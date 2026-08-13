#!/usr/bin/env python3
r"""SPE on the VITM 68 baseline — optimiser-driven RESAMPLING of the household
composition + re-sim (mechanism 2). Mirrors spe_driver but on the VITM engine:
the 68-baseline-weighted population (restore weights ride in) is resampled by a
seed/k-EXPLORED feature segmentation (demographics + 68-baseline travel signature),
then re-simulated via asim 24.1.4 and scored with score_vitm. Skims/land-use stay
fixed at the 68 baseline (SPE touches households only), so NO per-candidate skim
rewrite — each candidate just writes the resampled hh/persons.

    python spe_vitm.py --iters 15 --subsample 2000 --workers 3
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hdr_driver_vitm as HV          # VITM engine, base writer, load, asim, paths
import spe_mechanism as SPE           # scale_population, decode_parameters
import score_vitm as V

RUN_ROOT = HV.RUN_ROOT
RESTORE_NPZ = os.path.join(RUN_ROOT, "best_restore_vitm.npz")
BASE_TRIPS = os.path.join(RUN_ROOT, "out_best_restore_vitm", "final_trips.csv")
FAIL = -10.0
K_HDR = 6                              # the restore's HDR clustering
_DEMOG = ["HH_INCOME_BAND", "HHSIZE", "N_WORKERS", "AUTO_OWNERSHIP"]
_DISC = {"shopping", "othmaint", "othdiscr", "eatout", "social"}


def build_base(subsample, tag="spe_vitm"):
    """Materialise the 68-baseline-weighted VITM data dir (restore weights ride in)
    and return the baseline-weighted (households, persons) for resampling + the
    fixed-baseline dir (land_use + weighted skim) to hardlink into workers. The base
    is PER-OPTIMISER (tag) and guarded by .ready, so concurrent cmaes/ppo/lstm runs
    never collide writing/reading one shared 2 GB skim."""
    bdir = os.path.join(RUN_ROOT, f"{tag}_base_data")
    if not os.path.exists(os.path.join(bdir, ".ready")):
        hh, pe = HV.load_vitm(subsample)
        labels = HV.cluster_vitm(hh, K_HDR)
        base = HV.base_blocks(RESTORE_NPZ, K_HDR)
        os.makedirs(bdir, exist_ok=True)
        HV.write_vitm_inputs(np.zeros(HV.n_dim(K_HDR)), K_HDR, hh, pe, labels, bdir, base, HV.RESTORE_B)
        open(os.path.join(bdir, ".ready"), "w").close()   # only after a clean write
    base_hh = pd.read_csv(os.path.join(bdir, "synthetic_households_formatted.csv"))
    base_pe = pd.read_csv(os.path.join(bdir, "synthetic_persons_formatted.csv"))
    return bdir, base_hh, base_pe


def seg_features(base_hh, trips_path):
    """Standardised [demographics + 68-baseline trip count / work-share / disc-share]."""
    feats = base_hh[_DEMOG].fillna(0).to_numpy(float)
    n = w = d = np.zeros(len(base_hh))
    if os.path.exists(trips_path):
        t = pd.read_csv(trips_path, usecols=["household_id", "purpose"])
        idx = pd.Index(base_hh["household_id"].to_numpy())
        n = t.groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        w = t[t["purpose"] == "work"].groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        d = t[t["purpose"].isin(_DISC)].groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        w, d = w / np.clip(n, 1, None), d / np.clip(n, 1, None)
    X = np.column_stack([feats, n, w, d])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def feature_segment(seg, k, seed):
    if k <= 1 or len(seg) == 0:
        return np.zeros(len(seg), dtype=int)
    rng = np.random.default_rng(int(seed) + 1)
    proj = rng.standard_normal(seg.shape[1]); proj /= (np.linalg.norm(proj) + 1e-9)
    s = seg @ proj
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return np.clip(np.digitize(s, edges), 0, k - 1)


def write_resample(base_hh, base_pe, seg, raw, kmin, kmax, data_dir):
    """Decode -> segment -> scale_population -> overwrite hh/persons (land_use+skim
    already the fixed 68 baseline in data_dir). Returns resampled hh count."""
    HV._assert_run(data_dir)
    k, seed, weights = SPE.decode_parameters(raw, kmin, kmax)
    groups = feature_segment(seg, k, seed)
    p_out, hh_out = SPE.scale_population(base_pe, base_hh, weights, groups, zone_col=None)
    hh_out.to_csv(os.path.join(data_dir, "synthetic_households_formatted.csv"), index=False)
    p_out.to_csv(os.path.join(data_dir, "synthetic_persons_formatted.csv"), index=False)
    return len(hh_out)


def setup_workers(n, bdir, tag):
    workers = []
    for i in range(n):
        d = os.path.join(RUN_ROOT, f"{tag}_w{i}", "data")
        if not os.path.exists(os.path.join(d, ".ready")):
            os.makedirs(d, exist_ok=True)
            for item in os.listdir(bdir):
                if item == ".ready":
                    continue
                s, t = os.path.join(bdir, item), os.path.join(d, item)
                if item == "skims.omx":
                    # SPE keeps the skim FIXED at the 68 base, so workers don't each need a
                    # 2 GB copy (which antivirus intermittently locks mid-copy -> Errno 13).
                    # Hardlink it: instant, zero extra disk, read-only at run time.
                    if os.path.exists(t):
                        os.remove(t)
                    try:
                        os.link(s, t)
                    except OSError:
                        shutil.copyfile(s, t)                 # fallback if hardlink unsupported
                elif os.path.isdir(s):
                    shutil.copytree(s, t, dirs_exist_ok=True)
                else:
                    shutil.copyfile(s, t)
            open(os.path.join(d, ".ready"), "w").close()
        workers.append({"data": d, "out": os.path.join(RUN_ROOT, f"{tag}_w{i}", "out")})
    return workers


def evaluate_batch(cands, base_hh, base_pe, seg, kmin, kmax, workers):
    n = len(workers); scores = [FAIL] * len(cands)
    for s in range(0, len(cands), n):
        chunk = list(range(s, min(s + n, len(cands)))); running = []
        for slot, ci in enumerate(chunk):
            try:
                nhh = write_resample(base_hh, base_pe, seg, cands[ci], kmin, kmax, workers[slot]["data"])
            except Exception as e:
                print(f"  cand {ci} resample failed: {e}", flush=True); continue
            running.append((ci, slot, nhh, HV.run_vitm_asim(workers[slot]["data"], workers[slot]["out"])))
        for ci, slot, nhh, p in running:
            p.wait(); p._logf.close()
            if p.returncode == 0:
                c, _ = HV.score_dir(workers[slot]["out"], nhh, False)
                scores[ci] = c if (c and c > 0) else FAIL
    return scores


def main():
    ap = argparse.ArgumentParser(description="SPE resampling correction on the VITM 68 baseline")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RUN_ROOT, exist_ok=True)
    kmin, kmax, dim = args.kmin, args.kmax, 2 + args.kmax

    tag = f"spe_{args.optimizer}_vitm"
    print(f"[spe-vitm] materialising 68 baseline + seg features ({tag}_base_data, isolated) ...", flush=True)
    bdir, base_hh, base_pe = build_base(args.subsample, tag)
    seg = seg_features(base_hh, BASE_TRIPS)
    print(f"[spe-vitm] {len(base_hh)} hh, dim={dim} (2+kmax); skim/land-use FIXED at 68 base", flush=True)

    if args.smoke:
        d = os.path.join(RUN_ROOT, "spe_smoke", "data"); os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, ".ready")):
            shutil.copytree(bdir, d, dirs_exist_ok=True); open(os.path.join(d, ".ready"), "w").close()
        nhh = write_resample(base_hh, base_pe, seg, np.zeros(dim), kmin, kmax, d)
        p = HV.run_vitm_asim(d, os.path.join(RUN_ROOT, "spe_smoke", "out")); p.wait(); p._logf.close()
        c, _ = HV.score_dir(os.path.join(RUN_ROOT, "spe_smoke", "out"), nhh, False)
        print(f"[spe-vitm] SMOKE uniform resample -> {nhh} hh, composite={c:.2f} (~68 base)", flush=True)
        return

    ob = (-6, 6)
    if args.optimizer == "lstm":
        phi = np.load(os.path.join(HERE, "lstm_meta_weights.npz"))["phi"]
        opt = HV.H.make_optimiser("lstm", dim, args.popsize, lstm_phi=phi,
                                  lstm_config=HV.H.LSTMMetaConfig(bounds=ob))
    elif args.optimizer == "ppo":
        opt = HV.H.make_optimiser("ppo", dim, args.popsize, ppo_config=HV.H.PPOConfig(bounds=ob))
    else:
        opt = HV.H.make_optimiser("cmaes", dim, args.popsize, bounds=ob, sigma0=2.0)
    print(f"[spe-vitm] optimizer={args.optimizer}", flush=True)
    workers = setup_workers(max(1, min(args.workers, args.popsize)), bdir, tag)
    best, best_w, hist = FAIL, None, []; t0 = time.time()
    for it in range(args.iters):
        cands = opt.ask()
        scores = evaluate_batch(cands, base_hh, base_pe, seg, kmin, kmax, workers)
        opt.tell(cands, scores)
        w_b, s_b = opt.best_solution
        if s_b > best:
            best, best_w = s_b, np.asarray(w_b).copy()
            np.savez(os.path.join(RUN_ROOT, f"best_{tag}.npz"), w=best_w, kmin=kmin, kmax=kmax)
        hist.append({"iter": it, "best": float(best), "iter_max": float(np.max(scores))})
        json.dump(hist, open(os.path.join(RUN_ROOT, f"history_{tag}.json"), "w"), indent=2)
        print(f"[spe-vitm] it {it:3d}/{args.iters} best={best:6.2f} iter_max={np.max(scores):6.2f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        if opt.converged:
            print("[spe-vitm] converged."); break
    HV.COST.drain()
    cs = HV.COST.summarize(os.path.join(RUN_ROOT, "costs.jsonl"))
    if cs:
        print(f"[spe-vitm] cost: {cs['n_runs']} asim runs, "
              f"mean {cs['wall_s_mean']}s/run, peak RAM {cs['peak_rss_mb_max']}MB, "
              f"~{cs['output_mb_mean']}MB out/run", flush=True)
    print(f"[spe-vitm] DONE best composite={best:.2f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HDR driver for the transferred model: search for attribute weights with ActivitySim in the loop."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import openmatrix as omx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_mechanism as H
import cost_utils as COST

MAIN = HERE
BASELINE_DATA = os.path.join(HERE, "melb", "data")
CONFIGS = os.path.join(HERE, "melb", "configs")
RUN_ROOT = os.path.join(HERE, "hdr_run")


def _find(data_dir, name):
    """Case-insensitive file lookup (handles Households.csv vs households.csv)."""
    for f in os.listdir(data_dir):
        if f.lower() == name.lower():
            return os.path.join(data_dir, f)
    raise FileNotFoundError(os.path.join(data_dir, name))

sys.path.insert(0, MAIN)
import score_simulation as SCORE

K_MIN, K_MAX = 4, 8
POPSIZE = 16
DEFAULT_ITERS = 60
DEFAULT_WORKERS = 6

_HH_RENAME = {"HHID": "household_id", "PERSONS": "hhsize", "TAZ": "home_zone_id",
              "VEHICL": "auto_ownership", "workers": "num_workers"}


def _hh_id_col(df: pd.DataFrame) -> str:
    for c in ("HHID", "household_id", "hhno", "HHHID"):
        if c in df.columns:
            return c
    raise KeyError("no household id column found in households.csv")


def _mirror_baseline(dst, extra_ignore=()):
    """Copy the input data tree into dst, skipping the named files."""
    os.makedirs(dst, exist_ok=True)
    skip = {"households.csv", "persons.csv", "land_use.csv"} | {x.lower() for x in extra_ignore}

    def _ignore(d, names):
        return [n for n in names if n.lower() in skip]

    shutil.copytree(BASELINE_DATA, dst, dirs_exist_ok=True, ignore=_ignore)


def setup_run_data(tag=""):
    """Copy the input data tree into the run folder once."""
    data_dir = os.path.join(RUN_ROOT, f"{tag}_data" if tag else "data")
    sentinel = os.path.join(data_dir, ".hdr_data_ready")
    if os.path.exists(sentinel):
        return data_dir
    _mirror_baseline(data_dir)
    open(sentinel, "w").close()
    return data_dir


def _read_csv(path):
    """Read an input table with full float precision."""
    return pd.read_csv(path, float_precision="round_trip")


def load_baseline():
    hh = _read_csv(_find(BASELINE_DATA, "households.csv"))
    pe = _read_csv(_find(BASELINE_DATA, "persons.csv"))
    return hh, pe


def load_landuse():
    return _read_csv(_find(BASELINE_DATA, "land_use.csv"))


def cluster_once(hh_raw, persons, k_force=0):
    """Cluster households on demographic features; returns (labels, k)."""
    idc = _hh_id_col(hh_raw)
    hh = hh_raw.rename(columns=_HH_RENAME).copy()
    hh = hh.set_index("household_id")
    p = persons.copy()
    if "household_id" not in p.columns:
        raise KeyError("persons.csv must have a household_id column")
    kmin, kmax = (k_force, k_force) if k_force else (K_MIN, K_MAX)
    labels, k = H.cluster_households(p, hh, kmin, kmax, config=H.HDRConfig())
    label_by_id = pd.Series(labels.values, index=hh.index)
    return idc, label_by_id, k


def _assert_run_dir(data_dir):
    """Refuse to write weighted inputs anywhere but the run folder."""
    rr = os.path.abspath(RUN_ROOT)
    assert os.path.abspath(data_dir).startswith(rr), \
        f"refusing to write outside {rr}: {data_dir}"


def _split_theta(w_flat, k, cfg):
    """Split the flat optimizer vector into (per-cluster household, per-cluster person, global land-use, global skim) blocks."""
    n_hh, n_pe = len(cfg.attrs), len(cfg.person_attrs)
    n_lu, n_sk = len(cfg.lu_attrs), len(cfg.skim_attrs)
    w = np.asarray(w_flat, dtype=float)
    a, b, c = k * n_hh, k * n_hh + k * n_pe, k * n_hh + k * n_pe + n_lu
    return w[:a].reshape(k, n_hh), w[a:b].reshape(k, n_pe), w[b:c], w[c:c + n_sk]


def _cluster_of(id_series, label_by_id, k):
    cl = label_by_id.reindex(id_series).fillna(0).to_numpy().astype(int)
    return np.clip(cl, 0, k - 1)


def _with_weight_cols(df, cols, identity):
    """Copy of df in which every weight column exists, added at its neutral value where the input has none."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = identity
    return out


def _baseline_skims():
    p = os.path.join(BASELINE_DATA, "omx", "skims.omx")
    if os.path.exists(p):
        return p
    import glob
    return glob.glob(os.path.join(BASELINE_DATA, "omx", "*.omx"))[0]


def _write_weighted_skims(data_dir, w_sk, cfg):
    """Write a weighted copy of the skims into the run folder."""
    src = _baseline_skims()
    dst = os.path.join(data_dir, "omx", os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    mult = np.exp(np.clip(w_sk, *cfg.skim_log_clip))
    rules = [(H.SKIM_CORE_PATTERNS[a], float(mult[j])) for j, a in enumerate(cfg.skim_attrs)]
    f = omx.open_file(dst, "a")
    for name in f.list_matrices():
        for pat, fac in rules:
            if re.search(pat, name):
                if fac != 1.0:
                    f[name][:] = np.array(f[name]) * fac
                break
    f.close()


_PERSON_CDAP_COLS = ("cdap_work_w", "cdap_student_w", "cdap_nonmand_w")
_PERSON_SPEC = {
    "work":    ("cdap_work_w",    lambda p: p["pemploy"].isin((1, 2)).to_numpy().astype(float)),
    "student": ("cdap_student_w", lambda p: p["ptype"].isin((3, 6, 7)).to_numpy().astype(float)),
    "nonmand": ("cdap_nonmand_w", lambda p: np.ones(len(p))),
}


def _persons_with_workw(persons, label_by_id, k, w_pe, cfg):
    """Copy of the person table carrying the person weight columns."""
    p = _with_weight_cols(persons, _PERSON_CDAP_COLS, 0.0)
    cl = _cluster_of(p["household_id"], label_by_id, k)
    for j, attr in enumerate(cfg.person_attrs):
        col, indicator = _PERSON_SPEC[attr]
        lw = np.clip(w_pe[:, j], *cfg.work_log_clip)
        p[col] = p[col] + indicator(p) * lw[cl]
    return p


def write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, w_flat, data_dir, cfg):
    """Write households, persons, land use and skims with their weights into data_dir."""
    _assert_run_dir(data_dir)
    w_hh, w_pe, w_lu, w_sk = _split_theta(w_flat, k, cfg)
    df = _with_weight_cols(hh_raw, [f"w_{a}" for a in cfg.attrs], 1.0)
    mult = np.exp(np.clip(w_hh, *cfg.weight_log_clip))
    cl = _cluster_of(df[idc], label_by_id, k)
    for j, attr in enumerate(cfg.attrs):
        df[f"w_{attr}"] = df[f"w_{attr}"] * mult[cl, j]
    df.to_csv(os.path.join(data_dir, "households.csv"), index=False)
    _persons_with_workw(persons, label_by_id, k, w_pe, cfg).to_csv(
        os.path.join(data_dir, "persons.csv"), index=False)
    lu = _with_weight_cols(lu_raw, [f"w_{a}" for a in cfg.lu_attrs], 1.0)
    lmult = np.exp(np.clip(w_lu, *cfg.lu_weight_log_clip))
    for j, attr in enumerate(cfg.lu_attrs):
        lu[f"w_{attr}"] = lu[f"w_{attr}"] * float(lmult[j])
    lu.to_csv(os.path.join(data_dir, "land_use.csv"), index=False)
    _write_weighted_skims(data_dir, w_sk, cfg)


def write_baseline_inputs(hh_raw, lu_raw, persons, data_dir, cfg):
    """Write the inputs with every weight at its neutral value."""
    _assert_run_dir(data_dir)
    _with_weight_cols(hh_raw, [f"w_{a}" for a in cfg.attrs], 1.0).to_csv(
        os.path.join(data_dir, "households.csv"), index=False)
    _with_weight_cols(persons, _PERSON_CDAP_COLS, 0.0).to_csv(
        os.path.join(data_dir, "persons.csv"), index=False)
    _with_weight_cols(lu_raw, [f"w_{a}" for a in cfg.lu_attrs], 1.0).to_csv(
        os.path.join(data_dir, "land_use.csv"), index=False)


def report_drift(hh_raw, idc, label_by_id, k, w_flat, cfg):
    """Print the effective weights and the change in the weighted income aggregate."""
    w_hh, w_pe, w_lu, w_sk = _split_theta(w_flat, k, cfg)
    hh_mult = np.exp(np.clip(w_hh, *cfg.weight_log_clip))
    lu_mult = np.exp(np.clip(w_lu, *cfg.lu_weight_log_clip))
    sk_mult = np.exp(np.clip(w_sk, *cfg.skim_log_clip))
    print("[hdr] weight report (effective multipliers):")
    for j, a in enumerate(cfg.attrs):
        print(f"        household {a:8s} per-cluster: "
              f"{np.round(hh_mult[:, j], 3).tolist()}")
    for j, a in enumerate(cfg.person_attrs):
        odds = np.exp(np.clip(w_pe[:, j], *cfg.work_log_clip))
        print(f"        person   {a:8s} per-cluster: "
              f"CDAP odds x{np.round(odds, 3).tolist()}")
    for j, a in enumerate(cfg.lu_attrs):
        print(f"        land-use  {a:11s}: x{lu_mult[j]:.3f}")
    for j, a in enumerate(cfg.skim_attrs):
        print(f"        skim      {a:11s}: x{sk_mult[j]:.3f}")
    if "income" in cfg.attrs and "income" in hh_raw.columns:
        cl = _cluster_of(hh_raw[idc], label_by_id, k)
        eff = hh_mult[cl, cfg.attrs.index("income")]
        inc = hh_raw["income"].to_numpy()
        print(f"        income aggregate drift (weighted/input): "
              f"{(inc * eff).sum() / inc.sum():.3f}  (1.0 == no drift)")


def _activitysim_exe():
    cand = os.path.join(os.path.dirname(sys.executable), "Scripts", "activitysim.exe")
    return cand if os.path.exists(cand) else "activitysim"


def run_activitysim(data_dir, out_dir, log_path, label="", cost_log=None):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [_activitysim_exe(), "run", "-c", CONFIGS, "-d", data_dir, "-o", out_dir]
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    COST.RunCost(p, out_dir, label=(label or os.path.basename(out_dir)),
                 cost_log=(cost_log or os.path.join(RUN_ROOT, "costs.jsonl")),
                 extra={"engine": "melbourne", "data_mb": round(COST._dir_size_mb(data_dir), 1)})
    p.wait(); logf.close()
    return p.returncode == 0


def score_run(out_dir, n_households):
    trips_path = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(trips_path):
        alt = os.path.join(out_dir, "trips.csv")
        trips_path = alt if os.path.exists(alt) else trips_path
    if not os.path.exists(trips_path):
        return H.FAILURE_SCORE, None
    trips = pd.read_csv(trips_path)
    res = SCORE.evaluate_simulation(trips, n_households)
    return float(res["composite_score"]), res


def launch_activitysim(data_dir, out_dir, log_path):
    """Start an ActivitySim run as a background subprocess (non-blocking)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [_activitysim_exe(), "run", "-c", CONFIGS, "-d", data_dir, "-o", out_dir]
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    p._logf = logf
    COST.RunCost(p, out_dir, label=os.path.basename(out_dir),
                 cost_log=os.path.join(RUN_ROOT, "costs.jsonl"),
                 extra={"engine": "melbourne", "data_mb": round(COST._dir_size_mb(data_dir), 1)})
    return p


def setup_workers(n, persons_df, tag=""):
    """Create n worker folders, each with a copy of the input data tree."""
    pre = f"{tag}_" if tag else ""
    workers = []
    for i in range(n):
        ddir = os.path.join(RUN_ROOT, f"{pre}w{i}", "data")
        odir = os.path.join(RUN_ROOT, f"{pre}w{i}", "output")
        sentinel = os.path.join(ddir, ".hdr_data_ready")
        if not os.path.exists(sentinel):
            _mirror_baseline(ddir, extra_ignore=["persons.csv"])
            open(sentinel, "w").close()
        _with_weight_cols(persons_df, _PERSON_CDAP_COLS, 0.0).to_csv(
            os.path.join(ddir, "persons.csv"), index=False)
        workers.append({"data": ddir, "out": odir})
    return workers


def evaluate_batch(cands, hh_raw, lu_raw, persons, idc, label_by_id, k, cfg, workers, tagbase,
                   scorer=None):
    """Evaluate candidates, running up to len(workers) ActivitySim runs at once."""
    scorer = scorer or score_run
    n = len(workers)
    nhh = len(hh_raw)
    scores = [H.FAILURE_SCORE] * len(cands)
    metas = [None] * len(cands)
    for start in range(0, len(cands), n):
        chunk = list(range(start, min(start + n, len(cands))))
        running = []
        for slot, ci in enumerate(chunk):
            w = workers[slot]
            write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, cands[ci], w["data"], cfg)
            log_path = os.path.join(RUN_ROOT, f"asim_{tagbase}_c{ci}.log")
            running.append((ci, slot, launch_activitysim(w["data"], w["out"], log_path)))
        for ci, slot, p in running:
            p.wait()
            p._logf.close()
            if p.returncode == 0:
                scores[ci], metas[ci] = scorer(workers[slot]["out"], nhh)
    return scores, metas


def main():
    global BASELINE_DATA, CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--popsize", type=int, default=POPSIZE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="max concurrent ActivitySim runs")
    ap.add_argument("--data", default=BASELINE_DATA,
                    help="baseline data dir (persons/households/land_use/skims)")
    ap.add_argument("--configs", default=CONFIGS, help="ActivitySim configs dir")
    ap.add_argument("--smoke", action="store_true",
                    help="run ONE ActivitySim on the input at zero weights and report the score")
    ap.add_argument("--subsample", type=int, default=0,
                    help="optimize on an N-household subsample (faster, fewer crashes); also limits --smoke")
    ap.add_argument("--apply-best", dest="apply_best", action="store_true",
                    help="apply saved best_<optimizer>.npz weights to the FULL population and score")
    ap.add_argument("--k", type=int, default=0,
                    help="force number of household clusters (0 = auto-select by silhouette)")
    ap.add_argument("--warmstart", action="store_true",
                    help="cmaes: seed the search from the saved best_<tag>.npz (must match dim)")
    args = ap.parse_args()
    run_tag = args.optimizer
    BASELINE_DATA = args.data
    CONFIGS = args.configs

    os.makedirs(RUN_ROOT, exist_ok=True)
    cfg = H.HDRConfig()
    data_dir = setup_run_data(run_tag)
    hh_raw, persons = load_baseline()
    lu_raw = load_landuse()
    n_hh = len(hh_raw)
    print(f"[hdr] baseline: {n_hh} households, {len(persons)} persons, {len(lu_raw)} zones")

    if args.smoke:
        if args.subsample and args.subsample < n_hh:
            idc = _hh_id_col(hh_raw)
            n = args.subsample
            hh_sub = hh_raw.iloc[:n].copy()
            ids = set(hh_sub[idc].tolist())
            pe_sub = persons[persons["household_id"].isin(ids)].copy()
            print(f"[hdr] SMOKE subsample: {len(hh_sub)} households, {len(pe_sub)} persons")
            sdata = os.path.join(RUN_ROOT, "data_smoke")
            _mirror_baseline(sdata)
            write_baseline_inputs(hh_sub, lu_raw, pe_sub, sdata, cfg)
            n_hh_run = len(hh_sub)
        else:
            print("[hdr] SMOKE: writing households/persons/land_use at zero weights ...")
            write_baseline_inputs(hh_raw, lu_raw, persons, data_dir, cfg)
            sdata, n_hh_run = data_dir, n_hh
        out_dir = os.path.join(RUN_ROOT, "output_smoke")
        t0 = time.time()
        ok = run_activitysim(sdata, out_dir, os.path.join(RUN_ROOT, "asim_smoke.log"))
        if not ok:
            print("[hdr] SMOKE FAILED — see", os.path.join(RUN_ROOT, "asim_smoke.log"))
            sys.exit(1)
        score, res = score_run(out_dir, n_hh_run)
        print(f"[hdr] SMOKE OK in {time.time()-t0:.0f}s  composite={score:.2f}")
        if res:
            print(f"      trip_rate actual={res['trip_rate_actual']:.3f} "
                  f"target={res['trip_rate_target']:.3f}  "
                  f"(mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} "
                  f"trip={res['trip_rate_score']:.1f})")
        print("[hdr] If this matches your known baseline, the config edits are correct.")
        return

    if args.apply_best:
        bp = os.path.join(RUN_ROOT, f"best_{run_tag}.npz")
        if not os.path.exists(bp):
            print("[hdr] no saved weights at", bp)
            sys.exit(1)
        d = np.load(bp, allow_pickle=True)
        w = d["w"]
        saved_k = int(d["k"])
        idc, label_by_id, k = cluster_once(hh_raw, persons, saved_k)
        write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, w, data_dir, cfg)
        report_drift(hh_raw, idc, label_by_id, k, w, cfg)
        out_dir = os.path.join(RUN_ROOT, f"output_best_{run_tag}")
        print(f"[hdr] applying best {run_tag} weights to FULL population ({n_hh} hh) ...")
        ok = run_activitysim(data_dir, out_dir,
                             os.path.join(RUN_ROOT, f"asim_best_{run_tag}.log"))
        if not ok:
            print("[hdr] full-pop run FAILED — see log")
            sys.exit(1)
        score, res = score_run(out_dir, n_hh)
        print(f"[hdr] FULL-POP best {args.optimizer}: composite={score:.2f}")
        if res:
            print(f"      trip_rate={res['trip_rate_actual']:.3f} (target {res['trip_rate_target']:.3f})  "
                  f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} "
                  f"trip={res['trip_rate_score']:.1f}")
        return

    idc, label_by_id, k = cluster_once(hh_raw, persons, args.k)
    dim = (k * (len(cfg.attrs) + len(cfg.person_attrs))
           + len(cfg.lu_attrs) + len(cfg.skim_attrs))
    print(f"[hdr] clustered into k={k} groups -> dim={dim} "
          f"(({len(cfg.attrs)} hh + {len(cfg.person_attrs)} person) x {k} + "
          f"{len(cfg.lu_attrs)} land-use + {len(cfg.skim_attrs)} skim global)")

    if args.subsample and args.subsample < n_hh:
        srng = np.random.default_rng(0)
        pick = srng.choice(len(hh_raw), size=args.subsample, replace=False)
        hh_opt = hh_raw.iloc[pick].copy()
        ids = set(hh_opt[idc].tolist())
        pe_opt = persons[persons["household_id"].isin(ids)].copy()
        print(f"[hdr] optimizing on a {len(hh_opt)}-household subsample "
              f"({len(pe_opt)} persons)")
    else:
        hh_opt, pe_opt = hh_raw, persons

    olo, ohi = cfg.lu_weight_log_clip
    opt_bounds = (olo, ohi)
    opt_kwargs = (dict(bounds=opt_bounds, sigma0=(ohi - olo) / 4)
                  if args.optimizer == "cmaes" else {})
    if args.warmstart and args.optimizer == "cmaes":
        wp = os.path.join(RUN_ROOT, f"best_{run_tag}.npz")
        x0 = np.load(wp, allow_pickle=True)["w"] if os.path.exists(wp) else None
        if x0 is not None and len(x0) == dim:
            opt_kwargs["x0"] = x0
            opt_kwargs["sigma0"] = 0.6
            print(f"[hdr] warm-starting cmaes from {wp} (dim {dim}, sigma0=0.6)")
        else:
            print(f"[hdr] warmstart unavailable/dim-mismatch at {wp}; cold start")
    if args.optimizer == "lstm":
        npz = np.load(os.path.join(HERE, "lstm_meta_weights.npz"))
        phi = npz["phi"]
        lcfg = H.LSTMMetaConfig(bounds=opt_bounds)
        opt = H.make_optimiser("lstm", dim, args.popsize, lstm_phi=phi, lstm_config=lcfg)
    elif args.optimizer == "ppo":
        opt = H.make_optimiser("ppo", dim, args.popsize,
                               ppo_config=H.PPOConfig(bounds=opt_bounds))
    else:
        opt = H.make_optimiser("cmaes", dim, args.popsize, **opt_kwargs)

    n_workers = max(1, min(args.workers, args.popsize))
    workers = setup_workers(n_workers, pe_opt, run_tag)
    print(f"[hdr] running up to {n_workers} ActivitySim evaluations in parallel")
    history = []
    best_score, best_w = H.FAILURE_SCORE, None
    t_start = time.time()

    for it in range(args.iters):
        cands = opt.ask()
        scores, _ = evaluate_batch(cands, hh_opt, lu_raw, pe_opt, idc, label_by_id, k, cfg, workers,
                                   tagbase=f"{run_tag}_it{it}")
        opt.tell(cands, scores)
        w_best, s_best = opt.best_solution
        if s_best > best_score:
            best_score, best_w = s_best, np.asarray(w_best).copy()
            np.savez(os.path.join(RUN_ROOT, f"best_{run_tag}.npz"),
                     w=best_w, k=k, attrs=np.array(cfg.attrs),
                     person_attrs=np.array(cfg.person_attrs), lu_attrs=np.array(cfg.lu_attrs),
                     skim_attrs=np.array(cfg.skim_attrs),
                     label_by_id=label_by_id.values, id_index=label_by_id.index.values,
                     score=best_score)
        rec = {"iter": it, "best": float(best_score),
               "iter_mean": float(np.mean(scores)), "iter_max": float(np.max(scores))}
        history.append(rec)
        with open(os.path.join(RUN_ROOT, f"history_{run_tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[hdr] it {it:3d}/{args.iters}  best={best_score:6.2f}  "
              f"iter_max={np.max(scores):6.2f}  iter_mean={np.mean(scores):6.2f}  "
              f"elapsed={time.time()-t_start:.0f}s")
        if opt.converged:
            print("[hdr] optimiser converged.")
            break

    print(f"[hdr] DONE  best composite score = {best_score:.2f}")
    if best_w is not None:
        report_drift(hh_raw, idc, label_by_id, k, best_w, cfg)
    print(f"[hdr] best weights saved -> {os.path.join(RUN_ROOT, f'best_{run_tag}.npz')}")


if __name__ == "__main__":
    main()

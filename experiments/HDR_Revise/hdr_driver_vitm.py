#!/usr/bin/env python3
r"""
VITM HDR driver — the correction loop on the VITM Melbourne input data.

Mirrors hdr_driver.py but for the VITM ActivitySim model (asim 24.1.4 in the
VITM venv, VITM schema, reduced skim, score_vitm objective, the configs_vitm_hooks
overlay). Reuses hdr_mechanism's optimizer engines.

    cluster VITM households
      -> optimiser.ask()  (per-cluster income/work + global land-use/skim weights)
         -> for each candidate: write weighted households/persons/land_use + weighted
            skim -> run asim 24.1.4 -> score_vitm
      -> optimiser.tell(); track best.

--restore reverse-optimises to composite ~68 (the raw/uncalibrated VITM baseline,
analogous to Melbourne's faithful57). Forward (default) corrects toward observed.
Raw VITM inputs are never modified; everything is written into throwaway run dirs.

Run with the asim env's python (this driver), which shells out to the VITM venv:
    python hdr_driver_vitm.py --restore --iters 20 --subsample 3000 --workers 4
    python hdr_driver_vitm.py --restore --apply-best
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
import numpy as np, pandas as pd, openmatrix as omx, tables
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_mechanism as H           # optimizer engines
import score_vitm as V             # VITM composite scorer
import cost_utils as COST          # per-run wall-time / RAM / disk recorder

# ---- paths (VITM) ----
LANE   = r"G:\PhD programs\Ongoing works\lane_C"
VENV   = r"G:\VITM2_RC24v1_04_extsharing\.venv\Scripts\python.exe"
P2     = r"G:\PhD programs\Ongoing works\Phase 2 work - Calibration_SPSA"
DATA   = os.path.join(P2, "Data")                          # VITM population + skims_reduced.omx
REDUCED_SKIM = os.path.join(DATA, "skims_reduced.omx")
OVERLAY = os.path.join(P2, "vitm_run", "configs_vitm_hooks")
RUN_ROOT = os.path.join(P2, "vitm_run")
FAIL = -10.0
RESTORE_TARGET = 68.0

# ---- weight pool ----
HH_ATTRS   = ("income",)                                   # w_income (per cluster)
PERS_ATTRS = ("work",)                                     # cdap_work_w (per cluster)
LU_ATTRS   = ("employment", "enrollment")                 # global
SKIM_ATTRS = ("autotime", "autocost", "hovtime", "transit", "nonmotor")  # global
SKIM_CORE = {
    "autotime": r"^SOV_FREE_TIME__",
    "autocost": r"^SOV_FREE_(DISTANCE|TOLL)__",
    "hovtime":  r"^HOV[23]_FREE_TIME__",
    "transit":  r"^(WLK_ALLTRN_WLK_(IVT|IWAIT|XWAIT|WALK)|.*_COMPCOST)__",
    "nonmotor": r"^(WALK_DISTANCE|BIKE_DISTANCE)__",
}
LU_COLS = {"employment": ["EMP_TOTAL", "white_collar", "service", "health", "retail", "blue_collar"],
           "enrollment": ["ENROL_PR", "ENROL_SEC", "ENROL_TER", "primary", "secondary", "univ"]}
# bounds (log space). RESTORE_B = wide (reach 68 by degrading 88; also clips the
# saved restore = the 68 base). FWD_B = TIGHT plausible-calibration box for the
# FORWARD correction (skim ±0.7 -> x0.5-2.0; prevents the ÷7-skim overfit Melbourne hit).
RESTORE_B = {"income": 1.0, "work": 1.8, "lu": 1.4, "skim": 1.2}
FWD_B     = {"income": 0.8, "work": 1.6, "lu": 1.2, "skim": 0.7}
BOUNDS = RESTORE_B            # back-compat alias (base_blocks always uses RESTORE_B)


def n_dim(k):
    return k * (len(HH_ATTRS) + len(PERS_ATTRS)) + len(LU_ATTRS) + len(SKIM_ATTRS)


def split_theta(theta, k):
    a = k * len(HH_ATTRS); b = a + k * len(PERS_ATTRS); c = b + len(LU_ATTRS)
    w = np.asarray(theta, float)
    return w[:a].reshape(k, -1), w[a:b].reshape(k, -1), w[b:c], w[c:]


def base_blocks(npz_path, k):
    """Per-block clipped log-weights of the saved restore (the 68 baseline) — the
    forward correction composes ON TOP of these (forward θ=0 == the 68 baseline)."""
    w = np.load(npz_path, allow_pickle=True)["w"]
    wh, wp, wl, ws = split_theta(w, k)
    return {"income": np.clip(wh[:, 0], -BOUNDS["income"], BOUNDS["income"]),
            "work":   np.clip(wp[:, 0], -BOUNDS["work"], BOUNDS["work"]),
            "lu_emp": float(np.clip(wl[0], -BOUNDS["lu"], BOUNDS["lu"])),
            "lu_enr": float(np.clip(wl[1], -BOUNDS["lu"], BOUNDS["lu"])),
            "skim":   np.clip(ws, -BOUNDS["skim"], BOUNDS["skim"])}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_vitm(subsample, seed=0):
    hh = pd.read_csv(os.path.join(DATA, "synthetic_households_formatted.csv"))
    if subsample and subsample < len(hh):
        hh = hh.sample(n=subsample, random_state=seed).reset_index(drop=True)
    ids = set(hh["household_id"])
    keep = [c[c["household_id"].isin(ids)] for c in
            pd.read_csv(os.path.join(DATA, "synthetic_persons_formatted.csv"), chunksize=1_000_000)]
    pe = pd.concat(keep, ignore_index=True)
    return hh, pe


def cluster_vitm(hh, k):
    feats = hh[["HH_INCOME_BAND", "HHSIZE", "N_WORKERS", "AUTO_OWNERSHIP"]].fillna(0).to_numpy(float)
    labels = KMeans(k, random_state=0, n_init=5).fit_predict(StandardScaler().fit_transform(feats))
    return labels


def _assert_run(d):
    assert os.path.abspath(d).startswith(os.path.abspath(RUN_ROOT)), f"refuse to write outside {RUN_ROOT}: {d}"


def write_weighted_skim(skim_w, dst, retries=2):
    """Copy the reduced skim and scale matched cores in place (skim_w: attr->mult).
    Clean-overwrites and retries: copying a 2 GB OMX intermittently corrupts a deflate
    chunk (antivirus / disk caching on Windows), surfacing later as an inflate() read
    failure. We re-copy from the (verified-good) source and retry instead of crashing."""
    rules = [(SKIM_CORE[a], float(skim_w[a])) for a in SKIM_ATTRS]
    last = None
    for attempt in range(retries + 1):
        try:
            if os.path.exists(dst):
                os.remove(dst)                       # clean overwrite (no partial/locked file)
            shutil.copyfile(REDUCED_SKIM, dst)
            f = omx.open_file(dst, "a")
            try:
                for name in f.list_matrices():
                    for pat, mult in rules:
                        if re.search(pat, name):
                            if mult != 1.0:
                                f[name][:] = np.asarray(f[name]) * mult
                            break
            finally:
                f.close()
            return
        except Exception as e:                       # corrupted copy -> drop it and retry
            last = e
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                pass
            if attempt < retries:
                print(f"[skim] copy/scale failed (attempt {attempt+1}/{retries+1}), "
                      f"retrying: {str(e)[:80]}", flush=True)
    raise last


def write_vitm_inputs(theta, k, hh, pe, labels, data_dir, base=None, bnd=None):
    """Write weighted households/persons/land_use + weighted skim into data_dir.
    bnd = the box clipping THIS theta (FWD_B for forward, RESTORE_B for restore).
    base (optional) = base_blocks() of the restore (already clipped to RESTORE_B):
    the forward correction layers on top (clip(forward,FWD_B) + base), so theta==0
    reproduces the 68 baseline and the CORRECTION stays inside the plausible box."""
    _assert_run(data_dir)
    b = base or {}; bnd = bnd or BOUNDS
    w_hh, w_pe, w_lu, w_sk = split_theta(theta, k)
    hcl = labels
    inc = np.exp(np.clip(w_hh[:, 0], -bnd["income"], bnd["income"]) + b.get("income", 0.0))
    H_ = hh.copy(); H_["w_income"] = inc[hcl]
    H_.to_csv(os.path.join(data_dir, "synthetic_households_formatted.csv"), index=False)
    work = np.clip(w_pe[:, 0], -bnd["work"], bnd["work"]) + b.get("work", 0.0)
    p = pe.copy()
    pcl = p["household_id"].map(dict(zip(hh["household_id"], labels))).fillna(0).astype(int).to_numpy()
    is_worker = p["OCCUPATION"].isin([1, 2, 3, 4, 5]).to_numpy().astype(float)
    p["cdap_work_w"] = is_worker * work[pcl]
    p.to_csv(os.path.join(data_dir, "synthetic_persons_formatted.csv"), index=False)
    lu = pd.read_csv(os.path.join(DATA, "VITM2_Landuse_for_ActivitySim.csv"))
    lu["w_employment"] = float(np.exp(np.clip(w_lu[0], -bnd["lu"], bnd["lu"]) + b.get("lu_emp", 0.0)))
    lu["w_enrollment"] = float(np.exp(np.clip(w_lu[1], -bnd["lu"], bnd["lu"]) + b.get("lu_enr", 0.0)))
    lu.to_csv(os.path.join(data_dir, "VITM2_Landuse_for_ActivitySim.csv"), index=False)
    for f in ("business_shadow_prices.csv", "school_shadow_prices.csv", "workplace_shadow_prices.csv"):
        if not os.path.exists(os.path.join(data_dir, f)):
            shutil.copy(os.path.join(DATA, f), os.path.join(data_dir, f))
    bsk = b.get("skim", np.zeros(len(SKIM_ATTRS)))
    skim_w = {a: float(np.exp(np.clip(w_sk[j], -bnd["skim"], bnd["skim"]) + bsk[j]))
              for j, a in enumerate(SKIM_ATTRS)}
    write_weighted_skim(skim_w, os.path.join(data_dir, "skims.omx"))
    return len(H_)


def weight_report(theta, k, base, bnd):
    """Print the effective FORWARD-correction multipliers + the combined (base+correction)
    so weights can be sanity-checked for overfit (e.g. no ÷7 skims)."""
    b = base or {}
    w_hh, w_pe, w_lu, w_sk = split_theta(theta, k)
    print("[vitm] weight validity (forward correction x  |  combined w/ 68-base):")
    cinc = np.exp(np.clip(w_hh[:, 0], -bnd["income"], bnd["income"]))
    print(f"   income per-cluster  corr={np.round(cinc,2).tolist()}  "
          f"combined={np.round(np.exp(np.clip(w_hh[:,0],-bnd['income'],bnd['income'])+b.get('income',0)),2).tolist()}")
    cw = np.clip(w_pe[:, 0], -bnd["work"], bnd["work"])
    print(f"   work odds per-cluster corr=x{np.round(np.exp(cw),2).tolist()}")
    for j, a in enumerate(LU_ATTRS):
        c = np.exp(np.clip(w_lu[j], -bnd["lu"], bnd["lu"]))
        comb = np.exp(np.clip(w_lu[j], -bnd["lu"], bnd["lu"]) + (b.get("lu_emp", 0) if j == 0 else b.get("lu_enr", 0)))
        print(f"   land-use {a:10s} corr=x{c:.2f}  combined=x{comb:.2f}")
    bsk = b.get("skim", np.zeros(len(SKIM_ATTRS)))
    for j, a in enumerate(SKIM_ATTRS):
        c = np.exp(np.clip(w_sk[j], -bnd["skim"], bnd["skim"]))
        comb = np.exp(np.clip(w_sk[j], -bnd["skim"], bnd["skim"]) + bsk[j])
        flag = "  <-- check" if (comb < 0.25 or comb > 4) else ""
        print(f"   skim     {a:10s} corr=x{c:.2f}  combined=x{comb:.2f}{flag}")


def run_vitm_asim(data_dir, out_dir, label="", cost_log=None):
    env = os.environ.copy()
    for kk in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[kk] = "1"
    cmd = [VENV, "-m", "activitysim", "run", "-c", OVERLAY,
           "-c", os.path.join(LANE, "configs_mp"), "-c", os.path.join(LANE, "configs"),
           "-e", os.path.join(LANE, "extensions"), "-d", data_dir, "-o", out_dir]
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w")
    p = subprocess.Popen(cmd, env=env, cwd=LANE, stdout=logf, stderr=subprocess.STDOUT)
    p._logf = logf
    COST.RunCost(p, out_dir, label=(label or os.path.basename(out_dir)),
                 cost_log=(cost_log or os.path.join(RUN_ROOT, "costs.jsonl")),
                 extra={"engine": "vitm", "data_mb": round(COST._dir_size_mb(data_dir), 1)})
    return p


def score_dir(out_dir, n_hh, restore):
    import glob
    hits = glob.glob(os.path.join(out_dir, "**", "final_trips.csv"), recursive=True)
    if not hits:
        return FAIL, None
    res = V.evaluate_vitm(pd.read_csv(hits[0]), n_hh)
    comp = res["composite_score"]
    if not np.isfinite(comp):
        return FAIL, None
    if restore:                                  # reward proximity to 68
        return float(100.0 * np.exp(-abs(comp - RESTORE_TARGET) / 8.0)), res
    return float(comp), res


def setup_workers(n, tag):
    workers = []
    for i in range(n):
        d = os.path.join(RUN_ROOT, f"{tag}_w{i}", "data")
        o = os.path.join(RUN_ROOT, f"{tag}_w{i}", "out")
        os.makedirs(d, exist_ok=True)
        workers.append({"data": d, "out": o})
    return workers


def evaluate_batch(cands, k, hh, pe, labels, workers, restore, tag, base=None, bnd=None):
    n = len(workers); scores = [FAIL] * len(cands); metas = [None] * len(cands)
    nhh = len(hh)
    for s in range(0, len(cands), n):
        chunk = list(range(s, min(s + n, len(cands)))); running = []
        for slot, ci in enumerate(chunk):
            try:
                write_vitm_inputs(cands[ci], k, hh, pe, labels, workers[slot]["data"], base, bnd)
            except Exception as e:
                print(f"  cand {ci} write failed: {e}", flush=True); continue
            running.append((ci, slot, run_vitm_asim(workers[slot]["data"], workers[slot]["out"])))
        for ci, slot, p in running:
            p.wait(); p._logf.close()
            if p.returncode == 0:
                scores[ci], metas[ci] = score_dir(workers[slot]["out"], nhh, restore)
    return scores, metas


def main():
    ap = argparse.ArgumentParser(description="VITM HDR correction driver")
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--restore", action="store_true", help="reverse-optimize to composite ~68")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--apply-best", dest="apply_best", action="store_true")
    ap.add_argument("--warmstart", action="store_true",
                    help="seed CMA-ES from the saved best_<tag>.npz (resume after interruption)")
    ap.add_argument("--base-weights", dest="base_weights", default="",
                    help="restore npz (best_restore_vitm.npz) to layer the forward on; "
                         "at weights=0 the model IS the ~68 baseline")
    args = ap.parse_args()
    base = base_blocks(args.base_weights, args.k) if args.base_weights else None
    os.makedirs(RUN_ROOT, exist_ok=True)
    tag = ("restore" if args.restore else args.optimizer) + "_vitm"
    k = args.k

    if args.apply_best:
        bp = os.path.join(RUN_ROOT, f"best_{tag}.npz")
        d = np.load(bp, allow_pickle=True); theta = d["w"]; k = int(d["k"])
        base_ab = base_blocks(args.base_weights, k) if args.base_weights else None
        bnd_ab = RESTORE_B if args.restore else FWD_B
        hh, pe = load_vitm(args.subsample)       # SAME subsample (full 2.5M hh infeasible); seed 0
        labels = cluster_vitm(hh, k)             # deterministic (random_state=0) -> matches the run
        adir = os.path.join(RUN_ROOT, f"best_{tag}_data"); os.makedirs(adir, exist_ok=True)
        nhh = write_vitm_inputs(theta, k, hh, pe, labels, adir, base_ab, bnd_ab)
        out = os.path.join(RUN_ROOT, f"out_best_{tag}")
        print(f"[vitm] apply-best on {nhh}-hh subsample ...", flush=True)
        p = run_vitm_asim(adir, out); p.wait(); p._logf.close()
        comp, res = (score_dir(out, nhh, False))
        print(f"[vitm] composite = {comp:.2f}  trip_rate={res['trip_rate_actual']:.3f} "
              f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} trip={res['trip_rate_score']:.1f}")
        weight_report(theta, k, base_ab, bnd_ab)
        return

    hh, pe = load_vitm(args.subsample)
    labels = cluster_vitm(hh, k)
    dim = n_dim(k)
    print(f"[vitm] {len(hh)} hh / {len(pe)} persons, k={k}, dim={dim}, "
          f"{'RESTORE->68' if args.restore else 'FORWARD'}", flush=True)
    bnd = RESTORE_B if args.restore else FWD_B
    box = bnd["lu"]; opt_bounds = (-box, box)
    if args.optimizer == "lstm" and not args.restore:
        phi = np.load(os.path.join(HERE, "lstm_meta_weights.npz"))["phi"]
        opt = H.make_optimiser("lstm", dim, args.popsize, lstm_phi=phi,
                               lstm_config=H.LSTMMetaConfig(bounds=opt_bounds))
    elif args.optimizer == "ppo" and not args.restore:
        opt = H.make_optimiser("ppo", dim, args.popsize, ppo_config=H.PPOConfig(bounds=opt_bounds))
    else:                                            # cmaes (and always for --restore)
        opt_kw = dict(bounds=opt_bounds, sigma0=box / 2)
        if args.warmstart:
            wp = os.path.join(RUN_ROOT, f"best_{tag}.npz")
            if os.path.exists(wp) and len(np.load(wp, allow_pickle=True)["w"]) == dim:
                opt_kw["x0"] = np.load(wp, allow_pickle=True)["w"]; opt_kw["sigma0"] = box / 4
                print(f"[vitm] warm-start from {wp} (resuming)", flush=True)
        opt = H.make_optimiser("cmaes", dim, args.popsize, **opt_kw)
    print(f"[vitm] optimizer={args.optimizer} bounds={'RESTORE(wide)' if args.restore else 'FWD(tight)'}", flush=True)
    workers = setup_workers(max(1, min(args.workers, args.popsize)), tag)
    best, best_w, hist = FAIL, None, []
    t0 = time.time()
    for it in range(args.iters):
        cands = opt.ask()
        scores, metas = evaluate_batch(cands, k, hh, pe, labels, workers, args.restore, f"{tag}_it{it}", base, bnd)
        opt.tell(cands, scores)
        w_b, s_b = opt.best_solution
        if s_b > best:
            best, best_w = s_b, np.asarray(w_b).copy()
            np.savez(os.path.join(RUN_ROOT, f"best_{tag}.npz"), w=best_w, k=k)
        bi = int(np.argmax(scores))
        comp = metas[bi]["composite_score"] if metas[bi] else float("nan")
        hist.append({"iter": it, "best_obj": float(best), "iter_comp": float(comp)})
        json.dump(hist, open(os.path.join(RUN_ROOT, f"history_{tag}.json"), "w"), indent=2)
        extra = f" comp={comp:5.1f}(->68)" if args.restore else ""
        print(f"[vitm] it {it:3d}/{args.iters} best_obj={best:6.2f}{extra} "
              f"iter_max={np.max(scores):6.2f} elapsed={time.time()-t0:.0f}s", flush=True)
        if opt.converged:
            print("[vitm] converged."); break
    print(f"[vitm] DONE best_obj={best:.2f} -> {os.path.join(RUN_ROOT, f'best_{tag}.npz')}")
    if best_w is not None and not args.restore:
        weight_report(best_w, k, base, bnd)


if __name__ == "__main__":
    main()

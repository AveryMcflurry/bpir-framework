#!/usr/bin/env python3
"""HDR driver for VITM."""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
import numpy as np, pandas as pd, openmatrix as omx, tables
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_mechanism as H
import score_vitm as V
import cost_utils as COST

LANE   = os.path.join(HERE, "vitm", "model")
VENV   = os.environ.get("VITM_PYTHON", "python")
DATA   =os.path.join(HERE, "vitm", "data")
INPUT_SKIM = os.path.join(DATA, "skims.omx")
OVERLAY = os.path.join(LANE, "configs_vitm_hooks")
RUN_ROOT = os.path.join(HERE, "vitm_run")
FAIL = -10.0

HH_ATTRS   = ("income",)
PERS_ATTRS = ("work",)
LU_ATTRS   = ("employment", "enrollment")
SKIM_ATTRS = ("autotime", "autocost", "hovtime", "transit", "nonmotor")
SKIM_CORE = {
    "autotime": r"^SOV_FREE_TIME__",
    "autocost": r"^SOV_FREE_(DISTANCE|TOLL)__",
    "hovtime":  r"^HOV[23]_FREE_TIME__",
    "transit":  r"^(WLK_ALLTRN_WLK_(IVT|IWAIT|XWAIT|WALK)|.*_COMPCOST)__",
    "nonmotor": r"^(WALK_DISTANCE|BIKE_DISTANCE)__",
}
LU_COLS = {"employment": ["EMP_TOTAL", "white_collar", "service", "health", "retail", "blue_collar"],
           "enrollment": ["ENROL_PR", "ENROL_SEC", "ENROL_TER", "primary", "secondary", "univ"]}
BOUNDS = {"income": 0.8, "work": 1.6, "lu": 1.2, "skim": 0.7}


def n_dim(k):
    return k * (len(HH_ATTRS) + len(PERS_ATTRS)) + len(LU_ATTRS) + len(SKIM_ATTRS)


def split_theta(theta, k):
    a = k * len(HH_ATTRS); b = a + k * len(PERS_ATTRS); c = b + len(LU_ATTRS)
    w = np.asarray(theta, float)
    return w[:a].reshape(k, -1), w[a:b].reshape(k, -1), w[b:c], w[c:]


def _col(df, name, identity):
    """Weight column carried by the input (its identity value where the input has none)."""
    return df[name].to_numpy(float) if name in df.columns else np.full(len(df), float(identity))


def _read_csv(path, **kw):
    """Read an input table with full float precision."""
    return pd.read_csv(path, float_precision="round_trip", **kw)


def load_vitm(subsample, seed=0):
    hh = _read_csv(os.path.join(DATA, "synthetic_households_formatted.csv"))
    if subsample and subsample < len(hh):
        hh = hh.sample(n=subsample, random_state=seed).reset_index(drop=True)
    ids = set(hh["household_id"])
    keep = [c[c["household_id"].isin(ids)] for c in
            _read_csv(os.path.join(DATA, "synthetic_persons_formatted.csv"), chunksize=1_000_000)]
    pe = pd.concat(keep, ignore_index=True)
    return hh, pe


def cluster_vitm(hh, k):
    feats = hh[["HH_INCOME_BAND", "HHSIZE", "N_WORKERS", "AUTO_OWNERSHIP"]].fillna(0).to_numpy(float)
    labels = KMeans(k, random_state=0, n_init=5).fit_predict(StandardScaler().fit_transform(feats))
    return labels


def _assert_run(d):
    assert os.path.abspath(d).startswith(os.path.abspath(RUN_ROOT)), f"refuse to write outside {RUN_ROOT}: {d}"


def write_weighted_skim(skim_w, dst, retries=2):
    """Copy the input skim and scale the matched cores."""
    rules = [(SKIM_CORE[a], float(skim_w[a])) for a in SKIM_ATTRS]
    last = None
    for attempt in range(retries + 1):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copyfile(INPUT_SKIM, dst)
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
        except Exception as e:
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


def write_vitm_inputs(theta, k, hh, pe, labels, data_dir):
    """Write weighted households, persons, land use and skim into data_dir."""
    _assert_run(data_dir)
    w_hh, w_pe, w_lu, w_sk = split_theta(theta, k)
    hcl = labels
    inc = np.exp(np.clip(w_hh[:, 0], -BOUNDS["income"], BOUNDS["income"]))
    H_ = hh.copy(); H_["w_income"] = _col(H_, "w_income", 1.0) * inc[hcl]
    H_.to_csv(os.path.join(data_dir, "synthetic_households_formatted.csv"), index=False)
    work = np.clip(w_pe[:, 0], -BOUNDS["work"], BOUNDS["work"])
    p = pe.copy()
    pcl = p["household_id"].map(dict(zip(hh["household_id"], labels))).fillna(0).astype(int).to_numpy()
    is_worker = p["OCCUPATION"].isin([1, 2, 3, 4, 5]).to_numpy().astype(float)
    p["cdap_work_w"] = _col(p, "cdap_work_w", 0.0) + is_worker * work[pcl]
    p.to_csv(os.path.join(data_dir, "synthetic_persons_formatted.csv"), index=False)
    lu = _read_csv(os.path.join(DATA, "VITM2_Landuse_for_ActivitySim.csv"))
    lu["w_employment"] = _col(lu, "w_employment", 1.0) * float(np.exp(np.clip(w_lu[0], -BOUNDS["lu"], BOUNDS["lu"])))
    lu["w_enrollment"] = _col(lu, "w_enrollment", 1.0) * float(np.exp(np.clip(w_lu[1], -BOUNDS["lu"], BOUNDS["lu"])))
    lu.to_csv(os.path.join(data_dir, "VITM2_Landuse_for_ActivitySim.csv"), index=False)
    for f in ("business_shadow_prices.csv", "school_shadow_prices.csv", "workplace_shadow_prices.csv"):
        if not os.path.exists(os.path.join(data_dir, f)):
            shutil.copy(os.path.join(DATA, f), os.path.join(data_dir, f))
    skim_w = {a: float(np.exp(np.clip(w_sk[j], -BOUNDS["skim"], BOUNDS["skim"])))
              for j, a in enumerate(SKIM_ATTRS)}
    write_weighted_skim(skim_w, os.path.join(data_dir, "skims.omx"))
    return len(H_)


def write_baseline_inputs(hh, pe, data_dir):
    """Write the inputs with every weight at its neutral value."""
    return write_vitm_inputs(np.zeros(n_dim(1)), 1, hh, pe, np.zeros(len(hh), dtype=int), data_dir)


def weight_report(theta, k):
    """Print the weights found."""
    w_hh, w_pe, w_lu, w_sk = split_theta(theta, k)
    print("[vitm] learned weights (multipliers applied to the input):")
    cinc = np.exp(np.clip(w_hh[:, 0], -BOUNDS["income"], BOUNDS["income"]))
    print(f"   income per-cluster    x{np.round(cinc,2).tolist()}")
    cw = np.clip(w_pe[:, 0], -BOUNDS["work"], BOUNDS["work"])
    print(f"   work odds per-cluster x{np.round(np.exp(cw),2).tolist()}")
    for j, a in enumerate(LU_ATTRS):
        c = np.exp(np.clip(w_lu[j], -BOUNDS["lu"], BOUNDS["lu"]))
        print(f"   land-use {a:10s} x{c:.2f}")
    for j, a in enumerate(SKIM_ATTRS):
        c = np.exp(np.clip(w_sk[j], -BOUNDS["skim"], BOUNDS["skim"]))
        print(f"   skim     {a:10s} x{c:.2f}")


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


def score_dir(out_dir, n_hh):
    import glob
    hits = glob.glob(os.path.join(out_dir, "**", "final_trips.csv"), recursive=True)
    if not hits:
        return FAIL, None
    res = V.evaluate_vitm(pd.read_csv(hits[0]), n_hh)
    comp = res["composite_score"]
    if not np.isfinite(comp):
        return FAIL, None
    return float(comp), res


def setup_workers(n, tag):
    workers = []
    for i in range(n):
        d = os.path.join(RUN_ROOT, f"{tag}_w{i}", "data")
        o = os.path.join(RUN_ROOT, f"{tag}_w{i}", "out")
        os.makedirs(d, exist_ok=True)
        workers.append({"data": d, "out": o})
    return workers


def evaluate_batch(cands, k, hh, pe, labels, workers, tag):
    n = len(workers); scores = [FAIL] * len(cands); metas = [None] * len(cands)
    nhh = len(hh)
    for s in range(0, len(cands), n):
        chunk = list(range(s, min(s + n, len(cands)))); running = []
        for slot, ci in enumerate(chunk):
            try:
                write_vitm_inputs(cands[ci], k, hh, pe, labels, workers[slot]["data"])
            except Exception as e:
                print(f"  cand {ci} write failed: {e}", flush=True); continue
            running.append((ci, slot, run_vitm_asim(workers[slot]["data"], workers[slot]["out"])))
        for ci, slot, p in running:
            p.wait(); p._logf.close()
            if p.returncode == 0:
                scores[ci], metas[ci] = score_dir(workers[slot]["out"], nhh)
    return scores, metas


def main():
    ap = argparse.ArgumentParser(description="VITM HDR correction driver")
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--apply-best", dest="apply_best", action="store_true")
    ap.add_argument("--warmstart", action="store_true",
                    help="seed CMA-ES from the saved best_<tag>.npz (resume after interruption)")
    args = ap.parse_args()
    os.makedirs(RUN_ROOT, exist_ok=True)
    tag = args.optimizer + "_vitm"
    k = args.k

    if args.apply_best:
        bp = os.path.join(RUN_ROOT, f"best_{tag}.npz")
        d = np.load(bp, allow_pickle=True); theta = d["w"]; k = int(d["k"])
        hh, pe = load_vitm(args.subsample)
        labels = cluster_vitm(hh, k)
        adir = os.path.join(RUN_ROOT, f"best_{tag}_data"); os.makedirs(adir, exist_ok=True)
        nhh = write_vitm_inputs(theta, k, hh, pe, labels, adir)
        out = os.path.join(RUN_ROOT, f"out_best_{tag}")
        print(f"[vitm] apply-best on {nhh}-hh subsample ...", flush=True)
        p = run_vitm_asim(adir, out); p.wait(); p._logf.close()
        comp, res = score_dir(out, nhh)
        print(f"[vitm] composite = {comp:.2f}  trip_rate={res['trip_rate_actual']:.3f} "
              f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} trip={res['trip_rate_score']:.1f}")
        weight_report(theta, k)
        return

    hh, pe = load_vitm(args.subsample)
    labels = cluster_vitm(hh, k)
    dim = n_dim(k)
    print(f"[vitm] {len(hh)} hh / {len(pe)} persons, k={k}, dim={dim}", flush=True)
    box = BOUNDS["lu"]; opt_bounds = (-box, box)
    if args.optimizer == "lstm":
        phi = np.load(os.path.join(HERE, "lstm_meta_weights.npz"))["phi"]
        opt = H.make_optimiser("lstm", dim, args.popsize, lstm_phi=phi,
                               lstm_config=H.LSTMMetaConfig(bounds=opt_bounds))
    elif args.optimizer == "ppo":
        opt = H.make_optimiser("ppo", dim, args.popsize, ppo_config=H.PPOConfig(bounds=opt_bounds))
    else:
        opt_kw = dict(bounds=opt_bounds, sigma0=box / 2)
        if args.warmstart:
            wp = os.path.join(RUN_ROOT, f"best_{tag}.npz")
            if os.path.exists(wp) and len(np.load(wp, allow_pickle=True)["w"]) == dim:
                opt_kw["x0"] = np.load(wp, allow_pickle=True)["w"]; opt_kw["sigma0"] = box / 4
                print(f"[vitm] warm-start from {wp} (resuming)", flush=True)
        opt = H.make_optimiser("cmaes", dim, args.popsize, **opt_kw)
    print(f"[vitm] optimizer={args.optimizer}", flush=True)
    workers = setup_workers(max(1, min(args.workers, args.popsize)), tag)
    best, best_w, hist = FAIL, None, []
    t0 = time.time()
    for it in range(args.iters):
        cands = opt.ask()
        scores, metas = evaluate_batch(cands, k, hh, pe, labels, workers, f"{tag}_it{it}")
        opt.tell(cands, scores)
        w_b, s_b = opt.best_solution
        if s_b > best:
            best, best_w = s_b, np.asarray(w_b).copy()
            np.savez(os.path.join(RUN_ROOT, f"best_{tag}.npz"), w=best_w, k=k)
        bi = int(np.argmax(scores))
        comp = metas[bi]["composite_score"] if metas[bi] else float("nan")
        hist.append({"iter": it, "best_obj": float(best), "iter_comp": float(comp)})
        json.dump(hist, open(os.path.join(RUN_ROOT, f"history_{tag}.json"), "w"), indent=2)
        print(f"[vitm] it {it:3d}/{args.iters} best_obj={best:6.2f} "
              f"iter_max={np.max(scores):6.2f} elapsed={time.time()-t0:.0f}s", flush=True)
        if opt.converged:
            print("[vitm] converged."); break
    print(f"[vitm] DONE best_obj={best:.2f} -> {os.path.join(RUN_ROOT, f'best_{tag}.npz')}")
    if best_w is not None:
        weight_report(best_w, k)


if __name__ == "__main__":
    main()

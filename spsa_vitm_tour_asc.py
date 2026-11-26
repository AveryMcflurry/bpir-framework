#!/usr/bin/env python3
"""Calibration benchmark (SPSA) on VITM: tour-mode constants and tour-frequency coefficients."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.environ.get("BPIR_PACKAGE", HERE))
sys.path.insert(0, PKG)
import score_vitm as V

LANE = os.path.join(PKG, "vitm", "model")
CFG = os.path.join(LANE, "configs")
HOOKS = os.path.join(LANE, "configs_vitm_hooks")
DATA = os.path.join(PKG, "vitm", "data")
VENV = os.environ.get("VITM_PYTHON", "python")
RUN_ROOT = os.path.join(HERE, "vitm_tour_asc_run")
ARCHIVE = os.path.join(RUN_ROOT, "archive")
HIST = os.path.join(RUN_ROOT, "history_spsa_vitm_tour.json")
DIM, FAIL = 6, -10.0


def set_run_dir(name):
    """Point all outputs at ./<name>."""
    global RUN_ROOT, ARCHIVE, HIST
    RUN_ROOT = os.path.join(HERE, name)
    ARCHIVE = os.path.join(RUN_ROOT, "archive")
    HIST = os.path.join(RUN_ROOT, "history_spsa_vitm_tour.json")


N_HH =len(pd.read_csv(os.path.join(DATA, "synthetic_households_formatted.csv"), usecols=["household_id"]))
DATA_DIRS = [DATA]


def set_data_dir(paths):
    """Read the model inputs from the folders in `paths`, searched in that order, ahead of the package's."""
    global DATA_DIRS, N_HH
    paths = [os.path.abspath(p) for p in paths]
    DATA_DIRS = paths + [DATA]
    N_HH = len(pd.read_csv(os.path.join(paths[0], "synthetic_households_formatted.csv"), usecols=["household_id"]))

TOUR_BASE = pd.read_csv(os.path.join(CFG, "tour_mode_choice_coeffs.csv"), dtype=str, keep_default_na=False)
MAND_BASE = pd.read_csv(os.path.join(CFG, "mandatory_tour_frequency_coeffs.csv"))
NONMAND = [f for f in os.listdir(CFG) if f.startswith("non_mandatory_tour_frequency_coeffs_PTYPE")
           and not f.endswith("_base.csv")]
NONMAND_BASE = {f: pd.read_csv(os.path.join(CFG, f)) for f in NONMAND}

TOUR_LEVERS = [
    ("car_driver",    ("asc_Car_1_Occ_",)),
    ("car_passenger", ("asc_Car_2_Occ_", "asc_Car_3_Occ_")),
    ("active",        ("asc_Walking_", "asc_Bicycle_Cycling_")),
    ("transit",       ("asc_PT_walk_", "asc_PT_P_R_", "asc_PT_K_R_")),
]
for _name, _pref in TOUR_LEVERS:
    _n = int(TOUR_BASE["coefficient_name"].str.startswith(_pref).sum())
    assert _n > 0, f"no tour-mode constants found for lever {_name}"


def _free(df):
    if "constrain" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["constrain"].fillna("F").astype(str).str.upper().ne("T")


def apply_theta(theta, overlay):
    """Write the perturbed coefficient files into `overlay`."""
    os.makedirs(overlay, exist_ok=True)
    t_mode, t_mand, t_nonmand = theta[:4], theta[4], theta[5]
    tour = TOUR_BASE.copy()
    for (name, pref), t in zip(TOUR_LEVERS, t_mode):
        m = tour["coefficient_name"].str.startswith(pref)
        if float(t) != 0.0:
            tour.loc[m, "value"] = [repr(float(v) + float(t)) for v in tour.loc[m, "value"]]
    tour.to_csv(os.path.join(overlay, "tour_mode_choice_coeffs.csv"), index=False)
    mand = MAND_BASE.copy(); fr = _free(mand)
    msk = mand["coefficient_name"].str.contains("work|school|business", case=False, regex=True) & fr
    mand.loc[msk, "value"] = pd.to_numeric(mand.loc[msk, "value"], errors="coerce").fillna(0.0) + float(t_mand)
    mand.to_csv(os.path.join(overlay, "mandatory_tour_frequency_coeffs.csv"), index=False)
    for f, base in NONMAND_BASE.items():
        df = base.copy(); fr = _free(df)
        msk = df["coefficient_name"].str.contains("discr|maint|shop|social|eat|escort",
                                                  case=False, regex=True) & fr
        df.loc[msk, "value"] = pd.to_numeric(df.loc[msk, "value"], errors="coerce").fillna(0.0) + float(t_nonmand)
        df.to_csv(os.path.join(overlay, f), index=False)


def launch(theta, slot):
    overlay = os.path.join(RUN_ROOT, f"overlay_{slot}")
    out = os.path.join(RUN_ROOT, f"out_{slot}")
    apply_theta(theta, overlay)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    env = os.environ.copy()
    for kk in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[kk] = "1"
    cmd = [VENV, "-m", "activitysim", "run", "-c", overlay, "-c", HOOKS,
           "-c", os.path.join(LANE, "configs_mp"), "-c", CFG,
           "-e", os.path.join(LANE, "extensions")]
    for d in DATA_DIRS:
        cmd += ["-d", d]
    cmd += ["-o", out]
    logf = open(os.path.join(out, "run.log"), "w")
    p = subprocess.Popen(cmd, env=env, cwd=LANE, stdout=logf, stderr=subprocess.STDOUT)
    return {"p": p, "logf": logf, "out": out, "t0": time.time()}


def collect(job, label, theta):
    job["p"].wait(); job["logf"].close()
    rec = {"label": label, "theta": [float(x) for x in theta], "returncode": job["p"].returncode,
           "wall_s": round(time.time() - job["t0"], 1), "composite": FAIL}
    tp = os.path.join(job["out"], "final_trips.csv")
    if job["p"].returncode == 0 and os.path.exists(tp):
        res = V.evaluate_vitm(pd.read_csv(tp), N_HH)
        if np.isfinite(res["composite_score"]):
            rec.update(composite=float(res["composite_score"]),
                       trip_score=float(res["trip_rate_score"]), purpose_score=float(res["purpose_score"]),
                       mode_score=float(res["mode_score"]), trip_rate=float(res["trip_rate_actual"]),
                       mode_shares={k: float(v) for k, v in res["mode_probs_actual"].items()})
        dst = os.path.join(ARCHIVE, label); os.makedirs(dst, exist_ok=True)
        for f in ("final_trips.csv", "final_tours.csv"):
            if os.path.exists(os.path.join(job["out"], f)):
                shutil.copy(os.path.join(job["out"], f), os.path.join(dst, f))
    else:
        dst = os.path.join(ARCHIVE, label); os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(job["out"], "run.log"), os.path.join(dst, "run.log"))
    return rec


def evaluate_many(points, max_parallel):
    """Run the given (label, theta) points, up to max_parallel models at once."""
    recs = []
    for s in range(0, len(points), max_parallel):
        chunk = points[s:s + max_parallel]
        jobs = [launch(th, f"s{i}") for i, (_, th) in enumerate(chunk)]
        got = [collect(j, lab, th) for j, (lab, th) in zip(jobs, chunk)]
        for i, (lab, th) in enumerate(chunk):
            if got[i]["returncode"] != 0:
                got[i] = collect(launch(th, f"s{i}"), lab, th)
                got[i]["retried"] = True
        recs += got
    return recs


def fmt(rec):
    if "mode_score" not in rec:
        return f"{rec['label']:24s} FAILED (returncode {rec['returncode']})"
    ms = rec["mode_shares"]
    return (f"{rec['label']:24s} composite {rec['composite']:6.2f} | trip {rec['trip_score']:5.1f} "
            f"purpose {rec['purpose_score']:5.1f} mode {rec['mode_score']:5.1f} | rate {rec['trip_rate']:.3f} | "
            f"driver {100*ms['Private Vehicle - Driver']:.1f}% passenger {100*ms['Private Vehicle - Passenger']:.1f}% "
            f"walk {100*ms['Walking & Active Transport']:.1f}% cycle {100*ms['Cycling']:.1f}% "
            f"transit {100*ms['Public Transit']:.1f}% | {rec['wall_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--a", type=float, default=0.25)
    ap.add_argument("--c", type=float, default=0.08)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--resume", action="store_true", help="continue from the saved history")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--points", help="JSON file with [[label, [six lever values]], ...]; evaluates those points only")
    ap.add_argument("--run-dir", default="vitm_tour_asc_run", help="output folder under this script's folder")
    ap.add_argument("--data-dir", nargs="+",
                    help="folder(s) of model inputs to use ahead of the package's, searched in the order given")
    args = ap.parse_args()
    set_run_dir(args.run_dir)
    if args.data_dir:
        set_data_dir([os.path.join(HERE, d) for d in args.data_dir])
    os.makedirs(RUN_ROOT, exist_ok=True)
    tgt = V.MODE_PROBS_VITM
    print(f"[tour-asc] {N_HH} households; inputs read from {' then '.join(DATA_DIRS)}", flush=True)
    print("[tour-asc] observed targets: rate %.3f | " % V.TRIP_RATE
          + " ".join(f"{k.split(' - ')[-1].split(' ')[0].lower()} {100*v:.1f}%" for k, v in tgt.items()), flush=True)

    if args.points:
        pts = [(lab, np.array(th, dtype=float)) for lab, th in json.load(open(args.points))]
        assert all(len(th) == DIM for _, th in pts), "each point needs six lever values"
        recs = evaluate_many(pts, args.parallel)
        out = os.path.join(RUN_ROOT, os.path.splitext(os.path.basename(args.points))[0] + "_results.json")
        json.dump(recs, open(out, "w"), indent=2)
        for r in recs:
            print("[tour-asc] " + fmt(r), flush=True)
        return

    theta = np.zeros(DIM); best, best_rec = FAIL, None
    hist = []; rng = np.random.default_rng(0); k0 = 1
    if args.resume and os.path.exists(HIST):
        hist = json.load(open(HIST))
        for h in hist:
            rng.choice([-1.0, 1.0], size=DIM)
            if h["best"] > best:
                best = h["best"]
                best_rec = max((h["plus"], h["minus"]), key=lambda r: r["composite"])
        if hist:
            theta = np.array(hist[-1]["theta_after"]); k0 = hist[-1]["iter"] + 1
        print(f"[tour-asc] resuming at iteration {k0}, best so far {best:.2f}", flush=True)
    t0 = time.time()
    for k in range(k0, args.iters + 1):
        ak = args.a / ((k + args.A) ** 0.602)
        ck = args.c / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=DIM)
        rp, rm = evaluate_many([(f"it{k:03d}_p", theta + ck * delta), (f"it{k:03d}_m", theta - ck * delta)],
                               args.parallel)
        cp, cm = rp["composite"], rm["composite"]
        g = (cp - cm) / (2.0 * ck) * delta
        gnorm = float(np.linalg.norm(g))
        if gnorm > 5.0:
            g *= 5.0 / gnorm
        theta = np.clip(theta + ak * g, -4.0, 4.0)
        top = rp if cp >= cm else rm
        if top["composite"] > best:
            best, best_rec = top["composite"], top
        hist.append({"iter": k, "plus": rp, "minus": rm, "best": float(best),
                     "theta_after": [float(x) for x in theta],
                     "a_k": float(ak), "c_k": float(ck), "grad_norm": gnorm})
        json.dump(hist, open(HIST, "w"), indent=2)
        print(f"[tour-asc] it {k:2d}/{args.iters} C+={cp:6.2f} C-={cm:6.2f} best={best:6.2f} "
              f"mode+={rp.get('mode_score', float('nan')):5.1f} mode-={rm.get('mode_score', float('nan')):5.1f} "
              f"theta={np.round(theta, 3).tolist()} elapsed={time.time()-t0:.0f}s", flush=True)
    final = evaluate_many([("final_theta", theta)], 1)[0]
    json.dump({"final_theta": final, "best_evaluated_point": best_rec},
              open(os.path.join(RUN_ROOT, "result_spsa_vitm_tour.json"), "w"), indent=2)
    print("[tour-asc] best evaluated point: " + fmt(best_rec), flush=True)
    print("[tour-asc] final theta:          " + fmt(final), flush=True)
    print(f"[tour-asc] DONE best composite={best:.2f}", flush=True)


if __name__ == "__main__":
    main()

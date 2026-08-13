#!/usr/bin/env python3
r"""SPSA calibration on the VITM 68 baseline — the TRADITIONAL-calibration benchmark
that we contrast against the input-correction mechanisms (HDR / PWS / SPE).

Unlike the correction mechanisms (which weight the INPUTS), SPSA tunes the MODEL
COEFFICIENTS (mode-choice ASCs + IVT, and tour-frequency propensities) on the FIXED
68-baseline inputs — the restore weights are baked into the data once (incl. the
weighted skim, never rewritten), so each SPSA evaluation only rewrites tiny coeff
CSVs. It is scored with the SAME score_vitm composite (5-mode), so the number is
directly comparable to HDR/PWS/SPE. Per-run cost (wall / RAM / disk) is recorded
through cost_utils into vitm_run/costs.jsonl.

    python spsa_vitm.py --smoke               # theta=0 should reproduce ~68
    python spsa_vitm.py --iters 30            # full SPSA calibration

theta (6 dims, IDENTICAL structure to spsa_mel): [auto-driver, auto-passenger,
                 active, transit, mandatory tour-freq, non-mandatory tour-freq].
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hdr_driver_vitm as HV      # VITM engine paths + base writer + scorer
import score_vitm as V
import cost_utils as COST

LANE, VENV, OVERLAY, RUN_ROOT = HV.LANE, HV.VENV, HV.OVERLAY, HV.RUN_ROOT
CFG          = os.path.join(LANE, "configs")
DATA_68      = os.path.join(RUN_ROOT, "spsa_vitm_base")        # fixed 68-baseline inputs
SPSA_OVERLAY = os.path.join(RUN_ROOT, "configs_spsa_vitm")     # perturbed coeffs layer (first -c)
COST_LOG     = os.path.join(RUN_ROOT, "costs.jsonl")
RESTORE_NPZ  = os.path.join(RUN_ROOT, "best_restore_vitm.npz")
K_HDR        = 6                                                # the restore's HDR clustering
DIM          = 6                                                # symmetric with spsa_mel
FAIL         = -10.0

# base coefficient files (never modified; perturbed copies go to SPSA_OVERLAY)
MODE_BASE = pd.read_csv(os.path.join(CFG, "trip_mode_choice_coeffs.csv"))
MAND_BASE = pd.read_csv(os.path.join(CFG, "mandatory_tour_frequency_coeffs.csv"))
PCOLS = [c for c in MODE_BASE.columns if c != "coefficient_name"]
# Mode levers giving SPSA-VITM the SAME mode reach as SPSA-Melbourne: VITM has constants
# only for auto modes, so transit/active are tuned through their level-of-service (time)
# coefficients. Auto ASCs are additive; the LOS sensitivities are scaled multiplicatively
# by exp(0.5*theta) so they keep their (negative) sign no matter how far SPSA pushes.
TRANSIT_LOS = ["c_pnrTime", "c_knrTime", "c_firstWaitShort", "c_firstWaitLong", "c_xferWait"]
ACTIVE_LOS  = ["c_walkTimeShort", "c_walkTimeLong", "c_bikeTime"]
_need = ["sov_cons", "hov2_cons", "hov3_cons"] + TRANSIT_LOS + ACTIVE_LOS
_missing = [c for c in _need if c not in set(MODE_BASE["coefficient_name"])]
assert not _missing, f"spsa_vitm: mode coeff rows missing from base file: {_missing}"
NONMAND = [f for f in os.listdir(CFG) if f.startswith("non_mandatory_tour_frequency_coeffs_PTYPE")
           and not f.endswith("_base.csv")]
NONMAND_BASE = {f: pd.read_csv(os.path.join(CFG, f)) for f in NONMAND}


def _free(df):
    if "constrain" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["constrain"].fillna("F").astype(str).str.upper().ne("T")


def apply_theta(theta):
    """Write perturbed coefficient CSVs to SPSA_OVERLAY (layered first, so they
    override the base configs). Only unconstrained (constrain != 'T') rows move.
    6 levers: [auto-driver, auto-passenger, active, transit, mandatory, non-mandatory]."""
    os.makedirs(SPSA_OVERLAY, exist_ok=True)
    t_sov, t_hov, t_act, t_trn, t_mand, t_nonmand = theta
    mode = MODE_BASE.copy()

    def _add(names, val):                                  # additive (auto ASCs)
        m = mode["coefficient_name"].isin(names)
        mode.loc[m, PCOLS] = mode.loc[m, PCOLS].astype(float) + float(val)

    def _scale(names, t):                                  # multiplicative, sign-safe
        m = mode["coefficient_name"].isin(names)
        mode.loc[m, PCOLS] = mode.loc[m, PCOLS].astype(float) * float(np.exp(0.5 * t))

    _add(["sov_cons"], t_sov)                              # auto driver ASC
    _add(["hov2_cons", "hov3_cons"], t_hov)                # auto passenger ASC
    _scale(ACTIVE_LOS, t_act)                              # walk/bike (no constants exist -> LOS)
    _scale(TRANSIT_LOS, t_trn)                             # transit (no constants exist -> LOS)
    mode.to_csv(os.path.join(SPSA_OVERLAY, "trip_mode_choice_coeffs.csv"), index=False)
    # --- mandatory tour frequency: work/school/business propensity -> trip rate ---
    mand = MAND_BASE.copy(); fr = _free(mand)
    msk = mand["coefficient_name"].str.contains("work|school|business", case=False, regex=True) & fr
    mand.loc[msk, "value"] = pd.to_numeric(mand.loc[msk, "value"], errors="coerce").fillna(0.0) + float(t_mand)
    mand.to_csv(os.path.join(SPSA_OVERLAY, "mandatory_tour_frequency_coeffs.csv"), index=False)
    # --- non-mandatory tour frequency: discretionary/maintenance/shop -> trip rate ---
    for f, base in NONMAND_BASE.items():
        df = base.copy(); fr = _free(df)
        msk = df["coefficient_name"].str.contains("discr|maint|shop|social|eat|escort",
                                                  case=False, regex=True) & fr
        df.loc[msk, "value"] = pd.to_numeric(df.loc[msk, "value"], errors="coerce").fillna(0.0) + float(t_nonmand)
        df.to_csv(os.path.join(SPSA_OVERLAY, f), index=False)


def build_68_baseline(subsample):
    """Materialise the fixed 68-baseline inputs once (restore weights baked in, incl.
    the weighted skim). SPSA never rewrites these — only the coeff overlay."""
    ready = os.path.join(DATA_68, ".spsa_ready")
    hh, pe = HV.load_vitm(subsample)
    if os.path.exists(ready):
        return len(hh)
    labels = HV.cluster_vitm(hh, K_HDR)
    base = HV.base_blocks(RESTORE_NPZ, K_HDR)
    os.makedirs(DATA_68, exist_ok=True)
    HV.write_vitm_inputs(np.zeros(HV.n_dim(K_HDR)), K_HDR, hh, pe, labels, DATA_68, base, HV.RESTORE_B)
    open(ready, "w").close()
    return len(hh)


def run_score(theta, out_dir, label, n_hh):
    apply_theta(theta)
    env = os.environ.copy()
    for kk in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[kk] = "1"
    cmd = [VENV, "-m", "activitysim", "run",
           "-c", SPSA_OVERLAY, "-c", OVERLAY,
           "-c", os.path.join(LANE, "configs_mp"), "-c", os.path.join(LANE, "configs"),
           "-e", os.path.join(LANE, "extensions"), "-d", DATA_68, "-o", out_dir]
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w")
    p = subprocess.Popen(cmd, env=env, cwd=LANE, stdout=logf, stderr=subprocess.STDOUT)
    COST.RunCost(p, out_dir, label=label, cost_log=COST_LOG,
                 extra={"engine": "vitm", "method": "spsa"})
    p.wait(); logf.close()
    c, _ = HV.score_dir(out_dir, n_hh, False)
    return c if (c and np.isfinite(c)) else FAIL


def main():
    ap = argparse.ArgumentParser(description="SPSA calibration benchmark on the VITM 68 baseline")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--a", type=float, default=0.25)
    ap.add_argument("--c", type=float, default=0.08)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RUN_ROOT, exist_ok=True)

    print("[spsa-vitm] materialising fixed 68-baseline inputs ...", flush=True)
    n_hh = build_68_baseline(args.subsample)
    print(f"[spsa-vitm] {n_hh} hh; coeffs perturbed on the fixed 68 base (skim never rewritten)", flush=True)

    if args.smoke:
        c = run_score(np.zeros(DIM), os.path.join(RUN_ROOT, "spsa_vitm_smoke"), "spsa_vitm_smoke", n_hh)
        print(f"[spsa-vitm] SMOKE theta=0 -> composite={c:.2f} (should ~= 68 base)", flush=True)
        return

    theta = np.zeros(DIM); best, best_theta = FAIL, theta.copy()
    hist = []; rng = np.random.default_rng(0); t0 = time.time()
    for k in range(1, args.iters + 1):
        ak = args.a / ((k + args.A) ** 0.602)
        ck = args.c / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=DIM)
        cp = run_score(theta + ck * delta, os.path.join(RUN_ROOT, f"spsa_vitm_it{k:03d}_p"), f"spsa_vitm_it{k}_p", n_hh)
        cm = run_score(theta - ck * delta, os.path.join(RUN_ROOT, f"spsa_vitm_it{k:03d}_m"), f"spsa_vitm_it{k}_m", n_hh)
        # gradient ASCENT on the composite (the same objective HDR/PWS/SPE maximise)
        g = (cp - cm) / (2.0 * ck) * delta
        gnorm = np.linalg.norm(g)
        if gnorm > 5.0:
            g *= 5.0 / gnorm
        theta = np.clip(theta + ak * g, -4.0, 4.0)
        cur = max(cp, cm)
        if cur > best:
            best, best_theta = cur, theta.copy()
            np.savez(os.path.join(RUN_ROOT, "best_spsa_vitm.npz"), theta=best_theta, score=best)
        hist.append({"iter": k, "comp_plus": float(cp), "comp_minus": float(cm), "best": float(best)})
        json.dump(hist, open(os.path.join(RUN_ROOT, "history_spsa_vitm.json"), "w"), indent=2)
        print(f"[spsa-vitm] it {k:3d}/{args.iters} C+={cp:6.2f} C-={cm:6.2f} best={best:6.2f} "
              f"theta={np.round(theta,3)} elapsed={time.time()-t0:.0f}s", flush=True)
    COST.drain()
    cs = COST.summarize(COST_LOG)
    if cs:
        print(f"[spsa-vitm] cost: {cs['n_runs']} asim runs, mean {cs['wall_s_mean']}s/run, "
              f"peak RAM {cs['peak_rss_mb_max']}MB", flush=True)
    print(f"[spsa-vitm] DONE best composite={best:.2f}", flush=True)


if __name__ == "__main__":
    main()

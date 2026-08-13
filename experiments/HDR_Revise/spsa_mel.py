#!/usr/bin/env python3
r"""SPSA calibration on the Melbourne 57 baseline — the traditional-calibration
benchmark (Melbourne twin of spsa_vitm.py).

Same logic as spsa_vitm but on the Melbourne engine: the degraded 57-baseline INPUTS
are materialised once (faithful57 restore weights baked in, incl. weighted skim/land-
use), then SPSA tunes the MODEL COEFFICIENTS (mode ASCs + tour-frequency) by writing a
perturbed copy of configs_mel. Scored with the SAME 5-mode composite as HDR/PWS/SPE so
the number is directly comparable. Per-run cost (wall/RAM/disk) -> hdr_run/costs.jsonl.

    python spsa_mel.py --smoke           # theta=0 should reproduce ~the 57 baseline (5-mode)
    python spsa_mel.py --iters 30

theta (6 dims, IDENTICAL structure to spsa_vitm): [auto-driver, auto-passenger,
                 active, transit, mandatory tour-freq, non-mandatory tour-freq]
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hdr_driver as HD, hdr_mechanism as H
import cost_utils as COST
sys.path.insert(0, os.path.join(HERE, "..", "MainTrain"))
import score_simulation as MS
import score_vitm as V                      # 5-mode targets (MODE_PROBS_VITM)

RUN_ROOT     = HD.RUN_ROOT
CONFIGS_BASE = HD.CONFIGS                                       # configs_mel (HDR hooks)
SPSA_CFG     = os.path.join(RUN_ROOT, "configs_spsa_mel")       # perturbed copy
DATA_57      = os.path.join(RUN_ROOT, "spsa_mel_base")          # fixed 57-baseline inputs
COST_LOG     = os.path.join(RUN_ROOT, "costs.jsonl")
RESTORE57    = os.path.join(RUN_ROOT, "best_restore_faithful57.npz")
DIM          = 6                                                # symmetric with spsa_vitm
FAIL         = H.FAILURE_SCORE

# ---- 5-mode composite scorer (identical to the cross-dataset table) ----
MEL5 = {}
for raw, cat in MS.UNIFIED_MODE_MAPPING.items():
    if cat in ("Public Transit - Rail", "Public Transit - Bus"):
        MEL5[raw] = "Public Transit"
    elif cat == "Ride-Hailing & Taxi":
        continue
    else:
        MEL5[raw] = cat
MCATS = list(V.MODE_PROBS_VITM.keys()); MTGT = np.array(list(V.MODE_PROBS_VITM.values()))
PK = list(MS.OBSERVED_METRICS["purpose_probabilities"])
PTGT = np.array(list(MS.OBSERVED_METRICS["purpose_probabilities"].values()))
OBS_R = MS.OBSERVED_METRICS["trip_rate"]; W = MS.LOSS_WEIGHTS


def score5(out_dir, n):
    tp = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(tp):
        return FAIL
    t = pd.read_csv(tp); rate = len(t) / n
    tl = MS.calculate_trip_rate_loss(rate, OBS_R)
    ap = MS.get_probs(MS.map_column(t["purpose"], MS.PURPOSE_MAPPING), PK)
    pl = MS.calculate_dist_loss(ap, PTGT)
    mm = t["trip_mode"].map(MEL5).dropna()
    am = np.array([(mm == c).sum() / len(mm) for c in MCATS])
    ml = MS.calculate_dist_loss(am, MTGT)
    return float(100 * np.exp(-(W["trips"] * tl + W["purpose"] * pl + W["mode"] * ml)))


# ---- base coefficients (read once; perturbed copies go to SPSA_CFG) ----
MODE_BASE = pd.read_csv(os.path.join(CONFIGS_BASE, "trip_mode_choice_coefficients.csv"))
MAND_BASE = pd.read_csv(os.path.join(CONFIGS_BASE, "mandatory_tour_frequency_coefficients.csv"))
NONMAND = [f for f in os.listdir(CONFIGS_BASE)
           if f.startswith("non_mandatory_tour_frequency_coefficients_PTYPE") and not f.endswith("_base.csv")]
NONMAND_BASE = {f: pd.read_csv(os.path.join(CONFIGS_BASE, f)) for f in NONMAND}


def _free(df):
    if "constrain" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["constrain"].fillna("F").astype(str).str.upper().ne("T")


def apply_theta(theta):
    t_sov, t_sr, t_act, t_tr, t_mand, t_nonmand = theta
    mode = MODE_BASE.copy(); col = "coefficient_name"; fr = _free(mode)

    def add(mask, v):
        m = mask & fr
        mode.loc[m, "value"] = pd.to_numeric(mode.loc[m, "value"], errors="coerce").fillna(0.0) + float(v)

    add(mode[col].str.contains("sov", case=False), t_sov)
    add(mode[col].str.contains("sr2|sr3|shared|hov", case=False, regex=True), t_sr)
    add(mode[col].str.contains("bike|walk", case=False, regex=True)
        & ~mode[col].str.contains("trn|transit", case=False, regex=True), t_act)
    add(mode[col].str.contains("transit|wlk_trn|drv_trn|_trn_", case=False, regex=True), t_tr)
    mode.to_csv(os.path.join(SPSA_CFG, "trip_mode_choice_coefficients.csv"), index=False)

    mand = MAND_BASE.copy(); frm = _free(mand)
    msk = mand["coefficient_name"].str.contains("work|school|univ", case=False, regex=True) & frm
    mand.loc[msk, "value"] = pd.to_numeric(mand.loc[msk, "value"], errors="coerce").fillna(0.0) + float(t_mand)
    mand.to_csv(os.path.join(SPSA_CFG, "mandatory_tour_frequency_coefficients.csv"), index=False)

    # non-mandatory tour frequency (discretionary/maintenance/shopping) -> trip rate
    for f, base in NONMAND_BASE.items():
        df = base.copy(); fr2 = _free(df)
        msk2 = df["coefficient_name"].str.contains("discr|maint|shop|social|eat|escort",
                                                   case=False, regex=True) & fr2
        df.loc[msk2, "value"] = pd.to_numeric(df.loc[msk2, "value"], errors="coerce").fillna(0.0) + float(t_nonmand)
        df.to_csv(os.path.join(SPSA_CFG, f), index=False)


def setup(subsample):
    """Copy configs_mel -> SPSA_CFG once, and materialise the fixed 57-baseline data."""
    if not os.path.exists(os.path.join(SPSA_CFG, ".ready")):
        shutil.copytree(CONFIGS_BASE, SPSA_CFG, dirs_exist_ok=True)
        open(os.path.join(SPSA_CFG, ".ready"), "w").close()
    cfg = H.HDRConfig()
    if os.path.exists(os.path.join(DATA_57, ".ready")):
        return subsample
    hh_raw, persons = HD.load_baseline(); lu_raw = HD.load_landuse()
    srng = np.random.default_rng(0)
    pick = srng.choice(len(hh_raw), size=subsample, replace=False)
    idc = HD._hh_id_col(hh_raw)
    hh = hh_raw.iloc[pick].copy(); ids = set(hh[idc].tolist())
    pe = persons[persons["household_id"].isin(ids)].copy()
    k57 = int(np.load(RESTORE57, allow_pickle=True)["k"])
    idc2, label_by_id, k = HD.cluster_once(hh_raw, persons, k57)
    base_logw = HD._base_logw(RESTORE57, k, cfg)
    dim = k * (len(cfg.attrs) + len(cfg.person_attrs)) + len(cfg.lu_attrs) + len(cfg.skim_attrs)
    HD._mirror_baseline(DATA_57)
    HD.write_weighted_inputs(hh, lu_raw, pe, idc2, label_by_id, k, np.zeros(dim), DATA_57, cfg,
                             base_logw=base_logw)
    open(os.path.join(DATA_57, ".ready"), "w").close()
    return subsample


def run_score(theta, out_dir, label, n):
    apply_theta(theta)
    cmd = [HD._activitysim_exe(), "run", "-c", SPSA_CFG, "-d", DATA_57, "-o", out_dir]
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    COST.RunCost(p, out_dir, label=label, cost_log=COST_LOG,
                 extra={"engine": "melbourne", "method": "spsa"})
    p.wait(); logf.close()
    return score5(out_dir, n)


def main():
    ap = argparse.ArgumentParser(description="SPSA calibration benchmark on the Melbourne 57 baseline")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--subsample", type=int, default=5000)
    ap.add_argument("--a", type=float, default=0.25)
    ap.add_argument("--c", type=float, default=0.08)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RUN_ROOT, exist_ok=True)

    print("[spsa-mel] materialising fixed 57-baseline inputs ...", flush=True)
    n = setup(args.subsample)
    print(f"[spsa-mel] {n} hh; coeffs perturbed on the fixed 57 base (skim/land-use never rewritten)", flush=True)

    if args.smoke:
        c = run_score(np.zeros(DIM), os.path.join(RUN_ROOT, "spsa_mel_smoke"), "spsa_mel_smoke", n)
        print(f"[spsa-mel] SMOKE theta=0 -> 5-mode composite={c:.2f} (should ~= 57 base @5-mode ~59)", flush=True)
        return

    theta = np.zeros(DIM); best, best_theta = FAIL, theta.copy()
    hist = []; rng = np.random.default_rng(0); t0 = time.time()
    for k in range(1, args.iters + 1):
        ak = args.a / ((k + args.A) ** 0.602)
        ck = args.c / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=DIM)
        cp = run_score(theta + ck * delta, os.path.join(RUN_ROOT, f"spsa_mel_it{k:03d}_p"), f"spsa_mel_it{k}_p", n)
        cm = run_score(theta - ck * delta, os.path.join(RUN_ROOT, f"spsa_mel_it{k:03d}_m"), f"spsa_mel_it{k}_m", n)
        g = (cp - cm) / (2.0 * ck) * delta
        gnorm = np.linalg.norm(g)
        if gnorm > 5.0:
            g *= 5.0 / gnorm
        theta = np.clip(theta + ak * g, -4.0, 4.0)
        cur = max(cp, cm)
        if cur > best:
            best, best_theta = cur, theta.copy()
            np.savez(os.path.join(RUN_ROOT, "best_spsa_mel.npz"), theta=best_theta, score=best)
        hist.append({"iter": k, "comp_plus": float(cp), "comp_minus": float(cm), "best": float(best)})
        json.dump(hist, open(os.path.join(RUN_ROOT, "history_spsa_mel.json"), "w"), indent=2)
        print(f"[spsa-mel] it {k:3d}/{args.iters} C+={cp:6.2f} C-={cm:6.2f} best={best:6.2f} "
              f"theta={np.round(theta,3)} elapsed={time.time()-t0:.0f}s", flush=True)
    COST.drain()
    cs = COST.summarize(COST_LOG)
    if cs:
        print(f"[spsa-mel] cost: {cs['n_runs']} asim runs, mean {cs['wall_s_mean']}s/run, "
              f"peak RAM {cs['peak_rss_mb_max']}MB", flush=True)
    print(f"[spsa-mel] DONE best 5-mode composite={best:.2f}", flush=True)


if __name__ == "__main__":
    main()

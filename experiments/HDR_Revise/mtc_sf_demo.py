#!/usr/bin/env python3
r"""BPIR synthetic-truth validation on the San Francisco (MTC) example model.

Answers the editor's question "can the algorithm be applied to San Francisco itself?"
with a controlled IDENTICAL-TWIN (synthetic-truth) experiment — the standard validation
design in data assimilation, the tradition the paper itself invokes. The MTC example is
the estimation-context model (native SF coefficients, 5,000 households, 25 TAZ), so no
transfer misfit exists to correct; instead a KNOWN input-side error is injected and BPIR
must remove it. Because the truth is known, this validates what no real-data case can:
that the learned weights identify the injected error channels with the correct sign.

Protocol matches the paper (Methods 4.5-4.6): identical composite objective Eq.(1)
(w=0.25/0.40/0.35, alpha=2/4, FIVE mode categories), 20% of households withheld from
optimisation with reported scores on the hold-out, identical plausibility bounds, and
the identical mechanism code (hdr_driver / hdr_mechanism) with configs_mel — the same
SF-calibrated MTC configuration used throughout the paper, here on its home inputs.

  1. --targets   run the model at unit weights on the true SF inputs -> synthetic-truth
                 targets (trip rate, 9 purposes, 5 modes) + the 20% hold-out split.
  2. --degrade   inject the documented input-side error (bounds-edge attribute weights)
                 -> re-simulate -> the degraded baseline.
  3. --correct   HDR (CMA-ES) forward correction, optimised on the 80% training
                 households only.
  4. --apply-best  apply the learned weights to ALL households -> report the composite
                 on the full population AND on the 20% hold-out (the paper's protocol).
  5. --pws       Post-hoc Weighting Scheme floor on the degraded output (no re-sim).
  6. --report    summary + the weight-recovery diagnostic (learned vs injected).

    python mtc_sf_demo.py --targets
    python mtc_sf_demo.py --degrade
    python mtc_sf_demo.py --correct --iters 20 --workers 3
    python mtc_sf_demo.py --apply-best
    python mtc_sf_demo.py --pws
    python mtc_sf_demo.py --report
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "MainTrain"))
import hdr_driver as HD
import hdr_mechanism as H
import score_simulation as MS
from sklearn.cluster import KMeans

ROOT      = os.path.abspath(os.path.join(HERE, ".."))
MTC_DATA  = os.path.join(ROOT, "examples", "prototype_mtc", "data")
CONFIGS   = os.path.join(ROOT, "MainTrain", "configs_mel")   # SF-calibrated MTC configs + BPIR hooks
RUN_ROOT  = os.path.join(HERE, "mtc_run")
TARGETS   = os.path.join(RUN_ROOT, "mtc_targets.json")
DEGRADE_NPZ = os.path.join(RUN_ROOT, "degrade_mtc.npz")
FAIL      = H.FAILURE_SCORE
HOLDOUT_FRAC, HOLDOUT_SEED = 0.20, 0

# Paper's five mode categories (Eq. 1): MTC's raw modes aggregated exactly as Melbourne's
# were (rail+bus -> Public Transit; ride-hail excluded from the mode marginal).
MODE5 = {}
for raw, cat in MS.UNIFIED_MODE_MAPPING.items():
    if cat in ("Public Transit - Rail", "Public Transit - Bus"):
        MODE5[raw] = "Public Transit"
    elif cat == "Ride-Hailing & Taxi":
        continue
    else:
        MODE5[raw] = cat
W = MS.LOSS_WEIGHTS

# The injected input-side error (log-weights at the plausibility-box edges, so the
# inverse is inside the correction's own bounded search space): income deflated,
# worker propensity down, attractions deflated, driving slow+costly, transit/walk cheap.
DEG = {
    "income": -0.8, "work": -1.6, "lu": -1.2,
    "skim": {"autotime": +0.7, "hovtime": +0.7, "autocost": +0.7,
             "transit": -0.7, "nonmotor": -0.7},
}

_orig_mirror = HD._mirror_baseline


def _mirror_mtc(dst, extra_ignore=()):
    """Mirror + adapt layout: configs_mel expects omx/skims.omx; MTC ships it top-level.
    REAL COPY, never a hardlink — HDR rewrites the skim per candidate and a hardlink
    would corrupt the raw example file. The 25-zone skim is small (3.7 MB)."""
    import shutil
    _orig_mirror(dst, extra_ignore)
    omxd = os.path.join(dst, "omx"); os.makedirs(omxd, exist_ok=True)
    t = os.path.join(omxd, "skims.omx"); s = os.path.join(MTC_DATA, "skims.omx")
    if os.path.exists(t) and os.path.samefile(s, t):
        os.remove(t)
    if not os.path.exists(t):
        shutil.copyfile(s, t)


def _wire():
    HD.BASELINE_DATA = MTC_DATA
    HD.CONFIGS = CONFIGS
    HD.RUN_ROOT = RUN_ROOT
    HD._mirror_baseline = _mirror_mtc
    HD._baseline_skims = lambda: os.path.join(MTC_DATA, "skims.omx")
    os.makedirs(RUN_ROOT, exist_ok=True)


def _tg():
    return json.load(open(TARGETS))


def score_mtc(trips, n_hh):
    """The paper's Eq.(1) composite against the SF synthetic-truth targets: log-cosh on
    relative trip-rate error (alpha 2), purpose (9) and FIVE-mode deltas (alpha 4)."""
    t = _tg()
    PK = list(t["purpose_probabilities"]); PT = np.array(list(t["purpose_probabilities"].values()))
    MK = list(t["mode_probabilities"]);    MT = np.array(list(t["mode_probabilities"].values()))
    rate = len(trips) / n_hh
    tl = MS.calculate_trip_rate_loss(rate, t["trip_rate"])
    ap = MS.get_probs(MS.map_column(trips["purpose"], MS.PURPOSE_MAPPING), PK)
    pl = MS.calculate_dist_loss(ap, PT)
    mm = trips["trip_mode"].map(MODE5).dropna()
    am = np.array([(mm == c).sum() / max(len(mm), 1) for c in MK])
    ml = MS.calculate_dist_loss(am, MT)
    comp = float(100 * np.exp(-(W["trips"] * tl + W["purpose"] * pl + W["mode"] * ml)))
    return comp, {"composite_score": comp, "trip_rate_actual": rate, "trip_rate_target": t["trip_rate"],
                  "trip_rate_score": float(100 * np.exp(-tl / W["trips"])),
                  "purpose_score": float(100 * np.exp(-pl / W["purpose"])),
                  "mode_score": float(100 * np.exp(-ml / W["mode"]))}


def score_run_mtc(out_dir, n_hh):
    tp = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(tp):
        return FAIL, None
    return score_mtc(pd.read_csv(tp), n_hh)


def _holdout_split(hh):
    idc = HD._hh_id_col(hh)
    rng = np.random.default_rng(HOLDOUT_SEED)
    ids = hh[idc].to_numpy()
    hold = set(rng.choice(ids, size=int(len(ids) * HOLDOUT_FRAC), replace=False).tolist())
    return idc, hold


def run_targets():
    """Unit-weight run on the TRUE SF inputs -> synthetic-truth targets + hold-out split."""
    _wire()
    hh, pe = HD.load_baseline(); lu = HD.load_landuse()
    data_dir = os.path.join(RUN_ROOT, "data_targets"); HD._mirror_baseline(data_dir)
    HD.write_baseline_inputs(hh, lu, pe, data_dir, H.HDRConfig())
    out = os.path.join(RUN_ROOT, "output_targets")
    if not HD.run_activitysim(data_dir, out, os.path.join(RUN_ROOT, "asim_targets.log")):
        print("[mtc] target run FAILED — see asim_targets.log"); sys.exit(1)
    trips = pd.read_csv(os.path.join(out, "final_trips.csv"))
    purpose = MS.map_column(trips["purpose"], MS.PURPOSE_MAPPING).value_counts(normalize=True)
    mode = trips["trip_mode"].map(MODE5).dropna().value_counts(normalize=True)
    idc, hold = _holdout_split(hh)
    t = {"trip_rate": round(len(trips) / len(hh), 5),
         "n_households": len(hh), "n_trips": len(trips),
         "purpose_probabilities": {k: round(float(v), 6) for k, v in purpose.items()},
         "mode_probabilities": {k: round(float(v), 6) for k, v in mode.items()},
         "holdout_household_ids": sorted(int(x) for x in hold)}
    json.dump(t, open(TARGETS, "w"), indent=2)
    print(f"[mtc] SF synthetic truth: {len(hh)} hh, trip_rate={t['trip_rate']:.3f}, "
          f"5 modes; hold-out {len(hold)} hh (20%) -> {TARGETS}")


def _deg_theta(k, cfg):
    nA, nP = len(cfg.attrs), len(cfg.person_attrs)
    w = []
    for _ in range(k):
        w += [DEG["income"]] * nA + [DEG["work"]] * nP
    w += [DEG["lu"]] * len(cfg.lu_attrs)
    w += [DEG["skim"].get(a, 0.0) for a in cfg.skim_attrs]
    return np.array(w, dtype=float)


def run_degrade():
    """Inject the documented input-side error -> re-sim -> degraded SF baseline."""
    _wire()
    cfg = H.HDRConfig()
    hh, pe = HD.load_baseline(); lu = HD.load_landuse()
    idc, label_by_id, k = HD.cluster_once(hh, pe, 6)
    theta = _deg_theta(k, cfg)
    data_dir = os.path.join(RUN_ROOT, "data_degraded"); HD._mirror_baseline(data_dir)
    HD.write_weighted_inputs(hh, lu, pe, idc, label_by_id, k, theta, data_dir, cfg)
    out = os.path.join(RUN_ROOT, "output_degraded")
    if not HD.run_activitysim(data_dir, out, os.path.join(RUN_ROOT, "asim_degraded.log")):
        print("[mtc] degrade run FAILED"); sys.exit(1)
    score, res = score_run_mtc(out, len(hh))
    np.savez(DEGRADE_NPZ, w=theta, k=k, attrs=np.array(cfg.attrs),
             person_attrs=np.array(cfg.person_attrs), lu_attrs=np.array(cfg.lu_attrs),
             skim_attrs=np.array(cfg.skim_attrs),
             label_by_id=label_by_id.values, id_index=label_by_id.index.values,
             score=score)
    print(f"[mtc] DEGRADED SF baseline (Eq.1, 5-mode): composite={score:.2f} "
          f"(trip={res['trip_rate_score']:.0f} purpose={res['purpose_score']:.0f} "
          f"mode={res['mode_score']:.0f})  rate={res['trip_rate_actual']:.2f}")


def run_correct(iters, popsize, workers):
    """HDR (CMA-ES) forward correction, optimised on the 80% TRAINING households only."""
    _wire()
    cfg = H.HDRConfig()
    hh, pe = HD.load_baseline(); lu = HD.load_landuse()
    idc, label_by_id, k = HD.cluster_once(hh, pe, 6)
    base_logw = HD._base_logw(DEGRADE_NPZ, k, cfg)
    hold = set(_tg()["holdout_household_ids"])
    hh_tr = hh[~hh[idc].isin(hold)].copy()
    pe_tr = pe[pe["household_id"].isin(set(hh_tr[idc]))].copy()
    print(f"[mtc] optimising on {len(hh_tr)} training hh ({len(hold)} withheld)")
    dim = (k * (len(cfg.attrs) + len(cfg.person_attrs))
           + len(cfg.lu_attrs) + len(cfg.skim_attrs))
    olo, ohi = cfg.lu_weight_log_clip
    opt = H.make_optimiser("cmaes", dim, popsize, bounds=(olo, ohi), sigma0=(ohi - olo) / 4)
    ws = HD.setup_workers(max(1, min(workers, popsize)), pe_tr, "mtc")
    best, best_w, hist = FAIL, None, []
    t0 = time.time()
    for it in range(iters):
        cands = opt.ask()
        scores, _ = HD.evaluate_batch(cands, hh_tr, lu, pe_tr, idc, label_by_id, k, cfg, ws,
                                      tagbase=f"mtc_it{it}", scorer=score_run_mtc,
                                      base_logw=base_logw)
        opt.tell(cands, scores)
        w_b, s_b = opt.best_solution
        if s_b > best:
            best, best_w = s_b, np.asarray(w_b).copy()
            np.savez(os.path.join(RUN_ROOT, "best_mtc_cmaes.npz"), w=best_w, k=k, score=best)
        hist.append({"iter": it, "best": float(best), "iter_max": float(np.max(scores)),
                     "iter_mean": float(np.mean(scores))})
        json.dump(hist, open(os.path.join(RUN_ROOT, "history_mtc_cmaes.json"), "w"), indent=2)
        print(f"[mtc] it {it:3d}/{iters} best={best:6.2f} iter_max={np.max(scores):6.2f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        if opt.converged:
            print("[mtc] converged."); break
    print(f"[mtc] training-set corrected composite = {best:.2f}; "
          f"now run --apply-best for the hold-out score (the reported number)")


def run_apply_best():
    """Apply the learned weights to ALL households; report full + 20% hold-out scores."""
    _wire()
    cfg = H.HDRConfig()
    hh, pe = HD.load_baseline(); lu = HD.load_landuse()
    idc, label_by_id, k = HD.cluster_once(hh, pe, 6)
    base_logw = HD._base_logw(DEGRADE_NPZ, k, cfg)
    d = np.load(os.path.join(RUN_ROOT, "best_mtc_cmaes.npz"), allow_pickle=True)
    w = d["w"]
    data_dir = os.path.join(RUN_ROOT, "data_best"); HD._mirror_baseline(data_dir)
    HD.write_weighted_inputs(hh, lu, pe, idc, label_by_id, k, w, data_dir, cfg,
                             base_logw=base_logw)
    out = os.path.join(RUN_ROOT, "output_best")
    if not HD.run_activitysim(data_dir, out, os.path.join(RUN_ROOT, "asim_best.log")):
        print("[mtc] apply-best run FAILED"); sys.exit(1)
    trips = pd.read_csv(os.path.join(out, "final_trips.csv"))
    full, resF = score_mtc(trips, len(hh))
    hold = set(_tg()["holdout_household_ids"])
    trips_h = trips[trips["household_id"].isin(hold)]
    heldout, resH = score_mtc(trips_h, len(hold))
    json.dump({"full_population": resF, "holdout_20pct": resH},
              open(os.path.join(RUN_ROOT, "apply_best_scores.json"), "w"), indent=2)
    print(f"[mtc] APPLY-BEST full population ({len(hh)} hh): composite={full:.2f} "
          f"(trip={resF['trip_rate_score']:.0f} purpose={resF['purpose_score']:.0f} mode={resF['mode_score']:.0f})")
    print(f"[mtc] APPLY-BEST 20% HOLD-OUT ({len(hold)} hh, never seen by the optimiser): "
          f"composite={heldout:.2f} "
          f"(trip={resH['trip_rate_score']:.0f} purpose={resH['purpose_score']:.0f} mode={resH['mode_score']:.0f})")


def run_pws(k=8, bound=0.5, iters=60):
    """Post-hoc Weighting Scheme floor on the DEGRADED output (deterministic IPF, no re-sim)."""
    _wire()
    t = _tg()
    trips = pd.read_csv(os.path.join(RUN_ROOT, "output_degraded", "final_trips.csv"))
    hh = pd.read_csv(os.path.join(RUN_ROOT, "data_degraded", "households.csv"))
    idcol = HD._hh_id_col(hh)
    hh_ids = np.sort(hh[idcol].unique())
    OBS_R = t["trip_rate"]
    PK = list(t["purpose_probabilities"]); PT = np.array(list(t["purpose_probabilities"].values()))
    MK = list(t["mode_probabilities"]);    MT = np.array(list(t["mode_probabilities"].values()))
    tt = trips.copy()
    tt["_p"] = MS.map_column(tt["purpose"], MS.PURPOSE_MAPPING)
    tt["_m"] = tt["trip_mode"].map(MODE5)
    tc = tt.groupby("household_id").size().reindex(hh_ids, fill_value=0).to_numpy(float)
    pm = pd.crosstab(tt["household_id"], tt["_p"]).reindex(index=hh_ids, columns=PK, fill_value=0).to_numpy(float)
    mm = pd.crosstab(tt["household_id"], tt["_m"]).reindex(index=hh_ids, columns=MK, fill_value=0).to_numpy(float)
    mc = mm.sum(1)
    psh = np.divide(pm, tc[:, None], out=np.zeros_like(pm), where=tc[:, None] > 0)
    msh = np.divide(mm, np.clip(mc[:, None], 1, None))
    lab = KMeans(k, random_state=42, n_init=4).fit_predict(np.column_stack([tc / max(tc.max(), 1), psh, msh]))
    ct = np.array([tc[lab == c].sum() for c in range(k)]); cn = np.array([(lab == c).sum() for c in range(k)], float)
    cp = np.array([pm[lab == c].sum(0) for c in range(k)]); cmm = np.array([mm[lab == c].sum(0) for c in range(k)])
    cmc = np.array([mc[lab == c].sum() for c in range(k)])
    cps = cp / np.clip(ct[:, None], 1, None); cms = cmm / np.clip(cmc[:, None], 1, None)
    crate = ct / np.clip(cn, 1, None)
    def comp(wv):
        s_t = (wv * ct).sum(); s_n = (wv * cn).sum(); s_m = (wv * cmc).sum()
        if s_t <= 0 or s_n <= 0 or s_m <= 0:
            return -10.0
        l = (W["trips"] * MS.calculate_trip_rate_loss(s_t / s_n, OBS_R)
             + W["purpose"] * MS.calculate_dist_loss((wv[:, None] * cp).sum(0) / s_t, PT)
             + W["mode"] * MS.calculate_dist_loss((wv[:, None] * cmm).sum(0) / s_m, MT))
        return float(100 * np.exp(-l))
    lw = np.zeros(k); base = comp(np.ones(k)); best = base
    hist = [{"iter": 0, "best": round(base, 3)}]
    for i in range(1, iters + 1):
        wv = np.exp(np.clip(lw, -bound, bound)); ps = (wv[:, None] * cp).sum(0) / (wv * ct).sum()
        lw = np.clip(lw + 0.5 * (cps * np.log(np.clip(PT, 1e-9, None) / np.clip(ps, 1e-9, None))).sum(1), -bound, bound)
        wv = np.exp(np.clip(lw, -bound, bound)); ms = (wv[:, None] * cmm).sum(0) / (wv * cmc).sum()
        lw = np.clip(lw + 0.5 * (cms * np.log(np.clip(MT, 1e-9, None) / np.clip(ms, 1e-9, None))).sum(1), -bound, bound)
        wv = np.exp(np.clip(lw, -bound, bound)); rr = (wv * ct).sum() / (wv * cn).sum()
        lw = np.clip(lw + 0.3 * np.sign(OBS_R - rr) * np.sign(crate - rr)
                     * np.abs(np.log(np.clip(crate, 1e-9, None) / max(rr, 1e-9))), -bound, bound)
        c = comp(np.exp(np.clip(lw, -bound, bound)))
        best = max(best, c); hist.append({"iter": i, "best": round(best, 3)})
    json.dump(hist, open(os.path.join(RUN_ROOT, "history_mtc_pws.json"), "w"), indent=2)
    print(f"[mtc] PWS floor: degraded {base:.2f} -> raked {best:.2f} (+{best-base:.2f}, no re-sim)")


def run_report():
    """Summary + the weight-recovery diagnostic (learned correction vs injected error)."""
    _wire()
    print("\n=== BPIR on San Francisco (MTC example) — synthetic-truth validation ===")
    if os.path.exists(TARGETS):
        print(f"  {'Synthetic truth (unit weights)':46s} 100.00  (by construction)")
    if os.path.exists(DEGRADE_NPZ):
        print(f"  {'Degraded baseline (injected input error)':46s} {float(np.load(DEGRADE_NPZ)['score']):6.2f}")
    p = os.path.join(RUN_ROOT, "history_mtc_pws.json")
    if os.path.exists(p):
        print(f"  {'PWS floor (no re-simulation)':46s} {json.load(open(p))[-1]['best']:6.2f}")
    p = os.path.join(RUN_ROOT, "apply_best_scores.json")
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"  {'HDR corrected — full population':46s} {s['full_population']['composite_score']:6.2f}")
        print(f"  {'HDR corrected — 20% hold-out (unseen hh)':46s} {s['holdout_20pct']['composite_score']:6.2f}")
    # weight-recovery diagnostic: does the learned correction invert the injected error?
    bp = os.path.join(RUN_ROOT, "best_mtc_cmaes.npz")
    if os.path.exists(bp) and os.path.exists(DEGRADE_NPZ):
        cfg = H.HDRConfig()
        d = np.load(bp, allow_pickle=True); w = d["w"]; k = int(d["k"])
        nA, nP = len(cfg.attrs), len(cfg.person_attrs)
        hh_block = w[:k * (nA + nP)].reshape(k, nA + nP)
        inc = float(np.clip(hh_block[:, 0], *cfg.weight_log_clip).mean())
        wk = float(np.clip(hh_block[:, nA], *cfg.work_log_clip).mean()) if nP else 0.0
        lu_ = w[k * (nA + nP): k * (nA + nP) + len(cfg.lu_attrs)]
        lu = float(np.clip(lu_, *cfg.lu_weight_log_clip).mean())
        sk = np.clip(w[-len(cfg.skim_attrs):], *cfg.skim_log_clip)
        print("\n  Weight-recovery diagnostic (learned correction vs injected error; "
              "perfect recovery = opposite sign, similar magnitude):")
        print(f"    {'channel':10s} {'injected':>9s} {'learned':>9s}")
        print(f"    {'income':10s} {DEG['income']:+9.2f} {inc:+9.2f}")
        print(f"    {'work':10s} {DEG['work']:+9.2f} {wk:+9.2f}")
        print(f"    {'land use':10s} {DEG['lu']:+9.2f} {lu:+9.2f}")
        for j, a in enumerate(cfg.skim_attrs):
            print(f"    {'skim:'+a:10s} {DEG['skim'].get(a,0.0):+9.2f} {float(sk[j]):+9.2f}")


def main():
    ap = argparse.ArgumentParser(description="BPIR synthetic-truth validation on the SF (MTC) example")
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--apply-best", dest="apply_best", action="store_true")
    ap.add_argument("--pws", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.targets: run_targets()
    if a.degrade: run_degrade()
    if a.correct: run_correct(a.iters, a.popsize, a.workers)
    if a.apply_best: run_apply_best()
    if a.pws: run_pws()
    if a.report: run_report()
    if not any([a.targets, a.degrade, a.correct, a.apply_best, a.pws, a.report]):
        ap.print_help()


if __name__ == "__main__":
    main()

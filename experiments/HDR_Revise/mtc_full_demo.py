#!/usr/bin/env python3
r"""BPIR on the FULL-SCALE San Francisco MTC model against OBSERVED Bay Area targets.

The supplementary demonstration requested by the editor: the identical BPIR code applied
to the full 1,475-zone ActivitySim MTC example (2.88 M synthetic households) and scored
against the OBSERVED year-2000 Bay Area marginals from MTC's own calibration report
(Table 55 trip-mode-by-purpose targets; Table 8 households) — real observed aggregates,
exactly the two components the paper's framework requires (Methods 4.1). The ActivitySim
re-implementation was never re-calibrated to these targets, so its natural residual
misfit (baseline ~95.2: purpose 78.5, mode 91.8) is a genuine uncorrected starting point.

Protocol mirrors the paper: identical Eq.(1) composite (5 modes; purposes scored on
primary_purpose = tour purpose, matching the report's accounting; ride-hail excluded);
identical bounded HDR weighting; optimisation on a 50k-household representative sample
with the learned weights verified on a DISJOINT sample (the paper's generalisation
protocol); PWS as the no-re-simulation floor. ~12.5 min & 8.5 GB per evaluation.

    python mtc_full_demo.py --baseline              # score the uncorrected model
    python mtc_full_demo.py --correct --iters 15 --workers 3
    python mtc_full_demo.py --pws
    python mtc_full_demo.py --verify                # weights -> disjoint sample
    python mtc_full_demo.py --report
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

ROOT     = os.path.abspath(os.path.join(HERE, ".."))
FULL     = os.path.join(ROOT, "examples", "prototype_mtc_full")
FULL_DATA = os.path.join(FULL, "data")
CONFIGS  = os.path.join(FULL, "configs_bpir")       # stock full configs + BPIR hooks
RUN_ROOT = os.path.join(HERE, "mtc_full_run")
TARGETS  = os.path.join(RUN_ROOT, "mtc_full_targets.json")
FAIL     = H.FAILURE_SCORE
SAMPLE_N, VERIFY_N = 15_000, 45_000     # optimisation sample / disjoint verification sample
SAMPLE_SEED, VERIFY_SEED = 0, 1         # (50k x 3 workers thrashed disk+CPU: ~2 h/iter)

# Paper's five mode categories + the full model's pay/toll variants (absent in Melbourne)
MODE5 = {}
for raw, cat in MS.UNIFIED_MODE_MAPPING.items():
    if cat in ("Public Transit - Rail", "Public Transit - Bus"):
        MODE5[raw] = "Public Transit"
    elif cat == "Ride-Hailing & Taxi":
        continue                                   # not in the 2000 observations
    else:
        MODE5[raw] = cat
MODE5.update({"DRIVEALONEPAY": "Private Vehicle - Driver",
              "SHARED2PAY": "Private Vehicle - Passenger",
              "SHARED3PAY": "Private Vehicle - Passenger"})
W = MS.LOSS_WEIGHTS


_orig_mirror = HD._mirror_baseline


def _mirror_full(dst, extra_ignore=()):
    """Mirror WITHOUT the 734 MB top-level skims.omx: configs_bpir reads omx/skims.omx
    (so weighted skims are honoured), and _write_weighted_skims creates that file per
    candidate. Dirs that never rewrite skims (baseline/verify) get a REAL copy — never
    a hardlink, since _write_weighted_skims overwrites dst in-place."""
    _orig_mirror(dst, extra_ignore=tuple(extra_ignore) + ("skims.omx",))


def _ensure_raw_skim(data_dir):
    import shutil
    omxd = os.path.join(data_dir, "omx"); os.makedirs(omxd, exist_ok=True)
    t = os.path.join(omxd, "skims.omx")
    if not os.path.exists(t):
        shutil.copyfile(os.path.join(FULL_DATA, "skims.omx"), t)


def _wire():
    HD.BASELINE_DATA = FULL_DATA
    HD.CONFIGS = CONFIGS
    HD.RUN_ROOT = RUN_ROOT
    HD._mirror_baseline = _mirror_full
    HD._baseline_skims = lambda: os.path.join(FULL_DATA, "skims.omx")
    os.makedirs(RUN_ROOT, exist_ok=True)


def _tg():
    return json.load(open(TARGETS))


def score_mtc(trips, n_hh):
    """Eq.(1) against the observed Bay Area targets (purpose on primary_purpose)."""
    t = _tg()
    PK = list(t["purpose_probabilities"]); PT = np.array(list(t["purpose_probabilities"].values()))
    MK = list(t["mode_probabilities"]);    MT = np.array(list(t["mode_probabilities"].values()))
    rate = len(trips) / n_hh
    tl = MS.calculate_trip_rate_loss(rate, t["trip_rate"])
    ap = MS.get_probs(MS.map_column(trips["primary_purpose"], MS.PURPOSE_MAPPING), PK)
    pl = MS.calculate_dist_loss(ap, PT)
    mm = trips["trip_mode"].map(MODE5).dropna()
    am = np.array([(mm == c).sum() / max(len(mm), 1) for c in MK])
    ml = MS.calculate_dist_loss(am, MT)
    comp = float(100 * np.exp(-(W["trips"] * tl + W["purpose"] * pl + W["mode"] * ml)))
    return comp, {"composite_score": comp, "trip_rate_actual": rate,
                  "trip_rate_target": t["trip_rate"],
                  "trip_rate_score": float(100 * np.exp(-tl / W["trips"])),
                  "purpose_score": float(100 * np.exp(-pl / W["purpose"])),
                  "mode_score": float(100 * np.exp(-ml / W["mode"]))}


def score_run_mtc(out_dir, n_hh):
    tp = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(tp):
        return FAIL, None
    return score_mtc(pd.read_csv(tp, low_memory=False), n_hh)


def _subsample(n, seed):
    """Representative household sample + their persons (raw schema, untouched values)."""
    hh = pd.read_csv(os.path.join(FULL_DATA, "households.csv"))
    pe = pd.read_csv(os.path.join(FULL_DATA, "persons.csv"))
    idc = HD._hh_id_col(hh)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(hh), size=n, replace=False)
    hh_s = hh.iloc[pick].copy()
    ids = set(hh_s[idc].tolist())
    pe_s = pe[pe["household_id"].isin(ids)].copy()
    return hh_s, pe_s


def run_baseline():
    """Unit-weight run of the sample through the hooked configs -> uncorrected score.
    Also validates that the BPIR hooks reproduce the stock model at weights=1."""
    _wire()
    cfg = H.HDRConfig()
    hh, pe = _subsample(SAMPLE_N, SAMPLE_SEED)
    lu = HD.load_landuse()
    data_dir = os.path.join(RUN_ROOT, "data_baseline"); HD._mirror_baseline(data_dir)
    _ensure_raw_skim(data_dir)
    HD.write_baseline_inputs(hh, lu, pe, data_dir, cfg)
    out = os.path.join(RUN_ROOT, "output_baseline")
    if not HD.run_activitysim(data_dir, out, os.path.join(RUN_ROOT, "asim_baseline.log")):
        print("[mtc-full] baseline FAILED — see asim_baseline.log"); sys.exit(1)
    score, res = score_run_mtc(out, len(hh))
    json.dump(res, open(os.path.join(RUN_ROOT, "baseline_score.json"), "w"), indent=2)
    print(f"[mtc-full] UNCORRECTED SF baseline ({len(hh)} hh sample): composite={score:.2f} "
          f"(trip={res['trip_rate_score']:.0f} purpose={res['purpose_score']:.0f} "
          f"mode={res['mode_score']:.0f})  rate={res['trip_rate_actual']:.2f} vs {res['trip_rate_target']:.2f}")


def run_correct(iters, popsize, workers):
    """HDR forward correction of the NATURAL misfit (no degradation, no base weights)."""
    _wire()
    cfg = H.HDRConfig()
    hh, pe = _subsample(SAMPLE_N, SAMPLE_SEED)
    lu = HD.load_landuse()
    idc, label_by_id, k = HD.cluster_once(hh, pe, 6)
    dim = (k * (len(cfg.attrs) + len(cfg.person_attrs))
           + len(cfg.lu_attrs) + len(cfg.skim_attrs))
    olo, ohi = cfg.lu_weight_log_clip
    opt = H.make_optimiser("cmaes", dim, popsize, bounds=(olo, ohi), sigma0=(ohi - olo) / 4)
    ws = HD.setup_workers(max(1, min(workers, popsize)), pe, "mtcfull")
    best, best_w, hist = FAIL, None, []
    t0 = time.time()
    for it in range(iters):
        cands = opt.ask()
        scores, _ = HD.evaluate_batch(cands, hh, lu, pe, idc, label_by_id, k, cfg, ws,
                                      tagbase=f"mtcfull_it{it}", scorer=score_run_mtc)
        opt.tell(cands, scores)
        w_b, s_b = opt.best_solution
        if s_b > best:
            best, best_w = s_b, np.asarray(w_b).copy()
            np.savez(os.path.join(RUN_ROOT, "best_mtcfull_cmaes.npz"), w=best_w, k=k,
                     label_by_id=label_by_id.values, id_index=label_by_id.index.values,
                     score=best)
        hist.append({"iter": it, "best": float(best), "iter_max": float(np.max(scores)),
                     "iter_mean": float(np.mean(scores))})
        json.dump(hist, open(os.path.join(RUN_ROOT, "history_mtcfull_cmaes.json"), "w"), indent=2)
        print(f"[mtc-full] it {it:3d}/{iters} best={best:6.2f} iter_max={np.max(scores):6.2f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        if opt.converged:
            print("[mtc-full] converged."); break
    print(f"[mtc-full] corrected (training sample) composite = {best:.2f}; run --verify next")


def run_verify():
    """Apply the learned weights to a DISJOINT, larger sample (paper's generalisation)."""
    _wire()
    cfg = H.HDRConfig()
    d = np.load(os.path.join(RUN_ROOT, "best_mtcfull_cmaes.npz"), allow_pickle=True)
    w, k = d["w"], int(d["k"])
    hh, pe = _subsample(VERIFY_N, VERIFY_SEED)
    lu = HD.load_landuse()
    idc, label_by_id, _ = HD.cluster_once(hh, pe, k)
    data_dir = os.path.join(RUN_ROOT, "data_verify"); HD._mirror_baseline(data_dir)
    HD.write_weighted_inputs(hh, lu, pe, idc, label_by_id, k, w, data_dir, cfg)
    out = os.path.join(RUN_ROOT, "output_verify")
    if not HD.run_activitysim(data_dir, out, os.path.join(RUN_ROOT, "asim_verify.log")):
        print("[mtc-full] verify FAILED"); sys.exit(1)
    score, res = score_run_mtc(out, len(hh))
    json.dump(res, open(os.path.join(RUN_ROOT, "verify_score.json"), "w"), indent=2)
    print(f"[mtc-full] VERIFY on disjoint {VERIFY_N//1000}k sample (seed {VERIFY_SEED}): "
          f"composite={score:.2f} (trip={res['trip_rate_score']:.0f} "
          f"purpose={res['purpose_score']:.0f} mode={res['mode_score']:.0f})")


def run_pws(k=8, bound=0.5, iters=60):
    """Post-hoc Weighting Scheme floor on the BASELINE output (no re-simulation)."""
    _wire()
    t = _tg()
    trips = pd.read_csv(os.path.join(RUN_ROOT, "output_baseline", "final_trips.csv"), low_memory=False)
    hh = pd.read_csv(os.path.join(RUN_ROOT, "data_baseline", "households.csv"))
    idcol = HD._hh_id_col(hh)
    hh_ids = np.sort(hh[idcol].unique())
    OBS_R = t["trip_rate"]
    PK = list(t["purpose_probabilities"]); PT = np.array(list(t["purpose_probabilities"].values()))
    MK = list(t["mode_probabilities"]);    MT = np.array(list(t["mode_probabilities"].values()))
    tt = trips.copy()
    tt["_p"] = MS.map_column(tt["primary_purpose"], MS.PURPOSE_MAPPING)
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
    def parts(wv):
        s_t = (wv * ct).sum(); s_n = (wv * cn).sum(); s_m = (wv * cmc).sum()
        if s_t <= 0 or s_n <= 0 or s_m <= 0:
            return None
        tl = MS.calculate_trip_rate_loss(s_t / s_n, OBS_R)
        pl = MS.calculate_dist_loss((wv[:, None] * cp).sum(0) / s_t, PT)
        ml = MS.calculate_dist_loss((wv[:, None] * cmm).sum(0) / s_m, MT)
        return {"composite_score": float(100 * np.exp(-(W["trips"] * tl + W["purpose"] * pl + W["mode"] * ml))),
                "trip_rate_actual": float(s_t / s_n),
                "trip_rate_score": float(100 * np.exp(-tl / W["trips"])),
                "purpose_score": float(100 * np.exp(-pl / W["purpose"])),
                "mode_score": float(100 * np.exp(-ml / W["mode"]))}
    def comp(wv):
        r = parts(wv)
        return r["composite_score"] if r else -10.0
    lw = np.zeros(k); base = comp(np.ones(k)); best = base; best_lw = lw.copy()
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
        if c > best:
            best, best_lw = c, lw.copy()
        hist.append({"iter": i, "best": round(best, 3)})
    json.dump(hist, open(os.path.join(RUN_ROOT, "history_mtcfull_pws.json"), "w"), indent=2)
    res = parts(np.exp(np.clip(best_lw, -bound, bound)))
    json.dump(res, open(os.path.join(RUN_ROOT, "pws_score.json"), "w"), indent=2)
    print(f"[mtc-full] PWS floor: baseline {base:.2f} -> raked {best:.2f} (+{best-base:.2f}, no re-sim) "
          f"(trip={res['trip_rate_score']:.0f} purpose={res['purpose_score']:.0f} mode={res['mode_score']:.0f})")


def run_report():
    _wire()
    cfg = H.HDRConfig()
    print("\n=== BPIR on the full-scale San Francisco MTC model (observed 2000 targets) ===")
    p = os.path.join(RUN_ROOT, "baseline_score.json")
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"  {'Uncorrected baseline (estimated, never re-calibrated)':56s} {s['composite_score']:6.2f}"
              f"   (trip {s['trip_rate_score']:.0f} / purpose {s['purpose_score']:.0f} / mode {s['mode_score']:.0f})")
    p = os.path.join(RUN_ROOT, "pws_score.json")
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"  {'PWS floor (no re-simulation)':56s} {s['composite_score']:6.2f}"
              f"   (trip {s['trip_rate_score']:.0f} / purpose {s['purpose_score']:.0f} / mode {s['mode_score']:.0f})")
    elif os.path.exists(os.path.join(RUN_ROOT, "history_mtcfull_pws.json")):
        print(f"  {'PWS floor (no re-simulation)':56s} "
              f"{json.load(open(os.path.join(RUN_ROOT, 'history_mtcfull_pws.json')))[-1]['best']:6.2f}")
    p = os.path.join(RUN_ROOT, "best_mtcfull_cmaes.npz")
    if os.path.exists(p):
        print(f"  {'HDR corrected (training sample)':56s} {float(np.load(p, allow_pickle=True)['score']):6.2f}")
    p = os.path.join(RUN_ROOT, "verify_score.json")
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"  {'HDR corrected - DISJOINT ' + str(VERIFY_N//1000) + 'k verification sample':56s} {s['composite_score']:6.2f}"
              f"   (trip {s['trip_rate_score']:.0f} / purpose {s['purpose_score']:.0f} / mode {s['mode_score']:.0f})")
    # weight readout: where does the learned correction say the misfit lives?
    bp = os.path.join(RUN_ROOT, "best_mtcfull_cmaes.npz")
    if os.path.exists(bp):
        d = np.load(bp, allow_pickle=True); w = d["w"]; k = int(d["k"])
        nA, nP = len(cfg.attrs), len(cfg.person_attrs)
        blk = w[:k * (nA + nP)].reshape(k, nA + nP)
        print("\n  Learned weight readout (mean log-weight per channel; sign = direction of correction):")
        for j, a in enumerate(cfg.attrs):
            print(f"    hh:{a:12s} {float(np.clip(blk[:, j], *cfg.weight_log_clip).mean()):+7.2f}")
        for j, a in enumerate(cfg.person_attrs):
            print(f"    person:{a:8s} {float(np.clip(blk[:, nA + j], *cfg.work_log_clip).mean()):+7.2f}")
        lu = w[k * (nA + nP): k * (nA + nP) + len(cfg.lu_attrs)]
        for j, a in enumerate(cfg.lu_attrs):
            print(f"    lu:{a:12s} {float(np.clip(lu[j], *cfg.lu_weight_log_clip)):+7.2f}")
        sk = np.clip(w[-len(cfg.skim_attrs):], *cfg.skim_log_clip)
        for j, a in enumerate(cfg.skim_attrs):
            print(f"    skim:{a:10s} {float(sk[j]):+7.2f}")


def main():
    ap = argparse.ArgumentParser(description="BPIR on the full-scale SF MTC model, observed targets")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--pws", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.baseline: run_baseline()
    if a.correct: run_correct(a.iters, a.popsize, a.workers)
    if a.pws: run_pws()
    if a.verify: run_verify()
    if a.report: run_report()
    if not any([a.baseline, a.correct, a.pws, a.verify, a.report]):
        ap.print_help()


if __name__ == "__main__":
    main()

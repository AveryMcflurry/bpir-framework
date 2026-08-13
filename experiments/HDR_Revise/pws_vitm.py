#!/usr/bin/env python3
r"""PWS on the VITM 68 baseline — post-hoc household reweighting via bounded IPF raking.

Mirrors pws_driver.run_raking (mechanism 1: arithmetic, learning-free, NO asim re-sim)
but scores with the VITM categories/targets from score_vitm. Groups the 68-baseline
households by travel signature, then Iterative Proportional Fitting toward the observed
marginals. Re-weights the pre-simulated 68-baseline trips only.

    python pws_vitm.py --trips <68-baseline final_trips.csv> --k 15 --bound 1.0
"""
import argparse, os, json, sys
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_vitm as V
S = V.S                                    # score_simulation: loss fns + LOSS_WEIGHTS
RUN_ROOT = r"G:\PhD programs\Ongoing works\Phase 2 work - Calibration_SPSA\vitm_run"

OBS_R = V.TRIP_RATE
OBS_P = np.array(list(V.PURPOSE_PROBS.values()))
OBS_M = np.array(list(V.MODE_PROBS_VITM.values()))
W = S.LOSS_WEIGHTS


def build_matrices(trips, hh_ids):
    """hh_ids = ALL simulated households (incl. 0-trip, whose tcount=0). Counting only
    trip-having households would drop the 0-trip ones and INFLATE the trip rate."""
    t = trips.copy()
    t["_p"] = S.map_column(t["purpose"], V.PURPOSE_MAPPING_VITM)
    t["_m"] = S.map_column(t["trip_mode"], V.MODE_MAPPING_VITM)
    pk, mk = list(V.PURPOSE_PROBS.keys()), list(V.MODE_PROBS_VITM.keys())
    tcount = t.groupby("household_id").size().reindex(hh_ids, fill_value=0).to_numpy(float)
    pmat = pd.crosstab(t["household_id"], t["_p"]).reindex(index=hh_ids, columns=pk, fill_value=0).to_numpy(float)
    mmat = pd.crosstab(t["household_id"], t["_m"]).reindex(index=hh_ids, columns=mk, fill_value=0).to_numpy(float)
    return tcount, pmat, mmat


def _composite(tt, tn, tp, tm):
    l = (W["trips"] * S.calculate_trip_rate_loss(tt / tn, OBS_R)
         + W["purpose"] * S.calculate_dist_loss(tp / tt, OBS_P)
         + W["mode"] * S.calculate_dist_loss(tm / tt, OBS_M))
    return 100.0 * np.exp(-l)


def run_raking(trips_path, hh_path, k=8, bound=0.5, iters=60, seed=42):
    trips = pd.read_csv(trips_path, low_memory=False)
    hh_ids = np.sort(pd.read_csv(hh_path)["household_id"].unique())
    tcount, pmat, mmat = build_matrices(trips, hh_ids)
    n_hh = len(hh_ids)
    full = _composite(tcount.sum(), n_hh, pmat.sum(0), mmat.sum(0))
    psh = np.divide(pmat, tcount[:, None], out=np.zeros_like(pmat), where=tcount[:, None] > 0)
    msh = np.divide(mmat, tcount[:, None], out=np.zeros_like(mmat), where=tcount[:, None] > 0)
    feats = np.column_stack([tcount / max(tcount.max(), 1), psh, msh])
    labels = KMeans(k, random_state=seed, n_init=4).fit_predict(feats)
    ct = np.array([tcount[labels == c].sum() for c in range(k)])
    cn = np.array([(labels == c).sum() for c in range(k)], float)
    cp = np.array([pmat[labels == c].sum(0) for c in range(k)])
    cm = np.array([mmat[labels == c].sum(0) for c in range(k)])

    def composite_of(logw):
        w = np.exp(np.clip(logw, -bound, bound)); tt = float((w * ct).sum()); tn = float((w * cn).sum())
        if tt <= 0 or tn <= 0:
            return -10.0
        return _composite(tt, tn, (w[:, None] * cp).sum(0), (w[:, None] * cm).sum(0))

    cps = cp / np.clip(ct[:, None], 1, None); cms = cm / np.clip(ct[:, None], 1, None)
    crate = ct / np.clip(cn, 1, None)
    logw = np.zeros(k); best, best_logw = full, logw.copy()
    history = [{"iter": 0, "best": float(full), "iter_comp": float(full)}]
    for it in range(1, iters + 1):
        w = np.exp(np.clip(logw, -bound, bound)); T = float((w * ct).sum())
        ps = (w[:, None] * cp).sum(0) / T
        logw = np.clip(logw + 0.5 * (cps * np.log(np.clip(OBS_P, 1e-9, None) / np.clip(ps, 1e-9, None))).sum(1), -bound, bound)
        w = np.exp(np.clip(logw, -bound, bound)); T = float((w * ct).sum())
        ms = (w[:, None] * cm).sum(0) / T
        logw = np.clip(logw + 0.5 * (cms * np.log(np.clip(OBS_M, 1e-9, None) / np.clip(ms, 1e-9, None))).sum(1), -bound, bound)
        w = np.exp(np.clip(logw, -bound, bound)); r = float((w * ct).sum()) / float((w * cn).sum())
        logw = np.clip(logw + 0.3 * np.sign(OBS_R - r) * np.sign(crate - r)
                       * np.abs(np.log(np.clip(crate, 1e-9, None) / max(r, 1e-9))), -bound, bound)
        comp = composite_of(logw)
        if comp > best:
            best, best_logw = comp, logw.copy()
        history.append({"iter": it, "best": float(best), "iter_comp": float(comp)})
    json.dump(history, open(os.path.join(RUN_ROOT, "history_pws_vitm.json"), "w"), indent=2)
    w = np.exp(np.clip(best_logw, -bound, bound)); tt = float((w * ct).sum()); tn = float((w * cn).sum())
    print(f"[pws-vitm] {n_hh} hh, k={k}, bound={bound} (w in [{np.exp(-bound):.2f},{np.exp(bound):.2f}]); "
          f"arithmetic IPF, NO re-sim")
    print(f"   68 baseline {full:.2f} -> PWS-raked {best:.2f}  (+{best-full:.2f})  "
          f"weighted trip_rate {tt/tn:.2f} (obs {OBS_R:.2f})")
    return best, full


def main():
    ap = argparse.ArgumentParser(description="PWS (IPF raking) on the VITM 68 baseline")
    ap.add_argument("--trips", required=True, help="68-baseline final_trips.csv")
    ap.add_argument("--households", required=True, help="68-baseline households.csv (ALL hh, incl 0-trip)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bound", type=float, default=0.5, help="0.5 = Melbourne-consistent conservative floor")
    ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()
    run_raking(a.trips, a.households, a.k, a.bound, a.iters)


if __name__ == "__main__":
    main()

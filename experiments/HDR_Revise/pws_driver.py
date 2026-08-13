#!/usr/bin/env python3
"""
PWS driver — corrected Progressive Weighted Subsampling on a reconstructed baseline.

Mechanism 3 of the paper: an arithmetic, learning-free correction that RE-WEIGHTS the
(already-simulated) input population — it picks the household subset whose aggregate
travel best matches observed, without re-running ActivitySim. It can only correct by
choosing which households' existing trips to count (e.g. drop trip-over-predicting
households to pull the rate from 9.2 toward the observed 8.0), not by changing behaviour.

Fixes vs the shipped pws_mechanism.py (which was inert on a degraded baseline):
  1. HDR-space objective — map trip_mode/purpose to score_simulation's unified categories
     and use its observed targets + LOSS_WEIGHTS, so the composite is comparable to HDR.
  2. ADAPTIVE learning threshold — train weights from each iteration's top-percentile teams
     (not an absolute 90 that a degraded baseline never reaches), so the search actually
     learns which households help.
  3. Fixed trip-rate denominator — rate = trips / team_size (the selection count), and teams
     are drawn WITHOUT replacement (a true subset), so no double-counting.
  4. Precomputed per-household aggregate matrices -> team score is a vector row-sum (no
     pd.concat); ~100x faster and scales to large, representative subsets.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_driver as D            # noqa: E402  (brings in score_simulation as D.SCORE)
from pws_mechanism import LogCoshObjective  # noqa: E402

RUN_ROOT = D.RUN_ROOT
DEFAULT_TRIPS = os.path.join(RUN_ROOT, "output_best_restore_faithful57", "final_trips.csv")


def build_matrices(trips):
    """Per-household aggregate vectors in the HDR objective space (mapped categories)."""
    S = D.SCORE
    trips = trips.copy()
    trips["_p"] = S.map_column(trips["purpose"], S.PURPOSE_MAPPING)
    trips["_m"] = S.map_column(trips["trip_mode"], S.UNIFIED_MODE_MAPPING)
    purp_keys = list(S.OBSERVED_METRICS["purpose_probabilities"].keys())
    mode_keys = list(S.OBSERVED_METRICS["mode_probabilities"].keys())
    hh_ids = np.sort(trips["household_id"].unique())
    tcount = trips.groupby("household_id").size().reindex(hh_ids, fill_value=0).to_numpy(float)
    pmat = (pd.crosstab(trips["household_id"], trips["_p"])
            .reindex(index=hh_ids, columns=purp_keys, fill_value=0).to_numpy(float))
    mmat = (pd.crosstab(trips["household_id"], trips["_m"])
            .reindex(index=hh_ids, columns=mode_keys, fill_value=0).to_numpy(float))
    return hh_ids, tcount, pmat, mmat, purp_keys, mode_keys


def make_objective():
    S = D.SCORE
    o = S.OBSERVED_METRICS
    return LogCoshObjective(o["trip_rate"], o["purpose_probabilities"], o["mode_probabilities"],
                            weights=dict(S.LOSS_WEIGHTS and {"trip": S.LOSS_WEIGHTS["trips"],
                                         "purpose": S.LOSS_WEIGHTS["purpose"],
                                         "mode": S.LOSS_WEIGHTS["mode"]}))


def score_subset(sel, tcount, pmat, mmat, purp_keys, mode_keys, obj):
    """Composite for a subset (array of household indices). rate = trips / |subset|."""
    n = len(sel)
    nt = float(tcount[sel].sum())
    pc = pmat[sel].sum(axis=0)
    mc = mmat[sel].sum(axis=0)
    comp, parts = obj.score(int(round(nt)), n,
                            dict(zip(purp_keys, pc)), dict(zip(mode_keys, mc)))
    return comp, parts


def _phase(it, max_iter):
    f = it / max_iter
    if f <= 0.25:
        return "deterministic", 0.0
    if f <= 0.7:
        return "progressive", min(0.85, (f - 0.25) / 0.45 * 0.85)
    return "weighted", 1.0


def run(trips_path, team_frac=0.5, n_teams=120, max_iter=40, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    trips = pd.read_csv(trips_path, low_memory=False)
    hh_ids, tcount, pmat, mmat, pk, mk = build_matrices(trips)
    obj = make_objective()
    n_hh = len(hh_ids)
    team_size = max(50, int(team_frac * n_hh))
    full_comp, _ = score_subset(np.arange(n_hh), tcount, pmat, mmat, pk, mk, obj)
    if verbose:
        print(f"[pws] {n_hh} households, team_size={team_size} ({team_frac:.0%}); "
              f"full-population composite = {full_comp:.2f}")

    good_freq = np.zeros(n_hh)
    best, best_sel, best_parts = full_comp, np.arange(n_hh), None
    hist = []

    for it in range(1, max_iter + 1):
        phase, relax = _phase(it, max_iter)
        # sampling weights from learned good-frequency (uniform until learning kicks in)
        w = 0.1 + good_freq
        probs = w / w.sum()
        order = np.argsort(-good_freq)                 # best-known households first
        teams = []
        for _ in range(n_teams):
            if phase == "deterministic":
                core = order[:team_size]               # exploit current best set
                jitter = rng.choice(n_hh, size=max(1, team_size // 10), replace=False)
                sel = np.unique(np.concatenate([core[:team_size - len(jitter)], jitter]))
            elif phase == "progressive":
                n_det = int(team_size * (1 - relax))
                det = order[:n_det]
                pool = np.setdiff1d(np.arange(n_hh), det, assume_unique=False)
                rnd = rng.choice(pool, size=team_size - len(det), replace=False,
                                 p=(probs[pool] / probs[pool].sum()))
                sel = np.concatenate([det, rnd])
            else:
                sel = rng.choice(n_hh, size=team_size, replace=False, p=probs)
            teams.append(sel)
        scores = np.array([score_subset(s, tcount, pmat, mmat, pk, mk, obj)[0] for s in teams])
        # ADAPTIVE learning: top-decile teams of THIS iter reinforce their households
        thr = np.percentile(scores, 90)
        for s, sc in zip(teams, scores):
            if sc >= thr:
                good_freq[s] += 1
        bi = int(np.argmax(scores))
        if scores[bi] > best:
            best = float(scores[bi]); best_sel = teams[bi]
            _, best_parts = score_subset(best_sel, tcount, pmat, mmat, pk, mk, obj)
        hist.append({"it": it, "phase": phase, "best": best, "iter_best": float(scores[bi]),
                     "iter_mean": float(scores.mean())})
        if verbose:
            print(f"[pws] it {it:3d}/{max_iter} {phase:13s} relax={relax:.2f}  "
                  f"best={best:6.2f}  iter_best={scores[bi]:6.2f}  iter_mean={scores.mean():6.2f}")

    if verbose and best_parts:
        def disp(k, wkey):
            return 100 * np.exp(D.SCORE.LOSS_WEIGHTS[wkey] * np.log(best_parts[k] / 100))
        print(f"\n[pws] DONE  full-pop baseline={full_comp:.2f} -> best subset composite={best:.2f}  "
              f"(+{best - full_comp:.2f})")
        print(f"      best-subset components: trip={best_parts['trip_score']:.1f} "
              f"purpose={best_parts['purpose_score']:.1f} mode={best_parts['mode_score']:.1f} "
              f"trip_rate={best_parts['trip_rate_actual']:.2f} (obs {best_parts['trip_rate_target']:.2f})")
    return best, full_comp, best_sel, hist


def _composite(tt, tn, tp, tm, obj):
    """Composite from weighted aggregates: trips tt, households tn, purpose vec tp, mode vec tm."""
    l = (D.SCORE.LOSS_WEIGHTS["trips"] * obj.trip_rate_loss(tt / tn)
         + D.SCORE.LOSS_WEIGHTS["purpose"] * obj.distribution_loss(tp / tt, obj._target_purpose)
         + D.SCORE.LOSS_WEIGHTS["mode"] * obj.distribution_loss(tm / tt, obj._target_mode))
    return 100.0 * np.exp(-l)


def run_raking(trips_path, k=8, bound=0.5, iters=60, seed=42, verbose=True, save_history=True):
    """ARITHMETIC, LEARNING-FREE reweighting of the household sample (PWS = mechanism 1 of
    the progression: PWS no-learning -> SPE learning+design -> HDR full input weighting).
    Groups households by travel signature, then Iterative Proportional Fitting (raking) —
    a deterministic, bounded, log-linear adjustment toward the observed marginals. NO
    optimizer, NO learned weights. Re-weights pre-simulated trips only (no re-sim). Records
    a per-iteration convergence curve (history_pws.json) for plotting vs the other curves."""
    import json as _json
    from sklearn.cluster import KMeans

    trips = pd.read_csv(trips_path, low_memory=False)
    hh_ids, tcount, pmat, mmat, pk, mk = build_matrices(trips)
    obj = make_objective()
    n_hh = len(hh_ids)
    full = _composite(tcount.sum(), n_hh, pmat.sum(0), mmat.sum(0), obj)

    # cluster by travel signature: trip count + per-trip purpose/mode shares (0-trip hh -> 0)
    psh = np.divide(pmat, tcount[:, None], out=np.zeros_like(pmat), where=tcount[:, None] > 0)
    msh = np.divide(mmat, tcount[:, None], out=np.zeros_like(mmat), where=tcount[:, None] > 0)
    feats = np.column_stack([tcount / max(tcount.max(), 1), psh, msh])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(feats)
    ct = np.array([tcount[labels == c].sum() for c in range(k)])              # cluster trips
    cn = np.array([(labels == c).sum() for c in range(k)], float)             # cluster hh count
    cp = np.array([pmat[labels == c].sum(0) for c in range(k)])              # (k, n_purp)
    cm = np.array([mmat[labels == c].sum(0) for c in range(k)])              # (k, n_mode)

    def composite_of(logw):
        w = np.exp(np.clip(logw, -bound, bound))
        tt = float((w * ct).sum()); tn = float((w * cn).sum())
        if tt <= 0 or tn <= 0:
            return -10.0
        tp = (w[:, None] * cp).sum(0); tm = (w[:, None] * cm).sum(0)
        return _composite(tt, tn, tp, tm, obj)

    # ITERATIVE PROPORTIONAL FITTING (log-linear raking): deterministic, learning-free.
    # Each iteration cycles purpose -> mode -> trip-rate corrections, nudging each cluster's
    # log-weight by its composition's alignment with the target/current marginal ratios.
    obs_p, obs_m, obs_r = obj._target_purpose, obj._target_mode, obj.target_trip_rate
    cps = cp / np.clip(ct[:, None], 1, None)        # (k, n_purp) each cluster's purpose mix
    cms = cm / np.clip(ct[:, None], 1, None)        # (k, n_mode)
    crate = ct / np.clip(cn, 1, None)               # (k,) each cluster's trip rate
    logw = np.zeros(k)
    best, best_logw = full, logw.copy()
    history = [{"iter": 0, "best": full, "iter_best": full, "iter_mean": full}]   # iter 0 = baseline (w=1)
    for it in range(1, iters + 1):
        w = np.exp(np.clip(logw, -bound, bound)); T = float((w * ct).sum())
        ps = (w[:, None] * cp).sum(0) / T            # rake purpose
        logw = np.clip(logw + 0.5 * (cps * np.log(np.clip(obs_p, 1e-9, None) / np.clip(ps, 1e-9, None))).sum(1),
                       -bound, bound)
        w = np.exp(np.clip(logw, -bound, bound)); T = float((w * ct).sum())
        ms = (w[:, None] * cm).sum(0) / T            # rake mode
        logw = np.clip(logw + 0.5 * (cms * np.log(np.clip(obs_m, 1e-9, None) / np.clip(ms, 1e-9, None))).sum(1),
                       -bound, bound)
        w = np.exp(np.clip(logw, -bound, bound)); r = float((w * ct).sum()) / float((w * cn).sum())
        logw = np.clip(logw + 0.3 * np.sign(obs_r - r) * np.sign(crate - r)   # nudge trip rate
                       * np.abs(np.log(np.clip(crate, 1e-9, None) / max(r, 1e-9))), -bound, bound)
        comp = composite_of(logw)
        if comp > best:
            best, best_logw = comp, logw.copy()
        history.append({"iter": it, "best": float(best), "iter_best": float(comp), "iter_mean": float(comp)})
    if save_history:
        with open(os.path.join(RUN_ROOT, "history_pws.json"), "w") as f:
            _json.dump(history, f, indent=2)

    w = np.exp(np.clip(best_logw, -bound, bound))
    tt = float((w * ct).sum()); tn = float((w * cn).sum())
    tp = (w[:, None] * cp).sum(0); tm = (w[:, None] * cm).sum(0)
    if verbose:
        def disp(vec, tgt, wkey):
            l = obj.distribution_loss(vec / tt, tgt) if wkey != "trips" else obj.trip_rate_loss(tt / tn)
            return 100 * np.exp(-l)
        print(f"[rake] k={k} bound={bound} (w in [{np.exp(-bound):.2f},{np.exp(bound):.2f}])  "
              f"full-pop {full:.2f} -> raked {best:.2f}  (+{best - full:.2f})  [{iters} IPF iters, learning-free]")
        print(f"       trip={disp(None,None,'trips'):.1f} purpose={disp(tp,obj._target_purpose,'p'):.1f} "
              f"mode={disp(tm,obj._target_mode,'m'):.1f}  weighted trip_rate={tt/tn:.2f} (obs {obj.target_trip_rate:.2f})")
        print(f"       learning curve saved -> {os.path.join(RUN_ROOT, 'history_pws.json')}")
    return best, full, w, labels, history


def main():
    ap = argparse.ArgumentParser(description="Corrected PWS on a reconstructed baseline")
    ap.add_argument("--mode", choices=["subset", "raking"], default="raking",
                    help="raking = bounded IPF (arithmetic, learning-free, ~65); subset = old team-search (modest)")
    ap.add_argument("--k", type=int, default=15, help="raking: household clusters (15/bound1.0 -> ~66)")
    ap.add_argument("--bound", type=float, default=1.0,
                    help="raking: |log-weight| bound (1.0 -> w in [0.37,2.72]); tighter = more conservative, lower")
    ap.add_argument("--trips", default=DEFAULT_TRIPS, help="simulated final_trips.csv (default: 57 baseline)")
    ap.add_argument("--team-frac", type=float, default=0.5, help="subset size as fraction of households")
    ap.add_argument("--n-teams", type=int, default=120)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode == "raking":
        run_raking(a.trips, k=a.k, bound=a.bound, iters=a.iters, seed=a.seed)
    else:
        run(a.trips, team_frac=a.team_frac, n_teams=a.n_teams, max_iter=a.iters, seed=a.seed)


if __name__ == "__main__":
    main()

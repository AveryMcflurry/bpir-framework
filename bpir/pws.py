"""PWS: post-hoc weighting scheme (mechanism 1; no re-simulation).

Simulate the baseline once, cluster households by their simulated travel
signature, then rake bounded per-cluster log-weights toward the observed
marginals with damped iterative proportional fitting (IPF). Arithmetic and
learning-free: the weights reweight the OUTPUT aggregation only and never
re-enter the model, so PWS quantifies the no-propagation floor.
"""
import numpy as np

EPS = 1e-9


def build_matrices(trips, purpose_keys, mode_keys, id_col="household_id",
                   purpose_col="purpose", mode_col="mode"):
    """Per-household aggregates from the simulated trip table: trip-count
    vector plus purpose and mode count matrices, rows aligned to sorted ids."""
    import pandas as pd
    hh_ids = np.sort(trips[id_col].unique())
    tcount = trips.groupby(id_col).size().reindex(hh_ids, fill_value=0).to_numpy(float)
    pmat = (pd.crosstab(trips[id_col], trips[purpose_col])
            .reindex(index=hh_ids, columns=purpose_keys, fill_value=0).to_numpy(float))
    mmat = (pd.crosstab(trips[id_col], trips[mode_col])
            .reindex(index=hh_ids, columns=mode_keys, fill_value=0).to_numpy(float))
    return hh_ids, tcount, pmat, mmat


def cluster_by_signature(tcount, pmat, mmat, k, seed=0):
    """K-means over each household's travel signature: normalised trip count
    plus per-trip purpose and mode shares (zero-trip households map to zero)."""
    from sklearn.cluster import KMeans
    psh = np.divide(pmat, tcount[:, None], out=np.zeros_like(pmat), where=tcount[:, None] > 0)
    msh = np.divide(mmat, tcount[:, None], out=np.zeros_like(mmat), where=tcount[:, None] > 0)
    feats = np.column_stack([tcount / max(tcount.max(), 1.0), psh, msh])
    return KMeans(n_clusters=k, random_state=seed).fit_predict(feats)


def aggregate_clusters(labels, tcount, pmat, mmat, k):
    """Collapse households to k clusters: trips, household count, and purpose
    and mode trip counts per cluster."""
    ct = np.array([tcount[labels == c].sum() for c in range(k)])
    cn = np.array([(labels == c).sum() for c in range(k)], float)
    cp = np.array([pmat[labels == c].sum(0) for c in range(k)])
    cm = np.array([mmat[labels == c].sum(0) for c in range(k)])
    return ct, cn, cp, cm


def weighted_composite(logw, ct, cn, cp, cm, score, log_bound):
    """Composite of the weighted aggregates under w = exp(clip(logw))."""
    w = np.exp(np.clip(logw, -log_bound, log_bound))
    tt, tn = float((w * ct).sum()), float((w * cn).sum())
    if tt <= 0 or tn <= 0:
        return -np.inf
    tp, tm = (w[:, None] * cp).sum(0), (w[:, None] * cm).sum(0)
    return score(tt / tn, tp / tt, tm / tt)


def run_raking(tcount, pmat, mmat, target_rate, target_purpose, target_mode,
               score, k, log_bound, n_iters, damp, rate_damp, seed=0):
    """Bounded log-linear raking of cluster weights toward observed marginals.

    Each iteration cycles purpose -> mode -> trip rate. The purpose and mode
    steps nudge every cluster's log-weight by its own trip composition times
    the log-ratio of target to achieved shares (damped IPF); the trip-rate
    step nudges clusters whose own rate lies on the target side of the current
    weighted rate. Weights are w = exp(clip(logw, +-log_bound)), so the
    adjustment stays bounded and logw = 0 reproduces the baseline. `score` is
    the shared composite, score(trip_rate, purpose_shares, mode_shares); the
    best composite seen so far is tracked and returned.
    """
    target_purpose = np.asarray(target_purpose, float)
    target_mode = np.asarray(target_mode, float)
    labels = cluster_by_signature(tcount, pmat, mmat, k, seed)
    ct, cn, cp, cm = aggregate_clusters(labels, tcount, pmat, mmat, k)
    cps = cp / np.clip(ct[:, None], 1, None)      # each cluster's purpose mix
    cms = cm / np.clip(ct[:, None], 1, None)      # each cluster's mode mix
    crate = ct / np.clip(cn, 1, None)             # each cluster's trip rate

    logw = np.zeros(k)
    baseline = weighted_composite(logw, ct, cn, cp, cm, score, log_bound)
    best, best_logw = baseline, logw.copy()
    history = [{"iter": 0, "best": baseline, "iter_score": baseline}]
    for it in range(1, n_iters + 1):
        # rake purpose shares
        w = np.exp(np.clip(logw, -log_bound, log_bound))
        ps = (w[:, None] * cp).sum(0) / max(float((w * ct).sum()), EPS)
        step = (cps * np.log(np.clip(target_purpose, EPS, None) / np.clip(ps, EPS, None))).sum(1)
        logw = np.clip(logw + damp * step, -log_bound, log_bound)
        # rake mode shares
        w = np.exp(np.clip(logw, -log_bound, log_bound))
        ms = (w[:, None] * cm).sum(0) / max(float((w * ct).sum()), EPS)
        step = (cms * np.log(np.clip(target_mode, EPS, None) / np.clip(ms, EPS, None))).sum(1)
        logw = np.clip(logw + damp * step, -log_bound, log_bound)
        # nudge the trip rate through clusters on the target's side of it
        w = np.exp(np.clip(logw, -log_bound, log_bound))
        r = float((w * ct).sum()) / max(float((w * cn).sum()), EPS)
        step = (np.sign(target_rate - r) * np.sign(crate - r)
                * np.abs(np.log(np.clip(crate, EPS, None) / max(r, EPS))))
        logw = np.clip(logw + rate_damp * step, -log_bound, log_bound)

        comp = weighted_composite(logw, ct, cn, cp, cm, score, log_bound)
        if comp > best:
            best, best_logw = comp, logw.copy()
        history.append({"iter": it, "best": float(best), "iter_score": float(comp)})

    weights = np.exp(np.clip(best_logw, -log_bound, log_bound))
    return weights, labels, best, baseline, history

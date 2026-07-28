"""SPE: segmentation-based population exploration (mechanism 2).

Households are segmented over demographics plus their baseline travel
signature, and the optimiser explores the segmentation itself: the number
of segments k and the projection seed are part of the search alongside the
per-segment scaling weights. Each candidate is realised as INTEGER
RESAMPLING of whole household units: down-weighted segments are subsampled
(a true subset), up-weighted segments keep every original and add replicas
(replicas receive fresh ids downstream). No record is ever edited; only how
many times each real household is represented changes. The resampled
population is re-simulated, so the weights propagate through tour, trip and
mode choice.
"""
import numpy as np

EPS = 1e-9
FAILURE_SCORE = -10.0  # sentinel fed to the optimiser when a run fails


def decode_parameters(raw, k_min, k_max, log_bound):
    """Decode a raw optimiser vector [k_norm, seed_norm, w_1 .. w_kmax] into
    (k, seed, weights). k maps [-1, 1] onto [k_min, k_max]; the seed picks the
    segmentation; weights are exp(clip(w, +-log_bound)) normalised to relative
    segment shares, so raw == 0 gives a uniform, composition-preserving
    resample."""
    k_frac = (float(raw[0]) + 1.0) / 2.0
    k = int(np.clip(k_min + k_frac * (k_max - k_min), k_min, k_max))
    seed = int(abs(float(raw[1])) * 999_999) % 1_000_000
    w = np.zeros(k)
    given = np.asarray(raw[2:], float)
    w[:min(k, len(given))] = given[:k]
    w = np.exp(np.clip(w, -log_bound, log_bound))
    return k, seed, w / w.sum()


def segment_features(demographics, trip_count, work_share, discretionary_share):
    """Standardised per-household features: demographics plus the baseline
    travel signature (trip count, work and discretionary trip shares). The
    signature is what gives a segmentation leverage on the scored marginals."""
    X = np.column_stack([np.asarray(demographics, float),
                         np.asarray(trip_count, float),
                         np.asarray(work_share, float),
                         np.asarray(discretionary_share, float)])
    return (X - X.mean(0)) / (X.std(0) + EPS)


def feature_segment(features, k, seed):
    """k quantile bins along a seed-selected random projection of the feature
    space. Different seeds give different meaningful segmentations; the seed
    and k are exactly what the optimiser explores."""
    if k <= 1 or len(features) == 0:
        return np.zeros(len(features), dtype=int)
    rng = np.random.default_rng(int(seed) + 1)
    proj = rng.standard_normal(features.shape[1])
    proj /= (np.linalg.norm(proj) + EPS)
    s = features @ proj
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return np.clip(np.digitize(s, edges), 0, k - 1)


def scale_population(groups, weights, rng):
    """Integer resampling by segment: target = max(1, round(w * n)) households.
    Down: sample without replacement, a true subset. Up: keep every original
    and draw the extras with replacement; the real implementation copies each
    replicated household and its persons under fresh unique ids so the
    simulator sees no duplicates. Returns row indices of the resampled
    population; records themselves are untouched."""
    keep = []
    for g, w_g in enumerate(weights):
        members = np.flatnonzero(groups == g)
        if len(members) == 0:
            continue
        target = max(1, int(round(w_g * len(members))))
        if target <= len(members):
            keep.append(rng.choice(members, size=target, replace=False))
        else:
            keep.append(members)
            keep.append(rng.choice(members, size=target - len(members), replace=True))
    return np.concatenate(keep) if keep else np.zeros(0, dtype=int)


def optimise(simulate, score, features, k_min, k_max, log_bound, optimizer,
             n_iters, rng):
    """Ask/tell loop over segmentation-plus-scaling candidates. Per candidate:
    decode -> segment -> integer resample -> re-simulate -> composite score.
    simulate(indices) runs the travel model on the resampled population and
    returns its trip table (the real implementation writes the resampled
    households/persons into an isolated run directory and launches ActivitySim
    here); score(trips) is the shared composite. Failed or degenerate runs
    score FAILURE_SCORE so the optimiser treats them as dead candidates."""
    history = []
    for it in range(n_iters):
        candidates = optimizer.ask()
        scores = []
        for raw in candidates:
            k, seed, weights = decode_parameters(raw, k_min, k_max, log_bound)
            groups = feature_segment(features, k, seed)
            idx = scale_population(groups, weights, rng)
            try:
                s = float(score(simulate(idx)))
            except Exception:
                s = FAILURE_SCORE
            scores.append(s if np.isfinite(s) and s >= 0 else FAILURE_SCORE)
        optimizer.tell(candidates, scores)
        _, s_best = optimizer.best()
        history.append({"iter": it, "best": float(s_best),
                        "iter_max": float(np.max(scores)),
                        "iter_mean": float(np.mean(scores))})
        if getattr(optimizer, "converged", False):
            break
    return optimizer.best(), history

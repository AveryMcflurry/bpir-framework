"""SPE: Segmentation-based Population Exploration."""
import numpy as np

EPS = 1e-9
FAILURE_SCORE = -10.0


def decode_parameters(raw, k_min, k_max, log_bound):
    """Decode a raw optimiser vector into (k, seed, weights)."""
    k_frac = (float(raw[0]) + 1.0) / 2.0
    k = int(np.clip(k_min + k_frac * (k_max - k_min), k_min, k_max))
    seed = int(abs(float(raw[1])) * 999_999) % 1_000_000
    w = np.zeros(k)
    given = np.asarray(raw[2:], float)
    w[:min(k, len(given))] = given[:k]
    w = np.exp(np.clip(w, -log_bound, log_bound))
    return k, seed, w / w.sum()


def segment_features(demographics, trip_count, work_share, discretionary_share):
    """Standardised per-household features: demographics, trip count and trip shares of the starting run."""
    X = np.column_stack([np.asarray(demographics, float),
                         np.asarray(trip_count, float),
                         np.asarray(work_share, float),
                         np.asarray(discretionary_share, float)])
    return (X - X.mean(0)) / (X.std(0) + EPS)


def feature_segment(features, k, seed):
    """k quantile bins along a seed-selected random projection of the feature space."""
    if k <= 1 or len(features) == 0:
        return np.zeros(len(features), dtype=int)
    rng = np.random.default_rng(int(seed) + 1)
    proj = rng.standard_normal(features.shape[1])
    proj /= (np.linalg.norm(proj) + EPS)
    s = features @ proj
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return np.clip(np.digitize(s, edges), 0, k - 1)


def scale_population(groups, weights, rng):
    """Resample households by segment weight."""
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
    """Ask/tell loop over segmentation and scaling candidates."""
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

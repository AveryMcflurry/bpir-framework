"""HDR: Hierarchical Disaggregate-input Reweighting."""
import re

import numpy as np


def _start(table, col, identity):
    """Starting value of a weight column: what the input carries, else its neutral value."""
    return np.asarray(table[col], float) if col in table.columns else identity


def cluster_households(features, k_min, k_max, seed=0):
    """K-means over standardised household demographics, k auto-selected by silhouette within [k_min, k_max]."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = np.asarray(features, float)
    sd = X.std(0)
    X = (X - X.mean(0)) / np.where(sd == 0, 1.0, sd)
    best_k, best_sc, best_labels = k_min, -1.0, np.zeros(len(X), dtype=int)
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=seed).fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        if sc > best_sc:
            best_k, best_sc, best_labels = k, sc, labels
    return best_labels, best_k


def split_theta(theta, k, n_hh, n_person, n_lu, n_skim):
    """Split the flat vector into its blocks."""
    t = np.asarray(theta, float)
    a, b = k * n_hh, k * (n_hh + n_person)
    c = b + n_lu
    return (t[:a].reshape(k, n_hh), t[a:b].reshape(k, n_person),
            t[b:c], t[c:c + n_skim])


def effective_weights(theta, k, sizes, bounds):
    """Per-block effective weights."""
    blocks = dict(zip(("household", "person", "land_use", "skims"),
                      split_theta(theta, k, *sizes)))
    out = {n: np.clip(b, -bounds[n], bounds[n]) for n, b in blocks.items()}
    for n in ("household", "land_use", "skims"):
        out[n] = np.exp(out[n])
    return out


def attach_household_weights(households, clusters, mult, attrs):
    """One weight column per weighted household attribute, by cluster."""
    out = households.copy()
    cl = np.clip(np.asarray(clusters, int), 0, len(mult) - 1)
    for j, attr in enumerate(attrs):
        out["w_" + attr] = _start(out, "w_" + attr, 1.0) * mult[cl, j]
    return out


def attach_person_weights(persons, clusters, logodds, person_spec):
    """One weight column per weighted person attribute, by cluster."""
    out = persons.copy()
    cl = np.clip(np.asarray(clusters, int), 0, len(logodds) - 1)
    for j, (attr, (col, indicator)) in enumerate(person_spec.items()):
        out[col] = _start(out, col, 0.0) + np.asarray(indicator(out), float) * logodds[cl, j]
    return out


def attach_landuse_weights(land_use, mult, attrs):
    """One weight column per weighted land-use attribute, the same factor for every zone."""
    out = land_use.copy()
    for j, attr in enumerate(attrs):
        out["w_" + attr] = _start(out, "w_" + attr, 1.0) * float(mult[j])
    return out


def scale_skim_copy(skims, mult, core_patterns):
    """Scale a copy of the skim matrices by group."""
    rules = list(zip(core_patterns.values(), mult))
    out = {}
    for name, mat in skims.items():
        fac = next((float(m) for pat, m in rules if re.search(pat, name)), 1.0)
        out[name] = np.asarray(mat) * fac
    return out


def optimise(simulate, score, build_inputs, optimizer, n_iters):
    """Ask/tell loop over candidate weight vectors."""
    history = []
    for it in range(n_iters):
        candidates = optimizer.ask()
        scores = [float(score(simulate(build_inputs(theta)))) for theta in candidates]
        optimizer.tell(candidates, scores)
        _, s_best = optimizer.best()
        history.append({"iter": it, "best": float(s_best),
                        "iter_max": float(np.max(scores)),
                        "iter_mean": float(np.mean(scores))})
        if getattr(optimizer, "converged", False):
            break
    return optimizer.best(), history

"""HDR: hierarchical disaggregate-input reweighting (mechanism 3, flagship).

Learns bounded weights on input ATTRIBUTES, never on records. Theta layout:
[k clusters x household attrs | k clusters x person attrs | global land-use
terms | global skim (level-of-service) groups]. Household, land-use and skim
blocks become multipliers exp(clip(theta) + base), bounded in log space with
theta == 0 reproducing the baseline exactly; the person block stays an
additive log-odds shift on activity-pattern utilities because it enters them
linearly. Household weights ride in as appended w_<attr> columns (the
simulator applies effective = raw * w_<attr> at its annotate step); skim
weights scale a COPY of the matched matrices. Raw inputs stay byte-identical:
weighted copies live in throwaway run directories only.
"""
import re

import numpy as np


def cluster_households(features, k_min, k_max, seed=0):
    """K-means over standardised household demographics, k auto-selected by
    silhouette within [k_min, k_max]. Runs once; all candidates reuse it."""
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
    """Split the flat vector into its blocks.
    Layout: [k*n_hh | k*n_person | n_lu | n_skim]."""
    t = np.asarray(theta, float)
    a, b = k * n_hh, k * (n_hh + n_person)
    c = b + n_lu
    return (t[:a].reshape(k, n_hh), t[a:b].reshape(k, n_person),
            t[b:c], t[c:c + n_skim])


def effective_weights(theta, k, sizes, bounds, base_logw=None):
    """Per-block effective weights. `sizes` is (n_hh, n_person, n_lu, n_skim);
    `bounds` maps block name to its log half-width; `base_logw` (optional)
    composes a fixed baseline log-weight under each block, so the correction
    is layered on a locked baseline rather than re-fitted from scratch."""
    blocks = dict(zip(("household", "person", "land_use", "skims"),
                      split_theta(theta, k, *sizes)))
    base = base_logw or {}
    out = {n: np.clip(b, -bounds[n], bounds[n]) + base.get(n, 0.0)
           for n, b in blocks.items()}
    for n in ("household", "land_use", "skims"):   # person stays log-odds
        out[n] = np.exp(out[n])
    return out


def attach_household_weights(households, clusters, mult, attrs):
    """Append one w_<attr> column per weighted attribute, holding the row's
    cluster multiplier. Raw demographic values are untouched; the copy is a
    run-directory view, never written back over the source."""
    out = households.copy()
    cl = np.clip(np.asarray(clusters, int), 0, len(mult) - 1)
    for j, attr in enumerate(attrs):
        out["w_" + attr] = mult[cl, j]
    return out


def attach_person_weights(persons, clusters, logodds, person_spec):
    """person_spec: attr -> (column, indicator). Each weighted person attribute
    is a per-cluster log-odds shift written to its column for the persons the
    indicator selects (workers for the mandatory-pattern weight, everyone for
    the non-mandatory one); zero is benign elsewhere. `clusters` is each
    person's household cluster. Raw person fields are never edited."""
    out = persons.copy()
    cl = np.clip(np.asarray(clusters, int), 0, len(logodds) - 1)
    for j, (attr, (col, indicator)) in enumerate(person_spec.items()):
        out[col] = np.asarray(indicator(out), float) * logodds[cl, j]
    return out


def attach_landuse_weights(land_use, mult, attrs):
    """Global land-use weights: one w_<attr> column per term, identical for
    every zone (the correction is region-wide, not zone-targeted)."""
    out = land_use.copy()
    for j, attr in enumerate(attrs):
        out["w_" + attr] = float(mult[j])
    return out


def scale_skim_copy(skims, mult, core_patterns):
    """Scale a COPY of the level-of-service matrices: every core whose name
    matches a group's regex is multiplied by that group's factor (first match
    wins). The raw skim file is never overwritten; the real implementation
    copies it into the run directory and scales the matched cores in place."""
    rules = list(zip(core_patterns.values(), mult))
    out = {}
    for name, mat in skims.items():
        fac = next((float(m) for pat, m in rules if re.search(pat, name)), 1.0)
        out[name] = np.asarray(mat) * fac
    return out


def optimise(simulate, score, build_inputs, optimizer, n_iters):
    """Ask/tell loop. build_inputs(theta) materialises the weighted copies in a
    throwaway run directory using the attach_* / scale_skim_copy helpers;
    simulate(inputs) runs the frozen travel model on them and returns the trip
    table (the real implementation launches ActivitySim here); score(trips) is
    the shared composite. Best-so-far weights are tracked across iterations."""
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

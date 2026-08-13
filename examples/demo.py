"""Self-contained BPIR demo on a synthetic simulator (no data needed).

A toy 'simulator' maps five attribute weights to a trip rate and mode and
purpose shares. The demo learns bounded weights so the fixed simulator
reproduces synthetic observed targets, mirroring the HDR loop at miniature
scale. Runs in seconds on any desktop computer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bpir.objective import composite_score

rng = np.random.default_rng(0)
TRUE = np.array([0.6, -0.5, 0.55, -0.65, 0.45])          # hidden misfit
TARGETS = {"trip": 8.0, "purpose": np.array([0.5, 0.3, 0.2]),
           "mode": np.array([0.45, 0.2, 0.25, 0.1])}
W = {"trip": 0.25, "purpose": 0.40, "mode": 0.35}
A = {"trip": 2.0, "dist": 4.0}


def simulate(theta):
    """Stand-in for the fixed behavioural model; real runs use ActivitySim."""
    z = np.clip(theta, -0.7, 0.7) - TRUE
    trip = 8.0 + 3.0 * z[0]
    purpose = np.array([0.5, 0.3, 0.2]) + np.array([0.3 * z[1], -0.2 * z[1] + 0.2 * z[2], -0.1 * z[1] - 0.2 * z[2]])
    mode = np.array([0.45, 0.2, 0.25, 0.1]) + 0.4 * np.array([z[3], -z[3] / 2, -z[3] / 2, 0]) + 0.25 * np.array([z[4], 0, -z[4], 0])
    return trip, np.abs(purpose) / np.abs(purpose).sum(), np.abs(mode) / np.abs(mode).sum()


def score(theta):
    trip, purpose, mode = simulate(theta)
    return composite_score(trip, purpose, mode, TARGETS, W, A)


def main():
    try:
        from bpir.optimizers import CMAES
        opt = CMAES(np.zeros(5), 0.3)
        use = "CMA-ES"
    except Exception:
        opt = None
        use = "random search (install `cma` for CMA-ES)"
    print(f"optimiser: {use}")
    best, best_theta = score(np.zeros(5)), np.zeros(5)
    print(f"baseline composite: {best:.2f}")
    for it in range(30):
        cands = opt.ask() if opt else [best_theta + 0.2 * rng.standard_normal(5) for _ in range(10)]
        vals = [score(np.asarray(c)) for c in cands]
        if opt:
            opt.tell(cands, vals)
        i = int(np.argmax(vals))
        if vals[i] > best:
            best, best_theta = vals[i], np.asarray(cands[i])
    print(f"corrected composite after 30 iterations: {best:.2f}")
    print("learned weights (factors):", np.round(np.exp(np.clip(best_theta, -0.7, 0.7)), 2))
    print("expected: composite rises from ~70 to >95; total run time well under a minute.")


if __name__ == "__main__":
    main()

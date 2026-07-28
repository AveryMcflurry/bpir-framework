"""SPSA coefficient-calibration benchmark (holds all inputs fixed).

Six-lever vector, identical structure on every dataset: [auto-driver,
auto-passenger, active, transit, mandatory-frequency, non-mandatory-
frequency]. Each lever adds a shift to the unconstrained coefficients the
model exposes for that behaviour group. Standard simultaneous-perturbation
form: two full simulations per iteration, decaying gain schedule, Bernoulli
perturbation, norm-clipped ascent inside a bounded box.
"""
import numpy as np


def spsa(simulate_with_coeffs, score, theta0, a, c, A, n_iters, theta_clip,
         grad_norm_clip, rng):
    """Gain schedule a_k = a / (k + A)**0.602 and c_k = c / k**0.101, the
    standard SPSA decay exponents. Each iteration draws a Bernoulli +-1 delta,
    evaluates theta +- c_k * delta with two full simulations, forms the
    simultaneous-perturbation gradient estimate, norm-clips it and ascends,
    clipping theta to +-theta_clip. simulate_with_coeffs(theta) applies the
    lever shifts to a perturbed COPY of the model coefficient files and runs
    the travel model (the real implementation launches ActivitySim here);
    score(trips) is the shared composite. Returns (best_theta, best_score,
    history) with the best of the two evaluations tracked per iteration."""
    theta = np.array(theta0, dtype=float)
    best, best_theta, history = -np.inf, theta.copy(), []
    for k in range(1, n_iters + 1):
        a_k = a / (k + A) ** 0.602
        c_k = c / k ** 0.101
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        s_plus = float(score(simulate_with_coeffs(theta + c_k * delta)))
        s_minus = float(score(simulate_with_coeffs(theta - c_k * delta)))
        ghat = (s_plus - s_minus) / (2.0 * c_k) * delta  # for +-1, 1/delta == delta
        norm = np.linalg.norm(ghat)
        if norm > grad_norm_clip:
            ghat *= grad_norm_clip / norm
        theta = np.clip(theta + a_k * ghat, -theta_clip, theta_clip)  # ascent
        current = max(s_plus, s_minus)
        if current > best:
            best, best_theta = current, theta.copy()
        history.append({"iter": k, "plus": s_plus, "minus": s_minus,
                        "best": float(best)})
    return best_theta, best, history

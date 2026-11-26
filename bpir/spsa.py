"""Calibration benchmark: SPSA on model coefficients."""
import numpy as np


def spsa(simulate_with_coeffs, score, theta0, a, c, A, n_iters, theta_clip,
         grad_norm_clip, rng):
    """SPSA search on the coefficient levers with the standard gain schedule."""
    theta = np.array(theta0, dtype=float)
    best, best_theta, history = -np.inf, theta.copy(), []
    for k in range(1, n_iters + 1):
        a_k = a / (k + A) ** 0.602
        c_k = c / k ** 0.101
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        s_plus = float(score(simulate_with_coeffs(theta + c_k * delta)))
        s_minus = float(score(simulate_with_coeffs(theta - c_k * delta)))
        ghat = (s_plus - s_minus) / (2.0 * c_k) * delta
        norm = np.linalg.norm(ghat)
        if norm > grad_norm_clip:
            ghat *= grad_norm_clip / norm
        theta = np.clip(theta + a_k * ghat, -theta_clip, theta_clip)
        current = max(s_plus, s_minus)
        if current > best:
            best, best_theta = current, theta.copy()
        history.append({"iter": k, "plus": s_plus, "minus": s_minus,
                        "best": float(best)})
    return best_theta, best, history

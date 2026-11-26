"""Composite objective: one score for trip rate, purpose shares and mode shares."""
import numpy as np


def logcosh(x):
    return np.log(np.cosh(x))


def composite_score(trip_rate, purpose_shares, mode_shares, targets, weights, alphas):
    """Composite score from the trip-rate, purpose and mode losses."""
    L = weights["trip"] * logcosh(
        alphas["trip"] * (trip_rate - targets["trip"]) / targets["trip"])
    L += weights["purpose"] * np.sum(logcosh(
        alphas["dist"] * (np.asarray(purpose_shares) - np.asarray(targets["purpose"]))))
    L += weights["mode"] * np.sum(logcosh(
        alphas["dist"] * (np.asarray(mode_shares) - np.asarray(targets["mode"]))))
    return 100.0 * np.exp(-L)

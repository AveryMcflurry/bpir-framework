"""Composite objective: one score for trip rate, purpose split and mode split.

score = 100 * exp(-L),  L = w_t*logcosh(a_t*rel_trip_err)
                          + w_p*sum logcosh(a_d*purpose_diff)
                          + w_m*sum logcosh(a_d*mode_diff)
The identical objective is applied to every mechanism and benchmark.
"""
import numpy as np


def logcosh(x):
    return np.log(np.cosh(x))


def composite_score(trip_rate, purpose_shares, mode_shares, targets, weights, alphas):
    """targets/weights/alphas are dicts with keys trip / purpose / mode / dist."""
    L = weights["trip"] * logcosh(
        alphas["trip"] * (trip_rate - targets["trip"]) / targets["trip"])
    L += weights["purpose"] * np.sum(logcosh(
        alphas["dist"] * (np.asarray(purpose_shares) - np.asarray(targets["purpose"]))))
    L += weights["mode"] * np.sum(logcosh(
        alphas["dist"] * (np.asarray(mode_shares) - np.asarray(targets["mode"]))))
    return 100.0 * np.exp(-L)

"""VITM-adapted composite scorer.

IDENTICAL objective to score_simulation (same log-cosh losses, LOSS_WEIGHTS
trips=0.25/purpose=0.40/mode=0.35, alphas, composite = 100*exp(-total_loss)) and
the SAME observed trip-rate (8.032) and purpose targets (Melbourne VISTA). Only
the MODE bucketing adapts to VITM's coarser modes so the comparison is valid:

  * VITM lumps transit (WALK_ALLTRN / PNR_ALLTRN / KNR_ALLTRN, + SCHOOL_BUS) with
    no rail/bus split  -> observed Rail+Bus are COMBINED into one 'Public Transit'.
  * VITM has no TNC mode -> observed Ride-Hailing is DROPPED and the mode targets
    are RENORMALISED over the categories VITM models (so VITM isn't penalised for
    a mode it structurally can't produce).
  * 'business' purpose (VITM's business_location step) -> 'work'.

This leaves the Phase-1 score_simulation untouched (the 57-baseline results stand).
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MainTrain"))
import score_simulation as S  # noqa: E402  reuse formula + observed trip/purpose targets

# VITM raw mode -> unified category (transit combined; school bus -> transit)
MODE_MAPPING_VITM = {
    "DRIVEALONEFREE": "Private Vehicle - Driver",
    "SHARED2FREE": "Private Vehicle - Passenger",
    "SHARED3FREE": "Private Vehicle - Passenger",
    "WALK": "Walking & Active Transport",
    "BIKE": "Cycling",
    "WALK_ALLTRN": "Public Transit",
    "PNR_ALLTRN": "Public Transit",
    "KNR_ALLTRN": "Public Transit",
    "SCHOOL_BUS": "Public Transit",
}
# 'business' (VITM-only) folds into work
PURPOSE_MAPPING_VITM = dict(S.PURPOSE_MAPPING, business="work")

# observed mode targets: combine Rail+Bus -> Public Transit, drop TNC, renormalise
_md = S.OBSERVED_DATA_FULL["mode_dist"]
_VITM_MODE_DIST = {
    "Private Vehicle - Driver": _md["Private Vehicle - Driver"],
    "Private Vehicle - Passenger": _md["Private Vehicle - Passenger"],
    "Walking & Active Transport": _md["Walking & Active Transport"],
    "Cycling": _md["Cycling"],
    "Public Transit": _md["Public Transit - Rail"] + _md["Public Transit - Bus"],
}
_tot = sum(_VITM_MODE_DIST.values())
MODE_PROBS_VITM = {k: v / _tot for k, v in _VITM_MODE_DIST.items()}
PURPOSE_PROBS = S.OBSERVED_METRICS["purpose_probabilities"]
TRIP_RATE = S.OBSERVED_METRICS["trip_rate"]


def evaluate_vitm(trips_df, n_households):
    """Composite score of a VITM ActivitySim run, comparable to the Melbourne 57
    baseline (same formula, mode categories adapted to VITM)."""
    rate = len(trips_df) / n_households
    trip_loss = S.calculate_trip_rate_loss(rate, TRIP_RATE)

    pc = list(PURPOSE_PROBS.keys())
    ap = S.get_probs(S.map_column(trips_df["purpose"], PURPOSE_MAPPING_VITM), pc)
    purpose_loss = S.calculate_dist_loss(ap, np.array(list(PURPOSE_PROBS.values())))

    mc = list(MODE_PROBS_VITM.keys())
    am = S.get_probs(S.map_column(trips_df["trip_mode"], MODE_MAPPING_VITM), mc)
    mode_loss = S.calculate_dist_loss(am, np.array(list(MODE_PROBS_VITM.values())))

    total = (S.LOSS_WEIGHTS["trips"] * trip_loss
             + S.LOSS_WEIGHTS["purpose"] * purpose_loss
             + S.LOSS_WEIGHTS["mode"] * mode_loss)
    return {
        "composite_score": float(100.0 * np.exp(-total)),
        "trip_rate_score": float(100.0 * np.exp(-trip_loss / S.LOSS_WEIGHTS["trips"])),
        "purpose_score": float(100.0 * np.exp(-purpose_loss / S.LOSS_WEIGHTS["purpose"])),
        "mode_score": float(100.0 * np.exp(-mode_loss / S.LOSS_WEIGHTS["mode"])),
        "trip_rate_actual": rate, "trip_rate_target": TRIP_RATE,
        "mode_probs_actual": dict(zip(mc, am)), "mode_probs_target": MODE_PROBS_VITM,
        "purpose_probs_actual": dict(zip(pc, ap)),
        "purpose_probs_target": PURPOSE_PROBS,
    }

"""Composite score for VITM runs."""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_simulation as S

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
PURPOSE_MAPPING_VITM = dict(S.PURPOSE_MAPPING, business="work")

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
    """Composite score of a VITM run."""
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

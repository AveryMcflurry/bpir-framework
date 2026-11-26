#!/usr/bin/env python3
"""Run one model with every weight at its neutral value and print its composite score: python run_baseline.py melb|vitm|sf"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def melb_scores(trips, n):
    """Return (five-mode composite, seven-mode composite) for a trip table of the transferred model."""
    import score_simulation as MS

    res7 = MS.evaluate_simulation(trips, n)
    mel5 = {}
    for raw, cat in MS.UNIFIED_MODE_MAPPING.items():
        if cat in ("Public Transit - Rail", "Public Transit - Bus"):
            mel5[raw] = "Public Transit"
        elif cat == "Ride-Hailing & Taxi":
            continue
        else:
            mel5[raw] = cat
    obs7 = MS.OBSERVED_METRICS["mode_probabilities"]
    mk = ["Private Vehicle - Driver", "Private Vehicle - Passenger",
          "Walking & Active Transport", "Cycling", "Public Transit"]
    mt = np.array([obs7["Private Vehicle - Driver"], obs7["Private Vehicle - Passenger"],
                   obs7["Walking & Active Transport"], obs7["Cycling"],
                   obs7["Public Transit - Rail"] + obs7["Public Transit - Bus"]])
    mt = mt / mt.sum()
    w = MS.LOSS_WEIGHTS
    tl = MS.calculate_trip_rate_loss(len(trips) / n, MS.OBSERVED_METRICS["trip_rate"])
    pk = list(MS.OBSERVED_METRICS["purpose_probabilities"])
    pl = MS.calculate_dist_loss(MS.get_probs(MS.map_column(trips["purpose"], MS.PURPOSE_MAPPING), pk),
                                np.array(list(MS.OBSERVED_METRICS["purpose_probabilities"].values())))
    mm = trips["trip_mode"].map(mel5).dropna()
    ml = MS.calculate_dist_loss(np.array([(mm == c).sum() / len(mm) for c in mk]), mt)
    c5 = 100 * np.exp(-(w["trips"] * tl + w["purpose"] * pl + w["mode"] * ml))
    return float(c5), float(res7["composite_score"])


def melb():
    import hdr_driver as HD

    os.makedirs(HD.RUN_ROOT, exist_ok=True)
    n = len(pd.read_csv(os.path.join(HD.BASELINE_DATA, "households.csv"), usecols=[0]))
    out = os.path.join(HD.RUN_ROOT, "output_baseline")
    print(f"[melb] ActivitySim on {n} households (about 5 minutes) ...", flush=True)
    if not HD.run_activitysim(HD.BASELINE_DATA, out, os.path.join(HD.RUN_ROOT, "asim_baseline.log"),
                              label="baseline"):
        sys.exit("[melb] ActivitySim failed: see hdr_run/asim_baseline.log")
    trips = pd.read_csv(os.path.join(out, "final_trips.csv"))
    c5, c7 = melb_scores(trips, n)
    print(f"[melb] households = {n}   trips = {len(trips)}   trip rate = {len(trips) / n:.3f}")
    print(f"[melb] 5-mode composite = {c5:.2f}")
    print(f"[melb] 7-mode composite = {c7:.2f}")


def vitm():
    import hdr_driver_vitm as HV

    if not os.path.exists(HV.VENV):
        sys.exit("[vitm] VITM Python not found at\n         " + HV.VENV +
                 "\n       Set the environment variable VITM_PYTHON to the python.exe of the\n"
                 "       environment that ships with the VITM2 package (ActivitySim 24.1.4).")
    os.makedirs(HV.RUN_ROOT, exist_ok=True)
    n = len(pd.read_csv(os.path.join(HV.DATA, "synthetic_households_formatted.csv"), usecols=[0]))
    out = os.path.join(HV.RUN_ROOT, "out_baseline")
    print(f"[vitm] ActivitySim 24.1.4 on {n} households ...", flush=True)
    p = HV.run_vitm_asim(HV.DATA, out, label="baseline")
    p.wait()
    p._logf.close()
    if p.returncode != 0:
        sys.exit("[vitm] ActivitySim failed: see vitm_run/out_baseline/run.log")
    comp, res = HV.score_dir(out, n)
    if res is None:
        sys.exit("[vitm] no final_trips.csv was produced: see vitm_run/out_baseline/run.log")
    print(f"[vitm] households = {n}   trip rate = {res['trip_rate_actual']:.3f}")
    print(f"[vitm] composite = {comp:.2f}   trip = {res['trip_rate_score']:.1f}  "
          f"purpose = {res['purpose_score']:.1f}  mode = {res['mode_score']:.1f}")


def sf():
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "mtc_full_demo.py"), "--baseline"]))


if __name__ == "__main__":
    runs = {"melb": melb, "vitm": vitm, "sf": sf}
    if len(sys.argv) != 2 or sys.argv[1] not in runs:
        sys.exit(__doc__)
    runs[sys.argv[1]]()

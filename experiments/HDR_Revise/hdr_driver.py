#!/usr/bin/env python3
r"""
HDR attribute-weighting driver
==============================

Orchestrates the full correction loop:

    cluster households once
    -> optimiser.ask()  (per-cluster attribute weight vectors)
       -> for each candidate:  write weighted households.csv  ->  run ActivitySim
          ->  score trips vs VISTA targets (score_simulation.evaluate_simulation)
    -> optimiser.tell(candidates, scores)
    -> repeat; track best.

The raw inputs are NEVER overwritten. Each candidate writes weighted COPIES into the
throwaway run dir: households.csv (w_income/w_vot per cluster), persons.csv
(cdap_work_w worker-rate weight per cluster), land_use.csv (global w_parking/w_terminal/
w_employment) and omx/skims.omx (global hovtime/autocost/autotime LOS weights).
ActivitySim applies them at runtime in the annotate / CDAP / mode-choice steps.

Usage (run with the asim env's python):
    python hdr_driver.py --smoke                  # 1 run at weights=1.0 (baseline check)
    python hdr_driver.py --optimizer cmaes --iters 60
    python hdr_driver.py --optimizer ppo   --iters 60
    python hdr_driver.py --optimizer lstm  --iters 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import openmatrix as omx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_mechanism as H  # noqa: E402
import cost_utils as COST  # noqa: E402  per-run wall-time / RAM / disk recorder

# ===========================================================================
# CONFIGURE THESE PATHS for your machine
# ===========================================================================
MAIN = os.path.abspath(os.path.join(HERE, "..", "MainTrain"))
BASELINE_DATA = os.path.join(MAIN, "data_spsa")         # Melbourne baseline (30,195 hh / 77,428 persons); override with --data
CONFIGS = os.path.join(MAIN, "configs_mel")             # the configs we edited; override with --configs
RUN_ROOT = os.path.join(HERE, "hdr_run")                # working area (created)


def _find(data_dir, name):
    """Case-insensitive file lookup (handles Households.csv vs households.csv)."""
    for f in os.listdir(data_dir):
        if f.lower() == name.lower():
            return os.path.join(data_dir, f)
    raise FileNotFoundError(os.path.join(data_dir, name))

# scorer (your existing module in MainTrain)
sys.path.insert(0, MAIN)
import score_simulation as SCORE  # noqa: E402

K_MIN, K_MAX = 4, 8          # household-cluster search range
POPSIZE = 16
DEFAULT_ITERS = 60
DEFAULT_WORKERS = 6          # concurrent ActivitySim runs (16 phys cores / ~41GB free)

# raw -> ActivitySim column names, used only to build clustering features
_HH_RENAME = {"HHID": "household_id", "PERSONS": "hhsize", "TAZ": "home_zone_id",
              "VEHICL": "auto_ownership", "workers": "num_workers"}


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------

def _hh_id_col(df: pd.DataFrame) -> str:
    for c in ("HHID", "household_id", "hhno", "HHHID"):
        if c in df.columns:
            return c
    raise KeyError("no household id column found in households.csv")


def _mirror_baseline(dst, extra_ignore=()):
    """Copy the whole baseline data tree (preserving subdirs like ``omx/``) into
    dst, skipping the named files (the driver writes households / persons)."""
    os.makedirs(dst, exist_ok=True)
    # households + persons + land_use are written per-candidate by the driver (w_* /
    # cdap_work_w cols); persons now carries the per-cluster worker-rate weight.
    skip = {"households.csv", "persons.csv", "land_use.csv"} | {x.lower() for x in extra_ignore}

    def _ignore(d, names):
        return [n for n in names if n.lower() in skip]

    shutil.copytree(BASELINE_DATA, dst, dirs_exist_ok=True, ignore=_ignore)


def setup_run_data(tag=""):
    """Mirror the baseline tree into RUN_ROOT/[tag_]data once; households.csv is
    generated per candidate so it is skipped here. tag keeps parallel runs isolated."""
    data_dir = os.path.join(RUN_ROOT, f"{tag}_data" if tag else "data")
    sentinel = os.path.join(data_dir, ".hdr_data_ready")
    if os.path.exists(sentinel):
        return data_dir
    _mirror_baseline(data_dir)
    open(sentinel, "w").close()
    return data_dir


def load_baseline():
    hh = pd.read_csv(_find(BASELINE_DATA, "households.csv"))
    pe = pd.read_csv(_find(BASELINE_DATA, "persons.csv"))
    return hh, pe


def load_landuse():
    return pd.read_csv(_find(BASELINE_DATA, "land_use.csv"))


def cluster_once(hh_raw, persons, k_force=0):
    """Cluster households on demographic features; returns (labels_by_idcol, k).
    k_force>0 forces exactly that many clusters (overrides silhouette auto-select)."""
    idc = _hh_id_col(hh_raw)
    hh = hh_raw.rename(columns=_HH_RENAME).copy()
    hh = hh.set_index("household_id")
    p = persons.copy()
    if "household_id" not in p.columns:
        raise KeyError("persons.csv must have a household_id column")
    kmin, kmax = (k_force, k_force) if k_force else (K_MIN, K_MAX)
    labels, k = H.cluster_households(p, hh, kmin, kmax, config=H.HDRConfig())
    # map back to the raw id values (index of hh == household_id == raw idc values)
    label_by_id = pd.Series(labels.values, index=hh.index)
    return idc, label_by_id, k


def _assert_run_dir(data_dir):
    """SAFETY: never write a weighted input anywhere but the throwaway hdr_run tree
    (the originals in BASELINE_DATA must stay byte-untouched)."""
    rr = os.path.abspath(RUN_ROOT)
    assert os.path.abspath(data_dir).startswith(rr), \
        f"refusing to write outside {rr}: {data_dir}"


def _split_theta(w_flat, k, cfg):
    """Split the flat optimizer vector into (per-cluster household, per-cluster person,
    global land-use, global skim) blocks. Layout: [k*n_hh | k*n_pers | n_lu | n_sk]."""
    n_hh, n_pe = len(cfg.attrs), len(cfg.person_attrs)
    n_lu, n_sk = len(cfg.lu_attrs), len(cfg.skim_attrs)
    w = np.asarray(w_flat, dtype=float)
    a, b, c = k * n_hh, k * n_hh + k * n_pe, k * n_hh + k * n_pe + n_lu
    return w[:a].reshape(k, n_hh), w[a:b].reshape(k, n_pe), w[b:c], w[c:c + n_sk]


def _cluster_of(id_series, label_by_id, k):
    cl = label_by_id.reindex(id_series).fillna(0).to_numpy().astype(int)
    return np.clip(cl, 0, k - 1)


# bounds the RESTORE baseline (best_restore_faithful57.npz) was found under — used to
# evaluate the fixed base when composing the forward correction on top of it.
RESTORE_BOUNDS = {"hh": (-0.8, 0.8), "work": (-1.6, 1.6), "lu": (-2.0, 2.0), "skim": (-1.6, 1.6)}


def _base_logw(npz_path, k, cfg):
    """Load a fixed BASE weight set (e.g. the reconstructed 57 baseline) and return its
    effective LOG-weights per block, under the bounds it was trained with. The forward
    correction is then composed on top (add log-weights). Skims are padded with 0 for any
    attrs the base didn't have (e.g. transit/nonmotor added for the forward). cand==0 ->
    the writer reproduces the base state exactly (the 57 baseline)."""
    d = np.load(npz_path, allow_pickle=True)
    bw = np.asarray(d["w"], dtype=float)
    bk = int(d["k"])
    if bk != k:
        raise ValueError(f"base weights k={bk} != current clustering k={k}")
    n_hh, n_pe, n_lu = len(cfg.attrs), len(cfg.person_attrs), len(cfg.lu_attrs)
    n_sk_b = len(d["skim_attrs"]) if "skim_attrs" in d.files else 3
    a, b, c = k * n_hh, k * n_hh + k * n_pe, k * n_hh + k * n_pe + n_lu
    sk = np.clip(bw[c:c + n_sk_b], *RESTORE_BOUNDS["skim"])
    if len(sk) < len(cfg.skim_attrs):                       # pad new forward-only skims with 0
        sk = np.concatenate([sk, np.zeros(len(cfg.skim_attrs) - len(sk))])
    return {"hh": np.clip(bw[:a].reshape(k, n_hh), *RESTORE_BOUNDS["hh"]),
            "pe": np.clip(bw[a:b].reshape(k, n_pe), *RESTORE_BOUNDS["work"]),
            "lu": np.clip(bw[b:c], *RESTORE_BOUNDS["lu"]),
            "sk": sk}


def _baseline_skims():
    p = os.path.join(BASELINE_DATA, "omx", "skims.omx")
    if os.path.exists(p):
        return p
    import glob
    return glob.glob(os.path.join(BASELINE_DATA, "omx", "*.omx"))[0]


def _write_weighted_skims(data_dir, w_sk, cfg, base_sk=None):
    """Write a weighted COPY of skims.omx into data_dir/omx/ (the raw skims are NEVER
    overwritten). Copies the file, then scales only the matched cores in place by
    exp(clip(theta)[+base]) — ~5x faster than a full rewrite. base_sk (optional) composes
    a fixed baseline log-weight on top (forward correction on the reconstructed baseline)."""
    src = _baseline_skims()
    dst = os.path.join(data_dir, "omx", os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    logw = np.clip(w_sk, *cfg.skim_log_clip)
    if base_sk is not None:
        logw = logw + base_sk
    mult = np.exp(logw)
    rules = [(H.SKIM_CORE_PATTERNS[a], float(mult[j])) for j, a in enumerate(cfg.skim_attrs)]
    f = omx.open_file(dst, "a")
    for name in f.list_matrices():
        for pat, fac in rules:
            if re.search(pat, name):
                if fac != 1.0:
                    f[name][:] = np.array(f[name]) * fac
                break
    f.close()


# person-CDAP weight: attr -> (column, per-person indicator). lw (per-cluster log-odds)
# is added to the named CDAP utility for persons where the indicator is 1. All three
# columns are always written (0 == benign) so persons.csv stays schema-valid.
_PERSON_CDAP_COLS = ("cdap_work_w", "cdap_student_w", "cdap_nonmand_w")
_PERSON_SPEC = {
    "work":    ("cdap_work_w",    lambda p: p["pemploy"].isin((1, 2)).to_numpy().astype(float)),
    "student": ("cdap_student_w", lambda p: p["ptype"].isin((3, 6, 7)).to_numpy().astype(float)),
    "nonmand": ("cdap_nonmand_w", lambda p: np.ones(len(p))),
}


def _persons_with_workw(persons, label_by_id, k, w_pe, cfg, base_pe=None):
    """persons.csv copy carrying the CDAP person-weight columns. Each optimized person
    attr adds a per-cluster log-odds shift (clipped) to its CDAP utility for the persons
    its indicator selects (workers' Mandatory, all persons' NonMandatory, ...). base_pe
    (optional) composes the fixed baseline log-odds on top. Raw ptype/pemploy are never
    edited; the weights ride in via these derived columns."""
    p = persons.copy()
    cl = _cluster_of(p["household_id"], label_by_id, k)
    for col in _PERSON_CDAP_COLS:                       # default benign
        p[col] = 0.0
    for j, attr in enumerate(cfg.person_attrs):
        col, indicator = _PERSON_SPEC[attr]
        lw = np.clip(w_pe[:, j], *cfg.work_log_clip)
        if base_pe is not None:
            lw = lw + base_pe[:, j]                     # compose baseline log-odds
        p[col] = indicator(p) * lw[cl]
    return p


def write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, w_flat, data_dir, cfg,
                          base_logw=None):
    """Write households.csv (per-cluster household weight cols), persons.csv (per-cluster
    CDAP weights) and land_use.csv + skims (global weight cols) into data_dir. Raw VALUES
    are preserved; only w_* / cdap_* columns are appended/derived. base_logw (optional)
    composes a FIXED baseline log-weight under each block (forward correction layered on
    the reconstructed 57 baseline); at w_flat==0 the writer reproduces that baseline."""
    _assert_run_dir(data_dir)
    w_hh, w_pe, w_lu, w_sk = _split_theta(w_flat, k, cfg)
    bl = base_logw or {}
    # households (income, vot, ...): exp(log-weight [+ base]) multiplier per cluster
    df = hh_raw.copy()
    mult = np.exp(np.clip(w_hh, *cfg.weight_log_clip) + bl.get("hh", 0.0))   # (k, n_hh)
    cl = _cluster_of(df[idc], label_by_id, k)
    for j, attr in enumerate(cfg.attrs):
        df[f"w_{attr}"] = mult[cl, j]
    df.to_csv(os.path.join(data_dir, "households.csv"), index=False)
    # persons (CDAP): log-odds shift [+ base] on Mandatory / NonMandatory propensity
    _persons_with_workw(persons, label_by_id, k, w_pe, cfg, base_pe=bl.get("pe")).to_csv(
        os.path.join(data_dir, "persons.csv"), index=False)
    # land use (global, per-zone-uniform)
    lu = lu_raw.copy()
    lmult = np.exp(np.clip(w_lu, *cfg.lu_weight_log_clip) + bl.get("lu", 0.0))
    for j, attr in enumerate(cfg.lu_attrs):
        lu[f"w_{attr}"] = float(lmult[j])                        # same for all zones
    lu.to_csv(os.path.join(data_dir, "land_use.csv"), index=False)
    # skims (global level-of-service): weighted COPY of skims.omx
    _write_weighted_skims(data_dir, w_sk, cfg, base_sk=bl.get("sk"))


def write_baseline_inputs(hh_raw, lu_raw, persons, data_dir, cfg):
    """Weights == 1.0 (households/land use) and cdap_work_w == 0 (persons): effective ==
    raw, should reproduce the baseline exactly."""
    _assert_run_dir(data_dir)
    df = hh_raw.copy()
    for attr in cfg.attrs:
        df[f"w_{attr}"] = 1.0
    df.to_csv(os.path.join(data_dir, "households.csv"), index=False)
    p = persons.copy()
    for col in _PERSON_CDAP_COLS:
        p[col] = 0.0
    p.to_csv(os.path.join(data_dir, "persons.csv"), index=False)
    lu = lu_raw.copy()
    for attr in cfg.lu_attrs:
        lu[f"w_{attr}"] = 1.0
    lu.to_csv(os.path.join(data_dir, "land_use.csv"), index=False)


def report_drift(hh_raw, idc, label_by_id, k, w_flat, cfg):
    """Fidelity diagnostic: print effective multipliers (household, worker-rate, land
    use) and how far the weighted income aggregate drifts from raw (bounded, not
    distorted)."""
    w_hh, w_pe, w_lu, w_sk = _split_theta(w_flat, k, cfg)
    hh_mult = np.exp(np.clip(w_hh, *cfg.weight_log_clip))            # (k, n_hh)
    lu_mult = np.exp(np.clip(w_lu, *cfg.lu_weight_log_clip))
    sk_mult = np.exp(np.clip(w_sk, *cfg.skim_log_clip))
    print("[hdr] weight report (effective multipliers):")
    for j, a in enumerate(cfg.attrs):
        print(f"        household {a:8s} per-cluster: "
              f"{np.round(hh_mult[:, j], 3).tolist()}")
    for j, a in enumerate(cfg.person_attrs):
        odds = np.exp(np.clip(w_pe[:, j], *cfg.work_log_clip))
        print(f"        person   {a:8s} per-cluster: "
              f"CDAP odds x{np.round(odds, 3).tolist()}")
    for j, a in enumerate(cfg.lu_attrs):
        print(f"        land-use  {a:11s}: x{lu_mult[j]:.3f}")
    for j, a in enumerate(cfg.skim_attrs):
        print(f"        skim      {a:11s}: x{sk_mult[j]:.3f}")
    if "income" in cfg.attrs and "income" in hh_raw.columns:
        cl = _cluster_of(hh_raw[idc], label_by_id, k)
        eff = hh_mult[cl, cfg.attrs.index("income")]
        inc = hh_raw["income"].to_numpy()
        print(f"        income aggregate drift (weighted/raw): "
              f"{(inc * eff).sum() / inc.sum():.3f}  (1.0 == no drift)")


# ---------------------------------------------------------------------------
# ActivitySim + scoring
# ---------------------------------------------------------------------------

def _activitysim_exe():
    cand = os.path.join(os.path.dirname(sys.executable), "Scripts", "activitysim.exe")
    return cand if os.path.exists(cand) else "activitysim"


def run_activitysim(data_dir, out_dir, log_path, label="", cost_log=None):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [_activitysim_exe(), "run", "-c", CONFIGS, "-d", data_dir, "-o", out_dir]
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    COST.RunCost(p, out_dir, label=(label or os.path.basename(out_dir)),
                 cost_log=(cost_log or os.path.join(RUN_ROOT, "costs.jsonl")),
                 extra={"engine": "melbourne", "data_mb": round(COST._dir_size_mb(data_dir), 1)})
    p.wait(); logf.close()
    return p.returncode == 0


def score_run(out_dir, n_households):
    trips_path = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(trips_path):
        alt = os.path.join(out_dir, "trips.csv")
        trips_path = alt if os.path.exists(alt) else trips_path
    if not os.path.exists(trips_path):
        return H.FAILURE_SCORE, None
    trips = pd.read_csv(trips_path)
    res = SCORE.evaluate_simulation(trips, n_households)
    return float(res["composite_score"]), res


# Paper Fig-3 baseline targets (observed + Fig-3d differences) for the reverse
# reconstruction of the original worse baseline.
RESTORE_TRIP_RATE = 9.23                          # paper Fig 3b
RESTORE_MODE = {                                  # paper Fig 3d (observed + diffs)
    "Private Vehicle - Driver": 0.295, "Private Vehicle - Passenger": 0.396,
    "Walking & Active Transport": 0.001, "Cycling": 0.021,
    "Public Transit - Rail": 0.001, "Public Transit - Bus": 0.027,
    "Ride-Hailing & Taxi": 0.261,
}
RESTORE_PURPOSE = {                               # paper Fig 3c radar (pattern) + score 80.7
    "home": 0.227, "work": 0.267, "shopping": 0.130, "escort": 0.059,
    "othdiscr": 0.060, "social": 0.090, "othmaint": 0.085, "eatout": 0.050,
    "school": 0.033,
}
RESTORE_COMPOSITE_TARGET = 56.0                  # paper baseline composite vs observed
RESTORE_COMP_TARGET = {"trip": 95.7, "purpose": 80.7, "mode": 25.1}   # paper Fig-3 component scores


def score_run_restore(out_dir, n_households):
    """Score how well a run matches the PAPER's baseline (Fig 3: trip rate 9.23,
    mode shares, purpose shares) using the SAME composite weights as the real
    objective. --restore MAXIMIZES this; at a high match the reconstructed
    baseline's observed-composite -> ~56."""
    trips_path = os.path.join(out_dir, "final_trips.csv")
    if not os.path.exists(trips_path):
        return H.FAILURE_SCORE, None
    trips = pd.read_csv(trips_path)
    W = SCORE.LOSS_WEIGHTS
    rate = len(trips) / n_households
    trip_loss = SCORE.calculate_trip_rate_loss(rate, RESTORE_TRIP_RATE)
    pcats = list(RESTORE_PURPOSE.keys())
    actual_p = SCORE.get_probs(SCORE.map_column(trips["purpose"], SCORE.PURPOSE_MAPPING), pcats)
    purpose_loss = SCORE.calculate_dist_loss(actual_p, np.array([RESTORE_PURPOSE[c] for c in pcats]))
    mcats = list(RESTORE_MODE.keys())
    actual_m = SCORE.get_probs(SCORE.map_column(trips["trip_mode"], SCORE.UNIFIED_MODE_MAPPING), mcats)
    mode_loss = SCORE.calculate_dist_loss(actual_m, np.array([RESTORE_MODE[c] for c in mcats]))
    total = W["trips"] * trip_loss + W["purpose"] * purpose_loss + W["mode"] * mode_loss
    match = 100.0 * np.exp(-total)
    obs = SCORE.evaluate_simulation(trips, n_households)
    obs_comp = float(obs["composite_score"])
    if not (np.isfinite(obs_comp) and np.isfinite(match)):
        return H.FAILURE_SCORE, None      # degenerate run -> treat as failure (keep nan out of CMA/argmax)
    # OBJECTIVE: match the paper's REPORTED Fig-3 COMPONENT scores (trip 95.7 / purpose 80.7
    # / mode 25.1 vs observed) -> a BALANCED reconstruction. Targeting the composite alone let
    # mode over-degrade (passenger 0.50) while purpose stayed put; targeting each component
    # forces purpose to carry its share and caps mode at 25.1 (penalizes over-degradation).
    # `match` (paper distribution split) kept at 0.2 so the mode/purpose SPLITS stay paper-like.
    comp = {"trip": 100.0 * (obs["trip_rate_score"] / 100.0) ** W["trips"],
            "purpose": 100.0 * (obs["purpose_score"] / 100.0) ** W["purpose"],
            "mode": 100.0 * (obs["mode_score"] / 100.0) ** W["mode"]}
    comp_dev = sum(abs(comp[k] - RESTORE_COMP_TARGET[k]) for k in comp)
    comp_match = 100.0 * np.exp(-comp_dev / 12.0)
    objective = 0.8 * comp_match + 0.2 * match
    return float(objective), {"restore_match": float(match), "trip_rate_actual": rate,
                              "obs_composite": obs_comp, "comp_trip": comp["trip"],
                              "comp_purpose": comp["purpose"], "comp_mode": comp["mode"]}


def launch_activitysim(data_dir, out_dir, log_path):
    """Start an ActivitySim run as a background subprocess (non-blocking)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [_activitysim_exe(), "run", "-c", CONFIGS, "-d", data_dir, "-o", out_dir]
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    p._logf = logf
    COST.RunCost(p, out_dir, label=os.path.basename(out_dir),
                 cost_log=os.path.join(RUN_ROOT, "costs.jsonl"),
                 extra={"engine": "melbourne", "data_mb": round(COST._dir_size_mb(data_dir), 1)})
    return p


def setup_workers(n, persons_df, tag=""):
    """Create n isolated worker dirs mirroring the baseline tree (skims/land_use)
    once, and write the (sub)sample persons.csv into each. tag keeps parallel runs
    (3 optimizers in 3 terminals) from colliding on the same worker dirs."""
    pre = f"{tag}_" if tag else ""
    workers = []
    for i in range(n):
        ddir = os.path.join(RUN_ROOT, f"{pre}w{i}", "data")
        odir = os.path.join(RUN_ROOT, f"{pre}w{i}", "output")
        sentinel = os.path.join(ddir, ".hdr_data_ready")
        if not os.path.exists(sentinel):
            _mirror_baseline(ddir, extra_ignore=["persons.csv"])
            open(sentinel, "w").close()
        # schema-valid base persons.csv (cdap_* cols now in keep_columns); the per-
        # candidate writer overwrites it with the cluster-specific CDAP weights.
        persons_df.assign(**{c: 0.0 for c in _PERSON_CDAP_COLS}).to_csv(
            os.path.join(ddir, "persons.csv"), index=False)
        workers.append({"data": ddir, "out": odir})
    return workers


def evaluate_batch(cands, hh_raw, lu_raw, persons, idc, label_by_id, k, cfg, workers, tagbase,
                   scorer=None, base_logw=None):
    """Evaluate candidates, running up to len(workers) ActivitySim runs at once.
    Returns scores aligned to cands (FAILURE_SCORE on a failed run). ``scorer``
    defaults to score_run (observed targets); pass score_run_restore for --restore.
    base_logw (optional) layers a fixed baseline (e.g. the 57 reconstruction) under each."""
    scorer = scorer or score_run
    n = len(workers)
    nhh = len(hh_raw)
    scores = [H.FAILURE_SCORE] * len(cands)
    metas = [None] * len(cands)
    for start in range(0, len(cands), n):
        chunk = list(range(start, min(start + n, len(cands))))
        running = []
        for slot, ci in enumerate(chunk):
            w = workers[slot]
            write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, cands[ci], w["data"], cfg,
                                  base_logw=base_logw)
            log_path = os.path.join(RUN_ROOT, f"asim_{tagbase}_c{ci}.log")
            running.append((ci, slot, launch_activitysim(w["data"], w["out"], log_path)))
        for ci, slot, p in running:
            p.wait()
            p._logf.close()
            if p.returncode == 0:
                scores[ci], metas[ci] = scorer(workers[slot]["out"], nhh)
    return scores, metas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global BASELINE_DATA, CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--popsize", type=int, default=POPSIZE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="max concurrent ActivitySim runs")
    ap.add_argument("--data", default=BASELINE_DATA,
                    help="baseline data dir (persons/households/land_use/skims)")
    ap.add_argument("--configs", default=CONFIGS, help="ActivitySim configs dir")
    ap.add_argument("--smoke", action="store_true",
                    help="run ONE ActivitySim at weights=1.0 and report the score")
    ap.add_argument("--subsample", type=int, default=0,
                    help="optimize on an N-household subsample (faster, fewer crashes); also limits --smoke")
    ap.add_argument("--apply-best", dest="apply_best", action="store_true",
                    help="apply saved best_<optimizer>.npz weights to the FULL population and score")
    ap.add_argument("--k", type=int, default=0,
                    help="force number of household clusters (0 = auto-select by silhouette)")
    ap.add_argument("--restore", action="store_true",
                    help="reverse-optimize to reproduce the PAPER baseline (Fig 3) instead of observed")
    ap.add_argument("--warmstart", action="store_true",
                    help="cmaes: seed the search from the saved best_<tag>.npz (must match dim)")
    ap.add_argument("--base-weights", dest="base_weights", default="",
                    help="npz of a FIXED baseline (e.g. best_restore_faithful57.npz) to layer the "
                         "forward correction onto; at weights=0 the model IS that baseline (57)")
    args = ap.parse_args()
    run_tag = "restore" if args.restore else args.optimizer
    scorer = score_run_restore if args.restore else score_run
    BASELINE_DATA = args.data
    CONFIGS = args.configs

    os.makedirs(RUN_ROOT, exist_ok=True)
    cfg = H.HDRConfig()
    data_dir = setup_run_data(run_tag)
    hh_raw, persons = load_baseline()
    lu_raw = load_landuse()
    n_hh = len(hh_raw)
    print(f"[hdr] baseline: {n_hh} households, {len(persons)} persons, {len(lu_raw)} zones")

    # ---- smoke: weights=1.0 should reproduce the baseline ----
    if args.smoke:
        if args.subsample and args.subsample < n_hh:
            idc = _hh_id_col(hh_raw)
            n = args.subsample
            hh_sub = hh_raw.iloc[:n].copy()
            ids = set(hh_sub[idc].tolist())
            pe_sub = persons[persons["household_id"].isin(ids)].copy()
            print(f"[hdr] SMOKE subsample: {len(hh_sub)} households, {len(pe_sub)} persons")
            sdata = os.path.join(RUN_ROOT, "data_smoke")
            _mirror_baseline(sdata)
            write_baseline_inputs(hh_sub, lu_raw, pe_sub, sdata, cfg)
            n_hh_run = len(hh_sub)
        else:
            print("[hdr] SMOKE: writing households/persons/land_use at weights=1.0 ...")
            write_baseline_inputs(hh_raw, lu_raw, persons, data_dir, cfg)
            sdata, n_hh_run = data_dir, n_hh
        out_dir = os.path.join(RUN_ROOT, "output_smoke")
        t0 = time.time()
        ok = run_activitysim(sdata, out_dir, os.path.join(RUN_ROOT, "asim_smoke.log"))
        if not ok:
            print("[hdr] SMOKE FAILED — see", os.path.join(RUN_ROOT, "asim_smoke.log"))
            sys.exit(1)
        score, res = score_run(out_dir, n_hh_run)
        print(f"[hdr] SMOKE OK in {time.time()-t0:.0f}s  composite={score:.2f}")
        if res:
            print(f"      trip_rate actual={res['trip_rate_actual']:.3f} "
                  f"target={res['trip_rate_target']:.3f}  "
                  f"(mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} "
                  f"trip={res['trip_rate_score']:.1f})")
        print("[hdr] If this matches your known baseline, the config edits are correct.")
        return

    # ---- apply saved best weights to the FULL population (final paper number) ----
    if args.apply_best:
        bp = os.path.join(RUN_ROOT, f"best_{run_tag}.npz")
        if not os.path.exists(bp):
            print("[hdr] no saved weights at", bp)
            sys.exit(1)
        d = np.load(bp, allow_pickle=True)
        w = d["w"]
        saved_k = int(d["k"])   # cluster with the SAME k the weights were trained on
        idc, label_by_id, k = cluster_once(hh_raw, persons, saved_k)
        base_logw = _base_logw(args.base_weights, k, cfg) if args.base_weights else None
        write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k, w, data_dir, cfg,
                              base_logw=base_logw)
        report_drift(hh_raw, idc, label_by_id, k, w, cfg)
        out_dir = os.path.join(RUN_ROOT, f"output_best_{run_tag}")
        print(f"[hdr] applying best {run_tag} weights to FULL population ({n_hh} hh) ...")
        ok = run_activitysim(data_dir, out_dir,
                             os.path.join(RUN_ROOT, f"asim_best_{run_tag}.log"))
        if not ok:
            print("[hdr] full-pop run FAILED — see log")
            sys.exit(1)
        score, res = scorer(out_dir, n_hh)
        if args.restore:
            print(f"[hdr] FULL-POP RESTORE: composite={res['obs_composite']:.2f}  "
                  f"components trip={res['comp_trip']:.1f}/purpose={res['comp_purpose']:.1f}/mode={res['comp_mode']:.1f} "
                  f"(paper 95.7/80.7/25.1)  objective={score:.2f}")
            print("[hdr] -> this weighted land_use/households is the reconstructed original baseline.")
        else:
            print(f"[hdr] FULL-POP best {args.optimizer}: composite={score:.2f}")
            if res:
                print(f"      trip_rate={res['trip_rate_actual']:.3f} (target {res['trip_rate_target']:.3f})  "
                      f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} "
                      f"trip={res['trip_rate_score']:.1f}")
        return

    # ---- cluster the FULL population first (subsample & full share cluster defs) ----
    idc, label_by_id, k = cluster_once(hh_raw, persons, args.k)
    dim = (k * (len(cfg.attrs) + len(cfg.person_attrs))
           + len(cfg.lu_attrs) + len(cfg.skim_attrs))
    print(f"[hdr] clustered into k={k} groups -> dim={dim} "
          f"(({len(cfg.attrs)} hh + {len(cfg.person_attrs)} person) x {k} + "
          f"{len(cfg.lu_attrs)} land-use + {len(cfg.skim_attrs)} skim global)")
    base_logw = _base_logw(args.base_weights, k, cfg) if args.base_weights else None
    if base_logw is not None:
        print(f"[hdr] forward correction layered on {os.path.basename(args.base_weights)} "
              f"(weights=0 == the reconstructed baseline; optimizer corrects from there)")

    # ---- optimization population (optionally subsampled for speed + robustness) ----
    if args.subsample and args.subsample < n_hh:
        srng = np.random.default_rng(0)
        pick = srng.choice(len(hh_raw), size=args.subsample, replace=False)
        hh_opt = hh_raw.iloc[pick].copy()
        ids = set(hh_opt[idc].tolist())
        pe_opt = persons[persons["household_id"].isin(ids)].copy()
        print(f"[hdr] optimizing on a {len(hh_opt)}-household subsample "
              f"({len(pe_opt)} persons)")
    else:
        hh_opt, pe_opt = hh_raw, persons

    # widest plausibility box for the optimizer; per-group clipping happens in apply
    olo, ohi = cfg.lu_weight_log_clip
    opt_bounds = (olo, ohi)
    opt_kwargs = (dict(bounds=opt_bounds, sigma0=(ohi - olo) / 4)
                  if args.optimizer == "cmaes" else {})
    if args.warmstart and args.optimizer == "cmaes":
        wp = os.path.join(RUN_ROOT, f"best_{run_tag}.npz")
        x0 = np.load(wp, allow_pickle=True)["w"] if os.path.exists(wp) else None
        if x0 is not None and len(x0) == dim:
            opt_kwargs["x0"] = x0
            opt_kwargs["sigma0"] = 0.6     # tighter: local refinement around the seed
            print(f"[hdr] warm-starting cmaes from {wp} (dim {dim}, sigma0=0.6)")
        else:
            print(f"[hdr] warmstart unavailable/dim-mismatch at {wp}; cold start")
    if args.optimizer == "lstm":
        npz = np.load(os.path.join(HERE, "lstm_meta_weights.npz"))
        phi = npz["phi"]
        lcfg = H.LSTMMetaConfig(bounds=opt_bounds)
        opt = H.make_optimiser("lstm", dim, args.popsize, lstm_phi=phi, lstm_config=lcfg)
    elif args.optimizer == "ppo":
        opt = H.make_optimiser("ppo", dim, args.popsize,
                               ppo_config=H.PPOConfig(bounds=opt_bounds))
    else:
        opt = H.make_optimiser("cmaes", dim, args.popsize, **opt_kwargs)

    n_workers = max(1, min(args.workers, args.popsize))
    workers = setup_workers(n_workers, pe_opt, run_tag)
    print(f"[hdr] running up to {n_workers} ActivitySim evaluations in parallel")
    history = []
    best_score, best_w = H.FAILURE_SCORE, None
    t_start = time.time()

    for it in range(args.iters):
        cands = opt.ask()
        scores, metas = evaluate_batch(cands, hh_opt, lu_raw, pe_opt, idc, label_by_id, k, cfg, workers,
                                       tagbase=f"{run_tag}_it{it}", scorer=scorer, base_logw=base_logw)
        opt.tell(cands, scores)
        w_best, s_best = opt.best_solution
        if s_best > best_score:
            best_score, best_w = s_best, np.asarray(w_best).copy()
            np.savez(os.path.join(RUN_ROOT, f"best_{run_tag}.npz"),
                     w=best_w, k=k, attrs=np.array(cfg.attrs),
                     person_attrs=np.array(cfg.person_attrs), lu_attrs=np.array(cfg.lu_attrs),
                     skim_attrs=np.array(cfg.skim_attrs),
                     label_by_id=label_by_id.values, id_index=label_by_id.index.values,
                     score=best_score)
        # reconstructed composite (vs observed) of this iter's best-match candidate — for
        # --restore this is the number that should approach the paper baseline (~56)
        bi = int(np.argmax(scores))
        recon = (metas[bi].get("obs_composite") if metas[bi] else float("nan"))
        rec = {"iter": it, "best": float(best_score),
               "iter_mean": float(np.mean(scores)), "iter_max": float(np.max(scores))}
        if args.restore:
            rec["recon_composite"] = float(recon)
        history.append(rec)
        with open(os.path.join(RUN_ROOT, f"history_{run_tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
        extra = f"  recon_comp={recon:5.1f}" if args.restore else ""
        print(f"[hdr] it {it:3d}/{args.iters}  best={best_score:6.2f}  "
              f"iter_max={np.max(scores):6.2f}  iter_mean={np.mean(scores):6.2f}{extra}  "
              f"elapsed={time.time()-t_start:.0f}s")
        if opt.converged:
            print("[hdr] optimiser converged.")
            break

    print(f"[hdr] DONE  best composite score = {best_score:.2f}")
    if best_w is not None:
        report_drift(hh_raw, idc, label_by_id, k, best_w, cfg)
    print(f"[hdr] best weights saved -> {os.path.join(RUN_ROOT, f'best_{run_tag}.npz')}")


if __name__ == "__main__":
    main()

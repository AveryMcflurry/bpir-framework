from __future__ import annotations

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import time

# ============================================================
# 1. USER-CONFIGURABLE PATHS
# ============================================================

BASE_DIR = Path(".")
BASE_CONFIG_DIR   = BASE_DIR / "configs_spsa"       
BASE_DATA_DIR     = BASE_DIR / "data_spsa"          
SPSA_OUTPUT_ROOT  = BASE_DIR / "spsa_runs"          

# --- COEFFICIENT FILES ---
# 1. Mandatory
MANDATORY_COEFF_BASE   = BASE_CONFIG_DIR / "mandatory_tour_frequency_coefficients_base.csv"
MANDATORY_COEFF_ACTIVE = BASE_CONFIG_DIR / "mandatory_tour_frequency_coefficients.csv"

# 2. Mode
MODE_COEFF_BASE   = BASE_CONFIG_DIR / "trip_mode_choice_coefficients_base.csv"
MODE_COEFF_ACTIVE = BASE_CONFIG_DIR / "trip_mode_choice_coefficients.csv"

# 3. Non-Mandatory (8 Person Types)
NON_MAND_PREFIX = "non_mandatory_tour_frequency_coefficients" 
NON_MAND_SUFFIXES = [
    "PTYPE_FULL", "PTYPE_PART", "PTYPE_UNIVERSITY", "PTYPE_SCHOOL", 
    "PTYPE_PRESCHOOL", "PTYPE_RETIRED", "PTYPE_DRIVING", "PTYPE_NONWORK"
]

# Output Files
TRIPS_FILE_NAME      = "final_trips.csv"
HOUSEHOLDS_FILE_NAME = "final_households.csv"

# ============================================================
# 2. TARGETS & METRICS
# ============================================================

OBS_TRIP_RATE = 8.03209

PURPOSE_ORDER = [
    "home", "work", "shopping", "social", "escort",
    "eatout", "school", "othmaint", "othdiscr",
]

OBS_PURPOSE_SERIES = pd.Series({
    'home': 0.336749, 'work': 0.157069, 'shopping': 0.110626,
    'escort': 0.059154, 'othdiscr': 0.043783, 'social': 0.086337,
    'othmaint': 0.129523, 'eatout': 0.043997, 'school': 0.032762,
})[PURPOSE_ORDER]

MODE_ORDER = [
    "Private Vehicle - Driver", "Private Vehicle - Passenger",
    "Walking & Active Transport", "Cycling", "Public Transit - Rail",
    "Public Transit - Bus", "Ride-Hailing & Taxi",
]

OBS_MODE_SERIES = pd.Series({
    'Private Vehicle - Driver': 0.450678, 'Private Vehicle - Passenger': 0.202052,
    'Walking & Active Transport': 0.252056, 'Cycling': 0.015111,
    'Public Transit - Rail': 0.056734, 'Public Transit - Bus': 0.020686,
    'Ride-Hailing & Taxi': 0.002684,
})[MODE_ORDER]

# Weights
W_TRIP    = 20.0  
W_PURPOSE = 1.0
W_MODE    = 1.0
ALPHA_LOSS = 1.0

# Mappings
PURPOSE_MAP = {
    "home": "home", "work": "work", "shopping": "shopping",
    "social": "social", "escort": "escort", "eatout": "eatout",
    "school": "school", "univ": "school", "othmaint": "othmaint",
    "othdiscr": "othdiscr", "atwork": "othdiscr", 
}

MODE_MAP = {
    "DRIVEALONEFREE": "Private Vehicle - Driver", "SHARED2FREE": "Private Vehicle - Passenger",
    "SHARED3FREE": "Private Vehicle - Passenger", "TNC_SINGLE": "Ride-Hailing & Taxi",
    "TNC_SHARED": "Ride-Hailing & Taxi", "TAXI": "Ride-Hailing & Taxi",
    "BIKE": "Cycling", "WALK": "Walking & Active Transport",
    "WALK_LOC": "Public Transit - Bus", "WALK_EXP": "Public Transit - Bus",
    "DRIVE_LOC": "Public Transit - Bus", "DRIVE_EXP": "Public Transit - Bus",
    "WALK_HVY": "Public Transit - Rail", "WALK_COM": "Public Transit - Rail",
    "WALK_LRF": "Public Transit - Rail", "DRIVE_HVY": "Public Transit - Rail",
    "DRIVE_COM": "Public Transit - Rail", "DRIVE_LRF": "Public Transit - Rail",
}

# ============================================================
# 3. UTILS & LOADING
# ============================================================

def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Robust column detection with auto-injection of missing constraints."""
    if "coefficient" in df.columns: coef = "coefficient"
    elif "coefficient_name" in df.columns: coef = "coefficient_name"
    else: raise ValueError(f"No coefficient column found: {df.columns}")
    
    if "value" not in df.columns: raise ValueError("No value column")
    
    if "constrain" in df.columns: 
        cons = "constrain"
    elif "constraint" in df.columns: 
        cons = "constraint"
    else:
        # Auto-fix for files missing the constraint column
        print("   > ⚠️ 'constrain' column missing. Injecting 'F'.")
        df["constrain"] = "F"
        cons = "constrain"
        
    return coef, "value", cons

def load_base_file(active_path: Path, base_path: Path) -> pd.DataFrame:
    """Safely loads base file, creating it from active if missing."""
    if base_path.exists():
        return pd.read_csv(base_path)
    if not active_path.exists():
        # Fallback for alternative naming
        alt = active_path.name.replace("coefficients", "coeffs")
        if (active_path.parent / alt).exists():
            active_path = active_path.parent / alt
        else:
            raise FileNotFoundError(f"Missing {active_path}")
    
    print(f"   > Creating base from active: {active_path.name}")
    df = pd.read_csv(active_path)
    # Ensure columns exist before saving
    _detect_columns(df) 
    df.to_csv(base_path, index=False)
    return df

# --- LOAD BASES ---
print("Loading Coefficient Base Files...")

# 1. Mandatory
BASE_MAND = load_base_file(MANDATORY_COEFF_ACTIVE, MANDATORY_COEFF_BASE)
M_COL, M_VAL, M_CONS = _detect_columns(BASE_MAND)
BASE_MAND[M_VAL] = pd.to_numeric(BASE_MAND[M_VAL], errors="coerce")

# 2. Mode
BASE_MODE = load_base_file(MODE_COEFF_ACTIVE, MODE_COEFF_BASE)
MO_COL, MO_VAL, MO_CONS = _detect_columns(BASE_MODE)
BASE_MODE[MO_VAL] = pd.to_numeric(BASE_MODE[MO_VAL], errors="coerce")

# 3. Non-Mandatory
NON_MAND_BASES: Dict[str, pd.DataFrame] = {}
for suffix in NON_MAND_SUFFIXES:
    fname = f"{NON_MAND_PREFIX}_{suffix}.csv"
    fname_base = f"{NON_MAND_PREFIX}_{suffix}_base.csv"
    try:
        df = load_base_file(BASE_CONFIG_DIR / fname, BASE_CONFIG_DIR / fname_base)
        c, v, cons = _detect_columns(df)
        df[v] = pd.to_numeric(df[v], errors="coerce")
        NON_MAND_BASES[suffix] = df
        print(f" > Loaded Base for {suffix}")
    except Exception as e:
        print(f" > WARNING: Skipping {suffix}: {e}")

if not NON_MAND_BASES:
    print("CRITICAL: No non-mandatory files loaded.")
    exit(1)

# ============================================================
# 4. APPLY THETA
# ============================================================

def apply_theta(theta: np.ndarray) -> None:
    if len(theta) != 11: raise ValueError("Theta must be 11 dims")

    t_work, t_school, t_univ = theta[0:3]
    t_mode = theta[3:8]
    t_shop, t_social, t_other = theta[8:11]

    # 1. Mandatory
    mand = BASE_MAND.copy()
    m_free = mand[M_CONS].fillna("F") == "F"
    
    mask_work = (mand[M_COL].str.contains("work") & m_free)
    mand.loc[mask_work, M_VAL] += t_work
    
    mask_school = (mand[M_COL].str.contains("school") & ~mand[M_COL].str.contains("univ") & m_free)
    mand.loc[mask_school, M_VAL] += t_school
    
    mask_univ = (mand[M_COL].str.contains("univ") & m_free)
    mand.loc[mask_univ, M_VAL] += t_univ
    
    mand.to_csv(MANDATORY_COEFF_ACTIVE, index=False)

    # 2. Mode
    mode = BASE_MODE.copy()
    mo_free = mode[MO_CONS].fillna("F") == "F"
    
    mask_sov = mode[MO_COL].str.startswith("coef_sov") & mo_free
    mode.loc[mask_sov, MO_VAL] += t_mode[0]

    mask_sr = mode[MO_COL].str.startswith("coef_sr") & mo_free
    mode.loc[mask_sr, MO_VAL] += t_mode[1]

    mask_act = (mode[MO_COL].str.startswith("coef_bike") | mode[MO_COL].str.startswith("coef_walk")) & mo_free
    mode.loc[mask_act, MO_VAL] += t_mode[2]

    mask_tr = mode[MO_COL].str.contains("transit") & mo_free
    mode.loc[mask_tr, MO_VAL] += t_mode[3]

    mask_rh = (mode[MO_COL].str.contains("ride_hail") | mode[MO_COL].str.contains("taxi")) & mo_free
    mode.loc[mask_rh, MO_VAL] += t_mode[4]

    mode.to_csv(MODE_COEFF_ACTIVE, index=False)

    # 3. Non-Mandatory
    for suffix, df_base in NON_MAND_BASES.items():
        df_active = df_base.copy()
        c_col, c_val, c_cons = _detect_columns(df_active)
        is_free = df_active[c_cons].fillna("F") == "F"

        mask_shop = df_active[c_col].str.contains("shop", case=False) & is_free
        df_active.loc[mask_shop, c_val] += t_shop

        mask_soc = (
            df_active[c_col].str.contains("social", case=False) | 
            df_active[c_col].str.contains("escort", case=False) |
            df_active[c_col].str.contains("eat", case=False) |
            df_active[c_col].str.contains("visit", case=False)
        ) & is_free
        df_active.loc[mask_soc, c_val] += t_social

        mask_oth = (
            df_active[c_col].str.contains("oth", case=False) | 
            df_active[c_col].str.contains("maint", case=False) | 
            df_active[c_col].str.contains("discr", case=False)
        ) & is_free
        df_active.loc[mask_oth, c_val] += t_other

        fname = f"{NON_MAND_PREFIX}_{suffix}.csv"
        df_active.to_csv(BASE_CONFIG_DIR / fname, index=False)

# ============================================================
# 5. EXECUTION
# ============================================================

def run_activitysim(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ['activitysim', 'run', '-c', str(BASE_CONFIG_DIR), '-d', str(BASE_DATA_DIR), '-o', str(out_dir)]
    
    with open(out_dir / 'asim_stdout.log', 'wb') as so, open(out_dir / 'asim_stderr.log', 'wb') as se:
        try:
            print(f"   > Running ASim in {out_dir.name} ...")
            start_t = time.time()
            proc = subprocess.run(cmd, stdout=so, stderr=se, timeout=7200)
            print(f"   > Finished in {time.time() - start_t:.1f}s with code {proc.returncode}")
            return proc.returncode
        except Exception as e:
            print(f"   > Error launching ASim: {e}")
            return -99

def compute_indicators(out_dir: Path):
    trips_path = out_dir / TRIPS_FILE_NAME
    hh_path = out_dir / HOUSEHOLDS_FILE_NAME
    
    if not trips_path.exists():
        return 0.0, pd.Series(), pd.Series()

    trips = pd.read_csv(trips_path)
    
    if hh_path.exists():
        num_hh = len(pd.read_csv(hh_path))
    else:
        num_hh = trips['household_id'].nunique() if 'household_id' in trips else 1
    
    trip_rate = len(trips) / num_hh if num_hh > 0 else 0.0

    trips["purp"] = trips["purpose"].map(PURPOSE_MAP)
    p_share = trips["purp"].value_counts(normalize=True).reindex(PURPOSE_ORDER).fillna(0)

    trips["mode"] = trips["trip_mode"].map(MODE_MAP)
    m_share = trips["mode"].value_counts(normalize=True).reindex(MODE_ORDER).fillna(0)

    return float(trip_rate), p_share, m_share

def log_cosh(x, alpha):
    abs_ax = np.abs(alpha * x)
    return np.where(abs_ax > 50, abs_ax - np.log(2.0), np.log(np.cosh(alpha * x)))

def get_loss(trip_rate, p_share, m_share):
    L_trip = log_cosh((trip_rate - OBS_TRIP_RATE) / OBS_TRIP_RATE, ALPHA_LOSS)
    L_purp = log_cosh(p_share.values - OBS_PURPOSE_SERIES.values, ALPHA_LOSS).sum()
    L_mode = log_cosh(m_share.values - OBS_MODE_SERIES.values, ALPHA_LOSS).sum()
    return float(np.sum(W_TRIP * L_trip + W_PURPOSE * L_purp + W_MODE * L_mode))

# ============================================================
# 6. SPSA LOOP
# ============================================================

def spsa_calibration(max_iter=30):
    theta = np.zeros(11, dtype=float)
    
    # --- UPDATED HYPERPARAMETERS ---
    # REDUCED a=0.1 to prevent oscillation and crashes
    a = 0.2    
    c = 0.05   
    A = 3.0    
    
    # Inject Single Process
    settings_path = BASE_CONFIG_DIR / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, 'r') as f: content = f.read()
        if "rng_base_seed" not in content:
            print("   > Injecting deterministic settings...")
            with open(settings_path, 'a') as f:
                f.write("\nnum_processes: 1\nrng_base_seed: 0\n")

    for k in range(1, max_iter + 1):
        ak = a / ((k + A) ** 0.602)
        ck = c / (k ** 0.101)
        delta = np.random.choice([-1.0, 1.0], size=11)
        
        print(f"\n=== Iteration {k} / {max_iter} (ak={ak:.4f}) ===")

        # --- PLUS ---
        apply_theta(theta + ck * delta)
        out_p = SPSA_OUTPUT_ROOT / f"iter_{k:03d}_plus"
        
        if run_activitysim(out_p) == 0:
            tp, pp, mp = compute_indicators(out_p)
            L_p = get_loss(tp, pp, mp)
        else:
            print("   > ⚠️ Run (+) CRASHED. Skipping update.")
            continue # CRASH PROTECTION

        # --- MINUS ---
        apply_theta(theta - ck * delta)
        out_m = SPSA_OUTPUT_ROOT / f"iter_{k:03d}_minus"
        
        if run_activitysim(out_m) == 0:
            tm, pm, mm = compute_indicators(out_m)
            L_m = get_loss(tm, pm, mm)
        else:
            print("   > ⚠️ Run (-) CRASHED. Skipping update.")
            continue # CRASH PROTECTION

        print(f"   > L+={L_p:.4f}, L-={L_m:.4f} | Diff: {L_p - L_m:.4f}")
        print(f"   > TripRate+: {tp:.3f}, TripRate-: {tm:.3f}")

        # --- UPDATE ---
        grad = (L_p - L_m) / (2.0 * ck) * delta
        
        # Clip Gradient Norm
        gnorm = np.linalg.norm(grad)
        if gnorm > 5.0:
            grad = grad * (5.0 / gnorm)
            print(f"   > ✂️ Gradient clipped: {gnorm:.2f} -> 5.0")

        theta = theta - ak * grad
        
        # --- PARAMETER BOUNDING (Prevent Explosion) ---
        theta = np.clip(theta, -4.0, 4.0)

        print(f"   > Updated Theta: {np.round(theta, 3)}")

if __name__ == "__main__":
    print(f"Starting Baseline SPSA Calibration")
    spsa_calibration()
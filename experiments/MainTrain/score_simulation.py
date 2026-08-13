import numpy as np
import pandas as pd

# Observed data, mappings, weights
OBSERVED_DATA_FULL = {
    "total_trips": 242537,
    "total_households": 30196,
    "purpose_dist": {
        "home": 81674, "work": 38095, "shopping": 26831,
        "escort": 14347, "othdiscr": 10619, "social": 20940,
        "othmaint": 31414, "eatout": 10671, "school": 7946
    },
    "mode_dist": {
        "Private Vehicle - Driver": 109306,
        "Private Vehicle - Passenger": 49005,
        "Walking & Active Transport": 61133,
        "Cycling": 3665,
        "Public Transit - Rail": 13760,
        "Public Transit - Bus": 5017,
        "Ride-Hailing & Taxi": 651
    }
}
OBSERVED_METRICS = {
    "trip_rate": OBSERVED_DATA_FULL["total_trips"] / OBSERVED_DATA_FULL["total_households"],
    "purpose_probabilities": {k: v / OBSERVED_DATA_FULL["total_trips"] 
                             for k, v in OBSERVED_DATA_FULL["purpose_dist"].items()},
    "mode_probabilities": {k: v / OBSERVED_DATA_FULL["total_trips"] 
                          for k, v in OBSERVED_DATA_FULL["mode_dist"].items()}
}
UNIFIED_MODE_MAPPING = {
    'DRIVEALONEFREE': 'Private Vehicle - Driver',
    'SHARED2FREE': 'Private Vehicle - Passenger', 
    'SHARED3FREE': 'Private Vehicle - Passenger',
    'WALK': 'Walking & Active Transport', 
    'BIKE': 'Cycling',
    'WALK_HVY': 'Public Transit - Rail', 
    'WALK_LRF': 'Public Transit - Rail',
    'DRIVE_HVY': 'Public Transit - Rail', 
    'DRIVE_LRF': 'Public Transit - Rail',
    'WALK_COM': 'Public Transit - Rail', 
    'DRIVE_COM': 'Public Transit - Rail',
    'WALK_LOC': 'Public Transit - Bus', 
    'WALK_EXP': 'Public Transit - Bus',
    'DRIVE_LOC': 'Public Transit - Bus', 
    'DRIVE_EXP': 'Public Transit - Bus',
    'TNC_SINGLE': 'Ride-Hailing & Taxi', 
    'TNC_SHARED': 'Ride-Hailing & Taxi',
    'TAXI': 'Ride-Hailing & Taxi'
}
PURPOSE_MAPPING = {
    'home': 'home', 'work': 'work', 'atwork': 'work', 'shopping': 'shopping',
    'escort': 'escort', 'othdiscr': 'othdiscr', 'social': 'social',
    'othmaint': 'othmaint', 'eatout': 'eatout', 'school': 'school', 'univ': 'school'
}
LOSS_WEIGHTS = dict(trips=0.25, purpose=0.40, mode=0.35)

# ---- Log-cosh Loss Objective ----
def log_cosh(x, alpha=1.0):
    return np.log(np.cosh(alpha * x))

def calculate_trip_rate_loss(actual, target, alpha=2.0):
    rel_err = (actual - target) / target
    return log_cosh(rel_err, alpha)

def calculate_dist_loss(actual_probs, target_probs, alpha=4.0):
    abs_diffs = np.abs(actual_probs - target_probs)
    return np.sum(log_cosh(abs_diffs, alpha))

def map_column(series, mapping):
    mapped = series.map(mapping)
    mapped = mapped.fillna(series)
    return mapped

def get_probs(series, categories):
    counts = series.value_counts()
    total = len(series)
    return np.array([counts.get(cat, 0) / total for cat in categories])

def evaluate_simulation(trips_df, n_households):
    # 1. Trip rate
    actual_trip_rate = len(trips_df) / n_households
    target_trip_rate = OBSERVED_METRICS["trip_rate"]
    trip_loss = calculate_trip_rate_loss(actual_trip_rate, target_trip_rate)
    
    # 2. Purpose distribution
    purpose_cats = list(OBSERVED_METRICS["purpose_probabilities"].keys())
    mapped_purpose = map_column(trips_df['purpose'], PURPOSE_MAPPING)
    actual_purpose_probs = get_probs(mapped_purpose, purpose_cats)
    target_purpose_probs = np.array(list(OBSERVED_METRICS["purpose_probabilities"].values()))
    purpose_loss = calculate_dist_loss(actual_purpose_probs, target_purpose_probs)
    
    # 3. Mode distribution
    mode_cats = list(OBSERVED_METRICS["mode_probabilities"].keys())
    mapped_mode = map_column(trips_df['trip_mode'], UNIFIED_MODE_MAPPING)
    actual_mode_probs = get_probs(mapped_mode, mode_cats)
    target_mode_probs = np.array(list(OBSERVED_METRICS["mode_probabilities"].values()))
    mode_loss = calculate_dist_loss(actual_mode_probs, target_mode_probs)
    
    # Composite score
    total_loss = (LOSS_WEIGHTS['trips'] * trip_loss +
                  LOSS_WEIGHTS['purpose'] * purpose_loss +
                  LOSS_WEIGHTS['mode'] * mode_loss)
    composite_score = 100.0 * np.exp(-total_loss)
    # Component scores for diagnostics
    trip_rate_score = 100.0 * np.exp(-trip_loss / LOSS_WEIGHTS['trips'])
    purpose_score = 100.0 * np.exp(-purpose_loss / LOSS_WEIGHTS['purpose'])
    mode_score = 100.0 * np.exp(-mode_loss / LOSS_WEIGHTS['mode'])
    return {
        "composite_score": composite_score,
        "trip_rate_score": trip_rate_score,
        "purpose_score": purpose_score,
        "mode_score": mode_score,
        "trip_rate_actual": actual_trip_rate,
        "trip_rate_target": target_trip_rate,
        "purpose_probs_actual": dict(zip(purpose_cats, actual_purpose_probs)),
        "purpose_probs_target": dict(zip(purpose_cats, target_purpose_probs)),
        "mode_probs_actual": dict(zip(mode_cats, actual_mode_probs)),
        "mode_probs_target": dict(zip(mode_cats, target_mode_probs)),
    }

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # Change these as needed
    FINAL_TRIPS_CSV = "output/final_trips.csv"
    N_HOUSEHOLDS = 30196   # or load from households.csv

    trips = pd.read_csv(FINAL_TRIPS_CSV)
    result = evaluate_simulation(trips, N_HOUSEHOLDS)
    
    print("Composite Score: {:.2f}".format(result['composite_score']))
    print("Trip Rate Score: {:.2f}".format(result['trip_rate_score']))
    print("Purpose Score: {:.2f}".format(result['purpose_score']))
    print("Mode Score: {:.2f}".format(result['mode_score']))
    print("Actual Trip Rate: {:.4f}, Target: {:.4f}".format(
        result['trip_rate_actual'], result['trip_rate_target']))
    print("Purpose Actual vs Target:")
    for k in result['purpose_probs_actual']:
        print(f"  {k}: {result['purpose_probs_actual'][k]:.3f} vs {result['purpose_probs_target'][k]:.3f}")
    print("Mode Actual vs Target:")
    for k in result['mode_probs_actual']:
        print(f"  {k}: {result['mode_probs_actual'][k]:.3f} vs {result['mode_probs_target'][k]:.3f}")

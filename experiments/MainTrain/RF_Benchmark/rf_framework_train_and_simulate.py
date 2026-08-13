#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF-based disaggregate framework (paper-aligned).
Stops_VISTA is used *only* to rewrite PURPOSE & MODE (incl. last-purpose/last-mode).
Times of day remain from training data; we do not touch them.

Run:
python rf_framework_train_and_simulate.py \
  --tgm TGM.csv \
  --ftam1 FTAM1_trained_with_ids.csv \
  --ftam2 FTAM2_trained_with_ids.csv \
  --ntam1 NTAM1_with_ids.csv \
  --ntam2 NTAM2_with_ids.csv \
  --stops_vista Stop_VISTA.csv \
  --outdir ./rf_results
"""

from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================ I/O & small helpers ============================

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        cl = c.lower()
        if cl in lower:
            return lower[cl]
    by_norm = {_norm_name(c): c for c in df.columns}
    for c in candidates:
        nc = _norm_name(c)
        if nc in by_norm:
            return by_norm[nc]
    return None

def _coerce_join_keys_to_str(left: pd.DataFrame, right: pd.DataFrame, keys: List[str]):
    for k in keys:
        if k in left.columns:
            left[k] = left[k].astype("string")
        if k in right.columns:
            right[k] = right[k].astype("string")

def _canon_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


# ============================ Desired taxonomies ============================

DESIRED_PURPOSES = [
    "home","work","shopping","social","escort","eatout","school","othmaint","otherdiscr"
]

DESIRED_MODES = [
    "Walking","Cycling","Public Rail","Public Bus","Ride-Hailing&Taxi","Private Driver","Private Passenger"
]

# Purpose mapping (raw -> desired) per your spec
_PURPOSE_RAW_TO_DESIRED = {
    "At or Go Home": "home",
    "Change Mode": "otherdiscr",
    "Work Related": "work",
    "Buy Something": "shopping",
    "Social": "social",
    "Pick-up or Drop-off Someone": "escort",
    "Personal Business": "othmaint",
    "Recreational": "otherdiscr",
    "Education": "school",
    "Accompany Someone": "escort",
    "Pick-up or Deliver Something": "escort",
    "Other Purpose": "otherdiscr",
    # allow already-desired labels to pass
    "home": "home", "work": "work", "shopping": "shopping", "social": "social",
    "escort": "escort", "eatout": "eatout", "school": "school",
    "othmaint": "othmaint", "otherdiscr": "otherdiscr"
}
_PURPOSE_LUT = {_canon_text(k): v for k, v in _PURPOSE_RAW_TO_DESIRED.items()}

def normalize_purpose(s: pd.Series) -> pd.Series:
    if s is None: return s
    wanted = set(DESIRED_PURPOSES)
    def map_one(v):
        if pd.isna(v): return "otherdiscr"
        t = _canon_text(v)
        if t in _PURPOSE_LUT: return _PURPOSE_LUT[t]
        # heuristics for truncated/variant text
        if t.startswith("work"): return "work"
        if t.startswith("buy"): return "shopping"
        if "social" in t: return "social"
        if "educat" in t or "school" in t: return "school"
        if "pick" in t or "drop" in t or "deliver" in t or "accompany" in t or "escort" in t: return "escort"
        if "personal" in t or "business" in t or "errand" in t or "maint" in t: return "othmaint"
        if "home" in t: return "home"
        if "recreat" in t or "leisure" in t or "change mode" in t or "change" in t: return "otherdiscr"
        if "eat" in t or "restaurant" in t or "cafe" in t or "dine" in t: return "eatout"
        return "otherdiscr"
    out = s.map(map_one)
    return out.where(out.isin(wanted), "otherdiscr")

# Mode mapping (raw -> desired) per your spec
_MODE_RAW_TO_DESIRED = {
    "Car Driver": "Private Driver",
    "Car Passenger": "Private Passenger",
    "4WD Driver": "Private Driver",
    "4WD Passenger": "Private Passenger",
    "Ute Driver": "Private Driver",
    "Ute Passenger": "Private Passenger",
    "Van Driver": "Private Driver",
    "Van Passenger": "Private Passenger",
    "Truck Driver": "Private Driver",
    "Truck Passenger": "Private Passenger",
    "Motorcycle Rider": "Private Driver",
    "Motorcycle Passenger": "Private Passenger",
    "Train": "Public Rail",
    "Tram": "Public Rail",
    "Public Bus": "Public Bus",
    "School Bus": "Public Bus",
    "Taxi": "Ride-Hailing&Taxi",
    "Walking": "Walking",
    "Jogging": "Walking",
    "Mobility Scooter": "Walking",
    "Bicycle": "Cycling",
    "Other": "Private Driver",
    # allow already-desired labels to pass
    "Private Driver": "Private Driver",
    "Private Passenger": "Private Passenger",
    "Cycling": "Cycling",
    "Public Rail": "Public Rail",
    "Public Bus": "Public Bus",
    "Ride-Hailing&Taxi": "Ride-Hailing&Taxi",
    "Walking": "Walking"
}
_MODE_LUT = {_canon_text(k): v for k, v in _MODE_RAW_TO_DESIRED.items()}

def normalize_mode(s: pd.Series) -> pd.Series:
    if s is None: return s
    wanted = set(DESIRED_MODES)
    def map_one(v):
        if pd.isna(v): return "Walking"
        t = _canon_text(v)
        if t in _MODE_LUT: return _MODE_LUT[t]
        # heuristic catch-alls
        if t.startswith("car ") and "driver" in t: return "Private Driver"
        if t.startswith("car ") and "pass" in t: return "Private Passenger"
        if "4wd" in t and "driver" in t: return "Private Driver"
        if "4wd" in t and "pass" in t: return "Private Passenger"
        if "ute" in t and "driver" in t: return "Private Driver"
        if "ute" in t and "pass" in t: return "Private Passenger"
        if "van" in t and "driver" in t: return "Private Driver"
        if "van" in t and "pass" in t: return "Private Passenger"
        if "truck" in t and "driver" in t: return "Private Driver"
        if "truck" in t and "pass" in t: return "Private Passenger"
        if "motorcycle" in t and "pass" in t: return "Private Passenger"
        if "motorcycle" in t: return "Private Driver"
        if "train" in t or "tram" in t or "rail" in t: return "Public Rail"
        if "bus" in t: return "Public Bus"
        if "taxi" in t or "ride" in t or "uber" in t or "lyft" in t: return "Ride-Hailing&Taxi"
        if "bicycle" in t or "cycle" in t or "bike" in t: return "Cycling"
        if "walk" in t or "jog" in t or "scooter" in t: return "Walking"
        return "Private Driver"
    out = s.map(map_one)
    return out.where(out.isin(wanted), "Private Driver")


# ============================ RF plumbing ============================

@dataclass
class RFSpec:
    target_col: str
    feature_cols: List[str]
    model_path: str
    random_state: int = 42

def build_rf_pipeline(
    df: pd.DataFrame,
    features: List[str],
    random_state: int = 42,
    n_estimators: int = 1000,
    min_samples_leaf: int = 100,
) -> Pipeline:
    X = df[features]
    cat_cols = [c for c in features if X[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=-1))
        ]), num_cols),
    ])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        max_features="sqrt",       # paper: √p
        min_samples_leaf=min_samples_leaf,  # paper: 100
        bootstrap=True,
        oob_score=True,            # paper: OOB error as primary metric
        class_weight="balanced_subsample",         # paper: not specified -> None
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", rf)])

def train_rf(df: pd.DataFrame, spec: RFSpec) -> Pipeline:
    d = df.dropna(subset=[spec.target_col]).copy()
    y = d[spec.target_col].astype(str)
    pipe = build_rf_pipeline(d, spec.feature_cols, random_state=spec.random_state)
    pipe.fit(d[spec.feature_cols], y)
    try:
        print(f"\n== {spec.model_path} ==\nOOB accuracy: {pipe.named_steps['rf'].oob_score_:.4f}")
    except Exception as e:
        print(f"[WARN] OOB not available for {spec.model_path}: {e}")
    joblib.dump(pipe, spec.model_path)
    return pipe


# ============================ Model inputs ============================

def load_tgm(path: str) -> Tuple[pd.DataFrame, str, Optional[str], List[str]]:
    df = read_csv(path)
    pid = find_col(df, ["persid","person_id","PERID","pid"]) or "persid"
    hid = find_col(df, ["hhid","household_id","HHID"])
    m_col = find_col(df, ["motorized"])
    nm_col = find_col(df, ["non-motorized","nonmotorized","non_motorized"])
    if m_col is None or nm_col is None:
        raise ValueError("TGM must include 'motorized' and 'non-motorized' columns.")

    trips = (
        pd.to_numeric(df[m_col], errors="coerce").fillna(0).round().astype(int) +
        pd.to_numeric(df[nm_col], errors="coerce").fillna(0).round().astype(int)
    )
    keep = (trips >= 0) & (trips <= 10)  # 0..10 as in paper table
    df = df.loc[keep].copy()
    df["_tgm_trip_count_int"] = trips.loc[keep].values
    df["tgm_target"] = df["_tgm_trip_count_int"].astype(str)

    candidate_features = [
        "age","sex","relationship","carlicence","mbikelicence","studying",
        "mainact","worktype","dwelltype","owndwell","hhsize","hhinc",
        "AverageAvakiableBikes","AverageAvaliableCars","IRSAD","IRSD","IER","IEO",
        "Population"
    ]
    features = [c for c in candidate_features if c in df.columns]
    return df, pid, hid, features

def load_ftam1(path: str) -> Tuple[pd.DataFrame, str, List[str]]:
    df = read_csv(path)
    pid = find_col(df, ["persid","person_id","PERID","pid"]) or "persid"
    if "timeofday" in df.columns:
        df["timeofday"] = df["timeofday"].astype(str)
    features = [
        "age","sex","relationship","carlicence","mbikelicence","studying",
        "mainact","worktype","dwelltype","owndwell","hhsize","hhinc",
        "AverageAvakiableBikes","AverageAvaliableCars","IRSAD","IRSD","IER","IEO",
        "Population","motorized","non-motorized",
        "Is home the origin","Is it the last trip of day","timeofday"
    ]
    features = [c for c in features if c in df.columns]
    return df, pid, features

def load_ftam2(path: str) -> Tuple[pd.DataFrame, str, List[str]]:
    df = read_csv(path)
    pid = find_col(df, ["persid","person_id","PERID","pid"]) or "persid"
    if "timeofday" in df.columns:
        df["timeofday"] = df["timeofday"].astype(str)
    features = [
        "age","sex","relationship","carlicence","mbikelicence","studying",
        "mainact","worktype","dwelltype","owndwell","hhsize","hhinc",
        "AverageAvakiableBikes","AverageAvaliableCars","IRSAD","IRSD","IER","IEO",
        "Population","motorized","non-motorized",
        "Is home the origin","Is it the last trip of day","timeofday","trippurp",
        "IRSAD_destination","IRSD_destination","IER_destination","IEO_destination"
    ]
    features = [c for c in features if c in df.columns]
    return df, pid, features

def load_ntam1(path: str) -> Tuple[pd.DataFrame, str, Optional[str], List[str]]:
    df = read_csv(path)
    pid = find_col(df, ["persid","person_id","PERID","pid"]) or "persid"
    idx = find_col(df, ["trip_index","tripno","trip_no","trip_number","seq","trip_seq"]) or "trip_index"
    if "timeofday" in df.columns:
        df["timeofday"] = df["timeofday"].astype(str)
    if "lasttimeofday" in df.columns:
        df["lasttimeofday"] = df["lasttimeofday"].astype(str)
    features = [
        "age","sex","relationship","carlicence","mbikelicence","studying",
        "mainact","worktype","dwelltype","owndwell","hhsize","hhinc",
        "AverageAvakiableBikes","AverageAvaliableCars","IRSAD","IRSD","IER","IEO",
        "Population","motorized","non-motorized",
        "Is home the origin","Is it the last trip of day",
        "lasttimeofday","lasttrippurp","OriginIRSAD","OriginIRSD","OriginIER","OriginIEO",
        "lastmode","lasttriplength","timeofday"
    ]
    features = [c for c in features if c in df.columns]
    return df, pid, idx, features

def load_ntam2(path: str) -> Tuple[pd.DataFrame, str, Optional[str], List[str]]:
    df = read_csv(path)
    pid = find_col(df, ["persid","person_id","PERID","pid"]) or "persid"
    idx = find_col(df, ["trip_index","tripno","trip_no","trip_number","seq","trip_seq"]) or "trip_index"
    if "timeofday" in df.columns:
        df["timeofday"] = df["timeofday"].astype(str)
    if "lasttimeofday" in df.columns:
        df["lasttimeofday"] = df["lasttimeofday"].astype(str)
    features = [
        "age","sex","relationship","carlicence","mbikelicence","studying",
        "mainact","worktype","dwelltype","owndwell","hhsize","hhinc",
        "AverageAvakiableBikes","AverageAvaliableCars","IRSAD","IRSD","IER","IEO",
        "Population","motorized","non-motorized",
        "Is home the origin","Is it the last trip of day","lasttimeofday","lasttrippurp","lastmode","lasttriplength",
        "OriginIRSAD","OriginIRSD","OriginIER","OriginIEO","timeofday","trippurp",
        "IRSAD_destination","IRSD_destination","IER_destination","IEO_destination"
    ]
    features = [c for c in features if c in df.columns]
    return df, pid, idx, features


# ============================ Stops loader (purpose & mode ONLY) ============================

def load_stops_purpose_mode_only(path: str):
    """
    Load Stops_VISTA and produce:
      - trip_index per person (if not present, compute by file order)
      - mapped purpose and mode for current trip
      - mapped last purpose and last mode (shift by 1 per person)
    Uses columns: 'persid' (id), optional 'hhid', 'destpurp1' (purpose), 'mainmode' (mode).
    """
    sv = read_csv(path)

    spid = find_col(sv, ["persid","person_id","PERID","pid"])
    shid = find_col(sv, ["hhid","household_id","HHID"])
    purp_raw = find_col(sv, ["destpurp1"])  # <-- as specified
    mode_raw = find_col(sv, ["fullmode","mainmode"])   # prefer fullmode if it exists
    idx_raw  = find_col(sv, ["trip_index","tripno","trip_no","trip_number","seq","trip_seq","stopno"])

    if spid is None:
        raise ValueError("Stops_VISTA must contain a person id column (e.g., 'persid').")

    # If no index, create one by file order within person
    if idx_raw is None:
        sv["trip_index"] = sv.groupby(spid).cumcount() + 1
        idx_raw = "trip_index"

    # Map raw -> desired
    sv["purpose_mapped"] = normalize_purpose(sv[purp_raw]) if purp_raw else "otherdiscr"
    sv["mode_mapped"]    = normalize_mode(sv[mode_raw])   if mode_raw else "Walking"

    # Previous trip's mapped purpose/mode
    sv = sv.sort_values([spid, idx_raw]).copy()
    sv["lastpurpose_mapped"] = sv.groupby(spid)["purpose_mapped"].shift(1)
    sv["lastmode_mapped"]    = sv.groupby(spid)["mode_mapped"].shift(1)

    # For debugging
    print(f"[DBG] Stops columns used -> pid:{spid} hid:{shid} purpose:{purp_raw} mode:{mode_raw} idx:{idx_raw}")
    print("[DBG] Stops purpose_mapped head:", sv["purpose_mapped"].value_counts().head().to_dict())
    print("[DBG] Stops mode_mapped head:", sv["mode_mapped"].value_counts().head().to_dict())

    return sv, spid, shid, idx_raw, "purpose_mapped", "mode_mapped", "lastpurpose_mapped", "lastmode_mapped"


def relabel_from_stops_purpose_mode_only(
    ft1_df, ft1_pid,
    ft2_df, ft2_pid,
    nt1_df, nt1_pid, nt1_idx,
    nt2_df, nt2_pid, nt2_idx,
    stops_path: str,
):
    sv, spid, shid, sidx, spurp, smode, slastpurp, slastmode = load_stops_purpose_mode_only(stops_path)

    def join_keys_have(df, pid_name, hid_name):
        keys = []
        if pid_name and pid_name in df.columns:
            keys.append(pid_name)
        if hid_name and hid_name in df.columns:
            keys.append(hid_name)
        return keys

    # FTAM1: FIRST TRIP ONLY -> rewrite 'trippurp'
    if ft1_pid and (ft1_pid in ft1_df.columns):
        first = sv[sv[sidx] == 1][[spid, shid, spurp]].copy()
        first = first.rename(columns={
            spid: ft1_pid,
            shid: "hhid" if ("hhid" in ft1_df.columns) else shid,
            spurp: "trippurp"
        })
        k = join_keys_have(ft1_df, ft1_pid, "hhid" if "hhid" in ft1_df.columns and shid else None)
        if k:
            _coerce_join_keys_to_str(ft1_df, first, k)
            ft1_df = ft1_df.drop(columns=["trippurp"], errors="ignore").merge(first, on=k, how="left", suffixes=("", "_stops"))
            print("[DBG] FTAM1 labels:", ft1_df["trippurp"].value_counts(dropna=False).head().to_dict())
        else:
            print("[WARN] FTAM1 has no join keys (need persid, optional hhid). Skipping relabel for FTAM1.")

    # FTAM2: FIRST TRIP ONLY -> rewrite 'trippurp' & 'mode'
    if ft2_pid and (ft2_pid in ft2_df.columns):
        first = sv[sv[sidx] == 1][[spid, shid, spurp, smode]].copy()
        first = first.rename(columns={
            spid: ft2_pid,
            shid: "hhid" if ("hhid" in ft2_df.columns) else shid,
            spurp: "trippurp", smode: "mode"
        })
        k = join_keys_have(ft2_df, ft2_pid, "hhid" if "hhid" in ft2_df.columns and shid else None)
        if k:
            _coerce_join_keys_to_str(ft2_df, first, k)
            ft2_df = ft2_df.drop(columns=["trippurp","mode"], errors="ignore").merge(first, on=k, how="left", suffixes=("", "_stops"))
            print("[DBG] FTAM2 purp:", ft2_df["trippurp"].value_counts(dropna=False).head().to_dict())
            print("[DBG] FTAM2 mode:", ft2_df["mode"].value_counts(dropna=False).head().to_dict())
        else:
            print("[WARN] FTAM2 has no join keys (need persid, optional hhid). Skipping relabel for FTAM2.")

    # NTAM1: TRIPS >= 2 -> rewrite 'trippurp', 'lasttrippurp', 'lastmode'
    if nt1_pid and (nt1_pid in nt1_df.columns) and nt1_idx and (nt1_idx in nt1_df.columns):
        later = sv[sv[sidx] >= 2][[spid, shid, sidx, spurp, slastpurp, slastmode]].copy()
        later = later.rename(columns={
            spid: nt1_pid, shid: "hhid" if ("hhid" in nt1_df.columns) else shid,
            sidx: nt1_idx, spurp: "trippurp", slastpurp: "lasttrippurp", slastmode: "lastmode"
        })
        k = join_keys_have(nt1_df, nt1_pid, "hhid" if "hhid" in nt1_df.columns and shid else None) + [nt1_idx]
        _coerce_join_keys_to_str(nt1_df, later, k)
        nt1_df = nt1_df.drop(columns=["trippurp","lasttrippurp","lastmode"], errors="ignore").merge(later, on=k, how="left", suffixes=("", "_stops"))
        print("[DBG] NTAM1 purp:", nt1_df["trippurp"].value_counts(dropna=False).head().to_dict())
        print("[DBG] NTAM1 lastpurp:", nt1_df["lasttrippurp"].value_counts(dropna=False).head().to_dict())
        print("[DBG] NTAM1 lastmode:", nt1_df["lastmode"].value_counts(dropna=False).head().to_dict())
    else:
        print("[WARN] NTAM1 lacks persid and/or trip_index. Skipping relabel for NTAM1.")

    # NTAM2: TRIPS >= 2 -> rewrite 'trippurp', 'mode', 'lasttrippurp', 'lastmode'
    if nt2_pid and (nt2_pid in nt2_df.columns) and nt2_idx and (nt2_idx in nt2_df.columns):
        later = sv[sv[sidx] >= 2][[spid, shid, sidx, spurp, smode, slastpurp, slastmode]].copy()
        later = later.rename(columns={
            spid: nt2_pid, shid: "hhid" if ("hhid" in nt2_df.columns) else shid,
            sidx: nt2_idx, spurp: "trippurp", smode: "mode", slastpurp: "lasttrippurp", slastmode: "lastmode"
        })
        k = join_keys_have(nt2_df, nt2_pid, "hhid" if "hhid" in nt2_df.columns and shid else None) + [nt2_idx]
        _coerce_join_keys_to_str(nt2_df, later, k)
        nt2_df = nt2_df.drop(columns=["trippurp","mode","lasttrippurp","lastmode"], errors="ignore").merge(later, on=k, how="left", suffixes=("", "_stops"))
        print("[DBG] NTAM2 purp:", nt2_df["trippurp"].value_counts(dropna=False).head().to_dict())
        print("[DBG] NTAM2 mode:", nt2_df["mode"].value_counts(dropna=False).head().to_dict())
        print("[DBG] NTAM2 lastpurp:", nt2_df["lasttrippurp"].value_counts(dropna=False).head().to_dict())
        print("[DBG] NTAM2 lastmode:", nt2_df["lastmode"].value_counts(dropna=False).head().to_dict())
    else:
        print("[WARN] NTAM2 lacks persid and/or trip_index. Skipping relabel for NTAM2.")

    # Ensure mapped categories (idempotent)
    for df, cols in [
        (ft1_df, ["trippurp"]),
        (ft2_df, ["trippurp","mode"]),
        (nt1_df, ["trippurp","lasttrippurp","lastmode"]),
        (nt2_df, ["trippurp","mode","lasttrippurp","lastmode"]),
    ]:
        for c in cols:
            if c in df.columns:
                if "mode" in c:
                    df[c] = normalize_mode(df[c])
                else:
                    df[c] = normalize_purpose(df[c])

    return ft1_df, ft2_df, nt1_df, nt2_df


# ============================ Simulation (batched) ============================

@dataclass
class ModelStub:
    pass  # for spec placeholders during simulation call

def simulate_batched(
    tgm_df, tgm_pid, tgm_hid, tgm_features,
    rf_TGM,
    rf_FT1_purp, spec_FT1_purp,
    rf_FT1_tod,  spec_FT1_tod,   # may be None
    rf_FT2_mode, spec_FT2_mode,
    rf_NT1_purp, spec_NT1_purp,
    rf_NT1_tod,  spec_NT1_tod,   # may be None
    rf_NT2_mode, spec_NT2_mode,
):
    persons = tgm_df[[tgm_pid] + ([tgm_hid] if tgm_hid else []) + tgm_features].copy()
    persons["pred_trips"] = rf_TGM.predict(persons[tgm_features]).astype(int)

    def Xframe(base_df, feats):
        X = pd.DataFrame(index=base_df.index)
        for c in feats:
            X[c] = base_df[c] if c in base_df.columns else np.nan
        return X[feats]

    rows = []

    # Trip 1
    m1 = persons["pred_trips"] >= 1
    if m1.any():
        P = persons.loc[m1].copy()
        P["trip_index"] = 1

        if rf_FT1_tod is not None:
            tod1 = rf_FT1_tod.predict(Xframe(P, spec_FT1_tod.feature_cols))
        else:
            tod1 = np.array(["Unknown"]*len(P))

        P_ft1 = P.copy(); P_ft1["timeofday"] = tod1
        purp1 = rf_FT1_purp.predict(Xframe(P_ft1, spec_FT1_purp.feature_cols))

        P_ft2 = P.copy(); P_ft2["timeofday"] = tod1; P_ft2["trippurp"] = purp1
        mode1 = rf_FT2_mode.predict(Xframe(P_ft2, spec_FT2_mode.feature_cols))

        remain = np.clip(P["pred_trips"].astype("int64").values - 1, 0, None)

        rows.append(pd.DataFrame({
            "person_id": P[tgm_pid].values,
            "household_id": P[tgm_hid].values if tgm_hid else [None]*len(P),
            "trip_index": 1, "purpose": purp1, "timeofday": tod1, "mode": mode1
        }))

        state = pd.DataFrame({
            tgm_pid: P[tgm_pid].values,
            "household_id": P[tgm_hid].values if tgm_hid else [None]*len(P),
            "remaining": remain,
            "lasttrippurp": purp1,
            "lasttimeofday": tod1,
            "lastmode": mode1
        })
    else:
        state = pd.DataFrame(columns=[tgm_pid,"household_id","remaining","lasttrippurp","lasttimeofday","lastmode"])

    # Trips 2..N
    layer = 2
    while True:
        active = state[state["remaining"] > 0].copy()
        if active.empty:
            break
        base = persons.set_index(tgm_pid).join(active.set_index(tgm_pid), how="inner", rsuffix="_st").reset_index()

        if rf_NT1_tod is not None:
            todL = rf_NT1_tod.predict(Xframe(base, spec_NT1_tod.feature_cols))
        else:
            todL = np.array(["Unknown"]*len(base))

        base_p = base.assign(timeofday=todL)
        purpL = rf_NT1_purp.predict(Xframe(base_p, spec_NT1_purp.feature_cols))

        base_m = base.assign(trippurp=purpL, timeofday=todL)
        modeL = rf_NT2_mode.predict(Xframe(base_m, spec_NT2_mode.feature_cols))

        rows.append(pd.DataFrame({
            "person_id": base[tgm_pid].values,
            "household_id": base[tgm_hid].values if tgm_hid else [None]*len(base),
            "trip_index": layer, "purpose": purpL, "timeofday": todL, "mode": modeL
        }))

        key = state[tgm_pid].isin(base[tgm_pid])
        state.loc[key, "remaining"] = np.clip(state.loc[key, "remaining"].astype("int64").values - 1, 0, None)
        state.loc[key, "lasttrippurp"]  = purpL
        state.loc[key, "lasttimeofday"] = todL
        state.loc[key, "lastmode"]      = modeL

        layer += 1

    diary = pd.concat(rows, ignore_index=True) if rows else \
            pd.DataFrame(columns=["person_id","household_id","trip_index","purpose","timeofday","mode"])
    return diary


# ============================ Main driver ============================

def train_all_and_simulate(
    tgm_csv: str,
    ftam1_csv: str,
    ftam2_csv: str,
    ntam1_csv: str,
    ntam2_csv: str,
    stops_vista_csv: str,
    outdir: str = "./rf_results",
    random_state: int = 42,
):
    os.makedirs(outdir, exist_ok=True)

    # ---- Load base tables ----
    tgm_df, tgm_pid, tgm_hid, tgm_features = load_tgm(tgm_csv)
    ft1_df, ft1_pid, ft1_features = load_ftam1(ftam1_csv)
    ft2_df, ft2_pid, ft2_features = load_ftam2(ftam2_csv)
    nt1_df, nt1_pid, nt1_idx, nt1_features = load_ntam1(ntam1_csv)
    nt2_df, nt2_pid, nt2_idx, nt2_features = load_ntam2(ntam2_csv)

    # ---- Relabel PURPOSE & MODE only from Stops ----
    ft1_df, ft2_df, nt1_df, nt2_df = relabel_from_stops_purpose_mode_only(
        ft1_df, ft1_pid, ft2_df, ft2_pid, nt1_df, nt1_pid, nt1_idx, nt2_df, nt2_pid, nt2_idx,
        stops_path=stops_vista_csv
    )

    # ---- Train RFs (paper settings) ----
    rf_TGM = train_rf(tgm_df, RFSpec("tgm_target", tgm_features, os.path.join(outdir, "rf_TGM.joblib"), random_state))

    rf_FT1_purp = train_rf(ft1_df, RFSpec("trippurp", ft1_features, os.path.join(outdir, "rf_FTAM1_purpose.joblib"), random_state))
    rf_FT1_tod, spec_FT1_tod = None, None
    if "timeofday" in ft1_df.columns:
        spec_FT1_tod = RFSpec("timeofday", [c for c in ft1_features if c != "timeofday"], os.path.join(outdir, "rf_FTAM1_timeofday.joblib"), random_state)
        rf_FT1_tod = train_rf(ft1_df, spec_FT1_tod)

    rf_FT2_mode = train_rf(ft2_df, RFSpec("mode", ft2_features, os.path.join(outdir, "rf_FTAM2_mode.joblib"), random_state))

    rf_NT1_purp = train_rf(nt1_df, RFSpec("trippurp", nt1_features, os.path.join(outdir, "rf_NTAM1_purpose.joblib"), random_state))
    rf_NT1_tod, spec_NT1_tod = None, None
    if "timeofday" in nt1_df.columns:
        spec_NT1_tod = RFSpec("timeofday", [c for c in nt1_features if c != "timeofday"], os.path.join(outdir, "rf_NTAM1_timeofday.joblib"), random_state)
        rf_NT1_tod = train_rf(nt1_df, spec_NT1_tod)

    rf_NT2_mode = train_rf(nt2_df, RFSpec("mode", nt2_features, os.path.join(outdir, "rf_NTAM2_mode.joblib"), random_state))

    # ---- Simulate (batched) ----
    diary = simulate_batched(
        tgm_df, tgm_pid, tgm_hid, tgm_features,
        rf_TGM,
        rf_FT1_purp, RFSpec("trippurp", ft1_features, ""),
        rf_FT1_tod,  spec_FT1_tod,
        rf_FT2_mode, RFSpec("mode", ft2_features, ""),
        rf_NT1_purp, RFSpec("trippurp", nt1_features, ""),
        rf_NT1_tod,  spec_NT1_tod,
        rf_NT2_mode, RFSpec("mode", nt2_features, ""),
    )

    # ---- Outputs ----
    diary_path = os.path.join(outdir, "diary_compiled.csv")
    diary.to_csv(diary_path, index=False)

    total_trips = len(diary)
    total_households = int(tgm_df[tgm_hid].nunique()) if tgm_hid and tgm_hid in tgm_df.columns else len(tgm_df)
    trip_rate = float(total_trips) / float(total_households) if total_households > 0 else float("nan")

    p_counts = diary["purpose"].value_counts().reindex(DESIRED_PURPOSES, fill_value=0)
    purpose_df = pd.DataFrame({"purpose": p_counts.index, "count": p_counts.values})
    purpose_df["share"] = purpose_df["count"] / max(1, purpose_df["count"].sum())
    purpose_df.to_csv(os.path.join(outdir, "purpose_distribution.csv"), index=False)

    m_counts = diary["mode"].value_counts().reindex(DESIRED_MODES, fill_value=0)
    mode_df = pd.DataFrame({"mode": m_counts.index, "count": m_counts.values})
    mode_df["share"] = mode_df["count"] / max(1, mode_df["count"].sum())
    mode_df.to_csv(os.path.join(outdir, "mode_distribution.csv"), index=False)

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "trip_rate": trip_rate,
            "total_trips": total_trips,
            "total_households": total_households,
            "models": {
                "TGM": os.path.join(outdir, "rf_TGM.joblib"),
                "FTAM1_purpose": os.path.join(outdir, "rf_FTAM1_purpose.joblib"),
                "FTAM1_time": os.path.join(outdir, "rf_FTAM1_timeofday.joblib") if spec_FT1_tod else None,
                "FTAM2_mode": os.path.join(outdir, "rf_FTAM2_mode.joblib"),
                "NTAM1_purpose": os.path.join(outdir, "rf_NTAM1_purpose.joblib"),
                "NTAM1_time": os.path.join(outdir, "rf_NTAM1_timeofday.joblib") if spec_NT1_tod else None,
                "NTAM2_mode": os.path.join(outdir, "rf_NTAM2_mode.joblib"),
            }
        }, f, indent=2)

    print(json.dumps({
        "outputs": {
            "diary_compiled": diary_path,
            "purpose_distribution": os.path.join(outdir, "purpose_distribution.csv"),
            "mode_distribution": os.path.join(outdir, "mode_distribution.csv"),
            "summary": os.path.join(outdir, "summary.json"),
        },
        "trip_rate": trip_rate,
        "total_trips": total_trips,
        "total_households": total_households
    }, indent=2))


# ============================ CLI ============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RF framework with Stops-based PURPOSE & MODE relabel (time untouched)")
    ap.add_argument("--tgm", required=True)
    ap.add_argument("--ftam1", required=True)
    ap.add_argument("--ftam2", required=True)
    ap.add_argument("--ntam1", required=True)
    ap.add_argument("--ntam2", required=True)
    ap.add_argument("--stops_vista", required=True, help="Path to Stop_VISTA.csv (must have destpurp1, mainmode)")
    ap.add_argument("--outdir", default="./rf_results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_all_and_simulate(
        tgm_csv=args.tgm,
        ftam1_csv=args.ftam1,
        ftam2_csv=args.ftam2,
        ntam1_csv=args.ntam1,
        ntam2_csv=args.ntam2,
        stops_vista_csv=args.stops_vista,
        outdir=args.outdir,
        random_state=args.seed,
    )


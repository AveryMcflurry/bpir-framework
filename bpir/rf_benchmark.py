"""Random-Forest travel-diary emulator (model-replacing benchmark).

Chained system after Ghasri et al. (2017): a trip-generation model (TGM)
classifies each person's integer daily trip count; the first trip is assigned
by a two-stage first-trip model (FTAM1 purpose, then FTAM2 mode given
purpose); every subsequent trip is assigned by a next-trip model (NTAM1
purpose, NTAM2 mode) conditioned on the previous trip's purpose and mode.
Layers are evaluated in batch over everyone with trips remaining until all
predicted counts are exhausted. All sub-models are Random-Forest classifiers
behind a shared preprocessing pipeline, with out-of-bag accuracy as the
training metric.
"""
import numpy as np
import pandas as pd


def build_rf_pipeline(features, categorical, n_estimators, min_samples_leaf, seed=0):
    """Preprocessing plus classifier: categoricals are constant-imputed and
    one-hot encoded, numerics constant-imputed; the forest uses sqrt(p)
    features per split, bootstrap sampling and out-of-bag scoring."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    numeric = [c for c in features if c not in categorical]
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))]), list(categorical)),
        ("num", SimpleImputer(strategy="constant", fill_value=-1), numeric),
    ])
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, max_features="sqrt",
        min_samples_leaf=min_samples_leaf, bootstrap=True, oob_score=True,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])


def train_rf(df, target, features, categorical, **rf_kwargs):
    """Fit one sub-model on the labelled survey table; the target is treated
    as a class label (including the integer trip count of the TGM)."""
    d = df.dropna(subset=[target])
    pipe = build_rf_pipeline(features, categorical, **rf_kwargs)
    pipe.fit(d[features], d[target].astype(str))
    return pipe


def _frame(base, features):
    """Feature frame in model order; columns the state lacks become NaN and
    are handled by the pipeline's imputers."""
    X = pd.DataFrame(index=base.index)
    for c in features:
        X[c] = base[c] if c in base.columns else np.nan
    return X[features]


def simulate_diaries(persons, tgm, ftam1, ftam2, ntam1, ntam2,
                     tgm_features, ftam1_features, ftam2_features,
                     ntam1_features, ntam2_features):
    """Generate a synthetic diary for every person, one trip layer at a time.

    TGM predicts the trip count; trip 1 comes from FTAM1 (purpose) and FTAM2
    (mode given the predicted purpose); trips 2..N come from NTAM1/NTAM2 with
    the previous trip's purpose and mode carried in the state columns
    lasttrippurp / lastmode. Returns the long trip table."""
    P = persons.copy()
    P["pred_trips"] = tgm.predict(_frame(P, tgm_features)).astype(int)
    state = P[P["pred_trips"] >= 1].copy()
    if state.empty:
        return pd.DataFrame(columns=["person", "trip_index", "purpose", "mode"])

    purpose = ftam1.predict(_frame(state, ftam1_features))
    mode = ftam2.predict(_frame(state.assign(trippurp=purpose), ftam2_features))
    rows = [pd.DataFrame({"person": state.index, "trip_index": 1,
                          "purpose": purpose, "mode": mode})]
    state = state.assign(remaining=state["pred_trips"] - 1,
                         lasttrippurp=purpose, lastmode=mode)

    layer = 1
    while True:
        state = state[state["remaining"] > 0]
        if state.empty:
            break
        layer += 1
        purpose = ntam1.predict(_frame(state, ntam1_features))
        mode = ntam2.predict(_frame(state.assign(trippurp=purpose), ntam2_features))
        rows.append(pd.DataFrame({"person": state.index, "trip_index": layer,
                                  "purpose": purpose, "mode": mode}))
        state = state.assign(remaining=state["remaining"] - 1,
                             lasttrippurp=purpose, lastmode=mode)
    return pd.concat(rows, ignore_index=True)

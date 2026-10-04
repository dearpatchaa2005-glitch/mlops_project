"""Single source of truth for feature preparation.

Both the training script (src/training/train_tabular_baseline.py) and the
serving predictor (src/serving/predictor.py) import the constants and
functions in this module instead of redefining them. This is what prevents
training-serving skew: there is exactly one place that decides which columns
are features, which are categorical, and how a raw metadata row turns into
the matrix LightGBM sees -- at train time and at request time alike.

Design notes
-------------
- Column lists (CAT_COLS / DROP_COLS / NORM_COLS) are the contract. Changing
  them changes both training and serving at once, by construction.
- Category *values* are a different story: at train time we let pandas infer
  categories from the training data (`fit_categorical_dtypes`). At serve
  time we must NOT re-infer from a single request -- there's no "all
  categories" to see in one row. Instead we reuse the categories the trained
  model already captured in `booster.pandas_categorical`
  (`apply_fixed_categorical_dtypes`). An unseen category at serving time
  becomes NaN (LightGBM treats it as missing) rather than silently being
  assigned a new, meaningless category code.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

# Categorical feature columns (same three columns, same order, everywhere).
CAT_COLS: list[str] = ["sex", "anatom_site_general", "tbp_tile_type"]

# Columns present in the raw metadata that are never model features:
# identifiers, free-text/location fields not used, and a column excluded on
# purpose (tbp_lv_nevi_confidence comes from an external, unverified model --
# see docs/experiment_log.md "Findings").
DROP_COLS: list[str] = [
    "isic_id",
    "patient_id",
    "image_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "tbp_lv_nevi_confidence",
]

# Columns that participate in the optional per-patient normalization
# experiment (USE_PNORM). Not used by the current production model (see
# experiment_log.md: "+0.004 to +0.006 pAUC, below noise; not adopted --
# requires many lesions per patient at inference time"), but kept here so
# that if a future model *does* adopt it, training and serving read the same
# definition rather than two copies that can drift apart.
NORM_COLS: list[str] = [
    "tbp_lv_L", "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H",
    "tbp_lv_areaMM2", "clin_size_long_diam_mm", "tbp_lv_norm_color",
    "tbp_lv_norm_border", "tbp_lv_deltaLBnorm", "tbp_lv_color_std_mean",
    "tbp_lv_perimeterMM", "tbp_lv_minorAxisMM", "tbp_lv_stdL",
]

LABEL_COL = "malignant"
SPLIT_COLS = ["split", "fold"]


def add_patient_normalized_features(df: pd.DataFrame, use_pnorm: bool) -> pd.DataFrame:
    """Add <col>_pnorm and patient_n_lesions columns, or no-op.

    Mirrors the exact transform used in the pnorm experiment runs so results
    stay reproducible if this is ever turned back on. Not used by the
    production model (use_pnorm=False in both train_tabular_baseline.py and
    at serving time, since the predictor never calls this at all).
    """
    if not use_pnorm:
        return df
    out = df.copy()
    grp = out.groupby("patient_id")
    for c in NORM_COLS:
        out[c + "_pnorm"] = out[c] - grp[c].transform("mean")
    out["patient_n_lesions"] = grp["isic_id"].transform("count")
    return out


def get_feature_columns(df: pd.DataFrame, extra_drop: Iterable[str] = ()) -> list[str]:
    """Every column that is a model feature: everything except identifiers,
    the label, the split bookkeeping columns, and anything explicitly
    dropped. Column *order* follows df.columns, same as the original
    training script -- this matters because LightGBM's saved Booster
    records feature order, and the predictor must reproduce it exactly.
    """
    exclude = set(DROP_COLS) | {LABEL_COL} | set(SPLIT_COLS) | set(extra_drop)
    return [c for c in df.columns if c not in exclude]


def fit_categorical_dtypes(df: pd.DataFrame, cat_cols: Sequence[str] = CAT_COLS) -> pd.DataFrame:
    """Training-time categorical casting: categories are inferred from the
    data actually being fit on (the whole trainval frame, before the fold
    split), exactly as the original script did with `.astype("category")`.
    """
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out


def apply_fixed_categorical_dtypes(
    df: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Serving-time categorical casting: categories are NOT re-inferred from
    the request (a single row, or a small batch, can't show "all possible
    categories"). Instead we reuse the fixed category list the trained
    model already saw (booster.pandas_categorical), via
    `categories_from_booster` below. A value outside that fixed list becomes
    NaN, which LightGBM handles as a missing value rather than a silently
    wrong category code.
    """
    out = df.copy()
    for col, cats in categories.items():
        cats = list(cats)
        # Replace anything outside the fixed category list with NaN *before*
        # constructing the Categorical, rather than passing raw values with
        # out-of-list entries straight to pd.Categorical (deprecated in
        # pandas >=4 and slated to raise instead of warn).
        safe_values = out[col].where(out[col].isin(cats))
        out[col] = pd.Categorical(safe_values, categories=cats)
    return out


def categories_from_booster(booster, cat_cols: Sequence[str] = CAT_COLS) -> dict[str, list[str]]:
    """Extract the training-time category lists a saved LightGBM Booster
    remembers, keyed by column name, in the same order CAT_COLS was passed
    in at train time (LightGBM stores them positionally, not by name).
    """
    cats = booster.pandas_categorical or []
    return dict(zip(cat_cols, cats))


def prepare_for_inference(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Turn a raw metadata-shaped DataFrame (one or many rows) into exactly
    the matrix a booster trained with `feature_names`/`categories` expects.
    Used by the serving predictor; mirrors (does not duplicate) the training
    path's column selection + categorical casting.
    """
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    x = df[list(feature_names)].copy()
    x = apply_fixed_categorical_dtypes(x, categories)
    return x

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.preprocessing.features import (
    DROP_COLS,
    LABEL_COL,
    apply_fixed_categorical_dtypes,
    fit_categorical_dtypes,
    get_feature_columns,
)


def test_get_feature_columns_excludes_ids_label_and_split_bookkeeping():
    df = pd.DataFrame({
        "isic_id": ["a"], "patient_id": ["p"], "age_approx": [10.0],
        LABEL_COL: [0], "split": ["trainval"], "fold": [0],
    })
    features = get_feature_columns(df)
    assert "age_approx" in features
    for excluded in ["isic_id", "patient_id", LABEL_COL, "split", "fold"]:
        assert excluded not in features


def test_get_feature_columns_excludes_drop_cols():
    df = pd.DataFrame({c: [1] for c in DROP_COLS} | {"age_approx": [1.0]})
    features = get_feature_columns(df)
    assert features == ["age_approx"]


def test_fit_categorical_dtypes_infers_categories_from_data():
    df = pd.DataFrame({"sex": ["female", "male", "female"]})
    out = fit_categorical_dtypes(df, ["sex"])
    assert str(out["sex"].dtype) == "category"
    assert set(out["sex"].cat.categories) == {"female", "male"}


def test_apply_fixed_categorical_dtypes_maps_unseen_value_to_nan():
    """The exact behavior that prevents a serving-time surprise: a category
    never seen at training time must become missing, not a new code."""
    df = pd.DataFrame({"sex": ["female", "something_new"]})
    out = apply_fixed_categorical_dtypes(df, {"sex": ["female", "male"]})
    assert out["sex"].iloc[0] == "female"
    assert pd.isna(out["sex"].iloc[1])

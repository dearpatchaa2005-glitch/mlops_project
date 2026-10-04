"""Data schema for the ISIC 2024 metadata table.

Two schemas, both built from pandera so a violation raises a structured,
inspectable error (column, check, failing rows) instead of a model crashing
three steps later with a cryptic dtype error:

- `metadata_schema`   : the full raw metadata.csv used for training/eval.
                         Strict about columns the model depends on; used by
                         src/validation/validate_metadata.py, which is the
                         pipeline step that is supposed to stop everything
                         and alert when the data is bad.
- `serving_request_schema` : just the columns a single /predict request
                         needs (the model's feature columns, no label, no
                         split bookkeeping). Used by serving/app.py to
                         reject malformed requests with a clear 4xx instead
                         of letting bad input reach the model.

Ranges below are deliberately generous (they exist to catch corruption --
wrong units, swapped columns, garbage strings, impossible values -- not to
encode precise biological bounds we don't have ground truth for). Tightening
them is a reasonable follow-up once the whole team has looked at the real
value distributions in EDA.
"""
from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

from src.preprocessing.features import DROP_COLS, LABEL_COL

SEX_VALUES = ["female", "male"]
ANATOM_SITE_VALUES = [
    "anterior torso", "head/neck", "lower extremity",
    "posterior torso", "upper extremity",
]
TILE_TYPE_VALUES = ["3D: XP", "3D: white"]

# float feature columns and the range check used to catch obvious
# corruption (NaN/inf, wildly out-of-domain values, wrong dtype).
_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "age_approx": (0, 120),
    "clin_size_long_diam_mm": (0, 200),
    "tbp_lv_A": (-200, 200), "tbp_lv_Aext": (-200, 200),
    "tbp_lv_B": (-200, 200), "tbp_lv_Bext": (-200, 200),
    "tbp_lv_C": (0, 300), "tbp_lv_Cext": (0, 300),
    "tbp_lv_H": (-360, 360), "tbp_lv_Hext": (-360, 360),
    "tbp_lv_L": (-200, 200), "tbp_lv_Lext": (-200, 200),
    "tbp_lv_areaMM2": (0, 100_000),
    "tbp_lv_area_perim_ratio": (0, 1000),
    "tbp_lv_color_std_mean": (0, 1000),
    "tbp_lv_deltaA": (-500, 500),
    "tbp_lv_deltaB": (-500, 500),
    "tbp_lv_deltaL": (-500, 500),
    "tbp_lv_deltaLB": (-500, 500),
    "tbp_lv_deltaLBnorm": (-500, 500),
    "tbp_lv_eccentricity": (0, 1),
    "tbp_lv_minorAxisMM": (0, 1000),
    "tbp_lv_norm_border": (0, 1000),
    "tbp_lv_norm_color": (0, 1000),
    "tbp_lv_nevi_confidence": (0, 100),
    "tbp_lv_perimeterMM": (0, 5000),
    "tbp_lv_radial_color_std_max": (0, 1000),
    "tbp_lv_stdL": (0, 1000),
    "tbp_lv_stdLExt": (0, 1000),
    "tbp_lv_symm_2axis": (0, 1),
    "tbp_lv_symm_2axis_angle": (0, 360),
    "tbp_lv_x": (-2000, 2000),
    "tbp_lv_y": (-2000, 2000),
    "tbp_lv_z": (-2000, 2000),
}


def _float_column(lo: float, hi: float, nullable: bool = True) -> Column:
    return Column(
        float,
        checks=Check.in_range(lo, hi, include_min=True, include_max=True),
        nullable=nullable,
        coerce=True,
    )


_metadata_columns: dict[str, Column] = {
    "isic_id": Column(str, checks=Check.str_matches(r"^ISIC_\d+$"), unique=True, nullable=False),
    "patient_id": Column(str, checks=Check.str_matches(r"^IP_\d+$"), nullable=False),
    "image_type": Column(str, nullable=True, required=False),
    "tbp_lv_location": Column(str, nullable=True, required=False),
    "tbp_lv_location_simple": Column(str, nullable=True, required=False),
    "sex": Column(str, checks=Check.isin(SEX_VALUES), nullable=True),
    "anatom_site_general": Column(str, checks=Check.isin(ANATOM_SITE_VALUES), nullable=True),
    "tbp_tile_type": Column(str, checks=Check.isin(TILE_TYPE_VALUES), nullable=False),
}
for _name, (_lo, _hi) in _FLOAT_RANGES.items():
    _metadata_columns[_name] = _float_column(_lo, _hi)

# The label is required for the training table, optional for a generic
# metadata extract (e.g. a monitoring sample scored without ground truth).
_metadata_columns[LABEL_COL] = Column(
    int, checks=Check.isin([0, 1]), nullable=False, coerce=True, required=False,
)

metadata_schema = DataFrameSchema(
    _metadata_columns,
    strict=False,  # extra columns (e.g. split/fold added later) are fine
    coerce=False,
)

# Serving-time schema: exactly the model's feature columns, no label, no
# ids required. Built from the same column definitions so a tightened range
# above automatically tightens request validation too.
_serving_feature_names = [
    c for c in _metadata_columns
    if c not in DROP_COLS and c not in (LABEL_COL,)
]
serving_request_schema = DataFrameSchema(
    {name: _metadata_columns[name] for name in _serving_feature_names},
    strict=False,
    coerce=False,
)


def feature_columns() -> list[str]:
    return list(_serving_feature_names)

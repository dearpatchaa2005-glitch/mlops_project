"""Monitoring: data drift (feature distributions) and concept drift
(model performance), computed separately because they need different
inputs and mean different things:

- Data drift  = the INPUT distribution changed (e.g. a new imaging device,
  a new clinic sending different patient demographics). Detectable
  immediately, no ground truth needed. Measured here with PSI (Population
  Stability Index) for numeric features and a proportion-difference check
  for categoricals, against the training distribution.
- Concept drift = the relationship between inputs and the true label
  changed (the model is systematically wrong in a new way). Only
  detectable once ground truth catches up -- for this project that means
  waiting for a biopsy/follow-up result, which is inherently delayed. This
  script computes it only when a labeled "recent" window is supplied;
  otherwise it reports that there isn't enough labeled data yet rather
  than silently assuming no drift.

Usage
-----
    # data drift: compare live-logged requests against the training set
    python -m src.monitoring.drift data-drift \
        --reference data/raw/images/metadata.csv \
        --live runtime/logs/predictions.jsonl

    # concept drift: compare recent labeled outcomes against the
    # reference CV performance recorded in MLflow
    python -m src.monitoring.drift concept-drift \
        --recent-labeled path/to/recent_outcomes.csv --run-id <run_id>

Both subcommands write runtime/monitoring/drift_status.json (merging with
whatever the other subcommand last wrote), which serving/app.py reads and
republishes as Prometheus gauges on /metrics -- so Prometheus/Alertmanager
alert on the *same* numbers this script prints, not a second copy of the
logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.preprocessing.features import CAT_COLS, LABEL_COL

STATUS_PATH = Path("runtime/monitoring/drift_status.json")

# PSI thresholds, the usual rules of thumb: <0.1 no meaningful shift,
# 0.1-0.2 moderate (watch), >0.2 significant (alert).
PSI_WATCH = 0.1
PSI_ALERT = 0.2
CATEGORY_PROPORTION_ALERT = 0.15  # absolute proportion-point shift

NUMERIC_FEATURES = [
    "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext",
    "tbp_lv_B", "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H",
    "tbp_lv_Hext", "tbp_lv_L", "tbp_lv_Lext", "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean", "tbp_lv_deltaA",
    "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity", "tbp_lv_minorAxisMM", "tbp_lv_norm_border",
    "tbp_lv_norm_color", "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL", "tbp_lv_stdLExt", "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z",
]


def _read_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def _write_status(update: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status = _read_status()
    status.update(update)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def psi(reference: pd.Series, live: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples, binned on
    the reference distribution's quantiles so each reference bin starts
    with ~equal mass."""
    ref = reference.dropna().to_numpy()
    liv = live.dropna().to_numpy()
    if len(ref) < 20 or len(liv) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # reference has (almost) no spread; nothing meaningful to compare
    ref_counts, _ = np.histogram(ref, bins=edges)
    liv_counts, _ = np.histogram(liv, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    liv_pct = np.clip(liv_counts / liv_counts.sum(), 1e-6, None)
    return float(np.sum((liv_pct - ref_pct) * np.log(liv_pct / ref_pct)))


def category_shift(reference: pd.Series, live: pd.Series) -> float:
    """Max absolute proportion-point difference across categories."""
    ref_p = reference.value_counts(normalize=True, dropna=False)
    liv_p = live.value_counts(normalize=True, dropna=False)
    all_cats = set(ref_p.index) | set(liv_p.index)
    return max(abs(ref_p.get(c, 0.0) - liv_p.get(c, 0.0)) for c in all_cats)


def load_live_predictions(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def data_drift(reference_path: Path, live_path: Path) -> dict:
    ref = pd.read_csv(reference_path, low_memory=False)
    if "split" in ref.columns:
        ref = ref[ref["split"] == "trainval"]
    live = load_live_predictions(live_path)

    numeric_results = {}
    for col in NUMERIC_FEATURES:
        if col not in ref.columns or col not in live.columns:
            continue
        value = psi(ref[col], live[col])
        numeric_results[col] = value

    categorical_results = {}
    for col in CAT_COLS:
        if col not in ref.columns or col not in live.columns:
            continue
        categorical_results[col] = category_shift(ref[col], live[col])

    flagged = [c for c, v in numeric_results.items() if not np.isnan(v) and v >= PSI_ALERT]
    flagged += [c for c, v in categorical_results.items() if v >= CATEGORY_PROPORTION_ALERT]
    watch = [c for c, v in numeric_results.items() if not np.isnan(v) and PSI_WATCH <= v < PSI_ALERT]

    result = {
        "n_live_rows": len(live),
        "numeric_psi": {k: round(v, 4) for k, v in numeric_results.items() if not np.isnan(v)},
        "categorical_shift": {k: round(v, 4) for k, v in categorical_results.items()},
        "flagged_features": flagged,
        "watch_features": watch,
        "is_drift": len(flagged) > 0,
    }
    _write_status({"data_drift": result})
    return result


def concept_drift(recent_labeled_path: Path, run_id: str, reference_pauc: float | None = None) -> dict:
    """Recompute pAUC on a recently-labeled window and compare to the
    reference (CV) pAUC for `run_id`. Requires predictor + the same feature
    prep as training -- reuses src.serving.predictor so this never risks
    comparing "recent pAUC" computed a different way than "reference pAUC".
    """
    from src.evaluation.metrics import partial_auc
    from src.serving.predictor import Predictor

    if not recent_labeled_path.exists():
        result = {
            "status": "insufficient_labeled_data",
            "note": "No recently-labeled outcomes available yet (ground truth "
                    "for this task lags behind prediction -- see docs/experiment_log.md). "
                    "Not flagging drift in the absence of evidence.",
            "is_drift": False,
        }
        _write_status({"concept_drift": result})
        return result

    recent = pd.read_csv(recent_labeled_path, low_memory=False)
    predictor = Predictor()
    scores = predictor.predict(recent)
    recent_pauc = partial_auc(recent[LABEL_COL], scores)
    if reference_pauc is None:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        run = mlflow.get_run(run_id)
        reference_pauc = float(run.data.metrics.get("pauc_mean", float("nan")))

    drop = reference_pauc - recent_pauc
    result = {
        "status": "ok",
        "n_recent_rows": len(recent),
        "n_recent_malignant": int(recent[LABEL_COL].sum()),
        "reference_pauc": round(reference_pauc, 4),
        "recent_pauc": round(recent_pauc, 4),
        "pauc_drop": round(drop, 4),
        "is_drift": drop > 0.03,  # same magnitude as the fairness gate in configs/gates.yaml
    }
    _write_status({"concept_drift": result})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data-drift")
    p_data.add_argument("--reference", type=Path, default=Path("data/raw/images/metadata.csv"))
    p_data.add_argument("--live", type=Path, default=Path("runtime/logs/predictions.jsonl"))

    p_concept = sub.add_parser("concept-drift")
    p_concept.add_argument("--recent-labeled", type=Path, default=Path("does_not_exist.csv"))
    p_concept.add_argument("--run-id", default="16b43986f5704e0fafbef7cdcf7149b8")

    args = parser.parse_args()
    if args.command == "data-drift":
        out = data_drift(args.reference, args.live)
    else:
        out = concept_drift(args.recent_labeled, args.run_id)
    print(json.dumps(out, indent=2))
    sys.exit(1 if out.get("is_drift") else 0)

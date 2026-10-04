import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.run_context import RUN_ID  # noqa: E402

OUT_DIR = Path("artifacts/evaluation") / RUN_ID
N_BOOTSTRAPS = 1000
SEED = 42


def partial_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(1 - y_true, -y_score)
    max_fpr = 1 - 0.80
    stop = np.searchsorted(fpr, max_fpr, side="right")
    boundary = np.interp(
        max_fpr,
        fpr[stop - 1:stop + 1],
        tpr[stop - 1:stop + 1],
    )
    return float(np.trapezoid(
        np.append(tpr[:stop], boundary),
        np.append(fpr[:stop], max_fpr),
    ))


def calculate_metrics(y, scores):
    return {
        "pauc": partial_auc(y, scores),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


df = pd.read_csv(OUT_DIR / "test_predictions.csv")

required = ["isic_id", "patient_id", "malignant", "prediction_score"]
if df[required].isna().any().any():
    raise ValueError("พบค่าว่างในข้อมูล")
if df["isic_id"].duplicated().any():
    raise ValueError("พบรหัสภาพซ้ำ")
if not df["malignant"].isin([0, 1]).all():
    raise ValueError("label ต้องเป็น 0 หรือ 1")
if df["malignant"].nunique() != 2:
    raise ValueError("ต้องมีข้อมูลทั้งสองกลุ่ม")
if not np.isfinite(df["prediction_score"]).all():
    raise ValueError("คะแนนทำนายไม่ถูกต้อง")

y = df["malignant"].to_numpy(dtype=int)
scores = df["prediction_score"].to_numpy(dtype=float)

# แต่ละรายการเก็บตำแหน่งภาพทั้งหมดของผู้ป่วยหนึ่งคน
patient_groups = list(df.groupby("patient_id").indices.values())
n_patients = len(patient_groups)

point_estimates = calculate_metrics(y, scores)
bootstrap_scores = {name: [] for name in point_estimates}
rng = np.random.default_rng(SEED)
skipped = 0

print("patients:", n_patients)
print("bootstrap rounds:", N_BOOTSTRAPS)

for round_number in range(N_BOOTSTRAPS):
    # สุ่มผู้ป่วยแบบใส่คืน จำนวนเท่ากับผู้ป่วยเดิม
    selected = rng.integers(0, n_patients, size=n_patients)
    indices = np.concatenate([patient_groups[i] for i in selected])

    y_sample = y[indices]
    score_sample = scores[indices]

    # หากรอบใดเหลือเพียงคลาสเดียว จะคำนวณ ROC-AUC ไม่ได้
    if np.unique(y_sample).size < 2:
        skipped += 1
    else:
        result = calculate_metrics(y_sample, score_sample)
        for name, value in result.items():
            bootstrap_scores[name].append(value)

    if (round_number + 1) % 100 == 0:
        print(f"completed: {round_number + 1}/{N_BOOTSTRAPS}")

valid_rounds = N_BOOTSTRAPS - skipped
if valid_rounds < 900:
    raise RuntimeError("รอบที่คำนวณได้มีน้อยเกินไป ต้องตรวจข้อมูลก่อน")

summary = {
    "run_id": RUN_ID,
    "method": "patient_cluster_bootstrap_percentile",
    "confidence_level": 0.95,
    "seed": SEED,
    "requested_rounds": N_BOOTSTRAPS,
    "valid_rounds": valid_rounds,
    "skipped_rounds": skipped,
    "patients": n_patients,
    "metrics": {},
}

print("\n95% confidence intervals:")
for name, values in bootstrap_scores.items():
    low, high = np.percentile(values, [2.5, 97.5])
    summary["metrics"][name] = {
        "estimate": point_estimates[name],
        "ci_lower": float(low),
        "ci_upper": float(high),
    }
    print(
        f"{name}: {point_estimates[name]:.4f} "
        f"(95% CI: {low:.4f} - {high:.4f})"
    )

output_path = OUT_DIR / "test_confidence_intervals.json"
output_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_artifact(str(output_path), artifact_path="evaluation")

print("valid rounds:", valid_rounds)
print("skipped rounds:", skipped)
print("saved:", output_path)
print("Confidence interval report saved to MLflow")
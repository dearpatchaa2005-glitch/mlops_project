import json
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.run_context import RUN_ID  # noqa: E402

MODEL_DIR = Path("artifacts/models") / RUN_ID
OUT_DIR = Path("artifacts/evaluation") / RUN_ID

CAT_COLS = ["sex", "anatom_site_general", "tbp_tile_type"]


def partial_auc(y_true, y_score, min_tpr=0.80):
    # ใช้สูตรเดียวกับตอนประเมิน cross-validation
    v_gt = 1 - np.asarray(y_true)
    v_pred = -np.asarray(y_score)
    max_fpr = 1 - min_tpr

    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    tpr_at_limit = np.interp(
        max_fpr,
        fpr[stop - 1:stop + 1],
        tpr[stop - 1:stop + 1],
    )
    fpr = np.append(fpr[:stop], max_fpr)
    tpr = np.append(tpr[:stop], tpr_at_limit)
    return float(np.trapezoid(tpr, fpr))


# โหลดโมเดลที่เลือกไว้ครบทั้ง 5 folds
models = [
    lgb.Booster(model_file=str(MODEL_DIR / f"fold_{i}.txt"))
    for i in range(5)
]
features = models[0].feature_name()

if any(m.feature_name() != features for m in models):
    raise ValueError("รายชื่อหรือลำดับ features ของโมเดลไม่ตรงกัน")

# อ่านเฉพาะรายการที่กำหนดเป็น test ไว้แล้ว
splits = pd.read_csv("data/splits/splits.csv")
test_rows = splits.loc[
    splits["split"] == "test", ["isic_id", "malignant"]
].copy()

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
test = test_rows.merge(
    meta,
    on="isic_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not test["_merge"].eq("both").all():
    raise ValueError("มีภาพใน test ที่ไม่พบ metadata")

for col in CAT_COLS:
    test[col] = test[col].astype("category")

# จัดลำดับคอลัมน์ตามโมเดลที่บันทึกไว้
X_test = test[features]
y_test = test["malignant"].astype(int)

if set(y_test.unique()) != {0, 1}:
    raise ValueError("test ต้องมี label ทั้ง 0 และ 1 เท่านั้น")

# เฉลี่ยคะแนนจากโมเดลทั้ง 5
pred = np.mean(
    [model.predict(X_test) for model in models],
    axis=0,
)

if not np.isfinite(pred).all():
    raise ValueError("พบคะแนนทำนายที่ไม่ใช่ค่าจำนวนจริงที่ใช้ได้")

metrics = {
    "run_id": RUN_ID,
    "method": "mean_of_5_fold_models",
    "test_rows": len(test),
    "test_positive": int(y_test.sum()),
    "test_pauc": partial_auc(y_test, pred),
    "test_roc_auc": float(roc_auc_score(y_test, pred)),
    "test_average_precision": float(average_precision_score(y_test, pred)),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# บันทึกคะแนนทำนายรายภาพ
predictions = test[
    ["isic_id", "patient_id", "malignant"]
].copy()
predictions["prediction_score"] = pred

predictions_path = OUT_DIR / "test_predictions.csv"
predictions.to_csv(predictions_path, index=False)

# บันทึกคะแนนสรุป
output_path = OUT_DIR / "test_metrics.json"
output_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("run:", RUN_ID)
print("test rows:", metrics["test_rows"])
print("malignant:", metrics["test_positive"])
print(f"test pAUC: {metrics['test_pauc']:.4f}")
print(f"test ROC-AUC: {metrics['test_roc_auc']:.4f}")
print(f"test Average Precision: {metrics['test_average_precision']:.4f}")
print("saved:", output_path)
print("prediction rows:", len(predictions))
print("predictions saved:", predictions_path)

# แนบคะแนนและไฟล์ทั้งสองเข้า MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metrics({
        "test_pauc": metrics["test_pauc"],
        "test_roc_auc": metrics["test_roc_auc"],
        "test_average_precision": metrics["test_average_precision"],
    })
    mlflow.set_tag("test_evaluation_method", metrics["method"])
    mlflow.log_artifact(str(output_path), artifact_path="evaluation")
    mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")

print("Test metrics and predictions saved to MLflow")
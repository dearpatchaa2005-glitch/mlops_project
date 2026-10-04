import json
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.run_context import RUN_ID  # noqa: E402

MODEL_DIR = Path("artifacts/models") / RUN_ID
OUT_DIR = Path("artifacts/evaluation") / RUN_ID

N_WARMUP = 20
N_REQUESTS = 500
NUM_THREADS = 1
SEED = 42

CAT_COLS = ["sex", "anatom_site_general", "tbp_tile_type"]

# โหลดโมเดลก่อนเริ่มจับเวลา
models = [
    lgb.Booster(model_file=str(MODEL_DIR / f"fold_{i}.txt"))
    for i in range(5)
]
features = models[0].feature_name()

if any(model.feature_name() != features for model in models):
    raise ValueError("features ของโมเดลไม่ตรงกัน")

# เตรียมตัวอย่างจาก trainval โดยไม่ต้องใช้ label หรือ test
splits = pd.read_csv("data/splits/splits.csv")
sample_ids = (
    splits.loc[splits["split"] == "trainval", ["isic_id"]]
    .sample(n=N_REQUESTS, random_state=SEED)
)

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
sample = sample_ids.merge(
    meta,
    on="isic_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not sample["_merge"].eq("both").all():
    raise ValueError("ไม่พบ metadata ของตัวอย่างบางรายการ")

for col in CAT_COLS:
    sample[col] = sample[col].astype("category")

X = sample[features]
requests = [X.iloc[[i]].copy() for i in range(N_REQUESTS)]


def predict_one(row):
    scores = [
        model.predict(row, num_threads=NUM_THREADS)[0]
        for model in models
    ]
    return float(np.mean(scores))


# อุ่นเครื่องก่อนวัด เพื่อไม่รวมต้นทุนของการเรียกครั้งแรก
print("warming up...")
for i in range(N_WARMUP):
    predict_one(requests[i % N_REQUESTS])

# จำลองคำขอต่อเนื่องทีละรายการ ไม่มีคำขอพร้อมกัน
print("measuring...")
latencies_ms = []

for i, row in enumerate(requests):
    start = time.perf_counter()
    prediction = predict_one(row)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not np.isfinite(prediction):
        raise ValueError("พบคะแนนทำนายไม่ถูกต้อง")

    latencies_ms.append(elapsed_ms)

    if (i + 1) % 100 == 0:
        print(f"completed: {i + 1}/{N_REQUESTS}")

results = {
    "run_id": RUN_ID,
    "scope": "5-model ensemble prediction on prepared pandas row",
    "batch_size": 1,
    "concurrent_requests": 1,
    "num_threads_per_model": NUM_THREADS,
    "warmup_requests": N_WARMUP,
    "measured_requests": N_REQUESTS,
    "seed": SEED,
    "p50_ms": float(np.percentile(latencies_ms, 50)),
    "p95_ms": float(np.percentile(latencies_ms, 95)),
    "mean_ms": float(np.mean(latencies_ms)),
    "os": platform.platform(),
    "cpu": platform.processor(),
    "logical_cpu_count": os.cpu_count(),
    "python_version": platform.python_version(),
    "lightgbm_version": lgb.__version__,
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
report_path = OUT_DIR / "latency_report.json"
report_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

samples_path = OUT_DIR / "latency_samples.csv"
pd.DataFrame({
    "request_number": np.arange(1, N_REQUESTS + 1),
    "latency_ms": latencies_ms,
}).to_csv(samples_path, index=False)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metrics({
        "latency_p50_ms": results["p50_ms"],
        "latency_p95_ms": results["p95_ms"],
    })
    mlflow.log_artifact(str(report_path), artifact_path="evaluation")
    mlflow.log_artifact(str(samples_path), artifact_path="evaluation")

print(f"\np50: {results['p50_ms']:.3f} ms")
print(f"p95: {results['p95_ms']:.3f} ms")
print(f"mean: {results['mean_ms']:.3f} ms")
print("saved:", report_path)
print("Latency report saved to MLflow")
"""FastAPI serving for the skin-lesion tabular baseline.

    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health    liveness + which model version is loaded
GET  /metrics   Prometheus text exposition (requests, latency, scores)
GET  /slo       current rolling p50/p95 latency vs. the SLO in configs/gates.yaml
POST /predict   {"instances": [{...row...}, ...]} -> {"predictions": [...]}

Design notes
------------
- The model to load is resolved through the MLflow Model Registry
  ("production" alias of MODEL_NAME) when MLFLOW_TRACKING_URI / a registry
  entry is available, falling back to the RUN_ID baked into
  src/serving/predictor.py for local/dev use. See `load_predictor()`.
- Every request is validated against `serving_request_schema`
  (src/validation/schema.py -- the SAME schema module the data-validation
  gate uses) before it reaches the model. A bad request gets a 422 with the
  specific failing fields, not a stack trace from inside LightGBM. This is
  the "system stops and alerts on bad input" requirement, enforced at the
  API boundary instead of just at the batch-data boundary.
- Every prediction is appended to a local JSONL log
  (runtime/logs/predictions.jsonl) with features + score + timestamp. This
  is the raw feed src/monitoring/drift.py reads to compare "live" feature
  distributions against the training distribution.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pandera.errors import SchemaErrors
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, create_model

from src.serving.predictor import RUN_ID, Predictor
from src.validation.schema import feature_columns, serving_request_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

GATES_PATH = Path("configs/gates.yaml")
PRED_LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "runtime/logs/predictions.jsonl"))
MODEL_NAME = os.environ.get("MODEL_NAME", "skin_lesion_tabular")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
LATENCY_WINDOW = 500  # rolling window size for the /slo endpoint

# ---------------------------------------------------------------------------
# Model loading: try the registry's "production" alias first, fall back to
# the fixed RUN_ID baked into predictor.py for local/dev runs where the
# registry hasn't been set up yet (see README "Running the API").
# ---------------------------------------------------------------------------


def load_predictor() -> tuple[Predictor, dict]:
    info = {"source": "fixed_run_id", "run_id": RUN_ID, "model_name": MODEL_NAME}
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        model_dir = Path("artifacts/models") / mv.run_id
        if model_dir.exists():
            info = {
                "source": "registry", "run_id": mv.run_id,
                "model_name": MODEL_NAME, "version": mv.version,
            }
            return Predictor(model_dir=model_dir), info
        log.warning(
            "registry points at run %s but %s is missing locally; "
            "falling back to fixed RUN_ID", mv.run_id, model_dir,
        )
    except Exception as exc:  # noqa: BLE001 - any registry/lookup failure -> fallback
        log.info("no usable 'production' registry entry (%s); using fixed RUN_ID", exc)

    return Predictor(), info


app = FastAPI(
    title="Skin lesion risk scorer",
    description="Research demo for a course MLOps project. NOT a medical diagnostic tool.",
    version="0.1.0",
)
predictor, model_info = load_predictor()
_latencies_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "predict_requests_total", "Total /predict requests", ["status"],
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds", "End-to-end /predict latency (includes validation)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTION_SCORE = Histogram(
    "prediction_score", "Distribution of predicted malignancy scores",
    buckets=(0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
)

# Re-published from runtime/monitoring/drift_status.json, written by
# src/monitoring/drift.py. The API doesn't compute drift itself (that
# needs a reference dataset + a batch of live traffic); it just exposes
# whatever the last monitoring run found, so Prometheus/Alertmanager can
# alert on it (see infra/prometheus/alert_rules.yml).
DATA_DRIFT_MAX_PSI = Gauge("data_drift_max_psi", "Max feature PSI vs. training distribution")
DATA_DRIFT_DETECTED = Gauge("data_drift_detected", "1 if the last data-drift check flagged drift")
CONCEPT_DRIFT_PAUC_DROP = Gauge("concept_drift_pauc_drop", "reference pAUC - recent pAUC")
CONCEPT_DRIFT_DETECTED = Gauge("concept_drift_detected", "1 if the last concept-drift check flagged drift")
DRIFT_STATUS_PATH = Path("runtime/monitoring/drift_status.json")


def _refresh_drift_gauges() -> None:
    if not DRIFT_STATUS_PATH.exists():
        return
    try:
        status = json.loads(DRIFT_STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    dd = status.get("data_drift") or {}
    if dd.get("numeric_psi"):
        DATA_DRIFT_MAX_PSI.set(max(dd["numeric_psi"].values()))
    DATA_DRIFT_DETECTED.set(1 if dd.get("is_drift") else 0)
    cd = status.get("concept_drift") or {}
    if "pauc_drop" in cd:
        CONCEPT_DRIFT_PAUC_DROP.set(cd["pauc_drop"])
    CONCEPT_DRIFT_DETECTED.set(1 if cd.get("is_drift") else 0)


def _log_prediction(rows: list[dict], scores: list[float]) -> None:
    PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with PRED_LOG_PATH.open("a", encoding="utf-8") as f:
        for row, score in zip(rows, scores):
            f.write(json.dumps({"timestamp": ts, "score": score, **row}) + "\n")


# ---------------------------------------------------------------------------
# Request schema: built from the SAME column definitions as
# src/validation/schema.py, so the OpenAPI docs and the pandera validation
# can never describe two different contracts.
# ---------------------------------------------------------------------------
_pydantic_fields = {name: (Optional[float], None) if name not in (
    "sex", "anatom_site_general", "tbp_tile_type",
) else (Optional[str], None) for name in feature_columns()}
LesionRow = create_model("LesionRow", __config__=ConfigDict(extra="forbid"), **_pydantic_fields)


class PredictRequest(BaseModel):
    instances: list[LesionRow]


class PredictResponse(BaseModel):
    predictions: list[float]
    model: dict
    n_instances: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model_info,
        "n_features": len(predictor.features),
    }


@app.post("/reload")
def reload_model():
    """Re-resolve the 'production' registry alias and hot-swap the model
    in this running process -- no restart, no redeploy. This is what makes
    src/registry/rollback.py's alias flip actually take effect: call this
    right after a promote/rollback and the next /predict uses the new
    (or reverted) version.
    """
    global predictor, model_info
    old_info = model_info
    predictor, model_info = load_predictor()
    return {"previous": old_info, "current": model_info}


@app.get("/slo")
def slo():
    if not _latencies_ms:
        return {"status": "no requests served yet"}
    sorted_lat = sorted(_latencies_ms)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
    gates = yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))["gates"]
    slo_p95 = gates.get("latency_p95_ms_max")
    return {
        "window_size": len(_latencies_ms),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "slo_p95_ms": slo_p95,
        "within_slo": (p95 <= slo_p95) if slo_p95 is not None else None,
    }


@app.get("/metrics")
def metrics():
    _refresh_drift_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    start = time.perf_counter()
    rows = [r.model_dump() for r in body.instances]
    df = pd.DataFrame(rows)

    try:
        df = serving_request_schema.validate(df, lazy=True)
    except SchemaErrors as err:
        REQUEST_COUNT.labels(status="invalid").inc()
        failures = err.failure_cases[["column", "check", "index"]].to_dict(orient="records")
        raise HTTPException(status_code=422, detail={"schema_errors": failures})

    try:
        scores = predictor.predict(df).tolist()
    except ValueError as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    _latencies_ms.append(elapsed_ms)
    REQUEST_LATENCY.observe(elapsed_ms / 1000)
    for s in scores:
        PREDICTION_SCORE.observe(s)
    REQUEST_COUNT.labels(status="ok").inc()
    _log_prediction(rows, scores)

    return PredictResponse(
        predictions=scores,
        model=model_info,
        n_instances=len(scores),
    )


# ---------------------------------------------------------------------------
# Demo UI: pick a sample lesion image and see the model's risk score,
# reusing the exact same /predict logic above (so this can never drift from
# the real API contract). Not part of the graded API surface -- just a
# nicer way to show the model working during the presentation.
# ---------------------------------------------------------------------------
IMAGES_DIR = Path("data/raw/images")
GT_PATH = Path("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")
_metadata_df: pd.DataFrame | None = None


def _load_metadata() -> pd.DataFrame:
    global _metadata_df
    if _metadata_df is None:
        meta = pd.read_csv(IMAGES_DIR / "metadata.csv")
        gt = pd.read_csv(GT_PATH)
        merged = meta.merge(gt, on="isic_id", how="inner")
        merged["malignant"] = merged["malignant"].astype(int)
        _metadata_df = merged
    return _metadata_df


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return _DEMO_HTML


@app.get("/demo/samples")
def demo_samples(n: int = 12):
    df = _load_metadata()
    malignant_pool = df[df["malignant"] == 1]
    benign_pool = df[df["malignant"] == 0]
    n_mal = min(n // 2, len(malignant_pool))
    malignant = malignant_pool.sample(n_mal, random_state=42)
    benign = benign_pool.sample(n - n_mal, random_state=42)
    sample = pd.concat([malignant, benign]).sample(frac=1, random_state=42)
    return [{"isic_id": r.isic_id, "malignant_label": int(r.malignant)} for r in sample.itertuples()]


@app.get("/demo/image/{isic_id}")
def demo_image(isic_id: str):
    path = IMAGES_DIR / f"{isic_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"image not found: {path}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/demo/predict/{isic_id}")
def demo_predict(isic_id: str):
    df = _load_metadata()
    row = df[df["isic_id"] == isic_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"isic_id not found: {isic_id}")
    feats = row[feature_columns()]
    feats_clean = feats.where(pd.notnull(feats), None).to_dict(orient="records")[0]
    result = predict(PredictRequest(instances=[LesionRow(**feats_clean)]))

    display_keys = ["age_approx", "sex", "anatom_site_general", "tbp_tile_type",
                     "clin_size_long_diam_mm", "tbp_lv_nevi_confidence"]
    display = {k: feats_clean.get(k) for k in display_keys if k in feats_clean}

    return {
        "isic_id": isic_id,
        "true_label": int(row.iloc[0]["malignant"]),
        "predicted_score": result.predictions[0],
        "model": result.model,
        "features": display,
    }


_DEMO_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Skin lesion risk demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  select, button { font-size: 1rem; padding: 8px 12px; margin: 8px 0; width: 100%; }
  table { border-collapse: collapse; margin-top: 12px; width: 100%; }
  .result { margin-top: 16px; padding: 16px; border-radius: 8px; background: #f3f3f3; }
  .score { font-size: 2rem; font-weight: 700; }
  .note { color: #666; font-size: 0.85rem; margin-top: 24px; }
</style>
</head>
<body>
  <h1>Skin lesion risk scorer -- demo</h1>
  <p>เลือกตัวอย่างเคสจากชุดข้อมูล แล้วดูคะแนนความเสี่ยงที่โมเดลทำนาย (เรียก /predict ตัวจริงเบื้องหลัง)</p>
  <select id="picker"></select>
  <div id="feattable"></div>
  <div id="result"></div>
  <p class="note">Research demo for a course MLOps project. Not a medical diagnostic tool.</p>
<script>
async function loadSamples() {
  const res = await fetch('/demo/samples?n=12');
  const samples = await res.json();
  const sel = document.getElementById('picker');
  sel.innerHTML = samples.map(s => `<option value="${s.isic_id}">${s.isic_id} (${s.malignant_label ? 'malignant - dataset label' : 'benign - dataset label'})</option>`).join('');
  if (samples.length) showIsicId(samples[0].isic_id);
  sel.onchange = () => showIsicId(sel.value);
}
async function showIsicId(isicId) {
  document.getElementById('feattable').innerHTML = 'กำลังทำนาย...';
  document.getElementById('result').innerHTML = '';
  const res = await fetch(`/demo/predict/${isicId}`);
  const data = await res.json();
  const pct = (data.predicted_score * 100).toFixed(2);
  const rows = Object.entries(data.features).map(([k, v]) =>
    `<tr><td style="padding:4px 12px 4px 0; color:#666;">${k}</td><td style="padding:4px 0;"><b>${v}</b></td></tr>`
  ).join('');
  document.getElementById('feattable').innerHTML = `<table>${rows}</table>`;
  document.getElementById('result').innerHTML = `
    <div class="result">
      <div>Predicted malignancy risk:</div>
      <div class="score">${pct}%</div>
      <div>Dataset label: <b>${data.true_label ? 'malignant' : 'benign'}</b></div>
    </div>`;
}
loadSamples();
</script>
</body>
</html>
"""
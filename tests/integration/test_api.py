"""Integration test for serving/app.py, using FastAPI's TestClient (no
server process needed). Skipped automatically when the trained model
artifacts aren't present locally (e.g. on a fresh clone or CI, since
artifacts/ is gitignored) -- run this on a machine that has already run
the training pipeline at least once.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

RUN_ID = "16b43986f5704e0fafbef7cdcf7149b8"
MODEL_DIR = Path("artifacts/models") / RUN_ID

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason=f"model artifacts not present at {MODEL_DIR} (run the training "
           f"pipeline first; this is expected to skip in CI)",
)

GOOD_ROW = {
    "age_approx": 55, "sex": "female", "anatom_site_general": "upper extremity",
    "clin_size_long_diam_mm": 4.2, "tbp_tile_type": "3D: XP",
    "tbp_lv_A": 10.1, "tbp_lv_Aext": 9.8, "tbp_lv_B": 12.3, "tbp_lv_Bext": 11.9,
    "tbp_lv_C": 15.0, "tbp_lv_Cext": 14.7, "tbp_lv_H": 20.0, "tbp_lv_Hext": 19.5,
    "tbp_lv_L": 50.0, "tbp_lv_Lext": 49.2, "tbp_lv_areaMM2": 12.5,
    "tbp_lv_area_perim_ratio": 3.1, "tbp_lv_color_std_mean": 1.2,
    "tbp_lv_deltaA": 0.3, "tbp_lv_deltaB": -0.2, "tbp_lv_deltaL": 0.8,
    "tbp_lv_deltaLB": 1.1, "tbp_lv_deltaLBnorm": 2.2, "tbp_lv_eccentricity": 0.6,
    "tbp_lv_minorAxisMM": 2.0, "tbp_lv_norm_border": 1.5, "tbp_lv_norm_color": 1.8,
    "tbp_lv_perimeterMM": 14.0, "tbp_lv_radial_color_std_max": 0.9,
    "tbp_lv_stdL": 2.3, "tbp_lv_stdLExt": 2.1, "tbp_lv_symm_2axis": 0.7,
    "tbp_lv_symm_2axis_angle": 45.0, "tbp_lv_x": 10.0, "tbp_lv_y": -5.0, "tbp_lv_z": 2.0,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serving.app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_good_row_returns_score_in_unit_interval(client):
    r = client.post("/predict", json={"instances": [GOOD_ROW]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_instances"] == 1
    assert 0.0 <= body["predictions"][0] <= 1.0


def test_predict_bad_category_returns_422_with_schema_errors(client):
    bad = dict(GOOD_ROW, sex="alien")
    r = client.post("/predict", json={"instances": [bad]})
    assert r.status_code == 422
    assert "schema_errors" in r.json()["detail"]


def test_predict_batch_matches_single_row(client):
    batch = client.post("/predict", json={"instances": [GOOD_ROW, GOOD_ROW]}).json()
    single = client.post("/predict", json={"instances": [GOOD_ROW]}).json()
    assert batch["predictions"][0] == pytest.approx(single["predictions"][0])


def test_metrics_endpoint_exposes_request_counter(client):
    client.post("/predict", json={"instances": [GOOD_ROW]})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"predict_requests_total" in r.content

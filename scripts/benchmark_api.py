"""Real end-to-end latency + throughput benchmark, through the HTTP API --
not the model-only number in artifacts/evaluation/*/latency_report.json
(which docs/experiment_log.md already flags as excluding "API/network
overhead"). This script is what actually exercises that overhead: JSON
(de)serialization, pydantic validation, the pandera schema check, and the
loopback HTTP round trip.

Usage:
    uvicorn serving.app:app --port 8000 &
    python scripts/benchmark_api.py --url http://127.0.0.1:8000 \
        --n 500 --concurrency 8

Writes reports/performance/api_latency_report.json with p50/p95/throughput,
in the same shape check_gates.py-style consumers expect
(`p50_ms`, `p95_ms`, plus `throughput_rps`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROW = {
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
REPORT_PATH = Path("reports/performance/api_latency_report.json")


def one_request(url: str) -> float:
    payload = json.dumps({"instances": [ROW]}).encode()
    req = urllib.request.Request(
        f"{url}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        r.read()
    return (time.perf_counter() - t0) * 1000


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * pct), len(s) - 1)
    return s[idx]


def main(url: str, n: int, concurrency: int, warmup: int) -> dict:
    for _ in range(warmup):
        one_request(url)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = list(pool.map(lambda _: one_request(url), range(n)))
    wall_s = time.perf_counter() - t_start

    report = {
        "n_requests": n,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "mean_ms": round(statistics.mean(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "wall_time_s": round(wall_s, 3),
        "throughput_rps": round(n / wall_s, 2),
        "scope": "full HTTP round trip through /predict (1 row per request): "
                 "JSON parsing, pydantic validation, pandera schema check, "
                 "5-model predict, JSON response, loopback network.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    main(args.url, args.n, args.concurrency, args.warmup)

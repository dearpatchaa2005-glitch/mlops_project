"""Export a small, git-committed snapshot of the metrics check_gates.py
checks, so CI (GitHub Actions) can verify quality gates without needing
mlflow.db / artifacts/ / data/ -- all gitignored, none of them available
on a CI runner.

Run this locally right after training+evaluating a run (it reads exactly
what check_gates.py reads: mlflow.db, artifacts/evaluation/<run>/, the
oof predictions). Commit the resulting reports/gate_snapshot.json -- THAT
file, not the gitignored originals, is what travels to GitHub and what CI
checks against. src/evaluation/check_gates_ci.py applies the identical
threshold logic to this snapshot instead of to live MLflow data.

    python -m src.evaluation.export_gate_snapshot <run_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import pandas as pd

from src.evaluation.metrics import partial_auc

SNAPSHOT_PATH = Path("reports/gate_snapshot.json")


def export(run_id: str) -> dict:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run = mlflow.get_run(run_id)
    m = run.data.metrics

    snapshot = {
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "pauc_mean": m.get("pauc_mean"),
        "pauc_std": m.get("pauc_std"),
    }

    latency_path = Path("artifacts/evaluation") / run_id / "latency_report.json"
    if latency_path.exists():
        snapshot["model_latency_p95_ms"] = json.loads(latency_path.read_text())["p95_ms"]

    api_latency_path = Path("reports/performance/api_latency_report.json")
    if api_latency_path.exists():
        api = json.loads(api_latency_path.read_text())
        snapshot["api_latency_p95_ms"] = api["p95_ms"]
        snapshot["api_throughput_rps"] = api["throughput_rps"]

    model_dir = Path("artifacts/models") / run_id
    fold_files = list(model_dir.glob("fold_*.txt"))
    if fold_files:
        snapshot["model_size_mb"] = sum(f.stat().st_size for f in fold_files) / 1e6

    oof_path = Path("artifacts/evaluation") / run_id / "oof_predictions.csv"
    if oof_path.exists():
        meta = pd.read_csv("data/raw/images/metadata.csv",
                            usecols=["isic_id", "sex", "age_approx", "tbp_tile_type"],
                            low_memory=False)
        oof = pd.read_csv(oof_path)
        d = oof.merge(meta, on="isic_id", how="left")
        d["sex"] = d["sex"].fillna("unknown").astype(str)
        d["age_group"] = pd.cut(d["age_approx"], bins=[0, 50, 65, 90],
                                 labels=["<=50", "51-65", ">65"]).astype(object)
        d["age_group"] = d["age_group"].where(d["age_approx"].notna(), "unknown").astype(str)

        overall = partial_auc(d["malignant"], d["oof"])
        fairness = {}
        for col in ["sex", "age_group", "tbp_tile_type"]:
            for name, g in d.groupby(col):
                n_pos = int(g["malignant"].sum())
                if n_pos < 50:
                    continue
                gap = overall - partial_auc(g["malignant"], g["oof"])
                fairness[f"{col}={name}"] = {"gap": round(float(gap), 4), "n_pos": n_pos}
        snapshot["fairness_gaps"] = fairness

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH}")
    print(json.dumps(snapshot, indent=2))
    return snapshot


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.evaluation.export_gate_snapshot <run_id>")
    export(sys.argv[1])

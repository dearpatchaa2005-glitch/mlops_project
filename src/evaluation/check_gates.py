import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import yaml

from src.evaluation.metrics import partial_auc

GATES_PATH = Path("configs/gates.yaml")
EXPERIMENT = "skin_lesion_tabular_baseline"

cfg = yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))
gates = cfg["gates"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if runs.empty:
    sys.exit("no runs found")

run = runs.iloc[0]
pauc_mean = float(run["metrics.pauc_mean"])
pauc_std = float(run["metrics.pauc_std"])
print("run:", run["tags.mlflow.runName"], "| id:", run["run_id"])
print(f"pauc_mean={pauc_mean:.4f}  pauc_std={pauc_std:.4f}")
print()

checks = [
    ("pauc_mean >= pauc_mean_min", pauc_mean >= gates["pauc_mean_min"],
     f"{pauc_mean:.4f} vs {gates['pauc_mean_min']}"),
    ("pauc_std <= pauc_std_max", pauc_std <= gates["pauc_std_max"],
     f"{pauc_std:.4f} vs {gates['pauc_std_max']}"),
    ("pauc_mean >= best single feature", pauc_mean >= gates["pauc_min_vs_single_feature"],
     f"{pauc_mean:.4f} vs {gates['pauc_min_vs_single_feature']}"),
]

latency_path = Path("artifacts/evaluation") / run["run_id"] / "latency_report.json"
if "latency_p95_ms_max" in gates:
    if latency_path.exists():
        p95 = json.loads(latency_path.read_text(encoding="utf-8"))["p95_ms"]
        checks.append((
            "latency p95 <= latency_p95_ms_max",
            p95 <= gates["latency_p95_ms_max"],
            f"{p95:.2f} ms vs {gates['latency_p95_ms_max']} ms",
        ))
    else:
        checks.append(("latency report exists", False, f"missing {latency_path}"))

api_latency_path = Path("reports/performance/api_latency_report.json")
if "api_latency_p95_ms_max" in gates or "api_throughput_rps_min" in gates:
    if api_latency_path.exists():
        api_report = json.loads(api_latency_path.read_text(encoding="utf-8"))
        if "api_latency_p95_ms_max" in gates:
            checks.append((
                "api p95 <= api_latency_p95_ms_max",
                api_report["p95_ms"] <= gates["api_latency_p95_ms_max"],
                f"{api_report['p95_ms']:.2f} ms vs {gates['api_latency_p95_ms_max']} ms",
            ))
        if "api_throughput_rps_min" in gates:
            checks.append((
                "api throughput >= api_throughput_rps_min",
                api_report["throughput_rps"] >= gates["api_throughput_rps_min"],
                f"{api_report['throughput_rps']:.2f} rps vs {gates['api_throughput_rps_min']} rps",
            ))
    else:
        checks.append(("api latency report exists", False, f"missing {api_latency_path}; "
                        "run scripts/benchmark_api.py with the API running"))

if "model_size_mb_max" in gates:
    mdir = Path("artifacts/models") / run["run_id"]
    files = list(mdir.glob("fold_*.txt"))
    if len(files) != 5:
        checks.append(("model files exist", False, f"found {len(files)} of 5 in {mdir}"))
    else:
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        checks.append(("model size <= model_size_mb_max",
                       size_mb <= gates["model_size_mb_max"],
                       f"{size_mb:.3f} MB vs {gates['model_size_mb_max']} MB"))

if "fairness_pauc_gap_max" in gates:
    oof_path = Path("artifacts/evaluation") / run["run_id"] / "oof_predictions.csv"
    if not oof_path.exists():
        checks.append(("fairness oof file exists", False, f"missing {oof_path}"))
    else:
        pauc_fn = partial_auc

        oof = pd.read_csv(oof_path)
        meta = pd.read_csv("data/raw/images/metadata.csv",
                           usecols=["isic_id", "sex", "age_approx", "tbp_tile_type"],
                           low_memory=False)
        d = oof.merge(meta, on="isic_id", how="left")
        d["sex"] = d["sex"].fillna("unknown").astype(str)
        d["age_group"] = pd.cut(d["age_approx"], bins=[0, 50, 65, 90],
                                labels=["<=50", "51-65", ">65"]).astype(object)
        d["age_group"] = d["age_group"].where(d["age_approx"].notna(), "unknown").astype(str)

        overall = pauc_fn(d["malignant"], d["oof"])
        min_pos = gates.get("fairness_min_malignant_per_group", 50)
        for col in ["sex", "age_group", "tbp_tile_type"]:
            for name, g in d.groupby(col):
                n_pos = int(g["malignant"].sum())
                if n_pos < min_pos:
                    continue
                gap = overall - pauc_fn(g["malignant"], g["oof"])
                checks.append((f"fairness {col}={name} gap <= max",
                               gap <= gates["fairness_pauc_gap_max"],
                               f"gap {gap:.4f} vs {gates['fairness_pauc_gap_max']} (n_pos={n_pos})"))

failed = 0
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    failed += 0 if ok else 1

print()
print("RESULT:", "ALL GATES PASSED" if failed == 0 else f"{failed} GATE(S) FAILED")
sys.exit(1 if failed else 0)
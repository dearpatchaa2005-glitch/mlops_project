"""CI-safe quality gate check: same thresholds as src/evaluation/check_gates.py
and configs/gates.yaml, but reads reports/gate_snapshot.json (committed to
git) instead of mlflow.db / artifacts/ / data/ (all gitignored, none
available on a GitHub Actions runner). See
src/evaluation/export_gate_snapshot.py for how the snapshot is produced.

    python -m src.evaluation.check_gates_ci                      # default snapshot
    python -m src.evaluation.check_gates_ci path/to/snapshot.json # explicit (used by tests)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

GATES_PATH = Path("configs/gates.yaml")
DEFAULT_SNAPSHOT_PATH = Path("reports/gate_snapshot.json")


def check(snapshot_path: Path = DEFAULT_SNAPSHOT_PATH, gates_path: Path = GATES_PATH) -> int:
    if not snapshot_path.exists():
        print(f"[FAIL] no snapshot at {snapshot_path}; run "
              f"src.evaluation.export_gate_snapshot after training", file=sys.stderr)
        return 1

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    gates = yaml.safe_load(gates_path.read_text(encoding="utf-8"))["gates"]

    print(f"run: {snapshot.get('run_name')} | id: {snapshot.get('run_id')}")
    checks = []

    if "pauc_mean" in snapshot:
        checks.append((
            "pauc_mean >= pauc_mean_min",
            snapshot["pauc_mean"] >= gates["pauc_mean_min"],
            f"{snapshot['pauc_mean']:.4f} vs {gates['pauc_mean_min']}",
        ))
    if "pauc_std" in snapshot:
        checks.append((
            "pauc_std <= pauc_std_max",
            snapshot["pauc_std"] <= gates["pauc_std_max"],
            f"{snapshot['pauc_std']:.4f} vs {gates['pauc_std_max']}",
        ))
    if "pauc_mean" in snapshot and "pauc_min_vs_single_feature" in gates:
        checks.append((
            "pauc_mean >= best single feature",
            snapshot["pauc_mean"] >= gates["pauc_min_vs_single_feature"],
            f"{snapshot['pauc_mean']:.4f} vs {gates['pauc_min_vs_single_feature']}",
        ))
    if "model_latency_p95_ms" in snapshot and "latency_p95_ms_max" in gates:
        checks.append((
            "model latency p95 <= latency_p95_ms_max",
            snapshot["model_latency_p95_ms"] <= gates["latency_p95_ms_max"],
            f"{snapshot['model_latency_p95_ms']:.2f} ms vs {gates['latency_p95_ms_max']} ms",
        ))
    if "api_latency_p95_ms" in snapshot and "api_latency_p95_ms_max" in gates:
        checks.append((
            "api latency p95 <= api_latency_p95_ms_max",
            snapshot["api_latency_p95_ms"] <= gates["api_latency_p95_ms_max"],
            f"{snapshot['api_latency_p95_ms']:.2f} ms vs {gates['api_latency_p95_ms_max']} ms",
        ))
    if "api_throughput_rps" in snapshot and "api_throughput_rps_min" in gates:
        checks.append((
            "api throughput >= api_throughput_rps_min",
            snapshot["api_throughput_rps"] >= gates["api_throughput_rps_min"],
            f"{snapshot['api_throughput_rps']:.2f} rps vs {gates['api_throughput_rps_min']} rps",
        ))
    if "model_size_mb" in snapshot and "model_size_mb_max" in gates:
        checks.append((
            "model size <= model_size_mb_max",
            snapshot["model_size_mb"] <= gates["model_size_mb_max"],
            f"{snapshot['model_size_mb']:.3f} MB vs {gates['model_size_mb_max']} MB",
        ))
    if "fairness_gaps" in snapshot and "fairness_pauc_gap_max" in gates:
        for name, info in snapshot["fairness_gaps"].items():
            checks.append((
                f"fairness {name} gap <= max",
                info["gap"] <= gates["fairness_pauc_gap_max"],
                f"gap {info['gap']:.4f} vs {gates['fairness_pauc_gap_max']} (n_pos={info['n_pos']})",
            ))

    failed = 0
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        failed += 0 if ok else 1

    print()
    print("RESULT:", "ALL GATES PASSED" if failed == 0 else f"{failed} GATE(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SNAPSHOT_PATH
    sys.exit(check(path))

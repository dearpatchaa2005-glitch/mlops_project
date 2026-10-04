"""Proves check_gates_ci.py (the CI-safe gate checker) actually
discriminates pass from fail -- both outcomes, in one deterministic test,
rather than relying solely on someone remembering to trigger and then
revert a failing CI run manually. (CI evidence for both outcomes is still
worth capturing once as an additional artifact -- see README -- but this
test is what guarantees the mechanism itself is correct.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.check_gates_ci import check

GATES_PATH = Path("configs/gates.yaml")


def _write_snapshot(tmp_path: Path, **overrides) -> Path:
    snapshot = {
        "run_id": "test", "run_name": "test",
        "pauc_mean": 0.1582, "pauc_std": 0.0108,
        "model_size_mb": 0.708,
        **overrides,
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_passing_snapshot_exits_zero(tmp_path):
    snapshot_path = _write_snapshot(tmp_path)
    assert check(snapshot_path, GATES_PATH) == 0


def test_failing_snapshot_exits_nonzero(tmp_path):
    # pauc_mean below configs/gates.yaml's pauc_mean_min (0.14)
    snapshot_path = _write_snapshot(tmp_path, pauc_mean=0.05)
    assert check(snapshot_path, GATES_PATH) == 1


def test_oversized_model_fails_its_own_gate(tmp_path):
    snapshot_path = _write_snapshot(tmp_path, model_size_mb=999)
    assert check(snapshot_path, GATES_PATH) == 1


def test_missing_snapshot_file_fails_closed(tmp_path):
    assert check(tmp_path / "does_not_exist.json", GATES_PATH) == 1

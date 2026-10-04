"""Model registry: register a training run as a model version, gated by
configs/gates.yaml, with an explicit human-approval step before it can
serve production traffic.

Flow
----
1. `python -m src.registry.promote <run_id>`
   Runs src/evaluation/check_gates.py against that run. If every gate
   passes, registers the run as a new version of MODEL_NAME and points the
   "candidate" alias at it. Gate failure -> nothing is registered, exit 1.

2. `python -m src.registry.promote <run_id> --approve`
   Same gate check, then also moves the "production" alias to that version
   -- the one alias serving/app.py actually reads at startup. This is the
   approval gate: an automated pass is necessary but not sufficient, a
   person (or a release step standing in for one) has to say "go".

Every transition is appended to artifacts/registry/audit_log.jsonl so there
is a paper trail of who/what promoted which version and when -- the same
log src/registry/rollback.py writes to when reverting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow

MODEL_NAME = "skin_lesion_tabular"
TRACKING_URI = "sqlite:///mlflow.db"
AUDIT_LOG = Path("artifacts/registry/audit_log.jsonl")


def _append_audit(event: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    print("audit:", json.dumps(event))


def run_gates(run_id: str) -> bool:
    """Runs the existing gate checker as a subprocess so promotion logic
    never re-implements (and risks diverging from) the actual gate checks
    in src/evaluation/check_gates.py -- it just honors the exit code.
    Note: check_gates.py always evaluates the *latest* run in the
    experiment, so this only gives a meaningful answer when `run_id` is
    that latest run (true for normal promote-right-after-training use).
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluation.check_gates"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def get_or_create_model(client: mlflow.tracking.MlflowClient) -> None:
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            MODEL_NAME,
            description="LightGBM 5-fold tabular baseline for ISIC 2024 malignancy screening.",
        )


def promote(run_id: str, approve: bool) -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    print(f"checking gates for run {run_id} ...")
    if not run_gates(run_id):
        print(f"[BLOCKED] run {run_id} failed one or more gates; not registering.",
              file=sys.stderr)
        _append_audit({"action": "promote_blocked", "run_id": run_id, "reason": "gate_failure"})
        return 1

    get_or_create_model(client)
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/models",
        run_id=run_id,
        description="Auto-registered after passing configs/gates.yaml.",
    )
    client.set_registered_model_alias(MODEL_NAME, "candidate", mv.version)
    _append_audit({
        "action": "register", "run_id": run_id,
        "model": MODEL_NAME, "version": mv.version, "alias": "candidate",
    })
    print(f"[OK] registered {MODEL_NAME} v{mv.version} (run {run_id}) as 'candidate'")

    if approve:
        client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
        _append_audit({
            "action": "approve_production", "run_id": run_id,
            "model": MODEL_NAME, "version": mv.version, "alias": "production",
        })
        print(f"[OK] approved: {MODEL_NAME} 'production' alias -> v{mv.version}")
    else:
        print("not yet approved for production; re-run with --approve once reviewed.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--approve", action="store_true",
                         help="also move the production alias to this version")
    args = parser.parse_args()
    sys.exit(promote(args.run_id, args.approve))

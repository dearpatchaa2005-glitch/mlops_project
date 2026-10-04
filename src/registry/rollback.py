"""Demonstrate reverting the serving model to a previous registered
version.

    python -m src.registry.rollback --list                # show versions
    python -m src.registry.rollback --to 1                # revert to v1
    python -m src.registry.rollback --to-previous          # one step back

serving/app.py always reads the model it serves through the "production"
alias, resolved fresh on every /health call and cached for /predict (see
serving/app.py for the reload policy). So moving that alias here is a real
rollback of what the running service will use, not just a registry label
change -- restart or hit the app's /reload endpoint and traffic shifts to
the older version with no redeploy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow

from src.registry.promote import MODEL_NAME, TRACKING_URI, _append_audit


def list_versions(client) -> list:
    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production").version
    except mlflow.exceptions.MlflowException:
        current = None
    for v in versions:
        marker = " <- production" if v.version == current else ""
        print(f"  v{v.version}  run={v.run_id}  status={v.status}{marker}")
    return versions


def rollback(client, to_version: str | None, to_previous: bool) -> int:
    versions = list_versions(client)
    if not versions:
        print("[FAIL] no registered versions for", MODEL_NAME, file=sys.stderr)
        return 1

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production")
    except mlflow.exceptions.MlflowException:
        current = None

    if to_previous:
        if current is None:
            print("[FAIL] no current 'production' alias to roll back from", file=sys.stderr)
            return 1
        older = [v for v in versions if int(v.version) < int(current.version)]
        if not older:
            print("[FAIL] already at the oldest registered version", file=sys.stderr)
            return 1
        target = older[-1]
    elif to_version is not None:
        target = next((v for v in versions if v.version == str(to_version)), None)
        if target is None:
            print(f"[FAIL] no version {to_version} for {MODEL_NAME}", file=sys.stderr)
            return 1
    else:
        print("[FAIL] specify --to <version> or --to-previous", file=sys.stderr)
        return 1

    client.set_registered_model_alias(MODEL_NAME, "production", target.version)
    _append_audit({
        "action": "rollback",
        "model": MODEL_NAME,
        "from_version": current.version if current else None,
        "to_version": target.version,
        "run_id": target.run_id,
    })
    print(f"[OK] production rolled back: v{current.version if current else '?'} "
          f"-> v{target.version} (run {target.run_id})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--to", dest="to_version", default=None)
    group.add_argument("--to-previous", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    _client = mlflow.tracking.MlflowClient()

    if args.list:
        sys.exit(0 if list_versions(_client) is not None else 1)
    sys.exit(rollback(_client, args.to_version, args.to_previous))

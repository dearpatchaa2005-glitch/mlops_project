"""End-to-end pipeline, runnable with one command.

    python dags/skin_lesion_pipeline.py

Built with Prefect rather than Airflow: the team's dev machines are
Windows laptops, and Airflow's scheduler/webserver effectively requires
Docker or WSL2 there, while Prefect runs as a plain pip-installed library
with no extra services. It's still a DAG in the sense the course asks for
-- a graph of tasks with dependencies, runnable end to end with one command
or one click, with each step's success/failure visible (Prefect's local UI:
`prefect server start`, or just the console output below).

Stages: validate data -> make splits -> train -> evaluate -> check gates
-> register (gated) -> notify. A failure at any stage stops the ones after
it -- see `allow_failure=False` (the default) on each `.submit()`/call.

Scheduling: this script takes no arguments and always runs the full
pipeline against data/raw/images/metadata.csv. For the "training happens
automatically on a trigger" requirement, see docs/experiment_log.md's
retrain policy -- this script is what that trigger runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str], cwd: Path = ROOT, run_id: str | None = None) -> None:
    """Run a pipeline step as a subprocess.

    When `run_id` is given, it's exported as SKIN_LESION_RUN_ID so the
    evaluation scripts (src/evaluation/run_context.py) evaluate *that*
    freshly-trained run instead of the fixed baseline RUN_ID they fall
    back to for plain manual invocations. Without this, every DAG run
    would keep evaluating the old baseline model while check_gates.py
    checks the newly-trained run -- gates would always fail with
    "file not found" even when training itself succeeded.
    """
    logger = get_run_logger()
    env = {**os.environ, "SKIN_LESION_RUN_ID": run_id} if run_id else None
    logger.info("$ %s%s", " ".join(args), f"  [SKIN_LESION_RUN_ID={run_id}]" if run_id else "")
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=env)
    logger.info(result.stdout[-4000:])
    if result.returncode != 0:
        logger.error(result.stderr[-4000:])
        raise RuntimeError(f"step failed (exit {result.returncode}): {' '.join(args)}")


@task(retries=0)
def validate_data(metadata_path: str = "data/raw/images/metadata.csv") -> None:
    _run([PY, "-m", "src.validation.validate_metadata", metadata_path])


@task
def make_splits() -> None:
    _run([PY, "src/data/make_splits.py"])


@task
def check_splits() -> None:
    _run([PY, "src/data/check_splits.py"])


@task
def train() -> None:
    _run([PY, "src/training/train_tabular_baseline.py"])


@task
def evaluate(run_id: str) -> None:
    for script in [
        "src/evaluation/oof_fairness.py",
        "src/evaluation/evaluate_test.py",
        "src/evaluation/bootstrap_test.py",
        "src/evaluation/measure_latency.py",
        "src/evaluation/fairness_report.py",
    ]:
        if (ROOT / script).exists():
            _run([PY, script], run_id=run_id)


@task
def benchmark_api_if_running() -> None:
    """Best-effort: only runs if something is already serving on :8000
    (e.g. started separately with `uvicorn serving.app:app`). Skipped,
    not failed, if nothing is listening -- the API latency gate then
    reports a clear missing-report error instead of a false pass.
    """
    import urllib.request

    logger = get_run_logger()
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
    except Exception:
        logger.warning("no API listening on :8000; skipping scripts/benchmark_api.py "
                        "(start it with `uvicorn serving.app:app` to include this check)")
        return
    _run([PY, "scripts/benchmark_api.py"])


@task
def check_gates() -> bool:
    logger = get_run_logger()
    result = subprocess.run([PY, "src/evaluation/check_gates.py"], cwd=ROOT,
                             capture_output=True, text=True)
    logger.info(result.stdout)
    return result.returncode == 0


@task
def latest_run_id() -> str:
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=["skin_lesion_tabular_baseline"],
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    return runs.iloc[0]["run_id"]


@task
def register_candidate(run_id: str) -> None:
    _run([PY, "-m", "src.registry.promote", run_id])


@flow(name="skin-lesion-pipeline")
def skin_lesion_pipeline(metadata_path: str = "data/raw/images/metadata.csv") -> None:
    logger = get_run_logger()

    validate_data(metadata_path)
    make_splits()
    check_splits()
    train()

    # Resolve the run train() just created *before* evaluating, so every
    # evaluation script below writes under artifacts/evaluation/<run_id>/
    # for this exact run (see src/evaluation/run_context.py), not the
    # fixed baseline run_id they'd otherwise default to.
    run_id = latest_run_id()
    evaluate(run_id)
    benchmark_api_if_running()

    gates_passed = check_gates()
    if not gates_passed:
        logger.error("gates failed for run %s; NOT registering. "
                      "See output above for which check(s) failed.", run_id)
        raise RuntimeError("quality gates failed; pipeline stops here on purpose")

    register_candidate(run_id)
    logger.info(
        "pipeline complete. Run %s registered as 'candidate'. "
        "Promote to production explicitly with: "
        "python -m src.registry.promote %s --approve", run_id, run_id,
    )


if __name__ == "__main__":
    skin_lesion_pipeline()

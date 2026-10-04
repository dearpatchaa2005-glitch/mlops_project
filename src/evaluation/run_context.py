"""Single place that decides *which* MLflow run the evaluation / serving
scripts operate on.

Every script in src/evaluation/ (and src/serving/predictor.py) used to
hardcode the same literal run_id:

    RUN_ID = "16b43986f5704e0fafbef7cdcf7149b8"

That's fine for a human re-running one script against "the" winning run,
but it quietly breaks full automation: src/training/train_tabular_baseline.py
starts a *fresh* mlflow run (and a fresh artifacts/models/<run_id>/ folder)
on every call, so a DAG run that trains a new model and then calls these
scripts would keep evaluating the *old* run while check_gates.py (which
already looks up "whatever run is newest in mlflow", see its
mlflow.search_runs(..., order_by=["attributes.start_time DESC"])) checks the
*new* one -- every gate that depends on an evaluation artifact then fails
with a "file not found", even when training itself succeeded.

Fix: read the run id from the SKIN_LESION_RUN_ID environment variable when
it's set (dags/skin_lesion_pipeline.py sets it to the run that was *just*
trained, right after the train() task and before evaluate()), and fall back
to the fixed literal below for plain `python src/evaluation/whatever.py`
runs against the known-good baseline -- so nothing about the manual
single-script workflow changes.
"""
from __future__ import annotations

import os

# The team's current "blessed" baseline run (see docs/experiment_log.md).
# Used whenever SKIN_LESION_RUN_ID isn't set in the environment.
DEFAULT_RUN_ID = "16b43986f5704e0fafbef7cdcf7149b8"

RUN_ID = os.environ.get("SKIN_LESION_RUN_ID", DEFAULT_RUN_ID)

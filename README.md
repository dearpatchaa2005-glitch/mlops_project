# Skin Lesion Risk Triage — Tabular Baseline (CP413008 MLOps project)

A pre-screening risk-triage system for skin lesions, built on the
[ISIC 2024 Challenge (SLICE-3D)](https://www.kaggle.com/competitions/isic-2024-challenge)
dataset: metadata + crops from 3D total-body photography, binary
malignant/benign labels, heavily imbalanced (<0.1% malignant). Full
problem framing, stakeholders, and cost/revenue discussion are in the
team's AI Project Canvas and the course report (not part of this repo);
this README covers how the system is built and how to run it.

**Current scope:** a LightGBM model on the tabular metadata only (age,
sex, lesion site/size, and the other ~36 numeric/categorical columns ISIC
provides per image) — not yet the image-based CNN the canvas originally
scoped. This was a deliberate sequencing choice given the team's hardware
(a single laptop GPU) and the deadline: ship a correct, fully-automated,
gated pipeline end to end on the cheaper model first, then swap in an
image model behind the same validation/registry/serving/monitoring
scaffolding later — see "Known limitations" below.

- **Optimizing metric:** partial AUC at TPR ≥ 0.80 (pAUC, max 0.20) — the
  ISIC competition's own metric, chosen because a missed malignant case
  (false negative) costs far more than an unnecessary referral (false
  positive), so only the high-sensitivity region of the ROC curve matters.
- **Gating metrics:** see [`docs/slo.md`](docs/slo.md) — pAUC floor vs. a
  naive single-feature baseline, per-group fairness gap, model size, and
  both model-only and full-API p95 latency + throughput. A run that fails
  any gate is registered as `candidate` for visibility but never promoted
  to `production`.
- **Architecture diagram:** [`docs/architecture.md`](docs/architecture.md).
- **Experiment log (all CV rounds, held-out test, fairness, latency,
  drift demo, retrain policy):** [`docs/experiment_log.md`](docs/experiment_log.md).

## Repository layout

```
src/
  data/           EDA + integrity checks (not gated steps) + make_splits.py (gated: check_splits.py)
  preprocessing/  features.py -- the ONE feature-prep module used by both
                  train_tabular_baseline.py and serving/predictor.py
  validation/     schema.py (pandera schema) + validate_metadata.py (CLI gate)
  training/       train_tabular_baseline.py + ad-hoc sanity checks (check_metric.py etc.)
  evaluation/     metrics.py (shared pAUC), oof_fairness / evaluate_test /
                  bootstrap_test / measure_latency / fairness_report,
                  check_gates.py (local gate) + check_gates_ci.py +
                  export_gate_snapshot.py (CI-safe gate using a committed
                  snapshot instead of mlflow.db), run_context.py (which
                  run_id a script evaluates -- see below)
  registry/       promote.py (register + gate + alias), rollback.py
  serving/        predictor.py (inference; imports src/preprocessing/features.py)
  monitoring/     drift.py (PSI data drift + pAUC-drop concept drift)
serving/app.py     FastAPI app (/predict /health /slo /metrics /reload)
dags/              skin_lesion_pipeline.py -- Prefect flow, the "one command" DAG
scripts/           benchmark_api.py, smoke_test_api.py, simulate_drift.py
configs/gates.yaml  all gate thresholds, one file
docker/, docker-compose.yml  containerized serving + full observability stack
infra/             prometheus/, alertmanager/, grafana/ configs
.github/workflows/ci.yml  lint + data-validation + model-gates, on every push/PR
tests/             unit tests (preprocessing, metrics, schema, CI gate logic)
                   + integration tests (the live FastAPI app)
docs/              experiment_log.md, slo.md, architecture.md
```

## Running from a clean machine

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt           # api + tests + linting
pip install -r requirements/orchestration.txt # + Prefect, only if you'll run the DAG
```
`requirements/base.txt` / `serving.txt` / `dev.txt` / `orchestration.txt`
pin exact versions (checked against the team's working `.venv`) so a fresh
install reproduces the same environment `docs/experiment_log.md` was
produced with.

### 2. Data

Download the ISIC 2024 Challenge data from Kaggle and place it exactly as:

```
data/raw/images/metadata.csv
data/raw/metadata/ISIC_2024_Training_GroundTruth.csv
```
(`data/` is gitignored on purpose — see "What's NOT in this repo" below.)

### 3. Validate -> split -> train -> evaluate -> gate -> register (one command)

```bash
python dags/skin_lesion_pipeline.py
```
This runs, in order: schema validation (stops + writes an error report on
bad data — try it against `tests/fixtures/bad_metadata_sample.csv` to see
the stop-and-alert behavior), patient-level stratified splitting, training
(new MLflow run), the full evaluation suite, quality gates, and — only if
every gate passes — registers the new run as `candidate` in the model
registry. It is idempotent to re-run: each run gets a fresh MLflow
`run_id`, nothing is overwritten.

Or run each stage by hand (what the DAG above calls, in the same order):

```bash
python -m src.validation.validate_metadata data/raw/images/metadata.csv
python src/data/make_splits.py
python src/data/check_splits.py
python src/training/train_tabular_baseline.py
python src/evaluation/oof_fairness.py      # + evaluate_test.py, bootstrap_test.py,
                                            #   measure_latency.py, fairness_report.py
python -m src.evaluation.check_gates
```
These evaluation scripts default to evaluating the team's current blessed
baseline run (`RUN_ID` in `src/evaluation/run_context.py`) when run by
hand. The DAG instead sets `SKIN_LESION_RUN_ID=<the run it just trained>`
in the environment so they evaluate *that* run — see the comment at the
top of `run_context.py` if you're wiring in another step.

### 4. Promote / roll back the model registry

```bash
python -m src.registry.promote <run_id>              # registers as 'candidate' only
python -m src.registry.promote <run_id> --approve     # + 'production' alias (gates must pass)
python -m src.registry.rollback --list
python -m src.registry.rollback --to-previous         # or --to <version>
```
Every promote/rollback is appended to `artifacts/registry/audit_log.jsonl`.

### 5. Serve

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```
Resolves the `production` registry alias on startup, falling back to the
fixed baseline run if the registry has no `production` alias yet (e.g. a
fresh clone with no promotion run). After a promote/rollback, hit
`POST /reload` to hot-swap the running process onto the new alias with no
restart. Try it:
```bash
python scripts/smoke_test_api.py       # good + malformed request
python scripts/benchmark_api.py        # p50/p95 latency + throughput -> reports/performance/
curl localhost:8000/health
curl localhost:8000/slo
curl localhost:8000/metrics
```

### 6. Serve in Docker, with the full observability stack

```bash
docker compose up --build
```
Brings up the API (`:8000`), an MLflow UI (`:5000`), Prometheus (`:9090`),
Alertmanager (`:9093`) and Grafana (`:3000`, anonymous viewer enabled).
The image does **not** bake in model weights — `artifacts/`, `mlruns/` and
`mlflow.db` are bind-mounted (see `docker/Dockerfile`), so step 3 (or at
least a trained run + `mlflow.db`) must exist on the host first. **Not
build-tested in the environment this was developed in** (its sandbox
blocks Docker Hub pulls) — `docker compose config` was used to confirm the
compose file is syntactically valid, but the actual `docker compose up
--build` needs to be run and confirmed once on the team's own machine
before the demo.

### 7. Monitoring demo (data drift vs. concept drift)

```bash
python scripts/simulate_drift.py       # writes synthetic "shifted" live traffic
python -m src.monitoring.drift         # compares it to the training distribution
curl localhost:8000/metrics | grep drift   # with the API running
```
See `docs/experiment_log.md` → "Monitoring" for what each demo run showed
and the retrain-trigger policy.

### 8. Tests and linting (what `.github/workflows/ci.yml` runs)

```bash
ruff check src/ serving/ dags/ scripts/ tests/
python -m pytest tests/ -q
```
17 tests: feature-prep, the shared pAUC metric, the pandera schema (both a
clean and a corrupted fixture), the CI-safe gate checker (both a passing
and a failing snapshot, so CI's "show evidence of both pass and fail"
requirement is met programmatically, not just by screenshot), and
integration tests against the live FastAPI app (skip gracefully if no
trained model is present yet).

## What's NOT in this repo (by design, see `.gitignore`)

`data/`, `artifacts/`, `mlruns/`, `mlflow.db`, `runtime/`, `*.log` — all
generated by running the steps above, all machine/run-specific, and (for
`data/`) too large and under a dataset license that shouldn't be
redistributed. This is also why `requirements/*.txt` pin exact versions
and every MLflow run tags its own git commit + a hash of the exact
`splits.csv` it trained on (see `train_tabular_baseline.py`) — reproducing
a result means re-running the pipeline against the same data + commit, not
re-downloading someone's output.

## Known limitations / honest caveats for the report

- **Tabular, not image-based.** The canvas scoped a CNN/transfer-learning
  image model; what's implemented is a metadata-only LightGBM baseline.
  Defensible as a first increment (full gated pipeline before a bigger
  model), but say so explicitly in the report rather than implying it's
  the final model.
- **Docker build is untested**, not just unverified — the dev sandbox this
  was built in blocks Docker Hub registry pulls. Build and run
  `docker compose up --build` once for real before the demo.
- **API latency/throughput numbers need re-measuring** on real hardware —
  see the caveat at the bottom of `docs/slo.md`. The sequential numbers
  (p50=44 ms/p95=63 ms) are trustworthy; a concurrency test run in the same
  2-vCPU sandbox was not (client and server competed for the same cores).
- **`pauc_max_regression` gate (model not allowed to regress vs. the
  current production version) is declared in `configs/gates.yaml` but the
  comparison isn't implemented yet** in `check_gates.py` — currently only
  an absolute floor (`pauc_mean_min`), not a regression-vs-previous check.
- **Branches/PRs:** this repo was built and refactored through an AI
  pairing session working from a snapshot (zip) of the repo, which cannot
  push, commit, or open PRs — all commits, branches and PRs are made by
  the team themselves (grading rubric section 7 checks for actual branch +
  PR history, so this matters: commit this work through normal
  branch → PR → merge, not a single direct commit to main).
- **AI-assistance disclosure** belongs in the report document itself
  (what was AI-assisted and why), not as a separate file in this repo.

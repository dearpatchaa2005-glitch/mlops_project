# Service Level Objectives (SLO)

These are the SLOs this service is held to, where the thresholds live
(`configs/gates.yaml`), and how each one is actually measured rather than
just declared.

| SLO | Target | Measured by | Enforced by |
|---|---|---|---|
| Model-only inference latency (p95) | ≤ 50 ms | `src/evaluation/measure_latency.py` (500 requests, 20 warm-up, single thread) → `artifacts/evaluation/<run_id>/latency_report.json` | `src/evaluation/check_gates.py` gate `latency_p95_ms_max`; blocks promotion |
| Full HTTP API latency (p95), `/predict` | ≤ 150 ms | `scripts/benchmark_api.py` against a running `serving/app.py` → `reports/performance/api_latency_report.json` | gate `api_latency_p95_ms_max` |
| API throughput | ≥ 15 requests/sec | same benchmark as above | gate `api_throughput_rps_min` |
| Model quality: pAUC (TPR ≥ 0.80) | mean ≥ 0.14, std ≤ 0.03 across 5 folds | `src/training/train_tabular_baseline.py` (patient-level CV) | gates `pauc_mean_min`, `pauc_std_max` |
| Model beats a naive baseline | pAUC ≥ best single feature (0.0442) | same CV run | gate `pauc_min_vs_single_feature` |
| Fairness | per-group pAUC gap vs. overall ≤ 0.03 (groups with ≥ 50 malignant cases) | `src/evaluation/fairness_report.py`, `src/evaluation/oof_fairness.py` | gate `fairness_pauc_gap_max` |
| Model size | ≤ 5 MB | sum of the 5 fold model files | gate `model_size_mb_max` |
| Service availability | `GET /health` returns 200 with the currently-loaded model's info | `serving/app.py` | Docker `HEALTHCHECK`, Prometheus scrape target health |

**Business framing (why these, not just accuracy):** this is a pre-screening
triage aid, not a diagnosis. A false negative (missed malignant lesion) is
far more costly than a false positive (an unnecessary derm referral) --
that's why the optimizing metric is pAUC at TPR ≥ 0.80 rather than plain
accuracy or AUC: it only scores the model on the high-sensitivity region
clinicians actually operate in. The latency/throughput SLOs exist because
a triage tool that is slow to respond doesn't get used; the fairness and
size/latency gates exist so a model that is accurate on average but
unreliable for a subgroup, too slow, or too large to deploy cheaply is
never promoted, even if its headline pAUC looks good.

## How "within SLO" is actually checked, not just asserted

- **At promotion time (one-shot, gating):** `src/evaluation/check_gates.py`
  reads the latest MLflow run's metrics plus the latency/throughput reports
  above and prints a `[PASS]`/`[FAIL]` line per gate; `src/registry/promote.py`
  calls it and refuses to set the `production` alias if any gate fails
  (`candidate` is still registered either way, so a failed run is visible
  in the registry, not silently dropped). The CI-safe variant
  (`src/evaluation/check_gates_ci.py` + `reports/gate_snapshot.json`) runs
  the same checks in GitHub Actions, where `mlflow.db`/`artifacts/` aren't
  available.
- **At serving time (continuous):** every `/predict` call's latency is
  pushed into `REQUEST_LATENCY` (a Prometheus histogram) and a rolling
  in-process window; `GET /slo` reports the live p50/p95 over that window
  against `latency_p95_ms_max` right now, without waiting for the next
  benchmark run. `infra/prometheus/alert_rules.yml` also alerts on the
  Prometheus histogram directly, so an SLO breach under real traffic pages
  whoever owns Alertmanager, independent of the `/slo` endpoint.

## Known caveat on the API latency/throughput numbers

The measured 44 ms / 63 ms (p50/p95) numbers in `docs/experiment_log.md`
were taken **sequentially** on a 2-vCPU cloud sandbox with no other load --
trustworthy as a lower bound, not as a concurrent-load number. A
concurrency=8 run on the same sandbox showed p50=467 ms/p95=564 ms, but that
run had the benchmark client and the API server fighting over the same 2
vCPUs, which is a sandbox artifact, not a property of the service. **Before
quoting a throughput/latency number in the final report, re-run
`scripts/benchmark_api.py` on the team's actual deploy target** (or at
least a machine with more than 2 cores) and update
`reports/performance/api_latency_report.json` + the gate comment in
`configs/gates.yaml`.

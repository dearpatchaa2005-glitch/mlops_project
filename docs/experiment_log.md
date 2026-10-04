# Experiment log (tabular baseline, 5-fold patient-level CV on trainval)

| Run | Change | pAUC mean (std) | PR-AUC mean |
|---|---|---|---|
| baseline | 31 leaves, 300 rounds, spw=20 | 0.0211 (0.0125) | 0.0096 |
| no_xyz | drop tbp_lv_x/y/z | 0.0173 (0.0043) | 0.0124 |
| simple | 7 leaves, 150 rounds, mcs=500 | 0.0961 (0.0319) | 0.0338 |
| simple_spw1 | scale_pos_weight=1 | 0.1593 (0.0114) | 0.0809 |
| simple_spw1_no_nevi | drop tbp_lv_nevi_confidence | 0.1582 (0.0108) | 0.0883 |

## Findings
- Overfitting (complex model, spw=20) was the main cause of near-random pAUC.
- Metric function verified on synthetic data (random ~0.02, perfect = 0.20).
- Results do not depend on tbp_lv_nevi_confidence; excluded to avoid
  provenance concerns (the score comes from an external model, training
  data unverified).
- Several settings were tried on the same CV folds, so CV may be slightly
  optimistic. Test set (56 malignant) is untouched.
- Per-patient normalization gave +0.004 to +0.006 pAUC (below noise, under 1 std).
  patient_n_lesions is not a shortcut (removing it changed pAUC by 0.001).
  Not adopted: requires many lesions per patient at inference time.

## Limitations
- Not a medical diagnostic tool.
- Test set has few positives; expect wide uncertainty.
- Fairness and latency not yet measured.

| simple_spw1_no_nevi_pnorm | + 14 per-patient normalized features + patient_n_lesions | 0.1639 (0.0063) | 0.0956 |
| simple_spw1_no_nevi_pnorm_no_n | pnorm without patient_n_lesions | 0.1625 (0.0067) | 0.0922 |

## Held-out test evaluation

- Run ID: `16b43986f5704e0fafbef7cdcf7149b8`
- Model: LightGBM, 36 features, USE_PNORM=False.
- Prediction: average scores from all 5 saved fold models.
- Test set: 57,294 images, 147 patients, 56 malignant lesions.
- Test pAUC (TPR >= 0.80, maximum 0.20): 0.1473.
- Test ROC-AUC: 0.9248.
- Test Average Precision: 0.0378.
- Metrics and test_metrics.json saved to the same MLflow run.
- No model tuning was performed between the two test evaluations;
  the second execution recorded the same results in MLflow.
- These results evaluate the 5-model ensemble. CV results summarize
  individual fold models, so the two are not directly equivalent.
- Confidence intervals have not yet been calculated.

## Test confidence intervals

- Method: patient-level cluster bootstrap, percentile 95% CI.
- Fixed ensemble; 1,000 rounds, seed=42.
- Test patients: 147; patients with malignant lesions: 42.
- Valid rounds: 1,000; skipped rounds: 0.
- pAUC: 0.1473 (95% CI: 0.1249–0.1674).
- ROC-AUC: 0.9248 (95% CI: 0.8901–0.9522).
- Average Precision: 0.0378 (95% CI: 0.0216–0.0739).
- Report saved to MLflow as test_confidence_intervals.json.
- Intervals reflect test-patient sampling uncertainty,
  not retraining variability or performance at other sites.

## Model inference latency

- Run ID: `16b43986f5704e0fafbef7cdcf7149b8`
- Scope: prediction from 5 LightGBM models plus score averaging.
- Input: one prepared metadata row per request.
- Batch size: 1; concurrent requests: 1.
- Threads per model: 1.
- Warmup: 20 requests; measured: 500 requests.
- p50: 5.153 ms.
- p95: 7.613 ms.
- Mean: 5.687 ms.
- Excludes model loading, file reading, image feature extraction,
  and API/network overhead.
- Hardware and software details are in latency_report.json.
- Report and per-request timings saved to MLflow.
- Results describe this local benchmark, not an API latency guarantee.

## Exploratory fairness report (test set, run 16b43986...)
Not a gate: computed after viewing test results, each group has 13-38
malignant cases, no group-level CIs computed.

| Group | Rows | Malignant | pAUC | ROC-AUC |
|---|---|---|---|---|
| female | 25,981 | 17 | 0.1602 | 0.933 |
| male | 30,851 | 38 | 0.1368 | 0.916 |
| age <=50 | 16,926 | 13 | 0.1625 | 0.947 |
| age 51-65 | 28,271 | 26 | 0.1640 | 0.945 |
| age >65 | 11,847 | 17 | 0.1127 | 0.863 |

Observation: the >65 group scores lowest on both metrics; evidence is weak
(small counts). Future work: group-wise CV pAUC as a gate for the next model.

## Group-wise OOF pAUC (5-fold CV, patient-level bootstrap, run 16b43986...)

| Group | Malignant | pAUC | 95% CI |
|---|---|---|---|
| female | 92 | 0.1405 | 0.1116-0.1656 |
| male | 236 | 0.1652 | 0.1543-0.1742 |
| age <=50 | 66 | 0.1555 | 0.1298-0.1758 |
| age 51-65 | 173 | 0.1529 | 0.1357-0.1690 |
| age >65 | 95 | 0.1686 | 0.1493-0.1830 |
| tile XP | 159 | 0.1474 | 0.1298-0.1633 |
| tile white | 178 | 0.1671 | 0.1543-0.1782 |

Finding: no consistent subgroup gap; CIs overlap. The test-set gap for
age >65 (0.1127) was not reproduced in CV (0.1686), likely small-sample noise.
Not covered: skin tone (no variable), other imaging sites.

## API latency & throughput (through the HTTP service, not model-only)

The 7.6 ms figure above is the model-only cost (5 LightGBM predicts +
averaging), measured by `src/evaluation/measure_latency.py` with no API,
network, JSON or validation overhead -- that script says so explicitly.
`scripts/benchmark_api.py` measures the number that actually matters for
the SLO: a full `/predict` round trip.

- Sequential (concurrency=1), 100 requests, 10 warmup: p50 = 44.3 ms,
  p95 = 62.7 ms, throughput = 21.5 req/s.
- Concurrency=8, 300 requests: p50 = 467.8 ms, p95 = 564.3 ms. This run is
  **not** a usable hardware number -- it was measured on a 2-vCPU sandbox
  with the load generator and the server competing for the same 2 cores,
  which inflates latency under concurrency far beyond what a normal
  4-core+ dev machine would show. Re-run
  `python scripts/benchmark_api.py --n 300 --concurrency 8` on the actual
  dev/deploy machine before quoting a concurrent-load number in the report.
- `configs/gates.yaml` now has a separate `api_latency_p95_ms_max: 150`
  SLO (distinct from the model-only `latency_p95_ms_max: 50`), checked by
  `src/evaluation/check_gates.py` against
  `reports/performance/api_latency_report.json`. 150 ms was picked with
  headroom over the 63 ms sequential measurement; tighten it once a real
  concurrent-load number exists.

## Monitoring: data drift vs. concept drift

Implemented in `src/monitoring/drift.py`, exposed on `/metrics` via
`serving/app.py` (gauges `data_drift_max_psi`, `data_drift_detected`,
`concept_drift_pauc_drop`, `concept_drift_detected`), alerted on in
`infra/prometheus/alert_rules.yml`.

- **Data drift** (input distribution changed): PSI per numeric feature +
  proportion-shift per categorical, live traffic (`runtime/logs/predictions.jsonl`,
  written by every `/predict` call) vs. the training distribution.
  No ground truth needed, so this can run continuously.
- **Concept drift** (model is wrong in a new way): recomputes pAUC on a
  recently-labeled window and compares to the reference CV pAUC (0.1582).
  This metric is only as fresh as the ground truth is -- a biopsy/follow-up
  result lags behind the prediction -- so until a labeled window is
  supplied, the check reports "insufficient labeled data" rather than
  silently assuming no drift.
- Verified with `scripts/simulate_drift.py`: an unshifted sample does not
  flag, a sample with age shifted +25y and `tbp_tile_type` pushed to one
  category flags both (`age_approx` PSI = 3.4, `tbp_tile_type` shift =
  0.49); a sample with ~40% of labels flipped flags concept drift
  (pAUC drop = 0.134).

**Retrain policy (first pass):** retrain is triggered when either
(a) `DataDriftDetected` or `ConceptDriftSuspected` fires and stays firing
for more than one monitoring cycle (avoids retraining on a one-off blip),
or (b) on a fixed schedule regardless of alerts (monthly, pending the
team's call on cadence), whichever comes first. In this prototype the
retrain itself is a manual run of the pipeline DAG
(`dags/skin_lesion_pipeline.py`) kicked off by whoever is on call for the
alert; the hook point for making that automatic is the Alertmanager
webhook receiver in `infra/alertmanager/alertmanager.yml`, which currently
points at a placeholder (no CI/CD trigger wired to it yet).

## Fixed a retrain-time bug: evaluation scripts pinned to a fixed RUN_ID

`src/serving/predictor.py`, `src/evaluation/oof_fairness.py`,
`evaluate_test.py`, `bootstrap_test.py`, `measure_latency.py` and
`fairness_report.py` each used to hardcode
`RUN_ID = "16b43986f5704e0fafbef7cdcf7149b8"` -- fine for re-running one
script by hand against the one blessed baseline run, but it silently broke
full automation: `train_tabular_baseline.py` starts a *new* mlflow run
(and a new `artifacts/models/<run_id>/`) every time, while
`check_gates.py` already always checks whichever run is newest. Running
the DAG end to end therefore trained a new run, then kept evaluating the
*old* one, so every gate reading an evaluation artifact failed with
"file not found" even when training succeeded -- confirmed by actually
running `dags/skin_lesion_pipeline.py` end to end once: it correctly
stopped at `check_gates` ("quality gates failed; pipeline stops here on
purpose"), and running `check_gates.py` directly against that run showed
exactly this: `[FAIL] latency report exists (missing ...)` and
`[FAIL] fairness oof file exists (missing ...)`.

Fix: all six now import `RUN_ID` from `src/evaluation/run_context.py`,
which reads the `SKIN_LESION_RUN_ID` environment variable and falls back
to the same fixed baseline id for plain manual runs (`python
src/evaluation/whatever.py`, unchanged behavior). The DAG
(`dags/skin_lesion_pipeline.py`) now resolves the just-trained run id
right after `train()` and exports it as `SKIN_LESION_RUN_ID` for every
`evaluate()` subprocess. Also fixed: `oof_fairness.py` wrote its output
CSV without creating `artifacts/evaluation/<run_id>/` first, which only
ever worked because that directory already existed from a previous
manual `evaluate_test.py` run on the same fixed id -- added the missing
`mkdir(parents=True, exist_ok=True)`.

Re-verified after the fix (same run, synthetic sandbox data):
`latency report exists` and `fairness oof file exists` now `[PASS]`,
and every per-group fairness-gap check passes. The remaining gate
failures on that run (`pauc_mean`, `pauc_mean >= best single feature`)
are expected -- that run was trained on synthetic placeholder data
generated in this sandbox for drift-detection testing, not the real
ISIC data, so a near-random pAUC (~0.02) is the correct outcome, not a
bug. Re-run the DAG on the real dataset to get a real gate result.
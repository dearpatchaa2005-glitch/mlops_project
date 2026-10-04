# Architecture

```mermaid
flowchart TD
    subgraph DATA["Data"]
        RAW["ISIC 2024 metadata.csv +\nGroundTruth.csv (data/raw/)"]
        VALID["validate_metadata.py\n(pandera schema gate)"]
        SPLIT["make_splits.py\npatient-level StratifiedGroupKFold\n(train/val/test, 5 folds)"]
        RAW --> VALID
        VALID -- "schema OK" --> SPLIT
        VALID -- "schema violation\nexit 1 + error report" --> STOP1(["pipeline stops\n+ alerts"])
    end

    subgraph TRAIN["Train & evaluate"]
        FEAT["src/preprocessing/features.py\n(shared train/serve preprocessing)"]
        TRAINM["train_tabular_baseline.py\nLightGBM, 5-fold CV\n-> new MLflow run_id"]
        EVAL["evaluate():\noof_fairness, evaluate_test,\nbootstrap_test, measure_latency,\nfairness_report"]
        GATES["check_gates.py\npAUC / fairness / latency /\nsize gates vs configs/gates.yaml"]
        SPLIT --> FEAT --> TRAINM --> EVAL --> GATES
        GATES -- "FAIL" --> STOP2(["pipeline stops here\non purpose, nothing promoted"])
    end

    subgraph REGISTRY["Model registry"]
        PROMOTE["promote.py\nregisters version,\nalias 'candidate' always,\n'production' only if --approve"]
        ROLLBACK["rollback.py\nmove 'production' alias\nto an older version"]
        MLDB[("mlflow.db\n(tracking + registry)")]
        GATES -- "PASS" --> PROMOTE --> MLDB
        ROLLBACK <--> MLDB
    end

    subgraph SERVE["Serving"]
        API["serving/app.py (FastAPI,\nin a Docker container)\n/predict /health /slo /metrics /reload"]
        FEAT -. "same feature code,\nno train/serve skew" .-> API
        MLDB -- "resolve 'production' alias" --> API
        LOG[("runtime/logs/predictions.jsonl")]
        API --> LOG
    end

    subgraph MONITOR["Monitoring"]
        DRIFT["src/monitoring/drift.py\nPSI (data drift) +\npAUC drop (concept drift)"]
        PROM["Prometheus\n(scrapes /metrics)"]
        ALERT["Alertmanager"]
        GRAF["Grafana dashboards"]
        LOG --> DRIFT --> API
        API --> PROM --> ALERT
        PROM --> GRAF
        ALERT -- "sustained drift or\nSLO breach" --> RETRAIN(["retrain trigger\n(see docs/experiment_log.md\nretrain policy)"])
    end

    subgraph ORCH["Orchestration & CI/CD"]
        DAG["dags/skin_lesion_pipeline.py\n(Prefect flow: one command,\nruns every stage above in order)"]
        CI["GitHub Actions ci.yml:\nlint / data-validation / model-gates\n(pass + fail cases both shown)"]
    end

    RETRAIN -. "kicks off" .-> DAG
    DAG -. "drives" .-> RAW
    CI -. "runs on every push/PR,\nindependent of DAG" .-> VALID
    CI -. " " .-> GATES
```

## Reading the diagram

- **Data → Train → Registry → Serve → Monitor** is the main left-to-right
  path every prediction/retrain goes through.
- The two **"pipeline stops"** boxes are deliberate: a bad input file never
  reaches training (data-validation gate), and a bad model never reaches
  `production` (quality gate) -- both exit non-zero, which is what makes
  them real stops rather than warnings, whether run by hand, by the DAG, or
  by CI.
- **`src/preprocessing/features.py`** is shared, unchanged code between the
  training path and the serving path (dotted line) -- this is the
  training/serving-skew prevention mechanism, not just a convention.
- **Monitoring** reads the *live* traffic log the API itself writes, so
  drift detection doesn't need a second data pipeline.
- **Orchestration (Prefect)** and **CI/CD (GitHub Actions)** both wrap the
  same underlying scripts but serve different purposes: the DAG is "run the
  whole thing end to end from raw data to a registered candidate with one
  command"; CI is "catch a regression on every push/PR" and runs much
  faster because it checks a small committed snapshot
  (`reports/gate_snapshot.json`) instead of the full pipeline.

A rendered PNG/SVG export of this diagram (for the slide deck, where
Mermaid won't render) can be made with the `mermaid-cli` package
(`npx @mermaid-js/mermaid-cli -i docs/architecture.md -o architecture.png`)
or by pasting the block above into <https://mermaid.live>.

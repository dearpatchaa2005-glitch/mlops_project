# import os
# os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import partial_auc
from src.preprocessing.features import (
    CAT_COLS,
    add_patient_normalized_features,
    fit_categorical_dtypes,
    get_feature_columns,
)

SEED = 42
USE_PNORM = False
META_PATH = Path("data/raw/images/metadata.csv")
SPLITS_PATH = Path("data/splits/splits.csv")

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": 1,
    "seed": SEED,
    "verbose": -1,
}
N_ROUNDS = 150


meta = pd.read_csv(META_PATH, low_memory=False)
splits = pd.read_csv(SPLITS_PATH, usecols=["isic_id", "malignant", "split", "fold"])
df = meta.merge(splits, on="isic_id", how="inner")

# normalize ภายในผู้ป่วย: ค่ารอยโรค - ค่าเฉลี่ยของผู้ป่วยคนเดียวกัน (ไม่ใช้ label)
# (shared with src/evaluation/oof_fairness.py and fairness_report.py via
#  src/preprocessing/features.py -- single definition, used everywhere)
df = add_patient_normalized_features(df, USE_PNORM)
df = fit_categorical_dtypes(df, CAT_COLS)

features = get_feature_columns(df)
trainval = df[df["split"] == "trainval"].reset_index(drop=True)
print("features:", len(features), "| trainval rows:", len(trainval))

# mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("skin_lesion_tabular_baseline")

# with mlflow.start_run(run_name="lgbm_baseline"):
# with mlflow.start_run(run_name="lgbm_no_xyz"):
# with mlflow.start_run(run_name="lgbm_simple"):
# with mlflow.start_run(run_name="lgbm_simple_spw1"):
# with mlflow.start_run(run_name="lgbm_simple_spw1_no_nevi"):
# with mlflow.start_run(run_name="lgbm_simple_spw1_no_nevi_pnorm"):
with mlflow.start_run(run_name="lgbm_simple_spw1_no_nevi_final"):
    mlflow.log_params({**PARAMS, "n_rounds": N_ROUNDS, "n_features": len(features)})
    # Data version: MLflow auto-tags the code's git commit
    # (mlflow.source.git.commit) but has no notion of *data* version, and
    # data/ is gitignored so there's nothing for git to hash anyway. A
    # content hash of the exact splits file used is the cheapest thing
    # that actually identifies "which data" a run was trained on --
    # re-running make_splits.py with a changed GT file or TEST_FRACTION
    # changes this hash even if the filename doesn't change.
    import hashlib
    splits_hash = hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest()[:16]
    mlflow.set_tags({
        "data.splits_sha256_16": splits_hash,
        "data.splits_path": str(SPLITS_PATH),
        "data.trainval_rows": len(trainval),
    })
    mlflow.log_artifact("requirements/base.txt", artifact_path="environment")
    paucs, prs = [], []

    for fold in range(5):
        tr = trainval[trainval["fold"] != fold]
        va = trainval[trainval["fold"] == fold]
        model = lgb.train(
            PARAMS,
            lgb.Dataset(tr[features], tr["malignant"]),
            num_boost_round=N_ROUNDS,
        )

        model_dir = Path("artifacts/models") / mlflow.active_run().info.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir / f"fold_{fold}.txt"))

        pred = model.predict(va[features])
        pauc = partial_auc(va["malignant"], pred)
        pr = average_precision_score(va["malignant"], pred)
        paucs.append(pauc)
        prs.append(pr)
        mlflow.log_metrics({"pauc": pauc, "pr_auc": pr}, step=fold)
        print(f"fold {fold}: pAUC={pauc:.4f}  PR-AUC={pr:.4f}")

    mlflow.log_artifacts(str(model_dir), artifact_path="models")
    mlflow.log_metrics({
        "pauc_mean": float(np.mean(paucs)), "pauc_std": float(np.std(paucs)),
        "pr_auc_mean": float(np.mean(prs)), "pr_auc_std": float(np.std(prs)),
    })
    print(f"MEAN pAUC={np.mean(paucs):.4f} (+/- {np.std(paucs):.4f})  "
          f"PR-AUC={np.mean(prs):.4f} (+/- {np.std(prs):.4f})")
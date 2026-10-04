import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import partial_auc
from src.evaluation.run_context import RUN_ID

PRED = Path("artifacts/evaluation") / RUN_ID / "test_predictions.csv"

pred = pd.read_csv(PRED)
meta = pd.read_csv("data/raw/images/metadata.csv", usecols=["isic_id", "sex", "age_approx"],
                   low_memory=False)
df = pred.merge(meta, on="isic_id", how="left")
df["sex"] = df["sex"].fillna("unknown")
df["age_group"] = pd.cut(df["age_approx"], bins=[0, 50, 65, 90],
                         labels=["<=50", "51-65", ">65"]).astype(str)


def report(col):
    print(col)
    for name, g in df.groupby(col):
        n_pos = int(g["malignant"].sum())
        if n_pos < 5 or n_pos == len(g):
            print(f"  {name:10s} rows={len(g):6d} malignant={n_pos:3d}  (too few, skipped)")
            continue
        pauc = partial_auc(g["malignant"], g["prediction_score"])
        auc = roc_auc_score(g["malignant"], g["prediction_score"])
        print(f"  {name:10s} rows={len(g):6d} malignant={n_pos:3d}  pAUC={pauc:.4f}  ROC-AUC={auc:.3f}")
    print()


report("sex")
report("age_group")
import pandas as pd
from sklearn.metrics import roc_auc_score

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
sp = pd.read_csv("data/splits/splits.csv", usecols=["isic_id", "malignant", "split"])
df = meta.merge(sp, on="isic_id")
df = df[df["split"] == "trainval"]
y = df["malignant"]

cols = ["tbp_lv_nevi_confidence", "clin_size_long_diam_mm", "tbp_lv_areaMM2",
        "tbp_lv_norm_color", "tbp_lv_deltaLBnorm", "age_approx"]
for c in cols:
    s = df[c].fillna(df[c].median())
    auc = roc_auc_score(y, s)
    print(f"{c:28s} ROC-AUC={max(auc, 1 - auc):.3f}  (direction: {'+' if auc >= 0.5 else '-'})")
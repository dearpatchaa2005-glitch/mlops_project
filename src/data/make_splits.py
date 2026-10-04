from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
TEST_FRACTION = 0.15  # ปรับได้ (เช่น 0.20)
N_FOLDS = 5

META_PATH = Path("data/raw/images/metadata.csv")
GT_PATH = Path("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")
OUT_PATH = Path("data/splits/splits.csv")

meta = pd.read_csv(META_PATH, usecols=["isic_id", "patient_id"])
gt = pd.read_csv(GT_PATH)
df = meta.merge(gt, on="isic_id", how="inner")
df["malignant"] = df["malignant"].astype(int)
df["split"] = ""
df["fold"] = -1

# ขั้น 1: แยก test ออกก่อน (ตามผู้ป่วย) โดยใช้ StratifiedGroupKFold
# ที่จำนวน fold = 1/TEST_FRACTION แล้วหยิบ fold แรกมาเป็น test
n_outer = round(1 / TEST_FRACTION)
outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
trainval_idx, test_idx = next(outer.split(df, df["malignant"], groups=df["patient_id"]))
df.loc[test_idx, "split"] = "test"

# ขั้น 2: ส่วนที่เหลือทำ K-fold CV ตามผู้ป่วย
trainval = df.loc[trainval_idx]
inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (_, val_idx) in enumerate(
    inner.split(trainval, trainval["malignant"], groups=trainval["patient_id"])
):
    df.loc[trainval.index[val_idx], "fold"] = fold
df.loc[trainval_idx, "split"] = "trainval"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("saved:", OUT_PATH, "rows:", len(df))
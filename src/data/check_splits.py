import sys

import pandas as pd

df = pd.read_csv("data/splits/splits.csv")

print("rows per split/fold:")
g = df.groupby(["split", "fold"]).agg(
    rows=("isic_id", "count"),
    patients=("patient_id", "nunique"),
    malignant=("malignant", "sum"),
)
g["malignant_%"] = (g["malignant"] / g["rows"] * 100).round(3)
print(g)
print()

# ผู้ป่วยหนึ่งคนต้องอยู่ได้แค่ test หรือ fold เดียว
key = df["split"] + "_" + df["fold"].astype(str)
leak = (key.groupby(df["patient_id"]).nunique() > 1).sum()
print("patients appearing in more than one group (must be 0):", int(leak))

# Hard gate, not just a printout: a patient leaking across folds/test
# inflates CV scores and is exactly the kind of silent data bug the
# pipeline is supposed to stop on. Non-zero exit fails this step whether
# run by hand, from dags/skin_lesion_pipeline.py, or in CI.
if leak > 0:
    print(f"[FAIL] {leak} patient(s) leak across split/fold boundaries", file=sys.stderr)
    sys.exit(1)
print("[PASS] no patient-level leakage across splits/folds")
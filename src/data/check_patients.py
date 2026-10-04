import pandas as pd

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
gt = pd.read_csv("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")

df = meta[["isic_id", "patient_id"]].merge(gt, on="isic_id", how="inner")
print("rows after merge:", len(df))
print()

per_patient = df.groupby("patient_id").agg(
    lesions=("isic_id", "count"),
    malignant=("malignant", "sum"),
)

print("lesions per patient:")
print(per_patient["lesions"].describe().round(1))
print()
print("patients with >=1 malignant:", int((per_patient["malignant"] > 0).sum()))
print("patients with 0 malignant:  ", int((per_patient["malignant"] == 0).sum()))
print()
print("malignant per patient (patients with >=1):")
print(per_patient.loc[per_patient["malignant"] > 0, "malignant"].describe().round(1))
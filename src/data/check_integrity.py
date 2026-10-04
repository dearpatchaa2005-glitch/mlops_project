from pathlib import Path

import pandas as pd

IMAGES_DIR = Path("data/raw/images")
GT_PATH = Path("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")

df = pd.read_csv(GT_PATH)
csv_ids = set(df["isic_id"])
img_ids = {p.stem for p in IMAGES_DIR.glob("*.jpg")}

print("rows in CSV:        ", len(df))
print("unique IDs in CSV:  ", len(csv_ids))
print("images on disk:     ", len(img_ids))
print("in CSV, no image:   ", len(csv_ids - img_ids))
print("image, not in CSV:  ", len(img_ids - csv_ids))
print()
print("missing values:")
print(df.isna().sum())
print()
print("malignant counts:")
print(df["malignant"].value_counts())
print("malignant ratio: ", round(df["malignant"].mean() * 100, 4), "%")
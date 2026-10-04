import pandas as pd

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
gt = pd.read_csv("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")
df = meta.merge(gt, on="isic_id", how="inner")
df["malignant"] = df["malignant"].astype(int)

print("shape:", df.shape)
print()

# 1) ค่าที่หายไป (แสดงเฉพาะคอลัมน์ที่มีค่าหาย)
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("columns with missing values:")
if len(missing) == 0:
    print("  none")
else:
    print((missing / len(df) * 100).round(2).astype(str) + " %")
print()

# 2) ชนิดข้อมูล
print("dtypes:")
print(df.dtypes.value_counts())
print()

# 3) คอลัมน์หมวดหมู่
for col in ["sex", "anatom_site_general", "image_type", "tbp_tile_type", "tbp_lv_location_simple"]:
    print(col)
    print(df[col].value_counts(dropna=False).to_string())
    print()

# 4) อายุ
print("age_approx:")
print(df["age_approx"].describe().round(1))
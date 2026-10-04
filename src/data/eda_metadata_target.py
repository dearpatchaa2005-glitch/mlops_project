import pandas as pd

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
gt = pd.read_csv("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")
df = meta.merge(gt, on="isic_id", how="inner")
df["malignant"] = df["malignant"].astype(int)


def rate(col):
    g = df.groupby(col, dropna=False)["malignant"].agg(rows="count", malignant="sum")
    g["malignant_%"] = (g["malignant"] / g["rows"] * 100).round(3)
    print(g.to_string())
    print()


for col in ["sex", "anatom_site_general", "tbp_tile_type"]:
    print(col)
    rate(col)

df["age_group"] = pd.cut(df["age_approx"], bins=[0, 30, 40, 50, 60, 70, 90])
print("age_group")
rate("age_group")

# ตัวเลขที่น่าจะสัมพันธ์กับมะเร็ง
num_cols = ["clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_norm_color",
            "tbp_lv_norm_border", "tbp_lv_symm_2axis", "tbp_lv_nevi_confidence"]
print("mean of selected numeric features by malignant:")
print(df.groupby("malignant")[num_cols].mean().round(2).T.to_string())
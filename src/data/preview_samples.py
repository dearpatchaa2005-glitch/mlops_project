from pathlib import Path

import pandas as pd
from PIL import Image

IMAGES_DIR = Path("data/raw/images")
OUT_PATH = Path("reports/figures/sample_grid.png")
N = 12  # จำนวนรูปต่อกลุ่ม
CELL = 160  # ขนาดช่องต่อรูป (พิกเซล)
SEED = 42

df = pd.read_csv("data/splits/splits.csv")
mal = df[df["malignant"] == 1].sample(N, random_state=SEED)
ben = df[df["malignant"] == 0].sample(N, random_state=SEED)

canvas = Image.new("RGB", (N * CELL, 2 * CELL), "white")
for row, group in enumerate([ben, mal]):
    for col, isic_id in enumerate(group["isic_id"]):
        im = Image.open(IMAGES_DIR / f"{isic_id}.jpg").convert("RGB")
        im = im.resize((CELL, CELL))
        canvas.paste(im, (col * CELL, row * CELL))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT_PATH)
print("saved:", OUT_PATH)
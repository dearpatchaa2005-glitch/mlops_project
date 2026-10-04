from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGES_DIR = Path("data/raw/images")
OUT_PATH = Path("data/processed/image_stats.csv")


def scan(path_str):
    p = Path(path_str)
    try:
        with Image.open(p) as im:
            im.load()
            return p.stem, im.width, im.height, im.mode, ""
    except Exception as e:
        return p.stem, None, None, None, str(e)[:100]


if __name__ == "__main__":
    paths = [str(p) for p in sorted(IMAGES_DIR.glob("*.jpg"))]
    print("scanning:", len(paths))
    with Pool() as pool:
        rows = pool.map(scan, paths, chunksize=500)

    df = pd.DataFrame(rows, columns=["isic_id", "width", "height", "mode", "error"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print("unreadable:", int((df["error"] != "").sum()))
    print("modes:", df["mode"].value_counts().to_dict())
    print("non-square:", int((df["width"] != df["height"]).sum()))
    print("width min/median/max:", df["width"].min(), df["width"].median(), df["width"].max())
import random
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGES_DIR = Path("data/raw/images")
SAMPLE_SIZE = 2000
SEED = 42

paths = sorted(IMAGES_DIR.glob("*.jpg"))
random.seed(SEED)
sample = random.sample(paths, SAMPLE_SIZE)

sizes = Counter()
modes = Counter()
bad = []

for p in sample:
    try:
        with Image.open(p) as im:
            im.load()  # บังคับถอดรหัสทั้งรูป เพื่อจับไฟล์เสียจริงๆ
            sizes[im.size] += 1
            modes[im.mode] += 1
    except Exception as e:
        bad.append((p.name, str(e)))

print("sampled:", len(sample))
print("modes:", dict(modes))
print("distinct sizes:", len(sizes))
print("most common sizes (width, height):")
for size, n in sizes.most_common(5):
    print("  ", size, n)
print("min/max width: ", min(s[0] for s in sizes), max(s[0] for s in sizes))
print("min/max height:", min(s[1] for s in sizes), max(s[1] for s in sizes))
print("unreadable files:", len(bad))
for name, err in bad[:5]:
    print("  ", name, err)
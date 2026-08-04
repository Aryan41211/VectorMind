from PIL import Image
import os

images_dir = "data/raw/flickr30k/images"
corrupted = []
total = 0
for f in sorted(os.listdir(images_dir)):
    if not f.endswith(".jpg"):
        continue
    total += 1
    try:
        img = Image.open(os.path.join(images_dir, f))
        img.verify()
    except Exception as e:
        corrupted.append((f, str(e)))
        print(f"CORRUPT: {f}: {e}")

if not corrupted:
    print(f"All {total} images verified OK")
else:
    print(f"{len(corrupted)} corrupted images found out of {total}")

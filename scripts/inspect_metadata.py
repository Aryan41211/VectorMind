"""Inspect Flickr30k metadata structure."""
import json

with open("data/raw/flickr30k/captions.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"First entry keys: {list(data[0].keys())}")
print("Sample entry:")
for k, v in data[0].items():
    if isinstance(v, str) and len(v) > 100:
        print(f"  {k}: {v[:100]}...")
    else:
        print(f"  {k}: {v}")

# Check if filenames are consistent
filenames = [entry.get("filename", entry.get("image", "")) for entry in data[:10]]
print(f"\nFirst 10 filenames: {filenames}")

# Check captions per image
from collections import Counter

caption_counts = Counter(entry.get("filename", entry.get("image", "")) for entry in data)
print(f"\nCaptions per image (first 5): {dict(list(caption_counts.items())[:5])}")
print(f"Unique images: {len(caption_counts)}")

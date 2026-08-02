"""Check all splits in lmms-lab/flickr30k."""
from datasets import load_dataset

# Load without split to see all splits
ds = load_dataset("lmms-lab/flickr30k")
print("Available splits:", list(ds.keys()))
for split_name in ds:
    print(f"\nSplit '{split_name}': {len(ds[split_name])} examples")
    sample = ds[split_name][0]
    print(f"  Keys: {list(sample.keys())}")
    cap = sample["caption"]
    print(f"  Caption type: {type(cap)}, len: {len(cap)}")
    print(f"  Caption[0][:80]: {cap[0][:80]}")
    img = sample["image"]
    print(f"  Image size: {img.size}")

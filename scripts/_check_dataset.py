"""Verify lmms-lab/flickr30k caption structure."""
from datasets import load_dataset

ds = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
sample = next(iter(ds))
print(f"Keys: {list(sample.keys())}")
print(f"Caption type: {type(sample['caption'])}")
print(f"Caption count: {len(sample['caption'])}")
for i, cap in enumerate(sample["caption"]):
    print(f"  Caption {i}: {cap[:100]}...")
print(f"Filename: {sample['filename']}")
print(f"img_id: {sample['img_id']}")
print(f"sentids: {sample['sentids']}")
print(f"Image mode: {sample['image'].mode}, size: {sample['image'].size}")

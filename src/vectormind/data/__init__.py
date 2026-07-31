"""Data pipeline: Flickr30k dataset loading, transforms, and tokenization.

Populated in Phase 1 (ROADMAP.md). Contains the paired
image-caption Dataset/DataLoader implementation and train/val/test
splitting logic with zero image leakage across splits.
"""

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.dataset import Flickr30kDataset
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms

__all__ = [
    "CaptionTokenizer",
    "Flickr30kDataset",
    "create_dataloaders",
    "create_splits",
    "get_eval_transforms",
    "get_train_transforms",
]

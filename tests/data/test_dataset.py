"""Unit tests for vectormind.data.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vectormind.data.dataset import Flickr30kDataset


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create 3 small test images in tmp_path."""
    paths = []
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(i * 50, 100, 200 - i * 30))
        path = tmp_path / f"img_{i}.jpg"
        img.save(path)
        paths.append(path)
    return paths


@pytest.fixture
def sample_captions() -> list[str]:
    """3 sample captions."""
    return [
        "A person walking a dog in the park.",
        "A cat sleeping on a warm windowsill.",
        "Children playing in the backyard.",
    ]


@pytest.fixture
def mock_transform():
    """A simple transform that converts PIL to a [3, 64, 64] tensor."""

    def transform(img: Image.Image) -> torch.Tensor:
        return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

    return transform


@pytest.fixture
def mock_tokenizer():
    """A minimal mock tokenizer for testing."""

    class MockTokenizer:
        def encode(self, text: str):
            # Simple: just return input_ids as [1, 8] and attention_mask as [1, 8].
            ids = torch.tensor([[1] * min(len(text.split()), 8)])
            mask = torch.ones_like(ids)
            # Pad to length 8.
            if ids.shape[1] < 8:
                pad_len = 8 - ids.shape[1]
                ids = torch.cat([ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
                mask = torch.cat(
                    [mask, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
                )
            return {"input_ids": ids, "attention_mask": mask}

        def decode(self, token_ids):
            return [
                f"decoded_{i}"
                for i in range(token_ids.shape[0] if token_ids.dim() > 1 else 1)
            ]

        def __len__(self):
            return 8

    return MockTokenizer()


def test_dataset_length(
    sample_images: list[Path],
    sample_captions: list[str],
    mock_transform,
    mock_tokenizer,
) -> None:
    """Dataset length should equal the number of pairs."""
    dataset = Flickr30kDataset(
        image_paths=sample_images,
        captions=sample_captions,
        transform=mock_transform,
        tokenizer=mock_tokenizer,
        max_text_length=8,
    )
    assert len(dataset) == 3


def test_dataset_getitem_returns_correct_keys(
    sample_images: list[Path],
    sample_captions: list[str],
    mock_transform,
    mock_tokenizer,
) -> None:
    """__getitem__ should return a dict with expected keys."""
    dataset = Flickr30kDataset(
        image_paths=sample_images,
        captions=sample_captions,
        transform=mock_transform,
        tokenizer=mock_tokenizer,
        max_text_length=8,
    )
    item = dataset[0]

    assert "image" in item
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "caption_text" in item


def test_dataset_getitem_image_shape(
    sample_images: list[Path],
    sample_captions: list[str],
    mock_transform,
    mock_tokenizer,
) -> None:
    """Image tensor should have shape [3, H, W]."""
    dataset = Flickr30kDataset(
        image_paths=sample_images,
        captions=sample_captions,
        transform=mock_transform,
        tokenizer=mock_tokenizer,
        max_text_length=8,
    )
    item = dataset[0]

    assert item["image"].shape == (3, 64, 64)


def test_dataset_getitem_caption_text(
    sample_images: list[Path],
    sample_captions: list[str],
    mock_transform,
    mock_tokenizer,
) -> None:
    """caption_text should be the original caption string."""
    dataset = Flickr30kDataset(
        image_paths=sample_images,
        captions=sample_captions,
        transform=mock_transform,
        tokenizer=mock_tokenizer,
        max_text_length=8,
    )
    item = dataset[0]

    assert item["caption_text"] == sample_captions[0]


def test_dataset_raises_on_length_mismatch() -> None:
    """Flickr30kDataset should raise ValueError if paths/captions differ."""
    with pytest.raises(ValueError, match="must equal"):
        Flickr30kDataset(
            image_paths=[Path("a.jpg"), Path("b.jpg")],
            captions=["only one"],
            transform=None,
            tokenizer=None,
            max_text_length=8,
        )


def test_dataset_raises_on_empty() -> None:
    """Flickr30kDataset should raise ValueError if inputs are empty."""
    with pytest.raises(ValueError, match="non-empty"):
        Flickr30kDataset(
            image_paths=[],
            captions=[],
            transform=None,
            tokenizer=None,
            max_text_length=8,
        )


def test_dataset_missing_image_raises(
    sample_captions: list[str],
    mock_transform,
    mock_tokenizer,
) -> None:
    """Flickr30kDataset should raise if an image file doesn't exist."""
    dataset = Flickr30kDataset(
        image_paths=[Path("nonexistent.jpg")],
        captions=[sample_captions[0]],
        transform=mock_transform,
        tokenizer=mock_tokenizer,
        max_text_length=8,
    )
    with pytest.raises((FileNotFoundError, Exception)):
        dataset[0]

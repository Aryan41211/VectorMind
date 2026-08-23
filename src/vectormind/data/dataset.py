"""Flickr30k paired dataset for contrastive learning.

Purpose: provide a PyTorch Dataset that yields correctly paired
(image, caption) tensors from Flickr30k, one pair per __getitem__
call. Each image appears 5 times (once per caption) — this is the
expected behavior for contrastive training, not a duplication bug.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# One sample: three tensors plus the raw caption text carried through for
# debugging and qualitative analysis.
Sample = dict[str, Tensor | str]

logger = logging.getLogger(__name__)


class Flickr30kDataset(Dataset[Sample]):
    """A paired image-caption dataset for contrastive learning.

    Each item returns one (image, caption) pair. Because Flickr30k
    has 5 captions per image, each image path appears 5 times in the
    dataset with different captions — the DataLoader shuffles these
    naturally.

    Attributes:
        image_paths: List of image file paths (length N, where
            N = number_of_images * 5).
        captions: Corresponding caption strings (length N).
        transform: Image transform pipeline (from transforms.py).
        tokenizer: Caption tokenizer (from tokenizer.py).
        max_text_length: Maximum token sequence length.
    """

    def __init__(
        self,
        image_paths: list[Path],
        captions: list[str],
        transform: Callable[[Image.Image], Tensor],
        tokenizer: Any,
        max_text_length: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_paths: List of paths to image files, one per pair.
                Length must equal ``len(captions)``.
            captions: List of caption strings, one per pair.
            transform: A callable that takes a PIL Image and returns
                a transformed tensor (from ``get_train_transforms`` or
                ``get_eval_transforms``).
            tokenizer: A ``CaptionTokenizer`` instance with an
                ``encode`` method.
            max_text_length: Maximum token sequence length for padding/
                truncation.

        Raises:
            ValueError: If ``image_paths`` and ``captions`` have
                different lengths, or both are empty.

        Assumptions:
            All image paths exist and are readable. The transform
            handles resizing/normalization internally.

        Limitations:
            Images are loaded lazily on each __getitem__ call — this
            is memory-efficient for 30k images but means disk I/O
            happens during training. If this becomes a bottleneck,
            consider pre-caching in a future phase.
        """
        if len(image_paths) != len(captions):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) must equal "
                f"captions length ({len(captions)})."
            )
        if len(image_paths) == 0:
            raise ValueError("image_paths and captions must be non-empty.")

        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        logger.info(
            "Flickr30kDataset initialized with %d pairs (%d images × 5 captions)",
            len(self),
            len(self) // 5 if len(self) >= 5 else len(self),
        )

    def __len__(self) -> int:
        """Return the total number of image-caption pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        """Load and return a single image-caption pair.

        Args:
            idx: Index into the dataset.

        Returns:
            A dictionary with keys:
            - ``"image"``: Tensor of shape ``[3, H, W]`` (after transform)
            - ``"input_ids"``: Tensor of shape ``[max_text_length]``
            - ``"attention_mask"``: Tensor of shape ``[max_text_length]``
            - ``"caption_text"``: The raw caption string (for sanity checks)

        Raises:
            FileNotFoundError: If the image file at ``image_paths[idx]``
                does not exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.

        Assumptions:
            The image file exists and is a standard format (JPEG, PNG).

        Limitations:
            No error recovery for corrupted images — the exception
            propagates to the DataLoader worker, which will crash that
            batch. This is intentional: silent corruption is worse than
            a loud failure during development.
        """
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Lazy image loading — memory-efficient for ~30k images.
        pil_image = Image.open(image_path).convert("RGB")
        image: Tensor | Image.Image = (
            self.transform(pil_image) if self.transform is not None else pil_image
        )

        # Tokenize the caption.
        tokenized = self.tokenizer.encode(caption)

        return {
            "image": image,
            "input_ids": tokenized["input_ids"].squeeze(0),  # [max_length]
            "attention_mask": tokenized["attention_mask"].squeeze(0),  # [max_length]
            "caption_text": caption,
        }

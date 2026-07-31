"""Tokenizer for the VectorMind text encoder.

Purpose: wrap a pretrained BPE tokenizer (HuggingFace) to encode
Flickr30k captions into fixed-length token tensors for the text
encoder (ARCHITECTURE.md §3). The tokenizer is used purely as a
preprocessing utility — no pretrained embeddings or encoder weights
are loaded.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CaptionTokenizer:
    """Encode text captions into padded/truncated token tensors.

    Uses a pretrained BPE tokenizer from HuggingFace for tokenization
    only — the actual text representations are trained from scratch.

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer instance.
        max_length: Maximum sequence length (pad/truncate target).
    """

    def __init__(self, tokenizer_name: str, max_length: int) -> None:
        """Load a pretrained tokenizer.

        Args:
            tokenizer_name: HuggingFace tokenizer identifier, e.g.
                ``"bert-base-uncased"``.
            max_length: Maximum token sequence length. Inputs longer
                than this are truncated; shorter inputs are padded.

        Raises:
            OSError: If the tokenizer cannot be loaded from HuggingFace.

        Assumptions:
            The tokenizer is available locally or downloadable from
            HuggingFace Hub. The ``tokenizers`` and ``transformers``
            packages are installed (requirements.txt).

        Limitations:
            Uses the tokenizer's default truncation/padding strategy
            (right-pad with the model's pad_token_id).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT-2) lack a pad token — use eos.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.warning(
                "Tokenizer %s has no pad_token; falling back to eos_token.",
                tokenizer_name,
            )

        logger.info(
            "Loaded tokenizer %s (vocab_size=%d, max_length=%d)",
            tokenizer_name,
            len(self.tokenizer),
            max_length,
        )

    def encode(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """Encode one or more text strings into token tensors.

        Args:
            text: A single caption or a list of captions.

        Returns:
            A dictionary with keys ``"input_ids"`` and
            ``"attention_mask"``, each a ``torch.Tensor`` of shape
            ``[B, max_length]`` where ``B`` is the batch size (1 for
            a single string).

        Assumptions:
            Input strings are valid UTF-8 text in English (Flickr30k
            captions).

        Limitations:
            Does not return ``token_type_ids`` or ``overflow_to_sample_mapping``.
            Right-padding only.
        """
        encoded = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token tensors back to readable text.

        Args:
            token_ids: A tensor of shape ``[B, max_length]`` or
                ``[max_length]``.

        Returns:
            A list of decoded strings, one per row in the input.

        Assumptions:
            The token IDs are valid outputs of ``self.encode()``.

        Limitations:
            Special tokens ([CLS], [SEP], [PAD]) are included in the
            decoded text by default — this is fine for sanity checks
            where we want to see exactly what the tokenizer produced.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    def __len__(self) -> int:
        """Return the configured max sequence length."""
        return self.max_length

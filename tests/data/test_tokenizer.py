"""Unit tests for vectormind.data.tokenizer."""

from __future__ import annotations

import pytest

from vectormind.data.tokenizer import CaptionTokenizer


@pytest.fixture
def tokenizer() -> CaptionTokenizer:
    """A CaptionTokenizer using bert-base-uncased for testing."""
    return CaptionTokenizer(tokenizer_name="bert-base-uncased", max_length=77)


def test_encode_returns_input_ids_and_attention_mask(
    tokenizer: CaptionTokenizer,
) -> None:
    """encode() should return a dict with 'input_ids' and 'attention_mask'."""
    result = tokenizer.encode("A dog playing in the park.")

    assert "input_ids" in result
    assert "attention_mask" in result


def test_encode_output_shapes(tokenizer: CaptionTokenizer) -> None:
    """encode() should produce tensors of shape [1, max_length]."""
    result = tokenizer.encode("A dog playing in the park.")

    assert result["input_ids"].shape == (1, 77)
    assert result["attention_mask"].shape == (1, 77)


def test_encode_batch_shapes(tokenizer: CaptionTokenizer) -> None:
    """encode() with a list should produce [B, max_length] tensors."""
    texts = ["A dog", "A cat sitting on a mat", "Hello world"]
    result = tokenizer.encode(texts)

    assert result["input_ids"].shape == (3, 77)
    assert result["attention_mask"].shape == (3, 77)


def test_encode_padding(tokenizer: CaptionTokenizer) -> None:
    """Shorter texts should be padded with zeros (pad_token_id)."""
    result = tokenizer.encode("Hi")

    # The attention mask should have 0s for padding positions.
    mask = result["attention_mask"][0]
    num_real_tokens = mask.sum().item()
    num_padding = (mask == 0).sum().item()

    assert num_real_tokens + num_padding == 77
    assert num_padding > 0  # "Hi" is much shorter than 77.


def test_encode_truncation(tokenizer: CaptionTokenizer) -> None:
    """Long texts should be truncated to max_length."""
    long_text = "word " * 200  # 200 words, way more than 77 tokens.
    result = tokenizer.encode(long_text)

    assert result["input_ids"].shape == (1, 77)
    # Attention mask should be all 1s (no padding needed for long text).
    assert result["attention_mask"][0].sum().item() == 77


def test_decode_roundtrip(tokenizer: CaptionTokenizer) -> None:
    """decode(encode(text)) should recover the original text (approx)."""
    original = "A dog playing in the park."
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded["input_ids"])

    assert len(decoded) == 1
    # The decoded text should contain the original words (may have
    # special tokens like [CLS], [SEP], [PAD]).
    assert "dog" in decoded[0]
    assert "park" in decoded[0]


def test_decode_batch(tokenizer: CaptionTokenizer) -> None:
    """decode() with a batch should return a list of strings."""
    texts = ["Hello", "World"]
    encoded = tokenizer.encode(texts)
    decoded = tokenizer.decode(encoded["input_ids"])

    assert isinstance(decoded, list)
    assert len(decoded) == 2
    assert all(isinstance(d, str) for d in decoded)


def test_len_returns_max_length(tokenizer: CaptionTokenizer) -> None:
    """__len__ should return max_length."""
    assert len(tokenizer) == 77


def test_custom_max_length() -> None:
    """CaptionTokenizer should respect a custom max_length."""
    tok = CaptionTokenizer(tokenizer_name="bert-base-uncased", max_length=32)
    result = tok.encode("Short text")

    assert result["input_ids"].shape == (1, 32)
    assert len(tok) == 32

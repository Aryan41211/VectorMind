"""Text encoder: Transformer encoder for caption feature extraction.

Purpose: encode tokenized captions into fixed-dimensional feature vectors
that the projection head maps into the shared embedding space
(ARCHITECTURE.md §3).

Design decisions (locked in ARCHITECTURE.md §3, §8):
- Small Transformer encoder chosen over LSTM because self-attention
  handles variable-length captions without sequential bottlenecks and
  demonstrates modern architecture understanding.
- No pretrained embeddings or weights — token embeddings and positional
  embeddings are learned from scratch.
- Mean pooling (not CLS token) produces the pooled representation,
  which is more robust to variable-length inputs.
- Configured via configs/model.yaml — no hardcoded hyperparameters.

Input:  token IDs [B, max_seq_len] (from bert-base-uncased tokenizer)
Output: [B, embed_dim] (mean-pooled feature vector)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm architecture.

    Uses Layer Normalization before attention and FFN (pre-norm) rather
    than after (post-norm), which tends to train more stably at this
    scale.

    Attributes:
        attn_norm: Layer normalization before multi-head attention.
        attention: Multi-head self-attention.
        ffn_norm: Layer normalization before feed-forward network.
        ffn: Feed-forward network (two linear layers with GELU).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        """Initialize a Transformer encoder block.

        Args:
            embed_dim: Dimension of input/output embeddings.
            num_heads: Number of attention heads.
            ffn_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout probability applied after attention and FFN.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``num_heads``.

        Assumptions:
            All dimensions are positive and compatible. This block
            does not handle masking — the caller must provide the
            appropriate attention mask.

        Limitations:
            No rotary or learned relative positional embeddings. Uses
            standard absolute learned positional embeddings added at the
            token embedding level (see ``TextEncoder``).
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # Pre-norm: LayerNorm -> Attention -> Residual
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm: LayerNorm -> FFN -> Residual
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape ``[B, seq_len, embed_dim]``.
            attn_mask: Optional attention mask of shape
                ``[B * num_heads, seq_len, seq_len]`` or
                ``[seq_len, seq_len]``. Masks out positions that
                should not be attended to. ``True`` means masked.
            key_padding_mask: Optional mask of shape ``[B, seq_len]``.
                ``True`` means the position is padding and should be
                ignored.

        Returns:
            Output tensor of shape ``[B, seq_len, embed_dim]``.
        """
        # Self-attention with pre-norm
        normed = self.attn_norm(x)
        attn_out, _ = self.attention(
            query=normed,
            key=normed,
            value=normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Feed-forward network with pre-norm
        x = x + self.ffn(self.ffn_norm(x))

        return x


class TextEncoder(nn.Module):
    """Transformer text encoder for caption feature extraction.

    Encodes tokenized text into fixed-dimensional feature vectors via
    learned token and positional embeddings, a stack of Transformer
    blocks, and mean pooling over the sequence dimension.

    Architecture:
        Token Embedding + Positional Embedding -> Dropout ->
        N x TransformerBlock -> Mean Pool (over non-padding positions) ->
        [B, embed_dim]

    Attributes:
        vocab_size: Size of the vocabulary.
        max_seq_len: Maximum sequence length.
        embed_dim: Dimension of token and positional embeddings.
        num_layers: Number of Transformer blocks.
        token_embedding: Token embedding layer.
        position_embedding: Learned positional embedding.
        dropout: Dropout after embedding sum.
        blocks: Stack of Transformer blocks.
        final_norm: Final layer normalization.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the text encoder from configuration.

        Args:
            config: Model configuration dictionary loaded from
                ``configs/model.yaml``. Must contain the key
                ``text_encoder`` with sub-keys ``vocab_size``,
                ``max_seq_len``, ``embed_dim``, ``num_layers``,
                ``num_heads``, ``ffn_dim``, and ``dropout``.

        Raises:
            KeyError: If required config keys are missing.
            ValueError: If dimensions are incompatible.

        Assumptions:
            The config has been validated by ``utils.config.require_keys``
            before being passed here. The tokenizer (from Phase 1) uses
            the same vocab_size specified here (bert-base-uncased: 30522).

        Limitations:
            No rotary positional embeddings (RoPE) or ALiBi — standard
            learned absolute positions are simpler and sufficient at
            this project's scale.
        """
        super().__init__()

        txt_cfg = config["text_encoder"]
        self.vocab_size: int = txt_cfg["vocab_size"]
        self.max_seq_len: int = txt_cfg["max_seq_len"]
        self.embed_dim: int = txt_cfg["embed_dim"]
        self.num_layers: int = txt_cfg["num_layers"]

        num_heads: int = txt_cfg["num_heads"]
        ffn_dim: int = txt_cfg["ffn_dim"]
        dropout: float = txt_cfg["dropout"]

        # Learned embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Final layer normalization
        self.final_norm = nn.LayerNorm(self.embed_dim)

        # Initialize weights
        self._init_weights()

        logger.info(
            "TextEncoder initialized: vocab_size=%d, max_seq_len=%d, "
            "embed_dim=%d, num_layers=%d, num_heads=%d, ffn_dim=%d",
            self.vocab_size,
            self.max_seq_len,
            self.embed_dim,
            self.num_layers,
            num_heads,
            ffn_dim,
        )

    def _init_weights(self) -> None:
        """Initialize embedding and linear-layer weights.

        Normal distribution for embeddings, Xavier uniform for linear
        layers. This is the standard initialization for Transformer encoders
        and ensures stable forward passes from the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal attention mask (not used for bidirectional encoder).

        This is a utility for future extensibility. The standard text
        encoder uses bidirectional attention (no causal mask), so this
        method is not called in the default forward pass.

        Args:
            seq_len: Sequence length.
            device: Device for the mask tensor.

        Returns:
            Upper-triangular mask of shape ``[seq_len, seq_len]``.
            ``True`` means masked (not attended to).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokenized text into feature vectors.

        Args:
            input_ids: Token IDs of shape ``[B, max_seq_len]``.
            attention_mask: Optional mask of shape ``[B, max_seq_len]``.
                ``1`` indicates real tokens, ``0`` indicates padding.
                If ``None``, all positions are treated as real tokens.

        Returns:
            Pooled feature vectors of shape ``[B, embed_dim]`` where
            ``embed_dim`` is typically 256.

        Raises:
            RuntimeError: If input tensor shape is incompatible with
                the embedding dimensions.
        """
        B, seq_len = input_ids.shape

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Build key_padding_mask: True means "this position is padding"
        # Attention mask from tokenizer: 1 = real, 0 = pad
        # PyTorch key_padding_mask: True = ignore (mask out)
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # invert: 0-padding -> True

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Mean pooling over non-padding positions
        pooled = self._mean_pool(x, attention_mask)

        return pooled

    def _mean_pool(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute mean pooling over sequence positions.

        Averages the token representations across the sequence dimension,
        ignoring padding positions. This produces a single vector per
        sequence regardless of the actual caption length.

        Args:
            x: Token representations of shape ``[B, seq_len, embed_dim]``.
            attention_mask: Optional mask of shape ``[B, seq_len]``.
                ``1`` indicates real tokens, ``0`` indicates padding.

        Returns:
            Pooled representations of shape ``[B, embed_dim]``.

        Assumptions:
            Each sequence has at least one non-padding token (attention
            mask contains at least one ``1`` per row). The caller
            (CaptionTokenizer) ensures this via right-padding with a
            valid pad token.
        """
        if attention_mask is not None:
            # Expand mask to [B, seq_len, 1] for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum embeddings and divide by number of real tokens
            sum_embeddings = (x * mask_expanded).sum(dim=1)
            token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / token_counts
        else:
            # No mask: simple mean over all positions
            return x.mean(dim=1)

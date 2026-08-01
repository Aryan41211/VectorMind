"""Projection heads: map encoder outputs to the shared embedding space.

Purpose: both encoder towers produce task-specific feature vectors that
live in different dimensional spaces (image: 512, text: 256). The
projection head is a modality-specific module that maps each tower's
output into the shared D-dimensional embedding space used by the
contrastive loss (ARCHITECTURE.md §4).

Design decisions (locked in ARCHITECTURE.md §4):
- Single linear layer per modality (not a multi-layer MLP).
  At this project's scale (~30k training pairs), a deeper head risks
  overfitting to the projection while adding no measurable benefit.
- L2 normalization applied after projection so that the dot product
  between embeddings equals cosine similarity — this is what the
  InfoNCE contrastive loss operates on.
- Each modality gets its own projection head (swappable per
  ARCHITECTURE.md §2).
- Configured via configs/model.yaml embedding.shared_dim — no hardcoded
  dimensions.

Input:  encoder feature vectors [B, input_dim]
Output: L2-normalized embeddings [B, shared_dim]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Linear projection head with L2 normalization.

    Maps encoder outputs to the shared embedding space and normalizes
    them so that dot products correspond to cosine similarity.

    Attributes:
        projection: Linear layer mapping input_dim -> shared_dim.
        input_dim: Dimension of the encoder output fed into this head.
        shared_dim: Dimension of the shared embedding space.
    """

    def __init__(self, input_dim: int, shared_dim: int) -> None:
        """Initialize a projection head.

        Args:
            input_dim: Dimension of the input features from the encoder.
            shared_dim: Dimension of the target embedding space.

        Raises:
            ValueError: If ``input_dim`` or ``shared_dim`` is not
                positive.

        Assumptions:
            The caller is responsible for choosing a head per modality
            with the correct ``input_dim`` (512 for image, 256 for text).

        Limitations:
            Single linear layer only — no configurable depth or
            activation (YAGNI, as specified in ARCHITECTURE.md §4).
        """
        super().__init__()

        if input_dim <= 0 or shared_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive, got "
                f"input_dim={input_dim}, shared_dim={shared_dim}."
            )

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.projection = nn.Linear(input_dim, shared_dim)

        self._init_weights()

        logger.info(
            "ProjectionHead initialized: input_dim=%d, shared_dim=%d",
            input_dim,
            shared_dim,
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for the linear layer.

        Standard initialization for projection layers, consistent
        with the text encoder's linear layer initialization.
        """
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize the input features.

        Args:
            x: Encoder output of shape ``[B, input_dim]``.

        Returns:
            L2-normalized embeddings of shape ``[B, shared_dim]``.

        Raises:
            RuntimeError: If the last dimension of ``x`` does not
                match ``input_dim``.
        """
        projected: torch.Tensor = self.projection(x)
        normalized: torch.Tensor = nn.functional.normalize(projected, p=2, dim=-1)
        return normalized

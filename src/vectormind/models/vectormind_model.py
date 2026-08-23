"""VectorMind dual-encoder model: assembles towers + projection heads.

Purpose: top-level model that combines the image encoder, text encoder,
and modality-specific projection heads into a single nn.Module. Provides
separate encode_image() and encode_text() APIs for producing shared-
dimension embeddings, plus a learnable temperature parameter for the
contrastive loss (ARCHITECTURE.md §1, §7).

Design decisions (locked in ARCHITECTURE.md §7):
- Separate encode_image() / encode_text() rather than a combined
  forward(). The contrastive loss needs both modalities' embeddings
  but computes similarity itself; the model just produces the
  normalized vectors. The combined forward logic lives in the
  training loop (training/train_loop.py).
- Learnable temperature (not a fixed constant) — initialized via
  CLIP convention: log(1/0.07). Letting the model calibrate
  similarity sharpness is strictly better than hand-tuning a
  constant.
- All components (encoders, heads) are independently swappable
  behind defined interfaces (CLAUDE.md §2).

Input:
  - encode_image(): [B, 3, 224, 224]
  - encode_text():  token_ids [B, 77], attention_mask [B, 77] (optional)
Output:
  - Both return L2-normalized [B, 256] embeddings in the shared space.

Dependencies: ImageEncoder, TextEncoder, ProjectionHead (all from
src/vectormind.models). No dependency on training/ or data/
(FOLDER_STRUCTURE.md).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from vectormind.models.image_encoder import ImageEncoder
from vectormind.models.projection_head import ProjectionHead
from vectormind.models.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

# CLIP convention: temperature = log(1/0.07)
_INITIAL_LOG_TEMPERATURE: float = 2.659260036931346

# CLIP clamps its learnable logit scale at 100. Without a ceiling the
# optimizer can minimize contrastive loss by inflating this scalar
# instead of separating representations — which is exactly what Phase 4
# of this project did (14.3 -> 55 -> 500+, with the embedding space
# collapsing into a narrow cone). See docs/KNOWN_ISSUES.md §1.
DEFAULT_MAX_LOGIT_SCALE: float = 100.0


class VectorMindModel(nn.Module):
    """CLIP-style dual-encoder with shared embedding space.

    Assembles the image encoder, text encoder, modality-specific
    projection heads, and a learnable temperature into a single
    nn.Module. Exposes separate encode_image() and encode_text()
    methods that return L2-normalized embeddings in the shared
    embedding space.

    Attributes:
        image_encoder: CNN-based image feature extractor.
        text_encoder: Transformer-based text feature extractor.
        image_projection: Projection head mapping image features to
            the shared embedding space.
        text_projection: Projection head mapping text features to
            the shared embedding space.
        log_temperature: Learnable scalar (log space) controlling
            the sharpness of the similarity distribution.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize VectorMindModel from configuration.

        Args:
            config: Full model configuration dictionary loaded from
                ``configs/model.yaml``. Must contain ``image_encoder``,
                ``text_encoder``, and ``embedding`` sections with the
                required sub-keys:
                - image_encoder: in_channels, base_channels, output_dim
                - text_encoder: vocab_size, max_seq_len, embed_dim,
                  num_layers, num_heads, ffn_dim, dropout
                - embedding: shared_dim

        Raises:
            KeyError: If required config keys are missing.
            ValueError: If dimensions are incompatible (e.g.
                shared_dim mismatch between towers).

        Assumptions:
            The config has been validated by ``utils.config.require_keys``
            before being passed here. Image encoder output_dim (512)
            and text encoder output_dim (256) are both projected to
            the same shared_dim (256).

        Limitations:
            Temperature is a single scalar (per CLIP convention), not
            per-modality.
        """
        super().__init__()

        img_cfg = config["image_encoder"]
        txt_cfg = config["text_encoder"]
        emb_cfg = config["embedding"]

        shared_dim: int = emb_cfg["shared_dim"]

        # Encoders
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)

        # Projection heads — image encoder outputs img_cfg["output_dim"],
        # text encoder outputs txt_cfg["embed_dim"]
        self.image_projection = ProjectionHead(
            input_dim=img_cfg["output_dim"],
            shared_dim=shared_dim,
        )
        self.text_projection = ProjectionHead(
            input_dim=txt_cfg["embed_dim"],
            shared_dim=shared_dim,
        )

        # Learnable temperature (scalar, log space, CLIP init)
        self.log_temperature = nn.Parameter(torch.tensor(_INITIAL_LOG_TEMPERATURE))

        self._log_model_info(shared_dim)

    def _log_model_info(self, shared_dim: int) -> None:
        """Log model initialization summary.

        Args:
            shared_dim: Dimension of the shared embedding space.
        """
        img_out = self.image_encoder.output_dim
        txt_out = self.text_encoder.embed_dim
        n_params = sum(p.numel() for p in self.parameters())

        logger.info(
            "VectorMindModel initialized: "
            "image_encoder_out=%d, text_encoder_out=%d, "
            "shared_dim=%d, temperature_init=%.4f, "
            "total_params=%d",
            img_out,
            txt_out,
            shared_dim,
            self.log_temperature.exp().item(),
            n_params,
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value (exponentiated from log space).

        Returns:
            Scalar temperature tensor.
        """
        return self.log_temperature.exp()  # type: ignore[no-any-return]

    @torch.no_grad()
    def clamp_log_temperature(
        self,
        max_logit_scale: float = DEFAULT_MAX_LOGIT_SCALE,
    ) -> float:
        """Clamp the learnable logit scale to an upper bound, in place.

        Call this after every ``optimizer.step()``. The scalar the
        contrastive loss multiplies by is CLIP's ``logit_scale``, and it
        is an unconstrained parameter: driving it upward lowers the loss
        without improving the representation. Left unbounded in this
        project's Phase 4 run it reached 500+, and the embedding space
        collapsed into a narrow cone while Recall@K still looked
        acceptable. CLIP's own implementation clamps at 100.

        Args:
            max_logit_scale: Ceiling on ``exp(log_temperature)``. Must
                be greater than 1.0 — a ceiling at or below 1 would
                force similarities to be attenuated rather than scaled.

        Returns:
            The logit scale after clamping, as a float.

        Raises:
            ValueError: If ``max_logit_scale`` is not greater than 1.0.

        Assumptions:
            Called outside the autograd graph — the decorator enforces
            this, so the clamp does not appear as an operation in the
            next backward pass.
        """
        if max_logit_scale <= 1.0:
            raise ValueError(
                f"max_logit_scale must be > 1.0, got {max_logit_scale}."
            )

        ceiling = math.log(max_logit_scale)
        self.log_temperature.clamp_(max=ceiling)
        return float(self.log_temperature.exp().item())

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images into L2-normalized shared-space embeddings.

        Args:
            images: Batch of images, shape ``[B, 3, H, W]`` (typically
                ``[B, 3, 224, 224]``).

        Returns:
            L2-normalized embeddings of shape ``[B, shared_dim]``.

        Raises:
            RuntimeError: If input tensor shape is incompatible with
                the image encoder.
        """
        features: torch.Tensor = self.image_encoder(images)
        embeddings: torch.Tensor = self.image_projection(features)
        return embeddings

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokenized text into L2-normalized shared-space embeddings.

        Args:
            input_ids: Token IDs of shape ``[B, max_seq_len]``.
            attention_mask: Optional mask of shape ``[B, max_seq_len]``.
                ``1`` = real token, ``0`` = padding. If ``None``, all
                positions are treated as real tokens.

        Returns:
            L2-normalized embeddings of shape ``[B, shared_dim]``.

        Raises:
            RuntimeError: If input tensor shape is incompatible with
                the text encoder.
        """
        features: torch.Tensor = self.text_encoder(input_ids, attention_mask)
        embeddings: torch.Tensor = self.text_projection(features)
        return embeddings

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode both modalities and return embeddings plus temperature.

        This method is provided as a convenience for the training loop
        and evaluation scripts. For separate modality encoding, use
        encode_image() or encode_text().

        Args:
            images: Batch of images, shape ``[B, 3, H, W]``.
            input_ids: Token IDs of shape ``[B, max_seq_len]``.
            attention_mask: Optional mask of shape ``[B, max_seq_len]``.

        Returns:
            Dictionary with keys:
                ``image_embeddings``: [B, shared_dim], L2-normalized.
                ``text_embeddings``: [B, shared_dim], L2-normalized.
                ``temperature``: scalar tensor.
        """
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)

        return {
            "image_embeddings": image_emb,
            "text_embeddings": text_emb,
            "temperature": self.temperature,
        }

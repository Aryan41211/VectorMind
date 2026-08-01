"""Model definitions: image encoder, text encoder, projection heads.

Populated in Phase 2 (see ROADMAP.md). Design principle (CLAUDE.md §2):
image tower, text tower, and projection heads must each be independently
swappable behind defined interfaces — no component may assume a specific
encoder implementation. See ARCHITECTURE.md §2-4 for the chosen designs
and rationale.
"""

from vectormind.models.image_encoder import ImageEncoder
from vectormind.models.projection_head import ProjectionHead
from vectormind.models.text_encoder import TextEncoder
from vectormind.models.vectormind_model import VectorMindModel

__all__ = ["ImageEncoder", "ProjectionHead", "TextEncoder", "VectorMindModel"]

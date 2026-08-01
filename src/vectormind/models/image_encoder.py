"""Image encoder: ResNet-18-style CNN for visual feature extraction.

Purpose: encode images into fixed-dimensional feature vectors that the
projection head maps into the shared embedding space (ARCHITECTURE.md §2).

Design decisions (locked in ARCHITECTURE.md §2, §8):
- ResNet-18-style CNN chosen over ViT-from-scratch because CNNs learn
  more efficiently from small datasets (~30k images) due to
  convolutional inductive bias (locality, translation equivariance).
- No pretrained weights — the entire model is trained from scratch.
- Global average pooling produces a fixed-length feature vector
  regardless of spatial resolution.
- Configured via configs/model.yaml — no hardcoded hyperparameters.

Input:  [B, 3, 224, 224] (RGB images, normalized)
Output: [B, 512] (feature vector after global average pooling)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """ResNet basic block with two 3x3 convolutions and a skip connection.

    For the first block in each stage where spatial dimensions shrink,
    a 1x1 projection in the shortcut handles the dimension mismatch.

    Attributes:
        conv1: First 3x3 convolution with optional stride.
        bn1: Batch normalization after conv1.
        conv2: Second 3x3 convolution.
        bn2: Batch normalization after conv2.
        shortcut: Identity or 1x1 projection for dimension matching.
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize a basic residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Convolution stride for spatial downsampling (2 in
                the first block of each stage, 1 otherwise).

        Raises:
            ValueError: If ``in_channels`` or ``out_channels`` is not positive.

        Assumptions:
            Input spatial dimensions are compatible with the specified
            stride (no explicit size checks — caller ensures this).

        Limitations:
            No dilation support. Expansion factor is always 1.
        """
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"Channel dimensions must be positive, got "
                f"in_channels={in_channels}, out_channels={out_channels}."
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: identity when dimensions match, 1x1 projection otherwise.
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape ``[B, in_channels, H, W]``.

        Returns:
            Output tensor of shape ``[B, out_channels, H', W']`` where
            ``H'`` and ``W'`` depend on the stride.
        """
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))  # type: ignore[no-any-return]
        out = self.bn2(self.conv2(out))
        shortcut: torch.Tensor = self.shortcut(x)  # type: ignore[no-any-return]
        out = out + shortcut
        return self.relu(out)  # type: ignore[no-any-return]


class ImageEncoder(nn.Module):
    """ResNet-18-style CNN image encoder.

    Encodes RGB images into fixed-dimensional feature vectors via a
    series of residual blocks followed by global average pooling.
    The output dimension is determined by the final conv stage
    (default: 512).

    Architecture follows ResNet-18:
        conv1 (7x7, stride 2) -> maxpool -> 4 stages of residual blocks
        -> global average pool -> [B, output_dim]

    Attributes:
        in_channels: Number of input channels (3 for RGB).
        output_dim: Dimension of the output feature vector.
        conv1: Initial 7x7 convolution.
        bn1: Batch normalization after conv1.
        maxpool: Max pooling after conv1.
        stage1: First residual stage (64 channels).
        stage2: Second residual stage (128 channels).
        stage3: Third residual stage (256 channels).
        stage4: Fourth residual stage (512 channels).
        avgpool: Global average pooling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the image encoder from configuration.

        Args:
            config: Model configuration dictionary loaded from
                ``configs/model.yaml``. Must contain the key
                ``image_encoder`` with sub-keys ``in_channels``,
                ``base_channels``, and ``output_dim``.

        Raises:
            KeyError: If required config keys are missing.
            ValueError: If channel dimensions are incompatible.

        Assumptions:
            The config has been validated by ``utils.config.require_keys``
            before being passed here. The ``output_dim`` must equal
            ``base_channels * 8`` for the standard ResNet-18 channel
            progression (64 * 8 = 512).

        Limitations:
            Hardcoded to ResNet-18 topology (4 stages, 2 blocks each).
            For a different depth, this class would need to be refactored
            or generalized (YAGNI — not needed for this project's scope).
        """
        super().__init__()

        img_cfg = config["image_encoder"]
        in_channels: int = img_cfg["in_channels"]
        base_channels: int = img_cfg["base_channels"]
        self.output_dim: int = img_cfg["output_dim"]

        # Validate channel progression: 64 -> 128 -> 256 -> 512
        expected_output = base_channels * 8
        if self.output_dim != expected_output:
            raise ValueError(
                f"output_dim ({self.output_dim}) must equal "
                f"base_channels * 8 ({expected_output}) for ResNet-18. "
                f"Check configs/model.yaml."
            )

        # Stage 0: initial convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 1-4: residual blocks
        # Channel progression: base -> base*2 -> base*4 -> base*8
        self.stage1 = self._make_stage(
            base_channels, base_channels, num_blocks=2, stride=1
        )
        self.stage2 = self._make_stage(
            base_channels, base_channels * 2, num_blocks=2, stride=2
        )
        self.stage3 = self._make_stage(
            base_channels * 2, base_channels * 4, num_blocks=2, stride=2
        )
        self.stage4 = self._make_stage(
            base_channels * 4, base_channels * 8, num_blocks=2, stride=2
        )

        # Global average pooling: [B, C, H, W] -> [B, C]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights (Kaiming init for ReLU networks)
        self._init_weights()

        logger.info(
            "ImageEncoder initialized: in_channels=%d, base_channels=%d, "
            "output_dim=%d",
            in_channels,
            base_channels,
            self.output_dim,
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a residual stage with the specified number of blocks.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels for each block.
            num_blocks: Number of residual blocks in this stage.
            stride: Stride for the first block (subsequent blocks use 1).

        Returns:
            A ``nn.Sequential`` containing ``num_blocks`` BasicBlocks.

        Assumptions:
            ``num_blocks >= 1``.
        """
        blocks: list[BasicBlock] = []
        blocks.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal for Conv2d and ones for BatchNorm.

        This is the standard initialization for ReLU-activated networks
        and ensures stable forward passes from the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into feature vectors.

        Args:
            x: Batch of images, shape ``[B, in_channels, H, W]``.
                Expected to be 224x224 RGB images.

        Returns:
            Feature vectors of shape ``[B, output_dim]`` where
            ``output_dim`` is typically 512.

        Raises:
            RuntimeError: If input tensor shape is incompatible with
                the network (e.g. wrong number of channels).
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

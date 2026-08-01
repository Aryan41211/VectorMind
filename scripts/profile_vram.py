"""Phase 0.2 (ROADMAP.md): empirical VRAM profiling on the RTX 4050.

Purpose
-------
Determine the maximum batch size that fits in 6GB VRAM under mixed
precision, using encoder sizes representative of the planned Phase 2
architecture (ARCHITECTURE.md §2-4), including a contrastive-loss-shaped
forward/backward pass (since the loss's O(batch^2) similarity matrix is
itself a meaningful contributor to peak memory — see ARCHITECTURE.md §6).

This script's encoder/loss classes are intentionally minimal stand-ins
for profiling only. They are NOT the tested, final Phase 2/Phase 3
implementations (those live in src/vectormind/models/ and
src/vectormind/training/ with their own unit tests, per CLAUDE.md §4).

Usage
-----
    python scripts/profile_vram.py --config configs/profiling.yaml

Output
------
Writes measured results to the path in `profiling.results_path`
(default: logs/vram_profile_results.json) and logs progress to both
stdout and `profiling.log_path`.

After running, copy the reported "recommended_batch_size" into
ARCHITECTURE.md §6, replacing the placeholder text there.

Assumptions
-----------
- Run on the target machine (RTX 4050, 6GB VRAM) — results measured on
  any other GPU are not valid for this project's stated constraint.
- CUDA is available; this script raises immediately if it is not,
  rather than silently profiling on CPU (which would produce a
  meaningless and misleadingly large "safe" batch size).

Limitations
-----------
- The stand-in encoders approximate the planned architecture's memory
  footprint but are not byte-for-byte identical to the eventual Phase 2
  implementation. Treat the result as a well-informed empirical ceiling,
  not an exact guarantee — re-profile once the real Phase 2 model exists
  if the batch size matters at the margin.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ImageEncoderStub(nn.Module):
    """Minimal ResNet-18-style CNN stand-in for VRAM profiling.

    Purpose: approximate the memory footprint of the planned Phase 2
    image encoder (ARCHITECTURE.md §2) without depending on the actual
    implementation, which does not exist yet at Phase 0.

    Inputs: a batch of images, shape (batch, in_channels, image_size, image_size).
    Outputs: a pooled feature vector, shape (batch, base_channels * 8).

    Assumptions: input is already normalized/resized upstream.
    Limitations: not a real ResNet-18 (no residual connections); depth
    and channel progression are chosen to be memory-representative,
    not architecturally final.
    """

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_channels, out_channels, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = channels[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(images)
        pooled = self.pool(features).flatten(1)
        return pooled


class TextEncoderStub(nn.Module):
    """Minimal Transformer-encoder stand-in for VRAM profiling.

    Purpose: approximate the memory footprint of the planned Phase 2
    text encoder (ARCHITECTURE.md §3): a from-scratch Transformer with
    learned positional embeddings.

    Inputs: token ids, shape (batch, max_seq_len), dtype long.
    Outputs: a mean-pooled representation, shape (batch, embed_dim).

    Assumptions: token ids are already produced by tokenization
    (Phase 1); this stub uses random token ids for profiling since no
    real tokenizer/data pipeline exists yet at Phase 0.
    Limitations: uses mean pooling rather than a final pooling
    strategy decision, which is still open for Phase 2.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(
            positions
        )
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)
        return pooled


class ProfilingModel(nn.Module):
    """Combines the stub encoders + projection heads + a contrastive-shaped loss.

    Purpose: give the profiler a single forward+backward call that is
    representative of one real training step's memory footprint,
    including the O(batch^2) similarity matrix from the contrastive
    loss (ARCHITECTURE.md §5-6), which in-batch-negative batch-size
    profiling must account for.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_cfg = config["image_encoder"]
        text_cfg = config["text_encoder"]
        shared_dim = config["embedding"]["shared_dim"]

        self.image_encoder = ImageEncoderStub(
            in_channels=image_cfg["in_channels"],
            base_channels=image_cfg["base_channels"],
        )
        self.text_encoder = TextEncoderStub(
            vocab_size=text_cfg["vocab_size"],
            max_seq_len=text_cfg["max_seq_len"],
            embed_dim=text_cfg["embed_dim"],
            num_layers=text_cfg["num_layers"],
            num_heads=text_cfg["num_heads"],
            ffn_dim=text_cfg["ffn_dim"],
        )
        self.image_projection = nn.Linear(self.image_encoder.output_dim, shared_dim)
        self.text_projection = nn.Linear(text_cfg["embed_dim"], shared_dim)
        # Learnable temperature, initialized as in CLIP (ARCHITECTURE.md §5).
        self.logit_scale = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(1 / 0.07))))
        )

        self.image_size = image_cfg["image_size"]
        self.max_seq_len = text_cfg["max_seq_len"]
        self.vocab_size = text_cfg["vocab_size"]

    def forward_and_backward(self, batch_size: int, device: torch.device) -> None:
        """Run one representative forward + backward pass at `batch_size`."""
        images = torch.randn(
            batch_size, 3, self.image_size, self.image_size, device=device
        )
        token_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.max_seq_len),
            device=device,
        )

        image_features = self.image_projection(self.image_encoder(images))
        text_features = self.text_projection(self.text_encoder(token_ids))

        image_embeds = nn.functional.normalize(image_features, dim=-1)
        text_embeds = nn.functional.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        targets = torch.arange(batch_size, device=device)
        loss_i = nn.functional.cross_entropy(logits_per_image, targets)
        loss_t = nn.functional.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2

        loss.backward()


def _try_batch_size(
    model: ProfilingModel,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    warmup_iters: int,
    measure_iters: int,
) -> tuple[bool, float]:
    """Attempt `batch_size`; return (succeeded, peak_memory_bytes).

    On CUDA out-of-memory, returns (False, nan) and clears the cache so
    the next attempt starts from a clean state.

    A batch size is considered successful only if ALL iterations complete
    without OOM and the peak memory is within physical VRAM.
    """
    completed_iters = 0

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(warmup_iters + measure_iters):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                model.forward_and_backward(batch_size, device)
            optimizer.step()
            completed_iters += 1

        peak_memory = float(torch.cuda.max_memory_allocated(device))

        # Sanity check: peak memory must not exceed total VRAM
        total_vram = float(torch.cuda.get_device_properties(device).total_memory)
        if peak_memory > total_vram:
            logger.warning(
                "Batch size %d: peak memory (%.2f GB) exceeds total VRAM (%.2f GB). "
                "Treating as OOM.",
                batch_size,
                peak_memory / 1e9,
                total_vram / 1e9,
            )
            return False, float("nan")

        if completed_iters < warmup_iters + measure_iters:
            logger.warning(
                "Batch size %d: only %d/%d iterations completed.",
                batch_size,
                completed_iters,
                warmup_iters + measure_iters,
            )
            return False, float("nan")

        return True, peak_memory

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, float("nan")


def find_max_batch_size(
    model: ProfilingModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Exponential search then binary search for the max feasible batch size.

    Returns a results dict with the largest successful batch size, its
    peak memory usage, and a recommended batch size after applying the
    configured safety margin.
    """
    profiling_cfg = config["profiling"]
    min_bs = profiling_cfg["min_batch_size"]
    max_bs = profiling_cfg["max_batch_size"]
    use_amp = profiling_cfg["use_amp"]
    warmup_iters = profiling_cfg["warmup_iters"]
    measure_iters = profiling_cfg["measure_iters"]
    safety_margin = profiling_cfg["vram_safety_margin_fraction"]

    total_vram = float(torch.cuda.get_device_properties(device).total_memory)
    logger.info("Total VRAM on device: %.2f GB", total_vram / 1e9)

    # --- Exponential search: find an upper bound that fails ---
    last_success_bs = None
    last_success_peak_mem = None
    current_bs = min_bs
    first_failure_bs = None

    while current_bs <= max_bs:
        logger.info("Trying batch size %d ...", current_bs)
        succeeded, peak_mem = _try_batch_size(
            model, optimizer, current_bs, device, use_amp, warmup_iters, measure_iters
        )
        if succeeded:
            logger.info(
                "Batch size %d succeeded. Peak memory: %.2f GB",
                current_bs,
                peak_mem / 1e9,
            )
            last_success_bs = current_bs
            last_success_peak_mem = peak_mem
            current_bs *= 2
        else:
            logger.info("Batch size %d failed with OOM.", current_bs)
            first_failure_bs = current_bs
            break

    if last_success_bs is None:
        raise RuntimeError(
            f"Even the minimum batch size ({min_bs}) OOM'd. The model as "
            f"configured does not fit in this GPU's VRAM at all — reduce "
            f"encoder sizes in configs/profiling.yaml before re-running."
        )

    # --- Binary search between last success and first failure, if any ---
    if first_failure_bs is not None:
        low, high = last_success_bs, first_failure_bs
        while high - low > 1:
            mid = (low + high) // 2
            logger.info("Binary search: trying batch size %d ...", mid)
            succeeded, peak_mem = _try_batch_size(
                model, optimizer, mid, device, use_amp, warmup_iters, measure_iters
            )
            if succeeded:
                low = mid
                last_success_bs = mid
                last_success_peak_mem = peak_mem
            else:
                high = mid
        logger.info("Binary search converged: max feasible batch size = %d", low)

    assert last_success_peak_mem is not None  # guaranteed by the loop above
    peak_fraction_of_total = last_success_peak_mem / total_vram
    recommended_bs = last_success_bs
    if peak_fraction_of_total > (1 - safety_margin):
        # The largest batch that "succeeded" leaves less headroom than the
        # configured safety margin (needed for the memory queue, dataloader
        # workers, and allocator fragmentation) — step down by one search
        # increment to be conservative.
        recommended_bs = max(min_bs, int(last_success_bs * 0.9))
        logger.warning(
            "Max successful batch size (%d) uses %.1f%% of total VRAM, "
            "exceeding the %.0f%% safety margin. Recommending %d instead.",
            last_success_bs,
            peak_fraction_of_total * 100,
            (1 - safety_margin) * 100,
            recommended_bs,
        )

    return {
        "max_successful_batch_size": last_success_bs,
        "max_successful_peak_memory_bytes": last_success_peak_mem,
        "total_vram_bytes": total_vram,
        "peak_fraction_of_total_vram": peak_fraction_of_total,
        "recommended_batch_size": recommended_bs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/profiling.yaml",
        help="Path to the profiling config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    require_keys(config, ["image_encoder", "text_encoder", "embedding", "profiling"])
    profiling_cfg = config["profiling"]

    setup_logging(log_file=profiling_cfg["log_path"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script must be run on the target "
            "GPU (RTX 4050) — profiling on CPU would not measure anything "
            "meaningful for this project's VRAM constraint. Check your "
            "PyTorch install matches your CUDA driver version."
        )

    device = torch.device(profiling_cfg["device"])
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device))

    model = ProfilingModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    results = find_max_batch_size(model, optimizer, device, config)

    results_path = Path(profiling_cfg["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", results_path)
    logger.info(
        "RECOMMENDED BATCH SIZE: %d — copy this into ARCHITECTURE.md §6, "
        "replacing the placeholder text there.",
        results["recommended_batch_size"],
    )


if __name__ == "__main__":
    main()

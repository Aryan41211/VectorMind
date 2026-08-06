"""
VectorMind FAISS Index Builder — Offline Embedding Indexing

Builds FAISS indices from trained model embeddings for efficient
similarity search at serving time. Creates separate indices for
image→text and text→image retrieval directions.

Usage:
    python -m backend.index_builder \\
        --checkpoint checkpoints/train/best_model.pt \\
        --config configs/model.yaml \\
        --output backend/indices/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.vectormind.data.dataset import Flickr30kDataset
from src.vectormind.models.vectormind_model import VectorMindModel

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata for a built FAISS index."""
    index_type: str
    dimension: int
    num_vectors: int
    build_time_seconds: float
    checkpoint_path: str
    dataset_split: str
    creation_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_vectors": self.num_vectors,
            "build_time_seconds": self.build_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "dataset_split": self.dataset_split,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class IndexBuildResult:
    """Result of building indices for both directions."""
    image_index: faiss.Index
    text_index: faiss.Index
    image_embeddings: np.ndarray
    text_embeddings: np.ndarray
    image_metadata: IndexMetadata
    text_metadata: IndexMetadata
    captions_per_image: int
    total_images: int


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "IndexFlatIP",
) -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings: Numpy array of shape (N, D) with L2-normalized embeddings.
        index_type: FAISS index type to use.

    Returns:
        Built FAISS index.
    """
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # Normalize for inner product (cosine similarity)
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    logger.info(f"Built {index_type}: {num_vectors} vectors, dim={dimension}")
    return index


def load_model(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device | None = None,
) -> VectorMindModel:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.
        config: Model configuration dictionary.
        device: Device to load model onto.

    Returns:
        Loaded VectorMindModel in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VectorMindModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} to {device}")
    return model


@torch.no_grad()
def generate_embeddings(
    model: VectorMindModel,
    dataloader: DataLoader,
    device: torch.device,
    captions_per_image: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate image and text embeddings for all batches.

    Args:
        model: Trained VectorMindModel.
        dataloader: DataLoader yielding batches with image, input_ids, attention_mask.
        device: Device for inference.
        captions_per_image: Number of captions per image.

    Returns:
        Tuple of (image_embeddings, text_embeddings) as numpy arrays.
    """
    all_image_embeddings: list[np.ndarray] = []
    all_text_embeddings: list[np.ndarray] = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = images.shape[0]

        # Generate image embeddings
        image_emb = model.encode_image(images)  # [B, D]

        # Generate text embeddings
        text_emb = model.encode_text(input_ids, attention_mask)  # [B, D]

        all_image_embeddings.append(image_emb.cpu().numpy())
        all_text_embeddings.append(text_emb.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Processed {batch_idx + 1} batches")

    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    text_embeddings = np.concatenate(all_text_embeddings, axis=0)

    return image_embeddings, text_embeddings


def save_indices(
    output_dir: Path,
    image_index: faiss.Index,
    text_index: faiss.Index,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_metadata: IndexMetadata,
    text_metadata: IndexMetadata,
    captions_per_image: int,
    total_images: int,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> None:
    """
    Save FAISS indices and metadata to disk.

    Args:
        output_dir: Directory to save indices.
        image_index: FAISS index for image embeddings.
        text_index: FAISS index for text embeddings.
        image_embeddings: Image embedding matrix.
        text_embeddings: Text embedding matrix.
        image_metadata: Metadata for image index.
        text_metadata: Metadata for text index.
        captions_per_image: Number of captions per image.
        total_images: Total number of images.
        sample_metadata: Per-vector metadata mapping index to image/caption.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save indices
    faiss.write_index(image_index, str(output_dir / "image_index.faiss"))
    faiss.write_index(text_index, str(output_dir / "text_index.faiss"))
    logger.info(f"Saved FAISS indices to {output_dir}")

    # Save embeddings (for reconstruction / diagnostics)
    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "text_embeddings.npy", text_embeddings)
    logger.info(f"Saved embeddings to {output_dir}")

    # Save metadata
    metadata = {
        "image_index": image_metadata.to_dict(),
        "text_index": text_metadata.to_dict(),
        "captions_per_image": captions_per_image,
        "total_images": total_images,
    }
    with open(output_dir / "index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {output_dir / 'index_metadata.json'}")

    # Save sample metadata (per-vector image paths and captions)
    if sample_metadata is not None:
        with open(output_dir / "sample_metadata.json", "w", encoding="utf-8") as f:
            json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved sample metadata ({len(sample_metadata)} entries) to "
            f"{output_dir / 'sample_metadata.json'}"
        )


def build_indices(
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    data_config_path: str | Path,
    output_dir: str | Path,
    dataset_split: str = "test",
    device: torch.device | None = None,
) -> IndexBuildResult:
    """
    Main entry point: build FAISS indices for both search directions.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        model_config_path: Path to model configuration YAML.
        data_config_path: Path to data configuration YAML.
        output_dir: Directory to save indices.
        dataset_split: Dataset split to index (train/val/test).
        device: Device for inference.

    Returns:
        IndexBuildResult with indices and metadata.
    """
    import sys
    from pathlib import Path as PathLib

    # Add scripts/ to path for _data_helpers
    scripts_dir = PathLib(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from _data_helpers import load_flickr30k_from_hf
    from src.vectormind.data.dataloader import _collate_fn
    from src.vectormind.data.splitter import create_splits
    from src.vectormind.data.tokenizer import CaptionTokenizer
    from src.vectormind.data.transforms import get_eval_transforms
    from src.vectormind.utils.config import load_config

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    logger.info(f"Building indices from checkpoint: {checkpoint_path}")
    logger.info(f"Dataset split: {dataset_split}")
    logger.info(f"Output directory: {output_dir}")

    # Load configs
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)

    # Load model
    model = load_model(checkpoint_path, model_config, device)

    # Load Flickr30k data
    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)

    # Split data
    train_pairs, val_pairs, test_pairs = create_splits(
        data_config, [Path(p) for p in image_paths], captions
    )

    # Select split
    if dataset_split == "train":
        pairs = train_pairs
    elif dataset_split == "val":
        pairs = val_pairs
    else:
        pairs = test_pairs

    split_paths, split_caps = zip(*pairs)
    split_paths = list(split_paths)
    split_caps = list(split_caps)

    logger.info(f"Loaded {len(split_paths)} pairs for {dataset_split} split")

    # Create transform and tokenizer
    transform = get_eval_transforms(data_config)
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    max_text_length = data_config["dataset"]["max_text_length"]

    # Create dataset
    dataset = Flickr30kDataset(
        image_paths=split_paths,
        captions=split_caps,
        transform=transform,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    captions_per_image = 5  # Flickr30k has 5 captions per image

    # Generate embeddings
    start_time = time.time()
    image_embeddings, text_embeddings = generate_embeddings(
        model, dataloader, device, captions_per_image
    )
    embedding_time = time.time() - start_time
    logger.info(f"Embedding generation took {embedding_time:.2f}s")

    # Build FAISS indices
    start_time = time.time()
    image_index = build_faiss_index(image_embeddings.copy(), "IndexFlatIP")
    text_index = build_faiss_index(text_embeddings.copy(), "IndexFlatIP")
    build_time = time.time() - start_time

    # Create metadata
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    image_metadata = IndexMetadata(
        index_type="IndexFlatIP",
        dimension=image_embeddings.shape[1],
        num_vectors=image_embeddings.shape[0],
        build_time_seconds=build_time,
        checkpoint_path=str(checkpoint_path),
        dataset_split=dataset_split,
        creation_timestamp=creation_time,
    )
    text_metadata = IndexMetadata(
        index_type="IndexFlatIP",
        dimension=text_embeddings.shape[1],
        num_vectors=text_embeddings.shape[0],
        build_time_seconds=build_time,
        checkpoint_path=str(checkpoint_path),
        dataset_split=dataset_split,
        creation_timestamp=creation_time,
    )

    # Save to disk
    # Build per-vector sample metadata: maps FAISS index → image/caption info
    sample_metadata = []
    for i, (img_path, cap) in enumerate(zip(split_paths, split_caps)):
        path_obj = Path(img_path)
        sample_metadata.append({
            "index": i,
            "image_path": str(img_path),
            "filename": path_obj.name,
            "caption": cap,
        })

    save_indices(
        output_dir,
        image_index,
        text_index,
        image_embeddings,
        text_embeddings,
        image_metadata,
        text_metadata,
        captions_per_image,
        len(dataset),
        sample_metadata,
    )

    return IndexBuildResult(
        image_index=image_index,
        text_index=text_index,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_metadata=image_metadata,
        text_metadata=text_metadata,
        captions_per_image=captions_per_image,
        total_images=len(dataset),
    )


def main() -> None:
    """CLI entry point for index building."""
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for VectorMind retrieval"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/indices/",
        help="Output directory for indices",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to index",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu), auto-detected if not specified",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = None
    if args.device:
        device = torch.device(args.device)

    build_indices(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        output_dir=args.output,
        dataset_split=args.split,
        device=device,
    )


if __name__ == "__main__":
    main()

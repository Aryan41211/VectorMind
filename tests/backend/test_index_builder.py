"""Tests for backend/index_builder.py — FAISS index building utilities.

Covers:
- FAISS index construction
- Embedding normalization
- Metadata serialization
- Index saving/loading
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from backend.index_builder import (
    IndexMetadata,
    build_caption_metadata,
    build_faiss_index,
    deduplicate_image_embeddings,
    save_indices,
)


class TestBuildFaissIndex:
    """Tests for the build_faiss_index function."""

    def test_creates_index_flat_ip(self):
        """IndexFlatIP is created for inner product search."""
        embeddings = np.random.randn(100, 256).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatIP")
        assert isinstance(index, faiss.IndexFlatIP)
        assert index.ntotal == 100
        assert index.d == 256

    def test_creates_index_flat_l2(self):
        """IndexFlatL2 is created for L2 distance search."""
        embeddings = np.random.randn(50, 128).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatL2")
        assert isinstance(index, faiss.IndexFlatL2)
        assert index.ntotal == 50
        assert index.d == 128

    def test_embeddings_are_normalized(self):
        """Embeddings are L2-normalized before indexing."""
        embeddings = np.random.randn(100, 256).astype(np.float32)
        original_norms = np.linalg.norm(embeddings, axis=1).copy()
        # Random Gaussian rows are not unit-norm, so this confirms the
        # test is actually exercising normalization rather than being
        # handed already-normalized input.
        assert not np.allclose(original_norms, 1.0)
        build_faiss_index(embeddings, "IndexFlatIP")
        # After normalization, all vectors should have unit norm
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_unsupported_index_type_raises(self):
        """Unsupported index types raise ValueError."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        with pytest.raises(ValueError, match="Unsupported index type"):
            build_faiss_index(embeddings, "IndexIVFFlat")

    def test_search_returns_correct_shape(self):
        """Search returns correct number of results."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatIP")

        query = np.random.randn(1, 256).astype(np.float32)
        faiss.normalize_L2(query)

        k = 10
        distances, indices = index.search(query, k)
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)

    def test_search_results_are_sorted(self):
        """Search results are sorted by relevance (descending for IP)."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatIP")

        query = np.random.randn(1, 256).astype(np.float32)
        faiss.normalize_L2(query)

        distances, indices = index.search(query, 5)
        # Distances should be in descending order for IndexFlatIP
        assert all(distances[0][i] >= distances[0][i + 1]
                   for i in range(len(distances[0]) - 1))

    def test_empty_embeddings_returns_empty_index(self):
        """Empty embedding array returns an index with 0 vectors."""
        embeddings = np.array([], dtype=np.float32).reshape(0, 256)
        index = build_faiss_index(embeddings, "IndexFlatIP")
        assert index.ntotal == 0


class TestIndexMetadata:
    """Tests for IndexMetadata dataclass."""

    def test_creation(self):
        """IndexMetadata can be created with required fields."""
        metadata = IndexMetadata(
            index_type="IndexFlatIP",
            dimension=256,
            num_vectors=1000,
            build_time_seconds=1.5,
            checkpoint_path="checkpoints/model.pt",
            dataset_split="test",
        )
        assert metadata.index_type == "IndexFlatIP"
        assert metadata.dimension == 256
        assert metadata.num_vectors == 1000

    def test_to_dict(self):
        """IndexMetadata serializes to dictionary correctly."""
        metadata = IndexMetadata(
            index_type="IndexFlatIP",
            dimension=256,
            num_vectors=1000,
            build_time_seconds=1.5,
            checkpoint_path="checkpoints/model.pt",
            dataset_split="test",
            creation_timestamp="2026-08-06 12:00:00",
        )
        d = metadata.to_dict()
        assert d["index_type"] == "IndexFlatIP"
        assert d["dimension"] == 256
        assert d["num_vectors"] == 1000
        assert d["build_time_seconds"] == 1.5
        assert d["checkpoint_path"] == "checkpoints/model.pt"
        assert d["dataset_split"] == "test"
        assert d["creation_timestamp"] == "2026-08-06 12:00:00"

    def test_to_dict_json_serializable(self):
        """IndexMetadata dictionary is JSON-serializable."""
        metadata = IndexMetadata(
            index_type="IndexFlatIP",
            dimension=256,
            num_vectors=1000,
            build_time_seconds=1.5,
            checkpoint_path="checkpoints/model.pt",
            dataset_split="test",
        )
        d = metadata.to_dict()
        json_str = json.dumps(d)
        assert json_str is not None


def _image_records(n: int) -> list[dict]:
    """One index map record per image-index position."""
    return [
        {
            "index": i,
            "image_path": f"images/{i:06d}.jpg",
            "filename": f"{i:06d}.jpg",
            "captions": [f"caption {i}"],
        }
        for i in range(n)
    ]


def _caption_records(n: int) -> list[dict]:
    """One index map record per text-index position."""
    return [
        {
            "index": i,
            "caption": f"caption {i}",
            "image_path": f"images/{i:06d}.jpg",
            "filename": f"{i:06d}.jpg",
        }
        for i in range(n)
    ]


class TestSaveIndices:
    """Tests for the save_indices function."""

    def test_saves_faiss_indices(self):
        """FAISS indices are saved to disk."""
        image_embeddings = np.random.randn(100, 256).astype(np.float32)
        text_embeddings = np.random.randn(100, 256).astype(np.float32)

        image_index = build_faiss_index(image_embeddings.copy(), "IndexFlatIP")
        text_index = build_faiss_index(text_embeddings.copy(), "IndexFlatIP")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_indices(
                output_dir=output_dir,
                image_index=image_index,
                text_index=text_index,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                image_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=256,
                    num_vectors=100,
                    build_time_seconds=0.1,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                text_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=256,
                    num_vectors=100,
                    build_time_seconds=0.1,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                captions_per_image=5,
                total_images=100,
                image_samples=_image_records(100),
                caption_samples=_caption_records(100),
                save_embeddings=True,
            )

            assert (output_dir / "image_index.faiss").exists()
            assert (output_dir / "text_index.faiss").exists()
            assert (output_dir / "image_embeddings.npy").exists()
            assert (output_dir / "text_embeddings.npy").exists()
            assert (output_dir / "index_metadata.json").exists()
            assert (output_dir / "image_samples.json").exists()
            assert (output_dir / "caption_samples.json").exists()

    def test_metadata_json_is_valid(self):
        """Saved metadata JSON contains expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_indices(
                output_dir=output_dir,
                image_index=build_faiss_index(
                    np.random.randn(10, 64).astype(np.float32), "IndexFlatIP"
                ),
                text_index=build_faiss_index(
                    np.random.randn(10, 64).astype(np.float32), "IndexFlatIP"
                ),
                image_embeddings=np.random.randn(10, 64).astype(np.float32),
                text_embeddings=np.random.randn(10, 64).astype(np.float32),
                image_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=64,
                    num_vectors=10,
                    build_time_seconds=0.01,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                text_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=64,
                    num_vectors=10,
                    build_time_seconds=0.01,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                captions_per_image=5,
                total_images=10,
                image_samples=_image_records(10),
                caption_samples=_caption_records(10),
            )

            with open(output_dir / "index_metadata.json") as f:
                metadata = json.load(f)

            assert "image_index" in metadata
            assert "text_index" in metadata
            assert metadata["captions_per_image"] == 5
            assert metadata["total_images"] == 10

    def test_saved_index_can_be_loaded(self):
        """FAISS indices can be loaded from saved files."""
        embeddings = np.random.randn(100, 256).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatIP")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_indices(
                output_dir=output_dir,
                image_index=index,
                text_index=index,
                image_embeddings=embeddings,
                text_embeddings=embeddings,
                image_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=256,
                    num_vectors=100,
                    build_time_seconds=0.1,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                text_metadata=IndexMetadata(
                    index_type="IndexFlatIP",
                    dimension=256,
                    num_vectors=100,
                    build_time_seconds=0.1,
                    checkpoint_path="test.pt",
                    dataset_split="test",
                ),
                captions_per_image=5,
                total_images=100,
                image_samples=_image_records(100),
                caption_samples=_caption_records(100),
            )

            loaded_index = faiss.read_index(
                str(output_dir / "image_index.faiss")
            )
            assert loaded_index.ntotal == 100
            assert loaded_index.d == 256


class TestIndexSearchIntegration:
    """Integration tests for FAISS index search correctness."""

    def test_identical_vectors_return_high_score(self):
        """Query identical to indexed vector returns score ~1.0."""
        dim = 256
        vector = np.random.randn(1, dim).astype(np.float32)
        faiss.normalize_L2(vector)

        index = build_faiss_index(vector.copy(), "IndexFlatIP")
        distances, indices = index.search(vector, 1)

        assert indices[0][0] == 0
        assert distances[0][0] > 0.99

    def test_top_k_retrieval(self):
        """Top-K retrieval returns K results sorted by score."""
        dim = 256
        num_vectors = 1000
        embeddings = np.random.randn(num_vectors, dim).astype(np.float32)

        index = build_faiss_index(embeddings, "IndexFlatIP")

        query = np.random.randn(1, dim).astype(np.float32)
        faiss.normalize_L2(query)

        k = 10
        distances, indices = index.search(query, k)

        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        assert all(0 <= idx < num_vectors for idx in indices[0])
        assert all(distances[0][i] >= distances[0][i + 1]
                   for i in range(k - 1))

    def test_batch_query(self):
        """Batch queries return correct shapes."""
        dim = 128
        num_vectors = 500
        num_queries = 5
        k = 10

        embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
        index = build_faiss_index(embeddings, "IndexFlatIP")

        queries = np.random.randn(num_queries, dim).astype(np.float32)
        faiss.normalize_L2(queries)

        distances, indices = index.search(queries, k)

        assert distances.shape == (num_queries, k)
        assert indices.shape == (num_queries, k)


class TestDeduplicateImageEmbeddings:
    """Regression guard for docs/KNOWN_ISSUES.md §2.

    Flickr30k yields five (image, caption) rows per image, so the raw
    image embeddings contain five identical vectors per picture. Indexing
    all of them put 15,895 vectors in an index covering 3,179 images, and
    /search/text returned the same picture up to five times in one top-10.
    """

    @staticmethod
    def _five_captions_per_image(n_images: int, dim: int = 8):
        embeddings = np.repeat(
            np.arange(n_images, dtype=np.float32).reshape(n_images, 1),
            5,
            axis=0,
        ).repeat(dim, axis=1)
        paths = [f"images/{i:06d}.jpg" for i in range(n_images) for _ in range(5)]
        captions = [f"image {i} caption {c}" for i in range(n_images) for c in range(5)]
        return embeddings, paths, captions

    def test_collapses_five_rows_per_image(self):
        emb, paths, caps = self._five_captions_per_image(20)
        unique, records = deduplicate_image_embeddings(emb, paths, caps)
        assert emb.shape[0] == 100
        assert unique.shape[0] == 20
        assert len(records) == 20

    def test_preserves_embedding_dimension(self):
        emb, paths, caps = self._five_captions_per_image(20, dim=256)
        unique, _ = deduplicate_image_embeddings(emb, paths, caps)
        assert unique.shape[1] == 256

    def test_keeps_first_occurrence_in_order(self):
        emb, paths, caps = self._five_captions_per_image(4)
        unique, records = deduplicate_image_embeddings(emb, paths, caps)
        # Row i of the synthetic input encodes image id i.
        assert [row[0] for row in unique] == [0.0, 1.0, 2.0, 3.0]
        assert [r["filename"] for r in records] == [
            "000000.jpg",
            "000001.jpg",
            "000002.jpg",
            "000003.jpg",
        ]

    def test_gathers_every_caption_per_image(self):
        emb, paths, caps = self._five_captions_per_image(3)
        _, records = deduplicate_image_embeddings(emb, paths, caps)
        for i, record in enumerate(records):
            assert len(record["captions"]) == 5
            assert record["captions"][0] == f"image {i} caption 0"

    def test_index_field_matches_position(self):
        emb, paths, caps = self._five_captions_per_image(10)
        _, records = deduplicate_image_embeddings(emb, paths, caps)
        assert [r["index"] for r in records] == list(range(10))

    def test_handles_non_contiguous_rows(self):
        """Correctness must not depend on rows for one image being adjacent."""
        emb = np.array([[1.0], [2.0], [1.0], [3.0], [2.0]], dtype=np.float32)
        paths = ["a.jpg", "b.jpg", "a.jpg", "c.jpg", "b.jpg"]
        caps = ["a1", "b1", "a2", "c1", "b2"]
        unique, records = deduplicate_image_embeddings(emb, paths, caps)
        assert unique.shape[0] == 3
        assert [r["filename"] for r in records] == ["a.jpg", "b.jpg", "c.jpg"]
        assert records[0]["captions"] == ["a1", "a2"]
        assert records[1]["captions"] == ["b1", "b2"]

    def test_already_unique_input_is_unchanged(self):
        emb = np.arange(6, dtype=np.float32).reshape(3, 2)
        paths = ["a.jpg", "b.jpg", "c.jpg"]
        unique, records = deduplicate_image_embeddings(emb, paths, ["a", "b", "c"])
        assert np.array_equal(unique, emb)
        assert len(records) == 3

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            deduplicate_image_embeddings(
                np.zeros((3, 2), dtype=np.float32), ["a.jpg"], ["a"]
            )


class TestBuildCaptionMetadata:
    def test_one_record_per_caption(self):
        paths = [f"images/{i // 5:06d}.jpg" for i in range(15)]
        caps = [f"caption {i}" for i in range(15)]
        records = build_caption_metadata(paths, caps)
        assert len(records) == 15
        assert [r["index"] for r in records] == list(range(15))

    def test_carries_filename_and_caption(self):
        records = build_caption_metadata(["images/000007.jpg"], ["a dog runs"])
        assert records[0]["filename"] == "000007.jpg"
        assert records[0]["caption"] == "a dog runs"

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_caption_metadata(["a.jpg", "b.jpg"], ["only one"])


class TestSaveIndicesValidatesIndexMaps:
    """save_indices must refuse maps that do not line up with their index."""

    @staticmethod
    def _index(n: int, dim: int = 8):
        return build_faiss_index(
            np.random.randn(n, dim).astype(np.float32), "IndexFlatIP"
        )

    def _save(self, tmpdir, n_image_records, n_caption_records):
        meta = IndexMetadata(
            index_type="IndexFlatIP",
            dimension=8,
            num_vectors=10,
            build_time_seconds=0.01,
            checkpoint_path="test.pt",
            dataset_split="test",
        )
        save_indices(
            output_dir=Path(tmpdir),
            image_index=self._index(10),
            text_index=self._index(50),
            image_embeddings=np.random.randn(10, 8).astype(np.float32),
            text_embeddings=np.random.randn(50, 8).astype(np.float32),
            image_metadata=meta,
            text_metadata=meta,
            captions_per_image=5,
            total_images=10,
            image_samples=_image_records(n_image_records),
            caption_samples=_caption_records(n_caption_records),
        )

    def test_accepts_matching_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save(tmpdir, 10, 50)
            assert (Path(tmpdir) / "image_samples.json").exists()

    def test_rejects_image_map_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="image_samples has 9 records"):
                self._save(tmpdir, 9, 50)

    def test_rejects_caption_map_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="caption_samples has 49 records"):
                self._save(tmpdir, 10, 49)

    def test_skips_npy_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save(tmpdir, 10, 50)
            assert not (Path(tmpdir) / "image_embeddings.npy").exists()

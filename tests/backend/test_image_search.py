"""Tests for backend/routers/image_search.py — Image search endpoint.

Covers:
- Image search endpoint
- Image validation
- Error handling
- Response format
"""

from __future__ import annotations

import io

import faiss
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app_state, create_app


class MockModel:
    """Mock model for testing."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_image(self, image_tensor):
        """Return random embeddings."""
        batch_size = image_tensor.shape[0]
        embeddings = np.random.randn(batch_size, self.dim).astype(np.float32)
        return torch.tensor(embeddings)


@pytest.fixture
def mock_app_state():
    """Set up mock application state for testing."""
    # Save original state
    original_model = app_state.model
    original_device = app_state.device
    original_text_index = app_state.text_index
    original_loaded = app_state.loaded

    # Set up mock state
    dim = 256
    num_vectors = 1000

    app_state.model = MockModel(dim)
    app_state.device = torch.device("cpu")  # Use real torch device

    # Create a FAISS index with random embeddings
    embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    app_state.text_index = index
    app_state.loaded = True

    yield app_state

    # Restore original state
    app_state.model = original_model
    app_state.device = original_device
    app_state.text_index = original_text_index
    app_state.loaded = original_loaded


def create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a test image and return as bytes."""
    image = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestImageSearchEndpoint:
    """Tests for the /search/image endpoint."""

    def test_image_search_returns_200(self, mock_app_state):
        """Image search returns 200 OK with valid image."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_image_search_returns_results(self, mock_app_state):
        """Image search returns a list of results."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_image_search_respects_top_k(self, mock_app_state):
        """Image search respects top_k parameter."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image?top_k=5",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert len(data["results"]) <= 5

    def test_image_search_includes_query(self, mock_app_state):
        """Image search response includes the filename."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("my_photo.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["query"] == "image:my_photo.jpg"

    def test_image_search_includes_search_type(self, mock_app_state):
        """Image search response includes search_type."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["search_type"] == "image_to_text"

    def test_image_search_includes_latency(self, mock_app_state):
        """Image search response includes latency_ms."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_image_search_no_file_rejected(self, mock_app_state):
        """Image search rejects request without file."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post("/search/image")
        assert response.status_code == 422

    def test_image_search_service_unavailable(self):
        """Image search returns 503 when service is unavailable."""
        # Save original state
        original_loaded = app_state.loaded
        original_model = app_state.model

        app_state.loaded = False
        app_state.model = None

        try:
            app = create_app(test_mode=True)
            client = TestClient(app)
            image_bytes = create_test_image()
            response = client.post(
                "/search/image",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
            assert response.status_code == 503
        finally:
            app_state.loaded = original_loaded
            app_state.model = original_model


class TestImageValidation:
    """Tests for image validation."""

    def test_valid_jpeg_accepted(self, mock_app_state):
        """Valid JPEG image is accepted."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_valid_png_accepted(self, mock_app_state):
        """Valid PNG image is accepted."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        response = client.post(
            "/search/image",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_invalid_content_type_rejected(self, mock_app_state):
        """Non-image content type is rejected."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_corrupted_image_rejected(self, mock_app_state):
        """Corrupted image data is rejected."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", b"not image data", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_oversized_image_rejected(self, mock_app_state):
        """Image exceeding max size is rejected."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        # Create a large image (simulate 11MB)
        large_data = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/search/image",
            files={"file": ("large.jpg", large_data, "image/jpeg")},
        )
        assert response.status_code == 400


class TestImageSearchResultFormat:
    """Tests for the result format in image search responses."""

    def test_results_have_index(self, mock_app_state):
        """Each result has an index field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "index" in result

    def test_results_have_score(self, mock_app_state):
        """Each result has a score field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_results_have_caption(self, mock_app_state):
        """Each result has a caption field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "caption" in result

    def test_results_sorted_by_score(self, mock_app_state):
        """Results are sorted by score in descending order."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        image_bytes = create_test_image()
        response = client.post(
            "/search/image",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"top_k": 10},
        )
        data = response.json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

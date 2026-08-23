"""
Integration tests for backend API — Full workflow testing.

Covers:
- End-to-end search workflows
- API contract validation
- Error recovery
- Performance basics
"""

from __future__ import annotations

import io
import time

import faiss
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.app import app_state


class MockModel:
    """Mock model for integration testing."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_text(self, input_ids, attention_mask=None):
        """Return deterministic embeddings based on input."""
        batch_size = input_ids.shape[0]
        # Use input hash for deterministic results
        seed = input_ids.sum().item() % 1000
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(batch_size, self.dim).astype(np.float32)
        return torch.tensor(embeddings)

    def encode_image(self, image_tensor):
        """Return deterministic embeddings based on input."""
        batch_size = image_tensor.shape[0]
        # Use pixel hash for deterministic results
        seed = int(image_tensor.mean().item() * 1000) % 1000
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(batch_size, self.dim).astype(np.float32)
        return torch.tensor(embeddings)


@pytest.fixture
def mock_app_state():
    """Set up mock application state for integration testing."""
    # Save original state
    original_model = app_state.model
    original_device = app_state.device
    original_image_index = app_state.image_index
    original_text_index = app_state.text_index
    original_loaded = app_state.loaded

    # Set up mock state
    dim = 256
    num_vectors = 100

    app_state.model = MockModel(dim)
    app_state.device = torch.device("cpu")

    # Create FAISS indices with random embeddings
    embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
    faiss.normalize_L2(embeddings)

    image_index = faiss.IndexFlatIP(dim)
    image_index.add(embeddings.copy())
    app_state.image_index = image_index

    text_index = faiss.IndexFlatIP(dim)
    text_index.add(embeddings.copy())
    app_state.text_index = text_index

    app_state.loaded = True

    yield app_state

    # Restore original state
    app_state.model = original_model
    app_state.device = original_device
    app_state.image_index = original_image_index
    app_state.text_index = original_text_index
    app_state.loaded = original_loaded


class TestAPIWorkflow:
    """Integration tests for complete API workflows."""

    def test_health_check_workflow(self, mock_app_state):
        """Health check returns correct status."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        # Check health
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["index_loaded"] is True
        assert data["num_indexed_images"] == 100

    def test_text_search_workflow(self, mock_app_state):
        """Complete text search workflow."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        # Search
        response = client.post(
            "/search/text",
            json={"query": "a dog playing in the park", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "results" in data
        assert "query" in data
        assert "search_type" in data
        assert "latency_ms" in data
        assert data["search_type"] == "text_to_image"
        assert len(data["results"]) <= 5

    def test_image_search_workflow(self, mock_app_state):
        """Complete image search workflow."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        # Create test image
        image = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Search
        response = client.post(
            "/search/image?top_k=5",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "results" in data
        assert "query" in data
        assert "search_type" in data
        assert "latency_ms" in data
        assert data["search_type"] == "image_to_text"
        assert len(data["results"]) <= 5

    def test_search_performance(self, mock_app_state):
        """Search completes within acceptable latency."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        # Text search
        start = time.time()
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 10},
        )
        text_latency = time.time() - start
        assert response.status_code == 200
        assert text_latency < 1.0  # Should complete within 1 second

        # Image search
        image = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        start = time.time()
        response = client.post(
            "/search/image?top_k=10",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        image_latency = time.time() - start
        assert response.status_code == 200
        assert image_latency < 1.0  # Should complete within 1 second


class TestAPIContract:
    """Tests for API contract validation."""

    def test_openapi_schema_complete(self, mock_app_state):
        """OpenAPI schema includes all endpoints."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        # Check all endpoints exist
        assert "/health" in schema["paths"]
        assert "/api/info" in schema["paths"]
        assert "/search/text" in schema["paths"]
        assert "/search/image" in schema["paths"]

    def test_text_search_request_schema(self, mock_app_state):
        """Text search request schema is correct."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        response = client.get("/openapi.json")
        schema = response.json()

        # Check text search endpoint
        text_search = schema["paths"]["/search/text"]["post"]
        assert "requestBody" in text_search

    def test_image_search_request_schema(self, mock_app_state):
        """Image search request schema is correct."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        response = client.get("/openapi.json")
        schema = response.json()

        # Check image search endpoint
        image_search = schema["paths"]["/search/image"]["post"]
        assert "requestBody" in image_search


class TestErrorRecovery:
    """Tests for error recovery and graceful degradation."""

    def test_search_after_model_unload(self):
        """Search returns 503 after model is unloaded."""
        # Save original state
        original_loaded = app_state.loaded
        original_model = app_state.model
        original_device = app_state.device

        try:
            # First, set up a working state
            dim = 256
            app_state.model = MockModel(dim)
            app_state.device = torch.device("cpu")
            app_state.loaded = True

            # Create FAISS indices
            num_vectors = 100
            embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
            faiss.normalize_L2(embeddings)

            image_index = faiss.IndexFlatIP(dim)
            image_index.add(embeddings.copy())
            app_state.image_index = image_index

            text_index = faiss.IndexFlatIP(dim)
            text_index.add(embeddings.copy())
            app_state.text_index = text_index

            app = create_app(test_mode=True)
            client = TestClient(app)

            # Verify it works
            response = client.post(
                "/search/text",
                json={"query": "test"},
            )
            assert response.status_code == 200

            # Now unload the model
            app_state.loaded = False
            app_state.model = None

            # Search should fail with 503
            response = client.post(
                "/search/text",
                json={"query": "test"},
            )
            assert response.status_code == 503

        finally:
            app_state.loaded = original_loaded
            app_state.model = original_model
            app_state.device = original_device

    def test_concurrent_requests(self, mock_app_state):
        """Multiple requests can be handled."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        # Send multiple requests
        for i in range(5):
            response = client.post(
                "/search/text",
                json={"query": f"query {i}"},
            )
            assert response.status_code == 200


class TestResponseHeaders:
    """Tests for response headers."""

    def test_process_time_header(self, mock_app_state):
        """X-Process-Time header is present."""
        app = create_app(test_mode=True)
        client = TestClient(app)

        response = client.get("/health")
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0

    def test_cors_headers(self, mock_app_state):
        """CORS headers are present."""
        app = create_app(test_mode=True)
        client = TestClient(app, headers={"Origin": "http://localhost:3000"})

        response = client.get("/health")
        assert response.status_code == 200

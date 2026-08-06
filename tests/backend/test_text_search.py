"""
Tests for backend/routers/text_search.py — Text search endpoint.

Covers:
- Text search endpoint
- Request validation
- Error handling
- Response format
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import AppState, create_app
from backend.app import app_state


class MockModel:
    """Mock model for testing."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_text(self, text_input):
        """Return random embeddings."""
        batch_size = text_input["input_ids"].shape[0]
        embeddings = np.random.randn(batch_size, self.dim).astype(np.float32)
        return torch.tensor(embeddings)


@pytest.fixture
def mock_app_state():
    """Set up mock application state for testing."""
    # Save original state
    original_model = app_state.model
    original_device = app_state.device
    original_image_index = app_state.image_index
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
    app_state.image_index = index
    app_state.loaded = True

    yield app_state

    # Restore original state
    app_state.model = original_model
    app_state.device = original_device
    app_state.image_index = original_image_index
    app_state.loaded = original_loaded


class TestTextSearchEndpoint:
    """Tests for the /search/text endpoint."""

    def test_text_search_returns_200(self, mock_app_state):
        """Text search returns 200 OK with valid request."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog playing"},
        )
        assert response.status_code == 200

    def test_text_search_returns_results(self, mock_app_state):
        """Text search returns a list of results."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog playing"},
        )
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_text_search_respects_top_k(self, mock_app_state):
        """Text search respects top_k parameter."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 5},
        )
        data = response.json()
        assert len(data["results"]) <= 5

    def test_text_search_includes_query(self, mock_app_state):
        """Text search response includes the original query."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog playing"},
        )
        data = response.json()
        assert data["query"] == "a dog playing"

    def test_text_search_includes_search_type(self, mock_app_state):
        """Text search response includes search_type."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog"},
        )
        data = response.json()
        assert data["search_type"] == "text_to_image"

    def test_text_search_includes_latency(self, mock_app_state):
        """Text search response includes latency_ms."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog"},
        )
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_text_search_empty_query_rejected(self, mock_app_state):
        """Text search rejects empty query."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_text_search_missing_query_rejected(self, mock_app_state):
        """Text search rejects missing query field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"top_k": 10},
        )
        assert response.status_code == 422

    def test_text_search_service_unavailable(self):
        """Text search returns 503 when service is unavailable."""
        # Save original state
        original_loaded = app_state.loaded
        original_model = app_state.model

        app_state.loaded = False
        app_state.model = None

        try:
            app = create_app(test_mode=True)
            client = TestClient(app)
            response = client.post(
                "/search/text",
                json={"query": "a dog"},
            )
            assert response.status_code == 503
        finally:
            app_state.loaded = original_loaded
            app_state.model = original_model


class TestTextSearchResultFormat:
    """Tests for the result format in text search responses."""

    def test_results_have_index(self, mock_app_state):
        """Each result has an index field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "index" in result

    def test_results_have_score(self, mock_app_state):
        """Each result has a score field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_results_have_image_path(self, mock_app_state):
        """Each result has an image_path field."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 3},
        )
        data = response.json()
        for result in data["results"]:
            assert "image_path" in result

    def test_results_sorted_by_score(self, mock_app_state):
        """Results are sorted by score in descending order."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 10},
        )
        data = response.json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

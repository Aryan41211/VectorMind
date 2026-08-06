"""
Tests for backend/schemas.py — Pydantic request/response models.

Covers:
- Request validation
- Response serialization
- Field constraints
- Default values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageSearchRequest,
    ModelConfig,
    SearchResult,
    SearchResponse,
    ServerConfig,
    TextSearchRequest,
)


class TestTextSearchRequest:
    """Tests for TextSearchRequest schema."""

    def test_valid_request(self):
        """Valid text search request is accepted."""
        req = TextSearchRequest(query="a dog playing")
        assert req.query == "a dog playing"
        assert req.top_k == 10  # default

    def test_custom_top_k(self):
        """Custom top_k value is accepted."""
        req = TextSearchRequest(query="a dog", top_k=20)
        assert req.top_k == 20

    def test_empty_query_rejected(self):
        """Empty query string is rejected."""
        with pytest.raises(ValidationError):
            TextSearchRequest(query="")

    def test_whitespace_only_query_accepted(self):
        """Whitespace-only query is accepted (trimmed to empty)."""
        req = TextSearchRequest(query="   ")
        assert req.query == "   "  # Pydantic doesn't strip by default

    def test_top_k_minimum(self):
        """top_k=0 is rejected."""
        with pytest.raises(ValidationError):
            TextSearchRequest(query="test", top_k=0)

    def test_top_k_maximum(self):
        """top_k > 100 is rejected."""
        with pytest.raises(ValidationError):
            TextSearchRequest(query="test", top_k=101)

    def test_long_query_accepted(self):
        """Long query within limit is accepted."""
        long_query = "a " * 250  # 500 chars
        req = TextSearchRequest(query=long_query.strip())
        assert len(req.query) <= 500

    def test_query_too_long_rejected(self):
        """Query exceeding 500 chars is rejected."""
        with pytest.raises(ValidationError):
            TextSearchRequest(query="a " * 251)

    def test_missing_query_rejected(self):
        """Missing query field is rejected."""
        with pytest.raises(ValidationError):
            TextSearchRequest()


class TestImageSearchRequest:
    """Tests for ImageSearchRequest schema."""

    def test_default_top_k(self):
        """Default top_k is 10."""
        req = ImageSearchRequest()
        assert req.top_k == 10

    def test_custom_top_k(self):
        """Custom top_k is accepted."""
        req = ImageSearchRequest(top_k=25)
        assert req.top_k == 25

    def test_top_k_minimum(self):
        """top_k=0 is rejected."""
        with pytest.raises(ValidationError):
            ImageSearchRequest(top_k=0)


class TestSearchResult:
    """Tests for SearchResult schema."""

    def test_minimal_result(self):
        """Minimal result with only required fields."""
        result = SearchResult(rank=1, index=0, score=0.85)
        assert result.rank == 1
        assert result.index == 0
        assert result.score == 0.85
        assert result.caption is None
        assert result.image_path is None
        assert result.metadata is None

    def test_full_result(self):
        """Result with all fields populated."""
        result = SearchResult(
            rank=1,
            index=42,
            score=0.92,
            caption="a dog playing",
            image_path="data/images/42.jpg",
            filename="000042.jpg",
            image_url="/images/000042.jpg",
            metadata={"distance": 0.08},
        )
        assert result.rank == 1
        assert result.index == 42
        assert result.score == 0.92
        assert result.caption == "a dog playing"
        assert result.image_path == "data/images/42.jpg"
        assert result.filename == "000042.jpg"
        assert result.image_url == "/images/000042.jpg"
        assert result.metadata == {"distance": 0.08}

    def test_negative_index_accepted(self):
        """Negative index is accepted (no constraint defined)."""
        result = SearchResult(rank=1, index=-1, score=0.5)
        assert result.index == -1


class TestSearchResponse:
    """Tests for SearchResponse schema."""

    def test_valid_response(self):
        """Valid search response is created."""
        response = SearchResponse(
            results=[SearchResult(rank=1, index=0, score=0.85)],
            query="a dog",
            search_type="text_to_image",
            total_results=1,
            latency_ms=15.5,
        )
        assert len(response.results) == 1
        assert response.total_results == 1

    def test_empty_results(self):
        """Empty results list is valid."""
        response = SearchResponse(
            results=[],
            query="a dog",
            search_type="text_to_image",
            total_results=0,
            latency_ms=10.0,
        )
        assert len(response.results) == 0

    def test_missing_required_field(self):
        """Missing required field raises error."""
        with pytest.raises(ValidationError):
            SearchResponse(
                results=[],
                query="test",
                search_type="text_to_image",
                # missing total_results and latency_ms
            )


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self):
        """Healthy response is created."""
        response = HealthResponse(
            model_loaded=True,
            index_loaded=True,
            device="cuda",
            num_indexed_images=1000,
        )
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.num_indexed_images == 1000

    def test_unhealthy_response(self):
        """Unhealthy response is created."""
        response = HealthResponse(
            model_loaded=False,
            index_loaded=False,
            device="cpu",
            num_indexed_images=0,
        )
        assert response.model_loaded is False


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_with_message(self):
        """Error response with message."""
        response = ErrorResponse(error="Model not found")
        assert response.error == "Model not found"
        assert response.detail is None

    def test_error_with_detail(self):
        """Error response with detail."""
        response = ErrorResponse(
            error="Index load failed",
            detail="File not found at /path/to/index.faiss",
        )
        assert response.detail is not None


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_valid_config(self):
        """Valid model configuration."""
        config = ModelConfig(
            checkpoint_path="checkpoints/model.pt",
            index_path="backend/indices/",
        )
        assert config.device == "auto"  # default

    def test_custom_device(self):
        """Custom device is accepted."""
        config = ModelConfig(
            checkpoint_path="checkpoints/model.pt",
            index_path="backend/indices/",
            device="cuda",
        )
        assert config.device == "cuda"


class TestServerConfig:
    """Tests for ServerConfig schema."""

    def test_default_config(self):
        """Default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is False

    def test_custom_config(self):
        """Custom server configuration."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            workers=4,
            reload=True,
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.workers == 4


class TestSchemaSerialization:
    """Tests for schema JSON serialization."""

    def test_search_response_json(self):
        """SearchResponse serializes to JSON correctly."""
        response = SearchResponse(
            results=[SearchResult(rank=1, index=0, score=0.85)],
            query="a dog",
            search_type="text_to_image",
            total_results=1,
            latency_ms=15.5,
        )
        json_str = response.model_dump_json()
        assert "a dog" in json_str
        assert "text_to_image" in json_str

    def test_health_response_json(self):
        """HealthResponse serializes to JSON correctly."""
        response = HealthResponse(
            model_loaded=True,
            index_loaded=True,
            device="cuda",
            num_indexed_images=1000,
        )
        json_str = response.model_dump_json()
        assert "healthy" in json_str
        assert "cuda" in json_str

    def test_search_result_json(self):
        """SearchResult serializes to JSON correctly."""
        result = SearchResult(
            rank=1,
            index=42,
            score=0.92,
            caption="a dog playing",
        )
        json_str = result.model_dump_json()
        assert "42" in json_str
        assert "0.92" in json_str

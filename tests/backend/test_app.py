"""Tests for backend/app.py — FastAPI application setup and health checks.

Covers:
- Application creation
- Health check endpoint
- Root endpoint
- CORS configuration
- Request timing middleware
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app


class TestApplicationCreation:
    """Tests for FastAPI application creation."""

    def test_app_creation(self):
        """Application can be created."""
        app = create_app(test_mode=True)
        assert app.title == "VectorMind"
        assert app.version == "0.1.0"

    def test_app_has_health_endpoint(self):
        """Application has health check endpoint."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_app_has_root_endpoint(self):
        """Application has root endpoint."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self):
        """Health check returns 200 OK."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_model_not_loaded(self):
        """Health check reports model not loaded in test mode."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False
        assert data["index_loaded"] is False

    def test_health_check_has_required_fields(self):
        """Health check response has all required fields."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "index_loaded" in data
        assert "device" in data
        assert "num_indexed_images" in data

    def test_health_check_status_healthy(self):
        """Health check reports healthy status."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root and API info endpoints."""

    def test_api_info_returns_api_info(self):
        """API info endpoint returns API information."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/api/info")
        data = response.json()
        assert data["name"] == "VectorMind"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    def test_root_returns_200(self):
        """Root endpoint returns 200 OK (SPA or JSON)."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_origins(self):
        """CORS middleware allows all origins."""
        app = create_app(test_mode=True)
        client = TestClient(app, headers={"Origin": "http://example.com"})
        response = client.get("/health")
        assert response.status_code == 200


class TestTimingMiddleware:
    """Tests for request timing middleware."""

    def test_timing_header_present(self):
        """X-Process-Time header is present in responses."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        assert "x-process-time" in response.headers

    def test_timing_header_is_numeric(self):
        """X-Process-Time header contains a numeric value."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/health")
        process_time = float(response.headers["x-process-time"].removesuffix("ms"))
        assert process_time >= 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_path_serves_spa(self):
        """Unknown paths serve the SPA index.html (client-side routing)."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/nonexistent")
        assert response.status_code == 200

    def test_405_for_wrong_method(self):
        """Wrong HTTP method returns 405."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.post("/health")
        assert response.status_code == 405


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_available(self):
        """OpenAPI schema is accessible."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_schema_has_health_endpoint(self):
        """OpenAPI schema includes health endpoint."""
        app = create_app(test_mode=True)
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        assert "/health" in schema["paths"]

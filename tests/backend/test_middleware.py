"""Tests for the production middleware stack.

These guard behaviour that only shows up under conditions a manual click
through the demo never reaches: the 31st request in a minute, a body
larger than the limit, a handler that raises.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware import (
    REQUEST_ID_HEADER,
    SECURITY_HEADERS,
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)


def build_app(*middleware_specs: tuple[type, dict[str, object]]) -> FastAPI:
    """Minimal app carrying only the middleware under test."""
    app = FastAPI()
    for cls, kwargs in middleware_specs:
        app.add_middleware(cls, **kwargs)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search/text")
    async def search() -> dict[str, str]:
        return {"result": "ok"}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("handler exploded")

    return app


class TestRequestContext:
    def test_adds_a_request_id(self):
        client = TestClient(build_app((RequestContextMiddleware, {})))
        response = client.get("/health")
        assert response.headers[REQUEST_ID_HEADER]

    def test_echoes_a_client_supplied_id(self):
        """A caller correlating its own logs must get its id back."""
        client = TestClient(build_app((RequestContextMiddleware, {})))
        response = client.get("/health", headers={REQUEST_ID_HEADER: "abc123"})
        assert response.headers[REQUEST_ID_HEADER] == "abc123"

    def test_ids_are_unique_per_request(self):
        client = TestClient(build_app((RequestContextMiddleware, {})))
        first = client.get("/health").headers[REQUEST_ID_HEADER]
        second = client.get("/health").headers[REQUEST_ID_HEADER]
        assert first != second

    def test_reports_processing_time(self):
        client = TestClient(build_app((RequestContextMiddleware, {})))
        response = client.get("/health")
        assert response.headers["X-Process-Time"].endswith("ms")

    def test_logs_and_reraises_handler_errors(self, caplog):
        """The 500 handler does not know the request id; this must log it."""
        client = TestClient(
            build_app((RequestContextMiddleware, {})), raise_server_exceptions=False
        )
        with caplog.at_level("ERROR"):
            client.get("/boom")
        assert any("request failed" in record.message for record in caplog.records)


class TestSecurityHeaders:
    def test_applies_every_header(self):
        client = TestClient(build_app((SecurityHeadersMiddleware, {})))
        response = client.get("/health")
        for header, value in SECURITY_HEADERS.items():
            assert response.headers[header] == value

    def test_does_not_set_hsts(self):
        """HSTS over plain HTTP is meaningless; it belongs on the proxy."""
        client = TestClient(build_app((SecurityHeadersMiddleware, {})))
        response = client.get("/health")
        assert "Strict-Transport-Security" not in response.headers


class TestRateLimit:
    def test_allows_requests_under_the_limit(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 3}))
        )
        for _ in range(3):
            assert client.post("/search/text").status_code == 200

    def test_rejects_the_request_past_the_limit(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 3}))
        )
        for _ in range(3):
            client.post("/search/text")
        response = client.post("/search/text")
        assert response.status_code == 429
        assert response.json()["error"] == "rate_limited"

    def test_sets_retry_after(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 1}))
        )
        client.post("/search/text")
        response = client.post("/search/text")
        assert int(response.headers["Retry-After"]) >= 1

    def test_reports_remaining_budget(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 5}))
        )
        response = client.post("/search/text")
        assert response.headers["X-RateLimit-Limit"] == "5"
        assert response.headers["X-RateLimit-Remaining"] == "4"

    def test_does_not_meter_unlisted_paths(self):
        """Throttling /health would break the status dot for no benefit."""
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 1}))
        )
        for _ in range(10):
            assert client.get("/health").status_code == 200

    def test_window_slides(self):
        client = TestClient(
            build_app(
                (
                    RateLimitMiddleware,
                    {"max_requests": 2, "window_seconds": 0.25},
                )
            )
        )
        for _ in range(2):
            client.post("/search/text")
        assert client.post("/search/text").status_code == 429
        time.sleep(0.3)
        assert client.post("/search/text").status_code == 200

    def test_separate_clients_have_separate_budgets(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 1}))
        )
        assert (
            client.post(
                "/search/text", headers={"x-forwarded-for": "10.0.0.1"}
            ).status_code
            == 200
        )
        # A second address must not inherit the first one's spent budget,
        # or one visitor behind a proxy would lock out everyone else.
        assert (
            client.post(
                "/search/text", headers={"x-forwarded-for": "10.0.0.2"}
            ).status_code
            == 200
        )

    def test_uses_the_leftmost_forwarded_address(self):
        client = TestClient(
            build_app((RateLimitMiddleware, {"max_requests": 1}))
        )
        headers = {"x-forwarded-for": "10.0.0.9, 172.16.0.1, 192.168.1.1"}
        assert client.post("/search/text", headers=headers).status_code == 200
        assert client.post("/search/text", headers=headers).status_code == 429

    @pytest.mark.parametrize(
        "kwargs", [{"max_requests": 0}, {"window_seconds": 0}]
    )
    def test_rejects_nonsense_configuration(self, kwargs):
        with pytest.raises(ValueError):
            TestClient(build_app((RateLimitMiddleware, kwargs))).get("/health")


class TestMaxBodySize:
    def test_allows_a_body_under_the_limit(self):
        client = TestClient(build_app((MaxBodySizeMiddleware, {"max_bytes": 1024})))
        assert client.post("/search/text", content=b"x" * 100).status_code == 200

    def test_rejects_an_oversized_body(self):
        client = TestClient(build_app((MaxBodySizeMiddleware, {"max_bytes": 1024})))
        response = client.post("/search/text", content=b"x" * 2048)
        assert response.status_code == 413
        assert response.json()["error"] == "payload_too_large"

    def test_rejects_before_reading_the_body(self):
        """Content-Length is checked first, so the bytes are never buffered."""
        client = TestClient(build_app((MaxBodySizeMiddleware, {"max_bytes": 10})))
        response = client.post(
            "/search/text",
            content=b"x" * 5000,
            headers={"Content-Length": "5000"},
        )
        assert response.status_code == 413

    def test_rejects_nonsense_configuration(self):
        with pytest.raises(ValueError):
            TestClient(build_app((MaxBodySizeMiddleware, {"max_bytes": 0}))).get(
                "/health"
            )

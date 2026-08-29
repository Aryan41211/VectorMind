"""Tests for backend/metrics.py and the /metrics endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.metrics import METRICS, Metrics


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    """Give every test a clean global registry.

    The middleware writes into the process-wide ``METRICS`` singleton,
    and other test modules (test_app.py etc.) hit the same paths through
    their clients, so leaving state between tests would make the
    endpoint assertions depend on which other tests ran first.
    """
    METRICS.reset()


def test_record_increments_counter_and_histogram() -> None:
    """A request shows up in both the counter and the histogram."""
    m = Metrics()
    m.record("/search/text", 200, 12.5)
    m.record("/search/text", 200, 2000.0)
    m.record("/search/image", 429, 3.0)

    text = m.render()
    assert 'vectormind_http_requests_total{path="/search/text",status="200"} 2' in text
    assert 'vectormind_http_requests_total{path="/search/image",status="429"} 1' in text
    # The 2s observation lands in the +Inf/2500 buckets, not below 1000ms,
    # so the cumulative le=1000 count reflects only the 12.5ms one; le=1
    # reflects neither.
    assert 'vectormind_http_request_duration_ms_bucket{path="/search/text",le="1"} 0' in text
    assert 'vectormind_http_request_duration_ms_bucket{path="/search/text",le="25"} 1' in text
    assert 'vectormind_http_request_duration_ms_bucket{path="/search/text",le="1000"} 1' in text
    assert 'vectormind_http_request_duration_ms_bucket{path="/search/text",le="2500"} 2' in text
    assert 'vectormind_http_request_duration_ms_sum{path="/search/text"} 2012.500' in text
    assert 'vectormind_http_request_duration_ms_count{path="/search/text"} 2' in text
    assert 'vectormind_http_request_duration_ms_sum{path="/search/image"}' in text


def test_histogram_buckets_are_cumulative() -> None:
    """Each bucket's count includes every slower observation."""
    m = Metrics()
    m.record("/s", 200, 5.0)  # lands in le=5 and every slower bucket
    text = m.render()
    prefix = "vectormind_http_request_duration_ms_bucket{path=\"/s\","
    assert prefix + 'le="5"} 1' in text
    assert prefix + 'le="10"} 1' in text
    assert prefix + 'le="2500"} 1' in text
    assert prefix + 'le="+Inf"} 1' in text
    # Slower than the le=1 bucket, so it is absent there.
    assert prefix + 'le="1"} 0' in text


def test_render_has_up_and_uptime_gauges() -> None:
    """The registry always reports process liveness."""
    m = Metrics()
    text = m.render()
    assert "vectormind_up 1.0" in text
    assert "vectormind_uptime_seconds" in text


def test_metrics_endpoint_returns_prometheus_text() -> None:
    """GET /metrics returns versioned text and reflects recorded requests."""
    app = create_app(test_mode=True)
    client = TestClient(app)

    # Hit the app so the health path is recorded by the middleware.
    client.get("/health")
    client.get("/api/info")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    body = response.text
    assert 'vectormind_http_requests_total{path="/health",status="200"} 1' in body
    assert 'vectormind_http_requests_total{path="/api/info",status="200"} 1' in body
    assert "vectormind_up 1.0" in body


def test_metrics_endpoint_excludes_itself() -> None:
    """Scraping /metrics does not grow its own counter (no feedback loop)."""
    app = create_app(test_mode=True)
    client = TestClient(app)
    client.get("/metrics")
    client.get("/metrics")
    body = client.get("/metrics").text
    assert 'vectormind_http_requests_total{path="/metrics"' not in body

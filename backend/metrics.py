"""Dependency-free Prometheus text-format metrics for the serving app.

Purpose: expose the ``/metrics`` endpoint that deployment tooling asks
for, without pulling in prometheus_client. The Prometheus text
exposition format is stable and simple enough to emit by hand for the
two metric kinds a single-node demo needs — a counter and a histogram —
which keeps the serving image slim and the behaviour fully testable
with pure Python.

Why not prometheus_client: this project deliberately runs no
observability stack (docs/DEPLOYMENT.md), and the same rational that
kept tensorboard out of the serving image applies to a metrics client
that only a stack would consume. This module is the endpoint, not the
stack: it emits standard text a Prometheus server could scrape if one
is ever added.

Thread safety: all updates go through a single lock. The request path
is asyncio-concurrent, and a torn histogram read mid-update would emit
arithmetic that scrapers cannot sum — one lock is cheap relative to
the search work it protects.

Dependencies: stdlib only. No dependency on backend.app, matching
backend/middleware.py's import independence.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

# Latency buckets in milliseconds, cumulative. The range spans the
# measured CPU curve (8-19ms warm, ~2s cold) — docs/DEPLOYMENT.md and
# ROADMAP.md Phase 7.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    float("inf"),
)

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class Metrics:
    """Thread-safe counters and a latency histogram, in Prometheus form.

    Attributes:
        _counter: ``{(endpoint, status_code): count}`` request totals.
        _hist_count: ``{endpoint: count}`` histogram observation totals.
        _hist_sum: ``{endpoint: total_ms}`` sum of observed latencies.
        _hist_buckets: ``{endpoint: {bucket_le: cumulative_count}}``.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._counter: dict[tuple[str, str], int] = defaultdict(int)
        self._hist_count: dict[str, int] = defaultdict(int)
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_buckets: dict[str, dict[float, int]] = {}
        self._started_monotonic = time.monotonic()

    def reset(self) -> None:
        """Clear all recorded observations.

        Primarily for tests, which share the process-wide ``METRICS``
        singleton and need a clean slate; also useful if a deployment
        ever wants to rotate counters on a schedule.
        """
        with self._lock:
            self._counter.clear()
            self._hist_count.clear()
            self._hist_sum.clear()
            self._hist_buckets.clear()
            self._started_monotonic = time.monotonic()

    def record(self, endpoint: str, status_code: int, duration_ms: float) -> None:
        """Record one completed request.

        Args:
            endpoint: The request path used as the metric label.
            status_code: The HTTP status returned.
            duration_ms: End-to-end handler latency in milliseconds.
        """
        label = str(status_code)
        with self._lock:
            self._counter[(endpoint, label)] += 1
            buckets = self._hist_buckets.get(endpoint)
            if buckets is None:
                buckets = {b: 0 for b in LATENCY_BUCKETS_MS}
                self._hist_buckets[endpoint] = buckets
            self._hist_count[endpoint] += 1
            self._hist_sum[endpoint] += duration_ms
            # Cumulative histogram: every bucket with le >= value counts
            # this observation, so a scrape sees monotonically increasing
            # bucket counts across the series.
            for bucket_le in LATENCY_BUCKETS_MS:
                if duration_ms <= bucket_le:
                    buckets[bucket_le] += 1

    def render(self) -> str:
        """Render all metrics as Prometheus text exposition.

        Returns:
            The full ``text/plain`` body for the ``/metrics`` endpoint.
        """
        uptime_seconds = time.monotonic() - self._started_monotonic
        lines: list[str] = []
        lines.append("# HELP vectormind_up Whether the process is up.")
        lines.append("# TYPE vectormind_up gauge")
        lines.append(f"vectormind_up 1.0")
        lines.append("# HELP vectormind_uptime_seconds Seconds since startup.")
        lines.append("# TYPE vectormind_uptime_seconds gauge")
        lines.append(f"vectormind_uptime_seconds {uptime_seconds:.3f}")

        lines.append("# HELP vectormind_http_requests_total HTTP requests by path and status.")
        lines.append("# TYPE vectormind_http_requests_total counter")
        with self._lock:
            for (endpoint, status), count in sorted(self._counter.items()):
                lines.append(
                    f'vectormind_http_requests_total{{path="{endpoint}",'
                    f'status="{status}"}} {count}'
                )

            lines.append("# HELP vectormind_http_request_duration_ms HTTP latency histogram.")
            lines.append("# TYPE vectormind_http_request_duration_ms histogram")
            for endpoint in sorted(self._hist_buckets):
                for bucket_le in LATENCY_BUCKETS_MS:
                    bucket_key = f'le="{bucket_le:.6g}"' if bucket_le != float("inf") else 'le="+Inf"'
                    lines.append(
                        f'vectormind_http_request_duration_ms_bucket{{path="{endpoint}",'
                        f'{bucket_key}}} {self._hist_buckets[endpoint][bucket_le]}'
                    )
                lines.append(
                    f'vectormind_http_request_duration_ms_sum{{path="{endpoint}"}} '
                    f'{self._hist_sum[endpoint]:.3f}'
                )
                lines.append(
                    f'vectormind_http_request_duration_ms_count{{path="{endpoint}"}} '
                    f'{self._hist_count[endpoint]}'
                )
        return "\n".join(lines) + "\n"


# Single registry shared by the whole process.
METRICS = Metrics()
CONTENT_TYPE = _CONTENT_TYPE

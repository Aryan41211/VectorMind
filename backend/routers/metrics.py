"""Prometheus-format metrics endpoint.

Separated into its own router (rather than a closure in backend/app.py)
so the app module stays lean and the metrics contract lives beside the
registry it reads — backend/metrics.py.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.metrics import CONTENT_TYPE, METRICS

router = APIRouter()


@router.get("/metrics", tags=["health"], include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Expose request counters and a latency histogram as text.

    Exposed but not scraped by this stack — it is the endpoint an
    operator's Prometheus server would point at, instead of this
    project running its own observability stack (docs/DEPLOYMENT.md).
    The Content-Type is the versioned exposition format so a Prometheus
    scraper can parse it as-is.
    """
    return PlainTextResponse(content=METRICS.render(), media_type=CONTENT_TYPE)

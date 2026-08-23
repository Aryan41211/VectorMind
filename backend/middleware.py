"""Production middleware: request identity, rate limiting, security headers.

Purpose: the cross-cutting concerns a public endpoint needs and that the
application logic should not have to carry.

Scope note. The rate limiter here is deliberately in-process and
in-memory. That is correct for this deployment — one uvicorn worker on
one machine (ARCHITECTURE.md §11) — and wrong for any deployment with
more than one process, where each would enforce its own separate budget.
It is a guard against a single client hammering a 6GB laptop GPU, not a
security control. Anything multi-process needs shared state (Redis) or
an upstream limiter; see docs/FUTURE_IDEAS.md.

Dependencies: starlette only. No dependency on backend.app, so importing
this module cannot create the circular import that backend.routers has.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Header carrying a correlation id, echoed so a client can quote it when
# reporting a problem.
REQUEST_ID_HEADER = "X-Request-ID"

# Static headers applied to every response.
#
# HSTS is deliberately absent: it is meaningless over plain HTTP and
# actively harmful to set from an app that may be served without TLS —
# it belongs on the TLS terminator, which is where the certificate is.
SECURITY_HEADERS: dict[str, str] = {
    # The API serves JSON and the SPA serves its own assets; nothing here
    # should ever be sniffed into another type.
    "X-Content-Type-Options": "nosniff",
    # No part of this app is meant to be embedded.
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # This app uses no camera, microphone, or geolocation.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and timing to every response and log line.

    Without a correlation id, the access log and an error traceback
    cannot be tied together, which is the first thing anyone needs when
    diagnosing a request that failed in production.
    """

    async def dispatch(  # noqa: D102 - contract documented on the class
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            # Log before re-raising: the exception handler that turns this
            # into a 500 does not know the request id or the timing.
            logger.exception(
                "request failed | id=%s method=%s path=%s duration=%.1fms",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"

        # Health polls every 30s per open tab and would drown the log.
        if request.url.path not in ("/health", "/ready"):
            logger.info(
                "%s %s -> %d | id=%s duration=%.1fms",
                request.method,
                request.url.path,
                response.status_code,
                request_id,
                duration_ms,
            )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply the static security headers to every response."""

    async def dispatch(  # noqa: D102 - contract documented on the class
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-client rate limit on the search endpoints.

    A sliding window rather than a fixed one: a fixed window lets a
    client spend its whole budget at 0:59 and again at 1:01, producing a
    burst of double the intended rate against a single GPU.

    Only the paths in ``limited_prefixes`` are counted. Health polling
    and static assets are unmetered — throttling them would break the
    status indicator without protecting anything expensive.

    Attributes:
        max_requests: Requests permitted per window, per client.
        window_seconds: Window length.
        limited_prefixes: Path prefixes the limit applies to.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 30,
        window_seconds: float = 60.0,
        limited_prefixes: tuple[str, ...] = ("/search",),
    ) -> None:
        """Initialize the limiter.

        Args:
            app: The ASGI application to wrap.
            max_requests: Requests permitted per window, per client.
            window_seconds: Window length in seconds.
            limited_prefixes: Path prefixes to meter.

        Raises:
            ValueError: If ``max_requests`` or ``window_seconds`` is not
                positive.
        """
        super().__init__(app)
        if max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {max_requests}.")
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}."
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.limited_prefixes = limited_prefixes
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        """Identify the caller.

        Prefers the left-most X-Forwarded-For entry so a reverse proxy
        does not collapse every client into one bucket. This is
        spoofable and must not be relied on for anything but throttling.

        Args:
            request: The incoming request.

        Returns:
            A stable key for this client.
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(  # noqa: D102 - contract documented on the class
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not request.url.path.startswith(self.limited_prefixes):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        hits = self._hits[key]

        # Drop everything that has aged out of the window.
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.max_requests:
            retry_after = max(1, int(hits[0] + self.window_seconds - now) + 1)
            logger.warning(
                "rate limit hit | client=%s path=%s", key, request.url.path
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        f"Limit is {self.max_requests} searches per "
                        f"{int(self.window_seconds)}s. Retry in {retry_after}s."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)

        # Bound memory: an empty deque per client would otherwise
        # accumulate forever across many distinct addresses.
        if not hits:
            self._hits.pop(key, None)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - len(hits))
        )
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized uploads before they are read into memory.

    FastAPI would otherwise buffer the whole body before validation runs,
    so a large upload costs the memory regardless of being rejected
    afterwards. Checking Content-Length first refuses it at the door.

    A client can lie about or omit Content-Length; this is a guard
    against accidents, not an attacker. A real bound belongs in the
    reverse proxy (`client_max_body_size` in nginx).
    """

    def __init__(self, app: ASGIApp, max_bytes: int = 10 * 1024 * 1024) -> None:
        """Initialize the size guard.

        Args:
            app: The ASGI application to wrap.
            max_bytes: Largest request body accepted.

        Raises:
            ValueError: If ``max_bytes`` is not positive.
        """
        super().__init__(app)
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}.")
        self.max_bytes = max_bytes

    async def dispatch(  # noqa: D102 - contract documented on the class
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self.max_bytes:
                limit_mb = self.max_bytes / 1024 / 1024
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "detail": f"Request body exceeds {limit_mb:.0f}MB.",
                    },
                )
        return await call_next(request)

"""Concurrent load test against a running VectorMind API.

Purpose: measure how the API holds up under concurrent text-search load
— the percentile latency curve and throughput — instead of the single
serial benchmark in ROADMAP.md Phase 7, which cannot see contention.

The default concurrency (16) is deliberately modest: the backend runs as
a single uvicorn worker on one machine and the rate limiter budgets 30
searches/minute per client by default (configs/serving.yaml), so the
load driver sends each request with a fresh X-Forwarded-For to avoid
tripping its own client budget. Raise --concurrency to probe.

Usage:
    python scripts/load_test_api.py \
        --base-url http://127.0.0.1:8000 \
        --concurrency 16 --total 200

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_QUERY = "a dog playing in a park"
TIMEOUT_SECONDS = 30


def _send_search(base: str, query: str, top_k: int) -> tuple[bool, float]:
    """Send one text-search request, returning (ok, latency_ms)."""
    payload = {"query": query, "top_k": top_k}
    headers = {"X-Forwarded-For": f"10.0.0.{int(time.time() * 1000) % 255 + 1}"}
    start = time.perf_counter()
    try:
        r = requests.post(
            f"{base}/search/text",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        ok = r.status_code == 200
    except requests.RequestException:
        ok = False
    return ok, (time.perf_counter() - start) * 1000


def _percentile(values: np.ndarray, q: float) -> float:
    """Return the q-th percentile of an array, or nan when empty."""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def main(argv: list[str] | None = None) -> int:
    """Run the load test and print a summary."""
    parser = argparse.ArgumentParser(description="VectorMind concurrent load test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Backend URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent workers (default: 16)")
    parser.add_argument("--total", type=int, default=200,
                        help="Total requests to send (default: 200)")
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help=f"Query text (default: {DEFAULT_QUERY!r})")
    parser.add_argument("--top-k", type=int, default=10,
                        help="top_k per request (default: 10)")
    parser.add_argument("--max-error-rate", type=float, default=0.05,
                        help="Pass threshold for the error rate (default: 0.05)")
    args = parser.parse_args(argv)

    if args.concurrency <= 0 or args.total <= 0:
        print("concurrency and total must both be positive", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    print(f"Load test: {args.total} requests, concurrency={args.concurrency}")
    print(f"Target: POST {base}/search/text")

    work = [(base, args.query, args.top_k)] * args.total
    latencies: list[float] = []
    errors = 0
    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = pool.map(lambda item: _send_search(*item), work)
        for i, (ok, latency_ms) in enumerate(results):
            if ok:
                latencies.append(latency_ms)
            else:
                errors += 1
            if (i + 1) % 50 == 0 or i + 1 == args.total:
                print(f"  {i + 1}/{args.total} done (errors so far: {errors})")

    wall_seconds = time.perf_counter() - start_wall
    arr = np.array(latencies)
    throughput = args.total / wall_seconds if wall_seconds > 0 else float("inf")

    print("\n" + "=" * 46)
    print("Result")
    print("=" * 46)
    print(f"  Requests sent:      {args.total}")
    print(f"  Successful:         {len(latencies)}")
    print(f"  Errors:             {errors}")
    print(f"  Wall time:          {wall_seconds:.2f}s")
    print(f"  Throughput:         {throughput:.1f} req/s")
    print(f"  Error rate:         {errors / args.total:.1%}")
    print("  Latency (ms):")
    print(f"    min   {arr.min():.1f}" if arr.size else "    min   n/a")
    print(f"    avg   {arr.mean():.1f}" if arr.size else "    avg   n/a")
    print(f"    p50   {_percentile(arr, 50):.1f}")
    print(f"    p90   {_percentile(arr, 90):.1f}")
    print(f"    p95   {_percentile(arr, 95):.1f}")
    print(f"    p99   {_percentile(arr, 99):.1f}")
    print(f"    max   {arr.max():.1f}" if arr.size else "    max   n/a")

    error_rate = errors / args.total
    ok = error_rate <= args.max_error_rate
    print("\n" + ("PASS: error rate within threshold." if ok
                  else "FAIL: error rate exceeded threshold."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

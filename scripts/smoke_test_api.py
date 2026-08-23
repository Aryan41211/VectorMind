"""Smoke test: verify the VectorMind API endpoints are reachable and functional.

Purpose: validates that the backend serves health checks, text search,
image search, and static frontend files correctly. This is the Phase 6.5
acceptance gate for the serving layer.

Usage:
    python scripts/smoke_test_api.py [--base-url http://localhost:8000]

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 10


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="VectorMind API smoke test")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Backend URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    all_passed = True

    # --- Health ---
    print("\n1. Health endpoint")
    try:
        r = requests.get(f"{base}/health", timeout=TIMEOUT_SECONDS)
        all_passed &= _check("GET /health returns 200", r.status_code == 200)
        data = r.json()
        all_passed &= _check(
            "Response has status=healthy",
            data.get("status") == "healthy",
            f"got {data.get('status')}",
        )
    except Exception as e:
        all_passed &= _check("Health endpoint reachable", False, str(e))

    # --- API Info ---
    print("\n2. API info endpoint")
    try:
        r = requests.get(f"{base}/api/info", timeout=TIMEOUT_SECONDS)
        all_passed &= _check("GET /api/info returns 200", r.status_code == 200)
        data = r.json()
        all_passed &= _check(
            "Response has name=VectorMind",
            data.get("name") == "VectorMind",
            f"got {data.get('name')}",
        )
    except Exception as e:
        all_passed &= _check("API info endpoint reachable", False, str(e))

    # --- Text search ---
    print("\n3. Text search endpoint")
    try:
        payload = {"query": "a dog playing in a park", "top_k": 5}
        r = requests.post(
            f"{base}/search/text",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        all_passed &= _check("POST /search/text returns 200", r.status_code == 200)
        data = r.json()
        all_passed &= _check(
            "Response has results list",
            isinstance(data.get("results"), list),
            f"got {type(data.get('results')).__name__}",
        )
        all_passed &= _check(
            "Results count <= top_k",
            len(data.get("results", [])) <= 5,
            f"got {len(data.get('results', []))}",
        )
        if data.get("results"):
            first = data["results"][0]
            all_passed &= _check(
                "Result has required fields",
                all(k in first for k in ("filename", "score", "caption")),
                f"keys={list(first.keys())}",
            )
            all_passed &= _check(
                "Latency reported",
                "latency_ms" in data,
                f"latency_ms={data.get('latency_ms')}",
            )
    except Exception as e:
        all_passed &= _check("Text search endpoint reachable", False, str(e))

    # --- Image search ---
    print("\n4. Image search endpoint")
    try:
        # Create a tiny valid JPEG for the upload test
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            img.save(tmp.name, "JPEG")
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{base}/search/image?top_k=3",
                files={"file": ("test.jpg", f, "image/jpeg")},
                timeout=TIMEOUT_SECONDS,
            )
        all_passed &= _check("POST /search/image returns 200", r.status_code == 200)
        data = r.json()
        all_passed &= _check(
            "Response has results list",
            isinstance(data.get("results"), list),
            f"got {type(data.get('results')).__name__}",
        )

        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        all_passed &= _check("Image search endpoint reachable", False, str(e))

    # --- Frontend static serving ---
    print("\n5. Static frontend serving")
    try:
        r = requests.get(f"{base}/", timeout=TIMEOUT_SECONDS)
        all_passed &= _check("GET / returns 200", r.status_code == 200)
        all_passed &= _check(
            "Response is HTML",
            "html" in r.headers.get("content-type", ""),
            f"content-type={r.headers.get('content-type')}",
        )
    except Exception as e:
        all_passed &= _check("Static frontend reachable", False, str(e))

    # --- Summary ---
    print()
    if all_passed:
        print("All smoke tests passed.")
        return 0
    else:
        print("Some smoke tests FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

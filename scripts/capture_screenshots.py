r"""Capture screenshots of the running demo for the README.

Purpose: produce the walkthrough images from the *real* application
rather than mockups, so what a reader sees is what the model actually
returns. Every screenshot is a live query against a running stack.

Deliberately reproducible: the queries are fixed, the viewport is fixed,
and the theme is set explicitly rather than inherited from whatever the
capturing machine prefers. Re-running this after a retrain regenerates
the whole set consistently.

Usage:
    # with the stack running (see docs/DEPLOYMENT.md)
    python scripts/capture_screenshots.py --base-url http://localhost:8081

    # if the private-demo auth overlay is enabled
    python scripts/capture_screenshots.py --user demo --password '...'

Writes PNGs into docs/screenshots/.

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# A desktop viewport wide enough for the four-column result grid.
#
# Scale 1, not 2: a README renders these around 800px wide, so a 2880px
# source buys nothing visible and cost 13.7MB across four files. At 1x
# the set is under 4MB and looks identical in place.
VIEWPORT = {"width": 1440, "height": 900}
SCALE = 1

# Fixed so a rerun produces a comparable set. Chosen to show the model
# honestly: the first three are the kind of scene-level query it handles,
# the fourth is deliberately out of domain.
SHOTS: list[dict[str, str]] = [
    {
        "name": "01-idle",
        "query": "",
        "caption": "Idle state: search, example queries, and the metrics panel.",
    },
    {
        "name": "02-search-dog",
        "query": "a dog playing in a park",
        "caption": "Text to image, with similarity scores on every result.",
    },
    {
        "name": "03-search-street",
        "query": "a busy city street with cars",
        "caption": "Scene-level concepts are what the model handles best.",
    },
    {
        "name": "04-search-out-of-domain",
        "query": "a quarterly revenue chart",
        "caption": (
            "Out of domain. Flickr30k has no charts, so the scores drop "
            "and the results are the nearest photographs rather than "
            "matches — which is what the scores are there to show."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Capture README screenshots from the running demo"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8081",
        help="Where the demo is served.",
    )
    parser.add_argument(
        "--output",
        default="docs/screenshots",
        help="Directory to write PNGs into.",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="dark",
        help="Theme to capture. Set explicitly so results are reproducible.",
    )
    parser.add_argument("--user", default=None, help="Basic-auth username.")
    parser.add_argument("--password", default=None, help="Basic-auth password.")
    return parser.parse_args()


def main() -> None:
    """Capture every screenshot in SHOTS."""
    args = parse_args()
    setup_logging(level=logging.INFO)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright is not installed. It is a capture-time tool, not a "
            "project dependency:\n"
            "  pip install playwright && playwright install chromium"
        ) from None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    credentials = (
        {"username": args.user, "password": args.password}
        if args.user and args.password
        else None
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=SCALE,
            http_credentials=credentials,
            # The app reads prefers-color-scheme when no explicit choice
            # is stored; setting it here avoids depending on the host.
            color_scheme=args.theme,
        )
        page = context.new_page()

        logger.info("Loading %s", args.base_url)
        page.goto(args.base_url, wait_until="networkidle")

        # The health indicator polls on mount; let it settle so the
        # screenshot shows a resolved state rather than "Checking…".
        page.wait_for_timeout(2500)

        for shot in SHOTS:
            if shot["query"]:
                logger.info("Query: %s", shot["query"])
                page.fill("#search-query", shot["query"])
                page.press("#search-query", "Enter")
                # Results replace the skeleton; wait for a real card.
                page.wait_for_selector("img[loading='lazy']", timeout=30_000)
                # Let the grid's fade-in finish so nothing is captured
                # mid-animation.
                page.wait_for_timeout(1200)

            destination = output_dir / f"{shot['name']}.png"
            page.screenshot(path=str(destination), full_page=True)
            logger.info("Wrote %s", destination)

        browser.close()

    logger.info("Captured %d screenshots into %s", len(SHOTS), output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Environment verification script for VectorMind.

Checks all required dependencies and their versions, CUDA availability,
and mixed precision support.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version
from typing import Any


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v.major == 3 and v.minor == 12
    return ok, f"Python {v.major}.{v.minor}.{v.micro} (need 3.12.x)"


def check_torch() -> tuple[bool, str]:
    import torch
    ok = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if ok else "CPU"
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if ok else "N/A"
    return ok, f"PyTorch {torch.__version__} | CUDA: {ok} | Device: {device} | VRAM: {vram} GB"


def check_amp() -> tuple[bool, str]:
    import torch
    try:
        with torch.autocast(device_type="cuda", enabled=True):
            x = torch.randn(2, 2, device="cuda")
            _ = x @ x.T
        return True, "AMP (torch.autocast) works on CUDA"
    except Exception as e:
        return False, f"AMP failed: {e}"


def check_import(name: str, min_version: str | None = None) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        ok = True
        if min_version:
            from packaging.version import parse as parse_version
            ok = parse_version(ver) >= parse_version(min_version)
        return ok, f"{name} {ver}" + ("" if ok else f" (need >= {min_version})")
    except Exception as e:
        return False, f"{name}: NOT FOUND ({e})"


def main() -> int:
    print("=" * 60)
    print("VectorMind Environment Verification")
    print("=" * 60)

    checks: list[tuple[bool, str]] = []

    # Core
    checks.append(check_python())
    checks.append(check_torch())
    checks.append(check_amp())

    # Dependencies
    deps = [
        ("torchvision", "0.17"),
        ("transformers", "4.0"),
        ("tokenizers", "0.15"),
        ("faiss", "1.0"),
        ("cv2", "4.0"),
        ("PIL", "9.0"),
        ("fastapi", "0.100"),
        ("uvicorn", "0.20"),
        ("yaml", "5.0"),  # PyYAML
        ("wandb", "0.16"),
        ("pytest", "7.0"),
    ]

    for name, min_ver in deps:
        checks.append(check_import(name, min_ver))

    # Print results
    print("\nResults:")
    print("-" * 60)
    all_ok = True
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {msg}")

    print("-" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
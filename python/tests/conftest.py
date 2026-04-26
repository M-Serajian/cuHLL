"""pytest configuration for the cuhll Python bindings.

Tests requiring a CUDA GPU are gated by the `gpu` marker. If no GPU is
visible (nvidia-smi missing or returning no devices), every gpu-marked
test is skipped instead of failing — same pattern the C++ tests use,
keeps `pytest` runnable on laptops/CI without a GPU.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _gpu_available() -> bool:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False
    try:
        out = subprocess.run(
            [nvsmi, "-L"], capture_output=True, text=True, timeout=5
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


_HAS_GPU = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if _HAS_GPU:
        return
    skip_marker = pytest.mark.skip(
        reason="no CUDA GPU detected (nvidia-smi missing or reports no devices)"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)

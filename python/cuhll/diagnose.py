"""Environment diagnostic for cuhll.

Run before installing to verify your toolchain, or after a build/import
failure to collect everything that should accompany a bug report:

    cuhll-diagnose                              # if cuhll is installed
    python python/cuhll/diagnose.py             # standalone (works pre-install)

Prints versions and paths for: Python, OS, GCC/Clang, CMake, Ninja,
nvcc, NVIDIA driver, visible GPUs, scikit-build-core, pybind11, numpy,
plus any CC/CXX/CUDA_HOME/LD_LIBRARY_PATH/PATH overrides that affect
the build. The output is one self-contained block of text suitable for
pasting into a GitHub issue.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _run(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (result.stdout + result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"<not available: {exc!s}>"


def _which(prog):
    return shutil.which(prog) or "<not on PATH>"


def _import_version(name):
    try:
        module = __import__(name.replace("-", "_"))
        return getattr(module, "__version__", "?")
    except ImportError:
        return "<not installed>"


def main() -> int:
    print("=" * 72)
    print("cuHLL environment diagnostic")
    print("Include this entire output when reporting a build/import failure.")
    print("=" * 72)

    print("\n## Python")
    print(f"  version : {sys.version.split()[0]}")
    print(f"  exec    : {sys.executable}")
    print(f"  prefix  : {sys.prefix}")

    try:
        import cuhll
        print("\n## cuhll (installed)")
        print(f"  version : {cuhll.__version__}")
        print(f"  path    : {cuhll.__file__}")
    except ImportError as exc:
        print(f"\n## cuhll (not yet installed): {type(exc).__name__}: {exc}")

    print("\n## OS")
    print(f"  platform: {platform.platform()}")
    print(f"  arch    : {platform.machine()}")

    print("\n## Compilers and build tools")
    for tool, args in [
        ("g++",     ["g++", "--version"]),
        ("clang++", ["clang++", "--version"]),
        ("cmake",   ["cmake", "--version"]),
        ("ninja",   ["ninja", "--version"]),
        ("nvcc",    ["nvcc", "--version"]),
    ]:
        path = _which(tool)
        first_line = "n/a"
        if path != "<not on PATH>":
            out = _run(args)
            first_line = out.splitlines()[0] if out else "n/a"
        print(f"  {tool:<8} : {path}")
        print(f"           {first_line}")

    print("\n## NVIDIA driver / GPUs")
    if shutil.which("nvidia-smi"):
        print(_run(["nvidia-smi", "-L"]))
        print(_run(["nvidia-smi",
                    "--query-gpu=driver_version,name", "--format=csv"]))
    else:
        print("  nvidia-smi not on PATH (login node, CI runner, or driver missing)")

    print("\n## Python build/runtime packages")
    for pkg in ("pip", "scikit_build_core", "pybind11",
                "numpy", "pytest", "cmake", "ninja"):
        print(f"  {pkg:<20}: {_import_version(pkg)}")

    print("\n## Relevant environment variables")
    for var in ("CC", "CXX", "CUDA_HOME", "CUDAToolkit_ROOT",
                "CMAKE_CUDA_ARCHITECTURES", "CMAKE_BUILD_PARALLEL_LEVEL",
                "LD_LIBRARY_PATH", "PATH"):
        val = os.environ.get(var, "<unset>")
        if len(val) > 200:
            val = val[:200] + "...<truncated>"
        print(f"  {var}: {val}")

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

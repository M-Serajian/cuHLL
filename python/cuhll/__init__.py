"""cuHLL — CUDA-accelerated HyperLogLog sketches for genomic k-mers.

Quick examples
--------------

>>> import cuhll
>>> # Cardinality of one FASTA, single call
>>> est = cuhll.estimate("input.fasta", k=31)

>>> # Sketch in memory, manipulate, save
>>> s = cuhll.sketch("input.fasta", k=31, precision=14)
>>> s.estimate()
>>> s.write("input.hll")

>>> # Load a sketch
>>> s2 = cuhll.read("input.hll")

>>> # Set-union via `|` (returns new Sketch) or in-place via merge
>>> union = s | s2
>>> s.merge(s2)

>>> # Set-intersection cardinality via `&` or .intersect (returns int —
>>> # HLL can estimate intersection size but not the intersection itself)
>>> n_shared = s & s2
>>> n_shared = cuhll.intersect_estimate(s, s2)            # same thing

>>> # Many FASTAs at once (uses concurrent per-genome pipeline)
>>> sketches = cuhll.sketch_many(["a.fasta", "b.fasta"], k=31)        # in RAM
>>> result = cuhll.sketch_to_dir(["a.fasta", "b.fasta"],              # to disk
...                              output_dir="sketches/", k=31)
>>> result["a.fasta"]            # output .hll path for that input
>>> result.union_estimate        # union cardinality (free side effect)

>>> # One union estimate over a panel — concurrent path, no .hll kept
>>> total = cuhll.estimate_union(["a.fasta", "b.fasta"], k=31)

The on-disk `.hll` format is the stable interchange contract; see the
README for the byte-level layout. All cardinality estimates carry HLL's
inherent ~1.04/sqrt(2**precision) standard error.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Union


# ---------------------------------------------------------------------------
# Native-stdout suppression. The concurrent C++ pipeline writes its
# auto-tune banner, kernel-occupancy line, and per-genome
# "<path>\t<estimate>" progress lines via `printf` / `std::cout`.
# Python's contextlib.redirect_stdout only intercepts Python `print`, so
# to silence those for non-verbose users we have to dup2() the underlying
# file descriptor.
#
# Buffering caveat — `std::cout` accumulates in a userspace buffer when
# stdout is a pipe (which pytest, srun, redirection, etc. all set up).
# That buffer flushes at write() boundaries determined by the C runtime,
# not the dup2 timing. Without an explicit fflush(NULL) BEFORE we restore
# fd 1 to the original, any C-buffered bytes left at /dev/null-redirect
# time would write through *after* the restore and leak onto the user's
# terminal. The libc.fflush(None) call below forces the buffer drain
# while fd 1 still points at /dev/null, so the leak is impossible.
# ---------------------------------------------------------------------------
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    _libc.fflush.argtypes = [ctypes.c_void_p]
    _libc.fflush.restype = ctypes.c_int
except OSError:                         # non-glibc systems (rare; not Linux x86_64)
    _libc = None


@contextlib.contextmanager
def _silence_native_stdout(verbose: bool):
    """Redirect both stdout (fd 1) and stderr (fd 2) to /dev/null.

    The C++ pipeline writes per-genome estimates to stdout and the
    auto-tune banner + kernel-occupancy line to stderr; silencing one
    leaves the other visible. We dup2 both, fflush around the swap to
    avoid post-restore buffer leakage.
    """
    if verbose:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    if _libc is not None:
        _libc.fflush(None)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # Flush WHILE fds 1 and 2 still point at /dev/null so leftover
        # C-buffered bytes hit the bit bucket, not the user's terminal.
        if _libc is not None:
            _libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull_fd)
        os.close(saved_out)
        os.close(saved_err)

# pybind11-built extension. Importing it here surfaces native-library
# load errors (missing CUDA runtime, ABI mismatch) at `import cuhll` time
# instead of deep inside a function call.
from . import _bindings as _core

# Version is sourced from the installed package metadata (which scikit-
# build-core copies from pyproject.toml at build time). This keeps
# pyproject.toml as the single source of truth — no drift between
# pyproject.toml, CMakeLists.txt, and __init__.py. The fallback only
# fires when the package is imported from a source checkout that was
# never `pip install`ed (rare; mostly only matters in dev shells).
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__ = _pkg_version("cuhll")
    del _pkg_version, PackageNotFoundError
except Exception:
    __version__ = "0.0.0+unknown"

# Re-export constants.
kMinK             = _core.kMinK
kMaxK             = _core.kMaxK
kMinPrecision     = _core.kMinPrecision
kMaxPrecision     = _core.kMaxPrecision
kDefaultPrecision = _core.kDefaultPrecision
kDefaultChunkMB   = _core.kDefaultChunkMB
kHllFileVersion   = _core.kHllFileVersion

# Direct re-export — callers can introspect a .hll header without the
# wrapper class.
HllFileHeader = _core.HllFileHeader

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Sketch — Pythonic wrapper over the raw _Sketch from pybind11.
#
# Why wrap rather than expose _Sketch directly:
#   * Tracks `k` (the C++ Sketch class deliberately doesn't carry k, but
#     callers always know it). Lets `s.write(path)` work without re-typing
#     k, matches the natural numpy-style "the array knows its dtype".
#   * Validates k and canonical match on `merge` — the C++ side doesn't,
#     so a bare merge of two same-precision-different-k sketches would
#     silently produce nonsense.
#   * Adds `__or__` / `__ior__` so set-union reads as `a | b` / `a |= b`,
#     matching Python's `set` / `frozenset` operators.
# ---------------------------------------------------------------------------
class Sketch:
    """In-memory HyperLogLog sketch over canonical (or non-canonical) k-mers.

    Construct via :func:`cuhll.sketch`, :func:`cuhll.read`, or directly:

    >>> s = cuhll.Sketch(precision=14, canonical=True, k=31)
    """

    __slots__ = ("_impl", "_k")

    def __init__(
        self,
        precision: int = kDefaultPrecision,
        canonical: bool = True,
        *,
        k: int,
        _impl: "_core._Sketch | None" = None,
    ) -> None:
        if _impl is None:
            _impl = _core._Sketch(precision, canonical)
        self._impl = _impl
        self._k = int(k)

    # ----- properties --------------------------------------------------
    @property
    def k(self) -> int:
        """k-mer length the sketch was built for (read-only)."""
        return self._k

    @property
    def precision(self) -> int:
        """HLL precision (`p`); register count = `2 ** precision`."""
        return self._impl.precision

    @property
    def canonical(self) -> bool:
        """True iff the sketch counts canonical k-mers (`min(fwd, rc)`)."""
        return self._impl.canonical

    @property
    def sketch_bytes(self) -> int:
        """Size of the register state in host bytes (`4 * 2**precision`)."""
        return self._impl.sketch_bytes

    # ----- core ops ----------------------------------------------------
    def estimate(self) -> int:
        """Distinct k-mer cardinality estimate (uint64)."""
        return self._impl.estimate()

    def merge(self, other: "Sketch") -> "Sketch":
        """Union ``other`` into self (in place). Returns self for chaining.

        Raises ``ValueError`` if the two sketches were built with different
        ``k``, ``precision``, or ``canonical`` mode.
        """
        if not isinstance(other, Sketch):
            raise TypeError(f"merge expects a cuhll.Sketch, got {type(other).__name__}")
        if other._k != self._k:
            raise ValueError(f"k mismatch on merge: self.k={self._k}, other.k={other._k}")
        if other.precision != self.precision:
            raise ValueError(
                f"precision mismatch on merge: "
                f"self.precision={self.precision}, other.precision={other.precision}"
            )
        if other.canonical != self.canonical:
            raise ValueError(
                f"canonical-mode mismatch on merge: "
                f"self.canonical={self.canonical}, other.canonical={other.canonical}"
            )
        self._impl.merge(other._impl)
        return self

    def clone(self) -> "Sketch":
        """Deep copy: new Sketch with identical register state, k, mode."""
        return Sketch(
            precision=self.precision,
            canonical=self.canonical,
            k=self._k,
            _impl=self._impl.clone(),
        )

    def write(self, path: PathLike) -> None:
        """Persist this sketch to a ``.hll`` file (overwrites)."""
        _core.write_hll(os.fspath(path), self._impl, self._k)

    # ----- operators ---------------------------------------------------
    def __or__(self, other: "Sketch") -> "Sketch":
        """Set-union: ``a | b`` returns a new Sketch == a ∪ b."""
        out = self.clone()
        out.merge(other)
        return out

    def __ior__(self, other: "Sketch") -> "Sketch":
        """In-place union: ``a |= b``."""
        return self.merge(other)

    def intersect(self, other: "Sketch") -> int:
        """Estimate distinct k-mers in ``self ∩ other``.

        Uses the standard HLL inclusion-exclusion identity:

            |A ∩ B| ≈ max(0, |A| + |B| - |A ∪ B|)

        clamped at zero (the difference of two HLL estimates can go
        slightly negative when the true intersection is near zero).

        Caveat — HLL intersection is *much* less accurate than HLL
        union or single-set cardinality. Both terms on the right are
        themselves HLL estimates with ~1.04/√(2**precision) relative
        error, so the subtraction inflates the absolute error. When
        |A ∩ B| is much smaller than |A ∪ B| the relative error on the
        intersection can be enormous (catastrophic cancellation). For
        precision 14 (default) the rough rule of thumb is: trust the
        intersection only when it's >~5% of the union. Use a real exact
        counter (KMC, jellyfish) when you need precise small overlaps.

        Raises ``ValueError`` if ``k``, ``precision``, or ``canonical``
        differ between the two sketches.
        """
        if not isinstance(other, Sketch):
            raise TypeError(
                f"intersect expects a cuhll.Sketch, got {type(other).__name__}")
        if other._k != self._k:
            raise ValueError(
                f"k mismatch on intersect: "
                f"self.k={self._k}, other.k={other._k}")
        if other.precision != self.precision:
            raise ValueError(
                f"precision mismatch on intersect: "
                f"self.precision={self.precision}, "
                f"other.precision={other.precision}")
        if other.canonical != self.canonical:
            raise ValueError(
                f"canonical-mode mismatch on intersect: "
                f"self.canonical={self.canonical}, "
                f"other.canonical={other.canonical}")
        union_est = (self | other).estimate()
        return max(0, self.estimate() + other.estimate() - union_est)

    def __and__(self, other: "Sketch") -> int:
        """Set-intersection cardinality: ``a & b`` → int.

        Mirrors :meth:`intersect`. Returns the estimated cardinality
        of ``self ∩ other`` (an int), NOT a new Sketch — HLL doesn't
        support representing a true intersection sketch, only its
        cardinality via inclusion-exclusion.
        """
        return self.intersect(other)

    def __len__(self) -> int:
        """Number of HLL registers (``2 ** precision``)."""
        return 1 << self.precision

    def __repr__(self) -> str:
        return (
            f"<cuhll.Sketch k={self._k} precision={self.precision} "
            f"canonical={self.canonical} estimate={self.estimate()}>"
        )


# ---------------------------------------------------------------------------
# Top-level functions: the API surface most users will touch.
# ---------------------------------------------------------------------------

def estimate(
    fasta: PathLike,
    *,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    chunk_mb: int = kDefaultChunkMB,
    verbose: bool = False,
) -> int:
    """Distinct k-mer cardinality estimate for one FASTA — one-liner.

    Equivalent to ``sketch(fasta, k=k, ...).estimate()`` but doesn't keep
    the Sketch object around. ``verbose=True`` opts into the C++/CUDA
    pipeline's diagnostic banner (kernel occupancy, etc.); off by default.
    """
    return sketch(fasta, k=k, precision=precision, canonical=canonical,
                  chunk_mb=chunk_mb, verbose=verbose).estimate()


def sketch(
    fasta: PathLike,
    *,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    chunk_mb: int = kDefaultChunkMB,
    verbose: bool = False,
) -> Sketch:
    """Build an in-memory Sketch from a single FASTA file.

    ``verbose=True`` opts into the C++/CUDA pipeline's diagnostic
    banner (kernel-occupancy line); off by default for clean output.
    """
    with _silence_native_stdout(verbose):
        impl = _core.sketch_one_fasta(
            os.fspath(fasta), int(k), int(precision),
            bool(canonical), int(chunk_mb),
        )
    return Sketch(precision=precision, canonical=canonical, k=k, _impl=impl)


def sketch_many(
    fastas: Iterable[PathLike],
    *,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    chunk_mb: int = kDefaultChunkMB,
    verbose: bool = False,
) -> List[Sketch]:
    """Build one Sketch per input FASTA, returned as a list in input order.

    Uses cuHLL's **concurrent per-genome pipeline** (3+ CUDA streams +
    auto-tuned reader/writer pools, the same path :func:`sketch_to_dir`
    uses) so multiple genomes are processed in parallel on a single
    GPU. Sketches are written transiently to a system tempdir and
    immediately read back as Python Sketch objects, so the caller sees
    only in-memory results — the disk roundtrip is implementation
    detail, not API.

    The temp location respects ``$TMPDIR`` (Slurm: per-job node-local
    SSD; otherwise system ``/tmp``), so the I/O cost is microseconds
    per sketch and the parallelism benefit dominates.

    Result is bit-exact equivalent to a sequential
    ``[sketch(f, …) for f in fastas]`` loop — same `.hll` file format
    on disk, same registers in memory.

    Falls back to a sequential loop only if input filenames have
    colliding stems (e.g. ``dir1/genome.fa`` and ``dir2/genome.fa``
    would clash in the tempdir) or if there's a single input.
    """
    paths = [Path(os.fspath(f)) for f in fastas]
    if not paths:
        return []
    if len(paths) == 1:
        return [sketch(paths[0], k=k, precision=precision,
                       canonical=canonical, chunk_mb=chunk_mb,
                       verbose=verbose)]

    # If two inputs have the same filename stem they would write to the
    # same `<stem>.hll` in the tempdir and overwrite each other — fall
    # back to the safe sequential path.
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        return [sketch(p, k=k, precision=precision,
                       canonical=canonical, chunk_mb=chunk_mb,
                       verbose=verbose)
                for p in paths]

    with tempfile.TemporaryDirectory(prefix="cuhll_sketch_many_") as td:
        td_path = Path(td)
        with _silence_native_stdout(verbose):
            _core.sketch_per_genome_auto(
                [str(p) for p in paths], str(td_path),
                int(k), int(precision), bool(canonical),
            )
        return [read(td_path / f"{p.stem}.hll") for p in paths]


def sketch_union(
    fastas: Iterable[PathLike],
    *,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    chunk_mb: int = kDefaultChunkMB,
    verbose: bool = False,
) -> Sketch:
    """Build ONE merged Sketch over many FASTAs (treat the panel as one genome).

    Routes through the **concurrent per-genome pipeline** (3+ CUDA
    streams + auto-tuned reader/writer pools) and merges the resulting
    per-genome sketches in Python — the merge step is a register-wise
    max over 2**precision uint32s per sketch, microseconds even for
    thousands of inputs.

    For 2000 chr19 genomes this path runs ~5× faster than the original
    single-streaming-pass implementation (the GPU is starved by NFS
    reads in the sequential path; the concurrent pipeline saturates
    storage with parallel readers).

    ``chunk_mb`` is accepted for back-compat and ignored — the
    concurrent pipeline auto-tunes its own chunk size from
    GPU/CPU/RAM probes.

    Use this when you want the cardinality of the *combined* k-mer set
    across a panel of genomes (diversity / coverage estimation) and
    don't need per-genome breakdowns or persistent per-genome `.hll`
    files. If you DO want per-genome files, call :func:`sketch_to_dir`
    instead — it returns the same union estimate as a side effect.
    """
    del chunk_mb  # accepted for back-compat; the concurrent path auto-tunes.
    paths = [Path(os.fspath(f)) for f in fastas]
    if not paths:
        raise ValueError("sketch_union: empty input list")
    if len(paths) == 1:
        return sketch(paths[0], k=k, precision=precision,
                      canonical=canonical, verbose=verbose)

    # The concurrent path produces one Sketch per input; we then merge
    # them in Python. The merge is essentially free (register-wise max
    # of 2**precision uint32s per sketch — microseconds).
    sketches = sketch_many(paths, k=k, precision=precision,
                           canonical=canonical, verbose=verbose)
    out = sketches[0]
    for s in sketches[1:]:
        out.merge(s)
    return out


def estimate_union(
    fastas: Iterable[PathLike],
    *,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    chunk_mb: int = kDefaultChunkMB,
    verbose: bool = False,
) -> int:
    """Distinct k-mer cardinality across many FASTAs treated as one genome.

    Routes through the **concurrent per-genome pipeline** with a
    transient tempdir for the per-genome sketches — the C++ side
    already computes and returns the union cardinality from
    `sketch_per_genome_auto`, so this wrapper just captures it. ~5×
    faster than a sequential single-pass on large panels.

    For 2000 × 57 MB chr19 genomes on an L4 this completes in ~45s
    (vs ~225s for the old sequential implementation).

    ``chunk_mb`` is accepted for back-compat and ignored.
    """
    del chunk_mb
    paths = [Path(os.fspath(f)) for f in fastas]
    if not paths:
        raise ValueError("estimate_union: empty input list")
    if len(paths) == 1:
        return estimate(paths[0], k=k, precision=precision,
                        canonical=canonical, verbose=verbose)

    # Stem-collision fallback: if two inputs share a basename stem the
    # concurrent path would overwrite .hll files in the tempdir, so we
    # fall back to per-input sketching + Python merge (still concurrent
    # via sketch_many, just without the C++-side union shortcut).
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        return sketch_union(paths, k=k, precision=precision,
                            canonical=canonical, verbose=verbose).estimate()

    with tempfile.TemporaryDirectory(prefix="cuhll_estimate_union_") as td:
        with _silence_native_stdout(verbose):
            return int(_core.sketch_per_genome_auto(
                [str(p) for p in paths], td,
                int(k), int(precision), bool(canonical),
            ))


class SketchDirResult(Dict[str, str]):
    """Return value of :func:`sketch_to_dir`.

    Behaves exactly like a ``Dict[input_path, output_hll_path]`` —
    ``len()``, indexing, iteration, ``.items()``, ``.values()`` all
    work as before. Carries one extra attribute:

    Attributes:
        union_estimate (int): cardinality of the union of all input
            k-mer sets, computed by the concurrent C++ pipeline as a
            free side effect (no extra GPU/disk work). Useful for
            "panel diversity" workflows: get per-genome .hll files AND
            the merged union cardinality from a single call.

    Subclassing ``dict`` (not ``NamedTuple``) keeps every existing
    caller working — ``len(result)`` still returns the number of
    output files, not the number of return-value fields.
    """

    __slots__ = ("union_estimate",)

    def __init__(self, mapping: Dict[str, str], union_estimate: int) -> None:
        super().__init__(mapping)
        self.union_estimate = int(union_estimate)


def sketch_to_dir(
    fastas: Iterable[PathLike],
    *,
    output_dir: PathLike,
    k: int,
    precision: int = kDefaultPrecision,
    canonical: bool = True,
    verbose: bool = False,
) -> SketchDirResult:
    """Sketch many FASTAs and write each as ``<stem>.hll`` into ``output_dir``.

    Uses cuHLL's concurrent per-genome pipeline (pinned-host ring + 3
    streams + auto-tuned reader/writer pools). The output directory is
    created if it doesn't exist.

    Returns a :class:`SketchDirResult` — a dict-subclass mapping
    ``input_path -> output_hll_path`` (same as before), with an extra
    ``.union_estimate`` attribute carrying the union cardinality the
    pipeline computes for free. Existing dict-style access (``len``,
    indexing, iteration) is unchanged.

    No Python-side Sketch objects are constructed — sketches stream
    straight to disk, so memory stays flat regardless of input count.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(f) for f in fastas]
    paths_str = [str(f) for f in inputs]
    with _silence_native_stdout(verbose):
        union_est = _core.sketch_per_genome_auto(
            paths_str, str(out_dir), int(k), int(precision), bool(canonical)
        )
    mapping = {str(f): str(out_dir / f"{f.stem}.hll") for f in inputs}
    return SketchDirResult(mapping, int(union_est))


def intersect_estimate(a: Sketch, b: Sketch) -> int:
    """Estimate ``|a ∩ b|`` via HLL inclusion-exclusion (2-way).

    Identical to ``a.intersect(b)`` / ``a & b`` — provided as a top-level
    function for symmetry with :func:`estimate_union`. See
    :meth:`Sketch.intersect` for the math, error caveats, and validation
    rules.

    Returns 0 if the inclusion-exclusion subtraction goes negative
    (which happens when the true intersection is near zero and the
    individual HLL estimates' errors line up unfavorably).
    """
    return a.intersect(b)


def intersect_estimate_many(sketches: Sequence[Sketch]) -> int:
    """Estimate ``|⋂ sketches|`` via n-way inclusion-exclusion.

    For ``n`` sketches the identity is::

        |∩A_i| = Σ_{S⊆{1..n}, S≠∅} (-1)^(|S|+1) |∪_{i∈S} A_i|

    so the function evaluates ``2**n - 1`` HLL union estimates. Each
    union estimate is just a clone + register-wise max + count
    (microseconds for precision 14), so wall time is negligible — the
    real concern is *error*. Each term carries the standard HLL relative
    error (~1.04/√(2**precision)) and the inclusion-exclusion alternates
    signs, so the absolute error compounds rapidly.

    Practical guidance:
      * ``n == 2`` is the only case with reliable accuracy at default
        precision; even then, intersections that are <~5% of the union
        suffer catastrophic cancellation.
      * ``n == 3-4`` may be informative for "is this near zero or not"
        but treat the magnitude as approximate.
      * ``n > 6`` is almost always nonsense at p=14; emits a warning.
        Bump precision to 16+ if you genuinely need this.

    For exact small-set intersection, use a real k-mer counter (KMC).

    Raises ``ValueError`` for empty input, mismatched k/precision/canonical,
    or non-Sketch elements.
    """
    if not sketches:
        raise ValueError("intersect_estimate_many: empty sketches sequence")
    sketches = list(sketches)
    for i, s in enumerate(sketches):
        if not isinstance(s, Sketch):
            raise TypeError(
                f"intersect_estimate_many: element {i} is "
                f"{type(s).__name__}, expected cuhll.Sketch")
    s0 = sketches[0]
    for i, s in enumerate(sketches[1:], start=1):
        if s._k != s0._k:
            raise ValueError(
                f"k mismatch at sketch {i}: "
                f"sketch[0].k={s0._k}, sketch[{i}].k={s._k}")
        if s.precision != s0.precision:
            raise ValueError(
                f"precision mismatch at sketch {i}: "
                f"sketch[0].precision={s0.precision}, "
                f"sketch[{i}].precision={s.precision}")
        if s.canonical != s0.canonical:
            raise ValueError(
                f"canonical-mode mismatch at sketch {i}: "
                f"sketch[0].canonical={s0.canonical}, "
                f"sketch[{i}].canonical={s.canonical}")

    n = len(sketches)
    if n == 1:
        return s0.estimate()
    if n == 2:
        return sketches[0].intersect(sketches[1])
    if n > 6:
        warnings.warn(
            f"intersect_estimate_many: n={n} requires {(1 << n) - 1} HLL "
            f"union estimates and the inclusion-exclusion error compounds "
            f"exponentially. At precision={s0.precision} this estimate is "
            f"likely unreliable. Use precision >= 16, or a real k-mer "
            f"counter (KMC) for exact small-set intersections.",
            stacklevel=2)

    # Sum (-1)^(|S|+1) * |∪_{i∈S} sketches[i]| over every non-empty
    # subset S ⊆ {0..n-1}, encoded as a bitmask 1..2**n - 1.
    total = 0
    for mask in range(1, 1 << n):
        # Build the union sketch for this subset.
        first = True
        u: Sketch
        size = 0
        for i in range(n):
            if mask & (1 << i):
                size += 1
                if first:
                    u = sketches[i].clone()
                    first = False
                else:
                    u.merge(sketches[i])
        sign = 1 if (size % 2 == 1) else -1
        total += sign * u.estimate()
    return max(0, total)


def read(path: PathLike) -> Sketch:
    """Load a Sketch from a ``.hll`` file. ``k`` is taken from the header."""
    header = _core.read_hll_header(os.fspath(path))
    impl   = _core.read_hll(os.fspath(path))
    return Sketch(
        precision=int(header.precision_p),
        canonical=bool(header.canonical),
        k=int(header.k),
        _impl=impl,
    )


def read_header(path: PathLike) -> HllFileHeader:
    """Read just the 48-byte header of a ``.hll`` file (no register data)."""
    return _core.read_hll_header(os.fspath(path))


__all__ = [
    # Classes
    "Sketch",
    "HllFileHeader",
    "SketchDirResult",
    # Pipeline / construction
    "estimate",
    "sketch",
    "sketch_many",
    "sketch_to_dir",
    "sketch_union",
    "estimate_union",
    # Set-arithmetic estimators
    "intersect_estimate",
    "intersect_estimate_many",
    # File I/O
    "read",
    "read_header",
    # Constants
    "kMinK",
    "kMaxK",
    "kMinPrecision",
    "kMaxPrecision",
    "kDefaultPrecision",
    "kDefaultChunkMB",
    "kHllFileVersion",
    # Version
    "__version__",
]

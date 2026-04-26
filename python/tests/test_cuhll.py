"""Pytest suite for the cuhll Python bindings.

Mirrors the C++ Layer-1 hermetic tests:
  * Synthetic ground-truth cardinality estimation (within 4 sigma of HLL's
    expected error).
  * `.hll` round-trip — write then read recovers the same estimate and
    metadata.
  * Merge semantics — union of two synthetic sketches matches the size of
    the union of their ground-truth sets.
  * Operator overloads (`|`, `|=`) and Python-side validation
    (k-mismatch, precision-mismatch, canonical-mismatch raise ValueError).

GPU-bound tests carry the `gpu` marker and are skipped on hosts without
a visible CUDA device (see conftest.py).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

import cuhll


# ---------------------------------------------------------------------------
# Synthetic-sequence helpers (parallel to tests/test_synthetic.cu).
# ---------------------------------------------------------------------------

_BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(kmer: str) -> str:
    return kmer.translate(_COMPLEMENT)[::-1]


def _canonical(kmer: str) -> str:
    rc = _revcomp(kmer)
    return kmer if kmer < rc else rc


def _synth_sequence(n_distinct: int, k: int, seed: int) -> str:
    """Build a sequence whose k-mers (canonical) are exactly `n_distinct`
    independent random k-mers, separated by k 'N' characters so the parser
    sees no overlap-induced spurious k-mer.
    """
    rng = random.Random(seed)
    sep = "N" * k
    parts: list[str] = [sep]
    seen: set[str] = set()
    while len(seen) < n_distinct:
        kmer = "".join(rng.choices(_BASES, k=k))
        canon = _canonical(kmer)
        if canon in seen:
            continue
        seen.add(canon)
        parts.append(kmer)
        parts.append(sep)
    return "".join(parts)


def _write_fasta(path: Path, seq: str) -> None:
    """Write `seq` as a single-record FASTA wrapped at 80 cols."""
    with path.open("w") as f:
        f.write(">test\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


def _hll_4sigma(precision: int) -> float:
    """4 standard errors of HLL at the given precision."""
    m = 1 << precision
    return 4.0 * 1.04 / math.sqrt(m)


# ---------------------------------------------------------------------------
# Hermetic tests (don't need a GPU).
# ---------------------------------------------------------------------------

def test_constants_present() -> None:
    """Module-level constants match the C++ side."""
    assert cuhll.kMinK == 8
    assert cuhll.kMaxK == 32
    assert cuhll.kMinPrecision == 8
    assert cuhll.kMaxPrecision == 18
    assert cuhll.kDefaultPrecision == 14
    assert cuhll.kHllFileVersion == 2


def test_module_surface() -> None:
    """All public names enumerated in __all__ resolve."""
    for name in cuhll.__all__:
        assert hasattr(cuhll, name), f"cuhll.{name} missing"


# ---------------------------------------------------------------------------
# GPU-bound tests.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "n_distinct,k,precision",
    [
        (100_000, 21, 12),
        (200_000, 31, 14),
        ( 50_000, 15, 14),
    ],
)
def test_synthetic_within_4sigma(
    tmp_path: Path, n_distinct: int, k: int, precision: int
) -> None:
    """HLL estimate of `n_distinct` known k-mers is within 4 sigma of truth."""
    seed = (n_distinct ^ (k << 8) ^ (precision << 16)) & 0xFFFFFFFF
    seq = _synth_sequence(n_distinct, k, seed)
    fasta = tmp_path / "synth.fa"
    _write_fasta(fasta, seq)

    s = cuhll.sketch(fasta, k=k, precision=precision)
    est = s.estimate()

    rel_err = abs(est - n_distinct) / n_distinct
    tol = _hll_4sigma(precision)
    assert rel_err <= tol, (
        f"HLL estimate {est} for truth {n_distinct} "
        f"has relative error {rel_err:.4%} > tolerance {tol:.4%} (4 sigma)"
    )

    # Sketch metadata should match construction params.
    assert s.k == k
    assert s.precision == precision
    assert s.canonical is True
    assert len(s) == 1 << precision


@pytest.mark.gpu
def test_estimate_one_liner(tmp_path: Path) -> None:
    """`cuhll.estimate(...)` returns the same number as `sketch().estimate()`."""
    fasta = tmp_path / "synth.fa"
    _write_fasta(fasta, _synth_sequence(50_000, 31, seed=42))
    direct = cuhll.estimate(fasta, k=31)
    via_sketch = cuhll.sketch(fasta, k=31).estimate()
    assert direct == via_sketch


@pytest.mark.gpu
def test_hll_roundtrip(tmp_path: Path) -> None:
    """Write a sketch, read it back, recover identical estimate and metadata."""
    fasta = tmp_path / "synth.fa"
    _write_fasta(fasta, _synth_sequence(75_000, 31, seed=7))

    s = cuhll.sketch(fasta, k=31, precision=14)
    out = tmp_path / "synth.hll"
    s.write(out)

    header = cuhll.read_header(out)
    assert header.version == cuhll.kHllFileVersion
    assert header.precision_p == 14
    assert header.k == 31
    assert header.canonical == 1

    s2 = cuhll.read(out)
    assert s2.estimate() == s.estimate()
    assert s2.k == 31
    assert s2.precision == 14
    assert s2.canonical is True


@pytest.mark.gpu
def test_merge_set_union_matches_truth(tmp_path: Path) -> None:
    """Merging sketches of disjoint sets gives a sketch whose estimate
    is within 4 sigma of the true total cardinality."""
    k = 31
    p = 14

    seq_a = _synth_sequence(40_000, k, seed=1)
    seq_b = _synth_sequence(60_000, k, seed=2)
    fa = tmp_path / "a.fa"
    fb = tmp_path / "b.fa"
    _write_fasta(fa, seq_a)
    _write_fasta(fb, seq_b)

    a = cuhll.sketch(fa, k=k, precision=p)
    b = cuhll.sketch(fb, k=k, precision=p)

    union = a | b                            # operator
    a_clone = a.clone()
    a_clone.merge(b)                          # method

    # Both routes should give the same answer.
    assert union.estimate() == a_clone.estimate()

    # The two synthetic sets are independent random k-mers; collisions
    # negligible, so true total ≈ 100_000.
    truth = 40_000 + 60_000
    rel_err = abs(union.estimate() - truth) / truth
    assert rel_err <= _hll_4sigma(p)

    # In-place |= alias.
    a |= b
    assert a.estimate() == union.estimate()


@pytest.mark.gpu
def test_merge_validation_rejects_mismatch(tmp_path: Path) -> None:
    """Merging sketches with mismatched k / precision / canonical raises."""
    fa = tmp_path / "a.fa"
    fb = tmp_path / "b.fa"
    _write_fasta(fa, _synth_sequence(20_000, 21, seed=3))
    _write_fasta(fb, _synth_sequence(20_000, 31, seed=4))

    a21 = cuhll.sketch(fa, k=21, precision=14)
    b31 = cuhll.sketch(fb, k=31, precision=14)
    with pytest.raises(ValueError, match="k mismatch"):
        a21.merge(b31)

    a14 = cuhll.sketch(fa, k=21, precision=14)
    a12 = cuhll.sketch(fa, k=21, precision=12)
    with pytest.raises(ValueError, match="precision mismatch"):
        a14.merge(a12)

    can = cuhll.sketch(fa, k=21, precision=14, canonical=True)
    noc = cuhll.sketch(fa, k=21, precision=14, canonical=False)
    with pytest.raises(ValueError, match="canonical-mode mismatch"):
        can.merge(noc)


@pytest.mark.gpu
def test_sketch_many_parallel_matches_sequential(tmp_path: Path) -> None:
    """`sketch_many` (now parallel via the concurrent per-genome pipeline)
    must produce **bit-exact** Sketches relative to the sequential
    one-at-a-time path. Sanity check that the parallel implementation
    is a faithful drop-in for the sequential reference, and that every
    in-memory Sketch carries its expected metadata so downstream code
    can use it without surprises."""
    k = 31
    p = 14
    fa = tmp_path / "alpha.fa"
    fb = tmp_path / "beta.fa"
    fc = tmp_path / "gamma.fa"
    _write_fasta(fa, _synth_sequence(20_000, k, seed=101))
    _write_fasta(fb, _synth_sequence(30_000, k, seed=202))
    _write_fasta(fc, _synth_sequence(25_000, k, seed=303))
    paths = [fa, fb, fc]

    # Reference — sequential per-genome sketches.
    sequential = [cuhll.sketch(p_, k=k, precision=p) for p_ in paths]

    # Subject — parallel sketch_many (uses concurrent pipeline under the hood).
    parallel = cuhll.sketch_many(paths, k=k, precision=p)

    assert len(parallel) == len(sequential) == len(paths)
    for path_, seq_s, par_s in zip(paths, sequential, parallel):
        # Bit-exact estimate match — the parallel pipeline writes the same
        # registers as the sequential sketcher (same kernel, same hash, same
        # canonical mode).
        assert par_s.estimate() == seq_s.estimate(), (
            f"parallel/sequential drift on {path_.name}: "
            f"parallel={par_s.estimate()}, sequential={seq_s.estimate()}"
        )
        assert par_s.k == seq_s.k == k
        assert par_s.precision == seq_s.precision == p
        assert par_s.canonical == seq_s.canonical is True
        assert len(par_s) == len(seq_s) == (1 << p)

    # In-memory access pattern — caller can zip with input paths to map
    # names → sketches, just like the README documents.
    panel = dict(zip(paths, parallel))
    assert panel[fa].estimate() == parallel[0].estimate()
    assert panel[fb].estimate() == parallel[1].estimate()
    assert panel[fc].estimate() == parallel[2].estimate()


@pytest.mark.gpu
def test_sketch_union_matches_manual_merge(tmp_path: Path) -> None:
    """`sketch_union(paths)` produces the same Sketch as `sketch_many(paths)`
    followed by a manual merge — the convenience function is a faithful
    one-pass equivalent of the per-genome-then-merge pattern."""
    k = 31
    p = 14
    fa = tmp_path / "a.fa"
    fb = tmp_path / "b.fa"
    fc = tmp_path / "c.fa"
    _write_fasta(fa, _synth_sequence(20_000, k, seed=11))
    _write_fasta(fb, _synth_sequence(30_000, k, seed=22))
    _write_fasta(fc, _synth_sequence(25_000, k, seed=33))
    paths = [fa, fb, fc]

    # Path A — sketch each, merge in Python.
    sketches = cuhll.sketch_many(paths, k=k, precision=p)
    manual = sketches[0].clone()
    for s in sketches[1:]:
        manual.merge(s)

    # Path B — sketch_union in one streaming pass.
    union = cuhll.sketch_union(paths, k=k, precision=p)

    assert union.estimate() == manual.estimate(), (
        f"sketch_union estimate {union.estimate()} != manual merge {manual.estimate()}"
    )
    assert union.k == k and union.precision == p and union.canonical is True

    # estimate_union is the no-Sketch shortcut.
    bare = cuhll.estimate_union(paths, k=k, precision=p)
    assert bare == union.estimate()

    # And ground-truth: independent random k-mer sets, so the union of the
    # truth values is the sum (collisions negligible for these N).
    truth = 20_000 + 30_000 + 25_000
    rel_err = abs(union.estimate() - truth) / truth
    assert rel_err <= _hll_4sigma(p), (
        f"union estimate {union.estimate()} drifted {rel_err:.4%} from truth {truth}"
    )


@pytest.mark.gpu
def test_sketch_to_dir_writes_one_per_input(tmp_path: Path) -> None:
    """`sketch_to_dir` writes one .hll per input and returns the path map."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    fastas = []
    for i, n in enumerate([15_000, 25_000]):
        p = in_dir / f"genome_{i}.fa"
        _write_fasta(p, _synth_sequence(n, 31, seed=100 + i))
        fastas.append(p)

    paths = cuhll.sketch_to_dir(fastas, output_dir=out_dir, k=31)

    assert set(paths.keys()) == {str(p) for p in fastas}
    for src, dst in paths.items():
        assert Path(dst).is_file()
        assert Path(dst).suffix == ".hll"
        # The header should be loadable for every written file.
        h = cuhll.read_header(dst)
        assert h.k == 31

    # The concurrent C++ pipeline computes the union cardinality for
    # free; sketch_to_dir now exposes it via .union_estimate (was
    # previously discarded).
    assert hasattr(paths, "union_estimate")
    truth_union = 15_000 + 25_000   # disjoint random sets, collisions negligible
    rel_err = abs(paths.union_estimate - truth_union) / truth_union
    assert rel_err <= _hll_4sigma(14), (
        f"sketch_to_dir union_estimate {paths.union_estimate} "
        f"drifted {rel_err:.4%} from truth {truth_union}"
    )


# ---------------------------------------------------------------------------
# Intersection — 2-way and n-way, against synthetic ground truth.
# ---------------------------------------------------------------------------

def _build_overlapping_fastas(
    tmp_path: Path,
    k: int,
    *,
    only_a: int,
    only_b: int,
    shared: int,
    seed: int,
) -> tuple[Path, Path, int, int, int]:
    """Build two FASTAs whose canonical k-mer sets overlap by exactly `shared`.

    Returns (path_a, path_b, |A|, |B|, |A ∩ B|) where:
      |A|       == only_a + shared
      |B|       == only_b + shared
      |A ∩ B|   == shared

    All k-mers are drawn from independent random rolls and de-duplicated
    by canonical form, so collisions across the three buckets are zero
    by construction (the rejection sampling guarantees disjointness).
    """
    rng = random.Random(seed)
    pool: set[str] = set()

    def _draw(n: int) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            kmer = "".join(rng.choices(_BASES, k=k))
            canon = _canonical(kmer)
            if canon in pool:
                continue
            pool.add(canon)
            out.append(kmer)
        return out

    only_a_kmers = _draw(only_a)
    only_b_kmers = _draw(only_b)
    shared_kmers = _draw(shared)

    sep = "N" * k

    def _emit(kmers: list[str]) -> str:
        # Same separator pattern as _synth_sequence so the parser sees
        # exactly the intended k-mers, no overlap-induced extras.
        return sep + sep.join(kmers) + sep

    fa = tmp_path / "intersect_a.fa"
    fb = tmp_path / "intersect_b.fa"
    _write_fasta(fa, _emit(only_a_kmers + shared_kmers))
    _write_fasta(fb, _emit(only_b_kmers + shared_kmers))
    return fa, fb, only_a + shared, only_b + shared, shared


def _hll_intersect_tolerance(precision: int, union_size: int, intersect_size: int) -> float:
    """Rough 4-sigma tolerance for an HLL inclusion-exclusion intersection.

    The estimator is |A| + |B| - |A ∪ B|, three correlated HLL estimates
    whose variances roughly add. So absolute error scales with the union
    size, not the intersection size. Relative-to-truth tolerance is
    therefore amplified when |A ∩ B| << |A ∪ B|.
    """
    rse = 1.04 / math.sqrt(1 << precision)              # per-estimate RSE
    abs_err = 4.0 * math.sqrt(3) * rse * union_size     # 4-sigma, sqrt(3) for 3 terms
    return abs_err / max(intersect_size, 1)


@pytest.mark.gpu
def test_intersect_2way_recovers_known_overlap(tmp_path: Path) -> None:
    """Build A and B with a known overlap; HLL intersection recovers it."""
    k = 31
    p = 16  # bumped from default 14 — intersection has higher variance

    fa, fb, true_a, true_b, true_intersect = _build_overlapping_fastas(
        tmp_path, k, only_a=20_000, only_b=20_000, shared=30_000, seed=4242,
    )
    a = cuhll.sketch(fa, k=k, precision=p)
    b = cuhll.sketch(fb, k=k, precision=p)

    # Sanity: individual estimates within HLL's normal 4-sigma.
    assert abs(a.estimate() - true_a) / true_a <= _hll_4sigma(p)
    assert abs(b.estimate() - true_b) / true_b <= _hll_4sigma(p)

    # Intersection via the three equivalent surfaces.
    inter_method = a.intersect(b)
    inter_op     = a & b
    inter_topfn  = cuhll.intersect_estimate(a, b)
    assert inter_method == inter_op == inter_topfn

    true_union = true_a + true_b - true_intersect
    tol = _hll_intersect_tolerance(p, true_union, true_intersect)
    rel_err = abs(inter_method - true_intersect) / true_intersect
    assert rel_err <= tol, (
        f"|A ∩ B| estimate {inter_method:,} for truth {true_intersect:,} "
        f"has relative error {rel_err:.4%} > tolerance {tol:.4%} "
        f"(union={true_union:,}, p={p})"
    )


@pytest.mark.gpu
def test_intersect_disjoint_clamps_to_zero(tmp_path: Path) -> None:
    """When true intersection is exactly zero the estimator should clamp at 0
    (it can't go negative even if HLL noise wants it to)."""
    k = 31
    p = 16

    fa, fb, _ta, _tb, _ti = _build_overlapping_fastas(
        tmp_path, k, only_a=20_000, only_b=20_000, shared=0, seed=1729,
    )
    a = cuhll.sketch(fa, k=k, precision=p)
    b = cuhll.sketch(fb, k=k, precision=p)
    inter = a.intersect(b)

    assert inter >= 0   # the clamp guarantee
    # With ~40K union and zero shared, the estimate should be much smaller
    # than the union — call it < 5% of |A ∪ B| at p=16 (roughly 4-sigma).
    union_est = (a | b).estimate()
    assert inter < 0.05 * union_est, (
        f"intersect of disjoint sets returned {inter:,}, expected near 0 "
        f"(union estimate {union_est:,})"
    )


@pytest.mark.gpu
def test_intersect_validation_rejects_mismatch(tmp_path: Path) -> None:
    """Mismatched k/precision/canonical raises, just like merge."""
    fa = tmp_path / "a.fa"
    fb = tmp_path / "b.fa"
    _write_fasta(fa, _synth_sequence(10_000, 21, seed=11))
    _write_fasta(fb, _synth_sequence(10_000, 31, seed=22))

    a21 = cuhll.sketch(fa, k=21, precision=14)
    b31 = cuhll.sketch(fb, k=31, precision=14)
    with pytest.raises(ValueError, match="k mismatch"):
        a21.intersect(b31)

    a14 = cuhll.sketch(fa, k=21, precision=14)
    a12 = cuhll.sketch(fa, k=21, precision=12)
    with pytest.raises(ValueError, match="precision mismatch"):
        a14.intersect(a12)

    can = cuhll.sketch(fa, k=21, precision=14, canonical=True)
    noc = cuhll.sketch(fa, k=21, precision=14, canonical=False)
    with pytest.raises(ValueError, match="canonical-mode mismatch"):
        can.intersect(noc)

    with pytest.raises(TypeError, match="cuhll.Sketch"):
        a14.intersect("not a sketch")  # type: ignore[arg-type]


@pytest.mark.gpu
def test_intersect_many_3way_recovers_known_overlap(tmp_path: Path) -> None:
    """Three-way inclusion-exclusion against a synthetic ground truth.

    Construct three k-mer sets with controlled triple overlap. Each pair
    shares only the triple, no extra pairwise overlap, so the math is
    clean.
    """
    k = 31
    p = 16
    rng = random.Random(20251101)
    pool: set[str] = set()

    def _draw(n: int) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            kmer = "".join(rng.choices(_BASES, k=k))
            canon = _canonical(kmer)
            if canon in pool:
                continue
            pool.add(canon)
            out.append(kmer)
        return out

    only_a = _draw(15_000)
    only_b = _draw(15_000)
    only_c = _draw(15_000)
    triple = _draw(20_000)

    sep = "N" * k

    def _emit(kmers: list[str]) -> str:
        return sep + sep.join(kmers) + sep

    fa = tmp_path / "a.fa"; _write_fasta(fa, _emit(only_a + triple))
    fb = tmp_path / "b.fa"; _write_fasta(fb, _emit(only_b + triple))
    fc = tmp_path / "c.fa"; _write_fasta(fc, _emit(only_c + triple))

    a = cuhll.sketch(fa, k=k, precision=p)
    b = cuhll.sketch(fb, k=k, precision=p)
    c = cuhll.sketch(fc, k=k, precision=p)

    inter = cuhll.intersect_estimate_many([a, b, c])

    true_intersect = len(triple)        # 20_000
    true_union = 3 * 15_000 + 20_000    # 65_000
    tol = _hll_intersect_tolerance(p, true_union, true_intersect)
    rel_err = abs(inter - true_intersect) / true_intersect
    assert rel_err <= tol, (
        f"|A ∩ B ∩ C| estimate {inter:,} for truth {true_intersect:,} "
        f"has relative error {rel_err:.4%} > tolerance {tol:.4%}"
    )

    # Single-element call returns just the cardinality estimate.
    assert cuhll.intersect_estimate_many([a]) == a.estimate()

    # Two-element call delegates to the 2-way path.
    assert cuhll.intersect_estimate_many([a, b]) == a.intersect(b)


@pytest.mark.gpu
def test_intersect_many_validation(tmp_path: Path) -> None:
    """intersect_estimate_many enforces type and metadata invariants."""
    fa = tmp_path / "a.fa"
    _write_fasta(fa, _synth_sequence(10_000, 31, seed=99))
    a = cuhll.sketch(fa, k=31, precision=14)

    with pytest.raises(ValueError, match="empty"):
        cuhll.intersect_estimate_many([])

    with pytest.raises(TypeError, match="cuhll.Sketch"):
        cuhll.intersect_estimate_many([a, "not a sketch"])  # type: ignore[list-item]

    b_diff_k = cuhll.sketch(fa, k=31, precision=14)
    # Build a sketch object with mismatched k via direct construction
    # (sketches FROM file always carry the right k; we only mismatch via
    # explicit Sketch ctor or by re-sketching with a different k).
    fb = tmp_path / "b.fa"
    _write_fasta(fb, _synth_sequence(10_000, 21, seed=88))
    b21 = cuhll.sketch(fb, k=21, precision=14)
    with pytest.raises(ValueError, match="k mismatch"):
        cuhll.intersect_estimate_many([a, b21])

"""cuHLL end-to-end demo.

Run:
    python demo.py

What this script does:
  1.  Generates a ~200 Mbp synthetic FASTA inside the project's tmp/
      directory. The FASTA mixes:
        - random ACGT (high-entropy baseline, ~180 Mbp),
        - exact-repeat motifs (~20 Mbp total of repeats: tests that
          HLL collapses duplicate k-mers down to the distinct count),
        - a palindromic region (~1 Mbp self-reverse-complement: tests
          that canonical k-mers fold fwd/rc onto one register update),
          and
        - a short third record so the inter-record 'N' boundary is
          exercised.
  2.  Generates a gzipped FASTQ derived from the same bases — exercises
      cuhll's FASTQ + gzip auto-detection.
  3.  Runs every public cuhll function on these inputs so all of the
      pipeline paths (single-FASTA fast path, concurrent per-genome,
      concurrent shared-sketch union, .hll I/O) hit the kernel.
  4.  Re-runs the heaviest call under Nsight Systems and saves the
      timeline to <project>/tmp/profile_out/cuhll_demo.nsys-rep.

All intermediate files (the synthetic FASTA, FASTQ.gz, per-genome .hll
sketches) are created under <project>/tmp/cuhll_demo_*/ and deleted at
exit. The NSYS timeline survives — it's how you can open the run
afterward in `nsys-ui` or run `nsys stats` on it.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import cuhll


# --- knobs ------------------------------------------------------------
# Sizes chosen so the whole synthetic genome lands at ~200 Mbp — large
# enough that the GPU pipeline runs in steady state (auto-tune picks
# real stream counts, kernel SM occupancy is meaningful, H2D/kernel
# overlap is visible) rather than a few microseconds of one-shot work.
# That makes the NSYS timeline informative.
K                  = 31
BACKGROUND_BP      = 180_000_000   # random ACGT background (~180 Mbp)
REPEAT_MOTIF_LEN   = 200           # length of each exact-repeat motif
REPEAT_COUNT       = 5_000         # how many times each motif repeats
N_REPEAT_MOTIFS    = 20            # number of distinct motifs (→ 20 Mbp repeats)
PALINDROME_BP      = 1_000_000     # palindromic block (~1 Mbp)
TAIL_RECORD_BP     = 200_000       # final short record (intra-record N-break test)
SEED               = 42


def _rand_bases_bytes(n: int, rng: np.random.Generator) -> bytes:
    """Vectorized random ACGT generation. ~100× faster than
    random.choices('ACGT', k=n) for the 200 Mbp scale this demo runs at.
    """
    idx = rng.integers(0, 4, size=n, dtype=np.uint8)
    table = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)
    return table[idx].tobytes()


def _revcomp_bytes(b: bytes) -> bytes:
    table = bytes.maketrans(b"ACGTN", b"TGCAN")
    return b.translate(table)[::-1]


def make_challenging_fasta(out_path: Path) -> int:
    """Write a 3-record ~200 Mbp FASTA tailored to stress cuhll.

    Record 1: random ACGT background + N_REPEAT_MOTIFS distinct motifs,
              each repeated REPEAT_COUNT times. The repeats verify that
              HLL collapses duplicates down to a distinct count.
    Record 2: a palindromic block (sequence ++ its reverse complement).
              Every k-mer inside the palindrome has its reverse
              complement present, so canonical=True must fold fwd+rc
              onto the same register update.
    Record 3: short random fragment that exercises the boundary 'N'
              cuhll injects between FASTA records.

    Returns the total base count written.
    """
    rng = np.random.default_rng(SEED)

    bg = _rand_bases_bytes(BACKGROUND_BP, rng)
    motifs = [_rand_bases_bytes(REPEAT_MOTIF_LEN, rng) for _ in range(N_REPEAT_MOTIFS)]
    rec1 = bg + b"".join(m * REPEAT_COUNT for m in motifs)

    half = _rand_bases_bytes(PALINDROME_BP // 2, rng)
    rec2 = half + _revcomp_bytes(half)

    rec3 = _rand_bases_bytes(TAIL_RECORD_BP, rng)

    total_bp = 0
    LINE = 80
    with out_path.open("wb") as f:
        for i, body in enumerate((rec1, rec2, rec3), start=1):
            f.write(f">record_{i}_len{len(body)}\n".encode())
            for j in range(0, len(body), LINE):
                f.write(body[j : j + LINE])
                f.write(b"\n")
            total_bp += len(body)
    return total_bp


def make_fastq_gz_from_fasta(fa_path: Path, fq_gz_path: Path,
                             read_len: int = 150) -> None:
    """Chop the FASTA bases into read_len-bp reads, emit gzipped FASTQ.

    Quality is a constant Phred-40 string. Reads spanning the
    record-boundary 'N' (cuhll injects one between records) are skipped,
    so the FASTQ k-mer set is *almost* identical to the FASTA's — close
    enough that the two cardinalities should agree within HLL noise.

    Read the FASTA in chunks so we don't pull a 200 MB string into RAM.
    """
    qual_bytes = (b"I" * read_len) + b"\n"
    plus_bytes = b"+\n"
    nl = b"\n"

    # Accumulate sequence bytes only (skip headers) into one bytearray.
    seq = bytearray()
    with fa_path.open("rb") as fa:
        for line in fa:
            if line.startswith(b">"):
                continue
            seq.extend(line.rstrip(b"\n"))

    n_reads = 0
    with gzip.open(fq_gz_path, "wb", compresslevel=4) as fq:
        for start in range(0, len(seq) - read_len + 1, read_len):
            read = bytes(seq[start : start + read_len])
            if b"N" in read:
                continue
            fq.write(b"@read%d\n" % n_reads)
            fq.write(read)
            fq.write(nl)
            fq.write(plus_bytes)
            fq.write(qual_bytes)
            n_reads += 1


def run_demo(tmp: Path) -> None:
    print(f"cuhll version: {cuhll.__version__}")
    print(f"workdir (auto-cleaned): {tmp}")

    fa = tmp / "challenge.fasta"
    print(f"\n[gen] writing ~200 Mbp challenge FASTA → {fa.name}")
    t0 = time.perf_counter()
    total_bp = make_challenging_fasta(fa)
    sz_mb = fa.stat().st_size / (1024 ** 2)
    print(f"      {total_bp:,} bp, {sz_mb:.1f} MiB on disk, "
          f"generated in {time.perf_counter() - t0:.1f}s")

    fq_gz = tmp / "challenge.fastq.gz"
    print(f"\n[gen] writing FASTQ.gz of the same sequence → {fq_gz.name}")
    t0 = time.perf_counter()
    make_fastq_gz_from_fasta(fa, fq_gz)
    sz_mb = fq_gz.stat().st_size / (1024 ** 2)
    print(f"      {sz_mb:.1f} MiB on disk, "
          f"generated in {time.perf_counter() - t0:.1f}s")

    # 1. One FASTA → cardinality
    print("\n[1] cuhll.estimate(fa, k=31)")
    n = cuhll.estimate(fa, k=K)
    print(f"    {n:,} distinct canonical {K}-mers")

    # 2. Same content as FASTQ.gz (auto-detected by cuhll)
    print("\n[2] cuhll.estimate(fastq.gz, k=31)")
    n_fq = cuhll.estimate(fq_gz, k=K)
    diff_pct = (n_fq - n) / n * 100
    print(f"    {n_fq:,}    (vs FASTA: {diff_pct:+.3f}%; FASTQ skips "
          f"reads spanning the N-break, so a small diff is expected)")

    # 3. Sketch object
    print("\n[3] cuhll.sketch(fa, k=31) → Sketch")
    s = cuhll.sketch(fa, k=K)
    print(f"    {s!r}")

    # 4. Many files in parallel
    print("\n[4] cuhll.sketch_many on three copies of the FASTA")
    triplet = [fa, fa, fa]
    t0 = time.perf_counter()
    sketches = cuhll.sketch_many(triplet, k=K)
    print(f"    built {len(sketches)} sketches in "
          f"{time.perf_counter() - t0:.2f} s")
    print(f"    estimates: {[x.estimate() for x in sketches]}")

    # 5. Per-genome .hll on disk + free union
    out_dir = tmp / "sketches"
    out_dir.mkdir()
    print(f"\n[5] cuhll.sketch_to_dir → {out_dir.name}/")
    result = cuhll.sketch_to_dir(triplet, output_dir=out_dir, k=K)
    print(f"    wrote {len(result)} .hll files")
    print(f"    union estimate (returned for free): "
          f"{result.union_estimate:,}")

    # 6. In-GPU shared-sketch union (no .hll files)
    print("\n[6] cuhll.estimate_union (shared-sketch path, no disk I/O)")
    n_union = cuhll.estimate_union(triplet, k=K)
    print(f"    {n_union:,}    "
          f"(matches step 5? {n_union == result.union_estimate})")

    # 7. Set arithmetic
    print("\n[7] set arithmetic on two in-memory sketches")
    a, b = sketches[0], sketches[1]
    print(f"    |A| = {a.estimate():,}")
    print(f"    |B| = {b.estimate():,}")
    print(f"    |A ∪ B| = {(a | b).estimate():,}")
    print(f"    |A ∩ B| = {a & b:,}    (≈|A| because A and B are the same here)")

    # 8. n-way intersection
    print("\n[8] cuhll.intersect_estimate_many on 4 sketches")
    n_iws = cuhll.intersect_estimate_many(sketches + [sketches[0]])
    print(f"    {n_iws:,}")

    # 9. .hll round-trip
    print("\n[9] Sketch.write + cuhll.read round-trip")
    a.write(tmp / "a.hll")
    a2 = cuhll.read(tmp / "a.hll")
    same = a.estimate() == a2.estimate()
    print(f"    estimate after reload: {a2.estimate():,}  "
          f"(bit-identical to in-memory: {same})")


def run_nsys_profile(tmp: Path, profile_out: Path) -> int:
    """Run the heaviest cuhll call under nsys and save the timeline.

    Looks for nsys on PATH, then under /apps/compilers/cuda/13.2.1/bin/
    (HiPerGator default), then gives up.
    """
    candidates = [shutil.which("nsys"),
                  "/apps/compilers/cuda/13.2.1/bin/nsys",
                  "/apps/compilers/cuda/12.8.1/bin/nsys"]
    nsys = next((c for c in candidates if c and Path(c).exists()), None)
    if not nsys:
        print("\n[profile] nsys not found; skipping profile", file=sys.stderr)
        return 1

    profile_out.mkdir(parents=True, exist_ok=True)
    rep = profile_out / "cuhll_demo"

    # The profiled workload: estimate_union on a triplet — this is the
    # heaviest single call in the demo and exercises the full concurrent
    # pipeline.
    fa = tmp / "challenge.fasta"
    workload = (
        "import cuhll, sys; "
        f"p='{fa}'; "
        f"print(cuhll.estimate_union([p, p, p], k={K}, verbose=True))"
    )
    cmd = [
        nsys, "profile",
        "--trace=cuda,osrt", "--sample=none", "--cpuctxsw=none",
        "--output", str(rep), "--force-overwrite", "true",
        sys.executable, "-c", workload,
    ]
    print(f"\n[profile] running nsys → {rep}.nsys-rep", file=sys.stderr)
    r = subprocess.run(cmd)
    if r.returncode == 0:
        print(f"[profile] saved   : {rep}.nsys-rep", file=sys.stderr)
        print(f"[profile] view GUI: nsys-ui {rep}.nsys-rep", file=sys.stderr)
        print(f"[profile] view CLI: nsys stats {rep}.nsys-rep", file=sys.stderr)
    return r.returncode


def main() -> None:
    project_root = Path(__file__).resolve().parent
    project_tmp  = project_root / "tmp"
    project_tmp.mkdir(exist_ok=True)

    # Project tmp/ hosts both the auto-cleaned demo workdir and the
    # persistent profile timeline. Keeps everything under one directory
    # instead of spraying /tmp on the compute node (which is often small).
    with tempfile.TemporaryDirectory(prefix="cuhll_demo_",
                                     dir=str(project_tmp)) as tmp_s:
        tmp = Path(tmp_s)
        run_demo(tmp)
        run_nsys_profile(tmp, project_tmp / "profile_out")

    print()
    print(f"demo intermediates were under {project_tmp}/cuhll_demo_* "
          f"(cleaned at exit).")
    rep_path = project_tmp / "profile_out" / "cuhll_demo.nsys-rep"
    print(f"profile timeline saved to {rep_path}")
    print("open it with:")
    print(f"    nsys-ui {rep_path}")
    print("or print a text summary with:")
    print(f"    nsys stats {rep_path}")


if __name__ == "__main__":
    main()

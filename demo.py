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
# Sizes chosen so the whole synthetic genome lands at 5.131 Gbp —
# human-haploid-scale. The pipeline runs in true steady state: the
# auto-tune picks full stream counts, kernels run for seconds (not
# microseconds), and the NSYS timeline shows real H2D/kernel overlap.
#
# Memory note: the C++ reader slurps the whole FASTA into one
# std::string (~5 GB) and per-stream pinned buffers scale with input
# size, so the steps that pass `[fa, fa, fa]` to the concurrent
# pipeline can peak around 30 GB host RAM. Recommend `--mem=64G`
# (or larger) when running under srun.
K                  = 31
BACKGROUND_BP      = 4_500_000_000   # random ACGT background
REPEAT_MOTIF_LEN   = 200             # length of each exact-repeat motif
REPEAT_COUNT       = 5_000           # each motif repeats this many times (→ 1 Mbp/motif)
N_REPEAT_MOTIFS    = 600             # → 600 Mbp of repeats total
PALINDROME_BP      = 30_000_000      # palindromic block (~30 Mbp)
TAIL_RECORD_BP     = 1_000_000       # final short record (intra-record N-break test)
# Total bp = BACKGROUND_BP + N_REPEAT_MOTIFS * REPEAT_MOTIF_LEN * REPEAT_COUNT
#         + PALINDROME_BP + TAIL_RECORD_BP
#         = 4,500,000,000 + 600,000,000 + 30,000,000 + 1,000,000 = 5,131,000,000

# FASTQ.gz is generated separately at a smaller, fixed size so that
# (a) we don't spend minutes gzipping 5 Gbp, and (b) the FASTQ path is
# still exercised end-to-end. The two cardinalities therefore aren't
# expected to match — this is documented in the demo output.
FASTQ_SAMPLE_BP    = 200_000_000     # 200 Mbp of FASTQ.gz reads
FASTQ_READ_LEN     = 150
SEED               = 42


_BASE_TABLE = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)
_RC_TABLE = bytes.maketrans(b"ACGTN", b"TGCAN")
_GEN_CHUNK_BP = 100_000_000          # 100 Mbp per generation chunk — caps RAM


def _rand_bases_bytes(n: int, rng: np.random.Generator) -> bytes:
    """Vectorized random ACGT generation."""
    idx = rng.integers(0, 4, size=n, dtype=np.uint8)
    return _BASE_TABLE[idx].tobytes()


def _stream_random_bases(f, n: int, rng: np.random.Generator) -> None:
    """Write n random ACGT bases to file in fixed-size chunks. Peak
    memory is one chunk (~100 MB), regardless of n. Needed because the
    FASTA's background block is 4.5 Gbp — generating it as one numpy
    array would peak well above 4 GB of RAM."""
    remaining = n
    while remaining > 0:
        sz = min(_GEN_CHUNK_BP, remaining)
        f.write(_rand_bases_bytes(sz, rng))
        remaining -= sz


def make_challenging_fasta(out_path: Path) -> int:
    """Write a 3-record ~5.131 Gbp FASTA tailored to stress cuhll.

    Record 1: random ACGT background + N_REPEAT_MOTIFS distinct motifs,
              each repeated REPEAT_COUNT times. The repeats verify that
              HLL collapses duplicates down to a distinct count.
    Record 2: a palindromic block (sequence ++ its reverse complement).
              Every k-mer inside the palindrome has its reverse
              complement present, so canonical=True must fold fwd+rc
              onto the same register update.
    Record 3: short random fragment that exercises the boundary 'N'
              cuhll injects between FASTA records.

    Bases are written as one long line per record (no 80-char wrap).
    cuhll's reader strips any newline anyway; emitting one big write
    per record is dramatically faster than looping over millions of
    80-char slices in Python.

    Returns the total base count written.
    """
    rng = np.random.default_rng(SEED)
    total_bp = 0

    with out_path.open("wb") as f:
        # ---- record 1: random background + exact-repeat motifs ----
        f.write(b">record_1_random_plus_repeats\n")
        _stream_random_bases(f, BACKGROUND_BP, rng)
        total_bp += BACKGROUND_BP
        for _ in range(N_REPEAT_MOTIFS):
            motif = _rand_bases_bytes(REPEAT_MOTIF_LEN, rng)
            f.write(motif * REPEAT_COUNT)
            total_bp += REPEAT_MOTIF_LEN * REPEAT_COUNT
        f.write(b"\n")

        # ---- record 2: palindromic block ----
        f.write(b">record_2_palindrome\n")
        half = _rand_bases_bytes(PALINDROME_BP // 2, rng)
        f.write(half)
        f.write(half.translate(_RC_TABLE)[::-1])
        total_bp += PALINDROME_BP
        f.write(b"\n")

        # ---- record 3: tail random fragment ----
        f.write(b">record_3_tail\n")
        f.write(_rand_bases_bytes(TAIL_RECORD_BP, rng))
        total_bp += TAIL_RECORD_BP
        f.write(b"\n")

    return total_bp


def make_sample_fastq_gz(fq_gz_path: Path) -> int:
    """Generate a small standalone gzipped FASTQ — its purpose is to
    exercise cuhll's FASTQ + gzip auto-detection, not to match the 5 Gbp
    FASTA's cardinality. We emit FASTQ_SAMPLE_BP bases worth of
    independently-random reads (constant Phred-40 quality, fixed read
    length, no headers shared with the FASTA).

    Returns the total sequenced base count.
    """
    rng = np.random.default_rng(SEED + 1)
    qual_bytes = (b"I" * FASTQ_READ_LEN) + b"\n"
    plus_bytes = b"+\n"
    n_reads = FASTQ_SAMPLE_BP // FASTQ_READ_LEN
    total_bp = 0

    # Batch reads so we make ~one gzip write per 1000 reads instead of
    # per read. ~10× faster than the read-by-read version, important at
    # 200 Mbp scale where there are 1.3M reads.
    BATCH = 1000
    out_buf = bytearray()
    with gzip.open(fq_gz_path, "wb", compresslevel=4) as fq:
        for batch_start in range(0, n_reads, BATCH):
            batch_n = min(BATCH, n_reads - batch_start)
            block = _rand_bases_bytes(batch_n * FASTQ_READ_LEN, rng)
            out_buf.clear()
            for i in range(batch_n):
                read = block[i * FASTQ_READ_LEN : (i + 1) * FASTQ_READ_LEN]
                out_buf += b"@read%d\n" % (batch_start + i)
                out_buf += read
                out_buf += b"\n"
                out_buf += plus_bytes
                out_buf += qual_bytes
                total_bp += FASTQ_READ_LEN
            fq.write(out_buf)
    return total_bp


def run_demo(tmp: Path) -> None:
    print(f"cuhll version: {cuhll.__version__}")
    print(f"workdir (auto-cleaned): {tmp}")

    fa = tmp / "challenge.fasta"
    print(f"\n[gen] writing ~5.131 Gbp challenge FASTA → {fa.name}")
    t0 = time.perf_counter()
    total_bp = make_challenging_fasta(fa)
    sz_mb = fa.stat().st_size / (1024 ** 2)
    print(f"      {total_bp:,} bp, {sz_mb:.1f} MiB on disk, "
          f"generated in {time.perf_counter() - t0:.1f}s")

    fq_gz = tmp / "challenge.fastq.gz"
    print(f"\n[gen] writing standalone {FASTQ_SAMPLE_BP // 1_000_000} Mbp "
          f"FASTQ.gz sample → {fq_gz.name}")
    print("      (independent of the FASTA — exists to exercise the "
          "FASTQ + gzip path,")
    print("       not to numerically match the 5 Gbp FASTA's cardinality)")
    t0 = time.perf_counter()
    fq_bp = make_sample_fastq_gz(fq_gz)
    sz_mb = fq_gz.stat().st_size / (1024 ** 2)
    print(f"      {fq_bp:,} bp, {sz_mb:.1f} MiB on disk, "
          f"generated in {time.perf_counter() - t0:.1f}s")

    # 1. One FASTA → cardinality
    print("\n[1] cuhll.estimate(fa, k=31)")
    n = cuhll.estimate(fa, k=K)
    print(f"    {n:,} distinct canonical {K}-mers")

    # 2. Independent FASTQ.gz sample (auto-detected by cuhll)
    print("\n[2] cuhll.estimate(fastq.gz, k=31)  — small standalone sample")
    n_fq = cuhll.estimate(fq_gz, k=K)
    print(f"    {n_fq:,} distinct canonical {K}-mers in the {fq_bp:,} bp sample")

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
    """Profile the heaviest cuhll call under nsys, save the .nsys-rep,
    and print a text summary so the user gets useful output without
    needing the GUI (nsys-ui rarely works on HPC login nodes).

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
    if r.returncode != 0:
        print(f"[profile] nsys profile failed (rc={r.returncode})",
              file=sys.stderr)
        return r.returncode

    # Print a text summary right here — nsys-ui usually can't run on
    # cluster login nodes (no OpenGL / Qt platform plugin), so the text
    # tables from `nsys stats` are the most reliable way to actually read
    # the profile in place.
    rep_file = f"{rep}.nsys-rep"
    print(f"\n[profile] saved: {rep_file}", file=sys.stderr)
    print("[profile] text summary (top CUDA API calls + kernels + memcpys):",
          file=sys.stderr)
    stats_cmd = [
        nsys, "stats",
        "--force-overwrite", "true",
        "--force-export=true",   # nuke any stale .sqlite from a prior run
        "--report", "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum",
        rep_file,
    ]
    subprocess.run(stats_cmd)
    return 0


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
    print()
    print("re-print the text summary any time with:")
    print(f"    nsys stats --force-export=true {rep_path}")
    print("(the --force-export=true is needed when re-running over a stale")
    print(" .sqlite cache nsys leaves next to the .nsys-rep)")
    print()
    print("for the GUI timeline, copy the .nsys-rep to a workstation with")
    print("Nsight Systems installed and open it there:")
    print(f"    nsys-ui {rep_path}")
    print("(nsys-ui needs OpenGL + Qt6 + a real X server, so it usually")
    print(" can't run on HPC login nodes directly.)")


if __name__ == "__main__":
    main()

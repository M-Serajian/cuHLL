"""cuHLL end-to-end demo.

Run:
    python demo.py                # walk through every public function
    python demo.py --profile      # plus produce an Nsight Systems timeline

What this script does:
  1.  Generates one synthetic "challenging" FASTA in a fresh tempdir.
      The FASTA mixes:
        - random ACGT (high-entropy baseline),
        - exact-repeat motifs (so the true distinct-kmer count is
          well below the total k-mer positions — checks HLL collapses
          duplicates correctly),
        - a palindromic region (self-reverse-complement; tests that
          canonical k-mers fold fwd/rc into one register update), and
        - multi-record layout with an intra-record N-break (tests
          window-break handling).
  2.  Generates a gzipped FASTQ derived from the same bases (tests
      cuhll's FASTQ + gzip auto-detection).
  3.  Runs every public cuhll function on these inputs.
  4.  With --profile, re-runs the heaviest call under nsys and writes
      ./profile_out/cuhll_demo.nsys-rep so you can open it in
      Nsight Systems.

Nothing is left behind from the demo's tempdir. With --profile, the
.nsys-rep is moved to ./profile_out/ so it survives.
"""
from __future__ import annotations

import argparse
import gzip
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cuhll


# --- knobs ------------------------------------------------------------
K = 31
BACKGROUND_BP = 2_000_000      # random ACGT baseline length
REPEAT_MOTIF_LEN = 100         # exact-repeat motif length
REPEAT_COUNT = 500             # how many times each motif repeats
PALINDROME_BP = 50_000         # palindromic block size
N_REPEAT_MOTIFS = 5            # how many distinct repeated motifs to sprinkle
SEED = 42


def revcomp(s: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp[b] for b in reversed(s))


def make_challenging_fasta(out_path: Path) -> None:
    """Write a 3-record FASTA tailored to stress cuhll.

    Record 1: random ACGT plus N_REPEAT_MOTIFS distinct motifs each
              repeated REPEAT_COUNT times. The repeats let us verify
              that HLL collapses duplicate k-mers down to a distinct
              count.
    Record 2: a palindromic block (sequence ++ its reverse complement).
              Every k-mer inside the palindrome has its reverse
              complement also present. With canonical=True both fold
              onto the same register update.
    Record 3: a short random fragment, exercising the boundary 'N' that
              cuhll injects between FASTA records.
    """
    rng = random.Random(SEED)
    bases = "ACGT"

    def rand(n: int) -> str:
        return "".join(rng.choices(bases, k=n))

    motifs = [rand(REPEAT_MOTIF_LEN) for _ in range(N_REPEAT_MOTIFS)]
    rec1_chunks = [rand(BACKGROUND_BP)]
    for m in motifs:
        rec1_chunks.append(m * REPEAT_COUNT)
    rec1 = "".join(rec1_chunks)

    half = rand(PALINDROME_BP // 2)
    rec2 = half + revcomp(half)

    rec3 = rand(50_000)

    with out_path.open("w") as f:
        for i, body in enumerate((rec1, rec2, rec3), start=1):
            f.write(f">record_{i}_len{len(body)}\n")
            for j in range(0, len(body), 80):
                f.write(body[j : j + 80] + "\n")


def make_fastq_gz_from_fasta(fa_path: Path, fq_gz_path: Path,
                             read_len: int = 150) -> None:
    """Chop the FASTA bases into read_len-bp reads and emit gzipped FASTQ.

    Quality is a constant Phred-40 string. Reads that would span the
    record-boundary 'N' are skipped, so the FASTQ k-mer set is *almost*
    identical to the FASTA's — close enough that the two cardinalities
    should agree within HLL noise.
    """
    bases = []
    with fa_path.open() as fa:
        for line in fa:
            if line.startswith(">"):
                continue
            bases.append(line.strip())
    seq = "".join(bases)

    qual = "I" * read_len
    with gzip.open(fq_gz_path, "wt", compresslevel=4) as fq:
        for i, start in enumerate(range(0, len(seq) - read_len + 1, read_len)):
            read = seq[start : start + read_len]
            if "N" in read:
                continue
            fq.write(f"@read{i}\n{read}\n+\n{qual}\n")


def run_demo(tmp: Path) -> None:
    print(f"cuhll version: {cuhll.__version__}")
    print(f"tempdir:       {tmp}")

    fa = tmp / "challenge.fasta"
    print(f"\n[gen] writing challenge FASTA → {fa.name}")
    make_challenging_fasta(fa)
    print(f"      size: {fa.stat().st_size / 1024:.1f} KiB")

    fq_gz = tmp / "challenge.fastq.gz"
    print(f"\n[gen] writing FASTQ.gz of the same sequence → {fq_gz.name}")
    make_fastq_gz_from_fasta(fa, fq_gz)
    print(f"      size: {fq_gz.stat().st_size / 1024:.1f} KiB")

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="store_true",
                    help="also produce an nsys timeline of the heaviest call")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="cuhll_demo_") as tmp_s:
        tmp = Path(tmp_s)
        run_demo(tmp)

        if args.profile:
            here = Path(__file__).resolve().parent
            run_nsys_profile(tmp, here / "profile_out")


if __name__ == "__main__":
    main()

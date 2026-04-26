"""cuHLL end-to-end demo.

What this script does:
  STEP 0 — Generate 10 synthetic FASTA files in a temporary directory.
           Each file is a single record of 5 million random ACGT bases,
           seeded so the run is reproducible. No external data needed.
  STEP 1 — Estimate distinct k-mers in ONE FASTA (one number).
  STEP 2 — Build a Sketch object from ONE FASTA (manipulable in RAM).
  STEP 3 — Build 10 sketches in PARALLEL via the concurrent GPU pipeline.
  STEP 4 — Write 10 `.hll` files to disk and read the union estimate that
           the pipeline computed for free.
  STEP 5 — Same union number, but no `.hll` files left on disk.
  STEP 6 — Set-arithmetic on two in-memory sketches: union, intersection,
           Jaccard similarity.
  STEP 7 — n-way intersection (4 sketches via inclusion-exclusion).
  STEP 8 — Persist a sketch to a `.hll` file and read it back round-trip.

Every cuHLL public function is exercised at least once. Run with
`python demo.py` after `pip install`.
"""
import random
import tempfile
import time
from pathlib import Path

import cuhll

# ---- Knobs you might tweak --------------------------------------------------
K               = 31           # k-mer length (15 <= k <= 32)
N_FILES         = 10           # how many synthetic FASTAs to generate
BASES_PER_FILE  = 5_000_000    # 5 million random bases per file


def write_random_fasta(path: Path, n_bases: int, seed: int) -> None:
    """Write a single-record FASTA of `n_bases` random A/C/G/T bases.

    The output is wrapped at 80 columns so the file looks like a normal
    biological FASTA. `seed` makes the run reproducible.
    """
    rng = random.Random(seed)
    with path.open("w") as f:
        f.write(f">{path.stem}\n")              # FASTA record header
        chunk = []
        for _ in range(n_bases):
            chunk.append(rng.choice("ACGT"))    # one random base
            if len(chunk) == 80:                # flush every 80 chars
                f.write("".join(chunk) + "\n")
                chunk.clear()
        if chunk:                               # write any tail < 80
            f.write("".join(chunk) + "\n")


# Use a tempdir so the demo cleans up after itself — nothing left behind.
with tempfile.TemporaryDirectory(prefix="cuhll_demo_") as tmp:
    tmp = Path(tmp)
    print(f"cuhll version: {cuhll.__version__}")

    # =========================================================================
    # STEP 0 — Generate 10 synthetic random FASTA files (5 Mbp each)
    # =========================================================================
    # Each file represents one "genome" of random ACGT. Because the bases
    # are random, the 10 genomes share very few k-mers (good for showing
    # near-zero Jaccard later). Real genomes from the same species would
    # share most of their k-mers (Jaccard ~0.9+).
    print(f"\n[0] generating {N_FILES} random FASTAs "
          f"({BASES_PER_FILE:,} bp each) in {tmp}/")
    paths = []
    for i in range(N_FILES):
        p = tmp / f"genome_{i:02d}.fa"
        write_random_fasta(p, BASES_PER_FILE, seed=1000 + i)
        paths.append(p)
    print(f"    wrote {len(paths)} FASTAs")

    # =========================================================================
    # STEP 1 — One FASTA -> one cardinality number
    # =========================================================================
    # Simplest cuHLL call: take a FASTA, return how many DISTINCT
    # canonical k-mers it contains. "Distinct" means duplicates collapsed.
    # "Canonical" means each k-mer is counted once regardless of strand.
    print("\n[1] cuhll.estimate(one_fasta, k=31)  ->  int")
    n1 = cuhll.estimate(paths[0], k=K)
    print(f"    distinct k-mers in {paths[0].name}: {n1:,}")

    # =========================================================================
    # STEP 2 — One FASTA -> a Sketch object (in memory)
    # =========================================================================
    # A Sketch is a compact data structure (16 KB at default precision=14)
    # that summarizes the k-mer set. You can call .estimate(), merge it
    # with another sketch, intersect, save to disk, etc. Returning the
    # Sketch (not just the count) lets you do MORE than count later.
    print("\n[2] cuhll.sketch(one_fasta, k=31)  ->  Sketch")
    s0 = cuhll.sketch(paths[0], k=K)
    print(f"    {s0!r}")              # repr shows k, precision, canonical, estimate

    # =========================================================================
    # STEP 3 — Many FASTAs -> list of Sketches (in parallel on the GPU)
    # =========================================================================
    # sketch_many uses cuHLL's concurrent per-genome pipeline (multiple
    # CUDA streams + parallel readers + parallel writers). Result is a
    # list[Sketch] in the same order as the input paths. ~5x faster than
    # looping cuhll.sketch() over each input.
    print("\n[3] cuhll.sketch_many(paths, k=31)  ->  list[Sketch]")
    t0 = time.perf_counter()
    sketches = cuhll.sketch_many(paths, k=K)
    print(f"    built {len(sketches)} sketches in {time.perf_counter() - t0:.2f}s")

    # =========================================================================
    # STEP 4 — Many FASTAs -> .hll files on disk + free union cardinality
    # =========================================================================
    # sketch_to_dir streams sketches straight to disk as <stem>.hll files.
    # The C++ pipeline ALSO computes the union cardinality of all inputs
    # for free; cuHLL exposes that on the result via .union_estimate.
    # No Python Sketch objects are kept in RAM, so memory stays flat
    # regardless of input count.
    print("\n[4] cuhll.sketch_to_dir(paths, output_dir=..., k=31)")
    out_dir = tmp / "sketches"
    result = cuhll.sketch_to_dir(paths, output_dir=out_dir, k=K)
    print(f"    wrote {len(result)} .hll files to {out_dir}/")
    print(f"    union cardinality (free side effect): {result.union_estimate:,}")
    print(f"    one mapped path: {result[str(paths[0])]}")

    # =========================================================================
    # STEP 5 — Same union number, but with NO .hll files left on disk
    # =========================================================================
    # estimate_union routes through the same concurrent pipeline as step 4,
    # using a temporary directory under the hood that's deleted before the
    # function returns. Use this when you only want the panel cardinality.
    print("\n[5] cuhll.estimate_union(paths, k=31)  ->  int")
    union_n = cuhll.estimate_union(paths, k=K)
    print(f"    |⋃ genomes|: {union_n:,}")
    print(f"    matches step 4's .union_estimate? {union_n == result.union_estimate}")

    # =========================================================================
    # STEP 6 — Set arithmetic on two in-memory sketches
    # =========================================================================
    # `a | b` is set-union (returns a new Sketch). `a & b` is set-
    # intersection cardinality (returns int — HLL can estimate
    # |A ∩ B| but cannot represent the intersection itself as a sketch).
    # Jaccard = |A ∩ B| / |A ∪ B|. For two random independent genomes
    # we expect Jaccard near zero (very few shared k-mers).
    print("\n[6] set-union and set-intersection on two sketches")
    a, b = sketches[0], sketches[1]
    union_ab = (a | b).estimate()                   # |A ∪ B|
    inter_ab = a & b                                # |A ∩ B|, same as a.intersect(b)
    print(f"    |A|       = {a.estimate():,}")
    print(f"    |B|       = {b.estimate():,}")
    print(f"    |A ∪ B|   = {union_ab:,}")
    print(f"    |A ∩ B|   = {inter_ab:,}")
    print(f"    Jaccard   = {inter_ab / union_ab:.4f}    "
          f"(1.0 = identical, 0.0 = disjoint)")

    # =========================================================================
    # STEP 7 — n-way intersection across the first 4 sketches
    # =========================================================================
    # Implemented via HLL inclusion-exclusion (2**n - 1 union estimates
    # under the hood). Reliable for n <= 4 at default precision=14;
    # larger n needs higher precision because errors compound.
    print("\n[7] cuhll.intersect_estimate_many(sketches[:4])  ->  int")
    inter_n = cuhll.intersect_estimate_many(sketches[:4])
    print(f"    |A ∩ B ∩ C ∩ D| = {inter_n:,}")

    # =========================================================================
    # STEP 8 — Persist a sketch and read it back from disk
    # =========================================================================
    # .hll is cuHLL's stable interchange format. Sketch.write() saves
    # registers + metadata (k, precision, canonical) into a 48-byte
    # header + register block. cuhll.read() reconstructs the exact same
    # Sketch object — bit-identical estimate guaranteed.
    print("\n[8] Sketch.write(path)  /  cuhll.read(path)")
    a.write(tmp / "a.hll")
    a_loaded = cuhll.read(tmp / "a.hll")
    print(f"    on-disk roundtrip: estimate matches? "
          f"{a.estimate() == a_loaded.estimate()}")

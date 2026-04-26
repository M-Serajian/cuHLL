# 🧬 cuHLL

```
              _   _  _      _
  ___  _   _ | | | || |    | |
 / __|| | | || |_| || |    | |
| (__ | |_| ||  _  || |___ | |___
 \___| \__,_||_| |_||_____||_____|
```

**CUDA-accelerated HyperLogLog** for distinct k-mer cardinality estimation
on genomic FASTA inputs. cuHLL turns one or many FASTA files into compact,
mergeable HyperLogLog sketches on the GPU, persists them as `.hll` files,
and supports set-arithmetic (union, intersection) on those sketches from
both Python and C++.

Reference performance: **2,000 chr19 human genomes (~118 GB) in ~45 s on a
single L4 GPU** via the concurrent per-genome pipeline.

---

## 📦 Install

Build from source — the supported path today. You need a CUDA Toolkit
(`nvcc`) and a C++ compiler (GCC 11+) available at build time:

```bash
git clone https://github.com/M-Serajian/cuHLL.git
cd cuHLL
pip install .
```

That's it — `import cuhll` works in Python. On HPC systems, load the
modules first (e.g. `module load python/3.10 cuda/12.8.1 gcc/14.2.0`).
For a C++/CLI-only build (no Python) see
[Build from source (CMake)](#-build-from-source-cmake) below.

### PyPI install (coming soon)

```bash
pip install cuhll       # not yet active — wheel pipeline ready, first release pending
```

Once the first tagged release is published, `pip install cuhll` will
fetch a pre-built wheel that bundles `libcudart.so.12` — no CUDA
toolkit, no compiler, no module loads needed. Only requirement on the
target machine: an **NVIDIA driver**. The cibuildwheel pipeline that
builds these wheels is already in place at
[`.github/workflows/wheels.yml`](.github/workflows/wheels.yml); it
triggers on `git tag v*` and ships wheels for Python 3.9–3.13 on Linux
x86_64.

---

## 🐍 Python API

```python
import cuhll
```

### Functions

| function | inputs | returns | what it does |
|---|---|---|---|
| `cuhll.estimate(fasta, *, k)` | one FASTA path | `int` | distinct k-mer cardinality of one FASTA |
| `cuhll.sketch(fasta, *, k)` | one FASTA path | `Sketch` | build one in-memory sketch |
| `cuhll.sketch_many(paths, *, k)` | list of FASTA paths | `list[Sketch]` | build N sketches in parallel (concurrent pipeline) |
| `cuhll.sketch_to_dir(paths, *, output_dir, k)` | list of FASTAs + dir | `SketchDirResult` (dict + `.union_estimate`) | sketch each FASTA → write `<stem>.hll` to disk; also returns union cardinality |
| `cuhll.sketch_union(paths, *, k)` | list of FASTAs | `Sketch` | one merged panel sketch |
| `cuhll.estimate_union(paths, *, k)` | list of FASTAs | `int` | union cardinality across the panel (~45 s for 2k chr19 genomes / L4) |
| `cuhll.intersect_estimate(a, b)` | two `Sketch` | `int` | `\|A ∩ B\|` via inclusion-exclusion |
| `cuhll.intersect_estimate_many(sketches)` | list of `Sketch` | `int` | n-way intersection (reliable for n ≤ 4) |
| `cuhll.read(path)` | `.hll` path | `Sketch` | load a saved sketch |
| `cuhll.read_header(path)` | `.hll` path | `HllFileHeader` | metadata only, no register I/O |
| `Sketch.estimate()` | — | `int` | cardinality of this sketch |
| `Sketch.merge(other)` / `a \| b` | another `Sketch` | `Sketch` | set-union |
| `Sketch.intersect(other)` / `a & b` | another `Sketch` | `int` | set-intersection cardinality |
| `Sketch.write(path)` | path | `None` | persist sketch to `.hll` |
| `Sketch.clone()` | — | `Sketch` | deep copy of registers |

**Common keyword args** for all sketching functions:

| arg | default | meaning |
|---|---|---|
| `k` | required | k-mer length (15 ≤ k ≤ 32) |
| `precision` | `14` | HLL precision; `2**precision` registers, `~1.04/√(2**p)` relative error |
| `canonical` | `True` | count canonical k-mers (`min(fwd, rev_comp)`) |
| `verbose` | `False` | print pipeline diagnostics (kernel occupancy, per-genome estimates) |

### End-to-end runnable demo

A complete runnable demo lives at [`demo.py`](demo.py) in the repo
root. After `pip install`, just run:

```bash
python demo.py
```

The script generates 10 synthetic FASTA files (5 million random bases
each) in a tempdir, then exercises every public cuHLL function on
them — no external data needed. Full source is reproduced below for
reference.

```python
"""cuHLL end-to-end demo.

What this script does, top to bottom:
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
```

**HLL intersection caveat.** `|A| + |B| - |A ∪ B|` subtracts two HLL
estimates, so when the true intersection is much smaller than the union
the relative error explodes (catastrophic cancellation). Rule of thumb at
default `precision=14`: trust the intersection only when it's >5% of the
union. For small intersections, bump `precision=16+` or use an exact
counter (KMC, jellyfish).

---

## 🔧 Build from source (CMake)

For users who want the standalone CLI binary or the C++ static library
(no Python):

```bash
mkdir build && cd build
cmake ..                # default fatbin: sm_75/80/86/89/90 (Turing → Hopper)
cmake --build . -j
```

Output binaries land in `build/bin/`:
- `cuhll` — main CLI
- `cuhll_pack` — FASTA → `.cb2` packed-binary converter
- `cuco_probe` — minimal sm_70+ smoke test

Common flags:

| flag | what it does |
|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES=89` | build for one GPU only (smaller, faster compile) |
| `-DCMAKE_CUDA_ARCHITECTURES=NATIVE` | detect GPU on the build host (fails clean if none) |
| `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100"` | include Blackwell (sm_100) — needs CUDA ≥12.8 |
| `-DCUHLL_BUILD_TESTS=ON` | also build the C++ CTest suite |

CPM auto-fetches cuCollections, CCCL, and pybind11 the first time. The
C++/CUDA build doesn't need a GPU on the host — only at runtime.

---

## ▶️ CLI usage

```bash
cuhll --k 31 input.fasta                            # one estimate to stdout
cuhll --k 31 a.fa b.fa c.fa                         # union cardinality
cuhll --k 31 --output-dir sketches/ *.fa            # one .hll per input
```

| arg | default | meaning |
|---|---|---|
| `--k <int>` | required | k-mer length (15–32) |
| `--precision <int>` | 14 | HLL precision |
| `--no-canonical` | (canonical) | count both strands separately |
| `--output-dir <dir>` | (none) | write per-genome `.hll` files; otherwise prints union estimate |
| `--chunk-mb <int>` | 64 | streaming chunk size |

### `cuhll_pack` (offline FASTA → `.cb2`)

`.cb2` is cuHLL's filtered 2-bit-per-base format for repeat sketching:
```bash
cuhll_pack input.fasta out.cb2
cuhll --k 31 out.cb2                                # ~3× faster than re-parsing FASTA
```

---

## 📊 `.hll` file format

48-byte header + register block (little-endian, fixed layout for stable
interchange):

| offset | size | field | notes |
|---:|---:|---|---|
| 0  | 8 | magic       | `"CUHLLv02"` ASCII |
| 8  | 4 | version     | `2` (current) |
| 12 | 4 | precision_p | HLL `p`; register count = `2**p` |
| 16 | 4 | k           | k-mer length |
| 20 | 4 | hash_type   | `1` = xxhash_64 |
| 24 | 8 | n_registers | `2**p`, redundant for sanity |
| 32 | 8 | register_bytes | `4 * n_registers` |
| 40 | 1 | canonical   | `1` if canonical mode, else `0` |
| 41 | 7 | (reserved)  | zeroed |
| 48 | … | registers   | `n_registers` × `uint32_t` |

Same bytes from both Python (`Sketch.write`) and C++ (`cuhll::write_hll`).

---

## 📋 Requirements

### Build-time

| component | minimum | tested | how to get it |
|---|---|---|---|
| CUDA Toolkit (`nvcc`) | 12.0 | 12.4, 12.8.1, 12.9.1 | NVIDIA installer / `module load cuda/12.8.1` |
| GCC / G++ | 11 | 11, 12, 14.2.0 | distro pkg / `module load gcc/14.2.0` |
| Python | 3.9 | 3.10, 3.12 | system / `module load python/3.10` |
| CMake | 3.30 | 3.30.5, 4.3 | auto-fetched by pip's build env |
| GPU compute capability | sm_70 (Volta) | sm_89 (L4) | runtime only — no GPU needed at build time |
| OS | Linux x86_64 | RHEL 9 | — |

`scikit-build-core`, `pybind11`, `Ninja`, `cuCollections`, `CCCL`,
`pybind11` are all auto-fetched. No manual install needed.

### Runtime (after `pip install` of the source build, or once `cuhll-cu12` ships)

| component | minimum | notes |
|---|---|---|
| NVIDIA driver | sufficient for your CUDA major | the `cuhll-cu12` wheel will bundle `libcudart`, so no toolkit needed |
| Python | matches the wheel's `cpXY` | |
| `numpy` | ≥ 1.20 | auto-installed by pip |

### Test-only (`pip install ".[test]"`)

| component | role |
|---|---|
| `pytest` | Python test runner |
| KMC3 (`kmc`, `kmc_tools`) | optional ground-truth comparison via [`tools/validate_*_against_kmc.sh`](tools/) (HiPerGator: `module load kmc/3.2.1`) |

---

## 🧪 Tests

All commands below are run **from the project root** (`cuHLL/`). pytest
auto-discovers the test directory via `pyproject.toml`'s `testpaths`, so
plain `pytest` works — no need to type the path.

### Python tests (the common case)

```bash
cd /path/to/cuHLL                                   # the project root
pip install ".[test]"                               # one-time, installs pytest
pytest                                              # run all 17 Python tests
pytest -v                                           # verbose: print every test name
pytest -k intersect                                 # run only intersect-related tests
```

GPU-bound tests (any test marked `@pytest.mark.gpu`) auto-skip on hosts
without an NVIDIA GPU; on a GPU node they run.

### C++ CTest suite (opt-in)

```bash
cd /path/to/cuHLL
pip install ".[test]" --config-settings=cmake.define.CUHLL_BUILD_TESTS=ON
ctest --test-dir build/$(ls build | head -1) --output-on-failure
```

(Or build directly without pip via `cmake .. -DCUHLL_BUILD_TESTS=ON && make -j` then `ctest`.)

### KMC ground-truth validation (Layer 2)

```bash
cd /path/to/cuHLL
module load kmc/3.2.1                               # HiPerGator; or install KMC3 yourself
CUHLL_KMC_BIN=$(which kmc) CUHLL_KMC_TOOLS_BIN=$(which kmc_tools) \
    bash tools/validate_against_kmc.sh              # union vs KMC exact count
CUHLL_KMC_BIN=$(which kmc) CUHLL_KMC_TOOLS_BIN=$(which kmc_tools) \
    bash tools/validate_intersect_against_kmc.sh    # intersect vs KMC exact count
```

---

## 🐛 Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Failed to find nvcc.` | CUDA toolkit not on PATH | `module load cuda/12.8.1` then re-run |
| `error: identifier "TIME_UTC" is undefined` | CUDA 13 + GCC ≤ 10 | `module load gcc/12+` |
| `libcudart.so.12: cannot open shared object file` (or `.so.13`) | wheel's CUDA major ≠ runtime's | rebuild OR load matching `cuda/<major>.x` module |
| `conda/bin/ld: ... undefined reference to ...@GLIBC_2.34` | conda Python's bundled `ld` shadows system linker | auto-detected; CMake injects `-B /usr/bin/`. Disable with `-DCUHLL_KEEP_CONDA_LD=ON` |
| `ImportError: ...libstdc++.so.6: GLIBCXX_3.4.32 not found` | conda Python's old libstdc++ wins via DT_RPATH | auto-detected; bindings link libstdc++ statically |
| `Cryptic CMake error after toolchain switch` | stale `build/{wheel_tag}/` | `rm -rf build/ && pip install .` |
| `pip install .` succeeds but `import cuhll` is the old version | pip skipped the rebuild | `pip install --force-reinstall --no-deps .` |
| GPU utilization low (slow run, low `nvidia-smi` util) | I/O-bound (NFS / Lustre cold) | use `/blue` / node-local scratch; warm the page cache; add CPUs |

For a one-line environment dump useful in bug reports:
```bash
python -m cuhll.diagnose                            # works post-install anywhere
python python/cuhll/diagnose.py                     # works pre-install too
```

---

## 📚 C++ API reference

```cpp
#include <cuHLL/sketch.hpp>          // Sketch class
#include <cuHLL/pipeline.hpp>        // sketch_sequences_streaming
#include <cuHLL/concurrent.hpp>      // sketch_per_genome_auto
#include <cuHLL/hll_file.hpp>        // read_hll / write_hll
```

### Core types

| type | header | purpose |
|---|---|---|
| `cuhll::Sketch` | `sketch.hpp` | RAII over `cuco::hyperloglog`; carries precision + canonical flag |
| `cuhll::HllFileHeader` | `hll_file.hpp` | POD layout of a `.hll` header |

### Sketch operations

| call | does |
|---|---|
| `Sketch(precision, canonical)` | construct empty sketch |
| `sketch.estimate()` | uint64 cardinality estimate |
| `sketch.merge(other)` | in-place set-union (validates same precision/canonical) |
| `sketch.clone()` | deep copy |
| `sketch.copy_registers_to_host(uint32_t* out)` | dump register state for serialization |
| `sketch.load_registers_from_host(const uint32_t* in)` | inverse — build sketch from registers |

### Pipeline / file I/O

| call | does |
|---|---|
| `sketch_sequences_streaming(s, paths, k, chunk_mb)` | sequential single-pass union (low-memory; one CUDA stream) |
| `sketch_per_genome_auto(paths, output_dir, k, p, canonical)` | concurrent per-genome pipeline; writes `.hll`s; returns union cardinality |
| `read_hll(path)` / `write_hll(path, s, k)` | persist sketch to `.hll` |
| `read_hll_header(path)` | header-only metadata |

### Constants ([`common.hpp`](include/cuHLL/common.hpp))

`kMinK=15`, `kMaxK=32`, `kMinPrecision=8`, `kMaxPrecision=18`,
`kDefaultPrecision=14`, `kDefaultChunkMB=64`, `kHllFileVersion=2`.

### Linking from another CMake project

```cmake
add_subdirectory(cuHLL)                   # or FetchContent / CPMAddPackage
target_link_libraries(your_target PRIVATE cuhll_core)
```

`cuhll_core` is a static library carrying both host and device code,
device-symbol-resolved, position-independent. It encapsulates all
CUDA dependencies — your downstream project doesn't need to know cuco
exists.

---

## 🧠 How it works

cuHLL parses FASTA, hashes canonical k-mers with **ntHash → xxhash_64**
on the GPU, and updates registers in a `cuco::hyperloglog` sketch via
device-scope atomics (sm_70+ requirement).

The concurrent pipeline (`sketch_per_genome_auto`) overlaps:
- **multiple reader threads** pulling FASTA bytes off disk
- **3+ CUDA streams** running kernels in parallel with H2D copies
- **writer threads** draining finished sketches to `.hll`

Reader/writer/stream counts are auto-tuned at startup from CPU count, GPU
SM count, free VRAM, and host RAM. No flags. End-to-end throughput on
sustained 2,000-genome chr19 panels: **~2.5 GB/s on a single L4** —
saturating NFS storage rather than the GPU.

The sequential streaming path (`sketch_sequences_streaming`) is the
low-memory fallback (single stream, single reader, ~5× slower) used by
some single-FASTA entry points; the Python API routes large panels
through the concurrent path automatically.

---

## 📄 License

See [LICENSE](LICENSE).

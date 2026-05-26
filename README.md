# 🧬 cuHLL

A GPU implementation of HyperLogLog for counting distinct k-mers in
FASTA or FASTQ files (gzip is fine). Builds and merges sketches on the
GPU, writes them to `.hll` files, and lets you query them later from
either Python or C++.

On one NVIDIA L4 it runs the union of 2,048 chr19 genomes (~118 GB) in
about 30 seconds. The cardinality lands within HLL's expected error of
KMC3's exact count.

## Install

cuHLL is not on PyPI yet. Build it from source:

```bash
git clone https://github.com/M-Serajian/cuHLL.git
cd cuHLL
pip install .
```

You need a CUDA Toolkit (≥ 12.0) and GCC (≥ 11) somewhere on the
system. On HPC nodes that's usually `module load cuda/12.8.1 gcc/14.2.0`.
pip fetches a recent CMake itself, and zlib comes from your distro.
The CLI binary ends up at `build/<wheel-tag>/bin/cuhll`.

## Use

```python
import cuhll
cuhll.estimate("genome.fasta", k=31)                          # one file → int
cuhll.estimate_union(["a.fa", "b.fa.gz", "c.fastq.gz"], k=31) # panel union → int
cuhll.sketch_to_dir(paths, output_dir="sketches/", k=31)      # per-genome .hll + union
```

```bash
cuhll --k 31 --list manifest.txt              # union to stdout
cuhll --k 31 --keep-sketches --list m.txt     # also write .hll files
cuhll --help                                  # all flags
```

Inputs can be `.fasta`, `.fa`, `.fna`, `.fastq`, or `.fq`, optionally
gzipped — the format is detected from the file's contents, not its
name. Default precision is 14 (about 0.81% relative error), and k-mers
are canonical by default (a k-mer and its reverse complement count as
the same thing).

## Python API

| function | what it does |
|---|---|
| `estimate(path, k)` | distinct k-mer count for one file |
| `sketch(path, k)` | one in-memory `Sketch` |
| `sketch_many(paths, k)` | N sketches via the concurrent pipeline |
| `sketch_union(paths, k)` | one merged sketch (panel union) |
| `estimate_union(paths, k)` | union cardinality, no per-genome state |
| `sketch_to_dir(paths, output_dir, k)` | per-genome `.hll` + union |
| `intersect_estimate(a, b)` | $\|A \cap B\|$ via inclusion-exclusion |
| `intersect_estimate_many(sketches)` | n-way intersection (n ≤ 4 reliable) |
| `read(path)` | load a saved sketch |

[`demo.py`](demo.py) walks through every public function on a few
synthetic FASTAs it generates in a tempdir.

## Tests

```bash
pip install ".[test]"
pytest                          # 17 Python tests
```

To check the cuHLL estimates against an exact ground truth, install
KMC3 (`module load kmc/3.2.1` on HPC) and run
`bash tools/validate_against_kmc.sh`.

## Troubleshooting

| symptom | fix |
|---|---|
| `Failed to find nvcc.` | `module load cuda/12.8.1` |
| `cuHLL: GCC X.Y is too old` | `module load gcc/12.2.0` (or newer) |
| `CMake X.Y or higher is required` | `module unload cmake` — let pip fetch one |
| `Can't connect to HTTPS URL (SSL)` | HPC `python/X` module has no SSL — use system `python3` or a venv |
| `libstdc++.so.6: GLIBCXX_*` not found | rebuild with `-DCUHLL_STATIC_LIBSTDCXX=ON` (default on auto-discovered gcc) |

If you're filing a bug, please include the output of
`python -m cuhll.diagnose`.

## License

See [`LICENSE`](LICENSE).

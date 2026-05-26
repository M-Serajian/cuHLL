# 🧬 cuHLL

A GPU implementation of HyperLogLog for counting distinct k-mers in
FASTA or FASTQ files (gzip is fine). Builds and merges sketches on the
GPU, writes them to `.hll` files, and lets you query them later from
either Python or C++.

## Requirements

| component | minimum | notes |
|---|---|---|
| NVIDIA GPU | sm_70 (Volta) or newer | runtime only |
| NVIDIA driver | matches your CUDA major | runtime only |
| CUDA Toolkit | 12.0 | build-time; `module load cuda/12.8.1` on HPC |
| GCC / G++ | 11 | build-time; `module load gcc/14.2.0` on HPC |
| Python | 3.9 | for the Python package; use system Python or a venv |
| CMake | 4.0 | auto-fetched by the pip build env |
| zlib | any modern version | distro package; needed for `.gz` reading |
| Linux x86_64 | RHEL 9 / Ubuntu 20.04+ tested | other distros likely fine |

## Install

cuHLL is not on PyPI yet. Two ways to build it from source.

### 1. Python package (recommended)

```bash
git clone https://github.com/M-Serajian/cuHLL.git
cd cuHLL
pip install .
```

This builds the C++/CUDA library, the Python bindings, and the `cuhll`
CLI binary all at once. After it finishes, `import cuhll` works and
the CLI is at `build/<wheel-tag>/bin/cuhll`.

### 2. CMake-only (C++/CLI without Python)

If you only want the C++ library and the CLI:

```bash
git clone https://github.com/M-Serajian/cuHLL.git
cd cuHLL
mkdir build && cd build
cmake ..
cmake --build . -j
```

Output binaries land in `build/bin/cuhll` and `build/bin/cuco_probe`.
Common flags:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89    # one GPU arch (smaller, faster compile)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=NATIVE # detect from the GPU on the build host
cmake .. -DCUHLL_BUILD_TESTS=ON            # also build the C++ test suite
```

To link cuhll from another CMake project:

```cmake
add_subdirectory(cuHLL)
target_link_libraries(my_target PRIVATE cuhll_core)
```

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

To check cuHLL estimates against an exact ground truth, install KMC3
(`module load kmc/3.2.1` on HPC) and run
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

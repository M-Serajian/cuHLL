// main.cpp — cuHLL CLI entry point.
//
// Output: one integer on stdout — the estimated distinct k-mer count across
// all input FASTAs. Per-stage timings go to stderr under --verbose.

#include "cuHLL/common.hpp"
#include "cuHLL/concurrent.hpp"
#include "cuHLL/fasta.hpp"
#include "cuHLL/hll_file.hpp"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// --- path classification for auto-select routing (H4) ----------------------
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Strip a trailing ".gz" (case-insensitive) and return the inner extension.
// e.g. "/path/to/foo.fastq.gz" → ".fastq". For non-gzipped paths the
// path's own extension is returned.
std::string sequence_inner_ext(const std::string& p) {
    std::filesystem::path pp(p);
    const auto outer = lower(pp.extension().string());
    if (outer == ".gz") {
        return lower(std::filesystem::path(pp.stem()).extension().string());
    }
    return outer;
}

bool has_sequence_ext(const std::string& p) {
    const auto e = sequence_inner_ext(p);
    return e == ".fasta" || e == ".fa" || e == ".fna"
        || e == ".fastq" || e == ".fq";
}

// Validate one input path. cuhll's reader now transparently handles FASTA,
// FASTQ, and gzip-compressed variants of both via zlib.
std::string validate_input(const std::string& path) {
    if (!has_sequence_ext(path)) {
        throw std::runtime_error(
            "unknown input extension (expected .fasta/.fa/.fna/.fastq/.fq, "
            "optionally with .gz): " + path);
    }
    return path;
}

// --tmpdir > $TMPDIR > /tmp. Returns the parent; TempDir mkdtemp's a
// unique subdir under it.
std::string resolve_tmpdir_base(const std::string& cli_tmpdir) {
    if (!cli_tmpdir.empty()) return cli_tmpdir;
    if (const char* env = std::getenv("TMPDIR"); env && env[0] != '\0') {
        return std::string(env);
    }
    return "/tmp";
}

// RAII tempdir at <base>/cuhll-<pid>-<random6>/. mkdtemp(3) creates it
// atomically (no race even with many processes on the same shared FS)
// at mode 0700. Destructor removes it best-effort; a SIGKILL leaves it
// behind for tmpwatch to reap.
class TempDir {
public:
    explicit TempDir(const std::string& base_dir) {
        // mkdtemp wants a mutable buffer ending in "XXXXXX".
        std::string templ = base_dir + "/cuhll-" +
                            std::to_string(::getpid()) + "-XXXXXX";
        std::vector<char> buf(templ.begin(), templ.end());
        buf.push_back('\0');
        if (::mkdtemp(buf.data()) == nullptr) {
            const int err = errno;
            throw std::runtime_error(
                std::string("cuHLL: mkdtemp(") + templ + ") failed: "
                + std::strerror(err)
                + ". Pass --tmpdir <dir> to point at a writable parent.");
        }
        path_ = std::filesystem::path(buf.data());
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);  // best-effort
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
};

// dup2 /dev/null over fd 1 to swallow printf/std::cout from C++ callees,
// reverse on release(). The fflush before each dup2 is load-bearing —
// without it, libc-buffered bytes leak across the swap.
class StdoutSuppressor {
public:
    StdoutSuppressor() = default;
    ~StdoutSuppressor() { release(); }
    StdoutSuppressor(const StdoutSuppressor&) = delete;
    StdoutSuppressor& operator=(const StdoutSuppressor&) = delete;

    void engage() {
        if (engaged_) return;
        std::fflush(stdout);
        std::cout.flush();
        saved_fd_   = ::dup(STDOUT_FILENO);
        devnull_fd_ = ::open("/dev/null", O_WRONLY);
        if (saved_fd_ < 0 || devnull_fd_ < 0) {
            if (devnull_fd_ >= 0) ::close(devnull_fd_);
            if (saved_fd_   >= 0) ::close(saved_fd_);
            saved_fd_ = devnull_fd_ = -1;
            return;  // best-effort: skip suppression
        }
        ::dup2(devnull_fd_, STDOUT_FILENO);
        engaged_ = true;
    }
    void release() {
        if (!engaged_) return;
        std::fflush(stdout);
        std::cout.flush();
        ::dup2(saved_fd_, STDOUT_FILENO);
        ::close(saved_fd_);
        ::close(devnull_fd_);
        saved_fd_ = devnull_fd_ = -1;
        engaged_ = false;
    }
private:
    bool engaged_   = false;
    int  saved_fd_  = -1;
    int  devnull_fd_ = -1;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --k <K> [options] <input> [<input> ...]\n"
        "\n"
        "Inputs: FASTA or FASTQ (*.fasta / *.fa / *.fna / *.fastq / *.fq),\n"
        "        optionally gzip-compressed (*.gz). Format is auto-detected\n"
        "        from the gzip magic bytes and the first content byte.\n"
        "\n"
        "Options:\n"
        "  --k            k-mer length (required; %d <= k <= %d)\n"
        "  --precision    HLL precision (default %d; range %d..%d)\n"
        "  --chunk-mb     streaming chunk size in MiB (default %zu; min 1)\n"
        "  --canonical    (default) count canonical k-mers (min(fwd, rc))\n"
        "  --no-canonical count forward-strand k-mers only\n"
        "  --per-genome   emit one estimate per input plus a final UNION line\n"
        "  --output-dir D write one <stem>.hll per input to D (implies --per-genome)\n"
        "  --keep-sketches\n"
        "                 like --output-dir but auto-names the directory as\n"
        "                 ./cuhll_sketches_<YYYYMMDD-HHMMSS>_pid<PID>/ (implies\n"
        "                 --per-genome). Default behavior (neither flag) discards\n"
        "                 per-genome sketches in a tempdir after computing the union.\n"
        "  --tmpdir DIR   parent directory for transient per-genome sketches when\n"
        "                 neither --keep-sketches nor --output-dir is given. The\n"
        "                 subdirectory is created via mkdtemp(3) (atomic, mode 0700,\n"
        "                 collision-safe across concurrent processes/GPUs). Precedence:\n"
        "                 --tmpdir > $TMPDIR > /tmp. Useful on HPC where /tmp may be\n"
        "                 small; point at /blue/$GROUP/scratch for million-genome panels.\n"
        "  --list F       read input paths from manifest F (one path per line)\n"
        "  --verbose      print per-stage timings to stderr\n"
        "\n"
        "Output: one integer per line on stdout. Default mode emits a single line\n"
        "(the union cardinality); --per-genome emits <path>\\t<est> per input plus\n"
        "a trailing UNION line.\n",
        prog,
        cuhll::kMinK, cuhll::kMaxK,
        cuhll::kDefaultPrecision, cuhll::kMinPrecision, cuhll::kMaxPrecision,
        cuhll::kDefaultChunkMB);
}

int die_usage(const char* prog, const char* msg) {
    std::fprintf(stderr, "%s: %s\n", prog, msg);
    print_usage(prog);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    int k = -1;
    int precision = cuhll::kDefaultPrecision;
    int chunk_mb = static_cast<int>(cuhll::kDefaultChunkMB);
    bool verbose = false;
    bool per_genome = false;
    bool keep_sketches = false;        // --keep-sketches: auto-name an
                                       // output dir under cwd and keep
                                       // the per-genome .hll files there.
    std::string output_dir;            // empty = no per-genome sketch files
    std::string tmpdir_cli;            // --tmpdir override; empty = use
                                       // $TMPDIR, then /tmp.
    bool canonical = true;             // L1/L2: canonical k-mers (default)
    int  canonical_flag_seen = 0;      // 0=none, 1=--canonical, 2=--no-canonical
    std::vector<std::string> fasta_paths;

    auto need_value = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s: missing value for %s\n", argv[0], flag);
            std::exit(2);
        }
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--k") {
            k = std::stoi(need_value(i, "--k"));
        } else if (arg == "--precision") {
            precision = std::stoi(need_value(i, "--precision"));
        } else if (arg == "--list") {
            const std::string manifest_path = need_value(i, "--list");
            std::ifstream mf(manifest_path);
            if (!mf) {
                std::fprintf(stderr, "%s: cannot open manifest '%s'\n",
                             argv[0], manifest_path.c_str());
                return 2;
            }
            std::string line;
            while (std::getline(mf, line)) {
                while (!line.empty() &&
                       (line.back() == '\r' || line.back() == '\n' ||
                        line.back() == ' '  || line.back() == '\t')) {
                    line.pop_back();
                }
                if (line.empty() || line.front() == '#') continue;
                fasta_paths.push_back(line);
            }
        } else if (arg == "--chunk-mb") {
            chunk_mb = std::stoi(need_value(i, "--chunk-mb"));
        } else if (arg == "--per-genome") {
            per_genome = true;
        } else if (arg == "--canonical") {
            if (canonical_flag_seen == 2) {
                std::fprintf(stderr, "%s: --canonical and --no-canonical are mutually exclusive\n", argv[0]);
                return 2;
            }
            canonical = true;
            canonical_flag_seen = 1;
        } else if (arg == "--no-canonical") {
            if (canonical_flag_seen == 1) {
                std::fprintf(stderr, "%s: --canonical and --no-canonical are mutually exclusive\n", argv[0]);
                return 2;
            }
            canonical = false;
            canonical_flag_seen = 2;
        } else if (arg == "--keep-sketches") {
            keep_sketches = true;
        } else if (arg == "--tmpdir") {
            tmpdir_cli = need_value(i, "--tmpdir");
        } else if (arg == "--output-dir") {
            output_dir = need_value(i, "--output-dir");
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "%s: unknown flag '%s'\n", argv[0], arg.c_str());
            print_usage(argv[0]);
            return 2;
        } else {
            fasta_paths.push_back(std::move(arg));
        }
    }

    if (k < cuhll::kMinK || k > cuhll::kMaxK) {
        std::fprintf(stderr,
            "%s: --k must be in [%d, %d]; got %d. "
            "k > %d is not yet supported (single-uint64 data path; see README).\n",
            argv[0], cuhll::kMinK, cuhll::kMaxK, k, cuhll::kMaxK);
        return 2;
    }
    if (precision < cuhll::kMinPrecision || precision > cuhll::kMaxPrecision) {
        std::fprintf(stderr,
            "%s: --precision must be in [%d, %d]; got %d.\n",
            argv[0], cuhll::kMinPrecision, cuhll::kMaxPrecision, precision);
        return 2;
    }
    if (chunk_mb < 1) {
        std::fprintf(stderr,
            "%s: --chunk-mb must be >= 1; got %d.\n",
            argv[0], chunk_mb);
        return 2;
    }
    if (fasta_paths.empty()) {
        return die_usage(argv[0], "no FASTA input given");
    }

    double stream_ms = 0.0, estimate_ms = 0.0;

    auto t_total = clk::now();

    // Startup banner: always emit the mode so users see what was picked.
    std::fprintf(stderr, "[cuHLL] mode: %s\n",
                 canonical ? "canonical" : "non-canonical");

    try {
        // --keep-sketches: pick ./cuhll_sketches_<ts>_pid<PID>/ as the
        // output dir. PID guards against same-second collisions in
        // multi-GPU / parallel-batch jobs.
        if (keep_sketches && output_dir.empty()) {
            const std::time_t t = std::time(nullptr);
            std::tm tm_local{};
            localtime_r(&t, &tm_local);
            char stamp[32];
            std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm_local);
            output_dir = std::string("./cuhll_sketches_") + stamp +
                         "_pid" + std::to_string(::getpid());
            std::fprintf(stderr, "[cuHLL] --keep-sketches: writing per-genome "
                                 ".hll files to %s\n", output_dir.c_str());
        }

        const bool write_per_genome_files = !output_dir.empty();
        if (write_per_genome_files) {
            std::error_code ec;
            std::filesystem::create_directories(output_dir, ec);
            per_genome = true; // implicit: --output-dir forces per-genome
        }

        // Validate each input exactly once.
        std::vector<std::string> inputs;
        inputs.reserve(fasta_paths.size());
        for (const auto& p : fasta_paths) {
            inputs.push_back(validate_input(p));
        }

        // Multi-input modes all go through the concurrent pipeline; pure
        // union mode takes the shared-sketch path, the others go through
        // sketch_per_genome_auto. Single-input drops to the I2 fast path
        // below. CUHLL_INTERNAL_FORCE_SEQUENTIAL forces the old sequential
        // pipeline for benchmarking.
        if (inputs.size() >= 2 &&
            std::getenv("CUHLL_INTERNAL_FORCE_SEQUENTIAL") == nullptr) {
            const bool pure_union = !per_genome && !write_per_genome_files;
            auto t0 = clk::now();
            std::uint64_t union_est = 0;

            if (pure_union) {
                union_est = cuhll::union_estimate_auto(
                    inputs, k, precision, canonical);
                stream_ms = ms_since(t0);
                std::cout << union_est << std::endl;
            } else {
                // Per-genome lines wanted. Use a tempdir if no --output-dir.
                std::optional<TempDir> td;
                std::string target_dir = output_dir;
                if (target_dir.empty()) {
                    const std::string base = resolve_tmpdir_base(tmpdir_cli);
                    td.emplace(base);
                    target_dir = td->path().string();
                    std::fprintf(stderr,
                        "[cuHLL] transient sketches: %s (auto-removed on exit; "
                        "override base with --tmpdir or $TMPDIR; persist via "
                        "--keep-sketches or --output-dir)\n",
                        target_dir.c_str());
                }
                union_est = cuhll::sketch_per_genome_auto(
                    inputs, target_dir, k, precision, canonical);
                stream_ms = ms_since(t0);
                std::cout << "UNION\t" << union_est << std::endl;
            }

            if (verbose) {
                std::fprintf(stderr,
                    "[cuHLL] k=%d precision=%d inputs=%zu (concurrent path, %s)\n"
                    "[cuHLL] timings (ms): concurrent=%.3f total=%.3f\n",
                    k, precision, fasta_paths.size(),
                    pure_union ? "shared sketch" : "per-genome",
                    stream_ms, ms_since(t_total));
            }
            return 0;
        }

        if (per_genome) {
            // One sketch per input; print "<path>\t<est>" per line; merge all
            // into a union sketch and print "UNION\t<est>" last.
            // With --output-dir, also write <output_dir>/<stem>.hll per input.
            cuhll::Sketch union_sketch(precision, canonical);
            auto t0 = clk::now();
            for (const auto& path : inputs) {
                cuhll::Sketch s(precision, canonical);
                cuhll::sketch_sequences_streaming(
                    s, std::vector<std::string>{path}, k,
                    static_cast<std::size_t>(chunk_mb));
                const std::uint64_t est = s.estimate();
                std::cout << path << '\t' << est << '\n';
                if (write_per_genome_files) {
                    const std::string stem = std::filesystem::path(path).stem().string();
                    const std::string out_path = output_dir + "/" + stem + ".hll";
                    if (verbose && std::filesystem::exists(out_path)) {
                        std::fprintf(stderr, "[cuHLL] overwriting existing %s\n",
                                     out_path.c_str());
                    }
                    cuhll::write_hll(out_path, s, k);
                }
                union_sketch.merge(s);
            }
            stream_ms = ms_since(t0);

            auto t2 = clk::now();
            const std::uint64_t union_est = union_sketch.estimate();
            estimate_ms = ms_since(t2);
            std::cout << "UNION\t" << union_est << std::endl;

            if (verbose) {
                std::fprintf(stderr,
                    "[cuHLL] k=%d precision=%d chunk_mb=%d inputs=%zu per_genome=1\n"
                    "[cuHLL] timings (ms): stream(parse+H2D+kernel+merge)=%.3f "
                    "estimate=%.3f total=%.3f\n",
                    k, precision, chunk_mb, fasta_paths.size(),
                    stream_ms, estimate_ms, ms_since(t_total));
            }
        } else if (inputs.size() == 1) {
            // I2 fast path: single FASTA input, union mode. Kick off the
            // FASTA parse in a worker thread so it runs concurrently with
            // the Sketch ctor's CUDA lazy init (~80 ms on a fresh process).
            // By the time the parse future resolves, the Sketch is ready
            // and we can go straight to the single-stream kernel without
            // paying for the 3-slot streaming pipeline's pinned alloc.
            const std::string& path = inputs[0];
            auto parse_fut = std::async(std::launch::async, [&]() {
                return cuhll::read_fasta_concat(path);
            });
            cuhll::Sketch sketch(precision, canonical); // CUDA init runs here
            std::string seq = parse_fut.get();

            auto t0 = clk::now();
            cuhll::sketch_sequence_single_stream(sketch, seq.data(), seq.size(), k);
            stream_ms = ms_since(t0);

            auto t2 = clk::now();
            const std::uint64_t est = sketch.estimate();
            estimate_ms = ms_since(t2);

            if (verbose) {
                std::fprintf(stderr,
                    "[cuHLL] k=%d precision=%d chunk_mb=%d inputs=%zu (single-fasta fast path)\n"
                    "[cuHLL] timings (ms): async_parse+sketch_ctor~max(overlap) "
                    "stream(single)=%.3f estimate=%.3f total=%.3f\n"
                    "[cuHLL] sketch_bytes=%zu\n",
                    k, precision, chunk_mb, fasta_paths.size(),
                    stream_ms, estimate_ms, ms_since(t_total),
                    sketch.sketch_bytes());
            }
            std::cout << est << std::endl;
            return 0;
        } else {
            // Union mode, multiple FASTAs: one sketch, one streaming pass.
            cuhll::Sketch sketch(precision, canonical);
            auto t0 = clk::now();
            cuhll::sketch_sequences_streaming(sketch, inputs, k,
                                              static_cast<std::size_t>(chunk_mb));
            stream_ms = ms_since(t0);

            auto t2 = clk::now();
            const std::uint64_t est = sketch.estimate();
            estimate_ms = ms_since(t2);

            if (verbose) {
                std::fprintf(stderr,
                    "[cuHLL] k=%d precision=%d chunk_mb=%d inputs=%zu\n"
                    "[cuHLL] timings (ms): stream(parse+H2D+kernel)=%.3f "
                    "estimate=%.3f total=%.3f\n"
                    "[cuHLL] sketch_bytes=%zu\n",
                    k, precision, chunk_mb, fasta_paths.size(),
                    stream_ms, estimate_ms, ms_since(t_total),
                    sketch.sketch_bytes());
            }
            std::cout << est << std::endl;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: error: %s\n", argv[0], e.what());
        return 1;
    }

    return 0;
}

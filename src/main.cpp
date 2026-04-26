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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
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

bool has_cb2_ext(const std::string& p) {
    return lower(std::filesystem::path(p).extension().string()) == ".cb2";
}

bool has_fasta_ext(const std::string& p) {
    const auto e = lower(std::filesystem::path(p).extension().string());
    return e == ".fasta" || e == ".fa" || e == ".fna";
}

bool has_gzipped_fasta_ext(const std::string& p) {
    std::filesystem::path pp(p);
    if (lower(pp.extension().string()) != ".gz") return false;
    const auto inner = lower(std::filesystem::path(pp.stem()).extension().string());
    return inner == ".fa" || inner == ".fna" || inner == ".fasta";
}

std::string cb2_colocated_path(const std::string& fasta_path) {
    std::filesystem::path p(fasta_path);
    const auto parent = p.parent_path().string();
    return (parent.empty() ? "." : parent) + "/" + p.stem().string() + ".cb2";
}

std::size_t file_size_bytes(const std::string& path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<std::size_t>(sz);
}

bool is_newer_than(const std::string& a, const std::string& b) {
    std::error_code eca, ecb;
    const auto ta = std::filesystem::last_write_time(a, eca);
    const auto tb = std::filesystem::last_write_time(b, ecb);
    if (eca || ecb) return false;
    return ta > tb;
}

// Classify one input path into a (backend, canonical-path) pair.
// Backend: 0 = FASTA, 1 = cb2. Throws for unsupported (e.g. gzipped) inputs.
// Rules:
//   * `--cb2` override: treat every input as .cb2 regardless of extension.
//   * *.cb2 -> cb2 backend.
//   * *.fasta/*.fa/*.fna -> FASTA backend. If the file is >= threshold_mb
//     AND a <stem>.cb2 exists alongside AND the .cb2 is not older than the
//     FASTA, use that .cb2 instead (auto-accelerate).
//   * *.fa.gz / *.fasta.gz / *.fna.gz -> explicit error (compressed not
//     yet supported).
std::pair<int, std::string> classify_input(const std::string& path,
                                            bool force_cb2,
                                            int threshold_mb) {
    if (force_cb2) return {1, path};
    if (has_gzipped_fasta_ext(path)) {
        throw std::runtime_error("compressed FASTA not yet supported; gunzip first: "
                                 + path);
    }
    if (has_cb2_ext(path)) return {1, path};
    if (has_fasta_ext(path)) {
        const std::size_t sz_mb = file_size_bytes(path) / (1024ULL * 1024ULL);
        if (threshold_mb > 0 && sz_mb >= static_cast<std::size_t>(threshold_mb)) {
            const std::string cob = cb2_colocated_path(path);
            if (std::filesystem::exists(cob) && is_newer_than(cob, path)) {
                return {1, cob};
            }
        }
        return {0, path};
    }
    throw std::runtime_error("unknown input extension (expected .fasta/.fa/.fna or .cb2): "
                             + path);
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --k <K> [options] <input> [<input> ...]\n"
        "\n"
        "Inputs can be a mix of:\n"
        "  *.fasta / *.fa / *.fna   (plain FASTA)\n"
        "  *.cb2                    (offline 2-bit packed; see cuhll_pack)\n"
        "\n"
        "Auto-select: when a FASTA's size is >= --cb2-threshold-mb and a\n"
        "<stem>.cb2 sibling exists and is newer than the FASTA, cuhll uses the\n"
        ".cb2 transparently. Otherwise the FASTA path is used.\n"
        "\n"
        "Options:\n"
        "  --k            k-mer length (required; %d <= k <= %d)\n"
        "  --precision    HLL precision (default %d; range %d..%d)\n"
        "  --chunk-mb     streaming chunk size in MiB (default %zu; min 1)\n"
        "  --cb2-threshold-mb N   auto-use colocated .cb2 at or above this size\n"
        "                         (default 100; env CUHLL_CB2_THRESHOLD_MB overrides)\n"
        "  --cb2          force .cb2 interpretation of every input\n"
        "  --canonical    (default) count canonical k-mers (min(fwd, rc))\n"
        "  --no-canonical count forward-strand k-mers only\n"
        "  --per-genome   emit one estimate per input plus a final UNION line\n"
        "  --output-dir D write one <stem>.hll per input to D (implies --per-genome)\n"
        "  --list F       read input paths from manifest F (one path per line)\n"
        "  --verbose      print per-stage timings + routing decisions to stderr\n"
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
    bool cb2_mode = false;             // legacy override: force cb2 for all inputs
    int  cb2_threshold_mb = 100;       // FASTA >= this triggers colocated .cb2 reuse
    std::string output_dir;            // empty = no per-genome sketch files
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
        } else if (arg == "--cb2") {
            cb2_mode = true;
        } else if (arg == "--cb2-threshold-mb") {
            cb2_threshold_mb = std::stoi(need_value(i, "--cb2-threshold-mb"));
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
        const bool write_per_genome_files = !output_dir.empty();
        if (write_per_genome_files) {
            std::error_code ec;
            std::filesystem::create_directories(output_dir, ec);
            per_genome = true; // implicit: --output-dir forces per-genome
        }

        // Env override for auto-cb2 threshold.
        if (const char* env = std::getenv("CUHLL_CB2_THRESHOLD_MB")) {
            try { cb2_threshold_mb = std::stoi(env); }
            catch (...) { /* ignore, keep CLI value */ }
        }

        // Classify each input exactly once; log the routing under --verbose.
        struct Route { int backend; std::string input_path; std::string display; };
        std::vector<Route> routed;
        routed.reserve(fasta_paths.size());
        for (const auto& p : fasta_paths) {
            auto [be, canon] = classify_input(p, cb2_mode, cb2_threshold_mb);
            routed.push_back({be, canon, p});
            if (verbose) {
                std::fprintf(stderr,
                    "[cuHLL] route %s -> %s via %s\n",
                    p.c_str(), canon.c_str(),
                    (be == 1 ? "cb2" : "fasta"));
            }
        }

        // Milestone (j): automatic concurrent per-genome path.
        // Fires when --output-dir is set, there are >= 2 inputs, every input
        // resolves to FASTA, and the internal escape-hatch env var is NOT
        // set. Produces .hll files byte-for-byte identical to the sequential
        // path and prints the same per-genome + UNION lines on stdout.
        if (write_per_genome_files && routed.size() >= 2 &&
            std::getenv("CUHLL_INTERNAL_FORCE_SEQUENTIAL") == nullptr) {
            bool all_fasta = true;
            std::vector<std::string> fa_paths;
            fa_paths.reserve(routed.size());
            for (const auto& r : routed) {
                if (r.backend != 0) { all_fasta = false; break; }
                fa_paths.push_back(r.input_path);
            }
            if (all_fasta) {
                auto t0 = clk::now();
                const std::uint64_t union_est = cuhll::sketch_per_genome_auto(
                    fa_paths, output_dir, k, precision, canonical);
                stream_ms = ms_since(t0);
                std::cout << "UNION\t" << union_est << std::endl;
                if (verbose) {
                    std::fprintf(stderr,
                        "[cuHLL] k=%d precision=%d inputs=%zu (concurrent path)\n"
                        "[cuHLL] timings (ms): concurrent=%.3f total=%.3f\n",
                        k, precision, fasta_paths.size(),
                        stream_ms, ms_since(t_total));
                }
                return 0;
            }
            // Else fall through to sequential path (mixed FASTA+cb2).
        }

        if (per_genome) {
            // One sketch per input; print "<path>\t<est>" per line; merge all
            // into a union sketch and print "UNION\t<est>" last.
            // With --output-dir, also write <output_dir>/<stem>.hll per input.
            cuhll::Sketch union_sketch(precision, canonical);
            auto t0 = clk::now();
            for (const auto& r : routed) {
                cuhll::Sketch s(precision, canonical);
                if (r.backend == 1) {
                    cuhll::sketch_sequences_cb2_streaming(
                        s, std::vector<std::string>{r.input_path}, k,
                        static_cast<std::size_t>(chunk_mb));
                } else {
                    cuhll::sketch_sequences_streaming(
                        s, std::vector<std::string>{r.input_path}, k,
                        static_cast<std::size_t>(chunk_mb));
                }
                const std::uint64_t est = s.estimate();
                std::cout << r.display << '\t' << est << '\n';
                if (write_per_genome_files) {
                    const std::string stem = std::filesystem::path(r.display).stem().string();
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
        } else if (routed.size() == 1 && routed[0].backend == 0) {
            // I2 fast path: single FASTA input, union mode. Kick off the
            // FASTA parse in a worker thread so it runs concurrently with
            // the Sketch ctor's CUDA lazy init (~80 ms on a fresh process).
            // By the time the parse future resolves, the Sketch is ready
            // and we can go straight to the single-stream kernel without
            // paying for the 3-slot streaming pipeline's pinned alloc.
            const std::string& path = routed[0].input_path;
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
            // Union mode: split routed inputs by backend, feed each pipeline once.
            std::vector<std::string> fa_bucket, cb_bucket;
            for (const auto& r : routed) {
                if (r.backend == 1) cb_bucket.push_back(r.input_path);
                else                fa_bucket.push_back(r.input_path);
            }
            cuhll::Sketch sketch(precision, canonical);
            auto t0 = clk::now();
            if (!fa_bucket.empty()) {
                cuhll::sketch_sequences_streaming(sketch, fa_bucket, k,
                                                  static_cast<std::size_t>(chunk_mb));
            }
            if (!cb_bucket.empty()) {
                cuhll::sketch_sequences_cb2_streaming(sketch, cb_bucket, k,
                                                      static_cast<std::size_t>(chunk_mb));
            }
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

// cuhll_pairwise — CLI wrapper for cuhll::pairwise.
//
// Loads .hll files produced by cuhll (--output-dir or --keep-sketches),
// converts their int32 register arrays to the 8-bit form the pairwise
// kernel expects, computes the full 16K-register Jaccard for every pair,
// and writes pairs with Jaccard >= threshold to a TSV.
//
// Usage:
//   cuhll_pairwise --sketches-dir DIR [options] > out.tsv
//
// Options:
//   --threshold T    only output pairs with Jaccard >= T  (default 0.5)
//   --output FILE    write to FILE instead of stdout
//   --verbose        print timings to stderr

#include "cuHLL/pairwise/pairwise.cuh"
#include "cuHLL/io/hll_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string sketches_dir;
    std::string output_path;
    float       threshold = 0.5f;
    bool        verbose   = false;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --sketches-dir DIR [options]\n\n"
        "Compute all-pairs HLL Jaccard from .hll sketches.\n\n"
        "Required:\n"
        "  --sketches-dir DIR   directory of .hll files (cuhll's --output-dir output)\n\n"
        "Options:\n"
        "  --threshold T        Jaccard cutoff for emitting a pair     (default 0.5)\n"
        "                       (output filter only; every pair is still computed)\n"
        "  --output FILE        write to FILE instead of stdout\n"
        "  --verbose            print timings to stderr\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                usage(argv[0]); std::exit(2);
            }
        };
        if (arg == "--sketches-dir") { need(1); a.sketches_dir = argv[++i]; }
        else if (arg == "--threshold") { need(1); a.threshold = std::atof(argv[++i]); }
        else if (arg == "--output")    { need(1); a.output_path = argv[++i]; }
        else if (arg == "--verbose")   { a.verbose = true; }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); usage(argv[0]); std::exit(2); }
    }
    if (a.sketches_dir.empty()) {
        std::fprintf(stderr, "error: --sketches-dir is required\n");
        usage(argv[0]); std::exit(2);
    }
    return a;
}

// Load all .hll files from a directory, validate they share the same
// precision, and pack their registers into a contiguous uint8 buffer.
struct LoadedSketches {
    int n = 0;
    std::vector<std::string> paths;
    std::vector<std::uint8_t> packed;   // n × kBytesPerSketch
};

LoadedSketches load_sketches(const std::string& dir, bool verbose) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    LoadedSketches L;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".hll") {
            L.paths.push_back(e.path().string());
        }
    }
    std::sort(L.paths.begin(), L.paths.end());   // deterministic order
    L.n = static_cast<int>(L.paths.size());
    if (L.n < 2) {
        throw std::runtime_error("need >= 2 .hll files in " + dir + " (found " +
                                 std::to_string(L.n) + ")");
    }

    L.packed.resize((std::size_t)L.n * cuhll::pairwise::kBytesPerSketch);
    std::vector<std::uint32_t> tmp(cuhll::pairwise::kRegisters);

    for (int i = 0; i < L.n; ++i) {
        auto hdr = cuhll::read_hll_header(L.paths[i]);
        if (hdr.precision_p != cuhll::pairwise::kPrecision) {
            throw std::runtime_error(
                L.paths[i] + ": precision=" + std::to_string(hdr.precision_p) +
                " but cuhll_pairwise requires p=" +
                std::to_string(cuhll::pairwise::kPrecision));
        }
        std::ifstream f(L.paths[i], std::ios::binary);
        f.seekg(sizeof(cuhll::HllFileHeader), std::ios::beg);
        f.read(reinterpret_cast<char*>(tmp.data()),
               cuhll::pairwise::kRegisters * sizeof(std::uint32_t));
        cuhll::pairwise::pack_registers(
            tmp.data(),
            L.packed.data() + (std::size_t)i * cuhll::pairwise::kBytesPerSketch);
    }

    if (verbose) {
        auto dt = std::chrono::duration<double>(clock::now() - t0).count();
        std::fprintf(stderr,
            "[cuhll_pairwise] loaded %d sketches from %s in %.2f s\n",
            L.n, dir.c_str(), dt);
    }
    return L;
}

} // anonymous namespace

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    Args a = parse_args(argc, argv);

    LoadedSketches sk;
    try {
        sk = load_sketches(a.sketches_dir, a.verbose);
    } catch (std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const std::int64_t n_pairs = (std::int64_t)sk.n * (sk.n - 1) / 2;
    if (a.verbose) {
        std::fprintf(stderr,
            "[cuhll_pairwise] N=%d  pairs=%lld  threshold=%.3f\n",
            sk.n, (long long)n_pairs, a.threshold);
    }

    auto t0 = clock::now();
    std::vector<float> jaccards;
    try {
        jaccards = cuhll::pairwise::compute_pairwise_jaccard_exact(
            sk.packed.data(), sk.n);
    } catch (std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    auto dt = std::chrono::duration<double>(clock::now() - t0).count();
    if (a.verbose) {
        std::fprintf(stderr,
            "[cuhll_pairwise] pairwise computed in %.3f s (%.2f M pairs/s)\n",
            dt, n_pairs / dt / 1e6);
    }

    // Stream pairs >= threshold to stdout / output file, TSV format:
    //   sketch_i_path \t sketch_j_path \t jaccard
    std::ostream* out_stream = &std::cout;
    std::ofstream out_file;
    if (!a.output_path.empty()) {
        out_file.open(a.output_path);
        if (!out_file) {
            std::fprintf(stderr, "error: cannot write %s\n", a.output_path.c_str());
            return 1;
        }
        out_stream = &out_file;
    }

    std::int64_t emitted = 0;
    for (int i = 0; i < sk.n; ++i) {
        std::int64_t base = (std::int64_t)i * (2*(std::int64_t)sk.n - i - 1) / 2;
        for (int j = i + 1; j < sk.n; ++j) {
            std::int64_t k_out = base + (j - i - 1);
            float jac = jaccards[k_out];
            if (jac >= a.threshold) {
                (*out_stream) << sk.paths[i] << '\t' << sk.paths[j]
                              << '\t' << jac << '\n';
                ++emitted;
            }
        }
    }

    if (a.verbose) {
        std::fprintf(stderr,
            "[cuhll_pairwise] emitted %lld pairs above threshold %.3f\n",
            (long long)emitted, a.threshold);
    }
    return 0;
}

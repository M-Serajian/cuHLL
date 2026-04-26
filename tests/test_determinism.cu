// =============================================================================
// test_determinism.cu
//
// Formalizes the determinism invariant observed informally in milestone (d):
//   1. Two runs with the same k, chunk_mb, and input produce bit-identical
//      estimates (same-run repeatability).
//   2. Varying chunk_mb across 16 / 32 / 64 MiB does not change the estimate
//      (cross-chunk-size invariance — proves the ring overlap handling is
//      boundary-agnostic).
// =============================================================================

#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kPrecision = 14;
constexpr int kK         = 31;

std::uint64_t run_one(int chunk_mb, const std::string& fasta) {
    cuhll::Sketch sketch(kPrecision);
    cuhll::sketch_sequences_streaming(
        sketch,
        std::vector<std::string>{fasta},
        kK,
        static_cast<std::size_t>(chunk_mb));
    return sketch.estimate();
}

} // namespace

int main() {
    const std::string fasta =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_determinism");
    if (fasta.empty()) return 0;

    std::uint64_t e32_a = 0;
    std::uint64_t e32_b = 0;
    std::uint64_t e16   = 0;
    std::uint64_t e64   = 0;

    try {
        e32_a = run_one(32, fasta);
        e32_b = run_one(32, fasta);
        e16   = run_one(16, fasta);
        e64   = run_one(64, fasta);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_determinism: runtime error: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr,
        "[test_determinism] k=%d  fasta=%s\n"
        "  run1(chunk=32) = %llu\n"
        "  run2(chunk=32) = %llu\n"
        "  run (chunk=16) = %llu\n"
        "  run (chunk=64) = %llu\n",
        kK, fasta.c_str(),
        static_cast<unsigned long long>(e32_a),
        static_cast<unsigned long long>(e32_b),
        static_cast<unsigned long long>(e16),
        static_cast<unsigned long long>(e64));

    if (e32_a != e32_b) {
        std::fprintf(stderr,
            "[FAIL] test_determinism: same-chunk-size runs differ: "
            "run1=%llu run2=%llu\n",
            static_cast<unsigned long long>(e32_a),
            static_cast<unsigned long long>(e32_b));
        return 1;
    }
    if (e32_a != e16 || e32_a != e64) {
        std::fprintf(stderr,
            "[FAIL] test_determinism: cross-chunk-size estimates differ: "
            "chunk16=%llu chunk32=%llu chunk64=%llu\n",
            static_cast<unsigned long long>(e16),
            static_cast<unsigned long long>(e32_a),
            static_cast<unsigned long long>(e64));
        return 1;
    }

    std::fprintf(stderr, "[PASS] test_determinism\n");
    return 0;
}

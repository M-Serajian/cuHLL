// =============================================================================
// test_merge.cu
//
// Validates cuco::hyperloglog::merge as surfaced through cuhll::Sketch::merge.
//
// Setup:
//   - Load chr19/1.fasta via read_fasta_concat (filtered byte buffer).
//   - Split the filtered buffer at offset `half`.  A = [0, half), B = [half, end).
//   - Sketch A, sketch B, sketch AB_direct (the full buffer) independently.
//
// Note on the split: A∪B (set-wise) is MISSING exactly k-1 = 30 k-mers that
// span the split boundary compared to AB_direct. For a ~48M cardinality, 30
// extra k-mers is below HLL's resolution.
//
// Assertions:
//   1. merged = A.merge(B); |merged.estimate() - AB_direct.estimate()| is
//      within HLL bounds (generous 3 * sqrt(2) * sigma, since the two
//      estimates carry independent HLL noise).
//   2. merged.estimate() >= max(A.estimate(), B.estimate()) — merge can only
//      grow cardinality, never shrink.
//   3. B.estimate() is unchanged by the merge — cuco's merge is read-only on
//      its argument.
// =============================================================================

#include "cuHLL/io/fasta.hpp"
#include "cuHLL/pipeline/pipeline.hpp"
#include "cuHLL/sketch/sketch.hpp"
#include "test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr int    kPrecision = 14;
constexpr int    kK         = 31;
constexpr double kMRegs     = static_cast<double>(1 << kPrecision); // 16384

} // namespace

int main() {
    const std::string fasta =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_merge");
    if (fasta.empty()) return 0;

    std::string filtered;
    try {
        filtered = cuhll::read_fasta_concat(fasta);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_merge: cannot read %s: %s\n",
                     fasta.c_str(), e.what());
        return 1;
    }
    if (filtered.size() < 1'000'000) {
        std::fprintf(stderr,
            "[FAIL] test_merge: %s filtered buffer too small (%zu bytes)\n",
            fasta.c_str(), filtered.size());
        return 1;
    }

    const std::size_t half = filtered.size() / 2;

    cuhll::Sketch A(kPrecision);
    cuhll::Sketch B(kPrecision);
    cuhll::Sketch AB(kPrecision);

    cuhll::sketch_sequence_single_stream(A,  filtered.data(),        half,                       kK);
    cuhll::sketch_sequence_single_stream(B,  filtered.data() + half, filtered.size() - half,     kK);
    cuhll::sketch_sequence_single_stream(AB, filtered.data(),        filtered.size(),            kK);

    const std::uint64_t a_est          = A.estimate();
    const std::uint64_t b_est_before   = B.estimate();
    const std::uint64_t ab_direct_est  = AB.estimate();

    A.merge(B);

    const std::uint64_t merged_est     = A.estimate();
    const std::uint64_t b_est_after    = B.estimate();

    // Bounds.
    const double sigma_rel = 1.04 / std::sqrt(kMRegs);                 // ~0.00813
    // Two independent HLL estimates compared to each other carry noise of
    // sqrt(2) times a single estimate's sigma. Use 3*sqrt(2)*sigma as the
    // one-sided tolerance. For 48M, that's ~3.45% ≈ 1.66M.
    const double diff_bound_rel = 3.0 * std::sqrt(2.0) * sigma_rel;
    const double diff_bound_abs = diff_bound_rel *
        std::max(static_cast<double>(merged_est), static_cast<double>(ab_direct_est));
    const double diff_abs = std::fabs(static_cast<double>(merged_est) -
                                      static_cast<double>(ab_direct_est));

    std::fprintf(stderr,
        "[test_merge] k=%d  fasta=%s  half=%zu  total=%zu\n"
        "  A.est                = %llu\n"
        "  B.est (before merge) = %llu\n"
        "  AB_direct.est        = %llu\n"
        "  merged (A<-B).est    = %llu\n"
        "  B.est (after merge)  = %llu\n"
        "  |merged - direct|    = %.0f   (3*sqrt(2)*sigma bound = %.0f, rel %.4f)\n",
        kK, fasta.c_str(), half, filtered.size(),
        static_cast<unsigned long long>(a_est),
        static_cast<unsigned long long>(b_est_before),
        static_cast<unsigned long long>(ab_direct_est),
        static_cast<unsigned long long>(merged_est),
        static_cast<unsigned long long>(b_est_after),
        diff_abs, diff_bound_abs, diff_bound_rel);

    int fail = 0;

    // Assertion 1: merged ≈ AB_direct.
    if (diff_abs > diff_bound_abs) {
        std::fprintf(stderr,
            "[FAIL] test_merge: merged estimate deviates from AB_direct by "
            "%.0f, exceeding the 3*sqrt(2)*sigma bound of %.0f.\n",
            diff_abs, diff_bound_abs);
        fail = 1;
    }

    // Assertion 2: merged is monotone w.r.t. A and B individually.
    if (merged_est < a_est || merged_est < b_est_before) {
        std::fprintf(stderr,
            "[FAIL] test_merge: merged=%llu < max(A=%llu, B=%llu)\n",
            static_cast<unsigned long long>(merged_est),
            static_cast<unsigned long long>(a_est),
            static_cast<unsigned long long>(b_est_before));
        fail = 1;
    }

    // Assertion 3: merge is non-destructive on B.
    if (b_est_before != b_est_after) {
        std::fprintf(stderr,
            "[FAIL] test_merge: merge mutated B's estimate: %llu -> %llu\n",
            static_cast<unsigned long long>(b_est_before),
            static_cast<unsigned long long>(b_est_after));
        fail = 1;
    }

    if (fail) return 1;
    std::fprintf(stderr, "[PASS] test_merge\n");
    return 0;
}

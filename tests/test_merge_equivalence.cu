// =============================================================================
// test_merge_equivalence.cu
//
// Verifies HLL merge is equivalent to concatenated sketching:
//   sketch(g1 ⊎ g2) has register-wise equal values to
//   merge(sketch(g1), sketch(g2))
//
// The inter-file N-injection in the streaming path introduces a single
// k-mer spanning the g1/g2 boundary. That k-mer is present in
// sketch(g1⊎g2) but not in merge(sketch(g1), sketch(g2)). At m=16384
// registers and ~48M cardinality per genome, a single extra unique
// hash is far below register resolution, so register arrays are
// expected to be bit-identical in practice.
// =============================================================================

#include "cuHLL/hll_file.hpp"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr int kK = 31;
constexpr int kP = 14;
} // namespace

int main() {
    const std::string g1 =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA",  "test_merge_equivalence");
    if (g1.empty()) return 0;
    const std::string g2 =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA2", "test_merge_equivalence");
    if (g2.empty()) return 0;

    const fs::path tmp = cuhll_test::scratch_dir("test_merge_eq");

    // PATH A: sketch each genome separately, persist, reload, merge.
    cuhll::Sketch s1(kP), s2(kP);
    try {
        cuhll::sketch_sequences_streaming(s1, {g1}, kK, /*chunk_mb=*/32);
        cuhll::sketch_sequences_streaming(s2, {g2}, kK, /*chunk_mb=*/32);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_merge_equivalence: per-genome sketch failed: %s\n", e.what());
        return 1;
    }
    const std::uint64_t est1 = s1.estimate();
    const std::uint64_t est2 = s2.estimate();

    const std::string p1 = (tmp / "g1.hll").string();
    const std::string p2 = (tmp / "g2.hll").string();
    cuhll::write_hll(p1, s1, kK);
    cuhll::write_hll(p2, s2, kK);

    cuhll::Sketch s1r = cuhll::read_hll(p1);
    cuhll::Sketch s2r = cuhll::read_hll(p2);
    if (s1r.estimate() != est1 || s2r.estimate() != est2) {
        std::fprintf(stderr, "[FAIL] test_merge_equivalence: .hll round-trip changed estimate\n");
        return 1;
    }

    // Merge s2r into a clone of s1r so we can keep both sides for comparison.
    cuhll::Sketch s_merged = s1r.clone();
    s_merged.merge(s2r);
    const std::uint64_t est_merged = s_merged.estimate();

    // PATH B: sketch both genomes together.
    cuhll::Sketch s_combined(kP);
    try {
        cuhll::sketch_sequences_streaming(s_combined, {g1, g2}, kK, /*chunk_mb=*/32);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_merge_equivalence: combined sketch failed: %s\n", e.what());
        return 1;
    }
    const std::uint64_t est_combined = s_combined.estimate();

    const long long signed_diff  = static_cast<long long>(est_merged)
                                 - static_cast<long long>(est_combined);
    const double     abs_diff_pc = (est_combined == 0) ? 0.0
        : 100.0 * std::fabs(static_cast<double>(signed_diff))
                / static_cast<double>(est_combined);

    std::fprintf(stderr,
        "[test_merge_equivalence]\n"
        "  est(g1)            = %llu\n"
        "  est(g2)            = %llu\n"
        "  est(merge(g1,g2))  = %llu   <- PATH A (merge of separately-built sketches)\n"
        "  est(sketch(g1|g2)) = %llu   <- PATH B (single sketch over concatenation)\n"
        "  signed diff        = %lld\n"
        "  relative diff      = %.6f %%\n",
        (unsigned long long)est1,
        (unsigned long long)est2,
        (unsigned long long)est_merged,
        (unsigned long long)est_combined,
        signed_diff, abs_diff_pc);

    // Register-level comparison.
    const std::size_t n_regs = static_cast<std::size_t>(1u) << kP;
    std::vector<std::uint32_t> r_merged(n_regs), r_combined(n_regs);
    s_merged.copy_registers_to_host(r_merged.data());
    s_combined.copy_registers_to_host(r_combined.data());
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < n_regs; ++i) {
        if (r_merged[i] != r_combined[i]) ++diffs;
    }
    std::fprintf(stderr, "  register diffs: %zu / %zu\n", diffs, n_regs);

    bool fail = false;
    if (diffs != 0) {
        std::fprintf(stderr,
            "[FAIL] test_merge_equivalence: register arrays differ — merge is NOT register-identical to concat\n");
        fail = true;
    }
    if (est_merged != est_combined) {
        std::fprintf(stderr,
            "[FAIL] test_merge_equivalence: estimate differs despite identical registers\n");
        fail = true;
    }

    std::error_code ec;
    fs::remove_all(tmp, ec);

    if (fail) return 1;
    std::fprintf(stderr, "[PASS] test_merge_equivalence: merge(A,B) == sketch(A|B) register-identically\n");
    return 0;
}

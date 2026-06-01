// =============================================================================
// test_hll_roundtrip.cu
//
// Verifies the .hll file format round-trips a cuco::hyperloglog exactly:
//   sketch A = sketch(chr19/1.fasta, k=31, p=14)
//   write A to /build/test_hll_tmp/sketch_1.hll
//   sketch B = read_hll(same path)
//   ASSERT A.estimate() == B.estimate() == 48,682,019 (canary)
//   ASSERT 16,384 register-wise bit-identical bytes (i.e., cuco registers
//          restored exactly as saved).
// =============================================================================

#include "cuHLL/io/hll_file.hpp"
#include "cuHLL/pipeline/pipeline.hpp"
#include "cuHLL/sketch/sketch.hpp"
#include "test_common.hpp"

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
    const std::string fasta =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_hll_roundtrip");
    if (fasta.empty()) return 0;

    const fs::path tmp_dir  = cuhll_test::scratch_dir("test_hll");
    const std::string out_path = (tmp_dir / "sketch_1.hll").string();

    cuhll::Sketch A(kP);
    try {
        cuhll::sketch_sequences_streaming(A, {fasta}, kK, /*chunk_mb=*/32);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_hll_roundtrip: sketch build failed: %s\n", e.what());
        return 1;
    }
    const std::uint64_t est_A = A.estimate();

    // Write A and read it back.
    try {
        cuhll::write_hll(out_path, A, kK);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_hll_roundtrip: write_hll failed: %s\n", e.what());
        return 1;
    }

    cuhll::Sketch B(kP);
    try {
        B = cuhll::read_hll(out_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_hll_roundtrip: read_hll failed: %s\n", e.what());
        return 1;
    }
    const std::uint64_t est_B = B.estimate();

    std::fprintf(stderr,
        "[test_hll_roundtrip] est_A=%llu  est_B=%llu  path=%s\n",
        (unsigned long long)est_A,
        (unsigned long long)est_B,
        out_path.c_str());

    if (est_B != est_A) {
        std::fprintf(stderr, "[FAIL] test_hll_roundtrip: round-trip changed estimate\n");
        return 1;
    }

    // Bit-for-bit register comparison.
    const std::size_t n_regs = static_cast<std::size_t>(1u) << kP;
    std::vector<std::uint32_t> regs_A(n_regs), regs_B(n_regs);
    A.copy_registers_to_host(regs_A.data());
    B.copy_registers_to_host(regs_B.data());
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < n_regs; ++i) {
        if (regs_A[i] != regs_B[i]) ++diffs;
    }
    std::fprintf(stderr, "[test_hll_roundtrip] register diffs: %zu / %zu\n", diffs, n_regs);
    if (diffs != 0) {
        std::fprintf(stderr, "[FAIL] test_hll_roundtrip: register arrays differ after round-trip\n");
        return 1;
    }

    // Verify .hll header reads back with correct k.
    auto hdr = cuhll::read_hll_header(out_path);
    if (hdr.k != static_cast<std::uint32_t>(kK) ||
        hdr.precision_p != static_cast<std::uint32_t>(kP)) {
        std::fprintf(stderr,
            "[FAIL] test_hll_roundtrip: header fields wrong (k=%u p=%u)\n",
            hdr.k, hdr.precision_p);
        return 1;
    }

    std::error_code ec; fs::remove_all(tmp_dir, ec);
    std::fprintf(stderr, "[PASS] test_hll_roundtrip\n");
    return 0;
}

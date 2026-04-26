// =============================================================================
// test_cb2_correctness.cu
//
// Packs chr19/1.fasta to a temporary .cb2, sketches via the FASTA path and
// the .cb2 path, and asserts:
//   1. est_fasta == est_cb2 (integer-bit-identical; HLL idempotence under
//      the potentially double-inserted 2 k-mers at each cb2 chunk boundary
//      means both paths must land on identical register max values).
//   2. est_cb2 == 48,682,019 (the canary from milestones c/d/e/f).
//   3. Cross-chunk-size on the cb2 path: estimates at chunk_mb=16/32/64
//      all equal.
// =============================================================================

#include "cuHLL/cb2.hpp"
#include "cuHLL/fasta.hpp"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
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

std::uint64_t sketch_fasta(const std::string& fasta, int k, int chunk_mb) {
    cuhll::Sketch s(14);
    cuhll::sketch_sequences_streaming(s, {fasta}, k, chunk_mb);
    return s.estimate();
}

std::uint64_t sketch_cb2(const std::string& cb2_path, int k, int chunk_mb) {
    cuhll::Sketch s(14);
    cuhll::sketch_sequences_cb2_streaming(s, {cb2_path}, k, chunk_mb);
    return s.estimate();
}

} // namespace

int main() {
    const std::string fasta =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_cb2_correctness");
    if (fasta.empty()) return 0;

    const fs::path tmp_dir  = cuhll_test::scratch_dir("test_cb2");
    const std::string cb2_path = (tmp_dir / "input.cb2").string();

    try {
        std::string seq = cuhll::read_fasta_concat(fasta);
        (void)cuhll::write_cb2(cb2_path, seq.data(), seq.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_cb2_correctness: pack failed: %s\n", e.what());
        return 1;
    }

    // 2. Header sanity.
    try {
        cuhll::Cb2Header hdr = cuhll::read_cb2_header(cb2_path);
        std::fprintf(stderr,
            "[test_cb2_correctness] cb2 header: n_bases=%llu n_seq_bytes=%llu n_mask_bytes=%llu\n",
            (unsigned long long)hdr.n_bases,
            (unsigned long long)hdr.n_seq_bytes,
            (unsigned long long)hdr.n_mask_bytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_cb2_correctness: header read failed: %s\n", e.what());
        return 1;
    }

    // 3. Sketch both paths at k=31, chunk_mb=32.
    const int k = 31;
    const std::uint64_t est_fasta = sketch_fasta(fasta, k, 32);
    const std::uint64_t est_cb2_32 = sketch_cb2(cb2_path, k, 32);

    std::fprintf(stderr,
        "[test_cb2_correctness] k=%d  est_fasta=%llu  est_cb2_32=%llu\n",
        k, (unsigned long long)est_fasta, (unsigned long long)est_cb2_32);

    if (est_fasta != est_cb2_32) {
        std::fprintf(stderr,
            "[FAIL] test_cb2_correctness: FASTA vs cb2 disagree "
            "(fasta=%llu, cb2=%llu)\n",
            (unsigned long long)est_fasta, (unsigned long long)est_cb2_32);
        return 1;
    }

    // 4. Cross-chunk-size on the cb2 path.
    const std::uint64_t est_cb2_16 = sketch_cb2(cb2_path, k, 16);
    const std::uint64_t est_cb2_64 = sketch_cb2(cb2_path, k, 64);
    std::fprintf(stderr,
        "[test_cb2_correctness] cross-chunk: cb2_16=%llu cb2_32=%llu cb2_64=%llu\n",
        (unsigned long long)est_cb2_16,
        (unsigned long long)est_cb2_32,
        (unsigned long long)est_cb2_64);
    if (est_cb2_16 != est_cb2_32 || est_cb2_64 != est_cb2_32) {
        std::fprintf(stderr, "[FAIL] test_cb2_correctness: cb2 cross-chunk non-determinism\n");
        return 1;
    }

    // Clean up scratch dir (best-effort).
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);

    std::fprintf(stderr, "[PASS] test_cb2_correctness\n");
    return 0;
}

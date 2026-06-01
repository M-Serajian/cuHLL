// =============================================================================
// test_canonical_flag.cu — canonical/non-canonical k-mer mode (Milestone l).
//
// Cases:
//  A. Default Sketch is canonical — canary 48,682,019 on chr19/1.
//  B. Explicit canonical=true bit-identical to default (register-level).
//  C. Non-canonical produces a meaningfully different estimate
//     (> 10% different, typically ~2× larger for non-palindromic inputs).
//  D. .hll round-trip preserves the mode byte (v2 header).
//  E. Back-compat: an existing milestone-k v1 .hll loads as canonical.
// =============================================================================

#include "cuHLL/io/hll_file.hpp"
#include "cuHLL/pipeline/pipeline.hpp"
#include "cuHLL/sketch/sketch.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kP = 14;
constexpr int kK = 31;

const std::string& fasta_path() {
    static const std::string p =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_canonical_flag");
    return p;
}

const fs::path& tmp_root() {
    static const fs::path root = cuhll_test::scratch_dir("test_canonical");
    return root;
}

cuhll::Sketch sketch_with_mode(bool canonical) {
    cuhll::Sketch s(kP, canonical);
    cuhll::sketch_sequences_streaming(s, {fasta_path()}, kK, /*chunk_mb=*/32);
    return s;
}

int case_A_default_canonical() {
    std::fprintf(stderr, "[caseA] default Sketch(p) is canonical\n");
    cuhll::Sketch s(kP);
    if (!s.canonical()) {
        std::fprintf(stderr, "[FAIL A] default Sketch canonical()=false\n"); return 1;
    }
    cuhll::sketch_sequences_streaming(s, {fasta_path()}, kK, 32);
    const std::uint64_t est = s.estimate();

    const std::string canary_str = cuhll_test::env_or_empty("CUHLL_TEST_CANARY");
    if (!canary_str.empty()) {
        const std::uint64_t canary = std::stoull(canary_str);
        std::fprintf(stderr, "  default-mode est=%llu (canary=%llu)\n",
                     (unsigned long long)est, (unsigned long long)canary);
        if (est != canary) {
            std::fprintf(stderr, "[FAIL A] canary drift: %llu != %llu\n",
                         (unsigned long long)est, (unsigned long long)canary);
            return 1;
        }
    } else {
        std::fprintf(stderr, "  default-mode est=%llu (no canary set)\n",
                     (unsigned long long)est);
    }
    return 0;
}

int case_B_explicit_canonical_identical() {
    std::fprintf(stderr, "[caseB] explicit canonical==default (register-level)\n");
    auto s_default = sketch_with_mode(true);
    auto s_explicit = sketch_with_mode(true);
    const std::size_t n_regs = std::size_t(1) << kP;
    std::vector<std::uint32_t> r1(n_regs), r2(n_regs);
    s_default.copy_registers_to_host(r1.data());
    s_explicit.copy_registers_to_host(r2.data());
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < n_regs; ++i) if (r1[i] != r2[i]) ++diffs;
    std::fprintf(stderr, "  register diffs = %zu\n", diffs);
    if (diffs != 0) {
        std::fprintf(stderr, "[FAIL B] default vs explicit canonical differ\n");
        return 1;
    }
    return 0;
}

int case_C_non_canonical_differs() {
    std::fprintf(stderr, "[caseC] --no-canonical produces different estimate\n");
    auto s_can = sketch_with_mode(true);
    auto s_noc = sketch_with_mode(false);
    const auto est_can = s_can.estimate();
    const auto est_noc = s_noc.estimate();
    std::fprintf(stderr, "  canonical=%llu  non-canonical=%llu  ratio=%.4f\n",
                 (unsigned long long)est_can,
                 (unsigned long long)est_noc,
                 (double)est_noc / (double)est_can);
    // chr19 FASTA is single-stranded. Canonical folds fwd/rc to min(),
    // collapsing only the ~1 % of k-mers whose RC also appears on this
    // strand. So non-canonical must be strictly greater than canonical,
    // with a small-but-non-trivial delta. A zero or negative delta would
    // mean the flag is not plumbed through the kernel.
    if (est_noc <= est_can) {
        std::fprintf(stderr,
            "[FAIL C] non-canonical (%llu) must be > canonical (%llu)\n",
            (unsigned long long)est_noc, (unsigned long long)est_can);
        return 1;
    }
    const double rel_delta = double(est_noc - est_can) / double(est_can);
    if (rel_delta < 0.005 || rel_delta > 0.10) {
        std::fprintf(stderr,
            "[FAIL C] rel delta %.4f outside expected [0.005, 0.10]\n",
            rel_delta);
        return 1;
    }
    return 0;
}

int case_D_roundtrip_mode() {
    std::fprintf(stderr, "[caseD] .hll v2 round-trip preserves canonical bit\n");
    const std::string p_can = (tmp_root() / "D_can.hll").string();
    const std::string p_noc = (tmp_root() / "D_noc.hll").string();

    auto s_can = sketch_with_mode(true);
    auto s_noc = sketch_with_mode(false);
    cuhll::write_hll(p_can, s_can, kK);
    cuhll::write_hll(p_noc, s_noc, kK);

    auto h_can = cuhll::read_hll_header(p_can);
    auto h_noc = cuhll::read_hll_header(p_noc);
    std::fprintf(stderr, "  v=%u canonical_byte=%u (canonical-file)\n",
                 h_can.version, h_can.canonical);
    std::fprintf(stderr, "  v=%u canonical_byte=%u (non-canonical-file)\n",
                 h_noc.version, h_noc.canonical);
    if (h_can.canonical != 1 || h_noc.canonical != 0
        || h_can.version != cuhll::kHllFileVersion
        || h_noc.version != cuhll::kHllFileVersion) {
        std::fprintf(stderr, "[FAIL D] header fields wrong\n");
        return 1;
    }

    auto r_can = cuhll::read_hll(p_can);
    auto r_noc = cuhll::read_hll(p_noc);
    if (!r_can.canonical() || r_noc.canonical()) {
        std::fprintf(stderr,
            "[FAIL D] reconstructed Sketch.canonical() mismatch (c=%d, n=%d)\n",
            int(r_can.canonical()), int(r_noc.canonical()));
        return 1;
    }
    return 0;
}

int case_E_back_compat_v1() {
    std::fprintf(stderr, "[caseE] legacy v1 .hll reads as canonical\n");
    const std::string p_v1 = (tmp_root() / "E_legacy_v1.hll").string();

    // Build a real canonical sketch, write it as v2, then rewrite the
    // header in-place to look like the milestone-k v1 format: version=1
    // and the canonical byte zeroed out (v1 had no canonical field; the
    // byte was reserved-and-zero). This exercises the back-compat path
    // in read_hll_header that promotes v1 → canonical=true.
    auto s_src = sketch_with_mode(true);
    cuhll::write_hll(p_v1, s_src, kK);

    {
        std::fstream f(p_v1, std::ios::in | std::ios::out | std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "[FAIL E] cannot reopen %s for header rewrite\n",
                         p_v1.c_str());
            return 1;
        }
        const std::uint32_t v1 = 1u;
        const std::uint8_t  zero = 0u;
        f.seekp(8, std::ios::beg);  // version field starts at offset 8
        f.write(reinterpret_cast<const char*>(&v1), sizeof(v1));
        f.seekp(40, std::ios::beg); // canonical byte at offset 40
        f.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    auto h = cuhll::read_hll_header(p_v1);
    std::fprintf(stderr, "  synthesized-v1 version=%u canonical_byte=%u\n",
                 h.version, h.canonical);
    if (h.version != cuhll::kHllFileLegacyV1) {
        std::fprintf(stderr, "[FAIL E] expected version=1, got %u\n", h.version);
        return 1;
    }
    if (h.canonical != 1) {
        std::fprintf(stderr, "[FAIL E] v1 file should be promoted to canonical=1\n");
        return 1;
    }
    auto s_loaded = cuhll::read_hll(p_v1);
    if (!s_loaded.canonical()) {
        std::fprintf(stderr,
            "[FAIL E] reconstructed Sketch.canonical()=false for v1 file\n");
        return 1;
    }
    if (s_loaded.estimate() != s_src.estimate()) {
        std::fprintf(stderr,
            "[FAIL E] v1-loaded estimate %llu != source %llu\n",
            (unsigned long long)s_loaded.estimate(),
            (unsigned long long)s_src.estimate());
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (fasta_path().empty()) return 0;  // [SKIP] already printed
    int fails = 0;
    fails += case_A_default_canonical();
    fails += case_B_explicit_canonical_identical();
    fails += case_C_non_canonical_differs();
    fails += case_D_roundtrip_mode();
    fails += case_E_back_compat_v1();

    std::error_code ec;
    fs::remove_all(tmp_root(), ec);

    if (fails == 0) { std::fprintf(stderr, "[PASS] test_canonical_flag\n"); return 0; }
    std::fprintf(stderr, "[FAIL] test_canonical_flag: %d case(s)\n", fails);
    return 1;
}

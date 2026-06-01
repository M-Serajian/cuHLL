// =============================================================================
// test_tiny_exact.cu
//
// Ground truth: std::unordered_set<uint64_t> of canonical 2-bit-packed k-mers.
// This is independent of ntHash — it uses direct 2-bit encoding (A=0, C=1,
// G=2, T=3) and the "min(fwd_2bit, rc_2bit)" canonical choice. Any bug in
// cuHLL's ntHash + HLL pipeline that shifts cardinality will be caught.
//
// Synthetic sequence layout (deterministic, seed=42):
//   - 500 bases of pseudo-random ACGT
//   - 3 'N's inserted at known positions to exercise window reset
//   - 100 more pseudo-random ACGT
//   - 20-base DNA palindrome ("GATCAGGCATATGCCTGATC") — palindromes are a
//     canonical-hash edge case (fwd == rc, canonical hashes are small-ish)
//   - 500 more pseudo-random ACGT
//   - 'N' separator
//   - ~2 KB slice from chr19/1.fasta (real base distribution)
//   - 500 more pseudo-random ACGT
//
// Assertion: |cuHLL_est - exact| <= 3 * 1.04 / sqrt(m) * exact
//            for m = 16384 (precision=14) → ~2.44%
// Run at both k=21 and k=31.
// =============================================================================

#include "cuHLL/common/common.hpp"
#include "test_common.hpp"
#include "cuHLL/io/fasta.hpp"
#include "cuHLL/pipeline/pipeline.hpp"
#include "cuHLL/sketch/sketch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

constexpr int kPrecision = cuhll::kDefaultPrecision;           // 14
constexpr double kMRegs  = static_cast<double>(1 << kPrecision); // 16384

inline unsigned base2bit(char c) {
    switch (c) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:            return 4u;
    }
}

// Compute canonical 2-bit-packed k-mer for seq[0..k-1]. Returns false if any
// base is non-ACGT (caller should skip the window).
bool canonical_2bit(const char* seq, int k, std::uint64_t& out) {
    std::uint64_t fwd = 0;
    std::uint64_t rc  = 0;
    for (int i = 0; i < k; ++i) {
        unsigned c = base2bit(seq[i]);
        if (c > 3u) return false;
        fwd = (fwd << 2) | static_cast<std::uint64_t>(c);
        rc |= static_cast<std::uint64_t>(c ^ 3u) << (2 * i);
    }
    out = fwd < rc ? fwd : rc;
    return true;
}

std::size_t exact_distinct_canonical(const std::string& seq, int k) {
    std::unordered_set<std::uint64_t> s;
    s.reserve(seq.size());
    const std::size_t n = seq.size();
    if (n < static_cast<std::size_t>(k)) return 0;
    for (std::size_t i = 0; i + static_cast<std::size_t>(k) <= n; ++i) {
        std::uint64_t h = 0;
        if (canonical_2bit(seq.data() + i, k, h)) {
            s.insert(h);
        }
    }
    return s.size();
}

std::string build_synthetic(const std::string& chr19_slice) {
    std::mt19937 rng(42);
    const char alpha[] = "ACGT";
    auto rand_bases = [&](std::size_t n) {
        std::string r;
        r.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            r.push_back(alpha[rng() & 3u]);
        }
        return r;
    };

    std::string s;
    std::string block = rand_bases(500);
    block[100] = 'N';
    block[250] = 'N';
    block[400] = 'N';
    s += block;
    s += rand_bases(100);
    s += "GATCAGGCATATGCCTGATC"; // 20-base palindrome
    s += rand_bases(500);
    s.push_back('N');
    s += chr19_slice;
    s += rand_bases(500);
    return s;
}

int run_one_k(const std::string& seq, int k) {
    const std::size_t exact = exact_distinct_canonical(seq, k);

    cuhll::Sketch sketch(kPrecision);
    cuhll::sketch_sequence_single_stream(sketch, seq.data(), seq.size(), k);
    const std::uint64_t est = sketch.estimate();

    const double bound_rel = 3.0 * 1.04 / std::sqrt(kMRegs);   // ~0.02438
    const double bound_abs = bound_rel * static_cast<double>(exact);
    const double err = std::fabs(static_cast<double>(est) - static_cast<double>(exact));

    std::fprintf(stderr,
        "[test_tiny_exact] k=%d  seq_len=%zu  exact=%zu  cuHLL_est=%llu  "
        "err=%.0f  3sigma_bound=%.1f  (rel %.4f)\n",
        k, seq.size(), exact, static_cast<unsigned long long>(est),
        err, bound_abs, (exact ? err / static_cast<double>(exact) : 0.0));

    if (exact < 100) {
        std::fprintf(stderr, "[FAIL] test_tiny_exact: k=%d exact=%zu < 100 — "
                             "grow synthetic sequence.\n", k, exact);
        return 1;
    }
    if (err > bound_abs) {
        std::fprintf(stderr, "[FAIL] test_tiny_exact: k=%d err=%.0f > bound=%.1f\n",
                     k, err, bound_abs);
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const std::string fasta =
        cuhll_test::fasta_or_skip("CUHLL_TEST_FASTA", "test_tiny_exact");
    if (fasta.empty()) return 0;

    std::string chr19_slice;
    try {
        std::string filtered = cuhll::read_fasta_concat(fasta);
        // Take a 2 KB slice from offset 10000 to avoid the initial N-rich prefix
        // that chr19 has near its centromere start.
        constexpr std::size_t off = 10000;
        constexpr std::size_t len = 2000;
        if (filtered.size() < off + len) {
            throw std::runtime_error("FASTA filtered buffer too small for slice");
        }
        chr19_slice = filtered.substr(off, len);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] test_tiny_exact: cannot prepare chr19 slice: %s\n", e.what());
        return 1;
    }

    const std::string seq = build_synthetic(chr19_slice);

    int rc = 0;
    for (int k : {21, 31}) {
        rc |= run_one_k(seq, k);
    }

    if (rc != 0) return rc;
    std::fprintf(stderr, "[PASS] test_tiny_exact\n");
    return 0;
}

// =============================================================================
// test_synthetic.cu — Layer 1 hermetic correctness test.
//
// Generates a sequence containing exactly N distinct canonical k-mers
// (deterministic seed, no FASTA needed) and checks cuHLL's HLL estimate
// is within 4 standard errors of the true count. Self-contained — runs
// on any GPU host without external data, env vars, or scratch files.
//
// HLL standard error is sigma = 1.04 / sqrt(2^p). For p=14 sigma ≈ 0.81%
// so 4*sigma ≈ 3.25%. Each test case picks N >> 2^p to stay in HLL's
// asymptotic regime (avoiding the small-range bias that biases cardinality
// estimation when N is comparable to the register count).
// =============================================================================

#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_set>

namespace {

constexpr char kBases[4] = {'A', 'C', 'G', 'T'};

std::string revcomp(const std::string& s) {
    std::string r(s.size(), 'N');
    for (std::size_t i = 0; i < s.size(); ++i) {
        switch (s[s.size() - 1 - i]) {
            case 'A': r[i] = 'T'; break;
            case 'T': r[i] = 'A'; break;
            case 'C': r[i] = 'G'; break;
            case 'G': r[i] = 'C'; break;
            default:  r[i] = 'N'; break;
        }
    }
    return r;
}

std::string canonical_form(const std::string& kmer) {
    const std::string rc = revcomp(kmer);
    return (kmer < rc) ? kmer : rc;
}

// Build a sequence containing exactly `n_distinct` distinct canonical
// k-mers of length k. Each generated k-mer is concatenated with a run
// of k 'N' characters before and after, which guarantees the parser
// never produces an overlap-induced spurious k-mer (ntHash skips any
// window containing a non-ACGT base).
std::string synthesize_sequence(std::size_t n_distinct, int k,
                                std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<std::string> used;
    used.reserve(n_distinct * 2);

    const std::string sep(static_cast<std::size_t>(k), 'N');
    std::string out;
    out.reserve(n_distinct * static_cast<std::size_t>(2 * k) + k);
    out += sep;

    while (used.size() < n_distinct) {
        std::string kmer(static_cast<std::size_t>(k), 'A');
        for (int i = 0; i < k; ++i) {
            kmer[i] = kBases[rng() % 4];
        }
        if (used.insert(canonical_form(kmer)).second) {
            out += kmer;
            out += sep;
        }
    }
    return out;
}

struct TestCase {
    std::size_t n_distinct;
    int         k;
    int         precision;
};

int run_case(const TestCase& tc) {
    std::fprintf(stderr,
        "[case] N=%zu k=%d p=%d\n",
        tc.n_distinct, tc.k, tc.precision);

    const std::uint64_t seed =
        0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(tc.k) << 32)
                              ^ static_cast<std::uint64_t>(tc.precision);
    const std::string seq = synthesize_sequence(tc.n_distinct, tc.k, seed);

    cuhll::Sketch sketch(tc.precision);
    cuhll::sketch_sequence_single_stream(
        sketch, seq.data(), seq.size(), tc.k);
    const std::uint64_t est = sketch.estimate();

    const double m       = double(std::uint64_t(1) << tc.precision);
    const double sigma   = 1.04 / std::sqrt(m);
    const double tol     = 4.0 * sigma;
    const double truth   = double(tc.n_distinct);
    const double rel_err = std::abs(double(est) - truth) / truth;

    std::fprintf(stderr,
        "       est=%llu  truth=%zu  rel_err=%.4f%%  tol=%.4f%% (4*sigma)\n",
        (unsigned long long)est, tc.n_distinct,
        100.0 * rel_err, 100.0 * tol);

    if (rel_err > tol) {
        std::fprintf(stderr,
            "[FAIL] rel_err %.4f%% exceeds tolerance %.4f%%\n",
            100.0 * rel_err, 100.0 * tol);
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const TestCase cases[] = {
        {100000, 21, 12},
        {500000, 31, 14},
        { 50000, 15, 14},
        {200000, 25, 13},
    };

    int fails = 0;
    for (const auto& tc : cases) {
        fails += run_case(tc);
    }

    if (fails == 0) {
        std::fprintf(stderr, "[PASS] test_synthetic\n");
        return 0;
    }
    std::fprintf(stderr, "[FAIL] test_synthetic: %d case(s)\n", fails);
    return 1;
}

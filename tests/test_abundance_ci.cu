// test_abundance_ci.cu — confidence-interval & "never under-provision" sanity.
//
// Synthetic populations with a KNOWN in-abundance fraction p. Bottom-k samples the
// distinct keys by the real xxhash_64 finalizer (so the sampling is genuine,
// not a model). Over many independent trials we check:
//   * two-sided CI coverage of the true abundance count is near the nominal level,
//   * the one-sided UPPER bound covers the truth at >= the nominal rate
//     (the provisioning guarantee), both for exact F0 and for a noisy F0 that
//     carries a relative error term in the CI.

#include "cuHLL/abundance/abundance_estimator.hpp"
#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace cuhll::abundance;

static inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Coverage { int ci = 0, upper = 0, n = 0; };

// One trial: F0 distinct synthetic keys, each in-abundance with probability p.
// Returns whether truth fell in the two-sided CI and under the upper bound.
static void run_trial(std::uint64_t seed, std::uint64_t F0, double p,
                      std::uint64_t sample_size, const AbundanceConfig& base,
                      double f0_noise_rel, Coverage& cov) {
    std::uint64_t rng = seed;
    // Generate distinct keys + assign in/out-of-abundance counts.
    // In-abundance count = midpoint of (x,y); out-of-abundance = x (a tail, not in abundance).
    std::vector<std::pair<std::uint64_t,std::uint64_t>> keys; // (finalizer, count)
    keys.reserve(F0);
    std::uint64_t true_abundance = 0;
    const std::uint64_t in_count  = (base.x + base.y) / 2;        // strictly inside
    const std::uint64_t out_count = base.x;                       // == x: not in abundance
    for (std::uint64_t i = 0; i < F0; ++i) {
        std::uint64_t key = splitmix64(rng);
        double u = (double)(splitmix64(rng) >> 11) / (double)(1ull << 53);
        std::uint64_t cnt = (u < p) ? in_count : out_count;
        if (in_abundance(cnt, base)) ++true_abundance;
        keys.push_back({finalize(key), cnt});
    }
    // Bottom-k by finalizer.
    std::sort(keys.begin(), keys.end());
    std::uint64_t m = std::min<std::uint64_t>(sample_size, F0);
    std::uint64_t in_sample = 0;
    for (std::uint64_t i = 0; i < m; ++i)
        if (in_abundance(keys[i].second, base)) ++in_sample;

    // F0 estimate (optionally noisy) feeds the point estimate; the CI carries
    // the matching relative-error term.
    AbundanceConfig cfg = base;
    cfg.f0_rel_err = f0_noise_rel;
    double f0_est = (double)F0;
    if (f0_noise_rel > 0.0) {
        // deterministic signed perturbation within ~1 sigma
        double e = ((double)(splitmix64(rng) >> 11) / (double)(1ull<<53) - 0.5) * 2.0;
        f0_est = (double)F0 * (1.0 + f0_noise_rel * e);
    }
    RegionEstimate est = estimate_region((std::uint64_t)(f0_est + 0.5),
                                         in_sample, m, cfg);
    ++cov.n;
    if ((double)true_abundance >= est.lo && (double)true_abundance <= est.hi) ++cov.ci;
    if ((double)true_abundance <= est.upper) ++cov.upper;
}

int main() {
    AbundanceConfig base;
    base.k = 31; base.x = 1; base.y = 6;  // in-abundance counts in {2,3,4,5}; here use midpoint 3
    base.z = 1.645;                        // one-sided 95% / two-sided 90%
    const std::uint64_t F0 = 200000, sample = 4096;
    const int trials = 500;

    // (1) Exact F0.
    Coverage cov;
    for (int t = 0; t < trials; ++t)
        run_trial(0xC0FFEEull + 7919ull * t, F0, 0.10, sample, base, 0.0, cov);
    double ci_rate = (double)cov.ci / cov.n;
    double up_rate = (double)cov.upper / cov.n;
    std::printf("  exact-F0:  two-sided CI cover=%.3f  upper-bound cover=%.3f (n=%d)\n",
                ci_rate, up_rate, cov.n);
    CHECK(ci_rate   >= 0.85);   // nominal 0.90, allow trial noise
    CHECK(up_rate   >= 0.92);   // nominal 0.95 one-sided

    // (2) Noisy F0 (HLL p=14 theoretical ~0.81% rel err): provisioning must
    // still not under-predict at the nominal rate.
    Coverage cov2;
    double hll_rel = 1.04 / std::sqrt((double)(1u << 14));
    for (int t = 0; t < trials; ++t)
        run_trial(0xBEEF01ull + 6271ull * t, F0, 0.10, sample, base, hll_rel, cov2);
    double up2 = (double)cov2.upper / cov2.n;
    std::printf("  noisy-F0:  upper-bound cover=%.3f (rel=%.4f, n=%d)\n",
                up2, hll_rel, cov2.n);
    CHECK(up2 >= 0.92);

    // (3) Higher z = stronger provisioning -> ~never under-predicts.
    AbundanceConfig hz = base; hz.z = 3.0;
    Coverage cov3;
    for (int t = 0; t < trials; ++t)
        run_trial(0x5EED99ull + 5113ull * t, F0, 0.10, sample, hz, 0.0, cov3);
    double up3 = (double)cov3.upper / cov3.n;
    std::printf("  z=3.0:     upper-bound cover=%.3f (n=%d)\n", up3, cov3.n);
    CHECK(up3 >= 0.985);

    return report("test_abundance_ci");
}

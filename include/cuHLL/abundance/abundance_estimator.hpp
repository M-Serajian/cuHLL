#pragma once
// abundance_estimator.hpp — Phase 1 CPU oracle for k-mer abundance cardinality.
//
// ============================ SEMANTICS (CONFIRM) ============================
// count(kmer) = total number of OCCURRENCES of the canonical k-mer across the
//   whole input panel, as emitted by cuHLL's k-mer stream. This is the
//   abundance / multiplicity that KMC3 reports (KMC counts occurrences, which
//   is why it is the oracle here). It is NOT genome-frequency. Within a genome,
//   repeats are counted (every k-mer start position is one occurrence), exactly
//   as the kernel calls ref.add() per position and as KMC counts.
//
// Abundance conventions (flagged for confirmation in PHASE1_REPORT.md):
//   * in-abundance   : x <  count <  y     (STRICT both sides)
//   * tail >= x : count >= x
//   * tail <= y : count <= y
//   F0 = number of DISTINCT canonical k-mers.
//   F_abundance = #{distinct keys : x < count < y}, etc.
//
// Saturating counter: width = ceil(log2(cap+1)) bits. Default cap = y+1 so that
//   every threshold we need (x, y, y+1) is resolvable exactly: counts 0..y are
//   stored exactly and the value (y+1) means ">= y+1". This resolves in-abundance
//   (needs <y vs >=y), tail>=x, and tail<=y (needs <=y vs >=y+1).
// ============================================================================
//
// Two implementations that must agree on retained keys:
//   * brute_force()      — exact, full distinct-count table (no sampling).
//   * StreamingBottomK   — the actual algorithm the GPU sidecar will run: a
//                          bottom-k sample (k_sample distinct keys with the
//                          smallest xxhash_64 finalizer) carrying saturating
//                          per-key occurrence counters, maintained streaming.

#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "cuHLL/abundance/abundance_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuhll::abundance {

struct AbundanceConfig {
    int           k           = 31;
    std::uint64_t x           = 1;     // lower abundance edge (exclusive)
    std::uint64_t y           = 4;     // upper abundance edge (exclusive)
    std::uint64_t sample_size = 4096;  // bottom-k sample size (k_sample)
    std::uint32_t cap         = 0;     // 0 => auto = y+1
    double        f0_rel_err  = 0.0;   // relative error of F0 (HLL); 0 = exact
    double        z           = 1.645; // one-sided 95% by default

    std::uint32_t eff_cap() const {
        return cap ? cap : static_cast<std::uint32_t>(y + 1);
    }
    int counter_bits() const {
        std::uint64_t c = eff_cap();
        int b = 0;
        while ((1ull << b) <= c) ++b;   // ceil(log2(cap+1))
        return b;
    }
};

inline bool in_abundance(std::uint64_t count, const AbundanceConfig& c) {
    return count > c.x && count < c.y;
}
inline bool tail_ge_x(std::uint64_t count, const AbundanceConfig& c) {
    return count >= c.x;
}
inline bool tail_le_y(std::uint64_t count, const AbundanceConfig& c) {
    return count <= c.y;
}

// Exact distinct-count summary.
struct DistinctCounts {
    std::uint64_t f0        = 0;  // distinct keys
    std::uint64_t total_occ = 0;  // sum of all occurrences
    std::uint64_t abundance      = 0;  // x < count < y
    std::uint64_t ge_x      = 0;  // count >= x
    std::uint64_t le_y      = 0;  // count <= y
};

struct BruteForceResult {
    // Distinct (key -> raw occurrence count), RAW (no saturation) so this can
    // also serve as ground truth for the KMC comparison. Sorted by key.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> distinct;
    DistinctCounts counts;
};

// Exact abundance counts from an occurrence multiset. `occ` is consumed (sorted in
// place) for memory efficiency on large inputs.
inline BruteForceResult brute_force(std::vector<std::uint64_t> occ,
                                    const AbundanceConfig& cfg) {
    BruteForceResult r;
    std::sort(occ.begin(), occ.end());
    std::size_t i = 0;
    const std::size_t n = occ.size();
    r.counts.total_occ = n;
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && occ[j] == occ[i]) ++j;
        const std::uint64_t key   = occ[i];
        const std::uint64_t count = static_cast<std::uint64_t>(j - i);
        r.distinct.emplace_back(key, count);
        ++r.counts.f0;
        if (in_abundance(count, cfg))   ++r.counts.abundance;
        if (tail_ge_x(count, cfg)) ++r.counts.ge_x;
        if (tail_le_y(count, cfg)) ++r.counts.le_y;
        i = j;
    }
    return r;
}

// ---------------------------------------------------------------------------
// StreamingBottomK — the algorithm under test.
//
// Maintains the k_sample DISTINCT keys with the smallest finalizer, each with a
// saturating occurrence counter. Proven: the boundary key (rank == k_sample) is
// never evicted, so every retained key's counter reflects ALL its occurrences
// (counts are exact for retained keys).
// ---------------------------------------------------------------------------
class StreamingBottomK {
public:
    StreamingBottomK(std::uint64_t sample_size, std::uint32_t cap)
        : sample_size_(sample_size), cap_(cap) {}

    void add(std::uint64_t canonical) {
        auto it = counts_.find(canonical);
        if (it != counts_.end()) {
            if (it->second < cap_) ++it->second;
            return;
        }
        const std::uint64_t f = finalize(canonical);
        if (counts_.size() < sample_size_) {
            counts_.emplace(canonical, 1u);
            order_.emplace(f, canonical);
        } else {
            auto last = std::prev(order_.end());   // current max finalizer
            if (f < last->first) {
                counts_.erase(last->second);
                order_.erase(last);
                counts_.emplace(canonical, 1u);
                order_.emplace(f, canonical);
            }
            // else: finalizer >= current tau and tau is non-increasing ->
            // this key is not in the bottom-k; drop the occurrence.
        }
    }

    std::uint64_t size() const { return counts_.size(); }

    // tau = k_sample-th smallest finalizer (only meaningful once full).
    bool full() const { return counts_.size() >= sample_size_; }
    std::uint64_t tau() const { return order_.empty() ? 0 : order_.rbegin()->first; }

    const std::unordered_map<std::uint64_t, std::uint32_t>& retained() const {
        return counts_;
    }

private:
    std::uint64_t sample_size_;
    std::uint32_t cap_;
    std::unordered_map<std::uint64_t, std::uint32_t> counts_;       // key -> satcount
    std::set<std::pair<std::uint64_t, std::uint64_t>> order_;       // (finalizer,key)
};

// ---------------------------------------------------------------------------
// Estimator + confidence interval.
//
// p_hat = (#sampled keys in region) / m,  m = #retained sampled keys.
// point = F0 * p_hat.  CI combines the binomial sampling term with F0's
// relative error (delta method); one-sided UPPER bound is the provisioning
// number that must not under-predict.
// ---------------------------------------------------------------------------
struct RegionEstimate {
    std::uint64_t in_sample = 0;   // sampled keys in the region
    std::uint64_t m         = 0;   // sample size used
    double p_hat   = 0.0;
    double point   = 0.0;          // F0 * p_hat
    double sigma   = 0.0;
    double lo      = 0.0;          // two-sided Wald lower (reference, clamped >= 0)
    double hi      = 0.0;          // two-sided Wald upper (reference)
    double upper   = 0.0;          // one-sided WILSON upper + F0 term (the bound)
};

inline RegionEstimate estimate_region(std::uint64_t f0,
                                      std::uint64_t in_sample,
                                      std::uint64_t m,
                                      const AbundanceConfig& cfg) {
    RegionEstimate e;
    e.in_sample = in_sample;
    e.m         = m;
    if (m == 0) return e;
    e.p_hat = static_cast<double>(in_sample) / static_cast<double>(m);
    e.point = static_cast<double>(f0) * e.p_hat;
    const double mm     = static_cast<double>(m);
    const double rel_f0 = cfg.f0_rel_err;

    // Two-sided CI (reference only) — Wald, UNCHANGED.
    const double rel_var = rel_f0 * rel_f0 +
        (e.p_hat > 0.0 ? (1.0 - e.p_hat) / (e.p_hat * mm) : 0.0);
    e.sigma = e.point * std::sqrt(rel_var);
    e.lo    = std::max(0.0, e.point - cfg.z * e.sigma);
    e.hi    = e.point + cfg.z * e.sigma;

    // One-sided UPPER bound — UNCONDITIONAL >= 1-delta via a SPLIT-DELTA union
    // bound (replaces the quadrature assembly, which was not a provable bound
    // and dipped to ~0.989 at small F0 / high sampling ratio). The total risk
    // delta = 1 - Phi(z) is split (Bonferroni) between the two error sources:
    //   delta_p  = delta/2  -> exact Clopper-Pearson upper on p_hat
    //   delta_f0 = delta/2  -> one-sided F0 upper from the HLL relative error
    //                          F0_upper = F0 * (1 + z_{1-delta_f0} * rel_f0)
    //   F_upper  = F0_upper * p_upper
    // Validity: if F0_true <= F0_upper AND p_pop <= p_upper then
    //   F_true = F0_true*p_pop <= F0_upper*p_upper = F_upper. So
    //   P(F_true > F_upper) <= P(F0_true > F0_upper) + P(p_pop > p_upper)
    //                       <= delta_f0 + delta_p = delta   (union bound),
    // i.e. P(F_true <= F_upper) >= 1 - delta = 0.99, for ANY m/F0 — including
    // the adversarial small-input regime. Holds with no finite-population or
    // large-F0 assumption. point estimate and two-sided CI above are unchanged.
    const double delta_total = 0.5 * std::erfc(cfg.z / std::sqrt(2.0)); // 1-Phi(z)
    const double delta_p     = 0.5 * delta_total;
    const double delta_f0    = 0.5 * delta_total;
    const double z_f0        = normal_quantile(1.0 - delta_f0);
    const double p_upper     = clopper_pearson_upper(in_sample, m, delta_p);
    const double f0_upper    = static_cast<double>(f0) * (1.0 + z_f0 * rel_f0);
    e.upper = f0_upper * p_upper;
    return e;
}

struct AbundanceEstimate {
    std::uint64_t m = 0;
    RegionEstimate abundance, ge_x, le_y;
};

// Estimate all three regions from a retained sample (key -> saturating count).
inline AbundanceEstimate estimate_from_sample(
        std::uint64_t f0,
        const std::unordered_map<std::uint64_t, std::uint32_t>& sample,
        const AbundanceConfig& cfg) {
    std::uint64_t m = sample.size();
    std::uint64_t in_abundance_n = 0, ge_x_n = 0, le_y_n = 0;
    for (const auto& kv : sample) {
        const std::uint64_t c = kv.second;
        if (in_abundance(c, cfg))   ++in_abundance_n;
        if (tail_ge_x(c, cfg)) ++ge_x_n;
        if (tail_le_y(c, cfg)) ++le_y_n;
    }
    AbundanceEstimate be;
    be.m    = m;
    be.abundance = estimate_region(f0, in_abundance_n, m, cfg);
    be.ge_x = estimate_region(f0, ge_x_n,   m, cfg);
    be.le_y = estimate_region(f0, le_y_n,   m, cfg);
    return be;
}

}  // namespace cuhll::abundance

// test_abundance_estimate.cu — Phase 4: estimator + guarantee (IMPLEMENT AND MEASURE ONLY).
//
// p_hat by DIRECT classification (strict abundance; inclusive tails; distinct
// comparators) — never a difference of two tails. F_hat = F0 * p_hat. Combined
// variance = F0 HLL-error term + binomial p(1-p)/m. One-sided upper = F_hat+zσ.
//
// Part A: CI coverage with MANY more trials (Phase 1 used 500).
// Part B: on real chr19 test sets, the upper bound must be >= the oracle's true
//         abundance count (the never-under-provision guarantee), using the ACTUAL
//         GPU-counted bottom-k sample.
//
// Per instructions: MEASURE only. Do not modify the variance model here.

#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "cuHLL/abundance/abundance_estimator.hpp"
#include "cuHLL/abundance/abundance_sketch.cuh"
#include "cuHLL/io/fasta.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cuhll::abundance;

static inline std::uint64_t sm64(std::uint64_t& x){
    std::uint64_t z=(x+=0x9E3779B97F4A7C15ull);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return z^(z>>31);
}

static inline double unif(std::uint64_t& rng){
    return (double)(sm64(rng)>>11)/(double)(1ull<<53);
}
static inline double gauss(std::uint64_t& rng){
    double u1=unif(rng), u2=unif(rng);
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0*std::log(u1))*std::cos(6.283185307179586*u2);
}

// Coverage over `trials`. Abundance membership is independent of the (uniform)
// finalizer, so the bottom-k in-abundance count is Binomial(m,p) (drawn exactly) and
// the population truth is Binomial(F0,p) (normal approx; F0 large). F0 carries a
// modelled HLL relative error when f0_rel>0. Returns {two-sided, one-sided-upper}.
struct Cov { double ci, upper; };
// faithful=false: the original model — sample drawn from the fixed parameter p
//   (INCONSISTENT with the realized population truth ~ Binomial(F0,p)).
// faithful=true:  the real bottom-k model — the population is realized first
//   (truth in-abundance keys out of F0), then the m-sample is drawn FROM that fixed
//   population, so p_hat estimates truth/F0 (insamp ~ Binomial(m, truth/F0),
//   the m<<F0 approximation to the hypergeometric draw).
static Cov coverage(std::uint64_t seed, std::uint64_t F0, double p,
                    std::uint64_t m, double f0_rel, double z, int trials,
                    bool faithful=false) {
    AbundanceConfig cfg; cfg.z=z; cfg.f0_rel_err=f0_rel;
    int in_ci=0, in_up=0;
    std::uint64_t rng=seed;
    const double mu_t=(double)F0*p, sd_t=std::sqrt((double)F0*p*(1.0-p));
    for (int t=0;t<trials;++t){
        double truth = mu_t + sd_t*gauss(rng);             // Binomial(F0,p) ~ Normal
        if (truth < 0) truth = 0; if (truth > (double)F0) truth = (double)F0;
        const double p_eff = faithful ? (truth/(double)F0) : p;
        std::uint64_t insamp=0;                            // Binomial(m, p_eff) exact
        for (std::uint64_t j=0;j<m;++j) if (unif(rng)<p_eff) ++insamp;
        double f0e=(double)F0;
        if (f0_rel>0) f0e=(double)F0*(1.0+f0_rel*gauss(rng)); // modelled HLL F0 error
        RegionEstimate e=estimate_region((std::uint64_t)(f0e+0.5),insamp,m,cfg);
        if (truth>=e.lo && truth<=e.hi) ++in_ci;
        if (truth<=e.upper) ++in_up;
    }
    return { (double)in_ci/trials, (double)in_up/trials };
}

static void part_A() {
    const int T=200000;                       // SE@0.99 ~ 0.0002
    const double Z99 = 2.326347874;           // one-sided 99% (delta_total=0.01)
    std::printf("== Part A: split-delta (Bonferroni 0.005/0.005) one-sided 99%% upper, "
                "%d trials/config, FAITHFUL sampling, z=%.4f (SE ~ %.4f) ==\n",
                T, Z99, std::sqrt(0.99*0.01/T));
    const double hll = 1.04/std::sqrt((double)(1u<<14)); // HLL p=14 rel err ~0.81%
    // GATE = min one-sided upper coverage over ALL configs >= 0.99. Both the
    // adversarial small-input (high m/F0) and the real (low m/F0) regimes stay.
    struct Cfg{const char* name; std::uint64_t F0; double p; std::uint64_t m; double rel;};
    Cfg cfgs[] = {
        {"m=4096   exact-F0 p=0.100 F0=2M           ", 2000000ull,  0.100, 4096,  0.0},
        {"m=4096   HLL-F0   p=0.100 F0=2M           ", 2000000ull,  0.100, 4096,  hll},
        {"m=4096   HLL-F0   p=0.028 F0=2M           ", 2000000ull,  0.028, 4096,  hll},
        {"ADVERSARIAL HLL-F0 p=0.028 m=50000 F0=2M  (m/F0=0.025)", 2000000ull,  0.028, 50000, hll},
        {"REAL        HLL-F0 p=0.028 m=50000 F0=50M (m/F0=0.001)", 50000000ull, 0.028, 50000, hll},
    };
    double gate_min = 1.0;
    int idx = 0;
    for (auto& c : cfgs) {
        Cov cov = coverage(0xA11CEull + 1009ull*(++idx), c.F0, c.p, c.m, c.rel,
                           Z99, T, /*faithful=*/true);
        std::printf("  %s : two-sided=%.4f  one-sided-upper=%.4f\n",
                    c.name, cov.ci, cov.upper);
        gate_min = std::min(gate_min, cov.upper);
    }
    std::printf("  => GATE: min one-sided upper coverage (ALL configs incl. "
                "adversarial) = %.4f (target >= 0.99)\n", gate_min);
    CHECK(gate_min >= 0.99);
}

// Host exact abundance truth over the capped stream (full count map).
struct Truth { std::uint64_t f0, abundance, ge_x, le_y; };
static Truth true_abundance(const std::vector<std::string>& seqs, const AbundanceConfig& cfg){
    std::unordered_map<std::uint64_t,std::uint64_t> m; m.reserve(1u<<20);
    for (auto& s: seqs)
        enumerate_capped(reinterpret_cast<const unsigned char*>(s.data()),
            (std::int64_t)s.size(), cfg.k, true, [&](std::uint64_t key){ ++m[key]; });
    Truth t{0,0,0,0};
    for (auto& kv:m){ std::uint64_t c=kv.second; ++t.f0;
        if(in_abundance(c,cfg))++t.abundance; if(tail_ge_x(c,cfg))++t.ge_x; if(tail_le_y(c,cfg))++t.le_y; }
    return t;
}

static void part_B(const std::vector<std::string>& paths, std::uint64_t S,
                   std::uint64_t x, std::uint64_t y) {
    std::vector<std::string> seqs; for (auto&p:paths) seqs.push_back(cuhll::read_fasta_concat(p));
    AbundanceConfig cfg; cfg.k=31; cfg.x=x; cfg.y=y; cfg.sample_size=S;
    cfg.z=2.326347874; cfg.f0_rel_err = 1.04/std::sqrt((double)(1u<<14)); // 99% 1-sided; HLL F0

    Truth tr = true_abundance(seqs, cfg);
    auto g = cuhll::abundance::gpu_tau(seqs, 31, true, S);
    auto sample = cuhll::abundance::gpu_count(seqs, 31, true, g.tau, cfg.eff_cap(), 2*S+1024);
    std::unordered_map<std::uint64_t,std::uint32_t> smap;
    for (auto& p: sample) smap[p.first]=p.second;

    AbundanceEstimate be = estimate_from_sample(tr.f0, smap, cfg);
    const char* lbl = paths.size()==1?paths[0].c_str():"4-genome panel";
    std::printf("== Part B: real upper-bound vs truth (%s, S=%llu, abundance %llu<c<%llu) ==\n",
                lbl,(unsigned long long)S,(unsigned long long)x,(unsigned long long)y);
    std::printf("  F0_exact=%llu  true abundance=%llu\n",
                (unsigned long long)tr.f0,(unsigned long long)tr.abundance);
    std::printf("  EST abundance=%.0f  CI[%.0f,%.0f]  upper=%.0f  (truth %llu)  rel=%.3f%%\n",
                be.abundance.point, be.abundance.lo, be.abundance.hi, be.abundance.upper,
                (unsigned long long)tr.abundance,
                100.0*(be.abundance.point-(double)tr.abundance)/(double)tr.abundance);
    std::printf("  upper>=truth: abundance=%d  ge_x=%d  le_y=%d\n",
                (int)(be.abundance.upper>=(double)tr.abundance),
                (int)(be.ge_x.upper>=(double)tr.ge_x),
                (int)(be.le_y.upper>=(double)tr.le_y));
    CHECK(be.abundance.upper >= (double)tr.abundance);   // never under-provision
    CHECK(be.ge_x.upper >= (double)tr.ge_x);
    CHECK(be.le_y.upper >= (double)tr.le_y);
    CHECK(tr.abundance >= be.abundance.lo && tr.abundance <= be.abundance.hi); // truth in CI
}

int main(int argc, char** argv){
    // No genome args -> Part A coverage gate (CPU). With genome args -> Part B
    // only (run on the L4; skip the CPU coverage sim so the GPU isn't idle).
    if (argc>1){
        std::vector<std::string> paths(argv+1, argv+argc);
        if (paths.size()==1) part_B(paths, 50000, 1, 4);
        else                 part_B(paths, 50000, 1, 8);
    } else {
        part_A();
    }
    return report("test_abundance_estimate");
}

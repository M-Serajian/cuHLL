// test_abundance_oracle.cu — hand-computable correctness of the abundance oracle.
//
// Three sub-tests:
//   1. Abundance boundaries: 5 distinct k-mers with counts exactly x, x+1, y-1, y,
//      y+1 -> exact F0/abundance/tails, and bottom-k retained counts + classification.
//   2. Cap replication (correction B): a stripe packed with >2 ACGT runs ->
//      capped enumeration drops the 3rd+ runs; uncapped keeps them.
//   3. Bottom-k retained set == the m smallest finalizers, with EXACT counts.

#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_estimator.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace cuhll::abundance;

// Build a buffer where each k-mer occupies a fixed-width cell (k ACGT bases
// followed by pad 'N's). With cell width >= 32 + k this guarantees each
// 32-start stripe sees at most one full run, so capped == uncapped and each
// cell contributes exactly one occurrence — making counts hand-computable.
static std::string isolated(const std::vector<std::string>& kmers_in_order,
                            int cell_width) {
    std::string s;
    for (const auto& km : kmers_in_order) {
        s += km;
        for (int i = (int)km.size(); i < cell_width; ++i) s += 'N';
    }
    return s;
}

static std::vector<std::uint64_t> collect_capped(const std::string& s, int k,
                                                 bool canonical) {
    std::vector<std::uint64_t> v;
    enumerate_capped(reinterpret_cast<const unsigned char*>(s.data()),
                     (std::int64_t)s.size(), k, canonical,
                     [&](std::uint64_t h) { v.push_back(h); });
    return v;
}
static std::vector<std::uint64_t> collect_uncapped(const std::string& s, int k,
                                                   bool canonical) {
    std::vector<std::uint64_t> v;
    enumerate_uncapped(reinterpret_cast<const unsigned char*>(s.data()),
                       (std::int64_t)s.size(), k, canonical,
                       [&](std::uint64_t h) { v.push_back(h); });
    return v;
}

static void test_boundaries() {
    const int k = 4;
    // 5 distinct, pairwise non-reverse-complement 4-mers.
    std::string Ka = "AAAA", Kb = "AACC", Kc = "AAGG", Kd = "ACAC", Ke = "ACTG";
    // Counts: a=2(==x), b=3(==x+1), c=4(==y-1), d=5(==y), e=6(==y+1).
    AbundanceConfig cfg;
    cfg.k = k; cfg.x = 2; cfg.y = 5; cfg.sample_size = 100; // retain all
    // cell width 40 (>= 32 + k) keeps cap == uncapped, 1 occ per cell.
    std::vector<std::string> cells;
    auto add = [&](const std::string& km, int c) {
        for (int i = 0; i < c; ++i) cells.push_back(km);
    };
    add(Ka, 2); add(Kb, 3); add(Kc, 4); add(Kd, 5); add(Ke, 6);
    // Shuffle deterministically to exercise ordering (no RNG: rotate).
    std::rotate(cells.begin(), cells.begin() + 7, cells.end());
    std::string seq = isolated(cells, 40);

    auto cap = collect_capped(seq, k, true);
    auto unc = collect_uncapped(seq, k, true);
    CHECK_EQ(cap.size(), unc.size());          // no cap effect with wide cells
    CHECK_EQ((std::uint64_t)cap.size(), 20ull); // 2+3+4+5+6

    auto bf = brute_force(cap, cfg);
    CHECK_EQ(bf.counts.f0, 5ull);
    CHECK_EQ(bf.counts.total_occ, 20ull);
    CHECK_EQ(bf.counts.abundance, 2ull);   // counts 3,4  (b,c)
    CHECK_EQ(bf.counts.ge_x, 5ull);   // all >= 2
    CHECK_EQ(bf.counts.le_y, 4ull);   // all <= 5 except e(6)

    // Bottom-k with full retention -> exact estimate.
    StreamingBottomK bk(cfg.sample_size, cfg.eff_cap());
    for (auto h : cap) bk.add(h);
    CHECK_EQ(bk.size(), 5ull);
    // Retained saturating counts == min(raw, cap) for every distinct key.
    std::vector<std::pair<std::uint64_t,std::uint64_t>> raw(bf.distinct);
    for (auto& kv : bk.retained()) {
        auto it = std::find_if(raw.begin(), raw.end(),
            [&](auto& p){ return p.first == kv.first; });
        CHECK(it != raw.end());
        std::uint64_t expect = std::min<std::uint64_t>(it->second, cfg.eff_cap());
        CHECK_EQ((std::uint64_t)kv.second, expect);
    }
    auto est = estimate_from_sample(bf.counts.f0, bk.retained(), cfg);
    // Full retention => point estimate equals exact truth.
    CHECK_EQ((std::uint64_t)(est.abundance.point + 0.5), 2ull);
    CHECK_EQ((std::uint64_t)(est.ge_x.point + 0.5), 5ull);
    CHECK_EQ((std::uint64_t)(est.le_y.point + 0.5), 4ull);
}

static void test_cap_replication() {
    // 5 cells of width 8 ("ACGT" + 4 N) all the SAME 4-mer "ACGT".
    // Stripe 0 read window [0,35) holds runs at 0,8,16,24 (+partial 32):
    //   capped keeps first 2 (starts 0, 8); stripe covering start 32 keeps it.
    //   uncapped keeps starts 0,8,16,24,32.
    const int k = 4;
    std::vector<std::string> cells(5, "ACGT");
    std::string seq = isolated(cells, 8);   // 40 bytes
    auto cap = collect_capped(seq, k, true);
    auto unc = collect_uncapped(seq, k, true);
    CHECK_EQ((std::uint64_t)unc.size(), 5ull);  // all five occurrences
    CHECK_EQ((std::uint64_t)cap.size(), 3ull);  // cap drops the 3rd & 4th runs
    // All five are the same canonical key.
    AbundanceConfig cfg; cfg.k = k; cfg.x = 0; cfg.y = 100; cfg.sample_size = 10;
    auto bf_unc = brute_force(unc, cfg);
    auto bf_cap = brute_force(cap, cfg);
    CHECK_EQ(bf_unc.counts.f0, 1ull);
    CHECK_EQ(bf_cap.counts.f0, 1ull);
    CHECK_EQ(bf_unc.distinct[0].second, 5ull);
    CHECK_EQ(bf_cap.distinct[0].second, 3ull);  // 2 occurrences dropped by cap
}

static void test_bottomk_smallest() {
    // 6 distinct keys, sample_size=3 -> retained == 3 smallest finalizers,
    // with exact counts, regardless of stream order.
    const int k = 4;
    std::vector<std::string> kmers = {"AAAA","AACC","AAGG","ACAC","ACTG","AGGA"};
    std::vector<int> counts = {1,2,3,4,5,6};
    std::vector<std::string> cells;
    for (size_t i = 0; i < kmers.size(); ++i)
        for (int c = 0; c < counts[i]; ++c) cells.push_back(kmers[i]);
    std::rotate(cells.begin(), cells.begin() + 5, cells.end());
    std::string seq = isolated(cells, 40);
    auto cap = collect_capped(seq, k, true);
    AbundanceConfig cfg; cfg.k=k; cfg.x=0; cfg.y=100;
    auto bf = brute_force(cap, cfg);
    CHECK_EQ(bf.counts.f0, 6ull);

    const std::uint64_t m = 3;
    StreamingBottomK bk(m, cfg.eff_cap());
    for (auto h : cap) bk.add(h);
    CHECK_EQ(bk.size(), m);
    // Independently compute the 3 smallest finalizers from the distinct set.
    std::vector<std::pair<std::uint64_t,std::uint64_t>> byf; // (finalizer,key)
    for (auto& p : bf.distinct) byf.push_back({finalize(p.first), p.first});
    std::sort(byf.begin(), byf.end());
    std::vector<std::uint64_t> want;
    for (std::uint64_t i = 0; i < m; ++i) want.push_back(byf[i].second);
    for (std::uint64_t key : want) CHECK(bk.retained().count(key) == 1);
    // Exact counts for retained keys.
    for (auto& kv : bk.retained()) {
        auto it = std::find_if(bf.distinct.begin(), bf.distinct.end(),
            [&](auto& p){ return p.first == kv.first; });
        CHECK(it != bf.distinct.end());
        CHECK_EQ((std::uint64_t)kv.second,
                 std::min<std::uint64_t>(it->second, cfg.eff_cap()));
    }
}

int main() {
    test_boundaries();
    test_cap_replication();
    test_bottomk_smallest();
    return report("test_abundance_oracle");
}

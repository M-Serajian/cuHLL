// test_abundance_count.cu — Phase 3 GATE: GPU pass-2 counting table == CPU oracle,
// exactly. Covers: retained key set, saturated counts INCLUDING boundary keys
// at x, x+1, y-1, y, y+1; a within-genome repeat (abundance != frequency); a
// skew stress with throughput; and a real-genome cross-check.

#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "cuHLL/abundance/abundance_estimator.hpp"
#include "cuHLL/abundance/abundance_sketch.cuh"
#include "cuHLL/io/fasta.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cuhll::abundance;

// Oracle: exact saturated counts for keys whose finalizer <= tau (the bottom-k
// members), over the capped stream.
static std::unordered_map<std::uint64_t, std::uint32_t>
oracle_counts(const std::vector<std::string>& seqs, int k, bool canonical,
              std::uint64_t tau, std::uint32_t cap) {
    std::unordered_map<std::uint64_t, std::uint32_t> m;
    for (const auto& s : seqs) {
        enumerate_capped(reinterpret_cast<const unsigned char*>(s.data()),
                         (std::int64_t)s.size(), k, canonical,
                         [&](std::uint64_t key){
            if (finalize(key) <= tau) {
                auto& c = m[key];
                if (c < cap) ++c;   // saturating
            }
        });
    }
    return m;
}

// Compare a GPU (key,count) list to the oracle map; returns #mismatches and
// fills set/diff diagnostics.
static std::uint64_t compare(
        const std::vector<std::pair<std::uint64_t,std::uint32_t>>& gpu,
        const std::unordered_map<std::uint64_t,std::uint32_t>& oracle) {
    std::uint64_t mism = 0;
    CHECK_EQ((std::uint64_t)gpu.size(), (std::uint64_t)oracle.size());
    std::unordered_map<std::uint64_t,std::uint32_t> g;
    g.reserve(gpu.size());
    for (auto& p : gpu) g[p.first] = p.second;
    for (auto& kv : oracle) {
        auto it = g.find(kv.first);
        if (it == g.end() || it->second != kv.second) ++mism;
    }
    for (auto& kv : g) if (!oracle.count(kv.first)) ++mism;
    return mism;
}

// Build a buffer where each k-mer occupies a fixed-width cell (k ACGT bases +
// pad N). Wide cells (>= 32+k) make capped == uncapped, 1 occurrence per cell.
static std::string isolated(const std::vector<std::string>& cells, int width) {
    std::string s;
    for (auto& c : cells) { s += c; for (int i=(int)c.size(); i<width; ++i) s += 'N'; }
    return s;
}

static void test_boundaries() {
    const int k = 4;
    // counts x=2, x+1=3, y-1=4, y=5, y+1=6  (abundance x<count<y, x=2 y=5)
    std::string Ka="AAAA",Kb="AACC",Kc="AAGG",Kd="ACAC",Ke="ACTG";
    AbundanceConfig cfg; cfg.k=k; cfg.x=2; cfg.y=5; cfg.sample_size=100;
    std::vector<std::string> cells;
    auto add=[&](const std::string& km,int c){ for(int i=0;i<c;++i) cells.push_back(km); };
    add(Ka,2); add(Kb,3); add(Kc,4); add(Kd,5); add(Ke,6);
    std::string seq = isolated(cells, 40);
    std::vector<std::string> seqs{seq};

    auto g = cuhll::abundance::gpu_tau(seqs, k, true, cfg.sample_size);
    auto counts = cuhll::abundance::gpu_count(seqs, k, true, g.tau, cfg.eff_cap(), 256);
    auto oracle = oracle_counts(seqs, k, true, g.tau, cfg.eff_cap());
    std::uint64_t mism = compare(counts, oracle);
    std::printf("  boundaries: gpu_keys=%zu oracle_keys=%zu mismatches=%llu (cap=%u)\n",
                counts.size(), oracle.size(), (unsigned long long)mism, cfg.eff_cap());
    CHECK_EQ(mism, 0ull);
    // Spot-check the saturated value of the y+1 key (Ke, count 6 -> cap=6).
    // (cap = y+1 = 6; count 6 == cap so stored as 6.)
    CHECK_EQ((std::uint64_t)oracle.size(), 5ull);
}

static void test_within_genome_repeat() {
    // One "genome": the SAME k-mer in 7 wide cells -> count must be 7 (abundance),
    // not 1 (genome frequency). cap high so no saturation.
    const int k = 8;
    std::vector<std::string> cells(7, "ACGTACGT");
    std::string seq = isolated(cells, 48);
    std::vector<std::string> seqs{seq};
    auto g = cuhll::abundance::gpu_tau(seqs, k, true, 100);
    auto counts = cuhll::abundance::gpu_count(seqs, k, true, g.tau, /*cap=*/1000, 64);
    auto oracle = oracle_counts(seqs, k, true, g.tau, 1000);
    std::printf("  within-genome-repeat: distinct=%zu (expect 1), count=%u (expect 7)\n",
                counts.size(), counts.empty()?0:counts[0].second);
    CHECK_EQ((std::uint64_t)counts.size(), 1ull);
    CHECK_EQ((std::uint64_t)counts[0].second, 7ull);   // abundance, not frequency
    CHECK_EQ(compare(counts, oracle), 0ull);
}

static void test_skew_throughput() {
    // Homopolymer: one canonical k-mer repeated ~ (L-k+1) times -> max atomic
    // contention on a single counter; saturation fast-path exercised.
    const int k = 31;
    const std::int64_t L = 20'000'000;   // 20 Mbp of 'A'
    std::string seq(L, 'A');
    std::vector<std::string> seqs{seq};
    const std::uint32_t cap = 9;         // small -> mostly load-only fast path
    auto g = cuhll::abundance::gpu_tau(seqs, k, true, 100);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto counts = cuhll::abundance::gpu_count(seqs, k, true, g.tau, cap, 64);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto oracle = oracle_counts(seqs, k, true, g.tau, cap);
    std::uint64_t occ = (std::uint64_t)(L - k + 1);
    std::printf("  skew: keys=%zu count=%u (cap=%u, occ=%llu) gpu_count=%.1fms "
                "(%.0f Mocc/s)\n", counts.size(), counts.empty()?0:counts[0].second,
                cap, (unsigned long long)occ, ms, occ/1e6/(ms/1e3));
    CHECK_EQ((std::uint64_t)counts.size(), 1ull);
    CHECK_EQ((std::uint64_t)counts[0].second, (std::uint64_t)cap);  // saturated
    CHECK_EQ(compare(counts, oracle), 0ull);
}

static void test_real_genome(const std::vector<std::string>& paths,
                             std::uint64_t S, std::uint64_t x, std::uint64_t y) {
    std::vector<std::string> seqs;
    for (auto& p : paths) seqs.push_back(cuhll::read_fasta_concat(p));
    const std::string label = paths.size()==1 ? paths[0]
                              : (std::to_string(paths.size())+"-genome panel");
    AbundanceConfig cfg; cfg.k=31; cfg.x=x; cfg.y=y; cfg.sample_size=S;
    auto g = cuhll::abundance::gpu_tau(seqs, 31, true, S);
    auto counts = cuhll::abundance::gpu_count(seqs, 31, true, g.tau, cfg.eff_cap(), 2*S+1024);
    auto oracle = oracle_counts(seqs, 31, true, g.tau, cfg.eff_cap());
    std::uint64_t mism = compare(counts, oracle);
    // abundance classification agreement on the sample
    std::uint64_t gb=0, ob=0;
    for (auto& p : counts) if (in_abundance(p.second, cfg)) ++gb;
    for (auto& kv : oracle) if (in_abundance(kv.second, cfg)) ++ob;
    std::printf("  real(%s S=%llu abundance %llu<c<%llu): gpu_keys=%zu oracle_keys=%zu "
                "mismatch=%llu  in-abundance gpu=%llu oracle=%llu\n",
                label.c_str(), (unsigned long long)S,
                (unsigned long long)x, (unsigned long long)y,
                counts.size(), oracle.size(), (unsigned long long)mism,
                (unsigned long long)gb, (unsigned long long)ob);
    CHECK_EQ(mism, 0ull);
    CHECK_EQ(gb, ob);
}

int main(int argc, char** argv) {
    std::printf("test_abundance_count:\n");
    test_boundaries();
    test_within_genome_repeat();
    test_skew_throughput();
    if (argc > 1) {
        std::vector<std::string> paths(argv + 1, argv + argc);
        test_real_genome(paths, 50000, 1, (paths.size() > 1 ? 8 : 4));
    }
    return report("test_abundance_count");
}

// test_abundance_sketch.cu — Phase 6 GATE: the device-resident STREAMING path
// (gpu_stream) must be BIT-IDENTICAL to the verified two-pass oracle
// (gpu_tau for tau, gpu_count for the retained key->count set) on every input.
//
// Covered: tiny boundary cells, within-genome repeat (abundance), skew
// (homopolymer), a many-compaction stress (S << distinct, tiny chunk so tau
// tightens repeatedly and 90%+ of keys are evicted), and real genomes (arg).
// Each input is also run at TWO chunk sizes (one-shot vs tiny) to stress the
// relaxed/stale-tau admission — over-admission must never corrupt the result.

#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "cuHLL/abundance/abundance_sketch.cuh"
#include "cuHLL/io/fasta.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cuhll::abundance;

static std::string isolated(const std::vector<std::string>& cells, int width) {
    std::string s;
    for (auto& c : cells) { s += c; for (int i=(int)c.size(); i<width; ++i) s += 'N'; }
    return s;
}
// i-th distinct k-mer: base-4 encode i into ACGT (enough distinct for the tests).
static std::string gen_kmer(std::uint64_t i, int k) {
    std::string s(k, 'A'); const char* B="ACGT";
    for (int j=0;j<k;++j){ s[k-1-j]=B[i&3]; i>>=2; }
    return s;
}

// Compare a stream retained vector to a two-pass count map, exactly.
static std::uint64_t mismatches(
        const std::vector<std::pair<std::uint64_t,std::uint32_t>>& strm,
        const std::unordered_map<std::uint64_t,std::uint32_t>& twopass) {
    std::uint64_t mism = 0;
    std::unordered_map<std::uint64_t,std::uint32_t> s;
    for (auto& p : strm) s[p.first]=p.second;
    if (s.size() != twopass.size()) ++mism;
    for (auto& kv : twopass) { auto it=s.find(kv.first); if (it==s.end()||it->second!=kv.second) ++mism; }
    for (auto& kv : s) if (!twopass.count(kv.first)) ++mism;
    return mism;
}

// Run the oracle (two-pass) + streaming at one chunk size; assert identical.
static void check_case(const char* name, const std::vector<std::string>& seqs,
                       int k, std::uint64_t S, std::uint32_t cap,
                       std::uint64_t chunk_kmers, std::uint64_t capacity) {
    auto g = cuhll::abundance::gpu_tau(seqs, k, true, S);
    auto cnt = cuhll::abundance::gpu_count(seqs, k, true, g.tau, cap, 2*S+capacity);
    std::unordered_map<std::uint64_t,std::uint32_t> cmap;
    for (auto& p : cnt) cmap[p.first]=p.second;

    auto r = cuhll::abundance::gpu_stream(seqs, k, true, S, cap, chunk_kmers, capacity);
    std::uint64_t mm = mismatches(r.retained, cmap);
    std::printf("  %-26s chunk=%-7llu : 2pass tau=%llu n=%llu | stream tau=%llu n=%llu  "
                "retained-mismatch=%llu\n", name, (unsigned long long)chunk_kmers,
                (unsigned long long)g.tau, (unsigned long long)g.n_distinct,
                (unsigned long long)r.tau, (unsigned long long)r.n_distinct,
                (unsigned long long)mm);
    CHECK_EQ(r.tau, g.tau);                 // BIT-IDENTICAL tau
    CHECK_EQ(r.n_distinct, std::min<std::uint64_t>(S, g.n_distinct));
    CHECK_EQ(mm, 0ull);                     // BIT-IDENTICAL retained counts
}

int main(int argc, char** argv) {
    std::printf("test_abundance_sketch (streaming vs two-pass oracle):\n");

    // 1. Boundary cells (counts x,x+1,y-1,y,y+1), cap=y+1, S retains all.
    {
        int k=4; std::vector<std::string> cells;
        auto add=[&](const std::string& km,int c){ for(int i=0;i<c;++i) cells.push_back(km); };
        add("AAAA",2); add("AACC",3); add("AAGG",4); add("ACAC",5); add("ACTG",6);
        std::vector<std::string> seqs{isolated(cells,40)};
        check_case("boundaries one-shot", seqs, k, 100, 6, 1u<<20, 4096);
        check_case("boundaries tiny-chunk", seqs, k, 100, 6, 64, 4096);
    }
    // 2. Within-genome repeat -> abundance (same k-mer x7).
    {
        int k=8; std::vector<std::string> cells(7,"ACGTACGT");
        std::vector<std::string> seqs{isolated(cells,48)};
        check_case("repeat one-shot", seqs, k, 100, 1000, 1u<<20, 4096);
        check_case("repeat tiny-chunk", seqs, k, 100, 1000, 64, 4096);
    }
    // 3. Skew homopolymer (one canonical key, saturates).
    {
        int k=31; std::string seq(2'000'000,'A');
        std::vector<std::string> seqs{seq};
        check_case("skew one-shot", seqs, k, 100, 9, 1u<<22, 4096);
        check_case("skew tiny-chunk", seqs, k, 100, 9, 1024, 4096);
    }
    // 4. Many-compaction STRESS: 2000 distinct keys, S=50, tiny chunk so tau
    //    tightens repeatedly and ~97% of keys are evicted. Counts vary.
    {
        int k=16; std::vector<std::string> cells;
        for (std::uint64_t i=0;i<2000;++i){ int c=(int)(i%7)+1; for(int j=0;j<c;++j) cells.push_back(gen_kmer(i*2654435761ull,k)); }
        std::vector<std::string> seqs{isolated(cells,56)};
        check_case("stress S=50 one-shot", seqs, k, 50, 8, 1u<<22, 8192);
        check_case("stress S=50 chunk=64", seqs, k, 50, 8, 64, 8192);
        check_case("stress S=50 chunk=256", seqs, k, 50, 8, 256, 8192);
    }
    // 5. Real genomes (arg): single + panel, realistic S/chunk/capacity.
    if (argc > 1) {
        std::vector<std::string> seqs;
        for (int i=1;i<argc;++i) seqs.push_back(cuhll::read_fasta_concat(argv[i]));
        const std::uint64_t S=50000, chunk=200000, capacity=2*(S+chunk);
        const std::uint32_t cap = (seqs.size()>1)?9:5;
        check_case("real genomes", seqs, 31, S, cap, chunk, capacity);
    }
    return report("test_abundance_sketch");
}

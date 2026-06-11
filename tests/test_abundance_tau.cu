// test_abundance_tau.cu — Phase 2 GATE: GPU bottom-k tau == CPU oracle tau (exact).
//
// Oracle tau = S-th smallest distinct xxhash_64(canonical) over the capped
// k-mer stream, computed on the host with the Phase 1 enumerator + finalizer.
// GPU tau computed by the abundance kernel + sort/unique. They must match exactly.

#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_finalizer.hpp"
#include "cuHLL/abundance/abundance_sketch.cuh"
#include "cuHLL/io/fasta.hpp"
#include "abundance_test_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

using namespace cuhll::abundance;

struct OracleTau { std::uint64_t tau, n_distinct, n_occ; bool full; };

static OracleTau oracle_tau(const std::vector<std::string>& seqs, int k,
                            bool canonical, std::uint64_t S) {
    std::unordered_set<std::uint64_t> distinct;
    std::uint64_t n_occ = 0;
    for (const auto& s : seqs) {
        enumerate_capped(reinterpret_cast<const unsigned char*>(s.data()),
                         (std::int64_t)s.size(), k, canonical,
                         [&](std::uint64_t h){ distinct.insert(h); ++n_occ; });
    }
    std::vector<std::uint64_t> fz;
    fz.reserve(distinct.size());
    for (std::uint64_t key : distinct) fz.push_back(finalize(key));
    std::sort(fz.begin(), fz.end());
    OracleTau o;
    o.n_distinct = fz.size();
    o.n_occ = n_occ;
    o.full = o.n_distinct >= S;
    o.tau = fz.empty() ? 0 : (o.full ? fz[S - 1] : fz.back());
    return o;
}

int main(int argc, char** argv) {
    int k = 31;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--k") k = std::stoi(argv[++i]);
        else files.push_back(a);
    }
    if (files.empty()) { std::fprintf(stderr, "need input files\n"); return 2; }

    std::vector<std::string> seqs;
    for (auto& f : files) seqs.push_back(cuhll::read_fasta_concat(f));

    std::printf("test_abundance_tau: %zu file(s), k=%d\n", files.size(), k);
    const std::uint64_t Ss[] = {1000, 50000, 500000, 5000000};
    for (std::uint64_t S : Ss) {
        OracleTau o = oracle_tau(seqs, k, true, S);
        cuhll::abundance::TauResult g = cuhll::abundance::gpu_tau(seqs, k, true, S);
        std::printf("  S=%-8llu  oracle: tau=%llu distinct=%llu occ=%llu full=%d | "
                    "gpu: tau=%llu distinct=%llu occ=%llu full=%d\n",
                    (unsigned long long)S,
                    (unsigned long long)o.tau, (unsigned long long)o.n_distinct,
                    (unsigned long long)o.n_occ, o.full,
                    (unsigned long long)g.tau, (unsigned long long)g.n_distinct,
                    (unsigned long long)g.n_occ, g.full);
        CHECK_EQ(g.n_occ, o.n_occ);            // same capped occurrence multiset
        CHECK_EQ(g.n_distinct, o.n_distinct);  // same distinct finalizers
        CHECK_EQ((int)g.full, (int)o.full);
        CHECK_EQ(g.tau, o.tau);                // THE gate
    }
    return report("test_abundance_tau");
}

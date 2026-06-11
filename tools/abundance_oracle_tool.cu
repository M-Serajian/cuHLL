// abundance_oracle_tool.cu — Phase 1 CPU oracle driver for real genomes.
//
// For one or more FASTA inputs (read via cuHLL's own read_fasta_concat so the
// byte stream is identical to what the kernel sees), this:
//   * builds EXACT panel-wide occurrence-count tables for both the capped
//     (cuHLL-actual) and uncapped (platonic / KMC) k-mer streams,
//   * runs the StreamingBottomK algorithm (the GPU sidecar's design) across the
//     genomes as a streaming union accumulator,
//   * checks retained counts == brute force (estimator-correctness oracle),
//   * prints the abundance estimate + CI vs the exact truth,
//   * reports the capped-vs-uncapped enumeration gap (occurrences dropped),
//   * optionally compares the uncapped truth to a KMC3 histogram (three-way).
//
// Usage:
//   oracle_tool --k 31 --x X --y Y --sample S [--z Z] [--f0-rel R]
//               [--kmc-hist FILE] f1.fasta [f2.fasta ...]

#include "cuHLL/io/fasta.hpp"
#include "cuHLL/abundance/kmer_enumerator.hpp"
#include "cuHLL/abundance/abundance_estimator.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cuhll::abundance;

struct Tally {
    std::uint64_t f0 = 0, total_occ = 0, abundance = 0, ge_x = 0, le_y = 0;
};

static Tally tally_map(const std::unordered_map<std::uint64_t,std::uint64_t>& m,
                       const AbundanceConfig& cfg) {
    Tally t;
    for (auto& kv : m) {
        std::uint64_t c = kv.second;
        ++t.f0; t.total_occ += c;
        if (in_abundance(c, cfg))   ++t.abundance;
        if (tail_ge_x(c, cfg)) ++t.ge_x;
        if (tail_le_y(c, cfg)) ++t.le_y;
    }
    return t;
}

int main(int argc, char** argv) {
    AbundanceConfig cfg;
    cfg.sample_size = 100000;
    std::string kmc_hist;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* n){ return std::string(argv[++i]); };
        if      (a == "--k")       cfg.k = std::stoi(val("k"));
        else if (a == "--x")       cfg.x = std::stoull(val("x"));
        else if (a == "--y")       cfg.y = std::stoull(val("y"));
        else if (a == "--sample")  cfg.sample_size = std::stoull(val("s"));
        else if (a == "--z")       cfg.z = std::stod(val("z"));
        else if (a == "--f0-rel")  cfg.f0_rel_err = std::stod(val("r"));
        else if (a == "--kmc-hist")kmc_hist = val("h");
        else                       files.push_back(a);
    }
    if (files.empty()) { std::fprintf(stderr, "no input files\n"); return 2; }

    std::printf("== oracle_tool ==\n");
    std::printf("k=%d abundance: %llu < count < %llu  sample=%llu cap=%u (%d bits) z=%.3f f0_rel=%.4f\n",
                cfg.k, (unsigned long long)cfg.x, (unsigned long long)cfg.y,
                (unsigned long long)cfg.sample_size, cfg.eff_cap(),
                cfg.counter_bits(), cfg.z, cfg.f0_rel_err);
    std::printf("inputs (%zu):\n", files.size());
    for (auto& f : files) std::printf("  %s\n", f.c_str());

    // Load each file's byte stream once (identical to the kernel's input).
    std::vector<std::string> seqs;
    seqs.reserve(files.size());
    for (auto& f : files) seqs.push_back(cuhll::read_fasta_concat(f));

    // ---- Capped pass: exact map + the streaming algorithm together ----
    std::unordered_map<std::uint64_t,std::uint64_t> map_cap;
    map_cap.reserve(1u << 20);
    StreamingBottomK bk(cfg.sample_size, cfg.eff_cap());
    for (auto& s : seqs) {
        enumerate_capped(reinterpret_cast<const unsigned char*>(s.data()),
                         (std::int64_t)s.size(), cfg.k, true,
                         [&](std::uint64_t h){ ++map_cap[h]; bk.add(h); });
    }
    Tally cap = tally_map(map_cap, cfg);

    // Retained-count check: streaming retained == brute force (saturated).
    std::uint64_t mism = 0;
    for (auto& kv : bk.retained()) {
        auto it = map_cap.find(kv.first);
        std::uint64_t expect = (it == map_cap.end()) ? 0
            : std::min<std::uint64_t>(it->second, cfg.eff_cap());
        if (kv.second != expect) ++mism;
    }
    // Estimator from the streaming sample (F0 = exact distinct of capped stream).
    AbundanceEstimate be = estimate_from_sample(cap.f0, bk.retained(), cfg);

    std::printf("\n-- CAPPED (cuHLL-actual) stream --\n");
    std::printf("  F0(distinct)=%llu  total_occ=%llu\n",
                (unsigned long long)cap.f0, (unsigned long long)cap.total_occ);
    std::printf("  TRUE   abundance=%llu  ge_x=%llu  le_y=%llu\n",
                (unsigned long long)cap.abundance, (unsigned long long)cap.ge_x,
                (unsigned long long)cap.le_y);
    std::printf("  sample m=%llu  retained-count mismatches=%llu  tau=%llu\n",
                (unsigned long long)be.m, (unsigned long long)mism,
                (unsigned long long)bk.tau());
    std::printf("  EST abundance=%.0f  [%.0f, %.0f]  upper=%.0f   (truth %llu, in-CI=%d)\n",
                be.abundance.point, be.abundance.lo, be.abundance.hi, be.abundance.upper,
                (unsigned long long)cap.abundance,
                (cap.abundance >= be.abundance.lo && cap.abundance <= be.abundance.hi));
    std::printf("  EST ge_x=%.0f  upper=%.0f   (truth %llu)\n",
                be.ge_x.point, be.ge_x.upper, (unsigned long long)cap.ge_x);
    std::printf("  EST le_y=%.0f  upper=%.0f   (truth %llu)\n",
                be.le_y.point, be.le_y.upper, (unsigned long long)cap.le_y);
    {
        double rel = cap.abundance ? 100.0*(be.abundance.point-(double)cap.abundance)/(double)cap.abundance : 0.0;
        std::printf("  abundance relative error = %+.3f%%\n", rel);
    }
    std::fflush(stdout);
    std::unordered_map<std::uint64_t,std::uint64_t>().swap(map_cap); // free

    // ---- Uncapped pass: platonic truth (== KMC enumeration) ----
    std::unordered_map<std::uint64_t,std::uint64_t> map_unc;
    map_unc.reserve(1u << 20);
    for (auto& s : seqs) {
        enumerate_uncapped(reinterpret_cast<const unsigned char*>(s.data()),
                           (std::int64_t)s.size(), cfg.k, true,
                           [&](std::uint64_t h){ ++map_unc[h]; });
    }
    Tally unc = tally_map(map_unc, cfg);
    std::printf("\n-- UNCAPPED (platonic / KMC-equivalent) stream --\n");
    std::printf("  F0(distinct)=%llu  total_occ=%llu  abundance=%llu  ge_x=%llu  le_y=%llu\n",
                (unsigned long long)unc.f0, (unsigned long long)unc.total_occ,
                (unsigned long long)unc.abundance, (unsigned long long)unc.ge_x,
                (unsigned long long)unc.le_y);

    std::printf("\n-- CAP GAP (uncapped - capped) --\n");
    std::printf("  occurrences dropped by 2-run cap = %lld\n",
                (long long)unc.total_occ - (long long)cap.total_occ);
    std::printf("  distinct delta = %lld   abundance delta = %lld\n",
                (long long)unc.f0 - (long long)cap.f0,
                (long long)unc.abundance - (long long)cap.abundance);
    double dropfrac = unc.total_occ ?
        100.0*((double)unc.total_occ-(double)cap.total_occ)/(double)unc.total_occ : 0.0;
    std::printf("  dropped fraction of occurrences = %.6f%%\n", dropfrac);

    // ---- Three-way: uncapped vs KMC3 histogram ----
    if (!kmc_hist.empty()) {
        std::ifstream in(kmc_hist);
        if (!in) { std::fprintf(stderr, "cannot open kmc hist %s\n", kmc_hist.c_str()); }
        else {
            Tally kmc;
            std::uint64_t cnt, n;
            while (in >> cnt >> n) {
                kmc.f0 += n; kmc.total_occ += cnt * n;
                if (in_abundance(cnt, cfg))   kmc.abundance += n;
                if (tail_ge_x(cnt, cfg)) kmc.ge_x += n;
                if (tail_le_y(cnt, cfg)) kmc.le_y += n;
            }
            std::printf("\n-- KMC3 (exact, no cap) --\n");
            std::printf("  F0=%llu  abundance=%llu  ge_x=%llu  le_y=%llu\n",
                        (unsigned long long)kmc.f0, (unsigned long long)kmc.abundance,
                        (unsigned long long)kmc.ge_x, (unsigned long long)kmc.le_y);
            std::printf("  vs UNCAPPED:  dF0=%lld  dabundance=%lld  dge_x=%lld  dle_y=%lld\n",
                        (long long)unc.f0 - (long long)kmc.f0,
                        (long long)unc.abundance - (long long)kmc.abundance,
                        (long long)unc.ge_x - (long long)kmc.ge_x,
                        (long long)unc.le_y - (long long)kmc.le_y);
            std::printf("  (nonzero deltas at this scale => ntHash 64-bit collisions; expected ~0)\n");
        }
    }

    std::printf("\nRESULT: retained_mismatches=%llu\n", (unsigned long long)mism);
    return mism == 0 ? 0 : 1;
}

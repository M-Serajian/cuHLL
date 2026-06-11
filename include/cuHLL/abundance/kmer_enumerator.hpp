#pragma once
// kmer_enumerator.hpp — exact CPU replica of cuHLL's canonical k-mer stream.
//
// Phase 1 oracle component. This emits the SAME canonical ntHash values, in the
// same multiset, that cuHLL's GPU kernel inserts — so the abundance feature can be
// validated against what the GPU actually sees, not the platonic k-mer set.
//
// It reuses (calls, never modifies) the read-only device/host inline hash
// functions in cuHLL/kmer/nthash.cuh.
//
// Two modes:
//   * CAPPED  — replicates kmer_extract_kernel EXACTLY, including the
//               "at most 2 ACGT runs per 32-start stripe" cap
//               (src/kmer/kmer_kernel.cu:151-171). This is cuHLL's actual
//               stream and what the GPU feature will count.
//   * UNCAPPED — every k-mer in every ACGT run of length >= k, emitted once.
//               This is the platonic set KMC3 enumerates (ground truth).
//
// Each emitted value is the canonical ntHash min(fwd, rc) — the same uint64
// cuHLL passes to ref.add(). The xxhash_64 finalizer used for bottom-k
// sampling is applied later (abundance_finalizer.hpp), exactly as cuco does internally.

#include "cuHLL/kmer/nthash.cuh"
#include "cuHLL/common/common.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cuhll::abundance {

// Decode one byte to a base code 0..3 (ACGT, case-insensitive) or 4 (other).
// Mirrors nt_base_code_for_lut / the shared LUT in the kernel.
inline unsigned base_code(unsigned char c) {
    switch (c) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:            return 4u;
    }
}

// Compile-time constants mirrored from the kernel (read-only source of truth:
// include/cuHLL/kmer/kmer_kernel.cuh). Kept as local constants here — NOT a
// modification of the kernel; just matched values for the replica.
inline constexpr int kStripe = 32;   // kKmerExtractStripe

// Emit every canonical ntHash for the k-mers the kernel would emit from one
// in-memory sequence buffer `seq` (already parsed: headers stripped, single
// 'N' between records — produced by cuhll::read_fasta_concat).
//
// `Emit` is any callable void(std::uint64_t canonical).
//
// canonical=true  -> min(fwd, rc) (cuHLL default);
// canonical=false -> forward hash only (cuHLL --no-canonical).
template <typename Emit>
void enumerate_capped(const unsigned char* seq, std::int64_t len, int k,
                      bool canonical, Emit&& emit) {
    if (len < k) return;
    const std::int64_t n_positions = len - static_cast<std::int64_t>(k) + 1;

    // Iterate the 32-start stripes exactly as the kernel partitions them.
    for (std::int64_t s_start = 0; s_start < len; s_start += kStripe) {
        const std::int64_t owned_end =
            std::min<std::int64_t>(s_start + kStripe, n_positions);
        // Kernel guard: stripe_start_g < len && owned_end_g > stripe_start_g.
        if (!(s_start < len && owned_end > s_start)) continue;

        const std::int64_t read_end =
            std::min<std::int64_t>(owned_end + k - 1, len);

        // Phase 1: scan [s_start, read_end) for ACGT runs of length >= k,
        // keeping at most the FIRST 2 (the kernel's n_runs < 2 cap).
        std::int64_t run_s[2], run_e[2];
        int n_runs = 0;
        std::int64_t rs = -1;
        for (std::int64_t i = s_start; i < read_end; ++i) {
            if (base_code(seq[i]) <= 3u) {
                if (rs < 0) rs = i;
            } else {
                if (rs >= 0 && (i - rs) >= k && n_runs < 2) {
                    run_s[n_runs] = rs; run_e[n_runs] = i; ++n_runs;
                }
                rs = -1;
            }
        }
        if (rs >= 0 && (read_end - rs) >= k && n_runs < 2) {
            run_s[n_runs] = rs; run_e[n_runs] = read_end; ++n_runs;
        }

        // Phase 2+3: init at run start, then roll; emit k-mers whose START is
        // in the owned range [s_start, owned_end).
        for (int r = 0; r < n_runs; ++r) {
            const std::int64_t rs_l = run_s[r];
            const std::int64_t re_l = run_e[r];

            std::uint64_t fwd = 0, rc = 0;
            for (int j = 0; j < k; ++j) {
                const unsigned c = base_code(seq[rs_l + j]);
                fwd ^= cuhll::rotl64(cuhll::nt_seed(c), k - 1 - j);
                rc  ^= cuhll::rotl64(cuhll::nt_seed(cuhll::nt_complement_code(c)), j);
            }
            if (rs_l >= s_start && rs_l < owned_end)
                emit(canonical ? cuhll::nt_canonical(fwd, rc) : fwd);

            for (std::int64_t i = rs_l + k; i < re_l; ++i) {
                const unsigned co = base_code(seq[i - k]);
                const unsigned ci = base_code(seq[i]);
                fwd = cuhll::nt_hash_roll_fwd(fwd, co, ci, k);
                rc  = cuhll::nt_hash_roll_rc (rc,  co, ci, k);
                const std::int64_t ks = i - k + 1;
                if (ks >= s_start && ks < owned_end)
                    emit(canonical ? cuhll::nt_canonical(fwd, rc) : fwd);
            }
        }
    }
}

// Uncapped enumeration: every k-mer in every maximal ACGT run of length >= k,
// emitted exactly once. This equals KMC3's enumeration (canonical, N breaks
// runs, no stripe cap). Used as the ground-truth platonic set.
template <typename Emit>
void enumerate_uncapped(const unsigned char* seq, std::int64_t len, int k,
                        bool canonical, Emit&& emit) {
    if (len < k) return;
    std::int64_t i = 0;
    while (i < len) {
        // Find next maximal ACGT run.
        if (base_code(seq[i]) > 3u) { ++i; continue; }
        std::int64_t rs = i;
        while (i < len && base_code(seq[i]) <= 3u) ++i;
        std::int64_t re = i;  // [rs, re) is a maximal ACGT run
        if ((re - rs) < k) continue;

        std::uint64_t fwd = 0, rc = 0;
        for (int j = 0; j < k; ++j) {
            const unsigned c = base_code(seq[rs + j]);
            fwd ^= cuhll::rotl64(cuhll::nt_seed(c), k - 1 - j);
            rc  ^= cuhll::rotl64(cuhll::nt_seed(cuhll::nt_complement_code(c)), j);
        }
        emit(canonical ? cuhll::nt_canonical(fwd, rc) : fwd);
        for (std::int64_t p = rs + k; p < re; ++p) {
            const unsigned co = base_code(seq[p - k]);
            const unsigned ci = base_code(seq[p]);
            fwd = cuhll::nt_hash_roll_fwd(fwd, co, ci, k);
            rc  = cuhll::nt_hash_roll_rc (rc,  co, ci, k);
            emit(canonical ? cuhll::nt_canonical(fwd, rc) : fwd);
        }
    }
}

}  // namespace cuhll::abundance

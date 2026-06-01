#pragma once
// =============================================================================
// ntHash — recursive nucleotide hashing.
//
//   Mohamadi, Chu, Vandervalk, Birol. "ntHash: recursive nucleotide hashing
//   for high-throughput sequence analysis". Bioinformatics, 2016.
//   DOI: 10.1093/bioinformatics/btw397.
//
// This is the original ntHash, not ntHash2 (2022). The original variant is
// what ntCard / ntHits / most downstream genomics tooling expect.
//
// -----------------------------------------------------------------------------
// Per-base 64-bit seed values from the paper:
//
//   seedA = 0x3c8bfbb395c60474
//   seedC = 0x3193c18562a02b4c
//   seedG = 0x20323ed082572324
//   seedT = 0x295549f54be24456
//   seedN = 0
//
// Forward hash for a k-mer s[0..k-1]:
//   H_fwd = XOR_{i=0..k-1} rotl( seed(s[i]), k - 1 - i )
//
// Reverse-complement hash (the hash of the reverse-complement k-mer):
//   H_rc  = XOR_{i=0..k-1} rotl( seed(complement(s[i])), i )
//
// Canonical hash: H = min(H_fwd, H_rc) (unsigned lexicographic min).
//
// Rolling recurrences when the window slides one base (out = s[i] leaves,
// in = s[i+k] enters):
//   H_fwd_new = rotl(H_fwd, 1)
//             XOR rotl(seed(out), k)
//             XOR seed(in)
//   H_rc_new  = rotr(H_rc, 1)
//             XOR rotr(seed(complement(out)), 1)
//             XOR rotl(seed(complement(in)), k - 1)
//
// -----------------------------------------------------------------------------
// Implementation note: seeds are stored as scalar constexpr values (not as a
// __constant__ memory table). For a 4-entry table the compiler folds the
// scalars to immediates on both host and device, removing the storage
// footprint and the host/device dual-definition dance an array would need.
// For a larger seed table (per-codon, per-trimer, perfect-hash style), move
// to __constant__ memory.
// =============================================================================

#include "cuHLL/common/common.hpp"

#include <cstdint>

namespace cuhll {

// Seed constants.
inline constexpr std::uint64_t kNtSeedA = 0x3c8bfbb395c60474ull;
inline constexpr std::uint64_t kNtSeedC = 0x3193c18562a02b4cull;
inline constexpr std::uint64_t kNtSeedG = 0x20323ed082572324ull;
inline constexpr std::uint64_t kNtSeedT = 0x295549f54be24456ull;
inline constexpr std::uint64_t kNtSeedN = 0ull;

// 64-bit rotations. The `& 63` mask keeps behavior defined for r == 0 and
// r == 64, both of which the rolling recurrence can produce at the
// boundaries of the supported k range.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t rotl64(std::uint64_t x, int r) {
    unsigned s = static_cast<unsigned>(r) & 63u;
    return (x << s) | (x >> ((64u - s) & 63u));
}

CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t rotr64(std::uint64_t x, int r) {
    unsigned s = static_cast<unsigned>(r) & 63u;
    return (x >> s) | (x << ((64u - s) & 63u));
}

// Base encoding: 0/1/2/3 for A/C/G/T (case-insensitive). Returns 4 for any
// other byte (N, IUPAC ambiguity codes, gaps, garbage). A return > 3 is a
// signal to the caller that the current k-mer window must be reset rather
// than rolled.
CUHLL_HD CUHLL_FORCE_INLINE unsigned nt_base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:            return 4u;
    }
}

// Complement code: A<->T (0<->3), C<->G (1<->2).
CUHLL_HD CUHLL_FORCE_INLINE unsigned nt_complement_code(unsigned code) {
    return code ^ 3u;
}

// Seed lookup. A switch is used instead of an array indexed by `code`
// because nvcc does not always emit a device-side definition for namespace-
// scope `inline constexpr` arrays, while scalar constexpr values are folded
// to immediates on both sides. The compiler turns this into a jump table
// with no static storage.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_seed(unsigned code) {
    switch (code & 3u) {
        case 0u: return kNtSeedA;
        case 1u: return kNtSeedC;
        case 2u: return kNtSeedG;
        default: return kNtSeedT; // case 3u
    }
}

// Initial hashes over a contiguous k-base window s[0..k-1]. The caller must
// have verified that every base in [s, s+k) is ACGT (the window has been
// primed); these functions do not re-validate.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_hash_init_fwd(const char* s, int k) {
    std::uint64_t h = 0;
    for (int i = 0; i < k; ++i) {
        h ^= rotl64(nt_seed(nt_base_code(s[i])), k - 1 - i);
    }
    return h;
}

CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_hash_init_rc(const char* s, int k) {
    std::uint64_t h = 0;
    for (int i = 0; i < k; ++i) {
        h ^= rotl64(nt_seed(nt_complement_code(nt_base_code(s[i]))), i);
    }
    return h;
}

// Rolling updates. `code_out` is the base leaving the window (was at the
// left end); `code_in` is the base entering (at the right end). Both must
// be ACGT codes in [0, 3]; non-ACGT bases require resetting the window
// rather than rolling.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_hash_roll_fwd(
        std::uint64_t h, unsigned code_out, unsigned code_in, int k) {
    h = rotl64(h, 1);
    h ^= rotl64(nt_seed(code_out), k);
    h ^= nt_seed(code_in);
    return h;
}

CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_hash_roll_rc(
        std::uint64_t h, unsigned code_out, unsigned code_in, int k) {
    h = rotr64(h, 1);
    h ^= rotr64(nt_seed(nt_complement_code(code_out)), 1);
    h ^= rotl64(nt_seed(nt_complement_code(code_in)), k - 1);
    return h;
}

// Canonical = unsigned min(forward, reverse-complement). This is the hash
// that should be inserted into the HLL sketch for canonical k-mer counting.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_canonical(
        std::uint64_t fwd, std::uint64_t rc) {
    return fwd < rc ? fwd : rc;
}

} // namespace cuhll

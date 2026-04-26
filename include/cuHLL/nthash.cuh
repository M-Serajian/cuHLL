#pragma once
// =============================================================================
// ntHash (original variant, Mohamadi, Chu, Vandervalk, Birol 2016).
//
//   "ntHash: recursive nucleotide hashing for high-throughput sequence
//    analysis", Bioinformatics, 2016. DOI: 10.1093/bioinformatics/btw397.
//
// NOT ntHash2 (2022). The original variant is what ntCard / ntHits / most
// downstream genomics tooling rely on, and is the canonical reference.
//
// -----------------------------------------------------------------------------
// Per-base 64-bit seed table (Mohamadi 2016):
//
//     seedA = 0x3c8bfbb395c60474
//     seedC = 0x3193c18562a02b4c
//     seedG = 0x20323ed082572324
//     seedT = 0x295549f54be24456
//     seedN = 0
//
// Forward hash for k-mer s[0..k-1]:
//     H_fwd = XOR_{i=0..k-1} rotl(seed(s[i]), k-1-i)
//
// Reverse-complement hash (hash of the RC k-mer under the same formula):
//     H_rc  = XOR_{i=0..k-1} rotl(seed(comp(s[i])), i)
//
// Canonical hash:
//     H     = min(H_fwd, H_rc)     // lexicographic compare on unsigned values
//
// Rolling recurrences when the window slides by one base (out = s[i], in =
// s[i+k]):
//     H_fwd_new = rotl(H_fwd, 1)
//               XOR rotl(seed(out), k)
//               XOR seed(in)
//     H_rc_new  = rotr(H_rc, 1)
//               XOR rotr(seed(comp(out)), 1)
//               XOR rotl(seed(comp(in)), k - 1)
//
// -----------------------------------------------------------------------------
// Design note: seed storage.
// The spec called for `__constant__` memory for the seed table. For a 32-byte
// 4-entry table, a `constexpr` table is at least as fast (the PTX backend
// materializes the tiny table as immediate loads / register constants) and
// has the bonus of being directly usable from both host and device code with
// no `#ifdef __CUDA_ARCH__` dual-definition dance. If a future variant needs
// a larger seed table (per-codon / per-trimer / perfect-hash style), switch
// to `__constant__` here — that is where the `LD.CONST` cache actually helps.
// =============================================================================

#include "cuHLL/common.hpp"

#include <cstdint>

namespace cuhll {

// -----------------------------------------------------------------------------
// Seed constants (original ntHash, Mohamadi 2016).
// -----------------------------------------------------------------------------
inline constexpr std::uint64_t kNtSeedA = 0x3c8bfbb395c60474ull;
inline constexpr std::uint64_t kNtSeedC = 0x3193c18562a02b4cull;
inline constexpr std::uint64_t kNtSeedG = 0x20323ed082572324ull;
inline constexpr std::uint64_t kNtSeedT = 0x295549f54be24456ull;
inline constexpr std::uint64_t kNtSeedN = 0ull;

// -----------------------------------------------------------------------------
// Rotation helpers. Masking with `& 63` keeps behavior defined for r == 0 and
// r == 64, both of which the rolling recurrence relies on when k == 64 (not
// our current regime, but cheap safety).
// -----------------------------------------------------------------------------
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t rotl64(std::uint64_t x, int r) {
    unsigned s = static_cast<unsigned>(r) & 63u;
    return (x << s) | (x >> ((64u - s) & 63u));
}

CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t rotr64(std::uint64_t x, int r) {
    unsigned s = static_cast<unsigned>(r) & 63u;
    return (x >> s) | (x << ((64u - s) & 63u));
}

// -----------------------------------------------------------------------------
// Base encoding. Returns 0/1/2/3 for A/C/G/T (case-insensitive) and 4 for
// anything else (N, IUPAC ambiguity codes, gaps). A return of > 3 signals to
// the caller that the current k-mer window must be reset.
// -----------------------------------------------------------------------------
CUHLL_HD CUHLL_FORCE_INLINE unsigned nt_base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:            return 4u;
    }
}

CUHLL_HD CUHLL_FORCE_INLINE unsigned nt_complement_code(unsigned code) {
    return code ^ 3u; // A<->T (0<->3), C<->G (1<->2)
}

// A switch over scalar constexpr values is used instead of a table lookup:
// nvcc does not always emit a device-side definition for namespace-scope
// `inline constexpr` arrays (triggers an "undefined in device code" error at
// link time), while scalar constexpr values fold to immediates on both sides.
// The compiler turns this into a jump table / conditional select with no
// storage footprint.
CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_seed(unsigned code) {
    switch (code & 3u) {
        case 0u: return kNtSeedA;
        case 1u: return kNtSeedC;
        case 2u: return kNtSeedG;
        default: return kNtSeedT; // case 3u
    }
}

// -----------------------------------------------------------------------------
// Initial hashes over a contiguous k-base window s[0..k-1]. Caller must have
// verified that every base in [s, s+k) is ACGT (the window has been primed).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Rolling updates. `code_out` is the base leaving the window (was at the left
// end), `code_in` is the base entering (at the right end). Both must be ACGT
// codes 0..3; the caller is responsible for detecting non-ACGT bases and
// resetting the window instead of rolling.
// -----------------------------------------------------------------------------
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

CUHLL_HD CUHLL_FORCE_INLINE std::uint64_t nt_canonical(
        std::uint64_t fwd, std::uint64_t rc) {
    return fwd < rc ? fwd : rc;
}

} // namespace cuhll

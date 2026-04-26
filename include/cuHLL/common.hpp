#pragma once
#include <cstdint>
#include <cstddef>

// Host/device portability macros. When compiled with nvcc, CUHLL_HD expands to
// __host__ __device__; with a plain C++ compiler it is empty. This lets a
// single header be included from both .cpp and .cu translation units without
// duplicating the implementation.
#if defined(__CUDACC__)
    #define CUHLL_HD __host__ __device__
    #define CUHLL_DEV __device__
    #define CUHLL_GLOBAL __global__
#else
    #define CUHLL_HD
    #define CUHLL_DEV
    #define CUHLL_GLOBAL
#endif

#if defined(__CUDA_ARCH__)
    #define CUHLL_FORCE_INLINE __forceinline__
#else
    #define CUHLL_FORCE_INLINE inline
#endif

namespace cuhll {

// Supported k-mer length range.
//
// kMaxK is a HARD CONSTRAINT: the hot path uses a single-uint64 carrier
// for the 2-bit-packed canonical k-mer (2 bits * 32 bases = 64 bits).
// Widening this requires changing ntHash state, the kernel carrier
// type, and the cuco instantiation together. The extension point for a
// future 128-bit path (k up to 63) lives in a comment block in the
// k-mer extraction kernel.
//
// kMinK is conservative, not algorithmic. The kernel + ntHash + HLL
// math all work at any k >= 1, but for k < ~15 the cardinality space
// is small enough (4**8 = 65,536 possible 8-mers) that an exact
// counter beats HLL on both memory and accuracy. We keep the floor
// low (k=8) so users with unusual workloads aren't shut out, while
// recommending k >= 15 in practice. Bump down further if you have a
// real reason — nothing in the code path objects.
//
// Enforced:
//   - runtime check in the CLI rejects k outside [kMinK, kMaxK].
//   - static_assert on kMaxK in the kmer kernels (carrier-width invariant).
constexpr int kMinK = 8;
constexpr int kMaxK = 32;

// Single-uint64 hash-state invariant. If this ever needs to grow, the entire
// hot path (ntHash state, kernel carrier type, cuco instantiation) must
// widen together — see the FUTURE comment in src/kmer_kernel.cu.
static_assert(kMaxK <= 32,
              "cuHLL single-uint64 data path requires k <= 32");

// HLL precision bounds. p=14 (m=16384 registers) is the default and matches
// HyperLogLog++'s standard-error target of ~0.81%.
constexpr int kMinPrecision = 8;
constexpr int kMaxPrecision = 18;
constexpr int kDefaultPrecision = 14;

// Streaming chunk size defaults (MB). Large chunks amortize kernel-launch cost
// across many k-mers; small chunks overlap H2D and compute more aggressively.
constexpr std::size_t kDefaultChunkMB = 32;
constexpr int kDefaultNumStreams = 3;

} // namespace cuhll

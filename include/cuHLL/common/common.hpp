#pragma once
// Shared compile-time constants and host/device portability macros.
//
// Safe to include from both .cpp and .cu translation units.

#include <cstdint>
#include <cstddef>

// Host/device portability macros. With nvcc, CUHLL_HD expands to
// __host__ __device__; with a plain C++ compiler it is empty.
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
// kMaxK = 32 is a hard limit: the hot path packs the canonical k-mer state
// into a single 64-bit integer (2 bits per base, 32 bases). Going beyond 32
// requires widening ntHash, the kernel carrier type, and the cuco hash
// instantiation together.
//
// kMinK = 8 is a soft floor. The math works for any k >= 1, but for very
// small k the cardinality space is tiny (4**8 = 65,536 for 8-mers) and an
// exact counter outperforms HLL on both memory and accuracy. p >= 15 is
// recommended in practice; the floor exists so unusual workloads aren't
// shut out.
//
// Enforced at:
//   - CLI argument parsing (rejects k outside [kMinK, kMaxK]).
//   - static_assert on kMaxK in the k-mer kernel header.
constexpr int kMinK = 8;
constexpr int kMaxK = 32;

static_assert(kMaxK <= 32,
              "cuHLL single-uint64 data path requires k <= 32");

// HyperLogLog precision bounds. p=14 (m=16384 registers) is the default and
// matches HyperLogLog++'s standard-error target of ~0.81%.
constexpr int kMinPrecision = 8;
constexpr int kMaxPrecision = 18;
constexpr int kDefaultPrecision = 14;

// Streaming pipeline defaults. Larger chunks amortize kernel-launch cost
// across more k-mers; smaller chunks expose more H2D / compute overlap.
constexpr std::size_t kDefaultChunkMB = 32;
constexpr int kDefaultNumStreams = 3;

} // namespace cuhll

#pragma once
// CUDA-only. Do not include from pure-C++ translation units.
//
// k-mer extraction kernel + host launcher.
//
// Each thread owns kKmerExtractStripe k-mer start positions and reads
// (stripe + k - 1) bases so the last k-mer in the stripe has all its bases
// in scope. Ownership is defined by each k-mer's start index, so threads on
// adjacent stripes never double-insert and never drop a k-mer at boundaries.

#include "cuHLL/common/common.hpp"
#include "cuHLL/kmer/nthash.cuh"

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog_ref.cuh>

#include <cstdint>

namespace cuhll {

// Compile-time tunables.
//
//   kKmerExtractBlockSize : threads per block. Grid size is resolved at
//                           first launch via cudaOccupancyMaxActive…
//                           regardless of this value.
//   kKmerExtractStripe    : k-mer starts per thread. Larger stripes amortize
//                           per-k-mer rolling-hash cost; smaller stripes
//                           expose more parallelism. 32 keeps each block's
//                           working set inside L1 cache.
constexpr int kKmerExtractBlockSize = 256;
constexpr int kKmerExtractStripe    = 32;

static_assert(kMaxK <= 32, "Single-uint64 ntHash data path requires k <= 32");

// cuco hyperloglog device reference type used by the kernel.
//
// Hash is cuco::xxhash_64 because canonical = min(forward, reverse-comp) is
// not uniform in its top bits; xxhash_64 applied after the min restores the
// uniform distribution HLL's register selection requires.
//
// Must be kept in lockstep with SketchImpl::RefT in sketch_internal.cuh.
using SketchHllRef = cuco::hyperloglog_ref<
    std::uint64_t,
    cuda::thread_scope_device,
    cuco::xxhash_64<std::uint64_t>
>;

// The cuco device ref is passed BY VALUE into the kernel.
//
// cuco::hyperloglog_ref is a non-owning view: a span plus an empty hasher,
// ~16 bytes, trivially constexpr, no non-trivial destructor. Copying it into
// the kernel parameter block is free. If a future cuco variant introduces a
// non-trivial destructor on the ref type, switch this argument to a
// reference or raw pointer.

__global__ void __launch_bounds__(kKmerExtractBlockSize)
kmer_extract_kernel(const char* __restrict__ seq,
                    std::int64_t len,
                    int k,
                    int stripe_len,
                    SketchHllRef ref,
                    bool canonical);

// Host launcher. Resolves the grid size via
// cudaOccupancyMaxActiveBlocksPerMultiprocessor on first invocation and
// caches the result in a process-local static.
//
//   blocks_per_sm_out (optional): if non-null, receives the measured
//                                 blocks-per-SM value from the occupancy
//                                 query.
void launch_kmer_extract(const char* d_seq,
                         std::int64_t len,
                         int k,
                         SketchHllRef ref,
                         cudaStream_t stream,
                         bool canonical = true,
                         int* blocks_per_sm_out = nullptr);

} // namespace cuhll

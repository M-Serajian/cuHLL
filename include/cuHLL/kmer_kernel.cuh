#pragma once
// CUDA-only. Do not include from pure-C++ TUs.
//
// Stripe-per-thread k-mer extraction kernel + host launcher. Each thread owns
// `kKmerExtractStripe` k-mer START positions and reads (stripe + k - 1) bases
// so the last k-mer in the stripe has all its bases in scope. Ownership is
// defined by the k-mer's starting index so boundaries never double-insert
// and never drop a k-mer.

#include "cuHLL/common.hpp"
#include "cuHLL/nthash.cuh"

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog_ref.cuh>

#include <cstdint>

namespace cuhll {

// Compile-time kernel tunables.
//
// kKmerExtractBlockSize: 256 is the default. After an occupancy pass on the
// first successful build we may drop to 128 if it wins. Whichever is chosen,
// `launch_kmer_extract` resolves the grid via
// cudaOccupancyMaxActiveBlocksPerMultiprocessor at first call, so adjusting
// the block size doesn't require hand-tuning a grid count.
//
// kKmerExtractStripe: bases per thread (owned k-mer starts). Tunable trade-off:
//   - larger stripes amortize per-mer roll cost but leave tail threads
//     imbalanced on short sequences;
//   - smaller stripes expose more parallelism but raise per-byte overhead.
// 1024 is the project-wide default per the locked design.
constexpr int kKmerExtractBlockSize = 256;
constexpr int kKmerExtractStripe    = 1024;

// Single-uint64 hash-state invariant. Widening to k > 32 requires a 128-bit
// carrier — see the FUTURE comment inside the kernel.
static_assert(kMaxK <= 32, "Single-uint64 ntHash data path requires k <= 32");

// Type alias for the cuco hyperloglog device ref. Hash is cuco::xxhash_64
// because canonical = min(fwd, rc) is not uniform enough in its top bits for
// HLL's register-selection to produce a well-calibrated estimate — xxhash_64
// after the canonical-min restores uniformity. Keep this alias in lockstep
// with SketchImpl::RefT in sketch_internal.cuh.
using SketchHllRef = cuco::hyperloglog_ref<
    std::uint64_t,
    cuda::thread_scope_device,
    cuco::xxhash_64<std::uint64_t>
>;

// -----------------------------------------------------------------------------
// Device-ref passing.
//
// We pass the cuco device ref BY VALUE into the kernel. Justification:
//   - cuco::hyperloglog_ref is a non-owning view (internally a span + an
//     empty hasher). Copy is ~16 bytes and trivially constexpr.
//   - hyperloglog_ref has no non-trivial destructor, so copy into the kernel
//     parameter block is free and exit does not fire any cleanup.
//   - This matches the pattern in cuco's own examples / tests.
// If cuco ever gains a non-trivial destructor on its ref (e.g. atomic cleanup
// in a future variant), switch the kernel signature to
//   `SketchHllRef const& ref`  (reference to host-staged ref in constant
// memory) or to a raw pointer argument. For now, by-value is correct.
// -----------------------------------------------------------------------------

// Kernel declaration.
__global__ void __launch_bounds__(kKmerExtractBlockSize)
kmer_extract_kernel(const char* __restrict__ seq,
                    std::int64_t len,
                    int k,
                    int stripe_len,
                    SketchHllRef ref,
                    bool canonical);

// Host-side launcher. Resolves grid size via
// cudaOccupancyMaxActiveBlocksPerMultiprocessor on first invocation and caches
// the result in a process-local static. `blocks_per_sm_out` (if non-null)
// receives the measured value — useful for the milestone (c) report.
void launch_kmer_extract(const char* d_seq,
                         std::int64_t len,
                         int k,
                         SketchHllRef ref,
                         cudaStream_t stream,
                         bool canonical = true,
                         int* blocks_per_sm_out = nullptr);

} // namespace cuhll

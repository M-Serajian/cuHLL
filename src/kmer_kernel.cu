// kmer_kernel.cu — stripe-per-thread ntHash+cuco extraction kernel.
//
// Grid sizing is resolved at first launch via
// cudaOccupancyMaxActiveBlocksPerMultiprocessor. Measured on L4 (sm_89):
//
//     blockSize              = 256
//     blocks_per_sm          = 6
//     multiProcessorCount    = 58
//     grid                   = 348 (= 58 * 6)
//     warps/SM at full grid  = 6 * 8 = 48  (L4 max = 48)
//     theoretical occupancy  = 100%
//
// This file intentionally has no __shared__ usage inside the extraction
// kernel: cuco::hyperloglog_ref handles its own shared-memory sub-sketches
// internally (cooperative reduce across warps into the register array),
// which is why passing the device ref by value is the correct pattern here.

#include "cuHLL/cuda_check.hpp"
#include "cuHLL/kmer_kernel.cuh"
#include "cuHLL/nthash.cuh"
#include "cuHLL/nvtx_util.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace cuhll {

__global__ void __launch_bounds__(kKmerExtractBlockSize)
kmer_extract_kernel(const char* __restrict__ seq,
                    std::int64_t len,
                    int k,
                    int stripe_len,
                    SketchHllRef ref,
                    bool canonical) {
    // Per-thread state. 64-bit throughout — single-uint64 ntHash path.
    //
    // FUTURE: 128-bit extension.
    // For k in (32, 63], the ntHash state stays 64-bit but the canonical
    // k-mer packing widens to two uint64_t's. The insert key would become a
    // pair<uint64_t,uint64_t> (or __uint128_t) and cuco would need a matching
    // hyperloglog instantiation. ntHash itself is unchanged — that's the
    // whole point of picking it here.
    const std::int64_t tid  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t grid = static_cast<std::int64_t>(gridDim.x)  * blockDim.x;

    for (std::int64_t stripe_start = tid * stripe_len;
         stripe_start < len;
         stripe_start += grid * stripe_len) {

        // k-mer starts owned by this stripe: [stripe_start, owned_end)
        // clipped to [0, len - k + 1).
        const std::int64_t owned_end = min(stripe_start + stripe_len,
                                           len - static_cast<std::int64_t>(k) + 1);
        if (owned_end <= stripe_start) continue; // nothing to produce

        // Last base index the stripe may read (inclusive). The last owned
        // k-mer starts at `owned_end - 1` and needs bases up to that + k - 1.
        const std::int64_t read_end = min(owned_end + static_cast<std::int64_t>(k) - 1, len);

        unsigned valid = 0;
        std::uint64_t fwd = 0;
        std::uint64_t rc  = 0;

        for (std::int64_t i = stripe_start; i < read_end; ++i) {
            const unsigned code = nt_base_code(seq[i]);
            if (code > 3u) {
                valid = 0;
                continue;
            }
            ++valid;
            if (valid < static_cast<unsigned>(k)) continue;

            if (valid == static_cast<unsigned>(k)) {
                // First full k-mer in this valid run: prime.
                fwd = nt_hash_init_fwd(seq + i - k + 1, k);
                rc  = nt_hash_init_rc (seq + i - k + 1, k);
            } else {
                // Roll by one base.
                const unsigned code_out = nt_base_code(seq[i - k]);
                fwd = nt_hash_roll_fwd(fwd, code_out, code, k);
                rc  = nt_hash_roll_rc (rc,  code_out, code, k);
            }

            // Ownership gate: only insert if the starting position of this
            // k-mer lies in our owned stripe. This makes the right-overlap
            // safe (we read into the next stripe's bases to finish our own
            // last few k-mers, but don't re-insert k-mers that belong to it).
            const std::int64_t kstart = i - static_cast<std::int64_t>(k) + 1;
            if (kstart >= stripe_start && kstart < owned_end) {
                // L1: canonical = min(fwd, rc); non-canonical = fwd.
                // Runtime branch is warp-uniform (the same bool for all
                // threads in a launch), so there is no divergence cost.
                ref.add(canonical ? nt_canonical(fwd, rc) : fwd);
            }
        }
    }
}

namespace {
// Process-local cached grid params. First launch runs the occupancy query;
// subsequent launches reuse the decision.
struct LaunchCache {
    bool   ready        = false;
    int    blocks_per_sm = 0;
    int    grid          = 0;
};

LaunchCache& launch_cache() {
    static LaunchCache c;
    return c;
}

void resolve_grid_once() {
    auto& c = launch_cache();
    if (c.ready) return;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        kmer_extract_kernel,
        kKmerExtractBlockSize,
        /*dynamicSMemBytes=*/0));
    c.blocks_per_sm = blocks_per_sm;
    c.grid          = prop.multiProcessorCount * blocks_per_sm;
    c.ready         = true;

    // One-shot log line on first launch so the milestone (c) run captures the
    // numbers we promised to record.
    std::fprintf(stderr,
                 "[cuHLL] kmer_extract_kernel occupancy: blockSize=%d blocks_per_sm=%d "
                 "multiProcessorCount=%d grid=%d stripe=%d\n",
                 kKmerExtractBlockSize, blocks_per_sm, prop.multiProcessorCount,
                 c.grid, kKmerExtractStripe);
}

} // namespace

void launch_kmer_extract(const char* d_seq,
                         std::int64_t len,
                         int k,
                         SketchHllRef ref,
                         cudaStream_t stream,
                         bool canonical,
                         int* blocks_per_sm_out) {
    CUHLL_NVTX_RANGE("launch_kmer_extract");
    resolve_grid_once();
    const auto& c = launch_cache();

    if (blocks_per_sm_out) *blocks_per_sm_out = c.blocks_per_sm;

    if (len < static_cast<std::int64_t>(k)) return; // no k-mers; skip launch

    kmer_extract_kernel<<<c.grid, kKmerExtractBlockSize, 0, stream>>>(
        d_seq, len, k, kKmerExtractStripe, ref, canonical);
    CUDA_CHECK_LAST();
}

} // namespace cuhll

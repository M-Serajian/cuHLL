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

        // Phase 1: scan for valid runs (contiguous ACGT stretches >= k).
        // Absorbs all N-handling so Phases 2 and 3 are branch-free.
        // With stripe=32 and k=31, at most ~2 valid runs per stripe.
        struct Run { std::int64_t s, e; };
        constexpr int kMaxRuns = 2;
        Run runs[kMaxRuns];
        int n_runs = 0;
        std::int64_t rs = -1;

        for (std::int64_t i = stripe_start; i < read_end; ++i) {
            if (nt_base_code(seq[i]) <= 3u) {
                if (rs < 0) rs = i;
            } else {
                if (rs >= 0 && (i - rs) >= k && n_runs < kMaxRuns)
                    runs[n_runs++] = {rs, i};
                rs = -1;
            }
        }
        if (rs >= 0 && (read_end - rs) >= k && n_runs < kMaxRuns)
            runs[n_runs++] = {rs, read_end};

        // Phase 2 + 3: for each valid run, init once then roll.
        for (int r = 0; r < n_runs; ++r) {
            // Phase 2: prime fwd and rc hashes over the first k bases.
            std::uint64_t fwd = nt_hash_init_fwd(seq + runs[r].s, k);
            std::uint64_t rc  = nt_hash_init_rc (seq + runs[r].s, k);

            std::int64_t kstart = runs[r].s;
            if (kstart >= stripe_start && kstart < owned_end)
                ref.add(canonical ? nt_canonical(fwd, rc) : fwd);

            // Phase 3: pure rolling — no N-checks, no valid tracking.
            for (std::int64_t i = runs[r].s + k; i < runs[r].e; ++i) {
                const unsigned code_out = nt_base_code(seq[i - k]);
                const unsigned code_in  = nt_base_code(seq[i]);
                fwd = nt_hash_roll_fwd(fwd, code_out, code_in, k);
                rc  = nt_hash_roll_rc (rc,  code_out, code_in, k);

                kstart = i - static_cast<std::int64_t>(k) + 1;
                if (kstart >= stripe_start && kstart < owned_end)
                    ref.add(canonical ? nt_canonical(fwd, rc) : fwd);
            }
        }
    }
}

namespace {
// Process-local cache. blocks_per_sm + sm_count come from a one-shot
// occupancy query at first launch; the actual grid size is computed
// per-launch from the input length (see launch_kmer_extract).
struct LaunchCache {
    bool ready          = false;
    int  blocks_per_sm  = 0;
    int  sm_count       = 0;
    int  occupancy_grid = 0;   // sm_count × blocks_per_sm — used as the
                                // floor on small inputs so we don't
                                // under-fill the GPU.
};

LaunchCache& launch_cache() {
    static LaunchCache c;
    return c;
}

void resolve_occupancy_once() {
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
    c.blocks_per_sm  = blocks_per_sm;
    c.sm_count       = prop.multiProcessorCount;
    c.occupancy_grid = prop.multiProcessorCount * blocks_per_sm;
    c.ready          = true;

    std::fprintf(stderr,
                 "[cuHLL] kmer_extract_kernel occupancy: blockSize=%d "
                 "blocks_per_sm=%d multiProcessorCount=%d "
                 "occupancy_grid=%d stripe=%d\n",
                 kKmerExtractBlockSize, blocks_per_sm,
                 prop.multiProcessorCount, c.occupancy_grid,
                 kKmerExtractStripe);
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
    resolve_occupancy_once();
    const auto& c = launch_cache();

    if (blocks_per_sm_out) *blocks_per_sm_out = c.blocks_per_sm;

    if (len < static_cast<std::int64_t>(k)) return;

    // Grid sized to the input so each thread does at most one
    // grid-stride iteration. Stripe size is preserved (still 1024
    // positions per thread) so the rolling-hash savings within a
    // stripe stay intact — we only change how many blocks compete
    // for SMs.
    //
    // Floor at the SM-occupancy grid so small inputs still saturate
    // the device. Cap at int max for safety; CUDA's grid limit is
    // 2^31-1 per dim and we never come anywhere near that in practice.
    const std::int64_t n_positions = len - static_cast<std::int64_t>(k) + 1;
    const std::int64_t positions_per_block =
        static_cast<std::int64_t>(kKmerExtractBlockSize) *
        static_cast<std::int64_t>(kKmerExtractStripe);
    std::int64_t grid64 =
        (n_positions + positions_per_block - 1) / positions_per_block;
    if (grid64 < c.occupancy_grid) grid64 = c.occupancy_grid;
    constexpr std::int64_t kGridCap = (1LL << 30);
    if (grid64 > kGridCap) grid64 = kGridCap;
    const int grid = static_cast<int>(grid64);

    kmer_extract_kernel<<<grid, kKmerExtractBlockSize, 0, stream>>>(
        d_seq, len, k, kKmerExtractStripe, ref, canonical);
    CUDA_CHECK_LAST();
}

} // namespace cuhll

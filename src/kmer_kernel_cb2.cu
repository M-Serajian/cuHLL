// kmer_kernel_cb2.cu — .cb2 path extraction kernel.
//
// Reads 2-bit packed bases + 1-bit mask directly on the device. No ASCII
// conversion, no branch on the base's character — just a shift+mask for
// the base code and a shift+mask for the N-mask bit.

#include "cuHLL/cuda_check.hpp"
#include "cuHLL/kmer_kernel_cb2.cuh"
#include "cuHLL/nthash.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace cuhll {

namespace {

__device__ __forceinline__ unsigned cb2_base(const std::uint8_t* p, std::int64_t i) {
    return (p[i >> 2] >> ((static_cast<unsigned>(i) & 3u) << 1)) & 0x3u;
}

__device__ __forceinline__ unsigned cb2_mask(const std::uint8_t* m, std::int64_t i) {
    return (m[i >> 3] >> (static_cast<unsigned>(i) & 7u)) & 0x1u;
}

} // namespace

__global__ void __launch_bounds__(kKmerExtractBlockSize)
kmer_extract_kernel_cb2(const std::uint8_t* __restrict__ packed,
                        const std::uint8_t* __restrict__ mask,
                        std::int64_t n_bases,
                        int k,
                        int stripe_len,
                        SketchHllRef ref,
                        bool canonical) {
    const std::int64_t tid  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t grid = static_cast<std::int64_t>(gridDim.x)  * blockDim.x;

    for (std::int64_t stripe_start = tid * stripe_len;
         stripe_start < n_bases;
         stripe_start += grid * stripe_len) {

        const std::int64_t owned_end = min(stripe_start + stripe_len,
                                           n_bases - static_cast<std::int64_t>(k) + 1);
        if (owned_end <= stripe_start) continue;

        const std::int64_t read_end = min(owned_end + static_cast<std::int64_t>(k) - 1, n_bases);

        unsigned valid = 0;
        std::uint64_t fwd = 0;
        std::uint64_t rc  = 0;

        for (std::int64_t i = stripe_start; i < read_end; ++i) {
            // Mask bit set => this position is non-ACGT in the original FASTA;
            // break the window.
            if (cb2_mask(mask, i)) {
                valid = 0;
                continue;
            }
            const unsigned code = cb2_base(packed, i);

            ++valid;
            if (valid < static_cast<unsigned>(k)) continue;

            if (valid == static_cast<unsigned>(k)) {
                // Prime with the last k valid bases.
                std::uint64_t f = 0;
                std::uint64_t r = 0;
                for (int j = 0; j < k; ++j) {
                    const unsigned c = cb2_base(packed, i - k + 1 + j);
                    f ^= rotl64(nt_seed(c), k - 1 - j);
                    r ^= rotl64(nt_seed(nt_complement_code(c)), j);
                }
                fwd = f;
                rc  = r;
            } else {
                const unsigned code_out = cb2_base(packed, i - k);
                fwd = nt_hash_roll_fwd(fwd, code_out, code, k);
                rc  = nt_hash_roll_rc (rc,  code_out, code, k);
            }

            const std::int64_t kstart = i - static_cast<std::int64_t>(k) + 1;
            if (kstart >= stripe_start && kstart < owned_end) {
                ref.add(canonical ? nt_canonical(fwd, rc) : fwd);
            }
        }
    }
}

namespace {

struct Cb2LaunchCache {
    bool ready         = false;
    int  blocks_per_sm = 0;
    int  grid          = 0;
};

Cb2LaunchCache& cb2_launch_cache() {
    static Cb2LaunchCache c;
    return c;
}

void cb2_resolve_grid_once() {
    auto& c = cb2_launch_cache();
    if (c.ready) return;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        kmer_extract_kernel_cb2,
        kKmerExtractBlockSize,
        /*dynamicSMemBytes=*/0));
    c.blocks_per_sm = blocks_per_sm;
    c.grid          = prop.multiProcessorCount * blocks_per_sm;
    c.ready         = true;

    std::fprintf(stderr,
                 "[cuHLL] kmer_extract_kernel_cb2 occupancy: blockSize=%d "
                 "blocks_per_sm=%d multiProcessorCount=%d grid=%d stripe=%d\n",
                 kKmerExtractBlockSize, blocks_per_sm, prop.multiProcessorCount,
                 c.grid, kKmerExtractStripe);
}

} // namespace

void launch_kmer_extract_cb2(const std::uint8_t* d_packed,
                             const std::uint8_t* d_mask,
                             std::int64_t n_bases,
                             int k,
                             SketchHllRef ref,
                             cudaStream_t stream,
                             bool canonical,
                             int* blocks_per_sm_out) {
    cb2_resolve_grid_once();
    const auto& c = cb2_launch_cache();
    if (blocks_per_sm_out) *blocks_per_sm_out = c.blocks_per_sm;

    if (n_bases < static_cast<std::int64_t>(k)) return;

    kmer_extract_kernel_cb2<<<c.grid, kKmerExtractBlockSize, 0, stream>>>(
        d_packed, d_mask, n_bases, k, kKmerExtractStripe, ref, canonical);
    CUDA_CHECK_LAST();
}

} // namespace cuhll

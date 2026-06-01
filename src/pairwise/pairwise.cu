// pairwise.cu — all-pairs HLL Jaccard, single-pass full-precision.
//
//   - pack_registers(): int32 (cuco's native register width) -> uint8.
//     Safe because HLL register values are bounded by p + clz(64-p)
//     which is well under 255.
//   - precompute_cardinality_kernel(): per-sketch HLL cardinality, hoisted
//     out of the pairwise inner loop.
//   - exact_kernel(): 16384-register Jaccard for every pair, padded shared
//     memory, launch_bounds-pinned occupancy.
//   - compute_pairwise_jaccard_exact(): host-side convenience wrapper.

#include "cuHLL/pairwise/pairwise.cuh"
#include "cuHLL/common/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuhll {
namespace pairwise {

// Tile / thread mapping.
constexpr int kT       = 16;
constexpr int kThreads = 256;   // one thread per pair in the 16x16 tile

// ---------- conversion -------------------------------------------------------
void pack_registers(const std::uint32_t* src_int32,
                    std::uint8_t* dst_uint8,
                    int n_registers) {
    // Saturating cast — cuco values are bounded but a defensive clamp
    // prevents nonsense if a corrupted file ever sneaks through.
    for (int i = 0; i < n_registers; ++i) {
        std::uint32_t v = src_int32[i];
        dst_uint8[i] = static_cast<std::uint8_t>(v > 255u ? 255u : v);
    }
}

// =============================================================================
//  Precompute per-sketch HLL cardinality estimate.
//
//  eA[i] depends only on sketch i, not on any pair partner — so it is
//  computed once per sketch here and read N-1 times by the pairwise kernel
//  rather than recomputed on every pair.
//
//  One block per sketch, 256 threads. Each thread walks a strided slice of
//  the sketch's uchar4 chunks, accumulating partial register-sum and
//  zero-count values, then a tree reduction in shared memory produces the
//  final pair (sum, zeros). The HLL estimator with LinearCounting
//  small-range correction is applied by thread 0 and stored in eA_out[sid].
// =============================================================================
__global__ void precompute_cardinality_kernel(
    const std::uint8_t* __restrict__ sketches,
    int n,
    float* __restrict__ eA_out)
{
    constexpr int NT = 256;
    const int sid = blockIdx.x;
    if (sid >= n) return;
    const int tid = threadIdx.x;

    __shared__ float sm_lut[66];
    __shared__ float sm_sum[NT];
    __shared__ int   sm_z[NT];

    if (tid < 66) sm_lut[tid] = ldexpf(1.0f, -tid);
    __syncthreads();

    const uchar4* sk4 = reinterpret_cast<const uchar4*>(
        sketches + (std::size_t)sid * kBytesPerSketch);
    constexpr int n_chunks = kBytesPerSketch / 4;

    float local_sum = 0.f;
    int   local_z   = 0;
    #pragma unroll 4
    for (int r = tid; r < n_chunks; r += NT) {
        uchar4 v = sk4[r];
        local_sum += sm_lut[v.x] + sm_lut[v.y] + sm_lut[v.z] + sm_lut[v.w];
        local_z   += (v.x==0) + (v.y==0) + (v.z==0) + (v.w==0);
    }
    sm_sum[tid] = local_sum;
    sm_z[tid]   = local_z;
    __syncthreads();

    #pragma unroll
    for (int s = NT / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm_sum[tid] += sm_sum[tid + s];
            sm_z[tid]   += sm_z[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = sm_sum[0];
        int   total_z   = sm_z[0];
        const float m = (float)kRegisters;
        const float alpha_mm = (0.7213f / (1.0f + 1.079f / m)) * m * m;
        float e = alpha_mm / total_sum;
        if (e <= 2.5f * m && total_z != 0) e = m * logf(m / (float)total_z);
        eA_out[sid] = e;
    }
}

// =============================================================================
//  Pairwise kernel — full 16K-register Jaccard for every pair (i, j) with
//  i < j. Each block handles a kT x kT tile of pairs; one thread per pair.
//
//  Per-sketch cardinality estimates eA[i] are passed in via d_eA (produced
//  by precompute_cardinality_kernel above). Only the union cardinality eU
//  is accumulated per pair; Jaccard is then (eA[i] + eA[j] - eU) / eU.
//
//  __launch_bounds__(256, 2) caps regs/thread so two blocks can co-reside
//  on an SM; without it the compiler would spill into more registers and
//  the block-per-SM limit drops to one.
// =============================================================================
__global__ __launch_bounds__(256, 2)
void exact_kernel(const std::uint8_t* __restrict__ sketches,
                  int n,
                  const float* __restrict__ d_eA,
                  float* __restrict__ exact_j) {
    constexpr int stripe_size = 1024;
    constexpr int n_stripes   = kBytesPerSketch / stripe_size;

    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bj < bi) return;

    const int tid = threadIdx.x;
    const int ti  = tid / kT;
    const int tj  = tid % kT;
    const int gi  = bi * kT + ti;
    const int gj  = bj * kT + tj;
    const bool valid = (gi < n) && (gj < n) && (gi < gj);

    // Row stride is stripe_size + 4 (not stripe_size). The +4 breaks the
    // power-of-two stride that would otherwise map every row to the same
    // shared-memory bank, producing wide bank conflicts on the per-pair
    // reads of sm_B[tj][...] in the inner loop.
    constexpr int kPadStride = stripe_size + 4;
    __shared__ std::uint8_t sm_A[kT][kPadStride];
    __shared__ std::uint8_t sm_B[kT][kPadStride];
    __shared__ float        sm_lut[66];
    __shared__ float        sm_eA_i[kT];
    __shared__ float        sm_eA_j[kT];

    if (tid < 66) sm_lut[tid] = ldexpf(1.0f, -tid);
    if (tid < kT) {
        int idx_i = bi * kT + tid;
        int idx_j = bj * kT + tid;
        sm_eA_i[tid] = (idx_i < n) ? d_eA[idx_i] : 0.f;
        sm_eA_j[tid] = (idx_j < n) ? d_eA[idx_j] : 0.f;
    }

    // Per-pair union accumulators. eA[i] and eA[j] come from d_eA, so only
    // the union register-sum and zero-count are computed in the inner loop.
    // 4-way unrolled to expose ILP across the uchar4 lanes.
    float uSum0=0.f, uSum1=0.f, uSum2=0.f, uSum3=0.f;
    int   uZ0=0, uZ1=0, uZ2=0, uZ3=0;

    for (int sid = 0; sid < n_stripes; ++sid) {
        const int stripe_off = sid * stripe_size;
        const int half = kT * stripe_size;
        #pragma unroll
        for (int chunk = 0; chunk < (2 * kT * stripe_size) / (kThreads * 4); ++chunk) {
            int b_off = chunk * kThreads * 4 + tid * 4;
            std::uint8_t* dst;
            int sketch_idx, off_in_row;
            if (b_off < half) {
                int row = b_off / stripe_size;
                off_in_row = b_off % stripe_size;
                sketch_idx = bi * kT + row;
                dst = &sm_A[row][off_in_row];
            } else {
                int b2 = b_off - half;
                int row = b2 / stripe_size;
                off_in_row = b2 % stripe_size;
                sketch_idx = bj * kT + row;
                dst = &sm_B[row][off_in_row];
            }
            if (sketch_idx < n) {
                *reinterpret_cast<std::uint32_t*>(dst) =
                    *reinterpret_cast<const std::uint32_t*>(
                        sketches + (std::int64_t)sketch_idx * kBytesPerSketch
                                 + stripe_off + off_in_row);
            } else {
                *reinterpret_cast<std::uint32_t*>(dst) = 0;
            }
        }
        __syncthreads();
        if (valid) {
            const uchar4* pA = reinterpret_cast<const uchar4*>(&sm_A[ti][0]);
            const uchar4* pB = reinterpret_cast<const uchar4*>(&sm_B[tj][0]);
            #pragma unroll 4
            for (int r4 = 0; r4 < stripe_size / 4; ++r4) {
                uchar4 a = pA[r4]; uchar4 b = pB[r4];
                unsigned char ua = a.x > b.x ? a.x : b.x;
                unsigned char ub = a.y > b.y ? a.y : b.y;
                unsigned char uc = a.z > b.z ? a.z : b.z;
                unsigned char ud = a.w > b.w ? a.w : b.w;
                uSum0 += sm_lut[ua]; uSum1 += sm_lut[ub]; uSum2 += sm_lut[uc]; uSum3 += sm_lut[ud];
                uZ0 += (ua==0);  uZ1 += (ub==0);  uZ2 += (uc==0);  uZ3 += (ud==0);
            }
        }
        __syncthreads();
    }

    if (valid) {
        float uSum = (uSum0+uSum1)+(uSum2+uSum3);
        int   uZ   = uZ0+uZ1+uZ2+uZ3;
        const float m = (float)kRegisters;
        const float alpha_mm = (0.7213f / (1.0f + 1.079f / m)) * m * m;
        float eU = alpha_mm / uSum;
        if (eU <= 2.5f*m && uZ != 0) eU = m * logf(m / (float)uZ);
        float eA = sm_eA_i[ti];
        float eB = sm_eA_j[tj];
        std::int64_t k_out = (std::int64_t)gi * (2*(std::int64_t)n - gi - 1) / 2
                           + (gj - gi - 1);
        exact_j[k_out] = (eA + eB - eU) / eU;
    }
}

// =============================================================================
//  Device-resident driver. Caller supplies device pointers; this routine
//  precomputes per-sketch cardinalities into a scratch buffer, runs the
//  pairwise kernel, and frees the scratch.
// =============================================================================
void run_pairwise_exact(const std::uint8_t* d_sketches_packed,
                        int n,
                        float* d_jaccards_out) {
    float* d_eA = nullptr;
    CUDA_CHECK(cudaMalloc(&d_eA, (size_t)n * sizeof(float)));
    precompute_cardinality_kernel<<<n, 256>>>(d_sketches_packed, n, d_eA);
    CUDA_CHECK_LAST();

    int nb = (n + kT - 1) / kT;
    dim3 grid(nb, nb);
    exact_kernel<<<grid, kThreads>>>(d_sketches_packed, n, d_eA, d_jaccards_out);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFree(d_eA));
}

// =============================================================================
//  Host-resident convenience wrapper (used by CLI)
// =============================================================================
std::vector<float> compute_pairwise_jaccard_exact(
    const std::uint8_t* h_sketches_packed,
    int n) {
    if (n < 2) return {};
    const std::int64_t n_pairs = (std::int64_t)n * (n - 1) / 2;

    std::uint8_t* d_sk = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sk, (size_t)n * kBytesPerSketch));
    CUDA_CHECK(cudaMalloc(&d_out, n_pairs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sk, h_sketches_packed,
                          (size_t)n * kBytesPerSketch,
                          cudaMemcpyHostToDevice));

    run_pairwise_exact(d_sk, n, d_out);

    std::vector<float> h_out(n_pairs);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_pairs * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_sk));
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

} // namespace pairwise
} // namespace cuhll

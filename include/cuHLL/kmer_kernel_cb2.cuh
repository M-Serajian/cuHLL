#pragma once
// .cb2 path k-mer extraction kernel. Same stripe-per-thread layout as
// kmer_extract_kernel (FASTA path); the only differences are the read
// primitives: 2 bits per base from a packed byte array, 1 bit per base from
// the N-mask. Everything downstream (ntHash state, canonical min, cuco
// device-ref insert) is shared with the FASTA path.

#include "cuHLL/common.hpp"
#include "cuHLL/kmer_kernel.cuh"
#include "cuHLL/nthash.cuh"

#include <cstdint>

namespace cuhll {

__global__ void __launch_bounds__(kKmerExtractBlockSize)
kmer_extract_kernel_cb2(const std::uint8_t* __restrict__ packed,
                        const std::uint8_t* __restrict__ mask,
                        std::int64_t n_bases,
                        int k,
                        int stripe_len,
                        SketchHllRef ref,
                        bool canonical);

// Host-side launcher. Uses the same cached occupancy query as the FASTA
// launcher; records the decision to stderr on first call.
void launch_kmer_extract_cb2(const std::uint8_t* d_packed,
                             const std::uint8_t* d_mask,
                             std::int64_t n_bases,
                             int k,
                             SketchHllRef ref,
                             cudaStream_t stream,
                             bool canonical = true,
                             int* blocks_per_sm_out = nullptr);

} // namespace cuhll

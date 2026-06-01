#pragma once
// All-pairs HLL Jaccard between previously-sketched genomes.
//
// API contract:
//   Input:  N sketches, each packed as kBytesPerSketch (= kRegisters) bytes
//           of 8-bit register values in [0, 65]. Callers convert cuco's
//           native int32 register arrays via pack_registers() below.
//   Output: dense upper-triangular Jaccard vector of length N*(N-1)/2.
//           For pair (i, j) with i < j the linear index is
//             i * (2*N - i - 1) / 2 + (j - i - 1).
//
// Every pair gets a full kRegisters-element Jaccard — there is no filter
// stage or thresholded pruning. The driver precomputes each sketch's
// cardinality estimate once and reuses it across the N-1 pairs that
// involve that sketch, so the inner loop accumulates only the union
// cardinality per pair.

#include "cuHLL/common/common.hpp"

#include <cstdint>
#include <vector>
#include <string>

namespace cuhll {
namespace pairwise {

constexpr int kPrecision      = 14;
constexpr int kRegisters      = 1 << kPrecision;   // 16384 registers
constexpr int kBytesPerSketch = kRegisters;        // one byte per register

// Convert one sketch's int32 register array (cuco's native width) to the
// 8-bit packed form the pairwise kernel expects. Lossless: cuco's
// HyperLogLog never stores values above ~65.
void pack_registers(const std::uint32_t* src_int32,
                    std::uint8_t* dst_uint8,
                    int n_registers = kRegisters);

// Device-resident driver. The caller supplies device pointers; this routine
// allocates a small scratch buffer for the per-sketch cardinality cache,
// launches the kernels, and frees the scratch before returning.
//   d_sketches_packed : N x kBytesPerSketch bytes on the device.
//   n                 : number of sketches.
//   d_jaccards_out    : N*(N-1)/2 floats on the device.
void run_pairwise_exact(const std::uint8_t* d_sketches_packed,
                        int n,
                        float* d_jaccards_out);

// Host-resident convenience wrapper. Allocates device buffers, copies the
// sketches in, runs the pairwise kernel, copies the results back, and
// returns them as a std::vector. Convenient for one-shot use and tests;
// for tight loops that already manage device memory, call run_pairwise_exact
// directly.
std::vector<float> compute_pairwise_jaccard_exact(
    const std::uint8_t* h_sketches_packed,
    int n);

} // namespace pairwise
} // namespace cuhll

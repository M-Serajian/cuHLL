#pragma once
// CUDA-only internal header. Do NOT include from pure-C++ TUs (it pulls in
// cuco/cuda headers).
//
// Completes the `SketchImpl` forward-declared by sketch.hpp and gives CUDA
// TUs (sketch.cu, pipeline.cu, tests) direct access to the underlying
// cuco::hyperloglog object for device-ref extraction and merge calls.

#include "cuHLL/sketch.hpp"

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog.cuh>

#include <cstdint>

// ntHash output is uniform per k-mer, but canonical = min(fwd, rc) is NOT
// uniformity-preserving: it biases the top bits toward 0, which breaks HLL's
// register-selection (top p bits pick the register). xxhash_64 applied after
// min(fwd, rc) restores the uniform distribution HLL requires, at ~10 ns per
// insert on the L4 — negligible next to kernel wall time. See README for the
// empirical confirmation (~4.5x under-estimate with identity_hash, 1.18% error
// with xxhash_64 on chr19@k=31 vs KMC3).

namespace cuhll {

struct SketchImpl {
    using HasherT = cuco::xxhash_64<std::uint64_t>;
    using SketchT = cuco::hyperloglog<std::uint64_t, cuda::thread_scope_device, HasherT>;
    using RefT    = typename SketchT::template ref_type<>;

    SketchT sketch;

    explicit SketchImpl(int precision)
        : sketch{cuco::precision{precision}} {}
};

} // namespace cuhll

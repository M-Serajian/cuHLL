#pragma once
// CUDA-only internal header. Do not include from .cpp translation units —
// it pulls in cuco and CUDA headers.
//
// Completes the `SketchImpl` forward-declared by sketch.hpp so CUDA TUs
// (the kernel launchers, the pipeline, the tests) can reach the underlying
// cuco::hyperloglog and obtain its device reference for kernel arguments.
//
// Hash rationale (mirrors sketch.hpp): ntHash output is uniform, but the
// canonical = min(forward, reverse-complement) operation biases the top
// bits toward 0, which breaks HLL's register selection (the top p bits
// select the register). xxhash_64 applied after canonical-min restores the
// uniform distribution.

#include "cuHLL/sketch/sketch.hpp"

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog.cuh>

#include <cstdint>

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

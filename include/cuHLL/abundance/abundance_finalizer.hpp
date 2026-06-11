#pragma once
// abundance_finalizer.hpp — the uniform sampling hash for bottom-k.
//
// Correction A (Phase 0): bottom-k MUST sample on the uniform finalizer
// xxhash_64(canonical_ntHash), NOT the raw canonical ntHash. canonical =
// min(fwd, rc) biases the top bits toward 0, and bottom-k selects on magnitude
// (top bits) -> a raw-canonical sample is biased. xxhash_64 restores
// uniformity. This is the SAME hash cuco's hyperloglog applies internally for
// register selection (sketch_internal.cuh), so both sides of every comparison
// sample on the identical bit pattern.
//
// We call cuco's own XXHash_64 (host+device, constexpr) so the CPU reference
// and the future GPU sidecar are bit-identical by construction rather than by
// a hand-reimplementation that could drift.

#include <cuco/detail/hash_functions/xxhash.cuh>

#include <cstdint>

namespace cuhll::abundance {

// Default-seed (seed=0) xxhash_64 over the 8 bytes of the canonical key,
// matching cuco::xxhash_64<std::uint64_t> as used by the HLL.
inline std::uint64_t finalize(std::uint64_t canonical) {
    return cuco::detail::XXHash_64<std::uint64_t>{}(canonical);
}

}  // namespace cuhll::abundance

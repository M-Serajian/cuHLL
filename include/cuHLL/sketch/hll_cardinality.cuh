#pragma once
// Standalone HLL cardinality estimator that matches cuco's finalizer.
//
// cuco's Sketch::estimate() reduces an HLL register array to a cardinality
// by computing
//   z = sum_i 2^(-register_i)        // over all m = 2^p registers
//   v = count of register_i == 0
// and passing (z, v) to cuco::hyperloglog_ns::detail::finalizer(p), which
// applies the HyperLogLog++ linear-counting + bias correction.
//
// Calling the same finalizer here gives bit-identical cardinalities to
// Sketch::estimate() for any register array, which is useful in tests and
// in code paths (like the pairwise driver) that work directly on register
// bytes without owning a Sketch object.

#include <cstdint>
#include <cuco/detail/hyperloglog/finalizer.cuh>

namespace cuhll {

// Host-callable estimator. `registers` must point to 2^precision_p values
// accessible on the host.
inline std::uint64_t estimate_hll_registers_host(
        const std::uint32_t* registers, std::uint32_t precision_p) {
    const std::size_t m = static_cast<std::size_t>(1ULL) << precision_p;
    double z = 0.0;
    int    v = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const std::uint32_t r = registers[i];
        z += 1.0 / static_cast<double>(1ULL << r);
        if (r == 0) ++v;
    }
    cuco::hyperloglog_ns::detail::finalizer f(static_cast<int>(precision_p));
    return static_cast<std::uint64_t>(f(z, v));
}

} // namespace cuhll

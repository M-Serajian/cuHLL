#pragma once
// Device-side HLL cardinality estimator matching cuco's finalizer.
//
// cuco's `Sketch::estimate()` ends with a call to
// `cuco::hyperloglog_ns::detail::finalizer(p)(z, v)` where
//   z = Σ 2^(-register_i)   (sum over m registers)
//   v = count of register_i == 0
// and the finalizer applies HLL++ linear-counting + bias correction.
//
// Reusing that finalizer from our own kernels guarantees bit-identical
// output with Sketch::estimate() for any given register array.

#include <cstdint>
#include <cuco/detail/hyperloglog/finalizer.cuh>

namespace cuhll {

// Host-callable version: pure loop + finalizer. For tests / round-trip
// checks. Expects `registers` to be accessible on host.
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

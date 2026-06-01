#pragma once
// RAII NVTX range markers in a cuHLL-specific NVTX domain.
//
// Macros:
//   CUHLL_NVTX_RANGE(label)   — open a scoped range that closes at end of scope
//   CUHLL_NVTX_MARK(label)    — emit a point marker
//
// Both compile to no-ops when CUHLL_NVTX is undefined, so NVTX is optional
// at build time.

#ifdef CUHLL_NVTX
#include <nvtx3/nvtx3.hpp>

namespace cuhll {
struct nvtx_domain {
    static constexpr char const* name = "cuhll";
};
}  // namespace cuhll

#define CUHLL_NVTX_RANGE(label) \
    ::nvtx3::scoped_range_in<::cuhll::nvtx_domain> \
        _cuhll_nvtx_range_##__LINE__ { (label) }

#define CUHLL_NVTX_MARK(label) \
    ::nvtx3::mark_in<::cuhll::nvtx_domain>(label)
#else
#define CUHLL_NVTX_RANGE(label) ((void)0)
#define CUHLL_NVTX_MARK(label)  ((void)0)
#endif

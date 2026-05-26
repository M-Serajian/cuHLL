#pragma once
// RAII NVTX range markers, scoped to a cuhll-specific domain so traces
// don't mix with whatever cuco / CCCL emit. The whole thing compiles
// down to nothing when CUHLL_NVTX is undefined — useful for slim builds
// or platforms without the NVTX runtime.

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

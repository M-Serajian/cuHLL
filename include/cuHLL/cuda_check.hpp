#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cuhll {

inline void cuda_check_impl(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr,
                     "[cuHLL] CUDA error at %s:%d\n"
                     "  expression: %s\n"
                     "  reason:     %s\n",
                     file, line, expr, cudaGetErrorString(e));
        std::abort();
    }
}

} // namespace cuhll

#define CUDA_CHECK(expr) ::cuhll::cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() ::cuhll::cuda_check_impl(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__)

#pragma once
// CUDA error-checking macros.
//
//   CUDA_CHECK(expr)    — evaluates `expr`, aborts with diagnostic on
//                         non-success cudaError_t.
//   CUDA_CHECK_LAST()   — checks cudaGetLastError(). Use after kernel
//                         launches and other void-returning entry points.
//
// Both abort the process on failure; they are intended for unrecoverable
// errors at the boundary between cuHLL and the CUDA runtime.

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

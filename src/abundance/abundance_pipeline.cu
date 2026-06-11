// abundance_pipeline.cu — implementation of the additive k-mer abundance hook.
//
// Always compiled into cuhll_core; activated per-process via enabled() (the
// CUHLL_ABUNDANCE environment variable). Emits finalizers per stream, accumulates on
// the host, and reduces to tau at end-of-run. This is intentionally simple (a
// host sort) rather than the streaming bounded structure; it never touches the
// HLL sketch, so F0 is unaffected.

#include "cuHLL/abundance/abundance_pipeline.cuh"
#include "cuHLL/abundance/abundance_sketch.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace cuhll::abundance {
namespace {
constexpr std::uint64_t kU64Max = ~std::uint64_t(0);
std::mutex              g_mu;
std::vector<std::uint64_t> g_acc;
std::uint64_t           g_n_occ = 0;

std::uint64_t sample_from_env() {
    const char* s = std::getenv("CUHLL_ABUNDANCE_SAMPLE");
    if (s && *s) { char* e = nullptr; unsigned long long v = std::strtoull(s, &e, 10);
        if (v > 0) return (std::uint64_t)v; }
    return 50000;
}
}  // namespace

bool enabled() {
    // Opt-in for this process: the k-mer abundance sidecar runs only when
    // CUHLL_ABUNDANCE is set in the environment. Evaluated once.
    static const bool e = (std::getenv("CUHLL_ABUNDANCE") != nullptr);
    return e;
}

void reset() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_acc.clear();
    g_n_occ = 0;
}

void on_stream(const char* d_input, std::int64_t len, int k, bool canonical,
               cudaStream_t stream) {
    if (len < k) return;
    const std::int64_t n_pos = len - k + 1;
    std::uint64_t* d_out = nullptr;
    if (cudaMalloc(&d_out, (std::size_t)n_pos * sizeof(std::uint64_t)) != cudaSuccess)
        return;
    cudaMemsetAsync(d_out, 0xFF, (std::size_t)n_pos * sizeof(std::uint64_t), stream);
    cuhll::abundance::launch_emit_finalizers(d_input, len, k, canonical, d_out, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::uint64_t> h((std::size_t)n_pos);
    cudaMemcpy(h.data(), d_out, (std::size_t)n_pos * sizeof(std::uint64_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::lock_guard<std::mutex> lk(g_mu);
    for (std::uint64_t v : h)
        if (v != kU64Max) { g_acc.push_back(v); ++g_n_occ; }
}

std::uint64_t finalize_tau(std::uint64_t& n_distinct, std::uint64_t& n_occ,
                           std::uint64_t& sample_size) {
    std::lock_guard<std::mutex> lk(g_mu);
    sample_size = sample_from_env();
    n_occ = g_n_occ;
    std::sort(g_acc.begin(), g_acc.end());
    auto u = std::unique(g_acc.begin(), g_acc.end());
    n_distinct = (std::uint64_t)(u - g_acc.begin());
    if (n_distinct == 0) return 0;
    return (n_distinct >= sample_size) ? g_acc[sample_size - 1]
                                       : g_acc[n_distinct - 1];
}

}  // namespace cuhll::abundance

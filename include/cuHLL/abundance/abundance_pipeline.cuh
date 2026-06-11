#pragma once
// abundance_pipeline.cuh — hook used by the concurrent F0 pipeline to run the
// k-mer abundance sidecar. The hook is ADDITIVE: it runs the abundance emit
// kernel on the same device buffer the F0 kernel just consumed, and accumulates
// the xxhash_64(canonical) finalizers. It never touches the HLL sketch, so F0
// is unaffected. The abundance code is always compiled into cuhll_core but is
// OPT-IN at runtime via enabled(); when disabled the pipeline skips every hook
// below, so a plain F0 run does no extra work.

#include <cstdint>
#include <cuda_runtime.h>

namespace cuhll::abundance {

// Runtime opt-in. Returns true iff the k-mer abundance sidecar is requested
// for this process (environment variable CUHLL_ABUNDANCE is set). Evaluated once.
// When false, the pipeline must skip reset()/on_stream()/finalize_tau().
bool enabled();

// Clear the accumulator at the start of a run.
void reset();

// Run the abundance emit kernel on a per-stream device sequence (same d_input the
// F0 kernel used) and accumulate emitted finalizers. Synchronises `stream`.
void on_stream(const char* d_input, std::int64_t len, int k, bool canonical,
               cudaStream_t stream);

// Bottom-k tau over everything accumulated so far. Sample size S is read from
// $CUHLL_ABUNDANCE_SAMPLE (default 50000). Fills n_distinct / n_occ.
std::uint64_t finalize_tau(std::uint64_t& n_distinct, std::uint64_t& n_occ,
                           std::uint64_t& sample_size);

}  // namespace cuhll::abundance

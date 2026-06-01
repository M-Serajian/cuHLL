#pragma once
// Auto-tuned concurrent sketching pipeline for many-genome workloads.
//
// Public entry points:
//   survey_inputs            — stat the input files (no CUDA).
//   probe_host_gpu           — query CPU / RAM / GPU capabilities.
//   compute_auto_tune        — combine the survey and host/GPU probe into
//                              an orchestration plan.
//   compute_auto_tune_impl   — pure heuristic (no CUDA); for unit tests.
//   log_auto_tune            — print the plan to stderr.
//   sketch_per_genome_auto   — write one .hll per input plus the union.
//   union_estimate_auto      — return only the union cardinality.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuhll {

// Result of stat()ing every input.
struct InputSurvey {
    std::size_t n            = 0;
    std::size_t bytes_median = 0;
    std::size_t bytes_max    = 0;
    std::size_t bytes_total  = 0;
};

// Host + GPU capabilities snapshot. POD so tests can construct it without
// calling into CUDA.
struct HostGpuCaps {
    int         gpu_sm_count       = 0;
    std::size_t gpu_vram_free      = 0;
    std::size_t gpu_vram_total     = 0;
    int         cpu_count          = 0;
    std::size_t host_ram_available = 0;
};

// Output of the auto-tuner: the number of concurrent streams, reader and
// writer threads, and per-stream pinned buffer size to use for the run.
struct AutoTune {
    int         n_streams        = 0;
    int         n_readers        = 0;
    int         n_writers        = 0;
    std::size_t bytes_per_stream = 0;

    // Per-resource caps that the heuristic considered (reported for logs
    // and tests).
    std::size_t limit_vram            = 0;
    std::size_t limit_pinned          = 0;
    std::size_t limit_sm_concurrency  = 0;
    std::size_t limit_cpu             = 0;
    std::size_t limit_batch           = 0;
    const char* binding_limit         = "none";
};

// Stat every input. Throws std::runtime_error if a file cannot be stat'd.
InputSurvey survey_inputs(const std::vector<std::string>& paths);

// Query CPU count, available host RAM, and GPU device 0 SM count + VRAM.
HostGpuCaps probe_host_gpu();

// Heuristic only — no CUDA calls. Useful for unit tests.
AutoTune compute_auto_tune_impl(const InputSurvey& s, const HostGpuCaps& caps);

// Production wrapper: calls probe_host_gpu() and forwards to the heuristic.
AutoTune compute_auto_tune(const InputSurvey& s);

// Emit a one-block summary of the plan to stderr. Always logs (not gated
// on --verbose) so users can see what the pipeline decided.
void log_auto_tune(const AutoTune& at, const InputSurvey& s,
                   const HostGpuCaps& caps);

// Per-genome sketching. Reads each input, writes one <stem>.hll file under
// `output_dir`, prints "<display_path>\t<cardinality>" to stdout in input
// order, and returns the cardinality of the union of all inputs.
std::uint64_t sketch_per_genome_auto(
    const std::vector<std::string>& fasta_paths,
    const std::string& output_dir,
    int k,
    int precision_p,
    bool canonical = true);

// Union-only sketching. All streams insert into one shared HLL via
// atomicMax — no per-genome files and no merge step. Returns the union
// cardinality. Numerically equivalent to sketch_per_genome_auto's return
// value on the same input.
std::uint64_t union_estimate_auto(
    const std::vector<std::string>& fasta_paths,
    int k,
    int precision_p,
    bool canonical = true);

} // namespace cuhll

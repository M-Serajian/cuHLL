#pragma once
// Automatic concurrent per-genome sketching (Milestone j).
//
// Public entry points:
//   survey_inputs       — stat the input files (no CUDA).
//   compute_auto_tune   — turn the survey + hardware probe into a plan.
//   log_auto_tune       — one-shot stderr block describing the plan.
//   sketch_per_genome_auto — run the concurrent pipeline end-to-end.
//
// For users this is a black box: `cuhll --output-dir D --k 31 *.fasta`
// triggers it. No CLI flags or env vars control it (save one
// undocumented developer escape-hatch checked in main.cpp).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuhll {

// Result of stat()ing every input file.
struct InputSurvey {
    std::size_t n            = 0;
    std::size_t bytes_median = 0;
    std::size_t bytes_max    = 0;
    std::size_t bytes_total  = 0;
};

// Snapshot of the host + GPU capabilities used for the heuristic.
// Kept as a POD so unit tests can synthesize it without CUDA calls.
struct HostGpuCaps {
    int         gpu_sm_count      = 0;
    std::size_t gpu_vram_free     = 0;
    std::size_t gpu_vram_total    = 0;
    int         cpu_count         = 0;
    std::size_t host_ram_available= 0;
};

// Auto-tuned orchestration parameters.
struct AutoTune {
    int         n_streams        = 0;
    int         n_readers        = 0;
    int         n_writers        = 0;
    std::size_t bytes_per_stream = 0;

    // Limits as computed by the heuristic (for logging / unit tests).
    std::size_t limit_vram       = 0;
    std::size_t limit_pinned     = 0;
    std::size_t limit_sm_concurrency = 0;
    std::size_t limit_cpu        = 0;
    std::size_t limit_batch      = 0;
    const char* binding_limit    = "none";
};

// Stat the input set. Throws runtime_error if a file can't be stat'd.
InputSurvey survey_inputs(const std::vector<std::string>& paths);

// Pure heuristic (no CUDA). Exposed for unit tests.
AutoTune compute_auto_tune_impl(const InputSurvey& s, const HostGpuCaps& caps);

// Production wrapper that queries CUDA + /proc + cgroup + sched_getaffinity.
AutoTune compute_auto_tune(const InputSurvey& s);

// One stderr block describing the plan (always printed when the concurrent
// path fires; not gated on --verbose).
void log_auto_tune(const AutoTune& at, const InputSurvey& s,
                   const HostGpuCaps& caps);

// Probe host + GPU directly (matches compute_auto_tune's internals; exposed
// so log_auto_tune can print the same values that were fed to the heuristic).
HostGpuCaps probe_host_gpu();

// Per-genome sketching entry. Reads each FASTA path, produces one .hll per
// input under `output_dir`, and returns the UNION cardinality. Emits per-
// genome "<display_path>\t<est>" lines to stdout in input order.
std::uint64_t sketch_per_genome_auto(
    const std::vector<std::string>& fasta_paths,
    const std::string& output_dir,
    int k,
    int precision_p,
    bool canonical = true);

// Union-only entry. All streams hammer one shared sketch via atomicMax;
// no per-genome files, no merge step. Returns the same value as
// sketch_per_genome_auto on the same input.
std::uint64_t union_estimate_auto(
    const std::vector<std::string>& fasta_paths,
    int k,
    int precision_p,
    bool canonical = true);

} // namespace cuhll

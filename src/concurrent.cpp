#include "cuHLL/concurrent.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace cuhll {

// -----------------------------------------------------------------------------
// survey_inputs: stat each input path and compute N / median / max / total.
// -----------------------------------------------------------------------------
// Heuristic inflation when reading compressed inputs. Sequence content
// after decompressing + parsing is roughly the same magnitude as the
// compressed file size for FASTQ (4 lines/record × ~3× gzip ratio ×
// ~25% sequence fraction ≈ 1×), but for FASTA.gz it's ~3×. We pick a
// shared 4× upper bound so the per-stream pinned buffer is big enough
// for either format without inspecting the file's content.
constexpr std::size_t kGzipInflationFactor = 4;

static bool path_is_gzip(const std::string& p) {
    if (p.size() < 3) return false;
    auto e = p.substr(p.size() - 3);
    for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e == ".gz";
}

InputSurvey survey_inputs(const std::vector<std::string>& paths) {
    InputSurvey s;
    s.n = paths.size();
    if (paths.empty()) return s;

    std::vector<std::size_t> sizes;
    sizes.reserve(paths.size());
    for (const auto& p : paths) {
        struct stat st{};
        if (::stat(p.c_str(), &st) != 0) {
            throw std::runtime_error(std::string("cuHLL survey: cannot stat ")
                                     + p + ": " + std::strerror(errno));
        }
        std::size_t sz = static_cast<std::size_t>(st.st_size);
        if (path_is_gzip(p)) sz *= kGzipInflationFactor;
        sizes.push_back(sz);
        s.bytes_total += sz;
        if (sz > s.bytes_max) s.bytes_max = sz;
    }
    std::sort(sizes.begin(), sizes.end());
    s.bytes_median = sizes[sizes.size() / 2];
    return s;
}

// -----------------------------------------------------------------------------
// Host probes — no CUDA.
// -----------------------------------------------------------------------------
namespace {

std::size_t parse_meminfo_available() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::size_t kb = 0;
            // line format: "MemAvailable:    12345678 kB"
            const char* p = line.c_str();
            while (*p && !std::isdigit(static_cast<unsigned char>(*p))) ++p;
            while (*p && std::isdigit(static_cast<unsigned char>(*p))) {
                kb = kb * 10 + static_cast<std::size_t>(*p - '0');
                ++p;
            }
            return kb * 1024ULL;
        }
    }
    return 0;
}

std::size_t parse_cgroup_memory_limit() {
    // cgroup v2
    std::ifstream f2("/sys/fs/cgroup/memory.max");
    if (f2) {
        std::string s;
        f2 >> s;
        if (s == "max" || s.empty()) return std::numeric_limits<std::size_t>::max();
        try { return std::stoull(s); } catch (...) {}
    }
    // cgroup v1
    std::ifstream f1("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    if (f1) {
        std::size_t v = 0;
        f1 >> v;
        // cgroup v1 uses a huge sentinel (~9e18) for "no limit"
        if (v > (std::size_t{1} << 60)) return std::numeric_limits<std::size_t>::max();
        if (v > 0) return v;
    }
    return std::numeric_limits<std::size_t>::max();
}

int probe_cpu_affinity() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
    return CPU_COUNT(&set);
}

} // namespace

// -----------------------------------------------------------------------------
// probe_host_gpu — queries CPU + memory. GPU fields left zero; the CUDA
// wrapper in pipeline_concurrent.cu fills them in.
// -----------------------------------------------------------------------------
HostGpuCaps probe_host_gpu() {
    HostGpuCaps c;
    c.cpu_count = probe_cpu_affinity();
    if (c.cpu_count == 0) {
        c.cpu_count = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (c.cpu_count == 0) c.cpu_count = 1;

    const auto meminfo = parse_meminfo_available();
    const auto cglimit = parse_cgroup_memory_limit();
    c.host_ram_available = std::min(meminfo ? meminfo
                                             : std::numeric_limits<std::size_t>::max(),
                                    cglimit);
    if (c.host_ram_available == std::numeric_limits<std::size_t>::max()) {
        // Fallback: assume 8 GiB if nothing known.
        c.host_ram_available = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    // GPU fields filled by the CUDA wrapper.
    return c;
}

// -----------------------------------------------------------------------------
// compute_auto_tune_impl — the pure heuristic.
// -----------------------------------------------------------------------------
AutoTune compute_auto_tune_impl(const InputSurvey& s, const HostGpuCaps& caps) {
    AutoTune at{};

    // Per-stream worst-case buffer: largest input + 16 MiB slack.
    // Slack covers (a) filter-growth guard and (b) chunk alignment.
    const std::size_t slack = 16ULL * 1024 * 1024;
    at.bytes_per_stream = (s.bytes_max == 0 ? slack : s.bytes_max + slack);

    // Limit: fraction of free VRAM spendable on per-stream device buffers.
    // Use 50 %, 2× factor for allocator fragmentation / safety.
    if (at.bytes_per_stream > 0 && caps.gpu_vram_free > 0) {
        at.limit_vram = (caps.gpu_vram_free / 2ULL) / (at.bytes_per_stream * 2ULL);
    } else {
        at.limit_vram = 0;
    }

    // Limit: pinned host memory. 25% of host RAM budget.
    if (at.bytes_per_stream > 0 && caps.host_ram_available > 0) {
        at.limit_pinned = (caps.host_ram_available / 4ULL) / at.bytes_per_stream;
    } else {
        at.limit_pinned = 0;
    }

    // Limit: practical concurrent streams. Our extraction kernel targets
    // 100% occupancy at grid = 6 * n_sm, so multiple kernels in flight don't
    // run in parallel on the SMs — they serialize via the hardware scheduler.
    // The benefit of >1 stream is overlap of H2D/D2H (on the copy engine)
    // with kernel execution. 4–16 streams is enough to saturate the copy
    // engine on any realistic GPU; we scale slowly with SM count so
    // higher-end GPUs (B200 / H100 / GH200) still get more pipelining.
    {
        const std::size_t lo = 4ULL;
        const std::size_t hi = 16ULL;
        std::size_t sm_based = static_cast<std::size_t>(caps.gpu_sm_count) / 4ULL;
        if (sm_based < lo) sm_based = lo;
        if (sm_based > hi) sm_based = hi;
        at.limit_sm_concurrency = sm_based;
    }

    // Limit: CPU slots usable after carving out reader threads.
    at.limit_cpu = (caps.cpu_count >= 2)
        ? static_cast<std::size_t>(caps.cpu_count - 2)
        : 1ULL;
    if (at.limit_cpu < 2) at.limit_cpu = 2;

    // Limit: batch size (trivially can't exceed number of genomes).
    at.limit_batch = s.n;

    // Pick the minimum, clamp to [2, 64].
    std::size_t n = std::min({at.limit_vram, at.limit_pinned,
                              at.limit_sm_concurrency, at.limit_cpu,
                              at.limit_batch});
    if (n < 2) n = 2;
    if (n > 64) n = 64;
    // Final guard: never exceed batch size after the clamp.
    if (n > at.limit_batch && at.limit_batch > 0) n = at.limit_batch;
    at.n_streams = static_cast<int>(n);

    // Record which limit was binding. Ties: pick the first matching, ordered
    // by operational importance (batch > vram > pinned > sm > cpu).
    if (n == at.limit_batch)              at.binding_limit = "batch";
    else if (n == at.limit_vram)          at.binding_limit = "vram";
    else if (n == at.limit_pinned)        at.binding_limit = "pinned";
    else if (n == at.limit_sm_concurrency)at.binding_limit = "sm";
    else if (n == at.limit_cpu)           at.binding_limit = "cpu";
    else                                   at.binding_limit = "clamp";

    // Readers: cap at n_streams so we don't buffer ahead of the GPU.
    // Floor of 2 for minimum parallelism.
    int readers = caps.cpu_count - 2;
    if (readers < 2) readers = 2;
    if (readers > at.n_streams) readers = at.n_streams;
    at.n_readers = readers;

    // Writers: .hll files are tiny (~64 KB). One writer per ~8 CPUs is enough.
    int writers = std::max(1, caps.cpu_count / 8);
    at.n_writers = writers;

    return at;
}

// -----------------------------------------------------------------------------
// log_auto_tune — stderr block in a consistent format.
// -----------------------------------------------------------------------------
void log_auto_tune(const AutoTune& at, const InputSurvey& s,
                   const HostGpuCaps& caps) {
    std::fprintf(stderr,
        "[cuHLL auto-tune] inputs: N=%zu median=%.1f MB max=%.1f MB total=%.2f GB\n",
        s.n,
        static_cast<double>(s.bytes_median) / (1024.0 * 1024.0),
        static_cast<double>(s.bytes_max)    / (1024.0 * 1024.0),
        static_cast<double>(s.bytes_total)  / (1024.0 * 1024.0 * 1024.0));
    std::fprintf(stderr,
        "[cuHLL auto-tune] GPU: %d SMs, %.2f GB VRAM free of %.2f GB\n",
        caps.gpu_sm_count,
        static_cast<double>(caps.gpu_vram_free)  / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(caps.gpu_vram_total) / (1024.0 * 1024.0 * 1024.0));
    std::fprintf(stderr,
        "[cuHLL auto-tune] host: %d CPUs, %.2f GB RAM available (cgroup-aware)\n",
        caps.cpu_count,
        static_cast<double>(caps.host_ram_available) / (1024.0 * 1024.0 * 1024.0));
    std::fprintf(stderr,
        "[cuHLL auto-tune] per-stream buffer: %.1f MB\n",
        static_cast<double>(at.bytes_per_stream) / (1024.0 * 1024.0));
    std::fprintf(stderr,
        "[cuHLL auto-tune] limits: vram=%zu pinned=%zu sm=%zu cpu=%zu batch=%zu\n",
        at.limit_vram, at.limit_pinned, at.limit_sm_concurrency,
        at.limit_cpu, at.limit_batch);
    std::fprintf(stderr,
        "[cuHLL auto-tune] selected: n_streams=%d readers=%d writers=%d "
        "(bound by %s)\n",
        at.n_streams, at.n_readers, at.n_writers, at.binding_limit);
}

} // namespace cuhll

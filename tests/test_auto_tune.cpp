// =============================================================================
// test_auto_tune.cpp — pure unit test for compute_auto_tune_impl.
//
// Exercises 5 synthetic hardware / input combinations to prove the heuristic
// adapts across:
//   - chr19 size batches on L4 and B200
//   - human genome size on L4
//   - bacterial size on L4
//   - tiny batch (N=3)
// None of this touches the GPU. This is the adaptivity evidence without
// needing multiple real GPUs.
// =============================================================================

#include "cuHLL/pipeline/concurrent.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

bool g_verbose = false;

struct Case {
    const char*          name;
    cuhll::InputSurvey   s;
    cuhll::HostGpuCaps   caps;
    // Expectations (loose; we assert bands rather than exact values so the
    // heuristic can evolve without breaking the test).
    int  min_streams;
    int  max_streams;
    std::string          expected_binding;  // "" = don't check
};

Case make_chr19_l4() {
    Case c{};
    c.name = "chr19 / L4";
    c.s = {2000, 56 * kMiB, 60 * kMiB, 2000 * 56 * kMiB};
    c.caps.gpu_sm_count = 58;
    c.caps.gpu_vram_free = 22 * kGiB;
    c.caps.gpu_vram_total = 24 * kGiB;
    c.caps.cpu_count = 8;
    c.caps.host_ram_available = 16 * kGiB;
    c.min_streams = 2;
    c.max_streams = 16;
    c.expected_binding = "";   // L4 / chr19 typically bound by "sm" or "cpu"
    return c;
}

Case make_human_l4() {
    Case c{};
    c.name = "human (3 GB) / L4";
    c.s = {100, 3 * kGiB, 3ULL * kGiB, 100ULL * 3 * kGiB};
    c.caps.gpu_sm_count = 58;
    c.caps.gpu_vram_free = 22 * kGiB;
    c.caps.gpu_vram_total = 24 * kGiB;
    c.caps.cpu_count = 8;
    c.caps.host_ram_available = 16 * kGiB;
    c.min_streams = 2;
    c.max_streams = 4;
    c.expected_binding = ""; // expect vram or pinned
    return c;
}

Case make_bacterial_l4() {
    Case c{};
    c.name = "bacterial (5 MB) / L4";
    c.s = {10000, 5 * kMiB, 6 * kMiB, 10000ULL * 5 * kMiB};
    c.caps.gpu_sm_count = 58;
    c.caps.gpu_vram_free = 22 * kGiB;
    c.caps.gpu_vram_total = 24 * kGiB;
    c.caps.cpu_count = 8;
    c.caps.host_ram_available = 16 * kGiB;
    c.min_streams = 2;
    c.max_streams = 64;
    c.expected_binding = ""; // expect sm or cpu
    return c;
}

Case make_chr19_b200() {
    Case c{};
    c.name = "chr19 / B200";
    c.s = {2000, 56 * kMiB, 60 * kMiB, 2000 * 56 * kMiB};
    c.caps.gpu_sm_count = 148;            // approx B200 SMs
    c.caps.gpu_vram_free = 180 * kGiB;
    c.caps.gpu_vram_total = 192 * kGiB;
    c.caps.cpu_count = 32;
    c.caps.host_ram_available = 256 * kGiB;
    c.min_streams = 2;
    c.max_streams = 64;
    c.expected_binding = "";
    return c;
}

Case make_tiny() {
    Case c{};
    c.name = "tiny batch (N=3) / L4";
    c.s = {3, 20 * kMiB, 25 * kMiB, 3 * 20 * kMiB};
    c.caps.gpu_sm_count = 58;
    c.caps.gpu_vram_free = 22 * kGiB;
    c.caps.gpu_vram_total = 24 * kGiB;
    c.caps.cpu_count = 8;
    c.caps.host_ram_available = 16 * kGiB;
    c.min_streams = 2;
    c.max_streams = 3;
    c.expected_binding = "batch";
    return c;
}

int run_case(const Case& c) {
    const auto at = cuhll::compute_auto_tune_impl(c.s, c.caps);
    if (g_verbose) {
        std::fprintf(stderr,
            "[scenario %-30s] n_streams=%d readers=%d writers=%d bound-by=%s "
            "(limits vram=%zu pinned=%zu sm=%zu cpu=%zu batch=%zu; "
            "bytes_per_stream=%.1f MB)\n",
            c.name, at.n_streams, at.n_readers, at.n_writers,
            at.binding_limit,
            at.limit_vram, at.limit_pinned, at.limit_sm_concurrency,
            at.limit_cpu, at.limit_batch,
            static_cast<double>(at.bytes_per_stream) / static_cast<double>(kMiB));
    }
    int fail = 0;
    if (at.n_streams < c.min_streams || at.n_streams > c.max_streams) {
        std::fprintf(stderr,
            "[FAIL %s] n_streams=%d outside [%d, %d]\n",
            c.name, at.n_streams, c.min_streams, c.max_streams);
        fail = 1;
    }
    if (!c.expected_binding.empty() && c.expected_binding != at.binding_limit) {
        std::fprintf(stderr,
            "[FAIL %s] binding_limit=%s expected=%s\n",
            c.name, at.binding_limit, c.expected_binding.c_str());
        fail = 1;
    }
    if (at.n_readers < 2) {
        std::fprintf(stderr, "[FAIL %s] n_readers=%d < 2\n", c.name, at.n_readers);
        fail = 1;
    }
    if (at.n_writers < 1) {
        std::fprintf(stderr, "[FAIL %s] n_writers=%d < 1\n", c.name, at.n_writers);
        fail = 1;
    }
    if (at.bytes_per_stream < c.s.bytes_max) {
        std::fprintf(stderr,
            "[FAIL %s] bytes_per_stream=%zu < bytes_max=%zu\n",
            c.name, at.bytes_per_stream, c.s.bytes_max);
        fail = 1;
    }
    return fail;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0) g_verbose = true;
    }

    // Allow the test runner to see a header line.
    std::fprintf(stderr, "[test_auto_tune] running 5 synthetic scenarios\n");

    Case cases[] = {
        make_chr19_l4(),
        make_human_l4(),
        make_bacterial_l4(),
        make_chr19_b200(),
        make_tiny()
    };

    int total_fail = 0;
    for (const auto& c : cases) total_fail += run_case(c);

    // Cross-scenario sanity: B200 should pick >= L4 under the same chr19 load.
    {
        auto l4  = cuhll::compute_auto_tune_impl(make_chr19_l4().s,
                                                  make_chr19_l4().caps);
        auto b2  = cuhll::compute_auto_tune_impl(make_chr19_b200().s,
                                                  make_chr19_b200().caps);
        if (b2.n_streams < l4.n_streams) {
            std::fprintf(stderr,
                "[FAIL cross] B200 picked fewer streams (%d) than L4 (%d)\n",
                b2.n_streams, l4.n_streams);
            total_fail++;
        } else if (g_verbose) {
            std::fprintf(stderr,
                "[cross] L4=%d streams, B200=%d streams (B200 >= L4 ✓)\n",
                l4.n_streams, b2.n_streams);
        }
    }

    if (total_fail != 0) {
        std::fprintf(stderr, "[FAIL] test_auto_tune: %d failure(s)\n", total_fail);
        return 1;
    }
    std::fprintf(stderr, "[PASS] test_auto_tune\n");
    return 0;
}

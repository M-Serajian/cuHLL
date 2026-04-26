// pipeline_concurrent.cu — Milestone (j) automatic concurrent per-genome
// pipeline. One dispatcher thread owns all CUDA APIs; reader threads
// parse FASTAs on the CPU; writer threads persist .hll files. Per-stream
// sketches are CLEARED between genomes and their registers D2H'd via a
// pinned landing zone.

#include "cuHLL/concurrent.hpp"
#include "cuHLL/cuda_check.hpp"
#include "cuHLL/fasta.hpp"
#include "cuHLL/hll_file.hpp"
#include "cuHLL/kmer_kernel.cuh"
#include "cuHLL/sketch.hpp"
#include "cuHLL/sketch_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cuhll {

namespace {

// -------------------------------------------------------------------------
// Thread-safe bounded queue used for (a) input paths, (b) parsed genomes,
// (c) finished register arrays awaiting write.
// -------------------------------------------------------------------------
template <class T>
class MpmcQueue {
public:
    void close() {
        {
            std::lock_guard<std::mutex> g(mu_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    void push(T v) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&]{ return closed_ || q_.size() < cap_; });
        if (closed_) return;
        q_.push(std::move(v));
        lk.unlock();
        cv_not_empty_.notify_one();
    }

    // Returns nullopt when closed AND empty.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop();
        lk.unlock();
        cv_not_full_.notify_one();
        return v;
    }

    explicit MpmcQueue(std::size_t cap) : cap_(cap) {}

private:
    std::mutex              mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::queue<T>           q_;
    std::size_t             cap_;
    bool                    closed_ = false;
};

struct ParsedGenome {
    std::size_t       idx;
    std::string       display_path;
    std::string       bases;
};

struct WriteJob {
    std::size_t                    idx;
    std::string                    out_path;
    std::vector<std::uint32_t>     registers;
    std::uint32_t                  precision_p;
    std::uint32_t                  k;
};

// -------------------------------------------------------------------------
// Per-stream CUDA resources. One instance per n_streams.
// -------------------------------------------------------------------------
struct StreamSlot {
    cudaStream_t      stream  = nullptr;
    cudaEvent_t       done_event = nullptr;
    void*             d_input = nullptr;
    void*             h_input = nullptr;    // pinned
    std::uint32_t*    h_regs  = nullptr;    // pinned register landing zone
    std::unique_ptr<Sketch> sketch;         // owns device registers
    std::size_t       d_bytes  = 0;         // allocated device buffer size

    // In-flight state
    bool              busy = false;
    std::size_t       in_flight_idx = 0;
    std::string       in_flight_display;
    std::size_t       in_flight_bases = 0;
};

// -------------------------------------------------------------------------
// Probe GPU state (used by compute_auto_tune wrapper and log).
// -------------------------------------------------------------------------
void fill_gpu_caps(HostGpuCaps& c) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    c.gpu_sm_count   = prop.multiProcessorCount;
    c.gpu_vram_free  = free;
    c.gpu_vram_total = total;
}

// -------------------------------------------------------------------------
// Register max-reduction for CPU-side union accumulation.
// -------------------------------------------------------------------------
void merge_regs_max(std::uint32_t* dst, const std::uint32_t* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (src[i] > dst[i]) dst[i] = src[i];
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Public wrappers for host+GPU probe and auto-tune.
// -----------------------------------------------------------------------------
AutoTune compute_auto_tune(const InputSurvey& s) {
    HostGpuCaps caps = probe_host_gpu();
    fill_gpu_caps(caps);
    return compute_auto_tune_impl(s, caps);
}

// -----------------------------------------------------------------------------
// Concurrent per-genome pipeline.
// -----------------------------------------------------------------------------
std::uint64_t sketch_per_genome_auto(
        const std::vector<std::string>& fasta_paths,
        const std::string& output_dir,
        int k,
        int precision_p,
        bool canonical) {
    if (fasta_paths.empty()) return 0;

    // 1. Survey + auto-tune. Log once to stderr.
    InputSurvey survey = survey_inputs(fasta_paths);
    HostGpuCaps caps = probe_host_gpu();
    fill_gpu_caps(caps);
    AutoTune at = compute_auto_tune_impl(survey, caps);
    log_auto_tune(at, survey, caps);

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    const int        n_streams = at.n_streams;
    const std::size_t n_regs   = static_cast<std::size_t>(1ULL) << precision_p;
    const std::size_t reg_bytes = n_regs * sizeof(std::uint32_t);
    const std::size_t d_buf_bytes = at.bytes_per_stream;

    // 2. Allocate per-stream resources.
    std::vector<StreamSlot> slots(n_streams);
    for (int i = 0; i < n_streams; ++i) {
        auto& sl = slots[i];
        CUDA_CHECK(cudaStreamCreate(&sl.stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&sl.done_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&sl.d_input, d_buf_bytes));
        CUDA_CHECK(cudaHostAlloc(&sl.h_input, d_buf_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sl.h_regs),
                                  reg_bytes, cudaHostAllocDefault));
        sl.d_bytes = d_buf_bytes;
        sl.sketch = std::make_unique<Sketch>(precision_p, canonical);
    }

    // 3. Work queues.
    //    ready_q cap = 2 * n_streams so readers can stay one step ahead of
    //    the dispatcher; write_q unbounded is fine (items are tiny).
    struct InputItem { std::size_t idx; std::string path; };
    MpmcQueue<InputItem>    input_q(static_cast<std::size_t>(fasta_paths.size()));
    MpmcQueue<ParsedGenome> ready_q(static_cast<std::size_t>(std::max(4, n_streams * 2)));
    MpmcQueue<WriteJob>     write_q(static_cast<std::size_t>(fasta_paths.size()));

    for (std::size_t i = 0; i < fasta_paths.size(); ++i) {
        input_q.push({i, fasta_paths[i]});
    }
    input_q.close();

    // Per-input results (display path + estimate), pre-sized so we can fill
    // by index and print in order at the end.
    std::vector<std::uint64_t> per_genome_est(fasta_paths.size(), 0);

    // Union register accumulator (CPU side).
    std::vector<std::uint32_t> union_regs(n_regs, 0);

    // 4. Spawn readers.
    std::vector<std::thread> readers;
    readers.reserve(at.n_readers);
    std::atomic<bool> reader_fault{false};
    std::string      reader_error;
    std::mutex       reader_error_mu;
    for (int r = 0; r < at.n_readers; ++r) {
        readers.emplace_back([&]() {
            while (true) {
                auto item = input_q.pop();
                if (!item) break;
                try {
                    std::string bases = read_fasta_concat(item->path);
                    ready_q.push({item->idx, item->path, std::move(bases)});
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> g(reader_error_mu);
                    if (!reader_fault.exchange(true)) reader_error = e.what();
                    // Push an empty record so dispatcher's count stays correct.
                    ready_q.push({item->idx, item->path, std::string{}});
                }
            }
        });
    }

    // 5. Spawn writers.
    std::vector<std::thread> writers;
    writers.reserve(at.n_writers);
    for (int w = 0; w < at.n_writers; ++w) {
        writers.emplace_back([&]() {
            while (true) {
                auto job = write_q.pop();
                if (!job) break;
                try {
                    write_hll_registers(job->out_path, job->registers.data(),
                                        job->precision_p, job->k);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[cuHLL] write_hll failed for %s: %s\n",
                                 job->out_path.c_str(), e.what());
                }
            }
        });
    }

    // 6. Dispatcher: main thread drives CUDA. Two interleaved actions per
    //    iteration: drain completed streams, launch new work on free streams.
    std::size_t n_launched  = 0;
    std::size_t n_completed = 0;

    auto drain_slot = [&](int i) {
        auto& sl = slots[i];
        if (!sl.busy) return false;
        if (cudaEventQuery(sl.done_event) != cudaSuccess) return false;

        // Registers are in sl.h_regs. Copy out for the writer thread and
        // merge into union_regs.
        std::vector<std::uint32_t> regs_copy(sl.h_regs, sl.h_regs + n_regs);
        merge_regs_max(union_regs.data(), regs_copy.data(), n_regs);

        const std::string stem =
            std::filesystem::path(sl.in_flight_display).stem().string();
        WriteJob job;
        job.idx = sl.in_flight_idx;
        job.out_path = output_dir + "/" + stem + ".hll";
        job.registers = std::move(regs_copy);
        job.precision_p = static_cast<std::uint32_t>(precision_p);
        job.k = static_cast<std::uint32_t>(k);

        // Also compute per-genome estimate from the registers (cheap: a
        // single cudaMemcpy + cuco finalizer). We already have the Sketch's
        // current device state — estimate() reads its registers. Since the
        // D2H landed the same register bytes, est is stable.
        per_genome_est[sl.in_flight_idx] = sl.sketch->estimate();

        // Clear the sketch for reuse.
        sl.sketch->clear();

        write_q.push(std::move(job));

        sl.busy = false;
        ++n_completed;
        return true;
    };

    auto launch_slot = [&](int i, ParsedGenome& pg) {
        auto& sl = slots[i];
        const std::size_t n = pg.bases.size();
        if (n > sl.d_bytes) {
            throw std::runtime_error(
                "cuHLL concurrent: parsed genome exceeds per-stream buffer");
        }
        // H2D via pinned staging.
        if (n > 0) std::memcpy(sl.h_input, pg.bases.data(), n);
        CUDA_CHECK(cudaMemcpyAsync(sl.d_input, sl.h_input, n,
                                   cudaMemcpyHostToDevice, sl.stream));
        auto ref = sl.sketch->impl_ref().sketch.ref();
        if (n >= static_cast<std::size_t>(k)) {
            launch_kmer_extract(static_cast<const char*>(sl.d_input),
                                 static_cast<std::int64_t>(n),
                                 k, ref, sl.stream,
                                 sl.sketch->canonical());
        }
        // D2H registers for the .hll write.
        const void* d_regs = sl.sketch->impl_ref().sketch.sketch().data();
        CUDA_CHECK(cudaMemcpyAsync(sl.h_regs, d_regs, reg_bytes,
                                   cudaMemcpyDeviceToHost, sl.stream));
        CUDA_CHECK(cudaEventRecord(sl.done_event, sl.stream));

        sl.busy = true;
        sl.in_flight_idx     = pg.idx;
        sl.in_flight_display = std::move(pg.display_path);
        sl.in_flight_bases   = n;
        ++n_launched;
    };

    const std::size_t target = fasta_paths.size();

    while (n_completed < target) {
        // Drain anything completed.
        bool drained_any = false;
        for (int i = 0; i < n_streams; ++i) {
            if (drain_slot(i)) drained_any = true;
        }

        // If there's free slots AND parsed work, launch.
        int free_slot = -1;
        for (int i = 0; i < n_streams; ++i) {
            if (!slots[i].busy) { free_slot = i; break; }
        }
        if (free_slot >= 0 && n_launched < target) {
            auto pg = ready_q.pop();
            if (!pg) break; // readers all exited unexpectedly
            launch_slot(free_slot, *pg);
            continue;
        }

        // All slots busy; spin-wait briefly on events, then retry. Use
        // event-sync on the oldest busy slot to avoid busy-polling.
        if (!drained_any && free_slot < 0) {
            // Block until ANY event completes. cudaEventSynchronize on
            // slot 0 is safe — if 0 isn't done, it's at least waiting for
            // its own kernel, which is fine.
            int pick = 0;
            CUDA_CHECK(cudaEventSynchronize(slots[pick].done_event));
            drain_slot(pick);
        }
    }

    ready_q.close();
    write_q.close();
    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    if (reader_fault.load()) {
        throw std::runtime_error("cuHLL concurrent: reader failed: " + reader_error);
    }

    // 7. Emit per-genome estimates in input order.
    for (std::size_t i = 0; i < fasta_paths.size(); ++i) {
        std::printf("%s\t%llu\n", fasta_paths[i].c_str(),
                    static_cast<unsigned long long>(per_genome_est[i]));
    }

    // 8. Union estimate. Load the accumulated register array into a fresh
    //    Sketch on GPU and call estimate() — reuses cuco's finalizer.
    Sketch union_sketch(precision_p, canonical);
    union_sketch.load_registers_from_host(union_regs.data());
    const std::uint64_t union_est = union_sketch.estimate();

    // 9. Teardown.
    for (int i = 0; i < n_streams; ++i) {
        auto& sl = slots[i];
        if (sl.done_event) cudaEventDestroy(sl.done_event);
        if (sl.stream)     cudaStreamDestroy(sl.stream);
        if (sl.d_input)    cudaFree(sl.d_input);
        if (sl.h_input)    cudaFreeHost(sl.h_input);
        if (sl.h_regs)     cudaFreeHost(sl.h_regs);
    }

    return union_est;
}

} // namespace cuhll

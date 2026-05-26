// pipeline.cu — orchestrator for single-stream and multi-stream pipelines.

#include "cuHLL/cuda_check.hpp"
#include "cuHLL/fasta.hpp"
#include "cuHLL/kmer_kernel.cuh"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
#include "cuHLL/sketch_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuhll {

// =============================================================================
// Single-stream path (milestone c). Kept as a fallback / test entry point.
// =============================================================================
void sketch_sequence_single_stream(Sketch& sketch,
                                   const char* seq,
                                   std::size_t len,
                                   int k) {
    if (len == 0 || static_cast<std::int64_t>(len) < k) return;

    char* d_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seq, len));
    CUDA_CHECK(cudaMemcpy(d_seq, seq, len, cudaMemcpyHostToDevice));

    auto ref = sketch.impl_ref().sketch.ref();
    launch_kmer_extract(d_seq,
                        static_cast<std::int64_t>(len),
                        k,
                        ref,
                        /*stream=*/nullptr,
                        sketch.canonical());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_seq));
}

// =============================================================================
// Streaming pipeline (milestone d).
//
// Ring layout (3 slots):
//
//   slot i holds:
//     host_[i]    — `chunk_bytes` pinned host bytes  (cudaHostAlloc)
//     dev_[i]     — `chunk_bytes` device bytes        (cudaMalloc)
//     stream_[i]  — one CUDA stream, serializes H2D -> kernel -> event
//     event_[i]   — cudaEventDisableTiming, recorded after kernel launch;
//                   used by the CPU producer to know when it is safe to
//                   overwrite host_[i].
//
// The CPU producer (main thread) advances chunk_idx 0, 1, 2, 3, ...
//   slot = chunk_idx % 3
//   cudaEventSynchronize(event_[slot])   // on a never-recorded event this
//                                         // returns immediately, so the
//                                         // first pass through slots 0..2
//                                         // is not special-cased.
//   parse_chunk_into(host_[slot])
//   cudaMemcpyAsync(dev_[slot] <- host_[slot], stream_[slot])
//   launch_kmer_extract(dev_[slot], ..., stream_[slot])
//   cudaEventRecord(event_[slot], stream_[slot])
//
// When a chunk boundary falls inside a FASTA record (the common case for
// chr19 which is a single 55 MiB record), the parser copies the last
// (k-1) bytes of chunk i into the FRONT of chunk i+1. The kernel on chunk
// i+1 then emits k-mers starting at byte 0 of its buffer, which corresponds
// to the k-mer whose original position is L_i - (k-1) -- the first k-mer
// chunk i could not emit. No gaps, no double-insertion.
// =============================================================================

namespace {

class StreamingPipeline {
public:
    static constexpr int kNumSlots = 3;

    explicit StreamingPipeline(std::size_t chunk_bytes);
    ~StreamingPipeline();

    StreamingPipeline(const StreamingPipeline&)            = delete;
    StreamingPipeline& operator=(const StreamingPipeline&) = delete;

    char* host_slot(int slot) noexcept { return host_[slot]; }

    // Blocks the CPU until any kernel previously recorded on this slot's
    // event has completed. On an un-recorded event this returns immediately.
    void wait_slot_done(int slot) {
        CUDA_CHECK(cudaEventSynchronize(events_[slot]));
    }

    // Enqueue H2D + kernel + event record for `bytes` bytes already resident
    // in host_[slot]. `ref` is the cuco device ref, passed by value into the
    // kernel launcher (cuco ref is a trivially-copyable view).
    void submit(int slot, std::size_t bytes, int k, SketchHllRef ref,
                bool canonical) {
        if (bytes == 0) {
            // Still record the event so the ring sync logic stays uniform.
            CUDA_CHECK(cudaEventRecord(events_[slot], streams_[slot]));
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(dev_[slot], host_[slot], bytes,
                                   cudaMemcpyHostToDevice, streams_[slot]));
        launch_kmer_extract(dev_[slot],
                            static_cast<std::int64_t>(bytes),
                            k,
                            ref,
                            streams_[slot],
                            canonical);
        CUDA_CHECK(cudaEventRecord(events_[slot], streams_[slot]));
    }

    void wait_all() {
        for (int i = 0; i < kNumSlots; ++i) wait_slot_done(i);
    }

private:
    std::size_t  chunk_bytes_;
    char*        host_[kNumSlots]    = {nullptr, nullptr, nullptr};
    char*        dev_[kNumSlots]     = {nullptr, nullptr, nullptr};
    cudaStream_t streams_[kNumSlots] = {nullptr, nullptr, nullptr};
    cudaEvent_t  events_[kNumSlots]  = {nullptr, nullptr, nullptr};
};

StreamingPipeline::StreamingPipeline(std::size_t chunk_bytes)
    : chunk_bytes_(chunk_bytes) {
    for (int i = 0; i < kNumSlots; ++i) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_[i]),
                                 chunk_bytes_, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&dev_[i], chunk_bytes_));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
    }
}

StreamingPipeline::~StreamingPipeline() {
    // Best-effort teardown. CUDA_CHECK on destruction would abort on errors,
    // which is too aggressive for the RAII path; swallow errors here and
    // rely on CUDA_CHECK elsewhere to catch real faults.
    for (int i = 0; i < kNumSlots; ++i) {
        if (events_[i])  cudaEventDestroy(events_[i]);
        if (streams_[i]) cudaStreamDestroy(streams_[i]);
        if (dev_[i])     cudaFree(dev_[i]);
        if (host_[i])    cudaFreeHost(host_[i]);
    }
}

} // namespace

void sketch_sequences_streaming(Sketch& sketch,
                                const std::vector<std::string>& paths,
                                int k,
                                std::size_t chunk_mb) {
    if (paths.empty()) return;
    if (chunk_mb == 0) {
        throw std::invalid_argument("cuHLL: --chunk-mb must be >= 1");
    }

    const std::size_t chunk_bytes = chunk_mb * 1024ULL * 1024ULL;
    if (chunk_bytes <= static_cast<std::size_t>(k)) {
        throw std::invalid_argument(
            "cuHLL: --chunk-mb too small for this k (chunk_bytes must exceed k)");
    }

    StreamingPipeline pipe(chunk_bytes);
    auto ref = sketch.impl_ref().sketch.ref();
    const bool canonical = sketch.canonical();
    const std::size_t overlap = static_cast<std::size_t>(k - 1);

    std::uint64_t chunk_idx = 0;

    for (std::size_t file_i = 0; file_i < paths.size(); ++file_i) {
        FastaChunkReader reader(paths[file_i]);
        bool inject_file_break_N = (file_i > 0);

        while (true) {
            const int slot = static_cast<int>(chunk_idx % StreamingPipeline::kNumSlots);
            pipe.wait_slot_done(slot);

            char* const dst = pipe.host_slot(slot);
            std::size_t offset = 0;
            if (inject_file_break_N) {
                dst[0] = 'N';
                offset = 1;
                inject_file_break_N = false;
            }

            const std::size_t written = reader.next_chunk(dst + offset,
                                                          chunk_bytes - offset,
                                                          overlap);
            if (written == 0) {
                break; // EOF of this file
            }

            const std::size_t chunk_out = offset + written;
            pipe.submit(slot, chunk_out, k, ref, canonical);
            ++chunk_idx;
        }
    }

    pipe.wait_all();
}

} // namespace cuhll

// pipeline.cu — orchestrator for single-stream and multi-stream pipelines.

#include "cuHLL/cb2.hpp"
#include "cuHLL/cuda_check.hpp"
#include "cuHLL/fasta.hpp"
#include "cuHLL/kmer_kernel.cuh"
#include "cuHLL/kmer_kernel_cb2.cuh"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"
#include "cuHLL/sketch_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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

// =============================================================================
// Tier 1 — streaming .cb2 pipeline
//
// Per-slot state: packed host + mask host (pinned), packed device + mask
// device (plain), one CUDA stream, one event. Chunking is in **base-indices**,
// aligned to a multiple of 8 bases so that packed (2 bits/base) and mask
// (1 bit/base) slices are byte-aligned and trivially memcpy-able from the
// mmap'd file.
//
// Cross-chunk overlap is round_up(k-1, 8) bases — slightly more than the
// k-1 ntHash rolling overlap, but aligned to the mask byte boundary. The
// extra couple of overlapping k-mers are HLL-idempotent (double-insertion
// has no effect on the register max). This keeps the estimate bit-identical
// to the FASTA path, which is the test_cb2_correctness canary.
// =============================================================================
namespace {

constexpr int kCb2NumSlots = 3;

struct Cb2Pipe {
    std::size_t  chunk_bases = 0;
    std::size_t  packed_bytes_per_slot = 0;   // chunk_bases / 4
    std::size_t  mask_bytes_per_slot   = 0;   // chunk_bases / 8

    std::uint8_t*  host_packed[kCb2NumSlots] = {nullptr, nullptr, nullptr};
    std::uint8_t*  host_mask  [kCb2NumSlots] = {nullptr, nullptr, nullptr};
    std::uint8_t*  dev_packed [kCb2NumSlots] = {nullptr, nullptr, nullptr};
    std::uint8_t*  dev_mask   [kCb2NumSlots] = {nullptr, nullptr, nullptr};
    cudaStream_t streams[kCb2NumSlots]       = {nullptr, nullptr, nullptr};
    cudaEvent_t  events [kCb2NumSlots]       = {nullptr, nullptr, nullptr};

    explicit Cb2Pipe(std::size_t chunk_bases_in) {
        chunk_bases = chunk_bases_in;
        packed_bytes_per_slot = chunk_bases / 4;
        mask_bytes_per_slot   = chunk_bases / 8;
        for (int i = 0; i < kCb2NumSlots; ++i) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_packed[i]),
                                     packed_bytes_per_slot, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_mask[i]),
                                     mask_bytes_per_slot, cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc(&dev_packed[i], packed_bytes_per_slot));
            CUDA_CHECK(cudaMalloc(&dev_mask[i],   mask_bytes_per_slot));
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
        }
    }
    ~Cb2Pipe() {
        for (int i = 0; i < kCb2NumSlots; ++i) {
            if (events[i])      cudaEventDestroy(events[i]);
            if (streams[i])     cudaStreamDestroy(streams[i]);
            if (dev_packed[i])  cudaFree(dev_packed[i]);
            if (dev_mask[i])    cudaFree(dev_mask[i]);
            if (host_packed[i]) cudaFreeHost(host_packed[i]);
            if (host_mask[i])   cudaFreeHost(host_mask[i]);
        }
    }
    Cb2Pipe(const Cb2Pipe&) = delete;
    Cb2Pipe& operator=(const Cb2Pipe&) = delete;

    void wait_slot(int slot) { CUDA_CHECK(cudaEventSynchronize(events[slot])); }

    void submit(int slot, std::size_t chunk_bases_in_use, int k,
                SketchHllRef ref, bool canonical) {
        if (chunk_bases_in_use == 0) {
            CUDA_CHECK(cudaEventRecord(events[slot], streams[slot]));
            return;
        }
        const std::size_t p_bytes = (chunk_bases_in_use + 3) / 4;
        const std::size_t m_bytes = (chunk_bases_in_use + 7) / 8;
        CUDA_CHECK(cudaMemcpyAsync(dev_packed[slot], host_packed[slot], p_bytes,
                                   cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(dev_mask[slot],   host_mask[slot],   m_bytes,
                                   cudaMemcpyHostToDevice, streams[slot]));
        launch_kmer_extract_cb2(dev_packed[slot], dev_mask[slot],
                                static_cast<std::int64_t>(chunk_bases_in_use),
                                k, ref, streams[slot], canonical);
        CUDA_CHECK(cudaEventRecord(events[slot], streams[slot]));
    }

    void wait_all() {
        for (int i = 0; i < kCb2NumSlots; ++i) wait_slot(i);
    }
};

inline std::size_t round_up_mul(std::size_t x, std::size_t m) {
    return ((x + m - 1) / m) * m;
}

} // namespace

void sketch_sequences_cb2_streaming(Sketch& sketch,
                                    const std::vector<std::string>& cb2_paths,
                                    int k,
                                    std::size_t chunk_mb) {
    if (cb2_paths.empty()) return;
    if (chunk_mb == 0) {
        throw std::invalid_argument("cuHLL: --chunk-mb must be >= 1");
    }

    // chunk_bases: one Mi base == 1 MiB of logical sequence. Aligned to 8 bases
    // so packed and mask slices are byte-addressable.
    const std::size_t chunk_bases_raw = chunk_mb * 1024ULL * 1024ULL;
    const std::size_t chunk_bases     = (chunk_bases_raw / 8ULL) * 8ULL;
    if (chunk_bases <= static_cast<std::size_t>(k)) {
        throw std::invalid_argument(
            "cuHLL: --chunk-mb too small for this k (chunk_bases must exceed k)");
    }

    // Overlap: round_up(k-1, 8) so start offsets are mask-byte-aligned.
    const std::size_t overlap_bases = round_up_mul(static_cast<std::size_t>(k - 1), 8ULL);
    if (overlap_bases >= chunk_bases) {
        throw std::invalid_argument("cuHLL: chunk_bases too small vs k overlap");
    }

    Cb2Pipe pipe(chunk_bases);
    auto ref = sketch.impl_ref().sketch.ref();
    const bool canonical = sketch.canonical();

    std::uint64_t chunk_idx = 0;

    // Milestone (h) — pread-based chunking, no mmap, no page-cache buildup.
    //   * Open each .cb2 with O_RDONLY.
    //   * posix_fadvise(SEQUENTIAL) to hint prefetching.
    //   * pread header, then pread packed+mask slices directly into the
    //     per-slot pinned host buffers.
    //   * After each chunk, posix_fadvise(DONTNEED) over the just-read
    //     byte range so the kernel can drop those pages — the whole file
    //     is never allowed to accumulate in the page cache.
    //   * Close the fd when the file is fully processed.
    //
    // This is the fix for the "64 GB SLURM memory required" bug: at any
    // moment the only resident copies of the .cb2 data are the three
    // pinned ring slots (chunk_bases * 3/8 bytes each). For chunk_mb=32
    // that's ~36 MiB total resident, regardless of on-disk dataset size.
    auto pread_full = [](int fd, void* buf, std::size_t n, off_t off,
                         const char* what, const std::string& path) {
        std::size_t done = 0;
        char* p = static_cast<char*>(buf);
        while (done < n) {
            ssize_t r = ::pread(fd, p + done, n - done,
                                off + static_cast<off_t>(done));
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(
                    std::string("cuHLL cb2 pread(") + what + ") on " + path
                    + ": " + std::strerror(errno));
            }
            if (r == 0) break; // EOF
            done += static_cast<std::size_t>(r);
        }
        return done;
    };

    for (std::size_t file_i = 0; file_i < cb2_paths.size(); ++file_i) {
        const std::string& path = cb2_paths[file_i];

        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cuHLL: cannot open " + path + ": "
                                     + std::strerror(errno));
        }
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        Cb2Header hdr{};
        const std::size_t hdr_read = pread_full(fd, &hdr, sizeof(hdr), 0,
                                                "header", path);
        if (hdr_read != sizeof(hdr)) {
            ::close(fd);
            throw std::runtime_error("cuHLL: short header in " + path);
        }
        // Magic + version sanity (same invariant as Cb2Mmap).
        static const std::uint8_t kMagic[8] = {'C','B','2',0,0,0,0,1};
        if (std::memcmp(hdr.magic, kMagic, 8) != 0 || hdr.version != kCb2Version) {
            ::close(fd);
            throw std::runtime_error("cuHLL: bad .cb2 header in " + path);
        }

        const std::uint64_t total_bases  = hdr.n_bases;
        const std::uint64_t n_seq_bytes  = hdr.n_seq_bytes;
        const std::uint64_t n_mask_bytes = hdr.n_mask_bytes;
        if (total_bases == 0) { ::close(fd); continue; }

        const off_t packed_base_off = static_cast<off_t>(sizeof(Cb2Header));
        const off_t mask_base_off   = packed_base_off + static_cast<off_t>(n_seq_bytes);

        bool first_chunk_in_file = true;
        std::uint64_t start_base = 0;

        while (start_base < total_bases) {
            const int slot = static_cast<int>(chunk_idx % kCb2NumSlots);
            pipe.wait_slot(slot);

            const std::uint64_t chunk_end_base = std::min<std::uint64_t>(
                start_base + chunk_bases, total_bases);
            const std::uint64_t chunk_size = chunk_end_base - start_base;

            // start_base is a multiple of 8 ⇒ start/4 and start/8 are byte-aligned.
            const off_t p_src_off = packed_base_off + static_cast<off_t>(start_base >> 2);
            const off_t m_src_off = mask_base_off   + static_cast<off_t>(start_base >> 3);
            const std::size_t p_bytes      = (chunk_size + 3) / 4;
            const std::size_t m_bytes_calc = (chunk_size + 7) / 8;

            // Mask byte count in the source file may be 1 less than what
            // ceil(chunk_size/8) demands if the chunk straddles the last
            // byte of the file. Clip to what's actually available on disk.
            const std::uint64_t m_start_within_section = start_base >> 3;
            const std::size_t   m_src_avail            = (m_start_within_section < n_mask_bytes)
                ? static_cast<std::size_t>(n_mask_bytes - m_start_within_section)
                : 0;
            const std::size_t m_bytes = std::min(m_bytes_calc, m_src_avail);

            (void)pread_full(fd, pipe.host_packed[slot], p_bytes,
                             p_src_off, "packed", path);
            (void)pread_full(fd, pipe.host_mask  [slot], m_bytes,
                             m_src_off, "mask",   path);

            // Zero any tail past genuine data so the kernel sees 0s (no
            // spurious N-breaks) in the slop region of the pinned slot.
            if (m_bytes < pipe.mask_bytes_per_slot) {
                std::memset(pipe.host_mask[slot] + m_bytes, 0,
                            pipe.mask_bytes_per_slot - m_bytes);
            }

            // Cross-file window break: force a mask bit at chunk[0] on the
            // first chunk of every file after file 0.
            if (file_i > 0 && first_chunk_in_file) {
                pipe.host_mask[slot][0] |= 0x1u;
            }

            pipe.submit(slot, static_cast<std::size_t>(chunk_size), k, ref, canonical);

            // Release page cache for the bytes we just consumed. This is the
            // key hint that keeps the page cache from growing unboundedly.
            // We release the ADVANCE portion (what's exclusive to this chunk
            // vs the next overlap), not the overlap bytes — those get reused.
            if (chunk_end_base < total_bases) {
                const std::uint64_t advance_bases = chunk_size - overlap_bases;
                const off_t p_drop_off = p_src_off;
                const off_t m_drop_off = m_src_off;
                const std::size_t p_drop_bytes = advance_bases / 4;
                const std::size_t m_drop_bytes = advance_bases / 8;
                ::posix_fadvise(fd, p_drop_off, static_cast<off_t>(p_drop_bytes),
                                POSIX_FADV_DONTNEED);
                ::posix_fadvise(fd, m_drop_off, static_cast<off_t>(m_drop_bytes),
                                POSIX_FADV_DONTNEED);
            }

            ++chunk_idx;
            first_chunk_in_file = false;

            if (chunk_end_base == total_bases) break;
            start_base = chunk_end_base - overlap_bases;
        }

        // Final DONTNEED over the whole file just to be explicit.
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }

    pipe.wait_all();
}

} // namespace cuhll

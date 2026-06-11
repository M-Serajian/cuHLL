// abundance_stream.cu — DISK-STREAMING driver for the validated gpu_stream algorithm
// (via AbundanceStream). Reads each FASTA, feeds it, RELEASES it, moves on — host RAM
// holds at most one genome + the bounded sketch, never the whole corpus.
// gpu_stream itself is unchanged (AbundanceStream is its stateful form).
//
// Usage: stream_vram <manifest> [--validate]
//   default measure mode: disk-stream, print tau/n_distinct/total_kmers.
//   --validate: also load all seqs + run the two-pass oracle (gpu_tau/gpu_count)
//   and assert the disk-streamed tau + retained counts are BIT-IDENTICAL.
#include "cuHLL/abundance/abundance_sketch.cuh"
#include "cuHLL/abundance/abundance_estimator.hpp"
#include "cuHLL/io/fasta.hpp"
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static std::string read_retry(const std::string& path) {
    for (int attempt = 0; ; ++attempt) {
        try { return cuhll::read_fasta_concat(path); }
        catch (const std::exception& e) {
            if (attempt >= 8) throw;
            std::fprintf(stderr, "[retry %d] %s\n", attempt, e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// CHANGE (2): bounded prefetch. A reader thread reads genomes from disk IN ORDER
// into a FIFO of at most `max_depth` entries while the main thread feeds the GPU,
// overlapping NFS reads with compute. Host RAM stays bounded (~max_depth genomes
// in flight, NOT the whole corpus). Feeding order = manifest order, so the result
// is unchanged / bit-identical.
class Prefetcher {
public:
    Prefetcher(const std::vector<std::string>& paths, std::size_t max_depth)
        : paths_(paths), max_depth_(max_depth),
          reader_([this]{ read_loop(); }) {}
    ~Prefetcher() { if (reader_.joinable()) reader_.join(); }
    // Returns false when the stream is exhausted; rethrows a reader error.
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (err_) std::rethrow_exception(err_);
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front();
        cv_.notify_all();
        return true;
    }
private:
    void read_loop() {
        try {
            for (const auto& p : paths_) {
                std::string s = read_retry(p);
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this]{ return q_.size() < max_depth_; });
                q_.push_back(std::move(s));
                cv_.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(m_); err_ = std::current_exception();
        }
        std::lock_guard<std::mutex> lk(m_); done_ = true; cv_.notify_all();
    }
    const std::vector<std::string>& paths_;
    std::size_t max_depth_;
    std::deque<std::string> q_;
    std::mutex m_; std::condition_variable cv_;
    bool done_ = false; std::exception_ptr err_;
    std::thread reader_;
};

static int run(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: stream_vram <manifest> [--validate] [--abundance x y F0]\n"); return 2; }
    bool validate = false;
    std::uint64_t abundance_x = 0, abundance_y = 0, abundance_f0 = 0;  // --abundance x y F0
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--validate") validate = true;
        else if (a == "--abundance" && i + 3 < argc) {
            abundance_x  = std::strtoull(argv[i+1], nullptr, 10);
            abundance_y  = std::strtoull(argv[i+2], nullptr, 10);
            abundance_f0 = std::strtoull(argv[i+3], nullptr, 10);
            i += 3;
        }
    }

    const int k = 31;
    const std::uint64_t S = 50000, chunk = 200000;
    // Saturating counter cap MUST be > the abundance's upper edge so abundances are
    // resolved across the whole abundance (counts above the abundance saturate, not at a
    // tiny default). With --abundance x y, cap = y+1; otherwise a small default.
    const std::uint32_t cap = (abundance_f0 > 0) ? (std::uint32_t)(abundance_y + 1) : 9u;
    const std::uint64_t capacity = 2 * (S + chunk);

    std::vector<std::string> paths;
    { std::ifstream in(argv[1]); std::string p;
      while (std::getline(in, p)) if (!p.empty()) paths.push_back(p); }

    // --- DISK-STREAM with bounded PREFETCH: reader thread loads the next genome
    // while the GPU processes the current one; <= 2 genomes in flight. ---
    cuhll::abundance::AbundanceStream bs(k, true, S, cap, chunk, capacity);
    std::uint64_t total_kmers = 0;
    {
        Prefetcher pf(paths, /*max_depth=*/2);
        std::string seq;
        while (pf.next(seq)) {                              // next genome (prefetched)
            if ((int)seq.size() >= k) total_kmers += (std::uint64_t)seq.size() - k + 1;
            bs.process(seq);                                // feed; seq reused next iter
        }
    }                                                       // reader joined
    auto r = bs.finalize();
    std::printf("stream N=%zu : total_kmers=%llu tau=%llu n_distinct=%llu\n",
                paths.size(), (unsigned long long)total_kmers,
                (unsigned long long)r.tau, (unsigned long long)r.n_distinct);

    // Abundance-cardinality estimate from the streamed bottom-S sample, for the
    // direct comparison against KMC3's exact abundance count (split-delta 99% upper).
    if (abundance_f0 > 0) {
        cuhll::abundance::AbundanceConfig cfg;
        cfg.k = k; cfg.x = abundance_x; cfg.y = abundance_y; cfg.sample_size = S;
        cfg.z = 2.326347874;                                  // one-sided 99%
        cfg.f0_rel_err = 1.04 / std::sqrt((double)(1u << 14)); // HLL p=14 ~0.81%
        std::unordered_map<std::uint64_t, std::uint32_t> smap;
        for (auto& p : r.retained) smap[p.first] = p.second;
        cuhll::abundance::AbundanceEstimate be = cuhll::abundance::estimate_from_sample(abundance_f0, smap, cfg);
        std::printf("ABUNDANCE %llu<c<%llu  F0=%llu  sample_m=%llu  EST abundance=%.0f  "
                    "99%%upper=%.0f\n",
                    (unsigned long long)abundance_x, (unsigned long long)abundance_y,
                    (unsigned long long)abundance_f0, (unsigned long long)be.m,
                    be.abundance.point, be.abundance.upper);
    }

    if (validate) {
        // Load all (small panels only) and run the two-pass oracle.
        std::vector<std::string> seqs;
        for (const auto& p : paths) seqs.push_back(read_retry(p));
        auto g = cuhll::abundance::gpu_tau(seqs, k, true, S);
        auto cnt = cuhll::abundance::gpu_count(seqs, k, true, g.tau, cap, 2 * S + 4096);
        std::unordered_map<std::uint64_t, std::uint32_t> cmap;
        for (auto& pr : cnt) cmap[pr.first] = pr.second;
        std::uint64_t mism = (r.retained.size() != cmap.size()) ? 1 : 0;
        for (auto& pr : r.retained) { auto it = cmap.find(pr.first);
            if (it == cmap.end() || it->second != pr.second) ++mism; }
        const bool tau_ok = (r.tau == g.tau);
        std::printf("VALIDATE: oracle tau=%llu n=%llu | disk-stream tau=%llu n=%llu | "
                    "tau_match=%d retained_mismatch=%llu => %s\n",
                    (unsigned long long)g.tau, (unsigned long long)g.n_distinct,
                    (unsigned long long)r.tau, (unsigned long long)r.n_distinct,
                    (int)tau_ok, (unsigned long long)mism,
                    (tau_ok && mism == 0) ? "PASS" : "FAIL");
        return (tau_ok && mism == 0) ? 0 : 1;
    }
    return 0;
}

// Top-level guard: exit cleanly on any error (e.g. an NFS read that fails after
// retries) instead of letting the exception escape main -> std::terminate ->
// CORE DUMP. No more core files in the working directory.
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "stream_vram: fatal: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "stream_vram: fatal: unknown error\n");
        return 1;
    }
}

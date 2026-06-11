#pragma once
// abundance_sketch.cuh — GPU k-mer abundance sidecar (host API).
//
// Phase 2: bottom-k tau extraction over the XXHash_64(canonical) finalizer.
// Phase 3: fixed-capacity counting table for keys with finalizer <= tau.
//
// This is a SEPARATE module. It calls (never modifies) the read-only device
// inline hashes in cuHLL/kmer/nthash.cuh and uses cuco::XXHash_64 — the SAME
// finalizer the CPU oracle uses, so GPU and oracle agree bit-for-bit.
//
// The abundance kernel replicates kmer_extract_kernel's 2-runs-per-32-start-stripe
// cap exactly (validated against the oracle), so the k-mer occurrence multiset
// matches what cuHLL's F0 kernel sees.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuhll::abundance {

// Low-level: launch the cap-replicated emit kernel on a device sequence already
// resident in `d_seq`. `d_out` must have n_positions = (len-k+1) entries,
// preinitialised to ~0 (UINT64_MAX); emitted k-mer start positions receive the
// xxhash_64(canonical) finalizer. Async on `stream`.
void launch_emit_finalizers(const char* d_seq, std::int64_t len, int k,
                            bool canonical, std::uint64_t* d_out,
                            cudaStream_t stream);

struct TauResult {
    std::uint64_t tau        = 0;  // S-th smallest distinct finalizer
    std::uint64_t n_distinct = 0;  // distinct finalizers (== distinct canonical, modulo collisions)
    std::uint64_t n_occ      = 0;  // total emitted occurrences (capped stream)
    bool          full       = false; // true if n_distinct >= S
};

// Phase 2. Compute tau over a panel of host sequences (each already parsed by
// cuHLL's read_fasta_concat) on GPU device 0. Sample size S = bottom-k size.
TauResult gpu_tau(const std::vector<std::string>& seqs, int k, bool canonical,
                  std::uint64_t S);

// Phase 3. Count occurrences (capped stream) of every canonical key whose
// finalizer <= tau, saturating at `cap`, using a fixed-capacity table. Returns
// the retained (canonical_key -> saturating_count) pairs to the host.
std::vector<std::pair<std::uint64_t, std::uint32_t>>
gpu_count(const std::vector<std::string>& seqs, int k, bool canonical,
          std::uint64_t tau, std::uint32_t cap, std::uint64_t table_capacity);

// Phase 6. DEVICE-RESIDENT STREAMING bottom-k + counting in ONE pass. Maintains
// a bounded table (<= ~capacity entries) on device, processing the k-mer stream
// in chunks; tau is tightened BETWEEN chunks (a stale tau over-admits within a
// chunk — safe; under-admission impossible). NO full-stream collection. Must be
// BIT-IDENTICAL to gpu_tau (tau) + gpu_count (retained key->count) — verified in
// test_gpu_stream.
struct StreamResult {
    std::uint64_t tau        = 0;
    std::uint64_t n_distinct = 0;   // distinct keys retained (== min(S, total distinct))
    bool          full       = false;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> retained;  // key -> sat count
};
StreamResult gpu_stream(const std::vector<std::string>& seqs, int k, bool canonical,
                        std::uint64_t S, std::uint32_t cap,
                        std::uint64_t chunk_kmers, std::uint64_t table_capacity);

// Stateful streaming accumulator: the SAME algorithm as gpu_stream, but genomes
// are fed ONE AT A TIME (process()) so the caller can read a FASTA, feed it,
// release it, and move on — host RAM holds at most one genome + the bounded
// sketch, never the whole corpus. gpu_stream(vector) is exactly a thin wrapper
// that constructs a AbundanceStream, process()es each seq, and finalize()s — so its
// result is byte-identical to before (and to the two-pass oracle).
class AbundanceStream {
public:
    AbundanceStream(int k, bool canonical, std::uint64_t S, std::uint32_t cap,
               std::uint64_t chunk_kmers, std::uint64_t table_capacity);
    ~AbundanceStream();
    AbundanceStream(const AbundanceStream&) = delete;
    AbundanceStream& operator=(const AbundanceStream&) = delete;
    void         process(const std::string& seq);  // feed one genome
    StreamResult finalize();                        // bottom-S + tau
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cuhll::abundance

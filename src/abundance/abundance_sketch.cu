// abundance_sketch.cu — GPU k-mer abundance sidecar implementation.
//
// Calls (never modifies) the read-only cuHLL device hashes (nthash.cuh) and
// cuco::XXHash_64 (the same finalizer the CPU oracle uses).

#include "cuHLL/abundance/abundance_sketch.cuh"

#include "cuHLL/kmer/nthash.cuh"
#include <cuco/detail/hash_functions/xxhash.cuh>
#include <cuco/static_map.cuh>
#include <cuco/pair.cuh>

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace cuhll::abundance {

namespace {

constexpr int kStripe = 32;        // mirrors kKmerExtractStripe
constexpr int kBlock  = 128;
constexpr std::uint64_t kU64Max = ~std::uint64_t(0);

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e) \
            + " @" + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

// xxhash_64(canonical) — identical to cuco's HLL finalizer and to the CPU
// oracle's cuhll::abundance::finalize().
__device__ __forceinline__ std::uint64_t finalize_dev(std::uint64_t canon) {
    return cuco::detail::XXHash_64<std::uint64_t>{}(canon);
}

// Single source of truth for the capped traversal (mirrors enumerate_capped):
// one stripe of 32 k-mer starts; at most 2 ACGT runs; for every owned k-mer it
// calls emit(ks, canonical_key). Both the finalizer-emit and the counting
// kernels go through this, so they can never drift apart.
template <typename Emit>
__device__ __forceinline__ void for_each_capped_kmer(
        const char* __restrict__ seq, std::int64_t len, int k, int canonical,
        std::int64_t stripe, Emit emit) {
    const std::int64_t s_start = stripe * kStripe;
    if (s_start >= len) return;
    const std::int64_t n_positions = len - (std::int64_t)k + 1;
    const std::int64_t owned_end = min(s_start + kStripe, n_positions);
    if (!(s_start < len && owned_end > s_start)) return;
    const std::int64_t read_end = min(owned_end + (std::int64_t)k - 1, len);

    std::int64_t run_s[2], run_e[2];
    int n_runs = 0;
    std::int64_t rs = -1;
    for (std::int64_t i = s_start; i < read_end; ++i) {
        if (cuhll::nt_base_code(seq[i]) <= 3u) {
            if (rs < 0) rs = i;
        } else {
            if (rs >= 0 && (i - rs) >= k && n_runs < 2) {
                run_s[n_runs] = rs; run_e[n_runs] = i; ++n_runs;
            }
            rs = -1;
        }
    }
    if (rs >= 0 && (read_end - rs) >= k && n_runs < 2) {
        run_s[n_runs] = rs; run_e[n_runs] = read_end; ++n_runs;
    }

    for (int r = 0; r < n_runs; ++r) {
        const std::int64_t rs_l = run_s[r];
        const std::int64_t re_l = run_e[r];
        std::uint64_t fwd = 0, rc = 0;
        for (int j = 0; j < k; ++j) {
            const unsigned c = cuhll::nt_base_code(seq[rs_l + j]);
            fwd ^= cuhll::rotl64(cuhll::nt_seed(c), k - 1 - j);
            rc  ^= cuhll::rotl64(cuhll::nt_seed(cuhll::nt_complement_code(c)), j);
        }
        if (rs_l >= s_start && rs_l < owned_end)
            emit(rs_l, canonical ? cuhll::nt_canonical(fwd, rc) : fwd);
        for (std::int64_t i = rs_l + k; i < re_l; ++i) {
            const unsigned co = cuhll::nt_base_code(seq[i - k]);
            const unsigned ci = cuhll::nt_base_code(seq[i]);
            fwd = cuhll::nt_hash_roll_fwd(fwd, co, ci, k);
            rc  = cuhll::nt_hash_roll_rc (rc,  co, ci, k);
            const std::int64_t ks = i - k + 1;
            if (ks >= s_start && ks < owned_end)
                emit(ks, canonical ? cuhll::nt_canonical(fwd, rc) : fwd);
        }
    }
}

// One thread per 32-start stripe; writes the finalizer at the k-mer's start
// index; leaves non-emitted positions as kU64Max.
__global__ void emit_finalizers_kernel(const char* __restrict__ seq,
                                       std::int64_t len, int k, int canonical,
                                       std::uint64_t* __restrict__ out) {
    const std::int64_t stripe =
        (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for_each_capped_kmer(seq, len, k, canonical, stripe,
        [out] __device__ (std::int64_t ks, std::uint64_t key) {
            out[ks] = finalize_dev(key);
        });
}

struct NotMax {
    __device__ bool operator()(std::uint64_t x) const { return x != kU64Max; }
};

// Pass-2 filter: write the canonical KEY at its start index iff its finalizer
// <= tau (membership in the bottom-k sample); leave others as kU64Max.
__global__ void emit_keys_le_tau_kernel(const char* __restrict__ seq,
                                        std::int64_t len, int k, int canonical,
                                        std::uint64_t tau,
                                        std::uint64_t* __restrict__ out) {
    const std::int64_t stripe =
        (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for_each_capped_kmer(seq, len, k, canonical, stripe,
        [out, tau] __device__ (std::int64_t ks, std::uint64_t key) {
            if (finalize_dev(key) <= tau) out[ks] = key;
        });
}

// Saturating apply op for insert_or_apply. Op signature required by cuco:
// Op(cuda::atomic_ref<T,Scope>, T). Increments by `val` (always 1) but never
// past `cap`; once at cap it does no atomic store (the load-only fast path).
struct SaturatingAdd {
    std::uint32_t cap;
    template <typename AtomicRef, typename T>
    __device__ void operator()(AtomicRef ref, T const val) const {
        auto cur = ref.load(cuda::memory_order_relaxed);
        while (cur < cap) {
            const auto nxt = (cur + val > cap) ? cap : (cur + val);
            if (ref.compare_exchange_weak(cur, nxt, cuda::memory_order_relaxed))
                break;
        }
    }
};

struct MakeUnitPair {
    __device__ cuco::pair<std::uint64_t, std::uint32_t>
    operator()(std::uint64_t key) const { return {key, std::uint32_t{1}}; }
};

struct FinalizeOp {
    __device__ std::uint64_t operator()(std::uint64_t key) const {
        return finalize_dev(key);
    }
};

struct ZipToPair {
    __device__ cuco::pair<std::uint64_t, std::uint32_t>
    operator()(thrust::tuple<std::uint64_t, std::uint32_t> t) const {
        return {thrust::get<0>(t), thrust::get<1>(t)};
    }
};

// STREAMING emit: for the stripe range [stripe_begin, stripe_end), write the
// canonical key at its (chunk-local) start index iff its finalizer <= tau
// (the RELAXED, possibly stale, admission threshold). Non-admitted stay kU64Max.
__global__ void stream_emit_kernel(const char* __restrict__ seq,
                                   std::int64_t len, int k, int canonical,
                                   std::uint64_t tau,
                                   std::int64_t stripe_begin,
                                   std::int64_t stripe_end,
                                   std::int64_t base_pos,
                                   std::uint64_t* __restrict__ d_chunk) {
    const std::int64_t stripe =
        stripe_begin + (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (stripe >= stripe_end) return;
    for_each_capped_kmer(seq, len, k, canonical, stripe,
        [tau, base_pos, d_chunk] __device__ (std::int64_t ks, std::uint64_t key) {
            if (finalize_dev(key) <= tau) d_chunk[ks - base_pos] = key;
        });
}

}  // namespace

void launch_emit_finalizers(const char* d_seq, std::int64_t len, int k,
                            bool canonical, std::uint64_t* d_out,
                            cudaStream_t stream) {
    if (len < k) return;
    const std::int64_t n_stripes = (len + kStripe - 1) / kStripe;
    const int grid = (int)((n_stripes + kBlock - 1) / kBlock);
    emit_finalizers_kernel<<<grid, kBlock, 0, stream>>>(
        d_seq, len, k, canonical ? 1 : 0, d_out);
}

TauResult gpu_tau(const std::vector<std::string>& seqs, int k, bool canonical,
                  std::uint64_t S) {
    thrust::device_vector<std::uint64_t> all;
    for (const auto& s : seqs) {
        const std::int64_t len = (std::int64_t)s.size();
        if (len < k) continue;
        const std::int64_t n_pos = len - k + 1;

        thrust::device_vector<char> d_seq(s.begin(), s.end());
        thrust::device_vector<std::uint64_t> d_out(n_pos, kU64Max);

        launch_emit_finalizers(thrust::raw_pointer_cast(d_seq.data()), len, k,
                               canonical, thrust::raw_pointer_cast(d_out.data()),
                               /*stream=*/0);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        const std::size_t old = all.size();
        all.resize(old + n_pos);
        auto end = thrust::copy_if(d_out.begin(), d_out.end(),
                                   all.begin() + old, NotMax{});
        all.resize(end - all.begin());
    }

    TauResult r;
    r.n_occ = all.size();
    thrust::sort(all.begin(), all.end());
    auto u = thrust::unique(all.begin(), all.end());
    r.n_distinct = (std::uint64_t)(u - all.begin());
    r.full = r.n_distinct >= S;
    if (r.n_distinct == 0)      r.tau = 0;
    else if (r.full)            r.tau = all[S - 1];
    else                        r.tau = all[r.n_distinct - 1];
    return r;
}

std::vector<std::pair<std::uint64_t, std::uint32_t>>
gpu_count(const std::vector<std::string>& seqs, int k, bool canonical,
          std::uint64_t tau, std::uint32_t cap, std::uint64_t table_capacity) {
    // Pass 2: collect every OCCURRENCE of keys whose finalizer <= tau (the
    // bottom-k members). This set is small (~S distinct keys), so the bulk
    // count is cheap.
    thrust::device_vector<std::uint64_t> d_keys;
    for (const auto& s : seqs) {
        const std::int64_t len = (std::int64_t)s.size();
        if (len < k) continue;
        const std::int64_t n_pos = len - k + 1;
        thrust::device_vector<char> d_seq(s.begin(), s.end());
        thrust::device_vector<std::uint64_t> d_out(n_pos, kU64Max);
        const std::int64_t n_stripes = (len + kStripe - 1) / kStripe;
        const int grid = (int)((n_stripes + kBlock - 1) / kBlock);
        emit_keys_le_tau_kernel<<<grid, kBlock>>>(
            thrust::raw_pointer_cast(d_seq.data()), len, k, canonical ? 1 : 0,
            tau, thrust::raw_pointer_cast(d_out.data()));
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        const std::size_t old = d_keys.size();
        d_keys.resize(old + n_pos);
        auto end = thrust::copy_if(d_out.begin(), d_out.end(),
                                   d_keys.begin() + old, NotMax{});
        d_keys.resize(end - d_keys.begin());
    }

    // Fixed-capacity cuco table; insert-or-(saturating-)apply. No eviction —
    // membership is fixed by tau.
    using Map = cuco::static_map<std::uint64_t, std::uint32_t>;
    Map map{static_cast<std::size_t>(table_capacity),
            cuco::empty_key<std::uint64_t>{kU64Max},
            cuco::empty_value<std::uint32_t>{0u}};
    auto first = thrust::make_transform_iterator(d_keys.begin(), MakeUnitPair{});
    auto last  = thrust::make_transform_iterator(d_keys.end(),   MakeUnitPair{});
    map.insert_or_apply(first, last, SaturatingAdd{cap});

    const std::size_t n = map.size();
    thrust::device_vector<std::uint64_t> dk(n);
    thrust::device_vector<std::uint32_t> dv(n);
    map.retrieve_all(dk.begin(), dv.begin());

    std::vector<std::uint64_t> hk(n);
    std::vector<std::uint32_t> hv(n);
    thrust::copy(dk.begin(), dk.end(), hk.begin());
    thrust::copy(dv.begin(), dv.end(), hv.begin());
    std::vector<std::pair<std::uint64_t, std::uint32_t>> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {hk[i], hv[i]};
    return out;
}

namespace {
using AbundanceMap = cuco::static_map<std::uint64_t, std::uint32_t>;

// Sort the table by finalizer; set tau = m-th smallest (m = min(size,S)); if the
// table is over S, rebuild it to its bottom-S members (counts preserved). Keys
// with finalizer <= the final tau are never the ones dropped (their rank is
// <= S), so they survive every compaction with their full counts intact.
void compact_table(AbundanceMap& table, std::uint64_t S, std::uint32_t cap,
                   std::uint64_t& tau, bool rebuild) {
    const std::size_t n = table.size();
    if (n == 0) { tau = kU64Max; return; }
    thrust::device_vector<std::uint64_t> dk(n);
    thrust::device_vector<std::uint32_t> dv(n);
    table.retrieve_all(dk.begin(), dv.begin());
    thrust::device_vector<std::uint64_t> df(n);
    thrust::transform(dk.begin(), dk.end(), df.begin(), FinalizeOp{});
    auto vals = thrust::make_zip_iterator(thrust::make_tuple(dk.begin(), dv.begin()));
    thrust::sort_by_key(df.begin(), df.end(), vals);
    const std::size_t m = (n < S) ? n : (std::size_t)S;
    tau = df[m - 1];
    if (rebuild && n > S) {
        table.clear();
        auto pit = thrust::make_transform_iterator(
            thrust::make_zip_iterator(thrust::make_tuple(dk.begin(), dv.begin())),
            ZipToPair{});
        table.insert_or_apply(pit, pit + m, SaturatingAdd{cap});
    }
}
}  // namespace

// Stateful streaming accumulator. The per-genome chunk loop (process_seq) and
// the tail (finalize) are EXACTLY the body of the old gpu_stream — only the
// packaging changed, so behaviour is byte-identical.
struct AbundanceStream::Impl {
    int           k;
    bool          canonical;
    std::uint64_t S;
    std::uint32_t cap;
    std::int64_t  chunk_stripes;
    std::int64_t  chunk_cap;
    AbundanceMap       table;
    std::uint64_t tau = kU64Max;
    // CHANGE (1): d_chunk is double-buffered (2 buffers of chunk_cap). The emit
    // for chunk i+1 runs on emit_stream into one buffer while the table work for
    // chunk i runs on the default stream out of the other buffer — overlapping
    // the genome scan with the compaction. The table is ONLY ever touched by the
    // serial, ordered default-stream ops (insert/compact), so concurrency adds
    // no table race; emit only reads d_seq + a tau SNAPSHOT (kernel param by
    // value) and writes its own buffer.
    thrust::device_vector<std::uint64_t> d_chunk, d_cand;
    cudaStream_t emit_stream = nullptr;
    cudaEvent_t  evt[2]      = {nullptr, nullptr};

    Impl(int k_, bool canonical_, std::uint64_t S_, std::uint32_t cap_,
         std::uint64_t chunk_kmers, std::uint64_t table_capacity)
        : k(k_), canonical(canonical_), S(S_), cap(cap_),
          chunk_stripes(std::max<std::int64_t>(1, (std::int64_t)(chunk_kmers / kStripe))),
          chunk_cap(chunk_stripes * kStripe),
          table{static_cast<std::size_t>(table_capacity),
                cuco::empty_key<std::uint64_t>{kU64Max},
                cuco::empty_value<std::uint32_t>{0u}},
          d_chunk(2 * chunk_cap), d_cand(chunk_cap) {
        CUDA_OK(cudaStreamCreate(&emit_stream));
        CUDA_OK(cudaEventCreateWithFlags(&evt[0], cudaEventDisableTiming));
        CUDA_OK(cudaEventCreateWithFlags(&evt[1], cudaEventDisableTiming));
    }
    ~Impl() {
        if (evt[0]) cudaEventDestroy(evt[0]);
        if (evt[1]) cudaEventDestroy(evt[1]);
        if (emit_stream) cudaStreamDestroy(emit_stream);
    }

    // Launch emit for chunk `ci` into double-buffer slot `buf` on emit_stream,
    // using the CURRENT host `tau` snapshot (by value -> no read race vs the
    // default-stream compaction that may update `tau` concurrently). A staler
    // tau only over-admits (safe); under-admission is impossible.
    void launch_emit(std::int64_t ci, int buf, std::int64_t len,
                     const char* d_seq, std::int64_t n_stripes) {
        const std::int64_t sb = ci * chunk_stripes;
        const std::int64_t se = std::min(sb + chunk_stripes, n_stripes);
        const std::int64_t base = sb * kStripe;
        const std::int64_t clen = (se - sb) * kStripe;
        std::uint64_t* out = thrust::raw_pointer_cast(d_chunk.data()) + buf * chunk_cap;
        CUDA_OK(cudaMemsetAsync(out, 0xFF, (std::size_t)clen * sizeof(std::uint64_t),
                                emit_stream));
        const int nstr = (int)(se - sb);
        const int grid = (nstr + kBlock - 1) / kBlock;
        stream_emit_kernel<<<grid, kBlock, 0, emit_stream>>>(
            d_seq, len, k, canonical ? 1 : 0, tau, sb, se, base, out);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(evt[buf], emit_stream));
    }

    void process(const std::string& s) {
        const std::int64_t len = (std::int64_t)s.size();
        if (len < k) return;
        const std::int64_t n_stripes = (len + kStripe - 1) / kStripe;
        const std::int64_t n_chunks = (n_stripes + chunk_stripes - 1) / chunk_stripes;
        if (n_chunks <= 0) return;
        thrust::device_vector<char> d_seq(s.begin(), s.end());
        const char* d_seq_ptr = thrust::raw_pointer_cast(d_seq.data());

        launch_emit(0, 0, len, d_seq_ptr, n_stripes);   // prime the pipeline
        for (std::int64_t i = 0; i < n_chunks; ++i) {
            const int buf = (int)(i & 1);
            // Launch the NEXT chunk's emit so it overlaps this chunk's table work.
            if (i + 1 < n_chunks)
                launch_emit(i + 1, (int)((i + 1) & 1), len, d_seq_ptr, n_stripes);

            // --- table work for chunk i (default stream, UNCHANGED logic) ---
            const std::int64_t sb = i * chunk_stripes;
            const std::int64_t se = std::min(sb + chunk_stripes, n_stripes);
            const std::int64_t clen = (se - sb) * kStripe;
            std::uint64_t* cbuf = thrust::raw_pointer_cast(d_chunk.data()) + buf * chunk_cap;
            CUDA_OK(cudaStreamWaitEvent(/*default*/0, evt[buf], 0)); // wait emit(i)
            thrust::device_ptr<std::uint64_t> cb(cbuf);
            auto end = thrust::copy_if(cb, cb + clen, d_cand.begin(), NotMax{});
            const std::size_t ncand = end - d_cand.begin();
            if (ncand) {
                auto first = thrust::make_transform_iterator(d_cand.begin(), MakeUnitPair{});
                table.insert_or_apply(first, first + ncand, SaturatingAdd{cap});
            }
            if (table.size() > S) compact_table(table, S, cap, tau, /*rebuild=*/true);
        }
    }

    StreamResult finalize() {
        StreamResult r;
        compact_table(table, S, cap, tau, /*rebuild=*/false);  // sets tau exactly
        const std::size_t n = table.size();
        thrust::device_vector<std::uint64_t> dk(n);
        thrust::device_vector<std::uint32_t> dv(n);
        table.retrieve_all(dk.begin(), dv.begin());
        thrust::device_vector<std::uint64_t> df(n);
        thrust::transform(dk.begin(), dk.end(), df.begin(), FinalizeOp{});
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(dk.begin(), dv.begin()));
        thrust::sort_by_key(df.begin(), df.end(), vals);
        const std::size_t m = (n < S) ? n : (std::size_t)S;
        r.n_distinct = m;
        r.full = n >= S;
        r.tau = (m == 0) ? 0 : (std::uint64_t)df[m - 1];
        std::vector<std::uint64_t> hk(m);
        std::vector<std::uint32_t> hv(m);
        thrust::copy(dk.begin(), dk.begin() + m, hk.begin());
        thrust::copy(dv.begin(), dv.begin() + m, hv.begin());
        r.retained.resize(m);
        for (std::size_t i = 0; i < m; ++i) r.retained[i] = {hk[i], hv[i]};
        return r;
    }
};

AbundanceStream::AbundanceStream(int k, bool canonical, std::uint64_t S, std::uint32_t cap,
                       std::uint64_t chunk_kmers, std::uint64_t table_capacity)
    : p_(new Impl(k, canonical, S, cap, chunk_kmers, table_capacity)) {}
AbundanceStream::~AbundanceStream() = default;
void AbundanceStream::process(const std::string& seq) { p_->process(seq); }
StreamResult AbundanceStream::finalize() { return p_->finalize(); }

// Unchanged behaviour: construct a AbundanceStream, feed every seq, finalize.
StreamResult gpu_stream(const std::vector<std::string>& seqs, int k, bool canonical,
                        std::uint64_t S, std::uint32_t cap,
                        std::uint64_t chunk_kmers, std::uint64_t table_capacity) {
    AbundanceStream bs(k, canonical, S, cap, chunk_kmers, table_capacity);
    for (const auto& s : seqs) bs.process(s);
    return bs.finalize();
}

}  // namespace cuhll::abundance

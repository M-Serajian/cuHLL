// =============================================================================
// Milestone (b) test: validate cuHLL's ntHash implementation by cross-checking
// three independent paths on a fixed 100-base DNA test sequence (which includes
// an 'N' to exercise the window-reset path):
//
//   (1) GPU sliding-window           -> canonical hashes
//   (2) CPU sliding-window           -> canonical hashes
//   (3) CPU independent substring    -> canonical hashes
//       recompute (nt_hash_init_* on each valid k-mer separately; no rolling)
//
// (1) vs (2) catches device-vs-host compilation divergence.
// (2) vs (3) catches rolling-recurrence bugs by comparing against the simpler
// O(k) init formula over every valid k-mer — including across the N boundary.
//
// The first 10 canonical hashes at k=31 are written to
// bench/results/nthash_ground_truth.txt as a regression record.
// =============================================================================

#include "cuHLL/common/cuda_check.hpp"
#include "cuHLL/kmer/nthash.cuh"

#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// -----------------------------------------------------------------------------
// Test sequence: 50 ACGT bases, one N, 49 ACGT bases. Total 100 bases.
// Crafted so that with k=31:
//   - 20 k-mers fully on the left of the N
//   - 19 k-mers fully on the right of the N
// and with k=21:
//   - 30 on the left, 29 on the right.
// The bases are a mix rather than a simple "ACGT" repeat, so fwd != rc for
// most windows and the canonical min() is exercised.
// -----------------------------------------------------------------------------
static const char kSeq[] =
    "ACGTGCATGCATGCATATCGGCTAGCTAGCAGATCGATCGTACGTAGCTA"  // 50
    "N"                                                    //  1
    "GACTGACTAGCTAGCATCGATCGTAGCATGCATGCATAGATCGTAGCTA";   // 49
static constexpr int kSeqLen = static_cast<int>(sizeof(kSeq) - 1);
static_assert(kSeqLen == 100, "Test sequence must be exactly 100 bases");

// -----------------------------------------------------------------------------
// CPU sliding window (same algorithm the GPU kernel runs).
// -----------------------------------------------------------------------------
static void cpu_sliding(const char* seq, int len, int k, std::vector<std::uint64_t>& out) {
    out.clear();
    int valid = 0;
    std::uint64_t fwd = 0, rc = 0;
    for (int i = 0; i < len; ++i) {
        unsigned code = cuhll::nt_base_code(seq[i]);
        if (code > 3u) { valid = 0; continue; }
        ++valid;
        if (valid < k) continue;
        if (valid == k) {
            fwd = cuhll::nt_hash_init_fwd(seq + i - k + 1, k);
            rc  = cuhll::nt_hash_init_rc (seq + i - k + 1, k);
        } else {
            unsigned code_out = cuhll::nt_base_code(seq[i - k]);
            fwd = cuhll::nt_hash_roll_fwd(fwd, code_out, code, k);
            rc  = cuhll::nt_hash_roll_rc (rc,  code_out, code, k);
        }
        out.push_back(cuhll::nt_canonical(fwd, rc));
    }
}

// -----------------------------------------------------------------------------
// Independent CPU substring recompute: for every valid k-mer in the sequence,
// call nt_hash_init_{fwd,rc} fresh on its k bases. This is O(n*k) and only
// valid for small n, but it's the most rigorous ground truth we can compute
// cheaply without re-deriving the ntHash seed schedule.
// -----------------------------------------------------------------------------
static void cpu_substring_recompute(const char* seq, int len, int k,
                                    std::vector<std::uint64_t>& out) {
    out.clear();
    for (int p = 0; p + k <= len; ++p) {
        bool bad = false;
        for (int j = 0; j < k; ++j) {
            if (cuhll::nt_base_code(seq[p + j]) > 3u) { bad = true; break; }
        }
        if (bad) continue;
        std::uint64_t f = cuhll::nt_hash_init_fwd(seq + p, k);
        std::uint64_t r = cuhll::nt_hash_init_rc (seq + p, k);
        out.push_back(cuhll::nt_canonical(f, r));
    }
}

// -----------------------------------------------------------------------------
// GPU sliding-window kernel. Single-threaded on purpose: this test validates
// the ntHash device code itself, not the multi-threaded stripe kernel (which
// comes in milestone (c)).
// -----------------------------------------------------------------------------
__global__ void gpu_sliding_kernel(const char* __restrict__ seq, int len, int k,
                                   std::uint64_t* __restrict__ out, int* __restrict__ out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int idx = 0;
    int valid = 0;
    std::uint64_t fwd = 0, rc = 0;
    for (int i = 0; i < len; ++i) {
        unsigned code = cuhll::nt_base_code(seq[i]);
        if (code > 3u) { valid = 0; continue; }
        ++valid;
        if (valid < k) continue;
        if (valid == k) {
            fwd = cuhll::nt_hash_init_fwd(seq + i - k + 1, k);
            rc  = cuhll::nt_hash_init_rc (seq + i - k + 1, k);
        } else {
            unsigned code_out = cuhll::nt_base_code(seq[i - k]);
            fwd = cuhll::nt_hash_roll_fwd(fwd, code_out, code, k);
            rc  = cuhll::nt_hash_roll_rc (rc,  code_out, code, k);
        }
        out[idx++] = cuhll::nt_canonical(fwd, rc);
    }
    *out_count = idx;
}

static int run_for_k(int k, const char* d_seq, std::uint64_t* d_out, int* d_cnt,
                     double& out_gpu_ms, std::vector<std::uint64_t>& gpu_hashes_out) {
    // --- CPU sliding ---
    std::vector<std::uint64_t> cpu_hashes;
    cpu_sliding(kSeq, kSeqLen, k, cpu_hashes);

    // --- CPU substring recompute (independent algorithm) ---
    std::vector<std::uint64_t> cpu_sub;
    cpu_substring_recompute(kSeq, kSeqLen, k, cpu_sub);

    if (cpu_hashes.size() != cpu_sub.size()) {
        std::fprintf(stderr, "[k=%d] FAIL: sliding produced %zu hashes, substring produced %zu\n",
                     k, cpu_hashes.size(), cpu_sub.size());
        return 1;
    }
    int sub_mismatches = 0;
    for (size_t i = 0; i < cpu_hashes.size(); ++i) {
        if (cpu_hashes[i] != cpu_sub[i]) {
            if (sub_mismatches < 5) {
                std::fprintf(stderr,
                             "[k=%d] substring mismatch at i=%zu: sliding=0x%016llx substring=0x%016llx\n",
                             k, i,
                             (unsigned long long)cpu_hashes[i],
                             (unsigned long long)cpu_sub[i]);
            }
            ++sub_mismatches;
        }
    }
    if (sub_mismatches != 0) return 1;

    // --- GPU sliding ---
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    gpu_sliding_kernel<<<1, 1>>>(d_seq, kSeqLen, k, d_out, d_cnt);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    out_gpu_ms = static_cast<double>(ms);

    int gpu_count = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_count, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    gpu_hashes_out.assign(gpu_count, 0);
    CUDA_CHECK(cudaMemcpy(gpu_hashes_out.data(), d_out,
                          sizeof(std::uint64_t) * static_cast<size_t>(gpu_count),
                          cudaMemcpyDeviceToHost));

    if (static_cast<int>(cpu_hashes.size()) != gpu_count) {
        std::fprintf(stderr, "[k=%d] FAIL: cpu produced %zu, gpu produced %d\n",
                     k, cpu_hashes.size(), gpu_count);
        return 1;
    }
    int gpu_mismatches = 0;
    for (int i = 0; i < gpu_count; ++i) {
        if (cpu_hashes[i] != gpu_hashes_out[i]) {
            if (gpu_mismatches < 5) {
                std::fprintf(stderr,
                             "[k=%d] GPU mismatch at i=%d: cpu=0x%016llx gpu=0x%016llx\n",
                             k, i,
                             (unsigned long long)cpu_hashes[i],
                             (unsigned long long)gpu_hashes_out[i]);
            }
            ++gpu_mismatches;
        }
    }
    if (gpu_mismatches != 0) return 1;

    std::printf("[k=%d] produced=%d  sliding==substring: OK  gpu==cpu: OK  gpu_wall_ms=%.4f\n",
                k, gpu_count, static_cast<double>(ms));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}

int main() {
    std::printf("cuHLL nthash test (milestone b)\n");
    std::printf("sequence (%d bases, N at offset 50):\n  %s\n", kSeqLen, kSeq);

    char* d_seq = nullptr;
    std::uint64_t* d_out = nullptr;
    int* d_cnt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seq, static_cast<size_t>(kSeqLen)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 256));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_seq, kSeq, static_cast<size_t>(kSeqLen), cudaMemcpyHostToDevice));

    auto wall0 = std::chrono::steady_clock::now();

    int rc = 0;
    double gpu_ms_k21 = 0.0, gpu_ms_k31 = 0.0;
    std::vector<std::uint64_t> gpu_k31;

    {
        std::vector<std::uint64_t> gpu_k21;
        rc |= run_for_k(21, d_seq, d_out, d_cnt, gpu_ms_k21, gpu_k21);
    }
    rc |= run_for_k(31, d_seq, d_out, d_cnt, gpu_ms_k31, gpu_k31);

    auto wall1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();

    if (rc != 0) {
        std::fprintf(stderr, "nthash_test: FAIL\n");
        return rc;
    }

    // Print first 10 canonical hashes at k=31 to stdout.
    std::printf("first 10 canonical hashes at k=31 (hex):\n");
    for (int i = 0; i < 10 && i < static_cast<int>(gpu_k31.size()); ++i) {
        std::printf("  [%2d] 0x%016llx\n", i, (unsigned long long)gpu_k31[i]);
    }

    // Also write to bench/results/nthash_ground_truth.txt for regression checks.
    const char* gt_path = "bench/results/nthash_ground_truth.txt";
    FILE* f = std::fopen(gt_path, "w");
    if (!f) {
        std::fprintf(stderr, "[warn] could not open %s for writing: %s\n",
                     gt_path, std::strerror(errno));
    } else {
        std::fprintf(f, "# cuHLL ntHash ground truth  (Mohamadi 2016 original variant)\n");
        std::fprintf(f, "# generated by tests/test_nthash.cu\n");
        std::fprintf(f, "# sequence (100 bases, N at offset 50):\n");
        std::fprintf(f, "#   %s\n", kSeq);
        std::fprintf(f, "# k=31  canonical = min(fwd, rc)  (hex, 16 chars)\n");
        std::fprintf(f, "idx\tcanonical_hash\n");
        for (int i = 0; i < 10 && i < static_cast<int>(gpu_k31.size()); ++i) {
            std::fprintf(f, "%d\t0x%016llx\n", i, (unsigned long long)gpu_k31[i]);
        }
        std::fclose(f);
        std::printf("ground truth written to %s\n", gt_path);
    }

    std::printf("nthash_test: PASS   total_wall_ms=%.3f  kernel_ms(k=21)=%.4f  kernel_ms(k=31)=%.4f\n",
                wall_ms, gpu_ms_k21, gpu_ms_k31);

    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_cnt));
    return 0;
}

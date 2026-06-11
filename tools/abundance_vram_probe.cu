// abundance_vram_probe.cu — measured DEVICE (VRAM) footprint of the bounded counting
// table (cuco::static_map<uint64,uint32>), and its per-entry bytes.
//
// The table capacity depends only on the bottom-k sample size S (capacity =
// 2*S + 1024 in gpu_count), NOT on the number of genomes, so its VRAM is FLAT
// in genome count by construction. This probe measures the actual bytes via
// cudaMemGetInfo so the footprint number is measured, not theoretical.

#include <cuco/static_map.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using Map = cuco::static_map<std::uint64_t, std::uint32_t>;
static constexpr std::uint64_t kU64Max = ~std::uint64_t(0);

static void measure(std::uint64_t S) {
    const std::size_t cap = 2 * S + 1024;
    std::size_t free0 = 0, total = 0, free1 = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free0, &total);
    {
        Map map{cap, cuco::empty_key<std::uint64_t>{kU64Max},
                     cuco::empty_value<std::uint32_t>{0u}};
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free1, &total);
        const long long bytes = (long long)free0 - (long long)free1;
        std::printf("  S=%-7llu capacity=%-9zu  table VRAM = %lld bytes (%.3f MB)  "
                    "=> %.1f bytes/slot   [total VRAM %.1f GB]\n",
                    (unsigned long long)S, cap, bytes, bytes / 1.0e6,
                    (double)bytes / (double)cap, total / 1.0e9);
    }
}

int main() {
    // Warm up the CUDA context so its fixed overhead isn't charged to the table.
    void* warm = nullptr; cudaMalloc(&warm, 1 << 20); cudaFree(warm);
    std::printf("vram_probe: cuco::static_map<uint64,uint32> counting table\n");
    for (std::uint64_t S : {1000ull, 10000ull, 50000ull, 200000ull, 1000000ull})
        measure(S);
    std::printf("(capacity = 2*S+1024, independent of genome count -> VRAM flat in #genomes)\n");
    return 0;
}

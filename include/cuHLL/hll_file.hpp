#pragma once
// hll_file — persistent per-genome HyperLogLog sketches.
//
// File layout (little-endian, no padding; total header 48 bytes):
//
//   offset  size  field
//   -----   ----  -----
//   0       8     magic             "CUHLL" + {0,0,1}   (8 bytes)
//   8       4     version           uint32; 1 = original, 2 = adds canonical
//   12      4     precision_p       uint32 (HLL p; 8..18)
//   16      4     k                 uint32 k-mer length
//   20      4     hash_type         uint32 = 0 (cuco::xxhash_64)
//   24      8     n_registers       uint64 = 2^precision_p
//   32      8     register_bytes    uint64 = n_registers * 4   (cuco register = int32)
//   40      1     canonical         uint8 (v2 only): 1 = canonical, 0 = non-canonical.
//                                   Version-1 files stored 0 in this position as
//                                   part of the reserved zone and are interpreted
//                                   as canonical=1 on read (back-compat).
//   41      7     reserved          (zeroed)
//   48      register_bytes           raw cuco register data (bit-for-bit)
//
// On read, the loader validates magic + version + hash_type, then
// allocates a cuco::hyperloglog with the matching precision and
// cudaMemcpy's the register bytes straight back in.
//
// Milestone (l) bumps version to 2. V1 files (produced through milestone k)
// are still readable and treated as canonical, so no existing .hll needs
// re-sketching.

#include "cuHLL/sketch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cuhll {

struct HllFileHeader {
    std::uint8_t  magic[8];         // "CUHLL" + 3 bytes version-marker
    std::uint32_t version;          // 1 (milestones a-k) or 2 (milestone l+)
    std::uint32_t precision_p;      // HLL precision parameter
    std::uint32_t k;                // k-mer length this sketch was built for
    std::uint32_t hash_type;        // 0 = cuco::xxhash_64
    std::uint64_t n_registers;      // 2^precision_p
    std::uint64_t register_bytes;   // n_registers * sizeof(cuco register)
    std::uint8_t  canonical;        // v2: 1 = canonical, 0 = non-canonical
                                    // v1 files: stored 0 here as part of
                                    // reserved; read_hll_header promotes to 1
                                    // (canonical) for backward compat.
    std::uint8_t  reserved[7];      // zero
};
static_assert(sizeof(HllFileHeader) == 48, "HllFileHeader must be 48 bytes");

constexpr std::uint32_t kHllFileVersion  = 2u;     // bumped in milestone l
constexpr std::uint32_t kHllFileLegacyV1 = 1u;
constexpr std::uint32_t kHllHashXxhash64 = 0u;

// Read just the header; throws on magic / version / hash mismatch.
HllFileHeader read_hll_header(const std::string& path);

// Persist `s`'s HLL register state to `path`. Records `k` in the header.
// Overwrites `path` if it exists.
void write_hll(const std::string& path, const Sketch& s, int k);

// Reconstruct a Sketch from a .hll file. The returned Sketch has its
// precision set from the header, and its register state bit-copied from
// the file.
Sketch read_hll(const std::string& path);

// Low-level writer used by the concurrent pipeline: takes raw register
// bytes directly (already D2H-copied into host memory) and emits an .hll
// file. Caller supplies n_registers via precision_p and the byte count
// is derived. Output is byte-for-byte identical to write_hll() on a Sketch
// whose registers match `registers`.
void write_hll_registers(const std::string& path,
                         const std::uint32_t* registers,
                         std::uint32_t precision_p,
                         std::uint32_t k,
                         bool canonical = true);

} // namespace cuhll

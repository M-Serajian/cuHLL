#pragma once
// On-disk format for persisted HyperLogLog sketches.
//
// File layout (little-endian, no padding; 48-byte header followed by
// register_bytes of raw register data):
//
//   offset  size  field
//   ------  ----  -----
//   0       8     magic            "CUHLL" + {0, 0, 1}    (8 bytes)
//   8       4     version          uint32; 1 = legacy, 2 = adds canonical flag
//   12      4     precision_p      uint32 HLL p, in [8, 18]
//   16      4     k                uint32 k-mer length
//   20      4     hash_type        uint32 = 0 (cuco::xxhash_64)
//   24      8     n_registers      uint64 = 2^precision_p
//   32      8     register_bytes   uint64 = n_registers * 4 (cuco register width)
//   40      1     canonical        uint8 (v2): 1 = canonical, 0 = non-canonical.
//                                  v1 files always read back as canonical (1).
//   41      7     reserved         zero
//   48      register_bytes          raw cuco register data (bit-for-bit)
//
// Reading validates magic, version, and hash_type, then bit-copies the
// register block straight into a freshly constructed cuco::hyperloglog
// of matching precision.

#include "cuHLL/sketch/sketch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cuhll {

struct HllFileHeader {
    std::uint8_t  magic[8];         // "CUHLL" + 3-byte version marker
    std::uint32_t version;          // 1 (legacy) or 2 (current)
    std::uint32_t precision_p;      // HLL precision parameter
    std::uint32_t k;                // k-mer length this sketch was built for
    std::uint32_t hash_type;        // 0 = cuco::xxhash_64
    std::uint64_t n_registers;      // 2^precision_p
    std::uint64_t register_bytes;   // n_registers * sizeof(cuco register)
    std::uint8_t  canonical;        // v2: 1 = canonical, 0 = non-canonical.
                                    // v1 files stored 0 here; the reader
                                    // promotes them to canonical = 1.
    std::uint8_t  reserved[7];      // zero
};
static_assert(sizeof(HllFileHeader) == 48, "HllFileHeader must be 48 bytes");

constexpr std::uint32_t kHllFileVersion  = 2u;   // current on-disk version
constexpr std::uint32_t kHllFileLegacyV1 = 1u;
constexpr std::uint32_t kHllHashXxhash64 = 0u;

// Read just the file header. Throws std::runtime_error on magic mismatch,
// unsupported version, or unsupported hash_type.
HllFileHeader read_hll_header(const std::string& path);

// Write the sketch's register state to `path`. Records `k` in the header.
// Overwrites any existing file at `path`.
void write_hll(const std::string& path, const Sketch& s, int k);

// Read a sketch back from a .hll file. The returned Sketch has its
// precision set from the header and its register state restored bit-for-bit.
Sketch read_hll(const std::string& path);

// Low-level writer for the concurrent pipeline: takes already-host-resident
// register bytes and emits an .hll file. Output is byte-for-byte identical
// to write_hll() called on a Sketch whose registers match `registers`.
void write_hll_registers(const std::string& path,
                         const std::uint32_t* registers,
                         std::uint32_t precision_p,
                         std::uint32_t k,
                         bool canonical = true);

} // namespace cuhll

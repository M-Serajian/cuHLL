#pragma once
// cb2 — cuHLL Binary 2-bit format.
//
// Offline-packed representation of a FASTA record's filtered sequence bytes.
// Packing reduces data-path size by 8/3 (one 8-bit ACGT/N byte becomes 2 bits
// of payload + 1 bit of mask) AND removes FASTA parsing — the hot path for
// this input type is mmap + H2D + kernel.
//
// File layout (little-endian, no padding):
//
//   offset  size  field
//   -----   ----  -----
//   0       8     magic          = "CB2\0\0\0\0\1"
//   8       4     version        uint32 = 1
//   12      4     flags          uint32 = 0   (reserved)
//   16      8     n_bases        uint64
//   24      8     n_seq_bytes    uint64       = ceil(n_bases / 4)
//   32      8     n_mask_bytes   uint64       = ceil(n_bases / 8)
//   40      n_seq_bytes   packed_seq   (2 bits per base, A=0, C=1, G=2, T=3;
//                                        non-ACGT bases pack as A=0 here and
//                                        have their mask bit set)
//   40+nsb  n_mask_bytes  n_mask       (1 bit per base, bit set = non-ACGT
//                                        window-breaker)
//
// Within a byte, base i's 2 bits live in positions (2*(i%4)) and (2*(i%4)+1):
//   byte[0] = base0[1:0] | base1[3:2] | base2[5:4] | base3[7:6]
//
// Within the mask, base i's bit is (mask[i/8] >> (i%8)) & 1.
//
// Total on-disk size ≈ 40 + 3/8 * n_bases (62.5% compression vs raw FASTA
// filtered bytes).

#include <cstdint>
#include <cstddef>
#include <string>

namespace cuhll {

constexpr std::uint32_t kCb2Version = 1u;

struct Cb2Header {
    std::uint8_t  magic[8];      // "CB2\0\0\0\0\1"
    std::uint32_t version;       // = 1
    std::uint32_t flags;         // reserved, = 0
    std::uint64_t n_bases;       // number of bases represented
    std::uint64_t n_seq_bytes;   // ceil(n_bases/4)
    std::uint64_t n_mask_bytes;  // ceil(n_bases/8)
};
static_assert(sizeof(Cb2Header) == 40, "Cb2Header must be 40 bytes");

// Pack a host-side filtered sequence (ACGT/N bytes) into a .cb2 file on disk.
// Returns the number of bytes written. Throws std::runtime_error on I/O
// failure.
std::size_t write_cb2(const std::string& out_path,
                      const char* seq,
                      std::size_t n_bases);

// Read just the header from a .cb2 file; throws on magic or version mismatch.
Cb2Header read_cb2_header(const std::string& path);

// Memory-mapped view of a .cb2 file. The caller owns the lifetime; close()
// unmaps and closes the fd. The payload pointers are valid until close().
class Cb2Mmap {
public:
    explicit Cb2Mmap(const std::string& path);
    ~Cb2Mmap();
    Cb2Mmap(const Cb2Mmap&) = delete;
    Cb2Mmap& operator=(const Cb2Mmap&) = delete;
    Cb2Mmap(Cb2Mmap&& other) noexcept;
    Cb2Mmap& operator=(Cb2Mmap&&) noexcept;

    void close() noexcept;

    const Cb2Header& header()    const noexcept { return header_; }
    const std::uint8_t* packed() const noexcept { return packed_; }
    const std::uint8_t* mask()   const noexcept { return mask_; }
    std::uint64_t n_bases()      const noexcept { return header_.n_bases; }

private:
    int           fd_        = -1;
    void*         map_       = nullptr;
    std::size_t   map_size_  = 0;
    Cb2Header     header_{};
    const std::uint8_t* packed_ = nullptr;
    const std::uint8_t* mask_   = nullptr;
};

// Host-side decoders (used by tests and the packer round-trip).
inline unsigned cb2_get_base(const std::uint8_t* packed, std::uint64_t i) {
    return (packed[i >> 2] >> ((static_cast<unsigned>(i) & 3u) << 1)) & 0x3u;
}

inline unsigned cb2_get_mask_bit(const std::uint8_t* mask, std::uint64_t i) {
    return (mask[i >> 3] >> (static_cast<unsigned>(i) & 7u)) & 0x1u;
}

} // namespace cuhll

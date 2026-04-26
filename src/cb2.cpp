#include "cuHLL/cb2.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace cuhll {

namespace {

constexpr std::uint8_t kMagicBytes[8] = {'C','B','2',0,0,0,0,1};

inline unsigned ascii_to_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:            return 4u; // N / ambiguous / other
    }
}

void throw_errno(const std::string& what, const std::string& path) {
    throw std::runtime_error("cuHLL cb2: " + what + " (" + path + "): " + std::strerror(errno));
}

} // namespace

std::size_t write_cb2(const std::string& out_path,
                      const char* seq,
                      std::size_t n_bases) {
    const std::uint64_t nb = static_cast<std::uint64_t>(n_bases);
    const std::uint64_t n_seq_bytes  = (nb + 3u) / 4u;
    const std::uint64_t n_mask_bytes = (nb + 7u) / 8u;

    Cb2Header hdr{};
    std::memcpy(hdr.magic, kMagicBytes, 8);
    hdr.version      = kCb2Version;
    hdr.flags        = 0u;
    hdr.n_bases      = nb;
    hdr.n_seq_bytes  = n_seq_bytes;
    hdr.n_mask_bytes = n_mask_bytes;

    std::vector<std::uint8_t> packed(n_seq_bytes, 0u);
    std::vector<std::uint8_t> mask  (n_mask_bytes, 0u);

    for (std::uint64_t i = 0; i < nb; ++i) {
        unsigned code = ascii_to_code(seq[i]);
        if (code > 3u) {
            // Non-ACGT: pack as A (0) and set mask bit to break kernel window.
            mask[i >> 3] |= static_cast<std::uint8_t>(1u << (static_cast<unsigned>(i) & 7u));
            code = 0u;
        }
        packed[i >> 2] |= static_cast<std::uint8_t>(
            code << ((static_cast<unsigned>(i) & 3u) << 1));
    }

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) throw_errno("open for write", out_path);

    auto must_write = [&](const void* p, std::size_t n) {
        if (n == 0) return;
        if (std::fwrite(p, 1, n, f) != n) {
            std::fclose(f);
            throw_errno("write", out_path);
        }
    };

    must_write(&hdr, sizeof(hdr));
    must_write(packed.data(), packed.size());
    must_write(mask.data(),   mask.size());

    if (std::fclose(f) != 0) throw_errno("close", out_path);

    return sizeof(hdr) + packed.size() + mask.size();
}

Cb2Header read_cb2_header(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw_errno("open for read", path);

    Cb2Header hdr{};
    if (std::fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        std::fclose(f);
        throw std::runtime_error("cuHLL cb2: short header in " + path);
    }
    std::fclose(f);

    if (std::memcmp(hdr.magic, kMagicBytes, 8) != 0) {
        throw std::runtime_error("cuHLL cb2: bad magic in " + path);
    }
    if (hdr.version != kCb2Version) {
        throw std::runtime_error("cuHLL cb2: unsupported version in " + path);
    }
    const std::uint64_t expect_seq  = (hdr.n_bases + 3u) / 4u;
    const std::uint64_t expect_mask = (hdr.n_bases + 7u) / 8u;
    if (hdr.n_seq_bytes != expect_seq || hdr.n_mask_bytes != expect_mask) {
        throw std::runtime_error("cuHLL cb2: inconsistent size fields in " + path);
    }
    return hdr;
}

// -----------------------------------------------------------------------------
// Cb2Mmap
// -----------------------------------------------------------------------------
Cb2Mmap::Cb2Mmap(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw_errno("open", path);

    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        ::close(fd_); fd_ = -1;
        throw_errno("fstat", path);
    }
    map_size_ = static_cast<std::size_t>(st.st_size);
    if (map_size_ < sizeof(Cb2Header)) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("cuHLL cb2: file too small (" + path + ")");
    }
    map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        ::close(fd_); fd_ = -1;
        map_ = nullptr;
        throw_errno("mmap", path);
    }

    std::memcpy(&header_, map_, sizeof(header_));
    if (std::memcmp(header_.magic, kMagicBytes, 8) != 0) {
        close();
        throw std::runtime_error("cuHLL cb2: bad magic in " + path);
    }
    if (header_.version != kCb2Version) {
        close();
        throw std::runtime_error("cuHLL cb2: unsupported version in " + path);
    }
    const std::uint64_t expect = sizeof(Cb2Header)
                               + header_.n_seq_bytes
                               + header_.n_mask_bytes;
    if (static_cast<std::uint64_t>(map_size_) < expect) {
        close();
        throw std::runtime_error("cuHLL cb2: file shorter than header claims (" + path + ")");
    }

    const auto* base = static_cast<const std::uint8_t*>(map_);
    packed_ = base + sizeof(Cb2Header);
    mask_   = packed_ + header_.n_seq_bytes;
}

Cb2Mmap::~Cb2Mmap() {
    close();
}

Cb2Mmap::Cb2Mmap(Cb2Mmap&& other) noexcept
    : fd_(other.fd_),
      map_(other.map_),
      map_size_(other.map_size_),
      header_(other.header_),
      packed_(other.packed_),
      mask_(other.mask_) {
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_size_ = 0;
    other.packed_ = nullptr;
    other.mask_ = nullptr;
}

Cb2Mmap& Cb2Mmap::operator=(Cb2Mmap&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        map_ = other.map_;
        map_size_ = other.map_size_;
        header_ = other.header_;
        packed_ = other.packed_;
        mask_   = other.mask_;
        other.fd_ = -1;
        other.map_ = nullptr;
        other.map_size_ = 0;
        other.packed_ = nullptr;
        other.mask_   = nullptr;
    }
    return *this;
}

void Cb2Mmap::close() noexcept {
    if (map_ && map_ != MAP_FAILED) {
        ::munmap(map_, map_size_);
    }
    map_ = nullptr;
    map_size_ = 0;
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    packed_ = nullptr;
    mask_   = nullptr;
}

} // namespace cuhll

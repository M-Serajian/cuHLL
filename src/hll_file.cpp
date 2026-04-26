#include "cuHLL/hll_file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuhll {

namespace {

constexpr std::uint8_t kMagic[8] = {'C','U','H','L','L',0,0,1};

[[noreturn]] void err(const std::string& what, const std::string& path) {
    throw std::runtime_error("cuHLL hll_file: " + what + " (" + path + "): "
                             + std::strerror(errno));
}

} // namespace

HllFileHeader read_hll_header(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) err("open for read", path);

    HllFileHeader hdr{};
    if (std::fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        std::fclose(f);
        throw std::runtime_error("cuHLL hll_file: short header in " + path);
    }
    std::fclose(f);

    if (std::memcmp(hdr.magic, kMagic, 8) != 0) {
        throw std::runtime_error("cuHLL hll_file: bad magic in " + path);
    }
    if (hdr.version != kHllFileVersion && hdr.version != kHllFileLegacyV1) {
        throw std::runtime_error("cuHLL hll_file: unsupported version in " + path);
    }
    if (hdr.hash_type != kHllHashXxhash64) {
        throw std::runtime_error("cuHLL hll_file: unknown hash_type in " + path);
    }
    const std::uint64_t expect_regs  = 1ULL << hdr.precision_p;
    const std::uint64_t expect_bytes = expect_regs * 4ULL; // cuco register = int32
    if (hdr.n_registers != expect_regs || hdr.register_bytes != expect_bytes) {
        throw std::runtime_error("cuHLL hll_file: inconsistent size fields in " + path);
    }
    // Milestone (l) backward compat: v1 files stored 0 in what is now the
    // `canonical` byte (it was part of `reserved`). All sketches produced
    // before milestone (l) are canonical, so promote 0 -> 1 on v1 reads.
    if (hdr.version == kHllFileLegacyV1) {
        hdr.canonical = 1;
    } else {
        // v2: only 0 or 1 is valid.
        if (hdr.canonical != 0 && hdr.canonical != 1) {
            throw std::runtime_error("cuHLL hll_file: bad canonical byte in " + path);
        }
    }
    return hdr;
}

void write_hll(const std::string& path, const Sketch& s, int k) {
    const std::uint32_t p  = static_cast<std::uint32_t>(s.precision());
    const std::size_t n_regs = static_cast<std::size_t>(1ULL << p);
    std::vector<std::uint32_t> regs(n_regs);
    s.copy_registers_to_host(regs.data());
    write_hll_registers(path, regs.data(), p,
                        static_cast<std::uint32_t>(k), s.canonical());
}

void write_hll_registers(const std::string& path,
                         const std::uint32_t* registers,
                         std::uint32_t precision_p,
                         std::uint32_t k,
                         bool canonical) {
    HllFileHeader hdr{};
    std::memcpy(hdr.magic, kMagic, 8);
    hdr.version        = kHllFileVersion;
    hdr.precision_p    = precision_p;
    hdr.k              = k;
    hdr.hash_type      = kHllHashXxhash64;
    hdr.n_registers    = 1ULL << precision_p;
    hdr.register_bytes = hdr.n_registers * 4ULL;
    hdr.canonical      = canonical ? 1u : 0u;
    // reserved left zero.

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) err("open for write", path);

    auto must_write = [&](const void* p, std::size_t n) {
        if (std::fwrite(p, 1, n, f) != n) {
            std::fclose(f);
            err("write", path);
        }
    };
    must_write(&hdr, sizeof(hdr));
    must_write(registers, hdr.register_bytes);

    if (std::fclose(f) != 0) err("close", path);
}

Sketch read_hll(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) err("open for read", path);

    HllFileHeader hdr{};
    if (std::fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        std::fclose(f);
        throw std::runtime_error("cuHLL hll_file: short header in " + path);
    }
    if (std::memcmp(hdr.magic, kMagic, 8) != 0
        || (hdr.version != kHllFileVersion && hdr.version != kHllFileLegacyV1)
        || hdr.hash_type != kHllHashXxhash64) {
        std::fclose(f);
        throw std::runtime_error("cuHLL hll_file: invalid header in " + path);
    }

    // v1 stored 0 in what is now the canonical byte; all v1 sketches were
    // canonical, so promote here so the reconstructed Sketch remembers.
    bool canonical = (hdr.version == kHllFileLegacyV1) ? true : (hdr.canonical != 0);

    std::vector<std::uint32_t> regs(hdr.n_registers);
    if (std::fread(regs.data(), 1, hdr.register_bytes, f) != hdr.register_bytes) {
        std::fclose(f);
        throw std::runtime_error("cuHLL hll_file: short register body in " + path);
    }
    std::fclose(f);

    Sketch out(static_cast<int>(hdr.precision_p), canonical);
    out.load_registers_from_host(regs.data());
    return out;
}

} // namespace cuhll

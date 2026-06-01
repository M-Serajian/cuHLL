#include "cuHLL/io/fasta.hpp"
#include "cuHLL/common/nvtx_util.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <zlib.h>

namespace cuhll {

namespace {

constexpr int kGzipInflation = 4;

bool is_gzip(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char magic[2] = {0, 0};
    std::size_t got = std::fread(magic, 1, 2, f);
    std::fclose(f);
    return (got == 2 && magic[0] == 0x1f && magic[1] == 0x8b);
}

std::vector<char> slurp_plain(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cuHLL: stat failed: " + path);
    }
    const std::size_t sz = static_cast<std::size_t>(st.st_size);
    std::vector<char> out(sz);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cuHLL: cannot open: " + path);
    }
    std::size_t got = std::fread(out.data(), 1, sz, f);
    std::fclose(f);
    out.resize(got);
    return out;
}

std::vector<char> slurp_gzip(const std::string& path) {
    gzFile zf = gzopen(path.c_str(), "rb");
    if (!zf) {
        throw std::runtime_error("cuHLL: cannot open: " + path);
    }
    struct stat st{};
    std::size_t hint = 256 * 1024;
    if (::stat(path.c_str(), &st) == 0) {
        hint = static_cast<std::size_t>(st.st_size) * kGzipInflation;
    }
    constexpr int CHUNK = 256 * 1024;
    std::vector<char> out;
    out.reserve(hint);
    char buf[CHUNK];
    int n;
    while ((n = gzread(zf, buf, static_cast<unsigned>(CHUNK))) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    if (n < 0) {
        int errnum = 0;
        const char* gzerr = gzerror(zf, &errnum);
        gzclose(zf);
        throw std::runtime_error("cuHLL: read error on " + path + ": "
                                 + (gzerr && gzerr[0] ? gzerr : "zlib error"));
    }
    gzclose(zf);
    return out;
}

std::vector<char> slurp_all_bytes(const std::string& path) {
    return is_gzip(path) ? slurp_gzip(path) : slurp_plain(path);
}

// Parse FASTA bytes: one memchr per line (find '\n'), one byte check for
// '>' at line start, strip '\r' before '\n'. Non-ACGT characters that slip
// through (spaces, tabs) are harmless — the kernel's nt_base_code returns
// code=4, resetting the kmer window, same as an 'N'.
std::string parse_fasta_bytes(const char* base, std::size_t len) {
    std::string out;
    out.reserve(len);
    const char* const end = base + len;
    const char* p = base;
    bool first_record_seen = false;

    while (p < end) {
        if (*p == '\n' || *p == '\r') { ++p; continue; }
        if (*p == '>') {
            if (first_record_seen) out.push_back('N');
            first_record_seen = true;
            const char* nl = static_cast<const char*>(
                std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
            p = nl ? nl + 1 : end;
            continue;
        }
        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* le = nl ? nl : end;
        std::size_t n = static_cast<std::size_t>(le - p);
        if (n > 0 && p[n - 1] == '\r') --n;
        out.append(p, n);
        p = nl ? nl + 1 : end;
    }
    return out;
}

// 4-line FASTQ. Quality bytes can contain '@' (Phred+33 allows ASCII 33..126
// including '@'=64 and '+'=43), so we can NOT detect record boundaries by
// glyph — we count lines mod 4. Modern Illumina/ONT pipelines all emit
// single-line FASTQ records; multi-line sequence/quality isn't supported here.
std::string parse_fastq_bytes(const char* base, std::size_t len) {
    std::string out;
    out.reserve(len / 2);   // sequence is roughly half of FASTQ bytes
    const char* const end = base + len;
    const char* p = base;
    int line_in_record = 0; // 0=header @, 1=seq, 2=plus, 3=quality
    bool first_seq = true;

    while (p < end) {
        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* line_end = nl ? nl : end;

        if (line_in_record == 1) {
            if (!first_seq) out.push_back('N');
            first_seq = false;
            // Strip trailing CR + any in-line whitespace; copy the rest.
            for (const char* q = p; q < line_end; ++q) {
                const char c = *q;
                if (c == '\r' || c == ' ' || c == '\t') continue;
                out.push_back(c);
            }
        }
        line_in_record = (line_in_record + 1) % 4;
        if (!nl) break;
        p = nl + 1;
    }
    return out;
}

// Skip leading whitespace, dispatch by first content byte.
std::string parse_sequences(const char* base, std::size_t len) {
    const char* p = base;
    const char* const end = base + len;
    while (p < end && (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t')) {
        ++p;
    }
    if (p == end) return {};
    const std::size_t rem = static_cast<std::size_t>(end - p);
    if (*p == '>') return parse_fasta_bytes(p, rem);
    if (*p == '@') return parse_fastq_bytes(p, rem);
    throw std::runtime_error(
        "cuHLL: unknown sequence format — first non-whitespace byte must "
        "be '>' (FASTA) or '@' (FASTQ)");
}

// Variants that write into a caller-provided char* buffer instead of
// returning std::string. Used by the concurrent pipeline to parse directly
// into pinned memory (eliminates one full memcpy per genome).

std::size_t parse_fasta_into(const char* base, std::size_t len,
                             char* dst, std::size_t cap) {
    const char* const end = base + len;
    const char* p = base;
    bool first_record_seen = false;
    std::size_t pos = 0;

    while (p < end && pos < cap) {
        if (*p == '\n' || *p == '\r') { ++p; continue; }
        if (*p == '>') {
            if (first_record_seen && pos < cap) dst[pos++] = 'N';
            first_record_seen = true;
            const char* nl = static_cast<const char*>(
                std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
            p = nl ? nl + 1 : end;
            continue;
        }
        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* le = nl ? nl : end;
        std::size_t n = static_cast<std::size_t>(le - p);
        if (n > 0 && p[n - 1] == '\r') --n;
        if (pos + n > cap) n = cap - pos;
        std::memcpy(dst + pos, p, n);
        pos += n;
        p = nl ? nl + 1 : end;
    }
    return pos;
}

std::size_t parse_fastq_into(const char* base, std::size_t len,
                             char* dst, std::size_t cap) {
    const char* const end = base + len;
    const char* p = base;
    int line_in_record = 0;
    bool first_seq = true;
    std::size_t pos = 0;

    while (p < end) {
        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* line_end = nl ? nl : end;

        if (line_in_record == 1) {
            if (!first_seq && pos < cap) dst[pos++] = 'N';
            first_seq = false;
            for (const char* q = p; q < line_end && pos < cap; ++q) {
                const char c = *q;
                if (c == '\r' || c == ' ' || c == '\t') continue;
                dst[pos++] = c;
            }
        }
        line_in_record = (line_in_record + 1) % 4;
        if (!nl) break;
        p = nl + 1;
    }
    return pos;
}

std::size_t parse_sequences_into(const char* base, std::size_t len,
                                 char* dst, std::size_t cap) {
    const char* p = base;
    const char* const end = base + len;
    while (p < end && (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t'))
        ++p;
    if (p == end) return 0;
    const std::size_t rem = static_cast<std::size_t>(end - p);
    if (*p == '>') return parse_fasta_into(p, rem, dst, cap);
    if (*p == '@') return parse_fastq_into(p, rem, dst, cap);
    return 0;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// read_fasta_concat — name kept for backward compatibility. Transparently
// handles FASTA, FASTQ, and gzip-compressed variants of both. Returns the
// concatenated bases with a single 'N' injected between records so k-mers
// can't span unrelated sequences.
// -----------------------------------------------------------------------------
std::string read_fasta_concat(const std::string& path) {
    CUHLL_NVTX_RANGE("read_fasta_concat");
    auto bytes = slurp_all_bytes(path);
    if (bytes.empty()) return {};
    return parse_sequences(bytes.data(), bytes.size());
}

std::size_t read_fasta_into(const std::string& path, char* dst,
                            std::size_t capacity) {
    CUHLL_NVTX_RANGE("read_fasta_into");
    auto bytes = slurp_all_bytes(path);
    if (bytes.empty()) return 0;
    return parse_sequences_into(bytes.data(), bytes.size(), dst, capacity);
}

// -----------------------------------------------------------------------------
// FastaChunkReader
// -----------------------------------------------------------------------------
namespace {
constexpr std::size_t kRawBufBytes = 4ULL * 1024ULL * 1024ULL; // 4 MiB read-ahead
} // namespace

FastaChunkReader::FastaChunkReader(const std::string& path)
    : fin_(path, std::ios::binary),
      raw_buf_(kRawBufBytes) {
    if (!fin_) {
        throw std::runtime_error("cuHLL: cannot open FASTA: " + path);
    }
}

void FastaChunkReader::refill_raw() {
    if (raw_pos_ < raw_len_) return;
    if (raw_eof_) {
        raw_pos_ = raw_len_ = 0;
        return;
    }
    fin_.read(raw_buf_.data(), static_cast<std::streamsize>(raw_buf_.size()));
    raw_len_ = static_cast<std::size_t>(fin_.gcount());
    raw_pos_ = 0;
    if (raw_len_ == 0) raw_eof_ = true;
}

std::size_t FastaChunkReader::next_chunk(char* dst, std::size_t cap, std::size_t overlap) {
    // Phase 1: emit the tail from the previous call as this call's prefix.
    std::size_t out = 0;
    const std::size_t tail_sz = std::min(tail_.size(), overlap);
    if (tail_sz > 0) {
        std::memcpy(dst, tail_.data() + tail_.size() - tail_sz, tail_sz);
        out = tail_sz;
    }

    // Phase 2: parse fresh content into dst[out..cap).
    const std::size_t fresh_start = out;
    while (out < cap) {
        if (raw_pos_ >= raw_len_) refill_raw();
        if (raw_pos_ >= raw_len_) break; // EOF

        const char c = raw_buf_[raw_pos_++];

        if (c == '>') {
            // Start of a new FASTA record. Every boundary except the very
            // first one injects an 'N' as a window-breaker so k-mers can't
            // span across unrelated records.
            if (first_record_seen_) {
                dst[out++] = 'N';
            }
            first_record_seen_ = true;
            in_header_         = true;
            continue;
        }
        if (c == '\n' || c == '\r') { in_header_ = false; continue; }
        if (in_header_)             { continue; }
        if (c == ' ' || c == '\t')  { continue; }
        dst[out++] = c;
    }
    const std::size_t fresh_bytes = out - fresh_start;

    if (fresh_bytes == 0) {
        // EOF with no new bytes; the tail bytes we just copied are already
        // accounted for in the previous chunk and would produce zero new
        // k-mers in a standalone chunk. Return 0 so the orchestrator stops.
        eof_ = true;
        return 0;
    }

    // Phase 3: save the trailing `overlap` bytes of this call's output for
    // the next call's prefix.
    const std::size_t save = std::min(overlap, out);
    tail_.assign(dst + out - save, dst + out);

    return out;
}

} // namespace cuhll

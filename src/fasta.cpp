#include "cuHLL/fasta.hpp"
#include "cuHLL/nvtx_util.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace cuhll {

namespace {

// zlib's gzopen transparently handles both gzipped and plain files (it
// sniffs the gzip magic and falls through to a passthrough reader when
// absent). One code path, two formats.
std::vector<char> slurp_all_bytes(const std::string& path) {
    gzFile zf = gzopen(path.c_str(), "rb");
    if (!zf) {
        throw std::runtime_error("cuHLL: cannot open: " + path);
    }
    constexpr int CHUNK = 64 * 1024;
    std::vector<char> out;
    out.reserve(CHUNK);
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

// Parse FASTA bytes into the bases-with-'N'-between-records representation.
// memchr-based bulk scan — the inner loop only runs char-by-char in the
// rare case a sequence line contains whitespace or a mid-line '>'. On a
// 56 MB chr19 record this is ~near-memcpy-bandwidth on an L4 host.
std::string parse_fasta_bytes(const char* base, std::size_t len) {
    std::string out;
    out.reserve(len);
    const char* const end = base + len;
    const char* p = base;
    bool first_record_seen = false;

    while (p < end) {
        if (*p == '>') {
            if (first_record_seen) out.push_back('N');
            first_record_seen = true;
            const char* nl = static_cast<const char*>(
                std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
            if (!nl) break;
            p = nl + 1;
            continue;
        }
        if (*p == '\n' || *p == '\r') { ++p; continue; }

        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* region_end = nl ? nl : end;

        if (std::memchr(p, ' ',  static_cast<std::size_t>(region_end - p)) == nullptr &&
            std::memchr(p, '\t', static_cast<std::size_t>(region_end - p)) == nullptr &&
            std::memchr(p, '\r', static_cast<std::size_t>(region_end - p)) == nullptr &&
            std::memchr(p, '>',  static_cast<std::size_t>(region_end - p)) == nullptr) {
            out.append(p, static_cast<std::size_t>(region_end - p));
        } else {
            for (const char* q = p; q < region_end; ++q) {
                const char c = *q;
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
                if (c == '>') { region_end = q; break; }
                out.push_back(c);
            }
        }
        p = region_end;
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

}  // namespace

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

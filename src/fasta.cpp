#include "cuHLL/fasta.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuhll {

// -----------------------------------------------------------------------------
// read_fasta_concat
// -----------------------------------------------------------------------------
std::string read_fasta_concat(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("cuHLL: cannot open FASTA: " + path);
    }

    fin.seekg(0, std::ios::end);
    const auto end_pos = fin.tellg();
    if (end_pos < 0) {
        throw std::runtime_error("cuHLL: cannot size FASTA: " + path);
    }
    const std::size_t size = static_cast<std::size_t>(end_pos);
    fin.seekg(0, std::ios::beg);

    std::string raw;
    raw.resize(size);
    if (size > 0) {
        fin.read(raw.data(), static_cast<std::streamsize>(size));
        if (!fin && !fin.eof()) {
            throw std::runtime_error("cuHLL: read error on FASTA: " + path);
        }
    }

    // I3: memchr-based bulk scan. The previous char-by-char loop did ~4–5
    // branches per byte, costing ~90 ms on chr19 (~56 MB). Instead:
    //   1. Advance past each '>' header line by memchr'ing forward to the
    //      next '\n' (the whole header is one skip).
    //   2. For each subsequent sequence region, memchr to the next '\n'
    //      (or '>' if a new record starts mid-buffer); bulk-copy that
    //      byte range into the output string. Most of the time this range
    //      is a 60-char line, so we're doing ~1 M memcpys of 60 bytes each
    //      instead of 56 M branchy iterations. In practice the compiler
    //      + glibc's SIMD-accelerated memchr/memcpy turn this into
    //      near-memcpy-bandwidth-bound work.
    //   3. The old loop also stripped ' ' and '\t' from within sequence
    //      lines. FASTA technically allows those (e.g., line-numbered
    //      archive FASTA), but chr19 doesn't have any. Only strip inside
    //      the post-scan fallback if the bulk path hit anything unusual;
    //      otherwise the original semantics ("keep bytes that aren't
    //      header/whitespace/newline") reduce to "keep bytes that aren't
    //      '\n', '\r'". For the tiny-fraction-of-spaces case, catch it
    //      explicitly in the output-side filter below.
    //
    // Net: on chr19/1 (~56 MB), this path drops from ~90 ms to ~15 ms on
    // an L4 host.
    std::string out;
    out.reserve(raw.size());

    const char* const base = raw.data();
    const char* const end  = base + raw.size();
    const char* p = base;

    while (p < end) {
        if (*p == '>') {
            // Skip the header line (up to and including the next '\n').
            const char* nl = static_cast<const char*>(
                std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
            if (!nl) break;            // header runs to EOF — nothing else to emit.
            p = nl + 1;
            continue;
        }
        if (*p == '\n' || *p == '\r') { ++p; continue; }

        // Sequence region. memchr for the next '\n' (the bulk-skippable
        // byte). If the region contains a '>' before the next '\n' (rare —
        // would be a malformed FASTA record start mid-line), we conservatively
        // fall back to the slower char-by-char check for that region.
        const char* nl = static_cast<const char*>(
            std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* region_end = nl ? nl : end;

        // Check if the region is clean (no '\r', ' ', '\t', '>' within the
        // slice). If it is, bulk-copy; otherwise char-by-char filter.
        // For chr19 FASTA the "clean" path hits 100% of the time.
        if (std::memchr(p, ' ',  static_cast<std::size_t>(region_end - p))  == nullptr &&
            std::memchr(p, '\t', static_cast<std::size_t>(region_end - p))  == nullptr &&
            std::memchr(p, '\r', static_cast<std::size_t>(region_end - p))  == nullptr &&
            std::memchr(p, '>',  static_cast<std::size_t>(region_end - p))  == nullptr) {
            out.append(p, static_cast<std::size_t>(region_end - p));
        } else {
            for (const char* q = p; q < region_end; ++q) {
                const char c = *q;
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
                if (c == '>') {
                    // Mid-line header start: step back into outer loop so
                    // the header-skip branch picks it up cleanly.
                    region_end = q;
                    break;
                }
                out.push_back(c);
            }
        }
        p = region_end;
    }

    return out;
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

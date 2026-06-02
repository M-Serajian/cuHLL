#include "cuHLL/io/fasta.hpp"
#include "cuHLL/common/nvtx_util.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <zlib.h>          // streaming FastaChunkReader + zlib fallback decode
#ifdef CUHLL_HAVE_LIBDEFLATE
#include <cstdint>
#include <libdeflate.h>    // faster whole-file gzip decode (eager path)
#endif
#ifdef CUHLL_HAVE_ZSTD
#include <zstd.h>          // optional zstd-compressed input support (.zst)
#endif

namespace cuhll {

namespace {

constexpr int kGzipInflation = 4;

// Input compression format, detected from the file's leading magic bytes.
enum class Fmt { Plain, Gzip, Zstd };

Fmt detect_format(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return Fmt::Plain;  // open error surfaces later with a clear message
    unsigned char m[4] = {0, 0, 0, 0};
    std::size_t got = std::fread(m, 1, 4, f);
    std::fclose(f);
    if (got >= 2 && m[0] == 0x1f && m[1] == 0x8b) return Fmt::Gzip;
    if (got >= 4 && m[0] == 0x28 && m[1] == 0xb5 && m[2] == 0x2f && m[3] == 0xfd)
        return Fmt::Zstd;
    return Fmt::Plain;
}

// True iff `path` begins with the zstd magic (used to reject .zst in the
// streaming reader, which only supports plain/gzip).
bool is_zstd(const std::string& path) { return detect_format(path) == Fmt::Zstd; }

// Thread-local reusable scratch buffer. Grows monotonically, preserves
// contents on growth (like std::vector), but is NOT zero-initialized
// (`new char[]` leaves bytes indeterminate). Reusing one buffer per thread
// across files removes both the per-file memset and the first-touch page
// faults that show up when a fresh std::vector is allocated for every file
// (~7% of CPU in profiling, plus reduced memory-bandwidth pressure).
struct Scratch {
    std::unique_ptr<char[]> buf;
    std::size_t cap = 0;
    char* ensure(std::size_t n) {
        if (n > cap) {
            std::size_t ncap = cap ? cap : 4096;
            while (ncap < n) ncap *= 2;
            std::unique_ptr<char[]> nb(new char[ncap]);  // default-init: no zeroing
            if (buf && cap) std::memcpy(nb.get(), buf.get(), cap);  // preserve
            buf = std::move(nb);
            cap = ncap;
        }
        return buf.get();
    }
};

// A view over thread-local scratch; valid only until the next load on the
// same thread.
struct Bytes { const char* data; std::size_t size; };

// Read all raw bytes of `path` into scratch `s`. Returns bytes read.
std::size_t read_raw_into(const std::string& path, Scratch& s) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cuHLL: stat failed: " + path);
    }
    const std::size_t sz = static_cast<std::size_t>(st.st_size);
    char* dst = s.ensure(sz ? sz : 1);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cuHLL: cannot open: " + path);
    }
    std::size_t got = std::fread(dst, 1, sz, f);
    std::fclose(f);
    return got;
}

#ifdef CUHLL_HAVE_LIBDEFLATE
// Inflate gzip bytes comp[0..clen) into scratch `out` with libdeflate (~2x
// faster than zlib's gzread). Handles multi-member gzip (gzip(1), pigz, BGZF
// all produce back-to-back members) by looping, and grows `out` (preserving
// already-decoded members) if the initial size estimate was too small.
std::size_t gzip_inflate_into(const char* comp, std::size_t clen, Scratch& out,
                              const std::string& path) {
    libdeflate_decompressor* dec = libdeflate_alloc_decompressor();
    if (!dec) throw std::runtime_error("cuHLL: libdeflate_alloc_decompressor failed");

    out.ensure(clen * kGzipInflation + 4096);
    std::size_t out_pos = 0;
    const std::uint8_t* in = reinterpret_cast<const std::uint8_t*>(comp);
    std::size_t in_left = clen;

    // A gzip member is at least 18 bytes (10-byte header + 8-byte trailer).
    while (in_left >= 18 && in[0] == 0x1f && in[1] == 0x8b) {
        if (out.cap - out_pos < (1u << 16)) out.ensure(out.cap * 2);
        std::size_t got_in = 0, got_out = 0;

        // Fast path: a "plain" gzip member (deflate method, no optional header
        // fields: FLG==0) has a fixed 10-byte header followed by the raw
        // DEFLATE stream and an 8-byte trailer (CRC32 + ISIZE). Decoding the
        // raw DEFLATE directly skips libdeflate's CRC32 verification (~7% of
        // CPU), which we don't need for sketching. Any member with optional
        // fields (FEXTRA/FNAME/FCOMMENT/FHCRC) or a non-deflate method falls
        // back to the full, CRC-checked gzip decode — so robustness is intact.
        const bool simple = (in[2] == 8 /*deflate*/ && in[3] == 0 /*FLG==0*/);
        if (simple) {
            libdeflate_result r = libdeflate_deflate_decompress_ex(
                dec, in + 10, in_left - 10,
                out.buf.get() + out_pos, out.cap - out_pos,
                &got_in, &got_out);
            if (r == LIBDEFLATE_INSUFFICIENT_SPACE) { out.ensure(out.cap * 2); continue; }
            if (r != LIBDEFLATE_SUCCESS) {
                libdeflate_free_decompressor(dec);
                throw std::runtime_error("cuHLL: gzip(deflate) decode error on " + path);
            }
            out_pos += got_out;
            std::size_t consumed = 10 + got_in + 8;   // header + deflate + trailer
            if (consumed > in_left) consumed = in_left;
            in      += consumed;
            in_left -= consumed;
        } else {
            libdeflate_result r = libdeflate_gzip_decompress_ex(
                dec, in, in_left,
                out.buf.get() + out_pos, out.cap - out_pos,
                &got_in, &got_out);
            if (r == LIBDEFLATE_INSUFFICIENT_SPACE) { out.ensure(out.cap * 2); continue; }
            if (r != LIBDEFLATE_SUCCESS) {
                libdeflate_free_decompressor(dec);
                throw std::runtime_error("cuHLL: gzip decode error on " + path);
            }
            out_pos += got_out;
            in      += got_in;
            in_left -= got_in;
        }
    }
    libdeflate_free_decompressor(dec);
    // Trailing bytes that aren't a gzip member (e.g. zero padding) are ignored,
    // matching zlib's behavior of stopping at the stream end.
    return out_pos;
}
#else
// zlib fallback: incremental gzread into the growing scratch buffer.
std::size_t gzip_inflate_into(const std::string& path, Scratch& out) {
    gzFile zf = gzopen(path.c_str(), "rb");
    if (!zf) throw std::runtime_error("cuHLL: cannot open: " + path);
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        out.ensure(static_cast<std::size_t>(st.st_size) * kGzipInflation + 4096);
    }
    std::size_t out_pos = 0;
    constexpr unsigned CHUNK = 256 * 1024;
    for (;;) {
        if (out.cap - out_pos < CHUNK) out.ensure(out.cap * 2);
        int n = gzread(zf, out.buf.get() + out_pos,
                       static_cast<unsigned>(out.cap - out_pos));
        if (n <= 0) {
            if (n < 0) {
                int errnum = 0; const char* e = gzerror(zf, &errnum);
                gzclose(zf);
                throw std::runtime_error("cuHLL: read error on " + path + ": "
                                         + (e && e[0] ? e : "zlib error"));
            }
            break;
        }
        out_pos += static_cast<std::size_t>(n);
    }
    gzclose(zf);
    return out_pos;
}
#endif

#ifdef CUHLL_HAVE_ZSTD
// Decompress zstd bytes comp[0..clen) into scratch `out`. Uses the streaming
// API so it transparently handles single- and multi-frame inputs, growing
// `out` (preserving decoded bytes) when more space is needed.
std::size_t zstd_inflate_into(const char* comp, std::size_t clen, Scratch& out,
                              const std::string& path) {
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) throw std::runtime_error("cuHLL: ZSTD_createDCtx failed");
    out.ensure(clen * kGzipInflation + 4096);
    ZSTD_inBuffer in = { comp, clen, 0 };
    std::size_t out_pos = 0;
    while (in.pos < in.size) {
        // Keep >=1 MiB of headroom so ZSTD always has room to make progress.
        if (out.cap - out_pos < (1u << 20)) out.ensure(out.cap * 2);
        ZSTD_outBuffer ob = { out.buf.get(), out.cap, out_pos };
        std::size_t ret = ZSTD_decompressStream(dctx, &ob, &in);
        if (ZSTD_isError(ret)) {
            std::string e = ZSTD_getErrorName(ret);
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("cuHLL: zstd decode error on " + path + ": " + e);
        }
        out_pos = ob.pos;
    }
    ZSTD_freeDCtx(dctx);
    return out_pos;
}
#endif

// Load the (decompressed, if compressed) bytes of `path` into thread-local
// scratch and return a view. Handles plain, gzip, and zstd inputs (detected
// from magic bytes). The returned pointer is valid until the next
// load_sequence_bytes() call on the same thread.
Bytes load_sequence_bytes(const std::string& path) {
    thread_local Scratch raw;   // compressed bytes
    thread_local Scratch dec;   // plain / decompressed bytes
    const Fmt fmt = detect_format(path);

    if (fmt == Fmt::Plain) {
        std::size_t n = read_raw_into(path, dec);
        return {dec.buf.get(), n};
    }
    if (fmt == Fmt::Gzip) {
#ifdef CUHLL_HAVE_LIBDEFLATE
        std::size_t clen = read_raw_into(path, raw);
        std::size_t dlen = gzip_inflate_into(raw.buf.get(), clen, dec, path);
#else
        std::size_t dlen = gzip_inflate_into(path, dec);
#endif
        return {dec.buf.get(), dlen};
    }
    // Fmt::Zstd
#ifdef CUHLL_HAVE_ZSTD
    std::size_t clen = read_raw_into(path, raw);
    std::size_t dlen = zstd_inflate_into(raw.buf.get(), clen, dec, path);
    return {dec.buf.get(), dlen};
#else
    throw std::runtime_error(
        "cuHLL: input is zstd-compressed but cuHLL was built without libzstd: " + path);
#endif
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
    Bytes b = load_sequence_bytes(path);
    if (b.size == 0) return {};
    return parse_sequences(b.data, b.size);
}

std::size_t read_fasta_into(const std::string& path, char* dst,
                            std::size_t capacity) {
    CUHLL_NVTX_RANGE("read_fasta_into");
    Bytes b = load_sequence_bytes(path);
    if (b.size == 0) return 0;
    return parse_sequences_into(b.data, b.size, dst, capacity);
}

// -----------------------------------------------------------------------------
// FastaChunkReader
// -----------------------------------------------------------------------------
namespace {
constexpr std::size_t kRawBufBytes = 4ULL * 1024ULL * 1024ULL; // 4 MiB read-ahead
} // namespace

FastaChunkReader::FastaChunkReader(const std::string& path)
    : raw_buf_(kRawBufBytes) {
    // The streaming reader uses zlib, which decodes plain and gzip inputs but
    // would silently pass zstd bytes through as if uncompressed (→ garbage).
    // Reject .zst here with a clear message; zstd is supported on the eager
    // path (the default for batches and single-input union mode).
    if (is_zstd(path)) {
        throw std::runtime_error(
            "cuHLL: zstd input is not supported in the streaming path "
            "(used by --per-genome single-input); use the default mode: " + path);
    }
    gzFile zf = gzopen(path.c_str(), "rb");
    if (!zf) {
        throw std::runtime_error("cuHLL: cannot open input: " + path);
    }
    // Enlarge zlib's internal buffer (default 8 KiB) for better inflate
    // throughput on large inputs. Harmless for plain (non-gzip) files.
    gzbuffer(zf, 1u << 20);  // 1 MiB
    gzf_ = zf;
}

FastaChunkReader::~FastaChunkReader() {
    if (gzf_) gzclose(static_cast<gzFile>(gzf_));
}

void FastaChunkReader::refill_raw() {
    if (raw_pos_ < raw_len_) return;
    if (raw_eof_) {
        raw_pos_ = raw_len_ = 0;
        return;
    }
    const int n = gzread(static_cast<gzFile>(gzf_), raw_buf_.data(),
                         static_cast<unsigned>(raw_buf_.size()));
    if (n < 0) {
        int errnum = 0;
        const char* gzerr = gzerror(static_cast<gzFile>(gzf_), &errnum);
        throw std::runtime_error(
            std::string("cuHLL: read error: ")
            + (gzerr && gzerr[0] ? gzerr : "zlib error"));
    }
    raw_len_ = static_cast<std::size_t>(n);
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

        // Auto-detect the format from the first non-whitespace byte, then
        // fall through to handle this same byte under the chosen format.
        if (format_ == kFmtUnknown) {
            if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
            if (c == '>')      format_ = kFmtFasta;
            else if (c == '@') format_ = kFmtFastq;
            else throw std::runtime_error(
                "cuHLL: unknown sequence format — first non-whitespace byte "
                "must be '>' (FASTA) or '@' (FASTQ)");
        }

        if (format_ == kFmtFasta) {
            if (c == '>') {
                // Start of a new FASTA record. Every boundary except the
                // very first one injects an 'N' as a window-breaker so
                // k-mers can't span across unrelated records.
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
        } else { // kFmtFastq — 4-line records; emit only the sequence line.
            // Quality bytes can be any ASCII (incl. '@'/'+'), so boundaries
            // are tracked by line count mod 4, not by glyph.
            if (c == '\n') {
                fq_line_ = (fq_line_ + 1) % 4;
                if (fq_line_ == 1) {           // entering a sequence line
                    if (!fq_first_seq_) dst[out++] = 'N';
                    fq_first_seq_ = false;
                }
                continue;
            }
            if (c == '\r')              { continue; }
            if (fq_line_ != 1)          { continue; } // header / '+' / quality
            if (c == ' ' || c == '\t')  { continue; }
            dst[out++] = c;
        }
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

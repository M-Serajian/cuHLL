// =============================================================================
// test_fasta_stream.cpp — pure host test for FastaChunkReader.
//
// Proves the streaming chunk reader is byte-for-byte equivalent to the
// already-trusted eager reader (read_fasta_concat) across:
//   - FASTA and FASTQ
//   - plain and gzip-compressed (.gz) variants of each
//   - a range of chunk capacities, including tiny caps that force a chunk
//     boundary every few bytes (stressing cross-chunk parse state and
//     buffer refills)
//
// It also cross-checks that plain and gzip inputs decode identically, and
// that a FASTA and a FASTQ holding the same sequences yield the same bases.
//
// Self-contained: it generates its inputs (and writes the .gz variants via
// zlib directly), so it needs no external fixtures, no `gzip` binary, and
// no GPU.
// =============================================================================

#include "cuHLL/io/fasta.hpp"

#include <zlib.h>
#ifdef CUHLL_HAVE_ZSTD
#include <zstd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Tiny deterministic PRNG so the test is reproducible without <random> churn.
std::uint64_t g_state = 0x9E3779B97F4A7C15ULL;
char rnd_base() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return "ACGT"[g_state & 3u];
}

void write_file(const std::string& path, const std::string& data) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); std::exit(2); }
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

void write_gz(const std::string& path, const std::string& data) {
    gzFile z = gzopen(path.c_str(), "wb");
    if (!z) { std::fprintf(stderr, "gzopen failed: %s\n", path.c_str()); std::exit(2); }
    gzwrite(z, data.data(), static_cast<unsigned>(data.size()));
    gzclose(z);
}

#ifdef CUHLL_HAVE_ZSTD
void write_zst(const std::string& path, const std::string& data) {
    std::size_t bound = ZSTD_compressBound(data.size());
    std::vector<char> comp(bound);
    std::size_t n = ZSTD_compress(comp.data(), bound, data.data(), data.size(), 3);
    if (ZSTD_isError(n)) { std::fprintf(stderr, "zstd compress failed\n"); std::exit(2); }
    write_file(path, std::string(comp.data(), n));
}
#endif

// Write `data` as a gzip file split into `parts` back-to-back members
// (gzip "ab" appends a fresh member). Exercises the multi-member decode loop.
void write_gz_multimember(const std::string& path, const std::string& data, int parts) {
    write_file(path, "");  // truncate
    const std::size_t chunk = (data.size() + parts - 1) / parts;
    for (int i = 0; i < parts; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * chunk;
        if (off >= data.size()) break;
        const std::size_t n = std::min(chunk, data.size() - off);
        gzFile z = gzopen(path.c_str(), "ab");   // append a new member
        if (!z) { std::fprintf(stderr, "gzopen(ab) failed: %s\n", path.c_str()); std::exit(2); }
        gzwrite(z, data.data() + off, static_cast<unsigned>(n));
        gzclose(z);
    }
}

// Stream the whole file with overlap=0 and concatenate the fresh bytes; this
// must reproduce read_fasta_concat's output exactly.
std::string stream_all(const std::string& path, std::size_t cap) {
    cuhll::FastaChunkReader r(path);
    std::string out;
    std::vector<char> buf(cap);
    for (;;) {
        std::size_t n = r.next_chunk(buf.data(), buf.size(), /*overlap=*/0);
        if (n == 0) break;
        out.append(buf.data(), n);
    }
    return out;
}

int check_equiv(const std::string& path, std::size_t cap) {
    const std::string eager  = cuhll::read_fasta_concat(path);
    const std::string stream = stream_all(path, cap);
    if (eager != stream) {
        std::size_t i = 0;
        while (i < eager.size() && i < stream.size() && eager[i] == stream[i]) ++i;
        std::fprintf(stderr,
            "[FAIL test_fasta_stream] %s cap=%zu: eager(%zu) != stream(%zu), "
            "first diff at %zu\n",
            path.c_str(), cap, eager.size(), stream.size(), i);
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    std::fprintf(stderr, "[test_fasta_stream] streaming vs eager equivalence\n");

    // Build FASTA + FASTQ holding identical sequence content. ~6 MB of bases
    // so small chunk caps create thousands of boundaries.
    const int nreads = 40000, rlen = 100;
    std::string fasta, fastq;
    fasta.reserve(nreads * (rlen + 8));
    fastq.reserve(nreads * (2 * rlen + 12));
    for (int i = 0; i < nreads; ++i) {
        std::string seq;
        seq.reserve(rlen);
        for (int j = 0; j < rlen; ++j) seq.push_back(rnd_base());
        char hdr[32];
        std::snprintf(hdr, sizeof(hdr), "r%d", i);
        fasta += ">"; fasta += hdr; fasta += "\n"; fasta += seq; fasta += "\n";
        fastq += "@"; fastq += hdr; fastq += "\n"; fastq += seq;
        fastq += "\n+\n"; fastq += std::string(rlen, 'I'); fastq += "\n";
    }

    const std::string dir = "/tmp/cuhll_fasta_stream_test";
    std::string mk = "mkdir -p " + dir;
    if (std::system(mk.c_str()) != 0) { std::fprintf(stderr, "mkdir failed\n"); return 2; }

    const std::string fa  = dir + "/reads.fasta";
    const std::string faz = dir + "/reads.fasta.gz";
    const std::string fq  = dir + "/reads.fastq";
    const std::string fqz = dir + "/reads.fastq.gz";
    write_file(fa, fasta);   write_gz(faz, fasta);
    write_file(fq, fastq);   write_gz(fqz, fastq);

    int fail = 0;
    const std::size_t caps[] = {7, 64, 4096, 1u << 20};
    for (std::size_t cap : caps) {
        fail |= check_equiv(fa,  cap);
        fail |= check_equiv(faz, cap);
        fail |= check_equiv(fq,  cap);
        fail |= check_equiv(fqz, cap);
    }

    // Cross-checks on the eager decoder: plain==gz, and FASTA==FASTQ content.
    const std::string e_fa  = cuhll::read_fasta_concat(fa);
    const std::string e_faz = cuhll::read_fasta_concat(faz);
    const std::string e_fq  = cuhll::read_fasta_concat(fq);
    const std::string e_fqz = cuhll::read_fasta_concat(fqz);
    if (e_fa != e_faz) { std::fprintf(stderr, "[FAIL test_fasta_stream] fasta != fasta.gz\n"); fail = 1; }
    if (e_fq != e_fqz) { std::fprintf(stderr, "[FAIL test_fasta_stream] fastq != fastq.gz\n"); fail = 1; }
    if (e_fa != e_fq)  { std::fprintf(stderr, "[FAIL test_fasta_stream] fasta != fastq (same content)\n"); fail = 1; }

    // Buffer-growth path: highly compressible input so the decompressed size
    // far exceeds the 4x estimate, forcing the scratch buffer to grow (and
    // preserve already-decoded bytes) several times.
    std::string comp_fa;
    for (char b : std::string("ACGT"))
        comp_fa += ">" + std::string(1, b) + "\n" + std::string(1u << 20, b) + "\n";
    const std::string cfa  = dir + "/comp.fasta";
    const std::string cfaz = dir + "/comp.fasta.gz";
    write_file(cfa, comp_fa); write_gz(cfaz, comp_fa);
    for (std::size_t cap : caps) { fail |= check_equiv(cfa, cap); fail |= check_equiv(cfaz, cap); }
    if (cuhll::read_fasta_concat(cfaz) != cuhll::read_fasta_concat(cfa)) {
        std::fprintf(stderr, "[FAIL test_fasta_stream] compressible gz != plain (grow path)\n"); fail = 1; }

    // Multi-member gzip: the random FASTA written as 3 back-to-back members.
    const std::string mm = dir + "/reads.mm.fasta.gz";
    write_gz_multimember(mm, fasta, 3);
    for (std::size_t cap : caps) fail |= check_equiv(mm, cap);
    if (cuhll::read_fasta_concat(mm) != e_fa) {
        std::fprintf(stderr, "[FAIL test_fasta_stream] multi-member gz != single-member\n"); fail = 1; }

#ifdef CUHLL_HAVE_ZSTD
    // zstd input: decoding a .zst must match the plain source byte-for-byte.
    // (Streaming path rejects .zst, so this checks the eager zstd decoder.)
    const std::string faz_zst = dir + "/reads.fasta.zst";
    const std::string fqz_zst = dir + "/reads.fastq.zst";
    write_zst(faz_zst, fasta); write_zst(fqz_zst, fastq);
    if (cuhll::read_fasta_concat(faz_zst) != e_fa) {
        std::fprintf(stderr, "[FAIL test_fasta_stream] fasta.zst != plain fasta\n"); fail = 1; }
    if (cuhll::read_fasta_concat(fqz_zst) != e_fq) {
        std::fprintf(stderr, "[FAIL test_fasta_stream] fastq.zst != plain fastq\n"); fail = 1; }
#endif

    if (fail) {
        std::fprintf(stderr, "[FAIL test_fasta_stream]\n");
        return 1;
    }
    std::fprintf(stderr, "[PASS test_fasta_stream]\n");
    return 0;
}

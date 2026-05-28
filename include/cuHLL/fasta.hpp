#pragma once
// Sequence parsers — FASTA, FASTQ, and gzipped variants of both.
//
//   read_fasta_concat         — whole-file eager read, returned as one
//                                string of concatenated bases with 'N'
//                                injected between records. Auto-detects:
//                                  * gzip stream (magic 1f 8b)        → decompress
//                                  * '>' first byte                   → FASTA
//                                  * '@' first byte                   → FASTQ (4-line)
//                                The name is historical; FASTQ + .gz are
//                                handled transparently.
//   FastaChunkReader          — streaming chunk reader for plain FASTA.
//                                Used by the single-stream sequential pipeline.
//                                Headers stripped, whitespace removed,
//                                intra-file boundaries marked with a single 'N'.

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace cuhll {

std::string read_fasta_concat(const std::string& path);

// Parse directly into caller-provided buffer (e.g. pinned memory).
// Returns number of bytes written. Same format/correctness guarantees
// as read_fasta_concat.
std::size_t read_fasta_into(const std::string& path, char* dst,
                            std::size_t capacity);

class FastaChunkReader {
public:
    explicit FastaChunkReader(const std::string& path);

    // Fill dst with up to `cap` bytes of filtered FASTA content.
    //
    // Output layout:
    //   dst[0..tail_sz-1]     = the last `tail_sz` bytes emitted by the
    //                           previous next_chunk() call on this reader,
    //                           where tail_sz = min(overlap, prev.out).
    //                           Empty on the first call.
    //   dst[tail_sz..ret-1]   = newly parsed sequence bytes.
    //
    // Returns the total number of bytes written (tail + fresh). Returns 0 iff
    // there are no fresh bytes to parse AND no more useful work can be done
    // — i.e., EOF of the underlying file.
    //
    // The caller is expected to use overlap = k-1 on every call so that
    // k-mers spanning chunk boundaries are preserved. Overlap = 0 is legal
    // and means "no carry-over" (used for the very first chunk of each file,
    // where no predecessor bytes exist).
    std::size_t next_chunk(char* dst, std::size_t cap, std::size_t overlap);

    bool eof() const noexcept { return eof_; }

private:
    void refill_raw();

    std::ifstream        fin_;
    std::vector<char>    raw_buf_;
    std::size_t          raw_pos_   = 0;
    std::size_t          raw_len_   = 0;
    bool                 raw_eof_   = false;
    bool                 eof_       = false;

    bool                 in_header_          = false;
    bool                 first_record_seen_  = false;

    // Last `overlap` bytes of the previous emit, preserved across calls so
    // that chunk i+1 can begin with the tail of chunk i. Grows to at most
    // the largest `overlap` ever requested (k-1 in practice).
    std::vector<char>    tail_;
};

} // namespace cuhll

#pragma once
// Sequence parsers for FASTA, FASTQ, and gzipped variants of both.
//
// Two entry points:
//
//   read_fasta_concat(path)
//       Eager whole-file read. Returns a single std::string containing all
//       sequence bases concatenated together, with a single 'N' inserted
//       between records so k-mers cannot span a record boundary. Format is
//       detected from the file's first bytes:
//          1f 8b ...   gzip       (transparently decompressed)
//          >  ...      FASTA
//          @  ...      FASTQ (4-line records)
//       The function name is for historical reasons; FASTQ and .gz inputs
//       are handled by the same call.
//
//   FastaChunkReader (class below)
//       Streaming chunk-by-chunk reader for plain (non-gzipped) FASTA.
//       Used by the streaming pipeline so the input doesn't have to fit
//       in host RAM. Headers are stripped, whitespace removed, and the
//       last few bytes of each chunk are carried into the next chunk so
//       k-mers crossing chunk boundaries are not lost.

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace cuhll {

std::string read_fasta_concat(const std::string& path);

// Parse into a caller-provided buffer (e.g. pinned host memory). Returns
// the number of bytes written. Same format detection and 'N'-injection
// rules as read_fasta_concat.
std::size_t read_fasta_into(const std::string& path, char* dst,
                            std::size_t capacity);

class FastaChunkReader {
public:
    explicit FastaChunkReader(const std::string& path);

    // Fill `dst` with up to `cap` bytes of parsed sequence content.
    //
    // Output layout in `dst`:
    //   [0, tail_sz)         the trailing tail_sz bytes emitted by the
    //                        previous next_chunk() call (where
    //                        tail_sz = min(overlap, prev_returned_bytes));
    //                        empty on the first call.
    //   [tail_sz, returned)  fresh sequence bytes parsed from the file.
    //
    // Returns the total bytes written (tail + fresh). Returns 0 only when
    // there are no fresh bytes left to parse and the underlying file is
    // at EOF.
    //
    // The caller should pass overlap = k-1 so that every k-mer crossing a
    // chunk boundary appears in some chunk. Overlap = 0 is legal and
    // means "no carry-over" (used for the very first call on a file).
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

    // Holds the trailing bytes of the previous emit so the next call can
    // start with them. Grows to the largest overlap requested so far.
    std::vector<char>    tail_;
};

} // namespace cuhll

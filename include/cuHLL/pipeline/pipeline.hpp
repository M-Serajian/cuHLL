#pragma once
// Pure-C++ pipeline entry points for sketching a single Sketch from
// one or more input files. Safe to include from .cpp translation units.
//
// Two entry points:
//
//   sketch_sequence_single_stream
//       One-shot path: one H2D copy and one kernel launch on an
//       in-memory byte buffer. Useful for small inputs and tests.
//
//   sketch_sequences_streaming
//       Streaming path: a triple-buffered pinned host ring drives
//       3 CUDA streams with per-slot events. FASTA parsing, H2D copies,
//       and kernel launches overlap across slots so the GPU stays busy
//       while the next chunk is being read.

#include <cstddef>
#include <string>
#include <vector>

namespace cuhll {

class Sketch;

// One-shot path.
void sketch_sequence_single_stream(Sketch& sketch,
                                   const char* seq,
                                   std::size_t len,
                                   int k);

// Streaming path. Processes the inputs in `paths` one file at a time.
// Within a file, the parser injects a single 'N' between records so k-mer
// windows do not span record boundaries. Between files, the orchestrator
// injects an 'N' at the head of the first chunk of each subsequent file
// for the same reason.
//
// `chunk_mb` is the per-slot pinned-buffer size in MiB. Must be >= 1.
// Values that make chunk_bytes < k will degenerate the ring; the caller
// is expected to validate this.
void sketch_sequences_streaming(Sketch& sketch,
                                const std::vector<std::string>& paths,
                                int k,
                                std::size_t chunk_mb);

} // namespace cuhll

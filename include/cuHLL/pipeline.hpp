#pragma once
// Pure-C++ pipeline entry points. Safe to include from main.cpp.
//
// Two entry points live here now:
//
//   sketch_sequence_single_stream  — one-shot H2D + kernel on an in-memory
//                                    byte buffer. Used by small-file tests
//                                    and the milestone (c) fallback path.
//                                    One device allocation, one stream.
//
//   sketch_sequences_streaming     — milestone (d) production path. Triple-
//                                    buffered pinned ring + 3 CUDA streams +
//                                    per-slot events, overlapping FASTA parse,
//                                    H2D, and kernel across slots.

#include <cstddef>
#include <string>
#include <vector>

namespace cuhll {

class Sketch;

// Single-stream path (milestone c).
void sketch_sequence_single_stream(Sketch& sketch,
                                   const char* seq,
                                   std::size_t len,
                                   int k);

// Streaming pipeline (milestone d). Processes the FASTAs in `paths` one after
// another. Within a file, intra-file record boundaries are window-broken by
// the parser (injected 'N'); between files the orchestrator injects its own
// 'N' at the head of the first chunk of each secondary file for the same
// reason.
//
// `chunk_mb` is the per-slot pinned-buffer size in MiB. Must be >= 1. A value
// such that chunk_bytes < k degenerates the ring; the caller is expected to
// have validated this already (main.cpp enforces chunk_mb >= 1 at the CLI).
void sketch_sequences_streaming(Sketch& sketch,
                                const std::vector<std::string>& paths,
                                int k,
                                std::size_t chunk_mb);

// .cb2 path (Tier 1). Streams offline-packed 2-bit files into the same cuco
// sketch. Cross-file boundaries inject an 'N' window break via the mask.
// `chunk_mb` is interpreted in **mebibases** of logical sequence per chunk;
// per-chunk H2D is 3/8 of that in bytes (2 bits packed + 1 bit mask).
void sketch_sequences_cb2_streaming(Sketch& sketch,
                                    const std::vector<std::string>& cb2_paths,
                                    int k,
                                    std::size_t chunk_mb);

} // namespace cuhll

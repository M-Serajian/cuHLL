// =============================================================================
// _bindings.cpp — pybind11 wrapper for the cuHLL public C++ API.
//
// Surface mirrors include/cuHLL/{sketch,pipeline,hll_file,concurrent}.hpp.
// The Pythonic re-shaping (cuhll.estimate, cuhll.sketch, cuhll.sketch_many,
// cuhll.sketch_to_dir, the Sketch class with operator overloads and k-tracking)
// lives in python/cuhll/__init__.py — this file stays a thin, faithful binding.
//
// Compiled by pybind11_add_module (see CMakeLists.txt under
// `if(CUHLL_BUILD_PYTHON)`). Linked against cuhll_core for the actual
// CUDA work. No CUDA code lives in this file — it's plain C++17.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuHLL/common.hpp"
#include "cuHLL/concurrent.hpp"
#include "cuHLL/hll_file.hpp"
#include "cuHLL/pipeline.hpp"
#include "cuHLL/sketch.hpp"

namespace py = pybind11;

// One-shot helper used by the Pythonic `cuhll.sketch(path, k=...)` entry
// point. Always routes through the streaming pipeline so behaviour matches
// the cuhll CLI's default path for arbitrary FASTA sizes.
static cuhll::Sketch sketch_one_fasta(const std::string& path,
                                      int k,
                                      int precision,
                                      bool canonical,
                                      std::size_t chunk_mb) {
    cuhll::Sketch s(precision, canonical);
    cuhll::sketch_sequences_streaming(s, {path}, k, chunk_mb);
    return s;
}

// Sketch many FASTAs into a single Sketch (their union). Mirrors the
// CLI's default mode (`cuhll a.fa b.fa c.fa` with no --output-dir or
// --per-genome). One streaming pass through the pipeline accumulates
// k-mers from every input into the same registers, so it is faster
// than per-genome sketching followed by merge.
static cuhll::Sketch sketch_union_fastas(
        const std::vector<std::string>& paths,
        int k,
        int precision,
        bool canonical,
        std::size_t chunk_mb) {
    cuhll::Sketch s(precision, canonical);
    cuhll::sketch_sequences_streaming(s, paths, k, chunk_mb);
    return s;
}

PYBIND11_MODULE(_bindings, m) {
    m.doc() = "Low-level cuHLL bindings. Use the `cuhll` package for the "
              "Pythonic API; this module is private (subject to change).";

    // ------------------------------------------------------------------
    // Constants — mirror common.hpp + hll_file.hpp.
    // ------------------------------------------------------------------
    m.attr("kMinK")             = cuhll::kMinK;
    m.attr("kMaxK")             = cuhll::kMaxK;
    m.attr("kMinPrecision")     = cuhll::kMinPrecision;
    m.attr("kMaxPrecision")     = cuhll::kMaxPrecision;
    m.attr("kDefaultPrecision") = cuhll::kDefaultPrecision;
    m.attr("kDefaultChunkMB")   = cuhll::kDefaultChunkMB;
    m.attr("kHllFileVersion")   = cuhll::kHllFileVersion;

    // ------------------------------------------------------------------
    // HllFileHeader — read-only struct, exposed as a dataclass-like type
    // so callers can inspect a .hll file's metadata without loading
    // register data.
    // ------------------------------------------------------------------
    py::class_<cuhll::HllFileHeader>(m, "HllFileHeader",
        "On-disk header of a .hll file — version, k, precision, canonical "
        "flag, register-block size. See README's `.hll file format` section.")
        .def_readonly("version",        &cuhll::HllFileHeader::version)
        .def_readonly("precision_p",    &cuhll::HllFileHeader::precision_p)
        .def_readonly("k",              &cuhll::HllFileHeader::k)
        .def_readonly("hash_type",      &cuhll::HllFileHeader::hash_type)
        .def_readonly("n_registers",    &cuhll::HllFileHeader::n_registers)
        .def_readonly("register_bytes", &cuhll::HllFileHeader::register_bytes)
        .def_readonly("canonical",      &cuhll::HllFileHeader::canonical)
        .def("__repr__", [](const cuhll::HllFileHeader& h) {
            return "<cuhll.HllFileHeader version=" + std::to_string(h.version)
                 + " p=" + std::to_string(h.precision_p)
                 + " k=" + std::to_string(h.k)
                 + " canonical=" + (h.canonical ? "True" : "False") + ">";
        });

    // ------------------------------------------------------------------
    // Sketch — the core in-memory representation. Uses py::dynamic_attr
    // so the Pythonic wrapper in __init__.py can attach `.k` (the C++
    // Sketch class deliberately doesn't carry k).
    // ------------------------------------------------------------------
    py::class_<cuhll::Sketch>(m, "_Sketch", py::dynamic_attr(),
        "Raw cuHLL Sketch (RAII over cuco::hyperloglog). Prefer the "
        "Pythonic `cuhll.Sketch` wrapper which adds k-tracking and "
        "operator overloads.")
        .def(py::init<int, bool>(),
             py::arg("precision"), py::arg("canonical") = true,
             "Construct an empty Sketch with the given HLL precision and "
             "canonical mode.")
        .def("estimate", &cuhll::Sketch::estimate,
             "Distinct cardinality estimate as uint64.")
        .def("merge", &cuhll::Sketch::merge,
             py::arg("other"),
             "Union `other` into self (in place). Both sketches must have "
             "the same precision and canonical mode.")
        .def("clone", &cuhll::Sketch::clone,
             "Deep copy of this Sketch.")
        .def_property_readonly("precision", &cuhll::Sketch::precision)
        .def_property_readonly("canonical", &cuhll::Sketch::canonical)
        .def_property_readonly("sketch_bytes", &cuhll::Sketch::sketch_bytes,
             "Size of the register state in host bytes.");

    // ------------------------------------------------------------------
    // Top-level pipeline functions.
    // ------------------------------------------------------------------
    m.def("sketch_one_fasta", &sketch_one_fasta,
          py::arg("path"), py::arg("k"),
          py::arg("precision") = cuhll::kDefaultPrecision,
          py::arg("canonical") = true,
          py::arg("chunk_mb")  = cuhll::kDefaultChunkMB,
          "Sketch a single FASTA into a fresh Sketch (in memory).");

    m.def("sketch_union_fastas", &sketch_union_fastas,
          py::arg("paths"), py::arg("k"),
          py::arg("precision") = cuhll::kDefaultPrecision,
          py::arg("canonical") = true,
          py::arg("chunk_mb")  = cuhll::kDefaultChunkMB,
          "Sketch many FASTAs into ONE merged Sketch (their union). One "
          "streaming pass through the pipeline; faster than per-genome "
          "sketching followed by manual merge.");

    m.def("sketch_per_genome_auto", &cuhll::sketch_per_genome_auto,
          py::arg("paths"), py::arg("output_dir"),
          py::arg("k"), py::arg("precision"), py::arg("canonical"),
          "Concurrent per-genome path: sketches every input FASTA, writes "
          "<stem>.hll into output_dir, and returns the union cardinality "
          "as a uint64. The Python `cuhll.estimate_union` / "
          "`cuhll.sketch_to_dir` wrappers route through this.");

    // ------------------------------------------------------------------
    // .hll file I/O.
    // ------------------------------------------------------------------
    m.def("read_hll", &cuhll::read_hll,
          py::arg("path"),
          "Reconstruct a Sketch from a .hll file.");

    m.def("read_hll_header", &cuhll::read_hll_header,
          py::arg("path"),
          "Parse a .hll file header (no register data loaded).");

    m.def("write_hll", &cuhll::write_hll,
          py::arg("path"), py::arg("sketch"), py::arg("k"),
          "Persist a Sketch to a .hll file (overwrites).");
}

#pragma once
// Pure-C++ wrapper around cuco::hyperloglog.
//
// Header is safe to include from plain .cpp TUs (e.g., main.cpp, fasta.cpp).
// The cuco type lives inside an opaque SketchImpl that is completed only in
// CUDA TUs via `cuHLL/sketch_internal.cuh`.
//
// Hash choice: cuco::xxhash_64<uint64_t>. Initial design used
// cuco::identity_hash on the assumption that ntHash output is already
// uniform; that was wrong — ntHash per-kmer output is uniform but
// canonical = min(fwd, rc) biases the top bits toward 0, which breaks HLL's
// register selection (measured ~4.5x under-estimate on chr19@k=31 vs KMC3
// with identity_hash). xxhash_64 after the canonical-min restores uniformity.
// See sketch_internal.cuh for the actual instantiation.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuhll {

struct SketchImpl;
struct SketchImplDeleter {
    void operator()(SketchImpl*) const noexcept;
};

class Sketch {
public:
    // `canonical` controls whether the sketch counts canonical k-mers
    // (min(fwd, rc)) or non-canonical (fwd only). Default true matches all
    // prior milestones. Persisted in the .hll file header so that a round-
    // tripped sketch remembers its mode.
    explicit Sketch(int precision, bool canonical = true);
    ~Sketch();
    Sketch(Sketch&&) noexcept;
    Sketch& operator=(Sketch&&) noexcept;
    Sketch(const Sketch&) = delete;
    Sketch& operator=(const Sketch&) = delete;

    int  precision() const noexcept { return precision_; }
    bool canonical() const noexcept { return canonical_; }
    std::size_t sketch_bytes() const;

    void clear();
    void merge(const Sketch& other);
    std::uint64_t estimate() const;

    // Produce a deep copy of this sketch (same precision, same register state).
    // Uses cuco::hyperloglog::merge under the hood (merge into a freshly
    // zeroed sketch = plain copy of registers).
    Sketch clone() const;

    // Copy this sketch's HLL registers to a host-side buffer. `out` must hold
    // at least `sketch_bytes() / sizeof(uint32_t)` uint32 entries. cuco's
    // register type is int32_t in memory; we expose it as uint32_t for the
    // persistence layer — bit pattern is what matters for round-trip.
    void copy_registers_to_host(std::uint32_t* out) const;

    // Inverse: load HLL registers from a host-side buffer into this sketch.
    // `in` must hold exactly `sketch_bytes() / sizeof(uint32_t)` entries
    // matching this sketch's precision.
    void load_registers_from_host(const std::uint32_t* in);

    // CUDA-only access. Pure-C++ TUs can see the declaration but cannot use
    // the reference (SketchImpl is incomplete in this header). CUDA TUs
    // include sketch_internal.cuh to complete SketchImpl.
    SketchImpl& impl_ref() noexcept { return *impl_; }
    const SketchImpl& impl_ref() const noexcept { return *impl_; }

private:
    int  precision_;
    bool canonical_;
    std::unique_ptr<SketchImpl, SketchImplDeleter> impl_;
};

} // namespace cuhll

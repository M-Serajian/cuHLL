#pragma once
// Pure-C++ wrapper around cuco::hyperloglog.
//
// Safe to include from .cpp translation units. The cuco-typed payload lives
// inside an opaque `SketchImpl` that is completed only in CUDA TUs via
// `cuHLL/sketch/sketch_internal.cuh`.
//
// Hash choice: cuco::xxhash_64<uint64_t>. ntHash's per-k-mer output is
// uniform, but canonical = min(forward, reverse-complement) is not — it
// biases the top bits toward 0, which breaks HLL's register selection.
// xxhash_64 applied after the canonical min restores uniformity. The
// concrete instantiation lives in sketch_internal.cuh.

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
    // `precision` is the HLL p (number of registers = 2^p; default 14).
    // `canonical = true` counts canonical k-mers (min of forward and
    // reverse-complement); false counts forward-only. The flag is stored
    // in the .hll header so a round-tripped sketch remembers its mode.
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

    // Deep-copy: same precision, same register state.
    Sketch clone() const;

    // Copy this sketch's HLL registers to a host-side buffer. `out` must
    // hold at least sketch_bytes() / sizeof(uint32_t) entries. Registers
    // are exposed as uint32_t for persistence — cuco stores them as int32
    // internally but the bit pattern is what round-trips.
    void copy_registers_to_host(std::uint32_t* out) const;

    // Inverse of copy_registers_to_host. `in` must hold exactly
    // sketch_bytes() / sizeof(uint32_t) entries matching this sketch's
    // precision.
    void load_registers_from_host(const std::uint32_t* in);

    // CUDA-only access. .cpp TUs can see this declaration but cannot use
    // the reference (SketchImpl is incomplete here). CUDA TUs include
    // sketch_internal.cuh to complete the type.
    SketchImpl& impl_ref() noexcept { return *impl_; }
    const SketchImpl& impl_ref() const noexcept { return *impl_; }

private:
    int  precision_;
    bool canonical_;
    std::unique_ptr<SketchImpl, SketchImplDeleter> impl_;
};

} // namespace cuhll

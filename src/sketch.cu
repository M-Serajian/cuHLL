// sketch.cu — definitions for the host-facing Sketch class declared in
// sketch.hpp, backed by cuco::hyperloglog under the hood.

#include "cuHLL/cuda_check.hpp"
#include "cuHLL/sketch.hpp"
#include "cuHLL/sketch_internal.cuh"

namespace cuhll {

void SketchImplDeleter::operator()(SketchImpl* p) const noexcept {
    delete p;
}

Sketch::Sketch(int precision, bool canonical)
    : precision_(precision),
      canonical_(canonical),
      impl_(new SketchImpl(precision)) {}

Sketch::~Sketch() = default;
Sketch::Sketch(Sketch&&) noexcept = default;
Sketch& Sketch::operator=(Sketch&&) noexcept = default;

std::size_t Sketch::sketch_bytes() const {
    return impl_->sketch.sketch_bytes();
}

void Sketch::clear() {
    impl_->sketch.clear();
}

void Sketch::merge(const Sketch& other) {
    impl_->sketch.merge(other.impl_->sketch);
}

std::uint64_t Sketch::estimate() const {
    return static_cast<std::uint64_t>(impl_->sketch.estimate());
}

Sketch Sketch::clone() const {
    Sketch out(precision_, canonical_);
    // cuco merge: since `out` is fresh (all zero registers), merging our
    // state into it is equivalent to a copy. This avoids reaching into
    // cuco's internals for a raw byte-level memcpy.
    out.merge(*this);
    return out;
}

void Sketch::copy_registers_to_host(std::uint32_t* out) const {
    auto span = impl_->sketch.sketch();   // cuda::std::span<cuda::std::byte>
    const std::size_t bytes = span.size_bytes();
    CUDA_CHECK(cudaMemcpy(out, span.data(), bytes, cudaMemcpyDeviceToHost));
}

void Sketch::load_registers_from_host(const std::uint32_t* in) {
    auto span = impl_->sketch.sketch();
    const std::size_t bytes = span.size_bytes();
    CUDA_CHECK(cudaMemcpy(span.data(), in, bytes, cudaMemcpyHostToDevice));
}

} // namespace cuhll

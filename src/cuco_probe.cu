// Milestone (a) probe: confirm cuco::hyperloglog fetches, compiles under
// sm_89, and is loadable on an L4. No k-mer work, no ntHash, no pipeline.
// When cuHLL real targets land in milestone (c) this file goes away.

#include <cuco/hyperloglog.cuh>

#include <cstdint>
#include <cstdio>

int main() {
    using Sketch = cuco::hyperloglog<std::uint64_t>;

    Sketch sketch{cuco::precision{14}};
    std::printf("cuco_probe: precision=14 sketch_bytes=%zu\n",
                sketch.sketch_bytes());

    // Pull a device ref and read a host-side property off it to force the
    // ref_type instantiation (proof of concept that device-ref compiles too).
    auto ref = sketch.ref();
    std::printf("cuco_probe: ref sketch_bytes=%zu\n", ref.sketch_bytes());
    std::printf("cuco_probe: ok\n");
    return 0;
}

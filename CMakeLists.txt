cmake_minimum_required(VERSION 3.24)

# -----------------------------------------------------------------------------
# CUDA compute capability selection
#
# The project requires sm_70 (Volta) or newer: the HyperLogLog kernel in
# cuCollections uses device-scope atomic_ref, whose ptxas lowering requires
# `scopes on atomic operations` (sm_60+), and cuco pins sm_70 as its
# validated floor.
#
# Must be resolved before `project(... CUDA)` so CMake's CUDA language init
# sees the final architecture list.
#
# Knob: CMAKE_CUDA_ARCHITECTURES. Values accepted (in addition to any raw
# list CMake understands, e.g. "80;89;90" or "80-real"):
#
#     <unset>     Curated production list (CUHLL_DEFAULT_ARCHITECTURES).
#                 Builds anywhere CUDA is installed; the resulting fatbin
#                 runs on every GPU the project claims to support. This is
#                 the default so a bare `cmake ..` on a login node, CI
#                 runner, laptop, or container works out of the box.
#
#     NATIVE      Detect from the GPU on the build host. Explicit opt-in
#                 for the smallest, most-optimized single-arch binary.
#                 Fails cleanly when no GPU is visible instead of silently
#                 falling back to the nvcc default (sm_52).
#
#     ALL-MAJOR   One representative per supported major version. Smaller
#                 fatbin than the full curated list with broad coverage.
#
# No GPU or driver probing at configure time. This mirrors the rapids-cmake
# and cuCollections convention: build portability wins, because the build
# host (CI runners, login nodes, containers) often lacks a GPU entirely.
# The driver is a runtime concern, not a build concern.
# -----------------------------------------------------------------------------

# Hard floor: the lowest architecture the codebase will accept. The cuco
# HyperLogLog kernel uses device-scope atomics that require sm_60+; cuco
# itself pins sm_70 as its validated floor. This is NOT the same as the
# default fatbin list (CUHLL_DEFAULT_ARCHITECTURES) below — the default
# list is what we ship out of the box; the hard floor is what we'll still
# compile if a user opts into it explicitly.
set(CUHLL_MIN_ARCH 70)

# Curated list shipped in the default fatbin. Keep in ascending order.
# Update when a new GPU family is validated on real hardware; keep
# CUHLL_MIN_ARCH in sync only if the hard floor itself moves.
#   75 Turing   (T4)
#   80 Ampere   (A100)
#   86 Ampere   (A10, RTX 30xx)
#   89 Ada      (L4, RTX 40xx)
#   90 Hopper   (H100)
#
# sm_70 (Volta / V100) is intentionally NOT in the default list even
# though the hard floor still allows it. Reason: CUDA Toolkit 12.x emits
# a `Support for offline compilation for architectures prior to sm_75
# will be removed in a future release` warning for every pre-sm_75
# compile, producing dozens of warnings on a multi-TU build. The warning
# is correct — NVIDIA has announced Volta will be dropped — so mirroring
# NVIDIA's deprecation timing in the default list keeps the baseline
# build clean. V100 users opt back in with one flag:
#     cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
#
# Blackwell (sm_100) is intentionally NOT in the default either: it
# requires CUDA Toolkit >= 12.8 and the toolkit-version gate can only be
# evaluated after project(), which is too late to mutate the default
# list safely. Users on new hardware pass it explicitly:
#     cmake .. -DCMAKE_CUDA_ARCHITECTURES="...;100"
set(CUHLL_DEFAULT_ARCHITECTURES "75;80;86;89;90")

# Resolve the user-facing knob into a concrete list before project(). The
# _cuhll_arch_source string is only used for the status line below.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUHLL_DEFAULT_ARCHITECTURES}")
    set(_cuhll_arch_source "curated default")
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
    # Explicit opt-in to GPU detection. Probe for a visible GPU BEFORE
    # forwarding `native` to CMake/nvcc: without this check, CUDA 12.x
    # silently falls back to its default architecture (sm_52) on hosts
    # with no GPU, and the real failure surfaces much later as a cryptic
    # ptxas `scopes on atomic operations` error mid-build. This is the
    # same silent-fallback that bites users who run `cmake ..` on login
    # nodes; we catch it here with a clear, actionable message.
    find_program(CUHLL_NVIDIA_SMI nvidia-smi)
    set(_nvsmi_rc 1)
    set(_nvsmi_out "")
    if(CUHLL_NVIDIA_SMI)
        execute_process(
            COMMAND "${CUHLL_NVIDIA_SMI}" -L
            RESULT_VARIABLE _nvsmi_rc
            OUTPUT_VARIABLE _nvsmi_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(NOT _nvsmi_rc EQUAL 0 OR _nvsmi_out STREQUAL "")
        message(FATAL_ERROR
            "cuHLL: -DCMAKE_CUDA_ARCHITECTURES=NATIVE was requested but "
            "no GPU was detected on the build host (nvidia-smi is absent "
            "or reports no devices). Either build inside a GPU allocation "
            "(e.g. `srun --gres=gpu:1 ...`), or pass an explicit "
            "architecture list such as -DCMAKE_CUDA_ARCHITECTURES=89.")
    endif()
    unset(_nvsmi_rc)
    unset(_nvsmi_out)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(_cuhll_arch_source "build-host GPU (NATIVE)")
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL-MAJOR")
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
    set(_cuhll_arch_source "CMake all-major")
else()
    set(_cuhll_arch_source "user-provided")
endif()

# Floor validation for explicit numeric entries. Skip CMake's special
# tokens (`native`, `all`, `all-major`) — CMake owns their expansion.
set(_cuhll_special_archs "native;all;all-major")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_arch IN_LIST _cuhll_special_archs)
        continue()
    endif()
    # Entries may carry CMake 3.18+ suffixes: "80", "80-real", "80-virtual".
    string(REGEX MATCH "^([0-9]+)" _arch_num "${_arch}")
    if(NOT _arch_num)
        message(FATAL_ERROR
            "cuHLL: cannot parse CUDA architecture entry '${_arch}'. "
            "Use a numeric capability (e.g. '89'), an optional suffix "
            "('89-real', '89-virtual'), or one of: NATIVE, ALL-MAJOR.")
    endif()
    if(_arch_num LESS CUHLL_MIN_ARCH)
        message(FATAL_ERROR
            "cuHLL: requested sm_${_arch_num} is below the minimum "
            "sm_${CUHLL_MIN_ARCH}. The HyperLogLog kernel uses device-scope "
            "atomics which require Volta or newer. "
            "Override with -DCMAKE_CUDA_ARCHITECTURES=<list|NATIVE|ALL-MAJOR>.")
    endif()
endforeach()
unset(_cuhll_special_archs)

message(STATUS
    "cuHLL: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES} "
    "(${_cuhll_arch_source}; "
    "override with -DCMAKE_CUDA_ARCHITECTURES=<list|NATIVE|ALL-MAJOR>)")
unset(_cuhll_arch_source)

# -----------------------------------------------------------------------------
# Workaround: conda-bundled toolchain interfering with separately-installed
# gcc. Two distinct symptoms, one root cause (conda Python module activates
# its sysroot first):
#
# (1) Conda `ld` shadows the system linker on PATH.
#     Conda Pythons ship `bin/ld -> x86_64-conda-linux-gnu-ld`. Gcc resolves
#     `ld` via PATH lookup and picks up the conda binutils — which was
#     linked against an older glibc and fails to link binaries built with
#     the newer system glibc:
#         conda/bin/ld: libstdc++.so: undefined reference to
#             `pthread_create@GLIBC_2.34'
#     This blows up at CMake's compiler-id detection inside project(), so
#     the user never sees a useful error.
#
# (2) Conda libstdc++.so.6 is too old at runtime.
#     Even after (1) is fixed and the wheel builds, `import cuhll` fails:
#         ImportError: ...conda/lib/libstdc++.so.6: version
#             `GLIBCXX_3.4.32' not found
#     because the conda Python's executable carries DT_RPATH=$ORIGIN/../lib,
#     which forces every `dlopen` from that interpreter to find the conda
#     libstdc++ first — and it predates the gcc that compiled the binding.
#     Setting DT_RUNPATH on the binding doesn't help (executable RPATH
#     wins). The robust fix is to statically link libstdc++ and libgcc
#     into the binding so it has no runtime dependency on whichever
#     libstdc++ the host Python ships.
#
# Detect the conda toolchain (conda ld first on PATH AND distinct from
# system ld) and apply both fixes: redirect ld via `-B /usr/bin/`, and
# request static libstdc++/libgcc on the bindings target. Python /
# pip / cmake from the same module are unaffected — only `ld` lookup
# and the bindings' libstdc++ linkage change. Disable the entire
# workaround with -DCUHLL_KEEP_CONDA_LD=ON.
# -----------------------------------------------------------------------------
set(CUHLL_CONDA_LD_WORKAROUND OFF)
find_program(_cuhll_path_ld NAMES ld NO_CACHE)
if(_cuhll_path_ld)
    get_filename_component(_cuhll_ld_real "${_cuhll_path_ld}" REALPATH)
    if(_cuhll_ld_real MATCHES "conda-linux-gnu-ld$" AND NOT CUHLL_KEEP_CONDA_LD)
        set(CUHLL_CONDA_LD_WORKAROUND ON)
        if(NOT EXISTS "/usr/bin/ld")
            message(FATAL_ERROR
                "cuHLL: detected a conda-bundled linker first on PATH "
                "(${_cuhll_path_ld} -> ${_cuhll_ld_real}) but no system ld at "
                "/usr/bin/ld to redirect to. This conda ld is ABI-incompatible "
                "with the libstdc++ shipped by separately-installed gcc and "
                "will fail with `undefined reference to ...@GLIBC_2.xx`. "
                "Either unload the Python module providing the conda toolchain, "
                "install system binutils so /usr/bin/ld exists, or pass "
                "-DCUHLL_KEEP_CONDA_LD=ON if you've handled this another way.")
        endif()
        message(STATUS
            "cuHLL: detected conda-bundled linker at ${_cuhll_path_ld} "
            "(-> ${_cuhll_ld_real}); injecting `-B /usr/bin/` so gcc resolves "
            "ld to the system linker instead. "
            "Disable with -DCUHLL_KEEP_CONDA_LD=ON.")
        # CMake reads *_FLAGS_INIT during project() to seed *_FLAGS, which
        # is what the compiler-id test and every subsequent build command
        # use. Prepend rather than overwrite so user-supplied flags win.
        set(CMAKE_C_FLAGS_INIT    "-B /usr/bin/ ${CMAKE_C_FLAGS_INIT}")
        set(CMAKE_CXX_FLAGS_INIT  "-B /usr/bin/ ${CMAKE_CXX_FLAGS_INIT}")
        # nvcc forwards host-compiler flags via -Xcompiler; the same -B
        # makes nvcc's host link step pick up /usr/bin/ld too.
        set(CMAKE_CUDA_FLAGS_INIT "-Xcompiler=-B,/usr/bin/ ${CMAKE_CUDA_FLAGS_INIT}")
        # The compile-step _FLAGS variables don't reach the link command —
        # CMake invokes the linker via g++ with *_LINKER_FLAGS instead. Seed
        # those too so try_compile()'s link step (and every real link, e.g.
        # `g++ ... -ldl -lpthread`) routes through /usr/bin/ld.
        set(CMAKE_EXE_LINKER_FLAGS_INIT
            "-B /usr/bin/ ${CMAKE_EXE_LINKER_FLAGS_INIT}")
        set(CMAKE_SHARED_LINKER_FLAGS_INIT
            "-B /usr/bin/ ${CMAKE_SHARED_LINKER_FLAGS_INIT}")
        set(CMAKE_MODULE_LINKER_FLAGS_INIT
            "-B /usr/bin/ ${CMAKE_MODULE_LINKER_FLAGS_INIT}")
    endif()
    unset(_cuhll_ld_real)
endif()
unset(_cuhll_path_ld)

# scikit-build-core forwards pyproject.toml's [project].version into
# SKBUILD_PROJECT_VERSION at configure time, so a `pip install` build
# always sees the canonical version. For raw `cmake ..` builds outside
# pip, fall back to a hardcoded value — bump it together with
# pyproject.toml on a release. Either way, only one file needs editing
# in the common (pip) workflow.
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(_cuhll_project_version ${SKBUILD_PROJECT_VERSION})
else()
    set(_cuhll_project_version 1.0.0)
endif()

project(cuHLL
    VERSION ${_cuhll_project_version}
    DESCRIPTION "GPU-accelerated distinct canonical k-mer estimator backed by cuco::hyperloglog"
    LANGUAGES CXX CUDA)
unset(_cuhll_project_version)

# -----------------------------------------------------------------------------
# Language standards + warnings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# Centralize build outputs under build/bin and build/lib for a predictable
# layout matching modern CMake conventions (RAPIDS, gRPC, abseil). Multi-
# config generators (Ninja Multi-Config, Visual Studio) still split per-config
# subdirs underneath these automatically.
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# -----------------------------------------------------------------------------
# Dependencies (CPM + cuco + CUDA toolkit)
# -----------------------------------------------------------------------------
include(cmake/CPMDownload.cmake)

# cuco (NVIDIA/cuCollections). Pinned to the commit on `dev` that was green
# at milestone (a). The SHA is also recorded in README under Reproducibility.
# To update: bump GIT_TAG, rebuild the cuco_probe target under srun, and if
# green, update both this SHA and the README entry in the same commit.
set(CUHLL_CUCO_TAG "6477be2182668015f9a91e3a0bb7e248eceecd09"
    CACHE STRING "cuCollections git tag/SHA to fetch via CPM")

CPMAddPackage(
    NAME cuco
    GITHUB_REPOSITORY NVIDIA/cuCollections
    GIT_TAG ${CUHLL_CUCO_TAG}
    OPTIONS
      "CUCO_BUILD_TESTS OFF"
      "CUCO_BUILD_BENCHMARKS OFF"
      "CUCO_BUILD_EXAMPLES OFF"
)

find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Milestone (a) probe target
# -----------------------------------------------------------------------------
# One tiny CUDA executable that instantiates cuco::hyperloglog<uint64_t> on the
# host and prints its sketch byte count. Its only job is to prove that:
#   - CPM fetched cuco successfully
#   - <cuco/hyperloglog.cuh> resolves and compiles under our toolchain
#   - The link against CUDA::cudart works
#   - The final binary loads on an L4 (sm_89) without ptxas / loader errors
# Real `cuhll_core`, `cuhll_gpu`, and `cuhll` targets land in later milestones.
add_executable(cuco_probe src/cuco_probe.cu)
target_include_directories(cuco_probe PRIVATE include)
target_link_libraries(cuco_probe PRIVATE cuco::cuco CUDA::cudart)

# -----------------------------------------------------------------------------
# cuhll_core: shared static library containing everything the cuhll executable
# and the milestone (e) tests need. Tests link against this rather than
# recompiling each source from scratch.
# -----------------------------------------------------------------------------
add_library(cuhll_core STATIC
    src/fasta.cpp
    src/sketch.cu
    src/kmer_kernel.cu
    src/kmer_kernel_cb2.cu
    src/cb2.cpp
    src/hll_file.cpp
    src/pipeline.cu
    src/concurrent.cpp
    src/pipeline_concurrent.cu
)
target_include_directories(cuhll_core PUBLIC include)
target_link_libraries(cuhll_core PUBLIC cuco::cuco CUDA::cudart)
set_target_properties(cuhll_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# -----------------------------------------------------------------------------
# cuhll executable. main.cpp is plain C++; the CUDA TUs live in cuhll_core.
# -----------------------------------------------------------------------------
add_executable(cuhll src/main.cpp)
target_link_libraries(cuhll PRIVATE cuhll_core)
set_target_properties(cuhll PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# -----------------------------------------------------------------------------
# cuhll_pack tool: offline FASTA -> .cb2 converter (Tier 1).
# -----------------------------------------------------------------------------
add_executable(cuhll_pack tools/cuhll_pack.cpp)
target_link_libraries(cuhll_pack PRIVATE cuhll_core)

# -----------------------------------------------------------------------------
# Tests (CTest)
#
# Defaults OFF so end users get a fast `cmake .. && make -j` that builds
# only the runtime artifacts (cuhll, cuhll_pack, cuco_probe) plus the
# cuhll_core library. Developers and CI opt in with -DCUHLL_BUILD_TESTS=ON.
#
# Variable is project-namespaced (matches CUCO_BUILD_TESTS / CCCL_ENABLE_TESTING)
# so it cannot collide with a parent project's CTest BUILD_TESTING when cuHLL
# is consumed as a subdirectory.
# -----------------------------------------------------------------------------
option(CUHLL_BUILD_TESTS "Build cuHLL test suite (requires GPU to run)" OFF)
if(CUHLL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Python bindings (pybind11)
#
# Defaults OFF so the C++/CUDA build doesn't pull in Python development
# headers when nobody asked. scikit-build-core flips this to ON
# automatically when building via `pip install .` (see pyproject.toml's
# `tool.scikit-build.cmake.args`).
#
# The bindings target (`_bindings`) is a Python extension module that
# wraps cuhll_core. The Pythonic API surface lives in
# python/cuhll/__init__.py.
# -----------------------------------------------------------------------------
option(CUHLL_BUILD_PYTHON
    "Build Python bindings (requires Python development headers)" OFF)

if(CUHLL_BUILD_PYTHON)
    # pybind11 — fetched alongside cuco/CCCL via CPM for a uniform deps
    # story. Pinned to a recent stable release; bump together with the
    # version requirement in pyproject.toml's [build-system].requires.
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        GIT_TAG v2.13.6
        EXCLUDE_FROM_ALL TRUE
        SYSTEM TRUE
    )

    # Python extension module. The TU itself is plain C++ (pybind11
    # macros only), but we force CUDA compilation so CMake's nvcc-aware
    # link step embeds cuhll_core's fatbin into the .so. Without this
    # the MODULE library links without device symbols and `import cuhll`
    # fails at load time with `undefined symbol: fatbinData`. Same
    # pattern RAPIDS uses for cudf/cuml's pybind11 modules.
    set_source_files_properties(python/cuhll/_bindings.cpp
        PROPERTIES LANGUAGE CUDA)
    pybind11_add_module(_bindings MODULE python/cuhll/_bindings.cpp)
    target_link_libraries(_bindings PRIVATE cuhll_core)
    set_target_properties(_bindings PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION  ON
    )

    # When a conda Python's old libstdc++ would shadow gcc's at import
    # time (see the conda-toolchain workaround above project()), embed
    # libstdc++/libgcc statically into the binding so its runtime
    # behaviour is independent of whichever libstdc++ the host Python
    # ships. Adds ~1–2 MB to the .so but eliminates the GLIBCXX_3.4.xx
    # not-found class of import errors entirely.
    if(CUHLL_CONDA_LD_WORKAROUND)
        target_link_options(_bindings PRIVATE
            -static-libstdc++ -static-libgcc)
    endif()

    # Install destination is relative; scikit-build-core merges it under
    # the wheel layout (alongside python/cuhll/__init__.py).
    install(TARGETS _bindings DESTINATION cuhll)
endif()

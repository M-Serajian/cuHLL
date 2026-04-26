# cuHLL tests
#
# Each test is a standalone executable that prints "[PASS] <name>" or
# "[FAIL] <name>: <reason>" on stderr and exits 0 on pass, non-zero on fail.
#
# Tests that depend on a FASTA input emit "[SKIP] ..." and exit 0 when the
# CUHLL_TEST_FASTA (or CUHLL_TEST_FASTA2) environment variable is unset or
# points at a missing file. The SKIP_REGULAR_EXPRESSION property below
# promotes those to "Skipped" in CTest's summary (not "Passed").

set(_cuhll_skip_regex "\\[SKIP\\]")

# ----------------------------------------------------------------------
# test_nthash — self-contained, does not need cuhll_core or a FASTA.
# ----------------------------------------------------------------------
add_executable(test_nthash test_nthash.cu)
target_include_directories(test_nthash PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_nthash PRIVATE CUDA::cudart)
set_target_properties(test_nthash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_nthash
         COMMAND test_nthash
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ----------------------------------------------------------------------
# GPU tests linked against cuhll_core. Input FASTAs come from the
# CUHLL_TEST_FASTA / CUHLL_TEST_FASTA2 environment variables; tests
# skip cleanly if those are not set.
# ----------------------------------------------------------------------
set(_gpu_tests
    test_tiny_exact
    test_determinism
    test_merge
    test_cb2_correctness
    test_hll_roundtrip
    test_merge_equivalence
    test_canonical_flag
    test_synthetic
)
foreach(t IN LISTS _gpu_tests)
    add_executable(${t} ${t}.cu)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${t} PRIVATE cuhll_core)
    set_target_properties(${t} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME ${t}
             COMMAND ${t}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(${t} PROPERTIES
        SKIP_REGULAR_EXPRESSION "${_cuhll_skip_regex}")
endforeach()

# ----------------------------------------------------------------------
# test_auto_tune — pure C++, no GPU or FASTA needed.
# ----------------------------------------------------------------------
add_executable(test_auto_tune test_auto_tune.cpp)
target_link_libraries(test_auto_tune PRIVATE cuhll_core)
add_test(NAME test_auto_tune
         COMMAND test_auto_tune
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ----------------------------------------------------------------------
# test_against_kmc (Layer 2) — cross-implementation validation against
# KMC3's exact distinct-k-mer count.
#
# Resolution strategy mirrors rapids-cmake's "prefer system, fall back
# to source" pattern:
#
#   1. find_program(kmc) first. If the user has KMC pre-installed
#      (e.g. via HiPerGator's `kmc/3.2.1` module, or a system package),
#      we use that and pay no build cost.
#
#   2. Otherwise ExternalProject_Add fetches refresh-bio/KMC at a pinned
#      tag and builds it via its native Makefile under the build
#      directory. The compiled binary becomes a build-time dependency
#      of test_against_kmc.
#
# Either way the same `test_against_kmc` CTest entry runs the validation
# script against the resolved binary, so behaviour is identical across
# environments. Set CUHLL_TEST_FASTA at test time to validate against
# real data; otherwise the script synthesises a small FASTA on the fly.
# ----------------------------------------------------------------------
find_program(CUHLL_KMC_BIN kmc)
set(_kmc_test_deps "")

if(CUHLL_KMC_BIN)
    message(STATUS "cuHLL: using system KMC at ${CUHLL_KMC_BIN}")
else()
    message(STATUS "cuHLL: KMC not on PATH; will fetch and build via ExternalProject for tests")
    include(ExternalProject)
    set(_kmc_prefix "${CMAKE_BINARY_DIR}/_deps/kmc")
    ExternalProject_Add(kmc_external
        PREFIX            "${_kmc_prefix}"
        GIT_REPOSITORY    https://github.com/refresh-bio/KMC.git
        GIT_TAG           v3.2.1
        GIT_SHALLOW       TRUE
        UPDATE_DISCONNECTED TRUE
        BUILD_IN_SOURCE   TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     make -j
        INSTALL_COMMAND   ""
        BUILD_BYPRODUCTS  <SOURCE_DIR>/bin/kmc
    )
    ExternalProject_Get_Property(kmc_external SOURCE_DIR)
    set(CUHLL_KMC_BIN "${SOURCE_DIR}/bin/kmc"
        CACHE FILEPATH "kmc binary built by ExternalProject" FORCE)
    list(APPEND _kmc_test_deps kmc_external)
endif()

add_test(NAME test_against_kmc
         COMMAND bash ${CMAKE_SOURCE_DIR}/tools/validate_against_kmc.sh
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(test_against_kmc PROPERTIES
    ENVIRONMENT "CUHLL_BIN=$<TARGET_FILE:cuhll>;CUHLL_KMC_BIN=${CUHLL_KMC_BIN}")

# Drag the ExternalProject build (if any) into the default `make` so
# `make && ctest` is the canonical workflow even on hosts without KMC.
if(_kmc_test_deps)
    add_custom_target(cuhll_test_deps ALL DEPENDS ${_kmc_test_deps})
endif()

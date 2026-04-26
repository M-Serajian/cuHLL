# CPM.cmake bootstrap.
#
# Downloads CPM.cmake once on first configure and caches it under
# cmake/CPM.cmake so subsequent configures are offline-safe. The network
# operation runs at CMake configure time, which is allowed to happen on a
# login node (configure invokes no nvcc). The build step itself must be
# run inside an srun allocation per the project's compute rules.

set(CUHLL_CPM_VERSION "0.40.5" CACHE STRING "CPM.cmake version to fetch")
set(CUHLL_CPM_URL
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CUHLL_CPM_VERSION}/CPM.cmake"
    CACHE STRING "CPM.cmake download URL")

set(_cpm_local "${CMAKE_SOURCE_DIR}/cmake/CPM.cmake")

if(NOT EXISTS "${_cpm_local}")
    message(STATUS "cuHLL: fetching CPM.cmake v${CUHLL_CPM_VERSION} -> ${_cpm_local}")
    file(DOWNLOAD
         "${CUHLL_CPM_URL}"
         "${_cpm_local}"
         STATUS _cpm_status
         TLS_VERIFY ON
         SHOW_PROGRESS)
    list(GET _cpm_status 0 _cpm_code)
    if(NOT _cpm_code EQUAL 0)
        # Clean up the partial file so the next configure retries.
        file(REMOVE "${_cpm_local}")
        message(FATAL_ERROR
            "cuHLL: failed to download CPM.cmake (status=${_cpm_status}).\n"
            "  URL: ${CUHLL_CPM_URL}\n"
            "  Target: ${_cpm_local}\n"
            "Fixes:\n"
            "  - Run `cmake -S . -B build` from a node with internet access "
            "(the login node is fine; configure never invokes nvcc).\n"
            "  - Or manually place the file at ${_cpm_local}.")
    endif()
endif()

include("${_cpm_local}")

// tests/test_common.hpp — shared helpers for the CTest suite.
//
// Centralizes two concerns every test had baked in as hardcoded paths:
//
//   1. The FASTA input(s). Tests read them from CUHLL_TEST_FASTA (and, where
//      needed, CUHLL_TEST_FASTA2). If the variable is unset or the file is
//      missing, the test prints "[SKIP] ..." to stderr and returns 0.
//      CMakeLists registers SKIP_REGULAR_EXPRESSION so CTest reports the
//      test as "Skipped" rather than "Passed".
//
//   2. The scratch directory. Tests that need a writable working dir use
//      scratch_dir(name) which places it under the system temp directory
//      with the caller's PID appended so concurrent ctest runs never clash.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace cuhll_test {

inline std::string env_or_empty(const char* var) {
    const char* v = std::getenv(var);
    return (v && *v) ? std::string(v) : std::string{};
}

// Resolve a FASTA path from an environment variable. If unset or missing,
// prints a SKIP message the test's main() should forward as a 0-exit.
// Returns empty string to signal "skip".
inline std::string fasta_or_skip(const char* var, const char* test_name) {
    const std::string p = env_or_empty(var);
    if (p.empty()) {
        std::fprintf(stderr,
            "[SKIP] %s: %s is not set (set it to a FASTA file)\n",
            test_name, var);
        return {};
    }
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
        std::fprintf(stderr,
            "[SKIP] %s: %s points at a missing file: %s\n",
            test_name, var, p.c_str());
        return {};
    }
    return p;
}

// Per-process scratch directory under the system temp dir. Created fresh;
// caller is expected to clean it up at the end of the test run.
inline std::filesystem::path scratch_dir(const char* name) {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path();
    fs::path dir  = root / (std::string("cuhll_") + name + "_"
                            + std::to_string(::getpid()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

} // namespace cuhll_test

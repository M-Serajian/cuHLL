#pragma once
// Minimal test harness for the Phase 1 oracle tests.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace cuhll::abundance {
inline int& fail_count() { static int n = 0; return n; }

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
        ++::cuhll::abundance::fail_count(); \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va == _vb)) { \
        std::printf("  [FAIL] %s:%d  CHECK_EQ(%s, %s)  got %lld vs %lld\n", \
            __FILE__, __LINE__, #a, #b, \
            (long long)_va, (long long)_vb); \
        ++::cuhll::abundance::fail_count(); \
    } \
} while (0)

inline int report(const char* name) {
    if (fail_count() == 0) { std::printf("[PASS] %s\n", name); return 0; }
    std::printf("[FAIL] %s: %d failed checks\n", name, fail_count());
    return 1;
}
}  // namespace cuhll::abundance

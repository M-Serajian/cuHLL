#pragma once
// abundance_statistics.hpp — regularized incomplete beta I_x(a,b), its inverse (Beta
// quantile), and the one-sided Clopper-Pearson upper bound.
//
// No Beta-quantile exists in the project's dependencies (no Boost; the cccl hits
// are test-only), so this is hand-rolled. A subtly wrong beta inverse would
// silently corrupt the exact CP guarantee, so `test_betainc` validates it
// against known values (closed forms + scipy references) BEFORE the estimator
// uses it — see kmin_max/tests/test_abundance_statistics.cpp.
//
// Method: Lentz continued fraction for I_x(a,b) (Numerical Recipes betacf/betai),
// bisection for the inverse (bulletproof; monotone CDF on [0,1]).

#include <cmath>
#include <cstdint>

namespace cuhll::abundance {

inline double betacf(double a, double b, double x) {
    const int    MAXIT = 300;
    const double EPS   = 3.0e-14;
    const double FPMIN = 1.0e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c; h *= del;
        if (std::fabs(del - 1.0) < EPS) break;
    }
    return h;
}

// Regularized incomplete beta I_x(a,b) in [0,1].
inline double betai(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double lbeta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double bt = std::exp(lbeta + a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Inverse of I_x(a,b): the p-quantile of Beta(a,b). Bisection on [0,1].
inline double beta_quantile(double p, double a, double b) {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    double lo = 0.0, hi = 1.0, mid = 0.5;
    for (int i = 0; i < 200; ++i) {
        mid = 0.5 * (lo + hi);
        const double v = betai(a, b, mid);
        if (v < p) lo = mid; else hi = mid;
        if (hi - lo < 1.0e-14) break;
    }
    return mid;
}

// Inverse standard-normal CDF (Acklam's rational approximation; |err| < 1.2e-9).
// Used to turn a split confidence 1-delta_f0 into its z-quantile for the F0
// one-sided upper. Validated in test_betainc.
inline double normal_quantile(double p) {
    if (p <= 0.0) return -1e308;
    if (p >= 1.0) return  1e308;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
         2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
         2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    double q, r;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// One-sided Clopper-Pearson UPPER bound for k successes in m trials at
// confidence 1-delta. Exact (inverts the binomial tail): guaranteed >= nominal
// coverage. p_upper = BetaInv(1-delta, k+1, m-k); = 1 when k == m.
inline double clopper_pearson_upper(std::uint64_t k, std::uint64_t m,
                                    double delta) {
    if (m == 0) return 1.0;
    if (k >= m) return 1.0;
    return beta_quantile(1.0 - delta, static_cast<double>(k) + 1.0,
                         static_cast<double>(m - k));
}

}  // namespace cuhll::abundance

// test_abundance_statistics.cpp — MANDATORY self-test for the hand-rolled beta routines,
// run BEFORE the estimator uses them. References from scipy
// (beta.ppf / betainc) and closed forms.

#include "cuHLL/abundance/abundance_statistics.hpp"
#include "abundance_test_common.hpp"

#include <cmath>
#include <cstdio>

using namespace cuhll::abundance;

static void approx(const char* what, double got, double want, double tol=1e-6){
    double d = std::fabs(got - want);
    if (d > tol) {
        std::printf("  [FAIL] %s: got %.10f want %.10f (|d|=%.2e)\n", what, got, want, d);
        ++fail_count();
    } else {
        std::printf("  [ ok ] %s = %.10f (ref %.10f)\n", what, got, want);
    }
}

int main(){
    std::printf("test_abundance_statistics:\n");

    // (1) Forward I_x(a,b) vs closed forms / scipy.
    approx("I_0.5(1,1)",  betai(1,1,0.5), 0.5);
    approx("I_0.3(1,10)", betai(1,10,0.3), 1.0 - std::pow(0.7,10)); // 0.9717524751
    approx("I_0.5(3,1)",  betai(3,1,0.5), std::pow(0.5,3));         // 0.125
    // symmetry I_x(a,b) = 1 - I_{1-x}(b,a)
    approx("symmetry I_0.37(2.5,4.2)",
           betai(2.5,4.2,0.37), 1.0 - betai(4.2,2.5,0.63));

    // (2) Beta quantiles vs scipy beta.ppf.
    approx("qbeta(0.95,1,10)", beta_quantile(0.95,1,10), 0.2588655509);
    approx("qbeta(0.95,2,9)",  beta_quantile(0.95,2,9),  0.3941633024);
    approx("qbeta(0.95,4,7)",  beta_quantile(0.95,4,7),  0.6066242161);
    approx("qbeta(0.99,2,9)",  beta_quantile(0.99,2,9),  0.5043526629);
    approx("qbeta(0.99,4,7)",  beta_quantile(0.99,4,7),  0.7028835277);
    approx("qbeta(0.99,1,20)", beta_quantile(0.99,1,20), 0.2056717653);

    // (3) Clopper-Pearson upper: classic 0/10, 1/10, 3/10 at 95% and 99%.
    //     CP_upper(k,m,delta) = qbeta(1-delta, k+1, m-k).
    approx("CP 0/10 @95%", clopper_pearson_upper(0,10,0.05), 0.2588655509); // closed: 1-0.05^.1
    approx("CP 1/10 @95%", clopper_pearson_upper(1,10,0.05), 0.3941633024);
    approx("CP 3/10 @95%", clopper_pearson_upper(3,10,0.05), 0.6066242161);
    approx("CP 1/10 @99%", clopper_pearson_upper(1,10,0.01), 0.5043526629);
    approx("CP 3/10 @99%", clopper_pearson_upper(3,10,0.01), 0.7028835277);
    approx("CP 0/20 @99%", clopper_pearson_upper(0,20,0.01), 0.2056717653); // closed: 1-0.01^.05
    // k==m edge -> 1.0
    approx("CP 10/10 @99%", clopper_pearson_upper(10,10,0.01), 1.0);

    // (4) Round-trip: betai(beta_quantile(p,a,b),a,b) == p.
    for (double p : {0.5, 0.9, 0.95, 0.99}) {
        for (auto ab : {std::pair<double,double>{2,9}, {4,7}, {1,50}, {30,30}}) {
            double x = beta_quantile(p, ab.first, ab.second);
            double back = betai(ab.first, ab.second, x);
            char buf[64]; std::snprintf(buf,sizeof buf,"roundtrip p=%.2f a=%.0f b=%.0f",p,ab.first,ab.second);
            approx(buf, back, p, 1e-6);
        }
    }

    // (4b) Inverse standard-normal CDF (for the split-delta F0 z-quantile).
    approx("Phi^-1(0.975)", normal_quantile(0.975), 1.959963985, 1e-6);
    approx("Phi^-1(0.990)", normal_quantile(0.990), 2.326347874, 1e-6);
    approx("Phi^-1(0.995)", normal_quantile(0.995), 2.575829304, 1e-6);
    approx("Phi^-1(0.900)", normal_quantile(0.900), 1.281551566, 1e-6);

    // (5) Monotonicity in k.
    double prev=-1; bool mono=true;
    for (std::uint64_t k=0;k<=10;++k){ double u=clopper_pearson_upper(k,10,0.01);
        if (u<prev) mono=false; prev=u; }
    CHECK(mono);

    return report("test_abundance_statistics");
}

#include "kun/cellular/statistical_evaluation.hpp"
#include "tasks/transfer/doudizhu_eval_harness.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace kun;

void test_normal_quantile() {
    assert(normal_quantile(0.5) == 0.0);

    double z95 = normal_quantile(0.975);
    double z95_neg = normal_quantile(0.025);
    assert(std::abs(z95 - 1.959964) < 1e-4);
    assert(std::abs(z95_neg + 1.959964) < 1e-4);

    double z99 = normal_quantile(0.995);
    double z99_neg = normal_quantile(0.005);
    assert(std::abs(z99 - 2.575829) < 1e-4);
    assert(std::abs(z99_neg + 2.575829) < 1e-4);

    double z90 = normal_quantile(0.95);
    double z90_neg = normal_quantile(0.05);
    assert(std::abs(z90 - 1.644854) < 1e-4);
    assert(std::abs(z90_neg + 1.644854) < 1e-4);

    assert(normal_quantile(-0.5) == -10.0);
    assert(normal_quantile(1.5) == 10.0);
}

void test_wilson_score_interval() {
    // 0/100 test
    auto ci0 = wilson_score_interval(0, 100, 0.95);
    assert(ci0.point == 0.0);
    assert(ci0.lower == 0.0);
    assert(ci0.upper > 0.0 && ci0.upper < 0.05);

    // 100/100 test
    auto ci100 = wilson_score_interval(100, 100, 0.95);
    assert(ci100.point == 1.0);
    assert(ci100.lower > 0.95 && ci100.lower < 1.0);
    assert(ci100.upper == 1.0);

    // 500/1000 test
    auto ci500 = wilson_score_interval(500, 1000, 0.95);
    assert(std::abs(ci500.point - 0.5) < 1e-6);
    assert(ci500.lower < 0.5 && ci500.upper > 0.5);
    assert(std::abs(ci500.point - (ci500.lower + ci500.upper) / 2.0) < 1e-3);
    assert(ci500.lower > 0.46 && ci500.upper < 0.54);

    // Edge case total = 0
    auto ci_empty = wilson_score_interval(0, 0, 0.95);
    assert(ci_empty.point == 0.0);
    assert(ci_empty.lower == 0.0);
    assert(ci_empty.upper == 0.0);

    // Edge case successes > total
    auto ci_over = wilson_score_interval(120, 100, 0.95);
    assert(ci_over.point == 1.0);
    assert(ci_over.upper == 1.0);
}

void test_mcnemar_test() {
    // Zero discordant pairs -> p should be 1.0
    auto m_zero = mcnemar_test(0, 0, 100, 100, true);
    assert(m_zero.chi2 == 0.0);
    assert(m_zero.p_value == 1.0);
    assert(!m_zero.significant_01);

    // Symmetric discordant pairs (b = c) -> p should be 1.0
    auto m_sym = mcnemar_test(50, 50, 200, 200, true);
    assert(m_sym.chi2 == 0.0);
    assert(m_sym.p_value == 1.0);
    assert(!m_sym.significant_01);
    assert(!m_sym.significant_05);

    // Highly discordant pairs: b=100, c=10
    auto m_disc = mcnemar_test(100, 10, 500, 500, true);
    assert(m_disc.chi2 > 50.0);
    assert(m_disc.p_value < 1e-6);
    assert(m_disc.significant_01);
    assert(m_disc.significant_05);

    // Vector interface test
    std::vector<int> a = {1, 1, 0, 1, 0, 0, 1};
    std::vector<int> b = {1, 0, 1, 1, 0, 1, 0};
    auto m_vec = mcnemar_test(a, b, true);
    assert(m_vec.n == 7);
}

void test_paired_difference() {
    std::vector<double> x = {10.0, 20.0, 30.0, 40.0, 50.0};
    std::vector<double> y = {5.0,  15.0, 25.0, 35.0, 45.0};
    // diff is always 5.0, zero variance -> infinite t-statistic
    auto diff_res = analyze_paired_differences(x, y, 0.95);
    assert(std::abs(diff_res.mean_diff - 5.0) < 1e-6);
    assert(std::abs(diff_res.std_dev) < 1e-6);
    assert(std::isinf(diff_res.t_stat) && diff_res.t_stat > 0.0);
    assert(diff_res.p_value == 0.0);
    assert(diff_res.significant_01);

    // Identical samples -> zero mean diff, zero variance -> t = 0, p = 1.0
    auto diff_ident = analyze_paired_differences(x, x, 0.95);
    assert(std::abs(diff_ident.mean_diff) < 1e-6);
    assert(diff_ident.t_stat == 0.0);
    assert(diff_ident.p_value == 1.0);
    assert(!diff_ident.significant_05);

    // Random differences with zero mean
    std::vector<double> d = {1.0, -1.0, 2.0, -2.0, 0.5, -0.5};
    auto diff_zero = analyze_paired_differences(d, 0.95);
    assert(std::abs(diff_zero.mean_diff) < 1e-6);
    assert(!diff_zero.significant_05);
}

void test_doudizhu_eval_harness_quick() {
    // Quick run on 50 episodes to test harness plumbing
    DouDiZhuThreeBaselineHarness harness(50, 2026, 5.0, 40);
    assert(harness.total_episodes() == 50);
    assert(harness.baseline_0().total_episodes == 50);
    assert(harness.baseline_1().total_episodes == 50);

    // Evaluate heuristic policy with decision calls counting
    uint64_t policy_decisions = 0;
    auto report = harness.evaluate_policy("Heuristic Follow", [](const auto&, auto&) {
        return 1; // Always clean follow
    }, &policy_decisions);

    assert(report.total_episodes == 50);
    assert(report.target.total_episodes == 50);
    assert(report.target.win_rate >= 0.0 && report.target.win_rate <= 100.0);
    assert(policy_decisions > 0);

    // Test print_table output doesn't crash
    std::ostringstream oss;
    report.print_table(oss);
    assert(!oss.str().empty());
}

int main() {
    test_normal_quantile();
    test_wilson_score_interval();
    test_mcnemar_test();
    test_paired_difference();
    test_doudizhu_eval_harness_quick();

    std::cout << "[PASS] test_statistical_evaluation passed all assertions!\n";
    return 0;
}

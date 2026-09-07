#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 底座数学统计与实证检验引擎
// (Substrate Mathematical Statistics & Empirical Hypothesis Testing)
//
// 体系结构规范:
// 1. 纯粹数学与统计学原语: 威尔逊得分区间 (Wilson Score Interval)、
//    麦克尼马尔配对卡方检验 (McNemar's Test with Edwards' Continuity Correction)、
//    配对样本差值推断 (Paired Difference Estimation)。
// 2. 绝对铁律: 恪守 Rule 7 底座中立性与底座免疫准则，严禁任何具身业务/博弈/专有词汇。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>

namespace kun {
namespace cellular {

/**
 * @brief 置信区间容器 (Point Estimate & Confidence Interval)
 */
struct ConfidenceInterval {
    double point{0.0};
    double lower{0.0};
    double upper{0.0};
    double confidence_level{0.95};

    std::string to_string(int precision = 2, bool as_percentage = true) const {
        std::ostringstream oss;
        double scale = as_percentage ? 100.0 : 1.0;
        const char* suffix = as_percentage ? "%" : "";
        oss << std::fixed << std::setprecision(precision)
            << (point * scale) << suffix << " ["
            << (lower * scale) << suffix << ", "
            << (upper * scale) << suffix << "]";
        return oss.str();
    }
};

/**
 * @brief 麦克尼马尔配对二值检验结果 (McNemar's Test Result)
 */
struct McNemarResult {
    uint64_t n{0};               // 配对总样本数 (a + b + c + d)
    uint64_t b{0};               // 策略 A 成功且策略 B 失败
    uint64_t c{0};               // 策略 A 失败且策略 B 成功
    uint64_t a{0};               // 两者均成功
    uint64_t d{0};               // 两者均失败
    double chi2{0.0};            // 自由度 df=1 的卡方统计量
    double p_value{1.0};         // 双侧渐进 p 值
    bool significant_01{false};  // p < 0.01 (极显著差异)
    bool significant_05{false};  // p < 0.05 (显著差异)

    std::string to_string(int precision = 4) const {
        std::ostringstream oss;
        oss << "b=" << b << ", c=" << c
            << ", chi2=" << std::fixed << std::setprecision(precision) << chi2
            << ", p=" << std::scientific << std::setprecision(precision) << p_value
            << " (" << (significant_01 ? "p<0.01 显著" : (significant_05 ? "p<0.05 弱显著" : "不显著")) << ")";
        return oss.str();
    }
};

/**
 * @brief 配对差值统计推断结果 (Paired Continuous Difference Analysis)
 */
struct PairedDifferenceResult {
    size_t n{0};
    double mean_diff{0.0};
    double sample_var{0.0};
    double std_dev{0.0};
    double std_err{0.0};
    double ci_lower{0.0};
    double ci_upper{0.0};
    double t_stat{0.0};
    double p_value{1.0};
    bool significant_01{false};
    bool significant_05{false};

    std::string to_string(int precision = 3) const {
        std::ostringstream oss;
        oss << "mean_diff=" << std::fixed << std::setprecision(precision) << mean_diff
            << " +/- " << std_err << " 95% CI [" << ci_lower << ", " << ci_upper << "]"
            << ", t=" << t_stat
            << ", p=" << std::scientific << std::setprecision(precision) << p_value;
        return oss.str();
    }
};

/**
 * @brief 标准正态分布分位数函数近似 (Normal Quantile Function Probit Approximation)
 */
inline double normal_quantile(double p) {
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return 10.0;
    if (p == 0.5) return 0.0;
    if (std::abs(p - 0.975) < 1e-6) return 1.959963984540054;
    if (std::abs(p - 0.025) < 1e-6) return -1.959963984540054;
    if (std::abs(p - 0.995) < 1e-6) return 2.575829303548900;
    if (std::abs(p - 0.005) < 1e-6) return -2.575829303548900;
    if (std::abs(p - 0.950) < 1e-6) return 1.644853626951472;
    if (std::abs(p - 0.050) < 1e-6) return -1.644853626951472;

    // Rational approximation for central & tails
    double q = (p < 0.5) ? p : (1.0 - p);
    double t = std::sqrt(-2.0 * std::log(q));
    // Coefficients for rational approximation
    const double c0 = 2.515517;
    const double c1 = 0.802853;
    const double c2 = 0.010328;
    const double d1 = 1.432788;
    const double d2 = 0.189269;
    const double d3 = 0.001308;
    double z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0);
    return (p < 0.5) ? -z : z;
}

/**
 * @brief 威尔逊得分置信区间 (Wilson Score Interval for Binomial Proportions)
 *
 * 相比朴素 Wald 正态近似，小样本与极端比例 (0% 或 100%) 下严格覆盖无退化，无越界 [0, 1]。
 */
inline ConfidenceInterval wilson_score_interval(
    uint64_t successes,
    uint64_t total,
    double confidence_level = 0.95)
{
    ConfidenceInterval ci;
    confidence_level = std::clamp(confidence_level, 0.001, 0.9999);
    ci.confidence_level = confidence_level;
    if (total == 0) {
        ci.point = 0.0;
        ci.lower = 0.0;
        ci.upper = 0.0;
        return ci;
    }

    if (successes > total) {
        successes = total;
    }

    double n = static_cast<double>(total);
    double p_hat = static_cast<double>(successes) / n;
    ci.point = p_hat;

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha * 0.5);
    double z2 = z * z;

    double denom = 1.0 + z2 / n;
    double center = (p_hat + z2 / (2.0 * n)) / denom;
    double radicand = (p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n);
    double spread = (z / denom) * std::sqrt(std::max(0.0, radicand));

    if (successes == 0) {
        ci.lower = 0.0;
    } else {
        ci.lower = std::clamp(center - spread, 0.0, 1.0);
    }

    if (successes == total) {
        ci.upper = 1.0;
    } else {
        ci.upper = std::clamp(center + spread, 0.0, 1.0);
    }
    return ci;
}

/**
 * @brief 麦克尼马尔检验 (McNemar's Test for Paired Binary Outcomes)
 *
 * 计算两组完全配对的二值判决在不一致项 (b, c) 上的对称性。
 * 默认采用 Edwards 连续性修正: chi2 = (|b - c| - 1)^2 / (b + c)
 */
inline McNemarResult mcnemar_test(
    uint64_t b,
    uint64_t c,
    uint64_t a = 0,
    uint64_t d = 0,
    bool continuity_correction = true)
{
    McNemarResult res;
    res.b = b;
    res.c = c;
    res.a = a;
    res.d = d;
    res.n = a + b + c + d;

    uint64_t discordant = b + c;
    if (discordant == 0) {
        res.chi2 = 0.0;
        res.p_value = 1.0;
        res.significant_01 = false;
        res.significant_05 = false;
        return res;
    }

    double diff = std::abs(static_cast<double>(b) - static_cast<double>(c));
    if (continuity_correction) {
        double num = std::max(0.0, diff - 1.0);
        res.chi2 = (num * num) / static_cast<double>(discordant);
    } else {
        res.chi2 = (diff * diff) / static_cast<double>(discordant);
    }

    // 自由度为 1 的卡方分布右尾概率: p = erfc(sqrt(chi2 / 2))
    res.p_value = std::clamp(std::erfc(std::sqrt(res.chi2 * 0.5)), 0.0, 1.0);
    res.significant_01 = (res.p_value < 0.01);
    res.significant_05 = (res.p_value < 0.05);
    return res;
}

/**
 * @brief 基于配对向量的麦克尼马尔检验重载
 */
inline McNemarResult mcnemar_test(
    const std::vector<int>& outcomes_a,
    const std::vector<int>& outcomes_b,
    bool continuity_correction = true)
{
    size_t n = std::min(outcomes_a.size(), outcomes_b.size());
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i < n; ++i) {
        bool sa = (outcomes_a[i] != 0);
        bool sb = (outcomes_b[i] != 0);
        if (sa && sb) a++;
        else if (sa && !sb) b++;
        else if (!sa && sb) c++;
        else d++;
    }
    return mcnemar_test(b, c, a, d, continuity_correction);
}

/**
 * @brief 配对差值样本统计分析 (Paired Continuous Differences)
 */
inline PairedDifferenceResult analyze_paired_differences(
    const std::vector<double>& diffs,
    double confidence_level = 0.95)
{
    PairedDifferenceResult res;
    res.n = diffs.size();
    if (res.n == 0) return res;

    double sum = 0.0;
    for (double v : diffs) sum += v;
    res.mean_diff = sum / static_cast<double>(res.n);

    if (res.n < 2) {
        res.sample_var = 0.0;
        res.std_dev = 0.0;
        res.std_err = 0.0;
        res.ci_lower = res.mean_diff;
        res.ci_upper = res.mean_diff;
        res.t_stat = 0.0;
        res.p_value = 1.0;
        return res;
    }

    double sum_sq = 0.0;
    for (double v : diffs) {
        double d = v - res.mean_diff;
        sum_sq += d * d;
    }
    res.sample_var = sum_sq / static_cast<double>(res.n - 1);
    res.std_dev = std::sqrt(std::max(0.0, res.sample_var));
    res.std_err = res.std_dev / std::sqrt(static_cast<double>(res.n));

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha * 0.5);
    res.ci_lower = res.mean_diff - z * res.std_err;
    res.ci_upper = res.mean_diff + z * res.std_err;

    if (res.std_err > 1e-15) {
        res.t_stat = res.mean_diff / res.std_err;
        res.p_value = std::clamp(std::erfc(std::abs(res.t_stat) / std::sqrt(2.0)), 0.0, 1.0);
    } else {
        if (std::abs(res.mean_diff) < 1e-12) {
            res.t_stat = 0.0;
            res.p_value = 1.0;
        } else {
            res.t_stat = (res.mean_diff > 0.0) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
            res.p_value = 0.0;
        }
    }

    res.significant_01 = (res.p_value < 0.01);
    res.significant_05 = (res.p_value < 0.05);
    return res;
}

inline PairedDifferenceResult analyze_paired_differences(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double confidence_level = 0.95)
{
    size_t n = std::min(x.size(), y.size());
    std::vector<double> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        diffs[i] = x[i] - y[i];
    }
    return analyze_paired_differences(diffs, confidence_level);
}

} // namespace cellular

// Export at kun namespace level as well for convenience
using cellular::ConfidenceInterval;
using cellular::McNemarResult;
using cellular::PairedDifferenceResult;
using cellular::wilson_score_interval;
using cellular::mcnemar_test;
using cellular::analyze_paired_differences;
using cellular::normal_quantile;

} // namespace kun

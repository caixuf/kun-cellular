#pragma once
// ============================================================================
// population/pool_ecology.hpp — L1 系统层: 组合级生态评估原语 (纯数学)
// 职责: 给定共存池成员的逐日 returns 矩阵 + 任务层冻结的权重常数,
//       输出池级指标与个体级边际贡献 (LOO)、寄生度 (相关性)、生态位空缺度。
// 无任何业务语义; 常数全部由调用方注入, L1 不携带默认策略权重。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace kun {
namespace population {

// 任务层注入并冻结的评估常数 (v1 量化战役预注册值由 L2 传入)
struct PoolFitnessParams {
    double sharpe_weight = 2.0;             // 池级夏普权重
    double calmar_weight = 1.0;             // 池级卡玛权重
    double mdd_weight = 2.0;                // 池级最大回撤惩罚权重
    double parasitism_corr_threshold = 0.90;// 寄生判定相关阈
    double parasitism_penalty = 1.0;        // 寄生惩罚系数
    double regime_gap_penalty = 0.5;        // 生态位空缺惩罚系数
    uint32_t regime_count = 3;              // 时间轴等分体制数
    double periods_per_year = 252.0;        // 年化因子 (日频回放)
    double min_eval_days = 20;             // 最少评估样本 (低于此判无效)
};

struct MemberMetrics {
    double loo_contrib = 0.0;    // 留一法边际贡献: 移除该成员后池适应度跌幅
    double max_corr = 0.0;       // 与池内最相关成员的 |pearson r|
    double regime_gap = 0.0;     // 生态位空缺度: 池亏损且该成员也无正贡献的体制占比
    double score = 0.0;          // 综合个体分 = loo − 寄生惩罚 − 空缺惩罚
};

struct PoolMetrics {
    bool valid = false;
    double sharpe = 0.0;
    double ann_return = 0.0;
    double max_drawdown = 0.0;
    double calmar = 0.0;
    double fitness = 0.0;        // 池级适应度 = w_s·sharpe + w_c·calmar − w_m·mdd
    std::vector<MemberMetrics> members;
};

namespace detail {

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

inline double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    const double ma = mean(a), mb = mean(b);
    double num = 0.0, da = 0.0, db = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i] - ma, y = b[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    if (da <= 1e-18 || db <= 1e-18) return 0.0;
    return num / std::sqrt(da * db);
}

// 等权池逐日收益 (weights 为空时等权; 否则按给定权重)
inline std::vector<double> pool_returns(const std::vector<std::vector<double>>& rs,
                                        const std::vector<double>& weights) {
    const size_t n = rs.empty() ? 0 : rs[0].size();
    std::vector<double> out(n, 0.0);
    const size_t k = rs.size();
    if (k == 0 || n == 0) return out;
    for (size_t i = 0; i < k; ++i) {
        const double w = weights.empty() ? (1.0 / static_cast<double>(k)) : weights[i];
        for (size_t t = 0; t < n; ++t) out[t] += w * rs[i][t];
    }
    return out;
}

struct CurveStats {
    double sharpe = 0.0, ann_return = 0.0, mdd = 0.0, calmar = 0.0, fitness = 0.0;
};

inline CurveStats curve_stats(const std::vector<double>& rets, const PoolFitnessParams& p) {
    CurveStats s;
    const double n = static_cast<double>(rets.size());
    if (rets.size() < static_cast<size_t>(p.min_eval_days)) return s;
    const double sd = stdev(rets);
    s.sharpe = (sd > 1e-12) ? (mean(rets) / sd) * std::sqrt(p.periods_per_year) : 0.0;

    double equity = 1.0, peak = 1.0, cum_end = 1.0;
    for (double r : rets) {
        equity *= (1.0 + r);
        peak = std::max(peak, equity);
        if (peak > 1e-12) s.mdd = std::max(s.mdd, 1.0 - equity / peak);
        cum_end = equity;
    }
    const double cum = cum_end - 1.0;
    s.ann_return = (cum_end > 0.0 && n > 0.0)
        ? std::pow(cum_end, p.periods_per_year / n) - 1.0 : 0.0;
    s.calmar = (s.mdd > 1e-9) ? (s.ann_return / s.mdd) : (s.ann_return > 0.0 ? 99.0 : 0.0);
    s.fitness = p.sharpe_weight * s.sharpe + p.calmar_weight * s.calmar - p.mdd_weight * s.mdd;
    return s;
}

}  // namespace detail

// ── 组合级生态评估 ────────────────────────────────────────────────────────────
// 输入: rs[i] = 成员 i 的逐日 returns (时间对齐, 长度一致)
inline PoolMetrics evaluate_pool(const std::vector<std::vector<double>>& rs,
                                 const PoolFitnessParams& p,
                                 const std::vector<double>& weights = {}) {
    PoolMetrics out;
    const size_t k = rs.size();
    if (k == 0) return out;
    const size_t n = rs[0].size();
    for (const auto& r : rs) {
        if (r.size() != n || n < static_cast<size_t>(p.min_eval_days)) return out;
    }
    out.valid = true;

    // 池级
    const auto full = detail::pool_returns(rs, weights);
    const auto full_stats = detail::curve_stats(full, p);
    out.sharpe = full_stats.sharpe;
    out.ann_return = full_stats.ann_return;
    out.max_drawdown = full_stats.mdd;
    out.calmar = full_stats.calmar;
    out.fitness = full_stats.fitness;

    // 时间轴等分体制
    const uint32_t rc = std::max<uint32_t>(p.regime_count, 1);
    const size_t block = n / rc;

    out.members.resize(k);
    for (size_t i = 0; i < k; ++i) {
        MemberMetrics& mm = out.members[i];

        // 1) LOO 边际贡献
        std::vector<std::vector<double>> without_i;
        without_i.reserve(k - 1);
        for (size_t j = 0; j < k; ++j) {
            if (j != i) without_i.push_back(rs[j]);
        }
        if (!without_i.empty()) {
            const auto loo_stats = detail::curve_stats(detail::pool_returns(without_i, weights), p);
            mm.loo_contrib = full_stats.fitness - loo_stats.fitness;
        } else {
            mm.loo_contrib = full_stats.fitness;  // 单成员池: 贡献=全部
        }

        // 2) 寄生度 (与池内最相关成员)
        double worst = 0.0;
        for (size_t j = 0; j < k; ++j) {
            if (j == i) continue;
            worst = std::max(worst, std::fabs(detail::pearson(rs[i], rs[j])));
        }
        mm.max_corr = worst;
        const double parasitism = std::max(0.0, worst - p.parasitism_corr_threshold);

        // 3) 生态位空缺: 池亏损体制内该成员也无正贡献的占比
        uint32_t gap_blocks = 0, counted = 0;
        for (uint32_t b = 0; b < rc; ++b) {
            const size_t begin = b * block;
            const size_t end = (b == rc - 1) ? n : (b + 1) * block;
            if (end <= begin) continue;
            ++counted;
            double pool_block = 0.0, member_block = 0.0;
            for (size_t t = begin; t < end; ++t) {
                pool_block += full[t];
                member_block += rs[i][t];
            }
            if (pool_block < 0.0 && member_block <= 0.0) ++gap_blocks;
        }
        mm.regime_gap = counted > 0 ? static_cast<double>(gap_blocks) / counted : 0.0;

        mm.score = mm.loo_contrib
                 - p.parasitism_penalty * parasitism
                 - p.regime_gap_penalty * mm.regime_gap;
    }
    return out;
}

}  // namespace population
}  // namespace kun

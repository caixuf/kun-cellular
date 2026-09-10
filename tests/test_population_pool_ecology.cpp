// ============================================================================
// test_population_pool_ecology.cpp — L1 系统层: 组合级生态评估原语测试
// 不变量: LOO 方向性 / 寄生惩罚方向性 / 体制空缺计算 / 退化输入防御 / 冻结常数注入
// ============================================================================
#include "kun/cellular/population/pool_ecology.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using kun::population::PoolFitnessParams;
using kun::population::PoolMetrics;
using kun::population::evaluate_pool;

namespace {

std::vector<double> const_returns(size_t n, double v) {
    return std::vector<double>(n, v);
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 组合级生态评估原语测试" << std::endl;
    std::cout << "==================================================================" << std::endl;

    PoolFitnessParams p;  // v1 冻结常数 (夏普2.0/卡玛1.0/回撤2.0/寄生阈0.90/体制3)

    // ── 1. LOO 方向性: 拉动组合的核心成员移除后池适应度必然大跌 ──
    {
        const size_t n = 120;
        std::mt19937 rng(11);
        std::normal_distribution<double> noise(0.0, 0.002);
        std::vector<std::vector<double>> rs;
        // 成员 0: 稳定正 alpha; 成员 1/2: 纯噪声
        rs.push_back(const_returns(n, 0.004));
        rs.emplace_back();
        rs.emplace_back();
        for (size_t t = 0; t < n; ++t) {
            rs[1].push_back(noise(rng));
            rs[2].push_back(noise(rng));
        }
        auto m = evaluate_pool(rs, p);
        assert(m.valid);
        assert(m.members[0].loo_contrib > m.members[1].loo_contrib);
        assert(m.members[0].loo_contrib > m.members[2].loo_contrib);
        assert(m.sharpe > 0.0);
        std::cout << "  ✓ LOO 方向性: 核心成员贡献 (" << m.members[0].loo_contrib << ") 显著高于噪声成员" << std::endl;
    }

    // ── 2. 寄生惩罚方向性: 完全相同的两个成员 → corr=1 → 惩罚生效 ──
    {
        const size_t n = 100;
        std::vector<double> sig;
        std::mt19937 rng(22);
        std::normal_distribution<double> noise(0.0, 0.001);
        for (size_t t = 0; t < n; ++t) sig.push_back(0.001 + noise(rng));
        std::vector<std::vector<double>> rs{sig, sig, const_returns(n, 0.003)};  // 前两个互为副本
        auto m = evaluate_pool(rs, p);
        assert(m.valid);
        assert(m.members[0].max_corr > 0.999);
        const double expected_parasitism = m.members[0].max_corr - p.parasitism_corr_threshold;
        // score = loo − parasitism_penalty×parasitism − regime_gap 惩罚
        assert(m.members[0].score < m.members[0].loo_contrib);
        // 对照: 互不相关成员不触发寄生惩罚
        assert(m.members[2].max_corr < p.parasitism_corr_threshold
               || std::fabs(m.members[2].max_corr - p.parasitism_corr_threshold) < 1e-9);
        std::cout << "  ✓ 寄生惩罚: 副本成员 corr=" << m.members[0].max_corr
                  << " > 阈 " << p.parasitism_corr_threshold << ", score 被扣减" << std::endl;
    }

    // ── 3. 体制空缺: 池全程亏损 + 成员全程亏损 → gap=1.0; 全程盈利成员 gap=0 ──
    {
        const size_t n = 60;
        std::vector<std::vector<double>> rs{
            const_returns(n, 0.011), const_returns(n, 0.011), const_returns(n, -0.03)};
        auto m = evaluate_pool(rs, p);  // 池 = (0.011+0.011-0.03)/3 < 0 每日
        assert(m.valid);
        assert(m.members[2].regime_gap == 1.0);   // 池亏 & 成员亏 (全部体制)
        assert(m.members[0].regime_gap == 0.0);   // 成员恒正 → 无空缺
        std::cout << "  ✓ 体制空缺: 亏损跟随者 gap=1.0, 正贡献者 gap=0.0" << std::endl;
    }

    // ── 4. 退化输入防御 ──
    {
        PoolFitnessParams q = p;
        q.min_eval_days = 20.0;
        auto too_short = evaluate_pool({const_returns(10, 0.01)}, q);
        assert(!too_short.valid);                              // 样本不足
        auto mismatched = evaluate_pool({const_returns(50, 0.01), const_returns(40, 0.01)}, q);
        assert(!mismatched.valid);                             // 长度不一致
        auto single = evaluate_pool({const_returns(50, 0.01)}, q);
        assert(single.valid && single.members[0].loo_contrib > 0.0);  // 单成员池
        std::cout << "  ✓ 退化防御: 样本不足/长度不匹配判无效, 单成员池诚实计算" << std::endl;
    }

    // ── 5. 冻结常数注入生效 (权重改变 → 适应度改变; L1 不携带隐藏策略) ──
    {
        const size_t n = 80;
        std::mt19937 rng(77);
        std::normal_distribution<double> noise(0.0, 0.003);
        std::vector<double> sig;
        for (size_t t = 0; t < n; ++t) sig.push_back(0.002 + noise(rng));  // std>0 → 夏普项有效
        std::vector<std::vector<double>> rs{sig};
        PoolFitnessParams a = p, b = p;
        b.sharpe_weight = 5.0;
        const auto ma = evaluate_pool(rs, a);
        const auto mb = evaluate_pool(rs, b);
        assert(ma.sharpe != 0.0);
        assert(ma.fitness != mb.fitness);
        std::cout << "  ✓ 常数注入: 权重由任务层冻结注入, L1 零隐藏策略" << std::endl;
    }

    std::cout << "[PASS] test_population_pool_ecology all assertions passed!" << std::endl;
    return 0;
}

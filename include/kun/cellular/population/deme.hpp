#pragma once
// ============================================================================
// population/deme.hpp — L1 系统层: 岛屿deme (岛内代际演化容器)
// 概念吸收自底座 IslandEvolutionGrid::IslandDeme, 泛型化于任意个体类型。
// ============================================================================
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kun/cellular/population/individual_traits.hpp"

namespace kun {
namespace population {

// ── 代际演化参数 (任务层注入并冻结; L1 不携带任何业务常数) ────────────────────
struct EvolutionParams {
    float  mut_rate       = 0.12f;   // 变异触发率
    float  mut_sigma      = 0.25f;   // 变异强度
    size_t elite_keep     = 3;       // 精英保留数
    size_t tournament_k   = 3;       // 锦标赛选择窗口
    float  crossover_prob = 0.5f;    // 有性重组概率 (否则克隆父 A 后变异)
    float  migration_rate = 0.0625f; // 每代每岛迁移概率 (1/16)
};

template <typename Individual>
class Deme {
public:
    Deme(uint32_t deme_id, size_t pop_size, std::mt19937::result_type seed)
        : deme_id_(deme_id), rng_(seed) {
        individuals_.reserve(pop_size);
    }

    uint32_t deme_id() const { return deme_id_; }
    size_t size() const { return individuals_.size(); }
    size_t fitness_size() const { return fitness_.size(); }

    std::vector<Individual>& individuals() { return individuals_; }
    const std::vector<Individual>& individuals() const { return individuals_; }
    const std::vector<double>& fitness() const { return fitness_; }
    std::mt19937& rng() { return rng_; }

    // 种群初始化 (工厂由任务层注入; L1 不关心个体如何诞生)
    template <typename Factory>
    void seed_population(size_t pop_size, Factory&& factory) {
        individuals_.clear();
        fitness_.assign(pop_size, -1e18);   // 全部待评估
        for (size_t i = 0; i < pop_size; ++i) {
            individuals_.push_back(factory(rng_));
        }
    }

    // 适应度评估 (评估委托由任务层实现; 逐个体调用)
    // OpenMP 并行: 评估委托必须线程安全 (任务拷贝共享不可变预计算, 状态全局部)
    // fitness 约定: -1e18 = 待评估哨兵; 已有真实适应度的成员 (精英携带) 跳过重算
    template <typename Eval>
    void evaluate(Eval&& eval) {
        if (fitness_.size() != individuals_.size()) fitness_.assign(individuals_.size(), -1e18);
        #pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(individuals_.size()); ++i) {
            if (fitness_[i] > -1e17) continue;             // 精英 fitness 携带: 零重复评估
            double f = eval(individuals_[i]);
            fitness_[i] = std::isfinite(f) ? f : -1e18;
        }
        rebuild_ranking();
    }

    // 岛内代际推进: 精英保留 + 锦标赛亲本 + (杂交|克隆) + 变异, 种群规模守恒
    void evolve_generation(const EvolutionParams& params) {
        const size_t pop = individuals_.size();
        if (pop == 0 || fitness_.size() != pop) return;
        const size_t elite_n = std::min(params.elite_keep, pop);

        std::vector<Individual> next;
        next.reserve(pop);
        for (size_t i = 0; i < elite_n; ++i) {
            next.push_back(individuals_[ranking_[i]]);  // 精英原样保留
        }
        while (next.size() < pop) {
            const Individual& pa = tournament(params.tournament_k);
            const Individual& pb = tournament(params.tournament_k);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            Individual child = (u(rng_) < params.crossover_prob)
                ? CrossoverTrait<Individual>::cross(pa, pb, rng_)
                : pa;                                   // 克隆父 A (单亲)
            child.mutate(params.mut_rate, params.mut_sigma, rng_);
            next.push_back(std::move(child));
        }
        individuals_ = std::move(next);
        // 精英 fitness 携带: 精英个体未变, 保留其已知适应度 (跳过重算);
        // 后代全部标记待评估
        std::vector<double> carried(pop, -1e18);
        for (size_t i = 0; i < elite_n; ++i) {
            carried[i] = fitness_.empty() || fitness_.size() != pop ? -1e18 : fitness_[ranking_[i]];
        }
        fitness_ = std::move(carried);
        ranking_.clear();
    }

    // 锦标赛选择: 从 fitness 排名前 k 中取最优
    const Individual& tournament(size_t k) {
        if (ranking_.empty()) return individuals_.front();
        k = std::max<size_t>(k, 1);
        std::uniform_int_distribution<size_t> d(0, ranking_.size() - 1);
        size_t best = ranking_[d(rng_)];
        for (size_t t = 1; t < k; ++t) {
            size_t cand = ranking_[d(rng_)];
            if (fitness_[cand] > fitness_[best]) best = cand;
        }
        return individuals_[best];
    }

    // 非精英随机槽位 (迁移落点; 不会覆盖精英)
    size_t random_non_elite_slot(const EvolutionParams& params) {
        const size_t pop = individuals_.size();
        const size_t elite_n = std::min(params.elite_keep, pop);
        if (pop <= elite_n) return pop;  // 全员精英: 无非精英槽 (返回越界哨兵)
        std::uniform_int_distribution<size_t> d(elite_n, pop - 1);
        return d(rng_);
    }

    void install_at(size_t slot, const Individual& org) {
        if (slot >= individuals_.size()) return;
        individuals_[slot] = org;
        if (slot < fitness_.size()) fitness_[slot] = -1e18;  // 待重评估
    }

    // 生态位画像槽 (池评估阶段由任务层填充; deme 层只做透传存储)
    double best_fitness() const {
        double best = -1e18;
        for (double f : fitness_) best = std::max(best, f);
        return best;
    }

private:
    void rebuild_ranking() {
        ranking_.resize(fitness_.size());
        std::iota(ranking_.begin(), ranking_.end(), 0);
        std::sort(ranking_.begin(), ranking_.end(),
                  [this](size_t x, size_t y) { return fitness_[x] > fitness_[y]; });
    }

    uint32_t deme_id_;
    std::mt19937 rng_;
    std::vector<Individual> individuals_;
    std::vector<double> fitness_;
    std::vector<size_t> ranking_;   // 适应度降序索引
};

}  // namespace population
}  // namespace kun

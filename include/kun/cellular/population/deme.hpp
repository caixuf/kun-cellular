#pragma once
// ============================================================================
// population/deme.hpp — L1 系统层: 岛屿deme (岛内代际演化容器)
// 概念吸收自底座 IslandEvolutionGrid::IslandDeme, 泛型化于任意个体类型。
//
// 系统层纪律 (docs/population_ecology_v1_design.md):
//   - 仅消费 L0 底座公开 API, 零修改 L0
//   - 领域无关: 无任何业务名词
//   - 确定性: 全部随机源显式传入 (std::mt19937), 同种子位级可复现
//
// 并行归属: 本类的 evaluate 为串行 (单 deme 视角)。跨 deme 的并行调度由
// EcologyGrid 统一持有 (系统层单一调度点), 避免嵌套并行区过订。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kun/cellular/population/individual_traits.hpp"
#include "kun/cellular/population/population_eval.hpp"

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

    // 适应度降序排名 (仅前 keep_n 有序; 全范围仍是索引的一个排列)
    const std::vector<size_t>& ranking() const { return ranking_; }

    // 种群初始化 (工厂由任务层注入; L1 不关心个体如何诞生)
    template <typename Factory>
    void seed_population(size_t pop_size, Factory&& factory) {
        individuals_.clear();
        fitness_.assign(pop_size, -1e18);   // 全部待评估
        for (size_t i = 0; i < pop_size; ++i) {
            individuals_.push_back(factory(rng_));
        }
        ranking_valid_ = false;
    }

    // ── 待评估枚举 (系统层调度用: 网格收集全网格 pending, 统一并行) ─────────────
    // 约定: fitness <= -1e17 为待评估哨兵; 已有真实适应度的成员 (精英携带) 跳过。
    template <typename Fn>
    void for_each_pending(Fn&& fn) {
        if (fitness_.size() != individuals_.size()) fitness_.assign(individuals_.size(), -1e18);
        for (size_t i = 0; i < individuals_.size(); ++i) {
            if (fitness_[i] <= -1e17) fn(i, individuals_[i]);
        }
    }

    // 回写评估结果 (系统层调度用); 非有限值统一钳为待评估哨兵
    void set_fitness(size_t slot, double f) {
        if (slot >= fitness_.size()) return;
        fitness_[slot] = std::isfinite(f) ? f : -1e18;
        ranking_valid_ = false;
    }

    // 单 deme 串行评估便利入口 (跨 deme 并行由 EcologyGrid 持有; 见文件头)
    template <typename Eval>
    void evaluate(Eval&& eval) {
        for_each_pending([&](size_t slot, Individual& ind) {
            set_fitness(slot, static_cast<double>(eval(ind)));
        });
    }

    // 岛内代际推进: 精英保留 + 锦标赛亲本 + (杂交|克隆) + 变异, 种群规模守恒
    // 缓冲常驻复用: offspring_ 保持 pop 个活槽, 逐代按槽位原地写 (copy-assign 复用内部
    // vector 容量) → 稳态零堆分配。无默认构造的个体用 individuals_[0] 拷贝构造补齐。
    // rng 抽取序列与旧 push_back 版逐位一致 (抽取次数/顺序不变)。
    void evolve_generation(const EvolutionParams& params) {
        const size_t pop = individuals_.size();
        if (pop == 0 || fitness_.size() != pop) return;
        const size_t elite_n = std::min(params.elite_keep, pop);
        ensure_topk(elite_n);

        ensure_offspring_slots(pop);
        for (size_t i = 0; i < elite_n; ++i) {
            offspring_[i] = individuals_[ranking_[i]];  // 精英原地拷贝 (复用容量)
        }
        for (size_t slot = elite_n; slot < pop; ++slot) {
            const Individual& pa = tournament(params.tournament_k);
            const Individual& pb = tournament(params.tournament_k);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            if (u(rng_) < params.crossover_prob) {
                crossover_into(pa, pb, offspring_[slot], rng_);  // 原地杂交 (免分配)
            } else {
                offspring_[slot] = pa;                           // 克隆父 A (原地)
            }
            offspring_[slot].mutate(params.mut_rate, params.mut_sigma, rng_);
        }
        // 精英 fitness 携带 (跳过重算); 后代全部标记待评估
        fitness_next_.assign(pop, -1e18);
        for (size_t i = 0; i < elite_n; ++i) {
            fitness_next_[i] = fitness_[ranking_[i]];
        }
        std::swap(individuals_, offspring_);
        std::swap(fitness_, fitness_next_);
        ranking_valid_ = false;
        ranking_.clear();
    }

    // 锦标赛选择: 均匀抽 k 个 (确定性: 固定 rng 抽取序列), 取适应度最优
    // 与 ranking 顺序解耦 (partial_sort 尾部无序不影响选择语义)
    const Individual& tournament(size_t k) {
        const size_t pop = individuals_.size();
        if (pop == 0) return individuals_.front();
        k = std::max<size_t>(k, 1);
        std::uniform_int_distribution<size_t> d(0, pop - 1);
        size_t best = d(rng_);
        for (size_t t = 1; t < k; ++t) {
            size_t cand = d(rng_);
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
        ranking_valid_ = false;
    }

    double best_fitness() const {
        double best = -1e18;
        for (double f : fitness_) best = std::max(best, f);
        return best;
    }

    // 重算前 keep_n 的有序排名 (partial_sort, O(n + k·log k));
    // 全序比较器 (fitness 降序, index 升序) 保证并列时可移植确定性。
    void rebuild_topk(size_t keep_n) {
        const size_t pop = individuals_.size();
        ranking_.resize(pop);
        if (pop == 0) { ranking_valid_ = true; ranking_k_ = 0; return; }
        std::iota(ranking_.begin(), ranking_.end(), size_t{0});
        const size_t k = std::min(keep_n, pop);
        auto cmp = [this](size_t x, size_t y) {
            if (fitness_[x] != fitness_[y]) return fitness_[x] > fitness_[y];
            return x < y;  // 确定性并列打破
        };
        std::partial_sort(ranking_.begin(), ranking_.begin() + static_cast<ptrdiff_t>(k),
                          ranking_.end(), cmp);
        ranking_valid_ = true;
        ranking_k_ = k;
    }

private:
    // 保证 offspring_ 恰有 pop 个活槽; 个体无需默认构造 (用 individuals_[0] 拷贝构造补齐)。
    // 稳态: 槽数不变 → 逐代仅 copy-assign, 复用内部 buffer。前置条件: pop>0。
    // pop 收缩时 resize 销毁多余槽 (正确, 但会丢缓冲; 演化中 pop 通常固定)。
    void ensure_offspring_slots(size_t pop) {
        if (offspring_.size() == pop) return;
        if (offspring_.size() > pop) {
            // 收缩用 pop_back (resize 需默认构造, 个体可能不可默认构造)
            while (offspring_.size() > pop) offspring_.pop_back();
            return;
        }
        const Individual& proto = individuals_[0];
        while (offspring_.size() < pop) offspring_.push_back(proto);
    }

    void ensure_topk(size_t keep_n) {
        const size_t pop = individuals_.size();
        const size_t k = std::min(keep_n, pop);
        if (ranking_valid_ && ranking_.size() == pop && ranking_k_ >= k) return;
        rebuild_topk(keep_n);
    }

    uint32_t deme_id_;
    std::mt19937 rng_;
    std::vector<Individual> individuals_;
    std::vector<double> fitness_;
    std::vector<size_t> ranking_;   // 适应度降序索引 (前 ranking_k_ 有序)
    bool ranking_valid_{false};
    size_t ranking_k_{0};
    // 双缓冲 (容量复用)
    std::vector<Individual> offspring_;
    std::vector<double> fitness_next_;
};

}  // namespace population
}  // namespace kun

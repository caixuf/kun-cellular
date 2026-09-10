#pragma once
// ============================================================================
// population/ecology_grid.hpp — L1 系统层: 多岛生态网格 (迁移 + 杂交 + 代际)
// 概念吸收自底座 IslandEvolutionGrid (岛屿/迁移/红皇后), 泛型化于任意个体类型。
// 不变量: deme 规模守恒; 精英永不被迁移覆盖; 同种子位级可复现。
//
// 调度归属 (系统层单一调度点):
//   - 跨 deme 的待评估个体被摊平成单一列表, 在一个并行区内统一评估
//     (避免每 deme 每代重复 fork/join 的并行区过订)。
//   - 逐个体委托 (IndividualEvaluator): L1 跨 pending 并行。
//   - 批量委托 (BatchEvaluator): L1 交出整网格 pending 批, 任务自管批内并行。
//   - migrate 严格串行 (固定顺序 + 固定 rng 抽取序列), 保证确定性。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "kun/cellular/population/deme.hpp"

namespace kun {
namespace population {

// ── 执行配置 (与语义参数 EvolutionParams 分离) ───────────────────────────────
struct ExecutionConfig {
    bool parallel = true;  // individual 路径是否跨 pending 并行 (batch 路径由任务自管)
};

template <typename Individual>
class EcologyGrid {
public:
    EcologyGrid(size_t num_demes, size_t pop_per_deme, std::mt19937::result_type seed,
                ExecutionConfig cfg = {})
        : grid_rng_(seed), cfg_(cfg) {
        for (size_t i = 0; i < num_demes; ++i) {
            demes_.emplace_back(static_cast<uint32_t>(i), pop_per_deme,
                                static_cast<std::mt19937::result_type>(seed + 7919 * (i + 1)));
        }
    }

    size_t num_demes() const { return demes_.size(); }
    size_t pop_per_deme() const { return demes_.empty() ? 0 : demes_[0].size(); }
    std::vector<Deme<Individual>>& demes() { return demes_; }
    const std::vector<Deme<Individual>>& demes() const { return demes_; }

    uint64_t total_migrations() const { return total_migrations_; }

    template <typename Factory>
    void seed_all(Factory&& factory) {
        for (auto& d : demes_) d.seed_population(demes_[0].size() ? d.size() : 0, factory);
    }

    // 单代推进:
    //   1) 评估所有待评估 (gen0=全体; 之后通常为空, 快路径直接返回)
    //   2) 岛内演化 (含精英 fitness 携带)
    //   3) 跨岛迁移 (迁移个体本代即被评估)
    //   4) 评估 offspring ∪ migrants (单次遍历)
    template <typename Eval>
    void step_generation(Eval&& eval, const EvolutionParams& params) {
        evaluate_pending(eval);
        for (auto& d : demes_) d.evolve_generation(params);
        migrate(params);
        evaluate_pending(eval);
    }

    // 迁移 (严格串行): 精英基因流, 精英槽不覆盖; 固定 rng 抽取序列
    void migrate(const EvolutionParams& params) {
        if (demes_.size() < 2) return;
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t di = 0; di < demes_.size(); ++di) {
            if (u(grid_rng_) >= params.migration_rate) continue;
            size_t src = static_cast<size_t>(grid_rng_() % demes_.size());
            if (src == di) src = (src + 1) % demes_.size();
            Deme<Individual>& src_deme = demes_[src];
            if (src_deme.size() == 0 || src_deme.fitness_size() != src_deme.size()) continue;

            size_t best_idx = 0;
            double best_f = -1e18;
            for (size_t m = 0; m < src_deme.size(); ++m) {
                if (src_deme.fitness()[m] > best_f) {
                    best_f = src_deme.fitness()[m];
                    best_idx = m;
                }
            }
            size_t slot = demes_[di].random_non_elite_slot(params);
            if (slot >= demes_[di].size()) continue;
            demes_[di].install_at(slot, src_deme.individuals()[best_idx]);
            ++total_migrations_;
        }
    }

private:
    // 全网格扁平待评估调度: 收集 → (批量|并行)评估 → 定序回写
    template <typename Eval>
    void evaluate_pending(Eval&& eval) {
        using EvalT = std::remove_cvref_t<Eval>;
        flat_deme_.clear();
        flat_slot_.clear();
        flat_ptr_.clear();
        for (size_t di = 0; di < demes_.size(); ++di) {
            demes_[di].for_each_pending([&](size_t slot, Individual& ind) {
                flat_deme_.push_back(di);
                flat_slot_.push_back(slot);
                flat_ptr_.push_back(&ind);
            });
        }
        if (flat_ptr_.empty()) return;  // 快路径: 无待评估

        if constexpr (BatchEvaluator<EvalT, Individual>) {
            flat_fitness_.assign(flat_ptr_.size(), 0.0);
            ++batch_seq_;
            eval(BatchView<Individual>{flat_ptr_, flat_fitness_, batch_seq_});
        } else {
            flat_fitness_.resize(flat_ptr_.size());
            const bool par = cfg_.parallel;
            #pragma omp parallel for schedule(dynamic) if(par)
            for (int64_t j = 0; j < static_cast<int64_t>(flat_ptr_.size()); ++j) {
                flat_fitness_[static_cast<size_t>(j)] =
                    static_cast<double>(eval(*flat_ptr_[static_cast<size_t>(j)]));
            }
        }
        // 定序回写 (与并行调度无关 ⇒ 确定性)
        for (size_t j = 0; j < flat_ptr_.size(); ++j) {
            demes_[flat_deme_[j]].set_fitness(flat_slot_[j], flat_fitness_[j]);
        }
    }

    std::mt19937 grid_rng_;
    std::vector<Deme<Individual>> demes_;
    uint64_t total_migrations_{0};
    ExecutionConfig cfg_;
    // 扁平调度 scratch (容量复用)
    std::vector<size_t> flat_deme_;
    std::vector<size_t> flat_slot_;
    std::vector<Individual*> flat_ptr_;
    std::vector<double> flat_fitness_;
    uint64_t batch_seq_{0};
};

}  // namespace population
}  // namespace kun

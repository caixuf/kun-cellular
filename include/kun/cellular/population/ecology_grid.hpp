#pragma once
// ============================================================================
// population/ecology_grid.hpp — L1 系统层: 多岛生态网格 (迁移 + 杂交 + 代际)
// 概念吸收自底座 IslandEvolutionGrid (岛屿/迁移/红皇后), 泛型化于任意个体类型。
// 不变量: deme 规模守恒; 精英永不被迁移覆盖; 同种子位级可复现。
// ============================================================================
#include <cstdint>
#include <random>
#include <vector>

#include "kun/cellular/population/deme.hpp"

namespace kun {
namespace population {

template <typename Individual>
class EcologyGrid {
public:
    EcologyGrid(size_t num_demes, size_t pop_per_deme, std::mt19937::result_type seed)
        : grid_rng_(seed) {
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

    // 单代推进: 岛内评估与演化 → 跨岛迁移 (精英基因流, 精英槽不覆盖)
    template <typename Eval>
    void step_generation(Eval&& eval, const EvolutionParams& params) {
        for (auto& d : demes_) {
            d.evaluate(eval);
            d.evolve_generation(params);
            d.evaluate(eval);  // 子代立即评估 (父代精英在 evolve 内已保留)
        }
        migrate(params);
    }

    void migrate(const EvolutionParams& params) {
        if (demes_.size() < 2) return;
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t di = 0; di < demes_.size(); ++di) {
            if (u(grid_rng_) >= params.migration_rate) continue;
            // 源岛: 随机选另一岛, 取其随机精英
            size_t src = static_cast<size_t>(grid_rng_() % demes_.size());
            if (src == di) src = (src + 1) % demes_.size();
            Deme<Individual>& src_deme = demes_[src];
            if (src_deme.size() == 0 || src_deme.fitness_size() != src_deme.size()) continue;

            // 源岛精英 = 适应度最高且未被本代迁移污染的个体
            size_t best_idx = 0;
            double best_f = -1e18;
            for (size_t m = 0; m < src_deme.size(); ++m) {
                if (src_deme.fitness()[m] > best_f) {
                    best_f = src_deme.fitness()[m];
                    best_idx = m;
                }
            }
            // 目标岛: 只落在非精英槽 (精英永不被迁移覆盖)
            size_t slot = demes_[di].random_non_elite_slot(params);
            if (slot >= demes_[di].size()) continue;
            demes_[di].install_at(slot, src_deme.individuals()[best_idx]);
            ++total_migrations_;
        }
    }

private:
    std::mt19937 grid_rng_;
    std::vector<Deme<Individual>> demes_;
    uint64_t total_migrations_{0};
};

}  // namespace population
}  // namespace kun

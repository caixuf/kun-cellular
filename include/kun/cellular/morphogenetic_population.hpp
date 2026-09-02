#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <map>
#include <set>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/autonomous_replicator.hpp"

namespace kun {

// ============================================================================
// 1. 纯粹底层物种兼容性度量与两性基因重组 (Pure Genomic Speciation & Crossover)
// ============================================================================
struct SpeciationMetric {
    // 基于基因位点拓扑差分与权重差异的相容性距离 (NEAT 距离公理)
    static double compatibility_distance(const ReplicableGenome& g1, const ReplicableGenome& g2,
                                         double c1 = 1.0, double c2 = 0.4) {
        size_t n1 = g1.loci.size();
        size_t n2 = g2.loci.size();
        size_t min_n = std::min(n1, n2);
        size_t max_n = std::max(n1, n2);

        if (max_n == 0) return 0.0;

        double weight_diff_sum = 0.0;
        size_t matching_count = 0;

        for (size_t i = 0; i < min_n; ++i) {
            if (g1.loci[i].op_type == g2.loci[i].op_type) {
                weight_diff_sum += std::abs(g1.loci[i].weight_param - g2.loci[i].weight_param);
                matching_count++;
            }
        }

        size_t disjoint_count = max_n - matching_count;
        double avg_weight_diff = (matching_count > 0) ? (weight_diff_sum / matching_count) : 1.0;

        return (c1 * disjoint_count / double(max_n)) + (c2 * avg_weight_diff);
    }

    // 两性基因交叉重组 (Sexual Crossover)：模块化交换基因位点
    static ReplicableGenome sexual_crossover(const ReplicableGenome& mom, const ReplicableGenome& dad,
                                            std::mt19937& rng) {
        ReplicableGenome child;
        child.genome_id = rng();

        size_t min_len = std::min(mom.loci.size(), dad.loci.size());
        size_t crossover_point = rng() % (std::max(size_t(1), min_len));

        for (size_t i = 0; i < crossover_point && i < mom.loci.size(); ++i) {
            child.loci.push_back(mom.loci[i]);
        }
        for (size_t i = crossover_point; i < dad.loci.size(); ++i) {
            child.loci.push_back(dad.loci[i]);
        }

        child.base_replication_cost = (mom.base_replication_cost + dad.base_replication_cost) * 0.5;
        child.replication_threshold = (mom.replication_threshold + dad.replication_threshold) * 0.5;

        std::stringstream ss;
        ss << mom.lineage_hash.substr(0, std::min(size_t(4), mom.lineage_hash.size())) 
           << dad.lineage_hash.substr(0, std::min(size_t(4), dad.lineage_hash.size()));
        child.lineage_hash = ss.str();

        return child;
    }
};

// ============================================================================
// 2. 纯物理演化个体 (Pure Cellular Individual)
// ============================================================================
struct PureCellularIndividual {
    uint64_t id{0};
    uint32_t species_id{0};
    ReplicableGenome genome;
    CellularOrganism organism;
    double fitness{0.0};
    double adjusted_fitness{0.0};
    double energy_reserve{100.0};
    uint32_t age_generations{0};
    bool is_viable{false};

    // 从纯粹基因位点直接映射为物理元胞与突触网络（无人工发育阶段划分）
    bool express_phenotype() {
        organism = CellularOrganism();
        organism.cells.clear();
        organism.synapses.clear();

        if (genome.loci.empty()) {
            is_viable = false;
            return false;
        }

        uint32_t cid = 0;
        // 1. 基础感觉受体
        organism.cells.push_back({cid++, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -40.0f, 0.0f});
        organism.cells.push_back({cid++, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, 40.0f, 0.0f});

        static const CellType kOpTable[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS,
            CellType::OP_DELAY_N, CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_INHIBIT,
            CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX
        };

        // 2. 从基因位点直接表达中间代谢与门控细胞 (24类原语算子)
        for (size_t i = 0; i < genome.loci.size(); ++i) {
            const auto& loc = genome.loci[i];
            CellType ctype = kOpTable[loc.op_type % (sizeof(kOpTable) / sizeof(kOpTable[0]))];
            float px = -60.0f + static_cast<float>(i * 18.0);
            float py = (i % 2 == 0 ? 30.0f : -30.0f) + static_cast<float>(loc.weight_param * 10.0);
            organism.cells.push_back({cid++, ctype, loc.weight_param, loc.base_metabolic_rate, 0.0, 0.0, false, 0.0, 0, 0, px, py, 0.0f});
        }

        // 3. 动作效应器
        uint32_t act_pos_id = cid++;
        uint32_t act_neg_id = cid++;
        organism.cells.push_back({act_pos_id, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -40.0f, 0.0f});
        organism.cells.push_back({act_neg_id, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, 40.0f, 0.0f});

        // 4. 构建因果轴突前向连接
        organism.synapses.push_back({0, 2, 0, 1.0, true, 60.0f, -1.0f});
        organism.synapses.push_back({1, 2, 1, 1.0, true, 60.0f, -1.0f});

        for (uint32_t idx = 2; idx + 1 < act_pos_id; ++idx) {
            organism.synapses.push_back({idx, idx + 1, 0, genome.loci[(idx - 2) % genome.loci.size()].weight_param, true, 50.0f, -1.0f});
        }

        organism.synapses.push_back({act_pos_id - 1, act_pos_id, 0, 1.0, true, 60.0f, -1.0f});
        organism.synapses.push_back({act_pos_id - 1, act_neg_id, 0, -1.0, true, 60.0f, -1.0f});

        organism.compile();
        is_viable = (organism.cells.size() >= 4 && !organism.synapses.empty());
        return is_viable;
    }
};

// ============================================================================
// 3. 物种生态利基 (Species Ecological Niche)
// ============================================================================
struct SpeciesNiche {
    uint32_t species_id{0};
    ReplicableGenome representative_genome;
    std::vector<PureCellularIndividual> members;
    double max_fitness{0.0};
    uint32_t stagnant_generations{0};
    bool is_extinct{false};

    void calculate_shared_fitness() {
        if (members.empty()) return;
        double sz = static_cast<double>(members.size());
        for (auto& m : members) {
            m.adjusted_fitness = m.fitness / sz; // 显式适应度共享，保护物种多样性
        }
    }
};

// ============================================================================
// 4. 纯粹演化种群生态系统 (Pure Evolutionary Population Ecosystem)
// ============================================================================
class MorphogeneticPopulationEcosystem {
public:
    uint32_t generation{0};
    size_t target_population_size{40};
    double speciation_threshold{0.35};
    uint32_t next_species_id{1};
    uint64_t next_ind_id{1};

    std::vector<SpeciesNiche> species;
    std::mt19937 rng;

    explicit MorphogeneticPopulationEcosystem(size_t pop_size = 40, uint32_t seed = 42)
        : target_population_size(pop_size), rng(seed) {
        initialize_founder_population();
    }

    void initialize_founder_population() {
        species.clear();
        generation = 0;

        SpeciesNiche founder_species;
        founder_species.species_id = next_species_id++;

        for (size_t i = 0; i < target_population_size; ++i) {
            ReplicableGenome g = make_seed_genome();
            PureCellularIndividual ind;
            ind.id = next_ind_id++;
            ind.species_id = founder_species.species_id;
            ind.genome = g;
            ind.express_phenotype();
            founder_species.members.push_back(ind);
        }

        founder_species.representative_genome = founder_species.members[0].genome;
        species.push_back(founder_species);
    }

    // 运行一代演化周期 (Step Evolution Cycle)
    void step_evolution_cycle() {
        generation++;

        // 1. 评估个体在物理环境中的前向推演与适应度
        for (auto& sp : species) {
            for (auto& ind : sp.members) {
                if (ind.is_viable) {
                    double inputs[4] = {3620.0, 4500.0, 1.2, 0.4};
                    auto acts = ind.organism.forward(inputs);
                    double action_magnitude = std::abs(acts.positive_action) + std::abs(acts.negative_action);
                    ind.fitness = std::max(1.0, action_magnitude * 10.0 + ind.organism.cells.size() * 2.0);
                } else {
                    ind.fitness = 0.1;
                }
            }
            sp.calculate_shared_fitness();
        }

        // 2. 收集精英并基于物种内基因重组繁衍后代
        std::vector<PureCellularIndividual> offspring_pool;

        for (auto& sp : species) {
            if (sp.members.empty()) continue;

            // 排序选拔精英
            std::sort(sp.members.begin(), sp.members.end(), [](const auto& a, const auto& b) {
                return a.adjusted_fitness > b.adjusted_fitness;
            });

            // 精英保留
            offspring_pool.push_back(sp.members[0]);

            // 两性基因交叉重组与点突变
            size_t offspring_count = std::max(size_t(1), sp.members.size());
            for (size_t k = 1; k < offspring_count; ++k) {
                const auto& parent_a = sp.members[rng() % sp.members.size()];
                const auto& parent_b = sp.members[rng() % sp.members.size()];

                ReplicableGenome child_genome = SpeciationMetric::sexual_crossover(parent_a.genome, parent_b.genome, rng);
                child_genome = child_genome.replicate_with_mutation(rng, 0.08);

                PureCellularIndividual child;
                child.id = next_ind_id++;
                child.genome = child_genome;
                child.express_phenotype();
                offspring_pool.push_back(child);
            }
        }

        // 3. 物种动态重新聚类划分
        for (auto& sp : species) {
            sp.members.clear();
        }

        for (auto& ind : offspring_pool) {
            bool placed = false;
            for (auto& sp : species) {
                double dist = SpeciationMetric::compatibility_distance(ind.genome, sp.representative_genome);
                if (dist < speciation_threshold) {
                    ind.species_id = sp.species_id;
                    sp.members.push_back(ind);
                    placed = true;
                    break;
                }
            }

            if (!placed) {
                SpeciesNiche new_species;
                new_species.species_id = next_species_id++;
                new_species.representative_genome = ind.genome;
                ind.species_id = new_species.species_id;
                new_species.members.push_back(ind);
                species.push_back(new_species);
            }
        }

        // 4. 清理空利基
        species.erase(std::remove_if(species.begin(), species.end(), [](const SpeciesNiche& sp) {
            return sp.members.empty();
        }), species.end());

        // 补足种群规模恒定
        size_t current_pop = get_total_population();
        if (current_pop < target_population_size && !species.empty()) {
            size_t deficit = target_population_size - current_pop;
            for (size_t d = 0; d < deficit; ++d) {
                auto& sp = species[rng() % species.size()];
                ReplicableGenome g = sp.representative_genome.replicate_with_mutation(rng, 0.1);
                PureCellularIndividual ind;
                ind.id = next_ind_id++;
                ind.species_id = sp.species_id;
                ind.genome = g;
                ind.express_phenotype();
                sp.members.push_back(ind);
            }
        }
    }

    size_t get_total_population() const {
        size_t sum = 0;
        for (const auto& sp : species) {
            sum += sp.members.size();
        }
        return sum;
    }

    size_t get_species_count() const {
        return species.size();
    }

    double get_viable_rate() const {
        size_t viable = 0;
        size_t total = 0;
        for (const auto& sp : species) {
            for (const auto& ind : sp.members) {
                total++;
                if (ind.is_viable) viable++;
            }
        }
        return total > 0 ? (double(viable) / total) : 0.0;
    }

private:
    ReplicableGenome make_seed_genome() {
        ReplicableGenome g;
        g.genome_id = rng();
        g.lineage_hash = "Genesis01";

        for (uint32_t i = 0; i < 6; ++i) {
            GeneLocus loc;
            loc.gene_id = i;
            loc.op_type = static_cast<uint8_t>(4 + (i % 17)); // 24 基础原语类型
            loc.weight_param = 1.0;
            loc.base_metabolic_rate = 0.20;
            g.loci.push_back(loc);
        }
        return g;
    }
};

} // namespace kun

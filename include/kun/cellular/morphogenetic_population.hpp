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
// 1. 发育时钟与形态发生阶段 (Developmental Epoch & Morphogenesis Stages)
// ============================================================================
enum class DevelopmentalStage : uint8_t {
    ZYGOTE = 0,         // 受精卵阶段：单细胞起点与初始母源形态素
    CLEAVAGE = 1,       // 卵裂期：快速有丝分裂，细胞数量指数扩增
    GASTRULATION = 2,   // 原肠期：空间力场迁移与三维极性轴形成
    DIFFERENTIATION = 3,// 分化期：根据形态素浓度梯度确定细胞命运
    MATURE = 4,         // 成熟期：突触轴突定向投射与凋亡剪枝完成
    INVIABLE = 5        // 发育畸变/夭折：能量耗尽或未形成有效回路
};

inline const char* to_string(DevelopmentalStage stage) {
    switch (stage) {
        case DevelopmentalStage::ZYGOTE:         return "ZYGOTE";
        case DevelopmentalStage::CLEAVAGE:       return "CLEAVAGE";
        case DevelopmentalStage::GASTRULATION:   return "GASTRULATION";
        case DevelopmentalStage::DIFFERENTIATION: return "DIFFERENTIATION";
        case DevelopmentalStage::MATURE:         return "MATURE";
        case DevelopmentalStage::INVIABLE:       return "INVIABLE";
        default: return "UNKNOWN";
    }
}

// 细胞命运谱系分类
enum class CellFate : uint8_t {
    UNDIFFERENTIATED = 0, // 未分化干细胞
    RECEPTOR = 1,         // 感觉受体细胞 (输入层)
    METABOLIC = 2,        // 代谢联络神经元 (特征计算层)
    GATING = 3,           // 门控迟滞元胞 (决策控制层)
    EFFECTOR = 4          // 动作效应细胞 (输出执行层)
};

// ============================================================================
// 2. 形态素空间化学梯度场 (Morphogen Gradient Field)
// ============================================================================
struct MorphogenGradient {
    double anterior_posterior{0.5}; // A-P 前后轴极性梯度 [0.0, 1.0] (0:前/输入, 1:后/输出)
    double dorsal_ventral{0.5};    // D-V 背腹轴侧向梯度 [0.0, 1.0] (0:腹侧, 1:背侧)
    double local_density{0.0};     // 局部细胞微环境压强
};

// 胚胎细胞单元
struct EmbryonicCell {
    uint32_t cell_id{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double vx{0.0};
    double vy{0.0};
    double vz{0.0};
    MorphogenGradient morphogen;
    CellFate fate{CellFate::UNDIFFERENTIATED};
    uint8_t op_type{0};
    double param1{0.1};
    double param2{0.0};
    double energy{20.0};
    bool is_alive{true};
};

// 胚胎期突触轴突连接
struct EmbryonicSynapse {
    uint32_t from_id{0};
    uint32_t to_id{0};
    uint32_t port{0};
    double weight{1.0};
    bool is_active{true};
};

// ============================================================================
// 3. 胚胎形态发生引擎 (Embryo Morphogenesis Engine)
// ============================================================================
class EmbryoMorphogenesisEngine {
public:
    DevelopmentalStage stage{DevelopmentalStage::ZYGOTE};
    ReplicableGenome genome;
    std::vector<EmbryonicCell> cells;
    std::vector<EmbryonicSynapse> synapses;
    double developmental_energy{120.0}; // 母源卵黄初始能量 (ATP)
    uint32_t developmental_ticks{0};
    uint32_t next_cell_id{1};

    explicit EmbryoMorphogenesisEngine(const ReplicableGenome& g)
        : genome(g) {
        reset_to_zygote();
    }

    void reset_to_zygote() {
        stage = DevelopmentalStage::ZYGOTE;
        cells.clear();
        synapses.clear();
        next_cell_id = 1;
        developmental_ticks = 0;
        developmental_energy = 120.0;

        // 初始化受精卵原基细胞
        EmbryonicCell zygote;
        zygote.cell_id = 0;
        zygote.x = 0.0;
        zygote.y = 0.0;
        zygote.z = 0.0;
        zygote.morphogen.anterior_posterior = 0.5;
        zygote.morphogen.dorsal_ventral = 0.5;
        zygote.fate = CellFate::UNDIFFERENTIATED;
        zygote.energy = 30.0;
        cells.push_back(zygote);
    }

    // 单步形态发生推演 (Step Morphogenesis)
    bool step(std::mt19937& rng) {
        if (stage == DevelopmentalStage::MATURE || stage == DevelopmentalStage::INVIABLE) {
            return false;
        }

        developmental_ticks++;
        developmental_energy -= 0.5; // 发育基础代谢消耗

        if (developmental_energy <= 0.0) {
            stage = DevelopmentalStage::INVIABLE;
            return false;
        }

        switch (stage) {
            case DevelopmentalStage::ZYGOTE:
                step_zygote_activation(rng);
                break;
            case DevelopmentalStage::CLEAVAGE:
                step_cleavage_mitosis(rng);
                break;
            case DevelopmentalStage::GASTRULATION:
                step_gastrulation_migration();
                break;
            case DevelopmentalStage::DIFFERENTIATION:
                step_differentiation(rng);
                break;
            default:
                break;
        }

        return (stage != DevelopmentalStage::MATURE && stage != DevelopmentalStage::INVIABLE);
    }

    // 全程自动发育至成熟或终止
    DevelopmentalStage develop_to_maturity(std::mt19937& rng, uint32_t max_ticks = 200) {
        while (develop_ticks_remaining(max_ticks) && step(rng)) {
            // 继续发育
        }
        return stage;
    }

private:
    bool develop_ticks_remaining(uint32_t max_ticks) const {
        return developmental_ticks < max_ticks && stage != DevelopmentalStage::MATURE && stage != DevelopmentalStage::INVIABLE;
    }

    void step_zygote_activation(std::mt19937& rng) {
        // 受精卵激活：根据基因位点数量确定目标细胞增殖规模
        stage = DevelopmentalStage::CLEAVAGE;
        std::normal_distribution<double> dist(0.0, 1.0);

        // 第一次卵裂：分裂为前极性细胞与后极性细胞
        EmbryonicCell anterior = cells[0];
        anterior.cell_id = next_cell_id++;
        anterior.x = -10.0 + dist(rng);
        anterior.morphogen.anterior_posterior = 0.2; // 前端

        EmbryonicCell posterior = cells[0];
        posterior.cell_id = next_cell_id++;
        posterior.x = 10.0 + dist(rng);
        posterior.morphogen.anterior_posterior = 0.8; // 后端

        cells.clear();
        cells.push_back(anterior);
        cells.push_back(posterior);
        developmental_energy -= 10.0;
    }

    void step_cleavage_mitosis(std::mt19937& rng) {
        // 卵裂期有丝分裂：增殖至与基因组 locus 容量匹配 (8 ~ 32 细胞)
        size_t target_cell_count = std::clamp(genome.loci.size() + 4, size_t(8), size_t(32));

        if (cells.size() < target_cell_count) {
            std::vector<EmbryonicCell> new_daughters;
            std::normal_distribution<double> pos_noise(0.0, 5.0);

            for (auto& mother : cells) {
                if (cells.size() + new_daughters.size() >= target_cell_count) {
                    break;
                }
                if (mother.energy > 8.0 && developmental_energy > 5.0) {
                    mother.energy *= 0.5;
                    developmental_energy -= 2.0;

                    EmbryonicCell daughter = mother;
                    daughter.cell_id = next_cell_id++;
                    daughter.x += pos_noise(rng);
                    daughter.y += pos_noise(rng);
                    daughter.z += pos_noise(rng);
                    daughter.morphogen.dorsal_ventral = std::clamp(daughter.morphogen.dorsal_ventral + pos_noise(rng) * 0.05, 0.0, 1.0);
                    new_daughters.push_back(daughter);
                }
            }

            for (const auto& d : new_daughters) {
                cells.push_back(d);
            }
        }

        if (cells.size() >= target_cell_count || developmental_ticks >= 20) {
            stage = DevelopmentalStage::GASTRULATION;
        }
    }

    void step_gastrulation_migration() {
        // 原肠期空间力场自组织：兰纳-琼斯力场排斥与前后轴极性牵引
        const double k_repulse = 50.0;
        const double k_polar_pull = 2.0;

        for (size_t i = 0; i < cells.size(); ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;

            // 极性牵引：A-P 轴沿 X 轴拉伸排列 (-80 到 +80)
            double target_x = (cells[i].morphogen.anterior_posterior - 0.5) * 160.0;
            fx += (target_x - cells[i].x) * k_polar_pull;

            // 细胞间排斥力 (防止堆叠)
            for (size_t j = 0; j < cells.size(); ++j) {
                if (i == j) continue;
                double dx = cells[i].x - cells[j].x;
                double dy = cells[i].y - cells[j].y;
                double dz = cells[i].z - cells[j].z;
                double dist_sq = dx*dx + dy*dy + dz*dz + 1.0;
                double dist = std::sqrt(dist_sq);

                if (dist < 30.0) {
                    double rep = k_repulse / (dist_sq + 1e-3);
                    fx += (dx / dist) * rep;
                    fy += (dy / dist) * rep;
                    fz += (dz / dist) * rep;
                }
            }

            // 物理积分更新
            cells[i].vx = (cells[i].vx + fx * 0.02) * 0.75;
            cells[i].vy = (cells[i].vy + fy * 0.02) * 0.75;
            cells[i].vz = (cells[i].vz + fz * 0.02) * 0.75;

            cells[i].x += cells[i].vx;
            cells[i].y += cells[i].vy;
            cells[i].z += cells[i].vz;

            // 更新前后梯度坐标
            cells[i].morphogen.anterior_posterior = std::clamp((cells[i].x + 80.0) / 160.0, 0.0, 1.0);
        }

        if (developmental_ticks >= 40) {
            stage = DevelopmentalStage::DIFFERENTIATION;
        }
    }

    void step_differentiation(std::mt19937& rng) {
        // 分化期：根据前后极性梯度激活基因位点，特化为特定脑区原语
        // A-P 梯度 0.0 ~ 0.25: 感觉受体 (RECEPTOR)
        // A-P 梯度 0.25 ~ 0.70: 代谢/门控计算 (METABOLIC / GATING)
        // A-P 梯度 0.70 ~ 1.00: 动作效应器 (EFFECTOR)
        std::vector<uint32_t> receptors, metabolic_cells, effectors;

        for (size_t i = 0; i < cells.size(); ++i) {
            double ap = cells[i].morphogen.anterior_posterior;
            size_t locus_idx = i % std::max(size_t(1), genome.loci.size());
            const auto& locus = genome.loci[locus_idx];

            if (ap < 0.25) {
                cells[i].fate = CellFate::RECEPTOR;
                cells[i].op_type = static_cast<uint8_t>(locus.op_type % 4); // SENSE0 ~ SENSE3
                receptors.push_back(cells[i].cell_id);
            } else if (ap > 0.75) {
                cells[i].fate = CellFate::EFFECTOR;
                cells[i].op_type = static_cast<uint8_t>(21 + (locus.op_type % 4)); // ACT_POS, ACT_NEG, ACT_LOCK
                effectors.push_back(cells[i].cell_id);
            } else {
                if (ap > 0.55 && (locus.op_type % 3 == 0)) {
                    cells[i].fate = CellFate::GATING;
                    cells[i].op_type = static_cast<uint8_t>(15 + (locus.op_type % 6)); // 门控算子
                } else {
                    cells[i].fate = CellFate::METABOLIC;
                    cells[i].op_type = static_cast<uint8_t>(4 + (locus.op_type % 11)); // 代谢算子
                }
                metabolic_cells.push_back(cells[i].cell_id);
            }

            cells[i].param1 = locus.weight_param;
            cells[i].param2 = locus.base_metabolic_rate;
        }

        // 突触轴突生长定向投射 (Axon Pathfinding): Receptor -> Metabolic -> Effector
        synapses.clear();
        for (uint32_t r_id : receptors) {
            for (uint32_t m_id : metabolic_cells) {
                if (rng() % 100 < 60) {
                    EmbryonicSynapse syn{r_id, m_id, 0, 1.0, true};
                    synapses.push_back(syn);
                }
            }
        }
        for (uint32_t m_id : metabolic_cells) {
            for (uint32_t e_id : effectors) {
                if (rng() % 100 < 60) {
                    EmbryonicSynapse syn{m_id, e_id, 0, 1.0, true};
                    synapses.push_back(syn);
                }
            }
        }

        // 凋亡剪枝 (Apoptosis): 剔除零入度且零出度的孤立细胞
        std::set<uint32_t> connected_cells;
        for (const auto& syn : synapses) {
            connected_cells.insert(syn.from_id);
            connected_cells.insert(syn.to_id);
        }

        cells.erase(std::remove_if(cells.begin(), cells.end(), [&](const EmbryonicCell& c) {
            return connected_cells.find(c.cell_id) == connected_cells.end();
        }), cells.end());

        // 发育完成判定：至少有 1 个受体与 1 个效应器成功建立通路
        if (!receptors.empty() && !effectors.empty() && !synapses.empty()) {
            stage = DevelopmentalStage::MATURE;
        } else {
            stage = DevelopmentalStage::INVIABLE;
        }
    }
};

// ============================================================================
// 4. 物种分化与基因重组 (Speciation & Sexual Recombination)
// ============================================================================
struct SpeciationMetric {
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

    // 两性基因交叉重组 (Sexual Crossover)
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
        ss << mom.lineage_hash.substr(0, 4) << dad.lineage_hash.substr(0, 4);
        child.lineage_hash = ss.str();

        return child;
    }
};

// ============================================================================
// 5. 成熟形态发生个体 (Morphogenetic Individual)
// ============================================================================
struct MorphogeneticIndividual {
    uint64_t id{0};
    uint32_t species_id{0};
    ReplicableGenome genome;
    DevelopmentalStage stage{DevelopmentalStage::ZYGOTE};
    size_t mature_cell_count{0};
    size_t mature_synapse_count{0};
    double fitness{0.0};
    double adjusted_fitness{0.0};
    double energy_reserve{100.0};
    uint32_t age_generations{0};
    bool is_viable{false};

    // 执行形态发生发育
    bool develop(std::mt19937& rng) {
        EmbryoMorphogenesisEngine engine(genome);
        stage = engine.develop_to_maturity(rng);
        if (stage == DevelopmentalStage::MATURE) {
            mature_cell_count = engine.cells.size();
            mature_synapse_count = engine.synapses.size();
            is_viable = true;
            return true;
        }
        is_viable = false;
        return false;
    }
};

// 物种生态利基 (Species Ecological Niche)
struct SpeciesNiche {
    uint32_t species_id{0};
    ReplicableGenome representative_genome;
    std::vector<MorphogeneticIndividual> members;
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
// 6. 具身形态发生种群生态系统 (Morphogenetic Population Ecosystem)
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
            MorphogeneticIndividual ind;
            ind.id = next_ind_id++;
            ind.species_id = founder_species.species_id;
            ind.genome = g;
            ind.develop(rng);
            founder_species.members.push_back(ind);
        }

        founder_species.representative_genome = founder_species.members[0].genome;
        species.push_back(founder_species);
    }

    // 运行一代繁衍演化周期 (Step Generation)
    void step_evolution_cycle() {
        generation++;

        // 1. 评估所有成熟个体的环境适应度
        for (auto& sp : species) {
            for (auto& ind : sp.members) {
                if (ind.is_viable) {
                    // 适应度函数：网络规模适度 + 突触连接密度 + 基因组丰富度
                    double complexity_score = static_cast<double>(ind.mature_cell_count) * 2.0 + static_cast<double>(ind.mature_synapse_count);
                    ind.fitness = std::max(1.0, complexity_score + (ind.genome.loci.size() * 1.5));
                } else {
                    ind.fitness = 0.1;
                }
            }
            sp.calculate_shared_fitness();
        }

        // 2. 收集所有存活个体并繁殖产生后代
        std::vector<MorphogeneticIndividual> offspring_pool;

        for (auto& sp : species) {
            if (sp.members.empty()) continue;

            // 排序选拔精英
            std::sort(sp.members.begin(), sp.members.end(), [](const auto& a, const auto& b) {
                return a.adjusted_fitness > b.adjusted_fitness;
            });

            // 精英直接保留
            offspring_pool.push_back(sp.members[0]);

            // 在物种内进行基因交叉重组 (Sexual Crossover)
            size_t offspring_count = std::max(size_t(1), sp.members.size());
            for (size_t k = 1; k < offspring_count; ++k) {
                const auto& parent_a = sp.members[rng() % sp.members.size()];
                const auto& parent_b = sp.members[rng() % sp.members.size()];

                ReplicableGenome child_genome = SpeciationMetric::sexual_crossover(parent_a.genome, parent_b.genome, rng);
                child_genome = child_genome.replicate_with_mutation(rng, 0.08);

                MorphogeneticIndividual child;
                child.id = next_ind_id++;
                child.genome = child_genome;
                child.develop(rng);
                offspring_pool.push_back(child);
            }
        }

        // 3. 物种重新聚类划分 (Speciation Clustering)
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
                // 成立新物种生态利基
                SpeciesNiche new_species;
                new_species.species_id = next_species_id++;
                new_species.representative_genome = ind.genome;
                ind.species_id = new_species.species_id;
                new_species.members.push_back(ind);
                species.push_back(new_species);
            }
        }

        // 4. 清理灭绝物种
        species.erase(std::remove_if(species.begin(), species.end(), [](const SpeciesNiche& sp) {
            return sp.members.empty();
        }), species.end());

        // 保持种群规模恒定
        size_t current_pop = get_total_population();
        if (current_pop < target_population_size && !species.empty()) {
            size_t deficit = target_population_size - current_pop;
            for (size_t d = 0; d < deficit; ++d) {
                auto& sp = species[rng() % species.size()];
                ReplicableGenome g = sp.representative_genome.replicate_with_mutation(rng, 0.1);
                MorphogeneticIndividual ind;
                ind.id = next_ind_id++;
                ind.species_id = sp.species_id;
                ind.genome = g;
                ind.develop(rng);
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
        g.lineage_hash = "Zygote01";

        // 初始基础种子位点
        for (uint32_t i = 0; i < 8; ++i) {
            GeneLocus loc;
            loc.gene_id = i;
            loc.op_type = static_cast<uint8_t>(i % 6);
            loc.weight_param = 1.0;
            loc.base_metabolic_rate = 0.20;
            g.loci.push_back(loc);
        }
        return g;
    }
};

} // namespace kun

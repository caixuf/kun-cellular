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

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/autonomous_replicator.hpp"

namespace kun {

// ============================================================================
// 1. 胚胎发育阶段 (Embryonic Morphogenesis Developmental Stages)
// ============================================================================
enum class EmbryoStage : uint8_t {
    ZYGOTE = 0,         // 单细胞受精卵
    CLEAVAGE = 1,       // 快速卵裂扩增
    GASTRULATION = 2,   // 原肠极性轴迁移与力场弛豫
    DIFFERENTIATION = 3,// 形态素梯度诱导命运分化
    SYNAPTOGENESIS = 4, // 轴突定向投射与突触成型
    MATURE = 5          // 成熟机能适配器
};

inline const char* to_string(EmbryoStage s) {
    switch (s) {
        case EmbryoStage::ZYGOTE: return "ZYGOTE";
        case EmbryoStage::CLEAVAGE: return "CLEAVAGE";
        case EmbryoStage::GASTRULATION: return "GASTRULATION";
        case EmbryoStage::DIFFERENTIATION: return "DIFFERENTIATION";
        case EmbryoStage::SYNAPTOGENESIS: return "SYNAPTOGENESIS";
        case EmbryoStage::MATURE: return "MATURE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 2. 胚胎发育适配器引擎 (Embryo Morphogenesis Adapter Engine)
// ============================================================================
class EmbryoAdapterEngine {
public:
    uint64_t embryo_id{0};
    EmbryoStage current_stage{EmbryoStage::ZYGOTE};
    ReplicableGenome maternal_genome;
    CellularOrganism mature_organism;
    
    // 胚胎期形态素浓度梯度场 (Morphogen Gradients: Anterior-Posterior & Dorsal-Ventral)
    std::vector<float> morphogen_ap; // 前后轴浓度 (Anterior-Posterior: -120 to +140)
    std::vector<float> morphogen_dv; // 背腹轴浓度 (Dorsal-Ventral: -60 to +60)
    
    explicit EmbryoAdapterEngine(const ReplicableGenome& genome, uint64_t id = 1)
        : embryo_id(id), maternal_genome(genome) {}

    // 执行完整的胚胎自发诱导发育流程 (Develop to mature adapter)
    bool develop(std::mt19937& rng) {
        // Step 1: 受精卵启动
        current_stage = EmbryoStage::ZYGOTE;
        mature_organism = CellularOrganism();
        mature_organism.cells.clear();
        mature_organism.synapses.clear();

        // 初始单细胞 (受精卵原基)
        mature_organism.cells.push_back({0, CellType::OP_EMA, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});

        // Step 2: 卵裂期 (Cleavage) - 根据基因位点规模指数分裂扩增细胞群
        current_stage = EmbryoStage::CLEAVAGE;
        size_t target_cells = std::max(size_t(6), maternal_genome.loci.size() + 4);
        uint32_t next_cid = 1;
        while (mature_organism.cells.size() < target_cells) {
            float rx = std::uniform_real_distribution<float>(-40.0f, 40.0f)(rng);
            float ry = std::uniform_real_distribution<float>(-30.0f, 30.0f)(rng);
            mature_organism.cells.push_back({next_cid++, CellType::OP_EMA, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, rx, ry, 0.0f});
        }

        // Step 3: 原肠期 (Gastrulation) - 极性轴形成与 3D 兰纳-琼斯力场排斥迁移
        current_stage = EmbryoStage::GASTRULATION;
        for (size_t iter = 0; iter < 10; ++iter) {
            mature_organism.step_force_field_physics(0.02f);
        }

        // Step 4: 分化期 (Differentiation) - 依据沿极性轴的空间位置与形态素梯度确定细胞命运
        current_stage = EmbryoStage::DIFFERENTIATION;
        // 按 X 坐标 (前后轴) 排序分配命运
        std::sort(mature_organism.cells.begin(), mature_organism.cells.end(), [](const Cell& a, const Cell& b) {
            return a.x < b.x;
        });

        // 最前端 (X 极小处) 诱导为感受器 (Sensory Receptors)
        mature_organism.cells[0].type = CellType::SENSE_RAW_INPUT_0;
        mature_organism.cells[0].x = -120.0f; mature_organism.cells[0].y = -40.0f;
        mature_organism.cells[1].type = CellType::SENSE_RAW_INPUT_1;
        mature_organism.cells[1].x = -120.0f; mature_organism.cells[1].y = 40.0f;

        // 最后端 (X 极大处) 诱导为效应器 (Action Effectors)
        size_t n = mature_organism.cells.size();
        mature_organism.cells[n - 2].type = CellType::ACT_PRIMARY_POSITIVE;
        mature_organism.cells[n - 2].x = 140.0f; mature_organism.cells[n - 2].y = -40.0f;
        mature_organism.cells[n - 1].type = CellType::ACT_PRIMARY_NEGATIVE;
        mature_organism.cells[n - 1].x = 140.0f; mature_organism.cells[n - 1].y = 40.0f;

        // 中间层按基因位点表达代谢与门控细胞
        static const CellType kMidTypes[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS, CellType::OP_DELAY_N,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_DEADZONE
        };

        for (size_t i = 2; i < n - 2; ++i) {
            size_t locus_idx = (i - 2) % std::max(size_t(1), maternal_genome.loci.size());
            const auto& loc = maternal_genome.loci[locus_idx];
            mature_organism.cells[i].type = kMidTypes[loc.op_type % (sizeof(kMidTypes) / sizeof(kMidTypes[0]))];
            mature_organism.cells[i].param1 = loc.weight_param;
        }

        // Step 5: 突触发生 (Synaptogenesis) - 依据前向因果与局部近邻投射轴突连接
        current_stage = EmbryoStage::SYNAPTOGENESIS;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                // 仅向后轴正向投射 (前向因果拓扑)
                if (mature_organism.cells[j].x > mature_organism.cells[i].x + 10.0f) {
                    float dx = mature_organism.cells[j].x - mature_organism.cells[i].x;
                    float dy = mature_organism.cells[j].y - mature_organism.cells[i].y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < 160.0f) {
                        double syn_w = (j == n - 1) ? -1.0 : 1.0;
                        mature_organism.synapses.push_back({
                            mature_organism.cells[i].id,
                            mature_organism.cells[j].id,
                            0,
                            syn_w,
                            true,
                            60.0f,
                            -1.0f
                        });
                    }
                }
            }
        }

        // Step 6: 编译为成熟扁平拓扑 (Compile to Mature Flat Array)
        mature_organism.compile();
        current_stage = EmbryoStage::MATURE;
        return (mature_organism.cells.size() >= 4 && !mature_organism.synapses.empty());
    }
};

} // namespace kun

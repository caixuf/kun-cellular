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
    DIFFERENTIATION = 3,// 图灵形态发生素场诱导器官命运分化
    ADHESION = 4,       // 钙粘蛋白同嗜性粘附与器官外包膜成型
    SYNAPTOGENESIS = 5, // 轴突定向投射与突触成型
    MATURE = 6          // 成熟机能适配器
};

inline const char* to_string(EmbryoStage s) {
    switch (s) {
        case EmbryoStage::ZYGOTE: return "ZYGOTE";
        case EmbryoStage::CLEAVAGE: return "CLEAVAGE";
        case EmbryoStage::GASTRULATION: return "GASTRULATION";
        case EmbryoStage::DIFFERENTIATION: return "DIFFERENTIATION";
        case EmbryoStage::ADHESION: return "ADHESION";
        case EmbryoStage::SYNAPTOGENESIS: return "SYNAPTOGENESIS";
        case EmbryoStage::MATURE: return "MATURE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 1.5 图灵反应-扩散形态发生素动力学场 (Turing Reaction-Diffusion Morphogen Field)
// ============================================================================
struct MorphogenTuringField {
    static constexpr size_t GRID_SIZE = 32; // 极性空间剖分格点
    float u[GRID_SIZE]; // 激活素 U (自催化、慢扩散)
    float v[GRID_SIZE]; // 抑制素 V (侧向抑制、快扩散)

    float du{0.06f};    // U 扩散系数
    float dv{0.48f};    // V 扩散系数
    float rho{0.08f};   // 反应速率
    float mu_u{0.05f};  // U 降解率
    float mu_v{0.07f};  // V 降解率
    float sigma_u{0.02f};// 基础前驱素生成速率

    MorphogenTuringField() {
        init_field(12345);
    }

    // 初始化微小随机扰动的均匀场 (Symmetry Breaking Initial State)
    void init_field(uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> noise(-0.02f, 0.02f);
        for (size_t i = 0; i < GRID_SIZE; ++i) {
            u[i] = 1.0f + noise(rng);
            v[i] = 1.0f + noise(rng);
        }
    }

    // 运行 Gierer-Meinhardt 偏微分方程单步迭代
    void step(float dt = 0.20f) {
        float next_u[GRID_SIZE];
        float next_v[GRID_SIZE];

        for (size_t i = 0; i < GRID_SIZE; ++i) {
            // 拉普拉斯算子 ∇² (Neumann 边界条件 / 无通量绝热边界)
            float left_u  = (i > 0) ? u[i - 1] : u[0];
            float right_u = (i + 1 < GRID_SIZE) ? u[i + 1] : u[GRID_SIZE - 1];
            float lap_u = left_u - 2.0f * u[i] + right_u;

            float left_v  = (i > 0) ? v[i - 1] : v[0];
            float right_v = (i + 1 < GRID_SIZE) ? v[i + 1] : v[GRID_SIZE - 1];
            float lap_v = left_v - 2.0f * v[i] + right_v;

            // Gierer-Meinhardt 动力学方程
            float reaction_u = rho * (u[i] * u[i]) / (v[i] + 0.01f) - mu_u * u[i] + sigma_u;
            float reaction_v = rho * (u[i] * u[i]) - mu_v * v[i];

            next_u[i] = std::clamp(u[i] + (du * lap_u + reaction_u) * dt, 0.05f, 5.0f);
            next_v[i] = std::clamp(v[i] + (dv * lap_v + reaction_v) * dt, 0.05f, 5.0f);
        }

        for (size_t i = 0; i < GRID_SIZE; ++i) {
            u[i] = next_u[i];
            v[i] = next_v[i];
        }
    }

    // 沿前后轴空间坐标 ([-140, +160]) 连续插值采样形态发生素浓度
    float sample_u(float x) const {
        float norm_x = std::clamp((x + 140.0f) / 300.0f, 0.0f, 1.0f);
        float f_idx = norm_x * (GRID_SIZE - 1);
        size_t idx0 = static_cast<size_t>(std::floor(f_idx));
        size_t idx1 = std::min(idx0 + 1, GRID_SIZE - 1);
        float frac = f_idx - idx0;
        return u[idx0] * (1.0f - frac) + u[idx1] * frac;
    }

    float sample_ratio(float x) const {
        float norm_x = std::clamp((x + 140.0f) / 300.0f, 0.0f, 1.0f);
        float f_idx = norm_x * (GRID_SIZE - 1);
        size_t idx0 = static_cast<size_t>(std::floor(f_idx));
        size_t idx1 = std::min(idx0 + 1, GRID_SIZE - 1);
        float frac = f_idx - idx0;
        float u_val = u[idx0] * (1.0f - frac) + u[idx1] * frac;
        float v_val = v[idx0] * (1.0f - frac) + v[idx1] * frac;
        return u_val / (v_val + 0.01f);
    }
};

// ============================================================================
// 1.8 三维生物器官形态发生外包膜结构 (Organ Morphogenetic Capsule)
// ============================================================================
struct OrganCapsule {
    uint8_t organ_id{0};       // 0=SensoryColumn, 1=AssociationCortex, 2=MotorEffectorCore
    std::string name;          // 器官名称
    float center_x{0.0f};      // 3D 几何质心 X
    float center_y{0.0f};      // 3D 几何质心 Y
    float center_z{0.0f};      // 3D 几何质心 Z
    float radius_x{20.0f};     // 三维包络椭球半径 X
    float radius_y{20.0f};     // 三维包络椭球半径 Y
    float radius_z{20.0f};     // 三维包络椭球半径 Z
    uint32_t cell_count{0};    // 归属该器官的实体细胞数
    std::vector<uint32_t> member_cell_ids; // 成员细胞 ID 列表
};

// ============================================================================
// 2. 胚胎发育适配器引擎 (Embryo Morphogenesis Adapter Engine)
// ============================================================================
class EmbryoAdapterEngine {
public:
    uint64_t embryo_id{0};
    EmbryoStage current_stage{EmbryoStage::ZYGOTE};
    ReplicableGenome maternal_genome;
    CellularOrganism mature_organism;
    
    // 图灵反应-扩散形态发生素动力学场
    MorphogenTuringField morphogen_field;
    // 自主发育凝聚出的三维器官外包膜列表
    std::vector<OrganCapsule> organ_capsules;
    
    explicit EmbryoAdapterEngine(const ReplicableGenome& genome, uint64_t id = 1)
        : embryo_id(id), maternal_genome(genome) {}

    // 执行完整的胚胎自发诱导发育流程 (Develop to mature adapter)
    bool develop(std::mt19937& rng) {
        // Step 1: 受精卵启动
        current_stage = EmbryoStage::ZYGOTE;
        mature_organism = CellularOrganism();
        mature_organism.cells.clear();
        mature_organism.synapses.clear();
        organ_capsules.clear();

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
        // 求解图灵反应扩散偏微分方程，自发破缺对称性，形成头尾前后轴形态发生素浓度坡度
        morphogen_field.init_field(static_cast<uint32_t>(maternal_genome.genome_id ^ 0xABCD));
        for (int step = 0; step < 40; ++step) {
            morphogen_field.step(0.20f);
        }

        for (size_t iter = 0; iter < 12; ++iter) {
            mature_organism.step_force_field_physics(0.02f);
        }

        // Step 4: 图灵形态发生素场诱导细胞命运与器官分化 (Differentiation)
        current_stage = EmbryoStage::DIFFERENTIATION;
        // 按 X 坐标 (前后轴) 排序
        std::sort(mature_organism.cells.begin(), mature_organism.cells.end(), [](const Cell& a, const Cell& b) {
            return a.x < b.x;
        });

        // 为每个细胞采样形态素浓度并决定器官命运
        size_t n = mature_organism.cells.size();
        // 最前端 (X 极小处) 诱导为感受器 (Sensory Receptors) -> Organ 0
        mature_organism.cells[0].type = CellType::SENSE_RAW_INPUT_0;
        mature_organism.cells[0].x = -120.0f; mature_organism.cells[0].y = -40.0f;
        mature_organism.cells[0].organ_type = 0;
        mature_organism.cells[0].morphogen_concentration = morphogen_field.sample_u(-120.0f);

        mature_organism.cells[1].type = CellType::SENSE_RAW_INPUT_1;
        mature_organism.cells[1].x = -120.0f; mature_organism.cells[1].y = 40.0f;
        mature_organism.cells[1].organ_type = 0;
        mature_organism.cells[1].morphogen_concentration = morphogen_field.sample_u(-120.0f);

        // 最后端 (X 极大处) 诱导为效应器 (Action Effectors) -> Organ 2
        mature_organism.cells[n - 2].type = CellType::ACT_PRIMARY_POSITIVE;
        mature_organism.cells[n - 2].x = 140.0f; mature_organism.cells[n - 2].y = -40.0f;
        mature_organism.cells[n - 2].organ_type = 2;
        mature_organism.cells[n - 2].morphogen_concentration = morphogen_field.sample_u(140.0f);

        mature_organism.cells[n - 1].type = CellType::ACT_PRIMARY_NEGATIVE;
        mature_organism.cells[n - 1].x = 140.0f; mature_organism.cells[n - 1].y = 40.0f;
        mature_organism.cells[n - 1].organ_type = 2;
        mature_organism.cells[n - 1].morphogen_concentration = morphogen_field.sample_u(140.0f);

        // 中间层按基因位点表达代谢与门控细胞 -> Organ 1 (Association Cortex)
        static const CellType kMidTypes[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS, CellType::OP_DELAY_N,
            CellType::OP_OSCILLATOR,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_DEADZONE
        };

        for (size_t i = 2; i < n - 2; ++i) {
            size_t locus_idx = (i - 2) % std::max(size_t(1), maternal_genome.loci.size());
            const auto& loc = maternal_genome.loci[locus_idx];
            mature_organism.cells[i].type = kMidTypes[loc.op_type % (sizeof(kMidTypes) / sizeof(kMidTypes[0]))];
            mature_organism.cells[i].param1 = loc.weight_param;
            mature_organism.cells[i].organ_type = 1;
            mature_organism.cells[i].morphogen_concentration = morphogen_field.sample_u(mature_organism.cells[i].x);
        }

        // Step 5: 钙粘蛋白同嗜性粘附与器官外包膜成型 (Adhesion & Organ Capsule Cohesion)
        current_stage = EmbryoStage::ADHESION;
        // 同类器官细胞施加同嗜性吸引弹簧，凝聚为清晰的三维器官团簇
        const float cadherin_k = 0.08f;
        for (size_t iter = 0; iter < 10; ++iter) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    if (mature_organism.cells[i].organ_type == mature_organism.cells[j].organ_type) {
                        float dx = mature_organism.cells[j].x - mature_organism.cells[i].x;
                        float dy = mature_organism.cells[j].y - mature_organism.cells[i].y;
                        float dz = mature_organism.cells[j].z - mature_organism.cells[i].z;
                        float d = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-4f);
                        if (d > 45.0f) {
                            float f = (d - 45.0f) * cadherin_k;
                            float inv = f / d;
                            mature_organism.cells[i].x += dx * inv * 0.5f;
                            mature_organism.cells[i].y += dy * inv * 0.5f;
                            mature_organism.cells[i].z += dz * inv * 0.5f;
                            mature_organism.cells[j].x -= dx * inv * 0.5f;
                            mature_organism.cells[j].y -= dy * inv * 0.5f;
                            mature_organism.cells[j].z -= dz * inv * 0.5f;
                        }
                    }
                }
            }
        }

        // 拟合与生成 3 个器官外包膜几何参数
        compute_organ_capsules();

        // Step 6: 突触发生 (Synaptogenesis) - 依据前向因果与局部近邻投射轴突连接
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

        // Step 7: 编译为成熟扁平拓扑 (Compile to Mature Flat Array)
        mature_organism.compile();
        current_stage = EmbryoStage::MATURE;
        return (mature_organism.cells.size() >= 4 && !mature_organism.synapses.empty());
    }

    // 统计并拟合各器官的三维外包膜几何参数 (Centroid, Bounding Radius, Cell Members)
    void compute_organ_capsules() {
        organ_capsules.clear();
        static const char* kOrganNames[] = {"SensoryColumn", "AssociationCortex", "MotorEffectorCore"};

        for (uint8_t org_id = 0; org_id < 3; ++org_id) {
            OrganCapsule cap;
            cap.organ_id = org_id;
            cap.name = kOrganNames[org_id];

            float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
            for (const auto& c : mature_organism.cells) {
                if (c.organ_type == org_id) {
                    sum_x += c.x; sum_y += c.y; sum_z += c.z;
                    cap.cell_count++;
                    cap.member_cell_ids.push_back(c.id);
                }
            }

            if (cap.cell_count > 0) {
                cap.center_x = sum_x / cap.cell_count;
                cap.center_y = sum_y / cap.cell_count;
                cap.center_z = sum_z / cap.cell_count;

                float max_dx = 15.0f, max_dy = 15.0f, max_dz = 15.0f;
                for (const auto& c : mature_organism.cells) {
                    if (c.organ_type == org_id) {
                        max_dx = std::max(max_dx, std::abs(c.x - cap.center_x));
                        max_dy = std::max(max_dy, std::abs(c.y - cap.center_y));
                        max_dz = std::max(max_dz, std::abs(c.z - cap.center_z));
                    }
                }
                cap.radius_x = max_dx + 12.0f;
                cap.radius_y = max_dy + 12.0f;
                cap.radius_z = max_dz + 12.0f;
            }
            organ_capsules.push_back(cap);
        }
    }
};

} // namespace kun

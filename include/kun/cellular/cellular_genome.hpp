#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (Software-Defined Silicon Cellular Computer, SDSCC)
// 核心原语定义、算存一体局部状态、三维力敏自组织与 Kahn 拓扑零 GC 编译器
//
// 理论渊源: 约翰·冯·诺依曼自复制自动机理论 (1966) 与阿兰·图灵形态发生动力学 (1952)
// 架构特性: 原生算存一体、24 种离散物理动力学原语、零 GC 扁平内存排布、因果可证伪
// ============================================================================

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <optional>
#include <cmath>
#include <random>
#include <memory>
#include <sstream>
#include <fstream>
#include <iostream>
#include <functional>
#include <queue>
#include <atomic>
#include <thread>

namespace kun {

// ============================================================================
// 1. 细胞类型 (CellType): 生物计算原语
// ============================================================================
enum class CellType : uint8_t {
    // 【感知受体细胞 (Receptor / Sensory)】
    SENSE_RAW_INPUT_0 = 0,   // 输入通道 0 (基础空间观测通道 0)
    SENSE_RAW_INPUT_1 = 1,   // 输入通道 1 (基础空间观测通道 1)
    SENSE_RAW_INPUT_2 = 2,   // 输入通道 2 (基础空间观测通道 2)
    SENSE_RAW_INPUT_3 = 3,   // 输入通道 3 (基础空间观测通道 3)
    SENSE_CHANNEL     = 4,   // 任意维受体: output = inputs[floor(param2)] * param1
    
    // 【代谢运算细胞 (Metabolic Math Operators)】
    OP_EMA = 10,             // 指数平滑滤波 (具备衰减记忆)
    OP_DIFF = 11,            // 微分/变化率 (感知加速度与斜率)
    OP_INTEGRAL = 12,        // 积分累加 (能量积聚与趋势持久度)
    OP_SUM = 13,             // 突触信号线性叠加 (Input A + Input B)
    OP_SUB = 14,             // 突触信号差分对比 (Input A - Input B)
    OP_MULTIPLY = 15,        // 增益调节与调制 (Input A * Input B)
    OP_RATIO = 16,           // 相对比率 (Input A / (Input B + eps))
    OP_ABS = 17,             // 绝对值幅度 (|Input A|)
    OP_DELAY_N = 18,         // 延迟管道 (Sliding FIFO delay u(t) = x(t-k))
    OP_OSCILLATOR = 19,      // 范德波尔相弛豫振荡器 (Van der Pol Oscillator)
    OP_QUADRATIC = 20,       // 二次能量型 (Quadratic Lyapunov energy form)
    
    // 【门控逻辑神经元 (Gating / Threshold Neurons)】
    GATE_THRESHOLD = 24,     // 阶跃阈值激活 (Input > param ? 1 : 0)
    GATE_HYSTERESIS = 25,    // 迟滞比较器 (Schmitt Trigger，防震荡高频抖动)
    GATE_AND = 26,           // 协同兴奋门 (Input A > 0 && Input B > 0)
    GATE_INHIBIT = 27,       // 抑制性突触 (Input A * (1.0 - Input B))
    GATE_DEADZONE = 28,      // 中心死区噪声门 (Central deadband noise gate)
    GATE_MIN_MAX = 29,       // 极值包络门 (Extremum envelope gate)
    
    // 【效应/动作细胞 (Effector / Action)】
    ACT_PRIMARY_POSITIVE = 30, // 正向激发效应器 (Primary Positive Actuator)
    ACT_PRIMARY_NEGATIVE = 31, // 反向激发效应器 (Primary Negative Actuator)
    ACT_DEFENSIVE_RESET  = 32, // 防御性复位/阻尼归零效应器 (Defensive Reset / Damping Effector)
    ACT_IMMUNE_BLOCK     = 33, // 免疫阻断效应器 (Immune Inhibit / Safety Cutoff Effector)
    ACT_CHANNEL          = 34, // 任意维效应器: 写入动作张量槽 floor(param2)

    // 【认知联络与预测受体 (Cognitive & Predictive World Model)】
    PREDICT_SENSE_0      = 40, // 内部前瞻预测受体 0 (预测下一时刻输入0, 内部世界模型输出)
    PREDICT_SENSE_1      = 41, // 内部前瞻预测受体 1 (预测下一时刻输入1)
    ASSOCIATION_HUB      = 42  // 联络皮层中枢细胞 (皮层柱联想聚类, 概念吸引子表征)
};

inline const char* to_string(CellType t) {
    switch (t) {
        case CellType::SENSE_RAW_INPUT_0: return "Sense_Input0";
        case CellType::SENSE_RAW_INPUT_1: return "Sense_Input1";
        case CellType::SENSE_RAW_INPUT_2: return "Sense_Input2";
        case CellType::SENSE_RAW_INPUT_3: return "Sense_Input3";
        case CellType::SENSE_CHANNEL: return "Sense_Channel";
        case CellType::OP_EMA: return "Op_EMA";
        case CellType::OP_DIFF: return "Op_Diff";
        case CellType::OP_INTEGRAL: return "Op_Integral";
        case CellType::OP_SUM: return "Op_Sum";
        case CellType::OP_SUB: return "Op_Sub";
        case CellType::OP_MULTIPLY: return "Op_Multiply";
        case CellType::OP_RATIO: return "Op_Ratio";
        case CellType::OP_ABS: return "Op_Abs";
        case CellType::OP_DELAY_N: return "Op_DelayN";
        case CellType::OP_OSCILLATOR: return "Op_Oscillator";
        case CellType::OP_QUADRATIC: return "Op_Quadratic";
        case CellType::GATE_THRESHOLD: return "Gate_Threshold";
        case CellType::GATE_HYSTERESIS: return "Gate_Hysteresis";
        case CellType::GATE_AND: return "Gate_And";
        case CellType::GATE_INHIBIT: return "Gate_Inhibit";
        case CellType::GATE_DEADZONE: return "Gate_Deadzone";
        case CellType::GATE_MIN_MAX: return "Gate_MinMax";
        case CellType::ACT_PRIMARY_POSITIVE: return "Act_PosAction";
        case CellType::ACT_PRIMARY_NEGATIVE: return "Act_NegAction";
        case CellType::ACT_DEFENSIVE_RESET: return "Act_DefReset";
        case CellType::ACT_IMMUNE_BLOCK: return "Act_ImmuneLock";
        case CellType::ACT_CHANNEL: return "Act_Channel";
        case CellType::PREDICT_SENSE_0: return "Pred_Sense0";
        case CellType::PREDICT_SENSE_1: return "Pred_Sense1";
        case CellType::ASSOCIATION_HUB: return "Assoc_Hub";
        default: return "Cell_Unknown";
    }
}

inline CellType cell_type_from_string(const std::string& name) {
    if (name == "Sense_Input0") return CellType::SENSE_RAW_INPUT_0;
    if (name == "Sense_Input1") return CellType::SENSE_RAW_INPUT_1;
    if (name == "Sense_Input2") return CellType::SENSE_RAW_INPUT_2;
    if (name == "Sense_Input3") return CellType::SENSE_RAW_INPUT_3;
    if (name == "Sense_Channel") return CellType::SENSE_CHANNEL;
    if (name == "Op_EMA") return CellType::OP_EMA;
    if (name == "Op_Diff") return CellType::OP_DIFF;
    if (name == "Op_Integral") return CellType::OP_INTEGRAL;
    if (name == "Op_Sum") return CellType::OP_SUM;
    if (name == "Op_Sub") return CellType::OP_SUB;
    if (name == "Op_Multiply") return CellType::OP_MULTIPLY;
    if (name == "Op_Ratio") return CellType::OP_RATIO;
    if (name == "Op_Abs") return CellType::OP_ABS;
    if (name == "Op_DelayN") return CellType::OP_DELAY_N;
    if (name == "Op_Oscillator") return CellType::OP_OSCILLATOR;
    if (name == "Op_Quadratic") return CellType::OP_QUADRATIC;
    if (name == "Gate_Threshold") return CellType::GATE_THRESHOLD;
    if (name == "Gate_Hysteresis") return CellType::GATE_HYSTERESIS;
    if (name == "Gate_And") return CellType::GATE_AND;
    if (name == "Gate_Inhibit") return CellType::GATE_INHIBIT;
    if (name == "Gate_Deadzone") return CellType::GATE_DEADZONE;
    if (name == "Gate_MinMax") return CellType::GATE_MIN_MAX;
    if (name == "Act_PosAction") return CellType::ACT_PRIMARY_POSITIVE;
    if (name == "Act_NegAction") return CellType::ACT_PRIMARY_NEGATIVE;
    if (name == "Act_DefReset") return CellType::ACT_DEFENSIVE_RESET;
    if (name == "Act_ImmuneLock") return CellType::ACT_IMMUNE_BLOCK;
    if (name == "Act_Channel") return CellType::ACT_CHANNEL;
    if (name == "Pred_Sense0") return CellType::PREDICT_SENSE_0;
    if (name == "Pred_Sense1") return CellType::PREDICT_SENSE_1;
    if (name == "Assoc_Hub") return CellType::ASSOCIATION_HUB;
    return CellType::OP_EMA;
}

inline bool is_receptor_cell(CellType t) {
    return t == CellType::SENSE_RAW_INPUT_0 ||
           t == CellType::SENSE_RAW_INPUT_1 ||
           t == CellType::SENSE_RAW_INPUT_2 ||
           t == CellType::SENSE_RAW_INPUT_3 ||
           t == CellType::SENSE_CHANNEL;
}

inline bool is_effector_cell(CellType t) {
    return t == CellType::ACT_PRIMARY_POSITIVE ||
           t == CellType::ACT_PRIMARY_NEGATIVE ||
           t == CellType::ACT_DEFENSIVE_RESET ||
           t == CellType::ACT_IMMUNE_BLOCK ||
           t == CellType::ACT_CHANNEL;
}

inline size_t receptor_channel_index(CellType t, double param2) {
    switch (t) {
        case CellType::SENSE_RAW_INPUT_0: return 0;
        case CellType::SENSE_RAW_INPUT_1: return 1;
        case CellType::SENSE_RAW_INPUT_2: return 2;
        case CellType::SENSE_RAW_INPUT_3: return 3;
        case CellType::SENSE_CHANNEL:
            return (param2 >= 0.0) ? static_cast<size_t>(param2) : 0;
        default: return 0;
    }
}

inline size_t effector_channel_index(CellType t, double param2) {
    switch (t) {
        case CellType::ACT_PRIMARY_POSITIVE: return 0;
        case CellType::ACT_PRIMARY_NEGATIVE: return 1;
        case CellType::ACT_DEFENSIVE_RESET: return 2;
        case CellType::ACT_IMMUNE_BLOCK: return 3;
        case CellType::ACT_CHANNEL:
            return (param2 >= 0.0) ? static_cast<size_t>(param2) : 0;
        default: return 0;
    }
}

// ============================================================================
// 2. 突触连接 (Synapse): 细胞间传递与调控通路 (原生内嵌力学弹簧与放电粒子)
// ============================================================================
struct Synapse {
    uint32_t from_cell_id{0};  // 发射端细胞 ID (突触前膜)
    uint32_t to_cell_id{0};    // 接收端细胞 ID (突触后膜)
    uint8_t  to_port{0};       // 目标细胞端口号 (0=主输入, 1=辅助输入/门控端)
    double   weight{1.0};      // 传递强度 (可塑性突触权重)
    bool     is_active{true};  // 突触激活态

    // === 原生一等公民力学与全息可视化属性 ===
    float rest_length{60.0f};  // 弹簧物理静止长度
    float photon_pos{-1.0f};   // 神经电冲动放电光子位置 (0.0 -> 1.0)

    // === 终身可塑性与递归循环动力学参数 (Lifelong Plasticity & Recurrent Dynamics) ===
    double initial_weight{1.0}; // 基因始祖基础权重
    double hebbian_rate{0.005}; // 赫布/Oja 在线学习率 eta
    double hebbian_decay{0.02}; // Oja 自归一化衰减系数 alpha
    bool   is_recurrent{false}; // 是否为时序反馈循环突触
};

// ============================================================================
// 3. 细胞实例 (Cell): 独立生命计算节点 (原生内嵌 3D 坐标、力学与生物发光)
// ============================================================================
struct Cell {
    uint32_t id{0};
    CellType type{CellType::OP_EMA};
    
    // 细胞参数 (如 EMA 衰减系数 alpha, 门控阈值等)
    double param1{0.1};
    double param2{0.0};
    
    // 内部生物状态 (记忆与积分)
    double state_val{0.0};
    double prev_input{0.0};
    bool   latch_state{false};
    
    // 运行时当前放电电位 (Output Voltage)
    double output_val{0.0};
    
    // 细胞活跃度计数与代谢生命期 (用于凋亡剪枝)
    uint32_t activation_count{0};
    uint64_t last_active_ts{0};

    // === 原生一等公民 3D 物理空间坐标与力学属性 ===
    float x{0.0f}, y{0.0f}, z{0.0f};       // 培养皿空间坐标
    float vx{0.0f}, vy{0.0f}, vz{0.0f};    // 速度矢量
    float fx{0.0f}, fy{0.0f}, fz{0.0f};    // 受到的合力 (兰纳-琼斯力场)
    float glow_charge{0.0f};               // 生物发光强度 (0.0 ~ 1.0, 脉冲放电后渐变衰减)

    // === 高阶动态与零 GC 缓冲状态 (编译期内嵌连续内存，绝无堆分配) ===
    double aux_state{0.0};                 // 辅助内部动力学状态 (如 OP_OSCILLATOR 的速度变量 s2)
    double delay_buffer[16]{0.0};          // 静态滑动 FIFO 管道缓冲 (用于 OP_DELAY_N)
    uint8_t delay_idx{0};                  // FIFO 环形缓冲区游标 (用于 OP_DELAY_N)
    double prev_output_val{0.0};           // 上一时刻时序输出 (用于递归循环 1-step delay)

    // === 力敏转导应变与皮层沟回折叠属性 (Mechanotransduction & Cortical Folding) ===
    float physical_stress{0.0f};           // 物理力场累积剪切应力 ||F||
    float informational_strain{0.0f};      // 预测误差惊奇度应变
    uint16_t mitosis_cooldown{0};          // 力敏有丝分裂生物学冷却期
    float get_total_strain() const {
        return physical_stress * 0.05f + 2.0f * informational_strain;
    }

    // === 膜孔道与微观跨膜电位及代谢动力学 (Transmembrane Pores & Metabolic Gating) ===
    float membrane_potential{-70.0f};      // 跨膜电位 (mV)
    float membrane_pores[8]{0.20f, 0.20f, 0.10f, 0.10f, 0.40f, 0.30f, 0.05f, 0.05f}; // 8 个膜孔道开度
    uint8_t organ_type{1};                 // 所属器官归属 (0=Sensory, 1=Association, 2=Motor)
    float morphogen_concentration{0.50f};  // 局部感受到的图灵形态发生素浓度

    const char* type_name() const {
        return to_string(type);
    }
};

// ============================================================================
// 3.5 初始胚胎发生模式 (Seed Initialization Mode)
// ============================================================================
enum class SeedInitMode : uint8_t {
    HANDCRAFTED_PROGENITOR = 0, // Mode A: 9-cell Genesis seed (2 receptors, 3 metabolic, 1 gating, 3 effectors, 7 synapses)
    MINIMAL_RANDOM_GRAPH = 1,   // Mode B: 3-4 cells (2 receptors + 1 random metabolic/gating + 1-2 effectors with random weights U(-2, 2))
    DISCONNECTED_EMBRYO = 2,    // Mode C: Pure receptors and effectors with no intermediate connections at gen-0
    CONTRACT_PROGENITOR = 3     // Mode D: 15-cell contract progenitor (4 sensors, 4 actuators, 7 internal ops/gates)
};

inline const char* to_string(SeedInitMode mode) {
    switch (mode) {
        case SeedInitMode::HANDCRAFTED_PROGENITOR: return "Handcrafted_Progenitor";
        case SeedInitMode::MINIMAL_RANDOM_GRAPH: return "Minimal_Random_Graph";
        case SeedInitMode::DISCONNECTED_EMBRYO: return "Disconnected_Embryo";
        case SeedInitMode::CONTRACT_PROGENITOR: return "Contract_Progenitor";
        default: return "Unknown_Mode";
    }
}

// ── 演化约束消融实验配置 (Evolution Constraint Ablation Strategy) ──
enum class SkeletonLockMode : uint8_t {
    LOCKED = 0,    // 感受器与效应器严格不可修改/增删 (结构受限骨架)
    UNLOCKED = 1   // 感受器与效应器自由变异、增殖与凋亡 (自由开放形态发生)
};

enum class TypeWhitelistMode : uint8_t {
    CURATED_9 = 0, // 9 种基础代谢与门控算子白名单
    FULL_24 = 1    // 24 种全原语分类学 (完整算子空间)
};

enum class FitnessDriverMode : uint8_t {
    TASK_FITNESS_ONLY = 0, // 纯外部任务目标打分
    NOVELTY_SEARCH = 1,    // 纯内在动机：行为空间 KNN 稀缺度搜索
    HYBRID_CURIOSITY = 2   // 混合驱动：任务适应度 + 好奇心新颖性奖励
};

struct FunctionalCoverageContract {
    uint32_t active_sources_mask{0};
    uint32_t active_actuators_mask{0};
    size_t reachable_cells{0};
    uint32_t actuator_source_masks[8]{0};

    bool satisfies(uint32_t required_sources, uint32_t required_actuators) const {
        return ((active_sources_mask & required_sources) == required_sources) &&
               ((active_actuators_mask & required_actuators) == required_actuators);
    }
};

class CellularOrganism;

struct EvolutionConstraintConfig {
    SkeletonLockMode skeleton_lock{SkeletonLockMode::UNLOCKED}; // 默认自由形态发生 (允许探索新感官与新附肢)
    TypeWhitelistMode type_whitelist{TypeWhitelistMode::FULL_24};
    SeedInitMode seed_mode{SeedInitMode::HANDCRAFTED_PROGENITOR};
    FitnessDriverMode fitness_driver{FitnessDriverMode::TASK_FITNESS_ONLY};
    double novelty_weight{0.3}; // 好奇心奖励权重 alpha
    bool enable_mechanotransduction{true}; // 是否启入力敏转导与皮层沟回自发折叠

    // ── 资源上限守卫与动态代谢平衡 (Resource Bounds & Metabolic Equilibrium) ──
    size_t max_cells_limit{0};             // 细胞规模预算硬上限 (0 = 无上限，完全开放)
    size_t max_synapses_limit{0};          // 突触规模预算硬上限 (0 = 无上限，完全开放)
    bool enable_dynamic_metabolism{true};  // 启用动态代谢能耗调节
    double basal_metabolic_cost{0.00002};  // 细胞单位维持能耗 (热力学软阻尼标定)
    double synaptic_metabolic_cost{0.000005};// 突触单位通信能耗
    double immigrant_rate{0.05};           // 客卿移民比例

    // ── 事务性变异与依赖保护 (Transactional Mutation & Dependency Guard) ──
    bool enable_transaction_rollback{true};// 变异事务原子回滚：编译/预算/契约校验失败时自动回滚
    bool enable_dependency_guard{false};   // 严禁默认开启业务依赖守卫
    std::function<bool(const FunctionalCoverageContract&)> viability_contract_filter{nullptr}; // 通用功能图覆盖度契约过滤钩子
    std::function<bool(const CellularOrganism&)> topology_contract_guard{nullptr}; // 通用外围拓扑契约校验钩子 (底座完全解耦)
    std::function<bool(const CellularOrganism&)> viability_filter{nullptr};        // 通用外围生存力/准入过滤钩子 (例如亏损时不分裂)
    std::function<std::vector<double>(const CellularOrganism&)> behavior_extractor{nullptr}; // 通用外围行为特征提取钩子 (多样性/新颖性搜索)
    double fast_mutation_rate{0.80};       // 参数与局部权重漂移
    double medium_mutation_rate{0.45};     // 局部突触重连
    double slow_mutation_rate{0.35};       // 细胞增殖/力敏重塑

    // ── 鲍德温效应与可塑性跨代固化 (Baldwin Effect & Plasticity Crystallization) ──
    bool enable_baldwin_crystallization{true}; // 启用后天突触学习向先天基因权重的跨代固化
    double crystallization_rate{0.30};         // 突触固化速率
};

inline const char* to_string(SkeletonLockMode m) {
    return (m == SkeletonLockMode::LOCKED) ? "Skeleton_Locked" : "Skeleton_Unlocked";
}

inline const char* to_string(TypeWhitelistMode m) {
    return (m == TypeWhitelistMode::CURATED_9) ? "Curated_9_Primitives" : "Full_24_Primitives";
}

inline const char* to_string(FitnessDriverMode m) {
    switch (m) {
        case FitnessDriverMode::TASK_FITNESS_ONLY: return "Task_Fitness_Only";
        case FitnessDriverMode::NOVELTY_SEARCH: return "Novelty_Search";
        case FitnessDriverMode::HYBRID_CURIOSITY: return "Hybrid_Curiosity";
        default: return "Unknown_Fitness_Driver";
    }
}

// ============================================================================
// 3.5 3D 空间哈希网格 (SpatialHashGrid3D) — O(N) 兰纳-琼斯多体物理与局部邻域索引
// 零 GC、连续内存、无锁哈希链表结构，支持十万至百万级细胞亚毫秒级力场松弛
// ============================================================================
struct SpatialHashGrid3D {
    float cell_size{87.5f}; // 截断半径 r_cut = 2.5 * sigma
    float inv_cell_size{1.0f / 87.5f};
    uint32_t table_size{65536};
    uint32_t mask{65535};
    std::vector<int32_t> head;
    std::vector<int32_t> next;

    void init(size_t max_cells, float r_cut = 87.5f) {
        cell_size = r_cut;
        inv_cell_size = 1.0f / r_cut;
        uint32_t needed = 65536;
        while (needed < max_cells * 2 && needed < (1u << 22)) {
            needed <<= 1;
        }
        table_size = needed;
        mask = table_size - 1;
        head.assign(table_size, -1);
        next.assign(max_cells, -1);
    }

    inline uint32_t hash_coords(int32_t bx, int32_t by, int32_t bz) const {
        return (static_cast<uint32_t>(bx * 73856093) ^ 
                static_cast<uint32_t>(by * 19349663) ^ 
                static_cast<uint32_t>(bz * 83492791)) & mask;
    }

    void build(const std::vector<Cell>& cells) {
        if (next.size() < cells.size()) {
            next.assign(cells.size(), -1);
        }
        if (head.size() != table_size) {
            head.assign(table_size, -1);
        } else {
            std::fill(head.begin(), head.end(), -1);
        }

        for (size_t i = 0; i < cells.size(); ++i) {
            int32_t bx = static_cast<int32_t>(std::floor(cells[i].x * inv_cell_size));
            int32_t by = static_cast<int32_t>(std::floor(cells[i].y * inv_cell_size));
            int32_t bz = static_cast<int32_t>(std::floor(cells[i].z * inv_cell_size));
            uint32_t h = hash_coords(bx, by, bz);
            next[i] = head[h];
            head[h] = static_cast<int32_t>(i);
        }
    }
};

struct OrganismBlueprint {
    std::string lineage_name{"Contract-Progenitor"};
    std::vector<Cell> cells;
    std::vector<Synapse> synapses;
};

// ============================================================================
// 4. 多细胞有机体 (CellularOrganism): 细胞拓扑决策图 (DAG) + 力场物理引擎
// ============================================================================
class CellularOrganism {
public:
    uint64_t organism_id{0};
    uint64_t parent_organism_id{0};
    uint32_t generation{0};
    std::string lineage_name;   // 物种族谱名 (如 "Apex-Predator-V3")
    std::string mutation_receipt; // 突变凭证
    
    std::vector<Cell> cells;
    std::vector<Synapse> synapses;
    
    // 适应度评估 (综合回报、稳定性)
    double fitness_score{0.0};
    double cumulative_reward{0.0};
    double max_drawdown{0.0};
    uint32_t trade_count{0};
    std::unordered_map<std::string, double> custom_metrics; // 外围 Task 自定义业务度量挂载容器 (杜绝向基座增添业务字段)
    
    // 预编译扁平执行序列 (真·零 GC，无哈希表，无堆分配)
    struct CompiledSynapse {
        size_t from_idx;
        size_t to_idx;
        uint8_t to_port;
        double weight;
        double initial_weight;
        double hebbian_rate;
        double hebbian_decay;
        bool   is_recurrent;
    };
    struct CompiledActionCell {
        size_t cell_idx;
        CellType type;
    };
    std::vector<CompiledSynapse> compiled_synapses_;
    std::vector<CompiledActionCell> compiled_actions_;
    std::vector<size_t> execution_order_;
    // 出边 CSR 索引: forward_nd 内层传递由 O(细胞×突触) 降为 O(出度) — 百万级前向的生命线
    std::vector<size_t> out_start_;   // [cell_idx] 起始偏移, 长 N+1
    std::vector<size_t> out_edges_;   // compiled_synapses_ 的出边索引 (按 from_idx 分桶)
    mutable std::vector<double> flat_port_inputs_; // [cell_idx * 2 + port]
    mutable SpatialHashGrid3D spatial_grid_;       // 3D 空间哈希网格 (O(N) 多体力场)
    bool is_compiled_{false};
    bool is_compiled() const { return is_compiled_; }

    CellularOrganism() = default;

    // 回合间状态重置: 清零动态膜电位与记忆, 保留基因组 (参数/拓扑/坐标) 不变。
    // reset_plasticity: 是否将在线学习调整的突触权重重置回初始基因组权重
    void reset_state(bool reset_plasticity = false) {
        for (auto& c : cells) {
            c.state_val = 0.0;
            c.prev_input = 0.0;
            c.latch_state = false;
            c.output_val = 0.0;
            c.prev_output_val = 0.0;
            c.aux_state = 0.0;
            std::fill(std::begin(c.delay_buffer), std::end(c.delay_buffer), 0.0);
            c.delay_idx = 0;
        }
        std::fill(flat_port_inputs_.begin(), flat_port_inputs_.end(), 0.0);

        if (reset_plasticity) {
            for (auto& s : compiled_synapses_) {
                s.weight = s.initial_weight;
            }
            for (auto& s : synapses) {
                s.weight = s.initial_weight;
            }
        }
    }

    // 创建最简单细胞原生生物 (Archean Progenitor / Mode A: Handcrafted Progenitor)
    static CellularOrganism create_seed_organism(uint64_t id = 1) {
        CellularOrganism org;
        org.organism_id = id;
        org.generation = 0;
        org.lineage_name = "Genesis-0";

        // 感知细胞: 0=主输入, 1=辅助输入
        org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -40.0f, 0.0f});
        org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  40.0f, 0.0f});
        
        // 代谢细胞: EMA 滤波
        org.cells.push_back({2, CellType::OP_EMA, 0.05, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -40.0f, -30.0f, 0.0f}); // 慢线
        org.cells.push_back({3, CellType::OP_EMA, 0.20, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -40.0f,  30.0f, 0.0f}); // 快线
        
        // 门控对比细胞: 快线减慢线差值 + 迟滞比较
        org.cells.push_back({4, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 30.0f, 0.0f, 0.0f});
        org.cells.push_back({5, CellType::GATE_HYSTERESIS, 0.01, -0.01, 0.0, 0.0, false, 0.0, 0, 0, 80.0f, 0.0f, 0.0f});

        // 效应动作细胞
        org.cells.push_back({6, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -40.0f, 0.0f}); // 买/加速
        org.cells.push_back({7, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f,  40.0f, 0.0f}); // 卖/制动
        org.cells.push_back({8, CellType::ACT_IMMUNE_BLOCK, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, 100.0f, 0.0f});     // 免疫锁

        // 连接突触
        org.synapses.push_back({0, 2, 0, 1.0, true, 60.0f, -1.0f});
        org.synapses.push_back({0, 3, 0, 1.0, true, 60.0f, -1.0f});
        org.synapses.push_back({3, 4, 0, 1.0, true, 60.0f, -1.0f});
        org.synapses.push_back({2, 4, 1, 1.0, true, 60.0f, -1.0f});
        org.synapses.push_back({4, 5, 0, 1.0, true, 50.0f, -1.0f});
        org.synapses.push_back({5, 6, 0, 1.0, true, 60.0f, -1.0f});
        org.synapses.push_back({5, 7, 0, -1.0, true, 60.0f, -1.0f});

        org.compile();
        return org;
    }

    static CellularOrganism create_handcrafted_progenitor(uint64_t id = 1) {
        return create_seed_organism(id);
    }

    // Mode B: 最小随机图结构 (Minimal Random Graph: 3-4 cells: 2 receptors + 1 random metabolic/gating cell + 1 effector with random weights U(-2, 2))
    static CellularOrganism create_minimal_random_graph(uint64_t id = 1, uint32_t seed = 42) {
        CellularOrganism org;
        org.organism_id = id;
        org.generation = 0;
        org.lineage_name = "MinimalRandom-" + std::to_string(id);

        std::mt19937 prng(seed + static_cast<uint32_t>(id * 1013));
        std::uniform_real_distribution<double> dist_w(-2.0, 2.0);
        std::uniform_real_distribution<double> dist_param(-1.0, 1.0);

        static const CellType candidates[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS,
            CellType::OP_DELAY_N, CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_INHIBIT,
            CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX
        };
        std::uniform_int_distribution<size_t> dist_type(0, sizeof(candidates) / sizeof(candidates[0]) - 1);
        CellType mid_type = candidates[dist_type(prng)];

        // 2 感知受体细胞
        org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -30.0f, 0.0f});
        org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  30.0f, 0.0f});

        // 1 随机代谢/门控细胞
        org.cells.push_back({2, mid_type, dist_param(prng), dist_param(prng), 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});

        // 效应动作细胞 (3-4 cells: 1 积极动作 + 1 消极动作)
        org.cells.push_back({3, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, -30.0f, 0.0f});
        org.cells.push_back({4, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f,  30.0f, 0.0f});

        // 随机突触连接 U(-2, 2)
        std::uniform_int_distribution<int> dist_sense(0, 1);
        uint16_t s_id = static_cast<uint16_t>(dist_sense(prng));
        org.synapses.push_back({s_id, 2, 0, dist_w(prng), true, 60.0f, -1.0f});
        org.synapses.push_back({static_cast<uint16_t>(1 - s_id), 2, 1, dist_w(prng), true, 60.0f, -1.0f});
        org.synapses.push_back({2, 3, 0, dist_w(prng), true, 60.0f, -1.0f});
        org.synapses.push_back({2, 4, 0, dist_w(prng), true, 60.0f, -1.0f});

        org.compile();
        return org;
    }

    // Mode C: 无连接原始胚胎 (Disconnected Embryo: pure receptors and effectors with no intermediate connections at gen-0)
    static CellularOrganism create_disconnected_embryo(uint64_t id = 1) {
        CellularOrganism org;
        org.organism_id = id;
        org.generation = 0;
        org.lineage_name = "Embryo-" + std::to_string(id);

        // 纯感知受体细胞
        org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -60.0f, 0.0f});
        org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -20.0f, 0.0f});
        org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  20.0f, 0.0f});
        org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  60.0f, 0.0f});

        // 纯效应动作细胞
        org.cells.push_back({4, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f, -60.0f, 0.0f});
        org.cells.push_back({5, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f, -20.0f, 0.0f});
        org.cells.push_back({6, CellType::ACT_DEFENSIVE_RESET,  0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f,  20.0f, 0.0f});
        org.cells.push_back({7, CellType::ACT_IMMUNE_BLOCK,     0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f,  60.0f, 0.0f});

        // 突触完全为空 (0 connections at gen-0)
        org.synapses.clear();

        org.compile();
        return org;
    }

    static CellularOrganism create_from_blueprint(const OrganismBlueprint& bp, uint64_t id = 1) {
        CellularOrganism org;
        org.organism_id = id;
        org.generation = 0;
        org.lineage_name = bp.lineage_name;
        org.cells = bp.cells;
        org.synapses = bp.synapses;
        for (auto& s : org.synapses) {
            s.initial_weight = s.weight;
        }
        org.compile();
        return org;
    }

    // 通用契约原基：4 感受器 + 4 效应器 + 7 内部算子/门控
    static CellularOrganism create_contract_progenitor(uint64_t id = 1) {
        OrganismBlueprint bp;
        bp.lineage_name = "Contract-Progenitor";

        bp.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -60.0f, 0.0f});
        bp.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -20.0f, 0.0f});
        bp.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  20.0f, 0.0f});
        bp.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  60.0f, 0.0f});

        bp.cells.push_back({4, CellType::GATE_THRESHOLD, -1.0e9, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -40.0f, 0.0f, 0.0f});
        bp.cells.push_back({5, CellType::OP_SUM, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, -40.0f, 0.0f});
        bp.cells.push_back({6, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -60.0f, 0.0f});
        bp.cells.push_back({7, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -20.0f, 0.0f});
        bp.cells.push_back({8, CellType::ACT_DEFENSIVE_RESET, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f,  20.0f, 0.0f});
        bp.cells.push_back({9, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 40.0f, 0.0f});
        bp.cells.push_back({10, CellType::GATE_THRESHOLD, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 40.0f, 0.0f});
        bp.cells.push_back({11, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 80.0f, 0.0f});
        bp.cells.push_back({12, CellType::GATE_THRESHOLD, 3.5, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 80.0f, 0.0f});
        bp.cells.push_back({13, CellType::OP_SUM, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, 60.0f, 0.0f});
        bp.cells.push_back({14, CellType::ACT_IMMUNE_BLOCK, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, 100.0f, 0.0f});

        bp.synapses.push_back({0, 4, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({0, 5, 0, 0.35, true, 60.0f, -1.0f});
        bp.synapses.push_back({1, 5, 1, 0.95, true, 60.0f, -1.0f});
        bp.synapses.push_back({4, 5, 0, -5.25, true, 60.0f, -1.0f});
        bp.synapses.push_back({5, 6, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({2, 8, 0, 0.45, true, 60.0f, -1.0f});
        bp.synapses.push_back({4, 9, 0, 2.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({3, 9, 1, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({9, 10, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({1, 11, 1, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({11, 12, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({10, 13, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({12, 13, 1, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({13, 14, 0, 1.0, true, 60.0f, -1.0f});
        bp.synapses.push_back({13, 7, 0, -1.0, true, 60.0f, -1.0f});

        for (auto& s : bp.synapses) {
            s.initial_weight = s.weight;
            s.hebbian_rate = 0.0;
        }

        return create_from_blueprint(bp, id);
    }

    // 通用模式工厂分发器
    static CellularOrganism create_by_mode(SeedInitMode mode, uint64_t id = 1, uint32_t seed = 42) {
        switch (mode) {
            case SeedInitMode::HANDCRAFTED_PROGENITOR:
                return create_handcrafted_progenitor(id);
            case SeedInitMode::MINIMAL_RANDOM_GRAPH:
                return create_minimal_random_graph(id, seed);
            case SeedInitMode::DISCONNECTED_EMBRYO:
                return create_disconnected_embryo(id);
            case SeedInitMode::CONTRACT_PROGENITOR:
                return create_contract_progenitor(id);
        }
        return create_seed_organism(id);
    }

    // 拓扑排序与扁平执行编译 (Flat Array Compilation: 消除所有运行期堆分配)
    bool compile() {
        if (cells.empty()) {
            compiled_synapses_.clear();
            compiled_actions_.clear();
            execution_order_.clear();
            out_start_.clear();
            out_edges_.clear();
            is_compiled_ = true;
            return true;
        }

        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < cells.size(); ++i) {
            id_to_idx[cells[i].id] = i;
        }

        std::unordered_map<uint32_t, int> in_degrees;
        std::unordered_map<uint32_t, std::vector<uint32_t>> adj;
        std::unordered_map<uint32_t, std::vector<uint32_t>> rev_adj;
        for (const auto& c : cells) in_degrees[c.id] = 0;

        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            auto it_from = id_to_idx.find(syn.from_cell_id);
            auto it_to = id_to_idx.find(syn.to_cell_id);
            if (it_from != id_to_idx.end() && it_to != id_to_idx.end()) {
                adj[syn.from_cell_id].push_back(syn.to_cell_id);
                rev_adj[syn.to_cell_id].push_back(syn.from_cell_id);
                in_degrees[syn.to_cell_id]++;
            }
        }

        // 识别动作细胞与预测受体细胞
        compiled_actions_.clear();
        std::unordered_set<uint32_t> active_cell_ids;
        std::vector<uint32_t> rev_queue;

        for (size_t i = 0; i < cells.size(); ++i) {
            if (is_effector_cell(cells[i].type) ||
                cells[i].type == CellType::PREDICT_SENSE_0 ||
                cells[i].type == CellType::PREDICT_SENSE_1) {
                compiled_actions_.push_back({i, cells[i].type});
                active_cell_ids.insert(cells[i].id);
                rev_queue.push_back(cells[i].id);
            }
        }

        // 若暂无动作细胞（如刚初始化的胚胎），则默认全细胞保留
        if (compiled_actions_.empty()) {
            for (const auto& c : cells) active_cell_ids.insert(c.id);
        } else {
            // 反向活性分析 (从动作细胞反向追溯所有有贡献的细胞)
            size_t rev_head = 0;
            while (rev_head < rev_queue.size()) {
                uint32_t curr = rev_queue[rev_head++];
                auto it = rev_adj.find(curr);
                if (it != rev_adj.end()) {
                    for (uint32_t parent : it->second) {
                        if (active_cell_ids.insert(parent).second) {
                            rev_queue.push_back(parent);
                        }
                    }
                }
            }
        }

        // 拓扑排序 (Kahn's Algorithm)
        std::vector<uint32_t> queue;
        for (const auto& c : cells) {
            if (in_degrees[c.id] == 0) {
                queue.push_back(c.id);
            }
        }

        execution_order_.clear();
        size_t head = 0;
        while (head < queue.size()) {
            uint32_t u = queue[head++];
            if (active_cell_ids.find(u) != active_cell_ids.end()) {
                execution_order_.push_back(id_to_idx[u]);
            }

            for (uint32_t v : adj[u]) {
                in_degrees[v]--;
                if (in_degrees[v] == 0) {
                    queue.push_back(v);
                }
            }
        }

        // 环保护补齐
        if (execution_order_.size() < active_cell_ids.size()) {
            std::unordered_set<size_t> visited(execution_order_.begin(), execution_order_.end());
            for (uint32_t cid : active_cell_ids) {
                size_t idx = id_to_idx[cid];
                if (visited.find(idx) == visited.end()) {
                    execution_order_.push_back(idx);
                }
            }
        }

        // 建立执行顺序位阶索引 (用于一等公民循环连接与前向/反馈突触判定)
        std::vector<size_t> cell_rank(cells.size(), 999999);
        for (size_t r = 0; r < execution_order_.size(); ++r) {
            cell_rank[execution_order_[r]] = r;
        }

        // 编译有效突触 (自动区分前向突触与递归反馈突触)
        compiled_synapses_.clear();
        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            if (active_cell_ids.find(syn.from_cell_id) == active_cell_ids.end() ||
                active_cell_ids.find(syn.to_cell_id) == active_cell_ids.end()) {
                continue;
            }
            auto it_from = id_to_idx.find(syn.from_cell_id);
            auto it_to = id_to_idx.find(syn.to_cell_id);
            if (it_from != id_to_idx.end() && it_to != id_to_idx.end()) {
                size_t f_idx = it_from->second;
                size_t t_idx = it_to->second;
                bool is_loop = (cell_rank[f_idx] >= cell_rank[t_idx]); // 发送端位阶>=接收端位阶即为递归时序环

                compiled_synapses_.push_back({
                    f_idx, t_idx, syn.to_port,
                    syn.weight, syn.weight,
                    syn.hebbian_rate, syn.hebbian_decay,
                    is_loop
                });
            }
        }

        // 预分配扁平输入端口缓冲 (每个细胞 2 个端口)
        flat_port_inputs_.assign(cells.size() * 2, 0.0);

        // 构建出边 CSR 索引 (按 from_idx 分桶, 编译期一次, 前向 O(出度))
        out_start_.assign(cells.size() + 1, 0);
        for (const auto& s : compiled_synapses_) {
            out_start_[s.from_idx + 1]++;
        }
        for (size_t i = 0; i < cells.size(); ++i) {
            out_start_[i + 1] += out_start_[i];
        }
        out_edges_.resize(compiled_synapses_.size());
        {
            std::vector<size_t> cursor(out_start_.begin(), out_start_.end() - 1);
            for (size_t e = 0; e < compiled_synapses_.size(); ++e) {
                out_edges_[cursor[compiled_synapses_[e].from_idx]++] = e;
            }
        }

        is_compiled_ = true;
        return true;
    }

    // 胚胎分形发育生长函数 (从种子细胞快速发育扩增至 target_cells 规模，受 max_limit 预算硬上限保护)
    void develop_to_scale(size_t target_cells, size_t max_limit = 10000000) {
        target_cells = std::min(target_cells, max_limit);
        if (target_cells <= cells.size()) return;
        size_t needed = target_cells - cells.size();
        uint32_t current_id = static_cast<uint32_t>(cells.size());

        static const std::vector<CellType> pool = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS, CellType::OP_DELAY_N,
            CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_INHIBIT, CellType::GATE_DEADZONE,
            CellType::GATE_MIN_MAX, CellType::ASSOCIATION_HUB
        };

        const float sigma = 35.0f;
        float space_span = std::max(300.0f, sigma * std::cbrt(static_cast<float>(target_cells)) * 1.8f);
        size_t num_clusters = std::max<size_t>(1, target_cells / 100);
        size_t cells_per_cluster = std::max<size_t>(1, needed / num_clusters);

        cells.reserve(target_cells);
        synapses.reserve(target_cells * 2);

        std::mt19937 rng(1337 + static_cast<uint32_t>(organism_id));
        std::uniform_real_distribution<float> dist_uniform(-1.0f, 1.0f);
        std::normal_distribution<float> dist_gauss(0.0f, sigma * 0.8f);

        for (size_t c_idx = 0; c_idx < num_clusters && cells.size() < target_cells; ++c_idx) {
            float layer_t = static_cast<float>(c_idx + 0.5f) / static_cast<float>(num_clusters);
            float cx = -space_span * 0.45f + space_span * 0.9f * layer_t;
            float cy = dist_uniform(rng) * space_span * 0.4f;
            float cz = dist_uniform(rng) * space_span * 0.3f;

            for (size_t k = 0; k < cells_per_cluster && cells.size() < target_cells; ++k) {
                Cell new_c;
                new_c.id = current_id;
                new_c.type = pool[rng() % pool.size()];
                new_c.x = cx + dist_gauss(rng);
                new_c.y = cy + dist_gauss(rng);
                new_c.z = cz + dist_gauss(rng) * 0.75f;
                new_c.param1 = 0.1f + std::abs(dist_uniform(rng)) * 0.4f;
                new_c.param2 = dist_uniform(rng) * 0.5f;

                cells.push_back(new_c);

                if (current_id > 9) {
                    uint32_t offset = 1 + (rng() % std::min<uint32_t>(15, current_id));
                    uint32_t prev_id = (current_id >= offset) ? (current_id - offset) : 0;
                    
                    Synapse syn;
                    syn.from_cell_id = prev_id;
                    syn.to_cell_id = current_id;
                    syn.to_port = 0;
                    syn.weight = 0.01 + std::abs(dist_uniform(rng)) * 0.08;
                    syn.initial_weight = syn.weight;
                    syn.is_active = true;
                    synapses.push_back(syn);
                }

                current_id++;
            }
        }

        for (auto& c : cells) {
            if (c.type == CellType::ACT_PRIMARY_POSITIVE || c.type == CellType::ACT_PRIMARY_NEGATIVE) {
                c.x = space_span * 0.55f;
            }
        }

        is_compiled_ = false;
    }

    // ========================================================================
    // 4.5 宽契约供给 (N-Channel Wide Contract)
    //
    // 1) ensure_receptors: 保证 SENSE_CHANNEL 感受器覆盖 [0, n_channels) 通道,
    //    新增感受器以小权重随机接入既有核心网络 (为演化提供接线素材)。
    // 2) wire_global_bridge: 全细胞范围随机铺设长程突触, 让发育生长的新基质
    //    真正参与 感受→代谢→效应 通路 —— 否则百万细胞只是挂在核心旁边的死重
    //    (develop_to_scale 的局部前驱链接构性上无法被远处任务信号触达)。
    // ========================================================================
    void ensure_receptors(size_t n_channels, uint32_t seed = 7, size_t fan_out = 8) {
        std::vector<char> have(n_channels, 0);
        for (const auto& c : cells) {
            if (c.type == CellType::SENSE_CHANNEL ||
                c.type == CellType::SENSE_RAW_INPUT_0 || c.type == CellType::SENSE_RAW_INPUT_1 ||
                c.type == CellType::SENSE_RAW_INPUT_2 || c.type == CellType::SENSE_RAW_INPUT_3) {
                const size_t ch = receptor_channel_index(c.type, c.param2);
                if (ch < n_channels) have[ch] = 1;
            }
        }

        std::vector<uint32_t> internals;
        for (size_t i = 0; i < cells.size(); ++i) {
            const CellType t = cells[i].type;
            const bool is_sense = (t == CellType::SENSE_CHANNEL || t == CellType::SENSE_RAW_INPUT_0 ||
                                   t == CellType::SENSE_RAW_INPUT_1 || t == CellType::SENSE_RAW_INPUT_2 ||
                                   t == CellType::SENSE_RAW_INPUT_3);
            if (!is_sense) internals.push_back(static_cast<uint32_t>(i));
        }
        if (internals.empty()) return;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> pick(0, internals.size() - 1);
        std::uniform_real_distribution<double> w(-0.4, 0.4);
        uint32_t next_id = 0;
        for (const auto& c : cells) next_id = std::max(next_id, c.id);
        ++next_id;

        for (size_t ch = 0; ch < n_channels; ++ch) {
            if (have[ch]) continue;
            Cell nc;
            nc.id = next_id++;
            nc.type = CellType::SENSE_CHANNEL;
            nc.param1 = 1.0;
            nc.param2 = static_cast<double>(ch);
            nc.x = -180.0f;
            nc.y = static_cast<float>(ch) * 8.0f - 120.0f;
            cells.push_back(nc);
            for (size_t k = 0; k < fan_out; ++k) {
                Synapse syn;
                syn.from_cell_id = nc.id;
                syn.to_cell_id = internals[pick(rng)];
                syn.to_port = 0;
                syn.weight = w(rng);
                syn.initial_weight = syn.weight;
                syn.is_active = true;
                synapses.push_back(syn);
            }
        }
        is_compiled_ = false;
    }

    void wire_global_bridge(size_t n_edges, uint32_t seed = 11) {
        if (cells.size() < 2 || n_edges == 0) return;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(cells.size() - 1));
        std::uniform_real_distribution<double> w(-0.05, 0.05);  // 小权重: 桥接不淹没已训练通路
        synapses.reserve(synapses.size() + n_edges);
        for (size_t e = 0; e < n_edges; ++e) {
            const uint32_t a = pick(rng), b = pick(rng);
            if (a == b) continue;
            Synapse syn;
            syn.from_cell_id = a;
            syn.to_cell_id = b;
            syn.to_port = (e % 8 == 0) ? 1 : 0;
            syn.weight = w(rng);
            syn.initial_weight = syn.weight;
            syn.is_active = true;
            synapses.push_back(syn);
        }
        is_compiled_ = false;
    }

    // ========================================================================
    // 5. 严格 12-6 兰纳-琼斯势能与力场微物理引擎 (Lennard-Jones 12-6 Potential Engine)
    // V_LJ(r) = 4*epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
    // F_LJ(r) = -grad V = (24*epsilon / r^2) * [ 2*(sigma/r)^12 - (sigma/r)^6 ] * r_vec
    // 支持 3D 空间哈希网格加速 (Spatial Hashing)，将多体相互作用压至严格 O(N)
    // ========================================================================
    void step_force_field_physics(float dt = 0.016f) {
        const float epsilon = 15.0f;       // 势阱深度 (Potential Well Depth)
        const float sigma = 35.0f;         // 零势平衡距离 (Collision Diameter)
        const float sigma6 = std::pow(sigma, 6.0f);
        const float sigma12 = sigma6 * sigma6;
        const float r_cut = 2.5f * sigma;  // 截断半径 (Cutoff Radius)
        const float r_cut_sq = r_cut * r_cut;
        const float k_spring = 0.05f;      // 突触结构弹簧系数
        const float damping = 0.85f;       // 黏性阻尼

        // 1. 重置合力并衰减发光电位
        for (auto& c : cells) {
            c.fx = 0.0f; c.fy = 0.0f; c.fz = 0.0f;
            c.glow_charge *= 0.92f;
        }

        // 2. 严格 12-6 兰纳-琼斯多体非键结势能力场 (3D 近斥中吸远无)
        const size_t num_cells = cells.size();
        if (num_cells <= 64) {
            for (size_t i = 0; i < num_cells; ++i) {
                auto& ci = cells[i];
                for (size_t j = i + 1; j < num_cells; ++j) {
                    auto& cj = cells[j];
                    float dx = cj.x - ci.x;
                    float dy = cj.y - ci.y;
                    float dz = cj.z - ci.z;
                    float dist_sq = dx * dx + dy * dy + dz * dz + 1e-4f;

                    if (dist_sq < r_cut_sq && dist_sq > 1.0f) {
                        float dist = std::sqrt(dist_sq);
                        float r2 = dist_sq;
                        float r6 = r2 * r2 * r2;
                        float r12 = r6 * r6;
                        float sr6 = sigma6 / r6;
                        float sr12 = sigma12 / r12;
                        float f_mag = (24.0f * epsilon / r2) * (2.0f * sr12 - sr6);
                        f_mag = std::clamp(f_mag, -50.0f, 300.0f);
                        float inv_dist = f_mag / dist;

                        ci.fx -= dx * inv_dist;
                        ci.fy -= dy * inv_dist;
                        ci.fz -= dz * inv_dist;
                        cj.fx += dx * inv_dist;
                        cj.fy += dy * inv_dist;
                        cj.fz += dz * inv_dist;
                    }
                }
            }
        } else {
            if (spatial_grid_.head.empty() || spatial_grid_.next.size() < num_cells) {
                spatial_grid_.init(num_cells, r_cut);
            }
            spatial_grid_.build(cells);

            static const int32_t neighbor_offsets[14][3] = {
                {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                {1, 1, 0}, {1, -1, 0}, {1, 0, 1}, {1, 0, -1},
                {0, 1, 1}, {0, 1, -1},
                {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}
            };

            for (size_t i = 0; i < num_cells; ++i) {
                auto& ci = cells[i];
                int32_t bx = static_cast<int32_t>(std::floor(ci.x * spatial_grid_.inv_cell_size));
                int32_t by = static_cast<int32_t>(std::floor(ci.y * spatial_grid_.inv_cell_size));
                int32_t bz = static_cast<int32_t>(std::floor(ci.z * spatial_grid_.inv_cell_size));

                for (int off = 0; off < 14; ++off) {
                    int32_t nbx = bx + neighbor_offsets[off][0];
                    int32_t nby = by + neighbor_offsets[off][1];
                    int32_t nbz = bz + neighbor_offsets[off][2];
                    uint32_t h = spatial_grid_.hash_coords(nbx, nby, nbz);

                    int32_t j = spatial_grid_.head[h];
                    while (j != -1) {
                        if (static_cast<size_t>(j) > i) {
                            auto& cj = cells[j];
                            float dx = cj.x - ci.x;
                            float dy = cj.y - ci.y;
                            float dz = cj.z - ci.z;
                            float dist_sq = dx * dx + dy * dy + dz * dz + 1e-4f;

                            if (dist_sq < r_cut_sq && dist_sq > 1.0f) {
                                float dist = std::sqrt(dist_sq);
                                float r2 = dist_sq;
                                float r6 = r2 * r2 * r2;
                                float r12 = r6 * r6;
                                float sr6 = sigma6 / r6;
                                float sr12 = sigma12 / r12;
                                float f_mag = (24.0f * epsilon / r2) * (2.0f * sr12 - sr6);
                                f_mag = std::clamp(f_mag, -50.0f, 300.0f);
                                float inv_dist = f_mag / dist;

                                ci.fx -= dx * inv_dist;
                                ci.fy -= dy * inv_dist;
                                ci.fz -= dz * inv_dist;
                                cj.fx += dx * inv_dist;
                                cj.fy += dy * inv_dist;
                                cj.fz += dz * inv_dist;
                            }
                        }
                        j = spatial_grid_.next[j];
                    }
                }
            }
        }

        // 3. 突触弹簧引力场计算 (有向结构张力: 零堆分配，直取 compiled_synapses_)
        if (!is_compiled_) compile();
        for (size_t k = 0; k < compiled_synapses_.size(); ++k) {
            const auto& syn = compiled_synapses_[k];
            if (syn.from_idx >= cells.size() || syn.to_idx >= cells.size()) continue;
            auto& c1 = cells[syn.from_idx];
            auto& c2 = cells[syn.to_idx];

            float dx = c2.x - c1.x;
            float dy = c2.y - c1.y;
            float dz = c2.z - c1.z;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-4f);

            float delta = dist - 60.0f;
            float spring_f = k_spring * delta;

            float nx = dx / dist;
            float ny = dy / dist;
            float nz = dz / dist;

            c1.fx += nx * spring_f;
            c1.fy += ny * spring_f;
            c1.fz += nz * spring_f;

            c2.fx -= nx * spring_f;
            c2.fy -= ny * spring_f;
            c2.fz -= nz * spring_f;
        }

        for (auto& syn : synapses) {
            // 推进神经放电光子动画
            if (syn.photon_pos >= 0.0f) {
                syn.photon_pos += dt * 3.0f;
                if (syn.photon_pos > 1.0f) syn.photon_pos = -1.0f;
            }
        }

        // 4. 牛顿第二定律积分推进 (Verlet Velocity Step) 与力敏张力累积
        for (auto& c : cells) {
            float force_mag = std::sqrt(c.fx * c.fx + c.fy * c.fy + c.fz * c.fz);
            c.physical_stress = 0.80f * c.physical_stress + 0.20f * force_mag;
            if (c.mitosis_cooldown > 0) c.mitosis_cooldown--;

            c.vx = (c.vx + c.fx * dt) * damping;
            c.vy = (c.vy + c.fy * dt) * damping;
            c.vz = (c.vz + c.fz * dt) * damping;

            c.x += c.vx * dt;
            c.y += c.vy * dt;
            c.z += c.vz * dt;
        }
    }

    // ========================================================================
    // 计算 3D 大脑皮层沟回折叠指数 (Gyrification Index)
    // 衡量大脑从 2D 种子平坦表面向 3D 空间立体折叠的复杂度与受体面积膨胀率
    // ========================================================================
    double compute_cortical_folding_index() const {
        if (cells.size() <= 9) return 1.0;
        double sum_z = 0.0;
        double min_z = 1e9, max_z = -1e9;
        for (const auto& c : cells) {
            sum_z += c.z;
            min_z = std::min(min_z, static_cast<double>(c.z));
            max_z = std::max(max_z, static_cast<double>(c.z));
        }
        double mean_z = sum_z / cells.size();
        double var_z = 0.0;
        for (const auto& c : cells) {
            var_z += (c.z - mean_z) * (c.z - mean_z);
        }
        var_z /= cells.size();
        double height = max_z - min_z;
        return 1.0 + (std::sqrt(var_z) / 12.0) + (height / 35.0);
    }

    // 力学分相：有突触对的平均距离 vs 全体细胞对。>1 表示弹簧把耦合细胞拉成微柱。
    double compute_mechanical_demixing_ratio() const {
        if (cells.size() < 2) return 1.0;
        auto cell_xyz = [&](uint32_t id, float& x, float& y, float& z) -> bool {
            for (const auto& c : cells) {
                if (c.id == id) { x = c.x; y = c.y; z = c.z; return true; }
            }
            return false;
        };
        double syn_sum = 0.0;
        size_t syn_n = 0;
        for (const auto& s : synapses) {
            if (!s.is_active) continue;
            float x1, y1, z1, x2, y2, z2;
            if (!cell_xyz(s.from_cell_id, x1, y1, z1) || !cell_xyz(s.to_cell_id, x2, y2, z2)) continue;
            double dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
            syn_sum += std::sqrt(dx * dx + dy * dy + dz * dz);
            syn_n++;
        }
        double pair_sum = 0.0;
        size_t pair_n = 0;
        for (size_t i = 0; i < cells.size(); ++i) {
            for (size_t j = i + 1; j < cells.size(); ++j) {
                double dx = cells[i].x - cells[j].x;
                double dy = cells[i].y - cells[j].y;
                double dz = cells[i].z - cells[j].z;
                pair_sum += std::sqrt(dx * dx + dy * dy + dz * dz);
                pair_n++;
            }
        }
        if (syn_n == 0 || pair_n == 0) return 1.0;
        double mean_syn = syn_sum / static_cast<double>(syn_n);
        double mean_pair = pair_sum / static_cast<double>(pair_n);
        if (mean_syn < 1e-6) return mean_pair / 1e-6;
        return mean_pair / mean_syn;
    }

    using FunctionalCoverageContract = kun::FunctionalCoverageContract;

    // 通用功能图拓扑覆盖度分析 (Derive functional coverage from DAG reachability)
    FunctionalCoverageContract evaluate_coverage_contract() const {
        FunctionalCoverageContract result;
        std::unordered_map<uint32_t, size_t> id_to_index;
        std::vector<std::vector<size_t>> outgoing(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            id_to_index[cells[i].id] = i;
        }
        for (const auto& syn : synapses) {
            if (!syn.is_active || syn.is_recurrent) continue;
            auto from = id_to_index.find(syn.from_cell_id);
            auto to = id_to_index.find(syn.to_cell_id);
            if (from != id_to_index.end() && to != id_to_index.end()) {
                outgoing[from->second].push_back(to->second);
            }
        }

        std::vector<uint32_t> source_mask(cells.size(), 0);
        std::vector<size_t> queue;
        for (size_t i = 0; i < cells.size(); ++i) {
            uint32_t mask = 0;
            switch (cells[i].type) {
                case CellType::SENSE_RAW_INPUT_0: mask = 1u << 0; break;
                case CellType::SENSE_RAW_INPUT_1: mask = 1u << 1; break;
                case CellType::SENSE_RAW_INPUT_2: mask = 1u << 2; break;
                case CellType::SENSE_RAW_INPUT_3: mask = 1u << 3; break;
                case CellType::SENSE_CHANNEL: {
                    const size_t ch = receptor_channel_index(cells[i].type, cells[i].param2) & 31u;
                    mask = 1u << ch;
                    break;
                }
                default: break;
            }
            if (mask != 0) {
                source_mask[i] = mask;
                queue.push_back(i);
            }
        }

        for (size_t head = 0; head < queue.size(); ++head) {
            const size_t current = queue[head];
            for (size_t next : outgoing[current]) {
                const uint32_t merged = source_mask[next] | source_mask[current];
                if (merged != source_mask[next]) {
                    source_mask[next] = merged;
                    queue.push_back(next);
                }
            }
        }

        for (size_t i = 0; i < cells.size(); ++i) {
            if (source_mask[i] == 0) continue;
            ++result.reachable_cells;
            result.active_sources_mask |= source_mask[i];
            switch (cells[i].type) {
                case CellType::ACT_PRIMARY_POSITIVE:
                    result.active_actuators_mask |= (1u << 0);
                    result.actuator_source_masks[0] |= source_mask[i];
                    break;
                case CellType::ACT_PRIMARY_NEGATIVE:
                    result.active_actuators_mask |= (1u << 1);
                    result.actuator_source_masks[1] |= source_mask[i];
                    break;
                case CellType::ACT_DEFENSIVE_RESET:
                    result.active_actuators_mask |= (1u << 2);
                    result.actuator_source_masks[2] |= source_mask[i];
                    break;
                case CellType::ACT_IMMUNE_BLOCK:
                    result.active_actuators_mask |= (1u << 3);
                    result.actuator_source_masks[3] |= source_mask[i];
                    break;
                case CellType::ACT_CHANNEL: {
                    const size_t slot = static_cast<size_t>(std::clamp(std::floor(cells[i].param2), 0.0, 7.0));
                    result.active_actuators_mask |= (1u << slot);
                    result.actuator_source_masks[slot] |= source_mask[i];
                    break;
                }
                default:
                    break;
            }
        }
        return result;
    }

    // ── 规范化功能核心图 (Canonical Functional Core Graph) 与 Weisfeiler-Lehman (WL) 哈希 ──
    struct CanonicalCoreGraph {
        size_t core_node_count{0};
        size_t core_edge_count{0};
        uint64_t wl_hash{0};
        std::vector<uint32_t> core_cell_ids;
    };

    CanonicalCoreGraph extract_canonical_core_graph() const {
        CanonicalCoreGraph core;
        if (cells.empty()) return core;

        std::unordered_set<size_t> sensor_indices;
        std::unordered_set<size_t> actuator_indices;
        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < cells.size(); ++i) {
            id_to_idx[cells[i].id] = i;
            if (is_receptor_cell(cells[i].type)) {
                sensor_indices.insert(i);
            }
            if (cells[i].type == CellType::ACT_PRIMARY_POSITIVE ||
                cells[i].type == CellType::ACT_PRIMARY_NEGATIVE ||
                cells[i].type == CellType::ACT_DEFENSIVE_RESET ||
                cells[i].type == CellType::ACT_IMMUNE_BLOCK) {
                actuator_indices.insert(i);
            }
        }

        std::vector<std::vector<size_t>> fwd(cells.size()), bwd(cells.size());
        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            auto it_f = id_to_idx.find(syn.from_cell_id);
            auto it_t = id_to_idx.find(syn.to_cell_id);
            if (it_f != id_to_idx.end() && it_t != id_to_idx.end()) {
                fwd[it_f->second].push_back(it_t->second);
                bwd[it_t->second].push_back(it_f->second);
            }
        }

        std::vector<bool> reachable_from_sensor(cells.size(), false);
        std::vector<size_t> q;
        for (size_t s : sensor_indices) {
            reachable_from_sensor[s] = true;
            q.push_back(s);
        }
        for (size_t h = 0; h < q.size(); ++h) {
            size_t u = q[h];
            for (size_t v : fwd[u]) {
                if (!reachable_from_sensor[v]) {
                    reachable_from_sensor[v] = true;
                    q.push_back(v);
                }
            }
        }

        std::vector<bool> can_reach_actuator(cells.size(), false);
        q.clear();
        for (size_t a : actuator_indices) {
            can_reach_actuator[a] = true;
            q.push_back(a);
        }
        for (size_t h = 0; h < q.size(); ++h) {
            size_t u = q[h];
            for (size_t v : bwd[u]) {
                if (!can_reach_actuator[v]) {
                    can_reach_actuator[v] = true;
                    q.push_back(v);
                }
            }
        }

        std::vector<size_t> core_indices;
        std::vector<int> cell_to_core_map(cells.size(), -1);
        for (size_t i = 0; i < cells.size(); ++i) {
            if (reachable_from_sensor[i] && can_reach_actuator[i]) {
                cell_to_core_map[i] = static_cast<int>(core_indices.size());
                core_indices.push_back(i);
                core.core_cell_ids.push_back(cells[i].id);
            }
        }

        core.core_node_count = core_indices.size();
        if (core.core_node_count == 0) return core;

        struct CoreEdge {
            size_t from_core_idx;
            size_t to_core_idx;
            uint8_t to_port;
            int quant_weight;
        };
        std::vector<CoreEdge> core_edges;
        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            auto it_f = id_to_idx.find(syn.from_cell_id);
            auto it_t = id_to_idx.find(syn.to_cell_id);
            if (it_f != id_to_idx.end() && it_t != id_to_idx.end()) {
                int c_from = cell_to_core_map[it_f->second];
                int c_to = cell_to_core_map[it_t->second];
                if (c_from >= 0 && c_to >= 0) {
                    int qw = static_cast<int>(std::round(syn.weight * 20.0));
                    core_edges.push_back({static_cast<size_t>(c_from), static_cast<size_t>(c_to), syn.to_port, qw});
                }
            }
        }
        core.core_edge_count = core_edges.size();

        std::vector<uint64_t> colors(core.core_node_count);
        for (size_t k = 0; k < core.core_node_count; ++k) {
            size_t orig_idx = core_indices[k];
            const auto& c = cells[orig_idx];
            uint64_t type_val = static_cast<uint64_t>(c.type);
            int64_t p1_quant = static_cast<int64_t>(std::round(c.param1 * 10.0));
            uint64_t h = 14695981039346656037ULL;
            h = (h ^ type_val) * 1099511628211ULL;
            h = (h ^ static_cast<uint64_t>(p1_quant)) * 1099511628211ULL;
            colors[k] = h;
        }

        for (int round = 0; round < 3; ++round) {
            std::vector<uint64_t> next_colors(core.core_node_count);
            for (size_t k = 0; k < core.core_node_count; ++k) {
                std::vector<uint64_t> neighbor_hashes;
                for (const auto& edge : core_edges) {
                    if (edge.to_core_idx == k) {
                        uint64_t nh = colors[edge.from_core_idx];
                        nh = (nh ^ static_cast<uint64_t>(edge.to_port)) * 1099511628211ULL;
                        nh = (nh ^ static_cast<uint64_t>(edge.quant_weight)) * 1099511628211ULL;
                        neighbor_hashes.push_back(nh);
                    }
                }
                std::sort(neighbor_hashes.begin(), neighbor_hashes.end());
                uint64_t h = colors[k];
                for (uint64_t nh : neighbor_hashes) {
                    h = (h ^ nh) * 1099511628211ULL;
                }
                next_colors[k] = h;
            }
            colors = std::move(next_colors);
        }

        std::sort(colors.begin(), colors.end());
        uint64_t graph_hash = 14695981039346656037ULL;
        for (uint64_t c : colors) {
            graph_hash = (graph_hash ^ c) * 1099511628211ULL;
        }
        core.wl_hash = graph_hash;
        return core;
    }

    static size_t compute_core_graph_edit_distance(const CellularOrganism& a, const CellularOrganism& b) {
        auto ca = a.extract_canonical_core_graph();
        auto cb = b.extract_canonical_core_graph();
        if (ca.wl_hash == cb.wl_hash) return 0;

        // 1. 真实节点类型多重集直方图匹配 (Bipartite Vertex Label Substitution Cost)
        std::unordered_map<int, int> type_hist_a, type_hist_b;
        for (uint32_t id : ca.core_cell_ids) {
            for (const auto& c : a.cells) {
                if (c.id == id) { type_hist_a[static_cast<int>(c.type)]++; break; }
            }
        }
        for (uint32_t id : cb.core_cell_ids) {
            for (const auto& c : b.cells) {
                if (c.id == id) { type_hist_b[static_cast<int>(c.type)]++; break; }
            }
        }

        size_t label_cost = 0;
        std::unordered_set<int> all_types;
        for (const auto& kv : type_hist_a) all_types.insert(kv.first);
        for (const auto& kv : type_hist_b) all_types.insert(kv.first);
        for (int t : all_types) {
            int cnt_a = type_hist_a.count(t) ? type_hist_a[t] : 0;
            int cnt_b = type_hist_b.count(t) ? type_hist_b[t] : 0;
            label_cost += std::abs(cnt_a - cnt_b);
        }

        // 2. 真实边拓扑增删代价 (Edge Insertion/Deletion Cost)
        size_t edge_diff = (ca.core_edge_count > cb.core_edge_count) ? 
                           (ca.core_edge_count - cb.core_edge_count) : 
                           (cb.core_edge_count - ca.core_edge_count);

        size_t ged = label_cost + edge_diff;
        if (ged == 0 && ca.wl_hash != cb.wl_hash) {
            ged = 1;
        }
        return ged;
    }

    // 更新神经元信息惊奇度应变 (Prediction Error Strain)
    void update_informational_strain(double error_loss) {
        float abs_err = static_cast<float>(std::abs(error_loss));
        for (size_t i = 4; i < cells.size(); ++i) {
            float out_mag = static_cast<float>(std::abs(cells[i].output_val));
            cells[i].informational_strain = 0.80f * cells[i].informational_strain + 0.20f * (abs_err * out_mag);
        }
    }

    // 运行时代谢传导前向计算 (Forward Pass: 真·零 GC，纯连续数组计算)
    struct ActionOutputs {
        double positive_action{0.0}; // 买入/加速/左偏
        double negative_action{0.0}; // 卖出/制动/右偏
        double defensive_reset{0.0}; // 清仓/居中
        bool   immune_lock{false};   // 免疫熔断

        // === 预测编码与概念吸引子输出 (Predictive World Model & Thought Dynamics) ===
        double predicted_sense_0{0.0}; // 内部对输入0的前瞻预测
        double predicted_sense_1{0.0}; // 内部对输入1的前瞻预测
        double prediction_error{0.0};  // 预测惊奇度 (Surprise)
        double thought_energy{0.0};    // 内部神经节相空间总动能
        const char* thought_mode{"STABLE_ATTRACTOR"}; // "EXPLORATION", "FOCUS", "SURPRISE", "STABLE_ATTRACTOR"
    };

    ActionOutputs forward(const double inputs[4], bool enable_hebbian = true) {
        return forward_nd(inputs, 4, enable_hebbian);
    }

    ActionOutputs forward_nd(const double* inputs, size_t in_dim, bool enable_hebbian = true) {
        if (!is_compiled_) compile();
        if (inputs == nullptr) in_dim = 0;

        // 1. 清空扁平输入端口缓冲 (连续内存快速清零)
        double* __restrict port_ptr = flat_port_inputs_.data();
        std::memset(port_ptr, 0, flat_port_inputs_.size() * sizeof(double));

        Cell* __restrict cells_ptr = cells.data();
        auto* __restrict syn_ptr = compiled_synapses_.data();
        const size_t num_synapses = compiled_synapses_.size();

        // 2. 注入时序循环反馈信号 (一等公民 Recurrent Loops: 取上一时间步记忆状态 prev_output_val)
        for (size_t i = 0; i < num_synapses; ++i) {
            const auto& syn = syn_ptr[i];
            if (syn.is_recurrent) {
                double val = cells_ptr[syn.from_idx].prev_output_val * syn.weight;
                port_ptr[syn.to_idx * 2 + syn.to_port] += val;
            }
        }

        // 3. 按预编译拓扑顺序逐一激发细胞并即时前向传播
        const auto* __restrict order_ptr = execution_order_.data();
        const size_t num_ordered = execution_order_.size();

        for (size_t i = 0; i < num_ordered; ++i) {
            size_t idx = order_ptr[i];
            auto& c = cells_ptr[idx];
            double in0 = port_ptr[idx * 2 + 0];
            double in1 = port_ptr[idx * 2 + 1];

            switch (c.type) {
                case CellType::SENSE_RAW_INPUT_0:
                case CellType::SENSE_RAW_INPUT_1:
                case CellType::SENSE_RAW_INPUT_2:
                case CellType::SENSE_RAW_INPUT_3:
                case CellType::SENSE_CHANNEL: {
                    const size_t ch = receptor_channel_index(c.type, c.param2);
                    c.output_val = (ch < in_dim) ? inputs[ch] * c.param1 : 0.0;
                    break;
                }

                case CellType::OP_EMA: {
                    double alpha = std::clamp(c.param1, 0.001, 1.0);
                    if (c.activation_count == 0) c.state_val = in0;
                    else c.state_val = alpha * in0 + (1.0 - alpha) * c.state_val;
                    c.output_val = c.state_val;
                    break;
                }
                case CellType::OP_DIFF:
                    c.output_val = in0 - c.prev_input;
                    c.prev_input = in0;
                    break;
                case CellType::OP_INTEGRAL:
                    c.state_val = std::clamp(c.state_val + in0 * c.param1, -4.0, 4.0);
                    c.output_val = c.state_val;
                    break;
                case CellType::OP_SUM:
                    c.output_val = in0 + in1;
                    break;
                case CellType::OP_SUB:
                    c.output_val = in0 - in1;
                    break;
                case CellType::OP_MULTIPLY:
                    c.output_val = in0 * in1;
                    break;
                case CellType::OP_RATIO:
                    c.output_val = in0 / (std::abs(in1) > 1e-6 ? in1 : 1e-6);
                    break;
                case CellType::OP_ABS:
                    c.output_val = std::abs(in0);
                    break;
                case CellType::OP_DELAY_N: {
                    int k = std::clamp(static_cast<int>(std::floor(c.param1 * 16.0)), 1, 16);
                    size_t read_idx = (static_cast<size_t>(c.delay_idx) + 16 - static_cast<size_t>(k)) & 15;
                    c.output_val = c.delay_buffer[read_idx];
                    c.delay_buffer[c.delay_idx & 15] = in0;
                    c.delay_idx = static_cast<uint8_t>((c.delay_idx + 1) & 15);
                    break;
                }
                case CellType::OP_OSCILLATOR: {
                    if (c.activation_count == 0 && std::abs(c.state_val) < 1e-6 && std::abs(c.aux_state) < 1e-6) {
                        c.state_val = 0.1;
                    }
                    double mu = (std::abs(c.param1) > 1e-4) ? std::clamp(std::abs(c.param1), 0.01, 5.0) : 1.0;
                    double dt = (std::abs(c.param2) > 1e-4) ? std::clamp(std::abs(c.param2), 0.001, 0.2) : 0.05;

                    double s1 = c.state_val;
                    double s2 = c.aux_state;

                    double ds1 = s2;
                    double ds2 = mu * (1.0 - s1 * s1) * s2 - s1 + in0;

                    s1 += ds1 * dt;
                    s2 += ds2 * dt;

                    c.state_val = std::clamp(s1, -10.0, 10.0);
                    c.aux_state = std::clamp(s2, -10.0, 10.0);
                    c.output_val = c.state_val;
                    break;
                }
                case CellType::OP_QUADRATIC:
                    c.output_val = c.param1 * in0 * in0 + c.param2 * in0 * in1;
                    break;

                case CellType::GATE_THRESHOLD:
                    c.output_val = (in0 > c.param1) ? 1.0 : 0.0;
                    break;
                case CellType::GATE_HYSTERESIS:
                    if (in0 > c.param1) c.latch_state = true;
                    else if (in0 < c.param2) c.latch_state = false;
                    c.output_val = c.latch_state ? 1.0 : -1.0;
                    break;
                case CellType::GATE_AND:
                    c.output_val = (in0 > 0.0 && in1 > 0.0) ? 1.0 : 0.0;
                    break;
                case CellType::GATE_INHIBIT:
                    c.output_val = in0 * std::max(0.0, 1.0 - in1);
                    break;
                case CellType::GATE_DEADZONE:
                    c.output_val = (std::abs(in0) > std::abs(c.param1)) ? in0 : 0.0;
                    break;
                case CellType::GATE_MIN_MAX:
                    c.output_val = (c.param1 > 0.5) ? std::max(in0, in1) : std::min(in0, in1);
                    break;

                case CellType::ACT_PRIMARY_POSITIVE:
                case CellType::ACT_PRIMARY_NEGATIVE:
                case CellType::ACT_DEFENSIVE_RESET:
                case CellType::ACT_IMMUNE_BLOCK:
                case CellType::ACT_CHANNEL:
                case CellType::PREDICT_SENSE_0:
                case CellType::PREDICT_SENSE_1:
                    c.output_val = in0;
                    break;

                case CellType::ASSOCIATION_HUB:
                    c.output_val = std::tanh(in0 + in1 * c.param1);
                    break;
            }

            if (std::abs(c.output_val) > 1e-6) {
                c.activation_count++;
                c.glow_charge = std::min(1.0f, c.glow_charge + 0.3f);
            }

            // 膜穿透孔道微观动力学与跨膜电位演进 (Voltage & Gated Dynamics)
            float drive = static_cast<float>(c.output_val);
            if (drive > 0.05f) {
                c.membrane_pores[0] = std::clamp(c.membrane_pores[0] * 0.75f + 0.25f * std::min(1.0f, drive * 0.8f), 0.05f, 1.0f);
                c.membrane_pores[2] = std::clamp(c.membrane_pores[2] * 0.80f + 0.20f * std::min(1.0f, drive * 0.5f), 0.02f, 0.9f);
                c.membrane_potential += (35.0f - c.membrane_potential) * (c.membrane_pores[0] * 0.25f);
            } else {
                c.membrane_pores[0] = std::max(0.05f, c.membrane_pores[0] * 0.88f);
                c.membrane_pores[1] = std::clamp(c.membrane_pores[1] * 0.85f + 0.15f * 0.4f, 0.1f, 0.8f);
                c.membrane_potential += (-70.0f - c.membrane_potential) * (c.membrane_pores[1] * 0.20f);
            }
            c.membrane_pores[4] = std::clamp(0.2f + 0.6f * c.glow_charge, 0.1f, 0.95f);
            c.membrane_pores[5] = std::clamp(0.1f + 0.5f * (std::abs(drive) > 0.5f ? 0.6f : 0.1f), 0.05f, 0.85f);

            // 前向突触即时传递至后续位阶节点 (CSR 出边索引: O(出度))
            for (size_t k = out_start_[idx]; k < out_start_[idx + 1]; ++k) {
                const auto& syn = syn_ptr[out_edges_[k]];
                if (!syn.is_recurrent) {
                    port_ptr[syn.to_idx * 2 + syn.to_port] += c.output_val * syn.weight;
                }
            }
        }

        // 4. 终身在线 Oja 赫布可塑性学习 (Lifelong Hebbian Plasticity In-Place Update)
        if (enable_hebbian) {
            for (size_t i = 0; i < num_synapses; ++i) {
                auto& syn = syn_ptr[i];
                if (syn.hebbian_rate <= 1e-6) continue;

                double u_pre = cells_ptr[syn.from_idx].output_val;
                double u_post = cells_ptr[syn.to_idx].output_val;

                // Oja 规则: delta_w = eta * (u_pre * u_post - decay * u_post^2 * w)
                double delta_w = syn.hebbian_rate * (u_pre * u_post - syn.hebbian_decay * u_post * u_post * syn.weight);
                syn.weight = std::clamp(syn.weight + delta_w, -3.0, 3.0);
                if (!std::isfinite(syn.weight)) syn.weight = syn.initial_weight;
            }
        }

        // 5. 提交当前时刻输出至 prev_output_val 作为下一时刻循环记忆
        for (size_t i = 0; i < cells.size(); ++i) {
            cells_ptr[i].prev_output_val = cells_ptr[i].output_val;
        }

        // 6. 收集最终动作信号与概念相空间状态
        ActionOutputs actions{};
        double total_energy = 0.0;
        for (const auto& c : cells) {
            total_energy += c.output_val * c.output_val;
        }
        actions.thought_energy = total_energy;

        for (const auto& ac : compiled_actions_) {
            double val = cells_ptr[ac.cell_idx].output_val;
            if (ac.type == CellType::ACT_PRIMARY_POSITIVE) actions.positive_action = val;
            else if (ac.type == CellType::ACT_PRIMARY_NEGATIVE) actions.negative_action = val;
            else if (ac.type == CellType::ACT_DEFENSIVE_RESET) actions.defensive_reset = val;
            else if (ac.type == CellType::ACT_IMMUNE_BLOCK && val > 0.5) actions.immune_lock = true;
            else if (ac.type == CellType::PREDICT_SENSE_0) actions.predicted_sense_0 = val;
            else if (ac.type == CellType::PREDICT_SENSE_1) actions.predicted_sense_1 = val;
        }

        // 预测误差 (Surprise)
        const double in0_obs = (in_dim > 0) ? inputs[0] : 0.0;
        const double in1_obs = (in_dim > 1) ? inputs[1] : 0.0;
        double err0 = in0_obs - actions.predicted_sense_0;
        double err1 = in1_obs - actions.predicted_sense_1;
        actions.prediction_error = std::sqrt(err0 * err0 + err1 * err1);

        if (actions.prediction_error > 5.0) actions.thought_mode = "SURPRISE";
        else if (actions.thought_energy > 8.0) actions.thought_mode = "FOCUS";
        else if (actions.thought_energy < 0.2) actions.thought_mode = "EXPLORATION";
        else actions.thought_mode = "STABLE_ATTRACTOR";

        return actions;
    }

    void write_action_tensor(const ActionOutputs& acts, float* y, uint32_t out_dim) const {
        if (!y || out_dim == 0) return;
        for (uint32_t i = 0; i < out_dim; ++i) y[i] = 0.0f;
        if (out_dim > 0) y[0] = static_cast<float>(acts.positive_action);
        if (out_dim > 1) y[1] = static_cast<float>(acts.negative_action);
        if (out_dim > 2) y[2] = static_cast<float>(acts.defensive_reset);
        if (out_dim > 3) y[3] = acts.immune_lock ? 1.0f : 0.0f;
        for (const auto& ac : compiled_actions_) {
            if (ac.cell_idx >= cells.size()) continue;
            const Cell& c = cells[ac.cell_idx];
            if (!is_effector_cell(c.type)) continue;
            const size_t ch = effector_channel_index(c.type, c.param2);
            if (ch < out_dim) y[ch] = static_cast<float>(c.output_val);
        }
    }

    // ── 闭门心理推演与反事实想象 (Mental Simulation / Thought Rollout) ──
    // 切断真实外部输入，网络利用自身预测受体与时序递归环路在几微秒内内部试错自激推演未来轨迹！
    std::vector<ActionOutputs> simulate_mental_rollout(int rollout_steps = 5) {
        std::vector<ActionOutputs> imagined_trajectory;
        imagined_trajectory.reserve(rollout_steps);

        double simulated_inputs[4] = {
            cells.empty() ? 0.0 : cells[0].output_val,
            cells.size() > 1 ? cells[1].output_val : 0.0,
            0.0, 0.0
        };

        for (int step = 0; step < rollout_steps; ++step) {
            auto act = forward(simulated_inputs, false); // 心理推演中不污染真实外部突触
            imagined_trajectory.push_back(act);

            // 自回输：将内部预测作为下一步推演的世界模型输入
            simulated_inputs[0] = act.predicted_sense_0;
            simulated_inputs[1] = act.predicted_sense_1;
        }

        return imagined_trajectory;
    }

    // 导出全息 3D 可视化 JSON 数据帧 (包含物理坐标、发光电位、循环连接、突触可塑性与思维相空间动能)
    std::string to_json() const {
        std::ostringstream ss;
        double safe_fitness = std::isfinite(fitness_score) ? fitness_score : 0.0;
        double total_e = 0.0;
        for (const auto& c : cells) {
            if (std::isfinite(c.output_val)) total_e += c.output_val * c.output_val;
        }
        if (!std::isfinite(total_e)) total_e = 0.0;

        ss << "{\n";
        ss << "  \"organism_id\": " << organism_id << ",\n";
        ss << "  \"generation\": " << generation << ",\n";
        ss << "  \"lineage\": \"" << lineage_name << "\",\n";
        ss << "  \"fitness\": " << safe_fitness << ",\n";
        ss << "  \"thought_energy\": " << total_e << ",\n";
        ss << "  \"cells\": [\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            const auto& c = cells[i];
            double safe_out = std::isfinite(c.output_val) ? c.output_val : 0.0;
            float safe_x = std::isfinite(c.x) ? c.x : 0.0f;
            float safe_y = std::isfinite(c.y) ? c.y : 0.0f;
            float safe_z = std::isfinite(c.z) ? c.z : 0.0f;
            float safe_glow = std::isfinite(c.glow_charge) ? c.glow_charge : 0.0f;
            ss << "    {\"id\": " << c.id << ", \"type\": \"" << to_string(c.type) << "\", "
               << "\"param1\": " << (std::isfinite(c.param1) ? c.param1 : 0.0) << ", "
               << "\"param2\": " << (std::isfinite(c.param2) ? c.param2 : 0.0) << ", "
               << "\"output\": " << safe_out << ", \"activations\": " << c.activation_count << ", "
               << "\"x\": " << safe_x << ", \"y\": " << safe_y << ", \"z\": " << safe_z << ", "
               << "\"glow\": " << safe_glow << ", "
               << "\"organ\": " << static_cast<int>(c.organ_type) << ", "
               << "\"vm\": " << c.membrane_potential << ", "
               << "\"morph\": " << c.morphogen_concentration << "}"
               << (i + 1 < cells.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";
        ss << "  \"synapses\": [\n";
        for (size_t i = 0; i < synapses.size(); ++i) {
            const auto& s = synapses[i];
            double safe_w = std::isfinite(s.weight) ? s.weight : 0.0;
            double safe_init_w = std::isfinite(s.initial_weight) ? s.initial_weight : safe_w;
            float safe_p = std::isfinite(s.photon_pos) ? s.photon_pos : 0.0f;
            ss << "    {\"from\": " << s.from_cell_id << ", \"to\": " << s.to_cell_id << ", "
               << "\"port\": " << (int)s.to_port << ", "
               << "\"initial_weight\": " << safe_init_w << ", "
               << "\"weight\": " << safe_w << ", "
               << "\"active\": " << (s.is_active ? "true" : "false") << ", "
               << "\"recurrent\": " << (s.is_recurrent ? "true" : "false") << ", "
               << "\"hebb_rate\": " << s.hebbian_rate << ", "
               << "\"hebb_decay\": " << s.hebbian_decay << ", "
               << "\"photon_pos\": " << safe_p << "}"
               << (i + 1 < synapses.size() ? "," : "") << "\n";
        }
        ss << "  ]\n";
        ss << "}";
        return ss.str();
    }

    // ── 鲍德温效应与基因固化 (Baldwin Effect & Genetic Crystallization) ──
    // 将个体后天生命期内通过 Oja 在线学习到的有效突触权重，固化并写入为遗传初始基线！
    void crystallize_plasticity(double rate = 1.0) {
        rate = std::clamp(rate, 0.0, 1.0);
        for (auto& syn : synapses) {
            syn.initial_weight = syn.initial_weight * (1.0 - rate) + syn.weight * rate;
            syn.weight = syn.initial_weight;
        }
        for (auto& csyn : compiled_synapses_) {
            csyn.initial_weight = csyn.initial_weight * (1.0 - rate) + csyn.weight * rate;
            csyn.weight = csyn.initial_weight;
        }
    }

    // ── 全息检查点反序列化 (JSON Deserialization) ──
    static CellularOrganism from_json(const std::string& json_str) {
        CellularOrganism org;

        auto find_val = [&](const std::string& key) -> std::string {
            auto pos = json_str.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            auto colon = json_str.find(":", pos);
            if (colon == std::string::npos) return "";
            size_t start = colon + 1;
            while (start < json_str.size() && (json_str[start] == ' ' || json_str[start] == '\t' || json_str[start] == '\n' || json_str[start] == '\r')) start++;
            if (start >= json_str.size()) return "";
            if (json_str[start] == '\"') {
                size_t end = json_str.find('\"', start + 1);
                if (end == std::string::npos) return "";
                return json_str.substr(start + 1, end - start - 1);
            } else {
                size_t end = start;
                while (end < json_str.size() && json_str[end] != ',' && json_str[end] != '}' && json_str[end] != '\n' && json_str[end] != '\r') end++;
                return json_str.substr(start, end - start);
            }
        };

        std::string id_str = find_val("organism_id");
        if (!id_str.empty()) org.organism_id = static_cast<uint64_t>(std::stoull(id_str));
        std::string gen_str = find_val("generation");
        if (!gen_str.empty()) org.generation = static_cast<uint32_t>(std::stoul(gen_str));
        std::string lin_str = find_val("lineage");
        if (!lin_str.empty()) org.lineage_name = lin_str;
        std::string fit_str = find_val("fitness");
        if (!fit_str.empty()) org.fitness_score = std::stod(fit_str);

        // 解析 cells
        auto cells_pos = json_str.find("\"cells\"");
        if (cells_pos != std::string::npos) {
            auto open_bracket = json_str.find("[", cells_pos);
            auto close_bracket = json_str.find("]", open_bracket);
            if (open_bracket != std::string::npos && close_bracket != std::string::npos) {
                std::string cells_block = json_str.substr(open_bracket, close_bracket - open_bracket + 1);
                size_t pos = 0;
                while ((pos = cells_block.find("{", pos)) != std::string::npos) {
                    size_t end_brace = cells_block.find("}", pos);
                    if (end_brace == std::string::npos) break;
                    std::string cell_item = cells_block.substr(pos, end_brace - pos + 1);

                    auto parse_field = [&](const std::string& k) -> std::string {
                        auto p = cell_item.find("\"" + k + "\"");
                        if (p == std::string::npos) return "";
                        auto c = cell_item.find(":", p);
                        if (c == std::string::npos) return "";
                        size_t s = c + 1;
                        while (s < cell_item.size() && (cell_item[s] == ' ' || cell_item[s] == '\t')) s++;
                        if (s >= cell_item.size()) return "";
                        if (cell_item[s] == '\"') {
                            size_t e = cell_item.find('\"', s + 1);
                            if (e == std::string::npos) return "";
                            return cell_item.substr(s + 1, e - s - 1);
                        } else {
                            size_t e = s;
                            while (e < cell_item.size() && cell_item[e] != ',' && cell_item[e] != '}') e++;
                            return cell_item.substr(s, e - s);
                        }
                    };

                    uint32_t cid = static_cast<uint32_t>(std::stoul(parse_field("id").empty() ? "0" : parse_field("id")));
                    CellType ctype = cell_type_from_string(parse_field("type"));
                    double p1 = parse_field("param1").empty() ? 0.0 : std::stod(parse_field("param1"));
                    double p2 = parse_field("param2").empty() ? 0.0 : std::stod(parse_field("param2"));
                    float x = parse_field("x").empty() ? 0.0f : std::stof(parse_field("x"));
                    float y = parse_field("y").empty() ? 0.0f : std::stof(parse_field("y"));
                    float z = parse_field("z").empty() ? 0.0f : std::stof(parse_field("z"));

                    Cell c{cid, ctype, p1, p2, 0.0, 0.0, false, 0.0, 0, 0, x, y, z};
                    std::string organ_str = parse_field("organ");
                    if (!organ_str.empty()) c.organ_type = static_cast<uint8_t>(std::stoul(organ_str));
                    std::string vm_str = parse_field("vm");
                    if (!vm_str.empty()) c.membrane_potential = std::stof(vm_str);
                    std::string morph_str = parse_field("morph");
                    if (!morph_str.empty()) c.morphogen_concentration = std::stof(morph_str);
                    org.cells.push_back(c);
                    pos = end_brace + 1;
                }
            }
        }

        // 解析 synapses
        auto syn_pos = json_str.find("\"synapses\"");
        if (syn_pos != std::string::npos) {
            auto open_bracket = json_str.find("[", syn_pos);
            auto close_bracket = json_str.find("]", open_bracket);
            if (open_bracket != std::string::npos && close_bracket != std::string::npos) {
                std::string syn_block = json_str.substr(open_bracket, close_bracket - open_bracket + 1);
                size_t pos = 0;
                while ((pos = syn_block.find("{", pos)) != std::string::npos) {
                    size_t end_brace = syn_block.find("}", pos);
                    if (end_brace == std::string::npos) break;
                    std::string syn_item = syn_block.substr(pos, end_brace - pos + 1);

                    auto parse_field = [&](const std::string& k) -> std::string {
                        auto p = syn_item.find("\"" + k + "\"");
                        if (p == std::string::npos) return "";
                        auto c = syn_item.find(":", p);
                        if (c == std::string::npos) return "";
                        size_t s = c + 1;
                        while (s < syn_item.size() && (syn_item[s] == ' ' || syn_item[s] == '\t')) s++;
                        if (s >= syn_item.size()) return "";
                        if (syn_item[s] == '\"') {
                            size_t e = syn_item.find('\"', s + 1);
                            if (e == std::string::npos) return "";
                            return syn_item.substr(s + 1, e - s - 1);
                        } else {
                            size_t e = s;
                            while (e < syn_item.size() && syn_item[e] != ',' && syn_item[e] != '}') e++;
                            return syn_item.substr(s, e - s);
                        }
                    };

                    uint32_t from = static_cast<uint32_t>(std::stoul(parse_field("from").empty() ? "0" : parse_field("from")));
                    uint32_t to = static_cast<uint32_t>(std::stoul(parse_field("to").empty() ? "0" : parse_field("to")));
                    uint8_t port = static_cast<uint8_t>(std::stoul(parse_field("port").empty() ? "0" : parse_field("port")));
                    double weight = parse_field("weight").empty() ? 1.0 : std::stod(parse_field("weight"));
                    std::string init_w_str = parse_field("initial_weight");
                    double initial_weight = init_w_str.empty() ? weight : std::stod(init_w_str);
                    bool active = (parse_field("active") == "true" || parse_field("active").empty());
                    bool recurrent = (parse_field("recurrent") == "true");
                    double hebb_rate = parse_field("hebb_rate").empty() ? 0.005 : std::stod(parse_field("hebb_rate"));
                    double hebb_decay = parse_field("hebb_decay").empty() ? 0.02 : std::stod(parse_field("hebb_decay"));

                    Synapse s{from, to, port, weight, active, 60.0f, -1.0f};
                    s.initial_weight = initial_weight;
                    s.weight = weight;
                    s.is_recurrent = recurrent;
                    s.hebbian_rate = hebb_rate;
                    s.hebbian_decay = hebb_decay;
                    org.synapses.push_back(s);
                    pos = end_brace + 1;
                }
            }
        }

        org.compile();
        return org;
    }

    // ════════════════════════════════════════════════════════════════════════
    // ── 五层清晰分层持久化契约 (Stratified Persistence Architecture) ──
    // 1. genome.json    : 先天纯基因（拓扑、细胞类型/空间参数、遗传初始权重 initial_weight）
    // 2. epigenome.json : 表观遗传与可塑性调制（固化状态、学习率、适应度）
    // 3. runtime.ckpt   : 运行时操作态（实时放电电位、当前突触权值 weight、递归状态）
    // 4. memory.json    : 情节/世界模型记忆（经验回放池、不默认遗传）
    // ════════════════════════════════════════════════════════════════════════

    // 1. 导出纯先天基因 (genome.json)
    std::string export_genome_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"organism_id\": " << organism_id << ",\n";
        ss << "  \"generation\": " << generation << ",\n";
        ss << "  \"lineage\": \"" << lineage_name << "\",\n";
        ss << "  \"cells\": [\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            const auto& c = cells[i];
            ss << "    {\"id\": " << c.id << ", \"type\": \"" << to_string(c.type) << "\", "
               << "\"param1\": " << c.param1 << ", \"param2\": " << c.param2 << ", "
               << "\"x\": " << c.x << ", \"y\": " << c.y << ", \"z\": " << c.z << "}"
               << (i + 1 < cells.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";
        ss << "  \"synapses\": [\n";
        for (size_t i = 0; i < synapses.size(); ++i) {
            const auto& s = synapses[i];
            ss << "    {\"from\": " << s.from_cell_id << ", \"to\": " << s.to_cell_id << ", "
               << "\"port\": " << (int)s.to_port << ", "
               << "\"initial_weight\": " << s.initial_weight << ", "
               << "\"recurrent\": " << (s.is_recurrent ? "true" : "false") << ", "
               << "\"hebb_rate\": " << s.hebbian_rate << ", "
               << "\"hebb_decay\": " << s.hebbian_decay << "}"
               << (i + 1 < synapses.size() ? "," : "") << "\n";
        }
        ss << "  ]\n";
        ss << "}";
        return ss.str();
    }

    // 2. 导出表观遗传与调制态 (epigenome.json)
    std::string export_epigenome_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"organism_id\": " << organism_id << ",\n";
        ss << "  \"fitness_score\": " << fitness_score << ",\n";
        ss << "  \"plastic_drift_norms\": [\n";
        for (size_t i = 0; i < synapses.size(); ++i) {
            const auto& s = synapses[i];
            double drift = s.weight - s.initial_weight;
            ss << "    {\"from\": " << s.from_cell_id << ", \"to\": " << s.to_cell_id 
               << ", \"drift\": " << drift << "}"
               << (i + 1 < synapses.size() ? "," : "") << "\n";
        }
        ss << "  ]\n";
        ss << "}";
        return ss.str();
    }

    // 3. 导出运行时操作态 (runtime.ckpt)
    std::string export_runtime_ckpt() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"organism_id\": " << organism_id << ",\n";
        ss << "  \"live_cell_states\": [\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            const auto& c = cells[i];
            ss << "    {\"id\": " << c.id << ", \"output\": " << c.output_val 
               << ", \"activations\": " << c.activation_count << ", \"glow\": " << c.glow_charge << "}"
               << (i + 1 < cells.size() ? "," : "") << "\n";
        }
        ss << "  ],\n";
        ss << "  \"live_synapse_weights\": [\n";
        for (size_t i = 0; i < synapses.size(); ++i) {
            const auto& s = synapses[i];
            ss << "    {\"from\": " << s.from_cell_id << ", \"to\": " << s.to_cell_id 
               << ", \"weight\": " << s.weight << ", \"photon_pos\": " << s.photon_pos << "}"
               << (i + 1 < synapses.size() ? "," : "") << "\n";
        }
        ss << "  ]\n";
        ss << "}";
        return ss.str();
    }

    // 从先天基因初始化（不带任何后天未固化权重）
    static CellularOrganism from_genome_json(const std::string& json_str) {
        auto org = from_json(json_str);
        // 严格保证 live weight 重置为 initial_weight
        for (auto& syn : org.synapses) {
            syn.weight = syn.initial_weight;
        }
        for (auto& csyn : org.compiled_synapses_) {
            csyn.weight = csyn.initial_weight;
        }
        return org;
    }

    bool save_checkpoint_json(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) return false;
        ofs << to_json();
        return true;
    }

    static bool is_valid_cell_type_code(uint8_t op) {
        switch (static_cast<CellType>(op)) {
            case CellType::SENSE_RAW_INPUT_0:
            case CellType::SENSE_RAW_INPUT_1:
            case CellType::SENSE_RAW_INPUT_2:
            case CellType::SENSE_RAW_INPUT_3:
            case CellType::SENSE_CHANNEL:
            case CellType::OP_EMA:
            case CellType::OP_DIFF:
            case CellType::OP_INTEGRAL:
            case CellType::OP_SUM:
            case CellType::OP_SUB:
            case CellType::OP_MULTIPLY:
            case CellType::OP_RATIO:
            case CellType::OP_ABS:
            case CellType::OP_DELAY_N:
            case CellType::OP_OSCILLATOR:
            case CellType::OP_QUADRATIC:
            case CellType::GATE_THRESHOLD:
            case CellType::GATE_HYSTERESIS:
            case CellType::GATE_AND:
            case CellType::GATE_INHIBIT:
            case CellType::GATE_DEADZONE:
            case CellType::GATE_MIN_MAX:
            case CellType::ACT_PRIMARY_POSITIVE:
            case CellType::ACT_PRIMARY_NEGATIVE:
            case CellType::ACT_DEFENSIVE_RESET:
            case CellType::ACT_IMMUNE_BLOCK:
            case CellType::ACT_CHANNEL:
            case CellType::PREDICT_SENSE_0:
            case CellType::PREDICT_SENSE_1:
            case CellType::ASSOCIATION_HUB:
                return true;
            default:
                return false;
        }
    }

    static uint8_t cell_type_to_sdsc_opcode(CellType t) {
        return static_cast<uint8_t>(t);
    }

    bool save_checkpoint_bin(const std::string& filepath) const {
        if (cells.empty()) return false;
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) return false;

        uint32_t num_cells = static_cast<uint32_t>(cells.size());
        std::unordered_map<uint32_t, uint32_t> id_to_idx;
        for (uint32_t i = 0; i < num_cells; ++i) {
            id_to_idx[cells[i].id] = i;
        }

        std::vector<std::vector<std::pair<uint32_t, float>>> adj(num_cells);
        uint32_t num_synapses = 0;
        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            auto it_u = id_to_idx.find(syn.from_cell_id);
            auto it_v = id_to_idx.find(syn.to_cell_id);
            if (it_u != id_to_idx.end() && it_v != id_to_idx.end()) {
                uint32_t packed_target = (static_cast<uint32_t>(syn.to_port & 0xFF) << 24) | (it_v->second & 0x00FFFFFF);
                adj[it_u->second].push_back({packed_target, static_cast<float>(syn.weight)});
                num_synapses++;
            }
        }

        uint32_t in_dim = 0, out_dim = 0;
        for (const auto& c : cells) {
            if (c.type <= CellType::SENSE_CHANNEL || c.type == CellType::PREDICT_SENSE_0 || c.type == CellType::PREDICT_SENSE_1) in_dim++;
            if (c.type >= CellType::ACT_PRIMARY_POSITIVE && c.type <= CellType::ACT_CHANNEL) out_dim++;
        }

        uint64_t header_sz = 72;
        uint64_t cells_off = header_sz;
        uint64_t cells_sz = num_cells * 4;
        uint64_t row_ptr_off = cells_off + cells_sz;
        uint64_t row_ptr_sz = (num_cells + 1) * 4;
        uint64_t col_idx_off = row_ptr_off + row_ptr_sz;
        uint64_t col_idx_sz = num_synapses * 4;
        uint64_t weights_off = col_idx_off + col_idx_sz;
        uint64_t weights_sz = num_synapses * 4;
        uint64_t coords_off = weights_off + weights_sz;

        struct SDSC_BIN_HDR {
            uint32_t magic{0x53445343};
            uint32_t version{2};
            uint32_t n_cells{0};
            uint32_t n_syns{0};
            uint32_t in_d{0};
            uint32_t out_d{0};
            uint64_t c_off{0};
            uint64_t rp_off{0};
            uint64_t ci_off{0};
            uint64_t w_off{0};
            uint64_t coords_off{0};
            uint64_t extra{0};
        } hdr;
        hdr.n_cells = num_cells;
        hdr.n_syns = num_synapses;
        hdr.in_d = in_dim;
        hdr.out_d = out_dim;
        hdr.c_off = cells_off;
        hdr.rp_off = row_ptr_off;
        hdr.ci_off = col_idx_off;
        hdr.w_off = weights_off;
        hdr.coords_off = coords_off;

        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        for (const auto& c : cells) {
            uint8_t op = cell_type_to_sdsc_opcode(c.type);
            int8_t p1_i8 = static_cast<int8_t>(std::clamp(std::round(c.param1 * 64.0), -128.0, 127.0));
            int8_t p2_i8 = static_cast<int8_t>(std::clamp(std::round(c.param2 * 64.0), -128.0, 127.0));
            uint8_t p1_u8 = static_cast<uint8_t>(p1_i8);
            uint8_t p2_u8 = static_cast<uint8_t>(p2_i8);
            uint8_t flags = 0;
            if (c.type <= CellType::SENSE_CHANNEL || c.type == CellType::PREDICT_SENSE_0 || c.type == CellType::PREDICT_SENSE_1) flags |= 0x01;
            if (c.type >= CellType::ACT_PRIMARY_POSITIVE && c.type <= CellType::ACT_CHANNEL) flags |= 0x02;
            ofs.write(reinterpret_cast<const char*>(&op), 1);
            ofs.write(reinterpret_cast<const char*>(&p1_u8), 1);
            ofs.write(reinterpret_cast<const char*>(&p2_u8), 1);
            ofs.write(reinterpret_cast<const char*>(&flags), 1);
        }

        std::vector<uint32_t> row_ptr(num_cells + 1, 0);
        std::vector<uint32_t> col_idx;
        std::vector<float> weights;
        col_idx.reserve(num_synapses);
        weights.reserve(num_synapses);

        uint32_t curr = 0;
        for (uint32_t i = 0; i < num_cells; ++i) {
            row_ptr[i] = curr;
            for (const auto& edge : adj[i]) {
                col_idx.push_back(edge.first);
                weights.push_back(edge.second);
                curr++;
            }
        }
        row_ptr[num_cells] = curr;

        ofs.write(reinterpret_cast<const char*>(row_ptr.data()), row_ptr.size() * sizeof(uint32_t));
        if (!col_idx.empty()) {
            ofs.write(reinterpret_cast<const char*>(col_idx.data()), col_idx.size() * sizeof(uint32_t));
            ofs.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        }

        std::vector<float> coords(num_cells * 3, 0.0f);
        for (uint32_t i = 0; i < num_cells; ++i) {
            coords[i * 3 + 0] = static_cast<float>(cells[i].x);
            coords[i * 3 + 1] = static_cast<float>(cells[i].y);
            coords[i * 3 + 2] = static_cast<float>(cells[i].z);
        }
        ofs.write(reinterpret_cast<const char*>(coords.data()), coords.size() * sizeof(float));

        return true;
    }

    static CellType sdsc_opcode_to_cell_type(uint8_t op, uint8_t flags = 0) {
        if (is_valid_cell_type_code(op)) {
            return static_cast<CellType>(op);
        }
        // Legacy fallback mapping if loading older v1 checkpoints
        if (flags & 0x01) {
            if (op == 0) return CellType::SENSE_RAW_INPUT_0;
            if (op == 1) return CellType::SENSE_RAW_INPUT_1;
            if (op == 2) return CellType::SENSE_RAW_INPUT_2;
            if (op == 3) return CellType::SENSE_RAW_INPUT_3;
            return CellType::SENSE_CHANNEL;
        }
        if (flags & 0x02) {
            if (op == 21) return CellType::ACT_PRIMARY_POSITIVE;
            if (op == 22) return CellType::ACT_PRIMARY_NEGATIVE;
            if (op == 23) return CellType::ACT_DEFENSIVE_RESET;
            return CellType::ACT_CHANNEL;
        }
        switch (op) {
            case 0: return CellType::SENSE_RAW_INPUT_0;
            case 1: return CellType::SENSE_RAW_INPUT_1;
            case 2: return CellType::SENSE_RAW_INPUT_2;
            case 3: return CellType::SENSE_RAW_INPUT_3;
            case 4: return CellType::OP_SUM;
            case 5: return CellType::OP_INTEGRAL;
            case 8: return CellType::OP_EMA;
            case 10: return CellType::OP_ABS;
            case 11: return CellType::OP_MULTIPLY;
            case 12: return CellType::OP_DIFF;
            case 13: return CellType::OP_SUB;
            case 14: return CellType::OP_RATIO;
            case 15: return CellType::GATE_THRESHOLD;
            case 16: return CellType::GATE_HYSTERESIS;
            case 17: return CellType::GATE_DEADZONE;
            case 18: return CellType::GATE_INHIBIT;
            case 19: return CellType::GATE_AND;
            case 20: return CellType::GATE_MIN_MAX;
            case 21: return CellType::ACT_PRIMARY_POSITIVE;
            case 22: return CellType::ACT_PRIMARY_NEGATIVE;
            case 23: return CellType::ACT_DEFENSIVE_RESET;
            case 24: return CellType::ASSOCIATION_HUB;
            case 25: return CellType::OP_OSCILLATOR;
            default: return CellType::OP_EMA;
        }
    }

    static CellularOrganism load_checkpoint_json(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return CellularOrganism();
        std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        return from_json(str);
    }

    static CellularOrganism load_checkpoint_bin(const std::string& filepath) {
        CellularOrganism org;
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) return org;

        struct SDSC_BIN_HDR {
            uint32_t magic{0};
            uint32_t version{0};
            uint32_t n_cells{0};
            uint32_t n_syns{0};
            uint32_t in_d{0};
            uint32_t out_d{0};
            uint64_t c_off{0};
            uint64_t rp_off{0};
            uint64_t ci_off{0};
            uint64_t w_off{0};
            uint64_t coords_off{0};
            uint64_t extra{0};
        } hdr;

        ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!ifs || hdr.magic != 0x53445343 || hdr.version != 2 || hdr.n_cells == 0) {
            return org;
        }

        uint32_t num_cells = hdr.n_cells;
        uint32_t num_syns = hdr.n_syns;

        // 1. 读取 Cell 特征
        ifs.seekg(hdr.c_off);
        std::vector<uint8_t> cell_bytes(num_cells * 4);
        ifs.read(reinterpret_cast<char*>(cell_bytes.data()), cell_bytes.size());

        org.cells.resize(num_cells);
        for (uint32_t i = 0; i < num_cells; ++i) {
            uint8_t op = cell_bytes[i * 4 + 0];
            uint8_t p1_u8 = cell_bytes[i * 4 + 1];
            uint8_t p2_u8 = cell_bytes[i * 4 + 2];
            uint8_t flags = cell_bytes[i * 4 + 3];

            int8_t p1_i8 = static_cast<int8_t>(p1_u8);
            int8_t p2_i8 = static_cast<int8_t>(p2_u8);

            org.cells[i].id = i;
            org.cells[i].type = sdsc_opcode_to_cell_type(op, flags);
            org.cells[i].param1 = static_cast<double>(p1_i8) / 64.0;
            org.cells[i].param2 = static_cast<double>(p2_i8) / 64.0;
            org.cells[i].latch_state = false;
        }

        // 2. 读取坐标 (若存在)
        if (hdr.coords_off > 0) {
            ifs.seekg(hdr.coords_off);
            std::vector<float> coords(num_cells * 3, 0.0f);
            ifs.read(reinterpret_cast<char*>(coords.data()), coords.size() * sizeof(float));
            if (ifs) {
                for (uint32_t i = 0; i < num_cells; ++i) {
                    org.cells[i].x = coords[i * 3 + 0];
                    org.cells[i].y = coords[i * 3 + 1];
                    org.cells[i].z = coords[i * 3 + 2];
                }
            }
        }

        // 3. 读取 CSR 图结构并构建突触
        ifs.seekg(hdr.rp_off);
        std::vector<uint32_t> row_ptr(num_cells + 1);
        ifs.read(reinterpret_cast<char*>(row_ptr.data()), row_ptr.size() * sizeof(uint32_t));

        std::vector<uint32_t> col_idx(num_syns);
        std::vector<float> weights(num_syns);
        if (num_syns > 0) {
            ifs.seekg(hdr.ci_off);
            ifs.read(reinterpret_cast<char*>(col_idx.data()), num_syns * sizeof(uint32_t));
            ifs.seekg(hdr.w_off);
            ifs.read(reinterpret_cast<char*>(weights.data()), num_syns * sizeof(float));
        }

        for (uint32_t u = 0; u < num_cells; ++u) {
            uint32_t start_edge = row_ptr[u];
            uint32_t end_edge = row_ptr[u + 1];
            for (uint32_t e = start_edge; e < end_edge && e < num_syns; ++e) {
                uint32_t packed = col_idx[e];
                uint32_t v = packed & 0x00FFFFFF;
                uint8_t port = static_cast<uint8_t>((packed >> 24) & 0xFF);
                if (v < num_cells) {
                    Synapse s;
                    s.from_cell_id = org.cells[u].id;
                    s.to_cell_id = org.cells[v].id;
                    s.to_port = port;
                    s.weight = static_cast<double>(weights[e]);
                    s.initial_weight = s.weight;
                    s.is_active = true;
                    org.synapses.push_back(s);
                }
            }
        }

        org.compile();
        return org;
    }

    // ========================================================================
    // 7. 李雅普诺夫 BIBO 稳定性判定与自适应阻尼 (Lyapunov BIBO Stability Analyzer)
    // 物理公理: 存在反馈环路的非线性动力系统若环增益 >= 1.0 且无耗散阻尼，必发生数值发散
    // ========================================================================
    struct LyapunovReport {
        bool is_stable{true};
        double max_loop_gain{0.0};
        size_t detected_cycles_count{0};
        std::vector<std::vector<uint32_t>> unstable_cycles;
    };

    static double get_cell_operator_gain(CellType t, double p1) {
        (void)p1;
        switch (t) {
            case CellType::OP_EMA: return 0.85; // 指数衰减耗散
            case CellType::GATE_HYSTERESIS: return 0.50; // 施密特迟滞强抗震颤
            case CellType::GATE_DEADZONE: return 0.60;   // 死区滤波
            case CellType::GATE_THRESHOLD: return 0.30;  // 阶跃开关
            case CellType::OP_INTEGRAL: return 1.15;     // 累加器若无阻尼易发散
            case CellType::OP_DIFF: return 1.35;         // 微分高频放大
            case CellType::OP_SUM: return 1.0;
            case CellType::OP_SUB: return 1.0;
            case CellType::OP_MULTIPLY: return 1.0;
            case CellType::OP_RATIO: return 1.0;
            case CellType::OP_ABS: return 1.0;
            case CellType::OP_QUADRATIC: return 1.0;
            case CellType::OP_OSCILLATOR: return 0.95;   // 极限环自限
            default: return 1.0;
        }
    }

    LyapunovReport check_lyapunov_stability() const {
        LyapunovReport rep;
        if (cells.empty() || synapses.empty()) return rep;

        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < cells.size(); ++i) id_to_idx[cells[i].id] = i;

        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, double>>> adj;
        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            if (id_to_idx.count(syn.from_cell_id) && id_to_idx.count(syn.to_cell_id)) {
                adj[syn.from_cell_id].push_back({syn.to_cell_id, syn.weight});
            }
        }

        std::unordered_map<uint32_t, int> visited;
        std::vector<uint32_t> path;
        std::vector<double> weights_path;

        std::function<void(uint32_t)> dfs = [&](uint32_t u) {
            visited[u] = 1;
            path.push_back(u);

            for (const auto& edge : adj[u]) {
                uint32_t v = edge.first;
                double w = edge.second;
                weights_path.push_back(w);

                if (visited[v] == 1) {
                    rep.detected_cycles_count++;
                    auto it = std::find(path.begin(), path.end(), v);
                    size_t start_idx = std::distance(path.begin(), it);

                    double cycle_gain = 1.0;
                    bool has_dissipative_gate = false;
                    std::vector<uint32_t> cycle_nodes;

                    for (size_t k = start_idx; k < path.size(); ++k) {
                        uint32_t cid = path[k];
                        cycle_nodes.push_back(cid);
                        const auto& cell = cells[id_to_idx.at(cid)];
                        double node_gain = get_cell_operator_gain(cell.type, cell.param1);
                        cycle_gain *= node_gain;
                        if (cell.type == CellType::GATE_HYSTERESIS || 
                            cell.type == CellType::GATE_DEADZONE ||
                            cell.type == CellType::OP_EMA) {
                            has_dissipative_gate = true;
                        }
                    }
                    for (size_t k = start_idx; k < weights_path.size(); ++k) {
                        cycle_gain *= std::fabs(weights_path[k]);
                    }

                    if (cycle_gain > rep.max_loop_gain) {
                        rep.max_loop_gain = cycle_gain;
                    }

                    if (cycle_gain >= 1.0 && !has_dissipative_gate) {
                        rep.is_stable = false;
                        rep.unstable_cycles.push_back(cycle_nodes);
                    }
                } else if (visited[v] == 0) {
                    dfs(v);
                }
                weights_path.pop_back();
            }

            path.pop_back();
            visited[u] = 2;
        };

        for (const auto& c : cells) {
            if (visited[c.id] == 0) {
                dfs(c.id);
            }
        }

        return rep;
    }

    void enforce_lyapunov_stability(double max_allowable_gain = 0.95) {
        auto rep = check_lyapunov_stability();
        if (rep.is_stable || rep.unstable_cycles.empty()) return;

        for (const auto& cycle : rep.unstable_cycles) {
            std::unordered_set<uint32_t> cycle_set(cycle.begin(), cycle.end());
            for (auto& syn : synapses) {
                if (!syn.is_active) continue;
                if (cycle_set.count(syn.from_cell_id) && cycle_set.count(syn.to_cell_id)) {
                    syn.weight *= (max_allowable_gain / std::max(1.0, rep.max_loop_gain));
                    syn.initial_weight = syn.weight;
                }
            }
        }
        compile();
    }

    // 形式化风控连通性验证 (Inviolable Immune Path Reachability)
    bool verify_immune_connectivity() const {
        uint32_t immune_id = 0;
        bool has_immune_block = false;
        for (const auto& c : cells) {
            if (c.type == CellType::ACT_IMMUNE_BLOCK) {
                immune_id = c.id;
                has_immune_block = true;
                break;
            }
        }
        if (!has_immune_block) return false;

        std::unordered_map<uint32_t, std::vector<uint32_t>> adj;
        for (const auto& syn : synapses) {
            if (syn.is_active) {
                adj[syn.from_cell_id].push_back(syn.to_cell_id);
            }
        }

        for (const auto& c : cells) {
            if (is_receptor_cell(c.type)) {
                std::unordered_set<uint32_t> visited;
                std::vector<uint32_t> q = {c.id};
                visited.insert(c.id);
                size_t head = 0;
                while (head < q.size()) {
                    uint32_t curr = q[head++];
                    if (curr == immune_id) return true;
                    for (uint32_t nxt : adj[curr]) {
                        if (visited.insert(nxt).second) {
                            q.push_back(nxt);
                        }
                    }
                }
            }
        }
        return false;
    }

    // ========================================================================
    // 8. 原核到真核跃迁：超细胞共生封装 (Symbiotic Macro-Cell & Endosymbiosis)
    // ========================================================================
    struct SymbioticMacroCell {
        uint32_t macro_id{0};
        std::string label;
        std::vector<uint32_t> internal_cell_ids;
        std::vector<uint32_t> sensory_ports;
        std::vector<uint32_t> effector_ports;
    };
    std::vector<SymbioticMacroCell> macro_cells;

    bool form_symbiotic_macro_cell(const std::vector<uint32_t>& cluster_ids, const std::string& macro_label = "MacroCortex") {
        if (cluster_ids.size() < 2) return false;
        std::unordered_set<uint32_t> cluster_set(cluster_ids.begin(), cluster_ids.end());
        for (uint32_t cid : cluster_ids) {
            bool found = false;
            for (const auto& c : cells) {
                if (c.id == cid) { found = true; break; }
            }
            if (!found) return false;
        }

        SymbioticMacroCell mc;
        mc.macro_id = static_cast<uint32_t>(macro_cells.size() + 1);
        mc.label = macro_label;
        mc.internal_cell_ids = cluster_ids;

        for (const auto& syn : synapses) {
            if (!syn.is_active) continue;
            if (!cluster_set.count(syn.from_cell_id) && cluster_set.count(syn.to_cell_id)) {
                mc.sensory_ports.push_back(syn.to_cell_id);
            }
            if (cluster_set.count(syn.from_cell_id) && !cluster_set.count(syn.to_cell_id)) {
                mc.effector_ports.push_back(syn.from_cell_id);
            }
        }

        macro_cells.push_back(std::move(mc));
        return true;
    }
};

// ============================================================================
// 5.5 跨物种功能借用引擎与器官冷冻库 (Exaptation Engine & Organ Frozen Bank)
// 理论原则: 进化不造新轮子，全靠旧零件改用途 (如听小骨来自下颌骨)
// ============================================================================
struct FrozenOrgan {
    std::string name;
    std::string domain;
    std::vector<Cell> cells;
    std::vector<Synapse> internal_synapses;
    std::vector<uint32_t> input_ids;
    std::vector<uint32_t> output_ids;
};

class OrganFrozenBank {
public:
    static OrganFrozenBank& instance() {
        static OrganFrozenBank bank;
        return bank;
    }

    void deposit(const std::string& name, const std::string& domain,
                 const CellularOrganism& org, const std::vector<uint32_t>& cell_ids) {
        FrozenOrgan organ;
        organ.name = name;
        organ.domain = domain;
        std::unordered_set<uint32_t> cset(cell_ids.begin(), cell_ids.end());

        for (const auto& c : org.cells) {
            if (cset.count(c.id)) {
                organ.cells.push_back(c);
            }
        }
        for (const auto& s : org.synapses) {
            if (s.is_active && cset.count(s.from_cell_id) && cset.count(s.to_cell_id)) {
                organ.internal_synapses.push_back(s);
            }
        }
        if (!organ.cells.empty()) {
            organ.input_ids.push_back(organ.cells.front().id);
            organ.output_ids.push_back(organ.cells.back().id);
        }
        vault_[name] = std::move(organ);
    }

    bool exaptation_splice(const std::string& name, CellularOrganism& target,
                           uint32_t connect_from_sensor, uint32_t connect_to_actuator) {
        auto it = vault_.find(name);
        if (it == vault_.end()) return false;
        const auto& organ = it->second;

        uint32_t max_id = 0;
        for (const auto& c : target.cells) max_id = std::max(max_id, c.id);

        std::unordered_map<uint32_t, uint32_t> remap;
        for (const auto& c : organ.cells) {
            uint32_t new_id = ++max_id;
            remap[c.id] = new_id;
            Cell copied = c;
            copied.id = new_id;
            target.cells.push_back(copied);
        }

        for (const auto& s : organ.internal_synapses) {
            Synapse syn = s;
            syn.from_cell_id = remap[s.from_cell_id];
            syn.to_cell_id = remap[s.to_cell_id];
            target.synapses.push_back(syn);
        }

        if (!organ.input_ids.empty()) {
            uint32_t target_in = remap[organ.input_ids.front()];
            target.synapses.push_back({connect_from_sensor, target_in, 0, 1.0, true, 60.0f, -1.0f});
        }
        if (!organ.output_ids.empty()) {
            uint32_t target_out = remap[organ.output_ids.back()];
            target.synapses.push_back({target_out, connect_to_actuator, 0, 1.0, true, 60.0f, -1.0f});
        }

        target.compile();
        return true;
    }

    bool has_organ(const std::string& name) const {
        return vault_.find(name) != vault_.end();
    }

    std::vector<std::string> list_organs() const {
        std::vector<std::string> names;
        for (const auto& kv : vault_) names.push_back(kv.first);
        return names;
    }

private:
    OrganFrozenBank() {
        FrozenOrgan damper;
        damper.name = "schmitt_damping_column";
        damper.domain = "universal_cybernetics";
        Cell c1{1001, CellType::OP_EMA, 0.45, 0.0};
        Cell c2{1002, CellType::GATE_HYSTERESIS, -0.3, 0.3};
        damper.cells = {c1, c2};
        damper.internal_synapses = {{1001, 1002, 0, 1.2, true, 60.0f, -1.0f}};
        damper.input_ids = {1001};
        damper.output_ids = {1002};
        vault_[damper.name] = damper;
    }
    std::unordered_map<std::string, FrozenOrgan> vault_;
};

// ============================================================================
// 6. 形态发生进化引擎 (MorphogeneticEvolutionEngine)
// ============================================================================
class MorphogeneticEvolutionEngine {
public:
    // 行为特征空间新颖性档案 (Novelty Archive for Intrinsic Motivation / Open-Ended Curiosity)
    struct NoveltyArchive {
        std::vector<std::vector<double>> archive;
        size_t max_archive_size{200};
        size_t k_neighbors{5};

        double compute_novelty(const std::vector<double>& behavior_vec) const {
            if (archive.empty() || behavior_vec.empty()) return 1.0;
            std::vector<double> distances;
            distances.reserve(archive.size());
            for (const auto& a : archive) {
                double d2 = 0.0;
                size_t dim = std::min(behavior_vec.size(), a.size());
                for (size_t i = 0; i < dim; ++i) {
                    double diff = behavior_vec[i] - a[i];
                    d2 += diff * diff;
                }
                distances.push_back(std::sqrt(d2));
            }
            std::sort(distances.begin(), distances.end());
            size_t k = std::min(k_neighbors, distances.size());
            if (k == 0) return 1.0;
            double sum_dist = 0.0;
            for (size_t i = 0; i < k; ++i) sum_dist += distances[i];
            return sum_dist / static_cast<double>(k);
        }

        void add_behavior(const std::vector<double>& b) {
            if (b.empty()) return;
            archive.push_back(b);
            if (archive.size() > max_archive_size) {
                archive.erase(archive.begin());
            }
        }
    };

    explicit MorphogeneticEvolutionEngine(size_t population_size = 20, uint32_t seed = 42, SeedInitMode init_mode = SeedInitMode::HANDCRAFTED_PROGENITOR)
        : rng_(seed), population_size_(population_size) {
        constraint_cfg_.seed_mode = init_mode;
        init_population();
    }

    explicit MorphogeneticEvolutionEngine(size_t population_size, uint32_t seed, const EvolutionConstraintConfig& cfg)
        : rng_(seed), population_size_(population_size), constraint_cfg_(cfg) {
        init_population();
    }

    void init_population() {
        population_.clear();
        for (size_t i = 0; i < population_size_; ++i) {
            auto org = CellularOrganism::create_by_mode(constraint_cfg_.seed_mode, i + 1, static_cast<uint32_t>(rng_()));
            if (i > 0) {
                for (int m = 0; m < 3; ++m) mutate(org);
            }
            population_.push_back(std::move(org));
        }
    }

    void set_init_mode(SeedInitMode mode) {
        constraint_cfg_.seed_mode = mode;
        init_population();
    }

    // 种群并行评估 (尴尬并行: 个体间相互独立; 每线程克隆任务环境, 无数据竞争)
    // 演化主循环的标准入口 —— 1M 规模下评估吞吐随核数线性扩展。
    template <typename TaskT>
    std::vector<double> evaluate_population_parallel(
            TaskT& proto_env, const std::vector<uint32_t>& seeds,
            int max_steps, bool allow_plasticity, size_t n_threads = 0) {
        if (n_threads == 0) {
            n_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        }
        auto& pop = population_;
        std::vector<double> fits(pop.size(), 0.0);
        std::atomic<size_t> next{0};
        const size_t workers_n = std::min(n_threads, pop.size());
        std::vector<std::thread> workers;
        workers.reserve(workers_n);
        for (size_t t = 0; t < workers_n; ++t) {
            workers.emplace_back([&, t] {
                TaskT env = proto_env;   // 线程私有克隆 (每 episode reset 重播种, 确定性不丢)
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= pop.size()) break;
                    auto m = env.evaluate_organism(pop[i], seeds, max_steps, allow_plasticity);
                    fits[i] = m.mean_fitness;
                }
            });
        }
        for (auto& w : workers) w.join();
        return fits;
    }

    void set_constraint_config(const EvolutionConstraintConfig& cfg) {
        constraint_cfg_ = cfg;
        init_population();
    }

    const EvolutionConstraintConfig& get_constraint_config() const { return constraint_cfg_; }
    EvolutionConstraintConfig& constraint_config() { return constraint_cfg_; }
    SeedInitMode get_init_mode() const { return constraint_cfg_.seed_mode; }

    // ========================================================================
    // 9. 白垩纪大灭绝算子 (Chicxulub Extinction Operator)
    // 进化学原理: 恐龙不退场，哺乳类永为夜行小耗子。
    // 当种群长期停滞内卷时，抹杀排名前 80% 的成熟垄断者，保留边缘奇异变异体并施加相变冲击
    // ========================================================================
    struct ChicxulubImpactReport {
        uint32_t wiped_count{0};
        uint32_t survivors_count{0};
        double pre_extinction_best{0.0};
        double post_extinction_best{0.0};
        bool triggered{false};
    };

    ChicxulubImpactReport trigger_chicxulub_extinction(double wipeout_ratio = 0.8, double shock_scale = 2.5) {
        ChicxulubImpactReport report;
        if (population_.size() < 4) return report;

        report.pre_extinction_best = population_[0].fitness_score;
        size_t total = population_.size();
        size_t wipe_count = static_cast<size_t>(total * std::clamp(wipeout_ratio, 0.5, 0.95));
        size_t survivor_count = total - wipe_count;

        std::vector<CellularOrganism> survivors;
        for (size_t i = wipe_count; i < total; ++i) {
            survivors.push_back(std::move(population_[i]));
        }

        for (auto& s : survivors) {
            for (auto& syn : s.synapses) {
                std::normal_distribution<double> dist_shock(0.0, shock_scale * 0.2);
                syn.weight = std::clamp(syn.weight + dist_shock(rng_), -4.0, 4.0);
            }
            s.compile();
        }

        population_.clear();
        for (auto& s : survivors) {
            population_.push_back(s);
        }
        while (population_.size() < total) {
            std::uniform_int_distribution<size_t> dist_s(0, survivors.size() - 1);
            auto offspring = survivors[dist_s(rng_)];
            mutate(offspring);
            mutate(offspring);
            population_.push_back(std::move(offspring));
        }

        stagnation_generations_ = 0;
        adaptive_mutation_boost_ = shock_scale;
        report.wiped_count = static_cast<uint32_t>(wipe_count);
        report.survivors_count = static_cast<uint32_t>(survivor_count);
        report.triggered = true;
        return report;
    }

    // ── 生物变异操作 1: 细胞分裂增殖 (Mitosis / Add Cell — 开放式自组织演化) ──
    bool mutate_add_cell(CellularOrganism& org) {
        if ((constraint_cfg_.max_cells_limit != 0 && org.cells.size() >= constraint_cfg_.max_cells_limit) ||
            (constraint_cfg_.max_synapses_limit != 0 && org.synapses.size() + 2 > constraint_cfg_.max_synapses_limit)) {
            return mutate_add_synapse(org);
        }
        if (org.synapses.empty()) return mutate_add_synapse(org);
        std::uniform_int_distribution<size_t> dist_syn(0, org.synapses.size() - 1);
        size_t s_idx = dist_syn(rng_);
        auto& old_syn = org.synapses[s_idx];
        if (!old_syn.is_active) return false;

        static const CellType curated_9_candidates[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS,
            CellType::GATE_HYSTERESIS
        };

        static const CellType full_24_candidates[] = {
            CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_INTEGRAL,
            CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY,
            CellType::OP_RATIO, CellType::OP_ABS,
            CellType::OP_DELAY_N, CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
            CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS,
            CellType::GATE_AND, CellType::GATE_INHIBIT, CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX,
            CellType::PREDICT_SENSE_0, CellType::PREDICT_SENSE_1, CellType::ASSOCIATION_HUB
        };

        CellType new_type;
        if (constraint_cfg_.type_whitelist == TypeWhitelistMode::CURATED_9) {
            std::uniform_int_distribution<size_t> dist_type(0, sizeof(curated_9_candidates) / sizeof(curated_9_candidates[0]) - 1);
            new_type = curated_9_candidates[dist_type(rng_)];
        } else {
            std::uniform_int_distribution<size_t> dist_type(0, sizeof(full_24_candidates) / sizeof(full_24_candidates[0]) - 1);
            new_type = full_24_candidates[dist_type(rng_)];
        }

        // 解除骨架锁 (UNLOCKED) 约束下，允许演化出额外感受受体或效应器动作神经元
        if (constraint_cfg_.skeleton_lock == SkeletonLockMode::UNLOCKED) {
            std::uniform_real_distribution<double> dist_skel(0.0, 1.0);
            if (dist_skel(rng_) < 0.15) {
                static const CellType skel_candidates[] = {
                    CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
                    CellType::SENSE_RAW_INPUT_2, CellType::SENSE_RAW_INPUT_3,
                    CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE,
                    CellType::ACT_DEFENSIVE_RESET, CellType::ACT_IMMUNE_BLOCK,
                    CellType::SENSE_CHANNEL, CellType::ACT_CHANNEL
                };
                std::uniform_int_distribution<size_t> dist_sk(0, sizeof(skel_candidates) / sizeof(skel_candidates[0]) - 1);
                new_type = skel_candidates[dist_sk(rng_)];
            }
        }

        uint32_t new_id = 0;
        for (const auto& c : org.cells) new_id = std::max(new_id, c.id);
        new_id += 1;

        Cell new_cell;
        new_cell.id = new_id;
        new_cell.type = new_type;

        // 空间场发育引导: 新细胞诞生于原突触母体空间连线中点，并注入随机微扰
        size_t from_idx = 0, to_idx = 0;
        for (size_t i = 0; i < org.cells.size(); ++i) {
            if (org.cells[i].id == old_syn.from_cell_id) from_idx = i;
            if (org.cells[i].id == old_syn.to_cell_id) to_idx = i;
        }

        std::normal_distribution<float> dist_pos(0.0f, 15.0f);
        new_cell.x = (org.cells[from_idx].x + org.cells[to_idx].x) * 0.5f + dist_pos(rng_);
        new_cell.y = (org.cells[from_idx].y + org.cells[to_idx].y) * 0.5f + dist_pos(rng_);
        new_cell.z = (org.cells[from_idx].z + org.cells[to_idx].z) * 0.5f + dist_pos(rng_);

        std::uniform_real_distribution<double> dist_param(-1.0, 1.0);
        new_cell.param1 = dist_param(rng_);
        new_cell.param2 = dist_param(rng_);

        org.cells.push_back(new_cell);

        // 原突触断裂与重接：A -> NewCell -> B (保持原有权重的因果连续性)
        old_syn.is_active = false;

        Synapse syn1;
        syn1.from_cell_id = old_syn.from_cell_id;
        syn1.to_cell_id = new_id;
        syn1.to_port = 0;
        syn1.weight = old_syn.weight;
        syn1.initial_weight = syn1.weight;
        syn1.is_active = true;

        Synapse syn2;
        syn2.from_cell_id = new_id;
        syn2.to_cell_id = old_syn.to_cell_id;
        syn2.to_port = old_syn.to_port;
        syn2.weight = 1.0;
        syn2.initial_weight = syn2.weight;
        syn2.is_active = true;

        org.synapses.push_back(syn1);
        org.synapses.push_back(syn2);

        org.compile();
        return true;
    }

    // ── 生物变异操作 2: 突触跨界重连 (Synaptic Rewiring & Morphogenetic Spatial Wiring) ──
    bool mutate_add_synapse(CellularOrganism& org) {
        if (constraint_cfg_.max_synapses_limit != 0 && org.synapses.size() >= constraint_cfg_.max_synapses_limit) {
            return mutate_parameters(org);
        }
        if (org.cells.size() < 2) return false;
        for (int retry = 0; retry < 12; ++retry) {
            std::uniform_int_distribution<size_t> dist_cell(0, org.cells.size() - 1);
            size_t idx_a = dist_cell(rng_);
            size_t idx_b = dist_cell(rng_);
            if (idx_a == idx_b) continue;

            if (is_receptor_cell(org.cells[idx_b].type)) {
                continue;
            }

            // 空间场发育指引: 3D 物理空间欧氏距离越近，成键概率越高 (自发形成小世界皮层柱)
            float dx = org.cells[idx_b].x - org.cells[idx_a].x;
            float dy = org.cells[idx_b].y - org.cells[idx_a].y;
            float dz = org.cells[idx_b].z - org.cells[idx_a].z;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double conn_prob = std::exp(-dist / 80.0f) + 0.15;
            std::uniform_real_distribution<double> dist_p(0.0, 1.0);
            if (dist_p(rng_) > conn_prob && retry < 8) continue;

            uint32_t from_id = org.cells[idx_a].id;
            uint32_t to_id = org.cells[idx_b].id;

            std::uniform_real_distribution<double> dist_weight(-2.0, 2.0);
            std::uniform_int_distribution<int> dist_port(0, 1);

            Synapse new_syn{from_id, to_id, static_cast<uint8_t>(dist_port(rng_)), dist_weight(rng_), true, 60.0f, -1.0f};
            new_syn.initial_weight = new_syn.weight;
            new_syn.hebbian_rate = 0.005;
            new_syn.hebbian_decay = 0.02;

            org.synapses.push_back(new_syn);
            org.compile();
            return true;
        }
        return false;
    }

    // ── 生物变异操作 3: 细胞参数微调 + 突触可塑性 (Metabolic Drift & Plasticity) ──
    bool mutate_parameters(CellularOrganism& org) {
        if (org.cells.empty()) return false;
        std::uniform_int_distribution<size_t> dist_cell(0, org.cells.size() - 1);
        auto& c = org.cells[dist_cell(rng_)];

        std::normal_distribution<double> dist_noise(0.0, 0.05);
        c.param1 = std::clamp(c.param1 + dist_noise(rng_), -5.0, 5.0);
        c.param2 = std::clamp(c.param2 + dist_noise(rng_), -5.0, 5.0);

        if (!org.synapses.empty()) {
            std::uniform_int_distribution<size_t> dist_syn(0, org.synapses.size() - 1);
            auto& syn = org.synapses[dist_syn(rng_)];
            std::normal_distribution<double> dist_w(0.0, 0.15);
            syn.weight = std::clamp(syn.weight + dist_w(rng_), -3.0, 3.0);
            syn.initial_weight = syn.weight;

            // 突触学习率微调漂移 (慢时标系统发育演化)
            std::normal_distribution<double> dist_h(0.0, 0.002);
            syn.hebbian_rate = std::clamp(syn.hebbian_rate + dist_h(rng_), 0.0, 0.05);
            // 参数就地更新无需重复全图 compile，极大加速变异热路径
        }
        return true;
    }

    // ── 生物自我净化: 细胞凋亡与剪枝 (Apoptosis / Pruning) ──
    void prune_apoptosis(CellularOrganism& org) {
        std::unordered_set<uint32_t> useful_cells;
        for (const auto& c : org.cells) {
            if (is_effector_cell(c.type) ||
                c.type == CellType::PREDICT_SENSE_0 ||
                c.type == CellType::PREDICT_SENSE_1) {
                useful_cells.insert(c.id);
            }
        }

        bool expanded = true;
        while (expanded) {
            expanded = false;
            for (const auto& syn : org.synapses) {
                if (syn.is_active && useful_cells.count(syn.to_cell_id) && !useful_cells.count(syn.from_cell_id)) {
                    useful_cells.insert(syn.from_cell_id);
                    expanded = true;
                }
            }
        }

        if (constraint_cfg_.skeleton_lock == SkeletonLockMode::LOCKED) {
            // 受限骨架模式：受体与效应器器官细胞永远免疫凋亡，只凋亡中间无用悬空计算节点
            org.cells.erase(
                std::remove_if(org.cells.begin(), org.cells.end(), [&](const Cell& c) {
                    if (is_receptor_cell(c.type) ||
                        is_effector_cell(c.type) ||
                        c.type == CellType::PREDICT_SENSE_0 ||
                        c.type == CellType::PREDICT_SENSE_1) return false;
                    return !useful_cells.count(c.id);
                }),
                org.cells.end()
            );
        } else {
            // 自由形态发生模式：无用受体亦可凋亡
            org.cells.erase(
                std::remove_if(org.cells.begin(), org.cells.end(), [&](const Cell& c) {
                    return !useful_cells.count(c.id);
                }),
                org.cells.end()
            );
        }

        std::unordered_set<uint32_t> live_ids;
        for (const auto& c : org.cells) live_ids.insert(c.id);

        org.synapses.erase(
            std::remove_if(org.synapses.begin(), org.synapses.end(), [&](const Synapse& s) {
                return !s.is_active || !live_ids.count(s.from_cell_id) || !live_ids.count(s.to_cell_id);
            }),
            org.synapses.end()
        );

        org.compile();
    }

    // ── 生物变异操作 4: 力敏转导定向有丝分裂与皮层沟回拱起 (Mechanosensitive Mitosis) ──
    bool mutate_mechanosensitive_mitosis(CellularOrganism& org) {
        if ((constraint_cfg_.max_cells_limit != 0 && org.cells.size() >= constraint_cfg_.max_cells_limit) ||
            (constraint_cfg_.max_synapses_limit != 0 && org.synapses.size() + 2 > constraint_cfg_.max_synapses_limit) ||
            org.cells.size() < 5) return false;

        // 寻找综合力敏应变最高的候选母细胞
        size_t best_idx = 0;
        float max_strain = -1.0f;
        for (size_t i = 4; i < org.cells.size(); ++i) {
            if (org.cells[i].mitosis_cooldown == 0 && org.cells[i].get_total_strain() > max_strain) {
                max_strain = org.cells[i].get_total_strain();
                best_idx = i;
            }
        }
        if (max_strain < 0.0f) {
            std::uniform_int_distribution<size_t> dist_rand(4, org.cells.size() - 1);
            best_idx = dist_rand(rng_);
        }

        uint32_t parent_id = org.cells[best_idx].id;
        std::vector<size_t> incoming_syns;
        for (size_t k = 0; k < org.synapses.size(); ++k) {
            if (org.synapses[k].is_active && org.synapses[k].to_cell_id == parent_id) {
                incoming_syns.push_back(k);
            }
        }
        if (incoming_syns.empty()) return false;

        std::uniform_int_distribution<size_t> dist_syn(0, incoming_syns.size() - 1);
        size_t syn_idx = incoming_syns[dist_syn(rng_)];
        auto& target_syn = org.synapses[syn_idx];

        // 沿法向与应力方向拱起生成 3D 脑回
        const auto& parent = org.cells[best_idx];
        std::normal_distribution<float> dist_noise(0.0f, 3.0f);
        std::uniform_real_distribution<float> dist_z(15.0f, 30.0f);

        float new_x = parent.x + dist_noise(rng_);
        float new_y = parent.y + dist_noise(rng_);
        float new_z = parent.z + dist_z(rng_);

        uint32_t new_id = 0;
        for (const auto& cell : org.cells) new_id = std::max(new_id, cell.id);
        ++new_id;
        CellType new_type = CellType::OP_EMA;
        double new_param1 = 0.5;
        double new_param2 = 0.0;
        
        std::uniform_int_distribution<int> dist_type(0, 5);
        int t_idx = dist_type(rng_);
        if (t_idx == 0) {
            new_type = CellType::OP_EMA;
            std::uniform_real_distribution<double> dist_alpha(0.35, 0.85);
            new_param1 = dist_alpha(rng_);
        } else if (t_idx == 1) {
            new_type = CellType::OP_DIFF;
            new_param1 = 0.05;
        } else if (t_idx == 2) {
            new_type = CellType::GATE_HYSTERESIS;
            new_param1 = 0.5;
            new_param2 = -0.5;
        } else if (t_idx == 3) {
            new_type = CellType::OP_INTEGRAL;
            new_param1 = 0.02;
        } else if (t_idx == 4) {
            new_type = CellType::GATE_DEADZONE;
            new_param1 = 0.05;
        } else {
            new_type = CellType::OP_SUM;
            new_param1 = 0.0;
        }

        Cell new_cell{new_id, new_type, new_param1, new_param2, 0.0, 0.0, false, 0.0, 0, 0, new_x, new_y, new_z};
        new_cell.mitosis_cooldown = 15;
        org.cells[best_idx].mitosis_cooldown = 15;
        org.cells[best_idx].informational_strain *= 0.1f; // 卸载应变
        org.cells.push_back(new_cell);

        // 同源保真拆解突触: from -> new (1.0), new -> to (old_weight)
        target_syn.is_active = false;
        Synapse syn_in{target_syn.from_cell_id, new_id, 0, 1.0, true, 60.0f, -1.0f};
        syn_in.initial_weight = 1.0;
        Synapse syn_out{new_id, parent_id, target_syn.to_port, target_syn.weight, true, 60.0f, -1.0f};
        syn_out.initial_weight = target_syn.weight;

        org.synapses.push_back(syn_in);
        org.synapses.push_back(syn_out);

        org.compile();
        return true;
    }

    // 综合变异入口 (具备事务性原子回滚保障)
    bool mutate(CellularOrganism& org) {
        if (constraint_cfg_.viability_filter && !constraint_cfg_.viability_filter(org)) {
            return false; // 外围 Task 生存力准入拦截 (业务层决定何时不许繁殖)
        }
        // 预算预检: 若配置了非零硬上限且已超预算，跳过无效变异
        if ((constraint_cfg_.max_cells_limit != 0 && org.cells.size() > constraint_cfg_.max_cells_limit) ||
            (constraint_cfg_.max_synapses_limit != 0 && org.synapses.size() > constraint_cfg_.max_synapses_limit)) {
            return false;
        }
        CellularOrganism snapshot = org; // 变异前事务快照
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        bool any_mutated = false;

        // 1. 若当前有机体突触不足 4 条 (如原始胚胎冷启动)，强制优先建立基础突触连接
        size_t active_syns = 0;
        for (const auto& s : org.synapses) if (s.is_active) active_syns++;
        if (active_syns < 4) {
            for (int i = 0; i < 3; ++i) {
                if (mutate_add_synapse(org)) any_mutated = true;
            }
        }

        // 2. 突触权重与细胞参数微调 (细粒度梯度微扰，随停滞自适应增强)
        if (dist(rng_) < std::min(0.95, constraint_cfg_.fast_mutation_rate * adaptive_mutation_boost_)) {
            if (mutate_parameters(org)) any_mutated = true;
        }

        // 3. 突触结构重连 / 新增突触
        if (dist(rng_) < std::min(0.85, constraint_cfg_.medium_mutation_rate * adaptive_mutation_boost_)) {
            if (mutate_add_synapse(org)) any_mutated = true;
        }

        // 4. 力敏转导定向有丝分裂 (或传统随机有丝分裂)
        if (constraint_cfg_.enable_mechanotransduction) {
            if (dist(rng_) < std::min(0.70, constraint_cfg_.slow_mutation_rate * adaptive_mutation_boost_)) {
                if (mutate_mechanosensitive_mitosis(org)) any_mutated = true;
            }
        } else {
            if (dist(rng_) < std::min(0.70, constraint_cfg_.slow_mutation_rate * adaptive_mutation_boost_)) {
                if (mutate_add_cell(org)) any_mutated = true;
            }
        }

        // 5. 细胞自我凋亡与无用分支净化
        if (dist(rng_) < 0.05) {
            prune_apoptosis(org);
            any_mutated = true;
        }

        // 变异后事务校验：编译完整性、资源上限约束、功能依赖契约
        bool compile_ok = org.compile();
        if (compile_ok) {
            org.enforce_lyapunov_stability();
        }
        bool budget_ok = ((constraint_cfg_.max_cells_limit == 0 ||
                           org.cells.size() <= constraint_cfg_.max_cells_limit) &&
                          (constraint_cfg_.max_synapses_limit == 0 ||
                           org.synapses.size() <= constraint_cfg_.max_synapses_limit));
        bool contract_ok = true;
        if (constraint_cfg_.enable_dependency_guard) {
            contract_ok = constraint_cfg_.topology_contract_guard ?
                          constraint_cfg_.topology_contract_guard(org) :
                          (!constraint_cfg_.viability_contract_filter ||
                           constraint_cfg_.viability_contract_filter(org.evaluate_coverage_contract()));
        }

        if (constraint_cfg_.enable_transaction_rollback) {
            if (!compile_ok || !budget_ok || !contract_ok) {
                org = std::move(snapshot); // 原子回滚至变异前状态
                return false;
            }
        } else if (constraint_cfg_.enable_dependency_guard && !contract_ok) {
            org = std::move(snapshot);
            return false;
        }

        return any_mutated;
    }

    // ── 种群世代演化 (Evolve Next Generation — 无上限开放式自发演化) ──
    void evolve_generation() {
        // 1. 动态代谢能量平衡 (Dynamic Metabolic Energy Equilibrium):
        // 采用热力学乘性软阻尼，严禁直接相减负分导致复杂大脑被小个体逆淘汰
        if (constraint_cfg_.enable_dynamic_metabolism && constraint_cfg_.basal_metabolic_cost > 0.0) {
            for (auto& org : population_) {
                double raw_cost = static_cast<double>(org.cells.size()) * constraint_cfg_.basal_metabolic_cost +
                                  static_cast<double>(org.synapses.size()) * constraint_cfg_.synaptic_metabolic_cost;
                // 软阻尼因子: 能耗惩罚平滑受限在 [0, 0.20]，绝对不破坏正向适应度
                double metabolic_penalty = 0.20 * std::tanh(raw_cost * 1e-4);
                org.fitness_score = org.fitness_score * (1.0 - metabolic_penalty);
            }
        }

        // 2. 若启用了新颖性好奇心驱动，综合内在与外在动机评分
        if (constraint_cfg_.fitness_driver == FitnessDriverMode::NOVELTY_SEARCH ||
            constraint_cfg_.fitness_driver == FitnessDriverMode::HYBRID_CURIOSITY) {
            for (auto& org : population_) {
                if (constraint_cfg_.enable_dependency_guard) {
                    bool valid = constraint_cfg_.topology_contract_guard ?
                                 constraint_cfg_.topology_contract_guard(org) :
                                 (!constraint_cfg_.viability_contract_filter ||
                                  constraint_cfg_.viability_contract_filter(org.evaluate_coverage_contract()));
                    if (!valid) {
                        org.fitness_score = -1e12;
                        continue;
                    }
                }
                std::vector<double> b_vec;
                if (constraint_cfg_.behavior_extractor) {
                    b_vec = constraint_cfg_.behavior_extractor(org);
                } else {
                    b_vec = {
                        static_cast<double>(org.cells.size()),
                        static_cast<double>(org.synapses.size()),
                        org.fitness_score
                    };
                }
                double nov = novelty_archive_.compute_novelty(b_vec);
                novelty_archive_.add_behavior(b_vec);

                if (constraint_cfg_.fitness_driver == FitnessDriverMode::NOVELTY_SEARCH) {
                    org.fitness_score = nov * 100.0;
                } else if (constraint_cfg_.fitness_driver == FitnessDriverMode::HYBRID_CURIOSITY) {
                    double alpha = constraint_cfg_.novelty_weight;
                    org.fitness_score = (1.0 - alpha) * org.fitness_score + alpha * (nov * 50.0);
                }
            }
        }

        // Apply the dependency guard after every score transform, including novelty.
        if (constraint_cfg_.enable_dependency_guard) {
            for (auto& org : population_) {
                bool valid = constraint_cfg_.topology_contract_guard ?
                             constraint_cfg_.topology_contract_guard(org) :
                             (!constraint_cfg_.viability_contract_filter ||
                              constraint_cfg_.viability_contract_filter(org.evaluate_coverage_contract()));
                if (!valid) {
                    org.fitness_score = -1e12;
                }
            }
        }

        std::sort(population_.begin(), population_.end(), [](const CellularOrganism& a, const CellularOrganism& b) {
            return a.fitness_score > b.fitness_score;
        });

        // 3.0 鲍德温可塑性基因跨代固化 (Baldwin Effect Crystallization)
        // 仅对顶层精英个体、且每 10 代周期性固化，保护整体种群塑性隔离
        uint32_t current_gen = population_.empty() ? 0 : population_[0].generation;
        if (constraint_cfg_.enable_baldwin_crystallization && current_gen % 10 == 0) {
            size_t elite_count = std::max<size_t>(1, population_.size() / 10);
            for (size_t i = 0; i < elite_count && i < population_.size(); ++i) {
                population_[i].crystallize_plasticity(constraint_cfg_.crystallization_rate);
            }
        }

        // 自适应停滞检测 (Plateau Stagnation Detection)
        if (!population_.empty()) {
            double current_best = population_[0].fitness_score;
            if (current_best > best_historical_fitness_ + 1e-4) {
                best_historical_fitness_ = current_best;
                stagnation_generations_ = 0;
                adaptive_mutation_boost_ = 1.0;
            } else {
                stagnation_generations_++;
                if (stagnation_generations_ > 15) {
                    // 陷入停滞高原: 激变模式 (Hyper-mutation burst 动态加温)
                    adaptive_mutation_boost_ = std::min(2.5, 1.0 + (stagnation_generations_ - 15) * 0.08);
                }
            }
        }

        std::vector<CellularOrganism> next_gen;

        // 3.1 保留全局顶级精英 (Top Elites)
        size_t elite_count = std::max<size_t>(1, population_size_ / 8); // 40 个体保留前 5 个绝对优胜者
        for (size_t i = 0; i < elite_count && i < population_.size(); ++i) {
            next_gen.push_back(population_[i]);
        }

        // 3.2 注入客卿移民 (Immigrants - 保持种群多样性)
        size_t immigrant_count = static_cast<size_t>(std::round(population_size_ * constraint_cfg_.immigrant_rate));
        if (constraint_cfg_.immigrant_rate > 1e-6 && immigrant_count == 0 && population_size_ >= 8) {
            immigrant_count = 1;
        }
        immigrant_count = std::min<size_t>(immigrant_count, population_size_ / 4);
        for (size_t k = 0; k < immigrant_count && next_gen.size() < population_size_; ++k) {
            auto immigrant = CellularOrganism::create_by_mode(
                constraint_cfg_.seed_mode, 
                static_cast<uint32_t>(next_gen.size() + 1), 
                static_cast<uint32_t>(rng_())
            );
            for (int m = 0; m < 2 + static_cast<int>(k % 3); ++m) mutate(immigrant);
            immigrant.generation = population_.empty() ? 1 : population_[0].generation + 1;
            immigrant.lineage_name = "Immigrant-Gen" + std::to_string(immigrant.generation);
            next_gen.push_back(std::move(immigrant));
        }

        // 3.3 锦标赛选择 (Tournament Selection) 从全种群挑选父母繁殖，保持多样性
        std::uniform_int_distribution<size_t> dist_tour(0, population_.size() - 1);
        while (next_gen.size() < population_size_) {
            size_t c1 = dist_tour(rng_);
            size_t c2 = dist_tour(rng_);
            size_t c3 = dist_tour(rng_);
            size_t p_idx = (population_[c1].fitness_score >= population_[c2].fitness_score) ? 
                           ((population_[c1].fitness_score >= population_[c3].fitness_score) ? c1 : c3) :
                           ((population_[c2].fitness_score >= population_[c3].fitness_score) ? c2 : c3);

            CellularOrganism child = population_[p_idx];
            child.parent_organism_id = population_[p_idx].organism_id;
            child.organism_id = static_cast<uint32_t>(next_gen.size() + 1);
            child.generation = population_[p_idx].generation + 1;
            child.lineage_name = "Apex-Gen" + std::to_string(child.generation);
            mutate(child);
            next_gen.push_back(std::move(child));
        }

        population_ = std::move(next_gen);
    }

    CellularOrganism& get_champion() { return population_[0]; }
    const CellularOrganism& get_champion() const { return population_[0]; }
    const std::vector<CellularOrganism>& get_population() const { return population_; }
    std::vector<CellularOrganism>& get_population_mut() { return population_; }
    const std::vector<CellularOrganism>& population() const { return population_; }
    std::vector<CellularOrganism>& population() { return population_; }
    uint32_t get_stagnation_generations() const { return stagnation_generations_; }
    double get_adaptive_mutation_boost() const { return adaptive_mutation_boost_; }
    NoveltyArchive& novelty_archive() { return novelty_archive_; }

    // ── 种群级全息检查点保存 (Population Checkpoint Save) ──
    bool save_population_checkpoint(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) return false;
        ofs << "{\n";
        ofs << "  \"population_size\": " << population_.size() << ",\n";
        ofs << "  \"stagnation_generations\": " << stagnation_generations_ << ",\n";
        ofs << "  \"best_historical_fitness\": " << best_historical_fitness_ << ",\n";
        ofs << "  \"organisms\": [\n";
        for (size_t i = 0; i < population_.size(); ++i) {
            ofs << population_[i].to_json() << (i + 1 < population_.size() ? ",\n" : "\n");
        }
        ofs << "  ]\n";
        ofs << "}\n";
        return true;
    }

    // ── 种群级全息检查点加载与断点续演化 (Population Checkpoint Load & Warm Resume) ──
    bool load_population_checkpoint(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;
        std::stringstream buffer;
        buffer << ifs.rdbuf();
        std::string content = buffer.str();

        auto orgs_pos = content.find("\"organisms\"");
        if (orgs_pos == std::string::npos) return false;

        auto open_bracket = content.find("[", orgs_pos);
        auto close_bracket = content.rfind("]");
        if (open_bracket == std::string::npos || close_bracket == std::string::npos) return false;

        std::string list_str = content.substr(open_bracket + 1, close_bracket - open_bracket - 1);
        std::vector<CellularOrganism> loaded_pop;

        size_t depth = 0;
        size_t start = 0;
        bool in_obj = false;

        for (size_t i = 0; i < list_str.size(); ++i) {
            if (list_str[i] == '{') {
                if (depth == 0) { start = i; in_obj = true; }
                depth++;
            } else if (list_str[i] == '}') {
                depth--;
                if (depth == 0 && in_obj) {
                    std::string obj_str = list_str.substr(start, i - start + 1);
                    auto org = CellularOrganism::from_json(obj_str);
                    if (!org.cells.empty()) loaded_pop.push_back(std::move(org));
                    in_obj = false;
                }
            }
        }

        if (loaded_pop.empty()) return false;
        population_ = std::move(loaded_pop);
        population_size_ = population_.size();
        return true;
    }

private:
    std::mt19937 rng_;
    size_t population_size_{20};
    EvolutionConstraintConfig constraint_cfg_;
    std::vector<CellularOrganism> population_;
    NoveltyArchive novelty_archive_;
    uint32_t stagnation_generations_{0};
    double best_historical_fitness_{-1e9};
    double adaptive_mutation_boost_{1.0};
};

} // namespace kun

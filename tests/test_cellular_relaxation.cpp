#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_relaxation.hpp"

using namespace kun;

inline Cell make_cell(uint32_t id, CellType type, double param1 = 1.0, double param2 = 0.0) {
    Cell c;
    c.id = id;
    c.type = type;
    c.param1 = param1;
    c.param2 = param2;
    return c;
}

inline Synapse make_synapse(uint32_t from_id, uint32_t to_id, uint8_t port, double weight) {
    Synapse s;
    s.from_cell_id = from_id;
    s.to_cell_id = to_id;
    s.to_port = port;
    s.weight = weight;
    s.initial_weight = weight;
    s.is_active = true;
    return s;
}

// 1. 验证 K=1 动态松弛与标准单步前向推理的精确一致性 (Backward Compatibility & Identity at K=1)
void test_relaxation_identity_k1() {
    std::cout << "[Relaxation Test 1] 验证 K=1 思考步与原生前向反射的精确一致性 (Identity at K=1)...\n";
    CellularOrganism org1;
    org1.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org1.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_1, 1.0));
    org1.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org1.cells.push_back(make_cell(3, CellType::OP_EMA, 0.4));
    org1.cells.push_back(make_cell(4, CellType::ACT_PRIMARY_POSITIVE, 1.0)); // ch 0
    org1.cells.push_back(make_cell(5, CellType::ACT_PRIMARY_NEGATIVE, 1.0)); // ch 1

    org1.synapses.push_back(make_synapse(0, 2, 0, 1.5));
    org1.synapses.push_back(make_synapse(1, 2, 1, -0.5));
    org1.synapses.push_back(make_synapse(2, 3, 0, 0.8));
    org1.synapses.push_back(make_synapse(3, 4, 0, 1.2));
    org1.synapses.push_back(make_synapse(2, 5, 0, -1.0));
    org1.compile();

    CellularOrganism org2 = org1; // 完全同构副本

    double inps[4] = {0.75, 0.25, 0.0, 0.0};
    auto act_standard = org1.forward(inps, false);
    RelaxationTelemetry telem;
    auto act_relax1 = org2.forward_with_relaxation(inps, 1, 4, 1e-6, false, &telem);

    assert(std::abs(act_standard.positive_action - act_relax1.positive_action) < 1e-12);
    assert(std::abs(act_standard.negative_action - act_relax1.negative_action) < 1e-12);
    assert(std::abs(act_standard.defensive_reset - act_relax1.defensive_reset) < 1e-12);
    assert(std::abs(act_standard.thought_energy - act_relax1.thought_energy) < 1e-12);
    assert(telem.actual_steps == 1);
    assert(telem.bibo_stable == true);
    std::cout << "  ↳ K=1 行动输出精确匹配: Positive=" << act_relax1.positive_action 
              << ", Negative=" << act_relax1.negative_action << "\n";
}

// 2. 验证纯无环前向图在松弛步下的瞬时不动点收敛 (Instant Convergence on DAGs)
void test_pure_dag_instant_convergence() {
    std::cout << "[Relaxation Test 2] 验证纯前向拓扑在动态松弛下的单步不动点固化...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    org.synapses.push_back(make_synapse(0, 1, 0, 2.0));
    org.synapses.push_back(make_synapse(1, 2, 0, 1.5));
    org.compile();

    double inps[4] = {1.0, 0.0, 0.0, 0.0};
    RelaxationTelemetry telem;
    RelaxationConfig cfg;
    cfg.max_steps = 5;
    cfg.convergence_tol = 1e-8;
    cfg.early_stop = true;

    org.forward_with_relaxation(inps, 4, cfg, &telem);

    // 对于纯 DAG，第 1 步计算后状态即达到严格不动点，第 2 步变动严格为 0
    assert(telem.actual_steps <= 2);
    assert(telem.converged == true);
    assert(telem.step_deltas.size() >= 2);
    assert(telem.step_deltas[1] < 1e-12);
    std::cout << "  ↳ DAG 动态松弛步数: " << telem.actual_steps << " (第 2 步残差=" << telem.step_deltas[1] << ")\n";
}

// 3. 验证真实递归拓扑李雅普诺夫稳态吸引子收敛与指数收缩 (Dynamical Attractor Convergence & Contraction)
void test_dynamical_attractor_convergence() {
    std::cout << "[Relaxation Test 3] 验证真实时序循环反馈李雅普诺夫吸引子收敛与指数收缩...\n";
    CellularOrganism org;
    // 0: 感受器
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    // 1: 隐层叠加器 (OP_SUM)
    org.cells.push_back(make_cell(1, CellType::OP_SUM, 1.0));
    // 2: 隐层动态衰减器 (OP_EMA, alpha = 0.5)
    org.cells.push_back(make_cell(2, CellType::OP_EMA, 0.5));
    // 3: 动作效应器
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    // 前向突触
    org.synapses.push_back(make_synapse(0, 1, 0, 1.0));
    org.synapses.push_back(make_synapse(1, 2, 0, 0.8));
    // 反馈突触: 2 -> 1, 构成闭环循环反馈回路!
    // 环路静态增益: 0.8 * 0.5 = 0.4 < 1.0 (严格李雅普诺夫稳定吸引子)
    org.synapses.push_back(make_synapse(2, 1, 1, 0.5));
    org.synapses.push_back(make_synapse(2, 3, 0, 1.0));
    org.compile();

    // 理论不动点分析:
    // Input = 1.0
    // V_1 = 1.0 + 0.5 * V_2
    // V_2 = 0.8 * V_1
    // => V_1 = 1.0 + 0.4 * V_1 => V_1* = 1.0 / 0.6 = 1.666667
    // => V_2* = 0.8 * 1.666667 = 1.333333
    // => Act 3* = 1.333333

    double inps[4] = {1.0, 0.0, 0.0, 0.0};
    RelaxationTelemetry telem;
    RelaxationConfig cfg;
    cfg.max_steps = 36;
    cfg.convergence_tol = 1e-4;
    cfg.early_stop = false; // 观察完整轨迹

    auto acts = org.forward_with_relaxation(inps, 4, cfg, &telem);

    // 验证残差单调递减
    assert(telem.step_deltas.size() == 36);
    for (size_t i = 1; i < telem.step_deltas.size(); ++i) {
        assert(telem.step_deltas[i] <= telem.step_deltas[i - 1] + 1e-9);
    }

    // 验证终态残差收敛至容差之内
    assert(telem.step_deltas.back() < 1e-4);
    assert(telem.converged == true);
    assert(telem.spectral_contraction < 1.0);

    // 验证数值解逼近理论解析不动点
    assert(std::abs(org.cells[1].output_val - (1.0 / 0.6)) < 1e-2);
    assert(std::abs(acts.positive_action - (0.8 / 0.6)) < 1e-2);

    std::cout << "  ↳ 初始能量: " << telem.initial_energy << " -> 最终吸引子能量: " << telem.final_energy << "\n";
    std::cout << "  ↳ 步数 1 残差: " << telem.step_deltas[0] << ", 最终步残差: " << telem.step_deltas.back() << "\n";
    std::cout << "  ↳ 理论解析不动点: 1.333333, 实测终态输出: " << acts.positive_action 
              << " (相对误差: " << std::abs(acts.positive_action - 1.333333) / 1.333333 * 100.0 << "%)\n";
}

// 4. 验证李雅普诺夫 BIBO 有界性与发散阻断守卫 (Lyapunov BIBO Bounded Stability Guard)
void test_lyapunov_bibo_stability() {
    std::cout << "[Relaxation Test 4] 验证发散环路下的李雅普诺夫 BIBO 物理硬阻断与有界性保护...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    // 构建一个发散正反馈环路: 1 -> 2 (w=2.0), 2 -> 1 (w=2.0)
    // 环路增益 = 4.0 >> 1.0! 若无守卫必数值爆炸至无穷大
    org.synapses.push_back(make_synapse(0, 1, 0, 1.0));
    org.synapses.push_back(make_synapse(1, 2, 0, 2.0));
    org.synapses.push_back(make_synapse(2, 1, 1, 2.0));
    org.synapses.push_back(make_synapse(2, 3, 0, 1.0));
    org.compile();

    double inps[4] = {1.0, 0.0, 0.0, 0.0};
    RelaxationTelemetry telem;
    RelaxationConfig cfg;
    cfg.max_steps = 30;
    cfg.bibo_bound = 10.0; // 设定 10.0 BIBO 硬护栏 (小于硬件饱和边界 16.0)

    auto acts = org.forward_with_relaxation(inps, 4, cfg, &telem);

    // 验证未发生 NaN / Inf 浮点溢出
    assert(!std::isnan(acts.positive_action));
    assert(!std::isinf(acts.positive_action));
    for (const auto& c : org.cells) {
        assert(!std::isnan(c.output_val));
        assert(!std::isinf(c.output_val));
        assert(std::abs(c.output_val) <= cfg.bibo_bound + 1e-6);
    }
    // 发散被检出并标记
    assert(telem.bibo_stable == false);
    std::cout << "  ↳ BIBO 安全护栏成功拦截发散: 最大细胞电位已被死死钳位在界内 (|V| <= " 
              << cfg.bibo_bound << ")\n";
}

// 5. 验证确定性无抖动可复现性 (Deterministic Bit-For-Bit Execution)
void test_deterministic_execution() {
    std::cout << "[Relaxation Test 5] 验证纳秒级确定性执行与无抖动位级可复现性...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::OP_EMA, 0.3));
    org.cells.push_back(make_cell(2, CellType::OP_DIFF, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    org.synapses.push_back(make_synapse(0, 1, 0, 1.2));
    org.synapses.push_back(make_synapse(1, 2, 0, 0.6));
    org.synapses.push_back(make_synapse(2, 1, 0, 0.4));
    org.synapses.push_back(make_synapse(1, 3, 0, 1.0));
    org.compile();

    CellularOrganism clone1 = org;
    CellularOrganism clone2 = org;

    double inps[4] = {0.8421, 0.1234, -0.5678, 0.9999};
    RelaxationTelemetry telem1, telem2;

    auto act1 = clone1.forward_with_relaxation(inps, 8, 4, 1e-6, false, &telem1);
    auto act2 = clone2.forward_with_relaxation(inps, 8, 4, 1e-6, false, &telem2);

    assert(act1.positive_action == act2.positive_action);
    assert(act1.negative_action == act2.negative_action);
    assert(act1.defensive_reset == act2.defensive_reset);
    assert(act1.thought_energy == act2.thought_energy);
    assert(telem1.actual_steps == telem2.actual_steps);
    assert(telem1.final_energy == telem2.final_energy);
    assert(telem1.step_deltas.size() == telem2.step_deltas.size());
    for (size_t i = 0; i < telem1.step_deltas.size(); ++i) {
        assert(telem1.step_deltas[i] == telem2.step_deltas[i]);
    }
    std::cout << "  ↳ 双路完全同构松弛比对通过: 终态动能 " << telem1.final_energy 
              << " == " << telem2.final_energy << " (位级完全一致)\n";
}

// 6. 验证阻尼状态松弛抑制极限环振荡 (Damped Relaxation for Oscillation Suppression)
void test_damped_relaxation_oscillation_suppression() {
    std::cout << "[Relaxation Test 6] 验证阻尼松弛因子 (alpha < 1.0) 抑制二阶极限环振荡与不动点沉降...\n";
    CellularOrganism org;
    // 0: 输入 (钳位 0)
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    // 1: 负反馈振荡神经元 (OP_SUM)
    org.cells.push_back(make_cell(1, CellType::OP_SUM, 1.0));
    // 2: 效应器
    org.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    // 构成反相闭环自激环路: 1 -> 1 (w = -1.0)
    org.synapses.push_back(make_synapse(1, 1, 0, -1.0));
    org.synapses.push_back(make_synapse(1, 2, 0, 1.0));
    org.compile();

    // 初始状态打破平衡: 初始电位设为 2.0
    org.cells[1].prev_output_val = 2.0;

    // 对照组 A: 无阻尼标准松弛 (alpha = 1.0)
    // 状态将在 +2.0 与 -2.0 之间无休止翻转，永不收敛
    RelaxationConfig cfg_undamped;
    cfg_undamped.max_steps = 10;
    cfg_undamped.damping_factor = 1.0;
    cfg_undamped.convergence_tol = 1e-4;
    RelaxationTelemetry telem_undamped;

    double inps[4] = {0.0, 0.0, 0.0, 0.0};
    CellularOrganism org_undamped = org;
    org_undamped.forward_with_relaxation(inps, 4, cfg_undamped, &telem_undamped);
    assert(telem_undamped.converged == false);
    assert(telem_undamped.step_deltas.back() > 1.0); // 持续剧烈翻转振荡

    // 实验组 B: 阻尼松弛 (alpha = 0.5)
    // x^(k) = 0.5 * x^(k-1) + 0.5 * (-x^(k-1)) = 0.0! 一步平滑沉降至全局平衡不动点
    RelaxationConfig cfg_damped;
    cfg_damped.max_steps = 25;
    cfg_damped.damping_factor = 0.5;
    cfg_damped.convergence_tol = 1e-4;
    cfg_damped.early_stop = true;
    RelaxationTelemetry telem_damped;

    CellularOrganism org_damped = org;
    auto acts_damped = org_damped.forward_with_relaxation(inps, 4, cfg_damped, &telem_damped);
    assert(telem_damped.converged == true);
    assert(std::abs(acts_damped.positive_action) < 2e-4);
    assert(std::abs(org_damped.cells[1].output_val) < 2e-4);
    assert(std::abs(org_damped.cells[1].prev_output_val) < 2e-4);

    std::cout << "  ↳ 无阻尼残差: " << telem_undamped.step_deltas.back() 
              << " (发散翻转) vs 阻尼松弛终态输出: " << acts_damped.positive_action 
              << " (平稳不动点沉降)\n";
}

// 7. 验证 NaN / 非有限浮点数注入下的李雅普诺夫 BIBO 守卫净化 (BIBO NaN/Inf Sanitization Guard)
void test_bibo_nan_sanitization() {
    std::cout << "[Relaxation Test 7] 验证异常 NaN/Inf 浮点输入下的李雅普诺夫 BIBO 物理吸收与安全清洗...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::ACT_PRIMARY_POSITIVE, 1.0));
    org.synapses.push_back(make_synapse(0, 1, 0, 1.0));
    org.compile();

    // 恶意注入 NaN 输入
    double nan_inps[4] = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0};
    RelaxationTelemetry telem;
    RelaxationConfig cfg;
    cfg.max_steps = 4;
    cfg.bibo_bound = 10.0;

    auto acts = org.forward_with_relaxation(nan_inps, 4, cfg, &telem);

    assert(telem.bibo_stable == false);
    assert(!std::isnan(acts.positive_action));
    assert(!std::isinf(acts.positive_action));
    assert(!std::isnan(telem.final_energy));
    assert(!std::isnan(telem.max_potential_delta));
    for (const auto& c : org.cells) {
        assert(!std::isnan(c.output_val));
        assert(!std::isinf(c.output_val));
        assert(!std::isnan(c.prev_output_val));
    }
    std::cout << "  ↳ NaN 恶意输入被 BIBO 守卫成功吸收并归零: 终态输出 = " << acts.positive_action << "\n";
}

// 8. 验证灵活 API 重载与默认参数分发 (Overload Dispatch Ergonomics)
void test_overload_flexibility() {
    std::cout << "[Relaxation Test 8] 验证 forward_with_relaxation 多态重载与调用人体工学...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(1, CellType::ACT_PRIMARY_POSITIVE, 1.0));
    org.synapses.push_back(make_synapse(0, 1, 0, 1.0));
    org.compile();

    double inps[4] = {1.5, 0.0, 0.0, 0.0};
    RelaxationTelemetry telem;

    // 重载 1: (inputs, relaxation_steps)
    auto a1 = org.forward_with_relaxation(inps, 2);
    // 重载 2: (inputs, relaxation_steps, &telemetry)
    auto a2 = org.forward_with_relaxation(inps, 2, &telem);
    // 重载 3: (inputs, relaxation_steps, in_dim, &telemetry)
    auto a3 = org.forward_with_relaxation(inps, 2, 4, &telem);

    assert(std::abs(a1.positive_action - 1.5) < 1e-9);
    assert(std::abs(a2.positive_action - 1.5) < 1e-9);
    assert(std::abs(a3.positive_action - 1.5) < 1e-9);
    std::cout << "  ↳ 3 组重载函数一致通过，输出均为: " << a1.positive_action << "\n";
}

int main() {
    std::cout << "=======================================================================\n";
    std::cout << " 🧠 硅基细胞计算机 (SDSCC) 测试时动态松弛与吸引子计算单元测试\n";
    std::cout << "=======================================================================\n\n";

    test_relaxation_identity_k1();
    test_pure_dag_instant_convergence();
    test_dynamical_attractor_convergence();
    test_lyapunov_bibo_stability();
    test_deterministic_execution();
    test_damped_relaxation_oscillation_suppression();
    test_bibo_nan_sanitization();
    test_overload_flexibility();

    std::cout << "\n=======================================================================\n";
    std::cout << " 🎉 [全部通过] 测试时动态松弛计算与李雅普诺夫稳态吸引子形式化验证圆满成功!\n";
    std::cout << "=======================================================================\n";
    return 0;
}

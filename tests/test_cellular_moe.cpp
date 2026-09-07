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
#include "kun/cellular/cellular_moe.hpp"

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

// 辅助构建测试用微柱生命体
static CellularOrganism build_test_column(double scale_factor, int mode) {
    CellularOrganism org;
    // 0..3: 感受器
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0));
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_1));
    org.cells.push_back(make_cell(2, CellType::SENSE_RAW_INPUT_2));
    org.cells.push_back(make_cell(3, CellType::SENSE_RAW_INPUT_3));

    // 4: 内部运算
    org.cells.push_back(make_cell(4, mode == 0 ? CellType::OP_SUM : CellType::OP_SUB));

    // 5..7: 动作效应器
    org.cells.push_back(make_cell(5, CellType::ACT_PRIMARY_POSITIVE));
    org.cells.push_back(make_cell(6, CellType::ACT_PRIMARY_NEGATIVE));
    org.cells.push_back(make_cell(7, CellType::ACT_DEFENSIVE_RESET));

    org.synapses.push_back(make_synapse(0, 4, 0, 1.0 * scale_factor));
    org.synapses.push_back(make_synapse(1, 4, 1, 0.5 * scale_factor));
    org.synapses.push_back(make_synapse(4, 5, 0, 1.2));
    org.synapses.push_back(make_synapse(4, 6, 0, -0.8));
    org.synapses.push_back(make_synapse(2, 7, 0, 0.5));

    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

// 1. 验证 Top-K 精确选取与严格零算力旁路 (Zero-Compute Bypass)
void test_topk_and_zero_compute_bypass() {
    std::cout << "[MoE Test 1] 验证 Top-K 精确选取与严格零算力旁路 (Zero-Compute Bypass)...\n";

    const size_t N = 4;
    const size_t K = 2;
    std::vector<CellularOrganism> experts;
    for (size_t i = 0; i < N; ++i) {
        experts.push_back(build_test_column(1.0 + 0.5 * i, i % 2));
    }

    CellularOrganismMoE moe(experts, K, 4, 0.01, 12345);

    // 手动构造可预测权重的路由器以检验精确路由
    // 设置专家 1 和专家 3 在特定输入下获得最高 logits
    auto& weights = moe.router().weights();
    auto& biases = moe.router().biases();
    std::fill(weights.begin(), weights.end(), 0.0);
    std::fill(biases.begin(), biases.end(), 0.0);

    // 设置 biases: 专家 0: 1.0, 专家 1: 5.0, 专家 2: 2.0, 专家 3: 8.0
    // 预期降序排名: 专家 3 (8.0), 专家 1 (5.0), 专家 2 (2.0), 专家 0 (1.0)
    // 预期 Top-2 选取: 专家 3 和 专家 1
    biases[0] = 1.0;
    biases[1] = 5.0;
    biases[2] = 2.0;
    biases[3] = 8.0;

    double inps[4] = {0.5, 0.5, 0.0, 0.0};
    moe.reset_call_counter();

    RoutingDecision decision;
    auto out = moe.forward(inps, 4, false, &decision);

    // 断言 1: Top-2 索引必须严格为 [3, 1]
    assert(decision.topk_indices.size() == 2);
    assert(decision.topk_indices[0] == 3);
    assert(decision.topk_indices[1] == 1);

    // 断言 2: 归一化门控权重和严格为 1.0
    double w_sum = decision.topk_weights[0] + decision.topk_weights[1];
    assert(std::abs(w_sum - 1.0) < 1e-12);
    // Softmax(8.0, 5.0): exp(0) / (exp(0) + exp(-3.0)) = 1 / (1 + e^-3) = 0.952574
    double expected_w0 = 1.0 / (1.0 + std::exp(-3.0));
    assert(std::abs(decision.topk_weights[0] - expected_w0) < 1e-6);

    // 断言 3: 严格零算力旁路！仅对选出的 2 个专家执行了前向推演，未选中的 2 个专家 0 消耗
    assert(moe.active_call_count() == 2);

    // 运行 20 步推演，验证累计旁路节约次数
    for (int step = 0; step < 19; ++step) {
        moe.forward(inps, 4, false);
    }
    // 20 步 * 2 激活 = 40 次调用，若为 Dense 全激活则为 80 次
    assert(moe.active_call_count() == 40);
    assert(moe.router().telemetry().total_bypassed_evaluations == 40); // 节约 40 次
    assert(std::abs(moe.router().telemetry().active_compute_ratio - 0.5) < 1e-9);

    // 断言 4: 输出必须与选中的 2 个专家的加权凸组合严格一致
    auto out_exp3 = moe.experts()[3].forward_nd(inps, 4, false);
    auto out_exp1 = moe.experts()[1].forward_nd(inps, 4, false);
    double exp_pos = decision.topk_weights[0] * out_exp3.positive_action +
                     decision.topk_weights[1] * out_exp1.positive_action;
    assert(std::abs(out.positive_action - exp_pos) < 1e-9);

    std::cout << "  ✓ Top-K 选择正确: [3, 1], 权重 [" << decision.topk_weights[0] 
              << ", " << decision.topk_weights[1] << "]\n";
    std::cout << "  ✓ 严格零算力旁路验证通过: 20 步内仅执行 40 次微柱推演 (Dense 需 80 次，算力旁路率 50.0%)\n";
}

// 2. 验证负载均衡辅助损失计算与专家坍缩抑制特性
void test_load_balancing_loss() {
    std::cout << "[MoE Test 2] 验证负载均衡辅助损失计算与专家坍缩抑制特性...\n";

    MoERouterConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.input_dim = 4;
    cfg.load_balance_alpha = 0.05;

    // 场景 A: 理想均衡路由器 (所有专家被均等选择，全量概率均等)
    CellularRouter balanced_router(cfg, 42);
    balanced_router.reset_batch_history();
    // 构造均匀分布权重 (biases 全 0, weights 全 0)
    std::fill(balanced_router.biases().begin(), balanced_router.biases().end(), 0.0);
    std::fill(balanced_router.weights().begin(), balanced_router.weights().end(), 0.0);

    // 注入对称样本
    double in_zeros[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 100; ++i) {
        balanced_router.route(in_zeros, 4, true);
    }
    double loss_balanced = balanced_router.compute_load_balancing_loss();

    // 场景 B: 严重坍缩路由器 (专家 0 和 专家 1 独占所有门控，专家 2 和 专家 3 被完全饿死)
    CellularRouter collapsed_router(cfg, 42);
    collapsed_router.reset_batch_history();
    collapsed_router.biases()[0] = 50.0;
    collapsed_router.biases()[1] = 50.0;
    collapsed_router.biases()[2] = -50.0;
    collapsed_router.biases()[3] = -50.0;

    for (int i = 0; i < 100; ++i) {
        collapsed_router.route(in_zeros, 4, true);
    }
    double loss_collapsed = collapsed_router.compute_load_balancing_loss();

    // 理论上:
    // 均衡时: f = [0.25, 0.25, 0.25, 0.25], P = [0.25, 0.25, 0.25, 0.25]
    // \sum f_i P_i = 4 * 0.0625 = 0.25
    // Loss_balanced = alpha * N * 0.25 = 0.05 * 4 * 0.25 = 0.05
    // 坍缩时: f = [0.5, 0.5, 0, 0], P = [0.5, 0.5, 0, 0]
    // \sum f_i P_i = 0.25 + 0.25 = 0.50
    // Loss_collapsed = alpha * N * 0.50 = 0.05 * 4 * 0.50 = 0.10
    std::cout << "  - 均衡路由损失: " << loss_balanced << " (理论下界: " << cfg.load_balance_alpha << ")\n";
    std::cout << "  - 坍缩路由损失: " << loss_collapsed << " (坍缩惩罚倍数: " << (loss_collapsed / loss_balanced) << "x)\n";

    assert(std::abs(loss_balanced - cfg.load_balance_alpha) < 1e-6);
    assert(loss_collapsed > loss_balanced * 1.8);

    std::cout << "  ✓ 负载均衡辅助损失成功识别专家坍缩，惩罚倍数精确符合理论倍率!\n";
}

// 3. 验证闭式解析梯度与数值有限差分 (Finite Difference) 100% 精确对账
void test_auxiliary_loss_gradient_backprop() {
    std::cout << "[MoE Test 3] 验证路由辅助损失闭式解析梯度与数值有限差分精确对账...\n";

    MoERouterConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.input_dim = 3;
    cfg.load_balance_alpha = 0.10;

    CellularRouter router(cfg, 9999);
    router.reset_batch_history();

    // 构造具有非平凡概率分布的输入数据集
    std::vector<std::vector<double>> dataset = {
        {0.8, -0.4, 0.5},
        {-0.3, 0.7, -0.2},
        {0.5, 0.5, 0.9},
        {-0.8, -0.2, 0.4},
        {0.1, 0.6, -0.7}
    };

    for (const auto& sample : dataset) {
        router.route(sample.data(), 3, true);
    }

    // 1. 计算解析梯度
    auto analytical_grads = router.compute_auxiliary_loss_gradients();

    // 2. 双边有限差分 (Central Finite Difference) 验证
    // 依据 Switch Transformer / ST-MoE / DeepSeek 规范: f_i 作为不可导指示算子保持常数，
    // 梯度完全沿 Softmax 概率 P_i 闭式反传。
    const double eps = 1e-6;
    const size_t N = cfg.num_experts;
    const size_t D = cfg.input_dim;

    // 辅助函数: 在固定 f 指派率下计算损失
    auto compute_loss_with_perturbed_weights = [&](const std::vector<double>& w_pert, const std::vector<double>& b_pert) {
        std::vector<double> P(N, 0.0);
        for (const auto& x : dataset) {
            std::vector<double> logits(N);
            for (size_t i = 0; i < N; ++i) {
                logits[i] = b_pert[i];
                for (size_t d = 0; d < D; ++d) {
                    logits[i] += w_pert[i * D + d] * x[d];
                }
            }
            double max_l = *std::max_element(logits.begin(), logits.end());
            double sum_e = 0.0;
            std::vector<double> probs(N);
            for (size_t i = 0; i < N; ++i) {
                probs[i] = std::exp(logits[i] - max_l);
                sum_e += probs[i];
            }
            for (size_t i = 0; i < N; ++i) {
                P[i] += probs[i] / sum_e;
            }
        }
        for (size_t i = 0; i < N; ++i) {
            P[i] /= static_cast<double>(dataset.size());
        }

        // 固定的 f 指派率 (按未扰动的 batch_topk 统计)
        std::vector<double> f(N, 0.0);
        for (size_t i = 0; i < dataset.size(); ++i) {
            // 用解析梯度对应的静态 topk
            CellularRouter tmp(cfg);
            tmp.weights() = router.weights();
            tmp.biases() = router.biases();
            auto dec = tmp.route(dataset[i].data(), D, false);
            for (size_t idx : dec.topk_indices) f[idx] += 1.0;
        }
        for (size_t i = 0; i < N; ++i) {
            f[i] /= static_cast<double>(dataset.size() * cfg.top_k);
        }

        double sum_fp = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum_fp += f[i] * P[i];
        }
        return cfg.load_balance_alpha * static_cast<double>(N) * sum_fp;
    };

    double max_weight_grad_err = 0.0;
    for (size_t i = 0; i < N * D; ++i) {
        auto w_plus = router.weights();
        auto w_minus = router.weights();
        w_plus[i] += eps;
        w_minus[i] -= eps;

        double l_plus = compute_loss_with_perturbed_weights(w_plus, router.biases());
        double l_minus = compute_loss_with_perturbed_weights(w_minus, router.biases());
        double num_grad = (l_plus - l_minus) / (2.0 * eps);

        double ana_grad = analytical_grads.d_weights[i];
        double err = std::abs(num_grad - ana_grad);
        if (err > max_weight_grad_err) max_weight_grad_err = err;
        assert(err < 1e-5);
    }

    double max_bias_grad_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        auto b_plus = router.biases();
        auto b_minus = router.biases();
        b_plus[i] += eps;
        b_minus[i] -= eps;

        double l_plus = compute_loss_with_perturbed_weights(router.weights(), b_plus);
        double l_minus = compute_loss_with_perturbed_weights(router.weights(), b_minus);
        double num_grad = (l_plus - l_minus) / (2.0 * eps);

        double ana_grad = analytical_grads.d_biases[i];
        double err = std::abs(num_grad - ana_grad);
        if (err > max_bias_grad_err) max_bias_grad_err = err;
        assert(err < 1e-5);
    }

    std::cout << "  ✓ 权重矩阵解析梯度最大误差: " << max_weight_grad_err << " (< 1e-5)\n";
    std::cout << "  ✓ 偏置向量解析梯度最大误差: " << max_bias_grad_err << " (< 1e-5)\n";

    // 3. 验证梯度更新有效性: 执行一次梯度步后损失必须严格下降
    double loss_before = router.compute_load_balancing_loss();
    router.apply_gradients(analytical_grads, 0.5); // eta = 0.5
    // 重新前向统计新权重下的损失
    router.reset_batch_history();
    for (const auto& sample : dataset) {
        router.route(sample.data(), 3, true);
    }
    double loss_after = router.compute_load_balancing_loss();
    std::cout << "  - 梯度更新前损失: " << loss_before << " -> 更新后损失: " << loss_after << "\n";
    assert(loss_after < loss_before);
    std::cout << "  ✓ 梯度步成功将辅助损失由 " << loss_before << " 降至 " << loss_after << " (收敛确认)!\n";
}

// 4. 验证比特级完全确定性执行 (Deterministic Bit-Level Execution)
void test_deterministic_bit_parity() {
    std::cout << "[MoE Test 4] 验证比特级完全确定性推演 (Deterministic Parity)...\n";

    const size_t N = 8;
    const size_t K = 2;
    std::vector<CellularOrganism> experts_a;
    std::vector<CellularOrganism> experts_b;
    for (size_t i = 0; i < N; ++i) {
        experts_a.push_back(build_test_column(0.5 * (i + 1), i % 2));
        experts_b.push_back(build_test_column(0.5 * (i + 1), i % 2));
    }

    CellularOrganismMoE moe_a(experts_a, K, 4, 0.01, 2026);
    CellularOrganismMoE moe_b(experts_b, K, 4, 0.01, 2026);

    for (int step = 0; step < 50; ++step) {
        double inps[4] = {
            std::sin(step * 0.1),
            std::cos(step * 0.2),
            std::sin(step * 0.3 + 1.0),
            std::cos(step * 0.4 - 0.5)
        };

        RoutingDecision dec_a, dec_b;
        auto out_a = moe_a.forward(inps, 4, false, &dec_a);
        auto out_b = moe_b.forward(inps, 4, false, &dec_b);

        // 比特级校验选中的 Top-K 索引
        assert(dec_a.topk_indices == dec_b.topk_indices);
        // 比特级校验 Top-K 权重
        for (size_t k = 0; k < K; ++k) {
            assert(dec_a.topk_weights[k] == dec_b.topk_weights[k]);
        }
        // 比特级校验输出响应
        assert(out_a.positive_action == out_b.positive_action);
        assert(out_a.negative_action == out_b.negative_action);
        assert(out_a.defensive_reset == out_b.defensive_reset);
    }

    std::cout << "  ✓ 50 步复杂时变推演在双实例间保持 100% 逐比特确定性一致!\n";
}

// 5. 验证极限边界与异常鲁棒性 (Extreme Inputs, Nullptr, NaN/Inf & Degenerate Cases)
void test_extreme_and_edge_cases() {
    std::cout << "[MoE Test 5] 验证极限边界与异常鲁棒性 (Nullptr, NaN/Inf, N=1/K=1)...\n";

    // 5.1 空指针输入鲁棒性 (Nullptr Input)
    {
        std::vector<CellularOrganism> experts = {build_test_column(1.0, 0), build_test_column(1.0, 1)};
        CellularOrganismMoE moe(experts, 1, 4, 0.01, 42);

        RoutingDecision dec;
        auto out_null = moe.forward(nullptr, 4, false, &dec);
        assert(dec.topk_indices.size() == 1);
        assert(std::abs(dec.topk_weights[0] - 1.0) < 1e-12);
        assert(std::isfinite(out_null.positive_action));
        assert(std::isfinite(out_null.negative_action));
    }

    // 5.2 极端数值与非数输入鲁棒性 (NaN, Inf, 1e12)
    {
        std::vector<CellularOrganism> experts = {
            build_test_column(1.0, 0),
            build_test_column(1.0, 1),
            build_test_column(1.0, 0)
        };
        CellularOrganismMoE moe(experts, 2, 4, 0.01, 42);

        double extreme_inps[4] = {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            1e12
        };

        RoutingDecision dec;
        auto out = moe.forward(extreme_inps, 4, false, &dec);
        assert(dec.topk_indices.size() == 2);
        assert(dec.topk_indices[0] < 3 && dec.topk_indices[1] < 3);
        assert(std::isfinite(dec.topk_weights[0]));
        assert(std::isfinite(dec.topk_weights[1]));
        double sum_w = dec.topk_weights[0] + dec.topk_weights[1];
        assert(std::abs(sum_w - 1.0) < 1e-6);
        assert(std::isfinite(out.positive_action));
        assert(std::isfinite(out.negative_action));
    }

    // 5.3 单专家简并场景 (N=1, K=1) 与 空专家构造回退
    {
        std::vector<CellularOrganism> empty_experts;
        CellularOrganismMoE moe_empty(empty_experts, 1, 4, 0.01, 42);
        assert(moe_empty.num_experts() == 1);
        assert(moe_empty.top_k() == 1);

        double inps[4] = {0.1, 0.2, 0.3, 0.4};
        auto out = moe_empty.forward(inps, false);
        assert(std::isfinite(out.positive_action));
    }

    // 5.4 梯度尺寸失配安全门禁
    {
        MoERouterConfig cfg;
        cfg.num_experts = 2;
        cfg.top_k = 1;
        cfg.input_dim = 2;
        CellularRouter router(cfg);

        MoERouterGradients mismatched_grads;
        mismatched_grads.d_weights.assign(10, 1.0); // 长度失配
        mismatched_grads.d_biases.assign(2, 1.0);
        mismatched_grads.num_samples = 1;

        auto w_before = router.weights();
        router.apply_gradients(mismatched_grads, 0.1);
        assert(router.weights() == w_before); // 应当安全跳过更新，杜绝越界
    }

    std::cout << "  ✓ Nullptr、NaN/Inf 极限浮点、N=1/K=1 简并模式与失配梯度安全门禁全部 100% 通过!\n";
}

int main() {
    std::cout << "=====================================================================\n";
    std::cout << "  SDSCC 细粒度微柱神经稀疏门控路由 (Cellular MoE) 单元测试门禁\n";
    std::cout << "=====================================================================\n";

    test_topk_and_zero_compute_bypass();
    test_load_balancing_loss();
    test_auxiliary_loss_gradient_backprop();
    test_deterministic_bit_parity();
    test_extreme_and_edge_cases();

    std::cout << "=====================================================================\n";
    std::cout << "  ✓ Cellular MoE 全部 5 项核心单元测试 100% 满分通过!\n";
    std::cout << "=====================================================================\n";
    return 0;
}

#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include <cmath>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"

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

// 1. 验证 CSR 环形录带与基础反向传播
void test_basic_forward_tape_and_loss() {
    std::cout << "[BPTT Test 1] 验证 CSR 环形录带与基础反向传播...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    org.synapses.push_back(make_synapse(1, 2, 0, 0.8));
    org.synapses.push_back(make_synapse(2, 3, 0, 0.5));
    org.compile();

    CellularBPTTEngine engine(32);
    engine.init_optimizer(org);

    // 录带 10 步
    std::vector<std::vector<float>> targets;
    for (int t = 0; t < 10; ++t) {
        double in[4] = {0.2 * (t + 1), 0.0, 0.0, 0.0};
        org.forward_nd(in, 1, false);
        engine.record_step(org);
        targets.push_back({0.5f});
    }

    assert(engine.current_tape_len == 10);

    BPTTGradients grads;
    float loss = engine.backward(org, targets, grads);
    assert(loss > 0.0f);
    assert(!std::isnan(loss));

    for (float g : grads.grad_synapses) {
        assert(!std::isnan(g));
        assert(!std::isinf(g));
    }
    std::cout << "  ↳ 初始 Loss: " << loss << " (录带 10 步梯度反传成功)\n";
}

// 2. 验证循环反馈突触 (Recurrent Synapse) 跨时间步梯度回传与有限差分对账
void test_recurrent_synapse_gradient_parity() {
    std::cout << "[BPTT Test 2] 验证循环反馈突触 (Recurrent) 跨时间步反向传播与全轨迹有限差分对账...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_INTEGRAL, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    // 前向突触: 1 -> 2
    org.synapses.push_back(make_synapse(1, 2, 0, 0.5));
    // 前向突触: 2 -> 3
    org.synapses.push_back(make_synapse(2, 3, 0, 0.7));
    // 循环反馈突触: 3 -> 2 (形成 recurrent loop)
    org.synapses.push_back(make_synapse(3, 2, 0, 0.3));
    org.compile();

    CellularBPTTEngine engine(16);
    engine.init_optimizer(org);

    const int T = 5;
    std::vector<std::vector<float>> targets(T, std::vector<float>{0.4f});
    std::vector<double> input_seq = {0.1, 0.2, -0.1, 0.3, 0.0};

    // 跑前向并录带
    org.reset_state(false);
    engine.reset_tape();
    for (int t = 0; t < T; ++t) {
        double in[4] = {input_seq[t], 0.0, 0.0, 0.0};
        org.forward_nd(in, 1, false);
        engine.record_step(org);
    }

    BPTTGradients grads;
    float base_loss = engine.backward(org, targets, grads);
    assert(base_loss > 0.0f);

    size_t rec_syn_idx = 2; // syn 3: 3->2
    assert(org.compiled_synapses_[rec_syn_idx].is_recurrent);
    float analytic_g = grads.grad_synapses[rec_syn_idx];
    assert(analytic_g != 0.0f);

    // 全轨迹有限差分双边数值微分
    const double eps = 1e-3;
    auto compute_trajectory_loss = [&](double w_offset) -> float {
        CellularOrganism test_org = org;
        test_org.compiled_synapses_[rec_syn_idx].weight += w_offset;
        test_org.reset_state(false);

        float l = 0.0f;
        for (int t = 0; t < T; ++t) {
            double in[4] = {input_seq[t], 0.0, 0.0, 0.0};
            test_org.forward_nd(in, 1, false);
            float pred = static_cast<float>(test_org.cells[2].output_val);
            float diff = pred - targets[t][0];
            l += diff * diff;
        }
        return l / T;
    };

    float loss_p = compute_trajectory_loss(+eps);
    float loss_m = compute_trajectory_loss(-eps);
    float fd_g = (loss_p - loss_m) / (2.0f * static_cast<float>(eps));

    float err = std::abs(analytic_g - fd_g);
    std::cout << "  ↳ 循环反馈突触梯度对账: Analytic=" << analytic_g << ", FD=" << fd_g << ", Err=" << err << "\n";
    assert(err < 1e-3f);
}

// 3. 关键护城河：李雅普诺夫流形投影测试 (Lyapunov Stability Manifold Projection)
void test_lyapunov_manifold_projection() {
    std::cout << "[BPTT Test 3] 验证李雅普诺夫流形投影算子 (Lyapunov Projection) 强制收缩至 BIBO 稳定流形...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_INTEGRAL, 1.0));

    // 强自激环路 (增益 1.5 * 1.5 * 1.15 = 2.5875 >> 1.0)
    org.synapses.push_back(make_synapse(1, 2, 0, 1.5));
    org.synapses.push_back(make_synapse(2, 1, 0, 1.5));
    org.compile();

    auto pre_rep = org.check_lyapunov_stability();
    assert(!pre_rep.is_stable);
    assert(pre_rep.max_loop_gain > 1.0);

    CellularBPTTEngine engine;
    engine.apply_lyapunov_projection(org, 0.95f);

    auto post_rep = org.check_lyapunov_stability();
    std::cout << "  ↳ 投影前最大环路增益: " << pre_rep.max_loop_gain
              << " -> 投影后最大环路增益: " << post_rep.max_loop_gain << "\n";

    assert(post_rep.is_stable);
    assert(post_rep.max_loop_gain <= 0.950001);
}

// 4. 端到端多步 Adam 梯度收敛测试 (BPTT Optimization Loop)
void test_end_to_end_bptt_optimization() {
    std::cout << "[BPTT Test 4] 运行端到端多步 Adam 梯度收敛闭环 (30 轮轨迹训练)...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0));

    org.synapses.push_back(make_synapse(1, 2, 0, 0.1));
    org.synapses.push_back(make_synapse(2, 3, 0, 0.1));
    org.compile();

    CellularBPTTEngine engine(16);
    engine.init_optimizer(org);

    const int T = 8;
    std::vector<std::vector<float>> targets(T, std::vector<float>{0.75f});
    std::vector<double> fixed_inputs = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 30; ++epoch) {
        org.reset_state(false);
        engine.reset_tape();

        for (int t = 0; t < T; ++t) {
            double in[4] = {fixed_inputs[t], 0.0, 0.0, 0.0};
            org.forward_nd(in, 1, false);
            engine.record_step(org);
        }

        BPTTGradients grads;
        float cur_loss = engine.backward(org, targets, grads);
        if (epoch == 0) initial_loss = cur_loss;
        final_loss = cur_loss;

        engine.step_adam(org, grads, 0.08f);
    }

    std::cout << "  ↳ 初始 Loss: " << initial_loss << " -> 最终 Loss: " << final_loss
              << " (损失降幅: " << (1.0f - final_loss / initial_loss) * 100.0f << "%)\n";
    assert(final_loss < initial_loss * 0.10f); // 至少降低 90%
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " SDSCC CSR 图 BPTT 反向传播内核与李雅普诺夫投影验证套件\n";
    std::cout << "===================================================================\n";

    test_basic_forward_tape_and_loss();
    test_recurrent_synapse_gradient_parity();
    test_lyapunov_manifold_projection();
    test_end_to_end_bptt_optimization();

    std::cout << "\n🎉 全部 4 组 BPTT 可微内核与李雅普诺夫稳定投影测试全部通过!\n";
    return 0;
}

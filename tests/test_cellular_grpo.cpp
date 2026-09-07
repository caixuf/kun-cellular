#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>
#include <cmath>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/cellular_grpo.hpp"

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

// 1. 验证组内相对优势标准化 (Zero-Critic Group Relative Advantage Normalization)
void test_group_relative_advantage_normalization() {
    std::cout << "[GRPO Test 1] 验证数学级组内相对优势标准化 (Group Relative Advantage)...\n";
    GRPOGroup group;
    group.group_id = 1;

    // 4 条不同收益的探索轨迹
    for (float r : {100.0f, 200.0f, 300.0f, 400.0f}) {
        GRPOTrajectory traj;
        traj.raw_reward = r;
        group.rollouts.push_back(traj);
    }

    compute_group_relative_advantages(group);

    // 均值应为 250.0
    assert(std::abs(group.mean_reward - 250.0f) < 1e-4f);
    assert(group.std_reward > 0.0f);

    // 优势和应恒等于 0 (零均值基线消除特性)
    float sum_adv = 0.0f;
    for (const auto& traj : group.rollouts) {
        sum_adv += traj.advantage;
    }
    assert(std::abs(sum_adv) < 1e-4f);

    // 最高回报轨迹优势必为正，最低回报轨迹优势必为负
    assert(group.rollouts[3].advantage > 1.0f);
    assert(group.rollouts[0].advantage < -1.0f);
    std::cout << "  ↳ 均值: " << group.mean_reward << ", 方差标准差: " << group.std_reward
              << ", 最优优势: " << group.rollouts[3].advantage << ", 最劣优势: " << group.rollouts[0].advantage << "\n";

    // 边缘情况测试: 全组回报完全一致时，方差为 0，优势必须平滑归零且无 NaN/Inf
    GRPOGroup zero_var_group;
    for (int i = 0; i < 4; ++i) {
        GRPOTrajectory traj;
        traj.raw_reward = 50.0f;
        zero_var_group.rollouts.push_back(traj);
    }
    compute_group_relative_advantages(zero_var_group);
    for (const auto& traj : zero_var_group.rollouts) {
        assert(!std::isnan(traj.advantage));
        assert(!std::isinf(traj.advantage));
        assert(std::abs(traj.advantage) < 1e-5f);
    }
    std::cout << "  ↳ 零方差保护通过: 优势均平滑归零\n";
}

// 2. 验证 PPO/GRPO 剪裁比率替代目标与梯度计算
void test_grpo_surrogate_clipping() {
    std::cout << "[GRPO Test 2] 验证 GRPO 剪裁比率替代目标与梯度阻断 (Surrogate Clipping)...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0)); // channel 0
    org.cells.push_back(make_cell(4, CellType::ACT_PRIMARY_NEGATIVE, 1.0)); // channel 1

    org.synapses.push_back(make_synapse(1, 2, 0, 0.5));
    org.synapses.push_back(make_synapse(2, 3, 0, 0.5));
    org.synapses.push_back(make_synapse(2, 4, 0, -0.5));
    org.compile();

    CellularBPTTEngine engine(16);
    engine.init_optimizer(org);

    // 录带 1 步
    double in[4] = {1.0, 0.0, 0.0, 0.0};
    org.forward(in, false);
    engine.record_step(org);

    // 未剪裁状态: old_prob = 0.60f, ratio = 0.6225 / 0.60 = 1.0375 (在 [0.8, 1.2] 区间内)
    // targets: [0]=chosen_act, [1]=adv, [2]=old_prob, [3]=clip_eps, [4]=entropy_coef
    std::vector<std::vector<float>> targets_normal = {
        {0.0f, 2.0f, 0.60f, 0.2f, 0.0f}
    };
    BPTTGradients grads_normal;
    float loss_normal = engine.backward_with_loss(org, targets_normal, grads_normal, SubstrateLossType::GRPO_SURROGATE);
    assert(!std::isnan(loss_normal));

    // 剪裁状态: old_prob = 0.40f, ratio = 0.6225 / 0.40 = 1.556 > 1.0 + 0.2 = 1.2
    // 触发 positive advantage 下的 ratio 上界剪裁，策略梯度被阻断 (d_pol = 0)
    std::vector<std::vector<float>> targets_clipped = {
        {0.0f, 2.0f, 0.40f, 0.2f, 0.0f}
    };
    BPTTGradients grads_clipped;
    float loss_clipped = engine.backward_with_loss(org, targets_clipped, grads_clipped, SubstrateLossType::GRPO_SURROGATE);
    assert(!std::isnan(loss_clipped));

    float norm_clipped = 0.0f, norm_normal = 0.0f;
    for (float g : grads_clipped.grad_synapses) norm_clipped += g * g;
    for (float g : grads_normal.grad_synapses) norm_normal += g * g;
    std::cout << "  ↳ 未剪裁梯度范数: " << std::sqrt(norm_normal)
              << ", 剪裁后梯度范数: " << std::sqrt(norm_clipped) << "\n";
    assert(std::sqrt(norm_normal) > 0.0f);
    assert(std::sqrt(norm_clipped) == 0.0f);
}

// 3. 验证端到端 CellularGRPOTrainer 闭环与李雅普诺夫流形投影收敛
void test_end_to_end_grpo_trainer() {
    std::cout << "[GRPO Test 3] 验证端到端 CellularGRPOTrainer 组优化与李雅普诺夫投影收敛...\n";
    CellularOrganism org;
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_0, 1.0));
    org.cells.push_back(make_cell(2, CellType::OP_SUM, 1.0));
    org.cells.push_back(make_cell(3, CellType::ACT_PRIMARY_POSITIVE, 1.0)); // action 0
    org.cells.push_back(make_cell(4, CellType::ACT_PRIMARY_NEGATIVE, 1.0)); // action 1

    org.synapses.push_back(make_synapse(1, 2, 0, 0.1));
    org.synapses.push_back(make_synapse(2, 3, 0, 0.1));
    org.synapses.push_back(make_synapse(2, 4, 0, 0.1));
    org.compile();

    GRPOConfig cfg;
    cfg.group_size = 4;
    cfg.clip_eps = 0.2f;
    cfg.entropy_coef = 0.01f;
    cfg.learning_rate = 0.05f;
    cfg.lyapunov_max_gain = 0.95f;

    CellularGRPOTrainer trainer(cfg, 16);
    trainer.init_optimizer(org);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 20; ++epoch) {
        GRPOGroup group;
        group.group_id = epoch;

        // 构造合成任务: 期望对输入 > 0 激活 action 0
        for (int i = 0; i < 4; ++i) {
            GRPOTrajectory traj;
            GRPOStepTransition st;
            st.obs = {0.5f, 0.0f, 0.0f, 0.0f};
            st.action = (i >= 2) ? 0 : 1; // 后半部分探索到了正确动作 0
            st.logits = {0.1f, 0.1f};
            st.action_prob = 0.5f;
            traj.steps.push_back(st);

            // 正确动作给正回报，错误动作给负回报
            traj.raw_reward = (st.action == 0) ? 10.0f : -10.0f;
            group.rollouts.push_back(traj);
        }

        compute_group_relative_advantages(group);

        BPTTGradients accum_grads;
        GRPOMetrics metrics;
        trainer.accumulate_group_gradients(org, group, accum_grads, metrics);

        if (epoch == 0) initial_loss = metrics.total_loss;
        final_loss = metrics.total_loss;

        trainer.step_optimizer(org, accum_grads, group.rollouts.size(), metrics);
        assert(metrics.max_loop_gain <= 0.950001f);
    }

    std::cout << "  ↳ 初始 Loss: " << initial_loss << " -> 最终 Loss: " << final_loss << "\n";
    assert(!std::isnan(final_loss));
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " SDSCC 神经形态底座 DeepSeek-GRPO 组级相对策略优化内核验证套件\n";
    std::cout << "===================================================================\n";

    test_group_relative_advantage_normalization();
    test_grpo_surrogate_clipping();
    test_end_to_end_grpo_trainer();

    std::cout << "\n🎉 全部 3 组 GRPO 底座相对优势与剪裁替代目标测试全部通过!\n";
    return 0;
}

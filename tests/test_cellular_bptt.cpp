#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <array>
#include <vector>
#include <cmath>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/runtime_migration.hpp"

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

core::InitialParameterSeeds make_core_seeds(const core::GraphDefinition& graph) {
    core::InitialParameterSeeds seeds;
    for (const auto& cell : graph.cells) {
        const auto contract = core::contract_for(cell.type);
        assert(contract.has_value());
        for (size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            core::ParameterValue value = core::UnusedParameter{};
            switch (descriptor.value_type) {
                case core::ParameterValueType::Continuous:
                    value = core::ContinuousValue{1.0};
                    break;
                case core::ParameterValueType::ChannelIndex:
                    value = core::ChannelIndex{0};
                    break;
                case core::ParameterValueType::DelayTicks:
                    value = core::DelayTicks{1};
                    break;
                case core::ParameterValueType::MinMaxMode:
                    value = core::MinMaxMode::Min;
                    break;
                case core::ParameterValueType::Unused:
                    break;
            }
            seeds.cell_parameters.push_back(
                core::CellParameterSeed{
                    cell.id, static_cast<core::ParameterSlot>(slot), value});
        }
    }
    for (const auto& edge : graph.edges) {
        seeds.edge_weights.push_back(core::EdgeParameterSeed{edge.id, 1.0});
    }
    return seeds;
}

core::GraphDefinition make_core_bptt_graph(core::GraphRevision revision = core::GraphRevision{1}) {
    core::GraphDefinition graph;
    graph.identity = core::GraphIdentity{910};
    graph.revision = revision;
    graph.profile = SemanticProfile::LegacyCompatible;
    graph.cells = {
        {core::CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {core::CellId{2}, CellType::OP_SUM},
        {core::CellId{3}, CellType::ACT_PRIMARY_POSITIVE},
    };
    graph.edges = {
        {core::EdgeId{1}, core::CellId{1}, core::OutputPort{0},
         core::CellId{2}, core::InputPort{0}, core::EdgeDelay::Immediate},
        {core::EdgeId{2}, core::CellId{2}, core::OutputPort{0},
         core::CellId{3}, core::InputPort{0}, core::EdgeDelay::Immediate},
    };
    return graph;
}

core::ParameterBinding core_edge_binding(
    const core::RuntimeState& runtime,
    core::EdgeId edge) {
    for (const auto& parameter : runtime.parameters()) {
        if (parameter.binding.kind == core::ParameterBindingKind::EdgeWeight &&
            parameter.binding.edge == edge) {
            return parameter.binding;
        }
    }
    assert(false);
    return {};
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

void test_core_runtime_bptt_learning_window_binding() {
    std::cout << "[BPTT Test 5] 验证 Core RuntimeState 录带、梯度与 LearningWindow/Adam 绑定...\n";
    const auto graph = make_core_bptt_graph();
    const auto seeds = make_core_seeds(graph);
    auto compiled = core::GraphCompiler{}.compile(graph, seeds);
    assert(compiled.ok());
    auto runtime_result =
        core::RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    auto prepared = core::CompiledExecutor::prepare(compiled.graph);
    assert(prepared.ok());
    auto executor = std::move(prepared.executor);

    const auto edge_binding = core_edge_binding(*runtime, core::EdgeId{1});
    auto window = core::LearningWindow::open(
        910, *runtime, {edge_binding});
    CoreCellularBPTTEngine engine(8);
    for (int t = 0; t < 4; ++t) {
        const double input = 0.4 + 0.1 * t;
        assert(engine.record_step(
            *runtime, *executor, std::span<const double>(&input, 1)).ok());
    }

    CoreBPTTGradients gradients;
    const std::vector<std::vector<double>> targets(
        4, std::vector<double>{0.2});
    const auto backward = engine.backward(*runtime, targets, gradients, &window);
    assert(backward.ok());
    assert(backward.loss > 0.0);
    assert(gradients.gradients.size() == 1);
    assert(gradients.gradients[0].binding.edge == core::EdgeId{1});
    assert(std::isfinite(gradients.gradients[0].value));

    const std::array<double, 4> inputs{0.4, 0.5, 0.6, 0.7};
    const auto trajectory_loss = [&](double weight) {
        auto probe_result = runtime->fork_probe();
        assert(probe_result.ok());
        auto probe = std::move(probe_result.runtime);
        assert(probe->reset_episode().ok());
        assert(probe->set_parameter(
            edge_binding, core::ParameterValue{core::ContinuousValue{weight}})
                   .ok());
        double loss = 0.0;
        for (const double input : inputs) {
            assert(executor->step(
                *probe, std::span<const double>(&input, 1)).ok());
            double prediction = 0.0;
            for (const auto& cell : executor->last_measurement().cells) {
                if (cell.cell == core::CellId{3}) prediction = cell.output;
            }
            const double diff = prediction - 0.2;
            loss += diff * diff;
        }
        return loss / static_cast<double>(inputs.size());
    };
    const double weight = std::get<core::ContinuousValue>(
        *runtime->parameter_at(edge_binding.index)).value;
    const double epsilon = 1e-5;
    const double finite_difference =
        (trajectory_loss(weight + epsilon) -
         trajectory_loss(weight - epsilon)) /
        (2.0 * epsilon);
    assert(std::abs(
               finite_difference - gradients.gradients[0].value) < 1e-3);

    const double before = std::get<core::ContinuousValue>(
        *runtime->parameter_at(edge_binding.index)).value;
    const auto update = engine.step_adam(*runtime, window, gradients, 0.05);
    assert(update.ok());
    assert(update.updated_values == 1);
    const double after = std::get<core::ContinuousValue>(
        *runtime->parameter_at(edge_binding.index)).value;
    assert(after != before);

    CoreBPTTGradients forbidden = gradients;
    forbidden.gradients.push_back(
        {runtime->parameters()[0].binding, 1.0});
    const auto rejected = engine.step_adam(
        *runtime, window, forbidden, 0.05);
    assert(!rejected.ok());
    assert(rejected.error->code == CoreBPTTErrorCode::InvalidLearningWindow);

    const auto revised_graph = make_core_bptt_graph(core::GraphRevision{2});
    const auto revised_seeds = make_core_seeds(revised_graph);
    const auto revised = core::GraphCompiler{}.compile(revised_graph, revised_seeds);
    assert(revised.ok());
    const auto migrated = core::RuntimeMigration::rebind(
        *runtime, revised.graph, revised.initial_values);
    assert(migrated.ok());
    const auto stale_update = engine.step_adam(
        *runtime, window, gradients, 0.05);
    assert(!stale_update.ok());
    assert(stale_update.error->code == CoreBPTTErrorCode::InvalidLearningWindow);
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " SDSCC CSR 图 BPTT 反向传播内核与李雅普诺夫投影验证套件\n";
    std::cout << "===================================================================\n";

    test_basic_forward_tape_and_loss();
    test_recurrent_synapse_gradient_parity();
    test_lyapunov_manifold_projection();
    test_end_to_end_bptt_optimization();
    test_core_runtime_bptt_learning_window_binding();

    std::cout << "\n🎉 全部 5 组 BPTT 可微内核与李雅普诺夫稳定投影测试全部通过!\n";
    return 0;
}

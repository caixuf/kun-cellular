#ifndef KUN_CELLULAR_BPTT_HPP_
#define KUN_CELLULAR_BPTT_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_primitives.h"
#include "kun/cellular/sdsc_primitives_vjp.h"
#include "kun/cellular/generated_ops.hpp"

namespace kun {

/**
 * BPTT 录带单步快照 (Forward Tape Step)
 * 连续内存平铺，支持 64~256 步时间窗口的高速反向传播
 */
struct BPTTTapeStep {
    std::vector<float> port_inputs; // [N * 2]
    std::vector<float> cell_outputs; // [N]
    std::vector<float> state_pre;    // [N]
    std::vector<float> aux_pre;      // [N]
    std::vector<float> state_post;   // [N]
    std::vector<float> aux_post;     // [N]
};

/**
 * BPTT 全图参数梯度包
 */
struct BPTTGradients {
    std::vector<float> grad_synapses; // 与 compiled_synapses_ 对齐
    std::vector<float> grad_gains;    // 与 cells 对齐 (param1)

    void zero() {
        std::fill(grad_synapses.begin(), grad_synapses.end(), 0.0f);
        std::fill(grad_gains.begin(), grad_gains.end(), 0.0f);
    }
};

/**
 * 神经形态 CSR 图 BPTT 反向传播引擎 (Cellular BPTT Engine)
 * 
 * 核心特性:
 * 1. CSR 图时序反传: execution_order_ 逆序 + is_recurrent 递归突触跨步时序回传
 * 2. 直通估计器 (STE): 迟滞与门控原语反向直通无损，消除梯度消失
 * 3. 李雅普诺夫流形投影 (Lyapunov Projection): 每次更新后检测最大环增益并强制法向收缩
 * 4. 内置 Adam 优化器与梯度裁剪
 */
class CellularBPTTEngine {
public:
    size_t window_size{64};
    std::vector<BPTTTapeStep> tape;
    size_t current_tape_len{0};

    // Adam 优化器状态
    std::vector<float> m_synapses;
    std::vector<float> v_synapses;
    std::vector<float> m_gains;
    std::vector<float> v_gains;
    uint32_t adam_step{0};
    float beta1{0.9f};
    float beta2{0.999f};
    float eps_adam{1e-8f};
    float grad_clip_norm{1.0f};

    explicit CellularBPTTEngine(size_t max_window = 64)
        : window_size(max_window) {
        tape.resize(window_size);
    }

    void reset_tape() {
        current_tape_len = 0;
    }

    void init_optimizer(const CellularOrganism& org) {
        const size_t num_syn = org.compiled_synapses_.size();
        const size_t num_cells = org.cells.size();
        m_synapses.assign(num_syn, 0.0f);
        v_synapses.assign(num_syn, 0.0f);
        m_gains.assign(num_cells, 0.0f);
        v_gains.assign(num_cells, 0.0f);
        adam_step = 0;
    }

    /**
     * 前向录带：记录当前步推演状态
     */
    void record_step(const CellularOrganism& org) {
        if (current_tape_len >= window_size) {
            // 环形推进：丢弃最旧一步
            for (size_t t = 1; t < window_size; ++t) {
                tape[t - 1] = std::move(tape[t]);
            }
            current_tape_len = window_size - 1;
        }

        const size_t N = org.cells.size();
        auto& step = tape[current_tape_len];
        if (step.cell_outputs.size() != N) {
            step.port_inputs.resize(N * 2);
            step.cell_outputs.resize(N);
            step.state_pre.resize(N);
            step.aux_pre.resize(N);
            step.state_post.resize(N);
            step.aux_post.resize(N);
        }

        for (size_t i = 0; i < N; ++i) {
            const auto& c = org.cells[i];
            step.port_inputs[i * 2 + 0] = static_cast<float>(org.flat_port_inputs_[i * 2 + 0]);
            step.port_inputs[i * 2 + 1] = static_cast<float>(org.flat_port_inputs_[i * 2 + 1]);
            step.cell_outputs[i] = static_cast<float>(c.output_val);
            step.state_post[i] = static_cast<float>(c.state_val);
            step.aux_post[i] = static_cast<float>(c.aux_state);
        }
        current_tape_len++;
    }

    /**
     * 全图反向伴随方程推导 (Full-Graph BPTT Backward Pass)
     * 
     * @param org               待优化生命体
     * @param target_outputs    T 步的目标动作/标签矩阵 [T][action_dim]
     * @param grads             返回的突触权重与增益梯度
     * @return float            全轨迹均方误差损失 (MSE Loss)
     */
    float backward(
        const CellularOrganism& org,
        const std::vector<std::vector<float>>& target_outputs,
        BPTTGradients& grads
    ) {
        const size_t T = std::min(current_tape_len, target_outputs.size());
        if (T == 0) return 0.0f;

        const size_t N = org.cells.size();
        const size_t num_syn = org.compiled_synapses_.size();

        if (grads.grad_synapses.size() != num_syn) grads.grad_synapses.resize(num_syn);
        if (grads.grad_gains.size() != N) grads.grad_gains.resize(N);
        grads.zero();

        // 时序伴随梯度缓冲
        std::vector<float> delta_out(N, 0.0f);
        std::vector<float> delta_in_ports(N * 2, 0.0f);
        std::vector<float> delta_state(N, 0.0f);
        std::vector<float> delta_aux(N, 0.0f);
        std::vector<float> next_delta_state(N, 0.0f);
        std::vector<float> next_delta_aux(N, 0.0f);
        std::vector<float> next_delta_in_ports(N * 2, 0.0f);

        const auto* order_ptr = org.execution_order_.data();
        const size_t num_ordered = org.execution_order_.size();
        const auto* syn_ptr = org.compiled_synapses_.data();
        const auto* out_start_ptr = org.out_start_.data();
        const auto* out_edges_ptr = org.out_edges_.data();

        float total_loss = 0.0f;
        size_t loss_count = 0;

        // 沿时间轴逆序回传 (t = T-1 down to 0)
        for (int t = static_cast<int>(T) - 1; t >= 0; --t) {
            const auto& step = tape[t];
            std::fill(delta_out.begin(), delta_out.end(), 0.0f);
            std::fill(delta_in_ports.begin(), delta_in_ports.end(), 0.0f);

            // 1. 注入效应器动作监督误差 (MSE Loss)
            if (t < static_cast<int>(target_outputs.size())) {
                const auto& targets = target_outputs[t];
                for (const auto& ac : org.compiled_actions_) {
                    if (ac.cell_idx >= N) continue;
                    const auto& c = org.cells[ac.cell_idx];
                    if (!is_effector_cell(c.type)) continue;
                    size_t ch = effector_channel_index(c.type, c.param2);
                    if (ch < targets.size()) {
                        float pred = step.cell_outputs[ac.cell_idx];
                        float target = targets[ch];
                        float diff = pred - target;
                        delta_out[ac.cell_idx] += 2.0f * diff; // dL/dout
                        total_loss += diff * diff;
                        loss_count++;
                    }
                }
            }

            // 2. 注入跨步时序循环突触反馈梯度 (Recurrent Synapse Gradient: t -> t+1)
            // 突触传播: prev_output_val(t) * W 注入到了 t+1 步的 to_port
            if (t + 1 < static_cast<int>(T)) {
                for (size_t s_idx = 0; s_idx < num_syn; ++s_idx) {
                    const auto& syn = syn_ptr[s_idx];
                    if (syn.is_recurrent) {
                        float d_port = next_delta_in_ports[syn.to_idx * 2 + syn.to_port];
                        delta_out[syn.from_idx] += static_cast<float>(syn.weight) * d_port;
                        // 累积递归突触参数梯度: dL/dW = out(t) * delta_in(t+1)
                        grads.grad_synapses[s_idx] += step.cell_outputs[syn.from_idx] * d_port;
                    }
                }
            }

            // 3. 按 Kahn 拓扑排序逆序回传 (从汇点到源点)
            for (int r = static_cast<int>(num_ordered) - 1; r >= 0; --r) {
                size_t i = order_ptr[r];
                const auto& c = org.cells[i];
                uint8_t opcode = cell_type_to_sdsc_opcode(c.type);
                float g = static_cast<float>(c.param1);

                // (a) 收集同时间步非递归出边反传回来的梯度
                for (size_t k = out_start_ptr[i]; k < out_start_ptr[i + 1]; ++k) {
                    size_t s_idx = out_edges_ptr[k];
                    const auto& syn = syn_ptr[s_idx];
                    if (!syn.is_recurrent) {
                        float d_port = delta_in_ports[syn.to_idx * 2 + syn.to_port];
                        delta_out[i] += static_cast<float>(syn.weight) * d_port;
                        // 累积前向突触参数梯度: dL/dW = out(t) * delta_in(t)
                        grads.grad_synapses[s_idx] += step.cell_outputs[i] * d_port;
                    }
                }

                // 效应器恒等透传与受体增益梯度处理
                if (c.type == CellType::ACT_PRIMARY_POSITIVE ||
                    c.type == CellType::ACT_PRIMARY_NEGATIVE ||
                    c.type == CellType::ACT_DEFENSIVE_RESET ||
                    c.type == CellType::ACT_IMMUNE_BLOCK ||
                    c.type == CellType::ACT_CHANNEL ||
                    c.type == CellType::PREDICT_SENSE_0 ||
                    c.type == CellType::PREDICT_SENSE_1) {
                    delta_in_ports[i * 2 + 0] += delta_out[i];
                    continue;
                }
                if (c.type == CellType::SENSE_RAW_INPUT_0 ||
                    c.type == CellType::SENSE_RAW_INPUT_1 ||
                    c.type == CellType::SENSE_RAW_INPUT_2 ||
                    c.type == CellType::SENSE_RAW_INPUT_3 ||
                    c.type == CellType::SENSE_CHANNEL) {
                    float in_val = (c.param1 != 0.0) ? (step.cell_outputs[i] / static_cast<float>(c.param1)) : 0.0f;
                    grads.grad_gains[i] += delta_out[i] * in_val;
                    continue;
                }

                // (b) 调用该算子的 VJP 向量雅可比积
                float x_in = step.port_inputs[i * 2 + 0];
                float in1 = step.port_inputs[i * 2 + 1];
                float x_prim = x_in;
                if (c.type == CellType::OP_SUM) {
                    x_prim = x_in + in1;
                } else if (c.type == CellType::OP_SUB) {
                    x_prim = x_in - in1;
                } else if (c.type == CellType::OP_MULTIPLY) {
                    x_prim = x_in * in1;
                }
                float out_val = step.cell_outputs[i];
                float s_pre = (t > 0) ? tape[t - 1].state_post[i] : 0.0f;
                float a_pre = (t > 0) ? tape[t - 1].aux_post[i] : 0.0f;
                float s_post = step.state_post[i];
                float a_post = step.aux_post[i];
                float dy = delta_out[i];
                float ds_next = next_delta_state[i];
                float da_next = next_delta_aux[i];

                SdscOpVJP vjp = sdsc_primitive_vjp(
                    opcode, g, x_prim, s_pre, a_pre,
                    out_val, s_post, a_post,
                    dy, ds_next, da_next
                );

                // 特殊双端口反传处理 (OP_SUM, OP_SUB, OP_MULTIPLY 等精确链式法则)
                if (c.type == CellType::OP_SUM) {
                    delta_in_ports[i * 2 + 0] += vjp.dx;
                    delta_in_ports[i * 2 + 1] += vjp.dx;
                } else if (c.type == CellType::OP_SUB) {
                    delta_in_ports[i * 2 + 0] += vjp.dx;
                    delta_in_ports[i * 2 + 1] -= vjp.dx;
                } else if (c.type == CellType::OP_MULTIPLY) {
                    delta_in_ports[i * 2 + 0] += vjp.dx * in1;
                    delta_in_ports[i * 2 + 1] += vjp.dx * x_in;
                } else {
                    delta_in_ports[i * 2 + 0] += vjp.dx;
                }

                delta_state[i] = vjp.ds_prev;
                delta_aux[i] = vjp.da_prev;
                grads.grad_gains[i] += vjp.dg;
            }

            // 滑动到前一时刻
            next_delta_state = delta_state;
            next_delta_aux = delta_aux;
            next_delta_in_ports = delta_in_ports;
        }

        if (loss_count > 0) {
            float inv_m = 1.0f / static_cast<float>(loss_count);
            for (float& g : grads.grad_synapses) g *= inv_m;
            for (float& g : grads.grad_gains) g *= inv_m;
        }

        return loss_count > 0 ? (total_loss / loss_count) : 0.0f;
    }

    /**
     * Adam 优化更新 + 梯度裁剪
     */
    void step_adam(CellularOrganism& org, const BPTTGradients& grads, float lr = 0.005f) {
        adam_step++;
        const size_t num_syn = org.compiled_synapses_.size();
        const size_t num_cells = org.cells.size();

        if (m_synapses.size() != num_syn || m_gains.size() != num_cells) {
            init_optimizer(org);
        }

        // 1. 梯度范数裁剪 (Grad Norm Clip)
        double total_norm_sq = 0.0;
        for (float g : grads.grad_synapses) total_norm_sq += g * g;
        for (float g : grads.grad_gains) total_norm_sq += g * g;
        double total_norm = std::sqrt(total_norm_sq);

        float clip_factor = 1.0f;
        if (total_norm > grad_clip_norm && total_norm > 1e-6) {
            clip_factor = static_cast<float>(grad_clip_norm / total_norm);
        }

        const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(adam_step));
        const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(adam_step));

        // 2. 更新突触权重
        for (size_t i = 0; i < num_syn; ++i) {
            float g = grads.grad_synapses[i] * clip_factor;
            m_synapses[i] = beta1 * m_synapses[i] + (1.0f - beta1) * g;
            v_synapses[i] = beta2 * v_synapses[i] + (1.0f - beta2) * g * g;

            float m_hat = m_synapses[i] / bias_correction1;
            float v_hat = v_synapses[i] / bias_correction2;
            float delta = lr * m_hat / (std::sqrt(v_hat) + eps_adam);

            org.compiled_synapses_[i].weight = std::clamp(
                static_cast<double>(org.compiled_synapses_[i].weight - delta), -10.0, 10.0);
            // 同步回未编译基因组
            if (i < org.synapses.size()) {
                org.synapses[i].weight = org.compiled_synapses_[i].weight;
            }
        }

        // 3. 更新细胞增益参数 (param1) 并施加硬件原语边界 (PrimitiveBounds 严格闭包约束)
        for (size_t i = 0; i < num_cells; ++i) {
            float g = grads.grad_gains[i] * clip_factor;
            m_gains[i] = beta1 * m_gains[i] + (1.0f - beta1) * g;
            v_gains[i] = beta2 * v_gains[i] + (1.0f - beta2) * g * g;

            float m_hat = m_gains[i] / bias_correction1;
            float v_hat = v_gains[i] / bias_correction2;
            float delta = lr * m_hat / (std::sqrt(v_hat) + eps_adam);

            uint8_t op = cell_type_to_sdsc_eval_op(org.cells[i].type);
            PrimitiveBounds bounds = get_primitive_bounds(op);
            org.cells[i].param1 = std::clamp(
                org.cells[i].param1 - delta,
                static_cast<double>(bounds.param_g_min),
                static_cast<double>(bounds.param_g_max)
            );
        }

        // 4. 执行李雅普诺夫稳定流形投影 (Lyapunov Manifold Projection)
        apply_lyapunov_projection(org, 0.95f);
    }

    /**
     * 前向推演并连续录带
     */
    void forward_sequence(CellularOrganism& org, const std::vector<std::vector<float>>& inputs) {
        reset_tape();
        for (const auto& inp : inputs) {
            double d_inp[4] = {0.0, 0.0, 0.0, 0.0};
            for (size_t i = 0; i < std::min(size_t(4), inp.size()); ++i) {
                d_inp[i] = static_cast<double>(inp[i]);
            }
            org.forward(d_inp);
            record_step(org);
        }
    }

    const BPTTTapeStep& get_tape_step(size_t t) const {
        return tape[t];
    }

    /**
     * 关键创新：李雅普诺夫投影算子 (Lyapunov Projection Operator)
     * 保证任意梯度优化步后，系统最大环路增益严格收缩至 BIBO 稳定流形 (rho <= 0.95)
     */
    float apply_lyapunov_projection(CellularOrganism& org, float max_allowable_gain = 0.95f) {
        org.enforce_lyapunov_stability(static_cast<double>(max_allowable_gain));
        return static_cast<float>(org.check_lyapunov_stability().max_loop_gain);
    }
};

} // namespace kun

#endif /* KUN_CELLULAR_BPTT_HPP_ */

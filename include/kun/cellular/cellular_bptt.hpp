#ifndef KUN_CELLULAR_BPTT_HPP_
#define KUN_CELLULAR_BPTT_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/heredity.hpp"
#include "kun/cellular/core/kernel_bridge.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/sdsc_primitives.h"
#include "kun/cellular/sdsc_primitives_vjp.h"
#include "kun/cellular/generated_ops.hpp"

namespace kun {

/**
 * 通用损失函数规范 (Universal Substrate Loss Protocol)
 * 提供对标 PyTorch / Transformer 的标准化前向与反向梯度接口
 */
enum class SubstrateLossType {
    MSE = 0,               // 均方误差 (Mean Squared Error)
    CROSS_ENTROPY = 1,     // 动作分布交叉熵 / 负对数似然 (NLL)
    POLICY_GRADIENT = 2,   // 优势加权策略梯度 (Advantage-weighted Policy Gradient)
    SMOOTH_L1 = 3,         // 鲁棒 Huber 损失
    GRPO_SURROGATE = 4     // Group Relative Policy Optimization 剪裁替代目标
};

struct SubstrateLossGrad {
    float loss_val{0.0f};
    std::vector<float> dL_dout; // [channels] 对预测效应器输出的伴随导数
};

using SubstrateLossFn = std::function<SubstrateLossGrad(
    const std::vector<float>& preds,
    const std::vector<float>& targets
)>;

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
    /**
     * @brief 通用损失函数反向传播 (Universal BPTT Backward)
     * 支持经典 MSE、分类交叉熵 (Cross Entropy)、策略梯度强化学习 (Policy Gradient / PPO / GRPO),
     * 以及用户传入自定义可微损失回调函数 (Custom Loss Functor)。
     */
    float backward_with_loss(
        const CellularOrganism& org,
        const std::vector<std::vector<float>>& target_outputs,
        BPTTGradients& grads,
        SubstrateLossType loss_type = SubstrateLossType::MSE,
        const SubstrateLossFn& custom_loss_fn = nullptr
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

            // 1. 注入效应器动作监督误差 (通用损失计算)
            if (t < static_cast<int>(target_outputs.size())) {
                const auto& targets = target_outputs[t];

                // 提取当前步效应器输出通道与其对应的细胞索引
                std::vector<float> preds;
                std::vector<size_t> eff_cell_indices;
                for (const auto& ac : org.compiled_actions_) {
                    if (ac.cell_idx >= N) continue;
                    const auto& c = org.cells[ac.cell_idx];
                    if (!is_effector_cell(c.type)) continue;
                    size_t ch = effector_channel_index(c.type, c.param2);
                    if (ch >= preds.size()) {
                        preds.resize(ch + 1, 0.0f);
                        eff_cell_indices.resize(ch + 1, static_cast<size_t>(-1));
                    }
                    preds[ch] = step.cell_outputs[ac.cell_idx];
                    eff_cell_indices[ch] = ac.cell_idx;
                }

                if (custom_loss_fn) {
                    auto res = custom_loss_fn(preds, targets);
                    total_loss += res.loss_val;
                    for (size_t ch = 0; ch < res.dL_dout.size() && ch < eff_cell_indices.size(); ++ch) {
                        size_t c_idx = eff_cell_indices[ch];
                        if (c_idx < N) delta_out[c_idx] += res.dL_dout[ch];
                    }
                } else if (loss_type == SubstrateLossType::CROSS_ENTROPY) {
                    float max_p = preds.empty() ? 0.0f : *std::max_element(preds.begin(), preds.end());
                    float sum_exp = 0.0f;
                    std::vector<float> probs(preds.size());
                    for (size_t k = 0; k < preds.size(); ++k) {
                        probs[k] = std::exp(preds[k] - max_p);
                        sum_exp += probs[k];
                    }
                    if (sum_exp > 1e-7f) {
                        for (float& p : probs) p /= sum_exp;
                    }
                    for (size_t ch = 0; ch < preds.size() && ch < targets.size(); ++ch) {
                        float target = targets[ch];
                        float p = std::max(probs[ch], 1e-7f);
                        total_loss += (-target * std::log(p));
                        size_t c_idx = eff_cell_indices[ch];
                        if (c_idx < N) delta_out[c_idx] += (probs[ch] - target);
                    }
                } else if (loss_type == SubstrateLossType::POLICY_GRADIENT) {
                    // targets: [0]=chosen_action_idx, [1]=advantage
                    int chosen_act = targets.empty() ? 0 : static_cast<int>(targets[0]);
                    float adv = targets.size() > 1 ? targets[1] : 1.0f;

                    float max_p = preds.empty() ? 0.0f : *std::max_element(preds.begin(), preds.end());
                    float sum_exp = 0.0f;
                    std::vector<float> probs(preds.size());
                    for (size_t k = 0; k < preds.size(); ++k) {
                        probs[k] = std::exp(preds[k] - max_p);
                        sum_exp += probs[k];
                    }
                    if (sum_exp > 1e-7f) {
                        for (float& p : probs) p /= sum_exp;
                    }
                    float p_chosen = (chosen_act >= 0 && chosen_act < static_cast<int>(probs.size())) ? probs[chosen_act] : 1e-7f;
                    total_loss += (-adv * std::log(std::max(p_chosen, 1e-7f)));
                    for (size_t ch = 0; ch < preds.size(); ++ch) {
                        float grad = -adv * ((static_cast<int>(ch) == chosen_act ? 1.0f : 0.0f) - probs[ch]);
                        size_t c_idx = eff_cell_indices[ch];
                        if (c_idx < N) delta_out[c_idx] += grad;
                    }
                } else if (loss_type == SubstrateLossType::GRPO_SURROGATE) {
                    // DeepSeek-GRPO 剪裁比率替代目标与信息熵正则 (Group Relative Policy Optimization)
                    // targets: [0]=chosen_action_idx, [1]=advantage, [2]=old_prob, [3]=clip_eps, [4]=entropy_coef
                    int chosen_act = targets.empty() ? 0 : static_cast<int>(targets[0]);
                    float adv = targets.size() > 1 ? targets[1] : 0.0f;
                    float old_prob = targets.size() > 2 ? targets[2] : 0.0f;
                    float clip_eps = targets.size() > 3 ? targets[3] : 0.2f;
                    float entropy_coef = targets.size() > 4 ? targets[4] : 0.01f;

                    float max_p = preds.empty() ? 0.0f : *std::max_element(preds.begin(), preds.end());
                    float sum_exp = 0.0f;
                    std::vector<float> probs(preds.size());
                    for (size_t k = 0; k < preds.size(); ++k) {
                        probs[k] = std::exp(preds[k] - max_p);
                        sum_exp += probs[k];
                    }
                    if (sum_exp > 1e-7f) {
                        for (float& p : probs) p /= sum_exp;
                    }

                    float p_chosen = (chosen_act >= 0 && chosen_act < static_cast<int>(probs.size())) ? probs[chosen_act] : 1e-7f;
                    if (old_prob < 1e-7f) old_prob = p_chosen;
                    float ratio = p_chosen / std::max(old_prob, 1e-7f);

                    bool clipped = false;
                    if (adv > 0.0f && ratio > 1.0f + clip_eps) clipped = true;
                    else if (adv < 0.0f && ratio < 1.0f - clip_eps) clipped = true;

                    float surrogate = clipped ? (std::clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps) * adv) : (ratio * adv);
                    float pol_loss = -surrogate;

                    float entropy = 0.0f;
                    for (float p : probs) {
                        if (p > 1e-7f) entropy -= p * std::log(p);
                    }

                    total_loss += (pol_loss - entropy_coef * entropy);

                    for (size_t ch = 0; ch < preds.size(); ++ch) {
                        float d_pol = 0.0f;
                        if (!clipped) {
                            float delta_ak = (static_cast<int>(ch) == chosen_act ? 1.0f : 0.0f);
                            d_pol = -adv * ratio * (delta_ak - probs[ch]);
                        }
                        float d_ent = 0.0f;
                        if (entropy_coef > 0.0f) {
                            float p_k = std::max(probs[ch], 1e-7f);
                            d_ent = entropy_coef * probs[ch] * (std::log(p_k) + entropy);
                        }
                        float grad = d_pol + d_ent;
                        size_t c_idx = eff_cell_indices[ch];
                        if (c_idx < N) delta_out[c_idx] += grad;
                    }
                } else {
                    // 经典 MSE 损失
                    for (size_t ch = 0; ch < preds.size() && ch < targets.size(); ++ch) {
                        float diff = preds[ch] - targets[ch];
                        total_loss += diff * diff;
                        size_t c_idx = eff_cell_indices[ch];
                        if (c_idx < N) delta_out[c_idx] += 2.0f * diff;
                    }
                }
                loss_count++;
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
     * @brief 向后兼容的默认 MSE 反向传播接口
     */
    float backward(
        const CellularOrganism& org,
        const std::vector<std::vector<float>>& target_outputs,
        BPTTGradients& grads
    ) {
        return backward_with_loss(org, target_outputs, grads, SubstrateLossType::MSE);
    }

    /**
     * Adam 优化更新 + 梯度裁剪
     */
    void step_adam(CellularOrganism& org, const BPTTGradients& grads, float lr = 0.005f) {
        const size_t num_syn = org.compiled_synapses_.size();
        const size_t num_cells = org.cells.size();

        // [修复] init_optimizer 会重置 adam_step=0; 若在其后不重新递增,
        // 首步 bias_correction = 1-β⁰ = 0 → m̂ = 0/0 = NaN → 写穿全部细胞增益 (param1)
        if (m_synapses.size() != num_syn || m_gains.size() != num_cells) {
            init_optimizer(org);
        }
        adam_step++;

        // 尺寸守卫: 梯度向量与当前基因组不匹配 (如空窗/消融模式) 时直接跳过更新
        if (grads.grad_synapses.size() < num_syn || grads.grad_gains.size() < num_cells) {
            return;
        }

        // 0. NaN/Inf 梯度消毒 (nan_to_num): 任何非有限梯度分量归零,
        //    防止单点爆炸通过 Adam 写穿整个基因组 (Hebbian 路径有 isfinite 守卫, Adam 路径同样必须有)
        for (float& g : const_cast<std::vector<float>&>(grads.grad_synapses)) {
            if (!std::isfinite(g)) g = 0.0f;
        }
        for (float& g : const_cast<std::vector<float>&>(grads.grad_gains)) {
            if (!std::isfinite(g)) g = 0.0f;
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

        // 2. 更新突触权重 (带事务快照: 更新后出现非有限权重则整体回滚)
        std::vector<float> weight_snapshot(num_syn);
        for (size_t i = 0; i < num_syn; ++i) weight_snapshot[i] = org.compiled_synapses_[i].weight;
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
        // 事务回滚: 若 Adam 更新把任何权重写成 NaN/Inf, 恢复更新前快照
        bool any_nonfinite = false;
        for (size_t i = 0; i < num_syn; ++i) {
            if (!std::isfinite(org.compiled_synapses_[i].weight)) { any_nonfinite = true; break; }
        }
        if (any_nonfinite) {
            for (size_t i = 0; i < num_syn; ++i) {
                org.compiled_synapses_[i].weight = weight_snapshot[i];
                if (i < org.synapses.size()) org.synapses[i].weight = weight_snapshot[i];
            }
            // 快照亦不干净时退化为初值 (基因始祖权重)
            for (size_t i = 0; i < num_syn; ++i) {
                if (!std::isfinite(org.compiled_synapses_[i].weight)) {
                    double init_w = org.synapses[i].initial_weight;
                    org.compiled_synapses_[i].weight = init_w;
                    if (i < org.synapses.size()) org.synapses[i].weight = init_w;
                }
            }
        }
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

/**
 * Core RuntimeState BPTT adapter.
 *
 * This deliberately lives beside the legacy engine instead of changing its
 * CellularOrganism ABI.  The tape is populated from RuntimeState snapshots and
 * CompiledExecutor measurements, so gradients are taken through the same
 * kernels and typed live parameters used by production execution.
 *
 * Current core-native scope is the differentiable MSE path.  Other legacy loss
 * modes and the legacy Lyapunov projection remain available through
 * CellularBPTTEngine until their core contracts have first-class equivalents.
 */
struct CoreBPTTTapeStep {
    std::vector<double> inputs;
    std::vector<double> port_inputs;
    std::vector<core::InitialParameterValue> parameters;
    std::vector<core::RuntimeCellState> state_pre;
    std::vector<core::RuntimeCellState> state_post;
    core::GraphIdentity identity{};
    core::GraphRevision revision{};
    uint32_t semantic_version{0};
};

struct CoreBPTTGradients {
    std::vector<core::LearningGradient> gradients;
    double loss{0.0};

    void clear() {
        gradients.clear();
        loss = 0.0;
    }
};

enum class CoreBPTTErrorCode : uint8_t {
    InvalidRuntime,
    InvalidExecutor,
    StaleTape,
    InvalidTargets,
    UnsupportedLoss,
    InvalidLearningWindow,
};

struct CoreBPTTError {
    CoreBPTTErrorCode code{CoreBPTTErrorCode::InvalidRuntime};
    std::string reason;
};

struct CoreBPTTResult {
    double loss{0.0};
    std::size_t updated_values{0};
    std::optional<CoreBPTTError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class CoreCellularBPTTEngine final {
public:
    size_t window_size{64};
    std::vector<CoreBPTTTapeStep> tape;
    size_t current_tape_len{0};

    std::vector<double> m_parameters;
    std::vector<double> v_parameters;
    uint64_t adam_step{0};
    double beta1{0.9};
    double beta2{0.999};
    double eps_adam{1e-8};
    double grad_clip_norm{1.0};

    explicit CoreCellularBPTTEngine(size_t max_window = 64)
        : window_size(max_window), tape(window_size) {}

    void reset_tape() {
        current_tape_len = 0;
    }

    void init_optimizer(const core::RuntimeState& runtime) {
        m_parameters.assign(runtime.parameters().size(), 0.0);
        v_parameters.assign(runtime.parameters().size(), 0.0);
        adam_step = 0;
    }

    CoreBPTTResult record_step(
        core::RuntimeState& runtime,
        core::CompiledExecutor& executor,
        std::span<const double> inputs) {
        if (!executor.plan() || !runtime.bound_to(*executor.plan())) {
            return failure(
                CoreBPTTErrorCode::InvalidExecutor,
                "runtime and compiled executor are not bound to the same plan");
        }
        if (!inputs.empty() && inputs.data() == nullptr) {
            return failure(
                CoreBPTTErrorCode::InvalidRuntime,
                "non-empty input span has a null data pointer");
        }

        const auto before = runtime.snapshot();
        const auto executed = executor.step(runtime, inputs);
        if (!executed.ok()) {
            return failure(
                CoreBPTTErrorCode::InvalidRuntime,
                std::string(executed.error->reason));
        }
        const auto after = runtime.snapshot();
        const auto measurement = executed.measurement;

        if (current_tape_len >= window_size) {
            for (size_t t = 1; t < window_size; ++t) {
                tape[t - 1] = std::move(tape[t]);
            }
            current_tape_len = window_size - 1;
        }

        auto& step = tape[current_tape_len];
        const size_t cells = runtime.plan()->cells().size();
        step.inputs.assign(inputs.begin(), inputs.end());
        step.port_inputs.assign(cells * 2, 0.0);
        step.parameters.assign(
            before.parameters().begin(), before.parameters().end());
        step.state_pre.assign(before.cells().begin(), before.cells().end());
        step.state_post.assign(after.cells().begin(), after.cells().end());
        step.identity = runtime.identity();
        step.revision = runtime.revision();
        step.semantic_version = runtime.plan()->semantic_version();
        for (const auto& port : measurement.ports) {
            const auto index = cell_index(*runtime.plan(), port.cell);
            if (index.has_value() && port.port.value < 2) {
                step.port_inputs[*index * 2 + port.port.value] =
                    port.reduced_input;
            }
        }
        ++current_tape_len;
        return {};
    }

    CoreBPTTResult forward_sequence(
        core::RuntimeState& runtime,
        core::CompiledExecutor& executor,
        const std::vector<std::vector<double>>& inputs) {
        reset_tape();
        for (const auto& input : inputs) {
            const auto result = record_step(
                runtime,
                executor,
                std::span<const double>(input.data(), input.size()));
            if (!result.ok()) return result;
        }
        return {};
    }

    CoreBPTTResult backward(
        const core::RuntimeState& runtime,
        const std::vector<std::vector<double>>& target_outputs,
        CoreBPTTGradients& gradients,
        const core::LearningWindow* window = nullptr) const {
        if (!runtime.plan()) {
            return failure(
                CoreBPTTErrorCode::InvalidRuntime,
                "BPTT requires a runtime with an owned compiled plan");
        }
        if (current_tape_len == 0) {
            gradients.clear();
            return {};
        }
        if (target_outputs.size() < current_tape_len) {
            return failure(
                CoreBPTTErrorCode::InvalidTargets,
                "core BPTT requires one target row per recorded step");
        }
        if (window) {
            if (const auto valid = window->validate(runtime); !valid.ok()) {
                return failure(
                    CoreBPTTErrorCode::InvalidLearningWindow,
                    valid.error->reason);
            }
        }
        const auto& first = tape[0];
        if (first.state_pre.size() != runtime.plan()->cells().size()) {
            return failure(
                CoreBPTTErrorCode::StaleTape,
                "core BPTT tape dimensions do not match the runtime plan");
        }
        for (size_t t = 0; t < current_tape_len; ++t) {
            if (tape[t].state_pre.size() != first.state_pre.size() ||
                tape[t].state_post.size() != first.state_pre.size()) {
                return failure(
                    CoreBPTTErrorCode::StaleTape,
                    "core BPTT tape contains inconsistent graph dimensions");
            }
        }

        const auto plan = runtime.plan();
        if (first.state_post.size() != plan->cells().size()) {
            return failure(
                CoreBPTTErrorCode::StaleTape,
                "core BPTT tape is not bound to the current graph");
        }
        for (size_t t = 0; t < current_tape_len; ++t) {
            if (tape[t].identity != runtime.identity() ||
                tape[t].revision != runtime.revision() ||
                tape[t].semantic_version != plan->semantic_version()) {
                return failure(
                    CoreBPTTErrorCode::StaleTape,
                    "core BPTT tape graph identity or revision is stale");
            }
        }
        gradients.clear();
        const size_t cell_count = plan->cells().size();
        const size_t parameter_count = runtime.parameters().size();
        std::vector<double> grad_by_parameter(parameter_count, 0.0);
        std::vector<float> delta_out(cell_count, 0.0f);
        std::vector<float> delta_ports(cell_count * 2, 0.0f);
        std::vector<float> delta_state(cell_count, 0.0f);
        std::vector<float> delta_aux(cell_count, 0.0f);
        std::vector<float> next_delta_state(cell_count, 0.0f);
        std::vector<float> next_delta_aux(cell_count, 0.0f);
        std::vector<float> next_delta_ports(cell_count * 2, 0.0f);

        auto allowed = [window](const core::ParameterBinding& binding) {
            if (!window) return true;
            return std::find_if(
                       window->allowed_parameters().begin(),
                       window->allowed_parameters().end(),
                       [&](const auto& candidate) {
                           return same_binding(candidate, binding);
                       }) != window->allowed_parameters().end();
        };
        auto add_gradient = [&](const core::ParameterBinding& binding, double value) {
            if (binding.index >= grad_by_parameter.size() || !allowed(binding)) {
                return;
            }
            if (std::isfinite(value)) grad_by_parameter[binding.index] += value;
        };
        auto continuous_value = [](const CoreBPTTTapeStep& step, size_t index)
            -> std::optional<double> {
            if (index >= step.parameters.size()) return std::nullopt;
            const auto& value = step.parameters[index].value;
            if (const auto* continuous = std::get_if<core::ContinuousValue>(&value)) {
                return continuous->value;
            }
            return core::legacy_kernel_parameter(value);
        };

        double total_loss = 0.0;
        size_t loss_count = 0;
        for (int t = static_cast<int>(current_tape_len) - 1; t >= 0; --t) {
            const auto& step = tape[static_cast<size_t>(t)];
            if (step.parameters.size() != parameter_count) {
                return failure(
                    CoreBPTTErrorCode::StaleTape,
                    "core BPTT tape parameter bindings do not match the runtime");
            }
            std::fill(delta_out.begin(), delta_out.end(), 0.0f);
            std::fill(delta_ports.begin(), delta_ports.end(), 0.0f);

            std::vector<float> predictions;
            std::vector<size_t> effector_indices;
            for (size_t i = 0; i < cell_count; ++i) {
                const auto& cell = plan->cells()[i];
                if (!is_effector_cell(cell.type)) continue;
                const auto channel = effector_channel(step, cell);
                if (!channel.has_value()) continue;
                if (*channel >= predictions.size()) {
                    predictions.resize(*channel + 1, 0.0f);
                    effector_indices.resize(
                        *channel + 1,
                        static_cast<size_t>(-1));
                }
                predictions[*channel] =
                    static_cast<float>(step.state_post[i].output_val);
                effector_indices[*channel] = i;
            }
            const auto& targets = target_outputs[static_cast<size_t>(t)];
            for (size_t channel = 0;
                 channel < predictions.size() && channel < targets.size();
                 ++channel) {
                const float diff =
                    predictions[channel] - static_cast<float>(targets[channel]);
                total_loss += static_cast<double>(diff) * diff;
                const size_t index = effector_indices[channel];
                if (index < cell_count) delta_out[index] += 2.0f * diff;
            }
            if (!predictions.empty()) ++loss_count;

            if (t + 1 < static_cast<int>(current_tape_len)) {
                for (const auto& edge : plan->edges()) {
                    if (edge.delay != core::EdgeDelay::PreviousTick) continue;
                    const float d_port =
                        next_delta_ports[edge.target_index * 2 +
                                         edge.target_port.value];
                    const auto weight =
                        continuous_value(step, edge.weight_parameter_index);
                    if (!weight) continue;
                    delta_out[edge.source_index] +=
                        static_cast<float>(*weight) * d_port;
                    add_gradient(
                        runtime.parameters()[edge.weight_parameter_index].binding,
                        step.state_post[edge.source_index].output_val * d_port);
                }
            }

            for (int reverse = static_cast<int>(plan->execution_order().size()) - 1;
                 reverse >= 0; --reverse) {
                const size_t i = plan->execution_order()[static_cast<size_t>(reverse)];
                const auto& cell = plan->cells()[i];
                for (const auto& edge : plan->edges()) {
                    if (edge.source_index != i ||
                        edge.delay != core::EdgeDelay::Immediate) {
                        continue;
                    }
                    const float d_port =
                        delta_ports[edge.target_index * 2 +
                                    edge.target_port.value];
                    const auto weight =
                        continuous_value(step, edge.weight_parameter_index);
                    if (!weight) continue;
                    delta_out[i] += static_cast<float>(*weight) * d_port;
                    add_gradient(
                        runtime.parameters()[edge.weight_parameter_index].binding,
                        step.state_post[i].output_val * d_port);
                }

                if (is_effector_cell(cell.type)) {
                    delta_ports[i * 2] += delta_out[i];
                    continue;
                }
                if (is_receptor_cell(cell.type)) {
                    const auto channel = parameter_channel(step, cell);
                    const size_t input_channel =
                        channel.has_value() ? *channel : static_cast<size_t>(
                            cell.type == CellType::SENSE_RAW_INPUT_1
                                ? 1
                                : cell.type == CellType::SENSE_RAW_INPUT_2
                                    ? 2
                                    : cell.type == CellType::SENSE_RAW_INPUT_3 ? 3 : 0);
                    const double input =
                        input_channel < step.inputs.size() ? step.inputs[input_channel] : 0.0;
                    add_gradient(
                        runtime.parameters()[cell.parameter_indices[0]].binding,
                        delta_out[i] * input);
                    continue;
                }

                const auto gain =
                    continuous_value(step, cell.parameter_indices[0]);
                if (!gain) continue;
                const float x0 = static_cast<float>(step.port_inputs[i * 2]);
                const float x1 = static_cast<float>(step.port_inputs[i * 2 + 1]);
                float x_prim = x0;
                if (cell.type == CellType::OP_SUM) x_prim = x0 + x1;
                else if (cell.type == CellType::OP_SUB) x_prim = x0 - x1;
                else if (cell.type == CellType::OP_MULTIPLY) x_prim = x0 * x1;

                const auto& pre = step.state_pre[i];
                const auto& post = step.state_post[i];
                const auto vjp = sdsc_primitive_vjp(
                    cell_type_to_sdsc_opcode(cell.type),
                    static_cast<float>(*gain),
                    x_prim,
                    static_cast<float>(pre.state_val),
                    static_cast<float>(pre.aux_state),
                    static_cast<float>(post.output_val),
                    static_cast<float>(post.state_val),
                    static_cast<float>(post.aux_state),
                    delta_out[i],
                    next_delta_state[i],
                    next_delta_aux[i]);

                if (cell.type == CellType::OP_SUM) {
                    delta_ports[i * 2] += vjp.dx;
                    delta_ports[i * 2 + 1] += vjp.dx;
                } else if (cell.type == CellType::OP_SUB) {
                    delta_ports[i * 2] += vjp.dx;
                    delta_ports[i * 2 + 1] -= vjp.dx;
                } else if (cell.type == CellType::OP_MULTIPLY) {
                    delta_ports[i * 2] += vjp.dx * x1;
                    delta_ports[i * 2 + 1] += vjp.dx * x0;
                } else {
                    delta_ports[i * 2] += vjp.dx;
                }
                delta_state[i] = vjp.ds_prev;
                delta_aux[i] = vjp.da_prev;
                add_gradient(
                    runtime.parameters()[cell.parameter_indices[0]].binding,
                    vjp.dg);
            }

            next_delta_state = delta_state;
            next_delta_aux = delta_aux;
            next_delta_ports = delta_ports;
        }

        if (loss_count > 0) {
            const double inverse = 1.0 / static_cast<double>(loss_count);
            gradients.loss = total_loss * inverse;
            for (size_t index = 0; index < grad_by_parameter.size(); ++index) {
                grad_by_parameter[index] *= inverse;
            }
        }
        for (const auto& parameter : runtime.parameters()) {
            if (parameter.binding.kind == core::ParameterBindingKind::EdgeWeight) {
            if (allowed(parameter.binding)) {
                gradients.gradients.push_back(
                    {parameter.binding, grad_by_parameter[parameter.binding.index]});
            }
            continue;
        }
            const auto cell = find_cell(*plan, parameter.binding.cell);
            if (!cell.has_value()) continue;
            const auto contract = core::contract_for(plan->cells()[*cell].type);
            if (contract.has_value() &&
                core::is_continuous_trainable(
                    contract->get().parameters[
                        static_cast<size_t>(parameter.binding.slot)])) {
                if (allowed(parameter.binding)) {
                    gradients.gradients.push_back(
                        {parameter.binding, grad_by_parameter[parameter.binding.index]});
                }
            }
        }
        return {gradients.loss, 0, std::nullopt};
    }

    CoreBPTTResult step_adam(
        core::RuntimeState& runtime,
        core::LearningWindow& window,
        const CoreBPTTGradients& gradients,
        double learning_rate = 0.005) {
        if (const auto valid = window.validate(runtime); !valid.ok()) {
            return failure(
                CoreBPTTErrorCode::InvalidLearningWindow,
                valid.error->reason);
        }
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0) {
            return failure(
                CoreBPTTErrorCode::InvalidLearningWindow,
                "learning rate must be finite and positive");
        }
        std::vector<core::ParameterBinding> bindings;
        bindings.reserve(gradients.gradients.size());
        for (const auto& gradient : gradients.gradients) {
            bindings.push_back(gradient.binding);
            if (!std::isfinite(gradient.value) ||
                gradient.binding.index >= runtime.parameters().size()) {
                return failure(
                    CoreBPTTErrorCode::InvalidLearningWindow,
                    "core BPTT gradient is non-finite or unbound");
            }
        }
        if (const auto valid = window.validate_parameters(runtime, bindings);
            !valid.ok()) {
            return failure(
                CoreBPTTErrorCode::InvalidLearningWindow,
                valid.error->reason);
        }
        if (m_parameters.size() != runtime.parameters().size() ||
            v_parameters.size() != runtime.parameters().size()) {
            init_optimizer(runtime);
        }
        double norm_squared = 0.0;
        for (const auto& gradient : gradients.gradients) {
            norm_squared += gradient.value * gradient.value;
        }
        const double norm = std::sqrt(norm_squared);
        const double clip =
            norm > grad_clip_norm && norm > std::numeric_limits<double>::epsilon()
                ? grad_clip_norm / norm
                : 1.0;
        ++adam_step;
        const double correction1 = 1.0 - std::pow(beta1, static_cast<double>(adam_step));
        const double correction2 = 1.0 - std::pow(beta2, static_cast<double>(adam_step));
        std::vector<core::LearningGradient> normalized;
        normalized.reserve(gradients.gradients.size());
        for (const auto& gradient : gradients.gradients) {
            const size_t index = gradient.binding.index;
            const double clipped = gradient.value * clip;
            m_parameters[index] = beta1 * m_parameters[index] +
                                  (1.0 - beta1) * clipped;
            v_parameters[index] = beta2 * v_parameters[index] +
                                 (1.0 - beta2) * clipped * clipped;
            const double m_hat = m_parameters[index] / correction1;
            const double v_hat = v_parameters[index] / correction2;
            normalized.push_back({
                gradient.binding,
                m_hat / (std::sqrt(v_hat) + eps_adam)});
        }
        const auto updated = window.apply_sgd(
            runtime,
            std::span<const core::LearningGradient>(
                normalized.data(), normalized.size()),
            learning_rate);
        if (!updated.ok()) {
            return failure(
                CoreBPTTErrorCode::InvalidLearningWindow,
                updated.error->reason);
        }
        return {gradients.loss, updated.report.updated_values, std::nullopt};
    }

private:
    static CoreBPTTResult failure(
        CoreBPTTErrorCode code,
        std::string reason) {
        return {0.0, 0, CoreBPTTError{code, std::move(reason)}};
    }

    static bool same_binding(
        const core::ParameterBinding& lhs,
        const core::ParameterBinding& rhs) {
        return lhs.kind == rhs.kind &&
               lhs.index == rhs.index &&
               lhs.cell == rhs.cell &&
               lhs.edge == rhs.edge &&
               lhs.slot == rhs.slot;
    }

    static std::optional<size_t> cell_index(
        const core::CompiledGraph& plan,
        core::CellId id) {
        const auto cells = plan.cells();
        const auto it = std::lower_bound(
            cells.begin(), cells.end(), id,
            [](const core::CompiledCell& cell, core::CellId sought) {
                return cell.id < sought;
            });
        if (it == cells.end() || it->id != id) return std::nullopt;
        return static_cast<size_t>(it - cells.begin());
    }

    static std::optional<size_t> find_cell(
        const core::CompiledGraph& plan,
        core::CellId id) {
        return cell_index(plan, id);
    }

    static std::optional<size_t> effector_channel(
        const CoreBPTTTapeStep& step,
        const core::CompiledCell& cell) {
        switch (cell.type) {
            case CellType::ACT_PRIMARY_POSITIVE: return 0;
            case CellType::ACT_PRIMARY_NEGATIVE: return 1;
            case CellType::ACT_DEFENSIVE_RESET: return 2;
            case CellType::ACT_IMMUNE_BLOCK: return 3;
            case CellType::ACT_CHANNEL: {
                if (cell.parameter_indices[1] >= step.parameters.size()) {
                    return std::nullopt;
                }
                const auto& value = step.parameters[cell.parameter_indices[1]].value;
                const auto* channel =
                    std::get_if<core::ChannelIndex>(&value);
                return channel ? std::optional<size_t>{channel->value}
                               : std::nullopt;
            }
            default:
                return std::nullopt;
        }
    }

    static std::optional<size_t> parameter_channel(
        const CoreBPTTTapeStep& step,
        const core::CompiledCell& cell) {
        if (cell.parameter_indices[1] >= step.parameters.size()) {
            return std::nullopt;
        }
        const auto& value = step.parameters[cell.parameter_indices[1]].value;
        const auto* channel = std::get_if<core::ChannelIndex>(&value);
        return channel ? std::optional<size_t>{channel->value} : std::nullopt;
    }
};

} // namespace kun

#endif /* KUN_CELLULAR_BPTT_HPP_ */

#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 投机推演决策仲裁机制
// (Speculative Decision-Making & Dual-Process Verification Engine)
//
// 理论渊源:
//   1. 现代大模型投机解码 (Speculative Decoding / Leviathan et al., 2023):
//      小规模草稿模型 (Draft Model) 纳秒级极速生成先验假设，主干验证模型 (Verifier Model)
//      一次性校验流形一致性并决定接受 (Accept) 或回退修正 (Reject & Fallback)。
//   2. 认知双系统理论 (Dual-Process Theory / Kahneman System 1 & System 2):
//      系统 1 (瞬时条件反射核, 16 细胞) 提供直觉冲动；系统 2 (深思皮层, 64 细胞) 负责审慎阻尼。
//
// 绝对铁律: 严格恪守 Rule 7 Substrate Immunity 宪章，纯粹张量代数与决策仲裁，零具身业务词汇。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace kun {

/**
 * @brief 投机推演决策仲裁配置
 */
struct SpeculativeConfig {
    float acceptance_threshold{0.65f};  // 草稿被接纳的置信度下限
    float confidence_margin{0.15f};     // 草稿胜出动作与次优动作的概率间隔边界
    bool enable_soft_arbitration{false};// 是否采用软插值仲裁而非硬裁决
    float draft_weight{0.20f};          // 软仲裁时草稿所占权重
};

/**
 * @brief 投机决策运行时遥测指标
 */
struct SpeculativeTelemetry {
    uint64_t total_decisions{0};       // 总决策次数
    uint64_t accepted_drafts{0};      // 草稿直接被接纳次数 (Hit)
    uint64_t rejected_drafts{0};      // 草稿被否决回退次数 (Miss)
    double acceptance_rate{0.0};       // 投机接纳率 (Alpha = Hit / Total)
    double draft_latency_ns{0.0};      // 草稿阶段平均时延
    double verify_latency_ns{0.0};     // 验证阶段平均时延
    double effective_latency_ns{0.0};  // 加权有效决策时延
    double speedup_ratio{1.0};         // 相比全量验证模型的加速比
};

/**
 * @brief 投机决策单步仲裁输出
 */
struct SpeculativeDecision {
    int chosen_action{0};              // 最终采纳的动作序号
    bool draft_accepted{false};        // 本次是否直接采纳草稿决策
    float confidence{0.0f};            // 最终决策置信度
    std::vector<float> arbitrated_probs;// 仲裁后的归一化概率分布
};

/**
 * @brief 纯数学函数: Softmax 概率分布归一化
 */
inline std::vector<float> compute_softmax_distribution(const float* logits, size_t dim) {
    if (!logits || dim == 0) return {};
    float max_l = logits[0];
    for (size_t i = 1; i < dim; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    std::vector<float> probs(dim);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] = std::exp(logits[i] - max_l);
        sum_exp += probs[i];
    }
    float inv_sum = (sum_exp > 1e-7f) ? (1.0f / sum_exp) : 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= inv_sum;
    }
    return probs;
}

/**
 * @brief 投机决策核心仲裁引擎 (Cellular Speculative Arbitrator)
 */
class CellularSpeculativeEngine {
public:
    SpeculativeConfig config;
    SpeculativeTelemetry telemetry;

    explicit CellularSpeculativeEngine(const SpeculativeConfig& cfg = SpeculativeConfig{})
        : config(cfg) {}

    void reset_telemetry() {
        telemetry = SpeculativeTelemetry{};
    }

    /**
     * @brief 决策仲裁: 对比草稿 Logits 与验证 Logits
     *
     * @param draft_logits   16 细胞先验草稿核输出 [action_dim]
     * @param verify_logits  64 细胞深思皮层验证输出 [action_dim]
     * @param action_dim     动作离散维度
     * @return SpeculativeDecision 仲裁裁决结果
     */
    SpeculativeDecision arbitrate(
        const float* draft_logits,
        const float* verify_logits,
        size_t action_dim)
    {
        if (!draft_logits || !verify_logits || action_dim == 0) {
            return SpeculativeDecision{};
        }

        auto p_draft = compute_softmax_distribution(draft_logits, action_dim);
        auto p_verify = compute_softmax_distribution(verify_logits, action_dim);

        // 寻找草稿最优动作与次优动作
        int best_draft_act = 0;
        float best_draft_p = p_draft[0];
        for (size_t i = 1; i < action_dim; ++i) {
            if (p_draft[i] > best_draft_p) {
                best_draft_p = p_draft[i];
                best_draft_act = static_cast<int>(i);
            }
        }

        int best_verify_act = 0;
        float best_verify_p = p_verify[0];
        for (size_t i = 1; i < action_dim; ++i) {
            if (p_verify[i] > best_verify_p) {
                best_verify_p = p_verify[i];
                best_verify_act = static_cast<int>(i);
            }
        }

        telemetry.total_decisions++;

        SpeculativeDecision decision;
        // 接纳准则:
        // 1. 草稿动作与验证动作一致 (Consensus)
        // 2. 草稿置信度满足门禁: best_draft_p >= acceptance_threshold
        bool consensus = (best_draft_act == best_verify_act);
        bool confident = (best_draft_p >= config.acceptance_threshold);

        if (consensus && confident) {
            // 接纳草稿决策 (Fast-path Accept)
            decision.chosen_action = best_draft_act;
            decision.draft_accepted = true;
            decision.confidence = best_draft_p;
            decision.arbitrated_probs = p_draft;
            telemetry.accepted_drafts++;
        } else {
            // 否决回退: 由审慎验证主皮层强制纠偏 (Fallback Correction)
            decision.chosen_action = best_verify_act;
            decision.draft_accepted = false;
            decision.confidence = best_verify_p;
            decision.arbitrated_probs = p_verify;
            telemetry.rejected_drafts++;
        }

        if (telemetry.total_decisions > 0) {
            telemetry.acceptance_rate = static_cast<double>(telemetry.accepted_drafts) / 
                                        static_cast<double>(telemetry.total_decisions);
        }

        return decision;
    }

    /**
     * @brief 计算并更新有效时延与综合加速比
     */
    void update_latency_metrics(double t_draft_ns, double t_verify_ns) {
        telemetry.draft_latency_ns = t_draft_ns;
        telemetry.verify_latency_ns = t_verify_ns;

        // 接纳时仅产生草稿时延，否决时产生草稿 + 验证时延
        telemetry.effective_latency_ns = telemetry.acceptance_rate * t_draft_ns + 
                                         (1.0 - telemetry.acceptance_rate) * (t_draft_ns + t_verify_ns);

        if (telemetry.effective_latency_ns > 0.0) {
            telemetry.speedup_ratio = t_verify_ns / telemetry.effective_latency_ns;
        } else {
            telemetry.speedup_ratio = 1.0;
        }
    }
};

} // namespace kun

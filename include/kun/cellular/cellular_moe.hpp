#pragma once

#include "kun/cellular/cellular_router.hpp"
#include "kun/cellular/cellular_genome.hpp"

namespace kun {
/**
 * @brief 基于微柱神经基底的稀疏混合专家网络 (CellularOrganism MoE)
 * 
 * 核心特性:
 * - 聚合 N 个独立微柱生命体 (CellularOrganism)
 * - 门控路由器精确调度 Top-K 活跃专家
 * - 严格零算力旁路 (Zero-Compute Bypass): 未激活的 N-K 个微柱不消耗任何前向与循环计算
 */
class CellularOrganismMoE {
public:
    CellularOrganismMoE(
        const std::vector<CellularOrganism>& expert_organisms,
        size_t top_k = 2,
        size_t input_dim = 4,
        double load_balance_alpha = 0.01,
        uint32_t router_seed = 42)
        : experts_(expert_organisms),
          active_call_count_(0) {
        if (experts_.empty()) {
            CellularOrganism default_expert;
            default_expert.compile();
            experts_.push_back(default_expert);
        }
        if (top_k == 0) top_k = 1;
        if (top_k > experts_.size()) top_k = experts_.size();

        MoERouterConfig cfg;
        cfg.num_experts = experts_.size();
        cfg.top_k = top_k;
        cfg.input_dim = input_dim;
        cfg.load_balance_alpha = load_balance_alpha;
        router_ = CellularRouter(cfg, router_seed);

        for (auto& exp : experts_) {
            if (!exp.is_compiled()) {
                exp.compile();
            }
        }
    }

    size_t num_experts() const { return experts_.size(); }
    size_t top_k() const { return router_.config().top_k; }
    CellularRouter& router() { return router_; }
    const CellularRouter& router() const { return router_; }
    std::vector<CellularOrganism>& experts() { return experts_; }
    const std::vector<CellularOrganism>& experts() const { return experts_; }
    uint64_t active_call_count() const { return active_call_count_; }

    void reset_call_counter() { active_call_count_ = 0; }

    void reset_state(bool full_reset = false) {
        for (auto& exp : experts_) {
            exp.reset_state(full_reset);
        }
        router_.reset_batch_history();
    }

    /**
     * @brief 稀疏 MoE 4 维固定输入前向推演 (与 CellularOrganism::forward 接口完全对称)
     */
    CellularOrganism::ActionOutputs forward(
        const double inputs[4],
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        return forward_nd(inputs, 4, enable_hebbian, out_decision, record_for_aux_loss);
    }

    /**
     * @brief 稀疏 MoE 动态多维指针输入前向推演
     */
    CellularOrganism::ActionOutputs forward(
        const double* inputs,
        size_t in_dim,
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        return forward_nd(inputs, in_dim, enable_hebbian, out_decision, record_for_aux_loss);
    }

    /**
     * @brief 稀疏 MoE 核心前向推演 (具有严格的 Zero-Compute Bypass 与零堆分配)
     * 仅对被选中的 Top-K 专家执行实际的前向计算！
     */
    CellularOrganism::ActionOutputs forward_nd(
        const double* inputs,
        size_t in_dim = 4,
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        
        const auto& decision = router_.route(inputs, in_dim, record_for_aux_loss);
        if (out_decision) *out_decision = decision;

        CellularOrganism::ActionOutputs combined{};
        combined.thought_mode = "MOE_SPARSE";

        // === 严格零算力旁路 (Zero-Compute Bypass) ===
        for (size_t k = 0; k < decision.topk_indices.size(); ++k) {
            size_t idx = decision.topk_indices[k];
            if (idx >= experts_.size()) continue;
            double weight = decision.topk_weights[k];

            active_call_count_++;
            auto out = experts_[idx].forward_nd(inputs, in_dim, enable_hebbian);

            combined.positive_action += weight * out.positive_action;
            combined.negative_action += weight * out.negative_action;
            combined.defensive_reset += weight * out.defensive_reset;
            combined.predicted_sense_0 += weight * out.predicted_sense_0;
            combined.predicted_sense_1 += weight * out.predicted_sense_1;
            combined.prediction_error  += weight * out.prediction_error;
            combined.thought_energy    += weight * out.thought_energy;
            if (out.immune_lock) combined.immune_lock = true;
        }

        return combined;
    }

    /**
     * @brief 密集基准前向推演 (全量 N 个专家均激活计算，用于算力与时延消融对账，真·零 GC 零堆分配)
     */
    CellularOrganism::ActionOutputs forward_dense(
        const double* inputs,
        size_t in_dim = 4,
        bool enable_hebbian = false) {

        const size_t N = experts_.size();
        CellularOrganism::ActionOutputs combined{};
        combined.thought_mode = "DENSE_ALL_ACTIVE";
        if (N == 0) return combined;

        // 统一均权累加融合，杜绝堆分配
        double weight = 1.0 / static_cast<double>(N);
        for (size_t i = 0; i < N; ++i) {
            active_call_count_++;
            auto out = experts_[i].forward_nd(inputs, in_dim, enable_hebbian);
            combined.positive_action += weight * out.positive_action;
            combined.negative_action += weight * out.negative_action;
            combined.defensive_reset += weight * out.defensive_reset;
            combined.predicted_sense_0 += weight * out.predicted_sense_0;
            combined.predicted_sense_1 += weight * out.predicted_sense_1;
            combined.prediction_error  += weight * out.prediction_error;
            combined.thought_energy    += weight * out.thought_energy;
            if (out.immune_lock) combined.immune_lock = true;
        }

        return combined;
    }

private:
    std::vector<CellularOrganism> experts_;
    CellularRouter router_;
    uint64_t active_call_count_{0};
};

} // namespace kun

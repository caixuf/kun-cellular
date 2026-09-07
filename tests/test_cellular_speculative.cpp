#include "kun/cellular/cellular_speculative.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace kun;

void test_speculative_acceptance() {
    SpeculativeConfig cfg;
    cfg.acceptance_threshold = 0.60f;
    CellularSpeculativeEngine engine(cfg);

    // 动作 1 明显占优: p1 = e^3 / (e^3 + 2*e^0) = 20.08 / 22.08 ≈ 0.91 >= 0.60
    float draft_logits[3] = {0.0f, 3.0f, 0.0f};
    float verify_logits[3] = {0.2f, 2.5f, 0.1f};

    auto dec = engine.arbitrate(draft_logits, verify_logits, 3);
    assert(dec.chosen_action == 1);
    assert(dec.draft_accepted == true);
    assert(engine.telemetry.total_decisions == 1);
    assert(engine.telemetry.accepted_drafts == 1);
    assert(engine.telemetry.acceptance_rate == 1.0);
}

void test_speculative_rejection_divergence() {
    SpeculativeConfig cfg;
    cfg.acceptance_threshold = 0.50f;
    CellularSpeculativeEngine engine(cfg);

    // 草稿认为动作 0 最好，但主体验证模型认为动作 2 最好
    float draft_logits[3] = {3.0f, 0.0f, 0.0f};
    float verify_logits[3] = {0.0f, 0.5f, 3.5f};

    auto dec = engine.arbitrate(draft_logits, verify_logits, 3);
    assert(dec.chosen_action == 2); // 强制采纳验证模型动作 2
    assert(dec.draft_accepted == false);
    assert(engine.telemetry.total_decisions == 1);
    assert(engine.telemetry.rejected_drafts == 1);
    assert(engine.telemetry.acceptance_rate == 0.0);
}

void test_speculative_rejection_low_confidence() {
    SpeculativeConfig cfg;
    cfg.acceptance_threshold = 0.70f;
    CellularSpeculativeEngine engine(cfg);

    // 两者虽一致选动作 1，但草稿置信度不够高 (均分接近: p1 ≈ 0.45 < 0.70)
    float draft_logits[3] = {1.0f, 1.2f, 0.9f};
    float verify_logits[3] = {0.5f, 2.0f, 0.4f};

    auto dec = engine.arbitrate(draft_logits, verify_logits, 3);
    assert(dec.chosen_action == 1);
    assert(dec.draft_accepted == false); // 因置信度不足而回退
    assert(engine.telemetry.rejected_drafts == 1);
}

void test_speculative_speedup_calculation() {
    SpeculativeConfig cfg;
    CellularSpeculativeEngine engine(cfg);

    // 假设 80% 接纳率
    engine.telemetry.total_decisions = 100;
    engine.telemetry.accepted_drafts = 80;
    engine.telemetry.rejected_drafts = 20;
    engine.telemetry.acceptance_rate = 0.80;

    // 草稿 50ns, 验证 800ns
    // 有效时延 = 0.80 * 50 + 0.20 * (50 + 800) = 40 + 170 = 210ns
    // 加速比 = 800 / 210 ≈ 3.81x
    engine.update_latency_metrics(50.0, 800.0);
    assert(std::abs(engine.telemetry.effective_latency_ns - 210.0) < 1e-3);
    assert(std::abs(engine.telemetry.speedup_ratio - (800.0 / 210.0)) < 1e-3);
}

int main() {
    test_speculative_acceptance();
    test_speculative_rejection_divergence();
    test_speculative_rejection_low_confidence();
    test_speculative_speedup_calculation();
    std::cout << "[PASS] test_cellular_speculative all assertions passed successfully!\n";
    return 0;
}

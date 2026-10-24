#pragma once
// ============================================================================
// population/e_gate.hpp — Prefer-A E-Gate pipeline + E1/E2/E3 word discipline
//
// Ordered stages (all required for E2):
//   1. Pre-register — claim id, BD, metrics, seeds, scales, stop rule; learn≠exec
//   2. Anti-artifact — dual · ≥4×3-scale · OOD+paired green
//   3. Explainable — mechanism sketch tied to landed operators
//   4. Reproducible — cmd + seeds + tip hash
//
// E3 = failed stage 2 or 3. Bare "emerged/life/AGI" without E-tag is forbidden.
// Ban: execution scale marketed as learning scale.
// ============================================================================
#include "kun/cellular/population/anti_artifact.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace kun {
namespace population {

enum class EmergenceTag : std::uint8_t {
    E1 = 1,  // weak / descriptive
    E2 = 2,  // strong-pending — passed E-Gate
    E3 = 3   // artifact / failed anti-artifact or explainable
};

inline const char* emergence_tag_name(EmergenceTag t) {
    switch (t) {
        case EmergenceTag::E1: return "E1";
        case EmergenceTag::E2: return "E2";
        case EmergenceTag::E3: return "E3";
        default: return "E?";
    }
}

enum class EGateStage : std::uint8_t {
    PRE_REGISTER = 1,
    ANTI_ARTIFACT = 2,
    EXPLAINABLE = 3,
    REPRODUCIBLE = 4
};

struct EGatePreRegister {
    std::string claim_id;
    std::string behavioral_descriptor;
    std::vector<std::string> metrics;
    std::vector<std::uint32_t> seeds;
    std::vector<LearningScalePoint> learn_scales;
    std::uint32_t exec_scale{82};          // evaluation / demo size
    std::uint32_t learn_scale{82};         // scale agent learned at
    bool exec_equals_learn_claimed{false}; // if true AND scales differ → reject
    std::string stop_rule;
    std::string tip_hash;                  // reproducible tip
    std::string cmd;
    std::string mechanism_sketch;          // explainable stage input
};

struct EGateVerdict {
    EmergenceTag tag{EmergenceTag::E1};
    EGateStage failed_stage{EGateStage::PRE_REGISTER};
    bool stage1_ok{false};
    bool stage2_ok{false};
    bool stage3_ok{false};
    bool stage4_ok{false};
    bool pass_e2{false};
    std::string reason;
};

inline bool check_exec_learn_ban(const EGatePreRegister& pr, std::string& reason) {
    // Ban: claiming exec scale as learn scale when they differ.
    if (pr.exec_scale != pr.learn_scale && pr.exec_equals_learn_claimed) {
        reason = "ban: exec_scale marketed as learn_scale";
        return false;
    }
    // Pre-register must record both explicitly.
    if (pr.exec_scale == 0 || pr.learn_scale == 0) {
        reason = "pre-register must record exec_scale and learn_scale";
        return false;
    }
    return true;
}

inline EGateVerdict run_e_gate(const EGatePreRegister& pr,
                               const AntiArtifactEvidence& evidence) {
    EGateVerdict v;
    std::string ban_reason;

    // Stage 1: pre-register completeness + exec≠learn ban
    const bool has_id = !pr.claim_id.empty();
    const bool has_bd = !pr.behavioral_descriptor.empty();
    const bool has_metrics = pr.metrics.size() >= 2;
    const bool has_seeds = pr.seeds.size() >= kAntiArtifactMinSeeds;
    const bool has_scales = pr.learn_scales.size() >= kAntiArtifactMinScales;
    const bool has_stop = !pr.stop_rule.empty();
    const bool ban_ok = check_exec_learn_ban(pr, ban_reason);
    v.stage1_ok = has_id && has_bd && has_metrics && has_seeds && has_scales && has_stop && ban_ok;
    if (!v.stage1_ok) {
        v.tag = EmergenceTag::E1;
        v.failed_stage = EGateStage::PRE_REGISTER;
        v.reason = ban_ok ? "pre-register incomplete" : ban_reason;
        return v;
    }

    // Stage 2: anti-artifact
    const auto aa = evaluate_anti_artifact(evidence);
    v.stage2_ok = aa.pass;
    if (!v.stage2_ok) {
        v.tag = EmergenceTag::E3;  // failed stage 2 → E3
        v.failed_stage = EGateStage::ANTI_ARTIFACT;
        v.reason = aa.reason;
        return v;
    }

    // Stage 3: explainable
    v.stage3_ok = !pr.mechanism_sketch.empty();
    if (!v.stage3_ok) {
        v.tag = EmergenceTag::E3;  // failed stage 3 → E3
        v.failed_stage = EGateStage::EXPLAINABLE;
        v.reason = "mechanism sketch missing";
        return v;
    }

    // Stage 4: reproducible
    v.stage4_ok = !pr.tip_hash.empty() && !pr.cmd.empty() && !pr.seeds.empty();
    if (!v.stage4_ok) {
        v.tag = EmergenceTag::E1;
        v.failed_stage = EGateStage::REPRODUCIBLE;
        v.reason = "reproducible archive incomplete (cmd/seeds/tip)";
        return v;
    }

    v.pass_e2 = true;
    v.tag = EmergenceTag::E2;
    v.reason = "E-Gate green → E2";
    return v;
}

}  // namespace population
}  // namespace kun

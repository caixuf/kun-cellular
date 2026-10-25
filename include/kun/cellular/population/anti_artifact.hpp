#pragma once
// ============================================================================
// population/anti_artifact.hpp — Prefer-A anti-artifact harness
//
// Requirements:
//   - Dual metrics: ≥2 independent scores for any E2 claim
//   - ≥4 seeds × 3 learning scales (NOT FE LIFEFORM_SCALES viz)
//   - OOD + paired / McNemar (reuse statistical_evaluation.hpp)
// ============================================================================
#include "kun/cellular/statistical_evaluation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kun {
namespace population {

inline constexpr std::size_t kAntiArtifactMinSeeds = 4;
inline constexpr std::size_t kAntiArtifactMinScales = 3;

// Learning-scale axis (cells / columns / episode length) — NOT FE viz scales.
enum class LearningScaleAxis : std::uint8_t {
    CELLS = 0,
    COLUMNS = 1,
    EPISODE_LENGTH = 2
};

struct LearningScalePoint {
    LearningScaleAxis axis{LearningScaleAxis::CELLS};
    std::uint32_t value{82};   // Prefer-A default learn scale = 82 cells
    const char* label{"learn_cells_82"};
};

struct DualMetricPair {
    const char* name_a{"coverage"};
    const char* name_b{"novelty"};
    double value_a{0.0};
    double value_b{0.0};
    bool independent{true};  // caller asserts independence
};

struct AntiArtifactEvidence {
    DualMetricPair dual{};
    std::vector<std::uint32_t> seeds;           // ≥4
    std::vector<LearningScalePoint> scales;     // ≥3 learning scales
    bool ood_condition_present{false};
    std::string ood_label;
    cellular::McNemarResult mcnemar{};
    cellular::PairedDifferenceResult paired{};
    bool paired_used{false};
};

struct AntiArtifactVerdict {
    bool dual_ok{false};
    bool seeds_ok{false};
    bool scales_ok{false};
    bool ood_paired_ok{false};
    bool pass{false};
    std::string reason;
};

inline AntiArtifactVerdict evaluate_anti_artifact(const AntiArtifactEvidence& ev) {
    AntiArtifactVerdict v;
    v.dual_ok = ev.dual.independent &&
                (ev.dual.name_a != nullptr) && (ev.dual.name_b != nullptr) &&
                (std::string(ev.dual.name_a) != std::string(ev.dual.name_b));
    v.seeds_ok = ev.seeds.size() >= kAntiArtifactMinSeeds;
    v.scales_ok = ev.scales.size() >= kAntiArtifactMinScales;
    // Distinct scale values required
    if (v.scales_ok) {
        std::vector<std::uint32_t> vals;
        for (const auto& s : ev.scales) vals.push_back(s.value);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        if (vals.size() < kAntiArtifactMinScales) {
            v.scales_ok = false;
        }
    }
    const bool mcnemar_ok = ev.ood_condition_present && (ev.mcnemar.n > 0);
    const bool paired_ok = ev.ood_condition_present && ev.paired_used && (ev.paired.n > 0);
    v.ood_paired_ok = mcnemar_ok || paired_ok;

    v.pass = v.dual_ok && v.seeds_ok && v.scales_ok && v.ood_paired_ok;
    if (!v.dual_ok) v.reason = "dual metrics missing or not independent";
    else if (!v.seeds_ok) v.reason = "need >=4 seeds";
    else if (!v.scales_ok) v.reason = "need >=3 distinct learning scales";
    else if (!v.ood_paired_ok) v.reason = "need OOD + McNemar or paired differences";
    else v.reason = "anti-artifact green";
    return v;
}

// Prefer-A openness scale-sweep: max/min learning-scale value span >=4x (A2).
// Cheap optional helper — does NOT change evaluate_anti_artifact defaults, so
// existing E1-E3 evidence packs stay green without a span field.
inline constexpr double kAntiArtifactMinSpanRatio = 4.0;

inline double learning_scale_span_ratio(const std::vector<LearningScalePoint>& scales) {
    if (scales.empty()) return 0.0;
    std::uint32_t lo = scales.front().value;
    std::uint32_t hi = scales.front().value;
    for (const auto& s : scales) {
        if (s.value < lo) lo = s.value;
        if (s.value > hi) hi = s.value;
    }
    if (lo == 0u) return 0.0;
    return static_cast<double>(hi) / static_cast<double>(lo);
}

inline bool learning_scale_span_ok(const std::vector<LearningScalePoint>& scales,
                                   double min_ratio = kAntiArtifactMinSpanRatio) {
    std::vector<std::uint32_t> vals;
    vals.reserve(scales.size());
    for (const auto& s : scales) vals.push_back(s.value);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    if (vals.size() < kAntiArtifactMinScales) return false;
    return learning_scale_span_ratio(scales) + 1e-12 >= min_ratio;
}

// Helper: Prefer-A default 3 learning scales on cells axis (82 held as primary).
inline std::array<LearningScalePoint, 3> default_prefer_a_learning_scales() {
    return {{
        {LearningScaleAxis::CELLS, 41, "learn_cells_41"},
        {LearningScaleAxis::CELLS, 82, "learn_cells_82"},
        {LearningScaleAxis::CELLS, 164, "learn_cells_164"},
    }};
}

}  // namespace population
}  // namespace kun

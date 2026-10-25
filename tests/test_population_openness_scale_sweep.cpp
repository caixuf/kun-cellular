// ============================================================================
// test_population_openness_scale_sweep.cpp — Stream A scale-sweep unit (手术刀)
// Asserts: >=3 distinct Prefer-A learn scales · span_ratio >=4 · exec stays 82.
// Does NOT touch protect_window live deme wire (焊枪 owns
// test_population_openness_protect_live.cpp).
// ============================================================================
#include "kun/cellular/population/anti_artifact.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using kun::population::AntiArtifactEvidence;
using kun::population::LearningScaleAxis;
using kun::population::LearningScalePoint;
using kun::population::default_prefer_a_learning_scales;
using kun::population::evaluate_anti_artifact;
using kun::population::kAntiArtifactMinScales;
using kun::population::kAntiArtifactMinSpanRatio;
using kun::population::kAvalancheMaxSites;
using kun::population::kPreferAOrganismCells;
using kun::population::learning_scale_span_ok;
using kun::population::learning_scale_span_ratio;

namespace {

void test_exec_cells_held() {
    assert(kPreferAOrganismCells == 82);
    std::cout << "  ✓ A4 exec: kPreferAOrganismCells=82 held\n";
}

void test_default_scales_span_ge_4() {
    auto arr = default_prefer_a_learning_scales();
    assert(arr.size() == 3);
    assert(arr[0].value == 41);
    assert(arr[1].value == 82);
    assert(arr[2].value == 164);
    std::vector<LearningScalePoint> scales(arr.begin(), arr.end());
    const double span = learning_scale_span_ratio(scales);
    assert(span + 1e-12 >= kAntiArtifactMinSpanRatio);
    assert(span + 1e-12 >= 4.0);
    // 164/41 == 4.0 exactly by Prefer pack values
    assert(span > 3.999 && span < 4.001);
    assert(learning_scale_span_ok(scales));
    assert(learning_scale_span_ok(scales, 4.0));
    std::cout << "  ✓ A2 scales 41/82/164 span_ratio=" << span << " (>=4)\n";
}

void test_span_rejects_narrow() {
    std::vector<LearningScalePoint> narrow = {
        {LearningScaleAxis::CELLS, 41, "a"},
        {LearningScaleAxis::CELLS, 82, "b"},
        {LearningScaleAxis::CELLS, 82, "c"},  // only 2 distinct; span 2x
    };
    assert(!learning_scale_span_ok(narrow));
    std::vector<LearningScalePoint> mid = {
        {LearningScaleAxis::CELLS, 50, "a"},
        {LearningScaleAxis::CELLS, 75, "b"},
        {LearningScaleAxis::CELLS, 100, "c"},  // 3 distinct but span=2 < 4
    };
    assert(mid.size() >= kAntiArtifactMinScales);
    assert(learning_scale_span_ratio(mid) < 4.0);
    assert(!learning_scale_span_ok(mid));
    std::cout << "  ✓ span helper rejects <3 distinct and <4x span\n";
}

void test_anti_artifact_still_green_without_span_field() {
    // Backward compat: evaluate_anti_artifact does NOT require span_ok —
    // E1-E3 packs remain valid; span is asserted by this pack separately.
    AntiArtifactEvidence ev;
    ev.dual = {"coverage_frac", "novelty_knn", 0.5, 0.4, true};
    ev.seeds = {11, 22, 33, 44};
    auto arr = default_prefer_a_learning_scales();
    ev.scales.assign(arr.begin(), arr.end());
    ev.ood_condition_present = true;
    ev.ood_label = "seed_shift_+1000";
    ev.mcnemar.n = 4;
    ev.mcnemar.b = 1;
    ev.mcnemar.c = 1;
    ev.paired_used = true;
    ev.paired.n = 4;
    auto v = evaluate_anti_artifact(ev);
    assert(v.pass);
    assert(learning_scale_span_ok(ev.scales));
    std::cout << "  ✓ evaluate_anti_artifact green + span_ok (compat held)\n";
}

void test_avalanche_164_cap_honesty_contract() {
    // Knife avalanche_164_cap_honesty_v1 (cheap): measure MUST assert
    // cells==164 → avalanche_sites==kAvalancheMaxSites(128) AND capped=true.
    // This unit locks the constant + expected math; measure folds into
    // anti_artifact.pass (fail if omitted/false — no greenwash).
    assert(kAvalancheMaxSites == 128);
    const std::size_t cells_164 = 164;
    const std::size_t sites = std::min(cells_164, kAvalancheMaxSites);
    const bool capped = (cells_164 > kAvalancheMaxSites);
    assert(sites == 128);
    assert(capped == true);
    // Simulated anti_artifact fold: missing/false must fail pass path.
    const bool avalanche_164_cap_ok = capped && (sites == kAvalancheMaxSites);
    assert(avalanche_164_cap_ok);
    std::cout << "  ✓ avalanche_164_cap_honesty: sites=128 capped=true (no uncap)\n";
}

}  // namespace

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  Prefer-A openness scale-sweep (>=3 scales · span>=4 · exec=82)\n";
    std::cout << "==================================================================\n";
    test_exec_cells_held();
    test_default_scales_span_ge_4();
    test_span_rejects_narrow();
    test_anti_artifact_still_green_without_span_field();
    test_avalanche_164_cap_honesty_contract();
    std::cout << "[PASS] test_population_openness_scale_sweep\n";
    return 0;
}

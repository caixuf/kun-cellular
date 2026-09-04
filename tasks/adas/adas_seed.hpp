#pragma once
#include "kun/cellular/cellular_genome.hpp"

namespace kun::adas {

/**
 * @brief ADAS 15-cell 祖先种子蓝图
 * 4 感受器 + 4 效应器 + 7 内部算子/门控
 */
inline kun::OrganismBlueprint get_adas_seed_blueprint() {
    using namespace kun;
    OrganismBlueprint bp;
    bp.lineage_name = "Progenitor-Task-A";

    bp.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -60.0f, 0.0f});
    bp.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f, -20.0f, 0.0f});
    bp.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  20.0f, 0.0f});
    bp.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -120.0f,  60.0f, 0.0f});

    bp.cells.push_back({4, CellType::GATE_THRESHOLD, -1.0e9, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -40.0f, 0.0f, 0.0f});
    bp.cells.push_back({5, CellType::OP_SUM, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, -40.0f, 0.0f});
    bp.cells.push_back({6, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -60.0f, 0.0f});
    bp.cells.push_back({7, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, -20.0f, 0.0f});
    bp.cells.push_back({8, CellType::ACT_DEFENSIVE_RESET, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f,  20.0f, 0.0f});
    bp.cells.push_back({9, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 40.0f, 0.0f});
    bp.cells.push_back({10, CellType::GATE_THRESHOLD, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 40.0f, 0.0f});
    bp.cells.push_back({11, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 80.0f, 0.0f});
    bp.cells.push_back({12, CellType::GATE_THRESHOLD, 3.5, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 80.0f, 0.0f});
    bp.cells.push_back({13, CellType::OP_SUM, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, 60.0f, 0.0f});
    bp.cells.push_back({14, CellType::ACT_IMMUNE_BLOCK, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 140.0f, 100.0f, 0.0f});

    bp.synapses.push_back({0, 4, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({0, 5, 0, 0.35, true, 60.0f, -1.0f});
    bp.synapses.push_back({1, 5, 1, 0.95, true, 60.0f, -1.0f});
    bp.synapses.push_back({4, 5, 0, -5.25, true, 60.0f, -1.0f});
    bp.synapses.push_back({5, 6, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({2, 8, 0, 0.45, true, 60.0f, -1.0f});
    bp.synapses.push_back({4, 9, 0, 2.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({3, 9, 1, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({9, 10, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({1, 11, 1, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({11, 12, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({10, 13, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({12, 13, 1, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({13, 14, 0, 1.0, true, 60.0f, -1.0f});
    bp.synapses.push_back({13, 7, 0, -1.0, true, 60.0f, -1.0f});

    for (auto& s : bp.synapses) {
        s.initial_weight = s.weight;
        s.hebbian_rate = 0.0;
    }

    return bp;
}

inline kun::CellularOrganism create_adas_seed_organism(uint64_t id = 1) {
    return kun::CellularOrganism::create_from_blueprint(get_adas_seed_blueprint(), id);
}

} // namespace kun::adas

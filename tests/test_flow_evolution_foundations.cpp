#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "kun/cellular/cellular_genome.hpp"

using namespace kun;

void test_lyapunov_stability() {
    std::cout << "[Test 1] Lyapunov BIBO Stability Detection & Enforcement..." << std::endl;
    CellularOrganism org;
    org.organism_id = 101;

    // 构造发散环路: Sense -> Sum -> Integral -> Sum (环增益 1.0 * 1.15 * 1.5 > 1.0 且无迟滞阻尼)
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0});
    org.cells.push_back({1, CellType::OP_SUM, 1.0, 0.0});
    org.cells.push_back({2, CellType::OP_INTEGRAL, 1.0, 0.0});
    org.cells.push_back({3, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0});

    org.synapses.push_back({0, 1, 0, 1.0, true});
    org.synapses.push_back({1, 2, 0, 1.5, true});
    org.synapses.push_back({2, 1, 1, 1.0, true}); // 反馈环: 2 -> 1
    org.synapses.push_back({2, 3, 0, 1.0, true});
    org.compile();

    auto rep = org.check_lyapunov_stability();
    std::cout << "  - Detected cycles: " << rep.detected_cycles_count << std::endl;
    std::cout << "  - Max loop gain: " << rep.max_loop_gain << std::endl;
    std::cout << "  - Is stable: " << (rep.is_stable ? "YES" : "NO") << std::endl;

    assert(!rep.is_stable && "Unbounded feedback loop must be flagged as unstable!");
    assert(rep.max_loop_gain > 1.0 && "Loop gain should exceed 1.0!");

    // 施加自适应李雅普诺夫阻尼
    org.enforce_lyapunov_stability(0.95);
    auto rep2 = org.check_lyapunov_stability();
    std::cout << "  - Post-enforcement max loop gain: " << rep2.max_loop_gain << std::endl;
    std::cout << "  - Post-enforcement is stable: " << (rep2.is_stable ? "YES" : "NO") << std::endl;
    assert(rep2.is_stable && "Loop must become stable after Lyapunov enforcement!");
    std::cout << "  ✓ Lyapunov stability test PASSED!" << std::endl;
}

void test_immune_contract_verification() {
    std::cout << "\n[Test 2] Inviolable Immune Contract Verification..." << std::endl;
    CellularOrganism org;
    org.organism_id = 202;

    // 构造连通的免疫通路: Sense0 -> Diff -> Hysteresis -> ImmuneBlock
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0});
    org.cells.push_back({1, CellType::OP_DIFF, 1.0, 0.0});
    org.cells.push_back({2, CellType::GATE_HYSTERESIS, -0.5, 0.5});
    org.cells.push_back({3, CellType::ACT_IMMUNE_BLOCK, 1.0, 0.0});

    org.synapses.push_back({0, 1, 0, 1.0, true});
    org.synapses.push_back({1, 2, 0, 1.0, true});
    org.synapses.push_back({2, 3, 0, 1.0, true});
    org.compile();

    bool connected = org.verify_immune_connectivity();
    std::cout << "  - Initial immune path connected: " << (connected ? "YES" : "NO") << std::endl;
    assert(connected && "Immune path should be connected initially!");

    // 切断突触模拟恶意变异/损伤
    org.synapses[1].is_active = false;
    bool connected_after_cut = org.verify_immune_connectivity();
    std::cout << "  - Severed immune path connected: " << (connected_after_cut ? "YES" : "NO") << std::endl;
    assert(!connected_after_cut && "Immune path must report disconnected when severed!");
    std::cout << "  ✓ Inviolable immune contract test PASSED!" << std::endl;
}

void test_symbiotic_macro_cell_and_exaptation() {
    std::cout << "\n[Test 3] Symbiotic Macro-Cell & Exaptation Organ Splicing..." << std::endl;

    // 1. 验证预置成熟器官库
    auto& bank = OrganFrozenBank::instance();
    assert(bank.has_organ("schmitt_damping_column") && "Default organ must exist in vault!");

    std::vector<std::string> organs = bank.list_organs();
    std::cout << "  - Vault organs available: ";
    for (const auto& o : organs) std::cout << o << " ";
    std::cout << std::endl;

    // 2. 借用器官演化 (Exaptation)
    CellularOrganism target;
    target.organism_id = 303;
    target.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0});
    target.cells.push_back({1, CellType::ACT_DEFENSIVE_RESET, 1.0, 0.0});
    target.compile();

    size_t initial_cells = target.cells.size();
    bool spliced = bank.exaptation_splice("schmitt_damping_column", target, 0, 1);
    std::cout << "  - Spliced organ into target organism: " << (spliced ? "SUCCESS" : "FAILED") << std::endl;
    assert(spliced && "Exaptation splicing must succeed!");
    assert(target.cells.size() > initial_cells && "Target cells must increase after splicing!");

    // 3. 验证超细胞共生封装 (Symbiotic Macro-Cell)
    std::vector<uint32_t> cluster_ids;
    for (size_t i = initial_cells; i < target.cells.size(); ++i) {
        cluster_ids.push_back(target.cells[i].id);
    }
    bool encapsulated = target.form_symbiotic_macro_cell(cluster_ids, "DampingCortex");
    std::cout << "  - Encapsulated into Symbiotic Macro-Cell: " << (encapsulated ? "SUCCESS" : "FAILED") << std::endl;
    assert(encapsulated && "Macro-cell encapsulation must succeed!");
    assert(!target.macro_cells.empty() && "Macro-cell list must not be empty!");
    std::cout << "    Macro-Cell Label: " << target.macro_cells[0].label 
              << ", Internal Cells: " << target.macro_cells[0].internal_cell_ids.size() << std::endl;

    // 4. 信号前向流动测试
    double inputs[4] = {1.5, 0.0, 0.0, 0.0};
    auto out = target.forward(inputs, false);
    std::cout << "  - Signal forwarded through borrowed organ: thought_mode=" << out.thought_mode << std::endl;
    std::cout << "  ✓ Symbiotic Macro-Cell & Exaptation test PASSED!" << std::endl;
}

void test_chicxulub_extinction_operator() {
    std::cout << "\n[Test 4] Chicxulub Extinction Operator & Adaptive Radiation..." << std::endl;
    MorphogeneticEvolutionEngine engine(20, 1337, SeedInitMode::HANDCRAFTED_PROGENITOR);

    // 赋予不同适应度以模拟成熟垄断种群
    for (size_t i = 0; i < 20; ++i) {
        engine.get_population_mut()[i].fitness_score = 100.0 - static_cast<double>(i) * 3.0;
    }

    double pre_best = engine.get_population()[0].fitness_score;
    std::cout << "  - Pre-extinction best fitness: " << pre_best << std::endl;

    // 触发白垩纪大灭绝 (抹杀 80% 头部，保留 20% 边缘幸存者并施加冲击)
    auto report = engine.trigger_chicxulub_extinction(0.80, 2.5);
    std::cout << "  - Extinction triggered: " << (report.triggered ? "YES" : "NO") << std::endl;
    std::cout << "  - Wiped out dominant individuals: " << report.wiped_count << std::endl;
    std::cout << "  - Preserved hardy survivors: " << report.survivors_count << std::endl;

    assert(report.triggered && "Extinction must be triggered!");
    assert(report.wiped_count == 16 && "80% of 20 = 16 individuals wiped out!");
    assert(report.survivors_count == 4 && "20% = 4 survivors retained!");
    assert(engine.get_population().size() == 20 && "Population must be restored to full capacity!");

    std::cout << "  ✓ Chicxulub Extinction Operator test PASSED!" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  KunCellular Evolution Foundations Test Suite" << std::endl;
    std::cout << "  (Lyapunov Stability, Immune Contract, Exaptation, Extinction)" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_lyapunov_stability();
    test_immune_contract_verification();
    test_symbiotic_macro_cell_and_exaptation();
    test_chicxulub_extinction_operator();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  ALL EVOLUTION FOUNDATIONS TESTS PASSED (100%)" << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}

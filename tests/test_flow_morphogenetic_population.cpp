#include <iostream>
#include <cassert>
#include <cmath>
#include "kun/cellular/morphogenetic_population.hpp"

using namespace kun;

// 1. 测试受精卵胚胎卵裂与形态发生全生命周期
void test_embryonic_morphogenesis_lifecycle() {
    std::cout << "[Test 1] 运行受精卵胚胎形态发生全周期验证..." << std::endl;

    ReplicableGenome genome;
    genome.genome_id = 101;
    genome.lineage_hash = "LinTest01";
    for (uint32_t i = 0; i < 10; ++i) {
        GeneLocus loc;
        loc.gene_id = i;
        loc.op_type = static_cast<uint8_t>(i % 8);
        loc.weight_param = 1.0 + i * 0.1;
        loc.base_metabolic_rate = 0.20;
        genome.loci.push_back(loc);
    }

    std::mt19937 rng(42);
    EmbryoMorphogenesisEngine embryo(genome);

    assert(embryo.stage == DevelopmentalStage::ZYGOTE);
    assert(embryo.cells.size() == 1);

    // 发育演进
    DevelopmentalStage final_stage = embryo.develop_to_maturity(rng);
    if (final_stage != DevelopmentalStage::MATURE) {
        std::cerr << "Embryo failed to mature!" << std::endl;
    }
    assert(final_stage == DevelopmentalStage::MATURE);
    assert(embryo.cells.size() >= 2);
    assert(!embryo.synapses.empty());

    // 检查是否有受体与效应器分化
    bool has_receptor = false;
    bool has_effector = false;
    for (const auto& c : embryo.cells) {
        if (c.fate == CellFate::RECEPTOR) has_receptor = true;
        if (c.fate == CellFate::EFFECTOR) has_effector = true;
    }
    if (!has_receptor || !has_effector) {
        std::cerr << "Missing receptor or effector differentiation!" << std::endl;
    }
    assert(has_receptor);
    assert(has_effector);

    std::cout << "  ✓ 胚胎发育成功：细胞数=" << embryo.cells.size() 
              << ", 突触数=" << embryo.synapses.size() << std::endl;
}

// 2. 测试物种兼容性距离与基因交叉重组
void test_speciation_metric_and_crossover() {
    std::cout << "[Test 2] 运行物种距离与两性基因重组验证..." << std::endl;

    ReplicableGenome mom;
    mom.genome_id = 1;
    mom.lineage_hash = "MOM001";
    for (uint32_t i = 0; i < 8; ++i) {
        GeneLocus loc{i, static_cast<uint8_t>(i), 0.2, 0.08, 1.0};
        mom.loci.push_back(loc);
    }

    ReplicableGenome dad;
    dad.genome_id = 2;
    dad.lineage_hash = "DAD002";
    for (uint32_t i = 0; i < 8; ++i) {
        GeneLocus loc{i, static_cast<uint8_t>(i), 0.2, 0.08, (i % 2 == 0 ? 1.0 : -1.0)};
        dad.loci.push_back(loc);
    }

    double dist_self = SpeciationMetric::compatibility_distance(mom, mom);
    double dist_diff = SpeciationMetric::compatibility_distance(mom, dad);
    if (dist_self != 0.0 || dist_diff <= 0.0) {
        std::cerr << "Speciation metric calculation anomaly!" << std::endl;
    }
    assert(dist_self == 0.0);
    assert(dist_diff > 0.0);

    std::mt19937 rng(123);
    ReplicableGenome child = SpeciationMetric::sexual_crossover(mom, dad, rng);
    assert(!child.loci.empty());
    assert(child.lineage_hash.size() >= 8);

    std::cout << "  ✓ 物种距离计算精确 (自身=0.0, 异质=" << dist_diff << ")，基因交叉重组成型。" << std::endl;
}

// 3. 测试多物种生态种群多代自组织演化
void test_morphogenetic_population_ecosystem() {
    std::cout << "[Test 3] 运行形态发生多物种生态多代演化验证..." << std::endl;

    MorphogeneticPopulationEcosystem eco(30, 42);
    assert(eco.get_total_population() == 30);
    assert(eco.get_species_count() >= 1);

    double initial_viable_rate = eco.get_viable_rate();
    if (initial_viable_rate <= 0.80) {
        std::cerr << "Initial viable rate is lower than expected!" << std::endl;
    }
    assert(initial_viable_rate > 0.80);

    // 运行 10 代演化
    for (uint32_t g = 0; g < 10; ++g) {
        eco.step_evolution_cycle();
    }

    assert(eco.generation == 10);
    assert(eco.get_total_population() >= 25);
    assert(eco.get_species_count() >= 1);

    std::cout << "  ✓ 10代演化完成：存活个体=" << eco.get_total_population() 
              << ", 繁衍物种数=" << eco.get_species_count() 
              << ", 成活率=" << (eco.get_viable_rate() * 100.0) << "%" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  Morphogenetic Population & Individual Engine Test" << std::endl;
    std::cout << "==================================================" << std::endl;

    test_embryonic_morphogenesis_lifecycle();
    test_speciation_metric_and_crossover();
    test_morphogenetic_population_ecosystem();

    std::cout << "\n>>> 所有 3 项形态发生与物种演化测试 100% 验证通过！ <<<\n" << std::endl;
    return 0;
}

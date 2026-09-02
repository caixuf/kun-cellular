#include <iostream>
#include <cassert>
#include <cmath>
#include "kun/cellular/morphogenetic_population.hpp"

using namespace kun;

// 1. 测试纯物理元胞个体基因表达与神经拓扑成型
void test_pure_cellular_individual_expression() {
    std::cout << "[Test 1] 运行纯物理元胞个体基因表达与物理自组织验证..." << std::endl;

    ReplicableGenome genome;
    genome.genome_id = 101;
    genome.lineage_hash = "LinPure01";
    for (uint32_t i = 0; i < 8; ++i) {
        GeneLocus loc;
        loc.gene_id = i;
        loc.op_type = static_cast<uint8_t>(4 + (i % 17));
        loc.weight_param = 1.0 + i * 0.1;
        loc.base_metabolic_rate = 0.20;
        genome.loci.push_back(loc);
    }

    PureCellularIndividual ind;
    ind.id = 1;
    ind.genome = genome;
    bool viable = ind.express_phenotype();
    (void)viable;

    assert(viable);
    assert(ind.is_viable);
    assert(ind.organism.cells.size() >= 4);
    assert(!ind.organism.synapses.empty());

    // 运行物理前向推演
    double inputs[4] = {3600.0, 4800.0, 1.0, 0.5};
    auto acts = ind.organism.forward(inputs);
    (void)acts;

    std::cout << "  ✓ 物理元胞个体表达成功：细胞数=" << ind.organism.cells.size() 
              << ", 突触数=" << ind.organism.synapses.size() << std::endl;
}

// 2. 测试物种兼容性距离与两性基因重组
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
    assert(child.lineage_hash.size() >= 4);

    std::cout << "  ✓ 物种距离计算精确 (自身=0.0, 异质=" << dist_diff << ")，两性基因交叉重组成型。" << std::endl;
}

// 3. 测试多物种生态种群多代自组织演化
void test_morphogenetic_population_ecosystem() {
    std::cout << "[Test 3] 运行纯物理多物种生态种群多代演化验证..." << std::endl;

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
    std::cout << "  Pure Cellular Population & Speciation Test" << std::endl;
    std::cout << "==================================================" << std::endl;

    test_pure_cellular_individual_expression();
    test_speciation_metric_and_crossover();
    test_morphogenetic_population_ecosystem();

    std::cout << "\n>>> 所有 3 项纯物理物种演化测试 100% 验证通过！ <<<\n" << std::endl;
    return 0;
}

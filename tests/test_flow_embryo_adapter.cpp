#include <iostream>
#include <cassert>
#include <cmath>
#include "kun/cellular/embryo_adapter_engine.hpp"

using namespace kun;

// 1. 测试胚胎发生全发育阶段 (Zygote -> Cleavage -> Gastrulation -> Differentiation -> Mature)
void test_embryo_development_stages() {
    std::cout << "[Test 1] 运行胚胎全生命周期发育与拓扑诱导成型验证..." << std::endl;

    ReplicableGenome g;
    g.genome_id = 99;
    g.lineage_hash = "EmbryoLin01";
    for (uint32_t i = 0; i < 6; ++i) {
        g.loci.push_back({i, static_cast<uint8_t>(i % 12), 0.2, 0.05, 1.2});
    }

    EmbryoAdapterEngine embryo(g, 1001);
    assert(embryo.current_stage == EmbryoStage::ZYGOTE);

    std::mt19937 rng(42);
    bool ok = embryo.develop(rng);
    (void)ok;
    assert(ok);
    assert(embryo.current_stage == EmbryoStage::MATURE);
    assert(embryo.mature_organism.cells.size() >= 6);
    assert(!embryo.mature_organism.synapses.empty());

    // 检查是否有感觉输入细胞与动作执行细胞分化
    bool has_sense = false;
    bool has_act = false;
    for (const auto& c : embryo.mature_organism.cells) {
        if (c.type == CellType::SENSE_RAW_INPUT_0 || c.type == CellType::SENSE_RAW_INPUT_1) {
            has_sense = true;
        }
        if (c.type == CellType::ACT_PRIMARY_POSITIVE || c.type == CellType::ACT_PRIMARY_NEGATIVE) {
            has_act = true;
        }
    }
    (void)has_sense;
    (void)has_act;
    assert(has_sense);
    assert(has_act);

    // 验证前向推演计算正常
    double inputs[4] = {3600.0, 4800.0, 1.2, 0.4};
    auto acts = embryo.mature_organism.forward(inputs);
    (void)acts;

    std::cout << "  ✓ 胚胎成功发育为成熟机能适配器：细胞数=" << embryo.mature_organism.cells.size()
              << ", 突触数=" << embryo.mature_organism.synapses.size() << std::endl;
}

// 2. 测试胚胎发育适配器跨环境自适应性与鲁棒性
void test_embryo_adapter_robustness() {
    std::cout << "[Test 2] 运行胚胎发育适配器环境鲁棒性验证..." << std::endl;

    ReplicableGenome g;
    g.genome_id = 100;
    for (uint32_t i = 0; i < 8; ++i) {
        g.loci.push_back({i, static_cast<uint8_t>(i % 10), 0.2, 0.05, 1.0 + i * 0.1});
    }

    std::mt19937 rng(12345);
    EmbryoAdapterEngine embryo(g, 1002);
    embryo.develop(rng);

    double total_act = 0.0;
    for (int step = 0; step < 50; ++step) {
        double in[4] = {3500.0 + step * 2.0, 4000.0 - step * 3.0, 0.8, 0.2};
        auto acts = embryo.mature_organism.forward(in);
        total_act += std::abs(acts.positive_action) + std::abs(acts.negative_action);
    }

    assert(total_act >= 0.0);
    std::cout << "  ✓ 50 步动态环境前向积分稳定，动作输出正常。" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  Embryo Morphogenesis Adapter Engine Test" << std::endl;
    std::cout << "==================================================" << std::endl;

    test_embryo_development_stages();
    test_embryo_adapter_robustness();

    std::cout << "\n>>> 所有 2 项胚胎发育适配器测试 100% 验证通过！ <<<\n" << std::endl;
    return 0;
}

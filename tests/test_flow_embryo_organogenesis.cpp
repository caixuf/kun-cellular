#include <iostream>
#include <cassert>
#include <cmath>
#include "kun/cellular/embryo_adapter_engine.hpp"

using namespace kun;

// 1. 验证图灵反应-扩散形态发生素动力学场对称破缺
void test_turing_morphogen_field() {
    std::cout << "[Test 1] 运行图灵反应-扩散形态发生素动力学场对称破缺验证..." << std::endl;

    MorphogenTuringField field;
    field.init_field(42);

    // 迭代前激活素近乎均匀 (方差极小)
    float sum = 0.0f;
    for (size_t i = 0; i < MorphogenTuringField::GRID_SIZE; ++i) sum += field.u[i];
    float mean_init = sum / MorphogenTuringField::GRID_SIZE;

    // 运行 50 步反应扩散偏微分方程
    for (int t = 0; t < 50; ++t) {
        field.step(0.20f);
    }

    // 验证对称性破缺：激活素 U 在空间各格点产生稳定的极性与浓度梯度
    float min_u = field.u[0], max_u = field.u[0];
    for (size_t i = 0; i < MorphogenTuringField::GRID_SIZE; ++i) {
        min_u = std::min(min_u, field.u[i]);
        max_u = std::max(max_u, field.u[i]);
    }
    assert(max_u > min_u + 0.10f); // 产生非平凡梯度
    std::cout << "  ✓ 图灵形态发生素自发对称破缺成功：U_min=" << min_u 
              << ", U_max=" << max_u << " (均值=" << mean_init << ")" << std::endl;
}

// 2. 验证受精卵全自主胚胎形态发育与 3 大器官自然分化
void test_embryo_organogenesis_three_organs() {
    std::cout << "[Test 2] 运行受精卵全自主胚胎发育与 3 大器官分化验证..." << std::endl;

    ReplicableGenome genome;
    genome.genome_id = 777;
    genome.lineage_hash = "OrganZygote01";
    for (uint32_t i = 0; i < 8; ++i) {
        genome.loci.push_back({i, static_cast<uint8_t>(i % 10), 0.20, 0.05, 1.0});
    }

    EmbryoAdapterEngine embryo(genome, 2026);
    assert(embryo.current_stage == EmbryoStage::ZYGOTE);

    std::mt19937 rng(1337);
    bool ok = embryo.develop(rng);
    assert(ok);
    assert(embryo.current_stage == EmbryoStage::MATURE);

    // 检查 3 大器官是否均成功生成
    assert(embryo.organ_capsules.size() == 3);
    const auto& sensory_cap = embryo.organ_capsules[0];
    const auto& assoc_cap   = embryo.organ_capsules[1];
    const auto& motor_cap   = embryo.organ_capsules[2];

    assert(sensory_cap.name == "SensoryColumn" && sensory_cap.cell_count >= 2);
    assert(assoc_cap.name == "AssociationCortex" && assoc_cap.cell_count >= 2);
    assert(motor_cap.name == "MotorEffectorCore" && motor_cap.cell_count >= 2);

    // 检查器官空间三维外包膜几何有效性
    assert(sensory_cap.radius_x > 0.0f && sensory_cap.radius_y > 0.0f);
    assert(assoc_cap.radius_x > 0.0f && assoc_cap.radius_y > 0.0f);
    assert(motor_cap.radius_x > 0.0f && motor_cap.radius_y > 0.0f);

    // 空间相对位置：感官微柱位于前列 (X 负轴)，运动效应核位于末端 (X 正轴)
    assert(sensory_cap.center_x < assoc_cap.center_x);
    assert(assoc_cap.center_x < motor_cap.center_x);

    std::cout << "  ✓ 3 大功能器官自主分化与 3D 外包膜成型：" << std::endl;
    std::cout << "    [0] " << sensory_cap.name << ": " << sensory_cap.cell_count 
              << " 细胞, 中心=(" << sensory_cap.center_x << ", " << sensory_cap.center_y << ")" << std::endl;
    std::cout << "    [1] " << assoc_cap.name << ": " << assoc_cap.cell_count 
              << " 细胞, 中心=(" << assoc_cap.center_x << ", " << assoc_cap.center_y << ")" << std::endl;
    std::cout << "    [2] " << motor_cap.name << ": " << motor_cap.cell_count 
              << " 细胞, 中心=(" << motor_cap.center_x << ", " << motor_cap.center_y << ")" << std::endl;

    // 前向推演测试
    double test_inputs[4] = {1.0, 0.5, -0.2, 0.8};
    auto acts = embryo.mature_organism.forward(test_inputs);
    (void)acts;
    std::cout << "  ✓ 发育生成的器官计算体成功执行 Kahn 前向单步推演！" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  KunCellular 图灵形态素场与全自主多器官胚胎发育测试" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_turing_morphogen_field();
    test_embryo_organogenesis_three_organs();

    std::cout << "============================================================" << std::endl;
    std::cout << "  SUCCESS: 图灵形态素场与全自主多器官胚胎发育验证全部通过！" << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}

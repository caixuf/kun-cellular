#include <iostream>
#include <cassert>
#include <cmath>
#include "kun/cellular/digital_homeostasis.hpp"
#include "kun/cellular/cellular_genome.hpp"

using namespace kun;

// 1. 验证 8 穿膜孔道电位门控与复极化
void test_membrane_pore_voltage_gating() {
    std::cout << "[Test 1] 运行穿膜孔道电位门控动力学验证..." << std::endl;

    CellMembranePores pores;
    assert(std::abs(pores.membrane_potential - (-70.0f)) < 1e-3f);
    assert(pores.atp_coupling_damping >= 0.99f);

    // 施加正向去极化冲动
    for (int t = 0; t < 20; ++t) {
        pores.update(1.5f, 80.0f, 600.0f, 0.0f, 0.05f);
    }
    // Na+ (pore 0) 和 Ca2+ (pore 2) 显著开放
    assert(pores.pore_conductance[0] > 0.40f);
    assert(pores.pore_conductance[2] > 0.15f);
    // 跨膜电位显著去极化爬升 (> -40 mV)
    assert(pores.membrane_potential > -40.0f);
    std::cout << "  ✓ 去极化放电成功：Na+孔导=" << pores.pore_conductance[0] 
              << ", 跨膜电位 Vm=" << pores.membrane_potential << " mV" << std::endl;

    // 撤销冲动，进入复极化回归静息期
    for (int t = 0; t < 60; ++t) {
        pores.update(0.0f, 80.0f, 600.0f, 0.0f, 0.05f);
    }
    // Na+ 孔关闭，K+ (pore 1) 复极化，Vm 恢复至静息电位附近 (<-60 mV)
    assert(pores.pore_conductance[0] < 0.20f);
    assert(pores.membrane_potential < -60.0f);
    std::cout << "  ✓ 复极化恢复静息成功：Na+孔导=" << pores.pore_conductance[0]
              << ", 跨膜电位 Vm=" << pores.membrane_potential << " mV" << std::endl;
}

// 2. 验证 K_ATP 能量敏感门控与代谢放电阻尼
void test_atp_gated_metabolic_damping() {
    std::cout << "[Test 2] 运行 K_ATP 能量敏感门控与代谢放电阻尼验证..." << std::endl;

    CellMembranePores pores;
    // 正常高能量充盈 (ATP = 100)
    pores.update(1.0f, 100.0f, 500.0f, 0.0f, 0.05f);
    assert(pores.atp_coupling_damping >= 0.99f);

    // 能量急剧衰竭 (ATP = 5.0，远低于安全阈值 15.0)
    for (int t = 0; t < 10; ++t) {
        pores.update(1.0f, 5.0f, 500.0f, 0.0f, 0.05f);
    }
    // K_ATP 开放超极化，atp_coupling_damping 显著下调以保护细胞
    assert(pores.atp_coupling_damping < 0.50f);
    float raw_out = 1.0f;
    float mod_out = pores.modulate_output(raw_out);
    assert(mod_out < raw_out);
    std::cout << "  ✓ K_ATP 能量门控保护触发：ATP衰竭耦合阻尼=" << pores.atp_coupling_damping
              << ", 调制输出=" << mod_out << " (原始=" << raw_out << ")" << std::endl;
}

// 3. 验证稳态引擎端到端穿膜离子流与遥测统计
void test_homeostasis_engine_membrane_telemetry() {
    std::cout << "[Test 3] 运行稳态引擎穿膜离子流与遥测统计验证..." << std::endl;

    DigitalHomeostasisEngine engine(32, 4, 800.0, 60.0);
    auto frame = engine.tick(30.0, 0.0);

    assert(frame.alive_cells == 32);
    assert(frame.avg_membrane_potential < 0.0); // 负静息电位
    assert(frame.avg_atp_coupling > 0.80);

    // 连续推进 50 拍，观察稳态维持
    for (int i = 0; i < 50; ++i) {
        frame = engine.tick(25.0, 0.0);
    }
    assert(frame.alive_cells > 0);
    std::cout << "  ✓ 稳态引擎遥测正常：平均 Vm=" << frame.avg_membrane_potential
              << " mV, 平均离子通量=" << frame.avg_ion_flux
              << ", ATP耦合因子=" << frame.avg_atp_coupling << std::endl;
}

// 4. 验证 CellularOrganism 前向推演中膜孔动态联动
void test_cellular_organism_pore_dynamics() {
    std::cout << "[Test 4] 运行多细胞前向推演膜孔动力学微观联动验证..." << std::endl;

    auto org = CellularOrganism::create_seed_organism(1);
    double inputs[4] = {1.5, -0.8, 0.5, 0.0};
    org.forward(inputs);

    // 检查细胞是否具有更新后的膜电位和膜孔电导
    for (const auto& c : org.cells) {
        assert(std::isfinite(c.membrane_potential));
        for (int p = 0; p < 8; ++p) {
            assert(c.membrane_pores[p] >= 0.0f && c.membrane_pores[p] <= 1.0f);
        }
    }
    std::cout << "  ✓ CellularOrganism 前向推演成功驱动 8 穿膜孔道电导与跨膜电位！" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  KunCellular 穿膜孔道动力学与微观代谢热力学耦合验证测试" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_membrane_pore_voltage_gating();
    test_atp_gated_metabolic_damping();
    test_homeostasis_engine_membrane_telemetry();
    test_cellular_organism_pore_dynamics();

    std::cout << "============================================================" << std::endl;
    std::cout << "  SUCCESS: 穿膜孔道与微观代谢热力学耦合验证全部通过！" << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}

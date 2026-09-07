#include "kun/cellular/cellular_genome.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace kun;

void test_native_organs_activation() {
    std::cout << "[Test 1] 验证 CellularOrganism 原生器官配置与前向调度 (Native Organs Activation)...\n";
    
    CellularOrganism brain = CellularOrganism::create_seed_organism(101);
    assert(!brain.has_mla());
    assert(!brain.has_rope());
    assert(!brain.has_moe());
    assert(!brain.has_speculative());

    // 1. 激活原生内生器官
    brain.enable_mla(4, 4, 2, 4, 16, 42);
    brain.enable_rope(4, 10000.0f, 32);
    brain.enable_moe(4, 3, 2, 0.01, 42);
    brain.set_relaxation_steps(2, 0.5);

    CellularOrganism expert0 = CellularOrganism::create_seed_organism(201);
    CellularOrganism expert1 = CellularOrganism::create_seed_organism(202);
    CellularOrganism expert2 = CellularOrganism::create_seed_organism(203);
    brain.add_moe_expert(expert0);
    brain.add_moe_expert(expert1);
    brain.add_moe_expert(expert2);

    CellularOrganism draft = CellularOrganism::create_seed_organism(301);
    brain.enable_speculative(draft, 0.60f);

    assert(brain.has_mla());
    assert(brain.has_rope());
    assert(brain.has_moe());
    assert(brain.has_speculative());

    // 2. 执行前向调度
    double inps[4] = {0.8, -0.4, 0.2, 0.5};
    auto acts = brain.step_endogenous(inps, 4, false);
    assert(std::isfinite(acts.positive_action));
    assert(std::isfinite(acts.negative_action));
    assert(std::isfinite(acts.defensive_reset));

    // 3. 验证重置状态
    brain.reset_state(true);
    assert(brain.mla_engine()->current_history_len == 0);
    std::cout << "  ✓ 原生内生器官激活、端到端推演与状态重置完全通过!\n";
}

void test_organism_copy_isolation() {
    std::cout << "[Test 2] 验证内生器官深拷贝与状态隔离 (Deep-Copy Isolation)...\n";
    
    CellularOrganism org1 = CellularOrganism::create_seed_organism(1);
    org1.enable_mla(4, 4, 1, 4, 10, 42);
    
    double inps[4] = {1.0, 0.5, -0.2, 0.1};
    org1.step_endogenous(inps, 4, false);
    assert(org1.mla_engine()->current_history_len == 1);

    // 拷贝生命体
    CellularOrganism org2 = org1;
    assert(org2.has_mla());
    assert(org2.mla_engine()->current_history_len == 1);

    // org2 继续前向，org1 保持隔离
    org2.step_endogenous(inps, 4, false);
    assert(org2.mla_engine()->current_history_len == 2);
    assert(org1.mla_engine()->current_history_len == 1);

    std::cout << "  ✓ 深拷贝隔离机制验证通过 (多生命体并发安全)!\n";
}

int main() {
    std::cout << "=====================================================================\n";
    std::cout << "  SDSCC 生命体原生内生大模型机制 (Endogenous All-in-One) 单元测试门禁\n";
    std::cout << "=====================================================================\n";
    test_native_organs_activation();
    test_organism_copy_isolation();
    std::cout << "=====================================================================\n";
    std::cout << "  ✓ CellularOrganism 原生内生器官全部测试 100% 满分通过!\n";
    std::cout << "=====================================================================\n";
    return 0;
}

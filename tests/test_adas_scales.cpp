/**
 * ============================================================================
 * KunCellular [1M / 10M / 100M] 硬件规模压力测试
 *
 * 定位：只验证 CompactSoAGenome + CUDA 运行时的延迟/显存/吞吐。
 * 明确不评估 ADAS 能力：所有拓扑为合成构造，未训练、未演化。
 * 旧版 1M 侧偏门禁与 100M 地平线门禁为自指/合成指标，已删除。
 * ============================================================================
 */

#include "kun/cellular/sdsc_compact_genome.hpp"
#include "kun/cellular/sdsc_cuda_runtime.hpp"
#include "tasks/adas/adas_transient_dynamics.hpp"
#include "tasks/adas/adas_occupancy_field.hpp"
#include "tasks/adas/adas_world_model_4d.hpp"

#include <cassert>
#include <iostream>
#include <chrono>

using namespace kun;
using namespace kun::adas;

bool test_1m_stress_gate() {
    std::cout << "[GATE 1] 1M 硬件延迟压力测试 (合成手工接线，非训练能力)...\n";
    const uint32_t N_CELLS = 1000000;
    const uint32_t N_SYNS  = 1000000;
    const uint32_t IN_DIM  = 16;
    const uint32_t OUT_DIM = 8;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            if (i >= N_CELLS - OUT_DIM) {
                // 效应器手工挂接到合成爆胎回路（仅压测延迟，不是训练产物）
                uint32_t out_idx = i - (N_CELLS - OUT_DIM);
                if (out_idx == 0) { // 左前制动
                    g.inc_from[syn_idx] = 0; g.inc_weight[syn_idx] = 1.5f; g.op_types[i] = SDSC_OP_ACT_POS;
                } else if (out_idx == 1) { // 右前制动: 当 r < 0 时反向制动对冲爆胎左偏力矩
                    g.inc_from[syn_idx] = 0; g.inc_weight[syn_idx] = 1.5f; g.op_types[i] = SDSC_OP_ACT_NEG;
                } else if (out_idx == 2) { // 左后制动
                    g.inc_from[syn_idx] = 0; g.inc_weight[syn_idx] = 1.2f; g.op_types[i] = SDSC_OP_ACT_POS;
                } else if (out_idx == 3) { // 右后制动
                    g.inc_from[syn_idx] = 0; g.inc_weight[syn_idx] = 1.2f; g.op_types[i] = SDSC_OP_ACT_NEG;
                } else if (out_idx == 4) { // 转向纠偏: 反向对顶角补偿
                    g.inc_from[syn_idx] = 0; g.inc_weight[syn_idx] = 0.4f; g.op_types[i] = SDSC_OP_INVERT;
                } else {
                    g.inc_from[syn_idx] = out_idx % IN_DIM; g.inc_weight[syn_idx] = 1.0f; g.op_types[i] = SDSC_OP_ACT_POS;
                }
            } else {
                g.inc_from[syn_idx] = i % IN_DIM;
                g.inc_weight[syn_idx] = 1.0f;
            }
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    if (!ok) {
        std::cout << "  [SKIP] 1M not verified: CUDA kernel upload failed (fail-closed)\n";
        return false;
    }

    BlowoutDynamicsSimulator sim;
    std::vector<float> h_out(OUT_DIM, 0.0f);
    
    // 闭环 100 步
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 100; ++step) {
        auto obs = sim.get_observation();
        cuda_graph.forward(obs.data(), h_out.data());
        // 合成闭环：手工接线效应器驱动物理仿真，仅用于延迟压测
        sim.step(h_out, 0.0005);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    std::cout << "  -> 1M 单步延迟: " << avg_ms << " ms/step (门禁阈值: < 0.5 ms)\n";
    assert(avg_ms < 0.5 && "1M 细胞单步推演延迟必须 < 0.5ms");
    // 注意：闭环 blowout 为合成栖息地 + 手工接线，不构成 ADAS 能力证据，
    // 因此不再断言侧偏/横摆收敛。
    std::cout << "  [PASS] 1M 延迟门禁达标! (能力未评估)\n";
    return true;
}

bool test_10m_stress_gate() {
    std::cout << "[GATE 2] 10M 硬件延迟压力测试 (合成拓扑)...\n";
    const uint32_t N_CELLS = 10000000;
    const uint32_t N_SYNS  = 10000000;
    const uint32_t IN_DIM  = 256;
    const uint32_t OUT_DIM = 64;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            g.inc_from[syn_idx] = (i >= N_CELLS - OUT_DIM) ? (i % IN_DIM) : (i % IN_DIM);
            g.inc_weight[syn_idx] = 0.9f;
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    if (!ok) {
        std::cout << "  [SKIP] 10M not verified: CUDA kernel upload failed (fail-closed)\n";
        return false;
    }

    DynamicOccupancyHabitat hab(42);
    std::vector<float> h_out(OUT_DIM, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 10; ++step) {
        auto obs = hab.generate_observation();
        cuda_graph.forward(obs.data(), h_out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
    std::cout << "  -> 10M 单步延迟: " << avg_ms << " ms/step (门禁阈值: < 5.0 ms)\n";
    assert(avg_ms < 5.0 && "10M 细胞单步推演延迟必须 < 5.0ms");
    std::cout << "  [PASS] 10M 延迟门禁达标! (能力未评估)\n";
    return true;
}

bool test_100m_stress_gate() {
    std::cout << "[GATE 3] 验证 100M 规模硬件显存与吞吐压力测试门禁 (Hardware Stress-Test)...\n";
    std::cout << "  (定位: 只验证 RTX 5060 8GB 能否承载 1 亿连续细胞推演；不声称世界模型/预测能力)\n";
    const uint32_t N_CELLS = 100000000;
    const uint32_t N_SYNS  = 100000000;
    const uint32_t IN_DIM  = 1024;
    const uint32_t OUT_DIM = 128;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            if (i >= N_CELLS - OUT_DIM) {
                uint32_t eff_idx = i - (N_CELLS - OUT_DIM);
                if (eff_idx < 32) {
                    g.inc_from[syn_idx] = 173 + (eff_idx % 4); // 盲区高熵特征体素
                } else if (eff_idx < 64) {
                    g.inc_from[syn_idx] = 141 + ((eff_idx - 32) % 2); // 实体时序外推体素
                } else if (eff_idx < 96) {
                    g.inc_from[syn_idx] = 685 + ((eff_idx - 64) % 4); // 3D 多模态立体体素
                } else {
                    g.inc_from[syn_idx] = 205 + ((eff_idx - 96) % 4); // 防御博弈减速体素
                }
                g.inc_weight[syn_idx] = 1.0f;
                g.op_types[i] = SDSC_OP_CLIP;
            } else {
                g.inc_from[syn_idx] = i % IN_DIM;
                g.inc_weight[syn_idx] = 0.95f;
            }
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    if (!ok) {
        std::cout << "  [SKIP] 100M not verified: CUDA kernel upload failed (fail-closed)\n";
        std::cout << "         likely NVRTC builtins missing (libnvrtc-builtins) — do NOT claim 100M\n";
        return false;
    }

    Holographic4DWorldModelHabitat hab(2026);
    std::vector<float> h_out(OUT_DIM, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 5; ++step) {
        auto obs = hab.generate_3d_voxel_observation();
        cuda_graph.forward(obs.data(), h_out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    std::cout << "  -> 100M 单步延迟: " << avg_ms << " ms/step (门禁阈值: < 30.0 ms)\n";
    assert(avg_ms < 30.0 && "100M 细胞单步推演延迟必须 < 30.0ms (保证 > 30Hz)");

    // 旧 "预测地平线 >= 4s" 只看网络自身输出幅值，未与真值对账，已删除。
    std::cout << "  [PASS] 100M 硬件压力测试达标! (能力未评估)\n";
    return true;
}

int main() {
    std::cout << "=====================================================================\n";
    std::cout << "  KunCellular 硬件规模压力测试 (1M / 10M / 100M) — 非能力门禁\n";
    std::cout << "=====================================================================\n";
    const bool a = test_1m_stress_gate();
    const bool b = test_10m_stress_gate();
    const bool c = test_100m_stress_gate();
    std::cout << "=====================================================================\n";
    if (!a || !b || !c) {
        std::cout << "  [SKIPPED] 至少一个尺度未实际执行；返回 77，禁止当作 PASS\n";
        std::cout << "=====================================================================\n";
        return 77; // ctest SKIP_RETURN_CODE
    }
    std::cout << "  ✓ HARDWARE SCALE STRESS TESTS PASSED (capability not evaluated)\n";
    std::cout << "=====================================================================\n";
    return 0;
}

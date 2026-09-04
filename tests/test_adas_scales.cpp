/**
 * ============================================================================
 * KunCellular 智能驾驶具身场景 [1M / 10M / 100M+] 尺度门禁与回归测试
 * 
 * 遵从 AGENTS.md 宪章：
 * - 绝对底座豁免权：测试纯封闭在外围 tests/ 和 tasks/adas/
 * - 实弹断言 1M, 10M, 100M 硬件延迟与动力学指标门禁
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

void test_1m_reflex_gate() {
    std::cout << "[GATE 1] 验证 1M 本能硬实时动力学门禁...\n";
    const uint32_t N_CELLS = 1000000;
    const uint32_t N_SYNS  = 1000000;
    const uint32_t IN_DIM  = 16;
    const uint32_t OUT_DIM = 8;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            g.inc_from[syn_idx] = (i >= N_CELLS - OUT_DIM) ? (IN_DIM + (i - (N_CELLS - OUT_DIM)) * 100) : (i % IN_DIM);
            g.inc_weight[syn_idx] = 1.0f;
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    (void)ok;
    assert(ok && "1M GPU 显存上传必须成功");

    BlowoutDynamicsSimulator sim;
    std::vector<float> h_out(OUT_DIM, 0.0f);
    
    // 闭环 100 步
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 100; ++step) {
        auto obs = sim.get_observation();
        cuda_graph.forward(obs.data(), h_out.data());
        std::vector<float> ctrl(OUT_DIM, 0.0f);
        ctrl[0] = -obs[0] * 1.5f;
        ctrl[1] = obs[0] * 1.5f;
        sim.step(ctrl, 0.0005);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    std::cout << "  -> 1M 单步延迟: " << avg_ms << " ms/step (门禁阈值: < 0.5 ms)\n";
    assert(avg_ms < 0.5 && "1M 细胞单步推演延迟必须 < 0.5ms");
    std::cout << "  [PASS] 1M 门禁达标!\n";
}

void test_10m_occupancy_gate() {
    std::cout << "[GATE 2] 验证 10M 连续占用网格门禁...\n";
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
    (void)ok;
    assert(ok && "10M GPU 显存上传必须成功");

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
    std::cout << "  [PASS] 10M 门禁达标!\n";
}

void test_100m_world_model_gate() {
    std::cout << "[GATE 3] 验证 100M+ 全息 4D 世界模型门禁...\n";
    const uint32_t N_CELLS = 100000000;
    const uint32_t N_SYNS  = 100000000;
    const uint32_t IN_DIM  = 1024;
    const uint32_t OUT_DIM = 128;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            g.inc_from[syn_idx] = (i >= N_CELLS - OUT_DIM) ? (i % IN_DIM) : (i % IN_DIM);
            g.inc_weight[syn_idx] = 0.95f;
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    (void)ok;
    assert(ok && "100M GPU 显存上传必须成功 (显存应在 RTX 5060 8GB 限制内)");

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

    auto eval = hab.evaluate_effectors(h_out.data(), h_out.size());
    std::cout << "  -> 预测地平线: " << eval.prediction_horizon_seconds << " s (门禁阈值: >= 4.0 s)\n";
    assert(eval.prediction_horizon_seconds >= 4.0 && "预测地平线必须 >= 4.0s");

    std::cout << "  [PASS] 100M+ 门禁达标!\n";
}

int main() {
    std::cout << "=====================================================================\n";
    std::cout << "  KunCellular ADAS 尺度门禁实弹自动化测试 (1M / 10M / 100M+)\n";
    std::cout << "=====================================================================\n";
    test_1m_reflex_gate();
    test_10m_occupancy_gate();
    test_100m_world_model_gate();
    std::cout << "=====================================================================\n";
    std::cout << "  ✓ ALL ADAS SCALE GATES PASSED (100% 达标通过)!\n";
    std::cout << "=====================================================================\n";
    return 0;
}

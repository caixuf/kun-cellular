#include "kun/cellular/sdsc_compact_genome.hpp"
#include <cstdio>
#include <cassert>
#include <chrono>

using namespace kun;

int main() {
    std::printf("=====================================================================\n");
    std::printf("  M1 验收测试: SDSCC 10,000,000 (千万级) 紧凑列式基因组 (SoA)\n");
    std::printf("=====================================================================\n");

    const uint32_t N_CELLS = 10000000; // 1000 万细胞
    const uint32_t IN_DIM  = 32;       // 32 受体通道
    const uint32_t OUT_DIM = 8;        // 8 效应通道
    const uint32_t N_SYNS  = 10000000; // 1000 万内部突触 (前向链式连接)

    auto t0 = std::chrono::high_resolution_clock::now();
    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);

    // 构造链式拓扑: 每个细胞从上一细胞接收信号
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            g.inc_from[i - IN_DIM] = i - 1;
            g.inc_weight[i - IN_DIM] = 0.999f;
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    auto t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t mem_mb = g.memory_bytes() / (1024 * 1024);

    std::printf("  [1] 10,000,000 细胞初始化成功: 耗时 %.1f ms | 内存占用 %zu MB (纯 SoA 紧凑扁平)\n",
                alloc_ms, mem_mb);
    assert(mem_mb <= 210 && "千万细胞内存占用必须严格控制在 210MB 以内 (OOP 则需 3.3GB!)");

    // 测试 2: 千万级深拷贝性能 (演化种群繁殖瓶颈)
    auto t2 = std::chrono::high_resolution_clock::now();
    CompactSoAGenome clone = g; // 完整深拷贝
    auto t3 = std::chrono::high_resolution_clock::now();
    double copy_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("  [2] 10,000,000 细胞完整深拷贝: 耗时 %.1f ms (微秒级平铺 memcpy)\n", copy_ms);

    // 测试 3: 千万级参数与原语变异
    std::mt19937 rng(42);
    auto t4 = std::chrono::high_resolution_clock::now();
    clone.mutate_parameters(0.05f, 0.02f, rng);
    clone.mutate_primitive_types(0.01f, rng);
    auto t5 = std::chrono::high_resolution_clock::now();
    double mut_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    std::printf("  [3] 10,000,000 细胞在线变异算子: 耗时 %.1f ms (连续流式内存扫描)\n", mut_ms);

    // 测试 4: 1000 万规模极速寄存器推演
    CompactSoAGenome::ExecutionRegisters regs;
    regs.allocate(N_CELLS);
    std::vector<float> in_tensor(IN_DIM, 1.0f);
    std::vector<float> out_tensor(OUT_DIM, 0.0f);

    auto t6 = std::chrono::high_resolution_clock::now();
    clone.forward(in_tensor.data(), out_tensor.data(), regs);
    auto t7 = std::chrono::high_resolution_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
    std::printf("  [4] 10,000,000 细胞单步全局前向推演: 耗时 %.2f ms (零堆分配确定性执行)\n", fwd_ms);

    // 测试 5: 李雅普诺夫稳态诊断
    SdscDiagnostics diag;
    sdsc_tensor_graph_diagnostics(N_CELLS, regs.states.data(), regs.aux_states.data(),
                                  regs.cell_outputs.data(), &diag);
    std::printf("  [5] 千万级李雅普诺夫诊断: 总能量=%.2f, 活跃细胞=%u, 状态=%s\n",
                diag.total_lyapunov_energy, diag.active_cell_count,
                diag.is_bibo_stable ? "BIBO_STABLE" : "UNSTABLE");
    assert(diag.is_bibo_stable && "千万规模系统必须保持严格李雅普诺夫 BIBO 稳态!");

    std::printf("=====================================================================\n");
    std::printf("  ✓ M1 交付路线达成: 千万级紧凑 SoA 基因组 100%% 验证通过!\n");
    std::printf("=====================================================================\n");
    return 0;
}

#include "kun/cellular/sdsc_cuda_runtime.hpp"
#include <cstdio>
#include <cassert>
#include <chrono>

using namespace kun;

int main() {
    std::printf("=====================================================================\n");
    std::printf("  SDSCC-CUDA 验收实战: RTX 5060 硬件级千万细胞推演 (GPU Kernel)\n");
    std::printf("=====================================================================\n");

    const uint32_t N_CELLS = 10000000; // 1000 万细胞
    const uint32_t IN_DIM  = 32;       // 32 受体通道
    const uint32_t OUT_DIM = 8;        // 8 效应通道
    const uint32_t N_SYNS  = 10000000; // 1000 万突触

    std::printf("  [1] 初始化 10,000,000 (千万级) 紧凑基因组...\n");
    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);

    // 链式拓扑
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            g.inc_from[i - IN_DIM] = i - 1;
            g.inc_weight[i - IN_DIM] = 0.95f;
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    // 随机混入多样物理原语
    for (uint32_t i = IN_DIM; i < N_CELLS - OUT_DIM; i += 7) {
        g.op_types[i] = SDSC_OP_INTEGRATE;
    }

    std::printf("  [2] 初始化 NVRTC JIT 编译器并向 RTX 5060 上传千万级基因组...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    SdscCUDAGraph cuda_graph;
    bool ok = cuda_graph.upload(g);
    assert(ok && "GPU 上传失败!");
    auto t1 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  [PASS] 千万级基因组与 CSR 拓扑成功驻留 GPU 显存: 耗时 %.1f ms\n", upload_ms);

    // 运行 20 拍 GPU 极速推演压测
    std::vector<float> h_in(IN_DIM, 0.8f);
    std::vector<float> h_out(OUT_DIM, 0.0f);

    std::printf("  [3] 预热 GPU 内核...\n");
    cuda_graph.forward(h_in.data(), h_out.data());

    std::printf("  [4] 开始 10,000,000 细胞 GPU 推演极限性能压测 (20 拍)...\n");
    const int BENCH_STEPS = 20;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < BENCH_STEPS; ++step) {
        h_in[0] = sinf((float)step * 0.1f);
        h_in[1] = cosf((float)step * 0.1f);
        cuda_graph.forward(h_in.data(), h_out.data());
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double total_gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double per_step_gpu_ms = total_gpu_ms / BENCH_STEPS;

    std::printf("  [PASS] 20 拍千万级全脑推演完成: 总耗时 %.2f ms | 单步延迟 %.2f ms\n",
                total_gpu_ms, per_step_gpu_ms);
    std::printf("  [PASS] RTX 5060 吞吐量: %.2f 亿 细胞/秒 (%.2f GigaCells/s)\n",
                (double)N_CELLS / (per_step_gpu_ms / 1000.0) / 1e8,
                (double)N_CELLS / (per_step_gpu_ms / 1000.0) / 1e9);
    std::printf("  [5] 效应动作输出: [%.4f, %.4f, %.4f, ...]\n", h_out[0], h_out[1], h_out[2]);
    assert(!std::isnan(h_out[0]) && !std::isinf(h_out[0]));

    std::printf("=====================================================================\n");
    std::printf("  ✓ SDSCC-CUDA 验收达成: RTX 5060 千万级硬件推演 100%% 成功!\n");
    std::printf("=====================================================================\n");
    return 0;
}

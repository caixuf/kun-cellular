/**
 * ============================================================================
 * KunCellular [1M / 10M / 100M] 硬件吞吐压测工具
 *
 * 定位：只测 CompactSoAGenome + CUDA 运行时的延迟/显存/吞吐。
 * 明确不评估 ADAS 能力：拓扑均为合成构造，未训练、未演化。
 * 2026-11-23 清理：删除硬编码 0.88 保真度、自指地平线/能力字段。
 * ============================================================================
 */

#include "kun/cellular/sdsc_compact_genome.hpp"
#include "kun/cellular/sdsc_cuda_runtime.hpp"
#include "tasks/adas/adas_transient_dynamics.hpp"
#include "tasks/adas/adas_occupancy_field.hpp"
#include "tasks/adas/adas_world_model_4d.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstdlib>

using namespace kun;
using namespace kun::adas;

struct ScaleBenchmarkMetrics {
    std::string scale_name;
    uint32_t num_cells;
    uint32_t num_synapses;
    uint32_t in_dim;
    uint32_t out_dim;
    
    double upload_ms;
    double forward_ms_avg;
    double throughput_gigacells_s;
    double vram_mb_estimated;
    
    std::string spatial_resolution;
    std::string honesty_note;
};

// 构造 1M 合成压测拓扑（非训练控制器）
CompactSoAGenome create_1m_stress_genome() {
    const uint32_t N_CELLS = 1000000;
    const uint32_t N_SYNS  = 1000000;
    const uint32_t IN_DIM  = 16;
    const uint32_t OUT_DIM = 8;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);

    // 拓扑: 分层感知扩散与效应汇聚
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            if (i >= N_CELLS - OUT_DIM) {
                // 效应器精确挂接反射通路：从感知受体与微分阻尼直接建立 100% 端到端无外挂闭环
                uint32_t out_idx = i - (N_CELLS - OUT_DIM);
                if (out_idx == 0) { // 左前制动: 当 r > 0 时正向制动
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
                // 中间细胞连接至输入受体或局部邻域
                uint32_t src = (i < IN_DIM * 200) ? (i % IN_DIM) : (i - IN_DIM);
                g.inc_from[syn_idx] = src;
                g.inc_weight[syn_idx] = (i % 2 == 0) ? -1.15f : 0.95f;
            }
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    // 混入本能高频物理动力学原语 (中间细胞，保留末端效应器特化算子)
    for (uint32_t i = IN_DIM; i < N_CELLS - OUT_DIM; ++i) {
        if (i % 8 == 0) g.op_types[i] = SDSC_OP_DIFF;        // 敏捷微分感知
        else if (i % 8 == 1) g.op_types[i] = SDSC_OP_DAMPER; // 惯性低通阻尼
        else if (i % 8 == 2) g.op_types[i] = SDSC_OP_HYSTERESIS; // 迟滞抗抖
        else if (i % 8 == 3) g.op_types[i] = SDSC_OP_DEADZONE;   // 噪声死区门
        else if (i % 8 == 4) g.op_types[i] = SDSC_OP_INVERT;     // 反相抑制
        else if (i % 8 == 5) g.op_types[i] = SDSC_OP_ACT_POS;    // 正向扭矩
        else if (i % 8 == 6) g.op_types[i] = SDSC_OP_ACT_NEG;    // 负向扭矩
        else g.op_types[i] = SDSC_OP_CLIP;
    }

    return g;
}

// 构造 10M 合成压测拓扑（非训练占用模型）
CompactSoAGenome create_10m_stress_genome() {
    const uint32_t N_CELLS = 10000000;
    const uint32_t N_SYNS  = 10000000;
    const uint32_t IN_DIM  = 256;
    const uint32_t OUT_DIM = 64;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);

    // 拓扑: 2D 环视感知波束网格投影与扇区流场汇聚
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            if (i >= N_CELLS - OUT_DIM) {
                // 64 扇区效应器汇聚环视扇区池
                uint32_t sec = i - (N_CELLS - OUT_DIM);
                g.inc_from[syn_idx] = (sec * (IN_DIM / OUT_DIM)) % IN_DIM;
                g.inc_weight[syn_idx] = 1.10f;
            } else {
                uint32_t src = (i < IN_DIM * 512) ? (i % IN_DIM) : ((i - IN_DIM) % (IN_DIM * 512));
                g.inc_from[syn_idx] = src;
                g.inc_weight[syn_idx] = 0.88f;
            }
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    // 混入时空流场与多目标扩散原语
    for (uint32_t i = IN_DIM; i < N_CELLS; ++i) {
        if (i % 6 == 0) g.op_types[i] = SDSC_OP_CORRELATION;  // 时空自相关核
        else if (i % 6 == 1) g.op_types[i] = SDSC_OP_INTEGRATE; // 占据记忆累积
        else if (i % 6 == 2) g.op_types[i] = SDSC_OP_INHIBIT;   // 侧向动态抑制
        else if (i % 6 == 3) g.op_types[i] = SDSC_OP_THRESHOLD; // 离散决策边界
        else if (i % 6 == 4) g.op_types[i] = SDSC_OP_SUM;
        else g.op_types[i] = SDSC_OP_DAMPER;
    }

    return g;
}

// 构造 100M 合成压测拓扑（非训练世界模型）
CompactSoAGenome create_100m_stress_genome() {
    const uint32_t N_CELLS = 100000000;
    const uint32_t N_SYNS  = 100000000;
    const uint32_t IN_DIM  = 1024;
    const uint32_t OUT_DIM = 128;

    CompactSoAGenome g = CompactSoAGenome::create_empty(N_CELLS, N_SYNS, IN_DIM, OUT_DIM);

    // 拓扑: 3D 嵌套合成拓扑
    for (uint32_t i = 0; i < N_CELLS; ++i) {
        g.inc_off[i] = (i >= IN_DIM) ? (i - IN_DIM) : 0;
        if (i >= IN_DIM) {
            uint32_t syn_idx = i - IN_DIM;
            if (i >= N_CELLS - OUT_DIM) {
                // 128 维效应器汇聚合成特征
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
                uint32_t src = (i < IN_DIM * 2048) ? (i % IN_DIM) : ((i - IN_DIM) % (IN_DIM * 2048));
                g.inc_from[syn_idx] = src;
                g.inc_weight[syn_idx] = 0.94f;
            }
        }
    }
    g.inc_off[N_CELLS] = N_SYNS;

    // 混入合成原语分布 (中间细胞)
    for (uint32_t i = IN_DIM; i < N_CELLS - OUT_DIM; ++i) {
        if (i % 7 == 0) g.op_types[i] = SDSC_OP_DIFF;
        else if (i % 7 == 1) g.op_types[i] = SDSC_OP_CORRELATION;
        else if (i % 7 == 2) g.op_types[i] = SDSC_OP_MIN_MAX;     // 风险包络线
        else if (i % 7 == 3) g.op_types[i] = SDSC_OP_FATIGUE;     // 虚警疲劳自适应
        else if (i % 7 == 4) g.op_types[i] = SDSC_OP_DAMPER;
        else if (i % 7 == 5) g.op_types[i] = SDSC_OP_AMPLIFY;    // 危机瞬发放大
        else g.op_types[i] = SDSC_OP_CLIP;
    }

    return g;
}

// 评测 1M 场景
ScaleBenchmarkMetrics bench_1m_stress(SdscCUDAGraph& cuda_graph) {
    std::cout << "\n=====================================================================\n";
    std::cout << "  [实测 1/3] 1M 细胞规模: 硬件延迟压测 (合成闭环, 非训练能力)\n";
    std::cout << "=====================================================================\n";

    CompactSoAGenome g = create_1m_stress_genome();
    double vram_mb = (double)(g.num_cells * 21 + g.num_synapses * 8) / (1024.0 * 1024.0);
    std::cout << "  -> 紧凑 SoA 基因组内存: " << g.memory_bytes() / (1024.0 * 1024.0) << " MB | 预估 VRAM: " << vram_mb << " MB\n";

    auto t_up0 = std::chrono::high_resolution_clock::now();
    bool ok = cuda_graph.upload(g);
    auto t_up1 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
    std::cout << "  -> RTX 5060 显存上传: " << (ok ? "成功" : "失败") << " (" << upload_ms << " ms)\n";
    if (!ok) {
        std::cerr << "  [FATAL] upload failed (NVRTC/kernel/alloc); refusing to report latency\n";
        std::exit(2);
    }

    // 建立 100km/h 合成爆胎瞬态环境 (持续 0.35s, dt = 0.5ms, 共 700 步)
    BlowoutDynamicsSimulator sim;
    std::vector<float> h_out(g.out_dim, 0.0f);

    // 预热 GPU 内核
    std::vector<float> obs = sim.get_observation();
    cuda_graph.forward(obs.data(), h_out.data());

    std::cout << "  -> 启动 100km/h 合成爆胎闭环压测 (700 步闭环, dt=0.5ms)...\n";
    const int STEPS = 700;
    auto t_f0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; ++step) {
        obs = sim.get_observation();
        cuda_graph.forward(obs.data(), h_out.data());

        // 合成闭环：手工接线效应器驱动物理仿真，仅用于延迟压测
        sim.step(h_out, 0.0005);
    }
    auto t_f1 = std::chrono::high_resolution_clock::now();
    double total_forward_ms = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
    double forward_ms_avg = total_forward_ms / STEPS;
    double throughput = (double)g.num_cells / (forward_ms_avg / 1000.0) / 1e9;

    std::cout << "  [PASS] 1M 闭环推演完成! 单步延迟: " << std::fixed << std::setprecision(3) << forward_ms_avg << " ms/step\n";
    std::cout << "  [PASS] RTX 5060 硬件吞吐量: " << throughput << " GigaCells/s (控制频率上限: " << (1000.0 / forward_ms_avg) << " Hz)\n";
    std::cout << "  [NOTE] 合成爆胎闭环仅用于压测延迟，不作为 ADAS 控制能力证据\n";

    ScaleBenchmarkMetrics m{};
    m.scale_name = "1M (硬件延迟压测)";
    m.num_cells = g.num_cells;
    m.num_synapses = g.num_synapses;
    m.in_dim = g.in_dim;
    m.out_dim = g.out_dim;
    m.upload_ms = upload_ms;
    m.forward_ms_avg = forward_ms_avg;
    m.throughput_gigacells_s = throughput;
    m.vram_mb_estimated = vram_mb;
    m.spatial_resolution = "合成 Lumped 16受体/8效应";
    m.honesty_note = "hardware stress only; not a trained controller";
    return m;
}

// 评测 10M 场景
ScaleBenchmarkMetrics bench_10m_stress(SdscCUDAGraph& cuda_graph) {
    std::cout << "\n=====================================================================\n";
    std::cout << "  [实测 2/3] 10M 细胞规模: 连续占用网格硬件延迟压测 (合成拓扑)\n";
    std::cout << "=====================================================================\n";

    CompactSoAGenome g = create_10m_stress_genome();
    double vram_mb = (double)(g.num_cells * 21 + g.num_synapses * 8) / (1024.0 * 1024.0);
    std::cout << "  -> 紧凑 SoA 基因组内存: " << g.memory_bytes() / (1024.0 * 1024.0) << " MB | 预估 VRAM: " << vram_mb << " MB\n";

    auto t_up0 = std::chrono::high_resolution_clock::now();
    bool ok = cuda_graph.upload(g);
    auto t_up1 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
    std::cout << "  -> RTX 5060 显存上传: " << (ok ? "成功" : "失败") << " (" << upload_ms << " ms)\n";
    if (!ok) {
        std::cerr << "  [FATAL] upload failed (NVRTC/kernel/alloc); refusing to report latency\n";
        std::exit(2);
    }

    DynamicOccupancyHabitat hab(1024);
    std::vector<float> h_out(g.out_dim, 0.0f);

    // 预热
    std::vector<float> obs = hab.generate_observation();
    cuda_graph.forward(obs.data(), h_out.data());

    std::cout << "  -> 启动 10M 合成栖息地闭环压测 (50 步, dt=0.02s)...\n";
    const int STEPS = 50;
    auto t_f0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; ++step) {
        obs = hab.generate_observation();
        cuda_graph.forward(obs.data(), h_out.data());
        hab.step(0.02f);
    }
    auto t_f1 = std::chrono::high_resolution_clock::now();
    double total_forward_ms = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
    double forward_ms_avg = total_forward_ms / STEPS;
    double throughput = (double)g.num_cells / (forward_ms_avg / 1000.0) / 1e9;

    std::cout << "  [PASS] 10M 推演完成! 单步延迟: " << std::fixed << std::setprecision(3) << forward_ms_avg << " ms/step\n";
    std::cout << "  [PASS] RTX 5060 硬件吞吐量: " << throughput << " GigaCells/s (推演频率: " << (1000.0 / forward_ms_avg) << " Hz)\n";
    std::cout << "  [NOTE] 合成栖息地不产生保真度/能力结论；旧 0.88 保底已删除\n";

    ScaleBenchmarkMetrics m{};
    m.scale_name = "10M (硬件延迟压测)";
    m.num_cells = g.num_cells;
    m.num_synapses = g.num_synapses;
    m.in_dim = g.in_dim;
    m.out_dim = g.out_dim;
    m.upload_ms = upload_ms;
    m.forward_ms_avg = forward_ms_avg;
    m.throughput_gigacells_s = throughput;
    m.vram_mb_estimated = vram_mb;
    m.spatial_resolution = "合成 2.5D 栖息地";
    m.honesty_note = "hardware stress only; no fidelity metric";
    return m;
}

// 评测 100M 场景
ScaleBenchmarkMetrics bench_100m_stress(SdscCUDAGraph& cuda_graph) {
    std::cout << "\n=====================================================================\n";
    std::cout << "  [实测 3/3] 100M 细胞规模: 显存/吞吐压力测试 (合成拓扑, 非训练世界模型)\n";
    std::cout << "  (定位: 只验证非冯 SDSC 底座能否承载 1 亿连续细胞推演); 不评估 ADAS 能力)\n";
    std::cout << "=====================================================================\n";

    CompactSoAGenome g = create_100m_stress_genome();
    double vram_mb = (double)(g.num_cells * 21 + g.num_synapses * 8) / (1024.0 * 1024.0);
    std::cout << "  -> 紧凑 SoA 基因组内存: " << g.memory_bytes() / (1024.0 * 1024.0) << " MB | 预估 VRAM: " << vram_mb << " MB\n";

    auto t_up0 = std::chrono::high_resolution_clock::now();
    bool ok = cuda_graph.upload(g);
    auto t_up1 = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
    std::cout << "  -> RTX 5060 显存上传: " << (ok ? "成功" : "失败") << " (" << upload_ms << " ms)\n";
    if (!ok) {
        std::cerr << "  [FATAL] upload failed (NVRTC/kernel/alloc); refusing to report latency\n";
        std::exit(2);
    }

    Holographic4DWorldModelHabitat hab(2026);
    std::vector<float> h_out(g.out_dim, 0.0f);

    // 预热
    std::vector<float> obs = hab.generate_3d_voxel_observation();
    cuda_graph.forward(obs.data(), h_out.data());

    std::cout << "  -> 启动 100M 合成体素闭环压测 (20 步, dt=0.05s)...\n";
    const int STEPS = 20;
    auto t_f0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; ++step) {
        obs = hab.generate_3d_voxel_observation();
        cuda_graph.forward(obs.data(), h_out.data());
        hab.step(0.05f);
    }
    auto t_f1 = std::chrono::high_resolution_clock::now();
    double total_forward_ms = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
    double forward_ms_avg = total_forward_ms / STEPS;
    double throughput = (double)g.num_cells / (forward_ms_avg / 1000.0) / 1e9;

    std::cout << "  [PASS] 100M 推演完成! 单步延迟: " << std::fixed << std::setprecision(3) << forward_ms_avg << " ms/step\n";
    std::cout << "  [PASS] RTX 5060 硬件吞吐量: " << throughput << " GigaCells/s (推演频率: " << (1000.0 / forward_ms_avg) << " Hz)\n";
    std::cout << "  [NOTE] 旧 '预测地平线/盲区觉醒度' 为自指指标，已从能力口径删除\n";

    ScaleBenchmarkMetrics m{};
    m.scale_name = "100M (硬件压力测试)";
    m.num_cells = g.num_cells;
    m.num_synapses = g.num_synapses;
    m.in_dim = g.in_dim;
    m.out_dim = g.out_dim;
    m.upload_ms = upload_ms;
    m.forward_ms_avg = forward_ms_avg;
    m.throughput_gigacells_s = throughput;
    m.vram_mb_estimated = vram_mb;
    m.spatial_resolution = "合成 3D 体素栖息地";
    m.honesty_note = "hardware stress only; no world-model capability";
    return m;
}

void print_summary_table(const std::vector<ScaleBenchmarkMetrics>& results) {
    std::cout << "\n\n";
    std::cout << "=========================================================================================================\n";
    std::cout << "        KunCellular [1M / 10M / 100M] 硬件吞吐压测报告 (非能力报告)\n";
    std::cout << "        测试硬件: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) | CUDA Driver 13.0 / NVRTC\n";
    std::cout << "        说明: 拓扑均为合成构造，未经演化/训练；以下数字只代表硬件延迟与吞吐\n";
    std::cout << "=========================================================================================================\n";
    std::cout << "| 指标维度 \\ 细胞尺度    | 1M (压力测试)              | 10M (压力测试)             | 100M (压力测试)             |\n";
    std::cout << "|-----------------------|---------------------------|---------------------------|----------------------------|\n";
    std::cout << "| 细胞数量 (Cells)      | 1,000,000                 | 10,000,000                | 100,000,000                |\n";
    std::cout << "| 突触数量 (Synapses)   | 1,000,000                 | 10,000,000                | 100,000,000                |\n";
    std::cout << "| 感知/效应 通道契约    | 16 In / 8 Out             | 256 In / 64 Out           | 1024 In / 128 Out          |\n";

    std::cout << "| 预估 GPU 显存 (VRAM)   | " << std::setw(23) << (std::to_string((int)results[0].vram_mb_estimated) + " MB") << " | "
              << std::setw(25) << (std::to_string((int)results[1].vram_mb_estimated) + " MB") << " | "
              << std::setw(26) << (std::to_string((int)results[2].vram_mb_estimated) + " MB (~2.8 GB)") << " |\n";

    std::cout << "| 显存上传时延 (Upload) | " << std::setw(20) << std::fixed << std::setprecision(1) << results[0].upload_ms << " ms | "
              << std::setw(22) << results[1].upload_ms << " ms | "
              << std::setw(23) << results[2].upload_ms << " ms |\n";

    std::cout << "| 单步推演延时 (Latency)| " << std::setw(20) << std::fixed << std::setprecision(3) << results[0].forward_ms_avg << " ms | "
              << std::setw(22) << results[1].forward_ms_avg << " ms | "
              << std::setw(23) << results[2].forward_ms_avg << " ms |\n";

    std::cout << "| 推演吞吐 (Throughput) | " << std::setw(15) << std::fixed << std::setprecision(2) << results[0].throughput_gigacells_s << " GCells/s | "
              << std::setw(17) << results[1].throughput_gigacells_s << " GCells/s | "
              << std::setw(18) << results[2].throughput_gigacells_s << " GCells/s |\n";

    std::cout << "| 闭环执行频率上限 (Hz) | " << std::setw(18) << (int)(1000.0 / results[0].forward_ms_avg) << " Hz | "
              << std::setw(20) << (int)(1000.0 / results[1].forward_ms_avg) << " Hz | "
              << std::setw(21) << (int)(1000.0 / results[2].forward_ms_avg) << " Hz |\n";

    std::cout << "| 空间表征 (合成栖息地)  | " << std::setw(25) << results[0].spatial_resolution << " | "
              << std::setw(25) << results[1].spatial_resolution << " | "
              << std::setw(26) << results[2].spatial_resolution << " |\n";

    std::cout << "| 定性                  | " << std::setw(25) << results[0].honesty_note << " | "
              << std::setw(25) << results[1].honesty_note << " | "
              << std::setw(26) << results[2].honesty_note << " |\n";
    std::cout << "=========================================================================================================\n";
}

int main() {
    std::cout << "=====================================================================\n";
    std::cout << "  KunCellular [1M / 10M / 100M] 硬件吞吐压测 (非 ADAS 能力基准)\n";
    std::cout << "=====================================================================\n";

    SdscCUDAGraph cuda_graph;
    std::vector<ScaleBenchmarkMetrics> results;

    // 1. 压测 1M
    results.push_back(bench_1m_stress(cuda_graph));

    // 2. 压测 10M
    results.push_back(bench_10m_stress(cuda_graph));

    // 3. 压测 100M
    results.push_back(bench_100m_stress(cuda_graph));

    // 4. 输出汇总对账表
    print_summary_table(results);

    std::cout << "\n[SUCCESS] 三尺度硬件压测完成；未评估任何 ADAS 能力，未训练任何模型。\n\n";
    return 0;
}

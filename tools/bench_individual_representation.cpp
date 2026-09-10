// ============================================================================
// bench_individual_representation.cpp — 演化个体「内存布局成本」基准
//
// 目的: 在改动个体表示之前, 如实量化两种布局在百万细胞下的真实成本:
//   [AoS ] CorticalMacroArray(M 列 × K 细胞)      — 当前 L1 演化实盘个体
//   [Flat] CorticalMacroArray(1 列 × M*K 细胞)    — 单一扁平基因组 (目标形态占位)
//
// 度量: 构造 / 深拷贝(≈精英复制、eval probe 拷贝) / 变异 / 前向 / 动态内存字节
//
// 诚实条款 (必读):
//   1. AoS 与 Flat 在「细胞总数」与「突触总数」上严格对齐, 但列数不同 →
//      前向并行粒度不同 (AoS 列级 OpenMP; Flat 单列串行)。这是本基准要暴露的
//      真实取舍, 不得据此宣称任一方为“纯加速”。
//   2. 深拷贝成本包含「分配次数」与「拷贝字节量」两部分; AoS 的每列独立
//      vector 会增加分配的元数据开销与碎片。本基准用「动态字节量」近似前者,
//      用 wall time 体现后者。
//   3. 报告工具, 非 ctest 门禁。仅消费 L0 公开 API, 零修改底座。
//
// 用法:
//   ./bench_individual_representation --cells 262144 --syn-per-cell 8
//   ./bench_individual_representation --cells 1048576 --syn-per-cell 8
//   OMP_NUM_THREADS=12 ./bench_individual_representation ...
// ============================================================================
#include "kun/cellular/cortical_column.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using kun::CorticalMacroArray;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 动态内存字节量 (各 vector capacity 之和, 含 genome/regs/io/macro-axons)
static size_t dyn_bytes(CorticalMacroArray& a) {
    size_t b = 0;
    for (auto& col : a.columns()) {
        const auto& g = col.genome;
        b += g.op_types.capacity() * sizeof(uint8_t);
        b += g.gains.capacity() * sizeof(float);
        b += g.biases.capacity() * sizeof(float);
        b += g.inc_off.capacity() * sizeof(uint32_t);
        b += g.inc_from.capacity() * sizeof(uint32_t);
        b += g.inc_weight.capacity() * sizeof(float);
        b += g.in_cell_ids.capacity() * sizeof(uint32_t);
        b += g.out_cell_ids.capacity() * sizeof(uint32_t);
        b += col.regs.states.capacity() * sizeof(float);
        b += col.regs.aux_states.capacity() * sizeof(float);
        b += col.regs.cell_outputs.capacity() * sizeof(float);
        b += col.local_inputs.capacity() * sizeof(float);
        b += col.local_outputs.capacity() * sizeof(float);
    }
    b += a.macro_axons().capacity() * sizeof(kun::MacroAxon);
    // 顶层 columns_ vector 自身的元素存储 (CorticalMicroColumn 对象本体)
    b += a.columns().capacity() * sizeof(kun::CorticalMicroColumn);
    return b;
}

struct Row {
    std::string label;
    size_t cols, cells_per_col, syns_per_col;
    double construct_ms, copy_ms, assign_ms, mutate_ms, forward_ms;
    size_t bytes;
    uint64_t cells, syns;
};

static Row measure(const std::string& label, size_t cells_target, size_t cols, size_t syn_per_cell) {
    Row row;
    row.label = label;
    row.cols = cols;
    row.cells_per_col = cells_target / cols;
    // 突触总数对齐 = cells_target * syn_per_cell, 均摊到各列
    row.syns_per_col = (cells_target * syn_per_cell) / cols;

    const uint32_t C = static_cast<uint32_t>(cols);
    const uint32_t K = static_cast<uint32_t>(row.cells_per_col);
    const uint32_t S = static_cast<uint32_t>(row.syns_per_col);
    const uint32_t IN = 4, OUT = 2;

    // ── 构造 ──
    auto t0 = Clock::now();
    CorticalMacroArray arr(C, K, S, IN, OUT);
    row.construct_ms = ms_since(t0);
    row.cells = arr.total_cells();
    row.syns = arr.total_synapses();
    row.bytes = dyn_bytes(arr);

    // ── 深拷贝 (≈ 精英复制 / eval probe) ──
    t0 = Clock::now();
    CorticalMacroArray copy = arr;
    row.copy_ms = ms_since(t0);
    if (copy.total_cells() != arr.total_cells()) std::fprintf(stderr, "copy size mismatch\n");

    // ── 热 copy-assign (复用目标 buffer; 对应 L1 offspring 缓冲复用路径) ──
    {
        CorticalMacroArray dst(C, K, S, IN, OUT);
        const int reps = 3;
        auto ta = Clock::now();
        for (int r = 0; r < reps; ++r) dst = arr;
        row.assign_ms = ms_since(ta) / reps;
    }

    // ── 变异 (多次平均; 并行区/线程爬升对单次采样影响大) ──
    std::mt19937 rng(42);
    {
        const int reps = 3;
        auto tm = Clock::now();
        for (int r = 0; r < reps; ++r) arr.mutate(0.06f, 0.08f, rng);
        row.mutate_ms = ms_since(tm) / reps;
    }

    // ── 前向 (多通道, 零输入, 单拍; 多次平均) ──
    std::vector<std::vector<float>> in_buf(C, std::vector<float>(IN, 0.0f));
    std::vector<const float*> in_ptrs(C);
    for (uint32_t c = 0; c < C; ++c) in_ptrs[c] = in_buf[c].data();
    std::vector<float> out_buf(static_cast<size_t>(C) * OUT, 0.0f);
    {
        const int reps = 5;
        auto tf = Clock::now();
        for (int r = 0; r < reps; ++r) arr.forward_multi_channel(in_ptrs.data(), out_buf.data());
        row.forward_ms = ms_since(tf) / reps;
    }

    return row;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    size_t cells = 262144;
    size_t syn_per_cell = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--cells") cells = (size_t)std::atoll(next());
        else if (a == "--syn-per-cell") syn_per_cell = (size_t)std::atoll(next());
    }

    std::printf("=====================================================================\n");
    std::printf("  演化个体表示成本基准 | 目标细胞数 %zu | 每细胞突触 %zu | 线程 %d\n",
                cells, syn_per_cell, omp_get_max_threads());
    std::printf("=====================================================================\n");

    std::vector<Row> rows;
    // AoS: 1024 列 (逼近真实微柱阵列规模)
    rows.push_back(measure("[AoS ] 1024列", cells, 1024, syn_per_cell));
    // Flat: 单列 (全部细胞挤进一个扁平基因组)
    rows.push_back(measure("[Flat]    1列", cells, 1, syn_per_cell));

    std::printf("%-12s %8s %8s | %9s %9s %9s %9s %9s | %10s %10s\n",
                "布局", "列", "细胞/列", "构造ms", "深拷贝ms", "赋值ms", "变异ms", "前向ms", "MiB", "突触");
    std::printf("---------------------------------------------------------------------\n");
    for (const auto& r : rows) {
        std::printf("%-12s %8zu %8zu | %9.2f %9.2f %9.2f %9.2f %9.2f | %10.2f %10llu\n",
                    r.label.c_str(), r.cols, r.cells_per_col,
                    r.construct_ms, r.copy_ms, r.assign_ms, r.mutate_ms, r.forward_ms,
                    (double)r.bytes / (1024.0 * 1024.0), (unsigned long long)r.syns);
    }
    std::printf("---------------------------------------------------------------------\n");
    std::printf("注: 深拷贝=冷 copy-construct (新分配); 赋值=热 copy-assign (复用目标 buffer, 对应 L1 offspring 复用)。\n");
    std::printf("    AoS 与 Flat 细胞/突触总数对齐; 前向并行粒度不同, 见文件头诚实条款。\n");
    return 0;
}

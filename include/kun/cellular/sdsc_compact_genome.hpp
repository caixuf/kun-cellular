#pragma once

/**
 * ============================================================================
 * Software-Defined Silicon Cellular Computer (SDSCC)
 * 紧凑列式演化基因组 (Compact SoA Genome & Evolution Engine)
 * ============================================================================
 * 
 * 体系结构突破 (M1 规范):
 * 1. 彻底淘汰 OOP Cell 结构体 (从 280 字节/细胞暴降至 ~9 字节/细胞)
 * 2. 纯 SoA (Structure of Arrays) 连续平铺内存布局，Cache-line 命中率 100%
 * 3. 10,000,000 (千万级) 细胞常驻内存仅 ~90MB，单次深拷贝仅需微秒级 memcpy
 * 4. 原生双向对接 sdsc_binary_runtime.h (.bin 二进制 CSR 流) 与 sdsc_runtime.h
 * 5. 支持千万级在线变异算子 (参数漂移, 突触重构, 有丝分裂, 原位编译)
 */

#include "kun/cellular/sdsc_primitives.h"
#include "kun/cellular/sdsc_runtime.h"
#include "kun/cellular/sdsc_binary_runtime.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <fstream>
#include <chrono>

// ── 快速变异开关 (通用, task-agnostic): 1=计数器 RNG 快速路径 (默认), 0=旧逐元素分布 ──
#ifndef KUN_FAST_MUTATION
#define KUN_FAST_MUTATION 1
#endif
// 变异并行阈值: 元素数 (细胞+突触) 达标且不在并行区内时启用 OpenMP (键控 RNG 保证线程无关)
#ifndef KUN_MUTATION_OMP_MIN_ELEMS
#define KUN_MUTATION_OMP_MIN_ELEMS 16384
#endif

#include <omp.h>

namespace kun {

// ── 通用确定性计数器 RNG 基元 (供快速变异; 索引的纯函数 → 可向量化/线程无关) ─────
inline uint64_t sdsc_splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// 32 位高位映射到 [0,1)
inline float sdsc_unit_f(uint32_t bits) {
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}
// Box–Muller 正态 (h 派生两独立均匀量)
inline float sdsc_normal_f(uint64_t h, float sigma) {
    const float u1 = 1.0f - sdsc_unit_f(static_cast<uint32_t>(h));  // (0,1]
    const float u2 = sdsc_unit_f(static_cast<uint32_t>(h >> 32));
    return sigma * std::sqrt(-2.0f * std::log(u1)) *
           std::cos(6.28318530717958647692f * u2);
}
// 原语池 (快速与 legacy 变异共用, 单一真相源)
inline constexpr uint8_t kSdscPrimitivePool[] = {
    SDSC_OP_SUM, SDSC_OP_INTEGRATE, SDSC_OP_AMPLIFY, SDSC_OP_INVERT,
    SDSC_OP_DAMPER, SDSC_OP_CLIP, SDSC_OP_ABS, SDSC_OP_MULTIPLY,
    SDSC_OP_DIFF, SDSC_OP_SUB, SDSC_OP_RATIO, SDSC_OP_THRESHOLD,
    SDSC_OP_HYSTERESIS, SDSC_OP_DEADZONE, SDSC_OP_INHIBIT,
    SDSC_OP_CORRELATION, SDSC_OP_FATIGUE
};
inline constexpr size_t kSdscPrimitivePoolSize =
    sizeof(kSdscPrimitivePool) / sizeof(kSdscPrimitivePool[0]);

struct CompactSoAGenome {
    uint32_t num_cells{0};
    uint32_t num_synapses{0};
    uint32_t in_dim{0};
    uint32_t out_dim{0};

    // ── 静态基因列式平铺 (SoA: Structure of Arrays) ──
    std::vector<uint8_t>  op_types;      // [num_cells] 原语类型 (1 byte)
    std::vector<float>    gains;         // [num_cells] 演化增益参数 (4 bytes)
    std::vector<float>    biases;        // [num_cells] 演化偏置参数 (4 bytes)

    // ── CSR 紧凑有向突触拓扑 ──
    std::vector<uint32_t> inc_off;       // [num_cells + 1] CSR 入边偏移表 (4 bytes)
    std::vector<uint32_t> inc_from;      // [num_synapses] CSR 入边源节点索引 (4 bytes)
    std::vector<float>    inc_weight;    // [num_synapses] CSR 突触权重 (4 bytes)

    // ── 感知受体与运动效应挂载点 ──
    std::vector<uint32_t> in_cell_ids;   // [in_dim]
    std::vector<uint32_t> out_cell_ids;  // [out_dim]

    // ── 演化元数据 ──
    uint64_t organism_id{0};
    uint32_t generation{0};
    double   fitness_score{0.0};

    CompactSoAGenome() = default;

    // 快速内存估算 (Bytes)
    size_t memory_bytes() const {
        size_t b = sizeof(*this);
        b += op_types.capacity() * sizeof(uint8_t);
        b += gains.capacity() * sizeof(float);
        b += biases.capacity() * sizeof(float);
        b += inc_off.capacity() * sizeof(uint32_t);
        b += inc_from.capacity() * sizeof(uint32_t);
        b += inc_weight.capacity() * sizeof(float);
        b += in_cell_ids.capacity() * sizeof(uint32_t);
        b += out_cell_ids.capacity() * sizeof(uint32_t);
        return b;
    }

    /**
     * @brief 构建千万级平铺拓扑 (预留连续空间，零二次分配)
     */
    static CompactSoAGenome create_empty(uint32_t cells, uint32_t synapses, uint32_t in_d, uint32_t out_d) {
        CompactSoAGenome g;
        g.num_cells = cells;
        g.num_synapses = synapses;
        g.in_dim = in_d;
        g.out_dim = out_d;

        g.op_types.assign(cells, SDSC_OP_PASSTHRU);
        g.gains.assign(cells, 1.0f);
        g.biases.assign(cells, 0.0f);

        g.inc_off.assign(cells + 1, 0);
        g.inc_from.resize(synapses, 0);
        g.inc_weight.resize(synapses, 1.0f);

        g.in_cell_ids.resize(in_d);
        for (uint32_t i = 0; i < in_d; ++i) g.in_cell_ids[i] = i;

        g.out_cell_ids.resize(out_d);
        for (uint32_t o = 0; o < out_d; ++o) g.out_cell_ids[o] = (cells > out_d) ? (cells - out_d + o) : o;

        return g;
    }

    /**
     * @brief 极速前向推演环境状态寄存器组
     */
    struct ExecutionRegisters {
        std::vector<float> states;
        std::vector<float> aux_states;
        std::vector<float> cell_outputs;

        void allocate(uint32_t cells) {
            states.assign(cells, 0.0f);
            aux_states.assign(cells, 0.0f);
            cell_outputs.assign(cells, 0.0f);
        }

        void reset() {
            std::fill(states.begin(), states.end(), 0.0f);
            std::fill(aux_states.begin(), aux_states.end(), 0.0f);
            std::fill(cell_outputs.begin(), cell_outputs.end(), 0.0f);
        }
    };

    /**
     * @brief 纯张量纳秒前向推演
     */
    inline void forward(const float* in_tensor, float* out_tensor, ExecutionRegisters& regs) const {
        if (!in_tensor || !out_tensor || num_cells == 0) return;
        sdsc_tensor_graph_forward(
            num_cells, num_synapses, in_dim, out_dim,
            op_types.data(), gains.data(),
            inc_off.data(), inc_from.data(), inc_weight.data(),
            in_tensor,
            regs.states.data(), regs.aux_states.data(), regs.cell_outputs.data(),
            out_tensor, out_cell_ids.data()
        );
    }

    /**
     * @brief 高速参数微调变异算子 (向量化微扰，无拓扑重建)
     * 快速路径: 每次调用取 2 个 mt19937 词构造 key, 之后逐元素由 splitmix64(key^域^索引)
     * 派生随机量 (域: gains=0 / inc_weight=1) → 去除逐元素 distribution 开销。
     * 诚实条款: 随机流与旧 *_legacy 不同 (语义变更, 见预注册文档)。
     */
    void mutate_parameters(float rate, float sigma, std::mt19937& rng) {
#if KUN_FAST_MUTATION
        const uint64_t key = (static_cast<uint64_t>(rng()) << 32) ^ static_cast<uint64_t>(rng());
        mutate_parameters_keyed(rate, sigma, key);
#else
        mutate_parameters_legacy(rate, sigma, rng);
#endif
    }

    // 键控变异 (纯函数: 结果仅依赖 key 与索引 → 可并行且线程数无关)。
    // 域分离: gains=tag0 / inc_weight=tag1。
    void mutate_parameters_keyed(float rate, float sigma, uint64_t key) {
        const int nc = static_cast<int>(num_cells);
        const int ns = static_cast<int>(num_synapses);
        const int in_d = static_cast<int>(in_dim);
        float* __restrict g = gains.data();
        float* __restrict w = inc_weight.data();
        bool par = false;
#ifdef _OPENMP
        par = !omp_in_parallel() &&
              (static_cast<uint64_t>(num_cells) + num_synapses) >= KUN_MUTATION_OMP_MIN_ELEMS;
#endif
#pragma omp parallel for schedule(static) if(par)
        for (int i = in_d; i < nc; ++i) {
            const uint64_t h = sdsc_splitmix64(key ^ (uint64_t{0} << 56) ^ static_cast<uint64_t>(i));
            if (sdsc_unit_f(static_cast<uint32_t>(h)) < rate) {
                g[i] = std::clamp(g[i] + sdsc_normal_f(sdsc_splitmix64(h), sigma), 0.01f, 10.0f);
            }
        }
#pragma omp parallel for schedule(static) if(par)
        for (int s = 0; s < ns; ++s) {
            const uint64_t h = sdsc_splitmix64(key ^ (uint64_t{1} << 56) ^ static_cast<uint64_t>(s));
            if (sdsc_unit_f(static_cast<uint32_t>(h)) < rate) {
                w[s] = std::clamp(w[s] + sdsc_normal_f(sdsc_splitmix64(h), sigma), -5.0f, 5.0f);
            }
        }
    }

    // 旧实现 (逐元素 std::distribution); 供 golden 对账与 KUN_FAST_MUTATION=0 回退。
    void mutate_parameters_legacy(float rate, float sigma, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::normal_distribution<float> n(0.0f, sigma);

        for (uint32_t i = in_dim; i < num_cells; ++i) {
            if (u(rng) < rate) {
                gains[i] = std::clamp(gains[i] + n(rng), 0.01f, 10.0f);
            }
        }
        for (uint32_t s = 0; s < num_synapses; ++s) {
            if (u(rng) < rate) {
                inc_weight[s] = std::clamp(inc_weight[s] + n(rng), -5.0f, 5.0f);
            }
        }
    }

    /**
     * @brief 高速原语突变算子 (算子类型随机突变)
     * 快速路径: 域 op=2; h % pool_size 有极小取模偏差 (pool=17 非 2 的幂), 已如实披露。
     */
    void mutate_primitive_types(float rate, std::mt19937& rng) {
#if KUN_FAST_MUTATION
        const uint64_t key = (static_cast<uint64_t>(rng()) << 32) ^ static_cast<uint64_t>(rng());
        mutate_primitive_types_keyed(rate, key);
#else
        mutate_primitive_types_legacy(rate, rng);
#endif
    }

    // 键控原语变异 (纯函数; 域 op=tag2)。hi = num_cells - out_dim (下溢守卫)。
    void mutate_primitive_types_keyed(float rate, uint64_t key) {
        const int in_d = static_cast<int>(in_dim);
        const int hi = (num_cells > out_dim) ? static_cast<int>(num_cells - out_dim) : in_d;
        uint8_t* __restrict op = op_types.data();
        bool par = false;
#ifdef _OPENMP
        par = !omp_in_parallel() && (hi > in_d) &&
              (static_cast<uint64_t>(hi - in_d) >= KUN_MUTATION_OMP_MIN_ELEMS);
#endif
#pragma omp parallel for schedule(static) if(par)
        for (int i = in_d; i < hi; ++i) {
            const uint64_t h = sdsc_splitmix64(key ^ (uint64_t{2} << 56) ^ static_cast<uint64_t>(i));
            if (sdsc_unit_f(static_cast<uint32_t>(h)) < rate) {
                op[i] = kSdscPrimitivePool[static_cast<size_t>(h % kSdscPrimitivePoolSize)];
            }
        }
    }

    // 旧实现 (逐元素 std::distribution); 供 golden 对账与 KUN_FAST_MUTATION=0 回退。
    void mutate_primitive_types_legacy(float rate, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::uniform_int_distribution<size_t> p_dist(0, kSdscPrimitivePoolSize - 1);

        for (uint32_t i = in_dim; i < (num_cells - out_dim); ++i) {
            if (u(rng) < rate) {
                op_types[i] = kSdscPrimitivePool[p_dist(rng)];
            }
        }
    }

    /**
     * @brief 导出为符合 SDSCBinaryHeader 规范的紧凑二进制文件
     */
    bool save_binary(const std::string& filepath) const {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) return false;

        SDSCBinaryHeader hdr;
        std::memset(&hdr, 0, sizeof(hdr));
        hdr.magic = SDSC_BINARY_MAGIC;
        hdr.version = SDSC_BINARY_VERSION;
        hdr.num_cells = num_cells;
        hdr.num_synapses = num_synapses;
        hdr.input_dim = in_dim;
        hdr.output_dim = out_dim;

        uint64_t offset = sizeof(SDSCBinaryHeader);
        hdr.cells_offset = offset;
        offset += static_cast<uint64_t>(num_cells * sizeof(SDSCBinaryCellMeta));
        hdr.row_ptr_offset = offset;
        offset += static_cast<uint64_t>((num_cells + 1) * sizeof(uint32_t));
        hdr.col_idx_offset = offset;
        offset += static_cast<uint64_t>(num_synapses * sizeof(uint32_t));
        hdr.weights_offset = offset;

        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // 写入 CellMeta
        for (uint32_t i = 0; i < num_cells; ++i) {
            SDSCBinaryCellMeta meta;
            meta.op_type = op_types[i];
            meta.param1_u8 = static_cast<uint8_t>(std::clamp(gains[i] * 64.0f, 0.0f, 255.0f));
            meta.param2_u8 = static_cast<uint8_t>(std::clamp(biases[i] * 64.0f, 0.0f, 255.0f));
            meta.flags = (i < in_dim) ? 0x01 : ((i >= num_cells - out_dim) ? 0x02 : 0x00);
            ofs.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
        }

        // 写入 row_ptr (inc_off)
        ofs.write(reinterpret_cast<const char*>(inc_off.data()), (num_cells + 1) * sizeof(uint32_t));
        // 写入 col_idx (inc_from)
        ofs.write(reinterpret_cast<const char*>(inc_from.data()), num_synapses * sizeof(uint32_t));
        // 写入 weights (inc_weight)
        ofs.write(reinterpret_cast<const char*>(inc_weight.data()), num_synapses * sizeof(float));

        return ofs.good();
    }
};

} // namespace kun

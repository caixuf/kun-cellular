#pragma once

/**
 * ============================================================================
 * Software-Defined Silicon Cellular Computer (SDSCC)
 * 仿生哺乳动物新皮层微柱生态阵列 (Modular Cortical Micro-columns)
 * ============================================================================
 * 
 * 体系结构定位 (M2 规范):
 * 1. 突破单体巨大 DAG 编译瓶颈: 将图复杂度从 O(N^2) 分治为 O(M * k)
 * 2. 模拟真实哺乳动物大脑 6 层新皮层微柱架构:
 *    - 微柱 (CorticalMicroColumn): 1,000 ~ 10,000 细胞的纳秒级致密拓扑计算核
 *    - 皮层阵列 (CorticalMacroArray): M 个微柱 + 稀疏宏观神经束 (Macro-Axons)
 * 3. 线程无锁并发: 微柱内部无锁纯确定性推演，微柱间通过双缓冲突触交换流式冲动
 * 4. 轻松承载 10,000,000 ~ 100,000,000 (千万至亿级) 跨区域复合生命体
 */

#include "kun/cellular/sdsc_compact_genome.hpp"
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <algorithm>
#include <omp.h>

namespace kun {

// 柱间长程神经束突触 (Macro-Axon)
struct MacroAxon {
    uint32_t src_column_idx{0}; // 发射端微柱索引
    uint32_t src_cell_idx{0};   // 发射端内部细胞索引
    uint32_t dst_column_idx{0}; // 接收端微柱索引
    uint32_t dst_cell_idx{0};   // 接收端内部细胞索引
    float    weight{1.0f};      // 轴突传递强度
    float    delay_signal{0.0f};// 双缓冲传导信号
};

// 单个独立皮层微柱 (CorticalMicroColumn: 1K ~ 10K 细胞计算核)
struct CorticalMicroColumn {
    uint32_t column_id{0};
    CompactSoAGenome genome;
    CompactSoAGenome::ExecutionRegisters regs;

    std::vector<float> local_inputs;
    std::vector<float> local_outputs;

    void init(uint32_t id, uint32_t cells, uint32_t syns, uint32_t in_d, uint32_t out_d) {
        column_id = id;
        genome = CompactSoAGenome::create_empty(cells, syns, in_d, out_d);
        regs.allocate(cells);
        local_inputs.assign(in_d, 0.0f);
        local_outputs.assign(out_d, 0.0f);
    }

    void reset() {
        regs.reset();
        std::fill(local_inputs.begin(), local_inputs.end(), 0.0f);
        std::fill(local_outputs.begin(), local_outputs.end(), 0.0f);
    }

    inline void step() {
        genome.forward(local_inputs.data(), local_outputs.data(), regs);
    }
};

// 宏观皮层阵列 (CorticalMacroArray)
class CorticalMacroArray {
public:
    CorticalMacroArray(uint32_t num_columns, uint32_t cells_per_col,
                       uint32_t syns_per_col, uint32_t in_dim, uint32_t out_dim)
        : num_columns_(num_columns),
          cells_per_column_(cells_per_col),
          global_in_dim_(in_dim),
          global_out_dim_(out_dim) {
        
        columns_.resize(num_columns_);
        for (uint32_t c = 0; c < num_columns_; ++c) {
            columns_[c].init(c, cells_per_col, syns_per_col, in_dim, out_dim);
        }
    }

    uint64_t total_cells() const {
        return static_cast<uint64_t>(num_columns_) * cells_per_column_;
    }

    uint64_t total_synapses() const {
        uint64_t syn = 0;
        for (const auto& col : columns_) syn += col.genome.num_synapses;
        syn += macro_axons_.size();
        return syn;
    }

    // 建立微柱间稀疏长程轴突连接网络 (Small-World / 2D 网格互联)
    void wire_small_world_axons(uint32_t axons_per_col, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> col_dist(0, num_columns_ - 1);
        std::uniform_int_distribution<uint32_t> cell_dist(0, cells_per_column_ - 1);
        std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);

        macro_axons_.clear();
        macro_axons_.reserve(num_columns_ * axons_per_col);

        for (uint32_t c = 0; c < num_columns_; ++c) {
            for (uint32_t a = 0; a < axons_per_col; ++a) {
                MacroAxon axon;
                axon.src_column_idx = c;
                axon.src_cell_idx = cells_per_column_ > global_out_dim_ ? 
                                    (cells_per_column_ - global_out_dim_ + (a % global_out_dim_)) : (a % cells_per_column_);
                // 连接到近邻或远端柱
                uint32_t target_col = (c + 1 + (rng() % (num_columns_ > 1 ? (num_columns_ - 1) : 1))) % num_columns_;
                axon.dst_column_idx = target_col;
                axon.dst_cell_idx = a % global_in_dim_;
                axon.weight = w_dist(rng);
                axon.delay_signal = 0.0f;
                macro_axons_.push_back(axon);
            }
        }
    }

    void reset() {
        for (auto& col : columns_) col.reset();
        for (auto& axon : macro_axons_) axon.delay_signal = 0.0f;
    }

    /**
     * @brief 皮层宏阵列并行单拍前向推演
     * @param in_tensor 全局输入流 [global_in_dim_]
     * @param out_tensor 全局输出汇聚 [global_out_dim_]
     */
    void forward(const float* in_tensor, float* out_tensor) {
        if (!in_tensor || !out_tensor || num_columns_ == 0) return;

        // 1. 输入感受区注入: 广播/路由到初级感觉柱 (微柱 0)
        for (uint32_t i = 0; i < global_in_dim_; ++i) {
            columns_[0].local_inputs[i] = in_tensor[i];
        }

        // 2. 轴突脉冲信号交换 (跨柱双缓冲)
        for (const auto& axon : macro_axons_) {
            auto& dst_col = columns_[axon.dst_column_idx];
            if (axon.dst_cell_idx < dst_col.local_inputs.size()) {
                dst_col.local_inputs[axon.dst_cell_idx] += axon.delay_signal * axon.weight;
            }
        }

        // 3. 所有微柱高并发独立前向推演 (OpenMP 零锁并行)
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < static_cast<int>(num_columns_); ++c) {
            columns_[c].step();
        }

        // 4. 准备下一拍的轴突传导信号 (产生单拍轴突时延)
        for (auto& axon : macro_axons_) {
            const auto& src_col = columns_[axon.src_column_idx];
            uint32_t out_idx = axon.src_cell_idx >= (cells_per_column_ - global_out_dim_) ?
                               (axon.src_cell_idx - (cells_per_column_ - global_out_dim_)) : 0;
            if (out_idx < src_col.local_outputs.size()) {
                axon.delay_signal = src_col.local_outputs[out_idx];
            }
        }

        // 5. 运动联合皮层动作汇集 (读取尾部运动柱)
        const auto& motor_col = columns_[num_columns_ - 1];
        for (uint32_t o = 0; o < global_out_dim_; ++o) {
            out_tensor[o] = motor_col.local_outputs[o];
        }
    }

    std::vector<CorticalMicroColumn>& columns() { return columns_; }
    const std::vector<CorticalMicroColumn>& columns() const { return columns_; }
    std::vector<MacroAxon>& macro_axons() { return macro_axons_; }

private:
    uint32_t num_columns_{0};
    uint32_t cells_per_column_{0};
    uint32_t global_in_dim_{0};
    uint32_t global_out_dim_{0};
    std::vector<CorticalMicroColumn> columns_;
    std::vector<MacroAxon> macro_axons_;
};

} // namespace kun

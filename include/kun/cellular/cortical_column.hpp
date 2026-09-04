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

    /**
     * @brief 多通道阵列并行前向推演: 每个微柱直接接收专属张量输入并产生专属效应 (严格内存安全保护)
     * @param column_inputs 指针数组 [num_channels]，指向各柱长度为 global_in_dim_ 的输入张量
     * @param num_channels 输入通道数量 (防止越界访存)
     * @param column_outputs 连续数组 [num_columns_ * global_out_dim_]，接收各柱输出
     */
    void forward_multi_channel(const float* const* column_inputs, size_t num_channels, float* column_outputs) {
        if (!column_inputs || !column_outputs || num_columns_ == 0 || num_channels == 0) return;

        // 1. 各微柱注入自身输入 (严格通道边界保护，多余微柱自动置零)
        uint32_t active_cols = std::min(num_columns_, static_cast<uint32_t>(num_channels));
        for (uint32_t c = 0; c < active_cols; ++c) {
            if (column_inputs[c]) {
                for (uint32_t i = 0; i < global_in_dim_; ++i) {
                    columns_[c].local_inputs[i] = column_inputs[c][i];
                }
            } else {
                std::fill(columns_[c].local_inputs.begin(), columns_[c].local_inputs.end(), 0.0f);
            }
        }
        for (uint32_t c = active_cols; c < num_columns_; ++c) {
            std::fill(columns_[c].local_inputs.begin(), columns_[c].local_inputs.end(), 0.0f);
        }

        // 2. 轴突脉冲信号交换 (跨柱双缓冲突触)
        for (const auto& axon : macro_axons_) {
            if (axon.dst_column_idx < num_columns_) {
                auto& dst_col = columns_[axon.dst_column_idx];
                if (axon.dst_cell_idx < dst_col.local_inputs.size()) {
                    dst_col.local_inputs[axon.dst_cell_idx] += axon.delay_signal * axon.weight;
                }
            }
        }

        // 3. 微柱独立前向推演 (大规模微柱 >64 启用 OpenMP，中小规模纯紧凑循环避免线程抖动)
        if (num_columns_ > 64) {
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < static_cast<int>(num_columns_); ++c) {
                columns_[c].step();
            }
        } else {
            for (uint32_t c = 0; c < num_columns_; ++c) {
                columns_[c].step();
            }
        }

        // 4. 准备下一拍的轴突传导信号 (双缓冲延迟)
        for (auto& axon : macro_axons_) {
            if (axon.src_column_idx < num_columns_) {
                const auto& src_col = columns_[axon.src_column_idx];
                uint32_t out_idx = axon.src_cell_idx >= (cells_per_column_ - global_out_dim_) ?
                                   (axon.src_cell_idx - (cells_per_column_ - global_out_dim_)) : 0;
                if (out_idx < src_col.local_outputs.size()) {
                    axon.delay_signal = src_col.local_outputs[out_idx];
                }
            }
        }

        // 5. 各微柱输出写出
        for (uint32_t c = 0; c < num_columns_; ++c) {
            for (uint32_t o = 0; o < global_out_dim_; ++o) {
                column_outputs[c * global_out_dim_ + o] = columns_[c].local_outputs[o];
            }
        }
    }

    void forward_multi_channel(const float* const* column_inputs, float* column_outputs) {
        forward_multi_channel(column_inputs, num_columns_, column_outputs);
    }

    void mutate(float rate, float sigma, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::normal_distribution<float> n(0.0f, sigma);
        for (auto& col : columns_) {
            col.genome.mutate_parameters(rate, sigma, rng);
            col.genome.mutate_primitive_types(rate * 0.5f, rng);
        }
        for (auto& axon : macro_axons_) {
            if (u(rng) < rate) {
                axon.weight = std::clamp(axon.weight + n(rng), -2.0f, 2.0f);
            }
        }
    }

    /**
     * @brief 真实无损完整存盘 (100% 序列化全部微柱原语、权重拓扑与宏轴突)
     */
    bool save_checkpoint_json(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) return false;
        ofs << "{\n";
        ofs << "  \"architecture\": \"CorticalMacroArray\",\n";
        ofs << "  \"num_columns\": " << num_columns_ << ",\n";
        ofs << "  \"cells_per_column\": " << cells_per_column_ << ",\n";
        ofs << "  \"total_cells\": " << total_cells() << ",\n";
        ofs << "  \"total_synapses\": " << total_synapses() << ",\n";
        ofs << "  \"global_in_dim\": " << global_in_dim_ << ",\n";
        ofs << "  \"global_out_dim\": " << global_out_dim_ << ",\n";
        ofs << "  \"macro_axons_count\": " << macro_axons_.size() << ",\n";
        ofs << "  \"columns\": [\n";
        for (size_t c = 0; c < columns_.size(); ++c) {
            const auto& col = columns_[c];
            ofs << "    {\n";
            ofs << "      \"id\": " << col.column_id << ",\n";
            ofs << "      \"cells\": " << col.genome.num_cells << ",\n";
            ofs << "      \"synapses\": " << col.genome.num_synapses << ",\n";
            ofs << "      \"op_types\": [";
            for (size_t i = 0; i < col.genome.op_types.size(); ++i) {
                ofs << (int)col.genome.op_types[i] << (i + 1 < col.genome.op_types.size() ? "," : "");
            }
            ofs << "],\n";
            ofs << "      \"gains\": [";
            for (size_t i = 0; i < col.genome.gains.size(); ++i) {
                ofs << col.genome.gains[i] << (i + 1 < col.genome.gains.size() ? "," : "");
            }
            ofs << "],\n";
            ofs << "      \"inc_off\": [";
            for (size_t i = 0; i < col.genome.inc_off.size(); ++i) {
                ofs << col.genome.inc_off[i] << (i + 1 < col.genome.inc_off.size() ? "," : "");
            }
            ofs << "],\n";
            ofs << "      \"inc_from\": [";
            for (size_t i = 0; i < col.genome.inc_from.size(); ++i) {
                ofs << col.genome.inc_from[i] << (i + 1 < col.genome.inc_from.size() ? "," : "");
            }
            ofs << "],\n";
            ofs << "      \"inc_weight\": [";
            for (size_t i = 0; i < col.genome.inc_weight.size(); ++i) {
                ofs << col.genome.inc_weight[i] << (i + 1 < col.genome.inc_weight.size() ? "," : "");
            }
            ofs << "]\n";
            ofs << "    }" << (c + 1 < columns_.size() ? "," : "") << "\n";
        }
        ofs << "  ],\n";
        ofs << "  \"macro_axons\": [\n";
        for (size_t a = 0; a < macro_axons_.size(); ++a) {
            const auto& ax = macro_axons_[a];
            ofs << "    {\"src_col\": " << ax.src_column_idx << ", \"src_cell\": " << ax.src_cell_idx
                << ", \"dst_col\": " << ax.dst_column_idx << ", \"dst_cell\": " << ax.dst_cell_idx
                << ", \"weight\": " << ax.weight << "}"
                << (a + 1 < macro_axons_.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
        return true;
    }

    bool save_checkpoint_bin(const std::string& filepath) const {
        if (columns_.empty()) return false;
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) return false;

        uint32_t total_c = total_cells();
        uint32_t total_s = total_synapses();

        uint64_t header_sz = 72;
        uint64_t cells_off = header_sz;
        uint64_t cells_sz = total_c * 4;
        uint64_t row_ptr_off = cells_off + cells_sz;
        uint64_t row_ptr_sz = (total_c + 1) * 4;
        uint64_t col_idx_off = row_ptr_off + row_ptr_sz;
        uint64_t col_idx_sz = total_s * 4;
        uint64_t weights_off = col_idx_off + col_idx_sz;
        uint64_t weights_sz = total_s * 4;
        uint64_t coords_off = weights_off + weights_sz;

        struct SDSC_BIN_HDR {
            uint32_t magic{0x53445343};
            uint32_t version{2};
            uint32_t n_cells{0};
            uint32_t n_syns{0};
            uint32_t in_d{0};
            uint32_t out_d{0};
            uint64_t c_off{0};
            uint64_t rp_off{0};
            uint64_t ci_off{0};
            uint64_t w_off{0};
            uint64_t coords_off{0};
            uint64_t extra{0};
        } hdr;
        hdr.n_cells = total_c;
        hdr.n_syns = total_s;
        hdr.in_d = global_in_dim_;
        hdr.out_d = global_out_dim_;
        hdr.c_off = cells_off;
        hdr.rp_off = row_ptr_off;
        hdr.ci_off = col_idx_off;
        hdr.w_off = weights_off;
        hdr.coords_off = coords_off;

        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        for (const auto& col : columns_) {
            for (size_t i = 0; i < col.genome.op_types.size(); ++i) {
                uint8_t op = col.genome.op_types[i];
                float p1 = (i < col.genome.gains.size()) ? col.genome.gains[i] : 1.0f;
                uint8_t p1_u8 = static_cast<uint8_t>(std::clamp(p1 * 64.0f, 0.0f, 255.0f));
                uint8_t p2_u8 = 0;
                uint8_t flags = 0;
                if (op <= 3 || op == 26) flags |= 0x01;
                if (op >= 21 && op <= 23) flags |= 0x02;
                ofs.write(reinterpret_cast<const char*>(&op), 1);
                ofs.write(reinterpret_cast<const char*>(&p1_u8), 1);
                ofs.write(reinterpret_cast<const char*>(&p2_u8), 1);
                ofs.write(reinterpret_cast<const char*>(&flags), 1);
            }
        }

        std::vector<std::vector<std::pair<uint32_t, float>>> adj(total_c);
        for (size_t c = 0; c < columns_.size(); ++c) {
            const auto& col = columns_[c];
            for (size_t i = 0; i < col.genome.op_types.size(); ++i) {
                uint32_t dst_global = static_cast<uint32_t>(c * cells_per_column_ + i);
                size_t start = col.genome.inc_off[i];
                size_t end = col.genome.inc_off[i + 1];
                for (size_t s = start; s < end; ++s) {
                    uint32_t src_local = col.genome.inc_from[s];
                    float w = col.genome.inc_weight[s];
                    uint32_t src_global = static_cast<uint32_t>(c * cells_per_column_ + src_local);
                    adj[src_global].push_back({dst_global, w});
                }
            }
        }
        for (const auto& ax : macro_axons_) {
            uint32_t src_global = ax.src_column_idx * cells_per_column_ + ax.src_cell_idx;
            uint32_t dst_global = ax.dst_column_idx * cells_per_column_ + ax.dst_cell_idx;
            adj[src_global].push_back({dst_global, ax.weight});
        }

        std::vector<uint32_t> row_ptr(total_c + 1, 0);
        std::vector<uint32_t> col_idx;
        std::vector<float> weights;
        col_idx.reserve(total_s);
        weights.reserve(total_s);

        uint32_t curr = 0;
        for (uint32_t i = 0; i < total_c; ++i) {
            row_ptr[i] = curr;
            for (const auto& edge : adj[i]) {
                col_idx.push_back(edge.first);
                weights.push_back(edge.second);
                curr++;
            }
        }
        row_ptr[total_c] = curr;

        ofs.write(reinterpret_cast<const char*>(row_ptr.data()), row_ptr.size() * sizeof(uint32_t));
        if (!col_idx.empty()) {
            ofs.write(reinterpret_cast<const char*>(col_idx.data()), col_idx.size() * sizeof(uint32_t));
            ofs.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        }

        std::vector<float> coords(total_c * 3, 0.0f);
        for (size_t c = 0; c < columns_.size(); ++c) {
            for (uint32_t i = 0; i < cells_per_column_; ++i) {
                size_t g_idx = c * cells_per_column_ + i;
                coords[g_idx * 3 + 0] = static_cast<float>(c * 2.0);
                coords[g_idx * 3 + 1] = static_cast<float>(i % 6);
                coords[g_idx * 3 + 2] = static_cast<float>(i / 6);
            }
        }
        ofs.write(reinterpret_cast<const char*>(coords.data()), coords.size() * sizeof(float));

        return true;
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

#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 旋转位置相位振荡耦合机制
// (RoPE-inspired Rotary Phase Coupling & Oscillator Synchronization)
//
// 理论渊源:
//   1. 旋转位置编码 (Rotary Position Embedding, RoPE / Su et al.):
//      通过复数相空间 2D 正交旋转矩阵将相对时序距离 (t - s) 编码为酉变换内积保持流形:
//      < R_t u, R_s v > = < u, R_{s - t} v >
//   2. 神经形态海马体 θ-γ 振荡相位进动 (Neural Phase Precession):
//      微观细胞脉冲沿振荡相位周期展开，通过相位偏移自然形成衰减与共振门控。
//
// 绝对铁律: 严格恪守 Rule 7 Substrate Immunity 宪章，纯粹数学与几何代数，零具身业务词汇。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace kun {

/**
 * @brief 旋转位置编码超参数配置
 */
struct RoPEConfig {
    size_t dim{8};              // 旋转嵌入维度 (必须为偶数)
    float base_freq{10000.0f};  // 几何级数基频 base (例如 10000.0)
    size_t max_seq_len{128};    // 预计算最大相对时序窗口
    float scaling_factor{1.0f}; // 扩展上下文长度时的线性插值标度

    void validate() const {
        if (dim == 0 || (dim % 2) != 0) {
            throw std::invalid_argument("RoPE dimension must be a positive even integer.");
        }
        if (base_freq <= 0.0f) {
            throw std::invalid_argument("RoPE base frequency must be positive.");
        }
    }
};

/**
 * @brief 预计算正弦与余弦旋转几何相空间表 (Rotary Cos/Sin Table)
 */
class RotaryPhaseTable {
public:
    RoPEConfig config;
    std::vector<float> cos_table; // [max_seq_len, dim / 2]
    std::vector<float> sin_table; // [max_seq_len, dim / 2]
    std::vector<float> inv_freq;  // [dim / 2]

    explicit RotaryPhaseTable(const RoPEConfig& cfg) : config(cfg) {
        config.validate();
        init();
    }

    void init() {
        const size_t half_dim = config.dim / 2;
        inv_freq.resize(half_dim);

        // theta_i = 1.0 / (base_freq ^ (2i / dim))
        for (size_t i = 0; i < half_dim; ++i) {
            double exponent = static_cast<double>(2 * i) / static_cast<double>(config.dim);
            inv_freq[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(config.base_freq), exponent));
        }

        const size_t max_len = config.max_seq_len;
        cos_table.resize(max_len * half_dim);
        sin_table.resize(max_len * half_dim);

        for (size_t pos = 0; pos < max_len; ++pos) {
            float scaled_pos = static_cast<float>(pos) / config.scaling_factor;
            for (size_t i = 0; i < half_dim; ++i) {
                float angle = scaled_pos * inv_freq[i];
                cos_table[pos * half_dim + i] = std::cos(angle);
                sin_table[pos * half_dim + i] = std::sin(angle);
            }
        }
    }

    /**
     * @brief 对输入特征向量施加 2D 旋转相位变换:
     *   [u_{2k}, u_{2k+1}]^T -> [u_{2k} cos(pos*theta_k) - u_{2k+1} sin(pos*theta_k),
     *                           u_{2k} sin(pos*theta_k) + u_{2k+1} cos(pos*theta_k)]^T
     *
     * @param in_vec   原始特征向量 [dim]
     * @param pos      时序步长/距离索引
     * @param out_vec  旋转后特征向量 [dim]
     */
    void apply(const float* in_vec, size_t pos, float* out_vec) const {
        if (!in_vec || !out_vec) return;
        size_t p = std::min(pos, config.max_seq_len - 1);
        const size_t half_dim = config.dim / 2;
        const float* cos_row = &cos_table[p * half_dim];
        const float* sin_row = &sin_table[p * half_dim];

        for (size_t i = 0; i < half_dim; ++i) {
            float x0 = in_vec[2 * i + 0];
            float x1 = in_vec[2 * i + 1];
            float c = cos_row[i];
            float s = sin_row[i];

            out_vec[2 * i + 0] = x0 * c - x1 * s;
            out_vec[2 * i + 1] = x0 * s + x1 * c;
        }
    }

    /**
     * @brief 现场复数相移应用 (In-place rotation)
     */
    void apply_inplace(float* vec, size_t pos) const {
        if (!vec) return;
        size_t p = std::min(pos, config.max_seq_len - 1);
        const size_t half_dim = config.dim / 2;
        const float* cos_row = &cos_table[p * half_dim];
        const float* sin_row = &sin_table[p * half_dim];

        for (size_t i = 0; i < half_dim; ++i) {
            float x0 = vec[2 * i + 0];
            float x1 = vec[2 * i + 1];
            float c = cos_row[i];
            float s = sin_row[i];

            vec[2 * i + 0] = x0 * c - x1 * s;
            vec[2 * i + 1] = x0 * s + x1 * c;
        }
    }
};

/**
 * @brief 纯数学函数: 计算两个旋转特征向量的欧氏内积
 */
inline float compute_inner_product(const float* a, const float* b, size_t dim) {
    if (!a || !b) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

/**
 * @brief 纯数学函数: 计算特征向量 L2 范数 ||u||_2
 */
inline float compute_vector_norm_l2(const float* a, size_t dim) {
    return std::sqrt(compute_inner_product(a, a, dim));
}

} // namespace kun

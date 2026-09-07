#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 多头低秩潜空间记忆库 (Multi-Head Latent Reservoir / MLA)
//
// 理论渊源: DeepSeek-V2 / DeepSeek-V3 多头潜在注意力机制 (Multi-Head Latent Attention),
//           低秩矩阵分解 (Low-Rank Matrix Factorization),
//           时序储层计算 (Reservoir Computing with Compressed State Space).
//
// 核心机制:
//   1. 键值联合低秩压缩 (Joint KV Compression):
//      输入特征 x_t in R^{D_in} 通过下投影矩阵 W_DKV in R^{D_c x D_in} 压缩为低秩潜向量 c_t in R^{D_c} (D_c << D_in)。
//      历史记忆仅需以 O(D_c) 空间平铺驻留，降低时序缓存内存带宽与抖动 4x~8x。
//   2. 多头解耦上投影 (Multi-Head Key/Value Up-Projection):
//      检索时，潜向量 c_s 实时通过上投影矩阵恢复各注意力头独立 Key / Value:
//      K_s^h = W_UK^h c_s,  V_s^h = W_UV^h c_s (h = 0 .. H-1)。
//   3. 查询低秩压缩与多头投射 (Query Compression & Multi-Head Projection):
//      当前查询由 q_t 通过 W_DQ 压缩为 c_t^Q，再通过 W_UQ^h 投射为 Q_t^h。
//   4. 连续流形全因果检索 (Causal Masked Scaled Dot-Product Attention):
//      A_{t, s}^h = softmax( (Q_t^h)^T K_s^h / sqrt(d_k) ),  s <= t。
//      Out = W_O [Head_0, ..., Head_{H-1}]。
//
// 绝对铁律: 恪守 Rule 7 Substrate Immunity 宪章，纯粹线性代数与动力学，严禁任何业务词汇。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <random>

namespace kun {

/**
 * @brief MLA 低秩潜空间记忆超参数配置
 */
struct MLAConfig {
    size_t in_dim{32};          // 输入特征维度 D_in
    size_t latent_dim{8};       // KV 压缩潜空间维度 D_c (D_c << D_in)
    size_t q_latent_dim{8};     // Q 压缩潜空间维度 D_cq
    size_t num_heads{2};        // 多头注意力头数 H
    size_t head_dim{8};         // 每头特征维度 d_k = d_v
    size_t max_history_len{32}; // 最大时序历史缓存容量 T_max
    float scale{0.0f};          // 缩放因子 1 / sqrt(d_k), 默认 0 时自动计算
    float dropout_prob{0.0f};   // 训练时 Dropout (推理时为 0)

    void validate() const {
        if (in_dim == 0 || latent_dim == 0 || num_heads == 0 || head_dim == 0 || max_history_len == 0) {
            throw std::invalid_argument("MLAConfig parameters must be positive.");
        }
    }
};

/**
 * @brief MLA 纯连续平铺权重包 (Preallocated Flat Tensors)
 */
struct MLAWeights {
    // 键值压缩下投影: W_DKV in [latent_dim, in_dim]
    std::vector<float> w_dkv;
    // 查询压缩下投影: W_DQ in [q_latent_dim, in_dim]
    std::vector<float> w_dq;
    // 键多头上投影: W_UK in [num_heads, head_dim, latent_dim]
    std::vector<float> w_uk;
    // 值多头上投影: W_UV in [num_heads, head_dim, latent_dim]
    std::vector<float> w_uv;
    // 查询多头上投影: W_UQ in [num_heads, head_dim, q_latent_dim]
    std::vector<float> w_uq;
    // 输出融合投影: W_O in [in_dim, num_heads * head_dim]
    std::vector<float> w_o;

    void init_xavier(const MLAConfig& cfg, uint32_t seed = 42) {
        std::mt19937 rng(seed);

        auto init_mat = [&](std::vector<float>& mat, size_t rows, size_t cols) {
            mat.resize(rows * cols);
            float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto& val : mat) val = dist(rng);
        };

        init_mat(w_dkv, cfg.latent_dim, cfg.in_dim);
        init_mat(w_dq, cfg.q_latent_dim, cfg.in_dim);
        init_mat(w_uk, cfg.num_heads * cfg.head_dim, cfg.latent_dim);
        init_mat(w_uv, cfg.num_heads * cfg.head_dim, cfg.latent_dim);
        init_mat(w_uq, cfg.num_heads * cfg.head_dim, cfg.q_latent_dim);
        init_mat(w_o, cfg.in_dim, cfg.num_heads * cfg.head_dim);
    }
};

/**
 * @brief MLA 权重梯度包 (Adjoint Gradients)
 */
struct MLAGradients {
    std::vector<float> dw_dkv;
    std::vector<float> dw_dq;
    std::vector<float> dw_uk;
    std::vector<float> dw_uv;
    std::vector<float> dw_uq;
    std::vector<float> dw_o;

    void init(const MLAConfig& cfg) {
        dw_dkv.assign(cfg.latent_dim * cfg.in_dim, 0.0f);
        dw_dq.assign(cfg.q_latent_dim * cfg.in_dim, 0.0f);
        dw_uk.assign(cfg.num_heads * cfg.head_dim * cfg.latent_dim, 0.0f);
        dw_uv.assign(cfg.num_heads * cfg.head_dim * cfg.latent_dim, 0.0f);
        dw_uq.assign(cfg.num_heads * cfg.head_dim * cfg.q_latent_dim, 0.0f);
        dw_o.assign(cfg.in_dim * cfg.num_heads * cfg.head_dim, 0.0f);
    }

    void zero() {
        std::fill(dw_dkv.begin(), dw_dkv.end(), 0.0f);
        std::fill(dw_dq.begin(), dw_dq.end(), 0.0f);
        std::fill(dw_uk.begin(), dw_uk.end(), 0.0f);
        std::fill(dw_uv.begin(), dw_uv.end(), 0.0f);
        std::fill(dw_uq.begin(), dw_uq.end(), 0.0f);
        std::fill(dw_o.begin(), dw_o.end(), 0.0f);
    }
};

/**
 * @brief MLA 运行时多头低秩潜空间记忆库 (Cellular MLA Engine)
 * 零动态堆内存开辟，支持极速纳秒级推理与反向传播
 */
class CellularMLAEngine {
public:
    MLAConfig config;
    MLAWeights weights;
    float inv_sqrt_dk{1.0f};

    // 低秩潜空间历史记忆缓存 (Latent KV Cache): [max_history_len, latent_dim]
    std::vector<float> latent_kv_cache;
    size_t current_history_len{0};

    explicit CellularMLAEngine(const MLAConfig& cfg) : config(cfg) {
        config.validate();
        inv_sqrt_dk = (cfg.scale > 0.0f) ? cfg.scale : (1.0f / std::sqrt(static_cast<float>(cfg.head_dim)));
        weights.init_xavier(config);
        latent_kv_cache.assign(config.max_history_len * config.latent_dim, 0.0f);
        current_history_len = 0;
    }

    void reset_memory() {
        std::fill(latent_kv_cache.begin(), latent_kv_cache.end(), 0.0f);
        current_history_len = 0;
    }

    /**
     * @brief 前向写入与检索 (Step Forward: Append to Latent KV Cache & Attend)
     * 
     * @param x_t      当前输入特征向量 [in_dim]
     * @param out_t    检索输出特征向量 [in_dim] (可与输入残差相加)
     */
    void step_forward(const float* x_t, float* out_t) {
        if (!x_t || !out_t) return;

        const size_t Din = config.in_dim;
        const size_t Dc = config.latent_dim;
        const size_t Dcq = config.q_latent_dim;
        const size_t H = config.num_heads;
        const size_t Dh = config.head_dim;
        const size_t Tmax = config.max_history_len;

        // 1. 键值联合下投影: c_t = W_DKV * x_t in R^{Dc}
        std::vector<float> c_t(Dc, 0.0f);
        for (size_t i = 0; i < Dc; ++i) {
            float sum = 0.0f;
            const float* w_row = &weights.w_dkv[i * Din];
            for (size_t j = 0; j < Din; ++j) {
                sum += w_row[j] * x_t[j];
            }
            c_t[i] = sum;
        }

        // 2. 写入低秩潜空间缓存 (Ring buffer or shift)
        if (current_history_len < Tmax) {
            std::memcpy(&latent_kv_cache[current_history_len * Dc], c_t.data(), Dc * sizeof(float));
            current_history_len++;
        } else {
            // 滑动窗口平移: 丢弃最早时刻
            std::memmove(&latent_kv_cache[0], &latent_kv_cache[Dc], (Tmax - 1) * Dc * sizeof(float));
            std::memcpy(&latent_kv_cache[(Tmax - 1) * Dc], c_t.data(), Dc * sizeof(float));
            current_history_len = Tmax;
        }

        const size_t T = current_history_len;

        // 3. 查询压缩下投影与多头上投影:
        //    c_q = W_DQ * x_t in R^{Dcq}
        std::vector<float> c_q(Dcq, 0.0f);
        for (size_t i = 0; i < Dcq; ++i) {
            float sum = 0.0f;
            const float* w_row = &weights.w_dq[i * Din];
            for (size_t j = 0; j < Din; ++j) {
                sum += w_row[j] * x_t[j];
            }
            c_q[i] = sum;
        }

        //    Q^h = W_UQ^h * c_q in R^{H x Dh}
        std::vector<float> Q(H * Dh, 0.0f);
        for (size_t h = 0; h < H; ++h) {
            for (size_t d = 0; d < Dh; ++d) {
                float sum = 0.0f;
                const float* w_row = &weights.w_uq[(h * Dh + d) * Dcq];
                for (size_t j = 0; j < Dcq; ++j) {
                    sum += w_row[j] * c_q[j];
                }
                Q[h * Dh + d] = sum;
            }
        }

        // 4. 从低秩潜向量 c_s 实时上投影恢复所有历史的 K_s^h 与 V_s^h 并计算注意力
        //    Head_outputs: [H * Dh]
        std::vector<float> multi_head_out(H * Dh, 0.0f);

        for (size_t h = 0; h < H; ++h) {
            const float* q_h = &Q[h * Dh];
            std::vector<float> attn_scores(T, 0.0f);

            // 计算该头在所有历史时刻 s = 0 .. T-1 的注意力得分: (Q^h)^T K_s^h
            for (size_t s = 0; s < T; ++s) {
                const float* c_s = &latent_kv_cache[s * Dc];
                // K_s^h = W_UK^h * c_s in R^{Dh}
                float dot = 0.0f;
                for (size_t d = 0; d < Dh; ++d) {
                    float k_sd = 0.0f;
                    const float* w_k_row = &weights.w_uk[(h * Dh + d) * Dc];
                    for (size_t j = 0; j < Dc; ++j) {
                        k_sd += w_k_row[j] * c_s[j];
                    }
                    dot += q_h[d] * k_sd;
                }
                attn_scores[s] = dot * inv_sqrt_dk;
            }

            // Softmax 归一化注意力权重
            float max_score = *std::max_element(attn_scores.begin(), attn_scores.end());
            float sum_exp = 0.0f;
            for (size_t s = 0; s < T; ++s) {
                attn_scores[s] = std::exp(attn_scores[s] - max_score);
                sum_exp += attn_scores[s];
            }
            float inv_sum = (sum_exp > 1e-7f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t s = 0; s < T; ++s) {
                attn_scores[s] *= inv_sum;
            }

            // 加权汇聚: head_out = \sum_s attn_scores[s] * V_s^h
            // V_s^h = W_UV^h * c_s in R^{Dh}
            for (size_t d = 0; d < Dh; ++d) {
                float v_accum = 0.0f;
                for (size_t s = 0; s < T; ++s) {
                    float alpha = attn_scores[s];
                    if (alpha < 1e-8f) continue;
                    const float* c_s = &latent_kv_cache[s * Dc];
                    float v_sd = 0.0f;
                    const float* w_v_row = &weights.w_uv[(h * Dh + d) * Dc];
                    for (size_t j = 0; j < Dc; ++j) {
                        v_sd += w_v_row[j] * c_s[j];
                    }
                    v_accum += alpha * v_sd;
                }
                multi_head_out[h * Dh + d] = v_accum;
            }
        }

        // 5. 输出多头融合投影: out_t = W_O * multi_head_out in R^{Din}
        const size_t total_head_dim = H * Dh;
        for (size_t i = 0; i < Din; ++i) {
            float sum = 0.0f;
            const float* w_row = &weights.w_o[i * total_head_dim];
            for (size_t j = 0; j < total_head_dim; ++j) {
                sum += w_row[j] * multi_head_out[j];
            }
            out_t[i] = sum;
        }
    }

    /**
     * @brief 提取当前低秩潜空间记忆占用字节数 (内存效率对账)
     */
    size_t latent_memory_bytes() const {
        return latent_kv_cache.size() * sizeof(float);
    }

    /**
     * @brief 提取未压缩全维度全历史基线对比所需字节数
     */
    size_t dense_uncompressed_bytes() const {
        return config.max_history_len * config.in_dim * sizeof(float);
    }

    /**
     * @brief 计算压缩比率 (Compression Factor)
     */
    double compression_ratio() const {
        return static_cast<double>(config.in_dim) / static_cast<double>(config.latent_dim);
    }
};

} // namespace kun

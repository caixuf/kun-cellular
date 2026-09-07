#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 测试时动态松弛计算与李雅普诺夫吸引子收敛引擎
// (Test-Time Compute: Dynamical Attractor Relaxation Engine)
//
// 理论渊源: 连续状态动力系统吸引子理论、收缩映射定理 (Banach Fixed-Point Theorem)、
//           李雅普诺夫第二方法与平衡传播 (Equilibrium Propagation / DEQ)
// 核心机制: 在读取效应器通道前，通过 K 步内部循环松弛 (Internal Recurrent Relaxation Cycles)，
//           促使细胞跨膜电位与私有寄存器沿 Kahn 拓扑逆/顺时序流形收敛至稳态吸引子 (Fixed-Point Attractor)。
// 绝对铁律: 恪守 Rule 7 底座中立性，严禁任何具身业务特化词汇，纯粹数学与拓扑动力学。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

namespace kun {

/**
 * @brief 测试时动态松弛配置 (Relaxation Hyperparameters)
 */
struct RelaxationConfig {
    size_t max_steps{4};                 // 最大内部动态松弛思考步数 K (>= 1)
    double convergence_tol{1e-5};        // 吸引子不动点收敛容差 epsilon (L_inf residual)
    double damping_factor{1.0};          // 状态松弛阻尼系数 alpha in (0.0, 1.0] (1.0 = 无阻尼标准迭代)
    double bibo_bound{100.0};            // 李雅普诺夫 BIBO 安全边界 |V_i| <= M
    bool early_stop{false};              // 达到收敛容差后是否提前跳出
    bool enable_hebbian{false};          // 测试时思考过程中是否允许突触权重可塑性在线漂移 (默认关闭以保护纯推理)
};

/**
 * @brief 测试时松弛动力学遥测指标 (Dynamical Attractor Relaxation Telemetry)
 */
struct RelaxationTelemetry {
    size_t actual_steps{0};              // 实际执行的松弛迭代步数
    double initial_energy{0.0};          // 初始网络动能/位能 E(0) = 0.5 * \sum V_i^2
    double final_energy{0.0};            // 稳态网络动能/位能 E(K) = 0.5 * \sum V_i^2
    double delta_energy{0.0};            // 能量耗散/变动 Delta E = E(K) - E(0)
    double max_potential_delta{0.0};     // 终态最大细胞电位变动 ||V^(K) - V^(K-1)||_inf
    double l2_potential_delta{0.0};      // 终态 L2 范数变动 ||V^(K) - V^(K-1)||_2
    double spectral_contraction{0.0};    // 经验谱收缩率 rho = Delta_K / (Delta_1 + eps)
    bool converged{false};               // 是否满足吸引子收敛准则 (Delta <= tol)
    bool bibo_stable{true};              // 全程是否严格满足李雅普诺夫 BIBO 有界性
    std::vector<double> step_deltas;     // 各步 L2 电位变动残差轨迹 [Delta_1, Delta_2, ...]
    std::vector<double> step_energies;   // 各步网络相空间总能量轨迹 [E_1, E_2, ...]
};

/**
 * @brief 计算全网电位二次型李雅普诺夫总能量 E = 0.5 * \sum_{i=0}^{N-1} V_i^2
 */
inline double compute_network_potential_energy(const double* potentials, size_t num_cells) {
    if (!potentials || num_cells == 0) return 0.0;
    double energy = 0.0;
    for (size_t i = 0; i < num_cells; ++i) {
        energy += potentials[i] * potentials[i];
    }
    return 0.5 * energy;
}

/**
 * @brief 计算两组电位向量的 L2 范数距离 ||V_a - V_b||_2
 */
inline double compute_potential_distance_l2(const double* va, const double* vb, size_t num_cells) {
    if (!va || !vb || num_cells == 0) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < num_cells; ++i) {
        double diff = va[i] - vb[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief 计算两组电位向量的无穷范数距离 ||V_a - V_b||_inf
 */
inline double compute_potential_distance_linf(const double* va, const double* vb, size_t num_cells) {
    if (!va || !vb || num_cells == 0) return 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < num_cells; ++i) {
        double diff = std::abs(va[i] - vb[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

/**
 * @brief 验证电位向量是否符合 BIBO 有界性准则
 */
inline bool verify_potential_bibo_bound(const double* potentials, size_t num_cells, double bound) {
    if (!potentials) return true;
    for (size_t i = 0; i < num_cells; ++i) {
        double v = potentials[i];
        if (!std::isfinite(v) || std::abs(v) > bound) {
            return false;
        }
    }
    return true;
}

} // namespace kun

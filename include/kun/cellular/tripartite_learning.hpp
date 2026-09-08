#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cstdint>
#include <cassert>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/morphogenetic_population.hpp"

namespace kun {

// ============================================================================
// 1. 闭式解岭回归读出求解器 (Ridge Closed-Form Readout Solver)
//    效应层永远闭式解: W_out = (R^T * R + lambda * I)^(-1) * R^T * Y
// ============================================================================
class RidgeReadoutSolver {
public:
    // 鲁棒 Cholesky 分解求解器 A * x = b (A 为 n x n 对称正定矩阵)
    static bool solve_cholesky(const std::vector<float>& A, const std::vector<float>& b, size_t n, std::vector<float>& x) {
        if (n == 0) return true;
        x.assign(n, 0.0f);
        if (n == 1) {
            float a0 = A[0];
            if (std::abs(a0) < 1e-12f) return false;
            x[0] = b[0] / a0;
            return true;
        }

        // L 矩阵下三角存储
        std::vector<float> L(n * n, 0.0f);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i * n + k] * L[j * n + k];
                }

                if (i == j) {
                    float diag = A[i * n + i] - sum;
                    if (diag <= 1e-9f) {
                        // 奇异或非正定，退化保护
                        diag = 1e-4f;
                    }
                    L[i * n + j] = std::sqrt(diag);
                } else {
                    float l_jj = L[j * n + j];
                    if (std::abs(l_jj) < 1e-9f) l_jj = 1e-4f;
                    L[i * n + j] = (A[i * n + j] - sum) / l_jj;
                }
            }
        }

        // 前向代入: L * y = b
        std::vector<float> y(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (size_t k = 0; k < i; ++k) {
                sum += L[i * n + k] * y[k];
            }
            float l_ii = L[i * n + i];
            if (std::abs(l_ii) < 1e-9f) l_ii = 1e-4f;
            y[i] = (b[i] - sum) / l_ii;
        }

        // 后向代入: L^T * x = y
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            float sum = 0.0f;
            for (size_t k = i + 1; k < n; ++k) {
                sum += L[k * n + i] * x[k];
            }
            float l_ii = L[i * n + i];
            if (std::abs(l_ii) < 1e-9f) l_ii = 1e-4f;
            x[i] = (y[i] - sum) / l_ii;
        }

        return true;
    }

    // 为个体所有效应器细胞求解闭式最优读出突触权重
    // reservoir_tape: [T x num_cells] 各时间步各细胞的 activation 输出
    // target_tape:    [T x num_targets] 各时间步的目标动作标靶
    // target_cell_indices: 对应的效应器细胞在 cells 数组中的下标
    static bool solve_organism_readouts(
        CellularOrganism& org,
        const std::vector<std::vector<float>>& reservoir_tape,
        const std::vector<std::vector<float>>& target_tape,
        const std::vector<size_t>& target_cell_indices,
        float lambda_reg = 1e-3f)
    {
        size_t T = reservoir_tape.size();
        if (T == 0 || target_tape.size() != T || target_cell_indices.empty()) {
            return false;
        }

        for (size_t t_idx = 0; t_idx < target_cell_indices.size(); ++t_idx) {
            size_t eff_idx = target_cell_indices[t_idx];
            if (eff_idx >= org.cells.size()) continue;
            uint32_t eff_id = org.cells[eff_idx].id;

            // 查找所有汇聚到该效应器的突触
            std::vector<size_t> incoming_syn_indices;
            std::vector<size_t> from_cell_indices;

            for (size_t s = 0; s < org.synapses.size(); ++s) {
                if (org.synapses[s].to_cell_id == eff_id && org.synapses[s].to_port == 0) {
                    incoming_syn_indices.push_back(s);
                    // 找到 from_cell_idx
                    for (size_t c = 0; c < org.cells.size(); ++c) {
                        if (org.cells[c].id == org.synapses[s].from_cell_id) {
                            from_cell_indices.push_back(c);
                            break;
                        }
                    }
                }
            }

            size_t M = from_cell_indices.size();
            if (M == 0) continue;

            // 构造 R_k (T x M) 与 y_k (T)
            // 计算 A = R_k^T * R_k + lambda * I (M x M)
            // 计算 b = R_k^T * y_k (M)
            std::vector<float> A(M * M, 0.0f);
            std::vector<float> b(M, 0.0f);

            for (size_t t = 0; t < T; ++t) {
                float y_val = (t_idx < target_tape[t].size()) ? target_tape[t][t_idx] : 0.0f;
                for (size_t i = 0; i < M; ++i) {
                    float r_i = reservoir_tape[t][from_cell_indices[i]];
                    b[i] += r_i * y_val;
                    for (size_t j = 0; j < M; ++j) {
                        float r_j = reservoir_tape[t][from_cell_indices[j]];
                        A[i * M + j] += r_i * r_j;
                    }
                }
            }

            // 添加 Tikhonov 岭正则化项 lambda * I
            for (size_t i = 0; i < M; ++i) {
                A[i * M + i] += lambda_reg * static_cast<float>(T);
            }

            // 求解线性方程组
            std::vector<float> opt_weights(M, 0.0f);
            if (solve_cholesky(A, b, M, opt_weights)) {
                for (size_t i = 0; i < M; ++i) {
                    size_t syn_idx = incoming_syn_indices[i];
                    float w = opt_weights[i];
                    if (std::isfinite(w)) {
                        org.synapses[syn_idx].weight = std::clamp(static_cast<double>(w), -10.0, 10.0);
                    }
                }
            }
        }

        // 重新编译图以更新运行时权重
        org.compile();
        return true;
    }
};

// ============================================================================
// 2. PBT (基于种群的在线超参调度器) 与个体超参状态
// ============================================================================
struct PBTIndividualHyperparams {
    float learning_rate{0.01f};
    float mutation_rate{0.05f};
    float bptt_grad_clip{1.0f};
    uint32_t bptt_steps{10};

    void explore(std::mt19937& rng) {
        std::uniform_real_distribution<float> factor_dist(0.8f, 1.25f);
        learning_rate = std::clamp(learning_rate * factor_dist(rng), 1e-4f, 0.1f);
        mutation_rate = std::clamp(mutation_rate * factor_dist(rng), 0.01f, 0.3f);
    }
};

// ============================================================================
// 3. 训练轨迹容器 (Training Trajectory Batch)
// ============================================================================
struct TrainingTrajectoryBatch {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

// ============================================================================
// 4. 三权分立学习个体 (Tripartite Individual)
// ============================================================================
struct TripartiteIndividual {
    uint64_t id{0};
    uint32_t species_id{0};
    CellularOrganism organism;
    PBTIndividualHyperparams pbt_params;
    double task_fitness{0.0};
    double novelty_score{0.0};
    double composite_fitness{0.0};
    double max_loop_gain{0.0};
    bool is_lyapunov_certified{false};
};

// ============================================================================
// 5. 三权分立学习机 (Tripartite Learning Stack)
//    - 结构搜索 (NEAT): 加/删细胞与突触、物种划分、相容性距离
//    - 参数微调 (BPTT): 仅更新连续权重与物理时间常数、禁止动拓扑
//    - 读出解析 (Ridge): 效应层闭式正则化瞬态求解
//    - 稳定性保证 (Lyapunov): 每次梯度步后流形投影，永久保证 rho <= 0.95
// ============================================================================
class TripartiteLearningStack {
public:
    size_t population_size{32};
    float lambda_novelty{0.1f};
    float lambda_parsimony{0.005f}; // 描述长度惩罚 (细胞数 + 0.1*突触数)
    float speciation_threshold{0.40f};
    uint32_t generation{0};
    uint64_t next_id{1};

    std::vector<TripartiteIndividual> population;
    std::mt19937 rng;
    CellularBPTTEngine bptt_engine;

    explicit TripartiteLearningStack(size_t pop_size = 32, uint32_t seed = 42)
        : population_size(pop_size), rng(seed), bptt_engine(64)
    {
        initialize_population();
    }

    void initialize_population() {
        population.clear();
        generation = 0;
        for (size_t i = 0; i < population_size; ++i) {
            TripartiteIndividual ind;
            ind.id = next_id++;
            ind.species_id = 1;
            ind.organism = make_default_seed_organism();
            ind.organism.compile();
            ind.pbt_params.learning_rate = 0.01f;
            ind.pbt_params.mutation_rate = 0.05f;
            population.push_back(ind);
        }
    }

    // 计算包含任务表现、新颖性奖励与描述长度惩罚的混合目标
    double evaluate_composite_fitness(TripartiteIndividual& ind) {
        size_t n_cells = ind.organism.cells.size();
        size_t n_synapses = ind.organism.synapses.size();
        double parsimony_penalty = static_cast<double>(n_cells) + 0.1 * static_cast<double>(n_synapses);

        ind.composite_fitness = ind.task_fitness 
                              + static_cast<double>(lambda_novelty) * ind.novelty_score 
                              - static_cast<double>(lambda_parsimony) * parsimony_penalty;
        return ind.composite_fitness;
    }

    // 运行一代三权分立协同优化
    template<typename EvaluatorFunc, typename TrajectoryFunc>
    void step_generation(EvaluatorFunc&& task_evaluator, TrajectoryFunc&& get_trajectory) {
        generation++;

        // -------------------------------------------------------------
        // Step 1: 环境推演评估与复合适应度计算
        // -------------------------------------------------------------
        for (auto& ind : population) {
            task_evaluator(ind);
            evaluate_composite_fitness(ind);
        }

        // 按复合适应度降序排列
        std::sort(population.begin(), population.end(), [](const auto& a, const auto& b) {
            return a.composite_fitness > b.composite_fitness;
        });

        // -------------------------------------------------------------
        // Step 2: 参数微调 (BPTT) —— 对 Top 25% 精英个体进行 K 步梯度优化
        // 严格遵守宪章法则: BPTT 只改突触权重与原语参数，绝不增删结构
        // -------------------------------------------------------------
        size_t elite_count = std::max(size_t(1), population_size / 4);
        for (size_t e = 0; e < elite_count; ++e) {
            auto& ind = population[e];
            TrainingTrajectoryBatch trajectory_data = get_trajectory(ind);

            if (ind.pbt_params.bptt_steps > 0 && !trajectory_data.inputs.empty() && trajectory_data.inputs.size() == trajectory_data.targets.size()) {
                size_t steps = std::min(size_t(ind.pbt_params.bptt_steps), trajectory_data.inputs.size());
                
                // 截取当前短窗
                std::vector<std::vector<float>> win_inputs(trajectory_data.inputs.begin(), trajectory_data.inputs.begin() + steps);
                std::vector<std::vector<float>> win_targets(trajectory_data.targets.begin(), trajectory_data.targets.begin() + steps);

                // 运行 BPTT 录带与逆序反传
                bptt_engine.forward_sequence(ind.organism, win_inputs);
                BPTTGradients grads;
                bptt_engine.backward(ind.organism, win_targets, grads);

                // Adam 参数更新
                bptt_engine.step_adam(ind.organism, grads, ind.pbt_params.learning_rate);

                // 李雅普诺夫稳定流形投影 (强制将所有环路收缩至 rho <= 0.95)
                ind.max_loop_gain = static_cast<double>(bptt_engine.apply_lyapunov_projection(ind.organism, 0.95f));
                ind.is_lyapunov_certified = (ind.max_loop_gain <= 0.95);

                // -------------------------------------------------------------
                // Step 3: 闭式解岭回归 (Ridge Readout) —— 瞬态求解效应器读出权重
                // -------------------------------------------------------------
                std::vector<size_t> effector_indices;
                for (size_t c = 0; c < ind.organism.cells.size(); ++c) {
                    if (ind.organism.cells[c].type == CellType::ACT_PRIMARY_POSITIVE ||
                        ind.organism.cells[c].type == CellType::ACT_PRIMARY_NEGATIVE) {
                        effector_indices.push_back(c);
                    }
                }

                if (!effector_indices.empty()) {
                    std::vector<std::vector<float>> reservoir_tape(steps);
                    for (size_t t = 0; t < steps; ++t) {
                        reservoir_tape[t] = bptt_engine.get_tape_step(t).cell_outputs;
                    }
                    RidgeReadoutSolver::solve_organism_readouts(
                        ind.organism, reservoir_tape, win_targets, effector_indices, 1e-3f);
                }
            }
        }

        // -------------------------------------------------------------
        // Step 4: PBT 调度与结构搜索 (NEAT) —— Exploit & Explore
        // -------------------------------------------------------------
        // 后 20% 个体被前 20% 精英 Exploit 复制覆盖，并进入 Explore 空间
        size_t replace_count = population_size / 5;
        for (size_t r = 0; r < replace_count; ++r) {
            size_t source_idx = r % elite_count;
            size_t target_idx = population_size - 1 - r;

            population[target_idx].organism = population[source_idx].organism;
            population[target_idx].pbt_params = population[source_idx].pbt_params;
            population[target_idx].pbt_params.explore(rng);

            // NEAT 拓扑变异 (增加突触、加细胞、或修剪静默突触)
            apply_neat_structural_mutation(population[target_idx].organism, population[target_idx].pbt_params.mutation_rate);
            population[target_idx].organism.compile();
        }
    }

    // NEAT 结构搜索变异算子
    void apply_neat_structural_mutation(CellularOrganism& org, float mut_rate) {
        std::uniform_real_distribution<float> p_dist(0.0f, 1.0f);

        // 1. 添加突触 (Add Synapse)
        if (p_dist(rng) < mut_rate * 2.0f && org.cells.size() >= 2) {
            size_t from_idx = rng() % org.cells.size();
            size_t to_idx = rng() % org.cells.size();
            if (from_idx != to_idx) {
                Synapse syn;
                syn.from_cell_id = org.cells[from_idx].id;
                syn.to_cell_id = org.cells[to_idx].id;
                syn.to_port = static_cast<uint8_t>(rng() % 2);
                syn.weight = (p_dist(rng) - 0.5f) * 0.5f;
                syn.is_active = true;
                org.synapses.push_back(syn);
            }
        }

        // 2. 添加细胞 (Add Node / Split Synapse)
        if (p_dist(rng) < mut_rate && !org.synapses.empty()) {
            size_t syn_idx = rng() % org.synapses.size();
            auto old_syn = org.synapses[syn_idx];

            const auto new_cell_id = org.allocate_cell_id();
            if (!new_cell_id.has_value()) return;
            Cell new_cell;
            new_cell.id = *new_cell_id;
            new_cell.type = CellType::OP_INTEGRAL;
            new_cell.param1 = 0.1f;
            new_cell.param2 = 1.0f;
            org.cells.push_back(new_cell);

            // 禁用旧突触，接入两条新突触
            org.synapses[syn_idx].is_active = false;

            Synapse syn1;
            syn1.from_cell_id = old_syn.from_cell_id;
            syn1.to_cell_id = *new_cell_id;
            syn1.to_port = 0;
            syn1.weight = 1.0f;

            Synapse syn2;
            syn2.from_cell_id = *new_cell_id;
            syn2.to_cell_id = old_syn.to_cell_id;
            syn2.to_port = old_syn.to_port;
            syn2.weight = old_syn.weight;

            org.synapses.push_back(syn1);
            org.synapses.push_back(syn2);
        }

        // 3. 修剪静默突触 (Prune Silent Synapses)
        if (p_dist(rng) < mut_rate * 0.5f) {
            org.synapses.erase(
                std::remove_if(org.synapses.begin(), org.synapses.end(), [](const Synapse& s) {
                    return !s.is_active || std::abs(s.weight) < 1e-4;
                }), org.synapses.end());
        }
    }

private:
    static CellularOrganism make_default_seed_organism() {
        CellularOrganism org;
        // 感觉受体
        org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -30.0f, 0.0f});
        org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  30.0f, 0.0f});
        // 内部代谢
        org.cells.push_back({2, CellType::OP_INTEGRAL, 0.1, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});
        // 动作效应器
        org.cells.push_back({3, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, -30.0f, 0.0f});
        org.cells.push_back({4, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f,  30.0f, 0.0f});

        org.synapses.push_back({0, 2, 0, 0.5, true, 50.0f, -1.0f});
        org.synapses.push_back({1, 2, 1, 0.5, true, 50.0f, -1.0f});
        org.synapses.push_back({2, 3, 0, 0.8, true, 50.0f, -1.0f});
        org.synapses.push_back({2, 4, 0, -0.8, true, 50.0f, -1.0f});

        return org;
    }
};

} // namespace kun

#pragma once

#include "kun/cellular/evolvable_task.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iostream>

namespace kun {

/**
 * @brief 经典连续倒立摆控制任务 (CartPoleTask) — 遵循 EvolvableTask 接口
 * 物理动力学状态: [x (小车位置), x_dot (小车速度), theta (摆杆夹角 rad), theta_dot (角速度)]
 */
class CartPoleTask : public EvolvableTask {
public:
    explicit CartPoleTask(int max_steps = 200, uint32_t seed = 42)
        : max_steps_(max_steps), rng_(seed) {
        reset(seed);
    }

    const char* name() const override { return "CartPole-ContinuousPhysics"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 2; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        x_ = dist(rng_);
        x_dot_ = dist(rng_);
        theta_ = dist(rng_);
        theta_dot_ = dist(rng_);
        step_count_ = 0;
    }

    std::vector<float> current_observation() const override {
        // 归一化到 [-1, 1] 区间便于神经细胞激活
        return {
            std::clamp(x_ / 2.4f, -1.0f, 1.0f),
            std::clamp(x_dot_ / 3.0f, -1.0f, 1.0f),
            std::clamp(theta_ / 0.2094f, -1.0f, 1.0f), // 12度
            std::clamp(theta_dot_ / 3.0f, -1.0f, 1.0f)
        };
    }

    StepResult step(int action) override {
        step_count_++;

        // 物理动力学常数
        const float gravity = 9.8f;
        const float masscart = 1.0f;
        const float masspole = 0.1f;
        const float total_mass = masscart + masspole;
        const float length = 0.5f; // 半杆长
        const float polemass_length = masspole * length;
        const float force = (action == 1) ? 10.0f : -10.0f;
        const float tau = 0.02f; // 20ms 时间步长

        float costheta = std::cos(theta_);
        float sintheta = std::sin(theta_);

        float temp = (force + polemass_length * theta_dot_ * theta_dot_ * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) /
                         (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        // 半隐式欧拉积分
        x_ += tau * x_dot_;
        x_dot_ += tau * xacc;
        theta_ += tau * theta_dot_;
        theta_dot_ += tau * thetaacc;

        // 终止条件: 摆角超 12度 (0.2094 rad) 或 小车出界 (> 2.4m)
        bool failed = (std::abs(x_) > 2.4f || std::abs(theta_) > 0.2094f);
        bool timeout = (step_count_ >= max_steps_);

        StepResult res;
        res.obs = current_observation();
        res.done = (failed || timeout);
        res.success = (!failed && timeout);
        res.steps = step_count_;
        res.reward = failed ? 0.0 : (1.0 + (0.2094f - std::abs(theta_)) * 2.0); // 坚挺奖励 + 竖直姿态奖励
        res.min_dist_to_goal = std::abs(theta_);

        return res;
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        int act = (acts.positive_action >= acts.negative_action) ? 1 : 0;
        return step(act);
    }

    double current_fitness() const override {
        return static_cast<double>(step_count_) + (0.2094 - std::min(0.2094, (double)std::abs(theta_))) * 50.0;
    }

private:
    int max_steps_{200};
    int step_count_{0};
    float x_{0.0f};
    float x_dot_{0.0f};
    float theta_{0.0f};
    float theta_dot_{0.0f};
    std::mt19937 rng_;
};

/**
 * @brief 离散符号与序列规则学习任务 (SequenceRuleTask) — 遵循 EvolvableTask 接口
 * 规则目标: 识别时间序列异或奇偶校验 (XOR Parity Rule) 与动态周期转移
 */
class SequenceRuleTask : public EvolvableTask {
public:
    explicit SequenceRuleTask(int seq_length = 50, uint32_t seed = 42)
        : seq_length_(seq_length), rng_(seed) {
        reset(seed);
    }

    const char* name() const override { return "Sequence-SymbolicRuleLearning"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 2; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        step_idx_ = 0;
        correct_count_ = 0;

        // 生成确定性二值符号序列
        seq_.resize(seq_length_ + 4);
        std::bernoulli_distribution dist(0.5);
        for (size_t i = 0; i < seq_.size(); ++i) {
            seq_[i] = dist(rng_) ? 1.0f : 0.0f;
        }
    }

    std::vector<float> current_observation() const override {
        size_t idx = static_cast<size_t>(step_idx_);
        float b0 = (idx < seq_.size()) ? seq_[idx] : 0.0f;
        float b1 = (idx + 1 < seq_.size()) ? seq_[idx + 1] : 0.0f;
        float b2 = (idx + 2 < seq_.size()) ? seq_[idx + 2] : 0.0f;
        float progress = static_cast<float>(step_idx_) / static_cast<float>(seq_length_);
        return {b0, b1, b2, progress};
    }

    StepResult step(int action) override {
        // 目标预测规则: Rule(b0, b1, b2) = (b0 ^ b1) == b2 ? 1 : 0 (异或奇偶性)
        float b0 = seq_[step_idx_];
        float b1 = seq_[step_idx_ + 1];
        float b2 = seq_[step_idx_ + 2];
        int target = ((static_cast<int>(b0) ^ static_cast<int>(b1)) == static_cast<int>(b2)) ? 1 : 0;

        bool is_correct = (action == target);
        if (is_correct) correct_count_++;
        step_idx_++;

        bool done = (step_idx_ >= seq_length_);
        StepResult res;
        res.obs = current_observation();
        res.reward = is_correct ? 2.0 : -1.0;
        res.done = done;
        res.success = (correct_count_ >= static_cast<int>(seq_length_ * 0.85));
        res.steps = step_idx_;
        res.min_dist_to_goal = static_cast<double>(seq_length_ - correct_count_);

        return res;
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        int act = (acts.positive_action >= acts.negative_action) ? 1 : 0;
        return step(act);
    }

    double current_fitness() const override {
        double accuracy = static_cast<double>(correct_count_) / std::max(1.0, static_cast<double>(seq_length_));
        return accuracy * 100.0;
    }

private:
    int seq_length_{50};
    int step_idx_{0};
    int correct_count_{0};
    std::vector<float> seq_;
    std::mt19937 rng_;
};

/**
 * @brief 斗地主非完全信息不确定性对抗博弈环境 (DouDiZhuCardGameTask)
 * 验证: 3 人博弈（1 地主 vs 2 农民）、记牌（延迟记忆）、压牌与让牌决策（门控逻辑）
 */
class DouDiZhuCardGameTask : public EvolvableTask {
public:
    explicit DouDiZhuCardGameTask(int max_rounds = 40, uint32_t seed = 42)
        : max_rounds_(max_rounds), rng_(seed) {
        reset(seed);
    }

    const char* name() const override { return "DouDiZhu-ImperfectInfoGame"; }
    size_t obs_dim() const override { return 4; } // [己方手牌点数均值, 己方剩余张数比率, 场上上家牌力, 历史出牌强度]
    size_t act_dim() const override { return 2; } // 0: 过 (Pass), 1: 出牌 (Play)

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        agent_hand_strength_ = std::uniform_real_distribution<float>(0.3f, 0.9f)(rng_);
        agent_cards_left_ = 17;
        opp_cards_left_ = 17;
        table_card_strength_ = 0.0f;
        played_history_intensity_ = 0.0f;
        round_count_ = 0;
        agent_won_ = false;
        total_wins_ = 0;
        games_played_ = 0;
    }

    std::vector<float> current_observation() const override {
        return {
            agent_hand_strength_,
            static_cast<float>(agent_cards_left_) / 20.0f,
            table_card_strength_,
            played_history_intensity_
        };
    }

    StepResult step(int action) override {
        round_count_++;
        double reward = 0.0;
        bool done = false;

        // action: 0 = Pass, 1 = Play
        if (action == 1) { // 尝试出牌压制
            if (table_card_strength_ == 0.0f || agent_hand_strength_ >= table_card_strength_) {
                // 出牌成功
                float play_val = std::min(agent_hand_strength_, table_card_strength_ + 0.15f);
                table_card_strength_ = play_val;
                agent_cards_left_ -= std::uniform_int_distribution<int>(1, 2)(rng_);
                played_history_intensity_ += play_val * 0.2f;
                reward += 1.5;
            } else {
                // 违规越级出牌或牌力不足惩罚
                reward -= 1.0;
            }
        } else { // Pass
            if (table_card_strength_ > 0.0f && agent_hand_strength_ < table_card_strength_) {
                // 明智让牌/过牌，节省大牌
                reward += 0.8;
            } else if (table_card_strength_ == 0.0f) {
                // 首发随意过牌罚分
                reward -= 0.5;
            }
        }

        // 模拟对手 (农民联手/地主) 策略反馈
        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) > 0.45f) {
            float opp_play = std::uniform_real_distribution<float>(0.2f, 0.95f)(rng_);
            if (opp_play > table_card_strength_) {
                table_card_strength_ = opp_play;
                opp_cards_left_ -= std::uniform_int_distribution<int>(1, 2)(rng_);
                played_history_intensity_ += opp_play * 0.15f;
            } else {
                table_card_strength_ = 0.0f; // 对手要不起，清台
            }
        } else {
            table_card_strength_ = 0.0f; // 对手过牌，清台
        }

        if (agent_cards_left_ <= 0) {
            agent_won_ = true;
            reward += 10.0;
            done = true;
            total_wins_++;
        } else if (opp_cards_left_ <= 0 || round_count_ >= max_rounds_) {
            agent_won_ = false;
            reward -= 5.0;
            done = true;
        }

        games_played_++;

        StepResult res;
        res.obs = current_observation();
        res.reward = reward;
        res.done = done;
        res.success = agent_won_;
        res.steps = round_count_;
        res.min_dist_to_goal = static_cast<double>(agent_cards_left_);
        return res;
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        int act = (acts.positive_action >= acts.negative_action) ? 1 : 0;
        return step(act);
    }

    double current_fitness() const override {
        double win_rate = static_cast<double>(total_wins_) / std::max(1, games_played_);
        return win_rate * 100.0;
    }

private:
    int max_rounds_{40};
    int round_count_{0};
    float agent_hand_strength_{0.5f};
    int agent_cards_left_{17};
    int opp_cards_left_{17};
    float table_card_strength_{0.0f};
    float played_history_intensity_{0.0f};
    bool agent_won_{false};
    int total_wins_{0};
    int games_played_{0};
    std::mt19937 rng_;
};

/**
 * @brief 智驾极限交互拓扑工况任务 (UnprotectedIntersectionTask)
 * 涵盖：无保护左转博弈、无保护右转穿流、窄路多把掉头（U-Turn）、动态对向车博弈避让
 */
class UnprotectedIntersectionTask : public EvolvableTask {
public:
    enum class ManeuverType : uint8_t {
        UNPROTECTED_LEFT_TURN = 0,  // 无保护左转 (穿行对向车流)
        UNPROTECTED_RIGHT_TURN = 1, // 无保护右转 (汇入直行车流)
        MULTI_POINT_U_TURN = 2      // 窄路多把掉头 (D/R 换挡与极限舵角)
    };

    explicit UnprotectedIntersectionTask(int max_steps = 150, uint32_t seed = 42)
        : max_steps_(max_steps), rng_(seed) {
        reset(seed);
    }

    const char* name() const override { return "UnprotectedIntersection-ComplexManeuver"; }
    size_t obs_dim() const override { return 6; } // [横向偏差 cte, 航向偏差 d_psi, 纵向车速 v, 对向车TTC, 目标曲率 kappa, 剩余机动距离]
    size_t act_dim() const override { return 2; } // [方向盘转角 steer, 纵向加速度 a/油门刹车]

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        step_count_ = 0;
        collision_ = false;
        reached_target_ = false;
        
        // 随机选择当前机动工况
        uint8_t m_val = std::uniform_int_distribution<uint8_t>(0, 2)(rng_);
        maneuver_ = static_cast<ManeuverType>(m_val);

        x_ = 0.0f;
        y_ = 0.0f;
        psi_ = 0.0f;
        v_ = 4.0f;
        cte_ = 0.0f;
        
        if (maneuver_ == ManeuverType::UNPROTECTED_LEFT_TURN) {
            target_psi_ = 1.5708f; // 90度左转
            target_x_ = 25.0f;
            target_y_ = 25.0f;
            oncoming_ttc_ = std::uniform_real_distribution<float>(1.8f, 5.0f)(rng_);
            target_kappa_ = 0.08f;
        } else if (maneuver_ == ManeuverType::UNPROTECTED_RIGHT_TURN) {
            target_psi_ = -1.5708f; // 90度右转
            target_x_ = 15.0f;
            target_y_ = -15.0f;
            oncoming_ttc_ = std::uniform_real_distribution<float>(2.2f, 6.0f)(rng_);
            target_kappa_ = -0.12f;
        } else { // U-Turn
            target_psi_ = 3.14159f; // 180度掉头
            target_x_ = -5.0f;
            target_y_ = 12.0f;
            oncoming_ttc_ = 8.0f;
            target_kappa_ = 0.22f; // 极大曲率
        }
    }

    std::vector<float> current_observation() const override {
        float d_psi = target_psi_ - psi_;
        while (d_psi > 3.14159f) d_psi -= 6.28318f;
        while (d_psi < -3.14159f) d_psi += 6.28318f;

        float dist_rem = std::hypot(target_x_ - x_, target_y_ - y_);
        return {
            std::clamp(cte_ / 5.0f, -1.0f, 1.0f),
            std::clamp(d_psi / 3.14159f, -1.0f, 1.0f),
            std::clamp(v_ / 10.0f, 0.0f, 1.0f),
            std::clamp(oncoming_ttc_ / 6.0f, 0.0f, 1.0f),
            std::clamp(target_kappa_ / 0.25f, -1.0f, 1.0f),
            std::clamp(dist_rem / 40.0f, 0.0f, 1.0f)
        };
    }

    StepResult step(int action) override {
        // 离散映射: 0: 减速让行并向目标打舵, 1: 加速抢越并跟踪轨迹
        float steer = (target_kappa_ > 0) ? 0.35f : -0.35f;
        float accel = (action == 1) ? 1.5f : -2.5f;
        return step_internal(steer, accel);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        float steer = static_cast<float>(acts.positive_action - acts.negative_action);
        float accel = static_cast<float>(acts.positive_action * 2.5 - acts.defensive_action * 4.0);
        return step_internal(steer, accel);
    }

    double current_fitness() const override {
        double fit = reached_target_ ? 100.0 : 0.0;
        fit -= std::hypot(target_x_ - x_, target_y_ - y_) * 1.5;
        if (collision_) fit -= 80.0;
        return std::max(0.0, fit);
    }

private:
    StepResult step_internal(float steer, float accel) {
        step_count_++;
        float dt = 0.1f;

        // 动力学自行车单轨模型积分
        float L = 2.8f;
        v_ = std::clamp(v_ + accel * dt, 0.0f, 12.0f);
        x_ += v_ * std::cos(psi_) * dt;
        y_ += v_ * std::sin(psi_) * dt;
        psi_ += (v_ / L) * std::tan(std::clamp(steer, -0.6f, 0.6f)) * dt;

        // 对向车运动与 TTC 衰减
        oncoming_ttc_ -= dt;
        if (oncoming_ttc_ < 0.4f && oncoming_ttc_ > -0.4f) {
            // 对向车刚好到达冲突点
            if (std::abs(y_ - 10.0f) < 3.0f && v_ > 2.0f) {
                collision_ = true;
            }
        }

        float dist_rem = std::hypot(target_x_ - x_, target_y_ - y_);
        float d_psi = std::abs(target_psi_ - psi_);
        while (d_psi > 3.14159f) d_psi -= 6.28318f;
        d_psi = std::abs(d_psi);

        if (dist_rem < 2.5f && d_psi < 0.35f) {
            reached_target_ = true;
        }

        bool done = reached_target_ || collision_ || (step_count_ >= max_steps_);
        double reward = 0.0;
        if (reached_target_) reward += 50.0;
        if (collision_) reward -= 100.0;
        reward -= dist_rem * 0.1;
        reward -= d_psi * 0.2;

        StepResult res;
        res.obs = current_observation();
        res.reward = reward;
        res.done = done;
        res.success = reached_target_ && !collision_;
        res.steps = step_count_;
        res.min_dist_to_goal = dist_rem;
        return res;
    }

    int max_steps_{150};
    int step_count_{0};
    ManeuverType maneuver_{ManeuverType::UNPROTECTED_LEFT_TURN};
    float x_{0.0f}, y_{0.0f}, psi_{0.0f}, v_{4.0f}, cte_{0.0f};
    float target_x_{25.0f}, target_y_{25.0f}, target_psi_{1.5708f}, target_kappa_{0.08f};
    float oncoming_ttc_{3.0f};
    bool collision_{false};
    bool reached_target_{false};
    std::mt19937 rng_;
};

        StepResult res;
        res.obs = current_observation();
        res.reward = reward;
        res.done = done;
        res.success = agent_won_;
        res.steps = round_count_;
        res.min_dist_to_goal = static_cast<double>(std::max(0, agent_cards_left_));
        return res;
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        int act = (acts.positive_action >= acts.negative_action) ? 1 : 0;
        return step(act);
    }

    double current_fitness() const override {
        double win_rate = (games_played_ > 0) ? (static_cast<double>(total_wins_) / games_played_) : 0.0;
        double card_progress = static_cast<double>(17 - std::max(0, agent_cards_left_)) / 17.0;
        return (win_rate * 60.0) + (card_progress * 40.0);
    }

private:
    int max_rounds_{40};
    int round_count_{0};
    float agent_hand_strength_{0.5f};
    int agent_cards_left_{17};
    int opp_cards_left_{17};
    float table_card_strength_{0.0f};
    float played_history_intensity_{0.0f};
    bool agent_won_{false};
    int total_wins_{0};
    int games_played_{0};
    std::mt19937 rng_;
};

/**
 * @brief 跨域 Few-Shot 迁移学习加速比评测器 (CrossDomainTransferEvaluator)
 * 验证: 预演化母体网络迁移到新任务的收敛代数 vs 从零随机初始化的收敛代数
 * 门禁: 迁移加速比 (Transfer Speedup Ratio) > 1.5x
 */
class CrossDomainTransferEvaluator {
public:
    struct TransferReport {
        std::string source_domain;
        std::string target_domain;
        int scratch_convergence_generations{0};
        int transfer_convergence_generations{0};
        double acceleration_ratio{0.0};
        bool passes_m3_gate{false};
        std::string summary;
    };

    static TransferReport evaluate_transfer(
        EvolvableTask& target_task,
        const std::vector<uint32_t>& task_seeds,
        CellularOrganism pre_adapted_org,
        double target_fitness_threshold = 40.0,
        int max_generations = 25
    ) {
        TransferReport report;
        report.source_domain = "Maze-SpatialNavigation";
        report.target_domain = target_task.name();

        // 1. 从零随机初始化种群 (Scratch Baseline - 极小随机未分化胚胎母体)
        MorphogeneticEvolutionEngine scratch_engine(16, 42, SeedInitMode::MINIMAL_RANDOM_GRAPH);
        int scratch_gen = max_generations;

        for (int g = 1; g <= max_generations; ++g) {
            auto& pop = scratch_engine.population();
            double best_fit = -1e9;
            for (auto& org : pop) {
                auto m = target_task.evaluate_organism(org, task_seeds, 80, true);
                org.fitness_score = m.mean_fitness;
                if (m.mean_fitness > best_fit) best_fit = m.mean_fitness;
            }
            if (best_fit >= target_fitness_threshold) {
                scratch_gen = g;
                break;
            }
            scratch_engine.evolve_generation();
        }

        // 2. 跨域迁移种群 (Transfer via Pre-adapted Organism Seed)
        MorphogeneticEvolutionEngine transfer_engine(16, 42, SeedInitMode::HANDCRAFTED_PROGENITOR);
        // 将预演化的母体拓扑注入整个种群作为母体先验
        for (size_t i = 0; i < transfer_engine.population().size(); ++i) {
            transfer_engine.population()[i] = pre_adapted_org;
            if (i > 0) {
                transfer_engine.mutate(transfer_engine.population()[i]);
            }
        }

        int transfer_gen = max_generations;
        for (int g = 1; g <= max_generations; ++g) {
            auto& pop = transfer_engine.population();
            double best_fit = -1e9;
            for (auto& org : pop) {
                auto m = target_task.evaluate_organism(org, task_seeds, 80, true);
                org.fitness_score = m.mean_fitness;
                if (m.mean_fitness > best_fit) best_fit = m.mean_fitness;
            }
            if (best_fit >= target_fitness_threshold) {
                transfer_gen = g;
                break;
            }
            transfer_engine.evolve_generation();
        }

        report.scratch_convergence_generations = scratch_gen;
        report.transfer_convergence_generations = transfer_gen;
        report.acceleration_ratio = static_cast<double>(scratch_gen) / static_cast<double>(std::max(1, transfer_gen));
        report.passes_m3_gate = (report.acceleration_ratio >= 1.50 || (transfer_gen == 1 && scratch_gen >= 2));

        std::ostringstream oss;
        oss << "Cross-Domain Transfer [" << report.source_domain << " -> " << report.target_domain
            << "]: Scratch Generations=" << scratch_gen << ", Transfer Generations=" << transfer_gen
            << ", Speedup Ratio=" << std::fixed << std::setprecision(2) << report.acceleration_ratio
            << "x. M3 Gate=" << (report.passes_m3_gate ? "PASSED (>= 1.5x)" : "FAILED (< 1.5x)");
        report.summary = oss.str();

        return report;
    }
};

} // namespace kun

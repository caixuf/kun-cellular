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
 * @brief 斗地主标准54张实体牌非完全信息3人博弈环境 (DouDiZhuCardGameTask)
 * 架构规范：
 * 1. 真实 54 张牌副 (13 点数 * 4 + 小王 + 大王)，3 人按规发牌 (地主 20 张, 农民各 17 张)；
 * 2. 真实合法牌型引擎 (单牌 SOLO、对子 PAIR、炸弹 BOMB/王炸)，严格比较大小与出牌压制；
 * 3. 启发式农民协同博弈对手 (规避互压大牌、合规出牌、合法让牌)，杜绝随机数伪博弈。
 */
class DouDiZhuCardGameTask : public EvolvableTask {
public:
    enum TrickType { TRICK_NONE = 0, TRICK_SOLO = 1, TRICK_PAIR = 2, TRICK_BOMB = 3, TRICK_ROCKET = 4 };
    
    struct Trick {
        TrickType type{TRICK_NONE};
        int rank{-1};
        int owner{-1};
    };

    /**
     * @brief 15 维残局全量记牌器晶格 (Card Counting Lattice)
     * 准确追踪场上 15 个点数的余量分布（特别是 2 与双王、断张炸弹威胁），赋能残局精准收割
     */
    struct CardCountingLattice {
        std::array<int, 15> unseen{};      // 对手手中尚未打出的未知余量
        std::array<int, 15> played{};      // 全场已公开打出的牌张数

        void reset(const int my_hand[15]) {
            for (int r = 0; r < 15; ++r) {
                int total = (r < 13) ? 4 : 1;
                unseen[r] = total - my_hand[r];
                played[r] = 0;
            }
        }

        void record_play(int r, int count, int player) {
            played[r] += count;
            if (player != 0) {
                unseen[r] = std::max(0, unseen[r] - count);
            }
        }

        int unseen_2s() const { return unseen[12]; }
        int unseen_small_joker() const { return unseen[13]; }
        int unseen_big_joker() const { return unseen[14]; }
        int unseen_high_cards() const { return unseen[12] + unseen[13] + unseen[14]; }

        bool has_rocket_threat() const {
            return unseen[13] > 0 && unseen[14] > 0;
        }

        int unseen_bomb_threats() const {
            int bombs = 0;
            if (has_rocket_threat()) bombs++;
            for (int r = 0; r < 13; ++r) {
                if (unseen[r] == 4) bombs++;
            }
            return bombs;
        }

        int highest_unseen_rank() const {
            for (int r = 14; r >= 0; --r) {
                if (unseen[r] > 0) return r;
            }
            return -1;
        }
    };

    using NeuralBidEvaluator = std::function<bool(const std::vector<float>& hand_obs, double opp_max_score)>;

    explicit DouDiZhuCardGameTask(int max_rounds = 40, uint32_t seed = 42, double bidding_threshold = 16.0,
                                  NeuralBidEvaluator neural_bid = nullptr)
        : max_rounds_(max_rounds), bidding_threshold_(bidding_threshold),
          neural_bid_evaluator_(std::move(neural_bid)), rng_(seed) {
        reset(seed);
    }

    void set_neural_bid_evaluator(NeuralBidEvaluator eval) { neural_bid_evaluator_ = std::move(eval); }
    const NeuralBidEvaluator& neural_bid_evaluator() const { return neural_bid_evaluator_; }

    std::vector<float> initial_hand_observation() const {
        std::vector<float> obs(32, 0.0f);
        for (int r = 0; r < 15; ++r) {
            float max_c = (r < 13) ? 4.0f : 1.0f;
            obs[r] = std::clamp(static_cast<float>(hands_[0][r]) / max_c, 0.0f, 1.0f);
        }
        for (int r = 8; r < 15; ++r) {
            float max_unseen = (r < 13) ? 4.0f : 1.0f;
            int unseen_cnt = ((r < 13) ? 4 : 1) - hands_[0][r];
            obs[15 + (r - 8)] = std::clamp(static_cast<float>(unseen_cnt) / max_unseen, 0.0f, 1.0f);
        }
        obs[28] = 0.0f;
        obs[29] = 17.0f / 20.0f;
        obs[30] = 17.0f / 20.0f;
        obs[31] = 17.0f / 20.0f;
        return obs;
    }

    const char* name() const override { return "DouDiZhu-ImperfectInfoGame"; }
    size_t obs_dim() const override { return 32; } // 32 维高维全息感知晶格 (15手牌 + 7高牌记牌 + 6台面上下文 + 4角色余牌)
    size_t act_dim() const override { return 3; } // 0: 让牌(Pass/Hold), 1: 合规跟牌(Clean Follow), 2: 强行夺权(Power Seize)

    // 叫地主起手势能估值 (Bidding Filter)
    double evaluate_hand_potential(int p) const {
        double score = 0.0;
        int singles_below_9 = 0;
        bool has_bj = (hands_[p][13] > 0);
        bool has_rj = (hands_[p][14] > 0);
        if (has_bj && has_rj) score += 9.0; // 双王火箭
        else {
            if (has_rj) score += 3.8; // 大王
            if (has_bj) score += 2.8; // 小王
        }
        score += hands_[p][12] * 3.0; // 2 (核心控场大牌)
        score += hands_[p][11] * 1.1; // A
        score += hands_[p][10] * 0.5; // K

        for (int r = 0; r < 13; ++r) {
            if (hands_[p][r] == 4) score += 5.0; // 炸弹
            else if (hands_[p][r] == 3) score += (r >= 8 ? 1.8 : 0.9);
            else if (hands_[p][r] == 2 && r >= 9) score += 0.7;
            else if (hands_[p][r] == 1 && r <= 6) singles_below_9++;
        }
        score -= singles_below_9 * 0.5; // 散牌过多折损
        return score;
    }

    bool has_rocket(int p) const {
        return hands_[p][13] > 0 && hands_[p][14] > 0;
    }

    void play_rocket(int p) {
        hands_[p][13]--;
        hands_[p][14]--;
        cards_left_[p] -= 2;
        table_trick_ = Trick{TRICK_ROCKET, 14, p};
        record_card_played(13, 1, p);
        record_card_played(14, 1, p);
    }

    void record_card_played(int r, int count, int p) {
        lattice_.record_play(r, count, p);
        if (r >= 12) high_cards_played_ += count;
    }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);

        // 1. 初始化标准 54 张牌副 (13 点数 * 4 + 小王 + 大王)
        std::vector<int> deck;
        deck.reserve(54);
        for (int r = 0; r < 13; ++r) {
            for (int k = 0; k < 4; ++k) deck.push_back(r);
        }
        deck.push_back(13); // 小王
        deck.push_back(14); // 大王

        std::shuffle(deck.begin(), deck.end(), rng_);

        // 2. 初始发牌：3 人各 17 张，留 3 张底牌 (51..53)
        std::memset(hands_, 0, sizeof(hands_));
        for (int i = 0; i < 17; ++i) hands_[0][deck[i]]++;
        for (int i = 17; i < 34; ++i) hands_[1][deck[i]]++;
        for (int i = 34; i < 51; ++i) hands_[2][deck[i]]++;

        // 3. 叫地主门禁决策 (Bidding Filter)
        double score0 = evaluate_hand_potential(0);
        double score1 = evaluate_hand_potential(1);
        double score2 = evaluate_hand_potential(2);

        // 烂牌不盲叫地主，只有手牌大牌/成型牌势能充足时才叫地主拿 3 张底牌
        // 精准门禁：若配置神经叫牌核，则由端到端全息感知晶格自主决断；否则使用真实专家门限 (16.0)
        bool p0_bids = false;
        if (neural_bid_evaluator_) {
            auto hand_obs = initial_hand_observation();
            p0_bids = neural_bid_evaluator_(hand_obs, std::max(score1, score2));
        } else {
            p0_bids = (score0 >= bidding_threshold_ && score0 >= score1 && score0 >= score2);
        }
        if (p0_bids) {
            landlord_ = 0;
            role_ = 1; // 地主
            for (int i = 51; i < 54; ++i) hands_[0][deck[i]]++;
            cards_left_[0] = 20;
            cards_left_[1] = 17;
            cards_left_[2] = 17;
        } else {
            // 退居农民协同作战
            role_ = 0; // 农民
            landlord_ = (score1 >= score2) ? 1 : 2;
            for (int i = 51; i < 54; ++i) hands_[landlord_][deck[i]]++;
            cards_left_[landlord_] = 20;
            cards_left_[0] = 17;
            cards_left_[3 - landlord_] = 17;
        }

        // 4. 初始化 15 维记牌晶格
        lattice_.reset(hands_[0]);
        high_cards_played_ = 0;
        table_trick_ = Trick{TRICK_NONE, -1, -1};
        pass_count_ = 0;
        round_count_ = 0;
        agent_won_ = false;
        total_wins_ = 0;
        games_played_ = 0;
        current_turn_ = landlord_; // 地主拿底牌后先手出牌

        // 5. 若智能体不是地主，推进回合至智能体行动轮次 (current_turn_ == 0)
        while (current_turn_ != 0) {
            play_opponent_turn(current_turn_);
            if (cards_left_[1] <= 0 || cards_left_[2] <= 0) break;
        }
    }

    std::vector<float> current_observation() const override {
        std::vector<float> obs;
        obs.reserve(32);

        // 1. 15 dims: 己方手牌在 15 个点数等级上的分布 (3 到大王)
        // 点数 0..12 (3..2) 每张牌最多 4 张 (除以 4.0 归一化); 13 (小王), 14 (大王) 最多 1 张 (除以 1.0 归一化)
        for (int r = 0; r < 15; ++r) {
            float max_c = (r < 13) ? 4.0f : 1.0f;
            obs.push_back(std::clamp(static_cast<float>(hands_[0][r]) / max_c, 0.0f, 1.0f));
        }

        // 2. 7 dims: 记牌器中尚未打出的未知高牌分布 (J, Q, K, A, 2, 小王, 大王)
        // 点数 8..14: J(8), Q(9), K(10), A(11), 2(12), BJ(13), RJ(14)
        for (int r = 8; r < 15; ++r) {
            float max_unseen = (r < 13) ? 4.0f : 1.0f;
            obs.push_back(std::clamp(static_cast<float>(lattice_.unseen[r]) / max_unseen, 0.0f, 1.0f));
        }

        // 3. 6 dims: 台面牌型上下文 (牌型类别, 牌力点数, 牌型张数长度, 己方控台, 盟友控台, 地主控台)
        float trick_type = static_cast<float>(table_trick_.type) / 4.0f;
        float trick_rank = (table_trick_.type != TRICK_NONE && table_trick_.rank >= 0)
                               ? (static_cast<float>(table_trick_.rank) / 14.0f) : 0.0f;
        float trick_len = 0.0f;
        if (table_trick_.type == TRICK_SOLO) trick_len = 1.0f / 4.0f;
        else if (table_trick_.type == TRICK_PAIR) trick_len = 2.0f / 4.0f;
        else if (table_trick_.type == TRICK_BOMB) trick_len = 4.0f / 4.0f;
        else if (table_trick_.type == TRICK_ROCKET) trick_len = 2.0f / 4.0f;

        int teammate = (role_ == 1) ? -1 : (landlord_ == 1 ? 2 : 1);
        float owner_self = (table_trick_.type != TRICK_NONE && table_trick_.owner == 0) ? 1.0f : 0.0f;
        float owner_partner = (table_trick_.type != TRICK_NONE && role_ == 0 && table_trick_.owner == teammate) ? 1.0f : 0.0f;
        float owner_landlord = (table_trick_.type != TRICK_NONE && table_trick_.owner == landlord_) ? 1.0f : 0.0f;

        obs.push_back(std::clamp(trick_type, 0.0f, 1.0f));
        obs.push_back(std::clamp(trick_rank, 0.0f, 1.0f));
        obs.push_back(std::clamp(trick_len, 0.0f, 1.0f));
        obs.push_back(owner_self);
        obs.push_back(owner_partner);
        obs.push_back(owner_landlord);

        // 4. 4 dims: 玩家角色与各方余牌 (地主/农民角色, 己方余牌比率, 盟友余牌比率, 地主余牌比率)
        float player_role = (role_ == 1) ? 1.0f : 0.0f;
        float p0_cards = std::clamp(static_cast<float>(cards_left_[0]) / 20.0f, 0.0f, 1.0f);
        float partner_cards = (role_ == 0 && teammate >= 0) ? std::clamp(static_cast<float>(cards_left_[teammate]) / 20.0f, 0.0f, 1.0f) : 0.0f;
        float landlord_cards = std::clamp(static_cast<float>(cards_left_[landlord_]) / 20.0f, 0.0f, 1.0f);

        obs.push_back(player_role);
        obs.push_back(p0_cards);
        obs.push_back(partner_cards);
        obs.push_back(landlord_cards);

        return obs;
    }

    void play_opponent_turn(int p) {
        if (cards_left_[p] <= 0) return;

        bool is_landlord = (p == landlord_);
        int teammate = is_landlord ? -1 : (3 - landlord_ - p);
        int landlord_cards = is_landlord ? 0 : cards_left_[landlord_];

        bool played = false;

        if (table_trick_.type == TRICK_NONE) {
            // 0. 残局冲刺: 若自身剩 <= 2 张牌且能直接出完，立即终局获胜
            if (cards_left_[p] <= 2) {
                if (has_rocket(p)) {
                    play_rocket(p);
                    played = true;
                } else if (cards_left_[p] == 2) {
                    for (int r = 0; r < 13; ++r) {
                        if (hands_[p][r] == 2) {
                            hands_[p][r] -= 2;
                            cards_left_[p] -= 2;
                            table_trick_ = Trick{TRICK_PAIR, r, p};
                            record_card_played(r, 2, p);
                            played = true;
                            break;
                        }
                    }
                }
            }

            // 自由出牌: 农民协作，若盟友只剩 <= 2 张，喂送小牌保送
            if (!played && !is_landlord && teammate >= 0 && cards_left_[teammate] <= 2) {
                if (cards_left_[teammate] == 1) {
                    for (int r = 0; r < 15; ++r) {
                        if (hands_[p][r] >= 1) {
                            if ((r == 13 || r == 14) && has_rocket(p)) continue;
                            hands_[p][r]--;
                            cards_left_[p]--;
                            table_trick_ = Trick{TRICK_SOLO, r, p};
                            record_card_played(r, 1, p);
                            played = true;
                            break;
                        }
                    }
                } else if (cards_left_[teammate] == 2) {
                    for (int r = 0; r < 13; ++r) {
                        if (hands_[p][r] >= 2) {
                            hands_[p][r] -= 2;
                            cards_left_[p] -= 2;
                            table_trick_ = Trick{TRICK_PAIR, r, p};
                            record_card_played(r, 2, p);
                            played = true;
                            break;
                        }
                    }
                }
            }

            // 防守策略: 地主只剩 1 张牌时, 农民绝不出小单牌, 必出对子或大单牌
            if (!played && !is_landlord && landlord_cards == 1) {
                for (int r = 0; r < 13; ++r) {
                    if (hands_[p][r] >= 2 && hands_[p][r] < 4) {
                        hands_[p][r] -= 2;
                        cards_left_[p] -= 2;
                        table_trick_ = Trick{TRICK_PAIR, r, p};
                        record_card_played(r, 2, p);
                        played = true;
                        break;
                    }
                }
                if (!played) {
                    for (int r = 14; r >= 0; --r) {
                        if (hands_[p][r] >= 1 && hands_[p][r] < 4) {
                            if ((r == 13 || r == 14) && has_rocket(p)) continue;
                            hands_[p][r]--;
                            cards_left_[p]--;
                            table_trick_ = Trick{TRICK_SOLO, r, p};
                            record_card_played(r, 1, p);
                            played = true;
                            break;
                        }
                    }
                }
            }

            // 1. 优先出非炸弹对子
            if (!played) {
                for (int r = 0; r < 13; ++r) {
                    if (hands_[p][r] >= 2 && hands_[p][r] < 4) {
                        hands_[p][r] -= 2;
                        cards_left_[p] -= 2;
                        table_trick_ = Trick{TRICK_PAIR, r, p};
                        record_card_played(r, 2, p);
                        played = true;
                        break;
                    }
                }
            }

            // 2. 出非炸弹孤张单牌 (保全王炸)
            if (!played) {
                for (int r = 0; r < 15; ++r) {
                    if (hands_[p][r] == 1) {
                        if ((r == 13 || r == 14) && has_rocket(p)) continue;
                        hands_[p][r]--;
                        cards_left_[p]--;
                        table_trick_ = Trick{TRICK_SOLO, r, p};
                        record_card_played(r, 1, p);
                        played = true;
                        break;
                    }
                }
            }

            // 3. 出非炸弹单牌 (保全王炸)
            if (!played) {
                for (int r = 0; r < 15; ++r) {
                    if (hands_[p][r] >= 1 && hands_[p][r] < 4) {
                        if ((r == 13 || r == 14) && has_rocket(p)) continue;
                        hands_[p][r]--;
                        cards_left_[p]--;
                        table_trick_ = Trick{TRICK_SOLO, r, p};
                        record_card_played(r, 1, p);
                        played = true;
                        break;
                    }
                }
            }

            // 4. 若只剩王炸，直接打出王炸
            if (!played && has_rocket(p)) {
                play_rocket(p);
                played = true;
            }

            // 5. 兜底任意单牌
            if (!played) {
                for (int r = 0; r < 15; ++r) {
                    if (hands_[p][r] >= 1) {
                        hands_[p][r]--;
                        cards_left_[p]--;
                        table_trick_ = Trick{TRICK_SOLO, r, p};
                        record_card_played(r, 1, p);
                        played = true;
                        break;
                    }
                }
            }

            pass_count_ = 0;
            current_turn_ = (p + 1) % 3;
        } else {
            // 台面已有牌型，按规跟牌或过牌
            if (table_trick_.type == TRICK_ROCKET) {
                // 王炸为天牌，任何点数与炸弹均无法压制，必须过牌
                played = false;
                pass_count_++;
            } else if (!is_landlord && table_trick_.owner == teammate) {
                // 盟友控场: 绝不压死盟友的大牌
                bool landlord_passed = ((table_trick_.owner + 1) % 3 == landlord_);
                bool teammate_ready = (cards_left_[teammate] <= 2);
                bool high_block = (table_trick_.rank >= 8);

                if (landlord_passed || teammate_ready || high_block) {
                    played = false;
                    pass_count_++;
                } else {
                    // 地主在后且盟友牌力偏低(<8)：顶家接牌拦截地主
                    if (table_trick_.type == TRICK_SOLO) {
                        for (int r = std::max(8, table_trick_.rank + 1); r < 15; ++r) {
                            if (hands_[p][r] >= 1 && hands_[p][r] < 4) {
                                if ((r == 13 || r == 14) && has_rocket(p)) continue;
                                hands_[p][r]--;
                                cards_left_[p]--;
                                table_trick_ = Trick{TRICK_SOLO, r, p};
                                record_card_played(r, 1, p);
                                played = true;
                                break;
                            }
                        }
                    } else if (table_trick_.type == TRICK_PAIR) {
                        for (int r = std::max(8, table_trick_.rank + 1); r < 13; ++r) {
                            if (hands_[p][r] >= 2 && hands_[p][r] < 4) {
                                hands_[p][r] -= 2;
                                cards_left_[p] -= 2;
                                table_trick_ = Trick{TRICK_PAIR, r, p};
                                record_card_played(r, 2, p);
                                played = true;
                                break;
                            }
                        }
                    }
                    if (played) pass_count_ = 0;
                    else pass_count_++;
                }
            } else {
                int enemy_min_cards = 20;
                if (is_landlord) {
                    for (int opp = 0; opp < 3; ++opp) {
                        if (opp != p) enemy_min_cards = std::min(enemy_min_cards, cards_left_[opp]);
                    }
                } else {
                    enemy_min_cards = landlord_cards;
                }
                bool urgent = (!is_landlord && table_trick_.rank >= 12) || (enemy_min_cards <= 4) || (cards_left_[p] <= 4);

                if (table_trick_.type == TRICK_SOLO) {
                    // 1. 优先孤张
                    for (int r = table_trick_.rank + 1; r < 15; ++r) {
                        if (hands_[p][r] == 1) {
                            if ((r == 13 || r == 14) && has_rocket(p)) continue;
                            hands_[p][r]--;
                            cards_left_[p]--;
                            table_trick_ = Trick{TRICK_SOLO, r, p};
                            record_card_played(r, 1, p);
                            played = true;
                            break;
                        }
                    }
                    // 2. 对子/三张中的单牌
                    if (!played) {
                        for (int r = table_trick_.rank + 1; r < 15; ++r) {
                            if (hands_[p][r] >= 2 && hands_[p][r] < 4) {
                                hands_[p][r]--;
                                cards_left_[p]--;
                                table_trick_ = Trick{TRICK_SOLO, r, p};
                                record_card_played(r, 1, p);
                                played = true;
                                break;
                            }
                        }
                    }
                    // 3. 紧要关头炸弹或王炸
                    if (!played && urgent) {
                        for (int r = 0; r < 13; ++r) {
                            if (hands_[p][r] == 4) {
                                hands_[p][r] -= 4;
                                cards_left_[p] -= 4;
                                table_trick_ = Trick{TRICK_BOMB, r, p};
                                record_card_played(r, 4, p);
                                played = true;
                                break;
                            }
                        }
                        if (!played && has_rocket(p)) {
                            play_rocket(p);
                            played = true;
                        }
                    }
                } else if (table_trick_.type == TRICK_PAIR) {
                    for (int r = table_trick_.rank + 1; r < 13; ++r) {
                        if (hands_[p][r] >= 2 && hands_[p][r] < 4) {
                            hands_[p][r] -= 2;
                            cards_left_[p] -= 2;
                            table_trick_ = Trick{TRICK_PAIR, r, p};
                            record_card_played(r, 2, p);
                            played = true;
                            break;
                        }
                    }
                    if (!played && urgent) {
                        for (int r = 0; r < 13; ++r) {
                            if (hands_[p][r] == 4) {
                                hands_[p][r] -= 4;
                                cards_left_[p] -= 4;
                                table_trick_ = Trick{TRICK_BOMB, r, p};
                                record_card_played(r, 4, p);
                                played = true;
                                break;
                            }
                        }
                        if (!played && has_rocket(p)) {
                            play_rocket(p);
                            played = true;
                        }
                    }
                } else if (table_trick_.type == TRICK_BOMB) {
                    for (int r = table_trick_.rank + 1; r < 13; ++r) {
                        if (hands_[p][r] == 4) {
                            hands_[p][r] -= 4;
                            cards_left_[p] -= 4;
                            table_trick_ = Trick{TRICK_BOMB, r, p};
                            record_card_played(r, 4, p);
                            played = true;
                            break;
                        }
                    }
                    if (!played && has_rocket(p)) {
                        play_rocket(p);
                        played = true;
                    }
                }

                if (played) {
                    pass_count_ = 0;
                } else {
                    pass_count_++;
                }
            }

            if (pass_count_ == 2) {
                current_turn_ = table_trick_.owner;
                table_trick_ = Trick{TRICK_NONE, -1, -1};
                pass_count_ = 0;
            } else {
                current_turn_ = (p + 1) % 3;
            }
        }
    }

    StepResult step(int action) override {
        round_count_++;
        double reward = 0.0;
        bool done = false;

        bool is_landlord = (role_ == 1);
        int teammate = is_landlord ? -1 : (3 - landlord_);
        int landlord_cards = is_landlord ? 0 : cards_left_[landlord_];
        int enemy_min_cards = 20;
        if (is_landlord) {
            enemy_min_cards = std::min(cards_left_[1], cards_left_[2]);
        } else {
            enemy_min_cards = landlord_cards;
        }

        bool played = false;

        // 智能体三态离散决策: 0=让牌 (Pass/Hold), 1=合规跟牌 (Clean Follow), 2=强行夺权/拆牌突击 (Power Seize)
        if (action == 0) { // 让牌 (Pass / Strategic Hold: 保全手牌结构)
            if (table_trick_.type == TRICK_NONE) {
                // 自由出牌权违规 Pass: 规则禁止过牌, 强制按常规打出合理牌
                int prev_cards = cards_left_[0];
                play_opponent_turn(0);
                if (cards_left_[0] < prev_cards) played = true;
                reward -= 0.5;
            } else if (!is_landlord && table_trick_.owner == teammate) {
                reward += 1.5; // 盟友控场，审慎让牌协助盟友，互不压大牌
                pass_count_++;
                if (pass_count_ == 2) {
                    current_turn_ = table_trick_.owner;
                    table_trick_ = Trick{TRICK_NONE, -1, -1};
                    pass_count_ = 0;
                } else {
                    current_turn_ = 1;
                }
            } else {
                // 对手出牌: 若对手已到终局 (<=2张)，不可随意过牌送死，强制阻击
                if (enemy_min_cards <= 2) {
                    int prev_cards = cards_left_[0];
                    play_opponent_turn(0);
                    if (cards_left_[0] < prev_cards) {
                        played = true;
                        reward += 1.0;
                    }
                } else {
                    // 非终局时，仅在有非拆牌孤张时跟牌，否则保全对子/炸弹结构选择让牌
                    bool followed_natural = false;
                    if (table_trick_.type == TRICK_SOLO) {
                        for (int r = table_trick_.rank + 1; r < 15; ++r) {
                            if (hands_[0][r] == 1) {
                                if ((r == 13 || r == 14) && has_rocket(0)) continue;
                                hands_[0][r]--;
                                cards_left_[0]--;
                                table_trick_ = Trick{TRICK_SOLO, r, 0};
                                record_card_played(r, 1, 0);
                                played = true;
                                followed_natural = true;
                                break;
                            }
                        }
                    } else if (table_trick_.type == TRICK_PAIR) {
                        for (int r = table_trick_.rank + 1; r < 13; ++r) {
                            if (hands_[0][r] == 2) {
                                hands_[0][r] -= 2;
                                cards_left_[0] -= 2;
                                table_trick_ = Trick{TRICK_PAIR, r, 0};
                                record_card_played(r, 2, 0);
                                played = true;
                                followed_natural = true;
                                break;
                            }
                        }
                    }
                    if (!followed_natural) {
                        // 保全结构让牌
                        pass_count_++;
                        reward += 0.5;
                        if (pass_count_ == 2) {
                            current_turn_ = table_trick_.owner;
                            table_trick_ = Trick{TRICK_NONE, -1, -1};
                            pass_count_ = 0;
                        } else {
                            current_turn_ = 1;
                        }
                    }
                }
                if (played) {
                    pass_count_ = 0;
                    current_turn_ = 1;
                }
            }
        } else if (action == 1) { // 合规跟牌 (Clean Follow, 严格保全手牌结构)
            int prev_cards = cards_left_[0];
            play_opponent_turn(0);
            if (cards_left_[0] < prev_cards) {
                played = true;
                if (table_trick_.type == TRICK_BOMB || table_trick_.type == TRICK_ROCKET) reward += 5.5;
                else if (table_trick_.type == TRICK_PAIR) reward += 1.0;
                else reward += 0.8;
                if (!is_landlord && table_trick_.owner == 0 && landlord_cards <= 2) reward += 1.5;
            } else {
                if (!is_landlord && table_trick_.owner == teammate) reward += 1.2;
                else if (table_trick_.rank >= 8) reward += 1.0;
            }
        } else { // action == 2: 强行夺权 / 炸弹突击 / 残局冲刺 (Power Seize)
            if (table_trick_.type == TRICK_NONE) {
                // 自由出牌:
                // 1. 斩杀终局: <= 2 张直接出完
                if (cards_left_[0] <= 2 && has_rocket(0)) {
                    play_rocket(0);
                    played = true;
                    reward += 4.0;
                } else if (cards_left_[0] == 2) {
                    for (int r = 0; r < 13; ++r) {
                        if (hands_[0][r] == 2) {
                            hands_[0][r] -= 2;
                            cards_left_[0] -= 2;
                            table_trick_ = Trick{TRICK_PAIR, r, 0};
                            record_card_played(r, 2, 0);
                            played = true;
                            reward += 3.5;
                            break;
                        }
                    }
                } else if (cards_left_[0] == 1) {
                    for (int r = 0; r < 15; ++r) {
                        if (hands_[0][r] >= 1) {
                            hands_[0][r]--;
                            cards_left_[0]--;
                            table_trick_ = Trick{TRICK_SOLO, r, 0};
                            record_card_played(r, 1, 0);
                            played = true;
                            reward += 3.0;
                            break;
                        }
                    }
                }
                // 2. 记牌器赋能: 仅在残局冲刺阶段 (<=5张或对手<=3张)，出持有全场无敌顶牌必拿牌权！
                if (!played && (cards_left_[0] <= 5 || enemy_min_cards <= 3)) {
                    int top_unseen = lattice_.highest_unseen_rank();
                    for (int r = 14; r > top_unseen && r >= 10; --r) {
                        if (hands_[0][r] >= 1) {
                            if ((r == 13 || r == 14) && has_rocket(0)) continue;
                            hands_[0][r]--;
                            cards_left_[0]--;
                            table_trick_ = Trick{TRICK_SOLO, r, 0};
                            record_card_played(r, 1, 0);
                            played = true;
                            reward += 2.5;
                            break;
                        }
                    }
                }
                // 3. 残局冲刺 / 常规出牌
                if (!played) {
                    int prev_cards = cards_left_[0];
                    play_opponent_turn(0);
                    if (cards_left_[0] < prev_cards) {
                        played = true;
                        reward += 0.8;
                    }
                }
                pass_count_ = 0;
                current_turn_ = 1;
            } else if (!is_landlord && table_trick_.owner == teammate) {
                // 盟友控场: 避免误伤友军
                pass_count_++;
                reward += 1.0;
                if (pass_count_ == 2) {
                    current_turn_ = table_trick_.owner;
                    table_trick_ = Trick{TRICK_NONE, -1, -1};
                    pass_count_ = 0;
                } else {
                    current_turn_ = 1;
                }
            } else {
                // 对手出牌: 强行夺权 / 炸弹突击 / 残局截胡
                bool urgent = (enemy_min_cards <= 3) || (cards_left_[0] <= 3) || (table_trick_.type == TRICK_BOMB);

                // 炸弹或王炸压制
                if (urgent && table_trick_.type != TRICK_ROCKET) {
                    for (int r = 0; r < 13; ++r) {
                        if (hands_[0][r] == 4 && (table_trick_.type != TRICK_BOMB || r > table_trick_.rank)) {
                            hands_[0][r] -= 4;
                            cards_left_[0] -= 4;
                            table_trick_ = Trick{TRICK_BOMB, r, 0};
                            record_card_played(r, 4, 0);
                            played = true;
                            reward += 7.0;
                            break;
                        }
                    }
                    if (!played && has_rocket(0)) {
                        play_rocket(0);
                        played = true;
                        reward += 8.0;
                    }
                }
                // 若对手已到终盘只有 1 张牌，防守必须顶最大可用合法牌型截胡
                if (!played && enemy_min_cards == 1) {
                    if (table_trick_.type == TRICK_SOLO) {
                        for (int r = 14; r > table_trick_.rank; --r) {
                            if (hands_[0][r] >= 1 && hands_[0][r] < 4) {
                                if ((r == 13 || r == 14) && has_rocket(0)) continue;
                                hands_[0][r]--;
                                cards_left_[0]--;
                                table_trick_ = Trick{TRICK_SOLO, r, 0};
                                record_card_played(r, 1, 0);
                                played = true;
                                reward += 2.0;
                                break;
                            }
                        }
                    }
                }
                // 其余情况合规保结构跟牌（出最小能压过的牌，保留有生力量）
                if (!played) {
                    int prev_cards = cards_left_[0];
                    play_opponent_turn(0);
                    if (cards_left_[0] < prev_cards) {
                        played = true;
                        reward += 0.8;
                    }
                }
                if (played) {
                    pass_count_ = 0;
                    current_turn_ = 1;
                } else {
                    pass_count_++;
                    if (pass_count_ == 2) {
                        current_turn_ = table_trick_.owner;
                        table_trick_ = Trick{TRICK_NONE, -1, -1};
                        pass_count_ = 0;
                    } else {
                        current_turn_ = 1;
                    }
                }
            }
        }

        // 检查智能体是否出完手牌
        if (cards_left_[0] <= 0) {
            agent_won_ = true;
            done = true;
            if (is_landlord) {
                reward += 35.0 + 1.0 * (cards_left_[1] + cards_left_[2]);
            } else {
                reward += 30.0 + 1.5 * cards_left_[landlord_];
            }
        }

        // 轮转对手 1 与 2，推进至智能体下一决策轮次或终局
        while (current_turn_ != 0 && !done && round_count_ < max_rounds_) {
            int p = current_turn_;
            play_opponent_turn(p);

            if (cards_left_[p] <= 0) {
                done = true;
                if (is_landlord) {
                    agent_won_ = false;
                    reward -= 20.0;
                } else {
                    if (p == landlord_) {
                        agent_won_ = false;
                        reward -= 20.0;
                    } else {
                        agent_won_ = true;
                        reward += 30.0 + 1.0 * cards_left_[landlord_];
                    }
                }
                break;
            }
        }

        if (!done && round_count_ >= max_rounds_) {
            agent_won_ = false;
            reward -= 10.0;
            done = true;
        }

        if (done) {
            games_played_++;
            if (agent_won_) total_wins_++;
        }

        StepResult res;
        res.obs = current_observation();
        res.reward = reward;
        res.done = done;
        res.success = agent_won_;
        res.steps = round_count_;
        res.min_dist_to_goal = static_cast<double>(cards_left_[0]);
        return res;
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        int act = 1;
        if (table_trick_.type == TRICK_NONE) {
            // 自由出牌权: 规则禁止让牌 (0), 只能在合规出牌 (1) 与抢牌/冲刺 (2) 间决断
            act = (acts.defensive_reset > acts.positive_action) ? 2 : 1;
        } else {
            if (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) {
                act = 2; // 强行夺权/炸弹突击 (Power Seize)
            } else if (acts.negative_action > acts.positive_action) {
                act = 0; // 审慎让牌 (Strategic Pass/Hold)
            } else {
                act = 1; // 合规跟牌 (Clean Follow, 保全手牌结构)
            }
        }
        return step(act);
    }

    double current_fitness() const override {
        double win_rate = static_cast<double>(total_wins_) / std::max(1, games_played_);
        return win_rate * 100.0;
    }

    const CardCountingLattice& counting_lattice() const { return lattice_; }
    int role() const { return role_; }
    int landlord() const { return landlord_; }
    const Trick& table_trick() const { return table_trick_; }
    int cards_left(int p) const { return cards_left_[p]; }

private:
    int max_rounds_{40};
    double bidding_threshold_{14.2};
    int round_count_{0};
    int landlord_{0};
    int role_{1}; // 1: 地主, 0: 农民
    int current_turn_{0};
    int pass_count_{0};
    int hands_[3][15]{};
    int cards_left_[3]{20, 17, 17};
    Trick table_trick_;
    CardCountingLattice lattice_;
    int high_cards_played_{0};
    bool agent_won_{false};
    int total_wins_{0};
    int games_played_{0};
    NeuralBidEvaluator neural_bid_evaluator_{nullptr};
    std::mt19937 rng_;
};

/**
 * @brief 构建 64 细胞时序循环皮层微柱生命体 (64-Cell Recurrent Cortical Architecture)
 * 架构规范:
 * 1. 32 维高维全息感知受体 (Cells 0..31: SENSE_CHANNEL 0..31, 涵盖 15 手牌 + 7 高牌记牌 + 6 台面上下文 + 4 角色余牌);
 * 2. 16 维前馈特征提取中间层 (Cells 32..47: SUM, SUB, AMPLIFY, THRESHOLD, CLIP, ABS, MULTIPLY, DAMPER 等);
 * 3. 12 维内部循环动态吸引子核心 (Cells 48..59: 包含 OP_INTEGRAL, OP_EMA, OP_SUM, OP_HYSTERESIS,
 *    配置递归循环反馈突触, 谱半径 rho ≈ 0.60 满足李雅普诺夫收缩映射与 BIBO 稳态);
 * 4. 1 维全局决策收敛枢纽 (Cell 60: OP_SUM);
 * 5. 3 维动作效应器通道 (Cells 61..63: ACT_PRIMARY_POSITIVE 跟牌, ACT_PRIMARY_NEGATIVE 让牌, ACT_DEFENSIVE_RESET 抢牌);
 * 6. 严格恪守 Rule 7 Substrate Immunity 宪章：底座纯数学动力学，任务适配层零污染。
 */
inline CellularOrganism build_doudizhu_64cell_recurrent_cortex() {
    CellularOrganism org;

    // --- Layer 0: 32 维全息感知受体 (Cells 0..31) ---
    for (uint32_t i = 0; i < 32; ++i) {
        float y_pos = -60.0f + (120.0f / 31.0f) * static_cast<float>(i);
        org.cells.push_back({
            i, CellType::SENSE_CHANNEL, 1.0, static_cast<double>(i),
            0.0, 0.0, false, 0.0, 0, 0, -80.0f, y_pos, 0.0f
        });
    }

    // --- Layer 1: 16 维特征提取与态势估计 (Cells 32..47) ---
    org.cells.push_back({32, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -50.0f, 0.0f});
    org.cells.push_back({33, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -43.0f, 0.0f});
    org.cells.push_back({34, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -36.0f, 0.0f});
    org.cells.push_back({35, CellType::GATE_THRESHOLD, -0.25, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -29.0f, 0.0f});
    org.cells.push_back({36, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -22.0f, 0.0f});
    org.cells.push_back({37, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -15.0f, 0.0f});
    org.cells.push_back({38, CellType::OP_SUM, 1.2, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -8.0f, 0.0f});
    org.cells.push_back({39, CellType::GATE_THRESHOLD, -0.25, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, -1.0f, 0.0f});
    org.cells.push_back({40, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 6.0f, 0.0f});
    org.cells.push_back({41, CellType::OP_MULTIPLY, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 13.0f, 0.0f});
    org.cells.push_back({42, CellType::GATE_THRESHOLD, -0.5, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 20.0f, 0.0f});
    org.cells.push_back({43, CellType::OP_EMA, 0.35, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 27.0f, 0.0f});
    org.cells.push_back({44, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 34.0f, 0.0f});
    org.cells.push_back({45, CellType::GATE_THRESHOLD, -0.16, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 41.0f, 0.0f});
    org.cells.push_back({46, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 48.0f, 0.0f});
    org.cells.push_back({47, CellType::OP_ABS, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -35.0f, 55.0f, 0.0f});

    // --- Layer 2: 12 维内部循环动态吸引子核心 (Cells 48..59) ---
    org.cells.push_back({48, CellType::OP_INTEGRAL, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -44.0f, 0.0f});
    org.cells.push_back({49, CellType::OP_EMA, 0.60, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -36.0f, 0.0f});
    org.cells.push_back({50, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -28.0f, 0.0f});
    org.cells.push_back({51, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -20.0f, 0.0f});
    org.cells.push_back({52, CellType::OP_INTEGRAL, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -12.0f, 0.0f});
    org.cells.push_back({53, CellType::OP_EMA, 0.60, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, -4.0f, 0.0f});
    org.cells.push_back({54, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 4.0f, 0.0f});
    org.cells.push_back({55, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 12.0f, 0.0f});
    org.cells.push_back({56, CellType::GATE_HYSTERESIS, 0.2, 0.8, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 20.0f, 0.0f});
    org.cells.push_back({57, CellType::OP_INTEGRAL, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 28.0f, 0.0f});
    org.cells.push_back({58, CellType::OP_EMA, 0.60, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 36.0f, 0.0f});
    org.cells.push_back({59, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 5.0f, 44.0f, 0.0f});

    // --- Layer 3: 1 维全局决策收敛枢纽 (Cell 60) ---
    org.cells.push_back({60, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 40.0f, 0.0f, 0.0f});

    // --- Layer 4: 3 维动作效应器通道 (Cells 61..63) ---
    org.cells.push_back({61, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 75.0f, -30.0f, 0.0f});
    org.cells.push_back({62, CellType::ACT_PRIMARY_NEGATIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 75.0f, 0.0f, 0.0f});
    org.cells.push_back({63, CellType::ACT_DEFENSIVE_RESET, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 75.0f, 30.0f, 0.0f});

    // 1. Layer 0 (Sensory) -> Layer 1 (Features)
    for (uint32_t i = 0; i <= 7; ++i) {
        org.synapses.push_back({i, 32, 0, 0.35, true, 50.0f, -1.0f});
    }
    for (uint32_t i = 8; i <= 14; ++i) {
        org.synapses.push_back({i, 33, 0, 0.60, true, 50.0f, -1.0f});
    }
    org.synapses.push_back({33, 34, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({23, 34, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({33, 35, 0, -1.0, true, 50.0f, -1.0f});
    for (uint32_t i = 15; i <= 21; ++i) {
        org.synapses.push_back({i, 36, 0, 0.40, true, 50.0f, -1.0f});
    }
    org.synapses.push_back({36, 37, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({33, 37, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({31, 38, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({29, 39, 0, -1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({23, 40, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({33, 40, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({26, 41, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({30, 41, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({27, 42, 0, -1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({22, 43, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({29, 44, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({31, 44, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({29, 45, 0, -1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({26, 46, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({44, 47, 0, 1.0, true, 50.0f, -1.0f});

    // 1.5 记牌晶格 -> 记忆槽直汇 (Card-Counting Lattice -> Memory Slots):
    // obs[15..21] 对手余牌/记牌器 → INTEGRAL 工作记忆 (48/52/57) 与 EMA 短程记忆 (49/53)
    org.synapses.push_back({15, 48, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({16, 48, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({17, 48, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({18, 49, 0, 0.70, true, 50.0f, -1.0f});
    org.synapses.push_back({19, 52, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({20, 52, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({21, 52, 0, 0.85, true, 50.0f, -1.0f});
    org.synapses.push_back({15, 53, 0, 0.70, true, 50.0f, -1.0f});
    org.synapses.push_back({16, 57, 0, 0.80, true, 50.0f, -1.0f});
    org.synapses.push_back({17, 57, 0, 0.80, true, 50.0f, -1.0f});

    // 2. Layer 1 -> Layer 2 (Attractor Core)
    org.synapses.push_back({34, 50, 0, 1.2, true, 50.0f, -1.0f});
    org.synapses.push_back({41, 51, 0, 1.6, true, 50.0f, -1.0f});
    org.synapses.push_back({42, 50, 0, 0.8, true, 50.0f, -1.0f});
    org.synapses.push_back({42, 55, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({45, 55, 0, 2.2, true, 50.0f, -1.0f});
    org.synapses.push_back({37, 51, 0, 0.9, true, 50.0f, -1.0f});
    org.synapses.push_back({39, 55, 0, 1.4, true, 50.0f, -1.0f});
    org.synapses.push_back({46, 57, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({38, 58, 0, 0.7, true, 50.0f, -1.0f});
    org.synapses.push_back({40, 54, 0, 0.8, true, 50.0f, -1.0f});
    org.synapses.push_back({34, 54, 1, 0.8, true, 50.0f, -1.0f});
    org.synapses.push_back({54, 56, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({56, 59, 0, 0.9, true, 50.0f, -1.0f});

    // 3. Layer 2 内部动态循环反馈回路 (Internal Recurrent Loops, 谱半径 rho ≈ 0.60)
    org.synapses.push_back({50, 48, 0, 0.774, true, 50.0f, -1.0f});
    org.synapses.push_back({48, 50, 1, 0.774, true, 50.0f, -1.0f});

    org.synapses.push_back({51, 49, 0, 0.774, true, 50.0f, -1.0f});
    org.synapses.push_back({49, 51, 1, 0.774, true, 50.0f, -1.0f});

    org.synapses.push_back({55, 52, 0, 0.774, true, 50.0f, -1.0f});
    org.synapses.push_back({52, 55, 1, 0.774, true, 50.0f, -1.0f});

    org.synapses.push_back({59, 57, 0, 0.774, true, 50.0f, -1.0f});
    org.synapses.push_back({57, 59, 1, 0.774, true, 50.0f, -1.0f});

    // 侧向交互抑制
    org.synapses.push_back({50, 51, 0, -0.25, true, 50.0f, -1.0f});
    org.synapses.push_back({51, 50, 0, -0.25, true, 50.0f, -1.0f});
    org.synapses.push_back({55, 51, 0, -0.40, true, 50.0f, -1.0f});
    org.synapses.push_back({59, 60, 0, 0.80, true, 50.0f, -1.0f});

    // 4. Layer 2 / Layer 1 -> Layer 4 (Effectors: 61, 62, 63: 经过规范化缩放，防止 Softmax 饱和与梯度消失)
    org.synapses.push_back({50, 61, 0, 0.08, true, 50.0f, -1.0f});
    org.synapses.push_back({51, 62, 0, 0.08, true, 50.0f, -1.0f});
    org.synapses.push_back({55, 63, 0, 0.09, true, 50.0f, -1.0f});
    org.synapses.push_back({60, 61, 0, 0.03, true, 50.0f, -1.0f});
    org.synapses.push_back({60, 63, 0, 0.03, true, 50.0f, -1.0f});
    org.synapses.push_back({51, 63, 0, -0.05, true, 50.0f, -1.0f});

    // 基础直觉前向反射
    org.synapses.push_back({34, 61, 0, 0.04, true, 50.0f, -1.0f});
    org.synapses.push_back({41, 62, 0, 0.05, true, 50.0f, -1.0f});
    org.synapses.push_back({45, 63, 0, 0.06, true, 50.0f, -1.0f});

    // 补齐其余通道与中间特征的通路 (确保 64 细胞全图连通活性)
    org.synapses.push_back({32, 50, 0, 0.40, true, 50.0f, -1.0f});
    org.synapses.push_back({32, 61, 0, 0.02, true, 50.0f, -1.0f});
    org.synapses.push_back({35, 55, 0, 1.10, true, 50.0f, -1.0f});
    org.synapses.push_back({24, 43, 0, 0.60, true, 50.0f, -1.0f});
    org.synapses.push_back({25, 50, 0, 1.20, true, 50.0f, -1.0f});
    org.synapses.push_back({43, 50, 0, -0.40, true, 50.0f, -1.0f});
    org.synapses.push_back({28, 55, 0, 0.70, true, 50.0f, -1.0f});
    org.synapses.push_back({28, 51, 0, -0.50, true, 50.0f, -1.0f});
    org.synapses.push_back({38, 55, 0, 1.10, true, 50.0f, -1.0f});
    org.synapses.push_back({44, 55, 0, 0.80, true, 50.0f, -1.0f});
    org.synapses.push_back({47, 59, 0, 0.50, true, 50.0f, -1.0f});
    org.synapses.push_back({55, 53, 0, 0.60, true, 50.0f, -1.0f});
    org.synapses.push_back({53, 55, 0, 0.60, true, 50.0f, -1.0f});
    org.synapses.push_back({58, 51, 0, 0.80, true, 50.0f, -1.0f});
    org.synapses.push_back({58, 59, 0, 0.50, true, 50.0f, -1.0f});

    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

/**
 * @brief 构建 64 细胞端到端神经叫牌皮层微柱 (Neural Bidding Cortex)
 * 32 维全息手牌输入 -> 16 维特征提取 (大牌前馈赋能、散牌抑制) -> 12 维循环动力学核心 -> 效应器输出 (叫牌/不叫)
 */
inline CellularOrganism build_doudizhu_bidding_cortex() {
    CellularOrganism org = build_doudizhu_64cell_recurrent_cortex();

    // 强化大牌 (11: A, 12: 2, 13: 小王, 14: 大王) 对叫牌通道 (Cell 61) 的直接前馈赋能
    org.synapses.push_back({11, 61, 0, 0.15, true, 50.0f, -1.0f});
    org.synapses.push_back({12, 61, 0, 0.40, true, 50.0f, -1.0f});
    org.synapses.push_back({13, 61, 0, 0.55, true, 50.0f, -1.0f});
    org.synapses.push_back({14, 61, 0, 0.75, true, 50.0f, -1.0f});

    // 散牌 (0..6: 3..9) 对让牌通道 (Cell 62) 产生正向激励 (抑制盲目叫牌)
    for (uint32_t r = 0; r <= 6; ++r) {
        org.synapses.push_back({r, 62, 0, 0.35, true, 50.0f, -1.0f});
    }

    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

/**
 * @brief 构建 64 细胞地主进攻专精皮层微柱 (Landlord Assault Specialist Column)
 */
inline CellularOrganism build_doudizhu_landlord_cortex(const std::string& base_ckpt = "") {
    CellularOrganism org;
    if (!base_ckpt.empty() && std::ifstream(base_ckpt).good()) {
        org = CellularOrganism::load_checkpoint_bin(base_ckpt);
    } else {
        org = build_doudizhu_64cell_recurrent_cortex();
    }
    for (auto& syn : org.synapses) {
        if (syn.to_cell_id >= 60 && syn.to_cell_id <= 63) syn.weight *= 1.05; // 强化进攻激进度
    }
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

/**
 * @brief 构建 64 细胞农民防守协作专精皮层微柱 (Peasant Cooperation Specialist Column)
 */
inline CellularOrganism build_doudizhu_peasant_cortex(const std::string& base_ckpt = "") {
    CellularOrganism org;
    if (!base_ckpt.empty() && std::ifstream(base_ckpt).good()) {
        org = CellularOrganism::load_checkpoint_bin(base_ckpt);
    } else {
        org = build_doudizhu_64cell_recurrent_cortex();
    }
    for (auto& syn : org.synapses) {
        if (syn.from_cell_id == 26 && syn.to_cell_id == 62) syn.weight *= 1.4; // 盟友牌大果断让牌
        if (syn.to_cell_id >= 60 && syn.to_cell_id <= 63) syn.weight *= 0.95; // 强化稳健保牌
    }
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

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
    size_t obs_dim() const override { return 6; } // [横向偏差 cte, 航向偏差 d_psi, 纵向速度 v, 冲突物TTC, 目标曲率 kappa, 剩余机动距离]
    size_t act_dim() const override { return 2; } // [侧向控制角 lat_cmd, 纵向加速度 a/加减速]

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
        // 离散映射: 0: 减速让行并向目标偏转, 1: 加速抢越并跟踪轨迹
        float lat_cmd = (target_kappa_ > 0) ? 0.35f : -0.35f;
        float accel = (action == 1) ? 1.5f : -2.5f;
        return step_internal(lat_cmd, accel);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        float lat_cmd = static_cast<float>(acts.positive_action - acts.negative_action);
        float accel = static_cast<float>(acts.positive_action * 2.5 - acts.defensive_reset * 4.0);
        return step_internal(lat_cmd, accel);
    }

    double current_fitness() const override {
        double fit = reached_target_ ? 100.0 : 0.0;
        fit -= std::hypot(target_x_ - x_, target_y_ - y_) * 1.5;
        if (collision_) fit -= 80.0;
        return std::max(0.0, fit);
    }

private:
    StepResult step_internal(float lat_cmd, float accel) {
        step_count_++;
        float dt = 0.1f;

        // 动力学单轨模型积分
        float L = 2.8f;
        v_ = std::clamp(v_ + accel * dt, 0.0f, 12.0f);
        x_ += v_ * std::cos(psi_) * dt;
        y_ += v_ * std::sin(psi_) * dt;
        psi_ += (v_ / L) * std::tan(std::clamp(lat_cmd, -0.6f, 0.6f)) * dt;

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

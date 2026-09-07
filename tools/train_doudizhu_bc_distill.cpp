// ============================================================================
// 斗地主行为克隆蒸馏器 (Behavior Cloning Distillation)
// 教师: 内置启发式执行器 (play_opponent_turn, 数据/标签免费)
// 学生: 64 细胞递归皮层 (direct 语义: act0=过牌 act1=最低合法跟牌 act2=夺权)
// 模式: ./train_doudizhu_bc_distill gen [games]  → 生成数据集
//       ./train_doudizhu_bc_distill train        → 监督克隆训练 + 导出
// ============================================================================
#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include <cstdio>
#include <fstream>
#include <random>
#include <cmath>
#include <map>

using namespace kun;

// 直接语义的最低合法跟牌 (与任务层 direct act1 完全一致的标签判据)
static int lowest_beat_rank(const int hand[15], const kun::DouDiZhuCardGameTask::Trick& t) {
    if (t.type == kun::DouDiZhuCardGameTask::TRICK_SOLO) {
        for (int r = t.rank + 1; r < 15; ++r) {
            if (hand[r] >= 1 && hand[r] < 4) {
                if ((r == 13 || r == 14) && hand[13] > 0 && hand[14] > 0) continue;
                return r;
            }
        }
    } else if (t.type == kun::DouDiZhuCardGameTask::TRICK_PAIR) {
        for (int r = t.rank + 1; r < 13; ++r) if (hand[r] == 2) return r;
    } else if (t.type == kun::DouDiZhuCardGameTask::TRICK_BOMB) {
        for (int r = t.rank + 1; r < 13; ++r) if (hand[r] == 4) return r;
    }
    return -1;
}

static int gen_dataset(int games, const char* path) {
    std::ofstream f(path, std::ios::binary);
    int32_t hdr = games; f.write((char*)&hdr, 4);
    long samples = 0;
    int label_hist[3] = {0, 0, 0};
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(500000 + g * 131), 17.5);
        // 座 0 = 永远 act1 (委托启发式出牌); 每个决策点记录 (obs, 教师行为标签)
        bool done = false;
        while (!done) {
            // 决策前状态
            float obs[32];
            auto o = task.current_observation();
            for (int d = 0; d < 32; ++d) obs[d] = o[d];
            int pre_hand[15];
            for (int r = 0; r < 15; ++r) pre_hand[r] = (int)std::lround(std::clamp(o[r] * (r < 13 ? 4.0f : 1.0f), 0.f, 4.f));
            DouDiZhuCardGameTask::Trick pre = task.table_trick();
            int pre_cards = task.cards_left(0);

            auto res = task.step(1); // act1 委托: play_opponent_turn(0)

            // 标签: 座 0 本回合的行为分类 (direct 语义)
            int label;
            if (task.cards_left(0) == pre_cards) {
                label = 0; // 过牌 (含跟不上/盟友控场让牌)
            } else {
                DouDiZhuCardGameTask::Trick post = task.table_trick();
                if (post.type == kun::DouDiZhuCardGameTask::TRICK_ROCKET) label = 2;
                else if (post.type == kun::DouDiZhuCardGameTask::TRICK_BOMB && pre.type != kun::DouDiZhuCardGameTask::TRICK_BOMB) label = 2;
                else if (post.type == kun::DouDiZhuCardGameTask::TRICK_BOMB) label = (post.rank > pre.rank) ? 2 : 1;
                else if (pre.type == kun::DouDiZhuCardGameTask::TRICK_NONE) {
                    label = (pre_cards - task.cards_left(0) == 2) ? 2 : 1; // 对子终结=2, 常规低引=1
                } else {
                    int lb = lowest_beat_rank(pre_hand, pre);
                    label = (post.rank == lb) ? 1 : 2; // 最低压=1, 高牌截胡/炸弹=2
                }
            }
            label_hist[label]++;
            f.write((char*)obs, sizeof(float) * 32);
            f.write((char*)&label, 4);
            int32_t gid = g;
            f.write((char*)&gid, 4);
            samples++;
            done = res.done;
        }
    }
    f.close();
    printf("[数据集] %d 局 → %ld 样本 | 标签分布: 过牌=%d 跟牌=%d 夺权=%d → %s\n",
           games, samples, label_hist[0], label_hist[1], label_hist[2], path);
    return (int)samples;
}

static int train_bc(const char* path) {
    std::ifstream f(path, std::ios::binary);
    int32_t games; f.read((char*)&games, 4);
    std::vector<std::array<float, 32>> X;
    std::vector<int> Y;
    std::vector<int32_t> G;
    while (f.good()) {
        std::array<float, 32> x;
        int y; int32_t gid;
        f.read((char*)x.data(), sizeof(float) * 32);
        f.read((char*)&y, 4);
        f.read((char*)&gid, 4);
        if (!f.good()) break;
        X.push_back(x); Y.push_back(y); G.push_back(gid);
    }
    f.close();
    printf("[BC] 载入 %zu 样本\n", X.size());
    if (X.empty()) return 1;

    CellularOrganism org = build_doudizhu_64cell_recurrent_cortex();
    // BC 初始化修正: 效应器入权 (0.02-0.09) 太小 → logits 近均匀 → CE 梯度近零
    // 放大效应器入站突触 ×8 (任务层初始化, 合宪), 让 softmax 有表达范围
    for (auto& s : org.synapses) {
        if (s.to_cell_id >= 61 && s.to_cell_id <= 63) s.weight *= 4.0;
    }
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    kun::CellularBPTTEngine bptt;
    bptt.init_optimizer(org);

    if (std::getenv("DZ_BC_SENSITIVITY")) {
        // 信号通路诊断: 32 个通道逐一扰动, 测效应器输出敏感度
        std::vector<double> base(32, 0.0);
        org.reset_state(true);
        auto base_out = org.forward_nd(base.data(), 32, false);
        printf("[敏感度] 基线输出: pos=%.4f neg=%.4f def=%.4f\n",
               base_out.positive_action, base_out.negative_action, base_out.defensive_reset);
        for (int d = 0; d < 32; ++d) {
            std::vector<double> in(32, 0.0); in[d] = 1.0;
            org.reset_state(true);
            auto o = org.forward_nd(in.data(), 32, false);
            double dp = o.positive_action - base_out.positive_action;
            double dn = o.negative_action - base_out.negative_action;
            double dd = o.defensive_reset - base_out.defensive_reset;
            if (std::abs(dp) + std::abs(dn) + std::abs(dd) > 1e-4)
                printf("  ch%02d: Δpos=%+.4f Δneg=%+.4f Δdef=%+.4f\n", d, dp, dn, dd);
        }
        return 0;
    }
    if (std::getenv("DZ_BC_DEBUG")) {
        // 机制诊断: 单样本重复更新 (纯过拟合测试) + 梯度幅值
        std::vector<double> in(X[0].begin(), X[0].end());
        std::vector<float> tgt(3, 0.0f);
        tgt[Y[0] == 1 ? 0 : (Y[0] == 0 ? 1 : 2)] = 1.0f;
        double w0 = org.compiled_synapses_[0].weight;
        for (int it = 0; it < 200; ++it) {
            bptt.reset_tape();
            org.reset_state(true);
            org.forward_nd(in.data(), in.size(), false);
            bptt.record_step(org);
            BPTTGradients grads;
            float loss = bptt.backward_with_loss(org, {tgt}, grads, SubstrateLossType::CROSS_ENTROPY);
            float gmax = 0, gnonzero = 0;
            for (float g : grads.grad_synapses) { gmax = std::max(gmax, std::abs(g)); if (std::abs(g) > 1e-9f) gnonzero++; }
            bptt.step_adam(org, grads, 0.01f);
            if (it % 40 == 0)
                printf("[诊断] it=%d loss=%.4f max|g|=%.3e 非零梯度=%ld w0=%.5f\n", it, loss, gmax, (long)gnonzero, org.compiled_synapses_[0].weight);
        }
        printf("[诊断] w0 变化: %.5f → %.5f\n", w0, org.compiled_synapses_[0].weight);
        return 0;
    }

    const int EPOCHS = 60;
    const float LR = 0.006f;
    std::mt19937 rng(42);
    // 按局分组 (序列索引: 保留时序, BPTT 时间信用归位)
    std::vector<std::vector<size_t>> seqs;
    {
        std::map<int32_t, std::vector<size_t>> by_game;
        for (size_t s = 0; s < X.size(); ++s) by_game[G[s]].push_back(s);
        for (auto& kv : by_game) seqs.push_back(kv.second);
    }
    printf("[BC] 序列化: %zu 局序列\n", seqs.size());
    for (int ep = 1; ep <= EPOCHS; ++ep) {
        std::shuffle(seqs.begin(), seqs.end(), rng);
        double loss_sum = 0; long n = 0;
        for (auto& seq : seqs) {
            bptt.reset_tape();
            org.reset_state(true);
            std::vector<std::vector<float>> tgts;
            for (size_t s : seq) {
                std::vector<double> in(X[s].begin(), X[s].end());
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                std::vector<float> tgt(3, 0.075f);
                tgt[Y[s] == 1 ? 0 : (Y[s] == 0 ? 1 : 2)] = 0.85f;
                tgts.push_back(tgt);
            }
            BPTTGradients grads;
            double L = bptt.backward_with_loss(org, tgts, grads, SubstrateLossType::CROSS_ENTROPY);
            bptt.step_adam(org, grads, LR);
            loss_sum += L; n += (long)seq.size();
        }
        if (ep % 5 == 0 || ep == 1)
            printf("[BC] Epoch %d/%d | 平均 CE loss %.4f\n", ep, EPOCHS, loss_sum / (double)n);
    }

    // 训练后准确率 (direct 映射: act1 if pos>=neg&&pos>=def... 与任务层一致)
    long correct = 0;
    for (size_t s = 0; s < X.size(); ++s) {
        org.reset_state(true);
        std::vector<double> in(X[s].begin(), X[s].end());
        auto acts = org.forward_nd(in.data(), in.size(), false);
        int pred = (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) ? 2
                 : ((acts.negative_action > acts.positive_action) ? 0 : 1);
        if (pred == Y[s]) correct++;
    }
    printf("[BC] 训练后教师动作复现准确率: %.1f%% (%ld/%zu)\n", 100.0 * correct / X.size(), correct, X.size());
    org.save_checkpoint_bin("checkpoints/doudizhu_bc_distilled.bin");
    printf("[产物] checkpoints/doudizhu_bc_distilled.bin\n");
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "train";
    if (mode == "gen") {
        int games = argc > 2 ? std::atoi(argv[2]) : 2000;
        gen_dataset(games, "/tmp/opencode/doudizhu_bc_dataset.bin");
    } else {
        train_bc(argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_bc_dataset.bin");
    }
    return 0;
}

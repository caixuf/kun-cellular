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
#include <array>

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

// v2: rank 数据集 — 标签 = 教师实际打出的点数 (0..14) 或过牌 (15), 精确无歧义
static int gen_dataset_rank(int games, const char* path) {
    std::ofstream f(path, std::ios::binary);
    int32_t hdr = games; f.write((char*)&hdr, 4);
    long samples = 0;
    int label_hist[17] = {0};
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(900000 + g * 173), 17.5);
        task.set_rank_action_mode(true);
        bool done = false;
        while (!done) {
            auto o = task.current_observation();
            float obs[32];
            for (int d = 0; d < 32; ++d) obs[d] = o[d];
            int played_rank = task.teacher_play_capture();  // 立即捕获 (rotation 前)
            int label = (played_rank >= 0 && played_rank <= 14) ? played_rank : 15;
            auto res = task.settle_turn();                  // 轮转 + 结算
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
    printf("[数据集v2] %d 局 → %ld 样本 | 点数标签分布:", games, samples);
    for (int r = 0; r < 16; ++r) printf(" %d:%d", r, label_hist[r]);
    printf("\n→ %s\n", path);
    return (int)samples;
}

// DAgger: 模型驱动轨迹 (分布 = 模型自身), 教师标签在任务拷贝上只读标注 (根治分布偏移)
static int gen_dataset_dagger(int games, const char* path) {
    CellularOrganism org = build_doudizhu_rank_cortex_v3();
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    org.load_checkpoint_bin("checkpoints/doudizhu_bc_rank.bin");
    std::ofstream f(path, std::ios::binary);
    int32_t hdr = games; f.write((char*)&hdr, 4);
    long samples = 0;
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(7700000 + g * 211), 17.5);
        task.set_rank_action_mode(true);
        bool done = false;
        while (!done) {
            auto o = task.current_observation();
            float obs[32];
            for (int d = 0; d < 32; ++d) obs[d] = o[d];
            org.reset_state(true);
            std::vector<double> in(o.begin(), o.end());
            org.forward_nd(in.data(), in.size(), false);
            float y[19] = {0};
            for (int k = 0; k <= 15; ++k) y[3 + k] = (float)org.cells[64 + k].output_val;
            y[18] = (float)org.cells[79].output_val - 1.0f;
            // 教师标签: 状态拷贝 (模型动作未应用前), 只读标注
            DouDiZhuCardGameTask probe = task;
            int label_rank = probe.teacher_play_capture();
            int label = (label_rank >= 0 && label_rank <= 14) ? label_rank : 15;
            f.write((char*)obs, sizeof(float) * 32);
            f.write((char*)&label, 4);
            int32_t gid = g;
            f.write((char*)&gid, 4);
            samples++;
            auto res = task.step_rank_from_tensor(y);   // 模型动作驱动轨迹
            done = res.done;
        }
    }
    f.close();
    printf("[DAgger] %d 局 (模型驱动) -> %ld 样本\n-> %s\n", games, samples, path);
    return (int)samples;
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

static int train_bc_rank(const char* path) {
    std::ifstream f(path, std::ios::binary);
    int32_t games; f.read((char*)&games, 4);
    std::vector<std::array<float, 32>> X;
    std::vector<int> Y; std::vector<int32_t> G;
    while (f.good()) {
        std::array<float, 32> x; int y; int32_t gid;
        f.read((char*)x.data(), sizeof(float) * 32);
        f.read((char*)&y, 4); f.read((char*)&gid, 4);
        if (!f.good()) break;
        X.push_back(x); Y.push_back(y); G.push_back(gid);
    }
    f.close();
    printf("[BC-rank] 载入 %zu 样本\n", X.size());
    if (X.empty()) return 1;

    CellularOrganism org = build_doudizhu_rank_cortex_v3();
    for (auto& s : org.synapses) {
        if (s.to_cell_id >= 64) s.weight *= 4.0;   // rank 头入权放大 (logits 表达范围)
    }
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    kun::CellularBPTTEngine bptt;
    bptt.init_optimizer(org);

    // 按局分组 (序列 BPTT)
    std::vector<std::vector<size_t>> seqs;
    {
        std::map<int32_t, std::vector<size_t>> by_game;
        for (size_t s = 0; s < X.size(); ++s) by_game[G[s]].push_back(s);
        for (auto& kv : by_game) seqs.push_back(kv.second);
    }
    printf("[BC-rank] %zu 局序列\n", seqs.size());
    printf("[BC-rank] cells=%zu max_cell_id=%zu | 磁带窗口=%zu\n", org.cells.size(),
           [&]{ size_t m=0; for (auto& c : org.cells) m = std::max(m, (size_t)c.id); return m; }(), (size_t)0);

    const int EPOCHS = 120;
    const float LR = 0.02f;
    std::vector<std::array<double, 9>> grad_snap;
    std::mt19937 rng(42);
    for (int ep = 1; ep <= EPOCHS; ++ep) {
        std::shuffle(seqs.begin(), seqs.end(), rng);
        double loss_sum = 0; long n = 0;
        long seq_i = 0;
        for (auto& seq : seqs) {
            if (ep == 1 && seq_i % 200 == 0) printf("[BC-rank] ep1 seq %ld/%zu (len %zu)\n", seq_i, seqs.size(), seq.size());
            seq_i++;
            bptt.reset_tape();
            org.reset_state(true);
            std::vector<std::vector<float>> tgts;  // 19 通道: [0..2] 旧效应器 (小权重), [3+r] 点数, [18] 过牌
            for (size_t s : seq) {
                std::vector<double> in(X[s].begin(), X[s].end());
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                std::vector<float> tgt(19, 0.02f);
                tgt[3 + Y[s]] = (Y[s] == 15) ? 0.80f : 0.9f;   // 过牌类别降权 (活体超过 13pp 校准)
                tgts.push_back(tgt);
            }
            BPTTGradients grads;
            double L = bptt.backward_with_loss(org, tgts, grads, SubstrateLossType::CROSS_ENTROPY);
            // 逐层梯度诊断: 按突触目标细胞分桶 (rank头 64-80 / 吸引子 48-59 / 特征 32-47 / 受体侧 0-31)
            if (ep % 5 == 0 || ep == 1) {
                double gmax[4] = {0, 0, 0, 0}; long gcnt[4] = {0, 0, 0, 0};
                for (size_t si = 0; si < org.compiled_synapses_.size() && si < grads.grad_synapses.size(); ++si) {
                    float g = grads.grad_synapses[si];
                    int to = (int)org.compiled_synapses_[si].to_idx;
                    int b = (to >= 64) ? 0 : (to >= 48) ? 1 : (to >= 32) ? 2 : 3;
                    double ag = std::abs((double)g);
                    if (ag > gmax[b]) gmax[b] = ag;
                    if (ag > 1e-9) gcnt[b]++;
                }
                grad_snap.push_back({(double)L, gmax[0], gmax[1], gmax[2], gmax[3],
                                     (double)gcnt[0], (double)gcnt[1], (double)gcnt[2], (double)gcnt[3]});
            }
            bptt.step_adam(org, grads, LR);
            loss_sum += L; n += (long)seq.size();
        }
        if (ep % 5 == 0 || ep == 1) {
            printf("[BC-rank] Epoch %d/%d | CE %.4f\n", ep, EPOCHS, loss_sum / (double)n);
            if (!grad_snap.empty()) {
                auto& s = grad_snap.back();
                printf("  [梯度桶 max|g|] rank头=%.2e(%ld) 吸引子=%.2e(%ld) 特征=%.2e(%ld) 受体=%.2e(%ld)\n",
                       s[1], (long)s[5], s[2], (long)s[6], s[3], (long)s[7], s[4], (long)s[8]);
            }
        }
    }
    // 固定探针: 输出 + 入站权重分化诊断
    {
        printf("[权重诊断] 60→各rank头入权:");
        for (auto& s : org.compiled_synapses_) {
            int to = (int)s.to_idx;
            if (to >= 64 && to <= 70) printf(" w→%d=%.4f", to, (double)s.weight);
        }
        printf("\n");
        org.reset_state(true);
        std::vector<double> in(X[0].begin(), X[0].end());
        org.forward_nd(in.data(), in.size(), false);
        printf("[探针] 样本0 真实标签=%d | rank头输出:", Y[0]);
        for (int k = 0; k <= 16; ++k) printf(" c%d=%.3f", 3 + k, (double)org.cells[64 + k].output_val);
        printf("\n");
    }

    // 训练后复现准确率 + 混淆诊断 (argmax over 全部 17 头, 与 eval_rank 同映射)
    long correct = 0; long top3_hit = 0; long play_n = 0;
    int pp = 0, pq = 0, qp = 0, qq = 0;
    for (size_t s = 0; s < X.size(); ++s) {
        org.reset_state(true);
        std::vector<double> in(X[s].begin(), X[s].end());
        org.forward_nd(in.data(), in.size(), false);
        int best_r = 15; float bv = -1e9f;
        for (int k = 0; k <= 16; ++k) {
            float v = (float)org.cells[64 + k].output_val;
            if (v > bv) { bv = v; best_r = (k == 15) ? 15 : (k <= 14 ? k : best_r); }
        }
        if (best_r == Y[s]) correct++;
        bool tp = (best_r == 15), tq = (Y[s] == 15);
        if (tp && tq) pp++; else if (tp) pq++; else if (tq) qp++; else { qq++; play_n++; }
        if (!tq) {   // 出牌样本: top3 点数命中
            float v[15]; for (int r = 0; r < 15; ++r) v[r] = (float)org.cells[64 + r].output_val;
            int hits = 0; for (int t = 0; t < 3; ++t) {
                int am = 0; for (int r = 1; r < 15; ++r) if (v[r] > v[am]) am = r;
                if (am == Y[s]) hits++; v[am] = -1e9f;
            }
            if (hits > 0) top3_hit++;
        }
    }
    printf("[BC-rank] 单步复现准确率: %.1f%% (%ld/%zu)\n", 100.0 * correct / X.size(), correct, X.size());
    printf("[混淆 2x2] 过→过 %d 过→出 %d 出→过 %d 出→出 %d | 出牌样本点数Top3命中 %.1f%%\n",
           pp, pq, qp, qq, play_n > 0 ? 100.0 * top3_hit / play_n : 0.0);
    org.save_checkpoint_bin("checkpoints/doudizhu_bc_rank.bin");
    printf("[产物] checkpoints/doudizhu_bc_rank.bin\n");
    return 0;
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

    const int EPOCHS = 80;
    const float LR = 0.012f;
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

// rank 策略实战评测: 对启发式对手 N 局 (direct rank 语义)
// 混合策略评测: rank 头置信度 (top1-top2 margin) > theta 时接管, 否则回退启发式执行器
// 决定性实验: 若混合胜率 > 57.8% (纯委托) → 细胞智能有真实净贡献
static int eval_hybrid_games(int games, float theta, const char* ckpt) {
    CellularOrganism org = build_doudizhu_rank_cortex_v3();
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    org.load_checkpoint_bin(ckpt);
    int wins = 0; long hybrid_take = 0, heur_take = 0;
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        task.set_rank_action_mode(true);
        bool done = false;
        while (!done) {
            auto o = task.current_observation();
            std::vector<double> in(o.begin(), o.end());
            org.reset_state(true);
            org.forward_nd(in.data(), in.size(), false);
            float s[17];
            for (int k = 0; k <= 16; ++k) s[k] = (float)org.cells[64 + k].output_val;
            int b1 = 0, b2 = -1; float v1 = -1e9f, v2 = -1e9f;
            for (int k = 0; k < 17; ++k) {
                if (s[k] > v1) { v2 = v1; b2 = b1; v1 = s[k]; b1 = k; }
                else if (s[k] > v2) { v2 = s[k]; b2 = k; }
            }
            float margin = v1 - v2;
            if (margin > theta) {
                float y[19] = {0};
                for (int k = 0; k <= 15; ++k) y[3 + k] = s[k];
                y[18] = (float)org.cells[79].output_val - 1.0f;
                y[(b1 == 15) ? 18 : 3 + b1] = v1 + 3.0f;
                auto res = task.step_rank_from_tensor(y);
                done = res.done;
                if (res.success) wins++;
                hybrid_take++;
            } else {
                task.teacher_play_capture();   // 启发式执行器接管 (只动 seat 0)
                auto res = task.settle_turn();
                done = res.done;
                if (res.success) wins++;
                heur_take++;
            }
        }
    }
    printf("[混合 θ=%.1f] 胜率 %.1f%% (%d/%d) | rank接管 %.1f%% 启发式 %.1f%%\n",
           theta, 100.0 * wins / games, wins, games,
           100.0 * hybrid_take / (hybrid_take + heur_take), 100.0 * heur_take / (hybrid_take + heur_take));
    return wins;
}

static int eval_rank_games(int games, const char* ckpt_path) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(ckpt_path);
    org.compile();
    int wins = 0; long live_pass = 0, live_steps = 0;
    long task_forced = 0, task_voluntary = 0, prev_forced = 0, prev_voluntary = 0;
    double avg_out[17] = {0}, live_obs[32] = {0};
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        task.set_rank_action_mode(true);
        prev_forced = 0; prev_voluntary = 0;
        org.reset_state(true);
        bool done = false;
        while (!done) {
            auto o = task.current_observation();
            std::vector<double> in(o.begin(), o.end());
            org.reset_state(true);   // 活体累积验证: 每步重置 (无状态推理)
            org.forward_nd(in.data(), in.size(), false);
            float pre_cards_live = (float)task.cards_left(0);
            for (int d = 0; d < 32; ++d) live_obs[d] += o[d];
            float y[19] = {0};
            for (int k = 0; k <= 15; ++k) y[3 + k] = (float)org.cells[64 + k].output_val;
            y[18] = (float)org.cells[79].output_val - 1.0f;  // 过牌通道 + 保守偏置 (校准主动过, 扫描 0.5-3.0 等效)
            (void)pre_cards_live;
            auto res = task.step_rank_from_tensor(y);
            // 活体决策统计: 过牌判定 = 手牌数未减少 (比较 step 前快照)
            if (task.cards_left(0) == (int)pre_cards_live) live_pass++;
            live_steps++;
            for (int k = 0; k <= 16; ++k) avg_out[k] += (double)org.cells[64 + k].output_val;
            done = res.done;
            task_forced += task.forced_pass_count_ - prev_forced;
            task_voluntary += task.voluntary_pass_count_ - prev_voluntary;
            prev_forced = task.forced_pass_count_; prev_voluntary = task.voluntary_pass_count_;
            if (res.success) wins++;
        }
    }
    printf("[rank 实战] %d 局对启发式: 胜率 %.1f%% (%d/%d)\n", games, 100.0 * wins / games, wins, games);
    printf("[活体诊断] 步数 %ld 活体过牌率 %.1f%% (教师 46.8%%)\n", live_steps, 100.0 * live_pass / live_steps);
    printf("[过牌分解] 强制过 %ld 主动过 %ld\n", task_forced, task_voluntary);
    printf("[活体输出均值] rank头:", 0);
    for (int k = 0; k <= 16; ++k) printf(" c%d=%.2f", 64 + k, avg_out[k] / live_steps);
    printf("\n");
    // 数据集 obs 均值对照
    std::ifstream df("/tmp/opencode/doudizhu_bc_dataset_rank.bin", std::ios::binary);
    int32_t dgames; df.read((char*)&dgames, 4);
    double ds_obs[32] = {0}; size_t dn = 0;
    while (df.good()) {
        float xo[32]; int32_t xl, xg;
        df.read((char*)xo, 128); df.read((char*)&xl, 4); df.read((char*)&xg, 4);
        if (!df.good()) break;
        for (int d = 0; d < 32; ++d) ds_obs[d] += xo[d];
        dn++;
    }
    df.close();
    printf("[obs 均值对比] 通道: 活体 vs 数据集\n");
    for (int d : {23, 29, 15, 18, 21, 22, 25, 27})
        printf("  obs[%d] live=%.3f ds=%.3f\n", d, live_obs[d] / live_steps, ds_obs[d] / dn);
    return 0;
}

// 暖启动 GRPO: DAgger 蒸馏权重 -> on-policy 采样轨迹 -> 组相对优势 -> GRPO_SURROGATE 剪裁替代目标
static int train_rank_grpo(int iters, int group) {
    CellularOrganism org = build_doudizhu_rank_cortex_v3();
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    org.load_checkpoint_bin("checkpoints/doudizhu_bc_rank.bin");
    CellularBPTTEngine bptt;
    bptt.init_optimizer(org);   // 防御: 显式初始化优化器状态 (adam_step 时序)
    const float CLIP = 0.2f, ENT = 0.012f, LR = 0.008f;
    uint32_t lcg = 7717u;
    auto rnd01 = [&]() { lcg = lcg * 1664525u + 1013904223u; return (double)(lcg >> 8) / 16777216.0; };
    int total_wins = 0, total_games = 0;
    for (int it = 1; it <= iters; ++it) {
        struct StepRec { std::array<float, 32> obs; int chosen; float old_prob; int game; };
        std::vector<StepRec> steps;
        std::vector<double> game_reward(group, 0.0);
        std::vector<int> game_won(group, 0);
        for (int g = 0; g < group; ++g) {
            DouDiZhuCardGameTask task(40, (uint32_t)(5500000 + (long)it * 977 + g * 31), 17.5);
            task.set_rank_action_mode(true);
            bool done = false; double shaped = 0.0;
            while (!done) {
                auto o = task.current_observation();
                StepRec rec; std::copy(o.begin(), o.end(), rec.obs.begin());
                org.reset_state(true);
                std::vector<double> in(o.begin(), o.end());
                org.forward_nd(in.data(), in.size(), false);
                std::vector<float> preds(17);
                for (int k = 0; k <= 16; ++k) preds[k] = (float)org.cells[64 + k].output_val;
                // 合法无关的 softmax 采样 (探索); 通道语义: k=15 → 过牌(18), 其他 → 3+k
                double logits[17], mx = -1e30;
                for (int k = 0; k < 17; ++k) { logits[k] = (k == 15) ? (double)preds[15] - 1.0 : (double)preds[k]; if (logits[k] > mx) mx = logits[k]; }
                double exps[17], sum = 0;
                for (int k = 0; k < 17; ++k) { exps[k] = std::exp(logits[k] - mx); sum += exps[k]; }
                int chosen = 15; double acc = 0, r = rnd01() * sum;
                for (int k = 0; k < 17; ++k) { acc += exps[k]; if (r <= acc) { chosen = k; break; } }
                rec.chosen = (chosen == 15) ? 18 : 3 + chosen;
                rec.old_prob = (float)(exps[chosen] / std::max(sum, 1e-9));
                rec.game = g;
                float y[19] = {0};
                for (int k = 0; k <= 16; ++k) y[3 + k] = preds[k];
                y[18] = (float)org.cells[79].output_val - 1.0f;
                y[rec.chosen] = preds[chosen] + 3.0f;   // 选中通道强制最高 (applier argmax 对齐)
                auto res = task.step_rank_from_tensor(y);
                shaped += res.reward;
                steps.push_back(rec);
                done = res.done;
            }
            game_won[g] = (task.cards_left(0) <= 0) ? 1 : 0;
            game_reward[g] = 0.7 * game_won[g] - 0.7 * (1 - game_won[g]) + 0.02 * shaped;
            total_games++; total_wins += game_won[g];
        }
        // 组相对优势 (GRPO 核心)
        double mean = 0; for (double r : game_reward) mean += r; mean /= group;
        double var = 0; for (double r : game_reward) var += (r - mean) * (r - mean);
        double sd = std::sqrt(var / group) + 1e-6;
        // 逐决策重放反传 (当前权重 ≈ rollout 权重, on-policy)
        BPTTGradients grads;
        for (auto& st : steps) {
            double adv = (game_reward[st.game] - mean) / sd;
            org.reset_state(true);
            bptt.reset_tape();
            std::vector<double> in(st.obs.begin(), st.obs.end());
            org.forward_nd(in.data(), in.size(), false);
            std::vector<std::vector<float>> tgts = {{(float)st.chosen, (float)adv, st.old_prob, CLIP, ENT}};
            bptt.backward_with_loss(org, tgts, grads, SubstrateLossType::GRPO_SURROGATE);
        }
        bptt.step_adam(org, grads, LR);
        if (it % 20 == 0 || it == 1)
            printf("[GRPO-rank] iter %d/%d | 近期胜率 %.1f%% (%d 局)\n", it, iters, 100.0 * total_wins / total_games, total_games);
        if (it % 100 == 0) org.save_checkpoint_bin("checkpoints/doudizhu_bc_rank.bin");
    }
    org.save_checkpoint_bin("checkpoints/doudizhu_bc_rank.bin");
    printf("[GRPO-rank] 完成: %d 局 胜率 %.1f%%\n", total_games, 100.0 * total_wins / total_games);
    return 0;
}

// ============================ v5a 候选打分制 ============================
// 数据: 每决策 (obs32, K, K×8 候选特征, label, gid); 教师标签 = 匹配教师实际 (type,rank) 的候选
static int gen_dataset_cand(int games, const char* path) {
    std::ofstream f(path, std::ios::binary);
    const char magic[4] = {'D','D','Z','C'};
    int32_t ver = 3;
    f.write(magic, 4); f.write((char*)&ver, 4);
    int32_t hdr = games; f.write((char*)&hdr, 4);
    long samples = 0, skipped = 0;
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(6600000 + g * 191), 17.5);
        task.set_rank_action_mode(true);
        bool done = false;
        std::vector<std::vector<char>> buf;   // 按局缓冲 (终局回填团队胜负)
        auto write_sample = [&](const std::vector<float>& obs, const std::vector<DouDiZhuCardGameTask::CandPlay>& cs,
                                const std::vector<std::vector<float>>& feats, int label, int32_t gid) {
            std::vector<char> rec;
            rec.insert(rec.end(), (char*)obs.data(), (char*)obs.data() + 144);   // 36 floats (v3)
            int32_t K = (int32_t)cs.size();
            rec.insert(rec.end(), (char*)&K, (char*)&K + 4);
            for (auto& cf : feats) rec.insert(rec.end(), (char*)cf.data(), (char*)cf.data() + 48);
            rec.insert(rec.end(), (char*)&label, (char*)&label + 4);
            rec.insert(rec.end(), (char*)&gid, (char*)&gid + 4);
            int32_t won = -1;
            rec.insert(rec.end(), (char*)&won, (char*)&won + 4);
            buf.push_back(std::move(rec));
            samples++;
        };
        auto flush_game = [&](int32_t won) {
            for (auto& rec : buf) {
                int32_t* w = (int32_t*)(rec.data() + rec.size() - 4);
                *w = won;
                f.write(rec.data(), (std::streamsize)rec.size());
            }
        };
        StepResult res;
        while (!done) {
            // 因果顺序 (会诊 Task 3): 动作前一次性快照 obs/候选/特征
            auto o = task.current_observation();
            auto cands = task.enumerate_candidates(0);
            std::vector<std::vector<float>> feats;
            for (auto& c : cands) feats.push_back(task.candidate_features(c));
            std::vector<float> obs36(o.begin(), o.end());
            for (float v : task.seat_context()) obs36.push_back(v);   // 48 维公开信息 (座次)
            // 教师标签: 任务拷贝只读标注 (不推进真实账本)
            DouDiZhuCardGameTask probe = task;
            int lr = probe.teacher_play_capture();
            auto t_trick = probe.table_trick();
            int label = -1;
            for (size_t i = 0; i < cands.size(); ++i) {
                if (lr < 0) { if (cands[i].type == 0) { label = (int)i; break; } }
                else if (cands[i].type == (int)t_trick.type && cands[i].rank == t_trick.rank) { label = (int)i; break; }
            }
            // 真实 task 执行教师动作 (teacher_play_capture 应用同一确定性动作)
            task.teacher_play_capture();
            res = task.settle_turn();
            if (label < 0) { skipped++; done = res.done; continue; }   // 教师动作不在候选集 (记录为跳过, 无选择偏差风险)
            write_sample(obs36, cands, feats, label, (int32_t)g);
            done = res.done;
        }
        flush_game(res.success ? 1 : 0);   // 团队胜负 (农民队友出完也算胜)
    }
    f.close();
    printf("[v5数据] %d 局 -> %ld 样本 (跳过 %ld, won=团队res.success, 特征=动作前快照)\n-> %s\n", games, samples, skipped, path);
    return (int)samples;
}

// ---------- 共享: 按头定位 (param2), 不再依赖倒数第几个细胞 ----------
static size_t find_head_by_channel(const CellularOrganism& org, double ch) {
    for (size_t i = 0; i < org.cells.size(); ++i)
        if (org.cells[i].type == CellType::ACT_CHANNEL && org.cells[i].param2 == ch) return i;
    return (size_t)-1;
}
static void build_scorer_input(const std::vector<float>& obs, const std::vector<float>& cf, std::vector<double>& in) {
    in.resize(48);   // v3: 36 obs (32 + 4 座次) + 12 候选交互
    for (int d = 0; d < 36; ++d) in[d] = obs[d];
    for (int d = 0; d < 12; ++d) in[36 + d] = cf[d];
    // 受控消融 (会诊裁决): V5_ZERO_SEAT=1 → 座次 4 通道置零 (原图接线完全一致, 仅信息缺失)
    if (std::getenv("V5_ZERO_SEAT"))
        for (int d = 32; d < 36; ++d) in[d] = 0.0;
}
// 分数头专用 MSE (价值头导数恒 0 — 不用 target=0 冒充冻结)
static const SubstrateLossFn kScoreOnlyLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    float diff = preds[0] - targets[0];
    g.loss_val = diff * diff;
    g.dL_dout.resize(preds.size(), 0.0f);
    g.dL_dout[0] = 2.0f * diff;
    return g;
};
// listwise 候选 CE: dL/dscore = p_i - 1[i=label] (targets[0] 携带 -log p_label 供日志)
static std::vector<float> g_listwise_delta{0.0f};
static const SubstrateLossFn kListwiseLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    g.loss_val = targets.empty() ? 0.0f : targets[0];
    g.dL_dout.assign(preds.size(), 0.0f);
    if (!g_listwise_delta.empty()) g.dL_dout[0] = g_listwise_delta[0];
    return g;
};
// JSON 模型加载 (返回值! bin 为 static 量化格式会破坏通道号 → 拒绝)
static bool load_scorer_json(const char* path, CellularOrganism& org) {
    auto loaded = CellularOrganism::load_checkpoint_json(path);
    if (loaded.cells.empty()) { fprintf(stderr, "[错误] 模型加载失败: %s\n", path); return false; }
    org = std::move(loaded);
    return true;
}
static double score_candidate(CellularOrganism& org, size_t score_head,
                              const std::vector<float>& obs, const std::vector<float>& cf) {
    std::vector<double> in;
    build_scorer_input(obs, cf, in);
    org.reset_state(false);   // 修复: reset(true) 会恢复祖先权重 (抹掉训练)!
    org.forward_nd(in.data(), in.size(), false);
    return org.cells[score_head].output_val;
}

// ---------- DAgger: 评分器驱动 + 因果快照 + 团队 won ----------
static int gen_dataset_cand_dagger(int games, const char* path, const char* model_path) {
    CellularOrganism org = build_doudizhu_candidate_scorer();
    if (!load_scorer_json(model_path, org)) return 1;
    size_t score_head = find_head_by_channel(org, 0.0);
    if (score_head == (size_t)-1) { fprintf(stderr, "[错误] 找不到分数头\n"); return 1; }
    std::ofstream f(path, std::ios::binary);
    const char magic[4] = {'D','D','Z','C'};
    int32_t ver = 3;
    f.write(magic, 4); f.write((char*)&ver, 4);
    int32_t hdr = games; f.write((char*)&hdr, 4);
    long samples = 0, skipped = 0;
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(8800000 + g * 233), 17.5);
        bool done = false;
        std::vector<std::vector<char>> buf;
        auto write_sample = [&](const std::vector<float>& obs, const std::vector<DouDiZhuCardGameTask::CandPlay>& cs,
                                const std::vector<std::vector<float>>& feats, int label, int32_t gid) {
            std::vector<char> rec;
            rec.insert(rec.end(), (char*)obs.data(), (char*)obs.data() + 144);   // 36 floats (v3)
            int32_t K = (int32_t)cs.size();
            rec.insert(rec.end(), (char*)&K, (char*)&K + 4);
            for (auto& cf : feats) rec.insert(rec.end(), (char*)cf.data(), (char*)cf.data() + 48);
            rec.insert(rec.end(), (char*)&label, (char*)&label + 4);
            rec.insert(rec.end(), (char*)&gid, (char*)&gid + 4);
            int32_t won = -1;
            rec.insert(rec.end(), (char*)&won, (char*)&won + 4);
            buf.push_back(std::move(rec));
            samples++;
        };
        auto flush_game = [&](int32_t won) {
            for (auto& rec : buf) {
                int32_t* w = (int32_t*)(rec.data() + rec.size() - 4);
                *w = won;
                f.write(rec.data(), (std::streamsize)rec.size());
            }
        };
        StepResult res;
        while (!done) {
            // 动作前快照
            auto o = task.current_observation();
            auto cands = task.enumerate_candidates(0);
            std::vector<std::vector<float>> feats;
            for (auto& c : cands) feats.push_back(task.candidate_features(c));
            std::vector<float> obs36(o.begin(), o.end());
            for (float v : task.seat_context()) obs36.push_back(v);   // 48 维公开信息
            // 模型选候选
            float best = -1e30f; size_t best_i = 0; bool any_finite = false;
            for (size_t i = 0; i < cands.size(); ++i) {
                double sc = score_candidate(org, score_head, obs36, feats[i]);
                if (!std::isfinite(sc)) { fprintf(stderr, "[错误] 非有限分数, 终止\n"); return 1; }
                if (!any_finite || sc > best) { best = (float)sc; best_i = i; any_finite = true; }
            }
            // 教师标签: 拷贝只读标注 (模型动作未应用前)
            DouDiZhuCardGameTask probe = task;
            int lr = probe.teacher_play_capture();
            auto t_trick = probe.table_trick();
            int label = -1;
            for (size_t i = 0; i < cands.size(); ++i) {
                if (lr < 0) { if (cands[i].type == 0) { label = (int)i; break; } }
                else if (cands[i].type == (int)t_trick.type && cands[i].rank == t_trick.rank) { label = (int)i; break; }
            }
            res = task.play_candidate(cands[best_i]);   // 模型驱动
            if (label >= 0) write_sample(obs36, cands, feats, label, (int32_t)g);
            else skipped++;
            done = res.done;
        }
        flush_game(res.success ? 1 : 0);
    }
    f.close();
    printf("[v5-DAgger] %d 局 -> %ld 样本 (跳过 %ld)\n-> %s\n", games, samples, skipped, path);
    return 0;
}

// ---------- 纯 BC 训练: 梯度正确聚合 (K 候选求和/÷K, 每决策一次 Adam) ----------
static int train_bc_cand(const char* path) {
    CellularOrganism org = build_doudizhu_candidate_scorer();
    CellularBPTTEngine bptt;
    bptt.init_optimizer(org);
    std::ifstream f(path, std::ios::binary);
    char magic[4]; int32_t ver = 0;
    f.read(magic, 4); f.read((char*)&ver, 4);
    if (ver == 0 && std::memcmp(magic, "DDZC", 4) != 0) {
        // 旧格式 (无版本) — 显式拒绝 (会诊 Task 3.3)
        fprintf(stderr, "[错误] 旧版无版本数据 (需 v3 DDZC): %s\n", path); return 1;
    }
    if (ver != 3) { fprintf(stderr, "[错误] 数据版本 %d != 3\n", ver); return 1; }
    int32_t games; f.read((char*)&games, 4);
    struct Sample { std::array<float, 32> obs; std::array<float, 4> seat; std::vector<std::array<float, 12>> cands; int label; int32_t gid; int won; };
    std::vector<Sample> data;
    while (f.good()) {
        Sample s;
        f.read((char*)s.obs.data(), 128);
        f.read((char*)s.seat.data(), 16);
        int32_t K; f.read((char*)&K, 4);
        if (!f.good() || K <= 0 || K > 64) break;
        s.cands.resize(K);
        for (int i = 0; i < K; ++i) f.read((char*)s.cands[i].data(), 48);
        f.read((char*)&s.label, 4);
        f.read((char*)&s.gid, 4);
        f.read((char*)&s.won, 4);
        if (!f.good() || s.label < 0 || s.label >= K) break;
        data.push_back(std::move(s));
    }
    f.close();
    // DAgger 混合: 追加教师数据防遗忘 (V5_TEACHER_DATA)
    if (const char* extra = std::getenv("V5_TEACHER_DATA")) {
        std::ifstream f2(extra, std::ios::binary);
        char m2[4]; int32_t v2; f2.read(m2, 4); f2.read((char*)&v2, 4);
        if (v2 != 3) { fprintf(stderr, "[错误] 教师混合数据非 v3\n"); return 1; }
        int32_t g2; f2.read((char*)&g2, 4);
        long extra_n = 0;
        while (f2.good()) {
            Sample s;
            f2.read((char*)s.obs.data(), 128);
            f2.read((char*)s.seat.data(), 16);
            int32_t K; f2.read((char*)&K, 4);
            if (!f2.good() || K <= 0 || K > 64) break;
            s.cands.resize(K);
            for (int i = 0; i < K; ++i) f2.read((char*)s.cands[i].data(), 48);
            f2.read((char*)&s.label, 4);
            f2.read((char*)&s.gid, 4);
            f2.read((char*)&s.won, 4);
            if (!f2.good() || s.label < 0 || s.label >= K) break;
            data.push_back(std::move(s));
            extra_n++;
        }
        printf("[v5训练] 载入 %zu 决策 (含教师混合 %ld)\n", data.size(), extra_n);
    } else {
        printf("[v5训练] 载入 %zu 决策\n", data.size());
    }
    size_t score_head = find_head_by_channel(org, 0.0);
    size_t value_head = find_head_by_channel(org, 1.0);
    auto obs_of = [](const Sample* s) {
        std::vector<float> o(s->obs.begin(), s->obs.end());
        for (float v : s->seat) o.push_back(v);
        return o;   // 36 维
    };
    // 初始基线
    {
        long correct0 = 0;
        for (auto& s : data) {
            float best = -1e30f; int best_i = 0;
            for (size_t i = 0; i < s.cands.size(); ++i) {
                std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
                auto obs = obs_of(&s);
                double sc = score_candidate(org, score_head, obs, cf);
                if (sc > best) { best = (float)sc; best_i = i; }
            }
            if (best_i == s.label) correct0++;
        }
        printf("[v5初始基线] 单步复现 %.1f%%\n", 100.0 * correct0 / data.size());
    }
    double ce_sum = 0; long ce_cnt = 0;
    const int EPOCHS = std::getenv("V5_EPOCHS") ? std::atoi(std::getenv("V5_EPOCHS")) : 15;
    const float LR = std::getenv("V5_LR") ? (float)std::atof(std::getenv("V5_LR")) : 0.02f;
    const size_t MAXD = std::getenv("V5_MAXD") ? (size_t)std::atoll(std::getenv("V5_MAXD")) : data.size();
    const bool LISTWISE = std::getenv("V5_LOSS") && std::string(std::getenv("V5_LOSS")) == "ce";
    if (MAXD < data.size()) { data.resize(MAXD); printf("[微型] 截取 %zu 决策\n", MAXD); }
    std::mt19937 rng(42);
    for (int ep = 1; ep <= EPOCHS; ++ep) {
        std::shuffle(data.begin(), data.end(), rng);
        double mse_sum = 0; long mse_cnt = 0;
        for (auto& s : data) {
            BPTTGradients sum, one;
            sum.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
            sum.grad_gains.assign(org.cells.size(), 0.0f);
            const float inv_k = 1.0f / (float)s.cands.size();
            auto obs = obs_of(&s);
            if (LISTWISE) {
                // 第一遍: K 个分数 + softmax (log-sum-exp 稳定)
                std::vector<double> scores(s.cands.size());
                for (size_t i = 0; i < s.cands.size(); ++i) {
                    std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
                    scores[i] = score_candidate(org, score_head, obs, cf);
                }   // obs 36 维
                double mx = *std::max_element(scores.begin(), scores.end());
                double Z = 0; for (double v : scores) Z += std::exp(v - mx);
                std::vector<double> p(s.cands.size());
                for (size_t i = 0; i < s.cands.size(); ++i) p[i] = std::exp(scores[i] - mx) / Z;
                double ce = -std::log(std::max(p[s.label], 1e-9));
                // 第二遍: 逐候选举录带反传 (权重未变, 前向确定性)
                for (size_t i = 0; i < s.cands.size(); ++i) {
                    std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
                    std::vector<double> in;
                    build_scorer_input(obs, cf, in);
                    org.reset_state(false);
                    bptt.reset_tape();
                    org.forward_nd(in.data(), in.size(), false);
                    bptt.record_step(org);
                    g_listwise_delta[0] = (float)(p[i] - ((int)i == s.label ? 1.0 : 0.0));
                    std::vector<std::vector<float>> tgts = {{(float)ce}};
                    bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kListwiseLoss);
                    for (size_t j = 0; j < sum.grad_synapses.size(); ++j)
                        sum.grad_synapses[j] += one.grad_synapses[j];   // 候选 CE 梯度不除 K (会诊 4.9)
                    for (size_t j = 0; j < sum.grad_gains.size(); ++j)
                        sum.grad_gains[j] += one.grad_gains[j];
                }
                ce_sum += ce; ce_cnt++;
                bptt.step_adam(org, sum, LR);
                continue;
            }
            for (size_t i = 0; i < s.cands.size(); ++i) {
                std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
                std::vector<double> in;
                build_scorer_input(obs, cf, in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                float target = ((int)i == s.label) ? 1.0f : -0.2f;
                std::vector<std::vector<float>> tgts = {{target}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kScoreOnlyLoss);
                double d = org.cells[score_head].output_val - target;
                mse_sum += d * d; mse_cnt++;
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j)
                    sum.grad_synapses[j] += inv_k * one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j)
                    sum.grad_gains[j] += inv_k * one.grad_gains[j];
            }
            // 诊断: 梯度范数 + 权重位移
            double gnorm = 0, wshift = 0;
            for (size_t j = 0; j < sum.grad_synapses.size(); ++j) gnorm += sum.grad_synapses[j] * sum.grad_synapses[j];
            gnorm = std::sqrt(gnorm);
            for (size_t j = 0; j < org.compiled_synapses_.size(); ++j)
                wshift += std::abs(org.compiled_synapses_[j].weight - org.compiled_synapses_[j].initial_weight);
            if (ep == 1 && &data[0] == &s) {
                printf("[诊断] 首决策: 梯度范数 %.4f | 头入权样本:", gnorm);
                size_t sh = find_head_by_channel(org, 0.0);
                long cnt = 0;
                for (size_t j = 0; j < org.compiled_synapses_.size() && cnt < 5; ++j)
                    if (org.compiled_synapses_[j].to_idx == sh) { printf(" %.4f", org.compiled_synapses_[j].weight); cnt++; }
                printf("\n");
            }
            bptt.step_adam(org, sum, LR);
        }
        if (ep % 5 == 0 || ep == 1) {
            double wshift_total = 0;
            for (size_t j = 0; j < org.compiled_synapses_.size(); ++j)
                wshift_total += std::abs(org.compiled_synapses_[j].weight - org.compiled_synapses_[j].initial_weight);
            if (LISTWISE)
                printf("[v5训练-CE] Epoch %d/%d | 候选CE %.4f | 累计权重位移 %.4f\n", ep, EPOCHS, ce_sum / std::max(1L, ce_cnt), wshift_total);
            else
                printf("[v5训练] Epoch %d/%d | 候选MSE %.4f | 累计权重位移 %.4f\n", ep, EPOCHS, mse_sum / std::max(1L, mse_cnt), wshift_total);
        }
    }
    // 训练后单步 (44 维一致)
    long correct = 0;
    for (auto& s : data) {
        float best = -1e30f; int best_i = 0;
        for (size_t i = 0; i < s.cands.size(); ++i) {
            std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
            auto obs = obs_of(&s);
            double sc = score_candidate(org, score_head, obs, cf);
            if (sc > best) { best = (float)sc; best_i = i; }
        }
        if (best_i == s.label) correct++;
    }
    printf("[v5单步] 候选复现 %.1f%% (%ld/%zu)\n", 100.0 * correct / data.size(), correct, data.size());
    // JSON 保存 + 往返门禁 (会诊 Task 1.5: 保存前后分数一致)
    const char* out = "checkpoints/doudizhu_cand_scorer.json";
    if (!org.save_checkpoint_json(out)) { fprintf(stderr, "[错误] 保存失败\n"); return 1; }
    {
        CellularOrganism rt;
        if (!load_scorer_json(out, rt)) return 1;
        size_t sh2 = find_head_by_channel(rt, 0.0);
        long mismatch = 0;
        for (size_t si = 0; si < data.size() && si < 50; ++si) {
            auto& s = data[si];
            auto obs = obs_of(&s);
            for (size_t i = 0; i < s.cands.size(); ++i) {
                std::vector<float> cf(s.cands[i].begin(), s.cands[i].end());
                double a = score_candidate(org, score_head, obs, cf);
                double b = score_candidate(rt, sh2, obs, cf);
                if (std::abs(a - b) > 1e-4 * (1.0 + std::abs(a))) mismatch++;
            }
        }
        printf("[往返门禁] %ld/%ld 分点超差 %s\n", mismatch, std::min<size_t>(50, data.size()), mismatch ? "→ FAIL" : "→ PASS");
    }
    printf("[产物] %s\n", out);
    return 0;
}

// ---------- 统一基线评测协议 (会诊 Task 5): teacher/first/uniform/init/model ----------
static double wilson_lower(int wins, int n) {
    if (n == 0) return 0.0;
    double z = 1.959964, p = (double)wins / n, n2 = (double)n * n;
    double center = (p + z * z / (2 * n)) / (1 + z * z / n);
    double half = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n2)) / (1 + z * z / n);
    return center - half;
}
static int eval_cand_games(int games, const char* policy_arg) {
    std::string policy = policy_arg;
    CellularOrganism org;
    size_t score_head = (size_t)-1;
    bool use_model = false, use_teacher = false;
    if (policy == "teacher") use_teacher = true;
    else if (policy == "init") { org = build_doudizhu_candidate_scorer(); }
    else if (policy == "first" || policy == "uniform") { org = build_doudizhu_candidate_scorer(); }   // 修复: 规则策略曾被误分类为 model
    else if (policy == "model") { }
    else { policy = "model"; }
    if (policy == "model") {
        // main 已把路径传入 policy_arg? 约定: eval_cand <games> model <path>
        use_model = true;
    }
    // 模型路径由外部变量传递 (eval_model_path)
    extern std::string g_eval_model_path;
    if (use_model) {
        if (!load_scorer_json(g_eval_model_path.c_str(), org)) return 1;
        score_head = find_head_by_channel(org, 0.0);
        if (score_head == (size_t)-1) { fprintf(stderr, "[错误] 找不到分数头\n"); return 1; }
    } else if (!use_teacher) {
        score_head = find_head_by_channel(org, 0.0);
    }
    uint32_t lcg = 424243u;
    auto rnd01 = [&]() { lcg = lcg * 1664525u + 1013904223u; return (double)(lcg >> 8) / 16777216.0; };
    int wins = 0; long live_pass = 0, live_steps = 0;
    int ll_w = 0, ll_n = 0, fm_w = 0, fm_n = 0;   // 角色分层 (诊断协议)
    for (int g = 0; g < games; ++g) {
        DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        bool done = false;
        bool agent_is_landlord = false;
        {
            // 首个决策前的角色探测 (叫牌已由构造完成)
            DouDiZhuCardGameTask probe = task;
            int lr0 = probe.teacher_play_capture();
            (void)lr0;
            agent_is_landlord = (probe.role() == 1);
        }
        while (!done) {
            auto cands = task.enumerate_candidates(0);
            size_t pick = 0;
            if (use_teacher) {
                task.teacher_play_capture();
                auto res = task.settle_turn();
                live_steps++;
                done = res.done;
                if (res.success) wins++;
                if (done) {
                    if (agent_is_landlord) { ll_n++; if (res.success) ll_w++; }
                    else { fm_n++; if (res.success) fm_w++; }
                }
                continue;
            } else if (policy == "first") {
                pick = 0;
            } else if (policy == "uniform") {
                pick = (size_t)((int)(rnd01() * (double)cands.size()) % cands.size());
            } else {
                auto o = task.current_observation();
                std::vector<float> obs32(o.begin(), o.end());
                float best = -1e30f;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto cf = task.candidate_features(cands[i]);
                    double sc = score_candidate(org, score_head, obs32, cf);
                    if (!std::isfinite(sc)) { fprintf(stderr, "[错误] 非有限分数\n"); return 1; }
                    if (sc > best) { best = (float)sc; pick = i; }
                }
            }
            int pre = task.cards_left(0);
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            if (res.success) wins++;
            live_steps++;
            if (task.cards_left(0) == pre) live_pass++;
            if (done) {   // 角色分层按局计数 (修复: 此前按决策步数误计)
                if (agent_is_landlord) { ll_n++; if (res.success) ll_w++; }
                else { fm_n++; if (res.success) fm_w++; }
            }
        }
    }
    printf("[角色分层] 地主 %d/%d = %.1f%% | 农民 %d/%d = %.1f%%\n",
           ll_w, ll_n, ll_n ? 100.0 * ll_w / ll_n : 0.0, fm_w, fm_n, fm_n ? 100.0 * fm_w / fm_n : 0.0);
    printf("[v5实战/%s] %d 局: 胜率 %.1f%% (%d/%d) Wilson95下界 %.1f%% | 步数 %ld 过牌 %.1f%%\n",
           policy_arg, games, 100.0 * wins / games, wins, games, 100.0 * wilson_lower(wins, games),
           live_steps, 100.0 * live_pass / std::max(1L, live_steps));
    return 0;
}
std::string g_eval_model_path = "checkpoints/doudizhu_cand_scorer.json";

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "train";
    if (mode == "eval_rank") {
        int games = argc > 2 ? std::atoi(argv[2]) : 500;
        eval_rank_games(games, argc > 3 ? argv[3] : "checkpoints/doudizhu_bc_rank.bin");
        return 0;
    }
    if (mode == "gen_cand_dagger") {
        gen_dataset_cand_dagger(argc > 2 ? std::atoi(argv[2]) : 2000, "/tmp/opencode/doudizhu_cand_dagger.bin",
                                argc > 3 ? argv[3] : "checkpoints/doudizhu_cand_scorer.json");
        return 0;
    }
    if (mode == "gen_cand") {
        gen_dataset_cand(argc > 2 ? std::atoi(argv[2]) : 2000, "/tmp/opencode/doudizhu_cand.bin");
        return 0;
    }
    if (mode == "train_cand") {
        train_bc_cand(argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_cand.bin");
        return 0;
    }
    if (mode == "audit_conflicts") {
        // 会诊裁决: e_alias = Σ(n_g - max_a n_g,a)/N, 键 = 量化obs + 规范候选集合, 标签=(type,rank,count)
        std::ifstream f(argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_cand.bin", std::ios::binary);
        char magic[4]; int32_t ver; f.read(magic, 4); f.read((char*)&ver, 4);
        if (ver != 3) { fprintf(stderr, "[错误] 需 v3 数据\n"); return 1; }
        int32_t games; f.read((char*)&games, 4);
        std::map<std::string, std::map<std::string, long>> groups;   // 决策键 → 标签键 → 计数
        long N = 0;
        while (f.good()) {
            float ob[36]; int32_t K, label, gid, won;
            f.read((char*)ob, 144);
            f.read((char*)&K, 4);
            if (!f.good() || K <= 0 || K > 64) break;
            std::vector<std::array<float, 12>> cs(K);
            for (int i = 0; i < K; ++i) f.read((char*)cs[i].data(), 48);
            f.read((char*)&label, 4); f.read((char*)&gid, 4); f.read((char*)&won, 4);
            if (!f.good()) break;
            char key[512]; int off = 0;
            for (int d = 0; d < 36; ++d) off += snprintf(key + off, sizeof(key) - off, "%d,", (int)std::lround(ob[d] * 1000));
            std::vector<std::string> cand_keys;
            for (int i = 0; i < K; ++i) {
                char ck[64];
                snprintf(ck, sizeof(ck), "%d-%d-%d", (int)cs[i][0] ? 1 : (cs[i][1] ? 2 : (cs[i][2] ? 3 : (cs[i][3] ? 4 : 0))),
                         (int)std::lround(cs[i][5] * 14), (int)std::lround(cs[i][6] * 4));
                cand_keys.push_back(ck);
            }
            std::sort(cand_keys.begin(), cand_keys.end());
            for (auto& ck : cand_keys) { off += snprintf(key + off, sizeof(key) - off, "%s;", ck.c_str()); }
            char lk[64];
            snprintf(lk, sizeof(lk), "%d-%d-%d", (int)cs[label][0] ? 1 : (cs[label][1] ? 2 : (cs[label][2] ? 3 : (cs[label][3] ? 4 : 0))),
                     (int)std::lround(cs[label][5] * 14), (int)std::lround(cs[label][6] * 4));
            groups[key][lk]++;
            N++;
        }
        f.close();
        long alias_err = 0, dup_groups = 0, dup_n = 0;
        for (auto& [k, labels] : groups) {
            long n = 0, mx = 0;
            for (auto& [lk, c] : labels) { n += c; mx = std::max(mx, c); }
            alias_err += n - mx;
            if (n > 1) { dup_groups++; dup_n += n; }
        }
        printf("[冲突审计] N=%ld 组=%ld | e_alias=%.2f%% | 重复组 %ld 个 (覆盖 %ld 样本, %.1f%%)\n",
               N, (long)groups.size(), 100.0 * alias_err / std::max(1L, N), dup_groups, dup_n,
               100.0 * dup_n / std::max(1L, N));
        return 0;
    }
    if (mode == "eval_cand") {
        int games = argc > 2 ? std::atoi(argv[2]) : 500;
        std::string pol = argc > 3 ? argv[3] : "model";
        if (pol == "model") g_eval_model_path = argc > 4 ? argv[4] : "checkpoints/doudizhu_cand_scorer.json";
        eval_cand_games(games, pol.c_str());
        return 0;
    }
    if (mode == "eval_hybrid") {
        int games = argc > 2 ? std::atoi(argv[2]) : 500;
        float theta = argc > 3 ? (float)std::atof(argv[3]) : 1.0f;
        eval_hybrid_games(games, theta, argc > 4 ? argv[4] : "checkpoints/doudizhu_bc_rank.bin");
        return 0;
    }
    if (mode == "train_rank_grpo") {
        int iters = argc > 2 ? std::atoi(argv[2]) : 300;
        int group = argc > 3 ? std::atoi(argv[3]) : 8;
        train_rank_grpo(iters, group);
        return 0;
    }
    if (mode == "gen_dagger") {
        int games = argc > 2 ? std::atoi(argv[2]) : 2000;
        gen_dataset_dagger(games, "/tmp/opencode/doudizhu_bc_dagger.bin");
        return 0;
    }
    if (mode == "gen_rank") {
        int games = argc > 2 ? std::atoi(argv[2]) : 2000;
        gen_dataset_rank(games, "/tmp/opencode/doudizhu_bc_dataset_rank.bin");
        return 0;
    }
    if (mode == "train_rank") {
        train_bc_rank(argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_bc_dataset_rank.bin");
        return 0;
    }
    if (mode == "gen") {
        int games = argc > 2 ? std::atoi(argv[2]) : 2000;
        gen_dataset(games, "/tmp/opencode/doudizhu_bc_dataset.bin");
    } else {
        train_bc(argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_bc_dataset.bin");
    }
    return 0;
}

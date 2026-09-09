// P9-GROW (M10): 结构级进化 — 活动引导的功能等价分裂 (铁律: 细胞/突触无上限)
// 协议: docs/superpowers/plans/2026-09-09-online-rl-protocol.md §M10 (预注册)
//   - 每 grow_every 局: 欠票边(打分头入边中源激活最低者) → 插入 OP_SUM 恒等细胞
//     (from→new(原权) + new→to(1.0), 出生时函数逐位保持, 数值验证)
//   - 探测门: 500 局 holdout ≥ t0−2pp 接受, 否则换靶再试; 接受后 Adam 重置
//   - 躯体 82 → 无上限生长; 其余配方 = M2 课程 + 选择回退 + 池质量门
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/statistical_evaluation.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace kun;

namespace {

double wilson_lower(int wins, int n) {
    if (n == 0) return 0.0;
    const double z = 1.96, p = (double)wins / n;
    const double d = 1 + z * z / n;
    const double center = p + z * z / (2 * n);
    const double rad = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return (center - rad) / d;
}

size_t find_head_by_channel(const CellularOrganism& org, double ch) {
    for (size_t i = 0; i < org.cells.size(); ++i)
        if (org.cells[i].type == CellType::ACT_CHANNEL && org.cells[i].param2 == ch) return i;
    return (size_t)-1;
}

void build_scorer_input(const std::vector<float>& obs, const std::vector<float>& cf, std::vector<double>& in) {
    in.assign(56, 0.0);
    for (int d = 0; d < 44; ++d) in[d] = obs[d];
    for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
}

double score_org(CellularOrganism& org, size_t head, const std::vector<float>& obs, const std::vector<float>& cf) {
    std::vector<double> in;
    build_scorer_input(obs, cf, in);
    org.reset_state(false);
    org.forward_nd(in.data(), in.size(), false);
    return org.cells[head].output_val;
}

static std::vector<float> g_adv_delta{0.0f, 0.0f};
static const SubstrateLossFn kAdvLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    g.loss_val = targets.empty() ? 0.0f : targets[0];
    g.dL_dout.assign(preds.size(), 0.0f);
    if (!g_adv_delta.empty()) g.dL_dout[0] = g_adv_delta[0];
    if (g.dL_dout.size() > 1) g.dL_dout[1] = g_adv_delta[1];
    return g;
};

// 模型选候选 (池对手): 座位参数化观测 + 特征
size_t model_pick(CellularOrganism& org, size_t head, const DouDiZhuCardGameTask& task, int p,
                  const std::vector<DouDiZhuCardGameTask::CandPlay>& cands) {
    auto obs = task.observation44_for(p);
    size_t pick = 0;
    double best = -1e308;
    for (size_t i = 0; i < cands.size(); ++i) {
        auto cf = task.candidate_features_for(p, cands[i]);
        const double sc = score_org(org, head, obs, cf);
        if (sc > best) { best = sc; pick = i; }
    }
    return pick;
}

struct EvalResult {
    int wins = 0;
    long pass = 0, steps = 0;
    std::vector<uint8_t> won;
};

// 评测协议 (与 v2 eval_family 完全一致: seat 0 模型, 1/2 教师)
EvalResult eval_family(CellularOrganism& org, size_t head, int games, uint32_t seed_base) {
    EvalResult out;
    out.won.reserve(games);
    for (int g = 0; g < games; ++g) {
        kun::DouDiZhuCardGameTask task(40, seed_base + (uint32_t)(g * 97), 17.5);
        bool done = false;
        {
            kun::DouDiZhuCardGameTask probe = task;
            int lr0 = probe.teacher_play_capture();
            (void)lr0;
        }
        bool game_won = false;
        while (!done) {
            auto cands = task.enumerate_candidates(0);
            size_t pick = 0;
            {
                auto o = task.current_observation();
                std::vector<float> obs44(o.begin(), o.end());
                for (float v : task.seat_context()) obs44.push_back(v);
                double best = -1e308;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto cf = task.candidate_features(cands[i]);
                    const double sc = score_org(org, head, obs44, cf);
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            int pre = task.cards_left(0);
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            game_won = res.success;
            out.steps++;
            if (task.cards_left(0) == pre) out.pass++;
        }
        out.wins += game_won ? 1 : 0;
        out.won.push_back(game_won ? 1 : 0);
    }
    return out;
}

// ── M10 结构进化: 活动引导功能等价分裂 ──
// 返回: 0=成功分裂(函数恒等保持) 1=恒等验证失败(已回滚) 2=无靶边
int attempt_structural_split(
    CellularOrganism& org, size_t score_head,
    const std::deque<std::vector<double>>& act_buffer,
    uint32_t new_cell_id) {
    if (act_buffer.size() < 32) return 2;
    // 1. 活动测量: 每细胞 mean|output| (最近决策输入)
    std::vector<double> activation(org.cells.size(), 0.0);
    for (const auto& in : act_buffer) {
        org.reset_state(false);
        org.forward_nd(in.data(), in.size(), false);
        for (size_t i = 0; i < org.cells.size(); ++i)
            activation[i] += std::abs(org.cells[i].output_val);
    }
    for (double& a : activation) a /= (double)act_buffer.size();
    // 2. 欠票边 (M10 修订): ACT 头是带 latch 的有状态效应器 — head 入边分裂
    //    会摧毁 latch 状态机 (实证: 端口和恒等但输出 180→3)。生长靶改为
    //    特征层入边 (SENSE→特征细胞, 纯代数无状态): 最不活跃特征细胞的最弱源边
    const uint32_t head_id = org.cells[score_head].id;
    uint32_t max_id = 0;
    for (const auto& c : org.cells) max_id = std::max(max_id, c.id);
    std::vector<uint32_t> id2idx((size_t)max_id + 1, 0);   // 按 ID 索引 (ID 可达 201+)
    for (size_t i = 0; i < org.cells.size(); ++i) id2idx[org.cells[i].id] = (uint32_t)i;
    std::vector<double> feature_act;
    std::vector<uint32_t> feature_ids;
    for (const auto& c : org.cells)
        if (c.id >= 100 && c.id < 500 && (c.type == CellType::OP_SUM || c.type == CellType::OP_ABS)) {
            feature_ids.push_back(c.id);
            feature_act.push_back(activation[id2idx[c.id]]);
        }
    if (feature_ids.empty()) return 2;
    size_t fi = 0;
    { double worst_f = 1e308;
      for (size_t k = 0; k < feature_ids.size(); ++k)
          if (feature_act[k] < worst_f) { worst_f = feature_act[k]; fi = k; } }
    const uint32_t target_feature = feature_ids[fi];
    int target_raw = -1;
    double worst = 1e308;
    uint32_t from_id = 0;
    double w_orig = 0.0;
    for (size_t si = 0; si < org.synapses.size(); ++si) {
        const auto& s = org.synapses[si];
        if (!s.is_active || s.to_cell_id != target_feature) continue;
        const double act = activation[id2idx[s.from_cell_id]];
        if (act < worst) { worst = act; target_raw = (int)si; from_id = s.from_cell_id; w_orig = s.weight; }
    }
    if (target_raw < 0) return 2;
    // 3. 分裂: 原 from→head 灭活; from→new(w_orig) + new→head(1.0); OP_SUM g=1.0 恒等
    CellularOrganism backup = org;
    Cell nc;
    nc.id = new_cell_id; nc.type = CellType::OP_SUM;
    nc.param1 = 1.0; nc.param2 = 0.0;
    nc.x = 60.0f;   // 空间坐标 (培养皿), 其余物理属性默认
    org.cells.push_back(nc);
    Synapse s1, s2;
    s1.from_cell_id = from_id; s1.to_cell_id = new_cell_id; s1.to_port = 0;
    s1.weight = w_orig; s1.initial_weight = w_orig; s1.is_active = true;
    s1.rest_length = 50.0f; s1.photon_pos = -1.0f;
    s2.from_cell_id = new_cell_id; s2.to_cell_id = target_feature; s2.to_port = 0;
    s2.weight = 1.0; s2.initial_weight = 1.0; s2.is_active = true;
    s2.rest_length = 50.0f; s2.photon_pos = -1.0f;
    // 修复底座 latent bug 的工具侧规避: legacy step_adam 假设 raw↔compiled 索引 1:1,
    // 而 compile 会过滤非活性突触 → 任何 is_active=false 都会让之后的 Adam 把权重
    // 写进错位的 raw 槽位 (静默基因组污染)。因此分裂必须物理删除而非置非活性。
    org.synapses.erase(org.synapses.begin() + target_raw);
    org.synapses.push_back(s1);
    org.synapses.push_back(s2);
    org.compile();
    // 1:1 对账: raw 活性突触与 compiled 必须逐位对应 (数量 + 权重)
    {
        long active_raw = 0;
        for (const auto& s : org.synapses) if (s.is_active) ++active_raw;
        bool ok = (active_raw == (long)org.compiled_synapses_.size());
        if (ok)
            for (size_t i = 0; i < org.compiled_synapses_.size() && ok; ++i)
                ok = (std::abs(org.compiled_synapses_[i].weight - org.synapses[i].weight) < 1e-12);
        if (!ok) {
            printf("[生长] raw↔compiled 1:1 对账失败 (raw活性=%ld compiled=%zu) → 回滚\n",
                   active_raw, org.compiled_synapses_.size());
            org = std::move(backup);
            return 1;
        }
    }
    // 4. 出生恒等验证 (数值, 首个违例即回滚): 比对整条链路 — 靶特征细胞与最终分数头
    const size_t feat_idx_new = id2idx[target_feature];
    const size_t feat_idx_old = [&] {
        for (size_t i = 0; i < backup.cells.size(); ++i)
            if (backup.cells[i].id == target_feature) return i;
        return (size_t)0;
    }();
    int vrow = 0;
    for (const auto& in : act_buffer) {
        org.reset_state(false);
        org.forward_nd(in.data(), in.size(), false);
        const double s_new = org.cells[score_head].output_val;
        const double f_new = org.cells[feat_idx_new].output_val;
        backup.reset_state(false);
        backup.forward_nd(in.data(), in.size(), false);
        const double s_old = backup.cells[score_head].output_val;   // score_head 索引在两图中一致 (append-only)
        const double f_old = backup.cells[feat_idx_old].output_val;
        if (vrow == 0) {
            long act_before = 0, act_after = 0;
            for (const auto& s : backup.synapses)
                if (s.is_active && s.to_cell_id == target_feature) ++act_before;
            for (const auto& s : org.synapses)
                if (s.is_active && s.to_cell_id == target_feature) ++act_after;
            const size_t nc_idx = org.cells.size() - 1;
            // 逐细胞对比: 找出除新细胞外所有输出不一致的细胞 (compile 波及面)
            int ndiff = 0;
            for (size_t ci = 0; ci < backup.cells.size() && ndiff < 6; ++ci) {
                const double a = org.cells[ci].output_val, b = backup.cells[ci].output_val;
                if (std::abs(a - b) > 1e-6) {
                    printf("[生长诊断] 细胞 %u (idx %zu) 输出 org=%.4f backup=%.4f\n",
                           org.cells[ci].id, ci, a, b);
                    ++ndiff;
                }
            }
            auto port0sum = [&](const CellularOrganism& o, uint32_t hid) {
                std::unordered_map<uint32_t, size_t> m;
                for (size_t i = 0; i < o.cells.size(); ++i) m[o.cells[i].id] = i;
                double sum = 0.0;
                for (const auto& sy : o.synapses)
                    if (sy.is_active && sy.to_cell_id == hid && sy.to_port == 0)
                        sum += sy.weight * o.cells[m[sy.from_cell_id]].output_val;
                return sum;
            };
            printf("[生长诊断] 靶边 %u→特征%u w=%.4f | 特征入边 b/a=%ld/%ld | out_src=%.4f 新细胞out=%.4f | 实际port0 org=%.4f backup=%.4f 手工=%.4f/%.4f | f_old=%.4f f_new=%.4f\n",
                   from_id, target_feature, w_orig, act_before, act_after,
                   org.cells[id2idx[from_id]].output_val,
                   org.cells[nc_idx].output_val,
                   org.flat_port_inputs_[80 * 2 + 0],
                   backup.flat_port_inputs_[80 * 2 + 0],
                   port0sum(org, target_feature), port0sum(backup, target_feature),
                   f_old, f_new);
        }
        if (std::abs(s_new - s_old) > 1e-6 || std::abs(f_new - f_old) > 1e-6) {
            printf("[生长诊断] 失败行 %d: Δhead=%.4f Δfeat=%.4f\n",
                   vrow, std::abs(s_new - s_old), std::abs(f_new - f_old));
            printf("[生长] 恒等验证失败 (靶边 %u→特征%u) → 回滚\n", from_id, target_feature);
            org = std::move(backup);
            return 1;
        }
        ++vrow;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model = "checkpoints/doudizhu_cand_scorer.bin";
    int games = 20000, eval_every = 200, holdout_n = 500, pool_every = 500;
    double lr = 2e-4, r_max = 0.5, r_start = -1.0, value_w = 0.0, lr2 = 0.0;
    int r_period = 1000, lr_after = -1, pool_cap = 128, max_collapses = 8;
    int grow_every = 2000, grow_attempts = 12;
    double accept_tol = 2.0;
    const char* pool_list = "";
    const char* final_out = "checkpoints/pool/m10_final.bin";
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { printf("[错误] --%s 需要参数\n", what); exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--games")) games = std::atoi(need("games"));
        else if (!std::strcmp(argv[i], "--eval-every")) eval_every = std::atoi(need("eval-every"));
        else if (!std::strcmp(argv[i], "--holdout")) holdout_n = std::atoi(need("holdout"));
        else if (!std::strcmp(argv[i], "--pool-every")) pool_every = std::atoi(need("pool-every"));
        else if (!std::strcmp(argv[i], "--lr")) lr = std::atof(need("lr"));
        else if (!std::strcmp(argv[i], "--r-max")) r_max = std::atof(need("r-max"));
        else if (!std::strcmp(argv[i], "--r-start")) r_start = std::atof(need("r-start"));
        else if (!std::strcmp(argv[i], "--r-period")) r_period = std::atoi(need("r-period"));
        else if (!std::strcmp(argv[i], "--value-w")) value_w = std::atof(need("value-w"));
        else if (!std::strcmp(argv[i], "--lr-after")) lr_after = std::atoi(need("lr-after"));
        else if (!std::strcmp(argv[i], "--lr2")) lr2 = std::atof(need("lr2"));
        else if (!std::strcmp(argv[i], "--pool-cap")) pool_cap = std::atoi(need("pool-cap"));
        else if (!std::strcmp(argv[i], "--max-collapses")) max_collapses = std::atoi(need("max-collapses"));
        else if (!std::strcmp(argv[i], "--grow-every")) grow_every = std::atoi(need("grow-every"));
        else if (!std::strcmp(argv[i], "--grow-attempts")) grow_attempts = std::atoi(need("grow-attempts"));
        else if (!std::strcmp(argv[i], "--accept-tol")) accept_tol = std::atof(need("accept-tol"));
        else if (!std::strcmp(argv[i], "--pool-list")) pool_list = need("pool-list");
        else if (!std::strcmp(argv[i], "--final-out")) final_out = need("final-out");
        else if (!std::strcmp(argv[i], "--model")) model = need("model");
    }
    std::string final_path = final_out;
    const uint32_t TRAIN_BASE = 3100000u, HOLD_BASE = 9100000u;

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    const size_t score_head = find_head_by_channel(org, 0.0);
    if (score_head == (size_t)-1) { printf("[错误] 找不到分数头\n"); return 1; }
    const size_t value_head = find_head_by_channel(org, 1.0);
    if (value_w > 0.0 && value_head == (size_t)-1) { printf("[错误] --value-w 需要价值头\n"); return 1; }
    printf("[M2] 起点 cells=%zu lr=%.2e r_max=%.2f\n", org.cells.size(), lr, r_max);

    CellularBPTTEngine bptt(64);
    bptt.init_optimizer(org);

    EvalResult t0 = eval_family(org, score_head, holdout_n, HOLD_BASE);
    printf("[t0 基线/holdout] %d 局: %.1f%% (%d/%d) Wilson 下界 %.1f%% | 过牌 %.1f%%\n",
           holdout_n, 100.0 * t0.wins / holdout_n, t0.wins, holdout_n,
           100.0 * wilson_lower(t0.wins, holdout_n), 100.0 * t0.pass / std::max(1L, t0.steps));

    // 对手池 (M2c: --pool-list 预载快照, 保持课程连续性)
    std::vector<CellularOrganism> pool;
    {
        std::string list(pool_list);
        size_t pos = 0;
        while (!list.empty()) {
            const size_t comma = list.find(',');
            const std::string item = (comma == std::string::npos) ? list : list.substr(0, comma);
            if (comma == std::string::npos) list.clear(); else list.erase(0, comma + 1);
            if (item.empty()) continue;
            CellularOrganism loaded;
            auto bl = CellularOrganism::load_checkpoint_bin(item.c_str());
            if (!bl.cells.empty()) loaded = std::move(bl);
            else loaded = CellularOrganism::load_checkpoint_json(item.c_str());
            if (loaded.cells.empty()) { printf("[错误] 池加载失败: %s\n", item.c_str()); return 1; }
            pool.push_back(std::move(loaded));
            printf("[池预载] %s (池=%zu)\n", item.c_str(), pool.size());
        }
    }
    uint32_t lcg = 20260909u;
    auto rnd01 = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (double)(lcg >> 8) / 16777216.0;
    };

    std::deque<double> outcome_window;
    std::deque<std::vector<double>> act_buffer;   // M10: 最近决策输入 (活动测量)
    int collapses = 0;
    int growth_done = 0, growth_accepted = 0;
    int climb_streak = 0, prev_eval_wins = -1;   // 相位耦合: 连续爬升检测
    int best_wins = -1;
    std::string best_path;
    std::vector<std::string> eval_log;
    bool aborted = false;
    std::string abort_reason;
    long selfplay_decisions = 0, teacher_decisions = 0, student_decisions = 0;

    for (int g = 0; g < games && !aborted; ++g) {
        const double r = (r_start >= 0.0)
            ? std::min(r_max, r_start)
            : std::min(r_max, 0.05 * (double)(g / r_period));
        kun::DouDiZhuCardGameTask task(40, TRAIN_BASE + (uint32_t)(g * 97), 17.5);
        // 座位课程: 1/2 独立掷币 (r 概率池对手; 池空全教师)
        int model_seats = 0;
        if (!pool.empty()) {
            if (rnd01() < r) model_seats |= 1;
            if (rnd01() < r) model_seats |= 2;
        }
        size_t opp_head = (size_t)-1;
        CellularOrganism* opp = nullptr;
        if (model_seats) {
            opp = &pool[(size_t)(rnd01() * (double)pool.size()) % pool.size()];
            opp_head = find_head_by_channel(*opp, 0.0);
        }

        // 自博弈驱动: 全座位外部驱动 (seat 0 = 学生, 模型座位 = 池对手, 其余教师启发式)
        struct Snap { std::vector<float> obs; std::vector<std::vector<float>> feats; size_t chosen; };
        std::vector<Snap> snaps;
        bool done = false;
        bool game_won = false;
        int guard = 0;
        while (!done && guard++ < 400) {
            const int p = task.current_turn();
            if (p == 0) {
                auto cands = task.enumerate_candidates(0);
                auto o = task.current_observation();
                std::vector<float> obs44(o.begin(), o.end());
                for (float v : task.seat_context()) obs44.push_back(v);
                Snap s;
                s.obs = obs44;
                double best = -1e308;
                s.chosen = 0;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto cf = task.candidate_features(cands[i]);
                    s.feats.push_back(cf);
                    const double sc = score_org(org, score_head, obs44, cf);
                    if (sc > best) { best = sc; s.chosen = i; }
                }
                snaps.push_back(std::move(s));
                // M10: 候选输入入活动缓冲 (cap 600 行)
                {
                    const auto& bk = snaps.back();
                    for (size_t ci = 0; ci < bk.feats.size(); ++ci) {
                        std::vector<double> in(56, 0.0);
                        for (int d = 0; d < 44; ++d) in[d] = bk.obs[d];
                        for (int d = 0; d < 12; ++d) in[44 + d] = bk.feats[ci][d];
                        act_buffer.push_back(std::move(in));
                        if (act_buffer.size() > 600) act_buffer.pop_front();
                    }
                }
                task.play_candidate_at(0, cands[s.chosen]);
                if (task.cards_left(0) <= 0) {
                    done = true;
                    game_won = (task.landlord() == 0);   // 座 0 出完: 地主则胜, 农民则友方胜
                }
                student_decisions++;
            } else {
                auto cands = task.enumerate_candidates(p);
                if (cands.empty()) { printf("[错误] 空候选 @座%d\n", p); return 1; }
                if (opp && (model_seats & (1 << p))) {
                    const size_t pick = model_pick(*opp, opp_head, task, p, cands);
                    task.play_candidate_at(p, cands[pick]);
                    selfplay_decisions++;
                } else {
                    task.play_opponent_turn(p);   // 教师启发式执行器 (与评测协议同一执行器)
                    teacher_decisions++;
                }
                if (task.cards_left(p) <= 0) {
                    done = true;
                    game_won = (p != task.landlord());   // p≠0 出完且为农民 → 学生友方胜
                }
            }
        }
        if (!done) game_won = false;   // 轮转守卫兜底 (语义同 max_rounds 判负)

        // 优势加权 CE 一次更新 (M3: --value-w > 0 时启用价值基线 A_t = A − V(s_t))
        const double outcome = game_won ? 1.0 : -1.0;
        outcome_window.push_back(outcome);
        if ((int)outcome_window.size() > 256) outcome_window.pop_front();
        double mean = 0.0;
        for (double v : outcome_window) mean += v;
        mean /= outcome_window.size();
        double var = 0.0;
        for (double v : outcome_window) var += (v - mean) * (v - mean);
        var /= outcome_window.size();
        const double A = (outcome - mean) / std::max(std::sqrt(var), 0.25);
        const double value_target = game_won ? 1.0 : 0.0;

        BPTTGradients sum, one;
        sum.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
        sum.grad_gains.assign(org.cells.size(), 0.0f);
        // 价值基线 (M3, 预注册): V_t 取当前权重的选中行输出, 优势 A_t = A − V_t
        std::vector<double> snap_v(snaps.size(), 0.0);
        if (value_w > 0.0) {
            for (size_t si = 0; si < snaps.size(); ++si) {
                const auto& s = snaps[si];
                snap_v[si] = score_org(org, value_head, s.obs, s.feats[s.chosen]);
                // 价值头 MC 回报回归: delta_v = 2·value_w·(V − won)
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[s.chosen], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_adv_delta[0] = 0.0f;
                g_adv_delta[1] = (float)(2.0 * value_w * (snap_v[si] - value_target));
                std::vector<std::vector<float>> tgts = {{0.0f}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        size_t snap_idx = 0;
        for (const auto& s : snaps) {
            const double A_t = A - snap_v[snap_idx];
            ++snap_idx;
            std::vector<double> scores(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i)
                scores[i] = score_org(org, score_head, s.obs, s.feats[i]);
            const double mx = *std::max_element(scores.begin(), scores.end());
            double Z = 0.0;
            for (double v : scores) Z += std::exp(v - mx);
            std::vector<double> p(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i) p[i] = std::exp(scores[i] - mx) / Z;
            const double ce = -std::log(std::max(p[s.chosen], 1e-9));
            for (size_t i = 0; i < s.feats.size(); ++i) {
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[i], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_adv_delta[0] = (float)(A_t * (p[i] - ((int)i == (int)s.chosen ? 1.0 : 0.0)));
                g_adv_delta[1] = 0.0f;
                std::vector<std::vector<float>> tgts = {{(float)ce}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        bptt.step_adam(org, sum, (float)((lr_after > 0 && g >= lr_after) ? lr2 : lr));

        // M10 结构进化 (相位耦合版): 仅在爬升相位 (连续两次 eval 提升) 触发生长,
        // 且接受后保留 Adam 动量 (按 raw_index 重映射, 新参数零初始化)
        if ((g + 1) % grow_every == 0 && growth_done < grow_attempts && climb_streak >= 2) {
            ++growth_done;
            CellularOrganism pre_grow = org;   // 回滚快照
            // Adam 动量快照 (raw_index → m,v)
            std::unordered_map<size_t, std::pair<float, float>> adam_by_raw;
            for (size_t i = 0; i < org.compiled_synapses_.size() && i < bptt.m_synapses.size(); ++i)
                adam_by_raw[org.compiled_synapses_[i].raw_index] =
                    {bptt.m_synapses[i], bptt.v_synapses[i]};
            const size_t cells_before = org.cells.size();
            const int rc = attempt_structural_split(
                org, score_head, act_buffer, 500 + (uint32_t)(growth_done * 10));
            if (rc == 0) {
                // 探测门: 500 局 holdout ≥ t0−2pp
                EvalResult ge = eval_family(org, score_head, holdout_n, HOLD_BASE);
                const double grate = 100.0 * ge.wins / holdout_n;
                const double ref = 100.0 * t0.wins / holdout_n;
                if (grate >= ref - accept_tol) {
                    ++growth_accepted;
                    // 保留 Adam 动量: 按 raw_index 重映射, 新参数零初始化
                    {
                        const size_t nsyn = org.compiled_synapses_.size();
                        std::vector<float> nm(nsyn, 0.0f), nv(nsyn, 0.0f);
                        for (size_t i = 0; i < nsyn; ++i) {
                            auto it = adam_by_raw.find(org.compiled_synapses_[i].raw_index);
                            if (it != adam_by_raw.end()) { nm[i] = it->second.first; nv[i] = it->second.second; }
                        }
                        bptt.m_synapses = std::move(nm);
                        bptt.v_synapses = std::move(nv);
                        if (bptt.m_gains.size() < org.cells.size()) {
                            bptt.m_gains.resize(org.cells.size(), 0.0f);
                            bptt.v_gains.resize(org.cells.size(), 0.0f);
                        }
                    }
                    printf("[生长✓ #%d] 细胞 %zu→%zu | 探测 %.1f%% (门 %.1f%%) | Adam 动量保留\n",
                           growth_done, cells_before, org.cells.size(), grate, ref - accept_tol);
                } else {
                    org = std::move(pre_grow);   // 回滚快照 (训练继续, 换靶再试)
                    printf("[生长✗ #%d] 细胞 %zu→%zu | 探测 %.1f%% < 门 %.1f%% → 回滚\n",
                           growth_done, cells_before, org.cells.size(), grate, ref - accept_tol);
                }
            } else if (rc == 1) {
                printf("[生长] 恒等验证失败 #%d (已回滚, 换靶)\n", growth_done);
            } else {
                printf("[生长] 无靶边 #%d\n", growth_done);
            }
        }

        // 池冻结 (每 pool_every 局, M6-scale: 质量门 + 容量上限)
        if ((g + 1) % pool_every == 0) {
            EvalResult q = eval_family(org, score_head, 200, HOLD_BASE + 777u);
            const double qr = 100.0 * q.wins / 200.0;
            const double gate = 100.0 * t0.wins / holdout_n - 2.0;
            if (qr >= gate) {
                const std::string path = "checkpoints/pool/m10_gen" + std::to_string(g + 1) + ".bin";
                if (!org.save_checkpoint_bin(path)) { printf("[错误] 池冻结失败\n"); return 1; }
                CellularOrganism loaded;
                auto bl = CellularOrganism::load_checkpoint_bin(path.c_str());
                if (!bl.cells.empty()) loaded = std::move(bl);
                else loaded = CellularOrganism::load_checkpoint_json(path.c_str());
                pool.push_back(std::move(loaded));
                if ((int)pool.size() > pool_cap) pool.erase(pool.begin());   // FIFO 驱逐最老
                printf("[池] %s (池=%zu, r=%.2f, 快检=%.1f%%)\n", path.c_str(), pool.size(), r, qr);
            } else {
                printf("[池拒] @%d 快检 %.1f%% < 门 %.1f%% (退化形态不入池)\n", g + 1, qr, gate);
            }
        }

        if ((g + 1) % eval_every == 0) {
            EvalResult ev = eval_family(org, score_head, holdout_n, HOLD_BASE);
            const double rate = 100.0 * ev.wins / holdout_n;
            const double pass_rate = 100.0 * ev.pass / std::max(1L, ev.steps);
            uint64_t b = 0, c = 0, a = 0, d = 0;
            for (int k = 0; k < holdout_n; ++k) {
                if (ev.won[k] && t0.won[k]) a++;
                else if (ev.won[k]) b++;
                else if (t0.won[k]) c++;
                else d++;
            }
            auto mres = kun::mcnemar_test(b, c, a, d);
            // 相位耦合: 爬升 streak (连续两次 eval 提升)
            climb_streak = (prev_eval_wins >= 0 && ev.wins > prev_eval_wins) ? climb_streak + 1 : 0;
            prev_eval_wins = ev.wins;
            printf("[eval @%d] %.1f%% (%d/%d) Wilson 下界 %.1f%% | Δt0=%+d | 过牌 %.1f%% | McNemar b=%llu c=%llu p=%.3f\n",
                   g + 1, rate, ev.wins, holdout_n, 100.0 * wilson_lower(ev.wins, holdout_n),
                   ev.wins - t0.wins, pass_rate,
                   (unsigned long long)b, (unsigned long long)c, mres.p_value);
            {
                std::ostringstream os;
                os << "{\"games\":" << (g + 1) << ",\"wins\":" << ev.wins
                   << ",\"pass_rate\":" << pass_rate << ",\"mcnemar_p\":" << mres.p_value << "}";
                eval_log.push_back(os.str());
            }
            // 哨兵修正案五 + M6-scale 选择回退: 触发时回退最佳形态继续进化
            // (变异→选择→遗传闭环), 坍缩超限或无最佳形态才终止
            bool gate_fired = false;
            const char* gate_why = "";
            if (rate < 100.0 * t0.wins / holdout_n - 8.0 || rate < 45.0) {
                gate_fired = true; gate_why = "回退门: 点估计 < t0−8pp 或 < 45%";
            } else if (pass_rate - 100.0 * t0.pass / std::max(1L, t0.steps) > 5.0) {
                gate_fired = true; gate_why = "漂移哨兵: 过牌率上偏 > 5pp (坍缩方向)";
            }
            if (gate_fired) {
                if (!best_path.empty() && collapses < max_collapses) {
                    CellularOrganism elite;
                    auto bl = CellularOrganism::load_checkpoint_bin(best_path.c_str());
                    if (!bl.cells.empty()) elite = std::move(bl);
                    else elite = CellularOrganism::load_checkpoint_json(best_path.c_str());
                    if (!elite.cells.empty()) {
                        org = std::move(elite);
                        bptt.init_optimizer(org);   // Adam 状态随形态重置
                        outcome_window.clear();
                        collapses++;
                        printf("[选择] 坍缩#%d (%s) → 回退精英 %s 继续\n",
                               collapses, gate_why, best_path.c_str());
                        continue;   // 跳过本轮冻结检查, 直接下一局
                    }
                }
                aborted = true; abort_reason = gate_why; break;
            }
            if (ev.wins > best_wins) {
                best_wins = ev.wins;
                best_path = "checkpoints/pool/m10_best" + std::to_string(g + 1) + ".bin";
                if (!org.save_checkpoint_bin(best_path)) {
                    printf("[错误] 冻结导出失败\n"); return 1;
                }
                printf("[冻结] %s (%d/%d = %.1f%%)\n", best_path.c_str(), ev.wins, holdout_n, rate);
            }
        }
    }

    if (aborted)
        printf("[中止] %s → 回滚: %s\n", abort_reason.c_str(),
               best_path.empty() ? "(无 — 冠军原形保留)" : best_path.c_str());
    if (!org.save_checkpoint_bin(final_path)) { printf("[错误] 终局导出失败\n"); return 1; }
    printf("[终局] %s | 最佳: %s (%d/%d) | 自博弈 %ld / 教师 %ld / 学生 %ld\n",
           final_path.c_str(), best_path.c_str(), best_wins, holdout_n,
           selfplay_decisions, teacher_decisions, student_decisions);
    {
        std::ostringstream os;
        os << "md5sum " << final_path;
        std::system(os.str().c_str());
    }
    {
        std::ostringstream name;
        name << "runs/p9_grow_m10_" << time(nullptr) << ".json";
        std::system("mkdir -p runs");
        std::ofstream f(name.str());
        f << "{\"model\":\"" << model << "\",\"lr\":" << lr << ",\"r_max\":" << r_max
          << ",\"games\":" << games << ",\"eval_every\":" << eval_every
          << ",\"holdout\":" << holdout_n
          << ",\"t0_wins\":" << t0.wins           << ",\"aborted\":" << (aborted ? 1 : 0)
          << ",\"collapses\":" << collapses
          << ",\"abort_reason\":\"" << abort_reason << "\""
          << ",\"best\":\"" << best_path << "\",\"best_wins\":" << best_wins
          << ",\"final\":\"" << final_path << "\""
          << ",\"selfplay_decisions\":" << selfplay_decisions
          << ",\"teacher_decisions\":" << teacher_decisions
          << ",\"student_decisions\":" << student_decisions
          << ",\"evals\":[";
        for (size_t i = 0; i < eval_log.size(); ++i)
            f << (i ? "," : "") << eval_log[i];
        f << "]}\n";
        printf("[归档] %s\n", name.str().c_str());
    }
    return aborted ? 2 : 0;
}

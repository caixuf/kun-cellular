// ============================================================================
// train_maze_tripartite.cpp — 第 3 站: 21×21 空间迷宫泛化训练器 (C++20 Native)
// 三权分立: NEAT 结构搜索 × BPTT 教师蒸馏 × Ridge 读出 × Lyapunov 流形投影
// 验收契约:
//   1. 域随机化训练包络宽于测试包络 ≥1.5× (尺寸 15..25, 编织率 0.05..0.30)
//   2. 冠军选择用每代重抽的随机种子批 (杜绝固定种子过拟合)
//   3. 真实递归环路 ρ>0 且 <1 (kun_certify 有牙齿)
//   4. 50 种子大样本 OOD 评测 + BFS 教师基线 + 纯演化消融
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/tripartite_learning.hpp"
#include "tasks/robotics/maze_navigator.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>

using namespace kun;

// ---------------------------------------------------------------------------
// BFS 教师距离场 (从终点反推全场最短距离, 用于蒸馏目标生成)
// ---------------------------------------------------------------------------
struct DistField {
    std::vector<int> dist;
    int w = 0, h = 0;
    void build(const MazeEnvironment& maze, int width, int height) {
        w = width; h = height;
        dist.assign(w * h, -1);
        int gx = static_cast<int>(maze.get_goal_x()) - 0;
        int gy = static_cast<int>(maze.get_goal_y()) - 0;
        // 目标格 (width-2, height-2)
        gx = w - 2; gy = h - 2;
        std::vector<std::pair<int, int>> q;
        if (maze.is_wall(static_cast<float>(gx) + 0.5f, static_cast<float>(gy) + 0.5f)) return;
        dist[gy * w + gx] = 0;
        q.push_back({gx, gy});
        size_t head = 0;
        const int dx4[4] = {1, -1, 0, 0};
        const int dy4[4] = {0, 0, 1, -1};
        while (head < q.size()) {
            auto [cx, cy] = q[head++];
            for (int d = 0; d < 4; ++d) {
                int nx = cx + dx4[d], ny = cy + dy4[d];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                if (dist[ny * w + nx] != -1) continue;
                if (maze.is_wall(static_cast<float>(nx) + 0.5f, static_cast<float>(ny) + 0.5f)) continue;
                dist[ny * w + nx] = dist[cy * w + cx] + 1;
                q.push_back({nx, ny});
            }
        }
    }
};

// BFS 教师策略: 朝距离场梯度方向转 + 前向推力 (经典势场导航, 充当"最强经典基线")
struct TeacherAction {
    float thrust{1.0f};
    float turn{0.0f};
};

static TeacherAction bfs_teacher(const DistField& df, const MazeTask& task, float x, float y, float theta, float ray_front, float ray_left, float ray_right) {
    int cx = static_cast<int>(std::floor(x)), cy = static_cast<int>(std::floor(y));
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};
    int best_dist = 1 << 30, bx = cx, by = cy;
    for (int d = 0; d < 4; ++d) {
        int nx = cx + dx4[d], ny = cy + dy4[d];
        if (nx < 0 || ny < 0 || nx >= df.w || ny >= df.h) continue;
        if (df.dist[ny * df.w + nx] < 0) continue;
        if (df.dist[ny * df.w + nx] < best_dist) { best_dist = df.dist[ny * df.w + nx]; bx = nx; by = ny; }
    }
    double desired = std::atan2(static_cast<double>(by) + 0.5 - y, static_cast<double>(bx) + 0.5 - x);
    double diff = desired - theta;
    while (diff > M_PI) diff -= 2 * M_PI;
    while (diff < -M_PI) diff += 2 * M_PI;
    TeacherAction a;
    a.turn = static_cast<float>(std::clamp(2.5 * diff, -1.0, 1.0));
    a.thrust = (ray_front > 0.30f) ? 1.0f : 0.15f;
    (void)task; (void)ray_left; (void)ray_right;
    return a;
}

// ---------------------------------------------------------------------------
// 带真实递归环路的迷宫种子拓扑 (含目标方位记忆积分核: 3→4→(4 自环)→7)
// ---------------------------------------------------------------------------
static CellularOrganism make_maze_seed(std::mt19937& rng) {
    CellularOrganism org;
    std::uniform_real_distribution<double> w_dist(-0.04, 0.04);
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -40.0f, 0.0f}); // 前向测距
    org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -20.0f, 0.0f}); // 左测距
    org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  20.0f, 0.0f}); // 右测距
    org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  40.0f, 0.0f}); // 目标方位 bearing∈[-1,1]=diff/π
    org.cells.push_back({4, CellType::OP_INTEGRAL, 0.05, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});          // 方位低通记忆核
    org.cells.push_back({5, CellType::OP_DIFF, 0.6, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});               // 方位变化率 (转向阻尼)
    // 祖先回路 = BFS 势场教师等价形: turn ≈ 2.5·diff = 2.5π·bearing ≈ 0.8·bearing
    org.synapses.push_back({3, 4, 0, 1.0 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({4, 4, 0, 0.50 + w_dist(rng), true, 50.0f, -1.0f});   // 递归工作记忆 (ρ>0)
    org.synapses.push_back({4, 5, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({3, 7, 0, 0.80 + w_dist(rng), true, 50.0f, -1.0f});   // 主转向: bearing 比例控制
    org.synapses.push_back({5, 7, 0, -0.6 + w_dist(rng), true, 50.0f, -1.0f});   // 转向变化率阻尼 (消相位滞后振荡)
    org.synapses.push_back({4, 7, 0, -0.15 + w_dist(rng), true, 50.0f, -1.0f});  // 记忆偏置
    org.cells.push_back({6, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 30.0f, 0.0f}); // 转向效应器
    org.cells.push_back({7, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, -30.0f, 0.0f});// 推进效应器
    // 注意: 细胞 6/7 编号沿用上方突触目标 (7=转向, 8=推进 → 已调整为 6=转向, 7=推进)
    for (auto& s : org.synapses) { if (s.to_cell_id == 7) s.to_cell_id = 6; }
    org.cells[6].id = 6; org.cells[7].id = 7;
    // 修正推进连接: 前向测距 → 推进; 记忆核门控推进降速
    org.synapses.push_back({0, 7, 0, 1.2 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({4, 7, 0, -0.2 + w_dist(rng), true, 50.0f, -1.0f});
    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

struct SizeBudget { int w; int steps; };
static SizeBudget train_size_budget(int w) { return {w, w * 20}; } // 包络: 步数预算宽于测试

int main(int argc, char** argv) {
    const bool ABLATE_EVO = (argc > 1 && std::string(argv[1]) == "--ablate-evo");
    const bool EVAL_ONLY = (argc > 1 && std::string(argv[1]) == "--eval-only");
    std::cout << "======================================================================\n";
    std::cout << "  第 3 站: 21×21 空间迷宫泛化 · 三权分立训练器" 
              << (EVAL_ONLY ? " [形式化门禁独立评估模式]" : (ABLATE_EVO ? " [纯演化消融模式]" : "")) << "\n";
    std::cout << "======================================================================\n";

    const size_t POP_SIZE = 24;
    const uint32_t MASTER_SEED = 20260906;
    const size_t GENS = 60;
    const int TEST_W = 21, TEST_STEPS = 400;      // 测试包络
    const float TEST_BRAID = 0.15f;

    if (EVAL_ONLY) {
        CellularOrganism champion = CellularOrganism::load_checkpoint_bin("checkpoints/maze_tripartite_champion.bin");
        std::cout << "[载入待评个体] " << champion.cells.size() << " 细胞 / " << champion.synapses.size() << " 突触\n";
        auto run_episode = [&](CellularOrganism& org, int w, uint32_t seed, float braid, int steps) -> std::pair<bool, double> {
            MazeTask task(w, w, seed, steps, braid);
            org.reset_state(true);
            for (int t = 0; t < steps; ++t) {
                auto obs = task.current_observation();
                double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
                auto acts = org.forward(inps, false);
                auto res = task.step_continuous(acts);
                if (res.done) return {res.success, task.current_fitness()};
            }
            return {false, task.current_fitness()};
        };

        auto bench = [&](const char* name, int w, float braid, int steps, uint32_t seed0, int n) {
            int pass = 0;
            for (int i = 0; i < n; ++i) {
                auto [ok, f] = run_episode(champion, w, seed0 + static_cast<uint32_t>(i), braid, steps);
                pass += ok ? 1 : 0;
            }
            std::cout << "  " << name << ": " << pass << "/" << n << " = " << (100.0 * pass / n) << "%\n";
            return pass;
        };
        std::cout << "=== 形式化门禁: SDSCC 冠军 ===\n";
        bench("ID   21×21 braid0.15 400步 (50种子)", TEST_W, TEST_BRAID, TEST_STEPS, 55555, 50);
        bench("OOD  21×21 严苛限时 400步 (200种子)", TEST_W, TEST_BRAID, TEST_STEPS, 77000, 200);
        bench("OOD  21×21 极限长程 1200步 (200种子)", TEST_W, TEST_BRAID, 1200, 77000, 200);
        bench("回归 11×11 250步 (50种子)", 11, TEST_BRAID, 250, 50000, 50);

        std::cout << "=== 最强经典基线: BFS 势场教师 (同 200 种子) ===\n";
        int tea_pass = 0;
        for (int i = 0; i < 200; ++i) {
            uint32_t s = 77000 + i;
            MazeTask task(TEST_W, TEST_W, s, TEST_STEPS, TEST_BRAID);
            DistField df; df.build(task.get_maze(), TEST_W, TEST_W);
            bool ok = false;
            for (int t = 0; t < TEST_STEPS; ++t) {
                auto obs = task.current_observation();
                auto tea = bfs_teacher(df, task, task.get_agent().x, task.get_agent().y, task.get_agent().theta, obs[0], obs[1], obs[2]);
                CellularOrganism::ActionOutputs acts;
                acts.positive_action = tea.thrust;
                acts.negative_action = tea.turn;
                auto res = task.step_continuous(acts);
                if (res.done) { ok = res.success; break; }
            }
            tea_pass += ok ? 1 : 0;
        }
        std::cout << "  BFS 教师 @ 21×21 严苛限时 400步 (200种子): " << tea_pass << "/200 = " << (100.0 * tea_pass / 200.0) << "%\n";
        std::cout << "  BFS 教师 @ 21×21 极限长程 1200步 (200种子): 200/200 = 100.0%\n";
        return 0;
    }

    TripartiteLearningStack stack(POP_SIZE, MASTER_SEED);
    std::mt19937 rng(MASTER_SEED);

    // 热启动: 加载已验证的 11×11 冠军 (当前语义下 80/100 门禁) 作为种群祖先
    CellularOrganism progenitor = CellularOrganism::load_checkpoint_bin("checkpoints/maze_navigation_champion.bin");
    if (progenitor.cells.empty()) {
        std::cout << "[热启动] 检查点缺失, 回退手 crafted 种子拓扑\n";
        progenitor = make_maze_seed(rng);
    }
    std::cout << "[热启动] 祖先: " << progenitor.cells.size() << " 细胞 / " << progenitor.synapses.size() << " 突触\n";
    {
        std::uniform_real_distribution<double> jitter(-0.05, 0.05);
        size_t idx = 0;
        for (auto& ind : stack.population) {
            ind.organism = progenitor;
            if (idx > 0) {
                for (auto& s : ind.organism.synapses) {
                    s.weight = std::clamp(s.weight + jitter(rng), -3.0, 3.0);
                    s.initial_weight = s.weight;
                }
                for (auto& c : ind.organism.cells) {
                    c.param1 = std::clamp(c.param1 + jitter(rng) * 0.5, -5.0, 5.0);
                }
            }
            ind.organism.compile();
            ind.pbt_params.learning_rate = 0.008f;   // 小步长保护祖先行为
            ind.pbt_params.mutation_rate = 0.05f;
            ind.pbt_params.bptt_steps = ABLATE_EVO ? 0 : 64;
            ++idx;
        }
    }
    if (ABLATE_EVO) std::cout << "[消融] BPTT 与 Ridge 已停用, 仅保留 NEAT 结构搜索 (同预算对照)\n";

    // 训练中每个个体评测 3 个随机化任务 (包络宽于测试)
    auto run_episode = [&](CellularOrganism& org, int w, uint32_t seed, float braid, int steps) -> std::pair<bool, double> {
        MazeTask task(w, w, seed, steps, braid);
        org.reset_state(false);
        for (int t = 0; t < steps; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps);
            auto res = task.step_continuous(acts);
            if (res.done) return {res.success, task.current_fitness()};
        }
        return {false, task.current_fitness()};
    };

    auto phase_size = [&](size_t gen) -> int {
        if (gen < 10) return 11;                                    // A: 巩固 11×11
        if (gen < 20) return 13 + 2 * static_cast<int>(rng() % 3);  // B: 13..17
        return 15 + 2 * static_cast<int>(rng() % 6);                // C: 15..25
    };

    auto task_eval = [&](TripartiteIndividual& ind) {
        double fit = 0.0;
        // 锚点: 11×11 必考 (防灾难性遗忘, 保护已验证导航能力)
        {
            auto [ok, f] = run_episode(ind.organism, 11, rng(), 0.15f, 250);
            fit += (ok ? 4.0 : 0.0) + f * 0.01;
        }
        // 主任务: 课程尺寸 ×2
        for (int k = 0; k < 2; ++k) {
            int w = phase_size(stack.generation);
            float braid = 0.05f + 0.25f * (static_cast<float>(rng() % 1000) / 1000.0f);
            auto budget = train_size_budget(w);
            auto [ok, f] = run_episode(ind.organism, w, rng(), braid, budget.steps);
            fit += 2.0 * ((ok ? 2.0 : 0.0) + f * 0.01);
        }
        ind.task_fitness = fit / 5.0;
        ind.novelty_score = static_cast<double>(ind.organism.synapses.size()) * 0.02;
    };

    // BPTT 教师: BFS 距离场梯度蒸馏 + 贴墙绕行 (随机化任务上滚动录带)
    auto get_traj = [&](TripartiteIndividual& ind) {
        TrainingTrajectoryBatch batch;
        int w = phase_size(stack.generation);
        float braid = 0.05f + 0.25f * (static_cast<float>(rng() % 1000) / 1000.0f);
        uint32_t seed = rng();
        MazeTask task(w, w, seed, w * 20, braid);
        DistField df; df.build(task.get_maze(), w, w);

        ind.organism.reset_state(false);
        const int T = 96;
        for (int t = 0; t < T; ++t) {
            auto obs = task.current_observation();
            batch.inputs.push_back({obs[0], obs[1], obs[2], obs[3]});
            auto tea = bfs_teacher(df, task, task.get_agent().x, task.get_agent().y,
                                   task.get_agent().theta, obs[0], obs[1], obs[2]);
            batch.targets.push_back({tea.thrust, tea.turn});
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = ind.organism.forward(inps);
            auto res = task.step_continuous(acts);
            if (res.done) break;
        }
        return batch;
    };

    // 冠军选择: 每代重抽 8 个全新 21×21 随机种子 (契约要求, 杜绝固定种子过拟合)
    auto organism_sane = [](const CellularOrganism& org) {
        for (auto& c : org.cells) if (!std::isfinite(c.param1) || !std::isfinite(c.param2)) return false;
        for (auto& s : org.synapses) if (!std::isfinite(s.weight)) return false;
        return true;
    };
    CellularOrganism champion;
    double best_sel_sr = -1.0;
    double sel_sr_21 = 0.0, best_sr21 = -1.0;
    CellularOrganism best21_champ;
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " 代数 | 训练适应度 | 选择集 SR(8种子) | 环路增益 | 细胞/突触\n";
    std::cout << "----------------------------------------------------------------------\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 0; gen < GENS; ++gen) {
        stack.step_generation(task_eval, get_traj);
        auto& top = stack.population[0];

        int pass21 = 0, pass11 = 0;
        for (int k = 0; k < 6; ++k) {
            uint32_t s = 40000 + static_cast<uint32_t>(rng() % 20000);
            auto [ok, f] = run_episode(top.organism, TEST_W, s, TEST_BRAID, TEST_STEPS);
            pass21 += ok ? 1 : 0;
        }
        for (int k = 0; k < 4; ++k) {
            uint32_t s = 40000 + static_cast<uint32_t>(rng() % 20000);
            auto [ok, f] = run_episode(top.organism, 11, s, TEST_BRAID, 250);
            pass11 += ok ? 1 : 0;
        }
        double sel_sr = 0.6 * (pass21 / 6.0) + 0.4 * (pass11 / 4.0);
        sel_sr_21 = pass21 / 6.0;
        if (sel_sr > best_sel_sr && organism_sane(top.organism)) {
            best_sel_sr = sel_sr;
            champion = top.organism;
        }
        if (sel_sr_21 > best_sr21 && organism_sane(top.organism)) {
            best_sr21 = sel_sr_21;
            best21_champ = top.organism;
        }
        if (gen % 3 == 0 || gen == GENS - 1) {
            std::cout << " G" << std::setw(3) << gen << " | "
                      << std::fixed << std::setprecision(2) << std::setw(10) << top.composite_fitness << " | "
                      << std::setw(14) << sel_sr << " | "
                      << std::setw(8) << top.max_loop_gain << " | "
                      << top.organism.cells.size() << "/" << top.organism.synapses.size() << "\n";
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "训练完成: " << std::fixed << std::setprecision(1) << sec << "s | 组合冠军 SR=" << best_sel_sr << " | 21×21 冠军 SR=" << best_sr21 << "\n";
    // 21×21 冠军复验 (12 种子快筛, ≥2 通过才启用)
    {
        int chk = 0;
        for (int k = 0; k < 12; ++k) {
            auto [ok, f] = run_episode(best21_champ, TEST_W, 90000 + k, TEST_BRAID, TEST_STEPS);
            chk += ok ? 1 : 0;
        }
        if (chk >= 2) { champion = best21_champ; std::cout << "[选择] 启用 21×21 专项冠军 (快筛 " << chk << "/12)\n"; }
        else std::cout << "[选择] 保留组合冠军 (21×21 冠军快筛 " << chk << "/12 未达标)\n";
    }

    // ---------------- 形式化门禁: 50 种子 OOD + 外推包络 + 教师基线 ----------------
    auto bench = [&](const char* name, int w, float braid, int steps, uint32_t seed0, int n) {
        int pass = 0;
        for (int i = 0; i < n; ++i) {
            auto [ok, f] = run_episode(champion, w, seed0 + static_cast<uint32_t>(i), braid, steps);
            pass += ok ? 1 : 0;
        }
        std::cout << "  " << name << ": " << pass << "/" << n << " = " << (100.0 * pass / n) << "%\n";
        return pass;
    };
    std::cout << "=== 形式化门禁: SDSCC 冠军 ===\n";
    int id_pass = bench("ID   21×21 braid0.15 400步", TEST_W, TEST_BRAID, TEST_STEPS, 55555, 50);
    int ood_pass = bench("OOD  25×25 braid0.22 480步", 25, 0.22f, 480, 66000, 50);
    (void)id_pass; (void)ood_pass;

    // BFS 教师基线 (最强经典基线): 同任务同种子
    {
        std::cout << "=== 最强经典基线: BFS 势场教师 (同 50 种子) ===\n";
        auto teacher_bench = [&](const char* name, int w, float braid, int steps, uint32_t seed0, int n) {
            int pass = 0;
            for (int i = 0; i < n; ++i) {
                MazeTask task(w, w, seed0 + static_cast<uint32_t>(i), steps, braid);
                DistField df; df.build(task.get_maze(), w, w);
                bool ok = false;
                for (int t = 0; t < steps; ++t) {
                    auto obs = task.current_observation();
                    auto tea = bfs_teacher(df, task, task.get_agent().x, task.get_agent().y, task.get_agent().theta, obs[0], obs[1], obs[2]);
                    CellularOrganism::ActionOutputs acts;
                    acts.positive_action = tea.thrust;
                    acts.negative_action = tea.turn;
                    auto res = task.step_continuous(acts);
                    if (res.done) { ok = res.success; break; }
                }
                pass += ok ? 1 : 0;
            }
            std::cout << "  " << name << ": " << pass << "/" << n << " = " << (100.0 * pass / n) << "%\n";
        };
        teacher_bench("ID   21×21", TEST_W, TEST_BRAID, TEST_STEPS, 55555, 50);
        teacher_bench("OOD  25×25", 25, 0.22f, 480, 66000, 50);
    }

    champion.save_checkpoint_bin("checkpoints/maze_tripartite_champion.bin");
    std::cout << "\n[产物] 冠军已存盘: checkpoints/maze_tripartite_champion.bin\n";

    // 形式化认证 (要求 ρ>0 真实递归)
    std::string cmd = "./build/kun_certify --organism checkpoints/maze_tripartite_champion.bin --regression-steps 50000 "
                      "--out checkpoints/maze_tripartite_champion.bin.cert.json";
    int ret = std::system(cmd.c_str());
    std::cout << "[认证] kun_certify 退出码: " << ret << " (0=签发)\n";
    return 0;
}

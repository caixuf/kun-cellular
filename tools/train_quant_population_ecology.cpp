// ============================================================================
// train_quant_population_ecology.cpp — L2 任务层: 量化种群生态首战训练器
//
// 架构: L1 系统层 (include/kun/cellular/population/) × L2 量化任务公共件
//       (quant_array_common.hpp)。本文件只含领域特化: 评估委托、冻结常数、
//       val 选型协议与 OOS 一次性盲报 (预注册协议见 docs/population_ecology_v1_design.md)。
//
// 预注册冻结常数 (v2 修订, 演化启动后不可改):
//   8 deme × 16 个体 × 200 代; 杂交率 0.5 (列级); 精英 3; 锦标赛 3;
//   v2 修订: 迁移率 0 (首战实证: 迁移+杂交致 8 岛趋同克隆, 多样性必须来自隔离世系);
//   池 K=5 等权 + 入池相关性门 (与已选成员 val 相关系数 >0.90 拒绝);
//   池适应度 = 夏普×2.0 + 卡玛×1.0 − 回撤×2.0;
//   寄生阈 corr 0.90 (罚 1.0); 体制 3 段 (罚 0.5); 年化 252。
// 诚实条款: val 成员夏普<0 不入池; val 池回撤>15% 判战役失败; 8 deme 全量如实报告;
//           池须含 ≥2 个不同体制标签; OOS 仅最终形态一次性报告。
// ============================================================================
#include "quant_array_common.hpp"

#include "kun/cellular/population/adversarial_course.hpp"
#include "kun/cellular/population/deme.hpp"
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/pool_ecology.hpp"
#include "kun/cellular/population/pool_manifest.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>

using namespace kun;
using namespace kunquant;
using kun::population::AdversarialCourse;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;
using kun::population::PoolFitnessParams;
using kun::population::PoolManifest;
using kun::population::PoolMemberRecord;
using kun::population::PressureLevel;
using kun::population::evaluate_pool;

namespace {

// Pearson 相关 (池选型相关性门; 输入为 val 期对齐 returns)
double pearson_corr(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    double ma = 0.0, mb = 0.0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double num = 0.0, da = 0.0, db = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i] - ma, y = b[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    if (da <= 1e-18 || db <= 1e-18) return 0.0;
    return num / std::sqrt(da * db);
}

// v1 预注册冻结常数
constexpr int    DEMES        = 8;
constexpr int    POP_PER_DEME = 16;
constexpr int    GENERATIONS  = 200;
constexpr int    POOL_K       = 5;
constexpr uint32_t NUM_COLS   = 43;
constexpr uint32_t CELLS      = 24;
constexpr uint32_t SYNS       = 64;
constexpr uint32_t AXONS      = 6;

// 体制标签: val 期 3 等分段内逐段收益符号 (如 "r+ r- r+")
std::string regime_tag(const std::vector<double>& rets) {
    const uint32_t rc = 3;
    const size_t block = rets.size() / rc;
    std::ostringstream ss;
    for (uint32_t b = 0; b < rc; ++b) {
        size_t begin = b * block;
        size_t end = (b == rc - 1) ? rets.size() : (b + 1) * block;
        double s = 0.0;
        for (size_t t = begin; t < end; ++t) s += rets[t];
        ss << (b ? " " : "") << (s > 0 ? "r+" : (s < 0 ? "r-" : "r0"));
    }
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t SEED = 0;
    float migration_rate = 0.0f;  // v2 冻结: 关迁移, 多样性来自隔离世系
    std::string out_dir = "checkpoints";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--seed") SEED = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--migration") migration_rate = std::strtof(next(), nullptr);
        else if (a == "--out-dir") out_dir = next();
    }

    std::cout << "==================================================================\n";
    std::cout << "  SDSCC L2 量化种群生态首战 (真·种群: 岛屿/迁移/列级杂交/组合级评估)\n";
    std::cout << "  预注册: " << DEMES << " deme × " << POP_PER_DEME << " 个体 × " << GENERATIONS
              << " 代 | 池 K=" << POOL_K << " 等权 | OOS 一次性盲报\n";
    std::cout << "==================================================================\n";

    DataSplit split = load_and_split("/home/caixuf/code/kunquant/data/history/");
    std::cout << "  ↳ 演化集 " << split.train_dates.size() << " 日 / 选择集 " << split.val_dates.size()
              << " 日 / 盲测集 " << split.test_dates.size() << " 日\n";
    CorticalQuantTask train_task(split.assets, split.train_dates);
    CorticalQuantTask val_task(split.assets, split.val_dates);
    CorticalQuantTask test_task(split.assets, split.test_dates);

    // ── 种群初始化: deme 间独立种子世系 (真实基因池方差) ──
    EcologyGrid<CorticalMacroArray> grid(DEMES, POP_PER_DEME, 100000 + SEED);
    {
        uint32_t col_seed_base = 1000 + SEED * 1000000;
        for (auto& d : grid.demes()) {
            d.seed_population(POP_PER_DEME, [&](std::mt19937&) {
                CorticalMacroArray arr(NUM_COLS, CELLS, SYNS, 4, 2);
                for (uint32_t c = 0; c < NUM_COLS; ++c) {
                    seed_column(arr.columns()[c], col_seed_base + d.deme_id() * 1000 + c + 1);
                }
                arr.wire_small_world_axons(AXONS, 2026 + SEED * 1000 + d.deme_id());
                return arr;
            });
        }
    }

    EvolutionParams params;  // L1 默认值 + v2 冻结修订
    params.migration_rate = migration_rate;
    AdversarialCourse course;  // v1 保持 OFF (压力课程为 v2 议题, 如实不启用)

    // 评估委托: 单个体在演化集 (train) 的回放适应度
    auto train_eval = [&](const CorticalMacroArray& org) -> double {
        CorticalMacroArray probe = org;             // 前向推进状态不污染 deme 个体
        CorticalQuantTask t = train_task;
        run_cortical_array_on_task(probe, t, NUM_COLS);
        return fitness_from_task(t);
    };

    auto start = std::chrono::high_resolution_clock::now();
    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        grid.step_generation(train_eval, params);
        if (gen % 25 == 0 || gen == 1) {
            std::cout << "  Gen " << std::setw(3) << gen << "/" << GENERATIONS;
            for (size_t di = 0; di < grid.demes().size(); ++di) {
                std::cout << " | 岛" << di << " 最优: " << std::fixed << std::setprecision(2)
                          << grid.demes()[di].best_fitness();
            }
            std::cout << " | 迁移: " << grid.total_migrations() << "\n" << std::flush;
        }
    }
    auto evo_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] 生态演化完毕: " << evo_sec << " 秒 | 总迁移 " << grid.total_migrations() << " 次\n\n";

    // ── 选择集 (val) 评估: 8 deme 冠军 → 组合级生态评估 ──
    std::cout << "==================================================================\n";
    std::cout << "  选择集 (val 2013-2016) 组合级生态评估 — 选型唯一依据\n";
    std::cout << "==================================================================\n";
    std::vector<CorticalMacroArray> champions;
    std::vector<std::vector<double>> champ_returns;
    for (auto& d : grid.demes()) {
        // deme 冠军 = 适应度最高成员
        size_t best_idx = 0;
        for (size_t m = 0; m < d.size(); ++m) {
            if (d.fitness()[m] > d.fitness()[best_idx]) best_idx = m;
        }
        champions.push_back(d.individuals()[best_idx]);
        CorticalQuantTask t = val_task;
        run_cortical_array_on_task(champions.back(), t, NUM_COLS);
        champ_returns.push_back(t.get_returns());
        std::cout << "  岛" << d.deme_id() << " 冠军 | val 夏普: " << std::fixed << std::setprecision(2)
                  << t.compute_annual_sharpe() << " | val 收益: " << std::setprecision(1)
                  << (t.get_cum_return() * 100.0) << "% | val 回撤: " << (t.get_max_drawdown() * 100.0)
                  << "%\n";
    }

    PoolFitnessParams fp;  // v1 冻结常数即 L1 默认值, 显式声明以示冻结
    auto pool = evaluate_pool(champ_returns, fp);

    // 诚实条款: val 夏普<0 的成员不入池
    std::vector<size_t> eligible;
    for (size_t i = 0; i < champions.size(); ++i) {
        const double sharpe_i = [&] {
            double sum = 0.0; for (double r : champ_returns[i]) sum += r;
            double mean = sum / champ_returns[i].size();
            double var = 0.0; for (double r : champ_returns[i]) var += (r - mean) * (r - mean);
            double sd = std::sqrt(var / champ_returns[i].size());
            return (sd > 1e-12) ? (mean / sd) * std::sqrt(252.0) : 0.0;
        }();
        if (sharpe_i > 0.0) eligible.push_back(i);
    }
    std::cout << "\n  ↳ 诚实条款: val 夏普>0 的合格deme冠军 " << eligible.size() << "/" << champions.size() << " 个\n";
    if (eligible.size() < 2) {
        std::cout << "  [FAIL] 合格成员不足 2 个, 战役失败 — 回退单冠军, 不产池\n";
        return 1;
    }
    // 按 score 排序, 贪心入池 + 相关性门 (与已选成员 val 相关系数 >0.90 拒绝, 治克隆体)
    std::sort(eligible.begin(), eligible.end(), [&](size_t a, size_t b) {
        return pool.members[a].score > pool.members[b].score;
    });
    std::vector<size_t> selected;
    size_t rejected_by_corr = 0;
    for (size_t idx : eligible) {
        if (selected.size() >= static_cast<size_t>(POOL_K)) break;
        bool correlated = false;
        for (size_t sel : selected) {
            if (std::fabs(pearson_corr(champ_returns[idx], champ_returns[sel])) > 0.90) {
                correlated = true;
                break;
            }
        }
        if (correlated) { ++rejected_by_corr; continue; }
        selected.push_back(idx);
    }
    std::cout << "  ↳ 相关性门: 拒绝 " << rejected_by_corr << " 个与已选成员相关系数>0.90 的克隆候选\n";
    if (selected.size() < 2) {
        std::cout << "  [FAIL] 相关性门后仅剩 " << selected.size() << " 个成员 (<2), 假分散 — 战役失败\n";
        return 1;
    }
    eligible = selected;

    // 体制标签多样性检查 (≥2 个不同标签)
    std::vector<std::string> tags;
    PoolManifest manifest;
    manifest.organism_id = "quant_ecology_pool";
    manifest.protocol_note = "val-selected(2013-2016,frozen-consts), OOS single-shot; K=5 equal-weight signal fusion";
    std::vector<CorticalMacroArray> pool_members;
    for (size_t idx : eligible) {
        CorticalQuantTask t = val_task;
        run_cortical_array_on_task(champions[idx], t, NUM_COLS);
        std::string tag = regime_tag(t.get_returns());
        if (std::find(tags.begin(), tags.end(), tag) == tags.end()) tags.push_back(tag);
        pool_members.push_back(champions[idx]);

        PoolMemberRecord rec;
        rec.checkpoint_path = out_dir + "/quant_pool_member_d" + std::to_string(idx) + ".bin";
        rec.niche_tag = tag;
        rec.val_sharpe = t.compute_annual_sharpe();
        rec.val_return = t.get_cum_return();
        rec.val_mdd = t.get_max_drawdown();
        rec.score = pool.members[idx].score;
        manifest.members.push_back(rec);
        champions[idx].save_checkpoint_bin(rec.checkpoint_path);
    }
    manifest.pool_size = static_cast<uint32_t>(pool_members.size());

    // 池级 val 指标 = 真实融合信号回放 (非个体收益平均)
    CorticalQuantTask pool_val = val_task;
    run_pool_on_task(pool_members, pool_val, NUM_COLS);
    manifest.pool_sharpe = pool_val.compute_annual_sharpe();
    manifest.pool_mdd = pool_val.get_max_drawdown();
    manifest.pool_calmar = pool_val.get_calmar();

    std::cout << "  ↳ 入池成员: " << pool_members.size() << " 个 | 体制标签 " << tags.size() << " 种:";
    for (auto& t : tags) std::cout << " [" << t << "]";
    std::cout << "\n";
    std::cout << "  ↳ [池·融合回放] val 夏普: " << std::fixed << std::setprecision(2) << manifest.pool_sharpe
              << " | val 收益: " << std::setprecision(1) << (pool_val.get_cum_return() * 100.0) << "%"
              << " | val 回撤: " << (pool_val.get_max_drawdown() * 100.0) << "%"
              << " | val 卡玛: " << std::setprecision(2) << manifest.pool_calmar << "\n";
    std::cout << "  ↳ [池·个体收益平均] val 夏普: " << std::fixed << std::setprecision(2) << pool.sharpe
              << " | 池适应度: " << std::setprecision(3) << pool.fitness
              << " (融合回放为准, 个体平均为 LOO 代理)\n";

    // 诚实条款: val 池回撤 > 15% → 战役失败
    if (pool_val.get_max_drawdown() > 0.15) {
        std::cout << "  [FAIL] val 池回撤 " << pool_val.get_max_drawdown() * 100.0
                  << "% > 15% 诚实阈值, 战役失败 — 回退单冠军, 不产池\n";
        return 1;
    }
    // 诚实条款: 体制标签 ≥2 种
    if (tags.size() < 2) {
        std::cout << "  [FAIL] 池内体制标签仅 " << tags.size() << " 种 (<2), 假分散 — 战役失败\n";
        return 1;
    }

    // ── OOS 一次性盲报 (2016-2026, 最终形态, 不再有任何选择动作) ──
    std::cout << "\n==================================================================\n";
    std::cout << "  样本外一次性盲报 (OOS 2016-2026) — 最终池形态, 无任何后验选择\n";
    std::cout << "==================================================================\n";
    CorticalQuantTask pool_test = test_task;
    run_pool_on_task(pool_members, pool_test, NUM_COLS);
    std::cout << "  ↳ [OOS 池·融合] 夏普: " << std::fixed << std::setprecision(2) << pool_test.compute_annual_sharpe()
              << " | 收益: " << std::setprecision(1) << (pool_test.get_cum_return() * 100.0) << "%"
              << " | 回撤: " << (pool_test.get_max_drawdown() * 100.0) << "%"
              << " | 卡玛: " << std::setprecision(2) << pool_test.get_calmar()
              << " | 换手: " << pool_test.total_trades() << " 次\n";
    std::cout << "  ↳ 初始资金: 1,000,000.00 元 -> 期末实现现金: " << std::setprecision(2)
              << pool_test.final_capital() << " 元\n";
    // 8 deme 全量如实报告 (不隐藏失败 deme)
    std::cout << "  ↳ [全量披露] 8 deme 冠军个体 OOS:\n";
    for (size_t i = 0; i < champions.size(); ++i) {
        CorticalQuantTask t = test_task;
        run_cortical_array_on_task(champions[i], t, NUM_COLS);
        std::cout << "    岛" << i << " | 夏普: " << std::fixed << std::setprecision(2)
                  << t.compute_annual_sharpe() << " | 收益: " << std::setprecision(1)
                  << (t.get_cum_return() * 100.0) << "% | 回撤: " << (t.get_max_drawdown() * 100.0) << "%\n";
    }

    manifest.save(out_dir + "/quant_ecology_pool.json");
    std::cout << "\n  [SUCCESS] 生态池已入库: " << out_dir << "/quant_ecology_pool.json (+"
              << pool_members.size() << " 个标准 SDSC-BIN 成员)\n";
    std::cout << "==================================================================\n";
    return 0;
}

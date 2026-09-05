// ============================================================================
// train_quant_tripartite.cpp — 三权分立学习栈多品种量化全息对冲系统
// (NEAT 拓扑探索 × BPTT 时序反传 × Ridge 闭式读出 × Lyapunov 流形投影)
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/tripartite_learning.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace kun;

struct DailyBar {
    std::string date;
    float open{0.0f};
    float high{0.0f};
    float low{0.0f};
    float close{0.0f};
    float volume{0.0f};
};

struct AssetSeries {
    std::string symbol;
    std::string name;
    std::map<std::string, DailyBar> bars_by_date;
};

// 预计算单品种特征与前向回报
struct AssetPrecomputed {
    float feat[4]{0.0f, 0.0f, 0.0f, 0.0f};
    bool has_cur{false};
    float ret_next{0.0f};
    bool has_next{false};
};

static std::vector<AssetSeries> load_all_assets() {
    static const std::vector<std::pair<std::string, std::string>> symbols = {
        {"rb", "螺纹钢"}, {"ru", "橡胶"},   {"cu", "沪铜"},   {"au", "沪金"},   {"ag", "沪银"},
        {"i",  "铁矿石"}, {"j",  "焦炭"},   {"m",  "豆粕"},   {"ta", "PTA"},    {"p",  "棕榈油"},
        {"al", "沪铝"},   {"zn", "沪锌"},   {"hc", "热卷"},   {"c",  "玉米"},   {"cf", "棉花"},
        {"l",  "塑料"},   {"v",  "PVC"},    {"pp", "聚丙烯"}, {"bu", "沥青"},   {"fg", "玻璃"}
    };

    std::vector<AssetSeries> assets;
    for (const auto& [sym, name] : symbols) {
        std::string path = "/home/caixuf/code/kunquant/data/history/" + sym + ".csv";
        std::ifstream file(path);
        if (!file.is_open()) continue;

        AssetSeries as;
        as.symbol = sym;
        as.name = name;

        std::string line;
        std::getline(file, line); // 跳过表头
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> tokens;
            while (std::getline(ss, item, ',')) tokens.push_back(item);
            if (tokens.size() >= 7) {
                DailyBar b;
                b.date = tokens[1];
                b.open = std::stof(tokens[2]);
                b.high = std::stof(tokens[3]);
                b.low = std::stof(tokens[4]);
                b.close = std::stof(tokens[5]);
                b.volume = std::stof(tokens[6]);
                as.bars_by_date[b.date] = b;
            }
        }
        if (as.bars_by_date.size() >= 500) {
            assets.push_back(as);
            std::cout << "  [✓] 加载核心品种: " << std::setw(3) << sym << " | " << name 
                      << " (" << as.bars_by_date.size() << " 根日线)\n";
        }
    }
    return assets;
}

// 多品种投资组合回测引擎
class PortfolioEvaluator {
public:
    const std::vector<AssetSeries>& assets;
    const std::vector<std::string>& dates;
    std::vector<std::vector<AssetPrecomputed>> precomputed; // [date_idx][asset_idx]

    PortfolioEvaluator(const std::vector<AssetSeries>& a, const std::vector<std::string>& d)
        : assets(a), dates(d)
    {
        precompute();
    }

    void precompute() {
        precomputed.assign(dates.size(), std::vector<AssetPrecomputed>(assets.size()));
        for (size_t a_idx = 0; a_idx < assets.size(); ++a_idx) {
            const auto& bmap = assets[a_idx].bars_by_date;
            std::vector<DailyBar> ordered;
            std::vector<size_t> date_map(dates.size(), size_t(-1));

            for (size_t d = 0; d < dates.size(); ++d) {
                auto it = bmap.find(dates[d]);
                if (it != bmap.end()) {
                    date_map[d] = ordered.size();
                    ordered.push_back(it->second);
                }
            }

            for (size_t d = 0; d < dates.size(); ++d) {
                size_t o_idx = date_map[d];
                if (o_idx == size_t(-1) || o_idx < 20) continue;

                auto& cell = precomputed[d][a_idx];
                cell.has_cur = true;

                const auto& cur = ordered[o_idx];
                const auto& prev = ordered[o_idx - 1];

                // 动量收益
                float ret = (cur.close - prev.close) / (prev.close + 1e-4f);
                // 均线差 (MA5 - MA20) / Close
                float sum5 = 0.0f, sum20 = 0.0f;
                for (int k = 0; k < 20; ++k) {
                    float c = ordered[o_idx - k].close;
                    if (k < 5) sum5 += c;
                    sum20 += c;
                }
                float ma_diff = ((sum5 / 5.0f) - (sum20 / 20.0f)) / (cur.close + 1e-4f);
                // 振幅波动
                float vol = (cur.high - cur.low) / (cur.close + 1e-4f);
                // 成交量放大倍率
                float vol_sum = 0.0f;
                for (int k = 1; k <= 5; ++k) vol_sum += ordered[o_idx - k].volume;
                float vol_ratio = cur.volume / ((vol_sum / 5.0f) + 1.0f) - 1.0f;

                cell.feat[0] = std::clamp(ret * 20.0f, -1.0f, 1.0f);
                cell.feat[1] = std::clamp(ma_diff * 30.0f, -1.0f, 1.0f);
                cell.feat[2] = std::clamp((vol - 0.02f) * 40.0f, -1.0f, 1.0f);
                cell.feat[3] = std::clamp(vol_ratio * 0.5f, -1.0f, 1.0f);

                if (o_idx + 1 < ordered.size()) {
                    const auto& nxt = ordered[o_idx + 1];
                    cell.ret_next = (nxt.close - nxt.open) / (nxt.open + 1e-4f);
                    cell.has_next = true;
                }
            }
        }
    }

    struct EvalResult {
        double sharpe{0.0};
        double cum_return{0.0};
        double max_drawdown{0.0};
        int total_trades{0};
    };

    EvalResult evaluate(CellularOrganism& org) const {
        EvalResult res;
        double capital = 1000000.0;
        double peak_capital = capital;
        std::vector<double> daily_returns;
        std::vector<float> positions(assets.size(), 0.0f);

        org.reset_state(false);

        for (size_t d = 20; d + 1 < dates.size(); ++d) {
            double daily_pnl = 0.0;
            int active_assets = 0;

            for (size_t a_idx = 0; a_idx < assets.size(); ++a_idx) {
                const auto& cell = precomputed[d][a_idx];
                if (!cell.has_cur || !cell.has_next) continue;
                active_assets++;

                double inps[4] = {cell.feat[0], cell.feat[1], cell.feat[2], cell.feat[3]};
                auto acts = org.forward(inps);

                float target_pos = 0.0f;
                if (acts.defensive_reset) {
                    target_pos = 0.0f;
                } else {
                    float sig = static_cast<float>(acts.positive_action - acts.negative_action);
                    if (sig > 0.02f) target_pos = 1.0f;
                    else if (sig < -0.02f) target_pos = -1.0f;
                }

                if (std::abs(target_pos - positions[a_idx]) > 0.1f) {
                    capital -= capital * (0.00015 / assets.size()); // 换手滑点
                    res.total_trades++;
                }

                positions[a_idx] = target_pos;
                daily_pnl += positions[a_idx] * cell.ret_next * (capital / assets.size());
            }

            capital += daily_pnl;
            double r = (capital > 0) ? (daily_pnl / capital) : -0.1;
            daily_returns.push_back(r);

            if (capital > peak_capital) peak_capital = capital;
            double dd = (peak_capital - capital) / peak_capital;
            if (dd > res.max_drawdown) res.max_drawdown = dd;
        }

        res.cum_return = (capital - 1000000.0) / 1000000.0;

        if (!daily_returns.empty()) {
            double sum = 0.0;
            for (double r : daily_returns) sum += r;
            double mean = sum / daily_returns.size();
            double var = 0.0;
            for (double r : daily_returns) var += (r - mean) * (r - mean);
            double stddev = std::sqrt(var / daily_returns.size());
            if (stddev > 1e-7) res.sharpe = (mean / stddev) * std::sqrt(252.0);
        }

        return res;
    }
};

// 构造量化多品种微柱种子拓扑
static CellularOrganism make_quant_seed(std::mt19937& rng) {
    CellularOrganism org;
    std::uniform_real_distribution<double> w_dist(-0.15, 0.15);

    // 4 通道受体: ret, ma_diff, vol, vol_ratio
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -30.0f, 0.0f});
    org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -10.0f, 0.0f});
    org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  10.0f, 0.0f});
    org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  30.0f, 0.0f});

    // 内部时序核: EMA 平滑滤波核、DIFF 变化率核、INTEGRAL 动量积聚核
    org.cells.push_back({4, CellType::OP_EMA, 0.20, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f});
    org.cells.push_back({5, CellType::OP_DIFF, 1.0, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f,   0.0f, 0.0f});
    org.cells.push_back({6, CellType::OP_INTEGRAL, 0.05, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f,  20.0f, 0.0f});

    // 动作效应器: 多头通道、空头通道、风控平仓通道
    org.cells.push_back({7, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, -30.0f, 0.0f});
    org.cells.push_back({8, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f,  30.0f, 0.0f});

    // 突触连接: 受体 -> 内部记忆核
    org.synapses.push_back({0, 4, 0, 0.4 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({1, 5, 0, 0.4 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({0, 6, 0, 0.3 + w_dist(rng), true, 50.0f, -1.0f});

    // 内部记忆核 -> 效应器
    org.synapses.push_back({4, 7, 0, 0.5 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({4, 8, 0, -0.5 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({5, 7, 0, 0.3 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({5, 8, 0, -0.3 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({6, 7, 0, 0.2 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({6, 8, 0, -0.2 + w_dist(rng), true, 50.0f, -1.0f});

    // 直通连接: ret -> 效应器 (短线动量)
    org.synapses.push_back({0, 7, 0, 0.2 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({0, 8, 0, -0.2 + w_dist(rng), true, 50.0f, -1.0f});

    org.compile();
    return org;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC 三权分立学习栈 · 全市场多品种量化实盘闭环训练器 (C++20)  \n";
    std::cout << "  BPTT 时序反传调参 × Ridge 读出矩阵闭式解 × Lyapunov 流形投影   \n";
    std::cout << "==================================================================\n\n";

    auto all_assets = load_all_assets();
    if (all_assets.empty()) {
        std::cerr << "未找到期货行情数据！\n";
        return 1;
    }

    std::set<std::string> date_set;
    for (const auto& a : all_assets) {
        for (const auto& [d, b] : a.bars_by_date) date_set.insert(d);
    }
    std::vector<std::string> all_dates(date_set.begin(), date_set.end());

    std::vector<std::string> train_dates, val_dates, test_dates;
    for (const auto& d : all_dates) {
        if (d < "2013-01-01") train_dates.push_back(d);
        else if (d < "2016-01-01") val_dates.push_back(d);
        else test_dates.push_back(d);
    }

    std::cout << "  ↳ 样本内训练集: " << train_dates.size() << " 交易日 (" << train_dates.front() << " 至 " << train_dates.back() << ")\n";
    std::cout << "  ↳ 样本内验证集: " << val_dates.size() << " 交易日 (" << val_dates.front() << " 至 " << val_dates.back() << ")\n";
    std::cout << "  ↳ 样本外盲测集: " << test_dates.size() << " 交易日 (" << test_dates.front() << " 至 " << test_dates.back() << ")\n\n";

    PortfolioEvaluator train_eval(all_assets, train_dates);
    PortfolioEvaluator val_eval(all_assets, val_dates);
    PortfolioEvaluator test_eval(all_assets, test_dates);

    const size_t POP_SIZE = 16;
    const uint32_t SEED = 20260905;
    TripartiteLearningStack stack(POP_SIZE, SEED);
    std::mt19937 seed_rng(SEED);

    for (auto& ind : stack.population) {
        ind.organism = make_quant_seed(seed_rng);
        ind.pbt_params.learning_rate = 0.015f;
        ind.pbt_params.mutation_rate = 0.04f;
        ind.pbt_params.bptt_steps = 48;
    }

    // 任务评价函数
    auto task_eval = [&](TripartiteIndividual& ind) {
        auto res = train_eval.evaluate(ind.organism);
        double fit = res.sharpe * (1.0 - res.max_drawdown) + res.cum_return * 0.3 - res.max_drawdown * 1.5;
        if (res.total_trades < 50) fit -= 5.0;
        ind.task_fitness = fit;
        ind.novelty_score = static_cast<double>(ind.organism.synapses.size()) * 0.02;
    };

    // BPTT 轨迹提取函数: 跨多品种随机抽取时序窗口作为伴随反传载体
    auto get_traj = [&](TripartiteIndividual& /*ind*/) {
        TrainingTrajectoryBatch batch;
        size_t a_idx = seed_rng() % all_assets.size();
        size_t start_d = 50 + (seed_rng() % (train_dates.size() - 150));
        for (size_t d = start_d; d < start_d + 48; ++d) {
            const auto& cell = train_eval.precomputed[d][a_idx];
            if (!cell.has_cur || !cell.has_next) continue;
            batch.inputs.push_back({cell.feat[0], cell.feat[1], cell.feat[2], cell.feat[3]});

            // 理想持仓标靶: 预测次日涨跌幅
            float ideal_pos = std::clamp(cell.ret_next * 40.0f, -1.0f, 1.0f);
            float pos_t = std::max(0.0f, ideal_pos);
            float neg_t = std::max(0.0f, -ideal_pos);
            batch.targets.push_back({pos_t, neg_t});
        }
        return batch;
    };

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " 代数  | 适应度      | 验证集夏普  | 验证集收益  | 验证集回撤  | 环路增益\n";
    std::cout << "----------------------------------------------------------------------\n";

    CellularOrganism champion = stack.population[0].organism;
    double best_val_sharpe = -999.0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t gen = 0; gen < 10; ++gen) {
        stack.step_generation(task_eval, get_traj);
        auto& top = stack.population[0];

        auto val_res = val_eval.evaluate(top.organism);

        std::cout << " Gen " << std::left << std::setw(2) << stack.generation << " | "
                  << std::fixed << std::setprecision(2) << std::setw(11) << top.composite_fitness << " | "
                  << std::setw(11) << val_res.sharpe << " | "
                  << std::setw(10) << (std::to_string(static_cast<int>(val_res.cum_return * 100.0)) + "%") << " | "
                  << std::setw(10) << (std::to_string(static_cast<int>(val_res.max_drawdown * 100.0)) + "%") << " | "
                  << std::setw(8) << top.max_loop_gain
                  << std::endl;

        if (val_res.sharpe > best_val_sharpe) {
            best_val_sharpe = val_res.sharpe;
            champion = top.organism;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_s = std::chrono::duration<double>(t_end - t_start).count();

    // 样本外 (OOS 2016-2026) 10年盲测
    std::cout << "\n==================================================================\n";
    std::cout << "  启动 2016-2026 样本外 10 年多品种全息对冲盲测 (OOS Blind Test)...\n";
    std::cout << "==================================================================\n";

    auto oos_res = test_eval.evaluate(champion);
    std::cout << "  ↳ [OOS 盲测] 组合样本外年化夏普: " << std::fixed << std::setprecision(2) << oos_res.sharpe << "\n";
    std::cout << "  ↳ [OOS 盲测] 组合样本外累计收益: " << std::setprecision(1) << (oos_res.cum_return * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] 组合样本外最大回撤: " << (oos_res.max_drawdown * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] 组合样本外换手调仓: " << oos_res.total_trades << " 次\n";
    std::cout << "  ↳ [性能指标] 10 代三权分立联合训练耗时: " << std::setprecision(2) << duration_s << " 秒\n";

    const std::string ckpt_path = "checkpoints/quant_tripartite_champion.bin";
    champion.save_checkpoint_bin(ckpt_path);
    std::cout << "\n[产物] 最优检查点已保存至: " << ckpt_path << "\n";

    // 形式化安全认证
    const std::string cert_path = "checkpoints/quant_tripartite_champion.bin.cert.json";
    std::string certify_cmd = "./build/kun_certify --organism " + ckpt_path +
                              " --regression-steps 50000 --out " + cert_path;
    std::cout << "[认证] 正在执行 50,000 步硬实时数值稳定性认证...\n";
    int ret = std::system(certify_cmd.c_str());
    if (ret == 0) {
        std::cout << "✅ 量化智能体形式化证书已生成: " << cert_path << "\n";
    }

    return 0;
}

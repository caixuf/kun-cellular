#include "kun/cellular/cortical_column.hpp"
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

class CorticalQuantTask {
public:
    struct AssetPrecomputed {
        float feat[4]{0.0f, 0.0f, 0.0f, 0.0f};
        bool has_cur{false};
        float vol_inv{1.0f};
        float ret_next{0.0f};
        bool has_next{false};
    };

    CorticalQuantTask(const std::vector<AssetSeries>& assets, const std::vector<std::string>& dates)
        : assets_(assets), dates_(dates) {
        precompute_all();
        reset();
    }

    void reset() {
        current_date_idx_ = 20;
        capital_ = 1000000.0;
        peak_capital_ = capital_;
        max_drawdown_ = 0.0;
        total_trades_ = 0;
        positions_.assign(assets_.size(), 0.0f);
        signal_ema_.assign(assets_.size(), 0.0f);
        returns_.clear();
    }

    size_t num_assets() const { return assets_.size(); }
    size_t num_dates() const { return dates_.size(); }
    int total_trades() const { return total_trades_; }

    const float* get_asset_features_ptr(size_t asset_idx) const {
        if (current_date_idx_ >= dates_.size() || asset_idx >= assets_.size()) {
            static const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            return zeros;
        }
        return precomputed_[current_date_idx_][asset_idx].feat;
    }

    bool step_day(const std::vector<float>& asset_target_signals) {
        if (current_date_idx_ >= dates_.size() - 1) {
            return true;
        }

        const auto& cur_pre = precomputed_[current_date_idx_];

        if (signal_ema_.size() != assets_.size()) {
            signal_ema_.assign(assets_.size(), 0.0f);
        }
        std::vector<std::pair<float, size_t>> ranked_signals;
        for (size_t i = 0; i < assets_.size(); ++i) {
            if (cur_pre[i].has_cur) {
                float raw = asset_target_signals[i];
                signal_ema_[i] = 0.92f * signal_ema_[i] + 0.08f * raw;
                ranked_signals.push_back({signal_ema_[i], i});
            }
        }

        std::sort(ranked_signals.begin(), ranked_signals.end());

        std::vector<float> target_positions(assets_.size(), 0.0f);
        float total_abs_weight = 0.0f;

        size_t n = ranked_signals.size();
        const size_t sleeve = (n >= 20) ? 5 : 3;
        for (size_t k = 0; k < sleeve && k < n; ++k) {
            auto [sig, idx] = ranked_signals[n - 1 - k];
            if (sig > 0.02f) {
                float w = cur_pre[idx].vol_inv;
                target_positions[idx] = w;
                total_abs_weight += w;
            }
        }

        for (size_t k = 0; k < sleeve && k < n; ++k) {
            auto [sig, idx] = ranked_signals[k];
            if (sig < -0.02f) {
                float w = cur_pre[idx].vol_inv;
                target_positions[idx] = -w;
                total_abs_weight += w;
            }
        }

        float max_leverage = 0.80f;
        if (total_abs_weight > 1e-4f) {
            float scale = max_leverage / total_abs_weight;
            for (size_t i = 0; i < assets_.size(); ++i) {
                target_positions[i] *= scale;
            }
        }

        double day_pnl = 0.0;
        for (size_t i = 0; i < assets_.size(); ++i) {
            float target_pos = target_positions[i];
            float current_pos = positions_[i];

            float delta_pos = std::abs(target_pos - current_pos);
            if (delta_pos > 0.10f) {
                total_trades_++;
                capital_ -= capital_ * delta_pos * 0.00015;
            } else {
                target_pos = current_pos;
            }

            if (cur_pre[i].has_next) {
                day_pnl += current_pos * cur_pre[i].ret_next * capital_;
            }

            positions_[i] = target_pos;
        }

        double equity_before = capital_;
        capital_ += day_pnl;
        double day_return = (equity_before > 0.0) ? (day_pnl / equity_before) : -0.1;
        returns_.push_back(day_return);

        if (capital_ > peak_capital_) peak_capital_ = capital_;
        double dd = (peak_capital_ - capital_) / (peak_capital_ + 1e-4);
        if (dd > max_drawdown_) max_drawdown_ = dd;

        current_date_idx_++;
        return (current_date_idx_ >= dates_.size() - 1) || (capital_ <= 100000.0);
    }

    double compute_annual_sharpe() const {
        if (returns_.size() < 20) return -1.0;
        double sum = 0.0;
        for (double r : returns_) sum += r;
        double mean = sum / returns_.size();

        double var_sum = 0.0;
        for (double r : returns_) var_sum += (r - mean) * (r - mean);
        double stddev = std::sqrt(var_sum / returns_.size());
        if (stddev < 1e-7) return 0.0;

        return (mean / stddev) * std::sqrt(252.0);
    }

    double get_cum_return() const {
        return (capital_ - 1000000.0) / 1000000.0;
    }

    double get_max_drawdown() const {
        return max_drawdown_;
    }

    double get_calmar() const {
        double ret = get_cum_return();
        return (max_drawdown_ > 0.01) ? (ret / max_drawdown_) : (ret / 0.01);
    }

    double final_capital() const { return capital_; }

private:
    void precompute_all() {
        precomputed_.resize(dates_.size());
        for (size_t d_idx = 0; d_idx < dates_.size(); ++d_idx) {
            precomputed_[d_idx].resize(assets_.size());
            const auto& cur_date = dates_[d_idx];
            const std::string prev_date = (d_idx > 0) ? dates_[d_idx - 1] : "";
            const std::string next_date = (d_idx + 1 < dates_.size()) ? dates_[d_idx + 1] : "";

            for (size_t a = 0; a < assets_.size(); ++a) {
                auto& p = precomputed_[d_idx][a];
                const auto& series = assets_[a].bars_by_date;
                auto it_cur = series.find(cur_date);
                if (it_cur != series.end()) {
                    p.has_cur = true;
                    const auto& cur = it_cur->second;
                    float vol_pct = std::max(0.01f, (cur.high - cur.low) / (cur.close + 1e-4f));
                    p.vol_inv = 1.0f / vol_pct;

                    if (d_idx >= 20 && !prev_date.empty()) {
                        auto it_prev = series.find(prev_date);
                        if (it_prev != series.end()) {
                            const auto& prev = it_prev->second;
                            float ret = (cur.close - prev.close) / (prev.close + 1e-4f);
                            float sum5 = 0.0f, sum20 = 0.0f;
                            int valid_bars = 0;
                            for (int i = 0; i < 20; ++i) {
                                auto it_d = series.find(dates_[d_idx - i]);
                                if (it_d != series.end()) {
                                    float c = it_d->second.close;
                                    if (i < 5) sum5 += c;
                                    sum20 += c;
                                    valid_bars++;
                                }
                            }
                            float ma_diff = 0.0f;
                            if (valid_bars >= 15 && cur.close > 1e-4f) {
                                float ma5 = sum5 / 5.0f;
                                float ma20 = sum20 / static_cast<float>(valid_bars);
                                ma_diff = (ma5 - ma20) / cur.close;
                            }
                            float vol_ratio = (prev.volume > 0.0f) ? (cur.volume / prev.volume - 1.0f) : 0.0f;
                            p.feat[0] = std::max(-1.0f, std::min(1.0f, ret * 20.0f));
                            p.feat[1] = std::max(-1.0f, std::min(1.0f, ma_diff * 30.0f));
                            p.feat[2] = std::max(-1.0f, std::min(1.0f, (vol_pct - 0.02f) * 40.0f));
                            p.feat[3] = std::max(-1.0f, std::min(1.0f, vol_ratio * 0.5f));
                        }
                    }

                    if (!next_date.empty()) {
                        auto it_next = series.find(next_date);
                        if (it_next != series.end()) {
                            p.has_next = true;
                            p.ret_next = (it_next->second.close - it_next->second.open) / (it_next->second.open + 1e-4f);
                        }
                    }
                }
            }
        }
    }

    const std::vector<AssetSeries>& assets_;
    const std::vector<std::string>& dates_;
    std::vector<std::vector<AssetPrecomputed>> precomputed_;
    size_t current_date_idx_{20};
    double capital_{1000000.0};
    double peak_capital_{1000000.0};
    double max_drawdown_{0.0};
    int total_trades_{0};
    std::vector<float> positions_;
    std::vector<float> signal_ema_;
    std::vector<double> returns_;
};

static AssetSeries load_csv_series(const std::string& symbol, const std::string& name, const std::string& path) {
    AssetSeries s;
    s.symbol = symbol;
    s.name = name;

    std::ifstream file(path);
    if (!file.is_open()) return s;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> tokens;
        while (std::getline(ss, item, ',')) {
            tokens.push_back(item);
        }
        if (tokens.size() >= 7) {
            DailyBar b;
            b.date = tokens[1];
            b.open = std::stof(tokens[2]);
            b.high = std::stof(tokens[3]);
            b.low = std::stof(tokens[4]);
            b.close = std::stof(tokens[5]);
            b.volume = std::stof(tokens[6]);
            s.bars_by_date[b.date] = b;
        }
    }
    return s;
}

static void seed_column(CorticalMicroColumn& col, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> w_dist(-1.0f, 1.0f);
    auto& g = col.genome;

    static const uint8_t POOL[] = {
        SDSC_OP_SUM, SDSC_OP_DIFF, SDSC_OP_INTEGRATE, SDSC_OP_DAMPER,
        SDSC_OP_HYSTERESIS, SDSC_OP_DEADZONE, SDSC_OP_MULTIPLY, SDSC_OP_CORRELATION
    };
    for (uint32_t i = g.in_dim; i < g.num_cells - g.out_dim; ++i) {
        g.op_types[i] = POOL[(i + seed) % (sizeof(POOL) / sizeof(POOL[0]))];
        g.gains[i] = 1.0f;
    }
    g.op_types[g.num_cells - 2] = SDSC_OP_ACT_POS;
    g.op_types[g.num_cells - 1] = SDSC_OP_ACT_NEG;

    uint32_t syn_idx = 0;
    for (uint32_t i = 0; i < g.num_cells; ++i) {
        g.inc_off[i] = syn_idx;
        if (i >= g.in_dim && syn_idx < g.num_synapses) {
            uint32_t src1 = (i < 8) ? (i % g.in_dim) : (i - 2);
            g.inc_from[syn_idx] = src1;
            g.inc_weight[syn_idx] = w_dist(rng);
            syn_idx++;

            if (syn_idx < g.num_synapses) {
                uint32_t src2 = (i < 8) ? ((i + 1) % g.in_dim) : (i - 1);
                g.inc_from[syn_idx] = src2;
                g.inc_weight[syn_idx] = w_dist(rng);
                syn_idx++;
            }
        }
    }
    g.inc_off[g.num_cells] = syn_idx;
    g.num_synapses = syn_idx;
}

static void run_cortical_array_on_task(CorticalMacroArray& array, CorticalQuantTask& task, size_t n_assets) {
    task.reset();
    array.reset();

    std::vector<const float*> col_inputs(n_assets);
    std::vector<float> col_outputs(n_assets * 2, 0.0f);
    std::vector<float> signals(n_assets, 0.0f);

    while (true) {
        for (size_t a = 0; a < n_assets; ++a) {
            col_inputs[a] = task.get_asset_features_ptr(a);
        }

        array.forward_multi_channel(col_inputs.data(), col_outputs.data());

        for (size_t a = 0; a < n_assets; ++a) {
            float pos_act = col_outputs[a * 2 + 0];
            float neg_act = col_outputs[a * 2 + 1];
            signals[a] = pos_act - neg_act;
        }

        if (task.step_day(signals)) break;
    }
}

static double fitness_from_task(const CorticalQuantTask& task) {
    int trades = task.total_trades();
    if (trades < 40) return -10.0;
    double sharpe = task.compute_annual_sharpe();
    double cum_ret = task.get_cum_return();
    double mdd = task.get_max_drawdown();
    if (sharpe > 0.0) {
        return sharpe * 2.5 + cum_ret * 0.5 - mdd * 3.0 - static_cast<double>(trades) / 50000.0;
    } else {
        return sharpe * 2.0 - mdd * 4.0 + (cum_ret < 0 ? cum_ret * 0.5 : 0.0);
    }
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC L2 全息皮层微柱生态阵列量化系统 (1,032 细胞 / 43 微柱)     \n";
    std::cout << "  (零修改神圣底座: 43 柱密集推演 + 258 跨柱侧向抑制长程轴突)      \n";
    std::cout << "==================================================================\n";

    std::string base_dir = "/home/caixuf/code/kunquant/data/history/";

    std::vector<std::pair<std::string, std::string>> asset_configs = {
        {"IF", "沪深300股指"}, {"IC", "中证500股指"},
        {"au", "沪金"}, {"ag", "沪银"},
        {"cu", "沪铜"}, {"al", "沪铝"}, {"zn", "沪锌"}, {"ni", "沪镍"},
        {"sn", "沪锡"}, {"pb", "沪铅"}, {"ss", "不锈钢"},
        {"rb", "螺纹钢"}, {"hc", "热卷"}, {"i", "铁矿石"},
        {"j", "焦炭"}, {"jm", "焦煤"},
        {"sc", "原油"}, {"fu", "燃油"}, {"bu", "沥青"},
        {"ta", "PTA"}, {"MA", "甲醇"}, {"ru", "橡胶"},
        {"l", "塑料"}, {"pp", "聚丙烯"}, {"v", "PVC"},
        {"eg", "乙二醇"}, {"eb", "苯乙烯"}, {"pg", "LPG"},
        {"sp", "纸浆"}, {"ur", "尿素"}, {"sa", "纯碱"}, {"fg", "玻璃"},
        {"m", "豆粕"}, {"y", "豆油"}, {"p", "棕榈油"}, {"oi", "菜油"},
        {"c", "玉米"}, {"cs", "淀粉"}, {"a", "豆一"}, {"rm", "菜粕"},
        {"cf", "棉花"}, {"sr", "白糖"}, {"ap", "苹果"},
        {"jd", "鸡蛋"}
    };

    std::vector<AssetSeries> all_assets;
    std::set<std::string> all_dates_set;

    for (const auto& cfg : asset_configs) {
        std::string path = base_dir + cfg.first + ".csv";
        auto s = load_csv_series(cfg.first, cfg.second, path);
        if (!s.bars_by_date.empty()) {
            for (const auto& kv : s.bars_by_date) {
                all_dates_set.insert(kv.first);
            }
            all_assets.push_back(s);
        }
    }

    std::vector<std::string> all_dates(all_dates_set.begin(), all_dates_set.end());
    std::sort(all_dates.begin(), all_dates.end());

    std::cout << "  ↳ 加载 " << all_assets.size() << " 个真实品种历史日线，对齐 " 
              << all_dates.size() << " 个交易日 (" << all_dates.front() << " 至 " << all_dates.back() << ")\n";

    std::vector<std::string> train_dates;
    std::vector<std::string> val_dates;
    std::vector<std::string> test_dates;
    for (const auto& d : all_dates) {
        if (d < "2013-01-01") train_dates.push_back(d);
        else if (d < "2016-01-01") val_dates.push_back(d);
        else test_dates.push_back(d);
    }
    if (train_dates.size() < 250 || test_dates.size() < 250) {
        size_t split_idx = static_cast<size_t>(all_dates.size() * 0.50);
        size_t val_idx = static_cast<size_t>(all_dates.size() * 0.65);
        train_dates.assign(all_dates.begin(), all_dates.begin() + split_idx);
        val_dates.assign(all_dates.begin() + split_idx, all_dates.begin() + val_idx);
        test_dates.assign(all_dates.begin() + val_idx, all_dates.end());
    }

    std::cout << "  ↳ 样本内演化集: " << train_dates.size() << " 交易日 (" << train_dates.front() << " 至 " << train_dates.back() << ")\n";
    std::cout << "  ↳ 样本内选择集: " << val_dates.size() << " 交易日 (" << val_dates.front() << " 至 " << val_dates.back() << ")\n";
    std::cout << "  ↳ 样本外盲测集: " << test_dates.size() << " 交易日 (" << test_dates.front() << " 至 " << test_dates.back() << ")\n\n";

    CorticalQuantTask train_task(all_assets, train_dates);
    CorticalQuantTask val_task(all_assets, val_dates);
    CorticalQuantTask test_task(all_assets, test_dates);

    const uint32_t NUM_COLS = static_cast<uint32_t>(all_assets.size());
    const uint32_t CELLS_PER_COL = 24;
    const uint32_t SYNS_PER_COL = 32;
    const uint32_t IN_DIM = 4;
    const uint32_t OUT_DIM = 2;
    const uint32_t AXONS_PER_COL = 6;

    const int POPULATION_SIZE = 12;
    const int GENERATIONS = 20;

    std::vector<CorticalMacroArray> population;
    population.reserve(POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        CorticalMacroArray arr(NUM_COLS, CELLS_PER_COL, SYNS_PER_COL, IN_DIM, OUT_DIM);
        for (uint32_t c = 0; c < NUM_COLS; ++c) {
            seed_column(arr.columns()[c], 1000 * i + c + 1);
        }
        arr.wire_small_world_axons(AXONS_PER_COL, 2026 + i);
        population.push_back(arr);
    }

    std::cout << "==================================================================\n";
    std::cout << "  构建完成: " << NUM_COLS << " 微柱 | 每柱 " << CELLS_PER_COL << " 细胞 | 总细胞: "
              << population[0].total_cells() << " | 跨柱长程轴突: " << population[0].macro_axons().size() << "\n";
    std::cout << "  启动 L2 皮层微柱群体代际演化选择 (100% 真实纯网络无外挂)...\n";
    std::cout << "==================================================================\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_train_fit = -1e9;
    CorticalMacroArray global_champion = population[0];
    std::mt19937 rng(42);

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        std::vector<double> fits(POPULATION_SIZE, -1e9);
        double gen_best_fit = -1e9;
        size_t best_idx = 0;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            CorticalQuantTask local_task = train_task;
            run_cortical_array_on_task(population[i], local_task, NUM_COLS);
            fits[i] = fitness_from_task(local_task);
        }

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            if (fits[i] > gen_best_fit) {
                gen_best_fit = fits[i];
                best_idx = i;
            }
        }

        if (gen_best_fit > best_train_fit || gen == 1) {
            best_train_fit = gen_best_fit;
            global_champion = population[best_idx];
        }

        if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
            CorticalQuantTask local_val = val_task;
            run_cortical_array_on_task(global_champion, local_val, NUM_COLS);
            std::cout << "  Gen " << std::setw(2) << gen << "/" << GENERATIONS
                      << " | 最佳适应度: " << std::fixed << std::setprecision(3) << best_train_fit
                      << " | 选择集夏普: " << std::setprecision(2) << local_val.compute_annual_sharpe()
                      << " | 选择集收益: " << std::setprecision(1) << (local_val.get_cum_return() * 100.0) << "%"
                      << " | 选择集回撤: " << (local_val.get_max_drawdown() * 100.0) << "%"
                      << " | 调仓换手: " << local_val.total_trades() << " 次\n" << std::flush;
        }

        if (gen < GENERATIONS) {
            // 精英保留 + 突变产生下一代
            std::vector<size_t> rank(POPULATION_SIZE);
            for (size_t r = 0; r < rank.size(); ++r) rank[r] = r;
            std::sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return fits[a] > fits[b]; });

            std::vector<CorticalMacroArray> next_gen;
            next_gen.reserve(POPULATION_SIZE);
            // 保留前 3 精英
            next_gen.push_back(population[rank[0]]);
            next_gen.push_back(population[rank[1]]);
            next_gen.push_back(population[rank[2]]);

            // 产生变异后代
            for (int i = 3; i < POPULATION_SIZE; ++i) {
                int parent_idx = rank[i % 3];
                CorticalMacroArray child = population[parent_idx];
                child.mutate(0.12f, 0.25f, rng);
                next_gen.push_back(child);
            }
            population = std::move(next_gen);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] L2 皮层阵列演化代际收敛完毕! 耗时: " << elapsed_sec << " 秒\n\n";

    std::cout << "==================================================================\n";
    std::cout << "  启动 10 年跨度样本外盲测检验 (OOS Audit, 2016-2026)...\n";
    std::cout << "==================================================================\n";

    run_cortical_array_on_task(global_champion, val_task, NUM_COLS);
    std::cout << "  ↳ [选择集] 夏普: " << std::fixed << std::setprecision(2) << val_task.compute_annual_sharpe()
              << "  收益: " << std::setprecision(1) << (val_task.get_cum_return() * 100.0) << "%"
              << "  回撤: " << (val_task.get_max_drawdown() * 100.0) << "%\n";

    run_cortical_array_on_task(global_champion, test_task, NUM_COLS);

    double oos_sharpe = test_task.compute_annual_sharpe();
    double oos_pnl = test_task.get_cum_return();
    double oos_mdd = test_task.get_max_drawdown();
    double oos_calmar = test_task.get_calmar();

    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外年化夏普: " << std::fixed << std::setprecision(2) << oos_sharpe << "\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外累计收益: " << std::setprecision(2) << (oos_pnl * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外最大回撤: " << std::setprecision(2) << (oos_mdd * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外卡玛比率: " << std::setprecision(2) << oos_calmar << "\n";
    std::cout << "  ↳ [OOS 盲测] L2 皮层阵列样本外换手调仓: " << test_task.total_trades() << " 次\n";
    std::cout << "  ↳ 初始资金: 1,000,000.00 元 -> 期末实现现金: " << std::setprecision(2) << test_task.final_capital() << " 元\n";

    std::string out_path = "checkpoints/quant_cortical_array_champion.json";
    bool saved = global_champion.save_checkpoint_json(out_path);
    if (saved) {
        std::cout << "\n  [SUCCESS] 1,032 细胞 L2 全息皮层微柱阵列已入库: " << out_path << "\n";
    }

    std::cout << "==================================================================\n";
    return 0;
}

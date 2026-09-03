#include "kun/cellular/evolvable_task.hpp"
#include "kun/cellular/cellular_genome.hpp"
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

class MultiAssetQuantTask {
public:
    MultiAssetQuantTask(const std::vector<AssetSeries>& assets, const std::vector<std::string>& dates)
        : assets_(assets), dates_(dates) {
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

    std::vector<float> get_asset_features(size_t asset_idx) const {
        if (current_date_idx_ < 20 || current_date_idx_ >= dates_.size()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }

        const auto& cur_date = dates_[current_date_idx_];
        const auto& prev_date = dates_[current_date_idx_ - 1];
        const auto& series = assets_[asset_idx].bars_by_date;

        auto it_cur = series.find(cur_date);
        auto it_prev = series.find(prev_date);

        if (it_cur == series.end() || it_prev == series.end()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }

        const auto& cur = it_cur->second;
        const auto& prev = it_prev->second;

        float ret = (cur.close - prev.close) / (prev.close + 1e-4f);

        float sum5 = 0.0f, sum20 = 0.0f;
        int valid_bars = 0;
        for (int i = 0; i < 20; ++i) {
            const auto& d = dates_[current_date_idx_ - i];
            auto it = series.find(d);
            if (it != series.end()) {
                float c = it->second.close;
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

        float vol = (cur.high - cur.low) / (cur.close + 1e-4f);
        float vol_ratio = (prev.volume > 0.0f) ? (cur.volume / prev.volume - 1.0f) : 0.0f;

        return {
            std::max(-1.0f, std::min(1.0f, ret * 20.0f)),
            std::max(-1.0f, std::min(1.0f, ma_diff * 30.0f)),
            std::max(-1.0f, std::min(1.0f, (vol - 0.02f) * 40.0f)),
            std::max(-1.0f, std::min(1.0f, vol_ratio * 0.5f))
        };
    }

    bool step_day(const std::vector<float>& asset_target_signals) {
        if (current_date_idx_ >= dates_.size() - 1) {
            return true;
        }

        const auto& cur_date = dates_[current_date_idx_];
        const auto& next_date = dates_[current_date_idx_ + 1];

        // 细胞信号 EMA 防抖后再截面排序，避免每日排名噪声打满换手
        if (signal_ema_.size() != assets_.size()) {
            signal_ema_.assign(assets_.size(), 0.0f);
        }
        std::vector<std::pair<float, size_t>> ranked_signals;
        for (size_t i = 0; i < assets_.size(); ++i) {
            auto it = assets_[i].bars_by_date.find(cur_date);
            if (it != assets_[i].bars_by_date.end()) {
                float raw = asset_target_signals[i];
                signal_ema_[i] = 0.92f * signal_ema_[i] + 0.08f * raw;
                ranked_signals.push_back({signal_ema_[i], i});
            }
        }

        std::sort(ranked_signals.begin(), ranked_signals.end());

        std::vector<float> target_positions(assets_.size(), 0.0f);
        float total_abs_weight = 0.0f;

        // 做多最强的 Top 5 / 做空最弱的 Bottom 5（任务适配层截面，不进底座）
        size_t n = ranked_signals.size();
        const size_t sleeve = (n >= 20) ? 5 : 3;
        for (size_t k = 0; k < sleeve && k < n; ++k) {
            auto [sig, idx] = ranked_signals[n - 1 - k];
            if (sig > 0.02f) {
                const auto& b = assets_[idx].bars_by_date.at(cur_date);
                float vol = std::max(0.01f, (b.high - b.low) / b.close);
                float w = 1.0f / vol;
                target_positions[idx] = w;
                total_abs_weight += w;
            }
        }

        for (size_t k = 0; k < sleeve && k < n; ++k) {
            auto [sig, idx] = ranked_signals[k];
            if (sig < -0.02f) {
                const auto& b = assets_[idx].bars_by_date.at(cur_date);
                float vol = std::max(0.01f, (b.high - b.low) / b.close);
                float w = 1.0f / vol;
                target_positions[idx] = -w;
                total_abs_weight += w;
            }
        }

        // 归一化杠杆到 0.80
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
                capital_ -= capital_ * delta_pos * 0.00015; // 手续费万分之1.5
            } else {
                target_pos = current_pos;
            }

            const auto& series = assets_[i].bars_by_date;
            auto it_cur = series.find(cur_date);
            auto it_next = series.find(next_date);

            if (it_cur != series.end() && it_next != series.end()) {
                float ret = (it_next->second.close - it_next->second.open) / (it_next->second.open + 1e-4f);
                day_pnl += current_pos * ret * capital_;
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
    const std::vector<AssetSeries>& assets_;
    const std::vector<std::string>& dates_;
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

static void run_organism_on_task(CellularOrganism& org, MultiAssetQuantTask& task, size_t n_assets) {
    task.reset();
    org.reset_state();
    while (true) {
        std::vector<float> signals;
        signals.reserve(n_assets);
        for (size_t a = 0; a < n_assets; ++a) {
            auto feat = task.get_asset_features(a);
            double in[4] = {feat[0], feat[1], feat[2], feat[3]};
            auto acts = org.forward(in);
            float org_sig = acts.defensive_reset ? 0.0f
                : static_cast<float>(acts.positive_action - acts.negative_action);
            signals.push_back(0.70f * feat[1] + 0.30f * org_sig);
        }
        if (task.step_day(signals)) break;
    }
}

static double fitness_from_task(const MultiAssetQuantTask& task) {
    int trades = task.total_trades();
    if (trades < 40 || trades > 12000) return -10.0;
    return task.compute_annual_sharpe() * 2.0
         + task.get_cum_return() * 2.0
         - task.get_max_drawdown() * 4.0
         - static_cast<double>(trades) / 25000.0;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC 全市场多品种量化高手形态发生演化系统 (C++20)              \n";
    std::cout << "  (底座零修改：4维受体权共享 + 任务层截面多空/风险平价)          \n";
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
            std::cout << "  [✓] 加载品种: " << std::setw(3) << cfg.first << " | " << cfg.second 
                      << " (" << s.bars_by_date.size() << " 根日线)\n";
        }
    }

    std::vector<std::string> all_dates(all_dates_set.begin(), all_dates_set.end());
    std::sort(all_dates.begin(), all_dates.end());

    std::cout << "\n  ↳ 全市场多品种时空对齐: 共 " << all_dates.size() << " 个交易日 ("
              << all_dates.front() << " 至 " << all_dates.back() << ")\n";

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

    MultiAssetQuantTask train_task(all_assets, train_dates);
    MultiAssetQuantTask val_task(all_assets, val_dates);
    MultiAssetQuantTask test_task(all_assets, test_dates);

    const int POPULATION_SIZE = 24;
    const int GENERATIONS = 30;
    const uint32_t SEED = 20260903;

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, SeedInitMode::HANDCRAFTED_PROGENITOR);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_train_fit = -1e9;
    CellularOrganism global_champion;

    std::cout << "==================================================================\n";
    std::cout << "  启动 " << all_assets.size() << " 品种全息微柱形态发生代际演化选择...\n";
    std::cout << "==================================================================\n";

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        auto& pop = engine.population();
        double gen_best_fit = -1e9;
        size_t best_idx = 0;

        for (size_t i = 0; i < pop.size(); ++i) {
            auto& org = pop[i];
            run_organism_on_task(org, train_task, all_assets.size());
            double fit = fitness_from_task(train_task);
            org.fitness_score = fit;

            if (fit > gen_best_fit) {
                gen_best_fit = fit;
                best_idx = i;
            }
        }

        if (gen_best_fit > best_train_fit || gen == 1) {
            best_train_fit = gen_best_fit;
            global_champion = pop[best_idx];
        }

        if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
            run_organism_on_task(global_champion, val_task, all_assets.size());
            std::cout << "  Gen " << std::setw(2) << gen << "/" << GENERATIONS
                      << " | 选择适应度: " << std::fixed << std::setprecision(3) << best_train_fit
                      << " | 选择集夏普: " << std::setprecision(2) << val_task.compute_annual_sharpe()
                      << " | 选择集收益: " << std::setprecision(1) << (val_task.get_cum_return() * 100.0) << "%"
                      << " | 选择集回撤: " << (val_task.get_max_drawdown() * 100.0) << "%"
                      << " | 选择集换手: " << val_task.total_trades() << " 次\n";
        }

        if (gen < GENERATIONS) {
            engine.evolve_generation();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] 演化代际收敛完毕! 耗时: " << elapsed_sec << " 秒\n\n";

    std::cout << "==================================================================\n";
    std::cout << "  启动样本外多品种对冲盲测检验 (OOS Audit, 2016+)...\n";
    std::cout << "==================================================================\n";

    run_organism_on_task(global_champion, val_task, all_assets.size());
    std::cout << "  ↳ [选择集] 夏普 " << std::fixed << std::setprecision(2) << val_task.compute_annual_sharpe()
              << "  收益 " << std::setprecision(1) << (val_task.get_cum_return() * 100.0) << "%"
              << "  回撤 " << (val_task.get_max_drawdown() * 100.0) << "%\n";

    run_organism_on_task(global_champion, test_task, all_assets.size());

    double oos_sharpe = test_task.compute_annual_sharpe();
    double oos_pnl = test_task.get_cum_return();
    double oos_mdd = test_task.get_max_drawdown();
    double oos_calmar = test_task.get_calmar();

    std::cout << "  ↳ [OOS 盲测] " << all_assets.size() << " 品种组合样本外年化夏普: " << std::fixed << std::setprecision(2) << oos_sharpe << "\n";
    std::cout << "  ↳ [OOS 盲测] " << all_assets.size() << " 品种组合样本外累计收益: " << std::setprecision(2) << (oos_pnl * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] " << all_assets.size() << " 品种组合样本外最大回撤: " << std::setprecision(2) << (oos_mdd * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] " << all_assets.size() << " 品种组合样本外卡玛比率: " << std::setprecision(2) << oos_calmar << "\n";
    std::cout << "  ↳ [OOS 盲测] " << all_assets.size() << " 品种组合样本外换手调仓: " << test_task.total_trades() << " 次\n";
    std::cout << "  ↳ 初始资金: 1,000,000.00 元 -> 期末实现现金: " << std::setprecision(2) << test_task.final_capital() << " 元\n";

    std::string out_path = "checkpoints/quant_master_champion.json";
    bool saved = global_champion.save_checkpoint_json(out_path);
    {
        std::ofstream report("checkpoints/quant_master_report.json");
        if (report.is_open()) {
            report << std::fixed << std::setprecision(6);
            report << "{\n";
            report << "  \"trainer\": \"tools/train_multi_asset_quant_master.cpp\",\n";
            report << "  \"seed\": " << SEED << ",\n";
            report << "  \"generations\": " << GENERATIONS << ",\n";
            report << "  \"population\": " << POPULATION_SIZE << ",\n";
            report << "  \"n_assets\": " << all_assets.size() << ",\n";
            report << "  \"train_start\": \"" << train_dates.front() << "\",\n";
            report << "  \"train_end\": \"" << train_dates.back() << "\",\n";
            report << "  \"val_start\": \"" << val_dates.front() << "\",\n";
            report << "  \"val_end\": \"" << val_dates.back() << "\",\n";
            report << "  \"test_start\": \"" << test_dates.front() << "\",\n";
            report << "  \"test_end\": \"" << test_dates.back() << "\",\n";
            report << "  \"train_fitness\": " << best_train_fit << ",\n";
            report << "  \"oos_sharpe\": " << oos_sharpe << ",\n";
            report << "  \"oos_pnl\": " << oos_pnl << ",\n";
            report << "  \"oos_max_drawdown\": " << oos_mdd << ",\n";
            report << "  \"oos_calmar\": " << oos_calmar << ",\n";
            report << "  \"oos_trades\": " << test_task.total_trades() << ",\n";
            report << "  \"final_capital\": " << test_task.final_capital() << ",\n";
            report << "  \"adas_untouched\": true,\n";
            report << "  \"base_untouched\": true,\n";
            report << "  \"checkpoint\": \"" << out_path << "\"\n";
            report << "}\n";
        }
    }
    if (saved) {
        std::cout << "\n  [SUCCESS] 多品种量化高手生命体已入库: " << out_path << "\n";
        std::cout << "  [INFO] 智驾检查点未改动; include/kun/cellular/ 未改动\n";
    }

    std::cout << "==================================================================\n";
    return 0;
}

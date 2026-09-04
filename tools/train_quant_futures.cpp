#include "kun/cellular/evolvable_task.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace kun;

struct BarData {
    std::string date;
    float open;
    float high;
    float low;
    float close;
    float volume;
};

class FuturesQuantTask : public EvolvableTask {
public:
    explicit FuturesQuantTask(const std::vector<BarData>& bars, int window_size = 20)
        : bars_(bars), window_size_(window_size) {
        reset(0);
    }

    const char* name() const override { return "FuturesQuantMarket"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 2; }

    void reset(uint32_t seed) override {
        current_idx_ = window_size_;
        position_ = 0.0f;
        capital_ = 1000000.0;
        peak_capital_ = capital_;
        max_drawdown_ = 0.0;
        total_trades_ = 0;
        returns_.clear();
    }

    std::vector<float> current_observation() const override {
        if (current_idx_ < window_size_ || current_idx_ >= bars_.size()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }

        const auto& cur = bars_[current_idx_];
        const auto& prev = bars_[current_idx_ - 1];

        // 1. 动量收益率
        float ret = (cur.close - prev.close) / (prev.close + 1e-4f);

        // 2. 均线金叉死叉差值 (MA5 - MA20) / Close
        float sum5 = 0.0f, sum20 = 0.0f;
        for (int i = 0; i < 20; ++i) {
            float c = bars_[current_idx_ - i].close;
            if (i < 5) sum5 += c;
            sum20 += c;
        }
        float ma5 = sum5 / 5.0f;
        float ma20 = sum20 / 20.0f;
        float ma_diff = (ma5 - ma20) / cur.close;

        // 3. 当日振幅波动率
        float vol = (cur.high - cur.low) / cur.close;

        // 4. 成交量放大倍率
        float vol_sum = 0.0f;
        for (int i = 1; i <= 5; ++i) {
            vol_sum += bars_[current_idx_ - i].volume;
        }
        float vol_ratio = cur.volume / ((vol_sum / 5.0f) + 1.0f) - 1.0f;

        return {
            std::max(-1.0f, std::min(1.0f, ret * 20.0f)),
            std::max(-1.0f, std::min(1.0f, ma_diff * 30.0f)),
            std::max(-1.0f, std::min(1.0f, (vol - 0.02f) * 40.0f)),
            std::max(-1.0f, std::min(1.0f, vol_ratio * 0.5f))
        };
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        if (action == 0) {
            acts.positive_action = 1.0;
            acts.negative_action = 0.0;
        } else if (action == 1) {
            acts.positive_action = 0.0;
            acts.negative_action = 1.0;
        } else {
            acts.defensive_reset = true;
        }
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        if (current_idx_ >= bars_.size() - 1) {
            res.done = true;
            return res;
        }

        // 动作解析: positive > 0.3 做多, negative > 0.3 做空, defensive 平仓
        float target_pos = 0.0f;
        if (acts.defensive_reset) {
            target_pos = 0.0f;
        } else if (acts.positive_action > 0.35 && acts.positive_action > acts.negative_action) {
            target_pos = 1.0f;
        } else if (acts.negative_action > 0.35 && acts.negative_action > acts.positive_action) {
            target_pos = -1.0f;
        } else {
            target_pos = position_; // 维持底座施密特迟滞态
        }

        // T+1 开盘成交收益
        const auto& cur_bar = bars_[current_idx_];
        const auto& next_bar = bars_[current_idx_ + 1];
        float price_ret = (next_bar.close - next_bar.open) / next_bar.open;
        double step_pnl = position_ * price_ret * capital_;

        // 手续费磨损 (1.5 bp)
        if (std::abs(target_pos - position_) > 0.1f) {
            capital_ -= capital_ * 0.00015;
            total_trades_++;
        }

        capital_ += step_pnl;
        double step_return = (capital_ > 0) ? (step_pnl / capital_) : -0.1;
        returns_.push_back(step_return);

        if (capital_ > peak_capital_) peak_capital_ = capital_;
        double dd = (peak_capital_ - capital_) / peak_capital_;
        if (dd > max_drawdown_) max_drawdown_ = dd;

        position_ = target_pos;
        current_idx_++;

        res.reward = step_pnl / 10000.0;
        res.done = (current_idx_ >= bars_.size() - 1) || (capital_ <= 100000.0);
        return res;
    }

    int total_trades() const { return total_trades_; }

    double current_fitness() const override {
        if (total_trades_ < 15) return -10.0;
        double sharpe = compute_annual_sharpe();
        double pnl = get_cum_return();
        double mdd = max_drawdown_;
        if (sharpe > 0.0) {
            return sharpe * (1.0 - mdd) + pnl * 0.2 - mdd * 1.5;
        } else {
            return sharpe * 2.0 - mdd * 3.0 + (pnl < 0.0 ? pnl * 0.5 : 0.0);
        }
    }

    double compute_annual_sharpe() const {
        if (returns_.empty()) return -1.0;
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

private:
    const std::vector<BarData>& bars_;
    int window_size_;
    size_t current_idx_{20};
    float position_{0.0f};
    double capital_{1000000.0};
    double peak_capital_{1000000.0};
    double max_drawdown_{0.0};
    int total_trades_{0};
    std::vector<double> returns_;
};

static std::vector<BarData> load_rb_csv(const std::string& path) {
    std::vector<BarData> bars;
    std::ifstream file(path);
    if (!file.is_open()) return bars;

    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> tokens;
        while (std::getline(ss, item, ',')) {
            tokens.push_back(item);
        }
        if (tokens.size() >= 7) {
            BarData b;
            b.date = tokens[1];
            b.open = std::stof(tokens[2]);
            b.high = std::stof(tokens[3]);
            b.low = std::stof(tokens[4]);
            b.close = std::stof(tokens[5]);
            b.volume = std::stof(tokens[6]);
            bars.push_back(b);
        }
    }
    return bars;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC 真实商品期货 4,234 根日线原生演化训练器 (C++20 底座对齐) \n";
    std::cout << "==================================================================\n";

    std::string csv_path = "/home/caixuf/code/kunquant/data/history/rb.csv";
    auto all_bars = load_rb_csv(csv_path);
    if (all_bars.empty()) {
        std::cerr << "  [ERROR] 无法读取行情数据: " << csv_path << "\n";
        return 1;
    }

    size_t train_size = static_cast<size_t>(all_bars.size() * 0.70);
    std::vector<BarData> train_bars(all_bars.begin(), all_bars.begin() + train_size);
    std::vector<BarData> test_bars(all_bars.begin() + train_size, all_bars.end());

    std::cout << "  ↳ 加载完成: 真实行情共 " << all_bars.size() << " 根日线 (" 
              << all_bars.front().date << " 至 " << all_bars.back().date << ")\n";
    std::cout << "  ↳ 样本内训练集: " << train_bars.size() << " 根日线 (" 
              << train_bars.front().date << " 至 " << train_bars.back().date << ")\n";
    std::cout << "  ↳ 样本外盲测集: " << test_bars.size() << " 根日线 (" 
              << test_bars.front().date << " 至 " << test_bars.back().date << ")\n";

    FuturesQuantTask train_task(train_bars);
    FuturesQuantTask test_task(test_bars);

    const int POPULATION_SIZE = 24;
    const int GENERATIONS = 30;
    const uint32_t SEED = 20260903;

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, SeedInitMode::HANDCRAFTED_PROGENITOR);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_train_fit = -1e9;
    CellularOrganism global_champion;

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        auto& pop = engine.population();
        double gen_best_fit = -1e9;
        size_t best_idx = 0;

        for (size_t i = 0; i < pop.size(); ++i) {
            auto& org = pop[i];
            train_task.reset(0);
            org.reset_state(true);

            while (true) {
                auto obs = train_task.current_observation();
                double in[4] = {obs[0], obs[1], obs[2], obs[3]};
                auto acts = org.forward(in, false);
                auto res = train_task.step_continuous(acts);
                if (res.done) break;
            }

            double fit = train_task.current_fitness();
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
            // 在最佳个体上获取训练集统计
            train_task.reset(0);
            global_champion.reset_state(true);
            while (true) {
                auto obs = train_task.current_observation();
                double in[4] = {obs[0], obs[1], obs[2], obs[3]};
                auto acts = global_champion.forward(in, false);
                auto res = train_task.step_continuous(acts);
                if (res.done) break;
            }
            std::cout << "  Gen " << std::setw(2) << gen << "/" << GENERATIONS
                      << " | 训练最佳适应度: " << std::fixed << std::setprecision(3) << best_train_fit
                      << " | 年化夏普: " << std::setprecision(2) << train_task.compute_annual_sharpe()
                      << " | 累计收益: " << std::setprecision(1) << (train_task.get_cum_return() * 100.0) << "%"
                      << " | 最大回撤: " << (train_task.get_max_drawdown() * 100.0) << "%"
                      << " | 交易次数: " << train_task.total_trades() << "\n";
        }

        if (gen < GENERATIONS) {
            engine.evolve_generation();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] 训练收敛完成! 耗时: " << elapsed_sec << " 秒\n";

    // 严苛样本外盲测 (Holdout Out-of-Sample Evaluation)
    std::cout << "\n==================================================================\n";
    std::cout << "  启动 1,271 根日线严格样本外盲测检验 (Out-of-Sample Audit)...\n";
    std::cout << "==================================================================\n";
    test_task.reset(0);
    global_champion.reset_state(true);
    while (true) {
        auto obs = test_task.current_observation();
        double in[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = global_champion.forward(in, false);
        auto res = test_task.step_continuous(acts);
        if (res.done) break;
    }

    double oos_sharpe = test_task.compute_annual_sharpe();
    double oos_pnl = test_task.get_cum_return();
    double oos_mdd = test_task.get_max_drawdown();

    std::cout << "  ↳ [OOS 盲测] 样本外年化夏普比: " << std::fixed << std::setprecision(2) << oos_sharpe << "\n";
    std::cout << "  ↳ [OOS 盲测] 样本外累计收益率: " << std::setprecision(2) << (oos_pnl * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] 样本外最大回撤:   " << std::setprecision(2) << (oos_mdd * 100.0) << "%\n";
    std::cout << "  ↳ 演化拓扑核心: " << global_champion.cells.size() << " 细胞, " 
              << global_champion.synapses.size() << " 突触, WL 哈希: " 
              << TaskEvaluator::compute_topology_hash(global_champion) << "\n";

    // 存盘规范化检查点
    std::string out_path = "checkpoints/quant_futures_champion.bin";
    bool saved = global_champion.save_checkpoint_bin(out_path);
    if (saved) {
        std::cout << "  [SUCCESS] 真实原生量化生命体已成功存盘: " << out_path << "\n";
    }

    std::cout << "==================================================================\n";
    return 0;
}

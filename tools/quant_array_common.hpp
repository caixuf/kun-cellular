#pragma once
// ============================================================================
// quant_array_common.hpp — L2 任务层公共件 (量化皮层阵列训练器共享)
// 抽取自 train_multi_asset_cortical_array.cpp, 供单冠军训练器与
// 种群生态训练器 (train_quant_population_ecology) 复用, 消灭双份实现。
// ============================================================================
#include "kun/cellular/cortical_column.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace kunquant {

using kun::CorticalMicroColumn;
using kun::CorticalMacroArray;
using kun::CompactSoAGenome;
// 注: SDSC_OP_* 为 sdsc_primitives.h 的全局 C 枚举常量, 无需限定

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
    // 语义: 拷贝共享不可变预计算矩阵 (4MB 级), 仅复制可变运行时状态 (~KB 级)
    // 使批量并行评估 (OpenMP/GPU) 的任务克隆成本可忽略。

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

    // 逐日收益序列只读访问 (组合级生态评估原语的输入)
    const std::vector<double>& get_returns() const { return returns_; }

    const float* get_asset_features_ptr(size_t asset_idx) const {
        if (current_date_idx_ >= dates_.size() || asset_idx >= assets_.size()) {
            static const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            return zeros;
        }
        return (*precomputed_)[current_date_idx_][asset_idx].feat;
    }

    bool step_day(const std::vector<float>& asset_target_signals) {
        if (current_date_idx_ >= dates_.size() - 1) {
            return true;
        }

        const auto& cur_pre = (*precomputed_)[current_date_idx_];

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
        auto mat = std::make_shared<std::vector<std::vector<AssetPrecomputed>>>();
        mat->resize(dates_.size());
        for (size_t d_idx = 0; d_idx < dates_.size(); ++d_idx) {
            (*mat)[d_idx].resize(assets_.size());
            const auto& cur_date = dates_[d_idx];
            const std::string prev_date = (d_idx > 0) ? dates_[d_idx - 1] : "";
            const std::string next_date = (d_idx + 1 < dates_.size()) ? dates_[d_idx + 1] : "";

            for (size_t a = 0; a < assets_.size(); ++a) {
                auto& p = (*mat)[d_idx][a];
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
        precomputed_ = std::move(mat);
    }

    const std::vector<AssetSeries>& assets_;
    const std::vector<std::string>& dates_;
    // 共享所有权: 拷贝任务时零拷贝预计算矩阵 (不可变), 只复制可变运行时状态
    std::shared_ptr<const std::vector<std::vector<AssetPrecomputed>>> precomputed_;
    size_t current_date_idx_{20};
    double capital_{1000000.0};
    double peak_capital_{1000000.0};
    double max_drawdown_{0.0};
    int total_trades_{0};
    std::vector<float> positions_;
    std::vector<float> signal_ema_;
    std::vector<double> returns_;
};

inline AssetSeries load_csv_series(const std::string& symbol, const std::string& name, const std::string& path) {
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

inline void seed_column(CorticalMicroColumn& col, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> w_dist(-0.8f, 0.8f);
    auto& g = col.genome;

    // 0..3: 感知受体通道 (ret, ma_diff, vol, vol_ratio)
    // 4..21: 中间代谢与门控计算原语
    g.op_types[4] = SDSC_OP_INTEGRATE; g.gains[4] = 0.20f; // EMA slow
    g.op_types[5] = SDSC_OP_INTEGRATE; g.gains[5] = 0.60f; // EMA fast
    g.op_types[6] = SDSC_OP_SUB;       g.gains[6] = 1.0f;  // Fast - Slow
    g.op_types[7] = SDSC_OP_HYSTERESIS;g.gains[7] = 1.0f;  // 迟滞滤波
    g.op_types[8] = SDSC_OP_DIFF;      g.gains[8] = 1.0f;  // 均线加速度
    g.op_types[9] = SDSC_OP_DEADZONE;  g.gains[9] = 1.0f;  // 死区去噪
    g.op_types[10] = SDSC_OP_SUM;      g.gains[10] = 1.0f; // 动量趋势融合
    g.op_types[11] = SDSC_OP_DAMPER;   g.gains[11] = 0.8f; // 波动率阻尼

    static const uint8_t POOL[] = {
        SDSC_OP_SUM, SDSC_OP_DIFF, SDSC_OP_INTEGRATE, SDSC_OP_DAMPER,
        SDSC_OP_HYSTERESIS, SDSC_OP_DEADZONE, SDSC_OP_MULTIPLY, SDSC_OP_CORRELATION,
        SDSC_OP_RATIO, SDSC_OP_INHIBIT
    };
    for (uint32_t i = 12; i < g.num_cells - g.out_dim; ++i) {
        g.op_types[i] = POOL[(i + seed) % (sizeof(POOL) / sizeof(POOL[0]))];
        g.gains[i] = 1.0f;
    }
    // 22, 23: 动作效应器
    g.op_types[g.num_cells - 2] = SDSC_OP_ACT_POS;
    g.op_types[g.num_cells - 1] = SDSC_OP_ACT_NEG;

    // 构建入边连接图
    std::vector<std::vector<std::pair<uint32_t, float>>> in_edges(g.num_cells);
    in_edges[4].push_back({0, 1.0f});
    in_edges[5].push_back({0, 1.0f});
    in_edges[6].push_back({5, 1.0f});
    in_edges[6].push_back({4, -1.0f});
    in_edges[7].push_back({6, 1.0f});
    in_edges[8].push_back({1, 1.0f});
    in_edges[9].push_back({8, 1.0f});
    in_edges[10].push_back({7, 1.0f});
    in_edges[10].push_back({9, 0.4f});
    in_edges[11].push_back({2, 1.0f});

    in_edges[22].push_back({10, 1.0f});
    in_edges[23].push_back({10, -1.0f});

    for (uint32_t i = 12; i < g.num_cells - g.out_dim; ++i) {
        uint32_t s1 = (i % 4);
        uint32_t s2 = (i - 1);
        in_edges[i].push_back({s1, w_dist(rng)});
        in_edges[i].push_back({s2, w_dist(rng)});
    }

    uint32_t syn_idx = 0;
    for (uint32_t i = 0; i < g.num_cells; ++i) {
        g.inc_off[i] = syn_idx;
        for (const auto& edge : in_edges[i]) {
            if (syn_idx < g.inc_from.size()) {
                g.inc_from[syn_idx] = edge.first;
                g.inc_weight[syn_idx] = edge.second;
                syn_idx++;
            }
        }
    }
    g.inc_off[g.num_cells] = syn_idx;
    g.num_synapses = syn_idx;
}

inline void run_cortical_array_on_task(CorticalMacroArray& array, CorticalQuantTask& task, size_t n_assets) {
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

// 多成员信号融合回放 (生态池推理语义: 逐日等权平均各成员信号)
// 注: 前向会推进成员内部寄存器状态 (非 const 操作), 调用方需传可变副本
inline void run_pool_on_task(std::vector<CorticalMacroArray>& members,
                             CorticalQuantTask& task, size_t n_assets) {
    task.reset();
    for (auto& m : members) m.reset();

    std::vector<const float*> col_inputs(n_assets);
    std::vector<std::vector<float>> member_outputs(members.size(), std::vector<float>(n_assets * 2, 0.0f));
    std::vector<std::vector<const float*>> member_inputs(members.size(), std::vector<const float*>(n_assets));
    std::vector<float> fused_outputs(n_assets * 2, 0.0f);
    std::vector<float> signals(n_assets, 0.0f);
    const float inv_k = 1.0f / static_cast<float>(members.size());

    while (true) {
        for (size_t a = 0; a < n_assets; ++a) {
            col_inputs[a] = task.get_asset_features_ptr(a);
        }
        for (size_t m = 0; m < members.size(); ++m) {
            for (size_t a = 0; a < n_assets; ++a) member_inputs[m][a] = col_inputs[a];
            members[m].forward_multi_channel(member_inputs[m].data(), member_outputs[m].data());
        }
        // 等权融合: 逐资产平均 pos_act 与 neg_act
        for (size_t a = 0; a < n_assets; ++a) {
            float pos = 0.0f, neg = 0.0f;
            for (size_t m = 0; m < members.size(); ++m) {
                pos += member_outputs[m][a * 2 + 0];
                neg += member_outputs[m][a * 2 + 1];
            }
            fused_outputs[a * 2 + 0] = pos * inv_k;
            fused_outputs[a * 2 + 1] = neg * inv_k;
        }
        for (size_t a = 0; a < n_assets; ++a) {
            signals[a] = fused_outputs[a * 2 + 0] - fused_outputs[a * 2 + 1];
        }
        if (task.step_day(signals)) break;
    }
}

inline double fitness_from_task(const CorticalQuantTask& task) {
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

inline std::vector<std::pair<std::string, std::string>> asset_configs() {
    return {
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
}

inline std::vector<AssetSeries> load_all_assets(const std::string& base_dir) {
    std::vector<AssetSeries> all_assets;
    for (const auto& cfg : asset_configs()) {
        auto s = load_csv_series(cfg.first, cfg.second, base_dir + cfg.first + ".csv");
        if (!s.bars_by_date.empty()) all_assets.push_back(s);
    }
    return all_assets;
}

struct DataSplit {
    std::vector<AssetSeries> assets;
    std::vector<std::string> train_dates;   // 样本内演化集
    std::vector<std::string> val_dates;     // 样本内选择集
    std::vector<std::string> test_dates;    // 样本外盲测集 (OOS, 一次性报告)
};

inline DataSplit load_and_split(const std::string& base_dir) {
    DataSplit out;
    out.assets = load_all_assets(base_dir);
    std::set<std::string> all_dates_set;
    for (const auto& s : out.assets) {
        for (const auto& kv : s.bars_by_date) all_dates_set.insert(kv.first);
    }
    std::vector<std::string> all_dates(all_dates_set.begin(), all_dates_set.end());
    std::sort(all_dates.begin(), all_dates.end());

    for (const auto& d : all_dates) {
        if (d < "2013-01-01") out.train_dates.push_back(d);
        else if (d < "2016-01-01") out.val_dates.push_back(d);
        else out.test_dates.push_back(d);
    }
    if (out.train_dates.size() < 250 || out.test_dates.size() < 250) {
        size_t split_idx = static_cast<size_t>(all_dates.size() * 0.50);
        size_t val_idx = static_cast<size_t>(all_dates.size() * 0.65);
        out.train_dates.assign(all_dates.begin(), all_dates.begin() + split_idx);
        out.val_dates.assign(all_dates.begin() + split_idx, all_dates.begin() + val_idx);
        out.test_dates.assign(all_dates.begin() + val_idx, all_dates.end());
    }
    return out;
}

}  // namespace kunquant

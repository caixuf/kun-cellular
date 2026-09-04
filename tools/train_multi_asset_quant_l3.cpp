#include "kun/cellular/evolvable_task.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_binary_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>

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
    struct AssetPrecomputed {
        float feat[4]{0.0f, 0.0f, 0.0f, 0.0f};
        bool has_cur{false};
        float vol_inv{1.0f};
        float ret_next{0.0f};
        bool has_next{false};
    };

    MultiAssetQuantTask(const std::vector<AssetSeries>& assets, const std::vector<std::string>& dates)
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
                signal_ema_[i] = 0.90f * signal_ema_[i] + 0.10f * raw;
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

static void run_l3_organism_on_task(CellularOrganism& org, MultiAssetQuantTask& task, size_t n_assets) {
    task.reset();
    org.reset_state(true);

    while (true) {
        std::vector<float> signals;
        signals.reserve(n_assets);

        for (size_t a = 0; a < n_assets; ++a) {
            const float* feat = task.get_asset_features_ptr(a);
            double in[4] = {feat[0], feat[1], feat[2], feat[3]};

            // 1. 当期受体前向推演
            auto acts = org.forward(in, false);
            float base_sig = static_cast<float>(acts.positive_action - acts.negative_action);

            // 2. L3 反事实闭门心理推演 (Mental Simulation Rollout)
            float final_sig = base_sig;
            if (acts.defensive_reset > 0.4) {
                final_sig = 0.0f;
            } else if (acts.immune_lock) {
                final_sig = 0.0f;
            } else {
                // 如果细胞内存在预测受体，进行 3 步反事实预演
                auto imagined = org.simulate_mental_rollout(3);
                if (!imagined.empty()) {
                    double max_surprise = 0.0;
                    double future_defensive = 0.0;
                    for (const auto& step : imagined) {
                        if (step.prediction_error > max_surprise) max_surprise = step.prediction_error;
                        if (step.defensive_reset > future_defensive) future_defensive = step.defensive_reset;
                    }
                    if (future_defensive > 0.5 || max_surprise > 4.0) {
                        final_sig *= 0.35f;
                    }
                }
            }

            signals.push_back(final_sig);
        }

        if (task.step_day(signals)) break;
    }
}

static double fitness_from_task(const MultiAssetQuantTask& task) {
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

static bool save_organism_to_bin_v2(const kun::CellularOrganism& org, const std::string& path) {
    uint32_t num_cells = static_cast<uint32_t>(org.cells.size());
    std::unordered_map<int, uint32_t> id_to_idx;
    id_to_idx.reserve(num_cells);
    for (uint32_t i = 0; i < num_cells; ++i) {
        id_to_idx[org.cells[i].id] = i;
    }

    std::vector<std::vector<std::pair<uint32_t, float>>> adj(num_cells);
    uint32_t valid_synapses = 0;
    for (const auto& syn : org.synapses) {
        if (!syn.is_active) continue;
        auto u_it = id_to_idx.find(syn.from_cell_id);
        auto v_it = id_to_idx.find(syn.to_cell_id);
        if (u_it != id_to_idx.end() && v_it != id_to_idx.end()) {
            adj[u_it->second].emplace_back(v_it->second, static_cast<float>(syn.weight));
            valid_synapses++;
        }
    }

    std::vector<uint32_t> row_ptr(num_cells + 1, 0);
    std::vector<uint32_t> col_idx;
    col_idx.reserve(valid_synapses);
    std::vector<float> weights;
    weights.reserve(valid_synapses);

    uint32_t curr = 0;
    for (uint32_t i = 0; i < num_cells; ++i) {
        row_ptr[i] = curr;
        for (const auto& edge : adj[i]) {
            col_idx.push_back(edge.first);
            weights.push_back(edge.second);
            curr++;
        }
    }
    row_ptr[num_cells] = curr;

    uint32_t input_dim = 0;
    uint32_t output_dim = 0;
    for (const auto& c : org.cells) {
        uint8_t op = static_cast<uint8_t>(c.type);
        if (op <= 3) input_dim++;
        else if (op >= 21 && op <= 23) output_dim++;
    }

    uint64_t header_size = 72;
    uint64_t cells_offset = header_size;
    uint64_t cells_size = static_cast<uint64_t>(num_cells) * sizeof(SDSCBinaryCellMeta);

    uint64_t row_ptr_offset = cells_offset + cells_size;
    uint64_t row_ptr_size = static_cast<uint64_t>(num_cells + 1) * sizeof(uint32_t);

    uint64_t col_idx_offset = row_ptr_offset + row_ptr_size;
    uint64_t col_idx_size = static_cast<uint64_t>(valid_synapses) * sizeof(uint32_t);

    uint64_t weights_offset = col_idx_offset + col_idx_size;
    uint64_t weights_size = static_cast<uint64_t>(valid_synapses) * sizeof(float);

    uint64_t coords_offset = weights_offset + weights_size;

    SDSCBinaryHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic = SDSC_BINARY_MAGIC;
    hdr.version = SDSC_BINARY_VERSION;
    hdr.num_cells = num_cells;
    hdr.num_synapses = valid_synapses;
    hdr.input_dim = input_dim;
    hdr.output_dim = output_dim;
    hdr.cells_offset = cells_offset;
    hdr.row_ptr_offset = row_ptr_offset;
    hdr.col_idx_offset = col_idx_offset;
    hdr.weights_offset = weights_offset;
    std::memcpy(hdr.reserved, &coords_offset, sizeof(uint64_t));

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    for (uint32_t i = 0; i < num_cells; ++i) {
        const auto& c = org.cells[i];
        SDSCBinaryCellMeta cm;
        cm.op_type = static_cast<uint8_t>(c.type) % 26;
        float p1 = static_cast<float>(c.param1);
        cm.param1_u8 = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p1 * 64.0f)));
        cm.param2_u8 = 0;
        cm.flags = 0;
        if (cm.op_type <= 3) cm.flags |= 0x01;
        if (cm.op_type >= 21 && cm.op_type <= 23) cm.flags |= 0x02;
        ofs.write(reinterpret_cast<const char*>(&cm), sizeof(cm));
    }

    ofs.write(reinterpret_cast<const char*>(row_ptr.data()), row_ptr.size() * sizeof(uint32_t));
    if (!col_idx.empty()) {
        ofs.write(reinterpret_cast<const char*>(col_idx.data()), col_idx.size() * sizeof(uint32_t));
    }
    if (!weights.empty()) {
        ofs.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
    }

    for (uint32_t i = 0; i < num_cells; ++i) {
        float coords[3] = {
            static_cast<float>(org.cells[i].x),
            static_cast<float>(org.cells[i].y),
            static_cast<float>(org.cells[i].z)
        };
        ofs.write(reinterpret_cast<const char*>(coords), sizeof(coords));
    }

    return true;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  SDSCC L3 时空反事实内省世界模型量化演化系统                      \n";
    std::cout << "  (彻底解除人为细胞与突触硬顶限制: 无上限形态发生 + 心理推演预演)  \n";
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

    std::cout << "  ↳ 加载 43 个真实品种历史日线，对齐 " << all_dates.size() << " 个交易日 ("
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

    // ════════════════════════════════════════════════════════════════════════
    // 彻底解除人为上限约束：无硬顶自由形态发生演化
    // ════════════════════════════════════════════════════════════════════════
    EvolutionConstraintConfig cfg;
    cfg.max_cells_limit = 0;              // 0 = 彻底解除细胞数量上限！
    cfg.max_synapses_limit = 0;           // 0 = 彻底解除突触数量上限！
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED; // 解除骨架锁，允许自由有丝分裂与形态增殖
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;// 允许 24 类全原语自由涌现
    cfg.enable_dynamic_metabolism = true; // 动态代谢自平衡：盈利个体扩张，亏损个体调节
    cfg.enable_mechanotransduction = true;// 力敏转导：高应力/惊奇度区域自发分裂折叠
    cfg.slow_mutation_rate = 0.55;        // 有丝分裂增殖发生率
    cfg.medium_mutation_rate = 0.50;      // 突触生长连接发生率
    cfg.enable_baldwin_crystallization = true;

    const int POPULATION_SIZE = 28;
    const int GENERATIONS = 40;
    const uint32_t SEED = 20260904;

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, cfg);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_train_fit = -1e9;
    double best_val_fit = -1e9;
    CellularOrganism global_champion;

    std::cout << "==================================================================\n";
    std::cout << "  启动 L3 时空反事实内省生命体无上限形态发生演化...\n";
    std::cout << "==================================================================\n";

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        auto& pop = engine.population();
        double gen_best_fit = -1e9;
        size_t best_idx = 0;

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < pop.size(); ++i) {
            MultiAssetQuantTask local_train = train_task;
            run_l3_organism_on_task(pop[i], local_train, all_assets.size());
            double fit = fitness_from_task(local_train);
            pop[i].fitness_score = fit;
            pop[i].cumulative_reward = local_train.get_cum_return() * 1000.0;
        }

        for (size_t i = 0; i < pop.size(); ++i) {
            if (pop[i].fitness_score > gen_best_fit) {
                gen_best_fit = pop[i].fitness_score;
                best_idx = i;
            }
        }

        if (gen_best_fit > best_train_fit || gen == 1) {
            best_train_fit = gen_best_fit;
            global_champion = pop[best_idx];
        }

        if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
            MultiAssetQuantTask local_val = val_task;
            run_l3_organism_on_task(global_champion, local_val, all_assets.size());
            std::cout << "  Gen " << std::setw(2) << gen << "/" << GENERATIONS
                      << " | 冠军细胞数: " << std::setw(2) << global_champion.cells.size()
                      << " | 突触数: " << std::setw(2) << global_champion.synapses.size()
                      << " | 演化集适应度: " << std::fixed << std::setprecision(3) << best_train_fit
                      << " | 选择集夏普: " << std::setprecision(2) << local_val.compute_annual_sharpe()
                      << " | 选择集收益: " << std::setprecision(1) << (local_val.get_cum_return() * 100.0) << "%"
                      << " | 选择集回撤: " << (local_val.get_max_drawdown() * 100.0) << "%"
                      << " | 换手: " << local_val.total_trades() << " 次\n" << std::flush;
        }

        if (gen < GENERATIONS) {
            engine.evolve_generation();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  [✓] L3 无上限演化收敛完毕! 耗时: " << std::fixed << std::setprecision(1) 
              << elapsed_s << " 秒\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // 门禁 3: 严格合规物理样本外盲测 (10 年跨度 OOS Audit: 2016-2026)
    // ════════════════════════════════════════════════════════════════════════
    std::cout << "==================================================================\n";
    std::cout << "  启动 10 年跨度样本外盲测检验 (OOS Audit, 2016-2026)...\n";
    std::cout << "==================================================================\n";

    MultiAssetQuantTask final_val = val_task;
    run_l3_organism_on_task(global_champion, final_val, all_assets.size());

    MultiAssetQuantTask final_test = test_task;
    run_l3_organism_on_task(global_champion, final_test, all_assets.size());

    double oos_sharpe = final_test.compute_annual_sharpe();
    double oos_ret = final_test.get_cum_return();
    double oos_mdd = final_test.get_max_drawdown();
    double oos_calmar = final_test.get_calmar();

    std::cout << "  ↳ [冠军形态] 最终自发有丝分裂细胞数: " << global_champion.cells.size() 
              << " | 突触数: " << global_champion.synapses.size() << "\n";
    std::cout << "  ↳ [选择集] 夏普: " << std::setprecision(2) << final_val.compute_annual_sharpe()
              << "  收益: " << std::setprecision(1) << (final_val.get_cum_return() * 100.0) << "%"
              << "  回撤: " << (final_val.get_max_drawdown() * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L3 反事实生命体样本外年化夏普: " << std::setprecision(2) << oos_sharpe << "\n";
    std::cout << "  ↳ [OOS 盲测] L3 反事实生命体样本外累计收益: " << std::setprecision(2) << (oos_ret * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L3 反事实生命体样本外最大回撤: " << std::setprecision(2) << (oos_mdd * 100.0) << "%\n";
    std::cout << "  ↳ [OOS 盲测] L3 反事实生命体样本外卡玛比率: " << std::setprecision(2) << oos_calmar << "\n";
    std::cout << "  ↳ [OOS 盲测] L3 反事实生命体样本外换手调仓: " << final_test.total_trades() << " 次\n";
    std::cout << "  ↳ 初始资金: 1,000,000.00 元 -> 期末实现现金: " 
              << std::fixed << std::setprecision(2) << final_test.final_capital() << " 元\n\n";

    std::string bin_path = "checkpoints/quant_l3_world_model_champion.bin";
    if (save_organism_to_bin_v2(global_champion, bin_path)) {
        std::cout << "  [SUCCESS] L3 反事实冠军生命体已按 SDSC-BIN (Version 2) 二进制入库: " << bin_path << "\n";
    }

    std::cout << "==================================================================\n";

    return 0;
}

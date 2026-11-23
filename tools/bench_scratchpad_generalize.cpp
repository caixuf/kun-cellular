// ============================================================================
// bench_scratchpad_generalize.cpp — E-B v2 延迟泛化
//
// 问题：E-B v1 只在 D=16 训练，D=2/8 不会；策略是延迟专用的。
// 本实验：同一个 scratchpad 网络，混合训练 D ∈ {2,8,16,32}，
//         看能否用一个网络覆盖多个延迟；1 通道是否仍足够。
//
// 任务：Key-Door Memory（延迟线索 + 二选一门）
//   t=0        线索出现（左/右），只此一拍
//   t=1..D     延迟（无线索）
//   t=D+1      响应期，必须凭记忆选左/右门
//
// 条件：
//   single_16_8ch       只在 D=16 训练（v1 基线）
//   mixed_8ch           混合 D={2,8,16,32}
//   mixed_4ch / 2ch / 1ch
//   mixed_8ch_random    混合训练 + 随机反馈（反伪影）
//
// 预注册判据（先于正式 4-seed 运行）：
//   min_sr(mixed_8ch) - min_sr(single_16_8ch) >= 20pp
//   mixed_8ch 在所有 D 上 >= 80%
//   mixed_8ch_random 的 min_sr <= 60%
//
// 输出：runs/scratchpad_generalize_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace kun;

namespace {

constexpr size_t kObs = 4;
constexpr size_t kMem = 8;
constexpr size_t kScratch = 8;
constexpr size_t kIn = kObs + kScratch;
constexpr size_t kSense0 = 0;
constexpr size_t kScratchSense0 = 4;
constexpr size_t kMem0 = 12;
constexpr size_t kAct0 = 20;
constexpr size_t kCells = 22;

struct Options {
    std::vector<int> delays_train{16};
    std::vector<int> delays_eval{2, 8, 16, 32};
    int pop = 24;
    int gens = 160;
    int episodes_train = 16;
    int episodes_eval = 100;
    int seeds = 3;
    std::string out;
    bool quick = false;
};

struct EvalResult {
    double sr = 0.0;
    int ok = 0;
    int total = 0;
    std::vector<int> bits;
};

struct ConditionResult {
    std::string name;
    int channels = 0;
    bool random_feedback = false;
    std::vector<int> delays_train;
    int seed = 0;
    double train_fitness = 0.0;
    std::vector<EvalResult> eval;  // parallel to delays_eval
};

static std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
    return buf;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else o.push_back(c);
    }
    return o;
}

static CellularOrganism build_organism(uint32_t seed) {
    CellularOrganism org;
    org.cells.reserve(kCells);
    org.synapses.reserve(256);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> w(-1.0, 1.0);

    auto add_cell = [&](uint32_t id, CellType type, double p1 = 0.0, double p2 = 0.0) {
        Cell c;
        c.id = id;
        c.type = type;
        c.param1 = p1;
        c.param2 = p2;
        c.x = static_cast<float>(id) * 12.0f;
        org.cells.push_back(c);
    };
    auto add_syn = [&](uint32_t from, uint32_t to, double weight) {
        Synapse s;
        s.from_cell_id = from;
        s.to_cell_id = to;
        s.to_port = 0;
        s.weight = weight;
        s.initial_weight = weight;
        s.is_active = true;
        s.is_recurrent = false;
        org.synapses.push_back(s);
    };

    // 4 个外部观测受体
    add_cell(kSense0 + 0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0);
    add_cell(kSense0 + 1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0);
    add_cell(kSense0 + 2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0);
    add_cell(kSense0 + 3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0);

    // 8 个 scratchpad 受体：读 inputs[4..11]
    for (size_t i = 0; i < kScratch; ++i) {
        add_cell(static_cast<uint32_t>(kScratchSense0 + i), CellType::SENSE_CHANNEL,
                 1.0, static_cast<double>(kObs + i));
    }

    // 8 个记忆细胞：无状态 OP_SUM（内部不持留），
    // 唯一跨拍记忆来自任务层的 scratchpad 外部反馈环。
    for (size_t i = 0; i < kMem; ++i) {
        add_cell(static_cast<uint32_t>(kMem0 + i), CellType::OP_SUM, 1.0, 0.0);
    }

    // 2 个动作细胞
    add_cell(kAct0 + 0, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0);
    add_cell(kAct0 + 1, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0);

    // 观测 -> 记忆（含 bias 通道 3）
    for (size_t s = 0; s < kObs; ++s) {
        for (size_t m = 0; m < kMem; ++m) {
            add_syn(static_cast<uint32_t>(kSense0 + s), static_cast<uint32_t>(kMem0 + m), w(rng));
        }
    }
    // scratchpad -> 记忆（外部反馈写入口）
    for (size_t s = 0; s < kScratch; ++s) {
        for (size_t m = 0; m < kMem; ++m) {
            add_syn(static_cast<uint32_t>(kScratchSense0 + s), static_cast<uint32_t>(kMem0 + m), w(rng));
        }
    }
    // 注意：不建 memory->memory 同拍突触。
    // 跨拍记忆只允许走任务层 scratchpad 外部反馈，保证 no_scratchpad 是真前馈。
    // 记忆 -> 动作
    for (size_t m = 0; m < kMem; ++m) {
        add_syn(static_cast<uint32_t>(kMem0 + m), kAct0 + 0, w(rng));
        add_syn(static_cast<uint32_t>(kMem0 + m), kAct0 + 1, w(rng));
    }
    // 观测 -> 动作直连（帮助响应期）
    for (size_t s = 0; s < kObs; ++s) {
        add_syn(static_cast<uint32_t>(kSense0 + s), kAct0 + 0, w(rng) * 0.5);
        add_syn(static_cast<uint32_t>(kSense0 + s), kAct0 + 1, w(rng) * 0.5);
    }

    org.compile();
    return org;
}

static std::vector<double> export_genes(const CellularOrganism& org) {
    std::vector<double> g;
    g.reserve(org.synapses.size() + org.cells.size() * 2);
    for (const auto& s : org.synapses) if (s.is_active) g.push_back(s.weight);
    for (const auto& c : org.cells) { g.push_back(c.param1); g.push_back(c.param2); }
    return g;
}

static void apply_genes(CellularOrganism& org, const std::vector<double>& g) {
    size_t k = 0;
    for (auto& s : org.synapses) {
        if (!s.is_active) continue;
        if (k < g.size()) s.weight = g[k++];
    }
    for (auto& c : org.cells) {
        if (k < g.size()) c.param1 = g[k++];
        if (k < g.size()) c.param2 = g[k++];
    }
    org.compile();
}

// 单集：返回是否答对。
static bool episode(CellularOrganism& org, int delay, uint32_t seed,
                    int channels, bool random_feedback) {
    std::mt19937 rng(seed);
    const bool cue_left = (rng() % 2u) == 0;
    org.reset_state(false);

    double prev_mem[kMem] = {0};
    for (int t = 0; t <= delay + 1; ++t) {
        double in[kIn] = {0};
        if (t == 0) {
            in[cue_left ? 0 : 1] = 1.0;  // 线索只出现一拍
        }
        if (t == delay + 1) in[2] = 1.0;  // 响应期
        in[3] = 1.0;                      // bias

        for (int c = 0; c < static_cast<int>(kScratch); ++c) {
            if (c >= channels) { in[kObs + c] = 0.0; continue; }
            if (random_feedback) {
                std::uniform_real_distribution<double> u(-1.0, 1.0);
                in[kObs + c] = u(rng);
            } else {
                in[kObs + c] = prev_mem[c];
            }
        }

        auto acts = org.forward_nd(in, kIn, false);

        for (size_t c = 0; c < kMem; ++c) {
            prev_mem[c] = org.cells[kMem0 + c].output_val;
        }

        if (t == delay + 1) {
            const bool chose_left = acts.positive_action >= acts.negative_action;
            return chose_left == cue_left;
        }
    }
    return false;
}

static EvalResult evaluate(CellularOrganism& org, int delay, int episodes, uint32_t seed,
                           int channels, bool random_feedback) {
    EvalResult r;
    r.total = episodes;
    for (int e = 0; e < episodes; ++e) {
        const bool ok = episode(org, delay, seed + static_cast<uint32_t>(e) * 17u,
                               channels, random_feedback);
        if (ok) ++r.ok;
        r.bits.push_back(ok ? 1 : 0);
    }
    r.sr = episodes > 0 ? static_cast<double>(r.ok) / episodes : 0.0;
    return r;
}

static double mcnemar_exact(int b, int c) {
    const int n = b + c;
    if (n == 0) return 1.0;
    const int k = std::min(b, c);
    double tail = 0.0;
    for (int i = 0; i <= k; ++i) {
        tail += std::exp(std::lgamma(n + 1.0) - std::lgamma(i + 1.0) -
                         std::lgamma(n - i + 1.0) - n * std::log(2.0));
    }
    return std::min(1.0, 2.0 * tail);
}

struct ESResult {
    std::vector<double> genes;
    double fitness = 0.0;
};

static ESResult train_es(const Options& opt, uint32_t seed, int channels, bool random_feedback) {
    CellularOrganism base = build_organism(seed);
    std::vector<double> genes = export_genes(base);
    const size_t n = genes.size();
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::normal_distribution<double> mut(0.0, 0.25);

    auto make_org = [&](const std::vector<double>& g) {
        CellularOrganism org = base;
        apply_genes(org, g);
        return org;
    };

    auto fitness = [&](const std::vector<double>& g, uint32_t eval_seed) {
        CellularOrganism org = make_org(g);
        double sum = 0.0;
        for (int d : opt.delays_train) {
            sum += evaluate(org, d, opt.episodes_train, eval_seed + static_cast<uint32_t>(d) * 131u,
                            channels, random_feedback).sr;
        }
        return opt.delays_train.empty() ? 0.0 : sum / opt.delays_train.size();
    };

    std::vector<std::vector<double>> pop(static_cast<size_t>(opt.pop), genes);
    for (size_t i = 1; i < pop.size(); ++i) {
        for (size_t j = 0; j < n; ++j) pop[i][j] += mut(rng);
    }

    std::vector<double> best = genes;
    double best_fit = -1.0;
    const int elite = std::max(2, opt.pop / 6);

    for (int gen = 0; gen < opt.gens; ++gen) {
        const uint32_t eval_seed = seed + static_cast<uint32_t>(gen) * 7919u;
        std::vector<std::pair<double, size_t>> scored;
        scored.reserve(pop.size());
        for (size_t i = 0; i < pop.size(); ++i) {
            scored.emplace_back(fitness(pop[i], eval_seed), i);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (scored.front().first > best_fit) {
            best_fit = scored.front().first;
            best = pop[scored.front().second];
        }

        std::vector<std::vector<double>> next;
        next.reserve(pop.size());
        for (int i = 0; i < elite; ++i) next.push_back(pop[scored[i].second]);
        while (next.size() < pop.size()) {
            const auto& parent = pop[scored[rng() % elite].second];
            std::vector<double> child = parent;
            for (size_t j = 0; j < n; ++j) {
                if ((rng() % 100) < 70) child[j] += mut(rng);
                child[j] = std::max(-4.0, std::min(4.0, child[j]));
            }
            next.push_back(std::move(child));
        }
        pop.swap(next);
    }
    return {best, best_fit};
}

static bool has_ckpt_path() {
    return std::filesystem::exists("checkpoints") || std::filesystem::exists("../checkpoints");
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--quick") opt.quick = true;
        else if (a == "--out") opt.out = next();
        else if (a == "--gens") opt.gens = std::stoi(next());
        else if (a == "--pop") opt.pop = std::stoi(next());
        else if (a == "--seeds") opt.seeds = std::stoi(next());
        else if (a == "--delays-train") {
            std::string s = next();
            opt.delays_train.clear();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos) comma = s.size();
                opt.delays_train.push_back(std::stoi(s.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        }
        else if (a == "--eval-episodes") opt.episodes_eval = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: bench_scratchpad_generalize [--quick] [--gens N] [--pop N] "
                         "[--seeds N] [--delays-train 2,8,16,32] [--out path]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.gens = 12;
        opt.pop = 10;
        opt.seeds = 1;
        opt.episodes_train = 4;
        opt.episodes_eval = 12;
        opt.delays_train = {2, 16};
        opt.delays_eval = {2, 8, 16};
    }
    if (opt.out.empty()) opt.out = "runs/scratchpad_generalize_" + today() + ".json";
    if (!has_ckpt_path()) {
        // 任务不依赖 checkpoint；只是保证从 build/ 目录跑也能找到输出目录
    }

    std::cout << "E-B v2 delay generalization | delays_train=";
    for (size_t i = 0; i < opt.delays_train.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << opt.delays_train[i];
    }
    std::cout << " pop=" << opt.pop << " gens=" << opt.gens
              << " seeds=" << opt.seeds << " eval_eps=" << opt.episodes_eval << "\n";
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    struct Cond {
        std::string name;
        int channels;
        bool random;
        std::vector<int> delays;
    };
    std::vector<Cond> conds = {
        {"single_16_8ch", 8, false, {16}},
        {"mixed_8ch", 8, false, {2, 8, 16, 32}},
        {"mixed_4ch", 4, false, {2, 8, 16, 32}},
        {"mixed_2ch", 2, false, {2, 8, 16, 32}},
        {"mixed_1ch", 1, false, {2, 8, 16, 32}},
        {"mixed_8ch_random", 8, true, {2, 8, 16, 32}},
    };

    std::vector<ConditionResult> results;
    const uint32_t seed0 = 20261123u;
    for (const auto& c : conds) {
        for (int s = 0; s < opt.seeds; ++s) {
            const uint32_t seed = seed0 + static_cast<uint32_t>(s) * 101u;
            std::cout << "  train " << c.name << " seed=" << seed << " ... " << std::flush;
            Options co = opt;
            co.delays_train = c.delays;
            ESResult es = train_es(co, seed, c.channels, c.random);
            CellularOrganism org = build_organism(seed);
            apply_genes(org, es.genes);

            ConditionResult r;
            r.name = c.name;
            r.channels = c.channels;
            r.random_feedback = c.random;
            r.delays_train = c.delays;
            r.seed = static_cast<int>(seed);
            r.train_fitness = es.fitness;
            r.eval.reserve(opt.delays_eval.size());
            for (int d : opt.delays_eval) {
                r.eval.push_back(evaluate(org, d, opt.episodes_eval, seed ^ 0xABCDEFu,
                                          c.channels, c.random));
            }
            std::cout << "train_fit=" << es.fitness;
            for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
                std::cout << " D" << opt.delays_eval[i] << "=" << r.eval[i].sr;
            }
            std::cout << "\n";
            results.push_back(std::move(r));
        }
    }

    // 聚合：每个条件在每个 delay 上取 seed 均值
    auto mean_sr = [&](const std::string& name, size_t di) {
        double sum = 0.0;
        int n = 0;
        for (const auto& r : results) {
            if (r.name == name && di < r.eval.size()) { sum += r.eval[di].sr; ++n; }
        }
        return n ? sum / n : 0.0;
    };
    auto min_sr = [&](const std::string& name) {
        double m = 1.0;
        for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
            m = std::min(m, mean_sr(name, i));
        }
        return m;
    };

    const double single_min = min_sr("single_16_8ch");
    const double mixed_min = min_sr("mixed_8ch");
    const double random_min = min_sr("mixed_8ch_random");
    const double delta_min = mixed_min - single_min;
    const bool coverage_ok = mixed_min >= 0.80;
    const bool delta_ok = delta_min >= 0.20;
    const bool random_ok = random_min <= 0.60;
    const bool gate = coverage_ok && delta_ok && random_ok;

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"E-B v2 delay generalization\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"delays_train\":[";
    for (size_t i = 0; i < opt.delays_train.size(); ++i) {
        if (i) os << ",";
        os << opt.delays_train[i];
    }
    os << "],\"pop\":" << opt.pop << ",\"gens\":" << opt.gens
       << ",\"seeds\":" << opt.seeds << ",\"episodes_train\":" << opt.episodes_train
       << ",\"episodes_eval\":" << opt.episodes_eval << ",\"delays_eval\":[";
    for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
        if (i) os << ",";
        os << opt.delays_eval[i];
    }
    os << "]},\n  \"conditions\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) os << ",";
        os << "\n    {\"name\":\"" << json_escape(r.name)
           << "\",\"channels\":" << r.channels
           << ",\"random_feedback\":" << (r.random_feedback ? "true" : "false")
           << ",\"delays_train\":[";
        for (size_t k = 0; k < r.delays_train.size(); ++k) {
            if (k) os << ",";
            os << r.delays_train[k];
        }
        os << "],\"seed\":" << r.seed
           << ",\"train_fitness\":" << r.train_fitness
           << ",\"eval\":[";
        for (size_t j = 0; j < r.eval.size(); ++j) {
            if (j) os << ",";
            os << "{\"delay\":" << opt.delays_eval[j]
               << ",\"sr\":" << r.eval[j].sr
               << ",\"ok\":" << r.eval[j].ok
               << ",\"total\":" << r.eval[j].total
               << ",\"bits\":[";
            for (size_t k = 0; k < r.eval[j].bits.size(); ++k) {
                if (k) os << ",";
                os << r.eval[j].bits[k];
            }
            os << "]}";
        }
        os << "]}";
    }
    os << "\n  ],\n  \"aggregate\":{\"single_16_8ch_min\":" << single_min
       << ",\"mixed_8ch_min\":" << mixed_min
       << ",\"mixed_8ch_random_min\":" << random_min
       << ",\"delta_min\":" << delta_min
       << "},\n  \"gate\":{\"H2_v2\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"coverage_ok\":" << (coverage_ok ? "true" : "false")
       << ",\"delta_ok\":" << (delta_ok ? "true" : "false")
       << ",\"random_ok\":" << (random_ok ? "true" : "false") << "}\n}\n";
    os.close();

    std::cout << "\n== aggregate ==\n";
    for (const auto& c : conds) {
        std::cout << "  " << c.name << ":";
        for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
            std::cout << "  D" << opt.delays_eval[i] << "=" << mean_sr(c.name, i);
        }
        std::cout << "  min=" << min_sr(c.name) << "\n";
    }
    std::cout << "GATE: H2_v2 " << (gate ? "PASS" : "FAIL")
              << " (single_min=" << single_min
              << " mixed_min=" << mixed_min
              << " random_min=" << random_min
              << " delta_min=" << delta_min << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}

// ============================================================================
// bench_scratchpad_cot.cpp — E-B 串行 scratchpad 因果验证
//
// 任务：Key-Door Memory（延迟线索 + 二选一门）
//   t=0        线索出现（左/右），只此一拍
//   t=1..D     延迟（无线索），模拟走向门
//   t=D+1      响应期，必须凭记忆选左/右门
//
// 为什么这个任务干净：
//   前馈策略在 t=D+1 看不到线索，数学上无法可靠作答；
//   只有把线索写入跨拍状态（循环/scratchpad），才能在延迟后读出。
//
// 条件（同一拓扑、同一参数量，只改反馈内容）：
//   channels=0        无 scratchpad（反馈全 0）——消融基线
//   channels=1/2/4/8  真实反馈（上一拍记忆细胞输出）
//   channels=8 random 随机反馈（同分布、无信息）——反伪影对照
//
// 训练：固定拓扑 + (1+λ) ES 只调权重/参数；每个条件同预算、同 seed。
// 输出：runs/scratchpad_cot_<date>.json
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
    int delay_train = 16;
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
        return evaluate(org, opt.delay_train, opt.episodes_train, eval_seed,
                        channels, random_feedback).sr;
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
        else if (a == "--delay") opt.delay_train = std::stoi(next());
        else if (a == "--eval-episodes") opt.episodes_eval = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: bench_scratchpad_cot [--quick] [--gens N] [--pop N] "
                         "[--seeds N] [--delay N] [--out path]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.gens = 12;
        opt.pop = 10;
        opt.seeds = 1;
        opt.episodes_train = 6;
        opt.episodes_eval = 12;
        opt.delays_eval = {2, 8, 16};
    }
    if (opt.out.empty()) opt.out = "runs/scratchpad_cot_" + today() + ".json";
    if (!has_ckpt_path()) {
        // 任务不依赖 checkpoint；只是保证从 build/ 目录跑也能找到输出目录
    }

    std::cout << "E-B scratchpad CoT | delay_train=" << opt.delay_train
              << " pop=" << opt.pop << " gens=" << opt.gens
              << " seeds=" << opt.seeds << " eval_eps=" << opt.episodes_eval << "\n";
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    struct Cond { std::string name; int channels; bool random; };
    std::vector<Cond> conds = {
        {"no_scratchpad", 0, false},
        {"scratch_1ch", 1, false},
        {"scratch_2ch", 2, false},
        {"scratch_4ch", 4, false},
        {"scratch_8ch", 8, false},
        {"scratch_8ch_random", 8, true},
    };

    std::vector<ConditionResult> results;
    const uint32_t seed0 = 20261123u;
    for (const auto& c : conds) {
        for (int s = 0; s < opt.seeds; ++s) {
            const uint32_t seed = seed0 + static_cast<uint32_t>(s) * 101u;
            std::cout << "  train " << c.name << " seed=" << seed << " ... " << std::flush;
            ESResult es = train_es(opt, seed, c.channels, c.random);
            CellularOrganism org = build_organism(seed);
            apply_genes(org, es.genes);

            ConditionResult r;
            r.name = c.name;
            r.channels = c.channels;
            r.random_feedback = c.random;
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

    size_t hard_di = 0;
    for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
        if (opt.delays_eval[i] == 16) hard_di = i;
    }
    const double no_scratch = mean_sr("no_scratchpad", hard_di);
    const double full_scratch = mean_sr("scratch_8ch", hard_di);
    const double random_scratch = mean_sr("scratch_8ch_random", hard_di);
    const double delta = full_scratch - no_scratch;
    int b = 0, c = 0;
    for (const auto& rn : results) {
        if (rn.name != "no_scratchpad" || hard_di >= rn.eval.size()) continue;
        for (const auto& rf : results) {
            if (rf.name != "scratch_8ch" || rf.seed != rn.seed || hard_di >= rf.eval.size()) continue;
            const auto& bn = rn.eval[hard_di].bits;
            const auto& bf = rf.eval[hard_di].bits;
            for (size_t i = 0; i < bn.size() && i < bf.size(); ++i) {
                if (bn[i] && !bf[i]) ++b;
                if (!bn[i] && bf[i]) ++c;
            }
        }
    }
    const double p_mcnemar = mcnemar_exact(b, c);
    const bool ablation_ok = delta >= 0.15 && p_mcnemar < 0.05;
    const bool random_ok = (full_scratch - random_scratch) >= 0.15;
    const bool gate = ablation_ok && random_ok;

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"E-B serial scratchpad\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"delay_train\":" << opt.delay_train
       << ",\"pop\":" << opt.pop << ",\"gens\":" << opt.gens
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
           << ",\"seed\":" << r.seed
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
    os << "\n  ],\n  \"aggregate\":{\"no_scratchpad\":" << no_scratch
       << ",\"scratch_8ch\":" << full_scratch
       << ",\"scratch_8ch_random\":" << random_scratch
       << ",\"delta_full_vs_none\":" << delta
       << ",\"hard_delay\":" << opt.delays_eval[hard_di]
       << ",\"mcnemar_b\":" << b
       << ",\"mcnemar_c\":" << c
       << ",\"p_mcnemar\":" << p_mcnemar
       << "},\n  \"gate\":{\"H2\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"ablation_ok\":" << (ablation_ok ? "true" : "false")
       << ",\"random_ok\":" << (random_ok ? "true" : "false") << "}\n}\n";
    os.close();

    std::cout << "\n== aggregate @ D=" << opt.delays_eval[hard_di] << " ==\n";
    for (const auto& c : conds) {
        std::cout << "  " << c.name << ":";
        for (size_t i = 0; i < opt.delays_eval.size(); ++i) {
            std::cout << "  D" << opt.delays_eval[i] << "=" << mean_sr(c.name, i);
        }
        std::cout << "\n";
    }
    std::cout << "GATE: H2 " << (gate ? "PASS" : "FAIL")
              << " (no_scratch=" << no_scratch
              << " full=" << full_scratch
              << " random=" << random_scratch
              << " delta=" << delta
              << " b=" << b << " c=" << c << " p=" << p_mcnemar << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}

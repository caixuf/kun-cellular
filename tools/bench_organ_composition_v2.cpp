// ============================================================================
// bench_organ_composition_v2.cpp — E-C v2：4-way cue 冻结器官组合
//
// 为什么有 v2：
//   v1 是二选一，single_event 靠“有 pulse 就猜边”天然拿 50%，
//   overall best single = 75.6%，full 已 100%，+25pp 数学上不可达。
//   v2 改成 4-way cue（左/右/上/下），chance=25%，+25pp 可达。
//
// 预注册（先于数据）：
//   H3-v2 PASS 当且仅当
//     1) full - best_single >= 25pp
//     2) full - min_loo >= 20pp
//     3) full - monolith >= 10pp
//     4) 5 seeds × 200 held-out episodes；响应幅度门禁
//
// 任务：Conditional 4-Way Delayed Response
//   t=0        cue ∈ {0,1,2,3}，one-hot，只出现一拍
//   t=2..6     随机一拍 pulse（50% 概率完全没有 pulse）
//   t=10       有 pulse：选与 cue 同 index 的动作，且 max action >= 0.2
//              无 pulse：no-go（max action < 1e-6）
//
// 器官（内部参数冻结）：
//   delay   : 4 × OP_EMA（每 cue 一个）
//   event   : 1 × OP_INTEGRAL（锁存 pulse）
//   gate    : 4 × OP_MULTIPLY（delay_i × event）
//   readout : 4 × ACT_CHANNEL（param2=0..3）
//
// 对照：full / loo_delay / loo_event / loo_gate /
//       single_delay / single_event / single_gate / monolith
//
// 输出：runs/organ_composition_v2_<date>.json
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
#include <string>
#include <vector>

using namespace kun;

namespace {

constexpr size_t kCues = 4;
constexpr size_t kSense0 = 0;
constexpr size_t kPulseSense = 4;
constexpr size_t kDelay0 = 5;
constexpr size_t kEvent = 9;
constexpr size_t kGate0 = 10;
constexpr size_t kAct0 = 14;
constexpr size_t kCells = 18;
constexpr int kResponseTick = 10;

struct Options {
    int pop = 24;
    int gens = 120;
    int episodes_train = 24;
    int episodes_eval = 200;
    int seeds = 5;
    std::string out;
    bool quick = false;
};

struct EvalResult {
    double sr = 0.0;
    int ok = 0;
    int total = 0;
    int pulse_ok = 0;
    int pulse_total = 0;
    int nopulse_ok = 0;
    int nopulse_total = 0;
    std::vector<int> bits;
};

struct ConditionResult {
    std::string name;
    int seed = 0;
    double train_fitness = 0.0;
    EvalResult eval;
};

enum class Variant {
    Full, LooDelay, LooEvent, LooGate,
    SingleDelay, SingleEvent, SingleGate, Monolith,
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

static std::string variant_name(Variant v) {
    switch (v) {
        case Variant::Full: return "full";
        case Variant::LooDelay: return "loo_delay";
        case Variant::LooEvent: return "loo_event";
        case Variant::LooGate: return "loo_gate";
        case Variant::SingleDelay: return "single_delay";
        case Variant::SingleEvent: return "single_event";
        case Variant::SingleGate: return "single_gate";
        case Variant::Monolith: return "monolith";
    }
    return "?";
}

static void add_cell(CellularOrganism& org, uint32_t id, CellType type,
                     double p1 = 0.0, double p2 = 0.0) {
    Cell c;
    c.id = id;
    c.type = type;
    c.param1 = p1;
    c.param2 = p2;
    c.x = static_cast<float>(id) * 12.0f;
    org.cells.push_back(c);
}

static void add_syn(CellularOrganism& org, uint32_t from, uint32_t to,
                    uint8_t port, double weight) {
    Synapse s;
    s.from_cell_id = from;
    s.to_cell_id = to;
    s.to_port = port;
    s.weight = weight;
    s.initial_weight = weight;
    s.is_active = true;
    s.is_recurrent = false;
    org.synapses.push_back(s);
}

static CellularOrganism build_organism(Variant v, uint32_t seed) {
    CellularOrganism org;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> w(-1.0, 1.0);

    // 4 个 cue 受体 + 1 个 pulse 受体
    add_cell(org, kSense0 + 0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0);
    add_cell(org, kSense0 + 1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0);
    add_cell(org, kSense0 + 2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0);
    add_cell(org, kSense0 + 3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0);
    add_cell(org, kPulseSense, CellType::SENSE_CHANNEL, 1.0, 4.0);

    // delay 器官：4 个 EMA，param1=0.05 => 10 拍后约 0.6
    for (size_t i = 0; i < kCues; ++i) {
        add_cell(org, static_cast<uint32_t>(kDelay0 + i), CellType::OP_EMA, 0.05, 0.0);
    }
    // event 器官
    add_cell(org, kEvent, CellType::OP_INTEGRAL, 1.0, 0.0);
    // gate 器官：4 个乘法
    for (size_t i = 0; i < kCues; ++i) {
        add_cell(org, static_cast<uint32_t>(kGate0 + i), CellType::OP_MULTIPLY, 1.0, 0.0);
    }
    // 4 个 ACT_CHANNEL 动作细胞
    for (size_t i = 0; i < kCues; ++i) {
        add_cell(org, static_cast<uint32_t>(kAct0 + i), CellType::ACT_CHANNEL,
                 1.0, static_cast<double>(i));
    }

    if (v == Variant::Monolith) {
        // 同细胞数/同突触数，打乱 op 类型 + 随机连接；训练全部权重。
        std::vector<CellType> ops = {
            CellType::OP_EMA, CellType::OP_SUM, CellType::OP_SUB,
            CellType::OP_MULTIPLY, CellType::GATE_THRESHOLD,
            CellType::GATE_AND, CellType::OP_INTEGRAL,
        };
        for (size_t i = kDelay0; i < kAct0; ++i) {
            org.cells[i].type = ops[rng() % ops.size()];
            org.cells[i].param1 = w(rng);
        }
        for (size_t s = 0; s < 5; ++s) {
            for (size_t i = kDelay0; i < kAct0; ++i) {
                add_syn(org, static_cast<uint32_t>(kSense0 + s), static_cast<uint32_t>(i), 0, w(rng));
            }
        }
        for (size_t i = kDelay0; i < kGate0; ++i) {
            for (size_t j = kGate0; j < kAct0; ++j) {
                add_syn(org, static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0, w(rng));
            }
        }
        for (size_t i = kGate0; i < kAct0; ++i) {
            add_syn(org, static_cast<uint32_t>(i), static_cast<uint32_t>(kAct0 + (i - kGate0)), 0, w(rng));
        }
        org.compile();
        return org;
    }

    // cue_i -> delay_i
    for (size_t i = 0; i < kCues; ++i) {
        add_syn(org, static_cast<uint32_t>(kSense0 + i), static_cast<uint32_t>(kDelay0 + i), 0, 1.0);
    }
    // pulse -> event
    add_syn(org, kPulseSense, kEvent, 0, 1.0);
    // delay_i + event -> gate_i
    for (size_t i = 0; i < kCues; ++i) {
        add_syn(org, static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kGate0 + i), 0, 1.0);
        add_syn(org, kEvent, static_cast<uint32_t>(kGate0 + i), 1, 1.0);
        add_syn(org, static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0, 1.0);
    }

    auto set_weight = [&](uint32_t from, uint32_t to, uint8_t port, double weight) {
        for (auto& s : org.synapses) {
            if (s.from_cell_id == from && s.to_cell_id == to && s.to_port == port) {
                s.weight = weight;
                s.initial_weight = weight;
                return;
            }
        }
    };

    switch (v) {
        case Variant::Full:
            break;
        case Variant::LooDelay:
            for (size_t i = 0; i < kCues; ++i)
                set_weight(static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kGate0 + i), 0, 0.0);
            break;
        case Variant::LooEvent:
            for (size_t i = 0; i < kCues; ++i)
                set_weight(kEvent, static_cast<uint32_t>(kGate0 + i), 1, 0.0);
            break;
        case Variant::LooGate:
            for (size_t i = 0; i < kCues; ++i)
                set_weight(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0, 0.0);
            break;
        case Variant::SingleDelay:
            for (size_t i = 0; i < kCues; ++i) {
                set_weight(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0, 0.0);
                add_syn(org, static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kAct0 + i), 0, 1.0);
            }
            break;
        case Variant::SingleEvent:
            for (size_t i = 0; i < kCues; ++i) {
                set_weight(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0, 0.0);
                add_syn(org, kEvent, static_cast<uint32_t>(kAct0 + i), 0, 1.0);
            }
            break;
        case Variant::SingleGate:
            for (size_t i = 0; i < kCues; ++i) {
                set_weight(static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kGate0 + i), 0, 0.0);
                set_weight(kEvent, static_cast<uint32_t>(kGate0 + i), 1, 0.0);
            }
            break;
        default:
            break;
    }

    org.compile();
    return org;
}

static void enforce_variant(CellularOrganism& org, Variant v) {
    auto z = [&](uint32_t from, uint32_t to, uint8_t port) {
        for (auto& s : org.synapses) {
            if (s.from_cell_id == from && s.to_cell_id == to && s.to_port == port) s.weight = 0.0;
        }
    };
    switch (v) {
        case Variant::LooDelay:
            for (size_t i = 0; i < kCues; ++i)
                z(static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kGate0 + i), 0);
            break;
        case Variant::LooEvent:
            for (size_t i = 0; i < kCues; ++i)
                z(kEvent, static_cast<uint32_t>(kGate0 + i), 1);
            break;
        case Variant::LooGate:
            for (size_t i = 0; i < kCues; ++i)
                z(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0);
            break;
        case Variant::SingleDelay:
            for (size_t i = 0; i < kCues; ++i)
                z(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0);
            break;
        case Variant::SingleEvent:
            for (size_t i = 0; i < kCues; ++i)
                z(static_cast<uint32_t>(kGate0 + i), static_cast<uint32_t>(kAct0 + i), 0);
            break;
        case Variant::SingleGate:
            for (size_t i = 0; i < kCues; ++i) {
                z(static_cast<uint32_t>(kDelay0 + i), static_cast<uint32_t>(kGate0 + i), 0);
                z(kEvent, static_cast<uint32_t>(kGate0 + i), 1);
            }
            break;
        default:
            break;
    }
    org.compile();
}

static std::vector<double> export_genes(const CellularOrganism& org, Variant v) {
    std::vector<double> g;
    if (v == Variant::Monolith) {
        for (const auto& s : org.synapses) if (s.is_active) g.push_back(s.weight);
        for (const auto& c : org.cells) { g.push_back(c.param1); g.push_back(c.param2); }
        return g;
    }
    // 只训练器官间/读出连接；器官内部参数冻结。
    for (const auto& s : org.synapses) {
        if (!s.is_active) continue;
        const bool trainable =
            (s.from_cell_id >= kDelay0 && s.from_cell_id < kEvent &&
             s.to_cell_id >= kGate0 && s.to_cell_id < kAct0) ||
            (s.from_cell_id == kEvent && s.to_cell_id >= kGate0 && s.to_cell_id < kAct0) ||
            (s.to_cell_id >= kAct0 && s.to_cell_id < kCells);
        if (trainable) g.push_back(s.weight);
    }
    return g;
}

static void apply_genes(CellularOrganism& org, Variant v, const std::vector<double>& g) {
    size_t k = 0;
    if (v == Variant::Monolith) {
        for (auto& s : org.synapses) if (s.is_active && k < g.size()) s.weight = g[k++];
        for (auto& c : org.cells) {
            if (k < g.size()) c.param1 = g[k++];
            if (k < g.size()) c.param2 = g[k++];
        }
    } else {
        for (auto& s : org.synapses) {
            if (!s.is_active) continue;
            const bool trainable =
                (s.from_cell_id >= kDelay0 && s.from_cell_id < kEvent &&
                 s.to_cell_id >= kGate0 && s.to_cell_id < kAct0) ||
                (s.from_cell_id == kEvent && s.to_cell_id >= kGate0 && s.to_cell_id < kAct0) ||
                (s.to_cell_id >= kAct0 && s.to_cell_id < kCells);
            if (trainable && k < g.size()) s.weight = g[k++];
        }
    }
    org.compile();
}

static bool episode(CellularOrganism& org, bool has_pulse, int cue, int pulse_tick) {
    org.reset_state(false);
    for (int t = 0; t <= kResponseTick; ++t) {
        double in[5] = {0};
        if (t == 0) in[cue] = 1.0;
        if (has_pulse && t == pulse_tick) in[4] = 1.0;
        org.forward_nd(in, 5, false);
        if (t == kResponseTick) {
            double maxv = -1e9;
            int arg = 0;
            for (size_t i = 0; i < kCues; ++i) {
                const double v = org.cells[kAct0 + i].output_val;
                if (v > maxv) { maxv = v; arg = static_cast<int>(i); }
            }
            if (!has_pulse) return maxv < 1e-6;
            return maxv >= 0.2 && arg == cue;
        }
    }
    return false;
}

static EvalResult evaluate(CellularOrganism& org, int episodes, uint32_t seed) {
    EvalResult r;
    r.total = episodes;
    std::mt19937 rng(seed);
    for (int e = 0; e < episodes; ++e) {
        const bool has_pulse = (rng() % 100) < 50;
        const int cue = static_cast<int>(rng() % kCues);
        const int pulse_tick = 2 + static_cast<int>(rng() % 5);
        const bool ok = episode(org, has_pulse, cue, pulse_tick);
        if (ok) ++r.ok;
        if (has_pulse) { ++r.pulse_total; if (ok) ++r.pulse_ok; }
        else { ++r.nopulse_total; if (ok) ++r.nopulse_ok; }
        r.bits.push_back(ok ? 1 : 0);
    }
    r.sr = episodes > 0 ? static_cast<double>(r.ok) / episodes : 0.0;
    return r;
}

struct ESResult {
    std::vector<double> genes;
    double fitness = 0.0;
};

static ESResult train_es(const Options& opt, Variant v, uint32_t seed) {
    CellularOrganism base = build_organism(v, seed);
    std::vector<double> genes = export_genes(base, v);
    const size_t n = genes.size();
    std::mt19937 rng(seed ^ 0x5bd1e995u);
    std::normal_distribution<double> mut(0.0, 0.3);

    auto make_org = [&](const std::vector<double>& g) {
        CellularOrganism org = base;
        apply_genes(org, v, g);
        enforce_variant(org, v);
        return org;
    };

    auto fitness = [&](const std::vector<double>& g, uint32_t eval_seed) {
        CellularOrganism org = make_org(g);
        return evaluate(org, opt.episodes_train, eval_seed).sr;
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
        else if (a == "--eval-episodes") opt.episodes_eval = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: bench_organ_composition_v2 [--quick] [--gens N] [--pop N] "
                         "[--seeds N] [--out path]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.gens = 20;
        opt.pop = 12;
        opt.seeds = 1;
        opt.episodes_train = 8;
        opt.episodes_eval = 40;
    }
    if (opt.out.empty()) opt.out = "runs/organ_composition_v2_" + today() + ".json";
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    std::vector<Variant> variants = {
        Variant::Full, Variant::LooDelay, Variant::LooEvent, Variant::LooGate,
        Variant::SingleDelay, Variant::SingleEvent, Variant::SingleGate, Variant::Monolith,
    };

    std::cout << "E-C v2 (4-way) | pop=" << opt.pop << " gens=" << opt.gens
              << " seeds=" << opt.seeds << " eval_eps=" << opt.episodes_eval << "\n";

    std::vector<ConditionResult> results;
    const uint32_t seed0 = 20261123u;
    for (Variant v : variants) {
        for (int s = 0; s < opt.seeds; ++s) {
            const uint32_t seed = seed0 + static_cast<uint32_t>(s) * 101u;
            std::cout << "  train " << variant_name(v) << " seed=" << seed << " ... " << std::flush;
            ESResult es = train_es(opt, v, seed);
            CellularOrganism org = build_organism(v, seed);
            apply_genes(org, v, es.genes);
            enforce_variant(org, v);
            EvalResult ev = evaluate(org, opt.episodes_eval, seed ^ 0xABCDEFu);
            std::cout << "train_fit=" << es.fitness << " eval=" << ev.sr
                      << " (pulse " << ev.pulse_ok << "/" << ev.pulse_total
                      << ", nopulse " << ev.nopulse_ok << "/" << ev.nopulse_total << ")\n";
            results.push_back({variant_name(v), static_cast<int>(seed), es.fitness, ev});
        }
    }

    auto mean_sr = [&](const std::string& name) {
        double sum = 0.0;
        int n = 0;
        for (const auto& r : results) if (r.name == name) { sum += r.eval.sr; ++n; }
        return n ? sum / n : 0.0;
    };

    const double full = mean_sr("full");
    const double monolith = mean_sr("monolith");
    double best_single = 0.0;
    std::string best_single_name = "single_gate";
    for (const char* n : {"single_delay", "single_event", "single_gate"}) {
        const double v = mean_sr(n);
        if (v > best_single) { best_single = v; best_single_name = n; }
    }
    const double loo_delay = mean_sr("loo_delay");
    const double loo_event = mean_sr("loo_event");
    const double loo_gate = mean_sr("loo_gate");
    const double min_loo = std::min({loo_delay, loo_event, loo_gate});

    const bool super_add = (full - best_single) >= 0.25;
    const bool loo_ok = (full - min_loo) >= 0.20;
    const bool mono_ok = (full - monolith) >= 0.10;
    const bool gate = super_add && loo_ok && mono_ok;

    auto mean_pulse = [&](const std::string& name) {
        int ok = 0, total = 0;
        for (const auto& r : results) {
            if (r.name == name) { ok += r.eval.pulse_ok; total += r.eval.pulse_total; }
        }
        return total ? static_cast<double>(ok) / total : 0.0;
    };
    const double full_pulse = mean_pulse("full");
    const double best_single_pulse = mean_pulse(best_single_name);

    int b = 0, c = 0;
    for (const auto& rf : results) {
        if (rf.name != "full") continue;
        for (const auto& rm : results) {
            if (rm.name != "monolith" || rm.seed != rf.seed) continue;
            for (size_t i = 0; i < rf.eval.bits.size() && i < rm.eval.bits.size(); ++i) {
                if (rf.eval.bits[i] && !rm.eval.bits[i]) ++b;
                if (!rf.eval.bits[i] && rm.eval.bits[i]) ++c;
            }
        }
    }
    const double p_mono = mcnemar_exact(b, c);
    int b2 = 0, c2 = 0;
    for (const auto& rf : results) {
        if (rf.name != "full") continue;
        for (const auto& rs : results) {
            if (rs.name != best_single_name || rs.seed != rf.seed) continue;
            for (size_t i = 0; i < rf.eval.bits.size() && i < rs.eval.bits.size(); ++i) {
                if (rf.eval.bits[i] && !rs.eval.bits[i]) ++b2;
                if (!rf.eval.bits[i] && rs.eval.bits[i]) ++c2;
            }
        }
    }
    const double p_best = mcnemar_exact(b2, c2);

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"E-C v2 organ composition (4-way cue)\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"pop\":" << opt.pop << ",\"gens\":" << opt.gens
       << ",\"seeds\":" << opt.seeds << ",\"episodes_train\":" << opt.episodes_train
       << ",\"episodes_eval\":" << opt.episodes_eval << ",\"cues\":4},\n";
    os << "  \"conditions\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) os << ",";
        os << "\n    {\"name\":\"" << r.name << "\",\"seed\":" << r.seed
           << ",\"train_fitness\":" << r.train_fitness
           << ",\"sr\":" << r.eval.sr << ",\"ok\":" << r.eval.ok
           << ",\"total\":" << r.eval.total
           << ",\"pulse_ok\":" << r.eval.pulse_ok
           << ",\"pulse_total\":" << r.eval.pulse_total
           << ",\"nopulse_ok\":" << r.eval.nopulse_ok
           << ",\"nopulse_total\":" << r.eval.nopulse_total
           << ",\"bits\":[";
        for (size_t j = 0; j < r.eval.bits.size(); ++j) {
            if (j) os << ",";
            os << r.eval.bits[j];
        }
        os << "]}";
    }
    os << "\n  ],\n  \"aggregate\":{\"full\":" << full
       << ",\"best_single\":" << best_single
       << ",\"best_single_name\":\"" << best_single_name << "\""
       << ",\"min_loo\":" << min_loo
       << ",\"monolith\":" << monolith
       << ",\"full_vs_best_single\":" << (full - best_single)
       << ",\"full_vs_min_loo\":" << (full - min_loo)
       << ",\"full_vs_monolith\":" << (full - monolith)
       << ",\"full_pulse\":" << full_pulse
       << ",\"best_single_pulse\":" << best_single_pulse
       << ",\"full_vs_best_single_pulse\":" << (full_pulse - best_single_pulse)
       << ",\"p_full_vs_monolith\":" << std::scientific << p_mono << std::fixed
       << ",\"p_full_vs_best_single\":" << std::scientific << p_best << std::fixed
       << "},\n  \"gate\":{\"H3_v2\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"super_additive\":" << (super_add ? "true" : "false")
       << ",\"loo_ok\":" << (loo_ok ? "true" : "false")
       << ",\"monolith_ok\":" << (mono_ok ? "true" : "false") << "}\n}\n";
    os.close();

    std::cout << "\n== aggregate ==\n";
    for (Variant v : variants) {
        std::cout << "  " << std::setw(16) << variant_name(v) << " SR=" << mean_sr(variant_name(v)) << "\n";
    }
    std::cout << "GATE: H3_v2 " << (gate ? "PASS" : "FAIL")
              << " (full=" << full << " best_single=" << best_single
              << " [" << best_single_name << "] min_loo=" << min_loo
              << " monolith=" << monolith
              << " full_pulse=" << full_pulse
              << " best_single_pulse=" << best_single_pulse
              << " p_best=" << p_best << " p_mono=" << p_mono << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}

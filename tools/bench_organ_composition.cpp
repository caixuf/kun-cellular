// ============================================================================
// bench_organ_composition.cpp — E-C 冻结器官组合超加性验证
//
// 任务：Conditional Delayed Response（条件延迟响应）
//   t=0        线索 cue ∈ {left,right}，只出现一拍
//   t=2..6     随机一拍出现 pulse（50% 概率完全没有 pulse）
//   t=10       响应期：
//                - 有 pulse：选与 cue 同侧的门
//                - 无 pulse：no-go（两侧都不动）
//
// 四个冻结器官（内部参数固定，不参与训练）：
//   delay    : 2 × OP_EMA 保持 cue
//   event    : 1 × OP_ACCUMULATOR 锁存 pulse
//   compare  : 1 × OP_SUB 计算 left-right
//   gate     : 1 × OP_MULTIPLY 用 event 门控 compare
//
// 只训练器官之间的连接/读出权重（composition），或同参数单体的全部权重。
//
// 对照：
//   full            四器官全连
//   loo_delay       delay 输出置 0
//   loo_event       event 输出置 0
//   loo_compare     compare 输出置 0
//   loo_gate        gate 输出置 0
//   single_delay    只把 delay 接到读出
//   single_event    只把 event 接到读出
//   single_compare  只把 compare 接到读出
//   single_gate     只把 gate 接到读出（注意：gate 需要上游）
//   monolith        同细胞数/同突触数，但打乱 op 类型，训练全部权重
//
// 输出：runs/organ_composition_<date>.json
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

constexpr size_t kSense0 = 0;
constexpr size_t kDLeft = 4;
constexpr size_t kDRight = 5;
constexpr size_t kEvent = 6;
constexpr size_t kCompare = 7;
constexpr size_t kGate = 8;
constexpr size_t kActPos = 9;
constexpr size_t kActNeg = 10;
constexpr size_t kCells = 11;
constexpr int kResponseTick = 10;

struct Options {
    int pop = 24;
    int gens = 120;
    int episodes_train = 24;
    int episodes_eval = 200;
    int seeds = 5;
    std::string out;
    bool quick = false;
    bool debug = false;
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
    Full,
    LooDelay,
    LooEvent,
    LooCompare,
    LooGate,
    SingleDelay,
    SingleEvent,
    SingleCompare,
    SingleGate,
    Monolith,
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

    // 输入受体：cue_left, cue_right, pulse, response_phase
    add_cell(org, kSense0 + 0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0);
    add_cell(org, kSense0 + 1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0);
    add_cell(org, kSense0 + 2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0);
    add_cell(org, kSense0 + 3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0);

    // 器官细胞
    add_cell(org, kDLeft, CellType::OP_EMA, 0.05, 0.0);
    add_cell(org, kDRight, CellType::OP_EMA, 0.05, 0.0);
    add_cell(org, kEvent, CellType::OP_INTEGRAL, 1.0, 0.0);
    add_cell(org, kCompare, CellType::OP_SUB, 1.0, 0.0);
    add_cell(org, kGate, CellType::OP_MULTIPLY, 1.0, 0.0);
    add_cell(org, kActPos, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0);
    add_cell(org, kActNeg, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0);

    if (v == Variant::Monolith) {
        // 同细胞数/同突触数，但打乱 op 类型 + 随机连接；训练全部权重。
        std::vector<CellType> ops = {
            CellType::OP_EMA, CellType::OP_SUM, CellType::OP_SUB,
            CellType::OP_MULTIPLY, CellType::GATE_THRESHOLD,
            CellType::GATE_AND, CellType::OP_INTEGRAL,
        };
        for (size_t i = kDLeft; i <= kGate; ++i) {
            org.cells[i].type = ops[rng() % ops.size()];
            org.cells[i].param1 = w(rng);
        }
        // 随机连接：输入 + 内部 + 动作
        for (size_t s = 0; s < 4; ++s) {
            add_syn(org, static_cast<uint32_t>(kSense0 + s), kDLeft, 0, w(rng));
            add_syn(org, static_cast<uint32_t>(kSense0 + s), kDRight, 0, w(rng));
            add_syn(org, static_cast<uint32_t>(kSense0 + s), kEvent, 0, w(rng));
        }
        for (size_t i = kDLeft; i <= kGate; ++i) {
            for (size_t j = kDLeft; j <= kGate; ++j) {
                if (i == j) continue;
                add_syn(org, static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0, w(rng));
            }
        }
        for (size_t i = kDLeft; i <= kGate; ++i) {
            add_syn(org, static_cast<uint32_t>(i), kActPos, 0, w(rng));
            add_syn(org, static_cast<uint32_t>(i), kActNeg, 0, w(rng));
        }
        org.compile();
        return org;
    }

    // 固定器官内部连接
    add_syn(org, kSense0 + 0, kDLeft, 0, 1.0);
    add_syn(org, kSense0 + 1, kDRight, 0, 1.0);
    add_syn(org, kSense0 + 2, kEvent, 0, 1.0);

    // delay -> compare
    add_syn(org, kDLeft, kCompare, 0, 1.0);
    add_syn(org, kDRight, kCompare, 1, 1.0);
    // compare + event -> gate
    add_syn(org, kCompare, kGate, 0, 1.0);
    add_syn(org, kEvent, kGate, 1, 1.0);

    // 默认读出：gate -> 动作
    add_syn(org, kGate, kActPos, 0, 1.0);
    add_syn(org, kGate, kActNeg, 0, -1.0);

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
            set_weight(kDLeft, kCompare, 0, 0.0);
            set_weight(kDRight, kCompare, 1, 0.0);
            break;
        case Variant::LooEvent:
            set_weight(kEvent, kGate, 1, 0.0);
            break;
        case Variant::LooCompare:
            set_weight(kCompare, kGate, 0, 0.0);
            break;
        case Variant::LooGate:
            set_weight(kGate, kActPos, 0, 0.0);
            set_weight(kGate, kActNeg, 0, 0.0);
            break;
        case Variant::SingleDelay:
            set_weight(kGate, kActPos, 0, 0.0);
            set_weight(kGate, kActNeg, 0, 0.0);
            add_syn(org, kDLeft, kActPos, 0, 1.0);
            add_syn(org, kDRight, kActNeg, 0, 1.0);
            break;
        case Variant::SingleEvent:
            set_weight(kGate, kActPos, 0, 0.0);
            set_weight(kGate, kActNeg, 0, 0.0);
            add_syn(org, kEvent, kActPos, 0, 1.0);
            add_syn(org, kEvent, kActNeg, 0, 1.0);
            break;
        case Variant::SingleCompare:
            set_weight(kGate, kActPos, 0, 0.0);
            set_weight(kGate, kActNeg, 0, 0.0);
            set_weight(kDLeft, kCompare, 0, 0.0);
            set_weight(kDRight, kCompare, 1, 0.0);
            add_syn(org, kCompare, kActPos, 0, 1.0);
            add_syn(org, kCompare, kActNeg, 0, -1.0);
            break;
        case Variant::SingleGate:
            set_weight(kCompare, kGate, 0, 0.0);
            set_weight(kEvent, kGate, 1, 0.0);
            break;
        default:
            break;
    }

    org.compile();
    return org;
}

// LOO / single 的消融必须在每次 apply_genes 之后强制生效，
// 否则 ES 会把被消融的连接又学回来。
static void enforce_variant(CellularOrganism& org, Variant v) {
    auto z = [&](uint32_t from, uint32_t to, uint8_t port) {
        for (auto& s : org.synapses) {
            if (s.from_cell_id == from && s.to_cell_id == to && s.to_port == port) {
                s.weight = 0.0;
            }
        }
    };
    switch (v) {
        case Variant::LooDelay:
            z(kDLeft, kCompare, 0); z(kDRight, kCompare, 1); break;
        case Variant::LooEvent:
            z(kEvent, kGate, 1); break;
        case Variant::LooCompare:
            z(kCompare, kGate, 0); break;
        case Variant::LooGate:
            z(kGate, kActPos, 0); z(kGate, kActNeg, 0); break;
        case Variant::SingleDelay:
            z(kGate, kActPos, 0); z(kGate, kActNeg, 0); break;
        case Variant::SingleEvent:
            z(kGate, kActPos, 0); z(kGate, kActNeg, 0); break;
        case Variant::SingleCompare:
            z(kGate, kActPos, 0); z(kGate, kActNeg, 0); break;
        case Variant::SingleGate:
            z(kCompare, kGate, 0); z(kEvent, kGate, 1); break;
        default:
            break;
    }
    // 消融改的是 synapses，必须重新编译，否则 forward 仍走旧的 compiled_synapses_。
    org.compile();
}

static std::vector<double> export_genes(const CellularOrganism& org, Variant v) {
    std::vector<double> g;
    if (v == Variant::Monolith) {
        for (const auto& s : org.synapses) if (s.is_active) g.push_back(s.weight);
        for (const auto& c : org.cells) { g.push_back(c.param1); g.push_back(c.param2); }
        return g;
    }
    // 只训练器官间连接/读出；器官内部参数固定。
    for (const auto& s : org.synapses) {
        if (!s.is_active) continue;
        const bool trainable =
            (s.from_cell_id == kDLeft && s.to_cell_id == kCompare) ||
            (s.from_cell_id == kDRight && s.to_cell_id == kCompare) ||
            (s.from_cell_id == kCompare && s.to_cell_id == kGate) ||
            (s.from_cell_id == kEvent && s.to_cell_id == kGate) ||
            (s.to_cell_id == kActPos) || (s.to_cell_id == kActNeg);
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
                (s.from_cell_id == kDLeft && s.to_cell_id == kCompare) ||
                (s.from_cell_id == kDRight && s.to_cell_id == kCompare) ||
                (s.from_cell_id == kCompare && s.to_cell_id == kGate) ||
                (s.from_cell_id == kEvent && s.to_cell_id == kGate) ||
                (s.to_cell_id == kActPos) || (s.to_cell_id == kActNeg);
            if (trainable && k < g.size()) s.weight = g[k++];
        }
    }
    org.compile();
}

// 单集：返回是否答对。
static bool episode(CellularOrganism& org, uint32_t seed, bool has_pulse, bool cue_left,
                    int pulse_tick) {
    org.reset_state(false);
    for (int t = 0; t <= kResponseTick; ++t) {
        double in[4] = {0};
        if (t == 0) in[cue_left ? 0 : 1] = 1.0;
        if (has_pulse && t == pulse_tick) in[2] = 1.0;
        if (t == kResponseTick) in[3] = 1.0;
        auto acts = org.forward_nd(in, 4, false);
        if (t == kResponseTick) {
            const double diff = acts.positive_action - acts.negative_action;
            // 响应幅度门禁：禁止用 1e-13 的微小输出“符号正确”骗过判据。
            if (!has_pulse) return std::abs(diff) < 1e-6;
            return std::abs(diff) >= 0.2 && (cue_left ? (diff > 0.0) : (diff < 0.0));
        }
    }
    return false;
}

// 调试用：返回响应拍 (pos-neg) 与关键细胞输出。
static double debug_diff(CellularOrganism& org, bool has_pulse, bool cue_left, int pulse_tick,
                         double* dleft = nullptr, double* dright = nullptr,
                         double* event = nullptr, double* gate = nullptr) {
    org.reset_state(false);
    double diff = 0.0;
    for (int t = 0; t <= kResponseTick; ++t) {
        double in[4] = {0};
        if (t == 0) in[cue_left ? 0 : 1] = 1.0;
        if (has_pulse && t == pulse_tick) in[2] = 1.0;
        if (t == kResponseTick) in[3] = 1.0;
        auto acts = org.forward_nd(in, 4, false);
        if (t == kResponseTick) {
            diff = acts.positive_action - acts.negative_action;
            if (dleft) *dleft = org.cells[kDLeft].output_val;
            if (dright) *dright = org.cells[kDRight].output_val;
            if (event) *event = org.cells[kEvent].output_val;
            if (gate) *gate = org.cells[kGate].output_val;
        }
    }
    return diff;
}

static EvalResult evaluate(CellularOrganism& org, int episodes, uint32_t seed) {
    EvalResult r;
    r.total = episodes;
    std::mt19937 rng(seed);
    for (int e = 0; e < episodes; ++e) {
        const bool has_pulse = (rng() % 100) < 50;
        const bool cue_left = (rng() % 2) == 0;
        const int pulse_tick = 2 + static_cast<int>(rng() % 5);
        const bool ok = episode(org, seed + static_cast<uint32_t>(e) * 17u,
                               has_pulse, cue_left, pulse_tick);
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

static std::string variant_name(Variant v) {
    switch (v) {
        case Variant::Full: return "full";
        case Variant::LooDelay: return "loo_delay";
        case Variant::LooEvent: return "loo_event";
        case Variant::LooCompare: return "loo_compare";
        case Variant::LooGate: return "loo_gate";
        case Variant::SingleDelay: return "single_delay";
        case Variant::SingleEvent: return "single_event";
        case Variant::SingleCompare: return "single_compare";
        case Variant::SingleGate: return "single_gate";
        case Variant::Monolith: return "monolith";
    }
    return "?";
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
        else if (a == "--debug") opt.debug = true;
        else if (a == "--out") opt.out = next();
        else if (a == "--gens") opt.gens = std::stoi(next());
        else if (a == "--pop") opt.pop = std::stoi(next());
        else if (a == "--seeds") opt.seeds = std::stoi(next());
        else if (a == "--eval-episodes") opt.episodes_eval = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: bench_organ_composition [--quick] [--gens N] [--pop N] "
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
    if (opt.out.empty()) opt.out = "runs/organ_composition_" + today() + ".json";
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    std::vector<Variant> variants = {
        Variant::Full, Variant::LooDelay, Variant::LooEvent, Variant::LooCompare,
        Variant::LooGate, Variant::SingleDelay, Variant::SingleEvent,
        Variant::SingleCompare, Variant::SingleGate, Variant::Monolith,
    };

    std::cout << "E-C organ composition | pop=" << opt.pop << " gens=" << opt.gens
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
            if (opt.debug && (v == Variant::Full || v == Variant::SingleDelay) && s == 0) {
                for (bool hp : {false, true}) {
                    for (bool cl : {false, true}) {
                        double dl = 0, dr = 0, e = 0, g = 0;
                        double diff = debug_diff(org, hp, cl, 4, &dl, &dr, &e, &g);
                        std::cout << "    [debug single_delay] pulse=" << hp
                                  << " cue_left=" << cl << " diff=" << diff
                                  << " dL=" << dl << " dR=" << dr
                                  << " e=" << e << " g=" << g << "\n";
                    }
                }
            }
            results.push_back({variant_name(v), static_cast<int>(seed), es.fitness, ev});
        }
    }

    auto mean_sr = [&](const std::string& name) {
        double sum = 0.0;
        int n = 0;
        for (const auto& r : results) {
            if (r.name == name) { sum += r.eval.sr; ++n; }
        }
        return n ? sum / n : 0.0;
    };

    const double full = mean_sr("full");
    const double monolith = mean_sr("monolith");
    double best_single = 0.0;
    std::string best_single_name = "single_gate";
    for (const char* n : {"single_delay", "single_event", "single_compare", "single_gate"}) {
        const double v = mean_sr(n);
        if (v > best_single) { best_single = v; best_single_name = n; }
    }
    const double loo_delay = mean_sr("loo_delay");
    const double loo_event = mean_sr("loo_event");
    const double loo_compare = mean_sr("loo_compare");
    const double loo_gate = mean_sr("loo_gate");
    const double min_loo = std::min({loo_delay, loo_event, loo_compare, loo_gate});

    const bool super_add = (full - best_single) >= 0.25;
    const bool loo_ok = (full - min_loo) >= 0.20;
    const bool mono_ok = (full - monolith) >= 0.10;
    const bool gate = super_add && loo_ok && mono_ok;

    // full vs monolith McNemar（按 seed 配对）
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

    auto mean_pulse = [&](const std::string& name) {
        int ok = 0, total = 0;
        for (const auto& r : results) {
            if (r.name == name) { ok += r.eval.pulse_ok; total += r.eval.pulse_total; }
        }
        return total ? static_cast<double>(ok) / total : 0.0;
    };
    const double full_pulse = mean_pulse("full");
    const double best_single_pulse = mean_pulse(best_single_name);

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
    const double p_best_single = mcnemar_exact(b2, c2);

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"E-C organ composition\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"pop\":" << opt.pop << ",\"gens\":" << opt.gens
       << ",\"seeds\":" << opt.seeds << ",\"episodes_train\":" << opt.episodes_train
       << ",\"episodes_eval\":" << opt.episodes_eval << "},\n";
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
       << ",\"min_loo\":" << min_loo
       << ",\"monolith\":" << monolith
       << ",\"best_single_name\":\"" << best_single_name << "\""
       << ",\"full_vs_best_single\":" << (full - best_single)
       << ",\"full_vs_min_loo\":" << (full - min_loo)
       << ",\"full_vs_monolith\":" << (full - monolith)
       << ",\"full_pulse\":" << full_pulse
       << ",\"best_single_pulse\":" << best_single_pulse
       << ",\"full_vs_best_single_pulse\":" << (full_pulse - best_single_pulse)
       << ",\"p_full_vs_monolith\":" << std::scientific << p_mono << std::fixed
       << ",\"p_full_vs_best_single\":" << std::scientific << p_best_single << std::fixed
       << "},\n  \"gate\":{\"H3\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"super_additive\":" << (super_add ? "true" : "false")
       << ",\"loo_ok\":" << (loo_ok ? "true" : "false")
       << ",\"monolith_ok\":" << (mono_ok ? "true" : "false") << "}\n}\n";
    os.close();

    std::cout << "\n== aggregate ==\n";
    for (Variant v : variants) {
        std::cout << "  " << std::setw(16) << variant_name(v) << " SR=" << mean_sr(variant_name(v)) << "\n";
    }
    std::cout << "GATE: H3 " << (gate ? "PASS" : "FAIL")
              << " (full=" << full << " best_single=" << best_single
              << " [" << best_single_name << "] min_loo=" << min_loo
              << " monolith=" << monolith
              << " full_pulse=" << full_pulse
              << " best_single_pulse=" << best_single_pulse
              << " dpulse=" << (full_pulse - best_single_pulse)
              << " p_best=" << p_best_single
              << " p_mono=" << p_mono << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}

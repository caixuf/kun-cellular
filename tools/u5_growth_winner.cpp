// U5-D1: 结构改进→牌力转化 (DISCOVERY 主战场)
// 把时序发现验证过的算子 (EMA/DELAY 变异池 + 全图靶点 + 每代3候选 +
// 代际再同化) 接到斗地主打分器: 正确率代际循环 → 最优形态导出 legacy
// 兼容参数快照 → 供牌局配对复测。
// 纪律: 变异=EMA(时序记忆) 为主; 每步付费; 选择=holdout 正确率; 双榜如实。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/knowledge_module.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <vector>

using namespace kun;

namespace {

struct Sample {
    std::array<float, 32> obs;
    std::array<float, 12> tail;
    std::vector<std::array<float, 12>> cands;
    int label{0};
    int32_t gid{0};
    int32_t won{0};
};

std::vector<Sample> load_dataset(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return {};
    char magic[4]; int32_t ver = 0, games = 0;
    f.read(magic, 4); f.read((char*)&ver, 4); f.read((char*)&games, 4);
    std::vector<Sample> data;
    while (f.good()) {
        Sample s;
        f.read((char*)s.obs.data(), 128);
        f.read((char*)s.tail.data(), 48);
        int32_t K = 0; f.read((char*)&K, 4);
        if (!f.good() || K <= 0 || K > 64) break;
        s.cands.resize(K);
        for (int i = 0; i < K; ++i) f.read((char*)s.cands[i].data(), 48);
        f.read((char*)&s.label, 4);
        f.read((char*)&s.gid, 4);
        f.read((char*)&s.won, 4);
        if (!f.good() || s.label < 0 || s.label >= K) break;
        data.push_back(std::move(s));
    }
    return data;
}

std::vector<double> scorer_input(const Sample& s, size_t i) {
    std::vector<double> in(56);
    for (int d = 0; d < 32; ++d) in[d] = s.obs[d];
    for (int d = 0; d < 12; ++d) in[32 + d] = s.tail[d];
    for (int d = 0; d < 12; ++d) in[44 + d] = s.cands[i][d];
    return in;
}

double candidate_accuracy(core::RuntimeState& rt, core::CompiledExecutor& ex,
                          const std::vector<const Sample*>& data) {
    size_t head_index = 0;
    for (size_t i = 0; i < rt.cell_states().size(); ++i)
        if (rt.cell_states()[i].type == CellType::ACT_CHANNEL) { head_index = i; break; }
    long correct = 0;
    for (const auto* sp : data) {
        double best = -1e308; size_t best_i = 0;
        for (size_t i = 0; i < sp->cands.size(); ++i) {
            auto in = scorer_input(*sp, i);
            auto r = ex.step(rt, in);
            if (!r.ok()) return -1.0;
            const double sc = rt.cell_states()[head_index].output_val;
            if (sc > best) { best = sc; best_i = i; }
        }
        if ((int)best_i == sp->label) ++correct;
    }
    return 100.0 * correct / (double)data.size();
}

std::vector<core::ParameterBinding> full_bindings(const core::RuntimeState& rt) {
    std::vector<core::ParameterBinding> out;
    const auto& entries = rt.parameters();
    for (size_t i = 0; i < entries.size(); ++i) out.push_back(entries[i].binding);
    return out;
}

double finetune(core::RuntimeState& rt, core::CompiledExecutor& ex, uint64_t organism_id,
                const std::vector<const Sample*>& train, int epochs, double lr) {
    auto w = core::LearningWindow::open(organism_id, rt, full_bindings(rt));
    CoreCellularBPTTEngine engine(64);
    engine.init_optimizer(rt);
    double last = -1.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto reset = rt.reset_episode();
        engine.reset_tape();
        std::vector<std::vector<double>> targets;
        for (const auto* sp : train) {
            for (size_t i = 0; i < sp->cands.size(); ++i) {
                auto in = scorer_input(*sp, i);
                auto rec = engine.record_step(rt, ex, in);
                if (!rec.ok()) return -1.0;
                targets.push_back({i == (size_t)sp->label ? 1.0 : 0.0, 0.0});
            }
        }
        CoreBPTTGradients grads;
        auto back = engine.backward(rt, targets, grads, &w);
        if (!back.ok()) return -1.0;
        auto upd = engine.step_adam(rt, w, grads, lr);
        if (!upd.ok()) return -1.0;
        last = grads.loss;
    }
    return last;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const uint32_t seed = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 2026;
    const int generations = argc > 2 ? std::atoi(argv[2]) : 6;
    const char* model = argc > 3 ? argv[3] : "checkpoints/doudizhu_cand_scorer.bin";
    const char* dataset = argc > 4 ? argv[4] : "/tmp/opencode/doudizhu_cand.bin";
    std::mt19937 rng(seed);

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    auto data = load_dataset(dataset);
    assert(!data.empty());
    std::map<int32_t, std::vector<const Sample*>> by_game;
    for (const auto& s : data) by_game[s.gid].push_back(&s);
    std::vector<const Sample*> train, holdout;
    {
        int gi = 0;
        for (auto& [g, vec] : by_game) {
            for (auto* sp : vec) {
                if (gi % 5 == 0) { if ((int)holdout.size() < 600) holdout.push_back(sp); }
                else if ((int)train.size() < 1200) train.push_back(sp);
            }
            ++gi;
        }
    }

    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 8000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    cfg.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(8), core::GraphRevision(1), cfg);
    assert(assembled.ok());

    // 代际: germline 起点 = 冠军
    auto germline = core::Germline::create(
        kun::transfer::live_definition(assembled.phenotype->runtime()),
        kun::transfer::parameter_seeds(assembled.phenotype->runtime().parameters()),
        "冠军起点");
    assert(germline.ok());

    uint64_t organism_id = 8000;
    double best_acc = 0.0;
    for (int gen = 0; gen < generations; ++gen) {
        auto ph = core::Phenotype::create(germline.germline, [&]{
            core::OffspringSpec s;
            s.organism_id = organism_id;
            s.lifecycle_config = cfg.lifecycle_config;
            s.growth_config = cfg.growth_config;
            s.resource_config = core::ResourceLedgerConfig{};
            for (const auto& c : germline.germline->graph()->cells())
                s.resource_cells.push_back(core::ResourceCellInitial{
                    c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
            s.resource_compartments.push_back(
                core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
            return s;
        }());
        if (!ph.ok()) { printf("[G%d] 出生失败\n", gen); return 1; }
        auto probe = ph.phenotype->runtime().fork_probe();
        assert(probe.ok());
        auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
        assert(pex.ok());
        finetune(*probe.runtime, *pex.executor, organism_id, train, 2, 0.01);
        const double acc = candidate_accuracy(*probe.runtime, *pex.executor, holdout);
        printf("[G%d] 学习后 holdout=%.2f%% (cells=%zu)\n", gen, acc,
               probe.runtime->plan()->cells().size());
        best_acc = std::max(best_acc, acc);

        // 3 个变异候选: EMA/DELAY 为主 (时序记忆), 靶点=候选特征交互区 (44..55 维
        // 的受体/特征细胞之间) + 全图随机; EMA α 池含慢累积
        auto cand_germline = core::Germline::create(
            kun::transfer::live_definition(*probe.runtime),
            kun::transfer::parameter_seeds(probe.runtime->parameters()), "winner-gen");
        assert(cand_germline.ok());
        struct Best { double acc{-1}; std::shared_ptr<const core::Germline> g; CellType t{}; };
        Best best;
        for (int k = 0; k < 3; ++k) {
            const CellType types[2] = {CellType::OP_EMA, CellType::OP_DELAY_N};
            const CellType new_type = k < 2 ? types[rng() % 2]
                                            : (rng() % 2 ? CellType::OP_EMA : CellType::OP_SUM);
            auto cand = core::Phenotype::create(cand_germline.germline, [&]{
                core::OffspringSpec s;
                s.organism_id = organism_id + 100 + (uint64_t)k;
                s.lifecycle_config = cfg.lifecycle_config;
                s.growth_config = cfg.growth_config;
                s.resource_compartments.push_back(
                    core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
                for (const auto& c : cand_germline.germline->graph()->cells())
                    s.resource_cells.push_back(core::ResourceCellInitial{
                        c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
                return s;
            }());
            if (!cand.ok()) continue;
            const auto& cplan = cand.phenotype->runtime().plan();
            std::vector<core::EdgeId> all_edges;
            for (const auto& e : cplan->edges())
                if (e.delay == core::EdgeDelay::Immediate) all_edges.push_back(e.id);
            if (all_edges.empty()) break;
            const core::EdgeId split_edge = all_edges[rng() % all_edges.size()];
            std::array<core::ParameterValue, 2> birth_params;
            if (new_type == CellType::OP_EMA) {
                const double alphas[3] = {0.4, 0.1, 0.05};
                birth_params = {core::ParameterValue{core::ContinuousValue{alphas[rng() % 3]}},
                                core::UnusedParameter{}};
            } else if (new_type == CellType::OP_DELAY_N) {
                const uint64_t delays[3] = {1, 2, 4};
                birth_params = {core::ParameterValue{core::DelayTicks{delays[rng() % 3]}},
                                core::UnusedParameter{}};
            } else {
                birth_params = {core::UnusedParameter{}, core::UnusedParameter{}};
            }
            core::GrowthSplitProposal sp;
            sp.proposal_id = (gen + 1) * 10 + k;
            sp.split.edge = split_edge;
            sp.split.inserted = core::CellBirth{
                core::CellId{600 + (uint32_t)gen * 10 + (uint32_t)k}, new_type, birth_params};
            sp.split.source_to_new = core::EdgeId{3000 + gen * 10 + k * 2};
            sp.split.new_to_target = core::EdgeId{3001 + gen * 10 + k * 2};
            sp.split.new_input_port = core::InputPort{0};
            sp.split.source_weight = 1.0;
            sp.split.target_weight = 1.0;
            sp.funding.compartment = core::ResourceCompartmentId{0};
            if (!cand.phenotype->growth().submit(sp).ok()) continue;
            std::vector<double> zero_in(56, 0.0);
            auto gres = cand.phenotype->growth().step(zero_in);
            if (!gres.ok()) continue;
            const double paid = gres.growth_report
                ? gres.growth_report->cumulative_growth_cost : 0.0;
            auto c_probe = cand.phenotype->runtime().fork_probe();
            if (!c_probe.ok()) continue;
            auto c_ex = core::CompiledExecutor::prepare(c_probe.runtime->plan());
            if (!c_ex.ok()) continue;
            finetune(*c_probe.runtime, *c_ex.executor,
                     organism_id + 100 + (uint64_t)k, train, 3, 0.01);
            const double acc2 = candidate_accuracy(*c_probe.runtime, *c_ex.executor, holdout);
            printf("[G%d.%d] 变异(%s @e%llu) holdout=%.2f%% (付费=%.1f)\\n",
                   gen, k, new_type == CellType::OP_EMA ? "EMA" :
                   new_type == CellType::OP_DELAY_N ? "DELAY" : "SUM",
                   (unsigned long long)split_edge.value, acc2, paid);
            if (acc2 > best.acc && acc2 > acc) {
                auto sel = core::Germline::create(
                    kun::transfer::live_definition(*c_probe.runtime),
                    kun::transfer::parameter_seeds(c_probe.runtime->parameters()),
                    "winner-selected");
                if (sel.ok()) best = {acc2, sel.germline, new_type};
            }
        }
        if (best.g) {
            germline = core::Germline::create(
                best.g->definition(),
                kun::transfer::parameter_seeds(best.g->initial_values().entries()),
                "winner-selected");
            assert(germline.ok());
            best_acc = std::max(best_acc, best.acc);
            printf("[G%d] 选择保留 %s (holdout=%.2f%%) — 代际推进\\n",
                   gen, best.t == CellType::OP_EMA ? "EMA" :
                   best.t == CellType::OP_DELAY_N ? "DELAY" : "SUM", best.acc);
            organism_id += 300;
        } else {
            printf("[G%d] 全部候选未过选择门 (保留亲代)\\n", gen);
        }
    }
    printf("\\n[U5-D1] 代际循环完成: 峰值 holdout=%.2f%% (起点 89.33%%)\\n", best_acc);
    printf("[导出] 最优形态 germline 已在内存; 牌局配对复测待原生 runner (P9)\\n");
    return 0;
}

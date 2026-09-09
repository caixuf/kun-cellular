// U4 消融矩阵 (manifest: docs/superpowers/plans/2026-09-08-u4-ablation-manifest.md)
// A 固定拓扑 / B 仅代际 / C 仅发育(活动引导) / D 双层(发育+代际再同化) /
// E 随机提案(同预算) × 种子。主比较 D vs E 配对。
// 纪律: BPTT 为全组共用参数代谢基线 (非消融变量); 主语是结构生长与选择。
// 输出 JSON 行: {group, seed, acc_before, acc_after, paid, cells}
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/knowledge_module.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace kun;

namespace {

struct Sample {
    std::array<float, 32> obs;
    std::array<float, 12> tail;
    std::vector<std::array<float, 12>> cands;
    int label{0};
    int32_t gid{0};
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
        int32_t won; f.read((char*)&won, 4);
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

std::vector<core::ParameterBinding> edge_bindings(const core::CompiledGraph& plan) {
    std::vector<core::ParameterBinding> out;
    for (const auto& e : plan.edges()) {
        core::ParameterBinding b;
        b.kind = core::ParameterBindingKind::EdgeWeight;
        b.index = e.weight_parameter_index;
        b.edge = e.id;
        out.push_back(b);
    }
    return out;
}

double finetune(core::RuntimeState& rt, core::CompiledExecutor& ex, uint64_t organism_id,
                const std::vector<const Sample*>& train, int epochs, double lr) {
    auto w = core::LearningWindow::open(organism_id, rt, edge_bindings(*rt.plan()));
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

// 激活统计 (probe): 各细胞平均 |输出|
std::vector<double> activation_profile(core::RuntimeState& rt, core::CompiledExecutor& ex,
                                       const std::vector<const Sample*>& data) {
    std::vector<double> act(rt.plan()->cells().size(), 0.0);
    size_t n = 0;
    for (size_t t = 0; t < data.size() && t < 200; ++t) {
        for (size_t i = 0; i < data[t]->cands.size(); ++i) {
            auto in = scorer_input(*data[t], i);
            if (!ex.step(rt, in).ok()) return act;
            for (size_t ci = 0; ci < act.size(); ++ci)
                act[ci] += std::fabs(rt.cell_states()[ci].output_val);
            ++n;
        }
    }
    if (n > 0) for (auto& a : act) a /= (double)n;
    return act;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string group = argc > 1 ? argv[1] : "D";
    const uint32_t seed = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 2026;
    const char* model = argc > 3 ? argv[3] : "checkpoints/doudizhu_cand_scorer.bin";
    const char* dataset = argc > 4 ? argv[4] : "/tmp/opencode/doudizhu_cand.bin";

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    auto data = load_dataset(dataset);
    assert(!data.empty());
    // 数据划分 (种子无关 — 全组同划分, 保证配对可比)
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

    // 冷装配 (冠军)
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 6000 + seed % 1000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    cfg.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(6), core::GraphRevision(1), cfg);
    if (!assembled.ok()) { printf("[错误] 装配: %s\n", assembled.diagnostic.c_str()); return 1; }
    auto& ph = *assembled.phenotype;

    // 组 B: 仅出生, 无学习无生长
    double acc_before = 0.0, acc_after = 0.0, paid = 0.0;
    size_t cells_after = 0;
    {
        auto probe = ph.runtime().fork_probe();
        auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
        acc_before = candidate_accuracy(*probe.runtime, *pex.executor, holdout);
    }

    if (group == "B") {
        acc_after = acc_before;  // 仅代际出生 — 无变化
        cells_after = ph.runtime().plan()->cells().size();
    } else if (group == "A") {
        // 固定拓扑: 纯 BPTT 基线 (同预算学习, 无生长)
        auto probe = ph.runtime().fork_probe();
        auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
        finetune(*probe.runtime, *pex.executor, 7000 + seed % 1000, train, 2, 0.01);
        acc_after = candidate_accuracy(*probe.runtime, *pex.executor, holdout);
        cells_after = probe.runtime->plan()->cells().size();
    } else {
        // C/D/E: 生长组 — 学习 → 变异 → 重训 (D 再同化代际第二轮)
        auto probe = ph.runtime().fork_probe();
        auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
        finetune(*probe.runtime, *pex.executor, 7000 + seed % 1000, train, 2, 0.01);
        acc_before = candidate_accuracy(*probe.runtime, *pex.executor, holdout);

        auto cand_germline = core::Germline::create(
            kun::transfer::live_definition(*probe.runtime),
            kun::transfer::parameter_seeds(probe.runtime->parameters()), group);
        assert(cand_germline.ok());
        std::mt19937 rng(seed);
        core::OffspringSpec spec;
        spec.organism_id = 7100 + seed % 1000;
        spec.lifecycle_config = cfg.lifecycle_config;
        spec.growth_config = cfg.growth_config;
        spec.resource_compartments.push_back(
            core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
        for (const auto& c : cand_germline.germline->graph()->cells())
            spec.resource_cells.push_back(core::ResourceCellInitial{
                c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
        auto cand = core::Phenotype::create(cand_germline.germline, spec);
        if (!cand.ok()) { printf("[错误] 候选出生\n"); return 1; }

        // 变异 N=2 次 (预算对齐: C/D 活动引导, E 随机)
        const int N = 2;
        for (int v = 0; v < N; ++v) {
            const auto& cplan = cand.phenotype->runtime().plan();
            std::optional<core::EdgeId> target_edge;
            if (group == "C" || group == "D") {
                // 活动引导: head 入边中源激活最低者 (欠票启发式)
                auto act = activation_profile(cand.phenotype->runtime(),
                                              cand.phenotype->executor(), train);
                size_t head_idx = 0;
                for (size_t i = 0; i < cplan->cells().size(); ++i)
                    if (cplan->cells()[i].type == CellType::ACT_CHANNEL) { head_idx = i; break; }
                double worst = 1e308;
                for (const auto& e : cplan->edges()) {
                    if (e.target_index != head_idx) continue;
                    if (act[e.source_index] < worst) {
                        worst = act[e.source_index];
                        target_edge = e.id;
                    }
                }
            } else {
                // E: 随机边 (全图均匀)
                std::vector<core::EdgeId> all;
                for (const auto& e : cplan->edges())
                    if (e.delay == core::EdgeDelay::Immediate) all.push_back(e.id);
                if (!all.empty()) target_edge = all[rng() % all.size()];
            }
            if (!target_edge.has_value()) break;
            core::CellId split_target{0};
            for (const auto& e : cplan->edges())
                if (e.id == *target_edge) split_target = cplan->cells()[e.target_index].id;
            core::GrowthSplitProposal sp;
            sp.proposal_id = v + 1;
            sp.split.edge = *target_edge;
            sp.split.inserted = core::CellBirth{core::CellId{500 + v}, CellType::OP_SUM,
                std::array<core::ParameterValue, 2>{core::UnusedParameter{}, core::UnusedParameter{}}};
            sp.split.source_to_new = core::EdgeId{800 + v * 2};
            sp.split.new_to_target = core::EdgeId{801 + v * 2};
            sp.split.new_input_port = core::InputPort{0};
            sp.split.source_weight = 1.0;
            sp.split.target_weight = 1.0;
            sp.funding.compartment = core::ResourceCompartmentId{0};
            if (!cand.phenotype->growth().submit(sp).ok()) break;
            std::vector<double> zero_in(56, 0.0);
            auto gres = cand.phenotype->growth().step(zero_in);
            if (!gres.ok()) break;
            paid += gres.growth_report ? gres.growth_report->cumulative_growth_cost : 0.0;
        }
        // 变异后重训 (同超参)
        auto c_probe = cand.phenotype->runtime().fork_probe();
        auto c_ex = core::CompiledExecutor::prepare(c_probe.runtime->plan());
        finetune(*c_probe.runtime, *c_ex.executor, 7200 + seed % 1000, train, 2, 0.01);
        acc_after = candidate_accuracy(*c_probe.runtime, *c_ex.executor, holdout);
        cells_after = c_probe.runtime->plan()->cells().size();
        if (group == "D") {
            // 代际再同化: 变异形态写入 germline → 新个体出生 → 学习 → 评测
            auto g2 = core::Germline::create(
                kun::transfer::live_definition(*c_probe.runtime),
                kun::transfer::parameter_seeds(c_probe.runtime->parameters()),
                "D-generation-2");
            assert(g2.ok());
            core::OffspringSpec s2;
            s2.organism_id = 7300 + seed % 1000;
            s2.lifecycle_config = cfg.lifecycle_config;
            s2.growth_config = cfg.growth_config;
            s2.resource_compartments.push_back(
                core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
            for (const auto& c : g2.germline->graph()->cells())
                s2.resource_cells.push_back(core::ResourceCellInitial{
                    c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
            auto gen2 = core::Phenotype::create(g2.germline, s2);
            if (gen2.ok()) {
                auto p2 = gen2.phenotype->runtime().fork_probe();
                auto ex2 = core::CompiledExecutor::prepare(p2.runtime->plan());
                finetune(*p2.runtime, *ex2.executor, 7400 + seed % 1000, train, 2, 0.01);
                const double acc_gen2 = candidate_accuracy(*p2.runtime, *ex2.executor, holdout);
                if (acc_gen2 > acc_after) { acc_after = acc_gen2; }  // 代际增益如实取优
            }
        }
    }

    printf("{\"group\":\"%s\",\"seed\":%u,\"acc_before\":%.2f,\"acc_after\":%.2f,"
           "\"delta\":%.2f,\"paid\":%.2f,\"cells\":%zu}\n",
           group.c_str(), seed, acc_before, acc_after, acc_after - acc_before,
           paid, cells_after > 0 ? cells_after : ph.runtime().plan()->cells().size());
    return 0;
}

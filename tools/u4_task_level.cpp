// U4-TL: 七环任务级闭合 (斗地主真数据) — 环 3 盲测泛化 + 环 4 真结构改变
// 冠军 82 细胞 → 冷装配新核心 → probe 体 BPTT 微调 (候选软监督) →
// holdout 候选正确率 (盲测) → 真数据激活统计欠票边 → 付费分裂 → 对比。
// 纪律: 学习/评测走 probe (不推本体生命钟); BPTT 只是参数代谢工具,
// 环 4 主语是结构生长 (付费建造), 学习贡献分解呈现。
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
#include <random>
#include <map>
#include <optional>
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
    if (!f.good()) { printf("[错误] 数据集打不开: %s\n", path); return {}; }
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
    for (int d = 0; d < 12; ++d) in[32 + d] = s.tail[d];   // 32..43
    for (int d = 0; d < 12; ++d) in[44 + d] = s.cands[i][d];
    return in;
}

// 候选正确率 (probe 体): argmax 分数 == 教师所选候选
double candidate_accuracy(core::RuntimeState& rt, core::CompiledExecutor& ex,
                          const std::vector<const Sample*>& data) {
    // score head 通道槽 0: ACT_CHANNEL 中输出槽位
    size_t head_index = 0; int slot = 0;
    for (size_t i = 0; i < rt.cell_states().size(); ++i) {
        if (rt.cell_states()[i].type == CellType::ACT_CHANNEL) {
            if (slot == 0) { head_index = i; break; }
            ++slot;
        }
    }
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

// BPTT 候选软监督: 教师候选目标 1.0, 其他 0.0 (score head 通道行)
std::pair<int, double> finetune(
    core::RuntimeState& rt, core::CompiledExecutor& ex, uint64_t organism_id,
    const std::vector<const Sample*>& data, int epochs, double lr) {
    std::vector<core::ParameterBinding> bindings;
    for (const auto& e : rt.plan()->edges()) {
        core::ParameterBinding b;
        b.kind = core::ParameterBindingKind::EdgeWeight;
        b.index = e.weight_parameter_index;
        b.edge = e.id;
        bindings.push_back(b);
    }
    auto w = core::LearningWindow::open(organism_id, rt, bindings);
    CoreCellularBPTTEngine engine(64);
    engine.init_optimizer(rt);
    double last = -1.0;
    int used = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto reset = rt.reset_episode();
        engine.reset_tape();
        std::vector<std::vector<double>> targets;  // 每拍一行 [score_target, value_target(0)]
        for (const auto* sp : data) {
            for (size_t i = 0; i < sp->cands.size(); ++i) {
                auto in = scorer_input(*sp, i);
                auto rec = engine.record_step(rt, ex, in);
                if (!rec.ok()) { printf("[record 错误] %s\n", rec.error->reason.c_str()); return {-1, -1}; }
                targets.push_back({i == (size_t)sp->label ? 1.0 : 0.0, 0.0});
            }
        }
        CoreBPTTGradients grads;
        auto back = engine.backward(rt, targets, grads, &w);
        if (!back.ok()) { printf("[backward 错误] %s\n", back.error->reason.c_str()); return {-1, -1}; }
        auto upd = engine.step_adam(rt, w, grads, lr);
        if (!upd.ok()) { printf("[adam 错误]\n"); return {-1, -1}; }
        last = grads.loss;
        used = epoch + 1;
        printf("  [微调] epoch %d/%d loss=%.6f\n", epoch + 1, epochs, last);
    }
    return {used, last};
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model = argc > 1 ? argv[1] : "checkpoints/doudizhu_cand_scorer.bin";
    const char* dataset = argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_cand.bin";
    const int train_n = argc > 3 ? std::atoi(argv[3]) : 1500;
    const int holdout_n = argc > 4 ? std::atoi(argv[4]) : 800;

    // 1. 冠军 → 冷装配
    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else {
            auto j = CellularOrganism::load_checkpoint_json(model);
            if (j.cells.empty()) { printf("[错误] 冠军加载失败\n"); return 1; }
            org = std::move(j);
        }
    }
    printf("[冠军] cells=%zu syn=%zu\n", org.cells.size(), org.compiled_synapses_.size());
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 3000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(3), core::GraphRevision(1), cfg);
    if (!assembled.ok()) { printf("[装配失败] %s\n", assembled.diagnostic.c_str()); return 1; }
    core::Phenotype& ph = *assembled.phenotype;
    printf("[新核心] 冷装配 ✓ (内生 runtime/lifecycle/ledger/growth)\n");

    // 2. 数据 + 按 gid 划分 train/holdout (跨局盲测)
    auto data = load_dataset(dataset);
    if (data.empty()) return 1;
    std::map<int32_t, std::vector<const Sample*>> by_game;
    for (const auto& s : data) by_game[s.gid].push_back(&s);
    std::vector<int32_t> gids;
    for (auto& [g, _] : by_game) gids.push_back(g);
    std::mt19937 rng(20260908);
    std::shuffle(gids.begin(), gids.end(), rng);
    std::vector<const Sample*> train, holdout;
    for (size_t i = 0; i < gids.size(); ++i) {
        for (auto* sp : by_game[gids[i]]) {
            if ((int)train.size() < train_n && i % 5 != 0) train.push_back(sp);
            else if ((int)holdout.size() < holdout_n && i % 5 == 0) holdout.push_back(sp);
        }
    }
    printf("[数据] 总决策=%zu train=%zu (跨局) holdout=%zu (盲测)\n",
           data.size(), train.size(), holdout.size());

    // 3. 环 3: probe 盲测泛化 (微调前 → 微调后, holdout 与 train 同步)
    auto probe = ph.runtime().fork_probe();
    assert(probe.ok());
    auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
    assert(pex.ok());
    const double base_hold = candidate_accuracy(*probe.runtime, *pex.executor, holdout);
    const double base_train = candidate_accuracy(*probe.runtime, *pex.executor, train);
    printf("[环3 基线] train 正确率=%.2f%% | holdout 正确率=%.2f%% (冠军冷迁移即战力)\n",
           base_train, base_hold);
    auto [used, loss] = finetune(*probe.runtime, *pex.executor, 3000, train, 2, 0.01);
    if (used < 0) return 1;
    const double ft_hold = candidate_accuracy(*probe.runtime, *pex.executor, holdout);
    const double ft_train = candidate_accuracy(*probe.runtime, *pex.executor, train);
    printf("[环3 盲测泛化] 微调后 train=%.2f%% holdout=%.2f%% "
           "(Δholdout=%+.2fpp, train-holdout 泛化差=%.2fpp)\n",
           ft_train, ft_hold, ft_hold - base_hold, ft_train - ft_hold);

    // 4. 环 4: 真数据激活统计 → 欠票分裂 (付费建造, 独立兄弟体)
    {
        core::OffspringSpec spec;
        spec.organism_id = 3001;
        spec.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
        spec.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};  // 建造有价
        // 兄弟体从同冠军 germline 出生 (无性繁殖)
        auto g2 = core::Germline::create(
            kun::transfer::live_definition(*probe.runtime),
            kun::transfer::parameter_seeds(probe.runtime->parameters()),
            "U4-TL 冠军微调形态");
        assert(g2.ok());
        spec.resource_cells.clear();
        spec.resource_compartments.clear();
        spec.resource_compartments.push_back(
            core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
        for (const auto& c : g2.germline->graph()->cells())
            spec.resource_cells.push_back(core::ResourceCellInitial{
                c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
        auto sib = core::Phenotype::create(g2.germline, spec);
        if (!sib.ok()) {
            printf("[环4 错误] 兄弟体出生: %s\n",
                   sib.error ? sib.error->reason.c_str() : "?");
            return 1;
        }
        // 激活统计: probe 体回放 (研究行为不推本体生命钟 — U2.1 纪律)
        auto sib_probe = sib.phenotype->runtime().fork_probe();
        assert(sib_probe.ok());
        auto sib_pex = core::CompiledExecutor::prepare(sib_probe.runtime->plan());
        assert(sib_pex.ok());
        std::vector<double> act(sib_probe.runtime->plan()->cells().size(), 0.0);
        for (size_t t = 0; t < train.size() && t < 200; ++t) {
            const auto& s = *train[t];
            for (size_t i = 0; i < s.cands.size(); ++i) {
                auto in = scorer_input(s, i);
                auto r = sib_pex.executor->step(*sib_probe.runtime, in);
                if (!r.ok()) { printf("[统计 step 失败]\n"); return 1; }
                for (size_t ci = 0; ci < act.size(); ++ci)
                    act[ci] += std::fabs(sib_probe.runtime->cell_states()[ci].output_val);
            }
        }
        // 欠票靶点: score head 入边中源细胞激活最低者
        size_t head_idx = 0;
        for (size_t i = 0; i < sib_probe.runtime->plan()->cells().size(); ++i)
            if (sib_probe.runtime->plan()->cells()[i].type == CellType::ACT_CHANNEL) { head_idx = i; break; }
        std::optional<core::EdgeId> starved_edge;
        double starved_act = 1e308;
        for (const auto& e : sib_probe.runtime->plan()->edges()) {
            if (e.target_index != head_idx) continue;
            const size_t src = e.source_index;
            if (act[src] < starved_act) {
                starved_act = act[src];
                starved_edge = e.id;
            }
        }
        if (!starved_edge.has_value()) { printf("[环4] 无入边可分裂\n"); return 1; }
        printf("[环4 激活统计] 头部入边欠票源 act=%.4f → 分裂该因果边\n", starved_act);

        const size_t cells_before = sib.phenotype->runtime().plan()->cells().size();
        const auto rev_before = sib.phenotype->runtime().revision().value;
        core::GrowthSplitProposal sp;
        sp.proposal_id = 1;
        sp.split.edge = *starved_edge;
        sp.split.inserted = core::CellBirth{core::CellId{500}, CellType::OP_SUM,
            std::array<core::ParameterValue, 2>{core::UnusedParameter{}, core::UnusedParameter{}}};
        sp.split.source_to_new = core::EdgeId{900};
        sp.split.new_to_target = core::EdgeId{901};
        sp.split.new_input_port = core::InputPort{0};
        sp.split.source_weight = 1.0;
        sp.split.target_weight = 1.0;
        sp.funding.compartment = core::ResourceCompartmentId{0};
        assert(sib.phenotype->growth().submit(sp).ok());
        std::vector<double> first_in(56, 0.0);
        auto gres = sib.phenotype->growth().step(first_in);
        if (!gres.ok()) {
            printf("[环4 生长失败] growth: %s | lifecycle: %s\n",
                   gres.growth_error ? gres.growth_error->reason.c_str() : "-",
                   gres.lifecycle.error ? gres.lifecycle.error->reason.c_str() : "-");
            return 1;
        }
        const double paid = gres.growth_report ? gres.growth_report->cumulative_growth_cost : 0.0;
        printf("[环4 真结构改变] 付费=%.3f 细胞 %zu→%zu revision→%llu ✓\n",
               paid, cells_before, sib.phenotype->runtime().plan()->cells().size(),
               (unsigned long long)sib.phenotype->runtime().revision().value);
        assert(paid > 0.0);
        // 分裂后正确率 (SUM 直通, 结构变化即评)
        auto sex = core::CompiledExecutor::prepare(sib.phenotype->runtime().plan());
        assert(sex.ok());
        const double split_hold = candidate_accuracy(
            sib.phenotype->runtime(), *sex.executor, holdout);
        printf("[环4 分裂后 holdout=%.2f%% (基线 %.2f%%, Δ=%+.2fpp) — 结构/代谢贡献如实呈现\n",
               split_hold, ft_hold, split_hold - ft_hold);
    }

    printf("\n[U4-TL] 环3 (盲测泛化) + 环4 (真结构改变) 任务级执行完毕\n");
    return 0;
}

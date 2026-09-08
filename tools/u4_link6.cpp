// U4 环6 任务级: 冠军学得通路固化为知识模块 → 图书馆 R8 验证 →
// 无血缘 StrictCore 冠军变体冷边界借入 → 盲测正确率变化。
// 纪律: 知识移动走图书馆 (germline 本体), 不靠梯度; 借入付费 (资源结算)。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "tasks/transfer/knowledge_module.hpp"
#include "tasks/transfer/germline_library.hpp"
#include "tasks/transfer/knowledge_adoption.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model = argc > 1 ? argv[1] : "checkpoints/doudizhu_cand_scorer.bin";
    const char* dataset = argc > 2 ? argv[2] : "/tmp/opencode/doudizhu_cand.bin";

    std::remove("/tmp/opencode/u4_link6_library.sqlite3");

    // 1. 冠军 → 冷装配 (LegacyCompatible, 学得 live 权重)
    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else { org = CellularOrganism::load_checkpoint_json(model); }
    }
    assert(!org.cells.empty());
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 4000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto champ = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(4), core::GraphRevision(1), cfg);
    assert(champ.ok());
    printf("[冠军] 冷装配 ✓ cells=%zu\n", champ.phenotype->runtime().plan()->cells().size());

    // 2. 数据 → holdout (跨局)
    auto data = load_dataset(dataset);
    assert(!data.empty());
    std::map<int32_t, std::vector<const Sample*>> by_game;
    for (const auto& s : data) by_game[s.gid].push_back(&s);
    std::vector<const Sample*> holdout;
    {
        int gi = 0;
        for (auto& [g, vec] : by_game) {
            if (gi % 5 == 0)
                for (auto* sp : vec) if ((int)holdout.size() < 600) holdout.push_back(sp);
            ++gi;
        }
    }
    printf("[数据] holdout=%zu 决策 (盲测)\n", holdout.size());

    // 3. 固化学得通路: 冠军受体 0 → 直连 SUM 的学得边权 → StrictCore motif
    //    (知识固化 = live 参数写入先天模板; 模块契约: 单输入提取器)
    const auto& plan = *champ.phenotype->runtime().plan();
    const auto& entries = champ.phenotype->runtime().parameters();
    // 找受体 0 (id 0) 的直连出边 (immediate), 取其学得权重
    double learned_w = 0.0;
    bool found = false;
    for (const auto& e : plan.edges()) {
        const uint32_t s = plan.cells()[e.source_index].id.value;
        if (s != 0 || e.delay != core::EdgeDelay::Immediate) continue;
        // 学得权重
        const double wv = std::get<core::ContinuousValue>(
            entries[e.weight_parameter_index].value).value;
        if (std::fabs(wv) > 1e-9 && std::isfinite(wv)) {
            learned_w = wv;
            found = true;
            break;
        }
    }
    if (!found) { printf("[错误] 冠军受体 0 无学得直连边\n"); return 1; }
    printf("[固化] 冠军受体0→直通 学得权重=%.6f → StrictCore 知识模块\n", learned_w);

    core::GraphDefinition motif;
    motif.identity = core::GraphIdentity(41);
    motif.revision = core::GraphRevision{1};
    motif.profile = SemanticProfile::StrictCore;
    motif.semantic_version = 1;
    motif.cells.push_back({core::CellId{0}, CellType::SENSE_RAW_INPUT_0});
    motif.cells.push_back({core::CellId{1}, CellType::OP_SUM});
    motif.edges.push_back({core::EdgeId{1}, core::CellId{0}, core::OutputPort{0},
                           core::CellId{1}, core::InputPort{0}, core::EdgeDelay::Immediate});
    core::InitialParameterSeeds mseeds;
    mseeds.cell_parameters.push_back({core::CellId{0}, core::ParameterSlot::Param1,
                                      core::ParameterValue{core::ContinuousValue{1.0}}});
    mseeds.cell_parameters.push_back({core::CellId{0}, core::ParameterSlot::Param2,
                                      core::UnusedParameter{}});
    mseeds.cell_parameters.push_back({core::CellId{1}, core::ParameterSlot::Param1,
                                      core::UnusedParameter{}});
    mseeds.cell_parameters.push_back({core::CellId{1}, core::ParameterSlot::Param2,
                                      core::UnusedParameter{}});
    mseeds.edge_weights.push_back({core::EdgeId{1}, learned_w});
    auto knowledge = core::Germline::create(motif, mseeds, "冠军谱系: 受体0 直通提取器");
    assert(knowledge.ok());

    transfer::ModuleContract contract;
    contract.interface_id = "u4-receptor0-extractor";
    contract.environment = "doudizhu-holdout";
    contract.input_count = 1;
    contract.output = core::CellId{1};
    contract.semantic_profile = SemanticProfile::StrictCore;
    contract.semantic_version = 1;
    auto module = transfer::KnowledgeModule::from_germline(
        *knowledge.germline, contract, "champion-lineage-v1");

    // 4. 图书馆: 出版 → R8 验证 → 借阅
    transfer::GermlineLibraryStore store("/tmp/opencode/u4_link6_library.sqlite3");
    transfer::KnowledgeRef ref{"u4-receptor0-extractor", "v1"};
    store.publish(ref, "冠军受体0直通提取器", module, {});
    transfer::EvaluationProtocol protocol;
    protocol.protocol_id = "KUN-R8-APPROVED/u4-link6";
    protocol.environment = "doudizhu-holdout";
    protocol.absolute_tolerance = 0.0;
    auto make_trace = [&](uint64_t seed) {
        transfer::EvaluationTrace t; t.seed = seed;
        auto fresh = module.fresh_runtime();
        auto mex = core::CompiledExecutor::prepare(fresh->plan());
        assert(mex.ok());
        for (double xv : {0.5, -0.3, 1.0, 0.25, -0.75}) {
            std::vector<double> in{xv};
            auto r = mex.executor->step(*fresh, in);
            assert(r.ok());
            double out = 0.0;
            for (const auto& c : fresh->cell_states())
                if (c.cell.value == 1) out = c.output_val;
            t.frames.push_back({{xv}, {out}});
        }
        return t;
    };
    protocol.train.push_back(make_trace(1));
    protocol.train.push_back(make_trace(2));
    protocol.ood.push_back(make_trace(3));
    protocol.ood.push_back(make_trace(4));
    auto report = store.evaluate(ref, protocol, "u4-link6");
    assert(report.passed);
    auto borrowed = store.borrow(ref, contract, "strictcore-champion-variant");
    printf("[图书馆] publish→R8验证→borrow ✓ (borrow_event=%llu)\n",
           (unsigned long long)borrowed.event_sequence);

    // 5. 无血缘借入方: 冠军图的 StrictCore 编译版 (独立 germline, 独立 organism_id)
    core::GraphDefinition bdef;
    bdef.identity = core::GraphIdentity(42);
    bdef.revision = core::GraphRevision{1};
    bdef.profile = SemanticProfile::StrictCore;
    bdef.semantic_version = 1;
    for (const auto& c : plan.cells())
        bdef.cells.push_back({c.id, c.type});
    for (const auto& e : plan.edges())
        bdef.edges.push_back({e.id, plan.cells()[e.source_index].id, e.source_port,
                              plan.cells()[e.target_index].id, e.target_port, e.delay});
    // 附加未分化 RAW 受体 (冠军图全是 SENSE_CHANNEL 宽契约感受器;
    // adoption 的 motif 源契约 = SENSE_RAW_INPUT_0 — 年轻个体的胚胎感受器)
    bdef.cells.push_back({core::CellId{90}, CellType::SENSE_RAW_INPUT_0});
    // 动态选挂点目标: 第一个 input_port_count>=1 的非受体细胞
    {
        core::CellId hook_target{0}; bool found_t = false;
        CellType hook_type{CellType::OP_SUM};
        for (const auto& c : plan.cells()) {
            const auto ct = core::contract_for(c.type);
            if (ct.has_value() && ct->get().input_port_count >= 1 &&
                c.type != CellType::SENSE_CHANNEL && c.type != CellType::SENSE_RAW_INPUT_0) {
                hook_target = c.id; hook_type = c.type; found_t = true; break;
            }
        }
        assert(found_t);
        printf("[诊断] hook_target=%u type=%d ports=%d\n", hook_target.value,
               (int)hook_type, core::contract_for(hook_type)->get().input_port_count);
        bdef.edges.push_back({core::EdgeId{990}, core::CellId{90}, core::OutputPort{0},
                              hook_target, core::InputPort{0},
                              core::EdgeDelay::Immediate});
    }
    // 出生种子 (未训练): 受体 gain=1.0 — 借入方是无血缘的年轻个体
    auto birth_import = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(43), core::GraphRevision(1));
    assert(birth_import.ok());
    auto bseeds = kun::transfer::parameter_seeds(
        birth_import.snapshot->initial_parameter_values()->entries());
    // 借入方是年轻个体: 受体 gain 回到出生值 1.0 (adoption 的 unscaled-receptor 契约)
    for (auto& p : bseeds.cell_parameters)
        if (p.cell.value == 0 && p.slot == core::ParameterSlot::Param1)
            p.value = core::ParameterValue{core::ContinuousValue{1.0}};
    // 缺口语义: 受体 0 直通边权重削弱 (功能贫弱 → 借入补强)
    for (auto& w : bseeds.edge_weights) {
        for (const auto& e : plan.edges()) {
            if (e.id == w.edge) {
                const uint32_t s = plan.cells()[e.source_index].id.value;
                if (s == 0 && w.initial_weight == learned_w) w.initial_weight = learned_w * 0.1;
            }
        }
    }
    bseeds.cell_parameters.push_back(
        {core::CellId{90}, core::ParameterSlot::Param1,
         core::ParameterValue{core::ContinuousValue{1.0}}});
    bseeds.cell_parameters.push_back(
        {core::CellId{90}, core::ParameterSlot::Param2, core::UnusedParameter{}});
    bseeds.edge_weights.push_back({core::EdgeId{990}, 0.3});
    auto b_germline = core::Germline::create(bdef, bseeds, "StrictCore 冠军变体 (无血缘)");
    if (!b_germline.ok()) {
        printf("[环6 错误] B germline: %s\n",
               b_germline.error ? b_germline.error->reason.c_str() : "?");
        return 1;
    }
    core::OffspringSpec bspec;
    bspec.organism_id = 4001;  // 独立 id = 无血缘
    bspec.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    bspec.resource_compartments.push_back(
        core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
    for (const auto& c : b_germline.germline->graph()->cells())
        bspec.resource_cells.push_back(core::ResourceCellInitial{
            c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
    auto borrower = core::Phenotype::create(b_germline.germline, bspec);
    if (!borrower.ok()) {
        printf("[错误] 借入方出生: %s\n", borrower.error->reason.c_str());
        return 1;
    }
    auto b_probe = borrower.phenotype->runtime().fork_probe();
    assert(b_probe.ok());
    auto b_ex = core::CompiledExecutor::prepare(b_probe.runtime->plan());
    assert(b_ex.ok());
    const double acc_before = candidate_accuracy(*b_probe.runtime, *b_ex.executor, holdout);
    printf("[借入前] StrictCore 变体 holdout=%.2f%%\n", acc_before);

    // 6. 冷边界借入: 挂点 = 受体 0 的 immediate 直连边 (gain=1.0 ✓ coupling≠0)
    std::optional<core::EdgeId> hook;
    for (const auto& e : b_probe.runtime->plan()->edges()) {
        const uint32_t s = b_probe.runtime->plan()->cells()[e.source_index].id.value;
        if (s != 0 || e.delay != core::EdgeDelay::Immediate) continue;
        // 源须 unscaled receptor (gain=1.0)
        const auto gparam = b_probe.runtime->parameter(
            b_probe.runtime->plan()->cells()[e.source_index].id, core::ParameterSlot::Param1);
        if (!gparam.has_value()) continue;
        const double gain = std::get<core::ContinuousValue>(*gparam).value;
        if (std::fabs(gain - 1.0) > 1e-12) continue;
        const double coupling = std::get<core::ContinuousValue>(
            b_probe.runtime->parameters()[e.weight_parameter_index].value).value;
        if (coupling == 0.0) continue;
        hook = e.id;
        break;
    }
    {
        const auto g0 = borrower.phenotype->runtime().parameter(
            core::CellId{0}, core::ParameterSlot::Param1);
        printf("[实证] 借入方本体 受体0 gain=%s\n",
               g0.has_value()
                   ? std::to_string(std::get<core::ContinuousValue>(*g0).value).c_str()
                   : "无");
    }
    if (!hook.has_value()) {
        printf("[诊断] 受体0 出边:\n");
        for (const auto& e : b_probe.runtime->plan()->edges()) {
            const uint32_t s = b_probe.runtime->plan()->cells()[e.source_index].id.value;
            if (s != 0) continue;
            const auto gparam = b_probe.runtime->parameter(
                b_probe.runtime->plan()->cells()[e.source_index].id, core::ParameterSlot::Param1);
            const double gain = gparam.has_value()
                ? std::get<core::ContinuousValue>(*gparam).value : -999;
            const double coupling = std::get<core::ContinuousValue>(
                b_probe.runtime->parameters()[e.weight_parameter_index].value).value;
            printf("  edge=%llu delay=%d coupling=%.4f gain=%.4f\n",
                   (unsigned long long)e.id.value, (int)e.delay, coupling, gain);
        }
        return 1;
    }
    transfer::AdoptionRequest req;
    req.target_contract = contract;
    req.target_edge = *hook;
    req.expected_tick = borrower.phenotype->runtime().tick();
    req.funding.compartment = core::ResourceCompartmentId{0};
    req.cell_cost = 2.0;
    req.synapse_cost = 0.1;
    req.initial_energy = 5.0;
    {
        // 定位: adoption 的 require 精确复算 (edge 990)
        const auto& tp = *borrower.phenotype->runtime().plan();
        for (const auto& e : tp.edges()) {
            if (e.id.value != 990) continue;
            const auto& src_cell = tp.cells()[e.source_index];
            const auto g1 = borrower.phenotype->runtime().parameter(
                src_cell.id, core::ParameterSlot::Param1);
            printf("[复算] edge990 src_index=%zu id=%u type=%d (RAW枚举=%d) "
                   "param1=%s\n",
                   e.source_index, src_cell.id.value, (int)src_cell.type,
                   (int)CellType::SENSE_RAW_INPUT_0,
                   g1.has_value()
                       ? std::to_string(std::get<core::ContinuousValue>(g1.value()).value).c_str()
                       : "nullopt");
        }
    }
    std::vector<double> adopt_inputs{0.5};
    auto receipt = transfer::adopt_at_cold_boundary(
        *borrower.phenotype, borrowed, req, adopt_inputs);
    printf("[借入] 付费=%.3f 细胞 +%zu 边 +%zu revision %llu→%llu\n",
           receipt.paid_cost, receipt.inserted_cells.size(), receipt.new_edges.size(),
           (unsigned long long)receipt.revision_before.value,
           (unsigned long long)receipt.revision_after.value);

    // 7. 借入后盲测 (新 probe)
    auto b_probe2 = borrower.phenotype->runtime().fork_probe();
    assert(b_probe2.ok());
    auto b_ex2 = core::CompiledExecutor::prepare(b_probe2.runtime->plan());
    assert(b_ex2.ok());
    const double acc_after = candidate_accuracy(*b_probe2.runtime, *b_ex2.executor, holdout);
    printf("[环6 任务级结果] 借入前=%.2f%% → 借入后=%.2f%% (Δ=%+.2fpp) "
           "cells %zu→%zu 付费=%.3f\n",
           acc_before, acc_after, acc_after - acc_before,
           b_probe.runtime->plan()->cells().size(),
           b_probe2.runtime->plan()->cells().size(), receipt.paid_cost);
    // 判据: 借入真实改变行为 (付费>0, 图推进, 正确率有限变化 — 增益方向如实)
    assert(receipt.paid_cost > 0.0);
    assert(b_probe2.runtime->plan()->cells().size() ==
           b_probe.runtime->plan()->cells().size() + 1);  // 83 → 84

    printf("\n[环6 任务级] 知识跨谱系借入在斗地主数据上执行完毕\n");
    return 0;
}

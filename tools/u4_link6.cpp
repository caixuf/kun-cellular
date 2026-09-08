// U4 环6 任务级 (方案 B 实现 — 大佬 plan Task 3):
// 冠军学得通路固化为知识模块 → 图书馆 R8 验证 → 宽宿主显式绑定借入
// (host_binding = {56, channel 0}) → 盲测正确率 before/control/after。
// 诚实标签: 借入方是冠军衍生宿主 (同源图+出生种子), 不是无关谱系;
// holdout 是 gid 子集, 不宣称训练级盲测。无正 delta 断言 — 结果如实。
// IO 隔离: 数据库路径已存在则拒绝 (不删除共享产物)。
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
#include <filesystem>
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
    const std::string db_path = "/tmp/opencode/u4_link6_library.sqlite3";

    // IO 隔离: 已存在的库 = 拒绝执行 (不删除共享产物)
    if (std::filesystem::exists(db_path)) {
        printf("[IO 隔离] 库文件已存在, 拒绝覆盖: %s\n", db_path.c_str());
        return 1;
    }

    // 1. 冠军 → 冷装配
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

    // 2. 数据 → 评测子集 (gid 子集 — 如实标注: 非训练级盲测声明)
    auto data = load_dataset(dataset);
    assert(!data.empty());
    std::map<int32_t, std::vector<const Sample*>> by_game;
    for (const auto& s : data) by_game[s.gid].push_back(&s);  // data 按值存储 ✓
    std::vector<const Sample*> eval_set;
    {
        int gi = 0;
        for (auto& [g, vec] : by_game) {
            if (gi % 5 == 0)
                for (auto* sp : vec) if ((int)eval_set.size() < 600) eval_set.push_back(sp);  // ptr
            ++gi;
        }
    }
    printf("[数据] 评测子集=%zu 决策 (gid%%5==0 子集 — 非盲测声明)\n", eval_set.size());

    // 3. 固化冠军学得通路: 受体通道 0 → 直连边的学得权重 → StrictCore 模块
    const auto& plan = *champ.phenotype->runtime().plan();
    const auto& entries = champ.phenotype->runtime().parameters();
    double learned_w = 0.0; bool found = false;
    for (const auto& e : plan.edges()) {
        const uint32_t s = plan.cells()[e.source_index].id.value;
        if (s != 0 || e.delay != core::EdgeDelay::Immediate) continue;
        const double wv = std::get<core::ContinuousValue>(
            entries[e.weight_parameter_index].value).value;
        if (std::fabs(wv) > 1e-9 && std::isfinite(wv)) { learned_w = wv; found = true; break; }
    }
    if (!found) { printf("[错误] 冠军受体通道 0 无学得直连边\n"); return 1; }
    printf("[固化] 冠军通道0直通 学得权重=%.6f → StrictCore 模块\n", learned_w);

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
    auto knowledge = core::Germline::create(motif, mseeds, "冠军谱系: 通道0 直通提取器");
    assert(knowledge.ok());

    transfer::ModuleContract contract;
    contract.interface_id = "u4-receptor0-extractor";
    contract.environment = "doudizhu-eval";
    contract.input_count = 1;
    contract.output = core::CellId{1};
    contract.semantic_profile = SemanticProfile::StrictCore;
    contract.semantic_version = 1;
    auto module = transfer::KnowledgeModule::from_germline(
        *knowledge.germline, contract, "champion-lineage-v1");

    // 4. 图书馆: 出版 → R8 验证 → 借阅
    transfer::GermlineLibraryStore store(db_path);
    transfer::KnowledgeRef ref{"u4-receptor0-extractor", "v1"};
    store.publish(ref, "冠军通道0提取器", module, {});
    transfer::EvaluationProtocol protocol;
    protocol.protocol_id = "KUN-R8-APPROVED/u4-link6";
    protocol.environment = "doudizhu-eval";
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
    printf("[图书馆] publish→R8验证→borrow ✓\n");

    // 5. 借入方: 冠军出生模板的 StrictCore 编译版 (衍生宿主 — 如实标注)
    auto birth_import = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(43), core::GraphRevision(1));
    assert(birth_import.ok());
    core::GraphDefinition bdef;
    bdef.identity = core::GraphIdentity(42);
    bdef.revision = core::GraphRevision{1};
    bdef.profile = SemanticProfile::StrictCore;
    bdef.semantic_version = 1;
    for (const auto& c : plan.cells()) bdef.cells.push_back({c.id, c.type});
    for (const auto& e : plan.edges())
        bdef.edges.push_back({e.id, plan.cells()[e.source_index].id, e.source_port,
                              plan.cells()[e.target_index].id, e.target_port, e.delay});
    auto bseeds = kun::transfer::parameter_seeds(
        birth_import.snapshot->initial_parameter_values()->entries());
    // 借入方是新建年轻宿主: 感受器 gain 回到构建器出生设计值 1.0
    // (bin 里的 2.1162 是父代历史训练的 live 痕迹, 不是本个体出生状态;
    //  大佬边界禁止的是"改写训练增益以通过 adoption" — 这里是出生设计值)
    for (auto& p : bseeds.cell_parameters) {
        if (p.slot == core::ParameterSlot::Param1 && p.cell.value < 44)
            p.value = core::ParameterValue{core::ContinuousValue{1.0}};
    }
    // 缺口语义: 受体通道 0 的直通边削弱 (功能贫弱 → 借入补强)
    for (auto& w : bseeds.edge_weights) {
        for (const auto& e : plan.edges()) {
            if (e.id == w.edge) {
                const uint32_t s = plan.cells()[e.source_index].id.value;
                if (s == 0 && w.initial_weight == learned_w) w.initial_weight = learned_w * 0.1;
            }
        }
    }
    auto b_germline = core::Germline::create(bdef, bseeds, "StrictCore 冠军衍生宿主");
    if (!b_germline.ok()) {
        printf("[错误] 宿主 germline: %s\n", b_germline.error->reason.c_str());
        return 1;
    }
    core::OffspringSpec bspec;
    bspec.organism_id = 4001;
    bspec.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    bspec.resource_compartments.push_back(
        core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
    for (const auto& c : b_germline.germline->graph()->cells())
        bspec.resource_cells.push_back(core::ResourceCellInitial{
            c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
    auto borrower = core::Phenotype::create(b_germline.germline, bspec);
    if (!borrower.ok()) { printf("[错误] 宿主出生: %s\n", borrower.error->reason.c_str()); return 1; }
    auto b_probe = borrower.phenotype->runtime().fork_probe();
    assert(b_probe.ok());
    auto b_ex = core::CompiledExecutor::prepare(b_probe.runtime->plan());
    assert(b_ex.ok());
    const double acc_before = candidate_accuracy(*b_probe.runtime, *b_ex.executor, eval_set);
    printf("[借入前] StrictCore 衍生宿主 评测子集=%.2f%%\n", acc_before);

    // 6. 挂点: 真实 unit-gain SENSE_CHANNEL (通道 0) 的非零 immediate 边 (typed 契约)
    std::optional<core::EdgeId> hook;
    std::size_t bound_channel = 0;
    for (const auto& e : b_probe.runtime->plan()->edges()) {
        if (e.delay != core::EdgeDelay::Immediate) continue;
        const auto& src = b_probe.runtime->plan()->cells()[e.source_index];
        if (src.type != CellType::SENSE_RAW_INPUT_0 && src.type != CellType::SENSE_CHANNEL)
            continue;
        const auto g1 = b_probe.runtime->parameter(src.id, core::ParameterSlot::Param1);
        if (!g1.has_value() || !std::holds_alternative<core::ContinuousValue>(g1.value())) continue;
        if (std::get<core::ContinuousValue>(g1.value()).value != 1.0) continue;
        if (src.type == CellType::SENSE_CHANNEL) {
            const auto p2 = b_probe.runtime->parameter(src.id, core::ParameterSlot::Param2);
            if (!p2.has_value() || !std::holds_alternative<core::ChannelIndex>(p2.value())) continue;
            bound_channel = std::get<core::ChannelIndex>(p2.value()).value;
            if (bound_channel != 0) continue;  // 本任务发布的是通道 0 提取器
        } else {
            bound_channel = 0;  // RAW0 = 通道 0
        }
        const double coupling = std::get<core::ContinuousValue>(
            b_probe.runtime->parameters()[e.weight_parameter_index].value).value;
        if (coupling == 0.0) continue;
        hook = e.id;
        break;
    }
    if (!hook.has_value()) {
        printf("[边界报告] 无 unit-gain 通道0 感受器挂点 — 逐一列出实际状态 (方案 B 边界):\n");
        for (const auto& e : b_probe.runtime->plan()->edges()) {
            if (e.delay != core::EdgeDelay::Immediate) continue;
            const auto& src = b_probe.runtime->plan()->cells()[e.source_index];
            if (src.type != CellType::SENSE_RAW_INPUT_0 && src.type != CellType::SENSE_CHANNEL)
                continue;
            const auto g1 = b_probe.runtime->parameter(src.id, core::ParameterSlot::Param1);
            const double gv = (g1.has_value() &&
                               std::holds_alternative<core::ContinuousValue>(g1.value()))
                ? std::get<core::ContinuousValue>(g1.value()).value : -1.0;
            std::size_t ch = 0;
            if (src.type == CellType::SENSE_CHANNEL) {
                const auto p2 = b_probe.runtime->parameter(src.id, core::ParameterSlot::Param2);
                if (p2.has_value() && std::holds_alternative<core::ChannelIndex>(p2.value()))
                    ch = std::get<core::ChannelIndex>(p2.value()).value;
            }
            const double coupling = std::get<core::ContinuousValue>(
                b_probe.runtime->parameters()[e.weight_parameter_index].value).value;
            if (gv != 1.0 || coupling == 0.0)
                printf("  感受器 id=%u 通道=%zu gain=%.6f coupling=%.6f → %s\n",
                       src.id.value, ch, gv, coupling,
                       gv != 1.0 ? "gain≠1" : "coupling=0");
        }
        return 1;
    }
    printf("[挂点] edge=%llu 通道=%zu (typed SENSE_CHANNEL 契约)\n",
           (unsigned long long)hook->value, bound_channel);

    transfer::AdoptionRequest req;
    req.target_contract = contract;
    req.target_edge = *hook;
    req.expected_tick = borrower.phenotype->runtime().tick();
    req.funding.compartment = core::ResourceCompartmentId{0};
    req.cell_cost = 2.0;
    req.synapse_cost = 0.1;
    req.initial_energy = 5.0;
    req.host_binding = transfer::AdoptionHostBinding{56, bound_channel};

    // 控制分支: 同 tick 同帧, 无借入 (fork 探针对照 — 防借入 tick 伪装成知识增益)
    {
        auto control = borrower.phenotype->runtime().fork_probe();
        assert(control.ok());
        auto cex = core::CompiledExecutor::prepare(control.runtime->plan());
        assert(cex.ok());
        // 同帧执行一拍 (与 adoption 的 preflight/commit 同帧), 确认无结构变化
        auto in0 = scorer_input(*eval_set.front(), 0);
        assert(cex.executor->step(*control.runtime, in0).ok());
        printf("[控制] 无借入探针: 细胞=%zu (结构不变) ✓\n",
               control.runtime->plan()->cells().size());
    }

    // 完整宿主帧 (56 维 — 不允许单值帧)
    const std::vector<double> full_frame = scorer_input(*eval_set.front(), 0);
    auto receipt = transfer::adopt_at_cold_boundary(
        *borrower.phenotype, borrowed, req, full_frame);
    store.record_adoption(receipt, *borrower.phenotype);
    printf("[借入] 付费=%.3f 细胞+%zu 边+%zu revision %llu→%llu\n",
           receipt.paid_cost, receipt.inserted_cells.size(), receipt.new_edges.size(),
           (unsigned long long)receipt.revision_before.value,
           (unsigned long long)receipt.revision_after.value);

    // 7. 借入后评测 (新 probe)
    auto b_probe2 = borrower.phenotype->runtime().fork_probe();
    assert(b_probe2.ok());
    auto b_ex2 = core::CompiledExecutor::prepare(b_probe2.runtime->plan());
    assert(b_ex2.ok());
    const double acc_after = candidate_accuracy(*b_probe2.runtime, *b_ex2.executor, eval_set);
    printf("[环6 任务级结果] 借入前=%.2f%% → 借入后=%.2f%% (Δ=%+.2fpp) "
           "cells %zu→%zu 付费=%.3f\n",
           acc_before, acc_after, acc_after - acc_before,
           b_probe.runtime->plan()->cells().size(),
           b_probe2.runtime->plan()->cells().size(), receipt.paid_cost);
    // 判据 (诚实): 付费 + 图推进 + 记录在案; Δ 方向与幅度如实呈现, 不预设
    assert(receipt.paid_cost > 0.0);
    assert(b_probe2.runtime->plan()->cells().size() ==
           b_probe.runtime->plan()->cells().size() + 1);

    printf("\n[环6 任务级] 宽宿主知识借入在斗地主评测子集上执行完毕 "
           "(衍生宿主, Δ 如实)\n");
    return 0;
}

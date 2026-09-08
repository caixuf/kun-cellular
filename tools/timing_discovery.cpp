// 时序结构发现实验 (预注册协议见 docs/superpowers/plans/2026-09-08-timing-discovery-protocol.md)
// 命题: 演化引擎能从无时序结构的胚胎中自发发现可用的时序机制。
// 变异池含时序原语 (EMA/DELAY/滞回) + 代数对照; 环境选择 (DMS 正确率) 决定保留。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/knowledge_module.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

using namespace kun;

namespace {

// ── DMS 任务 ──
struct DmsSample {
    double sample_dir;              // ±1
    std::vector<std::vector<double>> frames;  // 10 拍 × in_dim
    double target;                  // ±1 (t8/t9 教师目标)
};

DmsSample make_dms(std::mt19937& rng) {
    std::uniform_real_distribution<double> dir(0.0, 1.0);
    DmsSample s;
    s.sample_dir = dir(rng) < 0.5 ? -1.0 : 1.0;
    s.frames.resize(10, std::vector<double>(4, 0.0));
    for (int t = 0; t < 3; ++t) s.frames[t][0] = s.sample_dir;   // 样本期
    s.frames[8][1] = 1.0;                                        // 测试脉冲
    s.target = s.sample_dir;
    return s;
}

const char* cell_type_name(CellType t) {
    switch (t) {
        case CellType::OP_EMA: return "OP_EMA";
        case CellType::OP_DELAY_N: return "OP_DELAY_N";
        case CellType::GATE_HYSTERESIS: return "GATE_HYSTERESIS";
        case CellType::OP_SUM: return "OP_SUM";
        case CellType::OP_SUB: return "OP_SUB";
        case CellType::OP_MULTIPLY: return "OP_MULTIPLY";
        default: return "?";
    }
}

// DMS 正确率: t9 时 ACT 通道 0 输出符号与样本方向一致
double dms_accuracy(core::RuntimeState& rt, core::CompiledExecutor& ex,
                    const std::vector<DmsSample>& set) {
    long correct = 0;
    for (const auto& s : set) {
        double out = 0.0;
        for (const auto& in : s.frames) {
            auto r = ex.step(rt, in);
            if (!r.ok()) return -1.0;
            for (const auto& c : rt.cell_states())
                if (c.type == CellType::ACT_CHANNEL) out = c.output_val;
        }
        if ((out > 0) == (s.sample_dir > 0)) ++correct;
    }
    return 100.0 * correct / (double)set.size();
}

// ── 胚胎 (无时序结构): in0/in1 受体 + SUM + ACT ch0 ──
CellularOrganism build_embryo() {
    CellularOrganism org;
    org.organism_id = 1;
    auto cell = [](uint32_t id, CellType t, double p1, double p2) {
        Cell c; c.id = id; c.type = t; c.param1 = p1; c.param2 = p2; return c;
    };
    auto syn = [](uint32_t f, uint32_t t, uint8_t p, double w) {
        Synapse s; s.from_cell_id = f; s.to_cell_id = t; s.to_port = p;
        s.weight = w; s.initial_weight = w; s.is_active = true; return s;
    };
    org.cells.push_back(cell(0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0));
    org.cells.push_back(cell(1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0));
    org.cells.push_back(cell(2, CellType::OP_SUM, 0.0, 0.0));
    org.cells.push_back(cell(3, CellType::ACT_CHANNEL, 1.0, 0.0));
    org.synapses.push_back(syn(0, 2, 0, 0.5));
    org.synapses.push_back(syn(1, 2, 1, 0.5));
    org.synapses.push_back(syn(2, 3, 0, 0.5));
    org.compile();
    return org;
}

// 完整参数窗口 (边权重 + 细胞参数, index 从 runtime 实测)
std::vector<core::ParameterBinding> full_bindings(const core::RuntimeState& rt) {
    std::vector<core::ParameterBinding> out;
    const auto& entries = rt.parameters();
    for (size_t i = 0; i < entries.size(); ++i) out.push_back(entries[i].binding);
    return out;
}

// BPTT 拟合 DMS (probe 体): 目标行 [ch0_target, 0]
std::pair<int, double> train_dms(core::RuntimeState& rt, core::CompiledExecutor& ex,
                                 uint64_t organism_id,
                                 const std::vector<DmsSample>& set,
                                 int epochs, double lr) {
    auto w = core::LearningWindow::open(organism_id, rt, full_bindings(rt));
    CoreCellularBPTTEngine engine(64);
    engine.init_optimizer(rt);
    double last = -1.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto reset = rt.reset_episode();
        engine.reset_tape();
        std::vector<std::vector<double>> targets;
        for (const auto& s : set) {
            for (int t = 0; t < 10; ++t) {
                auto rec = engine.record_step(rt, ex, s.frames[t]);
                if (!rec.ok()) return {-1, -1};
                // 教师行: t8/t9 时 ch0 = ±1, 其余拍 0
                const double tgt = (t >= 8) ? s.target : 0.0;
                targets.push_back({tgt, 0.0});
            }
        }
        CoreBPTTGradients grads;
        auto back = engine.backward(rt, targets, grads, &w);
        if (!back.ok()) { printf("[backward 错误] %s\n", back.error->reason.c_str()); return {-1, -1}; }
        auto upd = engine.step_adam(rt, w, grads, lr);
        if (!upd.ok()) { printf("[adam 错误]\n"); return {-1, -1}; }
        last = grads.loss;
    }
    return {epochs, last};
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool timing_pool = argc > 1 ? std::string(argv[1]) != "algebra" : true;
    const int generations = argc > 2 ? std::atoi(argv[2]) : 12;
    const uint32_t seed = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 2026;
    std::mt19937 rng(seed);

    printf("[时序结构发现] 变异池=%s 代数=%d seed=%u\n",
           timing_pool ? "含时序原语" : "纯代数对照", generations, seed);

    // 数据: train 12 / test 12 DMS 样本
    std::vector<DmsSample> train_set, test_set;
    for (int i = 0; i < 12; ++i) train_set.push_back(make_dms(rng));
    for (int i = 0; i < 12; ++i) test_set.push_back(make_dms(rng));

    // 出生: 胚胎 (无时序结构)
    auto embryo = build_embryo();
    auto imported = kun::migration::import_execution_snapshot(
        embryo, core::GraphIdentity(1), core::GraphRevision(1));
    assert(imported.ok());
    auto germline = core::Germline::create(
        imported.snapshot->graph_definition(),
        kun::transfer::parameter_seeds(
            imported.snapshot->initial_parameter_values()->entries()),
        "无时序结构胚胎");
    assert(germline.ok());

    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 5000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    cfg.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};

    const std::vector<CellType> timing_types = {
        CellType::OP_EMA, CellType::OP_DELAY_N, CellType::GATE_HYSTERESIS};
    const std::vector<CellType> algebra_types = {
        CellType::OP_SUM, CellType::OP_SUB, CellType::OP_MULTIPLY};
    const auto& pool = timing_pool ? timing_types : algebra_types;

    double best_acc = 0.0;
    uint64_t organism_id = 5000;
    std::vector<CellType> retained_timing;  // 结构账本: 被保留的时序原语

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
        if (!ph.ok()) { printf("[G%d] 出生失败: %s\n", gen,
            ph.error ? ph.error->reason.c_str() : "?"); return 1; }

        // 学习 (probe 体)
        auto probe = ph.phenotype->runtime().fork_probe();
        assert(probe.ok());
        auto pex = core::CompiledExecutor::prepare(probe.runtime->plan());
        assert(pex.ok());
        auto [ep, loss] = train_dms(*probe.runtime, *pex.executor, organism_id,
                                    train_set, 200, 0.02);
        if (ep < 0) return 1;
        const double acc = dms_accuracy(*probe.runtime, *pex.executor, test_set);
        printf("[G%d] 学习后 test 正确率=%.1f%% (loss=%.4f, cells=%zu)\n",
               gen, acc, loss, probe.runtime->plan()->cells().size());

        // 变异: 随机选类型, 分裂 head 通路某边 (变异随机, 选择不随机)
        const CellType new_type = pool[rng() % pool.size()];
        // 学习后的 live 同化候选
        auto candidate_germline = core::Germline::create(
            kun::transfer::live_definition(*probe.runtime),
            kun::transfer::parameter_seeds(probe.runtime->parameters()),
            timing_pool ? "timing-pool" : "algebra-pool");
        assert(candidate_germline.ok());

        // 在候选 (同形态) 上执行分裂
        auto cand = core::Phenotype::create(candidate_germline.germline, [&]{
            core::OffspringSpec s;
            s.organism_id = organism_id + 100;
            s.lifecycle_config = cfg.lifecycle_config;
            s.growth_config = cfg.growth_config;
            s.resource_config = core::ResourceLedgerConfig{};
            for (const auto& c : candidate_germline.germline->graph()->cells())
                s.resource_cells.push_back(core::ResourceCellInitial{
                    c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
            s.resource_compartments.push_back(
                core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});
            return s;
        }());
        if (!cand.ok()) { printf("[G%d] 候选出生失败\n", gen); return 1; }
        // 靶点: ACT 通道 0 的入边中随机一条 (通路上游)
        std::vector<core::EdgeId> head_in;
        const auto& cplan = cand.phenotype->runtime().plan();
        for (size_t i = 0; i < cplan->cells().size(); ++i) {
            if (cplan->cells()[i].type == CellType::ACT_CHANNEL) {
                for (const auto& e : cplan->edges())
                    if (e.target_index == i) head_in.push_back(e.id);
                break;
            }
        }
        if (head_in.empty()) { printf("[G%d] 无入边\n", gen); break; }
        const core::EdgeId split_edge = head_in[rng() % head_in.size()];
        // 挂边重定向 (源→新, 新→目标) 的权重取原边
        double src_w = 0.5;
        core::CellId split_target{0};
        core::CellId split_source{0};
        for (const auto& e : cplan->edges())
            if (e.id == split_edge) {
                split_target = cplan->cells()[e.target_index].id;
                split_source = cplan->cells()[e.source_index].id;
            }
        core::GrowthSplitProposal sp;
        sp.proposal_id = gen + 1;
        sp.split.edge = split_edge;
        const uint32_t new_id = 100 + (uint32_t)gen;
        sp.split.inserted = core::CellBirth{core::CellId{new_id}, new_type,
            std::array<core::ParameterValue, 2>{
                core::ParameterValue{core::ContinuousValue{0.4}},
                core::UnusedParameter{}}};
        sp.split.source_to_new = core::EdgeId{1000 + gen * 2};
        sp.split.new_to_target = core::EdgeId{1001 + gen * 2};
        sp.split.new_input_port = core::InputPort{0};
        sp.split.source_weight = src_w;
        sp.split.target_weight = 1.0;
        sp.funding.compartment = core::ResourceCompartmentId{0};
        if (!cand.phenotype->growth().submit(sp).ok()) {
            printf("[G%d] 提案拒绝\n", gen);
            break;
        }
        std::vector<double> zero_in(4, 0.0);
        auto gres = cand.phenotype->growth().step(zero_in);
        if (!gres.ok()) {
            printf("[G%d] 生长失败: %s (回滚 — 选择压力)\n", gen,
                   gres.growth_error ? gres.growth_error->reason.c_str() : "?");
            continue;  // 保留旧 germline
        }
        // 分裂后: 重训 + 评测 (变异体 vs 亲代 — 环境选择)
        auto c_probe = cand.phenotype->runtime().fork_probe();
        assert(c_probe.ok());
        auto c_ex = core::CompiledExecutor::prepare(c_probe.runtime->plan());
        assert(c_ex.ok());
        auto [ep2, loss2] = train_dms(*c_probe.runtime, *c_ex.executor,
                                      organism_id + 100, train_set, 200, 0.02);
        if (ep2 < 0) return 1;
        const double acc2 = dms_accuracy(*c_probe.runtime, *c_ex.executor, test_set);
        printf("[G%d] 变异(%s) 后 test=%.1f%% (付费=%.1f) — %s\n",
               gen, cell_type_name(new_type).c_str(), acc2,
               gres.growth_report ? gres.growth_report->cumulative_growth_cost : 0.0,
               acc2 > acc ? "选择保留" : "选择回滚");
        if (acc2 > acc && acc2 > best_acc) {
            best_acc = acc2;
            germline = core::Germline::create(
                kun::transfer::live_definition(*c_probe.runtime),
                kun::transfer::parameter_seeds(c_probe.runtime->parameters()),
                "timing-pool-selected");
            assert(germline.ok());
            if (new_type == CellType::OP_EMA || new_type == CellType::OP_DELAY_N ||
                new_type == CellType::GATE_HYSTERESIS)
                retained_timing.push_back(new_type);
            organism_id += 200;
        }
        // 个体代谢生命周期结束 (每代重建)
    }

    // 双榜
    printf("\n[MECHANISM_STATUS] 时序原语被保留: %zu 个 (%s)\n",
           retained_timing.size(),
           retained_timing.empty() ? "未出现" : "结构自发出现并被选择");
    printf("[DISCOVERY_STATUS] 最终 test 正确率=%.1f%% (判据 ≥80%% 机制成立)\n", best_acc);
    return 0;
}

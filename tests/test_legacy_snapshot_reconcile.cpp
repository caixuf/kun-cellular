// U1 双执行器对账: legacy CellularOrganism → execution_snapshot 冷导入 →
// 新核心 RuntimeState + ReferenceExecutor / CompiledExecutor
// 契约: LegacyCompatible 语义下新旧执行器逐细胞输出位级等价 (含记忆传播)
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/core/primitive_contract.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/compiled_executor.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <algorithm>
#include <random>
#include <vector>

using namespace kun;

namespace {

// 单通道单 episode: 相同输入序列下, legacy forward_nd vs 指定新执行器逐细胞对账
template <typename StepFn>
double reconcile_episode(CellularOrganism& org,
                         core::RuntimeState& runtime,
                         StepFn&& step_fn,
                         const std::vector<std::vector<double>>& inputs) {
    org.reset_state(false);
    auto reset = runtime.reset_episode();
    assert(!reset.error.has_value());
    double max_diff = 0.0;
    for (const auto& input : inputs) {
        org.forward_nd(input.data(), input.size(), false);
        std::map<uint32_t, double> legacy_out;
        for (const auto& c : org.cells) legacy_out[c.id] = c.output_val;
        auto result = step_fn(runtime, input);
        assert(result.ok());
        for (size_t i = 0; i < runtime.cell_states().size(); ++i) {
            const uint32_t id = runtime.cell_states()[i].cell.value;
            const auto it = legacy_out.find(id);
            const double legacy_v = it != legacy_out.end() ? it->second : 0.0;
            const double d = std::fabs(legacy_v - runtime.cell_states()[i].output_val);
            if (d > 1e-9) {
                printf("    [diff] cell %u type=%d legacy=%.9f new=%.9f\n",
                       id, (int)runtime.cell_states()[i].type, legacy_v,
                       runtime.cell_states()[i].output_val);
            }
            max_diff = std::max(max_diff, d);
        }
    }
    return max_diff;
}

inline Cell make_cell(uint32_t id, CellType type, double param1 = 1.0, double param2 = 0.0) {
    Cell c;
    c.id = id;
    c.type = type;
    c.param1 = param1;
    c.param2 = param2;
    return c;
}

inline Synapse make_synapse(uint32_t from_id, uint32_t to_id, uint8_t port, double weight) {
    Synapse s;
    s.from_cell_id = from_id;
    s.to_cell_id = to_id;
    s.to_port = port;
    s.weight = weight;
    s.initial_weight = weight;
    s.is_active = true;
    return s;
}

// 过滤无核心契约的细胞 (冷迁移按设计只支持契约覆盖类型)
CellularOrganism filter_contract_cells(const CellularOrganism& src) {
    CellularOrganism org = src;
    org.cells.erase(
        std::remove_if(org.cells.begin(), org.cells.end(),
                       [&](const Cell& c) { return !core::contract_for(c.type).has_value(); }),
        org.cells.end());
    std::map<uint32_t, bool> kept;
    for (const auto& c : org.cells) kept[c.id] = true;
    org.synapses.erase(
        std::remove_if(org.synapses.begin(), org.synapses.end(),
                       [&](const Synapse& s) {
                           return !kept[s.from_cell_id] || !kept[s.to_cell_id];
                       }),
        org.synapses.end());
    return org;
}

// 剪除目标契约不消费端口的突触 (legacy 随机图允许野端口; 新类型契约把关)
CellularOrganism prune_wild_ports(const CellularOrganism& src) {
    CellularOrganism org = src;
    std::map<uint32_t, kun::CellType> types;
    for (const auto& c : org.cells) types[c.id] = c.type;
    org.synapses.erase(
        std::remove_if(org.synapses.begin(), org.synapses.end(),
                       [&](Synapse& s) -> bool {
                           const auto contract = core::contract_for(types[s.to_cell_id]);
                           if (!contract.has_value()) return true;
                           auto consumed = [&](uint8_t port) {
                               return port < contract->get().input_port_count;
                           };
                           if (consumed(s.to_port)) return false;
                           const uint8_t other = static_cast<uint8_t>(1 - s.to_port);
                           if (consumed(other)) {
                               s.to_port = other;  // 重接到契约消费的端口
                               return false;
                           }
                           return true;  // 两端口皆不消费, 剪除
                       }),
        org.synapses.end());
    return org;
}

// 剪除无突触引用的孤儿细胞 (legacy 执行序只包含被引用细胞; 新契约把关拒收孤儿)
CellularOrganism prune_orphans(const CellularOrganism& src) {
    CellularOrganism org = src;
    std::map<uint32_t, bool> referenced;
    for (const auto& c : org.cells) referenced[c.id] = false;
    for (const auto& s : org.synapses) {
        referenced[s.from_cell_id] = true;
        referenced[s.to_cell_id] = true;
    }
    org.cells.erase(
        std::remove_if(org.cells.begin(), org.cells.end(),
                       [&](const Cell& c) { return !referenced[c.id]; }),
        org.cells.end());
    // 重编号: 保证 id 连续且与位置一致 (legacy 突触按编译索引, 新核心按稳定 ID)
    std::map<uint32_t, uint32_t> remap;
    for (size_t i = 0; i < org.cells.size(); ++i) {
        remap[org.cells[i].id] = static_cast<uint32_t>(i);
        org.cells[i].id = static_cast<uint32_t>(i);
    }
    for (auto& s : org.synapses) {
        s.from_cell_id = remap[s.from_cell_id];
        s.to_cell_id = remap[s.to_cell_id];
    }
    org.compile();
    return org;
}

void reconcile_organism(const CellularOrganism& src, size_t in_dim, int episodes) {
    auto imported = kun::migration::import_execution_snapshot(
        src, core::GraphIdentity(1), core::GraphRevision(1));
    if (!imported.ok()) {
        printf("[错误] 快照导入被拒: %s\n",
               imported.error ? imported.error->diagnostic().c_str() : "?");
        assert(false && "snapshot import rejected");
    }
    const auto& snap = *imported.snapshot;
    assert(snap.compiled_plan()->cells().size() == src.cells.size());
    assert(snap.cell_states().size() == src.cells.size());

    std::vector<core::RuntimeCellState> cells;
    cells.reserve(snap.cell_states().size());
    for (const auto& s : snap.cell_states()) {
        core::RuntimeCellState rc;
        rc.cell = s.cell;
        rc.type = s.type;
        rc.state_val = s.state_val;
        rc.aux_state = s.aux_state;
        rc.prev_input = s.prev_input;
        rc.output_val = s.output_val;
        rc.prev_output_val = s.prev_output_val;
        rc.delay_buffer = s.delay_buffer;
        rc.delay_idx = s.delay_idx;
        rc.latch_state = s.latch_state;
        rc.activation_count = s.activation_count;
        rc.initialized = true;
        cells.push_back(rc);
    }
    auto created = core::RuntimeState::from_imported_state(
        snap.compiled_plan(), snap.current_parameter_values(), cells, 0);
    assert(created.ok());

    core::ReferenceExecutor ref;
    auto compiled = core::CompiledExecutor::prepare(snap.compiled_plan());
    assert(compiled.ok());

    CellularOrganism org = src;  // 副本执行 (快照本身不可变)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int ep = 0; ep < episodes; ++ep) {
        std::vector<std::vector<double>> inputs(8);
        for (auto& v : inputs) {
            v.resize(in_dim);
            for (auto& x : v) x = uni(rng);
        }
        const double d_ref = reconcile_episode(
            org, *created.runtime,
            [&](core::RuntimeState& rt, const std::vector<double>& in) {
                return ref.step(rt, in);
            },
            inputs);
        const double d_cmp = reconcile_episode(
            org, *created.runtime,
            [&](core::RuntimeState& rt, const std::vector<double>& in) {
                return compiled.executor->step(rt, in);
            },
            inputs);
        printf("  episode %d: ref 最大差 %.3g | compiled 最大差 %.3g\n", ep, d_ref, d_cmp);
        assert(d_ref < 1e-9 && d_cmp < 1e-9);
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // 1. 创世种子 (EMA/迟滞门控/效应, 含内部状态)
    {
        printf("[创世种子 (剪枝孤儿)]\n");
        auto org = prune_orphans(CellularOrganism::create_handcrafted_progenitor(1));
        assert(!org.cells.empty());
        reconcile_organism(org, 4, 3);
    }
    // 2. 最小随机图 × 3 种子 (覆盖更多细胞类型与延迟缓冲)
    for (uint32_t seed : {42u, 7u, 2026u}) {
        printf("[最小随机图 seed=%u]\n", seed);
        auto raw = filter_contract_cells(
            CellularOrganism::create_minimal_random_graph(1, seed));
        printf("  [filter 后] cells=%zu syn=%zu\n", raw.cells.size(), raw.synapses.size());
        auto org = prune_orphans(prune_wild_ports(raw));
        assert(!org.cells.empty());
        reconcile_organism(org, 4, 3);
    }
    // 3. 状态丰富定制图: 延迟环缓冲 + 锁存 + 积分 + 最大最小 (冷迁移状态搬运重点覆盖)
    {
        printf("[状态丰富定制图]\n");
        CellularOrganism org;
        org.organism_id = 9;
        org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0));
        org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_1));
        org.cells.push_back(make_cell(2, CellType::OP_DELAY_N, 0.3, 0.0));   // 16 槽环缓冲
        org.cells.push_back(make_cell(3, CellType::OP_INTEGRAL, 0.05, 0.0)); // 积分状态
        org.cells.push_back(make_cell(4, CellType::OP_EMA, 0.15, 0.0));
        org.cells.push_back(make_cell(5, CellType::GATE_HYSTERESIS, 0.01, -0.01));
        org.cells.push_back(make_cell(6, CellType::OP_DIFF));
        org.cells.push_back(make_cell(7, CellType::GATE_MIN_MAX));
        org.cells.push_back(make_cell(8, CellType::ACT_PRIMARY_POSITIVE));
        org.cells.push_back(make_cell(9, CellType::ACT_PRIMARY_NEGATIVE));
        org.synapses.push_back(make_synapse(0, 2, 0, 1.0));
        org.synapses.push_back(make_synapse(2, 3, 0, 0.8));
        org.synapses.push_back(make_synapse(1, 4, 0, 1.2));
        org.synapses.push_back(make_synapse(4, 5, 0, 1.0));
        org.synapses.push_back(make_synapse(3, 6, 0, 1.0));
        org.synapses.push_back(make_synapse(2, 6, 0, -0.5));  // OP_DIFF 单端口, 双入同端口
        org.synapses.push_back(make_synapse(6, 7, 0, 1.0));
        org.synapses.push_back(make_synapse(5, 7, 1, 1.0));
        org.synapses.push_back(make_synapse(7, 8, 0, 1.0));
        org.synapses.push_back(make_synapse(7, 9, 0, -1.0));
        org.compile();
        assert(!org.cells.empty());
        reconcile_organism(org, 4, 3);
    }
    printf("[U1 双执行器对账] 全部通过: legacy 与新核心位级等价\n");
    return 0;
}

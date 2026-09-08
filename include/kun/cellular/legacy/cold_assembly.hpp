// 冷装配缝 (Cold Assembly Seam): legacy 存量组织体 → 新核心默认全装配。
// 这是 organism_adapter (出生模板) 与 execution_snapshot (live 状态) 的最终合流:
// 一个调用完成 快照导入 → Germline (live 参数种子) → Phenotype (runtime+ledger+
// lifecycle+executor 默认装配)。此后任何组织体的装配都是默认行为, 不再手接线。
#ifndef KUN_CELLULAR_LEGACY_COLD_ASSEMBLY_HPP_
#define KUN_CELLULAR_LEGACY_COLD_ASSEMBLY_HPP_

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/heredity.hpp"

#include <string>
#include <vector>

namespace kun::migration {

struct ColdAssemblyConfig {
    uint64_t organism_id{0};
    uint64_t rng_seed{0};
    // 资源默认: 每细胞初始能量/容量 (默认行为 = 生命活力充裕)
    double initial_energy{100.0};
    double initial_capacity{100.0};
    double environmental_resource{1e9};
    core::LifecycleConfig lifecycle_config{};
    core::ResourceLedgerConfig resource_config{};
    core::GrowthConfig growth_config{};  // 出生本能: 生长代价随出生声明
};

// live 参数 → 出生种子 (与 transfer::parameter_seeds 同形; 置于 core 侧避免
// 底座 include 任务层 —— 分层纪律: include/kun 不得反向依赖 tasks/)
inline core::InitialParameterSeeds cold_parameter_seeds(
    std::span<const core::InitialParameterValue> values) {
    core::InitialParameterSeeds seeds;
    for (const auto& p : values) {
        if (p.binding.kind == core::ParameterBindingKind::CellParameter)
            seeds.cell_parameters.push_back(
                {p.binding.cell, p.binding.slot, p.value});
        else
            seeds.edge_weights.push_back(
                {p.binding.edge,
                 std::get<core::ContinuousValue>(p.value).value});
    }
    return seeds;
}

struct ColdAssemblyResult {
    std::shared_ptr<const core::Germline> germline;
    std::unique_ptr<core::Phenotype> phenotype;
    std::string diagnostic;

    bool ok() const { return germline != nullptr && phenotype != nullptr; }
    explicit operator bool() const { return ok(); }
};

// 冠军/存量组织体的默认装配: 快照(含 live 权重) → germline → phenotype 全生态。
// 同化语义: 快照的 live 参数即为先天种子 (显式冷边界同化, 位级保真)。
inline ColdAssemblyResult assemble_phenotype(
    const CellularOrganism& organism,
    core::GraphIdentity identity,
    core::GraphRevision revision,
    const ColdAssemblyConfig& config = {}) {
    ColdAssemblyResult result;

    auto imported = import_execution_snapshot(organism, identity, revision);
    if (!imported.ok()) {
        result.diagnostic = imported.error
            ? imported.error->diagnostic() : "snapshot import failed";
        return result;
    }
    const auto& snap = *imported.snapshot;

    auto germline = core::Germline::create(
        snap.graph_definition(),
        cold_parameter_seeds(snap.initial_parameter_values()->entries()),
        "cold-assembly");
    if (!germline.ok()) {
        result.diagnostic = germline.error ? germline.error->reason : "germline create failed";
        return result;
    }
    result.germline = germline.germline;

    core::OffspringSpec spec;
    spec.organism_id = config.organism_id;
    spec.rng_seed = config.rng_seed;
    spec.lifecycle_config = config.lifecycle_config;
    spec.resource_config = config.resource_config;
    spec.growth_config = config.growth_config;
    for (const auto& c : germline.germline->graph()->cells()) {
        spec.resource_cells.push_back(core::ResourceCellInitial{
            c.id, core::ResourceCompartmentId{0},
            config.initial_energy, config.initial_capacity, 0.0});
    }
    spec.resource_compartments.push_back(core::ResourceCompartmentInitial{
        core::ResourceCompartmentId{0}, config.environmental_resource});

    auto phenotype = core::Phenotype::create(germline.germline, spec);
    if (!phenotype.ok()) {
        result.diagnostic = phenotype.error ? phenotype.error->reason
                                            : "phenotype create failed";
        result.germline = nullptr;
        return result;
    }
    result.phenotype = std::move(phenotype.phenotype);
    return result;
}

}  // namespace kun::migration

#endif  // KUN_CELLULAR_LEGACY_COLD_ASSEMBLY_HPP_

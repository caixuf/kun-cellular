#pragma once

#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/growth_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace kun::core {

class Phenotype;

struct GermlineVersion {
    uint64_t value{1};
    friend constexpr bool operator==(GermlineVersion, GermlineVersion) = default;
};

enum class HeredityErrorCode : uint8_t {
    InvalidGermline,
    InvalidOffspring,
    UnsupportedAssimilation,
    UndeclaredAssimilation,
    BindingMismatch,
    StaleLearningWindow,
    InvalidLearningWindow,
};

struct HeredityError {
    HeredityErrorCode code{HeredityErrorCode::InvalidGermline};
    std::string reason;
};

struct GermlineResult {
    std::shared_ptr<const class Germline> germline;
    std::optional<HeredityError> error;

    bool ok() const { return germline != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct PhenotypeResult {
    std::unique_ptr<Phenotype> phenotype;
    std::optional<HeredityError> error;

    bool ok() const { return phenotype != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct OffspringSpec {
    uint64_t organism_id{0};
    uint64_t rng_seed{0};
    LifecycleConfig lifecycle_config{};
    ResourceLedgerConfig resource_config{};
    std::vector<ResourceCellInitial> resource_cells;
    std::vector<ResourceCompartmentInitial> resource_compartments;
    // 生长是出生本能: 建造代价随出生声明 (默认零代价 = 开放生长)
    GrowthConfig growth_config{};
};

enum class AssimilationField : uint8_t {
    LiveParameters = 0,
    RuntimeMemory = 1,
    Resources = 2,
    OptimizerState = 3,
    LearningTape = 4,
};

enum class AssimilationFieldMask : uint8_t {
    None = 0,
    LiveParameters = 1u << 0,
};

struct AssimilationReport {
    GermlineVersion version_before{};
    GermlineVersion version_after{};
    AssimilationFieldMask copied_fields{AssimilationFieldMask::None};
    std::size_t copied_values{0};
    std::size_t total_values{0};
};

struct AssimilationResult {
    std::shared_ptr<const class Germline> germline;
    AssimilationReport report{};
    std::optional<HeredityError> error;

    bool ok() const { return germline != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class Germline final {
public:
    static GermlineResult create(
        GraphDefinition definition,
        InitialParameterSeeds seeds,
        std::string development_rules = {}) {
        const auto compiled = GraphCompiler{}.compile(definition, seeds);
        if (!compiled.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidGermline,
                compiled.error ? compiled.error->diagnostic()
                               : "germline graph compilation failed"}};
        }
        auto result = std::shared_ptr<const Germline>(
            new Germline(
                std::move(definition),
                std::move(development_rules),
                GermlineVersion{1},
                compiled.graph,
                compiled.initial_values));
        return {std::move(result), std::nullopt};
    }

    const GraphDefinition& definition() const { return definition_; }
    const std::string& development_rules() const { return development_rules_; }
    GermlineVersion version() const { return version_; }
    const std::shared_ptr<const CompiledGraph>& graph() const { return graph_; }
    const InitialParameterValues& initial_values() const {
        return *initial_values_;
    }

    PhenotypeResult spawn_offspring(const OffspringSpec& spec) const;

    AssimilationResult assimilate(
        const Phenotype& phenotype,
        std::span<const AssimilationField> declared) const;

private:
    Germline(
        GraphDefinition definition,
        std::string development_rules,
        GermlineVersion version,
        std::shared_ptr<const CompiledGraph> graph,
        std::shared_ptr<const InitialParameterValues> initial_values)
        : definition_(std::move(definition)),
          development_rules_(std::move(development_rules)),
          version_(version),
          graph_(std::move(graph)),
          initial_values_(std::move(initial_values)) {}

    GraphDefinition definition_;
    std::string development_rules_;
    GermlineVersion version_{};
    std::shared_ptr<const CompiledGraph> graph_;
    std::shared_ptr<const InitialParameterValues> initial_values_;

    friend class Phenotype;
    friend class LifecyclePersistence;
};

class Phenotype final {
public:
    Phenotype(Phenotype&&) = default;
    Phenotype& operator=(Phenotype&&) = default;
    Phenotype(const Phenotype&) = delete;
    Phenotype& operator=(const Phenotype&) = delete;

    static PhenotypeResult create(
        std::shared_ptr<const Germline> germline,
        OffspringSpec spec) {
        if (!germline || !germline->graph() || !germline->initial_values_) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                "phenotype requires an immutable germline template"}};
        }
        auto runtime_result = RuntimeState::create(
            germline->graph(), germline->initial_values_);
        if (!runtime_result.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                runtime_result.error ? runtime_result.error->reason
                                      : "offspring runtime creation failed"}};
        }
        auto ledger = ResourceLedger::attach(
            *runtime_result.runtime,
            std::move(spec.resource_config),
            std::span<const ResourceCellInitial>(
                spec.resource_cells.data(), spec.resource_cells.size()),
            std::span<const ResourceCompartmentInitial>(
                spec.resource_compartments.data(),
                spec.resource_compartments.size()));
        if (!ledger.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                ledger.error ? ledger.error->reason
                             : "offspring resource ledger creation failed"}};
        }
        auto prepared = CompiledExecutor::prepare(runtime_result.runtime->plan());
        if (!prepared.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                prepared.error ? std::string(prepared.error->reason)
                                : "offspring executor preparation failed"}};
        }
        auto executor = prepared.executor;
        auto lifecycle = CellularLifecycleController::create(
            *runtime_result.runtime,
            std::move(prepared.executor),
            std::move(ledger.ledger),
            spec.lifecycle_config);
        if (!lifecycle.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                lifecycle.error ? lifecycle.error->reason
                                 : "offspring lifecycle creation failed"}};
        }
        auto growth = CellularGrowthController::create(
            *lifecycle.controller, spec.growth_config);
        if (!growth.ok()) {
            return {nullptr, HeredityError{
                HeredityErrorCode::InvalidOffspring,
                growth.error ? growth.error->reason
                             : "offspring growth instinct creation failed"}};
        }
        auto result = std::unique_ptr<Phenotype>(new Phenotype(
            std::move(germline),
            std::move(runtime_result.runtime),
            std::move(executor),
            std::move(lifecycle.controller),
            std::move(growth.controller),
            spec.organism_id,
            spec.rng_seed));
        return {std::move(result), std::nullopt};
    }

    LifecycleStepResult step(
        std::span<const double> inputs,
        std::span<const ResourceSupply> supplies = {}) {
        return lifecycle_->step(inputs, supplies);
    }

    LifecycleStepResult step(
        std::span<const double> inputs,
        std::initializer_list<ResourceSupply> supplies) {
        return lifecycle_->step(inputs, supplies);
    }

    RuntimeState& runtime() { return *runtime_; }
    const RuntimeState& runtime() const { return *runtime_; }
    const ResourceLedger& ledger() const { return lifecycle_->ledger(); }
    CellularLifecycleController& lifecycle() { return *lifecycle_; }
    const CellularLifecycleController& lifecycle() const { return *lifecycle_; }
    const Germline& germline() const { return *germline_; }
    uint64_t organism_id() const { return organism_id_; }
    uint64_t next_random() { return rng_(); }

    // 出生本能: 执行/代谢/生命周期/生长全部内生, 调用方零接线。
    core::CompiledExecutor& executor() { return *executor_; }
    const core::CompiledExecutor& executor() const { return *executor_; }
    core::CellularGrowthController& growth() { return *growth_; }
    const core::CellularGrowthController& growth() const { return *growth_; }

private:
    Phenotype(
        std::shared_ptr<const Germline> germline,
        std::shared_ptr<RuntimeState> runtime,
        std::shared_ptr<core::CompiledExecutor> executor,
        std::unique_ptr<CellularLifecycleController> lifecycle,
        std::unique_ptr<CellularGrowthController> growth,
        uint64_t organism_id,
        uint64_t rng_seed)
        : germline_(std::move(germline)),
          runtime_(std::move(runtime)),
          executor_(std::move(executor)),
          lifecycle_(std::move(lifecycle)),
          growth_(std::move(growth)),
          organism_id_(organism_id),
          rng_(rng_seed) {}

    std::shared_ptr<const Germline> germline_;
    std::shared_ptr<RuntimeState> runtime_;
    std::shared_ptr<core::CompiledExecutor> executor_;
    std::unique_ptr<CellularLifecycleController> lifecycle_;
    std::unique_ptr<CellularGrowthController> growth_;
    uint64_t organism_id_{0};
    std::mt19937_64 rng_;

    friend class Germline;
    friend class LifecyclePersistence;
};

enum class LearningWindowErrorCode : uint8_t {
    InvalidBinding,
    StaleGraph,
    Consumed,
    ParameterNotAllowed,
};

struct LearningWindowError {
    LearningWindowErrorCode code{LearningWindowErrorCode::InvalidBinding};
    std::string reason;
};

struct LearningWindowResult {
    std::optional<LearningWindowError> error;
    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct LearningWindowResetReport {
    uint64_t reset_count{0};
    bool optimizer_state_reset{true};
};

struct LearningWindowResetResult {
    LearningWindowResetReport report{};
    std::optional<LearningWindowError> error;
    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct LearningGradient {
    ParameterBinding binding{};
    double value{0.0};
};

struct LearningUpdateReport {
    std::size_t updated_values{0};
    double learning_rate{0.0};
};

struct LearningUpdateResult {
    LearningUpdateReport report{};
    std::optional<LearningWindowError> error;
    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class LearningWindow final {
public:
    static LearningWindow open(
        uint64_t organism_id,
        const RuntimeState& runtime,
        std::span<const ParameterBinding> allowed) {
        LearningWindow result;
        result.organism_id_ = organism_id;
        result.identity_ = runtime.identity();
        result.revision_ = runtime.revision();
        result.semantic_version_ = runtime.plan()->semantic_version();
        result.allowed_.assign(allowed.begin(), allowed.end());
        return result;
    }

    static LearningWindow open(
        uint64_t organism_id,
        const RuntimeState& runtime,
        std::initializer_list<ParameterBinding> allowed) {
        return open(
            organism_id,
            runtime,
            std::span<const ParameterBinding>(
                allowed.begin(), allowed.size()));
    }

    LearningWindowResult validate(const RuntimeState& runtime) const {
        if (consumed_) {
            return {LearningWindowError{
                LearningWindowErrorCode::Consumed,
                "learning window has been consumed"}};
        }
        if (runtime.identity() != identity_ ||
            runtime.revision() != revision_ ||
            runtime.plan()->semantic_version() != semantic_version_) {
            return {LearningWindowError{
                LearningWindowErrorCode::StaleGraph,
                "learning window graph identity, revision, or semantic version is stale"}};
        }
        return {};
    }

    LearningWindowResult validate_parameters(
        const RuntimeState& runtime,
        std::span<const ParameterBinding> bindings) const {
        if (const auto result = validate(runtime); !result.ok()) return result;
        for (const auto& binding : bindings) {
            const auto found = std::find_if(
                allowed_.begin(), allowed_.end(),
                [&](const auto& allowed) {
                    return allowed.kind == binding.kind &&
                           allowed.index == binding.index &&
                           allowed.cell == binding.cell &&
                           allowed.edge == binding.edge &&
                           allowed.slot == binding.slot;
                });
            if (found == allowed_.end()) {
                return {LearningWindowError{
                    LearningWindowErrorCode::ParameterNotAllowed,
                    "gradient parameter is outside the declared learning set"}};
            }
        }
        return {};
    }

    LearningUpdateResult apply_sgd(
        RuntimeState& runtime,
        std::span<const LearningGradient> gradients,
        double learning_rate) {
        if (const auto result = validate(runtime); !result.ok()) {
            return {{}, result.error};
        }
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0) {
            return {{}, LearningWindowError{
                LearningWindowErrorCode::InvalidBinding,
                "learning rate must be finite and positive"}};
        }
        std::vector<ParameterBinding> bindings;
        bindings.reserve(gradients.size());
        for (const auto& gradient : gradients) {
            if (!std::isfinite(gradient.value)) {
                return {{}, LearningWindowError{
                    LearningWindowErrorCode::InvalidBinding,
                    "learning gradient must be finite"}};
            }
            bindings.push_back(gradient.binding);
        }
        if (const auto result = validate_parameters(runtime, bindings);
            !result.ok()) {
            return {{}, result.error};
        }
        for (const auto& gradient : gradients) {
            const auto current = runtime.parameter_at(gradient.binding.index);
            if (!current || !std::holds_alternative<ContinuousValue>(*current)) {
                return {{}, LearningWindowError{
                    LearningWindowErrorCode::ParameterNotAllowed,
                    "learning gradient targets a non-continuous parameter"}};
            }
            const double updated =
                std::get<ContinuousValue>(*current).value -
                learning_rate * gradient.value;
            if (!std::isfinite(updated)) {
                return {{}, LearningWindowError{
                    LearningWindowErrorCode::InvalidBinding,
                    "learning update produced a non-finite parameter"}};
            }
            if (const auto result = runtime.set_parameter(
                    gradient.binding,
                    ParameterValue{ContinuousValue{updated}});
                !result.ok()) {
                return {{}, LearningWindowError{
                    LearningWindowErrorCode::InvalidBinding,
                    result.error ? result.error->reason
                                 : "runtime rejected learning update"}};
            }
        }
        return {LearningUpdateReport{gradients.size(), learning_rate}, std::nullopt};
    }

    LearningWindowResetResult consume_after_graph_edit(
        const RuntimeState& runtime) {
        if (runtime.identity() != identity_) {
            return {{}, LearningWindowError{
                LearningWindowErrorCode::StaleGraph,
                "graph edit crossed organism graph identity"}};
        }
        consumed_ = true;
        ++reset_count_;
        revision_ = runtime.revision();
        semantic_version_ = runtime.plan()->semantic_version();
        return {LearningWindowResetReport{reset_count_, true}, std::nullopt};
    }

    void invalidate_after_checkpoint_restore() {
        consumed_ = true;
        ++reset_count_;
    }

    bool consumed() const { return consumed_; }
    uint64_t reset_count() const { return reset_count_; }
    uint64_t organism_id() const { return organism_id_; }
    GraphIdentity identity() const { return identity_; }
    GraphRevision revision() const { return revision_; }
    uint32_t semantic_version() const { return semantic_version_; }
    std::span<const ParameterBinding> allowed_parameters() const {
        return allowed_;
    }

private:
    uint64_t organism_id_{0};
    GraphIdentity identity_{};
    GraphRevision revision_{};
    uint32_t semantic_version_{0};
    std::vector<ParameterBinding> allowed_;
    bool consumed_{false};
    uint64_t reset_count_{0};
};

inline AssimilationResult Germline::assimilate(
    const Phenotype& phenotype,
    std::span<const AssimilationField> declared) const {
    bool live_parameters = false;
    AssimilationFieldMask mask = AssimilationFieldMask::None;
    for (const auto field : declared) {
        if (field != AssimilationField::LiveParameters) {
            return {nullptr, {}, HeredityError{
                HeredityErrorCode::UndeclaredAssimilation,
                "runtime memory, resources, optimizer, and tape are not assimilable"}};
        }
        live_parameters = true;
        mask = AssimilationFieldMask::LiveParameters;
    }
    if (!live_parameters) {
        return {nullptr, {}, HeredityError{
            HeredityErrorCode::UnsupportedAssimilation,
            "assimilation requires an explicitly declared field"}};
    }
    if (!phenotype.runtime().bound_to(*graph_)) {
        return {nullptr, {}, HeredityError{
            HeredityErrorCode::BindingMismatch,
            "live-parameter assimilation cannot implicitly assimilate postnatal topology"}};
    }
    std::vector<InitialParameterValue> values(
        initial_values_->entries().begin(),
        initial_values_->entries().end());
    if (values.size() != phenotype.runtime().parameters().size()) {
        return {nullptr, {}, HeredityError{
            HeredityErrorCode::BindingMismatch,
            "phenotype parameter store does not match the germline template"}};
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto& source = phenotype.runtime().parameters()[i];
        const auto& target = values[i];
        if (source.binding.kind != target.binding.kind ||
            source.binding.index != target.binding.index ||
            source.binding.cell != target.binding.cell ||
            source.binding.edge != target.binding.edge ||
            source.binding.slot != target.binding.slot) {
            return {nullptr, {}, HeredityError{
                HeredityErrorCode::BindingMismatch,
                "live-parameter assimilation binding identity differs"}};
        }
        values[i].value = source.value;
    }
    auto result = std::shared_ptr<const Germline>(
        new Germline(
            definition_,
            development_rules_,
            GermlineVersion{version_.value + 1},
            graph_,
            std::make_shared<const InitialParameterValues>(
                std::move(values))));
    return {
        std::move(result),
        AssimilationReport{
            version_,
            GermlineVersion{version_.value + 1},
            mask,
            phenotype.runtime().parameters().size(),
            phenotype.runtime().parameters().size()},
        std::nullopt};
}

inline PhenotypeResult Germline::spawn_offspring(
    const OffspringSpec& spec) const {
    auto owned = std::shared_ptr<const Germline>(
        new Germline(
            definition_,
            development_rules_,
            version_,
            graph_,
            initial_values_));
    return Phenotype::create(std::move(owned), spec);
}

}  // namespace kun::core

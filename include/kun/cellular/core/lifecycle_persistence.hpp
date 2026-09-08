#pragma once

#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/heredity.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace kun::core {

enum class CheckpointMode : uint8_t {
    NoOptimizer = 0,
    WithOptimizerState = 1,
};

enum class PersistenceErrorCode : uint8_t {
    InvalidInput,
    UnsupportedMode,
    BadMagic,
    UnsupportedVersion,
    Truncated,
    Corrupt,
    SchemaMismatch,
    RuntimeRestore,
    ResourceRestore,
    LifecycleRestore,
};

struct PersistenceError {
    PersistenceErrorCode code{PersistenceErrorCode::InvalidInput};
    std::string reason;
};

struct PersistenceResult {
    std::vector<uint8_t> bytes;
    std::optional<PersistenceError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

namespace persistence_detail {

class Writer final {
public:
    template <typename T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto offset = bytes_.size();
        bytes_.resize(offset + sizeof(T));
        std::memcpy(bytes_.data() + offset, &value, sizeof(T));
    }

    void raw(const void* data, std::size_t size) {
        const auto offset = bytes_.size();
        bytes_.resize(offset + size);
        if (size != 0) std::memcpy(bytes_.data() + offset, data, size);
    }

    void string(const std::string& value) {
        pod<uint64_t>(value.size());
        raw(value.data(), value.size());
    }

    std::vector<uint8_t> take() { return std::move(bytes_); }

private:
    std::vector<uint8_t> bytes_;
};

class Reader final {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    template <typename T>
    std::optional<T> pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (offset_ + sizeof(T) > bytes_.size()) {
            ok_ = false;
            return std::nullopt;
        }
        T result{};
        std::memcpy(&result, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return result;
    }

    std::optional<std::string> string() {
        const auto size = pod<uint64_t>();
        if (!size.has_value() || *size > bytes_.size() - offset_) {
            ok_ = false;
            return std::nullopt;
        }
        std::string result(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            static_cast<std::size_t>(*size));
        offset_ += static_cast<std::size_t>(*size);
        return result;
    }

    bool done() const { return ok_ && offset_ == bytes_.size(); }
    bool ok() const { return ok_; }
    // Count checks are derived solely from the remaining payload. They are
    // transport-safety guards, not graph-size limits or execution caps.
    std::size_t remaining() const {
        return offset_ <= bytes_.size() ? bytes_.size() - offset_ : 0;
    }

private:
    std::span<const uint8_t> bytes_;
    std::size_t offset_{0};
    bool ok_{true};
};

inline PersistenceError failure(PersistenceErrorCode code, std::string reason) {
    return {code, std::move(reason)};
}

inline uint64_t checksum(std::span<const uint8_t> bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const uint8_t byte : bytes) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

inline void write_parameter_value(Writer& writer, const ParameterValue& value) {
    writer.pod<uint8_t>(static_cast<uint8_t>(value.index()));
    std::visit(
        [&](const auto& item) {
            using Value = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Value, ContinuousValue>) {
                writer.pod<double>(item.value);
            } else if constexpr (std::is_same_v<Value, ChannelIndex>) {
                writer.pod<uint64_t>(item.value);
            } else if constexpr (std::is_same_v<Value, DelayTicks>) {
                writer.pod<uint64_t>(item.value);
            } else if constexpr (std::is_same_v<Value, MinMaxMode>) {
                writer.pod<uint8_t>(static_cast<uint8_t>(item));
            }
        },
        value);
}

inline std::optional<ParameterValue> read_parameter_value(Reader& reader) {
    const auto index = reader.pod<uint8_t>();
    if (!index.has_value()) return std::nullopt;
    switch (*index) {
        case 0: {
            const auto value = reader.pod<double>();
            return value.has_value()
                ? std::optional<ParameterValue>(ContinuousValue{*value})
                : std::nullopt;
        }
        case 1: {
            const auto value = reader.pod<uint64_t>();
            return value.has_value()
                ? std::optional<ParameterValue>(
                      ChannelIndex{static_cast<std::size_t>(*value)})
                : std::nullopt;
        }
        case 2: {
            const auto value = reader.pod<uint64_t>();
            return value.has_value()
                ? std::optional<ParameterValue>(DelayTicks{*value})
                : std::nullopt;
        }
        case 3: {
            const auto value = reader.pod<uint8_t>();
            if (!value.has_value() || *value > 1) return std::nullopt;
            return ParameterValue{static_cast<MinMaxMode>(*value)};
        }
        case 4:
            return ParameterValue{UnusedParameter{}};
        default:
            return std::nullopt;
    }
}

inline void write_graph(Writer& writer, const GraphDefinition& graph) {
    writer.pod<uint64_t>(graph.identity.value);
    writer.pod<uint64_t>(graph.revision.value);
    writer.pod<uint8_t>(static_cast<uint8_t>(graph.profile));
    writer.pod<uint32_t>(graph.semantic_version);
    writer.pod<uint64_t>(graph.cells.size());
    for (const auto& cell : graph.cells) {
        writer.pod<uint64_t>(cell.id.value);
        writer.pod<uint8_t>(static_cast<uint8_t>(cell.type));
    }
    writer.pod<uint64_t>(graph.edges.size());
    for (const auto& edge : graph.edges) {
        writer.pod<uint64_t>(edge.id.value);
        writer.pod<uint64_t>(edge.source.value);
        writer.pod<uint32_t>(edge.source_port.value);
        writer.pod<uint64_t>(edge.target.value);
        writer.pod<uint32_t>(edge.target_port.value);
        writer.pod<uint8_t>(static_cast<uint8_t>(edge.delay));
    }
}

inline std::optional<GraphDefinition> read_graph(Reader& reader) {
    const auto identity = reader.pod<uint64_t>();
    const auto revision = reader.pod<uint64_t>();
    const auto profile = reader.pod<uint8_t>();
    const auto semantic = reader.pod<uint32_t>();
    const auto cell_count = reader.pod<uint64_t>();
    if (!identity || !revision || !profile || !semantic || !cell_count ||
        *cell_count > reader.remaining()) {
        return std::nullopt;
    }
    GraphDefinition graph{
        GraphIdentity{*identity},
        GraphRevision{*revision},
        static_cast<SemanticProfile>(*profile),
        *semantic,
        {},
        {}};
    graph.cells.reserve(static_cast<std::size_t>(*cell_count));
    for (uint64_t i = 0; i < *cell_count; ++i) {
        const auto id = reader.pod<uint64_t>();
        const auto type = reader.pod<uint8_t>();
        if (!id || !type) return std::nullopt;
        graph.cells.push_back(
            CellDefinition{CellId{*id}, static_cast<CellType>(*type)});
    }
    const auto edge_count = reader.pod<uint64_t>();
    if (!edge_count || *edge_count > reader.remaining()) return std::nullopt;
    graph.edges.reserve(static_cast<std::size_t>(*edge_count));
    for (uint64_t i = 0; i < *edge_count; ++i) {
        const auto id = reader.pod<uint64_t>();
        const auto source = reader.pod<uint64_t>();
        const auto source_port = reader.pod<uint32_t>();
        const auto target = reader.pod<uint64_t>();
        const auto target_port = reader.pod<uint32_t>();
        const auto delay = reader.pod<uint8_t>();
        if (!id || !source || !source_port || !target || !target_port || !delay) {
            return std::nullopt;
        }
        graph.edges.push_back(
            EdgeDefinition{
                EdgeId{*id}, CellId{*source}, OutputPort{*source_port},
                CellId{*target}, InputPort{*target_port},
                static_cast<EdgeDelay>(*delay)});
    }
    return graph;
}

inline void write_parameters(
    Writer& writer,
    std::span<const InitialParameterValue> values) {
    writer.pod<uint64_t>(values.size());
    for (const auto& value : values) {
        writer.pod<uint8_t>(static_cast<uint8_t>(value.binding.kind));
        writer.pod<uint64_t>(value.binding.index);
        writer.pod<uint64_t>(value.binding.cell.value);
        writer.pod<uint64_t>(value.binding.edge.value);
        writer.pod<uint8_t>(static_cast<uint8_t>(value.binding.slot));
        write_parameter_value(writer, value.value);
    }
}

inline std::optional<std::vector<InitialParameterValue>> read_parameters(
    Reader& reader) {
    const auto count = reader.pod<uint64_t>();
    if (!count || *count > reader.remaining()) return std::nullopt;
    std::vector<InitialParameterValue> result;
    result.reserve(static_cast<std::size_t>(*count));
    for (uint64_t i = 0; i < *count; ++i) {
        const auto kind = reader.pod<uint8_t>();
        const auto index = reader.pod<uint64_t>();
        const auto cell = reader.pod<uint64_t>();
        const auto edge = reader.pod<uint64_t>();
        const auto slot = reader.pod<uint8_t>();
        const auto value = read_parameter_value(reader);
        if (!kind || !index || !cell || !edge || !slot || !value) {
            return std::nullopt;
        }
        result.push_back(
            InitialParameterValue{
                ParameterBinding{
                    static_cast<ParameterBindingKind>(*kind),
                    static_cast<std::size_t>(*index),
                    CellId{*cell},
                    EdgeId{*edge},
                    static_cast<ParameterSlot>(*slot)},
                *value});
    }
    return result;
}

inline InitialParameterSeeds seeds_from_parameters(
    std::span<const InitialParameterValue> values) {
    InitialParameterSeeds seeds;
    for (const auto& value : values) {
        if (value.binding.kind == ParameterBindingKind::CellParameter) {
            seeds.cell_parameters.push_back(
                CellParameterSeed{
                    value.binding.cell, value.binding.slot, value.value});
        } else {
            if (std::holds_alternative<ContinuousValue>(value.value)) {
                seeds.edge_weights.push_back(
                    EdgeParameterSeed{
                        value.binding.edge,
                        std::get<ContinuousValue>(value.value).value});
            }
        }
    }
    return seeds;
}

inline GraphDefinition graph_from_runtime(const RuntimeState& runtime) {
    GraphDefinition graph{
        runtime.identity(),
        runtime.revision(),
        runtime.profile(),
        runtime.plan()->semantic_version(),
        {},
        {}};
    for (const auto& cell : runtime.plan()->cells()) {
        graph.cells.push_back(CellDefinition{cell.id, cell.type});
    }
    for (const auto& edge : runtime.plan()->edges()) {
        graph.edges.push_back(
            EdgeDefinition{
                edge.id,
                runtime.plan()->cells()[edge.source_index].id,
                edge.source_port,
                runtime.plan()->cells()[edge.target_index].id,
                edge.target_port,
                edge.delay});
    }
    return graph;
}

inline void write_runtime_cells(
    Writer& writer,
    std::span<const RuntimeCellState> cells,
    uint64_t tick) {
    writer.pod<uint64_t>(tick);
    writer.pod<uint64_t>(cells.size());
    for (const auto& cell : cells) {
        writer.pod<uint64_t>(cell.cell.value);
        writer.pod<uint8_t>(static_cast<uint8_t>(cell.type));
        writer.pod<double>(cell.state_val);
        writer.pod<double>(cell.aux_state);
        writer.pod<double>(cell.prev_input);
        writer.pod<double>(cell.output_val);
        writer.pod<double>(cell.prev_output_val);
        for (const double value : cell.delay_buffer) writer.pod<double>(value);
        writer.pod<uint8_t>(cell.delay_idx);
        writer.pod<uint8_t>(cell.latch_state ? 1 : 0);
        writer.pod<uint32_t>(cell.activation_count);
        writer.pod<uint8_t>(cell.initialized ? 1 : 0);
    }
}

inline std::optional<std::pair<uint64_t, std::vector<RuntimeCellState>>>
read_runtime_cells(Reader& reader) {
    const auto tick = reader.pod<uint64_t>();
    const auto count = reader.pod<uint64_t>();
    if (!tick || !count || *count > reader.remaining()) return std::nullopt;
    std::vector<RuntimeCellState> cells;
    cells.reserve(static_cast<std::size_t>(*count));
    for (uint64_t i = 0; i < *count; ++i) {
        const auto id = reader.pod<uint64_t>();
        const auto type = reader.pod<uint8_t>();
        const auto state = reader.pod<double>();
        const auto aux = reader.pod<double>();
        const auto prev_input = reader.pod<double>();
        const auto output = reader.pod<double>();
        const auto prev_output = reader.pod<double>();
        if (!id || !type || !state || !aux || !prev_input || !output ||
            !prev_output) return std::nullopt;
        RuntimeCellState cell{
            CellId{*id}, static_cast<CellType>(*type), *state, *aux,
            *prev_input, *output, *prev_output};
        for (double& value : cell.delay_buffer) {
            const auto read = reader.pod<double>();
            if (!read) return std::nullopt;
            value = *read;
        }
        const auto delay_idx = reader.pod<uint8_t>();
        const auto latch = reader.pod<uint8_t>();
        const auto activation = reader.pod<uint32_t>();
        const auto initialized = reader.pod<uint8_t>();
        if (!delay_idx || !latch || !activation || !initialized) {
            return std::nullopt;
        }
        cell.delay_idx = *delay_idx;
        cell.latch_state = *latch != 0;
        cell.activation_count = *activation;
        cell.initialized = *initialized != 0;
        cells.push_back(cell);
    }
    return std::make_pair(*tick, std::move(cells));
}

inline void write_resource_config(
    Writer& writer,
    const ResourceLedgerConfig& config) {
    writer.pod<double>(config.dt);
    writer.pod<double>(config.maintenance_cost);
    writer.pod<double>(config.absorption_rate);
    writer.pod<double>(config.activity_scale);
    writer.pod<double>(config.activity_cost);
    writer.pod<double>(config.transmission_scale);
    writer.pod<double>(config.transmission_cost);
    writer.pod<uint64_t>(config.execution_costs.size());
    for (const auto& value : config.execution_costs) {
        writer.pod<uint8_t>(static_cast<uint8_t>(value.type));
        writer.pod<double>(value.cost);
    }
}

inline std::optional<ResourceLedgerConfig> read_resource_config(Reader& reader) {
    ResourceLedgerConfig config;
    const auto dt = reader.pod<double>();
    const auto maintenance = reader.pod<double>();
    const auto absorption = reader.pod<double>();
    const auto activity_scale = reader.pod<double>();
    const auto activity = reader.pod<double>();
    const auto transmission_scale = reader.pod<double>();
    const auto transmission = reader.pod<double>();
    const auto count = reader.pod<uint64_t>();
    if (!dt || !maintenance || !absorption || !activity_scale || !activity ||
        !transmission_scale || !transmission || !count ||
        *count > reader.remaining()) {
        return std::nullopt;
    }
    config.dt = *dt;
    config.maintenance_cost = *maintenance;
    config.absorption_rate = *absorption;
    config.activity_scale = *activity_scale;
    config.activity_cost = *activity;
    config.transmission_scale = *transmission_scale;
    config.transmission_cost = *transmission;
    for (uint64_t i = 0; i < *count; ++i) {
        const auto type = reader.pod<uint8_t>();
        const auto cost = reader.pod<double>();
        if (!type || !cost) return std::nullopt;
        config.execution_costs.push_back(
            ResourceExecutionCost{static_cast<CellType>(*type), *cost});
    }
    return config;
}

inline void write_resource_snapshot(
    Writer& writer,
    const ResourceSnapshot& snapshot) {
    writer.pod<uint64_t>(snapshot.cells().size());
    for (const auto& cell : snapshot.cells()) {
        writer.pod<uint64_t>(cell.cell.value);
        writer.pod<uint64_t>(cell.compartment.value);
        writer.pod<double>(cell.capacity);
        writer.pod<double>(cell.energy);
        writer.pod<double>(cell.structural_reserve);
        writer.pod<double>(cell.initial_structural_reserve);
        writer.pod<double>(cell.age);
        writer.pod<double>(cell.cumulative_activity);
        writer.pod<double>(cell.cumulative_transmission);
        writer.pod<double>(cell.cumulative_execution_cost);
        writer.pod<double>(cell.cumulative_maintenance_cost);
        writer.pod<double>(cell.cumulative_activity_cost);
        writer.pod<double>(cell.cumulative_transmission_cost);
        writer.pod<double>(cell.cumulative_paid_cost);
        writer.pod<double>(cell.cumulative_unpaid_cost);
        writer.pod<double>(cell.cumulative_growth_cost);
        writer.pod<uint64_t>(cell.execution_count);
        writer.pod<uint8_t>(cell.exhausted ? 1 : 0);
        writer.pod<uint8_t>(cell.active ? 1 : 0);
    }
    writer.pod<uint64_t>(snapshot.compartments().size());
    for (const auto& compartment : snapshot.compartments()) {
        writer.pod<uint64_t>(compartment.compartment.value);
        writer.pod<double>(compartment.environmental_resource);
    }
    const auto& totals = snapshot.totals();
    writer.pod<double>(totals.initial_total);
    writer.pod<double>(totals.cumulative_external_injection);
    writer.pod<double>(totals.cumulative_dissipation);
    writer.pod<double>(totals.cumulative_export);
    writer.pod<double>(totals.cumulative_paid_cost);
    writer.pod<double>(totals.cumulative_unpaid_cost);
    writer.pod<double>(totals.cumulative_growth_cost);
    writer.pod<uint64_t>(snapshot.last_measurement_tick());
    writer.pod<uint64_t>(snapshot.identity().value);
    writer.pod<uint64_t>(snapshot.revision().value);
    writer.pod<uint8_t>(static_cast<uint8_t>(snapshot.profile()));
}

inline std::optional<ResourceSnapshot> read_resource_snapshot(Reader& reader) {
    const auto cell_count = reader.pod<uint64_t>();
    if (!cell_count || *cell_count > reader.remaining()) return std::nullopt;
    std::vector<ResourceCellState> cells;
    cells.reserve(static_cast<std::size_t>(*cell_count));
    for (uint64_t i = 0; i < *cell_count; ++i) {
        ResourceCellState cell;
        const auto id = reader.pod<uint64_t>();
        const auto compartment = reader.pod<uint64_t>();
        if (!id || !compartment) return std::nullopt;
        cell.cell = CellId{*id};
        cell.compartment = ResourceCompartmentId{*compartment};
        double* doubles[] = {
            &cell.capacity, &cell.energy, &cell.structural_reserve,
            &cell.initial_structural_reserve, &cell.age,
            &cell.cumulative_activity, &cell.cumulative_transmission,
            &cell.cumulative_execution_cost,
            &cell.cumulative_maintenance_cost,
            &cell.cumulative_activity_cost,
            &cell.cumulative_transmission_cost,
            &cell.cumulative_paid_cost,
            &cell.cumulative_unpaid_cost,
            &cell.cumulative_growth_cost};
        for (double* value : doubles) {
            const auto read = reader.pod<double>();
            if (!read) return std::nullopt;
            *value = *read;
        }
        const auto executions = reader.pod<uint64_t>();
        const auto exhausted = reader.pod<uint8_t>();
        const auto active = reader.pod<uint8_t>();
        if (!executions || !exhausted || !active) return std::nullopt;
        cell.execution_count = *executions;
        cell.exhausted = *exhausted != 0;
        cell.active = *active != 0;
        cells.push_back(cell);
    }
    const auto compartment_count = reader.pod<uint64_t>();
    if (!compartment_count || *compartment_count > reader.remaining()) {
        return std::nullopt;
    }
    std::vector<ResourceCompartmentState> compartments;
    compartments.reserve(static_cast<std::size_t>(*compartment_count));
    for (uint64_t i = 0; i < *compartment_count; ++i) {
        const auto id = reader.pod<uint64_t>();
        const auto value = reader.pod<double>();
        if (!id || !value) return std::nullopt;
        compartments.push_back(
            ResourceCompartmentState{
                ResourceCompartmentId{*id}, *value});
    }
    ResourceTotals totals;
    double* total_doubles[] = {
        &totals.initial_total,
        &totals.cumulative_external_injection,
        &totals.cumulative_dissipation,
        &totals.cumulative_export,
        &totals.cumulative_paid_cost,
        &totals.cumulative_unpaid_cost,
        &totals.cumulative_growth_cost};
    for (double* value : total_doubles) {
        const auto read = reader.pod<double>();
        if (!read) return std::nullopt;
        *value = *read;
    }
    const auto tick = reader.pod<uint64_t>();
    const auto identity = reader.pod<uint64_t>();
    const auto revision = reader.pod<uint64_t>();
    const auto profile = reader.pod<uint8_t>();
    if (!tick || !identity || !revision || !profile) return std::nullopt;
    return ResourceSnapshot::from_parts(
        std::move(cells),
        std::move(compartments),
        totals,
        *tick,
        GraphIdentity{*identity},
        GraphRevision{*revision},
        static_cast<SemanticProfile>(*profile));
}

inline void write_lifecycle_config(Writer& writer, const LifecycleConfig& config) {
    writer.pod<double>(config.apoptotic_resource);
    writer.pod<double>(config.dormant_enter_resource);
    writer.pod<double>(config.dormant_exit_resource);
    writer.pod<uint64_t>(config.minimum_dwell_ticks);
    writer.pod<uint64_t>(config.cooldown_ticks);
}

inline std::optional<LifecycleConfig> read_lifecycle_config(Reader& reader) {
    const auto apoptosis = reader.pod<double>();
    const auto enter = reader.pod<double>();
    const auto exit = reader.pod<double>();
    const auto dwell = reader.pod<uint64_t>();
    const auto cooldown = reader.pod<uint64_t>();
    if (!apoptosis || !enter || !exit || !dwell || !cooldown) {
        return std::nullopt;
    }
    return LifecycleConfig{*apoptosis, *enter, *exit, *dwell, *cooldown};
}

inline void write_lifecycle_states(
    Writer& writer,
    std::span<const LifecycleCellState> states) {
    writer.pod<uint64_t>(states.size());
    for (const auto& state : states) {
        writer.pod<uint64_t>(state.cell.value);
        writer.pod<uint8_t>(static_cast<uint8_t>(state.state));
        writer.pod<uint64_t>(state.entered_tick);
        writer.pod<uint64_t>(state.cooldown_until);
    }
}

inline std::optional<std::vector<LifecycleCellState>> read_lifecycle_states(
    Reader& reader) {
    const auto count = reader.pod<uint64_t>();
    if (!count || *count > reader.remaining()) return std::nullopt;
    std::vector<LifecycleCellState> states;
    states.reserve(static_cast<std::size_t>(*count));
    for (uint64_t i = 0; i < *count; ++i) {
        const auto id = reader.pod<uint64_t>();
        const auto state = reader.pod<uint8_t>();
        const auto entered = reader.pod<uint64_t>();
        const auto cooldown = reader.pod<uint64_t>();
        if (!id || !state || !entered || !cooldown) return std::nullopt;
        states.push_back(
            LifecycleCellState{
                CellId{*id}, static_cast<LifecycleState>(*state),
                *entered, *cooldown});
    }
    return states;
}

}  // namespace persistence_detail

class LifecyclePersistence final {
public:
    // R5 exact continuation is deliberately limited to no-optimizer,
    // no-open-learning-window checkpoints.
    static PersistenceResult save(
        const Phenotype& phenotype,
        CheckpointMode mode = CheckpointMode::NoOptimizer) {
        if (mode == CheckpointMode::WithOptimizerState) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::UnsupportedMode,
                "optimizer state serialization is not supported in R5")};
        }
        persistence_detail::Writer writer;
        const char magic[] = {'K', 'U', 'N', 'R', '5', 'C', 'P', '1'};
        writer.raw(magic, sizeof(magic));
        writer.pod<uint32_t>(1);
        writer.pod<uint8_t>(static_cast<uint8_t>(mode));
        writer.pod<uint64_t>(phenotype.organism_id());
        writer.string(phenotype.germline().development_rules());
        writer.pod<uint64_t>(phenotype.germline().version().value);
        persistence_detail::write_graph(
            writer, phenotype.germline().definition());
        persistence_detail::write_parameters(
            writer, phenotype.germline().initial_values().entries());
        persistence_detail::write_graph(
            writer, persistence_detail::graph_from_runtime(phenotype.runtime()));
        persistence_detail::write_parameters(
            writer, phenotype.runtime().parameters());
        const auto runtime_snapshot = phenotype.runtime().snapshot();
        persistence_detail::write_runtime_cells(
            writer, runtime_snapshot.cells(), runtime_snapshot.tick());
        persistence_detail::write_resource_config(
            writer, phenotype.ledger().config());
        persistence_detail::write_resource_snapshot(
            writer, phenotype.ledger().snapshot());
        persistence_detail::write_lifecycle_config(
            writer, phenotype.lifecycle().config());
        persistence_detail::write_lifecycle_states(
            writer, phenotype.lifecycle().states());
        std::ostringstream rng_stream;
        rng_stream << phenotype.rng_;
        writer.string(rng_stream.str());
        auto bytes = writer.take();
        const auto hash = persistence_detail::checksum(bytes);
        const auto offset = bytes.size();
        bytes.resize(offset + sizeof(hash));
        std::memcpy(bytes.data() + offset, &hash, sizeof(hash));
        return {std::move(bytes), std::nullopt};
    }

    static PersistenceResult save(
        const Phenotype& phenotype,
        CheckpointMode mode,
        const LearningWindow* open_window) {
        if (open_window != nullptr && !open_window->consumed()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::UnsupportedMode,
                "R5 checkpoint save rejects an open learning window")};
        }
        return save(phenotype, mode);
    }

    static PersistenceResult restore_into(
        Phenotype& target,
        std::span<const uint8_t> bytes) {
        if (bytes.size() < sizeof(uint64_t)) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::Truncated,
                "checkpoint checksum is truncated")};
        }
        uint64_t stored_checksum = 0;
        std::memcpy(
            &stored_checksum,
            bytes.data() + bytes.size() - sizeof(uint64_t),
            sizeof(uint64_t));
        const auto payload = bytes.first(bytes.size() - sizeof(uint64_t));
        if (stored_checksum != persistence_detail::checksum(payload)) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::Corrupt,
                "checkpoint checksum does not match payload")};
        }
        persistence_detail::Reader reader(payload);
        const char expected_magic[] = {'K', 'U', 'N', 'R', '5', 'C', 'P', '1'};
        for (char expected : expected_magic) {
            const auto value = reader.pod<uint8_t>();
            if (!value || *value != static_cast<uint8_t>(expected)) {
                return {{}, persistence_detail::failure(
                    PersistenceErrorCode::BadMagic,
                    "checkpoint magic is invalid")};
            }
        }
        const auto version = reader.pod<uint32_t>();
        const auto mode = reader.pod<uint8_t>();
        if (!version || !mode) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::Truncated,
                "checkpoint header is truncated")};
        }
        if (*version != 1) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::UnsupportedVersion,
                "checkpoint schema version is unsupported")};
        }
        if (*mode != static_cast<uint8_t>(CheckpointMode::NoOptimizer)) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::UnsupportedMode,
                "optimizer checkpoint restore is unsupported in R5")};
        }

        const auto organism_id = reader.pod<uint64_t>();
        const auto rules = reader.string();
        const auto germline_version = reader.pod<uint64_t>();
        const auto germline_graph = persistence_detail::read_graph(reader);
        const auto germline_parameters = persistence_detail::read_parameters(reader);
        const auto phenotype_graph = persistence_detail::read_graph(reader);
        const auto phenotype_parameters = persistence_detail::read_parameters(reader);
        const auto runtime_cells = persistence_detail::read_runtime_cells(reader);
        const auto resource_config = persistence_detail::read_resource_config(reader);
        const auto resource_snapshot = persistence_detail::read_resource_snapshot(reader);
        const auto lifecycle_config = persistence_detail::read_lifecycle_config(reader);
        const auto lifecycle_states = persistence_detail::read_lifecycle_states(reader);
        const auto rng_state = reader.string();
        if (!organism_id || !rules || !germline_version ||
            !germline_graph || !germline_parameters || !phenotype_graph ||
            !phenotype_parameters || !runtime_cells || !resource_config ||
            !resource_snapshot || !lifecycle_config || !lifecycle_states ||
            !rng_state || !reader.done()) {
            return {{}, persistence_detail::failure(
                reader.ok() ? PersistenceErrorCode::Corrupt
                            : PersistenceErrorCode::Truncated,
                "checkpoint payload is incomplete or corrupt")};
        }

        const auto germline_compiled = GraphCompiler{}.compile(
            *germline_graph,
            persistence_detail::seeds_from_parameters(
                *germline_parameters));
        if (!germline_compiled.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::SchemaMismatch,
                "checkpoint germline graph cannot be compiled")};
        }
        const auto phenotype_compiled = GraphCompiler{}.compile(
            *phenotype_graph,
            persistence_detail::seeds_from_parameters(
                *phenotype_parameters));
        if (!phenotype_compiled.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::SchemaMismatch,
                "checkpoint phenotype graph cannot be compiled")};
        }
        auto runtime_result = RuntimeState::from_imported_state(
            phenotype_compiled.graph,
            std::make_shared<const InitialParameterValues>(
                *phenotype_parameters),
            std::span<const RuntimeCellState>(
                runtime_cells->second.data(), runtime_cells->second.size()),
            runtime_cells->first);
        if (!runtime_result.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::RuntimeRestore,
                runtime_result.error ? runtime_result.error->reason
                                      : "checkpoint runtime restore failed")};
        }
        auto ledger = ResourceLedger::from_snapshot(
            *runtime_result.runtime,
            *resource_config,
            *resource_snapshot);
        if (!ledger.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::ResourceRestore,
                ledger.error ? ledger.error->reason
                             : "checkpoint resource restore failed")};
        }
        auto prepared = CompiledExecutor::prepare(runtime_result.runtime->plan());
        if (!prepared.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::RuntimeRestore,
                "checkpoint executor preparation failed")};
        }
        auto lifecycle = CellularLifecycleController::create(
            *runtime_result.runtime,
            std::move(prepared.executor),
            std::move(ledger.ledger),
            *lifecycle_config);
        if (!lifecycle.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::LifecycleRestore,
                lifecycle.error ? lifecycle.error->reason
                                 : "checkpoint lifecycle restore failed")};
        }
        if (const auto error = lifecycle.controller->restore_states(
                std::span<const LifecycleCellState>(
                    lifecycle_states->data(), lifecycle_states->size()))) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::LifecycleRestore, error->reason)};
        }
        std::mt19937_64 rng;
        std::istringstream rng_stream(*rng_state);
        rng_stream >> rng;
        if (rng_stream.fail()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::Corrupt,
                "checkpoint RNG state is invalid")};
        }
        auto germline = std::shared_ptr<const Germline>(new Germline(
            *germline_graph,
            *rules,
            GermlineVersion{*germline_version},
            germline_compiled.graph,
            germline_compiled.initial_values));
        auto persisted_executor = prepared.executor;  // 与 lifecycle 共享同一执行视图
        auto persisted_growth = CellularGrowthController::create(
            *lifecycle.controller, GrowthConfig{});
        if (!persisted_growth.ok()) {
            return {{}, persistence_detail::failure(
                PersistenceErrorCode::Corrupt,
                persisted_growth.error
                    ? persisted_growth.error->reason
                    : "persisted growth instinct failed")};
        }
        auto candidate = std::unique_ptr<Phenotype>(new Phenotype(
            std::move(germline),
            std::move(runtime_result.runtime),
            std::move(persisted_executor),
            std::move(lifecycle.controller),
            std::move(persisted_growth.controller),
            *organism_id,
            0));
        candidate->rng_ = rng;
        target = std::move(*candidate);
        return {};
    }

    static PersistenceResult restore_into(
        Phenotype& target,
        std::span<const uint8_t> bytes,
        LearningWindow& external_window) {
        const auto result = restore_into(target, bytes);
        if (result.ok()) {
            external_window.invalidate_after_checkpoint_restore();
        }
        return result;
    }
};

}  // namespace kun::core

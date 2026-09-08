#pragma once

#include "kun/cellular/core/heredity.hpp"
#include <openssl/sha.h>
#include <bit>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace kun::transfer {
using namespace kun::core;

class KnowledgeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline void require(bool condition, const std::string& message) {
    if (!condition) throw KnowledgeError(message);
}

namespace wire {
using Bytes = std::vector<uint8_t>;
inline std::string digest(std::span<const uint8_t> bytes) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), hash);
    std::ostringstream out;
    for (auto byte : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
struct Writer {
    Bytes bytes;
    void integer(uint64_t v) {
        for (unsigned i = 0; i != 8; ++i) bytes.push_back(uint8_t(v >> (8 * i)));
    }
    void real(double value) {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
        require(std::isfinite(value), "wire: non-finite number");
        integer(std::bit_cast<uint64_t>(value));
    }
    void text(const std::string& value) {
        integer(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    void parameter(const ParameterValue& value) {
        integer(value.index());
        std::visit([&](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, ContinuousValue>) real(p.value);
            else if constexpr (std::is_same_v<T, ChannelIndex> || std::is_same_v<T, DelayTicks>)
                integer(p.value);
            else if constexpr (std::is_same_v<T, MinMaxMode>) integer(uint64_t(p));
        }, value);
    }
};
struct Reader {
    std::span<const uint8_t> bytes;
    std::size_t position{0};
    uint64_t integer() {
        require(bytes.size() - position >= 8, "wire: truncated integer");
        uint64_t v = 0;
        for (unsigned i = 0; i != 8; ++i) v |= uint64_t(bytes[position++]) << (8 * i);
        return v;
    }
    std::size_t count(std::size_t minimum_bytes = 8) {
        const auto n = integer();
        require(n <= (bytes.size() - position) / minimum_bytes, "wire: count exceeds payload");
        return static_cast<std::size_t>(n);
    }
    uint32_t u32() {
        auto n = integer();
        require(n <= UINT32_MAX, "wire: uint32 overflow");
        return static_cast<uint32_t>(n);
    }
    double real() {
        auto d = std::bit_cast<double>(integer());
        require(std::isfinite(d), "wire: non-finite number");
        return d;
    }
    std::string text() {
        const auto n = count(1);
        std::string value(reinterpret_cast<const char*>(bytes.data() + position), n);
        position += n;
        return value;
    }
    ParameterValue parameter() {
        switch (integer()) {
            case 0: return ContinuousValue{real()};
            case 1: {
                auto n = integer();
                require(n <= std::numeric_limits<std::size_t>::max(), "wire: channel overflow");
                return ChannelIndex{static_cast<std::size_t>(n)};
            }
            case 2: return DelayTicks{integer()};
            case 3: {
                auto n = integer(); require(n <= 1, "wire: unknown min/max mode");
                return static_cast<MinMaxMode>(n);
            }
            case 4: return UnusedParameter{};
            default: throw KnowledgeError("wire: unknown parameter type");
        }
    }
    void end() { require(position == bytes.size(), "wire: trailing payload"); }
};
} // namespace wire

struct KnowledgeRef {
    std::string object_id;
    std::string version;
    friend bool operator==(const KnowledgeRef&, const KnowledgeRef&) = default;
};

struct ModuleContract {
    std::string interface_id;
    std::string environment;
    std::size_t input_count{0};
    CellId output{};
    uint64_t timing_version{1};
    SemanticProfile semantic_profile{SemanticProfile::StrictCore};
    uint32_t semantic_version{1};
    bool compatible(const ModuleContract& other) const {
        return interface_id == other.interface_id && environment == other.environment &&
               input_count == other.input_count && timing_version == other.timing_version &&
               semantic_profile == other.semantic_profile &&
               semantic_version == other.semantic_version;
    }
};

inline InitialParameterSeeds parameter_seeds(std::span<const InitialParameterValue> values) {
    InitialParameterSeeds seeds;
    for (const auto& p : values) {
        if (p.binding.kind == ParameterBindingKind::CellParameter)
            seeds.cell_parameters.push_back({p.binding.cell, p.binding.slot, p.value});
        else
            seeds.edge_weights.push_back({p.binding.edge, std::get<ContinuousValue>(p.value).value});
    }
    return seeds;
}

inline GraphDefinition live_definition(const RuntimeState& runtime) {
    const auto& plan = *runtime.plan();
    GraphDefinition graph{plan.identity(), plan.revision(), plan.profile(),
                          plan.semantic_version(), {}, {}};
    for (const auto& c : plan.cells()) graph.cells.push_back({c.id, c.type});
    for (const auto& e : plan.edges())
        graph.edges.push_back({e.id, plan.cells()[e.source_index].id, e.source_port,
            plan.cells()[e.target_index].id, e.target_port, e.delay});
    return graph;
}

// An immutable executable description, never a checkpoint or a writable runtime.
class KnowledgeModule final {
public:
    static KnowledgeModule from_germline(const Germline& germline, ModuleContract contract,
                                        std::string producer_lineage) {
        return create(germline.definition(), parameter_seeds(germline.initial_values().entries()),
            std::move(contract), std::move(producer_lineage), "germline-template",
            "explicit birth-template export", germline.development_rules());
    }
    static KnowledgeModule from_phenotype(
        const Phenotype& phenotype, ModuleContract contract, std::string declaration,
        std::vector<CellId> selected = {}) {
        require(!declaration.empty(), "cultural publication requires an explicit declaration");
        auto graph = live_definition(phenotype.runtime());
        auto seeds = parameter_seeds(phenotype.runtime().parameters());
        if (!selected.empty()) {
            const std::set<CellId> keep(selected.begin(), selected.end());
            require(keep.size() == selected.size(), "duplicate selected cell");
            for (auto id : keep)
                require(phenotype.runtime().cell_state(id), "selected cell is missing");
            for (const auto& edge : graph.edges)
                require(!keep.contains(edge.target) || keep.contains(edge.source),
                        "selected motif has an undeclared incoming dependency");
            std::erase_if(graph.cells, [&](const auto& c) { return !keep.contains(c.id); });
            std::erase_if(graph.edges, [&](const auto& e) {
                return !keep.contains(e.source) || !keep.contains(e.target);
            });
            std::erase_if(seeds.cell_parameters, [&](const auto& p) { return !keep.contains(p.cell); });
            std::erase_if(seeds.edge_weights, [&](const auto& p) {
                return std::none_of(graph.edges.begin(), graph.edges.end(),
                    [&](const auto& e) { return e.id == p.edge; });
            });
        }
        return create(std::move(graph), std::move(seeds), std::move(contract),
            "organism:" + std::to_string(phenotype.organism_id()) + "/germline:" +
            std::to_string(phenotype.germline().definition().identity.value) + "/" +
            std::to_string(phenotype.germline().version().value),
            "phenotype-cultural", std::move(declaration), "");
    }
    const Germline& germline() const { return *germline_; }
    const ModuleContract& contract() const { return contract_; }
    const std::string& producer_lineage() const { return producer_; }
    const std::string& origin() const { return origin_; }
    const std::string& declaration() const { return declaration_; }
    std::shared_ptr<RuntimeState> fresh_runtime() const {
        auto result = RuntimeState::create(germline_->graph(),
            std::make_shared<const InitialParameterValues>(germline_->initial_values()));
        require(result.ok(), result.error ? result.error->reason : "fresh runtime failed");
        return result.runtime;
    }
    std::unique_ptr<Phenotype> spawn_birth(const OffspringSpec& spec) const {
        auto result = germline_->spawn_offspring(spec);
        require(result.ok(), result.error ? result.error->reason : "birth import failed");
        return std::move(result.phenotype);
    }
    wire::Bytes encode() const {
        wire::Writer w;
        w.text("KUN-NATIVE-KNOWLEDGE"); w.integer(1);
        w.text(contract_.interface_id); w.text(contract_.environment);
        w.integer(contract_.input_count); w.integer(contract_.output.value);
        w.integer(contract_.timing_version);
        w.integer(uint64_t(contract_.semantic_profile)); w.integer(contract_.semantic_version);
        w.text(producer_); w.text(origin_); w.text(declaration_);
        w.text(germline_->development_rules());
        const auto& g = germline_->definition();
        w.integer(g.identity.value); w.integer(g.revision.value);
        w.integer(uint64_t(g.profile)); w.integer(g.semantic_version);
        w.integer(g.cells.size());
        for (const auto& c : g.cells) { w.integer(c.id.value); w.integer(uint64_t(c.type)); }
        w.integer(g.edges.size());
        for (const auto& e : g.edges) {
            w.integer(e.id.value); w.integer(e.source.value); w.integer(e.source_port.value);
            w.integer(e.target.value); w.integer(e.target_port.value); w.integer(uint64_t(e.delay));
        }
        const auto seeds = parameter_seeds(germline_->initial_values().entries());
        w.integer(seeds.cell_parameters.size());
        for (const auto& p : seeds.cell_parameters) {
            w.integer(p.cell.value); w.integer(uint64_t(p.slot)); w.parameter(p.value);
        }
        w.integer(seeds.edge_weights.size());
        for (const auto& p : seeds.edge_weights) { w.integer(p.edge.value); w.real(p.initial_weight); }
        return std::move(w.bytes);
    }
    static KnowledgeModule decode(const wire::Bytes& bytes) {
        wire::Reader r{bytes};
        require(r.text() == "KUN-NATIVE-KNOWLEDGE" && r.integer() == 1,
                "unsupported knowledge artifact format/version");
        ModuleContract contract;
        contract.interface_id = r.text(); contract.environment = r.text();
        const auto width = r.integer();
        require(width <= std::numeric_limits<std::size_t>::max(), "input width overflow");
        contract.input_count = static_cast<std::size_t>(width);
        contract.output = CellId{r.integer()}; contract.timing_version = r.integer();
        const auto profile = r.integer();
        require(profile == uint64_t(SemanticProfile::StrictCore), "unsupported knowledge semantics");
        contract.semantic_profile = SemanticProfile::StrictCore;
        contract.semantic_version = r.u32();
        auto producer = r.text(); auto origin = r.text(); auto declaration = r.text();
        auto rules = r.text();
        GraphDefinition g;
        g.identity = GraphIdentity{r.integer()}; g.revision = GraphRevision{r.integer()};
        const auto graph_profile = r.integer();
        require(graph_profile == uint64_t(SemanticProfile::StrictCore), "unsupported knowledge semantics");
        g.profile = SemanticProfile::StrictCore; g.semantic_version = r.u32();
        auto count = r.count(16);
        for (std::size_t i = 0; i < count; ++i) {
            auto id = CellId{r.integer()}; const auto type = r.integer();
            require(type <= UINT8_MAX && contract_for_code(uint8_t(type)).has_value(),
                    "unknown cell type");
            g.cells.push_back({id, static_cast<CellType>(type)});
        }
        count = r.count(48);
        for (std::size_t i = 0; i < count; ++i) {
            EdgeDefinition e;
            e.id = EdgeId{r.integer()}; e.source = CellId{r.integer()};
            e.source_port = OutputPort{r.u32()}; e.target = CellId{r.integer()};
            e.target_port = InputPort{r.u32()};
            const auto delay = r.integer(); require(delay <= 1, "unknown edge timing");
            e.delay = static_cast<EdgeDelay>(delay); g.edges.push_back(e);
        }
        InitialParameterSeeds seeds;
        count = r.count(24);
        for (std::size_t i = 0; i < count; ++i) {
            const auto cell = CellId{r.integer()}; auto slot = r.integer();
            require(slot <= 1, "unknown parameter slot");
            seeds.cell_parameters.push_back({cell, static_cast<ParameterSlot>(slot), r.parameter()});
        }
        count = r.count(16);
        for (std::size_t i = 0; i < count; ++i) {
            auto edge = EdgeId{r.integer()}; auto value = r.real();
            seeds.edge_weights.push_back({edge, value});
        }
        r.end();
        auto result = create(std::move(g), std::move(seeds), std::move(contract),
            std::move(producer), std::move(origin), std::move(declaration), std::move(rules));
        require(result.encode() == bytes, "non-canonical native artifact");
        return result;
    }
private:
    static KnowledgeModule create(GraphDefinition graph, InitialParameterSeeds seeds,
        ModuleContract contract, std::string producer, std::string origin,
        std::string declaration, std::string rules) {
        require(graph.profile == SemanticProfile::StrictCore && graph.semantic_version == 1,
                "knowledge requires StrictCore semantic version 1");
        require(!contract.interface_id.empty() && !contract.environment.empty() &&
                contract.input_count > 0 && contract.timing_version == 1 &&
                contract.semantic_profile == SemanticProfile::StrictCore &&
                contract.semantic_version == 1,
                "missing or unsupported module interface/environment/timing");
        require(contract.semantic_profile == graph.profile &&
                contract.semantic_version == graph.semantic_version,
                "module semantic contract does not match graph");
        require(!producer.empty() && !declaration.empty(), "missing publication provenance");
        require(origin == "germline-template" || origin == "phenotype-cultural", "unknown publication origin");
        require(std::any_of(graph.cells.begin(), graph.cells.end(),
                    [&](const auto& c) { return c.id == contract.output; }), "missing output cell");
        for (const auto& c : graph.cells) {
            const auto code = uint64_t(c.type);
            if (code <= 3) require(code < contract.input_count, "input interface omits fixed receptor channel");
            if (c.type == CellType::PREDICT_SENSE_0 || c.type == CellType::PREDICT_SENSE_1)
                require(code - 40 < contract.input_count, "input interface omits prediction channel");
        }
        for (const auto& p : seeds.cell_parameters)
            if (const auto* channel = std::get_if<ChannelIndex>(&p.value))
                require(channel->value < contract.input_count, "channel outside declared interface");
        auto result = Germline::create(std::move(graph), std::move(seeds), std::move(rules));
        require(result.ok(), result.error ? result.error->reason : "native validation failed");
        KnowledgeModule module(result.germline, std::move(contract), std::move(producer),
            std::move(origin), std::move(declaration));
        module.fresh_runtime();
        return module;
    }
    KnowledgeModule(std::shared_ptr<const Germline> germline, ModuleContract contract,
        std::string producer, std::string origin, std::string declaration)
        : germline_(std::move(germline)), contract_(std::move(contract)),
          producer_(std::move(producer)), origin_(std::move(origin)),
          declaration_(std::move(declaration)) {}
    std::shared_ptr<const Germline> germline_;
    ModuleContract contract_;
    std::string producer_, origin_, declaration_;
};
} // namespace kun::transfer

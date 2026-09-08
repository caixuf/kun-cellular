#pragma once
#include "tasks/transfer/germline_library.hpp"
#include "kun/cellular/core/growth_controller.hpp"

namespace kun::transfer {

struct AdoptionRequest {
    ModuleContract target_contract;
    EdgeId target_edge;
    uint64_t expected_tick;
    GrowthFunding funding;
    double cell_cost;
    double synapse_cost;
    double initial_energy;
};
struct AdoptionWork {
    uint64_t attempts{0}, failed_attempts{0}, graph_executions{0}, cell_visits{0}, edge_visits{0};
};
struct AdoptionReceipt {
    KnowledgeRef source;
    std::string content_digest;
    uint64_t borrow_event, organism_id, boundary_tick;
    GraphRevision revision_before, revision_after;
    std::vector<CellId> inserted_cells;
    std::vector<EdgeId> new_edges;
    double paid_cost;
    ResourceGrowthReport funding;
    AdoptionWork work;
};

// Caller exclusively owns the phenotype at this declared cold boundary.
// target_edge is the explicit host/module composition boundary: v1 remaps a
// complete scalar receptor -> unary primitive motif onto that edge, allocating
// fresh stable IDs. There is no implicit semantic subgraph discovery. Full
// arbitrary graphs remain available only as explicit birth imports.
inline AdoptionReceipt adopt_at_cold_boundary(
    Phenotype& target, const BorrowedKnowledge& borrowed, const AdoptionRequest& request,
    std::span<const double> inputs, AdoptionWork* accumulated = nullptr) {
    AdoptionWork work;
    work.attempts = 1;
    auto accumulate = [&] {
        if (!accumulated) return;
        accumulated->attempts += work.attempts;
        accumulated->failed_attempts += work.failed_attempts;
        accumulated->graph_executions += work.graph_executions;
        accumulated->cell_visits += work.cell_visits;
        accumulated->edge_visits += work.edge_visits;
    };
    try {
        const auto& module = borrowed.module;
        require(wire::digest(module.encode()) == borrowed.content_digest, "borrowed content identity mismatch");
        require(request.target_contract.compatible(module.contract()), "adoption interface/environment/timing mismatch");
        require(target.runtime().profile() == module.germline().graph()->profile() &&
                target.runtime().plan()->semantic_version() == module.germline().graph()->semantic_version(),
                "adoption semantic mismatch");
        require(target.runtime().tick() == request.expected_tick, "stale cold learning boundary");
        require(inputs.size() == request.target_contract.input_count &&
                std::all_of(inputs.begin(), inputs.end(), [](double v) { return std::isfinite(v); }),
                "adoption tick input interface mismatch");
        require(std::isfinite(request.cell_cost) && request.cell_cost > 0 &&
                std::isfinite(request.synapse_cost) && request.synapse_cost > 0 &&
                std::isfinite(request.initial_energy) && request.initial_energy > 0,
                "adopted cells/edges require positive finite configured funding and endowment");
        const auto& graph = module.germline().definition();
        require(graph.cells.size() == 2 && graph.edges.size() == 1 &&
                module.contract().input_count == 1, "live adoption requires an explicit receptor/unary motif");
        const auto& edge = graph.edges.front();
        auto source = std::find_if(graph.cells.begin(), graph.cells.end(),
            [&](const auto& c) { return c.id == edge.source; });
        auto output = std::find_if(graph.cells.begin(), graph.cells.end(),
            [&](const auto& c) { return c.id == edge.target; });
        require(source != graph.cells.end() && output != graph.cells.end() &&
                source->type == CellType::SENSE_RAW_INPUT_0 && output->id == module.contract().output &&
                contract_for(output->type)->get().input_port_count >= 1 &&
                edge.target_port == InputPort{0} &&
                edge.delay == EdgeDelay::Immediate, "unsupported motif ports or causal timing");
        const auto& target_plan = *target.runtime().plan();
        auto link = std::find_if(target_plan.edges().begin(), target_plan.edges().end(),
            [&](const auto& e) { return e.id == request.target_edge; });
        require(link != target_plan.edges().end() && link->delay == EdgeDelay::Immediate,
                "live adoption requires an existing immediate target edge");
        require(target_plan.cells()[link->source_index].type == CellType::SENSE_RAW_INPUT_0 &&
                std::get<ContinuousValue>(*target.runtime().parameter(
                    target_plan.cells()[link->source_index].id, ParameterSlot::Param1)).value == 1.0,
                "motif boundary requires an unscaled scalar receptor");
        const auto resources = target.ledger().snapshot();
        for (const auto& state : target.lifecycle().states())
            require(state.state == LifecycleState::Active &&
                    resources.cell(state.cell)->energy > target.lifecycle().config().dormant_enter_resource,
                    "adoption requires an active cold structure boundary without pending death/dormancy");

        const auto edit_history = target.lifecycle().edit_history();
        uint64_t cell_max = 0, edge_max = 0;
        for (const auto& c : target_plan.cells()) cell_max = std::max(cell_max, c.id.value);
        for (const auto& e : target_plan.edges()) edge_max = std::max(edge_max, e.id.value);
        for (const auto id : edit_history.retired_cells) cell_max = std::max(cell_max, id.value);
        for (const auto id : edit_history.retired_edges) edge_max = std::max(edge_max, id.value);
        require(cell_max < UINT64_MAX && edge_max <= UINT64_MAX - 2,
                "stable ID namespace exhausted");
        const CellId inserted{cell_max + 1};
        const EdgeId incoming{edge_max + 1}, outgoing{edge_max + 2};
        const auto fresh = module.fresh_runtime();
        const double gain = std::get<ContinuousValue>(*fresh->parameter(source->id, ParameterSlot::Param1)).value;
        const auto seeds = parameter_seeds(module.germline().initial_values().entries());
        const double module_weight = seeds.edge_weights.front().initial_weight;
        const double coupling = std::get<ContinuousValue>(
            *target.runtime().parameter_at(link->weight_parameter_index)).value;
        require(std::isfinite(gain * module_weight) && gain * module_weight != 0 && coupling != 0,
                "adoption requires finite nonzero causal coupling");
        SplitEdgeAction split{request.target_edge,
            {inserted, output->type, {*fresh->parameter(output->id, ParameterSlot::Param1),
                                     *fresh->parameter(output->id, ParameterSlot::Param2)}},
            incoming, outgoing, InputPort{0}, gain * module_weight, coupling};
        GrowthConfig config;
        // R4 split births charge cell + endowment, but not their two created edges.
        // Include both edge construction costs in the funded birth charge; no free edges.
        config.cell_birth_cost = request.cell_cost + 2 * request.synapse_cost;
        config.synapse_birth_cost = request.synapse_cost;
        config.initial_energy = config.initial_capacity = request.initial_energy;
        const GrowthProposal proposal = GrowthSplitProposal{1, split, request.funding};
        auto probe = target.runtime().fork_probe();
        require(probe.ok(), "adoption probe allocation failed");
        auto executor = CompiledExecutor::prepare(probe.runtime->plan());
        require(executor.ok(), "adoption probe preparation failed");
        auto lifecycle = CellularLifecycleController::create(*probe.runtime, executor.executor,
            target.ledger().fork_probe(), target.lifecycle().config());
        require(lifecycle.ok(), "adoption probe lifecycle binding failed");
        require(!lifecycle.controller->restore_states(target.lifecycle().states()), "probe state restore failed");
        auto growth = CellularGrowthController::create(std::move(lifecycle.controller), config);
        require(growth.ok(), "adoption growth config rejected");
        require(growth.controller->submit(proposal).ok(), "adoption proposal rejected");
        const auto preflight = growth.controller->step(inputs);
        if (preflight.lifecycle.executed) {
            ++work.graph_executions;
            work.cell_visits += probe.runtime->plan()->cells().size();
            work.edge_visits += probe.runtime->plan()->edges().size();
        }
        require(preflight.ok(), preflight.growth_error ? preflight.growth_error->reason :
            (preflight.lifecycle.error ? preflight.lifecycle.error->reason : "adoption preflight failed"));
        auto actual = CellularGrowthController::create(target.lifecycle(), config);
        require(actual.ok() && actual.controller->submit(proposal).ok(), "adoption binding rejected");
        const auto old_revision = target.runtime().revision();
        const auto result = actual.controller->step(inputs);
        if (result.lifecycle.executed) {
            ++work.graph_executions;
            work.cell_visits += target.runtime().plan()->cells().size();
            work.edge_visits += target.runtime().plan()->edges().size();
        }
        require(result.ok() && result.growth_report.has_value(),
            result.growth_error ? result.growth_error->reason : "adoption commit failed");
        AdoptionReceipt receipt{borrowed.source, borrowed.content_digest, borrowed.event_sequence,
            target.organism_id(), request.expected_tick, old_revision, target.runtime().revision(),
            {inserted}, {incoming, outgoing}, result.growth_report->paid_cost,
            *result.growth_report, work};
        accumulate();
        return receipt;
    } catch (...) {
        work.failed_attempts = 1;
        accumulate();
        throw;
    }
}

// Durable observatory receipt, not a live checkpoint. The caller retains the
// receipt if persistence fails and can retry: SQLite and live RAM are not one transaction.
inline void GermlineLibraryStore::record_adoption(const AdoptionReceipt& receipt,
                                                  const Phenotype& target) {
    database::Transaction tx(db_);
    require(receipt.organism_id == target.organism_id() &&
            receipt.revision_after == target.runtime().revision(),
            "adoption receipt does not match the live target");
    require(receipt.paid_cost > 0 && std::isfinite(receipt.paid_cost) &&
            std::isfinite(receipt.funding.conservation_residual) &&
            std::abs(receipt.funding.conservation_residual) < 1e-9,
            "adoption receipt has invalid payment accounting");
    for (const auto id : receipt.inserted_cells)
        require(std::any_of(target.runtime().plan()->cells().begin(),
                            target.runtime().plan()->cells().end(),
                            [&](const auto& cell) { return cell.id == id; }),
                "adoption receipt references a missing inserted cell");
    for (const auto id : receipt.new_edges)
        require(std::any_of(target.runtime().plan()->edges().begin(),
                            target.runtime().plan()->edges().end(),
                            [&](const auto& edge) { return edge.id == id; }),
                "adoption receipt references a missing inserted edge");
    const auto e = load(receipt.source);
    require(wire::digest(e.module.encode()) == receipt.content_digest, "adoption/source identity mismatch");
    database::Statement borrow(db_, "SELECT event,object_id,version FROM events WHERE sequence=?");
    borrow.integer(1, receipt.borrow_event);
    require(borrow.row() && borrow.text(0) == "borrow" && borrow.text(1) == receipt.source.object_id &&
            borrow.text(2) == receipt.source.version, "adoption lacks matching borrow receipt");
    wire::Writer w;
    w.text("KUN-NATIVE-ADOPTION/v1"); w.text(receipt.content_digest);
    w.integer(receipt.organism_id); w.integer(receipt.boundary_tick);
    w.integer(receipt.revision_before.value); w.integer(receipt.revision_after.value);
    w.integer(receipt.inserted_cells.size());
    for (auto id : receipt.inserted_cells) w.integer(id.value);
    w.integer(receipt.new_edges.size());
    for (auto id : receipt.new_edges) w.integer(id.value);
    w.real(receipt.paid_cost); w.real(receipt.funding.conservation_residual);
    w.integer(receipt.funding.payments.size());
    for (const auto& payment : receipt.funding.payments) {
        w.integer(payment.sponsor.has_value());
        w.integer(payment.sponsor ? payment.sponsor->value : 0);
        w.integer(payment.compartment.value); w.real(payment.requested);
        w.real(payment.from_structural_reserve); w.real(payment.from_energy);
        w.real(payment.from_environment); w.real(payment.paid);
    }
    w.integer(receipt.work.graph_executions); w.integer(receipt.work.cell_visits);
    w.integer(receipt.work.edge_visits);
    const auto digest = wire::digest(w.bytes);
    database::Statement existing(db_, "SELECT receipt FROM adoptions WHERE borrow_event=? AND organism_id=? AND boundary_tick=?");
    existing.integer(1, receipt.borrow_event); existing.text(2, std::to_string(receipt.organism_id));
    existing.text(3, std::to_string(receipt.boundary_tick));
    if (existing.row()) {
        require(existing.blob(0) == w.bytes, "conflicting immutable adoption receipt");
        tx.commit();
        return;
    }
    database::Statement insert(db_, "INSERT INTO adoptions(object_id,version,borrow_event,organism_id,boundary_tick,receipt_digest,receipt) VALUES(?,?,?,?,?,?,?)");
    database::ref(insert, receipt.source); insert.integer(3, receipt.borrow_event);
    insert.text(4, std::to_string(receipt.organism_id)); insert.text(5, std::to_string(receipt.boundary_tick));
    insert.text(6, digest); insert.blob(7, w.bytes); insert.done();
    event("adopt", receipt.source, std::to_string(receipt.organism_id),
          "sha256:" + digest + "; paid_model_units=" + std::to_string(receipt.paid_cost) +
          "; native_graph_executions=" + std::to_string(receipt.work.graph_executions));
    tx.commit();
}
} // namespace kun::transfer

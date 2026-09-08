#include "tasks/transfer/germline_library.hpp"
#include "tasks/transfer/knowledge_adoption.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace kun;
using namespace kun::core;
using namespace kun::transfer;

template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const KnowledgeError&) { rejected = true; }
    assert(rejected);
}

ModuleContract contract() {
    return {"scalar-scale/v1", "deterministic-scalars/v1", 1, CellId{2}, 1};
}

std::shared_ptr<const Germline> genome(double gain = 2.0, std::size_t count = 2) {
    GraphDefinition graph{GraphIdentity{700}, GraphRevision{1},
                          SemanticProfile::StrictCore, 1, {}, {}};
    InitialParameterSeeds seeds;
    for (std::size_t i = 1; i <= count; ++i) {
        const CellId id{i};
        graph.cells.push_back({id, i == 1 ? CellType::SENSE_RAW_INPUT_0 : CellType::OP_SUM});
        seeds.cell_parameters.push_back({id, ParameterSlot::Param1,
            i == 1 ? ParameterValue{ContinuousValue{1.0}} : ParameterValue{UnusedParameter{}}});
        seeds.cell_parameters.push_back({id, ParameterSlot::Param2, UnusedParameter{}});
        if (i > 1) {
            graph.edges.push_back({EdgeId{i + 8}, CellId{i - 1}, OutputPort{0},
                                  id, InputPort{0}, EdgeDelay::Immediate});
            seeds.edge_weights.push_back({EdgeId{i + 8}, i == 2 ? gain : 1.0});
        }
    }
    const auto result = Germline::create(graph, seeds, "opaque-only");
    assert(result.ok());
    return result.germline;
}

OffspringSpec spec(const Germline& germline, uint64_t id, double energy = 100.0) {
    OffspringSpec s;
    s.organism_id = id;
    s.rng_seed = id;
    s.lifecycle_config.dormant_exit_resource = 1.0;
    s.resource_config.dt = 1;
    s.resource_config.activity_scale = 1;
    s.resource_config.transmission_scale = 1;
    s.resource_compartments.push_back({ResourceCompartmentId{1}, 0.0});
    for (const auto& cell : germline.definition().cells)
        s.resource_cells.push_back({cell.id, ResourceCompartmentId{1}, energy, energy, 0.0});
    return s;
}

KnowledgeModule module(double gain = 2.0) {
    return KnowledgeModule::from_germline(*genome(gain), contract(), "producer-A");
}

EvaluationProtocol protocol(double gain = 2.0) {
    return {"scale-trace/v1", "deterministic-scalars/v1", 1e-12,
            {{11, {{{-2.0}, {-2.0 * gain}}, {{1.0}, {gain}}}}},
            {{21, {{{-3.0}, {-3.0 * gain}}, {{0.5}, {0.5 * gain}}}}}};
}

void exec_sql(const std::string& path, const char* sql) {
    sqlite3* db = nullptr;
    assert(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    char* error = nullptr;
    const int result = sqlite3_exec(db, sql, nullptr, nullptr, &error);
    const std::string text = error ? error : "";
    sqlite3_free(error);
    sqlite3_close(db);
    if (result != SQLITE_OK) throw KnowledgeError(text);
}

int main() {
    const auto path = (std::filesystem::current_path() /
        ("r8-native-test-" + std::to_string(getpid()) + ".db")).string();
    std::filesystem::remove(path);
    const KnowledgeRef v1{"absolute", "1"};
    std::vector<uint8_t> original;
    {
        auto m = module();
        original = m.encode();
        auto loaded = KnowledgeModule::decode(original);
        assert(loaded.encode() == original);
        assert(loaded.fresh_runtime()->tick() == 0);
        rejects([&] { KnowledgeModule::decode({1, 2, 3, 5}); });
        auto damaged = original;
        damaged.push_back(0);
        rejects([&] { KnowledgeModule::decode(damaged); });
        wire::Reader header{original};
        header.text();
        auto unknown_artifact = original;
        unknown_artifact[header.position] = 99;
        rejects([&] { KnowledgeModule::decode(unknown_artifact); });
        auto wrong = contract(); wrong.output = CellId{999};
        rejects([&] { KnowledgeModule::from_germline(*genome(), wrong, "A"); });
        auto wrong_semantics = contract(); wrong_semantics.semantic_version = 2;
        rejects([&] { KnowledgeModule::from_germline(*genome(), wrong_semantics, "A"); });
        auto large_contract = contract(); large_contract.output = CellId{70};
        auto large = KnowledgeModule::from_germline(*genome(2, 70), large_contract, "A");
        assert(KnowledgeModule::decode(large.encode()).germline().graph()->cells().size() == 70);

        GermlineLibraryStore db(path);
        db.publish(v1, "Absolute mechanism", m, {});
        rejects([&] { db.publish(v1, "overwrite", module(3), {}); });
        assert(db.entry(v1).status == "candidate");
        rejects([&] { db.borrow(v1, contract(), "B"); });
        auto invalid = protocol(); invalid.ood.front().seed = invalid.train.front().seed;
        rejects([&] { db.evaluate(v1, invalid, "evaluator"); });
        invalid = protocol(); invalid.train.clear();
        rejects([&] { db.evaluate(v1, invalid, "evaluator"); });
        invalid = protocol(); invalid.environment = "wrong";
        rejects([&] { db.evaluate(v1, invalid, "evaluator"); });
        auto failed = db.evaluate(v1, protocol(3), "evaluator");
        assert(!failed.passed && failed.graph_executions == 8);
        assert(db.entry(v1).failures == 5);
        auto passed = db.evaluate(v1, protocol(), "evaluator");
        assert(passed.passed && passed.graph_executions == 8);
        assert(!passed.report_digest.empty());
        assert(db.entry(v1).status == "research-validated");
        assert(db.find(contract()).size() == 1);
        auto wrong_contract = contract(); wrong_contract.environment = "other";
        assert(db.find(wrong_contract).empty());
        rejects([&] { db.borrow(v1, wrong_contract, "B"); });
        auto borrowed = db.borrow(v1, contract(), "unrelated-B");
        auto born = borrowed.module.spawn_birth(spec(borrowed.module.germline(), 200));
        assert(born->runtime().tick() == 0 && born->organism_id() == 200);
        double x = -2;
        assert(born->step({&x, 1}).ok());
        assert(born->runtime().cell_state(CellId{2})->output_val == -4);

        const auto target_genome = genome(1);
        auto target = target_genome->spawn_offspring(spec(*target_genome, 300)).phenotype;
        assert(target->step({&x, 1}).ok());
        const auto germline_bytes = KnowledgeModule::from_germline(
            target->germline(), contract(), "unchanged").encode();
        const auto before_tick = target->runtime().tick();
        AdoptionRequest request{contract(), EdgeId{10}, before_tick,
            {CellId{1}, ResourceCompartmentId{1}}, 1.0, 0.25, 1.0};
        auto stale = request; stale.expected_tick = 0;
        rejects([&] { adopt_at_cold_boundary(*target, borrowed, stale, {&x, 1}); });
        auto receipt = adopt_at_cold_boundary(*target, borrowed, request, {&x, 1});
        db.record_adoption(receipt);
        db.record_adoption(receipt); // retry is idempotent, not a second adoption
        assert(receipt.source == v1 && receipt.paid_cost == 2.5);
        assert(receipt.inserted_cells.size() == 1 && receipt.new_edges.size() == 2);
        assert(receipt.work.graph_executions == 2);
        assert(target->ledger().snapshot().cell(CellId{1})->energy == 97.5);
        assert(std::abs(receipt.funding.conservation_residual) < 1e-12);
        assert(target->runtime().plan()->cells().size() == 3);
        assert(target->runtime().cell_state(CellId{2})->output_val == -4);
        assert(target->runtime().cell_state(CellId{1})->activation_count == 2);
        assert(target->runtime().tick() == before_tick + 1);
        assert(KnowledgeModule::from_germline(target->germline(), contract(), "unchanged").encode()
               == germline_bytes);
        auto child = target->germline().spawn_offspring(spec(target->germline(), 301)).phenotype;
        assert(child->runtime().plan()->cells().size() == 2);
        assert(child->runtime().tick() == 0);

        auto poor = target_genome->spawn_offspring(spec(*target_genome, 400, 1)).phenotype;
        request.expected_tick = 0;
        const auto old_plan = poor->runtime().plan();
        rejects([&] { adopt_at_cold_boundary(*poor, borrowed, request, {&x, 1}); });
        assert(poor->runtime().plan() == old_plan && poor->runtime().tick() == 0);
        auto state_graph = target_genome->definition();
        state_graph.cells[1].type = CellType::OP_EMA;
        auto state_seeds = parameter_seeds(target_genome->initial_values().entries());
        for (auto& p : state_seeds.cell_parameters)
            if (p.cell == CellId{2} && p.slot == ParameterSlot::Param1) p.value = ContinuousValue{0.5};
        const auto state_genome = Germline::create(state_graph, state_seeds).germline;
        assert(state_genome);
        auto stateful = state_genome->spawn_offspring(spec(*state_genome, 500)).phenotype;
        double initial_input = 4;
        assert(stateful->step({&initial_input, 1}).ok());
        assert(stateful->runtime().cell_state(CellId{2})->state_val != 0);
        auto control = stateful->runtime().fork_probe();
        for (const auto& p : control.runtime->parameters())
            if (p.binding.kind == ParameterBindingKind::EdgeWeight)
                assert(control.runtime->set_parameter(p.binding, ContinuousValue{2.0}).ok());
        auto control_executor = CompiledExecutor::prepare(control.runtime->plan());
        assert(control_executor.executor->step(*control.runtime, {&x, 1}).ok());
        auto state_request = request; state_request.expected_tick = stateful->runtime().tick();
        auto state_receipt = adopt_at_cold_boundary(*stateful, borrowed, state_request, {&x, 1});
        assert(state_receipt.work.graph_executions == 2);
        assert(stateful->runtime().cell_state(CellId{2})->state_val ==
               control.runtime->cell_state(CellId{2})->state_val);
        assert(stateful->runtime().cell_state(CellId{2})->output_val ==
               control.runtime->cell_state(CellId{2})->output_val);
        assert(stateful->runtime().cell_state(CellId{2})->activation_count ==
               control.runtime->cell_state(CellId{2})->activation_count);
        assert(poor->ledger().snapshot().cell(CellId{1})->energy == 1);
        auto timing = request; timing.target_contract.timing_version = 2;
        rejects([&] { adopt_at_cold_boundary(*poor, borrowed, timing, {&x, 1}); });
        assert(poor->runtime().plan() == old_plan && poor->runtime().tick() == 0);

        const auto internal = receipt.inserted_cells.front();
        const auto edge_id = receipt.new_edges.front();
        for (const auto& p : target->runtime().parameters())
            if (p.binding.kind == ParameterBindingKind::EdgeWeight && p.binding.edge == edge_id)
                assert(target->runtime().set_parameter(p.binding, ContinuousValue{3.0}).ok());
        auto selected = contract(); selected.output = internal;
        rejects([&] { KnowledgeModule::from_phenotype(*target, selected, "", {CellId{1}, internal}); });
        auto improved = KnowledgeModule::from_phenotype(*target, selected,
            "explicit cultural extraction: selected live parameters and structure",
            {CellId{1}, internal});
        assert(improved.origin() == "phenotype-cultural");
        assert(improved.fresh_runtime()->tick() == 0);
        db.publish({"absolute", "2"}, "Improved mechanism", improved, {v1});
        assert(db.evaluate({"absolute", "2"}, protocol(3), "evaluator").passed);
        assert(db.entry(v1).module.encode() == original);
        assert(db.entry({"absolute", "2"}).parents == std::vector<KnowledgeRef>{v1});
        db.retire(v1, "maintenance", "superseded");
        rejects([&] { db.borrow(v1, contract(), "too-late"); });
        assert(db.evaluate(v1, protocol(), "later-evaluator").passed);
        assert(db.entry(v1).status == "retired");
        rejects([&] { exec_sql(path, "DELETE FROM events"); });
        rejects([&] { exec_sql(path, "UPDATE knowledge_objects SET title='mutated'"); });
        rejects([&] { exec_sql(path, "DELETE FROM evidence"); });
        rejects([&] { exec_sql(path, "DELETE FROM adoptions"); });
        rejects([&] { exec_sql(path, "INSERT INTO parents VALUES('absolute','1','absolute','2')"); });
        db.publish({"large", "1"}, "Uncapped 70-cell native graph", large, {});
        assert(db.evaluate({"large", "1"}, protocol(), "large-evaluator").passed);
        auto large_borrow = db.borrow({"large", "1"}, large_contract, "large-reader");
        assert(large_borrow.module.fresh_runtime()->plan()->cells().size() == 70);
        rejects([&] { adopt_at_cold_boundary(*poor, large_borrow, request, {&x, 1}); });
    }
    {
        GermlineLibraryStore reopened(path);
        assert(reopened.entry(v1).module.encode() == original);
        assert(reopened.entry(v1).status == "retired");
        auto newer = reopened.borrow({"absolute", "2"}, contract(), "unrelated-C");
        double x = -2;
        auto runtime = newer.module.fresh_runtime();
        auto executor = CompiledExecutor::prepare(runtime->plan());
        assert(executor.executor->step(*runtime, {&x, 1}).ok());
        assert(runtime->cell_state(newer.module.contract().output)->output_val == -6);
        assert(!reopened.evaluate({"absolute", "2"}, protocol(), "negative-after-pass").passed);
        assert(reopened.entry({"absolute", "2"}).status == "candidate");
        reopened.evaluate({"absolute", "2"}, protocol(3), "revalidate");
    }
    std::thread t1([&] {
        GermlineLibraryStore db(path);
        for (int i = 0; i < 10; ++i) db.borrow({"absolute", "2"}, contract(), "thread-1");
    });
    std::thread t2([&] {
        GermlineLibraryStore db(path);
        for (int i = 0; i < 10; ++i) db.borrow({"absolute", "2"}, contract(), "thread-2");
    });
    t1.join(); t2.join();
    {
        GermlineLibraryStore db(path);
        assert(db.entry({"absolute", "2"}).borrows == 21);
        const auto failed_before = db.entry({"absolute", "2"}).failures;
        exec_sql(path, "CREATE TRIGGER reject_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'injected audit failure'); END");
        rejects([&] { db.borrow({"absolute", "2"}, contract(), "audit-failure"); });
        assert(db.entry({"absolute", "2"}).borrows == 21);
        assert(db.entry({"absolute", "2"}).failures == failed_before);
        exec_sql(path, "DROP TRIGGER reject_event");
    }
    exec_sql(path, "PRAGMA user_version=999");
    rejects([&] { GermlineLibraryStore unknown(path); });
    exec_sql(path, "PRAGMA user_version=3");
    exec_sql(path, "DROP TRIGGER evidence_no_update; UPDATE evidence SET manifest_digest='bad'");
    // A connection opened before corruption must still check report-to-column bindings.
    exec_sql(path, "CREATE TRIGGER evidence_no_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT,'immutable R8 record'); END");
    rejects([&] { GermlineLibraryStore corrupt_evidence(path); corrupt_evidence.entry(v1); });
    exec_sql(path, "DROP TRIGGER objects_no_update; UPDATE knowledge_objects SET artifact=x'0102'");
    rejects([&] { GermlineLibraryStore corrupt(path); corrupt.entry(v1); });
    std::filesystem::remove(path);
    std::cout << "R8 native typed artifacts, evidence, live funded transfer, persistence: passed\n";
}

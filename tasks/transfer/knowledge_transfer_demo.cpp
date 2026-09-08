#include "tasks/transfer/knowledge_adoption.hpp"
#include <iostream>

using namespace kun;
using namespace kun::core;
using namespace kun::transfer;

namespace {
ModuleContract interface() { return {"scalar-scale/v1", "deterministic-scalars/v1", 1, CellId{2}, 1}; }

std::shared_ptr<const Germline> genome(uint64_t identity, double gain) {
    GraphDefinition g{GraphIdentity{identity}, GraphRevision{1}, SemanticProfile::StrictCore, 1,
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0}, {CellId{2}, CellType::OP_SUM}},
        {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0}, EdgeDelay::Immediate}}};
    InitialParameterSeeds s{
        {{CellId{1}, ParameterSlot::Param1, ContinuousValue{1}},
         {CellId{1}, ParameterSlot::Param2, UnusedParameter{}},
         {CellId{2}, ParameterSlot::Param1, UnusedParameter{}},
         {CellId{2}, ParameterSlot::Param2, UnusedParameter{}}},
        {{EdgeId{10}, gain}}};
    auto result = Germline::create(g, s);
    require(result.ok(), "demo germline compilation failed");
    return result.germline;
}
OffspringSpec spec(const Germline& germline, uint64_t id) {
    OffspringSpec s;
    s.organism_id = id; s.rng_seed = id;
    s.lifecycle_config.dormant_exit_resource = 1;
    s.resource_config.activity_scale = s.resource_config.transmission_scale = 1;
    for (const auto& c : germline.definition().cells)
        s.resource_cells.push_back({c.id, ResourceCompartmentId{1}, 100, 100, 0});
    s.resource_compartments.push_back({ResourceCompartmentId{1}, 0});
    return s;
}
std::unique_ptr<Phenotype> born(uint64_t id, double gain) {
    auto g = genome(id, gain);
    auto result = g->spawn_offspring(spec(*g, id));
    require(result.ok(), "demo fresh phenotype creation failed");
    return std::move(result.phenotype);
}
EvaluationProtocol protocol(double expected_gain) {
    EvaluationProtocol p{"scale-gain-" + std::to_string(expected_gain) + "/v1",
                         interface().environment, 1e-12, {}, {}};
    // Explicit versioned seed-to-input mapping. OOD has unseen magnitudes.
    for (uint64_t seed : {11, 12, 101, 102}) {
        const double x = seed < 100 ? double(seed - 10) / 2.0 : double(seed - 100) + 2.0;
        EvaluationTrace trace{seed, {{{x}, {expected_gain * x}}, {{-x}, {-expected_gain * x}}}};
        (seed < 100 ? p.train : p.ood).push_back(std::move(trace));
    }
    return p;
}
struct Work {
    uint64_t evaluations{0}, failed_attempts{0}, executions{0}, cells{0}, edges{0};
    uint64_t cultural_extractions{0};
};
void step(Phenotype& p, double x, Work& work) {
    const auto result = p.step({&x, 1});
    ++work.executions;
    work.cells += p.runtime().plan()->cells().size();
    work.edges += p.runtime().plan()->edges().size();
    require(result.ok(), "demonstrator native step failed");
}
void set_weight(Phenotype& p, EdgeId edge, double gain) {
    for (const auto& parameter : p.runtime().parameters())
        if (parameter.binding.kind == ParameterBindingKind::EdgeWeight && parameter.binding.edge == edge) {
            require(p.runtime().set_parameter(parameter.binding, ContinuousValue{gain}).ok(),
                    "demo learning boundary rejected parameter");
            return;
        }
    throw KnowledgeError("demo learning edge missing");
}
struct Search {
    double selected_gain{0};
    bool success{false};
    uint64_t attempts{0}, training_executions{0};
};
Search learn(Phenotype& phenotype, EdgeId edge, double start, double expected_gain, Work& work) {
    Search result;
    const auto p = protocol(expected_gain);
    // Identical target/splits/budget/increment for no-book, wrong-book and B.
    for (unsigned trial = 0; trial < 6; ++trial) {
        const double gain = start + 0.5 * trial;
        set_weight(phenotype, edge, gain);
        ++result.attempts; ++work.evaluations;
        auto parameters = std::make_shared<const InitialParameterValues>(
            std::vector<InitialParameterValue>(phenotype.runtime().parameters().begin(),
                                                phenotype.runtime().parameters().end()));
        bool passed = true;
        for (const auto& trace : p.train) {
            auto fresh = RuntimeState::create(phenotype.runtime().plan(), parameters);
            require(fresh.ok(), "training reconstruction failed");
            auto executor = CompiledExecutor::prepare(fresh.runtime->plan());
            require(executor.ok(), "training executor preparation failed");
            for (const auto& frame : trace.frames) {
                auto evaluated = executor.executor->step(*fresh.runtime, frame.inputs);
                ++result.training_executions; ++work.executions;
                work.cells += fresh.runtime->plan()->cells().size();
                work.edges += fresh.runtime->plan()->edges().size();
                require(evaluated.ok(), "native training execution failed");
                passed &= std::abs(fresh.runtime->cell_state(CellId{2})->output_val -
                                    frame.expected.front()) <= p.absolute_tolerance;
            }
        }
        result.selected_gain = gain; result.success = passed;
        if (passed) return result;
        ++work.failed_attempts;
    }
    return result;
}
void print_search(const Search& s) {
    std::cout << "{\"selected_gain\":" << s.selected_gain << ",\"success\":" << (s.success ? "true" : "false")
              << ",\"attempts\":" << s.attempts << ",\"training_executions\":" << s.training_executions << "}";
}
AdoptionRequest request(const Phenotype& p) {
    return {interface(), EdgeId{10}, p.runtime().tick(),
            {CellId{1}, ResourceCompartmentId{1}}, 1.0, 0.25, 1.0};
}

int demo(const std::string& path) {
    const auto experiment_start = std::chrono::steady_clock::now();
    require(!std::filesystem::exists(path), "demo requires a new database path; existing libraries are never overwritten");
    GermlineLibraryStore db(path);
    Work work;
    AdoptionWork adoption_work;
    const double x = -2;
    auto A = born(10101, 0.5);
    const auto a_search = learn(*A, EdgeId{10}, 0.5, 2.0, work);
    auto learned = KnowledgeModule::from_phenotype(*A, interface(),
        "A explicitly publishes learned parameters and complete native structure; no runtime memory");
    ++work.cultural_extractions;
    db.publish({"scale", "1"}, "A learned scalar mechanism", learned, {});
    require(db.evaluate({"scale", "1"}, protocol(2), "native-producer-evaluator").passed, "A evidence failed");
    const auto original = db.entry({"scale", "1"}).module.encode();

    auto no_book = born(90901, 0.5);
    const auto scratch = learn(*no_book, EdgeId{10}, 0.5, 3.0, work);
    auto scratch_module = KnowledgeModule::from_phenotype(*no_book, interface(),
        "No-book control: publish only after completing from-scratch search, for equal held-out validation");
    ++work.cultural_extractions;
    db.publish({"control", "no-book-final"}, "From-scratch control, not borrowed", scratch_module, {});
    const auto scratch_report = db.evaluate({"control", "no-book-final"}, protocol(3), "same-target-control-evaluator");
    require(scratch_report.passed, "from-scratch control held-out evaluation failed");
    auto wrong_template = KnowledgeModule::from_germline(*genome(90902, -2), interface(), "wrong-book-control");
    db.publish({"control", "wrong"}, "Wrong target but executable mechanism", wrong_template, {});
    db.evaluate({"control", "wrong"}, protocol(-2), "native-control-evaluator");
    auto wrong_borrow = db.borrow({"control", "wrong"}, interface(), "wrong-target-consumer");
    auto wrong_individual = wrong_borrow.module.spawn_birth(spec(wrong_borrow.module.germline(), 90903));
    const auto wrong_search = learn(*wrong_individual, EdgeId{10}, -2, 3.0, work);
    require(!db.evaluate({"control", "wrong"}, protocol(3), "target-mismatch-negative").passed,
            "wrong-book negative control unexpectedly passed");

    auto B = born(20202, 1);
    const auto b_germline = &B->germline();
    step(*B, x, work);
    const double b_before = B->runtime().cell_state(CellId{2})->output_val;
    auto incompatible = interface(); incompatible.environment = "incompatible/v1";
    bool rejected = false;
    try { db.borrow({"scale", "1"}, incompatible, "incompatible-control"); }
    catch (const KnowledgeError&) { rejected = true; }
    require(rejected, "incompatible retrieval accepted");
    const auto compatible = db.find(interface());
    require(std::find(compatible.begin(), compatible.end(), KnowledgeRef{"scale", "1"}) != compatible.end(),
            "compatible native lookup failed");
    const auto borrowed = db.borrow({"scale", "1"}, interface(), "unrelated-B");
    const auto receipt_b = adopt_at_cold_boundary(*B, borrowed, request(*B), {&x, 1}, &adoption_work);
    db.record_adoption(receipt_b);
    const double b_after = B->runtime().cell_state(CellId{2})->output_val;
    const auto b_search = learn(*B, receipt_b.new_edges.front(), 2, 3, work);
    step(*B, x, work);
    const double b_improved = B->runtime().cell_state(CellId{2})->output_val;
    auto selected_contract = interface(); selected_contract.output = receipt_b.inserted_cells.front();
    auto improved = KnowledgeModule::from_phenotype(*B, selected_contract,
        "B explicitly publishes the selected adapted receptor/unary mechanism; excludes downstream host and memory",
        {CellId{1}, receipt_b.inserted_cells.front()});
    ++work.cultural_extractions;
    db.publish({"scale", "2"}, "B adapted scalar mechanism", improved, {{"scale", "1"}});
    require(db.evaluate({"scale", "2"}, protocol(3), "native-consumer-evaluator").passed, "B evidence failed");
    require(db.entry({"scale", "1"}).module.encode() == original, "old immutable version changed");

    auto C = born(30303, 1);
    const auto c_germline = &C->germline();
    step(*C, x, work);
    const double c_before = C->runtime().cell_state(CellId{2})->output_val;
    const auto newer = db.borrow({"scale", "2"}, interface(), "unrelated-C");
    const auto receipt_c = adopt_at_cold_boundary(*C, newer, request(*C), {&x, 1}, &adoption_work);
    db.record_adoption(receipt_c);
    const double c_after = C->runtime().cell_state(CellId{2})->output_val;
    auto birth_control = newer.module.spawn_birth(spec(newer.module.germline(), 40404));
    step(*birth_control, x, work);
    const double birth_output = birth_control->runtime().cell_state(newer.module.contract().output)->output_val;
    auto poor = born(50505, 1);
    auto too_costly = request(*poor); too_costly.cell_cost = 1000;
    try { adopt_at_cold_boundary(*poor, newer, too_costly, {&x, 1}, &adoption_work); }
    catch (const KnowledgeError&) {}
    require(poor->runtime().tick() == 0, "resource rejection partially changed consumer");
    const auto& library_work = db.work();
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - experiment_start).count();
    std::cout << "{\"experiment\":\"native scalar mechanism transfer; research only\","
              << "\"same_target_budget\":6,\"train_seeds\":[11,12],\"ood_seeds\":[101,102],"
              << "\"A\":"; print_search(a_search);
    std::cout << ",\"B\":{\"id\":20202,\"before\":" << b_before << ",\"after\":" << b_after
              << ",\"improved\":" << b_improved << ",\"growth_paid\":" << receipt_b.paid_cost
              << ",\"germline_unchanged\":" << (&B->germline() == b_germline ? "true" : "false")
              << ",\"training_executions\":" << b_search.training_executions << ",\"search\":";
    print_search(b_search);
    std::cout << "},\"C\":{\"id\":30303,\"before\":" << c_before << ",\"after\":" << c_after
              << ",\"growth_paid\":" << receipt_c.paid_cost << ",\"germline_unchanged\":"
              << (&C->germline() == c_germline ? "true" : "false")
              << "},\"controls\":{\"no_book\":"; print_search(scratch);
    std::cout << ",\"wrong_book\":"; print_search(wrong_search);
    std::cout << ",\"incompatible_rejected\":true,\"birth_import_output\":" << birth_output
              << "},\"total\":{\"attempted_evaluations\":" << work.evaluations + library_work.evaluation_attempts
              << ",\"graph_executions\":" << work.executions + library_work.graph_executions + adoption_work.graph_executions
              << ",\"cell_visits\":" << work.cells + library_work.cell_visits + adoption_work.cell_visits
              << ",\"edge_visits\":" << work.edges + library_work.edge_visits + adoption_work.edge_visits
              << ",\"failed_attempts\":" << work.failed_attempts + library_work.failed_evaluations +
                    library_work.failed_retrievals + adoption_work.failed_attempts
              << ",\"publication_validations\":" << library_work.publication_validations
              << ",\"read_validations\":" << library_work.read_validations
              << ",\"artifact_bytes_validated\":" << library_work.artifact_bytes_validated
              << ",\"cultural_extractions\":" << work.cultural_extractions
              << ",\"retrieval_attempts\":" << library_work.retrieval_attempts
              << ",\"adoption_attempts\":" << adoption_work.attempts
              << ",\"adoption_graph_executions\":" << adoption_work.graph_executions
              << ",\"experiment_wall_ns\":" << elapsed_ns
              << ",\"growth_paid\":" << receipt_b.paid_cost + receipt_c.paid_cost
              << "},\"claim\":\"B used fewer training trace executions in this fixed grid; producer, validation, cold compilation and growth are not free; no general learning advantage or deployment certification\"}\n";
    return 0;
}
int inspect(const std::string& path) {
    require(std::filesystem::exists(path), "native library is absent");
    GermlineLibraryStore db(path);
    const auto v2 = db.borrow({"scale", "2"}, interface(), "native-process-reopen");
    auto fresh = v2.module.fresh_runtime();
    const double x = -2;
    auto executor = CompiledExecutor::prepare(fresh->plan());
    require(executor.ok() && executor.executor->step(*fresh, {&x, 1}).ok(), "reopened native execution failed");
    std::cout << "{\"schema_version\":3,\"content_digest\":\"" << v2.content_digest
              << "\",\"fresh_output\":" << fresh->cell_state(v2.module.contract().output)->output_val << "}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: knowledge_transfer_demo demo|inspect DATABASE");
        if (std::string(argv[1]) == "demo") return demo(argv[2]);
        if (std::string(argv[1]) == "inspect") return inspect(argv[2]);
        throw KnowledgeError("unknown native demonstration command");
    } catch (const std::exception& e) {
        std::cerr << "R8: " << e.what() << '\n';
        return 1;
    }
}

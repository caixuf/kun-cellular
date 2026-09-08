#pragma once
#include "tasks/transfer/knowledge_module.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include <chrono>

namespace kun::transfer {

struct EvaluationFrame {
    std::vector<double> inputs;
    std::vector<double> expected;
};
struct EvaluationTrace {
    uint64_t seed;
    std::vector<EvaluationFrame> frames;
};
struct EvaluationProtocol {
    std::string protocol_id;
    std::string environment;
    double absolute_tolerance{0};
    std::vector<EvaluationTrace> train;
    std::vector<EvaluationTrace> ood;
};
struct EvaluationReport {
    bool passed{false};
    uint64_t graph_executions{0};
    uint64_t cell_visits{0};
    uint64_t edge_visits{0};
    uint64_t elapsed_ns{0};
    std::string content_digest, manifest_digest, report_digest;
    wire::Bytes bytes;
};

inline wire::Bytes encode_protocol(const EvaluationProtocol& p, const ModuleContract& contract) {
    require(!p.protocol_id.empty() && p.environment == contract.environment,
            "evaluation protocol/environment mismatch");
    require(std::isfinite(p.absolute_tolerance) && p.absolute_tolerance >= 0,
            "invalid evaluation tolerance");
    require(!p.train.empty() && !p.ood.empty(), "nonempty train and OOD seed manifests required");
    std::set<uint64_t> seen;
    wire::Writer w;
    w.text("KUN-TRACE-MANIFEST/v1"); w.text(p.protocol_id);
    w.text(p.environment); w.real(p.absolute_tolerance);
    for (const auto* split : {&p.train, &p.ood}) {
        w.integer(split->size());
        for (const auto& trace : *split) {
            require(seen.insert(trace.seed).second, "seed manifests must be unique and disjoint");
            require(!trace.frames.empty(), "empty evaluation trace");
            w.integer(trace.seed); w.integer(trace.frames.size());
            for (const auto& frame : trace.frames) {
                require(frame.inputs.size() == contract.input_count && frame.expected.size() == 1,
                        "trace interface/expected output mismatch");
                w.integer(frame.inputs.size());
                for (auto value : frame.inputs) w.real(value);
                w.real(frame.expected.front());
            }
        }
    }
    return std::move(w.bytes);
}

// The library accepts protocols, not caller-authored 'passed' attestations.
inline EvaluationReport evaluate_native(const KnowledgeModule& module, const EvaluationProtocol& p) {
    const auto start = std::chrono::steady_clock::now();
    const auto manifest = encode_protocol(p, module.contract());
    EvaluationReport result;
    result.content_digest = wire::digest(module.encode());
    result.manifest_digest = wire::digest(manifest);
    result.passed = true;
    wire::Writer observations;
    for (const auto* split : {&p.train, &p.ood}) {
        for (const auto& trace : *split) {
            auto compiled_state = module.fresh_runtime();
            auto reference_state = module.fresh_runtime();
            auto prepared = CompiledExecutor::prepare(compiled_state->plan());
            require(prepared.ok(), "evaluation executor preparation failed");
            for (const auto& frame : trace.frames) {
                const auto native = prepared.executor->step(*compiled_state, frame.inputs);
                const auto reference = ReferenceExecutor{}.step(*reference_state, frame.inputs);
                result.graph_executions += 2;
                result.cell_visits += 2 * module.germline().graph()->cells().size();
                result.edge_visits += 2 * module.germline().graph()->edges().size();
                bool passed = native.ok() && reference.ok();
                const double actual = compiled_state->cell_state(module.contract().output)->output_val;
                const double control = reference_state->cell_state(module.contract().output)->output_val;
                passed = passed && std::isfinite(actual) && std::isfinite(control) &&
                    std::abs(actual - control) <= p.absolute_tolerance &&
                    std::abs(actual - frame.expected.front()) <= p.absolute_tolerance;
                result.passed &= passed;
                observations.integer(trace.seed);
                observations.integer(native.ok()); observations.integer(reference.ok());
                observations.real(std::isfinite(actual) ? actual : 0.0);
                observations.real(std::isfinite(control) ? control : 0.0);
                observations.integer(passed);
                observations.text(native.error ? std::string(native.error->reason) : "");
                observations.text(reference.error ? reference.error->reason : "");
            }
        }
    }
    result.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    wire::Writer report;
    report.text("KUN-RESEARCH-TRACE-REPORT/v1");
    report.text(result.content_digest); report.text(result.manifest_digest);
    report.text(p.environment); report.text(p.protocol_id);
    report.integer(manifest.size());
    report.bytes.insert(report.bytes.end(), manifest.begin(), manifest.end());
    report.integer(result.passed); report.integer(result.graph_executions);
    report.integer(result.cell_visits); report.integer(result.edge_visits);
    report.integer(result.elapsed_ns);
    report.integer(observations.bytes.size());
    report.bytes.insert(report.bytes.end(), observations.bytes.begin(), observations.bytes.end());
    result.bytes = std::move(report.bytes);
    result.report_digest = wire::digest(result.bytes);
    return result;
}

struct StoredReport {
    EvaluationReport report;
    EvaluationProtocol protocol;
};

inline StoredReport validate_stored_report(const wire::Bytes& bytes, const KnowledgeModule& module) {
    wire::Reader r{bytes};
    require(r.text() == "KUN-RESEARCH-TRACE-REPORT/v1", "unsupported evidence report");
    StoredReport stored;
    auto& report = stored.report;
    report.content_digest = r.text(); report.manifest_digest = r.text();
    require(report.content_digest == wire::digest(module.encode()), "report/content identity mismatch");
    const auto environment = r.text(); const auto protocol_id = r.text();
    const auto manifest_size = r.count(1);
    const wire::Bytes manifest(r.bytes.begin() + r.position, r.bytes.begin() + r.position + manifest_size);
    r.position += manifest_size;
    require(wire::digest(manifest) == report.manifest_digest, "report/manifest digest mismatch");
    wire::Reader m{manifest};
    require(m.text() == "KUN-TRACE-MANIFEST/v1", "unsupported trace manifest");
    auto& p = stored.protocol;
    p.protocol_id = m.text(); p.environment = m.text(); p.absolute_tolerance = m.real();
    require(p.environment == environment && p.protocol_id == protocol_id, "report protocol/environment mismatch");
    for (auto* split : {&p.train, &p.ood}) {
        const auto count = m.count(16);
        for (std::size_t i = 0; i < count; ++i) {
            EvaluationTrace trace{m.integer(), {}};
            const auto frames = m.count(16);
            for (std::size_t j = 0; j < frames; ++j) {
                EvaluationFrame frame;
                const auto width = m.count(8);
                for (std::size_t k = 0; k < width; ++k) frame.inputs.push_back(m.real());
                frame.expected.push_back(m.real());
                trace.frames.push_back(std::move(frame));
            }
            split->push_back(std::move(trace));
        }
    }
    m.end();
    require(encode_protocol(p, module.contract()) == manifest, "noncanonical evidence manifest");
    const auto passed = r.integer(); require(passed <= 1, "invalid report result");
    report.passed = passed != 0;
    report.graph_executions = r.integer(); report.cell_visits = r.integer();
    report.edge_visits = r.integer(); report.elapsed_ns = r.integer();
    const auto observation_size = r.count(1);
    wire::Reader observations{r.bytes.subspan(r.position, observation_size)};
    r.position += observation_size; r.end();
    bool all_passed = true;
    uint64_t frames = 0;
    for (const auto* split : {&p.train, &p.ood}) {
        for (const auto& trace : *split) {
            for (const auto& frame : trace.frames) {
                ++frames;
                require(observations.integer() == trace.seed, "observation/seed mismatch");
                const auto native_ok = observations.integer(), reference_ok = observations.integer();
                require(native_ok <= 1 && reference_ok <= 1, "invalid executor result");
                const double actual = observations.real(), reference = observations.real();
                const auto result = observations.integer();
                const auto native_error = observations.text(), reference_error = observations.text();
                require(!native_ok || native_error.empty(), "inconsistent compiled error");
                require(!reference_ok || reference_error.empty(), "inconsistent reference error");
                const bool expected = native_ok && reference_ok &&
                    std::abs(actual - reference) <= p.absolute_tolerance &&
                    std::abs(actual - frame.expected.front()) <= p.absolute_tolerance;
                require(result <= 1 && bool(result) == expected, "report output verdict mismatch");
                all_passed &= expected;
            }
        }
    }
    observations.end();
    require(report.passed == all_passed && report.graph_executions == frames * 2 &&
            report.cell_visits == frames * 2 * module.germline().graph()->cells().size() &&
            report.edge_visits == frames * 2 * module.germline().graph()->edges().size(),
            "evidence summary/work count mismatch");
    report.report_digest = wire::digest(bytes);
    return stored;
}
} // namespace kun::transfer

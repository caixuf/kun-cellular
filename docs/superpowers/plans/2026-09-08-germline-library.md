# Germline Knowledge Library Implementation Plan

## Delivered scope

The native R8 implementation is complete for the deliberately narrow,
research-only transfer path:

- `tasks/transfer/knowledge_module.hpp` provides canonical typed native graph
  artifacts with stable IDs, initial parameter seeds, semantic/interface/
  environment contracts, and producer provenance. Publication is explicit and
  never captures runtime state or resources.
- `tasks/transfer/knowledge_evidence.hpp` evaluates each artifact with the
  native and reference executors over disjoint nonempty train/OOD seed
  manifests accepted by the bounded `KUN-R8-APPROVED/` research protocol
  registry. Reports bind content, environment, protocol, seeds, outputs, and
  work counts; failed reports are retained.
- `tasks/transfer/germline_library.hpp` provides SQLite schema 3 as the C++
  authority: immutable versions, parent refs, candidate/research-validated
  status, append-only events, and transactional borrow/evidence/adoption
  records.
- `tasks/transfer/knowledge_adoption.hpp` supports fresh offspring import and
  live cold-boundary composition through the existing growth/resource payment
  path. The host edge is an explicit composition/remapping boundary; arbitrary
  semantic subgraph discovery is not implemented.
- `tools/germline_library.py` is read-only display glue. It never creates or
  writes SQLite tables, and JSON is only an explicit snapshot/export format.
- `tasks/transfer/knowledge_transfer_demo.cpp` and the focused tests provide a
  measurable native A→B→C run with no-book, wrong-book, incompatible-contract,
  resource-failure, and birth-import controls.

## Deliberate limitations

This is not a production deployment certification system. Replay, Shadow,
multi-environment attestation, and a deployment-certified status are not
implemented and must not be inferred from the research validation evidence.
The live composition path currently supports one explicitly validated
receptor/unary motif; larger compatible modules use fresh birth import.
Native trace work is reported as attempted work in this phase; only live
structural adoption is charged through the resource ledger.

# R8 Native Knowledge Transfer Status

R8 is a peripheral, auditable knowledge layer analogous to books and cumulative
culture. It lets an unrelated organism reuse a validated native executable
module, then publish a distinct immutable child version. It does not modify the
generic cellular substrate.

## Implemented contract

- `tasks/transfer/knowledge_module.hpp` defines a typed, canonical
  `KnowledgeModule`: native graph definition, initial parameter seeds,
  interface/environment/timing/semantic contract, producer lineage, and
  explicit publication declaration. Arbitrary bytes, unknown schema or
  primitives, malformed graphs, and incompatible contracts are rejected.
- Stable graph/cell/edge IDs are preserved in artifacts. Live adoption has an
  explicit host-edge composition boundary and fresh ID allocation; it does not
  attempt arbitrary semantic subgraph discovery. Whole compatible modules can
  instead be imported as fresh offspring.
- `from_phenotype` is explicit cultural publication. It exports only selected
  live parameters and selected structure after fresh native reconstruction; it
  never serializes runtime state, resources, optimizer state, or execution tape.
- `tasks/transfer/germline_library.hpp` is the C++ control-plane authority.
  SQLite schema 3 stores immutable object versions, parent references,
  content-bound evidence, and append-only events. Python never creates tables
  or mutates the database.
- Candidate objects become `research-validated` only after deterministic native
  evaluation over disjoint nonempty train and OOD seed manifests accepted by
  the built-in bounded `KUN-R8-APPROVED/` research protocol registry. Failed
  evaluations remain stored and keep the object at `candidate`. There is no
  deployment certification tier in this implementation.
- `knowledge_adoption.hpp` provides fresh birth import and live cold-boundary
  adoption through the existing graph editor/growth/resource-payment path.
  Preflight executes on a fork, failure leaves the target unchanged, and a
  successful adoption preserves the target germline and existing runtime state.
  Persisting the resulting receipt is a separate SQLite transaction from the
  live mutation; the receipt is immutable and retry-safe rather than pretending
  RAM and the database form one atomic transaction.
- `tools/germline_library.py` is a read-only projection. JSON is only an
  explicit display/export format; malformed native databases produce explicit
  errors. Legacy `library/motifs/*.json` entries remain separately labelled
  `legacy_motif`.

## Measured A→B→C demonstration

`knowledge_transfer_demo` uses real native graphs and executor outputs:

- A learns gain `2`, publishes `scale/1`, and passes train/OOD evaluation.
- Unrelated B borrows `scale/1`, changes output from `-2` to `-4` at the live
  boundary, pays `2.5` resource units, then publishes `scale/2` with parent
  `scale/1`; B's germline remains unchanged.
- Unrelated C borrows `scale/2` and reaches `-6` without copying B's ancestry or
  runtime state.
- Controls include no-book search, wrong-book failure, incompatible-contract
  rejection, failed low-resource adoption, and fresh birth import.

The demo reports producer/evaluation/adoption work, failed attempts, bytes
validated, and wall time. Native trace work is reported as attempted work; only
live structural adoption is charged through the resource ledger in this phase.
It makes no general speedup claim and does not claim production Replay, Shadow,
or deployment certification.

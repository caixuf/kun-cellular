# Evidence-First Truthfulness Remediation Design

## Goal

Make every public capability claim in KunCellular traceable to reproducible repository evidence. The project may continue to expose ambitious research targets, but it must distinguish verified behavior, partial prototypes, unverified claims, and failed evaluations. The C/C++ cellular substrate remains unchanged; remediation is limited to evidence tooling, metadata, documentation, runtime presentation, and task-layer validation.

## Current evidence baseline

The audit establishes four categories:

- **Verified substrate:** C/C++ primitives, CSR execution, mmap loading, and the existing 34/34 local ctest result.
- **Verified small tasks:** domain-zoo controllers in the 8–20-cell range with train/ID/OOD measurements where reports exist.
- **Partial or contradicted assets:** nominal million/100-million scale entries whose SDSC-BIN headers contain only hundreds or thousands of cells.
- **Unsupported or failed claims:** random 1M export, statistically inadequate quant results, missing benchmark files, contradictory OOD gates, and payloads with unexplained trailing bytes.

No UI, README, or manifest field may upgrade a claim beyond the strongest evidence category.

## Architecture

### 1. Evidence extractor

Add a standard-library Python audit tool under `tools/` that reads, without modifying, the manifest, README, checkpoint headers and metadata, report JSON, run JSON, test paths, and binary file sizes. It emits a deterministic JSON evidence ledger containing:

- checkpoint header values and payload boundary calculations;
- cell/synapse counts from the binary versus manifest claims;
- whether a checkpoint has a training provenance record;
- report, test, and benchmark source paths;
- metric values, sample sizes, split labels, and verdicts;
- missing-file and contradictory-gate findings;
- a status per claim: `verified`, `partial`, `unverified`, or `failed`.

The extractor must never infer training or capability from a filename. Missing evidence is an explicit finding, not a success-shaped default.

### 2. Claim registry and consistency gate

Use the manifest as the machine-readable claim registry, adding explicit fields for:

- `claim_status`;
- `evidence_sources`;
- `checkpoint_cells` and `checkpoint_synapses`;
- `training_provenance`;
- `evaluation_split`, `sample_count`, and `verdict`;
- `scale_status` (`exported`, `nominal`, or `unverified`).

The audit tool validates README and manifest scale/count claims against binary headers, rejects references to missing tests, flags `gate: true` with a zero or failed OOD result, and reports any bytes outside the declared SDSC-BIN payload. It does not silently repair metadata.

### 3. Public presentation contract

The README, library API, and frontend must consume the same status vocabulary. A nominal target such as 100M is displayed as a target only when the checkpoint is smaller; a random export is labelled `unverified`, and a failed report remains `failed`. Capability prose must link to evidence sources and show sample size and split wherever a performance metric is shown.

The knowledge library should stop generating promotional book entries. Its identity and evidence shelves are populated from the ledger, including negative findings. The frontend must not show a green success badge for missing, failed, or statistically inadequate evidence.

### 4. Smart-driving regression track

ADAS is a separate task-layer workstream. Before any retraining, freeze the current baseline and reproduce the reported straight-segment instability. Then make the simulator and trainer share one environment contract, including signed CTE derivative, arc-length lookahead, actuator dynamics, and deterministic disturbance settings. Evaluation reports must include straight-segment average/max CTE, signed-CTE zero crossings, recovery time, full-track completion, and comparison to the previous checkpoint. A new checkpoint is promoted only if it improves the frozen baseline without regressing curve behavior.

This workstream cannot be declared successful from a single visual demo or aggregate completion rate.

## Validation and failure handling

Required checks are deterministic and offline:

1. Run the evidence extractor and compare its JSON to a checked-in schema.
2. Run binary boundary and header consistency checks for every manifest checkpoint.
3. Run documentation/manifest claim consistency checks.
4. Run existing substrate tests unchanged.
5. Run ADAS baseline and candidate evaluations with fixed seeds and retain both reports.
6. Run frontend checks only after the public status fields are wired.

Any missing report, missing test path, invalid gate, unexplained trailing bytes, or claim/evidence mismatch is a visible audit failure. The tooling must return a non-zero exit code for CI, while still emitting the complete ledger for diagnosis.

## Delivery order

1. Freeze and record the audit ledger and baseline metrics.
2. Implement the extractor and consistency checks.
3. Correct manifest and README claims using ledger statuses.
4. Wire the library/frontend status contract and remove promotional fallbacks.
5. Repair and re-evaluate ADAS environment behavior; promote only a measured improvement.
6. Resume rendering and knowledge-shelf improvements using truthful metadata.

Out of scope: changing `include/kun/cellular/` semantics, manufacturing replacement benchmarks, deleting failed reports, or converting nominal scale targets into claimed realized scale.

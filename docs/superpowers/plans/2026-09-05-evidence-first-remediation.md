# Evidence-First Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KunCellular's public capability claims reproducible and status-labelled, then restore intelligent-driving quality and only afterward improve visualization of truthful assets.

**Architecture:** A read-only Python evidence ledger becomes the source for manifest, README, library API, and frontend status. Binary/header, report, test-path, and gate consistency checks fail loudly in CI. ADAS evaluation remains in the peripheral task layer and uses a frozen baseline plus a shared trainer/runtime environment contract; the C/C++ substrate is unchanged.

**Tech Stack:** Python 3 standard library, existing JSON/SDSC-BIN readers, existing pytest/ctest/Node checks, vanilla ES modules, Three.js, Git.

---

## Task 1: Freeze the current evidence baseline

**Files:**
- Create: `tools/evidence_ledger.py`
- Create: `tools/evidence_schema.json`
- Create: `checkpoints/evidence_baseline.json`
- Test: `tests/test_evidence_ledger.py`

- [ ] **Step 1: Write tests for deterministic binary and claim extraction**

Add tests that construct a temporary SDSC-BIN v2 header and assert that `read_checkpoint_fact()` returns the header cell count, synapse count, declared payload end, actual file size, and a `trailing_bytes` finding. Add a manifest fixture with a missing test path and assert it is reported as `missing_source`, not treated as valid evidence.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m pytest tests/test_evidence_ledger.py -q
```

Expected: collection or import failure because `tools.evidence_ledger` does not exist.

- [ ] **Step 3: Implement read-only evidence extraction**

Implement these typed functions in `tools/evidence_ledger.py`:

```python
def read_checkpoint_fact(path: Path) -> dict: ...
def collect_manifest_claims(manifest_path: Path) -> list[dict]: ...
def collect_source_facts(repo_root: Path, claim: dict) -> dict: ...
def classify_claim(claim: dict, facts: dict) -> str: ...
def build_ledger(repo_root: Path) -> dict: ...
```

Use the existing SDSC-BIN v2 layout (`<IIIIIIQQQQQQ`, 72-byte header; 4-byte cells; CSR offsets; float32 coordinates). Never generate replacement cells, infer training from filenames, or mutate repository files. Emit stable sorted arrays and the status values `verified`, `partial`, `unverified`, and `failed`. Add `--repo-root`, `--output`, and `--check` CLI options; `--check` exits non-zero if any claim is `failed` or has a missing required source.

- [ ] **Step 4: Add and validate the JSON schema**

Define required top-level keys `schema_version`, `generated_at`, `claims`, and `findings`; require each claim to contain `organism_id`, `status`, `sources`, `checkpoint`, and `metrics`. In tests, validate the generated ledger shape without asserting wall-clock timestamps.

- [ ] **Step 5: Run tests and create the baseline ledger**

Run:

```bash
python3 -m pytest tests/test_evidence_ledger.py -q
python3 tools/evidence_ledger.py --repo-root . --output checkpoints/evidence_baseline.json
```

Expected: focused tests pass and the baseline contains explicit findings for known random, missing, contradictory, and trailing-byte cases.

- [ ] **Step 6: Commit the isolated evidence extractor**

```bash
git add tools/evidence_ledger.py tools/evidence_schema.json tests/test_evidence_ledger.py checkpoints/evidence_baseline.json
git commit -m "feat: add reproducible evidence ledger"
```

## Task 2: Add claim consistency and CI gates

**Files:**
- Modify: `tools/evidence_ledger.py`
- Create: `tools/ci/check_evidence_claims.py`
- Create: `tests/test_evidence_claims.py`
- Modify: existing CI entrypoint only if it already invokes repository checks

- [ ] **Step 1: Write failing checks for the audit findings**

Cover: manifest scale differing from header counts; README references to missing test files; `gate: true` with OOD success `0`; a declared payload ending before file EOF; and a report with `verdict: FAIL`. Assert each produces a finding with a stable code and non-zero CLI status.

- [ ] **Step 2: Implement explicit consistency rules**

Use finding codes `COUNT_MISMATCH`, `MISSING_SOURCE`, `INVALID_GATE`, `TRAILING_PAYLOAD`, `FAILED_EVALUATION`, and `UNSUPPORTED_CLAIM`. Do not auto-correct data. `check_evidence_claims.py` must print a concise table and return `1` for any error-level finding, `0` otherwise.

- [ ] **Step 3: Run focused checks**

```bash
python3 -m pytest tests/test_evidence_claims.py -q
python3 tools/ci/check_evidence_claims.py --repo-root .
```

Expected: tests pass; the current repository reports known discrepancies rather than silently succeeding. Record the initial finding list in `checkpoints/evidence_baseline.json`.

- [ ] **Step 4: Wire the check into the existing CI convention**

Add the command to the existing architecture/frontend check runner without changing its failure semantics. Keep `include/kun/cellular/` outside the change.

- [ ] **Step 5: Commit the consistency gate**

```bash
git add tools/ci/check_evidence_claims.py tools/evidence_ledger.py tests/test_evidence_claims.py
git commit -m "ci: enforce evidence and claim consistency"
```

## Task 3: Correct manifest and README claims

**Files:**
- Modify: `models/business_lifeforms/manifest.json`
- Modify: `README.md`
- Modify: `checkpoints/evidence_baseline.json`
- Test: `tests/test_manifest_evidence_contract.py`

- [ ] **Step 1: Add contract tests**

Assert every manifest organism has `claim_status`, `scale_status`, `checkpoint_cells`, `checkpoint_synapses`, `evidence_sources`, `evaluation_split`, `sample_count`, and `verdict`; assert nominal targets are not presented as realized checkpoint counts.

- [ ] **Step 2: Populate metadata from the ledger**

For each organism, set measured header counts separately from target scale. Set the random 1M export to `unverified`, the failed quant report to `failed`, and entries with only representative microcolumns to `partial`/`nominal`. Preserve failed reports and link them through `evidence_sources`.

- [ ] **Step 3: Rewrite README capability tables**

Replace unsupported “100M cells”, “+148.83%”, “82.5%”, and “19.06ns” assertions with measured values, target labels, sample sizes, splits, and source paths. State clearly that small domain-zoo tasks are verified while large-scale and unsupported claims are not.

- [ ] **Step 4: Run contract and consistency checks**

```bash
python3 -m pytest tests/test_manifest_evidence_contract.py -q
python3 tools/ci/check_evidence_claims.py --repo-root .
```

Expected: no unsupported public claim remains; any intentionally partial asset is labelled rather than hidden.

- [ ] **Step 5: Commit documentation and metadata correction**

```bash
git add README.md models/business_lifeforms/manifest.json checkpoints/evidence_baseline.json tests/test_manifest_evidence_contract.py
git commit -m "docs: align public claims with measured evidence"
```

## Task 4: Expose evidence status through the library and frontend

**Files:**
- Modify: `tools/cellular_live_backend.py`
- Modify: `frontend/cellular/organism_library.js`
- Modify: `frontend/cellular/app.js`
- Modify: `frontend/cellular/network_sync.js`
- Test: `tests/test_library_evidence_status.py`

- [ ] **Step 1: Test status propagation**

Call the library serialization path with verified, partial, unverified, and failed ledger entries. Assert the API preserves status, source paths, sample count, split, measured counts, and findings; assert no fallback creates exactly three promotional entries.

- [ ] **Step 2: Load the ledger as a read-only index**

Extend `SiliconLifeformLibrary.reload_books()` to read the ledger and manifest metadata, keeping existing unrelated working-tree behavior intact. Return six shelves (identity, contract, evidence, lineage, motifs, literature) with real counts and negative findings.

- [ ] **Step 3: Render status honestly**

In `organism_library.js`, render status badges and evidence links; in `app.js`/`network_sync.js`, display measured cells separately from nominal target scale. Do not show green success UI for `partial`, `unverified`, `failed`, or missing evidence.

- [ ] **Step 4: Validate frontend syntax and API tests**

```bash
python3 -m pytest tests/test_library_evidence_status.py -q
node --check frontend/cellular/organism_library.js
node --check frontend/cellular/app.js
node --check frontend/cellular/network_sync.js
```

- [ ] **Step 5: Commit the public status contract**

```bash
git add tools/cellular_live_backend.py frontend/cellular/organism_library.js frontend/cellular/app.js frontend/cellular/network_sync.js tests/test_library_evidence_status.py
git commit -m "feat: expose evidence status in lifeform library"
```

## Task 5: Freeze and improve the ADAS environment contract

**Files:**
- Modify: `tools/cellular_live_backend.py`
- Modify: `tools/train_adas_natural_champion.py`
- Test: `tests/test_adas_environment_contract.py`
- Create: `checkpoints/adas_baseline_regression.json`

- [ ] **Step 1: Add regression tests for signed derivative and distance lookahead**

Assert a signed CTE crossing produces a derivative with the correct sign, and that equal lookahead distance yields comparable preview behavior despite non-uniform track point spacing. Assert checkpoint cell opcode parsing uses 4-byte cells at offset `0`.

- [ ] **Step 2: Reproduce and freeze the old baseline**

Run the existing simulator with fixed seed/settings and save straight-segment average/max CTE, zero crossings, recovery time, curve metrics, completion, and checkpoint hash to `checkpoints/adas_baseline_regression.json`. Do not overwrite the existing champion before this file exists.

- [ ] **Step 3: Implement shared environment behavior**

Replace absolute-CTE differencing with signed CTE; replace `best_idx + 14` with arc-length lookahead using the existing track geometry; preserve trainer/runtime parity; add bounded actuator rate and first-order lag only in the peripheral environment; keep all changes out of `include/kun/cellular/`.

- [ ] **Step 4: Run focused environment tests**

```bash
python3 -m pytest tests/test_adas_environment_contract.py -q
```

Expected: all contract tests pass, including the 4-byte checkpoint parser assertion.

- [ ] **Step 5: Commit environment-only changes**

```bash
git add tools/cellular_live_backend.py tools/train_adas_natural_champion.py tests/test_adas_environment_contract.py checkpoints/adas_baseline_regression.json
git commit -m "fix: align ADAS training and live environment contracts"
```

## Task 6: Retrain and promote only a measured ADAS improvement

**Files:**
- Modify: `checkpoints/adas_track_champion.bin`
- Create: `checkpoints/adas_track_champion_regression.json`
- Modify: `models/business_lifeforms/manifest.json`

- [ ] **Step 1: Train with fixed reproducibility settings**

Run the existing trainer with the agreed fixed seed, 80 generations, and population 24. Preserve logs and do not replace the checkpoint until evaluation completes.

- [ ] **Step 2: Evaluate candidate and compare to baseline**

Require straight average CTE `< 1.0`, straight max CTE `< 3.0`, fewer than 6 signed-CTE zero crossings per lap, no regression in curve metrics, and full-track completion. If any threshold fails, retain the baseline and mark the candidate `failed`; do not claim improvement.

- [ ] **Step 3: Promote only if all gates pass**

Write the regression JSON with seed, trainer version/hash, metrics, split, and verdict. Replace the checkpoint only after the promotion gate passes, then update manifest evidence fields.

- [ ] **Step 4: Run existing ADAS and substrate validation**

```bash
python3 -m pytest tests/test_adas_scales.py -q
ctest --test-dir build --output-on-failure
```

Expected: ADAS tests and the existing substrate suite pass; no C/C++ base files change.

- [ ] **Step 5: Commit only a passing promotion**

```bash
git add checkpoints/adas_track_champion.bin checkpoints/adas_track_champion_regression.json models/business_lifeforms/manifest.json
git commit -m "feat: promote evidence-backed ADAS champion"
```

## Task 7: Resume truthful rendering and real connectivity

**Files:**
- Modify: `tools/cellular_live_backend.py`
- Modify: `frontend/cellular/manifold_system.js`
- Modify: `frontend/cellular/lod_system.js`
- Modify: `frontend/cellular/organ_view.js`
- Modify: `frontend/cellular/synapse_view.js`
- Modify: `frontend/cellular/config.js`
- Create: `frontend/cellular/palette.js`
- Modify: `models/business_lifeforms/manifest.json`
- Test: `tests/test_truthful_manifold_payload.py`

- [ ] **Step 1: Add payload tests**

Assert manifold payloads contain only checkpoint cells, reject missing coordinates instead of fabricating a lattice, and derive synapse segments from CSR rather than `np.random`. Assert opcode index `0..25` matches the canonical palette.

- [ ] **Step 2: Remove backend fabrication**

Delete procedural cell and random synapse fallbacks; return an explicit unavailable/error response when a truthful payload cannot be built. Include real counts and `synapses_truncated` metadata.

- [ ] **Step 3: Unify frontend palette and scale display**

Create `palette.js` as the only opcode palette source, use real-vs-nominal counts in HUD, and remove fixed “30,000 stars”/organism-scale fallbacks. Keep the backend payload free of presentation colors.

- [ ] **Step 4: Fix point cloud, synapse, and LOD behavior**

Use density-derived point size and premultiplied alpha, render real CSR LineSegments with bounded sampling, remove legacy random LOD clouds, and preserve CI-required `MIN_CELL_PIXELS`, `solidMaxDist`, exact distance guard, and `intersectsSphere` statements.

- [ ] **Step 5: Run targeted checks**

```bash
python3 -m pytest tests/test_truthful_manifold_payload.py -q
python3 tools/ci/check_architecture_discipline.py
python3 tools/ci/check_frontend.py
```

Expected: no random/fallback payload path remains and architecture/frontend gates pass.

- [ ] **Step 6: Commit rendering changes**

```bash
git add tools/cellular_live_backend.py frontend/cellular frontend/cellular/palette.js models/business_lifeforms/manifest.json tests/test_truthful_manifold_payload.py
git commit -m "fix: render only evidence-backed cellular assets"
```

## Task 8: Final audit and handoff

**Files:**
- Modify: `checkpoints/evidence_baseline.json`
- Create: `checkpoints/final_evidence_ledger.json`
- Create: `docs/EVIDENCE_STATUS.md`

- [ ] **Step 1: Run the complete verification set**

```bash
python3 tools/evidence_ledger.py --repo-root . --output checkpoints/final_evidence_ledger.json --check
python3 tools/ci/check_evidence_claims.py --repo-root .
python3 tools/ci/check_architecture_discipline.py
python3 tools/ci/check_frontend.py
ctest --test-dir build --output-on-failure
```

- [ ] **Step 2: Document remaining limitations**

Generate `docs/EVIDENCE_STATUS.md` from the final ledger, including verified substrate, verified small tasks, partial scale targets, failed evaluations, unverified random assets, sample sizes, and exact reproduction commands.

- [ ] **Step 3: Check unrelated worktree changes**

Review `git status --short` and ensure commits contain only files belonging to this plan. Do not reset, checkout, or overwrite pre-existing modifications.

- [ ] **Step 4: Commit the final ledger and status report**

```bash
git add checkpoints/final_evidence_ledger.json docs/EVIDENCE_STATUS.md
git commit -m "docs: publish final evidence status"
```

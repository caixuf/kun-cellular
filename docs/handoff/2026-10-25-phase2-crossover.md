# Handoff — Phase 2 crossover (Prefer A)

**Date:** 2026-10-25 Asia/Shanghai  
**Base tip:** `f09edc0c` / `f09edc0c2b035f544f7b55f56845b687ee7ac757` (Phase 1 `#10`)  
**Branch:** `feat/cellular-phase2-crossover`  
**Owners:** 焊枪 (L1 sexual + compat gate + tests) · 手术刀 (not escalated) · 颜料盒 (none)

## Surfaces landed (Prefer A)

| Surface | Sexual `cross_into` | Compat distance | Notes |
|---|---|---|---|
| **CorticalMacroArray** | Hardened (bool; shape reject leaves `out` unchanged) | Flatten columns → cell loci (`op_types` / `gains`) | Existing column 50/50 + axon rewire kept |
| **CellularOrganism (82-cell)** | **Added** — slot-aligned cell + synapse 50/50 | Cell loci (`CellType` / `param1`) | Syn contract: equal `cells.size()` + `synapses.size()`; champion path `cells==82` |

**Not claimed as Prefer-A A8:** legacy `ReplicableGenome` / `SpeciationMetric::sexual_crossover` (heap) — left untouched.

## Compatibility formula (parity vs SpeciationMetric)

Source mirror: `morphogenetic_population.hpp:25–48`.

```
distance = c1 * (disjoint_count / max_n) + c2 * avg_weight_diff
matching  = aligned index i with equal op/type
disjoint  = max_n - matching_count
avg_weight_diff = mean |weight_param| over matches; 1.0 if none
c1 = 1.0, c2 = 0.4
```

Adapters: `include/kun/cellular/population/compatibility_distance.hpp`  
Default threshold: **`speciation_threshold = 0.35`** (`EvolutionParams`, same as MorphogeneticEcosystem knob).  
Deme gate: before sexual `crossover_into`, require `mates_compatible(pa,pb,threshold)`; on fail **or** `cross_into==false` → asexual clone of parent A (reject/skip — **not** silent sexual success).

## Syn contract (82-organism)

- Parents must share `cells.size()` and `synapses.size()` (slot-aligned).  
- Offspring conserves both; Prefer-A champion path locks `cells.size()==82`.  
- Whole-slot take from A or B per cell / synapse; `reset_state` + `compile()`.  
- Steady-state zero-alloc: `out = a` then element assign reuses vector capacity when deme offspring slots already hold the shape.

## CI / local proof (2026-10-25 Asia/Shanghai, caixuf WSL)

```
ctest -R 'test_population_' → 9/9 PASS
  test_population_individual_traits ...... PASS  (A3–A7 + reject rewrite)
  test_population_buffer_reuse ........... PASS
  test_population_partial_selection ...... PASS
  test_population_parallel_determinism ... PASS
  test_population_batched_eval ........... PASS
  test_population_ecology_grid ........... PASS
  test_population_pool_ecology ........... PASS
  test_population_pool_manifest .......... PASS
  test_population_structured_wiring ...... PASS
```

Key new/rewritten coverage in `tests/test_population_individual_traits.cpp`:
- `test_incompatible_parents_reject` (was degrade/clone)
- `test_compat_distance_gate`
- `test_organism_82_sexual` (size / bit-identical / reject / capacity reuse)
- `test_cortical_zero_alloc_reuse`

## A1–A10

| ID | Verdict |
|---|---|
| A1 | PASS — branch off `f09edc0c`; untracked EMERGENCE/scripts left alone |
| A2 | PASS — cortical + 82-organism sexual landed |
| A3 | PASS — same-seed bit-identical (cortical + organism) |
| A4 | PASS — cells==82 + syn count; cortical columns/num_cells conserved |
| A5 | PASS — incompatible ≥ threshold or shape mismatch → reject/skip |
| A6 | PASS — `compatibility_distance` + formula documented |
| A7 | PASS — capacity-stable reuse on hot path |
| A8 | PASS — no MAP-Elites / NoveltyArchive / E-Gate / protect_window |
| A9 | PASS — win knives STOP; no DouZero/GPU/PSRO/ADAS/zoo/mutate; Phase3/4 not started |
| A10 | PASS — this handoff |

## NON-BLOCK / PARK

Legacy morphogenetic heap sexual · GPU py crossover · protect_window (P3) · FE none · untracked EMERGENCE/scripts.

## Escalate 手术刀

**No.** Prefer-A sexual + distance landed on existing CorticalMacroArray + CellularOrganism without a fourth genome ABI.

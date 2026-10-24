# Handoff — Phase 3 MAP-Elites / novelty (Prefer A)

**Date:** 2026-10-25 Asia/Shanghai  
**Base tip:** `2817e2f0` / `2817e2f0851c466b5204dd384f5e6e14600095a0` (Phase 2 `#11`)  
**Branch:** `feat/cellular-phase3-map-elites`  
**Owners:** 焊枪 (grid + protect_window + avalanche + E-Gate + tests) · 手术刀 (not escalated) · 颜料盒 (none)

## §七 product bet

Success = **novelty / openness** (archive coverage, gated E-tags).  
**Not** win-rate vs DouZero / PSRO / GPU. Win knives **STOP**. Phase 4 2-ply **PARK**.

## Surfaces landed (Prefer A)

| Surface | Path | Notes |
|---|---|---|
| MAP-Elites descriptor grid | `include/kun/cellular/population/map_elites_grid.hpp` | BD bins → elite by QD (coverage + novelty_weight×novelty); KNN ≠ done |
| Novelty stepping-stones | `include/kun/cellular/population/novelty_stepping_stone.hpp` | KNN wrap feeds grid; mid-goal not fitness-purged under NOVELTY/HYBRID |
| PopulationSpeciesNiche + protect_window | `include/kun/cellular/population/population_species_niche.hpp` | **Not** biosphere enum `SpeciesNiche` |
| Avalanche (minimal SOC) | `include/kun/cellular/population/avalanche_metric.hpp` | size histogram + branching; Lyapunov ≠ avalanche |
| Metrics bundle + handoff table | `include/kun/cellular/population/phase3_metrics.hpp` | Coverage · L1 distance · avalanche · Lyapunov companion |
| Anti-artifact harness | `include/kun/cellular/population/anti_artifact.hpp` | dual · ≥4×3 learn-scales · OOD+McNemar/paired |
| E-Gate | `include/kun/cellular/population/e_gate.hpp` | pre-register → anti-artifact → explainable → reproducible |
| E1 claim fixture | `docs/claims/E1-phase3-map-elites-fixture.md` | dry-run E1 |
| Tests | `tests/test_population_phase3_map_elites.cpp` | A1–A6 + A8 ban + A9 |

**Not claimed:** L0 heap `NoveltyArchive` as MAP-Elites; POET (#1); ASAL (#6); 200-cell probe; FE `LIFEFORM_SCALES` as learning scales; 4th genome ABI.

## Name disambiguation

| Name | Meaning |
|---|---|
| `kun::population::PopulationSpeciesNiche` | Prefer-A L1 niche + `protect_window` |
| `kun::SpeciesNiche` (`ecosystem_biosphere.hpp`) | Biosphere **enum** PRODUCER/HERBIVORE/… |
| `kun::SpeciesNiche` (`morphogenetic_population.hpp`) | Heap morphogenetic niche struct |

## Claim → metrics → E-Gate stage

| Claim | Metrics | E-Gate stage |
|---|---|---|
| grid_coverage_openness | MapElitesGrid.coverage_ratio + FunctionalCoverageContract | pre-register + anti-artifact |
| novelty_stepping_stone | NoveltySteppingStoneArchive.novelty + PreferAFitnessDriver | anti-artifact + explainable |
| species_protect_window | PopulationSpeciesNiche.protect_window + compatibility_distance | explainable |
| criticality_avalanche | AvalancheReport (size hist / branching) ; Lyapunov companion | anti-artifact + explainable |
| exec_vs_learn_scale | EGatePreRegister.learn_scale + exec_scale | pre-register |

## E1 / E2 / E3 word discipline

| Tag | Meaning | This knife |
|---|---|---|
| **E1** | Weak / descriptive | Fixture claim file tagged E1 |
| **E2** | Passed full E-Gate | Unit path can green; no marketed E2 campaign |
| **E3** | Failed stage 2 or 3 | Exercised in tests (anti-artifact fail → E3) |

## Ban: exec scale ≠ learn scale

Pre-register records both. Claiming `exec_scale` as `learn_scale` when they differ fails stage 1. Prefer-A line learns at **82** unless a registered multi-scale run says otherwise. FE `LIFEFORM_SCALES` is viz **NON-BLOCK**.

## CI / local proof

```
ctest -R test_population_phase3_map_elites
```

Expect `[PASS] test_population_phase3_map_elites all assertions passed!`

Also retain Phase 2 population suite green: `ctest -R 'test_population_'`.

## A0–A11

| ID | Verdict |
|---|---|
| A0 | PASS — tip `2817e2f0` + branch `feat/cellular-phase3-map-elites` (no interim `8edb8e3`) |
| A1 | PASS — L1 MAP-Elites grid + insert/replace tests |
| A2 | PASS — novelty stepping-stones; driver modes tested |
| A3 | PASS — protect_window on PopulationSpeciesNiche; name-disambiguated |
| A4 | PASS — Coverage · L1 distance · avalanche · Lyapunov companion + table |
| A5 | PASS — dual · ≥4×3 learning scales · OOD+McNemar |
| A6 | PASS — E-Gate pipeline + `docs/claims/E1-phase3-map-elites-fixture.md` |
| A7 | PASS — E1/E2/E3 in this handoff; no FE claim chips |
| A8 | PASS — exec≠learn documented + E-Gate stage-1 check |
| A9 | PASS — 82-cell / capacity-stable grid / det; no 4th ABI |
| A10 | PASS — no DouZero/GPU/PSRO; win STOP; Phase4 PARK; no ADAS/zoo/mutate |
| A11 | PASS — this handoff |

## NON-BLOCK / PARK / STOP

POET · ASAL · 200-cell probe · L0 KNN remains · LIFEFORM_SCALES FE · optional doctrine land skipped (untracked EMERGENCE/scripts left alone).  
Phase 4 **PARK**. Win knives **STOP**.

## Escalate 手术刀

**No.** Prefer-A grid/protect/avalanche/E-Gate landed without a fourth genome ABI.

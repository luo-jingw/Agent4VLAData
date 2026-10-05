# Plan

Plan Status: completed

# Problem

## Current

Local candidate ranking and continuation diagnostics are complete. They do not measure whether action-effect
structure transfers from other episodes or tasks without visual-neighbor retrieval. The physical pilot pools
tasks, uses privileged state and shares one physical schema across LIBERO-Goal tasks.

## Problem

Separate transferable physical effects from task utility, and distinguish evidence for representation transfer
from evidence for reconstruction, neighborhood retrieval or successful-action preference.

## Goal

Complete four bounded research rounds: primary-source audit of transferable dynamics, supervision audit,
existing-data transfer probe, and evidence synthesis. Record positive and negative observations with their scope.

# Structure

## Modules

Literature evidence; CPU cross-task effect diagnostic; research synthesis and open hypotheses.

## Responsibilities

The literature audit identifies mechanisms and limitations. The diagnostic isolates source-task coverage and
state/action conditioning. The synthesis maps measured facts to the existing r/q framework without replacing it.

## State Ownership

`plan.md` owns execution status. The diagnostic owns its immutable row manifest, grid, predictions and metrics
under `outputs/physics_world/cross_task_probe_v1/`. Verified conclusions belong to the research report;
unvalidated mechanisms belong to `opportunities.md`.

# Interface

## Interfaces

`CrossTaskProbeConfig` defines immutable paths, sample budget, seeds, ridge penalty and model/source grids.
`CrossTaskEffectProbe.run() -> None` reads existing transitions and writes complete measured results.

## Inputs

Primary papers; framework and predictor source; existing physical-world demo/development/fresh transitions.
Inputs to fitted predictors are current physical state and a ten-step candidate action chunk. No task text,
success labels, future state or continuation enters their features.

## Outputs

Source-linked research report, complete probe table, machine-readable provenance/predictions/resource use,
and explicitly unverified next hypotheses. No dataset relabeling or policy update is part of this study.

## State Changes

Read-only source data. New diagnostic code and output directory. No change to deployed policy or encoder weights.

# Flow

## Main Flow

Audit transferable dynamics → audit grounding/identifiability → freeze and run diagnostic grid → synthesize.
Existing local-retrieval and fixed-continuation choices do not constrain representation pretraining.

# Code Mapping

## Modules

- Literature and synthesis: `docs/cross_episode_representation_research.md`.
- Configuration: `ceam/world/cross_task_probe_config.py`.
- Diagnostic: `ceam/world/cross_task_effect_probe.py`.
- Entry point: `scripts/run_cross_task_effect_probe.py`.
- Independent saved-prediction validation and aggregation: `scripts/analyze_cross_task_effect_probe.py`.

## Interfaces

Configuration dataclass and diagnostic class in the files above; existing `WorldTransitionSet` remains unchanged.

## State

`outputs/physics_world/cross_task_probe_v1/{manifest,results,resources}.json`, predictions and results table;
`opportunities.md` owns future representation-learning hypotheses; `issues.md` owns unresolved limitations.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Round 1: verify primary evidence for cross-context effects, object/relational structure and decoder-free prediction.

### Files

`docs/cross_episode_representation_research.md`, `plan.md`.

### Structures

Evidence matrix separating training inputs, objective, transfer test and limitations.

### Affected Modules

Literature evidence only.

### Dependencies

None.

### Observation

Every mechanism claim has a primary source and inspected method section; paper findings and project inference
are separately identified. Record versions and retrieval date.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Round 2: identify what action labels, temporal structure and regularization can and cannot establish.

### Files

`docs/cross_episode_representation_research.md`, `plan.md`.

### Structures

Grounding and identifiability audit; distinguish inverse latent-action labels from forward candidate evaluation.

### Affected Modules

Literature evidence.

### Dependencies

Phase 1 evidence matrix.

### Observation

Include distractor counterevidence, cross-embodiment limitations and the distinction between noncollapse and
preservation of controllable effects. Avoid treating empirical failures as impossibility theorems.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Round 3: measure a bounded cross-task physical-effect transfer proxy using existing data, on CPU only.

### Files

`ceam/world/cross_task_probe_config.py`, `ceam/world/cross_task_effect_probe.py`,
`scripts/run_cross_task_effect_probe.py`, diagnostic output directory, report and `plan.md`.
Reproducible post-fit analysis: `scripts/analyze_cross_task_effect_probe.py`, captured by
`tmp/runlogs/cluster_run/` and fingerprinted in the diagnostic validation output.

### Structures

Grid: 10 target tasks × {own-task, other-nine-tasks, half-own-half-other} TRAIN pools ×
{state-only, action-only, additive state/action, state/action with relative-position/contact interactions} ×
3 sampling seeds = 360 fits. Each fit uses 256 windows, ridge penalty .01, TRAIN-only normalization and the
same target-task held-out candidate banks. No hyperparameter search. Outputs are EEF and four object xyz deltas.
Interactions are 12 object-relative xyz values plus 4 contact indicators, crossed with seven mean action values.
TRAIN normalization floors are .01 for state/relative coordinates and products, .05 for commands; the interaction
factors are standardized before multiplication. Other-task windows are balanced across the nine source tasks,
with uniform sampling without replacement within each task; mixed pools reuse half of each corresponding pool.
Absolute-error summaries weight tasks equally; effect R² pools within-bank SSE/SST and reports EEF/objects separately.
No candidate outcome filtering. Record all source row IDs before fitting. Report within-bank centered effect R²
and Euclidean errors separately for EEF and objects, plus all task cells. State-only and persistence are controls.

### Affected Modules

New standalone diagnostic; existing predictor and data modules are read-only.

### Dependencies

Phases 1–2 determine interpretation and limitations.

### Observation

Check source/evaluation episode disjointness, shapes, finite values, schema identity, bank start-state equality,
sample counts and input checksums. Both evaluation splits are reused research data, not untouched confirmation.
The initial input audit measured a maximum bank start-state difference of 8.6652e-15 and exited before fitting.
Record this floating-point spread directly, retain the failed audit, and quantify its metric contribution with
the state-only control. Bit identity is not a prerequisite for this aggregate physical prediction diagnostic.
Same-schema tasks do not establish visually dissimilar or cross-scene transfer. The privileged-state ridge grid
does not test learned visual representations, general feasibility or policy SR. Cap at one CPU process, two BLAS
threads, zero GPUs, no new simulator collection; expected minutes rather than a training campaign.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Round 4: reconcile literature, code audit and measured transfer; identify the smallest informative next study.

### Files

Report, `docs/framework.md`, `opportunities.md`, `issues.md`, `plan.md`,
`plans/cross_episode_representation_research.md`.

### Structures

Evidence/conclusion/limits table; separate proposed learning objective and falsification protocol.

### Affected Modules

Persistent research state.

### Dependencies

Phase 3 raw results.

### Observation

Completed: nine primary-source audits, 360 CPU ridge fits and ten persistence controls; all rows retained.
Independent pairwise SSE/SST validation agrees within 2.22e-16 / 6.94e-17. Zero GPUs; 5.956 s fit wall time,
10.813 s CPU time, 75.87 MiB peak RSS. Analysis adds .640 s wall / .555 s CPU. No worker remains.
The report links all raw evidence; ISSUE-046 records coverage limits and OPT-014 records future mechanisms.
No new policy SR, neural pretraining, BC/RL update or dataset publication occurred.

# Deferred Work

Completed continuation diagnostic: plans/continuation_sensitivity.md and docs/continuation_sensitivity_results.md.
All 478 branches and 20 independent replay results are complete; no experiment worker remains active.
Continuation-conditioned learning and matched architecture controls remain OPT-013. Restore sensitivity at
episode 292 remains ISSUE-045; compatibility and label ownership limits remain ISSUE-043/044.
No representation training, BC/RL update or action dataset publication is active.

Completed review: plans/long_horizon_matching_review.md and docs/long_horizon_matching_review.md.
Completed preference, retrieval and physical-world pilots: plans/preference_feedback.md,
plans/retrieval_refinement.md and plans/physical_world.md. Predictor improvements remain OPT-012;
precision/confidence and replay limits remain ISSUE-041/042. LIBERO-90 R6 interfaces remain ISSUE-039.

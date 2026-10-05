# Plan

Plan Status: completed

# Problem

## Current

The retrieval pilot supplies successful faster actions but frozen match does not establish a stable quality
shortlist advantage. LIBERO-Goal raw state is available. A CPU schema probe finds identical object/fixture names
and nq=41, nv=37 across ten tasks, permitting a shared task-independent physical predictor.

## Problem

Scene reconstruction alone cannot validate novel actions. Physical schema widths must match recorded arrays;
gripper control DOF differs from the accumulator state width. The recorded arrays have 174 columns while
the initial schema declares 173, requiring metadata correction and checkpoint regeneration before closure. A short-horizon action-conditioned predictor must
distinguish candidate consequences and expose unreliable predictions before it can replace candidate executions.

## Goal

Measure ten-step physical consequence prediction from structured three-dimensional state, comparing demonstration
training with added same-start TRAIN branches. Keep r/q/m unchanged. Predictive confidence and continuation task
success are distinct measurements. This pilot does not train a vision encoder, alter action labels or train RL/BC.

# Structure

## Modules

Physical state reader, transition collection, latent chunk predictor, training/checkpoint owner and comparison report.

## Responsibilities

The world predictor supports task-independent candidate consequence estimation. Task compatibility remains owned
by existing q/m. Physical prediction is an auxiliary experiment supporting r, not a redefinition of match.

## State Ownership

Existing PhysicsTaskEnv exclusively owns simulation state. The reader only observes it. Collection owns physical
transition records; training owns normalization/model weights; reporting owns calibration and held-out comparisons.

# Interface

## Interfaces

- PhysicalStateReader.read() -> ndarray[D] after simulation settling; schema() -> PhysicalSchema. Canonical joint/object/fixture names and
  field groups are explicit. All ten tasks must expose the same schema for a shared predictor.
- WorldTransitionSet: starts/ends [N,D], actions [N,10,7], task/episode/anchor/bank/kind provenance and separate
  continuation outcome availability. No language, task ID, outcome or future state enters predictor inputs.
- PhysicalWorldCollector.run_task(task) -> None: per-role transitions and source/restore observations.
- LatentChunkDynamics.forward(state, action_chunk) -> LatentPrediction: predicted physical delta, reconstructed
  start and predicted/current latent. All future states appear only in training targets.
- PhysicalWorldTraining.run() -> None: fixed normalization, model cells, predictions, checkpoints and resource values.
- PhysicalWorldReport.run() -> None: full grid, task/subset comparisons, calibrated forecast-error coverage and limits.

## Inputs

Current physical state includes generalized positions/velocities, EEF/object xyz and rotations, two-finger grasp
signatures, controller context and fixture poses. Actions use existing dataset conventions. Source data and existing
retrieval artifacts remain read-only; outputs use a new version.

## Outputs

Physical prediction error, candidate-centered effect R², contact prediction error, ensemble disagreement,
confidence/actual-error curves, continuation outcome rates among trusted forecasts, elapsed/RSS/VRAM and provenance.

## State Changes

No change to existing action representations, queries, match, simulators or raw caches. New auxiliary modules live
under ceam/world/. No learned feasibility guarantee or policy SR is claimed.

# Flow

## Main Flow

Replay TRAIN demonstrations → collect ten-step transitions and independent same-start TRAIN alternatives → collect
calibration and HELDOUT physical outcomes → fit fixed shared predictors → calibrate forecast confidence on TRAIN
calibration sources → measure development and fresh HELDOUT results → reconcile evidence and stop this bounded pilot.

# Code Mapping

## Modules

| Responsibility | Files |
|---|---|
| Physical schema/settled-state reader | ceam/world/physical_state.py |
| Transition persistence | ceam/world/world_transition_set.py |
| Collection | ceam/world/physical_world_collector.py, scripts/collect_physical_world.py |
| Typed run configuration | ceam/world/physical_world_config.py, configs/world/physical_pilot.json |
| Predictive interface/model | ceam/world/chunk_dynamics.py, ceam/world/latent_chunk_dynamics.py |
| TRAIN-only normalization | ceam/world/physical_normalization.py |
| Fit/predict/checkpoint owner | ceam/world/physical_world_training.py, scripts/train_physical_world.py |
| Complete report/calibration | ceam/world/physical_world_report.py, scripts/report_physical_world.py |
| Package exports | ceam/world/__init__.py |
| Runtime/resources | tmp/physics_world/, tmp/runlogs/physics_world/ |
| Outputs | outputs/physics_world/pilot_v1/ |
| Verified results | docs/physical_world_results.md |

## Interfaces

All boundaries map to the files above; the latent predictor defines a Protocol and typed prediction result.

## State

TRAIN/calibration/HELDOUT provenance lives with transition sets. Normalization uses only demonstration TRAIN rows,
shared across data ablations. Completed model cells and checkpoints are explicit files; no global hidden state.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Implement source/schema/collection and predictive interfaces, and observe one-task collection/training smoke
before collecting the complete grid.

### Files

Physical reader, config, transition set, collector, latent predictor, training/report interfaces, entry points
and runtime wrappers. Model interfaces are observed with the smoke data; the complete fitting grid is Phase 3.

### Structures

Typed physical schema and five source roles: demo TRAIN, branch TRAIN, calibration, development and fresh HELDOUT.

### Affected Modules

New ceam/world only, reusing exact source matching and read-only donor indexes.

### Dependencies

Existing retrieval and physics environment interfaces.

### Observation

Shapes/dtypes, exact source alignment, schema equality, split separation, original endpoint replay discrepancy,
raw restore errors, branch coverage, source success and runtime/RSS. Keep every failure.

Smoke observations: one task, four disjoint-role source replays; D=174 float32, actions [N,10,7]. Twenty-four
demonstration windows and twelve candidates each for branch/calibration/development/fresh. Original endpoint
max discrepancy 7.69e-15, restored-state discrepancy 6.64e-15. Collection took 48.08 s. All six fit cells,
three-member predictions and fourteen report cells completed. GPU smoke fitting/prediction took 5.72 s;
only node-63 GPU 0 was used. Evidence: outputs/physics_world/smoke/ and tmp/runlogs/physics_world/.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Collect all ten tasks and the fixed source/action grid.

### Files

Collection modules and outputs; no model fitting.

### Structures

First 24 TRAIN episodes/task provide demonstration windows at stride five. First 12 are donor sources;
TRAIN positions 12–15 provide four branch-query sources/task. TRAIN positions 24–26 provide three calibration
sources/task. First three HELDOUT sources/task form development; all remaining HELDOUT sources form confirmation.
Fresh confirmation has 13 episodes across nine tasks; task 1 has no fresh source and is reported explicitly.

For each branch source use three anchors at 1/4, 1/2, 3/4 first-success length, rounded as in retrieval. Physical
retrieval supplies up to eight distinct donors; add original and three arm-only alternatives: half-speed commands,
Gaussian perturbations sigma .1 and .3 clipped to [-1,1]. Preserve source gripper sequence. Execute every complete
ten-step chunk for a consistent physical endpoint. Calibration/evaluation continuation uses the original suffix
and records whether success occurs during the chunk or suffix. TRAIN branch outcomes do not supervise models.

### Affected Modules

New physical collection and existing read-only source/index interfaces.

### Dependencies

Phase 1.

### Observation

Complete manifests, coverage and replay metrics. Development donor outcomes may reuse existing verified branch
labels; added perturbations and fresh/calibration branches are executed. Source-overlap checks are explicit.

Complete collection: 5,135 demo windows from 240 episodes; 1,439 TRAIN branches from 40 episodes;
1,078 calibration and 1,078 development candidates from 30 episodes each; 467 fresh candidates from 13 episodes.
All ten task manifests and schemas are present. An independently repeated task-1 endpoint differs by 0.07304
in one generalized-velocity component, with worst xyz distance 4.56e-7 m. Numerical replay variation is recorded
in ISSUE-041; three-dimensional pose validation remains usable at this pilot's millimetre thresholds.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Define accumulator width by gripper actuator count; initialization has one control accumulator element and
the first step expands it to two actuator elements. Observe settled reader/schema equality on all tasks.
Correct schema metadata from observed state widths, check unchanged transition-array checksums, then fit
every shared model cell on one free GPU and save raw predictions with consistent checkpoint dimensions.

### Files

Latent model, training owner/config/entry point and checkpoint/prediction outputs.

### Structures

Grid: demonstration only / demonstration plus TRAIN branches × latent width 32 / 128, three ensemble members
seeds 0/1/2, plus ridge predictors for both datasets and persistence control. Model inputs contain no task ID or
instruction. Ridge penalty 1e-2. Latent hidden width 128, batch 256, AdamW lr 3e-4, weight decay 1e-4, 2,000 updates
per member. Mixed batches sample half demonstration and half branch rows; member resampling is by source episode.
Loss is standardized physical-delta MSE plus .1 start reconstruction MSE and .1 latent future consistency MSE.
TRAIN-only state std floor .01 and delta std floor .001; normalization is fixed from demo rows for all cells.
No calibration/test selection of checkpoints. Persist every member and all validation/evaluation predictions.

### Affected Modules

New world-model training. Existing r/q/m remains frozen and is not loaded by training.

### Dependencies

Phase 2.

### Observation

Loss components, per-member timing, parameter count, peak allocation/device memory, data checksum and finite outputs.
Estimated GPU budget below .5 GPU-hour, one GPU at a time; no node exceeds two project GPUs. CPU collection is
estimated at 1–3 core-hours on node-61/63 with five workers/node and thread limits two.

All ten full-task schemas and the one-task smoke schema match observed settled state [174].
Control DOF is one; initialization accumulator width is one and post-step actuator accumulator width is two.
Schema correction preserved every transition NPZ byte. All six fit cells completed with dimension-consistent
checkpoints: twelve latent members at 2,000 updates each plus two ridge predictors. Predictions are saved
for calibration/development/fresh. Only node-63 GPU 0 was used, with complete resource logs.


## Phase 4

Phase Status: completed

Round: 4

### Goal

Report every cell and reconcile the bounded experiment.

### Files

Report module/entry point, docs/physical_world_results.md, issues.md, opportunities.md and completed plan archive.

### Structures

Pooled and per-task development/fresh tables, original/donor/perturbation strata. Report EEF and maximum object
translation error, rotation matrix error, contact accuracy and within-bank centered xyz effect R². Calibrate
ensemble error/disagreement ratio using calibration-episode maxima pooled across tasks, 95th percentile; this is
an empirical diagnostic, not a distribution-free safety certificate. Report forecast trust thresholds 5/10/20 mm,
coverage, actual error violations and task failures among trusted predictions separately. Bootstrap source episodes
for descriptive uncertainty. Latency includes normalization and inference; compare with measured CPU physics.

### Affected Modules

New physical-world report and persistent research state.

### Dependencies

Phase 3.

### Observation

All model/data cells and missing confirmation task are visible. Stop after this pilot: vision reconstruction,
model-based action optimization, RL and BC need a subsequent evidence-based plan. Weak prediction or false
confidence is recorded as unresolved evidence, without automatically expanding the experiment.

All fourteen pooled cells, task/stratum rows, six paired comparisons and 42 calibrated-bound entries are
reported. Fresh mixed latent-128 effect R² is .658, mean worst xyz error 22.26 mm, p95 44.51 mm, grasp-change
recall 6.85%. All predefined confidence bounds retain zero candidates. ISSUE-042 records the unresolved
physical precision/contact and confidence-utility limits. OPT-012 scopes a subsequent predictor experiment;
no vision, label optimization or policy training was started. Fresh corrected-reader collection and strict
checkpoint loading complete. Full results and timed resource measurements are in docs/physical_world_results.md.

# Deferred Work

Latest completed retrieval experiment: `plans/retrieval_refinement.md`, with results in
`docs/retrieval_refinement_results.md`. Semantic preference learning remains under OPT-011. This physical predictor
pilot promotes only the auxiliary action-consequence validation mechanism; no policy training is active.

LIBERO-90 R6 remains proposed under `plans/libero90_pending_work.md` and `plans/libero90_r6_evaluation.md`;
ISSUE-039 is unresolved. Next-generation representation training remains under OPT-009.

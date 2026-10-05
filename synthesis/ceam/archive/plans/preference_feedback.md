# Plan

Plan Status: completed

# Problem

## Current

Frozen R6 ranking loses successful retrieval branches and has no established shortlist speed advantage over random.
Successful faster candidates exist, but speed-only oracle raises jerk. The physical-world pilot provides exact
same-start candidate actions and role/source provenance; its physical forecasts are insufficient to skip execution.

## Problem

Demonstration reconstruction supplies little explicit evidence about preferences between successful actions.
It is unknown whether fixed r can support faster/shorter preferences after feedback, or whether r needs adaptation.

## Goal

Collect actually executed local alternatives and learn compatibility and successful-action preference using shared
same-start banks. Compare fixed R6 with a task-independent residual r under identical feedback. Evaluate direct
selection and equal-budget physics-verified shortlists on development and remaining benchmark HELDOUT sources.
No original dataset edits, policy training, full-route optimization, visual encoder tuning or RL are in this pilot.

# Structure

## Modules

Typed preference banks/config; read-only feasibility observer; physics collection; rendered feature encoding;
training data owner; representation/query model and interface; preference objective; fitting; complete report.

## Responsibilities

r remains task-independent. A compatibility query owns feasible task-completion evidence; a separate preference
query owns ordering between feasible successful branches. Both use cosine with the same r. Preference supervision
extends successful-action comparison without changing the existing frozen match or replacing it with a speed scalar.

## State Ownership

PhysicsTaskEnv/LiberoTaskEnv exclusively own simulator state. The validity observer only reads state/contact data.
Collection owns actual trajectory metrics and banks. Encoding owns outcome-blind visual/text/R6 features.
Training owns new residual/query weights and TRAIN normalization. Reporting owns measured comparisons.

# Interface

## Interfaces

- PreferenceConfig.read(path) -> PreferenceConfig.
- PreferenceBank.save/load(path): same-start snapshot, [K,10,7] actions, role/source IDs, measured validity/success,
  first-success duration, EEF translation/rotation paths and jerk, physics-replay checks and raw measurements.
- BranchValidity.observe(action) -> None; result() -> ValidityResult: finite state, bounded commands, arm joint
  range violation and arm/environment contact at control-step endpoints. This is a declared observed-feasibility
  screen, not continuous-time collision certification. Intended gripper/object contact is allowed.
- PreferenceCollection.run_task(task, smoke) -> None: exact actions from physical-world role banks, actual suffixes.
- PreferenceEncoding.run(tasks) -> None: restore/query-image render, frozen DINO/PCA, frozen R6 action effects,
  frozen text/task and original query features. No outcome enters encoded inputs.
- PreferenceData.load(config, roles) -> PreferenceData: padded masked banks, episodes, input tensors and targets.
- PreferenceMatcher.forward(context, actions, base_r, base_q, task) -> PreferencePrediction.
- PreferenceObjective.forward(prediction, targets, kind) -> PreferenceLoss.
- PreferenceTraining.run() -> None; PreferenceReport.run() -> None.

## Inputs

Use physical-world training/calibration/development/fresh candidate lists verbatim: 120/90/90/39 banks,
40/30/30/13 source episodes, ten tasks except nine-task fresh. Donors and TRAIN branch queries are disjoint.
Development was used in retrieval research; remaining HELDOUT physical outcomes were reported by the world pilot.
These sources remain disjoint from preference training, but are not claimed to be untouched research holdouts.
Original instruction/observations/proprio and candidate actions are model inputs; privileged state is only for
collection/feasibility metrics, never matcher inputs. Demonstration caches and prior artifacts remain read-only.

## Outputs

Actual branch compatibility, successful Pareto pair counts, direct branch selection, source/task-stratified
completion time/path/jerk, preference accuracy, representation spread/effective rank and raw predictions.
Equal four-alternative shortlists with measured feasibility/success/Pareto acceptance isolate execution-budget utility.

## State Changes

New auxiliary preference modules and outputs only. R6 weights, original queries, world-model artifacts, raw data
and action labels are unchanged. No candidate is published as a refined demonstration by this experiment.

# Flow

## Main Flow

Read fixed candidate banks → replay independent sources → execute candidates plus source suffix → measure
feasibility/success/time/path → render exact common starts → encode outcome-blind features → train fixed grid →
measure all selectors on source-disjoint roles → reconcile evidence and stop this bounded pilot.

# Code Mapping

## Modules

| Responsibility | Files |
|---|---|
| Config / bank persistence | ceam/preference/preference_config.py, ceam/preference/preference_bank.py |
| Observed feasibility | ceam/preference/branch_validity.py |
| Physics collection | ceam/preference/preference_collection.py, scripts/collect_preference.py |
| Rendered input encoding | ceam/preference/preference_encoding.py, scripts/encode_preference.py |
| Dataset / targets | ceam/preference/preference_data.py |
| Model interface / implementation | ceam/preference/preference_matcher.py, ceam/preference/residual_preference_matcher.py |
| Objective | ceam/preference/preference_objective.py |
| Fit / report | ceam/preference/preference_training.py, ceam/preference/preference_report.py, scripts/train_preference.py, scripts/report_preference.py |
| Package / config | ceam/preference/__init__.py, configs/preference/feedback_pilot.json |
| Runtime / outputs | tmp/preference/, tmp/runlogs/preference/, outputs/preference/pilot_v1/ |
| Verified results | docs/preference_feedback_results.md |

## Interfaces

Typed dataclasses and Protocol; simulator state modifications use existing env interfaces only.

## State

Bank manifests and checksums preserve episode/task/role alignment. Padded candidates carry explicit masks.
No future metrics or privileged state enter input representations. Stored models include all residual weights.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Implement typed banks, observed-validity screen and collector; run a one-task one-anchor smoke.

### Files

Config, bank, validity, collection, collection entry point and runtime wrappers.

### Structures

All physical-world TRAIN branch, calibration, development and fresh candidates are retained, including failures.
Restore snapshot before anchor-1 plus producing action. Execute replacement plus original suffix through first
success or source horizon. Record complete prefix+continuation metrics at 20 Hz; prefix is common within bank.
Observed feasibility requires finite state and valid commands, arm joint range excess ≤1e-3 rad and no arm-link
contact with non-robot geometry at observed control-step endpoints. Finger/palm task contact is not forbidden.
Only this bounded screen is reported; actuator-limit/continuous-collision certification is outside its evidence.

### Affected Modules

New preference collection, reading existing world/retrieval sources and simulator interfaces.

### Dependencies

Existing physical-world/retrieval data and exact raw source matching.

### Observation

Counts/source IDs, shapes/dtypes, action equality, baseline success/steps, restore error, joint/contact measurements,
original validity and per-worker time/RSS. Do not discard invalid source/candidate cells.

One-task smoke collected four banks of twelve candidates. All originals are successful and feasible; candidate
feasible counts are 12/9/12/12, demonstrating the screen rejects some candidates rather than assuming legality.
World-start discrepancy is zero; runtime 76.87 s. Full collection additionally carries the common-prefix
validity observations into each branch, through an explicit typed initial ValidityResult.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Collect all tasks, encode actual restored 512² dual-camera inputs and validate data separation.

### Files

Collection outputs, encoding/data modules and entry points.

### Structures

One feature file/bank: base R6 vectors, PCA state, proprio, original query and task embedding, standardized actions.
Frozen DINO, PCA and task/original query weights. At most five CPU workers/node, threads two, nodes node-61/63;
one GPU for encoding. Observed scene/proprio restoration is reported rather than assumed from cached frames.

### Affected Modules

New collection/encoding/data only.

### Dependencies

Phase 1.

### Observation

Every planned bank; same source/candidates as world pilot; no source overlap across feedback roles; no input
future/outcome fields; finite encoded tensors and measured pair coverage before fitting.

All 339 planned banks completed: 120 TRAIN, 90 calibration, 90 development, 39 remaining HELDOUT.
TRAIN has 3,334 Pareto pairs; calibration/development/fresh have 2,810/2,844/1,307. Original observed-feasibility
rejections are 21/9/6/3 banks respectively; all were retained. Encoding rendered every common start, with
339 feature records, in 40.79 s on node-63 GPU 0. Model/objective forward/backward observations preserve
initial cosine ordering and expose action-residual gradients only when r adaptation is enabled.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Fit the complete controlled representation/supervision grid and preserve every prediction.

### Files

Matcher interface/model, objective, training owner/config/entry point, checkpoints/raw predictions.

### Structures

Grid: r fixed R6 / R6 plus learned residual × compatibility-only / compatibility plus Pareto preference × seeds
0/1/2: twelve runs. Frozen visual/PCA/R6 backbones and original query/task encoders; new small residual MLPs only.
r input: base r, PCA state, proprio and actions, no instruction. Two queries use current context, original query
and frozen task embedding. Hidden width 256, output 256, zero-initialized residual outputs preserve initial scores.
Train bank bootstrap by source episode, batch 16 banks, 2,000 updates, AdamW lr 3e-4, weight decay 1e-4.
Compatibility loss is BCE on observed-feasible successful branches. Preference loss is successful-feasible
within-bank Pareto pairs, no label for incomparable pairs; residual anchor penalty .01. Time uses first-success
steps; translation tolerance 1 mm. Better pairs have no worse time and translation, with strictly less time or
translation improvement >1 mm. Rotation and jerk remain separate measured tradeoffs, not silently encoded as
translation or command variation. No checkpoint selection using development/fresh outcomes.

### Affected Modules

New r/query residual models; existing r/q/m untouched.

### Dependencies

Phase 2.

### Observation

All twelve runs; objective components, pair availability, frozen-weight differences, representation norms/spread,
parameter counts, finite predictions and GPU time/memory. Calibration diagnostics are separate from task success.
Estimated budget ≤1 GPU-hour and ≤3 CPU core-hours for this pilot; at most one GPU/node is planned.

All twelve model cells completed at 2,000 updates with saved checkpoints and predictions for every role.
Frozen R6/query/encoder files remain read-only; learned parameters are only the declared residual modules.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Report every learned/control selector, reconcile hypotheses and close this bounded experiment.

### Files

Report module/entry point, results doc, issues/opportunities, completed plan archive.

### Structures

Controls: original, exact uniform candidates, frozen original match, successful-feasible Pareto oracle, random
four-alternative shortlist with actual acceptance. Learned selectors: compatibility argmax, preference argmax gated by
predicted compatibility ≥ original score, true-compatible preference ranking diagnostic and guided four-alternative
shortlist plus actual acceptance. Original stays available. Shortlist acceptance uses actual successful-feasible
candidates no worse than original time/path, then minimizes steps/path. This oracle acceptance is not learned
safety or policy SR. Random shortlist expectations enumerate subsets exactly when feasible.
Report development/fresh separately, all seeds, task/pair metrics, source-episode bootstrap intervals, failures,
all-branch duration and successful time/path/rotation/jerk, equal-budget gains and representation diagnostics.
No action labels or policies are trained after this report without a subsequent evidence-based plan.

### Affected Modules

Preference reporting and persistent project state.

### Dependencies

Phase 3.

### Observation

All 130 selector cells, 28 evaluation ranking rows, source/task/seed breakdowns and actual verification controls
are reported. Remaining-HELDOUT preference accuracy is 52.79% frozen, 61.49% fixed-r preference and 69.98%
adapted-r preference. Adapted-preference shortlists improve versus random by 0.0375 s and 10.83 mm of full path,
with episode-bootstrap intervals excluding zero. Direct compatibility-gated selection still loses successes;
compatibility calibration is weak, and faster/shorter supervision does not establish less jerk. No labels or
policies were created. All twelve checkpoints reload strictly and reproduce GPU predictions within 2.86e-6;
source actions and original checkpoint hashes match. Timed use is 1.237 CPU core-hours and .0439 GPU worker
wall-hours; only node-63 GPU 0 was used. Both nodes have no remaining experiment workers at 15:59:30 UTC.
Verified findings are in docs/preference_feedback_results.md; unresolved boundaries are ISSUE-043/044.

# Deferred Work

Completed retrieval and physical-world pilots: plans/retrieval_refinement.md and plans/physical_world.md.
Physical predictor changes remain under OPT-012; this pilot promotes only compatibility/preference feedback from
OPT-011. Remaining feedback/strategy work is OPT-013; policy training and vision reconstruction remain deferred.
LIBERO-90 R6 interfaces remain blocked by ISSUE-039 and are not part of this experiment.

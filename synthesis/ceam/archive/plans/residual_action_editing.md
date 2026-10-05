# Plan

Plan Status: completed

# Problem

## Current

Executed retrieval and preference studies show available local improvements and useful learned preferences,
but no robust label-utility gain over original demonstrations. Continuation changes candidate rankings.
OPT-015 defines fragment adaptation with a low-rank residual. Existing continuation starts retain twenty
source-disjoint-from-preference-training episodes across ten Goal tasks, with exact source actions and traces.

## Problem

It is unknown whether a bounded temporal residual adds useful candidates beyond direct donor transfer or
unstructured noise, whether preference models transfer to these candidates, and whether gains survive a
different continuation. Fixed ten-step edits isolate residual parameterization; they do not test phase removal,
variable duration, multiple-edit composition, or the full offline-editor hypothesis.

After three rounds, selection remains weak and source gains often fail to transfer. The first ranking grid
trains time/path preference while acceptance also limits jitter; the contribution of this target mismatch is
not isolated. A fourth controlled round changes only the training preference-pair jitter condition.

## Goal

Run three sequential experimental rounds: candidate availability, learned ranking, and continuation transfer.
Keep original candidates and all failures. Produce complete tables, code/config provenance and resource records.
Assess whether the evidence warrants a subsequent edited-dataset/student comparison.
Complete the target-alignment follow-up before closing that assessment.

# Structure

## Modules

- `ceam/editing/`: bounded proposal generation, source execution, reporting and learned selection diagnostics.
- Existing `ceam/preference/`: immutable starts/traces, validity measurements, encoders, preference targets,
  adapted matcher and frozen-policy continuation.

## Responsibilities

Proposal generation owns candidate actions/metadata. Collection owns executed outcomes. Reporting owns
acceptance and selection rules. Training owns fitted parameters and TRAIN-only normalization. Existing
environment classes remain the only simulator writers; no simulator/controller implementation changes.

## State Ownership

Inputs under outputs/preference/{continuation_v1,pilot_v1} are read-only. New files live under
outputs/residual_editing/pilot_v1. Each task worker owns its task's banks/branches. A single report process owns
aggregate tables. GPU processes and their lifecycle are recorded under tmp/runlogs/residual_editing.

# Interface

## Interfaces

- `ResidualProposalBank`: actions [K,10,7], family/reference ids, residual RMS/rank and seed provenance.
- `ResidualProposalGenerator.generate(start)`: finite controller-bounded proposals, original first.
- `ResidualCollection.run_task(task)`: source-suffix execution through existing ContinuationExecutor.
- `ResidualReport.run()`: exhaustive groups and matched-budget acceptance tables.
- `DirectPreferenceMatcher.forward(...)`: same inputs/output contract as ResidualPreferenceMatcher.
- `ResidualRanking.run(device)`: source-disjoint learning and new-bank ranking.
- `ResidualRanking.run(with_jitter=True)`: identical fitting with the additional within-pair jerk limit;
  owns a separate models/jitter directory and distinct prediction names.
- `ResidualContinuation.run_task(task, device)`: declared source-selected candidates under frozen policy.

## Inputs

Twenty previously inspected continuation starts from configs/preference/continuation_pilot.json, all ten tasks.
Existing task-local donor chunks and original ten-step gripper schedules. Source and frozen no-psi policy
continuations; original source horizon, 20 Hz physical control. Old TRAIN preference outcomes/features provide
ranking supervision; none of the twenty new outcome banks enter fitting or checkpoint selection.

## Outputs

Immutable NPZ proposal banks; per-branch JSON and full traces; preference-compatible banks for encoding;
model checkpoints/predictions; JSON/CSV/Markdown reports; measured resource ledger; verified results in
docs/residual_editing_results.md. No replacement labels published during these diagnostic rounds.

## State Changes

No raw data, old artifacts, environment, encoder weights or existing model checkpoints are modified.
Candidate start/metadata digests and code/config versions accompany new results.

# Flow

## Main Flow

1. Generate all cells before collection. For each of up to eight donors: direct transfer; rank-6 constant temporal
   residual; rank-12 two-mode DCT residual; full-rank IID residual. Each residual has one antithetic pair per donor,
   target arm RMS .05 and a shared maximal scale that keeps every command in [-1,1], without clipping directions.
   Add sixteen rank-12 edits around the original and one geometric command comparator that replaces arm xyz
   increments by their mean while retaining rotation/gripper. This preserves summed commanded displacement,
   not necessarily actual EEF displacement. Original plus at most 73 alternatives per start, <=1480 branches.
2. Execute all source-suffix branches. Repeat original at the end of each bank to observe restoration/order drift.
   Record whole/prefix/candidate/suffix metrics and full state traces. Geometric and residual mechanisms remain
   separate; zero residuals due to command-bound saturation are reported, not discarded.
3. Report success and observed validity separately. Primary acceptance requires successful-feasible source and
   candidate, steps <= source, translation <= source +1 mm and jerk <= source*1.05 +1e-6, with steps strictly
   shorter or translation at least 1 mm shorter. Five percent jerk tolerance is an engineering comparison bound,
   not a physical guarantee. Also report zero-tolerance jerk and time/path-only sensitivity. No worse-original
   claim for starts whose original fails the physical screen. Compare families at eight-candidate budgets using
   256 deterministic random subsets, plus full-bank availability. Resampling intervals cluster by source episode.
4. Encode all new banks with existing frozen encoders. Train direct MLP and adapted dual matcher on identical old
   TRAIN banks, original faster/shorter preference targets, 2000 updates, batch16, three seeds. New held-out
   reports include pair ranking, compatibility, original+four-candidate shortlists, exact-budget random controls
   and full-bank oracle. Jitter was not in training targets; strict acceptance measures whether it transfers.
5. For every start, evaluate original and source-verified winner for direct, rank6, rank12, IID, original-residual
   and geometric families under the common frozen policy. Deduplicate identical selected indices while retaining
   family attribution. Re-execute selected source controls in the rendered environment to diagnose backend or
   restoration differences. Maximum 140 distinct selections x2 continuations; one policy seed, explicitly scoped.
6. Assess the downstream gate: useful gains beyond equal-budget IID, usable learned selection, and consistency
   under continuation. If these are not supported, close with the located bottleneck. If all are supported,
   return to plan with concrete dataset/student interfaces before executing a matched three-seed policy grid.

Resource budget: <=8 aggregate CPU process-hours and <=4 GPU-hours for these rounds; <=4 CPU workers,
<=2 GPUs total and <=2 per node. Prefer CPU for physics and one GPU for encoding/training/continuation.
No node-70/71 GPUs or node-64 GPU3. All remote commands use scripts/cluster_run.sh. Record actual rather than
estimated GPU-hours. Budget exhaustion triggers a recorded partial result and revised plan, not extra cells.

# Code Mapping

## Modules

Proposal: ceam/editing/residual_proposals.py. CPU adapter: ceam/editing/physics_environment.py.
Collection: ceam/editing/residual_collection.py. Reporting: ceam/editing/residual_report.py.
Direct architecture: ceam/editing/direct_preference_matcher.py. Learning: ceam/editing/residual_ranking.py.
Ranking data: ceam/editing/residual_ranking_data.py. Selection measurements: ceam/editing/residual_selection_report.py.
Checkpoint/reuse audit: ceam/editing/residual_ranking_audit.py and scripts/audit_residual_editing.py.
Continuation: ceam/editing/residual_continuation.py. Configuration: ceam/editing/residual_config.py and
configs/editing/residual_pilot.json.

## Interfaces

Typed dataclasses in residual_proposals.py; existing ContinuationStart/Trace/Outcome and PreferenceBank formats.
Entry points scripts/{collect,report,rank,continue}_residual_editing.py; encode via scripts/encode_preference.py
with a separate preference-encoding configuration.

## State

outputs/residual_editing/pilot_v1/{proposals,branches,banks,features,models,predictions,reports,continuation}.
tmp/runlogs/residual_editing contains command, timestamps, process exit, CPU RSS/time and GPU samples.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Round 1: bounded candidate generation, controlled CPU execution and complete availability report.

### Files

ceam/editing/{__init__,residual_config,residual_proposals,physics_environment,residual_collection,residual_report}.py;
configs/editing/residual_pilot.json; scripts/{collect,report}_residual_editing.py.

### Structures

ResidualProposalBank and generator; CPU environment adapter; per-task collector and report.

### Affected Modules

New editing package; existing preference/refinement/simulation interfaces consumed read-only.

### Dependencies

Existing verified starts, traces, source actions and episode-disjoint donor library.

### Observation

Proposal bounds, RMS, basis rank and unchanged gripper; original replay and restoration error;
all-family execution outcomes, paired improvement and matched-budget tables; measured CPU time/RSS.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Round 2: determine whether preference selection transfers to structured edits and compare matched information.

### Files

ceam/editing/{direct_preference_matcher,residual_ranking,residual_ranking_data,residual_selection_report,residual_ranking_audit}.py;
scripts/{rank,audit}_residual_editing.py;
configs/editing/residual_encoding.json.

### Structures

Direct preference matcher and ranking runner consuming existing PreferencePrediction/Targets interfaces.

### Affected Modules

Editing training; existing frozen feature encoding and preference objective.

### Dependencies

Phase 1 banks and complete outcome report.

### Observation

Six model cells, TRAIN versus new-source pair accuracy, calibrated compatibility diagnostics,
equal-budget actual-verified shortlist utility, parameters, timings and GPU memory.

Both new architectures use BCE compatibility plus logistic faster/shorter pair loss, AdamW learning rate 3e-4,
weight decay 1e-4 and the same episode-bootstrap samples per seed. No representation-anchor penalty is applied
to either architecture. The direct MLP has two width-474 hidden layers and residual logits initialized at zero;
both architectures start at the original 10*cos match. This is a new matched-objective comparison, not an exact
reproduction of the older dual model's anchored training. Jerk enters acceptance, not these training targets.

After the matched grid, a fixed diagnostic evaluates all six historical fixed/adapted preference checkpoints
with their own saved normalization and no fitting. This isolates whether the observed new-grid dual weakness
also appears with the old anchored models. Reload all six new checkpoints and measure saved-prediction error.
This supplemental grid is exploratory; it does not change any new training cell or its recorded results.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Round 3: test source-selected edits under common-policy continuation, assess downstream gate and close.

### Files

ceam/editing/{residual_continuation,residual_continuation_report}.py; scripts/continue_residual_editing.py;
docs/residual_editing_results.md; issues.md; opportunities.md; plan.md.

### Structures

Selection manifest and deduplicated per-task continuation executor/report.

### Affected Modules

Editing continuation; existing rendered environment and frozen continuation policy.

### Dependencies

Phase 1 selection manifest and Phase 2 results.

### Observation

Paired source/rendered-source/policy results, repeated-original state/control consistency, success and
quality transfer, complete resource ledger and explicit evidence-based student-training decision.

There are 71 distinct source-selected candidates including originals, each run under rendered source and policy
continuation. Repeat original under both modes at each start adds forty controls, for 182 branches total.
The selection manifest is frozen before any policy-continuation result. GPU workers own disjoint task sets.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Isolate training-target alignment with the joint time/path/jitter acceptance condition.

### Files

ceam/editing/{residual_ranking,residual_selection_report,residual_ranking_audit}.py;
scripts/rank_residual_editing.py; docs/residual_editing_results.md; issues.md; opportunities.md; plan.md.

### Structures

An explicit with_jitter option intersects the existing TRAIN better-pair mask with
jerk_preferred <= jerk_other*1.05 +1e-6. Successful-feasible/time/path conditions remain unchanged.
Models, normalization and provenance use models/jitter; new prediction names have a _jerk suffix before seed.
All models are scored on both common time/path pairs and common joint-quality pairs; primary shortlist
acceptance is unchanged. No new query outcomes, candidate actions or continuation selections are collected.

### Affected Modules

Preference fitting, selection report and checkpoint audit only.

### Dependencies

All three prior rounds and their fixed raw data/results. Existing twenty query sources are already inspected,
so this is an exploratory mechanism check, not fresh confirmation.

### Observation

Six cells: direct/dual x seeds0/1/2, same 2000 updates, batch16, normalization, initialization and bootstrap
samples as Phase2. Record retained TRAIN preference pairs, both query pair metrics, verified shortlist
quality and paired differences versus each architecture's time/path-only counterpart. Reload checkpoints.
Training jerk comes from the older observable-pose measurement and query jerk from site boundaries;
relative within-bank supervision is evaluated, with no claim of calibrated absolute jerk prediction.
Additional GPU budget <=.1 GPU-hours within the original four-hour cap. Complete final closure after this grid.

All twelve new model checkpoints reproduce saved predictions after reloading. The target change does not
establish shortlist improvement. All four rounds are complete; downstream gate remains unmet, so no new
dataset or student-policy training is promoted. Results: docs/residual_editing_results.md. Twelve managed
jobs exited zero, .12833 GPU-hours, .37583 CPU collection process-hours; all experiment GPU allocations released.

# Deferred Work

Completed action-label literature review: docs/action_label_optimization_review.md. Eight primary sources,
controller and supervision-data boundaries are recorded. Phase/context/quality-conditioned editing remains
OPT-015; no experiment was promoted, no cluster job started, and additional GPU use was zero.

Completed frozen-teacher editing pilot: plans/frozen_teacher_editing_pilot.md and docs/teacher_editing_results.md.
All 59 numerical cells, 600 source-suffix branches and 36 common-teacher continuation branches are retained.
Teacher noise, TRAIN-calibrated noise and direct edits yield 24, 19 and 16 source-suffix Pareto improvements.
Calibration does not show stable added quality; continuation changes the success/efficiency trade-off.
ISSUE-047 records remaining ranking and continuation questions; OPT-015 retains richer dataset summaries,
compression and direct adaptation. No policy training or replacement dataset was produced. All pilot jobs exited;
approximately 0.442 GPU-hours, maximum one GPU on any node, final GPU allocation zero. Resource ledger:
outputs/teacher_editing/pilot_v1/resources.json. No execution phase is active.

Completed teacher/dataset calibration review: plans/teacher_dataset_calibration_review.md and
docs/teacher_dataset_calibration_review.md. Five primary papers and public interfaces underpin the pilot;
teacher codebooks, lossy compression and model adaptation remain future mechanisms.

Completed cross-episode representation research: plans/cross_episode_representation_research.md and
docs/cross_episode_representation_research.md. Nine primary sources, 360 CPU ridge fits and ten controls are
recorded; all workers exited and no GPU was used. Transferable interaction learning remains OPT-014;
source coverage and metric limitations are ISSUE-046. No neural pretraining, BC/RL update or label publication is active.

Completed continuation, preference, physical-world and retrieval studies are archived under plans/ with measured
results in docs/. Continuation-aware utility learning remains OPT-013; predictor improvements remain OPT-012.
Restore sensitivity is ISSUE-045, compatibility limits ISSUE-043/044, physical precision ISSUE-041/042 and
LIBERO-90 interface follow-up ISSUE-039. The prior literature audit is docs/long_horizon_matching_review.md.

# Plan

Plan Status: completed

# Problem

## Current

Preference labels use the original demonstration suffix and inherit immutable prefix validity. Their sensitivity
to a closed-loop continuation is unmeasured. Source IDs and the frozen no-psi seed-0 checkpoint are available.

## Problem

Candidate quality and compatibility with a particular suffix cannot be separated using existing aggregate banks.
The paired audit finds candidate-stage state sensitivity at episode 292; bit-identical-start sensitivity analysis
is required for attribution. Task 8 raw replay differs only in floating joint-excess values, with unchanged
discrete feasibility and success.

## Goal

Measure paired continuation outcomes at twenty prespecified TRAIN starts, at most 480 unique branches. Separate
prefix/candidate/continuation validity and quantify preference reversals. Stop after reporting; no model training,
action dataset publication or framework-definition change. This supplies evidence for the match interface.

# Structure

## Modules

Continuation configuration/source manifest; immutable candidate preparation; phase-local observation and records;
closed-loop continuation execution; paired analysis and replay verification.

## Responsibilities

Preparation owns original raw matching and fixed candidates. LiberoTaskEnv/PhysicsTaskEnv exclusively own simulator
writes. Read-only observers own measured traces; the executor owns branch phase and first-success stopping.
The report owns comparisons, without selecting a continuation or altering candidate labels after observing results.

## State Ownership

New artifacts live under outputs/preference/continuation_v1. Existing sources, checkpoints and experimental banks
are read-only. New continuation modules own their records; existing PreferenceBank and F1 executors are unchanged.

# Interface

## Interfaces

ContinuationConfig.read(path) returns explicit source IDs, paths and seed. ContinuationPreparation.run_task(task)
writes immutable ContinuationStart records. A start includes full source actions, snapshot-before-producing-action,
expected start state, prefix trace, candidate chunks/kinds/donor identities and metadata. ContinuationTrace holds
per-control-step commands, simulator states, controller/gripper state, EEF poses, observed proprio, success and
phase; phase outcomes hold ValidityResult and path/time/jerk. ContinuationExecutor.run(start, candidate, mode)
returns a typed branch record; mode is source or policy. FrozenContinuationPolicy.chunk(observation, task,
episode, decision) returns [10,7] float32. Report.run() checks coverage and writes all-cell and paired results.

## Inputs

LIBERO-Goal split fraction .1 seed 0. Task-order episode IDs:
0:30,31; 1:71,72; 2:105,106; 3:144,145; 4:194,195; 5:245,246; 6:291,292;
7:330,331; 8:369,370; 9:414,415. These are the first two sorted unused TRAIN episodes per task,
excluding every prior physical-world source, including donors, preference TRAIN and calibration.
Anchors use max(1,min(floor(length/2/10)*10,length-10)), before any rollout outcomes are measured.
Policy: outputs/runs/goal_proposal_nopsi_seed0/checkpoints/step_003000.pt, frozen eval mean, seed 0,
512-square dual cameras, original instruction tokens, replanning every ten executed continuation actions.
Candidate grid: original, up to eight physical neighbours from the fixed prior donor index, half arm command,
Gaussian arm noise .1/.3 clipped to [-1,1], original gripper commands; seed [20261002,task,episode,1].

## Outputs

Immutable starts and per-branch traces; exact grid/checkpoint/source hashes; phase-local and full-path metrics;
all-task table; paired eligibility flips and Pareto reversals with source-cluster uncertainty; resource log.

## State Changes

Only new diagnostic data and code. BranchValidity's environment annotation is generalized to a read-only Protocol;
its observation logic is unchanged. Candidate actions are frozen before their branch outcomes are collected.

# Flow

## Main Flow

Fix IDs/config/checksums → CPU raw matching and prefix replay → freeze candidate manifests → verify first-success
and phase aggregation → render paired continuations → independently replay recorded actions → report and close.
Each candidate stops at first success, including within its ten-step block. Both modes end at the source action
horizon. A success before the prescribed anchor is recorded as an unusable start, never replaced by a favorable one.
Phase labels are prefix/candidate/continuation. Full-path feasibility is their conjunction; prospective feasibility
is candidate plus continuation. Validity remains the existing sampled screen, not continuous safety certification.
EEF metrics use simulator site poses at control-step boundaries consistently in both modes. This differs from the
previous pilot's cached robosuite pose observables; no cross-pilot absolute path/jerk comparison is inferred.

# Code Mapping

## Modules

Config: configs/preference/continuation_pilot.json and ceam/preference/continuation_config.py.
Records: ceam/preference/continuation_records.py; artifact_digest.py owns file checksums. Preparation: ceam/preference/continuation_preparation.py.
Read-only physics: ceam/preference/continuation_observer.py; ceam/preference/branch_validity.py.
Frozen policy: ceam/preference/continuation_policy.py. Environment adapter: continuation_environment.py.
Executor: ceam/preference/preference_continuation.py. Task orchestration: continuation_collection.py.
Analysis: ceam/preference/continuation_report.py. Replay verification: continuation_verification.py. Integrity sensitivity: continuation_integrity.py and
scripts/audit_continuation.py. Commands: scripts/prepare_continuation.py,
scripts/collect_continuation.py, scripts/report_continuation.py, scripts/verify_continuation.py.
Targeted verification: scripts/verify_continuation.py, including observed early-success boundaries. Runtime wrappers/resource logs: tmp/continuation/.

## Interfaces

Typed dataclasses for records/config; Protocol for read-only validity environment and continuation action source.
NPZ/JSON are storage boundaries, not untyped module interfaces.

## State

Config and source IDs are checked in. Artifact manifest names all candidates and source/checkpoint hashes.
The simulator owns state; tracing copies state without mutating it. Reports never overwrite raw measurements.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Implement immutable preparation and explicit records.

### Files

Config, records, observer, preparation, prepare command; BranchValidity annotation.

### Structures

Source specifications, phase traces and candidate start records.

### Affected Modules

Diagnostic input and measurement ownership.

### Dependencies

Existing raw matching, donor index, snapshot and sampled validity interfaces.

### Observation

Twenty fixed IDs are available; record source/candidate hashes, exact raw match and restored-state error. No GPU
needed. All 20 starts are usable, 239 candidates and 478 paired branches frozen. Prefixes pass the sampled screen.
Maximum restore state error is 2.5525648494895137e-07. Preparation ran on node-63 CPU.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Implement fixed-policy continuation and first-success stopping.

### Files

continuation_policy.py, continuation_environment.py, preference_continuation.py, continuation_collection.py,
collect_continuation.py, continuation_verification.py and verify_continuation.py.

### Structures

Policy protocol, immutable phase-local branch outcomes.

### Affected Modules

Continuation execution.

### Dependencies

Phase 1 records and starts.

### Observation

Observed early-success/horizon cases; rendered original against CPU replay; same-start state/proprio errors;
phase totals and policy checkpoint/encoder provenance. Task 0 completed 48 branches in 78.43 seconds of
collection; 34 successful branches stopped within a continuation block. Independent CPU replay of the two
original branches reproduces every success flag and phase validity, max state error 1.20e-11. Projected
remaining collection is roughly 10–15 minutes on one GPU, below the initial budget estimate.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Execute the paired grid and prepare its fixed analysis interface with no outcome-driven source or checkpoint changes.

### Files

New output banks and tmp/continuation resource wrappers; continuation_report.py, report_continuation.py; plan.md.

### Structures

Up to 480 unique branches, two modes for every candidate at every usable start.

### Affected Modules

Diagnostic collection.

### Dependencies

Phase 2 verified execution.

### Observation

One GPU on node-63 initially, at most two workers on that GPU. No node exceeds two project GPUs. Guard before every
launch; existing measured step times suggest approximately .5–1.5 GPU-hours, revised from task-0 measurements.
Log wall/CPU/GPU memory, node/device, commands and code hashes. Record unusable starts and all failures.
Task 0 used node-63 GPU 0; after completion other users occupied that node and the GPU guard refused four
subsequent launch attempts. Remaining tasks use two workers on node-61 GPU 0. No concurrent use of node-63 remains.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Analyze paired outcomes and verify recorded trajectories independently.

### Files

continuation_report.py, continuation_integrity.py, report/verify/audit commands, reports, docs/continuation_sensitivity_results.md and issues.md.

### Structures

All-cell table; per-start and task differences; joint-eligible Pareto reversals; eligibility changes; prefix rejection
counts. Two thousand source-cluster bootstrap samples are descriptive intervals, not independent branch samples.

### Affected Modules

Diagnostic validation and reporting.

### Dependencies

Phase 3 complete grid.

### Observation

Replay candidate zero for the first episode of each task in both modes using recorded actions in a CPU environment
(20 validation replays, not new candidate exploration). Compare per-step states, success, feasibility and timing;
verify phase concatenation, original-candidate fidelity, gripper preservation and source/checkpoint hashes.
Keep every original cell. Add sensitivity results on starts with exactly identical pre-continuation state traces;
report all replay validity fields, rather than aborting on numerical/count disagreement. This audit does not change
the collected outcomes, their feasibility threshold or candidate selection.
All 478 branches are collected; all input hashes match. Twenty independent original-branch replay results match
success traces and discrete validity. Full-grid success flips are 46/239, strict preference reversals 191/448;
bit-identical-start sensitivity gives 44/216 and 176/380. Detailed results: docs/continuation_sensitivity_results.md.
No three-seed policy or causal architecture claim.

## Phase 5

Phase Status: completed

Round: 5

### Goal

Reconcile results, resource use and follow-up boundaries.

### Files

plan.md, issues.md, opportunities.md, verified documentation and plans/continuation_sensitivity.md.

### Structures

Completed experiment record and separately scoped future hypotheses.

### Affected Modules

Persistent project state.

### Dependencies

Phase 4 evidence.

### Observation

All workers exited by 17:01:49 EDT; GPU collection ended 16:54:22 EDT. Timed CPU 0.2569 core-hours,
device-interval-union GPU 0.1214 GPU-hours, device peak 2919 MiB, outputs approximately 36.3 MB.
Docs, issue records, deferred opportunities and artifact checks are reconciled; no training or label publication.
Completed-plan commit and archive preserve the diagnostic evidence and the remaining ISSUE-045.

# Deferred Work

Representation training, capacity-matched scorers and observability ablations remain OPT-013. Wider strategy
exploration, self-supervised objectives, action publication and BC/RL are outside this diagnostic grid.
Prior physical-world/retrieval/preference and literature-review plans remain archived under plans/.

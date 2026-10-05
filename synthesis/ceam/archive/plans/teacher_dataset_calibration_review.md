# Plan

Plan Status: completed

# Problem

## Current

OPT-015 defines a teacher- and dataset-conditioned offline action generator, with direct task adaptation as an
alternative output. Dataset calibration concerns useful input/output coordinates and scales. The objective is
improvement over demonstrations, with task completion and physical validity retained; global optimality is unnecessary.

## Problem

The relationship between teacher priors, dataset compression/calibration and action improvement is unverified.
Normalization, model compression, behavior imitation and quality improvement require separate evidence.

## Goal

Audit primary literature and available interfaces until a concrete, discriminating experimental opportunity is
supported, or a necessary mechanism changes the direction materially. Deliver evidence, assumptions, boundaries
and a minimal proposed comparison. Research includes no training, model downloads, rollouts or dataset relabeling.

# Structure

## Modules

- Evidence review: primary papers and official implementations, recorded in docs/teacher_dataset_calibration_review.md.
- Candidate assessment: hypotheses and the proposed experimental decision in OPT-015 of opportunities.md.

## Responsibilities

The review distinguishes published observations, inspected interfaces and deductions. Candidate assessment owns
unresolved mechanisms and explicitly separates action editing from online task adaptation.

## State Ownership

plan.md owns research phase state; the review owns verified evidence; opportunities.md owns unselected mechanisms.
No runtime state or framework interface changes. Completed research is archived in plans/teacher_dataset_calibration_review.md.

# Interface

## Interfaces

Research inputs -> evidence records with source, mechanism, information access, observed outcome and limits.
Evidence records -> candidate assessment with improvement source, inputs/outputs, controls and rejection conditions.

## Inputs

OPT-015; docs/framework.md; relevant prior findings; primary publications and public implementation files;
read-only local artifact and environment metadata. Teacher weights may be referenced but are not downloaded.

## Outputs

Verified review; updated OPT-015; a decision to propose a specific experiment or record the identified deviation.

## State Changes

Sequential evidence audit and synthesis phases. Experimental proposals remain unapproved and unexecuted.

# Flow

## Main Flow

Confirm scope -> audit related mechanisms -> inspect strongest candidate interfaces -> compare against alternatives
and direct BC -> apply stopping criteria -> persist evidence and candidate decision -> close research.

# Code Mapping

## Modules

Evidence review -> docs/teacher_dataset_calibration_review.md. Candidate assessment -> opportunities.md, OPT-015.

## Interfaces

Source/evidence records -> review tables. Proposed generator/adaptation interface and controls -> OPT-015.

## State

Research status -> plan.md, then plans/teacher_dataset_calibration_review.md. Resource record -> review.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Audit teacher-based action editing, data-calibrated coordinates/compression and direct adaptation. Inspect the
strongest relevant methods beyond abstracts, including access requirements, artifacts and reported controls.

### Files

docs/teacher_dataset_calibration_review.md; plan.md.

### Structures

Source table, mechanism/claim distinction and read-only compatibility audit.

### Affected Modules

Evidence review only.

### Dependencies

None.

### Observation

Every material external claim links to a primary source. Distinguish execution latency from physical completion
time, policy likelihood from action quality, and published results from local replication. Record inaccessible
artifacts and information gaps. No GPU or experiment jobs are started.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Assess alignment and stop at an evidence-backed experimental cut or a necessary departure from the target direction.
An experiment is worth proposing when the teacher interface is concrete, the quality hypothesis is explicit,
controls can distinguish it from calibration/BC alone, and observations could reject the mechanism.

### Files

opportunities.md; docs/teacher_dataset_calibration_review.md; plan.md; plans/teacher_dataset_calibration_review.md.

### Structures

Candidate decision, minimal proposed comparison and resource/compatibility prerequisites.

### Affected Modules

Candidate assessment and research state.

### Dependencies

Phase 1 evidence and interface audit.

### Observation

Identify which parts of the original idea the candidate retains. Keep unknowns in opportunities.md. Do not
claim novelty, superiority or feasibility from literature resemblance alone. Record the stopping reason and
leave any subsequent experiment as proposed work.

Completed: the evidence-backed stopping point is the proposed frozen-teacher candidate-quality comparison in
OPT-015. Five primary papers, public interfaces, exact-transform deductions and local dataset fields are recorded
in docs/teacher_dataset_calibration_review.md. Runtime/model compatibility remains a prerequisite for future
execution. No experiments, model downloads or GPU allocations occurred.

# Deferred Work

Completed cross-episode representation research: plans/cross_episode_representation_research.md and
docs/cross_episode_representation_research.md. Nine primary sources, 360 CPU ridge fits and ten controls are
recorded; all workers exited and no GPU was used. Transferable interaction learning remains OPT-014;
source coverage and metric limitations are ISSUE-046. No neural pretraining, BC/RL update or label publication is active.

Completed continuation, preference, physical-world and retrieval studies are archived under plans/ with measured
results in docs/. Continuation-aware utility learning remains OPT-013; predictor improvements remain OPT-012.
Restore sensitivity is ISSUE-045, compatibility limits ISSUE-043/044, physical precision ISSUE-041/042 and
LIBERO-90 interface follow-up ISSUE-039. The prior literature audit is docs/long_horizon_matching_review.md.

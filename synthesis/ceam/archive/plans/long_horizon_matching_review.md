# Plan

Plan Status: completed

# Problem

## Current

Preference feedback improves same-start ranking, while direct success/feasibility selection remains unreliable.
The fixed ten-step replacement and source suffix restrict which long-term consequences the labels measure.

## Problem

Long-horizon match semantics, continuation dependence and preference credit assignment are unresolved.
The observed-feasibility target contains prefix information absent from current matcher inputs.

## Goal

Review primary literature on trajectory preferences and reusable long-term representations; audit current
input/label interfaces; produce source-grounded findings and a bounded next-experiment proposal. This is a
research review with zero training or simulator collection and no change to the framework's public definition.

# Structure

## Modules

Primary-source evidence review; current-code label/input audit; proposed follow-up protocol.

## Responsibilities

The review records verified paper mechanisms and assumptions. The audit records actual implementation semantics.
Follow-up hypotheses and unimplemented choices remain in opportunities.md, separate from verified documentation.

## State Ownership

docs/long_horizon_matching_review.md owns the review. issues.md owns observed implementation limitations.
opportunities.md OPT-013 owns candidate research changes; plan.md owns this review's status.

# Interface

## Interfaces

Each source entry identifies its primary URL/version, inspected mechanism, policy/horizon assumptions and scope.
The audit maps each label to its input information and continuation owner. The follow-up protocol maps proposed
observations to existing code modules and states its stopping criteria.

## Inputs

Published primary papers/preprints, existing Agent4VLAData snapshot, framework and completed CEAM experiments,
and ceam/preference/{preference_collection,branch_validity,preference_data,preference_encoding}.py.

## Outputs

A primary-source comparison, input/label audit, recommendation with evidence boundaries and minimal validation
sequence. No literature result is represented as a CEAM measurement.

## State Changes

Research documentation, unresolved issues and opportunities only. Existing experiment artifacts are read-only.

# Flow

## Main Flow

Inspect primary mechanisms → audit current inputs/labels → compare architectural implications → record proposed
validation choices → verify references and reconcile persistent state.

# Code Mapping

## Modules

Evidence and audit: docs/long_horizon_matching_review.md. Unresolved limits: issues.md ISSUE-044.
Candidate validation protocol: opportunities.md OPT-013. Review archive: plans/long_horizon_matching_review.md.

## Interfaces

Existing preference modules are inspected only; no executable interface is changed.

## State

Primary citations and code paths belong to the review; proposed changes belong to OPT-013.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Verify mechanisms and limitations of primary literature on long-term representation and trajectory preference.

### Files

docs/long_horizon_matching_review.md.

### Structures

Source/version matrix with assumptions, evidence and exact differences from current CEAM.

### Affected Modules

Research documentation.

### Dependencies

Existing framework, source files and completed experiment evidence.

### Observation

Eight primary works inspected: successor features, forward-backward representation, contrastive RL, one-step FB,
regret preferences, CPL, Preference Transformer and TD-JEPA. Equations and policy dependencies were checked;
CEAM performance is not inferred from their benchmark results.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Reconcile current label/input semantics and specify a minimal follow-up validation proposal.

### Files

docs/long_horizon_matching_review.md, issues.md, opportunities.md, plan.md and archive.

### Structures

Audit table and proposed module-to-observation mapping, with immutable-prefix and continuation boundaries.

### Affected Modules

Research documentation and persistent state.

### Dependencies

Phase 1.

### Observation

Eight primary-source entries and three mechanism derivations are recorded in docs/long_horizon_matching_review.md.
The audit verifies current 9D proprio, immutable-prefix target inheritance, fixed suffix/horizon and missing
phase-local records. The existing focused executor has different first-success semantics and is not a drop-in
preference continuation adapter. ISSUE-044 records the evidence gap. OPT-013 proposes at most 480 paired
continuation branches followed by a stop for interpretation, with later architecture controls kept separate.
No training, simulator collection or GPU task ran; experiment outputs are unchanged. References, code paths,
persistent state boundaries and documentation formatting are checked before local commits.

# Deferred Work

Completed preference feedback: plans/preference_feedback.md and docs/preference_feedback_results.md.
Compatibility generalization and physical acceptance remain ISSUE-043/044; subsequent feedback, smoothness and
wider-strategy work is OPT-013. No refined action dataset, BC/RL run or new experiment is active.

Completed retrieval and physical-world pilots: plans/retrieval_refinement.md and plans/physical_world.md.
Physical predictor improvements remain OPT-012; precision/confidence and replay limits remain ISSUE-041/042.
LIBERO-90 R6 interfaces remain blocked by ISSUE-039. Future phases require their own concrete plan.

# Deferred LIBERO-90 and representation work

Plan Status: proposed

LIBERO-90 R6 remains blocked by ISSUE-039. Native normalization and topology interfaces are defined in `plans/libero90_r6_evaluation.md`. PCA/head training and transfer evaluation have not started. Next-generation representation learning remains a separate proposal under OPT-009.

## Phase 14: fit and evaluate the LIBERO-90 R6 baseline

Phase Status: blocked

Round: 10

### Goal

Fit PCA-256 on LIBERO-90 TRAIN frames, train the effect head for 70,000 steps, and measure its transfer and local candidate distinguishability.

### Files

LIBERO-90 interaction configs, PCA outputs, R6 run outputs, `docs/libero90_prep.md`.

### New structures

LIBERO-90 PCA and R6 baseline checkpoint.

### Affected modules

Existing PCA fitter, interaction trainer and representation diagnostics.

### Dependencies

Phases 6 and 8.

### Observation

Blocked by ISSUE-039: the existing full-store scene loader fails on heterogeneous task topologies, and the common-cache
model validation cannot evaluate Goal R6_all with native statistics alongside the LIBERO-90 refit.
Revised interface design is `plans/libero90_r6_evaluation.md`; it must be implemented and validated before execution.
No PCA/head job has started. Required observations remain:

R6_all PCA explained variance versus refit PCA; held-out prediction by store and scene; smooth-random and π-noise within-state variance share, effective rank, cosine, centered cosine, Spearman and state-conditioned R²; training time and memory.

## Phase 15: specify the next action-representation experiment

Phase Status: pending

Round: 11

### Goal

Use F1 and LIBERO-90 evidence to define a method-neutral self-supervised representation experiment. This phase writes the design and experiment grid; it does not implement a new encoder.

### Files

`docs/action_representation_learning.md`, `opportunities.md`, a subsequent `plan.md` after approval.

### New structures

Proposed `ActionRepresentationProtocol`, dataset-super-set manifest, candidate objective grid and fixed evaluation protocol.

### Affected modules

Architecture and experimental design only.

### Dependencies

Phases 7 and 14.

### Observation

The design maps every candidate objective to framework clauses, holds sampler/q/m evaluation fixed, separates non-collapse from useful separability, defines TRAIN/HELDOUT at episode and state level, and enumerates all experiment cells with resource estimates.

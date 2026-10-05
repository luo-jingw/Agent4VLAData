# Plan

Plan Status: completed

# Problem

## Current

The teacher/dataset review identifies a frozen flow decoder as an action-editing interface. CEAM has compatible
EEF observation fields, simulator replay and validity observers. A pinned public pi0.5 teacher is installed in
an isolated runtime; float32 computation passed the numerical reconstruction gate.

## Problem

Frozen-teacher reconstruction now passes a 0.001 max arm-command error gate at 40 Euler steps and 60 inverse
iterations with float32 compute. All 60 source inversions passed the gate. TRAIN calibration is frozen and all 600 query candidates are generated.
All 600 CPU branch outcomes and 36 common-teacher continuation cells are complete. Calibration has no stable
added Pareto yield; continuation changes the success/efficiency trade-off. All workers exited and GPU 0 is released.
Measured results: docs/teacher_editing_results.md; unresolved interpretation: ISSUE-047.

Whether dataset-calibrated teacher-space edits yield better successful/valid candidates than equally sized
direct action-space edits is unknown. Inversion error, edit magnitude and source continuation can confound results.

## Goal

Qualify a public pi0.5 LIBERO teacher, then compare original/reconstruction/direct/noise/calibrated-noise proposals
on a bounded paired pilot. Measure all candidates, quality, numerical error and resource use. No policy training
or published action-label replacement. This is an independent teacher-based generator experiment.

# Structure

## Modules

- Teacher artifacts/runtime: anonymous pinned public assets and a separate project-local Python environment.
- Teacher adapter: explicit observation preprocessing, normalization, cached velocity, decoding and inversion.
- Dataset preparation: TRAIN calibration chunks and held-out edit locations, images and replay provenance.
- Calibration/candidates: statistics and a fixed candidate grid.
- Execution/reporting: paired full-task branches, validity and time/path/jitter measurements.

## Responsibilities

The teacher owns its frozen parameters and original normalization. Calibration owns TRAIN statistics. The
generator owns candidate arrays. Existing simulator interfaces alone mutate simulator state. Reports own derived
comparisons, not action labels. Teacher and simulator environments communicate through explicit NPZ/JSON records.

## State Ownership

plan.md owns phases. outputs/teacher_editing/pilot_v1 owns manifest, prepared data, candidates, metrics and resource
records. runtimes/openpi/{pyproject.toml,uv.lock} owns teacher dependencies; root dependencies remain unchanged.

# Interface

## Interfaces

TeacherAdapter.encode(observation)->context; decode(context, noise)->normalized chunks;
invert(context, normalized chunks)->noise and reconstruction diagnostics. Calibration maps noise edit scales.
Typed records define source episode/frame, instruction, images, EEF state, original commands and artifact paths.

## Inputs

Public OpenPI source revision 215abfb217dbac7d5f1273282331b9b1866c0479; public pi05_libero checkpoint with pinned
GCS object generations; Goal TRAIN/HELDOUT split 0.1, seed 0; raw HDF5 replay sources; existing physics/validity APIs.

## Outputs

Frozen-teacher qualification and complete candidate table, source provenance, resource log and bounded conclusions.

## State Changes

Raw data/old weights stay read-only. New downloads and derived artifacts are project-local. No training updates.

# Flow

## Main Flow

Anonymous source/weight acquisition -> isolated runtime -> data and adapter qualification -> TRAIN calibration
and fixed candidate banks -> CPU source-suffix execution -> analysis. If meaningful gains survive controls,
evaluate deciding cases with a shared teacher continuation within the resource ceiling; otherwise report the
conditional/negative result without scaling training.

# Code Mapping

## Modules

scripts/prepare_teacher_assets.py; runtimes/openpi/pyproject.toml and uv.lock; scripts/teacher_env.sh;
ceam/teacher_editing/{records,teacher_adapter,dataset_preparation,dataset_calibration,candidate_generator,
branch_evaluation,report}.py; teacher_preprocessing.py, flow_kernel.py, replay_storage.py and observation_rendering.py
within ceam/teacher_editing isolate teacher transforms, numerical kernels, snapshot I/O and rendering; scripts/{prepare_teacher_data,qualify_teacher,generate_teacher_candidates,
evaluate_teacher_candidates,report_teacher_editing}.py.

## Interfaces

records.py owns structured cross-process data. teacher_adapter.py owns the teacher Protocol and concrete adapter;
dataset_preparation.py reads existing dataset/simulator interfaces without changing their state ownership.

## State

outputs/teacher_editing/pilot_v1; configs/teacher_editing/pilot.json; docs/teacher_editing_results.md;
plan.md, issues.md and PROJECT.md. Public source and weights -> outputs/teacher_assets/.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Prepare pinned assets and an isolated locked runtime; qualify teacher-compatible preprocessing and inversion.

### Files

scripts/prepare_teacher_assets.py; runtimes/openpi/; scripts/teacher_env.sh; ceam/teacher_editing/records.py,
teacher_adapter.py, dataset_preparation.py; scripts/prepare_teacher_data.py and qualify_teacher.py; PROJECT.md.

### Structures

Typed artifact, observation, source and qualification records; teacher Protocol; local file interchange.

### Affected Modules

Teacher runtime and dataset adapter only.

### Dependencies

Published source/weight availability and empty GPUs under the existing guard. Assets are anonymous public reads;
no GCS, Hugging Face or global account credentials are used.

### Observation

Qualification grid: first TRAIN episode of each selected task, both anchors; Euler steps {10,20} × fixed-point
iterations {5,10}. Admission requires finite reconstruction and arm-command max error ≤ 0.001 on every anchor;
select the admitted cell with the fewest velocity evaluations. All cells remain in the report.
Follow-up grid: the same six source chunks with Euler steps {20,40} × iterations {30,60}, plus four
matched teacher-generated controls at the first location, seed 0. This increases solver work without selecting
held-out outcomes. BF16 reconstruction and cache-boundary errors can be comparable to the required tolerance.
Precision isolation: repeat the 40-step/60-iteration cell on the same six source chunks and one teacher-generated
control with float32 compute. Checkpoint parameters retain identical BF16 values; only model compute dtype changes.
The official-reference call uses the same compute dtype. Retain both precision results. If neither compute mode
admits a solver, record qualification as blocked and do not launch candidates. No optimizer redesign is included.
Compare the official sampler with the same-graph implementation and separately cached prefix implementation.
The former must agree within 0.001 normalized units; report cache-boundary numerical effects separately.
Run every inversion diagnostic even if parity is unresolved, but do not admit a solver with unresolved parity.
Record hashes, versions, normalization/action-convention round trips, latent shape, inversion residuals, gripper
agreement, per-call time and peak GPU memory. Prefer one GPU, maximum two per node. Initial ceiling: 4 GPU-hours
and 40 GB new disk; qualification gives the estimate before full pilot launch. All remote work uses cluster_run.sh.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Qualification admitted 40 Euler steps / 60 fixed-point iterations with float32 compute and frozen BF16 parameter
values. Six source max arm errors ≤ 0.000382; teacher-generated control 0.000480; numerical reference parity zero.
Warm decode at 10 steps ~0.026 s; 40-step inversion ~6.1 s. Estimated pilot generation <1 GPU-hour, plus
qualification/rendering <0.2 GPU-hour, within the 4 GPU-hour ceiling.

Fix and run calibration/candidate generation after qualification. Task IDs 0, 3, 6 cover fixture articulation and
object transport/grasping. Start with first 8 TRAIN and first 2 HELDOUT episodes per task, anchors at one-third
and two-thirds of source first-success time. Ten-step edits; candidate seeds 0,1,2; 4 edits per seed per mechanism.
Data support and all exclusions are recorded. Sample count can only change through an explicit plan update.

### Files

configs/teacher_editing/pilot.json; ceam/teacher_editing/dataset_calibration.py and candidate_generator.py;
scripts/generate_teacher_candidates.py; generation_config.py; branch_evaluation.py and execution_records.py
under ceam/teacher_editing, scripts/evaluate_teacher_candidates.py; report.py and scripts/report_teacher_editing.py; plan.md.

### Structures

TRAIN-only per-dimension scales; source and accurate round-trip controls; direct-action, teacher-noise and
calibrated-noise candidate banks. Teacher-only proposals are a separate reference. Perturbation radii are fixed
from qualification/TRAIN data, not HELDOUT outcomes; match decoded arm-command RMS and preserve gripper schedule
in the local editor pilot. Physical RMS target is 0.03 in simulator arm-command units, with 0.003 matching
tolerance. Gaussian direct edits use teacher quantile scales. Noise directions use the same draws for calibrated
and uncalibrated variants; latent-radius bisection uses at most 18 decodes, without simulator feedback. Radius
matching failures and command-limit violations remain recorded proposals. No clipping. Calibration is the
per-coordinate TRAIN standard deviation, floored at 0.001 and rescaled to unit global RMS.
Teacher-only reference uses 12 proposals per location with its own binarized gripper. This restriction isolates arm edits and does not define the full generator scope.

### Affected Modules

Calibration and proposal generation; prepare the branch-observer implementation while generation runs.
Phase 3 owns execution and interpretation after the complete candidate bank is frozen.

### Dependencies

Phase 1 qualified decoder and prepared source records.

### Observation

Retain every generated candidate and decoding/scale error, including clipping/rejections. Report teacher forward
evaluations and elapsed time separately from the common execution budget. Three seeds refer to candidate draws.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Each branch replays the full original prefix from the matched raw initial state before inserting its candidate;
this avoids repeated mid-trajectory restoration as a branch-comparison confound. Metrics use actual simulator
control-boundary EEF positions, not delayed observables.

Execute all paired branches through source continuation and report candidate yield, success, observed validity,
first-success steps, EEF path and RMS jerk. Source controls identify simulator/reconstruction errors. Assess
continuation sensitivity for deciding positive cases before making a general-editor claim.

### Files

ceam/teacher_editing/branch_evaluation.py and report.py; scripts/evaluate_teacher_candidates.py,
report_teacher_editing.py; docs/teacher_editing_results.md; issues.md; plan.md.

### Continuation diagnostic

The 600 source-suffix cells are complete. Teacher-noise edits produce 24 Pareto improvements, calibration 19
and direct edits 16. Calibration has no added Pareto yield. The strongest teacher-related example reduces
174 to 129 steps, warranting the already scoped continuation check before attributing the improvement generally.

Fixed follow-up: locations e18_t58 and e288_t33, the two largest calibrated step improvements from different
episodes. For each mechanism, choose the fastest Pareto-improving source-suffix candidate, breaking ties by path;
if none is Pareto-improving, choose the fastest successful valid candidate. Include original, reconstruction,
direct, teacher noise, calibrated noise and teacher-only reference: 2 locations × 6 mechanisms × 3 common
continuation seeds = 36 cells. Selection is verifier-assisted on source-suffix outcomes, not a deployable ranking.
After the source prefix and ten-step candidate, all branches use the same frozen float32 teacher at ten-step
replanning intervals, a 400-step full-task ceiling and common noise streams keyed by continuation seed/chunk.
Record all failures, invalid teacher commands, full-task metrics and fresh images; do not clip actions.

Process interface: typed JSON requests identify NPZ observations, response action files and noise seeds. The
simulator process owns rendered state, and a separate teacher process owns JAX weights. Both use node-61 GPU 0;
there is no shared in-process dependency environment. Files: ceam/teacher_editing/{teacher_rpc,teacher_client,
teacher_server,rendered_branch,continuation_evaluation,continuation_report}.py; scripts/{teacher_server.sh,serve_teacher_actions.py,
evaluate_teacher_continuation.py,report_teacher_continuation.py}; configs/teacher_editing/continuation.json. Runtime cap 1 GPU-hour; combined
runs remain below the initial 4 GPU-hour and 40 GB bounds. No additional solver or generator search is included.

Completed outcome: all 36 cells ran. Calibration rescues two paired original failures but yields no paired Pareto
improvement among jointly successful cases. The selected source-suffix efficiency gains do not generally transfer
to the common teacher continuation. Total GPU wall time for the complete pilot is approximately 0.442 GPU-hours.

### Structures

Full-grid candidate outcomes, paired quality differences, verifier-assisted best-of-K and resource summary.

### Affected Modules

Execution observer and reporting.

### Dependencies

Phase 2 frozen candidate banks; existing PhysicsTaskEnv and BranchValidity.

### Observation

Quality flags are fixed before branch execution: faster = at least one fewer control step; shorter = at least
1 mm less EEF path; smoother = at least 1% lower RMS jerk. A Pareto improvement additionally allows no step
regression, at most 0.1 mm path increase and at most 0.1% jerk increase. Success, observed feasibility and edit
magnitude agreement are required. Report the raw values alongside these descriptive thresholds.
Retain all failures and restoration differences. Original fallback success does not count as generator success.
Use observed validity rather than physical safety certification. A source-suffix-only conclusion remains
conditional. Close if the mechanism is unhelpful, qualification fails, or the measured budget is exhausted;
record the exact evidence and remaining unrun cells rather than substituting another mechanism.

# Deferred Work

Completed teacher/dataset calibration review: plans/teacher_dataset_calibration_review.md and
docs/teacher_dataset_calibration_review.md. Five primary papers and public interfaces were audited; no model
or experiment jobs ran and no GPU was used. OPT-015 retains codebook and adaptation hypotheses.
The editing pilot above promotes the interface qualification and candidate comparison; remaining adaptations stay deferred.

Completed cross-episode representation research: plans/cross_episode_representation_research.md and
docs/cross_episode_representation_research.md. Nine primary sources, 360 CPU ridge fits and ten controls are
recorded; all workers exited and no GPU was used. Transferable interaction learning remains OPT-014;
source coverage and metric limitations are ISSUE-046. No neural pretraining, BC/RL update or label publication is active.

Completed continuation, preference, physical-world and retrieval studies are archived under plans/ with measured
results in docs/. Continuation-aware utility learning remains OPT-013; predictor improvements remain OPT-012.
Restore sensitivity is ISSUE-045, compatibility limits ISSUE-043/044, physical precision ISSUE-041/042 and
LIBERO-90 interface follow-up ISSUE-039. The prior literature audit is docs/long_horizon_matching_review.md.

# Dataset-backed action retrieval pilot

Plan Status: completed

# Problem

The local refinement pilot edits one ten-step block and does not test reuse of alternative demonstration actions.
Guided shortlisting improves over random for its fixed bank but reliable policy benefit over original remains
unestablished. A bounded retrieval experiment must measure candidate coverage and task-compatible ranking before
feedback training or dataset publication. The Agent4VLAData survey provides reference cautions about state aliasing,
execution feedback and downstream validation.

## Goal

Collect ten Goal tasks × first three HELDOUT demonstrations × three anchors at 1/4, 1/2 and 3/4 of the replay's
first-success length, rounded down to ten-step boundaries and constrained to contain a full chunk. Failed source
replays remain recorded and use their available length. Compare two retrieval modes on every start. Up to eight
distinct donor episodes plus original form each bank. Retain sources with zero donor coverage.

Donors are the first twelve TRAIN episodes per task under split fraction 0.1, seed 0. Index every tenth frame with
ten future source actions, through first success or the end of a failed donor. Donors and queries are disjoint.
Same-task retrieval controls instruction identity and movable-object topology; it does not test cross-task transfer.
Copy donor arm increments directly under the same robot/controller conventions and retain the query's gripper
sequence. This isolates arm-action reuse. No SE(3) retargeting or new action generator is introduced.

# Structure

- Source reader owns exact raw-demo matching and source actions; existing environments own simulator state.
- Physical descriptor reader owns read-only geometric indexing. Simulator object/contact information is privileged
  retrieval input; match itself retains the existing visual/proprio input.
- Task-local action index owns donor chunks, descriptors and donor provenance.
- Preparation owns same-start branch measurements and restore recipes.
- Scoring restores and encodes one actual query observation, then invokes frozen r/q/m.
- Reporting owns complete-grid validation and branch-level comparison. This experiment does not train policies,
  publish changed labels or establish full closed-loop policy SR.

# Interface

- `RetrievalEpisodeSource.read(record) -> RetrievalEpisode`: actions, raw name, initial state, fixture poses.
- `RetrievalStateReader.read(gripper_command) -> RetrievalState`: normalized descriptor, EEF xyz, contact signature and command mode.
- `DemoActionIndex.neighbours(state, mode, limit) -> tuple[RetrievedAction, ...]`: stable nearest order, one donor per
  episode; geometry uses EEF xyz, physical uses the full descriptor and equal contact signatures. Both require equal
  initial gripper command. No distance threshold or geometric similarity is treated as a guarantee of validity.
- `RetrievalPreparation.run_task(task_id) -> None`: task index, provenance and all query bank NPZs.
- `RetrievalBank`: query/task/anchor/mode, snapshot and producing action, expected proprio, candidate chunks and donor
  ids/frames/distances, measured success, steps, command variation, translation/rotation path length, acceleration
  RMS, jerk RMS and control timestep. Arrays have a variable candidate dimension including original at index zero.
- `RetrievalScoring.run(tasks, legacy_dir=None) -> None`: per-bank scores and render/restore measurements, no outcome
  fields enter scoring. Existing `FrozenMatchDemoScorer` accepts nonempty `[K,10,7]` chunks; nine-candidate behavior
  stays numerically unchanged.
- `RetrievalReport.run() -> None`: all cells, coverage, candidate oracle, task/source paired descriptive differences,
  resource intervals and explicit scope limits.

# Flow

Replay donor episodes on CPU → collect task-local physical descriptors and actions → replay HELDOUT sources →
choose three anchors → retrieve candidates with each mode → restore each common start → execute a candidate chunk
and the unchanged demonstration suffix until success or source horizon → measure outcomes → restore/render query
starts on one GPU → score all candidates under frozen checkpoints → compare selectors and report.

Replay and record all baseline failures. Do not discard an alternative because it fails or looks unusual.
Physical motion metrics cover the common prefix and changed continuation; failed and successful trajectories are
reported separately. Source command variation is a secondary measurement, not a physical smoothness proxy.

# Code Mapping

| Responsibility | File |
|---|---|
| Run config | `ceam/refinement/retrieval_config.py` |
| Source and raw match | `ceam/refinement/retrieval_episode_source.py` |
| Geometry/contact descriptors | `ceam/refinement/retrieval_state.py` |
| Retrieval protocol | `ceam/refinement/action_retriever.py` |
| Retrieved action record | `ceam/refinement/retrieved_action.py` |
| Donor lookup | `ceam/refinement/demo_action_index.py` |
| Bank persistence | `ceam/refinement/retrieval_bank.py` |
| CPU branch collection | `ceam/refinement/retrieval_preparation.py`, `scripts/prepare_retrieval_refinement.py` |
| Frozen scorer boundary | `ceam/refinement/frozen_match_demo_scorer.py` |
| GPU observation/scoring | `ceam/refinement/retrieval_scoring.py`, `scripts/score_retrieval_refinement.py` |
| Complete result grid | `ceam/refinement/retrieval_report.py`, `scripts/report_retrieval_refinement.py` |
| Run configuration | `configs/refinement/retrieval_pilot.json` |
| Runtime/resource evidence | `tmp/retrieval_refinement/`, `tmp/runlogs/retrieval_refinement/` |
| Artifacts | `outputs/retrieval_refinement/pilot_v1/` |
| Verified report | `docs/retrieval_refinement_results.md` |

# Experiment Grid

- Retrieval: geometry / physical; 90 starts each, maximum 180 banks and 1,620 branches.
- Frozen query checkpoints: original R6_all q seed 0 (`outputs/runs/query_match_r6all_seed0/checkpoints/step_005000.pt`)
  and focused q seeds 0/1/2 (`outputs/focused_query/runs/query_match_focused_seedN/checkpoints/step_005000.pt`).
- Selectors on each common bank: original, nearest donor, exact uniform donor and all-bank expectations, match argmax including
  original, and successful minimum-step oracle with command variation as tie-breaker. Also compare a match top-four
  donor shortlist plus original against exact random four-donor shortlisting using the same measured acceptance cost.
- Descriptor scales: translation 0.05 m, rotation matrix components 0.5, arm joints 0.3 rad, finger joints 0.01 m,
  fixture slide 0.05 m / hinge 0.3 rad. These define distance weighting and are pilot variables fixed for this grid.
- Chunk length 10, stride 10, original suffix continuation, native 20 Hz simulator control. Identical rendering,
  normalization and candidate outcomes for all query checkpoints. No test-time outcome oracle for match argmax.
- Branch outcome rates are conditional on demonstration suffix continuation; they are not learned-policy SR.

# Implementation Phases

## Phase 1: implement and observe interfaces

Phase Status: completed

Round: 1

Files: mapped retrieval modules/scripts/config and frozen scorer. Implement one task/source/anchor smoke, expose
raw matching, shapes, split separation, restore errors, duplicate donor counts, baseline replay agreement and
scorer nine-candidate numerical equivalence. Runtime scripts log elapsed/RSS and GPU resource samples.

Phase-1 observations: task 0, HELDOUT episode 4, anchor 30; both banks contain nine candidates. CPU restored proprio maximum error 3.46e-8; GPU restoration is exact. Original succeeds in 129 steps under both modes. Legacy nine-candidate scores and restored proprio match saved pilot results exactly. CPU smoke elapsed 66.30 s, peak RSS 2,102,832 KiB. Evidence: `outputs/retrieval_refinement/smoke/`, `tmp/runlogs/retrieval_refinement/`.

## Phase 2: collect complete CPU grid

Phase Status: completed

Round: 2

Depends on Phase 1. Run all ten tasks and both retrieval modes, no new GPU allocation for branch physics. Record
donor/source failures and missing-neighbour coverage explicitly; validate exact planned source and anchor coverage.

Phase-2 collection completed: 180 banks, all ten task manifests, 120 TRAIN donor episodes and 30 HELDOUT source episodes. All original-source outcomes and failed alternatives are retained. CPU work used node-61/node-63 with no GPU allocation.

## Phase 3: score every frozen checkpoint

Phase Status: completed

Round: 3

Depends on Phase 2. Score all available banks under all four checkpoints on a single free GPU; each node remains
limited to two project GPUs. Record restore errors, peak memory, timing, resource samples and checkpoint hashes.

Phase-3 scoring completed on one GPU on node-63: all 180 banks scored under four checkpoints; rendered state restoration error is zero throughout. Evidence: `outputs/retrieval_refinement/pilot_v1/scores/`, `tmp/runlogs/retrieval_refinement/score_full/`.

## Phase 4: report and reconcile

Phase Status: completed

Round: 4

Depends on Phase 3. Produce every grid cell, task breakdowns, paired source bootstrap intervals conditional on
the frozen checkpoints, coverage and resource totals. Record verified observations in docs and unresolved findings
in issues. Further feedback training, action-space retargeting and BC retraining require a separate measured plan.

Phase-4 observations: all 64 comparison cells, 180 banks and 1,618 branches reported. Original replay outcome and step discrepancies are zero. Physical retrieval supplies faster successful alternatives at 71/90 starts; frozen original-q direct match succeeds at 85.56%. All eight shortlist-versus-random step intervals span zero. Minimum-step physical oracle increases jerk from 6.933 to 7.852 m/s³. GPU resource samples cover 0.0183 GPU-hours, one card at a time; all workers exited. Verified results and limits: `docs/retrieval_refinement_results.md`. Reference review: `docs/agent4vla_reference_review.md`. Remaining mechanism questions: ISSUE-040 and OPT-011.

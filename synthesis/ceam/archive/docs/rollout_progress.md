# True ψ effects of rollout transitions against scene progress

Δ(o, a) is trained to regress the true effect z = ψ(o′) − ψ(o) (`docs/training.md`), so expanding Δ's training data moves its match m(Δ̂) = cos(Δ̂, q_l) toward the true match m(z) = cos(z, q_l); stage one (ψ, q_l) is unchanged. This page measures, on the closed-loop rollouts of data B (`docs/dynamics_data.md`), how often the true match of an executed chunk is positive when the scene does not progress toward the task's goal, and how the baseline Δ scores the same chunks. A chunk with m(z) > 0 is one that a Δ reproducing z exactly would judge valid.

## Data and code

- **Script**: `scripts/analyze_rollout_progress.py`, stages `scenes` (simulator), `effects` (GPU), `report` (CPU), `render` (simulator).
- **Rollouts**: `outputs/transitions/rollouts_r0/` (23,202 transitions: 19,812 `rollout`, 3,390 `rollout_stay`) and the traces `outputs/closed_loop/rollouts_r0_frozen_observed_rule/task<t>_match/trace.jsonl` (frozen retrieval, observed effects, rule 32/128/512, 100 sampled starts per task, env seed 1). A transition joins its trace on (task, episode, t): the scene before decision t is the scene before the transition; the scene before decision t + 1, or the trace's final scene after the last decision, is the scene after it.
- **Demonstrations**: every TRAIN episode (390), non-overlapping chunks t → t + 10 at t = 0, 10, 20, … with t + 10 ≤ T − 1 (4,534 chunks; 108 dropped because the goal already held at t, leaving 4,426). ψ from the feature cache, the chunk from the action cache.
- **Models**: stage one `outputs/runs/goal_instruction_contrast/checkpoints/step_005000.pt` (frozen trunk in float32 without autocast; q_l = the query direction of the task's original instruction, as in `scripts/run_closed_loop.py`); baseline Δ `outputs/runs/goal_dynamics_short/checkpoints/step_003000.pt`, executed chunk standardized with the action cache.
- **Outputs**: `outputs/analysis/rollout_progress_r0/`:
  - `scenes/task<TT>.json`: goal geometry of the 100 re-created sampled starts, and the replayed demonstration scenes;
  - `effects/baseline/{rollout,demo}.npz, manifest.json`: m(z), ‖z‖, m(Δ̂), cos(Δ̂, z), ‖Δ̂‖ per transition;
  - `report/baseline/{report.md, transitions.csv}`: all tables, one labeled row per transition; `report/baseline_d2cm/`: the threshold variant;
  - `renders/`: before / after images of example transitions;
  - `logs/`: launchers and stage logs.
- **Runs**: node-62 (L40S). `scenes` at commit `7f5d2ff`, 10 processes on GPUs 0, 2, 4, 5, 6, 168–226 s each; `effects` at `7f5d2ff` on GPU 7, 29 s; `report` and `render` at `1395735`. Launchers (`launch_stage1.sh`, `run_report.sh`, `run_render.sh`) and logs in `outputs/analysis/rollout_progress_r0/logs/`. `/tmp/robosuite.log` on node-62 was absent before and after; on node-71, which ran no simulator, it is unchanged (4522 bytes, md5 `201a68b5…`).

## Goal geometry

Traces store object positions and joint values but not the region sites that are goal targets. The `scenes` stage recovers them:

- **Rollout starts**: a fresh `LiberoTaskEnv` with env seed 1 runs `reset_sampled(10)` 100 times per task. Sampled start e depends only on the seed and the number of resets before it (`docs/libero_sim.md`), so this re-creates the rollout starts. After each reset the site pose and size are read through the read-only `libero_env`. All 1000 re-created starts equal the traced initial scenes exactly: max |Δ| over object positions and joint values 0.0, goal check equal in 1000 of 1000.
- **Moving sites**: fixtures do not move. The top-drawer interior (task 1) is attached to the drawer's slide joint and moves with it: S(q) = S(q_ref) + axis · (q − q_ref), with the world axis read at the start.
- **Demonstrations**: `DemoStateRestorer` replays each TRAIN episode. Frames 0, 10, 20, … are restored, and their scenes read with `SceneProbe`; the goal geometry is read at frame 0.
- **Renders**: the `render` stage re-simulates a rollout episode from its re-created start and the stored executed chunks. Every re-simulated scene equals the traced one (max |Δ| 0.0 over all decisions of the four rendered episodes).

## Progress labels

Each goal conjunct gives one or more components, each with a distance to LIBERO's goal set; LIBERO's predicates are in `libero/libero/envs/object_states/base_object_states.py` and `objects/site_object.py`.

| Task | BDDL goal | Components and distance to the goal set |
|---|---|---|
| 0 open the middle drawer | (Open wooden_cabinet_1_middle_region) | drawer slide q: max(0, q − (−0.14 m)) (WoodenCabinet open: q < max of default_open_ranges = −0.14) |
| 1 open the top drawer, bowl inside | (In akita_black_bowl_1 wooden_cabinet_1_top_region) | bowl to the drawer-interior box (in_box: site ± \|R·size\|, lower z bound 0.01 m lower), box taken at the drawer's after-scene q if open, else at q = −0.14; top-drawer slide as for task 0 |
| 2 push the plate to the stove front | (On plate_1 main_table_stove_front_region) | plate to the region's `under` box: δ = R·(p − S), \|δ_x\| < 0.04, \|δ_y\| < 0.04, 0.002 < δ_z < 0.107 |
| 3 bowl on the plate | (On akita_black_bowl_1 plate_1) | sqrt(max(0, ‖bowl_xy − plate_xy‖ − 0.03)² + max(0, plate_z − bowl_z)²), both at their current positions |
| 4 bowl on the stove | (On akita_black_bowl_1 flat_stove_1_cook_region) | bowl to the cook region's `under` box (half extents 0.075, 0.075; 0.0025 m thick) |
| 5 bowl on top of the cabinet | (On akita_black_bowl_1 wooden_cabinet_1_top_side) | bowl to the cabinet top's `under` box (0.125 × 0.094 half extents) |
| 6 cream cheese in the bowl | (On cream_cheese_1 akita_black_bowl_1) | as task 3, cream cheese to bowl |
| 7 wine bottle on the rack | (On wine_bottle_1 wine_rack_1_top_region) | bottle to the rack region's `under` box (tilted site frame) |
| 8 wine bottle on top of the cabinet | (On wine_bottle_1 wooden_cabinet_1_top_side) | bottle to the cabinet top's `under` box |
| 9 turn on the stove | (Turnon flat_stove_1) | knob hinge q: max(0, 0.5 rad − q) (FlatStove on: q ≥ min of default_turnon_ranges = 0.5) |

Choices behind the table:

- **Distance to the goal set**, not to the target's centre: the distance becomes 0 exactly where LIBERO's geometric test passes. For a movable target (tasks 3, 6), On requires a horizontal distance below 3 cm and the placed object's body at or above the target's. Lifting therefore does not count as progress; horizontal transport, and pushing the target toward the object, do. For region sites, the height band of `under` (up to 0.10 m above the surface) counts too: lifting a bottle up to the cabinet top is progress, and lifting far above the band is regress.
- **Not measured**: the contact condition of On (with the target object, or with the site's fixture). The final lowering onto a target is progress only through the goal check.
- **Task 1**: the goal has one conjunct, In, but the bowl can enter the box only when the drawer is open. The drawer becomes its own component with LIBERO's open condition, a proxy for "open enough". The bowl component is evaluated against the interior where it is once the drawer is open, so it measures the bowl's own motion, not the drawer's.

Labels:

- **Component**: progress when its distance falls by at least the threshold, regress when it rises by at least the threshold, none otherwise. Thresholds: 1 cm for positions and slide joints (the failure classifier's goal_move_m), 0.05 rad for the knob (goal_hinge_rad).
- **Transition**, rules in order:
  1. progress when LIBERO's goal check turns true, regress when it turns false;
  2. otherwise progress when some component progresses and none regresses, regress when some regresses and none progresses;
  3. mixed when both happen, none otherwise.

  "No progress or regress" (NP∪R) below means none or regress; mixed chunks (17) are excluded from it.
- **Preparatory sub-label** of a none chunk, first that applies:
  - grasped: a placed goal object becomes touched by both finger pads;
  - lifted: a placed goal object rises ≥ 1 cm;
  - approached: the end effector gets ≥ 1 cm closer to the nearest placed goal object or goal-joint body;
  - other.
- **Gripper state at the chunk start**: open (opening ≥ 0.06 m), closed holding a placed goal object (both pads touch it), closed not holding.
- **Disturbed**: a non-goal object moved ≥ 2 cm, or a non-goal joint beyond the classifier's other-joint threshold.
- **Episode groups**: demo (TRAIN chunks), success and failed (rollout chunks by episode outcome). Kind: chunk (`rollout`, demo) or stay (`rollout_stay`).
- **Intervals**: shares carry 95% percentile intervals from 2000 bootstrap resamples of episodes (transitions within an episode are dependent).

| Group | Kind | n | progress | none | regress | mixed | goal reached | disturbed |
|---|---|---|---|---|---|---|---|---|
| demo | chunk | 4426 | 1701 | 2679 | 44 | 2 | 376 | 44 |
| success | chunk | 2906 | 972 | 1919 | 15 | 0 | 28 | 32 |
| failed | chunk | 16906 | 979 | 15497 | 415 | 15 | 0 | 565 |
| success | stay | 16 | 2 | 14 | 0 | 0 | 0 | 0 |
| failed | stay | 3374 | 6 | 3357 | 11 | 0 | 0 | 22 |

The rollouts record only complete 10-step chunks. The chunk in which an episode succeeds usually stops early and is not stored, so goal completion appears in only 28 rollout chunks. The store's `goal_after` agrees with the traced goal check for every transition.

## Results

### True match against progress

Counts of executed chunks:

| Group | m(z) > 0: progress / none / regress | m(z) ≤ 0: progress / none / regress | m(z) > 0 overall | P(progress \| m(z) > 0) |
|---|---|---|---|---|
| demo | 1701 / 2674 / 41 | 0 / 5 / 3 | 0.998 | 0.39 |
| success | 941 / 1753 / 13 | 31 / 166 / 2 | 0.932 | 0.35 |
| failed | 882 / 9774 / 213 | 97 / 5723 / 202 | 0.644 | 0.08 |

Chunks, share [95% interval] (count):

| Quantity | demo | success | failed |
|---|---|---|---|
| m(z) > 0 \| NP∪R | 1.00 [0.99, 1.00] (2715/2723) | 0.91 [0.90, 0.93] (1766/1934) | **0.63** [0.62, 0.64] (9987/15912) |
| m(z) > 0 \| regress | 0.93 [0.84, 1.00] (41/44) | 0.87 [0.64, 1.00] (13/15) | **0.51** [0.46, 0.56] (213/415) |
| NP∪R \| m(z) > 0 | 0.61 [0.60, 0.63] (2715/4418) | 0.65 [0.64, 0.67] (1766/2707) | **0.92** [0.91, 0.93] (9987/10881) |
| regress \| m(z) > 0 | 0.009 [0.006, 0.013] (41/4418) | 0.005 [0.002, 0.008] (13/2707) | **0.020** [0.016, 0.023] (213/10881) |
| m(z) ≤ 0 \| progress | 0.00 (0/1701) | 0.03 [0.02, 0.05] (31/972) | 0.10 [0.08, 0.12] (97/979) |
| m(z) > 0 and NP∪R, share of all chunks | 0.61 | 0.61 | 0.59 |

Failures and successes come from different tasks: tasks 0, 1, 6, 7 and 8 have at most 14 successful episodes each. The task-standardized comparison averages per-task shares with the failed group's per-task denominators as weights. It uses the tasks where every group has at least 20 denominator transitions: 0, 1, 2, 3, 4, 5, 8, 9 (tasks 1, 2, 3, 4, 5, 8 for the last row):

| Quantity | failed | success | demo |
|---|---|---|---|
| NP∪R \| m(z) > 0 | 0.89 [0.88, 0.90] | 0.63 [0.61, 0.65] | 0.59 [0.58, 0.60] |
| regress \| m(z) > 0 | 0.02 [0.02, 0.02] | 0.00 [0.00, 0.01] | 0.01 [0.01, 0.01] |
| m(z) > 0 \| NP∪R | 0.61 [0.60, 0.62] | 0.93 [0.91, 0.95] | 1.00 [0.99, 1.00] |
| m(z) ≤ 0 \| progress | 0.09 [0.07, 0.10] | 0.03 [0.01, 0.06] | 0.00 [0.00, 0.00] |

What the no-progress chunks with m(z) > 0 are (share of the row):

| Group | n | grasped | lifted | approached | other | disturbed |
|---|---|---|---|---|---|---|
| demo | 2674 | 0.04 | 0.02 | 0.73 | 0.21 | 0.00 |
| success | 1753 | 0.02 | 0.01 | 0.77 | 0.21 | 0.00 |
| failed | 9774 | 0.01 | 0.01 | 0.43 | 0.56 | 0.02 |

m(z) > 0 among no-progress chunks by sub-label, and among NP∪R chunks by gripper state at the start:

| Group | grasped | lifted | approached | other | gripper open | closed, holding | closed, not holding |
|---|---|---|---|---|---|---|---|
| demo | 1.00 (109) | 1.00 (59) | 1.00 (1949) | 1.00 (562) | 1.00 (2435) | 1.00 (172) | 0.94 (116) |
| success | 0.82 (39) | 0.75 (20) | 0.92 (1458) | 0.90 (402) | 0.93 (1673) | 1.00 (69) | 0.77 (192) |
| failed | 0.74 (74) | 0.70 (91) | 0.75 (5612) | 0.56 (9720) | 0.67 (10306) | 0.67 (165) | 0.55 (5441) |

Distribution of m(z) by label (chunks; median [25%, 75%]):

| Group | progress | none | regress |
|---|---|---|---|
| demo | +0.470 [+0.432, +0.533] | +0.421 [+0.359, +0.467] | +0.448 [+0.405, +0.500] |
| success | +0.477 [+0.447, +0.560] | +0.396 [+0.275, +0.463] | +0.428 [+0.047, +0.540] |
| failed | +0.439 [+0.367, +0.484] | +0.103 [−0.078, +0.428] | +0.009 [−0.315, +0.427] |

In failed episodes, 23% of the NP∪R chunks score at or above the median progress chunk (+0.439).

AUC of m(z) as a progress detector over all tasks (progress vs regress / progress vs NP∪R): success 0.64 / 0.74, failed 0.73 / 0.72, demo 0.62 / 0.71. Against NP∪R by task in failed episodes it ranges from 0.60 (task 6) to 0.85 (task 0); per-task values are in `report/baseline/report.md`.

### Baseline Δ

Chunks, share [95% interval] (count):

| Quantity | demo | success | failed |
|---|---|---|---|
| m(Δ̂) > 0 \| NP∪R | 1.00 (2716/2723) | 0.95 [0.93, 0.96] (1829/1934) | **0.85** [0.84, 0.87] (13589/15912) |
| m(Δ̂) > 0 \| regress | 0.93 (41/44) | 0.87 (13/15) | **0.83** [0.78, 0.88] (346/415) |
| NP∪R \| m(Δ̂) > 0 | 0.61 (2716/4419) | 0.66 (1829/2779) | 0.94 (13589/14532) |
| regress \| m(Δ̂) > 0 | 0.009 (41/4419) | 0.005 (13/2779) | 0.024 (346/14532) |
| m(Δ̂) ≤ 0 \| progress | 0.00 (0/1701) | 0.02 (22/972) | 0.05 (50/979) |
| m(Δ̂) > 0 and NP∪R, share of all chunks | 0.61 | 0.63 | 0.80 |
| mean cos(Δ̂, z) on NP∪R chunks | 0.979 | 0.877 | 0.408 |

The signs of the true and predicted match on NP∪R chunks. z+Δ̂−, for example, counts chunks with m(z) > 0 and m(Δ̂) ≤ 0:

| Group | Subset | n | z+Δ̂+ | z−Δ̂+ | z+Δ̂− | z−Δ̂− | share of Δ̂+ with z+ |
|---|---|---|---|---|---|---|---|
| demo | NP∪R | 2723 | 2715 | 1 | 0 | 7 | 1.00 |
| success | NP∪R | 1934 | 1746 | 83 | 20 | 85 | 0.95 [0.94, 0.97] |
| failed | NP∪R | 15912 | 9057 | 4532 | 930 | 1393 | **0.67** [0.66, 0.68] |
| failed | regress | 415 | 191 | 155 | 22 | 47 | **0.55** [0.49, 0.61] |

In failed episodes, two thirds of the NP∪R chunks that Δ judges valid have a positive true match (0.55 of the regressing ones). The rest have m(z) ≤ 0 and come from Δ's own error: 4532 of 13589 NP∪R chunks and 155 of 346 regressing chunks.

### By task

Failed episodes unless stated; share (count):

| Task | failed chunks | m(z) > 0 \| NP∪R | m(Δ̂) > 0 \| NP∪R | m(z) > 0 \| regress | m(Δ̂) > 0 \| regress | success: m(z) > 0 \| NP∪R | demo: m(z) > 0 \| NP∪R |
|---|---|---|---|---|---|---|---|
| 0 | 2127 | 0.62 (2120) | 0.96 | – (0) | – | 0.92 (141) | 1.00 (391) |
| 1 | 2385 | 0.66 (2045) | 0.88 | 0.50 (125) | 0.81 | 0.95 (57) | 0.99 (358) |
| 2 | 1586 | 0.71 (1263) | 0.90 | 0.51 (70) | 1.00 | 0.99 (310) | 1.00 (274) |
| 3 | 687 | 0.59 (642) | 0.94 | 0.44 (36) | 0.92 | 0.83 (398) | 1.00 (242) |
| 4 | 1343 | 0.57 (1292) | 0.66 | 1.00 (1) | 1.00 | 0.83 (232) | 1.00 (235) |
| 5 | 940 | 0.45 (894) | 0.50 | 0.38 (26) | 0.62 | 0.86 (228) | 0.99 (184) |
| 6 | 2434 | 0.59 (2373) | 0.75 | 0.52 (109) | 0.71 | 1.00 (11) | 1.00 (227) |
| 7 | 2721 | 0.72 (2696) | 0.93 | 0.70 (10) | 1.00 | – (0) | 0.99 (359) |
| 8 | 2635 | 0.61 (2540) | 0.94 | 0.61 (38) | 1.00 | 1.00 (35) | 0.99 (195) |
| 9 | 48 | 0.79 (47) | 0.85 | – (0) | – | 0.98 (522) | 1.00 (258) |

### Stay

The loop scores stay as m = 0. The true effect of a stay chunk has median ‖z‖ 0.51 in failed episodes, and m(z) > 0 in 50% [48%, 52%] of the 3374 stay transitions (median m(z) −0.001). 3357 of them are labeled none.

### Examples

The failed-episode chunks with m(z) > 0 and regress, 213 in total, cluster in tasks 1 (63), 6 (57), 2 (36), 8 (23) and 3 (16). The four below were re-simulated and inspected in `renders/` (top row agent view before | after, bottom row wrist view):

| Task, episode, t | m(z) | m(Δ̂) | loop score | Scene change | Image |
|---|---|---|---|---|---|
| 6, 54, 12 | +0.705 | −0.041 | +0.010 | the gripper closes on the cream cheese without a two-pad grasp and drags it 6 cm away from the bowl (distance to the goal set 0.057 → 0.108 m); the plate is pushed 3.6 cm | `task6_ep54_t12.png` |
| 1, 70, 19 | +0.663 | +0.582 | +0.674 | the closed gripper, not holding the bowl, pushes the half-open top drawer 1.9 cm closed (q −0.104 → −0.085 m) | `task1_ep70_t19.png` |
| 1, 78, 28 | +0.659 | −0.108 | +0.574 | the bowl, pinned against the gripper without a two-pad grasp at the height of the drawer interior, drops 7.5 cm; the drawer is closed (distance to the interior box 0.022 → 0.098 m) | `task1_ep78_t28.png` |
| 2, 9, 21 | +0.385 | +0.524 | +0.543 | the plate is lifted and tilted by the gripper instead of pushed; distance to the stove-front region 0.042 → 0.071 m | `task2_ep9_t21.png` |

"Loop score" is the closed loop's own score of the executed choice (the match of the retrieved demonstration's effect). The full example list is in `report/baseline/report.md`.

### Threshold sensitivity

With 2 cm for positions and slide joints and 0.10 rad for the knob (`report/baseline_d2cm/`):

| Quantity (failed chunks) | 1 cm / 0.05 rad | 2 cm / 0.10 rad |
|---|---|---|
| m(z) > 0 \| NP∪R | 0.63 | 0.63 (10247/16220) |
| m(z) > 0 \| regress | 0.51 (213/415) | 0.50 (95/189) |
| NP∪R \| m(z) > 0 | 0.92 | 0.94 |
| m(Δ̂) > 0 \| NP∪R | 0.85 | 0.86 |
| m(Δ̂) > 0 \| regress | 0.83 | 0.76 (143/189) |
| share of Δ̂+ NP∪R chunks with z+ | 0.67 | 0.67 |
| NP∪R \| m(z) > 0, success / demo | 0.65 / 0.61 | 0.72 / 0.66 |

## Conclusion

A Δ that reproduced the true ψ effect would still judge most failing actions valid, so expanding only Δ's data cannot remove them:

- **Failed episodes**: the true match is positive for 63% of the executed chunks that make no progress or regress, and for 51% of the chunks that move the goal object or joint away from the goal. These are 59% of all failed-episode chunks.
- **Baseline Δ**: it judges 85% and 83% of the same chunks valid. Better Δ data can remove at most the part where Δ̂ is positive and z is not: 28% of NP∪R chunks and 37% of regressing chunks. That lowers the valid-without-progress rate from 85% to about 63%, not below.
- **A positive true match is weak evidence of progress on failure states**: 92% of the failed-episode chunks with m(z) > 0 make no progress or regress, against 65% in successful episodes and 61% in the demonstrations (task-standardized 0.89 vs 0.63 / 0.59). P(progress | m(z) > 0) is 0.08, against 0.35 and 0.39.
- **Regression under a positive match**: 2.0% of positive failed-episode chunks, against 0.5% (success) and 0.9% (demo). The difference is significant, and it holds after task standardization (0.02 vs 0.00 / 0.01), but the share is small. The dominant case is a positive match without progress.
- **Is ψ more permissive on failure states?** No. On demonstration and successful-episode states m(z) is positive for nearly every chunk, including the no-progress ones (1.00 / 0.91), which are mostly approach motions (73% / 77%). On failure states the true match still separates progress from the rest partly: median +0.44 for progress, +0.10 for none, +0.01 for regress, AUC 0.72. It is positive for:
  - 56% of the no-progress chunks with no preparatory effect;
  - 55% of the NP∪R chunks that start with the gripper closed on nothing.

  23% of NP∪R chunks score at or above the median progressing chunk.

## Caveats

- **Only goal-object and goal-joint motion counts as progress.** Approach, grasp, re-grasp and lift are "none" although they are needed. The sub-labels separate them, but whether a given approach is useful is not measured.
- **Regress also captures legitimate detours.** Example: lifting the bowl above the drawer interior before lowering it in. 44 of 4426 demonstration chunks (1.0%) are labeled regress, 41 of them with m(z) > 0. Regress shares of about 1% are therefore within label noise.
- **Contact is not measured**: the contact conditions of On are not in the distances.
- **Task 1 uses proxies**: the drawer component with LIBERO's open threshold, and the bowl target placed where the interior is once the drawer is open.
- **Chunk granularity**: a 10-step chunk hides progress that is undone within the same chunk.
- **The demonstrations are in-sample**: the TRAIN chunks are stage one's and Δ's training data (m(z) > 0 for 99.8%, mean cos(Δ̂, z) 0.98). Successful rollouts are the out-of-sample reference.
- **The actions are not a random sample**: the rollout chunks were selected by the loop from retrieved demonstration effects, so they are not a random sample of actions at failure states.
- **Task mix**: successes concentrate in tasks 2, 3, 4, 5 and 9, failures in 0, 1, 6, 7 and 8. The task-standardized rows compare within tasks but exclude tasks 6 and 7, which have no successful references.

## Rerunning with other dynamics heads

The labels and scenes do not depend on Δ. A retrained Δ (Phase 7) needs only the `effects` and `report` stages, under a new name. Run from the main checkout (default paths resolve under `$CEAM_ROOT`):

```bash
source scripts/ceam_env.sh    # CUDA_VISIBLE_DEVICES unset
$CEAM_PYTHON scripts/analyze_rollout_progress.py effects --name phase7 \
  --dynamics-checkpoints outputs/runs/goal_dynamics_short/checkpoints/step_003000.pt \
                         outputs/runs/goal_dynamics_<name>/checkpoints/step_003000.pt \
  --dynamics-labels baseline <name> --cuda-device <idle GPU> --code-commit <sha>
$CEAM_PYTHON scripts/analyze_rollout_progress.py report --effects phase7 --code-commit <sha>
```

Every Δ must be trained on the same stage-one checkpoint (checked). The report repeats the Δ tables, including the sign table of true and predicted match, for each label. A new rollout store (`--transitions`, `--rollout-run`) also needs the `scenes` stage, per task, under a new `--analysis-dir`:

```bash
$CEAM_PYTHON scripts/analyze_rollout_progress.py --analysis-dir <dir> --rollout-run <run> --transitions <store> \
  scenes --task <t> --cuda-device <idle GPU> --code-commit <sha>
```

## Goal-offset scores on the same chunks (second plan, phases 12 and 14)

Would a state-dependent direction toward the goal-head output, ĝ_l − ψ(o), or the same offset in the query subspace, cos(Uᵀz, Uᵀ(ĝ_l − ψ(o))), tell progress from its absence better than q_l?

**Rollout chunks.** The phase-8 rollout chunks (true effects), re-scored offline with `tmp/analysis/goal_match_rescore.py`:

| Target | P(m > 0 \| no progress or regress), failed | AUC progress vs NP∪R, failed |
|---|---|---|
| q_l | 0.63 | 0.72 |
| ĝ − ψ | 0.69 | 0.68 |
| U(ĝ − ψ) | 0.67 | 0.78 |

**Simulator-check chunks.** The phase-9 simulator-check chunks on HELDOUT demonstration states: demo, π mean, the MATCH choice under the four candidate settings, and the retrieval choice.

- These records hold only scalar scores, so the same chunks were executed again from the same restored states with the same seeds (`scripts/evaluate_proposals.py simulate`, commit `5fd9646`, node-64). The runs are in `outputs/analysis/proposals_goaldir{,_j1,_large,_small}/`.
- The re-execution reproduces the records: same choice and progress label for 516 of 516 chunks per setting, and the q_l match within 5e-4. Script: `tmp/analysis/p14_goaldir.py`.
- Over all 890 non-stay chunks, 599 of them without progress or regressing:

| Target | true effects: P(m > 0 \| NP∪R) | P(progress \| m > 0) | AUC | Δ̂ (+A): P(m > 0 \| NP∪R) | AUC |
|---|---|---|---|---|---|
| q_l | 0.95 | 0.33 | 0.70 | 0.99 | 0.58 |
| ĝ − ψ | 0.96 | 0.31 | 0.60 | 0.98 | 0.59 |
| U(ĝ − ψ) | 0.96 | 0.33 | 0.66 | 1.00 | 0.63 |

Neither goal-offset target lowers the share of positive scores without progress, and neither separates progress better on demonstration states. The approach motions of demonstrations count as "no progress" here, which is why the shares are close to 1. On the phase-8 failure states the subspace version separates slightly better (AUC 0.78 against 0.72), but its positive share without progress is higher (0.67 against 0.63).

## Offline scores of labeled chunks per stage-one checkpoint (phase 18)

`scripts/score_labeled_chunks.py` scores the labeled chunks of phases 8 and 9 by their true effects under any stage-one checkpoint. It needs no simulator and no closed loop. For each checkpoint and chunk, the true effect z = ψ(o′) − ψ(o) is computed with the checkpoint's own trunk. o is the frame before the chunk and o′ the frame after it. The effect is scored three ways:

- query: cos(z, q_l);
- goal: cos(z, ĝ_l − ψ(o)), in the full 256-d state space;
- goal_subspace: cos(Uᵀz, Uᵀ(ĝ_l − ψ(o))), the same offset in the coordinates of the query basis U_l (rank 8).

q_l, ĝ_l and U_l are the checkpoint's direction, goal-head output and basis for the task's original instruction. The stage-one objective uses ĝ only through Uᵀĝ, in GoalLoss (`docs/losses.md`).

### Chunks and their frames

| Set | Group | Frames | Labels |
|---|---|---|---|
| phase8 | failed, success (rollout chunks, 16,906 and 2,906; stays not scored) | `outputs/transitions/rollouts_r0`, joined on (task, episode, t) | `report/baseline/transitions.csv` (kind chunk) |
| phase8 | demo (TRAIN chunks t → t + 10, 4,426) | feature cache rows t and t + 10 | the same file |
| phase9 | demo, mean, retrieval (from the first run), match (every run), pooled | `outputs/transitions/phase9_chunks` | `executed.jsonl` of the runs below |

The phase-9 chunks were executed again with their frames recorded:

- **Command**: `evaluate_proposals.py simulate --transitions-out outputs/transitions/phase9_chunks --transitions-shard <shard>`. It writes o once per sample, then o′ and one transition per executed chunk, in `executed.jsonl` order.
- **Transition fields**:
  - kind: demo, mean, match or retrieval, with the suffix `_stay` for a stay choice;
  - episode: the dataset episode id; t: the frame index;
  - sigma: the chosen π candidate's noise scale for match, else 0.
- **Executed records** gain two fields:
  - `preparatory`: the `GoalProgress` sub-label, from the definitions above;
  - `transition`: the shard-local transition index.
- **Runs**: code `3e23723`, node-64, 264–280 s each. Launchers and logs are in `outputs/analysis/proposals_frames/logs/`.

| Setting | Run (`outputs/analysis/`) | Shard | CUDA |
|---|---|---|---|
| (0.25, 0.5, 1), J 3 | `proposals_frames/simulate` | `j3` | 1 |
| (0.25, 0.5, 1), J 1 | `proposals_frames_j1/simulate` | `j1` | 2 |
| (0.5, 1, 2), J 3 | `proposals_frames_large/simulate` | `large` | 3 |
| (0.1, 0.25, 0.5), J 3 | `proposals_frames_small/simulate` | `small` | 4 |

- **Shard size**: 645 frames (129 o, 516 o′) and 516 transitions.
- **Reproduction** (`logs/compare_recorded.py`, output `compare_recorded.txt`), against the recorded runs `proposals_goal*` (code `cfcfec6`) and `proposals_goaldir*` (code `5fd9646`):
  - In all eight comparisons, 516 of 516 records have the same candidate, stay and progress label.
  - The largest difference in the true q_l match is 5.1e-4: `j3` against `proposals_goaldir`, which ran on the same GPU. `j3` equals `proposals_goal` exactly.
  - Every other comparison differs by at most 9.7e-5. The goal-offset scores differ from `proposals_goaldir` by at most 1.4e-4.
  - Render noise changed no choice or label.
- **Across the four processes**: the demo, mean and retrieval records agree in candidate, progress and preparatory label. Their true match differs by at most 2.0e-4.

Pooled phase-9 chunks: 890 non-stay chunks. 599 of them are NP∪R, 456 of those approach-type.

### Groups, variants and quantities

- **Approach-type chunks**: none chunks whose preparatory sub-label is approached, grasped or lifted (`--approach-sublabels`). They move toward, grasp or lift a goal object, which the labels do not count as progress.
- **Variants** of every group:
  - (a) all labels;
  - (b) approach-type chunks excluded: removed from the group, so NP∪R holds only the none chunks labelled "other" and the regress chunks;
  - (c) the approach-type chunks alone, and the none chunks of each sub-label.
- **Quantities**:
  - P(s > 0 | NP∪R), P(s > 0 | regress), P(progress | s > 0);
  - the AUC of the score for progress vs NP∪R (Mann–Whitney, ties half), and the AUC of −s, which is 1 − AUC;
  - for (c), P(s > 0).

  Mixed chunks are in neither class but count among the chunks with s > 0.
- **Groups**: each group as a whole and per task.
- **Intervals**: 95% percentile intervals from 2000 bootstrap resamples of episodes. The resamples of a group are drawn once, from seed 0, the set and the group, and every checkpoint uses them. Differences between checkpoints are therefore paired.

### Interface and outputs

```bash
source scripts/ceam_env.sh
$CEAM_PYTHON scripts/score_labeled_chunks.py --checkpoints <stage one A> [<stage one B> …] --names A B \
  --out-dir outputs/analysis/labeled_chunk_scores/<name> --device cuda:<idle GPU> --code-commit <sha>
```

- **Options**:
  - `--device cpu` runs on the CPU, with `--cpu-threads` setting the thread count;
  - the inputs default to the files above (`--phase8-labels`, `--phase8-transitions`, `--phase9-runs`, `--phase9-transitions`);
  - the first `--phase9-runs` entry supplies the demo, mean and retrieval chunks.
- **Requirement**: every checkpoint must use the same feature cache, dataset, split fraction and seed (checked).
- **Output files**: an existing output directory is refused.
  - `config.json`: arguments, host, device, checkpoint steps, chunk counts, the convention checks, the consistency with the recorded scores, and the goal-head geometry.
  - `chunks.csv`: one row per chunk, with labels, sub-label, approach flag, the recorded scores, and each checkpoint's scores and ‖z‖.
  - `tables.csv`: every estimate (value, interval, numerator, denominator) by checkpoint, score, set, group, variant, task and quantity.
  - `comparisons.csv`: each later checkpoint minus the first, with paired intervals.
  - `score_comparisons.csv`: each score minus the query score of the same checkpoint, with paired intervals.
  - `report.md`, in this order: inputs, conventions and consistency, the side-by-side summary per score, the goal-head geometry, the paired comparisons, then per checkpoint the tables (a), (b) and (c) and phase8 failed by task.
- **Numerics**: the trunk runs in float32 without autocast, TF32 off, 256 frames per call. Deterministic algorithms are on, with the cuBLAS workspace fixed at `:4096:8`.
- **Cost**: on an L40S, 47 s for three checkpoints, each 25,128 chunks and 28,416 frames of ψ. One checkpoint takes 4.6 min on 16 CPU threads.

### Checks

Each run reports these checks in `report.md` under Conventions.

- **Frame order, z = ψ(o′) − ψ(o)**:
  - Phase-8 rollout store: current frame < future frame in 23,202 of 23,202 transitions. In 22,202 of 22,202 consecutive stored decisions t, t + 1 of one episode, the o′ frame of t is the o frame of t + 1.
  - Phase-9 store: in 516 of 516 samples, every executed chunk starts from one o frame, written before each o′.
  - Phase-8 demo chunks: o′ is o's feature-cache row + 10 in 4,426 of 4,426.
- **Progress means a goal distance fell**:
  - 3,652 of 3,652 phase-8 progress chunks have a goal component whose distance to its goal set decreased, or end with the goal holding: 3,459 with a decreased distance, 404 with the goal holding.
  - 474 of 474 regress chunks have a component whose distance increased.
- **Consistency with the recorded scores** under the fixed stage one, which recorded them:
  - m(z) of `transitions.csv`: largest difference 1.2e-6;
  - phase-9 `match_true`: 2.8e-6;
  - `match_true_goal`, the full-space offset from ψ(o) before the chunk: 3.1e-6;
  - `match_true_goal_subspace` (`evaluate_proposals.TaskGoals.matches`): 1.4e-5.
- **Determinism**: two runs on different GPUs of node-64 give byte-identical `chunks.csv` and `tables.csv`. So do the runs of the same checkpoints through different `ceam` packages below.
- **CPU against GPU**: the CPU run differs by up to 1.8e-3 in a score. It changes the sign of 1 query and 2 goal scores out of 25,128. Whole-group estimates agree within 1.6e-3, the largest being a phase-9 demo AUC. Per-task estimates on a few chunks differ by up to 0.11, an AUC over 9 chunks.
- **Known numbers reproduced**:
  - phase-8 failed chunks: 0.63 / AUC 0.72 for q_l, 0.69 / 0.68 for ĝ − ψ, 0.67 / 0.78 for Uᵀ(ĝ − ψ);
  - phase-9 pooled chunks: 0.95 / 0.70, 0.96 / 0.60 and 0.96 / 0.66.

### Baseline: fixed stage one

Runs:

- `outputs/analysis/labeled_chunk_scores/baseline`: q_l and ĝ − ψ, code `c07c877`, node-64 CUDA 5;
- `three_scores` (next section): the U-coordinate rows.

The table gives P(s > 0 | NP∪R) and the AUC under (a) all labels and (b) approach-type chunks excluded, P(progress | s > 0) under (a) and (b), and P(s > 0) of the approach-type chunks (c):

| Score | Set | Group | P(s>0 \| NP∪R) (a) | (b) | AUC (a) | (b) | P(progress \| s>0) (a) | (b) | approach-type s > 0 (c) |
|---|---|---|---|---|---|---|---|---|---|
| q_l | phase8 | failed | 0.63 (9987/15912) | 0.56 (5672/10135) | 0.72 | 0.76 | 0.08 | 0.13 | 0.75 (4315/5777) |
| q_l | phase8 | success | 0.91 (1766/1934) | 0.89 (373/417) | 0.74 | 0.65 | 0.35 | 0.72 | 0.92 (1393/1517) |
| q_l | phase8 | demo | 1.00 (2715/2723) | 0.99 (601/606) | 0.71 | 0.67 | 0.39 | 0.74 | 1.00 (2114/2117) |
| q_l | phase9 | demo | 0.93 (78/84) | 0.93 (13/14) | 0.73 | 0.60 | 0.35 | 0.76 | 0.93 (65/70) |
| q_l | phase9 | mean | 0.95 (81/85) | 1.00 (17/17) | 0.72 | 0.64 | 0.33 | 0.70 | 0.94 (64/68) |
| q_l | phase9 | match | 0.96 (331/346) | 0.89 (84/94) | 0.68 | 0.66 | 0.32 | 0.65 | 0.98 (247/252) |
| q_l | phase9 | retrieval | 0.95 (80/84) | 0.83 (15/18) | 0.75 | 0.66 | 0.33 | 0.73 | 0.98 (65/66) |
| q_l | phase9 | pooled | 0.95 (570/599) | 0.90 (129/143) | 0.70 | 0.65 | 0.33 | 0.69 | 0.97 (441/456) |
| ĝ − ψ | phase8 | failed | 0.69 (10928/15912) | 0.61 (6221/10135) | 0.68 | 0.73 | 0.07 | 0.12 | 0.81 (4707/5777) |
| ĝ − ψ | phase8 | success | 0.89 (1720/1934) | 0.69 (286/417) | 0.57 | 0.68 | 0.32 | 0.74 | 0.95 (1434/1517) |
| ĝ − ψ | phase8 | demo | 0.94 (2551/2723) | 0.78 (471/606) | 0.40 | 0.53 | 0.33 | 0.73 | 0.98 (2080/2117) |
| ĝ − ψ | phase9 | demo | 0.95 (80/84) | 0.93 (13/14) | 0.64 | 0.64 | 0.33 | 0.75 | 0.96 (67/70) |
| ĝ − ψ | phase9 | mean | 0.99 (84/85) | 0.94 (16/17) | 0.59 | 0.63 | 0.31 | 0.70 | 1.00 (68/68) |
| ĝ − ψ | phase9 | match | 0.96 (333/346) | 0.88 (83/94) | 0.59 | 0.61 | 0.30 | 0.63 | 0.99 (250/252) |
| ĝ − ψ | phase9 | retrieval | 0.93 (78/84) | 0.83 (15/18) | 0.60 | 0.53 | 0.32 | 0.71 | 0.95 (63/66) |
| ĝ − ψ | phase9 | pooled | 0.96 (575/599) | 0.89 (127/143) | 0.60 | 0.61 | 0.31 | 0.67 | 0.98 (448/456) |
| Uᵀ(ĝ − ψ) | phase8 | failed | 0.67 (10591/15912) | 0.59 (5966/10135) | 0.78 | 0.83 | 0.08 | 0.13 | 0.80 (4625/5777) |
| Uᵀ(ĝ − ψ) | phase8 | success | 0.95 (1837/1934) | 0.85 (356/417) | 0.72 | 0.77 | 0.34 | 0.73 | 0.98 (1481/1517) |
| Uᵀ(ĝ − ψ) | phase8 | demo | 0.99 (2702/2723) | 0.98 (593/606) | 0.63 | 0.67 | 0.39 | 0.74 | 1.00 (2109/2117) |
| Uᵀ(ĝ − ψ) | phase9 | demo | 0.98 (82/84) | 0.93 (13/14) | 0.65 | 0.65 | 0.34 | 0.76 | 0.99 (69/70) |
| Uᵀ(ĝ − ψ) | phase9 | mean | 0.99 (84/85) | 1.00 (17/17) | 0.62 | 0.60 | 0.32 | 0.70 | 0.99 (67/68) |
| Uᵀ(ĝ − ψ) | phase9 | match | 0.95 (329/346) | 0.85 (80/94) | 0.67 | 0.71 | 0.32 | 0.66 | 0.99 (249/252) |
| Uᵀ(ĝ − ψ) | phase9 | retrieval | 0.96 (81/84) | 0.89 (16/18) | 0.70 | 0.73 | 0.33 | 0.71 | 0.98 (65/66) |
| Uᵀ(ĝ − ψ) | phase9 | pooled | 0.96 (576/599) | 0.88 (126/143) | 0.66 | 0.69 | 0.33 | 0.69 | 0.99 (450/456) |

Intervals, for q_l:

- failed (a) 0.63 [0.62, 0.64], (b) 0.56 [0.55, 0.57];
- phase-9 pooled (a) 0.95 [0.92, 0.98], (b) 0.90 [0.84, 0.95];
- AUC: failed (b) 0.76 [0.75, 0.78], pooled (b) 0.65 [0.54, 0.75].

All intervals are in `report.md`.

- **Approach motions account for most of the positive scores without progress on demonstration states**:
  - 456 of the 599 phase-9 NP∪R chunks are approach-type, and 97% of them score positive under q_l.
  - Without them, the share of positive NP∪R chunks falls from 0.95 to 0.90 (q_l) and from 0.96 to 0.89 (ĝ − ψ). It stays near 0.9 on the 143 remaining chunks: 126 "other" none chunks and 17 regress.
  - P(progress | s > 0) rises from 0.33 to 0.69.
- **On failure states the shares fall less**:
  - Excluding approach motions lowers the positive share from 0.63 to 0.56 (q_l) and from 0.69 to 0.61 (ĝ − ψ).
  - The AUC rises from 0.72 to 0.76 and from 0.68 to 0.73.
  - P(progress | s > 0) is 0.13 against 0.72 (success) and 0.74 (demo): a positive score remains weak evidence of progress on failure states.
- **Approach motions themselves score positive** in 75% (q_l), 81% (ĝ − ψ) and 80% (Uᵀ(ĝ − ψ)) of failed-episode chunks. On success, demonstration and phase-9 states the share is 0.92–1.00.
- **ĝ − ψ on the TRAIN demonstration chunks** ranks progress below no progress: AUC 0.40, 0.53 without approach motions. Under q_l the AUC is 0.71 and 0.67; in Uᵀ coordinates 0.63 and 0.67.
- **The goal offset in Uᵀ coordinates** separates progress best on failure states: AUC 0.78, and 0.83 without approach motions, against 0.72 / 0.76 for q_l. Its positive share among NP∪R chunks is 0.67 / 0.59.
- **Per task**, failed episodes (`report.md`): excluding approach motions lowers the q_l share most in task 0, open the middle drawer, from 0.62 to 0.41. There the approach-type chunks score positive in 93% and the AUC rises from 0.85 to 0.93. Tasks 1, 3 and 5 hardly change: 0.66 → 0.66, 0.59 → 0.59 and 0.45 → 0.44.

### Flow-target stage ones (phase 17)

Run: `outputs/analysis/labeled_chunk_scores/three_scores`, node-64 CUDA 5, 47 s. All checkpoints are at step 5000:

| Name | Checkpoint (`outputs/runs/`) | Path target |
|---|---|---|
| fixed | `goal_instruction_contrast` | perpendicular |
| flow | `goal_instruction_contrast_flow` | interpolation in Uᵀ coordinates |
| flow_full | `goal_instruction_contrast_flow_full` | interpolation in the full space |

- **Code**: the scorer at `338dccd`, with the `ceam` package of `s3-phase-17` `ad9ab4e` (`git archive`), which reads `path_target`. The model, feature-cache, transition-store and checkpoint-store code of that package are identical to this branch.
- **Consistency**: the fixed and flow rows of the q_l and ĝ − ψ scores equal, in every estimate, the earlier runs:
  - `flow_vs_fixed`: `ceam` package `a17b920`;
  - `flow_variants_vs_fixed`: without the third score.

The acceptance quantities are P(s > 0 | NP∪R) and the AUC, under (a) all labels and (b) approach-type chunks excluded, plus P(s > 0) of the approach-type chunks (c). The AUC of −s is 1 − AUC. Phase 9 by chunk kind, the intervals and every other quantity are in `report.md`.

| Score | Group | Variant | P(s>0) fixed | flow | flow_full | AUC fixed | flow | flow_full |
|---|---|---|---|---|---|---|---|---|
| q_l | phase8 failed | (a) | 0.63 | 0.63 | 0.61 | 0.72 | 0.72 | 0.76 |
| q_l | phase8 failed | (b) | 0.56 | 0.56 | 0.53 | 0.76 | 0.75 | 0.80 |
| q_l | phase8 failed | (c) | 0.75 | 0.74 | 0.73 | | | |
| q_l | phase8 success | (a) | 0.91 | 0.89 | 0.93 | 0.74 | 0.71 | 0.76 |
| q_l | phase8 success | (b) | 0.89 | 0.86 | 0.88 | 0.65 | 0.63 | 0.68 |
| q_l | phase8 success | (c) | 0.92 | 0.90 | 0.95 | | | |
| q_l | phase8 demo | (a) | 1.00 | 1.00 | 0.95 | 0.71 | 0.72 | 0.67 |
| q_l | phase8 demo | (b) | 0.99 | 0.99 | 0.93 | 0.67 | 0.68 | 0.62 |
| q_l | phase8 demo | (c) | 1.00 | 1.00 | 0.96 | | | |
| q_l | phase9 pooled | (a) | 0.95 | 0.94 | 0.97 | 0.70 | 0.67 | 0.68 |
| q_l | phase9 pooled | (b) | 0.90 | 0.85 | 0.90 | 0.65 | 0.64 | 0.70 |
| q_l | phase9 pooled | (c) | 0.97 | 0.97 | 0.99 | | | |
| ĝ − ψ | phase8 failed | (a) | 0.69 | 0.72 | 0.59 | 0.68 | 0.41 | 0.53 |
| ĝ − ψ | phase8 failed | (b) | 0.61 | 0.66 | 0.59 | 0.73 | 0.45 | 0.53 |
| ĝ − ψ | phase8 failed | (c) | 0.81 | 0.84 | 0.59 | | | |
| ĝ − ψ | phase8 success | (a) | 0.89 | 0.82 | 0.64 | 0.57 | 0.17 | 0.43 |
| ĝ − ψ | phase8 success | (b) | 0.69 | 0.50 | 0.64 | 0.68 | 0.41 | 0.52 |
| ĝ − ψ | phase8 success | (c) | 0.95 | 0.91 | 0.64 | | | |
| ĝ − ψ | phase8 demo | (a) | 0.94 | 0.83 | 0.65 | 0.40 | 0.13 | 0.51 |
| ĝ − ψ | phase8 demo | (b) | 0.78 | 0.57 | 0.67 | 0.53 | 0.29 | 0.54 |
| ĝ − ψ | phase8 demo | (c) | 0.98 | 0.91 | 0.64 | | | |
| ĝ − ψ | phase9 pooled | (a) | 0.96 | 0.88 | 0.54 | 0.60 | 0.12 | 0.57 |
| ĝ − ψ | phase9 pooled | (b) | 0.89 | 0.69 | 0.57 | 0.61 | 0.25 | 0.56 |
| ĝ − ψ | phase9 pooled | (c) | 0.98 | 0.94 | 0.53 | | | |
| Uᵀ(ĝ − ψ) | phase8 failed | (a) | 0.67 | 0.62 | 0.57 | 0.78 | 0.80 | 0.43 |
| Uᵀ(ĝ − ψ) | phase8 failed | (b) | 0.59 | 0.56 | 0.52 | 0.83 | 0.84 | 0.46 |
| Uᵀ(ĝ − ψ) | phase8 failed | (c) | 0.80 | 0.73 | 0.67 | | | |
| Uᵀ(ĝ − ψ) | phase8 success | (a) | 0.95 | 0.90 | 0.61 | 0.72 | 0.70 | 0.37 |
| Uᵀ(ĝ − ψ) | phase8 success | (b) | 0.85 | 0.88 | 0.49 | 0.77 | 0.71 | 0.44 |
| Uᵀ(ĝ − ψ) | phase8 success | (c) | 0.98 | 0.90 | 0.65 | | | |
| Uᵀ(ĝ − ψ) | phase8 demo | (a) | 0.99 | 0.99 | 0.67 | 0.63 | 0.70 | 0.42 |
| Uᵀ(ĝ − ψ) | phase8 demo | (b) | 0.98 | 0.97 | 0.58 | 0.67 | 0.67 | 0.46 |
| Uᵀ(ĝ − ψ) | phase8 demo | (c) | 1.00 | 0.99 | 0.70 | | | |
| Uᵀ(ĝ − ψ) | phase9 pooled | (a) | 0.96 | 0.97 | 0.60 | 0.66 | 0.69 | 0.46 |
| Uᵀ(ĝ − ψ) | phase9 pooled | (b) | 0.88 | 0.89 | 0.52 | 0.69 | 0.69 | 0.48 |
| Uᵀ(ĝ − ψ) | phase9 pooled | (c) | 0.99 | 0.99 | 0.62 | | | |

Paired differences against fixed are given as [95% episode-bootstrap interval]. All of them are in `comparisons.csv` and `report.md`.

**q_l, flow**: no change beyond the resampling noise.

- Failed episodes: P(s > 0 | NP∪R) −0.00 [−0.01, +0.01], AUC +0.00 [−0.01, +0.01].
- With approach motions excluded: AUC −0.01 [−0.02, −0.00].
- Phase-9 pooled: P(s > 0 | NP∪R) −0.01 [−0.03, +0.02], AUC −0.03 [−0.08, +0.03].

**q_l, flow_full**: better on failure states, worse on the TRAIN demonstrations.

- Failed episodes:
  - P(s > 0 | NP∪R) −0.02 [−0.03, −0.02]; with approach motions excluded −0.03 [−0.04, −0.02].
  - AUC +0.04 [+0.03, +0.05], and +0.04 [+0.03, +0.05] with approach motions excluded.
- TRAIN chunks: P(s > 0 | NP∪R) −0.05 [−0.05, −0.04], AUC −0.03 [−0.05, −0.01].
- Phase-9 pooled: inside the noise (AUC −0.03 [−0.12, +0.06]).

**ĝ − ψ, full space**:

- flow ranks progress below no progress in every group: AUC 0.41 (failed), 0.17 (success), 0.13 (TRAIN), 0.12 (phase-9 pooled). AUC(−s) is correspondingly 0.59–0.88.
- flow_full brings every AUC to 0.43–0.57.

**Uᵀ(ĝ − ψ)**:

- flow keeps the fixed stage one's discrimination:
  - failed AUC 0.80 against 0.78, a difference of +0.02 [+0.01, +0.03];
  - phase-9 pooled 0.69 against 0.66, +0.03 [−0.03, +0.09].
- flow_full inverts it: AUC 0.43 (failed, −0.35 [−0.38, −0.32]), 0.37 (success), 0.42 (TRAIN), 0.46 (phase-9 pooled, −0.21 [−0.30, −0.11]).

**Goal-head geometry** (`GoalHeadGeometry`), on the 4,426 phase-8 TRAIN chunks and their 390 episodes:

- ψ_t and z are the chunk's start state and true effect.
- ψ_0 and ψ_T are its episode's first and last frames.
- "U" marks Uᵀ coordinates.

| Quantity | fixed | flow | flow_full |
|---|---|---|---|
| mean cos(ψ_T − ψ_t, ĝ − ψ_t) | +0.59 | +0.12 | +0.79 |
| the same in Uᵀ coordinates | +0.99 | +0.99 | +0.99 |
| mean cos(z, ψ_T − ψ_t) | +0.68 | +0.67 | +0.43 |
| AUC of cos(z, ψ_T − ψ_t), progress vs NP∪R | 0.86 | 0.88 | 0.58 |
| median ‖ĝ − ψ_T‖ / ‖ψ_T − ψ_0‖ | 0.517 | 0.837 | 0.322 |
| median ‖Uᵀ(ĝ − ψ_T)‖ / ‖Uᵀ(ψ_T − ψ_0)‖ | 0.007 | 0.008 | 0.006 |
| median ‖ψ_T − ψ_0‖ | 37.2 | 34.1 | 1169.6 |
| ‖Uᵀĝ‖ / ‖ĝ‖ (share of ‖ĝ‖ in span U), mean [min, max] over tasks | 0.62 [0.49, 0.73] | 0.52 [0.32, 0.74] | 0.56 [0.11, 0.86] |
| median ‖Uᵀ(ĝ − ψ_T)‖ / ‖ĝ − ψ_T‖ | 0.007 | 0.002 | 0.003 |

### Where the inversions come from

- **No implementation error was found**:
  - The frame-order and label checks above hold for every chunk source.
  - All three checkpoints use the same code path. For the fixed stage one, the scores equal the independent recorded implementations, including the U-coordinate score of `evaluate_proposals.py`, to 1.4e-5 or better.
  - Under the same code, flow scores q_l and Uᵀ(ĝ − ψ) as well as the fixed stage one does.
  - Only the full-space score inverts for flow, and only the U-coordinate score for flow_full.
- **flow, full space**: the inversion is a property of ĝ's component outside span(U).
  - In Uᵀ coordinates, ĝ sits at the demonstrations' last states under every checkpoint: 0.6–0.8% of the start-to-end distance, cosine +0.99 with the direction to the end state.
  - The difference ĝ − ψ_T lies more than 99% outside span(U) (median in-span share 0.002–0.007).
  - No loss constrains that part, because the objective sees ĝ only as Uᵀĝ.
  - Under flow, the full-space offset ĝ − ψ_t is nearly orthogonal to the direction to the end state: mean cosine +0.12, against +0.59 (fixed) and +0.79 (flow_full).
  - The episode's own end direction still separates progress well (AUC 0.88). So ψ orders the chunks correctly, and the full-space offset does not.
- **flow_full, U coordinates**: the inversion is a property of the trunk.
  - ĝ again sits at the end states in Uᵀ coordinates.
  - The chunks' true effects point less toward their episode's end state: mean cosine +0.43, against +0.68 / +0.67.
  - That direction separates progress only at AUC 0.58, against 0.86 / 0.88.
  - ψ's scale is about 30 times larger: median ‖ψ_T − ψ_0‖ 1170, against 37 / 34.
  - The effects of chunks that move the scene toward the goal are not the ones that move ψ toward the end state.

## Instance goal and observation-conditioned query (phase 19, branch C evidence)

Reference evidence on the progress proxy of this page (goal-object and goal-joint distances). The framework has no progress requirement, and these numbers drive no decision.

- **Run**: `outputs/analysis/labeled_chunk_scores/branch_c_evidence`, scorer at `f8e6a0b`, node-64 CUDA 5, 72 s. The launcher, log and reproduction check are `logs/run_branch_c_evidence.sh`, `logs/branch_c_evidence.log` and `logs/branch_c_reproduction_check.{py,txt}`.
- **Instance goal (oracle)**: cos(z, ψ_T − ψ(o)). ψ_T is ψ of the last feature-cache frame of the chunk's own demonstration: TRAIN for the phase-8 demo chunks, HELDOUT for phase 9. It uses privileged information and does not exist for rollout chunks.
- **Task-mean goal**: cos(z, ψ̄_T − ψ(o)), with ψ̄_T the mean ψ_T of the task's TRAIN demonstrations. This is a task-level end state, not the instance's own, and it is defined for every chunk.
- **obs**: `goal_instruction_contrast_obs` step 5000 (TEXT_AND_OBSERVATION), queried with the references its training used:
  - q(o, l) on the chunk's frame o, as the instruction contrast queried the direction with the effect's first frame;
  - ĝ and U on the episode's first frame o_0, as the path, goal and endpoint losses did (query_a with frame 0). o_0 is the decision-1 frame of a rollout and frame 0 of a demonstration.
  - ĝ and U queried on o instead, a reference training never used, separate progress worse in every group (`report.md`).
- **Checks**:
  - **Reproduction**: the fixed stage one's 3,960 q_l, ĝ − ψ and U-coordinate estimates, its 25,128 chunk scores and its geometry equal `three_scores` exactly.
  - **Frame source**: phase-9 o is a simulator frame and ψ_T a dataset frame. For the same state, ψ of the simulator frame lies 1–2% of ‖ψ_T − ψ(o)‖ from ψ of the dataset frame (direction cosine 0.9998).

(a) all labels, (b) approach-type chunks excluded, (c) the approach-type chunks alone:

| Score | Group | P(s>0 \| NP∪R) (a) | (b) | P(s>0 \| regress) | P(progress \| s>0) (a) | (b) | AUC (a) | (b) | approach-type s>0 (c) |
|---|---|---|---|---|---|---|---|---|---|
| fixed q_l | phase8 failed | 0.63 | 0.56 | 0.51 | 0.08 | 0.13 | 0.72 | 0.76 | 0.75 |
| fixed q_l | phase8 demo | 1.00 | 0.99 | 0.93 | 0.39 | 0.74 | 0.71 | 0.67 | 1.00 |
| fixed q_l | phase9 pooled | 0.95 | 0.90 | 0.76 | 0.33 | 0.69 | 0.70 | 0.65 | 0.97 |
| instance ψ_T − ψ (oracle), fixed ψ | phase8 failed | n/a | | | | | | | |
| instance ψ_T − ψ (oracle), fixed ψ | phase8 demo | 0.95 | 0.93 | 0.95 | 0.39 | 0.75 | **0.86** | 0.76 | 0.96 |
| instance ψ_T − ψ (oracle), fixed ψ | phase9 pooled | 0.91 | 0.92 | 0.94 | 0.33 | 0.67 | **0.85** | 0.79 | 0.91 |
| task-mean ψ̄_T − ψ, fixed ψ | phase8 failed | 0.66 | 0.60 | 0.55 | 0.08 | 0.13 | 0.76 | 0.78 | 0.79 |
| task-mean ψ̄_T − ψ, fixed ψ | phase8 demo | 0.95 | 0.93 | 0.95 | 0.39 | 0.75 | 0.85 | 0.75 | 0.96 |
| task-mean ψ̄_T − ψ, fixed ψ | phase9 pooled | 0.91 | 0.89 | 0.94 | 0.33 | 0.68 | 0.85 | 0.79 | 0.92 |
| obs q(o, l) | phase8 failed | 0.64 | 0.61 | 0.53 | 0.08 | 0.13 | **0.82** | 0.85 | 0.70 |
| obs q(o, l) | phase8 demo | 1.00 | 1.00 | 1.00 | 0.38 | 0.74 | 0.71 | 0.63 | 1.00 |
| obs q(o, l) | phase9 pooled | 0.88 | 0.97 | 0.94 | 0.35 | 0.68 | 0.69 | 0.62 | 0.86 |
| obs ĝ(o_0, l) − ψ | phase8 failed | 0.76 | 0.71 | 0.81 | 0.07 | 0.11 | 0.67 | 0.70 | 0.85 |
| obs ĝ(o_0, l) − ψ | phase8 demo | 0.84 | 0.77 | 0.93 | 0.40 | 0.76 | 0.60 | 0.64 | 0.86 |
| obs ĝ(o_0, l) − ψ | phase9 pooled | 0.81 | 0.80 | 0.94 | 0.36 | 0.70 | 0.73 | 0.69 | 0.81 |
| obs U(o_0)ᵀ(ĝ(o_0, l) − ψ) | phase8 failed | 0.61 | 0.55 | 0.58 | 0.06 | 0.11 | 0.55 | 0.61 | 0.70 |
| obs U(o_0)ᵀ(ĝ(o_0, l) − ψ) | phase8 demo | 0.76 | 0.74 | 0.84 | 0.42 | 0.77 | 0.43 | 0.55 | 0.77 |
| obs U(o_0)ᵀ(ĝ(o_0, l) − ψ) | phase9 pooled | 0.80 | 0.92 | 0.94 | 0.36 | 0.67 | 0.37 | 0.42 | 0.76 |

The fixed stage one's ĝ − ψ and U-coordinate rows are in the phase-18 table above. The success group, the scores of obs's own ψ and every interval are in `report.md`.

Paired differences, as [95% episode-bootstrap interval]:

- **Instance goal − q_l** (fixed): the AUC rises by +0.15 [+0.14, +0.16] on TRAIN chunks and by +0.15 [+0.10, +0.21] on phase 9. P(s>0 | NP∪R) changes by −0.05 [−0.05, −0.04] and −0.04 [−0.08, +0.01].
- **Task-mean goal − q_l** (fixed): the AUC rises by +0.04 [+0.03, +0.05] on failed chunks and by +0.15 on TRAIN chunks and phase 9. P(s>0 | NP∪R) on failed chunks rises by +0.04 [+0.03, +0.05].
- **obs − fixed, query**:
  - AUC: +0.10 [+0.08, +0.11] on failed chunks, −0.05 [−0.08, −0.02] on success chunks, −0.00 [−0.02, +0.02] on TRAIN chunks and −0.01 [−0.09, +0.07] on phase 9.
  - P(s>0 | NP∪R): +0.02 [+0.01, +0.03] on failed chunks.
- **obs − fixed, ĝ − ψ**: the AUC rises by +0.20 [+0.19, +0.22] on TRAIN chunks and by +0.12 [+0.06, +0.19] on phase 9, and changes by −0.01 [−0.03, +0.01] on failed chunks.
- **obs − fixed, U coordinates**: the AUC falls by 0.20–0.29 in every group, to 0.37–0.55, at or below chance.

Reading:

- **Ranking**: a direction from ψ(o) to the end state ranks the proxy's progress chunks above the others better than q_l does, with AUC 0.85–0.86 against 0.70–0.71 on demonstration states. The task-mean end state matches the instance's own within 0.01, so the gain does not need instance-specific information.
- **Sign**: no direction, including the oracle, makes a positive score a test of progress.
  - Under the oracle, 0.95 (TRAIN) and 0.91 (phase 9) of the NP∪R chunks score positive, and 0.93 and 0.92 without approach motions. Every other direction scores at least 0.74 of them positive on demonstration states.
  - Under every direction, P(progress | s > 0) is at most 0.42 on TRAIN chunks (base rate 0.38) and at most 0.36 on phase 9 (base rate 0.33).
  - The chunks without proxy progress move ψ toward the demonstration's end state as well.
- **Observation-conditioned query**: q(o, l) ranks failure-state chunks better (AUC 0.82 against 0.72) but leaves their positive share unchanged (0.64 against 0.63). On demonstration states its AUC is unchanged (0.71, 0.69). Its positive share falls on phase 9 (0.88 against 0.95) but not on TRAIN chunks.

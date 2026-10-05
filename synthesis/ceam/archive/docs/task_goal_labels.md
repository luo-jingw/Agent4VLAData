# Task-goal labels of stored states, chunk labels, coverage and the branch estimate (phase 20)

Every stored simulator state gets a 10-bit goal vector: for each LIBERO-Goal task, LIBERO's success check of that task's goal on the state. Chunks (o_t, a_t) of every data source then get a positive / negative / undetermined label per task (`plan_next.md`, section 1). This page records the labeller, its checks, the labels' coverage over the existing data, and a pilot of single-chunk branches from demonstration states that estimates how many branches the query / match training of `plan_next.md` (scheme A) needs.

**Verdict.**

- With single-chunk branches, a state yields a group with both a positive and a negative for task l almost only when l already holds at the state. The negatives there are breaks of the goal, not conflicts.
- Uniform sampling of demonstration states is inefficient: p_l(32) ≤ 0.02 per state, and no group at all for tasks 1, 2, 6, 9 in 100 states.
- Sampling the states where the goal holds gives p_l(32) of 0.49–1.00 for tasks 0–8 and 0.10 for task 9 (turn on the stove).
- The query / match observations in which a goal holds are too few for 200 groups except for task 7. Adding every TRAIN frame where a goal holds (the last 10 frames of each demonstration included) should reach 200 groups for 8 of 10 tasks, but not for tasks 1 and 9.

## Code

| Class | File | Responsibility |
|---|---|---|
| `PhysicsTaskEnv`, `PhysicsRollout` | `ceam/sim/physics_task_env.py` | one LIBERO task in a `ControlEnv` without renderer and cameras (no EGL device, no GL context): load a stored state, restore a `SimSnapshot`, execute chunks, snapshot |
| `RobosuiteRuntime.prepare_physics_only` | `ceam/sim/robosuite_runtime.py` | process settings for physics without rendering: requires `MUJOCO_GL=disable`, registers robosuite's macros with no file log and `MUJOCO_GPU_RENDERING = False` |
| `TaskGoal`, `TaskGoalLabeller` | `ceam/sim/task_goal_labeller.py` | every task's goal on the current state of one env |
| `TaskGoalConflicts`, `GoalPlacement` | `ceam/sim/task_goal_conflicts.py` | conflicting goal pairs derived from the predicate structure |
| `ChunkRecords`, `ChunkTaskLabeller`, `ChunkTaskLabels`, `ChunkLabel` | `ceam/data/chunk_task_labels.py` | chunk labels from the goal vectors of recorded states |
| `BranchCandidates`, `BranchCandidateConfig`, `BranchCandidateType` | `ceam/data/branch_candidates.py` | single-chunk candidates at a query / match observation |
| `ChunkBranchExecutor`, `BranchOutcomes` | `ceam/sim/chunk_branch_executor.py` | executes candidates from a stored state without rendering and labels their end states |

Scripts:

- `scripts/label_task_goals.py`, with subcommands `check`, `label` and `branch-labels` (simulator, CPU only);
- `scripts/report_task_goal_labels.py` (no simulator): chunk labels, coverage, groups, pilot rates and the estimate.

`LiberoTaskEnv`, `DemoStateRestorer` and every existing code path are unchanged; `RobosuiteRuntime.prepare` is unchanged.

## Goal vectors

- **Goals.** Task l's goal is the conjunction of its BDDL goal predicates (`data/task_relations/libero_goal.json`, dataset `task_index` order through `EpisodeIndex.task_texts`).
- **Evaluation.** `TaskGoalLabeller.label(sim_state, fixture_poses)`:
  1. loads the stored flattened MuJoCo state (79 values: time, qpos, qvel) and the fixture poses into a `PhysicsTaskEnv` of the state's own task: fixture poses written into the model, `ControlEnv.set_state`, `mj_forward`;
  2. evaluates every task's predicates with LIBERO's `eval_predicate_fn` on the env's `object_states_dict`. This is the computation of each task's own `_check_success`.
- **One scene.** The 10 LIBERO-Goal BDDL files declare the same regions, fixtures, objects and initial conditions and differ only in language, objects of interest and goal. Every goal is therefore evaluable in every task's env; the cross-task check below measures that the result does not depend on the env.
- **Label and live check.** The label is a function of the stored state alone. The sidecars' `goal_holds` is LIBERO's check in the live env after the frame's step, which reads body and site poses from the start of the step's last physics substep (`docs/demo_renders.md`). The two can differ; the next section counts the differences.

| Goal | Task |
|---|---|
| (Open wooden_cabinet_1_middle_region) | 0 open the middle drawer of the cabinet |
| (In akita_black_bowl_1 wooden_cabinet_1_top_region) | 1 open the top drawer and put the bowl inside |
| (On plate_1 main_table_stove_front_region) | 2 push the plate to the front of the stove |
| (On akita_black_bowl_1 plate_1) | 3 put the bowl on the plate |
| (On akita_black_bowl_1 flat_stove_1_cook_region) | 4 put the bowl on the stove |
| (On akita_black_bowl_1 wooden_cabinet_1_top_side) | 5 put the bowl on top of the cabinet |
| (On cream_cheese_1 akita_black_bowl_1) | 6 put the cream cheese in the bowl |
| (On wine_bottle_1 wine_rack_1_top_region) | 7 put the wine bottle on the rack |
| (On wine_bottle_1 wooden_cabinet_1_top_side) | 8 put the wine bottle on top of the cabinet |
| (Turnon flat_stove_1) | 9 turn on the stove |

## Chunk labels

A **record** is the time-ordered sequence of recorded states of one piece of data. A chunk (o_t, a_t) of a record has o_{t+1}, the state right after the chunk. For task l:

- **positive**: some recorded state after o_t in the same record satisfies l. There is no time window.
- **negative**:
  - break: l holds at o_t and not at o_{t+1};
  - conflict: o_{t+1} newly satisfies (true there, false at o_t) the goal of a task that conflicts with l, and does not satisfy l.
- **undetermined**: neither, or both.

**Conflicts** (`TaskGoalConflicts`): tasks i ≠ j conflict when an On / In conjunct of each has the same first argument (the placed object) and different second arguments (the place). Pairs:

- bowl: (1, 3), (1, 4), (1, 5), (3, 4), (3, 5), (4, 5);
- wine bottle: (7, 8).

Tasks 0, 2, 6 and 9 conflict with no task: goals on different objects (e.g. the drawer and the stove, or the plate and the bowl on it) are never negatives of each other. No labelled state satisfies both goals of a conflicting pair (0 of 141,664 states of the sources below).

Records per source (`report_task_goal_labels.py`):

| Source | Record | Chunks |
|---|---|---|
| `demo_renders` | one demonstration: every frame in t order | the store's transitions (t, t + 10); o_{t+1} is frame t + 10 |
| Phase 26 records (π-mean chains) | one π-mean episode: the decision states d = 1 … D restored without rendering, then the final state after the last decision's chunk | every decision d; o_{t+1} is decision d + 1 or the final state |
| collected stores (resim, counterfactual, trial2) | o, o′, and the chain continuing from o′ through the transitions whose current frame is the previous o′ (resim-mean chains, walks, pi-noise chains) | the store's transitions |
| pilot branches | [o, o′] | one per executed candidate; the demonstration chunk carries its `demo_renders` label |

A **group** is the set of chunks executed from one start state, counted when it holds at least two chunks.

## Checks

**Cross-task check** (`label_task_goals.py check`). Each state is evaluated in the env of every task. The reference is LIBERO's `check_success` of each task in its own env, after the same load and after a full restore (reset, fixture poses, `set_init_state`).

| Sample | States | True cases per goal | Every goal in every env = the goal's own env | Own env, load = full restore |
|---|---|---|---|---|
| `demo`: per goal 30 frames where the sidecar's goal holds, 30 in the 20 frames before it first holds, 30 of other tasks' shards | 900 | 30–39 (task B's own states) | 90,000 / 90,000 | 9,000 / 9,000 |
| `labelled`: per goal up to 30 states of other tasks' data where the label is true, 30 where it is false | 450 | 31–36 from other tasks for goals 0, 2, 3, 4, 5; goals 1, 6, 7, 8, 9 hold in no other task's store state | 45,000 / 45,000 | 4,500 / 4,500 |

**Label against the live check.** The own-task label is compared with the sidecar's `goal_holds`:

| Store | States | Label true | Live true | Label only | Live only |
|---|---|---|---|---|---|
| demo_renders | 52,895 | 5,474 | 5,462 | 88 | 76 |
| resim_phase26_mean (3 stores) | 20,349 | 11 | 0 | 11 | 0 |
| resim_phase26_sweep (3 stores) | 35,838 | 10 | 10 | 0 | 0 |
| counterfactual_eval_heldout | 8,514 | 208 | 214 | 2 | 8 |
| trial2 (5 stores) | 2,219 | 69 | 71 | 0 | 2 |

The disagreements are 0.3% of the demonstration frames. Hypothesis, not checked frame by frame: they are frames at the goal boundary, where the poses one substep apart decide. In the resim-mean stores the live check is False at every decision, because a π-mean episode ends at the first success.

**Phase 26 records relabelled without rendering.** Each record decision is restored as recorded (`PhysicsTaskEnv.restore_step`: the snapshot one step before the decision, then the producing step). The final state executes the last decision's chunk and stops at success.

| ψ | Decision goal vectors = resim-mean store sidecar of the same decision | Final chunk steps = record | Final success = record |
|---|---|---|---|
| fixed | 7,360 / 7,360 | 500 / 500 | 500 / 500 |
| flow | 6,507 / 6,507 | 500 / 500 | 500 / 500 |
| flow_full_sgd | 6,482 / 6,482 | 500 / 500 | 500 / 500 |

**Execution without rendering.** Restoring a `demo_renders` frame t from its sidecar snapshot (`PhysicsTaskEnv.restore`) and executing the demonstration chunk reproduces the sidecar state of frame t + 10:

- smoke on task 3, 61 transitions: qpos max |Δ| 3.0e-6, goal vectors equal 61 / 61;
- pilot DEMO candidates, 200 states: largest object distance ≤ 7.5e-8 m, goal vectors equal 200 / 200.

No GL context is imported: `mujoco.gl_context` and robosuite's `binding_utils` define no `GLContext` under `MUJOCO_GL=disable`. `/tmp/robosuite.log` on node-61 and node-65 was last modified before the runs.

## Goal prevalence

States where each goal holds (all splits):

| Source | States | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| demo_renders | 52,895 | 454 | 340 | 494 | 686 | 626 | 600 | 447 | 680 | 563 | 584 |
| Phase 26 records (decisions + final states) | 21,849 | 165 | 110 | 163 | 153 | 250 | 552 | 103 | 92 | 141 | 149 |
| resim_phase26_mean | 20,349 | 16 | 1 | 21 | 39 | 114 | 424 | 1 | 2 | 0 | 0 |
| resim_phase26_sweep | 35,838 | 99 | 0 | 129 | 221 | 720 | 2,758 | 9 | 0 | 0 | 0 |
| counterfactual_eval_heldout | 8,514 | 0 | 0 | 33 | 52 | 37 | 23 | 18 | 26 | 11 | 8 |

- No demonstration reaches another task's goal: every TRAIN demonstration chunk positive for task l belongs to a task-l demonstration.
- π-mean episodes do reach other tasks' goals: the bowl ends on the cabinet (g5) in task 3 and 4 episodes; the bowl on the stove (g4) in tasks 3, 5, 6; the plate in front of the stove (g2) in task 1; the middle drawer open (g0) in task 7. These are the goal-holding states of the resim stores.

## Chunk labels per source

Cells: positive / negative chunks. Negatives are split into b breaks and c conflicts. Every other chunk is undetermined.

| Source (split) | Chunks | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| demo_renders (TRAIN) | 43,701 | 4,849 / 0 | 5,779 / 1,304 (1304c) | 4,999 / 0 | 3,737 / 1,127 (1127c) | 4,070 / 1,137 (1137c) | 3,776 / 1,160 (1160c) | 3,225 / 0 | 5,616 / 417 (417c) | 3,970 / 394 (394c) | 3,559 / 0 |
| demo_renders (HELDOUT) | 4,864 | 554 / 0 | 593 / 149 (149c) | 547 / 0 | 404 / 129 (129c) | 503 / 129 (129c) | 462 / 130 (130c) | 407 / 0 | 660 / 37 (37c) | 335 / 40 (40c) | 395 / 0 |
| Phase 26 records (TRAIN, inits 0–39) | 16,434 | 1,510 / 0 | 1,581 / 344 (344c) | 1,629 / 1 (1b) | 792 / 337 (1b 336c) | 1,148 / 318 (318c) | 1,392 / 291 (5b 286c) | 816 / 0 | 975 / 113 (113c) | 1,140 / 69 (69c) | 945 / 0 |
| Phase 26 records (HELDOUT, inits 40–49) | 3,915 | 400 / 0 | 465 / 86 (86c) | 374 / 0 | 220 / 87 (1b 86c) | 288 / 83 (83c) | 313 / 78 (78c) | 223 / 0 | 303 / 29 (1b 28c) | 280 / 22 (22c) | 247 / 0 |
| resim_phase26_mean (TRAIN) | 15,234 | 29 / 0 | 18 / 75 (75c) | 75 / 1 (1b) | 47 / 74 (1b 73c) | 181 / 67 (67c) | 653 / 17 (4b 13c) | 0 / 0 | 14 / 0 | 0 / 1 (1c) | 0 / 0 |
| resim_phase26_sweep (TRAIN), rates only | 29,184 | 96 / 0 | 0 / 442 (442c) | 126 / 0 | 152 / 450 (8b 442c) | 634 / 448 (6b 442c) | 2,443 / 239 (239b) | 9 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| counterfactual_eval_heldout, rates only | 8,256 | 0 / 0 | 0 / 112 (112c) | 32 / 0 | 52 / 60 (60c) | 37 / 75 (75c) | 23 / 89 (89c) | 18 / 0 | 26 / 11 (11c) | 11 / 26 (26c) | 8 / 0 |

- **Demonstrations.** Every TRAIN demonstration chunk is positive for its own task except 121 undetermined chunks (positive and conflict at once, 69 of them in task 7).
  - Negatives come only from other tasks' demonstrations near their completion: conflicts of the bowl tasks (1, 3, 4, 5) and of the wine-bottle tasks (7, 8).
  - Tasks 0, 2, 6 and 9 have no negative chunk in any demonstration.
- **resim_phase26_mean vs Phase 26 records.** The resim-mean store lacks the last chunk of each episode, whose end state was not rendered. The Phase 26 records add it through the final state. That chunk is the positive of every successful episode.
- **trial2 stores** (2,113 chunks in all): positives for tasks 1, 3, 4, 5, 8 and negatives for tasks 1, 3, 4, 5, 7, at most 80 per task and store; trial2_init_random has none.
- **Use.** The resim_phase26_sweep states and counterfactual_eval_heldout serve only to estimate rates, never for training.

Groups with at least one positive and one negative (sources with several chunks executed at one observation):

| Source | Groups | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resim_phase26_sweep (TRAIN), rates only | 912 | 0 | 0 | 0 | 1 | 1 | 33 | 0 | 0 | 0 | 0 |
| resim_phase26_sweep (HELDOUT), rates only | 174 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| counterfactual_eval_heldout, rates only | 258 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial2_demo_branch | 32 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| trial2_demo_branch_eval | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial2_resim_branch | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- The 36 sweep groups of task 5 are π-failure states where the bowl already rests on top of the cabinet (the sweep's task-5 positives come from task-3 and task-4 episodes): some candidates keep it there, others knock it off (239 breaks in TRAIN).
- In the counterfactual set, groups hold positives for tasks 2–9 (up to 6 groups) and negatives for tasks 1, 3, 4, 5, 7, 8 (up to 16 groups), never both in one group.

## Pilot: single-chunk branches from TRAIN demonstration states

`label_task_goals.py branch-labels`: the start states are TRAIN query / match observations (`outputs/query_match/tables_mvp`; observation i is `demo_renders` transition i, so t ≤ T − 11). Each state is restored from its `demo_renders` sidecar snapshot, and every candidate runs once from a fresh restore without rendering.

Candidates per state (`BranchCandidateConfig` defaults, seed 0), 44 in all:

| Type | Count | Construction |
|---|---|---|
| DEMO | 1 | the observation's demonstration chunk (labelled with its `demo_renders` record) |
| π mean + small offsets | 1 + 10 | the tables' π seed-0 mean chunk (`ProposalCandidates` candidate 0) and `ProposalCandidates` offsets at scales 0.05 / 0.1 / 0.2, J = 1 |
| π large offsets | 11 | the same generator at scales 0.5 / 1 / 2, J = 1 |
| other-task demo | 11 | demonstration chunks of 11 of the tables' 64 other-task neighbours, random order |
| smooth random | 10 | `SmoothActionSampler`, J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, p = 0.3 |

The **base mix** is π 11, other-task 11 and smooth 10: 32 candidates, the pre-registered set. The π-large and DEMO candidates are variants.

Two designs, 100 states each:

- **stratified** (pre-registered): one state per task and position bin b = floor(10 t / T), 10 × 10;
- **goal-holds** (supplement): 10 states per task where the task's own goal holds at o. These fall in bins 5 (1 state), 7 (6), 8 (54) and 9 (39).

**Group model.** For state s, task l and candidate type c, p⁺ and p⁻ are the fractions of the executed candidates labelled positive and negative. With K_c candidates of each type:

P_s(both) = 1 − Π_c (1 − p⁺)^{K_c} − Π_c (1 − p⁻)^{K_c} + Π_c (1 − p⁺ − p⁻)^{K_c}.

p_l(K) is its mean over the design's states, with 95% intervals from 2,000 bootstrap resamples of the states.

### Stratified design

- **Goal-holding states.** 9 of the 100 states: g0 1, g3 1, g4 1, g5 1, g7 2, g8 2, g9 1.
- **Groups.** States whose 32 base candidates include a positive and a negative:

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| base 32 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 2 | 2 | 0 |
| base 32 + DEMO | 1 | 0 | 0 | 1 | 1 | 2 | 0 | 2 | 2 | 0 |

- **Where the groups are.** Every base-mix group is at a state where l already holds, in position bins 7–9. p_l(32) is 0 in bins 0–6 for every task.
- **The DEMO variant.** It adds one group: task 5 at a task-4 state, where the demonstration chunk is positive by its continuation and a branch places the bowl on the stove (conflict).
- **Positives without negatives.** Positives also exist at states where l does not hold, from candidates that complete l within one chunk (e.g. π candidates that turn on the stove). Those states have no negative for l: l does not hold there, and a conflicting place is out of reach in one chunk.
- **Negatives without positives.** Conflicts occur at other bowl or wine tasks' states, which have no positive for l.

p_l(32), base mix: 0.010 (tasks 0, 3, 4, 5), 0.020 (tasks 7, 8), 0 (tasks 1, 2, 6, 9; 95% upper bound about 0.03 from 0 of 100 states).

| Task | p_l(32) | States for 200 groups [95%] | Branches (K = 32) | CPU hours |
|---|---|---|---|---|
| 0, 3, 4, 5 | 0.010 | about 20,000 [6,700, ∞] | about 640,000 | about 26 |
| 7, 8 | 0.020 | about 10,000 [4,000, ∞] | about 325,000 | about 13 |
| 1, 2, 6, 9 | 0 of 100 states | not estimable (≥ 6,700 at the upper bound) | – | – |

Those states would still yield their groups only from the few among them where the goal holds.

### Goal-holds design

Candidates labelled positive / negative (b break, c conflict) per type and task, over the 100 states (in practice from the 10 states where l holds):

| Type (per task) | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| π mean + small (110) | 110 / 0 | 109 / 1b | 110 / 0 | 110 / 0 | 110 / 0 | 110 / 0 | 110 / 0 | 83 / 27b | 86 / 24b | 110 / 0 |
| π large (110) | 109 / 1b | 106 / 4b | 95 / 15b | 84 / 26b | 104 / 6b | 94 / 16b | 90 / 20b | 62 / 48b | 90 / 20b | 110 / 0 |
| other-task demo (110) | 70 / 40b | 84 / 26 (26b 1c) | 108 / 2b | 50 / 61 (60b 1c) | 75 / 36 (35b 1c) | 91 / 20b | 80 / 30b | 66 / 44b | 105 / 5b | 109 / 1b |
| smooth random (100) | 69 / 31b | 85 / 15b | 70 / 30b | 63 / 37b | 75 / 25b | 70 / 30b | 77 / 23b | 65 / 35b | 61 / 39b | 98 / 2b |

- **Negatives are breaks.** At goal-holding states the negatives are breaks, with 3 conflicts in 4,300 candidates.
- **Which candidates break.** π's small offsets break the goal only for the wine-bottle tasks (and once for task 1). Other-task chunks, smooth random chunks and π's large offsets break it for every task except 9.
- **Task 9.** Turning the stove off again happens in 3 of 430 candidates.

p_l(K) over the 10 states where l holds, base mix. With the DEMO chunk the values are the same: at these states it is positive and adds no negative.

| K | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.82 | 0.41 | 0.57 | 0.92 | 0.71 | 0.73 | 0.52 | 0.81 | 0.77 | 0.06 |
| 16 | 0.95 | 0.47 | 0.76 | 0.98 | 0.83 | 0.90 | 0.63 | 0.88 | 0.91 | 0.08 |
| 32 | 0.99 | 0.49 | 0.90 | 1.00 | 0.92 | 0.98 | 0.75 | 0.90 | 0.98 | 0.10 |
| 32, 95% interval | [0.98, 1.00] | [0.20, 0.79] | [0.81, 0.97] | [1.00, 1.00] | [0.82, 0.99] | [0.96, 1.00] | [0.55, 0.91] | [0.70, 1.00] | [0.96, 1.00] | [0.00, 0.29] |
| 64 | 1.00 | 0.50 | 0.97 | 1.00 | 0.97 | 1.00 | 0.85 | 0.90 | 1.00 | 0.10 |

With π's large offsets added (base + 11, K = 32 in those proportions): task 6 rises to 0.84, task 2 to 0.97 and task 7 to 0.95; tasks 4 and 8 drop by 0.02; the others change by at most 0.01. Task 1 is bimodal: in 5 of its 10 states no candidate breaks the goal. In the other 5, some candidate lifts the bowl out of the drawer.

### Estimate: branching every TRAIN state where the goal holds, K = 32

The available start states are those where l holds:

- **observations**: TRAIN query / match observations (t ≤ T − 11);
- **frames**: every TRAIN `demo_renders` frame, including the last 10 frames of each demonstration, which are not query / match observations.

Expected groups = available states × p_l(32 | l holds), with the interval of p_l(32). The frame column applies the rate measured at observations to the last 10 frames, which is a hypothesis.

| Task | p_l(32), l holds | Observations with l (demos) | Expected groups | Frames with l | Expected groups (frames) | States for 200 groups |
|---|---|---|---|---|---|---|
| 0 | 0.99 | 36 (25) | 36 [35, 36] | 410 | 407 [403, 410] | 202 |
| 1 | 0.49 | 33 (18) | 16 [7, 26] | 305 | 151 [60, 242] | 405 |
| 2 | 0.90 | 71 (24) | 64 [57, 69] | 402 | 362 [325, 391] | 222 |
| 3 | 1.00 | 183 (39) | 183 [182, 183] | 616 | 615 [614, 616] | 200 |
| 4 | 0.92 | 117 (30) | 107 [96, 116] | 556 | 510 [457, 551] | 218 |
| 5 | 0.98 | 131 (39) | 128 [125, 130] | 539 | 527 [515, 536] | 204 |
| 6 | 0.75 | 64 (32) | 48 [35, 58] | 403 | 301 [223, 365] | 267 |
| 7 | 0.90 | 297 (29) | 266 [207, 297] | 622 | 558 [434, 622] | 223 |
| 8 | 0.98 | 132 (35) | 130 [126, 132] | 523 | 514 [501, 521] | 204 |
| 9 | 0.10 | 90 (31) | 9 [0, 26] | 523 | 50 [0, 151] | 2,078 |

Phase 26 TRAIN decisions offer further goal-holding states, mostly where a π-mean episode left another task's goal satisfied: g5 382, g4 104, g3 29, g2 21, g0 16 decisions. Their frames are rendered in the resim-mean stores, and their snapshots are restorable.

**Cost.** Per branch (restore, 10 steps, snapshot, label) the pilots measured a mean of 0.146 s and 0.170 s per process, with 10 processes on a node shared with other jobs.

| Collection (K = 32, base mix) | States | Branches | CPU hours at 0.17 s | Wall, 16 processes | Tasks expected to reach 200 groups |
|---|---|---|---|---|---|
| goal-holding TRAIN observations | 1,154 | 36,928 | 1.7 | about 7 min | 7 (266); 3 at 183 |
| goal-holding TRAIN frames (needs π mean chunks and neighbours for the last 10 frames of each demonstration) | 4,899 | 156,768 | 7.4 | about 28 min | 0, 2, 3, 4, 5, 6, 7, 8 (hypothesis for the last 10 frames); not 1 (151) and 9 (50) |
| stratified over positions (not recommended) | about 20,000 | about 640,000 | about 30 | about 2 h | 0, 3, 4, 5, 7, 8 (point estimates from 1–2 states each) |

**Limits of these groups.**

- **What the groups contrast.** Every group is "keep the achieved goal" against "break it", at the end of a demonstration: frames in position bins 7–9, most in bins 8–9. Single-chunk branches from states before the goal holds give positives (completing chunks) or conflict negatives, never both at one state.
- **Diversity.** A task's goal-holding frames come from its 18–39 TRAIN demonstrations; consecutive frames of one demonstration are close to each other.
- **Estimate precision.** The rates at goal-holding states rest on 10 states per task. The intervals of tasks 1, 6, 7 and 9 are wide.

## Runs

- **Hosts.** node-61 and node-65, CPU only (no `--gpus`), through `scripts/cluster_run.sh`.
- **Environment.** `MUJOCO_GL=disable`; OMP / MKL / OPENBLAS / NUMBA threads 2. At most 11 of this phase's processes per node at a time. Node load before the launches: 10 (node-61) and 13 (node-65), 128 cores each.
- **Code.** Checks, labels and pilots: commit `4f33afb`, recorded in the outputs' provenance. The pilots ran after the removal of an unused report placeholder from `label_task_goals.py`, committed in `08a32f7`; `check`, `label` and `branch-labels` are identical in both. Report: `08a32f7`. Hashes in `tmp/s4p20/code_commit*.txt`.
- **Launchers.** `tmp/s4p20/launch_workers.sh`, `run_worker.sh`, `launch_report.sh`, `run_report.sh`. Logs: `tmp/runlogs/s4p20/`.

| Step | Host | Processes | Wall (EDT, 2026-09-29) | Work |
|---|---|---|---|---|
| `check --sample demo` | node-61 | 1 | 45 s | 900 states × 10 envs |
| `label` | node-65 | 10 | 00:23:24–00:25:06 | 119,815 store states, 20,349 record decisions, 1,500 final states |
| `check --sample labelled` | node-65 | 1 | 27 s | 450 states × 10 envs |
| `branch-labels` stratified | node-61 | 10 | 79 s | 100 × 44 branches |
| `branch-labels` goal-holds | node-65 | 10 | 91 s | 100 × 44 branches |
| report `final` | node-61 | 1 | under 1 min | – |

Throughput per process:

| Operation | Median | Count |
|---|---|---|
| stored state: load + 10 goals | 0.68 ms (mean 1.31) | 119,815 |
| record decision: restore + producing step + label | 18.1 ms | 20,349 |
| record final state: restore + chunk + label | 121 ms | 1,500 |
| pilot branch: restore + 10 steps + snapshot + label | 142 / 163 ms | 4,400 / 4,400 |

That is about 1,500 stored states per second per process; all stored states of the listed sources label in under 3 min of process time.

```bash
source scripts/ceam_env.sh   # CEAM_ROOT = main checkout, PYTHONPATH = the worktree
export MUJOCO_GL=disable OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
$CEAM_PYTHON scripts/label_task_goals.py check --name demo_sample --sample demo --per-goal 30
$CEAM_PYTHON scripts/label_task_goals.py label --worker <i> --workers 10
$CEAM_PYTHON scripts/label_task_goals.py check --name labelled_sample --sample labelled --per-goal 30
$CEAM_PYTHON scripts/label_task_goals.py branch-labels --name stratified_v1 --design stratified --bins 10 --states-per-cell 1 --seed 0 --worker <i> --workers 10
$CEAM_PYTHON scripts/label_task_goals.py branch-labels --name goal_holds_v1 --design goal-holds --states-per-task 10 --seed 0 --worker <i> --workers 10
$CEAM_PYTHON scripts/report_task_goal_labels.py --out final --pilots stratified_v1 goal_holds_v1 --write-chunks
```

## Outputs

All under `outputs/analysis/task_goal_labels/`:

| Path | Content |
|---|---|
| `goals/<store>/<shard>.npz` | `goals` [n, 10] per sidecar row, with the sidecar's `task_id`, `episode`, `t`, `goal_holds` (`sidecar_goal_holds`), `seconds`, provenance |
| `goals/phase26_record/<ψ>_task<TT>.npz` | per record row: `init`, `decision`, `goals`, the restored decision's `sim_state`, `accumulator`, `goal_orientation`; per episode: `final_goals`, `final_sim_state`, `final_steps`, `final_success`, `recorded_chunk_steps`, `recorded_success` |
| `check/demo_sample.{npz,json}`, `check/labelled_sample.{npz,json}` | the cross-task checks |
| `pilot/<name>/part_<i>_of_010.npz` | per state: observation, episode, t, T, task, bin, `goals_before`; per candidate: `kinds`, `scales`, `dct`, `source_observation`, `raw` chunk, `goals_after`, `next_sim_state`, `success`, `seconds`; DEMO checks |
| `chunks/<source>.npz` | per chunk: `label` [m, 10] int8 (1 positive, −1 negative, 0 undetermined), `positive`, `breaks`, `conflicts`, `split`, `source_task`, `kind`, `group`, `shard`, `item`, `start_goals`, `next_goals` |
| `report/final.md`, `report/final.json` | every table of this page; `dev1`–`dev4` are development runs of the same report |

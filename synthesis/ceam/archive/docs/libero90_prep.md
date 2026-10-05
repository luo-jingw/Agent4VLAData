# LIBERO-90 preparation (query-independent parts)

This page covers the parts of the move to LIBERO-90 that do not depend on the query or the match: the demonstration data, the rendered frames, the R6 action representation r(o, a) (`docs/framework.md` §1, action side; its distinguishability is §2, action side) and the no-ψ π, which is the sampler of §4 and the baseline of the final criterion (§3). Query and match work is not part of it.

## Plan

Estimates are marked as such. The CPU conversion and LeRobot v2.1 dataset counts are measured locally; later rendering, training and evaluation costs remain estimates.

### Data source

**LIBERO-Goal (existing).**

| Part | Source | Local path |
|---|---|---|
| LeRobot v2.1 episodes, 512² AV1 video, 20 fps | `Zhanguang/libero_mujoco3.3.2` (Fast-WAM release), `libero_goal_no_noops_lerobot.tar.gz` | `outputs/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot` (809 MB) |
| raw hdf5, 10 files × 50 demos | `yifengzhu-hf/LIBERO-datasets` at `f13aa24`, folder `libero_goal/` | `outputs/raw/LIBERO-datasets/libero_goal` (6.0 GB) |

The release was regenerated from the raw demos under MuJoCo 3.3.2 with the OpenVLA procedure (`docs/libero_sim.md`):

- the no-op rule;
- fixture poses of reset k + 1 after `seed(0)` for `demo_k`;
- raw `states[0]` and 10 settle no-ops;
- the no-op-filtered raw actions stepped in order;
- a demo kept only if LIBERO's done holds after its last action.

This keeps 433 of 500 demos.

**LIBERO-90.** No release in this format exists:

- the Fast-WAM release covers libero_10, goal, object and spatial only;
- `jesbu1/libero_90_lerobot` is 256², LeRobot v2.0, a different regeneration, and uses 73 instructions;
- `jesbu1/libero_90_openvla_processed` has 225 GB of hdf5, and one of its files is 1.8 kB.

The source is therefore the raw hdf5 of the same repository, regenerated here with the Fast-WAM/OpenVLA procedure under MuJoCo 3.3.2.

| Item | Value |
|---|---|
| raw source | `yifengzhu-hf/LIBERO-datasets` at `f13aa24a3da8c43c7225569f28c562979fa0e35a`, folder `libero_90/`, 90 hdf5 files, 66.66 GB, anonymous (`token=False`) |
| raw local path | `outputs/raw/LIBERO-datasets/libero_90/` (read-only after download) |
| tasks | 90 benchmark tasks in 20 scenes (10 kitchen, 6 living room, 4 study); 74 distinct instructions. 12 instructions occur in 2–3 scenes (28 tasks) |
| regenerated dataset | 3,949 retained episodes and 572,988 frames; one task retains 0 of 50 demos |
| output | `outputs/datasets/ceam_regenerated/libero_90_no_noops` in LeRobot v2.1 layout without videos: `data/chunk-*/episode_*.parquet`, `meta/{info,tasks,episodes,evaluation_tasks}.jsonl`, `meta/episode_sources.jsonl` |

Parquet columns follow the Goal release:

- `action` [7], dataset gripper convention: `(1 − raw) / 2`;
- `observation.states.joint_state` [7], `observation.states.gripper_state` [2], `observation.states.ee_state` [6];
- `frame_index`, `episode_index`, `index`, `task_index`.

`meta/episode_sources.jsonl` gives per episode:

- the benchmark task name (the BDDL stem, e.g. `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet`);
- its evaluation-task index, original task text and scene;
- the raw file and `demo_k`;
- the kept raw steps.

`meta/evaluation_tasks.jsonl` gives all 90 benchmark task identities independently of retained episodes. It therefore includes `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`, whose 50 replays all fail the deterministic final-step success rule.

**Conversion** (`scripts/convert_libero_raw.py`, new; CPU only, `PhysicsTaskEnv`, one process per task):

1. Compute the fixture poses of resets 1 … 50 after `seed(0)` in a fresh GL-free env of the task.
2. For each `demo_k` in numeric order:
   - apply the no-op rule (`DemoStateRestorer.keep_mask`);
   - reset to raw `states[0]` with the reset-(k + 1) fixture poses, zeroing the gripper accumulator;
   - take 10 settle no-ops;
   - step the kept actions one by one, recording before each step the proprio and end-effector state the policy observes;
   - keep the demo if done holds after the last action.

**Checks** (`scripts/convert_libero_raw.py check`):

- **(a) Procedure.** The converter run on raw LIBERO-Goal must reproduce the Fast-WAM release:
  - all 433 release episodes map uniquely to raw demos by equal length and exact float32 actions;
  - a fresh regeneration is payload-identical to an independent prior regeneration, excluding provenance and runtime;
  - replay-kept count and proprio divergence are measured outputs.
- **(b) MuJoCo 3.3.2 replay divergence on LIBERO-90:**
  - kept share per task;
  - first-done step against the episode end;
  - `|replay proprio − raw obs joint_states / gripper_states|` at the kept steps;
  - object positions of raw `states[raw_step]` against the replay, per task (the Goal table in `docs/demo_renders.md`).
- **(c) Render replay against the conversion replay.** In the demo render (S3), `DemoStateRestorer` replays each episode in the render env. Its proprio is compared with the parquet proprio of the GL-free conversion.

### Measured conversion validation

The full LIBERO-90 conversion ran on node-61, node-63, node-65 and node-66 with eight CPU workers, two per node. Every worker used `MUJOCO_GL=disable` and one OMP, MKL, OpenBLAS and NUMBA thread; no GPU was allocated. All 90 raw files produced loadable regeneration records. The slowest worker completed in 1,277.91 seconds. Conversion, reporting, assembly and successful exact validation used 9,053.59 user plus 113.07 system CPU-seconds. Per-worker peak RSS was 3.22–3.61 GiB; reporting, assembly and validation took 2.87, 12.94 and 20.61 seconds wall time. The records occupy 860,311,572 bytes and the assembled dataset occupies 106,970,794 bytes.

The assembled LeRobot v2.1 dataset contains 3,949 episodes and 572,988 frames. Its 74 `task_index` values represent the distinct original instructions, while `evaluation_tasks.jsonl` represents all 90 task-and-scene identities in 20 scenes. Twelve instructions occur in more than one scene and cover 28 evaluation tasks. All 3,949 episode-source rows map one-to-one to retained raw demos. Every parquet vector and index column was compared exactly with the corresponding regeneration payload. The complete per-task kept counts and replay divergence are in `docs/libero90_replay_report.md`.

`data/instructions/libero_90.json` contains five globally unique command forms for each instruction, 370 strings total, in dataset task-index order. `data/task_relations/libero_90.json` contains 90 benchmark-name-keyed task records and yields a 90 × 90 relation mask. It records 3,842 non-negative pairs and 163 negative pairs after excluding each task's diagonal.

The 2026-09-30 validation ran on node-65 with `MUJOCO_GL=disable` and no GPU allocation. The process cpuset exposed two logical CPUs. Ten Goal task processes completed in 164 seconds wall time and used 1,116.35 CPU-seconds in total. Six LIBERO-90 sample processes completed in 185 seconds wall time and used 641.62 CPU-seconds. Per-process peak RSS was 1.83–2.70 GiB. The outputs occupy 92 MiB for Goal and 57 MiB for the sample.

The empty-fixture serialization now writes explicit arrays with shape `[n_demo, 0, 3]` for positions and `[n_demo, 0, 4]` for quaternions. All six sample tasks produced loadable records:

| Benchmark task | Kept / demos | Frames | Fixture position / quaternion shape |
|---|---:|---:|---|
| `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet` | 43 / 50 | 4,693 | `[50, 1, 3]` / `[50, 1, 4]` |
| `KITCHEN_SCENE3_turn_on_the_stove` | 50 / 50 | 4,513 | `[50, 1, 3]` / `[50, 1, 4]` |
| `KITCHEN_SCENE6_close_the_microwave` | 46 / 50 | 10,142 | `[50, 1, 3]` / `[50, 1, 4]` |
| `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` | 0 / 50 | 6,502 | `[50, 0, 3]` / `[50, 0, 4]` |
| `LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray` | 48 / 50 | 10,722 | `[50, 0, 3]` / `[50, 0, 4]` |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | 34 / 50 | 8,740 | `[50, 1, 3]` / `[50, 1, 4]` |

The four sample tasks that had completed before this repair reproduced all 22 payload arrays exactly, excluding provenance and measured runtime. The two empty-fixture tasks are new successful serializations.

The fresh Goal run reproduced all 22 payload arrays of every task exactly against the previous run, excluding provenance and measured runtime. Its comparison with the release measured:

- 433 release episodes matched uniquely by length and float32 action; action maximum absolute difference was zero;
- 432 demos satisfied the replay's end-of-episode success rule; `push_the_plate_to_the_front_of_the_stove/demo_25` first satisfied the goal after action 135 but did not satisfy it after the final action;
- per-episode joint maximum absolute difference had median `1.24e-7`, p90 `9.72e-7`, and maximum `9.302508668e-3`; 17 episodes exceeded `1.5e-4`;
- gripper maximum absolute difference was `7.54e-4`;
- the largest joint differences were the same episodes and values already measured in `docs/demo_renders.md`, including release episode 91 at `9.3e-3`, episode 78 at `6.6e-3`, and episode 69 at `5.7e-3`.

Changing the replay keep rule from success after the final action to success after any action would retain 448 demos, not the release's 433. It adds 16 demos whose goal became true and later false:

- `push_the_plate_to_the_front_of_the_stove`: `demo_25`, `demo_31`, `demo_32`, `demo_38`, `demo_41`, `demo_43`;
- `put_the_cream_cheese_in_the_bowl`: `demo_4`, `demo_10`, `demo_26`, `demo_29`, `demo_42`;
- `put_the_wine_bottle_on_the_rack`: `demo_3`, `demo_16`, `demo_25`, `demo_42`;
- `put_the_wine_bottle_on_top_of_the_cabinet`: `demo_8`.

For `push_the_plate_to_the_front_of_the_stove/demo_25`, the raw length, no-op-filtered length and matched release length are all 153. The replay first reports done at zero-based action index 135 and reports false after action index 152. Length or no-op filtering therefore does not explain its kept-set difference.

The comparison command accepts the conversion because all 433 release episodes match uniquely and their float32 actions are exact. It reports replay-kept counts, kept-set differences and proprio errors without turning these measured replay differences into pass/fail thresholds. Two independent regenerations are payload-identical, and the measured divergence reproduces the established `docs/demo_renders.md` result. The six-task structural validation and Goal action validation are complete.

**Instructions.** `data/instructions/libero_90.json` holds the 74 instructions as `variants[0]`, plus four unique command-form paraphrases per instruction, as for Goal. π reads `variants[0]` only. `InstructionTable` requires at least two variants and unique strings, so the paraphrases keep it unchanged.

**Task identity.** A LeRobot `task_index` is per distinct instruction (74), as `EpisodeIndex` and the LeRobot format require: `episodes.jsonl` names tasks by text. The scene is an episode attribute (`episode_sources.jsonl`). The TRAIN / HELDOUT split is `EpisodeIndex`'s: heldout fraction 0.1, seed 0, per `task_index`. Per-scene HELDOUT counts are reported.

### Env support

| Component | On LIBERO-90 | Change (opt-in; Goal path unchanged) |
|---|---|---|
| `LiberoTaskEnv` | Selects the benchmark task by language. The 28 tasks whose instruction occurs in 2–3 scenes raise `ValueError`. Everything else is suite-independent: fixture poses from `fixtures_dict` (tables included), snapshot/restore, settle, cameras `agentview` / `robot0_eye_in_hand`, 7-D OSC_POSE, one robot. | Optional constructor argument `benchmark_task_name: str \| None = None`. When given, the task is selected by LIBERO `task.name` and its language must equal `task_text`. `None` runs the current code. **Protected module: needs approval.** |
| `DemoStateRestorer` | Works unchanged if the regeneration follows the release's convention (no-op rule, gripper mapping, fixture poses of reset k + 1 after `seed(0)`). The conversion follows it by construction. | none |
| `PhysicsTaskEnv` | Same language selection. | Same optional argument, plus `reset_to_state(sim_state, fixture_poses, settle_steps)` and `fixture_pose_sequence(seed, resets)` for the conversion. The existing methods are unchanged. |
| `SceneStateReader`, `SceneProbe` | Generic over objects and fixture joints. Construction raises if a scene has a free joint that is not an object's, or a fixture joint that is neither slide nor hinge. Not yet run on the 20 scenes. | Verify on one task per scene (S2). |
| task relations (`TaskRelationTable`, used by `SceneProbe`) | Keyed by instruction text. One text has different goals in different scenes: "put the black bowl on top of the cabinet" is `wooden_cabinet_1` in scene 1 and `white_cabinet_1` in scenes 4 and 5. | `data/task_relations/libero_90.json` keyed by benchmark task name; opt-in name lookup. |
| `SceneSidecar` | Per-shard arrays. `sim_state` has the scene's length (79 on Goal). | none |
| `TaskGoalLabeller` | Assumes one scene shared by all tasks, so it fails across scenes. | Per-scene goal subset (opt-in). Not needed for the L1 outputs; needed later for cross-task goal labels. |
| scripts: `render_demo_episodes.py`, `collect_interactions.py`, `collect_demo_perturbations.py`, `run_closed_loop.py` | `SUITE = "libero_goal"` constants; task found by text; `run_closed_loop.py` requires a stage-one checkpoint for ψ logging, retrieval tables and the start check. | Opt-in `--suite`, `--dataset-root`, `--raw-dir`, `--episode-sources`, with Goal defaults. `run_closed_loop.py` gets a π-only mode without a stage one. |
| π data (`ProposalTrainer`) | Reads a `FeatureCache` built from video frames; LIBERO-90 has no video. | `scripts/build_rendered_feature_cache.py` (new) writes the `FeatureCache` layout from the demo-render store. `ProposalTrainer` is unchanged. |

**Bit-identity checks on Goal**, with the base and new code, as in `docs/no_psi_baseline.md`:

- `LiberoTaskEnv` / `PhysicsTaskEnv` without the new argument: snapshot after `reset`, `restore` and a 10-step rollout, bitwise;
- `render_demo_episodes.py` on 2 Goal episodes: tokens, sidecar and checks equal to the stored `demo_renders` frames;
- `collect_interactions.py demo-branch` on one Goal shard slice: equal;
- π-mean closed loop, task 0, inits 0–4: results equal to `v9_mean_nopsi_s0`.

### Rendered frames

- **Store** `outputs/transitions/demo_renders_libero90` (`TransitionStore`, format unchanged):
  - shards `<split>_<benchmark task name>`; every frame of every kept episode restored by `DemoStateRestorer` at 512², both views, frozen DINOv2 pooled tokens [2, 64, 768] float16, raw proprio;
  - demo transitions (t, t + 10);
  - sidecar `scene.npz` as for Goal.
- **Checks**, as `docs/demo_renders.md` minus the video-frame cosine (no video):
  - restore proprio against the parquet;
  - SceneStateReader against SceneProbe;
  - raw against replay object positions;
  - goal at the last frame.
- **Rendered feature cache** `outputs/feature_cache/libero_90_rendered`, same layout as `FeatureCache`: the store's tokens ordered by episode id, render proprio, SigLIP tokens of the instruction strings. Action cache `outputs/action_cache/libero_90`.

| Item | Estimate |
|---|---|
| frames | about 575k |
| process time | 0.10 s per frame (Goal, replay included), about 16 process-hours |
| wall time | 1–1.5 h with 5 processes on each of 4 GPUs |
| GPU·h | 4–6 |
| disk | store about 113 GB; feature cache about 113 GB (a second copy of the tokens) |

### Measured demonstration rendering

The opt-in renderer preserves the LIBERO-Goal path as a separate execution path. A two-episode Goal run on
node-61 produced 276 frames and 256 transitions that were array-exact against the pre-change run for visual tokens,
raw proprio, scene sidecars, transition payloads and every non-timing check field. The run took 34.12 seconds wall
time, used 30.68 user plus 5.49 system CPU-seconds and peaked at 3,188,788 KiB RSS.

The full LIBERO-90 render ran from 2026-09-30 06:53:56 to 07:55:43 EDT with five processes on each of node-61
GPU 0/1 and node-63 GPU 0/1. Evaluation task `i` was assigned to worker `i mod 20`. The 20 workers used 56,126.56
user plus 1,606.05 system CPU-seconds; per-worker peak RSS was 3,852,696–4,918,428 KiB. The four GPU assignments
used 3.569 GPU·h. Ten-second monitoring measured peak device memory of 8,197 and 9,225 MiB on node-61 and 8,523 and
7,899 MiB on node-63. Every node used exactly two GPUs.

The completed store contains 178 shards, 3,949 episodes, 572,988 frames and 533,498 `(t, t + 10)` transitions.
It represents 89 benchmark task identities with retained demonstrations, all 74 instructions and all 20 scenes.
`meta/evaluation_tasks.jsonl` continues to represent all 90 task identities; the 0/50
`LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` task correctly has no rendered shard. All 572,988
global cache rows occur exactly once, every episode contains every source frame in order, and all scene sidecars are
loadable. Fifty-two shards use the explicit zero-fixture shape. No frame was refused. SceneProbe versus
SceneStateReader object and fixture-joint maximum errors are both zero.

Replay proprio against the regenerated parquet has p99 maximum-per-frame error `4.66e-10` and overall maximum
`1.181364e-3`. The maximum is episode 541, frame 175 of
`KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it`; it is a measured replay divergence, while
the scene readers agree exactly. The transition store occupies 113,468,119,033 bytes and its checks occupy
181,335,248 bytes. The aggregate SHA-256 over the ordered shard manifests is
`41fc10fa9a69442f1fae66202707ff60f2599c4627de149e3a354a9b68448cb6`; the aggregate SHA-256 over the ordered checks
files is `a1f57b127aeceb593e50c5b39877099df0ec3135b734114a0f885ad875bcba39`.

### Measured feature and action caches

`outputs/feature_cache/libero_90_rendered` contains 572,988 frames from 3,949 episodes and 370 text strings. A
deterministic 4,096-row sample was array-exact against the rendered store for visual tokens; de-standardized cache
proprio differed from render proprio by at most `2.384186e-7`. The cache occupies 112,738,523,990 bytes. Its main
arrays are:

| File | Shape and dtype | SHA-256 |
|---|---|---|
| `visual_tokens.npy` | `[572988, 2, 64, 768]` float16 | `a26ca0a66c221b04c694667a7e2af4389c2ac00ab774d8191abe97080df136a8` |
| `proprio.npy` | `[572988, 9]` float32 | `c70cb58ecd99f3895de718b7a83fb36acae1fa8e1ab21844bc83737a18937096` |
| `frame_table.npz` | 572,988 frame rows and 3,949 episode rows, int64 | `df0973443c00814ed2a53bbfac7096db8de62d57dc0d3ee7e4d1abe2d866f223` |
| `text_tokens.npy` | `[370, 64, 1152]` float16 | `7e478c7b3d850aead18d9b6e1f87afc84648c6a1a89fff2652744cdbb3da4f6e` |
| `text_mask.npy` | `[370, 64]` bool | `35703c5ee5aceaa3f930341802fd2af285946a7b84a1648f07c1b52d79c88f66` |

The feature builder ran on node-65 GPU 0 for 46:24.22 wall time, 103.24 user plus 2,208.71 system CPU-seconds,
and peaked at 203,981,768 KiB RSS (194.5 GiB). Monitoring measured 4,213 MiB peak device memory. An intermediate
snapshot observed about 164 GiB RSS with about 372 GiB still available. The later cross-shard reorder sustained only
10–20 MiB/s. The cache completed without memory pressure, but this reorder and memory footprint are the observed
construction bottleneck recorded in ISSUE-031.

`outputs/action_cache/libero_90` contains `actions.npy` `[572988, 7]` float32 with SHA-256
`af052e91581af924b82fca438cf465d5130865eaf2520502d0bcdabf66acb3c3` and
`frames_to_episode_end.npy` `[572988]` int64 with SHA-256
`e589195b39827a2c0034911e5d0de3f63f33c676093485761f9623c4b6a8e52d`. It occupies 20,628,505 bytes and took
11.63 seconds wall, 8.25 user plus 1.51 system CPU-seconds, with 566,004 KiB peak RSS.

### R6 on LIBERO-90

The recipe is `docs/query_match.md` "R6 on all data": frozen DINOv2 pooled features → fixed PCA-256 → effect head, L_pred per dim, 20,000 updates, batch 256, seed 0.

**Stores.** The six phase-26 re-simulations of Goal replay decisions of historical ψ π versions; they have no LIBERO-90 counterpart. Their roles, π decision states and candidate branches there, are covered by the π rollout and π-noise stores. Per-episode sampling is reduced so that the TRAIN stores fit in memory (node RAM 503 GB; about 150 GB of tokens).

| Store | Goal counterpart | Recipe | Transitions (est.) | Frames (est.) | Needs π |
|---|---|---|---|---|---|
| `demo_renders_libero90` | `demo_renders` | all demo (t, t + 10) | 530k | 575k | no |
| `demo_perturbations_libero90` | data A (`demo_perturbations`) | 4 TRAIN states per episode × (stay + 4 perturbed, σ ∈ {0.5, 1, 2}) | 71k | 85k | no |
| `random_demo_branch_libero90` | `random_demo_branch_train` | 2 TRAIN states per episode × 8 smooth random branches + a 4-chunk walk | 85k | 93k | no |
| `random_init_walk_libero90` | `random_init_walk` | 5 sampled starts per task × 20-chunk walk | 9k | 9.5k | no |
| `counterfactual_eval_heldout_libero90` | `counterfactual_eval_heldout` | 6 HELDOUT states per benchmark task × 32 smooth random chunks; evaluation only | 17k | 18k | no |
| `pi_rollouts_libero90` | `rollouts_r0`, resim mean | π-mean episodes from 10 sampled starts per task; starts 8–9 held out | 13k | 14k | yes |
| `pi_noise_libero90` | `pi_noise_nopsi`, resim sweep | 20 sampled starts per task × 30 decisions, K = 8 (mean + 7 noisy), starts 16–19 held out | 54k | 56k | yes |

The smooth random chunks use the `SmoothActionSampler` settings of Goal (J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch p = 0.3).

### Measured query-independent stores

`demo_perturbations_libero90` contains 20 shards, all 3,552 TRAIN episodes, 85,248 frames and 71,040
transitions: 14,208 `stay` and 56,832 `perturbed`. It samples four states per episode: 14,208 of the 479,578
valid `T - 10` states in total, or 2.9626%. There were no refusals. Replay proprio error was at most `2.384186e-7`, the
minimum rendered-versus-cache token cosine was 0.99999535, and 14,154/14,208 repeated restores were array-exact.
The store uses float16 `[N, 2, 64, 768]` visual tokens, float32 `[N, 9]` proprio and float32 `[M, 10, 7]`
actions. Its 80 files occupy 16,805,933,572 bytes and have aggregate tree SHA-256
`447daa3e196d2bf95e922a9760c36b2130f4813c3eee6fa11effab1e5d4023dd`.

The perturbation run used node-61 GPU 0/1 and node-63 GPU 0/1 from 2026-09-30 08:58:40 to approximately 09:28:31
EDT. The 20 workers used 31,776.61 user plus 2,291.53 system CPU-seconds, peaked at 5,066,112–5,265,356 KiB RSS,
and used 1.933 GPU·h by per-card completion time. Device-memory monitoring was not active for this run.

`random_demo_branch_libero90` contains one topology-stable shard for each of the 89 evaluation tasks with retained
demonstrations. All 3,552 TRAIN episodes contribute two states: 7,104 of 515,098 TRAIN frames, or 1.3792%. Eight
independent branches and one four-chunk walk per state produced 92,352 frames and 85,248 transitions, comprising
56,832 `random_branch` and 28,416 `random_walk`. The 0/50 task has sampling rate zero and remains represented in the
90-task metadata. There were no refusals; 43 shards have explicit zero-fixture shapes. Replay proprio error was at
most `1.557765e-7`, minimum token cosine was 0.99999541, and 63,777/63,936 repeated restores were array-exact. The
534 files occupy 18,306,919,647 bytes and have aggregate tree SHA-256
`791559343d148627f9f0eb0e77724a6870c35410d3fe63f70bcbcac87f61dacb`.

The clean random-branch run used the same four cards from 2026-09-30 10:34:56 to approximately 11:06:30 EDT. The
workers used 31,399.91 user plus 1,333.93 system CPU-seconds, peaked at 3,765,936–4,913,468 KiB RSS, and used 1.980
GPU·h. Monitoring measured peak device memory of 8,593 and 9,037 MiB on node-61 and 8,359 and 7,985 MiB on node-63.

`random_init_walk_libero90` contains one shard for each of all 90 evaluation task identities, including the 0/50
demonstration task. Five sampled starts per task and twenty chunks per start produced 9,450 frames and 9,000
`init_walk` transitions with no refusals; 44 shards have explicit zero-fixture shapes. Array shapes and dtypes match
the perturbation store. Its 540 files occupy 1,874,672,957 bytes and have aggregate tree SHA-256
`19f0a17d417d9f2c3aa97f5bc553306d20217ab0d753fbf05be34ed10d68f680`.

The init-random run used the same four cards from 2026-09-30 09:32:19 to approximately 09:34:23 EDT. The workers
used 1,909.68 user plus 152.59 system CPU-seconds, peaked at 3,161,084–3,812,836 KiB RSS, and used 0.131 GPU·h.
Device-memory monitoring was not active for this run.

`counterfactual_eval_heldout_libero90` selects one HELDOUT episode for each of the 89 evaluation tasks with retained
demonstrations: 89/397 = 22.4181% of HELDOUT episodes. Six states per selected episode are 534/13,067 = 4.0866% of
the selected episodes' frames and 534/57,890 = 0.9224% of all HELDOUT frames. Thirty-two independent branches per
state produced 17,622 frames and 17,088 `random_branch` transitions. The 0/50 task remains in evaluation metadata
with sampling rate zero. There were no refusals; 43 shards have explicit zero-fixture shapes. Replay proprio error
was at most `1.349766e-7`, minimum token cosine was 0.99999589, and 17,026/17,088 repeated restores were array-exact.
The 534 files occupy 3,494,696,934 bytes and have aggregate tree SHA-256
`ed6b38ade2f7ef3eda7b195ab5e15101d970cfcb55b6f03b281a996597df7487`.

The held-out run used the same four cards from 2026-09-30 11:09:33 to approximately 11:13:58 EDT. The workers used
4,337.39 user plus 182.17 system CPU-seconds, peaked at 3,834,996–4,837,476 KiB RSS, and used 0.271 GPU·h.
Monitoring measured peak device memory of 8,593 and 9,081 MiB on node-61 and 8,359 and 7,985 MiB on node-63.

The four complete query-independent stores use 4.315 physical GPU·h and 40,482,223,110 file bytes. Together with
the demonstration render and feature-cache build, the primary full Phase 6 artifacts used 8.658 physical GPU·h,
excluding short smoke and diagnostic runs. The invalid cross-task run and duplicate aborted restart interval used an
additional approximately 2.64 physical GPU·h; ISSUE-032 records those failures and the task-sharding correction.

**PCA (user decision).** The proposal is to refit the PCA on the LIBERO-90 TRAIN frames of these stores; `fit_feature_pca.py` is unchanged.

- R6_all's PCA was fitted on one scene. LIBERO-90 has 20 scenes and objects absent from Goal, so its leading directions likely miss LIBERO-90's variation (hypothesis).
- Evidence before the fit: the explained variance of the LIBERO-90 TRAIN frames under R6_all's PCA and under the refit, at k = 256. This is a CPU check.
- Reusing R6_all's PCA and effect head unchanged is reported as a transfer reference: held-out prediction on LIBERO-90.

**Effect head.** The config is `frozen_pca_delta_all_stores.json` with the LIBERO-90 stores (held-out rules as in the table), the refitted PCA, and LIBERO-90 proprio and action statistics.

- The recipe's 20,000 updates are about 7 passes over about 720k TRAIN transitions; the Goal run made about 25.
- The alternative is 70,000 updates, matching the Goal run's passes (user decision).

**Distinguishability check.** The phase-13/19 check:

- smooth random candidates on `counterfactual_eval_heldout_libero90` against G1: within-state variance share ≥ 0.05, effective rank ≥ 4, pairwise cosine mean ≤ 0.99;
- π-noise candidates at HELDOUT π states, reported only;
- within-state R², centered cosine and Spearman against the true effects;
- held-out prediction per store;
- a per-scene breakdown of all of these.

### No-ψ π baseline

- **Config**: `goal_proposal_nopsi_seed{0,1,2}.json` with the data section set to the rendered feature cache, the LIBERO-90 dataset root and the instruction table; heldout fraction 0.1, split seed 0. Architecture, optimizer, batch, bf16 and clipping are unchanged.
- **Updates (user decision)**: the recipe's 3,000 updates are about 1.5 passes over about 520k TRAIN frames; the Goal π made about 16. The proposal is 32,000 updates (16 passes; warmup 1,600, i.e. 5%), with checkpoints every 4,000 and HELDOUT MSE per checkpoint. About 1 h per seed at the measured 8.4–9.0 updates/s. Memory: about 115 GB RAM per process with the in-memory cache, so at most 2 seeds per node.
- **Closed loop**: π-mean, K = 1.
  - 90 benchmark tasks × benchmark inits 0–49, env seed 0, 10 settle steps;
  - max 400 steps, the LIBERO-90 horizon of the OpenVLA evaluation; it is checked against the longest regenerated demo;
  - 3 π seeds, 4,500 episodes each; the binomial SD of one run is ≤ 0.0075;
  - reported per seed, per scene and per task, with version-against-seed variance as in `docs/no_psi_baseline.md`.
- **GPU·h**: training 3–4; closed loop about 13,500 episodes, estimated at 8 s mean per episode, so about 30 process-hours, i.e. about 2 h wall on 4 GPUs and 8 GPU·h.

### Order, dependencies and cost

| Step | Depends on | Resource | Wall (est.) | GPU·h (est.) | Disk (est.) |
|---|---|---|---|---|---|
| S0 raw download | — | CPU, network | 15 min | 0 | 66.7 GB |
| S1 conversion and checks (a), (b) | S0 | CPU, 90 processes | 0.5 h | 0 | < 1 GB |
| S2 opt-in code changes, bit-identity on Goal, SceneProbe/Reader on 20 scenes, task relations | S1; approval of the `LiberoTaskEnv` change | 1 GPU for short checks | engineering, plus 0.5 h runs | 0.5 | small |
| S3 demo render and check (c) | S2 | 4 GPUs | 1–1.5 h | 4–6 | 113 GB |
| S4 rendered feature cache, action cache, instruction table | S3 | CPU, 1 GPU for minutes | 0.5 h | 0.1 | 113 GB |
| S5 π training, 3 seeds | S4 | 2–3 GPUs | 1.5 h | 3–4 | small |
| S6 π-independent stores | S3 | 4 GPUs | 1–1.5 h | 5–6 | 40 GB |
| S7 π closed loop, 3 seeds | S5 | 4 GPUs | 2 h | 8 | a few GB |
| S8 π rollout and π-noise stores | S5 | 4 GPUs | 0.5 h | 2 | 14 GB |
| S9 PCA (and the R6_all transfer variance) | S6, S8 | CPU | 0.5 h | 0 | small |
| S10 effect head | S9 | 1 GPU, about 160 GB RAM | 1.5 h (20k) / 4 h (70k) | 1.5–4 | small |
| S11 distinguishability and held-out prediction | S10 | 1 GPU | 0.3 h | 0.3 | small |

- **Parallel**:
  - S0–S1 run now;
  - S5 and S6 share the 4 GPUs;
  - S7 and S8 share the 4 GPUs;
  - S9 is CPU and can start when S8 ends.
- **Totals (estimate)**: about 25–32 GPU·h on node-61/node-65 CUDA 0/1, about 10–12 h wall after the plan is confirmed, and about 350 GB of disk (3.2 TB free).
- **Current GPU state (2026-09-29 14:59 EDT)**: every card on node-61 and node-65 holds about 38 GB of other processes at 36–78% utilization. `gpu_guard.sh` admits a card only when all its processes are the project's.

### Decisions needed

1. The opt-in `benchmark_task_name` argument of `LiberoTaskEnv`, a protected module.
2. Task identity per instruction (74, LeRobot `task_index`) with the scene as an episode attribute.
3. The PCA: refit on LIBERO-90 (proposed) or reuse R6_all's.
4. Training length: π 32,000 updates (proposed) against the recipe's 3,000; effect head 20,000 (recipe, proposed) against 70,000.
5. The reduced per-episode sampling of the stores in the R6 table.
6. The closed-loop protocol: 90 × 50 inits, 400 steps, 3 seeds.

# Interaction data: inventory of the existing real transitions

Joint ψ / Δ training (`docs/interaction_training.md`) uses real transitions (o, a, o′) with simulator-rendered frames. This page records which such transitions exist and how far their actions, states and effects reach beyond the demonstrations. It also records the outcome of the pre-registered rule that decides whether supplementary collection is needed (plan.md, phase 3). The sections from "Phase 6 collection (part 1)" on record the stores collected afterwards and the coverage with them added.

Code: `scripts/inventory_transitions.py`, with three subcommands:

- `pool` (CPU): the pooled frozen feature of every frame of the rendered stores (`--stores`);
- `resim` (one simulator process per task): restores every Phase 26 decision and re-simulates a sample of data A, for scene labels and frames;
- `report`: the statistics, `report.md`, `coverage.npz` and the figures (`--collected` adds collected stores, section "Coverage after collection").

## Sources and splits

Only the training part of each source enters the statistics.

| Source | Training part | Transitions | Distinct frames | Episodes | Kinds |
|---|---|---|---|---|---|
| `demo_renders` (`docs/demo_renders.md`) | shards `train_*` | 43,701 | 47,601 | 390 | demo |
| `demo_perturbations`, data A (`docs/dynamics_data.md`) | all (TRAIN demonstration states only) | 31,200 | 37,440 | 390 | stay 6,240; σ 0.5 8,333; σ 1 8,232; σ 2 8,395 |
| `rollouts_r0` | sampled starts 0–79 | 18,732 | 19,532 | 800 | rollout 15,875; rollout_stay 2,857 |
| Phase 26 records (`docs/branch_success_test.md`) | benchmark inits 0–39 | 15,234 | 16,434 | 1,200 | fixed 5,624; flow 4,801; flow_full_sgd 4,809 |

Held-out and evaluation-only parts are listed and left out of every statistic:

| Source | Part | Transitions | Frames |
|---|---|---|---|
| `demo_renders` | shards `heldout_*` | 4,864 | 5,294 |
| `rollouts_r0` | sampled starts 80–99 | 4,470 | 4,670 |
| Phase 26 records | benchmark inits 40–49 | 3,615 | 3,915 |
| `phase9_chunks` | evaluation only, 129 HELDOUT states (demo 516, match 506, match_stay 10, mean 516, retrieval 504, retrieval_stay 12) | 2,064 | 2,580 |
| `consistency_candidates` | evaluation only, 129 HELDOUT states (demo 129, perturbed 1,548, proposal 4,128, stay 129) | 5,934 | 6,063 |

The splits are those of `HeldoutRule` (`docs/interaction_training.md`): demo_renders by shard prefix `heldout`, rollouts_r0 by `episode` 80–99, data A has no held-out part. The Phase 26 split is by benchmark init.

**Phase 26 transitions.** A record row (`record/task<NN>/decisions.npz`, the arrays of `branch_success_test.DecisionTable`) holds one decision d of a π-mean episode: the SimSnapshot one env step before the decision's observation, the producing sim action, and π's mean chunk c_d. Rows d and d + 1 of one episode form the transition (o_d, c_d, o_{d+1}) when c_d ran all 10 steps. This holds for every row except the last of each episode: 20,349 rows in 1,500 episodes give 18,849 transitions (all three records, all inits). The records store no tokens, so `resim` restores every row with `LiberoTaskEnv.restore(snapshot, producing action)` to render and encode o_d.

## Definitions

**Actions.** Chunks [10, 7] standardized with `ActionCache.standardize`.

- Effective rank of the flattened chunks [70]: exp(entropy of the normalized covariance eigenvalues), the definition of `StateDistributionMetrics`.
- DCT: orthonormal DCT-II over the 10 steps of each arm dim; k = 0 is the chunk mean.
- Gripper: open when the command is above 0.5. In-chunk switches are counted over the 9 step pairs of a chunk. A boundary switch is a first step that differs from the command in force before the chunk:
  - demo and data A: the dataset action a_{t−1};
  - rollouts_r0 and Phase 26: the last step of the previous decision's chunk;
  - at t = 0 or decision 1: the settle command (open).
- Distance to the nearest TRAIN demo chunk: Euclidean, on the flattened standardized chunk, against the 43,701 TRAIN demo chunks. The demo's own distance is taken to the nearest TRAIN demo chunk of another episode. Its 95th percentile is the threshold of the criterion. The HELDOUT demo chunks, measured against all TRAIN chunks, check this choice (see the action table). Leaving out only the chunk itself is reported as a sensitivity: 27.8% of the HELDOUT demo chunks lie beyond that threshold.

**States.** The pooled frozen DINOv2 feature of a frame: per view, the float32 mean of its 64 pooled tokens; the two views concatenated (1,536-d, R0 of the interaction monitor).

- The distance of a frame to the nearest of the 47,601 TRAIN demo frames (every frame of the TRAIN shards).
- The demo's own distance leaves the frame's episode out.
- The 2-D PCA is fitted on the TRAIN demo frames.

**Effects**, per transition:

- object moved: some object's position changes by more than 1 cm;
- fixture changed: a drawer slide changes by more than 1 cm, or the stove knob by more than 0.05 rad;
- grasp changed: some object's grasp flag changes (`SceneProbe`: both finger pads in contact).

Label sources:

| Source | Scene before / after | Positions |
|---|---|---|
| demo_renders | sidecar `SceneState` of frames t and t + 10 | body `xpos` |
| rollouts_r0 | trace `SceneSnapshot` of decision t and of decision t + 1 (the episode's `final` scene after its last decision), joined on (task, `episode`, `t`) | body position (equal to `xpos`) |
| Phase 26 | objects and fixtures: the recorded `sim_state` of rows d and d + 1, decoded with `SceneStateReader.object_poses_from_sim_state` and the fixture joints' qpos addresses from the MuJoCo model; grasp: the `SceneState` read after restoring rows d and d + 1 | free-joint qpos, one env step before each decision |
| data A | re-simulated sample, `SceneState` after `restore(t)` and after `rollout(chunk)` | body `xpos` |

**Data A sample.** 2,000 of the 31,200 transitions, drawn without replacement with `numpy.random.default_rng(0)` in store order; 388 episodes. Per episode `DemoStateRestorer` replays the dataset episode (settle 10, env seed 0). Per transition: `restore(t)`, read the scene; `rollout(stored chunk)`, read the scene. The data A shares are estimates from this sample and are given with Wilson 95% intervals. A restore reproduces the observation of frame t; the re-simulation checks below measure how closely the re-simulated o′ reproduces the stored one.

## Action coverage

| Source | Chunks | Effective rank [70] | Top eigenvalue share | In-chunk step pairs switching | Chunks with an in-chunk switch | Boundary switch |
|---|---|---|---|---|---|---|
| demo TRAIN | 43,701 | 8.51 | 0.275 | 1.3% | 12.1% | 1.1% |
| data A | 31,200 | 8.08 | 0.203 | 1.3% | 11.3% | 1.0% |
| rollouts_r0 | 18,732 | 9.13 | 0.258 | 1.9% | 17.2% | 23.2% |
| Phase 26 | 15,234 | 7.85 | 0.306 | 1.5% | 12.4% | 5.0% |
| non-demo pooled | 65,166 | 8.58 | 0.207 | 1.5% | 13.2% | 8.3% |

The rollouts_r0 boundary rate is the same from the trace's `gripper_commands` (5,378 of 23,202, 23.2%), which equal the stored gripper column in every transition.

DCT energy, share of the summed arm-dim energy per frequency (k ≥ 5 below 0.1% everywhere). The energy column is the mean over chunks and arm dims of the sum of the 10 squared standardized actions:

| Source | Energy per arm dim and chunk | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|
| demo TRAIN | 10.39 | 90.4% | 8.1% | 1.1% | 0.3% | 0.1% |
| data A | 18.74 | 96.0% | 3.3% | 0.5% | 0.1% | 0.0% |
| rollouts_r0 | 8.06 | 88.8% | 9.4% | 1.3% | 0.3% | 0.1% |
| Phase 26 | 8.08 | 91.0% | 8.0% | 0.8% | 0.2% | 0.0% |

Share of each arm dim's energy at k ≥ 1:

| Source | x | y | z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| demo TRAIN | 11.6% | 8.2% | 11.3% | 8.9% | 12.8% | 4.7% |
| data A | 5.5% | 3.6% | 6.1% | 3.3% | 4.8% | 1.7% |
| rollouts_r0 | 13.5% | 11.9% | 14.1% | 11.5% | 11.4% | 4.9% |
| Phase 26 | 12.0% | 7.3% | 11.5% | 6.5% | 11.7% | 5.0% |

Distance to the nearest TRAIN demo chunk. The demo's own 95th percentile is **3.549** with the episode left out, and 2.271 with only the chunk left out.

| Chunks | n | p5 | Median | p95 | > 3.549 | > 2.271 |
|---|---|---|---|---|---|---|
| demo TRAIN, other episodes | 43,701 | 0.872 | 1.824 | 3.549 | 5.0% | 28.4% |
| demo TRAIN, other chunks | 43,701 | 0.287 | 1.015 | 2.271 | 0.0% | 5.0% |
| demo HELDOUT → TRAIN (check) | 4,864 | 0.846 | 1.780 | 3.536 | 4.9% | 27.8% |
| data A | 31,200 | 0.896 | 3.721 | 10.814 | 52.5% | 72.6% |
| data A, stay | 6,240 | 0.896 | 0.990 | 0.990 | 0.0% | 0.0% |
| data A, σ 0.5 | 8,333 | 1.642 | 2.820 | 4.416 | 21.7% | 75.3% |
| data A, σ 1 | 8,232 | 2.517 | 4.634 | 7.687 | 77.7% | 97.2% |
| data A, σ 2 | 8,395 | 4.124 | 8.057 | 13.577 | 97.4% | 99.8% |
| rollouts_r0, rollout | 15,875 | 0 | 0 | 0 | 0.0% | 0.0% |
| rollouts_r0, rollout_stay | 2,857 | 0.896 | 0.990 | 0.990 | 0.0% | 0.0% |
| Phase 26 | 15,234 | 0.830 | 1.531 | 2.797 | 1.5% | 13.2% |
| non-demo pooled | 65,166 | 0 | 1.508 | 9.005 | 25.5% | 37.8% |

- The retrieved rollouts_r0 chunks are TRAIN demo chunks: distance 0 up to the 95th percentile.
- The three Phase 26 records are close: p95 2.871 (fixed), 2.759 (flow), 2.738 (flow_full_sgd); 1.5–1.7% beyond 3.549.
- 98.6% of the pooled chunks beyond 3.549 are data A chunks. Their arm energy is 96% in the chunk mean (k = 0), so they are constant offsets of demo chunks.

Figures: `chunk_nn_distance.png`, `dct_spectrum.png`.

## State coverage

Distance to the nearest TRAIN demo frame. The demo's own 95th percentile, with the episode left out, is **11.880**.

| Frames | n | p5 | Median | p95 | > 11.880 |
|---|---|---|---|---|---|
| demo TRAIN, other episodes | 47,601 | 5.443 | 8.448 | 11.880 | 5.0% |
| demo HELDOUT → TRAIN (check) | 5,294 | 5.421 | 8.487 | 11.726 | 4.4% |
| data A, o | 6,240 | 0 | 0 | 0 | 0.0% |
| data A, o′ | 31,200 | 4.815 | 9.258 | 14.960 | 21.8% |
| data A, o′, own episode left out | 31,200 | 6.757 | 10.091 | 15.128 | 25.5% |
| rollouts_r0 | 19,532 | 5.603 | 12.942 | 21.784 | 56.0% |
| Phase 26 | 16,434 | 5.285 | 8.233 | 14.863 | 13.9% |
| non-demo pooled (data A o′, rollouts_r0, Phase 26) | 67,166 | 5.158 | 9.466 | 18.829 | 29.8% |

- Data A's o frames are restores of TRAIN demo frames. Their largest distance to the nearest demo render is 0.022.
- The re-simulated data A o′ lies at median 0 (max 0.078) from the stored o′ in this space.

2-D PCA of the TRAIN demo frames: PC 1 17.1%, PC 2 9.4% of their variance (`state_pca.png`). In these two components every source spans the demo range:

| Frames | PC 1 std | PC 2 std | PC 1, p1–p99 | PC 2, p1–p99 |
|---|---|---|---|---|
| demo TRAIN | 6.89 | 5.12 | −9.88 – 14.84 | −10.38 – 8.11 |
| data A, o′ | 6.55 | 4.74 | −10.36 – 14.42 | −10.53 – 7.67 |
| rollouts_r0 | 6.62 | 5.51 | −10.58 – 14.21 | −11.93 – 7.91 |
| Phase 26 | 6.86 | 5.04 | −10.19 – 14.61 | −10.21 – 8.01 |

Figures: `state_nn_distance.png`, `state_pca.png`.

## Effect coverage

| Source | Labelled transitions | Object moved | Fixture changed | Grasp changed | Any of the three |
|---|---|---|---|---|---|
| demo TRAIN | 43,701 | 40.2% | 7.4% | 12.3% | 49.4% |
| data A (sample) | 2,000 of 31,200 | 37.7% [35.6%, 39.8%] | 7.0% [6.0%, 8.2%] | 12.1% [10.7%, 13.6%] | 46.7% [44.5%, 48.8%] |
| rollouts_r0 | 18,732 | 15.7% | 2.3% | 2.6% | 18.3% |
| Phase 26 | 15,234 | 29.3% | 5.0% | 6.2% | 35.3% |

Object moved, per object:

| Source | bowl | cream cheese | wine bottle | plate |
|---|---|---|---|---|
| demo TRAIN | 19.1% | 5.5% | 10.6% | 6.6% |
| data A | 21.2% | 5.0% | 10.4% | 3.4% |
| rollouts_r0 | 7.9% | 4.0% | 1.3% | 5.1% |
| Phase 26 | 13.9% | 4.2% | 6.8% | 5.7% |

By kind. "Carried" is the share of the object-moving transitions with an object grasped at o; "displacement" is their median largest object displacement.

| Source | Kind | Labelled | Object moved | Carried | Displacement | Fixture changed | Grasp changed |
|---|---|---|---|---|---|---|---|
| data A | stay | 403 | 24.1% | 90.7% | 1.7 cm | 4.2% | 7.9% |
| data A | σ 0.5 | 543 | 42.2% | 66.4% | 7.8 cm | 7.4% | 14.5% |
| data A | σ 1 | 549 | 39.5% | 66.4% | 7.8 cm | 7.7% | 13.5% |
| data A | σ 2 | 505 | 41.8% | 54.5% | 9.5 cm | 8.1% | 11.3% |
| rollouts_r0 | rollout | 15,875 | 17.8% | 20.2% | 3.3 cm | 2.7% | 3.1% |
| rollouts_r0 | rollout_stay | 2,857 | 4.0% | 17.5% | 1.6 cm | 0.1% | 0.3% |
| Phase 26 | fixed | 5,624 | 28.0% | 52.0% | 7.5 cm | 4.3% | 5.9% |
| Phase 26 | flow | 4,801 | 30.5% | 54.8% | 7.7 cm | 5.9% | 6.6% |
| Phase 26 | flow_full_sgd | 4,809 | 29.8% | 53.9% | 7.6 cm | 4.9% | 6.1% |

### Re-simulation checks

**Data A sample** (2,000 transitions, 388 episodes, 0 refused):

- restored o: proprio identical to the stored o in all 2,000;
- re-simulated o′ against the stored o′:
  - proprio identical in 1,998, max |Δ| 5.96e-7;
  - token cosine 1.0000 at the minimum in both views;
  - success flag equal to the stored `goal_after` in 2,000 of 2,000.

On these 10-step chunks from demonstration states, the re-simulation reproduces the stored o′.

**Phase 26 restores** (all 20,349 rows, 0 refused):

- restored proprio identical to the recorded decision in 19,975 rows, max |Δ| 9.9e-5;
- decode of the restored `sim_state` against the `SceneState` read from it:
  - fixtures: 0;
  - objects: max 2.6 mm, the free-joint qpos against the body `xpos` (`docs/demo_renders.md`);
- on the training transitions, the decoded labels agree with the labels from the restored states in 98.8% (object moved) and 99.6% (fixture changed). The decoded snapshots lie one env step before each decision.

## Pre-registered decision

Plan.md, phase 3: supplementary collection is needed if either criterion holds over the non-demo training transitions. There are 65,166 of them: data A 31,200, rollouts_r0 18,732, Phase 26 15,234.

| Criterion | Measured | Holds when | Holds |
|---|---|---|---|
| chunks beyond the demo's own p95 from the nearest demo chunk (episode left out, 3.549) | 25.5% | < 20% | no |
| the same with the chunk-out p95 (2.271), sensitivity | 37.8% | < 20% | no |
| transitions moving an object (data A share from the sample) | 29.4% [28.4%, 30.4%] | < 10% | no |

**Outcome: neither criterion holds, so the pre-registered rule does not call for supplementary collection.**

The same shares without data A (rollouts_r0 and Phase 26, 33,966 transitions):

- chunks beyond 3.549: 0.7%;
- transitions moving an object: 21.8%.

The chunk criterion is met only through data A's constant offsets. Compared with the demonstrations, no existing source has:

- more in-chunk shape: arm energy at k ≥ 1 of 4.0% (data A) to 11.2% (rollouts_r0), against 9.6% for the demo;
- more in-chunk gripper switching: 1.3–1.9% of step pairs, against 1.3% for the demo.

## Runs

| Step | Host / device | Code | Wall time | Log |
|---|---|---|---|---|
| `pool` | node-65, CPU (4 threads) | `caae3f8` | 54 s (25 + 18 + 12 s) | `tmp/runlogs/s4p3/pool.log` |
| `resim`, tasks 0–4 | node-65 CUDA 1, 5 processes | `caae3f8` | 330–416 s per process | `tmp/runlogs/s4p3/resim_task0<N>.log` |
| `resim`, tasks 5–9 | node-65 CUDA 2, 5 processes | `caae3f8` | 284–383 s per process | same |
| `report` | node-65 CUDA 1 | `3ab8cb8` | 31 s | `tmp/runlogs/s4p3/report.log` |

- Launchers: `tmp/s4p3/launch_pool.sh`, `tmp/s4p3/launch_resim.sh`, `tmp/s4p3/run_report.sh`, with 2 threads per simulator process.
- Both GPUs were idle before launch (`tmp/runlogs/s4p3/resim_nvidia_smi.txt`).
- `resim` took 3,420 s in total over the 10 processes. Per Phase 26 row, the median restore took 35 ms and the median encode 42 ms. Per data A episode, the replay took a median of 3.1 s (max 8.4 s; 1,324 s over the 388 episodes).
- `/tmp/robosuite.log` on node-65 was not modified; its last write predates the runs.
- `e0e8b61` only removes unused code. Its report is identical: `report.md` apart from the header, `coverage.npz` byte for byte.

```bash
source scripts/ceam_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
$CEAM_PYTHON scripts/inventory_transitions.py pool
$CEAM_PYTHON scripts/inventory_transitions.py resim --task <t> --cuda-device <gpu> --code-commit <sha>   # t = 0 … 9
$CEAM_PYTHON scripts/inventory_transitions.py report --device cuda:<gpu> --code-commit <sha>
```

Every subcommand refuses to overwrite an existing output.

## Outputs

All outputs are in `outputs/analysis/interaction_inventory/`:

| Path | Contents |
|---|---|
| `report.md` | the tables above |
| `coverage.npz` | see below |
| `chunk_nn_distance.png`, `state_nn_distance.png`, `dct_spectrum.png`, `state_pca.png` | figures |
| `pooled/<store>.npy` | [n_frames, 1536] float32 pooled feature of every frame, in store frame order, for `demo_renders`, `demo_perturbations` and `rollouts_r0` (672 MB) |
| `resim/task<NN>.npz` | per task (134 MB in all), see below |

`resim/task<NN>.npz`:

- shared: `provenance` (JSON), `object_names`, `fixture_joint_names`, `fixture_joint_kinds`;
- per Phase 26 row, in record order (fixed, flow, flow_full_sgd):
  - `p26_name`, `p26_init`, `p26_decision`, `p26_steps_before`, `p26_chunk_steps`, `p26_chunk`;
  - `p26_message` (refusal, "" when restored), `p26_proprio_error`;
  - `p26_pooled` [1536];
  - `p26_restored_{object_positions, object_grasped, fixture_qpos}`;
  - `p26_decoded_{object_positions, fixture_qpos}`;
  - the decode checks, `p26_seconds` (restore, encode, read);
- per data A sample transition:
  - `a_index` (store transition index), `a_episode`, `a_t`, `a_task_id`, `a_kind`, `a_sigma`, `a_goal_after_{stored, resim}`, `a_message`;
  - `a_proprio_error_{o, next}`, `a_token_cosine_next`, `a_pooled_distance_next`;
  - `a_{o, next}_{object_positions, object_grasped, fixture_qpos}`;
  - timings.

`coverage.npz`. Source keys are `demo_TRAIN`, `data_A`, `rollouts_r0`, `Phase_26`, and `non-demo_pooled` for the chunk statistics. Rows follow each source's training transitions in store order (record row order for Phase 26).

- `chunk_eigenvalues_<source>` [70], `dct_energy_<source>` [6, 10];
- `chunk_nn_demo_loeo`, `chunk_nn_demo_loo`, `chunk_nn_demo_heldout`, `chunk_nn_<source>`, `chunk_threshold_loeo`, `chunk_threshold_loo`, `kind_<source>`;
- `frame_nn_demo_loeo`, `frame_nn_demo_HELDOUT_to_TRAIN_check`, `frame_nn_data_A_o`, `frame_nn_data_A_o_next`, `frame_nn_data_A_o_next_own_episode_left_out`, `frame_nn_rollouts_r0`, `frame_nn_Phase_26`, `frame_threshold`;
- `pca_mean`, `pca_components` [2, 1536], `pca_explained`, `pca_<frames>` [n, 2];
- `effect_<source>_{object_displacement [n, 4], joint_change [n, 4], grasp_changed, grasped_before, rows}`. `rows` are the labelled transitions: row indices of the source's training transitions, or store transition indices for data A;
- `decision_far_share`, `decision_far_share_loo`, `decision_moved_share` (estimate, low, high), `decision_needed`, `decision_far_share_without_data_a`, `decision_moved_share_without_data_a`.

## Phase 6 collection (part 1)

Two collections with `scripts/collect_interactions.py` (`docs/interaction_collection.md`), code commit `cf2f1b3`: the Phase 26 re-simulation, a training source with benchmark inits 40–49 held out, and the HELDOUT counterfactual evaluation set.

### Stores

| Store (`outputs/transitions/`) | Command | Shards | Frames | Transitions | TRAIN / HELDOUT |
|---|---|---|---|---|---|
| `resim_phase26_mean_fixed` | `resim-branch --name fixed --mode mean`, tasks 0–9, all inits | 10 | 7,360 | 6,860 | 5,624 / 1,236 |
| `resim_phase26_mean_flow` | the same, `--name flow` | 10 | 6,507 | 6,007 | 4,801 / 1,206 |
| `resim_phase26_mean_flow_full_sgd` | the same, `--name flow_full_sgd` | 10 | 6,482 | 5,982 | 4,809 / 1,173 |
| `resim_phase26_sweep_fixed` | `resim-branch --name fixed --mode sweep`, tasks 0–9, all inits | 12 | 17,028 | 16,512 | 14,592 / 1,920 |
| `resim_phase26_sweep_flow` | the same, `--name flow` | 11 | 9,504 | 9,216 | 7,296 / 1,920 |
| `resim_phase26_sweep_flow_full_sgd` | the same, `--name flow_full_sgd` | 9 | 9,306 | 9,024 | 7,296 / 1,728 |
| `counterfactual_eval_heldout` | `demo-branch --split heldout --fractions 0.1 0.4 0.7 0.25 0.55 0.85 --branches 32` | 4 | 8,514 | 8,256 | 0 / 8,256 |
| **total** | | **66** | **64,701** | **61,857** | **44,418 / 17,439** |

TRAIN / HELDOUT is the split of `HeldoutRule` through `InteractionSampler` (`docs/interaction_training.md`): episodes (40, 49) for the six resim stores, shard prefix `heldout` for the evaluation set.

- **resim mean**: every recorded decision of the 1,500 π-mean episodes is restored and encoded, 20,349 frames. A transition is (d, d + 1) of one episode with chunk d run for all 10 steps, 18,849 in all. Its action is the recorded π-mean chunk and o′ the restored decision d + 1.
- **resim sweep**: decisions 1, 6, 11, 16, 21, 26 of the 181 failed episodes (86, 48, 47). Each of the 32 stored candidates (`StoredBranches`) runs from a fresh restore: 1,086 states (516, 288, 282) × 32 = 34,752 transitions.
  - These states include two where a Phase 26 branch pass aborted one candidate on a black render: flow task 7, init 0, decision 21, candidate 30; flow_full_sgd task 7, init 33, decision 11, candidate 3. Here every candidate rendered.
  - The table "Every candidate executed" of `docs/branch_success_test.md` leaves these two states out (287 and 281 states).
- **Evaluation set**: 258 states (43 HELDOUT episodes × 6 fractions) × 32 random chunks, without walks.
  - `SmoothActionSampler` seed 0, J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch p = 0.3, env seed 0.
  - Shards `heldout_shard_<i>_of_004`, each with every fourth HELDOUT episode.
- **resim**: env seed 0; no random draws.
  - One shard per task, `<name>_<mode>_task<TT>`.
  - Six sweep tasks are split into two init ranges (`_init<a>-<b>`): fixed tasks 3, 4, 6; flow tasks 1, 6; flow_full_sgd task 1.
- Every manifest records the commit, host, CUDA device, settings and seeds. Refusals: 0 in all 66 shards.

### Checks

Scripts: `collect_interactions.py report` per store, `tmp/s4p6/check_phase6.py` (split, records, evaluation states, effects), `tmp/s4p6/check_eval_vs_demo_renders.py`, `tmp/s4p6/locate_exceptions.py`, `tmp/s4p6/diagnose_restores.py`.

**Split.**

- The held-out transitions of each resim store are exactly those with `episode` 40–49. All inits 40–49 are present in the mean stores; the sweep stores hold those of their failed episodes. TRAIN transitions have `episode` ≤ 39.
- The mask equals the direct `episode` test.
- The three mean stores give TRAIN 15,234 and HELDOUT 3,615, the Phase 26 record split of this page.
- `InteractionSampler` on the evaluation store: 8,256 HELDOUT transitions. Its TRAIN split raises `ValueError` (no transition).

**Sidecar ↔ store.** All 66 shards:

- sidecar rows = shard frames = checks rows, `frame` = 0 … n − 1;
- |store proprio − sidecar qpos|, arm: max 9.0e-3, shard medians 8.7e-5 to 1.1e-3. With the sidecar shifted by one row, the medians are 1.2e-2 to 0.28;
- `cache_row` = `frame_rows(episode, t)` at all 258 branch points.

**Chains and records.**

- Sim time o′ − o = 0.5 s in 61,857 / 61,857 transitions (max error 1.7e-13).
- Rollout o′ with parent o and step + 1: 43,008 / 43,008 (sweep 34,752, evaluation 8,256).
- resim mean:
  - o′ is the restored decision t + 1 of the same init in 18,849 / 18,849;
  - 1,500 chain starts, one per recorded episode;
  - the stored chunk equals the record's π-mean chunk in 18,849 / 18,849.
- resim sweep: the stored chunk equals the stored candidate named by the sidecar's `candidate` in 34,752 / 34,752. Every state has candidates 0–31 once.
- Every resim transition's (task, `episode`, t) equals its o's sidecar row and names a record row (init, decision).

**Evaluation set.**

- The 258 (episode, t) states are distinct and equal `BranchPoints` of the 43 HELDOUT episodes for the six fractions.
- Those at f ∈ {0.1, 0.4, 0.7} are the 129 `consistency_candidates` states; the other 129 are disjoint from them.
- 32 transitions per state.
- J counts 2,702 / 2,760 / 2,794; s counts 2,724 / 2,717 / 2,815; gripper switch rate 0.294; `sigma` equals the sidecar's `chunk_scale`.
- Per-entry RMS median 0.483 / 0.959 / 1.74 at s = 0.5 / 1 / 2. Share of arm entries at ±1: 0.000 / 0.012 / 0.111.

**Restore fidelity.**

| Frames | Reference | Result |
|---|---|---|
| resim mean decisions (20,349) | recorded proprio | bytes identical 19,975 (7,227 / 6,383 / 6,365); max \|Δ\| 9.9e-5 (fixed 1.3e-6, flow 9.9e-5, flow_full_sgd 8.0e-6) |
| resim sweep states (1,086) | recorded proprio | bytes identical 1,074 (508 / 284 / 282); max \|Δ\| 2.4e-7 |
| evaluation branch points (258) | dataset proprio | median 1.7e-8, p95 2.3e-7, max 1.5e-3 (episode 69, t = 204); 4 others above 1e-6, at most 2.4e-5 |
| evaluation branch points (258) | `demo_renders` frame of the same (episode, t) | proprio bytes identical 258 / 258; token cosine ≥ 0.999996 |
| evaluation branch points at f ∈ {0.1, 0.4, 0.7} (129) | `consistency_candidates` frame | proprio bytes identical 129 / 129; token cosine ≥ 0.999996 |
| evaluation branch points (258) | feature-cache video tokens | cosine agent mean 0.969 (min 0.960), wrist 0.939 (min 0.859) |

- The resim mean counts equal those of the Phase 3 restores (19,975 rows, max 9.9e-5).
- The branch points' errors against the dataset are those of `demo_renders`: in episode 69 the replay leaves the dataset from t = 181 (`docs/demo_renders.md`).

**Repeated restores.** Before each chunk a fresh restore is compared with the state's first restore, on proprio and both images together.

- They are equal in:
  - 8,067 / 8,256 (evaluation);
  - 16,238 / 16,512, 9,106 / 9,216, 8,775 / 9,024 (sweep fixed, flow, flow_full_sgd).
- The 822 exceptions fall in 127 states.
  - Sweep: 515 of 633 are in task 7.
  - Evaluation: the states with the most exceptions are episode 238 (task 5), 336 (task 7) and 412 (task 9), with 32 each.

`diagnose_restores.py` repeats the collection's call sequence in a new env (node-61 CUDA 0) at 12 states: 10 with 17–32 exceptions in the collection and 2 without. Over the 384 fresh restores:

- proprio 384 / 384 and agent image 384 / 384 are identical to the first restore;
- the wrist image differs in 110, by at most 1 intensity level in at most 17 pixels;
- the sim_state after the fresh restore equals the first restore's in 353. The 31 others are in one state, fixed task 7, init 20, decision 26;
- the re-executed o′ proprio equals the stored o′ in 384 / 384. Its sim_state equals the sidecar's in 354, with max difference 1.9e-12;
- four states with 32 exceptions in the collection are identical in all 32 restores of the rerun.

The collection's checks flag does not separate proprio from images.

### Effects

Per transition, from the sidecars (thresholds as above: object 1 cm, slide 1 cm, knob 0.05 rad):

| Transitions | n | Object moved | Fixture changed | Grasp changed | Any | Goal holds after |
|---|---|---|---|---|---|---|
| resim mean, fixed | 6,860 | 29.4% | 4.6% | 6.0% | 34.7% | 0 |
| resim mean, flow | 6,007 | 30.7% | 6.1% | 6.6% | 37.2% | 0 |
| resim mean, flow_full_sgd | 5,982 | 30.8% | 5.2% | 6.3% | 36.7% | 0 |
| resim mean, TRAIN (inits 0–39) | 15,234 | 29.9% | 5.2% | 6.2% | 35.8% | 0 |
| resim mean, HELDOUT (inits 40–49) | 3,615 | 31.6% | 5.5% | 6.7% | 37.7% | 0 |
| resim sweep, fixed | 16,512 | 21.2% | 1.9% | 5.9% | 24.0% | 0.05% |
| resim sweep, flow | 9,216 | 13.2% | 9.6% | 2.7% | 21.8% | 0.01% |
| resim sweep, flow_full_sgd | 9,024 | 14.0% | 4.4% | 1.4% | 18.6% | 0 |
| resim sweep, TRAIN | 29,184 | 16.6% | 4.6% | 3.7% | 21.2% | 0.03% |
| resim sweep, HELDOUT | 5,568 | 20.6% | 4.7% | 4.8% | 26.1% | 0.02% |
| evaluation set | 8,256 | 36.5% | 4.4% | 12.1% | 41.9% | 2.6% |

- The Phase 26 TRAIN transitions of the "Effect coverage" table above: 29.3% / 5.0% / 6.2% / 35.3%. Those labels come from the decoded recorded sim_state, one env step before each decision.
- A π-mean chunk that reaches the goal ends its episode before 10 steps, so no mean transition ends at the goal.

Share of chunks beyond the HELDOUT demo chunks' p95 distance to the nearest TRAIN demo chunk (arm dims 3.39 / all dims 3.53; `docs/interaction_collection.md`):

- evaluation set: 0.666 / 0.693;
- resim mean: 0.008 / 0.018 (fixed), 0.007 / 0.014 (flow), 0.005 / 0.015 (flow_full_sgd);
- resim sweep: 0.022 / 0.048, 0.026 / 0.044, 0.009 / 0.049.

### Runs, time and disk

- **Hosts**: node-61 CUDA 0 and 1, node-65 CUDA 1 and 2, all idle at launch.
  - 5 processes per GPU, 20 in all. Each ran its commands in sequence (`tmp/s4p6/slots/slot_<NN>.txt`).
  - OMP, MKL, OPENBLAS and NUMBA threads: 2.
- **Wall time**: 2026-09-28 16:47:35 to 16:59:07 EDT, 692 s. Per process 517–692 s; 12,530 process-seconds in all.
- **Load**: peak load average 10.1 (node-61) and 10.2 (node-65). GPU utilization samples ≤ 81%, memory ≤ 7.4 GiB per GPU.
- **Per transition** (the shards' collection seconds): evaluation 0.256 s, sweep 0.240–0.247 s, mean 0.058–0.063 s.
  - Per-shard medians: restore 26–37 ms, rollout 160–194 ms, encode 12–27 ms, scene read 3.1–4.0 ms.

| Store | Disk |
|---|---|
| `resim_phase26_mean_fixed` | 1.36 GiB |
| `resim_phase26_mean_flow` | 1.20 GiB |
| `resim_phase26_mean_flow_full_sgd` | 1.20 GiB |
| `resim_phase26_sweep_fixed` | 3.15 GiB |
| `resim_phase26_sweep_flow` | 1.76 GiB |
| `resim_phase26_sweep_flow_full_sgd` | 1.72 GiB |
| `counterfactual_eval_heldout` | 1.57 GiB |
| **total** | **11.96 GiB** (193.8 KiB per frame) |

Checks files and reports: 4.6 MiB.

Against the estimate of `docs/interaction_collection.md`:

| Source | Estimated process time / disk | Measured collection seconds / disk |
|---|---|---|
| evaluation set | 1,948 s / 1.6 GiB | 2,113 s / 1.57 GiB |
| resim mean | 1,131 s / 3.8 GiB | 1,150 s / 3.76 GiB |
| resim sweep | 7,784 s / 6.6 GiB | 8,463 s / 6.62 GiB |

**Outputs**:

- `outputs/analysis/interaction_collection/<store>/`: checks files `<shard>_checks.npz`, `report.txt`;
- `outputs/analysis/interaction_collection/phase6_part1_checks.json`: split counts, record and evaluation checks, effects;
- `outputs/analysis/interaction_collection/phase6_part1_restore_diagnosis.json`: the 12-state rerun.

**Launchers**: `tmp/s4p6/launch.sh`, `run_slot.sh`, `plan_slots.py` (slot packing), `after_collect.sh`. Logs: `tmp/runlogs/s4p6/`.

```bash
source scripts/ceam_env.sh   # CEAM_ROOT = main checkout; OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=2
$CEAM_PYTHON scripts/collect_interactions.py resim-branch --cuda-device <gpu> --code-commit cf2f1b3b9a6cf2ae00b601ea1fc13e9c6717b289 \
  --out-dir outputs/transitions/resim_phase26_<mode>_<name> \
  --analysis-dir outputs/analysis/interaction_collection/resim_phase26_<mode>_<name> \
  --name <name> --mode <mean|sweep> --tasks <t …> [--inits <a … b>]
$CEAM_PYTHON scripts/collect_interactions.py demo-branch --cuda-device <gpu> --code-commit cf2f1b3b9a6cf2ae00b601ea1fc13e9c6717b289 \
  --out-dir outputs/transitions/counterfactual_eval_heldout \
  --analysis-dir outputs/analysis/interaction_collection/counterfactual_eval_heldout \
  --split heldout --fractions 0.1 0.4 0.7 0.25 0.55 0.85 --branches 32 --walk-length 0 --seed 0 --env-seed 0 \
  --dct-coefficients 1 2 3 --scales 0.5 1.0 2.0 --gripper-switch-prob 0.3 --shard <i> --num-shards 4
```

## Phase 6 collection (part 2)

The random training data of plan.md "补充采集": demonstration-state branches and walks, pure random walks from sampled starts, and noisy π rollouts. Tool: `scripts/collect_interactions.py`; code commit `e0df3d9` (the collection code is unchanged since `cf2f1b3`).

### Stores

| Store (`outputs/transitions/`) | Command | Shards | Frames | Transitions | TRAIN / HELDOUT (HeldoutRule) | Disk |
|---|---|---|---|---|---|---|
| `random_demo_branch_train` | `demo-branch --split train --states-per-episode 8 --branches 8 --walk-length 4`, 26 shards `train_shard_<i>_of_026` | 26 | 40,560 | 37,440 (`random_branch` 24,960, `random_walk` 12,480) | 37,440 / 0 | 7.50 GiB |
| `random_init_walk` | `init-random --starts 20 --walk-length 20 --env-seed 100`, one shard per task | 10 | 4,200 | 4,000 (`init_walk`) | 4,000 / 0 | 0.78 GiB |
| `pi_noise_nopsi` | `pi-noise --starts 100 --decisions 30 --env-seed 2`, one shard per task | 10 | 31,000 | 30,000 (`pi_noise`) | 24,000 / 6,000 | 5.73 GiB |
| **total** | | **46** | **75,760** | **71,440** | **65,440 / 6,000** | **14.00 GiB** |

- **Smooth random chunks** (demo-branch, init-random):
  - `SmoothActionSampler` seed 0, J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2} (per-entry RMS), gripper switch p = 0.3;
  - demo-branch: env seed 0; 8 distinct branch points per episode from `default_rng([0, 0, episode])`.
- **pi-noise**:
  - π `goal_proposal_nopsi_seed0` step 3000;
  - K = 8: π's mean chunk plus 7 noisy candidates, scales 0.5 / 1 / 2 cycled, J = 3;
  - proposal seed 0, choice seed 0.
- **Held-out rule**: pi-noise `episode` 80–99. demo-branch and init-random have no held-out part: every demo-branch state is a TRAIN demonstration state, and init-random is training data.
- Every manifest records the commit, host, CUDA device, settings and seeds. Refusals: 0 in all 46 shards.

### Checks

Scripts: `collect_interactions.py report` per store, `tmp/s4p6/check_part2.py`, `tmp/s4p6/check_branch_points_vs_demo_renders.py`, `tmp/s4p6/check_pi_noise_detail.py`.

**Split** (`InteractionSampler` with the rules above):

- TRAIN 37,440 + 4,000 + 24,000 = 65,440; HELDOUT 6,000, all from pi-noise.
- The held-out pi-noise transitions are exactly starts 80–99; TRAIN starts are ≤ 79. The mask equals the direct `episode` test.
- All 390 demo-branch episodes are TRAIN episodes of the stage-one split.

**Sidecar ↔ store.** All 46 shards:

- sidecar rows = shard frames = checks rows, `frame` = 0 … n − 1;
- |store proprio − sidecar qpos|, arm: max 1.1e-2, shard medians 8.4e-4 to 1.0e-3. With the sidecar shifted by one row, the medians are 0.17 to 0.23;
- `cache_row` = `frame_rows(episode, t)` at all 3,120 branch points.

**Chains.**

- Sim time o′ − o = 0.5 s in 71,440 / 71,440 (max error 2.8e-13). Every o′ is a rollout frame with parent o and step + 1.
- `random_walk`: 3,120 chain starts, one per state; 9,360 continuations.
  - Each walk's first step runs from a fresh restore of the branch point.
  - The gripper command equals the previous chunk's unless switched: 9,360 / 9,360 (2,814 switched).
- `init_walk`: 200 chain starts (20 per task), 3,800 continuations, 20 transitions t = 1 … 20 per start. Gripper command: 3,800 / 3,800 (1,091 switched).
- `pi_noise`: 1,000 chain starts (100 per task), 29,000 continuations, 30 transitions t = 1 … 30 per start.

**demo-branch states.** The 3,120 restored (episode, t) states are distinct and equal `BranchPoints` (8 per TRAIN episode, seed 0). Each has 8 `random_branch` and 4 `random_walk` transitions.

**pi-noise.**

- The choice `default_rng([0, 4, task, start, decision]).integers(8)` equals the sidecar's `candidate` in 30,000 / 30,000.
- Candidates 0 … 7 were executed 3,682 / 3,898 / 3,771 / 3,723 / 3,737 / 3,759 / 3,743 / 3,687 times.
- `sigma` equals the sidecar's `chunk_scale`: 0 (π's mean chunk) 3,682; 0.5: 11,322; 1: 7,530; 2: 7,466.
- **Recomputation of the stored chunks.** π on each stored current frame on the GPU (node-65 CUDA 1) reproduces all 30,000 executed chunks exactly: arm difference 0, no differing gripper step.
  - The Phase 4 check (`tmp/s4p4/check_pi_noise.py`) on the CPU agrees on the choice, scale and J of every transition.
  - Its largest chunk difference is 1.0. A gripper step binarized on the other side of 0.5 would give this difference (hypothesis).
- **Success**: the share of starts that reach success at some decision is 0.208 (per task 0.01–0.53).

**Smooth random chunks.**

| | J = 1 / 2 / 3 | s = 0.5 / 1 / 2 | Gripper switch rate | Per-entry RMS median at s = 0.5 / 1 / 2 | Arm entries at ±1 at s = 0.5 / 1 / 2 |
|---|---|---|---|---|---|
| demo-branch | 12,452 / 12,467 / 12,521 | 12,504 / 12,358 / 12,578 | 0.300 | 0.489 / 0.962 / 1.75 (`random_branch`), 0.488 / 0.963 / 1.76 (`random_walk`) | 0.000 / 0.012 / 0.110–0.112 |
| init-random | 1,393 / 1,340 / 1,267 | 1,318 / 1,382 / 1,300 | 0.283 | 0.483 / 0.959 / 1.75 | 0.000 / 0.012 / 0.111 |

- With J = 1 the energy share of DCT coefficients ≥ J is ≤ 4.5e-31. With J = 2 or 3 its maximum is 7.3e-2; nonzero values come from clipping.
- DCT energy share of the arm dims by coefficient 0 / 1 / 2:
  - `random_branch`: 0.613 / 0.273 / 0.112;
  - `init_walk`: 0.624 / 0.269 / 0.106;
  - `pi_noise`: 0.533 / 0.259 / 0.202.

**Restore fidelity** (demo-branch branch points, 3,120):

- Against the `demo_renders` frame of the same (episode, t): proprio bytes identical 3,120 / 3,120, token cosine ≥ 0.999996.
- Against the dataset proprio:
  - median 1.8e-8, p95 2.3e-7, max 9.8e-4;
  - 63 branch points exceed 1e-6. They are `demo_renders`' own replay errors; the largest are in episodes 278 (task 6, t = 77–79) and 87 (task 2, t = 112–117).
- Token cosine against the feature-cache video frame: agent mean 0.969 (min 0.951), wrist 0.939 (min 0.774).
- Fresh restores before a chunk reproduce the first restore (proprio and both images) in 27,725 / 28,080. The 355 exceptions fall in 93 states; 264 of them are in task 7, as in part 1.

### Effects

Per transition, from the sidecars (object 1 cm, slide 1 cm, knob 0.05 rad):

| Transitions | n | Object moved | Fixture changed | Grasp changed | Any | Goal holds after |
|---|---|---|---|---|---|---|
| `random_branch` | 24,960 | 35.7% | 5.7% | 13.3% | 43.0% | 13.9% |
| `random_walk` | 12,480 | 31.3% | 5.5% | 9.5% | 37.8% | 14.2% |
| demo-branch, s = 0.5 / 1 / 2 | 12,504 / 12,358 / 12,578 | 28.9% / 34.4% / 39.4% | 4.4% / 5.5% / 7.0% | 11.4% / 11.9% / 12.8% | 35.9% / 41.3% / 46.5% | 15.2% / 13.4% / 13.3% |
| `init_walk` | 4,000 | 9.7% | 0.3% | 0.7% | 10.3% | 0 |
| `pi_noise`, TRAIN (starts 0–79) | 24,000 | 16.4% | 3.7% | 2.5% | 20.3% | 10.5% |
| `pi_noise`, HELDOUT (starts 80–99) | 6,000 | 17.0% | 4.2% | 2.8% | 21.5% | 9.6% |
| `pi_noise`, σ 0 / 0.5 / 1 / 2 | 3,682 / 11,322 / 7,530 / 7,466 | 14.1% / 14.7% / 17.5% / 19.5% | 3.5% / 3.6% / 3.7% / 4.2% | 3.0% / 2.6% / 2.6% / 2.4% | 18.2% / 18.7% / 21.3% / 23.8% | 10.5% / 10.4% / 10.2% / 10.4% |

Share of chunks beyond the HELDOUT demo chunks' p95 distance to the nearest TRAIN demo chunk (arm dims 3.39 / all dims 3.53):

- `random_branch` 0.661 / 0.690; `random_walk` 0.656 / 0.687;
- `init_walk` 0.645 / 0.676;
- `pi_noise` 0.717 / 0.741.

### Runs, time and disk

- **Hosts**: node-61 CUDA 0 and 1, node-65 CUDA 1 and 2. `gpu_guard.sh` passed on all four cards, which were idle at launch.
  - 5 queue workers per card, 20 in all. OMP, MKL, OPENBLAS and NUMBA threads: 2.
  - One job per shard, 46 jobs, longest first. Each job was claimed with an atomic `mkdir`, so a relaunch skips complete shards (`tmp/s4p6/plan_part2_queue.py`, `run_queue.sh`, `launch_queue.sh`, entry `start_part2.sh` through `scripts/cluster_run.sh`).
- **Wall time**: 2026-09-29 00:01:26 to 00:18:48 EDT, 1,042 s. 16,858 s of collection over the shards.
- **Load**: peak load average 10.4 (node-61) and 18.4 (node-65). On node-65, GPUs 0 and 7, idle at launch, were taken during the run by processes outside this collection. On our cards GPU memory stayed ≤ 7.5 GiB and utilization samples ≤ 72%.

| Source | Estimated process time / disk (`docs/interaction_collection.md`) | Measured collection seconds / disk | s per transition |
|---|---|---|---|
| demo-branch TRAIN | 10,184 s / 7.5 GiB | 9,944 s / 7.50 GiB | 0.266 (replay included) |
| init-random | 780 s / 0.8 GiB | 805 s / 0.78 GiB | 0.201 |
| pi-noise | 6,150 s / 5.7 GiB | 6,109 s / 5.73 GiB | 0.204 |

- Per-shard medians: restore 28–33 ms, rollout 148–188 ms, encode 13.5–18.3 ms, scene read 3.2–4.0 ms.
- Checks files and reports: 5.1 MiB.
- Part 1 and part 2 together: 133,297 transitions, 25.96 GiB.

**Outputs**:

- `outputs/analysis/interaction_collection/<store>/`: checks files and `report.txt`;
- `outputs/analysis/interaction_collection/phase6_part2_checks.json`;
- `outputs/analysis/interaction_collection/phase6_part2_pi_noise_recompute.json`.

Logs: `tmp/runlogs/s4p6/collect2/` (workers, launch states, load).

```bash
source scripts/ceam_env.sh   # CEAM_ROOT = main checkout; OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=2
SAMPLER="--dct-coefficients 1 2 3 --scales 0.5 1.0 2.0 --gripper-switch-prob 0.3 --seed 0"
$CEAM_PYTHON scripts/collect_interactions.py demo-branch --cuda-device <gpu> --code-commit e0df3d95c61d7ef409972426c418b57fe0b3e1ef \
  --out-dir outputs/transitions/random_demo_branch_train --analysis-dir outputs/analysis/interaction_collection/random_demo_branch_train \
  --split train --shard <i> --num-shards 26 --states-per-episode 8 --branches 8 --walk-length 4 --env-seed 0 $SAMPLER
$CEAM_PYTHON scripts/collect_interactions.py init-random --cuda-device <gpu> --code-commit e0df3d95c61d7ef409972426c418b57fe0b3e1ef \
  --out-dir outputs/transitions/random_init_walk --analysis-dir outputs/analysis/interaction_collection/random_init_walk \
  --tasks <t> --starts 20 --walk-length 20 --env-seed 100 $SAMPLER
$CEAM_PYTHON scripts/collect_interactions.py pi-noise --cuda-device <gpu> --code-commit e0df3d95c61d7ef409972426c418b57fe0b3e1ef \
  --out-dir outputs/transitions/pi_noise_nopsi --analysis-dir outputs/analysis/interaction_collection/pi_noise_nopsi \
  --tasks <t> --starts 100 --decisions 30 --env-seed 2 --candidates 8 --proposal-scales 0.5 1.0 2.0 --proposal-dct 3 \
  --proposal-seed 0 --choice-seed 0
```

## Coverage after collection

`scripts/inventory_transitions.py report --collected` adds the collected training stores to the four Phase 3 sources (`outputs/analysis/interaction_inventory/after_phase6/report.md`, `coverage.npz`).

- **Added stores**:
  - the three `resim_phase26_sweep_*` stores (held out: `episode` 40–49);
  - `random_demo_branch_train` and `random_init_walk` (no held-out part);
  - `pi_noise_nopsi` (held out: `episode` 80–99).
- **Not added separately**: the `resim_phase26_mean_*` stores. Their transitions are the Phase 26 source already counted: the same record pairs, 15,234 TRAIN transitions.
- **The evaluation set**: `counterfactual_eval_heldout` is not a training source.
- **How added stores enter the statistics**:
  - chunks and effects: as the store's training transitions;
  - effect labels: from the sidecar at o and o′;
  - state coverage: their reached frames, the rollout frames; their restored and reset frames are left out.

### Pre-registered criteria and totals (non-demo training transitions)

| Quantity | Phase 3 | After Phase 6 |
|---|---|---|
| transitions | 65,166 | 159,790 |
| chunks beyond the demo's own p95 (episode out, 3.549) | 25.5% | 40.1% |
| the same with the chunk-out p95 (2.271) | 37.8% | 59.6% |
| transitions moving an object (data A share estimated) | 29.4% [28.4%, 30.4%] | 25.8% [25.4%, 26.2%] |
| without data A: chunks beyond 3.549 | 0.7% (33,966 transitions) | 37.1% (128,590) |
| without data A: transitions moving an object | 21.8% | 22.9% |
| share of the far chunks that are data A chunks | 98.6% | 25.6% |
| effective rank of the flattened chunks [70] | 8.58 | 13.71 |
| top eigenvalue share | 0.207 | 0.159 |
| arm energy at DCT k ≥ 1 | 6.0% | 25.3% |
| gripper: in-chunk step pairs switching / chunks with an in-chunk switch | 1.5% / 13.2% | 1.6% / 13.0% |
| gripper: switch at the chunk boundary | 8.3% | 16.5% |
| frames beyond the demo's own p95 (11.880) | 29.8% (67,166 frames) | 41.3% (161,790) |

Neither criterion holds before or after, so the rule itself does not call for collection. After the collection, the far chunks no longer come mostly from data A's constant offsets:

- without data A, 37.1% of the chunks lie beyond the threshold;
- the arm energy above the chunk mean rises from 6.0% to 25.3%.

The smooth random chunks keep one gripper command for the whole chunk, so they switch only at chunk boundaries: 30.0% (demo-branch) and 28.3% (init-random), 0.0% within the chunk. π's noisy rollouts switch within 27.5% of their chunks, 3.5% of step pairs.

### Per source (training parts)

| Source | Transitions | Effective rank [70] | Arm energy at k ≥ 1 | Chunk distance median / beyond 3.549 | Reached frames beyond 11.880 | Object moved | Fixture changed | Grasp changed |
|---|---|---|---|---|---|---|---|---|
| demo TRAIN (reference) | 43,701 | 8.51 | 9.6% | 1.824 / 5.0% | 5.0% | 40.2% | 7.4% | 12.3% |
| data A | 31,200 | 8.08 | 4.0% | 3.721 / 52.5% | 21.8% (o′) | 37.7% (sample) | 7.0% | 12.1% |
| rollouts_r0 | 18,732 | 9.13 | 11.2% | 0.000 / 0.0% | 56.0% | 15.7% | 2.3% | 2.6% |
| Phase 26 (= resim mean) | 15,234 | 7.85 | 9.0% | 1.531 / 1.5% | 13.9% | 29.3% | 5.0% | 6.2% |
| `resim_phase26_sweep_fixed` | 14,592 | 8.49 | 15.4% | 1.926 / 4.1% | 42.5% | 20.4% | 2.2% | 5.3% |
| `resim_phase26_sweep_flow` | 7,296 | 8.19 | 14.0% | 1.934 / 4.1% | 43.8% | 12.5% | 9.5% | 2.9% |
| `resim_phase26_sweep_flow_full_sgd` | 7,296 | 9.43 | 15.6% | 1.990 / 5.3% | 49.2% | 13.1% | 4.5% | 1.2% |
| `random_demo_branch_train` | 37,440 | 15.44 | 38.4% | 4.918 / 68.7% | 34.0% | 34.3% | 5.6% | 12.0% |
| `random_init_walk` | 4,000 | 15.18 | 37.3% | 4.833 / 67.5% | 79.8% | 9.7% | 0.3% | 0.7% |
| `pi_noise_nopsi` | 24,000 | 19.12 | 51.6% | 5.003 / 73.8% | 74.8% | 16.4% | 3.7% | 2.5% |

- Arm energy at k ≥ 1 is 1 − the k = 0 share of the summed arm-dim energy.
- Chunk distance: to the nearest TRAIN demo chunk, flattened [70]. Reached frames: distance to the nearest TRAIN demo frame in the pooled DINOv2 feature.
- The demo TRAIN chunk and frame distances leave the episode out.

By kind:

- `random_branch` beyond 3.549: 24.8% / 83.0% / 98.0% at s = 0.5 / 1 / 2.
- `pi_noise`: 8.5% (π's mean chunk) and 60.5% / 99.6% / 100% at σ 0.5 / 1 / 2.
- The resim sweep candidates stay close to the demonstrations: 3.8–5.9% beyond 3.549 at every σ ≤ 0.2.

The six pooled feature files added for this report (three resim sweep stores and the three part 2 stores) are `outputs/analysis/interaction_inventory/pooled/<store>.npy` (`pool --stores`, 656 MiB).

### Report code and the unchanged Phase 3 path

Commit `2fb0671` adds two options to `scripts/inventory_transitions.py`:

- `pool --stores NAME …`: the stores to pool; the default is the three Phase 3 stores.
- `report --collected NAME[:FIRST-LAST] …`: collected stores (`CollectedSource`) as further training sources, with the held-out `episode` range.
  - Kinds are `<kind> σ <sigma>`.
  - Effects come from the sidecar SceneState at o and o′.
  - The gripper command before a chunk:
    - for a smooth random chunk, its command, or the opposite where the sidecar marks the chunk as switched;
    - otherwise the last step of the chunk that produced o, the reset command, or for a restored Phase 26 decision d the last step of the record's chunk d − 1.
  - The figures keep the four Phase 3 sources.

Without `--collected`, the report reproduces the Phase 3 report:

- rerun with 4 BLAS threads, as in Phase 3: `report.md` identical apart from the header and the report time, `coverage.npz` byte for byte (`regression_phase6_t4/`);
- with 8 threads: the four PCA projection arrays, the components and their explained variance differ by at most 4.4e-14 (`regression_phase6/`).

Runs, all 2026-09-29 EDT, commit `2fb0671`:

| Step | Host / device | Time |
|---|---|---|
| `pool --stores` of the six stores | node-65, CPU | 50 s |
| `report`, without and with `--collected` | node-65 CUDA 1 | 49 s and 53 s |
| regression rerun, 4 threads | node-65 CUDA 1 | 38 s |

```bash
source scripts/ceam_env.sh
$CEAM_PYTHON scripts/inventory_transitions.py pool --stores resim_phase26_sweep_fixed resim_phase26_sweep_flow \
  resim_phase26_sweep_flow_full_sgd random_demo_branch_train random_init_walk pi_noise_nopsi
$CEAM_PYTHON scripts/inventory_transitions.py report --device cuda:<gpu> --code-commit <sha> \
  --report-dir outputs/analysis/interaction_inventory/after_phase6 \
  --collected resim_phase26_sweep_fixed:40-49 resim_phase26_sweep_flow:40-49 resim_phase26_sweep_flow_full_sgd:40-49 \
  random_demo_branch_train random_init_walk pi_noise_nopsi:80-99
```

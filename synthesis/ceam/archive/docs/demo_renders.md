# Demo renders: the dataset episodes replayed in the simulator

All 433 LeRobot LIBERO-Goal episodes are replayed in the simulator. Every dataset frame is restored, rendered and encoded like data A (`docs/dynamics_data.md`). The result is a `TransitionStore` of rendered frames with the demonstration (t, t + 10) transitions, plus per-frame scene labels read from MuJoCo. Training uses these rendered frames. The dataset video frames (feature cache) are paired with them row by row for frame-domain diagnostics only.

Code:

- `ceam/sim/scene_state_reader.py`: `SceneState`, `SceneStateReader`.
- `scripts/render_demo_episodes.py`, with subcommands `render` and `report`.

## SceneStateReader

`SceneStateReader(task_env, probe).read() -> SceneState` reads the current state of a `LiberoTaskEnv` from MuJoCo model and data. It never writes the simulator.

- `probe` must be a `SceneProbe` of the same env. The grasp flags are `probe.snapshot().grasped`, so "grasped" has a single definition: both finger pads are in contact with the object.
- It works on any state the env's own methods leave, including a frame restored by `DemoStateRestorer`.

`SceneState` fields (world frame, SI units, quaternions (w, x, y, z), float64):

| Field | Shape | Source |
|---|---|---|
| `object_names` | n_obj | LIBERO `objects_dict` order. On LIBERO-Goal: `akita_black_bowl_1`, `cream_cheese_1`, `wine_bottle_1`, `plate_1` |
| `object_positions`, `object_quaternions` | [n_obj, 3], [n_obj, 4] | `data.xpos`, `data.xquat` of each object's root body |
| `object_grasped` | [n_obj] bool | `SceneProbe` grasp test |
| `fixture_joint_names`, `fixture_joint_qpos` | n_joint, [n_joint] | every slide or hinge joint outside the robot. On LIBERO-Goal: `wooden_cabinet_1_{top,middle,bottom}_level` (slides, m) and `flat_stove_1_button` (the knob, rad) |
| `eef_position`, `eef_quaternion` | [3], [4] | `site_xpos`, `site_xmat` of `robot.eef_site_id` (grip_site), the site of robosuite's `eef_pos` |
| `joint_qpos`, `gripper_qpos` | [7], [2] | `robot.robot_joints`, `robot.gripper.joints` via `jnt_qposadr` |

The reader checks its addressing at construction and raises if a check fails:

- each object's root body carries exactly one free joint and is a child of the world body;
- the free joints of the model are exactly the objects' joints;
- the robot has 7 arm and 2 finger joints;
- every remaining joint is a slide or hinge.

`object_poses_from_sim_state(sim_state)` decodes a flattened state [time, qpos, qvel] (79 values on LIBERO-Goal) into object positions and quaternions through the free joints' qpos addresses. This works for a raw hdf5 `states[k]` and for a `SimSnapshot.sim_state`.

**Timing within a step.** A read after a step sees the simulator after the step's last physics substep:

- The qpos fields (arm, fingers, fixture joints) and `sim_state` are that final state.
- Body and site poses are the kinematics MuJoCo computed at the start of the last substep.
- The observation is sampled one substep before the step ends (`docs/libero_sim.md`).

Measured over all 52,895 frames:

| Comparison | Median | Max |
|---|---|---|
| \|observation proprio − SceneState qpos\|, arm | 8.9e-4 rad | 5.4e-3 rad |
| \|observation proprio − SceneState qpos\|, fingers | 1.3e-6 | 3.8e-4 |
| object `xpos` vs the free-joint qpos of the same state | — | 3.6 mm (bowl), 0.8–1.9 mm (other objects) |

## Store, sidecar and checks

**Store** `outputs/transitions/demo_renders/` (`TransitionStore`, format unchanged):

- 20 shards `<split>_task<NN>`. The split is `EpisodeIndex` TRAIN / HELDOUT with heldout fraction 0.1 and seed 0, from the stage-one checkpoint's config. Each manifest's provenance also records `split` and `task_id`.
- Frames: every dataset frame once, in episode-id then t order. Each frame holds frozen DINOv2 pooled tokens [2, 64, 768] float16 (agent, wrist) from `ObservationEncoder` and raw proprio [9] float32.
- Transitions: one per t with t + 10 ≤ T − 1, the `StageTwoSampler` rule. Each has:
  - current frame t, future frame t + 10;
  - `actions`: the dataset's raw chunk a_t … a_{t+9};
  - `kind` "demo", `sigma` 0, `task_id`, `episode` (dataset episode id), `t`;
  - `goal_after`: LIBERO's success check at frame t + 10;
  - `episode_success` False, the store's value for non-rollout transitions.
- A transition's two frames are restores of one simulator replay, in which the chunk was executed from frame t to reach frame t + 10.

**Sidecar** `<shard>/scene.npz` holds one row per shard frame; row i is shard-local frame i. It is written before the manifest.

| Keys | Content |
|---|---|
| `frame`, `episode`, `t`, `task_id` | int64 [n] |
| `cache_row` | feature-cache row of the same dataset frame (the video frame) |
| `raw_demo_index`, `raw_step` | raw hdf5 `demo_<k>` and the raw step of the frame (`RawDemoMatch.raw_steps`) |
| `object_names`, `object_positions`, `object_quaternions`, `object_grasped`, `fixture_joint_names`, `fixture_joint_qpos`, `eef_position`, `eef_quaternion`, `joint_qpos`, `gripper_qpos` | `SceneState` fields, stacked over frames |
| `goal_holds` | LIBERO's success check at the frame |
| `sim_state` [n, 79], `gripper_accumulator` [n, 2], `goal_orientation` [n, 3, 3], `fixture_bodies`, `fixture_positions` [n, 3, 3], `fixture_quaternions` [n, 3, 4] | the env's `SimSnapshot` at the frame (after the producing step) |

**Checks** `outputs/analysis/demo_renders/<shard>_checks.npz`, per frame:

- restore proprio error against the dataset;
- token cosine per view against the feature cache;
- `SceneStateReader` − `SceneProbe` differences;
- observation proprio − `SceneState` qpos;
- raw-state and replay-state object distances;
- restore, encode and read times.

Per episode it holds the replay time; per shard, the refused frames. `outputs/analysis/demo_renders/report.txt` is the output of `render_demo_episodes.py report`.

**Refusals.** `LiberoTaskEnv.observe` raises on a black render inside `restore`. Such a frame would not be written; it would be recorded in the manifest (`refused_frames`) and in the checks file. The transitions touching it would be skipped and counted per episode. The run had 0 refused frames.

## Run

- Code: commit `7f4aabd`; the report additions are in `fec368c`, and the render path is unchanged.
- Host: node-61, 2 L40S (CUDA 0: tasks 0–4, CUDA 1: tasks 5–9), one process per task, all ten started together. Both GPUs were idle before launch.
- Launcher: `tmp/s4p1/launch_render.sh`. Logs: `tmp/runlogs/s4p1/render_task<NN>.log`. Report: `tmp/s4p1/report.sh`.

```bash
source scripts/ceam_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
$CEAM_PYTHON scripts/render_demo_episodes.py render --cuda-device 0 --code-commit <commit> --tasks 0
$CEAM_PYTHON scripts/render_demo_episodes.py report --report-file outputs/analysis/demo_renders/report.txt
```

- Thread pools: ten processes with the default thread pools oversubscribed the 128 cores (load average 360, about 5× slower per frame). With 2 threads each, a process runs at about 95% of one core.
- `/tmp/robosuite.log` on node-61 was not modified; its last write predates the run.

## Checks

### Frames and transitions

| Check | Result |
|---|---|
| frames | 52,895, equal to the feature cache |
| (episode, t) ↔ feature-cache row | `cache_row` = `frame_rows(episode, t)` for 52,895 / 52,895; every row 0 … 52,894 exactly once |
| episodes in a shard of another split or task | 0 |
| TRAIN transitions | 43,701 = `StageTwoSampler` valid pairs; 0 missing, 0 extra |
| HELDOUT transitions | 4,864 = valid pairs; 0 missing, 0 extra |
| current / future frame = (episode, t) / (episode, t + 10) | all |
| stored chunks, standardized, vs `ActionCache.chunks` | max difference 0 |
| `goal_after` true | 5,462 |
| refused frames | 0 |

### Restore proprio error against the dataset

| Frames | Max | p99 | Median |
|---|---|---|---|
| all (52,895) | 9.3e-3 | 1.3e-5 | 1.5e-8 |
| TRAIN (47,601) | 6.6e-3 | 1.0e-5 | 1.5e-8 |
| HELDOUT (5,294) | 9.3e-3 | 1.5e-3 | 1.5e-8 |

- 48 of 433 episodes exceed 1e-6, over 1,210 frames. 249 frames exceed 1e-4.
- The large errors are late in the episode, mostly in task 2, "push the plate to the front of the stove". Examples:
  - episode 91: first frame above 1e-5 at t = 145 of 153;
  - episode 78: t = 128 of 144;
  - episode 69 (task 1): t = 181 of 251.
- They come from the replay, not from the restore. A second replay through `LiberoTaskEnv`'s public methods, compared per frame (`tmp/s4p1/replay_divergence.py`):

| Episode | \|replay − dataset\| max | \|restore − replay\| max | \|restore now − stored\| max |
|---|---|---|---|
| 0 | 2.4e-7 | 0 | 0 |
| 91 | 9.3e-3 | 2.3e-10 | 0 |
| 78 | 6.6e-3 | 9.3e-10 | 0 |
| 69 | 5.7e-3 | 1.5e-5 | 0 |
| 77 | 1.7e-3 | 1.1e-4 | 1.1e-4 |
| 95 | 1.2e-3 | 0 | 0 |
| 278 | 1.1e-3 | 0 | 0 |

At these frames the rendered state differs slightly from the dataset frame's state. Within the simulator, the transitions stay consistent: both frames come from the same replay.

### Token cosine, render vs feature cache (video frame), same (episode, t)

| View | Mean | Min | p0.1 | p1 | p5 | Median | p95 | Max |
|---|---|---|---|---|---|---|---|---|
| agent | 0.9691 | 0.9460 | 0.9531 | 0.9601 | 0.9630 | 0.9689 | 0.9756 | 0.9822 |
| wrist | 0.9392 | 0.5771 | 0.8204 | 0.8678 | 0.8977 | 0.9437 | 0.9663 | 0.9858 |

- TRAIN and HELDOUT agree within 0.002 in mean.
- Per-task means are 0.967–0.973 (agent) and 0.925–0.948 (wrist).
- 19 frames have wrist cosine below 0.8:
  - 6 are the diverged ends of episodes 78 and 91 (minimum 0.577 at episode 78, t = 143);
  - the other 13, in episodes 101, 103, 109, 262, 268, 288 and 331, have restore proprio error ≤ 1e-6.
- Mean wrist cosine by restore proprio error: 0.9395 (< 1e-6, 51,685 frames), 0.9283 (1e-6 to 1e-4, 961), 0.9181 (> 1e-4, 249).

### SceneStateReader vs SceneProbe (same states, all 52,895 frames)

| Quantity | Max \|difference\| |
|---|---|
| object positions (each of the 4 objects) | 0 |
| fixture joint qpos | 0 |
| end-effector position | 0 |
| gripper qpos | 0 |

- Frames with a grasped object: 12,635. By object: bowl 6,251, cream cheese 1,525, wine bottle 4,734, plate 125.
- Frames where the goal holds: 5,462.

### Raw hdf5 `states[raw_step]` vs replay labels

Object position distance ‖raw decode − `SceneState`‖ over all frames, and over the frames where the object moved more than 1 cm from its t = 0 position:

| Object | Median | p90 | Max | t = 0 median | Frames moved > 1 cm | Moved: median | Moved: max |
|---|---|---|---|---|---|---|---|
| akita_black_bowl_1 | < 0.1 mm | 9.7 mm | 242 mm | 0.2 mm | 10,270 | 9.6 mm | 242 mm |
| cream_cheese_1 | < 0.1 mm | 0.2 mm | 89 mm | 0.0 mm | 3,435 | 8.6 mm | 89 mm |
| wine_bottle_1 | < 0.1 mm | 2.5 mm | 89 mm | 0.2 mm | 6,093 | 6.1 mm | 89 mm |
| plate_1 | < 0.1 mm | 0.2 mm | 60 mm | 0.2 mm | 3,615 | 12.3 mm | 60 mm |

Per task, over (frame, object) pairs where the object moved more than 1 cm:

| Task | Pairs | Median | p90 | Max |
|---|---|---|---|---|
| 0 | 0 | — | — | — |
| 1 | 3,092 | 18.9 mm | 57.0 mm | 242 mm |
| 2 | 4,323 | 12.9 mm | 33.6 mm | 89 mm |
| 3 | 2,316 | 6.7 mm | 19.0 mm | 50 mm |
| 4 | 2,527 | 7.5 mm | 19.0 mm | 50 mm |
| 5 | 2,865 | 10.5 mm | 24.8 mm | 162 mm |
| 6 | 2,215 | 4.7 mm | 14.1 mm | 31 mm |
| 7 | 3,446 | 8.6 mm | 23.8 mm | 89 mm |
| 8 | 2,629 | 3.8 mm | 15.7 mm | 35 mm |
| 9 | 0 | — | — | — |

- Static objects agree with the raw states. At t = 0 the distance is the 0.2 mm settling of the 10 settle steps.
- Moved objects differ by millimetres to centimetres, and by up to 24 cm in task 1, where the bowl is placed in the drawer. Examples:
  - episode 47: 226 mm, with restore proprio error 2.4e-7;
  - episode 51: 242 mm, with restore proprio error 3.0e-7.
- So the raw teleoperation trajectory differs from the re-simulation even where the robot follows the dataset exactly. The replay labels are the labels of the rendered and the dataset frames; the raw states are not.

### Timing and disk

| Item | Value |
|---|---|
| wall time | 681 s (slowest process, task 1) |
| per process | 425–681 s |
| per frame, median, ten processes on two GPUs | restore 35.2 ms, encode 29.4 ms, `SceneState` + probe + snapshot 3.2 ms |
| replay (`DemoStateRestorer` construction) | median 3.5 s per episode, 1,639 s in total |
| restore + encode + read | 3,678 s in total |
| store | 9.77 GiB, of which sidecars 67.4 MiB |
| checks files | 16.7 MiB |

## Using the store

- **TRAIN / HELDOUT**: select transitions by shard name prefix (`TransitionTable.shard` → `TransitionStore.shards[i].name`) or by the manifest's `provenance["split"]`.
- **Frame-domain pairing**: store frame (shard `first_frame` + sidecar row) ↔ feature-cache row `cache_row`.
  - 249 frames have restore proprio error above 1e-4, listed in the checks files. Their render shows a slightly different state from their video frame.
- **Scene labels**: the sidecar holds the scene labels of the demonstration frames; no separate label directory is written.

# LIBERO simulator conventions (LIBERO-Goal)

Measured with `scripts/inspect_libero_sim.py` on node-71, CUDA device 7, in the project `.venv` (mujoco 3.3.2, robosuite 1.4.0, libero 0.1.1, torch 2.7.1). The LeRobot dataset is `$CEAM_DATA_ROOT/libero_goal_no_noops_lerobot`; the raw demos are `outputs/raw/LIBERO-datasets/libero_goal/<task_name>_demo.hdf5`.

```bash
source scripts/ceam_env.sh
$CEAM_PYTHON scripts/inspect_libero_sim.py   # CUDA_VISIBLE_DEVICES unset; --cuda-device 7 (default); 164 s
```

Side-by-side renders are written to `outputs/analysis/libero_sim/`: `init_<benchmark id>_<task>.png` (benchmark init state after settling: env image row, then rotated row) and `episode_<id>_dataset_sim_diff.png` (per selected t and view: dataset frame | rotated sim frame | 4 × absolute difference).

## Runtime

**EGL device.** robosuite 1.4.0 renders on the EGL device whose *list index* is `MUJOCO_EGL_DEVICE_ID`, or `CUDA_VISIBLE_DEVICES` when that variable is unset, and asserts that `MUJOCO_EGL_DEVICE_ID` is a substring of `CUDA_VISIBLE_DEVICES` whenever the latter is set. On node-71, `eglQueryDevicesEXT` lists 17 devices: indices 0–7 are the NVIDIA devices, 8–15 Mesa DRM devices, 16 the Mesa software device. The EGL index and CUDA device (`EGL_CUDA_DEVICE_NV`) differ:

| EGL index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| CUDA device | 3 | 2 | 0 | 1 | 7 | 6 | 4 | 5 |

With `CUDA_VISIBLE_DEVICES=7`, robosuite renders on EGL index 7, which is CUDA device 5 (`nvidia-smi` lists the process as a graphics process on GPU 5). The script runs with `CUDA_VISIBLE_DEVICES` unset. It sets `MUJOCO_EGL_DEVICE_ID` to the EGL index whose `EGL_CUDA_DEVICE_NV` equals `--cuda-device`, and puts torch on `cuda:<id>`. While a render env is open, `nvidia-smi` lists the process as `C+G` on GPU 7 only.

**robosuite file log.** `robosuite/macros.py` sets `FILE_LOGGING_LEVEL = "DEBUG"`, so importing robosuite attaches a `FileHandler` on `/tmp/robosuite.log` and appends three "No private macro file found!" warnings. `RobosuiteRuntime` executes `robosuite/macros.py`, sets `FILE_LOGGING_LEVEL = None` and registers the module as `sys.modules["robosuite.macros"]` before `import robosuite`. robosuite's `log_utils` then uses that module (the logger has only a `StreamHandler`), and `/tmp/robosuite.log` is not modified by a run. robosuite's numba functions use `cache=True`; `NUMBA_CACHE_DIR` (set by `scripts/ceam_env.sh`) keeps the cache in the project.

**Init states.** `benchmark.get_task_init_states` calls `torch.load` with torch 2.7's default `weights_only=True` and fails (`numpy.core.multiarray._reconstruct` is not an allowed global). The files load with `torch.load(<init_states>/libero_goal/<task>.pruned_init, weights_only=False)` as float arrays of shape (50, 79).

**Task order.** `benchmark.get_benchmark_dict()["libero_goal"]()` orders tasks differently from the dataset's `task_index`, so tasks are mapped by language:

| Benchmark id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Dataset `task_index` | 0 | 4 | 8 | 1 | 5 | 2 | 6 | 9 | 3 | 7 |

## Environment construction

```python
env = OffScreenRenderEnv(
    bddl_file_name=f"{get_libero_path('bddl_files')}/libero_goal/{task_name}.bddl",
    camera_heights=512, camera_widths=512, hard_reset=False,
)
```

- All 10 tasks construct. `control_freq` = 20 Hz, model timestep 0.002 s (25 physics steps per `env.step`), horizon 1000. Once 1000 steps have run without a reset, `env.step` raises `ValueError("executing action in terminated episode")`.
- Image keys: `agentview_image` and `robot0_eye_in_hand_image`, each (512, 512, 3) uint8, the same resolution as the dataset videos.
- Robot keys: `robot0_joint_pos` (7), `robot0_joint_pos_cos`, `robot0_joint_pos_sin`, `robot0_joint_vel`, `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2), `robot0_gripper_qvel`, `robot0_proprio-state` (39).
- Object keys: `<object>_pos`, `<object>_quat`, `<object>_to_robot0_eef_pos`, `<object>_to_robot0_eef_quat` for each movable object, plus `object-state`.
- `env.env.sim.get_state().flatten()` has length 79, the same as the raw `states` rows.

**Black renders.** Some render contexts return all-zero frames for both cameras for the lifetime of the env. In repeated single-env runs with mujoco 3.3.7, such constructions took 11–14 s against 4–6 s for normal ones; `glGetError` and `glGetGraphicsResetStatus` were 0 and the renderer string was still `NVIDIA H100 NVL`. In the documented run, 2 of 14 constructions (every render env built during the run) had black first frames; each was rebuilt once and then rendered normally. `LiberoEnvFactory.render_env` builds with `hard_reset=False`, then calls `reset()`. It then makes the env's EGL context current and binds the offscreen framebuffer (`mujoco.mjr_setBuffer(mjFB_OFFSCREEN, con)`). It renders once, and rebuilds (up to 6 attempts) unless both image means are ≥ 40. Healthy frames have means of 110–120, and the black ones measured 0.0. With robosuite's default `hard_reset=True`, every `reset()` creates a new GL context. With mujoco 3.3.7 this produced black frames at random resets and a SIGABRT inside `mjr_readPixels` after 29–33 resets. Fixture placements are identical under hard and soft resets.

## Reset from a benchmark init state

```python
env.reset()
obs = env.set_init_state(init_states[i])
for _ in range(10):
    obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
```

- max |qvel| is 0 right after `set_init_state`. It is 0.49 and 0.98 after steps 1–2, while the gripper opens from its init width. After step 10 it is 0.005–0.051 across the 10 tasks. `done` is False during the settle steps for every task.
- Gripper value of the no-op: −1 opens the gripper (`robot0_gripper_qpos` ±0.0387 after 10 steps), +1 closes it (±0.0043).
- The dataset's first frame is the state after these 10 no-op steps (next section).

## Raw hdf5 demos

| Item | Value |
|---|---|
| Layout | `data/demo_<k>` for k = 0..49 in every task file |
| Arrays per demo | `actions` [T, 7] f64, `states` [T, 79] f64, `robot_states` [T, 9], `rewards` [T] u8, `dones` [T] u8, `obs/{agentview_rgb, eye_in_hand_rgb}` [T, 128, 128, 3] u8, `obs/{joint_states [T, 7], gripper_states [T, 2], ee_pos, ee_ori, ee_states}` |
| Demo attributes | `init_state` (equal to `states[0]`), `model_file` (the MJCF used during collection), `num_samples` |
| `data` attributes | `env_args` (OSC_POSE, `control_delta` true, `control_freq` 20, cameras `robot0_eye_in_hand`, `agentview` at 128×128, `ignore_done` true), `problem_info` (language instruction), `macros_image_convention` = `opengl`, `num_demos`, `total`, `bddl_file_name`, `env_name`, `tag` |
| Gripper values in `actions[:, 6]` | {−1, +1}; only −1 in "open the middle drawer of the cabinet" and "push the plate to the front of the stove" |

## LeRobot episodes and raw demos

- **No-op rule.** A raw step is a no-op when ‖a[:6]‖ < 1e-4 and a[6] equals the gripper value of the previous *kept* step. For the first step, ‖a[:6]‖ < 1e-4 alone makes it a no-op.
- **Match.** After no-op removal, every one of the 433 LeRobot episodes equals exactly one raw demo of its task (no ambiguous matches). The two have the same length and equal `action[:6]` after a float32 cast. The largest float64 difference is 2.9e-8, which is float32 rounding.
- **Coverage.** Matched episodes per dataset task 0–9: 42, 33, 39, 50, 50, 47, 39, 38, 45, 50, i.e. 433 of the 500 raw demos.
- **Order.** Within a task, the episode order is the raw demo names sorted as strings: `demo_0`, `demo_1`, `demo_10`, `demo_11`, …
- **Gripper mapping.** `lerobot = (1 − raw) / 2`: raw −1 (open) → 1, raw +1 (close) → 0, and back `raw = 1 − 2 · lerobot`. It has 0 mismatches over all matched steps; the opposite mapping mismatches 4024–6725 steps per task.

## Reproducing a dataset episode

LIBERO samples the fixture poses at every `reset()` from the global numpy RNG. The fixtures are the root bodies of the cabinet, flat stove and wine rack, stored as `model.body_pos` and `model.body_quat`. These poses are not part of the flattened state, so `set_init_state` does not restore them. The dataset episode of raw `demo_k` uses the fixture poses of the (k+1)-th `reset()` after `env.seed(0)` on a freshly constructed env of that task. Checked for demo_0 and demo_1 in the script run, and for demo_2 and demo_10 of task 0 (best of 40–52 candidate resets: resets 3 and 11). The demo's own `model_file` poses do not match: agent-view MSE at t = 0 is 143–257, against about 40 for the matching reset in the same comparison (rendered with mujoco 3.3.7). Placements can be computed without rendering in `ControlEnv(bddl_file_name, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)`, which yields the same poses as the render env.

Recipe:

1. Compute the fixture poses of reset k+1 after `seed(0)`.
2. In the render env, `reset()`, then write those poses.
3. `set_init_state(states[0])` of the raw demo.
4. Take 10 no-op steps `[0, 0, 0, 0, 0, 0, -1]`.
5. Step the LeRobot actions with gripper `1 − 2 · a[6]`.

The observation before action t corresponds to dataset frame t.

| Measured | Episode 0 (task 0, demo_0, T = 138) | Episode 115 (task 3, demo_1, T = 88) |
|---|---|---|
| max \|`joint_state` − `robot0_joint_pos`\|, t = 0, no settle steps | 1.3e-2 | 3.5e-2 |
| same, 10 settle steps, t = 0 / all t | 2.4e-6 / 3.3e-5 | 4.0e-6 / 5.6e-6 |
| max \|`gripper_state` − `robot0_gripper_qpos`\|, all t | 2.6e-4 | 2.6e-4 |
| agent-view MSE at t = 0 (rotated): matching fixtures / next reset's fixtures | 5.9 / 364.4 | 5.6 / 163.2 |
| first `done` (action index) / `done` after the last action | 126 / True | 74 / True |

So the dataset's `observation.states.joint_state` is `robot0_joint_pos` and `observation.states.gripper_state` is `robot0_gripper_qpos` at the same step. Dataset episodes continue after the first successful step: 11 more actions in episode 0, 13 in episode 115.

The table was measured with robosuite's gripper action accumulator (`gripper.current_action`) saturated open, carried over from earlier resets of the same env (ISSUE-008). If the accumulator is zero before the settle steps, as `LiberoTaskEnv.reset` sets it since commit `aba2587`, the replay is much closer to the dataset (phase 12 dev check, `tmp/runlogs/p12/dev/replay_acc.py`):

| Episode | accumulator at start | joint max, all t | joint median | gripper max | `done` after the last action |
|---|---|---|---|---|---|
| 0 | zero / saturated open | 1.3e-7 / 3.3e-5 | 5.5e-8 / 2.2e-6 | 2.9e-9 / 2.6e-4 | True / True |
| 4 | zero / saturated open | 1.2e-7 / 1.1e-5 | 5.0e-8 / 6.5e-6 | 2.1e-9 / 2.6e-4 | True / True |
| 128 | zero / saturated open | 1.2e-7 / 2.4e-4 | 7.1e-8 / 2.0e-4 | 1.8e-9 / 2.6e-4 | True / True |
| 303 | zero / saturated open | 1.5e-4 / 1.5e-3 | 8.0e-8 / 4.5e-6 | 1.6e-4 / 2.6e-4 | True / True |

The dataset is therefore consistent with a re-simulation from raw `states[0]` with the accumulator starting at zero. Episode 303 still deviates by up to 1.5e-4 at some steps.

## Restoring a dataset frame

`LiberoTaskEnv` is the only module that writes simulator state. Its public API (`ceam/sim/libero_task_env.py`):

| Method | What it does |
|---|---|
| `reset(init_index, settle_steps)` | Benchmark init state (unchanged). |
| `reset_sampled(settle_steps)` | New placement from LIBERO's reset sampling (see below). |
| `reset_to_state(sim_state, fixture_poses, settle_steps)` | `reset()`, fixture poses written, `set_init_state(sim_state)`, gripper accumulator zero, settle steps. |
| `snapshot() -> SimSnapshot` | Flattened MuJoCo state, gripper accumulator, controller `goal_ori`, fixture poses. |
| `restore(snapshot, sim_action)` | See the sequence below. |
| `execute(actions)` | Stops at success. |
| `rollout(actions) -> RolloutResult` | Every action, no stop at success; `first_success_step`. |
| `observe()` | The single observation conversion. |
| `fixture_poses_after_resets(seed, resets)` | Computed in a GL-free `ControlEnv`. |
| `to_sim_actions(actions)` | Dataset gripper g → sim 1 − 2g. |

`restore(snapshot, sim_action)` runs these steps in order:

1. `reset()`;
2. write the fixture poses;
3. `set_init_state`;
4. set the accumulator and `goal_ori`;
5. set `controller.new_update = True`;
6. execute `sim_action` as the producing step and return its observation.

`libero_env` remains a read-only property (`SceneProbe`, controller input range).

Raw `states[t_raw]` do not reproduce the dataset frames. Measured on 9 frames of episodes 4, 128 and 303:

- proprio error 1.3e-2 to 6.1e-2 rad;
- wrist-view token cosine 0.843–0.928;
- ‖ψ_sim − ψ_cache‖ 0.24–1.62.

`DemoStateRestorer` (`ceam/sim/demo_state_restorer.py`) therefore:

1. replays the dataset episode once: `reset_to_state` with raw `states[0]` and the matched fixture poses, then the recipe above with the accumulator at zero;
2. before the step that produces each frame t, records `snapshot()`;
3. restores frame t with `env.restore(snapshot, producing action)`.

Controller and observation behaviour the restore has to handle:

- After `reset()` + `set_init_state`, the OSC controller's first `set_goal` starts from the end-effector pose cached at reset, unless `new_update` is set. Without it, the first step after restoring is off by about 2e-2 rad in proprio.
- `goal_ori` carries over from the previous step when a step's rotation delta is exactly zero, as in stay chunks.
- Each forced observable update (one in `reset()`, one in `set_init_state`) advances robosuite's observable sampling timer by one 2 ms physics substep:
  - `reset()` + `set_init_state` samples observations one substep before the step ends;
  - `reset()` alone samples at the last substep.

  `reset_sampled` therefore calls `set_init_state(get_sim_state())` after `reset()`: the scene is unchanged and the sampling phase equals that of benchmark resets, the dataset replay and `restore`.

Fidelity over 129 restored frames of the HELDOUT split (`docs/dynamics_counterfactuals.md`):

- max proprio error 2.2e-5;
- mean DINOv2 token cosine: agent 0.969, wrist 0.945;
- after executing the demo chunk, proprio ≤ 1.6e-5 against frame t + 10.

Rendering is not fully deterministic from run to run:

- 125–128 of 129 samples give bit-identical repeated restores in a run; the exceptions are mostly task 7.
- Rerunning the Phase 12 check changes image-derived numbers by up to 2e-4 (29 of 1806 candidates), with identical proprio, ranks and choices. This holds for the code before and after the interface change.

A snapshot taken on a `reset_sampled` trajectory restores bit-exactly (50 of 50).

Timing on L40S at 512² with two cameras:

| Operation | Time |
|---|---|
| `restore` | median 40 ms |
| `rollout` of 10 actions | 189 ms |
| `reset_sampled(10)` | 187–190 ms |
| `snapshot` | 0.1 ms |

## Sampled initial states

`reset_sampled(settle_steps)` runs `env.reset()`, which samples object placements and fixture poses from the global numpy RNG, then `set_init_state(get_sim_state())`, a zero gripper accumulator and the settle steps.

- **Deterministic**: for a given env seed and call sequence, 50 of 50 resets were bit-identical across processes (seed 1 twice). Seed 2 differs: objects median 2.6 cm, fixtures up to 1.9 cm.
- **Global RNG**: `fixture_poses_after_resets` re-seeds the global numpy RNG, so it must not be interleaved with a sampled sequence.
- **At rest**: after 10 settle steps, max |qvel| is 7e-13 (robot) and 3.2e-3 (everything else). The goal never holds at the start.
- **In distribution**: start observations are as close to the dataset starts as benchmark inits are:

| Start | agent token cosine to the task's dataset starts | wrist token cosine | ψ distance to nearest dataset start |
|---|---|---|---|
| sampled | median 0.9395 | median 0.8703 | median 8.02 |
| benchmark inits 0–4 | median 0.9405 | median 0.8621 | median 7.48 |
| dataset starts, pairwise | 0.961–0.963 | 0.879–0.889 | — |

- **Reset count only**: sampled start e depends only on the env seed and on the number of resets before it, not on the steps taken in between. Reset-only calls reproduced the rollout starts bit-exactly (task 3: 100/100; task 0: 98/100, the other 2 within render noise). `run_sampled` still requires episodes 0, 1, 2, … in order in one process.
- **Robot pose**: the robot's start proprio is identical across all 1000 sampled starts (max difference 1.3e-14); only objects and fixtures vary.
- **node-61 EGL mapping**: CUDA 0–7 map to EGL devices 3, 2, 1, 0, 7, 6, 5, 4.

## Image orientation

A dataset frame is the env observation image rotated by 180°, `image[::-1, ::-1]`, for both cameras. robosuite's `IMAGE_CONVENTION` is `opengl`, so the observation arrays are stored bottom-up. MSE between the sim observation and the decoded dataset frame at the same t, in the reproduced episodes:

| Episode, t, view | as rendered | vertical flip | horizontal flip | rotate 180° |
|---|---|---|---|---|
| 0, t = 0, agent / wrist | 6857 / 8821 | 5806 / 6236 | 4765 / 9274 | **5.9 / 10.0** |
| 0, t = 69, agent / wrist | 6998 / 11262 | 6131 / 3179 | 4643 / 12477 | **7.8 / 10.7** |
| 0, t = 127, agent / wrist | 8859 / 7246 | 8296 / 1907 | 4919 / 6992 | **7.9 / 11.8** |
| 115, t = 0, agent / wrist | 6873 / 8317 | 5383 / 5927 | 4564 / 8946 | **5.6 / 9.2** |
| 115, t = 44, agent / wrist | 6818 / 9594 | 5744 / 3934 | 4538 / 7709 | **5.8 / 7.7** |
| 115, t = 77, agent / wrist | 7109 / 10773 | 6021 / 3055 | 4738 / 9381 | **8.7 / 14.4** |

## Done and success

`env.step` returns `done = env.check_success()`; LIBERO overrides robosuite's horizon-based flag. The two agree at every step of the 20 raw replays below. Stepping continues after success.

Raw `demo_0` replays of each task used `ControlEnv(..., has_offscreen_renderer=False, use_camera_obs=False)`: `seed(0)`, `reset()`, `set_init_state(states[0])`, then every raw action including no-ops, without settle steps. Fixtures came either from that reset or from the demo's `model_file`:

| Dataset task | Steps | First `done` step (reset fixtures / `model_file` fixtures) | `done` after last action |
|---|---|---|---|
| 0 open the middle drawer of the cabinet | 138 | 126 / 126 | True / True |
| 1 open the top drawer and put the bowl inside | 170 | 158 / 158 | True / True |
| 2 push the plate to the front of the stove | 155 | 142 / 142 | True / True |
| 3 put the bowl on the plate | 90 | never / never | False / False |
| 4 put the bowl on the stove | 94 | 82 / 82 | True / True |
| 5 put the bowl on top of the cabinet | 91 | 78 / 78 | True / True |
| 6 put the cream cheese in the bowl | 92 | 75 / 75 | True / True |
| 7 put the wine bottle on the rack | 347 | 250 / 134 | True / True |
| 8 put the wine bottle on top of the cabinet | 93 | 81 / 82 | True / True |
| 9 turn on the stove | 80 | 79 / 69 | True / True |

Every `demo_0` matches a LeRobot episode; task 3's is episode 114.

## Frozen-feature similarity

`DinoV2VisionEncoder` (`facebook/dinov2-base`, pooled grid 8, the cache's settings) encodes the rotated sim observations. The table gives the cosine similarity of the flattened [64 × 768] tokens against the cached tokens of the same (episode, t). The two references are the cosine between cached frames t and t+1, and between t and t+10, of the same episode.

| Episode, t | Agent: sim vs cache | Agent: cache t+1 / t+10 | Wrist: sim vs cache | Wrist: cache t+1 / t+10 |
|---|---|---|---|---|
| 0, 0 | 0.964 | 0.999 / 0.968 | 0.956 | 0.988 / 0.818 |
| 0, 69 | 0.963 | 0.978 / 0.915 | 0.952 | 0.935 / 0.683 |
| 0, 127 | 0.965 | 0.976 / 0.906 | 0.945 | 0.922 / 0.587 |
| 115, 0 | 0.968 | 1.000 / 0.969 | 0.950 | 0.988 / 0.655 |
| 115, 44 | 0.966 | 0.999 / 0.960 | 0.960 | 0.979 / 0.692 |
| 115, 77 | 0.965 | 0.986 / 0.972 | 0.942 | 0.934 / 0.837 |

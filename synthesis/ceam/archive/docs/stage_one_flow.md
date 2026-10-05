# Stage one: per-frame interpolation (flow) target

Each demonstration is a segment from its first state to its own end state g = ψ(o_T), and frame t's target is the point at t/T on that segment. The fixed stage one (`docs/stage_one.md`) constrains only the component of z_t − Uᵀsg[ψ₀] perpendicular to z_T − z₀ in the query coordinates z = Uᵀψ, so the position along the segment is free. This page records the interpolation target in two spaces, two stage-one runs retrained with it, and their offline comparison with the fixed stage one:

- **flow full (ψ)**: the interpolation target on the full 256-d state ψ; the primary variant.
- **flow (z)**: the same target in the query coordinates z = Uᵀψ; a control.
- **flow full (ψ), SIGReg on all frames**: flow full (ψ) with SIGReg on every frame of the batch instead of the uniform frames only (section "SIGReg on all frames").
- **flow full (ψ), SIGReg on all frames, sg[D_ψ]**: the previous variant with the normalizer D_ψ of the path term under stop-gradient (section "Stop-gradient on D_ψ").

Everything else is the fixed stage one. The step measures below show whether the target is realized; they do not rank the variants. The other diagnostics are sanity checks against the fixed stage one.

## Change

| `PathTarget` | Path term per uniform frame, averaged over frames and episodes | Runs |
|---|---|---|
| `PERPENDICULAR` | ‖(I − nnᵀ)(z_t − Uᵀsg[ψ₀])‖² / D, n = sg[(z_T − z₀)/√D] | fixed |
| `INTERPOLATION` | ‖z_t − ((1 − t/T)·Uᵀsg[ψ₀] + (t/T)·Uᵀsg[ψ_T])‖² / D | flow (z) |
| `INTERPOLATION_FULL` | ‖ψ_t − ((1 − t/T)·sg[ψ₀] + (t/T)·sg[ψ_T])‖² / D_ψ | flow full (ψ) |

- **Normalization**: D = ‖z_T − z₀‖² + ε and D_ψ = ‖ψ_T − ψ₀‖² + ε, both with gradients unless `path_normalizer_gradient` is false (section "Stop-gradient on D_ψ"); ε = 1e-6. ψ₀ and ψ_T enter each term only through its denominator. The full-space term does not involve U.
- **Replacement**: each interpolation residual contains the perpendicular residual of its space. The chosen term takes the place of the path term with the same weight 1.
- **t/T**: t is the index of the uniform frame and T the index of the episode's last frame, the frame of `end`. t ∈ {0, …, T − 1}, so t/T ∈ [0, 1).
- **Unchanged**:
  - SIGReg on the uniform frames; goal, endpoint, paraphrase and instruction contrast in their existing definitions (goal and endpoint in z);
  - all weights 1;
  - model, optimizer, 5000 steps, B = 64, M = 8, seed 0.

## Interfaces

| Interface | File | Change |
|---|---|---|
| `PathTarget` | `ceam/losses/path_loss.py` | enum: `PERPENDICULAR` = "perpendicular", `INTERPOLATION` = "interpolation", `INTERPOLATION_FULL` = "interpolation_full" |
| `PathLoss(target, epsilon).forward(projected, states)` | `ceam/losses/path_loss.py` | the target selects the term above; `epsilon` is used for D_ψ; the `PERPENDICULAR` branch is the previous code |
| `StageOneObjective(weights, path_target, sigreg_frames, state_dim, epsilon, negative_tasks)` | `ceam/losses/stage_one_objective.py` | builds `PathLoss(path_target, epsilon)` and passes it the projected episodes and the states; `sigreg_frames` is described in "SIGReg on all frames" |
| `StageOneBatch.uniform_fraction` | `ceam/data/stage_one_batch.py` | t/T of every uniform frame, [B, M] float32 |
| `StageOneSampler.sample` | `ceam/data/stage_one_sampler.py` | uniform_fraction = uniform_t / (length − 1) from the existing draws; the random stream is unchanged |
| `StageOneStates.uniform_fraction` | `ceam/losses/stage_one_states.py` | set from the batch by `StageOneTrainer._states` |
| `ProjectedEpisodes.uniform_fraction` | `ceam/losses/projected_episodes.py` | detached float32 copy made by `EpisodeProjector.project` |
| `StageOneConfig.path_target` | `ceam/training/stage_one_config.py` | see below |
| `ProgressReport` step measures | `ceam/diagnostics/progress_metrics.py` | see below |

`StageOneConfig.path_target`:

- It is optional when read (`OPTIONAL_FIELDS`; missing = `PERPENDICULAR`) and always written as the value string.
- Configs and checkpoints written before the field existed load unchanged.
- Code from before the field rejects configs that contain it, including the configs embedded in the flow checkpoints.

`ProgressReport` step measures:

- Per episode:
  - mean_t |p_t − t/T|;
  - p at t/T = 0.1 and 0.25, linear between frames;
  - whether p_t > 0.8 at some t/T < 0.2.
- Summaries: medians, p10 and p90, and the share of early-completing episodes.
- Spaces: computed for z (fields without prefix) and for ψ (`state_*`, p_t = ⟨ψ_t − ψ₀, n_ψ⟩ / ‖ψ_T − ψ₀‖).
- Printed by `scripts/diagnose_stage_one.py`.

Configs:

- `configs/stage_one/goal_instruction_contrast_flow_full.json` sets `"path_target": "interpolation_full"`, with `run_dir` = `$CEAM_ROOT/outputs/runs/goal_instruction_contrast_flow_full`.
- `configs/stage_one/goal_instruction_contrast_flow.json` sets `"path_target": "interpolation"`, with `run_dir` = `$CEAM_ROOT/outputs/runs/goal_instruction_contrast_flow`.
- Each differs from `goal_instruction_contrast.json` only in these two keys.

## Checks

### Loss values (`scripts/check_losses.py`, CPU, seed 0, ε = 1e-6)

- **Unchanged outputs**: every `PERPENDICULAR` value and every other term prints the same numbers as before the change.
- **Frames on the line**: frames sit on the line from start to end at position g, with uniform_fraction f uniform in [0, 1); U is random with r = 8, d = 256. The reference, in float64, is mean ((g − f)² + c²)(D − ε)/D for a perpendicular offset c‖end − start‖.

| Frames | z: PERPENDICULAR | z: INTERPOLATION | ψ: INTERPOLATION_FULL | reference |
|---|---|---|---|---|
| g = f (on the segment at t/T) | 6.0e-14 | 6.4e-14 | 1.1e-15 | 0 |
| g = f + 0.2 | 6.7e-14 | 0.0400 | 0.0400 | 0.0400 |
| g = 0 for f < 0.9, 1 after (late step) | 7.2e-14 | 0.2480 | 0.2598 | 0.2480 / 0.2598 |
| g = 1 − f (start and end swapped) | 6.2e-14 | 0.3264 | 0.3413 | 0.3264 / 0.3413 |
| g = f plus a perpendicular offset c = 0.1 | 0.0100 (z offset) | 0.0100 (z offset) | 0.0100 (ψ offset) | 0.0100 |

The z and ψ columns use different random draws, hence the two references.

- **Isotropic noise**: exact frames plus isotropic ψ noise of std σ.
  - INTERPOLATION_FULL: 5.00e-5 at σ = 0.01 and 4.96e-3 at σ = 0.1, against d σ² mean(1/D_ψ) = 4.99e-5 and 4.99e-3.
  - INTERPOLATION: 5.75e-5 and 6.30e-3, against r σ² mean(1/D) = 6.07e-5 and 6.07e-3.
- **Scale invariance**: scaling every ψ and ĝ by 0.1 or 10 changes INTERPOLATION_FULL by ≤ 2.1e-7 relative and INTERPOLATION by ≤ 1.2e-5 relative (from ε).
- **Gradients**:
  - INTERPOLATION_FULL reaches ψ₀, ψ_T and the uniform ψ, but not U.
  - With ψ₀ and ψ_T detached, only the uniform ψ receives a gradient.
  - With D detached, ψ₀ and ψ_T have no path to INTERPOLATION.
  - t/T has no autograd path.
- **Objective**: `StageOneObjective` returns float32 terms for float32 inputs, bf16 inputs, and bf16 inputs under autocast, for all three targets; the two bf16 cases agree exactly.

### Direction of the target and t/T

- **Code**: the start state has weight 1 − f and the end state weight f.
  - `path_loss.py`, INTERPOLATION: `target = (1.0 - fraction) * projected.start_target[:, None, :] + fraction * projected.end_target[:, None, :]`.
  - `path_loss.py`, INTERPOLATION_FULL: `target = (1.0 - fraction) * start.detach()[:, None, :] + fraction * end.detach()[:, None, :]`, with `start = states.start`, `end = states.end`.
  - `projected_episodes.py`: `start_target` = Uᵀ `states.start.detach()` and `end_target` = Uᵀ `states.end.detach()`.
- **Sampler** (`stage_one_sampler.py`):
  - `uniform_t = rng.integers(0, anchor_lengths[:, None] - 1, ...)` draws t ∈ {0, …, length − 2}.
  - `end` is gathered at `anchor_lengths[:, None] - 1`.
  - `uniform_fraction = (uniform_t / (anchor_lengths[:, None] - 1)).astype(np.float32)`.
- **Sampled frames**: the first TRAIN batch of the flow-full config (seed 0). Frame indices are identified by matching the batch's frames against the feature cache.

| Episode | Length | Frame of `end` | Uniform t | uniform_fraction | t / (length − 1) | Weight on ψ₀ |
|---|---|---|---|---|---|---|
| 362 | 97 | 96 | 91 | 0.947917 | 0.947917 | 0.052083 |
| 362 | 97 | 96 | 34 | 0.354167 | 0.354167 | 0.645833 |
| 260 | 110 | 109 | 89 | 0.816514 | 0.816514 | 0.183486 |
| 260 | 110 | 109 | 25 | 0.229358 | 0.229358 | 0.770642 |
| 297 | 95 | 94 | 86 | 0.914894 | 0.914894 | 0.085106 |
| 297 | 95 | 94 | 91 | 0.968085 | 0.968085 | 0.031915 |

- **Full batches**: over three whole batches, uniform_fraction matches t / (length − 1) of the replayed draws to 3e-8 (float32), and the horizon draws are unchanged.
- **Configs**:
  - Every file in `configs/stage_one/` and `outputs/runs/goal_instruction_contrast/config.json` loads as PERPENDICULAR and round-trips exactly.
  - `CheckpointStore.load` of the fixed checkpoint works.

### Bit identity of PERPENDICULAR

- **Setup**: the fixed config, shortened to 300 steps and logging every 10 steps, has no `path_target` key.
  - It was trained with the code before the change (`2b64f9f`), then with `a17b920` and with `ad9ab4e`.
  - All runs used the same GPU (node-61, CUDA 0, L40S).
- **Logged fields**: all 31 logged steps are bit-identical in every loss, lr, grad norm and logit scale.
- **Peak memory**: it is 2048 bytes higher, the size of the [64, 8] float32 uniform_fraction tensor.
- **Checkpoint**: all 112 parameter tensors at step 300 are bitwise equal.
- **Recorded run**: steps 0, 50, 100, 150 and 200 lie in the shared warmup, so they are the same steps as in the 5000-step schedule. They are bit-identical to the recorded fixed run (node-70, L40S), e.g. total at step 0 is 8.021472930908203.
- **Diagnostics**: the rerun of the fixed checkpoint reproduces all 117 recorded fields of its report exactly.

## Runs

| Run | Config | Commit | Host / GPU | Time | Speed | Peak memory |
|---|---|---|---|---|---|---|
| fixed | `goal_instruction_contrast.json` | recorded run | node-70, L40S | – | 2.1 steps/s | 12.86 GiB |
| flow (z) | `goal_instruction_contrast_flow.json` | `a17b920` | node-61, CUDA 1, L40S | 1810 s | 2.76 steps/s | 12.86 GiB |
| flow full (ψ) | `goal_instruction_contrast_flow_full.json` | `ad9ab4e` | node-61, CUDA 3, L40S | 2015 s | 2.48 steps/s | 12.86 GiB |

Checkpoints:

- `outputs/runs/goal_instruction_contrast_flow_full/checkpoints/step_005000.pt`;
- `outputs/runs/goal_instruction_contrast_flow/checkpoints/step_005000.pt`;
- both runs also saved steps 1000–4000.

Outputs:

- **Stdout**: `train.log` in each run directory.
- **Diagnostics of each checkpoint**: `<run_dir>/diagnostics/step_00N000/` (`report.json`, `progress_curves.png`, `subspace_overlap.png`), at commit `ad9ab4e`.
- **Fixed checkpoint at `ad9ab4e`**: `outputs/runs/goal_instruction_contrast_flow_full/diagnostics/fixed_goal_instruction_contrast_step_005000/`.
- **Comparison figures**:
  - `outputs/runs/goal_instruction_contrast_flow_full/training_curves_comparison.png`;
  - `outputs/runs/goal_instruction_contrast_flow_full/diagnostics/progress_comparison_{z,psi}_step_005000.png`. Each has a median and 10–90% band over the 43 HELDOUT episodes, and the first HELDOUT episode of every task.

## Training

Unweighted terms at selected steps; the last row is the mean of the last 11 logged steps (4500–4999). The path column is each run's own path term: perpendicular in z, interpolation in z, and interpolation in ψ respectively.

| Run | step | path | goal | endpoint | SIGReg | paraphrase | instruction contrast | logit scale s | total |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 0 | 0.6312 | 3.667 | 0.6425 | 0.5029 | 0.1574 | 2.421 | 14.29 | 8.021 |
| fixed | 1000 | 1.03e-3 | 1.21e-3 | 5.72e-4 | 0.0359 | 6.39e-4 | 0.1124 | 14.82 | 0.1518 |
| fixed | 3000 | 1.39e-3 | 3.00e-4 | 3.88e-4 | 0.0319 | 3.81e-4 | 0.0452 | 17.44 | 0.0796 |
| fixed | 4500–4999 | 5.36e-4 | 9.48e-5 | 1.64e-4 | 0.0318 | 2.98e-4 | 0.0111 | 18.29 | 0.0440 |
| flow (z) | 0 | 0.7284 | 3.667 | 0.6425 | 0.5029 | 0.1574 | 2.421 | 14.29 | 8.119 |
| flow (z) | 1000 | 3.29e-3 | 4.70e-4 | 7.46e-4 | 0.0375 | 2.36e-4 | 0.0821 | 15.37 | 0.1244 |
| flow (z) | 3000 | 3.65e-3 | 4.67e-4 | 5.80e-4 | 0.0384 | 7.68e-5 | 0.0475 | 18.98 | 0.0907 |
| flow (z) | 4500–4999 | 2.79e-3 | 2.09e-4 | 4.00e-4 | 0.0318 | 6.88e-5 | 0.0113 | 20.07 | 0.0465 |
| flow full (ψ) | 0 | 0.5375 | 3.667 | 0.6425 | 0.5029 | 0.1574 | 2.421 | 14.29 | 7.928 |
| flow full (ψ) | 1000 | 0.1414 | 7.34e-4 | 1.08e-3 | 0.0650 | 2.30e-4 | 0.1808 | 18.20 | 0.3893 |
| flow full (ψ) | 3000 | 0.0818 | 2.14e-4 | 5.48e-4 | 0.0652 | 6.19e-5 | 0.0854 | 25.73 | 0.2332 |
| flow full (ψ) | 4500–4999 | 0.0891 | 1.35e-4 | 2.31e-4 | 0.0526 | 4.15e-5 | 0.0661 | 27.51 | 0.2082 |

- **Step 0**: all terms other than the path term are identical across the three runs; the initialization and first batch are the same.
- **Full-space path term**: it drops to 0.085 by step 250; logged values from step 250 on range over 0.057–0.215 (0.14 at steps 500 and 1000, 0.089 mean over the last 11 logged steps).
- **flow full (ψ), other terms**: SIGReg (0.053) and the instruction contrast (0.066) end higher than in the other two runs, and the logit scale grows to 27.5.

## Offline diagnostics, step 5000

These use `scripts/diagnose_stage_one.py` on the 43 HELDOUT episodes. The action ranking uses k = 10 frames and 506 anchors, with observed outcomes standing in for E(o, a) (`docs/diagnostics.md`). The TRAIN rows use the same code on the 390 TRAIN episodes.

### Progress along the start-to-end direction

Spearman is Spearman(t, p_t); p is linear between frames. Medians over episodes.

| Space | Split | Quantity | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|---|---|
| z | HELDOUT | Spearman median / share of episodes with Spearman > 0 | 1.000 / 1.00 | 0.998 / 1.00 | 0.142 / 0.51 |
| z | HELDOUT | p at t/T = 0.1 / 0.25 / 0.5 / 0.75 | 0.151 / 0.289 / 0.520 / 0.804 | 0.060 / 0.182 / 0.460 / 0.731 | 0.730 / 0.730 / 0.734 / 0.722 |
| z | TRAIN | Spearman median / share > 0 | 1.000 / 1.00 | 1.000 / 1.00 | −0.003 / 0.50 |
| z | TRAIN | p at t/T = 0.1 / 0.25 / 0.5 / 0.75 | 0.185 / 0.308 / 0.537 / 0.829 | 0.108 / 0.261 / 0.499 / 0.752 | 0.733 / 0.736 / 0.729 / 0.712 |
| ψ | HELDOUT | Spearman median / share > 0 | 0.997 / 1.00 | 0.999 / 1.00 | 0.795 / 0.95 |
| ψ | HELDOUT | p at t/T = 0.1 / 0.25 / 0.5 / 0.75 | 0.114 / 0.181 / 0.343 / 0.685 | 0.231 / 0.311 / 0.484 / 0.781 | 0.628 / 0.632 / 0.638 / 0.643 |
| ψ | TRAIN | Spearman median / share > 0 | 1.000 / 1.00 | 1.000 / 1.00 | 0.794 / 0.99 |
| ψ | TRAIN | p at t/T = 0.1 / 0.25 / 0.5 / 0.75 | 0.111 / 0.192 / 0.370 / 0.703 | 0.227 / 0.369 / 0.519 / 0.797 | 0.626 / 0.625 / 0.631 / 0.640 |

Step measures on HELDOUT, with p10 / p90 in parentheses:

| Space | Quantity | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|---|
| z | mean_t \|p_t − t/T\| | 0.089 (0.038 / 0.137) | 0.055 (0.026 / 0.170) | 0.291 (0.253 / 0.433) |
| z | p at t/T = 0.1 | 0.151 (0.086 / 0.235) | 0.060 (−0.156 / 0.118) | 0.730 (0.114 / 0.975) |
| z | p at t/T = 0.25 | 0.289 (0.167 / 0.447) | 0.182 (−0.014 / 0.290) | 0.730 (0.163 / 0.936) |
| z | episodes with p_t > 0.8 at some t/T < 0.2 | 0 of 43 | 0 of 43 | 10 of 43 |
| ψ | mean_t \|p_t − t/T\| | 0.102 (0.063 / 0.138) | 0.077 (0.048 / 0.132) | 0.242 (0.203 / 0.429) |
| ψ | p at t/T = 0.1 | 0.114 (0.004 / 0.233) | 0.231 (0.089 / 0.318) | 0.628 (0.161 / 0.902) |
| ψ | p at t/T = 0.25 | 0.181 (0.100 / 0.291) | 0.311 (0.181 / 0.460) | 0.632 (0.276 / 0.897) |
| ψ | episodes with p_t > 0.8 at some t/T < 0.2 | 0 of 43 | 0 of 43 | 6 of 43 |

Over training (HELDOUT, z): mean_t |p_t − t/T| / p at 0.1 / p at 0.25.

| step | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|
| 1000 | 0.127 / 0.27 / 0.46 | 0.044 / 0.15 / 0.29 | 0.360 / 0.88 / 0.88 |
| 2000 | 0.108 / 0.26 / 0.41 | 0.035 / 0.10 / 0.26 | 0.311 / 0.81 / 0.81 |
| 3000 | 0.076 / 0.15 / 0.26 | 0.040 / 0.10 / 0.25 | 0.280 / 0.76 / 0.76 |
| 4000 | 0.088 / 0.15 / 0.30 | 0.045 / 0.08 / 0.21 | 0.291 / 0.74 / 0.75 |
| 5000 | 0.089 / 0.15 / 0.29 | 0.055 / 0.06 / 0.18 | 0.291 / 0.73 / 0.73 |

### Geometry of ψ (HELDOUT, all frames of every episode)

"Interior" means frames 1 … T − 1 of every episode, and the "interior mean" is their mean over all episodes. Values are medians over episodes, or over frames for the interior rows.

| Quantity | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|
| effective rank of cov(ψ), all frames / interior frames | 16.6 / 16.5 | 16.9 / 16.8 | 1.14 / 1.14 |
| share of variance on the top eigenvector, all / interior | 0.115 / 0.116 | 0.125 / 0.126 | 0.981 / 0.980 |
| ‖ψ₀ − interior mean‖ | 15.5 | 18.8 | 714 |
| ‖ψ_T − interior mean‖ | 30.4 | 24.5 | 457 |
| ‖ψ_t − interior mean‖, interior frames | 14.6 | 15.7 | 39.5 |
| ‖ψ_T − ψ₀‖ | 35.0 | 30.8 | 1156 |
| ‖ψ₁ − ψ₀‖ / ‖ψ_T − ψ_{T−1}‖ | 1.35 / 0.49 | 1.44 / 0.39 | 3.69 / 2.29 |
| ‖ψ_{t+1} − ψ_t‖, interior frames | 0.45 | 0.45 | 0.57 |
| per-dimension std of ψ, min / median / max | 0.74 / 1.11 / 1.69 | 0.56 / 1.03 / 1.65 | 1.56 / 8.77 / 34.51 |
| SIGReg statistic (4096 frames) | 0.0254 | 0.0302 | 0.0411 |

In flow full (ψ), p_t in ψ rises from 0 to about 0.63 within the first few frames, stays there, and rises to 1 within the last few frames (`progress_comparison_psi_step_005000.png`). Those are the medians over episodes; single episodes plateau elsewhere, e.g. near 0.05 (task 5, episode 223) or above 1 (task 9, episode 384). A plateau at 0.63 has along-segment interpolation residual ∫₀¹ (0.63 − f)² df = 0.100; the logged path term at the end of training is 0.089–0.099.

### Action ranking by t/T quarter

Fixed / flow (z) / flow full (ψ) in each cell.

| Quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | pooled |
|---|---|---|---|---|---|
| P(demo > stay) | 0.922 / 0.922 / 0.902 | 0.945 / 0.922 / 0.953 | 0.962 / 0.969 / 0.969 | 0.958 / 0.958 / 0.947 | 0.945 / 0.941 / 0.941 |
| P(demo > reverse) | 0.945 / 0.936 / 1.000 | 0.984 / 0.984 / 0.969 | 0.977 / 0.992 / 0.985 | 0.968 / 0.958 / 0.937 | 0.970 / 0.970 / 0.974 |
| P(demo > other-task action) | 0.960 / 0.970 / 0.975 | 0.974 / 0.945 / 0.953 | 0.963 / 0.969 / 0.978 | 0.959 / 0.956 / 0.946 | 0.964 / 0.960 / 0.964 |
| instruction top-1 (chance 0.1) | 0.758 / 0.719 / 0.654 | 0.844 / 0.867 / 0.859 | 0.962 / 0.969 / 0.969 | 0.958 / 0.958 / 0.937 | 0.870 / 0.866 / 0.840 |
| instruction pairwise | 0.916 / 0.887 / 0.883 | 0.954 / 0.928 / 0.950 | 0.962 / 0.969 / 0.972 | 0.958 / 0.958 / 0.944 | 0.945 / 0.932 / 0.934 |
| sensitivity (median std of candidate matches) | 0.156 / 0.139 / 0.124 | 0.187 / 0.160 / 0.170 | 0.191 / 0.166 / 0.211 | 0.204 / 0.184 / 0.199 | 0.186 / 0.160 / 0.173 |
| effective rank of demo effects | 9.57 / 10.80 / 1.01 | 7.87 / 8.03 / 1.47 | 10.16 / 9.60 / 1.18 | 9.03 / 9.56 / 1.25 | 15.07 / 14.83 / 1.05 |

Anchors per quarter: 153 / 128 / 130 / 95, pooled 506. One anchor is 0.65 points of a first-quarter P(demo > stay) or top-1 cell.

### Whole-model quantities

| Quantity | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|
| shared start P(demo > other-task first chunk) | 0.866 | 0.804 | 0.800 |
| effective rank of cov(ψ) | 16.6 | 16.9 | 1.1 |
| task-ID probe HELDOUT, ψ(o_T) | 1.000 | 1.000 | 0.884 |
| task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 1.000 | 1.000 | 0.895 |
| task-ID probe HELDOUT, ψ(o₀) control | 0.070 | 0.140 | 0.186 |
| displacement ratio ‖z_T − z₀‖ / RMS cross-episode | 1.65 | 2.86 | 2.84 |
| path deviation max_t ‖(I − nnᵀ)(z_t − z₀)‖ / ‖z_T − z₀‖ | 0.056 | 0.034 | 0.076 |
| goal error ‖Uᵀĝ − z_T‖ / ‖z_T − z₀‖ | 0.0082 | 0.0143 | 0.0097 |

One HELDOUT episode is 2.3 points of a probe accuracy.

### Query subspace statistics (medians over HELDOUT anchors or pairs)

| Quantity | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|
| subspace distance, paraphrase pairs | 1.25e-4 | 3.20e-5 | 1.12e-5 |
| subspace distance, different-task pairs | 0.256 | 0.162 | 0.096 |
| task overlap ‖U_iᵀU_j‖²/r, off-diagonal mean (min–max) | 0.749 (0.670–0.851) | 0.841 (0.799–0.921) | 0.906 (0.864–0.963) |
| endpoint distance, same task / ‖z_T − z₀‖ | 0.0116 | 0.0246 | 0.0149 |
| endpoint distance, other task / ‖z_T − z₀‖ | 1.150 | 0.994 | 0.300 |
| ‖U_aᵀ(ĝ_a − ψ_T)‖²/D_a, own endpoint | 6.70e-5 | 2.05e-4 | 9.50e-5 |
| same, same-task endpoints | 1.22e-4 | 4.71e-4 | 7.97e-4 |
| same, other-task endpoints | 1.421 | 1.097 | 0.117 |
| other-task / own ratio | 2.1e4 | 6.0e3 | 1.4e3 |
| fraction of other-task endpoints closer to ĝ_a than own | 0 | 0 | 0 |

### Over training (HELDOUT, pooled)

Each cell gives: P(demo > stay) / P(demo > other task) / top-1 / first-quarter top-1 / shared start / effective rank of demo effects / effective rank of ψ.

| step | fixed | flow (z) | flow full (ψ) |
|---|---|---|---|
| 1000 | 0.947 / 0.967 / 0.854 / 0.719 / 0.821 / 8.8 / 13.7 | 0.976 / 0.970 / 0.850 / 0.693 / 0.906 / 8.6 / 13.9 | 0.941 / 0.962 / 0.826 / 0.634 / 0.652 / 1.1 / 1.4 |
| 2000 | 0.980 / 0.966 / 0.885 / 0.797 / 0.924 / 13.4 / 14.6 | 0.980 / 0.966 / 0.860 / 0.745 / 0.923 / 11.4 / 15.1 | 0.947 / 0.963 / 0.862 / 0.699 / 0.624 / 1.1 / 1.2 |
| 3000 | 0.974 / 0.970 / 0.864 / 0.745 / 0.879 / 13.5 / 15.7 | 0.980 / 0.972 / 0.868 / 0.732 / 0.879 / 12.7 / 16.4 | 0.949 / 0.965 / 0.834 / 0.647 / 0.749 / 1.1 / 1.1 |
| 4000 | 0.951 / 0.968 / 0.874 / 0.758 / 0.862 / 14.6 / 16.4 | 0.947 / 0.963 / 0.870 / 0.745 / 0.816 / 14.5 / 16.7 | 0.939 / 0.964 / 0.834 / 0.654 / 0.775 / 1.1 / 1.1 |
| 5000 | 0.945 / 0.964 / 0.870 / 0.758 / 0.866 / 15.1 / 16.6 | 0.941 / 0.960 / 0.866 / 0.719 / 0.804 / 14.8 / 16.9 | 0.941 / 0.964 / 0.840 / 0.654 / 0.800 / 1.1 / 1.1 |

## Reading

### flow full (ψ)

- **Target not realized**: the interpolation target in ψ is not realized on TRAIN or HELDOUT.
  - Along the start-to-end direction in ψ, interior frames sit on a plateau at p ≈ 0.63 from t/T ≈ 0.03 to ≈ 0.95.
  - p at t/T = 0.1, 0.25, 0.5 and 0.75 is 0.63–0.64 on both splits.
  - In z the plateau is at 0.73, and the Spearman correlation with t is about 0 (0.14 HELDOUT, −0.003 TRAIN).
  - The logged path term (0.09–0.10) equals the along-segment residual of such a plateau.
- **Geometry**:
  - The start and end states lie far outside the interior frames: 714 and 457 from the interior mean, against 39.5 for interior frames.
  - ‖ψ_T − ψ₀‖ is 1156, against 35 in the fixed stage one.
  - One direction carries 98% of ψ's variance on HELDOUT frames, with or without the first and last frames.
  - The effective rank is 1.1 for ψ and 1.0–1.5 for demo effects in every quarter; the fixed stage one has 16.6 and 7.9–10.2.
- **Worse than fixed**:
  - First-quarter instruction top-1: 0.654 against 0.758.
  - First-quarter pairwise: 0.883 against 0.916; first-quarter P(demo > stay): 0.902 against 0.922.
  - Pooled top-1: 0.840 against 0.870; shared-start discrimination: 0.800 against 0.866.
  - Task-ID probes from ψ(o_T) and U_aᵀψ(o_T): 0.88–0.90 against 1.000.
  - Query subspaces of different tasks are closer: overlap 0.906 against 0.749, different-task distance 0.096 against 0.256.
  - The normalized squared distance of other-task endpoints to ĝ_a is 12× smaller (0.117 against 1.421), though none is closer than the own endpoint.
  - The instruction contrast ends at 0.066 against 0.011.
- **Better than fixed**:
  - First-quarter P(demo > reverse): 1.000 against 0.945.
  - First-quarter P(demo > other-task action): 0.975 against 0.960.
  - Second-quarter P(demo > stay): 0.953 against 0.945; second-quarter top-1: 0.859 against 0.844.
  - Pooled P(demo > reverse) and P(demo > other-task action) are equal or higher.

### flow (z)

- **Target realized**: the interpolation target in z is realized on TRAIN (p = 0.108 / 0.261 / 0.499 / 0.752 at t/T = 0.1 / 0.25 / 0.5 / 0.75).
  - On HELDOUT it lags early (0.060 / 0.182 / 0.460 / 0.731), and 10% of episodes have p ≤ −0.156 at t/T = 0.1.
  - mean_t |p_t − t/T| is 0.055 against 0.089, and ψ's effective rank is unchanged (16.9).
- **First two quarters, compared with fixed**:
  - First quarter: top-1 0.719 against 0.758, pairwise 0.887 against 0.916, sensitivity 0.139 against 0.156.
  - First quarter: P(demo > stay) is equal (0.922) and P(demo > other-task action) is higher (0.970 against 0.960).
  - Second quarter: top-1 is higher (0.867 against 0.844).
  - Second quarter: P(demo > stay) 0.922 against 0.945, P(demo > other-task action) 0.945 against 0.974, pairwise 0.928 against 0.954.
- **Other changes**:
  - Shared-start discrimination falls from 0.866 to 0.804.
  - Sensitivity is lower in every quarter (pooled 0.160 against 0.186).
  - Query subspaces of different tasks are closer: overlap 0.841 against 0.749, different-task distance 0.162 against 0.256. The probes stay at 1.000.

## SIGReg on all frames

flow full (ψ) above applies SIGReg to the uniform frames only, t ∈ {0, …, T − 2}, so ψ_T never enters it and ψ₀ enters rarely. This section retrains flow full (ψ) with SIGReg on every frame the batch encodes, including the start and end frames, as the original design C3 specifies (aggregate distribution over all frames). The path term keeps D_ψ with its gradient. Everything else is flow full (ψ).

### Change

| Interface | File | Change |
|---|---|---|
| `SigRegFrames` | `ceam/losses/stage_one_objective.py` | enum: `UNIFORM` = "uniform", `ALL` = "all" |
| `StageOneObjective(weights, path_target, sigreg_frames, state_dim, epsilon, negative_tasks)` | `ceam/losses/stage_one_objective.py` | SIGReg acts on `sigreg_states(states)`, [N, state_dim] |
| `StageOneConfig.sigreg_frames` | `ceam/training/stage_one_config.py` | see below |
| `StageOneTrainer` | `ceam/training/stage_one_trainer.py` | passes `config.sigreg_frames` to the objective |
| `check_sigreg_frames` | `scripts/check_losses.py` | printed after all earlier checks |

SIGReg's sample per batch of B episodes with M uniform frames each:

| `sigreg_frames` | States, in concatenation order | N at B = 64, M = 8 |
|---|---|---|
| `UNIFORM` | uniform ψ(o_t) [B·M] | 512 |
| `ALL` | start ψ(o₀) [B], end ψ(o_T) [B], partner_end ψ(o′_T) [B], uniform ψ(o_t) [B·M], uniform_future ψ(o_{t+k}) [B·M] | 1216 = B(3 + 2M) |

- Under `ALL`, start states are 1/19 (5.3%) of the sample and end states (end and partner_end) 2/19 (10.5%).
- `StageOneConfig.sigreg_frames` is optional when read (`OPTIONAL_FIELDS`; missing = `UNIFORM`) and always written as the value string. Configs and checkpoints written before the field existed load unchanged. Code from before the field rejects configs that contain it, which includes the config embedded in every checkpoint written from now on.
- `configs/stage_one/goal_instruction_contrast_flow_full_sigall.json` differs from `goal_instruction_contrast_flow_full.json` only in `"sigreg_frames": "all"` and `run_dir` = `$CEAM_ROOT/outputs/runs/goal_instruction_contrast_flow_full_sigall`.

### Checks

**Loss values** (`scripts/check_losses.py`, CPU, seed 0):

- Every value printed before the change is printed unchanged; the only addition is the `check_sigreg_frames` block.
- The objective's SIGReg term equals `SigRegLoss` on the expected [N, d] sample bitwise, with the same direction seed, for both settings.
- Gradients of the SIGReg term: under `UNIFORM` only the uniform ψ; under `ALL` ψ₀, ψ_T, ψ′_T, the uniform ψ and their futures.

| States | `UNIFORM` (N = 512) | `ALL` (N = 1216) |
|---|---|---|
| all ~ 𝒩(0, I) | 0.00217 | 0.00089 |
| start − 20e, end and partner_end + 20e, e a unit vector | 0.00217 | 0.00704 |

**Response to distant start states**: an `ALL`-shaped sample of 1216 states ~ 𝒩(0, I), d = 256, with the 64 start states shifted by r along one unit vector e; SIGReg mean over 20 direction draws, and |∂SIGReg/∂ψ₀ · e| mean over start states for one draw.

| r | 0 | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|
| SIGReg | 0.00088 | 0.00142 | 0.00361 | 0.00676 | 0.00692 | 0.00699 |
| gradient on ψ₀ along e | 1.3e-7 | 1.5e-6 | 1.6e-6 | 2.3e-7 | 2.3e-7 | 1.9e-7 |

**Configs**:

- Every file in `configs/stage_one/` and every stage-one `config.json` in `outputs/runs/` loads with `sigreg_frames` = `UNIFORM` (the new config: `ALL`) and round-trips exactly.
- Written back, they differ from the file only by the added `path_target` / `sigreg_frames` keys.
- `CheckpointStore.load` works for the step-5000 checkpoints of fixed, flow (z) and flow full (ψ).

**Bit identity of `UNIFORM`**:

- **Setup**: the fixed config, shortened to 300 steps and logging every 10 steps.
  - It was trained with the code before the change (`a2c4dc7`), then with `86c727d` without the key and with `"sigreg_frames": "uniform"`.
  - All runs used the same GPU (node-61, CUDA 6, L40S).
- **Logged fields**: all 31 logged steps are bit-identical in every loss, lr, grad norm, logit scale and peak memory, for both new runs.
- **Checkpoint**: all 112 parameter tensors at step 300 are bitwise equal.
- **Recorded runs**: steps 0, 50, 100, 150 and 200 lie in the shared warmup.
  - They are bit-identical to the recorded fixed run, e.g. total at step 0 is 8.021472930908203.
  - The flow-full config (no key), 300 steps with `86c727d`, is bit-identical to the recorded flow full (ψ) run at the same steps, e.g. total at step 0 is 7.9277262687683105.
- **Step 0 of the new run**: every term equals flow full (ψ) at step 0 except SIGReg (0.4968 against 0.5029), and hence total and grad norm. The initialization and first batch are the same.

### Run

| Run | Config | Commit | Host / GPU | Time | Speed | Peak memory |
|---|---|---|---|---|---|---|
| flow full (ψ), SIGReg on all frames | `goal_instruction_contrast_flow_full_sigall.json` | `86c727d` | node-61, CUDA 7, L40S | 2060 s | 2.43 steps/s | 12.86 GiB |

Outputs in `outputs/runs/goal_instruction_contrast_flow_full_sigall/`:

- `checkpoints/step_00N000.pt`, N = 1 … 5; `train.log`, `metrics.jsonl`;
- `diagnostics/step_00N000/` (`scripts/diagnose_stage_one.py` at `86c727d`);
- `collapse_check.json`: the collapse measures below, for the three step-5000 checkpoints and steps 1000–4000 of both full-space runs;
- `training_curves_comparison.png` and `diagnostics/progress_comparison_{z,psi}_step_005000.png`: fixed, flow full (ψ) and this run.

### Training

Unweighted terms, same layout as "Training" above. The SIGReg column is on 1216 states per batch here and on 512 in the other runs.

| Run | step | path | goal | endpoint | SIGReg | paraphrase | instruction contrast | logit scale s | total |
|---|---|---|---|---|---|---|---|---|---|
| SIGReg on all frames | 0 | 0.5375 | 3.667 | 0.6425 | 0.4968 | 0.1574 | 2.421 | 14.29 | 7.922 |
| SIGReg on all frames | 1000 | 0.2202 | 7.26e-4 | 4.11e-4 | 0.0747 | 2.32e-4 | 0.2259 | 18.00 | 0.5222 |
| SIGReg on all frames | 3000 | 0.2401 | 3.88e-4 | 4.97e-4 | 0.0571 | 5.53e-5 | 0.1052 | 23.23 | 0.4034 |
| SIGReg on all frames | 4500–4999 | 0.2526 | 8.75e-5 | 1.76e-4 | 0.0586 | 3.72e-5 | 0.0901 | 24.56 | 0.4015 |
| flow full (ψ) | 4500–4999 | 0.0891 | 1.35e-4 | 2.31e-4 | 0.0526 | 4.15e-5 | 0.0661 | 27.51 | 0.2082 |

- **Path term**: it falls to 0.053 at step 250, rises to 0.22 by step 1000 and stays at 0.24–0.27 from step 2000 on (0.268 at step 4999).
- **Other terms**: goal, endpoint and paraphrase end at the level of flow full (ψ); the instruction contrast ends at 0.090 against 0.066.

### Collapse check (HELDOUT, step 5000)

Conventions as in "Geometry of ψ" and "Progress along the start-to-end direction" above: all frames 0 … T of the 43 HELDOUT episodes; interior = frames 1 … T − 1; medians over episodes, or over frames for the interior rows. p_t = ⟨ψ_t − ψ₀, n_ψ⟩ / ‖ψ_T − ψ₀‖, linear between frames.

| Quantity | fixed | flow full (ψ), SIGReg on uniform frames | flow full (ψ), SIGReg on all frames |
|---|---|---|---|
| effective rank of cov(ψ), all frames / interior frames | 16.6 / 16.5 | 1.14 / 1.14 | 1.02 / 1.02 |
| share of ψ variance on the top eigenvector, all / interior | 0.115 / 0.116 | 0.981 / 0.980 | 0.997 / 0.997 |
| ‖ψ₀ − interior mean‖ | 15.5 | 714 | 1211 |
| ‖ψ_T − interior mean‖ | 30.4 | 457 | 130 |
| ‖ψ_t − interior mean‖, interior frames (their own spread) | 14.6 | 39.5 | 119 |
| ‖ψ₀ − interior mean‖ / interior spread | 1.06 | 18.1 | 10.2 |
| ‖ψ_T − interior mean‖ / interior spread | 2.08 | 11.6 | 1.09 |
| ‖ψ₀ − mean ψ₀‖ (spread of the start states) | 10.2 | 95.7 | 33.3 |
| ‖ψ_T − ψ₀‖ | 35.0 | 1156 | 1338 |
| ‖ψ₁ − ψ₀‖ / ‖ψ_T − ψ_{T−1}‖ | 1.35 / 0.49 | 3.69 / 2.29 | 0.18 / 1.32 |
| ψ: p at t/T = 0.1 / 0.25 / 0.5 / 0.75, HELDOUT | 0.114 / 0.181 / 0.343 / 0.685 | 0.628 / 0.632 / 0.638 / 0.643 | 0.986 / 0.989 / 0.992 / 0.993 |
| ψ: p at t/T = 0.1 / 0.25 / 0.5 / 0.75, TRAIN | 0.111 / 0.192 / 0.370 / 0.703 | 0.626 / 0.625 / 0.631 / 0.640 | 0.986 / 0.988 / 0.991 / 0.992 |
| ψ: p at t/T = 0.1, p10 / p90 | 0.004 / 0.233 | 0.161 / 0.902 | 0.208 / 0.994 |
| ψ: Spearman(t, p_t) median / share > 0 | 0.997 / 1.00 | 0.795 / 0.95 | 0.959 / 1.00 |
| ψ: mean_t \|p_t − t/T\| (p10 / p90) | 0.102 (0.063 / 0.138) | 0.242 (0.203 / 0.429) | 0.457 (0.131 / 0.477) |
| first frame with p_t ≥ 0.5 (p10 / p90) | 67 (53 / 110) | 3 (1 / 49) | 5 (2 / 39) |
| z: p at t/T = 0.1 / 0.25 / 0.5 / 0.75; Spearman median | 0.151 / 0.289 / 0.520 / 0.804; 1.000 | 0.730 / 0.730 / 0.734 / 0.722; 0.142 | 0.987 / 0.988 / 0.993 / 0.998; 0.915 |
| path term, mean of steps 4500–4999 (step 4999) | 5.4e-4 (5.3e-4) | 0.0891 (0.0993) | 0.2526 (0.2676) |
| SIGReg statistic (4096 HELDOUT frames) | 0.0254 | 0.0411 | 0.0625 |

Episodes are 105 frames long (median), so the first frame with p_t ≥ 0.5 is at t/T ≈ 0.04 in this run. A plateau at 0.99 from t/T = 0.04 on has along-segment residual ∫_{0.04}^{1} (0.99 − f)² df = 0.29.

Over training (HELDOUT, ψ): effective rank / top share / ‖ψ₀ − interior mean‖ / ‖ψ_T − interior mean‖ / interior spread / ‖ψ_T − ψ₀‖ / p at t/T = 0.1 / mean_t |p_t − t/T| / SIGReg statistic.

| step | flow full (ψ), SIGReg on uniform frames | flow full (ψ), SIGReg on all frames |
|---|---|---|
| 1000 | 1.35 / 0.952 / 295 / 92 / 21.4 / 379 / 0.796 / 0.304 / 0.052 | 1.16 / 0.978 / 344 / 46 / 21.3 / 384 / 0.914 / 0.408 / 0.066 |
| 2000 | 1.17 / 0.977 / 487 / 266 / 36.5 / 744 / 0.685 / 0.243 / 0.057 | 1.04 / 0.996 / 753 / 93 / 80.6 / 843 / 0.977 / 0.438 / 0.075 |
| 3000 | 1.14 / 0.981 / 633 / 386 / 41.0 / 1012 / 0.654 / 0.238 / 0.054 | 1.03 / 0.997 / 1055 / 113 / 100.9 / 1164 / 0.983 / 0.458 / 0.075 |
| 4000 | 1.15 / 0.979 / 705 / 448 / 37.0 / 1137 / 0.636 / 0.241 / 0.044 | 1.02 / 0.997 / 1191 / 126 / 114.6 / 1314 / 0.985 / 0.456 / 0.069 |
| 5000 | 1.14 / 0.981 / 714 / 457 / 39.5 / 1156 / 0.628 / 0.242 / 0.041 | 1.02 / 0.997 / 1211 / 130 / 119.2 / 1338 / 0.986 / 0.457 / 0.062 |

### Stage-one diagnostics, step 5000 (reference)

The fixed and flow full (ψ) columns are their reports at `ad9ab4e` ("Offline diagnostics, step 5000" above); `ceam/diagnostics/` and `scripts/diagnose_stage_one.py` are unchanged from `ad9ab4e` to `86c727d`, and the collapse check above recomputes the geometry of both at `86c727d` with the same values.

Action ranking by t/T quarter, HELDOUT, k = 10, 506 anchors (153 / 128 / 130 / 95); fixed / flow full (ψ) / flow full (ψ) with SIGReg on all frames in each cell.

| Quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | pooled |
|---|---|---|---|---|---|
| P(demo > stay) | 0.922 / 0.902 / 0.902 | 0.945 / 0.953 / 0.977 | 0.962 / 0.969 / 0.985 | 0.958 / 0.947 / 0.947 | 0.945 / 0.941 / 0.951 |
| P(demo > reverse) | 0.945 / 1.000 / 1.000 | 0.984 / 0.969 / 0.992 | 0.977 / 0.985 / 0.992 | 0.968 / 0.937 / 0.958 | 0.970 / 0.974 / 0.987 |
| P(demo > other-task action) | 0.960 / 0.975 / 0.975 | 0.974 / 0.953 / 0.968 | 0.963 / 0.978 / 0.984 | 0.959 / 0.946 / 0.953 | 0.964 / 0.964 / 0.971 |
| instruction top-1 (chance 0.1) | 0.758 / 0.654 / 0.673 | 0.844 / 0.859 / 0.852 | 0.962 / 0.969 / 0.969 | 0.958 / 0.937 / 0.947 | 0.870 / 0.840 / 0.846 |
| instruction pairwise | 0.916 / 0.883 / 0.880 | 0.954 / 0.950 / 0.957 | 0.962 / 0.972 / 0.979 | 0.958 / 0.944 / 0.952 | 0.945 / 0.934 / 0.939 |
| sensitivity (median std of candidate matches) | 0.156 / 0.124 / 0.118 | 0.187 / 0.170 / 0.150 | 0.191 / 0.211 / 0.171 | 0.204 / 0.199 / 0.176 | 0.186 / 0.173 / 0.157 |
| effective rank of demo effects | 9.57 / 1.01 / 1.00 | 7.87 / 1.47 / 1.13 | 10.16 / 1.18 / 1.26 | 9.03 / 1.25 / 1.22 | 15.07 / 1.05 / 1.00 |

| Quantity | fixed | flow full (ψ) | flow full (ψ), SIGReg on all frames |
|---|---|---|---|
| shared start P(demo > other-task first chunk) | 0.866 | 0.800 | 0.851 |
| task-ID probe HELDOUT, ψ(o_T) | 1.000 | 0.884 | 1.000 |
| task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 1.000 | 0.895 | 0.979 |
| task-ID probe HELDOUT, ψ(o₀) control | 0.070 | 0.186 | 0.116 |
| displacement ratio ‖z_T − z₀‖ / RMS cross-episode | 1.65 | 2.84 | 2.50 |
| path deviation max_t ‖(I − nnᵀ)(z_t − z₀)‖ / ‖z_T − z₀‖ | 0.056 | 0.076 | 0.013 |
| goal error ‖Uᵀĝ − z_T‖ / ‖z_T − z₀‖ | 0.0082 | 0.0097 | 0.0018 |
| task overlap ‖U_iᵀU_j‖²/r, off-diagonal mean | 0.749 | 0.906 | 0.904 |
| subspace distance, different-task pairs | 0.256 | 0.096 | 0.099 |
| per-dimension std of ψ, min / median / max | 0.74 / 1.11 / 1.69 | 1.56 / 8.77 / 34.51 | 0.80 / 14.68 / 59.53 |

### Reading

- **The collapse is not gone.** With SIGReg on all frames the full-space variant still collapses, and further than with SIGReg on the uniform frames:
  - effective rank 1.02 against 1.14 (fixed 16.6); the top eigenvector carries 99.7% of ψ's variance, with or without the first and last frames;
  - ‖ψ_T − ψ₀‖ is 1338 against 1156 (fixed 35);
  - mean_t |p_t − t/T| is 0.457 against 0.242 (fixed 0.102); the path term ends at 0.25 against 0.09.
- **What moved**: the end states and the plateau.
  - ψ_T now lies within the interior frames' own spread (130 against 119, ratio 1.09; fixed 2.08); with SIGReg on the uniform frames it was 11.6 spreads away.
  - The start states form one distant cluster: 1211 from the interior mean, 10.2 interior spreads, with a spread of 33 among themselves.
  - Interior frames sit next to the end state: p ≈ 0.99 at t/T = 0.1 … 0.75 on HELDOUT and TRAIN. The median episode leaves the start cluster within its first 5 frames (t/T ≈ 0.04); 10% of episodes are below p = 0.21 at t/T = 0.1.
- **Timing**: the collapse is present at step 1000 (rank 1.16, ψ₀ 344 from the interior mean against a spread of 21) and grows through step 5000.
- **SIGReg**: its statistic on HELDOUT frames is 0.062, higher than both other runs. For start states 5.3% of the sample, SIGReg levels off at about 0.007 once they are 100 or more from the rest, and its gradient on them stays below 2e-6 at every tested distance.
- **Reference diagnostics** (sanity only): action ranking is close to flow full (ψ): first-quarter top-1 0.673 against 0.654 (fixed 0.758); demo effects have effective rank 1.0–1.3 in every quarter. The task-ID probe from ψ(o_T) returns to 1.000, and shared-start discrimination is 0.851 against 0.800 (fixed 0.866).

## Stop-gradient on D_ψ

The full-space path term divides by D_ψ = ‖ψ_T − ψ₀‖² + ε only to make it invariant to a global scaling of ψ. With its gradient, D_ψ is also an optimizable quantity: the gradient descent step on ψ_T points along +(ψ_T − ψ₀) and the one on ψ₀ along −(ψ_T − ψ₀), so every step with a nonzero residual pushes the endpoints apart. This section retrains "flow full (ψ), SIGReg on all frames" with D_ψ under stop-gradient: ‖ψ_t − ((1 − t/T)·sg[ψ₀] + (t/T)·sg[ψ_T])‖² / sg[D_ψ]. ψ₀ and ψ_T then receive no gradient from the path term. Everything else is unchanged, including SIGReg on all frames and the projected D = ‖z_T − z₀‖² + ε of the goal and endpoint terms, which keeps its gradient.

### Change

| Interface | File | Change |
|---|---|---|
| `PathLoss(target, epsilon, normalizer_gradient)` | `ceam/losses/path_loss.py` | `normalizer_gradient = False` detaches D_ψ of `INTERPOLATION_FULL`; the value is unchanged. `False` with `PERPENDICULAR` or `INTERPOLATION` raises `ValueError`, because their D is the projected D shared with the goal and endpoint terms. |
| `StageOneObjective(weights, path_target, path_normalizer_gradient, sigreg_frames, state_dim, epsilon, negative_tasks)` | `ceam/losses/stage_one_objective.py` | passes `path_normalizer_gradient` to `PathLoss` |
| `StageOneConfig.path_normalizer_gradient` | `ceam/training/stage_one_config.py` | JSON boolean, see below |
| `StageOneTrainer` | `ceam/training/stage_one_trainer.py` | passes `config.path_normalizer_gradient` to the objective |
| `check_path_normalizer_gradient` | `scripts/check_losses.py` | printed after all earlier checks |

- `StageOneConfig.path_normalizer_gradient` is optional when read (`OPTIONAL_FIELDS`; missing = true, D_ψ with gradients) and always written as a JSON boolean; any other JSON value raises `ValueError`.
- Configs and checkpoints written before the field existed load unchanged. Code from before the field rejects configs that contain it, which includes the config embedded in every checkpoint written from now on.
- `configs/stage_one/goal_instruction_contrast_flow_full_sigall_sgd.json` differs from `goal_instruction_contrast_flow_full_sigall.json` only in `"path_normalizer_gradient": false` and `run_dir` = `$CEAM_ROOT/outputs/runs/goal_instruction_contrast_flow_full_sigall_sgd`.

### Checks

**Loss values and gradients** (`scripts/check_losses.py`, CPU, seed 0, random states, B = 64, M = 8, d = 256):

- Every value printed before the change is printed unchanged; the only addition is the `check_path_normalizer_gradient` block.
- `PathLoss` `INTERPOLATION_FULL`:

| `normalizer_gradient` | value | ‖∇ψ₀‖ | ‖∇ψ_T‖ | ‖∇ψ_t‖, uniform frames |
|---|---|---|---|---|
| true | 0.8291634321 | 9.240e-3 | 9.240e-3 | 3.562e-3 |
| false | 0.8291634321 (bitwise equal) | no autograd path | no autograd path | 3.562e-3 (bitwise equal) |

- **Direction of the D_ψ gradient** (true): per episode, cos(−∇ψ_T, ψ_T − ψ₀) = 1.000000 and cos(−∇ψ₀, ψ₀ − ψ_T) = 1.000000.
- **Scale invariance**: scaling every ψ by 0.1 or 10 changes the value by −7.2e-8 and 0 relative, identically under both settings.
- **Objective** (`StageOneObjective`, `INTERPOLATION_FULL`, `SigRegFrames.ALL`, SIGReg directions reseeded identically), true against false:
  - every term of the report is bitwise equal;
  - the path term loses its gradient on ψ₀ and ψ_T; its gradient on the uniform ψ is bitwise equal;
  - the gradients of goal + endpoint (ψ₀, ψ_T, ψ′_T, W_a of U_a) and of SIGReg are bitwise equal;
  - the total's gradient differs only on ψ₀ (0.2311 → 0.2298) and ψ_T (0.2230 → 0.2219).

**Configs**:

- Every file in `configs/stage_one/` and every stage-one `config.json` in `outputs/runs/` loads with `path_normalizer_gradient` = true (the new config: false) and round-trips exactly.
- Written back, they differ from the file only by the added `path_target` / `path_normalizer_gradient` / `sigreg_frames` keys.
- The values `"false"`, `0` and `null` raise `ValueError`.
- `CheckpointStore.load` works for the step-5000 checkpoints of all six earlier `goal_instruction_contrast*` runs, which read `path_normalizer_gradient` = true.

**Bit identity with the field missing**:

- **Setup**: the fixed config and `goal_instruction_contrast_flow_full_sigall.json`, each shortened to 300 steps and logging every 10 steps, without the key.
  - Each was trained with the code before the change (`54fb5f2`) and with `697b424`; the SIGReg-on-all-frames config also with `"path_normalizer_gradient": true`.
  - All five runs used the same GPU (node-62, CUDA 6, L40S).
- **Logged fields**: all 31 logged steps are bit-identical in every loss, lr, grad norm, logit scale and peak memory, for all three comparisons.
- **Checkpoint**: all 112 parameter tensors at step 300 are bitwise equal.
- **Recorded runs**: steps 0, 50, 100, 150 and 200 lie in the shared warmup and are bit-identical to the recorded runs, e.g. total at step 0 is 8.021472930908203 (fixed) and 7.921598434448242 (SIGReg on all frames).
- **Step 0 of the new run**: every term equals flow full (ψ) with SIGReg on all frames at step 0 (total 7.921598434448242); the grad norm is 392.86 against 393.81, the difference being the D_ψ gradient.

### Run

| Run | Config | Commit | Host / GPU | Time | Speed | Peak memory |
|---|---|---|---|---|---|---|
| flow full (ψ), SIGReg on all frames, sg[D_ψ] | `goal_instruction_contrast_flow_full_sigall_sgd.json` | `697b424` | node-62, CUDA 7, L40S | 3172 s | 1.58 steps/s | 12.86 GiB |

The sampler time per step rose from 0.10 s to about 0.5 s for part of the run, hence the lower speed.

Outputs in `outputs/runs/goal_instruction_contrast_flow_full_sigall_sgd/`:

- `checkpoints/step_00N000.pt`, N = 1 … 5; `train.log`, `metrics.jsonl`;
- `diagnostics/step_00N000/` (`scripts/diagnose_stage_one.py` at `697b424`);
- `collapse_check.json`: the collapse measures below, for the four step-5000 checkpoints and steps 1000–4000 of this run;
- `training_curves_comparison.png` and `diagnostics/progress_comparison_{z,psi}_step_005000.png`: fixed, flow full (ψ), SIGReg on all frames, and this run.

### Training

Unweighted terms, same layout as "Training" above. SIGReg is on 1216 states per batch in both rows.

| Run | step | path | goal | endpoint | SIGReg | paraphrase | instruction contrast | logit scale s | total |
|---|---|---|---|---|---|---|---|---|---|
| sg[D_ψ] | 0 | 0.5375 | 3.667 | 0.6425 | 0.4968 | 0.1574 | 2.421 | 14.29 | 7.922 |
| sg[D_ψ] | 1000 | 0.0226 | 1.65e-3 | 5.65e-4 | 0.0606 | 2.44e-4 | 0.0965 | 13.92 | 0.1821 |
| sg[D_ψ] | 3000 | 0.0104 | 2.54e-4 | 1.90e-4 | 0.0576 | 8.76e-5 | 0.0304 | 13.76 | 0.0990 |
| sg[D_ψ] | 4500–4999 | 7.15e-3 | 5.20e-5 | 7.83e-5 | 0.0557 | 5.39e-5 | 0.0101 | 13.84 | 0.0732 |
| SIGReg on all frames, D_ψ with gradient | 4500–4999 | 0.2526 | 8.75e-5 | 1.76e-4 | 0.0586 | 3.72e-5 | 0.0901 | 24.56 | 0.4015 |

- **Path term**: 0.030 at step 250, then falling through training (0.028 at 500, 0.017 at 2000, 0.0073 at 4000); logged values from step 250 on range over 0.0062–0.033. There is no plateau.
- **Other terms**: the instruction contrast ends at 0.010 (fixed 0.011, D_ψ with gradient 0.090); the logit scale stays at 13.7–14.3, where the other three runs grew it to 18.3–27.5.

### Collapse check (HELDOUT, step 5000)

Conventions as in "Collapse check" of "SIGReg on all frames". The HELDOUT interpolation residual is the path term's value on every interior frame of an episode, ‖ψ_t − ((1 − t/T)ψ₀ + (t/T)ψ_T)‖² / ‖ψ_T − ψ₀‖², averaged per episode.

| Quantity | fixed | flow full (ψ), SIGReg on uniform frames | SIGReg on all frames | SIGReg on all frames, sg[D_ψ] |
|---|---|---|---|---|
| effective rank of cov(ψ), all frames / interior frames | 16.6 / 16.5 | 1.14 / 1.14 | 1.02 / 1.02 | 8.56 / 8.56 |
| share of ψ variance on the top eigenvector, all / interior | 0.115 / 0.116 | 0.981 / 0.980 | 0.997 / 0.997 | 0.198 / 0.199 |
| ‖ψ₀ − interior mean‖ | 15.5 | 714 | 1211 | 2.72 |
| ‖ψ_T − interior mean‖ | 30.4 | 457 | 130 | 27.7 |
| ‖ψ_t − interior mean‖, interior frames (their own spread) | 14.6 | 39.5 | 119 | 11.9 |
| ‖ψ₀ − interior mean‖ / interior spread | 1.06 | 18.1 | 10.2 | 0.23 |
| ‖ψ_T − interior mean‖ / interior spread | 2.08 | 11.6 | 1.09 | 2.33 |
| ‖ψ₀ − mean ψ₀‖ / ‖ψ_T − mean ψ_T‖ (spread of the start / end states) | 10.2 / 29.9 | 95.7 / 184.5 | 33.3 / 93.7 | 0.90 / 27.1 |
| ‖ψ_T − ψ₀‖ | 35.0 | 1156 | 1338 | 27.5 |
| ‖ψ₁ − ψ₀‖ / ‖ψ₁₀ − ψ₀‖ / ‖ψ_T − ψ_{T−1}‖ | 1.35 / 15.1 / 0.49 | 3.69 / 748 / 2.29 | 0.18 / 1322 / 1.32 | 0.078 / 1.03 / 0.46 |
| ψ: p at t/T = 0.1 / 0.25 / 0.5 / 0.75, HELDOUT | 0.114 / 0.181 / 0.343 / 0.685 | 0.628 / 0.632 / 0.638 / 0.643 | 0.986 / 0.989 / 0.992 / 0.993 | 0.029 / 0.230 / 0.430 / 0.668 |
| ψ: p at t/T = 0.1 / 0.25 / 0.5 / 0.75, TRAIN | 0.111 / 0.192 / 0.370 / 0.703 | 0.626 / 0.625 / 0.631 / 0.640 | 0.986 / 0.988 / 0.991 / 0.992 | 0.035 / 0.238 / 0.446 / 0.659 |
| ψ: p at t/T = 0.1, p10 / p90 | 0.004 / 0.233 | 0.161 / 0.902 | 0.208 / 0.994 | 0.001 / 0.086 |
| ψ: p at t/T = 0.5, p10 / p90 | 0.176 / 0.465 | 0.493 / 0.898 | 0.580 / 0.995 | −0.104 / 0.539 |
| ψ: Spearman(t, p_t) median / share > 0 | 0.997 / 1.00 | 0.795 / 0.95 | 0.959 / 1.00 | 1.000 / 1.00 |
| ψ: mean_t \|p_t − t/T\| (p10 / p90) | 0.102 (0.063 / 0.138) | 0.242 (0.203 / 0.429) | 0.457 (0.131 / 0.477) | 0.061 (0.030 / 0.220) |
| first frame with p_t ≥ 0.5 (p10 / p90) | 67 (53 / 110) | 3 (1 / 49) | 5 (2 / 39) | 62 (47 / 94) |
| z: p at t/T = 0.1 / 0.25 / 0.5 / 0.75; Spearman median | 0.151 / 0.289 / 0.520 / 0.804; 1.000 | 0.730 / 0.730 / 0.734 / 0.722; 0.142 | 0.987 / 0.988 / 0.993 / 0.998; 0.915 | 0.072 / 0.272 / 0.463 / 0.765; 0.998 |
| HELDOUT interpolation residual, median / mean over episodes | 0.189 / 0.186 | 0.092 / 0.133 | 0.298 / 0.716 | 0.0065 / 0.038 |
| path term, mean of steps 4500–4999 (step 4999) | 5.4e-4 (5.3e-4) | 0.0891 (0.0993) | 0.2526 (0.2676) | 0.0071 (0.0083) |
| SIGReg statistic (4096 HELDOUT frames) | 0.0254 | 0.0411 | 0.0625 | 0.0655 |

The fixed stage one constrains only the perpendicular component in z, so its residual in ψ is large.

Measures of other degenerate solutions, same frames:

| Quantity | fixed | flow full (ψ), SIGReg on uniform frames | SIGReg on all frames | SIGReg on all frames, sg[D_ψ] |
|---|---|---|---|---|
| median ‖ψ_t‖, all frames | 14.8 | 12.9 | 11.8 | 11.9 |
| trace of cov(ψ), all frames | 320 | 4.3e4 | 1.3e5 | 219 |
| per-dimension std of ψ, min / median / max | 0.74 / 1.11 / 1.69 | 1.56 / 8.77 / 34.51 | 0.80 / 14.68 / 59.53 | 0.23 / 0.88 / 1.59 |
| within-episode spread of the interior frames; its ratio to ‖ψ_T − ψ₀‖ | 10.9; 0.31 | 13.3; 0.011 | 37.7; 0.028 | 5.96; 0.22 |
| ‖ψ_{t+1} − ψ_t‖, interior frames | 0.45 | 0.57 | 0.37 | 0.26 |
| path length Σ‖ψ_{t+1} − ψ_t‖ / ‖ψ_T − ψ₀‖ | 2.71 | 1.09 | 1.03 | 1.43 |
| max_t ‖(I − n_ψn_ψᵀ)(ψ_t − ψ₀)‖ / ‖ψ_T − ψ₀‖ | 0.587 | 0.047 | 0.014 | 0.062 |
| effective rank of cov(ψ₀) / of cov(ψ_T) over episodes | 5.75 / 8.97 | 1.01 / 1.51 | 1.00 / 1.03 | 6.10 / 8.05 |

The within-episode spread is the median over episodes of the median distance of an episode's interior frames from their own mean.

Over training (HELDOUT, ψ): effective rank / top share / ‖ψ₀ − interior mean‖ / ‖ψ_T − interior mean‖ / interior spread / ‖ψ_T − ψ₀‖ / spread of the start states / p at t/T = 0.1 / p at t/T = 0.5, p10 / mean_t |p_t − t/T| / SIGReg statistic.

| step | SIGReg on all frames, sg[D_ψ] |
|---|---|
| 1000 | 9.83 / 0.150 / 5.2 / 23.7 / 11.5 / 24.8 / 0.32 / 0.023 / 0.099 / 0.059 / 0.082 |
| 2000 | 9.27 / 0.175 / 4.1 / 26.0 / 11.9 / 26.5 / 0.47 / 0.029 / 0.022 / 0.055 / 0.067 |
| 3000 | 8.77 / 0.204 / 2.5 / 26.4 / 11.9 / 26.3 / 0.69 / 0.027 / −0.092 / 0.056 / 0.067 |
| 4000 | 8.58 / 0.202 / 2.5 / 27.9 / 12.0 / 28.0 / 0.81 / 0.028 / −0.115 / 0.063 / 0.066 |
| 5000 | 8.56 / 0.198 / 2.7 / 27.7 / 11.9 / 27.5 / 0.90 / 0.029 / −0.104 / 0.061 / 0.066 |

HELDOUT episodes with p < 0 at t/T = 0.5: 9 of 43 at step 5000, all in tasks 3, 4 and 5 (15 HELDOUT episodes), against 0 in fixed. In the plotted episodes of tasks 4 and 5 (`progress_comparison_psi_step_005000.png`), p drifts to about −0.1 to −0.2 and then jumps at t/T ≈ 0.5–0.6, at the frame where the fixed stage one also jumps. On TRAIN, p10 at t/T = 0.5 is 0.398.

### Stage-one diagnostics, step 5000 (reference)

The fixed, flow full (ψ) and SIGReg-on-all-frames columns are their reports from the sections above; `ceam/diagnostics/` and `scripts/diagnose_stage_one.py` are unchanged from `86c727d` to `697b424`, and the collapse check recomputes the geometry and progress of all three at `697b424` with the same values.

Action ranking by t/T quarter, HELDOUT, k = 10, 506 anchors (153 / 128 / 130 / 95); fixed / flow full (ψ) / SIGReg on all frames / sg[D_ψ] in each cell.

| Quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | pooled |
|---|---|---|---|---|---|
| P(demo > stay) | 0.922 / 0.902 / 0.902 / 0.961 | 0.945 / 0.953 / 0.977 / 0.898 | 0.962 / 0.969 / 0.985 / 0.977 | 0.958 / 0.947 / 0.947 / 0.958 | 0.945 / 0.941 / 0.951 / 0.949 |
| P(demo > reverse) | 0.945 / 1.000 / 1.000 / 0.982 | 0.984 / 0.969 / 0.992 / 0.898 | 0.977 / 0.985 / 0.992 / 0.992 | 0.968 / 0.937 / 0.958 / 0.979 | 0.970 / 0.974 / 0.987 / 0.961 |
| P(demo > other-task action) | 0.960 / 0.975 / 0.975 / 0.972 | 0.974 / 0.953 / 0.968 / 0.944 | 0.963 / 0.978 / 0.984 / 0.982 | 0.959 / 0.946 / 0.953 / 0.958 | 0.964 / 0.964 / 0.971 / 0.964 |
| instruction top-1 (chance 0.1) | 0.758 / 0.654 / 0.673 / 0.830 | 0.844 / 0.859 / 0.852 / 0.820 | 0.962 / 0.969 / 0.969 / 0.962 | 0.958 / 0.937 / 0.947 / 0.958 | 0.870 / 0.840 / 0.846 / 0.885 |
| instruction pairwise | 0.916 / 0.883 / 0.880 / 0.959 | 0.954 / 0.950 / 0.957 / 0.931 | 0.962 / 0.972 / 0.979 / 0.973 | 0.958 / 0.944 / 0.952 / 0.958 | 0.945 / 0.934 / 0.939 / 0.955 |
| sensitivity (median std of candidate matches) | 0.156 / 0.124 / 0.118 / 0.266 | 0.187 / 0.170 / 0.150 / 0.306 | 0.191 / 0.211 / 0.171 / 0.308 | 0.204 / 0.199 / 0.176 / 0.315 | 0.186 / 0.173 / 0.157 / 0.302 |
| effective rank of demo effects | 9.57 / 1.01 / 1.00 / 9.68 | 7.87 / 1.47 / 1.13 / 3.82 | 10.16 / 1.18 / 1.26 / 3.12 | 9.03 / 1.25 / 1.22 / 7.98 | 15.07 / 1.05 / 1.00 / 4.89 |

| Quantity | fixed | flow full (ψ) | SIGReg on all frames | sg[D_ψ] |
|---|---|---|---|---|
| shared start P(demo > other-task first chunk) | 0.866 | 0.800 | 0.851 | 0.968 |
| task-ID probe HELDOUT, ψ(o_T) | 1.000 | 0.884 | 1.000 | 1.000 |
| task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 1.000 | 0.895 | 0.979 | 1.000 |
| task-ID probe HELDOUT, ψ(o₀) control | 0.070 | 0.186 | 0.116 | 0.116 |
| displacement ratio ‖z_T − z₀‖ / RMS cross-episode | 1.65 | 2.84 | 2.50 | 1.48 |
| path deviation max_t ‖(I − nnᵀ)(z_t − z₀)‖ / ‖z_T − z₀‖ | 0.056 | 0.076 | 0.013 | 0.033 |
| goal error ‖Uᵀĝ − z_T‖ / ‖z_T − z₀‖ | 0.0082 | 0.0097 | 0.0018 | 0.0085 |
| task overlap ‖U_iᵀU_j‖²/r, off-diagonal mean | 0.749 | 0.906 | 0.904 | 0.873 |
| subspace distance, different-task pairs | 0.256 | 0.096 | 0.099 | 0.127 |
| other-task / own normalized endpoint distance to ĝ_a | 2.1e4 | 1.4e3 | 6.4e3 | 3.8e4 |

Over training (HELDOUT, pooled): P(demo > stay) / P(demo > other task) / top-1 / first-quarter top-1 / second-quarter P(demo > stay) / shared start / effective rank of demo effects / effective rank of ψ.

| step | SIGReg on all frames, sg[D_ψ] |
|---|---|
| 1000 | 0.974 / 0.962 / 0.842 / 0.732 / 0.977 / 0.920 / 7.9 / 9.8 |
| 2000 | 0.980 / 0.968 / 0.877 / 0.797 / 0.984 / 0.924 / 7.5 / 9.3 |
| 3000 | 0.974 / 0.964 / 0.872 / 0.791 / 0.977 / 0.947 / 5.7 / 8.8 |
| 4000 | 0.949 / 0.967 / 0.891 / 0.830 / 0.875 / 0.968 / 5.1 / 8.6 |
| 5000 | 0.949 / 0.964 / 0.885 / 0.830 / 0.898 / 0.968 / 4.9 / 8.6 |

### Reading

- **The rank-1 collapse is gone.** With D_ψ under stop-gradient and SIGReg on all frames:
  - the effective rank of ψ is 8.56 (fixed 16.6, D_ψ with gradient 1.02), and the top eigenvector carries 19.8% of the variance (fixed 11.5%, 99.7%);
  - ‖ψ_T − ψ₀‖ is 27.5 (fixed 35.0, 1338); the end states lie 2.33 interior spreads from the interior mean (fixed 2.08), and no start or end cluster lies far outside the interior frames;
  - the collapse is absent at every checkpoint from step 1000 on.
- **The interpolation target is realized** on HELDOUT and TRAIN:
  - p in ψ at t/T = 0.25 / 0.5 / 0.75 is 0.230 / 0.430 / 0.668 on HELDOUT and 0.238 / 0.446 / 0.659 on TRAIN;
  - Spearman(t, p_t) has median 1.000 with every episode positive; mean_t |p_t − t/T| is 0.061, against 0.102 for fixed;
  - the path term falls through training to 0.0071, and the HELDOUT residual has median 0.0065.
- **No overall shrinkage and no interior compression**:
  - the median ‖ψ_t‖ is 11.9 against 14.8 for fixed, and the total variance 219 against 320;
  - the interior frames spread 11.9 around their mean (fixed 14.6), and 0.22 of ‖ψ_T − ψ₀‖ within an episode (fixed 0.31);
  - there are no plateaus: the median episode reaches p = 0.5 at frame 62 (fixed 67).
- **Remaining deviations from fixed**:
  - *Start states*: they nearly coincide across episodes. Their spread is 0.90, against 10.2 for fixed, and the common start lies 0.23 interior spreads from the interior mean. The first frames barely move: ‖ψ₁₀ − ψ₀‖ is 1.03 against 15.1, and p at t/T = 0.1 is 0.029 against the target 0.1. The start-state spread grows over training, from 0.32 at step 1000 to 0.90 at step 5000.
  - *Rank*: the effective rank is half of fixed's and declined from 9.83 at step 1000 to 8.56, with no change from step 4000 to 5000. Demo effects have effective rank 3.1–3.8 in the middle quarters (fixed 7.9–10.2). The SIGReg statistic is 0.066 (fixed 0.025).
  - *Held-out tasks 3–5*: 9 of 43 HELDOUT episodes, all in tasks 3–5, move backwards along the start-to-end direction before t/T ≈ 0.5 (p10 of p at t/T = 0.5 is −0.10; TRAIN 0.40). The p10 fell from 0.099 at step 1000 to −0.092 at step 3000. Second-quarter P(demo > stay) and P(demo > reverse) fell from 0.977–0.992 at steps 1000–3000 to 0.875–0.930 at steps 4000–5000 (fixed 0.945 / 0.984).
- **Reference diagnostics** (sanity only):
  - first-quarter top-1 is 0.830 (fixed 0.758, D_ψ with gradient 0.673), and pooled top-1 is 0.885 (fixed 0.870);
  - shared-start discrimination is 0.968 (fixed 0.866); the task-ID probes are 1.000;
  - action sensitivity is higher in every quarter (pooled 0.302 against 0.186);
  - query subspaces of different tasks are closer than in fixed: overlap 0.873 against 0.749.

## Δ and π on the flow ψ versions (phase 27)

The dynamics head Δ and the proposal head π are trained on a frozen stage one, so a stage one other than the fixed one needs its own. This section trains both on two flow stage ones with the recipes used on the fixed stage one, checks them offline and in the simulator, and runs π's mean chunk in the closed loop as each stage one's baseline:

- **flow (z)**: `goal_instruction_contrast_flow`, step 5000 (section "Runs").
- **sg[D_ψ]**: flow full (ψ) with SIGReg on all frames and D_ψ under stop-gradient, `goal_instruction_contrast_flow_full_sigall_sgd`, step 5000 (section "Stop-gradient on D_ψ"). Its HELDOUT ψ has effective rank 8.56 and ‖ψ_T − ψ₀‖ 27.5 (`tmp/runlogs/s3p27/psi_rank.py`, the same values as the collapse check there).

Recipes:

- **Δ**: the +A recipe (`docs/dynamics_data.md`): demonstrations plus data A at fraction 0.5, 3000 steps, seed 0, batch 256.
- **π**: the recipe of `docs/proposal_branch.md`: behaviour cloning on TRAIN demonstrations, 3000 steps, seed 0, batch 256.
- **Configs**: each differs from the fixed stage one's config (`goal_dynamics_perturb.json`, `goal_proposal.json`) only in `stage_one_checkpoint` and `run_dir`.

| Stage one | Δ config → run directory | π config → run directory | ψ table (`build_state_table.py`) |
|---|---|---|---|
| flow (z) | `configs/stage_two/goal_dynamics_perturb_flow.json` → `outputs/runs/goal_dynamics_perturb_flow` | `configs/proposal/goal_proposal_flow.json` → `outputs/runs/goal_proposal_flow` | `outputs/state_tables/goal_instruction_contrast_flow_step5000.npy` |
| sg[D_ψ] | `configs/stage_two/goal_dynamics_perturb_flow_full_sgd.json` → `outputs/runs/goal_dynamics_perturb_flow_full_sgd` | `configs/proposal/goal_proposal_flow_full_sgd.json` → `outputs/runs/goal_proposal_flow_full_sgd` | `outputs/state_tables/goal_instruction_contrast_flow_full_sigall_sgd_step5000.npy` |

The ψ tables are [52895, 256] float32; 8 rows recomputed one at a time differ from them by at most 2.1e-6 and 1.7e-6. The closed loop needs the table of its stage one for the start-observation check and the ψ-retrieval index, and the evaluations below read ψ from it.

### Runs

Launchers and logs: `tmp/runlogs/s3p27/` (`launch_train_*.sh`, `launch_eval_*.sh`, `launch_v7.sh`, `logs/`). All GPUs are L40S.

| Job | flow (z): code `2a9b45a` | sg[D_ψ]: code `76cb2b4` |
|---|---|---|
| Δ, 3000 updates | node-62 CUDA 0; 271.3 s, 11.06 steps/s, 3.39 GiB | node-61 CUDA 2; 256.6 s, 11.69 steps/s, 3.39 GiB |
| π, 3000 updates | node-62 CUDA 2; 283.8 s, 10.57 steps/s, 3.30 GiB | node-61 CUDA 3; 309.0 s, 9.71 steps/s, 3.30 GiB |
| ψ table | node-62 CUDA 4; 10.4 s | node-61 CUDA 4; 43.7 s |
| counterfactual check | node-62 CUDA 0; 552 s | node-62 CUDA 5; 562 s |
| closed loop, 10 task processes | node-65 CUDA 0, 1, 2, 3, 5; 98–240 s each | node-61 CUDA 1, 3, 4, 6; 95–227 s each |

- **`/tmp/robosuite.log`**: on node-61, node-62 and node-65 it holds no entry from these runs; it changed during them only through processes of other users' robosuite environments.
- **Code**: `76cb2b4` adds the `path_normalizer_gradient` field of `StageOneConfig` (phase 24), which the sg[D_ψ] checkpoint's config contains; the Δ, π, closed-loop and evaluation code is the same in both commits.
- **Step 0**: π's MSE terms and gripper agreement equal the fixed run's in both runs (MSE 1.066, gripper agreement 0.588); the gradient norm differs (1.312 fixed, 1.309 flow (z), 1.315 sg[D_ψ]). Δ's demonstration MSE at step 0 is the mean z² of each ψ: 0.159 fixed, 0.137 flow (z), 0.034 sg[D_ψ]; the data-A part is 0.299, 0.298 and 0.136.

Training batches at the last logged step (2999), relative error / cos(Δ̂, z) for Δ and MSE for π:

| Head | fixed ψ | flow (z) | sg[D_ψ] |
|---|---|---|---|
| Δ, demonstration part | 0.214 / 0.966 | 0.192 / 0.971 | 0.333 / 0.938 |
| Δ, data-A part | 0.320 / 0.896 | 0.313 / 0.891 | 0.436 / 0.808 |
| π, TRAIN batch MSE | 0.015 | 0.015 | 0.016 |

### Δ on HELDOUT demonstration transitions

All 4864 valid HELDOUT (episode, t) of the feature cache, z = ψ(o_{t+10}) − ψ(o_t) from each stage one's ψ table, Δ̂ in float32 (`tmp/runlogs/s3p27/heldout_demo_error.py`, the phase-7 script with the ψ table and checkpoints as arguments; on the fixed-ψ heads it reproduces the phase-7 values):

| Head | median ‖z‖ | median ‖Δ̂‖ | median relative error | mean / median cos |
|---|---|---|---|---|
| fixed ψ, demonstrations only (`goal_dynamics_short`) | 4.00 | 3.94 | 0.215 | 0.922 / 0.988 |
| fixed ψ, +A (`goal_dynamics_perturb`) | 4.00 | 3.88 | 0.248 | 0.924 / 0.983 |
| flow (z), +A (`goal_dynamics_perturb_flow`) | 3.90 | 3.76 | 0.238 | 0.932 / 0.983 |
| sg[D_ψ], +A (`goal_dynamics_perturb_flow_full_sgd`) | 2.13 | 2.05 | 0.343 | 0.863 / 0.969 |

### HELDOUT counterfactual check

`scripts/check_dynamics_counterfactuals.py`, the phase-12 protocol (43 HELDOUT episodes × t/T 0.1, 0.4, 0.7; demo, stay and 4 perturbed chunks per σ ∈ {0.5, 1, 2}), seed 0, no π set. Outputs `outputs/dynamics_counterfactuals/goal_dynamics_perturb_flow{,_full_sgd}_step3000_heldout_seed0/`. True effects and matches are in each stage one's own ψ and q_l. Comparison: `tmp/runlogs/s3p27/cf_compare.py`, output `cf_compare_all.txt`.

- **Restoration**: in both runs the proprio and token fidelity of the restored frames equals the phase-7 run's (proprio max 2.2e-5; token cosine agent 0.969, wrist 0.945). Repeated restores returned exactly the same proprio and images in 98% of the samples for flow (z), as in phase 7, and in 91% for sg[D_ψ], whose check ran on a GPU that processes of another user also used.

Median relative error / mean cos by candidate group:

| Head | demo | stay | σ 0.5 | σ 1 | σ 2 |
|---|---|---|---|---|---|
| fixed ψ, demonstrations only | 0.199 / 0.953 | 1.397 / 0.501 | 0.265 / 0.912 | 0.435 / 0.819 | 0.655 / 0.669 |
| fixed ψ, +A | 0.230 / 0.951 | 0.729 / 0.683 | 0.252 / 0.927 | 0.304 / 0.897 | 0.384 / 0.854 |
| flow (z), +A | 0.211 / 0.953 | 0.695 / 0.661 | 0.242 / 0.932 | 0.316 / 0.883 | 0.379 / 0.862 |
| sg[D_ψ], +A | 0.379 / 0.889 | 0.972 / 0.519 | 0.426 / 0.834 | 0.493 / 0.770 | 0.598 / 0.720 |

Median true ‖z‖ of the demo / stay / σ 2 chunks: 4.78 / 1.40 / 7.92 for flow (z) and 2.11 / 0.46 / 3.01 for sg[D_ψ].

Ranking of the 14 candidates of each of the 129 samples (stay m = 0):

| Head | mean ρ | same top-1 | true m of the Δ̂ choice | true m: best / mean over candidates | true best in Δ̂ top 3 | Δ̂ m > 0 but true m ≤ 0: stay / σ 0.5 / σ 1 / σ 2 |
|---|---|---|---|---|---|---|
| fixed ψ, demonstrations only | 0.499 | 0.264 | 0.321 | 0.464 / 0.238 | 0.53 | 0.16 / 0.04 / 0.12 / 0.23 |
| fixed ψ, +A | 0.722 | 0.434 | 0.426 | 0.464 / 0.238 | 0.65 | 0.09 / 0.04 / 0.05 / 0.07 |
| flow (z), +A | 0.732 | 0.372 | 0.335 | 0.392 / 0.202 | 0.67 | 0.09 / 0.02 / 0.06 / 0.06 |
| sg[D_ψ], +A | 0.625 | 0.225 | 0.633 | 0.737 / 0.418 | 0.57 | 0.05 / 0.04 / 0.06 / 0.10 |

The match values of the stage ones are on different scales (the best true m of a sample averages 0.464, 0.392 and 0.737), so the "true m" columns compare within a row.

### π on HELDOUT

`scripts/evaluate_proposals.py curve` on every checkpoint with the stage one's ψ table; outputs `outputs/runs/goal_proposal_flow{,_full_sgd}/evaluation/curve/`. MSE of π's standardized mean chunk against the demonstration chunk on all valid (episode, t); fixed / flow (z) / sg[D_ψ]:

| step | TRAIN MSE | HELDOUT MSE | HELDOUT arm MSE | HELDOUT gripper MSE | HELDOUT gripper agreement |
|---|---|---|---|---|---|
| 500 | 0.1420 / 0.1468 / 0.1465 | 0.2646 / 0.2527 / 0.2497 | 0.2857 / 0.2708 / 0.2694 | 0.1380 / 0.1436 / 0.1312 | 0.958 / 0.957 / 0.962 |
| 1000 | 0.0744 / 0.0744 / 0.0740 | 0.2730 / 0.2592 / 0.2535 | 0.2964 / 0.2789 / 0.2732 | 0.1329 / 0.1406 / 0.1352 | 0.962 / 0.959 / 0.961 |
| 2000 | 0.0231 / 0.0232 / 0.0231 | 0.2717 / 0.2617 / 0.2597 | 0.2925 / 0.2805 / 0.2787 | 0.1469 / 0.1483 / 0.1455 | 0.961 / 0.961 / 0.962 |
| 3000 | 0.0149 / 0.0147 / 0.0147 | 0.2736 / 0.2631 / 0.2612 | 0.2939 / 0.2813 / 0.2796 | 0.1516 / 0.1543 / 0.1510 | 0.962 / 0.962 / 0.962 |

The fixed-ψ values are from `outputs/analysis/proposals_goal/curve/summary.txt`.

### π's mean chunk in the closed loop: `v7_mean_flow`, `v7_mean_flow_full_sgd`

Runs `outputs/closed_loop/v7_mean_flow` and `outputs/closed_loop/v7_mean_flow_full_sgd`, the v6 protocol (`docs/proposal_branch.md`, "v6") with each stage one, its ψ table and its π:

- `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`, π step 3000, candidate settings recorded as in v6 (scales 0.25, 0.5, 1.0, dct 1, seed 0; K = 1 draws no noise);
- `--checkpoint` the stage one, `--state-table` its ψ table (never queried for retrieval under the proposal source);
- seed 0, 300 steps, 10 settle steps, traced, `--check-start-observation`;
- task t on the (t mod n)-th GPU of the run's list above. During the sg[D_ψ] run, processes of other users occupied CUDA 1, 4 and 6 of node-61 as well.
- **Start observation**: for every task, in both runs, the check of the first start prints the same image means, token cosines to the cached starts and proprio distance as v6.

Each run paired with `v6_mean_default` (fixed ψ, π `goal_proposal`) by (task, init); exact two-sided McNemar test (`summarize_closed_loop.py --reference`, outputs `tmp/runlogs/s3p27/v7_mean_flow{,_full_sgd}_summary.txt`):

| Task | Text | success: v6 / flow (z) / sg[D_ψ] | flow (z) only / v6 only, p | sg[D_ψ] only / v6 only, p | decisions: flow (z) / sg[D_ψ] |
|---|---|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 0.96 / 0.98 / 1.00 | 2 / 1, 1 | 2 / 0, 0.5 | 12.7 / 12.8 |
| 1 | open the top drawer and put the bowl inside | 0.78 / 0.66 / 0.74 | 8 / 14, 0.29 | 9 / 11, 0.82 | 22.5 / 21.6 |
| 2 | push the plate to the front of the stove | 0.94 / 0.98 / 0.90 | 3 / 1, 0.625 | 3 / 5, 0.73 | 14.5 / 15.7 |
| 3 | put the bowl on the plate | 0.74 / 0.96 / 0.98 | 13 / 2, 0.0074 | 12 / 0, 4.9e-4 | 9.5 / 8.5 |
| 4 | put the bowl on the stove | 0.60 / 0.98 / 1.00 | 19 / 0, 3.8e-6 | 20 / 0, 1.9e-6 | 10.0 / 9.5 |
| 5 | put the bowl on top of the cabinet | 0.86 / 0.96 / 0.88 | 7 / 2, 0.18 | 5 / 4, 1 | 9.9 / 11.4 |
| 6 | put the cream cheese in the bowl | 0.62 / 0.72 / 0.80 | 11 / 6, 0.33 | 15 / 6, 0.078 | 15.6 / 14.1 |
| 7 | put the wine bottle on the rack | 0.80 / 0.90 / 0.84 | 9 / 4, 0.27 | 8 / 6, 0.79 | 15.5 / 16.6 |
| 8 | put the wine bottle on top of the cabinet | 0.98 / 0.90 / 0.94 | 1 / 5, 0.22 | 1 / 3, 0.625 | 12.2 / 11.1 |
| 9 | turn on the stove | 1.00 / 1.00 / 0.98 | 0 / 0, 1 | 0 / 1, 1 | 7.8 / 8.4 |
| **all** | | **0.828 / 0.904 / 0.906** | **73 / 35, 3.3e-4** | **75 / 36, 2.7e-4** | **13.0 / 13.0** |

- **Per episode**: v6 had 14.7 decisions and 143.2 env steps; flow (z) 126.3 and sg[D_ψ] 125.6 env steps.
- **sg[D_ψ] against flow (z)**: 37 / 36 discordant, p = 1 (`tmp/runlogs/s3p27/v7_full_sgd_vs_flow_summary.txt`).

Failure categories (`classify_closed_loop_failures.py`, outputs `tmp/runlogs/s3p27/v7_mean_flow{,_full_sgd}_failure_classification.txt`):

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| v6, fixed ψ | 414 | 0 | 4 | 8 | 54 | 17 | 3 |
| v7, flow (z) | 452 | 0 | 5 | 6 | 22 | 13 | 2 |
| v7, sg[D_ψ] | 453 | 0 | 3 | 5 | 16 | 19 | 4 |

"Lifted, not at target" failures by task (0–9), v6 / flow (z) / sg[D_ψ]:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 0 | 6 | 1 | 12 | 19 | 6 | 9 | 1 | 0 | 0 |
| flow (z) | 0 | 12 | 0 | 2 | 0 | 1 | 3 | 2 | 2 | 0 |
| sg[D_ψ] | 0 | 5 | 0 | 0 | 0 | 2 | 7 | 1 | 1 | 0 |

- **Scope of the comparison**: the π of different stage ones differ in their ψ and q_l inputs, which also changes their training trajectories; each π was trained once (seed 0), so a difference between runs is not attributed to ψ alone. Render differences between runs of the same π changed the success of 3 of 500 episodes in phase 13.

### Reading

- **Δ on flow (z)**: as accurate as on the fixed stage one. HELDOUT demonstration error 0.238 against 0.248; counterfactual errors within 0.04 of the fixed +A head in every group; ranking correlation 0.732 against 0.722. Its top-1 agreement with the true best candidate is lower (0.372 against 0.434), while the true best candidate is in its top 3 as often (0.67 against 0.65).
- **Δ on sg[D_ψ]**: less accurate relative to the effect size. ψ moves about half as far per chunk (median demonstration ‖z‖ 2.13 against 4.00), and the relative error is higher in every group: 0.343 on HELDOUT demonstrations, 0.38–0.60 on the counterfactual groups, 0.97 for stay. The ranking correlation is 0.625 and the top-1 agreement 0.225. False positives stay low (4–10% per group).
- **π**: HELDOUT MSE 0.263 (flow (z)) and 0.261 (sg[D_ψ]) against 0.274, with the same TRAIN fit (0.015).
- **Closed loop**: π's mean chunk succeeds in 0.904 (flow (z)) and 0.906 (sg[D_ψ]) of the 500 episodes against 0.828 on the fixed stage one (73 / 35, p = 3.3e-4; 75 / 36, p = 2.7e-4); the two flow stage ones do not differ (37 / 36). The gain is in placing the bowl (tasks 3 and 4: 32 / 2 and 32 / 0), where the fixed-ψ π lifts the bowl without placing it (31 such failures in the two tasks, 2 and 0 on the flow stage ones). 0.904 and 0.906 are the baselines of these stage ones for phases 23 and 26.

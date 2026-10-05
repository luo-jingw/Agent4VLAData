# Velocity query v(o, l)

The velocity query v(o, l) is a state-dependent match query. It replaces the text-only direction q_l in the match m(z, l) = cos(z, v(o, l)); candidates, effects and the stay rule are unchanged. v is trained by regression on a velocity of ψ along the demonstrations of the task named by l. A demonstration has frames 0 … T, and the regression target at frame t is one of two forms (`VelocityTarget`):

- **episode mean** (`EPISODE_MEAN`): v(o_t, l) ≈ sg[(ψ(o_T) − ψ(o_0)) / T], the velocity of the straight path from ψ(o_0) to ψ(o_T), the same at every frame of an episode;
- **remaining** (`REMAINING`): v(o_t, l) ≈ sg[(ψ(o_T) − ψ(o_t)) / (T − t)], t ≤ T − 1, the remaining displacement over the remaining frames. This is the conditional flow-matching velocity from the current state to the episode's end state.

Where ψ(o_t) lies at t/T of the segment from ψ(o_0) to ψ(o_T), the two targets are equal. Elsewhere the remaining target points at the end state and changes with the state. Both are in ψ units per frame. Stage one (ψ, q_l), Δ and π stay frozen.

This page covers:

- the model and the target option;
- four training runs: the episode mean on the fixed ψ, and the remaining target on the fixed ψ, the projected-flow ψ and the corrected full-space ψ;
- offline checks against q_l and against the text-only task means: velocity-field consistency, within-task constancy, selection consistency (true and predicted effects) and multimodality.

The offline checks confirm that v learned its target and does not reduce to one direction per task. They do not gate the closed loop. Metric definitions follow `docs/consistency_checks.md`.

Stage ones (step 5000, `outputs/runs/`):

| ψ | Run | Path target | ψ effective rank, HELDOUT |
|---|---|---|---|
| fixed | `goal_instruction_contrast` | perpendicular, U coordinates | 16.6 |
| flow | `goal_instruction_contrast_flow` | interpolation, U coordinates | 16.9 |
| corrected full space | `goal_instruction_contrast_flow_full_sigall_sgd` | interpolation in ψ, SIGReg on all frames, D_ψ without gradient (`docs/stage_one_flow.md`) | 8.56 |

## Model (`ceam/model/velocity_query.py`)

`VelocityQuery(VelocityQueryConfig)` maps (current `FrameBatch`, ψ(o) [..., 256], `TextTokens` [..., 64, 1152] with mask [..., 64]) to v [..., 256], in ψ units per frame. The model is the same for both targets.

- **Sequence of 195 tokens**, in this order:
  - readout token, a learned vector;
  - ψ token and proprio token, each from its own linear projection, with no position or type embedding;
  - 128 visual tokens: projection 768 → 512, plus a view embedding (agent, wrist) and a position embedding shared by the two views;
  - 64 text tokens: projection 1152 → 512, plus a learned text position embedding.
  - Every embedding starts from 𝒩(0, 0.02²).
- **Encoder**: 4 pre-norm Transformer encoder layers (width 512, 8 heads, FFN 2048, GELU, no dropout), bidirectional self-attention.
- **Mask**: one key padding mask [n, 195], True exactly at the text positions whose cached mask is False.
  - The readout, ψ, proprio and visual positions are never masked, so no row is fully masked.
  - SigLIP's pad id equals its `</s>` id, so `</s>` is masked too; the visible text keys are the instruction's word pieces.
  - Padded text positions still pass through the layers as queries, but no position attends to them.
- **Output**: LayerNorm and a linear map 512 → 256 on the readout position. The linear map starts at zero weight and bias, so an untrained query outputs v = 0.
- A trailing-shape or leading-dim mismatch raises `ValueError`.

Parameters (13,929,728):

| Part | Parameters |
|---|---|
| encoder layers × 4 | 12,609,536 |
| text projection 1152 → 512 | 590,336 |
| visual projection 768 → 512 | 393,728 |
| ψ projection 256 → 512 | 131,584 |
| output LayerNorm + linear 512 → 256 | 1,024 + 131,328 |
| visual position embedding (64) / text position embedding (64) / view embedding (2) | 32,768 / 32,768 / 1,024 |
| proprio projection / readout token | 5,120 / 512 |

Checks (`tmp/p22/inspect_velocity_query.py`, CPU):

- Output [2, 3, 256] for inputs with leading dims [2, 3]; max |v| = 0 at construction.
- With a random output layer:
  - replacing every padded text token by large random values changes v by exactly 0;
  - changing one visible word piece, ψ, proprio or the last wrist token changes v in every sample;
  - removing the mask changes v by up to 0.78.
- Flattening the leading dims gives the same output exactly; permuting the batch permutes the output (max |Δ| 8e-7).
- Train and eval mode agree to 9e-7.
- Config JSON round trip and checkpoint round trip (64 tensors) are exact.

## Training

### Code

- **Model**: `ceam/model/velocity_query.py` (`VelocityQuery`, `VelocityQueryConfig`).
- **Target**: `ceam/data/velocity_target.py` (`VelocityTarget`).
  - `frame_counts(lengths)`: the frames at which the target is defined, t ∈ {0, …, count − 1} with count = length (episode mean) or length − 1 (remaining).
  - `values(ψ(o_t), ψ(o_0), ψ(o_T), t, T)`: the targets [n, d], in the dtype of the inputs.
- **Data**: `ceam/data/velocity_sampler.py` (`VelocitySampler`, `VelocitySamplerConfig`), `ceam/data/velocity_batch.py` (`VelocityBatch`, which carries t as `frame_index` and T as `last_index`).
- **Configuration**: `ceam/training/velocity_config.py` (`VelocityTrainingConfig`).
- **Checkpoints**: `ceam/training/velocity_checkpoint_store.py` (`VelocityCheckpointStore`, `LoadedVelocity`).
- **Trainer**: `ceam/training/velocity_trainer.py` (`VelocityTrainer`, `VelocityMetrics`, `VelocityEvaluation`, `VelocityAccuracy`, `TargetStatistics`, `DemoFrameTable`, `VelocitySummary`).
- **Entry point**: `scripts/train_velocity_query.py --config configs/velocity/<name>.json`.

### Configuration and data

- The JSON config has the dataclass fields and uses the `$CEAM_ROOT` / `$CEAM_DATA_ROOT` path conventions of stage two. The feature cache, dataset, instructions and split come from the referenced stage-one checkpoint.
- **`target`**: `"episode_mean"` or `"remaining"`.
  - It is optional when read (`OPTIONAL_FIELDS`); a config without it trains the episode mean, so configs and checkpoints written before the field existed load unchanged.
  - It is always written. Code from before the field rejects configs that contain it, including the configs embedded in the remaining-target checkpoints.
- **Configs**: `configs/velocity/goal_velocity.json` (episode mean, fixed ψ, no `target` key); `goal_velocity_remaining_{fixed,flow,flow_full_sgd}.json` (remaining target on the three stage ones). They differ from `goal_velocity.json` only in `target`, `stage_one_checkpoint` and `run_dir`.
- A checkpoint holds the v state dict, the config JSON and the step; the stage-one parameters are not copied.
- An existing run directory is refused.
- **Samples** (`VelocitySampler`, TRAIN, seed 0): uniform with replacement over every (episode, t) at which the target is defined, T = length − 1:
  - episode mean: t ∈ {0, …, T}, 47,601 TRAIN frames;
  - remaining: t ∈ {0, …, T − 1}, 47,211 TRAIN frames, so T − t ≥ 1.
- **Text**: per frame, one instruction variant of the episode's task, uniform over its 5 variants (`InstructionTable`), with the cached SigLIP tokens of the feature cache. Inference uses the original instruction (variant 0).
- **ψ**: a table of every feature-cache row, computed once with the frozen trunk in float32.
- **Target**: `VelocityTarget.values` from the ψ table, constant in the loss (stop-gradient):
  - episode mean: (ψ[end] − ψ[start]) / T;
  - remaining: (ψ[end] − ψ[row]) / (T − t).

### Loss and optimizer

- **Loss**: MSE between v̂ and v over batch and the 256 dims, on the raw velocity vector, in float32.
- **Recipe**: the stage-two recipe, the same for both targets.
  - AdamW, lr 3e-4; weight decay 0.05 only on matrices whose name contains neither "token" nor "embedding" (21 tensors, 13,832,704 parameters; 43 tensors, 97,024 parameters without decay).
  - Linear warmup over 150 steps, then cosine decay to 0 at 3000 steps.
  - Batch 256, bf16 autocast, global gradient-norm clipping at 1.0, seed 0.
  - The targets are small (per-dim RMS 0.02), but AdamW's update does not depend on the gradient scale. The gradient norm stays at 1e-4 to 2e-2, below the clip.

### Metrics

- `metrics.jsonl`, every 50 steps, on the training batch: MSE; the target's mean square (the MSE of v̂ = 0) and mean norm; the mean ‖v̂‖; median ‖v̂ − v‖ / ‖v‖; mean cos(v̂, v); gradient norm; steps/s; sampling time; peak memory.
- `heldout.jsonl`, every 250 updates: the same accuracy with the original instruction, in eval mode with bf16 autocast, over every HELDOUT frame at which the target is defined: 5,294 frames for the episode mean, 5,251 for the remaining target.
- `target_statistics.json`: the target's scale per split, and two references:
  - v̂ = 0;
  - the text-only task mean, v̂ = the mean target of the frame's task over the TRAIN frames. This is the best text-only predictor of the regression target.
  - ‖v‖ per episode is taken at t = 0, where both targets equal (ψ_T − ψ_0) / T.
- A checkpoint every 1000 updates.

### Checks of the target option

Scripts and outputs: `outputs/analysis/velocity_query_remaining/checks/`.

- **Sampler** (`check_velocity_sampler.{py,txt}`, TRAIN, seed 0):
  - Episode mean: the draw arrays (rows, start rows, end rows, T, task) equal those of the sampler at the base commit `54fb5f2`, in dtype and value. The first three batches have the same rows, variants and T. t ranges over 0 … T, including 390 frames with t = T.
  - Remaining: T − t ranges over 1 … 346. No frame has t = T; 390 frames have t = T − 1. Three drawn batches have min T − t = 1.
  - In both, row − start row = t and end row − start row = T in every frame.
- **Target arithmetic** (`check_velocity_config.{py,txt}`), 1,000 random segments, float32:
  - frames on the segment at t/T: the two targets agree to 4.6e-7;
  - frames off the segment: mean cosine between the two targets 0.81;
  - the remaining target equals (ψ_T − ψ_t) / (T − t) exactly.
- **Configs** (same script):
  - Every file in `configs/velocity/`, `outputs/runs/goal_velocity_fixed/config.json` and the config embedded in its step-3000 checkpoint load, the last two as the episode mean.
  - JSON round trips are exact. Written back, a config without the key differs from the file only by the added `"target": "episode_mean"`.
  - An unknown target value and an unknown key are rejected.
- **Bit identity of the episode mean** (`compare_bit_identity.{py,txt}`):
  - Setup: `goal_velocity.json` shortened to 300 steps, logging every 10 steps, HELDOUT every 100, checkpoint at 300 (`bit_identity_{base,nokey,key}.json`).
  - It was trained with the base commit `54fb5f2`, then with the training code of `c023d89` without the key and with `"target": "episode_mean"`. The runs went one after another on node-61 CUDA 0 (L40S), the GPU of the recorded run.
  - All 31 logged steps are bit-identical in every field except timing and peak memory, for both new runs.
  - Peak memory is 1,024 bytes higher: the [256] float32 tensor of t.
  - All 3 HELDOUT evaluations are bit-identical; all 64 parameter tensors at step 300 are bitwise equal.
  - Steps 0, 50 and 100 lie in the shared warmup. They are bit-identical to the recorded run `goal_velocity_fixed`, e.g. MSE 4.1117286309599876e-04 at step 0.
  - `target_statistics.json`: every entry except the text-only task-mean reference is identical.
  - The task-mean reference differs by up to 4.4e-7 relative between any two runs, including the base run and the recorded run, which use the same code (2.1e-7). The per-task sums use CUDA `index_add_`, whose summation order is not fixed (`compare_target_statistics.txt`).

### Target scale

Frames at which the target is defined, each weighted equally. The last column is the share of the target's mean square explained by the text-only task mean, 1 − MSE(task mean) / MSE(0):

| ψ | Target | Split | Frames | mean ‖v‖ | MSE of v̂ = 0 | task mean: MSE / rel. error / cos | explained |
|---|---|---|---|---|---|---|---|
| fixed | episode mean | TRAIN | 47,601 | 0.307 | 4.04e-4 | 3.12e-5 / 0.274 / 0.964 | 0.92 |
| fixed | episode mean | HELDOUT | 5,294 | 0.285 | 3.47e-4 | 5.17e-5 / 0.340 / 0.946 | 0.85 |
| fixed | remaining | TRAIN | 47,211 | 0.420 | 7.89e-4 | 1.39e-4 / 0.349 / 0.938 | 0.82 |
| fixed | remaining | HELDOUT | 5,251 | 0.414 | 8.20e-4 | 1.97e-4 / 0.369 / 0.934 | 0.76 |
| flow | remaining | TRAIN | 47,211 | 0.343 | 5.08e-4 | 1.03e-4 / 0.377 / 0.911 | 0.80 |
| flow | remaining | HELDOUT | 5,251 | 0.337 | 5.01e-4 | 1.23e-4 / 0.390 / 0.904 | 0.75 |
| corrected full space | remaining | TRAIN | 47,211 | 0.286 | 4.11e-4 | 4.15e-5 / 0.179 / 0.990 | 0.90 |
| corrected full space | remaining | HELDOUT | 5,251 | 0.292 | 4.82e-4 | 1.33e-4 / 0.224 / 0.972 | 0.72 |

- **‖v‖ per episode at t = 0**, min / median / max:
  - fixed: TRAIN 0.109 / 0.355 / 0.546, HELDOUT 0.160 / 0.330 / 0.499;
  - flow: TRAIN 0.100 / 0.302 / 0.535, HELDOUT 0.142 / 0.279 / 0.477;
  - corrected full space: TRAIN 0.069 / 0.276 / 0.444, HELDOUT 0.093 / 0.254 / 0.443.
- **Size of the remaining target**: it is larger on average than the episode mean, 0.42 against 0.31 on the fixed ψ. In the fixed ψ, progress along the start-to-end segment lags t/T (p = 0.34 at t/T = 0.5, `docs/stage_one_flow.md`), so the remaining displacement per remaining frame exceeds the episode mean.
- **What the task explains**:
  - The task explains 82% of the remaining target's mean square on TRAIN and 76% on HELDOUT, against 92% and 85% for the episode mean.
  - The rest depends on the episode and, for the remaining target, on the frame.
  - On the corrected full-space ψ, the task explains 90% of the remaining target on TRAIN and 72% on HELDOUT.
  - That stage one places the frames close to their start-to-end segment (HELDOUT interpolation residual, median 0.0065), and all episodes start near one point (spread of ψ_0 0.90 against ‖ψ_T − ψ_0‖ 27.5, `docs/stage_one_flow.md`).
  - So ψ_T − ψ_0 has almost one direction per task, and the remaining target differs from the episode mean mainly through the timing along the segment.

### Runs

| Run (`outputs/runs/`) | Target | ψ | Commit | Host / GPU | Time | Speed, overall | Peak memory |
|---|---|---|---|---|---|---|---|
| `goal_velocity_fixed` | episode mean | fixed | `bd554c9` | node-61, CUDA 0, L40S | 371.5 s | 8.07 steps/s | 4.58 GiB |
| `goal_velocity_remaining_fixed` | remaining | fixed | `c023d89` | node-61, CUDA 1, L40S | 351.5 s | 8.53 steps/s | 4.58 GiB |
| `goal_velocity_remaining_flow` | remaining | flow | `c023d89` | node-61, CUDA 2, L40S | 337.9 s | 8.88 steps/s | 4.58 GiB |
| `goal_velocity_remaining_flow_full_sgd` | remaining | corrected full space | `35b870c` | node-61, CUDA 0, L40S | 328.8 s | 9.12 steps/s | 4.58 GiB |

- Each run is 3000 steps, seed 0. Checkpoints are `checkpoints/step_00{1,2,3}000.pt`; the checks use step 3000.
- Sampling takes 0.03–0.05 s per step.
- Opening the feature cache in memory takes 43–54 s, and the ψ table 9–19 s.
- `35b870c` merges the stage-one config field `path_normalizer_gradient` of phase 24 (`697b424`), which the corrected full-space checkpoint contains; the velocity-query code is that of `c023d89`.
- Launchers and logs: `outputs/analysis/velocity_query/logs/` (episode mean) and `outputs/analysis/velocity_query_remaining/logs/` (remaining target).

Training batch (TRAIN, random variants) and HELDOUT (original instruction), after the given number of updates. The training-batch columns of row 3000 are the batch of the last update. Each run is evaluated against its own target.

Episode mean, fixed ψ:

| Step | lr | TRAIN MSE | TRAIN rel. error | TRAIN cos | HELDOUT MSE | HELDOUT rel. error | HELDOUT cos | HELDOUT mean ‖v̂‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 2e-6 | 4.11e-4 | 1.000 | 0.000 | – | – | – | – |
| 50 | 1.0e-4 | 4.51e-5 | 0.336 | 0.945 | – | – | – | – |
| 250 | 3.0e-4 | 3.30e-5 | 0.282 | 0.960 | 5.72e-5 | 0.355 | 0.940 | 0.302 |
| 500 | 2.9e-4 | 1.89e-5 | 0.216 | 0.976 | 6.11e-5 | 0.373 | 0.936 | 0.302 |
| 1000 | 2.4e-4 | 2.52e-6 | 0.075 | 0.996 | 6.03e-5 | 0.391 | 0.938 | 0.306 |
| 1500 | 1.6e-4 | 1.09e-6 | 0.051 | 0.998 | 6.08e-5 | 0.388 | 0.939 | 0.309 |
| 2000 | 8.2e-5 | 7.52e-7 | 0.042 | 0.999 | 6.07e-5 | 0.383 | 0.939 | 0.308 |
| 3000 | 0 | 5.44e-7 | 0.037 | 0.999 | 6.14e-5 | 0.387 | 0.939 | 0.309 |

Remaining target, fixed ψ:

| Step | lr | TRAIN MSE | TRAIN rel. error | TRAIN cos | HELDOUT MSE | HELDOUT rel. error | HELDOUT cos | HELDOUT mean ‖v̂‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 2e-6 | 8.42e-4 | 1.000 | 0.000 | – | – | – | – |
| 50 | 1.0e-4 | 1.22e-4 | 0.334 | 0.947 | – | – | – | – |
| 250 | 3.0e-4 | 4.27e-5 | 0.196 | 0.983 | 1.22e-4 | 0.234 | 0.975 | 0.435 |
| 500 | 2.9e-4 | 3.23e-5 | 0.165 | 0.987 | 1.05e-4 | 0.204 | 0.977 | 0.417 |
| 1000 | 2.4e-4 | 1.43e-5 | 0.112 | 0.991 | 9.74e-5 | 0.176 | 0.982 | 0.412 |
| 1500 | 1.6e-4 | 7.82e-6 | 0.079 | 0.996 | 1.01e-4 | 0.171 | 0.984 | 0.425 |
| 2000 | 8.2e-5 | 5.56e-6 | 0.062 | 0.997 | 9.95e-5 | 0.167 | 0.984 | 0.423 |
| 2500 | 2.2e-5 | 3.63e-6 | 0.056 | 0.997 | 9.72e-5 | 0.165 | 0.985 | 0.422 |
| 3000 | 0 | 4.13e-6 | 0.053 | 0.997 | 9.74e-5 | 0.164 | 0.985 | 0.422 |

Remaining target, flow ψ:

| Step | lr | TRAIN MSE | TRAIN rel. error | TRAIN cos | HELDOUT MSE | HELDOUT rel. error | HELDOUT cos | HELDOUT mean ‖v̂‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 2e-6 | 5.27e-4 | 1.000 | 0.000 | – | – | – | – |
| 50 | 1.0e-4 | 7.84e-5 | 0.330 | 0.940 | – | – | – | – |
| 250 | 3.0e-4 | 3.41e-5 | 0.214 | 0.978 | 6.24e-5 | 0.254 | 0.963 | 0.340 |
| 500 | 2.9e-4 | 2.83e-5 | 0.171 | 0.985 | 5.57e-5 | 0.207 | 0.971 | 0.347 |
| 1000 | 2.4e-4 | 1.18e-5 | 0.121 | 0.991 | 5.08e-5 | 0.182 | 0.975 | 0.344 |
| 1500 | 1.6e-4 | 6.61e-6 | 0.094 | 0.994 | 5.13e-5 | 0.180 | 0.975 | 0.347 |
| 2000 | 8.2e-5 | 4.32e-6 | 0.069 | 0.996 | 5.09e-5 | 0.174 | 0.977 | 0.344 |
| 2500 | 2.2e-5 | 2.84e-6 | 0.060 | 0.996 | 5.01e-5 | 0.174 | 0.976 | 0.346 |
| 3000 | 0 | 3.88e-6 | 0.059 | 0.996 | 5.05e-5 | 0.176 | 0.977 | 0.345 |

Remaining target, corrected full-space ψ:

| Step | lr | TRAIN MSE | TRAIN rel. error | TRAIN cos | HELDOUT MSE | HELDOUT rel. error | HELDOUT cos | HELDOUT mean ‖v̂‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 2e-6 | 4.55e-4 | 1.000 | 0.000 | – | – | – | – |
| 50 | 1.0e-4 | 4.75e-5 | 0.281 | 0.959 | – | – | – | – |
| 250 | 3.0e-4 | 2.70e-5 | 0.243 | 0.978 | 1.37e-4 | 0.280 | 0.960 | 0.297 |
| 500 | 2.9e-4 | 2.49e-5 | 0.187 | 0.984 | 1.29e-4 | 0.241 | 0.966 | 0.274 |
| 1000 | 2.4e-4 | 1.26e-5 | 0.162 | 0.988 | 1.33e-4 | 0.205 | 0.968 | 0.275 |
| 1500 | 1.6e-4 | 7.30e-6 | 0.113 | 0.994 | 1.33e-4 | 0.196 | 0.972 | 0.291 |
| 2000 | 8.2e-5 | 4.88e-6 | 0.083 | 0.996 | 1.30e-4 | 0.181 | 0.974 | 0.293 |
| 2500 | 2.2e-5 | 3.26e-6 | 0.072 | 0.995 | 1.28e-4 | 0.176 | 0.974 | 0.289 |
| 3000 | 0 | 4.01e-6 | 0.073 | 0.996 | 1.28e-4 | 0.172 | 0.974 | 0.290 |

- **Episode mean**:
  - TRAIN error falls to 0.037, well below the task-mean reference (0.274): v learns each TRAIN episode's own velocity.
  - On HELDOUT the error is lowest at the first evaluation (step 250), then stays at 0.38–0.39, above the task mean (0.340). The per-episode part of the target does not transfer to new episodes.
  - A 200-step smoke run (20 warmup steps) ended at HELDOUT MSE 5.26e-5, rel. error 0.340, cos 0.946, which is the task-mean level.
- **Remaining target**:
  - On HELDOUT the error falls through training to 0.164 on the fixed ψ and 0.176 on the flow ψ. The task-mean references are 0.369 and 0.390.
  - The HELDOUT MSE is 0.49 and 0.41 times that of the task mean: 9.74e-5 against 1.97e-4, and 5.05e-5 against 1.23e-4.
  - The part of the remaining target that the task does not explain transfers to new episodes.
  - TRAIN error ends at 0.053 and 0.059, a third of the HELDOUT error.
  - On the corrected full-space ψ, the HELDOUT error ends at 0.172 against 0.224 for the task mean, but the HELDOUT MSE only at 1.28e-4 against 1.33e-4. v adds little to the task mean there, whose TRAIN relative error is already 0.179.

## Offline checks

### Scorer

`scripts/score_consistency.py` (`docs/consistency_checks.md`) scores velocity queries as follows.

- **Velocity queries** (`--velocity-checkpoints` / `--velocity-names`): each is scored as its own entry.
  - Its ψ is that of its stage one, which must be one of `--checkpoints`; the ψ tables are shared.
  - Its direction q is v(o, l) of the current frame o, with ψ(o) and each task's original instruction, in float32 without autocast. For demonstration frames, o is the frame t; for the restored states, o is the stored simulator frame.
  - The other-task control queries v with each other task's instruction at the same frame.
  - A velocity query has no ĝ or U, so only q enters its analyses. In the degeneracy control, same-task pairs of v count, as for the obs checkpoint's q(o, l).
- **Expected velocities**: every analysis uses three:
  - the demonstration's displacement ψ_{t+10} − ψ_t;
  - the episode velocity ψ_T − ψ_0;
  - the remaining displacement ψ_T − ψ_t, which has the direction of the remaining target.
- **Task-mean references** (`--task-mean-velocity`): every stage one that a velocity query uses gets two text-only reference entries. Each is the text-only optimum of the regression with its target.
  - `<stage one>_task_mean`: the mean of (ψ_T − ψ_0) / T over the task's TRAIN frames t ∈ {0, …, T};
  - `<stage one>_task_mean_remaining`: the mean of (ψ_T − ψ_t) / (T − t) over the task's TRAIN frames t ∈ {0, …, T − 1};
  - both weight frames as the sampler draws them, in float64.
- **Predicted effects** (`--dynamics-checkpoint`): a stage-two checkpoint whose stage one is among `--checkpoints`.
  - Δ̂_c = Δ(o, a_c) uses the stored frame o, ψ(o) and the stored executed chunk, standardized with the stage-two run's action cache, in float32 without autocast, as in the closed loop.
  - For every entry on that ψ, the sets `counterfactual_predicted` and `proposal_predicted` repeat the selection with m = cos(Δ̂_c, q). The choice, the stay decision, Spearman(m, ·) and the other-task control use Δ̂; every evaluation cosine stays cos(z_c, v) with the true effect.
- **Paired differences**, with 95% episode-bootstrap intervals:
  - every entry against the first, the stage one's q_l;
  - each velocity query against its own stage one, against the task-mean reference of its own target, and against every earlier velocity query on the same stage one.
- **Per t/T quarter**: the mean, quantiles, share > 0 and own − other.
- **Degeneracy control**: pairs from different tasks, from the same task and different episodes, and different frames of the same episode.
  - Vectors: the displacement, the episode velocity, the directions, and the two regression targets per frame.
  - For each vector also the relative spread around the task mean, sqrt(Σ_i ‖x_i − x̄_task(i)‖² / Σ_i ‖x_i‖²). It is 0 when x is constant within every task.

Checks (`outputs/analysis/velocity_query_remaining/checks/`):

- **Unchanged Phase 20 results**: with the fixed stage one alone and `goal_dynamics_short` (`outputs/analysis/velocity_query/check_scorer_fixed_short_6af22fb`, commit `6af22fb`), all 892 table rows and 1,548 state rows of Phase 20's `fixed` (`outputs/analysis/consistency/stage_one_step5000_f64`) are byte-identical.
- **Unchanged velocity-query results**: the command of the episode-mean scoring (`outputs/analysis/velocity_query/consistency_fixed`, `6af22fb`) was rerun at `eae446b` (`check_scorer_phase22_eae446b`). Every one of its 3,449 table rows, 2,321 comparison rows and 6,193 state rows is present byte-identically among the rerun's rows (`compare_regression.txt`).
- **Per-quarter control**: the runs below at `08f40bb` reproduce every row of the same runs at `eae446b` (`consistency_{fixed,flow}`) byte-identically and add the per-quarter own − other (`compare_rerun_08f40bb.txt`).
- **Multimodality check**: rerun at `eae446b` for the episode-mean query, `tables.csv` (393 rows) and `frames.csv` (97,131 rows) are byte-identical to `outputs/analysis/velocity_query/multimodality_fixed`.
- **Pairwise statistics**: on 700 synthetic vectors in 5 tasks and 35 episodes, the same-episode, same-task and different-task means and RMS cosines, and the relative spread, equal brute-force values to 1.1e-16 (`check_pairwise_episode.txt`).
- **Predicted effects**: with `goal_dynamics_short`, the Δ̂ used when the candidates were recorded, cos(Δ̂, q_l) recomputed from the stored frames and chunks equals the recorded predicted match within 2.9e-6 over the 1,806 counterfactual candidates. The true match agrees within 9.4e-6 (counterfactual) and 4.6e-6 (π).
- Every scoring run used node-61, an L40S.

### Runs

In `outputs/analysis/velocity_query_remaining/`, commit `08f40bb`, node-61 (L40S):

- **`consistency_fixed_08f40bb`** (CUDA 3, 353 s), fixed ψ:
  - q_l (`fixed`);
  - v, episode mean (`v_fixed`, `goal_velocity_fixed` step 3000);
  - v, remaining (`v_rem_fixed`, `goal_velocity_remaining_fixed` step 3000);
  - the task means of both targets;
  - predicted effects from `goal_dynamics_perturb` step 3000 (Δ trained on demonstrations + data A), the Δ of the closed loop.
- **`consistency_flow_08f40bb`** (CUDA 4, 181 s), flow ψ: q_l (`flow`); v, remaining (`v_rem_flow`, `goal_velocity_remaining_flow` step 3000); the task means of both targets. There is no Δ on this ψ, so only true effects are scored.
- **`consistency_flow_full_sgd_35b870c`** (commit `35b870c`, scorer as at `08f40bb`, CUDA 0, 185 s), corrected full-space ψ: q_l (`flow_full_sgd`); v, remaining (`v_rem_flow_full_sgd`, `goal_velocity_remaining_flow_full_sgd` step 3000); the task means of both targets; true effects only.

Store join, store consistency and the reproduction of the Phase 12 set give the same values as in Phase 20. The episode-mean query's step-1000 checkpoint scored within ±0.03 of step 3000 in every velocity-field and selection cell (`outputs/analysis/velocity_query/consistency_fixed`).

Column labels in the tables below:

- **q_l**: the stage one's text-only direction;
- **v, ep. mean** and **v, remaining**: the velocity queries with the two targets;
- **mean, ep.** and **mean, rem.**: the text-only task means of the two targets.

### Velocity-field consistency

HELDOUT demonstration frames with t + 10 ≤ T: 43 episodes, 4,864 frames. Each cell is the mean cosine [95% interval], with own − other in parentheses.

Fixed ψ:

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.40 [+0.38, +0.42] (+0.44) | +0.56 [+0.52, +0.60] (+0.56) | +0.63 [+0.60, +0.67] (+0.25) | +0.57 [+0.53, +0.61] (+0.58) | +0.53 [+0.50, +0.57] (+0.58) |
| ψ_T − ψ_0 | +0.72 [+0.70, +0.73] (+0.80) | +0.94 [+0.93, +0.95] (+0.89) | +0.79 [+0.77, +0.81] (+0.54) | +0.95 [+0.94, +0.95] (+0.92) | +0.81 [+0.79, +0.82] (+0.89) |
| ψ_T − ψ_t | +0.58 [+0.55, +0.60] (+0.64) | +0.83 [+0.81, +0.84] (+0.89) | +0.99 [+0.98, +1.00] (+0.53) | +0.84 [+0.83, +0.85] (+0.91) | +0.95 [+0.93, +0.95] (+0.99) |

Flow ψ:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.34 [+0.33, +0.36] (+0.38) | +0.62 [+0.59, +0.65] (+0.18) | +0.52 [+0.49, +0.55] (+0.51) | +0.49 [+0.46, +0.52] (+0.54) |
| ψ_T − ψ_0 | +0.64 [+0.60, +0.67] (+0.70) | +0.68 [+0.66, +0.71] (+0.44) | +0.90 [+0.88, +0.92] (+0.82) | +0.70 [+0.68, +0.73] (+0.80) |
| ψ_T − ψ_t | +0.50 [+0.49, +0.52] (+0.56) | +0.98 [+0.97, +0.99] (+0.43) | +0.75 [+0.73, +0.77] (+0.83) | +0.92 [+0.90, +0.93] (+0.97) |

Corrected full-space ψ:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.63 [+0.59, +0.67] (+0.70) | +0.80 [+0.74, +0.85] (+0.78) | +0.80 [+0.74, +0.85] (+0.84) | +0.79 [+0.73, +0.85] (+0.84) |
| ψ_T − ψ_0 | +0.76 [+0.75, +0.78] (+0.85) | +0.99 [+0.99, +1.00] (+0.99) | +1.00 [+1.00, +1.00] (+1.06) | +1.00 [+0.99, +1.00] (+1.05) |
| ψ_T − ψ_t | +0.74 [+0.71, +0.76] (+0.82) | +0.98 [+0.96, +0.99] (+0.97) | +0.98 [+0.96, +0.99] (+1.04) | +0.98 [+0.96, +0.99] (+1.04) |

Distribution over the frames, as median / p10 / p90 / share > 0.

Fixed ψ:

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.44 / +0.20 / +0.56 / 0.95 | +0.70 / +0.02 / +0.83 / 0.91 | +0.80 / +0.04 / +0.98 / 0.91 | +0.72 / +0.01 / +0.84 / 0.90 | +0.73 / −0.23 / +0.97 / 0.83 |
| ψ_T − ψ_0 | +0.72 / +0.67 / +0.76 / 1.00 | +0.95 / +0.89 / +0.97 / 1.00 | +0.80 / +0.64 / +0.92 / 1.00 | +0.96 / +0.90 / +0.98 / 1.00 | +0.81 / +0.76 / +0.87 / 1.00 |
| ψ_T − ψ_t | +0.53 / +0.47 / +0.79 / 1.00 | +0.84 / +0.71 / +0.94 / 1.00 | +1.00 / +0.99 / +1.00 / 1.00 | +0.85 / +0.72 / +0.95 / 1.00 | +0.96 / +0.88 / +0.99 / 1.00 |

Flow ψ:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.38 / +0.17 / +0.48 / 0.95 | +0.74 / +0.08 / +0.97 / 0.93 | +0.65 / +0.06 / +0.76 / 0.92 | +0.65 / −0.21 / +0.95 / 0.80 |
| ψ_T − ψ_0 | +0.66 / +0.47 / +0.73 / 1.00 | +0.69 / +0.49 / +0.88 / 1.00 | +0.92 / +0.83 / +0.97 / 1.00 | +0.70 / +0.57 / +0.82 / 1.00 |
| ψ_T − ψ_t | +0.49 / +0.43 / +0.62 / 1.00 | +1.00 / +0.98 / +1.00 / 1.00 | +0.76 / +0.59 / +0.92 / 1.00 | +0.95 / +0.84 / +0.99 / 1.00 |

Corrected full-space ψ:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.72 / +0.32 / +0.82 / 0.94 | +0.96 / +0.28 / +0.99 / 0.92 | +0.96 / +0.25 / +0.99 / 0.91 | +0.96 / +0.25 / +0.99 / 0.91 |
| ψ_T − ψ_0 | +0.79 / +0.68 / +0.83 / 1.00 | +1.00 / +0.99 / +1.00 / 1.00 | +1.00 / +1.00 / +1.00 / 1.00 | +1.00 / +0.99 / +1.00 / 1.00 |
| ψ_T − ψ_t | +0.78 / +0.63 / +0.83 / 1.00 | +1.00 / +0.97 / +1.00 / 1.00 | +1.00 / +0.97 / +1.00 / 1.00 | +1.00 / +0.98 / +1.00 / 1.00 |

Paired differences of the remaining-target v, mean / own − other [95% interval]:

| Velocity | fixed: v, remaining − q_l | fixed: v, remaining − v, ep. mean | fixed: v, remaining − mean, rem. | flow: v, remaining − q_l | flow: v, remaining − mean, rem. | corr. full: v, remaining − q_l | corr. full: v, remaining − mean, rem. |
|---|---|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.23 [+0.21, +0.25] / −0.19 [−0.22, −0.16] | +0.07 [+0.06, +0.08] / −0.31 [−0.35, −0.26] | +0.10 [+0.09, +0.11] / −0.33 [−0.37, −0.28] | +0.28 [+0.26, +0.30] / −0.20 [−0.23, −0.17] | +0.13 [+0.13, +0.14] / −0.36 [−0.39, −0.33] | +0.16 [+0.14, +0.18] / +0.08 [+0.05, +0.10] | +0.00 [+0.00, +0.00] / −0.06 [−0.09, −0.03] |
| ψ_T − ψ_0 | +0.07 [+0.05, +0.10] / −0.26 [−0.30, −0.21] | −0.15 [−0.16, −0.13] / −0.35 [−0.40, −0.30] | −0.02 [−0.02, −0.01] / −0.35 [−0.39, −0.30] | +0.04 [−0.00, +0.09] / −0.26 [−0.33, −0.18] | −0.02 [−0.03, −0.01] / −0.36 [−0.39, −0.32] | +0.23 [+0.21, +0.25] / +0.14 [+0.11, +0.17] | −0.00 [−0.00, −0.00] / −0.06 [−0.10, −0.03] |
| ψ_T − ψ_t | +0.41 [+0.38, +0.44] / −0.11 [−0.14, −0.08] | +0.16 [+0.15, +0.17] / −0.36 [−0.40, −0.31] | +0.04 [+0.04, +0.05] / −0.46 [−0.51, −0.40] | +0.48 [+0.47, +0.50] / −0.13 [−0.18, −0.08] | +0.06 [+0.06, +0.07] / −0.53 [−0.58, −0.48] | +0.24 [+0.22, +0.26] / +0.15 [+0.12, +0.18] | +0.00 [+0.00, +0.00] / −0.06 [−0.10, −0.03] |

With the episode-mean v against q_l on the fixed ψ, the differences are +0.16 [+0.13, +0.19] / +0.12 [+0.09, +0.14] against the displacement and +0.22 [+0.20, +0.24] / +0.09 [+0.08, +0.11] against ψ_T − ψ_0.

Against ψ_{t+10} − ψ_t by t/T quarter, mean (median / p10 / p90 / share > 0). Fixed ψ:

| Entry | t/T 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| q_l | +0.32 (+0.37 / +0.10 / +0.47 / 0.94) | +0.39 (+0.43 / +0.10 / +0.51 / 0.94) | +0.43 (+0.45 / +0.37 / +0.58 / 0.96) | +0.49 (+0.51 / +0.39 / +0.64 / 0.97) |
| v, ep. mean | +0.34 (+0.38 / −0.06 / +0.67 / 0.85) | +0.56 (+0.73 / −0.10 / +0.84 / 0.87) | +0.71 (+0.79 / +0.57 / +0.84 / 0.96) | +0.69 (+0.74 / +0.58 / +0.81 / 0.97) |
| v, remaining | +0.30 (+0.34 / −0.05 / +0.60 / 0.86) | +0.58 (+0.72 / −0.08 / +0.89 / 0.88) | +0.84 (+0.94 / +0.71 / +0.98 / 0.96) | +0.90 (+0.98 / +0.88 / +0.99 / 0.97) |
| mean, ep. | +0.33 (+0.37 / −0.09 / +0.68 / 0.82) | +0.58 (+0.75 / −0.10 / +0.85 / 0.87) | +0.72 (+0.80 / +0.59 / +0.86 / 0.96) | +0.70 (+0.76 / +0.60 / +0.82 / 0.97) |
| mean, rem. | +0.07 (+0.10 / −0.37 / +0.46 / 0.60) | +0.49 (+0.63 / −0.23 / +0.85 / 0.85) | +0.81 (+0.91 / +0.65 / +0.98 / 0.96) | +0.87 (+0.95 / +0.76 / +0.98 / 0.97) |
| v, remaining − q_l | −0.02 [−0.05, +0.01] | +0.19 [+0.12, +0.25] | +0.41 [+0.38, +0.44] | +0.41 [+0.36, +0.46] |
| v, remaining − v, ep. mean | −0.03 [−0.05, −0.01] | +0.02 [−0.01, +0.04] | +0.14 [+0.12, +0.15] | +0.21 [+0.18, +0.23] |
| v, remaining − mean, rem. | +0.23 [+0.21, +0.25] | +0.09 [+0.08, +0.10] | +0.03 [+0.02, +0.04] | +0.02 [+0.01, +0.03] |

Flow ψ:

| Entry | t/T 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| q_l | +0.28 (+0.32 / +0.05 / +0.41 / 0.93) | +0.33 (+0.37 / +0.12 / +0.43 / 0.93) | +0.37 (+0.39 / +0.32 / +0.46 / 0.97) | +0.42 (+0.46 / +0.35 / +0.53 / 0.97) |
| v, remaining | +0.30 (+0.32 / +0.00 / +0.59 / 0.90) | +0.54 (+0.66 / −0.05 / +0.85 / 0.88) | +0.84 (+0.93 / +0.70 / +0.98 / 0.97) | +0.89 (+0.96 / +0.85 / +0.99 / 0.97) |
| mean, ep. | +0.33 (+0.34 / −0.05 / +0.68 / 0.86) | +0.56 (+0.69 / +0.00 / +0.78 / 0.90) | +0.64 (+0.70 / +0.53 / +0.77 / 0.97) | +0.59 (+0.66 / +0.40 / +0.75 / 0.97) |
| mean, rem. | −0.01 (−0.01 / −0.36 / +0.34 / 0.49) | +0.43 (+0.52 / −0.18 / +0.80 / 0.83) | +0.80 (+0.89 / +0.65 / +0.96 / 0.97) | +0.85 (+0.92 / +0.73 / +0.97 / 0.98) |
| v, remaining − q_l | +0.02 [−0.01, +0.06] | +0.22 [+0.16, +0.27] | +0.47 [+0.43, +0.50] | +0.47 [+0.43, +0.50] |
| v, remaining − mean, rem. | +0.31 [+0.29, +0.33] | +0.12 [+0.11, +0.13] | +0.03 [+0.02, +0.05] | +0.04 [+0.02, +0.05] |

Corrected full-space ψ:

| Entry | t/T 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| q_l | +0.60 (+0.68 / +0.20 / +0.80 / 0.95) | +0.59 (+0.73 / −0.03 / +0.82 / 0.88) | +0.68 (+0.74 / +0.53 / +0.83 / 0.97) | +0.67 (+0.74 / +0.58 / +0.81 / 0.97) |
| v, remaining | +0.71 (+0.90 / +0.02 / +0.98 / 0.90) | +0.72 (+0.94 / −0.25 / +0.98 / 0.84) | +0.88 (+0.97 / +0.80 / +0.99 / 0.97) | +0.91 (+0.99 / +0.91 / +1.00 / 0.97) |
| mean, ep. | +0.71 (+0.90 / +0.02 / +0.98 / 0.90) | +0.72 (+0.94 / −0.25 / +0.98 / 0.84) | +0.88 (+0.97 / +0.79 / +0.99 / 0.97) | +0.90 (+0.98 / +0.92 / +0.99 / 0.97) |
| mean, rem. | +0.70 (+0.89 / +0.01 / +0.98 / 0.90) | +0.72 (+0.94 / −0.26 / +0.98 / 0.84) | +0.87 (+0.97 / +0.79 / +0.99 / 0.97) | +0.90 (+0.99 / +0.91 / +1.00 / 0.97) |
| v, remaining − q_l | +0.11 [+0.06, +0.15] | +0.13 [+0.09, +0.17] | +0.20 [+0.18, +0.22] | +0.23 [+0.20, +0.26] |
| v, remaining − mean, rem. | +0.01 [+0.00, +0.01] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.01] | +0.00 [−0.00, +0.00] |

Against ψ_T − ψ_t, the remaining-target v has mean cosine 0.95–1.00 in every quarter on every ψ, with p10 ≥ 0.95.

Other-task control against ψ_{t+10} − ψ_t by t/T quarter: own − other [95% interval], with the other-task cosine in parentheses.

| Entry | t/T 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| fixed: q_l | +0.36 [+0.32, +0.39] (−0.03) | +0.43 [+0.39, +0.46] (−0.04) | +0.48 [+0.44, +0.51] (−0.05) | +0.54 [+0.48, +0.59] (−0.05) |
| fixed: v, ep. mean | +0.24 [+0.18, +0.28] (+0.10) | +0.58 [+0.49, +0.65] (−0.01) | +0.74 [+0.69, +0.79] (−0.03) | +0.76 [+0.69, +0.82] (−0.07) |
| fixed: v, remaining | +0.15 [+0.12, +0.19] (+0.15) | +0.25 [+0.20, +0.30] (+0.33) | +0.29 [+0.24, +0.33] (+0.56) | +0.36 [+0.30, +0.42] (+0.54) |
| fixed: mean, ep. | +0.24 [+0.19, +0.29] (+0.08) | +0.60 [+0.52, +0.68] (−0.03) | +0.76 [+0.71, +0.81] (−0.04) | +0.78 [+0.71, +0.84] (−0.08) |
| fixed: mean, rem. | +0.12 [+0.07, +0.16] (−0.04) | +0.57 [+0.48, +0.66] (−0.08) | +0.85 [+0.78, +0.91] (−0.03) | +0.89 [+0.80, +0.95] (−0.01) |
| fixed: v, remaining − v, ep. mean | −0.08 [−0.11, −0.05] | −0.33 [−0.39, −0.26] | −0.45 [−0.52, −0.38] | −0.41 [−0.47, −0.34] |
| flow: q_l | +0.30 [+0.26, +0.33] (−0.02) | +0.36 [+0.32, +0.39] (−0.03) | +0.42 [+0.39, +0.44] (−0.05) | +0.48 [+0.42, +0.52] (−0.05) |
| flow: v, remaining | +0.08 [+0.06, +0.11] (+0.22) | +0.16 [+0.12, +0.19] (+0.39) | +0.21 [+0.17, +0.25] (+0.63) | +0.30 [+0.25, +0.35] (+0.59) |
| corr. full: q_l | +0.66 [+0.61, +0.72] (−0.07) | +0.66 [+0.57, +0.74] (−0.07) | +0.75 [+0.70, +0.80] (−0.08) | +0.75 [+0.67, +0.81] (−0.08) |
| corr. full: v, remaining | +0.74 [+0.64, +0.84] (−0.03) | +0.72 [+0.59, +0.83] (+0.01) | +0.87 [+0.79, +0.93] (+0.01) | +0.81 [+0.72, +0.89] (+0.10) |
| corr. full: mean, rem. | +0.74 [+0.64, +0.84] (−0.04) | +0.75 [+0.62, +0.87] (−0.03) | +0.93 [+0.87, +0.99] (−0.06) | +0.96 [+0.86, +1.03] (−0.05) |

Over all HELDOUT frames, the remaining-target v queried with another task's instruction still has cosine +0.38 (fixed) and +0.45 (flow) with the frame's own displacement, and +0.46 and +0.55 with its own remaining displacement. On the fixed ψ, the episode-mean v has 0.00 and −0.06 and q_l −0.04 and −0.06. On the corrected full-space ψ, the remaining-target v has +0.01 against the displacement, q_l −0.07.

TRAIN (390 episodes, 43,701 frames), mean (own − other):

| Velocity | fixed: q_l | fixed: v, ep. mean | fixed: v, remaining | fixed: mean, ep. | fixed: mean, rem. | flow: q_l | flow: v, remaining | flow: mean, ep. | flow: mean, rem. |
|---|---|---|---|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.45 (+0.50) | +0.63 (+0.63) | +0.70 (+0.32) | +0.64 (+0.66) | +0.60 (+0.65) | +0.39 (+0.43) | +0.69 (+0.23) | +0.59 (+0.58) | +0.55 (+0.60) |
| ψ_T − ψ_0 | +0.73 (+0.81) | +1.00 (+0.95) | +0.82 (+0.55) | +0.96 (+0.95) | +0.83 (+0.91) | +0.68 (+0.75) | +0.72 (+0.46) | +0.94 (+0.88) | +0.75 (+0.84) |
| ψ_T − ψ_t | +0.58 (+0.64) | +0.82 (+0.87) | +1.00 (+0.53) | +0.84 (+0.91) | +0.95 (+0.99) | +0.51 (+0.57) | +1.00 (+0.43) | +0.75 (+0.84) | +0.93 (+0.97) |

Corrected full-space ψ, TRAIN, q_l / v, remaining / mean, ep. / mean, rem.: +0.72 (+0.80) / +0.93 (+0.91) / +0.92 (+0.98) / +0.92 (+0.98) against ψ_{t+10} − ψ_t; +0.77 (+0.85) / +0.99 (+0.99) / +1.00 (+1.06) / +1.00 (+1.05) against ψ_T − ψ_0; +0.76 (+0.84) / +1.00 (+0.99) / +0.99 (+1.05) / +0.99 (+1.05) against ψ_T − ψ_t.

### Within-task constancy

Cosines between v of different states. The vectors are those of the HELDOUT frames with t + 10 ≤ T. The regression targets of the same frames are given for comparison.

- **same task**: pairs from the same task and different episodes, mean / RMS, with the standard deviation of the pairwise cosine sqrt(RMS² − mean²);
- **same episode**: pairs of different frames of the same episode, mean;
- **different tasks**: mean / RMS;
- **spread**: the relative spread around the task mean, sqrt(Σ_i ‖x_i − x̄_task(i)‖² / Σ_i ‖x_i‖²).

| ψ | Vector | same task: mean / RMS (sd) | same episode | different tasks: mean / RMS | spread |
|---|---|---|---|---|---|
| fixed | v, ep. mean | +0.972 / 0.972 (0.018) | +0.993 | +0.011 / 0.088 | 0.153 |
| fixed | target (ψ_T − ψ_0) / T | +0.896 / 0.897 (0.043) | +1.000 | +0.037 / 0.123 | 0.306 |
| fixed | v, remaining | +0.906 / 0.912 (0.104) | +0.909 | −0.046 / 0.123 | 0.363 |
| fixed | target (ψ_T − ψ_t) / (T − t) | +0.900 / 0.907 (0.117) | +0.908 | −0.046 / 0.124 | 0.429 |
| flow | v, remaining | +0.860 / 0.873 (0.150) | +0.864 | −0.049 / 0.172 | 0.396 |
| flow | target (ψ_T − ψ_t) / (T − t) | +0.855 / 0.869 (0.155) | +0.868 | −0.051 / 0.170 | 0.431 |
| corrected full space | v, remaining | +0.992 / 0.992 (0.012) | +0.992 | −0.055 / 0.127 | 0.190 |
| corrected full space | target (ψ_T − ψ_0) / T | +0.996 / 0.996 (0.009) | +1.000 | −0.056 / 0.121 | 0.146 |
| corrected full space | target (ψ_T − ψ_t) / (T − t) | +0.964 / 0.976 (0.153) | +0.978 | −0.054 / 0.135 | 0.367 |

On TRAIN the same quantities are: fixed episode-mean v 0.927 / 0.998 / 0.277 (same task, same episode, spread), fixed remaining v 0.905 / 0.909 / 0.378, flow remaining v 0.861 / 0.868 / 0.406, corrected full-space remaining v 0.987 / 0.989 / 0.223. On TRAIN, the corrected full-space remaining target has 0.985 / 0.991 / 0.233.

- **Episode-mean v**:
  - Within a task it is almost constant: states of different episodes have mean cosine 0.97, and states of one episode 0.99.
  - Its own target varies more across episodes (0.90); on HELDOUT, v stays near the task's mean direction.
- **Remaining-target v**:
  - Its within-task cosines equal those of its target: 0.906 against 0.900 across episodes and 0.909 against 0.908 within an episode (fixed ψ); 0.860 and 0.864 against 0.855 and 0.868 (flow ψ).
  - The standard deviation of the pairwise cosine is 0.10–0.15, against 0.02 for the episode-mean v.
  - Its spread around the task mean is 0.36 of its norm (target 0.43), against 0.15 for the episode-mean v.
  - On the fixed and flow ψ, v no longer reduces to one direction per task.
- **Remaining-target v on the corrected full-space ψ**:
  - Its target is itself nearly one direction per task: the episode velocities of a task have mean cosine 0.996, and the remaining target 0.964 on HELDOUT and 0.985 on TRAIN.
  - v follows the TRAIN level (0.992 across episodes, relative spread 0.19 against 0.37 for its HELDOUT target).

### Selection consistency

The 129 restored HELDOUT states of Phase 20. Each cell gives:

- the mean over states of cos(z, v) of the chosen / the unchosen candidates;
- chosen − unchosen [95% interval];
- the share of states whose chosen candidate exceeds the unchosen median;
- Spearman(m, cos);
- the states where MATCH would stay.

Random choice gives chosen − unchosen 0, share 0.50–0.51 and Spearman 0. The remaining-target v points along ψ_T − ψ_t at cosine 0.99, so its selection against ψ_T − ψ_t is nearly selection by that velocity itself.

**π set** (pool of 32: π's mean chunk and 31 offsets of scale 0.05–0.2), fixed ψ, true effects:

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.93 / +0.95; −0.01 [−0.02, −0.00]; 0.32; −0.03; 4 | +0.92 / +0.95; −0.03 [−0.05, −0.01]; 0.33; −0.05; 7 | +0.92 / +0.95; −0.02 [−0.04, −0.01]; 0.33; −0.00; 6 | +0.92 / +0.95; −0.03 [−0.05, −0.01]; 0.30; −0.04; 8 | +0.92 / +0.95; −0.03 [−0.05, −0.01]; 0.35; −0.04; 25 |
| ψ_T − ψ_0 | +0.56 / +0.53; +0.04 [+0.03, +0.05]; 0.88; +0.65; 4 | +0.57 / +0.52; +0.05 [+0.04, +0.06]; 0.96; +0.84; 7 | +0.57 / +0.52; +0.05 [+0.04, +0.06]; 0.84; +0.65; 6 | +0.58 / +0.52; +0.05 [+0.04, +0.06]; 0.98; +0.86; 8 | +0.57 / +0.52; +0.05 [+0.04, +0.06]; 0.91; +0.71; 25 |
| ψ_T − ψ_t | +0.62 / +0.57; +0.04 [+0.03, +0.06]; 0.81; +0.60; 4 | +0.63 / +0.57; +0.05 [+0.04, +0.07]; 0.91; +0.73; 7 | +0.63 / +0.57; +0.06 [+0.05, +0.07]; 0.99; +0.98; 6 | +0.63 / +0.57; +0.05 [+0.04, +0.07]; 0.89; +0.75; 8 | +0.63 / +0.57; +0.06 [+0.05, +0.07]; 0.98; +0.89; 25 |

**π set, fixed ψ, predicted effects** (choice with cos(Δ̂, ·), Δ = `goal_dynamics_perturb`):

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.92 / +0.95; −0.03 [−0.07, −0.00]; 0.37; +0.06; 3 | +0.93 / +0.95; −0.02 [−0.04, +0.00]; 0.38; +0.01; 7 | +0.93 / +0.95; −0.01 [−0.04, +0.00]; 0.41; +0.05; 4 | +0.93 / +0.95; −0.02 [−0.04, −0.00]; 0.38; +0.02; 7 | +0.93 / +0.95; −0.02 [−0.04, −0.00]; 0.38; +0.03; 24 |
| ψ_T − ψ_0 | +0.54 / +0.53; +0.01 [−0.01, +0.03]; 0.64; +0.28; 3 | +0.56 / +0.53; +0.03 [+0.02, +0.04]; 0.78; +0.44; 7 | +0.56 / +0.53; +0.03 [+0.02, +0.04]; 0.81; +0.40; 4 | +0.56 / +0.53; +0.03 [+0.02, +0.04]; 0.78; +0.45; 7 | +0.56 / +0.53; +0.03 [+0.02, +0.05]; 0.81; +0.41; 24 |
| ψ_T − ψ_t | +0.59 / +0.57; +0.02 [−0.01, +0.04]; 0.74; +0.37; 3 | +0.61 / +0.57; +0.04 [+0.03, +0.05]; 0.81; +0.45; 7 | +0.61 / +0.57; +0.04 [+0.03, +0.05]; 0.88; +0.59; 4 | +0.61 / +0.57; +0.04 [+0.03, +0.05]; 0.81; +0.47; 7 | +0.61 / +0.57; +0.04 [+0.03, +0.05]; 0.91; +0.55; 24 |

**π set, flow ψ, true effects**:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.94 / +0.94; −0.00 [−0.01, +0.00]; 0.35; +0.05; 5 | +0.92 / +0.94; −0.03 [−0.04, −0.01]; 0.26; −0.05; 7 | +0.93 / +0.94; −0.01 [−0.03, −0.00]; 0.26; −0.02; 7 | +0.91 / +0.94; −0.03 [−0.05, −0.01]; 0.26; −0.04; 25 |
| ψ_T − ψ_0 | +0.53 / +0.50; +0.03 [+0.02, +0.03]; 0.84; +0.53; 5 | +0.54 / +0.50; +0.03 [+0.02, +0.05]; 0.77; +0.54; 7 | +0.55 / +0.50; +0.04 [+0.03, +0.06]; 0.95; +0.77; 7 | +0.54 / +0.50; +0.04 [+0.03, +0.05]; 0.84; +0.63; 25 |
| ψ_T − ψ_t | +0.60 / +0.57; +0.03 [+0.02, +0.04]; 0.85; +0.58; 5 | +0.62 / +0.57; +0.05 [+0.04, +0.06]; 0.98; +0.97; 7 | +0.61 / +0.57; +0.04 [+0.04, +0.05]; 0.89; +0.69; 7 | +0.61 / +0.57; +0.05 [+0.04, +0.06]; 0.95; +0.84; 25 |

**π set, corrected full-space ψ, true effects**:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.87 / +0.90; −0.03 [−0.07, −0.00]; 0.40; +0.11; 4 | +0.87 / +0.90; −0.04 [−0.08, −0.00]; 0.44; +0.23; 7 | +0.87 / +0.90; −0.04 [−0.08, −0.00]; 0.45; +0.20; 7 | +0.87 / +0.90; −0.04 [−0.08, −0.00]; 0.47; +0.21; 7 |
| ψ_T − ψ_0 | +0.85 / +0.79; +0.07 [+0.03, +0.10]; 0.67; +0.40; 4 | +0.87 / +0.79; +0.08 [+0.05, +0.12]; 0.99; +0.95; 7 | +0.87 / +0.79; +0.08 [+0.05, +0.12]; 1.00; +0.98; 7 | +0.87 / +0.79; +0.08 [+0.05, +0.12]; 0.98; +0.95; 7 |
| ψ_T − ψ_t | +0.84 / +0.78; +0.07 [+0.03, +0.11]; 0.64; +0.39; 4 | +0.86 / +0.78; +0.08 [+0.05, +0.12]; 0.98; +0.95; 7 | +0.86 / +0.78; +0.08 [+0.05, +0.12]; 0.98; +0.95; 7 | +0.86 / +0.78; +0.08 [+0.05, +0.12]; 0.97; +0.94; 7 |

**Counterfactual set** (pool of 13: demo and 12 perturbed chunks), fixed ψ, true effects:

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.87 / +0.71; +0.15 [+0.11, +0.20]; 0.44; +0.47; 3 | +0.80 / +0.72; +0.08 [+0.03, +0.13]; 0.38; +0.39; 2 | +0.75 / +0.72; +0.02 [−0.03, +0.07]; 0.36; +0.32; 1 | +0.80 / +0.72; +0.07 [+0.02, +0.12]; 0.43; +0.38; 2 | +0.69 / +0.73; −0.04 [−0.09, +0.01]; 0.30; +0.16; 6 |
| ψ_T − ψ_0 | +0.61 / +0.40; +0.21 [+0.17, +0.24]; 0.88; +0.71; 3 | +0.64 / +0.40; +0.24 [+0.21, +0.27]; 0.97; +0.88; 2 | +0.62 / +0.40; +0.22 [+0.19, +0.25]; 0.84; +0.74; 1 | +0.64 / +0.40; +0.24 [+0.21, +0.27]; 0.96; +0.89; 2 | +0.60 / +0.40; +0.20 [+0.17, +0.24]; 0.84; +0.69; 6 |
| ψ_T − ψ_t | +0.67 / +0.46; +0.21 [+0.18, +0.25]; 0.87; +0.64; 3 | +0.71 / +0.46; +0.25 [+0.23, +0.28]; 0.89; +0.81; 2 | +0.73 / +0.46; +0.27 [+0.25, +0.30]; 1.00; +0.98; 1 | +0.71 / +0.46; +0.26 [+0.23, +0.28]; 0.92; +0.82; 2 | +0.72 / +0.46; +0.26 [+0.24, +0.29]; 0.99; +0.87; 6 |

**Counterfactual set, fixed ψ, predicted effects**:

| Velocity | q_l | v, ep. mean | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.90 / +0.71; +0.18 [+0.13, +0.23]; 0.60; +0.52; 3 | +0.81 / +0.72; +0.09 [+0.03, +0.14]; 0.47; +0.40; 2 | +0.76 / +0.72; +0.03 [−0.04, +0.10]; 0.43; +0.28; 1 | +0.82 / +0.72; +0.10 [+0.05, +0.15]; 0.47; +0.39; 2 | +0.68 / +0.73; −0.05 [−0.12, +0.01]; 0.30; +0.15; 4 |
| ψ_T − ψ_0 | +0.58 / +0.40; +0.18 [+0.15, +0.20]; 0.77; +0.53; 3 | +0.58 / +0.40; +0.18 [+0.15, +0.21]; 0.83; +0.64; 2 | +0.57 / +0.40; +0.16 [+0.12, +0.20]; 0.82; +0.58; 1 | +0.60 / +0.40; +0.20 [+0.17, +0.22]; 0.84; +0.66; 2 | +0.55 / +0.40; +0.14 [+0.10, +0.18]; 0.75; +0.52; 4 |
| ψ_T − ψ_t | +0.65 / +0.46; +0.18 [+0.15, +0.22]; 0.85; +0.51; 3 | +0.66 / +0.46; +0.20 [+0.16, +0.23]; 0.86; +0.65; 2 | +0.67 / +0.46; +0.21 [+0.17, +0.25]; 0.94; +0.76; 1 | +0.67 / +0.46; +0.21 [+0.18, +0.24]; 0.85; +0.67; 2 | +0.66 / +0.46; +0.19 [+0.16, +0.23]; 0.90; +0.69; 4 |

**Counterfactual set, flow ψ, true effects**:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.86 / +0.72; +0.14 [+0.10, +0.19]; 0.43; +0.48; 2 | +0.76 / +0.72; +0.04 [−0.02, +0.09]; 0.42; +0.33; 0 | +0.80 / +0.72; +0.07 [+0.02, +0.12]; 0.43; +0.41; 2 | +0.63 / +0.74; −0.11 [−0.16, −0.06]; 0.28; +0.13; 4 |
| ψ_T − ψ_0 | +0.57 / +0.39; +0.18 [+0.16, +0.21]; 0.83; +0.68; 2 | +0.57 / +0.39; +0.18 [+0.15, +0.20]; 0.80; +0.63; 0 | +0.59 / +0.39; +0.21 [+0.18, +0.23]; 0.92; +0.82; 2 | +0.52 / +0.39; +0.13 [+0.09, +0.16]; 0.67; +0.57; 4 |
| ψ_T − ψ_t | +0.67 / +0.46; +0.20 [+0.17, +0.24]; 0.84; +0.63; 2 | +0.73 / +0.46; +0.27 [+0.24, +0.29]; 1.00; +0.98; 0 | +0.70 / +0.46; +0.24 [+0.21, +0.26]; 0.88; +0.74; 2 | +0.69 / +0.46; +0.23 [+0.20, +0.26]; 0.86; +0.79; 4 |

**Counterfactual set, corrected full-space ψ, true effects**:

| Velocity | q_l | v, remaining | mean, ep. | mean, rem. |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | +0.82 / +0.62; +0.19 [+0.13, +0.26]; 0.68; +0.63; 0 | +0.81 / +0.63; +0.19 [+0.12, +0.26]; 0.69; +0.61; 0 | +0.81 / +0.63; +0.19 [+0.12, +0.26]; 0.71; +0.62; 0 | +0.81 / +0.62; +0.19 [+0.12, +0.26]; 0.68; +0.61; 0 |
| ψ_T − ψ_0 | +0.91 / +0.53; +0.38 [+0.33, +0.43]; 0.86; +0.76; 0 | +0.93 / +0.53; +0.40 [+0.36, +0.45]; 0.99; +0.98; 0 | +0.93 / +0.53; +0.40 [+0.36, +0.45]; 1.00; +0.98; 0 | +0.93 / +0.53; +0.40 [+0.36, +0.45]; 0.99; +0.97; 0 |
| ψ_T − ψ_t | +0.90 / +0.52; +0.38 [+0.34, +0.44]; 0.84; +0.72; 0 | +0.93 / +0.52; +0.41 [+0.36, +0.46]; 0.98; +0.96; 0 | +0.93 / +0.52; +0.41 [+0.36, +0.47]; 0.98; +0.96; 0 | +0.93 / +0.52; +0.41 [+0.36, +0.46]; 0.98; +0.95; 0 |

Paired differences of the remaining-target v, chosen − unchosen / Spearman(m, cos) / own − other of chosen − unchosen [95% interval]:

| Set | Velocity | against q_l | against v, ep. mean | against mean, rem. |
|---|---|---|---|---|
| π, fixed, true | ψ_{t+10} − ψ_t | −0.01 [−0.03, +0.00] / +0.02 [−0.07, +0.11] / −0.03 [−0.06, −0.00] | +0.00 [−0.00, +0.01] / +0.05 [−0.02, +0.11] / −0.00 [−0.01, +0.01] | +0.00 [−0.00, +0.01] / +0.03 [−0.01, +0.08] / −0.01 [−0.03, +0.02] |
| π, fixed, true | ψ_T − ψ_0 | +0.01 [+0.00, +0.01] / +0.00 [−0.07, +0.07] / −0.03 [−0.06, −0.01] | −0.00 [−0.01, −0.00] / −0.19 [−0.26, −0.12] / −0.02 [−0.03, −0.01] | −0.00 [−0.00, +0.00] / −0.05 [−0.10, −0.01] / −0.03 [−0.05, −0.01] |
| π, fixed, true | ψ_T − ψ_t | +0.01 [+0.01, +0.02] / +0.38 [+0.31, +0.45] / −0.03 [−0.06, −0.01] | +0.00 [+0.00, +0.01] / +0.25 [+0.19, +0.31] / −0.02 [−0.04, −0.01] | +0.00 [+0.00, +0.00] / +0.09 [+0.05, +0.13] / −0.03 [−0.05, −0.02] |
| π, fixed, predicted | ψ_{t+10} − ψ_t | +0.02 [−0.00, +0.05] / −0.01 [−0.10, +0.08] / +0.01 [−0.00, +0.04] | +0.00 [+0.00, +0.01] / +0.04 [−0.02, +0.11] / +0.00 [−0.01, +0.01] | +0.01 [+0.00, +0.01] / +0.02 [−0.02, +0.06] / +0.01 [+0.00, +0.03] |
| π, fixed, predicted | ψ_T − ψ_0 | +0.02 [+0.00, +0.05] / +0.12 [+0.05, +0.20] / −0.00 [−0.02, +0.01] | +0.00 [−0.00, +0.01] / −0.04 [−0.10, +0.02] / −0.01 [−0.02, −0.00] | −0.00 [−0.01, +0.00] / −0.01 [−0.04, +0.02] / −0.01 [−0.02, −0.00] |
| π, fixed, predicted | ψ_T − ψ_t | +0.02 [+0.00, +0.05] / +0.21 [+0.14, +0.30] / −0.01 [−0.03, +0.01] | +0.00 [−0.00, +0.01] / +0.14 [+0.08, +0.20] / −0.02 [−0.03, −0.01] | +0.00 [−0.00, +0.00] / +0.04 [+0.01, +0.07] / −0.02 [−0.03, −0.01] |
| π, flow, true | ψ_{t+10} − ψ_t | −0.02 [−0.04, −0.01] / −0.10 [−0.19, −0.03] / −0.04 [−0.07, −0.02] | – | +0.00 [−0.01, +0.01] / −0.01 [−0.07, +0.04] / −0.01 [−0.03, +0.01] |
| π, flow, true | ψ_T − ψ_0 | +0.01 [+0.00, +0.02] / +0.02 [−0.07, +0.10] / −0.02 [−0.04, −0.00] | – | −0.00 [−0.01, +0.00] / −0.09 [−0.15, −0.03] / −0.03 [−0.05, −0.01] |
| π, flow, true | ψ_T − ψ_t | +0.02 [+0.01, +0.02] / +0.39 [+0.32, +0.46] / −0.03 [−0.05, −0.01] | – | +0.00 [+0.00, +0.01] / +0.12 [+0.09, +0.17] / −0.03 [−0.05, −0.02] |
| π, corr. full, true | ψ_{t+10} − ψ_t | −0.01 [−0.02, +0.01] / +0.12 [+0.01, +0.21] / −0.02 [−0.06, +0.00] | – | +0.00 [−0.00, +0.00] / +0.02 [+0.01, +0.04] / −0.02 [−0.04, +0.00] |
| π, corr. full, true | ψ_T − ψ_0 | +0.01 [+0.01, +0.01] / +0.55 [+0.45, +0.66] / −0.01 [−0.03, +0.02] | – | +0.00 [+0.00, +0.00] / +0.00 [−0.02, +0.02] / −0.01 [−0.04, +0.01] |
| π, corr. full, true | ψ_T − ψ_t | +0.01 [+0.01, +0.02] / +0.56 [+0.46, +0.67] / −0.00 [−0.03, +0.02] | – | +0.00 [+0.00, +0.00] / +0.02 [−0.00, +0.03] / −0.01 [−0.04, +0.01] |
| counterfactual, fixed, true | ψ_{t+10} − ψ_t | −0.13 [−0.19, −0.08] / −0.15 [−0.21, −0.08] / −0.61 [−0.73, −0.51] | −0.06 [−0.10, −0.03] / −0.07 [−0.12, −0.02] / −0.24 [−0.31, −0.17] | +0.06 [+0.03, +0.10] / +0.15 [+0.11, +0.20] / −0.22 [−0.28, −0.16] |
| counterfactual, fixed, true | ψ_T − ψ_0 | +0.01 [−0.01, +0.03] / +0.04 [−0.02, +0.10] / −0.45 [−0.54, −0.36] | −0.02 [−0.04, −0.01] / −0.14 [−0.19, −0.10] / −0.25 [−0.31, −0.19] | +0.01 [−0.00, +0.03] / +0.06 [+0.03, +0.09] / −0.28 [−0.35, −0.22] |
| counterfactual, fixed, true | ψ_T − ψ_t | +0.06 [+0.04, +0.08] / +0.35 [+0.29, +0.40] / −0.45 [−0.54, −0.36] | +0.02 [+0.01, +0.03] / +0.18 [+0.15, +0.21] / −0.29 [−0.35, −0.23] | +0.01 [+0.00, +0.02] / +0.12 [+0.09, +0.14] / −0.33 [−0.39, −0.26] |
| counterfactual, fixed, predicted | ψ_{t+10} − ψ_t | −0.15 [−0.23, −0.08] / −0.24 [−0.30, −0.18] / −0.50 [−0.60, −0.41] | −0.06 [−0.13, +0.01] / −0.12 [−0.17, −0.08] / −0.21 [−0.30, −0.12] | +0.08 [+0.04, +0.13] / +0.12 [+0.09, +0.16] / −0.17 [−0.25, −0.10] |
| counterfactual, fixed, predicted | ψ_T − ψ_0 | −0.01 [−0.06, +0.02] / +0.06 [+0.00, +0.12] / −0.33 [−0.41, −0.26] | −0.02 [−0.06, +0.02] / −0.06 [−0.10, −0.02] / −0.20 [−0.27, −0.13] | +0.02 [+0.00, +0.04] / +0.06 [+0.03, +0.10] / −0.21 [−0.28, −0.15] |
| counterfactual, fixed, predicted | ψ_T − ψ_t | +0.03 [−0.01, +0.06] / +0.24 [+0.19, +0.29] / −0.32 [−0.40, −0.24] | +0.02 [−0.03, +0.05] / +0.10 [+0.07, +0.14] / −0.23 [−0.30, −0.16] | +0.02 [−0.00, +0.04] / +0.07 [+0.04, +0.09] / −0.25 [−0.32, −0.18] |
| counterfactual, flow, true | ψ_{t+10} − ψ_t | −0.11 [−0.16, −0.06] / −0.15 [−0.21, −0.10] / −0.52 [−0.62, −0.43] | – | +0.15 [+0.10, +0.19] / +0.20 [+0.15, +0.25] / −0.16 [−0.23, −0.09] |
| counterfactual, flow, true | ψ_T − ψ_0 | −0.01 [−0.02, +0.01] / −0.05 [−0.11, +0.01] / −0.36 [−0.42, −0.29] | – | +0.05 [+0.03, +0.08] / +0.06 [+0.01, +0.11] / −0.22 [−0.26, −0.17] |
| counterfactual, flow, true | ψ_T − ψ_t | +0.06 [+0.05, +0.08] / +0.35 [+0.29, +0.41] / −0.39 [−0.46, −0.33] | – | +0.04 [+0.03, +0.05] / +0.18 [+0.14, +0.23] / −0.33 [−0.38, −0.28] |
| counterfactual, corr. full, true | ψ_{t+10} − ψ_t | −0.01 [−0.03, +0.02] / −0.03 [−0.09, +0.04] / −0.31 [−0.39, −0.23] | – | −0.00 [−0.00, +0.00] / +0.00 [−0.00, +0.01] / −0.12 [−0.19, −0.05] |
| counterfactual, corr. full, true | ψ_T − ψ_0 | +0.02 [+0.01, +0.03] / +0.22 [+0.17, +0.27] / −0.27 [−0.35, −0.19] | – | −0.00 [−0.00, +0.00] / +0.00 [−0.00, +0.01] / −0.12 [−0.20, −0.05] |
| counterfactual, corr. full, true | ψ_T − ψ_t | +0.03 [+0.02, +0.04] / +0.24 [+0.19, +0.29] / −0.26 [−0.34, −0.18] | – | +0.00 [−0.00, +0.00] / +0.01 [+0.00, +0.01] / −0.12 [−0.20, −0.05] |

- **Pool statistics**, the same for every entry on one ψ:
  - The within-state std of cos(z, ψ_{t+10} − ψ_t) is 0.34 in the counterfactual set and 0.02–0.03 in the π set; the best candidate reaches 0.99 and 0.97. On the corrected full-space ψ the std is 0.44 and 0.05, the best 0.99 and 0.95.
  - Against ψ_T − ψ_t the std is 0.23–0.25 and 0.03, the best 0.73 and 0.62–0.63.
- **Own − other of chosen − unchosen**:
  - counterfactual set, fixed ψ, true effects, against ψ_{t+10} − ψ_t: +0.70 (q_l), +0.32 (episode-mean v), +0.08 (remaining v);
  - counterfactual set, corrected full-space ψ, against ψ_{t+10} − ψ_t: +0.83 (q_l), +0.52 (remaining v);
  - π set: −0.01 to +0.11 for every entry on every ψ.
- **By restored-state position f** (0.1 / 0.4 / 0.7), chosen − unchosen against ψ_{t+10} − ψ_t:
  - π set, fixed ψ, true effects: remaining v −0.05 / −0.03 / +0.01, q_l −0.00 / −0.03 / −0.00;
  - counterfactual set: remaining v −0.13 / +0.04 / +0.15, q_l +0.27 / +0.07 / +0.12.

### Multimodality check

`scripts/check_velocity_multimodality.py` takes U, the regression target, from the velocity query's config: (ψ_T − ψ_0) / T for the episode mean, (ψ_T − ψ_t) / (T − t) for the remaining target.

- **Runs**, at step 3000:
  - `outputs/analysis/velocity_query/multimodality_fixed` (episode mean, `6af22fb`);
  - `outputs/analysis/velocity_query_remaining/multimodality_{fixed,flow}` (remaining target, `eae446b`, node-61 CUDA 5 and 6, 25 s each);
  - `outputs/analysis/velocity_query_remaining/multimodality_flow_full_sgd` (remaining target, corrected full-space ψ, `35b870c`, node-61 CUDA 2, 28 s).
- **Frames**: every demonstration frame with t + 10 ≤ T: HELDOUT 4,864, TRAIN 43,701.
- **Two vectors per frame**: U, the regression target; D = ψ_{t+10} − ψ_t, the local displacement.
- **Neighbours**: in each space, the nearest frame of every other TRAIN episode of the same task, then the 10 episodes whose nearest frame is closest. A TRAIN query's own episode is excluded.
  - Spaces: ψ, and the frozen DINOv2 retrieval key (`outputs/state_tables/libero_goal_frozen_keys.npy`).
  - Mean distance of the 10th neighbour, HELDOUT: 1.89 in the fixed ψ, 1.29 in the corrected full-space ψ, 0.45 in the key space.
- **Agreement**: the mean pairwise cosine among the 10 neighbours' vectors. It is 1 when all nearby demonstrations move the same way. A neighbour pair with cosine < 0 marks opposing directions at nearby states.
- **Signs of averaging**, if they were there: an error of v that grows as the agreement falls; v nearer to the neighbours' mean than to its own vector; ‖v‖ shrinking where the agreement is low.

HELDOUT, ψ neighbours. With frozen-key neighbours the median agreement is 0.995 (fixed), 0.991 (flow) and 0.997 (corrected full space) for U, and 0.928, 0.927 and 0.953 for D (`report.md`).

| Quantity | fixed: v, ep. mean | fixed: v, remaining | flow: v, remaining | corr. full: v, remaining |
|---|---|---|---|---|
| U: agreement, median / p10 / p90 | 0.929 / 0.907 / 0.959 | 0.997 / 0.984 / 0.999 | 0.995 / 0.974 / 0.998 | 0.998 / 0.989 / 0.999 |
| U: frames with an opposing neighbour pair | 0 | 0 | 0 | 0 |
| U: cos(v, own) / cos(v, neighbours' mean) / cos(own, neighbours' mean) | 0.939 / 0.983 / 0.946 | 0.988 / 0.996 / 0.988 | 0.984 / 0.992 / 0.982 | 0.981 / 0.998 / 0.984 |
| U: cos(q_l, own) | 0.719 | 0.577 | 0.502 | 0.737 |
| U: Spearman(agreement, cos(v, own)) | −0.05 | +0.54 | +0.57 | +0.54 |
| U: relative error, median; ‖v‖ / mean ‖U_j‖ | 0.386; 0.99 | 0.154; 1.00 | 0.163; 0.99 | 0.163; 0.99 |
| U, lowest agreement quintile: agreement; cos(v, own); rel. error | 0.86–0.91; 0.943; 0.358 | 0.91–0.99; 0.954; 0.207 | 0.87–0.99; 0.941; 0.237 | 0.87–0.99; 0.970; 0.170 |
| D: agreement, median / p10 / p90 | 0.961 / 0.783 / 0.993 | 0.961 / 0.783 / 0.993 | 0.964 / 0.801 / 0.990 | 0.962 / 0.755 / 0.994 |
| D: frames with an opposing neighbour pair (frozen keys) | 127, 2.6% (4.1%) | 127, 2.6% (4.1%) | 77, 1.6% (3.7%) | 57, 1.2% (2.0%) |
| D: cos(v, own) / cos(v, neighbours' mean) / cos(own, neighbours' mean) | 0.563 / 0.647 / 0.885 | 0.634 / 0.724 / 0.885 | 0.622 / 0.708 / 0.890 | 0.795 / 0.964 / 0.826 |
| D: cos(q_l, own) | 0.401 | 0.401 | 0.344 | 0.632 |
| D: Spearman(agreement, cos(v, own)) | +0.65 | +0.78 | +0.69 | +0.71 |
| D, lowest agreement quintile: agreement; cos(v, own); cos(v, neighbours' mean); cos(q_l, own) | 0.29–0.88; 0.211; 0.466; 0.276 | 0.29–0.88; 0.235; 0.479; 0.276 | 0.47–0.89; 0.247; 0.448; 0.213 | 0.52–0.89; 0.406; 0.910; 0.389 |

D by quintile of its agreement, cos(v, own D), lowest to highest:

- fixed, episode-mean v: 0.211 / 0.447 / 0.655 / 0.741 / 0.760;
- fixed, remaining v: 0.235 / 0.423 / 0.695 / 0.869 / 0.947;
- flow, remaining v: 0.247 / 0.451 / 0.703 / 0.822 / 0.889;
- corrected full space, remaining v: 0.406 / 0.835 / 0.881 / 0.929 / 0.924.

- **The regression targets do not split.**
  - Nearby demonstrations have remaining velocities with agreement 0.997 (fixed) and 0.995 (flow), and no opposing pair; for the episode mean the agreement is 0.93.
  - The remaining-target v is at 0.988 to its own target and 0.954 in the lowest-agreement quintile. Its norm ratio is 0.97–1.02 in every quintile.
  - There is no sign that the single-direction regression averages distinct targets.
  - Its error is larger where the agreement is lower (Spearman +0.54, relative error 0.207 against 0.125–0.163). This follows the target: the frame's own target is less predictable from its neighbours there, cos(own, neighbours' mean) 0.952.
- **The local motion splits in some states**, as before.
  - The agreement of D is below 0.88 in a fifth of the HELDOUT frames, and 2.6% of the frames have opposing neighbours. These frames are concentrated early in the episode and in the three bowl tasks (`outputs/analysis/velocity_query/multimodality_fixed/report.md`).
  - There, the remaining-target v has cosine 0.235 with the frame's own displacement and 0.479 with the neighbours' mean, like the episode-mean v (0.211, 0.466).
  - In the highest-agreement quintiles it follows the local displacement more closely: 0.869 and 0.947, against 0.741 and 0.760.
  - On the corrected full-space ψ, v is at 0.406 to the frame's own displacement in the lowest-agreement quintile and at 0.910 to the neighbours' mean.

## Reading

The quantities above are implementation checks: did v learn its target, and does it vary with the state?

In short, on the fixed and flow ψ, v with the remaining target learned its target and varies with the state as much as the target does. On the local-displacement cosine it is clearly above q_l and the episode-mean v, in the second half of the episodes. On selection among π's small-offset candidates it is still at chance. Its direction depends less on the instruction than q_l's or the episode-mean v's. On the corrected full-space ψ the remaining target is nearly one direction per task, and v equals the text-only task mean on every check.

- **v learned the remaining target.**
  - HELDOUT relative error 0.164 (fixed ψ) and 0.176 (flow ψ), against 0.369 and 0.390 for the text-only task mean; the MSE is 0.49 and 0.41 times the task mean's.
  - The episode-mean v did not beat its task mean (0.387 against 0.340).
  - On the corrected full-space ψ: 0.172 against 0.224, with MSE 1.28e-4 against 1.33e-4.
  - Against ψ_T − ψ_t it has mean cosine 0.99 (fixed) and 0.98 (flow), p10 0.99 and 0.98, and a mean of at least 0.95 in every quarter.
- **On the fixed and flow ψ, v varies with the state and no longer reduces to a constant per task.**
  - Within a task, v at states of different episodes has mean cosine 0.906 (fixed) and 0.860 (flow), against 0.972 for the episode-mean v.
  - Along one episode it is 0.909 and 0.864, against 0.993.
  - These equal the values of its own target (0.900 / 0.908 and 0.855 / 0.868). Its relative spread around the task mean is 0.36–0.40, against 0.15.
- **Local-displacement cosine on the fixed and flow ψ: clearly higher than q_l and than the episode-mean v, in the second half of the episodes.**
  - HELDOUT mean 0.63 [0.60, 0.67] on the fixed ψ, against 0.40 for q_l (+0.23 [+0.21, +0.25]), 0.56 for the episode-mean v (+0.07 [+0.06, +0.08]) and 0.53–0.57 for the task means.
  - On the flow ψ: 0.62 against 0.34 for q_l (+0.28 [+0.26, +0.30]).
  - By t/T quarter on the fixed ψ, the gain over the episode-mean v is −0.03 [−0.05, −0.01], +0.02 [−0.01, +0.04], +0.14 and +0.21. In the last two quarters the median cosine is 0.94 and 0.98.
  - In the first quarter, v is at the level of q_l (0.30 against 0.32). The heavy low tail remains: p10 0.04 against 0.20 for q_l.
- **On the fixed and flow ψ, v depends less on the instruction than q_l and the episode-mean v.**
  - own − other against the local displacement is +0.25 (fixed) and +0.18 (flow), against +0.44 and +0.38 for q_l and +0.56 for the episode-mean v.
  - Queried with another task's instruction, v still has cosine +0.38 and +0.45 with the frame's own displacement. In the third quarter this is +0.56 and +0.63.
  - From the second quarter on, this other-task cosine exceeds own − other (fixed ψ: +0.33 against +0.25, then +0.56 against +0.29): the frame sets v's direction more than the instruction does.
  - On the counterfactual set its choices are the least task-specific: own − other of chosen − unchosen is +0.08, against +0.32 and +0.70.
- **Selection consistency on the π small-offset set is unchanged: at chance against the local displacement, on every ψ.**
  - Fixed ψ: chosen − unchosen −0.02 [−0.04, −0.01] with true effects and −0.01 [−0.04, +0.00] with Δ̂. q_l has −0.01 and −0.03, the episode-mean v −0.03 and −0.02. The paired differences with both are within ±0.02.
  - Flow ψ: −0.03 [−0.04, −0.01], −0.02 [−0.04, −0.01] below q_l.
  - Against ψ_T − ψ_t the choice is slightly better than q_l's: +0.06 against +0.04, share 0.99, Spearman +0.98 with true effects; +0.04, share 0.88, Spearman +0.59 with Δ̂. This velocity is v's own direction, and the π candidates' cosines spread by 0.03 within a state.
  - On the counterfactual set, against the local displacement, v chooses worse than q_l (−0.13 [−0.19, −0.08]) and the episode-mean v (−0.06 [−0.10, −0.03]). The loss is largest at f = 0.1, where v's choice is −0.13 against q_l's +0.27.
- **Multimodality**: the remaining target does not split among nearby demonstrations (agreement 0.995–0.998, no opposing pair), and v follows it. The local motion still splits early in the episodes and in the bowl tasks, and there v, like the episode-mean v, points near the neighbours' mean motion.
- **Corrected full-space ψ**: v coincides with the text-only task mean of its target.
  - Every paired difference to it is within ±0.02, except own − other (−0.06 in velocity-field consistency, −0.12 on the counterfactual set).
  - Within a task, v at different states has mean cosine 0.992. Its target has 0.964 on HELDOUT and 0.985 on TRAIN, and the episode velocities of a task have 0.996: in this ψ the remaining target is itself close to one direction per task.
  - Local-displacement cosine 0.80 [0.74, 0.85], against 0.63 for q_l (+0.16 [+0.14, +0.18]); by quarter the gain is +0.11 to +0.23.
  - v is instruction-specific here: own − other +0.78, and with another task's instruction the cosine with the frame's displacement is +0.01.
  - π set against the local displacement: −0.04 [−0.08, −0.00], at or below chance like q_l (−0.03 [−0.07, −0.00]).

## Full-space ψ

No velocity query was trained on the full-space stage one with SIGReg over all frames and D_ψ with gradient (`outputs/runs/goal_instruction_contrast_flow_full_sigall`, step 5000), because that ψ has collapsed. With D_ψ without gradient (`goal_instruction_contrast_flow_full_sigall_sgd`) it does not collapse, and the remaining-target v above is trained on that ψ. The collapse checks (5,294 HELDOUT frames; `collapse_check.json` in the runs, `docs/stage_one_flow.md`) give:

| Stage one | ψ effective rank | top eigenvector share | median ‖ψ_T − ψ_0‖ |
|---|---|---|---|
| fixed | 16.6 | 0.115 | 35 |
| full space, SIGReg on uniform frames | 1.14 | 0.981 | 1156 |
| full space, SIGReg on all frames, step 1000 / 5000 | 1.16 / 1.02 | 0.978 / 0.997 | 384 / 1338 |
| full space, SIGReg on all frames, D_ψ without gradient, step 1000 / 5000 | 9.83 / 8.56 | 0.150 / 0.198 | 24.8 / 27.5 |

The last row was measured again before training the velocity query (`outputs/analysis/velocity_query_remaining/checks/check_collapse_flow_full_sgd.txt`, steps 1000–5000): effective rank 9.83, 9.27, 8.77, 8.58, 8.56.

## Outputs

| Output | Path |
|---|---|
| training, episode mean, fixed ψ | `outputs/runs/goal_velocity_fixed` (`config.json`, `metrics.jsonl`, `heldout.jsonl`, `target_statistics.json`, `checkpoints/step_00{1,2,3}000.pt`) |
| training, remaining target, fixed ψ | `outputs/runs/goal_velocity_remaining_fixed` (same files) |
| training, remaining target, flow ψ | `outputs/runs/goal_velocity_remaining_flow` (same files) |
| training, remaining target, corrected full-space ψ | `outputs/runs/goal_velocity_remaining_flow_full_sgd` (same files) |
| consistency scoring, episode mean (fixed ψ, with the step-1000 checkpoint) | `outputs/analysis/velocity_query/consistency_fixed` (`report.md`, `tables.csv`, `comparisons.csv`, `states.csv`, `config.json`) |
| consistency scoring, both targets, fixed ψ | `outputs/analysis/velocity_query_remaining/consistency_fixed_08f40bb` |
| consistency scoring, remaining target, flow ψ | `outputs/analysis/velocity_query_remaining/consistency_flow_08f40bb` |
| consistency scoring, remaining target, corrected full-space ψ | `outputs/analysis/velocity_query_remaining/consistency_flow_full_sgd_35b870c` |
| multimodality, episode mean | `outputs/analysis/velocity_query/multimodality_fixed` (`report.md`, `tables.csv`, `frames.csv`, `config.json`) |
| multimodality, remaining target | `outputs/analysis/velocity_query_remaining/multimodality_{fixed,flow,flow_full_sgd}` |
| bit identity of the episode mean | `outputs/analysis/velocity_query_remaining/bit_identity/{base,nokey,key}` |
| regression checks of the scorer and the multimodality check | `outputs/analysis/velocity_query/check_scorer_fixed_short_6af22fb`, `outputs/analysis/velocity_query_remaining/{check_scorer_phase22_eae446b,check_multimodality_fixed_eae446b,consistency_fixed,consistency_flow}` |
| check scripts and outputs | `outputs/analysis/velocity_query_remaining/checks/` |
| launchers and logs | `outputs/analysis/velocity_query/logs/`, `outputs/analysis/velocity_query_remaining/logs/` |

## Closed loop with v(o, l)

The closed loop scores candidate effects against v(o, l) with `--match-target velocity`. v is computed at every decision from the current observation, and stay, the escalation and the stuck rule are those of the q_l match. This section covers the mechanism, its checks, and 10 tasks × 50 benchmark inits with v as the MATCH target and as the gate between π's mean chunk and stay, on the fixed ψ and on the projected-flow ψ. The baseline is π's mean chunk (`docs/closed_loop_mvp.md`, "Current default"). Scripts, launchers, process logs and analysis outputs are in `outputs/closed_loop/p23_runlogs/`, written R below.

### Mechanism

- **`MatchTarget.VELOCITY`** (`ceam/policy/candidate_selector.py`): `CandidateSelector.select(effects, state, velocity)` scores cos(z, v) with v [state_dim] passed at each call. v is required under VELOCITY and refused under every other target. VELOCITY needs no ĝ or U. Stay scores 0; MATCH, RANDOM and MEAN choose as before.
- **`VelocityMatch(query, instruction)`** (`ceam/policy/closed_loop_runner.py`): the velocity query in eval mode and the task's original instruction (variant 0, the instruction of q_l) as cached SigLIP tokens [1, 64, 1152] with mask [1, 64].
  - `ClosedLoopRunner(…, velocity)` takes it as its last argument (default None). It is required with match target VELOCITY and only then, and it is refused with effect source NONE.
  - Per decision, after encoding: v = query(frames, ψ(o), instruction), [256] float32, without autocast under inference mode, synchronized. Every escalation level of the decision, and the observed-effect comparison under retrieval, scores against this v.
  - π still takes q_l.
- **`scripts/run_closed_loop.py`**:
  - `--velocity-checkpoint PATH` (`VelocityCheckpointStore`) is required with `--match-target velocity` and refused otherwise; `--effect-source none` is refused with it.
  - The checkpoint's stage one must be `--checkpoint`, compared as resolved paths.
  - The decision line adds `‖v‖` before the scores and `v <ms>` between encode and π; the episode latency line adds the mean v time. The lines of the other targets are unchanged.
  - A recorded shard's provenance holds `match_target`, `velocity_checkpoint` and `velocity_step`.
- **Loading the remaining-target v**: its config has the `target` field, which the configuration code of this branch does not read. The runs with that v use a code snapshot of this branch's `bd492d6` plus the `ceam/` changes of `s3-phase-25` up to `eae446b` (the velocity-target option; R `s3-phase-25_ceam_cdb41c8_eae446b.patch`). The two change sets touch disjoint files.

### Checks

CPU unit checks (R `check_units.py`, output `check_units.txt`):

- **Existing targets**: QUERY, GOAL and GOAL_SUBSPACE under MATCH, RANDOM and MEAN choose and score bit-identically to the pre-change selector (`cdb41c8`) in 80 of 80 calls each. The calls have K from 1 to 512, float32 and float16 effects, and both signs, so they include stays.
- **VELOCITY**: the scores equal cos(z, v) exactly (max difference 0), and the choice is the argmax or stay, in 20 of 20 calls per strategy.
- **Refusals**, each with its own message: VELOCITY without v, or with v of shape [1, 256] or [257]; v under QUERY or GOAL; a runner with VELOCITY and no `VelocityMatch`, with a `VelocityMatch` and another target, with effect source NONE, or with a state_dim of 128. The runner's v equals the query called directly (max difference 0).
- **Script**: it refuses `--match-target velocity` without `--velocity-checkpoint`, `--velocity-checkpoint` with query or goal, and velocity with `--effect-source none`, before loading anything.

Closed-loop checks, task 3 (put the bowl on the plate), benchmark inits 0–4, seed 0, 300 steps, traced, `--check-start-observation`, node-65 (L40S). The check runs use the diagnostic entry of phase 13 with a per-call checksum of v added (R `diag_run.py`). Outputs are in `outputs/closed_loop/p23_checks/`; comparison R `compare_checks.py`, output `compare_checks.txt`.

| Check | Result |
|---|---|
| MATCH with q_l among π's candidates with scales 0.05/0.1/0.2, rule 32/128/512, Δ +A; this code and the pre-change code on the same GPU (`small_query_t3_{new,base}`) | 5 of 5 episodes identical in results, traces and every per-call checksum |
| The same against `p13_checks/proposal_small_t3_{a,b}` (code `b4487f6`) | 5 of 5 identical in results and traces, against each |
| The same against `v5_proposal_small_dynA` task 3 | All 5 differ from decision 1: `chosen_proposal`, `best_scores` and `chosen_score`. The pre-change `p13_checks` run differs from it in the same way. That run drew its noise from a per-process stream counter; since phase 13 the streams are keyed by decision (`docs/proposal_branch.md`). No current code reproduces it. |
| π's mean chunk, no Δ, against `v6_mean_default` task 3 (`mean_t3_new`) | Results identical in 5 of 5. Traces identical in 4; in init 2 only the final scene, after the last decision, differs. |
| π's mean chunk, this code against the pre-change code on the same GPU (`mean_t3_{new,base}`) | Results identical in 5 of 5. Init 2 splits: the first differing checksum is the agent-view image of the observation at decision 7, while the wrist image, proprio, qpos and qvel are bit-identical; the traces differ from the scene at decision 8. The pre-change run also differs from v6 from decision 8 of init 2. |
| `--checkpoint` the flow stage one with the fixed-ψ v | Refused: the velocity checkpoint was trained on the stage-one checkpoint `goal_instruction_contrast/…/step_005000.pt`, not on `--checkpoint` |
| Smoke run: MATCH with the episode-mean v (`small_vep_t3`) | 106 decisions, ‖v‖ 0.36–0.46, v 2.4 ms per decision with 3 processes on the GPU; 2 of 5 successes; init 3 escalates 19 times and stays 22 times |

### Runs

The v6 protocol (seed 0, 300 env steps, 10 settle steps, benchmark inits 0–49, traced, `--check-start-observation`), one process per task. Launchers R `launch_run.sh` and `run_one.sh`, process logs R `logs/<run>/`.

- **MATCH**: π's candidates with scales 0.05/0.1/0.2, dct 1, seed 0; rule 32/128/512 with patience 2, stuck → stay; Δ +A. This is the `v5_proposal_small_dynA` setting with another target.
- **Gate**: π's mean chunk against stay, K = 1, patience 2, stuck → stay, Δ +A. This is the `v5_proposal_gate_dynA` setting. K = 1 draws no noise, so that run's decisions do not depend on the noise-stream keying.

| Run (`outputs/closed_loop/`) | ψ | Configuration | v | Code | Host, CUDA |
|---|---|---|---|---|---|
| `v7_proposal_small_vrem_dynA` | fixed | MATCH | remaining (`goal_velocity_remaining_fixed`, step 3000) | `bd492d6` + `s3-phase-25` | node-65, 0–3, 5–7 |
| `v7_proposal_small_vep_dynA` | fixed | MATCH | episode mean (`goal_velocity_fixed`, step 3000) | `bd492d6` | node-65, 0, 2, 3, 5–7 |
| `v7_proposal_gate_vrem_dynA` | fixed | gate | remaining | `bd492d6` + `s3-phase-25` | node-61, 4–7 |
| `v7_proposal_small_query_dynA` | fixed | MATCH with q_l (reference) | – | `bd492d6` | node-61, 0–3 |
| `v7_proposal_small_vrem_flow` | flow | MATCH | remaining (`goal_velocity_remaining_flow`, step 3000) | `bd492d6` + `s3-phase-25` | node-62, 2–6 |
| `v7_proposal_gate_vrem_flow` | flow | gate | remaining (flow) | `bd492d6` + `s3-phase-25` | node-61, 4–7 |

- **Flow ψ**: the stage one is `goal_instruction_contrast_flow` step 5000, with its ψ table and its own heads, Δ `goal_dynamics_perturb_flow` and π `goal_proposal_flow` (step 3000, `docs/stage_one_flow.md`). The baseline is its π's mean chunk, `v7_mean_flow` (0.904).
- **References**: `v6_mean_default` is π's mean chunk (0.828). `v5_proposal_small_dynA` is MATCH with q_l under the per-process noise streams; `v5_proposal_gate_dynA` is the gate with q_l.
- **Analysis**: R `analyze_runs.py` (outputs `analysis_fixed.txt`, `analysis_flow.txt`), `classify_closed_loop_failures.py` (`failures_<run>.txt`) and R `first_decisions.py`.

### Fixed ψ

Success:

| Task | Text | π mean (v6) | MATCH, q_l | MATCH, q_l (v5) | MATCH, v episode mean | MATCH, v remaining | gate, q_l (v5) | gate, v remaining |
|---|---|---|---|---|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 0.96 | 0.56 | 0.58 | 0.04 | 0.30 | 0.96 | 0.96 |
| 1 | open the top drawer and put the bowl inside | 0.78 | 0.40 | 0.30 | 0.08 | 0.08 | 0.70 | 0.20 |
| 2 | push the plate to the front of the stove | 0.94 | 0.72 | 0.58 | 0.32 | 0.26 | 0.88 | 0.34 |
| 3 | put the bowl on the plate | 0.74 | 0.64 | 0.62 | 0.64 | 0.84 | 0.52 | 0.66 |
| 4 | put the bowl on the stove | 0.60 | 0.44 | 0.46 | 0.52 | 0.58 | 0.32 | 0.42 |
| 5 | put the bowl on top of the cabinet | 0.86 | 0.74 | 0.72 | 0.76 | 0.62 | 0.66 | 0.68 |
| 6 | put the cream cheese in the bowl | 0.62 | 0.58 | 0.60 | 0.32 | 0.20 | 0.62 | 0.40 |
| 7 | put the wine bottle on the rack | 0.80 | 0.54 | 0.48 | 0.04 | 0.14 | 0.74 | 0.62 |
| 8 | put the wine bottle on top of the cabinet | 0.98 | 0.64 | 0.70 | 0.50 | 0.62 | 0.98 | 0.98 |
| 9 | turn on the stove | 1.00 | 0.80 | 0.80 | 0.84 | 0.88 | 1.00 | 1.00 |
| **all** | | **0.828** | **0.606** | **0.584** | **0.406** | **0.452** | **0.738** | **0.626** |

Paired exact McNemar by (task, init), first-only / second-only successes:

| Comparison | counts | p |
|---|---|---|
| MATCH, v remaining vs π mean (v6) | 31 / 219 | 4.9e-36 |
| MATCH, v episode mean vs π mean (v6) | 30 / 241 | 4.2e-42 |
| gate, v remaining vs π mean (v6) | 6 / 107 | 5.1e-25 |
| MATCH, q_l vs π mean (v6) | 39 / 150 | 1.5e-16 |
| MATCH, v remaining vs MATCH, q_l | 71 / 148 | 2.1e-7 |
| MATCH, v episode mean vs MATCH, q_l | 59 / 159 | 8.7e-12 |
| MATCH, v remaining vs MATCH, v episode mean | 77 / 54 | 0.054 |
| gate, v remaining vs gate, q_l (v5) | 37 / 93 | 9.7e-7 |
| MATCH, q_l vs MATCH, q_l (v5) | 89 / 78 | 0.44 |

- **By task against v6**: MATCH with the remaining v is ahead only in task 3 (9 / 4) and level in task 4 (11 / 12). It loses 33–36 pairs each in tasks 0, 1, 2 and 7.
- **Against MATCH with q_l**: MATCH with the remaining v gains in tasks 3 and 4 (13 / 3, 12 / 5) and loses in tasks 0, 1, 2, 6 and 7 (4 / 17, 3 / 19, 2 / 25, 7 / 26, 4 / 24).
- **The gate with the remaining v** equals v6 in tasks 0, 8 and 9 (0 / 0) and loses in every other task, most in tasks 1 and 2 (1 / 30, 0 / 30).

Failure categories (`classify_closed_loop_failures.py`):

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| π mean (v6) | 414 | 0 | 4 | 8 | 54 | 17 | 3 |
| MATCH, q_l | 303 | 92 | 21 | 24 | 14 | 25 | 21 |
| MATCH, v episode mean | 203 | 191 | 31 | 29 | 18 | 21 | 7 |
| MATCH, v remaining | 226 | 116 | 43 | 34 | 34 | 40 | 7 |
| gate, v remaining | 313 | 152 | 2 | 3 | 19 | 10 | 1 |

- **Stuck** episodes are those whose last 5 decisions all stay. They are the largest category in every run that uses v: 116–191 episodes, against 92 with q_l and 0 for π's mean chunk.
- **MATCH with the remaining v**:
  - Stuck is concentrated in tasks 1 and 2 (36 and 26 episodes).
  - "Other moved" is concentrated in task 0 (26 episodes): the drawer stays closed while the bowl and the plate on the table are displaced. MATCH with q_l has 19 such episodes in task 0; π's mean chunk has none.
- **The gate with the remaining v**: 96 of its 152 stuck episodes are in tasks 1, 2 and 4 (37, 32 and 27).

Executed chunks, share of all decisions:

| Run | decisions | π's mean chunk | noisy, scale 0.05 / 0.1 / 0.2 | stay | first-level best ≤ 0 |
|---|---|---|---|---|---|
| MATCH, q_l | 9233 | 0 | 0.007 / 0.104 / 0.644 | 0.245 | 0.249 |
| MATCH, v episode mean | 10930 | 0 | 0.005 / 0.066 / 0.406 | 0.524 | 0.536 |
| MATCH, v remaining | 10437 | 0 | 0.007 / 0.094 / 0.558 | 0.341 | 0.348 |
| gate, q_l (v5) | 8273 | 0.660 | – | 0.340 | 0.340 |
| gate, v remaining | 9049 | 0.527 | – | 0.473 | 0.473 |

- **MATCH never executes π's mean chunk**, with v as with q_l: 0 of 9233–10930 decisions. It executes the largest offset (scale 0.2) in 85% of its non-stay decisions under every target (0.846–0.853).
- **In successful episodes**, MATCH with the remaining v stays in 6.2% of the decisions and the gate in 2.3%. In failed episodes the shares are 41.6% and 74.8%.
- **The gate** executes π's mean chunk in 97.7% of the decisions of its successful episodes and in 25.2% of those of its failed episodes.

Per episode, means:

| Run | decisions | env steps | stays | escalations | episodes reaching K 512 |
|---|---|---|---|---|---|
| π mean (v6) | 14.7 | 143.2 | 0 | 0 | 0 |
| MATCH, q_l | 18.5 | 182.0 | 4.53 | 3.92 | 0.218 |
| MATCH, v episode mean | 21.9 | 216.7 | 11.44 | 9.83 | 0.524 |
| MATCH, v remaining | 20.9 | 207.0 | 7.11 | 5.99 | 0.322 |
| gate, v remaining | 18.1 | 178.2 | 8.55 | 0 | 0 |

Where the stays begin (R `first_decisions.py`, output `first_decisions_fixed.txt`, per task: the share of episodes whose decision 1 has every candidate at a score ≤ 0; the share of episodes that end stuck):

- **Decision 1, episode-mean v**: every candidate scores ≤ 0 at decision 1 in 82% of the task-0 and 90% of the task-7 episodes (median best score −0.09 and −0.19). 50% and 82% of those tasks' episodes end stuck.
- **Decision 1, remaining v**: every candidate scores ≤ 0 at decision 1 in 6 of the 500 episodes; the median best score is +0.39, against +0.15 for q_l.
- **The gate with the remaining v in tasks 1 and 2**: 62 of its 69 stuck episodes there stay for the first time at decision 2, 4 at decision 1 and 3 later. After one executed mean chunk, cos(Δ̂(mean chunk), v) ≤ 0. Stay leaves the scene unchanged, so the next decisions see the same state and stay again. v6 succeeds in 28 of the 37 and 29 of the 32 corresponding inits.

‖v‖ in the closed loop, median (p10–p90): episode-mean v 0.284 (0.189–0.412), remaining v under MATCH 0.270 (0.166–0.503), under the gate 0.342 (0.202–0.556). The HELDOUT regression targets have mean norm 0.285 (episode mean) and 0.414 (remaining).

Decision latency, mean (median) in ms, from the decision lines (10 processes per node, 1–3 per GPU):

| Run | decisions | decision | encode | v | π | Δ, all levels | decision, one level | decision, to K 512 |
|---|---|---|---|---|---|---|---|---|
| MATCH, q_l | 9233 | 54.0 (26.4) | 18.0 (15.6) | – | 3.0 (2.3) | 31.0 (6.0) | 29.1 (24.0) | 146.5 (142.8) |
| MATCH, v episode mean | 10930 | 85.8 (46.8) | 16.7 (14.4) | 2.0 (1.7) | 2.7 (2.1) | 61.5 (12.2) | 28.0 (23.6) | 157.5 (143.9) |
| MATCH, v remaining | 10437 | 57.3 (24.8) | 15.5 (14.4) | 1.9 (1.8) | 2.4 (2.1) | 35.7 (4.8) | 26.4 (23.6) | 134.9 (120.6) |
| gate, v remaining | 9049 | 22.0 (20.9) | 15.9 (15.1) | 1.9 (1.9) | 2.2 (2.0) | 1.6 (1.5) | 22.0 (20.9) | – |

- v costs 1.8–2.0 ms per decision, about as much as π. v6 takes 15.5 ms per decision, the encoding and π.
- The mean decision time of MATCH follows the share of decisions that escalate to K = 512 (0.45 of the decisions with the episode-mean v, 0.28 with the remaining v, 0.21 with q_l), each costing 135–158 ms.

### Projected-flow ψ

Success, with the flow ψ's own Δ, π and remaining-target v:

| Task | π mean, fixed ψ (v6) | π mean, flow ψ (`v7_mean_flow`) | MATCH, v remaining, flow ψ | gate, v remaining, flow ψ |
|---|---|---|---|---|
| 0 | 0.96 | 0.98 | 0.40 | 0.36 |
| 1 | 0.78 | 0.66 | 0.20 | 0.22 |
| 2 | 0.94 | 0.98 | 0.16 | 0.32 |
| 3 | 0.74 | 0.96 | 0.74 | 0.76 |
| 4 | 0.60 | 0.98 | 0.70 | 0.66 |
| 5 | 0.86 | 0.96 | 0.50 | 0.78 |
| 6 | 0.62 | 0.72 | 0.44 | 0.72 |
| 7 | 0.80 | 0.90 | 0.24 | 0.90 |
| 8 | 0.98 | 0.90 | 0.68 | 0.90 |
| 9 | 1.00 | 1.00 | 0.98 | 1.00 |
| **all** | **0.828** | **0.904** | **0.504** | **0.662** |

| Comparison | counts | p |
|---|---|---|
| MATCH, v remaining, flow vs π mean, flow | 20 / 220 | 9.1e-44 |
| gate, v remaining, flow vs π mean, flow | 2 / 123 | 3.7e-34 |
| MATCH, v remaining, flow vs π mean, fixed (v6) | 39 / 201 | 1.8e-27 |
| gate, v remaining, flow vs π mean, fixed (v6) | 49 / 132 | 5.5e-10 |

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| π mean, flow | 452 | 0 | 5 | 6 | 22 | 13 | 2 |
| MATCH, v remaining, flow | 252 | 81 | 40 | 43 | 37 | 37 | 10 |
| gate, v remaining, flow | 331 | 144 | 1 | 3 | 10 | 10 | 1 |

- **Against π's mean chunk on the flow ψ**:
  - The gate loses in tasks 0–5 (2 / 123), where 140 of its 144 stuck episodes are, and is level in tasks 6–9 (0 / 0).
  - MATCH loses in every task; in task 9 by one pair.
- **Executed chunks**:
  - MATCH never executes π's mean chunk (0 of 10038); it stays in 26.5% of its decisions and executes scale 0.2 in 63.6% (86.5% of the non-stay decisions).
  - The gate executes π's mean chunk in 51.6% of its decisions: 98.1% in successful episodes, 19.3% in failed ones.
- **Decision 1**: every candidate scores ≤ 0 in 2 (MATCH) and 3 (gate) of the 500 episodes; the median best score is +0.53 and +0.49.
- **Latency and ‖v‖**: v takes 1.8–2.0 ms per decision; ‖v‖ has median 0.259 (MATCH) and 0.291 (gate).

### Reading

Can v(o, l) as the match target improve on π's mean chunk? In these closed loops, no.

- **Against π's mean chunk**: every configuration that lets the match with v decide is below π's mean chunk on the same ψ.
  - Fixed ψ: MATCH with the remaining v 0.452 and with the episode-mean v 0.406, the gate with the remaining v 0.626, against 0.828 (p ≤ 5.1e-25).
  - Flow ψ: MATCH 0.504 and gate 0.662, against 0.904 (p ≤ 3.7e-34).
  - On the fixed ψ, the only task where a v configuration is ahead of v6 is task 3 under MATCH with the remaining v (0.84 against 0.74, 9 / 4). On the flow ψ, no v configuration is ahead of that ψ's π mean chunk in any task; that π reaches 0.96 in task 3.
- **Against q_l with everything else fixed**: v is worse on the fixed ψ, as the MATCH target and as the gate.
  - MATCH: 0.452 (remaining) and 0.406 (episode mean) against 0.606 (71 / 148 and 59 / 159).
  - Gate: 0.626 against 0.738 (37 / 93).
  - The remaining target is ahead of the episode mean, 77 / 54 (p = 0.054).
  - The remaining v gains over q_l in the bowl-placing tasks 3 and 4 and loses in tasks 0, 1, 2, 6 and 7.
- **What v changes: more decisions with no positive candidate.**
  - The failures move to stuck: 116–191 stuck episodes against 92 with q_l.
  - The failures that do not stall are also more frequent with the remaining v: 158 against 105 with q_l, of which 43 against 21 displace another object while the goal object stays, and 34 against 14 lift the goal object without placing it.
  - With the episode-mean v, in tasks 0 and 7 every π candidate already scores ≤ 0 at the start state.
  - With the remaining v the start states are accepted (median best score +0.39). The gate's stuck episodes in tasks 1 and 2 mostly begin at decision 2 (62 of 69), after one of π's mean chunks: from there cos(Δ̂(mean chunk), v) ≤ 0, and stay does not change the state.
- **What v does not change**: MATCH never chooses π's mean chunk under any target, and 85% of its non-stay chunks carry the largest offset.
- **Cost**: v adds about 2 ms per decision. MATCH's cost comes from the escalations to K = 512 that its extra stays trigger.

### Outputs of the closed loop

| Output | Path |
|---|---|
| closed-loop runs | `outputs/closed_loop/v7_proposal_{small_vrem,small_vep,gate_vrem,small_query}_dynA`, `outputs/closed_loop/v7_proposal_{small_vrem,gate_vrem}_flow` (`task<t>_match/results.jsonl`, `trace.jsonl`) |
| closed-loop checks | `outputs/closed_loop/p23_checks/{mean_t3_new,mean_t3_base,small_query_t3_new,small_query_t3_base,small_vep_t3}` |
| launchers, process logs, check and analysis scripts and outputs | `outputs/closed_loop/p23_runlogs/` (`README.txt`) |

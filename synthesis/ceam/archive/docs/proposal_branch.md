# Proposal branch π

The proposal branch π generates candidate action chunks at the current state. In the closed loop's default, every decision executes π's mean chunk (the MEAN strategy); the match takes no part in the decision. Under the MATCH strategy, the dynamics head Δ predicts each candidate's effect and the rule m(Δ(o, a), l) = cos(Δ̂, q_l) selects, with stay scored m = 0. Stage one (ψ, q_l) and Δ are frozen. π is trained by behaviour cloning on TRAIN demonstrations only; there is no reward, value function or RL objective.

This page covers the model, its training, the candidate generator, their offline and simulator evaluation, π's candidates in the closed loop (round 0), and π's mean chunk as the closed-loop default (phase 13).

## Model (`ceam/model/proposal_head.py`)

`ProposalHead(ProposalConfig)` maps (current `FrameBatch`, ψ(o) [..., 256], q_l [..., 256]) to the mean chunk [..., 10, 7] in the action cache's standardized space.

- **Inputs**:
  - the 2 × 64 frozen DINOv2 tokens and the standardized proprio of the frame;
  - ψ(o) from the frozen trunk;
  - q_l, the frozen text-only query direction of the task's original instruction (`InstructionTable` variants[0], the dataset task text).
- **Sequence of 141 tokens**, in this order:
  - ψ token, q_l token and proprio token, each from its own linear projection, with no position or type embedding;
  - 128 visual tokens: projection, plus a view embedding and a position embedding shared by the two views;
  - 10 action slots. Slot k is the learned step embedding e_step_k; slots have no content projection.
- **Encoder**: 4 pre-norm Transformer encoder layers (width 512, 8 heads, FFN 2048, GELU, no dropout) with full bidirectional attention.
  - The encoder is called without a mask or key padding mask; each of the four self-attention calls receives `attn_mask` and `key_padding_mask` None and `is_causal` False.
- **Output**: each slot's output goes through LayerNorm and a linear map 512 → 7. Slot k gives step k of the chunk.
- **Initialization**: the output linear map (`action_head`) starts at zero weight and bias. An untrained π proposes the standardized chunk 0, the action cache's per-dim mean action, at every step (measured max |â| = 0).
- `ProposalConfig` has the fields of `DynamicsConfig` plus `query_dim` (256).
- A trailing-shape or leading-dim mismatch raises `ValueError`.

Parameters (13,315,079):

| Part | Parameters |
|---|---|
| encoder layers × 4 | 12,609,536 |
| visual projection | 393,728 |
| ψ projection, q_l projection | 131,584 each |
| position embedding (64) / view embedding (2) | 32,768 / 1,024 |
| step embedding (10), proprio projection | 5,120 each |
| output LayerNorm + linear 512 → 7 | 1,024 + 3,591 |

Checks (`tmp/p9/inspect_proposal.py`):

- Output [2, 3, 10, 7] for inputs with leading dims [2, 3]. The encoder input is [6, 141, 512].
- With a random output layer, changing ψ, q_l, proprio or the last visual token of the wrist view changes all 10 slots.
- Permuting the batch permutes the output exactly.

Cost, measured on an L40S (node-64):

| Workload | Time | Peak memory |
|---|---|---|
| batch 256 forward + backward + AdamW step, bf16 autocast | 54.6 ms | 3.14 GiB |
| the same in fp32 | 135 ms | 5.46 GiB |
| batch 1 forward, fp32 | 1.6 ms | – |

## Training

### Code

- **Configuration**: `ceam/training/proposal_config.py` (`ProposalTrainingConfig`).
- **Checkpoints**: `ceam/training/proposal_checkpoint_store.py` (`ProposalCheckpointStore`, `LoadedProposal`).
- **Trainer**: `ceam/training/proposal_trainer.py` (`ProposalTrainer`, `ProposalMetrics`, `ProposalSummary`).
- **Entry point**: `scripts/train_proposal.py --config configs/proposal/goal_proposal.json`.

### Configuration and data

- The JSON config has exactly the dataclass fields and uses the same `$CEAM_ROOT` / `$CEAM_DATA_ROOT` path conventions as stage two.
- The feature cache, dataset, instructions and split come from the referenced stage-one checkpoint.
- A checkpoint holds the π state dict, the config JSON and the step. The stage-one parameters are not copied into it.
- **Samples**: `StageTwoSampler` on TRAIN, uniform over every (episode, t) with t + 10 ≤ T − 1, with replacement, seed 0.
- **ψ**: a table of every feature-cache row, computed once with the frozen trunk in float32.
- **q_l**: one direction per task, from its original instruction. They are computed one text per call in float32, as the closed loop computes them, and indexed by the batch's task id. ‖q_l‖ ranges from 24.7 (task 7) to 35.6 (task 4).

### Loss and optimizer

- **Loss**: MSE between π's mean chunk and the standardized demonstration chunk, over batch, 10 steps and all 7 dims, in float32.
- **Optimizer and schedule**: the stage-two recipe.
  - AdamW, lr 3e-4. Weight decay 0.05 applies only to matrices whose name contains neither "token" nor "embedding": 21 tensors, 13,246,464 parameters.
  - Linear warmup over 150 steps, then cosine decay to 0 at 3000 steps.
  - Batch 256, bf16 autocast, global gradient-norm clipping at 1.0, seed 0.

### Metrics and checkpoints

- `metrics.jsonl` logs these fields every 50 steps: MSE, arm MSE, gripper MSE, gripper agreement (binarized at 0.5 in the dataset convention), gradient norm, steps/s, sampling time and peak memory.
- A checkpoint is written every 500 steps.
- An existing run directory is refused.

### Smoke run

- **Run**: 200 steps with 20 warmup steps.
- **Loss**: MSE 1.066 → 0.203 on TRAIN batches.
- **HELDOUT MSE at step 200**: 0.263.
- **Cost**: 57 s, 3.30 GiB peak.

### Full run

`outputs/runs/goal_proposal`, code `6722728`, node-61 CUDA 0.

- **Time**: the 3000 updates took 690.8 s. The whole process took 13 min 13 s, including loading the feature cache into memory and 29.6 s for the ψ table.
- **Memory**: 3.30 GiB peak GPU memory and 11.3 GiB peak host RSS.
- **Speed**: host-side gathering of each batch took 0.21–0.24 s per step, so training ran mostly at 3.4–3.7 steps/s (4.3 on average).
  - The process spent 498 s in the kernel against 304 s in user time, with 28.9 M minor page faults.
  - In other processes the same `StageTwoSampler` gathered a batch in 15–30 ms (the stage-two runs; a standalone profile on node-61) or in 190 ms (a standalone profile on node-64). The rate depends on the node's memory state, not on π.

TRAIN batches during training:

| step | lr | MSE | arm MSE | gripper MSE | gripper agreement | grad norm |
|---|---|---|---|---|---|---|
| 0 | 2.0e-6 | 1.066 | 1.068 | 1.054 | 0.588 | 1.31 |
| 150 | 3.0e-4 | 0.255 | 0.269 | 0.172 | 0.952 | 1.19 |
| 500 | 2.9e-4 | 0.142 | 0.150 | 0.091 | 0.973 | 0.61 |
| 1000 | 2.4e-4 | 0.076 | 0.079 | 0.060 | 0.982 | 0.48 |
| 1500 | 1.6e-4 | 0.042 | 0.046 | 0.020 | 0.995 | 0.37 |
| 2000 | 8.2e-5 | 0.023 | 0.023 | 0.020 | 0.994 | 0.23 |
| 2500 | 2.2e-5 | 0.016 | 0.018 | 0.006 | 0.999 | 0.15 |
| 2999 | 9.1e-11 | 0.015 | 0.017 | 0.006 | 0.999 | 0.14 |

Every checkpoint on all valid (episode, t) of each split (`scripts/evaluate_proposals.py curve`, `outputs/analysis/proposals_goal/curve/`; code `cfcfec6`, node-64 CUDA 1, 200 s):

| step | TRAIN MSE (43,701) | HELDOUT MSE (4864) | HELDOUT arm / gripper MSE | HELDOUT gripper agreement |
|---|---|---|---|---|
| 500 | 0.142 | 0.265 | 0.286 / 0.138 | 0.958 |
| 1000 | 0.074 | 0.273 | 0.296 / 0.133 | 0.962 |
| 1500 | 0.040 | 0.272 | 0.295 / 0.135 | 0.963 |
| 2000 | 0.023 | 0.272 | 0.293 / 0.147 | 0.961 |
| 2500 | 0.016 | 0.274 | 0.294 / 0.151 | 0.962 |
| 3000 | 0.015 | 0.274 | 0.294 / 0.152 | 0.962 |

HELDOUT MSE reaches about 0.27 by step 500 and stays there, while TRAIN MSE keeps falling to 0.015. From step 500 on, π fits the TRAIN chunks without improving on HELDOUT. The checkpoint used below is step 3000.

## Candidates (`ceam/policy/proposal_candidates.py`)

`ProposalCandidates(head, actions, count, scales, dct_coefficients, seed, input_min, input_max, gripper_flip)` generates the candidates.

`propose(encoded, psi, query) → ProposedChunks` takes:

- `encoded`: the frozen-encoder `FrameBatch` of one frame, [1, ...];
- `psi`: ψ(o);
- `query`: q_l.

π runs in eval mode, float32, under inference mode. From its standardized mean chunk m, the generator builds:

- **The gripper**: m's gripper column is unstandardized and binarized per step, g = 1 where it is ≥ 0.5, else 0.
- **Candidate 0**: m's arm dims with gripper g.
- **Candidate 1, only with `gripper_flip`**: the same with the gripper flipped at every step, 1 − g.
- **Noisy candidates**, from `first_noisy` (2 with `gripper_flip`, else 1) to count − 1: m's arm dims plus a smooth offset in the standardized space, gripper g.
  - The offset is δ = s · √(10 / J) · Σ_{j<J} c_j b_j, with c_j ~ N(0, I_6) and b_j the orthonormal DCT-II basis over the 10 steps (b_0 constant). J is `dct_coefficients`.
  - s cycles through `scales`, one per noisy candidate, starting at the first.
  - The mean squared offset per entry, averaged over the steps, is s². With J = 1 the offset is constant over the chunk, like the perturbations of data A (`docs/dynamics_data.md`).
- **Clipping**: every candidate's arm dims are unstandardized and clipped to the controller's input range, [−1, 1] in LIBERO.
- **Outputs** (`ProposedChunks`):
  - `raw` [K, 10, 7], the dataset-convention chunk as executed;
  - `standardized` [K, 10, 7], `ActionCache.standardize(raw)` on the head's device;
  - `scales` [K], 0 for candidate 0 and the flipped candidate;
  - `mean`, π's raw output [10, 7];
  - `stream`, the `NoiseStream` of the offsets.
- **Noise streams**: `sample(mean, count, stream, device)` builds count candidates from a mean chunk with offsets drawn from `numpy.random.default_rng([seed, *stream.words])`, the coefficients of the noisy candidates in order. The candidates depend only on the seed, the stream, the mean chunk and `gripper_flip`, and the count candidates of a stream are the first count of any larger number drawn from it. `propose(encoded, psi, query, stream)` is `sample` of π's mean chunk with `count` candidates on the given stream. The caller names the stream (phase 13, below): the closed loop keys it by episode and decision, the offline evaluation by sample position.
- The phase-9 evaluation (`scripts/evaluate_proposals.py`) uses `gripper_flip` on.

Checks (`tmp/p9/check_candidates.py`):

- On an untrained head with a random output layer (16 random inputs), and on the trained head at 64 HELDOUT frames:
  - the same seed gives bit-identical `raw` and `standardized` over every call; seed 1 differs in every call;
  - max |standardize(raw) − standardized| = 0, and max |unstandardize(standardized) − raw| = 6e-8;
  - the arm range is [−1, 1] and the gripper values are {0, 1};
  - candidate 0's arm dims equal clip(unstandardize(m)), and the binarized and flipped grippers are as specified;
  - the DCT basis is orthonormal to 2e-15.
- Offsets on unclipped candidates of the trained head:

| scales, J | RMS offset | per-step RMS | RMS step-to-step change | clipped arm entries |
|---|---|---|---|---|
| (1.0), J 1 | 0.93 | 0.93 at every step | 0 | 0.051 |
| (1.0), J 3 | 0.92 | 0.76–1.15 (larger at the chunk ends) | 0.38 | 0.051 |
| (1.0), J 10 | 0.96 | 0.93–1.00 | 1.35 | 0.052 |
| (0.25, 0.5, 1.0), J 3 | 0.50 | 0.41–0.63 | 0.21 | 0.028 |

## Offline evaluation on HELDOUT

Run: `scripts/evaluate_proposals.py offline`, output `outputs/analysis/proposals_goal/offline/` (`config.json`, `frames.npz`, `summary.txt`).

- **Code and host**: code `cfcfec6`; node-64 CUDA 2; 335 s.
- **Frames**: all 4864 valid HELDOUT (episode, t) of the feature cache.
- **Retrieval candidates**: from `DemoRetrieval` over TRAIN, 32 each.
  - frozen retrieval uses the key table `outputs/state_tables/libero_goal_frozen_keys.npy`;
  - ψ retrieval uses the ψ table `outputs/state_tables/goal_instruction_contrast_step5000.npy` as keys.
- **Predicted effects**: Δ̂ comes from the +A head (`outputs/runs/goal_dynamics_perturb/checkpoints/step_003000.pt`) on the cached frame, in float32.
- **Distances**: RMS over the 10 × 7 (all) or 10 × 6 (arm) entries of the standardized difference.
- **Settings**: a setting is written `K:scales/J`. The first setting, 32:(0.25, 0.5, 1)/3, defines the "π mean only" and "π + frozen retrieval" rows.

### (a) Mean chunk against the demonstration chunk

| chunk | MSE | arm MSE | gripper MSE | gripper agreement |
|---|---|---|---|---|
| π mean chunk | 0.274 | 0.294 | 0.152 | 0.962 |
| π candidate 0 (gripper binarized, arm clipped) | 0.275 | 0.294 | 0.164 | 0.962 |
| top-1 frozen-retrieval chunk | 0.501 | 0.546 | 0.232 | 0.946 |
| top-1 ψ-retrieval chunk | 0.505 | 0.546 | 0.257 | 0.940 |
| zero chunk (the mean action) | 1.021 | 1.018 | 1.038 | 0.601 |

π's mean chunk has about half the error of the demonstration chunk that follows the nearest retrieved TRAIN state.

### (b) Coverage: distance from the demonstration chunk to the nearest candidate

| source | K | all dims: median / mean / p90 | arm dims: median / mean / p90 |
|---|---|---|---|
| π mean only | 1 | 0.423 / 0.468 / 0.754 | 0.431 / 0.483 / 0.778 |
| π (0.25, 0.5, 1), J 3 | 32 | 0.399 / 0.436 / 0.684 | 0.406 / 0.450 / 0.705 |
| π (0.25, 0.5, 1), J 1 | 32 | 0.353 / 0.391 / 0.615 | 0.355 / 0.398 / 0.629 |
| π (0.25, 0.5, 1), J 10 | 32 | 0.421 / 0.461 / 0.727 | 0.430 / 0.478 / 0.765 |
| π (0.1, 0.25, 0.5), J 3 | 32 | 0.389 / 0.429 / 0.679 | 0.395 / 0.442 / 0.708 |
| π (0.5, 1, 2), J 3 | 32 | 0.418 / 0.453 / 0.710 | 0.427 / 0.469 / 0.744 |
| π (0.25, 0.5, 1), J 3 | 128 | 0.372 / 0.411 / 0.649 | 0.375 / 0.421 / 0.666 |
| frozen retrieval | 32 | 0.310 / 0.345 / 0.564 | 0.322 / 0.359 / 0.575 |
| ψ retrieval | 32 | 0.314 / 0.365 / 0.598 | 0.324 / 0.379 / 0.618 |
| π J 3 + frozen retrieval | 64 | 0.297 / 0.329 / 0.536 | 0.307 / 0.342 / 0.541 |
| π J 1 + frozen retrieval | 64 | 0.286 / 0.317 / 0.509 | 0.296 / 0.328 / 0.515 |

- Adding noisy samples brings π's nearest candidate only a little closer than the mean chunk alone (0.423 → 0.353 at best). Random offsets rarely point along the mean chunk's error.
- Constant offsets (J = 1) cover best. Rough offsets (J = 10) cover worst.
- The 32 frozen-retrieval candidates are nearer to the demonstration chunk than π's 32: 0.310 against 0.353.
  - π with J = 1 is nearer than frozen retrieval in 37% of frames; with J = 3, in 24%.
  - The union of both sources is nearest.

### (c) Candidates with a positive predicted match

| source | K | mean share of candidates with Δ̂ m > 0 | frames with at least one | mean largest Δ̂ m |
|---|---|---|---|---|
| π mean only | 1 | 0.975 | 0.975 | 0.425 |
| π (0.25, 0.5, 1), J 3 | 32 | 0.950 | 0.991 | 0.490 |
| π (0.25, 0.5, 1), J 1 | 32 | 0.923 | 0.991 | 0.499 |
| π (0.25, 0.5, 1), J 10 | 32 | 0.962 | 0.990 | 0.477 |
| π (0.1, 0.25, 0.5), J 3 | 32 | 0.969 | 0.988 | 0.470 |
| π (0.5, 1, 2), J 3 | 32 | 0.874 | 0.991 | 0.506 |
| π (0.25, 0.5, 1), J 3 | 128 | 0.949 | 0.991 | 0.511 |
| frozen retrieval | 32 | 0.906 | 0.987 | 0.474 |
| ψ retrieval | 32 | 0.923 | 0.976 | 0.458 |
| π J 3 + frozen retrieval | 64 | 0.928 | 0.991 | 0.498 |
| π J 1 + frozen retrieval | 64 | 0.914 | 0.991 | 0.505 |
| demonstration chunk | 1 | 0.977 (true z: 0.952) | – | 0.424 (true z: 0.401) |

- In 61 of the 4864 frames, no frozen-retrieval candidate has a positive Δ̂ match. π has one in 17 of them, under both J = 3 and J = 1. π has none in 44 frames.
- By time point f = t / (T − 1 − 10), π (J = 3) against frozen retrieval:

| f | frames | nearest distance π / frozen | share of Δ̂ m > 0, π / frozen |
|---|---|---|---|
| [0, 0.25) | 1225 | 0.291 / 0.249 | 0.909 / 0.756 |
| [0.25, 0.5) | 1198 | 0.423 / 0.299 | 0.952 / 0.955 |
| [0.5, 0.75) | 1208 | 0.426 / 0.333 | 0.950 / 0.921 |
| [0.75, 1] | 1233 | 0.450 / 0.374 | 0.988 / 0.993 |

At these demonstration states, both sources almost always hold a candidate that Δ̂ scores positive. The frozen-retrieval candidates' lower positive share concentrates in the first quarter of the episodes.

## Simulator check

### Protocol

`scripts/evaluate_proposals.py simulate`, code `cfcfec6`, node-64 CUDA 3–6, one process per candidate setting, 267–285 s each (0.8–0.9 s per sample).

- **Samples**: the 129 samples of the phase-12 protocol, 43 HELDOUT episodes × t = round(f · (T − 11)) with f ∈ {0.1, 0.4, 0.7}.
- **Restoration**: each frame is restored with `DemoStateRestorer` (`LiberoTaskEnv.restore`) and encoded with the closed loop's `ObservationEncoder`.
- **Executed chunks**: four chunks are executed, each for all 10 steps with `LiberoTaskEnv.rollout`, restoring the frame before each:
  - **demo**: the demonstration chunk;
  - **mean**: π's candidate 0;
  - **match**: the MATCH choice among π's 32 candidates, with Δ̂ from the +A head on the simulator frame. Stay executes 10 zero-delta steps with the gripper command in force at t;
  - **retrieval**: the MATCH choice among the 32 frozen-retrieval candidates of the frame, as the first retrieval of a closed-loop decision chooses under predicted effects.
- **Measures**:
  - the true match m = cos(ψ(o′) − ψ(o), q_l);
  - the goal-progress label of the scene change (`GoalProgress` of `scripts/analyze_rollout_progress.py`, thresholds of `docs/rollout_progress.md`): progress, none, regress or mixed toward the BDDL goal.
- **Paired differences**: mean [95% interval from 10,000 bootstrap resamples of episodes] (samples with difference > 0 / = 0 / < 0).
- **Outputs**:
  - `outputs/analysis/proposals_goal/simulate/`, for (0.25, 0.5, 1), J 3;
  - `outputs/analysis/proposals_goal_j1/simulate/`, for J 1;
  - `outputs/analysis/proposals_goal_large/simulate/`, for (0.5, 1, 2), J 3;
  - `outputs/analysis/proposals_goal_small/simulate/`, for (0.1, 0.25, 0.5), J 3.
  - Each holds `executed.jsonl`, `samples.jsonl` and `summary.txt`.
- **Reproducibility across the four processes**: the demo, mean and retrieval rows are shared by all four. Their true m differs by at most 2.6e-4 between processes, and the choices and progress labels are identical.

### Executed chunks, 129 samples

The π candidates use (0.25, 0.5, 1), J 1 in the "match, J 1" row and (0.25, 0.5, 1), J 3 in the "match, J 3" row:

| chunk | mean true m | true m > 0 | mean Δ̂ m | stay | progress | regress | RMS distance to demo |
|---|---|---|---|---|---|---|---|
| demo | 0.369 | 0.930 | 0.381 | 0 | 0.349 | 0.008 | 0 |
| π mean | 0.384 | 0.938 | 0.390 | 0 | 0.341 | 0.000 | 0.458 |
| π match, J 3 | 0.384 | 0.938 | 0.452 | 2 | 0.287 | 0.031 | 0.865 |
| π match, J 1 | 0.410 | 0.953 | 0.465 | 3 | 0.326 | 0.031 | 0.878 |
| frozen-retrieval match | 0.374 | 0.930 | 0.437 | 3 | 0.326 | 0.008 | 0.683 |

Paired differences of the true match:

| comparison | all | f = 0.1 | f = 0.4 | f = 0.7 |
|---|---|---|---|---|
| π mean − demo | +0.015 [+0.004, +0.028] (81/0/48) | +0.013 [+0.003, +0.026] | +0.014 [−0.007, +0.041] | +0.017 [+0.001, +0.040] |
| π match J 3 − demo | +0.014 [−0.014, +0.040] (93/0/36) | +0.021 [−0.020, +0.052] | +0.017 [−0.040, +0.062] | +0.004 [−0.054, +0.048] |
| π match J 1 − demo | +0.040 [+0.013, +0.067] (100/0/29) | +0.046 [+0.028, +0.066] | +0.057 [+0.002, +0.114] | +0.017 [−0.028, +0.052] |
| π match J 1 − π mean | +0.025 [+0.002, +0.047] (98/0/31) | +0.033 [+0.017, +0.049] | +0.043 [−0.013, +0.098] | +0.000 [−0.041, +0.025] |
| frozen-retrieval match − demo | +0.005 [−0.019, +0.027] (89/0/40) | +0.002 [−0.026, +0.027] | −0.006 [−0.058, +0.042] | +0.018 [−0.024, +0.055] |
| π match J 3 − frozen-retrieval match | +0.009 [−0.013, +0.030] (67/2/60) | +0.019 [−0.007, +0.045] | +0.024 [−0.037, +0.079] | −0.014 [−0.077, +0.037] |
| π match J 1 − frozen-retrieval match | +0.035 [+0.009, +0.061] (78/3/48) | +0.044 [+0.021, +0.074] | +0.064 [+0.011, +0.119] | −0.002 [−0.052, +0.045] |

Paired differences of progress (1 when labelled progress, else 0):

| comparison | difference |
|---|---|
| π mean − demo | −0.008 [−0.054, +0.039] (4/120/5) |
| π match J 3 − demo | −0.062 [−0.124, −0.008] (2/117/10) |
| π match J 1 − demo | −0.023 [−0.085, +0.039] (6/114/9) |
| frozen-retrieval match − demo | −0.023 [−0.078, +0.031] (5/116/8) |
| π match J 1 − frozen-retrieval match | +0.000 [−0.062, +0.062] (7/115/7) |

Mean true m by time point:

| f | demo | π mean | π match J 3 | π match J 1 | frozen-retrieval match |
|---|---|---|---|---|---|
| 0.1 | 0.294 | 0.306 | 0.315 | 0.340 | 0.296 |
| 0.4 | 0.360 | 0.374 | 0.377 | 0.417 | 0.354 |
| 0.7 | 0.455 | 0.472 | 0.459 | 0.472 | 0.474 |

- **π's mean chunk**:
  - Executed from the restored HELDOUT state, it scores slightly higher than the demonstration chunk by the true match (+0.015).
  - Its progress share is the same as the demonstration chunk's (0.341 against 0.349).
  - Δ̂ judges it almost without bias: mean Δ̂ m − true m = 0.006.
- **The MATCH choice among π's candidates**:
  - With J = 1 it adds +0.025 over the mean chunk and +0.035 over the frozen-retrieval choice.
  - With J = 3 it adds nothing over the mean chunk (−0.000) and lowers the progress share against the demonstration (−0.062).
  - Δ̂ is optimistic about the chunk it chooses: mean Δ̂ m − true m is 0.052 with J 1, 0.065 with J 3 and 0.060 for the frozen-retrieval choice.
- **Samples where Δ̂ m > 0 but true m ≤ 0**: 3 with J 1, 6 with J 3.
- **Regressions**: 4 of the 129 π choices are labelled regress under J 3 and under J 1. Under J 1 all four have a positive true match (+0.43 to +0.47); under J 3, two of the four do (for example +0.658 at episode 128, t 66). A positive m(z) does not imply scene progress, as in `docs/rollout_progress.md`.
- **The gripper-flipped candidate**: MATCH chose it in 7 (J 1) to 18 (small scales) of the 129 samples, 4–6 of them on task 9.
  - Its mean true m is 0.34 (J 1) and 0.29 (J 3), against 0.43 and 0.41 for the chosen noisy candidates.
  - Under J 3 and under the small scales it is the worst MATCH choice: at episode 223, t 66 (task 5), true m −0.584, labelled regress, against the demonstration's +0.403. Δ̂ scored it +0.471.
  - Δ has no gripper-flipped chunks in its training data (data A keeps the demonstration's gripper column), so its predictions for this candidate are extrapolations.

### Candidate settings for the MATCH choice

Paired against (0.25, 0.5, 1), J 3, on the same 129 samples (`tmp/p9/compare_settings.py`):

| setting | mean true m of the MATCH choice | true m difference | progress difference |
|---|---|---|---|
| (0.25, 0.5, 1), J 3 | 0.384 | – | – |
| (0.25, 0.5, 1), J 1 | 0.410 | +0.026 [−0.001, +0.054] (69/9/51) | +0.039 [−0.016, +0.093] (8/118/3) |
| (0.5, 1, 2), J 3 | 0.390 | +0.006 [−0.024, +0.036] (67/10/52) | +0.039 [−0.023, +0.101] (9/116/4) |
| (0.1, 0.25, 0.5), J 3 | 0.376 | −0.008 [−0.026, +0.009] (43/14/72) | +0.016 [−0.031, +0.062] (6/119/4) |

## Candidate setting for the closed loop

π's candidates in the closed loop use count 32, scales (0.25, 0.5, 1.0), **dct_coefficients 1** (constant offsets over the chunk) and seed 0. `scripts/evaluate_proposals.py` uses this setting by default.

- **Offline coverage**: among the 32-candidate settings, J = 1 has the nearest candidate to the HELDOUT demonstration chunk (median 0.353 against 0.399 for J = 3). It is nearer than frozen retrieval in 37% of frames against 24%.
- **Consistency with Δ**: constant offsets are the form of the perturbations in data A, on which the +A head was trained (σ ∈ {0.5, 1, 2}). The scales 0.25–1.0 lie inside or just below that range.
- **Simulator check**:
  - The MATCH choice under J = 1 has the highest true match: 0.410. That is +0.040 over the demonstration chunk, +0.025 over π's mean chunk and +0.035 over the frozen-retrieval choice; all three intervals exclude 0.
  - Its progress share is not distinguishable from the demonstration's.
  - It has the fewest Δ̂-positive, true-non-positive choices (3).
- **Selection caveat**: the simulator comparison chose J = 1 among four settings on the same 129 samples. The paired advantage of J = 1 over J = 3 on those samples is +0.026 [−0.001, +0.054]. The offline coverage on 4864 frames supports the same choice independently.
- **Other settings**:
  - Larger scales (0.5, 1, 2) lower the share of Δ̂-positive candidates (0.874) without improving the choice.
  - Smaller scales (0.1, 0.25, 0.5) make the flipped-gripper variant the most frequent non-mean choice (18 of 129).
- **Gripper-flipped candidate**: Δ has not been trained on gripper flips. Its choices score below the noisy candidates on average and, under J = 3, include the worst choice. The closed loop leaves it out unless `--proposal-gripper-flip` is given.

## Closed loop (phase 10)

### Candidate sources

`ClosedLoopRunner(…, candidate_source, proposals)` (`ceam/policy/closed_loop_runner.py`) takes a `CandidateSource`; `scripts/run_closed_loop.py --candidate-source {retrieval,proposal,mixed}`.

- **RETRIEVAL** (default): the chunks after the K TRAIN states with the nearest retrieval keys, as before.
- **PROPOSAL**: the K candidates of a level come from `ProposalCandidates` (`--proposal-checkpoint`) for the current observation's frozen-encoder frames, ψ(o) and q_l. Retrieval is not queried.
- **MIXED**: the K retrieved candidates followed by the K proposed candidates of the level, scored together (2K candidates).
- **Effects**: PROPOSAL and MIXED require `--effect-source predicted`; proposed chunks have no observed effect. PROPOSAL under the MEAN strategy also accepts `--effect-source none` (phase 13).
- **Stay and the rule** are unchanged. Stay scores m = 0 and wins ties. After at least `stay_patience` consecutive stays, a decision whose selection is stay scores the later levels of the schedule, K = 128 and then 512.
- **Levels and noise streams under proposals**:
  - π runs once per decision (`propose`, count = the first K). In the phase-10 runs (code `63ada75`), its candidates came from stream n = the number of decisions the runner's `ProposalCandidates` served before, counted over all episodes of the process from 0, so an episode's candidates depended on the number of decisions of the episodes run before it in the same process. The current code keys the stream by (seed, task, start, init, decision) (phase 13, below).
  - A later level draws K candidates from the same mean chunk and stream (`sample`). They are the earlier level's candidates followed by more noisy samples on the same scale cycle.
- **Settings**: the phase-9 setting by default: count = first K = 32, `--proposal-scales 0.25,0.5,1.0`, `--proposal-dct 1`, `--proposal-seed 0`.
  - The gripper-flipped candidate is off unless `--proposal-gripper-flip` is given. Without it, candidate 0 is π's mean chunk and candidates 1 … K − 1 are noisy.
  - The clipping range is the arm controller's input range, read from the env.
  - π's stage-one checkpoint must be `--checkpoint` and its action cache `--action-cache`.
- **Match target**: unchanged. The `goal_subspace` target with PROPOSAL and the `goal` target with MIXED each ran a one-episode check. The phase-10 runs use `query`.

### Where executed chunks come from

- **`EpisodeResult`**:
  - `candidate_source`;
  - `proposal_chunks`, the executed chunks proposed by π;
  - `own_task_fraction`, now the share of own-task chunks among the executed retrieved chunks (the same as before under RETRIEVAL);
  - `mean_effect_cosine` and `choice_agreement`, NaN under PROPOSAL and MIXED.
- **`DecisionTrace`**:
  - `chosen_source`: retrieval, proposal or stay;
  - `chosen_task` and `chosen_row`: −1 for stay as before, −2 (`ClosedLoopRunner.NOT_RETRIEVED`) for a proposed chunk;
  - `chosen_proposal`: the index among the level's proposed candidates (0 = π's mean chunk), else −1;
  - `candidate_counts` holds the level's K. Under MIXED a level scores 2K candidates. `own_task_candidates` counts retrieved candidates only.
- **Recorded transitions** (`--record-transitions`):
  - kinds `rollout` (retrieved), `rollout_proposal` and `rollout_stay`;
  - `RecordedEpisode.proposal_transitions`;
  - the shard provenance holds the candidate source and the proposal settings.
- **Scripts**:
  - `summarize_closed_loop.py` prints the candidate source and the proposed share of executed chunks. Older results read as retrieval with 0 proposed chunks.
  - `classify_closed_loop_failures.py` recognises a stay by `chosen_source` (in older traces, by `chosen_task` −1). Its stuck-onset table has an "after proposed chunk" column.
- **Decision log line**: encoding time, π time (`propose`, including candidate generation) and the Δ time of every level, synchronized.

### Checks

Code `63ada75`, node-61, task 3 with the round-0 protocol below. Outputs are in `outputs/closed_loop/p10_checks/`; the comparison script is `tmp/p10/compare_checks.py`.

- **RETRIEVAL** reproduces `v3_frozen_predicted_rule_dyn_perturb` task 3, inits 0–4, episode for episode.
  - Every results field except the wall time is identical, and so is every trace field of every decision and scene.
  - The five episodes are 3 successes and 2 failures; both failures escalate to K = 512.
- **MIXED**: identical results and traces across three runs of 5 episodes: two identical commands on different GPUs and one with `--record-transitions`. Sampled starts (`--start sampled --inits 0-2`) with recording on and off are also identical.
- **PROPOSAL**: sampled starts with recording on and off are identical.
  - On benchmark starts, two identical commands differ in 4 of 5 episodes, first at decisions 15–28. Recording on against off differs in 1 of 5 episodes, at decision 27.
  - Per-call checksums locate the cause (`tmp/p10/diag_determinism.py`, `tmp/p10/diag_images.py`, 4 processes each). Up to the first difference, the following are bit-identical across processes: the observations, frozen tokens, ψ, π's mean chunk, the candidates and Δ̂.
  - The first difference is the rendered wrist image. At bit-identical qpos and qvel, 1–2 pixels differ by one intensity level.
  - π's candidates carry the resulting change of π's mean chunk into the executed actions, and the next state differs (qpos max |Δ| 3–6e-6). A retrieval episode changes only if such a difference flips a choice, because retrieved chunks are fixed demonstration chunks.
  - Repeated renders of one state within a process are identical (`tmp/p10/diag_render.py`).
- **Recorded transitions** agree with the traces: each complete chunk is one transition, of the kind given by its decision's `chosen_source`. This holds in all 16 recorded episodes.
- **Other proposal settings**: another `--proposal-seed` gives different episodes. With `--proposal-gripper-flip`, the flipped candidate was executed once in 59 chunks.

### Round 0: 10 tasks × 50 benchmark inits

Protocol:

- the phase-7 protocol of `v3_frozen_predicted_rule_dyn_perturb`: seed 0, 300 env steps, 10 settle steps, `--check-start-observation`, `--trace`;
- rule 32/128/512 with patience 2, the Δ +A head (`goal_dynamics_perturb`, step 3000), `--match-target query`;
- π `outputs/runs/goal_proposal` step 3000 with the setting above;
- frozen retrieval keys where retrieval is used.

Code `63ada75`, launcher `tmp/p10/launch_round0.sh`, one process per task. Analysis: `tmp/p10/round0_analysis.py`; failure classification: `tmp/p10/logs/round0_failure_classification.txt`.

| Run (`outputs/closed_loop/`) | Candidates | Host, GPUs |
|---|---|---|
| `v5_proposal_dynA` | PROPOSAL | node-61, CUDA 0–7 |
| `v5_mixed_dynA` | MIXED | node-64, CUDA 1–7 |
| `v5_proposal_dynA_replicate` | PROPOSAL, the same command again | node-65, CUDA 0–3, 5–7 |
| `v5_proposal_mean_only_dynA` | π's mean chunk executed at every decision | node-61, CUDA 0–7 |

The last run uses `--escalation 1 --stay-patience 0 --stuck-action best` (`tmp/p10/launch_mean_only.sh`). Its only candidate is π's mean chunk, which is executed even when Δ̂ scores it ≤ 0.

Success:

| Task | retrieval, observed | retrieval, Δ +A | proposal | mixed | proposal, replicate | π mean chunk only |
|---|---|---|---|---|---|---|
| 0 | 0.16 | 0.20 | 0.04 | 0.02 | 0.04 | 0.96 |
| 1 | 0.00 | 0.02 | 0.02 | 0.00 | 0.02 | 0.78 |
| 2 | 0.32 | 0.28 | 0.04 | 0.06 | 0.10 | 0.94 |
| 3 | 0.70 | 0.46 | 0.16 | 0.12 | 0.16 | 0.74 |
| 4 | 0.46 | 0.26 | 0.22 | 0.16 | 0.22 | 0.60 |
| 5 | 0.50 | 0.74 | 0.18 | 0.18 | 0.18 | 0.86 |
| 6 | 0.08 | 0.10 | 0.00 | 0.00 | 0.00 | 0.60 |
| 7 | 0.00 | 0.06 | 0.02 | 0.04 | 0.02 | 0.80 |
| 8 | 0.08 | 0.18 | 0.06 | 0.02 | 0.06 | 0.98 |
| 9 | 0.96 | 0.96 | 0.46 | 0.66 | 0.62 | 1.00 |
| **all** | **0.326** | **0.326** | **0.120** | **0.126** | **0.142** | **0.826** |

Paired exact McNemar (first-only / second-only successes):

| Comparison | counts | p |
|---|---|---|
| proposal vs retrieval, Δ +A | 18 / 121 | 6.2e-20 |
| proposal vs retrieval, observed | 22 / 125 | 1.1e-18 |
| mixed vs retrieval, Δ +A | 17 / 117 | 1.5e-19 |
| mixed vs retrieval, observed | 17 / 117 | 1.5e-19 |
| mixed vs proposal | 40 / 37 | 0.82 |
| proposal, replicate vs proposal | 13 / 2 | 0.0074 |
| π mean chunk only vs proposal | 362 / 9 | 1.4e-94 |
| π mean chunk only vs retrieval, Δ +A | 267 / 17 | 6.0e-59 |
| π mean chunk only vs retrieval, observed | 272 / 22 | 5.5e-56 |

Failure categories (`classify_closed_loop_failures.py`):

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| retrieval, observed | 163 | 35 | 54 | 59 | 59 | 117 | 13 |
| retrieval, Δ +A | 163 | 80 | 44 | 60 | 44 | 82 | 27 |
| proposal | 60 | 61 | 49 | 89 | 78 | 121 | 42 |
| mixed | 63 | 54 | 64 | 98 | 51 | 129 | 41 |
| proposal, replicate | 71 | 62 | 48 | 88 | 79 | 114 | 38 |
| π mean chunk only | 413 | 0 | 4 | 9 | 55 | 16 | 3 |

Per episode, means:

| Run | stays | escalations | episodes reaching K 512 | decisions | env steps | proposed share of executed chunks |
|---|---|---|---|---|---|---|
| retrieval, observed | 3.64 | 1.87 | 0.284 | 23.4 | 232.7 | 0 |
| retrieval, Δ +A | 5.27 | 3.93 | 0.308 | 23.5 | 234.1 | 0 |
| proposal | 4.28 | 3.04 | 0.214 | 27.7 | 276.1 | 1 |
| mixed | 3.93 | 2.89 | 0.200 | 27.5 | 274.6 | 0.641 |
| proposal, replicate | 4.32 | 3.06 | 0.220 | 27.2 | 271.0 | 1 |
| π mean chunk only | 0 | 0 | 0 | 14.7 | 143.7 | 1 |

- **Stays** concentrate in failed episodes: 4.80 against 0.43 per episode under PROPOSAL, 4.44 against 0.40 under MIXED.
- **Escalation**: under PROPOSAL, 1522 decisions scored the K = 128 level. It gave a positive candidate in 41 of them, and K = 512 in 67 more; 1414 decisions stayed. Under MIXED the counts are 37 at K = 128, 44 at K = 512 and 1366 stays, out of 1447 escalations.
- **π mean chunk only**: Δ̂ scored the mean chunk ≤ 0 in 1.47 decisions per episode, and these chunks were executed anyway.

Executed chunks by source:

- **MIXED**: 64.1% of the executed chunks are proposed (7557) and 35.9% retrieved (4237).
  - By task the proposed share ranges from 0.55 (task 3) to 0.76 (task 2).
  - It is 0.693 in successful episodes and 0.638 in failed ones.
  - Of the executed retrieved chunks, 0.656 come from the episode's task, against 0.746 under RETRIEVAL.
- **MATCH almost never executes π's mean chunk**: 19 of 11,696 proposed chunks under PROPOSAL and 1 of 7557 under MIXED.
- **MATCH favours the largest offsets**: scale 1.0 accounts for 63.6% (PROPOSAL) and 65.9% (MIXED) of the executed noisy chunks, although it is about a third of the candidates. Scale 0.5 accounts for 26.5% and 25.9%, scale 0.25 for 9.9% and 8.2%.

Decision latency, mean over decisions in ms, from the per-decision log lines. The round-0 runs had 10 processes per node; the RETRIEVAL row is the 5-episode check on node-61, which shared the node with 12 other processes:

| Source | encode | π | Δ, level K 32 | π + Δ, level K 32 | Δ, level K 128 | Δ, level K 512 | decision ending at K 32 | decision ending at K 512 |
|---|---|---|---|---|---|---|---|---|
| RETRIEVAL | 14.0 | – | 4.4 | – | 16.8 | 72.9 | 20.3 | 112.7 |
| PROPOSAL (node-61) | 13.9 | 2.2 | 4.9 | 7.1 | 17.3 | 75.3 | 21.4 | 117.5 |
| MIXED (node-64) | 14.2 | 2.4 | 9.7 | 12.1 | 36.5 | 154.9 | 28.3 | 224.9 |

π adds 2.2–2.4 ms per decision (median 2.1). Δ on K candidates costs the same whether they are retrieved or proposed. MIXED doubles the candidates of each level, so it doubles the Δ time.

Run-to-run variation of PROPOSAL:

- The replicate on another node reaches 0.142 against 0.120.
- 307 of the 500 episodes are identical in the two runs.
- 64 episodes differ from decision 1 on. In tasks 2 and 9, an earlier episode of the process changed length, which shifted the per-process noise stream of every later episode.
  - In task 2, init 19 took 9 decisions in one run and 30 in the other; inits 20–49 differ from decision 1.
  - In task 9, init 18 took 9 and 13 decisions; inits 19–49 differ from decision 1.
- The remaining differing episodes agree up to some decision and then diverge. The code is deterministic given the observations (Checks), so these divergences start at a rendered image. Tasks 7 and 8, where success is rare, have many of them: 44 and 39 of 50 episodes differ.
- The 13 / 2 discordance comes from task 9 (10 / 2) and task 2 (3 / 0). A paired comparison of PROPOSAL runs thus carries a discordance of this size between two runs of the same command.

Findings:

- **π's candidates lower success under MATCH.** Choosing among π's 32 candidates by Δ̂'s match reaches 0.120 (replicate 0.142), and adding them to the retrieved candidates reaches 0.126. The paired baseline, retrieval with the same Δ, reaches 0.326 (p < 1e-18 for both sources). The two π sources do not differ from each other (40 / 37).
- **No task improves.** The largest losses are in task 5 (0.74 → 0.18) and task 9 (0.96 → 0.46–0.66).
- **The failures move from stuck to motion without completion**. Stuck episodes fall from 80 to 54–62. Nothing moved rises from 60 to 88–98, displaced without lift from 82 to 114–129, and partial joint motion from 27 to 38–42.
- **π's mean chunk alone succeeds in 0.826 of the episodes**, with no stuck episodes. This run and the proposal run differ only in which candidate is executed and in whether a decision can stay.
  - The phase-9 simulator check scored the MATCH choice among π's J = 1 candidates +0.025 above π's mean chunk in the true match of a single chunk.
  - In the closed loop, choosing that way instead of executing the mean chunk lowers success from 0.826 to 0.120 (362 / 9).

## Diagnostics after round 0: small offsets and the stay gate

Two runs that change hyperparameters only, both with the protocol of round 0 (launcher `tmp/runlogs/s3p10/launch_v5b.sh`, main commit `5fec84e`, node-61; summary `tmp/analysis/v5b_summary.py`):

- `v5_proposal_small_dynA`: MATCH among π's candidates with small constant offsets, scales 0.05 / 0.1 / 0.2; rule 32/128/512.
- `v5_proposal_gate_dynA`: π's mean chunk against stay only. K = 1, no escalation, stuck → stay, so the mean chunk is executed only when its predicted match is > 0.

| Task | retrieval, Δ +A | π, MATCH, scales 0.25–1 | π, MATCH, scales 0.05–0.2 | π mean vs stay | π mean, always executed |
|---|---|---|---|---|---|
| 0 | 0.20 | 0.04 | 0.58 | 0.96 | 0.96 |
| 1 | 0.02 | 0.02 | 0.30 | 0.70 | 0.78 |
| 2 | 0.28 | 0.04 | 0.58 | 0.88 | 0.94 |
| 3 | 0.46 | 0.16 | 0.62 | 0.52 | 0.74 |
| 4 | 0.26 | 0.22 | 0.46 | 0.32 | 0.60 |
| 5 | 0.74 | 0.18 | 0.72 | 0.66 | 0.86 |
| 6 | 0.10 | 0.00 | 0.60 | 0.62 | 0.60 |
| 7 | 0.06 | 0.02 | 0.48 | 0.74 | 0.80 |
| 8 | 0.18 | 0.06 | 0.70 | 0.98 | 0.98 |
| 9 | 0.96 | 0.46 | 0.80 | 1.00 | 1.00 |
| **all** | **0.326** | **0.120** | **0.584** | **0.738** | **0.826** |

Paired exact McNemar (first-only / second-only successes):

| Comparison | counts | p |
|---|---|---|
| small offsets vs mean always | 40 / 161 | 2.1e-18 |
| small offsets vs scales 0.25–1 | 245 / 13 | 1.2e-56 |
| mean vs stay, against mean always | 7 / 51 | 2.4e-9 |

Failure categories (success / stuck / other moved / nothing moved / lifted, not at target / displaced / partial joint):

| Run | counts |
|---|---|
| small offsets | 292 / 89 / 15 / 26 / 22 / 39 / 17 |
| mean vs stay | 369 / 103 / 0 / 6 / 14 / 5 / 3 |

Every way of letting m = cos(Δ̂, q_l) decide lowers success relative to executing π's mean chunk:

- **As an argmax among nearby candidates**: success falls from 0.826 to 0.584 with small offsets, and to 0.120 with larger ones.
- **As a gate between the mean chunk and stay**: success falls to 0.738. In 103 episodes the mean chunk's predicted match was ≤ 0, and the policy stayed until the step limit.

## π's mean chunk as the closed-loop default (phase 13)

### Selection

`SelectionStrategy.MEAN` (`--strategy mean`) executes candidate 0 of the proposal source, π's mean chunk, at every decision (`CandidateSelector.FIRST`).

- It never selects stay, so the stay escalation and the stuck rule never apply. `--stay-patience` and `--stuck-action` have no effect, as under RANDOM.
- The match takes no part in the decision: not as the choice, not as a secondary ranking, not as a gate.
- **Requirements**. `ClosedLoopRunner` and `run_closed_loop.py` each refuse a violation with its own message:
  - `--candidate-source proposal`; retrieval and mixed are refused;
  - a single `--escalation` K, since a decision never escalates. `--escalation 1` makes π's mean chunk the only candidate;
  - `--effect-source none` or `predicted`. `observed` is refused, because π's chunks have no observed effect.
- **Effect sources under MEAN**:
  - `none` (`EffectSource.NONE`): no dynamics checkpoint is loaded and Δ never runs. `CandidateSelector.select_unscored` returns candidate 0 with NaN scores. This is the default configuration and the cheapest path: a decision is the encoding and one π forward.
  - `predicted` with `--dynamics-checkpoint`: Δ̂ and the match of the level's candidates are computed and recorded (`best_scores`, `chosen_score`, `mean_selected_score`). The choice is still candidate 0.
  - `none` is refused for MATCH and RANDOM, and `predicted` for RANDOM.
- **Records**:
  - results: `strategy` mean, `effect_source` none or predicted, `stay_decisions`, `escalations` and `forced_moves` 0, `max_candidates` K;
  - every decision's trace: `chosen_source` proposal, `chosen_proposal` 0, `stuck` and `forced` False; under `none`, `chosen_score` and `best_scores` are NaN, and the decision log line reads "unscored";
  - recorded transitions are of kind `rollout_proposal`.
- **Default command**: `run_closed_loop.py --strategy mean --candidate-source proposal --escalation 1 --effect-source none --proposal-checkpoint outputs/runs/goal_proposal/checkpoints/step_003000.pt`, with the benchmark protocol flags.

### Noise streams keyed by decision

`ProposalCandidates.propose(encoded, psi, query, stream)` and `sample(mean, count, stream, device)` take the stream as a `NoiseStream`. The offsets come from `numpy.random.default_rng([seed, *stream.words])`.

- **Closed loop**: `NoiseStream.episode(task, start, init, decision)`, entropy [proposal seed, dataset task_index, start code (0 benchmark, 1 sampled), init index, decision number from 1].
  - A decision's candidates depend only on its observation and this key. The number of decisions of earlier episodes in the process does not shift them.
  - A later escalation level K draws from the same stream, so its candidates are the earlier level's followed by more noisy samples.
- **Offline evaluation** (`scripts/evaluate_proposals.py`): `NoiseStream.position(n)` for the n-th sample, entropy [seed, n].
  - These are the phase-9 streams of the per-call counter. The candidates are bit-identical to the pre-phase-13 code in 12 of 12 calls under each of three candidate settings (`tmp/runlogs/s3p13/check_units.py`).
- **No two keys share a stream**:
  - Every word must be an int in [0, 2³²), so it is one uint32 word of the entropy. A larger int splits into several words: [2³²] and [0, 1] give the same stream.
  - Each kind has a fixed number of words, so distinct keys of a kind are distinct entropy arrays of one length.
  - numpy's `SeedSequence` pads an entropy shorter than its 4-word pool with zeros, so [a, b] and [a, b, 0] give the same stream. The episode entropy has 5 words, so its last word is always mixed in, and it never equals a zero-padded shorter entropy.
  - Measured: 30,000 episode keys (10 tasks × 2 starts × 50 inits × 30 decisions) and 20,000 position keys give 50,000 distinct first draws.

### Checks

Code `b4487f6`, node-61 (L40S), the round-0 protocol (seed 0, 300 steps, 10 settle steps, traced, `--check-start-observation`).

- Outputs: `outputs/closed_loop/p13_checks/`.
- Launchers: `tmp/runlogs/s3p13/launch_checks.sh` and `tmp/runlogs/s3p13/launch_checks_c2.sh`.
- Comparison: `tmp/runlogs/s3p13/compare_checks.py`, output `tmp/runlogs/s3p13/compare_checks.txt`.
- The diagnostic entry `tmp/runlogs/s3p13/diag_run.py` runs `run_closed_loop.py` unchanged. It prints a checksum per call of every observation (both images, proprio, MuJoCo qpos and qvel), encoding, π output and executed chunk.

| Check | Runs | Result |
|---|---|---|
| MEAN with Δ reproduces `v5_proposal_mean_only_dynA`, task 3, inits 0–4, on CUDA 3 as that run | `mean_dyn_t3` | Results identical but `strategy`, `forced_moves` and wall time. Traces identical in every field of every decision and scene, `chosen_score` and `best_scores` included, but `stuck` and `forced`. 3 successes; env steps 80 / 79 / 300 / 79 / 300; decisions 8 / 8 / 30 / 8 / 30. The earlier run marked the 13 decisions whose mean chunk scored ≤ 0 as stuck and forced (patience 0, stuck action best); MEAN marks none. |
| MEAN without Δ executes the same chunks | `mean_nodyn_t3` against `mean_dyn_t3`, both CUDA 3 | All 84 executed chunks and every observation, encoding and π checksum identical. Results and traces identical but the scores, NaN without Δ. Δ ran 84 times with the checkpoint and never without it. |
| PROPOSAL + MATCH, scales 0.05 / 0.1 / 0.2, the same command on CUDA 0 and CUDA 1 | `proposal_small_t3_{a,b}`, task 3, inits 0–4 | All 5 episodes identical: results, traces and checksums. |
| The same, task 7, inits 0–9 | `proposal_small_t7_{a,b}` | 8 of 10 episodes split, at decisions 7–15. In each, the first differing checksum is the wrist image of one observation, while the agent image, proprio, qpos and qvel are bit-identical. Inits 2 and 7 are identical in both runs. |
| The same, scales 0.25 / 0.5 / 1.0 (the phase-10 check setting) | `proposal_t3_{a,b}`, task 3, inits 0–4 | All 5 episodes identical. Phase 10 saw 4 splits in the same comparison, so these render differences occur at random. |
| Recording with MEAN, no Δ, benchmark starts | `mean_rec_t3` against `mean_nodyn_t3`, both CUDA 3 | Results identical. Traces identical but init 2 from decision 21: at decision 20 the agent-view image differs while the wrist image, proprio, qpos and qvel are bit-identical. Render differences also occur between two processes on the same GPU. |
| Recording with MEAN, sampled starts 0–2 | `mean_sampled_rec_t3` against `mean_sampled_t3`, both CUDA 4 | Results and traces identical. |
| Recorded store | `p13_checks/transitions_mean` | Two shards, 127 transitions and 135 frames, all of kind `rollout_proposal`. Decision number, kind and episode success agree with the traces in 8 of 8 episodes. The provenance holds strategy mean, effect source none, escalation [1], code commit `b4487f6` and no dynamics checkpoint. |
| RETRIEVAL + MATCH reproduces `v3_frozen_predicted_rule_dyn_perturb`, task 3, inits 0–4 | `retrieval_t3`, CUDA 3 | Results and traces identical in every field but wall time and the phase-10 fields. |

Unit checks on the CPU (`tmp/runlogs/s3p13/check_units.py`):

- the stream prefix over levels 32 / 128 / 512;
- order independence: the same key gives the same candidates before and after another key;
- MEAN chooses candidate 0 also when every score is −1, where MATCH stays;
- the runner and the script refuse each invalid combination with its message.

### v6: 10 tasks × 50 benchmark inits

Run `outputs/closed_loop/v6_mean_default`, code `b4487f6`, launcher `tmp/runlogs/s3p13/launch_v6.sh`.

- **Placement**: node-61, task t on CUDA t mod 8, as `v5_proposal_mean_only_dynA`. 86–214 s per task process.
- **Settings**: `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`, π `goal_proposal` step 3000.
  - The ψ retrieval space is never queried under the proposal source; no key table.
  - Seed 0, 300 steps, 10 settle steps, traced, `--check-start-observation`.
- **Analysis**: `summarize_closed_loop.py` against v5, `classify_closed_loop_failures.py`, and `tmp/runlogs/s3p13/v6_analysis.py` for the episode-level comparison and latency. Outputs are in `tmp/runlogs/s3p13/v6_*.txt`.

The reference is π's mean chunk run as a phase-10 diagnostic, with Δ scoring it and stuck decisions forced:

| Task | success, v6 | success, v5 π mean only | v6 only / v5 only | decisions, v6 | env steps, v6 |
|---|---|---|---|---|---|
| 0 | 0.96 | 0.96 | 0 / 0 | 13.3 | 129.4 |
| 1 | 0.78 | 0.78 | 0 / 0 | 21.4 | 210.5 |
| 2 | 0.94 | 0.94 | 0 / 0 | 14.4 | 139.3 |
| 3 | 0.74 | 0.74 | 0 / 0 | 14.0 | 137.3 |
| 4 | 0.60 | 0.60 | 0 / 0 | 17.7 | 173.9 |
| 5 | 0.86 | 0.86 | 0 / 0 | 12.0 | 115.6 |
| 6 | 0.62 | 0.60 | 1 / 0 | 17.9 | 176.2 |
| 7 | 0.80 | 0.80 | 1 / 1 | 17.5 | 172.0 |
| 8 | 0.98 | 0.98 | 0 / 0 | 10.5 | 101.2 |
| 9 | 1.00 | 1.00 | 0 / 0 | 8.1 | 77.2 |
| **all** | **0.828** | **0.826** | **2 / 1** | **14.7** | **143.2** |

- **Paired exact McNemar** against v5: 2 / 1 discordant, p = 1.
- **Per episode**, v5 had 14.7 decisions and 143.7 env steps.

Failure categories (`classify_closed_loop_failures.py`):

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| v6 | 414 | 0 | 4 | 8 | 54 | 17 | 3 |
| v5 π mean only | 413 | 0 | 4 | 9 | 55 | 16 | 3 |

Episodes against v5: 456 of 500 are identical in the results and in every trace field but `stuck`, `forced` and the scores. The 44 others:

- **By task**: task 7 29, task 1 4, task 6 3, task 8 3, task 9 3, task 3 2.
- **Where they begin**: all begin after identical recorded scenes.
  - In 37 the first difference is the scene at decision d, between 4 and 30, after identical scenes at every earlier decision.
  - In 2, the gripper commands and executed steps of decision d differ as well.
  - In 5, every decision record is identical and only the final scene, after the last chunk, differs.
  - So the chunk executed from an identical recorded scene differed.
- **Cause**: the code path is bit-reproducible given the observations (the checks above), so these are render differences at an identical state, the mechanism the checksummed splits show.
- **Task 7**: it ran on CUDA 7 in both runs and accounts for 29 of the 44. In phase 10, tasks 7 and 8 also differed most between runs.
- **Success**: 3 episodes change it. Task 6 init 19 and task 7 init 9 succeed only in v6; task 7 init 20 succeeds only in v5.

Latency per decision, in ms, mean (median). Both runs had 10 processes on the 8 GPUs of node-61:

| Run | decisions | decision | encode | π | Δ | execute per env step |
|---|---|---|---|---|---|---|
| v6, no Δ | 7345 | 15.5 (15.1) | 13.5 (13.3) | 1.9 (1.8) | 0 | 16.8 |
| v5 π mean only, Δ on 1 candidate | 7367 | 17.6 (17.3) | 14.0 (13.7) | 1.9 (1.9) | 1.5 (1.4) | 17.1 |

Findings:

- **The default reproduces the diagnostic.** Executing π's mean chunk at every decision reaches 0.828, against 0.826 for the phase-10 diagnostic (2 / 1, p = 1). None of its 86 failures is stuck. 74 moved a goal object or joint without completing the task: 54 lifted but not at the target, 17 displaced without lift, 3 partial joint motion.
- **The differences between the two runs come from render noise**, in 44 episodes, mostly task 7.
- **Cost**: without Δ, a decision costs 15.5 ms, the encoding (13.5 ms) and one π forward (1.9 ms).

## Seed control and the new-v closed loop on the flow ψ versions (phase 28)

Phase 27 ran π's mean chunk on three stage ones with one π training each (seed 0): 0.828 on the fixed stage one, 0.904 on flow (z) and 0.906 on sg[D_ψ] (`docs/stage_one_flow.md`, "Δ and π on the flow ψ versions"). This section trains π with seeds 1 and 2 on each of the three stage ones and runs the same closed loop, to compare the differences between stage ones with the differences between π seeds. It also runs MATCH with the remaining-target velocity query v on sg[D_ψ], next to the phase-23 run on flow (z).

### Runs

Code `44aa302`. Launchers, process logs and analysis outputs are in `tmp/runlogs/s3p28/`, written S below: `env.sh`, `run_one.sh`, `launch_train.sh`, `launch_mean.sh`, `launch_vrem.sh`, `launch_rerun_s0.sh`, `launch_eval.sh`, `analyze_seeds.sh`, `logs/`.

- **π configs**: `configs/proposal/{goal_proposal,goal_proposal_flow,goal_proposal_flow_full_sgd}_seed{1,2}.json` → `outputs/runs/<config name>`. Each is identical to its base config except `seed` and `run_dir`.
- **What the seed sets**: `torch.manual_seed` right before π is built, and the TRAIN sampler. So seed k gives the same initialization and the same batch order on every stage one.
  - At step 0, the MSE and gripper agreement of seed k are equal on the three stage ones: 1.1039 / 0.593 for seed 1 and 1.0419 / 0.625 for seed 2. The gradient norms differ in the third decimal.
- **π's mean chunk in the closed loop**: the v6 protocol with each stage one and its ψ table, as in phase 27.
  - `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`, π step 3000, candidate settings recorded as scales 0.25/0.5/1.0, dct 1, seed 0;
  - env seed 0, 300 steps, 10 settle steps, traced, `--check-start-observation`.
  - Run directories: `outputs/closed_loop/v8_mean_{fixed,flow,flow_full_sgd}_s{1,2}`. Seed 0 is `v6_mean_default`, `v7_mean_flow` and `v7_mean_flow_full_sgd`.
- **Reproduction control**: the three seed-0 π were run again with this code, into `outputs/closed_loop/v8_mean_{fixed,flow,flow_full_sgd}_s0_rerun`.
- **Start observation**: in every new run and every task, the check prints the same block as the seed-0 run on the same stage one.
- **HELDOUT MSE**: `scripts/evaluate_proposals.py curve` at step 3000, outputs `outputs/runs/<π run>/evaluation/curve/`.
- **Analysis**:
  - S `seed_analysis.py` (output `seed_analysis.txt`);
  - `summarize_closed_loop.py` against the seed-0 run (`v8_*_summary.txt`);
  - `classify_closed_loop_failures.py` (`v8_*_failure_classification.txt`).

Placement, all on node-61 (L40S), 2026-09-27:

| Job | CUDA | Time |
|---|---|---|
| MATCH with v on sg[D_ψ], the only job on the node | task t on t mod 8 | 23:15–23:20 |
| π, seeds 1 / 2 | fixed 0 / 1, flow (z) 2 / 3, sg[D_ψ] 4 / 5 | 23:20–23:28 |
| `v8_mean_fixed_s1`, `v8_mean_fixed_s2` | 0, 1, 6, 7 | 23:25–23:29 |
| `v8_mean_flow_s2`, `v8_mean_flow_full_sgd_s2` | 3, 5 | 23:26–23:31 |
| `v8_mean_flow_s1`, `v8_mean_flow_full_sgd_s1` | 2, 4 | 23:28–23:33 |
| HELDOUT MSE, seeds 1 and 2 | fixed 0, flow (z) 1, sg[D_ψ] 6 | 23:29–23:30 |
| seed-0 reruns | fixed 0, 1; flow (z) 2, 3; sg[D_ψ] 4, 5 | 23:35–23:39 |

- **GPU occupancy** (S `logs/gpu_check_*.txt`):
  - From about 23:30, processes of other jobs occupied CUDA 7 (a feature-cache build); from 23:31 they also occupied CUDA 6 (two stage-one trainings). The seed-0 reruns therefore ran on CUDA 0–5, and no job of this section shared a GPU with them.
  - A cap of two GPUs per node took effect at 23:42. Every job above had finished by then, and no GPU job was started afterwards.
- **π training**:
  - 3000 updates took 261–278 s (10.8–11.5 steps/s), with 3.30 GiB peak GPU memory each.
  - Seed 1 on flow (z) and on sg[D_ψ] took 419 and 420 s (7.1 steps/s): host-side batch gathering was slower while six trainings shared the node.
- **Closed-loop task processes**: 95–301 s each.
- **`/tmp/robosuite.log`** on node-61 has the same size, modification time (22:36:09) and checksum before and after these runs (S `robosuite_nnmc61_{before,after}.txt`).

### π's TRAIN and HELDOUT fit

HELDOUT MSE of π's mean chunk at step 3000 (4864 HELDOUT (episode, t)). The seed-0 values are from `outputs/analysis/proposals_goal/curve/` and `outputs/runs/goal_proposal_flow{,_full_sgd}/evaluation/curve/`.

| Stage one | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| fixed | 0.2736 | 0.2648 | 0.2730 | 0.2705 ± 0.0049 |
| flow (z) | 0.2631 | 0.2783 | 0.2657 | 0.2690 ± 0.0081 |
| sg[D_ψ] | 0.2612 | 0.2744 | 0.2655 | 0.2670 ± 0.0067 |

- **TRAIN fit**: the TRAIN batch MSE at the last logged step is 0.0135–0.0156 for all nine π.
- **HELDOUT gripper agreement**: 0.960–0.963.
- **HELDOUT MSE and closed-loop success** (table below), across the nine π: Pearson r = −0.88, Spearman ρ = −0.82.

### π's mean chunk: success per seed

| Stage one | seed 0 | seed 1 | seed 2 | mean | SD over seeds | range |
|---|---|---|---|---|---|---|
| fixed | 0.828 | 0.858 | 0.872 | **0.853** | 0.022 | 0.044 |
| flow (z) | 0.904 | 0.796 | 0.884 | **0.861** | 0.057 | 0.108 |
| sg[D_ψ] | 0.906 | 0.806 | 0.906 | **0.873** | 0.058 | 0.100 |

- **Binomial noise**: the SD of one 500-episode success rate at these rates is 0.015–0.016.
- **Reproduction**: the seed-0 reruns reach 0.828, 0.904 and 0.906 again. They differ from the original runs in 2, 0 and 0 of 500 episodes (fixed: task 7, one each way).
  - For a given π, the protocol reproduces the outcome of each episode up to these 2.
  - The original seed-0 runs used earlier code (`b4487f6`, `76cb2b4`) and other GPUs, so neither code nor placement separates them from seeds 1 and 2.
- **Per episode**: 13.0–14.7 decisions, except seed 1 on flow (z) and on sg[D_ψ], with 15.6 and 15.2.

Success per task (50 episodes per run):

| Task | Text | fixed s0 / s1 / s2 | flow (z) s0 / s1 / s2 | sg[D_ψ] s0 / s1 / s2 | mean: fixed / flow (z) / sg[D_ψ] |
|---|---|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 0.96 / 0.98 / 0.96 | 0.98 / 1.00 / 0.94 | 1.00 / 0.98 / 0.94 | 0.967 / 0.973 / 0.973 |
| 1 | open the top drawer and put the bowl inside | 0.78 / 0.88 / 0.88 | 0.66 / 0.84 / 0.84 | 0.74 / 0.86 / 0.86 | 0.847 / 0.780 / 0.820 |
| 2 | push the plate to the front of the stove | 0.94 / 0.96 / 0.92 | 0.98 / 0.84 / 0.94 | 0.90 / 0.92 / 0.94 | 0.940 / 0.920 / 0.920 |
| 3 | put the bowl on the plate | 0.74 / 0.68 / 0.80 | 0.96 / 0.70 / 0.78 | 0.98 / 0.50 / 0.92 | 0.740 / 0.813 / 0.800 |
| 4 | put the bowl on the stove | 0.60 / 0.94 / 0.84 | 0.98 / 0.38 / 1.00 | 1.00 / 0.34 / 0.92 | 0.793 / 0.787 / 0.753 |
| 5 | put the bowl on top of the cabinet | 0.86 / 0.84 / 0.94 | 0.96 / 0.60 / 0.86 | 0.88 / 0.72 / 0.92 | 0.880 / 0.807 / 0.840 |
| 6 | put the cream cheese in the bowl | 0.62 / 0.62 / 0.80 | 0.72 / 0.84 / 0.82 | 0.80 / 0.92 / 0.72 | 0.680 / 0.793 / 0.813 |
| 7 | put the wine bottle on the rack | 0.80 / 0.78 / 0.72 | 0.90 / 0.86 / 0.78 | 0.84 / 0.94 / 0.86 | 0.767 / 0.847 / 0.880 |
| 8 | put the wine bottle on top of the cabinet | 0.98 / 0.90 / 0.88 | 0.90 / 0.90 / 0.88 | 0.94 / 0.88 / 0.98 | 0.920 / 0.893 / 0.933 |
| 9 | turn on the stove | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.993 / 1.000 / 0.993 |
| **all** | | **0.828 / 0.858 / 0.872** | **0.904 / 0.796 / 0.884** | **0.906 / 0.806 / 0.906** | **0.853 / 0.861 / 0.873** |

Failure categories (`classify_closed_loop_failures.py`; no run has a stuck episode):

| Run | success | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint | lifted, not at target, in tasks 3 and 4 |
|---|---|---|---|---|---|---|---|
| fixed, seed 0 (v6) | 414 | 4 | 8 | 54 | 17 | 3 | 31 |
| fixed, seed 1 | 429 | 8 | 1 | 42 | 20 | 0 | 18 |
| fixed, seed 2 | 436 | 4 | 12 | 26 | 20 | 2 | 17 |
| flow (z), seed 0 | 452 | 5 | 6 | 22 | 13 | 2 | 2 |
| flow (z), seed 1 | 398 | 2 | 0 | 76 | 24 | 0 | 45 |
| flow (z), seed 2 | 442 | 1 | 9 | 28 | 18 | 2 | 11 |
| sg[D_ψ], seed 0 | 453 | 3 | 5 | 16 | 19 | 4 | 0 |
| sg[D_ψ], seed 1 | 403 | 1 | 3 | 82 | 11 | 0 | 57 |
| sg[D_ψ], seed 2 | 453 | 2 | 3 | 25 | 15 | 2 | 8 |

### Paired comparisons

Exact two-sided McNemar by (task, init). "Counts" are the episodes that only the first run solves / only the second run solves.

Between seeds of one stage one:

| Comparison | success | counts, p | tasks 3 and 4: counts, p |
|---|---|---|---|
| fixed: seed 1 vs seed 0 | 0.858 / 0.828 | 63 / 48, 0.18 | 25 / 11, 0.029 |
| fixed: seed 2 vs seed 0 | 0.872 / 0.828 | 69 / 47, 0.051 | 24 / 9, 0.014 |
| fixed: seed 2 vs seed 1 | 0.872 / 0.858 | 52 / 45, 0.54 | 13 / 12, 1 |
| flow (z): seed 1 vs seed 0 | 0.796 / 0.904 | 36 / 90, 1.6e-6 | 2 / 45, 1.6e-11 |
| flow (z): seed 2 vs seed 0 | 0.884 / 0.904 | 42 / 52, 0.35 | 2 / 10, 0.039 |
| flow (z): seed 2 vs seed 1 | 0.884 / 0.796 | 89 / 45, 1.8e-4 | 40 / 5, 7.9e-8 |
| sg[D_ψ]: seed 1 vs seed 0 | 0.806 / 0.906 | 39 / 89, 1.2e-5 | 1 / 58, 2.1e-16 |
| sg[D_ψ]: seed 2 vs seed 0 | 0.906 / 0.906 | 37 / 37, 1 | 1 / 8, 0.039 |
| sg[D_ψ]: seed 2 vs seed 1 | 0.906 / 0.806 | 86 / 36, 6.9e-6 | 51 / 1, 2.4e-14 |

Between stage ones at one seed:

| Comparison | success | counts, p | tasks 3 and 4: counts, p |
|---|---|---|---|
| seed 0: flow (z) vs fixed | 0.904 / 0.828 | 73 / 35, 3.3e-4 | 32 / 2, 6.9e-8 |
| seed 0: sg[D_ψ] vs fixed | 0.906 / 0.828 | 75 / 36, 2.7e-4 | 32 / 0, 4.7e-10 |
| seed 0: sg[D_ψ] vs flow (z) | 0.906 / 0.904 | 37 / 36, 1 | 3 / 1, 0.62 |
| seed 1: flow (z) vs fixed | 0.796 / 0.858 | 49 / 80, 0.008 | 11 / 38, 1.4e-4 |
| seed 1: sg[D_ψ] vs fixed | 0.806 / 0.858 | 48 / 74, 0.023 | 5 / 44, 7.6e-9 |
| seed 1: sg[D_ψ] vs flow (z) | 0.806 / 0.796 | 57 / 52, 0.70 | 12 / 24, 0.065 |
| seed 2: flow (z) vs fixed | 0.884 / 0.872 | 52 / 46, 0.61 | 14 / 7, 0.19 |
| seed 2: sg[D_ψ] vs fixed | 0.906 / 0.872 | 50 / 33, 0.078 | 14 / 4, 0.031 |
| seed 2: sg[D_ψ] vs flow (z) | 0.906 / 0.884 | 47 / 36, 0.27 | 8 / 5, 0.58 |

Discordant episodes, out of 500 (mean over the three pairs):

| Pairs | discordant |
|---|---|
| seeds of fixed | 111, 116, 97 (108) |
| seeds of flow (z) | 126, 94, 134 (118) |
| seeds of sg[D_ψ] | 128, 74, 122 (108) |
| flow (z) vs sg[D_ψ], same seed | 73, 109, 83 (88) |
| fixed vs flow (z), same seed | 108, 129, 98 (112) |
| fixed vs sg[D_ψ], same seed | 111, 122, 83 (105) |

### Stage one against seed over the nine runs

The unit here is one run's success rate over 500 episodes: three stage ones × three seeds.

- **One-way** (stage one as the factor, the seeds as replicates):
  - The SD of the three stage-one means is 0.010; the pooled SD of the seeds within a stage one is 0.049.
  - F(2, 6) = 0.13, p = 0.88.
  - Exact permutation of this F over all 1680 assignments of the nine runs to three groups of three: p = 0.81. The smallest attainable p is 0.0036.
- **Two-way, stage one × seed** (one run per cell; seed k shares π's initialization and batch order across stage ones):
  - Stage one: F(2, 4) = 0.20, p = 0.83.
  - Seed: F(2, 4) = 2.64, p = 0.19.
  - The residual, the stage-one × seed interaction, has mean square 1.5e-3, larger than that of the stage one (3.0e-4).
- **Two-way on the two flow stage ones only** (2 × 3):
  - Seed: F(2, 2) = 130, p = 0.008. Stage one: F(1, 2) = 3.8, p = 0.19.
  - The seed means are 0.905 (seed 0), 0.801 (seed 1) and 0.895 (seed 2).
  - The two flow stage ones share the seed pattern; the fixed stage one does not (its seed 1 is above its seed 0).
- **Exact permutation of the difference of means**:
  - Fixed against the six flow runs (84 assignments): +0.014, one-sided p = 0.31. The smallest attainable p is 0.012.
  - sg[D_ψ] against flow (z) (20 assignments): +0.011, two-sided p = 0.50. The smallest attainable p is 0.10.
- **Bootstrap over seeds** (each stage one's three rates resampled with replacement, 20,000 draws), difference of means with 95% interval:
  - flow (z) − fixed: +0.009 [−0.052, +0.059];
  - sg[D_ψ] − fixed: +0.020 [−0.047, +0.068];
  - sg[D_ψ] − flow (z): +0.011 [−0.058, +0.081].
- **Tasks 3 and 4** (100 episodes per run):
  - Success: fixed 0.67 / 0.81 / 0.82, flow (z) 0.97 / 0.54 / 0.89, sg[D_ψ] 0.99 / 0.42 / 0.92.
  - The pooled seed SD within a stage one is 0.23, against 0.017 between the stage-one means; F(2, 6) = 0.02.
- **The other eight tasks** (400 episodes per run):
  - Success: fixed 0.868 / 0.870 / 0.885, flow (z) 0.887 / 0.860 / 0.882, sg[D_ψ] 0.885 / 0.902 / 0.902.
  - F(2, 6) = 3.37, p = 0.10; the permutation p is 0.13.
  - Bootstrap: sg[D_ψ] − fixed +0.023 [+0.011, +0.034], sg[D_ψ] − flow (z) +0.020 [+0.005, +0.037].
  - This split was chosen after seeing where the seed variance is.

### Reading: stage one against seed

- **The differences between stage ones do not exceed those between seeds.**
  - Averaged over three seeds, π's mean chunk succeeds in 0.853 (fixed), 0.861 (flow (z)) and 0.873 (sg[D_ψ]) of the episodes. The seeds of one stage one span 0.044–0.108.
  - The between-stage-one SD (0.010) is a fifth of the within-stage-one seed SD (0.049). Over all tasks, every seed-level test gives the stage-one effect p ≥ 0.19, and every bootstrap interval contains 0.
- **The phase-27 gain is not reproduced.**
  - The comparison 0.828 against 0.904 / 0.906 (73 / 35 and 75 / 36) pairs the lowest of the three fixed-ψ seeds with the highest or tied-highest flow seeds.
  - At seed 1 the fixed stage one is ahead of both flow stage ones: 80 / 49 (p = 0.008) and 74 / 48 (p = 0.023).
  - At seed 2 no stage-one difference over all tasks reaches p < 0.05.
- **The seed-to-seed differences are differences between the trained π**, not between runs. The same π reproduces up to 2 of 500 episodes. Seeds of one stage one differ in 74–134 episodes, with p down to 1.6e-6.
- **Where the seed matters: the bowl-placing tasks 3 and 4.**
  - There the seed SD is 0.23 against 0.012 over the other tasks.
  - The failure that phase 27 found only on the fixed stage one, the bowl lifted but not placed, appears on every stage one, depending on the seed. It occurs in tasks 3 and 4 in 31 episodes with fixed seed 0, and in 45 and 57 with seed 1 on flow (z) and sg[D_ψ].
  - Seed 1 gives π the same initialization and batch order on every stage one; it fails these tasks on both flow stage ones and not on the fixed one.
- **HELDOUT MSE ranks the nine π much as the closed loop does** (ρ = −0.82). The two seed-1 flow π have the highest HELDOUT MSE (0.278, 0.274).
- **What three seeds per stage one can establish**:
  - the seed-0 ranking of the stage ones does not hold at seeds 1 and 2;
  - the seed variance of π (SD 0.02–0.06 in success rate) is larger than the differences between the stage-one means and larger than the binomial noise of one run (0.015);
  - the seed differences are π differences.
- **What they cannot establish**:
  - They cannot exclude a stage-one effect smaller than the seed spread. With three runs per stage one and a seed SD of 0.049, a two-sample t-test at 5% needs a difference of about 0.11 (2.78 × 0.049 × √(2/3)).
  - An exact permutation test with three runs per arm cannot reach a two-sided p below 0.10.
  - The rate at which a seed produces a π that fails tasks 3 and 4, as seed 1 does on the flow stage ones, is estimated from three draws per stage one.
  - The advantage of sg[D_ψ] on the other eight tasks is in a subset chosen after the fact, and the seed-level tests on that subset give p = 0.10–0.20.

### MATCH with the remaining-target v on sg[D_ψ]

Run `outputs/closed_loop/v8_proposal_small_vrem_flow_full_sgd`, launcher S `launch_vrem.sh`. It is the configuration of the phase-23 flow (z) run `v7_proposal_small_vrem_flow` (`docs/velocity_query.md`, "Closed loop with v(o, l)"), with the sg[D_ψ] stage one and its heads:

- **Candidates**: MATCH among π's candidates. π is `goal_proposal_flow_full_sgd` step 3000 (seed 0), with scales 0.05/0.1/0.2, dct 1, seed 0 and no gripper-flipped candidate.
- **Rule**: 32/128/512, patience 2, stuck → stay.
- **Δ**: `goal_dynamics_perturb_flow_full_sgd` step 3000 (+A).
- **Match target**: `--match-target velocity` with `goal_velocity_remaining_flow_full_sgd` step 3000, the remaining target.
- **Retrieval**: `--retrieval-space frozen` with the frozen key table, which the proposal source never queries.
- **Stage one**: `goal_instruction_contrast_flow_full_sigall_sgd` step 5000 with its ψ table.
- **Protocol**: seed 0, 300 steps, 10 settle steps, traced, `--check-start-observation`.
  - The start-observation block equals that of `v7_mean_flow_full_sgd` in all 10 tasks.
- **Code**: the phase-23 run used a snapshot of `bd492d6` plus the phase-25 velocity-target patch. This run uses `44aa302`, which contains both.
- **Analysis**:
  - S `vrem_analysis.py`, output `vrem_analysis.txt`. It has the tables of phase 23's `analyze_runs.py` and `first_decisions.py`, with the latency parsed from each run's process logs.
  - `classify_closed_loop_failures.py`, output `v8_proposal_small_vrem_flow_full_sgd_failure_classification.txt`.

Success; the pair counts are MATCH only / π mean only on sg[D_ψ]:

| Task | π mean, sg[D_ψ] | MATCH, v, sg[D_ψ] | counts, p | π mean, flow (z) | MATCH, v, flow (z) (phase 23) |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.94 | 0 / 3, 0.25 | 0.98 | 0.40 |
| 1 | 0.74 | 0.44 | 5 / 20, 0.0041 | 0.66 | 0.20 |
| 2 | 0.90 | 0.72 | 3 / 12, 0.035 | 0.98 | 0.16 |
| 3 | 0.98 | 0.88 | 1 / 6, 0.12 | 0.96 | 0.74 |
| 4 | 1.00 | 0.82 | 0 / 9, 0.0039 | 0.98 | 0.70 |
| 5 | 0.88 | 0.50 | 3 / 22, 1.6e-4 | 0.96 | 0.50 |
| 6 | 0.80 | 0.72 | 6 / 10, 0.45 | 0.72 | 0.44 |
| 7 | 0.84 | 0.66 | 6 / 15, 0.078 | 0.90 | 0.24 |
| 8 | 0.94 | 0.56 | 2 / 21, 6.6e-5 | 0.90 | 0.68 |
| 9 | 0.98 | 0.96 | 1 / 2, 1 | 1.00 | 0.98 |
| **all** | **0.906** | **0.720** | **27 / 120, 3.4e-15** | **0.904** | **0.504** |

| Comparison | counts | p |
|---|---|---|
| MATCH, v, sg[D_ψ] vs π mean, sg[D_ψ] | 27 / 120 | 3.4e-15 |
| MATCH, v, flow (z) vs π mean, flow (z) (phase 23) | 20 / 220 | 9.1e-44 |
| MATCH, v, sg[D_ψ] vs MATCH, v, flow (z) | 170 / 62 | 8.2e-13 |
| MATCH, v, sg[D_ψ] vs π mean, fixed (v6) | 57 / 111 | 3.8e-5 |

Failure categories:

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| π mean, sg[D_ψ] | 453 | 0 | 3 | 5 | 16 | 19 | 4 |
| MATCH, v, sg[D_ψ] | 360 | 30 | 6 | 26 | 31 | 35 | 12 |
| π mean, flow (z) | 452 | 0 | 5 | 6 | 22 | 13 | 2 |
| MATCH, v, flow (z) | 252 | 81 | 40 | 43 | 37 | 37 | 10 |

- **Stuck** episodes of MATCH on sg[D_ψ] are in tasks 1–6 and 8, at most 6 per task.
- **"Nothing moved"**: 11 of its 26 such episodes are in task 8.
- **Partial joint motion**: 10 of its 12 are in task 1.

Executed chunks and per-episode means:

| Run | decisions | π's mean chunk | noisy, scale 0.05 / 0.1 / 0.2 | stay | first-level best ≤ 0 | per episode: decisions, env steps, stays, escalations | episodes reaching K 512 |
|---|---|---|---|---|---|---|---|
| MATCH, v, sg[D_ψ] | 8115 | 0 | 0.021 / 0.158 / 0.711 | 0.110 | 0.117 | 16.2, 159.2, 1.79, 1.26 | 0.130 |
| MATCH, v, flow (z) | 10038 | 0 | 0.007 / 0.093 / 0.636 | 0.265 | 0.274 | 20.1, 198.6, 5.31, 4.48 | 0.280 |

- **Stays by outcome**: MATCH on sg[D_ψ] stays in 2.0% of the decisions of its successful episodes and in 19.4% of those of its failed ones. On flow (z) the shares are 4.4% and 34.2%.
- **Largest offset**: 80% of its non-stay chunks carry the largest offset (scale 0.2).
- **Decision 1**: every candidate scores ≤ 0 in 15 of the 500 episodes (tasks 3, 4 and 5); the median best score is +0.72, against +0.53 on flow (z).
- **‖v‖**: median 0.255 (p10 0.115, p90 0.404); 0.259 on flow (z).

Decision latency in ms, mean (median):

| Run | decisions | decision | encode | v | π | Δ, all levels | decision, one level | decision, to K 512 | execute per env step |
|---|---|---|---|---|---|---|---|---|---|
| MATCH, v, sg[D_ψ] | 8115 | 30.4 (22.5) | 14.1 (13.8) | 1.7 (1.7) | 2.1 (2.0) | 11.8 (4.7) | 23.3 (22.4) | 118.0 (117.5), 605 decisions | 16.1 |
| π mean, sg[D_ψ] (phase 27) | 6482 | 17.9 (16.7) | 15.6 (14.6) | – | 2.3 (2.0) | – | 17.9 (16.7) | – | 17.0 |
| MATCH, v, flow (z) (phase 23) | 10038 | 55.5 (26.7) | 16.4 (14.8) | 2.0 (1.8) | 2.7 (2.3) | 32.5 (6.0) | 28.3 (25.1) | 151.6 (144.7), 2203 decisions | 17.1 |

Each run had 10 processes, placed as follows:

- this run: on 8 GPUs of an otherwise idle node-61;
- the phase-27 π-mean run: on 4 GPUs of node-61;
- the phase-23 run: on 5 GPUs of node-62, shared with other users' processes.

### Reading: MATCH with v on sg[D_ψ]

- **Below π's mean chunk on the same stage one**: 0.720 against 0.906 (27 / 120, p = 3.4e-15). It is behind in every task, significantly (p < 0.05) in tasks 1, 2, 4, 5 and 8.
  - The gap (−0.186) is larger than the spread of π's mean chunk over the three π seeds on this stage one (0.806–0.906).
  - The MATCH run draws its candidates from the seed-0 π, whose mean chunk reaches 0.906.
- **Above the same configuration on flow (z)**: 0.720 against 0.504 (170 / 62, p = 8.2e-13), with
  - fewer stays: 11.0% of the decisions against 26.5%;
  - fewer stuck episodes: 30 against 81;
  - fewer episodes reaching K = 512: 0.13 against 0.28;
  - a higher best score at decision 1: +0.72 against +0.53.
- **Unchanged from flow (z)**: MATCH never executes π's mean chunk (0 of 8115 decisions), and 80% of its non-stay chunks carry the largest offset.
- **Cost**: 30.4 ms per decision against 17.9 ms for π's mean chunk. v takes 1.7 ms, and the escalations to K = 512 (7.5% of the decisions) take 118 ms each.

# Joint ψ / Δ training on real transitions

ψ (`StateTrunk`) and Δ (`DynamicsHead`) are trained together from scratch on real transitions (o, a, o′) of TransitionStores. Every frame is a simulator render; the feature cache is read only for its proprio statistics. Neither module's architecture changes (width 512, depth 4, ψ 256-d, the Δ of stage two). The same trainer runs the control R6, where ψ is a fixed PCA projection of the frozen DINOv2 pooled features and only Δ is trained ("Fixed encoder (R6)").

| Class | File |
|---|---|
| `HeldoutRule`, `InteractionStore`, `InteractionBatch`, `InteractionSampler` | `ceam/data/interaction_sampler.py` |
| `InteractionWeights`, `InteractionLossReport`, `InteractionObjective` | `ceam/losses/interaction_objective.py` |
| `InteractionSourceConfig`, `MonitorConfig`, `FixedEncoderConfig`, `InteractionConfig` | `ceam/training/interaction_config.py` |
| `CriteriaValues`, `MonitorDiagnostics`, `MonitorMeasurement`, `ConflictCriteria`, `ScaleValues`, `ScaleMeasurement`, `ScaleCollapseCriterion`, `CriteriaStopRule`, `TrajectoryPairs`, `InteractionMonitor` | `ceam/training/interaction_monitor.py` |
| `InteractionCheckpointStore`, `LoadedInteraction`, `LoadedFixedEncoderInteraction` | `ceam/training/interaction_checkpoint_store.py` |
| `InteractionTrainer`, `InteractionStates`, `InteractionMetrics`, `SourceShare`, `InteractionSummary` | `ceam/training/interaction_trainer.py` |
| `FittedSource`, `FeaturePca` | `ceam/data/feature_pca.py` |
| `FrozenPcaEncoder` | `ceam/model/frozen_pca_encoder.py` |

Entry: `scripts/train_interaction.py --config <json> --device cuda:N`; `scripts/fit_feature_pca.py --config <json> --out <npz> --code-commit <sha>` fits the PCA of R6. Checks:

- `scripts/check_interaction_training.py --device cuda:N --scratch <new dir> [--checkpoint <joint checkpoint>]`;
- `scripts/check_scale_criterion.py [--config <json> --device cuda:N [--pca <npz>]]` (K4);
- `scripts/check_frozen_pca_encoder.py --pca <npz> [--pca-repeat <npz>] --config <json> --device cuda:N [--checkpoint <fixed-encoder checkpoint>] [--joint-checkpoint <checkpoint>] [--scratch <dir>]` (R6);
- `scripts/compare_interaction_runs.py --a <run dir> --b <run dir>`: every logged value and every checkpoint tensor of two runs over their common steps, timings excluded.

## Objective

With Δ̂ = Δ(o, ψ(o), a) and z = ψ(o′) − ψ(o), `InteractionObjective` has exactly three terms:

- L_pred = mean over the batch of ‖Δ̂ − z‖², the squared norm summed over the 256 state dimensions. This is 256 times the per-element mean squared error of stage two (`InteractionMetrics.prediction_per_dim` logs L_pred / 256).
- SIGReg_ψ = `SigRegLoss` on the 2B states [ψ(o); ψ(o′)].
- SIGReg_Δ = `SigRegLoss` on the B effects Δ̂.
- total = w_pred · L_pred + w_ψ · SIGReg_ψ + w_Δ · SIGReg_Δ, weights 1, 1, 1 by default (`weights` in the config).
- Per-dim L_pred: w_pred = 1/256 (0.00390625, exact in binary floating point) makes the prediction term the mean over the 256 dimensions, ‖Δ̂ − z‖² / 256. `joint_all.json` and `frozen_pca_delta.json` use it. The logged `prediction` stays the unweighted sum over dimensions and `prediction_per_dim` the mean; `total` includes the weight. The objective code is unchanged; every other config keeps its weights (`smoke.json`: 1, 1, 1).

No tensor is detached and there is no EMA: L_pred reaches ψ through ψ(o′), through ψ(o) in the target, and through ψ(o) as Δ's input. The objective casts its inputs to float32 and runs with autocast disabled; `SigRegLoss` draws fresh directions from torch's RNG, first for the states, then for the effects. `forward` returns (total, report) with the three unweighted terms detached.

## Sampler and split

`InteractionBatch` has four fields, `current` and `future` (`FrameBatch`: float16 tokens [B, 2, 64, 768], proprio [B, 9] standardized with the feature cache's statistics as `TransitionStore.proprio`), `actions` ([B, 10, 7], `ActionCache.standardize`) and `source` ([B] int64, the store's position in the config), and two optional history fields, `history` (the previous chunk) and `previous_state` (ψ of the previous frame), which are None unless the sampler is given previous chunks or previous states (`docs/r_short_history.md`). The sampler keeps, per store, only o / o′ frame indices, standardized chunks and kind labels; task_id, episode, t, goal_after and episode_success are read only through `HeldoutRule` to form the split. `kinds(pooled)` returns kind labels for evaluation grouping (stay chunks); they are not in the batch.

Sampling: the pool is the concatenation of every store's transitions in the split; each batch draws B pooled indices independently and uniformly (with replacement) with `np.random.default_rng(seed)`, so a store contributes in proportion to its size. `gather(pooled)` reads given indices without touching the generator.

`HeldoutRule(episodes, shard_prefixes)`: a transition is held out iff its `episode` lies in the inclusive range or its shard name starts with one of the prefixes. A prefix that matches no shard raises `ValueError`.

| Store | `episode` field | Held-out rule | TRAIN / HELDOUT |
|---|---|---|---|
| `demo_perturbations` (data A) | dataset episode id (verified: all 390 episodes are TRAIN episodes of the stage-one split, heldout fraction 0.1, seed 0) | none | 31,200 / 0 |
| `rollouts_r0` | sampled start index 0–99 within each task's shard (verified: `ClosedLoopRunner` passes the init index to `begin_episode`; 100 values per shard) | episodes 80–99 | 18,732 / 4,470 (533 `rollout_stay`) |
| `demo_renders` | dataset episode id; shards `train_taskNN` / `heldout_taskNN` in the writer `scripts/render_demo_episodes.py`; the store is not built yet, and the rule raises `ValueError` if no shard name starts with `heldout` | shard prefix `heldout` | — |
| Phase-26 re-simulation | the rule selects initial states 40–49 only if the writer stores the per-task init index in `episode` | episodes 40–49 | — |

In `rollouts_r0` consecutive decisions of one episode share frames: all 22,202 (t, t + 1) decision pairs have future frame = next current frame, so the transitions chain into 10-step trajectories.

## Config

`InteractionConfig` JSON has exactly its dataclass fields at every level (unknown or missing keys raise `ValueError`), except `fixed_encoder`, `history` and `previous_state`, which may be absent (meaning none) and are written only when set; paths may use `$CEAM_ROOT` / `$CEAM_DATA_ROOT`. A source is `{store_dir, heldout: {episodes: [first, last] | null, shard_prefixes: [...]}}`; the monitor is `{first_step, samples, seed, stay_kinds, trajectory_store_dir, trajectory_heldout}`; the fixed encoder is `{pca_file}`, the history `{previous_chunk_dir}`, the previous state `{previous_state_dir}` (`docs/r_short_history.md`). Configs and checkpoints written before `fixed_encoder` existed read unchanged, and their `to_json_text` is byte-identical to the files.

| Config | Sources | Optimizer | Monitor | Run dir |
|---|---|---|---|---|
| `configs/interaction/joint_all.json` (template) | demo_renders (prefix `heldout`), demo_perturbations (none), rollouts_r0 (80–99) | weights 1/256, 1, 1; AdamW lr 3e-4, wd 0.05, warmup 500, cosine to 0 at 20,000; B 256; clip 1.0; bf16 autocast; checkpoint every 2,000 | from step 4,000, 4,096 samples, seed 0, stay kinds `stay`, `rollout_stay`; K3 on demo_renders HELDOUT | `outputs/runs/interaction_joint_all_seed0` |
| `configs/interaction/frozen_pca_delta.json` (R6 template) | as joint_all; fixed encoder `outputs/feature_pca/provisional_joint_all_3stores/pca.npz` | weights 1/256, 0, 0; otherwise as joint_all | as joint_all | `outputs/runs/interaction_frozen_pca_delta_seed0` |
| `configs/interaction/frozen_pca_delta_all_stores_prev_chunk.json`, `frozen_pca_delta_all_stores_prev_state.json` | R6_all (`frozen_pca_delta_all_stores.json`) with a short history in the effect head (`docs/r_short_history.md`) | as R6_all | as R6_all | `outputs/runs/interaction_frozen_pca_delta_all_stores_prev_{chunk,state}_seed0` |
| `configs/interaction/smoke.json` | demo_perturbations (none), rollouts_r0 (80–99) | as above, but warmup 100 and 500 steps; checkpoint every 100 | from step 200; K3 on rollouts_r0 episodes 80–99 | `outputs/runs/interaction_smoke` |
| `configs/interaction/smoke_prediction_per_dim.json` | as smoke | as smoke | as smoke | `outputs/runs/interaction_smoke_prediction_per_dim` |

`smoke_prediction_per_dim.json` differs from `smoke.json` only in `weights.prediction` = 1/256, so its prediction term is the per-element mean squared error. `smoke.json` keeps the sum over dimensions.

The store lists of `joint_all.json` and `frozen_pca_delta.json` are the stores that exist before the random training collection; the final lists, and the PCA fitted on the final list, are set when the runs are launched. The trainer refuses a PCA fitted on other stores ("Fixed encoder (R6)").

## Training step and outputs

1. Draw B TRAIN transitions.
2. `InteractionTrainer.forward`: the trunk runs once on the [B, 2] frames (o, o′), then Δ̂ = dynamics(o, ψ(o), a), both under bf16 autocast when `autocast_bf16`.
3. `InteractionObjective` in float32; backward; global gradient-norm clipping; AdamW over both modules (weight decay only on ≥ 2-D weights whose name contains neither "token" nor "embedding"); linear warmup, cosine decay to 0 at `total_steps`.

Initialization: `torch.manual_seed(seed)`, then the trunk (with a fixed encoder: the encoder, which draws no random numbers), then the dynamics head (its output layer is zero, so Δ̂ = 0 at step 0).

Run directory:

- `config.json`;
- `metrics.jsonl`: one `InteractionMetrics` line every `log_every` updates and at every update followed by a checkpoint or a K4 evaluation: lr, total and the three terms, L_pred / 256, batch mean cos(Δ̂, z), median ‖Δ̂ − z‖/‖z‖, median ‖z‖ and ‖Δ̂‖, mean per-dimension std of [ψ(o); ψ(o′)], gradient norm (global, trunk, Δ) before clipping, steps/s and sampling time excluding checkpoint and monitor time, peak memory, per-source count and L_pred;
- `checkpoints/step_{N:06d}.pt`: trunk and dynamics state dicts, the config JSON text, the step; written every `checkpoint_every` updates and after the last; an existing file is refused;
- `monitor.jsonl`: one `MonitorMeasurement` per checkpoint;
- `scale.jsonl`: one `ScaleMeasurement` per K4 evaluation, the step-0 reference first;
- `monitor_stop.md`: written when the monitor (K0–K4) stops training.

Exit status of `scripts/train_interaction.py`: 0 when every step ran, 3 when the monitor stopped training by any of K0–K4 (after writing the checkpoint and `monitor_stop.md`), 1 on an exception.

## Monitor: conflict and collapse criteria

The criteria are those fixed in plan.md ("冲突与坍缩判据"); the thresholds are constants of `ConflictCriteria`.

**Evaluation data**, fixed at construction from `monitor.seed`:

- held-out set: `samples` (4,096) pooled indices of the HELDOUT split of the training sources, without replacement (`default_rng([seed, 0])`); stay = kind in `stay_kinds`;
- K3 pairs: of the trajectory store's held-out transitions, min(4,096, M) drawn without replacement (`default_rng([seed, 1])`) give (o_t, o_{t+10}); 4,096 pairs (i, j) of held-out current frames from different episodes, episode = (shard, episode), drawn with `default_rng([seed, 2])`, give the denominator;
- R0: the frozen DINOv2 pooled feature of a frame is, per view, the float32 mean of its 64 tokens, the two views concatenated (1,536-d).

Both modules run in eval mode, float32, without autocast or gradient.

**Criteria**, with z = ψ(o′) − ψ(o) and Δ̂ on the held-out set:

| Criterion | Holds at a checkpoint when |
|---|---|
| K0 collapse | effective rank of ψ(o) or of Δ̂ ≤ 2, or the largest covariance eigenvalue share of either ≥ 0.9 |
| K1 inflated effects | median ‖Δ̂‖/‖z‖ > 1.5, or median ‖Δ̂‖ over stay transitions > median ‖z‖ over the other transitions |
| K2 prediction against SIGReg_Δ | R² = 1 − Σ‖Δ̂ − z‖² / Σ‖z − z̄‖² < 0.3 and SIGReg(Δ̂) > 10 × 1.06 / N |
| K3 temporal roughness | K3 ratio = E‖ψ(o_{t+10}) − ψ(o_t)‖² / E‖ψ(o_i) − ψ(o_j)‖² > 0.25 and > 2 × the same ratio on R0 |

- Effective rank = exp(entropy of the normalized covariance eigenvalues), the definition of `StateDistributionMetrics`.
- SIGReg(Δ̂): `SigRegLoss(256)` on the CPU in float32 with directions seeded 0 in a forked RNG, as `StateDistributionMetrics`; N = 4,096, so the floor is 2.588e-4.
- Without stay transitions in the held-out set, the stay half of K1 does not hold.

**Stop rule** (`CriteriaStopRule`): every checkpoint is measured and logged. Only checkpoints of step ≥ `first_step` (4,000 in `joint_all.json`) count. K0 stops at the first counted checkpoint where it holds; K1, K2 and K3 each stop when the same criterion holds at two consecutive counted checkpoints. On a stop the trainer writes `monitor_stop.md` (criteria met, every criterion value at this and the previous counted checkpoint with thresholds, the diagnostics) and ends the loop. K4 has its own schedule and rule ("K4, scale collapse"), and a stop by either writes the same report.

**Diagnostics logged at every checkpoint** (`MonitorDiagnostics`):

- Var(z)/Var(ψ) = tr Cov(z) / tr Cov(ψ(o));
- correlation of ⟨ψ(o), u_k⟩ and ⟨ψ(o′), u_k⟩ along each principal direction u_k of Cov(ψ(o)) (all 256 values, eigenvalue-weighted mean, min, 10th percentile, median);
- ‖Δ̂‖ and ‖z‖ percentiles 10/25/50/75/90 on stay transitions;
- the K3 ratio and the R0 ratio;
- median adjacent-frame cosine cos(ψ(o), ψ(o′)), raw and centered, on the held-out set and on the K3 pairs;
- mean cos(Δ̂, z), median ‖Δ̂ − z‖/‖z‖, median ‖z‖ and ‖Δ̂‖;
- every criterion value and flag.

### K4, scale collapse

K0–K3 do not change when ψ and Δ̂ are multiplied by one positive constant, so they cannot see a run that shrinks everything. K4 (plan.md, "K4 尺度坍缩") measures the scale itself, on the same 4,096 held-out transitions as K0–K3 (`ScaleCollapseCriterion`, float64):

| Quantity | Definition | Reference |
|---|---|---|
| σ_ψ | √(mean over the 256 dims of the variance of ψ(o)), divisor N − 1 | step 0, measured before the first update |
| mean ‖Δ̂‖ | mean of ‖Δ(o, ψ(o), a)‖ | step 500 (Δ̂ is identically 0 at step 0: the dynamics output layer is zero) |

- **K4a**: σ_ψ < 0.1 × σ_ψ(step 0). **K4b**: mean ‖Δ̂‖ < 0.1 × mean ‖Δ̂‖(step 500); it cannot hold before step 500, nor when the step-500 value is 0.
- **Schedule**: after every update n with n a multiple of 500 (n ≥ 500), and after every checkpointed update. It does not depend on `first_step`, which applies to K0–K3 only.
- **Rule**: a single evaluation where K4a or K4b holds stops training. If that update is not a scheduled checkpoint, it becomes the last one: its checkpoint is written and measured by K0–K3 as usual. The trainer then writes `monitor_stop.md` (criteria met, K0–K3 values, a K4 table with both references and ratios, the diagnostics) and exits with status 3.
- **Cost**: an evaluation computes only ψ(o) and Δ̂ of the held-out set and their norms and variances: 1.7–3.4 s on an L40S in the final smoke runs of Phase 9, against 2.9–9.3 s for K0–K3 at the same checkpoints. The CUDA blocks cached by the step-0 evaluation are released before the training loop, so the logged peak memory does not depend on it.
- **Logged at every evaluation** (`scale.jsonl`, and a `scale:` line in the log): step, whether the update is a scheduled checkpoint, σ_ψ, mean ‖Δ̂‖, both references, both ratios, K4a, K4b, the stop.

## Fixed encoder (R6)

R6 (plan.md, "对照组") trains Δ on a fixed ψ: the frozen DINOv2 pooled features reduced to 256 dims by a fixed PCA.

**Encoder** (`FrozenPcaEncoder`): ψ(o) = (pooled(o) − mean) @ V₂₅₆, where pooled(o) is, per view, the float32 mean of the frame's 64 frozen tokens, the two views concatenated (1,536-d, the feature of R0). mean [1,536] and V₂₅₆ [1,536, 256] are float32 buffers, so the module has no parameters. It reads no proprio, runs in float32 with autocast disabled and without gradient, and accepts any leading dims. The projection is not whitened.

**PCA** (`scripts/fit_feature_pca.py` → `FeaturePca` .npz):

- Frames: the TRAIN frames of every source of a given interaction config, i.e. per store the distinct current and future frames of the transitions its held-out rule does not hold out; each frame once, the stores pooled.
- Method: exact. Mean and covariance (divisor N − 1) in float64; `numpy.linalg.eigh`; eigenvalues descending, clipped at 0; each eigenvector signed so that its largest-magnitude entry is positive; the first 256 kept.
- File: the mean, the 256 components, all 1,536 eigenvalues and explained-variance ratios (float64), the fitted sources (store, held-out rule, shard names, TRAIN frame count) and the provenance (config path and text, commit, host, time, self-check).
- Self-check printed by the script on the fitting frames: variance of each projected coordinate against its eigenvalue, largest projected covariance off the diagonal, orthonormality, eigen-equation residual.

**Trainer** (`fixed_encoder: {pca_file}` in the config):

- Before creating the run directory, the trainer loads the PCA and requires that it was fitted on exactly the config's sources: the same store directories in the same order, the same held-out rules, and the store's current shard names. A PCA fitted on other stores, or on a store that has since gained shards, raises `ValueError`. The PCA must also match the trunk's shape fields (1,536 = n_views × visual_token_dim features, state_dim components).
- ψ(o), ψ(o′) come from the encoder; only the `DynamicsHead` is optimized (AdamW over its parameters; `trunk_grad_norm` is logged as 0).
- Objective: `frozen_pca_delta.json` sets the weights to 1/256, 0, 0, so the loss is L_pred per dim alone. Both SIGReg statistics are still computed and logged; with weight 0 they add nothing to the loss or the gradient.
- The monitor is unchanged: K0–K3 on the fixed ψ and the trained Δ̂, K4 as above. σ_ψ of a fixed encoder never changes, so K4a cannot hold; K4b applies to Δ̂.
- Checkpoint: the encoder's buffers are stored under `fixed_encoder` instead of `trunk`, so a checkpoint holds the projection it was trained with. `InteractionCheckpointStore.load_fixed_encoder` loads it (encoder, dynamics, config, step); `load` refuses it, and `load_fixed_encoder` refuses a trainable-trunk checkpoint.
- Evaluation: `FrozenPcaDeltaModel` (`--model NAME frozen_pca CHECKPOINT.pt` of `scripts/evaluate_representation.py`, docs/interaction_representation.md).

## Verification

`scripts/check_interaction_training.py` with the code of commit `9e0ca04` (run from the working tree just before that commit), node-69 CUDA 0, log `tmp/runlogs/s4p2/check_c.log`; `--checkpoint outputs/runs/interaction_smoke/checkpoints/step_000300.pt`.

**1. Objective against a direct re-implementation.** The direct version sums per-sample squared norms in a Python loop and writes the Epps–Pulley statistic out itself (same RNG order).

| Weights, B | total diff | L_pred diff | SIGReg_ψ diff | SIGReg_Δ diff | max gradient diff (max gradient) |
|---|---|---|---|---|---|
| 1, 1, 1; 256 | 2.9e-6 (of 11.51) | 2.9e-6 | 4.7e-10 | 0 | 4.7e-10 (7.6e-3) |
| 1, 1, 1; 7 | 0 | 0 | 0 | 0 | 7.5e-9 (0.27) |
| 0.5, 2, 3; 256 | 1.9e-6 (of 6.45) | 2.9e-6 | 4.7e-10 | 0 | 2.3e-10 (3.8e-3) |
| 0.5, 2, 3; 7 | 0 | 0 | 0 | 0 | 7.5e-9 (0.13) |

bf16 inputs, with or without autocast, give the float32 total of the same values exactly (11.423368), as float32 scalars.

**2. Gradient paths into ψ.** Trunk gradient of L_pred through `InteractionTrainer.forward` on 8 random frames, and through each path alone with the other two detached:

| Setting | ‖g(full)‖ | via ψ(o′) | via ψ(o) in the target | via ψ(o) as Δ input | ‖g(full) − Σ‖ / ‖g(full)‖ |
|---|---|---|---|---|---|
| zero-initialized effect head, float32 | 852.0 | 606.7 | 580.2 | 0 | 3.9e-7 |
| random effect head, float32 | 916.2 | 798.2 | 766.4 | 26.9 | 6.0e-7 |
| random effect head, bf16 autocast | 915.6 | 797.9 | 766.0 | 26.9 | 4.5e-3 |

The Δ-input path is exactly 0 at initialization because the effect head's weights are zero; it is open after the first update. SIGReg_ψ alone gives trunk gradient 3.0 and dynamics gradient 0; SIGReg_Δ alone gives 0.17 and 2.27.

**3. Sampler.**

- `InteractionBatch` fields: `current`, `future`, `actions`, `source`.
- Two synthetic stores and a copy of the first whose task_id (50 entries), t (58), goal_after (27) and episode_success (32) differ: five batches of 64 with the same seed are identical (tokens, proprio, actions, source). No sampler attribute is named after task, success, goal, instruction or episode.
- `HeldoutRule` masks equal the direct masks for an episode range, a shard prefix and both; a prefix matching no shard raises `ValueError`.
- `gather` of all held-out transitions equals direct `TransitionStore` reads, `ActionCache.standardize` and the store's kinds.
- `proprio_stats.json` read alone equals `FeatureCache.proprio_mean` / `proprio_std`.
- Real stores: TRAIN (31,200, 18,732), HELDOUT (0, 4,470; 3,937 rollout, 533 rollout_stay). Forty batches of 256 have rollouts share 0.3760 against the pool share 0.3752.

**4. Criteria on synthetic inputs** (N = 4,096, d = 256, ψ ~ 𝒩(0, I), z ~ 0.3·𝒩(0, I), 10 % stay transitions with z scaled by 0.15):

| Case | Values | K0 | K1 | K2 | K3 |
|---|---|---|---|---|---|
| Δ̂ = z + 0.1·noise | ranks 248 / 247.5, ratio 1.06, R² 0.88 | F | F | F | F |
| ψ on one direction | ψ rank 1.0, top 1.00 | T | F | F | F |
| Δ̂ on one direction | Δ̂ rank 1.0, R² 0.004, SIGReg 0.40 | T | F | T | F |
| Δ̂ = 3z | ratio 3.0, R² −3.0, SIGReg 0.016 | F | T | T | F |
| stay Δ̂ = 2 × non-stay z | stay ‖Δ̂‖ 9.54 vs non-stay ‖z‖ 4.79 | F | T | F | F |
| Δ̂ = 0.05·noise | R² −0.03, SIGReg 0.40 | F | F | T | F |
| Δ̂ = 𝒩(0, I) noise | ratio 3.37, R² −12.3, SIGReg 2.7e-4 | F | T | F | F |
| K3 0.30, R0 0.10 / 0.30, 0.20 / 0.20, 0.05 | | F | F | F | T / F / F |

The effective rank equals `StateDistributionMetrics` (22.614130 both), and the SIGReg statistic of 𝒩(0, I) at N = 4,096 is 2.502e-4 in both. Stop rule with first_step 4,000: K1 at 2,000, 4,000, 6,000 stops at 6,000; K1 at 2,000 and 4,000 only does not stop; K1, K2, K1 at 4,000, 6,000, 8,000 does not stop; K2 and K3 at 4,000 and 6,000 stop at 6,000 with both; K0 at 2,000 does not stop; K0 at 4,000 stops.

**5. bf16 against float32** on one real batch of 256 (initialized trunk, randomized effect head): median relative error 5.2e-3 for ψ(o) and 1.4e-2 for z, where ‖z‖/‖ψ(o)‖ is 0.34.

**6. Config and checkpoint.** `from_json(to_json(config)) == config` for all three configs; a checkpoint loads with parameters equal to the saved tensors and the run's config.

## Smoke runs

node-69, NVIDIA L40S, seed 0; logs in `tmp/runlogs/s4p2/`. The K3 trajectories are rollouts_r0 episodes 80–99 (chained 10-step decisions), standing in for the demo_renders HELDOUT trajectories of `joint_all.json`.

| Run | Config | Code | GPU | Log |
|---|---|---|---|---|
| `outputs/runs/interaction_smoke` | `configs/interaction/smoke.json` | working tree before `9e0ca04` | CUDA 0 | `smoke.log` |
| `outputs/runs/interaction_smoke_rerun` | `smoke.json` with only `run_dir` changed (`tmp/s4p2/smoke_rerun.json`) | `9e0ca04` | CUDA 0 | `smoke_rerun.log` |
| `outputs/runs/interaction_smoke_prediction_per_dim` | `configs/interaction/smoke_prediction_per_dim.json` | working tree before `9e0ca04` | CUDA 1 | `smoke_prediction_per_dim.log` |

The working tree before `9e0ca04` differs from it only in writing no metrics line at checkpoint steps and in the stop report's number format. The rerun reproduces the first run exactly: every field of the 30 common metrics lines except the timings, and every field of the three monitor records except the monitor time, are equal (including peak memory).

**Throughput and memory** (all runs): median 4.13–4.17 steps/s (range 3.97–4.49) without checkpoints and the monitor, of which sampling from the in-memory stores takes 0.076–0.079 s per step; peak memory 8.19 GiB; the monitor takes 6.2–6.5 s per checkpoint; the stores load in 1–9 s from NFS. At this rate 20,000 steps take about 80 minutes.

### Weights 1, 1, 1 (`outputs/runs/interaction_smoke_rerun`)

| step | L_pred | SIGReg_ψ | SIGReg_Δ | ψ std | median ‖z‖ | median ‖Δ̂‖ | cos(Δ̂, z) |
|---|---|---|---|---|---|---|---|
| 0 | 12.68 | 0.4956 | 0.4015 | 0.2546 | 3.369 | 0 | 0 |
| 10 | 1.338 | 0.6122 | 0.4015 | 0.0878 | 1.091 | 0.150 | 0.155 |
| 20 | 0.0662 | 0.5051 | 0.4014 | 0.0189 | 0.254 | 0.125 | 0.213 |
| 50 | 0.00399 | 0.4058 | 0.4015 | 0.00443 | 0.0543 | 0.0380 | 0.122 |
| 100 | 0.00272 | 0.4021 | 0.4015 | 0.00296 | 0.0414 | 0.0351 | 0.091 |
| 200 | 0.00084 | 0.4086 | 0.4015 | 0.00180 | 0.0289 | 0.0107 | 0.155 |
| 299 | 0.00062 | 0.4017 | 0.4015 | 0.00153 | 0.0250 | 0.0078 | 0.149 |

L_pred falls by shrinking ψ: the per-dimension std of ψ drops from 0.25 to 0.0015. Both SIGReg terms sit at 0.4015, the statistic of a point mass at 0 (∫(1 − e^{−t²/2})² e^{−t²/2} dt on the 17 knots = 0.401496).

| step | counted | ψ rank / top share | Δ̂ rank / top share | median ‖Δ̂‖/‖z‖ | stay ‖Δ̂‖ vs non-stay ‖z‖ | R² | SIGReg(Δ̂) | K3 / R0 | K0 K1 K2 K3 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | no | 58.4 / 0.108 | 26.8 / 0.219 | 1.125 | 0.0347 vs 0.0328 | −0.988 | 0.4015 | 0.252 / 0.289 | F T T F |
| 200 | yes | 66.1 / 0.104 | 35.4 / 0.180 | 0.605 | 0.0095 vs 0.0184 | −0.102 | 0.4015 | 0.258 / 0.289 | F F T F |
| 300 | yes | 66.6 / 0.108 | 39.9 / 0.169 | 0.524 | 0.0062 vs 0.0145 | 0.002 | 0.4015 | 0.260 / 0.289 | F F T F |

K2 held at steps 200 and 300, so the run stopped after the step-300 checkpoint with exit status 3 and wrote `monitor_stop.md`. At step 300: Var(z)/Var(ψ) 0.515; direction correlation eigenvalue-weighted 0.747, min 0.301, median 0.435; median adjacent cosine raw 0.998, centered 0.74 (held-out and trajectory pairs alike); stay ‖Δ̂‖ percentiles 10/50/90 0.0054 / 0.0062 / 0.0073 against stay ‖z‖ 0.0012 / 0.0059 / 0.011.

### Prediction weight 1/256 (`outputs/runs/interaction_smoke_prediction_per_dim`)

| step | L_pred | SIGReg_ψ | SIGReg_Δ | ψ std | median ‖z‖ | median ‖Δ̂‖ | cos(Δ̂, z) |
|---|---|---|---|---|---|---|---|
| 0 | 12.68 | 0.4956 | 0.4015 | 0.2546 | 3.369 | 0 | 0 |
| 10 | 19.16 | 0.2939 | 0.4015 | 0.351 | 4.033 | 0.213 | 0.165 |
| 50 | 32.93 | 0.1008 | 0.2496 | 0.747 | 3.306 | 5.519 | 0.345 |
| 100 | 20.94 | 0.0959 | 0.2100 | 0.845 | 4.458 | 5.279 | 0.515 |
| 200 | 17.40 | 0.0656 | 0.1372 | 0.960 | 5.323 | 7.733 | 0.719 |
| 300 | 13.39 | 0.0580 | 0.1348 | 0.997 | 4.970 | 6.718 | 0.691 |
| 400 | 11.38 | 0.0474 | 0.1071 | 1.030 | 6.746 | 7.638 | 0.751 |
| 499 | 10.41 | 0.0472 | 0.1138 | 1.021 | 5.364 | 7.398 | 0.738 |

(L_pred is the unweighted term; the total uses L_pred / 256.)

| step | counted | ψ rank / top share | Δ̂ rank / top share | median ‖Δ̂‖/‖z‖ | stay ‖Δ̂‖ vs non-stay ‖z‖ | R² | SIGReg(Δ̂) | K3 / R0 | K0 K1 K2 K3 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | no | 3.4 / 0.519 | 2.0 / 0.786 | 1.323 | 3.31 vs 4.29 | 0.455 | 0.240 | 0.097 / 0.289 | F F F F |
| 200 | yes | 3.8 / 0.461 | 2.6 / 0.554 | 1.344 | 2.44 vs 4.44 | 0.758 | 0.170 | 0.165 / 0.289 | F F F F |
| 300 | yes | 3.8 / 0.433 | 2.6 / 0.574 | 1.197 | 1.84 vs 5.33 | 0.875 | 0.148 | 0.210 / 0.289 | F F F F |
| 400 | yes | 3.9 / 0.403 | 2.7 / 0.523 | 1.228 | 1.41 vs 5.55 | 0.905 | 0.130 | 0.244 / 0.289 | F F F F |
| 500 | yes | 3.9 / 0.404 | 2.7 / 0.515 | 1.201 | 1.36 vs 5.65 | 0.917 | 0.131 | 0.251 / 0.289 | F F F F |

No criterion held; the run ended at step 500 with exit status 0. ψ keeps unit scale, but on the held-out rollouts its effective rank is 3.9 and Δ̂'s 2.7 (K0 thresholds: ≤ 2, top share ≥ 0.9). At step 500: Var(z)/Var(ψ) 0.483; direction correlation eigenvalue-weighted 0.740, min 0.249, median 0.376; median adjacent cosine raw 0.940, centered 0.939; stay ‖Δ̂‖ percentiles 10/50/90 0.87 / 1.36 / 3.35 against stay ‖z‖ 0.12 / 0.65 / 3.13; mean cos(Δ̂, z) 0.670, median relative error 0.621.

## Phase 9 verification: K4, per-dim L_pred, fixed encoder

node-69, NVIDIA L40S, CUDA 0 and 1; launchers `tmp/s4p9/launch_round{1..5}.sh`, logs `tmp/runlogs/s4p9/`. Final code: commit `ee6e6af`. The monitor's CPU arithmetic ran with 4 BLAS threads in every run compared below.

### K4 on synthetic inputs

`scripts/check_scale_criterion.py` (log `final_check_scale_criterion.log`). N = 4,096, d = 256; ψ at step 0 has per-dimension std 0.25; Δ̂ at step 500 has std 0.3 per dimension. Evaluations at steps 500, 1,000, …, 3,000 multiply ψ or Δ̂ by the listed factors:

| Case | σ_ψ ratio | mean ‖Δ̂‖ ratio | K4a | K4b |
|---|---|---|---|---|
| ψ × 1, 0.5, 0.2, 0.101, 0.099, 0.01 | 1.000, 0.500, 0.200, 0.1010, 0.0990, 0.0100 | 1 | F F F F T T | F |
| Δ̂ × 1, 0.5, 0.2, 0.101, 0.099, 0.01 | 1 | 1.000, 0.500, 0.200, 0.1010, 0.0990, 0.0100 | F | F F F F T T |
| ψ grows to × 4, Δ̂ to × 1.7 (like the per-dim smoke run) | 1.5 … 4.0 | 1.0 … 1.7 | F | F |
| ψ and Δ̂ fluctuate within × 0.5 … × 2 | 0.5 … 2.0 | 0.5 … 2.0 | F | F |
| checkpoint at step 100 with σ_ψ 0.2546 → 0.00296 (the sum-reduction smoke run) | 0.0116 | no reference yet | T | F |
| Δ̂ ≡ 0 at step 500 | 1 | undefined | F | F |

- Schedule: `scheduled(n)` is true for n = 500, 1,000, 1,500, 2,000, 2,500, 20,000 and false for 0, 1, 100, 200, 499, 501.
- σ_ψ equals √(tr Cov(ψ) / d) to 6.5e-13 (σ_ψ 1.259).

Through `InteractionMonitor` on the 4,096 held-out transitions of `smoke.json` (fresh trunk, dynamics output layer randomized after step 0):

| Evaluation | σ_ψ (ratio) | mean ‖Δ̂‖ (ratio) | stop |
|---|---|---|---|
| step 0 | 0.262483 (1) | 0 | — |
| 500, output layer randomized | 0.262483 (1) | 7.389 (reference) | — |
| 1,000, unchanged | 0.262483 (1) | 7.389 (1) | — |
| 1,500, dynamics output layer × 0.05 | 0.262483 (1) | 0.3695 (0.0500) | K4b |
| 2,000, trunk output layer × 0.05 | 0.013124 (0.0500) | 7.384 (0.9993) | K4a |
| 2,500, restored | 0.262483 (1) | 7.389 (1) | — |
| fixed encoder (provisional PCA), steps 0, 500, 1,000 | 1.08946 (1) at all three | 0, 7.451, 7.451 (1) | — |

### Sum-reduction smoke run stopped by K4a

| Run | Config | Code | Stop | K4 at the stop |
|---|---|---|---|---|
| `outputs/runs/s4p9_final_smoke_sum` | `smoke.json` (run dir changed; weights 1, 1, 1; checkpoint every 100) | `ee6e6af` | K4a at step 100, exit status 3 | σ_ψ 0.002861 against 0.262483 at step 0 (ratio 0.0109); mean ‖Δ̂‖ 0.0352, no K4b reference yet |
| `outputs/runs/s4p9_new_smoke_sum_ckpt1000` | `smoke.json` with 1,000 steps and a checkpoint every 1,000 | `5e8243c` | K4a at step 500, exit status 3 | σ_ψ 0.000861 (ratio 0.0033); mean ‖Δ̂‖ 0.00613, which is also its K4b reference |

- With checkpoints every 100 steps, the first K4 evaluation is the checkpoint of step 100, and it stops the run. K0–K3 at that checkpoint (K1 and K2 hold) do not count, because `first_step` is 200. The batch per-dimension std of ψ in `metrics.jsonl`, a close but different quantity, already falls from 0.0878 at step 10 to 0.0189 at step 20, below 0.1 × σ_ψ(0) = 0.026.
- With checkpoints every 1,000 steps, the first K4 evaluation is the scheduled one at step 500. That update is not a checkpoint, so the stop writes `step_000500.pt`, measures K0–K3 on it (K2 holds there: R² −0.107, SIGReg(Δ̂) 0.4015, but it needs two checkpoints) and writes `monitor_stop.md` with the K4 table. `5e8243c` differs from `ee6e6af` in the release of the step-0 CUDA cache and in checking the PCA before creating the run directory, neither of which the K4 decision reads.
- Without K4 the same configuration ran to step 300 and was stopped by K2 (`s4p9_base_smoke_sum`, code `ffe766b`), as in Phase 2.

### Existing configurations unchanged

`scripts/compare_interaction_runs.py`: every field of every `metrics.jsonl` line except steps/s and sampling time, every field of every `monitor.jsonl` line except its time, `config.json` without `run_dir`, and every tensor of every checkpoint, over the steps both runs logged. `ffe766b` is the code before this phase (extracted to `tmp/s4p9/base_code_ffe766b`); all runs on CUDA 0.

| Config | Before (`ffe766b`) | After | Common steps | Compared | Differences |
|---|---|---|---|---|---|
| `smoke_prediction_per_dim.json` | `s4p9_base_smoke_prediction_per_dim` (500 steps, exit 0) | `s4p9_final_smoke_prediction_per_dim` (`ee6e6af`, 500 steps, exit 0) | 0–500 | 55 metrics lines, 5 monitor lines, 5 checkpoints (615 tensors) | 0 |
| `smoke.json` | `s4p9_base_smoke_sum` (K2 at 300) | `s4p9_final_smoke_sum` (`ee6e6af`, K4a at 100) | 0–100 | 11 metrics lines, 1 monitor line, 1 checkpoint (123 tensors) | 0 |

- The same comparisons with `0d0bd5e` (`s4p9_new2_*`) also show 0 differences.
- With `5e8243c`, which evaluates K4 at step 0 without releasing the CUDA cache afterwards, every value was equal except the logged `peak_memory_bytes`, lower by 5.8 MB from step 0 on (`compare_base_vs_new_smoke_prediction_per_dim.log`). Releasing the cached blocks of the step-0 evaluation removed the difference.
- Against the Phase 2 rerun `interaction_smoke_rerun` (run without a thread limit), the metrics lines and the checkpoint of step 100 are equal; monitor values differ in the last digits (for example ψ effective rank 58.412529492418436 vs 58.41252949241849), from the CPU BLAS thread count.
- `joint_all.json` now has weights 1/256, 1, 1; `smoke.json` and `smoke_prediction_per_dim.json` are unchanged. Every config in `configs/interaction/` round-trips, and its written text equals the file.

### Provisional PCA

`outputs/feature_pca/provisional_joint_all_3stores/pca.npz`, fitted on the sources of `joint_all.json` (log `pca_fit_provisional.log`; code `5e8243c`, whose `fit_feature_pca.py` is the final one; node-69 CPU, 8 threads, 23 s). The random training collection does not exist yet, so this PCA only serves the checks and the smoke run.

| Store | Frames | TRAIN frames | HELDOUT frames |
|---|---|---|---|
| demo_renders | 52,895 | 47,601 | 5,294 |
| demo_perturbations | 37,440 | 37,440 | 0 |
| rollouts_r0 | 24,202 | 19,532 | 4,670 |
| total | | 104,573 | |

No frame belongs to both splits, and every frame belongs to one.

| Components k | 1 | 8 | 16 | 32 | 64 | 128 | **256** | 512 | 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| variance explained | 0.156 | 0.483 | 0.610 | 0.715 | 0.802 | 0.875 | **0.9337** | 0.975 | 0.997 |

- The effective rank of the 1,536-d feature covariance is 78.1. The 256th eigenvalue is 1.84e-3 of the first.
- Self-check on the fitting frames: variance of each projected coordinate against its eigenvalue, max relative error 2.9e-15; largest projected covariance off the diagonal, 3.5e-16 of the top eigenvalue; max |VᵀV − I| 2.2e-15; eigen residual 2.5e-16 of the top eigenvalue.
- Reproducibility: a second fit (`tmp/s4p9/pca_repeat/pca.npz`) has bitwise-equal mean, components, eigenvalues and ratios, and equal sources.
- Encoder against a float64 NumPy projection, on 1,024 random TRAIN frames per store (σ of the projection 1.01–1.08): max |difference| 3.6e-6–4.7e-6 on CUDA and 3.0e-6–3.8e-6 on the CPU. [B, 2] frames under bf16 autocast with gradient enabled give float32 values without gradient, equal to the [2B] call without autocast. The buffers equal the float32 cast of the file; the module has no parameters.

### Fixed-encoder smoke run (R6 trainer)

`outputs/runs/s4p9_final_frozen_pca_delta_smoke`: `frozen_pca_delta.json` with 500 steps, warmup 100, a checkpoint every 100, a log line every 10 and `first_step` 200 (`tmp/s4p9/configs/final_r6_smoke.json`); code `ee6e6af`, CUDA 1, seed 0. TRAIN pool 93,633 transitions (demo_renders 43,701, demo_perturbations 31,200, rollouts_r0 18,732), HELDOUT 9,334; monitor on 4,096 held-out transitions (238 stay), K3 on demo_renders HELDOUT (R0 ratio 0.3724). Trained parameters: 13,315,840 (dynamics only). 5.5 steps/s including checkpoints and the monitor; peak memory 3.18 GiB.

Batch values (`metrics.jsonl`; L_pred summed over dims, the loss is L_pred / 256; SIGReg_ψ is logged with weight 0):

| step | L_pred | L_pred / 256 | SIGReg_ψ | SIGReg_Δ | median ‖z‖ | median ‖Δ̂‖ | cos(Δ̂, z) | median ‖Δ̂ − z‖/‖z‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 175.2 | 0.684 | 0.0185 | 0.4015 | 12.90 | 0 | 0 | 1.000 |
| 50 | 179.6 | 0.702 | 0.0184 | 0.3715 | 13.78 | 2.27 | 0.359 | 0.956 |
| 100 | 138.2 | 0.540 | 0.0194 | 0.2873 | 13.23 | 4.84 | 0.481 | 0.881 |
| 200 | 109.3 | 0.427 | 0.0199 | 0.1875 | 13.19 | 7.44 | 0.575 | 0.791 |
| 300 | 93.9 | 0.367 | 0.0182 | 0.1599 | 12.39 | 8.12 | 0.580 | 0.778 |
| 400 | 90.9 | 0.355 | 0.0204 | 0.1522 | 12.93 | 8.16 | 0.627 | 0.748 |
| 499 | 90.4 | 0.353 | 0.0195 | 0.1486 | 13.36 | 8.46 | 0.638 | 0.727 |

At step 499, L_pred per source is 77.5 (demo_renders), 96.6 (demo_perturbations) and 107.2 (rollouts_r0).

Monitor on the held-out set:

| step | counted | ψ rank / top share | Δ̂ rank / top share | median ‖Δ̂‖/‖z‖ | stay ‖Δ̂‖ vs non-stay ‖z‖ | R² | SIGReg(Δ̂) | K3 / R0 | K0 K1 K2 K3 | σ_ψ | mean ‖Δ̂‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | no | 49.3 / 0.170 | 14.9 / 0.199 | 0.384 | 1.84 vs 12.69 | 0.245 | 0.289 | 0.343 / 0.372 | F F T F | 1.0494 | 4.72 |
| 200 | yes | 49.3 / 0.170 | 25.6 / 0.143 | 0.570 | 2.03 vs 12.69 | 0.384 | 0.193 | 0.343 / 0.372 | F F F F | 1.0494 | 6.90 |
| 300 | yes | 49.3 / 0.170 | 30.7 / 0.146 | 0.643 | 2.07 vs 12.69 | 0.440 | 0.158 | 0.343 / 0.372 | F F F F | 1.0494 | 7.74 |
| 400 | yes | 49.3 / 0.170 | 34.3 / 0.133 | 0.665 | 1.94 vs 12.69 | 0.469 | 0.149 | 0.343 / 0.372 | F F F F | 1.0494 | 7.93 |
| 500 | yes | 49.3 / 0.170 | 35.4 / 0.125 | 0.678 | 1.94 vs 12.69 | 0.477 | 0.144 | 0.343 / 0.372 | F F F F | 1.0494 | 8.07 |

- No criterion stopped the run; it ended at step 500 with exit status 0. σ_ψ stays at its step-0 value, as it must for a fixed encoder. The ψ columns and the K3 ratio do not change between checkpoints for the same reason.
- At step 500: Var(z)/Var(ψ) 0.598; direction correlation eigenvalue-weighted 0.711, min 0.186; mean cos(Δ̂, z) 0.600, median relative error 0.757; median ‖z‖ 12.38, median ‖Δ̂‖ 8.42; median adjacent cosine raw 0.696, centered 0.694; stay ‖Δ̂‖ percentiles 10/50/90 1.51 / 1.94 / 2.96 against stay ‖z‖ 0.94 / 5.35 / 9.60.
- A run of the same config with `5e8243c` (`s4p9_frozen_pca_delta_smoke`) is equal in every logged value and every checkpoint tensor, except `peak_memory_bytes` (3,407,051,776 vs 3,417,644,032), which depends on the release of the step-0 cache.

Checkpoint and loaders (`final_check_frozen_pca_encoder.log`): the projection stored in `step_000500.pt` equals the float32 cast of the PCA file. `FrozenPcaDeltaModel.encode` and `effect` equal the loaded modules bitwise. `InteractionCheckpointStore.load` and `JointInteractionModel` refuse the fixed-encoder checkpoint; `FrozenPcaDeltaModel` refuses a trainable-trunk checkpoint, which `JointInteractionModel` still loads. The trainer refuses the provisional PCA on `smoke.json`'s two sources before creating a run directory.

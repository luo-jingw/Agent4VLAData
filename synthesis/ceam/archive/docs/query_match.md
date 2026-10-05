# Query q(o, l) and match m(r, q) on a frozen action representation (MVP, phases 10–12; code for phases 14–15)

The MVP of `docs/framework.md`: a frozen action representation r(o, a), a task encoding e(l), a query q(o, l), and a match m(r, q) trained by contrast within one observation. In the closed loop, a replaceable sampler (the no-ψ π) proposes candidates, and the candidate with the highest match is executed. This page covers the components, the tables, the training and its checks, the offline evaluation, the closed-loop integration, and the closed loops of 10 tasks × 50 benchmark inits over three seeds against a uniform choice among the same candidates and against π's mean chunk. The later sections cover the code added for phases 14–15 (a smooth random candidate source, R6 as r, two training options), its checks and the G2 power pilot; then R6 replaces r, with its pre-registered within-observation distinguishability check (phase 13). The last section evaluates the smooth random sampler from HELDOUT demonstration states, argmax against a uniform choice, with the two q versions on R6 (phase 18).

## Components and the framework

| Component | Framework clause | Implementation |
|---|---|---|
| r(o, a), frozen | 1 (action representation, task-free) | the output of the effect head `goal_dynamics_perturb` step 3000 (trained with data A on the fixed stage one) for the frames of o, ψ(o) of the fixed stage one `goal_instruction_contrast` step 5000, and the standardized chunk a. This is a historical implementation used as one candidate r. No gradient reaches it. |
| e(l) | 1 (the task input of the query) | `TaskEncoder`: masked mean of the frozen SigLIP tokens, trainable projection |
| task-side non-collapse | 2 (query side: on the task encoding, not on q) | SIGReg on e(l) of the 10 tasks, one random instruction variant each, per step |
| q(o, l) | 1 (query, observation- and task-conditioned) | `QueryNetwork`: Transformer over readout, proprio, 2 × 64 frozen DINOv2 tokens and e(l); does not take ψ |
| m(r, q) | 3 (equivalence of r with the query) | `Match`: s · cos(r, q), s learnable, clamped at `match_max_scale` (100 in the MVP) |
| training contrast | 3 (within one observation: the demonstration chunk accepted, other reachable chunks distinguished) | InfoNCE over {a⁺} ∪ N(o), all 33 chunks at the same o |
| sampler | 4 (replaceable) | the no-ψ π, K = 32: mean + 31 noisy (scales 0.05/0.1/0.2, DCT J = 1) |
| selection | 4 (execute the highest match, re-observe) | `SelectionStrategy.ARGMAX` with `MatchTarget.QUERY_NETWORK`; no stay option, no escalation |
| control | final criterion (random-candidate control) | `SelectionStrategy.UNIFORM`: a uniform candidate among the same 32 |

Engineering choices that are not framework requirements: the tables, the exclusion radius d_min, the source mix of N(o), bf16 autocast of the trainable modules, and the clamp of s.

## Code

| Class | File |
|---|---|
| `QueryMatchObservations`, `QueryMatchTables` | `ceam/data/query_match_tables.py` |
| `SameObservationCandidates`, `SameObservationConfig`, `CandidateBatch`, `NegativeSource` | `ceam/data/same_observation_candidates.py` |
| `TaskEncoder`, `TaskEncoderConfig` | `ceam/model/task_encoder.py` |
| `QueryNetwork`, `QueryNetworkConfig` | `ceam/model/query_network.py` |
| `Match` | `ceam/model/match.py` |
| `ActionRepresentation` (the frozen r) | `ceam/model/action_representation.py` |
| `QueryMatchObjective`, `QueryMatchWeights`, `QueryMatchLossReport` | `ceam/losses/query_match_objective.py` |
| `QueryMatchTrainingConfig` | `ceam/training/query_match_config.py` |
| `QueryMatchCheckpointStore`, `LoadedQueryMatch` | `ceam/training/query_match_checkpoint_store.py` |
| `QueryMatchTrainer`, `AcceptanceEvaluator`, `TaskTexts`, `ObservationFrames`, `CandidateStatistics` | `ceam/training/query_match_trainer.py` |
| `SelectionStrategy.ARGMAX` / `UNIFORM`, `MatchTarget.QUERY_NETWORK` | `ceam/policy/candidate_selector.py` |
| `QueryNetworkMatch`, runner integration | `ceam/policy/closed_loop_runner.py` |
| `RepresentationKind` (stage one + stage two, or R6) | `ceam/model/action_representation.py` |
| `ActionRepresentationLoader`, `LoadedActionRepresentation` | `ceam/training/action_representation_loader.py` |
| `SmoothCandidates`, `SmoothProposedChunks`; `CandidateSource.SMOOTH`, `ChunkSource.SMOOTH` | `ceam/policy/smooth_candidates.py`; `ceam/policy/closed_loop_runner.py` |
| `DemoStartPoint`, `DemoStartSchedule`; `EpisodeStart.DEMO`, `ClosedLoopRunner.run_demo` | `ceam/policy/demo_start.py`; `ceam/policy/closed_loop_runner.py` |

Scripts: `scripts/build_query_match_tables.py`, `scripts/train_query_match.py`, `scripts/evaluate_query_match.py` (`acceptance`, `tasks`, `sweep`), `scripts/rescore_branch_candidates.py score --query-match`, `scripts/run_closed_loop.py` (`--strategy argmax|uniform`, `--match-target query_network --query-match-checkpoint`, `--start demo --demo-levels`). Configs: `configs/query_match/mvp_seed{0,1,2}.json`.

`same_observation_candidates.py` and the three `query_match_*` training modules import `ceam.policy.proposal_candidates`, so they are imported by module path and are not exported from `ceam/data/__init__.py` and `ceam/training/__init__.py`; `import ceam.data` and `import ceam.training` do not load `ceam.policy`.

## Tables (`scripts/build_query_match_tables.py`)

Output `outputs/query_match/tables_mvp/` (manifest last). One observation per `demo_renders` transition (t, t + 10): 48,565 observations, 43,701 TRAIN and 4,864 HELDOUT, the frames with a full 10-step chunk and a future frame (the `StageTwoSampler` rule; the same 43,701 TRAIN rows as the retrieval pool). Frames are the simulator renders of the store; proprio is standardized with the feature cache's statistics, as `ObservationEncoder` does in the closed loop.

| Table | Content |
|---|---|
| `observations.npz` | store frame, feature-cache row, episode, t, task, split, gripper command in force (a_{t−1}'s gripper; 1.0, the settle no-op's, at t = 0), raw demonstration chunk |
| `psi.npy` | ψ(o) of the fixed stage one, float32 without autocast |
| `pi_mean_seed{0,1,2}.npy` | the standardized mean chunk of the no-ψ π of seed 0, 1, 2 (`goal_proposal_nopsi_seed{0,1,2b}` step 3000), conditioned on the observation task's original instruction (variants[0], the text of its closed loop); float32, eval mode, batch 256 |
| `neighbors.npz` | per observation, the 64 TRAIN observations of other tasks nearest by the frozen-feature key of the rendered frames (`RetrievalKeys.frozen`), nearest first, with their key distances |
| `manifest.json` | d_min and its statistics, π checkpoints and their MSE, checks, provenance |

Measured (`tmp/runlogs/s4mvp/logs/build_tables.log`, 87 s on node-62 CUDA 0):

- **Checks**: stored chunks against `ActionCache.chunks` max |Δ| 0; 0 split and 0 task mismatches against the `EpisodeIndex`; every neighbour is of another task and in TRAIN. Key distance of the nearest neighbour: median 0.546 (of the 64th: 0.620); the 64 neighbours come from a median of 22 distinct episodes (first 2,000 observations).
- **d_min** = 1.9007 (standardized 70-d chunk space). Quantiles of the nearest same-task chunk of another TRAIN episode: p10 1.106, p25 1.462, p50 1.901, p75 2.494, p90 3.198; per-task medians 1.466 (task 2) to 2.364 (task 7).
- **π on the rendered frames**, MSE of the standardized mean chunk against the demonstration chunk:

| π | TRAIN | HELDOUT |
|---|---|---|
| seed 0 | 0.0486 | 0.2632 |
| seed 1 | 0.0477 | 0.2674 |
| seed 2 | 0.0457 | 0.2695 |

On the dataset video frames the same π have HELDOUT MSE 0.2535, 0.2640 and 0.2648 (`docs/no_psi_baseline.md`).

## Same-observation candidates (`SameObservationCandidates`)

Per observation o with demonstration chunk a⁺, K = 32 other chunks at the same o, in this order:

| Source | Count | Construction |
|---|---|---|
| smooth random | 12 | `SmoothActionSampler` with the collection settings (J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch 0.3), gripper_now the command in force at o, clipped to [−1, 1] |
| other-task demonstration chunks | 8 | the demonstration chunks of the 64 other-task neighbours of o, taken in a random order |
| π noisy candidates | 12 | `ProposalCandidates.sample` of the table's mean chunk of π seed k at o: candidates 1, 2, … (scales 0.05/0.1/0.2 cycled, DCT J = 1, no gripper flip), never candidate 0 |

- **Exclusion**: a candidate whose standardized 70-d distance to a⁺ is below d_min is excluded, and its source moves to its next draw.
- **Cap**: each source considers at most 4 × its count (3 retries). Slots a source cannot fill within its cap are filled with further smooth random draws of the same stream (fills, up to 64 draws, also excluded within d_min; RuntimeError if they run out, never observed).
- **Determinism**: the candidates of observation i under draw key d depend only on (seed, d, i). Smooth chunks come from sampler key (1, d, i, 0), the neighbour order from `default_rng([seed, 2, d, i])`, and π's noise from `NoiseStream` (3, d, i) of `ProposalCandidates` seeded with seed. Training uses seed = the config seed and d = the update number. The evaluation uses seed = config seed + 1000 (held out) and d = 0.
- **Generators**: π's candidates come from `ProposalCandidates` objects of the loaded π with each task's original instruction. `sample` does not run π; the mean chunk comes from the table.

Checks (`tmp/runlogs/s4mvp/check_candidates.py`, output `tmp/runlogs/s4mvp/logs/check_candidates.log`, CPU):

- 256 TRAIN observations: the smallest distance of any candidate to a⁺ is 1.9008 (d_min 1.9007); 0 below d_min.
- Re-drawing with the same key is identical; key 1 changes 99.7% of the slots. An observation drawn alone equals its row in a batch.
- The π candidates of one observation equal `ProposalCandidates.sample` (49 draws, stream (3, 0, o)) filtered by d_min, and its smooth chunks equal `SmoothActionSampler.sample` filtered by d_min.

Exclusion and fill rates, 2,048 observations each (draw key 3):

| Split | source | excluded / considered | shortfall / quota | observations with a shortfall |
|---|---|---|---|---|
| TRAIN | smooth | 0 / 24,576 | 0 | 0 |
| TRAIN | other task | 40 / 16,424 (0.0024) | 0 | 0 |
| TRAIN | π noise | 45,010 / 65,975 (0.682) | 3,611 / 24,576 (0.147) | 769 |
| HELDOUT | smooth | 1 / 24,577 | 0 | 0 |
| HELDOUT | other task | 28 / 16,412 (0.0017) | 0 | 0 |
| HELDOUT | π noise | 4,268 / 28,688 (0.149) | 156 / 24,576 (0.006) | 40 |

- **Fills**: 5.5% of the TRAIN slots and 0.24% of the HELDOUT slots, none excluded.
- **Why π noise is excluded on TRAIN**: π fits its TRAIN demonstrations. Its mean chunk lies within d_min of a⁺ at 69.3% of the TRAIN observations (median distance 1.59), against 8.9% on HELDOUT (median 3.47). A constant offset of scale s has norm s · √60 over the 60 arm entries, 1.5 at s = 0.2, the order of d_min.
- **Distance to a⁺** (median, TRAIN): smooth 11.6, other task 9.4, π noise 2.29, fill 11.8.
- **Training batches** (`metrics.jsonl`, over all updates): π noise excluded in 0.686 / 0.693 / 0.703 of its draws (seeds 0 / 1 / 2), and 1.80 / 1.90 / 2.01 fills per observation (of 32 slots).

## Models and loss

- **`TaskEncoder`**: the mean of the SigLIP tokens whose cached mask is True, in float32 → LayerNorm(1152) → Linear(1152, 512) → GELU → Linear(512, 256) = e(l). There is no output normalization. Parameters: 723,968.
- **`QueryNetwork`**: 131 tokens [readout, proprio, 128 visual tokens, e(l)] with the embeddings and 4-layer pre-norm encoder of the other heads (width 512, 8 heads, FFN 2048). q = Linear(LayerNorm(readout)), 256-d, default initialization. It takes no ψ. Parameters: 13,306,624.
- **`Match`**: m = s · cos(r, q) with s = exp(min(logit_scale, log match_max_scale)), match_max_scale 100 in the MVP configs, logit_scale initialized to log(1/0.07) (the convention of `InstructionContrastLoss`). Computed in float32 outside autocast.
- **`ActionRepresentation`**: r = dynamics(frames, trunk(frames), standardized chunk). Both modules are in eval mode with requires_grad False, and every call runs in float32 without autocast under `torch.no_grad`, the computation of the closed loop's predicted effects.
- **`QueryMatchObjective`**: loss = InfoNCE + SIGReg, weight 1 each.
  - InfoNCE: mean over observations of logsumexp_j m_j − m_{a⁺}, over the 33 chunks at the observation.
  - SIGReg: `SigRegLoss(256)` on e(l) of the 10 tasks, one uniform instruction variant each.
  - No term acts on q.

## Training (`QueryMatchTrainer`, `configs/query_match/mvp_seed{0,1,2}.json`)

- **Data**: the 43,701 TRAIN observations of the tables, 64 per update, uniform with replacement. Each observation's task is its episode's task, and its text a uniform instruction variant of that task (5 per task). The frames are read from `demo_renders` held in memory.
- **Per update**:
  - {a⁺} ∪ N(o) (draw key = update);
  - r of the 64 × 33 chunks, float32, no gradient;
  - e(l), q(o, l) and the 10 SIGReg encodings in bf16 autocast;
  - m and the loss in float32.
- **Optimizer**: AdamW, lr 3e-4, weight decay 0.05 on matrices without 'token' or 'embedding' in their name, 150 warmup updates, then cosine decay to 0 at 5,000 updates; gradient clipping 1.0. These are the optimizer settings of the proposal and velocity heads.
- **Seeds**: seed k (k = 0, 1, 2) sets the initialization, the batch sampler, the variants and the training candidate draws, and uses π seed k for the noisy negatives (`goal_proposal_nopsi_seed{0,1,2b}`).
- **Outputs**: `outputs/runs/query_match_mvp_seed{0,1,2}/`, with `config.json`, `metrics.jsonl` (every 50 updates), `evaluation.jsonl` (every 500 updates: acceptance on 1,024 TRAIN and 1,024 HELDOUT observations, held-out candidate seed), `tables_check.json`, `frozen_check.json` and checkpoints every 1,000 updates.
- **Placement and speed**: seeds 0 and 1 on node-62 CUDA 0 and 2, seed 2 on node-63 CUDA 0. 2.2–2.3 updates/s. Per update, drawing the candidates takes 0.08 s and r takes 0.33 s. Loading the store takes 150 s on node-62. Peak GPU memory 6.04 GiB.

## Checks of the training (phase 10)

Scripts and outputs are in `tmp/runlogs/s4mvp/` (R below).

| Check | Result |
|---|---|
| r frozen: requires_grad of r's 26,490,880 parameters | False for all; `representation_grad_free` true at every logged update of the three runs (every r parameter has grad None after the backward) |
| r frozen: parameters after training (`frozen_check.json`, 123 tensors) | max |Δ| 0 against the loaded parameters and 0 against the checkpoint files, in all three runs |
| no success or outcome field read | the tables hold no such field (`observations.npz`: cache_row, demo_chunk, episode, frame, gripper_now, heldout, t, task_id); the new modules and scripts name `goal_after` / `episode_success` only in the docstrings saying they are not read (R `no_success_fields.txt`). `TransitionStore.open` loads the store's transition table, whose columns include them; the builder reads current_frame, actions, kind, task_id, episode, t and shard of it, and the trainer reads only frames |
| loss against a direct re-implementation, random r, q, e (R `check_units.py`, 3 trials, float64 loops, SIGReg with the same random directions) | InfoNCE |Δ| ≤ 2.9e-7, SIGReg |Δ| ≤ 3.0e-8, total = InfoNCE + SIGReg |
| the same on real r and q (R `check_overfit.py`, 16 TRAIN observations) | InfoNCE 3.497795 against 3.497795 (|Δ| 1.4e-7); log 33 = 3.4965 at initialization |
| ψ of the tables against a recomputation (64 observations, every run) | max |Δ| 1.2e-6 |
| `TaskEncoder`: padded tokens changed to 1000 | e(l) max |Δ| 0; a visible token changed: 0.30 |
| `QueryNetwork`: permutation of the batch; forward arguments | permuted exactly; (frames, task), no ψ |
| `Match`: s at logit_scale 10 | 100 (the clamp); initial s 14.29 |

**Overfitting a tiny subset** (R `check_overfit.py`, logs `check_overfit*.log`): 16 TRAIN observations, one fixed candidate draw, AdamW without weight decay, bf16 autocast as in training.

| Run | InfoNCE at update 0 → 400 → 3000 | s at 3000 | top-1 of a⁺ | rank of a⁺ |
|---|---|---|---|---|
| lr 3e-4, 400 updates | 3.498 → 1.652 | – (16.2 at 400) | 0 → 0.8125 (from update 50) | 13.6 → 1.38 |
| lr 3e-4, 3000 updates | 3.498 → 1.61 → 1.072 | 36.1 | 0.8125 | 1.19 |
| lr 1e-3, 3000 updates | 3.498 → 1.52 → 0.480 | 100 (clamp, from update 2100) | 0.8125 | 1.19 |

- **The limit**: the loss falls as long as s grows, and stays at 0.4804 once s reaches its clamp. In 3 of the 16 observations one negative stays above a⁺.
- **Feasibility**: q = r̂⁺ itself would rank a⁺ first at every observation (512 TRAIN observations, top-1 1.000 in float32 and in bf16; R `diagnose_r.py`). At s ≤ 100 the InfoNCE optimum does not coincide with that ordering.
- **Geometry of r within an observation** (R `diagnose_r.txt`, 512 TRAIN observations × 33 chunks):
  - cos(r⁺, r) of the π-noise candidates: median 0.9975, p90 0.9997, max 0.99999; ‖r − r⁺‖/‖r⁺‖ median 0.136.
  - Smooth chunks: median cos 0.655. Other-task chunks: median cos 0.834.
  - Per observation, the closest negative has cos(r⁺, r_j) median 0.99906, and above 0.999 in 52% of the observations.
  - Within-state share of r's variance 0.547.

Closed-loop mechanics: CPU unit checks (R `check_units.py`, output `check_units.txt`) and closed-loop checks in "Closed loop" below.

## Training results

Batch metrics, mean over windows of 500 updates (R `training_summary.txt`; InfoNCE of 33 equal logits is log 33 = 3.497):

| Seed | updates | InfoNCE | SIGReg | top-1 of a⁺ | rank of a⁺ | s |
|---|---|---|---|---|---|---|
| 0 | 0–499 / 2000–2499 / 4500–4999 | 2.868 / 2.411 / 2.112 | 0.121 / 0.056 / 0.054 | 0.103 / 0.244 / 0.372 | 8.10 / 4.61 / 3.17 | 15.0 / 25.8 / 31.3 |
| 1 | same | 2.840 / 2.427 / 2.101 | 0.119 / 0.055 / 0.054 | 0.109 / 0.220 / 0.359 | 7.61 / 4.42 / 3.16 | 15.1 / 25.7 / 31.2 |
| 2 | same | 2.884 / 2.426 / 2.110 | 0.121 / 0.054 / 0.054 | 0.103 / 0.230 / 0.371 | 7.88 / 4.66 / 3.38 | 15.1 / 25.5 / 31.1 |

- ‖q‖ rises from 14.5 to 19–20 in the first 2,000 updates, and ‖e(l)‖ settles at 14.5.
- The gradient norm before clipping is 0.8–2.5.

Periodic evaluation (`evaluation.jsonl`: 1,024 TRAIN and 1,024 HELDOUT observations, held-out candidate seed, original instruction), top-1 of a⁺ at 500 / 2,500 / 5,000 updates:

| Seed | TRAIN | HELDOUT | HELDOUT InfoNCE |
|---|---|---|---|
| 0 | 0.144 / 0.272 / 0.447 | 0.209 / 0.244 / 0.250 | 2.943 / 3.050 / 3.134 |
| 1 | 0.140 / 0.265 / 0.373 | 0.223 / 0.229 / 0.249 | 2.904 / 3.004 / 3.099 |
| 2 | 0.112 / 0.229 / 0.364 | 0.208 / 0.257 / 0.248 | 2.944 / 2.935 / 2.974 |

The HELDOUT top-1 stays at 0.21–0.27 from update 500 on, while the TRAIN top-1 keeps rising.

### Same-observation acceptance at step 5000

`evaluate_query_match.py acceptance`: every TRAIN (43,701) and HELDOUT (4,864) observation, candidates drawn with the held-out seed config seed + 1000, the task's original instruction. "Beats <source>": a⁺ above every candidate of that source. Swapped: the instruction of each of the 9 other tasks. Chance top-1 is 1/33 = 0.030; chance rank is 17.

| Seed | split | top-1 | rank mean / median | beats smooth / other task / π noise | swapped top-1 / rank | a⁺ higher under the own text than under another |
|---|---|---|---|---|---|---|
| 0 | TRAIN | 0.402 | 3.08 / 2 | 0.897 / 0.934 / 0.462 | 0.138 / 8.91 | 0.906 |
| 0 | HELDOUT | 0.255 | 7.84 / 7 | 0.758 / 0.775 / 0.307 | 0.127 / 11.84 | 0.886 |
| 1 | TRAIN | 0.397 | 3.06 / 2 | 0.904 / 0.938 / 0.454 | 0.135 / 8.66 | 0.882 |
| 1 | HELDOUT | 0.231 | 8.04 / 8 | 0.762 / 0.776 / 0.289 | 0.112 / 11.89 | 0.871 |
| 2 | TRAIN | 0.380 | 3.18 / 2 | 0.901 / 0.936 / 0.435 | 0.151 / 8.09 | 0.886 |
| 2 | HELDOUT | 0.238 | 7.82 / 7 | 0.774 / 0.787 / 0.294 | 0.126 / 11.35 | 0.859 |

- **π-noise candidates in these sets**: on HELDOUT, 16–18% of their draws are excluded, and fills take 524–578 of 155,648 slots. On TRAIN the rates are 69–70% and 5.6–6.3% of the slots.
- **r within these sets**: per-state effective rank 2.5–2.6, pooled within-state effective rank 14.6–14.8, within-state share 0.52–0.55. The negatives include smooth and other-task chunks far from a⁺.

### Task encodings

`evaluate_query_match.py tasks`: e(l) of all 50 instruction variants (10 tasks × 5), float32. Nearest-task accuracy counts a variant as correct when its nearest other variant, and its nearest leave-one-out task centroid, belong to its task.

| Encoding | within-task cosine distance | between-task cosine distance (mean / closest task pair) | nearest-variant / nearest-centroid accuracy | mean norm |
|---|---|---|---|---|
| e(l), seed 0 / 1 / 2 | 0.0014 / 0.0015 / 0.0013 | 1.111 / 1.085–1.091 | 1.0 / 1.0 | 14.5 |
| mean-pooled frozen SigLIP tokens (the input) | 0.086 | 0.253 / 0.123 | 1.0 / 1.0 | 28.2 |

## Closed loop (phase 11)

### Mechanism

- **`SelectionStrategy.ARGMAX`** (`--strategy argmax`): the candidate with the largest score (`np.argmax`, the first on ties). It never stays, whatever the sign of the scores.
- **`SelectionStrategy.UNIFORM`** (`--strategy uniform`): a uniform candidate. The draw is `numpy.random.default_rng([--seed, 0x554E4946, task, start code, init, decision])`, so it depends only on the decision's key; it uses the key of π's noise stream in another entropy domain. It never stays. With `--effect-source predicted` the scores are computed and logged but not used; with `none` nothing is scored.
- **Requirements**: both strategies require `--candidate-source proposal` and one `--escalation` K; argmax requires `--effect-source predicted`. The runner and the script refuse violations with their own messages. Stay, the escalation and the stuck rule never apply.
- **`MatchTarget.QUERY_NETWORK`** (`--match-target query_network --query-match-checkpoint CKPT`):
  - At each decision, q(o, l) = query(frames, e(l)) is computed once from the current encoded frames, in float32 without autocast under inference mode. e(l) is computed once per process from the task's original instruction (variants[0], the text the no-ψ π and q_l take).
  - The candidates' scores are the checkpoint's `Match` m(Δ̂, q), with Δ̂ = r from the effect head (`--effect-source predicted`).
  - The script refuses a checkpoint whose ψ, effect head or action cache differ from `--checkpoint`, `--dynamics-checkpoint` and `--action-cache`.
  - The decision line prints ‖q‖ and the query's time. A recorded shard's provenance holds `query_match_checkpoint` and `query_match_step`.
- **Loop plumbing**: the loop's stage one (`--checkpoint`) is the fixed stage one, because r's ψ is the ψ the runner computes (`ObservationEncoder`). The no-ψ π takes nothing from it.

### Existing paths bit-identical

| Check | Result |
|---|---|
| `CandidateSelector` MATCH / RANDOM / MEAN × QUERY / GOAL / GOAL_SUBSPACE / VELOCITY against the pre-change selector (`7fca0fe`), K 1–512, float32 and float16, both signs | 72 / 72 calls identical in index and scores, each of the 12 combinations; MEAN `select_unscored` identical |
| π-mean closed loop, the `v9_mean_nopsi_s0` command, task 0, inits 0–4, node-62 CUDA 2: new code (`mvp_checks/mean_nopsi_s0_t0_new`) and base code (`…_base`) against `v9_mean_nopsi_s0` | 5 / 5 episodes identical in results (wall time excepted) and in every trace field, for both |
| MATCH with q_l among π's candidates (0.05/0.1/0.2), rule 32/128/512, Δ +A, fixed ψ: the `v7_proposal_small_query_dynA` command, task 3, inits 0–4: new code (`mvp_checks/small_query_t3_new`) against base code and against `v7_proposal_small_query_dynA` | 5 / 5 identical in results and traces against each |
| refusals of `run_closed_loop.py` before any loading | argmax with effect none, argmax with retrieval, uniform with an escalation schedule, query_network without checkpoint, checkpoint without query_network, query_network with effect none; the pre-existing random-with-predicted refusal unchanged |

Unit checks of the new modes (R `check_units.txt`):

- ARGMAX selects the best candidate when every score is negative (MATCH stays there); ties go to the first.
- UNIFORM: scored and unscored draws are equal and independent of call order. Over 15,000 decision keys the 32 counts are 410–523 (χ² 34.1, 31 dof). The correlation with the first draw of π's noise stream of the same decision is −0.004.
- QUERY_NETWORK scores equal 20 · cos exactly at s = 20.
- Each missing or superfluous argument is refused.
- The sweep scorer (`QueryMatchScorer`) equals the runner-style computation (Δ̂, q, `CandidateSelector` ARGMAX with the query-network target) on 8 stored sweep states: max |Δm| 0, argmax equal 8 / 8 (R `check_scorer.py`).

### Smoke (tasks 3 and 6, inits 0–1, seed-0 step-1000 checkpoint, π seed 0, node-63 CUDA 1)

Outputs `outputs/closed_loop/mvp_smoke/{argmax,uniform,uniform_unscored}_s0_step1000/`; summary R `smoke_summary.txt`.

| Mode | task 3 success (env steps) | task 6 success (env steps) | decisions | stays | chosen = best score | distinct candidates chosen |
|---|---|---|---|---|---|---|
| argmax, query network | 2 / 2 (156, 76) | 2 / 2 (88, 91) | 24, 19 | 0 | 100% | 12, 14; candidate 0 never |
| uniform, scored | 1 / 2 (79, 300) | 1 / 2 (300, 99) | 38, 40 | 0 | 0%, 7% | 25, 25 |
| uniform, unscored | 1 / 2 (79, 300) | 1 / 2 (300, 99) | 38, 40 | 0 | – | 25, 25 |

- The scored and unscored uniform runs execute the same episodes, since the draw does not depend on the scores.
- Decision latency on the shared GPU (6 processes): 28–35 ms with the query network (encode 17–19 ms, q 1.4–1.9 ms, π 3–8 ms, Δ 6–7 ms), against 22–25 ms unscored.

## Offline: the branch test's sweep states

- **States**: every candidate of π was executed at decisions 1, 6, …, 26 of π's failed episodes (`docs/branch_success_test.md`, phase 26). These are 1,084 states in 181 episodes, each with 32 candidates and each candidate's simulator outcome: 516 states of `fixed`, 287 of `flow` and 281 of `flow_full_sgd` (the states with an outcome for all 32 candidates).
- **Candidates**: the ψ π's mean + 31 noisy candidates, with scales 0.05/0.1/0.2, DCT J = 1 and the same construction as the closed loop's.
- **Scoring**: `rescore_branch_candidates.py score --query-match` scores them from the stored frames (`branch_rescore/<name>/observations`), one state at a time as in the closed loop. It computes r with the fixed ψ and the data-A head, q with the checkpoint's query network and e(l) of the original instruction, and m = s · cos(r, q). It equals the runner's computation exactly (see "Closed loop").
- **Reference**: `fixed`, the q_l match of the fixed stage one with the same r (cos(r, q_l)).
- **Statistics**: pooled over the three names, 95% intervals from a bootstrap over episodes (2,000 resamples, seed 0). Output `outputs/analysis/query_match/sweep/mvp_pooled.{md,json}`; per name `mvp_{fixed,flow,flow_full_sgd}.{md,json}`.

| Scoring | best (top 1) | all (random) | top 1 − random [95% CI] | top 1 − same scale [95% CI] | within-state AUC (422 states) | same-scale AUC (413 states) |
|---|---|---|---|---|---|---|
| q_l (`fixed`) | 0.210 | 0.170 | +0.040 [+0.023, +0.057] | +0.026 [+0.009, +0.044] | 0.522 [0.505, 0.540] | 0.526 [0.505, 0.545] |
| q(o, l), seed 0 | 0.205 | 0.170 | +0.035 [+0.018, +0.051] | +0.020 [+0.003, +0.037] | 0.502 [0.483, 0.521] | 0.520 [0.501, 0.539] |
| q(o, l), seed 1 | 0.201 | 0.170 | +0.031 [+0.015, +0.046] | +0.018 [−0.000, +0.035] | 0.503 [0.485, 0.520] | 0.513 [0.494, 0.531] |
| q(o, l), seed 2 | 0.201 | 0.170 | +0.031 [+0.014, +0.047] | +0.018 [+0.000, +0.034] | 0.507 [0.488, 0.526] | 0.519 [0.498, 0.538] |

Paired against q_l with the same r (difference, 95% episode bootstrap):

| q(o, l) seed | top-1 success (only / q_l only, McNemar p) | top 1 − same scale | within-state AUC | same-scale AUC |
|---|---|---|---|---|
| 0 | −0.006 [−0.026, +0.016] (57 / 63, 0.65) | −0.006 [−0.028, +0.017] | −0.020 [−0.041, −0.002] | −0.005 [−0.027, +0.017] |
| 1 | −0.009 [−0.030, +0.011] (52 / 62, 0.40) | −0.009 [−0.031, +0.014] | −0.020 [−0.038, −0.002] | −0.013 [−0.032, +0.008] |
| 2 | −0.009 [−0.029, +0.011] (54 / 64, 0.41) | −0.009 [−0.030, +0.013] | −0.016 [−0.036, +0.002] | −0.007 [−0.029, +0.015] |

- **Scale of the best candidate**: scale 0.2 in 80% of the states under q(o, l) (0.1: 18%, 0.05: 1–2%), and in 84% under q_l.
- **Per name, top 1 − same scale**:
  - fixed: q(o, l) +0.023 to +0.032, q_l +0.040;
  - flow: q(o, l) +0.015 to +0.023, q_l −0.006;
  - flow_full_sgd: q(o, l) −0.006 to +0.003, q_l +0.034.
  - No per-name paired difference against q_l has a McNemar p below 0.13.
- **Reading**:
  - The top candidate of q(o, l) succeeds more often than a random candidate of the state: +0.031 to +0.035 for each seed, interval above 0.
  - Against the other candidates of its scale, its gain is +0.018 to +0.020, with intervals from −0.000 to +0.037.
  - It does not differ from q_l with the same r in top-1 success (p ≥ 0.40).
  - Its within-state AUC is 0.502–0.507, and 0.016–0.020 below q_l's.

## Closed loop, 10 tasks × 50 benchmark inits (phase 12)

### Runs

Per seed k ∈ {0, 1, 2}, with query / match seed k and no-ψ π seed k (π_k = `outputs/runs/goal_proposal_nopsi_seed{0,1,2b}/checkpoints/step_003000.pt`, Q_k = `outputs/runs/query_match_mvp_seed{k}/checkpoints/step_005000.pt`):

```bash
bash tmp/runlogs/s4mvp/launch_closed_loop.sh v11_qm_argmax_s<k> argmax <π_k> <Q_k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu>
bash tmp/runlogs/s4mvp/launch_closed_loop.sh v11_qm_random_s<k> uniform <π_k> <Q_k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu>
```

- **Processes**: one per task, 2 threads each. Each runs `scripts/run_closed_loop.py` with:
  - `--max-env-steps 300 --settle-steps 10 --seed 0 --trace --check-start-observation`;
  - `--checkpoint goal_instruction_contrast/step_005000.pt --state-table goal_instruction_contrast_step5000.npy`;
  - `--strategy argmax|uniform --candidate-source proposal --escalation 32 --effect-source predicted --dynamics-checkpoint goal_dynamics_perturb/step_003000.pt`;
  - `--match-target query_network --query-match-checkpoint Q_k`;
  - `--proposal-checkpoint π_k --proposal-scales 0.05,0.1,0.2 --proposal-dct 1 --proposal-seed 0`.
- **Runs**:
  - argmax: `outputs/closed_loop/v11_qm_argmax_s{0,1,2}/task<t>_argmax/`;
  - uniform control: `outputs/closed_loop/v11_qm_random_s{0,1,2}/task<t>_uniform/`, also scored, so its traces hold the matches.
- **Reference**: π's mean chunk, `v9_mean_nopsi_s{0,1,2}` (`docs/no_psi_baseline.md`). It uses the same env seed, the same extra reset of the start check and the same init order, so the episodes pair by (task, init).
- **Placement**: code `bf82639`, launched 2026-09-28 17:36 PDT. Seed 0 ran on node-62 CUDA 0/2, seed 1 on node-63 CUDA 0/1, seed 2 on node-65 CUDA 1/5, 20 processes per node, all GPUs idle at launch. Every process took 2.3–5.6 min.
- **Analysis**: R `analyze_phase12.py` (output `analyze_phase12.txt`) and `classify_closed_loop_failures.py` (R `failures_v11_qm_*.txt`; the v9 classifications are `tmp/runlogs/s3p29/v9_mean_nopsi_s*_failure_classification.txt`).

### Success

| Selection | seed 0 | seed 1 | seed 2 | mean | SD over seeds |
|---|---|---|---|---|---|
| argmax of m(r, q) | 0.846 | 0.842 | 0.840 | **0.843** | 0.003 |
| uniform among the same 32 candidates | 0.884 | 0.866 | 0.878 | **0.876** | 0.009 |
| π's mean chunk (v9) | 0.940 | 0.922 | 0.918 | **0.927** | 0.012 |

The binomial SD of one 500-episode run is 0.012–0.016.

Paired exact two-sided McNemar by (task, init). Each cell gives the episodes only the first solves / only the second solves, and p:

| Pair | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| argmax vs mean | 25 / 72, 1.9e-6 | 26 / 66, 3.7e-5 | 31 / 70, 1.3e-4 | 82 / 208, 8.7e-14 |
| argmax vs uniform | 47 / 66, 0.090 | 49 / 61, 0.29 | 46 / 65, 0.087 | 142 / 192, 0.0073 |
| uniform vs mean | 20 / 48, 9.1e-4 | 22 / 50, 0.0013 | 30 / 50, 0.033 | 72 / 148, 3.3e-7 |

Seed-level tests of the 9 run rates (as `docs/no_psi_baseline.md`):

- SD of the three version means 0.042, against a pooled SD within version of 0.009.
- One-way F(2, 6) = 69.8, p = 7e-5. The exact permutation over the 280 partitions into three groups of 3 gives p = 0.0036, the smallest attainable.
- Two-way version × seed: version F(2, 4) = 142, p = 1.9e-4; seed F(2, 4) = 4.1, p = 0.11.
- Pairwise exact permutation (20 assignments), one-sided in the observed direction. For each pair p = 0.05, the smallest attainable.
- Bootstrap over seeds (20,000 draws, seed 0):
  - argmax − mean −0.084 [−0.097, −0.075];
  - argmax − uniform −0.033 [−0.041, −0.024];
  - uniform − mean −0.051 [−0.064, −0.037].

### Per task

Success, mean over the three seeds (50 episodes per run; per-seed values in R `analyze_phase12.txt`):

| Task | argmax | uniform | mean (v9) |
|---|---|---|---|
| 0 open the middle drawer | 0.820 | 0.913 | 0.913 |
| 1 open the top drawer, bowl inside | 0.767 | 0.813 | 0.820 |
| 2 push the plate to the stove front | 0.807 | 0.893 | 0.973 |
| 3 bowl on the plate | 0.953 | 0.967 | 0.987 |
| 4 bowl on the stove | 0.967 | 0.947 | 0.980 |
| 5 bowl on top of the cabinet | 0.953 | 0.933 | 0.960 |
| 6 cream cheese in the bowl | 0.707 | 0.680 | 0.860 |
| 7 wine bottle on the rack | 0.647 | 0.780 | 0.873 |
| 8 wine bottle on the cabinet | 0.833 | 0.873 | 0.907 |
| 9 turn on the stove | 0.973 | 0.960 | 0.993 |

Failure categories, summed over the three seeds (1,500 episodes each; no episode is stuck):

| Selection | success | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|
| argmax | 1264 | 30 | 42 | 74 | 79 | 11 |
| uniform | 1314 | 19 | 32 | 70 | 55 | 10 |
| mean (v9) | 1390 | 8 | 13 | 51 | 28 | 10 |

Under argmax, "displaced without lift" is 29 in task 2 (the plate push; 4 under the mean chunk, over the three seeds) and 22 in task 7 (7 under the mean chunk).

### Decisions, choices and latency

| Run | decisions / env steps per episode | π's mean chunk chosen | chosen noisy scale 0.05 / 0.1 / 0.2 | decision latency, ms (encode, q, π, Δ) |
|---|---|---|---|---|
| argmax s0 / s1 / s2 | 14.5 / 141.0; 14.4 / 139.8; 14.4 / 140.3 | 1 of 7,240; 0 of 7,178; 0 of 7,205 | 0.020 / 0.194 / 0.787; 0.025 / 0.186 / 0.790; 0.026 / 0.183 / 0.792 | 62.8 (34.2, 3.9, 7.0, 14.7); 68.6; 60.5 |
| uniform s0 / s1 / s2 | 13.8 / 133.9; 14.1 / 137.0; 13.9 / 135.2 | 0.030; 0.031; 0.033 | 0.360 / 0.327 / 0.312; 0.362 / 0.323 / 0.316; 0.360 / 0.327 / 0.312 | 63.4; 68.8; 61.5 |
| mean (v9) s0 / s1 / s2 | 12.4 / 119.5; 12.8 / 124.3; 12.8 / 124.2 | 1 | – | 13.2 end to end (`docs/no_psi_baseline.md`, one process on the GPU) |

- **Latency**: the loop latencies were measured with 10 processes per GPU; `scripts/measure_efficiency.py` has not measured the argmax configuration alone.
- **Uniform draws**: 1/32 = 0.031 of the uniform choices fall on candidate 0, and the noisy scales cycle 11 / 10 / 10 over candidates 1–31.
- **Uniform scores**: under uniform, the best score exceeds the chosen one by 0.89–0.93 on average.

### Offline metrics next to the closed loop

| Seed | closed loop argmax / uniform / mean | HELDOUT top-1 of a⁺ | HELDOUT a⁺ beats every π-noise candidate | TRAIN a⁺ beats every π-noise candidate | sweep top 1 − random | sweep same-scale AUC |
|---|---|---|---|---|---|---|
| 0 | 0.846 / 0.884 / 0.940 | 0.255 | 0.307 | 0.462 | +0.035 | 0.520 |
| 1 | 0.842 / 0.866 / 0.922 | 0.231 | 0.289 | 0.454 | +0.031 | 0.513 |
| 2 | 0.840 / 0.878 / 0.918 | 0.238 | 0.294 | 0.435 | +0.031 | 0.519 |

### The frozen r among π's candidates

The within-observation distinguishability of the candidates under r uses the Phase-5 sufficiency metrics (`docs/interaction_representation.md`, "Effect distribution"): the per-state effective rank of the centered r, the effective rank of the pooled within-state covariance, and the within-state share Σ_s Σ_a ‖r_sa − r̄_s‖² / Σ ‖r − r̄‖². This r is the R1 Δ̂ of that page.

| Set (candidates per state) | per-state effective rank | pooled within-state effective rank | within-state share | source |
|---|---|---|---|---|
| sweep states, π's 32 candidates (1,084 states) | 2.23 (median 2.10) | 9.3 | 0.020 | R `diagnose_sweep_r.txt` |
| consistency_proposal, π's 32 candidates (129 HELDOUT states) | 1.85 | 9.1 | 0.017 | `docs/interaction_representation.md` |
| trial_resim_candidates, π's 32 candidates (12 states) | 1.87 | 3.4 | 0.036 | same |
| consistency_counterfactual, demo + stay + 12 perturbations (129 states) | 2.30 | 14.4 | 0.521 | same |
| training candidate sets {a⁺} ∪ N(o), HELDOUT (4,864 states) | 2.52 | 14.6 | 0.52 | acceptance above |

- **Pairwise cos** of the r of π's 32 candidates at a sweep state: median 0.9973, p10 0.964; within one offset scale, median 0.9980. The centered spread ‖r − r̄_o‖ is 9% of ‖r̄_o‖ (median).
- **Matches within a sweep state**: m ranges over 2.2 (median, s ≈ 31), and the best candidate leads the second by 0.15.
- **Linear readouts of the simulator outcome** on the sweep states (R `oracle_readout.py`, output `oracle_readout.txt`; a diagnostic that uses the outcomes, never used for training):
  - Method: cross-fitted L2 logistic regression (5 outer / 3 inner folds grouped by episode) on features centered within each state. The scores are ranked within each state; within-state AUC over 422 states, same-scale AUC over 413.

| Scores | within-state AUC [95% CI] | same-scale AUC [95% CI] |
|---|---|---|
| logistic on r (256) | 0.552 [0.534, 0.570] | 0.545 [0.525, 0.564] |
| logistic on the standardized chunk (70) | 0.541 [0.522, 0.560] | 0.537 [0.518, 0.555] |
| logistic on r and the chunk | 0.559 [0.540, 0.579] | 0.559 [0.538, 0.578] |
| logistic on the offset scale alone | 0.599 [0.581, 0.616] | 0.5 (constant within a scale) |
| m(r, q(o, l)), seed 0 | 0.502 [0.483, 0.521] | 0.520 [0.501, 0.539] |
| cos(r, q_l) | 0.522 [0.505, 0.540] | 0.526 [0.505, 0.545] |

### Reading

- **Argmax does not beat π's mean chunk, nor a uniform choice among the same candidates.**
  - Over three seeds it succeeds in 0.843, against 0.876 for the uniform choice and 0.927 for the mean chunk.
  - The paired McNemar is significant against the mean chunk at every seed (pooled 82 / 208), and pooled against the uniform choice (142 / 192, p = 0.0073).
  - Every seed-level test separates the three selections at the smallest attainable p.
- **Leaving π's mean chunk costs success in this protocol.** The uniform choice among π's candidates loses 0.051 against the mean chunk. The losses concentrate in the plate push (task 2), the cream cheese (task 6) and the wine bottle on the rack (task 7).
- **Argmax chooses the largest offsets.** In 79% of its decisions it takes a candidate of scale 0.2, against 31% under the uniform choice, and never π's mean chunk. Of its 50 extra failures over the uniform choice, 24 are displacements without lift.
- **r among π's candidates.** Among π's 32 candidates of one observation, r's within-state share is 0.017–0.036 and its per-state effective rank 1.9–2.2, with pairwise cos 0.997.
  - The training candidate sets, which include far smooth and other-task chunks, have share 0.52.
  - A linear readout of r fitted on the simulator outcomes ranks the successful candidates of a sweep state with within-state AUC 0.552 [0.534, 0.570]; the same readout of the chunk itself gives 0.541.
- **q against r.**
  - Given this r, a⁺ is separable at every observation (q = r̂⁺ ranks a⁺ first at 512 / 512 TRAIN observations).
  - q ranks a⁺ first at 0.23–0.26 of the HELDOUT observations and above every π-noise candidate at 0.29–0.31 (TRAIN 0.44–0.46).
  - On the sweep states its within-state AUC (0.502) is below the linear readout of r (0.552) and below q_l on the same r (0.522).
  - The overfitting check shows the loss floor that s ≤ 100 sets for π-noise candidates at cos(r⁺, r) ≈ 0.999.

## Code for phases 14–15

Code `81dc09f` (branch s4-p14). Checks and smokes on node-62 CUDA 2 (L40S), 2026-09-28 21:32–21:50 EDT; scripts and outputs in `tmp/runlogs/s4p14/` (R below).

### Smooth random candidates in the closed loop (framework 4)

- **`CandidateSource.SMOOTH`** (`--candidate-source smooth`): the K = `--escalation` candidates of a decision are `SmoothCandidates.sample`, i.e. `SmoothActionSampler` chunks around the dataset mean action:
  - J uniform in `--smooth-dct` (default 1,2,3) and per-entry RMS s uniform in `--smooth-scales` (default 0.5,1,2; standardized units, those of π's noise scales);
  - arm dims clipped to the controller's input range;
  - the gripper command switched from the last executed one (1.0 after the reset) with probability `--smooth-gripper-switch` (default 0.3), for the whole chunk.
  - No π and no mean chunk.
- **Determinism**: `numpy.random.default_rng([--smooth-seed, task, start code, init, decision])`, the words of `NoiseStream.episode`, so a decision's candidates depend only on its key.
- **Strategies**: `argmax` (requires `--effect-source predicted`) and `uniform` (predicted, or none). The uniform draw is the MVP's, keyed by decision, and independent of the candidates. The runner and the script refuse match, random and mean, a π checkpoint, observed effects and escalation schedules.
- **Records**: results carry `candidate_source` smooth and count executed smooth chunks in `proposal_chunks`; traces carry `chosen_source` smooth and the candidate index in `chosen_proposal`; recorded transitions have kind `rollout_smooth`. The decision line prints the chosen candidate's J, s and switch.
- **`--run-config FILE`** writes the run's `RolloutConfig`, which now holds `smooth_count`, `smooth_dct_coefficients`, `smooth_scales`, `smooth_gripper_switch_prob`, `smooth_seed` and `representation_kind`.

### R6 as the frozen r (framework 1)

- **`RepresentationKind`**: `stage_one_two` (the MVP r) or `frozen_pca_delta` (R6). Under R6, ψ is the fixed PCA projection of the frozen DINOv2 pooled feature and the effect head is the one trained on it, both from one fixed-encoder interaction checkpoint.
- **`ActionRepresentation(encoder, dynamics)`** takes a `StateTrunk` or a `FrozenPcaEncoder`. `ActionRepresentationLoader.load(kind, stage_one, effect checkpoint, action cache, device)` builds either. For R6 it refuses a run whose feature cache is not the stage one's or whose action cache is not the given one.
- **Config**: optional `representation_kind`. Under `frozen_pca_delta`, `dynamics_checkpoint` is the interaction checkpoint, and `stage_one_checkpoint` provides only the feature cache, split and instruction table.
- **Tables**: `build_query_match_tables.py --fixed-encoder-checkpoint CKPT` computes `psi.npy` with R6's encoder. The manifest records `representation_kind`, `fixed_encoder_checkpoint` and `fixed_encoder_step`. The trainer refuses tables whose ψ source differs from the config's r; the MVP tables, without these keys, count as `stage_one_two`.
- **Trainer**: the frozen check includes the encoder's buffers.
- **Closed loop**: `--representation-kind frozen_pca_delta --dynamics-checkpoint CKPT` feeds the effect head the PCA ψ of the current frames (`ClosedLoopRunner` `effect_encoder`). It requires `--match-target query_network`, since the other targets score in the loop's ψ space. `--checkpoint` stays the loop's stage one (the fixed stage one), and a query / match checkpoint must have the loop's kind.
- **Rescoring and the acceptance evaluation**: they read r through `QueryMatchCheckpointStore.representation`, so they accept R6 unchanged.

### Options for phase 15 (framework 3)

| Field | Default | Effect |
|---|---|---|
| `candidates.exclude_within_d_min` | true | false: every source contributes its first draws as drawn, with no exclusion within d_min, no resampling and no fills |
| `match_max_scale` | 100 | the clamp of s (`Match(max_scale)`; not stored in the state dict) |

Both fields, and `representation_kind`, are optional in the JSON and omitted when at their default, so the MVP configs and the configs stored in the MVP checkpoints read and write the same text.

### Checks

Existing paths, new code against the base code (`ac5951e`) and the reference runs:

| Check | Result |
|---|---|
| closed loop, argmax: the `v11_qm_argmax_s0` command, task 3, inits 0–4 | new and base code: 5 / 5 episodes identical to `v11_qm_argmax_s0` in results (wall time excepted) and in every trace field |
| closed loop, π mean: the `v9_mean_nopsi_s0` command, task 0, inits 0–4 | the same, 5 / 5 against `v9_mean_nopsi_s0` |
| closed loop, uniform: the `v11_qm_random_s0` command, task 3, inits 0–4, two runs per code | every episode has the reference's chosen candidates, success and env steps; see below |
| `CandidateSelector` (every strategy × target, K 1–512, float32 / float16, both signs, 72 calls each), `Match` (8 logit values), the three MVP config texts and the stored step-5000 config round trip, `SameObservationCandidates` (512 observations × seeds 0 / 1000 × draw keys 0 / 1 / 4999), r and m of the MVP seed-0 checkpoint (8 observations × 33 chunks) | 122 / 122 arrays and 5 / 5 texts bit-identical (R `compare_dumps.txt`) |
| training: `mvp_seed0` with 40 updates, evaluation every 20 on 128 + 128 observations | metrics (timings excepted), evaluations, tables and frozen checks, and both checkpoints bit-identical (R `compare_training.txt`) |

**Uniform runs** (R `compare_scores.txt`):

- In the first new-code run, init 2 first differs from the reference in a logged score at decision 3, where the scene is identical (14.70814 against 14.70780). The end effector then differs by 1.2e-6 m at decision 4 and by 2.1 mm at decision 8, and the scores by up to 0.74. The chosen candidates, gripper commands and steps are identical at all 8 decisions, and so are success and env steps.
- In the repeat, init 2 is identical to the reference.
- Init 3 differs from the reference only in the final scene (2e-8) and one score (6e-5), in all four new- and base-code runs.
- Argmax and mean were identical in every field.

Unit checks (R `check_units.txt`, CPU):

| Check | Result |
|---|---|
| `SmoothCandidates` against `SmoothActionSampler.sample(K, g, (task, start, init, decision))` | raw, J, s, switch equal; standardized = `ActionCache.standardize(raw)` |
| determinism | same key equal; another decision, init, task, start or seed differs; K = 8 is the prefix of K = 32; a 1-word stream and gripper 0.5 refused |
| 64,000 chunks (2 × 1,000 decisions × 32) | arm entries in [−1, 1]; gripper constant per chunk and switched ⇔ ≠ gripper_now; switch rate 0.3000; J shares 0.340 / 0.332 / 0.328; s shares 0.330 / 0.334 / 0.337 |
| per-entry RMS by s without clipping (range ±1e9), J = 1 / 2 / 3 | s 0.5: 0.499 / 0.502 / 0.501; s 1: 1.007 / 1.001 / 0.999; s 2: 1.993 / 1.993 / 2.010; energy above J ≤ 1.9e-15 |
| clipped share of arm entries, s = 0.5 / 1 / 2 | 0.0001 / 0.013 / 0.110 |
| uniform choice over 1,000 decisions | counts 23–46 of 32 slots (χ² 25.9, 31 dof); chosen s shares 0.34 / 0.34 / 0.33 |
| runner and script refusals | smooth with match, random, mean, observed, none under argmax, a π checkpoint, escalation 32,64, K ≠ count; R6 without the query-network target; a query / match checkpoint of the other kind; an R6 checkpoint read as stage two (R `check_units.txt`, `check_refusals.txt`) |
| R6 through the loader against `FrozenPcaDeltaModel` (16 observations × 8 chunks) | ψ and r max \|Δ\| 0; the runner's effects with the effect encoder = `ActionRepresentation.of` exactly |
| `exclude_within_d_min` false, 1,024 TRAIN and 1,024 HELDOUT observations | considered = 12 / 8 / 12, excluded 0, fills 0; each source's slots equal its first draws; kept within d_min of a⁺: π noise 0.525 (TRAIN), 0.077 (HELDOUT), other sources ≤ 0.003 |
| `Match(1000)` | s = 500 at logit log 500, 1000 at logit 10; the MVP step-5000 match loads with s 31.3145 under either clamp |

### Smokes

R6 pipeline with the phase-9 smoke checkpoint `s4p9_final_frozen_pca_delta_smoke` step 500:

- **Tables**: `outputs/query_match/p14_smoke_tables_r6` took 121 s. Every table except `psi.npy` is bit-identical to `tables_mvp`: observations, the three π mean chunks, neighbours and d_min 1.9007. `psi.npy` matches the encoder recomputed on the CPU within 4.8e-6 (R `compare_tables.txt`).
- **Training**: q on the R6 smoke r, `outputs/runs/p14_smoke_qm_r6`, 300 updates at 2.2 / s.
  - InfoNCE 3.49 → 2.64.
  - Frozen check: 66 tensors (2 encoder buffers and the effect head), change 0, grad None at every update; tables check 1.4e-6.
- **Rescoring**: `QueryMatchScorer` equals the closed loop's computation on 8 sweep states, with max |Δm| 0 and argmax equal 8 / 8, for the R6 smoke q and for the MVP q (R `check_scorer_r6.txt`).
- **Phase-15 options**: `mvp_seed0` with both options, 40 updates. Its training candidates were kept as drawn (considered 6,912 / 4,608 / 6,912 for 576 observations, 0 excluded, 0 fills).

Closed-loop smokes, tasks 3 and 6, inits 0–1, K 32, smooth seed 0 (R `smoke_mvp_summary.txt`, `smoke_r6_summary.txt`; outputs `outputs/closed_loop/p14_smoke/`):

| Run | r / q | Success | Checks |
|---|---|---|---|
| smooth, argmax | MVP r, MVP q seed 0 | 0 / 4 | every chosen chunk smooth and the best-scoring |
| smooth, uniform, scored / unscored | MVP r, MVP q seed 0 | 0 / 4, 0 / 4 | scored and unscored runs execute the same episodes |
| smooth, argmax | R6 smoke r, its q | 0 / 4 | every choice the best-scoring |
| smooth, uniform | R6 smoke r, its q | 0 / 4 | same chosen candidates and scenes as the MVP-r uniform run |
| π seed 0, argmax | R6 smoke r, its q | 3 / 4 | every choice the best-scoring |
| π seed 0, uniform | R6 smoke r, its q | 2 / 4 | – |

The smooth sampler adds 0.6–4.4 ms per decision (episode means, with 8–12 processes on the GPU).

### G2 power pilot

Setup (R `launch_pilot.sh`, `pilot_summary.txt`; outputs `outputs/closed_loop/p14_pilot_smooth_uniform_s0/task<t>_uniform/`):

- Uniform choice among 32 smooth random candidates, smooth seed 0.
- 10 tasks × benchmark inits 0–4, with the env seed, step limit and start check of `v9_mean_nopsi_s0`.
- The choice does not depend on the scores; the candidates were scored with the MVP r and MVP q seed 0.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform, smooth candidates | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/50 |
| π mean (`v9_mean_nopsi_s0`, same inits) | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 49/50 |

- Every pilot episode ran the full 300 steps.
- A movable object was grasped at some decision in 8 of 50 episodes. A movable object moved more than 1 cm, or a fixture joint more than 0.05, in 33 of 50.
- The uniform success rate, 0.000, is below the 5% precondition of G2. From this starting point the random-sampler evaluation has no statistical power.

### Commands for the phase-14 runs

Launchers in `tmp/runlogs/s4p14/phase14/`. Code W is the s4-p14 worktree, or the main checkout once merged. R6 is Phase 13's `outputs/runs/interaction_frozen_pca_delta_nine_stores_seed0/checkpoints/step_020000.pt`; override `R6` if Phase 13 ends at another step. The configs `configs/query_match/r6_seed{0,1,2}.json` are `mvp_seed{0,1,2}.json` with `representation_kind` frozen_pca_delta, that checkpoint, `tables_r6` and `query_match_r6_seed<k>`.

```bash
P=$CEAM_ROOT/tmp/runlogs/s4p14/phase14   # on the compute node, GPU checked before each launch
bash $P/build_tables_r6.sh <cuda> <commit>                           # outputs/query_match/tables_r6
bash $P/train_r6.sh r6_seed<k> <cuda>                                # k = 0, 1, 2: outputs/runs/query_match_r6_seed<k>
bash $P/launch_closed_loop.sh v12_r6_argmax_s<k> argmax pi <k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu> <commit>
bash $P/launch_closed_loop.sh v12_r6_random_s<k> uniform pi <k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu> <commit>
bash $P/launch_closed_loop.sh v12_r6_smooth_argmax_s<k> argmax smooth <k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu> <commit>
bash $P/launch_closed_loop.sh v12_r6_smooth_random_s<k> uniform smooth <k> 0-49 0,1,2,3,4,5,6,7,8,9 <gpu>,<gpu> <commit>
```

The π runs pair with `v9_mean_nopsi_s<k>` by (task, init).

## R6 as r: within-observation distinguishability (phase 13)

R6 is a candidate implementation of r(o, a) (framework clause 1):

- ψ_R6(o) is a fixed PCA-256 of the frozen DINOv2 pooled feature: per view the mean of the frame's 64 tokens, the two views concatenated (1,536-d); `FrozenPcaEncoder`, `docs/interaction_training.md`, "Fixed encoder (R6)".
- r(o, a) = Δ̂(o, ψ_R6(o), a) is the output of an effect head trained with L_pred alone to regress z = ψ_R6(o′) − ψ_R6(o).

The regression target supervises r; it is an engineering choice, not a framework requirement. The check below is clause 2 on the action side: whether the r of different candidates at one observation can be told apart. It was measured before any q is trained on this r.

### PCA

`outputs/feature_pca/r6_nine_stores/pca.npz`: `scripts/fit_feature_pca.py --config configs/interaction/frozen_pca_delta_nine_stores.json --threads 4`, code `76da757`, node-62 CPU, 115 s (log `tmp/runlogs/s4p13/logs/fit_pca.log`).

| Store | Held-out rule | TRAIN frames | HELDOUT frames |
|---|---|---|---|
| `demo_renders` | shards `heldout_*` | 47,601 | 5,294 |
| `demo_perturbations` (data A) | none | 37,440 | 0 |
| `rollouts_r0` | sampled starts 80–99 | 19,532 | 4,670 |
| `resim_phase26_mean_{fixed,flow,flow_full_sgd}` | inits 40–49 | 6,024 / 5,201 / 5,209 | 1,336 / 1,306 / 1,273 |
| `resim_phase26_sweep_{fixed,flow,flow_full_sgd}` | inits 40–49 | 15,048 / 7,524 / 7,524 | 1,980 / 1,980 / 1,782 |
| total | | 151,103 | |

No frame is in both splits.

| Components k | 1 | 8 | 16 | 32 | 64 | 128 | **256** | 512 | 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| variance explained | 0.155 | 0.476 | 0.606 | 0.713 | 0.801 | 0.874 | **0.9336** | 0.975 | 0.997 |

- The covariance has effective rank 79.4. The 256th eigenvalue is 0.0841, 1.85e-3 of the first (45.49).
- Self-check on the fitting frames:
  - projected variance against eigenvalue, max relative error 3.4e-15;
  - largest projected covariance off the diagonal, 4.3e-16 of the top eigenvalue;
  - max |VᵀV − I| 2.7e-15;
  - eigen residual 2.7e-16 of the top eigenvalue.

### Effect head

- **Config**: `configs/interaction/frozen_pca_delta_nine_stores.json`. It is `frozen_pca_delta.json` with:
  - the nine sources of the PCA table, in that order;
  - this PCA as the fixed encoder;
  - its own run dir.
  - Everything else is unchanged: weights 1/256, 0, 0 (L_pred per dim only); AdamW lr 3e-4, weight decay 0.05, warmup 500, cosine to 0 at 20,000; batch 256; clip 1.0; bf16 autocast; seed 0; checkpoints every 2,000.
  - Monitor: K0–K3 counted from step 4,000 on 4,096 pooled HELDOUT transitions (126 stay), K3 on the `demo_renders` HELDOUT trajectories; K4 every 500 updates.
- **Data**: TRAIN pool 138,051 transitions, HELDOUT 18,517. Trained parameters: 13,315,840 (dynamics head only).
- **Run**: `outputs/runs/interaction_frozen_pca_delta_nine_stores_seed0`, code `76da757`, node-62 CUDA 0 (idle at launch), 2026-09-28 18:22–19:15 PDT.
  - 2,899 s for 20,000 updates: 6.9 updates/s with the monitor, median 7.7 without it.
  - Peak GPU memory 3.18 GiB; the nine stores are held in memory.
  - Exit status 0. No criterion held at any checkpoint, and no `monitor_stop.md` was written.

Batch values, means over the logged lines of each window (`metrics.jsonl`; the loss is L_pred / 256):

| Updates | L_pred / 256 | cos(Δ̂, z) | median ‖Δ̂ − z‖/‖z‖ | median ‖z‖ | median ‖Δ̂‖ |
|---|---|---|---|---|---|
| 0 | 0.664 | 0 | 1.000 | 12.50 | 0 |
| 1–500 | 0.453 | 0.500 | 0.847 | 12.37 | 5.51 |
| 1,500–2,000 | 0.210 | 0.753 | 0.583 | 12.22 | 10.20 |
| 3,500–4,000 | 0.158 | 0.812 | 0.498 | 12.22 | 10.56 |
| 7,500–8,000 | 0.106 | 0.865 | 0.407 | 12.27 | 11.23 |
| 11,500–12,000 | 0.080 | 0.889 | 0.361 | 12.22 | 11.40 |
| 19,500–19,999 | 0.060 | 0.917 | 0.310 | 12.29 | 11.65 |

L_pred / 256 per source over updates 19,500–19,999:

| Source | L_pred / 256 |
|---|---|
| demo_renders | 0.052 |
| data A | 0.071 |
| rollouts_r0 | 0.068 |
| resim mean fixed / flow / flow_full_sgd | 0.064 / 0.063 / 0.063 |
| resim sweep fixed / flow / flow_full_sgd | 0.056 / 0.049 / 0.052 |

Monitor on the 4,096 pooled HELDOUT transitions (`monitor.jsonl`; ψ is fixed, so its rank and K3 do not change: ψ rank 48.1, top share 0.173, K3 0.343 against R0 0.372):

| Step | Δ̂ rank / top share | median ‖Δ̂‖/‖z‖ | stay ‖Δ̂‖ vs non-stay ‖z‖ | R² | SIGReg(Δ̂) | cos(Δ̂, z) | K0 K1 K2 K3 |
|---|---|---|---|---|---|---|---|
| 2,000 (not counted) | 63.1 / 0.109 | 0.814 | 2.25 vs 12.10 | 0.561 | 0.102 | 0.666 | F F F F |
| 4,000 | 69.7 / 0.104 | 0.831 | 2.77 vs 12.10 | 0.584 | 0.094 | 0.683 | F F F F |
| 8,000 | 75.5 / 0.100 | 0.884 | 3.14 vs 12.10 | 0.583 | 0.076 | 0.691 | F F F F |
| 12,000 | 80.2 / 0.096 | 0.908 | 3.63 vs 12.10 | 0.570 | 0.068 | 0.690 | F F F F |
| 16,000 | 83.0 / 0.093 | 0.921 | 3.88 vs 12.10 | 0.555 | 0.064 | 0.684 | F F F F |
| 20,000 | 83.5 / 0.093 | 0.923 | 3.93 vs 12.10 | 0.552 | 0.063 | 0.683 | F F F F |

- **K4**: σ_ψ stays 1.0375, the step-0 value, as it must for a fixed encoder. Mean ‖Δ̂‖ rises from 7.74 at step 500 to 10.94 at step 20,000 (ratio 1.41); neither K4a nor K4b holds.
- **Held-out R²**: it peaks at 0.584 at step 4,000 and ends at 0.552. The batch cos rises to 0.917, while the held-out cos stays at 0.68–0.69.
- **Final diagnostics**: Var(z)/Var(ψ) 0.577. Direction correlation: eigenvalue-weighted 0.721, min 0.185. Stay ‖Δ̂‖ quartiles 2.70 / 3.93 / 6.32, against stay ‖z‖ 3.47 / 6.03 / 7.73.

Held-out prediction per source at step 20,000, all HELDOUT transitions (`outputs/analysis/query_match/r6_gate/heldout.json`, script `tmp/runlogs/s4p13/heldout_r6.py`):

- Each representation is measured in its own ψ space: R6 against z_R6 (median ‖z‖ 10.5–12.8), the MVP r against the fixed stage one's z (median ‖z‖ 2.5–4.0).
- The MVP effect head was trained on demonstrations and data A only, so every row is held out for it.

| Held-out part | n | R6: R² / cos mean / median rel. error | MVP r: R² / cos mean / median rel. error |
|---|---|---|---|
| demo_renders `heldout_*` | 4,864 | 0.632 / 0.775 / 0.598 | 0.854 / 0.924 / 0.244 |
| rollouts_r0 starts 80–99 | 4,470 | 0.450 / 0.599 / 0.798 | 0.315 / 0.484 / 0.924 |
| — of which rollout_stay | 533 | −0.181 / 0.233 / 1.122 | 0.009 / 0.086 / 1.321 |
| resim mean fixed / flow / flow_full_sgd, inits 40–49 | 1,236 / 1,206 / 1,173 | 0.671–0.684 / 0.781–0.782 / 0.534–0.543 | 0.772–0.843 / 0.848–0.880 / 0.236–0.250 |
| resim sweep fixed | 1,920 | 0.461 / 0.617 / 0.797 | 0.555 / 0.562 / 0.822 |
| resim sweep flow | 1,920 | 0.511 / 0.640 / 0.754 | 0.819 / 0.633 / 0.503 |
| resim sweep flow_full_sgd | 1,728 | 0.449 / 0.609 / 0.780 | 0.770 / 0.699 / 0.648 |

### Candidate sets

- **π's noisy candidates** at the 1,084 sweep states of "Offline: the branch test's sweep states", in two versions:
  - **(a) stored**: the 32 candidates the branch test executed. They come from each name's ψ π (`goal_proposal`, `goal_proposal_flow`, `goal_proposal_flow_full_sgd`): mean + 31 noisy, scales 0.05/0.1/0.2 cycled, DCT J = 1, proposal seed 0, stream `NoiseStream.episode(task, 0, init, d)`.
    - These are the candidates of the MVP r's 0.020 above.
    - Their o′ is in the re-simulated sweep stores, so their true effects exist.
  - **(b) no-ψ π seed 0**: the mean chunk of `goal_proposal_nopsi_seed0` step 3000 at the stored observation, with the task's original instruction, plus 31 noisy candidates on the same stream, seed and scales, with no gripper flip and controller range [−1, 1].
    - These are the candidates the MVP closed loop of seed 0 would propose at these states. No o′ exists for them.
    - Check: `ProposalCandidates.sample` of each stored candidate 0 on this stream reproduces the 32 stored candidates within 1e-6 in 1,077 of 1,084 states (max 0.0175 in the others). So (b) has the offsets of (a).
    - The mean chunks of (a) and (b) differ by a median 3.12 in the standardized 70-d chunk space (p90 6.21).
- **Smooth random candidates**: `counterfactual_eval_heldout`, 258 HELDOUT demonstration states × 32 `SmoothActionSampler` chunks (J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch 0.3), with o′.
- **How r and z are computed**:
  - For the sweep states, r comes from the stored observation tokens (`branch_rescore/<name>/observations`), as in the MVP. z = ψ(o′) − ψ(o) comes from the o and o′ frames of `resim_phase26_sweep_<name>`, with the candidate index read from the o′ sidecar.
  - Store checks: the store chunks equal the stored candidates exactly. ψ(o) of the store frame and of the stored observation differ by at most 1.8e-4 relative.
  - For the smooth random set, r and z come from the store's o and o′.
- **The MVP r**: `goal_dynamics_perturb` step 3000 on `goal_instruction_contrast` step 5000. On (a) it reproduces the MVP diagnostic exactly: share 0.019709, per-state rank 2.2283, pooled rank 9.3264, median cos 0.99733.
- **True effects in the MVP's space**: z_MVP = ψ(o′) − ψ(o) of that stage one.

### Metrics

- **Within-state share**: Σ_s Σ_a ‖r_sa − r̄_s‖² / Σ ‖r − r̄‖². It depends on the spread between states, so it compares representations only within one set and one space.
- **Within-state effective rank**: per state, exp of the entropy of the normalized eigenvalues of the 32 centered vectors, averaged over states; the median is also given.
- **Pairwise cos**: the cos of the raw vectors over the 32 × 31 ordered pairs of each state, pooled over states; mean and median.
  - The plan's reference value for the MVP r, 0.997, is the median of this distribution; its mean is 0.984.
- Of predicted against true effects in the same space:
  - within-state R² = 1 − Σ‖(r − r̄_s) − (z − z̄_s)‖² / Σ‖z − z̄_s‖²;
  - Spearman correlation of the 496 pairwise distances per state (mean over states);
  - ratio of the mean centered norms of r and z per state (median over states).

Script `tmp/runlogs/s4p13/gate_r6.py` (float64 metrics; models in float32 without autocast), node-62 CUDA 0, 97 s. Output `outputs/analysis/query_match/r6_gate/gate.json` and `effects.pt` (every r and z).

### Pre-registered gate (step 20,000)

Thresholds: share ≥ 0.05, per-state effective rank ≥ 4, mean pairwise cos ≤ 0.99; a set passes when all three hold. The phase stops without training q only if both the π set and the smooth random set fail.

| Candidates | r | share | rank mean (median) | pairwise cos mean (median) | passes |
|---|---|---|---|---|---|
| π, (a) stored, 1,084 states | R6 | **0.070** | **4.18** (4.14) | **0.911** (0.960) | **yes** |
| π, (b) no-ψ π seed 0, 1,084 states | R6 | **0.053** | **4.51** (4.45) | **0.932** (0.967) | **yes** |
| smooth random, 258 states | R6 | **0.512** | **11.00** (11.06) | **0.520** (0.533) | **yes** |
| π, (a) | MVP r | 0.020 | 2.23 (2.10) | 0.984 (0.997) | no |
| π, (b) | MVP r | 0.013 | 2.25 (2.08) | 0.986 (0.998) | no |
| smooth random | MVP r | 0.679 | 3.41 (3.02) | 0.400 (0.493) | no (rank) |

**R6 passes on both candidate kinds.** With the median cos in place of the mean it also passes (≤ 0.967). Per name, the π sets have share 0.062–0.076 (a) and 0.051–0.057 (b), rank 4.14–4.53, cos mean 0.901–0.939.

The 1,084 sweep states split by whether their stored candidates' transitions trained the effect head (inits 0–39 are TRAIN in the re-simulated sweep stores):

| Sweep states | n | (a) share / rank / cos mean | (b) share / rank / cos mean | (a) within-state R² of r against z / centered cos |
|---|---|---|---|---|
| inits 0–39, trained on | 910 | 0.073 / 4.06 / 0.905 | 0.055 / 4.45 / 0.930 | 0.399 / 0.517 |
| inits 40–49, held out | 174 | 0.050 / 4.83 / 0.939 | 0.046 / 4.85 / 0.943 | 0.006 / 0.180 |

The same quantities at earlier checkpoints of the run (`tmp/runlogs/s4p13/gate_step{002000,010000}/gate.json`):

| Step | (a) share / rank / cos mean | (b), inits 40–49: share / rank | (a) within-state R², all / inits 40–49 | smooth random share / rank / cos mean |
|---|---|---|---|---|
| 2,000 | 0.019 / 3.61 / 0.978 | 0.022 / 3.59 | 0.098 / 0.065 | 0.450 / 7.38 / 0.623 |
| 10,000 | 0.045 / 3.88 / 0.941 | 0.039 / 4.40 | 0.247 / 0.049 | 0.495 / 9.81 / 0.553 |
| 20,000 | 0.070 / 4.18 / 0.911 | 0.046 / 4.85 | 0.336 / 0.006 | 0.512 / 11.00 / 0.520 |

### True effects against r

π's stored candidates (a), 1,084 states:

| Vectors | share | rank mean | pooled within-state rank | pairwise cos mean / median |
|---|---|---|---|---|
| r, R6 | 0.070 | 4.18 | 79.9 | 0.911 / 0.960 |
| **z in R6 space** | **0.175** | **11.75** | 127.2 | **0.768** / 0.828 |
| r, MVP | 0.020 | 2.23 | 9.3 | 0.984 / 0.997 |
| z in the MVP's ψ space | 0.051 | 2.83 | 14.0 | 0.866 / 0.986 |

- R6: within-state R² of r against z 0.336, pairwise-distance Spearman 0.823, spread ratio 0.505.
  - Inits 0–39: 0.399 / 0.837 / 0.516.
  - Inits 40–49: 0.006 / 0.751 / 0.444.
- MVP: 0.195 / 0.639 / 0.691; on inits 40–49, within-state R² 0.189.
- z in R6 space on inits 40–49 alone: share 0.162, rank 12.11, cos mean 0.781.

Smooth random candidates, 258 HELDOUT states:

| Vectors | share | rank mean | pooled within-state rank | pairwise cos mean / median |
|---|---|---|---|---|
| r, R6 | 0.512 | 11.00 | 76.0 | 0.520 / 0.533 |
| z in R6 space | 0.643 | 13.75 | 92.6 | 0.359 / 0.360 |
| r, MVP | 0.679 | 3.41 | 15.0 | 0.400 / 0.493 |
| z in the MVP's ψ space | 0.724 | 3.13 | 14.9 | 0.282 / 0.345 |

- R6: within-state R² 0.182, Spearman 0.724, spread ratio 0.802, centered cos 0.467.
- MVP: 0.349 / 0.609 / 0.678 / 0.631.

### Reading

- **The candidates' effects differ, and the MVP's ψ space hides it.** In R6 space the true effects of π's 32 candidates at a sweep state have share 0.175, per-state rank 11.8 and mean cos 0.77.
  - In the fixed stage one's ψ space the same effects have share 0.051, rank 2.8 and median cos 0.986.
  - The MVP r (0.020, 2.2, 0.997) sits close to its own space's true effects.
- **The R6 r passes the gate** on π's stored candidates, on π-seed-0's candidates and on the smooth random candidates.
  - Among π's candidates it carries about half of the true within-state spread: spread ratio 0.505, share 0.070 against 0.175.
- **On held-out sweep states the spread does not follow the true effects.**
  - Most of the 1,084 gate states (910, inits 0–39) had their stored candidates' transitions in the effect head's training data. There r's within-state variation follows the true effects: within-state R² 0.40.
  - On the 174 states of held-out inits, r has the gate's size of spread (share 0.050 and 0.046, rank 4.8), but its within-state R² against the true effects is 0.006 (centered cos 0.18). The ranking of pairwise distances still agrees (Spearman 0.75).
  - Over training, this R² on held-out inits falls from 0.065 (step 2,000) to 0.006, while the share of set (b) there rises from 0.022 to 0.046.
- **On the smooth random candidates of held-out demonstration states**, R6's r is well separated: share 0.51, rank 11.0, cos 0.52. Its within-state R² against the true effects is 0.18, and the MVP r's is 0.35 in its own space.

## Random-sampler evaluation from held-out demonstration states (phase 18)

From the benchmark inits, a uniform choice among 32 smooth random candidates never succeeded (G2 power pilot, 0 / 50). This evaluation starts instead late in the HELDOUT demonstrations, where fewer steps remain. Only the starting point changes (framework clause 4): the sampler, the two selections and the success check are those of the closed loop above.

### Protocol (fixed before the runs)

- **Starts**: each of the 43 HELDOUT LIBERO-Goal episodes (the `EpisodeIndex` split of the fixed stage one: heldout fraction 0.1, seed 0). Each starts at three levels f ∈ {0.6, 0.75, 0.9}, at frame t_s = round(f · (T − 1)), with T the episode's frame count. All three levels are reported.
- **Step limit**: H = min(300, max(50, 2 · (T − t_s))) env steps from the start. Success is LIBERO's check (the env's done flag).
- **Candidates**: 32 smooth random chunks per decision (J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch 0.3), without π. They are keyed by (sampler seed k, episode, level, decision), so both selections face the same candidates at the same decision index.
- **Selections**: argmax of m(r, q) with the query-network target, with r = R6 (`interaction_frozen_pca_delta_nine_stores_seed0` step 20,000); and a uniform choice among the same 32. Query seed k is paired with sampler seed k, k = 0, 1, 2. Two query versions are evaluated on this R6:
  - (a) Phase 14, the MVP recipe (`query_match_r6_seed<k>`);
  - (b) Phase 15 (`query_match_r6_fix_seed<k>`).
- **Instruction**: the episode's task text (variants[0]), as in the MVP.
- **Report**: per level, the success rates, the relative lift (p_argmax − p_uniform) / p_uniform, and the exact McNemar, paired by (episode, level), per seed and pooled. A level where the uniform choice succeeds in less than 5% is not a main conclusion.

**Every start is a held-out state for both learned parts of the selection.**

- The query was trained on the 43,701 TRAIN demonstration observations only.
- R6's PCA was fitted on the TRAIN frames of its nine stores. Its effect head was trained on their TRAIN transitions. The `demo_renders` `heldout_*` shards were held out, and data A was drawn from TRAIN states only (`docs/dynamics_data.md`).
- Neither saw a frame of the 43 HELDOUT demonstrations.

### Code: the demo start (framework 4)

- **`DemoStartSchedule`** (`ceam/policy/demo_start.py`) gives the starts of an episode.
  - Levels are exact thousandths: `--demo-levels 0.6,0.75,0.9` gives 600 / 750 / 900.
  - t_s = round(f · (T − 1)) is computed in integers. Ties round up: f = 0.75 has 8 ties among the 43 episodes, f = 0.9 has 7.
  - H = min(`--max-env-steps`, max(50, 2 · (T − t_s))), with `--max-env-steps` 300.
- **`DemoStartPoint`** holds (episode, level, T, t_s, H). Its `start_index` = episode · 1000 + level in thousandths (e.g. 128750).
  - This is the init word of the decision's streams. The smooth candidates of decision d use `default_rng([smooth seed k, task, 2, start_index, d])`, and the uniform draw uses `default_rng([--seed, UNIFORM_WORD, task, 2, start_index, d])`.
  - It is also the `init_index` of the results and trace lines.
- **`EpisodeStart.DEMO`** (start code 2) and **`ClosedLoopRunner.run_demo(restorer, point)`** start an episode here.
  - The runner restores frame t_s with `DemoStateRestorer.restore`, with no settle step, and runs the unchanged decision loop.
  - The loop's step limit is H, and the gripper command in force is the restored frame's (the dataset's a_{t_s − 1}). A benchmark or sampled start keeps `max_env_steps` and the reset gripper command.
  - `EpisodeRun.start_proprio` holds the raw proprio of the start observation.
- **`run_closed_loop.py --start demo`**:
  - `--inits` are dataset episode ids. Each must be a HELDOUT episode of the task, or the script refuses it.
  - Per episode, `DemoStarts` replays the dataset episode once into a `DemoStateRestorer`, from its raw demo in `--raw-dir` with 10 settle steps. It then runs the levels in order.
  - After each start it compares the start observation with the `demo_renders` frame (episode, t_s) (`DemoRenderFrames`: the sidecar's (episode, t) → store frame). It writes a `DemoStartRecord` line to `<out>/demo_starts.jsonl`: the start, H, the gripper in force, the raw demo, the store frame, the max |Δ| of the raw proprio, and per view the float64 cosine and the max |Δ| of the frozen tokens.
  - The run config records `demo_levels`, `demo_renders` and `raw_dir`. `--check-start-observation` and `--record-transitions` are refused with `--start demo`.
- Selection, candidates, scoring and the success check are the code of phases 11 and 14, unchanged.

### Checks

Code `dd1b660` (branch s4-p18), and `6e4a652` for the float64 token cosine of the start check. Scripts and outputs are in `tmp/runlogs/s4p18/` (S below). Checks ran on node-62 CUDA 0 / 2 and node-63 CUDA 0, 2026-09-28 22:52–23:10 EDT.

Existing paths, new code against the reference runs (S `compare_checks.txt`, `compare_mean_repeat.txt`; outputs `outputs/closed_loop/p18_checks/`):

| Check | Result |
|---|---|
| argmax: the `v11_qm_argmax_s0` command, task 3, inits 0–4 | 5 / 5 episodes identical to `v11_qm_argmax_s0` in results (wall time excepted) and in every trace field |
| uniform: the `v11_qm_random_s0` command, task 3, inits 0–4 | 5 / 5 identical in chosen candidates, success and env steps. Init 3 differs in one logged score (22.18474 against 22.18480) and in the mean selected score, as in the phase-14 check of the base code |
| mean: the `v9_mean_nopsi_s0` command, task 0, inits 0–4 | first run: 4 / 5 identical; init 2 ended at 112 env steps against 116 (the end effector differs by 2.5e-7 m at decision 2). Repeat: 5 / 5 identical. Base code `e0df3d9`, same GPU: 5 / 5 in results, init 2 differs by 7.6e-7 m at decision 12 |
| smooth argmax on R6: the `p14_smoke` `smooth_argmax_r6` command (Phase-9 smoke R6, its 300-step q), task 3, inits 0–1 | 2 / 2 identical in results and traces |
| smooth uniform, unscored: the `p14_smoke` `smooth_uniform_unscored` command, task 3, inits 0–1 | 2 / 2 identical in results and traces |

- The mean-chunk difference also appears with the base code, so it is run-to-run variation on the shared GPU, not a code change.
- Benchmark and sampled starts pass `max_env_steps` and the reset gripper command to the loop as before; the loop body is otherwise unchanged.

Unit checks (S `check_units.txt`, CPU):

| Check | Result |
|---|---|
| t_s and H against exact rational half-up rounding, T = 1 … 800, the three levels | 0 mismatches |
| level parsing | 0.6005, 1.0, 0 and a repeated level refused |
| start codes | benchmark 0, sampled 1, demo 2 |
| smooth candidates under demo keys | the same key gives identical chunks; another level, episode, decision, sampler seed or start code changes 32 / 32 chunks; gripper_now changes only the gripper entries |
| uniform draw under 2,000 demo keys | scored = unscored 2,000 / 2,000; 49–75 draws per slot of 32, expected 62.5 (χ² 24.6, 31 dof) |
| refusals (S `check_refusals.txt`) | `--check-start-observation`, `--record-transitions`, the bad levels, a TRAIN episode and another task's episode are each refused with their own message |

**Smoke** (S `smoke_summary.txt`; outputs `outputs/closed_loop/p18_smoke/`): task 3, HELDOUT episodes 128 and 131, the three levels, smooth seed 0. Three modes: argmax with R6 and the Phase-14 q of seed 0 at step 1,000; uniform scored with the same q; uniform unscored.

- **Restore**: every start's raw proprio equals the `demo_renders` frame's exactly (max |Δ| 0), and so do its frozen tokens (max |Δ| 0).
- **Step limit**: every failed episode ran exactly H env steps (86, 54, 72 and 50), and no episode exceeded H. The first decision observes the restored frame at env step 0.
- **Selection**:
  - argmax took the best-scoring candidate at every decision;
  - the scored and unscored uniform runs executed the same candidates, gripper commands, steps and scenes at every decision.
- **Candidates**: the executed candidates' J, s and switch, recomputed from their keys, give the executed gripper command at every decision.

### Starts

The 43 HELDOUT episodes have T = 79–251 frames (median 105). Their episode ids per task (S `heldout_episodes.txt`; every start in S `starts.tsv`) are:

- task 0: 4, 18, 26, 32
- task 1: 58, 69, 74
- task 2: 76, 91, 96, 107
- task 3: 128, 131, 140, 148, 158
- task 4: 170, 180, 189, 206, 212
- task 5: 223, 225, 236, 238, 248
- task 6: 281, 288, 290, 293
- task 7: 303, 307, 318, 336
- task 8: 341, 345, 352, 367
- task 9: 384, 397, 398, 412, 430

| Level f | H: min / median / mean / max | H = 50 (the floor) | starts where LIBERO's goal already holds at t_s |
|---|---|---|---|
| 0.6 | 64 / 86 / 99.8 / 202 | 0 | 0 |
| 0.75 | 50 / 54 / 64.0 / 126 | 13 | 1 (episode 107, task 2) |
| 0.9 | 50 / 50 / 50.0 / 52 | 42 | 23 (tasks 2–9: 2, 4, 5, 2, 2, 1, 2, 5) |

- The goal flags come from the `demo_renders` sidecar (`goal_holds` at (episode, t_s)). The initial scenes of the runs' traces give the same counts, identical in all nine runs.
- No start reaches the 300 cap.

**More than half of the f = 0.9 starts come after the goal is reached.** In 23 of its 43 starts, LIBERO's goal already holds at t_s.

- Such a start usually counts as a success after its first env step under either selection. In those runs, 57, 57 and 59 of the 69 goal-at-start episodes succeed at step 1 (uniform, (a), (b)); all but 2, 0 and 1 succeed within H.
- They raise both success rates, add almost no discordant pairs, and shrink the relative lift.
- The pre-registered f = 0.9 results below include them. The not-pre-registered table after them excludes the starts where the goal holds at t_s: 23 at f = 0.9, 1 at f = 0.75, none at f = 0.6.

### Runs

One process per (run, task), which runs the task's HELDOUT episodes at the three levels. Every process runs `scripts/run_closed_loop.py` with:

- `--task-index t --inits <its HELDOUT episode ids> --start demo --demo-levels 0.6,0.75,0.9 --max-env-steps 300 --settle-steps 10 --seed 0 --trace`;
- `--checkpoint goal_instruction_contrast/step_005000.pt --state-table goal_instruction_contrast_step5000.npy` (the loop's stage one, as in the MVP; π is not used);
- `--candidate-source smooth --escalation 32 --smooth-dct 1,2,3 --smooth-scales 0.5,1.0,2.0 --smooth-gripper-switch 0.3 --smooth-seed k`;
- uniform: `--strategy uniform --effect-source none`;
- argmax: `--strategy argmax --representation-kind frozen_pca_delta --effect-source predicted --dynamics-checkpoint interaction_frozen_pca_delta_nine_stores_seed0/checkpoints/step_020000.pt --match-target query_network --query-match-checkpoint Q`, where Q is `query_match_r6_seed<k>` (a) or `query_match_r6_fix_seed<k>` (b), `checkpoints/step_005000.pt`.

| Run | Selection | q | Smooth seed |
|---|---|---|---|
| `p18_demo_uniform_s<k>` | uniform, unscored | none | k |
| `p18_demo_argmax_r6_s<k>` | argmax | (a) `query_match_r6_seed<k>` step 5,000 | k |
| `p18_demo_argmax_r6fix_s<k>` | argmax | (b) `query_match_r6_fix_seed<k>` step 5,000 | k |

- **One uniform run per seed serves both versions.** The uniform choice uses neither r nor q, and its draw depends only on the decision's key. So `p18_demo_uniform_s<k>` is the control of both `p18_demo_argmax_r6_s<k>` and `p18_demo_argmax_r6fix_s<k>`.
  - It runs unscored. The smoke shows that a scored run executes the same episodes.
- **Uniform draw**: `--seed 0` in every run, as in phases 12 and 14. The candidates differ between seeds through the smooth seed k.
- **Pairing**: starts pair by `init_index` (episode, level) within a seed. At the same decision index the two selections face the same 32 candidates, since the sampler seed, episode, level and decision are the same.
- **Launchers**: `tmp/runlogs/s4p18/` `make_jobs.sh` writes a run's 10 jobs to `queues/<run>.txt`. `worker.sh` workers take jobs by atomic `mkdir` claims, and before each job wait for ≥ 6,000 MiB free and no other project's process on their GPU. `start_workers.sh` starts them; `launch_queue.sh` is the entry for `scripts/cluster_run.sh`, the gate of the argmax launches, which ran `gpu_guard.sh` first.
- **Placement**: node-63 CUDA 0 and 6 (idle at launch), 4 workers per card, at most 1,882 MiB per process. Code `6e4a652`.
  - Uniform: 22:58–23:03 EDT.
  - Argmax, as each q checkpoint became final (the Phase 14 / 15 trainings on node-62, which ended at 00:08–00:35 EDT): (b) seeds 1 and 2 and (a) seed 0 at 00:13–00:17; (b) seed 0 and (a) seed 1 at 00:29–00:32; (a) seed 2 at 00:35–00:38 EDT.
  - A process took 27–47 s, of which the episode's replay into the restorer took 2.3–6.8 s (median 3.5 s).
- **Checks over the 1,161 episodes** (S `analysis_runs_v2.txt`):
  - every start's raw proprio equals its `demo_renders` frame's exactly;
  - the frozen tokens are bitwise equal in 1,157. The other 4 are one start (episode 303, f = 0.9) in 4 of the 9 runs: wrist token max |Δ| 0.0078, 1 − cos 2.6e-8;
  - no episode exceeds H, and every failure ran exactly H env steps;
  - argmax took the best-scoring candidate at all 4,136 of its decisions;
  - the executed gripper command matches the chosen candidate's, recomputed from its key, at every decision.

### Results (pre-registered)

Success rates over the 43 starts of a level. Lift = (p_argmax − p_uniform) / p_uniform. McNemar is exact and two-sided on the pairs (episode, level): discordant pairs "argmax only / uniform only", then p. "Pooled" sums the three seeds (129 pairs), as in phase 12. (a) is the Phase-14 q (MVP recipe), (b) the Phase-15 q (no exclusion within d_min, s ≤ 1000); both on the current R6.

**f = 0.6.** Uniform succeeds in 0.016 pooled, below 5%: this level is not a main conclusion.

| | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| uniform | 0.023 | 0.023 | 0.000 | 0.016 |
| argmax (a) | 0.233 | 0.093 | 0.233 | 0.186 |
| argmax (b) | 0.163 | 0.186 | 0.186 | 0.178 |
| lift (a) | +9.0 | +3.0 | undefined (uniform 0) | +11.0 |
| lift (b) | +6.0 | +7.0 | undefined (uniform 0) | +10.5 |
| McNemar (a) | 9 / 0, p 0.0039 | 4 / 1, p 0.38 | 10 / 0, p 0.0020 | 23 / 1, p 3.0e-6 |
| McNemar (b) | 7 / 1, p 0.070 | 8 / 1, p 0.039 | 8 / 0, p 0.0078 | 23 / 2, p 1.9e-5 |

**f = 0.75.**

| | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| uniform | 0.209 | 0.279 | 0.279 | 0.256 |
| argmax (a) | 0.442 | 0.488 | 0.512 | 0.481 |
| argmax (b) | 0.488 | 0.512 | 0.442 | 0.481 |
| lift (a) | +1.11 | +0.75 | +0.83 | +0.88 |
| lift (b) | +1.33 | +0.83 | +0.58 | +0.88 |
| McNemar (a) | 11 / 1, p 0.0063 | 11 / 2, p 0.022 | 11 / 1, p 0.0063 | 33 / 4, p 1.1e-6 |
| McNemar (b) | 13 / 1, p 0.0018 | 13 / 3, p 0.021 | 12 / 5, p 0.14 | 38 / 9, p 2.5e-5 |

**f = 0.9.** 23 of the 43 starts begin with the goal already holding; see "Starts".

| | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| uniform | 0.814 | 0.860 | 0.814 | 0.829 |
| argmax (a) | 0.884 | 0.930 | 0.884 | 0.899 |
| argmax (b) | 0.953 | 0.860 | 0.953 | 0.922 |
| lift (a) | +0.086 | +0.081 | +0.086 | +0.084 |
| lift (b) | +0.171 | 0.000 | +0.171 | +0.112 |
| McNemar (a) | 4 / 1, p 0.38 | 4 / 1, p 0.38 | 5 / 2, p 0.45 | 13 / 4, p 0.049 |
| McNemar (b) | 6 / 0, p 0.031 | 3 / 3, p 1 | 6 / 0, p 0.031 | 15 / 3, p 0.0075 |

In the form of G2 (pooled McNemar p < 0.05 in favour of argmax, and argmax above uniform at every seed):

| Level | (a) | (b) |
|---|---|---|
| 0.6 | holds, but uniform < 5% | holds, but uniform < 5% |
| 0.75 | holds | holds |
| 0.9 | holds (p 0.049) | fails: seed 1 ties |

G2 itself was defined on the benchmark inits, where it had no power.

### Starts where the goal does not hold at t_s (not pre-registered)

The same comparison without the starts whose goal already holds at t_s. f = 0.6 keeps all 43 starts and is unchanged. f = 0.75 loses 1 start (42 left); f = 0.9 loses 23 (20 left).

| Level | selection | seed 0 | seed 1 | seed 2 | pooled | lift pooled | McNemar pooled |
|---|---|---|---|---|---|---|---|
| 0.75 | uniform | 0.190 | 0.262 | 0.262 | 0.238 | | |
| 0.75 | argmax (a) | 0.429 | 0.476 | 0.500 | 0.468 | +0.97 | 33 / 4, p 1.1e-6 |
| 0.75 | argmax (b) | 0.476 | 0.500 | 0.429 | 0.468 | +0.97 | 38 / 9, p 2.5e-5 |
| 0.9 | uniform | 0.650 | 0.700 | 0.650 | 0.667 | | |
| 0.9 | argmax (a) | 0.750 | 0.850 | 0.750 | 0.783 | +0.18 | 11 / 4, p 0.12 |
| 0.9 | argmax (b) | 0.900 | 0.750 | 0.900 | 0.850 | +0.28 | 13 / 2, p 0.0074 |

- Per seed at f = 0.9: (a) 3 / 1 (p 0.63), 4 / 1 (0.38), 4 / 2 (0.69); (b) 5 / 0 (0.063), 3 / 2 (1), 5 / 0 (0.063).
- At f = 0.75 the per-seed discordant pairs equal the pre-registered ones: the excluded start succeeds under every selection.

**(a) against (b)**, same seed and start, pooled: at f = 0.6, 12 / 11 (p 1); at f = 0.75, 14 / 14 (p 1); at f = 0.9, 4 / 7 (p 0.55). No per-seed p is below 0.2.

### Per task

Successes pooled over the three seeds, (a) / (b) / uniform, out of 3 × the task's HELDOUT episodes:

| Task | n | f = 0.6 | f = 0.75 | f = 0.9 |
|---|---|---|---|---|
| 0 open the middle drawer | 12 | 1 / 0 / 0 | 1 / 0 / 0 | 6 / 8 / 6 |
| 1 open the top drawer, bowl inside | 9 | 0 / 0 / 0 | 0 / 0 / 0 | 8 / 9 / 5 |
| 2 push the plate to the stove front | 12 | 0 / 0 / 0 | 3 / 4 / 3 | 10 / 8 / 7 |
| 3 bowl on the plate | 15 | 3 / 5 / 0 | 8 / 7 / 5 | 15 / 14 / 14 |
| 4 bowl on the stove | 15 | 4 / 5 / 0 | 12 / 13 / 7 | 15 / 15 / 14 |
| 5 bowl on top of the cabinet | 15 | 2 / 2 / 0 | 11 / 11 / 4 | 15 / 15 / 14 |
| 6 cream cheese in the bowl | 12 | 3 / 3 / 0 | 5 / 5 / 4 | 12 / 12 / 12 |
| 7 wine bottle on the rack | 12 | 1 / 0 / 0 | 5 / 6 / 0 | 9 / 11 / 8 |
| 8 wine bottle on the cabinet | 12 | 2 / 1 / 0 | 5 / 6 / 3 | 11 / 12 / 12 |
| 9 turn on the stove | 15 | 8 / 7 / 2 | 12 / 10 / 7 | 15 / 15 / 15 |

### Where the choices fall

The executed candidates' J, s and gripper switch, recomputed from their keys, over all decisions of the three seeds. The uniform choice shows the candidates' own shares.

| Selection | f | decisions | J = 1 / 2 / 3 | s = 0.5 / 1 / 2 | gripper switched |
|---|---|---|---|---|---|
| uniform | 0.6 | 1,345 | 0.324 / 0.344 / 0.332 | 0.309 / 0.349 / 0.341 | 0.285 |
| uniform | 0.75 | 746 | 0.328 / 0.338 / 0.334 | 0.328 / 0.335 / 0.336 | 0.312 |
| uniform | 0.9 | 234 | 0.346 / 0.312 / 0.342 | 0.368 / 0.321 / 0.312 | 0.338 |
| argmax (a) | 0.6 | 1,256 | 0.368 / 0.327 / 0.305 | 0.283 / 0.369 / 0.349 | 0.249 |
| argmax (a) | 0.75 | 634 | 0.347 / 0.315 / 0.338 | 0.287 / 0.353 / 0.360 | 0.248 |
| argmax (a) | 0.9 | 192 | 0.406 / 0.281 / 0.312 | 0.354 / 0.339 / 0.307 | 0.328 |
| argmax (b) | 0.6 | 1,256 | 0.368 / 0.346 / 0.287 | 0.303 / 0.312 / 0.385 | 0.249 |
| argmax (b) | 0.75 | 616 | 0.383 / 0.323 / 0.294 | 0.302 / 0.338 / 0.360 | 0.237 |
| argmax (b) | 0.9 | 182 | 0.379 / 0.242 / 0.379 | 0.324 / 0.368 / 0.308 | 0.330 |

- Argmax takes J = 1 somewhat more often: 0.35–0.41, against 0.32–0.35 under the uniform choice. It switches the gripper less often at f = 0.6 and 0.75: 0.24–0.25, against 0.29–0.31.
- Its scale shares stay within 0.28–0.39. Among π's candidates, the MVP argmax took the largest scale in 79% of decisions; here there is no such preference.

Mean env steps per episode (the mean H is 99.8, 64.0 and 50.0 at the three levels):

| Level | uniform | argmax (a) | argmax (b) |
|---|---|---|---|
| 0.6 | 99.3 | 92.6 | 92.9 |
| 0.75 | 54.0 | 45.4 | 44.2 |
| 0.9 | 11.4 | 7.7 | 6.8 |

### Reading

- **At f = 0.75, argmax of m(r, q) nearly doubles the success of a uniform choice among the same smooth random candidates.**
  - Both q versions succeed in 0.481 of 129 starts pooled, against 0.256; lift +0.88.
  - Pooled McNemar: 33 / 4 (p 1.1e-6) for (a) and 38 / 9 (p 2.5e-5) for (b). Argmax is higher at every seed, with per-seed p < 0.05 in 5 of the 6 comparisons.
  - Without the one start whose goal holds at t_s, the result is the same (0.468 against 0.238).
- **At f = 0.6 the uniform choice almost never succeeds** (2 of 129), so by the protocol this level is not a main conclusion. Argmax succeeds in 24 (a) and 23 (b) of the 129; pooled McNemar 23 / 1 and 23 / 2.
- **At f = 0.9 most starts are already done.**
  - Uniform succeeds in 0.829, argmax in 0.899 (a) and 0.922 (b); lift +0.08 and +0.11, pooled p 0.049 and 0.0075.
  - Over the 20 starts where the goal does not yet hold (not pre-registered), the lift is +0.18 (a, p 0.12) and +0.28 (b, p 0.0074).
- **The two q versions do not differ in this evaluation**: pooled discordant pairs 12 / 11, 14 / 14 and 4 / 7.
- **The gain is spread over the tasks.** At f = 0.75 it is largest for the bowl on the stove, the bowl on top of the cabinet and the wine bottle on the rack (12–13 / 11 / 5–6 against 7 / 4 / 0). The two drawer tasks stay at or near 0 at f = 0.6 and 0.75 under every selection.
- **Scope.** Every start is a held-out state for q and for R6: a HELDOUT demonstration frame that neither the query's training nor R6's PCA and effect head saw. The candidates are random smooth chunks, not π's.

## Phase 14 and 15 results on R6

r is R6: `interaction_frozen_pca_delta_nine_stores_seed0` step 20,000 (phase 13), `--representation-kind frozen_pca_delta`. Two versions of q were trained with seeds 0/1/2:

| Version | Config | Difference from the MVP recipe | Runs |
|---|---|---|---|
| r6 (phase 14) | `configs/query_match/r6_seed<k>.json` | r only | `outputs/runs/query_match_r6_seed<k>` |
| r6fix (phase 15) | `configs/query_match/r6_fix_seed<k>.json` | r; `candidates.exclude_within_d_min` false; `match_max_scale` 1000 | `outputs/runs/query_match_r6_fix_seed<k>` |

- **Tables**: `outputs/query_match/tables_r6` (67 s). They equal `tables_mvp` apart from `psi.npy`; d_min is 1.9007.
- **Code**: `a3371c6` for the trainings and closed loops, and `472c2b2` for the offline evaluation (`evaluate_query_match.py acceptance --candidate-exclusion`) and the task-7 rerun. `run_closed_loop.py` is the same in both.
- **Trainings**: node-62 CUDA 0 and 2, three per GPU, 2026-09-28 22:25 to 29 00:35 EDT, at 0.68–0.84 updates/s. The r frozen check is clean in all six runs (66 tensors, change 0, grad None at every update).
- **Aborted starts**: the two seed-2 runs were first started on node-66 CUDA 2 and stopped after 18 s, because another user's job started on that GPU. Their run directories hold only `config.json`, kept as `query_match_r6_{seed2,fix_seed2}_aborted_nnmc66_2225edt`.

Scripts, logs and analysis outputs are in `tmp/runlogs/s4p14/phase14/` (R14 below).

### Closed loop, π's candidates, 10 tasks × 50 benchmark inits

**Runs**: `v12_r6_argmax_s<k>` and `v12_r6fix_argmax_s<k>`, argmax of m(r, q) among π seed k's 32 candidates, with the settings of `v11_qm_argmax_s<k>` (`launch_closed_loop.sh`, `closed_loop_waves.sh`). They ran on node-62 CUDA 0/2, 00:09–00:40 EDT, one run of 10 processes launched at a time.

**Controls**:

- `v11_qm_random_s<k>`: the uniform choice among the same candidates. It does not use the scores, so it serves every r and q of seed k.
- `v9_mean_nopsi_s<k>`: π's mean chunk.
- The MVP argmax `v11_qm_argmax_s<k>` is a reference.

**Pairing** (R14 `pairing.txt`):

- Every v12 process log names the same π checkpoint and the same candidate settings as the v11 control (count 32, scales 0.05 / 0.1 / 0.2, DCT J 1, proposal seed 0, no flip).
- In every episode of the first processes, the scene after the reset and at decision 1 equals the control's, and so do the image means that decision 1 prints. The noise stream is keyed by (task, start, init, decision), so decision 1 draws the same candidates as the control.

**Refused episodes.** `LiberoTaskEnv.observe` refuses a near-black frame, and the process stops. This happened twice, in task 7 (the wine bottle on the rack), as the wrist image darkened over a few decisions (mean 52.7 → 6.0 → 1.2):

- `v12_r6_argmax_s0` during init 31;
- `v12_r6fix_argmax_s0` during init 38.

A second process ran the remaining inits into `task7_argmax_rest` (`rerun_task7.sh`) and repeated the last completed init. It does not reproduce the first process:

- the repeated episode differs from the first run of it (both fail);
- in every one of its episodes, the decision-1 image means differ from the control's, although the scene is equal.

The tables therefore use the first processes only: task 7 inits 31–49 of `v12_r6_argmax_s0` and 38–49 of `v12_r6fix_argmax_s0` are left out of those runs and of their pairs (481 and 488 episodes).

- **Sensitivity** (R14 `closed_loop_analysis_with_rerun.txt`): adding the second processes' episodes gives 0.884 / 0.870 for these two runs, 0.900 / 0.873 as means over seeds, and r6 against uniform pooled 158 / 122, p = 0.036.
- Counting the refused episode as a failure changes the r6 mean from 0.900 to 0.899.

**Success**:

| Selection | seed 0 | seed 1 | seed 2 | mean | SD over seeds |
|---|---|---|---|---|---|
| argmax, r6 | 0.886 (481) | 0.904 | 0.912 | **0.901** | 0.014 |
| argmax, r6fix | 0.871 (488) | 0.886 | 0.864 | **0.874** | 0.011 |
| uniform among the same candidates (v11) | 0.884 | 0.866 | 0.878 | **0.876** | 0.009 |
| π's mean chunk (v9) | 0.940 | 0.922 | 0.918 | **0.927** | 0.012 |
| argmax, MVP r and q (v11) | 0.846 | 0.842 | 0.840 | 0.843 | 0.003 |

The binomial SD of one 500-episode run is 0.012–0.016.

By benchmark init: R6's effect head was trained on re-simulated transitions of the phase-26 records of inits 0–39 (training states); inits 40–49 are held out, 100 episodes per run.

| Selection | inits 0–39: seed 0 / 1 / 2, mean | inits 40–49: seed 0 / 1 / 2, mean |
|---|---|---|
| argmax, r6 | 0.880 (391) / 0.905 / 0.907, **0.897** | 0.911 (90) / 0.900 / 0.930, **0.914** |
| argmax, r6fix | 0.867 (398) / 0.892 / 0.855, **0.871** | 0.889 (90) / 0.860 / 0.900, **0.883** |
| uniform (v11) | 0.887 / 0.868 / 0.877, **0.877** | 0.870 / 0.860 / 0.880, **0.870** |
| π's mean chunk (v9) | 0.945 / 0.930 / 0.910, **0.928** | 0.920 / 0.890 / 0.950, **0.920** |
| argmax, MVP (v11) | 0.843 / 0.833 / 0.838, 0.838 | 0.860 / 0.880 / 0.850, 0.863 |

Paired exact two-sided McNemar by (task, init). Each cell gives the episodes only the first solves / only the second solves, and p:

| Pair | seed 0 | seed 1 | seed 2 | pooled | pooled, inits 0–39 | pooled, inits 40–49 |
|---|---|---|---|---|---|---|
| r6 vs uniform | 47 / 45, 0.92 | 55 / 36, 0.059 | 55 / 38, 0.097 | **157 / 119, 0.026** | 122 / 97, 0.10 | 35 / 22, 0.11 |
| r6 vs mean | 24 / 52, 0.0018 | 32 / 41, 0.35 | 37 / 40, 0.82 | **93 / 133, 0.0093** | 74 / 112, 0.0065 | 19 / 21, 0.88 |
| r6fix vs uniform | 47 / 53, 0.62 | 53 / 43, 0.36 | 50 / 57, 0.56 | 150 / 153, 0.91 | 118 / 125, 0.70 | 32 / 28, 0.70 |
| r6fix vs mean | 24 / 59, 1.6e-4 | 31 / 49, 0.057 | 32 / 59, 0.0061 | 87 / 167, 5.8e-7 | 67 / 136, 1.5e-6 | 20 / 31, 0.16 |
| r6fix vs r6 | 40 / 45, 0.67 | 35 / 44, 0.37 | 34 / 58, 0.016 | 109 / 147, 0.021 | 89 / 118, 0.051 | 20 / 29, 0.25 |
| r6 vs MVP argmax | 59 / 40, 0.070 | 65 / 34, 0.0024 | 62 / 26, 1.6e-4 | 186 / 100, 4.1e-7 | 151 / 80, 3.5e-6 | 35 / 20, 0.058 |
| r6fix vs MVP argmax | 58 / 45, 0.24 | 63 / 41, 0.039 | 60 / 48, 0.29 | 181 / 134, 0.0094 | 147 / 106, 0.012 | 34 / 28, 0.53 |
| uniform vs mean | 20 / 48, 9.1e-4 | 22 / 50, 0.0013 | 30 / 50, 0.033 | 72 / 148, 3.3e-7 | 59 / 120, 6.0e-6 | 13 / 28, 0.028 |

The per-seed McNemar of each split is in R14 `closed_loop_analysis.txt`.

Relative gain over the uniform choice, (p_argmax − p_uniform) / p_uniform:

| Selection | per seed (all inits) | pooled: all / inits 0–39 / inits 40–49 |
|---|---|---|
| r6 | +0.002, +0.044, +0.039 | +0.028 / +0.023 / +0.050 |
| r6fix | −0.015, +0.023, −0.016 | −0.003 / −0.007 / +0.015 |
| MVP argmax | −0.043, −0.028, −0.043 | −0.038 / −0.046 / −0.008 |

Seed-level tests of the run rates (as `docs/no_psi_baseline.md`; all inits):

- **{r6, uniform, mean}**:
  - one-way F(2, 6) = 14.3, p = 0.0052; exact permutation over the 280 partitions p = 0.0036; two-way version F(2, 4) = 10.4, p = 0.026.
  - r6 − uniform +0.025: exact permutation one-sided p = 0.05, the smallest attainable; bootstrap over seeds [+0.010, +0.039].
  - r6 − mean −0.026: p = 0.05; bootstrap [−0.042, −0.010].
- **{r6fix, uniform, mean}**:
  - one-way F = 23.2, p = 0.0015; permutation p = 0.032.
  - r6fix − uniform −0.002: permutation p ≥ 0.45 either way; bootstrap [−0.016, +0.011].
- **Argmax versions**:
  - r6fix − r6 −0.027, bootstrap [−0.043, −0.011];
  - r6 − MVP argmax +0.058, bootstrap [+0.044, +0.069], permutation p = 0.05.
- On inits 40–49 alone, the three-way one-way F for r6 is 5.43, p = 0.045 (permutation p = 0.068).

Per task, mean over the three seeds (the refused and rerun episodes of task 7 left out):

| Task | r6 | r6fix | uniform | mean (v9) | MVP argmax |
|---|---|---|---|---|---|
| 0 | 0.907 | 0.913 | 0.913 | 0.913 | 0.820 |
| 1 | 0.773 | 0.760 | 0.813 | 0.820 | 0.767 |
| 2 | 0.927 | 0.867 | 0.893 | 0.973 | 0.807 |
| 3 | 0.967 | 0.973 | 0.967 | 0.987 | 0.953 |
| 4 | 0.993 | 0.953 | 0.947 | 0.980 | 0.967 |
| 5 | 0.973 | 0.960 | 0.933 | 0.960 | 0.953 |
| 6 | 0.813 | 0.727 | 0.680 | 0.860 | 0.707 |
| 7 | 0.754 | 0.699 | 0.780 | 0.873 | 0.647 |
| 8 | 0.893 | 0.867 | 0.873 | 0.907 | 0.833 |
| 9 | 0.987 | 1.000 | 0.960 | 0.993 | 0.973 |

**Where argmax picks** (R14 `closed_loop_analysis.txt`, section 4). π's mean chunk is chosen in 0–0.05% of the decisions of every argmax run, against 2.9–3.5% under the uniform choice. Among the noisy candidates, the shares of scales 0.05 / 0.1 / 0.2:

| Runs | all inits | inits 0–39 | inits 40–49 |
|---|---|---|---|
| r6, three seeds | 0.048–0.055 / 0.246–0.259 / 0.692–0.700 | 0.047–0.058 / 0.247–0.260 / 0.687–0.701 | 0.043–0.056 / 0.240–0.256 / 0.688–0.717 |
| r6fix | 0.032–0.036 / 0.212–0.213 / 0.752–0.756 | 0.033–0.037 / 0.209–0.215 / 0.748–0.758 | 0.028–0.031 / 0.200–0.223 / 0.748–0.770 |
| MVP argmax | 0.020–0.026 / 0.183–0.194 / 0.787–0.792 | – | – |
| uniform | 0.360–0.362 / 0.323–0.327 / 0.312–0.316 | – | – |

- Episodes average 12.8–13.7 decisions for the R6 runs, 14.4 for the MVP argmax and 12.4–12.8 for π's mean chunk.

**Failure categories** (`scripts/classify_closed_loop_failures.py`, R14 `failures_<run>.txt`), summed over the three seeds. Seed-0 task 7 is classified over both processes (50 episodes, init 30 twice, the refused init absent).

| Selection | success | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|
| argmax, r6 | 1349 | 11 | 23 | 55 | 56 | 6 |
| argmax, r6fix | 1309 | 16 | 26 | 70 | 74 | 5 |
| uniform (v11) | 1314 | 19 | 32 | 70 | 55 | 10 |
| mean (v9) | 1390 | 8 | 13 | 51 | 28 | 10 |
| MVP argmax (v11) | 1264 | 30 | 42 | 74 | 79 | 11 |

### Training

Batch metrics over updates 4500–4999 and the periodic evaluation at update 5000 (R14 `offline_summary.txt`):

| Run | InfoNCE / top-1 / rank of a⁺ / s | π noise excluded; fills per observation | eval top-1 TRAIN / HELDOUT; HELDOUT InfoNCE |
|---|---|---|---|
| r6 seeds 0 / 1 / 2 | 1.128 / 1.111 / 1.164; 0.749–0.790; 1.38–1.43; 32.0–32.1 | 0.685–0.703; 1.80–2.01 | 0.751–0.766 / 0.221–0.233; 3.23–3.33 |
| r6fix seeds 0 / 1 / 2 | 1.519 / 1.535 / 1.573; 0.614–0.649; 1.77–1.91; 32.4–32.5 | 0; 0 | 0.607–0.655 / 0.217–0.253; 3.16–3.29 |
| MVP seeds 0 / 1 / 2 | 2.101–2.112; 0.359–0.372; 3.16–3.38; 31.1–31.3 | 0.685–0.703; 1.80–2.01 | 0.364–0.447 / 0.248–0.250; 2.97–3.13 |

- **The clamp of s does not bind.** s settles at 32.0–32.5 in both versions, below the MVP clamp of 100 and far below 1000. The two versions differ in practice only in the exclusion.

### Offline

**Sweep states of the branch test** (1,084 states, 181 episodes):

- Scores: `rescore_branch_candidates.py score --query-match qm_<r6|r6fix>_seed<k>`, `outputs/analysis/branch_rescore/<name>/scores/`.
- Pooled report: `outputs/analysis/query_match/sweep/r6_pooled.{md,json}`.
- Split by init, the mean-chunk comparison and the pick shares: R14 `sweep_extra.txt`.
- The states' candidate 0 is the ψ π's mean chunk; it succeeds in 0.014 of the states (the states are decisions of failed π-mean episodes). 910 states come from inits 0–39 and 174 from inits 40–49.
- 95% intervals come from a bootstrap over episodes.

| Scoring | best | best − all | best − same scale | best − mean chunk | within-state AUC | same-scale AUC | top: scale 0.05 / 0.1 / 0.2 |
|---|---|---|---|---|---|---|---|
| q_l, MVP r | 0.210 | +0.040 [+0.023, +0.057] | +0.026 [+0.009, +0.044] | +0.196 [+0.173, +0.221] | 0.522 [0.505, 0.540] | 0.526 [0.505, 0.545] | 0.012 / 0.144 / 0.844 |
| MVP q, mean of seeds | 0.202 | +0.032 | +0.018 | +0.188 | 0.504 | 0.517 | 0.017 / 0.181 / 0.802 |
| r6 seed 0 | 0.198 | +0.028 [+0.012, +0.046] | +0.016 [−0.001, +0.033] | +0.185 [+0.161, +0.208] | 0.494 [0.476, 0.513] | 0.514 [0.496, 0.532] | 0.059 / 0.227 / 0.714 |
| r6 seed 1 | 0.200 | +0.030 [+0.013, +0.048] | +0.014 [−0.004, +0.032] | +0.186 [+0.164, +0.211] | 0.498 [0.480, 0.517] | 0.513 [0.494, 0.534] | 0.049 / 0.203 / 0.747 |
| r6 seed 2 | 0.194 | +0.023 [+0.009, +0.037] | +0.008 [−0.009, +0.022] | +0.180 [+0.160, +0.201] | 0.499 [0.481, 0.518] | 0.520 [0.501, 0.538] | 0.069 / 0.218 / 0.713 |
| r6fix seed 0 | 0.197 | +0.027 [+0.011, +0.044] | +0.013 [−0.004, +0.030] | +0.184 [+0.161, +0.208] | 0.500 [0.482, 0.519] | 0.513 [0.493, 0.531] | 0.047 / 0.207 / 0.746 |
| r6fix seed 1 | 0.204 | +0.034 [+0.018, +0.050] | +0.018 [+0.001, +0.036] | +0.190 [+0.167, +0.215] | 0.508 [0.491, 0.525] | 0.523 [0.505, 0.542] | 0.054 / 0.188 / 0.758 |
| r6fix seed 2 | 0.195 | +0.024 [+0.010, +0.040] | +0.012 [−0.003, +0.028] | +0.181 [+0.157, +0.206] | 0.499 [0.484, 0.516] | 0.511 [0.494, 0.528] | 0.043 / 0.202 / 0.755 |

By init (mean of the three seeds; the per-seed rows with intervals are in `sweep_extra.txt`):

| Scoring | inits 0–39: best − all / best − same scale / same-scale AUC | inits 40–49: best − all / best − same scale / within-state AUC / same-scale AUC |
|---|---|---|
| q_l, MVP r | +0.037 / +0.024 / 0.527 | +0.057 / +0.039 / 0.511 / 0.517 |
| MVP q | +0.033 / +0.020 / 0.514 | +0.029 / +0.010 / 0.517 / 0.532 |
| r6 | +0.029 / +0.015 / 0.518 | +0.019 / +0.002 / 0.476 / 0.507 |
| r6fix | +0.027 / +0.014 / 0.517 | +0.034 / +0.017 / 0.497 / 0.510 |

- Paired against the MVP q of the same seed, no difference of an R6 q has an interval excluding 0, in any subset (best success, best − same scale, both AUCs).
- On inits 40–49, the r6 within-state AUC is 0.462–0.494 (intervals up to 0.495–0.536).

**Same-observation acceptance** (`evaluate_query_match.py acceptance`, held-out candidate seed; outputs `outputs/analysis/query_match/acceptance/`):

- Observations: 4,864 TRAIN (seeded choice) and all 4,864 HELDOUT for R6; every TRAIN observation for the MVP.
- Each R6 checkpoint is evaluated on its own candidate distribution and, with `--candidate-exclusion`, on the other version's (files `<name>_exclusion_<on|off>.json`).

| Checkpoints | candidates | TRAIN top-1 | HELDOUT top-1 | HELDOUT rank mean | HELDOUT: a⁺ beats every π-noise candidate | HELDOUT swapped top-1 | HELDOUT a⁺ higher under the own text |
|---|---|---|---|---|---|---|---|
| r6, seeds 0 / 1 / 2 | with exclusion | 0.762 / 0.768 / 0.761 | 0.228 / 0.217 / 0.224 | 8.40–8.75 | 0.282–0.300 | 0.131–0.144 | 0.698–0.721 |
| r6 | without | 0.676–0.684 | 0.210–0.222 | 8.50–8.85 | 0.275–0.294 | 0.127–0.143 | 0.698–0.721 |
| r6fix, seeds 0 / 1 / 2 | without | 0.635 / 0.632 / 0.621 | 0.245 / 0.211 / 0.233 | 8.30–8.67 | 0.288–0.321 | 0.133–0.147 | 0.670–0.710 |
| r6fix | with exclusion | 0.671–0.683 | 0.215–0.249 | 8.22–8.59 | 0.292–0.326 | 0.134–0.148 | 0.670–0.710 |
| MVP, seeds 0 / 1 / 2 | with exclusion | 0.402 / 0.397 / 0.380 | 0.255 / 0.231 / 0.238 | 7.82–8.04 | 0.289–0.307 | 0.112–0.127 | 0.859–0.886 |

- **r within these candidate sets**, R6 against the MVP r: per-state effective rank 7.1–7.6 against 2.5–2.6; pooled within-state rank 79–82 against 15; within-state share 0.39–0.42 against 0.52–0.55.
- **Task encodings** (`tasks/<name>.json`): every checkpoint separates the 10 tasks. Within-task cosine distance is 0.0013–0.0015, between tasks 1.111, and nearest-variant and nearest-centroid accuracy 1.00, as in the MVP.

### Reading

- **With R6 as r, argmax over π's candidates is above the uniform choice among the same candidates:**
  - success 0.901 against 0.876, pooled McNemar 157 / 119, p = 0.026;
  - relative gain +0.028, with argmax above uniform in seeds 1 and 2 and equal in seed 0;
  - it stays below π's mean chunk (0.927; 93 / 133, p = 0.009).
- **Against the MVP**, the R6 argmax succeeds in 0.058 more episodes (186 / 100, p = 4e-7). Its choices shift from scale 0.2 (79% → 69–70%) toward scales 0.1 and 0.05.
- **Held-out inits 40–49** (100 episodes per run):
  - the R6 argmax succeeds in 0.914, against 0.870 for the uniform choice (35 / 22, p = 0.11) and 0.920 for the mean chunk (19 / 21, p = 0.88);
  - on inits 0–39 the mean chunk leads it by 0.031 (74 / 112, p = 0.0065).
- **The phase-15 version is no better than the uniform choice.** It succeeds in 0.874 (150 / 153, p = 0.91) and falls 0.027 below the MVP-recipe version (109 / 147, p = 0.021). Its q keeps π's candidates near the demonstration in training and lowers TRAIN acceptance (0.62–0.64 against 0.76). s never approaches either clamp.
- **The offline measures do not follow the closed-loop gain:**
  - On the branch test's sweep states, both R6 versions rank candidates as the MVP q does (same-scale AUC 0.511–0.523, no paired difference); on the held-out inits, the r6 within-state AUC is 0.46–0.49.
  - On TRAIN demonstration observations, acceptance nearly doubles (top-1 0.76 against 0.40); on HELDOUT observations it stays at 0.21–0.25, the MVP's level.
  - On HELDOUT observations, the own text raises a⁺ above another task's text less often than in the MVP (0.67–0.72 against 0.86–0.89).

## R6 on all data (phase 19)

This is the recipe of phase 13 applied to all twelve training stores, after the Phase 6 part-2 collection (`docs/interaction_data.md`). It covers steps 1–2 of phase 19; q is not retrained here.

Names used below:

- **R6_all**: this run.
- **R6_nine**: phase 13's R6, trained on nine stores.
- **MVP r**: the fixed stage one with `goal_dynamics_perturb`.

### PCA

`outputs/feature_pca/r6_all_stores/pca.npz`: `scripts/fit_feature_pca.py --config configs/interaction/frozen_pca_delta_all_stores.json --threads 4`, code `020325e`, node-65 CPU, 246 s, 2026-09-29 00:38–00:42 EDT (log `tmp/runlogs/s4p19/logs/fit_pca.log`).

| Store | Held-out rule | TRAIN frames | HELDOUT frames |
|---|---|---|---|
| the nine stores of phase 13 | as in phase 13 | 151,103 | as in phase 13 |
| `random_demo_branch_train` | none | 40,560 | 0 |
| `random_init_walk` | none | 4,200 | 0 |
| `pi_noise_nopsi` | starts 80–99 | 24,800 | 6,200 |
| total | | 220,663 | |

No frame is in both splits.

| Components k | 1 | 8 | 16 | 32 | 64 | 128 | **256** | 512 | 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| variance explained | 0.149 | 0.464 | 0.589 | 0.697 | 0.787 | 0.864 | **0.9270** | 0.973 | 0.997 |

- The covariance has effective rank 86.2. The 256th eigenvalue is 0.0934, 2.08e-3 of the first (44.83).
- Self-check on the fitting frames:
  - projected variance against eigenvalue, max relative error 7.2e-15;
  - largest off-diagonal projected covariance, 2.9e-16 of the top eigenvalue;
  - max |VᵀV − I| 2.0e-15;
  - eigen residual 2.0e-16 of the top eigenvalue.

### Effect head

- **Config**: `configs/interaction/frozen_pca_delta_all_stores.json`. It is `frozen_pca_delta_nine_stores.json` with:
  - the three part-2 stores appended, with the rules of the PCA table;
  - this PCA as the fixed encoder;
  - its own run dir.
  - The recipe is unchanged: L_pred per dim only, 20,000 updates, seed 0, batch 256, K0–K4.
- **Data**: TRAIN pool 203,491 transitions; the part-2 stores add 37,440 + 4,000 + 24,000. HELDOUT pool 24,517, of which 6,000 are from `pi_noise_nopsi`.
- **Monitor set**: 4,096 pooled HELDOUT transitions, 78 of them stay transitions. This set is not the one of phase 13, so the monitor R² of the two runs is not directly comparable.
- **Run**: `outputs/runs/interaction_frozen_pca_delta_all_stores_seed0`, code `020325e`, node-65 CUDA 1.
  - Launched through `scripts/cluster_run.sh --gpus 1`; `gpu_guard.sh` passed and the card was idle at launch.
  - The twelve stores took 489 s to load into memory. Training ran from about 00:51 to 01:38 EDT, 2026-09-29.
  - 2,810 s for 20,000 updates: 7.1 updates/s with the monitor, median 7.7 without it. Peak GPU memory 3.18 GiB.
  - Exit status 0. No criterion held at any checkpoint, and no `monitor_stop.md` was written.

Batch values (`metrics.jsonl`; window means of the logged lines):

| Updates | L_pred / 256 | cos(Δ̂, z) | median ‖Δ̂ − z‖/‖z‖ | median ‖Δ̂‖ |
|---|---|---|---|---|
| 0 | 0.681 | 0 | 1.000 | 0 |
| 1,500–2,000 | 0.277 | 0.705 | 0.664 | 9.60 |
| 7,500–8,000 | 0.160 | 0.817 | 0.494 | 10.60 |
| 19,500–19,999 | 0.099 | 0.875 | 0.397 | 11.14 |

Per-source L_pred / 256 over updates 19,500–19,999:

| Source | L_pred / 256 |
|---|---|
| demo_renders | 0.076 |
| data A | 0.112 |
| rollouts_r0 | 0.109 |
| resim mean | 0.088–0.091 |
| resim sweep | 0.059–0.072 |
| random_demo_branch_train | 0.116 |
| random_init_walk | 0.145 |
| pi_noise_nopsi | 0.133 |

For comparison, the phase-13 run's final values were 0.060 overall and 0.049–0.056 on the resim sweep stores.

Monitor (ψ is fixed: rank 52.1, top share 0.167, K3 0.343 against R0 0.372):

| Step | Δ̂ rank / top share | median ‖Δ̂‖/‖z‖ | R² | SIGReg(Δ̂) | cos(Δ̂, z) | K0 K1 K2 K3 |
|---|---|---|---|---|---|---|
| 2,000 (not counted) | 63.8 / 0.103 | 0.772 | 0.523 | 0.112 | 0.650 | F F F F |
| 6,000 | 74.1 / 0.102 | 0.823 | 0.565 | 0.095 | 0.684 | F F F F |
| 12,000 | 82.3 / 0.093 | 0.874 | 0.560 | 0.075 | 0.689 | F F F F |
| 20,000 | 84.0 / 0.092 | 0.892 | 0.550 | 0.069 | 0.688 | F F F F |

- **K4**: σ_ψ stays 1.0488. Mean ‖Δ̂‖ goes from 7.44 at step 500 to 10.69 at step 20,000 (ratio 1.44); neither K4a nor K4b holds.
- **Final diagnostics**:
  - Var(z)/Var(ψ) 0.577;
  - direction correlation, eigenvalue-weighted 0.719, min 0.227;
  - stay ‖Δ̂‖ quartiles 2.34 / 3.31 / 4.71, against stay ‖z‖ 2.28 / 5.22 / 7.11.

Held-out prediction per source at step 20,000, all HELDOUT transitions (`outputs/analysis/query_match/r6_all_gate/heldout.json`). Each r is measured against z in its own ψ space:

| Held-out part | n | R6_all: R² / cos mean / median rel. error | R6_nine | MVP r |
|---|---|---|---|---|
| demo_renders `heldout_*` | 4,864 | 0.677 / 0.799 / 0.561 | 0.632 / 0.775 / 0.598 | 0.854 / 0.924 / 0.244 |
| rollouts_r0 starts 80–99 | 4,470 | 0.519 / 0.624 / 0.753 | 0.450 / 0.599 / 0.798 | 0.315 / 0.484 / 0.924 |
| — rollout_stay | 533 | −0.011 / 0.253 / 1.042 | −0.181 / 0.233 / 1.122 | 0.009 / 0.086 / 1.321 |
| resim mean, inits 40–49 (three stores) | 1,173–1,236 | 0.707–0.710 / 0.795–0.801 / 0.510–0.528 | 0.671–0.684 / 0.781–0.782 / 0.534–0.543 | 0.772–0.843 / 0.848–0.880 / 0.236–0.250 |
| resim sweep fixed / flow / flow_full_sgd, inits 40–49 | 1,920 / 1,920 / 1,728 | 0.558 / 0.554 / 0.528 | 0.461 / 0.511 / 0.449 | 0.555 / 0.819 / 0.770 |
| pi_noise_nopsi starts 80–99 | 6,000 | 0.377 / 0.590 / 0.829 | 0.153 / 0.496 / 0.959 | 0.234 / 0.418 / 0.970 |

R6_all predicts every held-out part better than R6_nine.

### Distinguishability check (step 20,000)

The sets, candidates, metrics and thresholds are those of phase 13:

- π's candidates at the 1,084 sweep states: (a) the stored ψ-π candidates, which have true effects; (b) the no-ψ π seed-0 candidates on the same stream.
- The 258 HELDOUT states of `counterfactual_eval_heldout` × 32 smooth random chunks.

The script is `tmp/runlogs/s4p19/gate_r6_p19.py`, the phase-13 script with several R6 checkpoints and the verdict per subset. It ran on node-65 CUDA 1 in 103 s. Output: `outputs/analysis/query_match/r6_all_gate/{gate.json,effects.pt}`. The phase-13 checks repeat identically:

- stored candidates reproduced from their candidate 0 in 1,077 of 1,084 states;
- store chunks equal to the stored candidates;
- ψ(o) of the store frame and of the stored observation within 1.8e-4 relative.

Only the 174 sweep states of inits 40–49 and the counterfactual states are held out for all three r. The resim sweep stores of inits 0–39, which hold the stored candidates' transitions at the other 910 sweep states, are TRAIN data of both R6 runs. The MVP's effect head saw no sweep or counterfactual transition.

**Gate numbers** (share ≥ 0.05, rank mean ≥ 4, cos mean ≤ 0.99):

| States | Candidates | r | share | rank mean (median) | cos mean (median) | passes |
|---|---|---|---|---|---|---|
| sweep, all 1,084 | π (a) | R6_all | 0.0504 | 3.69 (3.62) | 0.938 (0.975) | no (rank) |
| | | R6_nine | 0.070 | 4.18 (4.14) | 0.911 (0.960) | yes |
| | | MVP r | 0.020 | 2.23 (2.10) | 0.984 (0.997) | no |
| | π (b) | R6_all | 0.044 | 3.88 (3.82) | 0.947 (0.977) | no (share, rank) |
| | | R6_nine | 0.053 | 4.51 (4.45) | 0.932 (0.967) | yes |
| | | MVP r | 0.013 | 2.25 (2.08) | 0.986 (0.998) | no |
| sweep, inits 0–39 (910, trained on) | π (a) | R6_all | 0.052 | 3.63 (3.53) | 0.935 (0.975) | no (rank) |
| | | R6_nine | 0.073 | 4.06 (3.95) | 0.905 (0.958) | yes |
| | π (b) | R6_all | 0.045 | 3.85 (3.77) | 0.945 (0.976) | no |
| | | R6_nine | 0.055 | 4.45 (4.39) | 0.930 (0.967) | yes |
| **sweep, inits 40–49 (174, held out)** | π (a) | R6_all | 0.042 | 3.99 (3.99) | 0.951 (0.976) | no (share, rank) |
| | | R6_nine | 0.050 | 4.83 (4.71) | 0.939 (0.968) | yes |
| | | MVP r | 0.018 | 2.27 (2.01) | 0.985 (0.997) | no |
| | π (b) | R6_all | 0.039 | 4.08 (4.01) | 0.954 (0.978) | no (share) |
| | | R6_nine | 0.046 | 4.85 (4.79) | 0.943 (0.970) | no (share) |
| | | MVP r | 0.015 | 2.26 (2.00) | 0.986 (0.998) | no |
| **counterfactual (258, held out)** | smooth random | R6_all | 0.553 | 9.14 (9.17) | 0.485 (0.496) | **yes** |
| | | R6_nine | 0.512 | 11.00 (11.06) | 0.520 (0.533) | yes |
| | | MVP r | 0.679 | 3.41 (3.02) | 0.400 (0.493) | no (rank) |

**R6_all passes on the smooth random candidates and fails on π's candidates.** On all 1,084 states, set (a) misses the rank threshold (3.69) and set (b) the share and rank thresholds (0.044, 3.88). Per name, set (a) has share 0.044–0.056, rank 3.64–3.71 and cos mean 0.930–0.946.

**True effects in each space** (identical sets of o and o′; the two PCAs give nearly the same values):

| States | Candidates | z in | share | rank mean | cos mean (median) |
|---|---|---|---|---|---|
| sweep, all | π (a) | R6_all space | 0.176 | 11.78 | 0.768 (0.827) |
| sweep, inits 40–49 | π (a) | R6_all space | 0.162 | 12.13 | 0.781 (0.841) |
| sweep, inits 40–49 | π (a) | MVP ψ space | 0.066 | 2.75 | 0.890 (0.990) |
| counterfactual | smooth random | R6_all space | 0.644 | 13.76 | 0.358 (0.359) |
| counterfactual | smooth random | MVP ψ space | 0.724 | 3.13 | 0.282 (0.345) |

**r against the true effects** (same space; within-state R² = 1 − Σ‖(r − r̄_s) − (z − z̄_s)‖² / Σ‖z − z̄_s‖²):

| States | r | within-state R² | centered cos | Spearman of pairwise distances | spread ratio | set R² |
|---|---|---|---|---|---|---|
| sweep, inits 0–39, π (a) | R6_all | 0.291 | 0.421 | 0.809 | 0.402 | 0.866 |
| | R6_nine | 0.399 | 0.517 | 0.837 | 0.516 | 0.890 |
| | MVP r | 0.197 | 0.334 | 0.638 | 0.694 | 0.681 |
| **sweep, inits 40–49, π (a)** | R6_all | **0.069** | 0.215 | 0.751 | 0.377 | 0.553 |
| | R6_nine | 0.006 | 0.180 | 0.751 | 0.444 | 0.481 |
| | MVP r | 0.189 | 0.361 | 0.641 | 0.657 | 0.698 |
| **counterfactual, smooth random** | R6_all | **0.425** | 0.628 | 0.820 | 0.811 | 0.564 |
| | R6_nine | 0.182 | 0.467 | 0.724 | 0.802 | 0.367 |
| | MVP r | 0.349 | 0.631 | 0.609 | 0.678 | 0.428 |

### Reading

- **On the held-out smooth random candidates, R6_all's within-state variation follows the true effects much better than R6_nine's**: within-state R² 0.43 against 0.18, and Spearman 0.82 against 0.72. It stays well separated (share 0.55, rank 9.1, cos 0.48). The part-2 stores are TRAIN demonstration-state branches with the same smooth sampler, so the counterfactual set's HELDOUT states are the closest held-out match to them.
- **On π's candidates at held-out sweep states**, R6_all's within-state R² is 0.069, against 0.006 for R6_nine. Both are below the MVP r's 0.189 in its own space. In that space the true spread among π's candidates is small (share 0.066, median cos 0.990).
- **R6_all's spread among π's candidates is smaller than R6_nine's**: share 0.050 against 0.070 and rank 3.7 against 4.2 over all 1,084 states. Most of the difference is on the 910 states whose candidates trained both heads, where R6_nine fits them more closely (within-state R² 0.40 against 0.29; final L_pred / 256 on the sweep stores 0.049–0.056 against 0.059–0.072).
- **Gate verdict under the phase-13 rule**: R6_all passes on the smooth random candidates and fails on π's candidates (both versions, all states and held-out states alike), so by that rule it continues.
- **Share of π's candidates' true spread that r carries**: spread ratio 0.38–0.40 for R6_all, against a true share of 0.16–0.18 in R6 space.

## Paths

| What | Where |
|---|---|
| tables | `outputs/query_match/tables_mvp/` |
| runs | `outputs/runs/query_match_mvp_seed{0,1,2}/` |
| offline evaluation | `outputs/analysis/query_match/{acceptance,tasks,sweep}/`; sweep scores `outputs/analysis/branch_rescore/{fixed,flow,flow_full_sgd}/scores/qm_mvp_seed{0,1,2}.{npz,json}` |
| closed-loop checks and smoke | `outputs/closed_loop/mvp_checks/`, `outputs/closed_loop/mvp_smoke/` |
| closed loops (phase 12) | `outputs/closed_loop/v11_qm_argmax_s{0,1,2}/task<t>_argmax/`, `outputs/closed_loop/v11_qm_random_s{0,1,2}/task<t>_uniform/` (`results.jsonl`, `trace.jsonl`) |
| phase-14 code: checks, smokes, pilot | `outputs/closed_loop/p14_checks/`, `outputs/closed_loop/p14_smoke/`, `outputs/closed_loop/p14_pilot_smooth_uniform_s0/`, `outputs/query_match/p14_smoke_tables_r6/`, `outputs/runs/p14_bitid_qm_{new,base}/`, `outputs/runs/p14_smoke_qm_{r6,phase15_options}/`; scripts and outputs `tmp/runlogs/s4p14/`, phase-14 launchers `tmp/runlogs/s4p14/phase14/` |
| R6 (phase 13) | config `configs/interaction/frozen_pca_delta_nine_stores.json`; PCA `outputs/feature_pca/r6_nine_stores/pca.npz`; run `outputs/runs/interaction_frozen_pca_delta_nine_stores_seed0/` (r = `checkpoints/step_020000.pt`); gate and held-out prediction `outputs/analysis/query_match/r6_gate/{gate.json,effects.pt,heldout.json}`; launchers, scripts, logs and the training summary `tmp/runlogs/s4p13/` (`env.sh`, `fit_pca.sh`, `train.sh`, `run_gate.sh`, `run_heldout.sh`, `final_eval.sh`, `gate_r6.py`, `heldout_r6.py`, `summarize_training.py`, `training_summary.txt`, `gate_step{002000,010000}/`, `logs/`) |
| demo starts (phase 18) | runs `outputs/closed_loop/p18_demo_uniform_s{0,1,2}/task<t>_uniform/`, `outputs/closed_loop/p18_demo_argmax_{r6,r6fix}_s{0,1,2}/task<t>_argmax/` (`results.jsonl`, `trace.jsonl`, `demo_starts.jsonl`, `run_config.json`); checks and smoke `outputs/closed_loop/p18_checks/`, `outputs/closed_loop/p18_smoke/`; launchers, queues, worker and job logs, check and analysis scripts and their outputs `tmp/runlogs/s4p18/` (`env.sh`, `run_one.sh`, `run_one_base.sh`, `make_jobs.sh`, `worker.sh`, `start_workers.sh`, `launch_queue.sh`, `launch_checks*.sh`, `launch_smoke.sh`, `check_units.py`, `check_refusals.sh`, `analyze_demo.py`, `analysis_entry.sh`, `check_units.txt`, `check_refusals.txt`, `compare_checks.txt`, `compare_mean_repeat.txt`, `smoke_summary.txt`, `analysis_runs_v2.txt`, `starts.tsv`, `heldout_episodes.txt`, `queues/`, `claims/`, `logs/`) |
| phases 14–15 on R6 | tables `outputs/query_match/tables_r6/`; runs `outputs/runs/query_match_r6_seed{0,1,2}/`, `outputs/runs/query_match_r6_fix_seed{0,1,2}/`; closed loops `outputs/closed_loop/v12_r6_argmax_s{0,1,2}/task<t>_argmax/`, `outputs/closed_loop/v12_r6fix_argmax_s{0,1,2}/task<t>_argmax/` (seed 0 also `task7_argmax_rest/`); offline `outputs/analysis/query_match/{acceptance,tasks}/r6_*`, `sweep/r6_pooled.*`, sweep scores `outputs/analysis/branch_rescore/<name>/scores/qm_{r6,r6fix}_seed<k>.*`; launchers, logs and analysis `tmp/runlogs/s4p14/phase14/` (`closed_loop_analysis.txt`, `closed_loop_analysis_with_rerun.txt`, `pairing.txt`, `sweep_extra.txt`, `offline_summary.txt`, `failures_v12_*.txt`, `logs/`) |
| R6 on all data (phase 19) | config `configs/interaction/frozen_pca_delta_all_stores.json`; PCA `outputs/feature_pca/r6_all_stores/pca.npz`; run `outputs/runs/interaction_frozen_pca_delta_all_stores_seed0/` (r = `checkpoints/step_020000.pt`); check and held-out prediction `outputs/analysis/query_match/r6_all_gate/{gate.json,effects.pt,heldout.json}`; entry scripts (through `scripts/cluster_run.sh`), scripts, logs and the training summary `tmp/runlogs/s4p19/` (`start.sh`, `pipeline.sh`, `status.sh`, `summarize.sh`, `test_scripts.sh`, `env.sh`, `gate_r6_p19.py`, `heldout_r6_p19.py`, `summarize_training.py`, `tables.py`, `training_summary_fixed.txt`, `logs/`) |
| launchers, logs, check scripts | `tmp/runlogs/s4mvp/` (`env.sh`, `launch.sh`, `build_tables.sh`, `train_one.sh`, `run_one.sh`, `run_one_base.sh`, `launch_checks*.sh`, `launch_smoke.sh`, `launch_closed_loop.sh`, `evaluate_all.sh`, `run_diagnostics.sh`, `check_*.py`, `diagnose_r.py`, `diagnose_sweep_r.py`, `oracle_readout.py`, `summarize_*.py`, `analyze_phase12.py`, their `.txt` outputs, `failures_v11_qm_*.txt`, `logs/`) |

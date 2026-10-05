# Velocity-field and selection consistency of stage-one checkpoints (phase 20)

Two offline acceptance metrics for a stage-one checkpoint, both in the checkpoint's own ψ:

- **Velocity-field consistency**: on demonstration frames, do the match query directions point where stage one's per-frame target says the state moves?
- **Selection consistency**: at a demonstration state, does the candidate the match selects move ψ along that state's expected velocity more than the candidates it does not select?

Scene-based progress labels (`docs/rollout_progress.md`) play no part here.

## Checkpoints

All at step 5000, with the same feature cache, split and instructions:

| Name | Run (`outputs/runs/`) | Path target | Query conditioning |
|---|---|---|---|
| fixed | `goal_instruction_contrast` | perpendicular in U coordinates | text only |
| flow | `goal_instruction_contrast_flow` | interpolation in U coordinates | text only |
| flow_full | `goal_instruction_contrast_flow_full` | interpolation in the full ψ space (ψ effective rank 1.1, ISSUE-016) | text only |
| obs (reference) | `goal_instruction_contrast_obs` | perpendicular in U coordinates | text and observation |

## Definitions

**Notation.**

- An episode has frames 0 … T, T = length − 1. ψ_t is the checkpoint's trunk on the feature-cache frame t.
- The horizon is k = 10, the chunk length.
- Every direction uses the task's original instruction l (`InstructionTable` variants[0]).
- A text-only checkpoint has one q_l, ĝ_l and U_l (256 × 8, orthonormal columns) per task.
- obs is queried with the references its training used: q(o, l) with the current frame o, and ĝ(o_0, l), U(o_0, l) with the first dataset frame o_0 of the frame's episode.

**Velocities.**

- Displacement: v_k = ψ_{t+k} − ψ_t.
- Episode velocity: v_ep = ψ_T − ψ_0.
- Both come from the feature cache.
- Stage one's path term (`docs/stage_one_flow.md`) ties these velocities to the query:
  - fixed and obs keep Uᵀψ_t on the line through Uᵀψ_0 and Uᵀψ_T, at any position;
  - flow places Uᵀψ_t at position t/T of that segment;
  - flow_full places ψ_t at position t/T of the full-space segment from ψ_0 to ψ_T.
  - The goal loss puts Uᵀĝ at Uᵀψ_T.
  - So in U coordinates, ĝ − ψ_t, ψ_T − ψ_0 and the displacements of fixed, flow and obs are aligned by the training objective itself.

### Velocity-field consistency

- **Frames**: every frame t with t + k ≤ T.
  - HELDOUT: 43 episodes, 4,864 frames.
  - TRAIN: 390 episodes, 43,701 frames.
- **Cosines per frame**, with the frame's own task:

| Name | Cosine |
|---|---|
| cos(q, v) | cos(q_l, v) |
| cos(g, v) | cos(ĝ_l − ψ_t, v), full 256-d space |
| cos(gU, v) | cos(U_lᵀ(ĝ_l − ψ_t), U_lᵀv), query coordinates |

each for v = v_k and v = v_ep.
- **Summaries** over the frames of a split:
  - mean, median, p10, p90 and the share > 0;
  - the same by t/T quarter.
- **Other-task control**: the same cosine with each of the other nine tasks' q, ĝ and U, averaged over those tasks, and own − other. A direction that only follows a velocity shared by all tasks scores as well with another task's instruction.
- **Degeneracy control**: the mean cosine and the root-mean-square cosine between the unit vectors of different states.
  - Pairs are taken from different tasks, or from the same task and different episodes.
  - Vectors:
    - v_k, one per frame;
    - v_ep, one per episode;
    - the own-task q and ĝ − ψ_t, one per frame. For a text-only q only pairs of different tasks count, since one task has one q.
    - U_lᵀv_k, same task only, text-only checkpoints only, where the episodes of a task share U_l.
  - The averages are exact, from group sums: Σ_{i,j} cos_ij = ‖Σ u_i‖² and Σ_{i,j} cos²_ij = ‖Σ u_i u_iᵀ‖_F².
  - For random directions in 256 dimensions the RMS cosine is 1/√256 = 0.06.
  - If the velocities of all states point the same way, any direction aligned with that way scores a high velocity consistency without carrying information about the task or the state.

### Selection consistency

- **States**: the 129 restored HELDOUT states of the phase-12 protocol, 43 episodes × t = round(f · (T − k)), f ∈ {0.1, 0.4, 0.7}.
- **Candidate sets**, executed in the simulator from the restored state:
  - counterfactual: demo, stay and 12 perturbed chunks (σ ∈ {0.5, 1, 2} × 4 draws), as in `docs/dynamics_counterfactuals.md`;
  - π: 32 candidates of π (`goal_proposal` step 3000): its mean chunk and 31 constant offsets (J = 1) with scales cycling 0.05, 0.1, 0.2; gripper flip off.
- **Effects**: z_c = ψ(o′_c) − ψ(o) under each checkpoint, from the stored simulator frames o and o′_c. The new checkpoints have no Δ, so only true effects are used.
- **Scores**:
  - m_q = cos(z, q);
  - m_g = cos(z, ĝ − ψ(o));
  - m_gU = cos(Uᵀz, Uᵀ(ĝ − ψ(o))).
- **Pool and choice**:
  - The pool is the non-stay candidates: 13 in the counterfactual set, 32 in the π set.
  - The chosen candidate is argmax_pool m.
  - A state whose pool maximum is ≤ 0 is one where MATCH, which scores stay m = 0, would stay. Such states are counted as stay choices, and the argmax is still evaluated there.
- **Expected velocity v** at the state:
  - v_k = ψ_{t+k} − ψ_t of the same (episode, t);
  - v_ep = ψ_T − ψ_0 of the episode.
  - Both come from the feature cache. The evaluation cosine is cos(z_c, v) in the full space, for every score.
- **Per state**:
  - the chosen candidate's cosine;
  - the mean and median cosine of the unchosen pool;
  - chosen − unchosen mean;
  - whether the chosen cosine exceeds the unchosen median;
  - the Spearman correlation over the pool between m(z_c) and cos(z_c, v), with average ranks;
  - the best cosine in the pool and the within-state standard deviation of the cosine;
  - the cosine of the reference candidate: demo in the counterfactual set, π's mean chunk in the π set.
- **Random-choice baseline**: the exact expectation over a uniform choice from the pool.
  - chosen − unchosen is 0;
  - Spearman is 0;
  - the share above the unchosen median is computed per state.
- **Intervals**: 95% percentile intervals from 2,000 bootstrap resamples of episodes. The resamples are drawn once per analysis, and every checkpoint uses the same ones, so differences against fixed are paired.

## Candidate frames

The frames of o and every o′ are stored, so any stage-one checkpoint can be scored without the simulator.

**Code**: `scripts/check_dynamics_counterfactuals.py`, commit `e5b8d21`. Frame recording and the π set are options of the phase-12 script, so the counterfactual set, its restoration and its records use the phase-12 code path, on which the reproduction check below relies.

- `--transitions-out <store> --transitions-shard <name>` writes, per sample, o once and then, per executed candidate in execution order, o′ and one transition (`TransitionStoreWriter`).
  - Execution order: the 14 counterfactual candidates in `candidate_index` order, then the 32 π candidates in `candidate_index` order.
  - Transition kind: demo, stay, perturbed or proposal.
  - sigma: σ of a perturbed chunk or the π candidate's scale; 0 for demo, stay and π's mean chunk.
  - episode: the dataset episode id; t: the frame index.
  - `candidates.jsonl` and `proposals.jsonl` hold each candidate's shard-local `transition` index, next to its candidate index, σ or scale and its true match under the fixed stage one.
- `--proposal-checkpoint` (with `--proposal-setting`, default `32:0.05,0.1,0.2/1`) executes π's candidates from the same restored state, each after restoring the state again.
  - π must share Δ's stage one and action cache (checked); π runs on the restored frame with ψ(o) and q_l of that stage one.
  - Offsets come from `NoiseStream.position(n)`. n is the sample's index in the full protocol order: every HELDOUT episode by task id then episode id, × f in order. It does not depend on `--episodes`, so a sample's candidates do not depend on how episodes are split over processes.
  - One `ProposalRecord` per π candidate: episode, t, f, n, candidate index, scale, ‖z‖, true match, success, transition.

**Runs**: 8 shards `s0`–`s7`, node-61 CUDA 0–7, one process each, 175–215 s.

- Each shard covers a contiguous block of 5–6 episodes in protocol order.
- Launchers and logs: `outputs/analysis/consistency_candidates/logs/`.
- Run directories: `outputs/analysis/consistency_candidates/s{0..7}/`, holding `config.json`, `candidates.jsonl`, `proposals.jsonl`, `samples.jsonl` and `summary.txt`.
- Store: `outputs/transitions/consistency_candidates/`, 6,063 frames and 5,934 transitions (1.2 GB). Per state: 1 o frame, 14 + 32 o′ frames and 46 transitions.
- `/tmp/robosuite.log` on node-61 was 24,231 bytes (Sep 25 01:32) before and after the runs.

## Scorer

`scripts/score_consistency.py`:

```bash
source scripts/ceam_env.sh
$CEAM_PYTHON scripts/score_consistency.py --checkpoints <fixed> <flow> <flow_full> <obs> --names fixed flow flow_full obs \
  --runs outputs/analysis/consistency_candidates/s{0..7} --out-dir outputs/analysis/consistency/<name> \
  --device cuda:<idle GPU> --code-commit <sha>
```

- **Inputs**:
  - the feature cache of the checkpoints, which must share it, together with the dataset, split and instructions (checked);
  - the transition store `--transitions`, default `outputs/transitions/consistency_candidates`;
  - the run directories `--runs`;
  - the phase-12 run `--reference-candidates` for the reproduction check.
  - The first checkpoint is the reference of the paired differences.
- **Numerics**:
  - The trunks and query modules run in float32 without autocast, TF32 off, 256 frames per call, deterministic algorithms, cuBLAS workspace `:4096:8`.
  - Every difference, projection and cosine of ψ, q, ĝ and U is computed in float64.
  - The π candidates' scores and velocity cosines nearly tie within a state, so their ranks depend on rounding. With the cosines in float32 (`outputs/analysis/consistency/stage_one_step5000`, commit `c3da30e`):
    - no choice, stay decision or cosine moves by more than 1e-6;
    - per-state Spearman correlations move by up to 0.044 and the random-choice shares by up to 0.031;
    - every aggregate estimate moves by at most 0.003 (`logs/compare_f32_f64.txt`).
- **Pass**: one pass over the 52,895 feature-cache frames and the 6,063 store frames runs every checkpoint on each batch. The obs query runs for every (frame, task) pair.
- **Cost**: 93 s on an L40S (node-61) for the four checkpoints, 85 s of it the pass.
- **Outputs**: an existing directory is refused.
  - `config.json`: arguments, provenance, frame counts and the checks;
  - `states.csv`: one row per checkpoint, set, score, velocity and state;
  - `tables.csv`: every estimate with its interval;
  - `comparisons.csv`: every later checkpoint minus the first, paired;
  - `report.md`: all tables.

## Checks

Run `outputs/analysis/consistency/stage_one_step5000_f64` (commit `54073ce`, node-61 CUDA 3). Check scripts and outputs are in `outputs/analysis/consistency/logs/`.

- **Reproduction of the phase-12 counterfactual set** (`outputs/dynamics_counterfactuals/goal_dynamics_short_step3000_heldout_seed0/candidates.jsonl`):
  - all 1,806 records are matched on (episode, t, candidate index);
  - the chunks are the same in all 1,806: kind, σ, draw, offset and clipped fraction are equal;
  - the true match under the fixed stage one differs by at most 5.8e-4, median 0; the Δ̂ match by at most 2.1e-6; ‖z‖ by at most 0.14% relative;
  - the MATCH choice under the true match (stay scored 0) is the same in 129 of 129 states, and the sign of the true match in 1,806 of 1,806.
  - The original run used node-65 CUDA 5, these runs node-61. Repeated executions of the same chunks differed by up to 5.1e-4 in phase 18 (`docs/rollout_progress.md`).
- **Store join**: all 5,934 transitions are joined to their records.
  - current frame < future frame in all 5,934;
  - episode, t, kind and σ or scale agree with the record in all 5,934;
  - all 46 candidates of a state start from one stored o in 129 of 129 states.
- **Store consistency**: m(z) = cos(z, q_l) under the fixed stage one, recomputed from the stored frames, equals the value the simulator run recorded within 9.4e-6 (counterfactual) and 4.6e-6 (π).
- **Determinism**: the same command on CUDA 4 writes byte-identical `states.csv`, `tables.csv` and `comparisons.csv`.
- **Pairwise averages**: the group-sum formulas equal brute-force averages over all pairs to 2e-16 on 700 synthetic vectors in 5 groups and 35 subgroups (`check_pairwise.{py,txt}`).
- **Vectorized selection**: a per-state loop with `scipy.stats.spearmanr` gives the same choices, stay decisions and cosines in all 6,192 state rows (`compare_loop_vectorized_f32.txt`).

## Results

### Velocity-field consistency

HELDOUT, 4,864 frames. Each cell is the mean cosine [95% interval], with own − other in parentheses: the own-task cosine minus the mean over the other nine tasks.

| direction | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.40 [+0.38, +0.42] (+0.44) | +0.34 [+0.33, +0.36] (+0.38) | +0.37 [+0.35, +0.38] (+0.41) | +0.42 [+0.40, +0.45] (+0.62) |
| q | ψ_T − ψ_0 | +0.72 [+0.70, +0.73] (+0.80) | +0.64 [+0.60, +0.67] (+0.70) | +0.10 [+0.08, +0.12] (+0.11) | +0.31 [+0.27, +0.34] (+0.73) |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.34 [+0.31, +0.37] (+0.44) | +0.03 [+0.01, +0.04] (+0.06) | +0.24 [+0.19, +0.31] (+0.02) | +0.27 [+0.25, +0.30] (+0.07) |
| ĝ − ψ | ψ_T − ψ_0 | +0.48 [+0.44, +0.52] (+0.70) | -0.15 [-0.18, -0.11] (+0.07) | +0.74 [+0.66, +0.79] (+0.05) | +0.17 [+0.13, +0.21] (+0.20) |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +0.90 [+0.86, +0.93] (+1.13) | +0.91 [+0.87, +0.94] (+1.00) | +0.18 [+0.07, +0.29] (-0.13) | +0.57 [+0.48, +0.66] (+0.16) |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +0.98 [+0.95, +0.99] (+1.18) | +1.00 [+1.00, +1.00] (+1.04) | +0.93 [+0.89, +0.96] (+0.18) | +0.91 [+0.87, +0.94] (-0.06) |

Distribution over the frames, as median / p10 / p90 / share > 0:

| direction | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.44 / +0.20 / +0.56 / 0.95 | +0.38 / +0.17 / +0.48 / 0.95 | +0.37 / +0.14 / +0.62 / 0.96 | +0.46 / +0.14 / +0.63 / 0.94 |
| q | ψ_T − ψ_0 | +0.72 / +0.67 / +0.76 / 1.00 | +0.66 / +0.47 / +0.73 / 1.00 | +0.08 / +0.06 / +0.22 / 1.00 | +0.32 / +0.04 / +0.56 / 0.92 |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.45 / -0.32 / +0.72 / 0.80 | +0.09 / -0.44 / +0.39 / 0.60 | +0.32 / -0.42 / +0.81 / 0.69 | +0.31 / -0.10 / +0.56 / 0.83 |
| ĝ − ψ | ψ_T − ψ_0 | +0.65 / -0.28 / +0.83 / 0.84 | -0.11 / -0.67 / +0.34 / 0.42 | +0.79 / +0.73 / +0.85 / 0.97 | +0.25 / -0.42 / +0.58 / 0.76 |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +1.00 / +0.82 / +1.00 / 0.96 | +1.00 / +0.97 / +1.00 / 0.95 | +0.39 / -0.86 / +1.00 / 0.60 | +0.99 / -0.96 / +1.00 / 0.79 |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +1.00 / +0.98 / +1.00 / 0.99 | +1.00 / +1.00 / +1.00 / 1.00 | +0.99 / +0.85 / +1.00 / 0.99 | +1.00 / +0.90 / +1.00 / 0.96 |

Paired differences against fixed:

| direction | velocity | statistic | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | mean | -0.06 [-0.07, -0.05] | -0.03 [-0.05, -0.02] | +0.02 [+0.00, +0.04] |
| q | ψ_{t+10} − ψ_t | own − other | -0.06 [-0.07, -0.05] | -0.03 [-0.05, -0.02] | +0.18 [+0.16, +0.20] |
| q | ψ_T − ψ_0 | mean | -0.08 [-0.12, -0.05] | -0.62 [-0.64, -0.59] | -0.41 [-0.46, -0.37] |
| q | ψ_T − ψ_0 | own − other | -0.09 [-0.13, -0.07] | -0.68 [-0.71, -0.65] | -0.07 [-0.13, -0.00] |
| ĝ − ψ | ψ_{t+10} − ψ_t | mean | -0.31 [-0.34, -0.29] | -0.10 [-0.15, -0.04] | -0.07 [-0.09, -0.04] |
| ĝ − ψ | ψ_{t+10} − ψ_t | own − other | -0.38 [-0.41, -0.34] | -0.41 [-0.44, -0.39] | -0.37 [-0.39, -0.34] |
| ĝ − ψ | ψ_T − ψ_0 | mean | -0.63 [-0.67, -0.59] | +0.26 [+0.18, +0.32] | -0.31 [-0.34, -0.28] |
| ĝ − ψ | ψ_T − ψ_0 | own − other | -0.62 [-0.67, -0.57] | -0.65 [-0.67, -0.62] | -0.50 [-0.53, -0.47] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | mean | +0.01 [-0.02, +0.03] | -0.72 [-0.82, -0.61] | -0.33 [-0.41, -0.25] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | own − other | -0.14 [-0.19, -0.09] | -1.26 [-1.33, -1.18] | -0.97 [-1.06, -0.88] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | mean | +0.02 [+0.01, +0.05] | -0.05 [-0.09, -0.02] | -0.07 [-0.10, -0.04] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | own − other | -0.14 [-0.25, -0.04] | -1.00 [-1.08, -0.90] | -1.24 [-1.29, -1.17] |

Mean by t/T quarter; each cell lists fixed / flow / flow_full / obs:

| direction | velocity | t/T 0–0.25 | t/T 0.25–0.5 | t/T 0.5–0.75 | t/T 0.75–1 |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.32 / +0.28 / +0.30 / +0.29 | +0.39 / +0.33 / +0.33 / +0.42 | +0.43 / +0.37 / +0.46 / +0.48 | +0.49 / +0.42 / +0.38 / +0.53 |
| q | ψ_T − ψ_0 | +0.72 / +0.64 / +0.10 / +0.32 | +0.72 / +0.64 / +0.10 / +0.28 | +0.72 / +0.64 / +0.10 / +0.35 | +0.72 / +0.65 / +0.10 / +0.27 |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.39 / +0.30 / +0.15 / +0.17 | +0.52 / +0.17 / +0.17 / +0.35 | +0.46 / -0.10 / +0.28 / +0.39 | -0.18 / -0.40 / +0.42 / +0.14 |
| ĝ − ψ | ψ_T − ψ_0 | +0.79 / +0.27 / +0.79 / +0.45 | +0.71 / -0.00 / +0.77 / +0.28 | +0.41 / -0.38 / +0.73 / +0.15 | -0.22 / -0.64 / +0.63 / -0.36 |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +0.92 / +0.88 / +0.17 / +0.73 | +0.84 / +0.91 / +0.19 / +0.24 | +0.93 / +0.93 / +0.07 / +0.72 | +0.91 / +0.92 / +0.35 / +0.59 |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +1.00 / +1.00 / +0.92 / +1.00 | +1.00 / +1.00 / +0.94 / +0.99 | +0.99 / +1.00 / +0.93 / +0.95 | +0.90 / +1.00 / +0.92 / +0.60 |

TRAIN, 43,701 frames: mean (own − other). The intervals and the other statistics are in `report.md`.

| direction | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.45 (+0.50) | +0.39 (+0.43) | +0.39 (+0.44) | +0.48 (+0.70) |
| q | ψ_T − ψ_0 | +0.73 (+0.81) | +0.68 (+0.75) | +0.10 (+0.11) | +0.33 (+0.76) |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.37 (+0.49) | +0.01 (+0.07) | +0.26 (+0.03) | +0.27 (+0.07) |
| ĝ − ψ | ψ_T − ψ_0 | +0.48 (+0.71) | -0.15 (+0.10) | +0.76 (+0.05) | +0.19 (+0.21) |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +0.97 (+1.23) | +0.98 (+1.10) | +0.15 (-0.15) | +0.60 (+0.20) |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +0.98 (+1.21) | +0.99 (+1.07) | +0.94 (+0.18) | +0.95 (-0.02) |

### Degeneracy control

Each cell gives the mean cosine between different states for pairs from different tasks / from the same task (different episodes), then the root-mean-square cosine in the same order. "–" marks a pair type that is not defined.

HELDOUT:

| vector | fixed | flow | flow_full | obs |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | -0.01 / +0.39; +0.20 / +0.60 | -0.01 / +0.34; +0.22 / +0.57 | +0.08 / +0.36; +0.43 / +0.61 | +0.01 / +0.07; +0.24 / +0.43 |
| ψ_T − ψ_0 | +0.04 / +0.89; +0.13 / +0.90 | +0.12 / +0.82; +0.20 / +0.83 | +0.89 / +0.90; +0.98 / +0.99 | +0.24 / +0.89; +0.29 / +0.89 |
| q | -0.10 / –; +0.17 / – | -0.09 / –; +0.14 / – | -0.11 / –; +0.12 / – | -0.01 / +0.25; +0.23 / +0.48 |
| ĝ − ψ_t | -0.00 / +0.68; +0.16 / +0.76 | -0.05 / +0.71; +0.15 / +0.75 | +0.90 / +0.94; +0.94 / +0.98 | +0.06 / +0.39; +0.17 / +0.50 |
| Uᵀ(ψ_{t+10} − ψ_t) | – / +0.82; – / +0.92 | – / +0.83; – / +0.97 | – / +0.29; – / +0.72 | – / –; – / – |

TRAIN:

| vector | fixed | flow | flow_full | obs |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | -0.02 / +0.46; +0.20 / +0.62 | -0.01 / +0.41; +0.21 / +0.59 | +0.09 / +0.39; +0.45 / +0.62 | +0.01 / +0.07; +0.24 / +0.44 |
| ψ_T − ψ_0 | +0.02 / +0.93; +0.12 / +0.93 | +0.07 / +0.88; +0.17 / +0.89 | +0.94 / +0.96; +0.98 / +1.00 | +0.24 / +0.92; +0.29 / +0.92 |
| q | -0.10 / –; +0.17 / – | -0.10 / –; +0.14 / – | -0.11 / –; +0.12 / – | +0.00 / +0.24; +0.23 / +0.48 |
| ĝ − ψ_t | -0.00 / +0.67; +0.16 / +0.76 | -0.05 / +0.71; +0.14 / +0.76 | +0.90 / +0.94; +0.94 / +0.98 | +0.06 / +0.40; +0.18 / +0.52 |
| Uᵀ(ψ_{t+10} − ψ_t) | – / +0.94; – / +0.96 | – / +0.97; – / +0.99 | – / +0.28; – / +0.71 | – / –; – / – |

### Selection consistency, counterfactual set

129 states; pool of 13 (demo and 12 perturbed). Mean over states of cos(z, v) for the chosen candidate / the unchosen pool, then chosen − unchosen [95% interval]:

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.87 / +0.71; +0.15 [+0.11, +0.20] | +0.86 / +0.72; +0.14 [+0.10, +0.19] | +0.82 / +0.72; +0.11 [+0.06, +0.15] | +0.78 / +0.69; +0.09 [+0.04, +0.14] |
| q | ψ_T − ψ_0 | +0.61 / +0.40; +0.21 [+0.17, +0.24] | +0.57 / +0.39; +0.18 [+0.16, +0.21] | +0.15 / +0.05; +0.10 [+0.04, +0.17] | +0.21 / +0.16; +0.05 [+0.03, +0.07] |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.81 / +0.72; +0.09 [+0.03, +0.14] | +0.42 / +0.75; -0.33 [-0.43, -0.23] | +0.58 / +0.74; -0.15 [-0.23, -0.08] | +0.50 / +0.71; -0.21 [-0.26, -0.17] |
| ĝ − ψ | ψ_T − ψ_0 | +0.62 / +0.40; +0.22 [+0.19, +0.25] | +0.24 / +0.42; -0.18 [-0.25, -0.10] | +0.49 / +0.02; +0.47 [+0.39, +0.54] | +0.24 / +0.16; +0.08 [+0.05, +0.12] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +0.89 / +0.71; +0.18 [+0.14, +0.22] | +0.90 / +0.71; +0.19 [+0.15, +0.23] | +0.53 / +0.74; -0.21 [-0.30, -0.13] | +0.74 / +0.69; +0.05 [+0.00, +0.10] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +0.59 / +0.40; +0.19 [+0.17, +0.21] | +0.55 / +0.39; +0.16 [+0.14, +0.18] | +0.29 / +0.04; +0.25 [+0.18, +0.32] | +0.25 / +0.15; +0.10 [+0.07, +0.12] |

Share of states whose chosen candidate exceeds the unchosen median [95% interval]; mean within-state Spearman(m, cos(z, v)) [95% interval]; states where MATCH would stay:

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | 0.44 [0.34, 0.54]; +0.47 [+0.40, +0.53]; 3 | 0.43 [0.35, 0.50]; +0.48 [+0.42, +0.54]; 2 | 0.45 [0.37, 0.53]; +0.43 [+0.35, +0.50]; 0 | 0.55 [0.47, 0.63]; +0.46 [+0.37, +0.53]; 0 |
| q | ψ_T − ψ_0 | 0.88 [0.82, 0.93]; +0.71 [+0.65, +0.76]; 3 | 0.83 [0.75, 0.90]; +0.68 [+0.61, +0.74]; 2 | 0.57 [0.48, 0.67]; +0.08 [-0.00, +0.17]; 0 | 0.60 [0.53, 0.67]; +0.23 [+0.16, +0.29]; 0 |
| ĝ − ψ | ψ_{t+10} − ψ_t | 0.40 [0.31, 0.48]; +0.40 [+0.33, +0.46]; 4 | 0.22 [0.15, 0.31]; -0.03 [-0.13, +0.07]; 12 | 0.31 [0.24, 0.39]; +0.18 [+0.10, +0.25]; 17 | 0.17 [0.11, 0.24]; -0.14 [-0.22, -0.07]; 1 |
| ĝ − ψ | ψ_T − ψ_0 | 0.83 [0.78, 0.88]; +0.68 [+0.63, +0.73]; 4 | 0.43 [0.36, 0.51]; +0.08 [+0.00, +0.15]; 12 | 0.91 [0.82, 0.98]; +0.85 [+0.74, +0.93]; 17 | 0.71 [0.63, 0.78]; +0.31 [+0.23, +0.39]; 1 |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | 0.59 [0.49, 0.69]; +0.54 [+0.48, +0.61]; 0 | 0.71 [0.64, 0.78]; +0.57 [+0.51, +0.61]; 0 | 0.29 [0.22, 0.36]; +0.11 [+0.03, +0.19]; 12 | 0.53 [0.44, 0.61]; +0.31 [+0.25, +0.38]; 5 |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | 0.88 [0.82, 0.93]; +0.65 [+0.60, +0.70]; 0 | 0.84 [0.78, 0.90]; +0.59 [+0.55, +0.63]; 0 | 0.74 [0.67, 0.80]; +0.47 [+0.40, +0.54]; 12 | 0.74 [0.66, 0.81]; +0.40 [+0.34, +0.46]; 5 |

Random-choice baseline: chosen − unchosen and Spearman are 0. The other statistics per checkpoint, for v = ψ_{t+10} − ψ_t and ψ_T − ψ_0, as random share > median / best cosine in the pool / within-state std of the cosine / reference (demo) candidate:

| velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | 0.50 / +0.99 / +0.34 / +0.99 | 0.51 / +0.99 / +0.34 / +0.99 | 0.51 / +0.99 / +0.33 / +0.99 | 0.50 / +1.00 / +0.28 / +1.00 |
| ψ_T − ψ_0 | 0.51 / +0.65 / +0.24 / +0.52 | 0.51 / +0.61 / +0.21 / +0.50 | 0.51 / +0.54 / +0.32 / +0.15 | 0.51 / +0.39 / +0.14 / +0.19 |

Other-task control: chosen − unchosen when each other task's direction makes the choice, averaged over the other tasks; then own − other [95% interval]:

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | -0.54; +0.70 [+0.59, +0.82] | -0.46; +0.60 [+0.51, +0.69] | -0.39; +0.50 [+0.41, +0.59] | -0.36; +0.45 [+0.37, +0.54] |
| q | ψ_T − ψ_0 | -0.40; +0.60 [+0.53, +0.68] | -0.30; +0.49 [+0.43, +0.55] | -0.13; +0.24 [+0.16, +0.32] | -0.16; +0.21 [+0.17, +0.24] |
| ĝ − ψ | ψ_{t+10} − ψ_t | -0.50; +0.59 [+0.48, +0.70] | -0.38; +0.05 [-0.00, +0.11] | -0.18; +0.02 [+0.01, +0.04] | -0.24; +0.03 [-0.00, +0.06] |
| ĝ − ψ | ψ_T − ψ_0 | -0.34; +0.56 [+0.50, +0.63] | -0.23; +0.05 [+0.01, +0.10] | +0.46; +0.01 [+0.01, +0.02] | +0.02; +0.07 [+0.05, +0.09] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | -0.46; +0.64 [+0.54, +0.75] | -0.26; +0.44 [+0.36, +0.52] | -0.14; -0.07 [-0.14, -0.01] | -0.12; +0.17 [+0.12, +0.23] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | -0.34; +0.53 [+0.46, +0.60] | -0.17; +0.33 [+0.27, +0.38] | +0.28; -0.03 [-0.08, +0.01] | +0.01; +0.08 [+0.06, +0.11] |

Paired differences against fixed:

| score | velocity | statistic | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | chosen − unchosen | -0.01 [-0.05, +0.03] | -0.05 [-0.10, +0.01] | -0.06 [-0.13, -0.01] |
| q | ψ_{t+10} − ψ_t | Spearman(m, cos) | +0.02 [-0.03, +0.07] | -0.03 [-0.11, +0.04] | -0.01 [-0.10, +0.07] |
| q | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.10 [-0.17, -0.02] | -0.20 [-0.29, -0.11] | -0.24 [-0.34, -0.15] |
| q | ψ_T − ψ_0 | chosen − unchosen | -0.02 [-0.04, -0.00] | -0.10 [-0.17, -0.04] | -0.16 [-0.19, -0.12] |
| q | ψ_T − ψ_0 | Spearman(m, cos) | -0.03 [-0.09, +0.03] | -0.62 [-0.72, -0.53] | -0.48 [-0.57, -0.39] |
| q | ψ_T − ψ_0 | own − other chosen − unchosen | -0.11 [-0.16, -0.07] | -0.36 [-0.47, -0.26] | -0.40 [-0.48, -0.32] |
| ĝ − ψ | ψ_{t+10} − ψ_t | chosen − unchosen | -0.41 [-0.51, -0.31] | -0.24 [-0.34, -0.14] | -0.30 [-0.36, -0.24] |
| ĝ − ψ | ψ_{t+10} − ψ_t | Spearman(m, cos) | -0.43 [-0.53, -0.33] | -0.22 [-0.33, -0.11] | -0.54 [-0.63, -0.46] |
| ĝ − ψ | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.54 [-0.66, -0.41] | -0.57 [-0.68, -0.46] | -0.56 [-0.67, -0.46] |
| ĝ − ψ | ψ_T − ψ_0 | chosen − unchosen | -0.40 [-0.46, -0.33] | +0.25 [+0.18, +0.32] | -0.14 [-0.17, -0.10] |
| ĝ − ψ | ψ_T − ψ_0 | Spearman(m, cos) | -0.61 [-0.70, -0.52] | +0.17 [+0.04, +0.27] | -0.37 [-0.47, -0.28] |
| ĝ − ψ | ψ_T − ψ_0 | own − other chosen − unchosen | -0.51 [-0.59, -0.43] | -0.55 [-0.63, -0.48] | -0.50 [-0.57, -0.43] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | chosen − unchosen | +0.00 [-0.03, +0.04] | -0.39 [-0.49, -0.30] | -0.13 [-0.20, -0.07] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | Spearman(m, cos) | +0.02 [-0.03, +0.07] | -0.43 [-0.55, -0.31] | -0.23 [-0.31, -0.15] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.20 [-0.28, -0.12] | -0.72 [-0.84, -0.60] | -0.47 [-0.57, -0.36] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | chosen − unchosen | -0.03 [-0.05, -0.01] | +0.06 [-0.01, +0.13] | -0.09 [-0.13, -0.06] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | Spearman(m, cos) | -0.06 [-0.11, -0.01] | -0.18 [-0.26, -0.10] | -0.25 [-0.32, -0.17] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | own − other chosen − unchosen | -0.20 [-0.25, -0.15] | -0.56 [-0.65, -0.48] | -0.44 [-0.52, -0.37] |

### Selection consistency, π set

129 states; pool of 32 (π's mean chunk and 31 offsets). The tables have the same layout as for the counterfactual set; the reference candidate is π's mean chunk.

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | +0.93 / +0.95; -0.01 [-0.02, -0.00] | +0.94 / +0.94; -0.00 [-0.01, +0.00] | +0.91 / +0.92; -0.01 [-0.02, +0.01] | +0.89 / +0.90; -0.02 [-0.03, -0.00] |
| q | ψ_T − ψ_0 | +0.56 / +0.53; +0.04 [+0.03, +0.05] | +0.53 / +0.50; +0.03 [+0.02, +0.03] | +0.18 / +0.17; +0.01 [-0.02, +0.03] | +0.19 / +0.18; +0.01 [+0.00, +0.02] |
| ĝ − ψ | ψ_{t+10} − ψ_t | +0.93 / +0.95; -0.01 [-0.03, -0.00] | +0.91 / +0.94; -0.03 [-0.07, -0.01] | +0.89 / +0.92; -0.03 [-0.06, -0.01] | +0.88 / +0.91; -0.03 [-0.04, -0.01] |
| ĝ − ψ | ψ_T − ψ_0 | +0.57 / +0.52; +0.04 [+0.04, +0.06] | +0.50 / +0.50; -0.00 [-0.03, +0.02] | +0.32 / +0.16; +0.16 [+0.13, +0.19] | +0.20 / +0.18; +0.03 [+0.02, +0.04] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | +0.94 / +0.95; -0.01 [-0.02, +0.00] | +0.94 / +0.94; -0.01 [-0.02, +0.00] | +0.89 / +0.92; -0.03 [-0.06, -0.00] | +0.90 / +0.90; -0.00 [-0.01, +0.01] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | +0.56 / +0.53; +0.03 [+0.02, +0.04] | +0.52 / +0.50; +0.01 [+0.00, +0.02] | +0.25 / +0.17; +0.08 [+0.06, +0.11] | +0.19 / +0.18; +0.02 [+0.01, +0.03] |

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | 0.32 [0.22, 0.42]; -0.03 [-0.11, +0.06]; 4 | 0.35 [0.27, 0.43]; +0.05 [-0.04, +0.14]; 5 | 0.33 [0.25, 0.42]; +0.04 [-0.07, +0.15]; 1 | 0.40 [0.33, 0.49]; +0.12 [+0.03, +0.21]; 6 |
| q | ψ_T − ψ_0 | 0.88 [0.82, 0.93]; +0.65 [+0.59, +0.71]; 4 | 0.84 [0.78, 0.91]; +0.53 [+0.43, +0.61]; 5 | 0.48 [0.40, 0.57]; +0.01 [-0.11, +0.13]; 1 | 0.60 [0.52, 0.70]; +0.15 [+0.06, +0.25]; 6 |
| ĝ − ψ | ψ_{t+10} − ψ_t | 0.39 [0.31, 0.47]; +0.05 [-0.02, +0.13]; 5 | 0.32 [0.24, 0.40]; -0.10 [-0.20, -0.00]; 33 | 0.29 [0.21, 0.36]; -0.10 [-0.22, +0.02]; 33 | 0.29 [0.23, 0.35]; +0.02 [-0.08, +0.11]; 16 |
| ĝ − ψ | ψ_T − ψ_0 | 0.80 [0.73, 0.86]; +0.55 [+0.48, +0.61]; 5 | 0.53 [0.45, 0.60]; +0.09 [+0.01, +0.17]; 33 | 0.96 [0.90, 1.00]; +0.88 [+0.77, +0.95]; 33 | 0.71 [0.63, 0.78]; +0.28 [+0.21, +0.35]; 16 |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | 0.36 [0.27, 0.44]; +0.03 [-0.05, +0.11]; 3 | 0.40 [0.33, 0.48]; +0.06 [-0.03, +0.14]; 1 | 0.36 [0.27, 0.44]; -0.02 [-0.11, +0.06]; 50 | 0.38 [0.29, 0.47]; +0.08 [-0.01, +0.16]; 16 |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | 0.73 [0.66, 0.80]; +0.38 [+0.32, +0.44]; 3 | 0.64 [0.55, 0.73]; +0.20 [+0.14, +0.27]; 1 | 0.77 [0.70, 0.84]; +0.45 [+0.38, +0.52]; 50 | 0.67 [0.59, 0.74]; +0.21 [+0.12, +0.30]; 16 |

Random share > median / best cosine / within-state std / reference (π's mean chunk):

| velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|
| ψ_{t+10} − ψ_t | 0.50 / +0.97 / +0.03 / +0.95 | 0.50 / +0.97 / +0.02 / +0.95 | 0.50 / +0.97 / +0.04 / +0.93 | 0.50 / +0.96 / +0.04 / +0.91 |
| ψ_T − ψ_0 | 0.50 / +0.58 / +0.03 / +0.53 | 0.50 / +0.55 / +0.02 / +0.51 | 0.50 / +0.33 / +0.07 / +0.17 | 0.50 / +0.24 / +0.03 / +0.18 |

Other-task control:

| score | velocity | fixed | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | -0.04; +0.03 [-0.00, +0.06] | -0.03; +0.02 [-0.00, +0.05] | -0.03; +0.02 [-0.00, +0.06] | -0.05; +0.03 [+0.01, +0.05] |
| q | ψ_T − ψ_0 | -0.03; +0.07 [+0.05, +0.10] | -0.02; +0.05 [+0.03, +0.07] | -0.01; +0.02 [-0.01, +0.05] | -0.04; +0.05 [+0.04, +0.06] |
| ĝ − ψ | ψ_{t+10} − ψ_t | -0.05; +0.03 [+0.00, +0.07] | -0.04; +0.01 [-0.00, +0.02] | -0.03; +0.00 [-0.00, +0.00] | -0.03; +0.00 [-0.00, +0.01] |
| ĝ − ψ | ψ_T − ψ_0 | -0.02; +0.06 [+0.04, +0.10] | -0.01; +0.01 [-0.00, +0.02] | +0.16; +0.00 [+0.00, +0.01] | +0.02; +0.01 [+0.01, +0.02] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | -0.04; +0.03 [+0.01, +0.07] | -0.02; +0.02 [-0.00, +0.05] | -0.03; +0.00 [-0.01, +0.01] | -0.02; +0.02 [+0.01, +0.03] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | -0.02; +0.05 [+0.03, +0.08] | -0.01; +0.03 [+0.01, +0.05] | +0.08; +0.00 [-0.02, +0.02] | +0.00; +0.01 [+0.00, +0.02] |

Paired differences against fixed:

| score | velocity | statistic | flow | flow_full | obs |
|---|---|---|---|---|---|
| q | ψ_{t+10} − ψ_t | chosen − unchosen | +0.01 [-0.00, +0.02] | +0.00 [-0.01, +0.02] | -0.01 [-0.03, +0.02] |
| q | ψ_{t+10} − ψ_t | Spearman(m, cos) | +0.08 [-0.03, +0.18] | +0.07 [-0.06, +0.20] | +0.15 [+0.01, +0.28] |
| q | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.00 [-0.02, +0.01] | -0.00 [-0.02, +0.01] | +0.00 [-0.03, +0.03] |
| q | ψ_T − ψ_0 | chosen − unchosen | -0.01 [-0.03, -0.00] | -0.03 [-0.06, -0.01] | -0.03 [-0.04, -0.01] |
| q | ψ_T − ψ_0 | Spearman(m, cos) | -0.12 [-0.20, -0.05] | -0.64 [-0.77, -0.51] | -0.50 [-0.60, -0.38] |
| q | ψ_T − ψ_0 | own − other chosen − unchosen | -0.02 [-0.04, -0.01] | -0.05 [-0.08, -0.02] | -0.02 [-0.05, +0.01] |
| ĝ − ψ | ψ_{t+10} − ψ_t | chosen − unchosen | -0.02 [-0.06, +0.01] | -0.02 [-0.04, +0.01] | -0.01 [-0.03, +0.00] |
| ĝ − ψ | ψ_{t+10} − ψ_t | Spearman(m, cos) | -0.15 [-0.26, -0.04] | -0.16 [-0.29, -0.03] | -0.03 [-0.14, +0.07] |
| ĝ − ψ | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.03 [-0.07, +0.00] | -0.03 [-0.07, -0.00] | -0.03 [-0.07, -0.00] |
| ĝ − ψ | ψ_T − ψ_0 | chosen − unchosen | -0.05 [-0.08, -0.02] | +0.11 [+0.09, +0.14] | -0.02 [-0.03, -0.00] |
| ĝ − ψ | ψ_T − ψ_0 | Spearman(m, cos) | -0.45 [-0.55, -0.36] | +0.34 [+0.20, +0.44] | -0.27 [-0.35, -0.18] |
| ĝ − ψ | ψ_T − ψ_0 | own − other chosen − unchosen | -0.05 [-0.09, -0.03] | -0.06 [-0.09, -0.03] | -0.05 [-0.08, -0.03] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | chosen − unchosen | +0.00 [-0.01, +0.02] | -0.02 [-0.05, +0.01] | +0.01 [-0.01, +0.02] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | Spearman(m, cos) | +0.03 [-0.06, +0.13] | -0.05 [-0.17, +0.06] | +0.05 [-0.07, +0.17] |
| Uᵀ(ĝ − ψ) | ψ_{t+10} − ψ_t | own − other chosen − unchosen | -0.01 [-0.03, +0.00] | -0.03 [-0.06, -0.00] | -0.01 [-0.05, +0.02] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | chosen − unchosen | -0.02 [-0.03, -0.01] | +0.05 [+0.03, +0.08] | -0.01 [-0.03, +0.00] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | Spearman(m, cos) | -0.18 [-0.26, -0.10] | +0.07 [-0.01, +0.15] | -0.17 [-0.28, -0.06] |
| Uᵀ(ĝ − ψ) | ψ_T − ψ_0 | own − other chosen − unchosen | -0.03 [-0.04, -0.01] | -0.05 [-0.08, -0.03] | -0.04 [-0.07, -0.01] |

By f and every other statistic: `report.md` and `tables.csv`.

## Reading

### Velocity-field consistency

- **q**:
  - fixed is the most consistent text-only checkpoint: 0.40 against ψ_{t+10} − ψ_t and 0.72 against ψ_T − ψ_0, own − other +0.44 / +0.80.
  - flow is lower by 0.06 and 0.08; both intervals exclude 0.
  - flow_full is lower by 0.03 against the displacement and by 0.62 against ψ_T − ψ_0, where it reaches only 0.10. Its q does not point along its episode velocities, which all share one direction.
- **ĝ − ψ in the full space**:
  - flow_full is the only checkpoint above fixed: 0.74 against ψ_T − ψ_0, a difference of +0.26 [+0.18, +0.32].
  - The same cosine with another task's ĝ is 0.69, so own − other is 0.05, against 0.70 for fixed.
  - The degeneracy control shows why. In flow_full, ψ_T − ψ_0 of episodes of different tasks have mean cosine 0.89 (fixed 0.04), and ĝ − ψ_t of states of different tasks 0.90 (fixed 0.00). Every task's offset and every episode's velocity point along the collapsed direction of ψ (effective rank 1.1), so they align whatever the instruction.
  - flow's full-space offset is nearly orthogonal or opposite: −0.15 against ψ_T − ψ_0. The part of ĝ outside span(U) is unconstrained (ISSUE-017).
  - fixed's full-space offset also turns negative in the last quarter (−0.18, −0.22), where its U-coordinate offset is at 0.91 and 0.90.
- **Uᵀ(ĝ − ψ)**:
  - fixed reaches 0.90 / 0.98 and flow 0.91 / 1.00. flow is equal against the displacement (+0.01 [−0.02, +0.03]) and higher by +0.02 [+0.01, +0.05] against ψ_T − ψ_0.
  - flow's own − other is lower, by 0.14 in both cases.
  - Both values are what the goal and path losses train: Uᵀĝ at Uᵀψ_T, and Uᵀψ_t on the start-to-end segment. Within a task, the displacements Uᵀ(ψ_{t+10} − ψ_t) of different episodes have mean cosine 0.82–0.97.
  - flow_full reaches 0.18 / 0.93, with other-task values 0.31 / 0.75.
- **obs (reference)**:
  - Its q is the most task-specific against the local displacement: own − other +0.62 against +0.44, mean +0.02 [+0.00, +0.04].
  - It is lower against ψ_T − ψ_0 (0.31) and in U coordinates, where own − other is +0.16 / −0.06.

### Selection consistency

- **fixed**: on the counterfactual set, every score chooses candidates that move ψ more along the expected velocity than the unchosen ones.
  - chosen − unchosen is +0.09 to +0.22 and Spearman +0.40 to +0.71.
  - Choosing with another task's direction does worse: own − other is +0.53 to +0.70.
  - Against ψ_{t+10} − ψ_t the choice exceeds the unchosen median in only 0.40–0.59 of states (random 0.50). This velocity is the demonstration's own next displacement, and the demo candidate has cosine 0.99 with it. The q match chooses a σ = 2 chunk in 58 of 129 states and the demo chunk in 4.
- **flow**: equal to fixed within the intervals for q and Uᵀ(ĝ − ψ).
  - chosen − unchosen differs by −0.01 to +0.00 against the displacement, and by −0.02 / −0.03 against ψ_T − ψ_0, where the intervals exclude 0.
  - On the counterfactual set, own − other of these two scores is lower by 0.10–0.20.
  - Its full-space ĝ − ψ chooses against both velocities (−0.33, −0.18) and stays in 12 states.
- **flow_full**: above fixed only for the ĝ-based scores against ψ_T − ψ_0.
  - Full-space ĝ − ψ, counterfactual set: +0.47 against +0.22, a difference of +0.25 [+0.18, +0.32], with share 0.91 and Spearman +0.85.
  - Full-space ĝ − ψ, π set: +0.16 against +0.04, a difference of +0.11 [+0.09, +0.14].
  - Uᵀ(ĝ − ψ): +0.06 [−0.01, +0.13] (counterfactual) and +0.05 [+0.03, +0.08] (π).
  - Another task's ĝ produces the same selection. own − other is +0.01 / +0.00 in the full space and −0.03 / +0.00 in U coordinates, against +0.56 / +0.06 and +0.53 / +0.05 for fixed.
  - Every other combination is at or below fixed.
  - The ĝ − ψ scores stay in 17 (counterfactual) and 33 (π) states, and the U-coordinate score in 12 and 50.
- **π set, all checkpoints**: the 32 candidates lie close together.
  - The within-state std of cos(z, ψ_{t+10} − ψ_t) is 0.02–0.04, against 0.28–0.34 in the counterfactual set.
  - Selection against the displacement is at chance or slightly below: chosen − unchosen is −0.03 to −0.00, and Spearman is −0.10 to +0.12.
  - Against ψ_T − ψ_0, fixed's q choice beats the unchosen by +0.04 [+0.03, +0.05], with share 0.88, Spearman +0.65 and own − other +0.07. flow is lower by 0.01 [0.00, 0.03].
- **obs (reference)**: below fixed in every combination on the counterfactual set, and equal or below on the π set.

### Is any checkpoint clearly better than fixed?

No.

- **flow** matches fixed in U coordinates, within ±0.03 on both metrics. Its q velocity consistency is lower by 0.06–0.08, its selections on the counterfactual set are less task-specific (own − other lower by 0.10–0.54), and its full-space ĝ − ψ is not usable.
- **flow_full** is better only for ĝ − ψ against ψ_T − ψ_0.
  - Full space: +0.26 in velocity consistency, +0.25 and +0.11 in selection consistency. Its selection in U coordinates is also slightly above fixed on the π set.
  - Under the other-task control, the own − other margins of these quantities are 0.05, 0.01 and 0.00, against 0.70, 0.56 and 0.06 for fixed.
  - They coincide with the degeneracy: states of different tasks have mean cosine 0.89–0.90 in these vectors.
  - The gain comes from the collapse of ψ, not from consistency with the instruction. Every other quantity of flow_full is at or below fixed.
- **obs**, as reference, has the most task-specific q against the local displacement and is weaker in everything else.

# No-ψ π baseline and inference cost (phase 29)

The proposal head π (`ceam/model/proposal_head.py`) is trained by behaviour cloning on the mean action chunk, conditioned on ψ(o) and q_l of a frozen stage one (`docs/proposal_branch.md`). This page describes an opt-in π input option and a π that takes neither ψ nor q_l, only the frozen DINOv2 tokens, proprio and the frozen SigLIP tokens of the instruction. It compares that π with the π of the fixed, flow (z) and sg[D_ψ] stage ones, three seeds each, in the π-mean closed loop. It also measures the per-decision inference cost of the closed-loop configurations.

## π input options

### Model

`ProposalConfig` has three input switches and two text sizes:

| Field | Default | Meaning |
|---|---|---|
| `state_input` | `true` | ψ(o) [state_dim] is a token (its own linear projection) |
| `query_input` | `true` | q_l [query_dim] is a token (its own linear projection) |
| `instruction_input` | `false` | the cached SigLIP tokens of the instruction are tokens |
| `text_token_dim` | 1152 | SigLIP token width |
| `max_text_tokens` | 64 | cached tokens per instruction |

- **Sequence**: [ψ, q_l, proprio, 2 × 64 DINOv2 tokens, 64 instruction tokens, 10 action slots]; a part is present only when its input is on. The defaults give the sequence of the existing π, [ψ, q_l, proprio, visual tokens, action slots].
- **Instruction tokens**: handled as in `VelocityQuery`. Each token gets a linear projection plus a learned position embedding e_txt_i (initialized from 𝒩(0, 0.02²)). The encoder gets a key padding mask that is True exactly at the padded text positions (text mask False; SigLIP's pad id equals its `</s>` id, so `</s>` is hidden too). Every other position is visible. Without instruction tokens the encoder runs without a mask, as before.
- **Initialization order**: a part whose input is off creates no parameters, and the text parameters come after the step embedding. The default configuration therefore draws its initialization exactly as before.
- **Parameters**: 13,315,079 with the defaults; 13,675,015 for the no-ψ π (no ψ and q_l projections; text projection and text position embedding added).
- **Forward**: `forward(frames, state, query, text=None)`. Each of state, query and text must be given exactly when its input is on (ValueError otherwise).

### Training config and trainer

- **`ProposalTrainingConfig`**: `stage_one_checkpoint` may be null, and a `data` section (`ProposalDataConfig`: `cache_dir`, `dataset_root`, `instructions_path`, `heldout_fraction`, `split_seed`) may be given.
  - A π that takes ψ or q_l needs `stage_one_checkpoint` and takes its data from that stage one's config; `data` must be null.
  - A π that takes neither has no stage one: `stage_one_checkpoint` must be null and `data` given.
  - JSON: the five new proposal fields and `data` may be missing when read, and then take their defaults. Every field is always written.
- **`ProposalTrainer`**:
  - loads a stage one only when π takes ψ or q_l;
  - builds the ψ table only for the ψ input and the q_l table only for the q_l input;
  - with instruction tokens, builds a table of the cached SigLIP tokens and mask of every dataset task's original instruction (`InstructionTable` variants[0], the dataset task text), indexed by task_index. This is the same text, the same per-task choice, and the same absence of paraphrase sampling that q_l uses.
  - Per step, π gets its inputs: ψ(o) = table[current_rows], q_l = directions[task_id], instruction tokens[task_id] with mask[task_id].
- **`ProposalCheckpointStore.load`**: loads the stage one only when the config names one (`LoadedProposal.stage_one` is None otherwise).

### Closed loop

- **`ProposalCandidates`** takes the task's instruction tokens at construction (required exactly when π takes them). `propose(encoded, psi, query, stream)` passes π only its own inputs; psi and query may be None when π does not take them.
- **`ClosedLoopRunner`** is unchanged: it passes the encoded frames, ψ(o) and q_l, and `ProposalCandidates` selects π's inputs.
- **`scripts/run_closed_loop.py`**: with instruction tokens, π gets the task's original instruction (the text q_l is computed from) as cached SigLIP tokens.
  - For a π without a stage one, the stage-one check is skipped. `--checkpoint` then serves only the loop itself: the ψ the runner computes and logs, the retrieval tables, the start-observation check and the scene probe's task relations. It must share π's feature cache, dataset root and instruction table (ValueError otherwise).
  - The "proposal head" line prints π's inputs.
- **`scripts/train_proposal.py`** prints π's inputs, stage one and data.
- **`scripts/evaluate_proposals.py`** refuses a π without a stage one.

### Default behaviour

The existing configs and checkpoints behave bit-identically. The base code is `e2b0082`, the new code is `dd723a0`, and both ran on node-65 CUDA 0. The dump and comparison scripts are R `verify_dump.py` and `verify_compare.py` (R = `tmp/runlogs/s3p29/`).

| Check | Result |
|---|---|
| `ProposalHead(ProposalConfig())` after `torch.manual_seed(0)` | 63 tensors, bit-identical |
| `goal_proposal`, `goal_proposal_flow`, `goal_proposal_flow_full_sgd` configs | parse to the old fields plus the defaults; round trip equal |
| `goal_proposal_flow` step 3000 on 256 feature-cache rows, float32 and bf16 autocast | bit-identical (max \|Δ\| 0) |
| `ProposalCandidates` K = 32 on 8 frames, raw and standardized | bit-identical |
| 30 training steps of `goal_proposal_flow`, every step logged, twice with the new code | mse, arm, gripper, gripper accuracy, grad norm and lr equal at all 30 steps; step-30 parameters bit-identical |
| π-mean closed loop, flow π, task 0, inits 0–4, new code twice (`outputs/closed_loop/p29_checks/mean_flow_t0_{new,new2}`) | results equal to `v7_mean_flow` in 5 / 5 both times; traces equal in 4 / 5 and 5 / 5 |

- **The one trace difference** (init 3 of the first run): the scene before decision 3 differs from `v7_mean_flow` by less than 1e-6 m in the end-effector position.
- **Base code in the same loop** (`p29_checks/mean_flow_t0_base`): success equals `v7_mean_flow` in 5 of 5 episodes. In 3 of them the episode takes 1–5 env steps more or fewer.
- π's forward and candidates are bit-identical between the two codes (above), so these differences arise in the simulated episode, as between runs of one π in `docs/proposal_branch.md`.

No-ψ checks (CPU, R `cpu_checks.py`):

- Changing the padded text tokens leaves the output unchanged (max |Δ| 0); changing a visible token changes it (0.19).
- ψ given to the no-ψ head, missing text, text given to a default head, a no-ψ config with a stage one or without data, and a ψ config without a stage one are each refused.

## Configs and training

`configs/proposal/goal_proposal_nopsi_seed{0,1,2}.json` differ from `goal_proposal_flow.json` only in:

- `stage_one_checkpoint` null;
- `state_input` false, `query_input` false, `instruction_input` true;
- the `data` section, which holds the values of the stage-one configs: feature cache `outputs/feature_cache/libero_goal`, dataset `libero_goal_no_noops_lerobot`, instructions `data/instructions/libero_goal.json`, heldout fraction 0.1, split seed 0;
- `seed` and `run_dir`.

The optimizer (AdamW, lr 3e-4, weight decay 0.05, 150 warmup, 3000 steps), batch (256 transitions, chunk 10), bf16 autocast and gradient clipping are unchanged.

- **Run directories**: `outputs/runs/goal_proposal_nopsi_seed0`, `goal_proposal_nopsi_seed1` and `goal_proposal_nopsi_seed2b`. `goal_proposal_nopsi_seed2` holds an incomplete first seed-2 run (checkpoints 500, 1000, 1500; see "Placement"); it is not used.
- **What the seed sets**: the seed is set with `torch.manual_seed` before π is built, and it also seeds the TRAIN sampler.
  - Seed k gives the no-ψ π the same batch order as the ψ π of seed k.
  - It does not give the same initialization, because the no-ψ π creates a different set of parameters.
- **Training**: 3000 updates in 332, 341 and 359 s (8.4–9.0 steps/s), with 4.66 GiB peak GPU memory. The ψ π took 3.30 GiB; the no-ψ sequence is 203 tokens instead of 141.
- **TRAIN MSE at the last step**: 0.0156, 0.0139 and 0.0149. The nine ψ π reach 0.0135–0.0156 (`docs/proposal_branch.md`).

## Closed loop

The protocol is the `v7_mean_flow` command line (R `run_one.sh`, `launch_mean.sh`) with only `--proposal-checkpoint` changed:

- `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`, scales 0.25/0.5/1.0, dct 1, proposal seed 0 (K = 1 draws no noise);
- env seed 0, 300 steps, 10 settle steps, benchmark inits 0–49, traced, `--check-start-observation`;
- `--checkpoint` the flow (z) stage one `goal_instruction_contrast_flow` step 5000, and `--state-table` its ψ table. The no-ψ π takes nothing from them.

Runs: `outputs/closed_loop/v9_mean_nopsi_s{0,1,2}`, π step 3000.

- **Start states**: for every task of every run, the start-observation block (image means, token cosines, proprio distance, ψ distances) is identical to `v7_mean_flow`'s (R `start_observation_check.txt`). The scene after the reset (`EpisodeTrace.initial`) of all 1500 episodes equals `v7_mean_flow`'s.
- **Per episode**: 12.4, 12.8 and 12.8 decisions and 119.5, 124.3 and 124.2 env steps. The ψ π take 13.0–15.6 decisions.
- **Analysis**:
  - R `nopsi_seed_analysis.py` (output `nopsi_seed_analysis.txt`) extends the phase-28 analysis (`tmp/runlogs/s3p28/seed_analysis.py`; unchanged copy R `seed_analysis_p28_copy.py`) to four versions;
  - `summarize_closed_loop.py` against `v7_mean_flow` (R `v9_mean_nopsi_s*_summary.txt`);
  - `classify_closed_loop_failures.py` (R `v9_mean_nopsi_s*_failure_classification.txt`);
  - R `analyze_nopsi.py` (initial-scene check);
  - R `heldout_mse.py` (HELDOUT MSE).

The ψ π runs are:

- fixed: `v6_mean_default`, `v8_mean_fixed_s{1,2}`;
- flow (z): `v7_mean_flow`, `v8_mean_flow_s{1,2}`;
- sg[D_ψ]: `v7_mean_flow_full_sgd`, `v8_mean_flow_full_sgd_s{1,2}`.

### Success per seed

| π | seed 0 | seed 1 | seed 2 | mean | SD over seeds | range |
|---|---|---|---|---|---|---|
| no ψ | 0.940 | 0.922 | 0.918 | **0.927** | 0.012 | 0.022 |
| fixed ψ | 0.828 | 0.858 | 0.872 | **0.853** | 0.022 | 0.044 |
| flow (z) ψ | 0.904 | 0.796 | 0.884 | **0.861** | 0.057 | 0.108 |
| sg[D_ψ] ψ | 0.906 | 0.806 | 0.906 | **0.873** | 0.058 | 0.100 |

- **Binomial noise**: the SD of one 500-episode success rate is 0.012 at 0.927, and 0.015–0.016 for the ψ π.
- **Rank**: the three no-ψ runs are the three highest of the twelve.

### Paired comparisons

Exact two-sided McNemar by (task, init), no-ψ π against the ψ π of the same seed. Each cell gives the episodes only the no-ψ π solves / only the ψ π solves, and p. "Pooled" sums the discordant counts of the three seeds.

| ψ π | tasks | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|---|
| fixed | all | 83 / 27, 8.3e-8 | 62 / 30, 0.0011 | 57 / 34, 0.021 | 202 / 91, 7.8e-11 |
| fixed | 3 and 4 | 33 / 2, 3.7e-8 | 19 / 0, 3.8e-6 | 17 / 2, 7.3e-4 | 69 / 4, 2.4e-16 |
| fixed | other 8 | 50 / 25, 0.0052 | 43 / 30, 0.16 | 40 / 32, 0.41 | 133 / 87, 0.0023 |
| flow (z) | all | 43 / 25, 0.038 | 97 / 34, 3.3e-8 | 50 / 33, 0.078 | 190 / 92, 5.4e-9 |
| flow (z) | 3 and 4 | 3 / 2, 1 | 46 / 0, 2.8e-14 | 11 / 3, 0.057 | 60 / 5, 4.9e-13 |
| flow (z) | other 8 | 40 / 23, 0.043 | 51 / 34, 0.082 | 39 / 30, 0.34 | 130 / 87, 0.0042 |
| sg[D_ψ] | all | 41 / 24, 0.046 | 96 / 38, 5.7e-7 | 41 / 35, 0.57 | 178 / 97, 1.2e-6 |
| sg[D_ψ] | 3 and 4 | 1 / 2, 1 | 58 / 0, 6.9e-18 | 8 / 3, 0.23 | 67 / 5, 6.4e-15 |
| sg[D_ψ] | other 8 | 40 / 22, 0.030 | 38 / 38, 1 | 33 / 32, 1 | 111 / 92, 0.21 |

- **Between the no-ψ seeds**: seed 1 vs 0 gives 26 / 35 (p = 0.31), seed 2 vs 0 gives 26 / 37 (p = 0.21), and seed 2 vs 1 gives 34 / 36 (p = 0.90). In tasks 3 and 4: 2 / 0, 2 / 3 and 0 / 3.
- **Discordant episodes** out of 500 (mean over the three pairs):
  - seeds of the no-ψ π: 61, 63, 70 (65);
  - no ψ against fixed at the same seed: 110, 92, 91 (98);
  - no ψ against flow (z) at the same seed: 68, 131, 83 (94);
  - no ψ against sg[D_ψ] at the same seed: 65, 134, 76 (92).
  - For comparison, the seeds of one ψ stage one differ in 74–134 episodes (`docs/proposal_branch.md`).

### Version against seed

The unit is one run's success rate: four versions × three seeds. The tests are those of phase 28 (`docs/proposal_branch.md`, "Stage one against seed over the nine runs"), extended to four versions, and repeated for the no-ψ π against the nine ψ runs pooled.

(A) Four versions:

| Test | all tasks | tasks 3 and 4 | other 8 tasks |
|---|---|---|---|
| SD of the four version means / pooled SD of the seeds within version | 0.033 / 0.043 | 0.102 / 0.198 | 0.018 / 0.013 |
| one-way F(3, 8), p | 1.82, 0.22 | 0.80, 0.53 | 6.14, 0.018 |
| exact permutation of the one-way F (15,400 partitions into four groups of 3), p | 0.21 | 0.53 | 0.014 |
| two-way version × seed: version F(3, 6), p | 2.46, 0.16 | 0.96, 0.47 | 5.66, 0.035 |
| two-way version × seed: seed F(2, 6), p | 2.40, 0.17 | 1.78, 0.25 | 0.69, 0.54 |
| exact permutation, no ψ against fixed / flow (z) / sg[D_ψ] (20 assignments each), one-sided p | 0.05 / 0.05 / 0.05 | 0.05 / 0.10 / 0.15 | 0.05 / 0.05 / 0.15 |
| bootstrap over seeds, no ψ − fixed, 95% interval | +0.074 [+0.052, +0.097] | +0.217 [+0.157, +0.310] | +0.038 [+0.023, +0.055] |
| bootstrap over seeds, no ψ − flow (z) | +0.065 [+0.022, +0.129] | +0.183 [+0.017, +0.440] | +0.036 [+0.018, +0.054] |
| bootstrap over seeds, no ψ − sg[D_ψ] | +0.054 [+0.013, +0.116] | +0.207 [−0.003, +0.560] | +0.016 [+0.001, +0.033] |

- The smallest attainable one-sided p with three runs per arm is 0.05, and the smallest two-sided p is 0.10.
- The bootstrap resamples each version's three rates with replacement (20,000 draws, seed 0).
- Between the three ψ versions the bootstrap reproduces the phase-28 values: flow (z) − fixed +0.009 [−0.052, +0.059], sg[D_ψ] − fixed +0.020 [−0.042, +0.068], sg[D_ψ] − flow (z) +0.011 [−0.058, +0.081].

(B) No ψ (3 runs) against the nine ψ runs pooled:

| Test | all tasks | tasks 3 and 4 | other 8 tasks |
|---|---|---|---|
| mean: no ψ / ψ pooled | 0.927 / 0.862 | 0.983 / 0.781 | 0.913 / 0.883 |
| SD of the two group means / pooled SD within the groups | 0.046 / 0.039 | 0.143 / 0.177 | 0.021 / 0.015 |
| SD over runs: no ψ / ψ pooled | 0.012 / 0.043 | 0.015 / 0.198 | 0.015 / 0.015 |
| one-way F(1, 10), p | 6.16, 0.032 | 2.93, 0.12 | 9.26, 0.012 |
| exact permutation (220 choices of 3 of the 12 runs), one-sided / two-sided p | 0.0045 / 0.027 | 0.023 / 0.096 | 0.014 / 0.018 |
| two-way (no ψ / mean of the three ψ at the seed) × seed: group F(1, 2), p | 9.71, 0.089 | 3.79, 0.19 | 8.31, 0.10 |
| same, seed F(2, 2), p | 1.32, 0.43 | 0.71, 0.59 | 0.69, 0.59 |
| bootstrap, no ψ − ψ pooled, 95% interval | +0.064 [+0.038, +0.094] | +0.202 [+0.089, +0.329] | +0.030 [+0.017, +0.046] |

- Over all tasks, 0.0045 (1/220) is the smallest attainable one-sided p.
- The bootstrap resamples the no-ψ rates, and each ψ version's rates within the version.
- The mean of the three ψ π per seed is 0.879, 0.820 and 0.887; the no-ψ π reaches 0.940, 0.922 and 0.918.
- The split into tasks 3 and 4 and the other eight tasks is the one phase 28 chose after seeing where the ψ-π seed variance is.

### Per task

Success per task, 50 episodes per run:

| Task | Text | no ψ s0 / s1 / s2 | mean: no ψ / fixed / flow (z) / sg[D_ψ] |
|---|---|---|---|
| 0 | open the middle drawer of the cabinet | 0.94 / 0.92 / 0.88 | 0.913 / 0.967 / 0.973 / 0.973 |
| 1 | open the top drawer and put the bowl inside | 0.84 / 0.80 / 0.82 | 0.820 / 0.847 / 0.780 / 0.820 |
| 2 | push the plate to the front of the stove | 0.96 / 0.98 / 0.98 | 0.973 / 0.940 / 0.920 / 0.920 |
| 3 | put the bowl on the plate | 0.98 / 1.00 / 0.98 | 0.987 / 0.740 / 0.813 / 0.800 |
| 4 | put the bowl on the stove | 0.98 / 1.00 / 0.96 | 0.980 / 0.793 / 0.787 / 0.753 |
| 5 | put the bowl on top of the cabinet | 0.98 / 0.94 / 0.96 | 0.960 / 0.880 / 0.807 / 0.840 |
| 6 | put the cream cheese in the bowl | 0.90 / 0.80 / 0.88 | 0.860 / 0.680 / 0.793 / 0.813 |
| 7 | put the wine bottle on the rack | 0.88 / 0.88 / 0.86 | 0.873 / 0.767 / 0.847 / 0.880 |
| 8 | put the wine bottle on top of the cabinet | 0.96 / 0.90 / 0.86 | 0.907 / 0.920 / 0.893 / 0.933 |
| 9 | turn on the stove | 0.98 / 1.00 / 1.00 | 0.993 / 0.993 / 1.000 / 0.993 |
| **all** | | **0.940 / 0.922 / 0.918** | **0.927 / 0.853 / 0.861 / 0.873** |

The per-seed values of the ψ π are in `docs/proposal_branch.md`, "π's mean chunk: success per seed".

Tasks 3 and 4 (the bowl-placing tasks), 100 episodes per run:

| π | seed 0 | seed 1 | seed 2 | mean | per-task SD over seeds, task 3 / task 4 |
|---|---|---|---|---|---|
| no ψ | 0.98 | 1.00 | 0.97 | 0.983 | 0.012 / 0.020 |
| fixed | 0.67 | 0.81 | 0.82 | 0.767 | 0.060 / 0.175 |
| flow (z) | 0.97 | 0.54 | 0.89 | 0.800 | 0.133 / 0.352 |
| sg[D_ψ] | 0.99 | 0.42 | 0.92 | 0.777 | 0.262 / 0.360 |

The other eight tasks, 400 episodes per run:

| π | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| no ψ | 0.930 | 0.902 | 0.905 | 0.913 |
| fixed | 0.868 | 0.870 | 0.885 | 0.874 |
| flow (z) | 0.887 | 0.860 | 0.882 | 0.877 |
| sg[D_ψ] | 0.885 | 0.902 | 0.902 | 0.897 |

Failure categories of the no-ψ runs (`classify_closed_loop_failures.py`; no episode is stuck):

| Run | success | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint | lifted, not at target, in tasks 3 and 4 |
|---|---|---|---|---|---|---|---|
| no ψ, seed 0 | 470 | 2 | 3 | 16 | 7 | 2 | 2 |
| no ψ, seed 1 | 461 | 2 | 5 | 20 | 11 | 1 | 0 |
| no ψ, seed 2 | 459 | 4 | 5 | 15 | 10 | 7 | 3 |

In the ψ runs this failure (lifted, not at target, in tasks 3 and 4) counts 31, 18 and 17 on the fixed stage one, 2, 45 and 11 on flow (z), and 0, 57 and 8 on sg[D_ψ] (`docs/proposal_branch.md`).

### HELDOUT MSE

MSE of π's standardized mean chunk against the demonstration chunk on the 4864 HELDOUT (episode, t) at step 3000 (R `heldout_mse.py`, output `heldout_mse.txt`; float32, ψ recomputed with the π's frozen trunk). The nine ψ values equal the phase-28 `evaluate_proposals.py curve` values to the four printed decimals.

| π | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| no ψ | 0.2535 | 0.2640 | 0.2648 | 0.2608 ± 0.0063 |
| fixed | 0.2736 | 0.2648 | 0.2730 | 0.2705 ± 0.0049 |
| flow (z) | 0.2631 | 0.2783 | 0.2657 | 0.2690 ± 0.0081 |
| sg[D_ψ] | 0.2612 | 0.2744 | 0.2655 | 0.2670 ± 0.0067 |

- **Gripper agreement** of the no-ψ π on HELDOUT: 0.961–0.964, against 0.960–0.963 for the ψ π.
- **Correlation with success**: over the twelve π, HELDOUT MSE and closed-loop success correlate with Pearson r = −0.89 and Spearman ρ = −0.84. Over the nine ψ π the values are −0.88 and −0.85.

### Reading

- **The π without ψ and q_l succeeds more often than every ψ π.**
  - Over three seeds it averages 0.927, against 0.853–0.873 for the three stage ones.
  - Each of its seeds (0.918–0.940) is above every one of the nine ψ runs (0.796–0.906).
  - Against the ψ π of the same seed, it solves more episodes alone than it loses in all nine pairs. Pooled over seeds: 202 / 91, 190 / 92 and 178 / 97.
  - Against the nine ψ runs pooled, the seed-level tests give one-sided permutation p = 0.0045 (the smallest attainable), F(1, 10) p = 0.032, and a bootstrap interval of +0.038 to +0.094.
  - The two-way test with seed as a block gives p = 0.089, on 2 residual degrees of freedom.
- **The seed-level tests of the four versions do not separate them** (F(3, 8) p = 0.22, permutation p = 0.21). The ψ π's seed spread dominates that test.
  - With three runs per arm, no pairwise permutation test can go below one-sided p = 0.05. No ψ against each ψ version reaches that bound over all tasks.
- **Tasks 3 and 4**:
  - The no-ψ π places the bowl in 0.97–1.00 of these episodes at every seed (SD 0.015), and lifts it without placing it in 0–3 episodes per run.
  - The ψ π range from 0.42 to 0.99 there, and fail this way in up to 57 episodes. Pooled over seeds, the discordant counts are 69 / 4, 60 / 5 and 67 / 5.
- **The other eight tasks**:
  - The no-ψ π is ahead of fixed and flow (z) (bootstrap +0.038 and +0.036). Against sg[D_ψ] it is +0.016 [+0.001, +0.033], with pooled McNemar 111 / 92 (p = 0.21).
  - On task 0 it is lower than every ψ version (0.913 against 0.967–0.973). Of its 13 failures there, 12 leave the drawer unmoved and 1 opens it partly.
- **Seed stability**: the no-ψ seeds differ in 61–70 episodes (no pair with p < 0.2); the seeds of one ψ stage one differ in 74–134. The no-ψ π's run SD (0.012) equals the binomial SD of one run.
- **HELDOUT MSE**: the no-ψ π has the lowest mean HELDOUT MSE (0.261 against 0.267–0.271), and the ranking by HELDOUT MSE follows the closed-loop ranking across all twelve π (ρ = −0.84).
- **Scope**:
  - The no-ψ π and the ψ π share the recipe, the data, the batch order of each seed and the evaluation protocol.
  - They differ in π's inputs: ψ and q_l against the instruction tokens.
  - They also differ in π's initialization at a given seed.

## Inference cost

`scripts/measure_efficiency.py` runs each configuration's decision the way `run_closed_loop.py` does: batch 1, float32 without autocast, under inference mode, without the simulator.

- **Setup**: node-65, CUDA 0 (L40S), 2026-09-27 23:51. No other process was on CUDA 0; a no-ψ training ran on CUDA 1. Load average 4.6 at the start and 3.5 at the end. Code `27fc147`.
- **Timing**: 20 warm-up decisions, then 300 timed decisions (100 calls for the per-episode rows). Each decision sits between `torch.cuda.synchronize` calls. The input frames cycle over 32 feature-cache rows spread over the cache.
- **Output**: `outputs/efficiency/no_psi_baseline_nnmc65_cuda0.json`; log R `logs/efficiency_nnmc65_cuda0.log`.

The rows:

- **dinov2_encode** (shared, per decision): DINOv2 on the two 512 × 512 views, each its own batch, as `ObservationEncoder` runs it, plus the proprio standardization. The images are random uint8; the cost does not depend on the pixel values.
- **siglip_text** (shared, per episode): the SigLIP text tower on one instruction. The closed loop reads the cached tokens instead.
- **q_l_query** (per episode): the query module's q_l, for the ψ configurations.
- **(a) psi_mean**: flow (z) ψ π's mean chunk: the trunk's ψ(o), then π through `ProposalCandidates` at K = 1.
- **(b) candidates_K8 / K32**: the MATCH setting of `v7_proposal_small_vrem_flow` at its first level. That is ψ(o), π's mean chunk and K − 1 noisy candidates (scales 0.05/0.1/0.2, dct 1, seed 0, no gripper flip), Δ (`goal_dynamics_perturb_flow`) on the K chunks, and cos scoring with `CandidateSelector` against q_l (query) or against v(o, l) of `goal_velocity_remaining_flow`, computed per decision (velocity).
- **(c) nopsi_mean**: the no-ψ π's mean chunk (seed 0) with the instruction tokens, K = 1. No stage one runs.

The columns:

- **active params**: the parameters of the modules that run at every decision; for the per-episode rows, at every episode.
- **param × samples**: Σ parameters × samples each module processes per decision; Δ runs on K samples.
- **latency**: median, with p10–p90 in brackets.
- **end to end**: the median time of DINOv2 on the two views plus the configuration's decision, fed with the encoded frames.
- **weights**: the GPU memory of the modules that run per decision. The DINOv2 encoder and the per-episode query are loaded before and not counted.
- **peak**: `max_memory_allocated` during the timed decisions above the allocation before those modules were loaded, so their weights plus activations.
- **end-to-end peak**: the absolute peak during the end-to-end decisions, including DINOv2's weights (331 MiB).

| Row | Scope | Modules | Active params | Param × samples | Latency, ms | End to end, ms | Weights, MiB | Peak, MiB | End-to-end peak, MiB |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_encode | per decision, shared | DINOv2 ViT-B/14, 2 views | 86.58 M | 173.16 M | 11.45 (11.04–12.14) | – | 331 | 342 | – |
| siglip_text | per episode, shared | SigLIP so400m text tower | 449.73 M | 449.73 M | 10.23 (9.95–10.56) | – | 1716 | 1727 | – |
| q_l_query | per episode, ψ configurations | query module | 9.43 M | 9.43 M | 1.59 (1.58–1.63) | – | 37 | 38 | – |
| (a) psi_mean | per decision | trunk, π (ψ) | 26.49 M | 26.49 M | 2.96 (2.92–3.00) | 14.18 | 103 | 108 | 497 |
| (b) candidates_K8_query | per decision | trunk, π (ψ), Δ × 8 | 39.81 M | 133.02 M | 4.97 (4.91–5.08) | 16.28 | 155 | 179 | 562 |
| (b) candidates_K8_velocity | per decision | trunk, π (ψ), Δ × 8, v | 53.74 M | 146.95 M | 6.54 (6.48–6.72) | 17.83 | 209 | 234 | 616 |
| (b) candidates_K32_query | per decision | trunk, π (ψ), Δ × 32 | 39.81 M | 452.60 M | 7.95 (7.89–8.07) | 19.09 | 155 | 245 | 628 |
| (b) candidates_K32_velocity | per decision | trunk, π (ψ), Δ × 32, v | 53.74 M | 466.53 M | 9.49 (9.42–9.63) | 20.71 | 209 | 299 | 681 |
| (c) nopsi_mean | per decision | π (no ψ) | 13.68 M | 13.68 M | 1.98 (1.97–2.01) | 13.19 | 54 | 62 | 411 |

- **The frozen DINOv2 encoding dominates every configuration**: 11.5 ms of the 13.2–20.7 ms per decision.
- **Beyond the encoder**:
  - the no-ψ π costs 2.0 ms and 13.7 M parameters per decision;
  - ψ π's mean chunk costs 3.0 ms and 26.5 M (trunk and π), plus q_l once per episode (1.6 ms, 9.4 M);
  - candidates with Δ and the match cost 5.0 ms (K = 8) and 8.0 ms (K = 32) against q_l, and 6.5 and 9.5 ms against v, with 39.8 M and 53.7 M active parameters.
- **Memory**: the GPU memory of the per-decision modules is 54 MiB (no-ψ π) to 209 MiB (trunk, π, Δ, v). With Δ, the peak exceeds the weights by 24 MiB at K = 8 and 90 MiB at K = 32.
- **The SigLIP text tower** (1716 MiB, 10.2 ms) runs once per instruction; the closed loop reads its tokens from the feature cache.

## Placement and GPUs

All jobs ran on node-65 (L40S), 2026-09-27 to 28, with the launchers in R. No process of another user was on CUDA 0–2 at any launch (nvidia-smi before each launch).

| Job | CUDA | Time | Code |
|---|---|---|---|
| bit-identity checks (dumps, 30-step trainings) | 0 | 23:37–23:39 | `e2b0082` / `dd723a0` |
| π training, seeds 0 / 1 / 2 | 0 / 1 / 2 | 23:39–23:46 (seed 2 stopped at 23:43) | `dd723a0` |
| closed-loop checks, task 0 (`p29_checks`) | 0 | 23:40–23:42 | `e2b0082` / `dd723a0` |
| `v9_mean_nopsi_s0` / `v9_mean_nopsi_s1`, 10 task processes each | 0 / 1 | 23:45–23:50 | `dd723a0` |
| π training, seed 2 (`goal_proposal_nopsi_seed2b`) | 1 | 23:50–23:57 | `4834eb5` |
| inference-cost measurement | 0 | 23:51–23:52 | `27fc147` |
| `v9_mean_nopsi_s2`: even tasks / odd tasks | 0 / 1 | 23:57–00:01 | `4834eb5` |
| HELDOUT MSE | 0 | 23:57–23:59 | `4834eb5` |

- **GPU cap**: a cap of two GPUs per node took effect at 23:42, during the three trainings. From then on only CUDA 0 and 1 were used.
  - The seed-2 training on CUDA 2 was stopped (SIGTERM, at about update 1750) after its step-1500 checkpoint had been written. `outputs/runs/goal_proposal_nopsi_seed2` keeps that incomplete run.
  - Seed 2 was trained again, with the same config and a new `run_dir`, into `goal_proposal_nopsi_seed2b` on CUDA 1.
- **Commits**: `4834eb5` changes only the seed-2 `run_dir` relative to `dd723a0`; `27fc147` adds `scripts/measure_efficiency.py`.
- **Timing**: closed-loop task processes took 1.6–4.5 min.
- **`/tmp/robosuite.log`** on node-65 was last modified at 23:13:25, before these jobs.

## Paths

| What | Where |
|---|---|
| configs | `configs/proposal/goal_proposal_nopsi_seed{0,1,2}.json` |
| π runs | `outputs/runs/goal_proposal_nopsi_seed{0,1,2b}` (step 3000 used); `goal_proposal_nopsi_seed2` incomplete |
| closed loops | `outputs/closed_loop/v9_mean_nopsi_s{0,1,2}/task<t>_mean/{results,trace}.jsonl` |
| closed-loop checks | `outputs/closed_loop/p29_checks/{mean_flow_t0_new,mean_flow_t0_new2,mean_flow_t0_base,nopsi_smoke_t0}` |
| inference cost | `outputs/efficiency/no_psi_baseline_nnmc65_cuda0.json`, `scripts/measure_efficiency.py` |
| launchers, logs, analysis | `tmp/runlogs/s3p29/` (`env.sh`, `run_one.sh`, `launch_train.sh`, `launch_mean.sh`, `efficiency.sh`, `heldout_run.sh`, `analyze.sh`, `logs/`, `nopsi_seed_analysis.{py,txt}`, `heldout_mse.{py,txt}`, `analyze_nopsi.{py,txt}`, `start_observation_check.txt`, `v9_mean_nopsi_s*_{summary,failure_classification}.txt`, `verify_*`, `cpu_checks.py`, `code_base/`) |

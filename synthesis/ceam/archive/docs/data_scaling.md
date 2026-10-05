# Data scaling: π success against the share of TRAIN demonstrations (phase 31)

This page compares two π at 25 %, 50 % and 100 % of the LIBERO-Goal TRAIN demonstrations:

- the no-ψ π (`docs/no_psi_baseline.md`), which reads the frozen DINOv2 tokens, proprio and the frozen SigLIP instruction tokens;
- the projected-flow ψ π (flow (z)), which reads ψ(o) and q_l of a frozen stage one trained with the `goal_instruction_contrast_flow` recipe.

At 25 % and 50 %, both π and the flow stage one are retrained on the same per-task subset. Every π has three seeds (0, 1, 2). Each π runs in the π-mean closed loop, 10 tasks × 50 benchmark inits. The 100 % points are the existing runs.

## Result

**ψ does not help at low data; the flow ψ π falls further behind the no-ψ π as the data shrinks.**

| π | 25 % | 50 % | 100 % |
|---|---|---|---|
| no ψ, mean ± SD over 3 seeds | 0.869 ± 0.006 | 0.899 ± 0.008 | 0.927 ± 0.012 |
| flow (z) ψ, mean ± SD over 3 seeds | 0.527 ± 0.054 | 0.795 ± 0.036 | 0.861 ± 0.058 |
| no ψ − flow (z), bootstrap 95 % interval | +0.343 [+0.290, +0.397] | +0.103 [+0.065, +0.134] | +0.065 [+0.022, +0.129] |
| paired McNemar pooled over seeds, only no ψ / only flow | 604 / 90, p = 2.4e-94 | 252 / 97, p = 5.1e-17 | 190 / 92, p = 5.4e-9 |

- **Every no-ψ run beats every flow run at the same fraction**, and the lead grows as the data shrinks:
  - at 25 %: 0.862–0.874 against 0.472–0.580;
  - at 50 %: 0.890–0.904 against 0.764–0.834.
- **Version × fraction interaction**: F(2, 12) = 26.35, p < 0.0001 (two-way ANOVA, seeds as replicates).
  - From 100 % to 25 %, the gap grows by +0.277 [+0.203, +0.347].
  - From 100 % to 50 %, it grows by +0.038 [−0.029, +0.097].
- **Cost of going from 100 % to 25 %**:
  - the no-ψ π loses 0.057 [0.046, 0.071];
  - the flow ψ π loses 0.335 [0.263, 0.407].
- **Seed spread**: the flow ψ π's seed SD is 0.036–0.058 at every fraction. The no-ψ π's is 0.006–0.012, about the binomial SD of one run (0.012–0.015).

## TRAIN subset

`TrainSubset(fraction, seed)` is defined in `ceam/data/train_subset.py`. Stage one and π both use this class, so they select the same episodes.

- **Selection.** One `numpy.random.default_rng(seed)` is used across the tasks, in task_id order. For each task it permutes the n TRAIN episode ids, sorted ascending, and keeps the first ceil(fraction · n). Before the ceiling, fraction · n is rounded to 9 decimals.
- **Nesting.** The permutations do not depend on the fraction. For one seed, a smaller fraction selects a subset of what a larger fraction selects, and fraction 1 keeps all of TRAIN.
- **Unchanged parts.** HELDOUT (heldout_fraction 0.1, split seed 0) and the task list stay the same.
- **API.**
  - `apply(index)` returns the `EpisodeIndex` restricted to the selected TRAIN episodes.
  - `write_selection(index, path)` writes `train_episodes.json`. It holds fraction, seed, n_train and n_selected, plus, per task, task_id, task_text, n_train, n_selected and episode_ids.

**Stage one.**

- **Config.** `StageOneConfig.train_subset` is `None` or a `TrainSubset`, stored under the JSON key `train_subset`, e.g. `{"fraction": 0.25, "seed": 0}`. When the key is missing it reads as `None`, and it is written only when set.
- **Trainer.** With a subset, `StageOneTrainer` writes `run_dir/train_episodes.json` and passes the restricted index to `StageOneSampler`.

**π.**

- **Config.** `ProposalDataConfig.train_subset` works the same way. In JSON it is `data.train_subset`: optional, and written only when set.
- **Where π gets its subset.**
  - A π on a stage one (flow ψ) takes `train_subset` from that stage one's stored config. `ProposalTrainer` builds its `ProposalDataConfig` from the stage-one config, including `train_subset`.
  - The no-ψ π takes it from its own `data` section.
- **Trainer.** With a subset, `ProposalTrainer` writes `run_dir/train_episodes.json` and passes `TrainSubset.apply(index)` to `StageTwoSampler`.
- **Evaluation data.** The ψ table and the task directions cover the whole feature cache. The closed loop and `scripts/diagnose_stage_one.py` build the full index.

### Subset sizes

LIBERO-Goal has 390 TRAIN and 43 HELDOUT episodes. The subsets use seed 0.

| task | n TRAIN | 25 % | 50 % |
|---|---|---|---|
| 0 open the middle drawer of the cabinet | 38 | 10 | 19 |
| 1 open the top drawer and put the bowl inside | 30 | 8 | 15 |
| 2 push the plate to the front of the stove | 35 | 9 | 18 |
| 3 put the bowl on the plate | 45 | 12 | 23 |
| 4 put the bowl on the stove | 45 | 12 | 23 |
| 5 put the bowl on top of the cabinet | 42 | 11 | 21 |
| 6 put the cream cheese in the bowl | 35 | 9 | 18 |
| 7 put the wine bottle on the rack | 34 | 9 | 17 |
| 8 put the wine bottle on top of the cabinet | 41 | 11 | 21 |
| 9 turn on the stove | 45 | 12 | 23 |
| all | 390 | 103 | 198 |

### Checks

| Check | Result |
|---|---|
| Per-task subset sizes | ceil(f · n) for every task |
| Nesting | 25 % ⊂ 50 % for every task |
| Fraction 1 | equals the full TRAIN split, record for record |
| HELDOUT | identical to the full index |
| Default serialization, stage one (23 configs: `configs/stage_one/*.json` and the run `config.json` files) | byte-identical to the code at `fcfefdc` |
| Default serialization, π (25 configs: `configs/proposal/*.json` and the π run `config.json` files, including the no-ψ `data` configs) | byte-identical to the code at `fcfefdc` |
| 30 training steps of `goal_instruction_contrast_flow`, every step logged, base `e2b0082` vs new `abf4eba` (R `verify_stage_one_launch.sh`) | all losses, logit scale, grad norm and lr equal at all 30 steps; step-30 trunk and query tensors bit-identical (max \|Δ\| 0); no `train_episodes.json` written |
| 30 training steps each of `goal_proposal_nopsi_seed0` and `goal_proposal_flow`, base `fcfefdc` vs new `abf4eba` (R `verify_launch.sh`) | mse, arm, gripper, gripper accuracy, grad norm and lr equal at all 30 steps; step-30 parameters bit-identical (max \|Δ\| 0); no `train_episodes.json` written |
| `train_episodes.json` of the 12 π against that of their stage one's subset (`goal_instruction_contrast_flow_frac{25,50}_restart`) | byte-identical in 12 of 12 |
| Training logs | 103 TRAIN episodes at 25 % and 198 at 50 %, for stage one and all 12 π |

R = `tmp/runlogs/s3p31/`.

## Stage one

The configs are `configs/stage_one/goal_instruction_contrast_flow_frac{25,50}_restart.json`, and the runs are `outputs/runs/goal_instruction_contrast_flow_frac{25,50}_restart`. The configs differ from `goal_instruction_contrast_flow.json` only in `run_dir` and `train_subset` (fraction 0.25 / 0.5, seed 0). Both runs use stage-one seed 0 and 5000 steps, starting from step 0.

Both runs finished 5000 updates, in 3817 s and 3665 s (1.31 and 1.36 steps/s, while sharing the GPU with π trainings and closed loops). Peak memory was 12.86 GiB.

Four other stage-one runs were stopped early: `goal_instruction_contrast{,_flow}_frac{25,50}`, at step 1000–2000. They are kept and not used.

### Diagnostics at step 5000

`scripts/diagnose_stage_one.py` builds the full index. So for every fraction:

- all rows below use the 43 HELDOUT episodes;
- the task-ID probe is fit on all 390 TRAIN episodes;
- the action ranking draws its other-task candidates from all 390 TRAIN episodes.

The 100 % column is `goal_instruction_contrast_flow`. The table comes from `tmp/s3p31/diag_table.py` (output R `stage_one_diagnostics_table.txt`).

| Quantity | 25 % | 50 % | 100 % |
|---|---|---|---|
| progress Spearman median, z | 0.969 | 0.987 | 0.998 |
| mean_t \|p_t − t/T\| median, z | 0.175 | 0.133 | 0.055 |
| mean_t \|p_t − t/T\| median, ψ | 0.088 | 0.091 | 0.077 |
| effective rank of cov(ψ) | 18.5 | 17.3 | 16.9 |
| SIGReg statistic | 0.0249 | 0.0267 | 0.0302 |
| goal error median | 0.0308 | 0.0224 | 0.0143 |
| paraphrase distance median | 1.69e-05 | 2.51e-05 | 3.20e-05 |
| different-task distance median | 0.232 | 0.162 | 0.162 |
| P(demo > stay), pooled | 0.885 | 0.915 | 0.941 |
| P(demo > reverse), pooled | 0.914 | 0.944 | 0.970 |
| P(demo > other-task action), pooled | 0.935 | 0.956 | 0.960 |
| instruction top-1, pooled (chance 0.1) | 0.814 | 0.824 | 0.866 |
| instruction pairwise, pooled | 0.888 | 0.913 | 0.932 |
| shared start P(demo > other-task first chunk) | 0.498 | 0.631 | 0.804 |
| task-ID probe HELDOUT, ψ(o_T) | 1.000 | 1.000 | 1.000 |
| task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 0.995 | 1.000 | 1.000 |
| task-ID probe HELDOUT, ψ(o₀) control | 0.186 | 0.047 | 0.140 |

The training terms below are the mean of the last 11 logged steps (4500–4999).

| Run | TRAIN episodes | path | goal | endpoint | SIGReg | paraphrase | instruction contrast | total |
|---|---|---|---|---|---|---|---|---|
| 25 % | 103 | 1.42e-3 | 2.41e-5 | 4.14e-5 | 0.0223 | 5.02e-5 | 0.0023 | 0.0261 |
| 50 % | 198 | 2.53e-3 | 1.01e-4 | 2.08e-4 | 0.0282 | 5.80e-5 | 0.0099 | 0.0410 |
| 100 % | 390 | 2.79e-3 | 2.09e-4 | 4.00e-4 | 0.0318 | 6.87e-5 | 0.0112 | 0.0465 |

- **On TRAIN, the terms fall as the data shrinks.** Goal, endpoint and instruction contrast at 25 % are 5–10 times lower than at 100 %.
- **On HELDOUT, the goal-directed measures degrade as the data shrinks**:
  - goal error: 0.014 → 0.031;
  - P(demo > stay): 0.941 → 0.885;
  - shared-start P(demo > other-task first chunk): 0.804 → 0.498;
  - z-progress deviation: 0.055 → 0.175.
- **The task-ID probe on ψ(o_T) stays at 1.000.**

## π training

Configs: `configs/proposal/goal_proposal_{nopsi,flow}_frac{25,50}_seed{0,1,2}.json`. Each is its seed's base config (`goal_proposal_nopsi_seed{s}`, or `goal_proposal_flow` / `goal_proposal_flow_seed{1,2}`) with three kinds of change:

- `run_dir`;
- for no ψ, `data.train_subset` = {fraction 0.25 / 0.5, seed 0};
- for flow, `stage_one_checkpoint` = its fraction's `_restart` stage one at step 5000.

The recipe is otherwise unchanged: 3000 AdamW updates at lr 3e-4, 256 transitions per batch, chunk 10. Runs: `outputs/runs/goal_proposal_{nopsi,flow}_frac{25,50}_seed{0,1,2}`, using step 3000.

- **TRAIN MSE at the last step** (standardized mean chunk):
  - no ψ: 0.0026–0.0027 at 25 % and 0.0057–0.0059 at 50 %;
  - flow: 0.0025–0.0027 at 25 % and 0.0055–0.0057 at 50 %;
  - at 100 %: 0.014–0.016 for the no-ψ π and 0.0135–0.0156 for the nine ψ π (`docs/no_psi_baseline.md`).
- **Training time and memory**:
  - no ψ: 1634–1733 s with 3 runs per GPU, next to the stage one, 4.66 GiB;
  - flow: 666–807 s with 3 runs per GPU, 3.30 GiB.

### HELDOUT MSE

This is the MSE of π's standardized mean chunk against the demonstration chunk, at step 3000, over the 4864 HELDOUT (episode, t). It is computed in float32, with ψ recomputed by the π's frozen trunk.

- Script: R `heldout_mse.py`, a copy of `tmp/runlogs/s3p29/heldout_mse.py` with the 12 new π. Output: R `heldout_mse.txt`.
- The 100 % rows come from `tmp/runlogs/s3p29/heldout_mse.txt`.

| π | fraction | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|---|
| no ψ | 25 % | 0.3053 | 0.2980 | 0.2949 | 0.2994 ± 0.0053 |
| no ψ | 50 % | 0.2754 | 0.2704 | 0.2729 | 0.2729 ± 0.0025 |
| no ψ | 100 % | 0.2535 | 0.2640 | 0.2648 | 0.2608 ± 0.0063 |
| flow (z) | 25 % | 0.3207 | 0.3204 | 0.3152 | 0.3188 ± 0.0031 |
| flow (z) | 50 % | 0.2825 | 0.2895 | 0.2864 | 0.2861 ± 0.0035 |
| flow (z) | 100 % | 0.2631 | 0.2783 | 0.2657 | 0.2690 ± 0.0081 |

- **Gripper agreement on HELDOUT**: 0.952–0.955 at 25 % and 0.958–0.962 at 50 %.
- **Correlation with closed-loop success**: over the 18 runs, Pearson r = −0.83 and Spearman ρ = −0.86.

## Closed loop

**Protocol.** The runs use the `v7_mean_flow` / `v9_mean_nopsi` command line (R `run_one.sh`, `launch_mean.sh`, `run_queue.sh`), with only `--proposal-checkpoint`, `--checkpoint` and `--state-table` changed:

- `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`, scales 0.25/0.5/1.0, dct 1, proposal seed 0 (K = 1 draws no noise);
- env seed 0, 300 steps, 10 settle steps, benchmark inits 0–49, traced, `--check-start-observation`;
- no-ψ π: `--checkpoint goal_instruction_contrast_flow` step 5000 and its ψ table, as in `v9_mean_nopsi`. The π takes nothing from them.
- flow ψ π: `--checkpoint` is its own stage one, `goal_instruction_contrast_flow_frac{25,50}_restart` step 5000, because π takes ψ from it. `--state-table` is that stage one's ψ table, `outputs/state_tables/goal_instruction_contrast_flow_frac{25,50}_restart_step5000.npy`, built with `scripts/build_state_table.py`. Recomputing 8 rows one at a time gives max |Δψ| of 1.9e-6 and 1.7e-6.

**Runs.** `outputs/closed_loop/v10_mean_{nopsi,flow}_frac{25,50}_s{0,1,2}`. All 12 runs have 500 episodes, and every task process exited 0.

**Start states.**

- **Scene after the reset** (`EpisodeTrace.initial`: object positions, joint values, end-effector position): equal to `v7_mean_flow`'s in 6000 of 6000 episodes, max |Δ| 0. Checked with `tmp/s3p31/check_initial_states.py` (output R `initial_scene_check.txt`).
- **Start-observation block of each task** (image means, token cosines, proprio): identical to `v7_mean_flow`'s in all 120 task processes. Output: R `start_observation_check.txt`.
- **ψ-distance line.**
  - For the no-ψ runs, it is computed with the same flow stage one as `v7_mean_flow`. It is identical in 59 of 60 task processes. In `v10_mean_nopsi_frac50_s2` task 8, its median reads 20.053 against 20.052.
  - For the flow runs, it comes from their own stage one and is not compared.

**Analysis.**

- R `data_scaling_analysis.py` (output `data_scaling_analysis.txt`) copies the phase-29 tests of `tmp/runlogs/s3p29/nopsi_seed_analysis.py` (unchanged copy: R `nopsi_seed_analysis_p29_copy.py`). It applies them to two versions at each fraction, then adds the change with the fraction and the version × fraction interaction.
- `summarize_closed_loop.py` compares every run against `v7_mean_flow` (R `v10_mean_*_summary.txt`).
- `classify_closed_loop_failures.py` classifies every run's failures (R `v10_mean_*_failure_classification.txt`, summary `failure_summary.txt`).

### Success per run

| π | fraction | seed 0 | seed 1 | seed 2 | mean | SD over seeds | range | binomial SD of one run |
|---|---|---|---|---|---|---|---|---|
| no ψ | 25 % | 0.862 | 0.874 | 0.872 | **0.869** | 0.006 | 0.012 | 0.015 |
| no ψ | 50 % | 0.904 | 0.902 | 0.890 | **0.899** | 0.008 | 0.014 | 0.013 |
| no ψ | 100 % | 0.940 | 0.922 | 0.918 | **0.927** | 0.012 | 0.022 | 0.012 |
| flow (z) | 25 % | 0.528 | 0.472 | 0.580 | **0.527** | 0.054 | 0.108 | 0.022 |
| flow (z) | 50 % | 0.764 | 0.834 | 0.788 | **0.795** | 0.036 | 0.070 | 0.018 |
| flow (z) | 100 % | 0.904 | 0.796 | 0.884 | **0.861** | 0.058 | 0.108 | 0.016 |

The 100 % runs are `v9_mean_nopsi_s{0,1,2}` (no ψ), and `v7_mean_flow` with `v8_mean_flow_s{1,2}` (flow).

- **Decisions per episode**:
  - no ψ: 13.8–13.9 at 25 % and 13.2–13.4 at 50 %;
  - flow: 19.3–21.5 at 25 % and 14.6–16.0 at 50 %.
- **Env steps**:
  - no ψ: 134–136 at 25 % and 128–130 at 50 %;
  - flow: 190–212 at 25 % and 142–157 at 50 %.

### No ψ against flow ψ at each fraction

Exact two-sided McNemar by (task, init), at the same π seed. Each cell gives episodes only the no-ψ π solves / only the flow π solves, and p. "Pooled" sums the discordant counts over the seeds.

| fraction | tasks | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|---|
| 25 % | all | 196 / 29, 1.2e-31 | 232 / 31, 3.1e-39 | 176 / 30, 2.5e-26 | 604 / 90, 2.4e-94 |
| 25 % | 3 and 4 | 70 / 2, 1.1e-18 | 71 / 1, 3.1e-20 | 78 / 2, 5.4e-21 | 219 / 5, 3.4e-58 |
| 25 % | other 8 | 126 / 27, 1.7e-16 | 161 / 30, 7.2e-23 | 98 / 28, 2.7e-10 | 385 / 85, 1.2e-46 |
| 50 % | all | 105 / 35, 2.6e-9 | 71 / 37, 0.0014 | 76 / 25, 3.7e-7 | 252 / 97, 5.1e-17 |
| 50 % | 3 and 4 | 21 / 2, 6.6e-5 | 10 / 1, 0.012 | 17 / 2, 7.3e-4 | 48 / 5, 7.1e-10 |
| 50 % | other 8 | 84 / 33, 2.7e-6 | 61 / 36, 0.014 | 59 / 23, 8.7e-5 | 204 / 92, 6.6e-11 |
| 100 % | all | 43 / 25, 0.038 | 97 / 34, 3.3e-8 | 50 / 33, 0.078 | 190 / 92, 5.4e-9 |
| 100 % | 3 and 4 | 3 / 2, 1 | 46 / 0, 2.8e-14 | 11 / 3, 0.057 | 60 / 5, 4.9e-13 |
| 100 % | other 8 | 40 / 23, 0.043 | 51 / 34, 0.082 | 39 / 30, 0.34 | 130 / 87, 0.0042 |

The 100 % rows reproduce `docs/no_psi_baseline.md`.

**Discordant episodes** out of 500 (mean over the three pairs):

| Pairs | 25 % | 50 % | 100 % |
|---|---|---|---|
| seeds within no ψ | 86, 101, 97 (94.7) | 73, 85, 80 (79.3) | 61, 63, 70 (64.7) |
| seeds within flow (z) | 158, 168, 188 (171.3) | 133, 132, 117 (127.3) | 126, 94, 134 (118.0) |
| no ψ vs flow (z), same seed | 225, 263, 206 (231.3) | 140, 108, 101 (116.3) | 68, 131, 83 (94.0) |

**Version against seed**, on the run-level success rates: phase-29 test (A) with two versions.

| Test, all tasks | 25 % | 50 % | 100 % |
|---|---|---|---|
| SD of the 2 version means / pooled SD of the seeds within version | 0.242 / 0.039 | 0.073 / 0.026 | 0.046 / 0.042 |
| one-way F(1, 4), p | 119.1, 0.0004 | 24.2, 0.0079 | 3.72, 0.13 |
| two-way version × seed: version F(1, 2), p | 114.3, 0.0086 | 24.7, 0.038 | 4.64, 0.16 |
| two-way version × seed: seed F(2, 2), p | 0.92, 0.52 | 1.04, 0.49 | 1.49, 0.40 |
| exact permutation (20 assignments), one-sided p, no ψ higher | 0.05 | 0.05 | 0.05 |
| bootstrap over seeds, no ψ − flow (z), 95 % interval | +0.343 [+0.292, +0.397] | +0.103 [+0.065, +0.134] | +0.065 [+0.022, +0.129] |

- **Permutation bound.** With three runs per arm, 0.05 is the smallest attainable one-sided permutation p. The no-ψ π reaches it at every fraction, over all tasks, over the other eight tasks, and in tasks 3 and 4 at 25 % and 50 % (0.10 at 100 %).
- **Other eight tasks.** The version F(1, 4) is 35.4 (p = 0.004) at 25 %, 25.8 (p = 0.007) at 50 % and 8.6 (p = 0.042) at 100 %.

### Success against the data fraction within each π

| Test, all tasks | no ψ | flow (z) |
|---|---|---|
| mean at 25 % / 50 % / 100 % | 0.869 / 0.899 / 0.927 | 0.527 / 0.795 / 0.861 |
| SD of the 3 fraction means / pooled SD of the seeds | 0.029 / 0.009 | 0.177 / 0.050 |
| one-way F(2, 6), p | 31.3, 0.0007 | 37.8, 0.0004 |
| exact permutation of F over the 280 partitions of the 9 runs, p | 0.0036 (the smallest attainable) | 0.0071 |
| bootstrap 25 % − 100 % | −0.057 [−0.071, −0.046] | −0.335 [−0.407, −0.263] |
| bootstrap 50 % − 100 % | −0.028 [−0.041, −0.016] | −0.066 [−0.125, −0.001] |
| bootstrap 25 % − 50 % | −0.029 [−0.038, −0.021] | −0.269 [−0.328, −0.209] |
| McNemar pooled over seeds, 25 % vs 100 % (only 25 % / only 100 %) | 85 / 171, 8.3e-8 | 81 / 583, 1.3e-94 |
| McNemar pooled, 50 % vs 100 % | 85 / 127, 0.0047 | 158 / 257, 1.3e-6 |
| McNemar pooled, 25 % vs 50 % | 105 / 149, 0.0069 | 123 / 526, 3.2e-60 |

The McNemar pairs join the π of the same seed at two fractions, by (task, init).

### Version × fraction

Two-way ANOVA on the 18 run-level rates, with the three seeds of a cell as replicates:

| Tasks | version F(1, 12) | fraction F(2, 12) | version × fraction F(2, 12) | no ψ − flow at 25 % / 50 % / 100 % | change of the gap, 100 % → 25 % | change of the gap, 100 % → 50 % |
|---|---|---|---|---|---|---|
| all | 101.6 | 48.8 | 26.35 (p < 0.0001) | +0.343 / +0.103 / +0.065 | +0.277 [+0.203, +0.347] | +0.038 [−0.029, +0.097] |
| 3 and 4 | 57.3 | 21.1 | 16.10 (p = 0.0004) | +0.713 / +0.143 / +0.183 | +0.530 [+0.277, +0.700] | −0.040 [−0.287, +0.140] |
| other 8 | 63.8 | 38.6 | 16.35 (p = 0.0004) | +0.250 / +0.093 / +0.036 | +0.214 [+0.145, +0.282] | +0.058 [+0.021, +0.091] |

The change intervals are bootstrap intervals: within each (version, fraction), the seeds are resampled with replacement, 20,000 draws, seed 0.

### Per task

Mean success over the three seeds, 50 episodes per run:

| Task | no ψ 25 % | no ψ 50 % | no ψ 100 % | flow 25 % | flow 50 % | flow 100 % |
|---|---|---|---|---|---|---|
| 0 open the middle drawer of the cabinet | 0.860 | 0.947 | 0.913 | 0.560 | 0.933 | 0.973 |
| 1 open the top drawer and put the bowl inside | 0.740 | 0.713 | 0.820 | 0.507 | 0.633 | 0.780 |
| 2 push the plate to the front of the stove | 0.907 | 0.987 | 0.973 | 0.480 | 0.900 | 0.920 |
| 3 put the bowl on the plate | 0.940 | 0.987 | 0.987 | 0.233 | 0.800 | 0.813 |
| 4 put the bowl on the stove | 0.947 | 0.980 | 0.980 | 0.227 | 0.880 | 0.787 |
| 5 put the bowl on top of the cabinet | 0.980 | 0.933 | 0.960 | 0.473 | 0.720 | 0.807 |
| 6 put the cream cheese in the bowl | 0.653 | 0.667 | 0.860 | 0.327 | 0.427 | 0.793 |
| 7 put the wine bottle on the rack | 0.720 | 0.867 | 0.873 | 0.653 | 0.773 | 0.847 |
| 8 put the wine bottle on top of the cabinet | 0.947 | 0.907 | 0.907 | 0.813 | 0.893 | 0.893 |
| 9 turn on the stove | 1.000 | 1.000 | 0.993 | 0.993 | 0.993 | 1.000 |

The per-seed values are in R `data_scaling_analysis.txt`.

### Tasks 3 and 4

Tasks 3 and 4 are the bowl-placing tasks, with 100 episodes per run. The other eight tasks give 400 episodes per run.

| π | fraction | tasks 3 and 4: seed 0 / 1 / 2 | mean | other 8 tasks: seed 0 / 1 / 2 | mean |
|---|---|---|---|---|---|
| no ψ | 25 % | 0.93 / 0.94 / 0.96 | 0.943 | 0.845 / 0.858 / 0.850 | 0.851 |
| no ψ | 50 % | 0.98 / 0.99 / 0.98 | 0.983 | 0.885 / 0.880 / 0.868 | 0.878 |
| no ψ | 100 % | 0.98 / 1.00 / 0.97 | 0.983 | 0.930 / 0.902 / 0.905 | 0.913 |
| flow (z) | 25 % | 0.25 / 0.24 / 0.20 | 0.230 | 0.598 / 0.530 / 0.675 | 0.601 |
| flow (z) | 50 % | 0.79 / 0.90 / 0.83 | 0.840 | 0.757 / 0.818 / 0.777 | 0.784 |
| flow (z) | 100 % | 0.97 / 0.54 / 0.89 | 0.800 | 0.887 / 0.860 / 0.882 | 0.877 |

Failure categories (`classify_closed_loop_failures.py`; no episode is stuck):

| Run | success | goal object not moved, other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint | tasks 3 and 4: other moved / nothing moved / lifted / displaced |
|---|---|---|---|---|---|---|---|
| no ψ 25 %, s0 / s1 / s2 | 431 / 437 / 436 | 1 / 12 / 4 | 14 / 15 / 11 | 25 / 21 / 33 | 29 / 15 / 14 | 0 / 0 / 2 | 0–0 / 0–1 / 2–3 / 2–4 |
| no ψ 50 %, s0 / s1 / s2 | 452 / 451 / 445 | 3 / 3 / 8 | 8 / 18 / 17 | 19 / 14 / 15 | 15 / 14 / 15 | 3 / 0 / 0 | 0–0 / 0–0 / 1–1 / 0–1 |
| flow 25 %, s0 / s1 / s2 | 264 / 236 / 290 | 80 / 115 / 65 | 28 / 30 / 28 | 55 / 46 / 61 | 66 / 68 / 51 | 7 / 5 / 5 | 28–35 / 9–11 / 13–28 / 13–19 |
| flow 50 %, s0 / s1 / s2 | 382 / 417 / 394 | 24 / 9 / 29 | 13 / 6 / 15 | 56 / 35 / 33 | 24 / 31 / 23 | 1 / 2 / 6 | 1–5 / 0–1 / 8–20 / 0–3 |

In the task 3 and 4 column, each entry is the range over the three seeds.

- **Tasks 3 and 4 at 25 %.** The flow π's most frequent failure is "goal object not moved, other object moved", in 28–35 episodes per run. At 100 %, its failures there are mostly "lifted, not at target" (2, 45 and 11 per seed; `docs/no_psi_baseline.md`).
- **The no-ψ π** places the bowl in 0.93–0.96 of the task 3 and 4 episodes at 25 %, and in 0.98–0.99 at 50 %.

### Reading

- **At every fraction, the no-ψ π is ahead of the flow ψ π.**
  - All 9 same-seed pairs favour the no-ψ π, with McNemar p ≤ 0.078 in each pair and p ≤ 0.0014 at 25 % and 50 %.
  - The seed-level tests reach the smallest attainable permutation p (0.05) at every fraction.
  - F(1, 4) has p = 0.0004 at 25 % and 0.0079 at 50 %, against 0.13 at 100 %.
- **Less data widens the gap.**
  - It is +0.065 at 100 %, +0.103 at 50 % and +0.343 at 25 % (version × fraction F(2, 12) = 26.35).
  - The flow ψ π loses 0.335 from 100 % to 25 %, and the no-ψ π loses 0.057.
- **Tasks 3 and 4 carry the largest share at 25 %.** The gap there is +0.713; over the other eight tasks it is +0.250. The gap over the other eight tasks also grows monotonically (+0.036, +0.093, +0.250).
- **Seed stability.**
  - The flow ψ π's seeds differ in 118–171 episodes on average at every fraction, and its run SD is 0.036–0.058.
  - The no-ψ π's seeds differ in 65–95 episodes, and its run SD (0.006–0.012) is at most the binomial SD of one run.
- **HELDOUT MSE** is lower for the no-ψ π at each fraction (0.299 against 0.319 at 25 %, 0.273 against 0.286 at 50 %). It rises for both as the data shrinks, and over the 18 runs it tracks closed-loop success (ρ = −0.86).
- **The flow stage one itself generalizes less at low data.**
  - Its HELDOUT goal error doubles from 100 % to 25 %.
  - P(demo > stay) drops from 0.941 to 0.885.
  - Its shared-start instruction ranking drops from 0.804 to 0.498.
  - Meanwhile its TRAIN losses fall.
  - The no-ψ π has no trained input module; its inputs are the frozen DINOv2 and SigLIP features.
  - That the ψ π's loss at low data comes from this stage-one degradation is a hypothesis. The comparison here changes the stage one and π together.
- **Scope.**
  - At each fraction, the two π share the subset, the π recipe, the batch order of each seed, the evaluation protocol and the initial scenes.
  - They differ in π's inputs (ψ and q_l against the instruction tokens), and hence in whether a stage one is trained on the subset.

## Placement and GPUs

No process of another user was on the target GPU at any launch (nvidia-smi before each launch; `run_queue.sh` also checks the owner of every process on the GPU before each closed-loop run). All times are 2026-09-28.

| Job | Host, CUDA | Time | Code |
|---|---|---|---|
| stage one 25 % / 50 % | node-65, 0 / 1 | 07:40–08:44 / 07:40–08:41 | `fcfefdc` (stage-one code as in `abf4eba`) |
| no-ψ π, 25 % seeds 0–2 / 50 % seeds 0–2 | node-65, 0 / 1 | 07:44–08:12 | `abf4eba` |
| π bit-identity trainings | node-65, 0 | 07:46–07:50 | `fcfefdc` / `abf4eba` |
| `v10_mean_nopsi_frac25_s{0,1,2}` / `v10_mean_nopsi_frac50_s{0,1,2}`, one run at a time per GPU | node-65, 0 / 1 | 08:13–08:33 | `abf4eba` |
| stage-one diagnostics and ψ tables, 25 % / 50 % | node-61, 0 / 1 | 11:33–11:36 | `abf4eba` |
| flow π, 25 % seeds 0–2 / 50 % seeds 0–2 | node-61, 0 / 1 | 11:33–11:47 / 11:33–11:45 | `abf4eba` |
| `v10_mean_flow_frac50_s{0,1,2}` | node-61, 1 | 11:46–12:03 | `abf4eba` |
| `v10_mean_flow_frac25_s{0,1,2}` | node-61, 0 | 11:47–12:07 | `abf4eba` |
| stage-one bit-identity trainings | node-61, 0 | 11:47–11:49 | `e2b0082` / `abf4eba` |
| HELDOUT MSE | node-61, 1 | 11:48–11:49 | `abf4eba` |
| analysis (CPU) | node-61 | 12:07–12:09 | `abf4eba` |

- **Node switch.** After 08:44, node-65 CUDA 0 and 1 went to another user's job. The remaining jobs ran on node-61 CUDA 0 and 1.
- **Closed-loop duration.** A closed-loop run (10 task processes) took 5.2–6.6 min.
- **`/tmp/robosuite.log`.**
  - On node-61: last modified 2026-09-27 22:36:09, 24,576 bytes, both before and after these jobs.
  - On node-65: last modified at 11:45, after these jobs had ended (08:44).

## Paths

| What | Where |
|---|---|
| subset code | `ceam/data/train_subset.py`; `ceam/training/{stage_one_config,stage_one_trainer,proposal_config,proposal_trainer}.py` |
| stage-one configs | `configs/stage_one/goal_instruction_contrast_flow_frac{25,50}_restart.json` |
| stage-one runs | `outputs/runs/goal_instruction_contrast_flow_frac{25,50}_restart` (with `train_episodes.json` and `diagnostics/step_005000/`) |
| ψ tables | `outputs/state_tables/goal_instruction_contrast_flow_frac{25,50}_restart_step5000.npy` |
| π configs | `configs/proposal/goal_proposal_{nopsi,flow}_frac{25,50}_seed{0,1,2}.json` |
| π runs | `outputs/runs/goal_proposal_{nopsi,flow}_frac{25,50}_seed{0,1,2}` (with `train_episodes.json`; step 3000 used) |
| closed loops | `outputs/closed_loop/v10_mean_{nopsi,flow}_frac{25,50}_s{0,1,2}/task<t>_mean/{results,trace}.jsonl` |
| stopped stage-one runs, not used | `outputs/runs/goal_instruction_contrast{,_flow}_frac{25,50}`; configs `configs/stage_one/goal_instruction_contrast{,_flow}_frac{25,50}{,_restart}.json` |
| launchers, logs, analysis | `tmp/runlogs/s3p31/` (`env.sh`, `launch_stage_one.sh`, `launch_post_stage_one.sh`, `launch_train_pi.sh`, `run_one.sh`, `launch_mean.sh`, `run_queue.sh`, `heldout_run.sh`, `analyze.sh`, `verify_*`, `logs/`, `data_scaling_analysis.{py,txt}`, `nopsi_seed_analysis_p29_copy.py`, `heldout_mse.{py,txt}`, `heldout_mse_p29_copy.py`, `stage_one_diagnostics_table.txt`, `start_observation_check.txt`, `initial_scene_check.txt`, `failure_summary.txt`, `v10_mean_*_{summary,failure_classification}.txt`); `tmp/s3p31/` (`diag_table.py`, `check_initial_states.py`, config checks) |

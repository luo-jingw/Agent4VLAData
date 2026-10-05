# Failure displacements as negatives in the instruction contrast (phases 15 and 16)

**Status (user decision, 2026-09-28):** the failure-negative stage one does not pass the early-stage acceptance, and the user accepted that result: OPT-008 ends here and phase 16 is not run. The implementation (`failure_negatives`, off by default) and `scripts/rescore_branch_candidates.py` are kept.

The instruction contrast of stage one (`docs/stage_one_instruction_contrast.md`) asks which instruction a demonstration effect belongs to: the anchor is the effect z = ψ(o_{t+k}) − ψ(o_t), the positive its own instruction's direction q_l, the negatives the directions of the tasks that are its negatives in the task-relation table. It never sees a displacement that started under an instruction and did not complete it. This page adds the other direction of the same contrast (instruction → effect), in which the displacements of failed closed-loop episodes are negatives of their task's q_l, trains a stage one with it and checks it against the fixed stage one. Failure displacements enter only as contrastive negatives; no success probability, value or reward is regressed, and the branch-test outcomes of phase 26 are used only for evaluation.

## Implementation

### Data: `FailureEffectSampler`

`ceam/data/failure_effect_sampler.py`, with the batch `FailureEffectBatch` (`ceam/data/failure_effect_batch.py`).

- **Source**: the transition store `outputs/transitions/rollouts_r0` (data B, `docs/dynamics_data.md`): 100 sampled starts per task, closed loop with frozen retrieval, observed effects and rule 32/128/512, env seed 1. Every decision of an episode is one transition (kind `rollout` or `rollout_stay`), decisions t = 1 … 30 of a failed episode with chained frames (the frame after decision t is the frame before decision t + 1).
- **Failure**: LIBERO's success of the episode (`episode_success`), constant within every episode (checked). An episode is used when it failed and its sampled start is below `first_heldout_episode` = 80; starts 80–99 are held out for evaluation.
- **Displacement**: z⁻ = ψ(o_end) − ψ(o_t), with o_t the frame before a decision t drawn uniformly and o_end the last frame recorded in the episode (the frame after decision 30). The frames are the stored frozen DINOv2 tokens and proprio (standardized with the feature cache's statistics); ψ is the trunk being trained, with gradient.
- **Counts** (failed episodes with start 0–79 per task 0–9): 69, 76, 52, 26, 47, 43, 78, 80, 76, 2 (549 episodes, 16,470 decisions). Every task has at least one; task 9 (turn on the stove) has two.
- **Per batch**: for every task, `per_task` (episode, decision) pairs drawn uniformly with replacement from all decisions of the task's used episodes. `per_task` = 51 = ⌊B·M / n_tasks⌋ = ⌊64 · 8 / 10⌋: as many failure displacements per task as the batch has uniform frames per task on average. A batch therefore adds 510 displacements (1020 frames) to the 1216 frames of the stage-one batch.
- **Randomness**: the sampler owns `numpy.random.default_rng([seed, 1])`, independent of the stage-one sampler's `default_rng(seed)`, so the stage-one batches are those of the same config without failure negatives.

### Loss: the reverse direction

`InstructionContrastLoss.reverse` (`ceam/losses/instruction_contrast_loss.py`), with the states of the failure displacements in `FailureEffectStates` (`ceam/losses/failure_effect_states.py`).

- **Anchor**: q_l of every task l, the direction of the one instruction variant per task that the batch draws for the existing direction's negatives (`text_all`).
- **Positive**: each uniform-frame effect z⁺ of the batch (the existing effect of the forward direction, k uniform to the episode end), with l its anchor's task.
- **Negatives of task l**: the failure displacements of task l, and the uniform-frame effects of every anchor whose task is a negative of l in the task-relation table.
- **Form**, per positive: logsumexp(s·cos(z⁺, q_l) ∪ {s·cos(z⁻_j, q_l)}) − s·cos(z⁺, q_l) over those negatives; the same s = exp(min(logit_scale, log 100)) and cosine convention as the forward direction; averaged over the positives whose task has at least one negative. No stop-gradient; float32.
- **Weight**: `StageOneObjective` adds the reverse term with the instruction-contrast weight (1): total = … + 1 · forward + 1 · reverse. The forward direction is unchanged.
- **Trainer** (`ceam/training/stage_one_trainer.py`): with failure negatives, a second trunk call on the [2, N] frames (o_t, o_end) inside the same bf16 autocast. SIGReg and the other terms see the same states as before; the failure frames enter only the reverse term. `metrics.jsonl` gains `instruction_contrast_reverse`.
- **Config** (`ceam/training/stage_one_config.py`): the optional section `failure_negatives` with `transitions_dir`, `per_task` and `first_heldout_episode`; missing when read means none, and it is written only when present. It requires text-only queries (one direction per task). `configs/stage_one/goal_instruction_contrast_failneg.json` is the fixed config (`goal_instruction_contrast.json`) with this section and its own run directory.
- `scripts/check_losses.py` passes no failure states to the objective and compares report fields that are present.

## Verification

Scripts and outputs in R = `tmp/s3p15/verify/`; all ran on node-69.

- **Reverse term against a direct implementation** (R `check_reverse_loss.py`, output `check_reverse_loss.txt`): a float64 loop over positives and negatives, on four random cases with 3–5 tasks, some task pairs not negative, failures for all, some or no tasks. Largest difference 7.2e-7 in value and 7.8e-7 relative in every gradient (effects, directions, failure states, logit scale). Closed forms: every negative orthogonal to the positive gives log(1 + 3e^{−s}) exactly; a failure displacement equal to the positive gives log(2 + 2e^{−s}); no negative gives 0.
- **Default off is bit-identical** (R `run_short.py`, `compare_runs.py`, output `compare_runs.txt`): 30-step runs (log every step, one checkpoint) of the base code (`git archive fcfefdc`, the branch point) and this code on the same config:
  - `goal_instruction_contrast.json`: two base runs on CUDA 0 and 1 and the new run; every logged loss, the logit scale and the gradient norm equal at all 30 steps, and all 112 parameter tensors of the checkpoint bitwise equal;
  - `goal_instruction_contrast_frac25.json` (phase 31's TRAIN subset): the same, base against new.
- **Config round trip** (R `dump_config_texts.py`, output `roundtrip.txt`): every config of `configs/stage_one/` and the configs stored in two checkpoints parse and write to the same text under the base and the new code; the failure-negative config round-trips.
- **Existing checks**: `scripts/check_losses.py --device cpu` prints the same output under both codes (R `check_losses_{base,new}.txt`).
- **Same batches**: step 0 of the failure-negative run logs the fixed run's step-0 SIGReg, path, goal, endpoint, paraphrase and forward contrast exactly (the stage-one batch and initialization are the same).

## Training

- **Run**: `configs/stage_one/goal_instruction_contrast_failneg.json`, from scratch, 5000 steps, seed 0; code `9260642`; node-69 CUDA 0 (L40S), 2026-09-28 07:52–08:43. Output `outputs/runs/goal_instruction_contrast_failneg` (checkpoints every 1000 steps), log `tmp/runlogs/s3p15/train_stage_one_failneg.log`.
- **Cost**: 3019.5 s, 1.66 steps/s, peak 22.50 GiB GPU memory; the fixed run logs 2.09 steps/s and 12.86 GiB.
- **Curves** (`outputs/analysis/stage_one_failneg/training_curves.md`; R2 = `tmp/s3p15/analysis/`, script R2 `training_curves.py`), means of the logged lines in each step window, fixed / failure negatives:

| steps | forward contrast | reverse contrast | logit scale s | SIGReg | total | gradient norm |
|---|---|---|---|---|---|---|
| 0 | 2.421 / 2.421 | – / 6.651 | 14.29 / 14.29 | 0.503 / 0.503 | 8.02 / 14.67 | 411 / 411 |
| 50–200 | 0.644 / 0.630 | – / 1.804 | 14.34 / 14.40 | 0.076 / 0.146 | 0.740 / 2.622 | 3.5 / 14.0 |
| 250–950 | 0.194 / 0.227 | – / 0.484 | 14.46 / 14.89 | 0.049 / 0.068 | 0.249 / 0.788 | 1.3 / 5.0 |
| 1000–1950 | 0.106 / 0.123 | – / 0.256 | 15.48 / 15.66 | 0.039 / 0.058 | 0.148 / 0.443 | 0.84 / 2.74 |
| 2000–2950 | 0.077 / 0.065 | – / 0.124 | 16.85 / 16.50 | 0.036 / 0.054 | 0.117 / 0.248 | 0.71 / 2.69 |
| 3000–3950 | 0.025 / 0.022 | – / 0.030 | 17.84 / 17.14 | 0.033 / 0.050 | 0.059 / 0.105 | 0.55 / 2.06 |
| 4000–4999 | 0.011 / 0.011 | – / 0.013 | 18.26 / 17.50 | 0.031 / 0.045 | 0.043 / 0.070 | 0.38 / 0.87 |

- The reverse term falls from 6.65 to 0.013 and ends at 0.029 (step 4999). The forward term stays close to the fixed run's in every window (largest difference 0.033, steps 250–950). The logit scale ends at 17.5 against 18.3; SIGReg stays about 1.5 times higher and the gradient norm 2–4 times higher.

## Acceptance (phase 15)

Checks on the 43 HELDOUT episodes, both step-5000 checkpoints under the same code:

- `scripts/diagnose_stage_one.py`: `outputs/runs/goal_instruction_contrast_failneg/diagnostics/step_005000/` and `outputs/analysis/stage_one_failneg/diagnostics_fixed_step005000/` (logs `tmp/runlogs/s3p15/diagnose_{failneg,fixed}_step5000.log`).
- Paired intervals: R2 `early_stage_paired.py` recomputes the per-anchor indicators of `ActionRankingMetrics` with its own helpers (anchors, encoding, directions, frozen-feature neighbours); every aggregate equals `report.json` (largest difference 0 for both checkpoints). Intervals are 95% percentiles over 2000 bootstrap resamples of the HELDOUT episodes, one draw matrix for both checkpoints. Output `outputs/analysis/stage_one_failneg/early_stage_paired.json`; tables `acceptance_tables.md` (R2 `format_acceptance.py`).
- **Rule**: an early segment (t/T 0–0.25, 0.25–0.5, or the first half pooled) is clearly worse when, for an action-ranking quantity (demo against stay, reverse, other-task action), instruction top-1 or shared-start discrimination, the paired difference failure negatives − fixed is at most −0.03 with its 95% interval entirely below 0.

### Sanity checks by t/T quarter

| quantity | t/T | fixed | failneg | failneg − fixed [95% CI] |
|---|---|---|---|---|
| P(m(demo) > m(stay)) | 0–0.25 | 0.922 | 0.948 | +0.026 [-0.034, +0.099] |
| P(m(demo) > m(stay)) | 0.25–0.5 | 0.945 | 0.859 | -0.086 [-0.153, -0.030] |
| P(m(demo) > m(stay)) | 0.5–0.75 | 0.962 | 0.969 | +0.008 [+0.000, +0.024] |
| P(m(demo) > m(stay)) | 0.75–1 | 0.958 | 0.958 | +0.000 [+0.000, +0.000] |
| P(m(demo) > m(stay)) | first half | 0.932 | 0.907 | -0.025 [-0.069, +0.014] |
| P(m(demo) > m(reverse)), t ≥ k | 0–0.25 | 0.945 | 0.955 | +0.009 [-0.064, +0.088] |
| P(m(demo) > m(reverse)), t ≥ k | 0.25–0.5 | 0.984 | 0.891 | -0.094 [-0.169, -0.034] |
| P(m(demo) > m(reverse)), t ≥ k | 0.5–0.75 | 0.977 | 0.985 | +0.008 [-0.015, +0.033] |
| P(m(demo) > m(reverse)), t ≥ k | 0.75–1 | 0.968 | 0.947 | -0.021 [-0.052, +0.000] |
| P(m(demo) > m(reverse)), t ≥ k | first half | 0.966 | 0.920 | -0.046 [-0.098, -0.004] |
| P(m(demo) > m(other-task NN action)), t > 0 | 0–0.25 | 0.960 | 0.945 | -0.014 [-0.066, +0.037] |
| P(m(demo) > m(other-task NN action)), t > 0 | 0.25–0.5 | 0.974 | 0.918 | -0.056 [-0.110, -0.013] |
| P(m(demo) > m(other-task NN action)), t > 0 | 0.5–0.75 | 0.963 | 0.975 | +0.012 [-0.004, +0.034] |
| P(m(demo) > m(other-task NN action)), t > 0 | 0.75–1 | 0.959 | 0.957 | -0.002 [-0.007, +0.000] |
| P(m(demo) > m(other-task NN action)), t > 0 | first half | 0.967 | 0.930 | -0.037 [-0.071, -0.009] |
| instruction top-1 | 0–0.25 | 0.758 | 0.784 | +0.026 [-0.034, +0.083] |
| instruction top-1 | 0.25–0.5 | 0.844 | 0.852 | +0.008 [-0.025, +0.042] |
| instruction top-1 | 0.5–0.75 | 0.962 | 0.969 | +0.008 [+0.000, +0.024] |
| instruction top-1 | 0.75–1 | 0.958 | 0.958 | +0.000 [+0.000, +0.000] |
| instruction top-1 | first half | 0.797 | 0.815 | +0.018 [-0.023, +0.057] |
| instruction pairwise | 0–0.25 | 0.916 | 0.946 | +0.030 [-0.014, +0.085] |
| instruction pairwise | 0.25–0.5 | 0.954 | 0.914 | -0.040 [-0.088, -0.003] |
| instruction pairwise | 0.5–0.75 | 0.962 | 0.976 | +0.015 [+0.000, +0.037] |
| instruction pairwise | 0.75–1 | 0.958 | 0.958 | +0.000 [+0.000, +0.000] |
| instruction pairwise | first half | 0.933 | 0.931 | -0.002 [-0.028, +0.025] |
| shared start, t = 0 | 0–0.25 | 0.866 | 0.949 | +0.083 [+0.027, +0.147] |
| effective rank of demo effects | quarters 1–4, pooled | 9.6 / 7.9 / 10.2 / 9.0 / 15.1 | 8.9 / 6.4 / 7.0 / 9.2 / 11.7 | |

Other diagnostics, fixed → failure negatives:

| Quantity | fixed | failure negatives |
|---|---|---|
| effective rank of cov(ψ), 5294 HELDOUT frames | 16.58 | 12.02 |
| SIGReg statistic, 4096 frames | 0.025 | 0.042 |
| action sensitivity, median std of candidate matches (pooled) | 0.186 | 0.259 |
| task subspace overlap ‖U_iᵀU_j‖²_F / r, off-diagonal mean | 0.749 | 0.727 |
| linear probe of the task from ψ(o_T), TRAIN / HELDOUT | 1.000 / 1.000 | 1.000 / 1.000 |
| linear probe from ψ(o₀) (control), TRAIN / HELDOUT | 0.815 / 0.070 | 0.590 / 0.163 |
| goal error ‖Uᵀĝ − z_T‖ / ‖z_T − z₀‖, median | 0.0082 | 0.0149 |
| path deviation, median | 0.056 | 0.045 |
| progress p at t/T = 0.25, median | 0.289 | 0.272 |

### By task

The second-quarter drop comes from three tasks, all bowl pick-and-place (R2 `per_task_bins.py`, log `tmp/runlogs/s3p15/per_task_bins_step5000.log`); fixed → failure negatives (anchors or pairs):

| Task | demo > stay | demo > reverse | demo > other-task action |
|---|---|---|---|
| 3 put the bowl on the plate | 0.82 → 0.45 (11) | 1.00 → 0.64 (11) | 0.91 → 0.57 (99) |
| 4 put the bowl on the stove | 1.00 → 0.62 (13) | 1.00 → 0.77 (13) | 0.97 → 0.81 (117) |
| 5 put the bowl on top of the cabinet | 0.69 → 0.46 (13) | 0.85 → 0.46 (13) | 0.89 → 0.74 (117) |

In the second quarter every other task is at 1.00 in both, or higher with failure negatives. In these three tasks a second-quarter demonstration chunk has a match ≤ 0 under its own q_l at 38–55% of the anchors, against 0–31% for the fixed stage one. In the first quarter, task 4 gains (demo > stay 0.56 → 0.94) and task 5 loses (0.87 → 0.60).

### By training step

Paired differences failure negatives − fixed at the checkpoints of both runs (R2 `early_stage_paired.py`, outputs `early_stage_paired_step{1000,2000,3000,4000}.json`):

| step | quantity | t/T 0–0.25 | t/T 0.25–0.5 | first half |
|---|---|---|---|---|
| 1000 | P(m(demo) > m(stay)) | +0.098 [+0.053, +0.144] | +0.016 [-0.014, +0.047] | +0.060 [+0.033, +0.089] |
| 1000 | P(m(demo) > m(reverse)), t ≥ k | +0.027 [+0.000, +0.057] | +0.008 [+0.000, +0.025] | +0.017 [+0.004, +0.033] |
| 1000 | P(m(demo) > m(other-task NN action)), t > 0 | +0.002 [-0.015, +0.016] | +0.003 [-0.014, +0.025] | +0.003 [-0.008, +0.015] |
| 1000 | instruction top-1 | -0.033 [-0.091, +0.021] | -0.008 [-0.047, +0.032] | -0.021 [-0.056, +0.011] |
| 1000 | shared start, t = 0 | +0.081 [+0.031, +0.139] |  |  |
| 2000 | P(m(demo) > m(stay)) | -0.013 [-0.040, +0.000] | -0.023 [-0.055, +0.000] | -0.018 [-0.036, -0.004] |
| 2000 | P(m(demo) > m(reverse)), t ≥ k | -0.009 [-0.028, +0.000] | -0.008 [-0.025, +0.000] | -0.008 [-0.021, +0.000] |
| 2000 | P(m(demo) > m(other-task NN action)), t > 0 | +0.003 [-0.013, +0.018] | -0.003 [-0.018, +0.008] | +0.000 [-0.011, +0.009] |
| 2000 | instruction top-1 | -0.092 [-0.169, -0.020] | +0.000 [-0.042, +0.042] | -0.050 [-0.102, -0.003] |
| 2000 | shared start, t = 0 | -0.011 [-0.033, +0.011] |  |  |
| 3000 | P(m(demo) > m(stay)) | -0.013 [-0.039, +0.013] | -0.086 [-0.157, -0.030] | -0.046 [-0.081, -0.017] |
| 3000 | P(m(demo) > m(reverse)), t ≥ k | -0.027 [-0.072, +0.000] | -0.078 [-0.143, -0.029] | -0.055 [-0.101, -0.020] |
| 3000 | P(m(demo) > m(other-task NN action)), t > 0 | -0.007 [-0.030, +0.012] | -0.019 [-0.040, -0.003] | -0.014 [-0.030, +0.000] |
| 3000 | instruction top-1 | +0.046 [-0.012, +0.103] | +0.008 [-0.031, +0.049] | +0.028 [-0.004, +0.061] |
| 3000 | shared start, t = 0 | +0.051 [+0.010, +0.098] |  |  |
| 4000 | P(m(demo) > m(stay)) | +0.026 [-0.026, +0.090] | -0.117 [-0.198, -0.050] | -0.039 [-0.083, +0.000] |
| 4000 | P(m(demo) > m(reverse)), t ≥ k | -0.009 [-0.075, +0.056] | -0.117 [-0.205, -0.046] | -0.067 [-0.127, -0.016] |
| 4000 | P(m(demo) > m(other-task NN action)), t > 0 | -0.020 [-0.067, +0.023] | -0.057 [-0.104, -0.019] | -0.040 [-0.072, -0.015] |
| 4000 | instruction top-1 | +0.033 [-0.026, +0.089] | -0.023 [-0.073, +0.017] | +0.007 [-0.035, +0.048] |
| 4000 | shared start, t = 0 | +0.091 [+0.037, +0.153] |  |  |
| 5000 | P(m(demo) > m(stay)) | +0.026 [-0.034, +0.099] | -0.086 [-0.153, -0.030] | -0.025 [-0.069, +0.014] |
| 5000 | P(m(demo) > m(reverse)), t ≥ k | +0.009 [-0.064, +0.088] | -0.094 [-0.169, -0.034] | -0.046 [-0.098, -0.004] |
| 5000 | P(m(demo) > m(other-task NN action)), t > 0 | -0.014 [-0.066, +0.037] | -0.056 [-0.110, -0.013] | -0.037 [-0.071, -0.009] |
| 5000 | instruction top-1 | +0.026 [-0.034, +0.083] | +0.008 [-0.025, +0.042] | +0.018 [-0.023, +0.057] |
| 5000 | shared start, t = 0 | +0.083 [+0.027, +0.147] |  |  |

### Verdict: not passed

- **Action ranking, second quarter**: all three quantities fall by 0.056–0.094 with intervals below 0; demo against reverse and against other-task actions also fall in the first half pooled (−0.046, −0.037). The fall appears between steps 2000 and 3000 and persists to step 5000.
- **Instruction top-1**: unchanged in both early quarters (+0.026, +0.008; intervals include 0).
- **Shared-start discrimination**: higher, 0.866 → 0.949 (+0.083 [+0.027, +0.147]).
- **Effective rank**: lower, of ψ (16.6 → 12.0) and of the demonstration effects in every quarter but the last.

By the rule above the early segments are clearly worse, so phase 16 does not run on this stage one.

## Downstream and evaluation (phase 16)

Not run: the stage one of phase 15 did not pass its acceptance. The configs and the tool for the main readouts are in place and checked.

- **Configs**: `configs/stage_two/goal_dynamics_perturb_failneg.json` (Δ, demo + data A, the `goal_dynamics_perturb` recipe) and `configs/proposal/goal_proposal_failneg_seed{0,1,2}.json` (π, the `goal_proposal` recipe), each on `goal_instruction_contrast_failneg` step 5000.

### Re-scoring the branch test's every-candidate states

`scripts/rescore_branch_candidates.py` scores the phase-26 every-candidate states (`docs/branch_success_test.md`) with any stage one and its Δ, without executing any candidate. Outputs `outputs/analysis/branch_rescore/<name>/`.

- `observe`: per sweep state, `LiberoTaskEnv.restore` of the recorded decision (the branch test's restore: the snapshot and the recorded producing step), then the frozen DINOv2 tokens and raw proprio of the observation, the stored candidates, every candidate's outcome, the recorded q_l match and the restore checks. The record has no frozen features of its observations, so this is the only simulator step.
- `score`: per model, ψ′, Δ′ of every candidate and cos(Δ̂′, q′_l), one state at a time as the closed loop computes them.
- `report`: the every-candidate readouts of the branch test (best − all, best − same scale, within-state AUC over all candidates) plus a within-state AUC restricted to pairs of equal offset scale, and paired differences against a reference scoring with the branch test's episode bootstrap (seed 0, 2000 resamples).

Checks (node-69 CUDA 1, code `77c2072`; logs `tmp/runlogs/s3p15/observe_*.log`, `score_fixed.log`, `rescore_report_*.log`):

| Set | states | restore: proprio / agent / wrist identical | max \|ψ − recorded ψ\| | own stage one and Δ: max \|match − recorded\|, best equal |
|---|---|---|---|---|
| fixed | 516 | 508 / 498 / 461 | 1.8e-3 | 1.9e-5, 516 of 516 |
| flow | 287 | 283 / 281 / 264 | 1.9e-3 | 3.3e-5, 287 of 287 |
| flow_full_sgd | 281 | 281 / 276 / 259 | 4.2e-3 | 7.5e-5, 281 of 281 |

The re-scored matches reproduce the branch test's table exactly, from the recorded scores and from the re-observed states:

| Set | scoring | best − all | best − same scale | within-state AUC | same-scale AUC |
|---|---|---|---|---|---|
| fixed | fixed ψ q_l, recorded and re-scored | +0.049 [+0.030, +0.070] | +0.040 [+0.020, +0.061] | 0.542 [0.513, 0.572] | 0.542 [0.509, 0.576] |
| flow | flow ψ q_l, recorded and re-scored | +0.018 [−0.014, +0.050] | −0.008 [−0.042, +0.027] | 0.492 [0.466, 0.521] | 0.497 [0.472, 0.523] |
| flow | fixed ψ, its Δ and q_l | +0.018 [−0.013, +0.049] | −0.006 [−0.041, +0.030] | 0.503 [0.476, 0.534] | 0.513 [0.482, 0.547] |
| flow_full_sgd | flow_full_sgd ψ q_l, recorded and re-scored | +0.003 [−0.027, +0.033] | −0.014 [−0.047, +0.019] | 0.474 [0.442, 0.506] | 0.485 [0.452, 0.520] |
| flow_full_sgd | fixed ψ, its Δ and q_l | +0.046 [+0.008, +0.084] | +0.034 [−0.006, +0.075] | 0.512 [0.482, 0.543] | 0.512 [0.480, 0.544] |

States with both outcomes: 184 (fixed), 122 (flow), 116 (flow_full_sgd); with a same-scale pair of both outcomes: 183, 119, 111.

On the flow_full_sgd states, the fixed ψ's best candidate succeeds more often than that ψ's own (paired +0.043 [−0.004, +0.086], 24 / 12 discordant states, McNemar p 0.065; within-state AUC +0.039 [+0.005, +0.071]); on the flow states the two do not differ (9 / 9).

Command, per name after a Δ exists for the new stage one:

```bash
source scripts/ceam_env.sh    # CEAM_ROOT = main checkout, CUDA_VISIBLE_DEVICES unset
$CEAM_PYTHON scripts/rescore_branch_candidates.py observe --name <branch-test name> --cuda-device <gpu> --code-commit <sha>
$CEAM_PYTHON scripts/rescore_branch_candidates.py score --name <name> --model <label>=<stage one ckpt>,<Δ ckpt> \
  --device cuda:<gpu> --code-commit <sha>
$CEAM_PYTHON scripts/rescore_branch_candidates.py report --name <name> --models fixed <label> --reference fixed
```

## Paths, runs and GPUs

| What | Path |
|---|---|
| stage one | `outputs/runs/goal_instruction_contrast_failneg/` (config, metrics, checkpoints, diagnostics) |
| acceptance | `outputs/analysis/stage_one_failneg/` (training curves, fixed diagnostics, paired early-stage JSON at steps 1000–5000, tables) |
| branch-test re-scoring | `outputs/analysis/branch_rescore/{fixed,flow,flow_full_sgd}/` (observations, scores of fixed, flow and flow_full_sgd, report) |
| verification | `tmp/s3p15/verify/` |
| analysis scripts | `tmp/s3p15/analysis/` |
| launchers and logs | `tmp/s3p15/*.sh`, `tmp/s3p15/cl/`, `tmp/runlogs/s3p15/` |

- **GPUs**: node-69 (L40S) CUDA 0 and 1 only; no process of another user was on either at any launch. `/tmp/robosuite.log` on node-69 existed before (2026-09-17, 281,249 bytes) and is unchanged (same md5) after the simulator runs.
- **Code**: `9260642` (stage one, verification, acceptance), `77c2072` (re-scoring), `841e988` (configs).

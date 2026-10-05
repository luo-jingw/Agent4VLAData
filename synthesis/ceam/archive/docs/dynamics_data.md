# Dynamics-head training data from the simulator

Δ(o, a) is trained to regress ψ(o′) − ψ(o) (`docs/training.md`). This document records the simulated transitions added to the demonstration transitions and their effect on Δ.

## Data A: executed perturbations and stay from TRAIN demonstration states

Collection: `scripts/collect_demo_perturbations.py`; store `outputs/transitions/demo_perturbations/` (32 shards, 6.9 GB).

- **Code and host**: commit `4db445e`; node-70, 8 L40S × 4 processes. Launchers: `tmp/v3/launch_collect_demo_perturbations.sh` (shard 0) and `tmp/v3/launch_collect_rest.sh` (shards 1–31); logs `tmp/v3/collect_a/`.
- **Time points**: every TRAIN episode (390), 16 distinct time points t ≤ T − 11, drawn with a Generator seeded [0, episode_id].
- **Per time point**:
  1. restore frame t (`DemoStateRestorer`);
  2. encode o_t;
  3. run five candidates, each restoring t again before `env.rollout(chunk)` and encoding o′.
- **Candidates**: stay (10 zero-delta steps with the gripper command in force at t), and 4 perturbed chunks. A perturbed chunk is the demo chunk a_t … a_{t+9} plus a constant offset σ·ε on the 6 arm dims in the standardized action space (ε ~ N(0, I), σ uniform on {0.5, 1, 2} per chunk, stream [0, episode_id, t, 1]), with the gripper column unchanged, clipped to the controller range [−1, 1].
- **Totals**: 6240 time points, 37,440 frames, 31,200 transitions (6240 stay, 24,960 perturbed).
- **Wall time**: 283–371 s per shard, about 1.1 s per time point.
- **Restore fidelity per shard**: proprio max ≤ 1.7e-5; token cosine mean agent 0.970, wrist 0.939; repeated restores identical in 201–202 of 208 time points.
- **`/tmp/robosuite.log` on node-70**: unchanged (1885 bytes, md5 `27d2a5d0…`).

On a 22-episode sample, the median true ‖z‖ was:

| kind | median ‖z‖ |
|---|---|
| stay | 1.01 |
| σ 0.5 | 4.34 |
| σ 1 | 4.74 |
| σ 2 | 6.07 |

These are lower than the phase-12 HELDOUT medians because t is uniform over the episode rather than at 0.1 / 0.4 / 0.7 of it.

Zero-shot error of the baseline Δ (`goal_dynamics_short/step_003000`, demonstrations only) on data A:

| Group | n | median ‖z‖ | median ‖Δ̂‖ | median relative error | cosine mean / median |
|---|---|---|---|---|---|
| demo chunk (feature cache), same (episode, t) | 6240 | 4.45 | 4.42 | 0.112 | 0.985 / 0.995 |
| stay | 6240 | 1.14 | 1.96 | 1.315 | 0.618 / 0.788 |
| σ 0.5 | 8333 | 4.49 | 4.43 | 0.231 | 0.922 / 0.985 |
| σ 1 | 8232 | 5.05 | 4.65 | 0.393 | 0.802 / 0.960 |
| σ 2 | 8395 | 6.61 | 4.82 | 0.669 | 0.623 / 0.858 |
| all of data A | 31,200 | 4.18 | 4.01 | 0.487 | 0.749 / 0.941 |

The same pattern holds on the HELDOUT counterfactual check: error grows with σ, ‖Δ̂‖ is too small for large offsets, and stay is overpredicted.

## Data B: closed-loop rollouts from sampled starts (round 0)

Configuration: frozen retrieval, observed effects, rule 32/128/512, patience 2 — the best closed-loop configuration (`docs/closed_loop_mvp.md`).

- **Starts**: 100 sampled starts per task (`--start sampled --inits 0-99 --seed 1`), 300 env steps, traced.
- **Code and host**: commit `9a63623`; node-61, 10 processes, about 10 min wall.
- **Outputs**:
  - results and traces: `outputs/closed_loop/rollouts_r0_frozen_observed_rule/`;
  - transitions: `outputs/transitions/rollouts_r0/` (one shard per task, 4.77 GB);
  - launcher and reports: `tmp/runlogs/s3p5/rollouts_r0/`.
- **Recording rule**:
  - Every decision's encoded frame and executed chunk is recorded, and the next observation is the future frame, so consecutive transitions share frames.
  - Only complete 10-step chunks are kept. A chunk cut short by success or by the step limit is dropped (296 dropped).
  - `kind` is "rollout", or "rollout_stay" for stay decisions.
  - `t` is the decision number, so a transition joins its trace decision on (episode, t).
- **Checks**:
  - Recording does not change episodes: benchmark inits reproduce v3 exactly with and without recording.
  - A replay of the stored chunks re-encodes the stored frames: tokens bit-equal except render noise ≤ 7.8e-3; proprio exact.

| Task | success (100 sampled) | benchmark v3 (50) | transitions | rollout / rollout_stay | from successful episodes |
|---|---|---|---|---|---|
| 0 | 0.14 | 0.16 | 2757 | 2298 / 459 | 177 |
| 1 | 0.05 | 0.00 | 2940 | 2475 / 465 | 90 |
| 2 | 0.39 | 0.32 | 2384 | 2139 / 245 | 554 |
| 3 | 0.72 | 0.70 | 1404 | 1249 / 155 | 564 |
| 4 | 0.40 | 0.46 | 2159 | 1702 / 457 | 359 |
| 5 | 0.46 | 0.50 | 2032 | 1352 / 680 | 412 |
| 6 | 0.02 | 0.08 | 2955 | 2449 / 506 | 15 |
| 7 | 0.00 | 0.00 | 3000 | 2721 / 279 | 0 |
| 8 | 0.08 | 0.08 | 2825 | 2700 / 125 | 65 |
| 9 | 0.98 | 0.96 | 746 | 727 / 19 | 686 |
| **all** | **0.324** | **0.326** | **23,202** | **19,812 / 3390** | **2922** |

- Frames: 24,202 (one chain per episode: transitions + 1000).
- Success on sampled starts equals success on benchmark inits (two-proportion p = 0.94). The failure categories have the same distribution:

| Starts | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| sampled (1000) | 324 | 51 | 110 | 128 | 115 | 230 | 42 |
| benchmark (500) | 163 | 35 | 54 | 59 | 59 | 117 | 13 |

- Shard manifests store each episode's `EpisodeResult`, including NaN fields (`mean_effect_cosine`, `choice_agreement` under observed effects). Python's json reads them; strict JSON parsers do not.

## Retrained dynamics heads: contribution of each data source (phase 7)

Three heads were trained with the phase-6 trainer, 3000 steps, seed 0, batch 256, otherwise as `goal_dynamics_short`:

| Head | Data | Run directory |
|---|---|---|
| +A | demonstrations + data A (fraction 0.5) | `outputs/runs/goal_dynamics_perturb` |
| +B | demonstrations + data B round 0 (0.5) | `outputs/runs/goal_dynamics_rollout` |
| +A+B | demonstrations + 0.25 A + 0.25 B | `outputs/runs/goal_dynamics_both` |

- **Training**: commit `b0e2dcf`; node-64, one L40S each, 473–520 s; logs `tmp/v3/train_goal_dynamics_*.log`.
- **Evaluation launchers**: `tmp/v3/eval_dyn_closed_loop.sh` (node-61) and `tmp/v3/eval_dyn_counterfactuals.sh` (node-64), commit `cdca0e4`.

### Error on all HELDOUT demonstration transitions

4864 transitions from the feature cache (`tmp/analysis/p7_heldout_demo_error.py`):

| Head | median relative error | mean cos |
|---|---|---|
| baseline | 0.215 | 0.922 |
| +A | 0.248 | 0.924 |
| +B | 0.266 | 0.922 |
| +A+B | 0.249 | 0.930 |

Accuracy on demonstrations drops a little because demonstrations fill only half of each batch.

### HELDOUT counterfactual check

Phase-12 protocol, seed 0; outputs `outputs/dynamics_counterfactuals/goal_dynamics_{perturb,rollout,both}_step3000_heldout_seed0/`; comparison script `tmp/analysis/p7_cf_compare.py`.

Median relative error / mean cos by candidate group:

| Head | demo | stay | σ 0.5 | σ 1 | σ 2 |
|---|---|---|---|---|---|
| baseline | 0.199 / 0.953 | 1.397 / 0.501 | 0.265 / 0.912 | 0.435 / 0.819 | 0.655 / 0.669 |
| +A | 0.230 / 0.951 | 0.729 / 0.683 | 0.252 / 0.927 | 0.304 / 0.897 | 0.384 / 0.854 |
| +B | 0.242 / 0.936 | 1.073 / 0.421 | 0.298 / 0.890 | 0.464 / 0.798 | 0.638 / 0.699 |
| +A+B | 0.213 / 0.948 | 0.820 / 0.621 | 0.254 / 0.916 | 0.325 / 0.881 | 0.416 / 0.830 |

Ranking of the 14 candidates of each state:

| Head | mean ρ | same top-1 | true m of the Δ̂ choice | true best in Δ̂ top 3 | Δ̂ m > 0 but true m ≤ 0: stay / σ 0.5 / σ 1 / σ 2 |
|---|---|---|---|---|---|
| baseline | 0.499 | 0.264 | 0.321 | 0.53 | 0.16 / 0.04 / 0.12 / 0.23 |
| +A | 0.722 | 0.434 | 0.426 | 0.65 | 0.09 / 0.04 / 0.05 / 0.07 |
| +B | 0.480 | 0.163 | 0.361 | 0.38 | 0.10 / 0.04 / 0.09 / 0.16 |
| +A+B | 0.663 | 0.372 | 0.411 | 0.59 | 0.12 / 0.04 / 0.06 / 0.07 |

Data A is built the same way as the check's perturbed candidates (same σ values, from TRAIN rather than HELDOUT states), so the check is in-distribution for +A and +A+B.

### Closed loop

Frozen retrieval, predicted effects, rule 32/128/512; 10 tasks × 50 benchmark inits, paired with `v3_frozen_predicted_rule` (baseline head); outputs `outputs/closed_loop/v3_frozen_predicted_rule_dyn_{perturb,rollout,both}/`; script `tmp/analysis/p7_closed_loop.py`.

| Task | observed effects | Δ baseline | Δ +A | Δ +B | Δ +A+B |
|---|---|---|---|---|---|
| 0 | 0.16 | 0.06 | 0.20 | 0.30 | 0.34 |
| 1 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 |
| 2 | 0.32 | 0.12 | 0.28 | 0.32 | 0.24 |
| 3 | 0.70 | 0.26 | 0.46 | 0.22 | 0.36 |
| 4 | 0.46 | 0.28 | 0.26 | 0.30 | 0.36 |
| 5 | 0.50 | 0.42 | 0.74 | 0.58 | 0.62 |
| 6 | 0.08 | 0.06 | 0.10 | 0.06 | 0.18 |
| 7 | 0.00 | 0.00 | 0.06 | 0.02 | 0.06 |
| 8 | 0.08 | 0.12 | 0.18 | 0.02 | 0.20 |
| 9 | 0.96 | 0.86 | 0.96 | 0.80 | 0.92 |
| **all** | **0.326** | **0.218** | **0.326** | **0.264** | **0.328** |

Paired exact McNemar tests (first-only / second-only successes):

| Comparison | counts | p |
|---|---|---|
| +A vs baseline | 80 / 26 | 1.4e-7 |
| +B vs baseline | 68 / 45 | 0.038 |
| +A+B vs baseline | 83 / 28 | 1.7e-7 |
| +A vs +B | 70 / 39 | 0.0039 |
| +A+B vs +A | 62 / 61 | 1.0 |
| +A+B vs +B | 83 / 51 | 0.0072 |
| +A vs observed effects | 65 / 65 | 1.0 |
| +A+B vs observed effects | 74 / 73 | 1.0 |
| +B vs observed effects | 50 / 81 | 0.0085 |

Per episode and failure categories (`outputs/analysis/v3_dyn_failure_classification.txt`):

| Head | stays / escalations per episode | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|---|
| baseline | 6.6 / 5.1 | 109 | 133 | 50 | 51 | 35 | 92 | 30 |
| +A | 5.3 / 3.9 | 163 | 80 | 44 | 60 | 44 | 82 | 27 |
| +B | 8.7 / 6.7 | 132 | 166 | 37 | 28 | 50 | 61 | 26 |
| +A+B | 8.8 / 6.8 | 164 | 181 | 25 | 21 | 29 | 53 | 27 |

Contribution of each source:

- **Data A** (executed perturbations and stay from demonstration states) carries the gain. It closes the gap between predicted and observed effects in the closed loop (0.218 → 0.326; +A vs observed 65 / 65). On the counterfactual check it halves the σ = 2 error (0.655 → 0.384), cuts the σ = 2 false positives from 23% to 7%, and raises the ranking correlation from 0.50 to 0.72.
- **Data B** (round-0 rollouts, 87% from failed episodes) helps less: +0.046 in the closed loop (p = 0.038), and no gain on the counterfactual ranking (ρ 0.48). Adding B to A changes success by +0.002 (62 / 61), but it makes Δ more reluctant to move: stays per episode rise 5.3 → 8.8 and stuck episodes 80 → 181, while other failure categories shrink.

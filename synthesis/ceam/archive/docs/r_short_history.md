# r with a short history (phase 24)

The action representation r(o, a) of `docs/framework.md` clause 1 is conditioned on the observation and the action. Here the observation also holds a short history, and r is compared with R6_all (`docs/query_match.md`, "R6 on all data (phase 19)") with only that input changed. Two histories are tested:

- **R6_prev**: the action chunk executed just before o (10 steps, standardized like a);
- **R6_state**: ψ of the frame observed one decision (10 env steps) before o, under the same fixed PCA as ψ(o).

Everything else is R6_all's: the twelve training stores and their held-out rules, the PCA `outputs/feature_pca/r6_all_stores/pca.npz`, L_pred per dim only, 20,000 updates, seed 0, batch 256, the same monitor. Both are opt-in; without them every existing path is unchanged.

## Code

| Class | File |
|---|---|
| `ChunkHistory`, `PreviousChunks` | `ceam/data/previous_chunks.py` |
| `StateHistory`, `PreviousStates` | `ceam/data/previous_states.py` |
| `HistoryDynamicsHead` | `ceam/model/history_dynamics_head.py` |
| `PreviousStateDynamicsHead` | `ceam/model/previous_state_dynamics_head.py` |
| `InteractionEffect`, `EffectHead` | `ceam/training/interaction_effect.py` |
| `HistoryConfig`, `PreviousStateConfig` (fields `history`, `previous_state` of `InteractionConfig`) | `ceam/training/interaction_config.py` |
| `LoadedHistoryFixedEncoderInteraction`, `LoadedPreviousStateFixedEncoderInteraction` (`InteractionCheckpointStore.load_fixed_encoder_history`, `load_fixed_encoder_previous_state`) | `ceam/training/interaction_checkpoint_store.py` |

Scripts: `scripts/build_previous_chunks.py`, `scripts/build_previous_states.py`. Configs: `configs/interaction/frozen_pca_delta_all_stores_prev_chunk.json`, `frozen_pca_delta_all_stores_prev_state.json`; each is `frozen_pca_delta_all_stores.json` with its own run dir and one added key.

**Effect heads.** Both wrap a `DynamicsHead(config)` (`head`) and use its layers; `DynamicsHead` is unchanged. `head` is built first, so under the trainer's seed its initial values equal those of R6_all's effect head (a `DynamicsHead` built under the same seed; checked bitwise); the history parameters are drawn afterwards. The output layer is zero at construction, so Δ̂ = 0 at step 0 as in R6_all.

| | `HistoryDynamicsHead` | `PreviousStateDynamicsHead` |
|---|---|---|
| history input | previous chunk [10, 7], standardized | ψ(o_prev) [256] |
| tokens added | 10: `history_projection`(chunk step) + `history_step_embedding` | 1: `previous_state_projection`(ψ(o_prev)) |
| no history | a learned `no_history_token` (+ the step embedding) at the 10 positions | a learned `no_previous_state_token` |
| sequence | [readout, ψ, proprio, 128 visual, 10 history, 10 action] | [readout, ψ, history, proprio, 128 visual, 10 action] |
| parameters | 13,325,568 (R6_all 13,315,840; +9,728) | 13,447,936 (+132,096) |

Where a transition has no history, its stored zeros are not read: the output does not change when they are replaced by other values (checked).

**Data path.** `InteractionSampler` takes optional `previous_chunks` / `previous_states` (one file per store); the batch then carries `history` / `previous_state`. They add no random draw: the drawn indices and every other batch field are those without them (checked on real stores, 5 batches of 64 per split). `InteractionEffect.predict` is the one place where the trainer and the monitor call the effect head; it calls a `DynamicsHead` exactly as before and refuses a head and a batch that do not match. `previous_state` needs `fixed_encoder` and excludes `history`.

## The history of each transition

A `TransitionStore` records no previous chunk: its table holds the frames of o and o′, the chunk, kind, σ, task, `episode`, `t` and the success flags (`docs/interaction_data.md`). The previous chunk is recovered per store by explicit rules (`STORE_RULES` of `scripts/build_previous_chunks.py`), tried in this order:

1. **chain**: the transition of the same store whose o′ is this o. Consecutive decisions of a rollout, a walk or a rendered demonstration share that frame; no frame is the o′ of two transitions.
2. **demo**: o is a demonstration frame (episode, t) or a restore of it; the previous chunk is the dataset actions a_{t−10} … a_{t−1} of the action cache, when t ≥ 10.
3. **record**: o is a fresh restore of recorded decision d of benchmark init `episode`; the previous chunk is the recorded π-mean chunk of decision d − 1, read from the transition (task, init, d − 1) of the re-simulated mean store of the same name.

Otherwise the transition has no previous chunk ("none"): the first decision after the settle steps of a benchmark init (rollouts, π-mean records, random walks, noisy π), a restored decision 1, and demonstration states with t < 10. The previous frame of R6_state follows the same rules: the current frame of the chained transition, the `demo_renders` frame (episode, t − 10), or the restored frame of decision d − 1 in the mean store; `PreviousStates` holds ψ of it under the R6_all PCA, and exists exactly where a previous chunk exists.

| Store | Rules | TRAIN: with / without | HELDOUT: with / without |
|---|---|---|---|
| `demo_renders` | chain | 39,801 / 3,900 | 4,434 / 430 |
| `demo_perturbations` (data A) | demo | 28,335 / 2,865 | — |
| `rollouts_r0` | chain | 17,932 / 800 | 4,270 / 200 |
| `resim_phase26_mean_{fixed,flow,flow_full_sgd}` | chain | 5,224 / 400, 4,401 / 400, 4,409 / 400 | 1,136 / 100, 1,106 / 100, 1,073 / 100 |
| `resim_phase26_sweep_{fixed,flow,flow_full_sgd}` | record | 12,160 / 2,432, 6,080 / 1,216, 6,080 / 1,216 | 1,600 / 320, 1,600 / 320, 1,440 / 288 |
| `random_demo_branch_train` | chain (walk steps 2–4), demo (branches, walk step 1) | 34,929 / 2,511 | — |
| `random_init_walk` | chain | 3,800 / 200 | — |
| `pi_noise_nopsi` | chain | 23,200 / 800 | 5,800 / 200 |
| **all twelve** | | **186,351 / 17,140 (91.6 % with)** | **22,459 / 2,058 (91.6 % with)** |
| `counterfactual_eval_heldout` (evaluation) | demo | — | 7,552 / 704 (236 / 22 states) |

Checks (`outputs/previous_chunks/all_stores/report.json`, `outputs/previous_states/all_stores_r6_all_pca/report.json`):

- In `demo_renders` every chained previous chunk equals the action-cache chunk a_{t−10} … a_{t−1} of its (episode, t): 44,235 compared, max |Δ| 0. Every chained previous frame is its frame (episode, t − 10).
- Every chain link joins the same task and `episode`; t(o) − t(previous) is 10 in `demo_renders`, 1 in the rollout, mean, walk and noisy-π stores, and 0 in the demo-branch walks (whose `t` is the branch point).
- Stores without the chain rule (data A, the sweep stores, the evaluation set) contain no chain link.
- Record rule: every sweep transition with d > 1 finds decision d − 1 in its mean store (0 missing). The sweep states without history are exactly those at decision 1 (2,752 / 1,536 / 1,504 transitions = 86 / 48 / 47 states).
- ψ(o_prev) of a chained transition equals ψ(o) of the chained transition (512 compared, max |Δ| 0). For data A, the `demo_renders` frame (episode, t) and data A's own o frame give the same ψ (256 compared, max relative difference 0).
- Every file reloads equal to what was written.

`PreviousChunks.load` and `PreviousStates.load` refuse a file built for other shards or another transition count, and `PreviousStates.load` one built with another PCA than the run's fixed encoder.

## Existing paths unchanged

`scripts/compare_interaction_runs.py` (every logged value except timings, every checkpoint tensor, the config) on runs of the same config with the code before this phase (`6defed9`, extracted to `tmp/s5p24/base_code_6defed9`) and with this phase's code (configs in `tmp/s5p24/configs/`). The `6defed9` and `2bd0f58` runs ran on node-63 CUDA 1, the `a5ebc7e` runs on CUDA 0 (both NVIDIA L40S).

| Config | Code | Compared | Differences |
|---|---|---|---|
| R6 smoke (`tmp/s4p9/configs/final_r6_smoke.json`, 500 updates, fixed encoder) | `2bd0f58` (first variant) and `a5ebc7e` (both variants) against `6defed9` | 55 metrics lines, 5 monitor lines, 6 scale lines, 5 checkpoints (330 tensors), each | 0, 0 |
| `smoke.json` (trainable trunk; stopped by K4a at update 100 in all three) | the same | 11 metrics lines, 1 monitor line, 2 scale lines, 1 checkpoint (123 tensors), each | 0, 0 |
| R6 smoke with the previous chunk (`history_smoke.json`) | `a5ebc7e` against `2bd0f58` | 55 metrics lines, 5 monitor lines, 6 scale lines, 5 checkpoints (350 tensors) | 0 |

Peak GPU memory, which is compared as a logged value, is equal as well (3,417,644,032 bytes for the R6 smoke). Every config in `configs/interaction/` reads and writes back to its exact file text. `InteractionCheckpointStore.load_fixed_encoder` still loads R6_all's checkpoint and refuses a checkpoint with history; the loaders of the two variants refuse every other checkpoint.

Other checks (`tmp/runlogs/s5p24/check_history.py`, `check_previous_state.py`; logs `logs/check_history.log`, `logs/check_previous_state.log`):

- Under one seed, the shared layers of both new heads equal a `DynamicsHead` bitwise (64 tensors); Δ̂ = 0 at construction.
- With a randomized output layer: rows without history give the same output whatever the stored previous chunk or state (max |Δ| 0); rows with history change (max |Δ| 0.71 and 0.29); leading dims [2, 3] equal the flat call.
- bf16 autocast against float32: median relative difference 0.0064 (previous chunk) and 0.0067 (previous state). GPU against CPU in float32 for `HistoryDynamicsHead`: max |Δ| 7.8e-4 on outputs of order 1.
- `InteractionEffect` on a `DynamicsHead` is bitwise the direct call; it refuses each mismatched head and batch.

## Runs

| | R6_prev | R6_state |
|---|---|---|
| config | `frozen_pca_delta_all_stores_prev_chunk.json` | `frozen_pca_delta_all_stores_prev_state.json` |
| history files | `outputs/previous_chunks/all_stores/` | `outputs/previous_states/all_stores_r6_all_pca/` |
| run | `outputs/runs/interaction_frozen_pca_delta_all_stores_prev_chunk_seed0` | `outputs/runs/interaction_frozen_pca_delta_all_stores_prev_state_seed0` |
| code | `2bd0f58` | `a5ebc7e` |
| host, GPU | node-63 CUDA 0 (L40S), through `scripts/cluster_run.sh --gpus 0` | node-63 CUDA 1, `--gpus 1` |
| time (EDT, 2026-09-29) | 02:09–03:24 | 02:50–04:14 |
| store loading | 425 s | 359 s |
| 20,000 updates | 3,991 s with the monitor; median 6.7 updates/s without it | 4,595 s; median 6.8 updates/s |
| peak GPU memory | 3.38 GiB | 3.20 GiB |
| exit | 0; no criterion held, no `monitor_stop.md` | 0; the same |

`gpu_guard.sh` passed at every launch; no other user's process was on either card. R6_all ran alone on node-65 at a median 7.7 updates/s with 3.18 GiB; here other runs of this phase shared the node, so the update rates are not comparable. The TRAIN and HELDOUT pools, the monitor's 4,096 held-out transitions (78 stay) and σ_ψ at step 0 (1.04877) are R6_all's.

## Training and the monitor

Batch values, means over the logged lines of each window (`metrics.jsonl`; the loss is L_pred / 256):

| Updates | R6_all: L_pred / 256, cos(Δ̂, z) | R6_prev | R6_state |
|---|---|---|---|
| 0 | 0.681, 0 | 0.681, 0 | 0.681, 0 |
| 1,500–2,000 | 0.277, 0.705 | 0.274, 0.707 | 0.276, 0.705 |
| 7,500–8,000 | 0.160, 0.817 | 0.159, 0.818 | 0.159, 0.818 |
| 19,500–19,999 | 0.0991, 0.875 | 0.0981, 0.876 | 0.0984, 0.876 |

Per source over updates 19,500–19,999 the three differ by at most 0.006 in L_pred / 256 (largest: `random_init_walk` 0.144 / 0.138 / 0.138).

Monitor, the same 4,096 held-out transitions for all three (`monitor.jsonl`):

| Step | R6_all: R² (Δ̂ rank) | R6_prev | R6_state |
|---|---|---|---|
| 2,000 | 0.523 (63.8) | 0.527 (64.1) | 0.524 (63.4) |
| 6,000 | 0.565 (74.1) | 0.571 (73.8) | 0.566 (74.4) |
| 10,000 | 0.564 (80.9) | 0.572 (80.4) | 0.564 (80.2) |
| 14,000 | 0.558 (82.7) | 0.569 (81.9) | 0.562 (83.2) |
| 20,000 | 0.550 (84.0) | 0.562 (83.9) | 0.554 (84.2) |

- All three reach their highest held-out R² between steps 6,000 and 10,000 and decline slightly afterwards.
- K0–K4 hold nowhere. Mean ‖Δ̂‖ at step 20,000 against step 500: 10.69 / 7.44 (R6_all), 10.63 / 7.41 (R6_prev), 10.69 / 7.40 (R6_state).
- Stay transitions (78) at step 20,000, ‖Δ̂‖ quartiles: 2.34 / 3.31 / 4.71 (R6_all), 1.35 / 2.81 / 5.21 (R6_prev), 2.31 / 3.39 / 5.16 (R6_state), against ‖z‖ 2.28 / 5.22 / 7.11.

## Held-out prediction per source (step 20,000)

All HELDOUT transitions of the training sources; each r against z in the R6_all ψ space (`outputs/analysis/r_short_history/both/heldout.json`, script `tmp/runlogs/s5p24/heldout_p24.py`). "No history" evaluates the same trained head with the no-history token for every transition. R² / mean cos(Δ̂, z):

| Held-out part | n | R6_all | R6_prev | R6_prev, no history | R6_state | R6_state, no history |
|---|---|---|---|---|---|---|
| demo_renders `heldout_*` | 4,864 | 0.677 / 0.799 | **0.682** / 0.802 | 0.648 / 0.782 | 0.679 / 0.800 | 0.661 / 0.790 |
| rollouts_r0 starts 80–99 | 4,470 | 0.519 / 0.624 | **0.532** / 0.629 | 0.492 / 0.616 | 0.521 / 0.625 | 0.494 / 0.616 |
| — rollout_stay | 533 | −0.011 / 0.253 | **0.070** / 0.260 | −0.357 / 0.240 | 0.003 / 0.255 | −0.107 / 0.252 |
| resim mean fixed, inits 40–49 | 1,236 | 0.708 / 0.801 | **0.710** / 0.802 | 0.674 / 0.781 | 0.708 / 0.801 | 0.689 / 0.790 |
| resim mean flow | 1,206 | 0.707 / 0.795 | **0.718** / 0.801 | 0.682 / 0.781 | 0.712 / 0.797 | 0.696 / 0.789 |
| resim mean flow_full_sgd | 1,173 | 0.710 / 0.798 | **0.719** / 0.800 | 0.683 / 0.780 | 0.711 / 0.797 | 0.697 / 0.787 |
| resim sweep fixed | 1,920 | 0.558 / 0.656 | 0.559 / 0.665 | 0.543 / 0.660 | **0.567** / 0.670 | 0.539 / 0.661 |
| resim sweep flow | 1,920 | 0.554 / 0.672 | **0.576** / 0.684 | 0.539 / 0.668 | 0.557 / 0.675 | 0.546 / 0.671 |
| resim sweep flow_full_sgd | 1,728 | 0.528 / 0.655 | **0.555** / 0.645 | 0.502 / 0.629 | 0.521 / 0.646 | 0.528 / 0.642 |
| pi_noise_nopsi starts 80–99 | 6,000 | 0.377 / 0.590 | **0.384** / 0.595 | 0.366 / 0.585 | 0.375 / 0.591 | 0.360 / 0.586 |

- The R6_all column reproduces phase 19's `heldout.json` exactly (77 values, max |difference| 0).
- R6_prev's R² is at or above R6_all's on every part: +0.001 to +0.026, and +0.082 on the rollout stay transitions. R6_state differs from R6_all by −0.008 to +0.010 (rollout stay +0.014).
- On the 2,058 of 24,517 held-out transitions without a previous chunk, where a head sees its no-history token in both evaluations, R² differs from R6_all's by −0.003 to +0.027 (R6_prev) and −0.004 to +0.017 (R6_state), per part.
- With the history replaced by the no-history token, both heads fall below R6_all on every part: R6_prev by 0.010 to 0.034 (rollout stay 0.346), R6_state by 0.0004 (sweep flow_full_sgd) to 0.026 (rollouts_r0), rollout stay 0.095. The heads use the history.

## Distinguishability check (step 20,000)

The phase-19 check (`docs/query_match.md`, "R6 on all data (phase 19)") with the history of each state (`tmp/runlogs/s5p24/gate_p24.py`, node-63 CUDA 1, 185 s; `outputs/analysis/r_short_history/both/gate/{gate.json,effects.pt}`). The history is shared by the 32 candidates of a state:

- sweep state (task, init, decision d): the recorded π-mean chunk of decision d − 1, or ψ of its restored frame; the 181 states at d = 1 (their history is the no-history token) are exactly those without one. 145 of the 174 held-out states have a history.
- counterfactual state (episode, t): the dataset actions a_{t−10} … a_{t−1}, or ψ of `demo_renders` frame (episode, t − 10); 236 of the 258 states have one.

The phase-19 checks repeat identically (stored candidates reproduced from their candidate 0 in 1,077 of 1,084 states, store chunks equal to the stored candidates), and the R6_all rows reproduce phase 19's `gate.json` exactly (329 values, max |difference| 0).

**Gate numbers** (share ≥ 0.05, rank mean ≥ 4, cos mean ≤ 0.99), share / rank mean / cos mean:

| States | Candidates | R6_all | R6_prev | R6_state |
|---|---|---|---|---|
| sweep, all 1,084 | π (a) | 0.0504 / 3.69 / 0.938 (no) | 0.0511 / 3.73 / 0.936 (no) | 0.0504 / 3.69 / 0.937 (no) |
| | π (b) | 0.0436 / 3.88 / 0.947 (no) | 0.0438 / 3.91 / 0.946 (no) | 0.0434 / 3.87 / 0.947 (no) |
| sweep, inits 0–39 (910, trained on) | π (a) | 0.0519 / 3.63 / 0.935 (no) | 0.0526 / 3.67 / 0.934 (no) | 0.0518 / 3.63 / 0.935 (no) |
| | π (b) | 0.0445 / 3.85 / 0.945 (no) | 0.0448 / 3.86 / 0.944 (no) | 0.0442 / 3.84 / 0.945 (no) |
| **sweep, inits 40–49 (174, held out)** | π (a) | 0.0424 / 3.99 / 0.951 (no) | 0.0426 / 4.06 / 0.949 (no) | 0.0431 / 4.03 / 0.951 (no) |
| | π (b) | 0.0389 / 4.08 / 0.954 (no) | 0.0391 / 4.12 / 0.953 (no) | 0.0396 / 4.07 / 0.953 (no) |
| **counterfactual (258, held out)** | smooth random | 0.553 / 9.14 / 0.485 (**yes**) | 0.555 / 9.12 / 0.480 (**yes**) | 0.554 / 9.12 / 0.482 (**yes**) |

With the history, as without it, r passes on the smooth random candidates and fails on π's candidates, in both versions and on held-out states alike. Against R6_all, every entry moves by at most 0.0014 in share, 0.07 in rank mean and 0.005 in cos mean. The no-history evaluation lowers the share among π's candidates on all 1,084 states (a) to 0.048 (R6_prev) and 0.047 (R6_state).

**r against the true effects** (same ψ space; true effects in R6_all space: share 0.176, rank 11.78, cos mean 0.768 on all sweep states (a); 0.644, 13.76, 0.358 on the counterfactual set). Within-state R² / centered cos / Spearman of pairwise distances / spread ratio:

| States | R6_all | R6_prev | R6_prev, no history | R6_state | R6_state, no history |
|---|---|---|---|---|---|
| sweep inits 0–39, π (a) | 0.291 / 0.421 / 0.809 / 0.402 | 0.293 / 0.423 / 0.809 / 0.406 | 0.245 / 0.376 / 0.791 / 0.408 | 0.290 / 0.421 / 0.809 / 0.399 | 0.254 / 0.389 / 0.797 / 0.390 |
| **sweep inits 40–49, π (a)** | 0.069 / 0.215 / 0.751 / 0.377 | 0.070 / 0.215 / 0.757 / 0.384 | 0.056 / 0.206 / 0.756 / 0.384 | 0.065 / 0.211 / 0.754 / 0.379 | 0.065 / 0.209 / 0.754 / 0.366 |
| — of which d > 1 (145) | 0.046 / 0.180 / 0.738 / 0.356 | 0.047 / 0.179 / 0.744 / 0.350 | 0.030 / 0.168 / 0.743 / 0.370 | 0.041 / 0.174 / 0.742 / 0.359 | 0.041 / 0.172 / 0.741 / 0.343 |
| sweep d = 1 (181, no history) | 0.307 / 0.473 / 0.840 / 0.499 | 0.318 / 0.483 / 0.844 / 0.509 | = R6_prev | 0.308 / 0.474 / 0.841 / 0.491 | = R6_state |
| **counterfactual, smooth random** | 0.425 / 0.628 / 0.820 / 0.811 | 0.433 / 0.635 / 0.821 / 0.812 | 0.403 / 0.614 / 0.810 / 0.807 | 0.428 / 0.631 / 0.820 / 0.807 | 0.419 / 0.624 / 0.818 / 0.800 |

## Reading

- **Held-out prediction improves slightly with the previous chunk, not with the previous frame.** R6_prev predicts every held-out part at least as well as R6_all (R² +0.001 to +0.026; +0.082 on rollout stay transitions), and its monitor R² at step 20,000 is 0.562 against 0.550. R6_state stays within 0.01 of R6_all on every part except the rollout stay transitions (+0.014).
- **Both heads read their history.** Replacing it with the no-history token lowers held-out R² below R6_all's (for example demo_renders 0.648 and 0.661 against 0.677) and lowers the within-state R² against true effects (sweep inits 0–39: 0.245 and 0.254 against 0.291).
- **Within-observation distinguishability does not change.** Among π's 32 candidates at a sweep state the history is one shared input, and the gate numbers move by at most 0.0014 in share and 0.07 in rank mean: π's candidates still fail (share 0.05, rank 3.7 over all 1,084 states), the smooth random candidates still pass.
- **Agreement with the true effects on held-out states is unchanged.** Within-state R² on the 174 held-out sweep states is 0.070 (R6_prev) and 0.065 (R6_state) against 0.069; on the counterfactual states 0.433 and 0.428 against 0.425.
- Each variant is one run (seed 0), as R6_all. The spread of these quantities between seeds has not been measured, so it is not known whether differences of 0.01 in R² exceed it.

## Paths

| What | Where |
|---|---|
| history files | `outputs/previous_chunks/all_stores/<store>.npz` and `report.json`; `outputs/previous_states/all_stores_r6_all_pca/<store>.npz` and `report.json` |
| runs | `outputs/runs/interaction_frozen_pca_delta_all_stores_prev_chunk_seed0/`, `outputs/runs/interaction_frozen_pca_delta_all_stores_prev_state_seed0/` (r = `checkpoints/step_020000.pt`) |
| evaluation | `outputs/analysis/r_short_history/both/{gate/gate.json,gate/effects.pt,heldout.json}` (all three r); `outputs/analysis/r_short_history/prev_chunk/` (R6_all and R6_prev only, from the first pipeline; equal to the corresponding entries of `both/`: 1,638 gate and 609 held-out values, max |difference| 0) |
| smoke and comparison runs | `outputs/runs/s5p24_history_smoke{,_final}/`, `outputs/runs/s5p24_bitid_{r6,joint}_smoke_{base,new,final}/` |
| entry scripts, scripts, logs, tables | `tmp/runlogs/s5p24/` (`env.sh`, `start.sh`, `step_a.sh`, `train_hist.sh`, `bitid.sh`, `build_states.sh`, `step_b.sh`, `bitid2.sh`, `check_history.py`, `check_previous_state.py`, `gate_hist_p24.py`, `heldout_hist_p24.py`, `gate_p24.py`, `heldout_p24.py`, `tables_p24.py`, `tables_both.txt`, `logs/`); configs of the smoke and comparison runs `tmp/s5p24/configs/`; code before this phase `tmp/s5p24/base_code_6defed9/` |

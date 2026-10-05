# Closed-loop MVP on LIBERO-Goal

Policy without new learned mechanisms (`ceam/policy/`, `ceam/sim/`; entry `scripts/run_closed_loop.py`, summary `scripts/summarize_closed_loop.py`). At every decision point the current observation (two 512×512 views rotated 180°, proprio = `robot0_joint_pos` + `robot0_gripper_qpos`) is encoded with the frozen DINOv2 and the stage-one trunk to ψ(o); the K = 32 TRAIN-demonstration states nearest in ψ (with a full 10-step chunk after them; ψ table `outputs/state_tables/goal_instruction_contrast_step5000.npy`) give the candidate chunks; "stay" holds the current gripper for 10 steps with effect 0.

- MATCH: argmax of cos(z, q_l) over the K candidates and stay (ties → stay), z = the candidate's demonstrated effect ψ(o_{t+10}) − ψ(o_t), q_l = the stage-one query direction of the task's original text.
- RANDOM: one of the K candidates uniformly.

Protocol: the 10 LIBERO-Goal tasks, the 50 LIBERO benchmark init states per task, `env.seed(0)`, 10 settle steps, at most 300 env steps, success = LIBERO's `done`. Each process first checks the start observation against the task's dataset episode starts (wrist-view DINOv2 cosine 0.85–0.87 on every run, against 0.88 between dataset starts). Runs: match on node-70, random on node-69 (L40S, task i on CUDA device i mod 8), identical seeds and init order, so episodes pair by (task, init). Outputs `outputs/closed_loop/mvp_observed/task<i>_<strategy>/results.jsonl`.

## Success rate (50 inits per task)

| Task (dataset index) | match | random | match own-task fraction | match mean stay decisions |
|---|---|---|---|---|
| 0 open the middle drawer of the cabinet | 0.10 | 0.02 | 0.64 | 15.7 |
| 1 open the top drawer and put the bowl inside | 0.06 | 0.02 | 0.76 | 14.6 |
| 2 push the plate to the front of the stove | 0.60 | 0.04 | 0.86 | 5.7 |
| 3 put the bowl on the plate | 0.24 | 0.04 | 0.69 | 14.4 |
| 4 put the bowl on the stove | 0.24 | 0.00 | 0.43 | 19.6 |
| 5 put the bowl on top of the cabinet | 0.46 | 0.04 | 0.58 | 12.8 |
| 6 put the cream cheese in the bowl | 0.12 | 0.04 | 0.67 | 19.1 |
| 7 put the wine bottle on the rack | 0.02 | 0.02 | 0.54 | 19.6 |
| 8 put the wine bottle on top of the cabinet | 0.04 | 0.08 | 0.79 | 12.2 |
| 9 turn on the stove | 0.88 | 0.40 | 0.92 | 2.6 |
| **all (500 episodes)** | **0.276** | **0.070** | 0.70 | 13.6 |

Random picks own-task candidates in 13% of decisions (never stays).

Paired by (task, init): match-only successes 118, random-only 15, both 20; exact McNemar two-sided p = 5.1e-21.

Failures: 269 of the 362 failed match episodes have ≥ 10 stay decisions (224 have ≥ 20). Successful match episodes average 0.14 stay decisions and execute own-task chunks in 94% of decisions; failed ones average 18.8 stay decisions and 59% own-task chunks. A stay leaves the state unchanged, so retrieval returns the same candidates and the policy stays until the step limit (ISSUE-006).

Timing: decisions take 16–42 ms (encoding 15–27 ms); episodes 2–12 s; `/tmp/robosuite.log` unchanged by all runs.

## Predicted effects and the stay-escalation rule (phases 8–9)

Predicted effects: z = Δ(o_current, a_candidate) from `outputs/runs/goal_dynamics_short/checkpoints/step_003000.pt`, same candidates. Stay escalation: after 2 consecutive stay decisions, a decision whose selection at K = 32 is stay retrieves again with K = 128, then 512, until a candidate scores > 0; counter and K reset after a non-stay chunk. Runs: `outputs/closed_loop/mvp_predicted/`, `mvp_observed_escalation/`, `mvp_predicted_escalation/` (task i on CUDA device i mod 8 of node-70 / node-69, seed 0, `--check-start-observation`).

| Task | random | observed | predicted | observed + rule | predicted + rule |
|---|---|---|---|---|---|
| 0 | 0.02 | 0.10 | 0.04 | 0.12 | 0.02 |
| 1 | 0.02 | 0.06 | 0.02 | 0.08 | 0.02 |
| 2 | 0.04 | 0.60 | 0.40 | 0.58 | 0.46 |
| 3 | 0.04 | 0.24 | 0.40 | 0.28 | 0.38 |
| 4 | 0.00 | 0.24 | 0.18 | 0.30 | 0.22 |
| 5 | 0.04 | 0.46 | 0.38 | 0.56 | 0.42 |
| 6 | 0.04 | 0.12 | 0.08 | 0.10 | 0.10 |
| 7 | 0.02 | 0.02 | 0.02 | 0.00 | 0.02 |
| 8 | 0.08 | 0.04 | 0.14 | 0.04 | 0.14 |
| 9 | 0.40 | 0.88 | 0.86 | 0.88 | 0.86 |
| **all** | **0.070** | **0.276** | **0.252** | **0.294** | **0.264** |

Paired exact McNemar (first-only / second-only successes): predicted vs observed 47 / 59, p = 0.285; predicted vs random 103 / 12, p = 3.4e-19; observed + rule vs observed 19 / 10, p = 0.136; predicted + rule vs predicted 10 / 4, p = 0.18; predicted + rule vs observed + rule 54 / 69, p = 0.207.

Prediction quality inside the loop: mean cos(Δ̂, observed effect) over the candidates 0.904 (0.861–0.973 by task); the predicted and observed choices agree in 43.9% of decisions, mostly both staying; when both pick a chunk they pick the same one in 14.6% of decisions (Spearman of the K scores 0.37).

Escalation rarely finds a positive candidate: of 3964 expanded decisions (observed + rule), 139 found one at K = 128, 264 at K = 512 and 3561 still stayed (predicted + rule: 91 / 101 / 4327 of 4519). Mean stay decisions per episode drop from 13.6 to 10.0 (observed) and 10.8 (predicted). Decision latency on L40S: 14 ms observed, 19–20 ms predicted at K = 32, 41 ms at K = 128, 121 ms at K = 512.

## Failure breakdown from decision logs

Rules, applied in order to failed episodes: stuck = the last ≥ 5 decisions are stays; early wrong = ≥ 2 of the first 3 executed chunks come from another task; own chunks, not completed = ≥ 80% of executed chunks from the own task and < 5 stays; otherwise mixed/other.

| Version | success | stuck | own chunks, not completed | early wrong | mixed/other |
|---|---|---|---|---|---|
| observed | 138 | 283 | 66 | 4 | 9 |
| predicted | 126 | 278 | 66 | 3 | 27 |
| observed + rule | 147 | 204 | 87 | 36 | 26 |
| predicted + rule | 132 | 223 | 85 | 4 | 56 |

In every stuck episode the candidate set at the last decision contains no own-task state (observed: 283 / 283; predicted + rule: 201 / 223), up to K = 512 with the rule. In the observed runs without the rule, own-task candidates are absent from the first decision on in 140 of the 283 stuck episodes; in the other 143 they vanish for good right after an own-task chunk was executed (142 / 143). "Own chunks, not completed" concentrates on the drawer tasks (0, 1) and the wine-bottle tasks (7, 8).

## Traced failure classification (phase 10)

`--trace` records every decision (candidate counts, own-task candidates, best score per K, choice, gripper commands) and a scene snapshot (`ceam/sim/scene_probe.py`: movable-object positions, drawer slides and stove knob, LIBERO's goal check, end-effector position, gripper opening, fingerpad contacts). Traced reruns of observed + rule and predicted + rule reproduce the untraced runs episode for episode. `scripts/classify_closed_loop_failures.py` assigns each failed episode one category, in order: stuck (last 5 decisions are stays); goal object never moved (< 1 cm; drawer slide < 1 cm; knob < 0.05 rad), split by whether another object moved ≥ 2 cm; goal object moved but goal not reached, split into lifted (≥ 3 cm) and displaced without lift; partial joint motion for articulated goals; other.

| Run | success | stuck | goal not moved, other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint | other |
|---|---|---|---|---|---|---|---|---|
| observed + rule | 147 | 204 | 27 | 19 | 48 | 29 | 26 | 0 |
| predicted + rule | 132 | 223 | 33 | 17 | 25 | 44 | 26 | 0 |

Observed + rule by task: task 0 (middle drawer) 21 stuck and 22 "other object moved"; task 1 (top drawer + bowl) 26 partial joint motion (drawer opened, bowl not placed); tasks 6, 7, 8 are dominated by stuck (37, 31, 32).

Where own-task candidates vanish (observed + rule, 204 stuck episodes): never present in 68; in the other 136 they vanish right after an own-task chunk (135) at a median decision 10. At that moment the gripper is open in 62, closed and holding the goal object in 7, closed on nothing in 67 (median end-effector–object distance 0.21 m). Task 6: all 33 vanish with the gripper closed on nothing 0.13–0.34 m from the cream cheese (a missed grasp); task 0: all 14 with the gripper open and the drawer closed; task 7: 22 of 31 never had own-task candidates.

Nearest own-task training state at the 3561 stuck decisions (pool: 43701 TRAIN retrieval rows): its rank by ψ distance has median 2571 and is never ≤ 512; by frozen-feature similarity (mean over views of the cosine of L2-normalised mean DINOv2 tokens) median 177, ≤ 32 in 30% and ≤ 512 in 74% of decisions. ψ distance to the nearest state overall 1.45 against 15.3 to the nearest own-task state (dataset starts are ~14 apart). HELDOUT dataset frames have rank 1 in both spaces.

## Stuck-handling experiments (phase 11)

Observed effects, rule 32/128/512, traced; E1 executes the first 5 actions of each chunk (score still the 10-step effect); E2 executes the best candidate of the K = 512 set when the escalation ends in stay; E3 both. Outputs `outputs/closed_loop/{e1_execute5,e2_stuck_best,e3_execute5_stuck_best,e_best_predicted}/`.

| Run | all | vs observed + rule (first-only / second-only, p) |
|---|---|---|
| observed + rule | 0.294 | — |
| E1 execute 5 | 0.250 | 49 / 71, p = 0.055 |
| E2 stuck → best | 0.300 | 17 / 14, p = 0.72 |
| E3 both | 0.246 | 46 / 70, p = 0.032 |
| E2 with predicted effects | 0.286 | 61 / 65, p = 0.79 |

None of the 220 E2 episodes with forced moves succeeds (955 forced moves, none an own-task chunk). These comparisons were run before the gripper-reset fix (ISSUE-008): the start state of 128 of 500 episodes depended on the previous episode; restricted to the 372 episodes with identical starts, E2 vs observed + rule has 0 discordant pairs, E1 27 / 54 (p = 0.0036), E3 23 / 52 (p = 0.0011).

## Reruns with the gripper-reset fix (v2)

Commit `aba2587` zeroes robosuite's gripper action accumulator at every reset (ISSUE-008), so every episode's start depends only on its init index and position in the sequence and is identical across configurations. All five main configurations rerun with tracing on node-70 (8 L40S, 50 processes, 2 CPU threads each; launcher `tmp/v2/launch.sh`), seed 0, inits 0–49, 300 env steps, 10 settle steps, `--check-start-observation` (wrist cosine 0.850–0.889, agent 0.929–0.963). Outputs `outputs/closed_loop/v2_{random,observed,predicted,observed_rule,predicted_rule}/`; summary `outputs/analysis/v2_summary.txt`, classification `outputs/analysis/v2_failure_classification.txt`. These are the reference numbers; earlier runs remain on disk.

| Task | random | observed | predicted | observed + rule | predicted + rule |
|---|---|---|---|---|---|
| 0 open the middle drawer | 0.00 | 0.08 | 0.14 | 0.10 | 0.14 |
| 1 open the top drawer, bowl inside | 0.02 | 0.06 | 0.02 | 0.08 | 0.02 |
| 2 push the plate to the stove front | 0.08 | 0.56 | 0.42 | 0.56 | 0.50 |
| 3 bowl on the plate | 0.08 | 0.16 | 0.36 | 0.20 | 0.34 |
| 4 bowl on the stove | 0.02 | 0.26 | 0.12 | 0.32 | 0.12 |
| 5 bowl on top of the cabinet | 0.04 | 0.50 | 0.38 | 0.60 | 0.38 |
| 6 cream cheese in the bowl | 0.02 | 0.12 | 0.04 | 0.12 | 0.04 |
| 7 wine bottle on the rack | 0.00 | 0.02 | 0.02 | 0.02 | 0.02 |
| 8 wine bottle on top of the cabinet | 0.10 | 0.22 | 0.14 | 0.22 | 0.18 |
| 9 turn on the stove | 0.24 | 0.90 | 0.92 | 0.90 | 0.92 |
| **all (500)** | **0.060** | **0.288** | **0.256** | **0.312** | **0.266** |

Paired by (task, init), exact McNemar (first-only / second-only successes):

| Comparison | Counts | p |
|---|---|---|
| observed vs random | 126 / 12 | 3.9e-25 |
| predicted vs random | 113 / 15 | 8.9e-20 |
| observed + rule vs random | 138 / 12 | 2.6e-28 |
| predicted vs observed | 47 / 63 | 0.15 |
| observed + rule vs observed | 12 / 0 | 4.9e-4 |
| predicted + rule vs predicted | 7 / 2 | 0.18 |
| predicted + rule vs observed + rule | 53 / 76 | 0.052 |

Mean per episode: random 288 env steps; observed 244 steps, 13.5 stays; predicted 249, 13.4; observed + rule 240, 9.4 stays, 7.3 escalations; predicted + rule 248, 10.4, 8.7. Match executes own-task chunks in 69–73% of decisions, random in 12%.

Failure categories (traced, classifier as above):

| Run | success | stuck | goal not moved, other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| observed | 144 | 275 | 11 | 11 | 28 | 14 | 17 |
| predicted | 128 | 270 | 23 | 12 | 22 | 29 | 16 |
| observed + rule | 156 | 188 | 23 | 22 | 60 | 27 | 24 |
| predicted + rule | 133 | 215 | 36 | 20 | 30 | 41 | 25 |

Observed + rule, stuck episodes (188): own-task candidates never present in 63, vanished after an own-task chunk in 125 (median decision 10); at that moment the gripper was open in 51, closed and holding the goal object in 7, closed on nothing in 67 (median 0.16 m from the goal object; task 6: 29 of 33). By task, stuck dominates tasks 0, 6, 7, 8; "lifted, not at target" concentrates on the bowl tasks 3 and 4 (15, 18); partial joint motion on task 1 (24: drawer opened, bowl not placed); task 0 fails by moving another object in 17 episodes.

## Frozen-feature retrieval (second plan, phases 1–2)

Candidates can be retrieved in a second space: `--retrieval-space frozen --key-table outputs/state_tables/libero_goal_frozen_keys.npy`, built with `scripts/build_state_table.py --space frozen`.

- **Key table**: (52895, 1536) float32, md5 `4ae477a3862b40ad680be4b22c9d2b93`; about 14 s on CPU.
- **Key**: per view, the mean of the 64 frozen DINOv2 tokens, L2-normalised; the two views concatenated. Euclidean ranking on the key equals ranking by the mean over views of the token-mean cosines, the similarity of the phase-10 neighbour analysis. The table reproduces that analysis exactly: median own-task rank 177; ≤ 512 in 74% of stuck decisions.
- **Unchanged**: effects, scores and the rule. The default `psi` reproduces v2 decision for decision (tasks 2 and 6, inits 0–4, all trace fields).
- **Neighbours**: querying HELDOUT frames against all rows, the nearest other frame is in the same episode for 96.6% of frames (psi 74.4%).
- **Latency**: about 1.2 ms more per retrieval than psi on L40S.

Runs `outputs/closed_loop/v3_frozen_{observed,predicted}_rule/`:

- launcher `tmp/v3/launch_frozen.sh`;
- node-70, 20 processes, commit `5c3d2fc`;
- same protocol and flags as v2 with the rule (32/128/512, patience 2), traced;
- summary script `tmp/analysis/v3_frozen_summary.py`; classification `outputs/analysis/v3_frozen_failure_classification.txt`.

| Task | psi, observed + rule | frozen, observed + rule | psi, predicted + rule | frozen, predicted + rule |
|---|---|---|---|---|
| 0 | 0.10 | 0.16 | 0.14 | 0.06 |
| 1 | 0.08 | 0.00 | 0.02 | 0.00 |
| 2 | 0.56 | 0.32 | 0.50 | 0.12 |
| 3 | 0.20 | 0.70 | 0.34 | 0.26 |
| 4 | 0.32 | 0.46 | 0.12 | 0.28 |
| 5 | 0.60 | 0.50 | 0.38 | 0.42 |
| 6 | 0.12 | 0.08 | 0.04 | 0.06 |
| 7 | 0.02 | 0.00 | 0.02 | 0.00 |
| 8 | 0.22 | 0.08 | 0.18 | 0.12 |
| 9 | 0.90 | 0.96 | 0.92 | 0.86 |
| **all** | **0.312** | **0.326** | **0.266** | **0.218** |

Paired exact McNemar (first-only / second-only successes):

| Comparison | Counts | p |
|---|---|---|
| frozen vs psi, observed + rule | 81 / 74 | 0.63 |
| frozen vs psi, predicted + rule | 45 / 69 | 0.031 |
| frozen: predicted vs observed | 36 / 90 | 1.6e-6 |

Mean per episode:

| Run | stay decisions | escalations | own-task chunks executed |
|---|---|---|---|
| psi, observed + rule | 9.4 | 7.3 | 70% |
| frozen, observed + rule | 3.6 | 1.9 | 87% |
| psi, predicted + rule | 10.4 | 8.7 | 73% |
| frozen, predicted + rule | 6.6 | 5.1 | 74% |

Failure categories (same classifier as v2):

| Run | success | stuck | goal not moved, other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| psi, observed + rule | 156 | 188 | 23 | 22 | 60 | 27 | 24 |
| frozen, observed + rule | 163 | 35 | 54 | 59 | 59 | 117 | 13 |
| psi, predicted + rule | 133 | 215 | 36 | 20 | 30 | 41 | 25 |
| frozen, predicted + rule | 109 | 133 | 50 | 51 | 35 | 92 | 30 |

With observed effects, frozen retrieval removes most stuck episodes (188 → 35), but success does not change significantly: the failures move to the goal object being moved without reaching the goal. Episode by episode (`tmp/analysis/v3_transitions.py`), of the psi-stuck episodes:

| Under frozen retrieval | episodes |
|---|---|
| displaced without lift | 50 |
| success | 48 |
| nothing moved | 30 |
| other object moved | 24 |
| lifted, not at target | 21 |

In the other direction, 31 psi successes become "displaced without lift". By task:

- the bowl tasks 3 and 4 gain (0.20 → 0.70, 0.32 → 0.46);
- the plate push (task 2) and the wine bottle on the cabinet (task 8) lose;
- task 0 fails mostly by moving another object (37).

With predicted effects, frozen retrieval gets stuck with own-task candidates present. In 87 of its 133 stuck episodes own-task candidates are in the final set. Among the decisions whose escalation ended in stay, the observed effects of the same candidates would have chosen a chunk in 94% (2219 of 2356), against 59% under psi retrieval (2474 of 4159). Δ(o_current, a) scores the chunks of visually similar demonstration states ≤ 0 at the current state.

Decision latency: 20 ms mean at K = 32 (observed); 25–29 ms mean decision time with predicted effects.

## Sampled starts and transition recording

`scripts/run_closed_loop.py` options:

- `--start sampled` runs episodes from `LiberoTaskEnv.reset_sampled`. `--inits` must then be `0-(N−1)`, run in order in one process; `init_index` is the sampled-episode index.
- `results.jsonl` and `trace.jsonl` gain a `start` field ("benchmark" or "sampled"). Lines without it read as benchmark.
- The summary and classification scripts pair episodes on (task, start, init).
- `--record-transitions DIR --code-commit SHA` writes one transition-store shard per process (`docs/data.md`, `docs/dynamics_data.md`).
- For sampled starts, `--check-start-observation` checks every episode's start rather than doing one extra reset, which would shift the sampled sequence.

## Matching against ĝ_l − ψ(o) (second plan, phase 12)

Inference-only experiment:

- **Change**: the match target (`--match-target`, `ceam/policy/candidate_selector.py`, commit `7834ba2`). `goal` scores cos(z, ĝ_l − ψ(o)), with ĝ_l the stage-one goal head's output for the task's original instruction. `goal_subspace` scores cos(Uᵀz, Uᵀ(ĝ_l − ψ(o))).
- **Why the subspace arm**: the goal loss constrains ĝ only in the U coordinates. In U, ĝ is within 0.3–1.0% of the start-to-end distance from the demonstrations' end states; in the full 256-d space it is 37–58% away; only 49–73% of ‖ĝ‖ lies in span(U) (`tmp/analysis/goal_head_check.py`).
- **Unchanged**: everything else, identical to the best configurations — frozen retrieval, rule 32/128/512, patience 2, predicted effects from the data-A head, 10 tasks × 50 benchmark inits, traced.
- **Check**: `--match-target query` reproduces v3 episode for episode.
- **Runs**: launcher `tmp/v3/launch_goal_match.sh` (node-61); outputs `outputs/closed_loop/v4_{goal,goalU}_{observed,predicted_dynA}/`; script `tmp/analysis/p12_goal_match.py`.

| Task | q_l, observed | q_l, Δ +A | ĝ − ψ, observed | ĝ − ψ, Δ +A | U(ĝ − ψ), observed | U(ĝ − ψ), Δ +A |
|---|---|---|---|---|---|---|
| 0 | 0.16 | 0.20 | 0.00 | 0.00 | 0.18 | 0.08 |
| 1 | 0.00 | 0.02 | 0.00 | 0.00 | 0.02 | 0.02 |
| 2 | 0.32 | 0.28 | 0.00 | 0.00 | 0.36 | 0.44 |
| 3 | 0.70 | 0.46 | 0.40 | 0.24 | 0.42 | 0.28 |
| 4 | 0.46 | 0.26 | 0.60 | 0.24 | 0.62 | 0.46 |
| 5 | 0.50 | 0.74 | 0.62 | 0.24 | 0.52 | 0.38 |
| 6 | 0.08 | 0.10 | 0.00 | 0.00 | 0.06 | 0.10 |
| 7 | 0.00 | 0.06 | 0.00 | 0.00 | 0.02 | 0.04 |
| 8 | 0.08 | 0.18 | 0.00 | 0.00 | 0.38 | 0.34 |
| 9 | 0.96 | 0.96 | 0.16 | 0.02 | 0.90 | 0.80 |
| **all** | **0.326** | **0.326** | **0.178** | **0.074** | **0.348** | **0.294** |

Paired exact McNemar (first-only / second-only successes):

| Comparison | counts | p |
|---|---|---|
| ĝ − ψ vs q_l, Δ +A | 17 / 143 | 5.3e-26 |
| ĝ − ψ vs q_l, observed | 35 / 109 | 4.9e-10 |
| U(ĝ − ψ) vs q_l, Δ +A | 61 / 77 | 0.20 |
| U(ĝ − ψ) vs q_l, observed | 71 / 60 | 0.38 |

Failure categories (`outputs/analysis/v4_goal_match_failure_classification.txt`), with the q_l runs for reference:

| Run | success | stuck | other moved | nothing moved | lifted, not at target | displaced, no lift | partial joint |
|---|---|---|---|---|---|---|---|
| q_l, observed | 163 | 35 | 54 | 59 | 59 | 117 | 13 |
| q_l, Δ +A | 163 | 80 | 44 | 60 | 44 | 82 | 27 |
| ĝ − ψ, observed | 89 | 88 | 145 | 46 | 53 | 76 | 3 |
| ĝ − ψ, Δ +A | 37 | 89 | 174 | 39 | 70 | 88 | 3 |
| U(ĝ − ψ), observed | 174 | 24 | 108 | 47 | 35 | 97 | 15 |
| U(ĝ − ψ), Δ +A | 147 | 97 | 62 | 50 | 68 | 67 | 9 |

Re-scoring the phase-8 labeled chunks offline (`tmp/analysis/goal_match_rescore.py`; the q_l rows reproduce phase 8):

| Effect | Target | P(m > 0 \| no progress or regress): failed / success / demo | P(m > 0 \| regress), failed | AUC progress vs no progress or regress: failed / success / demo |
|---|---|---|---|---|
| true | q_l | 0.63 / 0.91 / 1.00 | 0.51 | 0.72 / 0.74 / 0.71 |
| true | ĝ − ψ | 0.69 / 0.89 / 0.94 | 0.58 | 0.68 / 0.57 / 0.40 |
| true | U(ĝ − ψ) | 0.67 / 0.95 / 0.99 | 0.53 | 0.78 / 0.72 / 0.63 |
| Δ +A | q_l | 0.80 / 0.94 / 0.99 | 0.73 | 0.71 / 0.71 / 0.70 |
| Δ +A | ĝ − ψ | 0.74 / 0.91 / 0.93 | 0.60 | 0.70 / 0.55 / 0.41 |
| Δ +A | U(ĝ − ψ) | 0.77 / 0.97 / 0.99 | 0.65 | 0.77 / 0.70 / 0.60 |

Findings:

- **Full-space ĝ − ψ(o)**: success drops sharply (Δ +A 0.326 → 0.074; observed 0.326 → 0.178). The failures move to disturbing other objects (145–174 episodes), and task 9 falls from 0.96 to 0.02–0.16.
- **U-projected offset**: success is statistically unchanged against q_l.
- **Phase-8 problem not mitigated**: with true effects the share of no-progress-or-regress chunks with a positive score is 0.63 under q_l, 0.69 under ĝ − ψ and 0.67 under U(ĝ − ψ). A state-dependent direction toward the goal-head output does not remove the positive scores without progress.

## Current default: the no-ψ π's mean chunk (user decision, 2026-09-28)

By the user's decision of 2026-09-28, the default π is the no-ψ π (`docs/no_psi_baseline.md`). It takes the frozen DINOv2 tokens, proprio and the frozen SigLIP instruction tokens, and neither ψ nor q_l.

The command is the `v9_mean_nopsi_s0` one:

- strategy: `--strategy mean --candidate-source proposal --escalation 1 --effect-source none`;
- π: `--proposal-checkpoint outputs/runs/goal_proposal_nopsi_seed0/checkpoints/step_003000.pt`.

The loop still needs a stage one for its own plumbing: logged ψ, retrieval tables, the start-observation check and the scene probe. π takes nothing from it. The default uses the flow (z) stage one:

- `--checkpoint outputs/runs/goal_instruction_contrast_flow/checkpoints/step_005000.pt`;
- `--state-table outputs/state_tables/goal_instruction_contrast_flow_step5000.npy`.

Results:

- 10 × 50 benchmark inits over seeds 0 / 1 / 2 (`outputs/closed_loop/v9_mean_nopsi_s{0,1,2}`): 0.940 / 0.922 / 0.918, i.e. 0.927 ± 0.012.
- Latency: 13.2 ms per decision end to end (DINOv2 11.5, π 2.0), measured with `scripts/measure_efficiency.py`.

Rule (user, 2026-09-28): any closed-loop conclusion that compares differently trained models uses at least 3 π seeds, tested as in `docs/proposal_branch.md` (seed control) and `docs/no_psi_baseline.md`.

## Previous default: π's mean chunk with ψ and q_l (second plan, phase 13)

By the user's decision of 2026-09-27, the closed loop's default was `--strategy mean --candidate-source proposal --escalation 1 --effect-source none` with π `outputs/runs/goal_proposal/checkpoints/step_003000.pt`:

- Every decision executes π's mean chunk. The match takes no part in the decision, and no Δ is run.
- Run `outputs/closed_loop/v6_mean_default`: 10 × 50 benchmark inits, success 0.828.
- Failures: stuck 0, other moved 4, nothing moved 8, lifted not at target 54, displaced 17, partial joint 3.
- Latency: 15.5 ms per decision (encode 13.5, π 1.9).
- Details: `docs/proposal_branch.md`.

Retrieval + MATCH and the other strategies remain available for comparisons.

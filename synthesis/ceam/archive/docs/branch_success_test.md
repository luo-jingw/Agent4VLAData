# Branch test with simulator success as the label (phase 26)

At a state where executing π's mean chunk at every decision leads to failure, does the candidate that the match ranks first, executed once and followed by π's mean chunk at every later decision, reach success more often than π's mean chunk does? Success is LIBERO's goal check within the episode's 300 env steps. No demonstration or scene-based label enters.

Tool: `scripts/branch_success_test.py`, subcommands `record`, `validate`, `branch`, `report`, `summary`. Outputs: `outputs/analysis/branch_test/<name>/`, one name per ψ; the side-by-side table is `outputs/analysis/branch_test/summary/summary.md`.

## ψ versions

Each ψ uses its own π, Δ and π-mean failures.

| Name | Stage one (`outputs/runs/`, step 5000) | π (step 3000) | Δ (step 3000) | π-mean run (`outputs/closed_loop/`) | Queries |
|---|---|---|---|---|---|
| fixed | `goal_instruction_contrast` | `goal_proposal` | `goal_dynamics_perturb` | `v6_mean_default` (0.828) | q_l; v, episode mean (`goal_velocity_fixed`); v, remaining (`goal_velocity_remaining_fixed`) |
| flow | `goal_instruction_contrast_flow` | `goal_proposal_flow` | `goal_dynamics_perturb_flow` | `v7_mean_flow` (0.904) | q_l; v, remaining (`goal_velocity_remaining_flow`) |
| flow_full_sgd | `goal_instruction_contrast_flow_full_sigall_sgd` | `goal_proposal_flow_full_sgd` | `goal_dynamics_perturb_flow_full_sgd` | `v7_mean_flow_full_sgd` (0.906) | q_l; v, remaining (`goal_velocity_remaining_flow_full_sgd`) |

- q_l: the stage one's text-only query direction of the original instruction.
- v(o, l): the velocity query at the current observation with the original instruction (`docs/velocity_query.md`). All velocity queries are at step 3000. The episode-mean target is (ψ_T − ψ_0)/T; the remaining target is (ψ_T − ψ_t)/(T − t).

## Protocol

### States where π fails (`record`)

One process per task re-runs the task's π-mean closed loop under the protocol of the reference run:

- `LiberoTaskEnv` with env seed 0;
- one extra reset to init 0 before the episodes (the start check of `--check-start-observation`, which advances LIBERO's fixture RNG);
- benchmark inits 0–49 in order, so each episode is the same reset of the env as in the reference and gets the same fixture poses;
- 10 settle steps, at most 300 env steps; the scene probe reads the scene at every decision, as the reference trace did.

Every decision d executes π's mean chunk c_d: candidate 0 of a count-1 generator on the stream `NoiseStream.episode(task, 0, init, d)`, which draws no noise.

Execution is split so that every decision state can be restored:

- each chunk runs as `execute(c[:-1])`, `snapshot()`, `execute(c[-1:])`;
- the settle steps run as `reset(init, 9)`, `snapshot()`, one settle step;
- the snapshot and the sim action of the step after it are recorded; `LiberoTaskEnv.restore(snapshot, action)` executes that step and returns the observation of decision d.

Each episode is compared with the reference's result and trace, exactly: every decision's scene (object positions, joints, end effector, gripper) and gripper commands, and the final scene.

A state is used when its episode fails in the reference and in the re-run.

### Branch points and branches (`branch`)

- **Branch points**: every decision of every failed episode, 30 per episode (a failed episode runs 30 chunks of 10 steps).
- **At each state**:
  1. restore the recorded state and encode the observation;
  2. π's 32 candidates on the stream `NoiseStream.episode(task, 0, init, d)`: candidate 0 is π's mean chunk; candidates 1–31 add a constant offset (DCT coefficient 1) in the standardized action space, scales 0.05 / 0.1 / 0.2 cycled from candidate 1; no gripper flip;
  3. Δ̂ = Δ(o, a_c) of every candidate with this ψ's Δ, as the closed loop computes it, and the match cos(Δ̂, q) for each query.
- **Branch kinds**:
  - **mean**: candidate 0, π's mean chunk;
  - **top 1, top 2, top 3**: the best three by match, per query;
  - **bottom 1**: the worst by match, per query;
  - **random 1, random 2**: two distinct noisy candidates drawn uniformly from 1–31 with `numpy.random.default_rng([1, task, 0, init, d])`, independent of the scores.
- **Execution**: each distinct candidate of a state runs once, from a fresh restore: the candidate chunk, then π's mean chunk at decisions d + 1, d + 2, … until success or 300 env steps counted from the episode start. Kinds and queries that pick the same candidate share its outcome.
- **Every-candidate sweep**: a second pass (`--top-k 32`) executes all 32 candidates at decisions 1, 6, 11, 16, 21 and 26. At those states the success share of all candidates is the exact success of a uniformly random candidate, and the match's ranking of the successful candidates can be measured.
- **Later queries**: the candidates of a state are fixed by the first pass that scores it (`candidates_<pass>.npz`). A later pass restores the state, scores the stored candidates with its queries and executes only the branches not yet executed.

### Comparisons (`report`, `summary`)

- **Paired over states**: the success of one branch kind against another at the same states; exact McNemar on the discordant states; 95% interval of the difference from a bootstrap over episodes (2000 resamples), because the 30 states of an episode are not independent.
- **Paired over episodes**: whether any branch point of the episode succeeds under each kind.
- **Breakdowns**: by stage (decision number d in bins of 5; a failed episode has T = 30 decisions, so the bins are sixths of t/T), by task, by the failure category of the reference episode (`scripts/classify_closed_loop_failures.py`, default thresholds), and by the offset scale of the executed candidate.
- **Every-candidate states**: the best candidate against all candidates and against the other candidates of its offset scale; the within-state AUC of the match for success, P(match of a successful candidate > match of a failed one), ties ½, over states with both outcomes.

## Checks

### Re-run against the reference

| ψ | episodes | identical to the reference in every decision scene | reference failures | still failing in the re-run | of those, identical | outcome changes |
|---|---|---|---|---|---|---|
| fixed | 500 | 450 | 86 | 86 | 72 | 1: task 7 init 9, a reference success that fails in the re-run (first difference at decision 8); not used |
| flow | 500 | 447 | 48 | 48 | 38 | 0 |
| flow_full_sgd | 500 | 458 | 47 | 47 | 37 | 0 |

Differing episodes begin after identical scenes. As between earlier runs of the same command, they are render noise, most of them in task 7 (27 of fixed's 50).

### Split execution

`validate` on the fixed ψ, tasks 3, 4 and 7, inits 0–2. Each episode runs three times: from `reset`, and twice from one restore of the same start (split, then plain).

- **Split against plain chunk execution**, both from the restored start:
  - tasks 3 and 4: all 6 episodes identical at every decision (proprio bytes, both image digests, the chunk) and in the final MuJoCo state;
  - task 7: all 3 differ first at a wrist image, with the proprio identical (render noise, as in the reference comparison).
- **Start from `reset` against start from `restore`**:
  - the first observation is identical, but the final MuJoCo state differs in 9 of 9 episodes;
  - in 5 of 9 the observations diverge, first at decisions 8–23, by up to 0.19 in proprio later in a failing episode.
  - A restored state reproduces the decision's observation, not the whole later trajectory.

### Restores and the π-mean branch

| ψ | states | proprio identical to the recorded decision | agent / wrist image identical | π's mean chunk identical to the recorded chunk | repeated restores identical to the state's first | π-mean branch replays every later recorded observation | π-mean branch succeeds |
|---|---|---|---|---|---|---|---|
| fixed | 2580 | 2534 (max \|Δ\| 1.3e-6) | 2485 / 2320 | 2302 | 36,284 of 36,836 | 1247 | 31 (1.2%) |
| flow | 1440 | 1418 (max \|Δ\| 9.9e-5) | 1411 / 1346 | 1340 | 19,198 of 19,458 | 656 | 18 (1.3%) |
| flow_full_sgd | 1410 | 1405 (max \|Δ\| 3.7e-9) | 1382 / 1308 | 1302 | 18,200 of 18,657 | 654 | 12 (0.9%) |

The π-mean branch re-executes the failing policy from the restored state. Its successes, about 1% of states, are the rate at which restore imprecision and render noise alone turn a recorded failure into a success.

### Aborted branches and completeness

- **Aborted**: 5 branches, all task 7 (flow: init 0 decision 21, init 10 decisions 3 and 4; flow_full_sgd: init 33 decisions 4 and 11). There the wrist image's mean falls to 1.0–1.6 with the agent view normal, below `LiberoTaskEnv.BLACK_FRAME_MEAN` (2.0), and `LiberoTaskEnv.observe` refuses the observation. These branches have no outcome, and a comparison that needs one of them leaves out its state.
- **Completeness**: every branch any pass needed has an outcome or an abort, and no state and candidate was executed twice. Outcomes: fixed 36,836, flow 19,458, flow_full_sgd 18,657.

## Results

The side-by-side tables are generated in `outputs/analysis/branch_test/summary/summary.md`; per-ψ details in `<name>/report/report.md`.

### Best candidate against π's mean chunk, over states

"top 1 only / mean only" are the discordant states; p is the exact McNemar p; intervals are the 95% episode bootstrap.

| ψ | query | states | π mean | top 1 | top 1 only / mean only | McNemar p | top 1 − mean | random 1 | random 2 | bottom 1 | top 1 − random 1 (p) | top 1 − bottom 1 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | q_l | 2580 | 0.012 | 0.131 | 319 / 11 | 1.0e-79 | +0.119 [+0.101, +0.141] | 0.099 | 0.107 | 0.115 | +0.032 [+0.021, +0.044] (3.4e-7) | +0.017 [+0.000, +0.033] (0.019) |
| fixed | v, episode mean | 2580 | 0.012 | 0.128 | 309 / 10 | 5.0e-78 | +0.116 [+0.096, +0.139] | 0.099 | 0.107 | 0.119 | +0.029 [+0.017, +0.041] (6.2e-6) | +0.009 [−0.008, +0.026] (0.23) |
| fixed | v, remaining | 2580 | 0.012 | 0.126 | 305 / 11 | 1.0e-75 | +0.114 [+0.095, +0.136] | 0.099 | 0.107 | 0.122 | +0.027 [+0.016, +0.038] (1.3e-5) | +0.004 [−0.012, +0.020] (0.60) |
| flow | q_l | 1438 | 0.013 | 0.177 | 245 / 9 | 7.5e-61 | +0.164 [+0.131, +0.199] | 0.153 | 0.169 | 0.153 | +0.024 [+0.006, +0.044] (0.013) | +0.024 [+0.007, +0.042] (0.013) |
| flow | v, remaining | 1439 | 0.013 | 0.179 | 247 / 8 | 1.4e-62 | +0.166 [+0.133, +0.200] | 0.153 | 0.169 | 0.171 | +0.026 [+0.008, +0.045] (0.0042) | +0.008 [−0.014, +0.031] (0.44) |
| flow_full_sgd | q_l | 1410 | 0.009 | 0.172 | 241 / 10 | 1.3e-58 | +0.164 [+0.136, +0.194] | 0.165 | 0.157 | 0.175 | +0.008 [−0.013, +0.028] (0.44) | −0.004 [−0.026, +0.016] (0.78) |
| flow_full_sgd | v, remaining | 1410 | 0.009 | 0.173 | 236 / 4 | 1.6e-64 | +0.165 [+0.133, +0.198] | 0.165 | 0.157 | 0.172 | +0.009 [−0.012, +0.027] (0.40) | +0.000 [−0.022, +0.021] (1.0) |

- **Top 2 and top 3**: within 0.021 of top 1 for every query and ψ. Some top-3 candidate succeeds in 0.19–0.26 of the states (`report.md`).
- **π's mean chunk is never the best candidate** (0 of 5,430 states, every query): some small offset always has a higher match.
- **The best candidate has the largest offset scale**: 0.2 in 80–85% of states, 0.1 in 14–18%, 0.05 in under 3%.

### Over episodes: success at any branch point

| ψ | query | episodes | π mean | top 1 | top 1 only / mean only | McNemar p | random 1 | bottom 1 |
|---|---|---|---|---|---|---|---|---|
| fixed | q_l | 86 | 0.081 | 0.895 | 70 / 0 | 1.7e-21 | 0.814 | 0.849 |
| fixed | v, episode mean | 86 | 0.081 | 0.884 | 69 / 0 | 3.4e-21 | 0.814 | 0.860 |
| fixed | v, remaining | 86 | 0.081 | 0.884 | 69 / 0 | 3.4e-21 | 0.814 | 0.837 |
| flow | q_l | 48 | 0.125 | 0.979 | 41 / 0 | 9.1e-13 | 0.854 | 0.917 |
| flow | v, remaining | 48 | 0.125 | 0.938 | 39 / 0 | 3.6e-12 | 0.854 | 0.896 |
| flow_full_sgd | q_l | 47 | 0.106 | 0.979 | 41 / 0 | 9.1e-13 | 0.957 | 0.979 |
| flow_full_sgd | v, remaining | 47 | 0.106 | 0.957 | 40 / 0 | 1.8e-12 | 0.957 | 0.979 |

With 30 branch points per episode, nearly every failed episode has some decision where a single offset chunk leads to success, whichever kind of candidate is executed.

### Every candidate executed

Decisions 1, 6, 11, 16, 21, 26 of every failed episode (a state with an aborted candidate is left out).

| ψ | states | all candidates (a random candidate) | π's mean chunk | offset scale 0.05 / 0.1 / 0.2 | states where some candidate succeeds |
|---|---|---|---|---|---|
| fixed | 516 | 0.141 | 0.016 | 0.135 / 0.143 / 0.158 | 184 |
| flow | 287 | 0.202 | 0.017 | 0.187 / 0.214 / 0.225 | 122 |
| flow_full_sgd | 281 | 0.193 | 0.007 | 0.189 / 0.196 / 0.211 | 116 |

| ψ | query | best | best − all | best − same-scale candidates | within-state AUC |
|---|---|---|---|---|---|
| fixed | q_l | 0.190 | +0.049 [+0.030, +0.070] | +0.040 [+0.020, +0.061] | 0.542 [0.513, 0.572] |
| fixed | v, episode mean | 0.176 | +0.036 [+0.013, +0.060] | +0.019 [−0.006, +0.045] | 0.524 [0.493, 0.554] |
| fixed | v, remaining | 0.169 | +0.028 [+0.006, +0.051] | +0.014 [−0.011, +0.038] | 0.506 [0.477, 0.535] |
| flow | q_l | 0.220 | +0.018 [−0.014, +0.050] | −0.008 [−0.042, +0.027] | 0.492 [0.466, 0.521] |
| flow | v, remaining | 0.206 | +0.004 [−0.025, +0.031] | −0.019 [−0.051, +0.011] | 0.478 [0.450, 0.504] |
| flow_full_sgd | q_l | 0.196 | +0.003 [−0.027, +0.033] | −0.014 [−0.047, +0.019] | 0.474 [0.442, 0.506] |
| flow_full_sgd | v, remaining | 0.203 | +0.010 [−0.019, +0.037] | −0.009 [−0.040, +0.021] | 0.478 [0.441, 0.513] |

States with both successful and failed candidates: 184 (fixed), 122 (flow), 116 (flow_full_sgd).

### Where the best candidate succeeds

Cells: top 1 / random 1 / π mean success share (states).

**Stage** (decision bin):

| ψ | query | 1–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 |
|---|---|---|---|---|---|---|---|
| fixed | q_l | 0.50 / 0.37 / 0.02 (430) | 0.19 / 0.16 / 0.03 (430) | 0.07 / 0.05 / 0.01 (430) | 0.02 / 0.01 / 0.00 (430) | 0.00 / 0.01 / 0.00 (430) | 0.00 / 0.00 / 0.00 (430) |
| fixed | v, episode mean | 0.47 / 0.37 / 0.02 | 0.18 / 0.16 / 0.03 | 0.07 / 0.05 / 0.01 | 0.01 / 0.01 / 0.00 | 0.02 / 0.01 / 0.00 | 0.01 / 0.00 / 0.00 |
| fixed | v, remaining | 0.48 / 0.37 / 0.02 | 0.17 / 0.16 / 0.03 | 0.07 / 0.05 / 0.01 | 0.01 / 0.01 / 0.00 | 0.02 / 0.01 / 0.00 | 0.01 / 0.00 / 0.00 |
| flow | q_l | 0.58 / 0.49 / 0.02 (238) | 0.25 / 0.24 / 0.02 (240) | 0.19 / 0.16 / 0.03 (240) | 0.04 / 0.03 / 0.01 (240) | 0.02 / 0.00 / 0.00 (240) | 0.00 / 0.00 / 0.00 (240) |
| flow | v, remaining | 0.56 / 0.49 / 0.02 | 0.28 / 0.24 / 0.02 | 0.19 / 0.16 / 0.03 | 0.05 / 0.03 / 0.01 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| flow_full_sgd | q_l | 0.58 / 0.59 / 0.01 (235) | 0.29 / 0.25 / 0.00 (235) | 0.11 / 0.11 / 0.02 (235) | 0.03 / 0.03 / 0.01 (235) | 0.02 / 0.01 / 0.00 (235) | 0.00 / 0.00 / 0.00 (235) |
| flow_full_sgd | v, remaining | 0.60 / 0.59 / 0.01 | 0.28 / 0.25 / 0.00 | 0.11 / 0.11 / 0.02 | 0.03 / 0.03 / 0.01 | 0.03 / 0.01 / 0.00 | 0.00 / 0.00 / 0.00 |

Top 1 − random 1 is significant (McNemar p < 0.05) only in decisions 1–5: on the fixed ψ q_l +0.126 (p 2.1e-5), v episode mean +0.105 (p 2.8e-4), v remaining +0.114 (p 5.8e-5); on the flow ψ q_l +0.084 (p 0.045). Every other cell has p ≥ 0.06.

**Failure category of the reference episode**:

| ψ | query | lifted, not at target | displaced without lift | partial joint motion | goal object not moved, nothing moved | goal object not moved, other moved |
|---|---|---|---|---|---|---|
| fixed | q_l | 0.15 / 0.11 / 0.02 (1620) | 0.13 / 0.10 / 0.00 (510) | 0.14 / 0.10 / 0.00 (90) | 0.07 / 0.07 / 0.00 (240) | 0.03 / 0.02 / 0.00 (120) |
| fixed | v, episode mean | 0.15 / 0.11 / 0.02 | 0.11 / 0.10 / 0.00 | 0.13 / 0.10 / 0.00 | 0.08 / 0.07 / 0.00 | 0.01 / 0.02 / 0.00 |
| fixed | v, remaining | 0.14 / 0.11 / 0.02 | 0.12 / 0.10 / 0.00 | 0.13 / 0.10 / 0.00 | 0.09 / 0.07 / 0.00 | 0.00 / 0.02 / 0.00 |
| flow | q_l | 0.27 / 0.24 / 0.03 (660) | 0.12 / 0.09 / 0.00 (388) | 0.10 / 0.10 / 0.00 (60) | 0.08 / 0.06 / 0.00 (180) | 0.08 / 0.05 / 0.00 (150) |
| flow | v, remaining | 0.27 / 0.24 / 0.03 | 0.10 / 0.09 / 0.00 | 0.17 / 0.10 / 0.00 | 0.09 / 0.06 / 0.00 | 0.07 / 0.05 / 0.00 |
| flow_full_sgd | q_l | 0.23 / 0.21 / 0.02 (480) | 0.16 / 0.18 / 0.01 (570) | 0.13 / 0.10 / 0.00 (120) | 0.10 / 0.12 / 0.01 (150) | 0.08 / 0.03 / 0.00 (90) |
| flow_full_sgd | v, remaining | 0.23 / 0.21 / 0.02 | 0.16 / 0.18 / 0.01 | 0.12 / 0.10 / 0.00 | 0.13 / 0.12 / 0.01 | 0.06 / 0.03 / 0.00 |

Top 1 − random 1 is significant (p < 0.05) only on the fixed ψ: for "lifted, not at target" with every query (q_l +0.038, p 1.3e-5; v episode mean +0.039, p 5.4e-6; v remaining +0.033, p 6.3e-5), and for q_l on "displaced without lift" (+0.029, p 0.032). On flow and flow_full_sgd every cell has p ≥ 0.067.

**Task** (dataset task index; fixed has no failed episode in task 9, flow_full_sgd none in tasks 0 and 4):

| ψ | query | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | q_l | 0.20 / 0.18 / 0.00 (60) | 0.22 / 0.18 / 0.04 (330) | 0.20 / 0.17 / 0.01 (90) | 0.09 / 0.06 / 0.00 (390) | 0.08 / 0.04 / 0.00 (600) | 0.09 / 0.06 / 0.00 (210) | 0.16 / 0.12 / 0.01 (570) | 0.14 / 0.13 / 0.04 (300) | 0.00 / 0.00 / 0.00 (30) | – |
| flow | q_l | 0.13 / 0.20 / 0.00 (30) | 0.26 / 0.24 / 0.03 (510) | 0.13 / 0.10 / 0.00 (30) | 0.28 / 0.23 / 0.00 (60) | 0.03 / 0.00 / 0.00 (30) | 0.08 / 0.10 / 0.00 (60) | 0.10 / 0.05 / 0.00 (420) | 0.24 / 0.21 / 0.01 (148) | 0.11 / 0.11 / 0.00 (150) | – |
| flow_full_sgd | q_l | – | 0.21 / 0.18 / 0.01 (390) | 0.27 / 0.27 / 0.01 (150) | 0.10 / 0.13 / 0.00 (30) | – | 0.11 / 0.09 / 0.01 (180) | 0.16 / 0.15 / 0.02 (300) | 0.15 / 0.15 / 0.01 (240) | 0.12 / 0.16 / 0.00 (90) | 0.13 / 0.07 / 0.00 (30) |

The v rows are within 0.05 of the q_l rows in every task. Top 1 − random 1 is significant (p < 0.05) on the fixed ψ in tasks 3 and 4 with every query (+0.027 to +0.044) and in task 6 with q_l and v remaining (+0.033, +0.039); on the flow ψ in task 6 with both queries (+0.045, +0.036) and in task 1 with v remaining (+0.041). On flow_full_sgd every cell has p ≥ 0.18.

## Reading

- **Leaving π's mean chunk once rescues many failures; which candidate matters little.** At π's error states, the match's best candidate followed by π succeeds in 0.13 (fixed) and 0.17–0.18 (flow, flow_full_sgd) of the states, against about 0.01 for π's mean chunk, for every query and ψ (p ≤ 1.3e-58). A random small offset succeeds in 0.10–0.17, and the match's worst candidate in 0.12–0.18.
- **The match's own contribution is small and limited to the fixed ψ.**
  - Over a random offset, the best candidate gains +0.03 on the fixed ψ, +0.02–0.03 on flow and nothing on flow_full_sgd.
  - Part of that is the best candidate's preference for the largest offset, which succeeds more often. Against candidates of the same scale, only q_l on the fixed ψ keeps an advantage: +0.040, within-state AUC 0.54 [0.51, 0.57].
  - The velocity queries, with the episode-mean or the remaining target, rank the successful candidates no better than chance on any ψ (AUC 0.47–0.52). So does q_l on the flow and flow_full_sgd ψ.
- **Where the best candidate beats π's mean chunk**:
  - **Stage**: early. In decisions 1–5 about half of the states are rescued; after decision 15 almost none (≤ 0.05). π's failures are fragile in the first third of the episode and locked in after the middle.
  - **Failure type**: every category, most often "lifted, not at target", the most common failure type. "Goal object not moved, other moved" is rescued least.
  - **Task**: every task with failures; fixed task 8, a single failed episode, only with v remaining (1 of 30 states).
  - **Common to random offsets**: the same states are rescued by random offsets. For fixed q_l, the best and random 1 both succeed in 166 states; the best alone in 173, random 1 alone in 90.
  - **Match-specific part**: the gain over random sits in decisions 1–5 on the fixed ψ (and flow q_l), in the fixed ψ's "lifted, not at target" episodes and in its tasks 3, 4 and 6 (bowl on the plate, bowl on the stove, cream cheese in the bowl).
- **Answer to the offline question.** With simulator success as the label, the match improves on π at π's error states mainly by selecting some deviation from π's mean chunk. Among π's small-offset candidates it identifies the rescuing ones only weakly: with q_l on the fixed ψ, and not with the velocity queries or on the flow ψ versions.

## Runs

- **Hosts**: L40S GPUs on node-64 and node-69. Record and validate ran on node-64. The fixed branch passes ran on node-69. flow ran on node-69; flow_full_sgd on node-64, with its last sweep episodes on node-69.
- **Load**: 3 processes per GPU; one GPU of node-69 ran 4 for the last 9 minutes.
- **Code**: commits `0b75c03`–`3c8459b` of this page's branch. Recording, scoring and branch execution are the same in all of them. Later commits changed the report, loaded the velocity queries through `VelocityCheckpointStore` (same weights) and added the abort record.
- **Recovery passes**: each re-scores its episodes from the stored candidates and executes only the missing branches. They covered:
  - shards that stopped at a refused wrist observation before the abort record existed (flow `p1r`, `p1r2`; flow_full_sgd `p1r2`);
  - shards that stopped on a GPU fault (flow_full_sgd `sweep_r2`).
- **Throughput**: a 300-step branch takes about 5.5 s with rendering, a state with about 10 branches about 40 s. The fixed main pass (25,382 branches, 18 processes) took 74 min.

## Using the tool

Every command below runs after `source scripts/ceam_env.sh`, with `CEAM_ROOT` set to the main checkout and `--code-commit` giving the commit of the code:

```bash
# states: one process per task, benchmark inits 0–49 in order, against the ψ's π-mean run
$CEAM_PYTHON scripts/branch_success_test.py record --name <name> --cuda-device <gpu> \
  --stage-one-checkpoint <ψ> --proposal-checkpoint <π> --task-index <t> \
  --reference-dir outputs/closed_loop/<π-mean run>/task<t>_mean --code-commit <sha>
# branches: n shards over the failed episodes
$CEAM_PYTHON scripts/branch_success_test.py branch --name <name> --pass-name <pass> --shard <i> --shards <n> \
  --cuda-device <gpu> --stage-one-checkpoint <ψ> --proposal-checkpoint <π> --dynamics-checkpoint <Δ> \
  --velocity-checkpoints <v …> --velocity-names <names …> --code-commit <sha>
# every candidate at decisions 1, 6, …, 26: the same with --top-k 32 --decision-stride 5
$CEAM_PYTHON scripts/branch_success_test.py report --name <name>
$CEAM_PYTHON scripts/branch_success_test.py summary --names fixed flow flow_full_sgd --out summary
```

A new query on an existing name is a new `branch` pass that lists every query. It reuses the stored candidates and the executed branches, and runs only the branches its query adds. A new ψ is a new name, with its own `record` from its own π-mean run.

## Outputs

`outputs/analysis/branch_test/`:

| Path | Contents |
|---|---|
| `<name>/record/task<t>/` | `decisions.npz` (snapshots, producing actions, observations' proprio, ψ, image digests, chunks), `episodes.jsonl` (against the reference), `config.json` |
| `<name>/validate/task<t>.jsonl` | split against plain execution (fixed: tasks 3, 4, 7) |
| `<name>/branch/task<t>_init<i>/` | `states_<pass>.jsonl` (restore checks, scores, branch candidates), `outcomes_<pass>.jsonl`, `candidates_<pass>.npz`, `aborted_<pass>.jsonl` |
| `<name>/branch/pass_<pass>/` | per shard: settings, episodes, provenance, and the completion record |
| `<name>/report/` | `report.md`, `comparisons.csv`, `sweep.csv`, `targets.json` |
| `summary/summary.md` | query × ψ, paired against π's mean chunk; by stage, failure category and task; every-candidate states |

`smoke_p26` and `smoke_summary` hold the development checks of the tool.

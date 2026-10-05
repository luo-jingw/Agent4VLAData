# Focused within-start experiment on π's failure states (LIBERO-Goal)

The experiment tests whether the query q(o, l), trained within one observation on simulator outcomes, chooses better than π at states where π's mean chunk fails. At each start state, 8 candidate chunks are executed. After each candidate, the same π's mean chunk continues until the record ends. The candidates are then compared by whether the start's task is reached. r is fixed to R6_all.

| Component | Framework clause (`docs/framework.md`) | Implementation |
|---|---|---|
| r(o, a) | 1 (action representation, task-free) | R6_all, frozen: `outputs/runs/interaction_frozen_pca_delta_all_stores_seed0/checkpoints/step_020000.pt`, PCA `outputs/feature_pca/r6_all_stores/pca.npz` |
| q(o, l), m = s · cos(r, q) | 1, 3 | `QueryNetwork`, `Match`, `TaskEncoder`, as in `docs/query_match.md` |
| task-side non-collapse | 2 (query side) | SIGReg on e(l) of the 10 tasks (`SchemeAObjective`) |
| within-start contrast | 3 (within one observation: consistent chunks accepted, other reachable chunks distinguished) | `SchemeAObjective` on the positives and negatives of one start and its task |
| candidate sampler for the data | 4 (replaceable sampler) | π's noisy offsets, other-task demonstration chunks, smooth random chunks |
| selection in the closed loop | 4 | argmax of m over π's 32 candidates (the `v12` / `p23` protocol) |
| labels | final criterion (simulator success in branch tests) | LIBERO's success check of the start's task |

Engineering choices, not framework requirements: π's mean chunk as the continuation that defines a label, the step limit that ends a record, the choice of decisions 1–15, and the candidate proportions.

## Protocol

- **Starts.**
  - The starts are states of π-mean closed-loop episodes on TRAIN benchmark inits 0–39 that fail. π is the no-ψ π, seeds 0/1/2, under the protocol of `v9_mean_nopsi_s<k>`.
  - The episodes are re-run with a restore recipe at every decision: the snapshot before the chunk's last step, and that step's sim action.
  - Every failed episode contributes its decisions 1–15, the states at env steps 0, 10, …, 140.
  - HELDOUT inits 40–49 and the Phase-26 sweep states never enter training.
- **Candidates**, 8 per start, all at the same observation:
  - 4 π noisy chunks: `ProposalCandidates` offsets of the start's π seed under the start task's original instruction, scales 0.05 / 0.1 / 0.2 / 0.05, DCT J = 1, no gripper flip. π's mean chunk is not among the 8.
  - 2 other-task demonstration chunks: a random 2 of the 64 nearest other-task TRAIN observations by the frozen key of the rendered start (`outputs/iterative/start_pool/train_keys.npz`, the phase-22 construction).
  - 2 smooth random chunks: `SmoothActionSampler`, J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, gripper switch 0.3 from the command in force.
- **Continuation and record end.**
  - Each candidate runs its full 10 steps.
  - Then the start's own π seed runs its mean chunk at every later decision.
  - The record ends at the start task's success (LIBERO's done) or at the original episode's step limit, whichever comes first. The limit is 300 env steps from the episode start, so 300 − 10 (d − 1) steps remain after decision d.
- **Labels**, for the start's task only:
  - positive: the task is reached before the record ends;
  - negative: it is not reached.
  - A branch stopped by a black render has no label.
- **q.** q is trained by scheme A (`ceam/losses/scheme_a_objective.py`) on the (start, task) sets that hold a positive and a negative. r stays frozen.
- **Evaluation (a).**
  - States: HELDOUT inits 40–49 failure episodes, decisions 1–15.
  - At each state the same 8-candidate mix is drawn, and π's mean chunk is added as a 9th branch.
  - The continuation success of three choices is compared, paired by state: q's argmax among the 8 (q seeds 0/1/2, read from the same branches), π's mean chunk, and a seeded uniform draw among the same 8. The exact mean over the 8 is reported with the draw.
- **Evaluation (b).**
  - The π-candidate closed loop, 10 tasks × 50 benchmark inits, seeds 0/1/2: q seed k with π seed k.
  - It follows the protocol of `p23_final_argmax_s<k>` (`tmp/s5p22/closed_loop.sh`).
  - It is paired by (task, init) with `v9_mean_nopsi_s<k>`, `v11_qm_random_s<k>` and `p23_round0_argmax_s<k>`, which exist.

## Estimate

All numbers below are computed from existing data by `tmp/s6f1/f1_estimate.py`. Its output is `tmp/s6f1/out/f1_estimate.{txt,json}`. The script ran read-only on node-62, CPU only, 2026-09-29 15:00–15:11 EDT, with code `7edc4b6`.

**Data used**

| Source | Content | Use here |
|---|---|---|
| `outputs/closed_loop/v9_mean_nopsi_s{0,1,2}` | no-ψ π-mean closed loops, 10 × inits 0–49, results and traces | failure episodes per task and split, failure categories, scene at each decision |
| Phase-26 branch outcomes `outputs/analysis/branch_test/{fixed,flow,flow_full_sgd}/branch` | ψ-π failure states. Every decision has the mean chunk, top / bottom / two uniform noisy candidates (pass `p1` and recoveries). Decisions 1, 6, …, 26 have all 32 candidates (the sweep, 1,084 states). The continuation is the ψ π's mean chunk up to the episode's step limit. | state features that predict mixed outcomes, per-decision rates, branch timing |
| Phase-22 records `outputs/iterative/round_{1..4}/records` with the start pool | 1,000 single-candidate branches of four kinds. The continuation is the no-ψ π seed 0 mean chunk, up to 300 steps from the decision point. | success odds of each candidate kind, steps to success, record timing |

**Main assumption.** The analysis sources are ψ-π failure states continued by the ψ π, and here they stand in for no-ψ-π failure states continued by the no-ψ π. The failure-category mix is similar (below), but no data measures the no-ψ π at its own failure states. The collection measures the realized share directly (section 3).

### 1. Failure episodes and states

**No-ψ π-mean failures** (`v9_mean_nopsi`). A failed episode runs the full 300 env steps, 30 decisions, in every case.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN inits 0–39, π seed 0 | 2 | 6 | 2 | 1 | 1 | 1 | 3 | 5 | 0 | 1 | 22 |
| TRAIN, seed 1 | 4 | 5 | 1 | 0 | 0 | 2 | 7 | 4 | 5 | 0 | 28 |
| TRAIN, seed 2 | 6 | 8 | 0 | 1 | 2 | 2 | 5 | 6 | 6 | 0 | 36 |
| TRAIN, seeds 0/1/2 | 12 | 19 | 3 | 2 | 3 | 5 | 15 | 15 | 11 | 1 | **86** |
| HELDOUT inits 40–49, seeds 0/1/2 | 1 | 8 | 1 | 0 | 0 | 1 | 6 | 4 | 3 | 0 | **24** (8 / 11 / 5) |

- **Failure categories**, TRAIN, 3 seeds: lifted, not at target 37; displaced without lift 23; goal object not moved, nothing moved 10; partial joint motion 10; goal object not moved, other moved 6.
- **Scene at decisions 1–12** of these episodes (1,032 states):
  - nothing grasped and the goal object not lifted: 947;
  - grasped and lifted: 42;
  - lifted without a grasp: 27;
  - grasped, not lifted: 16.

**Restorability.** No existing record restores these states.
- The `v9` traces hold the scene probe of every decision, not the simulator state.
- `branch_success_test.py record`, which produced the Phase-26 states, requires a stage one and a π trained on it. It does not load the no-ψ π, whose stage one is null.
- The resim stores and the Phase-26 records belong to the ψ π.
- Step E below therefore re-runs the 1,500 `v9` episodes with a new recorder that keeps a one-step restore recipe at every decision, as phase 26 and phase 22 do.
- Phase 26's re-run of its reference was identical in every decision scene in 90–92% of episodes, and changed the outcome of 1 of 1,500. Its failures are used as recorded; their count can differ slightly from the 86 / 24 above.

**Record end.** The record ends at the original episode's step limit. Two measurements support this choice:
- Phase 26, the source of the estimate, used it.
- The closed loop counts success within that limit.

The alternative, 300 steps from the decision point as in phase 22, changes few labels for decisions ≤ 15:
- after a candidate at π closed-loop starts, 94% of phase-22 successes came within 150 steps of the decision point, and 98% within 190;
- the median is 54 steps and p90 is 127 steps.

**More starts.** Starts can be added from TRAIN-side π-mean episodes on sampled starts of a fresh env seed (`--start sampled`). These are new scenes, disjoint from benchmark inits 40–49. The optional extension is sized in section 4.

### 2. Which states give a mixed group

**By position** (sweep, 8 of the 31 noisy candidates drawn without replacement, exact). In a failed episode, position and remaining budget are the same variable, because every failed episode runs 300 steps.

| Decision (env steps before) | 1 (0) | 6 (50) | 11 (100) | 16 (150) | 21 (200) | 26 (250) |
|---|---|---|---|---|---|---|
| states | 181 | 181 | 180 | 181 | 180 | 181 |
| per-candidate success | 0.575 | 0.284 | 0.128 | 0.046 | 0.013 | 0.004 |
| states with any success | 1.000 | 0.724 | 0.317 | 0.188 | 0.072 | 0.028 |
| P(mixed), 8 π noisy | **0.830** | **0.554** | **0.244** | **0.127** | **0.046** | **0.019** |

Every decision 1–30 is covered by the two uniform noisy candidates of pass `p1` (181 episodes per decision). There, P(the two differ) × the sweep's ratio P8 / P2 (2.29–2.87) gives P8 at each decision. It reproduces the sweep at its six decisions (0.835 / 0.494 / 0.233 / 0.123 / 0.028 / 0.016):

| d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16–20 | 21–30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P8 | 0.84 | 0.77 | 0.81 | 0.84 | 0.66 | 0.49 | 0.52 | 0.33 | 0.29 | 0.23 | 0.23 | 0.23 | 0.25 | 0.18 | 0.12 | 0.00–0.12 | 0.00–0.04 |

**Other features** at decisions 1, 6 and 11 (P(mixed), 8 π noisy; states in parentheses):

| Feature | Values |
|---|---|
| task | 0: 0.39 (9); 1: 0.74 (123); 2: 0.70 (27); 3: 0.39 (48); 4: 0.41 (63); 5: 0.47 (45); 6: 0.55 (129); 7: 0.55 (68); 8: 0.27 (27); 9: 0.00 (3); all 0.54 |
| failure category of the episode | lifted, not at target 0.60 (275); displaced without lift 0.56 (147); partial joint motion 0.46 (27); nothing moved 0.44 (57); other object moved 0.30 (36) |
| scene at the state | nothing grasped, not lifted 0.59 (452); grasped, not lifted 0.56 (24); grasped and lifted 0.39 (24); lifted without a grasp 0.16 (42) |
| offset scale (per-candidate success, all sweep states) | 0.05: 0.163; 0.1: 0.175; 0.2: 0.189 |

- Reweighting the category rows by the no-ψ TRAIN failure categories moves 0.544 to 0.532.
- 92% of the no-ψ candidate states fall in the scene class with the highest rate.
- Task, category and scene therefore add little to a position rule. The task differences rest on few states for tasks 0, 2, 8 and 9.

**Candidate type** (phase-22 records at π closed-loop starts: 479 records, no-ψ continuation). Own-task success odds relative to π's small offsets, Mantel–Haenszel over start group × decision bin:

| Kind | π small (0.05–0.2) | π large (0.5–2) | other-task demo | smooth random |
|---|---|---|---|---|
| odds ratio | 1 | 0.25 | 0.18 | 0.15 |
| success at failed ψ-π starts (n) | 0.63 (8) | 0.13 (16) | 0.00 (7) | 0.18 (17) |
| success at pi-noise starts (n) | 0.68 (22) | 0.33 (67) | 0.21 (42) | 0.18 (120) |

- The other kinds are chosen by the phase-22 selections, not uniformly.
- Of the 10 failed-ψ-π starts executed twice, none had mixed outcomes.

**The mix.** At each sweep state the rate of π's small offsets is θ = (k + 0.5) / 32. The other kinds shift its log-odds by the ratios above, and the 8 candidates are taken as independent. The mix changes P(mixed) against 8 π noisy by these factors:

| Mix | d 1 | d 6 | d 11 | d 16 | d 21 | d 26 |
|---|---|---|---|---|---|---|
| **4 π small + 2 other + 2 smooth (proposed)** | 1.04 | 0.92 | 0.86 | 0.74 | 0.66 | 0.62 |
| 3 π small + 3 other + 2 smooth | 1.02 | 0.88 | 0.81 | 0.67 | 0.57 | 0.52 |
| 2 π small + 2 π large + 2 other + 2 smooth | 1.00 | 0.83 | 0.76 | 0.60 | 0.49 | 0.44 |

- Positives come mostly from π's offsets.
- The proposed mix keeps 4 of them, because the closed loop chooses among π's offsets.

**Rule and share of mixed starts** (proposed mix, per-decision P8 × factor). **Rule: decisions 1–15 of every failed episode**, with no further filter.

| Decisions used | 1–5 | 1–8 | 1–10 | 1–12 | **1–15** | 1–20 | 1–30 (all) |
|---|---|---|---|---|---|---|---|
| mixed groups per failed episode | 4.08 | 5.32 | 5.80 | 6.20 | **6.67** | 6.86 | 6.96 |
| share of starts with ≥ 1 positive and ≥ 1 negative | 0.82 | 0.67 | 0.58 | 0.52 | **0.44** | 0.34 | 0.23 |
| of these, the 4 π offsets alone mixed, per episode | 3.06 | 4.07 | 4.46 | 4.82 | 5.24 | 5.43 | 5.53 |

- Decisions 1–15 give 96% of the groups that all 30 decisions give, from half the starts.
- Beyond decision 15, a start yields ≤ 0.12.

### 3. Start count

**Proposed: 1,290 starts.** These are the 86 TRAIN failure episodes of π seeds 0/1/2 × decisions 1–15, each start continued by its own π seed. The expected number of mixed groups is **≈ 570**. With π seed 0 alone: 22 episodes, 330 starts, ≈ 150 groups.

| Task | Failed episodes, seeds 0/1/2 (seed 0) | Starts | Expected mixed groups, 3 seeds (seed 0) | Sweep task factor (states) |
|---|---|---|---|---|
| 0 | 12 (2) | 180 | 80 (13) | 0.71 (9) |
| 1 | 19 (6) | 285 | 127 (40) | 1.37 (123) |
| 2 | 3 (2) | 45 | 20 (13) | 1.28 (27) |
| 3 | 2 (1) | 30 | 13 (7) | 0.72 (48) |
| 4 | 3 (1) | 45 | 20 (7) | 0.76 (63) |
| 5 | 5 (1) | 75 | 33 (7) | 0.86 (45) |
| 6 | 15 (3) | 225 | 100 (20) | 1.01 (129) |
| 7 | 15 (5) | 225 | 100 (33) | 1.01 (68) |
| 8 | 11 (0) | 165 | 73 (0) | 0.50 (27) |
| 9 | 1 (1) | 15 | 7 (7) | – (3) |
| all | 86 (22) | 1,290 | **574 (147)** | |

- The expected groups use the pooled per-decision curve. The task factor (the task's sweep P(mixed) at decisions 1/6/11 over the pooled 0.544) shows the spread between tasks and is not applied.
- **Short tasks:**
  - 9 (7 groups) and 3 (13) are short;
  - 2 and 4 (20 each) are borderline, and 5 has 33.
  - These tasks rarely fail under π's mean chunk (1–6 failures in 150 episodes).
- **Sensitivity** (mixed groups per episode at decisions 1–15; total for 86 episodes):
  - every candidate's success odds at a state × 0.25 / 0.5 / 2: 4.94 / 5.91 / 7.21, totals 425 / 508 / 620;
  - the no-ψ π's failures locked in twice as early (decision d taking the rate of decision 2d − 1): about 3.8 per episode, about 330 in total.
  - The ≥ 300 target holds in each case with the three seeds, and in none with seed 0 alone.
- **Per-task floor, optional extension.** π-mean episodes on sampled starts of a fresh env seed:
  - Size: starts 0–299 for tasks 2, 3, 4 and 9, and 0–99 for task 5; π seeds 0/1/2; 3,900 episodes.
  - Expected new failures, at the benchmark failure rates of these tasks: 24 / 12 / 18 / 12 / 6.
  - The first failed episodes are used until a task has 8, about 25 episodes and 375 starts.
  - This lifts tasks 2, 3, 4 and 5 to ≈ 53 groups each. Task 9 likely stays short (≈ 47 if its rate of 1 in 150 holds).
- **Check after collection.** The measured share of mixed starts replaces the 0.44 above. The extension is run only if the count or a task floor is missed.

### 4. Cost

**Timing basis** (measured):

| Measurement | Value | Source |
|---|---|---|
| branch execution, per simulator process | 19.46 ms per env step + 0.17 s per branch, 5 processes per GPU | phase-22 records (1,000, node-61); phase 26 gives 19.61 ms per step at 3 per GPU (74,951 branches) |
| executed steps per branch, proposed mix, decisions 1–15, episode-limit end | 206 (1,645 per start of 8; 1,875 with π's mean chunk as a 9th branch) | per-decision executed steps of phase 26's uniform noisy branches; other kinds by their odds and the remaining budget |
| π-mean closed loop, 50 episodes per task process | 1.6–4.5 min, 10 processes per GPU | `docs/no_psi_baseline.md` |
| π-candidate closed loop, 10 × 50, one seed | 4.6 min on 2 GPUs (10 processes) | phase 23, 07:03:54–07:08:31 |
| scheme-1 q on R6_all, 5,000 updates | 2,904 s alone on one GPU | round 0, seed 1 |

**Steps**, on 2 nodes × 2 GPUs, 5 simulator processes per GPU (20 in all). GPU·h counts the GPUs held × wall time.

| Step | Work | Wall | GPU·h |
|---|---|---|---|
| E. record π-mean episodes with restore recipes | 1,500 episodes (10 tasks × inits 0–49 × π seeds 0/1/2), 30 task processes | ≈ 6 min | 0.4 |
| C. training collection | 1,290 starts × 8 branches; 34 s process time per start | ≈ 40 min | 2.7 |
| L. labels and groups | CPU | ≈ 3 min | – |
| R. render and encode the start frames | ≈ 1,650 frames | ≈ 2 min | 0.1 |
| Q. q, seeds 0/1/2, scheme A, 5,000 updates of 64 sets | one GPU per seed, in parallel | 15–50 min | 0.8–2.5 |
| A. evaluation (a): branches | 360 states (24 HELDOUT episodes × decisions 1–15) × 9 branches; 39 s per state; runs during Q, as it needs no q | ≈ 14 min | 0.9 |
| A. evaluation (a): scoring and analysis | 360 × 8 candidates × 3 q | ≈ 3 min | 0.1 |
| B. evaluation (b) | 3 × 500 episodes | ≈ 10 min | 0.5 |
| **total** | | **≈ 1.5–2 h** | **≈ 5.5–7** |
| optional extension (section 3) | 3,900 π-mean episodes + 375 starts × 8 branches | ≈ 25 min | 1.3 |

- **Q timing.** The upper bound is the scheme-1 rate on R6_all (48 min alone). Scheme A computes r for 8 chunks per set instead of 33 and draws no candidates. 15 min is estimated from that and is not measured.
- **Implementation before step E** (no GPU; estimated 3–5 h with smoke tests):
  - a no-ψ π-mean recorder;
  - the 8-branch executor with compact records;
  - the labels and contrast sets of this protocol, own task only, positive or negative;
  - the evaluation (a) readout.

**q training** (proposal; see the open points):
- Scheme A (`SchemeATrainer`, `ceam/losses/scheme_a_objective.py`): contrast sets = (start, start task) with ≥ 1 positive and ≥ 1 negative, about 570.
- Settings as in phase 21: 64 sets per update, 5,000 updates, lr 3e-4, warmup 150, cosine, weight decay 0.05, clip 1.0, bf16, match clamp 1000.
- At 570 sets that is about 560 draws of each set.
- q starts from the round-0 q (`outputs/runs/query_match_r6all_seed<k>/checkpoints/step_005000.pt`), which was trained on R6_all. This is phase 21's default for scheme A.
- No demonstration tables are mixed in.
- r stays R6_all, frozen.

**Evaluation (a) size.**
- 360 states, of which ≈ 160 have mixed outcomes (24 × 6.67); only there can the choice matter.
- Phase 26's episode bootstrap of best − random over 2,580 states in 86 episodes had half-width 0.012. Scaled to 360 states in 24 episodes, the half-width is about 0.02–0.03 over all states.
- The per-task split is not usable: tasks 3, 4 and 9 have no HELDOUT failure.
- A free secondary readout: rescoring the 1,084 sweep states' stored π candidates with the new q (`rescore_branch_candidates.py score --query-match`, as for the R6 q in `docs/query_match.md`). It takes minutes on one GPU. These states are π-candidate-only and continued by the ψ π.

**Disk** (compact records only: sim states, restore recipes, chunks, goal vectors, labels):

| Item | Size |
|---|---|
| training records, 10,320 branches at ≈ 50 KB (phase 22: 36 KB per branch at 141 steps) | ≈ 0.5 GB (another 0.5 GB if a staging copy is kept, as in phase 22) |
| evaluation (a) records, 3,240 branches | ≈ 0.2 GB |
| π-mean episode records, 1,500 episodes | ≈ 0.04 GB |
| rendered start frames, ≈ 1,650 at ≈ 0.19 MB (`pi_noise_nopsi`: 5.73 GiB for 31,000 frames) | ≈ 0.3 GB |
| q checkpoints, 3 seeds × 5 at 56 MB | ≈ 0.8 GB |
| closed loops, 3 runs | ≈ 0.03 GB |
| **total** | **≈ 2–2.5 GB** |

### 5. Open points

1. **Which π seeds give the failure states and continue them.**
   - Proposed: seeds 0, 1 and 2, each start continued by the π whose episode failed there. This gives 86 episodes and ≈ 570 groups.
   - With seed 0 alone: 22 episodes and ≈ 150 groups, below 300 without about 25 more failed episodes from sampled starts.
   - A related question is whether the sampled-start extension, TRAIN-side scenes of a fresh env seed, may fill tasks 2, 3, 4, 5 and 9.
2. **Other-task demonstration chunks.**
   - Proposed: strictly other tasks, the phase-22 construction.
   - Same-task chunks from other demonstrations would be the candidates most likely to complete the task. Their rate at failure states has not been measured.
   - Other-task chunks succeed with odds 0.18 of π's small offsets, so they mostly supply negatives.
3. **q initialization and data.**
   - (A) the round-0 q seed k, then scheme A on these sets only (proposed);
   - (B) a fresh q on these sets only, about 570 sets for a 13 M-parameter network;
   - (C) the round-0 q with the demonstration tables mixed into the updates. This needs a trainer that mixes scheme 1 and scheme A, which does not exist.

## Implemented mechanism

Phase 2 implements the collection boundary without changing `ActionRepresentation`, `QueryNetwork`, `Match` or
`SchemeATrainer`:

- `FocusedStartRecords` stores typed one-step restore recipes, split identity and the expected proprio in a
  pickle-free NPZ. Fixture poses are flat with offsets, so records from tasks with different fixture counts remain
  explicit.
- `FocusedCandidateSet` requires exactly four π small-offset chunks, two other-task demonstration chunks and two
  smooth chunks, in that order. Kind, offset scale and demonstration source are stored beside every chunk.
- `ContinuationBranchExecutor` restores every candidate from the same recorded start, executes the full candidate
  chunk, then continues with the π seed that produced that start until task success or the original episode limit.
  A black render is an unlabelled `render_abort` outcome.
- `FocusedContrastBuilder` uses labelled TRAIN branches only. It assigns a label only to the start's own task and
  emits `GroupMembers` and `ContrastSets` only when that task has both a positive and a negative branch. HELDOUT
  branches cannot enter scheme-A training.
- `FocusedQueryMetrics` reports all-candidate, π-only and the independently executed π-mean baseline, seeded and
  exact-uniform baselines, oracle availability, source/scale strata, pre/post-q changes and episode-cluster bootstrap
  intervals.
- `scripts/run_focused_query.py` owns recording, branch collection, contrast construction, diagnostics and store
  inspection. `configs/query_match/focused_seed{0,1,2}.json` fix R6_all, initialize from the corresponding round-0 q,
  use scheme A for 5,000 updates and clamp match scale at 1,000.

### Validation

The CPU synthetic smoke covered two starts with different split identities, all eight candidate roles, same-policy
continuation, mixed-label contrast construction, NPZ round-trip equality and checksum preservation, and every
diagnostic path. It produced `[2, 8, 10, 7]` float32 chunks, zero restore error and one TRAIN contrast set; the
HELDOUT start was excluded from training.

The simulator smoke ran on node-62 CUDA 0 on 2026-09-30 EDT:

| Measurement | Observed value |
|---|---:|
| task / π seed | task 1 / seed 0 |
| π-mean episodes replayed | 50 |
| reference / reproduced failures | 8 / 8 |
| episode outcome mismatches | 0 |
| retained failure episodes | TRAIN init 2, HELDOUT init 47 |
| retained starts | 30, decisions 1–15 of each episode |
| smoke starts branched | 2, decision 1 of each retained episode |
| branches | 16 |
| candidate success labels | 13 positive, 3 negative |
| render aborts | 0 |
| maximum proprio restore error | 0 |
| TRAIN contrast set | 1 set, 6 positive and 2 negative members |
| branch-store checksum | `64a8240ecbf0c847cf2d40296d781e61d8a925faf4803ab0c71ad8a48fff7ec9` |
| recorder wall time | 215.0 s |
| branch collection wall time | 71.9 s |
| branch collection peak allocated GPU memory | 1,217,854,976 bytes |
| GPU allocation | node-62 CUDA 0 only; approximately 0.080 GPU·h total |

The branch store reload preserved every array exactly, including shapes, dtypes and the checksum. The TRAIN start had
six successful candidates and the HELDOUT start had seven; both therefore exercised the mixed-outcome path rather than
only the serialization path. Smoke artifacts are under `tmp/phase2_real_smoke/` and are not experiment results.

## Full branch collection

The three no-ψ π seeds replayed all 1,500 benchmark episodes. An episode supplies starts only when both the reference
run and the replay fail. Replay produced 111 failures against 110 reference failures, with 9 episode outcome
mismatches; their intersection contains 106 failure episodes. Decisions 1–15 of those episodes produced 1,590 starts.
The recorder writes an explicit report with zero starts and a null checksum when a task-seed shard has no retained
failure, and does not create an empty start store.

| Task | Reference failures | Replayed failures | Outcome mismatches | Retained TRAIN episodes | Retained HELDOUT episodes | TRAIN starts | HELDOUT starts | TRAIN mixed groups | HELDOUT mixed groups |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 | 13 | 0 | 12 | 1 | 180 | 15 | 157 | 12 |
| 1 | 27 | 26 | 1 | 18 | 8 | 270 | 120 | 214 | 109 |
| 2 | 4 | 4 | 0 | 3 | 1 | 45 | 15 | 40 | 13 |
| 3 | 2 | 3 | 1 | 2 | 0 | 30 | 0 | 20 | 0 |
| 4 | 3 | 3 | 0 | 3 | 0 | 45 | 0 | 20 | 0 |
| 5 | 6 | 6 | 0 | 5 | 1 | 75 | 15 | 30 | 9 |
| 6 | 21 | 21 | 0 | 15 | 6 | 225 | 90 | 148 | 53 |
| 7 | 19 | 21 | 6 | 13 | 4 | 195 | 60 | 130 | 37 |
| 8 | 14 | 14 | 0 | 11 | 3 | 165 | 45 | 53 | 19 |
| 9 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **All** | **110** | **111** | **9** | **82** | **24** | **1,230** | **360** | **812** | **252** |

The 1,590 starts produced 12,720 branches: 4,249 positive labels, 8,470 labelled negatives and one unlabelled render
abort. Maximum proprio restore error was `6.693881005048752e-09`. TRAIN contributes 9,839 labelled members and 812
mixed groups, a measured mixed-start share of 0.6602. HELDOUT has 252 mixed starts, a share of 0.7000, and remains
excluded from training. The TRAIN total exceeds the 300-group gate, so the sampled-start extension is not run.

The 44 branch shards reload with their recorded checksums. Their start identities are unique, and the canonical
contrast arrays contain 9,839 `GroupMembers` rows and 812 `ContrastSets` rows with width 8. Every contrast set has at
least one positive and one negative, every member index is in range, and padding uses member `-1` with role `0`.

| Artifact | Location / checksum |
|---|---|
| start stores and reports | `outputs/focused_query/starts/`, `outputs/focused_query/reports/record/` |
| branch stores and reports | `outputs/focused_query/branches/`, `outputs/focused_query/reports/collect/` |
| contrast arrays | `outputs/focused_query/contrasts/members_sets.npz`, sha256 `ae57514418a417563c0653ffa859023a52539dfdc3f78bde4dad4618e8fb9580` |
| global start index | `outputs/focused_query/contrasts/start_index.npz`, sha256 `1f41523e5c538f31f9b8f6242eadefe35e834944277e23057144d55e2f3f377b` |
| measured summary | `outputs/focused_query/reports/contrast_summary.json` |
| shard and resource manifests | `outputs/focused_query/resources/` |

### Collection resources

All simulator work ran on node-62 CUDA 0 and CUDA 2. GPU 1 was not used. Each card ran at most five simulator
processes at a time. The memory sampler queried both allocated cards once per second.

| Stage | EDT interval / active wall | Peak nvidia-smi used memory, CUDA 0 / 2 | GPU·h |
|---|---:|---:|---:|
| recorder, initial segment | 2026-09-30 03:26:31; 227.471 s | 8,589 / 8,589 MiB | 0.1264 |
| recorder, resumed segment | 2026-09-30 03:32:04–03:41:10; 545.646 s | 8,590 / 8,589 MiB | 0.3031 |
| 12,720-branch collection | 2026-09-30 03:45:15–05:10:57; 5,141.297 s | 11,170 / 11,170 MiB | 2.8563 |
| **GPU total** | **5,914.414 s of active allocation** | **11,170 / 11,170 MiB peak** | **3.2858** |

Peak torch allocated memory in any collection process was 1,217,854,976 bytes. The 44 collection processes used
47,475.000 process-seconds in total. Contrast construction and validation used CPU only. The complete
`outputs/focused_query/` tree occupies 15,406,864 bytes after contrast construction.

## Focused q training and paired offline diagnostics

The π-mean baseline is stored independently from the fixed eight candidates. Its 25 stores embed and checksum the
same start records and contain 1,590 mean chunks with shape `[10, 7]`. Collection found 53 successful continuations
(44/1,230 TRAIN and 9/360 HELDOUT), zero render aborts and maximum proprio restore error
`6.693881005048752e-09`. The eight-candidate stores and the 812 TRAIN contrast sets are unchanged.

The 1,590 start frames were restored and encoded once into `outputs/focused_query/start_frames/`; the store occupies
299 MiB and its maximum proprio error is `6.693881005048752e-09`. Each Scheme-A run used all 812 TRAIN contrast sets,
64 sets per update and 5,000 updates. R6_all remained frozen in every run: its 66 parameter tensors had zero maximum
change, zero difference from the checkpoint files and no gradient at any update.

| q seed | training seconds | final TRAIN pair AUC | final TRAIN top-positive | peak torch allocated bytes |
|---:|---:|---:|---:|---:|
| 0 | 551.017 | 0.861863 | 1.000000 | 1,810,013,184 |
| 1 | 572.123 | 0.861004 | 1.000000 | 1,810,013,184 |
| 2 | 540.173 | 0.857103 | 1.000000 | 1,810,013,184 |

The same 12,720 candidate branches were scored before and after training. Rows with an unlabelled candidate are
excluded, leaving 1,229 TRAIN starts and all 360 HELDOUT starts. Exact uniform is the per-start mean over all eight
outcomes; seeded uniform is one deterministic draw. π mean is the separately executed same-policy branch.

| q seed | checkpoint | split | pair AUC | top-positive | all candidates | π-only | exact uniform | seeded uniform | π mean | oracle |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | round 0 | TRAIN | 0.5547 | 0.5191 | 0.3645 | 0.3572 | 0.3247 | 0.3255 | 0.0358 | 0.6819 |
| 0 | trained | TRAIN | 0.8621 | 1.0000 | 0.6819 | 0.4776 | 0.3247 | 0.3255 | 0.0358 | 0.6819 |
| 0 | round 0 | HELDOUT | 0.6018 | 0.5952 | 0.4417 | 0.4389 | 0.3660 | 0.3806 | 0.0250 | 0.7250 |
| 0 | trained | HELDOUT | 0.6025 | 0.5357 | 0.4000 | 0.4167 | 0.3660 | 0.3806 | 0.0250 | 0.7250 |
| 1 | round 0 | TRAIN | 0.5513 | 0.5240 | 0.3678 | 0.3556 | 0.3247 | 0.3076 | 0.0358 | 0.6819 |
| 1 | trained | TRAIN | 0.8611 | 1.0000 | 0.6819 | 0.4622 | 0.3247 | 0.3076 | 0.0358 | 0.6819 |
| 1 | round 0 | HELDOUT | 0.6135 | 0.6310 | 0.4667 | 0.4722 | 0.3660 | 0.3667 | 0.0250 | 0.7250 |
| 1 | trained | HELDOUT | 0.5963 | 0.5516 | 0.4111 | 0.4444 | 0.3660 | 0.3667 | 0.0250 | 0.7250 |
| 2 | round 0 | TRAIN | 0.5475 | 0.5179 | 0.3637 | 0.3596 | 0.3247 | 0.3238 | 0.0358 | 0.6819 |
| 2 | trained | TRAIN | 0.8571 | 1.0000 | 0.6819 | 0.4597 | 0.3247 | 0.3238 | 0.0358 | 0.6819 |
| 2 | round 0 | HELDOUT | 0.6126 | 0.6111 | 0.4528 | 0.4556 | 0.3660 | 0.3639 | 0.0250 | 0.7250 |
| 2 | trained | HELDOUT | 0.5908 | 0.5873 | 0.4361 | 0.4417 | 0.3660 | 0.3639 | 0.0250 | 0.7250 |

Training fits the collected TRAIN groups: all-candidate selection reaches the measured oracle on all three seeds.
This improvement does not transfer to the held-out failure episodes. Relative to round 0, trained all-candidate
success changes by −0.0417, −0.0556 and −0.0167 for seeds 0, 1 and 2; held-out pair AUC changes by +0.0007,
−0.0172 and −0.0218. The trained selectors remain above exact uniform. Their episode-bootstrap all-candidate minus
exact-uniform intervals are `[0.0056, 0.0615]`, `[0.0226, 0.0698]` and `[0.0344, 0.1024]`; all-candidate minus π-mean
intervals are `[0.3000, 0.4500]`, `[0.3083, 0.4694]` and `[0.3444, 0.4833]`. Each interval uses 2,000 draws over the
24 HELDOUT episodes. The complete TRAIN/HELDOUT reports include π-only intervals, source/scale strata, per-task
counts and pre/post score changes in `outputs/focused_query/reports/offline_metrics.json`.

Artifacts are under `outputs/focused_query/mean/`, `start_frames/`, `scores/` and
`runs/query_match_focused_seed{0,1,2}/`. Full π-mean collection used node-62 CUDA 0 and 2 from 06:00:08 to 06:28:27
EDT, with 7,215.085 process-seconds and 3,026.586 device-active seconds (0.8407 GPU·h). Start rendering used CUDA 2
for 45 s. Round-0 and trained scoring used 42 s of device allocation. Training used 559, 580 and 549 s of allocation
for seeds 0, 1 and 2 (0.4689 GPU·h). The principal full runs therefore used 1.3338 GPU·h; smoke, round-trip,
diagnostic and resource-validation calls bring measured Phase-5 allocation to 1.36 GPU·h. Mean collection peaked at
692,905,472 torch-allocated bytes per process; scoring at 1,626,850,816 bytes per process; training at
1,810,013,184 bytes per process. The final `outputs/focused_query/` tree occupies 1,174,394,328 bytes.

## Focused q closed-loop evaluation

The trained q checkpoints were evaluated with the same paired 10 tasks × 50 benchmark initial states used by the
round-0, uniform and π-mean references. Query seed k used no-ψ π seed k, 32 proposal candidates, R6_all predicted
effects and a 300-step limit. Every one of the 30 task runs contains exactly 50 unique initial states. The saved run
configs have the expected q, π, R6_all and stage-one checkpoints and code commit.

| q / π seed | trained q | round-0 q | uniform | π mean | trained − round 0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.820 | 0.916 | 0.884 | 0.940 | −0.096 |
| 1 | 0.806 | 0.900 | 0.866 | 0.922 | −0.094 |
| 2 | 0.788 | 0.890 | 0.878 | 0.918 | −0.102 |
| **mean** | **0.805** | **0.902** | **0.876** | **0.927** | **−0.097** |

The paired counts below put trained q first. Each seed row contains 500 pairs and each pooled row contains 1,500.

| Reference | Seed | both | trained only | reference only | neither | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| round 0 | 0 | 383 | 27 | 75 | 15 | 2.15e-6 |
| round 0 | 1 | 369 | 34 | 81 | 16 | 1.38e-5 |
| round 0 | 2 | 359 | 35 | 86 | 20 | 4.00e-6 |
| round 0 | pooled | 1,111 | 96 | 242 | 51 | 1.10e-15 |
| uniform | 0 | 370 | 40 | 72 | 18 | 0.00322 |
| uniform | 1 | 358 | 45 | 75 | 22 | 0.00785 |
| uniform | 2 | 353 | 41 | 86 | 20 | 8.06e-5 |
| uniform | pooled | 1,081 | 126 | 233 | 60 | 1.75e-8 |
| π mean | 0 | 389 | 21 | 81 | 9 | 1.71e-9 |
| π mean | 1 | 378 | 25 | 83 | 14 | 1.88e-8 |
| π mean | 2 | 367 | 27 | 92 | 14 | 1.73e-9 |
| π mean | pooled | 1,134 | 73 | 256 | 37 | 5.69e-25 |

The version effect is larger than seed variation. Across trained and round-0 q, the sample SD of the two version
means is 0.0688 and the pooled within-version seed SD is 0.0147. A two-way version × seed calculation gives version
F(1,2) = 1639.69, p = 0.00061. The trained-minus-round-0 differences are −0.096, −0.094 and −0.102, so the degradation
has the same sign and similar size for all three q / π pairs.

| Task | trained q | round-0 q | uniform | π mean | trained − round 0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.820 | 0.913 | 0.913 | 0.913 | −0.093 |
| 1 | 0.587 | 0.853 | 0.813 | 0.820 | −0.267 |
| 2 | 0.760 | 0.913 | 0.893 | 0.973 | −0.153 |
| 3 | 0.927 | 0.980 | 0.967 | 0.987 | −0.053 |
| 4 | 0.967 | 0.973 | 0.947 | 0.980 | −0.007 |
| 5 | 0.887 | 0.927 | 0.933 | 0.960 | −0.040 |
| 6 | 0.647 | 0.747 | 0.680 | 0.860 | −0.100 |
| 7 | 0.640 | 0.827 | 0.780 | 0.873 | −0.187 |
| 8 | 0.833 | 0.900 | 0.873 | 0.907 | −0.067 |
| 9 | 0.980 | 0.987 | 0.960 | 0.993 | −0.007 |

The 293 trained-q failures contain no stuck episodes: 32 move another object without moving the goal object, 44 move
nothing, 111 lift without reaching the target, 87 displace without lifting and 19 make only partial joint motion.
The loss is concentrated in tasks 1, 2 and 7, but trained q is below round 0 on every task.

This closes the focused test with a negative result. Same-start success contrasts are sufficient to fit the collected
TRAIN branches, but the resulting q does not transfer to either held-out failure starts or its own closed-loop state
distribution. The trained selector is worse than uniform candidate selection, so it adds a systematically harmful
preference rather than merely failing to exploit R6_all. The offline HELDOUT change and closed-loop change have the
same direction.

The three runs used node-61, node-63 and node-65 respectively, GPUs 0 and 1 on each node, from 11:36:14–11:41:07 EDT.
Wall times were 293, 266 and 280 seconds; each GPU peaked at 9,769 MiB. The six devices used 0.4661 physical GPU·h in
total. The machine-readable validation and paired metrics are in
`outputs/focused_query/reports/closed_loop_metrics.json`; raw summaries, failure classifications and one-second
resource samples are under `tmp/runlogs/phase7/`.

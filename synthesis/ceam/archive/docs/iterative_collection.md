# Iterative closed-loop collection (phases 21–23)

The protocol is fixed in plan.md, section "夜间自主推进（用户指令，2026-09-29 美东约 01:50）". Each round collects single-candidate branches at fixed start states with the current r and q, labels them with the task predicates, retrains r's effect head on all data so far and q (scheme A on all groups once at least 300 groups hold both a positive and a negative, otherwise the scheme-1 recipe on the demonstration tables; every round 1–4 used scheme 1), and measures a low-cost closed-loop metric. Framework mapping (`docs/framework.md`): the collection is data for the action representation r (clause 1) and for the within-observation comparison of the match (clause 3); the candidate pool is a replaceable sampler (clause 4). No component or loss outside the framework is added.

The interface between collection (phase 22) and training (phase 21) is `tmp/plan3/iterative_interface.md`.

Times are EDT, 2026-09-29.

## Collection

Code: `scripts/iterative_collect.py` (subcommands `pool`, `keys`, `collect`, `verify`, `label`, `report`); record formats `ceam/data/iterative_records.py` (`StartPool`, `RoundRecords`, `RestoreRecipe`). Existing classes are used unchanged: `LiberoTaskEnv`, `ProposalCandidates`, `SmoothActionSampler`, `ActionRepresentationLoader`, `QueryMatchCheckpointStore`, `ObservationEncoder`, `TaskGoalLabeller`, `ChunkTaskLabeller`.

### Start pool

`outputs/iterative/start_pool/start_pool.npz`, 2,000 starts, written once (`pool`, CPU, code `eb70322`, node-61, 21 s). No HELDOUT state and no sweep state enters.

| Source | Origin | Starts | Selection | Restore recipe | Original chunk and its labels |
|---|---|---|---|---|---|
| `demo_train` | TRAIN observations of the query / match tables (`demo_renders` transitions, 1 ≤ t ≤ T − 11) | 1,000: 100 per task | 20 per task where the task's own goal holds at o (200 = 20%); 80 per task stratified by position b = floor(10 t / T) among the others, 8 per bin; a bin short of 8 is filled from the task's other bins (tasks 3, 4, 5, 6, 9: bin 9 has 0–5, filled 3–8) | sidecar snapshot of frame t − 1 and the dataset action a_{t−1} | the demonstration chunk; labels of `chunks/demo_renders.npz` (its demonstration as the record) |
| `pi_closed_loop` | Phase-26 records (fixed, flow, flow_full_sgd), benchmark inits 0–39 | 500: 50 per task, uniform over the eligible decisions | excluded: decisions 1, 6, 11, 16, 21, 26 of every episode that failed in the reference or the re-run (a superset of the 1,086 sweep states), and inits 40–49 | the record's snapshot and producing action | the recorded π-mean chunk; labels of `chunks/phase26_records.npz` |
| `pi_closed_loop` | `pi_noise_nopsi`, TRAIN starts 0–79 | 500: 50 per task, uniform over steps 1–29 | starts 80–99 excluded | the parent frame's sidecar snapshot and the 10 actions of the chunk that produced the frame | the executed noisy π chunk; labels from its chain (the start's 31 frames, goal vectors computed here, 36–40 chains per task) |

Draws: `numpy.random.default_rng([22, word, task, …])`. Per start the pool holds the task, origin, sim state, fixture poses, the 10-task goal vector, the gripper command in force, the restore recipe, the original chunk with its executed steps and labels, the state after it with its goal vector and its own restore recipe (`pool_report.json` has the counts).

| Source | Own goal holds at the start | Original chunk positive for the own task | Original chunks negative (all tasks) |
|---|---|---|---|
| demo | 200 | 987 / 1,000 | 154 (conflicts of the bowl and bottle tasks) |
| Phase-26 records | 0 (an episode ends at its first success) | 376 / 500 | 52 |
| pi-noise | 54 | 92 / 500 | 4 |

**Restore recipes.** A recipe is a snapshot and one or more sim actions: `LiberoTaskEnv.restore(snapshot, first action)`, then `rollout` of the rest. Replaying the recipes of 120 starts (12 per task, `verify`, node-65) reaches the stored state within max |Δ| 1.7e-7 over qpos and qvel (median 2e-15); the recipes of the states after the original chunks within 6.3e-7. In an earlier 12-start check one next-state recipe (of 12) differed by 0.18 in one velocity entry.

**Other-task keys.** `outputs/iterative/start_pool/train_keys.npz`: `RetrievalKeys.frozen` of the rendered frame of every TRAIN table observation (43,701 × 1,536, `keys`, 13 s), the key of the tables' neighbours.

### One round

`collect` runs on the GPU, one worker per task (worker k takes the decision points whose task ≡ k mod 10).

- **Plan.** `default_rng([22, 100, round])` draws 250 starts from the pool without replacement (a start can recur in later rounds) and permutes the selection types 100 exploit, 50 low-match, 50 uncertain, 50 random.
- **Restore** the start with its recipe and encode the rendered observation.
- **Candidates (32)**, all at the same observation:
  - 8 π noisy, scales 0.05 / 0.1 / 0.2 cycled, and 8 π noisy, scales 0.5 / 1 / 2 cycled: no-ψ π seed 0 (`goal_proposal_nopsi_seed0` step 3000) under the start task's original instruction, `ProposalCandidates` offsets with DCT J = 1, no gripper flip, streams (22, round, start, 1 | 2); π's mean chunk itself is not a candidate;
  - 8 other-task demonstration chunks: a random 8 of the 64 nearest other-task TRAIN observations by frozen key of the rendered observation;
  - 8 smooth random chunks (`SmoothActionSampler`, J 1/2/3, s 0.5/1/2, gripper switch 0.3 from the command in force).
  - A 33rd, fresh smooth chunk is drawn for the random selection.
- **Scores.** r(o, a) of the 33 chunks with the round's r (`frozen_pca_delta`), q_k(o, l) of the start task's original instruction for q seeds k = 0, 1, 2, m_k = the seed's `Match`. A q trained on another r is refused.
- **Selection** over the 32: exploit = argmax of the mean m over seeds; low-match = argmin of the mean m; uncertain = argmax of the standard deviation of m over seeds; random = the fresh smooth chunk.
- **Execution.** The selected chunk runs in full (10 steps, `rollout`, no stop at success), so a goal-holding start can show a break. If the start task's goal then holds (LIBERO's done), the record ends; otherwise π's mean chunk (no-ψ π seed 0, the start task's instruction) runs at every later boundary with `execute` (stop at success), until the task's success or 300 env steps from the decision point. Each env step runs alone after a snapshot, so every boundary keeps the snapshot before its last step and that step's action. Frames are rendered only for π; the record stores no frame and no feature.
- **Labels** (`label`, CPU). The 10-task goal vector of every boundary state (`TaskGoalLabeller` on the stored state) and, per task l, with `ChunkTaskLabeller` (record = the boundaries, chunk = boundary 0 → 1):
  - positive: l holds at some boundary ≥ 1;
  - negative: a break (l holds at boundary 0, not at 1) or a conflict (boundary 1 newly satisfies a task that conflicts with l and not l);
  - both, or neither: undetermined.
- **Records** `outputs/iterative/round_<n>/records/shard_<k>_of_10.npz` (`RoundRecords`, fields in its docstring): the executed chunk, its selection and kind, the scores of all 32 under the three q, the pool's chunks, boundaries flat with `n_boundaries` / `boundary_offsets`, each boundary's one-step restore recipe and goal vector, the continuation chunks and their executed steps, end reason, labels, provenance (r and q checkpoints, commit, host, GPU).
- A black render before the candidate chunk completes skips the decision point (listed in the provenance); one during the continuation ends the record with `render_abort`.

Smoke (not a round; `outputs/iterative_smoke/round_1`, R6_nine with the phase-14 q, 2 decision points in each of tasks 3 and 7): the four records' goal vectors at boundary 0 equal the pool's, the executed chunk equals its pool entry, all four ended in task success and are positive for the own task only. Replaying every boundary recipe of those records (34 boundaries) reaches the stored state within 2.8e-7.

## Round training

### Components and the framework

| Component | Framework clause | Implementation |
|---|---|---|
| r(o, a) | 1 (action representation, task-free) | R6: fixed PCA-256 of the frozen DINOv2 pooled feature, and an effect head. Each round continues the previous round's effect head (round 1 continues R6_all). The regression target ψ(o′) − ψ(o) is R6's historical engineering choice and is unchanged. |
| e(l) | 1 (the task input of the query) | `TaskEncoder`, as in the MVP |
| task-side non-collapse | 2 (query side) | SIGReg on e(l) of all 10 tasks, one uniform instruction variant each, every update, as in the MVP |
| q(o, l), m = s · cos(r, q) | 1, 3 | `QueryNetwork` and `Match`, as in the MVP |
| scheme 1 (MVP recipe) | 3 | InfoNCE over {a⁺} ∪ N(o) on the demonstration observations of the tables, 5,000 updates (`configs/query_match/r6all_seed<k>.json`) |
| scheme A | 3 (within one observation; consistent actions accepted, reachable inconsistent ones distinguished) | `SchemeAObjective`, defined after this table |
| selection, sampler | 4 | phase 22 (not here) |

For one start o and task l, the chunks executed from o split into positives P, definite negatives N (a break or a conflict caused by the chunk itself) and undetermined chunks. Scheme A's loss for (o, l) is

L = −log Σ_{p∈P} exp m_p + log Σ_{c∈P∪N} exp m_c,

computed only when P and N are both non-empty. Undetermined chunks are left out of the denominator. The positives are not ranked against each other.

No new component or loss beyond `plan_next.md` section 2 and the pre-registered protocol is introduced. The following are engineering choices, not framework requirements:
- the relaxed clamp of s under scheme A (1000, the phase-15 value; plan_next: "温度上限放宽");
- the warmup of the r continuation (200 updates);
- the balanced sampling of the r sources;
- the exploration cap rule, described below.

### Code

| Class | File |
|---|---|
| `RecipeRenderer`, `RenderPlan`, `RenderWorker`, `RenderCache`, `ShardIndex` | `ceam/training/iterative_render_cache.py` |
| `BalancedSourceSampler`, `EffectHeadContinuation`, `HeldoutPrediction`, `SameObservationGroups`, `GroupMembers`, `ContrastSets`, `SchemeATrainer`, `StartFrames` | `ceam/training/iterative_round.py` |
| `SchemeAObjective`, `SchemeAWeights`, `SchemeALossReport` | `ceam/losses/scheme_a_objective.py` |
| `QueryMatchTrainingConfig.init_checkpoint`, `QueryMatchTrainer.load_initial` | `ceam/training/query_match_config.py`, `ceam/training/query_match_trainer.py` |
| the record reader (`StartPool`, `RoundRecords`, `RestoreRecipe`; phase 22) | `ceam/data/iterative_records.py` |

- Script: `scripts/iterate_train.py`, with the subcommand `run` (the default) and the internal subcommands `render-worker`, `r` and `q-scheme-a`.
- Round-0 configs: `configs/query_match/r6all_seed{0,1,2}.json`. Each is `r6_seed<k>.json` with the R6_all effect head, `tables_r6all` and its own run directory.

### Steps of `iterate_train.py --round n`

1. **Re-render.** For every round m ≤ n whose cache lacks `render_done.json`, one worker process per task renders every record's boundaries and the starts the round uses first. Workers alternate over the GPUs.
   - Each frame comes from its restore recipe: `LiberoTaskEnv.restore(snapshot, first sim action)`, then `rollout` of the remaining actions, at 512 px.
   - Both views are encoded one at a time by the frozen DINOv2 (8 × 8 pooled), as in `ObservationEncoder`.
   - Three working TransitionStores are written under `tmp/iterative/render_cache/round_<n>/`:

     | Store | Content |
     |---|---|
     | `iter_start` | start frames |
     | `iter_candidate` | (b0, candidate chunk, b1) |
     | `iter_continuation` | (b_i, π-mean chunk, b_{i+1}) |

   - A transition is written only if its chunk ran all 10 steps; a chunk stopped by success is shorter.
   - Earlier rounds' shards are linked in by relative symbolic links, so each store covers rounds 1 … n.
   - `render_index.npz` in each shard maps frames to (round, record, boundary, start). It also records the max |Δ| between the state after the recipe and the recorded state.
   - The cache is never deleted.
2. **r.** `EffectHeadContinuation` continues the previous effect head:
   - 4,000 updates, AdamW, lr 1e-4, weight decay 0.05, warmup 200, cosine to 0, batch 256, bf16, clip 1.0;
   - the fixed PCA is taken from the checkpoint's own buffers;
   - sources: the 12 training stores with their held-out rules, plus `iter_candidate` and `iter_continuation`;
   - each batch draws the source uniformly over the 14, then a transition uniformly within it;
   - the start pool's original transitions are not a separate source: they are already TRAIN transitions of `demo_renders`, `resim_phase26_mean_*` and `pi_noise_nopsi`.

   The checkpoint is `r/checkpoints/step_<previous + 4000>.pt`, in `InteractionCheckpointStore` format. Its config lists the 14 sources and the R6_all PCA file. `r/heldout.json` gives R², mean cos and median relative error, before and after training, on:
   - every held-out transition of each store;
   - the iteration stores' transitions of round n, which the previous head has not seen;
   - the iteration stores' transitions of all rounds.
3. **Tables.** `build_query_match_tables.py --fixed-encoder-checkpoint <new r>` writes `tables/`. The trainer requires the tables' fixed-encoder checkpoint to be the config's r. Its `psi.npy` is compared with `tables_r6all`'s; the PCA is fixed, so the difference should be 0.
4. **Groups and the scheme.** A group is (start, task l):
   - members: the start's original chunk and every executed candidate at that start over rounds 1 … n, each with its [10] labels;
   - a group counts when it has a +1 and a −1 for l;
   - counts are reported per task, per start source and in total;
   - scheme A is used if the total is ≥ 300, otherwise scheme 1; the choice is written to `groups.json` and `report.json`.

   **Exploration cap** ("探索数据不超过训练数据的一半"): exploration members are records with selection `low_match`, `uncertain` or `random`. If more than half of the members that enter contrast sets are exploration members, randomly chosen exploration members (seeded) are dropped until the share is ≤ 0.5. The counts before and after are reported.
   - Scheme 1 reads no records, so its exploration share is 0.
   - In r's batches the two iteration sources make up 2 of 14 sources.
5. **q, seeds 0/1/2**, run in parallel: seeds 0 and 2 on the first GPU, seed 1 on the second.
   - **Initialization** (`--q-init-scheme1`, lead decision at about 04:20, from round 2 on):
     - Scheme 1 starts from the round-0 recipe's own initialization (`recipe`, the default). Continuing would only add epochs on the same demonstration tables, so a round's q now differs from round 0 only through r and its tables.
     - `continue` initializes from `--prev-q` (`init_checkpoint`). Round 1 was trained this way (code `f42f4c8`; `outputs/iterative/round_1/models/q_init_note.json`).
     - Scheme A always continues from `--prev-q`, because it trains on new data.
   - **Scheme 1**: `train_query_match.py` with `r6all_seed<k>.json`, changed only in r, tables and run directory (and `init_checkpoint` under `continue`).
   - **Scheme A**: `SchemeATrainer`. Settings as scheme 1: 64 contrast sets per update, 5,000 updates, lr 3e-4, warmup 150, cosine, weight decay 0.05, clip 1.0, bf16, seed k. The clamp of s is 1000. Contrast sets are balanced over the two start sources. Checkpoints use the `QueryMatchCheckpointStore` format with the round's config, so `run_closed_loop.py` loads them unchanged.
6. **Report.** `report.json` holds:
   - the r and q checkpoint paths, the scheme and the group counts;
   - render totals, r's held-out prediction before and after, and the ψ comparison of the tables;
   - the training curves (`metrics.jsonl`, `evaluation.jsonl`), the frozen checks and the timings.

### Split q mode (`--remote-q-seeds`)

The q step is the longest step. With `--remote-q-seeds 2`, seed 2 is trained by a q-only job on another node (`iterate_train.py q-seed`, entry `tmp/runlogs/s5p21/iterate_q.sh ROUND SEED GPU [--out=<models dir>]`), and seeds 0 and 1 take one GPU each on node-62.

The q-only job:
- polls for `<models>/q_prep.json`, which `run` writes once r, the tables and the three q configs exist. It holds the scheme and each seed's exact command, with the device left open.
- writes nothing into the models directory before that file exists. It stops if `run_failed.json` appears, or after 4 h.
- re-checks its card with `scripts/gpu_guard.sh`, every minute for up to 20 min, then runs the seed's command on its card.
- writes `q_seed<k>_placement.json`: host, device, command, times, exit, and a status of running, done or failed.
- refuses a seed whose placement file already exists.

The node-62 job waits up to 3 h for status done, then writes `report.json` last, with each seed's placement.

Default behaviour without the option is unchanged.

### Commands

Entry scripts, through `scripts/cluster_run.sh`: `tmp/runlogs/s5p21/iterate_round.sh` and `tmp/runlogs/s5p21/iterate_q.sh`. The code is the s5-p21 worktree at the commit in `tmp/runlogs/s5p21/commit.txt`.

```bash
# round n on node-62 CUDA 0/2; seed 2 trained by a q-only job elsewhere (drop --remote-q-seeds=2 for the unsplit form)
bash scripts/cluster_run.sh --gpus 0,2 --free-mib 20000 node-62 $CEAM_ROOT/tmp/runlogs/s5p21/iterate_round.sh <n> 0 2 --remote-q-seeds=2
bash scripts/cluster_run.sh --gpus <g> --free-mib 20000 <node> $CEAM_ROOT/tmp/runlogs/s5p21/iterate_q.sh <n> 2 <g>
```

- Round 1 continues R6_all and the round-0 q. Round n > 1 reads `outputs/iterative/round_<n-1>/models/report.json`.
- Outputs are in `outputs/iterative/round_<n>/models/`: `report.json` (written last), `r/`, `tables/`, `groups.json`, `configs/`, `q_prep.json`, `q_seed<k>/`, `q_seed<k>_placement.json` and `logs/`.
- The render cache is `tmp/iterative/render_cache/round_<n>/`.
- The main log is `tmp/runlogs/s5p21/logs/iterate_round<n>_<time>.log`.

### Verification

Code `35e6aa3`, `d34277f` and `f42f4c8` (branch s5-p21), node-62 CUDA 2, 2026-09-29 02:14–02:49. Scripts and outputs are in `tmp/runlogs/s5p21/` (R below). The tests use reduced step counts: r 60 updates, warmup 10, memory-mapped stores, evaluation sets of ≤ 256 transitions; q 30 updates.

| Check | Result |
|---|---|
| MVP path bit-identical: `mvp_seed0` recipe, 40 updates, evaluation every 20 on 128 + 128 observations (R `configs/bitid_mvp_new.json`), new code against phase 14's `outputs/runs/p14_bitid_qm_new` (R `compare_training.txt`) | metrics without timings, evaluations, tables and frozen checks identical; both checkpoints' parameters bit-identical; configs equal apart from run_dir |
| Config texts: every `configs/query_match/*.json` and the stored step-5000 configs of `query_match_{mvp,r6,r6_fix}_seed0`, with the new code and with the base tree `b59e7bd` (R `check_config_texts.py`, `config_texts_{new,base}.json`) | `to_json_text` identical between the two codes; every stored config text reproduced exactly; `init_checkpoint` is written only when set and reads back equal (R `check_units.txt`) |
| (A) end to end on phase 22's `outputs/iterative_smoke` (4 records, real start pool; R `logs/test_smoke_round1.log`, `test_smoke/`) | render, r, tables, groups (0 → scheme 1), q from the checkpoint, report: exit 0 in 453 s. The tables' ψ equals `tables_r6all` exactly. The frozen check is clean. The restored-state error is ≤ 2.8e-7. |
| (B) synthetic records from 18 real `demo_renders` states with fake labels (R `make_synthetic.py`, `synthetic/`, `check_synthetic.txt`): round 1 forced to scheme A, round 2 scheme 1 continuing from round 1, the round-1 cache linked into round 2 | both rounds exit 0 |
| (B) re-render fidelity, every rendered frame against the stored `demo_renders` frame its recipe reproduces | start frames 18 / 18 tokens bit-identical; record boundaries 181 / 190 bit-identical, the rest within 0.0078 (one fp16 step), min cos 0.999995; proprio identical; restored-state error ≤ 2.6e-7 |
| (B) groups: an independent recount from the npz files | 54 (round 1) and 81 (round 2) groups with a positive and a negative, equal per task to `groups.json` |
| scheme-A loss against a direct float64 loop (35 random sets) | max \|Δ\| 1.1e-6; a set without a negative is refused |
| exploration cap on a constructed case (share 0.667) | 20 exploration members dropped, share 0.474; the same seed gives the same result (R `check_units.txt`) |
| balanced sampler (source sizes 1,000 / 10 / 0 / 50,000 / 3) | shares 0.250 / 0.251 / 0 / 0.250 / 0.250; in the r batches of (A) and (B) the 14 sources get 24–60 of 512 draws each |
| r checkpoint | loads through `ActionRepresentationLoader` (frozen_pca_delta) and `FrozenPcaDeltaModel`; encoder buffers equal R6_all's (no change during training); step = previous + updates; the config lists the 14 sources |
| q checkpoints | load through `QueryMatchCheckpointStore`; config's r = the round's r; `init_checkpoint` = the previous q; scheme-1 configs differ from `r6all_seed<k>.json` only in r, tables, run_dir, init_checkpoint and (tests only) steps; scheme A adds the clamp 1000 |
| `--q-init-scheme1 recipe` (code `df460c1`): the scheme-1 config written with the round-0 r and tables, trained on node-62 CUDA 0 and stopped after update 150 (R `check_recipe_init.txt`) | config equal to `r6all_seed0.json` apart from run_dir, without init_checkpoint; metric lines of updates 0, 50, 100 and 150 and `tables_check.json` bit-identical to `query_match_r6all_seed0` (timings excepted) |
| split mode, twice on (A)'s records (R `test_split/`, `test_split2/`, `logs/iterate_q_round1_seed2_*.log`) | the q-only job was launched before the prep job, waited 210 s and 181 s for `q_prep.json`, passed gpu_guard (second test) and trained seed 2; `report.json` records the three placements; r results identical across the three runs of (A)'s records |

Test timings (the GPU shared with round-0 training): render 16–25 s for 4–28 records; r 20–27 s; tables 138–216 s; q 18 s (scheme A) and 166–187 s (scheme 1, mostly loading `demo_renders`).

### Round 0

r is R6_all (`interaction_frozen_pca_delta_all_stores_seed0` step 20,000, phase 19). q was trained with the scheme-1 (MVP) recipe, seeds 0/1/2, 5,000 updates, on `configs/query_match/r6all_seed<k>.json`. These configs are `r6_seed<k>.json` with R6_all, `tables_r6all` and their own run directories.

- **Tables** `outputs/query_match/tables_r6all`: 67 s on node-62 CUDA 0. d_min 1.9007 and the π MSEs are those of `tables_mvp` / `tables_r6`; only `psi.npy` comes from R6_all's encoder.
- **Runs** `outputs/runs/query_match_r6all_seed{0,1,2}`: code `b59e7bd`, node-62, started 01:58 EDT. Both GPUs were idle at launch.

  | Seed | GPU | Time for 5,000 updates | Finished |
  |---|---|---|---|
  | 1 | CUDA 2, alone | 2,904 s | 02:47 |
  | 0 | CUDA 0, shared with seed 2 | 4,637 s | 03:16 |
  | 2 | CUDA 0, shared with seed 0 | 4,638 s | 03:16 |

  Peak GPU memory 6.0 GiB. The r frozen check is clean in all three: 66 tensors, change 0, grad None at every update.
- **Offline evaluation**: `evaluate_query_match.py tasks` and `acceptance --observations 4864`, node-62 CUDA 2, 02:47–03:22. Outputs are `outputs/analysis/query_match/{tasks,acceptance}/r6all_seed<k>.json`, with 4,864 seeded TRAIN observations and all 4,864 HELDOUT observations, held-out candidate seed. The launcher is R `round0_eval.sh`; the summary is R `summarize_round0.py` → `round0_summary.txt`.

Same-observation acceptance, candidates with the exclusion (the recipe's). Chance top-1 is 1/33. The r6 (phase 14, R6_nine) and MVP rows are copied from `docs/query_match.md` for comparison.

| q | TRAIN top-1 | HELDOUT top-1 | HELDOUT rank mean | HELDOUT: a⁺ beats every smooth / other-task / π-noise candidate | HELDOUT swapped top-1 | HELDOUT a⁺ higher under the own text |
|---|---|---|---|---|---|---|
| r6all, seeds 0 / 1 / 2 | 0.718 / 0.730 / 0.728 | 0.235 / 0.237 / 0.227 | 8.10–8.35 | 0.720–0.724 / 0.714–0.716 / 0.291–0.305 | 0.146–0.170 | 0.706–0.711 |
| r6 (phase 14), seeds 0 / 1 / 2 | 0.762 / 0.768 / 0.761 | 0.228 / 0.217 / 0.224 | 8.40–8.75 | 0.691–0.698 / 0.680–0.704 / 0.282–0.300 | 0.131–0.144 | 0.698–0.721 |
| MVP, seeds 0 / 1 / 2 (every TRAIN observation) | 0.402 / 0.397 / 0.380 | 0.255 / 0.231 / 0.238 | 7.82–8.04 | 0.758–0.774 / 0.775–0.787 / 0.289–0.307 | 0.112–0.127 | 0.859–0.886 |

- **r within these candidate sets** (R6_all against R6_nine): per-state effective rank 6.2–6.6 against 7.2–7.6; pooled within-state rank 70–72 against 79–82; within-state share 0.42–0.44 against 0.40–0.42.
- **Training batches**, updates 4,500–4,999: InfoNCE 1.19–1.26, top-1 0.71–0.75, s 31.6–31.9. The r6 runs had 1.11–1.16, 0.75–0.79 and 32.0–32.1.
- **Periodic evaluation at update 5,000** (1,024 + 1,024 observations): TRAIN top-1 0.683–0.738, HELDOUT top-1 0.222–0.261.

Task encodings (`tasks`, all 50 instruction variants):

| q | within-task cos distance | between-task mean / closest pair | nearest variant / centroid accuracy | ‖e(l)‖ |
|---|---|---|---|---|
| r6all, seeds 0 / 1 / 2 | 0.0014 / 0.0014 / 0.0016 | 1.111 / 1.091–1.093 | 1.00 / 1.00 | 14.4–14.5 |
| r6 (phase 14) | 0.0014–0.0015 | 1.111 / 1.088–1.091 | 1.00 / 1.00 | 14.4–14.5 |

Reading:
- On R6_all the MVP recipe gives HELDOUT acceptance at the level of the r6 and MVP q: top-1 0.23–0.24 against 0.22–0.23 and 0.23–0.26.
- TRAIN acceptance is somewhat lower than r6's (0.72–0.73 against 0.76–0.77). R6_all's r spreads the candidates of one observation less: per-state rank 6.2–6.6 against 7.2–7.6.
- The task encodings separate all 10 tasks, as before.
- Closed-loop numbers of round 0 are phase 23's.

### Round-by-round training records

Summaries come from `report.json` via R `summarize_round.py`, written to R `round<n>_summary.txt`. There was no round 5: the final evaluation (phase 23) started with the round-4 models.

**Overview.** q metrics come from the periodic evaluation at update 5,000 (1,024 TRAIN and 1,024 HELDOUT observations), as ranges over seeds 0/1/2. r's held-out R² is taken on every held-out transition of the listed stores.

| Round | Trained (EDT) | Scheme; q init; groups with + and − | q TRAIN top-1 | q HELDOUT top-1 | q HELDOUT InfoNCE | r | r held-out R²: demo_renders / rollouts_r0 / pi_noise_nopsi / resim mean / resim sweep | Time: render / r / tables / q |
|---|---|---|---|---|---|---|---|---|
| 0 | 01:58–03:16 | 1; fresh (recipe); — | 0.683–0.738 | 0.222–0.261 | 3.12–3.29 | R6_all (step 20,000) | 0.677 / 0.519 / 0.377 / 0.707–0.710 / 0.528–0.558 | tables 67 s; q 48 min (seed 1) and 77 min (seeds 0, 2 sharing a GPU) |
| 1 | 03:19–04:09 | 1; continue; 4 | 0.869–0.904 | 0.216–0.239 | 4.38–4.46 | step 24,000 | 0.671 / 0.511 / 0.364 / 0.703–0.707 / 0.518–0.550 | 1.1 / 8.4 / 1.2 / 38.7 min |
| 2 | 04:26–05:17 | 1; recipe; 4 | 0.720–0.733 | 0.211–0.264 | 3.20–3.28 | step 28,000 | 0.668 / 0.507 / 0.360 / 0.700–0.705 / 0.518–0.546 | 0.9 / 8.4 / 1.3 / 40.3 min |
| 3 | 05:20–06:10 | 1; recipe; 4 | 0.720–0.748 | 0.222–0.237 | 3.23–3.28 | step 32,000 | 0.667 / 0.505 / 0.358 / 0.699–0.704 / 0.521–0.543 | 1.0 / 8.1 / 2.2 / 38.3 min |
| 4 | 06:13–07:04 | 1; recipe; 6 | 0.725–0.745 | 0.225–0.256 | 3.24–3.30 | step 36,000 | 0.666 / 0.503 / 0.356 / 0.696–0.702 / 0.517–0.536 | 1.0 / 8.8 / 2.4 / 38.3 min |

**Same-data R²** on round n's new transitions, which the compared models did not train on (`outputs/iterative/round_<n>/models/r_same_data_eval.json`):

| Round n | Candidates: n; R6_all / r_{n−1} / r_n (trained on these) | Continuations: n; R6_all / r_{n−1} / r_n (trained on these) |
|---|---|---|
| 1 | 250; 0.627 / (r_0 = R6_all) / 0.994 | 3,002; 0.314 / (r_0 = R6_all) / 0.784 |
| 2 | 250; 0.585 / 0.576 / 0.981 | 3,232; 0.259 / 0.239 / 0.687 |
| 3 | 250; 0.584 / 0.568 / 0.970 | 3,253; 0.284 / 0.266 / 0.644 |
| 4 | 250; 0.591 / 0.573 / 0.958 | 3,385; 0.257 / 0.247 / 0.603 |

Round 1's values come from its `r/heldout.json`, before and after training.

**Round 1.** It trained 03:19–04:09 (49 min) on code `f42f4c8`, in the split mode. q was continued from the round-0 q (`q_init` continue; see the note file).

Placement and timings:
- node-62 CUDA 0/2 ran the prep and seeds 0 and 1. Seed 2 ran on node-66 cuda:1, launched by phase 22, and waited 661 s for `q_prep.json`.
- render 63 s, r 501 s (105 s of it loading the stores; 4,000 updates at 11.1/s), tables 73 s, q 2,321 s.

Data and scheme:
- Records: 250 (250 starts). Rendered 4,066 frames; 250 candidate and 3,002 continuation transitions. 110 continuation chunks stopped at success were skipped. 0 refusals.
- Tables: ψ equals `tables_r6all` exactly.
- Groups with a positive and a negative: 4, one each for tasks 1, 3, 5 and 7, all at demo_train starts. Scheme 1.
- Restore check. One start frame (start 1676, a pi-noise start with a 10-action recipe) is 0.17 from the pool's `sim_state`. This equals phase 22's own `restore_error` for that record, so the rendered frame is the state the collection used. Of 3,316 boundary frames, 6 differ from the recorded state by more than 1e-4 (max 0.092); the rest by ≤ 2e-6.

r, held-out prediction before (R6_all) and after (round-1 r):

| set | n | R² | mean cos | median rel. error |
|---|---|---|---|---|
| demo_renders `heldout_*` | 4,864 | 0.677 → 0.671 | 0.799 → 0.796 | 0.561 → 0.568 |
| rollouts_r0 starts 80–99 | 4,470 | 0.519 → 0.511 | 0.624 → 0.620 | 0.753 → 0.759 |
| resim mean, inits 40–49 (3 stores) | 1,173–1,236 | 0.707–0.710 → 0.703–0.707 | 0.795–0.801 → 0.794–0.800 | 0.510–0.528 → 0.515–0.527 |
| resim sweep, inits 40–49 (3 stores) | 1,728–1,920 | 0.528–0.558 → 0.518–0.550 | 0.655–0.672 → 0.652–0.670 | 0.684–0.738 → 0.690–0.746 |
| pi_noise_nopsi starts 80–99 | 6,000 | 0.377 → 0.364 | 0.590 → 0.585 | 0.829 → 0.835 |
| round-1 candidate transitions (not seen by R6_all) | 250 | 0.627 → 0.994 | 0.751 → 0.994 | 0.584 → 0.078 |
| round-1 continuation transitions (not seen by R6_all) | 3,002 | 0.314 → 0.784 | 0.483 → 0.766 | 0.932 → 0.560 |

- The before values reproduce phase 19's held-out table (`docs/query_match.md`), e.g. demo_renders 0.677, pi_noise_nopsi 0.377.
- Balanced sampling gives the 250 candidate transitions 1/14 of every batch, about 290 draws each. Their R² of 0.994 after training is fit, not generalization. The held-out R² of the twelve stores falls by 0.003–0.013.

q, scheme 1, continuing from round 0 (updates 4,500–4,999 of the batches; periodic evaluation at update 5,000 on 1,024 + 1,024 observations):

| seed | placement | InfoNCE / top-1 / s | TRAIN top-1 / HELDOUT top-1 / HELDOUT InfoNCE |
|---|---|---|---|
| 0 | node-62 cuda:0 | 0.499 / 0.912 / 64.4 | 0.904 / 0.239 / 4.382 |
| 1 | node-62 cuda:2 | 0.511 / 0.896 / 64.1 | 0.869 / 0.217 / 4.423 |
| 2 | node-66 cuda:1 | 0.557 / 0.885 / 63.9 | 0.884 / 0.216 / 4.459 |

- Under scheme 1, q's data are the unchanged demonstration tables. The recipe fixes the batch and candidate order by the seed, so round 1 replays round 0's 5,000 batches from the round-0 q; only r differs.
- Against round 0: TRAIN top-1 rose from 0.68–0.74 to 0.87–0.90; HELDOUT top-1 stayed at 0.22–0.24; HELDOUT InfoNCE rose from 3.12–3.29 to 4.38–4.46; s doubled from 32 to 64.

**Round 2.** It trained 04:26–05:17 (51 min) on code `df460c1`, in the split mode (seed 2 on node-66 cuda:1). q followed the recipe (`q_init` recipe).

Timings: render 56 s, r 506 s (store loading 112 s, 4,000 updates at 11.7/s), tables 75 s, q 2,420 s.

Data and scheme:
- Records: 250, of which 213 at starts not used before. Rendered 4,265 frames; 250 candidate and 3,232 continuation transitions; 112 short chunks skipped; 0 refusals. Max restore-state error 1.1e-4.
- Groups with a positive and a negative: still 4 over rounds 1–2, one each for tasks 1, 3, 5 and 7. Scheme 1.

r, held-out R² of the twelve stores, R6_all → round 1 → round 2:
- demo_renders 0.677 → 0.671 → 0.668;
- rollouts_r0 0.519 → 0.511 → 0.507;
- pi_noise_nopsi 0.377 → 0.364 → 0.360;
- resim sweep 0.528–0.558 → 0.518–0.550 → 0.518–0.546.

**Same data** (`eval_r_same_data.py`, R; `outputs/iterative/round_2/models/r_same_data_eval.json`), on round 2's new transitions, which neither model trained on:

| round-2 transitions | n | R6_all: R² / mean cos | r_1: R² / mean cos |
|---|---|---|---|
| candidates | 250 | 0.585 / 0.736 | 0.576 / 0.731 |
| continuations | 3,232 | 0.259 / 0.461 | 0.239 / 0.458 |

- **Distribution shift:** R6_all's own R² falls from round-1 data to round-2 data, from 0.627 to 0.585 on candidates and from 0.314 to 0.259 on continuations. The two rounds' records were selected by different q.
- **Same data:** r_1 is below R6_all, by 0.009 on candidates and 0.020 on continuations. The round-1 continuation did not improve the prediction of unseen iteration transitions.

q (recipe init), at update 5,000:

| seed | placement | batches 4,500–4,999: InfoNCE / top-1 / s | TRAIN top-1 / HELDOUT top-1 / HELDOUT InfoNCE |
|---|---|---|---|
| 0 | node-62 cuda:0 | 1.216 / 0.749 / 32.0 | 0.733 / 0.264 / 3.255 |
| 1 | node-62 cuda:2 | 1.185 / 0.732 / 31.9 | 0.720 / 0.246 / 3.204 |
| 2 | node-66 cuda:1 | 1.234 / 0.699 / 31.7 | 0.723 / 0.211 / 3.284 |

**Round 3.** It trained 05:20–06:10 (50 min) on code `df460c1`, in the split mode (seed 2 on node-66 cuda:1), with `q_init` recipe.

Timings: render 58 s, r 487 s (store loading 114 s, 4,000 updates at 12.2/s), tables 129 s, q 2,296 s.

Data and scheme:
- Records: 250, of which 190 at new starts. Rendered 4,271 frames; 250 candidate and 3,253 continuation transitions; 115 short chunks skipped; 0 refusals. Max restore-state error 6.2e-3.
- Groups with a positive and a negative: still 4 over rounds 1–3 (tasks 1, 3, 5, 7). Scheme 1.

r, held-out R² of the twelve stores, R6_all → r_1 → r_2 → r_3:
- demo_renders 0.677 → 0.671 → 0.668 → 0.667;
- rollouts_r0 0.519 → 0.511 → 0.507 → 0.505;
- pi_noise_nopsi 0.377 → 0.364 → 0.360 → 0.358;
- resim sweep 0.528–0.558 → … → 0.521–0.543.

Same data, round 3's new transitions (`outputs/iterative/round_3/models/r_same_data_eval.json`):

| round-3 transitions | n | R6_all: R² / mean cos | r_2: R² / mean cos | r_3 (trained on these): R² |
|---|---|---|---|---|
| candidates | 250 | 0.584 / 0.749 | 0.568 / 0.742 | 0.970 |
| continuations | 3,253 | 0.284 / 0.464 | 0.266 / 0.463 | 0.644 |

q (recipe init), at update 5,000:

| seed | placement | batches 4,500–4,999: InfoNCE / top-1 / s | TRAIN top-1 / HELDOUT top-1 / HELDOUT InfoNCE |
|---|---|---|---|
| 0 | node-62 cuda:0 | 1.193 / 0.751 / 32.0 | 0.748 / 0.233 / 3.269 |
| 1 | node-62 cuda:2 | 1.161 / 0.763 / 31.9 | 0.720 / 0.237 / 3.232 |
| 2 | node-66 cuda:1 | 1.226 / 0.706 / 31.8 | 0.736 / 0.222 / 3.279 |

**Across rounds 2 and 3**, on the same unseen transitions, the continued r predicts slightly worse than R6_all: by 0.009–0.016 on candidates and 0.018–0.020 on continuations. The iteration so far does not improve r on unseen iteration data. Groups with both a positive and a negative stay at 4, far below the threshold of 300, so every round used scheme 1.

**Round 4.** It trained 06:13–07:04 (50 min) on code `df460c1`, in the split mode (seed 2 on node-66 cuda:1), with `q_init` recipe.

Timings: render 60 s, r 529 s (store loading 114 s, 4,000 updates at 10.9/s), tables 144 s, q 2,298 s.

Data and scheme:
- Records: 250, of which 159 at new starts. Rendered 4,347 frames; 250 candidate and 3,385 continuation transitions; 98 short chunks skipped; 0 refusals. Max restore-state error 3.1e-3.
- Groups with a positive and a negative: 6 over rounds 1–4 (task 1: 1, task 3: 1, task 5: 2, task 7: 2). Scheme 1.

r, held-out R² of the twelve stores, R6_all → r_4:
- demo_renders 0.677 → 0.666;
- rollouts_r0 0.519 → 0.503;
- pi_noise_nopsi 0.377 → 0.356;
- resim mean 0.707–0.710 → 0.696–0.702;
- resim sweep 0.528–0.558 → 0.517–0.536.

Same data, round 4's new transitions (`outputs/iterative/round_4/models/r_same_data_eval.json`):

| round-4 transitions | n | R6_all: R² / mean cos | r_3: R² / mean cos | r_4 (trained on these): R² |
|---|---|---|---|---|
| candidates | 250 | 0.591 / 0.737 | 0.573 / 0.728 | 0.958 |
| continuations | 3,385 | 0.257 / 0.460 | 0.247 / 0.457 | 0.603 |

q (recipe init), at update 5,000:

| seed | placement | batches 4,500–4,999: InfoNCE / top-1 / s | TRAIN top-1 / HELDOUT top-1 / HELDOUT InfoNCE |
|---|---|---|---|
| 0 | node-62 cuda:0 | 1.166 / 0.753 / 32.0 | 0.745 / 0.256 / 3.236 |
| 1 | node-62 cuda:2 | 1.151 / 0.747 / 31.9 | 0.733 / 0.241 / 3.242 |
| 2 | node-66 cuda:1 | 1.214 / 0.699 / 31.8 | 0.725 / 0.225 / 3.305 |

**Over rounds 2–4**, on the same unseen transitions:
- The continued r_{n−1} predicts slightly worse than R6_all, by 0.009–0.018 R² on candidates and 0.010–0.020 on continuations.
- Each round's r fits its own round's transitions: R² 0.96–0.98 on candidates, 0.60–0.69 on continuations. This does not carry over to the next round.
- The held-out R² of the twelve stores falls by 0.008–0.022 over the four rounds. Per round it moves by at most 0.013; one source rose by 0.003 in round 3.

## Rounds

Round n collects with the models of round n − 1 (round 0: r = R6_all, `interaction_frozen_pca_delta_all_stores_seed0` step 20,000; q = `query_match_r6all_seed{0,1,2}` step 5,000), then `scripts/iterate_train.py` (phase 21, "Round training") trains the round-n models, then the round-n metric runs.

**Low-cost metric.** π-candidate closed loop, argmax of m(r, q seed 0) among no-ψ π seed 0's 32 candidates (mean + 31 noisy, scales 0.05 / 0.1 / 0.2, DCT J 1), 10 tasks × benchmark inits 40–44, the settings of `v12_r6_argmax_s0` (`tmp/s5p22/closed_loop.sh`: env seed 0, 300 env steps, 10 settle steps, `--check-start-observation`). The process starts at init 40, so its fixture poses differ from those of inits 40–44 inside a 0–49 run; the metric pairs only across rounds. Runs `outputs/closed_loop/p22_metric_round<n>/`.

### Placement and wall times

Collection: node-61 CUDA 0/1, 10 workers (5 per card). Labels and report: node-61 CPU. Training: phase 21's `iterate_round.sh` on node-62 CUDA 0/2 (render, r, tables, q seeds 0 and 1) and `iterate_q.sh` on node-66 CUDA 1 (q seed 2), launched by this phase through the gate. Metric: node-65 CUDA 0/1, 10 processes. Every GPU launch passed `gpu_guard.sh`; the cards were idle of other users at every launch.

| Round | Collection | Labels + report | Training | Metric |
|---|---|---|---|---|
| 0 | – | – | – (round-0 q: phase 21, until 03:16) | 03:17:10–03:17:48 |
| 1 | 03:17:08–03:19:14 | 03:19:18–03:19:52 | 03:19:43–04:09:02 | 04:10:09–04:10:45 (`p22_metric_round1_retry`) |
| 2 | 04:10:07–04:12:08 | 04:12:11–04:12:43 | 04:26:25–05:17:23 | 05:17:38–05:18:21 |
| 3 | 05:17:36–05:19:47 | 05:19:52–05:20:25 | 05:20:16–06:09:48 | 06:10:10–06:10:47 |
| 4 | 06:10:09–06:12:30 | 06:12:35–06:13:12 | 06:13:02–07:03:33 | 07:04:06–07:04:50 (node-63 CUDA 0/1) |

- **Loading failure at 04:09.** The round-1 q configs carry phase 21's optional `init_checkpoint` key, which this branch's `QueryMatchTrainingConfig` refused. The first round-2 collection and the first round-1 metric (`p22_metric_round1`, empty task directories) stopped at loading, before any record or episode. Phase 21's `ceam/training/query_match_config.py` was taken verbatim (`c050fa9`; the key is optional and absent keys keep the old behaviour), and both were relaunched.
- **Hold 04:12–04:26.** The lead held the round-2 training until phase 21's scheme-1 fallback trained q from the round-0 recipe's initialization instead of continuing it (phase 21, "Round training"). Round 1's q is the continued one.
- A collection of 250 decisions takes about 2 min (workers 47–138 s including model loading, 2.7–3.0 s per record); no decision point was skipped and no record ended in a render abort.

### Per round

| Round | Records (exploit / low / unc / random) | Own-task success | Labels, all tasks: + / − / 0 | Labels, own task: + / − / 0 | Groups with + and − after the round (tasks) | Scheme, q init | Metric (inits 40–44) |
|---|---|---|---|---|---|---|---|
| 0 | – | – | – | – | 0 | scheme 1, round-0 recipe | 46/50 = 0.92 |
| 1 | 250 (100 / 50 / 50 / 50) | 168 | 0.064 / 0.014 / 0.922 | 0.616 / 0.016 / 0.368 | 4 (t1:1, t3:1, t5:1, t7:1) | 1, q init continue | 47/50 = 0.94 |
| 2 | 250 (100 / 50 / 50 / 50) | 160 | 0.061 / 0.007 / 0.932 | 0.580 / 0.000 / 0.420 | 4 (t1:1, t3:1, t5:1, t7:1) | 1, q init recipe | 45/50 = 0.90 |
| 3 | 250 (100 / 50 / 50 / 50) | 161 | 0.063 / 0.006 / 0.931 | 0.592 / 0.004 / 0.404 | 4 (t1:1, t3:1, t5:1, t7:1) | 1, q init recipe | 49/50 = 0.98 |
| 4 | 250 (100 / 50 / 50 / 50) | 152 | 0.062 / 0.010 / 0.929 | 0.584 / 0.004 / 0.412 | 6 (t1:1, t3:1, t5:2, t7:2) | 1, q init recipe | 43/50 = 0.86 |

"+ / − / 0" are the shares of positive, negative and undetermined labels over the 250 records × 10 tasks (all tasks) or over the 250 records for the start's own task. The groups count starts of the pool with a positive and a negative for the same task among the original chunk and the executed candidates of rounds 1 … n (this phase's count; phase 21's `groups.json` gives the same 4, 4, 4 and 6). The metric runs are `p22_metric_round0`, `p22_metric_round1_retry`, `p22_metric_round2`, `p22_metric_round3`, `p22_metric_round4`; against round 0 their discordant pairs are 3 / 2, 3 / 4, 4 / 1 and 2 / 5 (McNemar p ≥ 0.37). With 50 episodes the binomial SD of a rate near 0.92 is 0.04.

New labels by selection type: positives / negatives over all tasks, in parentheses the positives for the own task.

| Round | exploit + / − (own +) | low-match + / − (own +) | uncertain + / − (own +) | random + / − (own +) |
|---|---|---|---|---|
| 1 | 81 / 17 (78) | 24 / 4 (24) | 29 / 5 (27) | 27 / 8 (25) |
| 2 | 78 / 7 (77) | 21 / 3 (18) | 27 / 6 (25) | 27 / 2 (25) |
| 3 | 73 / 5 (71) | 26 / 0 (24) | 28 / 8 (26) | 30 / 3 (27) |
| 4 | 66 / 6 (65) | 30 / 5 (27) | 36 / 4 (32) | 22 / 9 (22) |

- **Scheme.** Every round used scheme 1: the same-observation groups with a positive and a negative stayed at 4–6, far below the 300 of scheme A.
- **Why the groups stay few.** π's continuation is conditioned on the start's own task, so positives are almost only for that task (own-task positives are 94–96% of all positives). Negatives are mostly conflicts for the other bowl and bottle tasks, and breaks of the own task are rare (0–4 per round): a same-task pair of a positive and a negative at one start needs a break of an already achieved goal next to a completion. After round 4, 5 of the 6 groups are demo starts where the own goal holds: the demonstration chunk keeps it (positive) and one executed candidate breaks it (2 random smooth, 1 low-match π large offset, 1 low-match and 1 uncertain other-task chunk). The sixth is a task-1 demo start whose demonstration chunk is a conflict for task 5 and whose round-4 uncertain candidate (an other-task chunk) led to task 5's goal holding at a later boundary.
- **Exploit against the other selections.** Exploit chunks reached the own task in 65–78 of 100 records per round, the other selections in 18–32 of 50 (36–64%). Exploit picks π's small offsets in 47–63 of 100 records and π's large offsets in 25–41; low-match picks smooth chunks in 25–31 of 50.
- **Low-match against uncertain** (new labels over rounds 1–4, all tasks): low-match 101 positive / 12 negative, uncertain 120 positive / 23 negative.
- **Restore.** The restored start's state matched the pool's within 3.6e-6 in 999 of the 1,000 records (median about 5e-15); one pi-noise start of round 1 (start 1676) differed by 4.9e-4 in qpos and 0.17 in one velocity.

**r's held-out R²** (phase 21's `report.json`; "before" is the previous round's r, "after" this round's r). The last column is the previous r on the new round's own records, which it has not seen; the "after" values on those records are training data and are not given.

| Round | demo_renders before → after | rollouts_r0 before → after | resim_phase26_mean_fixed before → after | resim_phase26_sweep_fixed before → after | pi_noise_nopsi before → after | this round's records, previous r: candidate / continuation |
|---|---|---|---|---|---|---|
| 1 | 0.677 → 0.671 | 0.519 → 0.511 | 0.708 → 0.704 | 0.558 → 0.548 | 0.377 → 0.364 | 0.627 / 0.314 |
| 2 | 0.671 → 0.668 | 0.511 → 0.507 | 0.704 → 0.700 | 0.548 → 0.545 | 0.364 → 0.360 | 0.576 / 0.239 |
| 3 | 0.668 → 0.667 | 0.507 → 0.505 | 0.700 → 0.699 | 0.545 → 0.543 | 0.360 → 0.358 | 0.568 / 0.266 |
| 4 | 0.667 → 0.666 | 0.505 → 0.503 | 0.699 → 0.696 | 0.543 → 0.536 | 0.358 → 0.356 | 0.573 / 0.247 |

- The previous r predicts the new round's candidate transitions with R² 0.57–0.63 and the π-continuation transitions with 0.24–0.31. Each round's training lowers R² on the held-out parts in the table by 0.001–0.013.

## Final evaluation

π-candidate closed loops, 10 tasks × benchmark inits 0–49, three seeds: q seed k with no-ψ π seed k (k = 0, 1, 2), argmax of m(r, q) among π's 32 candidates, the protocol of `v12_r6_argmax_s<k>` (`tmp/s5p22/closed_loop.sh`). Two (r, q):

- **round 0**: R6_all and `query_match_r6all_seed<k>` step 5,000; runs `outputs/closed_loop/p23_round0_argmax_s<k>/`;
- **final**: the round-4 models (r `outputs/iterative/round_4/models/r/checkpoints/step_036000.pt`, q `outputs/iterative/round_4/models/q_seed<k>/checkpoints/step_005000.pt`); runs `outputs/closed_loop/p23_final_argmax_s<k>/`.

Controls, paired by (task, init): `v9_mean_nopsi_s<k>` (π's mean chunk) and `v11_qm_random_s<k>` (uniform among the same 32 candidates; it uses neither r nor q). The initial scene of every episode of the round-0 runs equals both controls' (1,500 / 1,500). Inits 0–39 are training states of R6 (phase-26 records re-simulated) and of the start pool; inits 40–49 are held out. Analysis `tmp/s5p22/analyze_closed_loop.py` (exact two-sided McNemar on discordant pairs).

Success rates (500 episodes per seed; 400 for inits 0–39, 100 for inits 40–49):

| Selection | inits | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|---|
| final (round 4) | all | 0.914 | 0.916 | 0.890 | **0.907** |
| round 0 | all | 0.916 | 0.900 | 0.890 | **0.902** |
| uniform (v11) | all | 0.884 | 0.866 | 0.878 | 0.876 |
| π's mean chunk (v9) | all | 0.940 | 0.922 | 0.918 | 0.927 |
| final | 0–39 | 0.910 | 0.912 | 0.890 | 0.904 |
| round 0 | 0–39 | 0.917 | 0.905 | 0.900 | 0.908 |
| uniform | 0–39 | 0.887 | 0.868 | 0.877 | 0.877 |
| mean chunk | 0–39 | 0.945 | 0.930 | 0.910 | 0.928 |
| final | 40–49 | 0.930 | 0.930 | 0.890 | 0.917 |
| round 0 | 40–49 | 0.910 | 0.880 | 0.850 | 0.880 |
| uniform | 40–49 | 0.870 | 0.860 | 0.880 | 0.870 |
| mean chunk | 40–49 | 0.920 | 0.890 | 0.950 | 0.920 |

Paired exact two-sided McNemar by (task, init): episodes only the first solves / only the second solves, p.

| Pair | seed 0 | seed 1 | seed 2 | pooled | pooled, inits 0–39 | pooled, inits 40–49 |
|---|---|---|---|---|---|---|
| final vs mean chunk | 26 / 39, 0.14 | 29 / 32, 0.80 | 35 / 49, 0.16 | **90 / 120, 0.045** | 70 / 99, 0.031 | 20 / 21, 1 |
| final vs uniform | 54 / 39, 0.15 | 57 / 32, 0.011 | 55 / 49, 0.62 | **166 / 120, 0.0077** | 129 / 97, 0.039 | 37 / 23, 0.092 |
| final vs round 0 | 34 / 35, 1 | 40 / 32, 0.41 | 42 / 42, 1 | **116 / 109, 0.69** | 85 / 89, 0.82 | 31 / 20, 0.16 |
| round 0 vs mean chunk | 24 / 36, 0.16 | 32 / 43, 0.25 | 31 / 45, 0.14 | 87 / 124, 0.013 | 68 / 93, 0.058 | 19 / 31, 0.12 |
| round 0 vs uniform | 52 / 36, 0.11 | 53 / 36, 0.089 | 53 / 47, 0.62 | 158 / 119, 0.022 | 126 / 90, 0.017 | 32 / 29, 0.80 |

Per task, mean over the three seeds (all inits):

| Task | final | round 0 | uniform | mean chunk |
|---|---|---|---|---|
| 0 | 0.940 | 0.913 | 0.913 | 0.913 |
| 1 | 0.840 | 0.853 | 0.813 | 0.820 |
| 2 | 0.933 | 0.913 | 0.893 | 0.973 |
| 3 | 0.960 | 0.980 | 0.967 | 0.987 |
| 4 | 0.993 | 0.973 | 0.947 | 0.980 |
| 5 | 0.947 | 0.927 | 0.933 | 0.960 |
| 6 | 0.833 | 0.747 | 0.680 | 0.860 |
| 7 | 0.760 | 0.827 | 0.780 | 0.873 |
| 8 | 0.873 | 0.900 | 0.873 | 0.907 |
| 9 | 0.987 | 0.987 | 0.960 | 0.993 |

- **Runs.** Round 0: seed 1 on node-65 CUDA 0/1, 02:48–02:52; seeds 0 and 2 on node-65 and node-61 CUDA 0/1, 03:19–03:24. Final: seed 0 on node-61, seed 1 on node-65, seed 2 on node-63 (CUDA 0/1 each), 07:03:54–07:08:31, after round 4's `report.json` (07:03:33). All 3,000 episodes completed; no render refusal. The initial scene of every final and round-0 episode equals both controls' (1,500 / 1,500 each). Episodes average 12.8 decisions (final) and 12.8–13.2 (round 0).
- **Reading.**
  - After four rounds, argmax over π's candidates succeeds in 0.907, against 0.902 with the round-0 models: the rounds did not change closed-loop success (pooled 116 / 109, p = 0.69; inits 40–49 31 / 20, p = 0.16).
  - Both stay above the uniform choice among the same candidates (final 166 / 120, p = 0.0077; round 0 158 / 119, p = 0.022) and below π's mean chunk (final 90 / 120, p = 0.045; round 0 87 / 124, p = 0.013).
  - On the held-out inits 40–49 the final models reach 0.917, the mean chunk's level (0.920; 20 / 21).
  - The round-0 arm with R6_all and its fresh q (0.902) is at the level of phase 14's R6_nine argmax (`v12_r6_argmax`, 0.901).

### Paths (phases 22–23)

| What | Where |
|---|---|
| start pool, its report, TRAIN keys | `outputs/iterative/start_pool/{start_pool.npz,pool_report.json,train_keys.npz}` |
| round records | `outputs/iterative/round_<n>/records/shard_<k>_of_10.npz` (labelled), `round_<n>/staging/` (before labels), `round_<n>/collection_report.json` |
| smoke records (not a round) | `outputs/iterative_smoke/round_{1,2}/` (R6_nine with the phase-14 q) |
| metric runs | `outputs/closed_loop/p22_metric_round{0,1_retry,2,3,4}/`; failed load attempt `p22_metric_round1/` (empty); launcher smokes `p22_smoke_launcher/`, `p22_smoke_metric_r6nine/` |
| final evaluation | `outputs/closed_loop/p23_final_argmax_s{0,1,2}/`, `outputs/closed_loop/p23_round0_argmax_s{0,1,2}/` |
| entry scripts, analysis, summaries | `tmp/s5p22/` (`env.sh`, `cpu_job.sh`, `gpu_job.sh`, `collect_round.sh`, `closed_loop.sh`, `status.sh`, `analyze_closed_loop.py`, `summarize_round.py`, `round_table.py`, `analysis_final.txt`, `analysis_round0_final.txt`, `metric_round<n>.txt`, `round<n>_summary.txt`, `round_tables.txt`, `commit.txt`) |
| logs | `tmp/runlogs/s5p22/` (pool, keys, verify, `iterative/round_<n>[_HHMMSS]/collect_worker_<k>.log`, `round<n>_{label,report}.log`, `<run>/task<t>.log`) |

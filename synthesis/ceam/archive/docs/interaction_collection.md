# Interaction collection: smooth random chunks, re-simulation and scene sidecars

Tools that collect simulated transitions (o, a, o′) for joint ψ / Δ training (`docs/interaction_training.md`), each with per-frame scene labels. Every frame is a simulator render encoded like the feature cache.

| Class | File |
|---|---|
| `SmoothActionSampler`, `SmoothChunks` | `ceam/data/smooth_action_sampler.py` |
| `SceneSidecar`, `SceneRow`, `FrameProvenance` | `ceam/data/scene_sidecar.py` |
| subcommands `demo-branch`, `init-random`, `resim-branch`, `pi-noise`, `report` | `scripts/collect_interactions.py` |

## SmoothActionSampler

`SmoothActionSampler(actions, chunk_length, dct_coefficients, scales, gripper_switch_prob, seed, input_min, input_max).sample(n, gripper_now, key) -> SmoothChunks`

- **Arm (6 dims)**, in the `ActionCache`-standardized space: δ = s · √(k / J) · Σ_{c<J} x_c b_c, with x_c ~ 𝒩(0, I₆) and b_c the orthonormal 10-step DCT-II basis (`dct_basis`, equal to `ProposalCandidates.dct_basis`; b₀ constant).
  - The chunk is centred on 0, the dataset mean action.
  - s is the per-entry scale: the mean squared entry, averaged over the steps, is s². This is `ProposalCandidates`' normalization of its noisy offsets, so the scales of the random chunks and of π's noisy candidates are in the same units. With J = 1 the chunk is constant.
  - Unstandardized with the `ActionCache` statistics, then clipped to the controller's input range ([−1, 1] on every arm dim).
- **Gripper**: with probability p the chunk's command is 1 − `gripper_now`, otherwise `gripper_now`, for all 10 steps.
- **J and s** are drawn per chunk, uniformly from `dct_coefficients` and `scales`. Collection defaults: J ∈ {1, 2, 3}, s ∈ {0.5, 1, 2}, p = 0.3.
- **Determinism**: `numpy.random.default_rng([seed, *key])`, `key` exactly 4 ints in [0, 2³²) (5-word entropy, never zero-padded). Per chunk, in order: index of J, index of s, a uniform for the switch, x [J, 6]. `sample(n, …)` is a prefix of `sample(n′ ≥ n, …)`.
- `SmoothChunks`: `raw` [n, 10, 7] float32 (dataset convention, as executed), `dct_coefficients` [n], `scales` [n], `switched` [n], `clipped_fraction` [n].

Checks (`tmp/s4p4/check_sampler_sidecar.py`, 60,000 unclipped chunks):

| Quantity | Result |
|---|---|
| basis vs `ProposalCandidates.dct_basis` | max difference 0; ‖BBᵀ − I‖∞ 2.0e-15 |
| same key / prefix / other key / other seed | equal / equal / different / different |
| J, s shares | 0.332 / 0.339 / 0.329; 0.331 / 0.335 / 0.335 |
| gripper switch rate | 0.299 |
| per-entry RMS for s = 0.5 / 1 / 2 (every J) | 0.500–0.501 / 0.998–1.004 / 2.002–2.003 |
| energy share of DCT coefficients ≥ J | ≤ 1.9e-15 |
| mean standardized arm entry | within ±0.0084 of 0 |
| clipped share of arm entries, range [−1, 1], s = 0.5 / 1 / 2 | 0.0001 / 0.012 / 0.111 |

**Normalization.** The sampler uses δ = s · √(k / J) · Σ_{c<J} x_c b_c with x_c ~ 𝒩(0, I₆), equivalently coefficients x_c ~ 𝒩(0, s² k / J), rather than coefficients x_c ~ 𝒩(0, s²). The latter gives a per-entry RMS of s·√(J/10), 0.16–1.1 for these J and s. The reason is coverage: with 2,000 offline draws (J, s uniform, unclipped), the share of chunks farther from every TRAIN demonstration chunk than the 95th percentile of HELDOUT demonstration chunks (arm dims, below) is 0.651 with the implemented form and 0.283 with coefficients 𝒩(0, s²) (`report --normalization-comparison`). The implemented form is also the one of π's noisy candidates, so both use s in the same units.

## SceneSidecar

`<shard>/scene.npz`, one row per shard frame (row i = shard-local frame i), written before the shard's manifest.

- The keys of `docs/demo_renders.md` (`frame`, `episode`, `t`, `task_id`, `cache_row`, `raw_demo_index`, `raw_step`; the `SceneState` fields; `goal_holds`; `sim_state` [n, 79], `gripper_accumulator`, `goal_orientation`, `fixture_bodies`, `fixture_positions`, `fixture_quaternions`).
- Collected shards add the `FrameProvenance` columns:

| Key | Content |
|---|---|
| `origin` | `restore` (a dataset frame or a recorded decision state restored), `reset` (a `reset_sampled` start), `rollout` (the state after a chunk executed from another frame of the shard) |
| `parent_frame` | for a rollout frame, the shard-local frame its chunk ran from; −1 otherwise |
| `step` | chunks executed since the last restore or reset (0 for restore / reset frames) |
| `chunk_dct`, `chunk_scale` | J and s of the random DCT part of the chunk that produced a rollout frame: a smooth random chunk, or the noise offset of a noisy π candidate; 0 and 0.0 for π's mean chunk, a recorded chunk, and frames that are not rollout frames |
| `chunk_switched` | the smooth random chunk switched the gripper command; False otherwise |
| `candidate` | index of the π candidate (ProposalCandidates order, 0 = π's mean chunk) whose execution produced the frame (resim sweep, pi-noise); −1 otherwise |

`cache_row`, `raw_demo_index`, `raw_step` are −1 except on restored dataset frames.

`SceneSidecar.read` checks row counts, `frame` = 0 … n − 1 and per-frame shapes; the provenance keys must be all present or all absent (`provenance` is None for the demo_renders sidecars). `write` refuses an existing file. Reading `demo_renders/heldout_task00/scene.npz` (594 frames, no provenance) and writing it back gives the same keys, arrays and dtypes; `SceneSidecar.of_rows` round-trips.

## collect_interactions.py

Common to the collecting subcommands:

- One shard per process or task (`TransitionStoreWriter`, format unchanged; refused if it exists). Frames: frozen DINOv2 pooled tokens [2, 64, 768] float16 (`ObservationEncoder`, each view on its own) and raw proprio [9].
- Chunks run with `LiberoTaskEnv.rollout` (every action; no stop at success). `goal_after` is LIBERO's check after the last action.
- After every frame, the scene is read at the same state: `SceneStateReader`, `SceneProbe.goal_holds`, `LiberoTaskEnv.snapshot`.
- Checks file `<analysis-dir>/<shard>_checks.npz` (`CollectionChecks`), one row per frame:
  - timings: restore, reset / rollout, encode, scene read;
  - for restored frames with a reference: max |proprio − reference|, proprio bytes equal (recorded decisions only), token cosine against the feature cache (dataset frames only);
  - for frames whose chunk ran from a fresh restore: whether that restore reproduced the state's first restore (proprio and both images);
  - refusals.
- Refusal: an observation that `LiberoTaskEnv.observe` refuses (black render) inside a restore, reset or rollout. The frame and its transitions are not written; the refusal goes to the checks file and the manifest.
- Manifest provenance: `CollectionConfig` (command, commit, host, GPU, stage one, caches, split), the subcommand's settings (sampler settings with the controller range), per-episode or per-start summaries (including seconds), the sidecar size, refusals.

| Subcommand | States | Transitions (kind) | `episode` / `t` | `sigma` | Shard |
|---|---|---|---|---|---|
| `demo-branch` | `--split` episodes (stage-one `EpisodeIndex`: heldout fraction 0.1, seed 0), `--shard i --num-shards n`; per episode `DemoStateRestorer` replay, branch points `--states-per-episode N` (distinct t uniform on 0 … T − 1, `default_rng([seed, 0, episode])`) or `--fractions` (t = round(f · (T − 11)), the rule of `check_dynamics_counterfactuals.py`) | `random_branch`: `--branches K` chunks `sample(K, command at t, (1, episode, t, 0))`, each from a fresh `restore(t)`; `random_walk`: a fresh `restore(t)`, then `--walk-length L` chunks in sequence, chunk ℓ = `sample(1, previous command, (2, episode, t, ℓ))`, sharing frames | dataset episode id / branch point t (sidecar `step` gives the walk step) | s | `<split>_shard_<i>_of_<n>` |
| `init-random` | per task of `--tasks`, `reset_sampled` starts e = 0 … `--starts` − 1 in order, env seed `--env-seed` (default 100) | `init_walk`: `--walk-length L` chunks in sequence, chunk ℓ = `sample(1, previous command, (3, task, e, ℓ))` | start e / ℓ (from 1); sidecar t = chunks executed | s | `task<TT>_sampled_<first>-<last>` |
| `resim-branch --mode mean` | every recorded decision of `outputs/analysis/branch_test/<--name>/record/task<TT>` (all 50 inits, or `--inits`), restored with `LiberoTaskEnv.restore(recorded snapshot, producing action)` | `resim_mean`: decision d → decision d + 1 when both frames exist and chunk d ran 10 steps; actions = the recorded π-mean chunk; o′ = the recorded state of d + 1; `episode_success` = the re-run episode's success | benchmark init index / decision d | 0 | `<name>_mean_task<TT>[_init<a>-<b>]` |
| `resim-branch --mode sweep` | decisions d ≡ 1 (mod `--decision-stride` 5) of the failed episodes with stored candidates (`StoredBranches`) | `resim_candidate`: each of the 32 stored candidates from a fresh restore; sidecar `candidate` = its index, `chunk_scale` / `chunk_dct` its offset's scale and J (the branch passes' settings) | benchmark init index / decision d | candidate's offset scale (0 for π's mean chunk, then 0.05 / 0.1 / 0.2 cycled) | `<name>_sweep_task<TT>[_init<a>-<b>]` |
| `pi-noise` | per task of `--tasks`, `reset_sampled` starts e = 0 … `--starts` − 1 in order, env seed `--env-seed` (default 2) | `pi_noise`: `--decisions` (30) decisions per start whatever the success; at decision d, `ProposalCandidates` of π on the encoded current frame, then one candidate chosen uniformly and executed with `rollout` (next section); frames chained | start e / d (from 1); sidecar t = decisions executed | the chosen candidate's noise scale (0 for π's mean chunk) | `pi_task<TT>_sampled_<first>-<last>` |

- `episode_success` is False except under `resim-branch --mode mean` and `pi-noise` (LIBERO's success at some step of the episode).
- The ψ name (fixed / flow / flow_full_sgd) is the shard-name prefix and `psi_name` in the manifest, with the record's stage one and π.
- **Held-out rules** (`HeldoutRule`, `docs/interaction_training.md`): demo-branch HELDOUT shards by prefix `heldout`; resim by `episode` 40–49; pi-noise by `episode` 80–99, the sampled start index as in rollouts_r0; init-random `episode` is the start index.
- **Evaluation set**: `--split heldout --fractions 0.1 0.4 0.7 0.25 0.55 0.85 --branches 32`. Over all 43 HELDOUT episodes:
  - f ∈ {0.1, 0.4, 0.7} gives 129 states, exactly the (episode, t) states of `consistency_candidates`;
  - the six fractions give 258 distinct states (shortest episode T = 79).
- **Phase-26 sizes** (`tmp/s4p4/inspect_inputs.py`):
  - records: 7,360 / 6,507 / 6,482 decisions (fixed / flow / flow_full_sgd), 20,349 in all, with 18,849 (d, d + 1) pairs;
  - sweep decisions with stored candidates: 516 / 288 / 282, 1,086 in all.

`report --store S --analysis-dir A --report-file F [--normalization-comparison]` (CPU) prints the checks below.

**Chunk coverage reference.** The report measures the Euclidean distance, in standardized units, from each chunk to the nearest TRAIN demonstration chunk (every t ≤ T − 10 of the TRAIN episodes, 44,091 chunks), on the 6 arm dims × 10 steps (60-d) or on all 70 dims. The reference is the same distance for the 4,907 HELDOUT demonstration chunks:

| | Median | p95 |
|---|---|---|
| arm dims (60-d) | 1.73 | 3.39 |
| all dims (70-d) | 1.78 | 3.53 |

## pi-noise: noisy π rollouts

`pi-noise` runs π's candidates directly, without the closed-loop runner:

- **π**: `--proposal-checkpoint`, default the no-ψ π `goal_proposal_nopsi_seed0` step 3000; a π that takes ψ or q_l is refused. Its feature cache must be the encoder's and its action cache `--action-cache`.
- **Conditioning**: the task's original instruction (the first `InstructionTable` variant) as cached SigLIP tokens of π's feature cache, as in the closed loop. The tokens are π's input only; the store holds frames, actions and labels.
- **Candidates** at decision d: `ProposalCandidates` with K = `--candidates` (8): π's mean chunk, then 7 noisy candidates with scales `--proposal-scales` (0.5, 1, 2) cycled from candidate 1 and `--proposal-dct` J = 3, seed `--proposal-seed` 0, no gripper flip, on the closed loop's stream `NoiseStream.episode(task, 1 (sampled), e, d)`.
- **Choice**: uniform over the K candidates, `default_rng([--choice-seed, 4, task, e, d])`, one draw per decision.
- **Execution**: `rollout` of the chosen chunk (no stop at success), `--decisions` 30 per start (300 env steps after 10 settle steps).
- **Recording**: the start frame (origin `reset`), then each o′ (origin `rollout`, step d; sidecar `candidate` = the chosen index, `chunk_scale` = its scale, `chunk_dct` = 3, or 0 for π's mean chunk); transitions from each frame to the next.

`run_closed_loop.py --strategy random --candidate-source proposal` is refused for every `--effect-source` by its argument checks and by `ClosedLoopRunner` (`tmp/s4p4/pi_random_check.sh`):

| `--effect-source` | Refusal |
|---|---|
| none | "--effect-source none leaves the candidates unscored; only --strategy mean chooses without scores" |
| observed | "--candidate-source proposal has no observed effects" |
| predicted (flow Δ) | "--effect-source predicted requires --strategy match or mean" |

The closed loop with the same generator under `--strategy mean` (`--escalation 8 --proposal-scales 0.5,1,2 --proposal-dct 3 --effect-source none --start sampled --inits 0-3 --record-transitions --trace`, task 3; `tmp/s4p4/pi_mean_plumbing.sh`; `outputs/closed_loop/trial_pi_mean_plumbing/task3_mean`, store `outputs/transitions/trial_pi_mean_plumbing`):

- 4 sampled starts, all successful; 31 recorded transitions (`rollout_proposal`) join the trace's decisions on (task, episode = sampled start, t = decision number) 31 / 31, with equal gripper commands, chosen = π's mean chunk, K = 8, consecutive decisions sharing the frame 27 / 27;
- the trace's `SceneSnapshot` has object positions, fixture joints, `goal_holds`, end-effector position, gripper qpos and grasped objects, and no object orientations, arm joint qpos, `sim_state` or fixture poses; the store's `sigma` is 0 for every closed-loop transition;
- 15.3 ms per decision (encode 13.1, π 2.2), 16.6–17.8 ms per env step.

## Trial

- **Host**: node-61, 2 L40S (CUDA 0 and 1), idle before every launch; `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = `OPENBLAS_NUM_THREADS` = 2.
- **Round B** (current code, commit `abeae59`): the stores `outputs/transitions/trial2_*`, eight processes at once, four per GPU (`tmp/s4p4/launch_trial_b.sh`, `after_trial_b.sh`).
- **Round A** (commit `5d389f2`, the same commands without pi-noise, six processes, three per GPU; `launch_trial_a.sh`): stores `trial_*`. Every value that the round-A reports print is reproduced in round B, except the timings. The round-A sidecars and those of the throughput runs have no `chunk_scale` column, so the current `SceneSidecar.read` refuses them.
- **Throughput runs** (commit `5d389f2`; `launch_throughput.sh`, `throughput_driver.sh`): demo-branch TRAIN, one episode per process (one per task: 2, 44, 75, 114, 165, 214, 261, 301, 338, 383), 8 × (8 + 4) = 96 transitions each, P processes per GPU on both GPUs (`trial_throughput_{1,3,5}pgpu`).
- **Outputs**: checks and reports `outputs/analysis/interaction_collection/<store>/` (`report.txt`); logs `tmp/runlogs/s4p4/`.

Round B:

| Store / shard | Command | GPU | Transitions | Frames | Refusals |
|---|---|---|---|---|---|
| `trial2_demo_branch/train_shard_00{0,1}_of_002` | TRAIN episodes 1, 43, 164, 262; 8 branch points × (8 branches + 4-chunk walk) | 0, 0 | 192 + 192 | 208 + 208 | 0 |
| `trial2_demo_branch_eval/heldout_shard_000_of_001` | HELDOUT episodes 4, 131; 6 fractions × 32 branches | 0 | 384 | 396 | 0 |
| `trial2_init_random/task0{2,6}_sampled_000-009` | tasks 2, 6; 10 starts × 20 chunks; env seed 100 | 1 | 200 + 200 | 210 + 210 | 0 |
| `trial2_resim_branch/fixed_mean_task03_init30-49` | fixed ψ, task 3, inits 30–49 | 1 | 261 | 281 | 0 |
| `trial2_resim_branch/flow_sweep_task06_init04-05` | flow ψ, task 6, inits 4, 5: 12 states × 32 | 1 | 384 | 396 | 0 |
| `trial2_pi_noise/pi_task0{3,8}_sampled_000-004` | tasks 3, 8; 5 starts × 30 decisions; env seed 2 | 0, 1 | 150 + 150 | 155 + 155 | 0 |

### Sidecar ↔ store

For every shard (the 9 round-B shards, the 7 round-A shards and the 18 throughput shards):

- sidecar rows = shard frames = checks rows, `frame` = 0 … n − 1;
- restored dataset frames: `cache_row` = `frame_rows(episode, t)` for every branch point (16 + 16 TRAIN, 12 HELDOUT, 8 per throughput shard);
- |store proprio − sidecar `joint_qpos`|: median 6.0e-4 to 1.08e-3 per shard, max ≤ 5.8e-3, the observation-vs-qpos lag of `docs/demo_renders.md` (median 8.9e-4, max 5.4e-3). The same comparison with the sidecar shifted by one row gives medians 2.9e-2 to 2.7e-1.

### Chained frames

The rows name the round-A stores; the round-B stores give the same counts.

| Store | Transitions with sim time o′ − o = 0.5 s (10 steps) | o′ from a rollout: parent = o, step + 1 | Walk continuation (o′ of step ℓ = o of step ℓ + 1) | Gripper command = previous one unless switched |
|---|---|---|---|---|
| trial_demo_branch | 384 / 384 | 384 / 384 | `random_walk`: 32 chain starts + 96 / 96 | 96 / 96 (34 switched) |
| trial_demo_branch_eval | 384 / 384 | 384 / 384 | — | — |
| trial_init_random | 400 / 400 | 400 / 400 | `init_walk`: 20 chain starts + 380 / 380 | 380 / 380 (105 switched) |
| trial_resim_branch | 645 / 645 | 384 / 384 (sweep) | `resim_mean`: 20 chain starts + 241 / 241 (o′ = restored next decision) | — |
| trial2_pi_noise | 300 / 300 | 300 / 300 | `pi_noise`: 10 chain starts + 290 / 290 | — |
| throughput P = 1 / 3 / 5 | 192 / 576 / 960, all | all | all | all |

The largest |Δtime − 0.5 s| is 1.7e-13.

### Restore fidelity

| Frames | Reference | max \|proprio − reference\| | Other |
|---|---|---|---|
| demo branch points, TRAIN (32) | dataset proprio | 2.4e-7 (median 1.6e-8) | token cosine vs feature cache: agent mean 0.971 (min 0.965), wrist 0.936 (min 0.864) |
| demo branch points, HELDOUT (12) | dataset proprio | 6.9e-8 | agent 0.969, wrist 0.939 |
| throughput branch points (16 / 48 / 80) | dataset proprio | 9.5e-8 / 1.2e-7 / 2.4e-7 | — |
| resim decisions (293) | recorded proprio | 3.7e-9 | bit-identical 286 / 293 |
| HELDOUT branch points in `consistency_candidates` (6) | its stored frame | 0 (6 / 6 bit-identical) | token cosine ≥ 0.999996 |

Fresh restores before a chunk reproduce the state's first restore in 288 / 288 (TRAIN), 384 / 384 (HELDOUT), 384 / 384 (sweep), and 144 / 144, 432 / 432, 710 / 720 in the throughput runs. The 10 exceptions are in task 7, episode 301 (9), and task 8, episode 338 (1): the render nondeterminism of `docs/libero_sim.md`.

### Smoothness and coverage of the chunks

DCT energy share of the standardized arm dims by coefficient 0 … 5 (the rest < 0.001):

| Chunks | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| TRAIN demonstrations | 0.856 | 0.121 | 0.016 | 0.004 | 0.001 | 0.001 |
| `random_branch` (TRAIN trial) | 0.616 | 0.276 | 0.107 | 0.001 | 0.000 | 0.000 |
| `random_walk` | 0.589 | 0.299 | 0.111 | 0.001 | 0.000 | 0.000 |
| `init_walk` | 0.633 | 0.257 | 0.109 | 0.001 | 0.000 | 0.000 |
| `resim_candidate` (π, flow) | 0.828 | 0.141 | 0.023 | 0.003 | 0.002 | 0.001 |
| `resim_mean` (π, fixed) | 0.746 | 0.222 | 0.024 | 0.005 | 0.001 | 0.001 |
| `pi_noise` (no-ψ π, J = 3) | 0.543 | 0.264 | 0.186 | 0.004 | 0.002 | 0.001 |

- **Energy above J** (all random-chunk stores): with J = 1 the energy share of coefficients ≥ J is ≤ 4.2e-31. With J = 2 or 3 its median per store is at most 4.8e-4 and its maximum 5.5e-2; without clipping it is ≤ 1.9e-15 (previous section), so the nonzero values come from clipping.
- **Per-entry RMS** of the executed random chunks (round-A stores), median at s = 0.5 / 1 / 2: 0.46–0.49 / 0.94–0.97 / 1.67–1.78.
- **Clipping**: share of arm entries at the ±1 limit 0 / 0.004–0.013 / 0.095–0.109.
- **Gripper switch rate**: 0.301 (TRAIN branches), 0.352 (walks), 0.326 (evaluation), 0.268 (init-random).

| Kind (store) | n | Arm distance to nearest TRAIN demo chunk, median / p95 | Share beyond the HELDOUT p95 (arm 3.39 / all dims 3.53) |
|---|---|---|---|
| `random_branch` (trial_demo_branch) | 256 | 4.56 / 10.9 | 0.668 / 0.703 |
| `random_walk` (trial_demo_branch) | 128 | 4.12 / 12.5 | 0.617 / 0.641 |
| `random_branch` (trial_demo_branch_eval) | 384 | 4.32 / 12.1 | 0.635 / 0.669 |
| `init_walk` | 400 | 4.74 / 12.3 | 0.660 / 0.695 |
| `random_branch` / `random_walk` (throughput P = 5) | 640 / 320 | 4.58 / 4.41 | 0.664 / 0.656 (arm) |
| `resim_candidate` | 384 | 1.79 / 2.70 | 0.000 / 0.104 |
| `resim_mean` | 261 | 1.51 / 2.39 | 0.031 / 0.057 |
| `pi_noise` | 300 | 4.34 / 11.9 | 0.707 / 0.760 |

### Effects (o → o′ from the sidecar)

| Kind (store) | Object moved > 1 cm | Fixture joint changed (1 cm / 0.05 rad) | Grasp changed | Goal holds after |
|---|---|---|---|---|
| `random_branch` (trial_demo_branch, tasks 0, 1, 4, 6) | 0.246 | 0.070 | 0.129 | 0.094 |
| `random_walk` (trial_demo_branch) | 0.180 | 0.055 | 0.102 | 0.086 |
| `random_branch` (throughput P = 5, tasks 0–9) | 0.342 | 0.052 | 0.136 | 0.100 |
| `random_walk` (throughput P = 5) | 0.300 | 0.047 | 0.091 | 0.091 |
| `random_branch` (evaluation, tasks 0, 3) | 0.242 | 0.023 | 0.133 | 0.052 |
| `init_walk` (tasks 2, 6) | 0.070 | 0.000 | 0.000 | 0.000 |
| `resim_candidate` (flow, task 6) | 0.083 | 0.000 | 0.083 | 0.000 |
| `resim_mean` (fixed, task 3) | 0.337 | 0.000 | 0.088 | 0.000 |
| `pi_noise` (tasks 3, 8) | 0.210 | 0.000 | 0.040 | 0.043 |

A π-mean chunk that reaches the goal ends its episode before 10 steps, so no `resim_mean` transition ends at the goal.

**pi-noise checks** (`tmp/s4p4/check_pi_noise.py`, CPU): for each of the 300 transitions, π on the stored current frame with the same stream and the choice generator gives the stored chunk within 7.0e-4 (the stored chunks come from π on the GPU); the recomputed choice equals the sidecar's `candidate` 300 / 300; the candidate's scale equals `sigma` and `chunk_scale` 300 / 300, `chunk_dct` 300 / 300. Chosen candidates 0 … 7: 44, 43, 33, 38, 38, 31, 37, 36 (the executed scales: 0 for 44 chunks, 0.5 for 117, 1 for 64, 2 for 75). One of the 10 episodes reaches success (task 8, start 4, at decision 18) and continues to decision 30.

### Timing and disk

Per-frame medians under the round-A and round-B loads:

| Step | Median |
|---|---|
| restore | 27.5–34.5 ms |
| rollout of 10 actions, or `reset_sampled` | 153–174 ms |
| encode | 13.8–17.1 ms |
| scene read (reader, probe, snapshot) | 3.2–3.9 ms |

Collection time per transition, from the per-episode or per-start seconds (DemoStateRestorer replay and π included); process start-up (models, env builds) excluded:

| Source | s / transition, round A | round B | Shard wall time outside the episodes / starts (env builds; demo-branch also model loading) |
|---|---|---|---|
| demo-branch TRAIN (8 × (8 + 4) per episode) | 0.246, 0.271 | 0.251, 0.273 | 9.2–10.7 s per shard |
| evaluation set (6 × 32 per episode) | 0.236 | 0.239 | 9.4–9.8 s |
| init-random (10 × 20 per task) | 0.197, 0.198 | 0.195, 0.190 | 1.6–6.3 s per task |
| resim mean (restore + encode per decision) | 0.057 | 0.063 | 5.8–6.0 s |
| resim sweep (32 per state) | 0.214 | 0.234 | 5.0–6.0 s |
| pi-noise (30 decisions per start) | — | 0.204, 0.206 | 5.9–6.0 s per task |

Scaling with processes per GPU (demo-branch TRAIN, 96 transitions per process):

| Processes per GPU | Processes | s / transition per process (mean over shards) | Range over shards | Node throughput (transitions / s) | GPU utilization (peak sample) | Load average (peak) |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.280 | 0.245–0.316 | 7.1 | — | — |
| 3 | 6 | 0.272 | 0.250–0.318 | 22 | — | ≤ 6.2 |
| 5 | 10 | 0.283 | 0.259–0.329 | 35 | 57 % / 71 %, 7.4 GiB per GPU | 6.2 |

The per-process rate is unchanged from 1 to 5 processes per GPU; the node throughput grows in proportion to the number of processes. The slowest shard in every run is the one with episode 44 (T = 299, the longest replay).

**Disk**:

- 193.9 KiB per frame: tokens 192.0 KiB, the transition table, and a sidecar of 1.40 KiB per frame.
- Per transition: 210.0 KiB (demo-branch with walks), 199.9 KiB (32 branches per state), 203.5 KiB (init-random), 208.6 KiB (resim mean), 200.3 KiB (pi-noise).
- Checks files and reports: under 1 MiB for all trial stores.

## Estimate for Phase 6

Volumes from plan.md; pi-noise at 100 sampled starts × 10 tasks × 30 decisions. Per-transition times are the means of rounds A and B above, except demo-branch, which uses the mean of the P = 3 throughput run (0.272 s). Disk uses 193.9 KiB per frame and each source's frames per transition.

| Source | Transitions | Process time | Disk |
|---|---|---|---|
| demo-branch TRAIN: 390 episodes × 8 × (8 + 4) | 37,440 | 10,184 s | 7.5 GiB |
| evaluation set: 43 × 6 × 32 | 8,256 | 1,948 s | 1.6 GiB |
| init-random: 10 tasks × 20 starts × 20 | 4,000 | 780 s | 0.8 GiB |
| resim mean: 3 ψ × 10 tasks, 20,349 restores | 18,849 | 1,131 s | 3.8 GiB |
| resim sweep: 1,086 states × 32 | 34,752 | 7,784 s | 6.6 GiB |
| pi-noise: 10 tasks × 100 starts × 30 | 30,000 | 6,150 s | 5.7 GiB |
| **total** | **133,297** | **27,977 s** | **26.0 GiB** |

At 25,000 pi-noise transitions (about 83 starts per task) pi-noise takes 5,125 s and 4.8 GiB, and the total is 26,952 s and 25.1 GiB.

Wall time for 2 nodes × 2 GPUs:

- 3 processes per GPU (12 processes): about 39 min;
- 5 processes per GPU (20 processes): about 23 min.

Each process adds about 10 s of start-up and 5–10 s per env build. The largest single shards:

- resim sweep for fixed task 4: 120 states × 32 = 3,840 transitions, about 14 min;
- pi-noise, one process per task: 3,000 transitions, about 10 min.

These figures assume the second node runs at node-61's L40S rates.

## Commands

```bash
source scripts/ceam_env.sh   # CEAM_ROOT = main checkout, OMP/MKL/OPENBLAS_NUM_THREADS=2 with several processes
$CEAM_PYTHON scripts/collect_interactions.py demo-branch --cuda-device 0 --code-commit <sha> \
  --out-dir outputs/transitions/<store> --analysis-dir outputs/analysis/interaction_collection/<store> \
  --split train --shard <i> --num-shards <n> --states-per-episode 8 --branches 8 --walk-length 4
$CEAM_PYTHON scripts/collect_interactions.py demo-branch ... --split heldout \
  --fractions 0.1 0.4 0.7 0.25 0.55 0.85 --branches 32
$CEAM_PYTHON scripts/collect_interactions.py init-random ... --tasks <t …> --starts 20 --walk-length 20
$CEAM_PYTHON scripts/collect_interactions.py resim-branch ... --name <fixed|flow|flow_full_sgd> --mode <mean|sweep> --tasks <t …>
$CEAM_PYTHON scripts/collect_interactions.py pi-noise ... --tasks <t> --starts 100 --decisions 30
$CEAM_PYTHON scripts/collect_interactions.py report --store <store dir> --analysis-dir <dir> --report-file <file>
```

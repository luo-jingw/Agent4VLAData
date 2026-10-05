# Stage-one training

`ceam/training/`: `StageOneConfig` / `OptimizerConfig` (`stage_one_config.py`), `CheckpointStore` / `LoadedCheckpoint` (`checkpoint_store.py`), `StageOneTrainer` / `TrainingSummary` / `TrainingMetrics` (`stage_one_trainer.py`). Entry: `scripts/train_stage_one.py --config <json> --device cuda`.

## Config

JSON with exactly the dataclass fields at every level (missing or unknown keys raise `ValueError`). Path strings may use `$CEAM_ROOT` and `$CEAM_DATA_ROOT`, expanded on read; `to_json` writes paths under the current roots back as these variables, so a run's `config.json` resolves on any cluster node after `source scripts/ceam_env.sh`.

## Step

1. Sample a `StageOneBatch` (TRAIN split).
2. `StateTrunk` once on all B·(3 + 2M) frames (start, end, partner end, uniform, uniform future) → `StageOneStates`.
3. `QueryModule` on text_a and on text_b (reference = start frames under `TEXT_AND_OBSERVATION`), and on `text_all` (one random variant per task) for the instruction contrast: once per step under `TEXT_ONLY`; under `TEXT_AND_OBSERVATION` for every (b, m, task) with the uniform frame o_t as reference, plus text_a on o_t for the own direction.
4. `StageOneObjective` (float32) with `batch.task_id`, the own direction and the negative mask from `TaskRelationTable.from_json(config.task_relations_path).negative_mask(index.task_texts())`; trunk and query forward under bf16 autocast when `autocast_bf16`.
5. AdamW (weight decay only on ≥ 2-D weights whose name contains neither "token" nor "embedding"), global grad-norm clipping, linear warmup then cosine decay to 0 at `total_steps`.

`metrics.jsonl` gets one `TrainingMetrics` line at step 0, every `log_every` steps and the last step: step (0-based update index, losses before that update), lr, each loss term, grad norm before clipping, steps/s, sampler seconds per step, peak memory. `checkpoints/step_{N:06d}.pt` holds the parameters after N updates. An existing `run_dir` is refused.

## Measured

`configs/stage_one/smoke.json` (LIBERO-Goal, TEXT_ONLY, 200 steps, B = 64, M = 8, bf16 autocast, shared H100): 5.5–10 steps/s depending on other users' load, 7.7 GiB peak memory. Checkpoint save → load is bit-exact; `from_json(to_json(config)) == config`. A 20-step TEXT_AND_OBSERVATION run trains without error.

Configs written before `task_relations_path` and `weights.task_contrast` existed (`outputs/runs/smoke_goal_text_only`, `smoke_goal_text_obs_20`) no longer load through the strict `from_json`.

Smoke with task contrast (`configs/stage_one/smoke.json`, weight 1, 200 steps): the term goes from 3.13 to 0.004; path, goal, endpoint and paraphrase end at 8.7e-3, 4.2e-3, 2.3e-3, 1.8e-3 (without the term: 1.2e-4, 2.1e-4, 3.6e-4, 5.8e-4); same peak memory.

The sampler draws uniform frames t ∈ {0, …, T−2} and horizons k uniform on {1, …, T−1−t}. `TrainingMetrics` logs `instruction_contrast` and `logit_scale` (as s = exp(parameter)). The learnable log scale moves at most about one learning rate per AdamW step: over 200 steps s went 14.29 → 14.35.

Run directories from before the instruction contrast (`ab_goal_contrast*`, `smoke_goal_contrast1`, `smoke_goal_text_*`) hold configs that no longer load.

Smoke with the instruction contrast (`configs/stage_one/smoke.json`, 200 steps, shared H100): the contrast goes 2.42 → 0.25; 3–5 steps/s, 12.9 GiB peak under `TEXT_ONLY`; `TEXT_AND_OBSERVATION` (5120 + 512 query evaluations per step) trains at 1.0–2.2 steps/s with 28.0 GiB peak.

## Stage two (dynamics head)

`ceam/training/stage_two_config.py` (`StageTwoConfig`), `stage_two_checkpoint_store.py` (`StageTwoCheckpointStore`, `LoadedStageTwo`), `stage_two_trainer.py` (`StageTwoTrainer`, `StageTwoMetrics`, `StageTwoSummary`); sampler `ceam/data/stage_two_sampler.py` (`StageTwoSampler`, uniform over valid (episode, t) with t + k ≤ T − 1, with replacement). Entry: `scripts/train_stage_two.py --config <json>`. Feature cache, dataset root, split and index seed come from the referenced stage-one checkpoint's config. The trainer freezes stage one, computes ψ for every feature-cache row once (52895 × 256 in 12.6 s), and regresses Δ(o_t, a_{t:t+k}) to ψ(o_{t+k}) − ψ(o_t) with MSE; metrics: MSE, median ‖Δ̂ − z‖/‖z‖, mean cos(Δ̂, z).

Smoke (`configs/stage_two/smoke.json`, 200 steps, batch 256, shared H100): 16.8 steps/s, 3.4 GiB peak; relative error 1.00 → 0.35, cos(Δ̂, z) 0 → 0.93; all 112 stage-one tensors unchanged; checkpoint round trip exact.

Short training (`configs/stage_two/goal_dynamics_short.json`, 3000 steps, warmup 150, shared H100): 18 steps/s, 3.4 GiB peak; on TRAIN batches the relative error goes 1.00 → 0.257 (step 500) → 0.128 (step 3000) and cos(Δ̂, z) 0 → 0.952 → 0.986. Checkpoint used by the closed loop: `outputs/runs/goal_dynamics_short/checkpoints/step_003000.pt`. HELDOUT prediction error is not measured.

### Mixed demonstration and simulated transitions

`StageTwoConfig.transition_sources` lists `TransitionSourceConfig(store_dir, fraction)` entries.

- **Reading**: optional in JSON (missing = none), so older configs and run directories load unchanged; always written.
- **Batch composition**:
  - source i contributes n_i = round(B·f_i) transitions;
  - the demonstrations contribute B − Σ n_i;
  - one concatenated forward pass, demonstrations first;
  - loss unchanged: MSE to ψ(o′) − ψ(o) over the whole batch.
- **Sampling** (`TransitionSampler`): uniform over the store's transitions with replacement.
  - Seed `(config.seed, i + 1)` per source; `default_rng(s)` equals `default_rng([s, 0])`, so no source repeats the demonstration stream.
  - Action chunks standardized with the action-cache statistics.
  - Proprio standardized with the feature-cache statistics.
- **ψ tables**: one per store, computed once with the frozen trunk.
- **Metrics**: `metrics.jsonl` adds `sources: {name: {transitions, mse, relative_error, effect_cosine}}`, with "demo" always present.
- **Checks**:
  - With no sources the trainer is bit-identical to the previous one over all 3000 steps (61 logged steps, 64 checkpoint tensors; node-63 L40S). The recorded H100 run differs only by GPU (step 2999: mse 0.001348 vs 0.001330).
  - At step 0 (Δ̂ = 0) each part's MSE equals the mean z² recomputed offline (demo 0.15860, data A 0.29866).
  - `in_memory_cache` false gives bit-identical results.
- **Cost**:
  - Peak GPU memory 3.39 GiB at B = 256.
  - Opening data A in memory: 62 s (6.86 GiB).
  - The ψ table of data A (37,440 frames): 40–88 s on L40S.
  - Speed is bound by host-side gathering: 9.6–10.6 steps/s mixed against 11–12.7 demo-only in quiet periods on node-63.
- **Configs**: `configs/stage_two/goal_dynamics_{perturb,rollout,both}.json` differ from `goal_dynamics_short.json` only in `run_dir` and the sources:
  - perturb: data A at 0.5;
  - rollout: data B round 0 at 0.5;
  - both: 0.25 + 0.25.

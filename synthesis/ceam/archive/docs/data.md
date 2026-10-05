# Data

## Source datasets

LIBERO in LeRobot v2.1 format, read-only, under
`$LOCAL_HOME/workspace/jingwu/data/libero_mujoco3.3.2/<suite>_no_noops_lerobot`.

| Field | Value |
|---|---|
| Frame rate | 20 fps |
| Cameras | `observation.images.image` (agent view), `observation.images.wrist_image`; 512×512 AV1 mp4, one file per episode |
| Proprio used | `observation.states.joint_state` (7) + `observation.states.gripper_state` (2) = 9 |
| Action | 7: end-effector delta xyz, rpy, gripper command |
| Parquet | one file per episode, `data/chunk-{chunk:03d}/episode_{id:06d}.parquet`, `frame_index` is 0..T−1 |

## Module

- `ceam/data/episode_index.py` — `EpisodeIndex` lists episodes from `meta/tasks.jsonl` and `meta/episodes.jsonl`; `task_id` is the dataset `task_index`. Each episode has exactly one task. TRAIN / HELDOUT is drawn per task with one seeded permutation: `round(fraction · n)` episodes of each task are HELDOUT.
- `ceam/data/lerobot_episode_reader.py` — `LeRobotEpisodeReader.read` returns all frames of one episode as `EpisodeArrays`. It raises `ValueError` if `frame_index` is not 0..T−1 or the decoded frame count differs from the episode length.
- `ceam/data/frame_batch.py`, `ceam/data/text_tokens.py` — shared tensor containers `FrameBatch` and `TextTokens`.
- `ceam/data/instruction_table.py` — `InstructionTable` reads `data/instructions/<suite>.json`: per task, `variants[0]` is the dataset task text and `variants[1..4]` are hand-written paraphrases with the same objects, spatial reference and target. Construction raises `ValueError` if `variants[0]` differs from the task text, a task has fewer than 2 variants, or a task text or variant repeats (within or across tasks).

## Measured

`scripts/inspect_dataset.py`, heldout fraction 0.1, seed 0:

| Suite | Tasks | Episodes (train / heldout) | Frames | Length min / median / max | Episode 0 read time |
|---|---|---|---|---|---|
| libero_goal | 10 | 390 / 43 | 52895 | 75 / 105 / 347 | 1.44 s (138 frames) |
| libero_spatial | 10 | 391 / 43 | 53229 | 75 / 123 / 193 | 0.74 s (98 frames) |

Two builds with the same seed give identical splits. Every task has at least 30 TRAIN episodes.

Instruction tables: 10 tasks × 5 variants per suite, 50 distinct strings each; `variants[0]` matches `meta/tasks.jsonl` for every task of both suites.

## Feature cache

`ceam/data/feature_cache_builder.py` (`FeatureCacheBuilder`, `FeatureCacheProvenance`, `FeatureCacheSummary`) encodes every frame of every episode (TRAIN and HELDOUT, ordered by episode id, each view in 64-frame chunks) and every instruction string once; `ceam/data/feature_cache.py` (`FeatureCache`) reads it. `ceam.data` exports `FeatureCache` only: the builder depends on `ceam.encoders`, which depends on `ceam.data`. Build: `scripts/build_feature_cache.py --suite <suite> --out-dir <dir>`; an existing `out_dir` is refused.

Layout of `outputs/feature_cache/<suite>/`: `visual_tokens.npy` float16 [N, 2, 64, 768] (view 0 agent, view 1 wrist), `proprio.npy` float32 [N, 9] raw, `proprio_stats.json` (per-dim mean, std floored at 1e-6, over all frames), `frame_table.npz` (`row_episode_id`, `row_t`, `episode_ids`, `episode_start_row`, `episode_length`), `text_tokens.npy` float16 [S, 64, 1152], `text_mask.npy` bool [S, 64], `strings.json`, `manifest.json` (provenance, shapes, dtypes, counts; written last). Row of (episode, t) = `episode_start_row[episode] + t`.

| Suite | Frames | Strings | Size | Build time (shared GPU) |
|---|---|---|---|---|
| libero_goal | 52895 | 50 | 9.7 GiB | 493 s (107 frames/s) |
| libero_spatial | 53229 | 50 | 9.8 GiB | 476 s (112 frames/s) |

Checks on both caches: cached visual tokens equal a re-encode of the same 64-frame chunk exactly and differ from a single-frame re-encode by ≤ 0.0078 (relative Frobenius ≤ 7.9e-5); raw proprio equals the reader's; 10000/10000 random (episode, t) round-trip through `frame_rows`; standardized proprio has mean 0 and std 1 per dimension; strings re-encoded in a different batch differ by ≤ 0.0078 with identical masks. Reading 704 random rows from the memory map: 215–285 ms with cold pages, 9–14 ms warm.

`FeatureCache.open(cache_dir, in_memory)` raises `FileNotFoundError` when `manifest.json` is missing (incomplete build). `in_memory=True` loads `visual_tokens.npy` into RAM (2.2 s with warm page cache, 93 s cold); the read-only memory map stalls for 1.7–3.3 s now and then on random rows even when mostly cached, so training uses `in_memory=True`.

## Stage-one sampler

`ceam/data/stage_one_sampler.py` (`StageOneSampler`, `SamplerConfig`) and `ceam/data/stage_one_batch.py` (`StageOneBatch`). Per batch, with a generator seeded at construction: B anchor episodes without replacement from one split; per anchor a partner episode of the same task and split; M uniform frames t ∈ {0, …, T−1}; two distinct instruction variants. Frames: start t = 0, end t = T − 1, partner_end t = T′ − 1, uniform [B, M]; all B·(3 + M) rows gathered in one call. Construction raises `ValueError` if B exceeds the split, a task has fewer than 2 episodes in the split, or fewer than 2 variants.

Measured on libero_goal TRAIN (B = 64, M = 8), 1000 batches: every structural check (start/end/partner rows, same-task partner of a different episode, uniform frames from the anchor, text_a ≠ text_b, variants of the anchor's task) holds for every sample; uniform t/T counts match the exact expectation; 0 HELDOUT frames. Sampling to GPU: median 31 ms in memory, 40 ms memory-mapped (warm).

## Task relations

`ceam/data/bddl_task_goals.py` (`BddlTaskGoals`) parses the `:init` and `(:goal (And …))` predicates of a LIBERO BDDL file; `ceam/data/task_relation_table.py` (`TaskRelationTable`, `TaskRelation`) reads `data/task_relations/<suite>.json`, built by `scripts/build_task_relations.py`. Two different tasks are not negatives of each other when one goal-predicate set contains the other or either goal holds at init; `negative_mask(task_texts)` returns the [n, n] negative matrix with a False diagonal.

- LIBERO-Goal: all 10 tasks share one `:init` section; the 10 goals are distinct single predicates, none holding at init; every pair of different tasks is a negative pair.
- LIBERO-Spatial: 10 distinct `:init` sections, all with the goal `(On akita_black_bowl_1 plate_1)`; no pair is a negative pair.

## Action cache

`ceam/data/action_cache_builder.py` (`ActionCacheBuilder`, `ActionCacheSummary`; not exported from `ceam.data`) and `ceam/data/action_cache.py` (`ActionCache`). Build: `scripts/build_action_cache.py --suite <suite> --feature-cache <dir> --out-dir <dir>`. Row r of the action cache is row r of the feature cache (`FeatureCache.frame_rows`); the builder refuses a feature cache whose `n_frames` differs from the index and checks every row is written once. `LeRobotEpisodeReader.read_actions` reads the parquet action column only.

Layout of `outputs/action_cache/<suite>/`: `actions.npy` float32 [N, 7] raw, `frames_to_episode_end.npy` int64 [N] (T − t), `action_stats.json` (per-dim mean, std floored at 1e-6, over all rows), `manifest.json` (dataset root, feature cache dir, N, episodes; written last). `ActionCache.chunks(rows, length)` returns standardized [..., length, 7] chunks and raises `ValueError` for chunks crossing an episode end, rows outside [0, N), non-integer rows or length < 1.

libero_goal: 52895 rows, built in 0.8 s; cached actions equal the parquet actions exactly for all 433 episodes. Raw per-dim std: 0.399, 0.351, 0.491, 0.055, 0.078, 0.100, 0.482; the gripper column takes {0, 1} (19332 / 33563 rows), standardized to {−1.318, 0.759}.

## Transition store

`ceam/data/transition_store.py` holds simulated transitions (o, a, o′) of chunk length 10: frames, transitions and their metadata, one directory per shard (`<root>/<shard>/`).

Shard files:

| File | Content |
|---|---|
| `visual_tokens.bin` | float16 [n, 2, 64, 768], appended frame by frame; view 0 agent, view 1 wrist, exactly as `ObservationEncoder` produces them |
| `proprio.bin` | float32 [n, 9], raw (`robot0_joint_pos` + `robot0_gripper_qpos`) |
| `transitions.npz` | One column per field of `TransitionRecord` (below) |
| `manifest.json` | Counts, shapes, dtypes, bytes and provenance; written last |

`TransitionRecord` fields:

- `current_frame`, `future_frame`: shard-local frame indices.
- `actions`: [10, 7], raw dataset convention, as executed.
- `kind`: "stay", "perturbed", "rollout", "rollout_stay".
- `sigma`, `task_id`, `episode`, `t`.
- `goal_after`: LIBERO's done after the last action.
- `episode_success`.

Writing and reading:

- **Writer**: `TransitionStoreWriter(out_dir, shard)` refuses an existing shard, accepts only [10, 7] chunks, and requires both frames to be already written.
- **Reader**: `TransitionStore.open(root, in_memory)`.
  - Global frame index: frames of the earlier shards (ordered by name) + local index.
  - `proprio(frames, mean, std)` standardizes like `FeatureCache.proprio`.
  - A shard without a manifest raises `FileNotFoundError`.
- **Size**: 196,644 B per frame, about 376 B per transition in the table.

`ceam/data/counterfactual_chunks.py` (`CounterfactualChunks`) builds the stay and perturbed chunks for both `scripts/check_dynamics_counterfactuals.py` and `scripts/collect_demo_perturbations.py`, so the two use one construction. It gives output identical to the phase-12 construction on 3444 candidates.

Seeding note: numpy seed sequences treat `[a, b]` and `[a, b, 0]` as the same seed, so the perturbation stream uses `[seed, episode_id, t, 1]`.

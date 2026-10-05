# LIBERO-90 no-ψ π baseline

## Protocol

Three independently trained no-ψ proposal models use 32,000 updates, batch size 256 and the same LIBERO-90 TRAIN
set. The policy conditions on the observation and original instruction. Closed-loop evaluation executes the
proposal mean, with one candidate and no r/q/m ranking: 90 benchmark tasks × 50 initial states per seed, environment
seed 0, ten settle steps and a 400-step limit. Evaluation tasks retain scene-specific benchmark identity; their
instructions contain 74 distinct texts across 20 scenes.

## Training and closed-loop results

Full-set checkpoint curves contain 479,578 TRAIN and 53,920 HELDOUT transitions per checkpoint.

| π seed | updates | final TRAIN MSE | final HELDOUT MSE | successes / episodes | SR |
|---:|---:|---:|---:|---:|---:|
| 0 | 32,000 | 0.1699 | 0.2810 | 4213 / 4500 | 93.62% |
| 1 | 32,000 | 0.1769 | 0.2807 | 4216 / 4500 | 93.69% |
| 2 | 32,000 | 0.1714 | 0.2785 | 4134 / 4500 | 91.87% |
| all | — | — | — | 12563 / 13500 | 93.06% |

The aggregation validates every benchmark task's identity, exactly 50 rows and unique initial-state indices 0–49.
Per-task, per-scene and per-seed results are in `outputs/analysis/libero90_baseline/seeds012.json` and `seeds012.md`.
The full checkpoint curves are in `outputs/analysis/proposals_libero90/curve_seed{0,1,2}/curve/summary.txt`.
HELDOUT MSE reaches its lowest recorded value at 24,000 updates for all three seeds and rises slightly afterward;
closed-loop evaluation uses the fixed final 32,000-update checkpoint.

## Query-independent stores

| store | complete task shards | frames | transitions | file bytes |
|---|---:|---:|---:|---:|
| `pi_rollouts_libero90` | 90 | 13,639 | 12,746 | 2,688,222,016 |
| `pi_noise_libero90` | 90 | 55,786 | 53,986 | 11,055,447,729 |

Every manifest's declared visual-token, proprio and transition file size matches the actual file size.
Both complete stores load through `TransitionStore.open`. π-noise has ninety loadable scene sidecars with exactly
55,786 aligned frames; π-mean uses the existing recorder's feature/transition-only contract. Sampled first/last
visual-token batches have shape `[2, 2, 64, 768]` and dtype float16. The measured load validation is in
`outputs/analysis/libero90_recovery_20260930/store_load_validation.json`.
Rollouts use proposal seed 0 and sampled starts 0–9; downstream HELDOUT episodes are 8–9. π-noise uses twenty sampled
starts per task, up to thirty decisions per start, eight candidates with scales 0.5/1/2 and DCT dimension 3;
downstream HELDOUT episodes are 16–19. Final counts report recorded valid transitions.

## Execution and resources

The initial full evaluation used GPU 1 on node-61, node-63, node-65 and node-66, with five task processes per card.
The node-66 resource log ends at 15:02:47 EDT and no project processes were present there during the 19:17 EDT check.
Eight seed-2 tasks were incomplete. Tasks 36 and 58 contained 45 and 40 rows; their directories are preserved under
`outputs/analysis/libero90_recovery_20260930/aborted_seed2_eval{36,58}_mean`. All eight incomplete tasks were rerun
from initial states 0–49 on node-61 GPU 1 with the same checkpoint and evaluation settings. These reruns replace
the two partial task runs in aggregation; partial rows are never counted twice. The remaining twenty store tasks
ran on node-63 GPU 1, completing π-noise at 19:32:34 EDT. Every launch passed the project GPU guard.

Resource CSVs measure elapsed card allocation rather than summed worker time. Training plus complete checkpoint
curves used 4.3794 GPU·h, the initially logged closed-loop intervals used 2.7475 GPU·h, recovery used 0.0822 GPU·h,
rollout collection used 0.2433 GPU·h and π-noise used 0.6264 GPU·h. The node-66 interval is a logged lower bound;
its exact process termination time is unknown. Peak measured memory was 5,871 MiB for training/curves, 9,716 MiB
for initial closed loops, 7,594 MiB for recovery, 8,607 MiB for rollout collection and 9,303 MiB for π-noise.
Short checks and aborted sampler runs are excluded from these totals. No node used more than two project GPUs.
Machine-readable resource measurements are in
`outputs/analysis/libero90_recovery_20260930/resources_complete.json`.

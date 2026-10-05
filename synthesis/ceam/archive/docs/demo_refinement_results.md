# Goal demonstration refinement pilot

## Verified preparation

Thirty source episodes are the first three TRAIN episodes of each Goal task under the existing episode split.
Each source supplies its original ten-step block and eight alternatives at approximately one third of its first
successful trajectory. Gripper commands and all actions outside that block are identical. Random and guided use
four alternatives plus original; guided ranks with fixed R6_all and focused q seed 0. Both choose a successful
candidate using measured first-success steps, then standardized arm-command total variation, retaining original
when none improves it. Every branch is executed from the same raw initial state and fixture poses.

All thirty anchor restorations have zero maximum proprio error. All selected full trajectories were re-rendered
and reached success; their executed step counts equal the CPU branch measurements. The feature/action caches retain
433 episode ids, the same 43 HELDOUT episodes, instruction tokens and original normalization statistics.
Visual tokens are `[N, 2, 64, 768]` float16, proprio is `[N, 9]` float32 and actions are `[N, 7]` float32. HELDOUT
feature and action arrays are array-exact to source. Selected episodes are rendered at 512 pixels in all conditions;
the remaining legacy features came from 224-pixel images. The original condition controls that shared change.

## Selected source trajectories

| condition | replaced source episodes / 30 | total steps | mean steps | mean standardized arm total variation |
|---|---:|---:|---:|---:|
| original | 0 | 3531 | 117.70 | 33.1251 |
| random | 28 | 3379 | 112.63 | 31.2649 |
| guided | 29 | 3397 | 113.23 | 31.4045 |

Both refinement conditions reduce the specified source costs. Guided has eighteen more steps and higher aggregate
command variation than random on these sources. This measures source-trajectory cost, not policy improvement or
physical robustness. The figures are in `outputs/demo_refinement/pilot_v1/reports/selection_summary.json`.

## Candidate-bank measurements

Of the 270 executed branches, 266 reached success. The full-bank lowest-cost choices total 3,373 steps;
random shortlist choices total 3,379 and guided choices total 3,397. Guided selects the full-bank oracle candidate
on 18/30 sources and random on 14/30. The two conditions select the same candidate on 13 sources. Across source
cost comparisons, random is better on ten, guided on seven and thirteen are equal.
The figures and per-source/task costs are in `outputs/demo_refinement/pilot_v1/reports/bank_diagnostics.json`.

## Training compatibility

The original, random and guided caches contain 52,426, 52,274 and 52,292 frames respectively. Their dataset roots
contain episode metadata for feature-based training, without video or parquet data. The existing no-ψ ProposalTrainer
completed a separate 100-update smoke at 8.98 updates/s with 4.66 GiB peak CUDA allocation. This smoke is outside
the nine policy comparison cells. A synthetic 4,500-row report validated exact cell completeness, paired differences
and a separate inits 40–49 readout; synthetic data is confined to `tmp/demo_refinement/synthetic_report/`.

## Policy comparison protocol

The full grid is three data conditions × π seeds 0/1/2, each trained for 3,000 updates, batch 256, IID sampling,
the same architecture and optimizer schedule. Every checkpoint is evaluated on ten Goal tasks × fifty benchmark
initial states with a 300-step horizon, environment seed 0 and ten settle steps. Inits 40–49 are reported separately
because focused q training used 0–39. The policy executes its mean chunk without a test-time r/q/m selector.
The direct mean adapter loads each checkpoint's own cache paths and preserves the baseline pre-episode reset/encode.
A known Goal no-ψ seed-0 checkpoint reproduced all five task-0 init-0–4 success outcomes and step counts exactly:
125, 121, 116, 129, 117. Independently published-cache evaluation also completed its smoke.

## Measured collection resources

The thirty-source scoring/re-rendering run used GPUs 0/1 on node-61 and node-63 from approximately 20:59:41 to
21:02:09 EDT on 2026-09-30. Per-card monitored intervals sum to 0.1069 GPU·h, with peak device memory 1,549 MiB.
CPU branch preparation, the single-source smoke, cache publication and compatibility training are separate.
Resource CSVs and time/RSS measurements are under `tmp/runlogs/demo_refinement/`; the collection summary is
`outputs/demo_refinement/pilot_v1/reports/render_resources.json`.

## Complete policy results

All nine checkpoints completed 3,000 updates and all 4,500 benchmark episodes completed.

| condition | seed 0 SR | seed 1 SR | seed 2 SR | pooled SR | inits 40–49 pooled SR |
|---|---:|---:|---:|---:|---:|
| original | 90.80% | 89.20% | 86.80% | 88.93% | 86.33% |
| random | 86.80% | 86.40% | 88.20% | 87.13% | 86.67% |
| guided | 88.00% | 89.40% | 93.40% | 90.27% | 88.67% |

| comparison | pooled SR difference (pp) | seed 0 / 1 / 2 differences (pp) | paired 95% interval (pp) |
|---|---:|---|---|
| random − original | -1.80 | -4.00 / -2.80 / +1.40 | [-3.87, +0.27] |
| guided − original | +1.33 | -2.80 / +0.20 / +6.60 | [-0.73, +3.40] |
| guided − random | +3.13 | +1.20 / +3.00 / +5.20 | [+1.13, +5.20] |

Intervals resample paired task/initial-state positions while keeping the three trained seeds together.
They describe benchmark uncertainty conditional on these checkpoints, not uncertainty over future training seeds.
The JSON also reports pooled exact McNemar values; these are descriptive because repeats across seeds share benchmark positions.

Guided exceeds random on all three π seeds and by 3.13 pp pooled. Guided versus original is +1.33 pp pooled,
with seed differences −2.80, +0.20 and +6.60 pp and a paired interval spanning zero. This establishes an offline
shortlist advantage over the fixed random shortlist for these cells; a reliable gain over original demonstrations
across training seeds remains unestablished. Lower source step/command costs alone do not establish better BC data.

## Per-task policy SR

| task | original | random | guided |
|---|---:|---:|---:|
| open the middle drawer of the cabinet | 95.33% | 90.00% | 96.67% |
| open the top drawer and put the bowl inside | 65.33% | 66.67% | 76.00% |
| push the plate to the front of the stove | 86.00% | 79.33% | 83.33% |
| put the bowl on the plate | 96.00% | 97.33% | 97.33% |
| put the bowl on the stove | 94.67% | 98.67% | 99.33% |
| put the bowl on top of the cabinet | 97.33% | 92.67% | 96.00% |
| put the cream cheese in the bowl | 71.33% | 66.67% | 80.00% |
| put the wine bottle on the rack | 93.33% | 90.67% | 85.33% |
| put the wine bottle on top of the cabinet | 90.00% | 89.33% | 88.67% |
| turn on the stove | 100.00% | 100.00% | 100.00% |

## Scope of the conclusion

Thirty of 390 TRAIN source episodes were eligible for edits, at one ten-step anchor each; random and guided
replace 28 and 29 respectively. The random data version uses one
fixed shortlist seed and guided uses one frozen focused-query checkpoint. The experiment does not vary source
selection, edit locations or refinement seeds. All three policies share re-rendered original controls, source
normalization and unedited legacy features; their absolute SR is not directly comparable to the older 92.67% Goal
baseline trained on the unrevised cache. Inits 40–49 were excluded from focused-q training; that subset is not a
claim that every representation or policy training observation is unseen.

## Complete execution resources

The monitored rendering and grid intervals sum to 2.6747 GPU·h, including abandoned
training and failed legacy evaluation attempts. Peak device memory is 7,299 MiB.
These CSV intervals exclude independent smokes and CPU-only work and omit unsampled start/end gaps.
The pilot artifact directory and policy runs occupy 35.94 GB of logical file bytes.
Every node used at most two project GPUs. All pilot GPU workers and the CPU report driver completed by 22:14:13 EDT
on 2026-09-30; final training/evaluation allocations were node-61 0/1 and node-65 1/2.

CPU branch preparation took 322.61 and 247.03 seconds wall on two shards, totaling 551.20 user and 15.86 system
CPU-seconds, with about 2.01 GiB maximum RSS. Three-cache publication took 124.19 seconds wall and 19.21 GiB
maximum RSS. Separate time/RSS records are in `outputs/demo_refinement/pilot_v1/reports/cpu_time_records.json`.
The cache manifest’s inherited `build_seconds` identifies the source build; the publication timing is the measured
cache-build command above.

Raw cells, comparison counts and intervals: `outputs/demo_refinement/pilot_v1/reports/policy_comparison.{json,md}`.
Resource intervals: `outputs/demo_refinement/pilot_v1/reports/execution_resources.json`.
Driver identities: `outputs/demo_refinement/pilot_v1/execution.json`.

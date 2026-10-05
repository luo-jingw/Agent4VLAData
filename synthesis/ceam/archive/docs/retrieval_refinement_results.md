# Dataset-backed action retrieval results

## Outcome

The LIBERO-Goal pilot collected 180 same-start candidate banks and 1,618 executed branches. Demonstration retrieval
supplies successful faster alternatives at most evaluated starts, but frozen match ranking does not reliably improve
the four-candidate offline shortlist over exact random shortlisting. Direct match selection damages some successful
source demonstrations. Minimum-step selection also increases measured physical jerk.

These are branch outcomes under an unchanged demonstration suffix. No policy was trained on these edits and no
learned-policy SR was measured in this experiment.

## Protocol

Configuration: `configs/refinement/retrieval_pilot.json`. Design: `plans/retrieval_refinement.md`.
Artifacts: `outputs/retrieval_refinement/pilot_v1/`; execution logs: `tmp/runlogs/retrieval_refinement/`.

- Ten Goal tasks; first twelve TRAIN donors per task and first three HELDOUT query demonstrations per task under
  `EpisodeIndex` held-out fraction 0.1, seed 0. The 120 donors and 30 queries are disjoint. All replay successfully.
- Index 1,311 donor chunks at stride ten, stopping at first success. Each query contributes anchors at 1/4, 1/2 and
  3/4 of its replay first-success length, rounded down to ten-step boundaries.
- Retrieve from the same task, with one chunk per distinct donor episode. Geometry uses EEF xyz. Physical retrieval
  uses EEF position/orientation, arm/finger joint positions, object positions relative to EEF and fixture joint
  positions, with equal two-finger object-contact signatures. Both require the same initial gripper command.
- Directly copy the six arm-command components of each ten-step donor chunk and retain the source's entire
  ten-step gripper sequence. No pose retargeting, timing change or gripper optimization is performed.
- Restore every alternative to exactly the same query start, execute the replacement chunk and original source
  suffix, and stop at first task success or the original episode horizon. Keep failures.
- Score restored 512² dual-camera query observations under four frozen checkpoints, with R6_all fixed. Scoring
  consumes observations, instruction and candidate actions; measured branch outcomes do not enter the scorer.
- Compare eight selectors per retrieval/checkpoint: original, nearest donor, exact uniform all-bank and donor-only
  expectations, match argmax including original, successful minimum-step oracle, match top-four donors plus original
  with measured acceptance, and exact random four donors plus original with the same measured acceptance.

The acceptance cost is success first, then first-success steps, then standardized six-arm-command total variation.
All-bank oracle and both shortlist editors use measured outcomes. Keeping a successful original in every shortlist
explains their 100% branch success; this is not a learned ability to predict success.

## Coverage and controls

| Retrieval | Starts | Donors per bank | Donor branches | Donor branch success | Starts with a successful faster donor |
|---|---:|---:|---:|---:|---:|
| Geometry | 90 | 8 | 720 | 73.75% | 74/90 |
| Physical | 90 | 7–8 | 718 | 77.16% | 71/90 |

Candidate chunks are float32: 178 banks have shape `[9,10,7]` and two have shape `[8,10,7]`. No start has zero donor
coverage. The donor-success percentages above weight individual branches. Selector results
below weight starts equally; physical donor-only uniform success is 77.14% because two banks have seven donors.

Original branch success is 100% in both modes, with mean 113.267 first-success steps. All original branches agree
with independent source replay: zero outcome disagreements and zero step differences. CPU restored-proprio maximum
absolute error is 1.1891e-7; rendered-query restoration error is zero for all 180 banks. The legacy nine-candidate
scorer reproduces saved scores and restored proprio with maximum absolute differences of zero.

## Direct selection

| Selector / query checkpoint | Geometry branch success | Physical branch success |
|---|---:|---:|
| Original | 100.00% | 100.00% |
| Nearest donor | 82.22% | 82.22% |
| Uniform all-bank expectation | 76.67% | 79.69% |
| Uniform donor-only expectation | 73.75% | 77.14% |
| Match, original q seed 0 | 83.33% | 85.56% |
| Match, focused q seed 0 | 72.22% | 78.89% |
| Match, focused q seed 1 | 71.11% | 76.67% |
| Match, focused q seed 2 | 75.56% | 80.00% |

Original q beats uniform all-bank by +6.67 pp in geometry, interval [0.49, 12.72] pp, and +5.86 pp in physical
retrieval, interval [1.73, 10.22] pp. This is a limited positive compatibility-ranking observation. Original q still
loses 15/90 and 13/90 source successes respectively. Focused q does not consistently outperform uniform selection
on this candidate distribution; this experiment does not isolate the reason.

Physical retrieval changes the candidate bank as well as its metric and contact eligibility. Its better average
donor validity cannot be attributed to one descriptor component without further controlled comparison.

## Offline shortlisting and speed

Both shortlist editors and the oracle retain 100% branch success.

| Selector / query checkpoint | Geometry mean steps | Physical mean steps |
|---|---:|---:|
| Original | 113.267 | 113.267 |
| Successful all-bank oracle | 109.244 | 109.456 |
| Exact random four-donor shortlist | 110.294 | 110.342 |
| Match shortlist, original q seed 0 | 110.078 | 110.256 |
| Match shortlist, focused q seed 0 | 110.256 | 110.267 |
| Match shortlist, focused q seed 1 | 110.478 | 110.533 |
| Match shortlist, focused q seed 2 | 110.244 | 110.222 |

Original-q match minus random shortlisting is −0.217 steps in geometry, interval [−0.532, 0.101], and −0.087 steps
in physical retrieval, interval [−0.294, 0.135]. All eight match-versus-random step intervals span zero. These data do
not establish a stable shortlist speed advantage. The physical-bank oracle saves 3.811 steps, or 3.36% of source
duration; this is the upper bound for this bank and acceptance cost, not an upper bound for all possible policies.

Intervals use 2,000 episode bootstrap resamples, seed 20261001, drawing three source episodes within each of the ten
fixed tasks and keeping each episode's three anchors together. They condition on the fixed donor pool and frozen
checkpoints. No correction for multiple comparisons is applied; intervals are descriptive. Focused checkpoints
vary query training seeds; these are not policy training seeds.

## Physical smoothness

Metrics include the common trajectory prefix and executed continuation at native 20 Hz. EEF translation path is in
metres; rotation path is in radians. Acceleration RMS and jerk RMS are norms of finite differences of EEF xyz in
m/s² and m/s³. Command variation is measured separately and is not physical jerk.

| Physical-bank selector | Steps on success | Translation path m | Acceleration RMS m/s² | Jerk RMS m/s³ |
|---|---:|---:|---:|---:|
| Original | 113.267 | 0.827 | 0.494 | 6.933 |
| Successful all-bank oracle | 109.456 | 0.799 | 0.520 | 7.852 |
| Exact random shortlist | 110.342 | 0.806 | 0.512 | 7.570 |
| Original-q match shortlist | 110.256 | 0.808 | 0.510 | 7.521 |

Successful minimum-step selection raises jerk by about 13.3% over original. The current lexicographic cost does
not optimize physical smoothness. These measurements cover demonstration branches, not policy chunk-boundary
jitter. Success-conditional and failure-conditional motion metrics are stored separately for every selector in
`reports/comparison.json`; all-branch executed steps are also reported to expose failures.

## Task breakdown

Physical retrieval, original q. Each task has nine starts. Uniform donor success is the start-weighted expectation.

| Task | Instruction | Uniform donor success | Direct match success | Original steps | Oracle steps |
|---|---|---:|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 76.39% | 77.78% | 141.33 | 133.67 |
| 1 | open the top drawer and put the bowl inside | 65.28% | 77.78% | 196.00 | 194.89 |
| 2 | push the plate to the front of the stove | 70.83% | 77.78% | 122.67 | 120.44 |
| 3 | put the bowl on the plate | 75.00% | 77.78% | 78.00 | 73.67 |
| 4 | put the bowl on the stove | 94.44% | 100.00% | 98.33 | 96.78 |
| 5 | put the bowl on top of the cabinet | 90.28% | 100.00% | 89.00 | 85.78 |
| 6 | put the cream cheese in the bowl | 57.54% | 77.78% | 95.33 | 87.67 |
| 7 | put the wine bottle on the rack | 70.83% | 77.78% | 151.00 | 146.00 |
| 8 | put the wine bottle on top of the cabinet | 87.50% | 100.00% | 84.67 | 82.22 |
| 9 | turn on the stove | 83.33% | 88.89% | 76.33 | 73.44 |

## Execution and resources

Project environment: CPython 3.10.20, torch 2.7.1+cu126, MuJoCo 3.3.2, robosuite 1.4.0, LIBERO `48929ba`.
CPU branches ran on node-61 tasks 0–4 and node-63 tasks 5–9, five workers per node, thread limits two, no rendering.
The full preparation used 0.662 CPU core-hours; slowest task wall time was 468.85 s; highest worker RSS was
2,091,876 KiB. The smoke worker used 66.30 s and 2,102,832 KiB.

| GPU run | Node / card | Monitored interval EDT, 2026-10-01 | Worker wall time | Peak device memory |
|---|---|---|---:|---:|
| Smoke | node-61 / 1 | 23:28:19–23:28:41 | 23.51 s | 1,735 MiB |
| Full 180-bank scoring | node-63 / 0 | 23:38:27–23:39:11 | 45.05 s | 1,860 MiB |

The two-second resource samples cover 0.0183 GPU-hours; combined worker wall time corresponds to 0.0190 GPU-hours.
Full scoring records 843,854,336 peak Torch allocated bytes, excluding renderer/driver memory. Only one GPU was
used at a time, below the two-GPU-per-node limit. Both nodes had no retrieval Python workers at the final process
check at 23:46:09 EDT. Derived retrieval artifacts occupy approximately 3.5 MB, reusing existing caches and models.

Entry points through `scripts/cluster_run.sh`:

```text
node-61/node-63: tmp/retrieval_refinement/launch_prepare.sh <lane>
node-63 --gpus 0 --free-mib 16000: tmp/retrieval_refinement/score.sh full 0
node-61: tmp/retrieval_refinement/report.sh
```

The persistent scripts are `scripts/prepare_retrieval_refinement.py`, `scripts/score_retrieval_refinement.py` and
`scripts/report_retrieval_refinement.py`; each accepts the checked-in config. Runtime wrappers, checkpoint SHA256s,
resource CSVs and time/RSS observations are preserved. Report parsing uses explicit timestamp format for Python
3.10 compatibility. Its final complete run exited zero; no further GPU execution was required to fix reporting.

## Limits

This is same-task privileged-state retrieval with unchanged gripper commands and direct donor arm increments.
Object orientations, velocities, one-finger contact and temporal context are absent from the distance descriptor.
It therefore does not establish physical state equivalence, cross-task transfer, vision-only retrieval or safe
replacement under a newly learned continuation policy. Only one local block is replaced at a time; full-route
redundancy and alternative task strategies were not tested. Three anchors do not test a successful trajectory's
integrated-match constraint. Neither self-supervised feedback learning nor RL was run.

Open findings and follow-up mechanisms are recorded in ISSUE-040 and OPT-011. The complete 64-cell grid, paired
intervals, task results and separate failure metrics are in `outputs/retrieval_refinement/pilot_v1/reports/`.

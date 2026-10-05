# Frozen-teacher action editing pilot

## Verified setup

Public OpenPI source is pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`. Checkpoint object generations and
SHA256 values are in `outputs/teacher_assets/manifest.json`; all 16 GCS objects also passed CRC32C verification.
Assets occupy 12.446 GB. The separate `runtimes/openpi/uv.lock` resolves Python 3.11.15, JAX 0.5.3, Flax 0.10.2
and Orbax 0.11.13. The simulator keeps its original Python 3.10 environment. No model training occurs.

The public pi05_libero configuration has horizon 10, action dimension 32 and discrete_state_input=False.
Its pi05 suffix omits continuous state tokens: this checkpoint conditions actions on images/instruction,
even though the input record retains EEF position, axis-angle and finger positions. Its immutable quantile
normalization is kept. Dataset gripper 1 means open; simulator/teacher gripper −1 means open.
Images are already rotated by the CEAM renderer and are only resized to 224×224 at the teacher boundary.

Thirty episodes from Goal tasks 0, 3 and 6 were exactly matched to raw HDF5 actions, including gripper conversion.
All 30 source replays succeeded. Eight TRAIN and two HELDOUT episodes per task yield 48 calibration and 12 query
locations, at one-third and two-thirds of first-success time. All 60 locations were rendered anew at 512×512.
Maximum physics/render restore proprio difference was 1.190e-7. This is source replay success, not policy SR.
TRAIN/HELDOUT separate local calibration episodes; they do not establish exclusion from the public teacher training data.

## Numerical qualification

The fixed gate is max arm-command reconstruction error ≤0.001, finite output and unchanged binarized gripper.
The initial 24-cell grid and 28-cell convergence follow-up did not admit a BF16 solver. A precision isolation
cell at 40 Euler steps / 60 fixed-point iterations admitted float32 computation. Parameters retain identical
BF16 values in both compute modes; no weight updates occur. Six source maxima were ≤0.000382; the generated
control maximum was 0.000480. Float32 computation passed this numerical gate; no trajectory-quality conclusion
follows from reconstruction accuracy.

Same-graph implementation and official sampling agreed exactly on the qualification reference. Separating
prefix caching changed BF16 outputs by 0.001953125 normalized units in the recorded final comparisons;
with float32 computation, the reference difference was 0.000461578 normalized / 0.000080884 physical units.
Warm ten-step decoding was 0.0258 s; 40-step / 60-iteration inversion was about 6.12 s after compilation.
The float32 qualification measured 7.232 GB peak live JAX allocation and 8.592 GB peak allocator pool.

All numerical cells below use three tasks, six source locations, 10×32 latent arrays and a single L40S.
Generated controls use seed 0 at e0_t42. Time includes compilation for the first call of each shape/static setting.

| Compute | Target | Location | Euler steps | Inverse iterations | Max arm error | RMS arm error | Max normalized error | Seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BF16 | source | e0_t42 | 10 | 5 | 0.017684 | 0.00620037 | 0.0292969 | 3.112 |
| BF16 | source | e0_t42 | 10 | 10 | 0.0118995 | 0.00438022 | 0.0146486 | 3.011 |
| BF16 | source | e0_t42 | 20 | 5 | 0.0084883 | 0.00310132 | 0.0117188 | 2.989 |
| BF16 | source | e0_t42 | 20 | 10 | 0.00640742 | 0.00197015 | 0.00683594 | 3.305 |
| BF16 | source | e0_t84 | 10 | 5 | 0.034783 | 0.00832264 | 0.0371093 | 0.168 |
| BF16 | source | e0_t84 | 10 | 10 | 0.0118995 | 0.00490739 | 0.0205078 | 0.292 |
| BF16 | source | e0_t84 | 20 | 5 | 0.00781372 | 0.00276216 | 0.0112304 | 0.334 |
| BF16 | source | e0_t84 | 20 | 10 | 0.00503436 | 0.00145798 | 0.00634766 | 0.569 |
| BF16 | source | e114_t23 | 10 | 5 | 0.0155609 | 0.00480242 | 0.024414 | 0.168 |
| BF16 | source | e114_t23 | 10 | 10 | 0.00945882 | 0.00331421 | 0.0136718 | 0.285 |
| BF16 | source | e114_t23 | 20 | 5 | 0.00697257 | 0.00197767 | 0.00842285 | 0.332 |
| BF16 | source | e114_t23 | 20 | 10 | 0.00606312 | 0.00141482 | 0.00732428 | 0.568 |
| BF16 | source | e114_t46 | 10 | 5 | 0.0247142 | 0.00787614 | 0.0263672 | 0.167 |
| BF16 | source | e114_t46 | 10 | 10 | 0.0201375 | 0.00591728 | 0.0214844 | 0.285 |
| BF16 | source | e114_t46 | 20 | 5 | 0.0121282 | 0.00387876 | 0.0129393 | 0.332 |
| BF16 | source | e114_t46 | 20 | 10 | 0.00732279 | 0.00200511 | 0.0078125 | 0.569 |
| BF16 | source | e261_t25 | 10 | 5 | 0.0338678 | 0.00868777 | 0.0361328 | 0.168 |
| BF16 | source | e261_t25 | 10 | 10 | 0.0145513 | 0.00518309 | 0.0253907 | 0.285 |
| BF16 | source | e261_t25 | 20 | 5 | 0.00808406 | 0.00302471 | 0.0156251 | 0.332 |
| BF16 | source | e261_t25 | 20 | 10 | 0.00454728 | 0.00154234 | 0.0107422 | 0.570 |
| BF16 | source | e261_t50 | 10 | 5 | 0.0266775 | 0.00808524 | 0.0429686 | 0.169 |
| BF16 | source | e261_t50 | 10 | 10 | 0.0148051 | 0.00460638 | 0.0214844 | 0.286 |
| BF16 | source | e261_t50 | 20 | 5 | 0.0106926 | 0.00285093 | 0.0146485 | 0.333 |
| BF16 | source | e261_t50 | 20 | 10 | 0.00575757 | 0.00131937 | 0.00683588 | 0.569 |
| BF16 | source | e0_t42 | 20 | 30 | 0.00274612 | 0.000812778 | 0.00390625 | 4.837 |
| BF16 | teacher_generated | e0_t42 | 20 | 30 | 0.00329 | 0.00110928 | 0.00585938 | 1.536 |
| BF16 | source | e0_t42 | 20 | 60 | 0.00164482 | 0.000479103 | 0.00195312 | 6.040 |
| BF16 | teacher_generated | e0_t42 | 20 | 60 | 0.00131762 | 0.000384692 | 0.00195324 | 2.976 |
| BF16 | source | e0_t42 | 40 | 30 | 0.00164494 | 0.00041136 | 0.00195312 | 6.116 |
| BF16 | teacher_generated | e0_t42 | 40 | 30 | 0.000808418 | 0.000265597 | 0.00341809 | 3.072 |
| BF16 | source | e0_t42 | 40 | 60 | 0.00137301 | 0.000263131 | 0.00146484 | 9.009 |
| BF16 | teacher_generated | e0_t42 | 40 | 60 | 0.0012337 | 0.000284797 | 0.00146484 | 5.932 |
| BF16 | source | e0_t84 | 20 | 30 | 0.00137293 | 0.00053931 | 0.00341797 | 1.534 |
| BF16 | source | e0_t84 | 20 | 60 | 0.00123379 | 0.000369729 | 0.00244141 | 2.973 |
| BF16 | source | e0_t84 | 40 | 30 | 0.00101045 | 0.000292129 | 0.00292969 | 3.069 |
| BF16 | source | e0_t84 | 40 | 60 | 0.000915349 | 0.000298079 | 0.00146484 | 5.936 |
| BF16 | source | e114_t23 | 20 | 30 | 0.00228834 | 0.000512395 | 0.00390625 | 1.535 |
| BF16 | source | e114_t23 | 20 | 60 | 0.00242522 | 0.000488542 | 0.00292969 | 2.976 |
| BF16 | source | e114_t23 | 40 | 30 | 0.000808381 | 0.000262902 | 0.00219727 | 3.070 |
| BF16 | source | e114_t23 | 40 | 60 | 0.00123 | 0.000298133 | 0.00170898 | 5.936 |
| BF16 | source | e114_t46 | 20 | 30 | 0.0018307 | 0.000513509 | 0.00488281 | 1.532 |
| BF16 | source | e114_t46 | 20 | 60 | 0.00183076 | 0.000606935 | 0.00292969 | 2.969 |
| BF16 | source | e114_t46 | 40 | 30 | 0.00137305 | 0.000350937 | 0.00268555 | 3.065 |
| BF16 | source | e114_t46 | 40 | 60 | 0.000915408 | 0.0002899 | 0.00146484 | 5.927 |
| BF16 | source | e261_t25 | 20 | 30 | 0.0016169 | 0.000512593 | 0.00488281 | 1.532 |
| BF16 | source | e261_t25 | 20 | 60 | 0.0016169 | 0.000505767 | 0.00244141 | 2.970 |
| BF16 | source | e261_t25 | 40 | 30 | 0.00114417 | 0.000288798 | 0.00219727 | 3.063 |
| BF16 | source | e261_t25 | 40 | 60 | 0.000808392 | 0.00023456 | 0.000976562 | 5.931 |
| BF16 | source | e261_t50 | 20 | 30 | 0.0018307 | 0.000519305 | 0.00585938 | 1.534 |
| BF16 | source | e261_t50 | 20 | 60 | 0.0012126 | 0.000383537 | 0.00195342 | 2.970 |
| BF16 | source | e261_t50 | 40 | 30 | 0.000686467 | 0.000250707 | 0.00244141 | 3.067 |
| BF16 | source | e261_t50 | 40 | 60 | 0.0010106 | 0.000289828 | 0.00128174 | 5.929 |
| FP32 | source | e0_t42 | 40 | 60 | 0.000216275 | 7.54649e-05 | 0.00109919 | 8.752 |
| FP32 | teacher_generated | e0_t42 | 40 | 60 | 0.000480473 | 0.000119699 | 0.00155132 | 6.108 |
| FP32 | source | e0_t84 | 40 | 60 | 0.000111043 | 4.31738e-05 | 0.00105404 | 6.107 |
| FP32 | source | e114_t23 | 40 | 60 | 0.00033766 | 7.28674e-05 | 0.0017283 | 6.120 |
| FP32 | source | e114_t46 | 40 | 60 | 0.000335336 | 7.48043e-05 | 0.00133743 | 6.115 |
| FP32 | source | e261_t25 | 40 | 60 | 0.00038147 | 6.38355e-05 | 0.000862382 | 6.121 |
| FP32 | source | e261_t50 | 40 | 60 | 0.000216842 | 5.03062e-05 | 0.000932395 | 6.119 |

All 59 cells were finite and retained their binarized gripper commands. Three earlier interface attempts
are retained under `qualification/attempt_{1,2,3}`; they tested timestep arithmetic before the same-graph
comparison isolated caching-boundary numerical effects. Their cached/reference max differences were
0.0029296875, 0.001953125 and 0.0029296875 normalized units. These attempts did not run candidate evaluation.

## Artifacts and commands

Artifact root: `outputs/teacher_editing/pilot_v1`. Source/camera artifacts are under `data/`; every inversion
cell preserves target, reconstructed action and noise arrays. Gateway command/script checksums and node/GPU
declarations are in `tmp/runlogs/cluster_run/invocations.log`.

CPU preparation: `scripts/prepare_teacher_data.py`. Rendering: `scripts/render_teacher_data.py --cuda-device 0`.
Qualification: `scripts/qualify_teacher.py --cuda-device 0 --assets outputs/teacher_assets --data-dir
outputs/teacher_editing/pilot_v1/data --output <qualification-directory>`, with grids `{10,20}×{5,10}`,
`{20,40}×{30,60} --generated-control`, and `40×60 --generated-control --compute-dtype float32`.
All remote runs used `scripts/cluster_run.sh`, node-61 GPU 0 for rendering/inference and node-63 CPU for replay.

## Candidate bank

All 60 TRAIN/query chunks passed the fixed source reconstruction gate: max physical arm-command error
0.000515, with every gripper sign preserved. Calibration uses exactly 48 TRAIN chunks; query episodes do
not enter its moments. The per-coordinate raw standard deviation has min/median/max 0.522/0.917/1.859.
The first seven coordinate columns have median 1.361; the 25 padded columns have median 0.844. These are
noise-coordinate statistics, not intrinsic dimension or trajectory-quality measurements. Local edits use
standard deviation scales; the latent mean cancels when editing around a source code. No codebook is learned.

The fixed bank contains 600 proposals: 12 source, 12 reconstruction and 144 each of direct normalized-action
edits, uncalibrated teacher-noise edits, calibrated-noise edits and teacher-only reference proposals.
The three local editing mechanisms preserve each source gripper schedule. Teacher-only draws use their own
binarized gripper and are a separate reference. Each location has 3 candidate seeds × 4 draws per mechanism.
Seeds describe candidate draws; they are not policy-training seeds.

| Mechanism | Proposals | Mean arm-command RMS edit | Min–max RMS | Commands outside arm limits |
|---|---:|---:|---:|---:|
| Direct | 144 | 0.0300000 | 0.0300000–0.0300000 | 19 |
| Teacher noise | 144 | 0.0300202 | 0.0297097–0.0303853 | 7 |
| Calibrated noise | 144 | 0.0300149 | 0.0290676–0.0302931 | 5 |
| Teacher only | 144 | 0.0385927 | 0.0124399–0.134172 | 0 |
| Reconstruction | 12 | 0.00007254 | 0.00003487–0.00018902 | 0 |

All local edits meet the declared 0.03±0.003 matching interval. RMS is measured in the six simulator arm-command
coordinates; it does not equate translation in metres with rotation in radians, or equalize temporal roughness.
No command is clipped or resampled after a limit violation. Invalid proposals retain their budget slots and
count as failures; executed-branch counts therefore differ. Fewer violations alone cannot establish better
quality among legal executions. Original fallback is separate from generator success.

Generation took 714.6 s on node-61 GPU 0. Peak live JAX allocation was 7.240 GB, peak allocator pool 8.693 GB;
a process memory sample was 8,742 MiB. TRAIN plus query inversion used 148,800 velocity evaluations. Excluding
shared source inversion, uncalibrated radius fitting used 61,040, calibrated fitting 60,120 and teacher-only
reference draws 5,760. Direct editing uses no teacher velocity evaluations. The reconstruction entries refer
to shared query inversion and are not an additional inversion cost. Teacher computation and simulator budgets
are distinct; matching proposal counts does not match generation latency.

## Source-suffix execution results

The complete 600-cell table is `outputs/teacher_editing/pilot_v1/report/all_cells.csv`; JSON measurements and
full EEF/action traces are under `execution/`. Every branch replays its original prefix from the matched raw
initial state, applies its ten-step replacement and then replays the remaining source actions until first success
or source exhaustion. All original control steps match preparation exactly. All successful branches passed the
observed command/joint/contact screen. This is not a newly trained policy's benchmark SR.

A quality improvement requires success, observed feasibility and matched edit magnitude. Faster means ≥1 fewer
control step; shorter means ≥1 mm less EEF path; smoother means ≥1% lower RMS jerk. Pareto flags additionally
allow no step increase, at most 0.1 mm path increase and at most 0.1% jerk increase. These thresholds were fixed
before execution. Metrics use actual control-boundary EEF positions at the simulator control timestep.

| Mechanism | Proposals | SR | Valid success | Matched | Faster | Shorter | Smoother | Pareto | Locations / episodes with Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 12 | 12/12 | 12 | n/a | 0 | 0 | 0 | 0 | 0 / 0 |
| reconstruction | 12 | 12/12 | 12 | n/a | 0 | 1 | 2 | 1 | 1 / 1 |
| direct | 144 | 123/144 | 123 | 144 | 28 | 38 | 37 | 16 | 6 / 4 |
| teacher_noise | 144 | 129/144 | 129 | 144 | 36 | 59 | 35 | 24 | 8 / 6 |
| calibrated_noise | 144 | 131/144 | 131 | 144 | 35 | 60 | 37 | 19 | 6 / 5 |
| teacher_only | 144 | 126/144 | 126 | n/a | 30 | 56 | 56 | 34 | 9 / 6 |

Locations with Pareto improvements correspond to verifier-assisted best-of-12 availability, pooling the three
candidate seeds. They do not establish a learned selector's performance. Seed-specific Pareto counts are direct
5/1/10, teacher noise 8/8/8, calibrated noise 7/5/7 and teacher-only 14/11/9, with 48 proposals per seed/method.

The direct method succeeds in 123/125 executed legal proposals; teacher noise 129/137; calibration 131/139.
Thus calibration's two additional successes over uncalibrated noise coincide with two fewer rejected commands;
the number of executed failures is eight in both. Calibration yields fewer Pareto improvements, 19 versus 24.
The teacher-only reference has different magnitudes and gripper freedom, so its 34 Pareto improvements do not
isolate a superior editing coordinate system. Source controls are successful by construction and are not a
sampling baseline that ought to have lower SR than perturbed proposals.

Numerical reconstruction preserved success in 12/12 branches but changed completion by one step in three cases.
It also produced one Pareto improvement and one 7.70 mm path increase. Small command errors can matter under
contact and thresholded success; identity in action norm does not imply identical trajectory outcomes.

One selected calibrated candidate, `e18_t58_calibrated_noise_s1_d2.npz`, changes 174 to 129 steps, EEF path
0.92983 to 0.61776 m and RMS jerk 11.7897 to 8.6252 m/s³ under source continuation. At this location, the fastest
successful direct candidate takes 171 steps, uncalibrated teacher noise 130 and teacher-only 129. This is a
verifier-selected case; it does not demonstrate an advantage of the calibration statistics.

## Common-teacher continuation diagnostic

The complete 36-cell table is `outputs/teacher_editing/pilot_v1/continuation/all_cells.csv`. Selection and
seeds were fixed in `configs/teacher_editing/continuation.json` before these rollouts. Locations e18_t58 and
e288_t33 are the two strongest calibrated step-improvement cases from different episodes. Each mechanism
contributes its fastest Pareto-improving source-suffix candidate, breaking ties by path; when there is no Pareto
candidate, its fastest successful valid proposal is used. Original and reconstruction controls are retained.
This verifier-assisted, selected-case diagnostic is not an unbiased evaluation of a learned selector.

Each branch replays the full source prefix and ten-step candidate, then uses the same frozen teacher, replanning
every ten actions up to 400 total control steps. Seeds 0/1/2 now identify continuation noise streams, paired by
location and chunk number; they are distinct from the candidate-generation seeds retained in selected filenames.
The teacher uses float32 computation, the same frozen BF16 parameter values and 40 integration steps. Fresh
rendered observations and action responses are retained in `continuation/rpc/`. The physical state field is
measured at control boundaries; this checkpoint ignores that field. Teacher and renderer run in separate Python
environments on the same single GPU. The state wrapper delegates every simulator mutation to LiberoTaskEnv.

Rescue means successful/valid continuation when its paired original control fails; regression means losing a
successful/valid original continuation. Paired Pareto compares only jointly successful/valid branches under the
same declared quality thresholds. Successful-only mean times do not compare rescued failures fairly.

| Source location | Mechanism | Success | Valid success | Rescues | Regressions | Paired Pareto |
|---|---|---:|---:|---:|---:|---:|
| e18_t58 | original | 1/3 | 1 | 0 | 0 | 0 |
| e18_t58 | reconstruction | 1/3 | 1 | 0 | 0 | 0 |
| e18_t58 | direct | 0/3 | 0 | 0 | 1 | 0 |
| e18_t58 | teacher_noise | 1/3 | 1 | 0 | 0 | 0 |
| e18_t58 | calibrated_noise | 2/3 | 2 | 1 | 0 | 0 |
| e18_t58 | teacher_only | 1/3 | 1 | 0 | 0 | 0 |
| e288_t33 | original | 2/3 | 2 | 0 | 0 | 0 |
| e288_t33 | reconstruction | 2/3 | 2 | 0 | 0 | 0 |
| e288_t33 | direct | 2/3 | 2 | 0 | 0 | 0 |
| e288_t33 | teacher_noise | 3/3 | 3 | 1 | 0 | 0 |
| e288_t33 | calibrated_noise | 3/3 | 3 | 1 | 0 | 0 |
| e288_t33 | teacher_only | 3/3 | 3 | 1 | 0 | 1 |

At e18_t58, the original branch succeeds only for continuation seed 1, in 136 steps. The source-suffix winners
from teacher noise and calibration both require 191 steps under that same continuation seed. Calibration also
rescues seed 2, reaching success in 172 steps. Thus the 174→129 source-suffix speed gain does not carry over as
a comparable speed gain under the shared teacher continuation. At e288_t33, teacher-noise, calibrated and
teacher-only candidates rescue one original failure each. Only one teacher-only case satisfies the paired
Pareto criterion; the teacher-noise and calibrated candidates have none in these selected continuations.
Some failed drawer continuations produce arm contacts and joint-limit excesses; these remain observed validity
failures. This diagnostic does not measure the public checkpoint's standard benchmark SR.

## Conclusions and limits

**Measured conclusion.** Frozen-teacher proposal editing is implemented and produces successful/valid source-suffix
improvements. In this fixed bank, uncalibrated teacher noise provides 24 Pareto improvements against 16 direct
edits, while adding TRAIN variance calibration reduces that count to 19. Calibration rejects fewer out-of-range
commands but does not show a consistent additional trajectory-quality benefit. The common-continuation check
shows success rescues alongside lost efficiency gains. Improvement depends on the complete continuation.

**Confidence.** These are exploratory observations from six query episodes, twelve locations, three candidate
seeds and two selected continuation cases. They support neither a general superiority claim nor rejection of
teacher/dataset-conditioned generation. The direct baseline is an unclipped Gaussian edit, not the strongest
bounded or temporally structured optimizer. Teacher-only proposals differ in magnitude and gripper freedom.
The raw noise space contains 25 padded coordinate columns per timestep, so coordinate variance is not a semantic
or intrinsic-dimensionality measurement. No learned candidate ranking, capacity compression, codebook training,
BC/RL update or downstream trained-policy SR was measured. Edited trajectories are not published as a replacement
dataset. Remaining quality/continuation questions are ISSUE-047 and OPT-015.

The bounded pilot is complete. There is no basis here to expand variance-calibrated editing into large-scale
training. A stronger follow-up must distinguish meaningful trajectory context and quality information from
coordinate scaling, and include a stronger direct-edit baseline and declared continuation/deployment protocol.

## Resources and completion

The full ledger is `outputs/teacher_editing/pilot_v1/resources.json`. All nine GPU runs, including failed numerical
interface attempts and rendering, total approximately 0.442 GPU·h by declared single-GPU wall time. Four endpoints
use final log timestamps, so this is approximate. Maximum concurrent GPU count was one, always node-61 GPU 0.
The final renderer/teacher device-memory sample was 9,163 MiB. All inference and simulator workers exited;
GPU 0 had 0 MiB used in the 2026-10-03 21:50 EDT post-run check. CPU source-suffix workers on node-63 took
424/234/265 seconds for tasks 0/3/6 respectively. New artifact/runtime storage is bounded by a conservative
30 GB bound; measured covered directories total 28.097 GB and include pre-existing uv cache/Python files.
This remains below the 40 GB limit; it is not a measurement of incremental storage alone.

Inference/generation entry: `scripts/generate_teacher_candidates.py --config configs/teacher_editing/pilot.json
--cuda-device 0` in the isolated teacher environment. CPU execution entry: `scripts/evaluate_teacher_candidates.py
--config configs/teacher_editing/pilot.json --output outputs/teacher_editing/pilot_v1/execution --task-id <0|3|6>`.
Continuation entry: `scripts/evaluate_teacher_continuation.py --config configs/teacher_editing/continuation.json
--cuda-device 0` in the simulator environment; its teacher worker starts through `scripts/teacher_server.sh`.
Exact executed scripts and hashes remain under `tmp/runlogs/cluster_run/`; syntax/annotation checks and the
complete numerical/rollout grids provide validation. The original root dependency lockfile remains the simulator's.

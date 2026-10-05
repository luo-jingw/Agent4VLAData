# Executed preference feedback

## Result

Executed same-start alternatives supply useful faster/shorter preference supervision. Across three training
seeds, preference-pair accuracy on remaining HELDOUT starts rises from 52.79% for the frozen match to 61.49%
with fixed action representations and a learned preference query, and 69.98% with a task-independent action
residual as well. Compatibility-only learning yields 52.79% and 54.07%, respectively.

The learned preferences improve an equal-budget shortlist that still uses actual physics outcomes for acceptance.
On remaining HELDOUT starts, adapted-representation preference shortlists save 0.0375 s and 10.83 mm of complete
EEF translation path relative to random shortlists. Direct selection remains unreliable. This experiment
establishes local feedback-ranking utility, not a refined dataset, learned-policy SR, global strategy optimality,
or a replacement for execution validation.

## Sources and observed feasibility

The candidate actions are unchanged from the physical-world pilot: original, up to eight retrieved actions,
half-arm action, and two arm-noise actions with preserved gripper commands. Each candidate replaces one ten-step
block and is followed by the original source continuation until first success or the demonstration horizon.
Time, EEF translation/rotation and RMS third-difference jerk include the common original prefix at 20 Hz.
Reported simulated durations exclude compute/render latency.

The speed preference concerns total time to successful task completion, not each command's magnitude or each
step's velocity. A preferred continuation may contain slower movements, pauses or preparation steps. Candidate
blocks are ranked by their executed continuation outcomes; the trajectory preference is not copied onto every
individual action as a local speed label. The experiment imposes no requirement that local speed or match rise
monotonically. Its ordering remains conditional on the fixed source suffix, rather than measuring the best
possible future strategy after each action.

| Role | Source episodes | Banks | Candidates | Successful and feasible | Preference pairs | Original screen rejections |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN branch | 40 | 120 | 1,439 | 923 | 3,334 | 21 |
| Calibration | 30 | 90 | 1,078 | 765 | 2,810 | 9 |
| Development | 30 | 90 | 1,078 | 776 | 2,844 | 6 |
| Remaining HELDOUT | 13 | 39 | 467 | 356 | 1,307 | 3 |

Roles are source-disjoint. TRAIN branch sources are disjoint from donor sources. Development covers all ten
LIBERO-Goal tasks; remaining HELDOUT covers nine tasks and excludes task 1. These evaluation episodes were
already examined by retrieval/physical-world experiments; they are held out from preference learning, not an
untouched final research test. No unseen-task or unseen-instruction generalization is measured.

Observed feasibility requires finite qpos/qvel, arm commands within [-1,1] with 1e-6 tolerance, gripper commands
in {0,1}, seven Panda arm joints within their ranges with at most 1e-3 rad excess, and no moving arm-link contact
with non-robot geometry at sampled control-step endpoints. Palm/finger task contact is allowed. This screen
does not establish continuous-time collision avoidance, actuator safety or real-robot legality.

All original branches succeed, but some exceed the declared joint-range tolerance. Two development banks and
one remaining-HELDOUT bank have no successful-feasible candidate. These banks are retained. Original fallback
preserves benchmark success but does not certify the conjunction of success and observed feasibility.

## Representation and supervision

Actual restored dual 512² camera images are encoded by frozen DINOv2/PCA/R6. Frozen original-query and text/task
features provide query inputs. The action residual receives normalized base r, PCA state, proprio and candidate
actions; it has no instruction input. Two query residuals operate in the same action representation space:
compatibility and preference. Both use 10 times cosine; compatibility also has a learned scalar bias.

The controlled grid is fixed/adapted r × compatibility-only/compatibility-plus-preference × seeds 0/1/2.
Each model trains 2,000 AdamW updates on episode-bootstrap TRAIN banks with batch size 16, learning rate 3e-4,
weight decay 1e-4, width/output 256, and residual anchor coefficient .01. Zero final-layer initialization
preserves initial cosine ordering. The original encoders, action representation and query files are unchanged.
TRAIN normalization is saved. Evaluation outcomes do not enter model inputs or checkpoint selection.

Compatibility targets are measured success AND observed feasibility. Preference supervision compares only
successful-feasible candidates: no worse completion steps and translation path within 1 mm, with fewer steps
or translation improvement greater than 1 mm. Incomparable pairs carry no ordering target. Rotation and jerk
are separately reported; they are not optimized by this target. The compatibility-only preference query stays
at its zero-initialized residual, although adapting r can alter its scores.

## Complete controlled ranking grid

Accuracy is over measured within-bank candidate pairs, with tied predictions receiving half credit. The three
seed results and their mean are shown; these are training seeds, not three independent policy evaluations.

| Representation / targets | Development preference accuracy, seeds 0/1/2 | Mean | Remaining HELDOUT, seeds 0/1/2 | Mean |
|---|---|---:|---|---:|
| Frozen original match | 46.98 | 46.98% | 52.79 | 52.79% |
| Fixed r / compatibility only | 46.98 / 46.98 / 46.98 | 46.98% | 52.79 / 52.79 / 52.79 | 52.79% |
| Fixed r / compatibility + preference | 63.47 / 62.80 / 62.48 | 62.92% | 63.05 / 60.60 / 60.83 | 61.49% |
| Adapted r / compatibility only | 49.54 / 48.80 / 52.95 | 50.43% | 52.79 / 55.24 / 54.17 | 54.07% |
| Adapted r / compatibility + preference | 70.15 / 67.86 / 70.75 | 69.58% | 73.14 / 68.02 / 68.78 | 69.98% |

For three-seed mean predictions, remaining-HELDOUT preference improvements over frozen matching are +8.70
percentage points for fixed r, with episode-bootstrap 95% interval [2.29,15.89], and +17.19 for adapted r,
with interval [8.45,24.76]. Intervals resample source episodes, preserving their three anchors; they do not
include uncertainty over tasks or the broader research selection process.

TRAIN preference accuracies are 82.31% for fixed-r preference learning and 87.65% for adapted-r preference
learning. The held-out drop remains substantial. Compatibility pair-ranking accuracies on remaining HELDOUT
are 56.08% and 55.71%, respectively. Their Brier scores are .199 and .255, and ten-bin ECE values are .163 and
.255. The compatibility head is not an effective safety/success certificate.

## Selection and actual-validation budget

Direct preference selection permits candidates whose predicted compatibility is at least the original's
predicted score, then selects highest preference. Shortlists contain original plus four alternatives selected
by preference; random controls enumerate equally sized subsets exactly. Alternatives include the geometric
and noise controls, not just retrieved donors. Actual acceptance requires success, observed feasibility, no
more steps than original and translation no greater than original +1 mm, then minimizes steps/path. If none
qualify, it returns the original. The unrestricted oracle uses the same acceptance rule on all candidates.

The budget comparison is retrospective over fully executed banks. It measures selection utility at a common
five-candidate budget; collection itself executed every candidate. True-compatible selection and shortlist
acceptance use measured labels, and cannot be deployed without an actual verifier.

| Selector | Development success / combined criterion | Complete time | Remaining HELDOUT success / combined criterion | Complete time |
|---|---:|---:|---:|---:|
| Original | 100.00% / 93.33% | 5.6633 s | 100.00% / 92.31% | 4.9154 s |
| Frozen match, direct | 86.67% / 76.67% | 5.7100 s | 82.05% / 71.79% | 5.0756 s |
| Fixed r + preference, direct gate | 84.81% / 78.52% | 5.7044 s | 94.02% / 84.62% | 4.9880 s |
| Adapted r + preference, direct gate | 77.41% / 71.48% | 5.7526 s | 85.47% / 82.91% | 5.0231 s |
| Random shortlist + actual acceptance | 100.00% / 95.02% | 5.5285 s | 100.00% / 94.17% | 4.8508 s |
| Fixed r + preference shortlist | 100.00% / 95.56% | 5.4994 s | 100.00% / 94.87% | 4.8321 s |
| Adapted r + preference shortlist | 100.00% / 94.81% | 5.5031 s | 100.00% / 94.87% | 4.8132 s |
| Actual all-candidate oracle | 100.00% / 96.67% | 5.4617 s | 100.00% / 97.44% | 4.7974 s |

Learned rows average all three seeds. Complete time includes failures at their executed horizons; shorter failed
executions do not count as faster task completion. Here shortlist rows all preserve task success, so their
complete-duration comparisons avoid that particular confound.

| Preference shortlist versus random | Time difference | Episode-bootstrap 95% interval | Complete translation difference | Interval |
|---|---:|---|---:|---|
| Fixed r, development | -0.0290 s | [-0.0412,-0.0166] | -3.71 mm | [-6.81,-0.62] |
| Adapted r, development | -0.0253 s | [-0.0410,-0.0083] | -7.58 mm | [-11.04,-4.18] |
| Fixed r, remaining HELDOUT | -0.0187 s | [-0.0359,-0.0029] | -3.19 mm | [-8.24,1.82] |
| Adapted r, remaining HELDOUT | -0.0375 s | [-0.0568,-0.0206] | -10.83 mm | [-16.53,-5.90] |

All three adapted-preference seeds have negative time and path differences versus random on remaining HELDOUT.
Relative to fixed-r preference learning, adaptation's remaining-HELDOUT time gain has interval
[-0.0432,0.0021] s, which includes zero; its path gain is -7.64 mm with interval [-15.97,-0.84].
Adaptation does not establish a consistent additional time gain across both evaluation sets.

Among successful-feasible selections on remaining HELDOUT, RMS jerk is 5.63 for original, 7.03 for random
shortlist, 7.03 for fixed-preference shortlist and 6.96 for adapted-preference shortlist. These conditional
subsets differ. Faster/shorter acceptance does not establish improved smoothness; jerk can rise relative to
the original. Rotation, all seeds, every selector and task are retained in the full report.

## Structural and resource observations

All twelve models preserve instruction independence of r under an instruction-feature swap: maximum difference
zero. On remaining HELDOUT, fixed r has norm 1, within-bank RMS .404 and effective rank 52.90. Adapted-preference
r has norm 1.98–2.50, within-bank RMS 1.45–1.79 and effective rank 11.46–13.14. Its sampled candidates remain
distinct; covariance becomes more concentrated. These measurements do not prove noncollapse in new settings.
Frozen-r preference/query learning already improves ranking, so this result does not attribute all preference
information to the action representation alone.

All actions equal their physical-world sources exactly. Maximum candidate restore proprio discrepancy is
1.19e-7; rendered-feature restore discrepancy is zero. Source R6/query file SHA256 values match encoding
provenance. Twelve checkpoints load strictly on CPU and reproduce saved GPU predictions within 2.86e-6.
Fixed-r compatibility predictions are identical with/without the additional preference loss, verifying that
the separate query heads isolate these targets when r is fixed. Feature schema has no outcome/state label fields.

CPU collection used five workers each on node-61 and node-63 with two math threads per worker. Only node-63 GPU 0
was used for smoke/full encoding and training, sequentially. Timed jobs total 1.237 CPU core-hours and .0439
GPU worker wall-hours, with sampled GPU peak 1,548 MiB and maximum process RSS 2,881,116 KiB. Small CPU coverage
and structural-smoke helpers and gateway overhead are outside those timed totals. Output size is approximately
49 MB. At 2026-10-02 15:59:30 UTC, both nodes report no remaining preference experiment workers.

Environment: project CPython 3.10.20, torch 2.7.1+cu126, MuJoCo 3.3.2, robosuite 1.4.0. CPU collection uses
MUJOCO_GL=disable; encoding uses the existing EGL runtime. No dependency changes were needed. All sixteen added package/entry-point files compile in the project environment;
the finalized typed training record imports successfully.

## Artifacts and reproduction

- Design: `plans/preference_feedback.md`.
- Configuration: `configs/preference/feedback_pilot.json`.
- Entry points: `scripts/collect_preference.py`, `scripts/encode_preference.py`, `scripts/train_preference.py`,
  `scripts/report_preference.py`; all run through `scripts/cluster_run.sh` in the project environment.
- Banks/manifests/features: `outputs/preference/pilot_v1/{banks,tasks,features}/`.
- Models, TRAIN normalization, source hashes and 48 role prediction files: `outputs/preference/pilot_v1/{models,predictions}/`.
- Full 130 selector cells and task/seed comparisons: `outputs/preference/pilot_v1/reports/comparison.{json,md}`.
- Checkpoint/source checks and group intervals: `reports/validation.json`; all-role ranking: `reports/all_role_ranking.json`.
- Compute observations: `reports/resources.json`, `reports/process_nnmc{61,63}.json`, `tmp/runlogs/preference/`.
- Supplementary observation scripts: `tmp/preference/{validate.py,validate.sh,train_metrics.sh,process_snapshot.sh}`.

Collection/encoding write new bank/feature paths exclusively. A repeat must use a distinct output directory;
checkpoint fitting must retain the same source/feature provenance. No refined labels, BC runs or RL jobs were
created by this experiment.

# Physical consequence pilot

Completed 2026-10-02 EDT. Shared task-independent prediction learns coarse candidate consequences, but the
measured precision and grasp-event coverage do not support replacing simulator validation. No action labels,
r/q/m weights or policies were changed. This experiment supplies an auxiliary action-effect mechanism for r;
q/m retain task compatibility and local match-based search retains candidate selection.

## Data and physical interface

Sources are official LIBERO-Goal simulation teleoperation demonstrations, regenerated under MuJoCo 3.3.2.
The existing EpisodeIndex uses HELDOUT fraction .1 and seed 0. The first 24 TRAIN episodes/task provide
stride-five ten-step demonstration windows; donor sources are the first twelve. TRAIN positions 12–15
provide branch sources, positions 24–26 calibration. Three HELDOUT episodes/task form development, already
used by the retrieval pilot. All remaining thirteen HELDOUT episodes across nine tasks form fresh confirmation.
Task 1 has no additional HELDOUT episode. Fresh confirmation is small and contains no unseen task or topology.

| role | transitions | source episodes | tasks | candidate banks |
|---|---:|---:|---:|---:|
| demo TRAIN | 5,135 | 240 | 10 | — |
| branch TRAIN | 1,439 | 40 | 10 | 120 |
| calibration | 1,078 | 30 | 10 | 90 |
| development | 1,078 | 30 | 10 | 90 |
| fresh | 467 | 13 | 9 | 39 |

All 313 source replays reach success. Demo windows end no later than first success. Branch anchors use
one-quarter, one-half and three-quarters of first-success duration, rounded to ten steps. Each bank contains
original, up to eight physical-neighbour arm chunks, half-scale arm commands and Gaussian arm perturbations
sigma .1/.3 clipped to [-1,1]. Each retains the source gripper sequence. These are local candidate actions,
not unrestricted strategies. The half-scale control is not time retiming.

The settled state is float32 [174], with actions float32 [10,7]. All tasks share nq=41, nv=37, joint order,
four object names and three fixture names. Schema fields are:

| field | width |
|---|---:|
| qpos / qvel | 41 / 37 |
| EEF xyz / rotation matrix | 3 / 9 |
| four object xyz / rotation matrices | 12 / 36 |
| two-finger grasp signatures | 4 |
| controller goal orientation | 9 |
| gripper actuator accumulator | 2 |
| three fixture poses | 21 |

Gripper control DOF is one. robosuite initializes its accumulator with one element, then the first control
step expands it to two actuator elements. PhysicalStateReader defines width by actuator count and reads only
a settled simulation; it checks array/schema equality. All ten schemas were observed against settled [174]
states. Metadata correction preserved every transition NPZ byte. Checkpoints and normalization use dimension
174, and twelve latent checkpoints load with strict state-dict checks. EEF/object fields precede the affected
schema offset; all physical comparison values agree exactly across the metadata correction.

Original endpoint replay maximum over heterogeneous state components is 0.0730366; restore maximum is
2.26e-6. The larger discrepancy is generalized velocity at task 1, episode 69, anchor 170. An independent
repeat gives worst xyz distance 4.56e-7 m, maximum qpos element difference 7.09e-5 and maximum object
rotation-matrix element difference 1.10e-5. This is recorded separately in ISSUE-041. It is not a 73 mm
translation error. No source was removed.

## Fixed model grid

All models share demonstration-TRAIN normalization: state std floor .01, delta std floor .001, action std floor
.05. Inputs are current structured physical state and the ten-step action chunk. No instruction, task ID,
continuation outcome or future state enters inputs. Training/calibration/development/fresh source sets are
disjoint; branch TRAIN sources are a subset of demo TRAIN. Success labels do not supervise training.

Persistence predicts unchanged state. Ridge predicts standardized physical delta from normalized state and
flattened actions, with penalty .01. Latent predictors use a state encoder, action-conditioned latent transition,
physical-delta decoder and start reconstruction decoder; hidden width 128, latent widths 32/128. Each ensemble
has members seeds 0/1/2 with source-episode bootstrap, batch 256, 2,000 AdamW updates, lr 3e-4, weight decay
1e-4. Mixed batches contain half demo and half actual TRAIN branches. Loss is delta MSE plus .1 start
reconstruction MSE plus .1 future-latent consistency MSE with detached future targets. This grid does not isolate
any individual loss or test SIGReg. Checkpoints use final updates, with no evaluation-based checkpoint selection.

Every candidate executes the full ten steps to measure its endpoint, even when success occurs inside the chunk.
Calibration and HELDOUT branches then use the original suffix until first success or demonstration end.
Development original/donor continuation labels reuse the verified retrieval banks. Their physical endpoints are
new executions. Perturbed and calibration/fresh continuation labels are measured by new executions. A successful
fixed-suffix branch is neither a policy rollout SR nor a certificate for other continuations.

## Physical forecast results

EEF error is mean Euclidean distance. Object error is mean per-row maximum across four objects. Worst xyz is
maximum across EEF and objects. Centered effect R² centers both predicted and measured xyz within each
same-start candidate bank; a state-only persistence predictor has zero. This isolates action distinctions from
predicting the start state. Intervals resample source episodes 1,000 times, preserving anchor/candidate dependence;
they are descriptive and conditional on these fitted models.

| model | split | EEF mm | max object mm | mean worst xyz mm [95% interval] | p95 worst xyz mm | effect R² |
|---|---|---:|---:|---|---:|---:|
| persistence | development | 70.69 | 30.68 | 71.00 [64.43, 78.32] | 126.75 | 0.000 |
| persistence | fresh | 75.06 | 29.69 | 75.56 [66.68, 83.85] | 127.88 | 0.000 |
| demo_ridge | development | 16.20 | 17.58 | 23.16 [21.28, 25.06] | 55.49 | 0.401 |
| demo_ridge | fresh | 15.39 | 20.93 | 25.47 [21.27, 30.89] | 61.15 | 0.416 |
| demo_latent32 | development | 26.96 | 18.43 | 31.12 [28.37, 33.80] | 56.73 | 0.489 |
| demo_latent32 | fresh | 27.27 | 20.83 | 33.63 [30.30, 36.95] | 57.91 | 0.463 |
| demo_latent128 | development | 17.77 | 12.25 | 20.68 [19.19, 22.17] | 41.83 | 0.621 |
| demo_latent128 | fresh | 17.22 | 14.57 | 22.18 [18.51, 26.93] | 46.93 | 0.645 |
| mixed_ridge | development | 13.82 | 16.62 | 21.74 [19.75, 24.09] | 52.35 | 0.574 |
| mixed_ridge | fresh | 13.36 | 20.29 | 24.14 [19.92, 29.72] | 59.45 | 0.572 |
| mixed_latent32 | development | 26.50 | 19.72 | 31.73 [29.23, 34.17] | 58.47 | 0.498 |
| mixed_latent32 | fresh | 26.67 | 21.47 | 33.72 [30.07, 37.64] | 59.88 | 0.485 |
| mixed_latent128 | development | 17.08 | 12.04 | 20.04 [18.59, 21.55] | 43.21 | 0.642 |
| mixed_latent128 | fresh | 16.95 | 15.10 | 22.26 [19.14, 26.58] | 44.51 | 0.658 |

The 128-width latent cells distinguish candidate consequences better than persistence and the linear cells
on the pooled xyz effect measurement. Ridge has lower EEF error than the latent models; latent-128 predicts
object positions better. Width 32 is worse than 128 on these physical targets. No single cell dominates all
physical or contact measurements.

Adding branches reduces ridge's mean worst xyz error on fresh sources by 1.33 mm, interval [-2.33,-0.47].
For latent-128 the paired change is +0.08 mm, interval [-1.19,1.28]; for latent-32 +0.10 mm, interval
[-2.27,2.51]. This grid establishes neither a neural prediction advantage from added branches nor a task-match
advantage. Mixed latent-128's fresh effect R² is .658, versus .645 for demo latent-128; no uncertainty interval
for that R² difference is measured.

Mixed latent-128 has fresh mean worst xyz error 22.26 mm. Its donor-only centered effect R² is .750,
with mean worst xyz error 21.25 mm. Larger sigma-.3 perturbations have mean worst xyz error 33.92 mm,
versus 21.06 mm for original. Fresh per-task means range from 17.53 to 45.73 mm; each task has only one or
two fresh sources. Prediction of coarse action consequences does not establish millimetre feasibility.

## Grasp transitions

The contact field means simultaneous contact with both fingerpads, not arbitrary collision or surface contact.
Fresh endpoints are positive on 7.55% of object signatures. Only 3.91% of signatures change from the start.
High overall accuracy therefore does not demonstrate event prediction. Brier uses clipped regression outputs.

| model | fresh grasp accuracy % | positive recall % | transition recall % | Brier |
|---|---:|---:|---:|---:|
| persistence | 96.09 | 58.16 | 0.00 | 0.0391 |
| demo_ridge | 97.27 | 75.89 | 35.62 | 0.0219 |
| demo_latent32 | 96.09 | 58.16 | 0.00 | 0.0297 |
| demo_latent128 | 96.25 | 60.28 | 4.11 | 0.0262 |
| mixed_ridge | 96.31 | 63.83 | 10.96 | 0.0231 |
| mixed_latent32 | 96.09 | 58.16 | 0.00 | 0.0299 |
| mixed_latent128 | 96.36 | 61.70 | 6.85 | 0.0271 |

Mixed latent-128 predicts only 6.85% of the 73 actual changed grasp signatures. Demo ridge predicts 35.62%;
persistence and both latent-32 cells predict none. Grasp-event prediction remains substantially incomplete.

## Forecast confidence

For each model, ensemble disagreement is maximum xyz standard-deviation norm across bodies. Calibration
computes error/(disagreement+1 mm), takes each calibration episode's maximum across candidates/anchors,
then its pooled 95th percentile using the higher order statistic. The resulting ratio scales each candidate's
forecast bound. Ridge and persistence have one member, so their bound is constant. This is an empirical
calibration diagnostic, without a distribution-free or physical safety guarantee.

All fourteen model/split cells retain zero candidates at all three predetermined 5/10/20 mm bounds. There are
no accepted candidates on which to measure an actual error violation or continuation failure rate; these rates
are undefined, not zero. The calibration cannot justify omitting simulator execution in this pilot.

| model | fresh minimum bound mm | fresh median bound mm | actual error ≤20 mm % |
|---|---:|---:|---:|
| persistence | 143.20 | 143.20 | 11.99 |
| demo_ridge | 87.61 | 87.61 | 52.68 |
| demo_latent32 | 55.96 | 160.49 | 18.63 |
| demo_latent128 | 29.48 | 80.02 | 58.67 |
| mixed_ridge | 82.50 | 82.50 | 56.10 |
| mixed_latent32 | 51.68 | 138.62 | 19.49 |
| mixed_latent128 | 32.45 | 91.37 | 52.68 |

Demo latent-128 has 58.67% of realized endpoints within 20 mm, yet its smallest calibrated fresh bound is
29.48 mm. Mixed latent-128's minimum is 32.45 mm and median 91.37 mm. This is conservative rejection,
not an observed false-acceptance rate. A calibrated coordinate error bound would still not establish rotation
consistency, collision freedom along the chunk, contact correctness or task success.

Fresh original fixed-suffix success is 39/39; donor 252/311, half-scale 30/39, sigma-.1 35/39 and sigma-.3
31/39. These measured outcomes remain separate from predicted physical errors and do not enter learning.
No refined actions were accepted, no dataset was relabelled and no policy SR was measured.

## Boundaries

This experiment uses privileged simulator state, not a learned reconstruction from RGB/RGB-D. State inputs
include pose, velocity and controller context; which groups can be omitted has not been tested. The decoder
predicts unconstrained physical arrays, without enforcing joint/body consistency or SO(3). Only ten-step
endpoints are predicted, not intervening collision/contact histories, multi-chunk rollouts or alternate strategies.
The donor bank and gripper choices limit exploration. Member disagreement can miss shared errors.

Full-state forecast accuracy is a stricter auxiliary target than the framework's required same-observation
separability. These results neither establish nor invalidate learned task-equivalence matching, semantic
elimination of redundant operations, multi-task RL benefit or action-label optimization. q/m were not evaluated
with these latent encodings. The completed pilot stops before vision training, learned action optimization,
RL or BC. Open physical limitations are recorded in ISSUE-042; subsequent work is scoped in OPT-012.

## Execution and resources

Compute uses the project .venv: CPython 3.10.20, torch 2.7.1+cu126, MuJoCo 3.3.2, robosuite 1.4.0.
CPU physics uses MUJOCO_GL=disable and OMP/MKL/OpenBLAS thread limits two. Full collection uses five CPU
workers each on node-61 and node-63. Every GPU fitting run uses only node-63 GPU 0 through cluster_run and
its GPU guard. No node uses more than one project GPU for this pilot.

Final six-cell fitting/prediction takes 57.21 s. Mixed latent-128 three-member inference
median is 0.795 ms, including normalization, GPU transfer, model mean and host output, on one candidate.
Ridge inference is about .02 ms. Observed CPU physics ten-step medians range 124–147 ms/task. Prediction
latency excludes obtaining the physical state, sampling candidates and any vision reconstruction.

Timed full collection uses 0.780 CPU core-hours. GPU worker wall
time totals 0.041 GPU-hours, including smoke, initial fit,
schema-consistent regeneration and two failed shard-validation preflights. GPU sampled peak is 571 MiB;
PyTorch peak allocation is 39.77 MiB. Peak CPU collector RSS is
2.01 GiB per worker. Main outputs occupy about 47.8 MiB.
Additional schema/replay diagnostic CPU time and gateway overhead are outside the timed-worker totals.


Entry points after project environment setup, using a new output version for collection:

```sh
$CEAM_PYTHON scripts/collect_physical_world.py --config configs/world/physical_pilot.json --tasks 0 1 2 3 4 5 6 7 8 9
$CEAM_PYTHON scripts/train_physical_world.py --config configs/world/physical_pilot.json --cuda-device 0
$CEAM_PYTHON scripts/report_physical_world.py --config configs/world/physical_pilot.json
```

Remote commands run through `scripts/cluster_run.sh`; shell wrappers and copied configurations are preserved
under `tmp/physics_world/` and `tmp/runlogs/physics_world/`. Collection writes exclusively and refuses existing
transition files. No dependencies were added. A fresh four-source collection using the corrected reader completes with [174] states, 24 demo windows
and 12 candidates in each non-demo role. Original endpoint discrepancy is 7.69e-15 and restore discrepancy
6.64e-15. Shape/finiteness, source separation, all-task schemas, model grid
coverage, strict checkpoint loading and byte-preserving schema correction were observed. Both compute nodes
have no remaining pilot collector/trainer/reporter process at closure.

## Evidence

- `configs/world/physical_pilot.json`: fixed grid and project-scoped data paths.
- `ceam/world/`: state schema, normalization, transition records, model interface, collection/training/report owners.
- `outputs/physics_world/pilot_v1/data/` and `tasks/`: transition sets, task/source manifests, schemas and replay probes.
- `models/data_provenance.json`: source roles and transition SHA256; `models/training.json`: all fit cells and resources.
- `predictions/`: every calibration/development/fresh member prediction.
- `reports/comparison.json`: all fourteen pooled cells, task/stratum metrics, paired intervals and 42 trust comparisons.
- `reports/observations.json`: finite shapes, loaded checkpoints, actual bounds and separate continuation outcomes.
- `reports/resources.json` and `tmp/runlogs/physics_world/`: CPU/GPU time, RSS, allocation and sampled GPU usage.
- `tasks/schema_correction.json`: all-task settled [174] widths and unchanged transition bytes.

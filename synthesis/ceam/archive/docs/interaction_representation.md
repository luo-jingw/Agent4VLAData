# Representation evaluation without task success

One evaluation suite compares any state encoder ψ, with or without an effect model Δ, on five groups of metrics that do not involve task success (plan.md, "评估（不涉及任务成功率）" 1–5): collapse and conflict, prediction, linear probes of the scene state, the distribution of action effects, and the frame domain. Every metric is unchanged when ψ and Δ̂ are multiplied by one positive constant, except the raw SIGReg statistic, which is reported next to its scale-free version.

## Code

| Class | File | Role |
|---|---|---|
| `RepresentationModel`, `FrozenFeatureModel`, `TrunkDynamicsModel`, `StageOneTwoModel`, `JointInteractionModel`, `FrozenPcaDeltaModel`, `ModelProvenance` | `ceam/diagnostics/representation_model.py` | `encode(frames) -> ψ`, `effect(frames, ψ, actions) -> Δ̂` |
| `FrameSource`, `StoreFrameSource`, `CacheFrameSource`, `ModelEmbeddings`, `EmbeddingPass` | `ceam/diagnostics/representation_embeddings.py` | runs all models over one read of each chunk of frames |
| `TransitionSetConfig`, `RepresentationEvalConfig` | `ceam/diagnostics/representation_eval_config.py` | evaluation sets and fixed data (JSON) |
| `EvaluationTransitions`, `DemoRenderFrames`, `DataAStates` | `ceam/diagnostics/representation_eval_data.py` | selected transitions; rendered demonstration frames with their sidecar identity and restore error; data-A states |
| `SceneLabelTable` | `ceam/diagnostics/scene_label_table.py` | per-frame `SceneState` of a store from its `scene.npz` sidecars |
| `StateGroups` | `ceam/diagnostics/state_groups.py` | transitions grouped by their current frame (one state, several actions) |
| `GroupFolds` | `ceam/diagnostics/group_folds.py` | cross-validation folds that keep a group in one fold |
| `RidgeReadout`, `RidgeResult` | `ceam/diagnostics/ridge_readout.py` | ridge regression, λ by grouped CV |
| `LogisticReadout`, `CrossFittedScores` | `ceam/diagnostics/logistic_readout.py` | L2 logistic regression, λ by grouped inner CV, cross-fitted scores, AUC |
| `RepresentationCollapseMetrics`, `SpectrumSummary`, `SigRegSummary`, `ConflictDiagnostics` | `ceam/diagnostics/representation_collapse_metrics.py` | 评估 1, collapse and conflict |
| `EffectPredictionMetrics`, `PredictionSummary`, `WithinStateSummary` | `ceam/diagnostics/effect_prediction_metrics.py` | 评估 2, prediction |
| `RepresentationProbeMetrics`, `ProbeTargetBuilder`, `ProbeTargets`, `ProbeInput`, `ProbeScore` | `ceam/diagnostics/representation_probe_metrics.py` | 评估 3, linear probes |
| `EffectDistributionMetrics`, `SceneChangeLabeler`, `SceneChangeLabels`, `SufficiencySummary`, `ClassifierScore`, `DisplacementScore` | `ceam/diagnostics/effect_distribution_metrics.py` | 评估 4, effect distribution |
| `FrameDomainMetrics`, `DomainDistance`, `DomainScores`, `DomainSeparability` | `ceam/diagnostics/frame_domain_metrics.py` | 评估 5, frame domain |

Entry points:

- `scripts/evaluate_representation.py {collapse,prediction,probe,effects,domain} --config <json> --model ... --out-dir <dir> --device cuda:N --code-commit <sha>` writes `<dir>/<subcommand>.json` (`{"provenance", "results"}`; non-finite values are `null`); `report --out-dir <dir>` writes `<dir>/report.md` from the JSON files present. An existing output file is refused.
- `scripts/check_representation_metrics.py --device <dev>`: the unit checks below.

## Models

`--model` is repeatable; the order is kept.

| Option | Class | ψ | Δ̂ |
|---|---|---|---|
| `NAME frozen` | `FrozenFeatureModel` | frozen DINOv2 pooled feature: per view the float32 mean of the 64 tokens, views concatenated, 1,536-d (InteractionMonitor's R0) | none |
| `NAME stage_one_two S1.pt S2.pt` | `StageOneTwoModel` | stage-one `StateTrunk` (`CheckpointStore`) | stage-two `DynamicsHead` (`StageTwoCheckpointStore`); construction fails unless the stage-two config names S1 |
| `NAME joint CKPT.pt` | `JointInteractionModel` | joint trunk (`InteractionCheckpointStore`) | joint dynamics head |
| `NAME frozen_pca CKPT.pt` | `FrozenPcaDeltaModel` (R6) | the fixed PCA projection of the R0 feature to 256 dims, from the checkpoint's own copy (`InteractionCheckpointStore.load_fixed_encoder`; docs/interaction_training.md, "Fixed encoder (R6)") | the dynamics head trained on it |

- `joint` refuses a fixed-encoder checkpoint and `frozen_pca` a trainable-trunk checkpoint (`ValueError`).
- ψ of R6 reads no proprio; its Δ reads proprio like the other heads. Caches come from the run's config.

- Inputs follow `FrameBatch`: float16 tokens, proprio standardized with the feature-cache statistics; action chunks standardized with the action-cache statistics. The script checks that every model's feature cache and action cache equal the config's.
- Both modules run in eval mode, float32, without autocast or gradient.
- Δ̂ = Δ(o, ψ(o), a) from the current frame o; z = ψ(o′) − ψ(o). Both are computed on the frames of the evaluated store, so a model trained on video frames is evaluated on simulator renders wherever the store holds renders.

## Evaluation config

`RepresentationEvalConfig` (JSON with exactly its fields; `$CEAM_ROOT` / `$CEAM_DATA_ROOT` expanded). A transition set is `{name, store_dir, heldout, kinds, stay_kinds, counterfactual, scene_labels, fold_group}`:

- `heldout`: null selects every transition, otherwise the transitions a `HeldoutRule` holds out;
- `kinds`: kinds kept (empty: all); `stay_kinds`: stay transitions;
- `counterfactual`: transitions sharing a current frame are the actions of one state;
- `scene_labels`: per-frame `scene.npz` sidecars exist (the demo_renders format, also written by `collect_interactions.py`);
- `fold_group`: what grouped folds hold out together, `state` (one (task, episode, t)) or `episode` (one (task, episode)).

Fixed data: the feature cache and action cache (`libero_goal`), `demo_renders` with its held-out rule (shard prefix `heldout`), the checks files `outputs/analysis/demo_renders/<shard>_checks.npz`, the restore-error threshold 1e-4, the data-A store `demo_perturbations`, K3 draws (4,096, seed 0), folds (outer 5, inner 3, seed 0).

## Metrics

Notation: a set holds transitions (o, a, o′); z = ψ(o′) − ψ(o); Δ̂ = Δ(o, ψ(o), a). Covariances, norms and medians are float64. "Effective rank" is exp(entropy of the covariance eigenvalues divided by their sum), the definition of `StateDistributionMetrics`; "top share" is the largest eigenvalue divided by their sum.

### Readouts

- **Ridge** (`RidgeReadout`): features standardized with the fitting rows' per-dimension mean and std (a dimension with std ≤ 1e-9 × the largest is set to 0), targets centered; objective (1/n)‖Y − XW‖² + λ‖W‖²; λ ∈ {1e-8, …, 1e3} per target group, the grid value with the largest pooled out-of-fold R² over grouped folds (`GroupFolds`: distinct groups permuted with seed 0, group j to fold j mod K); refit on all fitting rows.
- **Logistic** (`LogisticReadout`): standardized features, mean log-loss + λ‖w‖² (bias free), Newton's method in float64; λ ∈ {1e-7, …, 1} by the smallest held-out log-loss over 3 grouped inner folds (seed + 1). Cross-fitting: 5 grouped outer folds (seed); every row is scored by the model fitted on the other folds.
- **AUC**: Mann–Whitney U / (n₊ n₋), average ranks for ties.
- **Pooled R²** of a group of columns: 1 − Σ‖ŷ − y‖² / Σ‖y − ȳ‖² over rows and the group's columns, ȳ the per-column mean of the evaluated rows.

Both readouts standardize their inputs, so their scores do not change when a feature dimension is rescaled.

### 1. Collapse and conflict (`collapse`)

Per model and transition set:

- ψ over the set's distinct frames, z and Δ̂ over its transitions: effective rank and top share.
- SIGReg statistic (`SigRegLoss(d)`, directions seeded 0 as `ConflictCriteria.sigreg_statistic`) of ψ (frames) and Δ̂ on at most 4,096 rows drawn with seed 0:
  - raw: the vectors as they are, the value the training objective sees; it depends on the scale;
  - normalized: the vectors divided by s, s² = tr Cov / d; scale-free. The comparison uses this one.
  - floor 1.06 / N.
- Conflict diagnostics (`ConflictDiagnostics`, equal to `ConflictCriteria.diagnose` / `evaluate` to 0 in the checks):
  - Var(z)/Var(ψ) = tr Cov(z) / tr Cov(ψ(o));
  - correlation of ⟨ψ(o), u_k⟩ and ⟨ψ(o′), u_k⟩ along the principal directions u_k of Cov(ψ(o)): eigenvalue-weighted mean, min, median;
  - median adjacent-frame cosine cos(ψ(o), ψ(o′)), raw and centered on the mean ψ(o);
  - stay rows: median stay ‖Δ̂‖ / median non-stay ‖z‖ (K1's second half as a ratio; holds when > 1), with median stay ‖z‖ / median non-stay ‖z‖ as the reference;
  - median ‖Δ̂‖/‖z‖ and set R² (K1, K2); `CriteriaValues` of each set with the K flags as if the set were the monitor's held-out set.
- K3 (per model): E‖ψ(o_{t+10}) − ψ(o_t)‖² / E‖ψ(o_i) − ψ(o_j)‖² on the pairs of `InteractionMonitor` (4,096 held-out demo_renders transitions and 4,096 pairs of current frames of different episodes, seed 0), with R0's ratio on the same pairs.

### 2. Prediction (`prediction`)

Per model with Δ and set, all transitions and each kind:

- mean and median cos(Δ̂, z), median ‖Δ̂ − z‖/‖z‖, set R² = 1 − Σ‖Δ̂ − z‖²/Σ‖z − z̄‖².
- Counterfactual sets (states = transitions sharing a current frame):
  - within-state R² = 1 − Σ_s Σ_a ‖(Δ̂_sa − Δ̂̄_s) − (z_sa − z̄_s)‖² / Σ_s Σ_a ‖z_sa − z̄_s‖², over states with ≥ 2 actions;
  - within-state ranking agreement: per state with ≥ 3 actions, the Spearman correlation between the pairwise distances ‖Δ̂_a − Δ̂_b‖ and ‖z_a − z_b‖ over the pairs a < b; mean and median over states.

### 3. Linear probes (`probe`)

- **Frames**: all 52,895 demo_renders frames; fit on the 47,601 TRAIN frames (grouped CV by episode), score R² on the 5,294 HELDOUT frames.
- **Targets** (`ProbeTargetBuilder`, from the sidecar `SceneState`). Within a group every column has one unit and the group R² is pooled in that unit:
  - object_position (m), all four objects and per object; object_orientation, the first two rotation-matrix columns (unitless), all objects and per object;
  - fixture/<joint>, one joint each (m or rad); fixtures, the four joints each divided by its TRAIN std;
  - arm_joints (rad), gripper (m), eef_position (m), eef_orientation (rotation-matrix columns).
- **Inputs**: proprio [9] (the store's proprio, standardized), each model's ψ, and [proprio ⊕ ψ]. R0 enters as a model.
- **Object beyond the arm**:
  - incremental R² = R²([proprio ⊕ ψ]) − R²(proprio) on the object groups, all frames;
  - t0: fit on the 390 TRAIN first frames, score on the 43 HELDOUT first frames, object position groups;
  - ungrasped: per object, fit and score only on frames where that object is not grasped; pooled over objects as 1 − ΣSSE/ΣSST.
- The labels are read one physics substep after the frame's observation (`docs/demo_renders.md`): arm qpos differs from the observed proprio by a median 8.9e-4 rad, object positions by up to 3.6 mm. This bounds how close to 1 the arm-joint R² of an image representation can come, and is small against the object-position spread.

### 4. Effect distribution (`effects`)

- **Sufficiency**, counterfactual sets, for Δ̂ and z (`SufficiencySummary`):
  - per-state effective rank of the state's centered effects (mean, median over states with ≥ 2 actions);
  - effective rank of the pooled within-state covariance;
  - within-state share Σ_s Σ_a ‖e_sa − ē_s‖² / Σ ‖e − ē‖².
- **Scene changes**, sets with scene labels (`SceneChangeLabeler`, SceneState of o and o′):
  - object moved: some object's ‖p′ − p‖ > 1 cm; moved/<object> per object (one-vs-rest);
  - fixture changed: some fixture joint |q′ − q| > 0.01 in its unit (1 cm for the drawer slides, 0.01 rad for the stove knob);
  - grasp changed: some object's grasp flag differs.
  - AUC of the cross-fitted logistic readout from Δ̂ and from z; no classifier when a class spans fewer fold groups than outer folds (`ClassifierScore.note`).
  - R² (pooled, m) of the cross-fitted ridge readout from Δ̂ and from z to the object displacement vector [4 × 3] and to the end-effector displacement [3].
  - Fold groups: `state` (one (task, episode, t)) for counterfactual sets, `episode` for demo_renders HELDOUT, whose states follow each other in time.

### 5. Frame domain (`domain`)

- Pairs: every demo_renders frame (render) with feature-cache row `cache_row` (the dataset video frame of the same state), 52,895; and the 6,240 data-A current frames (render) with the video frame of their (episode, t).
- Distance: median, p10, p90 over states of ‖ψ(video) − ψ(render)‖, divided by the median ‖z‖ of the 48,565 demo_renders chunks (render frames). The median ‖z‖ on the video frames of the same chunks is listed.
- Separability: logistic readout video vs render, cross-fitted over 5 folds grouped by episode (both frames of a state in one fold); accuracy of logit > 0 ⇔ video and AUC.
- Subsets: all states; the 249 frames with restore proprio error > 1e-4 (their render shows a slightly different state from the video frame) and the others; the HELDOUT episodes; data A (its own cross-fitting by episode). The 19 frames whose wrist-view token cosine between render and video is below 0.8 (`docs/demo_renders.md`) are in the "all" and "≤ 1e-4" rows except the 6 of episodes 78 and 91, which have restore error > 1e-4.

## Unit checks

`scripts/check_representation_metrics.py`, commit `6cd75a8`, node-69, on the CPU and on CUDA 0 (logs `tmp/runlogs/s4p5/check_metrics_6cd75a8_{cpu,cuda0}.log`); both print identical output.

| Check | Measured | Expected |
|---|---|---|
| effective rank / top share, exact spectra (n 5,000, d 64): one direction; 8 equal; 64 equal; geometric 0.5 over 16; k⁻¹ over 64 | 1.000000 / 1.000000; 8.000000 / 0.125000; 64.000000 / 0.015625; 3.999262 / 0.500008; 29.117882 / 0.210797 | identical (analytic entropy rank) |
| ridge probe, planted target, population R² 0.8 (p 64, fit 20,000, evaluate 5,000) | 0.7931 (CV 0.8030) | 0.8 |
| ridge probe, target independent of the features | 0.0001 (λ 10) | ≈ 0 |
| the same with feature dims rescaled by 1e3 / 1e-3 | 0.7931, 0.0001 | unchanged |
| incremental R²: R²(proprio), R²(proprio ⊕ ψ), increment (population 0.3, 0.8, 0.5) | 0.2900, 0.7925, 0.5024 | 0.3, 0.8, 0.5 |
| ridge at the chosen λ 1e-3 vs ordinary least squares | max \|Δ prediction\| 3.95e-3 (prediction std 1.01) | small |
| planted effects, 200 states × 16 actions, d 32 (set R² / within-state R² / ranking ρ): Δ̂ = z | 1.0000 / 1.0000 / 1.0000 | 1 / 1 / 1 |
| Δ̂ = state mean + 0.5 noise | 0.7550 / −0.2450 / −0.0025 | 0.75 / −0.25 / 0 |
| Δ̂ = z + 0.5 noise | 0.9509 / 0.7501 / 0.7795 | 0.95 / 0.75 / < 1 |
| Δ̂ = 3 z | −3.0146 / −3.0000 / 1.0000 (cos 1.0000) | −3 / −3 / 1 |
| within-state share of z (state spread 2, within spread 1) | 0.1843 | (A − 1)/(5A) = 0.1875 |
| per-state effects of exact rank 3 | per-state rank 3.000000 | 3 |
| AUC: separable scores; random scores (200,000); Gaussian classes d′ = 1; ties | 1.000000; 0.4986; 0.7611; 0.5000 | 1; 0.5; Φ(1/√2) = 0.7602; 0.5 |
| cross-fitted logistic: separable classes; planted logistic model (true logit as score); labels independent of x | 1.0000; 0.8127 (0.8163); 0.4916 | 1; ≤ 0.8163; ≈ 0.5 |
| `conflict` vs `ConflictCriteria.diagnose` / `evaluate` (Var ratio, direction correlations, adjacent cosines, ‖Δ̂‖/‖z‖, R², stay ratio) | differences 0 | 0 |
| every metric on (ψ, Δ̂) vs (c ψ, c Δ̂), c = 7.3 and 1e-3 | differences ≤ 2.2e-16, except SIGReg raw (0.115 → 1.068, 0.401) | unchanged except SIGReg raw |
| grouped folds: groups split across folds | 0 | 0 |
| rotation-matrix columns vs `mujoco.mju_quat2Mat`, 1,000 quaternions | max \|difference\| 5.6e-16 | 0 |
| scene-change labels on planted displacements (2 cm, 0.5 cm, 0, 1.1 cm; knob 0.3; grasp flip) | as planted | as planted |
| domain distance, ‖offset‖ 0.25, median ‖z‖ 0.5 | 0.500000 | 0.5 |

Cross-fitted AUC without any signal falls slightly below 0.5 (0.4916 here): each outer fold's intercept is fitted on the other folds, whose class balance is opposite to the held-out fold's. An AUC below 0.5 on a small set therefore means no signal.

## Baseline: R0 and R1

- **Models**: R0 = `frozen`; R1 = `stage_one_two outputs/runs/goal_instruction_contrast/checkpoints/step_005000.pt outputs/runs/goal_dynamics_perturb/checkpoints/step_003000.pt` (stage one on dataset video frames; Δ on demonstrations + data A).
- **Config**: `configs/representation_eval/phase5_sets.json`.
- **Code**: commit `6cd75a8`. Host node-69, NVIDIA L40S: CUDA 0 ran collapse (32 s), prediction (13 s), probe (33 s), effects (508 s) in sequence; CUDA 1 ran domain (573 s). Both GPUs were idle at launch.
- **Launcher**: `tmp/s4p5/launch_baseline.sh`; logs `tmp/runlogs/s4p5/baseline_<subcommand>.log`.
- **Outputs**: `outputs/analysis/representation_eval/r0_r1_phase5/{collapse,prediction,probe,effects,domain}.json` and `report.md` (every table, including per-kind prediction rows and all classifier rows).

### Evaluation sets

| Set | Store, selection | Transitions | Frames | States | Stay |
|---|---|---|---|---|---|
| demo_renders_heldout | demo_renders, shards `heldout_*` | 4,864 | 5,294 | — | 0 |
| rollouts_r0_heldout | rollouts_r0, starts 80–99 | 4,470 | 4,670 | — | 533 |
| consistency_candidates | all (129 HELDOUT states × 14 counterfactual + 32 π) | 5,934 | 6,063 | 129 | 129 |
| consistency_counterfactual | kinds demo, stay, perturbed | 1,806 | 1,935 | 129 | 129 |
| consistency_proposal | kind proposal (π) | 4,128 | 4,257 | 129 | 0 |
| phase9_chunks | all (129 HELDOUT states × 4 shards × demo, mean, retrieval, match) | 2,064 | 2,580 | 516 | 22 |
| trial_demo_branch_eval | phase-4 trial, HELDOUT episodes 4 and 131, 6 time points, 32 smooth random chunks | 384 | 396 | 12 | 0 |
| trial_demo_branch | phase-4 trial, TRAIN states, kind random_branch (8 per state) | 256 | 288 | 32 | 0 |
| trial_throughput_5pgpu | phase-4 trial, TRAIN states of all 10 tasks, kind random_branch (8 per state) | 640 | 720 | 80 | 0 |
| trial_resim_candidates | phase-4 trial, trial_resim_branch kind resim_candidate (32 π candidates, task 6) | 384 | 396 | 12 | 0 |

demo_renders and the four trial stores carry scene sidecars, so the scene-change readouts run on demo_renders_heldout and the trial sets. The trial stores are the small trial collections of `collect_interactions.py`; the shards read are listed in each JSON's `provenance.store_shards`.

### 1. Collapse and conflict

K3 on the held-out rendered demonstration trajectories (4,096 pairs): R0 0.372, R1 0.095.

Effective rank / top share of ψ over the set's frames and of z; SIGReg of ψ, normalized (raw):

| Set | R0 ψ | R0 z | R0 SIGReg ψ | R1 ψ | R1 z | R1 Δ̂ | R1 SIGReg ψ | R1 SIGReg Δ̂ |
|---|---|---|---|---|---|---|---|---|
| demo_renders_heldout | 56.9 / 0.172 | 149.7 / 0.078 | 0.914 (0.581) | 16.6 / 0.115 | 15.2 / 0.158 | 14.4 / 0.175 | 0.027 (0.026) | 0.072 (0.250) |
| rollouts_r0_heldout | 80.0 / 0.148 | 203.4 / 0.060 | 0.793 (0.534) | 14.0 / 0.192 | 16.9 / 0.131 | 12.6 / 0.240 | 0.037 (0.042) | 0.137 (0.292) |
| consistency_counterfactual | 66.6 / 0.159 | 118.3 / 0.086 | 0.921 (0.585) | 17.5 / 0.092 | 16.9 / 0.143 | 15.0 / 0.169 | 0.026 (0.029) | 0.060 (0.196) |
| consistency_proposal | 45.5 / 0.162 | 79.7 / 0.098 | 0.950 (0.597) | 16.5 / 0.105 | 12.1 / 0.252 | 11.1 / 0.296 | 0.029 (0.033) | 0.059 (0.240) |
| phase9_chunks | 50.9 / 0.165 | 89.7 / 0.091 | 0.951 (0.598) | 16.9 / 0.099 | 12.8 / 0.237 | 12.0 / 0.263 | 0.027 (0.032) | 0.059 (0.233) |
| trial_throughput_5pgpu | 56.6 / 0.159 | 123.0 / 0.064 | 0.897 (0.576) | 16.2 / 0.110 | 14.1 / 0.174 | 13.4 / 0.197 | 0.030 (0.030) | 0.102 (0.222) |

The SIGReg floor 1.06 / N is 2.6e-4 at N = 4,096. On the small sets (12–32 states) the spectra are bounded by the number of distinct states; `report.md` lists them.

Conflict diagnostics (R0 has no Δ̂):

| Set | Model | Var(z)/Var(ψ) | direction corr. weighted / min / median | adjacent cos raw / centered | median ‖Δ̂‖/‖z‖ | R² | stay ‖z‖ / non-stay ‖z‖ | stay ‖Δ̂‖ / non-stay ‖z‖ |
|---|---|---|---|---|---|---|---|---|
| demo_renders_heldout | R0 | 0.753 | 0.639 / −0.071 / −0.008 | 0.914 / 0.615 | — | — | — | — |
| demo_renders_heldout | R1 | 0.187 | 0.919 / −0.009 / 0.173 | 0.975 / 0.975 | 0.976 | 0.854 | — | — |
| rollouts_r0_heldout | R0 | 0.579 | 0.716 / −0.113 / 0.145 | 0.927 / 0.715 | — | — | 0.453 | — |
| rollouts_r0_heldout | R1 | 0.247 | 0.877 / 0.301 / 0.462 | 0.981 / 0.981 | 0.851 | 0.315 | 0.171 | 0.166 |
| consistency_counterfactual | R0 | 0.963 | 0.634 / −0.135 / 0.005 | 0.897 / 0.525 | — | — | 0.488 | — |
| consistency_counterfactual | R1 | 0.416 | 0.838 / −0.069 / 0.049 | 0.928 / 0.928 | 0.980 | 0.776 | 0.221 | 0.197 |
| phase9_chunks | R0 | 0.842 | 0.682 / −0.032 / 0.001 | 0.909 / 0.588 | — | — | 0.476 | — |
| phase9_chunks | R1 | 0.267 | 0.907 / −0.025 / 0.020 | 0.937 / 0.939 | 0.976 | 0.837 | 0.417 | 0.303 |
| trial_throughput_5pgpu | R1 | 0.389 | 0.829 / −0.067 / 0.017 | 0.974 / 0.975 | 0.901 | 0.517 | — | — |
| trial_resim_candidates | R1 | 0.237 | 0.911 / −0.170 / 0.003 | 0.996 / 0.996 | 2.211 | 0.724 | — | — |

Read with the pre-registered thresholds (`CriteriaValues` per set, all set transitions): R1 meets no criterion on any set except K1 on trial_resim_candidates (median ‖Δ̂‖/‖z‖ 2.21: π candidates at 12 states of task 6, whose true effects are small). SIGReg(Δ̂) of R1 (0.196–0.292 raw) exceeds 10 × floor on every set, so K2 would hold wherever R² < 0.3; R² is 0.315 on rollouts_r0_heldout and 0.310 on trial_demo_branch_eval.

### 2. Prediction (R1)

| Set | cos mean / median | median rel. error | R² | within-state R² | ranking ρ mean / median |
|---|---|---|---|---|---|
| demo_renders_heldout | 0.924 / 0.983 | 0.245 | 0.854 | — | — |
| rollouts_r0_heldout | 0.484 / 0.629 | 0.924 | 0.315 | — | — |
| rollouts_r0_heldout, rollout_stay | 0.086 / 0.083 | 1.321 | 0.009 | — | — |
| consistency_candidates | 0.934 / 0.983 | 0.234 | 0.847 | 0.670 | 0.926 / 0.947 |
| consistency_counterfactual | 0.882 / 0.969 | 0.313 | 0.776 | 0.691 | 0.852 / 0.896 |
| consistency_proposal | 0.957 / 0.986 | 0.219 | 0.900 | 0.181 | 0.806 / 0.842 |
| phase9_chunks | 0.930 / 0.983 | 0.234 | 0.837 | 0.295 | 0.718 / 0.829 |
| trial_demo_branch_eval | 0.654 / 0.798 | 0.758 | 0.310 | 0.235 | 0.540 / 0.557 |
| trial_demo_branch | 0.712 / 0.836 | 0.726 | 0.509 | 0.406 | 0.553 / 0.624 |
| trial_throughput_5pgpu | 0.678 / 0.829 | 0.693 | 0.517 | 0.442 | 0.592 / 0.649 |
| trial_resim_candidates | 0.429 / 0.602 | 2.115 | 0.724 | −0.166 | 0.626 / 0.635 |

- On rendered frames R1 reproduces its video-frame accuracy on the HELDOUT demonstration chunks: median relative error 0.245 and mean cos 0.924, against 0.248 and 0.924 on the feature-cache frames (`docs/dynamics_data.md`, +A head).
- consistency_counterfactual per kind equals the phase-7 simulator evaluation of the same head (`docs/dynamics_data.md`): demo 0.230 / 0.951, stay 0.729 / 0.683 (median relative error / mean cos).
- Smooth random chunks from demonstration states (trial sets) are predicted far worse than demonstration-like chunks: R² 0.31–0.52, within-state R² 0.24–0.44.

### 3. Linear probes (HELDOUT R²)

All frames (fit 47,601 TRAIN, score 5,294 HELDOUT):

| Target | proprio | R0 | proprio ⊕ R0 | R1 | proprio ⊕ R1 |
|---|---|---|---|---|---|
| object_position | 0.396 | 0.936 | 0.939 | 0.924 | 0.928 |
| — bowl | 0.323 | 0.938 | 0.941 | 0.945 | 0.947 |
| — cream cheese | 0.043 | 0.801 | 0.801 | 0.692 | 0.692 |
| — wine bottle | 0.563 | 0.952 | 0.956 | 0.930 | 0.939 |
| — plate | 0.328 | 0.942 | 0.941 | 0.938 | 0.939 |
| object_orientation | 0.220 | 0.339 | 0.340 | 0.308 | 0.312 |
| — bowl | 0.140 | 0.269 | 0.269 | 0.176 | 0.173 |
| — cream cheese | −0.011 | −0.046 | −0.049 | −0.009 | −0.009 |
| — wine bottle | 0.565 | 0.816 | 0.819 | 0.768 | 0.783 |
| — plate | 0.006 | 0.105 | 0.109 | 0.264 | 0.270 |
| fixtures | 0.191 | 0.933 | 0.936 | 0.968 | 0.969 |
| — top drawer | 0.096 | 0.984 | 0.984 | 0.993 | 0.994 |
| — middle drawer | 0.226 | 0.929 | 0.928 | 0.949 | 0.950 |
| — stove knob | 0.243 | 0.889 | 0.897 | 0.967 | 0.968 |
| arm_joints | 1.000 | 0.967 | 1.000 | 0.953 | 1.000 |
| gripper | 1.000 | 0.910 | 1.000 | 0.930 | 1.000 |
| eef_position | 0.931 | 0.988 | 0.995 | 0.970 | 0.991 |
| eef_orientation | 0.930 | 0.975 | 0.989 | 0.960 | 0.980 |

The bottom drawer does not move in any demonstration frame, so its R² is undefined. Incremental R² of [proprio ⊕ ψ] over proprio: object position +0.543 (R0), +0.532 (R1); object orientation +0.120, +0.092.

Objects beyond the arm:

| Target | proprio | R0 | proprio ⊕ R0 | R1 | proprio ⊕ R1 |
|---|---|---|---|---|---|
| t0 (fit 390, score 43 first frames): object_position | −0.025 | 0.868 | 0.868 | 0.641 | 0.625 |
| — bowl / cream cheese / wine bottle / plate | −0.08 / −0.00 / −0.03 / −0.00 | 0.897 / 0.825 / 0.931 / 0.857 | 0.896 / 0.827 / 0.930 / 0.853 | 0.760 / 0.527 / 0.562 / 0.842 | 0.759 / 0.491 / 0.536 / 0.862 |
| ungrasped, pooled over objects | 0.198 | 0.884 | 0.888 | 0.870 | 0.874 |
| — bowl / cream cheese / wine bottle / plate | 0.128 / 0.012 / 0.220 / 0.328 | 0.886 / 0.712 / 0.855 / 0.942 | 0.890 / 0.703 / 0.869 / 0.941 | 0.923 / 0.450 / 0.802 / 0.938 | 0.925 / 0.450 / 0.816 / 0.938 |

### 4. Effect distribution

Sufficiency (per-state effective rank mean; pooled within-state effective rank; within-state share):

| Set | R0 z | R1 Δ̂ | R1 z |
|---|---|---|---|
| consistency_counterfactual (14 per state) | 8.39; 142.0; 0.521 | 2.30; 14.4; 0.521 | 2.34; 16.9; 0.568 |
| consistency_proposal (32) | 13.24; 225.8; 0.143 | 1.85; 9.1; 0.017 | 2.44; 11.0; 0.022 |
| consistency_candidates (46) | 14.04; 174.8; 0.335 | 2.32; 14.5; 0.265 | 2.38; 16.8; 0.313 |
| phase9_chunks (4) | 2.50; 140.2; 0.253 | 1.44; 11.5; 0.098 | 1.47; 13.0; 0.119 |
| trial_demo_branch_eval (32) | 15.84; 86.6; 0.668 | 3.59; 11.1; 0.753 | 3.83; 10.9; 0.762 |
| trial_demo_branch (8) | 5.52; 80.9; 0.562 | 2.42; 11.5; 0.534 | 2.29; 12.3; 0.602 |
| trial_throughput_5pgpu (8) | 5.43; 125.4; 0.575 | 2.23; 14.1; 0.552 | 1.95; 13.9; 0.606 |
| trial_resim_candidates (32) | 12.71; 92.7; 0.166 | 1.87; 3.4; 0.036 | 2.94; 5.4; 0.018 |

Scene-change readouts. AUC; "—" where a class spans fewer fold groups than the 5 outer folds:

| Set (fold groups) | Label (positives) | R0 z | R1 Δ̂ | R1 z |
|---|---|---|---|---|
| demo_renders_heldout (43 episodes) | object moved (1,956) | 0.916 | 0.982 | 0.975 |
| | bowl (945) / cream cheese (318) / wine bottle (470) / plate (307, 4 episodes) | 0.929 / 0.809 / 0.910 / — | 0.988 / 0.939 / 0.996 / — | 0.966 / 0.921 / 0.982 / — |
| | fixture changed (361) | 0.946 | 0.983 | 0.987 |
| | grasp changed (579) | 0.801 | 0.905 | 0.890 |
| trial_throughput_5pgpu (80 states) | object moved (219) | 0.700 | 0.698 | 0.723 |
| | bowl (141) / wine bottle (49) / plate (16) | 0.647 / 0.772 / 0.611 | 0.707 / 0.837 / 0.468 | 0.663 / 0.853 / 0.349 |
| | fixture changed (38) | 0.734 | 0.730 | 0.403 |
| | grasp changed (87) | 0.551 | 0.605 | 0.653 |
| trial_demo_branch (32 states) | object moved (63) | 0.754 | 0.841 | 0.747 |
| | bowl (40) | 0.806 | 0.853 | 0.715 |
| | fixture changed (18) | 0.470 | 0.424 | 0.373 |
| | grasp changed (33) | 0.776 | 0.843 | 0.745 |

trial_demo_branch_eval (object motion in 4 of 12 states) and trial_resim_candidates (1 of 12) are too small for any classifier.

R² of the displacement vectors (object [4 × 3]; end effector [3]):

| Set | R0 z | R1 Δ̂ | R1 z |
|---|---|---|---|
| demo_renders_heldout | 0.494; 0.857 | 0.747; 0.871 | 0.440; 0.729 |
| trial_throughput_5pgpu | 0.052; 0.640 | 0.060; 0.301 | 0.031; 0.149 |
| trial_demo_branch | −0.052; 0.721 | −0.040; 0.341 | −0.040; 0.090 |
| trial_demo_branch_eval | 0.007; 0.675 | 0.087; 0.161 | −0.058; 0.119 |
| trial_resim_candidates | −0.713; 0.823 | −0.233; 0.384 | −0.249; 0.070 |

### 5. Frame domain

52,895 demonstration states, of which 249 have restore proprio error > 1e-4; data A 6,240 states. Median ‖z‖ of the 48,565 demo chunks: R0 14.06 (render) / 14.29 (video); R1 3.957 / 3.964.

| Model | distance / median ‖z‖: all [p10, p90] | error > 1e-4 | error ≤ 1e-4 | data A |
|---|---|---|---|---|
| R0 | 0.520 [0.416, 0.707] | 0.603 [0.481, 0.859] | 0.520 [0.416, 0.706] | 0.516 [0.415, 0.704] |
| R1 | 0.063 [0.039, 0.258] | 0.064 [0.042, 0.114] | 0.063 [0.039, 0.259] | 0.065 [0.039, 0.307] |

| Model | separability accuracy / AUC: all | error > 1e-4 | error ≤ 1e-4 | HELDOUT episodes | data A |
|---|---|---|---|---|---|
| R0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| R1 | 0.883 / 0.944 | 0.863 / 0.936 | 0.883 / 0.944 | 0.882 / 0.938 | 0.884 / 0.944 |

The inner CV chose λ = 1e-7, the grid's smallest, in every fold for both models. With λ fixed in the same outer folds (`tmp/s4p5/domain_lambda_sweep.py`, log `tmp/runlogs/s4p5/domain_lambda_sweep.log`), R1's AUC is 0.917 at λ 1e-5, 0.941 at 1e-6 and 0.944 at every λ from 1e-7 to 1e-12, so the reported value is the unregularized one.

### Reading

- **Collapse**: on the sets with at least 80 states, R1's ψ has effective rank 14.0–17.5 (R0: 45.5–80.0) and meets no pre-registered collapse or conflict criterion on the held-out demonstration and rollout sets. Its K3 ratio is a quarter of R0's.
- **Frame domain**: R1 moves by 6% of a median chunk effect between the video frame and the render of one state (R0: 52%), yet a linear readout still tells the two apart (AUC 0.944). The frames with restore error > 1e-4 and the data-A renders behave like the rest.
- **Object state**: ψ of R1 on renders encodes object positions nearly as well as R0 (R² 0.924 vs 0.936) and adds +0.53 R² over proprio. With the arm fixed (t = 0) it falls behind R0 (0.641 vs 0.868), most for the cream cheese and the wine bottle. Fixtures are encoded better by R1 than R0 (0.968 vs 0.933). Orientation is low for every input: the bowl and plate are rotation-symmetric, and the cream cheese's orientation varies little.
- **Effects**: within one state R1's effects span about 2 dimensions (per-state rank 2.3 of Δ̂ and z on the 14-candidate set, against 8.4 for R0's z), and the π candidates' effects nearly coincide (within share 0.02). On demonstration chunks Δ̂ and z of R1 separate object motion, fixture change and grasp change (AUC 0.89–1.00). On smooth random chunks from demonstration states, z of R1 hardly encodes the end-effector displacement (R² 0.09–0.15 against 0.64–0.72 for R0), and object displacement is not linearly decodable from any representation (R² ≤ 0.09).

## R6 through the evaluator (smoke checkpoint)

The R6 path is exercised on one set, demo_renders_heldout, with the smoke checkpoint of the fixed-encoder trainer. That checkpoint has 500 steps and uses the provisional PCA fitted before the random training collection existed (docs/interaction_training.md, "Phase 9 verification"). The numbers check the path; they are not the R6 result, which Phase 7 trains on the final stores.

- **Models**:
  - R0 = `frozen`;
  - R6_smoke = `frozen_pca outputs/runs/s4p9_frozen_pca_delta_smoke/checkpoints/step_000500.pt`. That checkpoint is bitwise equal to that of the final-code rerun `s4p9_final_frozen_pca_delta_smoke`.
- **Config**: `tmp/s4p9/configs/r6_eval_one_set.json`, which is `phase5_sets.json` restricted to demo_renders_heldout.
- **Code**: commit `0d0bd5e`; node-69 CUDA 1; collapse took 8 s, prediction 2 s.
- **Outputs**: `outputs/analysis/representation_eval/r6_smoke_phase9/{collapse,prediction}.json` and `report.md`; logs in `tmp/runlogs/s4p9/eval_r6_*.log`.

| Model | ψ frames rank / top | z rank / top | Δ̂ rank / top | SIGReg ψ normalized (raw) | SIGReg Δ̂ normalized (raw) | K3 (R0) |
|---|---|---|---|---|---|---|
| R0 (1,536-d) | 56.9 / 0.172 | 149.7 / 0.078 | — | 0.914 (0.581) | — | 0.372 (0.372) |
| R6_smoke (256-d) | 41.0 / 0.183 | 86.9 / 0.090 | 34.2 / 0.135 | 0.023 (0.023) | 0.021 (0.115) | 0.343 (0.372) |

| Model | Var(z)/Var(ψ) | dir. corr. weighted / min / median | adjacent cos raw / centered | cos mean / median | median rel. error | R² | K0–K3 |
|---|---|---|---|---|---|---|---|
| R6_smoke | 0.694 | 0.667 / −0.046 / 0.107 | 0.650 / 0.649 | 0.695 / 0.731 | 0.693 | 0.543 | F F F F |

The PCA subtracts the training mean, so the raw adjacent-frame cosine of R6 (0.650) is close to its centered value (0.649). R0's raw cosine (0.914) includes the feature mean.

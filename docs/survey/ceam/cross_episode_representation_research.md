# Cross-episode action-effect representation research

Evidence reviewed on 2026-10-03 EDT. This study examines transferable action-conditioned effects, supervision
and the coverage of existing CEAM diagnostics. Literature mechanisms are verified against primary method
sections; paper results have not been independently reproduced here.

## Round 1: transferable structure without complete reconstruction

| Primary source and inspected version | Actual mechanism / evidence | Scope and limitation |
|---|---|---|
| [DINO-WM, v1, §§3–4](https://arxiv.org/html/2411.04983v1) | Frozen spatial DINOv2 patch features; action/history-conditioned prediction; goal-image planning. Pixel decoder is optional and trained separately. | Supports latent prediction without reconstruction as the control objective. Uses recorded actions; goal-image control does not establish instruction-conditioned equivalence classes. |
| [V-JEPA 2, v1, §§3–4](https://arxiv.org/html/2506.09985v1) | Video pretraining followed by an action-conditioned predictor on a frozen encoder; deployment in two new robot labs. | Robot post-training uses EEF states and actions derived from consecutive EEF-state changes. “Unlabeled” excludes reward/task/success annotation, not physical measurements. Pick-and-place uses supplied subgoal images; camera sensitivity and long horizons remain limitations. |
| [C-SWM, v2, §§2–3](https://arxiv.org/pdf/1911.12247v2) | Shared object encoders and graph dynamics learn contrastive state/action/next-state structure without pixel reconstruction. | Demonstrates structured environments and combinatorial generalization. Its object-factorized action interface and simple scenes are stronger assumptions than continuous-arm manipulation from arbitrary camera views. |
| [SE(2)-equivariant pushing, CoRL 2022, §2](https://proceedings.mlr.press/v205/kim23b/kim23b.pdf) | Transforms object poses and pushing commands into relative coordinates; shared dynamics respect planar rigid transformations. | Provides a concrete mechanism for reusing interactions across different positions/orientations. Uses object shape/pose estimation and physical supervision. Tabletop SE(2) symmetry cannot be imposed on arbitrary arm configurations, gravity directions or obstacles. |

These mechanisms provide existence evidence for transferring structure across observations without requiring
whole-scene visual-neighbor retrieval. They do not establish that every pair of visually different observations
shares an interchangeable action effect. Relevant relations, embodiment and contact conditions still matter.

## Round 2: what makes a representation informative about control

| Primary source and inspected version | Actual mechanism / evidence | Scope and limitation |
|---|---|---|
| [LAPA, v1, §3](https://arxiv.org/html/2410.11758v1) | An inverse model extracts quantized latent actions from current/future frames; a language-conditioned policy learns these labels; action-labeled fine-tuning maps to robot control. | The inverse latent can be computed after observing the outcome. It is not a forward evaluation of a proposed, unexecuted action. Useful pretraining evidence does not resolve prospective candidate evaluation. |
| [Latent Action Learning Requires Supervision in the Presence of Distractors, v1, §§3–5](https://arxiv.org/html/2502.00379v1) | Controlled distractor experiments reveal that latent actions can encode background/camera changes. Reusing scarce action labels during latent learning improves results. | Study covers four modified control environments with privileged tuning. Cross-embodiment results remain comparable to BC from the same action labels. This is counterevidence to automatic grounding, not an impossibility theorem for all self-supervision. |
| [LeJEPA / SIGReg, v1, §§4–5](https://arxiv.org/html/2511.08544v1) | Regularizes projected embedding distributions toward an isotropic Gaussian, combined with a predictive agreement objective. | Distributional noncollapse does not specify which physical distinctions must survive. A regularizer on the population is different from a test of action sensitivity at a fixed observation. |
| [LeWorldModel, v1, §§3–6](https://arxiv.org/html/2603.19312v1) | Jointly learns a visual encoder and action-conditioned latent predictor with prediction loss plus SIGReg; performs goal-image planning without reconstruction or reward labels. | A close concrete baseline for the proposed learning direction. Requires action-labeled offline coverage. Reports short-horizon and low-diversity limitations; these experiments do not establish a universal cross-scene representation or CEAM's dual-query advantage. |
| [Physically Grounded JEPA, v1, §§II–III](https://arxiv.org/html/2609.03565v1) | Adds inverse-action prediction and training-only physical-state supervision. On four tasks, state alignment improves over its IDM-only ablation. | September 2026 preprint; baseline means are imported from LeWorldModel, while new variants use three seeds. Lower temporal straightening can accompany better planning. It motivates a controlled auxiliary-supervision ablation, not a requirement to reconstruct all state. |

The distinction between physical-state supervision and full scene reconstruction is substantive: a compact
contact, relative-motion or proprioceptive target can ground an effect without reconstructing appearance.
Whether these targets improve CEAM rather than distract it remains unmeasured.

An analytical counterexample to noncollapse as sufficient action grounding is a representation of a rich,
episode-specific background that remains constant during an action. It can vary across episodes while ignoring
all candidate commands at one start. A low temporal prediction loss and a diverse global representation alone
therefore do not certify useful candidate distinctions. This is an objective-level example, not an observed
failure diagnosis of the current neural model.

## Round 3: implementation and data coverage

Verified source observations:

- `ceam/model/frozen_pca_encoder.py` averages cached patch tokens per view before PCA. DINO-WM's spatial-token
  predictor is not equivalent to this input bottleneck. This source inspection does not measure retained contact information.
- `ceam/world/physical_world_training.py` pools all ten Goal tasks. Its ridge predicts an additive function
  `W_s s + W_a a + b`. At one start, the predicted candidate difference is `W_a(a_i-a_j)`, independent of state.
  That baseline can detect generic command effects without learning context-dependent interactions.
- Existing physical shards have a 174-dimensional privileged-state schema, ten-step seven-dimensional actions,
  four named objects and three named fixtures. A task-held-out probe within this schema does not certify transfer
  between unlike visual scenes, embodiments or object inventories.
- Candidate effects are distinct from success and trajectory quality. The continuation diagnostic's label flips
  do not refute transferable local physical effects: see `docs/continuation_sensitivity_results.md`.

### Fixed CPU diagnostic

Protocol: `plans/cross_episode_representation_research.md`. Raw artifacts:
`outputs/physics_world/cross_task_probe_v1/`. The complete 360-fit grid and ten persistence controls are in
`results.md` / `results.json`; `manifest.json` records input checksums, source row IDs and the full grid before fits.

Data comprise 5,135 demonstration windows from 240 TRAIN episodes and 1,545 candidate transitions in 129 banks
from 43 disjoint evaluation episodes. Development and fresh splits were reused from prior research; neither is
new confirmation. All commands are ten-step chunks. Each target task uses exactly 256 TRAIN windows per fit,
three sampling seeds and one fixed ridge penalty .01. The same target banks serve every model/source condition.
Other-task data are balanced over nine tasks; mixed uses 128 own-task and 128 other-task windows. Source diversity
therefore changes together with coverage; this is not an estimate of adding data while retaining all own-task data.

Models: state-only, 70-dimensional action-only, additive state/action, and additive plus 112 explicit interaction
features. The latter crosses 12 object-relative positions and four bilateral-grasp indicators with seven mean
action components. The full action time sequence remains in the additive features. Feature normalizers use only
the selected TRAIN windows. No language, task identity, success or continuation label enters the predictors.

Outputs are EEF xyz and four object xyz changes. They exclude drawer/stove joint motion, rotation, collision and
future feasibility. Object errors average all four objects, including stationary ones.

Within each bank, subtract the candidate mean from both true and predicted effects. Effect R² is
`1 - sum(centered_prediction_error²) / sum(centered_true_effect²)`. Zero is the prediction that cannot distinguish
candidates; negative means worse than that reference. Absolute errors below are mean Euclidean errors, averaged
equally over target tasks and sampling seeds. R² instead pools SSE/SST, so larger-effect banks have more weight.
Seed ranges describe sampling variation, not confidence intervals or independent policy seeds.

| Source | Model | EEF mm | Object mm | Effect R² | EEF R² | Object R² | Seed R² range |
|---|---|---|---|---|---|---|---|
| none | persistence | 70.5784 | 7.5691 | 0.0000 | 0.0000 | 0.0000 | 0.0000 to 0.0000 |
| own | state | 36.8614 | 6.2607 | -0.0000 | 0.0000 | 0.0000 | -0.0000 to 0.0000 |
| own | action | 19.5121 | 6.5073 | -2.0070 | -0.2940 | -5.6200 | -2.3342 to -1.7903 |
| own | additive | 16.4425 | 6.1122 | -0.0554 | 0.4204 | -1.0589 | -0.0817 to -0.0375 |
| own | interaction | 14.4598 | 4.1564 | 0.4817 | 0.6199 | 0.1901 | 0.4378 to 0.5132 |
| other | state | 67.5797 | 13.1634 | 0.0000 | 0.0000 | -0.0000 | -0.0000 to 0.0000 |
| other | action | 35.3557 | 25.6135 | -7.5675 | -1.7575 | -19.8222 | -8.5057 to -6.9398 |
| other | additive | 35.8691 | 15.8492 | -0.7458 | -0.0544 | -2.2040 | -0.8010 to -0.6645 |
| other | interaction | 31.4095 | 12.9662 | 0.0012 | 0.2209 | -0.4622 | -0.0221 to 0.0246 |
| mixed | state | 36.8587 | 7.4889 | 0.0000 | 0.0000 | -0.0000 | 0.0000 to 0.0000 |
| mixed | action | 25.8003 | 17.4492 | -6.4159 | -1.5100 | -16.7635 | -7.4885 to -5.6738 |
| mixed | additive | 19.5864 | 9.2394 | -0.5984 | 0.1387 | -2.1532 | -0.7632 to -0.5100 |
| mixed | interaction | 16.8481 | 5.5930 | 0.3866 | 0.4836 | 0.1820 | 0.2861 to 0.4597 |

The explicit interaction model improves both EEF and object mean errors over the additive model in all three
source regimes. Its overall effect R² changes from −.0554 to .4817 for own-task data, −.7458 to .0012 for other-task
data, and −.5984 to .3866 for mixed data. More features and a different inductive bias both change; this comparison
does not isolate either as the sole cause or establish an advantage for dual representations.

Only-other-task interaction training retains some EEF discrimination, R² .2209, but object discrimination is
negative, −.4622. The almost-zero aggregate is therefore not evidence of uniformly successful physical transfer.
Mixed data improve over other-only data but do not beat own-only data at this fixed sample budget. No conclusion
about large-scale pretraining follows from this small flat ridge model.

| Task | Own EEF / object mm | Other EEF / object mm | Mixed EEF / object mm |
|---|---|---|---|
| 0 | 15.04 / 0.01 | 25.84 / 8.28 | 18.72 / 1.92 |
| 1 | 26.58 / 5.73 | 41.38 / 9.04 | 25.54 / 5.95 |
| 2 | 17.52 / 4.71 | 38.53 / 18.31 | 20.10 / 5.71 |
| 3 | 10.03 / 4.89 | 20.99 / 6.99 | 14.28 / 5.46 |
| 4 | 9.43 / 4.52 | 26.94 / 10.60 | 14.10 / 5.30 |
| 5 | 13.32 / 5.29 | 26.50 / 11.67 | 15.76 / 5.87 |
| 6 | 17.54 / 11.17 | 29.55 / 20.64 | 20.87 / 15.10 |
| 7 | 6.71 / 2.58 | 37.99 / 14.75 | 10.74 / 3.67 |
| 8 | 16.46 / 2.67 | 25.67 / 14.16 | 14.63 / 4.78 |
| 9 | 11.96 / 0.00 | 40.70 / 15.22 | 13.74 / 2.17 |

### Coverage observations

A post-fit descriptive audit counts windows with object displacement above 5 mm. This threshold defines an
inspection statistic only; it was not used to select TRAIN examples, fit models or filter evaluation candidates.
Object order is bowl, cream cheese, plate, wine bottle.

- All 390 TRAIN windows with plate motion above 5 mm occur in task 2. Holding out task 2 removes this motion
  coverage entirely. The only recorded initial bilateral plate contacts also occur there.
- All 137 initial bilateral cream-cheese contacts occur in task 6. Other tasks contain some cream-cheese motion,
  so absence of the grasp proxy does not mean absence of every interaction with this object.
- `PhysicalStateReader.read()` defines its contact fields as simultaneous contact with both finger pads. A zero
  flag does not distinguish free motion from one-sided pushing contact.
- The predictor assigns separate coefficients to named object output slots. It has no explicit parameter sharing
  between bowl, plate and bottle dynamics. This diagnostic consequently does not test learning a common object
  interaction rule and reusing it for a held-out object.
- Object effect SST is exactly zero for task 9; its object-only R² is undefined and recorded as null. Task 0 has
  very small object variation, making per-task object R² unstable. The complete table retains these cells.

### Validation and resource record

Run:

```bash
bash $CEAM_ROOT/scripts/cluster_run.sh node-61 $CEAM_ROOT/tmp/run_cross_task_effect_probe.sh
```

Analysis uses the same gateway with `tmp/analyze_cross_task_probe.sh`. The analysis source is
`scripts/analyze_cross_task_effect_probe.py`; its SHA256 is recorded in `validation.json`.

All 360 fits completed at 02:02:52 EDT on 2026-10-03. One process, two BLAS threads, zero GPUs. Fit-and-report wall
time was 5.956 s; CPU time 10.813 s; peak RSS 75.87 MiB; initial outputs 6.56 MB. Independent post-fit analysis used
0.640 s wall time and 0.555 s CPU time. These tiny ridge fits are not neural training runs or a throughput benchmark.

Input checksums remain unchanged. Independent pairwise-difference recomputation of bank-centered statistics
agrees within 2.22e-16 for SSE and 6.94e-17 for SST. The state-only control has maximum absolute effect R² 1.22e-15.
The initial audit stopped before fitting on an exact-start check: bank start spread was 8.6652e-15 in mixed state
units. That failed audit is retained in `cross_task_probe_v1_initial_audit/audit.json`; the completed run records
the spread observationally. It has no measurable action-discrimination contribution in the state-only control.

The experiment process exited. No GPU allocation, simulator rollout, policy update or action-label publication
was performed in this study. No new policy SR was measured.

## Round 4: deductions and evidence boundaries

**Local matching does not identify a globally shared coordinate system.** For dot-product matching, apply an
observation-dependent orthogonal matrix U(o) to both representations: r′ = U(o)r and q′ = U(o)q. Their local score
is unchanged because UᵀU = I. Cosine and Euclidean distance are likewise unchanged under common orthogonal
transforms. This algebraic ambiguity means that good same-observation matching alone cannot prove cross-context
representation alignment. It does not prevent transfer through shared parameters or additional objectives.

**A forward effect model and a task utility model have different evidence requirements.** Physical transition
targets can come from successes, failures and unrelated task episodes; prospective utility also depends on task,
continuation and horizon. The observed continuation-label reversals concern utility supervision. They do not
imply that task language must enter the physical-effect representation.

**No complete reconstruction requirement follows from the evidence.** Published latent predictors provide
concrete alternatives. Conversely, neither low prediction loss, global noncollapse nor a well-shaped match score
certifies feasibility for unsupported actions. The present probe does not resolve camera observability or
long-horizon uncertainty.

| Question | Evidence level after this study |
|---|---|
| Can latent action prediction work without reconstructing all pixels/3D? | Supported by external primary experiments; not newly reproduced here. |
| Does state-dependent action structure matter in the small physical diagnostic? | Explicit interaction features improve measured errors under all three source regimes. |
| Does arbitrary other-task demonstration data suffice here? | No reliable object-effect transfer in the fixed flat ridge grid; relevant contact/motion coverage is uneven. |
| Does this refute transferable structure across unlike observations? | No: neither cross-scene visual transfer nor shared object dynamics was tested. |
| Does CEAM dual matching outperform a direct scorer, BC or RL? | Unresolved; this study contains no matched policy comparison or new SR. |

Future mechanisms and a falsification protocol are recorded in OPT-014. Coverage and interpretation limitations
are tracked in ISSUE-046.

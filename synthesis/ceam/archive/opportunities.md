# OPT-001

Status: open

Area: 查询 / 指代

## Observation

LIBERO-Goal 上，以当前帧为参考的状态依赖查询（TEXT_AND_OBSERVATION）与仅文本查询的动作排序相近（`docs/stage_one_instruction_contrast.md`），训练慢一倍以上，阶段一定版为仅文本查询。

## Opportunity

在需要指代的数据上（同一场景有多个同类物体、指令按空间关系指代），启用状态依赖查询 q(o, l)。

## Expected Mechanism

指代对象由当前观测决定，仅文本查询无法区分"哪一个"。

## Required Evidence

同场景"指代错误"终态作为负样本的数据（ISSUE-003），以及在其上仅文本与状态依赖查询的动作排序对比。

## Promotion Condition

有了检验指代的数据之后。

# OPT-002

Status: open

Area: 诊断 / 物体状态探针

## Observation

LIBERO-Goal 原始 hdf5（含仿真状态）已下载到 `outputs/raw/LIBERO-datasets/libero_goal/`；阶段一 ψ(o_T) 的任务 ID 探针在 HELDOUT 上为 1.000，未触发物体状态探针的使用条件。

## Opportunity

用仿真状态中的物体位姿作标签，从 ψ 线性探测物体状态，定位表征缺失的信息。

## Expected Mechanism

按动作序列把去零动作的 episode 与原始 demo 对齐，取对应帧的物体位姿。

## Required Evidence

对齐成功率与探针精度。

## Promotion Condition

ψ 层面的任务信息或动作排序出现无法由现有指标解释的缺失时。

# OPT-003

Status: open

Area: 指令对比项 / 温度

## Observation

logit_scale 与其余参数共用 AdamW 学习率，5000 步内 s 从 14.3 变为 18–31（阶段一定版保持现状）。

## Opportunity

为 logit_scale 设独立参数组与更大的学习率，让温度由数据决定。

## Expected Mechanism

Adam 每步参数位移约为学习率，独立学习率决定温度可移动的范围。

## Required Evidence

不同 logit_scale 学习率下温度的收敛值与动作排序、闭环成功率的变化。

## Promotion Condition

闭环成功率对温度敏感时。

# OPT-004

Status: open

Area: SIGReg 权重 / 表征有效秩

## Observation

SIGReg 权重 10 使 ψ 有效秩 16.6 → 31.5、动作效果有效秩 15.1 → 20.7，同时共享起点区分度 0.87 → 0.66、候选 match 离散度 0.19 → 0.08；阶段一定版为权重 1。

## Opportunity

若 E(o, a) 需要更丰富的表征（例如区分细粒度动作），重新权衡 SIGReg 权重或其作用范围。

## Expected Mechanism

更高的各向同性约束把方差分散到与指令无关的方向，提高秩但稀释 match 的对比度。

## Required Evidence

阶段二中 E 的预测误差与反事实排序随 SIGReg 权重的变化。

## Promotion Condition

阶段二离线评估或闭环中发现表征容量不足时。

# OPT-005

Status: open

Area: 阶段二 / 离线评估与正式训练

## Observation

阶段二原计划中的离线评估工具（用 Δ 的预测替换真实后果的动作排序、倒放与偏移扰动候选、共享起点反事实）与 10000 步正式训练，推迟到闭环 MVP 的结果之后再定。

## Opportunity

以离线指标诊断闭环失败的原因（Δ 的预测误差、候选排序），并以更长训练提升 Δ。

## Expected Mechanism

离线排序与闭环成功率的差距可定位到预测误差或候选质量。

## Required Evidence

闭环 MVP 的成功率（真实效果与预测效果两种来源）。

## Promotion Condition

闭环结果需要解释或 Δ 的短时训练不足时。

# OPT-009

Status: open

Area: 任务无关动作表征学习

## Observation

- 当前 R6_all 使用冻结 DINOv2、固定 PCA-256 和效果头。它在广域随机候选上具有可分性，但在 π 产生的相近候选上表现较弱：留出状态内方差占比为 0.042，平均候选余弦为 0.951，真实效果状态内 R² 为 0.069。
- 四轮小规模交互续训主要记忆当轮转移，没有改善未见数据。现有具体类还把动作表征绑定到 ψ、状态差分和 `DynamicsHead`，这些不是框架要求。
- 纯行为克隆只约束动作采样器复现示范动作，不能独立保证 `r(o, a)` 在同一观测下形成对下游选择有用的结构。

## Opportunity

用示教数据的超集学习任务无关的动作表征。数据覆盖示教、同状态动作扰动与分支、策略 rollout、失败与恢复轨迹以及结构化随机动作。学习方法保持开放，可比较预测式联合嵌入、对比学习、掩码学习和 SIGReg 等非坍缩约束；JEPA 是候选方法族之一。

先定义不依赖 ψ、状态差分或特定预测头的动作表征接口，再以相同数据、查询协议和候选集合比较学习目标。编码器冻结、部分适配或联合训练是实验变量。

## Expected Mechanism

- 多来源动作与结果提供比示教动作更宽的支持集，减少对单一路径的记忆。
- 同观测候选比较直接约束下游需要区分的效果；效果等价候选允许重合。
- SIGReg 或其他方差与协方差约束防止表征坍缩；预测或对比目标保留动作相关结构。
- 固定查询与 match 的评估协议把动作表征改进和查询改进分开。

## Required Evidence

- F1 证明当前 q/m 能否利用 R6_all 已有信号，并报告候选 oracle 与 π-only 排序。
- LIBERO-90 R6 基线给出跨 90 个任务的可分性、候选排序和闭环结果。
- 新表征在未见状态上提高同观测可分性和固定 q/m 下的候选排序；报告来源与动作尺度分层结果。
- 三种子闭环相对 R6 与无 ψ π 的配对结果，同时报告单步延迟、显存和参数量。

## Promotion Condition

F1 与 LIBERO-90 R6 基线完成后，依据两者证据批准独立的动作表征学习计划，再确定方法网格、数据配比和编码器训练范围。

# OPT-010

Status: open

Area: Offline demonstration refinement

## Observation

The thirty-source pilot's guided shortlist exceeds random by +1.20 / +3.00 / +5.20 pp across π seeds. Guided
versus original is +1.33 pp pooled with a benchmark-paired interval spanning zero and heterogeneous seed effects.
Each source has one edited ten-step block; one source selection and one refinement seed were used.

Current selection minimizes first-success steps, then standardized six-dimensional arm-command total variation.
Variation only affects equal-step comparisons; it does not measure physical end-effector acceleration or jerk.
Random refinement produces lower aggregate source costs than guided refinement but lower pooled policy SR.

The G2 Orion resampling reference builds observation-anchored future joint-target chunks using second-difference
density outside grasp/release windows and time-uniform sampling inside those windows. Its playback separately
assigns time from waypoint distance and a duration floor, interpolates targets and blends replans. The exported
chunks do not include each waypoint's original time. Event-window sampling alone therefore does not establish
preservation of measured physical dwell durations during playback. The supplied report measures reconstruction
and playback behavior rather than a controlled policy SR improvement. These are reference-code observations,
not CEAM measurements. Reference: `g2-orion05-resample-training.zip`, `progress_resample.py`,
`chunk_playback.py`, `build_new_dataset.py` and `docs/RESAMPLING_REPORT.md`.

## Opportunity

Confirm the offline use of r/q/m with repeated source/refinement seeds and more policy seeds, then compare broader
edit coverage or a quality objective tied to policy learnability. Preserve same-bank random and original controls.

Develop demonstration-quality objectives for faster successful completion, fewer unnecessary task operations and
less physical jitter. Candidate coverage should include pause removal, local route shortcuts and redundant
subsequence edits as well as smoothing. Required grasp/contact/release events and physical waiting intervals are
feasibility constraints; a slower successful demonstration is not automatically an inferior training example.
Faster completion is a trajectory-level preference. Necessary alignment, grasp settling, waits and preparation
may be locally slow; per-step velocity maximization is not the quality objective.

Use the reference's separation of path sampling, execution timing and event protection as design input.
CEAM's seven-dimensional end-effector increment commands differ from the reference's absolute joint targets.
Retiming must reconstruct a physical pose trajectory and produce commands through the actual controller
conventions; nonuniformly sampled increment commands cannot simply replace fixed-time action labels.
Measured edited trajectories require simulator replay and regenerated observations. BC targets may instead pair
original observations with proposed actions, under the supervision boundaries defined in OPT-015.

## Expected Mechanism

Compatibility ranking may reduce harmful local edits while executed trajectory quality chooses among compatible
candidates. Its contribution must be distinguished from ordinary smoothing and policy training variation.

Geometric editing may reduce local pauses, path length and motion irregularity. Task-conditioned comparison is
needed to distinguish necessary operations from redundant ones and to compare alternative successful strategies.
Current match scores express compatibility, not calibrated progress, remaining duration or strategy optimality.
A task-conditioned preference objective would require separate same-start branch evidence and validation.
The representation can retain its existing separability requirement; a monotone distance to a fixed goal is not
a prerequisite for demonstration refinement.

A positive aggregate match on successful trajectories is a candidate auxiliary training constraint. It permits
local negative matches and imposes no negative-aggregate requirement on failed trajectories. Its contribution to
same-start ranking and trajectory quality requires measurement. Dot-product matching additionally requires a
defined treatment of embedding norms; changing the similarity function does not itself supply quality evidence.

## Required Evidence

Stable guided-versus-random and guided-versus-original differences, explicit task coverage, task-quality criteria,
benchmark pairing and complete compute/data budgets.

Measure source success, first-success duration, physical position/orientation path length, acceleration/jerk and
event preservation. Separate demonstration jitter from policy jitter at chunk boundaries. Compare original,
geometry-only and semantic-guided refinement with a shared candidate bank, editing budget and physics budget;
include random selection to identify the ranking contribution. Train policies under the same protocol and report
paired SR, successful completion time and physical jitter. Report duration together with failures so that faster
completion among surviving easy episodes cannot conceal lost task coverage. Geometric fidelity or lower source
cost alone does not establish better policy learning.

## Promotion Condition

An independent confirmation experiment defines its sources, quality objective and resource grid before execution.
Broader event detection, controller conversion and semantic preference targets require explicit interfaces in
that experiment's plan. The completed retrieval mechanism and measured physical branch metrics are recorded in
`docs/retrieval_refinement_results.md`; confirmation, retiming and broader edits remain future work.

# OPT-012

Status: open

Area: Local match-based action search / learned physical consequence support

## Observation

The privileged-state pilot predicts nontrivial same-start action effects: fresh effect R² .658 for mixed
latent-128, versus zero for persistence. Physical error remains centimetre-scale, contact changes are missed
and fixed confidence bounds retain no candidate. Adding branches improves linear mean xyz forecasting by
1.33 mm, while both neural data-ablation intervals span zero. Results are in `docs/physical_world_results.md`.

## Opportunity

Keep local candidate search with task-independent r and task-conditioned q/m. Improve the physical predictor
as auxiliary evidence for candidate consequences and for deciding which candidates need actual execution.
A task-independent physical forecast does not own task success or successful-action preference.

First compare the current predictor with an object-relative pose/velocity encoding and explicit grasp-change
outputs on shorter and ten-step horizons. Use actual TRAIN branches, failures and recovery/contact transitions;
keep exact same-start candidate controls and measure event prevalence/recall rather than overall contact accuracy.
Define module boundaries, horizon grid, source budget and confidence protocol before collecting additional data.

After physical distinctions and calibrated coverage improve, compare match-guided candidate search with and
without physical prediction under the same simulator-query budget. Predicted effects can support r; q/m still
judge the instruction in the current observation. Actual executions update transition evidence and can support
compatibility/preference learning under OPT-013. Own predicted high match or low forecast error is not an
improved-action label. No constant-positive cumulative-match constraint certifies task success.

A vision-to-3D encoder is a subsequent measurement: learn observable state from camera data against training
physical targets, quantify occlusion/state error and propagate its uncertainty into consequence prediction.
Privileged-input pilot accuracy does not establish camera observability. Voxel similarity alone does not encode
velocity, controller state or grasp phase. History is an optional candidate if missing state is measured.

## Expected Mechanism

Same-start executed alternatives train task-independent action distinctions without requiring monotone local
match or progress. Physical forecasts can reduce execution queries only where held-out consequence errors and
coverage permit. Real execution remains necessary for uncertain or contact-sensitive decisions. A later
interaction-based policy improvement loop can use this mechanism without changing match into a speed reward.

## Required Evidence

Disjoint source episodes; current persistence/ridge/latent controls; candidate-centered translation/rotation and
event predictions; rare-contact failures; conditional error/coverage and actual simulator-query savings;
physical consistency and path validation. Confidence and task success remain separate reports.

Action labels require verified speed, unnecessary-operation and jitter tradeoffs, regenerated observations and
independent confirmation. Learned-policy claims require paired three-seed SR, completion time and jitter.
The current world-model ensembles are not policy seeds or a measured SR improvement.

## Promotion Condition

A subsequent approved plan fixes the minimal predictor change, source/horizon grid, uncertainty ownership,
simulator budget, independent confirmation and stopping condition. Vision reconstruction, RL and BC are separate
phases activated by evidence; they are not automatically launched from this opportunity.

# OPT-013

Status: open

Area: Representation controls after continuation-sensitive preference diagnostics

## Observation

The completed pilot improves held-out preference ranking, while direct success/feasibility selection remains
unreliable. The completed continuation diagnostic measures 46/239 success-label changes and 191/448 strict preference
reversals; the exact-state sensitivity subset retains 176/380 reversals. Evidence and validity boundaries are in
docs/continuation_sensitivity_results.md. The literature audit is docs/long_horizon_matching_review.md.

## Opportunity

Define learning labels under an explicit fixed continuation and horizon, retaining separate success, prospective
feasibility and trajectory preference targets. Then compare frozen-r query learning, adapted-r matching and an approximately
capacity-matched direct scorer on identical validated inputs and data. Add an action-only/geometric control.
Instruction permutation measures dependence; semantic correctness needs multiple verified tasks in the same scene.
Observe velocity/history ablations before making history a required input. Preserve partial preferences and
jitter tradeoffs; complete-trajectory speed does not require uniform local speed.

Successor, contrastive and latent-predictive objectives remain possible mechanisms. A fixed behavior continuation
and a task-conditioned policy family require different interpretations. Native FB/TD-JEPA policy conditioning
changes the interface of a strict task-free r when task-specific policy codes enter it.

## Expected Mechanism

Controlled comparisons can distinguish extra input/capacity from the contribution of dual representations.
Explicit continuation labels may improve generalization; that effect has not been measured. Broader strategy
exploration and self-supervised pretraining require their own objectives and evidence.

## Required Evidence

The continuation diagnostic first establishes label coverage and sensitivity. Subsequent model comparisons need
matched information, capacity and budgets, independent sources and calibrated feasibility checks. High model
scores alone do not label better actions. Published edits require independent replay and regenerated observations.
Policy-performance comparisons require paired three-seed SR, completion time and jitter. An RL advantage claim
requires a shared-budget baseline.

## Promotion Condition

A separate concrete plan specifies the learning target, source split, model interfaces and compute budget after
the diagnostic results are interpreted. No representation training, policy update or dataset publication is active.

# OPT-014

Status: open

Area: Transferable task-independent action effects across episodes and contexts

## Observation

Local retrieval and continuation-quality diagnostics do not measure learning a common interaction rule across
visually different observations. Primary latent-world-model methods demonstrate alternatives to full reconstruction.
The CPU transfer grid exposes uneven object/contact coverage and the limitations of additive, object-slot-specific
prediction. The source-linked audit and all measurements are in `docs/cross_episode_representation_research.md`;
coverage limitations are ISSUE-046.

## Opportunity

Learn r(o, a, robot_state) from cross-episode interactions while retaining q(o, instruction) for task-conditioned
selection. Shared physical structure can be learned without requiring visually similar episodes to be retrieved
or different observations to have identical embeddings. Complete 3D reconstruction is an optional reference,
not a required objective. The architecture and degree of sharing remain open.

The leading minimal study compares action-conditioned latent prediction with and without explicit preservation
of candidate-dependent effects. A generic form is a predictor from r to a training-only future/effect embedding,
with noncollapse regularization and an optional measured relative-motion/contact target. The future observation
supplies supervision only; it is not an input for scoring an unexecuted candidate. A difference of state embeddings
is one implementation, not a requirement. Unconditional pulling together of unrelated episodes is not a defined
positive-pair rule.

LeWorldModel's prediction-plus-SIGReg is a useful baseline. An auxiliary inverse-action head is an ablation, not a
default requirement: forcing exact recovery of every command may unnecessarily separate actions with equivalent
effects. Physical supervision can instead target selected observable relations without reconstructing a scene.

Before visual training, a small comparison of shared versus separate object-dynamics parameters can isolate
cross-object sharing using the existing privileged-state diagnostic. It must use the same feature interface,
TRAIN windows and evaluation banks. A positive privileged result establishes only a mechanism proxy; a camera-based
representation experiment is needed to establish observable transfer.

## Expected Mechanism

Shared interaction parameters may reuse action effects across positions, objects and task contexts. Action-conditioned
predictive evidence and same-start alternatives may prevent a diverse representation from relying only on static
appearance. Query/matching learning supplies task utility; its supervision retains continuation and horizon semantics.
Task-independent inputs do not imply absence of object or affordance information in r.

An exploration loop can prioritize uncertain or underrepresented interaction modes and retain executed outcomes,
including failures. Predicted high match is not its own confirmation label. Exploration and model learning require
an explicit data-acquisition budget; uncertainty calibration remains unverified.

## Required Evidence

1. Define two transfer conditions separately: a previously observed interaction in a new visual/spatial context,
   and an interaction mode absent from training. Audit action/relative-state/contact support before interpreting
   either condition. Near visual neighbors may be a baseline, not the only source of training examples.
2. Use held-out episodes and contexts with same-start executed alternatives. Report EEF, active-object, fixture
   articulation and contact effects separately as applicable. Distinguish endpoint accuracy from differences among
   candidates and preserve full failure counts. Camera/proprio inputs and training-only physical labels are explicit.
3. Hold source counts, information and optimization budget fixed. Compare state-only, action-only, latent predictive,
   shared-structure and capacity controls; permuting actions tests dependence. An ignored-action predictor must not
   receive credit for predicting static scene information. Preserve all grid cells.
4. Freeze r before comparing q/m with a direct scorer on matched data and inputs. Assess task compatibility and
   trajectory preferences separately; faster completion does not require higher speed on every step. Policy claims
   require paired three-seed SR, completion time, path and jitter, with an RL baseline for any claimed RL advantage.

## Promotion Condition

A new plan defines one minimal sharing/learning mechanism, source coverage, held-out contexts, prediction targets,
resource ceiling and stopping criterion. The completed review does not launch neural pretraining, BC/RL or dataset
relabeling. OPT-013 remains a separate task-utility diagnostic; it does not narrow representation pretraining to
same-task donors or a fixed continuation.

# OPT-015

Status: open

Area: Teacher- and dataset-conditioned action improvement and task specialization

## Observation

[ImageWAM, arXiv:2606.19531v1, §§3.2–3.4](https://arxiv.org/html/2606.19531v1) uses image-editing intermediate
KV features to condition an action expert. It trains the image diffusion branch and action expert using future
frames and expert actions, with the VLM conditioning components frozen. Its inference uses a single editing-branch
step without decoding an image. This establishes a particular use of pretrained visual priors; it does not establish
that local calibration or capacity reduction alone improves demonstrations. A generic pretrained image editor and
an embodied action teacher have different interfaces and training evidence.

Five primary papers and public implementations are audited in docs/teacher_dataset_calibration_review.md.
Frozen-teacher editing and offline noise-space adaptation have external precedents; improved duration, path and
jitter of successful source demonstrations remain unverified. The frozen pi0.5 interface has passed numerical
qualification in `docs/teacher_editing_results.md`; the completed candidate-quality pilot is recorded in `docs/teacher_editing_results.md` and
`plans/frozen_teacher_editing_pilot.md`.
Codebook learning, lossy compression and task adaptation remain alternatives.

## Opportunity

Learn an offline action generator with the conceptual interface
`G(teacher_weights, dataset, episode_and_time_interval) -> replacement_action_sequence`.
The dataset contains observations, instructions, recorded actions and their temporal/episode associations;
available robot state is retained. The interval identifies the original action sequence to optimize. The complete
dataset provides context beyond a single observation, including other episodes and the recorded future of the
target episode. Original actions are editable source material, not assumed optimal labels.
The objective is measurable improvement over source demonstrations while retaining task completion and physical
validity. Global optimality and coverage of every possible improvement are not requirements.

The generator produces candidate replacement sequences for demonstration improvement. It is not constrained
to an online policy's current-observation interface or to reproducing the teacher's original action distribution.
An alternative output is a task-specialized model obtained by distillation or fine-tuning, without materializing
an edited dataset. The completed pilot implements action editing; task-model adaptation remains a future interface.
This opportunity is independent of
the dual-representation framework.

The target domain may have a narrower range of relevant functions than the teacher's original training domain.
This does not imply a linear input/output subspace, low intrinsic rank or a small required network. Compression
may extract relevant teacher components, compile a dataset-specific generator or summarize dataset context.
These mechanisms have distinct latency, memory and information-retention consequences; none is selected yet.
Teacher weights can participate through initialization, parameter extraction or an explicit conditioning mechanism.
A universal network consuming flattened weights from arbitrary teachers is not a required interface.

An equivalent operational interface may factor out reusable context:
`context = C(teacher_weights, dataset)` followed by
`G_local(context, observation, instruction, original_action_sequence, temporal_context) -> replacement_sequence`.
Dataset information may reside in specialized parameters, a context cache or representative examples. A bare
`(observation, instruction, action)` tuple does not generally preserve the full dataset's information; sufficiency
must be assessed for the intended edits. Dataset compression preserves relevant information approximately unless
an exact sufficient representation is established. Restricting outputs to demonstrated actions could exclude
improvements that the generator is meant to discover.

## Expected Mechanism

Teacher weights provide learned visual/action priors; the dataset supplies scene-specific evidence and trajectory
context; temporal selection specifies the edit. Offline access to trajectory context may support refinements
that a teacher's ordinary online rollout does not produce. A superior unmodified teacher policy is therefore not
a prerequisite for this hypothesis. Its rollout remains a useful control.

Compression can reduce generation cost or provide useful regularization. A systematic trajectory-improvement
claim requires identifying its source of quality information: teacher priors, dataset outcomes, trajectory
preferences or additional consequence evidence. A separate scalar reward is not required. Matching teacher
outputs alone supplies no new shorter/faster/smoother objective. Full-task duration, path and jitter remain
distinct preferences subject to physical validity and task completion; fast demonstrations need not move faster
at every step. Preserve necessary multimodality.

An instruction-conditioned edited future is not automatically an action-conditioned transition predictor for
arbitrary candidate commands. The selected teacher's actual interface must determine whether it proposes actions,
predicts their consequences, or evaluates them. Model confidence or visual plausibility is not execution evidence.

### Candidate mechanisms and assessment

The three mechanisms address different questions: what dataset information to retain, how to calibrate its
representation for action generation, and how to use the retained knowledge. They can be combined, but none
implies either of the others.

| Candidate | Potential contribution | Unresolved assumption |
|---|---|---|
| Dataset codebook | Reuse interaction patterns across episodes and reduce conditioning cost | A compact context preserves distinctions needed for useful edits, including rare contacts and recovery actions |
| Dataset-calibrated normalization and representation | Put inputs and outputs into suitable coordinates and scales; optionally reduce irrelevant dimensions | The calibrated representation retains useful distinctions and supports reliable decoding into physical actions |
| Task-model distillation or fine-tuning | Transfer teacher knowledge and dataset context directly into a deployable model | The transferred information improves learning beyond an equally initialized BC baseline and is usable from deployment inputs |

**Codebook.** Quantizing action chunks alone does not summarize the dataset's observation, instruction and
transition relationships. A candidate memory could represent context-conditioned interaction fragments and their
effects, preserving temporal links. This is a hypothesis about the useful unit of compression, not a fixed schema.
Compression quality should concern action-relevant distinctions, not only average reconstruction error or feature
variance. Common prototypes can preserve common suboptimal behavior; quantization can erase rare but important
details. A codebook does not by itself rank candidates by trajectory quality.
[VQ-BeT](https://arxiv.org/html/2403.03181v2) provides a precedent for residual quantization of action chunks with
a continuous correction head. It does not establish compression of full-dataset context or improvement of source
demonstrations. If the output is a fine-tuned model, preserving training effect may be a different compression
objective from preserving a memory for generation; [Dataset Distillation](https://arxiv.org/abs/1811.10959)
illustrates the former distinction without establishing its effectiveness for this robotics setting.

**Dataset-calibrated normalization and representation.** The engineering objective is a useful input/output
parameterization for the current data domain, analogous to normalizing model inputs and actions and decoding
outputs back into physical units. Dataset statistics or learned structure can define scales, compatible local
coordinates or a compact representation. Subspace and manifold models are optional parameterizations; they do
not impose a requirement to contain a globally optimal solution. A restricted representation can be useful if it
supports improvements over the source demonstrations at lower learning or search cost.

Invertible normalization admits an inverse transformation. A lossy projection or learned code instead requires
a decoder whose reconstruction and physical-action errors must be measured. Low-variance directions can still
matter for contact. Reusing a pretrained teacher requires an explicit correspondence with its existing input/output
conventions: changing a numerical scale must not silently change the meaning of its actions. Normalization alone
does not reduce network capacity, establish physical validity or supply a trajectory-quality ranking. Its value
is assessed through conditioning, retained information and useful action improvements, without an optimality proof.

**Direct task adaptation.** Teacher priors, richer trajectory context and a suitable parameterization may improve
finite-data generalization. Preference-aware targets may additionally improve the behavior being learned. These
are different claims. Predicting the same demonstration actions through a codebook or diffusion model can still
be BC; BC is not restricted to single-action squared-error regression. A fair comparison uses the same teacher
initialization, deployment inputs, data and declared optimization/parameter budgets. Offline access to a recorded
future is valid for editing, but an online student cannot always recover decisions that depend on hidden information
absent from its inputs. Distillation must identify which context is available at deployment and which knowledge
can be transferred into parameters. Compression alone does not add evidence about unobserved consequences.

The main open design question is what compression must preserve: source-action reconstruction, controllable
effects, quality ordering or downstream training behavior. Dataset-calibrated coordinates provide a smaller first
mechanism than a jointly learned codebook and generator; the completed variance-scaling pilot below supplies
its current evidence. Exact inversion followed by exact decoding
reconstructs the source; an intended edit or changed generation distribution is necessary for improvement.

### Frozen-teacher editing pilot

The bounded editing comparison is complete in `docs/teacher_editing_results.md`. Teacher noise, TRAIN-calibrated
noise and direct edits produced 24, 19 and 16 source-suffix Pareto improvements respectively. Calibration did not
add stable quality yield; common-teacher continuation exposed success/efficiency trade-offs. ISSUE-047 records
the unresolved ranking and continuation questions. Codebook learning, lossy compression, richer trajectory
conditioning and direct model adaptation remain future mechanisms.

### Phase context and quality conditioned action labels

Evidence: `docs/action_label_optimization_review.md` audits eight primary sources. S2I and OLAF establish
relabeling at original observations for BC; SkiP and HYDRA expose phase and controller dependencies;
Beyond Success adds explicit quality supervision. These precedents support several candidate mechanisms,
but do not validate their combination in CEAM.

**Proposed priority.** Develop the offline editor around trajectory phase, intended effect and a source of quality
preference. Preserve teacher proposals as one candidate source. Include a bounded, temporally structured direct
editor as the baseline; simple variance calibration and codebook construction have lower priority until retained
information is linked to editing quality. This ordering is a research recommendation, not an active phase.

A candidate editor would identify a segment and a useful future effect from the complete recorded episode,
combine it with task/observation/state context and selected dataset fragments, and propose a coherent replacement
chunk. Recorded futures specify intent and context; they are not the realized futures of edits. The current
teacher does not accept a future-goal condition, so adding such conditioning requires an explicit adapter or
external selection mechanism. The ten-step arm-only restriction belongs to the completed pilot and is not a
permanent scope boundary: phase duration and gripper timing can become declared editing variables.

Quality preference could come from same-start executed alternatives, a small set of segment comparisons, or
verified task-specific constraints. Automatic duration/path/jerk measurements are useful after execution;
commands alone do not determine those outcomes. A learned preference ranker need not use online RL, but it still
needs evidence distinguishing better from worse candidates. The dual representations could supply effect and
task compatibility features if separately validated; their current match score is not a measured efficiency value.

Cross-episode fragments are a candidate for the first simple editing baseline: retrieve alternatives near a
segment's start context, adapt their actions with a low-dimensional residual and search for a better complete
route. Retrieved fragments are references rather than immutable edges. Transfer needs
object-relative/contact context and evidence of compatibility at both ends. Similar EEF locations alone do not
establish interchangeable states. A learned reachability/effect representation is an alternative to full 3D scene
reconstruction, with its own coverage and calibration requirements. Learned dataset context need not be a
discrete codebook.

### Simple editing operators and their information requirements

The candidate editor can combine the following operators. Their mechanisms and comparison targets differ;
none is assumed to improve a trajectory merely because it reduces the number of stored points.

| Operator | Candidate transformation | Required distinction |
|---|---|---|
| Geometric resampling | Allocate fewer targets to easy-to-interpolate portions and more to complex portions | Lower representation error at a fixed point budget is distinct from a shorter path or execution time. The G2 reference retains every observation and resamples its future action targets. |
| Geometric path editing | Smooth local deviations or replace a detour with a shorter connecting curve | Actual path changes need collision/contact checks; reconstruction fidelity alone does not evaluate shortcut quality. |
| Time allocation | Remove verified unnecessary pauses and vary speed along a retained path | Motion-critical waits, controller tracking and object dynamics constrain compression. Physical duration must be explicit. |
| Task-conditioned point or segment removal | Preserve required effects and remove candidate redundant visits, adjustments or operation cycles | Task phase labels and visual similarity do not establish that an operation is dispensable or that a replacement connection is executable. |
| Retrieved fragment adaptation | Select an alternative behavior from the dataset and adjust it with a low-rank residual | Both transferred effects and connecting dynamics require evaluation; donor preference need not transfer unchanged. |

[AWE](https://proceedings.mlr.press/v229/shi23b.html) provides a geometric waypoint-extraction precedent based on
bounded reconstruction error. [UVD](https://zcczhang.github.io/UVD/) extracts visual subgoals from pretrained
embedding changes. These source descriptions establish candidate extraction mechanisms; neither establishes
that points omitted from an extracted representation can be deleted from physical execution without consequence.

Task-conditioned deletion has two distinct scopes: dropping intermediate targets while retaining an operation,
and removing an operation itself. The latter requires checking what the operation changes or reveals and whether
its effect is still needed by later stages. Similar boundary images can hide changes in grasp stability, velocity
or object state. Necessary waiting can also have little visible motion. Candidate deletions therefore require a
new connection and complete-continuation evaluation, not simply concatenation of surviving delta commands.

An offline temporal mapping from each original observation to its edited target/segment must remain explicit.
Gripper and contact events constrain timing separately from continuous arm geometry. Low-rank residuals can
adapt the source trajectory, a geometric shortcut or a retrieved fragment; they are a parameterization shared
across operators rather than an exclusive candidate source. Semantic information can guide edit location,
required effects and candidate ranking; geometry, timing and residual parameters specify the executable change.
Task compatibility and efficiency preference remain separate validation questions.

Initial comparisons should isolate these operators before attributing an advantage to their combination. For a
semantic ranking claim, use a common proposal bank; if semantics changes generation or protected events, measure
that separate contribution. A complete editor combining all operators is not an implementation requirement.

### Dataset fragment search with low rank residual adaptation

Hypothesis: the dataset contains reference fragments whose adaptation and compatible composition improve a
source demonstration, even when no complete donor episode dominates it.
`docs/retrieval_refinement_results.md` supplies partial evidence for candidate availability:
physical retrieval found a successful faster donor at 71 of 90 starts, under the original suffix. That study
replaced one fixed ten-step block at a time; it did not evaluate residual adaptation or compositions of multiple
replacements.

A candidate parameterization for a fixed-duration arm-action chunk is
`vec(A_edit) = vec(A_reference_j) + B(context, j) z`, where `A_reference_j` has shape `[H, 6]`, B has shape
`[6H, k]`, and k is smaller than 6H. A fixed basis or a basis shared across training examples makes k the number
of per-candidate residual coordinates. If B is also freely optimized for every candidate, small k alone does not
restrict the available residual directions. Bases may encode temporal deformation or learned context-dependent
adjustments; no rank or basis is selected yet. Translation and rotation scales require explicit conventions.

The candidate search jointly selects the reference and residual coefficients. Selecting the highest-scoring
unaltered donor first can exclude a donor that becomes better after a small adaptation. Independent donor modes
are preserved instead of averaged. A graph over task-relevant boundary states is an optional search structure;
an adapted fragment's endpoint is determined by its edited execution and can differ from the recorded endpoint.
Spatial neighborhoods provide candidate lookup; object state, grasp/contact mode, task phase and necessary
motion history determine eligibility. Endpoint matching must retain velocity or equivalent boundary information
when evaluating joining costs such as jerk. A short fragment is comparable to a longer one only through its
achieved effect and remaining task cost. Source-specific donor quality is a retrieval prior, not a measured
value after transfer to a different start.

Low rank does not enforce small amplitude, temporal smoothness or physical feasibility. These properties need
their own parameterization, bounds or evaluation. The six continuous arm commands admit additive residuals;
gripper switching and variable-duration edits need explicit event and timing representations. They are not
obtained by smoothing binary commands or changing the simulator's control period. A basis over desired EEF
poses would require a declared conversion back to controller commands; it is a different interface from a
residual directly over the existing action labels.

The search retains the original complete route as an incumbent. For a common whole-trajectory preference J and
a feasible route set C containing the original route, an exact maximizer has J at least as large as the original.
The same comparison holds for a validated candidate accepted only when it improves the incumbent. With learned
scores or approximate state matching this only establishes a predicted ordering; actual non-regression must be
checked under a declared execution and uncertainty protocol. Retaining original edges alone is insufficient if
the search discards their complete route or if a local replacement changes the state required by the suffix.

Independent local maxima need not form the best feasible route. A fragment's ranking must include its resulting
state and continuation, and the connecting motions count toward duration, path and smoothness. A bounded first
comparison could retain several compatible routes and validate complete compositions against the unchanged
source, including a local-greedy control. Success and physical validity precede efficiency preference; if the
claim covers all three quality measures, acceptance must also enforce the declared non-regression tolerances
for duration, path and jitter. Measured monotonicity is scoped to evaluated starts and execution conditions.
This baseline can use recorded fragments and a fixed residual basis without a teacher. Learned bases or
teacher-guided residuals remain candidate mechanisms whose extra information must be isolated. Direct transfer,
residual adaptation and multiple-fragment composition need separate comparisons. Search-space, boundary-state
interface and validation budget remain to be specified in a plan.

### Development priorities for an offline editor

The primary design decisions are the unit of improvement, the evidence for improvement and the learned target.
Network architecture follows the information interface and candidate parameterization. A useful initial unit is
a coherent segment between task-relevant events, evaluated through its resulting state and declared continuation.
An event boundary describes a required effect, not necessarily one exact recorded pose. Success and declared
physical checks precede time/path/jitter preferences; faster completion does not imply faster individual steps.
Incomparable quality outcomes remain explicit until a tradeoff rule is selected.

Existing results locate different bottlenecks. `docs/preference_feedback_results.md` reports remaining-HELDOUT
pair accuracy of 52.79% for frozen matching and 69.98% for adapted representations with preference supervision,
while direct selection remains unreliable. `docs/continuation_sensitivity_results.md` reports 176/380 strict
preference reversals in its exact-restoration subset after changing the continuation. Consequently a donor's
quality is conditional on its transfer context and subsequent execution. These results do not isolate a benefit
of dual representations over a direct scorer with the same information, training outcomes and candidate bank.

Three learning responsibilities can be examined separately: transferable action effects and reachability;
task-conditioned compatibility and continuation-dependent preference; and efficient proposal generation or
residual adaptation. Cross-episode temporal data can supervise effects, while comparisons of executed alternatives
provide preference evidence. Observed transitions alone do not label unexecuted alternatives. Geometric costs
can be computed on measured trajectories; predicting physical effects and costs before execution remains a
learning problem. Full 3D reconstruction is one possible source of evidence, not a required intermediate target.

An offline editor's context may contain observation history, robot state, instruction, the source segment,
required future effects and retrieved dataset fragments. Its editable variables can be reference choice and
low-dimensional residual coefficients, with explicit duration and gripper events when those operators are
enabled. Decoding must produce controller-compatible actions and an anchor/temporal mapping. Dataset summaries
and teacher computations may be reused across queries; training a hypernetwork over teacher weights is not an
initial requirement. Fixed-duration adaptation and variable-duration phase editing remain distinct interfaces.

The downstream policy's information is a separate constraint. Offline future context can guide editing, but a
student must infer a useful action from its deployment inputs. Evaluation should examine whether edits depend on
hidden distinctions, remove useful approach/recovery coverage or create incompatible labels for indistinguishable
inputs. `docs/demo_refinement_results.md` already shows that lower source costs do not suffice: guided versus
original student SR was +1.33 percentage points pooled, with an interval spanning zero. A new editor must establish
both executed improvement and learnability under matched student inputs and training.

**Remaining evidence sequence.** The fixed-duration residual, architecture, continuation and jitter-target
diagnostics are complete in `docs/residual_editing_results.md`. They locate useful local smoothness priors but
do not establish reliable learned shortlisting or continuation-independent quality. Repeating the same
ten-step perturbation grid is not the next candidate study.

A phase-level editor should first specify required effects, eligible reference fragments, boundary/continuation
context and a controller-compatible timing interface. Semantic removal and variable duration are candidate
mechanisms; neither is required to be combined with all other operators. Preserve source routes and unedited
donor choices while evaluating adaptations. Measure candidate availability before attributing limitations to
network capacity. Teacher conditioning and learned residual bases remain possible sources of information.

New preference evidence should distinguish quality under its declared continuation and use consistent physical
measurements. A direct scorer remains the matched-information control for dual representations. Use independent
TRAIN alternatives and newly held-out sources; previously inspected query branches are exploratory evidence.
A matched student comparison still requires common initialization, architecture, deployment inputs and training
budget, at least three training seeds, and both SR and execution-quality measurements. Phase editing,
multi-fragment composition, learned generators and new label publication remain future work.

## Required Evidence

- For teacher-conditioned editing, compare original sequences, teacher-only proposals and dataset-conditioned
  generator proposals from matched starts, under declared search/execution budgets. Teacher-only performance is
  a control, not an admission gate.
- Isolate teacher priors, full-trajectory conditioning and compression where feasible. Define whether an experiment
  measures editing of a supplied dataset, transfer to held-out episodes, or both. Access to the edited episode's
  recorded future is valid offline context, but is not evidence about an unexecuted replacement's actual outcome.
- Measure compression cost and retained edit quality separately. Record which information is preserved in model
  parameters or dataset summaries and what must be passed at each edit. Include perturbations and states reached
  by edited sequences when assessing coverage. Teacher-weight interfaces must be explicit before implementation.
- Distinguish BC supervision targets from executed trajectory data. A recorded context paired with a replacement
  action can be a BC target without regenerated images; validate target quality and downstream distribution shift.
  A replacement chunk must be coherent from its anchor, rather than assembled from independently relabeled old
  states. For measured new trajectories, execute edits and record their corresponding observations. Original
  future frames cannot be treated as measured consequences of replacement actions or as verified RL transitions.
- Dataset utility requires downstream policies trained under a common protocol, with paired three-seed SR and
  efficiency measurements. Better generated demonstrations alone do not establish improved learned policies.
- For direct adaptation, compare against BC from the same pretrained teacher under matched deployment inputs
  and declared budgets. Separate compression/fidelity, generalization and trajectory-quality claims; editing
  evidence is not required for a method that outputs only adapted model parameters.

## Promotion Condition

A separate plan selects the output interface and specifies the teacher/weight interface, retained dataset
information, objective, validity evidence and compute budget. Action editing additionally defines temporal
selection and whether outputs are supervision pairs, coherent chunks or regenerated trajectories; direct
adaptation defines deployment inputs and the BC comparator.
Compression is optional until its role is defined. The frozen-teacher local arm-edit pilot is complete. Any richer dataset summary must specify the information
retained beyond coordinate variance and how its contribution will be isolated. Fixed-basis residual diagnostics
are complete in `plans/residual_action_editing.md`. Learned generator training, task-model adaptation and
action-label publication require a separate plan.

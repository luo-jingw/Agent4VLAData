# Action label optimization evidence review

Review date: 2026-10-04, America/New_York. Eight primary papers cover direct action relabeling, temporal
abstraction, teacher editing, quality supervision and cross-episode stitching. Published results below belong
to their source settings; they are not CEAM replications. This review extends
[the teacher calibration review](teacher_dataset_calibration_review.md) with evidence specific to better
action supervision. Proposed mechanisms and priorities are recorded in OPT-015 in `opportunities.md`.

Existing work demonstrates that changing action supervision can improve learned robot policies. Its gains
depend on the information used to construct the targets and on the deployed controller. Dataset normalization,
teacher likelihood, physical feasibility and trajectory quality are different quantities.

## Primary evidence

| Work and inspected source | Mechanism and reported evidence | Scope and limitations |
|---|---|---|
| [S2I, v2, III and IV, appendix V-C](https://arxiv.org/html/2409.19917v2) | Uses three reference expert demonstrations, segment representations and a geometric path simplifier; relabels actions at retained original observations. Image-based DP on Can-30 improves last-ten-checkpoint mean SR from 0.547 to 0.624 across five seeds. | Combines selection and optimization. Reference quality information and geometric assumptions contribute; this is not fully unsupervised preference discovery. BC-RNN uses relative actions, while the other evaluated policies use absolute actions. |
| [SkiP, v1, sections 3 and 4, appendices G and I](https://arxiv.org/html/2605.15536v1) | Labels free-motion segments toward the next key segment and preserves dense targets near precise interaction. RoboMimic average SR is 0.772 versus CoA-rev's 0.723; successful-run steps are 144.1 versus 211.8. | All evaluated interfaces use absolute targets. Delta-action extension remains future work. Low motion frequency can miss critical continuous contacts; individual tasks regress. Step count and elapsed time are separate metrics. |
| [HYDRA, v2, section 4.1 and appendix D.4](https://arxiv.org/html/2306.17237v2) | Recomputes sparse-phase labels with a waypoint controller and learns hybrid sparse/dense control. Its controller ablation reports 90% versus 58% SR with and without the online waypoint controller. | Requires mode annotations and a compatible execution mechanism. Its improvement cannot be attributed solely to replacing labels. It explicitly discusses state-distribution shift after relabeling. |
| [SAIL, v1, sections 4.2 to 4.4 and 7](https://arxiv.org/html/2506.11948v1) | Predicts achieved EEF poses instead of teleoperator command targets, combines a tracking controller with phase-dependent speed and consistent chunk scheduling. | Changes controller and runtime behavior as well as supervision. Faster robot-object dynamics remain a stated limitation. Achieved poses are not interchangeable with the raw commands of an unchanged controller. |
| [DemoDiffusion, v3, III-C and IV-B](https://arxiv.org/html/2506.20668v3) | Partially noises a retargeted human trajectory and denoises it with a frozen robot policy using fresh observations. Real-robot mean SR is 83.8%, versus 52.5% for retargeting and 13.8% for the base policy. | Shows trajectory information and teacher priors can complement each other. It does not test efficiency optimization of successful robot demonstrations or downstream training on edited labels. |
| [OLAF, v1, II-B to II-D](https://arxiv.org/html/2310.17555v1) | Uses human verbal corrections to select action adjustments before intervention, retains recorded observations, then trains the policy on relabeled and original samples. | Supplies additional corrective semantics through human feedback. It establishes a supervision interface, not a fully autonomous offline editor or measured counterfactual transitions. |
| [Beyond Success, v1, sections 3 to 5](https://arxiv.org/html/2511.22555v1) | Trains an offline quality critic using annotated constraint-critical segments and selectively ranks VLA candidates. SmolVLA elegant success rate rises from 49.8% to 67.2%; the success-only reward ablation reaches 56.8%. | This is inference-time selection with extra manual quality supervision. Elegant success includes task-specific constraints and differs from ordinary SR, duration, path length and jerk. It does not report an edited-label student comparison. |
| [Model-based trajectory stitching, journal version, sections 3 and 4](https://link.springer.com/article/10.1007/s10994-023-06392-z) | Uses learned transition likelihoods and values to choose better continuations across trajectories; inverse dynamics supplies connecting actions. Evaluates BC and offline-RL combinations. | Evidence is from state-based D4RL locomotion with recorded rewards. Geometric proximity only narrows candidates; learned reachability and value remain necessary parts of the method. This is not demonstrated transfer to image-based manipulation. |

The Beyond Success numerical claims above are from the inspected arXiv v1. Its
[CVPR 2026 entry](https://openaccess.thecvf.com/content/CVPR2026/html/Mao_Beyond_Success_Refining_Elegant_Robot_Manipulation_from_Mixed-Quality_Data_via_CVPR_2026_paper.html)
confirms publication; the proceedings PDF could not be fetched in this review, so version equality is not assumed.

## Public implementation availability

The [S2I repository](https://github.com/Junxix/S2I) exposes representation training, selection/optimization
evaluation, dataset utilities and downstream policy adaptations. Its README notes sampler and rollout changes,
so integration is broader than copying an action array. Code was inspected through its public tree and entry
pages; it was not installed or executed.

The [Beyond Success repository](https://github.com/Mwuqiu/BeyondSuccess) exposes only a README in the inspected
public tree. Its description mentions code and checkpoints, but runnable artifacts were not visible there.
The SkiP paper was accessible; its linked project site did not load, so runnable-code availability is unverified.

## Supervision targets and executed transitions

Three data products have different meanings:

| Product | Meaning | Valid use and evidence boundary |
|---|---|---|
| Recorded context with a replacement action | A proposed better action at an observed state or history | Can be a BC supervision target. It need not have produced the original next observation. Target quality and deployment coverage still require evaluation. |
| Recorded context with a replacement action chunk | A proposed plan starting from that context | The chunk must describe a coherent continuation from its anchor. Concatenating independently relabeled actions at different original states does not establish this coherence. |
| Executed edited trajectory | Observations and transitions generated by executing the replacement commands | Supports rollout metrics and measured transition supervision. New observations follow the edited execution; a model-generated future must be identified as a prediction. |

S2I and OLAF explicitly retain observations for relabeled supervision; HYDRA explains the resulting deployment
distribution issue. Therefore, regenerating every image is not a prerequisite for constructing BC targets.
The logical restriction is narrower: `(o_t, replacement_action, original_o_next)` is not evidence that the
replacement caused `original_o_next`. Such tuples cannot be treated as measured dynamics or RL transitions.

An editor may use the recorded future as offline context or as a desired outcome. That future does not certify
the consequence of a different action. A downstream student still operates on its deployment inputs; labels
that distinguish hidden situations solely through unavailable future information can conflict at deployment.

## CEAM compatibility and measured boundaries

CEAM's current dataset uses seven-dimensional EEF delta commands at 20 Hz and a binary gripper field.
`ceam/sim/libero_task_env.py` maps the gripper convention and sends commands to the existing simulator controller.
`ceam/teacher_editing/teacher_preprocessing.py` applies the public checkpoint's fixed action quantiles. Thus an
absolute future waypoint is not a drop-in replacement for a LIBERO action. The current interface contains no
SkiP-style controller that turns a distant target into a complete motion within one policy step.

The [completed teacher pilot](../../../synthesis/ceam/archive/docs/teacher_editing_results.md) supplies these measurements:

| Mechanism | Source-suffix Pareto improvements | Out-of-range proposals | Successful legal executions |
|---|---:|---:|---:|
| Direct Gaussian editing | 16/144 | 19/144 | 123/125 |
| Teacher noise editing | 24/144 | 7/144 | 129/137 |
| Dataset variance calibrated noise | 19/144 | 5/144 | 131/139 |

These are candidate-bank results from six query episodes, not trained-policy SR. The three local edit families
use ten-step chunks and preserve the source gripper schedule. They do not test changing the gripper timing or
replacing a complete manipulation phase. Teacher-only proposals have independent gripper outputs and different
edit magnitudes, so they do not isolate the effect of relaxing that restriction.

One selected calibrated edit reduces source-suffix completion from 174 to 129 steps, yet requires 191 steps
against the original's 136 under one common teacher-continuation seed. Its local benefit therefore cannot be
assigned independently of continuation. The pilot tests neither full-dataset trajectory conditioning nor a
learned quality selector. Variance calibration has no consistent additional quality yield in that bank.

The checkpoint in this pilot ignores robot state, although state is retained in the observation record. Access
to state in a future editor would be an additional input path, not an already-used teacher capability.

## What the evidence establishes

The direct relabeling studies provide downstream policy evidence that source actions need not remain immutable
BC targets. The temporal methods specify where coarse progress is appropriate and where dense control is
retained. Teacher editing supplies context-dependent proposals. Quality-critic and corrective-feedback methods
add information about desirable behavior. Stitching methods add reachability and continuation-value estimates.
These are separate information sources, not consequences of coordinate normalization alone.

Reducing time, reducing EEF path and reducing jitter are distinct outcomes. A shorter sampled command sequence
does not establish a faster physical execution; a smoother command signal does not establish a smoother
contact trajectory. Task completion, physical validity, elapsed duration, measured path and motion smoothness
need separate reporting. Success-conditioned efficiency can also change when the successful episode set changes.

None of the inspected sources validates the complete CEAM proposal of compressing teacher weights and an
entire dataset into a generator of better source labels. The sources establish component precedents with
different assumptions. Current CEAM results supply a working proposal interface and execution measurements,
but no teacher-edited-dataset student evaluation. The separate non-teacher pilot in the demonstration-refinement report already includes nine student policies. Research novelty and end-to-end benefit remain unestablished.

## Review resources

This review used public read-only web sources and local documents/source inspection. No credentials, cluster
jobs, GPU allocations, model training, simulator rollouts or dataset modifications were used. Additional GPU
consumption was zero. Source versions and section references are recorded above; untested designs remain in
`opportunities.md` rather than an active execution plan.

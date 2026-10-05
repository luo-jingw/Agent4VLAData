# Teacher and dataset calibration: evidence review

Review date: 2026-10-03. Five primary papers were examined beyond abstracts, together with public model and
inversion interfaces. External results below are published observations, not CEAM replications. Candidate
mechanisms and the experimental proposal belong to OPT-015 in opportunities.md.

## Scope and evidence questions

The target is improvement over existing demonstrations using teacher weights, dataset context and an episode/time
edit location. Dataset calibration may change coordinates, scales or conditioning. Direct task adaptation is an
alternative output. Neither global optimality nor a mandatory codebook is required. This direction is independent
of the dual-representation architecture in docs/framework.md.

The audit distinguishes action editing, representation/calibration, policy adaptation and model compression.
Physical completion time, command-generation latency, candidate quality and learned-policy SR are separate outcomes.

## Primary evidence

### DemoDiffusion

Source: [2506.20668v3, §§III-C, IV-B, Limitations](https://arxiv.org/html/2506.20668v3).

Partially noise a retargeted human trajectory and denoise it with a frozen policy under current observations.
The real-robot study reports 83.8% mean SR versus 52.5% for retargeting and 13.8% for the base policy, over eight
tasks with ten trials each. It uses task-dependent noise levels and closed-loop observation feedback. Timing
alignment remains a stated limitation. This supports combining trajectory information and a teacher whose
ordinary rollout is weak. It does not establish improving successful robot demonstrations' duration, path or
jitter, compressing a dataset, or predicting counterfactual outcomes without execution.

### Flow Reversal Steering and offline DSBC

Source: [2606.13675v2, §§4.1–4.3, 5.3, E.2](https://arxiv.org/html/2606.13675v2).

FRS reverses then reapplies a frozen flow policy to reference actions. Finite Euler error changes outputs;
exact integration reconstructs the reference. DSBC learns a small noise policy from inverted actions. Offline
LIBERO-90 DSBC reports aggregate SR increasing from 30% to 40% across 89 success-filtered tasks. Main BC controls
use small action policies rather than fine-tuning the same VLA. LIBERO DSBC repeats a learned 7D noise vector
through the chunk and pads other dimensions. These results support offline latent adaptation, not automatic
efficiency improvement of source trajectories. The [project page](https://flow-reversal-steering.github.io/)
listed code as forthcoming at inspection.

### FlowDAgger

Source: [2607.08877v1, §§4–5](https://arxiv.org/html/2607.08877v1).

Per-step fixed-point inversion maps corrective actions to a frozen generator's noise input. A small policy learns
these targets using intervention and successful-rollout buffers. Twelve MetaWorld tasks report mean SR .78,
versus .53 base and .71 SFT. SFT uses offline demonstrations; DAgger methods receive on-policy corrections.
These information differences matter for interpreting the comparison. Inverting a joint world-action process
also differs from inverting an action head. Evidence supports a useful adaptation interface, not improvement
from an unchanged offline dataset alone. The [official repository](https://github.com/microsoft/FlowDAgger)
contains separate pi0.5/JAX and GR00T/PyTorch implementations with scripted intervention examples.

### FAST

Source: [2501.09747v1, §V-B](https://arxiv.org/html/2501.09747v1).

FAST applies training-data quantile normalization, DCT, coefficient quantization and BPE to action chunks.
It provides a concrete precedent for dataset-dependent scaling and compact temporal action coding. Its target
is a useful action representation for policy learning; it does not encode full observation/instruction/transition
context or optimize the quality of existing trajectories. Compression fidelity and downstream control remain
different measurements. A codebook is therefore an optional representation mechanism, not a source of quality
supervision by itself.

### OneDP

Source: [2410.21257v1, §§2.2, 3](https://arxiv.org/html/2410.21257v1).

OneDP trains a one-step action generator by matching a pretrained diffusion policy using reverse-KL gradients
on diffused actions. The stochastic variant slightly exceeds its DDPM teacher on average and matches the EDM
teacher's average in the reported simulation settings. Real-robot evaluation includes completion time; reduced
prediction delays improve responsiveness. This establishes that practical distillation can preserve or improve
control, not that compression supplies a universal better-action target. Latency-driven gains must be separated
from changes to a fixed-rate simulated action sequence and from efficiency of recorded demonstrations.

## Deductions from explicit interfaces

These deductions state their mathematical conditions; they are not new empirical results.

### Exact encoding and decoding do not edit an action

Fix teacher weights W and conditioning c, and let F_W(c,z) be its deterministic noise-to-action map in its own
normalized coordinates. If I_W is an exact inverse on the relevant domain, then

`F_W(c, I_W(c, A)) = A`.

An edit must enter through a changed latent, conditioning, projection, stochastic step, learning objective or
approximate numerical procedure. Reconstruction error alone is not evidence of improvement. Keeping an accurate
round trip as a control separates intended edits from solver-induced changes.

### Dataset calibration and teacher conventions are separate transforms

Let N_T map physical actions into a frozen teacher's original coordinates, and S_D map latent codes into
dataset-calibrated coordinates. A correctly composed edit has the interface

`z = I_W(c, N_T(A))`

`u = S_D(z)`

`A_new = N_T_inverse(F_W(c, S_D_inverse(Edit(u))))`.

With Edit equal to identity and exact inverses, the original action is recovered. Replacing N_T's statistics
without compensating the model interface instead changes what physical commands its outputs represent.
Lossy codes require measured decoding errors. A Gaussian prior, small code distance or finite latent norm does
not imply physical feasibility or trajectory quality.

### Noise-space adaptation is not automatically dimensional compression

An invertible flow preserves dimension. A compact steering parameterization introduces an additional restriction;
its size does not describe the size of the frozen decoder. Since F_W depends on c, the same latent change can
have different action effects at different observations. A global codebook of inverted noises is consequently
not guaranteed to identify shared interaction semantics. Likewise, a small learned steering model does not
remove the teacher's inference memory or compute.

Under the stronger condition that demonstration actions at each c follow the teacher's distribution exactly,
an exact invertible map sends those demonstrations back to the teacher's original noise prior. For a standard
Gaussian prior, population mean/std calibration of those codes is therefore the identity. A narrow physical
task domain does not by itself imply a narrow or anisotropic noise distribution. Deviations from that prior
can indicate a dataset/teacher distribution difference; they do not identify which distribution is better.

### Better candidate generation and better selection are distinct claims

A generator can increase the frequency of useful actions without predicting which candidate succeeds. If a
simulator selects successful improvements from its samples, the result measures a verifier-assisted editor.
Including the original successful trajectory in every bank can preserve selected success even for a poor
generator. All-candidate success/feasibility and improvement yield are necessary to interpret such a result.

## Read-only integration audit

Public implementation references are mutable main-branch files inspected on the review date. Exact revisions,
weights and assets must be pinned in any execution plan. No compatibility run was performed.

| Item | Observed interface or local evidence | Implication |
|---|---|---|
| Candidate teacher | OpenPI lists `gs://openpi-assets/checkpoints/pi05_libero`; its sampler accepts an explicit noise tensor | A frozen flow decoder has a concrete public entry point; weights are not installed here |
| Velocity access | `Pi0.sample_actions` computes the velocity inside a nested sampling step with a cached observation prefix | Inversion needs an explicit adapter to this computation; an HTTP action-only endpoint is insufficient |
| Model shape | Default Pi0Config uses action dimension 32 and horizon 50; particular checkpoint configs can override defaults | Inspect the selected checkpoint; physical action dimension 7 does not identify latent shape |
| Normalization | OpenPI defines Normalize/Unnormalize for mean/std and quantile statistics | Preserve checkpoint statistics; dataset calibration needs its own declared transform |
| Teacher state | OpenPI LIBERO uses EEF position, quaternion converted to axis-angle, and two finger positions | Eight physical state values, different from CEAM's existing nine-value joint/finger feature input |
| Available dataset | Goal metadata contains `observation.state[8]`, `ee_state[6]`, joint[7], gripper[2], two 512-square videos, action[7], 20 fps | Required pose fields already exist; no need to infer absence from the existing CEAM feature cache |
| Dataset conversion | `scripts/convert_libero_raw.py` builds state from EEF position, axis-angle and finger positions | Metadata labels roll/pitch/yaw do not establish an Euler-angle representation |
| Gripper | CEAM dataset g=1 means open; `LiberoTaskEnv.to_sim_actions` applies `1-2g` | Account for teacher, dataset and simulator conventions exactly once |
| Images | CEAM observes rotated simulator images; OpenPI LIBERO also rotates both views, then resizes with padding to 224 | Different render resolutions and MuJoCo versions still need a paired interface check |
| Environment | CEAM pins Python 3.10, torch 2.7.1, transformers 4.53.3; current OpenPI declares Python >=3.11 and a JAX/Flax stack | The inspected package cannot simply be installed into CEAM's current interpreter; integration is future work |
| Local weights | `.hf/hub` contains DINOv2-base and SigLIP encoder caches | No reviewed generative teacher checkpoint is present in the project model cache |

Official sources: [sampler](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py),
[model configuration](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py),
[transforms](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py),
[LIBERO example](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/main.py),
[checkpoint catalogue](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md),
[dependencies](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/pyproject.toml).

Local evidence: `outputs/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/meta/info.json`,
`scripts/convert_libero_raw.py:521`, `ceam/data/lerobot_episode_reader.py:28`,
`ceam/sim/libero_task_env.py:324`, `pyproject.toml`, `uv.lock`, and directory names under `.hf/hub`.

## Relation to existing project results

`docs/demo_refinement_results.md` already records that simulator-selected random candidates can improve source
trajectory costs. A teacher editor must therefore be compared with equal-budget direct action editing, not only
with original demonstrations. `docs/retrieval_refinement_results.md` likewise separates candidate availability,
ranking and measured acceptance. ISSUE-044 and ISSUE-045 record continuation dependence and restore sensitivity;
a local replacement's outcome cannot be presented as independent of its continuation or restoration protocol.

## Research disposition and resource record

The evidence supports a testable comparison between dataset-calibrated editing through a frozen flow teacher
and equally budgeted editing in normalized action coordinates. Published interfaces remove the need to posit a
new universal weight-consuming network. Whether this improves successful demonstrations' quality remains open
and is specified as a hypothesis in OPT-015. Codebooks, pruning, online RL and 3D reconstruction are not required
to test this comparison. The research stopped at this experimental opportunity.

Work consisted of public primary-source browsing, read-only source/metadata inspection, a standard-library JSON
metadata read and documentation edits. No teacher weights or repositories were downloaded; no model inference,
training, simulator runs, new SR measurements or dataset writes occurred. GPU allocation and GPU-hours: zero.
Runtime, peak memory and local speedups of the proposed editor are unmeasured. The published OpenPI memory
guidance is not a measured budget for inversion or for this project's environment.

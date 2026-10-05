# Stage-one losses

`ceam/losses/`. Projected coordinates z = Uᵀψ with U from `QueryOutput.basis`; D = ‖z_T − z₀‖² + ε with gradients. Stop-gradient applies only to ψ₀ and ψ_T inside `start_target` = Uᵀsg[ψ₀] and `end_target` = Uᵀsg[ψ_T]; U always receives gradients. All arithmetic is float32 (inputs cast, autocast disabled for the matmuls).

| Class | File | Value |
|---|---|---|
| `SigRegLoss` | `sigreg_loss.py` | Epps–Pulley statistic of 256 fresh random unit projections of ψ against 𝒩(0, 1): trapezoid over 17 knots in [−3, 3] of \|φ̂(t) − e^{−t²/2}\|² e^{−t²/2}, averaged over directions (not multiplied by N) |
| `EpisodeProjector` | `projected_episodes.py` | builds `ProjectedEpisodes` from `StageOneStates` and `QueryOutput` |
| `PathLoss` | `path_loss.py` | mean over (b, t) of ‖(I − nnᵀ)(z_t − start_target)‖² / D, n = sg[(z_T − z₀)/√D] |
| `GoalLoss` | `goal_loss.py` | mean_b ‖Uᵀĝ − end_target‖² / D |
| `EndpointLoss` | `endpoint_loss.py` | mean_b ‖z_T − z′_T‖² / D |
| `ParaphraseLoss` | `paraphrase_loss.py` | mean_b (1 − ‖U_aᵀU_b‖²_F / r) + mean_b (1 − cos(q_a, q_b)) |
| `InstructionContrastLoss` | `instruction_contrast_loss.py` | for every uniform frame o_t with horizon k ∈ [1, T−1−t] and effect z = ψ(o_{t+k}) − ψ(o_t): logits s·cos(z, q) for the anchor's own sampled variant and for one random variant of every negative task (`TaskRelationTable.negative_mask`); loss = logsumexp(logits − own); s = exp(min(logit_scale, log 100)); mean over samples with at least one negative (0, in the graph, when none); cos of a zero vector is 0 |
| `StageOneObjective` | `stage_one_objective.py` | SIGReg on the uniform frames (default `sigreg_frames: uniform`) or on every encoded frame, i.e. start, end, partner end, uniform and uniform future (`all`); path (target `perpendicular`, `interpolation` in z, or `interpolation_full` in ψ, see `docs/stage_one_flow.md`), goal, endpoint on query a; instruction contrast on the uniform → future effects with the own direction passed explicitly; paraphrase on (a, b); weighted sum in `StageOneLossReport` |

## Measured (`scripts/check_losses.py`, CPU, seed 0, ε = 1e-6)

- SIGReg: 2.4e-4 for 𝒩(0, I) at N = 4096 (finite-sample floor ≈ 1.06/N; 2.0e-3 at N = 512), 0.386 for 0.1·𝒩(0, I), 0.511 for 3·𝒩(0, I), 0.168–0.396 for rank-1 data.
- PathLoss: 9.3e-14 on an exactly straight projected path; with isotropic ψ noise σ it matches (r − 1)σ²·mean(1/D).
- Scaling every ψ and ĝ by 0.1 or 10 changes path, goal and endpoint by < 1.4e-5 relative (from ε).
- Noise-only path with zero true displacement: PathLoss ≈ 1.0 at σ = 0.1; ε dominates D below σ ≈ 1e-3 and pulls the ratio toward 0 (0.80 at σ = 1e-3, 1.7e-3 at σ = 1e-5).
- ParaphraseLoss: 0 for equal subspaces (including U_b = U_a R), 1 for orthogonal subspaces, 0.969 ≈ 1 − r/d for independent random subspaces.
- Gradients: ψ₀ reaches the path and goal terms only through D; with D detached it has no path to them. SIGReg reaches only the uniform-frame ψ under the default `sigreg_frames: uniform`.
- `StageOneObjective` gives the same float32 report for float32 inputs, bf16 inputs, and bf16 inputs under autocast.
- InstructionContrastLoss: equals log(1 + 9e^{−s·cos}) with 9 orthogonal negatives (0°: 5.4e-6, 90°: log 10, 180°: 16.48 at s = 14.29); s is clamped at 100; invariant to scaling ψ (rel. 9e-8); 0 with an all-False mask; a mask with True on its diagonal is rejected; gradients reach ψ of both frames, the own and other directions and logit_scale. The ψ-gradient per unit angle scales as 1/‖z‖, so short-horizon effects get larger gradients per sample.

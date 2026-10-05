# Dynamics counterfactual check (phase 12)

The check executes counterfactual action chunks from demonstration states in the simulator and compares the stage-two dynamics head Δ(o, a) with the true effect z = ψ(o′) − ψ(o).

Code:

- `ceam/sim/demo_state_restorer.py` (`DemoStateRestorer`);
- `scripts/check_dynamics_counterfactuals.py`;
- commit `811b5ba` (merged in `4de8935`).

Run:

- `outputs/dynamics_counterfactuals/goal_dynamics_short_step3000_heldout_seed0/`, holding `config.json` (arguments, commit, host, checkpoints), `candidates.jsonl`, `samples.jsonl` and `summary.txt`. The log is `tmp/runlogs/p12/check_dynamics_counterfactuals_seed0.log`.
- node-65 CUDA 5, seed 0, 503 s.
- Checkpoints: Δ `outputs/runs/goal_dynamics_short/checkpoints/step_003000.pt`; stage one `goal_instruction_contrast/step_005000.pt`.

## Samples and candidates

- **Samples**: all 43 HELDOUT LIBERO-Goal episodes × 3 time points t = round(f · (T − 1 − 10)), f ∈ {0.1, 0.4, 0.7}, giving 129 samples.
- **Candidates**: 14 per sample, 1806 in total. Each is executed for real from the restored frame t, all 10 actions, with no stop at success.
  - demo: the demonstration chunk a_t … a_{t+9};
  - stay: 10 zero-delta steps with the gripper command in force at frame t;
  - perturbed: σ ∈ {0.5, 1, 2} × 4 draws. The demo chunk plus a constant offset σ · ε in the standardized space of the 6 arm dimensions (ε ~ N(0, I); gripper unchanged), then clipped to the controller's input range [−1, 1]. Δ receives the clipped chunk; 7.4% of perturbed entries are clipped (σ = 2: 14.6%). The generator is seeded with [seed, episode_id, t].

Frame restoration is described in `docs/libero_sim.md` (Restoring a dataset frame).

## Restoration fidelity (129 samples)

- **Frame t vs the dataset**: max proprio error 2.2e-5. Mean DINOv2 token cosine: agent 0.969 (min 0.960), wrist 0.945 (min 0.861).
- **After the demo chunk vs dataset frame t + 10**:
  - proprio ≤ 1.6e-5;
  - ‖ψ_sim(o′) − ψ_cache(o_{t+10})‖ / ‖z_cache‖ has median 0.052;
  - mean cos(z_sim, z_cache) 0.990 (min 0.870).
- **Repeated restores** are bit-identical in 126 of 129 samples. The other 3 (task 7) differ only in rendering: proprio identical, 2 wrist pixels off by one level.
- **ψ gap at the restored frame**: median ‖ψ_sim(o_t) − ψ_cache(o_t)‖ is 0.835 at f = 0.1, 0.199 at f = 0.4 and 0.259 at f = 0.7. Proprio and token cosines are at the same levels at all three time points. The f = 0.1 gap comes from ψ changing steeply near episode starts; for example, episode 430 at t = 8 has distance 9.5 against ‖z_cache‖ 18.6.
- **Offline reference on the demo chunks**: Δ̂ from the cached frame t has median relative error 0.190 and mean cosine 0.953; Δ̂ from the simulator render gives 0.199 / 0.953.

## Prediction error

Relative error is ‖Δ̂ − z‖ / ‖z‖; cosine is cos(Δ̂, z).

| Candidates (n) | median ‖z‖ | median ‖Δ̂‖ | median relative error | cosine mean / median |
|---|---|---|---|---|
| demo (129) | 4.75 | 4.72 | 0.199 | 0.953 / 0.987 |
| stay (129) | 1.24 | 2.11 | 1.397 | 0.501 / 0.573 |
| perturbed σ = 0.5 (516) | 4.95 | 4.80 | 0.265 | 0.912 / 0.981 |
| perturbed σ = 1 (516) | 5.22 | 4.98 | 0.435 | 0.819 / 0.952 |
| perturbed σ = 2 (516) | 7.86 | 5.59 | 0.655 | 0.669 / 0.856 |

Median relative error by time point (demo / σ 0.5 / σ 1 / σ 2 / stay):

| f | demo | σ 0.5 | σ 1 | σ 2 | stay |
|---|---|---|---|---|---|
| 0.1 | 0.17 | 0.23 | 0.41 | 0.65 | 1.75 |
| 0.4 | 0.26 | 0.43 | 0.56 | 0.84 | 1.37 |
| 0.7 | 0.17 | 0.23 | 0.34 | 0.50 | 1.29 |

Two patterns stand out:

- The error grows with the distance from the demonstrated action. Δ also underestimates the size of large deviations: at σ = 2, median ‖Δ̂‖ is 5.6 against a median ‖z‖ of 7.9.
- The true effect of stay is not zero: median ‖z‖ is 1.24, a quarter of a demo chunk's. The closed loop nevertheless scores stay as zero effect (m = 0). Δ also overpredicts the effect of stay (1.40 relative error). The training data has no stay transitions (ISSUE-001).

## Match scores and ranking

The match score is m = cos(effect, q_l) under the task's text-only query.

Share of candidates with a positive score, true (z) vs predicted (Δ̂):

| Candidates | true m > 0 | Δ̂ m > 0 | same sign | Δ̂ m > 0 but true m ≤ 0 | mean true m / Δ̂ m |
|---|---|---|---|---|---|
| demo | 0.93 | 0.97 | 0.96 | 0.04 | 0.369 / 0.389 |
| stay (true or predicted effect) | 0.80 | 0.83 | 0.72 | 0.16 | 0.222 / 0.267 |
| σ = 0.5 | 0.90 | 0.93 | 0.94 | 0.04 | 0.338 / 0.364 |
| σ = 1 | 0.80 | 0.90 | 0.87 | 0.12 | 0.264 / 0.323 |
| σ = 2 | 0.61 | 0.81 | 0.75 | 0.23 | 0.140 / 0.278 |

Δ is optimistic away from the demonstrations. Its predicted match falls much less with σ than the true match does: from σ = 0.5 to σ = 2, the mean predicted m drops 0.086 and the mean true m drops 0.198. At σ = 2, 23% of chunks score positive under Δ̂ but not under z.

Ranking over the 14 candidates of a sample, with stay scored m = 0 as in the closed loop:

| Subset | mean Spearman ρ | median ρ | same top-1 | stay chosen (z / Δ̂) | demo above all perturbed (z / Δ̂) |
|---|---|---|---|---|---|
| all (129) | 0.499 | 0.565 | 0.264 | 0.023 / 0.023 | 0.031 / 0.070 |
| f = 0.1 | 0.637 | 0.732 | 0.279 | 0.047 / 0.070 | 0.047 / 0.186 |
| f = 0.4 | 0.471 | 0.508 | 0.256 | 0.023 / 0.000 | 0.023 / 0.023 |
| f = 0.7 | 0.388 | 0.486 | 0.256 | 0 / 0 | 0.023 / 0 |

- The true score of the chosen candidate averages 0.464 when choosing by z and 0.321 when choosing by Δ̂; the demo chunk's own averages 0.369.
- The true best candidate is in Δ̂'s top 3 in 53% of samples (median rank 3).
- Mean ρ per task ranges from 0.36 (task 9) to 0.67 (task 3).
- Under z, the demo chunk scores above all 12 perturbed chunks in only 3.1% of samples. In the rest, at least one perturbed chunk's true effect is better aligned with q_l than the demonstration's.

Script for the sign table and the chosen-candidate scores: `tmp/analysis/p12_extra.py`.

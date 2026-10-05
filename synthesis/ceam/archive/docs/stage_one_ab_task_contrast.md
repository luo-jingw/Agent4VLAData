# Stage one: task-contrast A/B on LIBERO-Goal

Configs `ab_goal_contrast0.json` and `ab_goal_contrast1.json` (removed after the instruction contrast replaced the task contrast; see git history before commit `24144fa`): LIBERO-Goal, TEXT_ONLY, d = 256, r = 8, B = 64, M = 8, 5000 steps (warmup 250, lr 3e-4 cosine to 0), bf16 autocast, seed 0; the only difference is `weights.task_contrast` 0 vs 1. Trained on node-70 (L40S), 4.5–4.7 steps/s, 7.6 GiB peak. Diagnostics (`scripts/diagnose_stage_one.py`) on the 43 HELDOUT episodes every 1000 steps; outputs under `outputs/runs/ab_goal_contrast{0,1}/diagnostics/`.

## Step 5000 (medians over HELDOUT episodes)

| Quantity | contrast 0 | contrast 1 |
|---|---|---|
| Task overlap ‖U_iᵀU_j‖²/r, off-diagonal mean (min–max) | 0.996 (0.993–0.998) | 0.838 (0.737–0.937) |
| ‖U_aᵀ(ĝ_a − ψ_T)‖² / D_a, own endpoint | 1.1e-4 | 9.3e-3 |
| same, other-task endpoints | 5.8e-4 | 12.5 |
| other-task / own ratio | 5.0 | 1188 |
| fraction of other-task endpoints closer to ĝ_a than own | 0.41 | 0 |
| task-ID probe HELDOUT, ψ(o_T) | 0.837 | 0.907 |
| task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 0.414 | 0.900 |
| task-ID probe HELDOUT, ψ(o₀) control | 0.093 | 0.116 |
| progress Spearman(t, p_t) | 0.35 | 0.73 |
| path deviation / ‖z_T − z₀‖ | 0.011 | 0.16 |
| goal error / ‖z_T − z₀‖ | 0.010 | 0.096 |
| displacement ratio | 4.22 | 0.89 |
| effective rank of cov(ψ) | 55.9 | 5.0 |
| SIGReg statistic | 0.0054 | 0.022 |

## Over training (contrast 0 → contrast 1)

| Step | overlap off-diag | other/own ratio | U_aᵀψ(o_T) probe | ψ(o_T) probe | effective rank |
|---|---|---|---|---|---|
| 1000 | 0.981 / 0.826 | 3.5 / 80 | 0.749 / 0.926 | 1.000 / 0.953 | 25.6 / 3.0 |
| 2000 | 0.995 / 0.829 | 4.9 / 588 | 0.707 / 0.916 | 0.977 / 0.953 | 37.7 / 4.5 |
| 3000 | 0.997 / 0.845 | 6.0 / 6169 | 0.542 / 0.916 | 0.930 / 0.907 | 46.9 / 4.6 |
| 4000 | 0.994 / 0.846 | 4.5 / 7755 | 0.472 / 0.895 | 0.837 / 0.907 | 53.9 / 4.5 |
| 5000 | 0.996 / 0.838 | 5.0 / 1188 | 0.414 / 0.900 | 0.837 / 0.907 | 55.9 / 5.0 |

## Progress curves

- contrast 0: on every task, the projected progress p_t reaches ≈ 1 within the first ~5% of the episode and stays there. The projected coordinate separates "the episode has started" from the start frame, identically for all tasks.
- contrast 1: on most episodes p_t stays near 0 for ~90% of the episode and jumps to 1 in the last 5–10%; a few episodes (e.g. task 7, episode 303; task 5, episode 225) rise gradually.

## Findings

- Without the task-contrast term, longer training makes the query more instruction-independent: overlap stays at 0.99+, other-task endpoints stay within a few percent of the path length of ĝ_a, and the task information in U_aᵀψ(o_T) decays (probe 0.75 → 0.41) while ψ(o_T) keeps most of it.
- With the term, other-task end states sit ~3.5 path lengths from ĝ_a (squared ratio 12.5), no other-task endpoint is closer than the own endpoint, and each 8-d subspace keeps task identity (probe 0.90).
- With the term, the projected path is a late step rather than gradual progress, and ψ concentrates its variance in ~5 directions.

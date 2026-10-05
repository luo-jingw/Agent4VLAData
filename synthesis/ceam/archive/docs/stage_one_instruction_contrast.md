# Stage one: instruction contrast on action effects (LIBERO-Goal)

Match m(z, l) = cos(z, q(o, l)) with z = ψ(o_{t+k}) − ψ(o_t); instruction contrast with learnable logit scale and horizons k uniform on [1, T−1−t]; negatives from `data/task_relations/libero_goal.json` (all task pairs). Three runs, 5000 steps each (warmup 250, lr 3e-4 cosine, B = 64, M = 8, bf16, seed 0), trained on node-70 (L40S, shared with another user's jobs; 2.1 steps/s text-only, 0.9 steps/s text-and-observation):

- `configs/stage_one/goal_instruction_contrast.json` — TEXT_ONLY;
- `configs/stage_one/goal_instruction_contrast_obs.json` — TEXT_AND_OBSERVATION, q queried with the current frame o_t as reference;
- `configs/stage_one/goal_instruction_contrast_sigreg10.json` — TEXT_ONLY, SIGReg weight 10.

Evaluation (`ActionRankingReport`, k = 10 frames, 43 HELDOUT episodes, 506 anchors): observed k-step outcomes stand in for E(o, a), so every number is the ranking the match would give if E predicted outcomes exactly. "Other-task action" at t > 0 is the k-step effect at another task's nearest frame in frozen features (median feature distance 0.47–0.81 against 0.59 for a k-frame step within an episode).

### Step 5000, pooled over t/T

| quantity | TEXT_ONLY | TEXT_AND_OBS (q on o_t) | TEXT_ONLY, SIGReg ×10 |
|---|---|---|---|
| P(demo > stay) | 0.945 | 0.933 | 0.933 |
| P(demo > reverse) | 0.970 | 0.940 | 0.963 |
| P(demo > other-task action) | 0.964 | 0.920 | 0.966 |
| instruction top-1 (chance 0.1) | 0.870 | 0.897 | 0.854 |
| instruction pairwise | 0.945 | 0.973 | 0.928 |
| sensitivity (median std of matches) | 0.186 | 0.225 | 0.080 |
| effective rank of demo effects | 15.065 | 17.955 | 20.699 |
| shared start P(demo > other first chunk) | 0.866 | 0.859 | 0.660 |
| effective rank of cov(ψ) | 16.6 | 22.3 | 31.5 |
| SIGReg statistic | 0.0254 | 0.0165 | 0.0137 |
| task-ID probe HELDOUT ψ(o_T) / ψ(o₀) | 1.000 / 0.070 | 1.000 / 0.070 | 1.000 / 0.163 |
| progress Spearman (auxiliary) | 1.00 | 0.92 | 0.99 |

TEXT_ONLY: logit scale s 14.29 → 18.30; instruction contrast 2.421 → 0.0119

TEXT_AND_OBS (q on o_t): logit scale s 14.29 → 19.42; instruction contrast 2.296 → 0.0077

TEXT_ONLY, SIGReg ×10: logit scale s 14.29 → 31.18; instruction contrast 2.421 → 0.0620

### By t/T quarter, step 5000


TEXT_ONLY

| quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| P(demo > stay) | 0.922 | 0.945 | 0.962 | 0.958 |
| P(demo > reverse) | 0.945 | 0.984 | 0.977 | 0.968 |
| P(demo > other-task action) | 0.960 | 0.974 | 0.963 | 0.959 |
| instruction top-1 (chance 0.1) | 0.758 | 0.844 | 0.962 | 0.958 |
| instruction pairwise | 0.916 | 0.954 | 0.962 | 0.958 |
| sensitivity (median std of matches) | 0.156 | 0.187 | 0.191 | 0.204 |
| effective rank of demo effects | 9.566 | 7.867 | 10.156 | 9.031 |

TEXT_AND_OBS (q on o_t)

| quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| P(demo > stay) | 0.824 | 0.961 | 0.992 | 0.989 |
| P(demo > reverse) | 0.845 | 0.945 | 0.985 | 0.979 |
| P(demo > other-task action) | 0.862 | 0.903 | 0.951 | 0.968 |
| instruction top-1 (chance 0.1) | 0.791 | 0.867 | 0.992 | 0.979 |
| instruction pairwise | 0.946 | 0.981 | 0.992 | 0.979 |
| sensitivity (median std of matches) | 0.160 | 0.234 | 0.236 | 0.277 |
| effective rank of demo effects | 11.141 | 13.966 | 14.055 | 10.624 |

TEXT_ONLY, SIGReg ×10

| quantity | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| P(demo > stay) | 0.876 | 0.953 | 0.969 | 0.947 |
| P(demo > reverse) | 0.936 | 0.977 | 0.977 | 0.958 |
| P(demo > other-task action) | 0.970 | 0.951 | 0.990 | 0.947 |
| instruction top-1 (chance 0.1) | 0.673 | 0.898 | 0.954 | 0.947 |
| instruction pairwise | 0.862 | 0.946 | 0.972 | 0.947 |
| sensitivity (median std of matches) | 0.062 | 0.077 | 0.089 | 0.101 |
| effective rank of demo effects | 10.598 | 12.954 | 15.905 | 9.692 |

### Over training (pooled): demo>stay / demo>other / top-1 / shared start / eff. rank effects

- TEXT_ONLY: 1000: 0.95/0.97/0.85/0.82/8.8; 2000: 0.98/0.97/0.89/0.92/13.4; 3000: 0.97/0.97/0.86/0.88/13.5; 4000: 0.95/0.97/0.87/0.86/14.6; 5000: 0.94/0.96/0.87/0.87/15.1
- TEXT_AND_OBS (q on o_t): 1000: 0.96/0.92/0.87/0.91/13.7; 2000: 0.96/0.93/0.88/0.90/15.3; 3000: 0.95/0.93/0.90/0.87/17.0; 4000: 0.94/0.92/0.90/0.87/17.7; 5000: 0.93/0.92/0.90/0.86/18.0
- TEXT_ONLY, SIGReg ×10: 1000: 0.94/0.96/0.83/0.40/8.6; 2000: 0.93/0.96/0.85/0.63/13.2; 3000: 0.92/0.97/0.85/0.59/14.8; 4000: 0.93/0.97/0.86/0.61/19.8; 5000: 0.93/0.97/0.85/0.66/20.7


## Findings

- Against the endpoint task contrast (`docs/stage_one_ab_task_contrast.md`, same evaluation after ISSUE-004), the first half of the episode is no longer at chance: first-quarter instruction top-1 0.76 (TEXT_ONLY) against 0.13, and shared-start discrimination 0.87 against 0.47–0.50.
- TEXT_ONLY and TEXT_AND_OBSERVATION reach similar pooled ranking; TEXT_AND_OBSERVATION has higher instruction top-1 (0.90 vs 0.87) and pairwise accuracy (0.97 vs 0.95) and lower first-quarter accuracy against stay, reverse and other-task actions (0.82–0.86 vs 0.92–0.96).
- SIGReg weight 10 raises the effective rank of ψ (31.5 vs 16.6) and of demo effects (20.7 vs 15.1) while lowering shared-start discrimination (0.66 vs 0.87) and the spread of candidate matches (0.08 vs 0.19).
- The learnable logit scale moved from 14.29 to 18.3–31.2 over 5000 steps.

# Stage-one diagnostics

`ceam/diagnostics/`, entry `scripts/diagnose_stage_one.py` (`--checkpoint <pt>` → `<run_dir>/diagnostics/step_XXXXXX/`, or `--random-init --config <json> --out-dir <dir>` for an untrained baseline). Computed on HELDOUT episodes in float32; the query uses each task's original text and, under `TEXT_AND_OBSERVATION`, the episode's first frame. Output: `report.json`, `progress_curves.png`, `subspace_overlap.png`.

| Report | File | Quantities |
|---|---|---|
| `StateDistributionReport` | `state_distribution_metrics.py` | SIGReg statistic (4096 frames), effective rank of cov(ψ), per-dimension std |
| `ProgressReport` | `progress_metrics.py` | Spearman(t, p_t) with p_t = ⟨z_t − z₀, n⟩ / ‖z_T − z₀‖; p_t curves |
| `DisplacementReport` | `displacement_metrics.py` | ‖z_T − z₀‖ / RMS projected distance of 256 cross-episode frame pairs; max path deviation / ‖z_T − z₀‖; goal error / ‖z_T − z₀‖ |
| `QueryReport` | `query_metrics.py` | subspace distance 1 − ‖U_aᵀU_b‖²/r for paraphrase and different-task pairs; task × task overlap matrix; endpoint distances same vs other task; task discrimination ‖U_aᵀ(ĝ_a − ψ_T^b)‖² / D_a for own, same-task and other-task endpoints, their ratio, and the fraction of other-task endpoints closer to ĝ_a than the own endpoint |

| `ActionRankingReport` | `action_ranking_metrics.py` | match m(z, l) = cos(z, q_l), m(0, l) = 0; HELDOUT anchors t = 0, k, 2k, … (k = `--horizon`, default 10); candidates: demo effect ψ(o_{t+k}) − ψ(o_t), stay 0, reverse ψ(o_{t−k}) − ψ(o_t), other tasks' k-step effects (at t = 0 every other-task first chunk, at t > 0 each other task's nearest frame in frozen features); per t/T quarter and pooled: P(demo > stay / reverse / other task) under the own instruction, instruction top-1 and pairwise accuracy of the demo effect, median std of candidate matches, effective rank of demo effects; shared-start P(demo > other-task first chunk); nearest-neighbour feature distances |
| `LinearProbeReport` | `linear_probe_metrics.py` | task-ID softmax regression (float64, features standardized on TRAIN, L2 1e-3 on W, full-batch LBFGS), fit on TRAIN final frames, scored on TRAIN and HELDOUT: from ψ(o_T), from U_aᵀψ(o_T) for each task a's basis, and from ψ(o₀) as a control |

Each array is stored per episode (or pair) with median, p10 and p90. The probe standardizes features, so it shows whether task information is present, not how large the separation is relative to the path; with 256-d features and 390 TRAIN episodes it fits TRAIN perfectly even on noise (HELDOUT on noise: 0.05–0.19 over 20 draws), so only its HELDOUT column is informative, with one episode = 2.3 points. The displacement ratio varies by ~9% between pair-sampling seeds.

## Reference values (LIBERO-Goal, TEXT_ONLY)

| Quantity (median) | Random init | Smoke, 200 steps |
|---|---|---|
| Effective rank of cov(ψ) | 33.5 | 10.8 |
| Progress Spearman | 0.71 | 0.51 |
| Path deviation | 1.10 | 0.018 |
| Goal error | 2.55 | 0.013 |
| Different-task subspace distance | 0.188 | 0.0076 |
| Task overlap, off-diagonal mean | 0.806 | 0.992 |
| Other-task / own goal distance ratio | 0.98 | 13.8 |
| Other-task endpoints closer to ĝ_a than own | 0.51 | 0.05 |
| Task-ID probe HELDOUT, ψ(o_T) | 1.000 | 0.977 |
| Task-ID probe HELDOUT, U_aᵀψ(o_T), mean over a | 0.793 | 0.928 |
| Task-ID probe HELDOUT, ψ(o₀) control | 0.070 | 0.070 |

Action ranking, k = 10, pooled over the four quarters (random init: all quantities at chance; shared start 0.460). A fixed per-task direction equal to the mean TRAIN demo effect on the random-init trunk reaches 0.72 over stay, 0.58 over other tasks, instruction top-1 0.20 and shared start 0.53. Instruction-contrast smoke checkpoint (200 steps): over stay 0.919, over reverse 0.955, over other tasks 0.934, instruction top-1 0.777 (0.536 in the first quarter), pairwise 0.900, shared start 0.474, effective rank of demo effects 5.0.

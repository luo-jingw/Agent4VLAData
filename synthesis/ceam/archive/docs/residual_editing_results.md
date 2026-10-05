# Bounded residual action editing

## Protocol

Twenty previously inspected continuation-diagnostic starts span ten LIBERO-Goal tasks. They are disjoint from
the old preference TRAIN branch sources, but are not an untouched final research test. Each edit replaces one
ten-step arm-command block and preserves its gripper schedule. Source-suffix execution uses the source horizon
and stops at first task success. Full-route EEF position/rotation and jerk come from simulator site boundaries.
No variable duration, semantic deletion or multi-fragment composition is evaluated.

There are 1,473 candidate executions and twenty repeated-original controls. One source has seven eligible donors;
all others have eight. Rank-6/12 residuals use one/two fixed DCT temporal modes for each arm coordinate. Each
donor has an antithetic pair per family; IID uses all ten modes. Requested arm-command RMS is .05. One scalar
per proposal scales its entire residual to preserve command bounds. Original-centered rank-12 edits have sixteen
proposals. The geometric command comparator replaces xyz increments by their mean; it preserves summed commanded
translation, not necessarily the physical path or endpoint.

Acceptance requires both original and candidate to succeed and pass the existing sampled physical screen,
completion steps no greater, translation at most original +1 mm, and RMS jerk at most original*1.05 +1e-6.
Steps must strictly decrease or translation decrease by more than 1 mm. All twenty originals succeed; eighteen
pass the physical screen. Original fallback for the other two preserves success without certifying feasibility.
Zero-tolerance jerk and time/path-only sensitivities are retained in the per-source CSV.

## Source-suffix candidate availability

| family | candidates | success | success and sampled feasibility | sources with a full-bank accepted edit /20 | mean steps with up to eight alternatives |
|---|---:|---:|---:|---:|---:|
| original | 20 | 100% | 90% | — | 110.450 |
| direct donor | 159 | 81.76% | 71.70% | 10 | 108.700 |
| donor + rank-6 residual | 318 | 77.67% | 67.61% | 11 | 109.472 |
| donor + rank-12 residual | 318 | 78.62% | 68.24% | 10 | 109.562 |
| donor + IID residual | 318 | 79.87% | 70.13% | 2 | 110.377 |
| original + rank-12 residual | 320 | 90.00% | 79.69% | 16 | 108.832 |
| geometric command averaging | 20 | 90.00% | 80.00% | 2 | 110.450 |

The final column includes the unchanged original and actual acceptance, averaging 256 deterministic subsets per
source. Geometric has only one alternative and one direct pool has seven. It does not measure learned selection
or policy SR. Rank-6, rank-12 and original-centered rank-12 save .905, .815 and 1.545 steps versus donor IID;
descriptive source-episode bootstrap intervals are [-1.652,-.276], [-1.564,-.220] and [-3.217,-.415]. These intervals
do not cover new task sampling or model training. Donor-centered low-rank edits do not exceed direct transfer's
mean saved steps in this grid. The original-centered comparison changes the reference as well as residual rank.

Realized residual RMS means are .04957, .04952, .04925 and .04956 respectively. No generated residual becomes zero.
Command-bound scaling introduces small magnitude differences; rank alone is not perfectly isolated by that rule.
Full-bank accepted-source counts use different family pool sizes and are not matched-budget evidence.

Without a jerk constraint, direct, rank-6, rank-12, IID and original-centered rank-12 each find a time/path
improvement at 17/20 starts. With zero permitted jerk increase the counts are 8, 6, 5, 2 and 10; with the primary
five-percent tolerance they are 10, 11, 10, 2 and 16. The low-frequency residual advantage over IID concerns the
joint quality criterion, not evidence of superior task semantics. Geometric command averaging gives 9, 1 and 2
under those respective acceptance rules. Neither this comparator nor the residuals remove redundant operations.

## Numerical and execution checks

All candidate banks regenerate exactly from source starts and the saved configuration. The original is retained,
commands are finite/in bounds, and gripper schedules are unchanged. Original repetition preserves all success
outcomes and step counts. Eighteen repeats have bit-identical full state traces; episode 292 differs by at most
8.671e-10 and episode 414 by 1.371e-13. Per-bank restore spreads and low-rank reconstruction errors are saved.

Artifacts: `outputs/residual_editing/pilot_v1/{proposals,branches,banks,reports}`. Tables and validation are
`reports/{availability.json,availability.md,per_source.csv,validation.json,selections.json}`. Commands, code
checksums, CPU timing and node identities are in `tmp/runlogs/residual_editing/cpu*`. Resource state is
`outputs/residual_editing/pilot_v1/resources.json`.

## Learned selection on new candidates

Both new models use identical TRAIN-only normalization, 120 old preference banks from forty TRAIN source
episodes, 2,000 updates, batch16, AdamW 3e-4 and weight decay 1e-4, with three episode-bootstrap seeds.
The direct MLP has 749,396 parameters and the dual matcher 747,265. Inputs are the same frozen visual/state,
task, original r/q and candidate-action features. Both begin at 10*cos and optimize BCE compatibility plus
logistic time/path preference loss. Neither new model uses the old representation-anchor penalty.
No query outcome enters training or checkpoint selection. Query scoring covers 28,075 within-start preference
pairs across twenty sources; those pairs are not independent observations.

| selector | query pair accuracy, three-seed mean | verified shortlist mean steps | steps versus random, episode-bootstrap 95% interval |
|---|---:|---:|---|
| frozen original score | 42.40% | 110.000 | +.151 [-.443,.811] |
| new direct scorer | 69.59% | 109.500 | -.349 [-.752,.026] |
| new dual matcher | 59.52% | 109.900 | +.051 [-.470,.722] |
| historical adapted preference matcher | 59.76% | 109.600 | -.249 [-.601,.080] |
| historical fixed-r preference matcher | 46.48% | 109.950 | +.101 [-.375,.722] |
| random shortlist | — | 109.849 | — |
| full candidate oracle | — | 107.850 | — |

Shortlists contain four alternatives plus original and use actual outcomes with the same jitter-aware
acceptance rule as round 1. Historical models are a supplemental zero-shot diagnostic using their saved
normalization, with no retraining. Six new checkpoint reloads exactly reproduce saved predictions. Their
results are in `reports/{ranking.json,ranking_selections.json,checkpoint_audit.json}` and `models/training.json`.

The direct architecture ranks these candidates better than the tested dual architecture, but its verified
shortlist improvement over random has an interval crossing zero. The historical adapted model has similar
pair accuracy to the new dual model, so the new grid's removed anchor alone does not explain the weak transfer.
Candidate distribution and source population differ from the earlier feedback study; their effects are not
separately identified. All models retain learned frozen action/query features, so this is not a comparison
between representation learning and a model without representations.

Without actual verification, new direct selectors succeed on 15/20, 17/20 and 19/20 starts; new dual selectors
on 15/20, 16/20 and 19/20, versus 20/20 for original. Selection permits predicted compatibility at least as large
as the original then maximizes predicted preference. These are branch-selection outcomes, not SR of a newly
trained action policy. The source-horizon and physical-screen limitations remain unchanged.

## Common-policy continuation

The source-verified family winners and unchanged originals form 71 distinct candidate indices. Each executes
under rendered source continuation and the same frozen no-psi policy; forty repeated-original controls bring
the grid to 182 executions. The selection manifest was frozen before policy outcomes. A family with no accepted
edit retains its original. Pool sizes differ by family, so this is a transfer diagnostic, not an equal-budget
generator ranking. No new policy is trained.

| source-selected family | edits /20 | source improvements | policy improvements | policy branch success /20 |
|---|---:|---:|---:|---:|
| unchanged original | 0 | 0 | 0 | 20 |
| direct donor | 10 | 10 | 3 | 18 |
| donor + rank-6 | 11 | 11 | 2 | 16 |
| donor + rank-12 | 10 | 10 | 1 | 18 |
| donor + IID | 2 | 2 | 1 | 20 |
| original + rank-12 | 16 | 16 | 3 | 18 |
| geometric command averaging | 2 | 2 | 1 | 20 |

All twenty CPU versus rendered original-source controls reproduce success and step counts. Maximum translation
and jerk differences are 1.157e-13 m and 6.097e-12 respectively. Rendered source winners reproduce all acceptance
counts. Sixty-four of 71 paired candidate-stage state traces are bit-identical. Episode 292/candidate59 differs
by .03994 in flattened simulator state; six episode414 pairs differ by at most 1.355e-11. Forty repeated originals
reproduce steps and success sequences, but episode292's policy repeat has full-state maximum difference 2.84025.
This is the unresolved restoration sensitivity in ISSUE-045; exact replay is not established for that source.

The sensitivity analysis excludes episodes292/414, leaving eighteen starts with exact paired candidate-stage
states and sixteen successful-feasible originals. Direct, rank-6, rank-12, IID, original-centered rank-12 and
geometric source improvements are 9, 10, 9, 1, 14 and 2; policy improvements are 2, 1, 0, 0, 2 and 1. Policy
success losses remain 2, 4, 2, 0, 2 and 0. Continuation dependence is not confined to discrepant restorations.
Full and sensitivity tables are in `reports/continuation.json`, per-source quality is in
`reports/continuation_per_source.json`, and raw traces are under `continuation/branches`.

Failure to transfer to this frozen policy does not invalidate a source-verified edited trajectory or prove that
a student trained on its regenerated observations would fail. It measures compatibility with that continuation.
Downstream label utility remains unmeasured for this editor.

## Jitter-target follow-up

A fourth exploratory round changes only the TRAIN preference-pair target: the preferred candidate must also
have jerk <= the other candidate*1.05 +1e-6. It retains 2,269 of 3,334 TRAIN pairs. Both architectures repeat all
three seeds with the same input normalization, initialization, episode-bootstrap samples and optimization.
The common query joint-quality mask has 16,611 pairs. The query set was already inspected before this mechanism
check. TRAIN jerk uses the earlier observable-pose measurement, so relative within-bank learning is assessed;
absolute jerk calibration across the two measurement pipelines is not established.

| model | time/path pair accuracy | joint-quality pair accuracy | verified shortlist mean steps |
|---|---:|---:|---:|
| direct, time/path target | 69.59% | 66.71% | 109.500 |
| direct, jitter-aware target | 66.14% | 64.86% | 109.783 |
| dual, time/path target | 59.52% | 57.40% | 109.900 |
| dual, jitter-aware target | 58.72% | 58.23% | 109.783 |

Means cover three training seeds. Jitter-aware minus time/path-only shortlist step differences are +.283
[-.233,1.050] for direct and -.117 [-.300,.017] for dual, using source-episode bootstrap intervals. Neither
target-aligned model establishes improvement over random. This constrained target change does not resolve
the observed selection limitation; other supervision designs and consistent new physical measurements remain
untested. Checkpoints and provenance are under `models/jitter`, predictions have `_jerk` in their names, and
the common `reports/ranking.json` retains all cells and both preference metrics.
All twelve newly trained checkpoints reload with zero maximum difference from their saved predictions.

## Decision and resource measurements

The four-round diagnostic is complete. Low-frequency residual proposals have measured local utility under
the jitter-aware source criterion, but learned shortlists have no established gain over random and source
improvements often fail to transfer to the tested policy. The declared downstream evidence gate is not met;
no refined dataset is published and no student-policy grid is launched. These results do not evaluate learned
residual generation, semantic phase editing, different action durations, multi-edit composition or new tasks.

Twelve managed jobs exited zero. Timed CPU collection intervals total .37583 process-hours. Reserved GPU intervals
total .12833 GPU-hours, including startup; peak sampled device memory is 1,869 MiB. GPU work used node-61, initially
card0 and then cards0/1 for disjoint continuation tasks. At most two project GPUs were used on any node. Final
inventory at 2026-10-04 02:45 EDT shows card0 at zero MiB and card1 at its four-MiB idle baseline. CPU collection
used node-61 and node-65; all collection workers exited. Artifacts occupy approximately 163 MB in logical file bytes.
Read-only report generation and small local audits are outside the timed worker intervals. The resource ledger
retains every job's timestamps, CPU times/RSS, GPU samples, exit status and command-log location.

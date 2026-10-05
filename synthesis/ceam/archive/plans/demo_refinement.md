# Demonstration refinement pilot

Plan Status: completed

## Problem

The Goal trained selector has 80.47% closed-loop SR against 92.67% for π mean. This experiment tests a different use
of the existing frozen r/q/m: offline local demonstration refinement, followed by ordinary behavior cloning.
Task compatibility and trajectory quality have distinct owners. Match scores rank search candidates; simulator
success and measured trajectory cost decide whether an edited demonstration is accepted.

The pilot edits three TRAIN demonstrations per Goal task, thirty source episodes total. Source selection uses
the existing episode split, seed 0, sorted episode ids, and no branch outcomes. Every other TRAIN episode and every
HELDOUT episode remain identical across conditions. The number of source demonstrations is identical across cells.
The legacy Goal evaluation CLI requires a stage-one/state table and rejects proposal caches that differ from that
stage one's training cache. Refined policies therefore need an explicit no-stage-one evaluation adapter.

## Structure

| Module | Responsibility | Owned state |
|---|---|---|
| Candidate preparation | Match raw demo, construct local action alternatives, execute same-start branches | Immutable per-episode candidate banks and CPU outcomes |
| Compatibility scoring | Frozen R6_all and focused q seed 0 score each candidate at one observed anchor | Candidate match vector |
| Selection | Same-size random or match-ranked shortlist; true success and quality select within it | Selected candidate ids |
| Re-rendering | Replay each selected full trajectory and encode its actual observations | RefinedEpisode feature/action arrays |
| Cache publication | Assemble three matching training views without modifying source caches | Feature/action caches and episode metadata |
| Policy comparison | Train and evaluate all nine cells using the existing no-ψ π | Checkpoints, closed-loop rows and aggregate report |

## Interfaces

- `DemoRefinementBank.save/load(Path)` owns raw source identity, fixture poses, initial state, anchor restore
  recipe, nine complete action sequences, branch successes, executed steps and arm total variation.
- `DemoCandidateScorer.score(FrameBatch, str, ndarray) -> ndarray` is a Protocol; `FrozenMatchDemoScorer` implements
  it using existing query and representation checkpoint loaders. It does not alter learned models.
- `DemoRefinementSelection.choose(bank, scores, condition, shortlist_size, seed) -> int` accepts an explicit
  `RefinementCondition`: original, random or guided. Original candidate 0 is always available.
- `RefinedEpisode.save/load(Path)` owns source episode/task identity, raw executed actions, rendered visual tokens
  and raw proprio, with exactly one observation preceding each stored action.
- `DemoRefinementCacheBuilder.build(condition) -> RefinementCacheSummary` publishes complete caches for the existing
  ProposalTrainer. Dataset-root output is an episode-metadata training view, not a complete LeRobot video release.
- `DemoRefinementPolicyEvaluator(DemoPolicyEvaluationConfig).run() -> tuple[ProposalMeanEpisodeResult, ...]` loads the proposal's own data and normalization
  through its stored config and writes `ProposalMeanEpisodeResult` rows with `ProposalMeanRunner`. It owns no r/q/m.

## Flow

CPU prepare → immutable candidate bank → GPU restore anchor and score → explicit selection → GPU replay and encode
the entire selected trajectory → publish caches → train π → dedicated Goal mean evaluator → paired comparison.

One ten-step anchor is at approximately one third of the original successful trajectory. Eight alternatives keep
the original gripper sequence: four local smoothing variants and four low-dimensional DCT perturbations. Candidate
0 is the original. Branches use identical raw initial state, fixture poses and ten settle steps. Each branch ends
at its first task success or the source episode's action horizon. All nine branches are measured to make the
shortlist comparison reusable and auditable; each editing condition sees four alternative candidates plus original.

Quality is lexicographic: fewer steps to success, then lower total variation of ActionCache-standardized arm commands.
Only successful candidates with strictly lower cost than original may replace it. Guided uses the top four match
scores; random uses four candidates without replacement from the same eight, seed 0. Quality is measured by actual
execution and is not inferred from match. The optimizer can retain original unchanged when no candidate improves it.

The original control re-renders the same thirty source trajectories under the same first-success convention.
The legacy unedited cache was encoded from 224-pixel images; all three conditions re-render their selected source
episodes at 512 pixels. This observation-source change is shared by the original control.
Unedited episodes use the same source features/actions across cells. The original source proprio/action statistics
and instruction token tables remain fixed for every cell. Episode ids and the TRAIN/HELDOUT split remain unchanged.
Changing an action never reuses the source's subsequent observations in that edited episode.

## Code Mapping

| Module / interface | Files |
|---|---|
| Bank | `ceam/refinement/demo_bank.py` |
| Preparation | `ceam/refinement/demo_preparation.py`, `scripts/prepare_demo_refinement.py` |
| Selection | `ceam/refinement/demo_selection.py` |
| Scorer interface / implementation | `ceam/refinement/demo_candidate_scorer.py`, `ceam/refinement/frozen_match_demo_scorer.py` |
| Rendered episode | `ceam/refinement/refined_episode.py` |
| Renderer | `ceam/refinement/demo_renderer.py`, `scripts/render_demo_refinement.py` |
| Caches | `ceam/refinement/demo_cache_builder.py`, `scripts/build_demo_refinement_caches.py` |
| Training / evaluation | Generated `configs/proposal/demo_refinement_*_seed*.json`, existing training/evaluation CLIs |
| Paired report | `ceam/refinement/policy_comparison.py`, `scripts/report_demo_refinement.py` |
| Policy evaluation adapter | `ceam/refinement/policy_evaluation.py`, `scripts/evaluate_demo_refinement_policy.py` |
| Evaluation config | `ceam/refinement/policy_evaluation_config.py` |
| Outcomes | `outputs/demo_refinement/`, `docs/demo_refinement_results.md` after measurement |

## Tasks

1. Define and validate bank preparation/selection with CPU trajectories and a synthetic same-start selection check.
2. Score and re-render one real source episode; report restore error, branch outcomes, chosen ids, frame shape and time.
3. Collect all thirty source episodes, publish all three caches and verify episode/action/frame alignment and unchanged
   heldout identity. Short smoke training validates existing trainer compatibility before full grid execution.
4. Run the complete policy grid and its closed-loop evaluation. No cell is selected or omitted based on its result.
   The evaluator uses each proposal's own cache and is validated with a known Goal baseline checkpoint and the
   compatibility-smoke checkpoint before formal policy evaluation.
5. Report every cell, paired outcomes, task breakdown, source replacement rates, cost distributions and resource use.

## Experiment Grid

| Cell | Demonstration condition | π seed | Updates | Closed-loop episodes |
|---|---|---:|---:|---:|
| original-0 | re-rendered original | 0 | 3,000 | 500 |
| original-1 | re-rendered original | 1 | 3,000 | 500 |
| original-2 | re-rendered original | 2 | 3,000 | 500 |
| random-0 | random shortlist refinement | 0 | 3,000 | 500 |
| random-1 | random shortlist refinement | 1 | 3,000 | 500 |
| random-2 | random shortlist refinement | 2 | 3,000 | 500 |
| guided-0 | r/q/m shortlist refinement | 0 | 3,000 | 500 |
| guided-1 | r/q/m shortlist refinement | 1 | 3,000 | 500 |
| guided-2 | r/q/m shortlist refinement | 2 | 3,000 | 500 |

All cells use the Goal no-ψ π architecture, instruction tokens, batch 256, existing 3,000-update optimizer schedule,
bf16, clipping and IID sampling. Evaluate task × benchmark inits 0–49, environment seed 0, ten settle steps and
300-step horizon. The seed-paired difference is the primary comparison; no test split selects source demos or models.
Before the episode sequence, reset and encode the first init once, matching the existing Goal baseline's
`--check-start-observation` reset sequence without requiring its unrelated stage-one/state table.
Benchmark inits 40–49 are also reported separately because the frozen focused q was trained from inits 0–39.

## Resources and continuation

Every node uses at most two GPUs. CPU branch preparation uses no GPU. Measured throughput, disk, memory and completed cells are recorded in
`docs/demo_refinement_results.md` and `outputs/demo_refinement/pilot_v1/reports/`. Source caches/raw data are read-only;
all outputs use a separate version. LIBERO-90 PCA/R6 and next-representation design remain pending after this pilot.
The pilot tests the current frozen implementation; it does not train a new representation or prove a general
demonstration-refinement advantage from trajectory cost alone.

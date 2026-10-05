# Stage one: final configuration

Stage one learns the task-agnostic state representation ψ(o) (`StateTrunk`) and the instruction query (`QueryModule`) from frozen DINOv2 / SigLIP features of LIBERO-Goal demonstrations. The configuration below is fixed; later stages build on it without adding constraints or modules to stage one.

| Setting | Value |
|---|---|
| Config | `configs/stage_one/goal_instruction_contrast.json` |
| Model | `outputs/runs/goal_instruction_contrast/checkpoints/step_005000.pt` |
| Query conditioning | TEXT_ONLY (q depends on the instruction only) |
| Match | m(z, l) = cos(z, q_l), z = ψ(o′) − ψ(o); m(0, l) = 0; ĝ is not used in the match |
| Losses and weights | SIGReg 1, path 1, goal 1, endpoint 1, paraphrase 1 (subspace + direction), instruction contrast 1 |
| logit_scale | learnable, init log(1/0.07), shared AdamW learning rate (moves to s = 18.3 in 5000 steps) |
| Training | 5000 steps, B = 64, M = 8, lr 3e-4 cosine, warmup 250, bf16 autocast |

## Decisions

- **Match on action effects, not on distance to a goal.** Distance improvement toward ĝ was near zero far from the goal, which left the first half of every episode at chance (`docs/stage_one_ab_task_contrast.md`, action-level evaluation). The direction query does not depend on the distance to the goal.
- **Instruction contrast instead of an endpoint contrast.** Without any contrast the query subspace became instruction-independent; the endpoint contrast fixed task discrimination only near the end of episodes. The instruction contrast on k-step effects (k uniform to the episode end) gives action-level ranking of 0.92–0.98 in every quarter of the episode, shared-start discrimination 0.87 and instruction top-1 0.87 (observed outcomes standing in for E; `docs/stage_one_instruction_contrast.md`).
- **Text-only query.** Querying with the current frame gave similar ranking on LIBERO-Goal and trains 2.3× slower; state-dependent queries are deferred to data that tests reference resolution.
- **SIGReg weight 1.** Weight 10 raises the effective rank of ψ but lowers shared-start discrimination to 0.66 and halves the spread of candidate matches.
- **LIBERO-Spatial is not evaluated in stage one.** All Spatial tasks share one goal predicate, so the instruction contrast is 0 there and its results would reflect scene discrimination only (ISSUE-003).
- **Evaluation criterion.** Progress and progress-curve shape are auxiliary observations; decisions use action-level ranking. The final criterion of the framework is closed-loop task success.

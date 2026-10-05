# Agent4VLAData reference review

## Snapshot

Source: [Agent4VLAData](https://github.com/luo-jingw/Agent4VLAData).
The public main-branch archive was downloaded anonymously, without credentials, into
`tmp/reference/agent4vla-main.zip`. SHA256:
`2ed2392c7af02fcdfeb1605d646228e1ed9016f9fb6d70fb16380f1f97925c31`.
Extracted reference: `tmp/reference/Agent4VLAData-main/`. The reference repository was not modified.

The snapshot README and `docs/reports/top_level_view.md` describe a 58-paper research stage. The current conceptual
entry is `idea/conditional-action-representation/README.md`; older design material lives in `design/history/`.
Some metadata-count text and gap entries still describe earlier survey scopes. This review does not independently
verify all cited papers, literature novelty claims or externally reported numerical results.

## Conceptual mapping

The conditional action representation note defines task-independent `r=f(o,a)`, conditional representation
distributions, goal-conditioned match queries, an embodiment-specific decoder and future-aware representation
learning targets. It explicitly permits multiple valid action modes and does not guarantee optimality.

CEAM's `docs/framework.md` keeps the task-independent representation and task-conditioned query/match split.
The retrieval pilot implements its replaceable candidate sampler using observed donor actions and leaves r/q/m
frozen. It does not require a latent decoder or a learned conditional distribution. Actual same-start execution
measures whether a retrieved action remains compatible after transfer.

The survey separates task legality from execution efficiency and registers observation gaps and state aliasing.
Those distinctions are reflected in the pilot's success measurement, conditional physical motion metrics and
same-start alternatives. The reference's numerical evidence belongs to its stated source setups; the measured
CEAM results are recorded separately in `docs/retrieval_refinement_results.md`.

## Verification boundaries

Task-independent encoding is a module boundary, not a theorem that conditioning an encoder always causes collapse.
Choosing argmax over match scores still uses the full representation to compute scores; it does not by itself
turn the representation into a one-dimensional latent variable. Sampling versus argmax concerns the search policy.

Missing physical information cannot be recovered merely by assigning it to the query's responsibility. A
deterministic representation with identical inputs produces the same output; a candidate-dependent query needs
enough contextual information and usable action distinctions for ranking. CEAM's current frozen matcher does not
implement a general distribution over hidden physical states.

The local pilot measures single replacement blocks and fixed-suffix outcomes. It establishes neither novel
literature coverage nor globally superior strategies. Compatibility ranking, successful-branch quality preference
and downstream policy learning remain separate validation questions.

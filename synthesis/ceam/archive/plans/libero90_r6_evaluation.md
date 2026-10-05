# LIBERO-90 R6 evaluation interfaces

Plan Status: proposed

## Problem

The existing representation CLI requires every model to name one common feature/action cache. Goal R6_all was
trained with Goal normalization; the LIBERO-90 baseline uses LIBERO-90 normalization. Their transfer comparison
requires a model-owned normalization contract. `SceneLabelTable.read` also requires every shard to share object
and fixture names. The 89-task counterfactual store fails that check on a cabinet task. Whole-store scene-label
concatenation therefore cannot implement the required multi-scene physical readouts.

PCA fitting on TRAIN frames and the 70,000-update effect-head recipe remain the intended baseline. The revised
interfaces must cover their complete evaluation before training starts.

## Structure

| Module | Responsibility | Owned state |
|---|---|---|
| Native input normalization | Convert raw store proprio/actions using each checkpoint's recorded training statistics | One immutable statistics pair per model |
| Topology scene labels | Read sidecars in groups with equal object/fixture names and retain global frame identity | Typed scene groups and global-to-group row maps |
| Evaluation identity | Associate episode/shard provenance with benchmark task and scene | `LiberoEpisodeMetadata` task records |
| Transfer evaluation | Evaluate each model in its native coordinates on identical raw transitions | Prediction and within-state result rows |
| PCA transfer variance | Measure old and refitted projections on the same centered LIBERO-90 TRAIN features | Variance-capture report |

## Interface

- `RepresentationInputNormalizer` is a Protocol exposing `frames(raw_visual, raw_proprio, device) -> FrameBatch`
  and `actions(raw_chunks, device) -> Tensor`. `CheckpointInputNormalizer` loads the checkpoint's named caches.
- `TopologySceneGroup` is an immutable dataclass containing global frame ids, object/fixture identities and the
  existing physical label columns. `TopologySceneLabels.read(store) -> tuple[TopologySceneGroup, ...]` groups
  equal topology; each sidecar row is represented once. Empty fixture arrays remain explicit.
- `LiberoEpisodeMetadata` owns the 90 benchmark-task identities and scene membership. Equal instruction text does
  not establish equal task identity. Store-to-task attribution must be checked against writer provenance.
- `Libero90RepresentationEvaluator` receives models, their native normalizers, raw transition sets and explicit
  task/scene identities. Each model's target is its own encoded future minus current. Native MSE and MSE divided
  by true-effect energy are reported separately; different PCA coordinate systems are not treated as one space.
- `PcaTransferVariance` reports captured centered variance divided by total variance for both projection matrices
  on exactly the refit TRAIN frame set. Projection-origin reconstruction error is a separate quantity.

## Flow

Validate store episode/provenance and split meanings → validate topology-group and native-normalization interfaces
→ define six-source training config → fit PCA and reference variance → short resource observation → train effect
head for 70,000 updates → native prediction and within-state diagnostics → task and scene reports.

## Code Mapping

| Mechanism | Files |
|---|---|
| Normalizer interface / implementation | `ceam/diagnostics/representation_input_normalizer.py`, `ceam/diagnostics/checkpoint_input_normalizer.py` |
| Typed topology groups / reader | `ceam/diagnostics/topology_scene_group.py`, `ceam/diagnostics/topology_scene_labels.py` |
| Existing identity owner | `ceam/data/libero_episode_metadata.py` |
| Dedicated transfer evaluator | `ceam/diagnostics/libero90_representation_evaluator.py`, `scripts/evaluate_libero90_representation.py` |
| PCA variance | `ceam/diagnostics/pca_transfer_variance.py`, `scripts/report_pca_transfer_variance.py` |
| Existing fit/train mechanisms | `scripts/fit_feature_pca.py`, `scripts/train_interaction.py` |
| Baseline config / measured evidence | `configs/interaction/libero90_frozen_pca_delta.json`, `docs/libero90_prep.md` |

The existing Goal evaluation CLI retains its common-cache contract. The new transfer evaluator has an explicit
model-native contract rather than weakening that check.

## Tasks

1. Verify per-store episode meanings, task attribution, TRAIN/HELDOUT separation and frame counts. Observation:
   per-source split counts and task/scene coverage, including the task without retained demonstrations.
2. Implement native normalizers and topology readers. Observation: array shapes, native normalization error versus
   existing loaders, sidecar row coverage and successful loading of different/zero-fixture topologies.
3. Define the six training sources and fixed PCA/refit configuration. Observation: source loading, memory estimate
   from actual file sizes, PCA TRAIN frame identity, projected variance and numerical self-checks.
4. Run a short effect-head resource observation, then the fixed 70,000-update baseline. Observation: throughput,
   memory, checkpoint step and prediction values.
5. Measure transfer and refit prediction plus smooth-random and held-out π-noise distinguishability. Observation:
   within-state variance, effective rank, cosine, centered cosine, Spearman and state-conditioned R², with task and
   scene counts. Cross-validation groups use task/episode/state identities, never instruction text alone.

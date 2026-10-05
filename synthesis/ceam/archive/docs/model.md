# Model

## StateTrunk (`ceam/model/state_trunk.py`)

`FrameBatch` with arbitrary leading dims → ψ [..., state_dim], unnormalized. Sequence per frame: [readout token, proprio token, 2 × 64 visual tokens]; visual tokens get a linear projection plus learned view and position embeddings. Pre-norm Transformer encoder (width 512, depth 4, 8 heads, FFN 4×width, GELU, no dropout), then LayerNorm and a linear head on the readout token. Raises `ValueError` if the leading dims of `visual_tokens` and `proprio` differ.

## QueryModule (`ceam/model/query_module.py`)

`TextTokens` (and a reference `FrameBatch` under `TEXT_AND_OBSERVATION`) → `QueryOutput(basis U [B, d, r], goal ĝ [B, d])`. rank + 2 learned query tokens pass through a pre-norm Transformer decoder (width 512, depth 2) over the context: projected text tokens with position embeddings and padding mask, plus projected reference visual tokens under `TEXT_AND_OBSERVATION` (reference proprio unused). The first r outputs form a [d, r] matrix; U is its float32 reduced QR factor, computed outside autocast. Output r gives ĝ and output r + 1 the match direction q (`QueryOutput.direction`); `QueryOutput.logit_scale` is the module's learnable scalar log s, initialized to log(1/0.07), used to scale cosine matches in contrastive terms. A reference given under `TEXT_ONLY`, or missing under `TEXT_AND_OBSERVATION`, raises `ValueError`.

Under bf16 autocast, `basis` is float32 while ψ and `goal` are bf16.

## Measured

`scripts/inspect_model.py`, d = 256, r = 8:

| Quantity | Value |
|---|---|
| StateTrunk parameters | 13,175,040 |
| QueryModule parameters | 9,433,345 (TEXT_ONLY), 9,860,865 (TEXT_AND_OBSERVATION) |
| max \|UᵀU − I\| | 7.7e-7 (TEXT_ONLY), 7.2e-7 (TEXT_AND_OBSERVATION) |
| StateTrunk fwd+bwd, 704 frames, peak memory | 13.9 GB fp32, 7.8 GB bf16 autocast |
| StateTrunk fwd+bwd, 704 frames, latency | 58–118 ms bf16, 267–558 ms fp32 (GPU shared, indicative) |

Gradients reach the query parameters through both `basis` and `goal`. Padded text positions do not affect the output.

## DynamicsHead (`ceam/model/dynamics_head.py`)

Δ(o, a) for E(o, a) = ψ(o) + Δ(o, a). Inputs: current `FrameBatch`, ψ(o) [..., 256], standardized action chunk [..., 10, 7]; leading dims must agree (`ValueError` otherwise). Sequence of 141 tokens: readout, ψ, proprio, 2 × 64 visual (projection + view + position embeddings), 10 actions (projection + step embedding); 4 pre-norm Transformer encoder layers (width 512, 8 heads, FFN 4×width); LayerNorm and a linear head on the readout. The head's weight and bias start at zero, so an untrained model predicts Δ = 0 (E = ψ) and only the head receives gradient on the first step. 13,315,840 parameters; batch 256 forward+backward 52 ms / 3.2 GiB under bf16 autocast, 160 ms / 5.6 GiB in fp32 (shared H100).

# Frozen encoders

Both encoders load from the local Hugging Face snapshot (`snapshot_download(..., local_files_only=True)`), run in float32 under `torch.inference_mode()` with every parameter frozen, and return float16.

| Encoder | File | Input | Output |
|---|---|---|---|
| `DinoV2VisionEncoder` (`facebook/dinov2-base`) | `ceam/encoders/dinov2_vision_encoder.py` | uint8 RGB [N, H, W, 3] | [N, 64, 768] |
| `SiglipTextEncoder` (`google/siglip-so400m-patch14-384`, text tower) | `ceam/encoders/siglip_text_encoder.py` | strings | `TextTokens`: tokens [S, 64, 1152], mask [S, 64] |

Vision preprocessing: bicubic antialiased resize to 224×224, scale to [0, 1], ImageNet mean/std. `encode` disables cuDNN TF32 inside a `torch.backends.cudnn.flags` context and restores the previous setting afterwards: with PyTorch's default `cudnn.allow_tf32=True`, the patch-embedding convolution runs in TF32 and the same frame encoded alone or inside a batch of 64 differs by up to 0.71 absolute (1.2e-2 relative Frobenius error). The 16×16 patch tokens of `last_hidden_state` (CLS dropped) are average-pooled 2×2 to 8×8 and flattened row-major.

Text: padded and truncated to 64 tokens and encoded from `input_ids` without an attention mask, as SigLIP was trained. `mask` is `input_ids != pad_token_id`; the pad token is `</s>`, so the end-of-sequence token is excluded from the mask.

## Decision: SigLIP tokenizer from `tokenizer.json`

The slow `SiglipTokenizer` needs `sentencepiece`, which the shared `lerobot` environment does not have and which is not installed there. The encoder loads the repository's `tokenizer.json` through `PreTrainedTokenizerFast`. Compared with `spiece.model`, the 32000 pieces and scores, the NFKC precompiled charsmap and byte fallback are identical, and the normalizer applies lowercase → punctuation removal → whitespace collapse → strip, then appends `</s>`. One difference: `tokenizer.json` does not remove `/`, `<` or `>`, while the slow tokenizer removes all of `string.punctuation`. No instruction string in `data/instructions/` contains these characters.

## Measured

`scripts/inspect_encoders.py`, GPU shared with another workload at ~97% utilization, so latencies are upper bounds:

| Quantity | Vision | Text |
|---|---|---|
| Parameters (frozen) | 86.6 M | 449.7 M |
| Repeat-encode max abs diff | 0.0 | 0.0 |
| Latency | batch 64 frames: 97–215 ms | 2 sentences: 13–29 ms |
| Peak memory above load | 1074.6 MiB at batch 64 | — |

Outputs are bitwise reproducible across processes. With TF32 disabled, 16 frames of LIBERO-Goal episode 5 encoded alone versus inside a batch of 64 differ by at most 0.0078 absolute and 5.7e-5 relative Frobenius error (agent and wrist views), from fp32 kernel selection depending on batch size.

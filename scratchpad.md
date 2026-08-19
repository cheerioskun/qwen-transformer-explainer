# Project scratchpad

Working memory for the Qwen Transformer Explainer journey. Update this at the end of each session. Keep durable decisions, verified facts, open questions, and the next concrete step here; polished teaching belongs in the explainers.

## Purpose

Learn transformer inference from two connected views:

1. **Model view:** follow values through a readable Qwen3 implementation and expose intermediate tensors.
2. **Engine view:** connect those operations to how an inference engine stores weights, builds compute graphs, caches keys and values, and runs kernels.

This repository owns the model-view explainer. The `llama.cpp` experiments remain a separate project.

## Ground rules

- Introduce a term before relying on it.
- Label illustrations as illustrations; do not present invented values as model outputs.
- Prefer measurements and tensors from a real pinned model.
- Build one understandable vertical slice before broad model support.
- Keep the original GPT-2 explainer runnable while we learn from it.

## Baseline

- Fork: `cheerioskun/qwen-transformer-explainer`
- Upstream: `poloclub/transformer-explainer`
- Upstream revision at fork: `bfe50afba10b9b560b84143ee1107d977defa74f`
- Original execution: a custom instrumented GPT-2 implementation exported to ONNX; ONNX Runtime Web runs it in the browser and returns attention tensors plus final logits.
- Original browser model: roughly 626 MB split into 63 ten-megabyte chunks.
- Local build repair: Vite 6.4.x is required by the current Svelte plugin; Sass also needs the project root in `loadPaths`.
- `npm run build` passes. `npm run check` does not: the upstream code currently reports many existing TypeScript errors.

## Verified Qwen3-0.6B facts

Source: `Qwen/Qwen3-0.6B` config at revision `c1899de289a04d12100db370d81485cdf75e47ca`.

- 28 decoder blocks
- hidden size: 1,024
- MLP intermediate size: 3,072
- 16 query heads, 8 key/value heads
- head dimension: 128
- vocabulary: 151,936 tokens
- context limit in config: 40,960 positions
- RMSNorm with epsilon `1e-6`
- rotary position embeddings (RoPE), base frequency `1,000,000`
- SiLU-gated MLP (commonly described as SwiGLU)
- input embedding and output projection weights are tied

Measured with `llama-tokenize` using the BF16 GGUF:

```text
The capital of France is
785   -> 'The'
6722  -> ' capital'
315   -> ' of'
9625  -> ' France'
374   -> ' is'
```

## Important architecture gap

Adapting the UI is not a model-name replacement. GPT-2 uses learned positional embeddings, LayerNorm, ordinary multi-head attention, and a GELU MLP. Qwen3 uses RoPE, RMSNorm, grouped-query attention, and a gated SiLU MLP. The visual grammar and exported tensor names must change accordingly.

## Proposed learning sequence

1. One next-token prediction: establish vocabulary and tensor shapes.
2. Tokenization and embeddings.
3. One Qwen3 decoder block in readable Python.
4. Causal self-attention, then grouped-query attention.
5. Residual stream, RMSNorm, and SwiGLU.
6. Logits, softmax, temperature, top-k/top-p, and sampling.
7. Autoregressive decoding and the KV cache.
8. Prefill versus decode.
9. Match selected tensors and logits against Hugging Face Qwen3.
10. Connect the readable implementation to GGUF and `llama.cpp` execution.

## Session log

### 2026-08-13 — fork and baseline

- Forked and cloned the upstream explainer.
- Inspected its model export and browser inference path.
- Repaired the clean install/build dependency mismatch without using legacy peer resolution.
- Confirmed the development server serves the page and the first ONNX chunk.
- Started lesson 1 as a standalone, self-contained HTML explainer.

## Next step

Read lesson 1, then implement a tiny script that loads the official Qwen3 weights and captures the exact tensors involved in a single next-token prediction. Do not add generation or a KV cache yet.

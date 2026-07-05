# How vramwatch attributes VRAM

This document explains exactly how vramwatch turns two data sources — the GPU
driver and the inference loader — into the weights/KV/other breakdown, what is
measured versus estimated, and where the numbers can be wrong. If you only read
one thing: **the KV-cache formula is exact for an unquantized cache (f16/bf16/f32)
and a small, deliberately-conservative over-estimate for a quantized one;
everything else in the split is a best-effort estimate, and the output labels it
as such.**

## The two inputs

For each GPU, vramwatch gathers:

1. **Driver truth** — from `nvidia-smi` or `rocm-smi`:
   - total / used / free VRAM for the device,
   - per-process VRAM: NVIDIA via `nvidia-smi --query-compute-apps`; AMD/Intel via
     the kernel’s `/proc/<pid>/fdinfo` DRM interface on Linux (deduplicated by DRM
     client id and mapped to a device by PCI address).

2. **Loader truth** — which models are resident and their shape:
   - **Ollama**: `GET /api/ps` gives each model’s name and `size_vram`; `POST
     /api/show` gives `model_info` (architecture: `block_count`,
     `attention.head_count`, `attention.head_count_kv`, `attention.key_length`,
     `embedding_length`, `context_length`).
   - **llama.cpp**: `GET /props` gives the running context (`n_ctx`) and the model
     file path. vramwatch then reads the **GGUF file header** directly (no tensor
     data) to recover the same architecture fields and the file size.

## The KV-cache formula

The attention KV cache grows linearly with context. Per token it costs:

```
KV bytes/token = 2 · n_layers · n_kv_heads · head_dim · (kv_bits / 8)
```

- `2` — one Key tensor and one Value tensor.
- `n_kv_heads` — the number of **key/value** heads. For grouped-query attention
  (GQA) this is smaller than the number of query heads, which is exactly why a
  modern 70B model has a much smaller KV cache than its size suggests. For
  multi-head attention (MHA), `n_kv_heads == n_heads`.
- `head_dim` — per-head dimension (`key_length` if the model reports it, else
  `embedding_length / n_heads`).
- `kv_bits` — bits per cache element: 16 for f16/bf16 (the default), 32 for f32.
  Block-quantized caches also store a per-block f16 scale (and an f16 min for the
  `_1` variants) over 32 elements, so their true cost is higher than the nominal
  width: `q8_0` ≈ 8.5, `q5_0` ≈ 5.5, `q5_1` = 6, `q4_0` ≈ 4.5, `q4_1` = 5 bits.
  `--kv-cache-type` rounds these **up** (q8_0→9, q5→6, q4→5) so the estimate is
  conservative — an OOM predictor should never under-count.

### Worked example — Llama-3-8B at 8k context, f16 cache

`n_layers = 32`, `n_kv_heads = 8` (GQA), `head_dim = 128`, `kv_bits = 16`:

```
KV/token = 2 · 32 · 8 · 128 · (16/8) = 131,072 bytes = 128 KiB
KV @ 8192 = 128 KiB · 8192 = 1.00 GiB
```

Declare a `q8_0` cache (`--kv-cache-type q8_0`, modelled at 9 bits) and the estimate
drops to ~576 MiB; `q4_0` (5 bits) to ~320 MiB — versus 1 GiB at f16. That’s the
difference between “fits” and “OOM”, which is why declaring the dtype matters.

## Attribution: tiling the device

Given the footprint of the inference process on a device, vramwatch splits it into
`weights`, `KV cache`, and `compute` (activations, scratch, the CUDA/HIP context),
then adds `other apps` (everything else on the device) and `free`. **The five
segments always sum exactly to the device total.**

The footprint is chosen in priority order:

1. per-process driver VRAM for the inference process — matched first by **PID**,
   then by **process name** (e.g. an `ollama` / `llama-server` process). This uses
   the *real* resident VRAM, including runtime overhead the loader doesn’t report.
2. the loader’s reported VRAM (`size_vram` for Ollama), else
3. `weights + KV` derived from the GGUF file and the formula (llama.cpp).

Then:

- **KV cache** is the loader-reported value if present, otherwise the formula
  estimate (labelled `estimated`).
- **Weights** are loader-reported if present (they win any conflict with an
  estimated KV), otherwise `footprint − KV` (Ollama) or the GGUF file size
  (llama.cpp).
- **Compute** is whatever footprint remains after weights and KV.
- **Other apps** is `device used − inference footprint`.
- **Free** is `device total − device used`.

### Guardrails

- Reported (ground-truth) figures always win over estimated ones — an
  over-estimated KV can never shrink a loader-reported weights value.
- An estimated KV cache can never claim the entire footprint (weights must be
  resident too); if the estimate exceeds the footprint — which usually means the
  cache is quantized but wasn’t declared — vramwatch caps it and prints a warning.
- All arithmetic is unsigned and clamped, so the segments never underflow or
  exceed the device total.

## Prediction

Because KV grows linearly, the maximum context that still fits is:

```
max_context ≈ current_context + (free_VRAM ÷ KV_bytes_per_token)
```

capped at the model’s trained context length. `predict --context N` answers the
inverse — whether a target `N` fits — by holding weights and compute constant and
scaling only the KV cache:

```
needed(N) = weights + compute + KV/token · N
fits      = needed(N) ≤ 98% of device total
```

If `N` exceeds the model’s trained context, vramwatch says so even when it fits in
VRAM.

## What’s measured vs. estimated

| Figure | Source | Trust |
|--------|--------|-------|
| Device total / used / free | driver | measured |
| Per-process VRAM (NVIDIA) | driver | measured |
| KV cache | formula (`arch × ctx × dtype`) | estimated — exact at f16/bf16/f32, conservative (rounded up) for quantized |
| Weights (Ollama) | `footprint − KV` | estimated |
| Weights (llama.cpp) | GGUF file size | estimated (assumes full offload) |
| Compute overhead | footprint remainder | estimated |
| Max context before OOM | `free ÷ KV/token` | estimated, linear |

## Known sources of error

- **Undeclared quantized KV cache** → KV over-estimated (and weights
  correspondingly under-estimated). Fix: `--kv-cache-type`.
- **Partial GPU offload in llama.cpp** → GGUF file size over-states VRAM weights.
- **AMD**: no per-process VRAM, so if another process shares the GPU the “other
  apps” bucket may absorb some of the model’s footprint or vice-versa.
- **Flash-attention / paged KV** implementations may allocate the KV cache in
  blocks; the formula gives the logical size, which can differ from the physical
  reservation by a small margin.

vramwatch aims to be *useful and honest*, not a substitute for an allocator-level
profiler. When in doubt, the `estimated` label is telling you the truth.

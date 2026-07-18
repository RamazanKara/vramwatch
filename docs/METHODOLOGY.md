# Methodology

vramwatch has two related engines:

1. `fit` predicts a model before it is loaded.
2. `watch` attributes a model that is already resident.

Both use the same GGUF architecture fields and the same provenance vocabulary,
but they answer different questions. This document describes the arithmetic and
where uncertainty enters.

## Provenance

Every important value is assigned one of five sources:

| Badge | JSON value | Meaning |
|---|---|---|
| `[M]` | `measured` | sampled from a driver or OS counter |
| `[R]` | `loader_reported` | returned by a loader API |
| `[E]` | `model_estimated` | derived from metadata or other observations |
| `[A]` | `assumed` | conservative vramwatch policy |
| `[U]` | `user_supplied` | supplied explicitly, such as `--vram` |

A mathematically derived value is not relabelled “measured” merely because its
inputs were measured.

## Preflight metadata resolution

`fit` needs two facts without loading model tensors:

- the exact byte size of the selected GGUF artifact or shard set; and
- architecture fields sufficient to estimate the KV cache.

For a local GGUF, vramwatch reads metadata from the file header and uses its file
size. For Hugging Face it reads the model API (`?blobs=true`), selects a GGUF by
`--quant` or `--file`, sums every shard, then reads a bounded ranged response from
the first shard. For Ollama it reads the OCI manifest and config, sums
model/projector layers, then ranges the model blob. A direct HTTPS URL is ranged
and must expose its complete size through `Content-Range` or an equivalent
linked-size header.

Remote metadata has a hard 16 MiB budget, and the response is closed as soon as
the architecture fields are parsed. If a server ignores Range and would force a
larger response, the request is stopped. Unknown file sizes, incomplete shard sets, ambiguous
files, malformed GGUF headers, and incomplete architectures fail closed; none can
turn into a zero-byte optimistic prediction.

The architecture fields are:

- transformer block count;
- KV-head count (falling back to attention-head count for MHA);
- key-head dimension (`attention.key_length`, else embedding/head count);
- value-head dimension (`attention.value_length`, else key dimension); and
- trained context length, when present.

`general.file_type` is used to identify common GGUF quantizations. Selection also
recognizes quantization names in filenames. The selected artifact's actual byte
size drives weight residency; vramwatch does not estimate weights from parameter
count and nominal bits.

## KV-cache estimate

For a full attention cache with one sequence:

```text
KV elements = context × layers × KV heads × (key dimension + value dimension)
KV bytes    = KV elements × element width
```

This handles grouped-query/multi-query attention and models whose key and value
dimensions differ. If no value dimension is present, it reduces to the familiar:

```text
2 × context × layers × KV heads × head dimension × element width
```

Widths used by preflight prediction are exact rational values for the common GGML
block formats:

| Cache type | Effective bits/element |
|---|---:|
| f32 | 32 |
| f16 / bf16 | 16 |
| q8_0 | 8.5 |
| q5_0 | 5.5 |
| q5_1 | 6 |
| q4_0 | 4.5 |
| q4_1 | 5 |

The half-bit overhead is the per-block scale (and, for `_1`, minimum) amortized
over 32 values. Live watch stores an integer bit width in its architecture model,
so quantized cache widths are rounded upward there (`q8_0` → 9, `q5` → 6,
`q4` → 5). That preserves the no-under-count rule.

This is a logical full-cache estimate. Backend padding, graph layout, paged-cache
allocation, parallel sequences, sliding-window/hybrid attention, recurrent state,
and separate K/V cache types can change physical allocation. These effects are a
reason the value is labelled `[E]`, even for an unquantized cache.

### Worked KV example

For 32 layers, 8 KV heads, 128-dimensional keys and values, f16, and 8192 tokens:

```text
KV/token = 32 × 8 × (128 + 128) × 2 bytes
         = 131,072 bytes = 128 KiB

KV total = 128 KiB × 8192 = 1 GiB
```

At 32,768 tokens, the same cache is 4 GiB.

## The `conservative-v1` fit policy

The prediction exposes both an expected footprint (used later to score accuracy)
and a conservative launch requirement (used for the verdict):

```text
weights          = selected GGUF/shard bytes                         [E]
KV               = architecture × requested context × cache width   [E]
runtime expected = max(64 MiB, 10% of weights), rounded to 16 MiB    [A]
runtime ceiling  = max(256 MiB, 15% of weights), rounded to 16 MiB   [A]

expected footprint     = weights + KV + runtime expected
conservative footprint = weights + KV + runtime ceiling
safety margin          = max(512 MiB, 5% of capacity), rounded to 16 MiB [A]
required               = conservative footprint + safety margin
```

The GGUF size is labelled estimated GPU residency because it assumes the entire
artifact is offloaded to one accelerator. File headers and alignment are included,
which is slightly conservative. The runtime terms cover backend context, graph,
scratch, allocator, and activation memory without pretending to model a specific
backend allocator.

For each accelerator:

```text
fits on device = required ≤ accelerator capacity
fits right now = required ≤ currently available accelerator memory
```

The second result is `unknown` when usage could not be measured. A true zero-free
sample is distinct and returns `does_not_fit`. If requested context exceeds the
GGUF's trained context, the verdict is `context_unsupported` even if the byte
budget would fit.

All prediction additions and multiplications saturate on overflow. Hostile or
implausibly large metadata therefore becomes “does not fit,” never a wrapped small
number.

The policy deliberately does not combine several GPUs. Each target is evaluated
for full residency. Tensor/row splitting and partial CPU offload require
loader-specific planning and are outside `conservative-v1`.

## Live device and loader inputs

`watch` composes device observations with loader observations.

Device inputs:

- NVIDIA: total/used/free and compute-process memory from `nvidia-smi`.
- AMD with AMD SMI: capacity and usage from `amd-smi`; on Linux, process memory
  is augmented from DRM `/proc/<pid>/fdinfo` and mapped by PCI address.
- Windows non-NVIDIA: capacity from the display-adapter registry and dedicated
  usage from the `GPU Adapter Memory` performance counter when mapping is
  unambiguous.
- Apple silicon: Metal's recommended maximum working set is the accelerator
  budget. Non-overlapping Mach VM free + inactive pages form a conservative
  current reclaimable estimate, clamped to that budget. Speculative pages are
  already included in `free_count`, so they are not added again. This is unified
  system memory, not dedicated VRAM, and is labelled as such.

Loader inputs:

- Ollama: `/api/ps` for resident identity/context/VRAM and `/api/show` for GGUF
  architecture and local blob path.
- llama.cpp server: `/props` for identity/context/model path, then a local GGUF
  header when the loopback server's path is readable.

## Live attribution

The inference footprint is selected in this order:

1. driver-measured process memory matched by loader PID, then by a narrow loader
   process name (`ollama*`, `llama-*`);
2. loader-reported model VRAM; or
3. estimated weights + KV when no footprint is exposed.

Inside that footprint:

- loader-reported KV wins; otherwise KV is estimated from architecture;
- readable GGUF size supplies estimated fully-offloaded weights;
- if weights are unavailable, they are the footprint remainder after KV;
- compute/runtime is the remaining inference footprint;
- other processes are device used minus inference footprint; and
- free memory is device total minus device used.

Reported weights win conflicts with estimated KV. An estimated KV is capped so it
cannot consume the entire inference footprint. Every segment is clamped and the
segments tile device capacity exactly.

## Prediction ledger and accuracy

`fit` stores its result locally unless `--no-record` is set. A resident model is
paired only when identity (name or a comparable digest), context, and available
quantization agree, and only when exactly one model is resident on that device.

Watch waits for three footprint samples within 2% before persisting an observation.
`report` may take a current matching observation immediately. Driver process memory
is `[M]`; loader model VRAM is `[R]`; an attributed fallback is `[E]`.

Accuracy compares the *expected* footprint, not the conservative launch ceiling:

```text
signed error %   = 100 × (predicted - observed) / observed
absolute error % = abs(signed error %)
```

A positive signed error means vramwatch over-predicted; a negative value means it
under-predicted. The conservative margin is excluded so it does not make the
estimator look artificially inaccurate.

## Known error sources

- partial GPU offload makes GGUF size overstate GPU-resident weights;
- undeclared KV quantization makes the default f16 cache estimate too large;
- loader/backend graph and scratch allocations differ from the generic runtime
  policy;
- another process can allocate after the current-availability sample;
- fragmented allocators can fail despite sufficient aggregate free memory;
- process counters may be unavailable because of driver, OS, or permissions; and
- hybrid/sliding-window/recurrent architectures can allocate less or differently
  than a full attention cache.

vramwatch's goal is a conservative, inspectable planning estimate whose uncertainty
is visible—not an allocator-level proof.

# Real-hardware validation

vramwatch's engine is unit-tested against fixtures, but the numbers below were
produced by running it against real hardware and a real inference server, then
cross-checked against independent ground truth. This page records what has been
validated so far. For what hasn't, see the Road to 1.0 in the README.

## AMD Radeon RX 7900 XT + Ollama, Windows 11

Hardware: AMD Radeon RX 7900 XT (20 GiB, gfx1100), Windows 11.
Loader: Ollama 0.31.1 (native Windows AMD/ROCm), model `qwen2.5:0.5b`.

### Device VRAM (Windows provider)

vramwatch reads the total from the registry and usage from the built-in
`GPU Adapter Memory\Dedicated Usage` performance counter. Both matched ground
truth exactly, and tracked live:

| vramwatch used | `typeperf` counter | note |
|---|---|---|
| 5.33 GiB | 5.33 GiB | idle desktop |
| 965.8 MiB | 966 MiB | later sample |
| 76.0 MiB | 76 MiB | picks the real GPU adapter, not the 0-usage software one |

Total was 19.98 GiB, matching the registry `qwMemorySize` (`0x4ff000000`).

### Weights / KV split (Ollama loader + KV formula)

With `qwen2.5:0.5b` resident on the GPU, Ollama's `/api/ps` reported
`size_vram = 459 MB`. vramwatch reads the model's GGUF blob (via the path in
`/api/show`) for a measured weight size, and splits that footprint as:

- weights **379.4 MiB** (the real GGUF blob size) + KV cache **48 MiB** + compute
  **32.1 MiB** = **459.5 MiB ≈ 459 MB**. The split sums to Ollama's own reported
  VRAM, with the compute/scratch VRAM correctly separated from the weights.

The KV figure matches the model's real architecture exactly. `/api/show` reports
qwen2.5:0.5b as 24 layers, 2 KV heads (GQA), embedding 896 / 14 heads = 64 head_dim,
f16 cache:

```
KV/token = 2 · 24 · 2 · 64 · 2 bytes = 12,288 bytes = 12 KiB   (vramwatch: "~12.0 KiB/token")
KV @ 4096  = 12 KiB · 4096  =  48 MiB   (vramwatch: 48 MiB)
KV @ 16384 = 12 KiB · 16384 = 192 MiB   (vramwatch: 192 MiB, grew exactly 4× with 4× context)
```

The linear KV growth with context, which is the main thing the tool exists to show,
was confirmed by reloading the model at 4× the context and watching the KV segment
grow exactly 4×.

## llama.cpp + a real GGUF, Windows 11

Loader: llama.cpp `llama-server` b9873 (Vulkan backend on the same RX 7900 XT),
serving the qwen2.5:0.5b GGUF (379 MB q4) with `-ngl 99`.

This exercises a completely different code path from Ollama: vramwatch reads
`/props` for the model path and context, then parses the **GGUF file header
directly** for the architecture and weight size (the Ollama path gets those from
`/api/show` instead). Both reached the same answer:

- weights **379.4 MiB**, which is the GGUF file's actual size on disk (379 MB) read
  straight from the file.
- KV cache **48 MiB** at ctx 4096, i.e. the same 12 KiB/token the Ollama run
  produced, because the GGUF parser recovered the same arch (24 layers, 2 KV heads,
  head_dim 64) that `/api/show` reported.

So the GGUF header parser (previously only tested against a synthetic fixture) reads
a real model file correctly, and two independent loaders agree on the split.

This run also surfaced and fixed a real Windows bug: llama.cpp's `/props` returns a
Windows path, and the model name was showing the full path because `path.Base`
doesn't split on backslashes.

## Still to validate

- NVIDIA hardware (the `nvidia-smi` path).
- AMD on Linux (the `rocm-smi` provider and `/proc/<pid>/fdinfo` per-process path).
  These are currently tested only against captured fixtures. Note that **WSL2 can't
  stand in for this**: it exposes the GPU through `/dev/dxg` (dxgkrnl), with no
  `amdgpu` kernel module and no `/sys/class/kfd` topology, so `rocm-smi`/`amd-smi`
  fail with "amdgpu not found in modules" and `/proc/<pid>/fdinfo` carries no `drm-*`
  memory keys. Validating this path needs native Linux on AMD (bare metal or PCI
  passthrough), not WSL.
- Multiple GPUs, and a range of drivers.

If you run vramwatch on your hardware, posting the result (and any mismatch) is the
most useful contribution. See the issues link in the README.

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
`size_vram = 459 MB`. vramwatch split that footprint as:

- weights **411.5 MiB** + KV cache **48 MiB** = **459.5 MiB ≈ 459 MB**. The split
  sums to Ollama's own reported VRAM.

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

## Still to validate

- NVIDIA hardware (the `nvidia-smi` path).
- AMD on Linux (the `rocm-smi` provider and `/proc/<pid>/fdinfo` per-process path).
  These are currently tested only against captured fixtures.
- Multiple GPUs, and a range of drivers.
- llama.cpp with a real GGUF (the split via GGUF-derived weights).

If you run vramwatch on your hardware, posting the result (and any mismatch) is the
most useful contribution. See the issues link in the README.

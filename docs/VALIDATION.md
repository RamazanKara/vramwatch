# Validation status

Validation has two layers. Hardware-free contract tests cover bounded remote GGUF
resolution, sharded-size accounting, quant selection, exact KV arithmetic,
overflow fail-closed behavior, current-vs-capacity verdicts, local ledger pairing,
privacy-safe SVG rendering, and all vendor/loader parsers. Historical hardware measurements below are retained from earlier runs; they are
not new measurements of this revision.

The new `conservative-v1` preflight policy still needs a broad public accuracy
corpus across backends, model families, and drivers. The local prediction ledger
and `report` accuracy card exist specifically to make those field results
comparable without telemetry.

## Maintenance checks, 2026-10-09

On Windows 11/amd64 with Go 1.26.8, the CLI examples exercised local GGUF fit,
Ollama and Hugging Face metadata, inferred Hub references, explicit file/revision
selection, direct HTTPS GGUF ranges, quantized KV, manual budgets, JSON, exit
codes, demo/mock watch, doctor, and report SVG creation/overwrite protection.
Local-path examples used a small synthetic GGUF; no full model was downloaded.
The installed Go command and the checksum-verified v0.7.1 Windows release binary
both ran successfully. License/release/CLI dependency badges returned HTTP 200.
The CI badge was removed because Actions is unavailable due to billing.

The Windows provider detected an AMD Radeon 780M. No inference loader was
running, so doctor correctly failed the loader check; live model attribution and
accuracy remain unverified in this pass. Registry probes are checked separately.
Linux amd64 and arm64 cross-builds passed.
WSL was blocked by the sandbox with `E_ACCESSDENIED`; native Linux/macOS execution,
Linux procfs, Metal, the shell installer, and CI/release execution were not
revalidated here. The historical field results below should not be read as
coverage of those gaps.

Statement coverage on Windows/amd64 (Go 1.26.8):

| Package | Before | After |
|---|---:|---:|
| total | 65.5% | 71.0% |
| internal/model | 65.2% | 69.6% |
| internal/engine | 90.3% | 90.3% |
| internal/source | 34.1% | 86.4% |
| internal/render | 85.4% | 85.4% |
| internal/fit | 76.5% | 76.9% |
| internal/loader | 72.2% | 78.4% |
| internal/gpu | 71.6% | 72.1% |
| internal/gguf | 73.4% | 73.4% |
| internal/ledger | 70.5% | 72.7% |
| cmd/vramwatch | 40.5% | 54.4% |
| tools/reportfixture | 0.0% | 0.0% |

The baseline failed two Unix-mode permission assertions and a CRLF-sensitive
JSON golden comparison on Windows. These checks are now portable. The final
`make fmt vet test build` gate passes; `make test` includes `-race`. The final
coverage profile combines the full race-enabled run with the refreshed GPU
package run after the mixed-adapter fix. `make card` and `make gif` regenerate
the committed assets unchanged (SVG line endings normalized). `make run` passed;
`make demo` displayed its loop and was stopped after sampling.

Both modules were tidied; Go is now pinned to 1.26.8 and the GIF generator uses
`golang.org/x/image` v0.46.0. Source `govulncheck` scans of both modules found no
vulnerabilities with Go 1.26.8. The existing v0.7.1 Windows release binary, built
with Go 1.26.3, flags 13 standard-library advisories in a binary scan. Those
archives were not republished in this maintenance pass.

## Live registry metadata smoke, 2026-07-18

The documented model-first command was run against both live metadata paths with
an 8 GiB manual target and no local model download:

```sh
vramwatch fit ollama:llama3.2:3b-instruct --quant q4_k_m --context 32768 --vram 8GiB --no-record --json
vramwatch fit hf:bartowski/Llama-3.2-1B-Instruct-GGUF --quant q4_k_m --context 32768 --vram 8GiB --no-record --json
```

- Ollama resolved `llama3.2:3b-instruct-q4_K_M`, a 2,019,377,376-byte model
  layer, after 3,914 metadata bytes. Its manifest digest matched Ollama's running
  model identity, and the ranged GGUF header recovered 28 layers and 8 KV heads.
- Hugging Face resolved the pinned `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, an
  807,694,464-byte artifact, after 29,468 metadata bytes and recovered 8 KV heads.

Both returned `fits` for the manual target. The byte counts prove the incremental
range reader stopped after architecture metadata rather than consuming the model
artifact; its enforced ceiling remains 16 MiB for unusually large headers.

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

With `qwen2.5:0.5b` resident on the GPU, the original run recorded the
following MiB breakdown. vramwatch reads the model's GGUF blob (via the path in
`/api/show`) for the exact GGUF file size used as estimated residency, and splits
that footprint as:

- weights **379.4 MiB** (GGUF blob size, treated as estimated full-offload
  residency) + KV cache **48 MiB** + compute
  **32.1 MiB** = **459.5 MiB** (about 482 MB). The split sums to Ollama's own reported
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
serving the qwen2.5:0.5b GGUF (379.4 MiB q4) with `-ngl 99`.

This exercises a completely different code path from Ollama: vramwatch reads
`/props` for the model path and context, then parses the **GGUF file header
directly** for the architecture and weight size (the Ollama path gets those from
`/api/show` instead). Both reached the same answer:

- weights **379.4 MiB**, the GGUF file's recorded size on disk, used as an
  estimated full-offload residency value.
- KV cache **48 MiB** at ctx 4096, i.e. the same 12 KiB/token the Ollama run
  produced, because the GGUF parser recovered the same arch (24 layers, 2 KV heads,
  head_dim 64) that `/api/show` reported.

So the GGUF header parser (previously only tested against a synthetic fixture) reads
a real model file correctly, and two independent loaders agree on the split.

This run also surfaced and fixed a real Windows bug: llama.cpp's `/props` returns a
Windows path, and the model name was showing the full path because `path.Base`
doesn't split on backslashes.

## Awaiting field reports

These paths are implemented and pass their fixture tests; they simply haven't been
run on native hardware here yet, so real-world results from users are the next
validation step:

- NVIDIA hardware (the `nvidia-smi` path).
- AMD on Linux (the `amd-smi` provider and `/proc/<pid>/fdinfo` per-process path).
  These are currently tested against captured fixtures. Note that **WSL2 can't
  stand in for this**: it exposes the GPU through `/dev/dxg` (dxgkrnl), with no
  `amdgpu` kernel module and no `/sys/class/kfd` topology, so `rocm-smi`/`amd-smi`
  fail with "amdgpu not found in modules" and `/proc/<pid>/fdinfo` carries no `drm-*`
  memory keys. Validating this path needs native Linux on AMD (bare metal or PCI
  passthrough), not WSL.
- Multiple GPUs, and a range of drivers.
- Apple silicon unified memory through the native Metal provider.
- Preflight expected-footprint error across Ollama/llama.cpp backends and larger
  context windows. Share `vramwatch report --svg` cards or scrubbed JSON when
  filing these results.

If you run vramwatch on your hardware, posting the result (and any mismatch) is the
most useful contribution. See the [issue tracker](https://github.com/RamazanKara/vramwatch/issues).

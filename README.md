<h1 align="center">vramwatch</h1>

<p align="center"><em>The flame graph for “why won’t this model fit.”</em></p>

<p align="center">
  A single, dependency-free binary that live-traces where every megabyte of your
  local-LLM VRAM went — <strong>weights vs KV cache vs other apps</strong> — and
  predicts how much context fits before you OOM.
</p>

<p align="center">
  <a href="https://github.com/RamazanKara/vramwatch/actions/workflows/ci.yml"><img src="https://github.com/RamazanKara/vramwatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/RamazanKara/vramwatch/releases"><img src="https://img.shields.io/github/v/release/RamazanKara/vramwatch?sort=semver" alt="Release"></a>
  <img src="https://img.shields.io/badge/deps-0-brightgreen" alt="Zero dependencies">
</p>

<p align="center"><img src="docs/demo.gif" alt="vramwatch watching the KV cache grow until OOM" width="720"></p>

---

`nvidia-smi` and `rocm-smi` tell you a GPU is using 23.5 of 24 GiB. They can’t tell
you **why**: how much is model weights, how much is the KV cache that grows with
your context, and how much is the desktop compositor you forgot about. So when a
70B model that “should fit” OOMs at 22 GiB, you’re guessing.

vramwatch attributes VRAM **inside** the inference process and shows you the split
live:

```text
vramwatch v1.0.0

GPU 0  AMD Radeon RX 7900 XTX  (amd, driver 6.7.0)
[███████████████████████████████████████████████]  23.75 GiB / 24.00 GiB used
  █ weights      19.50 GiB   81.2%  (ollama, estimated)
  █ KV cache      2.50 GiB   10.4%  (ollama, estimated)
  █ other apps    1.75 GiB    7.3%
  █ free         256.0 MiB    1.0%
  model: llama3:70b-q4  ctx 8192/8192
  ⚠ OOM risk: headroom 256.0 MiB • ~320.0 KiB/token • max context ≈ 8,192 tokens
```

## Why vramwatch

- **Within-process attribution.** Not “process X uses 22 GiB” — but *of that 22 GiB,
  19.5 is weights and 2.5 is KV cache*. That’s the number that tells you whether a
  longer context or a bigger quant will fit.
- **OOM prediction.** It knows your model’s KV-cache growth per token, so it tells
  you the **max context that fits** and answers *“will 32k fit?”* before you try it.
- **Quantized-KV aware.** Running a `q8_0`/`q4_0` KV cache? Pass `--kv-cache-type`
  and the estimate is right, not 2–4× too high.
- **AMD/ROCm is a peer, not an afterthought.** Most VRAM tooling is CUDA-only.
- **Zero friction, zero deps.** One static binary. No Python, no CUDA toolkit, no
  account, nothing uploaded. `curl | sh` and go.
- **Honest.** Anything derived rather than measured is labelled `estimated`, and the
  method is [documented in full](docs/METHODOLOGY.md).

## vramwatch vs. the usual tools

|                                   | vramwatch | `nvidia-smi` | `nvtop` / `nvitop` |
|-----------------------------------|:---:|:---:|:---:|
| Device total / used / free        | ✅ | ✅ | ✅ |
| Per-process VRAM                  | ✅ (NVIDIA) | ✅ | ✅ |
| **Weights vs KV-cache split**     | ✅ | ❌ | ❌ |
| **Max context before OOM**        | ✅ | ❌ | ❌ |
| **“Will 32k context fit?”**       | ✅ | ❌ | ❌ |
| Shareable SVG scorecard           | ✅ | ❌ | ❌ |
| AMD / ROCm                        | ✅ | ❌ | ✅ (nvtop) |
| Single static binary, no Python   | ✅ | n/a | ❌ (nvitop) |

vramwatch doesn’t replace `nvtop` for live GPU utilisation graphs — it answers the
one question those tools can’t: *what is my model actually spending VRAM on?*

## Install

```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/RamazanKara/vramwatch/master/install.sh | sh

# With Go
go install github.com/RamazanKara/vramwatch/cmd/vramwatch@latest
```

Windows: grab the `.zip` from [Releases](https://github.com/RamazanKara/vramwatch/releases),
or `go install` as above.

## Usage

```sh
vramwatch watch                      # live TUI (updates as the KV cache grows)
vramwatch snapshot                   # one-shot breakdown
vramwatch snapshot --json            # machine-readable
vramwatch snapshot --svg card.svg    # branded scorecard to share
vramwatch predict --context 32768    # will 32k context fit? what's the max?
vramwatch devices                    # what GPUs/loaders did I detect?
```

Running a quantized KV cache (Ollama `OLLAMA_KV_CACHE_TYPE`, llama.cpp `--cache-type-k`)?
Tell vramwatch so the estimate matches:

```sh
vramwatch watch --kv-cache-type q8_0        # or export VRAMWATCH_KV_CACHE_TYPE=q8_0
```

No GPU handy? Every command takes `--source`:

```sh
vramwatch watch --source demo   # synthetic card whose KV cache grows until OOM
vramwatch snapshot --source mock:testdata/scenarios/24gb-70b-oom.json
```

`snapshot --svg` writes a shareable scorecard:

<p align="center"><img src="docs/sample/vramwatch-card.svg" alt="vramwatch SVG scorecard" width="640"></p>

### `predict`

```text
$ vramwatch predict --context 32768
GPU 0  AMD Radeon RX 7900 XTX
  model: llama3:70b-q4   ~320.0 KiB/token
  headroom: 256.0 MiB
  max context that fits: ~8,192 tokens   (OOM risk now)
  target 32,768 tokens: WON'T FIT (needs 29.50 GiB, card has 24.00 GiB)
```

## How it works

vramwatch combines two sources per GPU:

1. **The driver** (`nvidia-smi` / `rocm-smi`) — device total/used/free, plus
   per-process VRAM on NVIDIA. This is ground truth.
2. **The loader** — which models are resident and their architecture:
   - **Ollama** via `/api/ps` (VRAM) + `/api/show` (architecture).
   - **llama.cpp** via `/props` (context) + **reading the GGUF file’s header**
     directly for the architecture and weight size.

It then splits the inference process’s footprint. The KV cache is computed with the
standard formula:

```
KV bytes/token = 2 (K and V) · n_layers · n_kv_heads · head_dim · bytes_per_element
```

which is GQA/MQA-aware (`n_kv_heads`) and dtype-aware (`bytes_per_element`, set by
`--kv-cache-type`). The segments always tile the device exactly:
**weights + KV + compute + other apps + free = total**.

**The full method — including a worked example and exactly what’s measured vs.
estimated — is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).**

## Accuracy: measured vs. estimated

| Figure | How it’s obtained | Trust |
|--------|-------------------|-------|
| Device total / used / free | Driver (`nvidia-smi`/`rocm-smi`) | measured |
| Per-process VRAM (NVIDIA) | Driver compute-apps query | measured |
| KV cache | `arch × context × dtype` (formula) | **estimated** — exact once `--kv-cache-type` matches your cache |
| Weights (Ollama) | `process VRAM − KV` | **estimated** |
| Weights (llama.cpp) | GGUF file size | **estimated** (assumes full GPU offload) |
| Max context before OOM | `free ÷ KV-bytes-per-token` | **estimated**, linear |

Everything in the estimated rows is labelled `estimated` in the output. See the
[FAQ](docs/FAQ.md) if your numbers don’t match what you expect.

## Supported

| GPU vendor | via         | device totals | per-process | notes |
|------------|-------------|:---:|:---:|-------|
| NVIDIA     | `nvidia-smi`| ✅ | ✅ | full support |
| AMD        | `rocm-smi`  | ✅ | ❌ | per-process not collected in v1.0; footprint comes from the loader |

| Loader   | via                       | model + VRAM | weights/KV split |
|----------|---------------------------|:---:|:---:|
| Ollama   | `/api/ps`, `/api/show`    | ✅ | ✅ (arch from the API) |
| llama.cpp| `/props` + GGUF header    | ✅ (from GGUF) | ✅ (arch + weights from the GGUF file) |

## Limitations

vramwatch is deliberately honest about what it can and can’t know:

- **Weights/KV are estimated, not allocator-hooked.** v1.0 does not intercept the
  CUDA/HIP allocator; it derives the split from the loader’s reported footprint (or
  the GGUF file) plus the model architecture. The KV formula is exact for a given
  dtype; weights are the remainder (Ollama) or the file size (llama.cpp).
- **KV dtype defaults to f16.** vramwatch can’t read the loader’s cache-type setting,
  so pass `--kv-cache-type q8_0` (or set `VRAMWATCH_KV_CACHE_TYPE`) if you quantized
  it. With the right dtype the KV figure is exact.
- **llama.cpp weights assume full GPU offload.** The GGUF file size ≈ VRAM weights
  only when every layer is on the GPU; with partial offload it over-reports weights.
- **AMD per-process VRAM is not collected.** On AMD the inference footprint comes
  from the loader’s reported VRAM (the weights/KV split still works). Per-process
  driver attribution is NVIDIA-only for now.
- **Prediction is linear** in the KV cache and holds weights/overhead constant — a
  good planning estimate, not a guarantee.

**Roadmap:** allocator-level attribution, KV-dtype auto-detection, ROCm per-process
data, partial-offload awareness, and vLLM / MLX / Apple-Metal providers.

## Docs

- [Methodology](docs/METHODOLOGY.md) — the attribution model and KV math in depth.
- [FAQ](docs/FAQ.md) — “why estimated?”, “my numbers don’t match `nvidia-smi`”, etc.
- [Contributing](CONTRIBUTING.md) — adding GPU/loader providers.

## Building

```sh
make build     # -> ./vramwatch
make test
make card      # regenerate the sample scorecard
make gif       # regenerate the demo GIF
```

No third-party dependencies — standard library only.

## License

[Apache-2.0](LICENSE) © Ramazan Kara

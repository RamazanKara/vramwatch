# Changelog

All notable changes to vramwatch are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- `docs/VALIDATION.md` — end-to-end real-hardware validation (AMD RX 7900 XT +
  Ollama on Windows): device VRAM matches the registry/counter, the weights/KV
  split sums to Ollama's reported VRAM, and the KV cache grows exactly with context
  (matching the model's real GQA architecture).

## [0.4.0] - 2026-07-05

Focus: real-hardware validation → **vramwatch now works on Windows AMD**.

### Added
- **Windows GPU provider.** On Windows — where AMD's consumer driver ships no
  `rocm-smi` — vramwatch reads the real VRAM size from the registry
  (`HardwareInformation.qwMemorySize`) and usage from the built-in `GPU Adapter
  Memory\Dedicated Usage` performance counter (`typeperf`), with no extra tooling;
  NVIDIA stays on `nvidia-smi`. Validated live against a real **Radeon RX 7900 XT** —
  total/used match the registry and the counter exactly. (Discrete Intel Arc cards
  go through the same path but are **untested**; integrated GPUs, which report no
  dedicated VRAM, are not detected. Multi-GPU usage is left unattributed rather than
  guessed.)
- `VendorIntel`, and an OS-specific provider hook so more platforms can plug in.

### Fixed
- Before this, vramwatch reported "no GPUs detected" on Windows with an AMD card.

## [0.3.0] - 2026-07-05

Focus: complete per-process attribution.

### Added
- **Per-process VRAM for AMD on Linux**, read from the kernel’s
  `/proc/<pid>/fdinfo` DRM interface (deduplicated by DRM client id, mapped to a
  device by PCI address). Previously per-process was NVIDIA-only. The reader is
  vendor-neutral (amdgpu/i915), though only AMD devices are surfaced for now.
- The inference **footprint is now matched by process name** (`ollama` /
  `llama-server`) when a loader doesn’t report a PID, so per-process VRAM actually
  improves the footprint on NVIDIA and AMD instead of being collected and ignored.

### Changed
- The `rocm-smi` query adds `--showbus` to recover each device’s PCI address.

## [0.2.0] - 2026-07-05

Focus: reduce the estimation limitations and document the method in full. Still
0.x — the tool is young and hasn't been validated on a broad range of real
hardware yet (see “Road to 1.0” in the README).

### Added
- **KV cache dtype support** — `--kv-cache-type` (and `$VRAMWATCH_KV_CACHE_TYPE`)
  so a quantized cache (`q8_0`, `q4_0`, `f32`, …) is estimated correctly instead of
  silently assuming f16.
- **GGUF header parsing** (`internal/gguf`) — vramwatch reads the model file’s
  header directly, giving **llama.cpp a real weights/KV split** for the first time
  (architecture + weight size), not just a context number.
- Weights derived from a GGUF file size are labelled `estimated`, and attribution
  derives the footprint from weights+KV when a loader (llama.cpp) reports no VRAM.
- Animated demo GIF hero, generated reproducibly (`make gif`).

### Documentation
- New [METHODOLOGY.md](docs/METHODOLOGY.md) — the attribution model and KV math in
  depth, with a worked example and a measured-vs-estimated breakdown.
- New [FAQ.md](docs/FAQ.md).
- README rewritten: a comparison to `nvidia-smi`/`nvtop`/`nvitop`, an accuracy
  table, the new `--kv-cache-type` workflow, and trimmed, honest limitations.

## [0.1.0] - 2026-07-05

Initial public release. Includes, on top of the core tool, the fixes from a full
adversarial code review (correct exit codes, robust `nvidia-smi`/`rocm-smi`
parsing, reported-weights-win attribution, known-arch prediction fallback,
`install.sh` fallback, Windows ANSI).

First public release.

### Added
- `watch` — live TUI stacked VRAM bar that updates as the KV cache grows, with
  a per-segment legend, resident models, and an OOM-risk line.
- `snapshot` — one-shot breakdown to the console, `--json`, or an `--svg`
  branded scorecard (the shareable artifact). `--static` for reproducible output.
- `predict` — max context that fits before OOM, and a `--context N` fit check
  for a target context length.
- `devices` — diagnostics: detected GPU providers, loader providers, and GPUs.
- Within-process VRAM attribution: weights vs KV cache vs compute vs other apps,
  tiling the device exactly.
- KV-cache estimation from model architecture (GQA/MQA aware, quantized-cache
  aware) using the standard `2 · layers · kv_heads · head_dim · bytes` formula.
- GPU providers: NVIDIA (`nvidia-smi`) and AMD (`rocm-smi`), including
  per-process attribution where the driver reports it.
- Loader providers: Ollama (first-class, pulls architecture from `/api/show`)
  and llama.cpp (best-effort context + model name via `/props`).
- `demo` and `mock:PATH` data sources for hardware-free demos, tests, and CI.
- Single static, dependency-free binary for Linux, macOS, and Windows.

[Unreleased]: https://github.com/RamazanKara/vramwatch/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/RamazanKara/vramwatch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/RamazanKara/vramwatch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/RamazanKara/vramwatch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/RamazanKara/vramwatch/releases/tag/v0.1.0

# Changelog

All notable changes to vramwatch are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-07-05

Switch the AMD provider from `rocm-smi` to `amd-smi`.

### Changed
- **AMD GPUs are now read via `amd-smi` instead of `rocm-smi`.** `amd-smi` is AMD's
  current SMI CLI; `rocm-smi` is deprecated in its favour, so vramwatch no longer
  calls `rocm-smi` at all. It runs `amd-smi static --json` (identity + capacity) and
  `amd-smi metric --mem-usage --json` (live VRAM used/free), joined on the per-GPU
  index. The new parser handles `amd-smi`'s JSON-array root and `{value, unit}`
  leaves, tolerates any field or block collapsing to `"N/A"`, and corrects the
  `MB`-labelled values (which are really MiB). The name-fallback safeguard is
  preserved: a GPU with no product name shows as `AMD GPU N`, and an entry with
  neither VRAM numbers nor identity is skipped rather than shown as a phantom.
- The `amd-smi` parser is hardened against degraded output (found via adversarial
  review): a card `amd-smi` can name but not size — common on Windows, where many
  queries return `"N/A"` — is kept with an unknown capacity instead of vanishing;
  `used` is derived from `total − free` (and vice-versa) when only one is reported;
  a quoted or float `gpu` index still joins; and an implausibly huge VRAM value is
  rejected rather than saturating the total.

## [0.5.0] - 2026-07-05

Feature-complete. Every planned GPU vendor (NVIDIA, AMD) and loader (Ollama,
llama.cpp) is implemented and covered by fixture tests. Still `0.x` — the CLI and
JSON shapes can change as field reports arrive — but the capabilities are shipped.
See "Feature status" in the README.

### Added
- **Measured weights for Ollama.** vramwatch now reads the model's GGUF blob (via
  the path in `/api/show`) for a real weight size, instead of leaving weights as the
  `footprint − KV` remainder. This separates compute/scratch VRAM from weights.
  Validated on real hardware: qwen2.5:0.5b split as weights 379.4 MiB (the blob
  size) + KV 48 MiB + compute 32.1 MiB, summing to Ollama's reported `size_vram`.
- A golden test pins the `--json` snapshot schema, so an accidental change to the
  machine-readable output (a field added, removed, renamed, or reformatted) fails
  CI. Regenerate deliberately with `-update-golden`.

### Fixed
- llama.cpp model names on Windows showed the full path (`path.Base` doesn't split
  on `\`); now the basename is shown. Found while validating against a real GGUF.
- **rocm-smi parser hardened against real-world output.** Three defects that would
  bite a real Linux+AMD user are fixed: (1) a card value that nests a JSON object
  (ROCm 6/7 emit these — metrics, MI300 partition info) no longer makes the whole
  parse fail and drop *every* AMD GPU; (2) a card that reports no VRAM
  (headless/masked) is skipped instead of appearing as a phantom 0-byte GPU; (3) the
  GPU name no longer falls back to the hex device id (`0x744c`) — it uses the product
  name or a clean `AMD GPU N`. Each is covered by a regression test.

### Documentation
- `docs/VALIDATION.md`: end-to-end real-hardware validation (AMD RX 7900 XT +
  Ollama on Windows). Device VRAM matches the registry/counter, the weights/KV
  split sums to Ollama's reported VRAM, and the KV cache grows exactly with context
  (matching the model's real GQA architecture).
- Reframed the README "Road to 1.0" section as "Feature status" (feature-complete,
  with NVIDIA and AMD-on-Linux implemented + fixture-tested and awaiting field
  reports), and renamed VALIDATION.md's "Still to validate" to "Awaiting field
  reports". Added an `amd-smi` fallback to the roadmap.

## [0.4.0] - 2026-07-05

Focus: real-hardware validation. vramwatch now works on Windows AMD.

### Added
- **Windows GPU provider.** On Windows, where AMD's consumer driver ships no
  `rocm-smi`, vramwatch reads the real VRAM size from the registry
  (`HardwareInformation.qwMemorySize`) and usage from the built-in `GPU Adapter
  Memory\Dedicated Usage` performance counter (`typeperf`), with no extra tooling;
  NVIDIA stays on `nvidia-smi`. Validated live against a real Radeon RX 7900 XT,
  where total/used match the registry and the counter exactly. (Discrete Intel Arc
  cards go through the same path but are **untested**; integrated GPUs, which report
  no dedicated VRAM, are not detected. Multi-GPU usage is left unattributed rather
  than guessed.)
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
  `llama-server`) when a loader doesn’t report a PID, so per-process VRAM improves
  the footprint on NVIDIA and AMD instead of being collected and ignored.

### Changed
- The `rocm-smi` query adds `--showbus` to recover each device’s PCI address.

## [0.2.0] - 2026-07-05

Focus: reduce the estimation limitations and document the method in full. Still
0.x, so the tool is young and hasn't been validated on a broad range of real
hardware yet (see “Road to 1.0” in the README).

### Added
- **KV cache dtype support** via `--kv-cache-type` (and `$VRAMWATCH_KV_CACHE_TYPE`),
  so a quantized cache (`q8_0`, `q4_0`, `f32`, …) is estimated correctly instead of
  silently assuming f16.
- **GGUF header parsing** (`internal/gguf`): vramwatch reads the model file’s
  header directly, giving **llama.cpp a real weights/KV split** for the first time
  (architecture + weight size) instead of only a context number.
- Weights derived from a GGUF file size are labelled `estimated`, and attribution
  derives the footprint from weights+KV when a loader (llama.cpp) reports no VRAM.
- Animated demo GIF hero, generated reproducibly (`make gif`).

### Documentation
- New [METHODOLOGY.md](docs/METHODOLOGY.md), covering the attribution model and KV
  math in depth, with a worked example and a measured-vs-estimated breakdown.
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
- `watch`: live TUI stacked VRAM bar that updates as the KV cache grows, with
  a per-segment legend, resident models, and an OOM-risk line.
- `snapshot`: one-shot breakdown to the console, `--json`, or an `--svg`
  branded scorecard (the shareable artifact). `--static` for reproducible output.
- `predict`: max context that fits before OOM, and a `--context N` fit check
  for a target context length.
- `devices`: diagnostics for detected GPU providers, loader providers, and GPUs.
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

[Unreleased]: https://github.com/RamazanKara/vramwatch/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/RamazanKara/vramwatch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/RamazanKara/vramwatch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/RamazanKara/vramwatch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/RamazanKara/vramwatch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/RamazanKara/vramwatch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/RamazanKara/vramwatch/releases/tag/v0.1.0

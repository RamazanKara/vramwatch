# Changelog

All notable changes to vramwatch are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-05

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

[Unreleased]: https://github.com/RamazanKara/vramwatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RamazanKara/vramwatch/releases/tag/v0.1.0

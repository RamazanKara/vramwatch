# Contributing to vramwatch

Thanks for your interest! vramwatch is a small, dependency-free Go tool, and
contributions are welcome, especially new GPU/loader providers and better
attribution.

## Development

```sh
git clone https://github.com/RamazanKara/vramwatch
cd vramwatch
go test ./...          # run the suite
make build             # build ./vramwatch
make demo              # live TUI against the synthetic demo source
make gif               # regenerate the animated README walkthrough
```

The shipped module has **no third-party Go dependencies** and should stay that
way: the value proposition is one binary with no language runtime or package
environment. Keep the root `go.mod` free of `require` lines beyond the standard
library. The standalone `docs/gifgen` module may use `x/image` to produce the
README asset; it is not linked into vramwatch. Native macOS builds link only
Apple's system Foundation/Metal frameworks.

## Ground rules

- `gofmt` clean, `go vet ./...` clean, `go test ./...` green. CI enforces all
  three plus `-race`.
- Parsing logic (vendor CLI output, loader JSON) must be a **pure function**
  with a fixture-based test, so it can be verified without a GPU. See
  `internal/gpu/*_test.go` and `internal/loader/loader_test.go`.
- Attribution must keep tiling the device exactly: the segments always sum to
  `gpu.TotalBytes`. `internal/engine` has tests that assert this.
- Be honest in output. Set provenance explicitly: `measured`, `loader_reported`,
  `model_estimated`, `assumed`, or `user_supplied`. Don't present a derived value
  as ground truth.
- Fit must fail closed. Unknown artifact size, incomplete architecture/shards, or
  arithmetic overflow may not degrade to a smaller optimistic prediction.
- Shareable SVGs may not gain hostnames, paths, URL queries, PIDs, bus IDs, or
  hardware serials. Add a privacy regression test for any new field.

## Adding a GPU provider

Implement `gpu.Provider` (`Name`, `Vendor`, `Available`, `Sample`) and register
it in `gpu.All()`. Keep the actual command execution thin and put the parsing in
a tested pure function. Set `CapacitySource` and `UsageSource`; if usage cannot
be queried, use `ProvenanceAssumed` so `fit` reports current availability as
unknown. Apple platform providers also require native macOS CI coverage because
the shipped implementation uses cgo and system frameworks.

## Adding a loader provider

Implement `loader.Provider` (`Name`, `Available`, `Models`) and register it in
`loader.All()`. If you can extract architecture (layers, KV heads, head dim),
fill `model.Arch` so the engine can compute the weights/KV split. When your
loader exposes a GGUF file path, `internal/gguf.Read` gives you the architecture
and weight size for free; see the llama.cpp provider for the pattern.

Set `VRAMBytes` only when the loader exposes a resident footprint, and identify it
with `VRAMSource`. An exact GGUF file size may populate `WeightsBytes`, but GPU
residency still needs `Estimated: true` because partial offload is possible. Leave
`KVCacheBytes` at zero to let the engine estimate it from the architecture (which
also lets `--kv-cache-type` apply). Populate `Quantization`, `Digest`, and
`ArtifactPath` when available; the ledger uses them to avoid pairing a prediction
with the wrong resident model.

## Changing preflight prediction

The public policy is versioned as `conservative-v1` in `internal/fit`. A policy
change must update the version, methodology, JSON fixtures, and tests for
fits-on-device, fits-now, unknown current usage, unsupported context, and overflow.
Remote resolver tests use a custom HTTP transport and must continue proving that
only metadata plus a bounded range is read.

## Commit / PR

- One logical change per PR; describe the user-visible effect.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Sign off your commits (`git commit -s`).

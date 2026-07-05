# Contributing to vramwatch

Thanks for your interest! vramwatch is a small, dependency-free Go tool, and
contributions are welcome — especially new GPU/loader providers and better
attribution.

## Development

```sh
git clone https://github.com/RamazanKara/vramwatch
cd vramwatch
go test ./...          # run the suite
make build             # build ./vramwatch
make demo              # live TUI against the synthetic demo source
```

There are **no third-party dependencies** and there should stay none: the whole
value proposition is a single static binary. Please keep `go.mod` free of
`require` lines beyond the standard library.

## Ground rules

- `gofmt` clean, `go vet ./...` clean, `go test ./...` green. CI enforces all
  three plus `-race`.
- Parsing logic (vendor CLI output, loader JSON) must be a **pure function**
  with a fixture-based test, so it can be verified without a GPU. See
  `internal/gpu/*_test.go` and `internal/loader/loader_test.go`.
- Attribution must keep tiling the device exactly: the segments always sum to
  `gpu.TotalBytes`. `internal/engine` has tests that assert this.
- Be honest in output. Anything derived rather than measured is marked
  `estimated`. Don't present an estimate as ground truth.

## Adding a GPU provider

Implement `gpu.Provider` (`Name`, `Vendor`, `Available`, `Sample`) and register
it in `gpu.All()`. Keep the actual command execution thin and put the parsing in
a tested pure function.

## Adding a loader provider

Implement `loader.Provider` (`Name`, `Available`, `Models`) and register it in
`loader.All()`. If you can extract architecture (layers, KV heads, head dim),
fill `model.Arch` so the engine can compute the weights/KV split.

## Commit / PR

- One logical change per PR; describe the user-visible effect.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Sign off your commits (`git commit -s`).

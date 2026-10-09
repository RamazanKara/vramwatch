# Releasing locally

GitHub Actions is currently unavailable. Run the local gate and build the release
artifacts below from the same commit on every build machine. There is no
`make release` target; `make build VERSION=v0.7.2` builds only the host binary.
The next patch release is `v0.7.2`; keep its notes under `[Unreleased]` until release.

## Gate

Use Go 1.27.2 or newer (both CI workflows take the version from `go.mod`). Check
[Go releases](https://go.dev/dl/) and the vulnerability database again on release
day. The CLI has no third-party modules; `docs/gifgen` is a separate module.

With GNU make, a POSIX shell, and a C compiler:

```sh
export GOTOOLCHAIN=$(go env GOVERSION)
make fmt vet test build VERSION=v0.7.2
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck ./...
(cd docs/gifgen && go test ./... && go vet ./... && govulncheck ./...)
```

On Windows without make or a C compiler, use PowerShell 7.3+:

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:PATH = "C:\Program Files\Go\bin;$env:PATH"
$env:CGO_ENABLED = '0'
$env:GOTOOLCHAIN = go env GOVERSION
$files = @(git ls-files -co --exclude-standard '*.go')
if (gofmt -l $files) { throw 'Run gofmt -w on the listed files' }
go vet ./...
go test -count=1 -timeout=60s ./...
go build -ldflags '-X main.Version=v0.7.2' -o dist/vramwatch.exe ./cmd/vramwatch
go install golang.org/x/vuln/cmd/govulncheck@latest
& "$(go env GOPATH)/bin/govulncheck.exe" ./...
Push-Location docs/gifgen
try {
    go test ./...
    go vet ./...
    & "$(go env GOPATH)/bin/govulncheck.exe" ./...
} finally { Pop-Location }
```

The Windows fallback omits `-race`: it requires cgo and a compatible C compiler.
Run `go test -race -timeout=60s ./...` on Linux with a C compiler before release.
Unix permission assertions in the ledger tests are skipped on Windows; run them
on Linux too. Native macOS testing is needed for the cgo Metal provider. A Darwin
cross-build with `CGO_ENABLED=0` compiles the fallback without Metal and is not a
substitute for the macOS release artifacts.

Run each parser fuzzer for 30 seconds (the same commands work in PowerShell and a
POSIX shell). Two workers keep GGUF fuzzing's memory use bounded. Retain only
small regression inputs in `testdata/fuzz`; generated coverage inputs stay in the
Go build cache.

```sh
go test ./internal/gguf -run='^$' -fuzz='^FuzzReadGGUF$' -fuzztime=30s -parallel=2
go test ./internal/gpu -run='^$' -fuzz='^FuzzNvidiaCSV$' -fuzztime=30s -parallel=2
go test ./internal/gpu -run='^$' -fuzz='^FuzzAMDSMI$' -fuzztime=30s -parallel=2
go test ./internal/gpu -run='^$' -fuzz='^FuzzWindowsOutput$' -fuzztime=30s -parallel=2
go test ./internal/gpu -run='^$' -fuzz='^FuzzFdinfo$' -fuzztime=30s -parallel=2
go test ./internal/loader -run='^$' -fuzz='^FuzzOllamaMetadata$' -fuzztime=30s -parallel=2
go test ./internal/loader -run='^$' -fuzz='^FuzzLlamaProps$' -fuzztime=30s -parallel=2
go test ./internal/fit -run='^$' -fuzz='^FuzzArtifactReferences$' -fuzztime=30s -parallel=2
go test ./cmd/vramwatch -run='^$' -fuzz='^FuzzParseByteSize$' -fuzztime=30s -parallel=2
```

## Windows and Linux archives

Use a clean checkout of the release commit. Finalize the changelog, commit and
publish its version tag as a separate maintainer action before publishing the
release. Do not build from an unrelated or dirty checkout. The commands below
only build and package; they do not create a tag or publish anything.

In PowerShell 7.3+, from the repository root, use GNU tar and gzip from Git for
Windows (adjust its installation path below if needed). Explicit modes keep the
Linux binaries executable when packaged on Windows:

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:PATH = "C:\Program Files\Git\usr\bin;$env:PATH"
$Version = 'v0.7.2'
$releaseDir = "dist/$Version"
New-Item -ItemType Directory -Path $releaseDir | Out-Null
$env:CGO_ENABLED = '0'
try {
    foreach ($target in @('linux/amd64', 'linux/arm64', 'windows/amd64')) {
        $env:GOOS, $env:GOARCH = $target.Split('/')
        $stage = "$releaseDir/stage/$($env:GOOS)_$($env:GOARCH)"
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        $binary = if ($env:GOOS -eq 'windows') { 'vramwatch.exe' } else { 'vramwatch' }
        go build -trimpath -ldflags "-s -w -X main.Version=$Version" -o "$stage/$binary" ./cmd/vramwatch
        Copy-Item README.md, LICENSE -Destination $stage
        $name = "vramwatch_${Version}_$($env:GOOS)_$($env:GOARCH)"
        if ($env:GOOS -eq 'windows') {
            Compress-Archive -Path "$stage/$binary", "$stage/README.md", "$stage/LICENSE" -DestinationPath "$releaseDir/$name.zip"
        } else {
            tar --mode=0755 -cf "$releaseDir/$name.tar" -C $stage $binary
            tar --mode=0644 -rf "$releaseDir/$name.tar" -C $stage README.md LICENSE
            gzip "$releaseDir/$name.tar"
        }
    }
} finally {
    Remove-Item Env:GOOS, Env:GOARCH -ErrorAction SilentlyContinue
}
& "$releaseDir/stage/windows_amd64/vramwatch.exe" version
```

Inspect each archive: its root must contain `vramwatch` (or `vramwatch.exe`),
`README.md`, and `LICENSE`. Run the native binary's `version` command on each
platform and confirm the release version. Linux cross-builds can be made on
Windows, but their execution must be checked on Linux.

## Native macOS archives

On both an Intel Mac and an Apple-silicon Mac with Xcode Command Line Tools,
check out the same release commit and run this in a POSIX shell:

```sh
set -eu
VERSION=v0.7.2
ARCH=$(go env GOHOSTARCH)
STAGE=$(mktemp -d)
export CGO_ENABLED=1 GOOS=darwin GOARCH="$ARCH"
go test -race -timeout=60s ./...
govulncheck ./...
go build -trimpath -ldflags "-s -w -X main.Version=$VERSION" -o "$STAGE/vramwatch" ./cmd/vramwatch
"$STAGE/vramwatch" version
cp README.md LICENSE "$STAGE/"
mkdir -p "dist/$VERSION"
tar -C "$STAGE" -czf "dist/$VERSION/vramwatch_${VERSION}_darwin_${ARCH}.tar.gz" vramwatch README.md LICENSE
```

Copy both macOS archives into the Windows release directory, alongside the three
cross-built archives. Keep the exact asset names: `install.sh` relies on them.

## Checksums and publication

In the same PowerShell session, create checksums from only the five archives:

```powershell
$names = @(
    "vramwatch_${Version}_darwin_amd64.tar.gz",
    "vramwatch_${Version}_darwin_arm64.tar.gz",
    "vramwatch_${Version}_linux_amd64.tar.gz",
    "vramwatch_${Version}_linux_arm64.tar.gz",
    "vramwatch_${Version}_windows_amd64.zip"
)
$assets = @($names | ForEach-Object { "$releaseDir/$_" })
$lines = foreach ($asset in $assets) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $asset).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($asset))"
}
[IO.File]::WriteAllText("$PWD/$releaseDir/SHA256SUMS", ($lines -join "`n") + "`n", [Text.Encoding]::ASCII)
Copy-Item "$releaseDir/SHA256SUMS" "$releaseDir/checksums.txt"
```

Publish both `SHA256SUMS` and the identical `checksums.txt`: existing installers
fetch the latter. On Linux, verify downloaded archives with
`sha256sum --check SHA256SUMS`; on macOS use `shasum -a 256 --check SHA256SUMS`.

Only when the gate, native checks, archives, and notes have been reviewed, use
[GitHub CLI release create](https://cli.github.com/manual/gh_release_create).
The tag must already exist on GitHub and identify the exact build commit.
Copy the release's changelog section to `dist/v0.7.2-notes.md`, then run:

```powershell
gh release create $Version @assets "$releaseDir/SHA256SUMS" "$releaseDir/checksums.txt" --title $Version --notes-file "dist/$Version-notes.md" --verify-tag
```

Do not upload the staging directories or unrelated contents of `dist`.

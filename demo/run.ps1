# Live demo: a synthetic 24 GiB card whose KV cache grows until it OOMs.
# Great for recording the README GIF. No GPU required.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
go run ./cmd/vramwatch watch --source demo --color @args

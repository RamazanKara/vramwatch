#!/bin/sh
# Live demo: a synthetic 24 GiB card whose KV cache grows until it OOMs.
# Great for recording the README GIF. No GPU required.
set -eu
cd "$(dirname "$0")/.."
go run ./cmd/vramwatch watch --source demo --color "$@"

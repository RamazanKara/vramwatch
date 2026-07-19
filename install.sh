#!/bin/sh
# vramwatch installer.
#
#   curl -fsSL https://raw.githubusercontent.com/RamazanKara/vramwatch/main/install.sh | sh
#
# Downloads the latest release binary for your OS/arch and installs it into
# INSTALL_DIR (default: /usr/local/bin, or ~/.local/bin without write access).
set -eu

REPO="RamazanKara/vramwatch"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

say()  { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$os" in
  linux)  os="linux" ;;
  darwin) os="darwin" ;;
  *) die "unsupported OS '$os'. Windows users: download the .zip from https://github.com/$REPO/releases" ;;
esac

arch="$(uname -m)"
case "$arch" in
  x86_64|amd64) arch="amd64" ;;
  aarch64|arm64) arch="arm64" ;;
  *) die "unsupported architecture '$arch'" ;;
esac

# Resolve the latest release tag.
api="https://api.github.com/repos/$REPO/releases/latest"
if have curl; then
  tag="$(curl -fsSL "$api" | grep -m1 '"tag_name"' | cut -d'"' -f4)"
elif have wget; then
  tag="$(wget -qO- "$api" | grep -m1 '"tag_name"' | cut -d'"' -f4)"
else
  die "need curl or wget"
fi
[ -n "$tag" ] || die "could not determine the latest release tag"

asset="vramwatch_${tag}_${os}_${arch}.tar.gz"
url="https://github.com/$REPO/releases/download/$tag/$asset"
checksums_url="https://github.com/$REPO/releases/download/$tag/checksums.txt"
say "downloading $asset ..."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if have curl; then
  curl -fsSL "$url" -o "$tmp/$asset" || die "download failed: $url"
  curl -fsSL "$checksums_url" -o "$tmp/checksums.txt" || die "checksum download failed: $checksums_url"
else
  wget -qO "$tmp/$asset" "$url" || die "download failed: $url"
  wget -qO "$tmp/checksums.txt" "$checksums_url" || die "checksum download failed: $checksums_url"
fi

expected="$(awk -v file="$asset" '$2 == file { print $1; exit }' "$tmp/checksums.txt")"
[ -n "$expected" ] || die "$asset is missing from checksums.txt"
if have sha256sum; then
  actual="$(sha256sum "$tmp/$asset" | awk '{ print $1 }')"
elif have shasum; then
  actual="$(shasum -a 256 "$tmp/$asset" | awk '{ print $1 }')"
else
  die "need sha256sum or shasum to verify the release"
fi
[ "$actual" = "$expected" ] || die "checksum mismatch for $asset"
say "checksum verified"

tar -xzf "$tmp/$asset" -C "$tmp" vramwatch || die "extract failed"

# Choose a writable install dir. Try the privileged dir first (non-interactive
# sudo so a piped `curl | sh` never blocks on a password prompt); on any
# failure (no sudo, not a sudoer, or a password is required) fall back to ~/.local/bin.
if [ -w "$INSTALL_DIR" ] || [ "$(id -u)" -eq 0 ]; then
  install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch"
elif have sudo && sudo -n install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch" 2>/dev/null; then
  say "installed to $INSTALL_DIR (via sudo)"
else
  INSTALL_DIR="$HOME/.local/bin"
  mkdir -p "$INSTALL_DIR"
  install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch"
  say "installed to $INSTALL_DIR (add it to your PATH)"
fi

say "installed vramwatch $tag -> $INSTALL_DIR/vramwatch"
say "run: vramwatch watch"

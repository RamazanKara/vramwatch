#!/bin/sh
# vramwatch installer.
#
#   curl -fsSL https://raw.githubusercontent.com/RamazanKara/vramwatch/master/install.sh | sh
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
say "downloading $asset ..."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if have curl; then
  curl -fsSL "$url" -o "$tmp/$asset" || die "download failed: $url"
else
  wget -qO "$tmp/$asset" "$url" || die "download failed: $url"
fi

tar -xzf "$tmp/$asset" -C "$tmp" vramwatch || die "extract failed"

# Choose a writable install dir.
if [ ! -w "$INSTALL_DIR" ] && [ "$(id -u)" -ne 0 ]; then
  if have sudo; then
    say "installing to $INSTALL_DIR (via sudo)"
    sudo install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch"
  else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch"
    say "installed to $INSTALL_DIR (add it to your PATH)"
  fi
else
  install -m 0755 "$tmp/vramwatch" "$INSTALL_DIR/vramwatch"
fi

say "installed vramwatch $tag -> $INSTALL_DIR/vramwatch"
say "run: vramwatch watch"

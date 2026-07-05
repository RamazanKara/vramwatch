//go:build !linux

package gpu

import "github.com/RamazanKara/vramwatch/internal/model"

// procVRAM has no /proc to read outside Linux; per-process VRAM for AMD/Intel is
// not collected on this platform.
func procVRAM() map[string][]model.Proc { return nil }

//go:build linux

package gpu

import "github.com/RamazanKara/vramwatch/internal/model"

// procVRAM reads per-process VRAM from the kernel's /proc DRM fdinfo interface.
func procVRAM() map[string][]model.Proc { return procVRAMIn("/proc") }

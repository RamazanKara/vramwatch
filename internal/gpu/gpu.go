// Package gpu discovers GPUs and their device-level VRAM usage by shelling out
// to the vendor tools (nvidia-smi, rocm-smi). Parsing is split into pure
// functions so it can be unit-tested against captured fixtures without a GPU.
package gpu

import (
	"context"
	"os/exec"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// Provider samples device-level VRAM for one GPU vendor.
type Provider interface {
	// Name is the human label for the provider (e.g. "nvidia-smi").
	Name() string
	// Vendor is the GPU vendor this provider reports.
	Vendor() model.Vendor
	// Available reports whether the provider can run on this host.
	Available(ctx context.Context) bool
	// Sample returns the current per-device VRAM state.
	Sample(ctx context.Context) ([]model.GPU, error)
}

// All returns every built-in provider.
func All() []Provider { return []Provider{&Nvidia{}, &AMD{}} }

// DetectAvailable returns the providers usable on this host.
func DetectAvailable(ctx context.Context) []Provider {
	var out []Provider
	for _, p := range All() {
		if p.Available(ctx) {
			out = append(out, p)
		}
	}
	return out
}

// Sample runs every available provider and concatenates their GPUs, then
// augments non-NVIDIA devices with per-process VRAM from /proc fdinfo (Linux).
func Sample(ctx context.Context) ([]model.GPU, error) {
	var gpus []model.GPU
	for _, p := range DetectAvailable(ctx) {
		g, err := p.Sample(ctx)
		if err != nil {
			continue // a failing vendor tool shouldn't blank out the others
		}
		gpus = append(gpus, g...)
	}
	attachProcs(gpus, procVRAM())
	return gpus, nil
}

// attachProcs fills per-process VRAM for AMD/Intel/unknown devices that don't
// already have it (NVIDIA gets it from nvidia-smi). It matches a device to a
// /proc-fdinfo PCI address when known; if there's a single such device and a
// single DRM device in the data, it attaches directly.
func attachProcs(gpus []model.GPU, byPdev map[string][]model.Proc) {
	if len(byPdev) == 0 {
		return
	}
	var targets []int
	for i := range gpus {
		if gpus[i].Vendor != model.VendorNVIDIA && len(gpus[i].Procs) == 0 {
			targets = append(targets, i)
		}
	}
	if len(targets) == 0 {
		return
	}
	unmatched := 0
	for _, i := range targets {
		if procs, ok := byPdev[normalizePCI(gpus[i].PCIBus)]; ok && gpus[i].PCIBus != "" {
			gpus[i].Procs = procs
		} else {
			unmatched++
		}
	}
	// Single ambiguous device + single DRM device: attach directly.
	if unmatched == 1 && len(targets) == 1 && len(byPdev) == 1 {
		for _, procs := range byPdev {
			gpus[targets[0]].Procs = procs
		}
	}
}

// normalizePCI canonicalises a PCI address to the full "0000:03:00.0" form.
func normalizePCI(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return ""
	}
	if strings.Count(s, ":") == 1 { // "03:00.0" is missing the domain
		s = "0000:" + s
	}
	return s
}

// run executes a command with a timeout and returns combined stdout.
func run(ctx context.Context, name string, args ...string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, name, args...).Output()
	return string(out), err
}

// lookPath reports whether an executable is on PATH.
func lookPath(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func trimField(s string) string { return strings.TrimSpace(s) }

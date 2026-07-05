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

// All returns every built-in provider, including any OS-specific ones.
func All() []Provider {
	return append([]Provider{&Nvidia{}, &AMD{}}, platformProviders()...)
}

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

// attachProcs fills per-process VRAM for AMD/unknown devices that don't already
// have it (NVIDIA gets it from nvidia-smi). It binds a device to /proc-fdinfo
// data strictly by PCI address; a GPU whose known PCI address doesn't appear in
// the data is left alone rather than being given a foreign device's processes.
func attachProcs(gpus []model.GPU, byPdev map[string][]model.Proc) {
	if len(byPdev) == 0 {
		return
	}
	// PCI addresses already claimed by a device, so the fallback never hands one
	// GPU's processes to another.
	owned := map[string]bool{}
	for i := range gpus {
		if b := normalizePCI(gpus[i].PCIBus); b != "" {
			owned[b] = true
		}
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

	// 1. Exact PCI match.
	var unmatched []int
	for _, i := range targets {
		if b := normalizePCI(gpus[i].PCIBus); b != "" {
			if procs, ok := byPdev[b]; ok {
				gpus[i].Procs = procs
				continue
			}
		}
		unmatched = append(unmatched, i)
	}

	// 2. Fallback ONLY for a single device whose PCI address is unknown (so it
	//    couldn't match), when there's exactly one DRM device in the data and
	//    that device isn't owned by another GPU.
	if len(unmatched) == 1 && len(byPdev) == 1 {
		i := unmatched[0]
		if normalizePCI(gpus[i].PCIBus) == "" {
			for pdev, procs := range byPdev {
				if !owned[pdev] {
					gpus[i].Procs = procs
				}
			}
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

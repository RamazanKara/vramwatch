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

// Sample runs every available provider and concatenates their GPUs, reindexing
// so device indices are globally unique in display order.
func Sample(ctx context.Context) ([]model.GPU, error) {
	var gpus []model.GPU
	for _, p := range DetectAvailable(ctx) {
		g, err := p.Sample(ctx)
		if err != nil {
			continue // a failing vendor tool shouldn't blank out the others
		}
		gpus = append(gpus, g...)
	}
	return gpus, nil
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

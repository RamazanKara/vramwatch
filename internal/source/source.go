// Package source provides the observation inputs to the engine: a live source
// that composes the GPU and loader providers, and a mock source that replays a
// captured scenario file (used for demos, tests, and CI without a GPU).
package source

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/gpu"
	"github.com/RamazanKara/vramwatch/internal/loader"
	"github.com/RamazanKara/vramwatch/internal/model"
)

// Source yields raw GPU and loader observations for the engine to attribute.
type Source interface {
	Collect(ctx context.Context) (gpus []model.GPU, models []model.LoaderModel, err error)
	// Describe names the source for display.
	Describe() string
}

// Live composes the auto-detected GPU and loader providers.
type Live struct{}

func (Live) Collect(ctx context.Context) ([]model.GPU, []model.LoaderModel, error) {
	gpus, _ := gpu.Sample(ctx)
	models, _ := loader.Models(ctx)
	return gpus, models, nil
}

func (Live) Describe() string { return "live (nvidia-smi/rocm-smi + ollama/llama.cpp)" }

// Scenario is the on-disk format a Mock source replays.
type Scenario struct {
	GPUs   []model.GPU         `json:"gpus"`
	Models []model.LoaderModel `json:"models"`
}

// Mock replays a fixed Scenario.
type Mock struct {
	Path     string
	Scenario Scenario
}

func (m *Mock) Collect(ctx context.Context) ([]model.GPU, []model.LoaderModel, error) {
	return m.Scenario.GPUs, m.Scenario.Models, nil
}

func (m *Mock) Describe() string {
	if m.Path != "" {
		return "mock:" + m.Path
	}
	return "mock"
}

// LoadMock reads a scenario JSON file into a Mock source.
func LoadMock(path string) (*Mock, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s Scenario
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parse scenario %s: %w", path, err)
	}
	// Default unspecified GPU indices to their slice position.
	for i := range s.GPUs {
		if s.GPUs[i].Vendor == "" {
			s.GPUs[i].Vendor = model.VendorUnknown
		}
	}
	return &Mock{Path: path, Scenario: s}, nil
}

// Demo synthesises a 24 GiB card running an 8B model whose KV cache grows with
// wall-clock time, looping. It exists so `vramwatch watch --source demo` shows
// the headline behaviour (the KV cache eating free VRAM until OOM) with no
// GPU present. It is a demo aid, never a measurement.
type Demo struct {
	Start time.Time
}

func (d Demo) Describe() string { return "demo (synthetic KV-cache growth)" }

func (d Demo) Collect(ctx context.Context) ([]model.GPU, []model.LoaderModel, error) {
	start := d.Start
	if start.IsZero() {
		start = time.Now()
	}
	const loop = 24 * time.Second
	elapsed := time.Since(start) % loop
	frac := float64(elapsed) / float64(loop)
	// Ease-in so the cache accelerates into the danger zone; 2k -> ~180k tokens.
	tokens := 2048 + int(frac*frac*(180000-2048))

	const (
		total    = 24 * model.GiB
		weights  = 5600 * model.MiB // ~8B at q5
		overhead = 800 * model.MiB
		other    = 1024 * model.MiB // desktop/compositor
	)
	arch := model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16}
	kv := uint64(2*32*8*128*16/8) * uint64(tokens)
	vram := uint64(weights) + kv + uint64(overhead)
	used := vram + uint64(other)
	if used > total {
		used = total
	}
	free := uint64(total) - used

	gpu := model.GPU{
		Index: 0, Name: "AMD Radeon RX 7900 XTX (demo)", Vendor: model.VendorAMD, Driver: "6.7.0",
		TotalBytes: total, UsedBytes: used, FreeBytes: free,
		Procs: []model.Proc{{PID: 4242, Name: "ollama", UsedBytes: vram}},
	}
	m := model.LoaderModel{
		Loader: "ollama", Name: "llama3:8b-demo", PID: 4242, GPUIndex: 0,
		VRAMBytes: vram, ContextTokens: tokens, ContextMax: 200000, Arch: arch, Estimated: true,
	}
	return []model.GPU{gpu}, []model.LoaderModel{m}, nil
}

// FromSpec resolves a --source flag value to a Source. Recognised forms:
//
//	""            -> Live
//	"live"        -> Live
//	"demo"        -> Demo (synthetic growing KV cache)
//	"mock:PATH"   -> Mock reading PATH
//	"PATH.json"   -> Mock reading PATH
func FromSpec(spec string) (Source, error) {
	spec = strings.TrimSpace(spec)
	switch {
	case spec == "" || spec == "live":
		return Live{}, nil
	case spec == "demo":
		return Demo{Start: time.Now()}, nil
	case strings.HasPrefix(spec, "mock:"):
		return LoadMock(strings.TrimPrefix(spec, "mock:"))
	case strings.HasSuffix(spec, ".json"):
		return LoadMock(spec)
	default:
		return nil, fmt.Errorf("unrecognised --source %q (use 'live', 'demo', 'mock:PATH', or a .json path)", spec)
	}
}

//go:build windows

package gpu

import (
	"context"
	"sort"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// winGPUClassKey is the registry Class key for display adapters. Its subkeys
// (0000, 0001, …) hold each adapter's real VRAM size (qwMemorySize), device id
// (vendor), and description (name).
const winGPUClassKey = `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`

// Windows reports GPU VRAM on Windows, where the vendor CLIs (rocm-smi) usually
// aren't present. Total VRAM comes from the registry; usage from the built-in
// "GPU Adapter Memory" performance counter via typeperf. NVIDIA is left to
// nvidia-smi (which ships with its Windows driver) to avoid double-counting.
type Windows struct{}

func (Windows) Name() string                       { return "windows-gpu" }
func (Windows) Vendor() model.Vendor               { return model.VendorUnknown }
func (Windows) Available(ctx context.Context) bool { return lookPath("typeperf") }

func (Windows) Sample(ctx context.Context) ([]model.GPU, error) {
	qw := regValues(ctx, "HardwareInformation.qwMemorySize")
	if len(qw) == 0 {
		return nil, nil
	}
	dev := regValues(ctx, "MatchingDeviceId")
	desc := regValues(ctx, "DriverDesc")

	subkeys := make([]string, 0, len(qw))
	for k := range qw {
		subkeys = append(subkeys, k)
	}
	sort.Strings(subkeys)

	var gpus []model.GPU
	for _, sk := range subkeys {
		total := parseRegUint(qw[sk])
		if total == 0 {
			continue // no dedicated VRAM (virtual display, or an integrated GPU)
		}
		name := desc[sk]
		vendor := vendorFromDeviceID(dev[sk])
		if vendor == model.VendorNVIDIA || looksNVIDIA(name) {
			continue // NVIDIA is reported by nvidia-smi; don't double-count
		}
		if name == "" {
			name = "GPU"
		}
		// FreeBytes defaults to the whole card; usage is filled in below only
		// when it can be attributed to a device unambiguously.
		gpus = append(gpus, model.GPU{Index: len(gpus), Name: name, Vendor: vendor, TotalBytes: total, FreeBytes: total})
	}
	if len(gpus) == 0 {
		return nil, nil
	}

	// Device usage comes from the GPU Adapter Memory perf counter, keyed by an
	// opaque adapter LUID. We can only map it to a physical card unambiguously
	// when there is exactly one non-NVIDIA GPU (the common case). With several,
	// there is no reliable LUID<->registry join, so usage is left unknown (the
	// card reports full free) rather than guessed onto the wrong device.
	if len(gpus) == 1 {
		if out, err := run(ctx, "typeperf", `\GPU Adapter Memory(*)\Dedicated Usage`, "-sc", "1"); err == nil {
			var used uint64
			for _, u := range parseTypeperfAdapter(out) {
				if u > used {
					used = u // the real GPU dominates; software adapters report ~0
				}
			}
			if used > gpus[0].TotalBytes {
				used = gpus[0].TotalBytes
			}
			gpus[0].UsedBytes = used
			gpus[0].FreeBytes = gpus[0].TotalBytes - used
		}
	}
	return gpus, nil
}

func regValues(ctx context.Context, name string) map[string]string {
	out, err := run(ctx, "reg", "query", winGPUClassKey, "/s", "/v", name)
	if err != nil {
		return nil
	}
	return parseRegValues(out, name)
}

func platformProviders() []Provider { return []Provider{Windows{}} }

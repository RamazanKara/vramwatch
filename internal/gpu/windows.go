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
			continue
		}
		vendor := vendorFromDeviceID(dev[sk])
		if vendor == model.VendorNVIDIA {
			continue // handled by nvidia-smi
		}
		name := desc[sk]
		if name == "" {
			name = "GPU"
		}
		gpus = append(gpus, model.GPU{Index: len(gpus), Name: name, Vendor: vendor, TotalBytes: total})
	}
	if len(gpus) == 0 {
		return nil, nil
	}

	// Device usage from the GPU Adapter Memory perf counter, keyed by LUID.
	usage := map[string]uint64{}
	if out, err := run(ctx, "typeperf", `\GPU Adapter Memory(*)\Dedicated Usage`, "-sc", "1"); err == nil {
		usage = parseTypeperfAdapter(out)
	}
	// Pair each GPU with an adapter LUID by usage (highest first). This is exact
	// for a single GPU (the common case) and best-effort for multiple.
	luids := make([]string, 0, len(usage))
	for l := range usage {
		luids = append(luids, l)
	}
	sort.Slice(luids, func(i, j int) bool { return usage[luids[i]] > usage[luids[j]] })
	for i := range gpus {
		if i >= len(luids) {
			break
		}
		used := usage[luids[i]]
		if used > gpus[i].TotalBytes {
			used = gpus[i].TotalBytes
		}
		gpus[i].UsedBytes = used
		gpus[i].FreeBytes = gpus[i].TotalBytes - used
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

package gpu

import (
	"context"
	"encoding/json"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// AMD samples AMD GPUs via amd-smi, the AMD SMI CLI that supersedes rocm-smi.
// Identity + capacity come from `amd-smi static --json` and live VRAM usage from
// `amd-smi metric --mem-usage --json`; the two are joined on the per-GPU "gpu"
// index. Parsing is pure so it can be unit-tested against captured fixtures.
type AMD struct{}

func (AMD) Name() string                       { return "amd-smi" }
func (AMD) Vendor() model.Vendor               { return model.VendorAMD }
func (AMD) Available(ctx context.Context) bool { return lookPath("amd-smi") }

func (AMD) Sample(ctx context.Context) ([]model.GPU, error) {
	// static is the device set + identity + capacity; metric adds live usage.
	staticOut, serr := run(ctx, "amd-smi", "static", "--json")
	if strings.TrimSpace(staticOut) == "" {
		if serr != nil {
			return nil, serr
		}
		return nil, nil
	}
	// metric is best-effort: amd-smi can fail some queries (e.g. on Windows or an
	// unsupported card) yet still describe the device. Missing usage leaves used=0
	// rather than dropping the GPU.
	metricOut, _ := run(ctx, "amd-smi", "metric", "--mem-usage", "--json")

	gpus, perr := parseAMDSMI(staticOut, metricOut)
	if perr != nil {
		if serr != nil {
			return nil, serr // the command failed; don't report a parse of its garbage
		}
		return nil, perr
	}
	return gpus, nil
}

// amdUsage is the live VRAM usage for one GPU, from `amd-smi metric`.
type amdUsage struct {
	total, used, free uint64
}

// parseAMDSMI merges `amd-smi static --json` (identity + capacity) with
// `amd-smi metric --mem-usage --json` (live usage), joined on the "gpu" index.
// The metric output is optional; when a GPU has no metric entry, usage is zero.
func parseAMDSMI(staticJSON, metricJSON string) ([]model.GPU, error) {
	var arr []map[string]json.RawMessage
	if err := json.Unmarshal([]byte(staticJSON), &arr); err != nil {
		return nil, err
	}
	usage := parseAMDMetric(metricJSON)

	var gpus []model.GPU
	for _, el := range arr {
		idx, ok := amdInt(el["gpu"])
		if !ok {
			continue // not a per-GPU element
		}
		u := usage[idx] // zero value if this GPU had no metric entry

		total := u.total
		if total == 0 {
			total = amdVRAMSize(el["vram"]) // static capacity fallback
		}
		used := u.used
		free := u.free
		// Fill a missing side of the used/free/total triangle when we know total
		// and one of the other two (amd-smi can report "N/A" for either).
		if total > 0 {
			if used == 0 && free > 0 && free <= total {
				used = total - free
			}
			if free == 0 && total >= used {
				free = total - used
			}
		}

		name := amdBlockString(el["asic"], "market_name")
		driver := amdBlockString(el["driver"], "version")
		// Drop only a truly empty entry: no VRAM numbers AND no identity. amd-smi
		// (notably on Windows) can collapse vram/mem_usage to "N/A" while still
		// naming a real card, so a card with any VRAM signal or a real name/driver
		// is kept — with an unknown (TotalBytes=0) capacity if need be — rather
		// than vanishing from the report.
		if total == 0 && used == 0 && free == 0 && name == "" && driver == "" {
			continue
		}
		if name == "" {
			name = "AMD GPU " + strconv.Itoa(idx)
		}
		gpus = append(gpus, model.GPU{
			Index:      idx,
			Name:       name,
			Vendor:     model.VendorAMD,
			Driver:     driver,
			PCIBus:     amdBlockString(el["bus"], "bdf"),
			TotalBytes: total,
			UsedBytes:  used,
			FreeBytes:  free,
		})
	}
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	return gpus, nil
}

// parseAMDMetric reads `amd-smi metric --mem-usage --json` into per-GPU usage,
// keyed by the "gpu" index. Any parse failure yields an empty map (best-effort).
func parseAMDMetric(metricJSON string) map[int]amdUsage {
	out := map[int]amdUsage{}
	if strings.TrimSpace(metricJSON) == "" {
		return out
	}
	var arr []map[string]json.RawMessage
	if err := json.Unmarshal([]byte(metricJSON), &arr); err != nil {
		return out
	}
	for _, el := range arr {
		idx, ok := amdInt(el["gpu"])
		if !ok {
			continue
		}
		var mem map[string]json.RawMessage
		if json.Unmarshal(el["mem_usage"], &mem) != nil {
			continue // mem_usage absent or collapsed to the string "N/A"
		}
		out[idx] = amdUsage{
			total: amdBytes(mem["total_vram"]),
			used:  amdBytes(mem["used_vram"]),
			free:  amdBytes(mem["free_vram"]),
		}
	}
	return out
}

// amdInt reads an integer leaf (the "gpu" index). amd-smi normally emits a bare
// int, but some builds render it as a JSON float (0.0) or a quoted string ("0");
// tolerate both, matching the leniency amdBytes uses for value leaves. A
// non-numeric value ("N/A") or null is rejected.
func amdInt(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var num json.Number
	if json.Unmarshal(raw, &num) != nil {
		return 0, false
	}
	if i, err := num.Int64(); err == nil {
		return int(i), true
	}
	f, err := num.Float64()
	if err != nil {
		return 0, false
	}
	return int(f), true
}

// amdBlockString reads a string field from a block that amd-smi may render as an
// object OR collapse to the bare string "N/A" when the underlying query fails.
// Returns "" if the block isn't an object, the key is absent, or the value is
// "N/A".
func amdBlockString(block json.RawMessage, key string) string {
	if len(block) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(block, &m) != nil {
		return "" // block was "N/A" or otherwise not an object
	}
	var s string
	if json.Unmarshal(m[key], &s) != nil {
		return ""
	}
	s = strings.TrimSpace(s)
	if s == "N/A" {
		return ""
	}
	return s
}

// amdVRAMSize reads the static vram.size ({value,unit}) leaf into bytes. The vram
// block can itself be the bare string "N/A".
func amdVRAMSize(vramBlock json.RawMessage) uint64 {
	if len(vramBlock) == 0 {
		return 0
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(vramBlock, &m) != nil {
		return 0
	}
	return amdBytes(m["size"])
}

// amdBytes normalizes an amd-smi numeric leaf to bytes. The leaf is either a
// {"value": N, "unit": "MB"} object, a bare number (very old builds), or the
// string "N/A" (missing). amd-smi labels VRAM "MB" but the value is actually
// MiB-magnitude (bytes/1048576), so MB and MiB both scale by 1<<20.
func amdBytes(raw json.RawMessage) uint64 {
	if len(raw) == 0 {
		return 0
	}
	// Bare number (old builds emit an int with no unit wrapper).
	var num json.Number
	if json.Unmarshal(raw, &num) == nil {
		return amdScale(num, "MB")
	}
	// {value, unit} object.
	var vu struct {
		Value json.Number `json:"value"`
		Unit  string      `json:"unit"`
	}
	if json.Unmarshal(raw, &vu) == nil && vu.Value != "" {
		return amdScale(vu.Value, vu.Unit)
	}
	return 0 // "N/A" or an unexpected shape
}

// maxPlausibleVRAM caps a single GPU's VRAM at 1 PiB; a larger figure is a
// garbage amd-smi leaf (Windows/error paths), not a real capacity.
const maxPlausibleVRAM = float64(uint64(1) << 50)

// amdScale converts a value + unit to bytes. amd-smi's VRAM "MB" is really MiB,
// so MB and MiB are treated identically (1<<20). A non-finite or implausibly
// large result is rejected (returns 0) rather than saturating uint64.
func amdScale(n json.Number, unit string) uint64 {
	f, err := n.Float64()
	if err != nil || f < 0 || math.IsNaN(f) {
		return 0
	}
	var mult float64
	switch strings.ToUpper(strings.TrimSpace(unit)) {
	case "B", "BYTES":
		mult = 1
	case "KB", "KIB":
		mult = 1 << 10
	case "GB", "GIB":
		mult = 1 << 30
	case "TB", "TIB":
		mult = 1 << 40
	default: // "MB"/"MIB"/unknown: amd-smi VRAM values are MiB-magnitude
		mult = 1 << 20
	}
	bytes := f * mult
	if math.IsInf(bytes, 0) || math.IsNaN(bytes) || bytes >= maxPlausibleVRAM {
		return 0
	}
	return uint64(bytes)
}

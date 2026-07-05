package gpu

import (
	"context"
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// AMD samples AMD GPUs via rocm-smi's JSON output.
type AMD struct{}

func (AMD) Name() string                       { return "rocm-smi" }
func (AMD) Vendor() model.Vendor               { return model.VendorAMD }
func (AMD) Available(ctx context.Context) bool { return lookPath("rocm-smi") }

func (AMD) Sample(ctx context.Context) ([]model.GPU, error) {
	out, err := run(ctx, "rocm-smi", "--showmeminfo", "vram", "--showproductname", "--showdriverversion", "--json")
	if err != nil {
		return nil, err
	}
	return parseROCm(out)
}

// parseROCm parses rocm-smi --json output. Values are strings; keys and their
// spelling drift across ROCm versions, so lookups try several spellings.
func parseROCm(jsonStr string) ([]model.GPU, error) {
	var raw map[string]map[string]string
	if err := json.Unmarshal([]byte(jsonStr), &raw); err != nil {
		return nil, err
	}
	systemDriver := getFirst(raw["system"], "Driver version", "Driver Version")

	// Stable ordering by card number.
	var keys []string
	for k := range raw {
		if strings.HasPrefix(k, "card") {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return cardNum(keys[i]) < cardNum(keys[j]) })

	var gpus []model.GPU
	for _, k := range keys {
		card := raw[k]
		total := parseUint(getFirst(card, "VRAM Total Memory (B)", "VRAM Total Memory (b)", "vram_total"))
		used := parseUint(getFirst(card, "VRAM Total Used Memory (B)", "VRAM Total Used Memory (b)", "vram_used"))
		name := getFirst(card, "Card Series", "Card series", "Card Model", "Card model", "GPU ID")
		if name == "" {
			name = "AMD GPU " + strconv.Itoa(cardNum(k))
		}
		driver := getFirst(card, "Driver version", "Driver Version")
		if driver == "" {
			driver = systemDriver
		}
		var free uint64
		if total >= used {
			free = total - used
		}
		gpus = append(gpus, model.GPU{
			Index:      cardNum(k),
			Name:       name,
			Vendor:     model.VendorAMD,
			Driver:     driver,
			TotalBytes: total,
			UsedBytes:  used,
			FreeBytes:  free,
		})
	}
	return gpus, nil
}

func getFirst(m map[string]string, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k]; ok && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func cardNum(k string) int {
	return atoiDefault(strings.TrimPrefix(k, "card"), 0)
}

func parseUint(s string) uint64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	n, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		// tolerate floats or trailing units (e.g. "1.5 GB")
		fields := strings.Fields(s)
		if len(fields) == 0 {
			return 0
		}
		if f, ferr := strconv.ParseFloat(fields[0], 64); ferr == nil && f >= 0 {
			return uint64(f)
		}
		return 0
	}
	return n
}

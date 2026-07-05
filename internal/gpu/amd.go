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
	out, err := run(ctx, "rocm-smi", "--showmeminfo", "vram", "--showproductname", "--showdriverversion", "--showbus", "--json")
	if err != nil {
		return nil, err
	}
	return parseROCm(out)
}

// parseROCm parses rocm-smi --json output. Values are strings; keys and their
// spelling drift across ROCm versions, so lookups try several spellings.
func parseROCm(jsonStr string) ([]model.GPU, error) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonStr), &doc); err != nil {
		return nil, err
	}
	// rocm-smi groups output as {"card0": {...}, "system": {...}}. Decode each
	// block leniently into a flat string map: a nested object/array value (ROCm
	// 6/7 emit these, e.g. "GPU Metrics" or MI300 partition info) is skipped
	// rather than making json.Unmarshal fail for the whole document and dropping
	// every GPU. A bare number is coerced to its string form so a future numeric
	// value still parses.
	raw := make(map[string]map[string]string, len(doc))
	for block, blob := range doc {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(blob, &fields); err != nil {
			continue // block isn't an object (unexpected shape) — ignore it
		}
		m := make(map[string]string, len(fields))
		for k, v := range fields {
			var s string
			if err := json.Unmarshal(v, &s); err == nil {
				m[k] = s
				continue
			}
			var num json.Number
			if err := json.Unmarshal(v, &num); err == nil {
				m[k] = num.String()
			}
		}
		raw[block] = m
	}
	systemDriver := getFirst(raw["system"], "Driver version", "Driver Version")

	// Stable ordering by card number. Only real card blocks (cardN) are GPUs;
	// a sibling like "system" or a stray key is not.
	var keys []string
	for k := range raw {
		if isCardKey(k) {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return cardNum(keys[i]) < cardNum(keys[j]) })

	var gpus []model.GPU
	for _, k := range keys {
		card := raw[k]
		total := parseUint(getFirst(card, "VRAM Total Memory (B)", "VRAM Total Memory (b)", "vram_total"))
		used := parseUint(getFirst(card, "VRAM Total Used Memory (B)", "VRAM Total Used Memory (b)", "vram_used"))
		if total == 0 {
			continue // no meminfo: a masked/headless/failed-probe entry, not a usable GPU
		}
		// Name from a genuine product-name key only. "Card Model"/"GPU ID" are hex
		// device ids (e.g. 0x744c), so they are NOT names — fall through to the
		// clean "AMD GPU N" default instead of showing a hex id.
		name := getFirst(card, "Card Series", "Card series", "card_series", "Market Name", "market_name", "Device Name", "device_name")
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
			PCIBus:     getFirst(card, "PCI Bus", "pci_bus", "PCI Bus ID"),
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

// isCardKey reports whether a top-level rocm-smi key names a real GPU (cardN),
// not a sibling block like "system" or some other stray key.
func isCardKey(k string) bool {
	rest := strings.TrimPrefix(k, "card")
	if rest == k || rest == "" {
		return false
	}
	for _, r := range rest {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
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

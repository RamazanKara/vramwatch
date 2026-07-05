package gpu

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// procVRAMIn reads per-PCI-device, per-PID VRAM usage from a /proc-style tree.
// It parses the vendor-neutral DRM fdinfo interface (amdgpu, i915, …), which is
// how AMD and Intel per-process VRAM is obtained. The root is injectable so the
// walk is fully testable against a fixture tree.
func procVRAMIn(root string) map[string][]model.Proc {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	byDev := map[string]map[int]uint64{} // pdev -> pid -> vram (deduped by client id)
	names := map[int]string{}

	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || pid <= 0 {
			continue
		}
		procDir := filepath.Join(root, e.Name())
		fds, err := os.ReadDir(filepath.Join(procDir, "fdinfo"))
		if err != nil {
			continue // not readable (permissions) or no fds
		}
		seen := map[string]bool{}
		hit := false
		for _, fd := range fds {
			data, err := os.ReadFile(filepath.Join(procDir, "fdinfo", fd.Name()))
			if err != nil {
				continue
			}
			pdev, client, vram, isDRM := parseFdinfo(string(data))
			if !isDRM || vram == 0 || pdev == "" {
				continue
			}
			if client != "" {
				key := pdev + "/" + client
				if seen[key] {
					continue // same client id already counted (dup'd fd)
				}
				seen[key] = true
			}
			if byDev[pdev] == nil {
				byDev[pdev] = map[int]uint64{}
			}
			byDev[pdev][pid] += vram
			hit = true
		}
		if hit {
			names[pid] = readComm(procDir)
		}
	}

	out := map[string][]model.Proc{}
	for pdev, pids := range byDev {
		for pid, vram := range pids {
			out[pdev] = append(out[pdev], model.Proc{PID: pid, Name: names[pid], UsedBytes: vram})
		}
	}
	return out
}

func readComm(procDir string) string {
	b, err := os.ReadFile(filepath.Join(procDir, "comm"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// parseFdinfo extracts the DRM device (pdev), client id, and VRAM usage from a
// single /proc/<pid>/fdinfo/<fd> file. isDRM is false for non-DRM fds.
func parseFdinfo(content string) (pdev, client string, vram uint64, isDRM bool) {
	for _, line := range strings.Split(content, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "drm-driver":
			isDRM = true
		case "drm-pdev":
			pdev = normalizePCI(val)
		case "drm-client-id":
			client = val
		case "drm-resident-vram", "drm-memory-vram", "drm-total-vram":
			if v := parseDRMBytes(val); v > vram {
				vram = v // keep the largest of the reported VRAM keys
			}
		}
	}
	return pdev, client, vram, isDRM
}

// parseDRMBytes parses a fdinfo memory value like "4096 KiB" into bytes.
func parseDRMBytes(s string) uint64 {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0
	}
	n, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0
	}
	unit := "b"
	if len(fields) > 1 {
		unit = strings.ToLower(fields[1])
	}
	switch unit {
	case "kib", "kb":
		return n * model.KiB
	case "mib", "mb":
		return n * model.MiB
	case "gib", "gb":
		return n * model.GiB
	default:
		return n
	}
}

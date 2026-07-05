package gpu

import (
	"math"
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
	// pdev -> pid -> client -> vram. Deduplicating by DRM client id makes a
	// process's dup'd fds count once; distinct clients are summed. When a driver
	// omits drm-client-id, all its client-less fds collapse to one pseudo-client
	// so they can't inflate the total.
	acc := map[string]map[int]map[string]uint64{}
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
			if client == "" {
				client = "-" // collapse client-less fds to one entry
			}
			if acc[pdev] == nil {
				acc[pdev] = map[int]map[string]uint64{}
			}
			if acc[pdev][pid] == nil {
				acc[pdev][pid] = map[string]uint64{}
			}
			if vram > acc[pdev][pid][client] {
				acc[pdev][pid][client] = vram // max within a client (dup'd fds)
			}
			hit = true
		}
		if hit {
			names[pid] = readComm(procDir)
		}
	}

	out := map[string][]model.Proc{}
	for pdev, pids := range acc {
		for pid, clients := range pids {
			var sum uint64
			for _, v := range clients {
				sum += v
			}
			out[pdev] = append(out[pdev], model.Proc{PID: pid, Name: names[pid], UsedBytes: sum})
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
//
// The VRAM keys are NOT interchangeable: drm-resident-vram is physical residency,
// drm-total-vram is an upper bound that includes evicted buffers. We report
// residency, preferring drm-resident-vram, then the legacy drm-memory-vram, and
// only falling back to drm-total-vram when neither is present.
func parseFdinfo(content string) (pdev, client string, vram uint64, isDRM bool) {
	var resident, memory, total uint64
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
		case "drm-resident-vram":
			resident = parseDRMBytes(val)
		case "drm-memory-vram":
			memory = parseDRMBytes(val)
		case "drm-total-vram":
			total = parseDRMBytes(val)
		}
	}
	switch {
	case resident > 0:
		vram = resident
	case memory > 0:
		vram = memory
	default:
		vram = total
	}
	return pdev, client, vram, isDRM
}

// parseDRMBytes parses a fdinfo memory value like "4096 KiB" into bytes,
// saturating instead of wrapping on overflow.
func parseDRMBytes(s string) uint64 {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0
	}
	n, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0
	}
	var mult uint64 = 1
	if len(fields) > 1 {
		switch strings.ToLower(fields[1]) {
		case "kib", "kb":
			mult = model.KiB
		case "mib", "mb":
			mult = model.MiB
		case "gib", "gb":
			mult = model.GiB
		}
	}
	if mult > 1 && n > math.MaxUint64/mult {
		return math.MaxUint64 // saturate rather than wrap
	}
	return n * mult
}

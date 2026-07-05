package gpu

import (
	"context"
	"strconv"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// Nvidia samples NVIDIA GPUs via nvidia-smi.
type Nvidia struct{}

func (Nvidia) Name() string                       { return "nvidia-smi" }
func (Nvidia) Vendor() model.Vendor               { return model.VendorNVIDIA }
func (Nvidia) Available(ctx context.Context) bool { return lookPath("nvidia-smi") }

const (
	nvGPUQuery  = "index,name,memory.total,memory.used,memory.free,driver_version,gpu_uuid"
	nvAppsQuery = "gpu_uuid,pid,process_name,used_memory"
)

func (Nvidia) Sample(ctx context.Context) ([]model.GPU, error) {
	gpuOut, err := run(ctx, "nvidia-smi", "--query-gpu="+nvGPUQuery, "--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	gpus, uuidToIdx := parseNvidiaGPUs(gpuOut)

	appsOut, err := run(ctx, "nvidia-smi", "--query-compute-apps="+nvAppsQuery, "--format=csv,noheader,nounits")
	if err == nil {
		parseNvidiaApps(appsOut, gpus, uuidToIdx)
	}
	return gpus, nil
}

// parseNvidiaGPUs parses the --query-gpu CSV (memory fields in MiB) into GPUs
// and a uuid->slice-index map for associating processes.
func parseNvidiaGPUs(csv string) ([]model.GPU, map[string]int) {
	var gpus []model.GPU
	idx := map[string]int{}
	for _, line := range strings.Split(csv, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := splitCSV(line)
		if len(f) < 7 {
			continue
		}
		g := model.GPU{
			Index:      atoiDefault(f[0], len(gpus)),
			Name:       f[1],
			Vendor:     model.VendorNVIDIA,
			TotalBytes: mibToBytes(f[2]),
			UsedBytes:  mibToBytes(f[3]),
			FreeBytes:  mibToBytes(f[4]),
			Driver:     f[5],
		}
		uuid := f[6]
		idx[uuid] = len(gpus)
		gpus = append(gpus, g)
	}
	return gpus, idx
}

// parseNvidiaApps parses the --query-compute-apps CSV (used_memory in MiB) and
// attaches processes to their GPU by uuid. The query is a fixed 4 columns
// (gpu_uuid, pid, process_name, used_memory) but nvidia-smi does not quote
// fields, so a process name containing a comma would shift a naive split. The
// first two and last columns are stable, so we anchor on those and rejoin the
// middle as the name.
func parseNvidiaApps(csv string, gpus []model.GPU, uuidToIdx map[string]int) {
	for _, line := range strings.Split(csv, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "no running") {
			continue
		}
		f := splitCSV(line)
		if len(f) < 4 {
			continue
		}
		i, ok := uuidToIdx[f[0]]
		if !ok {
			continue
		}
		last := len(f) - 1
		name := strings.Join(f[2:last], ",") // rejoin any commas in the process name
		gpus[i].Procs = append(gpus[i].Procs, model.Proc{
			PID:       atoiDefault(f[1], 0),
			Name:      name,
			UsedBytes: mibToBytes(f[last]),
		})
	}
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = trimField(parts[i])
	}
	return parts
}

// mibToBytes converts a possibly-noisy MiB string ("24564", "24564 MiB",
// "[N/A]") to bytes.
func mibToBytes(s string) uint64 {
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "MiB"))
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || f < 0 {
		return 0
	}
	return uint64(f * model.MiB)
}

func atoiDefault(s string, def int) int {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return def
	}
	return n
}

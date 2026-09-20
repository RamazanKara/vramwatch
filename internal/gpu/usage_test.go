package gpu

import (
	"fmt"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestNvidiaUsageProvenance(t *testing.T) {
	for _, tc := range []struct {
		name, used, free string
		known            bool
	}{
		{"unavailable", "[N/A]", "[N/A]", false},
		{"invalid", "NaN", "NaN", false},
		{"idle", "0", "8192", true},
		{"used only", "2048", "[N/A]", true},
		{"free only", "[N/A]", "6144", true},
		{"full", "[N/A]", "0", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gpus, _ := parseNvidiaGPUs(fmt.Sprintf("0, NVIDIA Test, 8192, %s, %s, 550, GPU-test", tc.used, tc.free))
			if len(gpus) != 1 {
				t.Fatalf("expected one GPU, got %d", len(gpus))
			}
			if got := gpus[0].UsageSource == model.ProvenanceMeasured; got != tc.known {
				t.Errorf("measured usage = %v, want %v: %+v", got, tc.known, gpus[0])
			}
			if tc.name == "full" && gpus[0].UsedBytes != 8192*model.MiB {
				t.Errorf("full GPU reported %d used bytes", gpus[0].UsedBytes)
			}
			if tc.name == "free only" && gpus[0].UsedBytes != 2048*model.MiB {
				t.Errorf("used bytes = %d, want %d", gpus[0].UsedBytes, 2048*model.MiB)
			}
		})
	}
}

func TestAMDUsageProvenance(t *testing.T) {
	const stat = `[{"gpu":0,"asic":{"market_name":"AMD Test"},"vram":{"size":8192}}]`
	for _, tc := range []struct {
		name, metric string
		known        bool
		used         uint64
	}{
		{"total only", `{"total_vram":8192}`, false, 0},
		{"unavailable", `{"total_vram":8192,"used_vram":"N/A","free_vram":"N/A"}`, false, 0},
		{"invalid", `{"total_vram":8192,"used_vram":-1,"free_vram":1e30}`, false, 0},
		{"idle", `{"used_vram":0}`, true, 0},
		{"used only", `{"used_vram":2048}`, true, 2048 * model.MiB},
		{"free only", `{"free_vram":6144}`, true, 2048 * model.MiB},
		{"full", `{"free_vram":0}`, true, 8192 * model.MiB},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gpus, err := parseAMDSMI(stat, `[{"gpu":0,"mem_usage":`+tc.metric+`}]`)
			if err != nil || len(gpus) != 1 {
				t.Fatalf("parse: gpus=%v err=%v", gpus, err)
			}
			if got := gpus[0].UsageSource == model.ProvenanceMeasured; got != tc.known {
				t.Errorf("measured usage = %v, want %v: %+v", got, tc.known, gpus[0])
			}
			if gpus[0].UsedBytes != tc.used {
				t.Errorf("used bytes = %d, want %d", gpus[0].UsedBytes, tc.used)
			}
		})
	}
}

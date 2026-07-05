package gpu

import (
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

const nvGPUFixture = `0, NVIDIA GeForce RTX 4090, 24564, 1876, 22688, 550.90.07, GPU-abc123
1, NVIDIA GeForce RTX 3090, 24576, 512, 24064, 550.90.07, GPU-def456`

const nvAppsFixture = `GPU-abc123, 4242, ollama, 22000
GPU-def456, 9001, python, 300`

func TestParseNvidiaGPUs(t *testing.T) {
	gpus, idx := parseNvidiaGPUs(nvGPUFixture)
	if len(gpus) != 2 {
		t.Fatalf("want 2 gpus, got %d", len(gpus))
	}
	if gpus[0].Name != "NVIDIA GeForce RTX 4090" {
		t.Errorf("name = %q", gpus[0].Name)
	}
	if gpus[0].TotalBytes != 24564*model.MiB {
		t.Errorf("total = %d", gpus[0].TotalBytes)
	}
	if gpus[0].Driver != "550.90.07" {
		t.Errorf("driver = %q", gpus[0].Driver)
	}
	if idx["GPU-abc123"] != 0 || idx["GPU-def456"] != 1 {
		t.Errorf("uuid index map wrong: %v", idx)
	}

	parseNvidiaApps(nvAppsFixture, gpus, idx)
	if len(gpus[0].Procs) != 1 || gpus[0].Procs[0].PID != 4242 || gpus[0].Procs[0].UsedBytes != 22000*model.MiB {
		t.Errorf("proc parse wrong: %+v", gpus[0].Procs)
	}
	if gpus[1].Procs[0].Name != "python" {
		t.Errorf("gpu1 proc: %+v", gpus[1].Procs)
	}
}

func TestParseNvidiaAppsNoProcesses(t *testing.T) {
	gpus, idx := parseNvidiaGPUs(nvGPUFixture)
	parseNvidiaApps("No running processes found", gpus, idx)
	if len(gpus[0].Procs) != 0 {
		t.Errorf("expected no procs, got %+v", gpus[0].Procs)
	}
}

const rocmFixture = `{
  "card0": {
    "VRAM Total Memory (B)": "25753026560",
    "VRAM Total Used Memory (B)": "24696061952",
    "Card Series": "Radeon RX 7900 XTX",
    "Card Model": "0x744c",
    "Driver version": "6.7.0"
  },
  "system": {
    "Driver version": "6.7.0"
  }
}`

func TestParseROCm(t *testing.T) {
	gpus, err := parseROCm(rocmFixture)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	g := gpus[0]
	if g.Name != "Radeon RX 7900 XTX" {
		t.Errorf("name = %q", g.Name)
	}
	if g.Vendor != model.VendorAMD {
		t.Errorf("vendor = %q", g.Vendor)
	}
	if g.TotalBytes != 25753026560 {
		t.Errorf("total = %d", g.TotalBytes)
	}
	if g.FreeBytes != 25753026560-24696061952 {
		t.Errorf("free = %d", g.FreeBytes)
	}
	if g.Driver != "6.7.0" {
		t.Errorf("driver = %q", g.Driver)
	}
}

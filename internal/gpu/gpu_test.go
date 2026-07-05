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

func TestParseNvidiaAppsCommaInName(t *testing.T) {
	gpus, idx := parseNvidiaGPUs(nvGPUFixture)
	// A process launched from a path containing a comma must not shift columns.
	parseNvidiaApps("GPU-abc123, 4242, /opt/app,v2/python, 512", gpus, idx)
	if len(gpus[0].Procs) != 1 {
		t.Fatalf("want 1 proc, got %d", len(gpus[0].Procs))
	}
	p := gpus[0].Procs[0]
	if p.Name != "/opt/app,v2/python" {
		t.Errorf("name = %q, want %q", p.Name, "/opt/app,v2/python")
	}
	if p.UsedBytes != 512*model.MiB {
		t.Errorf("used = %d, want %d (memory must come from the last field)", p.UsedBytes, uint64(512*model.MiB))
	}
}

func TestParseUintEdgeCases(t *testing.T) {
	cases := map[string]uint64{
		"":            0,
		"   ":         0,
		"\t":          0,
		"25753026560": 25753026560,
		"1.5 GB":      1,
		"[N/A]":       0,
	}
	for in, want := range cases {
		if got := parseUint(in); got != want {
			t.Errorf("parseUint(%q) = %d, want %d", in, got, want)
		}
	}
}

const rocmFixture = `{
  "card0": {
    "VRAM Total Memory (B)": "25753026560",
    "VRAM Total Used Memory (B)": "24696061952",
    "Card Series": "Radeon RX 7900 XTX",
    "Card Model": "0x744c",
    "PCI Bus": "0000:03:00.0",
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
	if g.PCIBus != "0000:03:00.0" {
		t.Errorf("pci bus = %q", g.PCIBus)
	}
}

// A nested object value (ROCm 6/7 emit these) must not make json.Unmarshal fail
// for the whole document and drop every AMD GPU.
func TestParseROCmNestedObjectValue(t *testing.T) {
	const j = `{
	  "card0": {
	    "VRAM Total Memory (B)": "25753026560",
	    "VRAM Total Used Memory (B)": "1000",
	    "Card Series": "Radeon RX 7900 XTX",
	    "GPU Metrics": {"temperature": "45.0"}
	  },
	  "system": {"Driver version": "6.7.0"}
	}`
	gpus, err := parseROCm(j)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu despite the nested object, got %d", len(gpus))
	}
	if gpus[0].Name != "Radeon RX 7900 XTX" || gpus[0].TotalBytes != 25753026560 {
		t.Errorf("card mis-parsed: %+v", gpus[0])
	}
}

// A card that reports no meminfo (headless/masked) must not appear as a phantom
// 0-byte GPU.
func TestParseROCmSkipsPhantomCard(t *testing.T) {
	const j = `{
	  "card0": {"VRAM Total Memory (B)": "25753026560", "Card Series": "Radeon RX 7900 XTX"},
	  "card1": {"PCI Bus": "0000:44:00.0"}
	}`
	gpus, err := parseROCm(j)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 real gpu, got %d: %+v", len(gpus), gpus)
	}
	if gpus[0].Index != 0 {
		t.Errorf("kept the wrong card: %+v", gpus[0])
	}
}

// With no human-readable product name, fall back to "AMD GPU N", never the hex
// device id from "GPU ID"/"Card Model".
func TestParseROCmNameFallbackNotHex(t *testing.T) {
	const j = `{"card0": {"VRAM Total Memory (B)": "25753026560", "GPU ID": "0x744c", "Card Model": "0x744c"}}`
	gpus, err := parseROCm(j)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	if gpus[0].Name != "AMD GPU 0" {
		t.Errorf("name = %q, want \"AMD GPU 0\" (not a hex id)", gpus[0].Name)
	}
}

// Newer ROCm may spell the product name market_name.
func TestParseROCmMarketName(t *testing.T) {
	const j = `{"card0": {"VRAM Total Memory (B)": "25753026560", "market_name": "Radeon RX 7900 XT"}}`
	gpus, err := parseROCm(j)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 || gpus[0].Name != "Radeon RX 7900 XT" {
		t.Fatalf("market_name not used as name: %+v", gpus)
	}
}

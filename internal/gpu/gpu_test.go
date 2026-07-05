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

// amd-smi static + metric fixtures, joined on the "gpu" index. GPU 1's asic block
// has collapsed to the bare string "N/A" (a real amd-smi failure mode), and the
// metric array is in the opposite GPU order to static to exercise the join.
const amdStaticFixture = `[
  {"gpu":0,"asic":{"market_name":"Radeon RX 7900 XT"},"bus":{"bdf":"0000:03:00.0"},"driver":{"name":"amdgpu","version":"6.7.0"},"vram":{"type":"GDDR6","size":{"value":20464,"unit":"MB"}}},
  {"gpu":1,"asic":"N/A","bus":{"bdf":"0000:44:00.0"},"driver":{"name":"amdgpu","version":"6.7.0"},"vram":{"type":"GDDR6","size":{"value":20464,"unit":"MB"}}}
]`

const amdMetricFixture = `[
  {"gpu":1,"mem_usage":{"total_vram":{"value":20464,"unit":"MB"},"used_vram":{"value":1000,"unit":"MB"},"free_vram":{"value":19464,"unit":"MB"}}},
  {"gpu":0,"mem_usage":{"total_vram":{"value":20464,"unit":"MB"},"used_vram":{"value":5000,"unit":"MB"},"free_vram":{"value":15464,"unit":"MB"}}}
]`

func TestParseAMDSMI(t *testing.T) {
	gpus, err := parseAMDSMI(amdStaticFixture, amdMetricFixture)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 2 {
		t.Fatalf("want 2 gpus, got %d", len(gpus))
	}
	g0 := gpus[0]
	if g0.Index != 0 || g0.Name != "Radeon RX 7900 XT" {
		t.Errorf("gpu0 identity wrong: %+v", g0)
	}
	if g0.Vendor != model.VendorAMD || g0.Driver != "6.7.0" || g0.PCIBus != "0000:03:00.0" {
		t.Errorf("gpu0 fields wrong: %+v", g0)
	}
	// "MB" values are MiB-magnitude: 20464 * 1 MiB.
	if g0.TotalBytes != 20464*model.MiB {
		t.Errorf("gpu0 total = %d, want %d", g0.TotalBytes, uint64(20464*model.MiB))
	}
	// Joined to gpu 0's metric entry (used 5000), not gpu 1's (used 1000), despite
	// the metric array being in the opposite order.
	if g0.UsedBytes != 5000*model.MiB {
		t.Errorf("gpu0 used = %d, want %d (join by gpu index)", g0.UsedBytes, uint64(5000*model.MiB))
	}
	if g0.FreeBytes != 15464*model.MiB {
		t.Errorf("gpu0 free = %d", g0.FreeBytes)
	}
	// GPU 1's asic collapsed to "N/A": name falls back, but usage still joins.
	g1 := gpus[1]
	if g1.Name != "AMD GPU 1" {
		t.Errorf("gpu1 name = %q, want the fallback", g1.Name)
	}
	if g1.UsedBytes != 1000*model.MiB {
		t.Errorf("gpu1 used = %d, want %d", g1.UsedBytes, uint64(1000*model.MiB))
	}
}

// Without metric output, a GPU is still reported from static with its capacity;
// usage is left at zero rather than guessed.
func TestParseAMDSMINoMetric(t *testing.T) {
	gpus, err := parseAMDSMI(amdStaticFixture, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 2 {
		t.Fatalf("want 2 gpus, got %d", len(gpus))
	}
	g0 := gpus[0]
	if g0.TotalBytes != 20464*model.MiB || g0.UsedBytes != 0 || g0.FreeBytes != 20464*model.MiB {
		t.Errorf("no-metric fallback wrong: %+v", g0)
	}
}

// Missing fields ("N/A" blocks) and bare-number leaves (old amd-smi builds) are
// tolerated, and free is computed from total-used when the CLI reports "N/A".
func TestParseAMDSMITolerant(t *testing.T) {
	const stat = `[{"gpu":0,"asic":{"market_name":"N/A"},"bus":"N/A","driver":"N/A","vram":{"size":16384}}]`
	const metric = `[{"gpu":0,"mem_usage":{"total_vram":16384,"used_vram":{"value":2048,"unit":"MB"},"free_vram":"N/A"}}]`
	gpus, err := parseAMDSMI(stat, metric)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	g := gpus[0]
	if g.Name != "AMD GPU 0" { // asic market_name "N/A" -> fallback
		t.Errorf("name = %q", g.Name)
	}
	if g.PCIBus != "" || g.Driver != "" { // bus/driver blocks were the bare string "N/A"
		t.Errorf("expected empty bus/driver, got bus=%q driver=%q", g.PCIBus, g.Driver)
	}
	if g.TotalBytes != 16384*model.MiB { // bare-number total
		t.Errorf("total = %d, want %d", g.TotalBytes, uint64(16384*model.MiB))
	}
	if g.UsedBytes != 2048*model.MiB {
		t.Errorf("used = %d", g.UsedBytes)
	}
	if g.FreeBytes != (16384-2048)*model.MiB { // free_vram "N/A" -> total - used
		t.Errorf("free = %d, want %d", g.FreeBytes, uint64((16384-2048)*model.MiB))
	}
}

// A device with NO VRAM info AND no identity (masked/unsupported) is skipped, not
// shown as a phantom 0-byte GPU. Note bdf alone doesn't make it a real GPU.
func TestParseAMDSMISkipsPhantom(t *testing.T) {
	const stat = `[
	  {"gpu":0,"asic":{"market_name":"Radeon RX 7900 XT"},"bus":{"bdf":"0000:03:00.0"},"vram":{"size":{"value":20464,"unit":"MB"}}},
	  {"gpu":1,"asic":"N/A","bus":{"bdf":"0000:44:00.0"},"vram":"N/A"}
	]`
	gpus, err := parseAMDSMI(stat, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 real gpu, got %d: %+v", len(gpus), gpus)
	}
	if gpus[0].Index != 0 {
		t.Errorf("kept the wrong gpu: %+v", gpus[0])
	}
}

// A card amd-smi can name but not size (vram + mem_usage both "N/A", common on
// Windows) is kept with an unknown capacity rather than dropped.
func TestParseAMDSMIKeepsIdentifiableCard(t *testing.T) {
	const stat = `[{"gpu":0,"asic":{"market_name":"Radeon RX 7900 XT"},"bus":{"bdf":"0000:03:00.0"},"driver":{"version":"6.7.0"},"vram":"N/A"}]`
	const metric = `[{"gpu":0,"mem_usage":"N/A"}]`
	gpus, err := parseAMDSMI(stat, metric)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want the identifiable card kept, got %d gpus", len(gpus))
	}
	g := gpus[0]
	if g.Name != "Radeon RX 7900 XT" || g.PCIBus != "0000:03:00.0" || g.Driver != "6.7.0" {
		t.Errorf("identity lost: %+v", g)
	}
	if g.TotalBytes != 0 {
		t.Errorf("total = %d, want 0 (unknown capacity)", g.TotalBytes)
	}
}

// When metric gives total and free but used is "N/A", used is derived as
// total - free rather than left at 0 (which would draw a too-empty bar).
func TestParseAMDSMIUsedFromFree(t *testing.T) {
	const stat = `[{"gpu":0,"asic":{"market_name":"X"},"vram":{"size":{"value":100,"unit":"MB"}}}]`
	const metric = `[{"gpu":0,"mem_usage":{"total_vram":{"value":100,"unit":"MB"},"used_vram":"N/A","free_vram":{"value":40,"unit":"MB"}}}]`
	gpus, err := parseAMDSMI(stat, metric)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	g := gpus[0]
	if g.UsedBytes != 60*model.MiB || g.FreeBytes != 40*model.MiB || g.TotalBytes != 100*model.MiB {
		t.Errorf("used-from-free wrong: used=%d free=%d total=%d", g.UsedBytes, g.FreeBytes, g.TotalBytes)
	}
}

// The "gpu" join key may be a quoted string on some builds; it must still parse
// and join static<->metric.
func TestParseAMDSMIQuotedIndex(t *testing.T) {
	const stat = `[{"gpu":"0","asic":{"market_name":"X"},"vram":{"size":{"value":8192,"unit":"MB"}}}]`
	const metric = `[{"gpu":"0","mem_usage":{"used_vram":{"value":1024,"unit":"MB"}}}]`
	gpus, err := parseAMDSMI(stat, metric)
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	if gpus[0].Index != 0 || gpus[0].UsedBytes != 1024*model.MiB {
		t.Errorf("quoted index not joined: %+v", gpus[0])
	}
}

// An implausibly huge VRAM leaf is rejected (0), not saturated to a bogus total.
func TestParseAMDSMIRejectsGarbageVRAM(t *testing.T) {
	const stat = `[{"gpu":0,"asic":{"market_name":"X"},"vram":{"size":{"value":1e30,"unit":"GB"}}}]`
	gpus, err := parseAMDSMI(stat, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 { // kept via its name, but the garbage capacity is rejected
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	if gpus[0].TotalBytes != 0 {
		t.Errorf("garbage total not rejected: %d", gpus[0].TotalBytes)
	}
}

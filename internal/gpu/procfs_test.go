package gpu

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

const drmFdinfo = `pos:	0
flags:	02000002
drm-driver:	amdgpu
drm-pdev:	0000:03:00.0
drm-client-id:	10
drm-memory-vram:	4194304 KiB
drm-memory-gtt:	1024 KiB
`

func TestParseFdinfo(t *testing.T) {
	pdev, client, vram, isDRM := parseFdinfo(drmFdinfo)
	if !isDRM {
		t.Fatal("expected a DRM fd")
	}
	if pdev != "0000:03:00.0" || client != "10" {
		t.Errorf("pdev=%q client=%q", pdev, client)
	}
	if vram != 4*model.GiB {
		t.Errorf("vram = %d, want %d", vram, uint64(4*model.GiB))
	}
	// A non-DRM fd.
	_, _, _, isDRM = parseFdinfo("pos:\t0\nflags:\t02\nmnt_id:\t8\n")
	if isDRM {
		t.Error("non-DRM fd should not be flagged DRM")
	}
}

func TestParseDRMBytes(t *testing.T) {
	cases := map[string]uint64{
		"4194304 KiB": 4 * model.GiB,
		"2 MiB":       2 * model.MiB,
		"1 GiB":       1 * model.GiB,
		"512":         512,
		"":            0,
		"garbage":     0,
	}
	for in, want := range cases {
		if got := parseDRMBytes(in); got != want {
			t.Errorf("parseDRMBytes(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestNormalizePCI(t *testing.T) {
	cases := map[string]string{
		"03:00.0":      "0000:03:00.0",
		"0000:03:00.0": "0000:03:00.0",
		"0000:0A:00.0": "0000:0a:00.0",
		"":             "",
	}
	for in, want := range cases {
		if got := normalizePCI(in); got != want {
			t.Errorf("normalizePCI(%q) = %q, want %q", in, got, want)
		}
	}
}

// writeFd writes a fake /proc/<pid>/fdinfo/<fd> file.
func writeFd(t *testing.T, root string, pid int, fd string, content string) {
	t.Helper()
	dir := filepath.Join(root, strconv.Itoa(pid), "fdinfo")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, fd), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeComm(t *testing.T, root string, pid int, name string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, strconv.Itoa(pid), "comm"), []byte(name+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestProcVRAMIn(t *testing.T) {
	root := t.TempDir()
	// pid 1234: two fds, same client id 10 (dup) -> counted once = 4 GiB.
	writeFd(t, root, 1234, "3", drmFdinfo)
	writeFd(t, root, 1234, "4", drmFdinfo)               // dup client -> not double counted
	writeFd(t, root, 1234, "5", "pos:\t0\nflags:\t02\n") // non-DRM
	writeComm(t, root, 1234, "ollama")
	// pid 2000: distinct client 20, 2 GiB, same device.
	writeFd(t, root, 2000, "7", "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t20\ndrm-memory-vram:\t2097152 KiB\n")
	writeComm(t, root, 2000, "python")
	// a non-numeric entry that must be ignored.
	os.MkdirAll(filepath.Join(root, "self"), 0o755)

	got := procVRAMIn(root)
	procs := got["0000:03:00.0"]
	if len(procs) != 2 {
		t.Fatalf("want 2 procs, got %d: %+v", len(procs), procs)
	}
	byPID := map[int]model.Proc{}
	for _, p := range procs {
		byPID[p.PID] = p
	}
	if byPID[1234].UsedBytes != 4*model.GiB || byPID[1234].Name != "ollama" {
		t.Errorf("pid 1234 wrong (dedup failed?): %+v", byPID[1234])
	}
	if byPID[2000].UsedBytes != 2*model.GiB {
		t.Errorf("pid 2000 = %+v", byPID[2000])
	}
}

func TestProcVRAMInMissingRoot(t *testing.T) {
	if procVRAMIn(filepath.Join(t.TempDir(), "nope")) != nil {
		t.Error("missing root should yield nil")
	}
}

func TestAttachProcs(t *testing.T) {
	procs := map[string][]model.Proc{"0000:03:00.0": {{PID: 1, Name: "ollama", UsedBytes: 4 * model.GiB}}}

	// Match by PCI bus.
	gpus := []model.GPU{{Index: 0, Vendor: model.VendorAMD, PCIBus: "0000:03:00.0"}}
	attachProcs(gpus, procs)
	if len(gpus[0].Procs) != 1 {
		t.Error("AMD GPU should get procs via PCI match")
	}

	// Single ambiguous AMD GPU + single DRM device: attach directly.
	gpus = []model.GPU{{Index: 0, Vendor: model.VendorAMD}}
	attachProcs(gpus, procs)
	if len(gpus[0].Procs) != 1 {
		t.Error("single AMD GPU should get the single device's procs")
	}

	// NVIDIA GPU with existing procs is left alone.
	gpus = []model.GPU{{Index: 0, Vendor: model.VendorNVIDIA, Procs: []model.Proc{{PID: 9}}}}
	attachProcs(gpus, procs)
	if len(gpus[0].Procs) != 1 || gpus[0].Procs[0].PID != 9 {
		t.Error("NVIDIA procs should not be overwritten")
	}
}

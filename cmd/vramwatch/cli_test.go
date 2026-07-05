package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"strings"
	"testing"
)

const oomMock = "mock:../../testdata/scenarios/24gb-70b-oom.json"
const fitMock = "mock:../../testdata/scenarios/16gb-13b-fits.json"

// capture redirects os.Stdout for the duration of fn and returns what it wrote.
func capture(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	runErr := fn()
	w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String(), runErr
}

func TestCmdSnapshotConsole(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdSnapshot([]string{"--source", oomMock, "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"weights", "KV cache", "OOM risk", "llama3:70b-q2_K"} {
		if !strings.Contains(out, want) {
			t.Errorf("snapshot output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "\x1b[") {
		t.Error("--no-color output should not contain ANSI escapes")
	}
}

func TestCmdSnapshotJSON(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdSnapshot([]string{"--source", oomMock, "--json"})
	})
	if err != nil {
		t.Fatal(err)
	}
	var snap map[string]any
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatalf("snapshot --json is not valid JSON: %v\n%s", err, out)
	}
	if !strings.Contains(out, `"oom_risk": true`) {
		t.Error("expected oom_risk true in JSON output")
	}
}

func TestCmdSnapshotSVG(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/card.svg"
	_, err := capture(t, func() error {
		return cmdSnapshot([]string{"--source", oomMock, "--svg", path, "--static"})
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(data, []byte("<svg")) {
		t.Error("SVG file does not start with <svg")
	}
}

func TestCmdPredictFits(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdPredict([]string{"--source", oomMock, "--context", "32768", "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "WON'T FIT") {
		t.Errorf("expected 32k to not fit on the OOM scenario:\n%s", out)
	}
}

func TestCmdPredictHealthy(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdPredict([]string{"--source", fitMock, "--context", "8192", "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "FITS") {
		t.Errorf("expected 8k to fit on the 16 GiB scenario:\n%s", out)
	}
	if !strings.Contains(out, "exceeds trained context") {
		t.Errorf("expected trained-context caveat (model trained to 4096):\n%s", out)
	}
}

func TestCmdWatchOnce(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdWatch([]string{"--source", oomMock, "--once", "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "weights") {
		t.Errorf("watch --once should render a frame:\n%s", out)
	}
}

func TestCmdDevices(t *testing.T) {
	out, err := capture(t, func() error {
		return cmdDevices([]string{"--source", oomMock, "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"GPU providers", "Loader providers", "Devices", "RX 7900 XTX"} {
		if !strings.Contains(out, want) {
			t.Errorf("devices output missing %q:\n%s", want, out)
		}
	}
}

func TestCmdSnapshotBadSource(t *testing.T) {
	_, err := capture(t, func() error {
		return cmdSnapshot([]string{"--source", "bogus"})
	})
	if err == nil {
		t.Error("expected an error for an unrecognised --source")
	}
	// A bad source is a runtime error, not a usage error.
	var ue *usageError
	if errors.As(err, &ue) {
		t.Error("bad --source value should be a runtime error, not a usageError")
	}
}

func TestResolveKVBits(t *testing.T) {
	t.Setenv("VRAMWATCH_KV_CACHE_TYPE", "") // isolate from the ambient env
	// Quantized widths are rounded up for the GGML per-block scale overhead.
	cases := map[string]int{"": 0, "f16": 16, "BF16": 16, "f32": 32, "q8_0": 9, "q5_0": 6, "q4_0": 5, "q4_1": 5}
	for in, want := range cases {
		got, err := resolveKVBits(in)
		if err != nil || got != want {
			t.Errorf("resolveKVBits(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	if _, err := resolveKVBits("bogus"); err == nil {
		t.Error("bogus dtype should error")
	}
}

// kvSegmentBytes runs snapshot --json with extra args and returns the KV-cache
// segment size of the first breakdown.
func kvSegmentBytes(t *testing.T, extra ...string) uint64 {
	t.Helper()
	out, err := capture(t, func() error {
		return cmdSnapshot(append([]string{"--source", oomMock, "--json"}, extra...))
	})
	if err != nil {
		t.Fatal(err)
	}
	var snap struct {
		Breakdowns []struct {
			Segments []struct {
				Kind  string `json:"kind"`
				Bytes uint64 `json:"bytes"`
			} `json:"segments"`
		} `json:"breakdowns"`
	}
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatal(err)
	}
	for _, s := range snap.Breakdowns[0].Segments {
		if s.Kind == "kv_cache" {
			return s.Bytes
		}
	}
	t.Fatal("no kv_cache segment found")
	return 0
}

func TestCmdKVCacheType(t *testing.T) {
	t.Setenv("VRAMWATCH_KV_CACHE_TYPE", "")
	f16 := kvSegmentBytes(t)                           // 16 bits
	q8 := kvSegmentBytes(t, "--kv-cache-type", "q8_0") // 9 bits (8.5 rounded up)
	q4 := kvSegmentBytes(t, "--kv-cache-type", "q4_0") // 5 bits (4.5 rounded up)
	// KV scales linearly with the bit width.
	if q8*16 != f16*9 {
		t.Errorf("q8_0 KV (%d) should be 9/16 of f16 (%d)", q8, f16)
	}
	if q4*16 != f16*5 {
		t.Errorf("q4_0 KV (%d) should be 5/16 of f16 (%d)", q4, f16)
	}
	if !(q4 < q8 && q8 < f16) {
		t.Errorf("expected q4 < q8 < f16, got %d %d %d", q4, q8, f16)
	}
}

func TestCmdHelpReturnsErrHelp(t *testing.T) {
	// flag prints usage to stderr; silence it for a clean test run.
	old := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w
	err := cmdSnapshot([]string{"--help"})
	w.Close()
	os.Stderr = old
	if !errors.Is(err, flag.ErrHelp) {
		t.Errorf("--help should return flag.ErrHelp (so main exits 0), got %v", err)
	}
}

func TestCmdBadFlagIsUsageError(t *testing.T) {
	old := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w
	err := cmdSnapshot([]string{"--nope"})
	w.Close()
	os.Stderr = old
	var ue *usageError
	if !errors.As(err, &ue) {
		t.Errorf("an unknown flag should be a usageError (so main exits 2), got %v", err)
	}
}

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
	for _, want := range []string{"weights", "KV cache", "OOM risk", "llama3:70b-q4"} {
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

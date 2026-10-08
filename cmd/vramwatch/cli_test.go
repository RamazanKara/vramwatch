package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

const oomMock = "mock:../../testdata/scenarios/24gb-70b-oom.json"

// capture redirects os.Stdout for the duration of fn and returns what it wrote.
func capture(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	var buf bytes.Buffer
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(&buf, r)
		r.Close()
		close(done)
	}()
	runErr := fn()
	w.Close()
	os.Stdout = old
	<-done
	return buf.String(), runErr
}

func TestCmdWatchOnce(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	out, err := capture(t, func() error {
		return cmdWatch([]string{"--source", oomMock, "--once", "--no-color"})
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"weights", "KV cache", "OOM risk", "[M]", "[E]"} {
		if !strings.Contains(out, want) {
			t.Errorf("watch --once output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "\x1b[") {
		t.Error("--no-color output should not contain ANSI escapes")
	}
}

func TestCmdWatchBadSourceIsRuntimeError(t *testing.T) {
	_, err := capture(t, func() error { return cmdWatch([]string{"--source", "bogus", "--once"}) })
	if err == nil {
		t.Error("expected an error for an unrecognised --source")
	}
	var ue *usageError
	if errors.As(err, &ue) {
		t.Error("bad source should be a runtime error, not a usage error")
	}
}

func TestCmdWatchRejectsPositionalArguments(t *testing.T) {
	for _, args := range [][]string{
		{"typo", "--once"},
		{"--source", "demo", "--once", "unexpected"},
	} {
		_, err := capture(t, func() error { return cmdWatch(args) })
		var usage *usageError
		if !errors.As(err, &usage) {
			t.Errorf("cmdWatch(%q) error = %v, want usage error", args, err)
		}
	}
}

func TestDoctorLoaderDiagnostics(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ps     string
		status string
	}{
		{"idle loader", `{"models":[]}`, "warn"},
		{"resident model", `{"models":[{"name":"fixture","size_vram":1024}]}`, "pass"},
		{"invalid response", `{`, "fail"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
			t.Setenv("PATH", t.TempDir())
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/version":
					w.Write([]byte(`{"version":"fixture"}`))
				case "/api/ps":
					w.Write([]byte(tc.ps))
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			t.Setenv("LLAMACPP_HOST", srv.URL)
			out, err := capture(t, func() error { return cmdDoctor([]string{"--json", "--verbose"}) })
			var result doctorEnvelope
			if e := json.Unmarshal([]byte(out), &result); e != nil {
				t.Fatal(e)
			}
			if result.SchemaVersion != 1 || result.Command != "doctor" || result.Healthy != (err == nil) {
				t.Fatalf("envelope = %+v, error = %v", result, err)
			}
			if !result.Healthy {
				var exit *exitError
				if !errors.As(err, &exit) || exit.code != 1 {
					t.Fatalf("unhealthy doctor error = %v", err)
				}
			}
			found := false
			for _, check := range result.Checks {
				if check.ID == "loader.ollama" {
					found = true
					if check.Status != tc.status || (tc.status == "fail" && (check.Evidence == "" || check.Remediation == "")) {
						t.Errorf("loader check = %+v", check)
					}
				}
			}
			if !found {
				t.Fatal("missing loader diagnostic")
			}
		})
	}
}

func TestResolveKVBits(t *testing.T) {
	t.Setenv("VRAMWATCH_KV_CACHE_TYPE", "")
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

func TestCmdHelpReturnsErrHelp(t *testing.T) {
	old := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w
	err := cmdWatch([]string{"--help"})
	w.Close()
	os.Stderr = old
	if !errors.Is(err, flag.ErrHelp) {
		t.Errorf("--help should return flag.ErrHelp, got %v", err)
	}
}

func TestCmdBadFlagIsUsageError(t *testing.T) {
	old := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w
	err := cmdWatch([]string{"--nope"})
	w.Close()
	os.Stderr = old
	var ue *usageError
	if !errors.As(err, &ue) {
		t.Errorf("unknown flag should be a usage error, got %v", err)
	}
}

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	fitengine "github.com/RamazanKara/vramwatch/internal/fit"
	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
	"github.com/RamazanKara/vramwatch/internal/source"
)

func TestCmdWatchJSON(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	empty := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(empty, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, source string
		devices      int
	}{
		{"fixture", oomMock, 1},
		{"demo", "demo", 1},
		{"no devices", empty, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := capture(t, func() error {
				return cmdWatch([]string{"--source", tc.source, "--json", "--once", "--color", "--kv-cache-type", "q8_0"})
			})
			if err != nil {
				t.Fatal(err)
			}
			if strings.Count(out, "\n") != 1 || strings.Contains(out, "\x1b") {
				t.Fatalf("expected one JSON line without terminal controls: %q", out)
			}
			var env watchEnvelope
			if err := json.Unmarshal([]byte(out), &env); err != nil {
				t.Fatal(err)
			}
			if env.SchemaVersion != 1 || env.Command != "watch" || env.Snapshot.Version != Version || env.Snapshot.Timestamp.IsZero() || len(env.Snapshot.Breakdowns) != tc.devices {
				t.Fatalf("invalid watch envelope: %+v", env)
			}
		})
	}
}

func TestWatchJSONStreamRecordsStableObservation(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	watchTrack = watchTrackState{}
	t.Cleanup(func() { watchTrack = watchTrackState{} })
	rec, err := ledger.Save(fitengine.Result{
		Artifact: fitengine.Artifact{CanonicalID: "model"}, Context: 4096, ExpectedFootprintBytes: 2 * model.MiB,
	}, "ollama")
	if err != nil {
		t.Fatal(err)
	}
	s := &source.Mock{Scenario: source.Scenario{
		GPUs:   []model.GPU{{Index: 0, Name: "fixture", TotalBytes: model.GiB, UsedBytes: model.MiB, FreeBytes: model.GiB - model.MiB}},
		Models: []model.LoaderModel{{Loader: "ollama", Name: "model", GPUIndex: 0, ContextTokens: 4096, VRAMBytes: model.MiB, VRAMSource: model.ProvenanceReported}},
	}}
	out, err := capture(t, func() error {
		for i := 0; i < 3; i++ {
			if err := renderFrame(context.Background(), s, render.Options{Color: true}, 0, true, "footer", true); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(out, "\n"), "\n")
	if len(lines) != 3 || strings.Contains(out, "\x1b") || strings.Contains(out, "footer") {
		t.Fatalf("invalid NDJSON stream: %q", out)
	}
	for _, line := range lines {
		var env watchEnvelope
		if err := json.Unmarshal([]byte(line), &env); err != nil {
			t.Fatal(err)
		}
		if len(env.Snapshot.Breakdowns) != 1 || env.Snapshot.Breakdowns[0].GPU.Name != "fixture" {
			t.Fatalf("missing snapshot: %+v", env)
		}
	}
	rec, err = ledger.Load(rec.ID)
	if err != nil || rec.Observation == nil || rec.Observation.FootprintBytes != model.MiB || rec.Observation.Provenance != model.ProvenanceReported {
		t.Fatalf("JSON watch lost observation tracking: %+v, %v", rec.Observation, err)
	}
}

func TestWatchWriteErrors(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	for _, tc := range []struct {
		name string
		json bool
	}{{"console", false}, {"JSON", true}} {
		t.Run(tc.name, func(t *testing.T) {
			closed, err := os.Create(filepath.Join(t.TempDir(), "closed"))
			if err != nil {
				t.Fatal(err)
			}
			closed.Close()
			old := os.Stdout
			os.Stdout = closed
			defer func() { os.Stdout = old }()
			err = renderFrame(context.Background(), &source.Mock{}, render.Options{}, 0, false, "", tc.json)
			if !errors.Is(err, os.ErrClosed) {
				t.Fatalf("write error = %v, want closed stdout", err)
			}
		})
	}
}

func TestCmdFitContextLimits(t *testing.T) {
	path := commandGGUF(t)
	for _, tc := range []struct {
		name, budget, context string
		json                  bool
		limit, code           int
	}{
		{"JSON fits", "1GiB", "4096", true, 8192, 0},
		{"JSON no room", "512MiB", "4096", true, 0, 3},
		{"JSON unsupported context", "1GiB", "16384", true, 8192, 3},
		{"console fits", "1GiB", "4096", false, 8192, 0},
		{"console no room", "512MiB", "4096", false, 0, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := []string{path, "--context", tc.context, "--vram", tc.budget, "--no-record", "--no-color"}
			if tc.json {
				args = append(args, "--json")
			}
			out, err := capture(t, func() error { return cmdFit(args) })
			var exit *exitError
			if (tc.code == 0 && err != nil) || (tc.code != 0 && (!errors.As(err, &exit) || exit.code != tc.code)) {
				t.Fatalf("exit error = %v, want code %d", err, tc.code)
			}
			if tc.json {
				var env fitEnvelope
				if err := json.Unmarshal([]byte(out), &env); err != nil {
					t.Fatal(err)
				}
				if len(env.Result.Targets) != 1 {
					t.Fatalf("targets = %+v", env.Result.Targets)
				}
				got := env.Result.Targets[0]
				if got.MaxContextOnDevice == nil || got.MaxContextNow == nil || *got.MaxContextOnDevice != tc.limit || *got.MaxContextNow != tc.limit {
					t.Fatalf("context limits = %+v, want %d", got, tc.limit)
				}
			} else if !strings.Contains(out, "[E] max context: "+commas(tc.limit)+" on device / "+commas(tc.limit)+" right now (tokens)") {
				t.Fatalf("missing context limits:\n%s", out)
			}
		})
	}
}

func TestCmdReportMarkdown(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	t.Setenv("PATH", t.TempDir())
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)
	t.Setenv("LLAMACPP_HOST", srv.URL)
	rec, err := ledger.Save(fitengine.Result{
		Artifact: fitengine.Artifact{Source: fitengine.SourceURL, CanonicalID: "https://private.example/model.gguf?token=secret", Reference: "https://private.example/model.gguf?token=secret", Filename: "model.gguf"},
		Context:  4096, KVCacheType: "F16", ExpectedFootprintBytes: model.GiB,
		Targets: []fitengine.TargetResult{{Target: fitengine.Target{GPUName: "fixture GPU", CapacityBytes: 8 * model.GiB}, FitsNow: fitengine.VerdictFits}},
	}, "llama.cpp")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name           string
		file, existing bool
		force, dash    bool
	}{
		{"stdout", false, false, false, false},
		{"explicit stdout", false, false, false, true},
		{"file", true, false, false, false},
		{"protect existing", true, true, false, false},
		{"force", true, true, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := []string{"--prediction", rec.ID, "--markdown", "--static"}
			path := filepath.Join(t.TempDir(), "report.md")
			if tc.file {
				args = append(args, "--output", path)
			} else if tc.dash {
				args = append(args, "--output", "-")
			}
			if tc.existing {
				if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if tc.force {
				args = append(args, "--force")
			}
			out, err := capture(t, func() error { return cmdReport(args) })
			if tc.existing && !tc.force {
				data, readErr := os.ReadFile(path)
				if err == nil || readErr != nil || string(data) != "original" {
					t.Fatalf("existing report changed: %q, %v, %v", data, err, readErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.file {
				if !strings.Contains(out, path) {
					t.Fatalf("output path missing: %q", out)
				}
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				out = string(data)
			}
			for _, want := range []string{"# vramwatch report", "model.gguf", "fixture GPU", "4,096", "FITS", "| Accuracy | pending |"} {
				if !strings.Contains(out, want) {
					t.Errorf("missing %q:\n%s", want, out)
				}
			}
			for _, private := range []string{"private.example", "token=", "secret", "| Generated |"} {
				if strings.Contains(out, private) {
					t.Errorf("static shareable report contains %q", private)
				}
			}
		})
	}
}

func TestReportOutputFlagConflicts(t *testing.T) {
	for _, args := range [][]string{
		{"--json", "--svg"}, {"--json", "--markdown"}, {"--svg", "--markdown"},
		{"--json", "--svg", "--markdown"}, {"--output", "report.md"}, {"--json", "--force"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			_, err := capture(t, func() error { return cmdReport(args) })
			var usage *usageError
			if !errors.As(err, &usage) {
				t.Fatalf("error = %v, want usage error", err)
			}
		})
	}
}

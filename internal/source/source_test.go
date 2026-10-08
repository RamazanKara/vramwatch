package source

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
)

const oomScenario = "../../testdata/scenarios/24gb-70b-oom.json"

func TestMockSpecsAndInvalidScenarios(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scenario.json")
	for _, tc := range []struct {
		name string
		data string
		ok   bool
	}{
		{"empty object", `{}`, true},
		{"unknown vendor", `{"gpus":[{"index":7,"total_bytes":1024}]}`, true},
		{"truncated", `{"gpus":[`, false},
		{"wrong shape", `{"gpus":"invalid"}`, false},
		{"negative capacity", `{"gpus":[{"total_bytes":-1}]}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(tc.data), 0o600); err != nil {
				t.Fatal(err)
			}
			for _, spec := range []string{path, "mock:" + path} {
				s, err := FromSpec(spec)
				if (err == nil) != tc.ok {
					t.Fatalf("FromSpec(%q) error = %v", spec, err)
				}
				if !tc.ok {
					continue
				}
				gpus, models, err := s.Collect(context.Background())
				if err != nil || len(models) != 0 || s.Describe() != "mock:"+path {
					t.Fatalf("mock = %v, %v, %v", gpus, models, err)
				}
				if tc.name == "unknown vendor" && (len(gpus) != 1 || gpus[0].Index != 7 || gpus[0].Vendor != model.VendorUnknown) {
					t.Fatalf("mock GPU identity = %+v", gpus)
				}
			}
		})
	}
}

func TestDemoAttribution(t *testing.T) {
	for _, tc := range []struct {
		name    string
		elapsed time.Duration
		oom     bool
	}{
		{"start", 0, false},
		{"near limit", 22 * time.Second, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, err := FromSpec("demo")
			if err != nil {
				t.Fatal(err)
			}
			demo := s.(Demo)
			demo.Start = time.Now().Add(-tc.elapsed)
			gpus, models, err := demo.Collect(context.Background())
			if err != nil || len(gpus) != 1 || len(models) != 1 {
				t.Fatalf("demo shape = %v, %v, %v", gpus, models, err)
			}
			if gpus[0].UsedBytes+gpus[0].FreeBytes != gpus[0].TotalBytes || gpus[0].UsageSource != model.ProvenanceAssumed || models[0].VRAMSource != model.ProvenanceAssumed {
				t.Fatalf("synthetic memory/provenance = %+v, %+v", gpus[0], models[0])
			}
			snap := engine.Build(gpus, models, engine.Options{})
			if got := snap.Breakdowns[0].Prediction.OOMRisk; got != tc.oom {
				t.Errorf("OOM risk = %v, want %v", got, tc.oom)
			}
		})
	}
}

func TestFromSpec(t *testing.T) {
	if s, err := FromSpec("live"); err != nil || s.Describe() == "" {
		t.Fatalf("live: %v", err)
	}
	if s, err := FromSpec(""); err != nil {
		t.Fatalf("empty: %v", err)
	} else if _, ok := s.(Live); !ok {
		t.Fatal("empty spec should be Live")
	}
	if _, err := FromSpec("bogus"); err == nil {
		t.Fatal("expected error for unrecognised spec")
	}
	if _, err := FromSpec("mock:does-not-exist.json"); err == nil {
		t.Fatal("expected error for missing mock file")
	}
}

func TestLoadMockAndBuild(t *testing.T) {
	m, err := LoadMock(oomScenario)
	if err != nil {
		t.Fatal(err)
	}
	gpus, models, err := m.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 || len(models) != 1 {
		t.Fatalf("scenario shape: %d gpus, %d models", len(gpus), len(models))
	}

	snap := engine.Build(gpus, models, engine.Options{Version: "test"})
	b := snap.Breakdowns[0]

	// Segments must tile the device exactly.
	var sum uint64
	for _, s := range b.Segments {
		sum += s.Bytes
	}
	if sum != gpus[0].TotalBytes {
		t.Errorf("segments (%d) != total (%d)", sum, gpus[0].TotalBytes)
	}
	// KV must match the standard formula for the scenario arch.
	kv, _ := b.Segment(model.KindKVCache)
	if kv.Bytes != engine.KVCacheBytes(models[0].Arch, 8192) {
		t.Errorf("kv = %d", kv.Bytes)
	}
	// This scenario is designed to be at OOM risk.
	if b.Prediction == nil || !b.Prediction.OOMRisk {
		t.Errorf("expected OOM risk, got %+v", b.Prediction)
	}
}

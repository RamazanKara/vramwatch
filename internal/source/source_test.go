package source

import (
	"context"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
)

const oomScenario = "../../testdata/scenarios/24gb-70b-oom.json"

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

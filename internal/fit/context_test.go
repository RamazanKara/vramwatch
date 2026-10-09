package fit

import (
	"encoding/json"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestPredictMaxContext(t *testing.T) {
	const fixed = 1024 + 256*model.MiB + 512*model.MiB
	for _, tc := range []struct {
		name           string
		capacity, free uint64
		known, manual  bool
		modelMax       int
		dtype          string
		valueDim       int
		device, now    int
	}{
		{"busy", fixed + 40, fixed + 8, true, false, 100, "f16", 1, 10, 2},
		{"unknown usage", fixed + 40, 0, false, false, 100, "f16", 1, 10, -1},
		{"full device", fixed + 40, 0, true, false, 100, "f16", 1, 10, 0},
		{"unknown capacity", 0, 0, false, false, 100, "f16", 1, -1, -1},
		{"weights exceed budget", fixed - 1, fixed - 1, true, false, 100, "f16", 1, 0, 0},
		{"no room for KV", fixed, fixed, true, false, 100, "f16", 1, 0, 0},
		{"manual", fixed + 40, 0, false, true, 100, "f16", 1, 10, 10},
		{"model limit", fixed + 40, fixed + 40, true, false, 7, "f16", 1, 7, 7},
		{"unknown model limit", fixed + 40, fixed + 40, true, false, 0, "f16", 1, 10, 10},
		{"quantized rounding", fixed + 10, fixed + 9, true, false, 100, "q4_0", 1, 8, 8},
		{"asymmetric heads", fixed + 60, fixed + 8, true, false, 100, "f16", 2, 10, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := Artifact{WeightBytes: 1024, ContextMax: tc.modelMax,
				Arch: model.Arch{Layers: 1, KVHeads: 1, HeadDim: 1, ValueDim: tc.valueDim, KVTypeBits: 16}}
			target := Target{CapacityBytes: tc.capacity, AvailableBytes: tc.free, AvailableKnown: tc.known, Manual: tc.manual}
			r, err := Predict(a, []Target{target}, PredictOptions{Context: 20, KVCacheType: tc.dtype})
			if err != nil {
				t.Fatal(err)
			}
			got := r.Targets[0]
			memoryOnly := false
			for _, warning := range r.Warnings {
				if warning == "model context limit is unknown; maximum context estimates are memory-only" {
					memoryOnly = true
				}
			}
			if memoryOnly != (tc.modelMax == 0) {
				t.Errorf("context warning = %v for model limit %d", r.Warnings, tc.modelMax)
			}
			for _, limit := range []struct {
				name string
				got  *int
				want int
			}{
				{"device", got.MaxContextOnDevice, tc.device},
				{"now", got.MaxContextNow, tc.now},
			} {
				if limit.want < 0 {
					if limit.got != nil {
						t.Errorf("%s = %d, want unknown", limit.name, *limit.got)
					}
				} else if limit.got == nil || *limit.got != limit.want {
					t.Errorf("%s = %v, want %d", limit.name, limit.got, limit.want)
				}
			}
			data, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatal(err)
			}
			for key, want := range map[string]int{"max_context_on_device": tc.device, "max_context_now": tc.now} {
				if value, ok := fields[key]; ok != (want >= 0) || (want == 0 && string(value) != "0") {
					t.Errorf("%s = %s (present %v), want %d", key, value, ok, want)
				}
			}
		})
	}
}

func TestPredictMaxContextFailsClosedOnOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name    string
		weights uint64
		arch    model.Arch
	}{
		{"weights", ^uint64(0), testArtifact().Arch},
		{"KV", 1, model.Arch{Layers: maxInt, KVHeads: maxInt, HeadDim: maxInt, KVTypeBits: 16}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := Artifact{WeightBytes: tc.weights, Arch: tc.arch}
			r, err := Predict(a, []Target{{CapacityBytes: ^uint64(0), AvailableBytes: ^uint64(0), AvailableKnown: true}}, PredictOptions{Context: 1})
			if err != nil {
				t.Fatal(err)
			}
			got := r.Targets[0]
			if got.FitsOnDevice != VerdictDoesNotFit || got.FitsNow != VerdictDoesNotFit || *got.MaxContextOnDevice != 0 || *got.MaxContextNow != 0 {
				t.Fatalf("overflow became an optimistic recommendation: %+v", got)
			}
		})
	}
}

func FuzzMaxContext(f *testing.F) {
	f.Add(uint64(1<<30), uint64(800<<20), uint16(8192), uint8(32), uint8(8), uint8(128), uint8(0))
	f.Add(^uint64(0), uint64(0), uint16(0), uint8(0), uint8(0), uint8(0), uint8(1))
	f.Add(uint64(0), ^uint64(0), uint16(1), uint8(0), uint8(0), uint8(0), uint8(2))
	f.Fuzz(func(t *testing.T, budget, fixed uint64, cap uint16, layers, heads, dim, kind uint8) {
		a := Artifact{ContextMax: int(cap), Arch: model.Arch{
			Layers: int(layers) + 1, KVHeads: int(heads) + 1, HeadDim: int(dim) + 1, KVTypeBits: 16,
		}}
		dtype := []string{"f16", "q4_0", "q8_0"}[int(kind)%3]
		got := maxContext(a, dtype, fixed, budget)
		if got < 0 || (a.ContextMax > 0 && got > a.ContextMax) {
			t.Fatalf("context %d exceeds model limit %d", got, a.ContextMax)
		}
		fits := func(n int) bool {
			kv, err := kvBytes(a.Arch, n, dtype)
			if err != nil {
				t.Fatal(err)
			}
			required := saturatingAdd(fixed, kv)
			return required != ^uint64(0) && required <= budget
		}
		if got > 0 && !fits(got) {
			t.Fatalf("recommended context %d does not fit", got)
		}
		if got < int(^uint(0)>>1) && (a.ContextMax == 0 || got < a.ContextMax) && fits(got+1) {
			t.Fatalf("context %d is not maximal", got)
		}
	})
}

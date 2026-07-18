package fit

import (
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func testArtifact() Artifact {
	return Artifact{
		CanonicalID:  "owner/model@abc",
		Source:       SourceHF,
		Quantization: "Q4_K_M",
		WeightBytes:  4 * model.GiB,
		WeightBasis:  "test fixture",
		Arch: model.Arch{
			Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, ValueDim: 128, KVTypeBits: 16,
		},
		ContextMax: 131072,
	}
}

func TestPredictConservativeFitAndProvenance(t *testing.T) {
	a := testArtifact()
	target := Target{GPUIndex: 0, GPUName: "test GPU", MemoryKind: model.MemoryDedicated, CapacityBytes: 16 * model.GiB, AvailableBytes: 10 * model.GiB, AvailableKnown: true}
	r, err := Predict(a, []Target{target}, PredictOptions{Context: 8192, KVCacheType: "f16", KVCacheTypeExplicit: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.KVCache.Bytes != model.GiB {
		t.Fatalf("KV cache = %d, want 1 GiB", r.KVCache.Bytes)
	}
	if r.Weights.Provenance != model.ProvenanceEstimated || r.RuntimeCeiling.Provenance != model.ProvenanceAssumed {
		t.Fatalf("wrong provenance: weights=%s runtime=%s", r.Weights.Provenance, r.RuntimeCeiling.Provenance)
	}
	if r.ExpectedFootprintBytes >= r.ConservativeFootprintBytes {
		t.Fatalf("expected footprint %d should be below conservative %d", r.ExpectedFootprintBytes, r.ConservativeFootprintBytes)
	}
	if got := r.Targets[0]; got.FitsOnDevice != VerdictFits || got.FitsNow != VerdictFits {
		t.Fatalf("verdict = %+v, want fits/fits", got)
	}
	if r.Confidence != "high" || len(r.Warnings) != 0 {
		t.Fatalf("confidence/warnings = %q %v", r.Confidence, r.Warnings)
	}
}

func TestPredictDistinguishesZeroAvailableFromUnknown(t *testing.T) {
	a := testArtifact()
	knownZero := Target{CapacityBytes: 16 * model.GiB, AvailableKnown: true}
	unknown := Target{CapacityBytes: 16 * model.GiB}
	r, err := Predict(a, []Target{knownZero, unknown}, PredictOptions{Context: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if r.Targets[0].FitsNow != VerdictDoesNotFit {
		t.Errorf("known zero free memory = %s, want does_not_fit", r.Targets[0].FitsNow)
	}
	if r.Targets[1].FitsNow != VerdictUnknown {
		t.Errorf("unknown free memory = %s, want unknown", r.Targets[1].FitsNow)
	}
	if r.Confidence != "medium" || !r.KVCacheTypeAssumed {
		t.Errorf("implicit KV type should lower confidence: %+v", r)
	}
}

func TestPredictRejectsUnsupportedContext(t *testing.T) {
	a := testArtifact()
	a.ContextMax = 4096
	r, err := Predict(a, []Target{{CapacityBytes: 80 * model.GiB, AvailableBytes: 80 * model.GiB, AvailableKnown: true}}, PredictOptions{Context: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if r.Targets[0].FitsOnDevice != VerdictContextTooLong || r.Targets[0].FitsNow != VerdictContextTooLong {
		t.Fatalf("oversized context verdict = %+v", r.Targets[0])
	}
}

func TestKVBytesUsesAsymmetricValueDimensionAndQuantWidth(t *testing.T) {
	a := model.Arch{Name: "test", Layers: 2, KVHeads: 1, HeadDim: 8, ValueDim: 4, KVTypeBits: 16}
	f16, err := kvBytes(a, 100, "f16")
	if err != nil {
		t.Fatal(err)
	}
	q8, err := kvBytes(a, 100, "q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if f16 != 4800 || q8 != 2550 {
		t.Fatalf("KV bytes f16/q8 = %d/%d, want 4800/2550", f16, q8)
	}
}

func TestCeilDivNearUint64LimitDoesNotWrap(t *testing.T) {
	n := ^uint64(0) - 1
	want := n / 32
	if n%32 != 0 {
		want++
	}
	if got := ceilDiv(n, 32); got != want {
		t.Fatalf("ceilDiv(%d, 32) = %d, want %d", n, got, want)
	}
}

func TestPredictNeverWrapsOverflowIntoFits(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	a := testArtifact()
	a.Arch = model.Arch{Name: "hostile", Layers: maxInt, KVHeads: maxInt, HeadDim: maxInt, ValueDim: maxInt, KVTypeBits: 16}
	r, err := Predict(a, []Target{{CapacityBytes: ^uint64(0), AvailableBytes: ^uint64(0), AvailableKnown: true}}, PredictOptions{Context: maxInt})
	if err != nil {
		t.Fatal(err)
	}
	if r.Targets[0].FitsOnDevice == VerdictFits || r.ConservativeFootprintBytes != ^uint64(0) {
		t.Fatalf("overflow became optimistic: %+v", r)
	}
}

func TestPredictRefusesIncompleteInputs(t *testing.T) {
	a := testArtifact()
	a.WeightBytes = 0
	if _, err := Predict(a, []Target{{CapacityBytes: model.GiB}}, PredictOptions{Context: 1}); err == nil {
		t.Error("zero weight size should fail")
	}
	a = testArtifact()
	if _, err := Predict(a, nil, PredictOptions{Context: 1}); err == nil {
		t.Error("missing target should fail")
	}
	if _, err := Predict(a, []Target{{CapacityBytes: model.GiB}}, PredictOptions{Context: 0}); err == nil {
		t.Error("zero context should fail")
	}
}

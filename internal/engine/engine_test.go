package engine

import (
	"testing"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestKVBytesPerToken(t *testing.T) {
	// Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128, f16 cache.
	// Known value: 128 KiB/token, i.e. 1 GiB at 8192 ctx.
	a := model.Arch{Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16}
	if got := KVBytesPerToken(a); got != 131072 {
		t.Fatalf("KVBytesPerToken = %d, want 131072", got)
	}
	if got := KVCacheBytes(a, 8192); got != 1*model.GiB {
		t.Fatalf("KVCacheBytes(8192) = %d, want %d", got, uint64(model.GiB))
	}
	// q8_0 cache halves the per-token cost.
	q8 := model.Arch{Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 8}
	if got := KVBytesPerToken(q8); got != 65536 {
		t.Fatalf("q8 KVBytesPerToken = %d, want 65536", got)
	}
	// Unknown arch => 0.
	if got := KVBytesPerToken(model.Arch{Layers: 32}); got != 0 {
		t.Fatalf("unknown arch KVBytesPerToken = %d, want 0", got)
	}
}

// A 24 GiB card running llama3:70b-q2_K at 8k context, nearly full.
func scenario70B() (model.GPU, []model.LoaderModel) {
	gpu := model.GPU{
		Index: 0, Name: "AMD Radeon RX 7900 XTX", Vendor: model.VendorAMD, Driver: "6.10.5",
		TotalBytes: 24 * model.GiB,
		UsedBytes:  24*model.GiB - 256*model.MiB,
		FreeBytes:  256 * model.MiB,
		Procs:      []model.Proc{{PID: 4242, Name: "ollama", UsedBytes: 22 * model.GiB}},
	}
	// 70B q4: ~19 GiB weights reported by loader; arch drives KV estimate.
	m := model.LoaderModel{
		Loader: "ollama", Name: "llama3:70b-q2_K", PID: 4242, GPUIndex: 0,
		VRAMBytes:     22 * model.GiB,
		ContextTokens: 8192, ContextMax: 8192,
		Arch: model.Arch{Name: "llama", Layers: 80, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	return gpu, []model.LoaderModel{m}
}

func TestAttributeGPUTilesExactly(t *testing.T) {
	gpu, models := scenario70B()
	segs, _ := AttributeGPU(gpu, models)

	var sum uint64
	kinds := map[model.SegmentKind]uint64{}
	for _, s := range segs {
		sum += s.Bytes
		kinds[s.Kind] = s.Bytes
	}
	if sum != gpu.TotalBytes {
		t.Fatalf("segments sum to %d, want total %d", sum, gpu.TotalBytes)
	}
	// KV at 8192 for 80-layer GQA f16 = 2*80*8*128*16/8 * 8192 = 2.5 GiB.
	wantKV := KVCacheBytes(models[0].Arch, 8192)
	if kinds[model.KindKVCache] != wantKV {
		t.Fatalf("kv segment = %d, want %d", kinds[model.KindKVCache], wantKV)
	}
	// Weights = process footprint (22 GiB) - KV.
	if kinds[model.KindWeights] != 22*model.GiB-wantKV {
		t.Fatalf("weights = %d, want %d", kinds[model.KindWeights], uint64(22*model.GiB)-wantKV)
	}
	// Other apps = device used (24 GiB - 256 MiB) - inference proc (22 GiB).
	if kinds[model.KindOtherProcess] != 2*model.GiB-256*model.MiB {
		t.Fatalf("other = %d, want %d", kinds[model.KindOtherProcess], uint64(2*model.GiB-256*model.MiB))
	}
	if kinds[model.KindFree] != 256*model.MiB {
		t.Fatalf("free = %d, want %d", kinds[model.KindFree], uint64(256*model.MiB))
	}
}

func TestAttributeGPUNoLoader(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorNVIDIA, TotalBytes: 8 * model.GiB, UsedBytes: 3 * model.GiB, FreeBytes: 5 * model.GiB}
	segs, _ := AttributeGPU(gpu, nil)
	var sum, other, free uint64
	for _, s := range segs {
		sum += s.Bytes
		switch s.Kind {
		case model.KindOtherProcess:
			other = s.Bytes
		case model.KindFree:
			free = s.Bytes
		}
	}
	if sum != 8*model.GiB || other != 3*model.GiB || free != 5*model.GiB {
		t.Fatalf("no-loader attribution wrong: sum=%d other=%d free=%d", sum, other, free)
	}
}

func TestPredict(t *testing.T) {
	gpu, models := scenario70B()
	p := Predict(gpu, models, DefaultOOMThreshold)
	if p == nil {
		t.Fatal("expected a prediction")
	}
	if !p.OOMRisk {
		t.Errorf("expected OOM risk with 512 MiB free")
	}
	if p.KVBytesPerToken != KVBytesPerToken(models[0].Arch) {
		t.Errorf("kv/token = %d", p.KVBytesPerToken)
	}
	// 512 MiB free / (2.5GiB/8192 per tok) ~ a few hundred extra tokens, capped
	// at the trained 8192 context.
	if p.MaxContextFits != 8192 {
		t.Errorf("MaxContextFits = %d, want capped at 8192", p.MaxContextFits)
	}
}

func TestWillContextFit(t *testing.T) {
	gpu, models := scenario70B()
	// Currently at 8k and nearly full: 32k must not fit.
	fit, ok := WillContextFit(gpu, models, 32768)
	if !ok {
		t.Fatal("expected fit computation")
	}
	if fit.Fits {
		t.Errorf("32k context should not fit on a nearly-full 24 GiB card")
	}
	if fit.KVAtTarget != KVCacheBytes(models[0].Arch, 32768) {
		t.Errorf("KVAtTarget wrong: %d", fit.KVAtTarget)
	}
	// The scenario model is trained to 8192, so 32768 exceeds it.
	if !fit.ExceedsTrained || fit.ModelContextMax != 8192 {
		t.Errorf("expected ExceedsTrained with trained max 8192, got %+v", fit)
	}
}

// Reported weights must survive a conflict with an over-estimated KV cache.
func TestAttributeReportedWeightsWin(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorAMD, TotalBytes: 24 * model.GiB, UsedBytes: 20 * model.GiB, FreeBytes: 4 * model.GiB}
	m := model.LoaderModel{
		Loader: "ollama", Name: "x", GPUIndex: 0,
		VRAMBytes:     20 * model.GiB,
		WeightsBytes:  19 * model.GiB, // loader-reported ground truth
		ContextTokens: 26000,          // KV estimate ~8 GiB, over the footprint
		Arch:          model.Arch{Name: "llama", Layers: 80, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	segs, _ := AttributeGPU(gpu, []model.LoaderModel{m})
	var sum uint64
	kinds := map[model.SegmentKind]uint64{}
	for _, s := range segs {
		sum += s.Bytes
		kinds[s.Kind] = s.Bytes
	}
	if sum != gpu.TotalBytes {
		t.Fatalf("segments sum %d != total %d", sum, gpu.TotalBytes)
	}
	if kinds[model.KindWeights] != 19*model.GiB {
		t.Errorf("reported weights should be preserved at 19 GiB, got %s", model.HumanBytes(kinds[model.KindWeights]))
	}
	if kinds[model.KindKVCache] > 1*model.GiB {
		t.Errorf("estimated KV should be shrunk to fit, got %s", model.HumanBytes(kinds[model.KindKVCache]))
	}
}

// When the KV estimate exceeds the footprint, weights must not be starved to 0.
func TestAttributeKVOvershootLeavesWeights(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorNVIDIA, TotalBytes: 4 * model.GiB, UsedBytes: 4 * model.GiB}
	m := model.LoaderModel{
		Loader: "ollama", Name: "tiny-longctx", GPUIndex: 0,
		VRAMBytes:     4 * model.GiB,
		ContextTokens: 16384, // KV estimate ~5 GiB > footprint
		Arch:          model.Arch{Name: "llama", Layers: 80, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	segs, warns := AttributeGPU(gpu, []model.LoaderModel{m})
	kinds := map[model.SegmentKind]uint64{}
	var sum uint64
	for _, s := range segs {
		sum += s.Bytes
		kinds[s.Kind] = s.Bytes
	}
	if sum != gpu.TotalBytes {
		t.Fatalf("segments sum %d != total %d", sum, gpu.TotalBytes)
	}
	if kinds[model.KindWeights] == 0 {
		t.Error("weights must never be 0 for a resident model (KV must not take the whole footprint)")
	}
	if kinds[model.KindKVCache] >= gpu.TotalBytes {
		t.Error("KV must not claim the entire footprint")
	}
	found := false
	for _, w := range warns {
		if len(w) > 0 {
			found = true
		}
	}
	if !found {
		t.Error("expected a warning that the KV estimate exceeded the footprint")
	}
}

// Prediction must fall back to a smaller known-arch model instead of giving up
// because the largest model has an unknown architecture.
func TestPredictPrefersKnownArch(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorNVIDIA, TotalBytes: 24 * model.GiB, UsedBytes: 18 * model.GiB, FreeBytes: 6 * model.GiB}
	unknown := model.LoaderModel{Loader: "ollama", Name: "novel-arch", GPUIndex: 0, VRAMBytes: 12 * model.GiB}
	known := model.LoaderModel{
		Loader: "ollama", Name: "llama3:8b", GPUIndex: 0, VRAMBytes: 6 * model.GiB,
		ContextTokens: 4096, ContextMax: 8192,
		Arch: model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	models := []model.LoaderModel{unknown, known}

	p := Predict(gpu, models, DefaultOOMThreshold)
	if p == nil {
		t.Fatal("expected a prediction from the smaller known-arch model")
	}
	if p.Model != "llama3:8b" {
		t.Errorf("prediction should be for the known model, got %q", p.Model)
	}
	if _, ok := WillContextFit(gpu, models, 4096); !ok {
		t.Error("WillContextFit should succeed using the known-arch model")
	}
}

// A user-declared quantized KV cache scales the estimate by its bit width.
func TestKVBitsOverride(t *testing.T) {
	gpu, models := scenario70B() // arch is f16 (16 bits)
	f16 := Build([]model.GPU{gpu}, models, Options{Version: "t"})
	q8 := Build([]model.GPU{gpu}, models, Options{Version: "t", KVBits: 8})

	kvF16, _ := f16.Breakdowns[0].Segment(model.KindKVCache)
	kvQ8, _ := q8.Breakdowns[0].Segment(model.KindKVCache)
	if kvQ8.Bytes*16 != kvF16.Bytes*8 {
		t.Errorf("KV at 8 bits (%s) should be 8/16 of f16 KV (%s)", model.HumanBytes(kvQ8.Bytes), model.HumanBytes(kvF16.Bytes))
	}
	// The override must also flow into the prediction's per-token figure.
	if q8.Breakdowns[0].Prediction.KVBytesPerToken*16 != f16.Breakdowns[0].Prediction.KVBytesPerToken*8 {
		t.Error("prediction kv/token should scale with the bit override")
	}
}

// A loader-reported (measured) KV cache must NOT be second-guessed by the flag.
func TestKVBitsOverrideKeepsReportedKV(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorNVIDIA, TotalBytes: 16 * model.GiB, UsedBytes: 10 * model.GiB, FreeBytes: 6 * model.GiB}
	m := model.LoaderModel{
		Loader: "x", Name: "reported-kv", GPUIndex: 0, VRAMBytes: 10 * model.GiB,
		KVCacheBytes:  3 * model.GiB, // measured, not estimated
		ContextTokens: 0,             // no per-token context reported
		Arch:          model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	snap := Build([]model.GPU{gpu}, []model.LoaderModel{m}, Options{Version: "t", KVBits: 8})
	kv, _ := snap.Breakdowns[0].Segment(model.KindKVCache)
	if kv.Bytes != 3*model.GiB {
		t.Errorf("reported KV should stay 3 GiB under the override, got %s", model.HumanBytes(kv.Bytes))
	}
}

// llama.cpp reports weights (from GGUF) but no VRAM total; attribution must
// still produce a real weights/KV split from the derived footprint.
func TestLlamaCppFootprintFallback(t *testing.T) {
	gpu := model.GPU{Index: 0, Vendor: model.VendorNVIDIA, TotalBytes: 16 * model.GiB, UsedBytes: 10 * model.GiB, FreeBytes: 6 * model.GiB}
	m := model.LoaderModel{
		Loader: "llama.cpp", Name: "Meta-Llama-3-8B.Q4_K_M", GPUIndex: 0,
		WeightsBytes:  5 * model.GiB, // from the GGUF file size; no VRAMBytes reported
		ContextTokens: 8192,
		Arch:          model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	segs, _ := AttributeGPU(gpu, []model.LoaderModel{m})
	var sum uint64
	kinds := map[model.SegmentKind]uint64{}
	for _, s := range segs {
		sum += s.Bytes
		kinds[s.Kind] = s.Bytes
	}
	if sum != gpu.TotalBytes {
		t.Fatalf("segments sum %d != total %d", sum, gpu.TotalBytes)
	}
	if kinds[model.KindWeights] != 5*model.GiB {
		t.Errorf("weights should be the GGUF-reported 5 GiB, got %s", model.HumanBytes(kinds[model.KindWeights]))
	}
	// KV at 8192 for llama-8b f16 = 1 GiB.
	if kinds[model.KindKVCache] != 1*model.GiB {
		t.Errorf("kv should be 1 GiB, got %s", model.HumanBytes(kinds[model.KindKVCache]))
	}
	// Prediction must work for llama.cpp too (arch known via GGUF).
	if p := Predict(gpu, []model.LoaderModel{m}, DefaultOOMThreshold); p == nil {
		t.Error("expected a prediction for a GGUF-backed llama.cpp model")
	}
}

// When a loader reports no PID, the footprint should come from the driver's
// process VRAM matched by NAME (e.g. an "ollama" process) instead of the
// loader's self-reported size.
func TestFootprintFromNamedProcess(t *testing.T) {
	gpu := model.GPU{
		Index: 0, Vendor: model.VendorAMD, TotalBytes: 24 * model.GiB,
		UsedBytes: 22*model.GiB + 512*model.MiB, FreeBytes: 24*model.GiB - (22*model.GiB + 512*model.MiB),
		// comm-style truncated name, driver-measured VRAM (includes runtime overhead).
		Procs: []model.Proc{{PID: 5001, Name: "ollama_llama_se", UsedBytes: 22 * model.GiB}},
	}
	m := model.LoaderModel{
		Loader: "ollama", Name: "llama3:8b", GPUIndex: 0, PID: 0, // no PID from /api/ps
		VRAMBytes:     20 * model.GiB, // loader self-report (lower than the real 22)
		ContextTokens: 8192,
		Arch:          model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16},
	}
	segs, _ := AttributeGPU(gpu, []model.LoaderModel{m})
	kinds := map[model.SegmentKind]uint64{}
	var sum uint64
	for _, s := range segs {
		sum += s.Bytes
		kinds[s.Kind] = s.Bytes
	}
	if sum != gpu.TotalBytes {
		t.Fatalf("segments sum %d != total %d", sum, gpu.TotalBytes)
	}
	// Footprint = 22 GiB (the named process), so weights = 22 GiB - KV, NOT
	// derived from the 20 GiB self-report. KV @ 8192 = 1 GiB.
	kv := KVCacheBytes(m.Arch, 8192)
	if kinds[model.KindWeights] != 22*model.GiB-kv {
		t.Errorf("weights = %s, want %s (footprint from named process)",
			model.HumanBytes(kinds[model.KindWeights]), model.HumanBytes(22*model.GiB-kv))
	}
	// Other apps = device used - 22 GiB footprint = 512 MiB.
	if kinds[model.KindOtherProcess] != 512*model.MiB {
		t.Errorf("other apps = %s, want 512 MiB", model.HumanBytes(kinds[model.KindOtherProcess]))
	}
}

func TestBuildDeterministic(t *testing.T) {
	gpu, models := scenario70B()
	now := time.Date(2026, 7, 5, 12, 0, 0, 0, time.UTC)
	snap := Build([]model.GPU{gpu}, models, Options{Version: "test", Host: "bench", Now: now})
	if len(snap.Breakdowns) != 1 {
		t.Fatalf("want 1 breakdown, got %d", len(snap.Breakdowns))
	}
	b := snap.Breakdowns[0]
	if b.Prediction == nil {
		t.Fatal("expected a prediction on the breakdown")
	}
	if !snap.Timestamp.Equal(now) {
		t.Errorf("timestamp not injected")
	}
}

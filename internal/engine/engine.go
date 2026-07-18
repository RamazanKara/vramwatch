// Package engine turns raw driver + loader observations into an attributed
// VRAM breakdown and an out-of-memory prediction. This is where vramwatch
// earns its keep: splitting an inference process's VRAM into weights vs KV
// cache vs compute, and projecting how much context fits before OOM.
package engine

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// DefaultOOMThreshold is the free-VRAM level below which a device is flagged
// as at risk of an out-of-memory failure on the next allocation.
const DefaultOOMThreshold = 512 * model.MiB

// Options tunes the attribution and prediction behaviour.
type Options struct {
	Version      string
	Host         string
	Now          time.Time // injectable for deterministic output; zero => time.Now()
	OOMThreshold uint64    // 0 => DefaultOOMThreshold
	// KVBits overrides the KV-cache element size (in bits) for every model, so a
	// quantized cache (q8_0 conservatively 9, q4_0 conservatively 5) is
	// estimated correctly instead of the f16 default. 0 leaves each model's own
	// value untouched.
	KVBits int
}

// maxPredictTokens caps projected context so a uint64->int narrowing can never
// overflow (matters only on 32-bit builds); far beyond any real context window.
const maxPredictTokens = 1 << 30

func (o Options) now() time.Time {
	if o.Now.IsZero() {
		return time.Now()
	}
	return o.Now
}

func (o Options) oomThreshold() uint64 {
	if o.OOMThreshold == 0 {
		return DefaultOOMThreshold
	}
	return o.OOMThreshold
}

// KVBytesPerToken returns the number of VRAM bytes one additional context
// token adds to the KV cache for a model with the given architecture.
//
// The formula is:
//
//	(key_dim + value_dim)
//	  * n_layers
//	  * n_kv_heads      (grouped-query / multi-query aware)
//	  * bytes_per_element
//
// KVTypeBits is used in bit units so quantized caches (e.g. q8_0 = 8 bits)
// use a conservative integer width. Returns 0 when the architecture is unknown.
func KVBytesPerToken(a model.Arch) uint64 {
	if !a.KnownForKV() {
		return 0
	}
	valueDim := a.ValueDim
	if valueDim == 0 {
		valueDim = a.HeadDim
	}
	headWidth := saturatingAdd(uint64(a.HeadDim), uint64(valueDim))
	bits := uint64(a.Layers)
	for _, factor := range []uint64{uint64(a.KVHeads), headWidth, uint64(a.KVTypeBits)} {
		bits = saturatingMul(bits, factor)
	}
	if bits == ^uint64(0) {
		return bits
	}
	bytes := bits / 8
	if bits%8 != 0 {
		bytes++
	}
	return bytes
}

// KVCacheBytes returns the total KV cache size for a model at ctxTokens.
func KVCacheBytes(a model.Arch, ctxTokens int) uint64 {
	if ctxTokens <= 0 {
		return 0
	}
	return saturatingMul(KVBytesPerToken(a), uint64(ctxTokens))
}

func saturatingAdd(a, b uint64) uint64 {
	c := a + b
	if c < a {
		return ^uint64(0)
	}
	return c
}

func saturatingMul(a, b uint64) uint64 {
	if a == 0 || b == 0 {
		return 0
	}
	if a > ^uint64(0)/b {
		return ^uint64(0)
	}
	return a * b
}

// modelKV resolves a model's KV-cache bytes, preferring a loader-reported value
// and falling back to the architecture-based estimate. reported is true when
// the figure came from the loader rather than being derived.
func modelKV(m model.LoaderModel) (kv uint64, reported bool) {
	if m.KVCacheBytes > 0 {
		return m.KVCacheBytes, true
	}
	return KVCacheBytes(m.Arch, m.ContextTokens), false
}

// AttributeGPU produces the fully-tiled segment list for one device given the
// models resident on it. The segments always sum exactly to gpu.TotalBytes.
//
// Reported figures (loader-provided weights/KV) are treated as ground truth and
// win any conflict; estimated figures are shrunk to fit the real footprint. An
// estimated KV never claims the whole footprint, since weights are always
// resident too.
func AttributeGPU(gpu model.GPU, models []model.LoaderModel) ([]model.Segment, []string) {
	segs, warnings, _ := attributeGPU(gpu, models)
	return segs, warnings
}

func attributeGPU(gpu model.GPU, models []model.LoaderModel) ([]model.Segment, []string, model.GPU) {
	var warnings []string

	total := gpu.TotalBytes
	used := gpu.UsedBytes
	if total == 0 {
		// Nothing sensible to attribute; surface a single unknown block.
		return []model.Segment{{Kind: model.KindFree, Label: "unknown", Bytes: 0, Source: string(gpu.Vendor)}}, warnings, gpu
	}
	if used > total {
		used = total
	}

	// Aggregate reported weights and KV (reported-or-estimated) across models.
	var reportedWeights, kvBytes, sumModelVRAM uint64
	weightsReported, anyModelEstimated := false, false
	kvReported := len(models) > 0
	for _, m := range models {
		if m.WeightsBytes > 0 {
			reportedWeights = saturatingAdd(reportedWeights, m.WeightsBytes)
			weightsReported = true
		}
		kv, rep := modelKV(m)
		kvBytes = saturatingAdd(kvBytes, kv)
		if !rep {
			kvReported = false
		}
		sumModelVRAM = saturatingAdd(sumModelVRAM, m.VRAMBytes)
		if m.Estimated {
			anyModelEstimated = true // e.g. weights derived from a GGUF file size
		}
		if !m.Arch.KnownForKV() && m.KVCacheBytes == 0 {
			warnings = append(warnings, fmt.Sprintf("model %q: architecture unknown, KV cache not estimated", m.Name))
		}
	}

	// Inference-process footprint on this device: prefer per-process driver
	// figures for the loader PIDs; else the loader's own reported VRAM total;
	// else (e.g. llama.cpp, which reports no VRAM) derive it from the model's
	// weights plus the estimated KV cache.
	infProc := procUsedFor(gpu, models)
	if infProc == 0 {
		infProc = sumModelVRAM
	}
	if infProc == 0 {
		infProc = saturatingAdd(reportedWeights, kvBytes)
	}
	usageProv := gpu.UsageSource
	if usageProv == "" {
		usageProv = model.ProvenanceMeasured
	}
	if infProc > used {
		if usageProv == model.ProvenanceAssumed || gpu.MemoryKind == model.MemoryUnified {
			used = infProc
			if used > total {
				used = total
			}
			usageProv = model.ProvenanceEstimated
			if gpu.MemoryKind == model.MemoryUnified {
				warnings = append(warnings, "unified-memory availability is a system-pressure proxy; used/free memory is floored by the resident model footprint")
			} else {
				warnings = append(warnings, "device usage is unavailable; used/free memory is inferred from the resident model")
			}
		} else {
			infProc = used
		}
	}
	if infProc > used {
		infProc = used
	}

	weights, kv := reportedWeights, kvBytes
	if kv > infProc {
		kv = infProc
	}
	weightsEstimated := false
	if weightsReported {
		// Reported weights win any conflict with an (often estimated) KV.
		if weights > infProc {
			weights = infProc
		}
		if kv > infProc-weights {
			kv = infProc - weights
		}
	} else {
		// Weights are derived from whatever the footprint leaves after KV. An
		// estimated KV must never claim the entire footprint.
		maxKV := infProc
		if !kvReported && infProc > 0 {
			maxKV = infProc - infProc/10 // reserve >=10% for weights
		}
		if kv > maxKV {
			if !kvReported {
				warnings = append(warnings, "KV-cache estimate exceeded the model footprint (a quantized KV cache would explain this); the weights/KV split is approximate")
			}
			kv = maxKV
		}
		weights = infProc - kv
		weightsEstimated = weights > 0
	}
	compute := infProc - weights - kv
	other := used - infProc
	free := total - used

	loaderSrc := loaderSource(models)
	segs := make([]model.Segment, 0, 5)
	if weights > 0 {
		estimated := weightsEstimated || anyModelEstimated
		prov := model.ProvenanceReported
		if estimated {
			prov = model.ProvenanceEstimated
		}
		segs = append(segs, model.Segment{Kind: model.KindWeights, Label: model.KindWeights.DefaultLabel(), Bytes: weights, Source: loaderSrc, Estimated: estimated, Provenance: prov})
	}
	if kv > 0 {
		prov := model.ProvenanceReported
		if !kvReported {
			prov = model.ProvenanceEstimated
		}
		segs = append(segs, model.Segment{Kind: model.KindKVCache, Label: model.KindKVCache.DefaultLabel(), Bytes: kv, Source: loaderSrc, Estimated: !kvReported, Provenance: prov})
	}
	if compute > 0 {
		segs = append(segs, model.Segment{Kind: model.KindCompute, Label: model.KindCompute.DefaultLabel(), Bytes: compute, Source: "footprint remainder", Estimated: true, Provenance: model.ProvenanceEstimated})
	}
	if other > 0 {
		segs = append(segs, model.Segment{Kind: model.KindOtherProcess, Label: model.KindOtherProcess.DefaultLabel(), Bytes: other, Source: "device minus model", Estimated: true, Provenance: model.ProvenanceEstimated})
	}
	segs = append(segs, model.Segment{Kind: model.KindFree, Label: model.KindFree.DefaultLabel(), Bytes: free, Source: "driver", Provenance: usageProv})
	gpu.UsedBytes = used
	gpu.FreeBytes = free
	gpu.UsageSource = usageProv
	return segs, warnings, gpu
}

// procUsedFor returns the driver-measured VRAM of the inference process on this
// device. It matches first by PID (when a loader reports one), then by process
// name (e.g. an "ollama" / "llama-server" process), which lets
// per-process VRAM stand in for a footprint the loader doesn't self-report.
// Returns 0 when nothing matches (the caller falls back to loader-reported VRAM).
func procUsedFor(gpu model.GPU, models []model.LoaderModel) uint64 {
	if len(gpu.Procs) == 0 {
		return 0
	}
	// 1. Exact PID match.
	pids := map[int]bool{}
	for _, m := range models {
		if m.PID > 0 {
			pids[m.PID] = true
		}
	}
	if len(pids) > 0 {
		var sum uint64
		for _, p := range gpu.Procs {
			if pids[p.PID] {
				sum = saturatingAdd(sum, p.UsedBytes)
			}
		}
		if sum > 0 {
			return sum
		}
	}
	// 2. Name match by loader. Take the single LARGEST match (the inference
	//    process dominates VRAM) rather than a sum, so a co-resident bystander
	//    with a similar name can't inflate the footprint.
	var best uint64
	for _, p := range gpu.Procs {
		if loaderMatches(p.Name, models) && p.UsedBytes > best {
			best = p.UsedBytes
		}
	}
	return best
}

// loaderMatches reports whether a process name looks like one of the resident
// loaders' server processes. It matches the executable basename against a
// narrow prefix set (e.g. "ollama*", "llama-*"), not a full-path substring, so
// an unrelated process on the same path won't collide.
func loaderMatches(rawName string, models []model.LoaderModel) bool {
	n := procBase(rawName)
	for _, m := range models {
		switch strings.ToLower(m.Loader) {
		case "ollama":
			if strings.HasPrefix(n, "ollama") {
				return true
			}
		case "llama.cpp":
			if strings.HasPrefix(n, "llama-") || n == "llama.cpp" {
				return true
			}
		}
	}
	return false
}

// procBase lower-cases a process name and reduces it to its executable
// basename (handling both / and \ separators and a trailing .exe).
func procBase(raw string) string {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if i := strings.LastIndexAny(raw, `/\`); i >= 0 {
		raw = raw[i+1:]
	}
	return strings.TrimSuffix(raw, ".exe")
}

func loaderSource(models []model.LoaderModel) string {
	if len(models) == 0 {
		return "driver"
	}
	return models[0].Loader
}

// Predict projects how much context the largest model on a device can hold
// before exhausting VRAM. Returns nil when no model with a known KV size is
// resident on the device.
func Predict(gpu model.GPU, models []model.LoaderModel, oomThreshold uint64) *model.Prediction {
	m, ok := primaryModel(models)
	if !ok {
		return nil
	}
	kvPerTok := KVBytesPerToken(m.Arch)
	if kvPerTok == 0 {
		return nil
	}
	free := gpu.FreeBytes
	if gpu.TotalBytes > 0 && gpu.UsedBytes <= gpu.TotalBytes {
		free = gpu.TotalBytes - gpu.UsedBytes
	}

	additional := free / kvPerTok
	if additional > maxPredictTokens {
		additional = maxPredictTokens // guard the uint64->int narrowing below
	}
	current := m.ContextTokens
	if current < 0 {
		current = 0
	}
	maxInt := int(^uint(0) >> 1)
	maxFits := maxInt
	if additional <= uint64(maxInt-current) {
		maxFits = current + int(additional)
	}
	if m.ContextMax > 0 && maxFits > m.ContextMax {
		maxFits = m.ContextMax
	}
	return &model.Prediction{
		Model:           m.Name,
		KVBytesPerToken: kvPerTok,
		ContextTokens:   m.ContextTokens,
		MaxContextFits:  maxFits,
		ModelContextMax: m.ContextMax,
		HeadroomBytes:   free,
		OOMRisk:         free < oomThreshold,
	}
}

// ContextFit describes whether a target context length fits on a device for
// the currently-loaded primary model.
type ContextFit struct {
	Model           string
	TargetContext   int
	NeededBytes     uint64 // weights + KV(target) + current compute overhead
	TotalBytes      uint64
	Fits            bool // fits in VRAM
	KVAtTarget      uint64
	ModelContextMax int  // model's trained context length (0 if unknown)
	ExceedsTrained  bool // target exceeds the trained context (fits in VRAM but risky)
}

// WillContextFit computes whether targetCtx fits for the primary model on the
// device, holding weights and compute overhead constant and scaling only the
// KV cache. Leaves a 2% device slack. Returns false, ok=false if no
// KV-known model is loaded.
func WillContextFit(gpu model.GPU, models []model.LoaderModel, targetCtx int) (ContextFit, bool) {
	if targetCtx < 0 {
		return ContextFit{}, false
	}
	m, ok := primaryModel(models)
	if !ok {
		return ContextFit{}, false
	}
	kvPerTok := KVBytesPerToken(m.Arch)
	if kvPerTok == 0 {
		return ContextFit{}, false
	}
	// Everything in the footprint that isn't the current KV cache (weights +
	// compute overhead) is held constant; only the KV cache scales with context.
	curKV, _ := modelKV(m)
	footprint := m.VRAMBytes
	if pu := procUsedFor(gpu, models); pu > 0 {
		footprint = pu
	}
	if footprint == 0 {
		footprint = saturatingAdd(m.WeightsBytes, curKV) // e.g. llama.cpp: weights from GGUF + KV
	}
	var base uint64
	if footprint > curKV {
		base = footprint - curKV
	}
	kvTarget := saturatingMul(kvPerTok, uint64(targetCtx))
	needed := saturatingAdd(base, kvTarget)
	budget := gpu.TotalBytes - gpu.TotalBytes/50 // 98% of total
	return ContextFit{
		Model:           m.Name,
		TargetContext:   targetCtx,
		NeededBytes:     needed,
		TotalBytes:      gpu.TotalBytes,
		Fits:            needed <= budget,
		KVAtTarget:      kvTarget,
		ModelContextMax: m.ContextMax,
		ExceedsTrained:  m.ContextMax > 0 && targetCtx > m.ContextMax,
	}, true
}

// primaryModel returns the largest-VRAM resident model whose architecture is
// known well enough to compute a KV cache. That is the model predictions and
// fit checks are about. A bigger model with an unknown architecture is skipped
// in favour of a smaller, fully-known one rather than giving up entirely.
func primaryModel(models []model.LoaderModel) (model.LoaderModel, bool) {
	var best model.LoaderModel
	found := false
	for _, m := range models {
		if !m.Arch.KnownForKV() {
			continue
		}
		if !found || m.VRAMBytes > best.VRAMBytes {
			best, found = m, true
		}
	}
	return best, found
}

// Build assembles a full Snapshot from raw GPU and loader observations,
// attributing every device and attaching predictions.
func Build(gpus []model.GPU, models []model.LoaderModel, opts Options) model.Snapshot {
	now := opts.now()
	host := opts.Host
	if host == "" {
		host, _ = os.Hostname()
	}
	snap := model.Snapshot{
		Version:   opts.Version,
		Host:      host,
		Timestamp: now,
	}
	if opts.KVBits > 0 {
		models = withKVBits(models, opts.KVBits)
	}
	for _, g := range gpus {
		if g.MemoryKind == "" {
			g.MemoryKind = model.MemoryDedicated
		}
		if g.BudgetBytes == 0 {
			g.BudgetBytes = g.TotalBytes
		}
		if g.CapacitySource == "" {
			g.CapacitySource = model.ProvenanceMeasured
		}
		if g.UsageSource == "" {
			g.UsageSource = model.ProvenanceMeasured
		}
		devModels := modelsForDevice(g, gpus, models)
		segs, warns, attributedGPU := attributeGPU(g, devModels)
		g = attributedGPU
		b := model.Breakdown{
			GPU:        g,
			Segments:   segs,
			Models:     devModels,
			Prediction: Predict(g, devModels, opts.oomThreshold()),
			Warnings:   warns,
			Timestamp:  now,
		}
		snap.Breakdowns = append(snap.Breakdowns, b)
	}
	return snap
}

// withKVBits returns a copy of models with the KV element size set to bits for
// every model whose KV cache is *estimated* (KVCacheBytes == 0), so a
// user-declared quantized cache is estimated with the right dtype. A loader that
// reported an exact KVCacheBytes is left untouched; a measured value is never
// second-guessed by the flag.
func withKVBits(models []model.LoaderModel, bits int) []model.LoaderModel {
	out := make([]model.LoaderModel, len(models))
	copy(out, models)
	for i := range out {
		if out[i].KVCacheBytes == 0 && out[i].Arch.KnownForKV() {
			out[i].Arch.KVTypeBits = bits
		}
	}
	return out
}

// modelsForDevice returns the models mapped to device g. A model with
// GPUIndex == g.Index maps there; a model with GPUIndex < 0 (unknown device)
// is attached to the sole GPU when there is exactly one.
func modelsForDevice(g model.GPU, allGPUs []model.GPU, models []model.LoaderModel) []model.LoaderModel {
	var out []model.LoaderModel
	single := len(allGPUs) == 1
	for _, m := range models {
		switch {
		case m.GPUIndex == g.Index:
			out = append(out, m)
		case m.GPUIndex < 0 && single:
			out = append(out, m)
		}
	}
	return out
}

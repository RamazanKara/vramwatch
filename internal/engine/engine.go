// Package engine turns raw driver + loader observations into an attributed
// VRAM breakdown and an out-of-memory prediction. This is where vramwatch
// earns its keep: splitting an inference process's VRAM into weights vs KV
// cache vs compute, and projecting how much context fits before OOM.
package engine

import (
	"fmt"
	"os"
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
// The standard formula is:
//
//	2 (one K tensor + one V tensor)
//	  * n_layers
//	  * n_kv_heads      (grouped-query / multi-query aware)
//	  * head_dim
//	  * bytes_per_element
//
// KVTypeBits is used in bit units so quantized caches (e.g. q8_0 = 8 bits)
// are handled exactly. Returns 0 when the architecture is unknown.
func KVBytesPerToken(a model.Arch) uint64 {
	if !a.KnownForKV() {
		return 0
	}
	bits := uint64(2) * uint64(a.Layers) * uint64(a.KVHeads) * uint64(a.HeadDim) * uint64(a.KVTypeBits)
	return bits / 8
}

// KVCacheBytes returns the total KV cache size for a model at ctxTokens.
func KVCacheBytes(a model.Arch, ctxTokens int) uint64 {
	if ctxTokens <= 0 {
		return 0
	}
	return KVBytesPerToken(a) * uint64(ctxTokens)
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
	var warnings []string

	total := gpu.TotalBytes
	used := gpu.UsedBytes
	if total == 0 {
		// Nothing sensible to attribute; surface a single unknown block.
		return []model.Segment{{Kind: model.KindFree, Label: "unknown", Bytes: 0, Source: string(gpu.Vendor)}}, warnings
	}
	if used > total {
		used = total
	}

	// Aggregate reported weights and KV (reported-or-estimated) across models.
	var reportedWeights, kvBytes, sumModelVRAM uint64
	weightsReported, kvReported := false, false
	for _, m := range models {
		if m.WeightsBytes > 0 {
			reportedWeights += m.WeightsBytes
			weightsReported = true
		}
		kv, rep := modelKV(m)
		kvBytes += kv
		if rep {
			kvReported = true
		}
		sumModelVRAM += m.VRAMBytes
		if !m.Arch.KnownForKV() && m.KVCacheBytes == 0 {
			warnings = append(warnings, fmt.Sprintf("model %q: architecture unknown, KV cache not estimated", m.Name))
		}
	}

	// Inference-process footprint on this device: prefer per-process driver
	// figures for the loader PIDs; else the loader's own VRAM total.
	infProc := procUsedFor(gpu, models)
	if infProc == 0 {
		infProc = sumModelVRAM
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
		if weights+kv > infProc {
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
		segs = append(segs, model.Segment{Kind: model.KindWeights, Label: model.KindWeights.DefaultLabel(), Bytes: weights, Source: loaderSrc, Estimated: weightsEstimated})
	}
	if kv > 0 {
		segs = append(segs, model.Segment{Kind: model.KindKVCache, Label: model.KindKVCache.DefaultLabel(), Bytes: kv, Source: loaderSrc, Estimated: !kvReported})
	}
	if compute > 0 {
		segs = append(segs, model.Segment{Kind: model.KindCompute, Label: model.KindCompute.DefaultLabel(), Bytes: compute, Source: "driver"})
	}
	if other > 0 {
		segs = append(segs, model.Segment{Kind: model.KindOtherProcess, Label: model.KindOtherProcess.DefaultLabel(), Bytes: other, Source: "driver"})
	}
	segs = append(segs, model.Segment{Kind: model.KindFree, Label: model.KindFree.DefaultLabel(), Bytes: free, Source: "driver"})
	return segs, warnings
}

// procUsedFor sums the driver-reported VRAM of processes whose PID matches a
// loader model on this device. Returns 0 when no match is found.
func procUsedFor(gpu model.GPU, models []model.LoaderModel) uint64 {
	if len(gpu.Procs) == 0 {
		return 0
	}
	pids := map[int]bool{}
	for _, m := range models {
		if m.PID > 0 {
			pids[m.PID] = true
		}
	}
	if len(pids) == 0 {
		return 0
	}
	var sum uint64
	for _, p := range gpu.Procs {
		if pids[p.PID] {
			sum += p.UsedBytes
		}
	}
	return sum
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
	maxFits := m.ContextTokens + int(additional)
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
	var base uint64
	if footprint > curKV {
		base = footprint - curKV
	}
	kvTarget := kvPerTok * uint64(targetCtx)
	needed := base + kvTarget
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
// known well enough to compute a KV cache — the one predictions and fit checks
// are about. A bigger model with an unknown architecture is skipped in favour
// of a smaller, fully-known one rather than giving up entirely.
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
	for _, g := range gpus {
		devModels := modelsForDevice(g, gpus, models)
		segs, warns := AttributeGPU(g, devModels)
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

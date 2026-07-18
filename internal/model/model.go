// Package model defines the core data types vramwatch uses to describe GPU
// memory, the inference models resident on a GPU, and the attributed VRAM
// breakdown that is the whole point of the tool.
package model

import (
	"fmt"
	"time"
)

// Vendor identifies the GPU vendor a device belongs to.
type Vendor string

const (
	VendorNVIDIA  Vendor = "nvidia"
	VendorAMD     Vendor = "amd"
	VendorIntel   Vendor = "intel"
	VendorApple   Vendor = "apple"
	VendorUnknown Vendor = "unknown"
)

// SegmentKind names one contiguous slice of a GPU's VRAM. Together the
// segments of a Breakdown tile the entire device: every byte is accounted
// for as exactly one kind.
type SegmentKind string

const (
	// KindWeights is the model parameters resident in VRAM.
	KindWeights SegmentKind = "weights"
	// KindKVCache is the attention key/value cache, which grows with context.
	KindKVCache SegmentKind = "kv_cache"
	// KindCompute is everything the inference process holds that is neither
	// weights nor KV cache: activations, scratch buffers, the CUDA/HIP
	// context, graph allocator slack, etc.
	KindCompute SegmentKind = "compute"
	// KindOtherProcess is VRAM used by processes other than the inference
	// loader: the desktop compositor, a browser, another model, a zombie.
	KindOtherProcess SegmentKind = "other_process"
	// KindFree is unused VRAM available for new allocations.
	KindFree SegmentKind = "free"
)

// Segment is one accounted slice of a GPU's VRAM.
type Segment struct {
	Kind       SegmentKind `json:"kind"`
	Label      string      `json:"label"`
	Bytes      uint64      `json:"bytes"`
	Source     string      `json:"source"`              // provider/method that produced this figure
	Estimated  bool        `json:"estimated,omitempty"` // true when derived rather than reported
	Provenance Provenance  `json:"provenance,omitempty"`
}

// Provenance describes how a displayed value was obtained. A loader report is
// kept distinct from a direct driver/OS measurement, and neither is confused
// with model math or a policy assumption.
type Provenance string

const (
	ProvenanceMeasured  Provenance = "measured"
	ProvenanceReported  Provenance = "loader_reported"
	ProvenanceEstimated Provenance = "model_estimated"
	ProvenanceAssumed   Provenance = "assumed"
	ProvenanceUser      Provenance = "user_supplied"
)

// MemoryKind distinguishes a dedicated accelerator pool from Apple unified
// memory and a manual planning budget.
type MemoryKind string

const (
	MemoryDedicated MemoryKind = "dedicated_vram"
	MemoryUnified   MemoryKind = "unified_memory"
	MemoryManual    MemoryKind = "manual"
)

// Proc is a process holding VRAM on a device, as reported by the driver.
type Proc struct {
	PID       int    `json:"pid"`
	Name      string `json:"name"`
	UsedBytes uint64 `json:"used_bytes"`
}

// GPU is device-level memory from a provider. CapacitySource and UsageSource
// state whether those values were measured or had to be assumed.
type GPU struct {
	Index          int        `json:"index"`
	Name           string     `json:"name"`
	Vendor         Vendor     `json:"vendor"`
	Driver         string     `json:"driver,omitempty"`
	PCIBus         string     `json:"pci_bus,omitempty"` // e.g. "0000:03:00.0"; used to map /proc fdinfo to a device
	TotalBytes     uint64     `json:"total_bytes"`
	UsedBytes      uint64     `json:"used_bytes"`
	FreeBytes      uint64     `json:"free_bytes"`
	Procs          []Proc     `json:"processes,omitempty"`
	MemoryKind     MemoryKind `json:"memory_kind,omitempty"`
	BudgetBytes    uint64     `json:"accelerator_budget_bytes,omitempty"`
	CapacitySource Provenance `json:"capacity_provenance,omitempty"`
	UsageSource    Provenance `json:"usage_provenance,omitempty"`
}

// Arch holds the architecture parameters needed to compute a model's KV
// cache size. Zero values mean "unknown" and disable KV estimation.
type Arch struct {
	Name       string `json:"name,omitempty"`
	Layers     int    `json:"layers,omitempty"`       // transformer blocks
	KVHeads    int    `json:"kv_heads,omitempty"`     // key/value heads (GQA/MQA aware)
	HeadDim    int    `json:"head_dim,omitempty"`     // per-head dimension
	ValueDim   int    `json:"value_dim,omitempty"`    // value-head dimension; zero => HeadDim
	KVTypeBits int    `json:"kv_type_bits,omitempty"` // effective integer bits per KV element (16=f16, 9≈q8_0, ...)
}

// KnownForKV reports whether Arch has everything needed to compute KV size.
func (a Arch) KnownForKV() bool {
	return a.Layers > 0 && a.KVHeads > 0 && a.HeadDim > 0 && a.KVTypeBits > 0
}

// LoaderModel is a model an inference loader reports as resident in VRAM.
type LoaderModel struct {
	Loader        string     `json:"loader"` // ollama | llama.cpp | ...
	Name          string     `json:"name"`
	PID           int        `json:"pid,omitempty"`
	GPUIndex      int        `json:"gpu_index"`  // best-effort device mapping; -1 if unknown
	VRAMBytes     uint64     `json:"vram_bytes"` // total resident footprint; see VRAMSource
	WeightsBytes  uint64     `json:"weights_bytes,omitempty"`
	KVCacheBytes  uint64     `json:"kv_cache_bytes,omitempty"`
	ContextTokens int        `json:"context_tokens,omitempty"` // configured n_ctx
	ContextMax    int        `json:"context_max,omitempty"`    // trained context length
	Arch          Arch       `json:"arch"`
	Estimated     bool       `json:"estimated,omitempty"` // weights/kv were derived, not reported
	Quantization  string     `json:"quantization,omitempty"`
	Digest        string     `json:"digest,omitempty"`
	ArtifactPath  string     `json:"artifact_path,omitempty"`
	VRAMSource    Provenance `json:"vram_provenance,omitempty"`
}

// Prediction estimates how much context a model can hold before it runs out
// of VRAM, using the linear-in-tokens growth of the KV cache.
type Prediction struct {
	Model           string `json:"model"`
	KVBytesPerToken uint64 `json:"kv_bytes_per_token"`
	ContextTokens   int    `json:"context_tokens"`    // current
	MaxContextFits  int    `json:"max_context_fits"`  // given current free headroom
	ModelContextMax int    `json:"model_context_max"` // 0 if unknown
	HeadroomBytes   uint64 `json:"headroom_bytes"`    // free VRAM at time of prediction
	OOMRisk         bool   `json:"oom_risk"`          // free headroom below the risk threshold
}

// Breakdown is a single GPU with its VRAM fully attributed.
type Breakdown struct {
	GPU        GPU           `json:"gpu"`
	Segments   []Segment     `json:"segments"`
	Models     []LoaderModel `json:"models,omitempty"`
	Prediction *Prediction   `json:"prediction,omitempty"`
	Warnings   []string      `json:"warnings,omitempty"`
	Timestamp  time.Time     `json:"timestamp"`
}

// Used returns the sum of all non-free segments.
func (b Breakdown) Used() uint64 {
	var u uint64
	for _, s := range b.Segments {
		if s.Kind != KindFree {
			u += s.Bytes
		}
	}
	return u
}

// Segment returns the first segment of the given kind and whether it exists.
func (b Breakdown) Segment(k SegmentKind) (Segment, bool) {
	for _, s := range b.Segments {
		if s.Kind == k {
			return s, true
		}
	}
	return Segment{}, false
}

// Snapshot is a full observation across all GPUs at one instant.
type Snapshot struct {
	Version    string      `json:"vramwatch_version"`
	Host       string      `json:"host,omitempty"`
	Breakdowns []Breakdown `json:"breakdowns"`
	Timestamp  time.Time   `json:"timestamp"`
}

// Byte-size units.
const (
	KiB = 1 << 10
	MiB = 1 << 20
	GiB = 1 << 30
	TiB = 1 << 40
)

// HumanBytes formats a byte count using binary (IEC) units with two
// significant fractional digits for GiB and above.
func HumanBytes(b uint64) string {
	switch {
	case b >= TiB:
		return fmt.Sprintf("%.2f TiB", float64(b)/TiB)
	case b >= GiB:
		return fmt.Sprintf("%.2f GiB", float64(b)/GiB)
	case b >= MiB:
		return fmt.Sprintf("%.1f MiB", float64(b)/MiB)
	case b >= KiB:
		return fmt.Sprintf("%.1f KiB", float64(b)/KiB)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// DefaultKind returns a stable display label for a segment kind.
func (k SegmentKind) DefaultLabel() string {
	switch k {
	case KindWeights:
		return "weights"
	case KindKVCache:
		return "KV cache"
	case KindCompute:
		return "compute"
	case KindOtherProcess:
		return "other apps"
	case KindFree:
		return "free"
	default:
		return string(k)
	}
}

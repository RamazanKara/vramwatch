package fit

import (
	"fmt"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

const PolicyVersion = "conservative-v1"

type Component struct {
	Bytes      uint64           `json:"bytes"`
	Provenance model.Provenance `json:"provenance"`
	Basis      string           `json:"basis"`
}

type Target struct {
	GPUIndex        int              `json:"gpu_index"`
	GPUName         string           `json:"gpu_name"`
	MemoryKind      model.MemoryKind `json:"memory_kind"`
	CapacityBytes   uint64           `json:"capacity_bytes"`
	CapacitySource  model.Provenance `json:"capacity_provenance"`
	AvailableBytes  uint64           `json:"available_bytes,omitempty"`
	AvailableKnown  bool             `json:"available_known"`
	AvailableSource model.Provenance `json:"available_provenance,omitempty"`
	Manual          bool             `json:"manual,omitempty"`
}

type Verdict string

const (
	VerdictFits           Verdict = "fits"
	VerdictDoesNotFit     Verdict = "does_not_fit"
	VerdictUnknown        Verdict = "unknown"
	VerdictContextTooLong Verdict = "context_unsupported"
)

type TargetResult struct {
	Target            Target    `json:"target"`
	FitsOnDevice      Verdict   `json:"fits_on_device"`
	FitsNow           Verdict   `json:"fits_now"`
	SafetyMargin      Component `json:"safety_margin"`
	RequiredBytes     uint64    `json:"required_bytes"`
	DeviceSpareBytes  int64     `json:"device_spare_bytes"`
	CurrentSpareBytes int64     `json:"current_spare_bytes,omitempty"`
}

type Result struct {
	Policy                     string         `json:"policy"`
	Artifact                   Artifact       `json:"artifact"`
	Context                    int            `json:"context"`
	KVCacheType                string         `json:"kv_cache_type"`
	KVCacheTypeAssumed         bool           `json:"kv_cache_type_assumed"`
	Weights                    Component      `json:"weights"`
	KVCache                    Component      `json:"kv_cache"`
	RuntimeExpected            Component      `json:"runtime_expected"`
	RuntimeCeiling             Component      `json:"runtime_ceiling"`
	ExpectedFootprintBytes     uint64         `json:"expected_footprint_bytes"`
	ConservativeFootprintBytes uint64         `json:"conservative_footprint_bytes"`
	Targets                    []TargetResult `json:"targets"`
	Confidence                 string         `json:"confidence"`
	Warnings                   []string       `json:"warnings,omitempty"`
}

type PredictOptions struct {
	Context             int
	KVCacheType         string
	KVCacheTypeExplicit bool
}

func Predict(a Artifact, targets []Target, opts PredictOptions) (Result, error) {
	if opts.Context <= 0 {
		return Result{}, fmt.Errorf("--context must be greater than zero")
	}
	if a.WeightBytes == 0 {
		return Result{}, fmt.Errorf("model weight size is unknown; refusing an optimistic fit prediction")
	}
	if len(targets) == 0 {
		return Result{}, fmt.Errorf("no accelerator target was provided")
	}
	kvType := strings.ToUpper(strings.TrimSpace(opts.KVCacheType))
	if kvType == "" {
		kvType = "F16"
	}
	kv, err := kvBytes(a.Arch, opts.Context, kvType)
	if err != nil {
		return Result{}, err
	}
	expectedRuntime := max64(64*model.MiB, roundUp(percentCeil(a.WeightBytes, 10), 16*model.MiB))
	ceilingRuntime := max64(256*model.MiB, roundUp(percentCeil(a.WeightBytes, 15), 16*model.MiB))
	expected := saturatingAdd(saturatingAdd(a.WeightBytes, kv), expectedRuntime)
	conservative := saturatingAdd(saturatingAdd(a.WeightBytes, kv), ceilingRuntime)
	r := Result{
		Policy: PolicyVersion, Artifact: a, Context: opts.Context, KVCacheType: kvType, KVCacheTypeAssumed: !opts.KVCacheTypeExplicit,
		Weights:                Component{Bytes: a.WeightBytes, Provenance: model.ProvenanceEstimated, Basis: a.WeightBasis + " mapped to fully resident weights"},
		KVCache:                Component{Bytes: kv, Provenance: model.ProvenanceEstimated, Basis: "architecture × context × KV element width"},
		RuntimeExpected:        Component{Bytes: expectedRuntime, Provenance: model.ProvenanceAssumed, Basis: "max(64 MiB, 10% of weights)"},
		RuntimeCeiling:         Component{Bytes: ceilingRuntime, Provenance: model.ProvenanceAssumed, Basis: "max(256 MiB, 15% of weights)"},
		ExpectedFootprintBytes: expected, ConservativeFootprintBytes: conservative, Confidence: "high",
	}
	if a.Quantization == "" {
		r.Confidence = "medium"
		r.Warnings = append(r.Warnings, "weight quantization could not be identified; file size is known, but verify that this is the intended artifact")
	}
	if !opts.KVCacheTypeExplicit {
		r.Confidence = "medium"
		r.Warnings = append(r.Warnings, "KV cache type assumed f16; declare --kv-cache-type if the loader uses a quantized cache")
	}
	contextOK := a.ContextMax == 0 || opts.Context <= a.ContextMax
	if !contextOK {
		r.Warnings = append(r.Warnings, fmt.Sprintf("requested context %d exceeds model context %d", opts.Context, a.ContextMax))
	}
	for _, t := range targets {
		capacity := t.CapacityBytes
		margin := max64(512*model.MiB, roundUp(percentCeil(capacity, 5), 16*model.MiB))
		required := saturatingAdd(conservative, margin)
		tr := TargetResult{
			Target: t, FitsOnDevice: verdict(required, capacity, contextOK), DeviceSpareBytes: delta(capacity, required), RequiredBytes: required,
			SafetyMargin: Component{Bytes: margin, Provenance: model.ProvenanceAssumed, Basis: "max(512 MiB, 5% of accelerator capacity)"},
		}
		if t.AvailableKnown || t.Manual {
			available := t.AvailableBytes
			if t.Manual && available == 0 {
				available = capacity
			}
			tr.FitsNow = knownVerdict(required, available, contextOK)
			tr.CurrentSpareBytes = delta(available, required)
		} else {
			tr.FitsNow = VerdictUnknown
		}
		r.Targets = append(r.Targets, tr)
	}
	return r, nil
}

func verdict(required, budget uint64, contextOK bool) Verdict {
	if !contextOK {
		return VerdictContextTooLong
	}
	if budget == 0 {
		return VerdictUnknown
	}
	if required <= budget {
		return VerdictFits
	}
	return VerdictDoesNotFit
}

func knownVerdict(required, budget uint64, contextOK bool) Verdict {
	if !contextOK {
		return VerdictContextTooLong
	}
	if required <= budget {
		return VerdictFits
	}
	return VerdictDoesNotFit
}

type rational struct{ num, den uint64 }

func kvWidth(name string) (rational, error) {
	switch strings.ToUpper(name) {
	case "F32":
		return rational{32, 1}, nil
	case "F16", "FP16", "BF16":
		return rational{16, 1}, nil
	case "Q8_0", "Q8":
		return rational{17, 2}, nil
	case "Q5_0", "Q5":
		return rational{11, 2}, nil
	case "Q5_1":
		return rational{6, 1}, nil
	case "Q4_0", "Q4":
		return rational{9, 2}, nil
	case "Q4_1":
		return rational{5, 1}, nil
	default:
		return rational{}, fmt.Errorf("unknown --kv-cache-type %q", name)
	}
}

func kvBytes(a model.Arch, context int, dtype string) (uint64, error) {
	if !a.KnownForKV() {
		return 0, fmt.Errorf("model architecture is incomplete")
	}
	w, err := kvWidth(dtype)
	if err != nil {
		return 0, err
	}
	vd := a.ValueDim
	if vd == 0 {
		vd = a.HeadDim
	}
	units := uint64(context)
	headWidth := saturatingAdd(uint64(a.HeadDim), uint64(vd))
	for _, factor := range []uint64{uint64(a.Layers), uint64(a.KVHeads), headWidth} {
		units = saturatingMul(units, factor)
	}
	n := saturatingMul(units, w.num)
	d := w.den * 8
	if n == ^uint64(0) {
		return n, nil
	}
	return ceilDiv(n, d), nil
}

func ceilDiv(n, d uint64) uint64 {
	if d == 0 {
		return ^uint64(0)
	}
	q := n / d
	if n%d != 0 {
		return saturatingAdd(q, 1)
	}
	return q
}

func roundUp(v, unit uint64) uint64 {
	if unit == 0 {
		return v
	}
	if v > ^uint64(0)-(unit-1) {
		return ^uint64(0)
	}
	return ((v + unit - 1) / unit) * unit
}
func percentCeil(v, p uint64) uint64 {
	whole := saturatingMul(v/100, p)
	fraction := ((v%100)*p + 99) / 100
	return saturatingAdd(whole, fraction)
}
func max64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
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
func delta(a, b uint64) int64 {
	if a >= b {
		d := a - b
		if d > uint64(^uint64(0)>>1) {
			return int64(^uint64(0) >> 1)
		}
		return int64(d)
	}
	d := b - a
	if d > uint64(^uint64(0)>>1) {
		return -int64(^uint64(0) >> 1)
	}
	return -int64(d)
}

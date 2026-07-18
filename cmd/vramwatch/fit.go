package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	fitengine "github.com/RamazanKara/vramwatch/internal/fit"
	"github.com/RamazanKara/vramwatch/internal/gpu"
	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/model"
)

type fitEnvelope struct {
	SchemaVersion int              `json:"schema_version"`
	Command       string           `json:"command"`
	RecordID      string           `json:"record_id,omitempty"`
	Result        fitengine.Result `json:"result"`
}

func cmdFit(args []string) error {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	quant := fs.String("quant", "", "weight quantization, for example q4_k_m")
	ctxTokens := fs.Int("context", 0, "target context length in tokens (required)")
	kvType := fs.String("kv-cache-type", "", "KV cache type: f16|bf16|f32|q8_0|q5_0|q5_1|q4_0|q4_1 (default f16)")
	loaderName := fs.String("loader", "auto", "target loader: auto|ollama|llama.cpp")
	gpuIndex := fs.Int("gpu", -1, "evaluate only this GPU index")
	manualVRAM := fs.String("vram", "", "manual accelerator budget, for example 24GiB")
	revision := fs.String("revision", "", "Hugging Face revision")
	file := fs.String("file", "", "exact GGUF filename when a repository is ambiguous")
	asJSON := fs.Bool("json", false, "print stable machine-readable JSON")
	noRecord := fs.Bool("no-record", false, "do not save this prediction to the local ledger")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch fit MODEL --quant q4_k_m --context 32768\n\nMODEL accepts a local GGUF, HTTPS URL, hf:owner/repo, or ollama:name:tag.\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, interspersedFitArgs(args)); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return &usageError{fmt.Errorf("fit requires exactly one MODEL")}
	}
	if *ctxTokens <= 0 {
		return &usageError{fmt.Errorf("--context is required and must be greater than zero")}
	}
	if *manualVRAM != "" && *gpuIndex >= 0 {
		return &usageError{fmt.Errorf("--vram and --gpu are mutually exclusive")}
	}
	if *gpuIndex < -1 {
		return &usageError{fmt.Errorf("--gpu must be zero or greater")}
	}
	loader := *loaderName
	if loader == "auto" {
		if strings.HasPrefix(fs.Arg(0), "ollama:") || (!strings.Contains(fs.Arg(0), "/") && !fileExistsCLI(fs.Arg(0))) {
			loader = "ollama"
		} else {
			loader = "llama.cpp"
		}
	}
	if loader != "ollama" && loader != "llama.cpp" {
		return &usageError{fmt.Errorf("unknown --loader %q", loader)}
	}

	artifact, err := fitengine.Resolve(context.Background(), fs.Arg(0), fitengine.ResolveOptions{Quant: *quant, Revision: *revision, File: *file, HFToken: os.Getenv("HF_TOKEN")})
	if err != nil {
		return err
	}
	targets, err := fitTargets(context.Background(), *gpuIndex, *manualVRAM)
	if err != nil {
		if *manualVRAM != "" {
			return &usageError{err}
		}
		return err
	}
	result, err := fitengine.Predict(artifact, targets, fitengine.PredictOptions{Context: *ctxTokens, KVCacheType: *kvType, KVCacheTypeExplicit: *kvType != ""})
	if err != nil {
		return &usageError{err}
	}
	var recordID string
	if !*noRecord {
		rec, err := ledger.Save(result, loader)
		if err != nil {
			return fmt.Errorf("save prediction ledger: %w", err)
		}
		recordID = rec.ID
	}
	if *asJSON {
		b, err := json.MarshalIndent(fitEnvelope{SchemaVersion: 1, Command: "fit", RecordID: recordID, Result: result}, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
	} else {
		printFit(result, recordID, cf.resolve())
	}
	if fitSucceeded(result) {
		return nil
	}
	if fitIndeterminate(result) {
		return &exitError{code: 1, quiet: true, err: fmt.Errorf("fit could not be determined")}
	}
	return &exitError{code: 3, quiet: true, err: fmt.Errorf("model does not fit")}
}

// interspersedFitArgs permits the natural documented form
// "fit MODEL --quant ... --context ..." even though the standard flag package
// stops parsing at its first positional argument. The FlagSet still owns all
// validation and error messages; this only moves positional arguments last.
func interspersedFitArgs(args []string) []string {
	valueFlags := map[string]bool{
		"quant": true, "context": true, "kv-cache-type": true, "loader": true,
		"gpu": true, "vram": true, "revision": true, "file": true,
	}
	var flags, positional []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			positional = append(positional, args[i+1:]...)
			break
		}
		if !strings.HasPrefix(a, "-") || a == "-" {
			positional = append(positional, a)
			continue
		}
		flags = append(flags, a)
		name := strings.TrimLeft(strings.SplitN(a, "=", 2)[0], "-")
		if valueFlags[name] && !strings.Contains(a, "=") && i+1 < len(args) {
			i++
			flags = append(flags, args[i])
		}
	}
	return append(flags, positional...)
}

func fitTargets(ctx context.Context, index int, manual string) ([]fitengine.Target, error) {
	if manual != "" {
		n, err := parseByteSize(manual)
		if err != nil {
			return nil, err
		}
		return []fitengine.Target{{GPUIndex: 0, GPUName: "manual budget", MemoryKind: model.MemoryManual, CapacityBytes: n, CapacitySource: model.ProvenanceUser, AvailableBytes: n, AvailableKnown: true, AvailableSource: model.ProvenanceUser, Manual: true}}, nil
	}
	gpus, err := gpu.Sample(ctx)
	if err != nil {
		return nil, err
	}
	var out []fitengine.Target
	for _, g := range gpus {
		if index >= 0 && g.Index != index {
			continue
		}
		cap := g.BudgetBytes
		if cap == 0 {
			cap = g.TotalBytes
		}
		kind := g.MemoryKind
		if kind == "" {
			kind = model.MemoryDedicated
		}
		availableKnown := g.UsageSource == model.ProvenanceMeasured || g.UsageSource == model.ProvenanceReported
		capacitySource := g.CapacitySource
		if capacitySource == "" {
			capacitySource = model.ProvenanceMeasured
		}
		out = append(out, fitengine.Target{GPUIndex: g.Index, GPUName: g.Name, MemoryKind: kind, CapacityBytes: cap, CapacitySource: capacitySource, AvailableBytes: g.FreeBytes, AvailableKnown: availableKnown, AvailableSource: g.UsageSource})
	}
	if len(out) == 0 {
		if index >= 0 {
			return nil, fmt.Errorf("GPU %d was not detected; run vramwatch doctor", index)
		}
		return nil, fmt.Errorf("no accelerator detected; run vramwatch doctor or pass --vram")
	}
	return out, nil
}

func printFit(r fitengine.Result, id string, color bool) {
	fmt.Printf("%s  %s  %s\n", bold(color, r.Artifact.CanonicalID), r.Artifact.Quantization, dimc(color, fmt.Sprintf("ctx %s", commas(r.Context))))
	fmt.Printf("  [E] weights          %10s  %s\n", model.HumanBytes(r.Weights.Bytes), r.Weights.Basis)
	fmt.Printf("  [E] KV cache         %10s  %s\n", model.HumanBytes(r.KVCache.Bytes), r.KVCacheType)
	fmt.Printf("  [A] runtime expected %10s\n", model.HumanBytes(r.RuntimeExpected.Bytes))
	fmt.Printf("  [A] runtime ceiling  %10s\n", model.HumanBytes(r.RuntimeCeiling.Bytes))
	fmt.Printf("      expected total   %10s\n", model.HumanBytes(r.ExpectedFootprintBytes))
	fmt.Printf("      conservative     %10s  %s confidence\n", model.HumanBytes(r.ConservativeFootprintBytes), r.Confidence)
	for _, t := range r.Targets {
		freeNow := "[?] unknown"
		if t.Target.AvailableKnown {
			freeNow = "[" + fitProvenanceBadge(t.Target.AvailableSource) + "] " + model.HumanBytes(t.Target.AvailableBytes)
		}
		fmt.Printf("\n  GPU %d  %s  %s  [%s] %s total / %s free now\n", t.Target.GPUIndex, t.Target.GPUName, fitMemoryKind(t.Target.MemoryKind), fitProvenanceBadge(t.Target.CapacitySource), model.HumanBytes(t.Target.CapacityBytes), freeNow)
		fmt.Printf("    required: %s  (includes [A] %s safety margin)\n", model.HumanBytes(t.RequiredBytes), model.HumanBytes(t.SafetyMargin.Bytes))
		fmt.Printf("    on device: %s", fitWord(t.FitsOnDevice, color))
		if t.DeviceSpareBytes != 0 {
			fmt.Printf("  (%s)", signedBytes(t.DeviceSpareBytes))
		}
		fmt.Println()
		fmt.Printf("    right now: %s", fitWord(t.FitsNow, color))
		if t.CurrentSpareBytes != 0 {
			fmt.Printf("  (%s)", signedBytes(t.CurrentSpareBytes))
		}
		fmt.Println()
	}
	for _, w := range r.Warnings {
		fmt.Printf("  %s %s\n", dimc(color, "note:"), w)
	}
	if id != "" {
		fmt.Printf("\n  prediction %s saved locally\n", id)
	}
	fmt.Println("  [M] measured  [R] loader-reported  [E] model-estimated  [A] assumed  [U] user-supplied")
}

func fitProvenanceBadge(p model.Provenance) string {
	switch p {
	case model.ProvenanceMeasured:
		return "M"
	case model.ProvenanceReported:
		return "R"
	case model.ProvenanceAssumed:
		return "A"
	case model.ProvenanceUser:
		return "U"
	default:
		return "E"
	}
}

func fitMemoryKind(k model.MemoryKind) string {
	switch k {
	case model.MemoryUnified:
		return "(unified memory)"
	case model.MemoryManual:
		return "(manual budget)"
	default:
		return "(dedicated VRAM)"
	}
}

func fitWord(v fitengine.Verdict, color bool) string {
	switch v {
	case fitengine.VerdictFits:
		return greenc(color, "FITS")
	case fitengine.VerdictDoesNotFit:
		return redc(color, "DOES NOT FIT")
	case fitengine.VerdictContextTooLong:
		return redc(color, "CONTEXT UNSUPPORTED")
	default:
		return dimc(color, "UNKNOWN")
	}
}
func fitSucceeded(r fitengine.Result) bool {
	for _, t := range r.Targets {
		if t.FitsNow == fitengine.VerdictFits || (t.FitsNow == fitengine.VerdictUnknown && t.FitsOnDevice == fitengine.VerdictFits) {
			return true
		}
	}
	return false
}

func fitIndeterminate(r fitengine.Result) bool {
	for _, t := range r.Targets {
		if t.FitsNow == fitengine.VerdictUnknown && t.FitsOnDevice == fitengine.VerdictUnknown {
			return true
		}
	}
	return false
}
func signedBytes(v int64) string {
	if v >= 0 {
		return model.HumanBytes(uint64(v)) + " spare"
	}
	return model.HumanBytes(uint64(-v)) + " short"
}
func fileExistsCLI(p string) bool { _, err := os.Stat(p); return err == nil }

func parseByteSize(raw string) (uint64, error) {
	s := strings.ToUpper(strings.TrimSpace(raw))
	mult := uint64(1)
	units := []struct {
		s string
		m uint64
	}{{"TIB", model.TiB}, {"TB", model.TiB}, {"GIB", model.GiB}, {"GB", model.GiB}, {"MIB", model.MiB}, {"MB", model.MiB}, {"KIB", model.KiB}, {"KB", model.KiB}, {"B", 1}}
	for _, u := range units {
		if strings.HasSuffix(s, u.s) {
			mult = u.m
			s = strings.TrimSpace(strings.TrimSuffix(s, u.s))
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f <= 0 || math.IsNaN(f) || math.IsInf(f, 0) || f >= float64(^uint64(0))/float64(mult) {
		return 0, fmt.Errorf("invalid byte size %q", raw)
	}
	return uint64(f * float64(mult)), nil
}

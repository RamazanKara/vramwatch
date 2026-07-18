package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
)

type reportEnvelope struct {
	SchemaVersion int             `json:"schema_version"`
	Command       string          `json:"command"`
	Record        ledger.Record   `json:"record"`
	Hardware      *reportHardware `json:"hardware,omitempty"`
}
type reportHardware struct {
	GPUIndex      int              `json:"gpu_index"`
	GPUName       string           `json:"gpu_name"`
	MemoryKind    model.MemoryKind `json:"memory_kind"`
	CapacityBytes uint64           `json:"capacity_bytes"`
	Driver        string           `json:"driver,omitempty"`
}

func cmdReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ContinueOnError)
	predictionID := fs.String("prediction", "", "prediction ledger ID (default latest)")
	asJSON := fs.Bool("json", false, "print stable machine-readable JSON")
	asSVG := fs.Bool("svg", false, "write a shareable SVG card")
	output := fs.String("output", "", "output path ('-' for stdout)")
	force := fs.Bool("force", false, "replace an existing output file")
	static := fs.Bool("static", false, "omit the live timestamp for deterministic output")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch report [--svg] [--prediction ID]\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return &usageError{fmt.Errorf("report takes no positional arguments")}
	}
	if *asJSON && *asSVG {
		return &usageError{fmt.Errorf("--json and --svg are mutually exclusive")}
	}
	if *output != "" && !*asSVG {
		return &usageError{fmt.Errorf("--output requires --svg")}
	}
	if *force && !*asSVG {
		return &usageError{fmt.Errorf("--force requires --svg")}
	}
	var rec ledger.Record
	var err error
	if *predictionID != "" {
		rec, err = ledger.Load(*predictionID)
	} else {
		rec, err = ledger.Latest()
	}
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("no saved prediction; run vramwatch fit first")
	}
	if err != nil {
		return err
	}
	hw, footprint, prov, source := observePrediction(rec)
	if footprint > 0 {
		updated, e := ledger.UpdateObservation(rec.ID, footprint, prov, source)
		if e != nil {
			return fmt.Errorf("save prediction observation: %w", e)
		}
		rec = updated
	}
	env := reportEnvelope{SchemaVersion: 1, Command: "report", Record: rec, Hardware: hw}
	if *asJSON {
		b, err := json.MarshalIndent(env, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
		return nil
	}
	card := cardFromReport(rec, hw, *static)
	if *asSVG {
		svg := render.ReportSVG(card)
		dest := *output
		if dest == "" {
			dest = "vramwatch-report-" + time.Now().Format("20060102-150405") + ".svg"
		}
		if dest == "-" {
			fmt.Println(svg)
			return nil
		}
		if err := writeReportSVG(dest, svg+"\n", *force); err != nil {
			return err
		}
		abs, _ := filepath.Abs(dest)
		fmt.Println(abs)
		return nil
	}
	printReport(rec, hw, cf.resolve())
	return nil
}

func writeReportSVG(path, data string, force bool) error {
	if force {
		dir := filepath.Dir(path)
		f, err := os.CreateTemp(dir, ".vramwatch-report-*.tmp")
		if err != nil {
			return err
		}
		tmp := f.Name()
		defer os.Remove(tmp)
		if err := f.Chmod(0o644); err != nil {
			f.Close()
			return err
		}
		if _, err := f.WriteString(data); err != nil {
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		return os.Rename(tmp, path)
	}
	flags := os.O_WRONLY | os.O_CREATE
	flags |= os.O_EXCL
	f, err := os.OpenFile(path, flags, 0o644)
	if errors.Is(err, os.ErrExist) {
		return fmt.Errorf("%s already exists (use --force)", path)
	}
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if _, err := f.WriteString(data); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}

func observePrediction(rec ledger.Record) (*reportHardware, uint64, model.Provenance, string) {
	snap, _, err := collect(context.Background(), "", 0)
	if err != nil {
		return nil, 0, "", ""
	}
	for _, bd := range snap.Breakdowns {
		if len(bd.Models) != 1 {
			continue
		}
		m := bd.Models[0]
		if !recordMatchesModel(rec, m) {
			continue
		}
		hw := &reportHardware{GPUIndex: bd.GPU.Index, GPUName: bd.GPU.Name, MemoryKind: bd.GPU.MemoryKind, CapacityBytes: bd.GPU.TotalBytes, Driver: bd.GPU.Driver}
		if hw.MemoryKind == "" {
			hw.MemoryKind = model.MemoryDedicated
		}
		if bd.GPU.BudgetBytes > 0 {
			hw.CapacityBytes = bd.GPU.BudgetBytes
		}
		var fp uint64
		for _, s := range bd.Segments {
			if s.Kind == model.KindWeights || s.Kind == model.KindKVCache || s.Kind == model.KindCompute {
				fp = saturatingBytes(fp, s.Bytes)
			}
		}
		prov := model.ProvenanceEstimated
		source := "attributed model footprint"
		if measured := measuredLoaderFootprint(bd.GPU, bd.Models); measured > 0 {
			fp = measured
			prov = model.ProvenanceMeasured
			source = "driver process memory"
		} else if m.VRAMBytes > 0 {
			prov = loaderVRAMProvenance(m)
			source = m.Loader + " model footprint"
			fp = m.VRAMBytes
		}
		return hw, fp, prov, source
	}
	return nil, 0, "", ""
}

func measuredLoaderFootprint(gpu model.GPU, models []model.LoaderModel) uint64 {
	pids := map[int]bool{}
	for _, m := range models {
		if m.PID > 0 {
			pids[m.PID] = true
		}
	}
	var sum uint64
	for _, p := range gpu.Procs {
		if pids[p.PID] {
			sum = saturatingBytes(sum, p.UsedBytes)
		}
	}
	if sum > 0 {
		return sum
	}
	var best uint64
	for _, p := range gpu.Procs {
		if processMatchesLoader(p.Name, models) && p.UsedBytes > best {
			best = p.UsedBytes
		}
	}
	return best
}

func recordMatchesModel(rec ledger.Record, m model.LoaderModel) bool {
	a := rec.Prediction.Artifact
	// Ollama exposes the manifest digest for a running model. Other artifact
	// sources use different digest namespaces (for example a Hub commit), so only
	// compare digests when they describe the same kind of identity.
	digestsKnown := a.Source == "ollama" && a.Digest != "" && m.Digest != ""
	if digestsKnown && !digestEqual(a.Digest, m.Digest) {
		return false
	}
	names := []string{a.CanonicalID, a.Filename, strings.TrimSuffix(filepath.Base(a.Filename), filepath.Ext(a.Filename))}
	match := false
	for _, n := range names {
		if n != "" && (strings.EqualFold(n, m.Name) || strings.EqualFold(filepath.Base(n), m.Name)) {
			match = true
		}
	}
	if !match && digestsKnown {
		match = true
	}
	if !match {
		return false
	}
	if rec.Prediction.Context != m.ContextTokens {
		return false
	}
	if a.Quantization != "" && m.Quantization != "" && !strings.EqualFold(a.Quantization, m.Quantization) {
		return false
	}
	return true
}

func digestEqual(a, b string) bool {
	normalize := func(s string) string {
		s = strings.ToLower(strings.TrimSpace(s))
		return strings.TrimPrefix(s, "sha256:")
	}
	return normalize(a) == normalize(b)
}

func cardFromReport(rec ledger.Record, hw *reportHardware, static bool) render.ReportCard {
	targetIndex := reportTargetIndex(rec, hw)
	c := render.ReportCard{Version: Version, GeneratedAt: time.Now(), PredictionID: rec.ID, Loader: rec.Loader, Model: shareableModelName(rec), Quant: rec.Prediction.Artifact.Quantization, Context: rec.Prediction.Context, KVType: rec.Prediction.KVCacheType, PredictedBytes: rec.Prediction.ExpectedFootprintBytes, Fits: reportFitStatus(rec, targetIndex)}
	if hw != nil {
		c.GPUName = hw.GPUName
		c.MemoryKind = hw.MemoryKind
		c.CapacityBytes = hw.CapacityBytes
		c.Driver = hw.Driver
	} else if targetIndex >= 0 {
		t := rec.Prediction.Targets[targetIndex].Target
		c.GPUName = t.GPUName
		c.MemoryKind = t.MemoryKind
		c.CapacityBytes = t.CapacityBytes
	}
	if rec.Observation != nil {
		c.ObservedBytes = rec.Observation.FootprintBytes
		c.ObservationProvenance = rec.Observation.Provenance
		c.AbsoluteErrorPct = rec.Observation.AbsoluteErrorPct
		c.SignedErrorPct = rec.Observation.SignedErrorPct
	}
	if static {
		c.GeneratedAt = time.Time{}
	}
	return c
}

func shareableModelName(rec ledger.Record) string {
	a := rec.Prediction.Artifact
	if (a.Source == "local" || a.Source == "url") && a.Filename != "" {
		return filepath.Base(strings.ReplaceAll(a.Filename, `\`, "/"))
	}
	if a.CanonicalID != "" {
		return a.CanonicalID
	}
	return filepath.Base(strings.ReplaceAll(a.Filename, `\`, "/"))
}

func reportTargetIndex(rec ledger.Record, hw *reportHardware) int {
	if hw != nil {
		fallback := -1
		for i, t := range rec.Prediction.Targets {
			if t.Target.GPUIndex != hw.GPUIndex {
				continue
			}
			if strings.EqualFold(t.Target.GPUName, hw.GPUName) {
				return i
			}
			if fallback < 0 {
				fallback = i
			}
		}
		if fallback >= 0 {
			return fallback
		}
	}
	for i, t := range rec.Prediction.Targets {
		if t.FitsNow == "fits" || (t.FitsNow == "unknown" && t.FitsOnDevice == "fits") {
			return i
		}
	}
	if len(rec.Prediction.Targets) > 0 {
		return 0
	}
	return -1
}

func reportFitStatus(rec ledger.Record, targetIndex int) string {
	if targetIndex < 0 || targetIndex >= len(rec.Prediction.Targets) {
		return "unknown"
	}
	t := rec.Prediction.Targets[targetIndex]
	if t.FitsNow == "context_unsupported" || t.FitsOnDevice == "context_unsupported" {
		return "context unsupported"
	}
	if t.FitsNow == "fits" || (t.FitsNow == "unknown" && t.FitsOnDevice == "fits") {
		return "fits"
	}
	if t.FitsNow == "does_not_fit" || t.FitsOnDevice == "does_not_fit" {
		return "does not fit"
	}
	return "unknown"
}

func printReport(rec ledger.Record, hw *reportHardware, color bool) {
	fmt.Printf("%s  prediction %s\n", bold(color, "vramwatch report"), rec.ID)
	fmt.Printf("  model: %s  %s  ctx %s\n", rec.Prediction.Artifact.CanonicalID, rec.Prediction.Artifact.Quantization, commas(rec.Prediction.Context))
	if hw != nil {
		fmt.Printf("  hardware: %s  %s  %s\n", hw.GPUName, fitMemoryKind(hw.MemoryKind), model.HumanBytes(hw.CapacityBytes))
	} else if targetIndex := reportTargetIndex(rec, nil); targetIndex >= 0 {
		t := rec.Prediction.Targets[targetIndex].Target
		fmt.Printf("  hardware: %s  %s  %s\n", t.GPUName, fitMemoryKind(t.MemoryKind), model.HumanBytes(t.CapacityBytes))
	}
	fmt.Printf("  predicted: %s  [E]\n", model.HumanBytes(rec.Prediction.ExpectedFootprintBytes))
	if rec.Observation == nil {
		fmt.Println("  observed: pending — load the model and run watch/report again")
		fmt.Println("  accuracy: pending")
		return
	}
	fmt.Printf("  observed: %s  [%s]\n", model.HumanBytes(rec.Observation.FootprintBytes), fitProvenanceBadge(rec.Observation.Provenance))
	fmt.Printf("  accuracy: within %.1f%%  (signed error %+.1f%%)\n", rec.Observation.AbsoluteErrorPct, rec.Observation.SignedErrorPct)
}

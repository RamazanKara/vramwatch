package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	fitengine "github.com/RamazanKara/vramwatch/internal/fit"
	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
)

func commandGGUF(t *testing.T) string {
	t.Helper()
	var body bytes.Buffer
	strKV := func(key, value string) {
		cmdString(&body, key)
		cmdU32(&body, 8)
		cmdString(&body, value)
	}
	uintKV := func(key string, value uint32) {
		cmdString(&body, key)
		cmdU32(&body, 4)
		cmdU32(&body, value)
	}
	strKV("general.architecture", "llama")
	uintKV("general.file_type", 15)
	uintKV("llama.block_count", 2)
	uintKV("llama.attention.head_count", 2)
	uintKV("llama.attention.head_count_kv", 1)
	uintKV("llama.attention.key_length", 8)
	uintKV("llama.attention.value_length", 8)
	uintKV("llama.context_length", 8192)

	var out bytes.Buffer
	out.WriteString("GGUF")
	cmdU32(&out, 3)
	cmdU64(&out, 0)
	cmdU64(&out, 8)
	out.Write(body.Bytes())
	path := filepath.Join(t.TempDir(), "launch-Q4_K_M.gguf")
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func cmdU32(w io.Writer, v uint32) { _ = binary.Write(w, binary.LittleEndian, v) }
func cmdU64(w io.Writer, v uint64) { _ = binary.Write(w, binary.LittleEndian, v) }
func cmdString(w io.Writer, s string) {
	cmdU64(w, uint64(len(s)))
	_, _ = io.WriteString(w, s)
}

func TestCmdFitAcceptsDocumentedModelFirstOrdering(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	path := commandGGUF(t)
	out, err := capture(t, func() error {
		return cmdFit([]string{path, "--quant", "q4_k_m", "--context", "4096", "--vram", "1GiB", "--json"})
	})
	if err != nil {
		t.Fatal(err)
	}
	var env fitEnvelope
	if err := json.Unmarshal([]byte(out), &env); err != nil {
		t.Fatalf("fit JSON: %v\n%s", err, out)
	}
	if env.Command != "fit" || env.RecordID == "" || env.Result.Artifact.Quantization != "Q4_K_M" {
		t.Fatalf("fit envelope = %+v", env)
	}
	if len(env.Result.Targets) != 1 || env.Result.Targets[0].FitsNow != fitengine.VerdictFits {
		t.Fatalf("fit verdict = %+v", env.Result.Targets)
	}
	if _, err := ledger.Load(env.RecordID); err != nil {
		t.Fatalf("prediction was not recorded: %v", err)
	}
}

func TestCmdFitPrintsResultAndReturnsScriptableNonFit(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	path := commandGGUF(t)
	out, err := capture(t, func() error {
		return cmdFit([]string{path, "--context=4096", "--vram=512MiB", "--json", "--no-record"})
	})
	var ee *exitError
	if err == nil || !strings.Contains(err.Error(), "does not fit") || !errors.As(err, &ee) || ee.code != 3 {
		t.Fatalf("non-fit error = %#v", err)
	}
	var env fitEnvelope
	if json.Unmarshal([]byte(out), &env) != nil || env.Result.Targets[0].FitsNow != fitengine.VerdictDoesNotFit {
		t.Fatalf("non-fit JSON missing verdict:\n%s", out)
	}
}

func TestFitExitClassificationPreservesIndeterminate(t *testing.T) {
	unknown := fitengine.Result{Targets: []fitengine.TargetResult{{
		FitsOnDevice: fitengine.VerdictUnknown,
		FitsNow:      fitengine.VerdictUnknown,
	}}}
	if fitSucceeded(unknown) || !fitIndeterminate(unknown) {
		t.Fatalf("unknown target was classified as success or definite non-fit: %+v", unknown.Targets)
	}
	knownNonFit := fitengine.Result{Targets: []fitengine.TargetResult{{
		FitsOnDevice: fitengine.VerdictDoesNotFit,
		FitsNow:      fitengine.VerdictDoesNotFit,
	}}}
	if fitSucceeded(knownNonFit) || fitIndeterminate(knownNonFit) {
		t.Fatalf("known non-fit was classified as indeterminate: %+v", knownNonFit.Targets)
	}
}

func TestReportCardIsPrivacySafeAndShowsAccuracy(t *testing.T) {
	rec := ledger.Record{
		ID: "0123456789abcdef", Loader: "llama.cpp",
		Prediction: fitengine.Result{
			Artifact: fitengine.Artifact{Source: fitengine.SourceLocal, CanonicalID: `/Users/alice/private/models/secret.gguf`, Filename: "secret.gguf", Quantization: "Q4_K_M"},
			Context:  32768, KVCacheType: "F16", ExpectedFootprintBytes: 7 * model.GiB,
			Targets: []fitengine.TargetResult{{Target: fitengine.Target{GPUName: "RTX 4090", CapacityBytes: 24 * model.GiB}, FitsOnDevice: fitengine.VerdictFits, FitsNow: fitengine.VerdictUnknown}},
		},
		Observation: &ledger.Observation{FootprintBytes: 7200 * model.MiB, Provenance: model.ProvenanceMeasured, AbsoluteErrorPct: 0.4, SignedErrorPct: -0.4},
	}
	card := cardFromReport(rec, nil, true)
	svg := render.ReportSVG(card)
	if card.Model != "secret.gguf" || strings.Contains(svg, "/Users/alice") {
		t.Fatalf("shareable report leaked local path: model=%q\n%s", card.Model, svg)
	}
	for _, want := range []string{"secret.gguf", "RTX 4090", "32,768", "within 0.4%", "FITS"} {
		if !strings.Contains(svg, want) {
			t.Errorf("report missing %q", want)
		}
	}
	if strings.Contains(svg, "0001-01-01") {
		t.Error("static report contains a timestamp")
	}

	rec.Prediction.Artifact.Source = fitengine.SourceURL
	rec.Prediction.Artifact.CanonicalID = "https://example.test/model.gguf?token=super-secret"
	rec.Prediction.Artifact.Filename = "model.gguf"
	if got := shareableModelName(rec); got != "model.gguf" || strings.Contains(got, "token") {
		t.Errorf("URL model name = %q", got)
	}
}

func TestReportCardUsesTheTargetThatProducedItsStatus(t *testing.T) {
	rec := ledger.Record{ID: "0123456789abcdef", Prediction: fitengine.Result{
		Artifact: fitengine.Artifact{CanonicalID: "model", Quantization: "Q4_K_M"},
		Context:  4096,
		Targets: []fitengine.TargetResult{
			{Target: fitengine.Target{GPUIndex: 0, GPUName: "small", CapacityBytes: 4 * model.GiB}, FitsOnDevice: fitengine.VerdictDoesNotFit, FitsNow: fitengine.VerdictDoesNotFit},
			{Target: fitengine.Target{GPUIndex: 1, GPUName: "large", CapacityBytes: 24 * model.GiB}, FitsOnDevice: fitengine.VerdictFits, FitsNow: fitengine.VerdictFits},
		},
	}}
	card := cardFromReport(rec, nil, true)
	if card.GPUName != "large" || card.Fits != "fits" {
		t.Fatalf("card mixed target and status: %+v", card)
	}
	console, err := capture(t, func() error {
		printReport(rec, nil, false)
		return nil
	})
	if err != nil || !strings.Contains(console, "hardware: large") || strings.Contains(console, "hardware: small") {
		t.Fatalf("console report selected the wrong target: err=%v\n%s", err, console)
	}

	rec.Prediction.Targets = []fitengine.TargetResult{{
		Target:       fitengine.Target{GPUIndex: 0, GPUName: "unknown"},
		FitsOnDevice: fitengine.VerdictUnknown,
		FitsNow:      fitengine.VerdictUnknown,
	}}
	card = cardFromReport(rec, nil, true)
	if card.Fits != "unknown" {
		t.Fatalf("unknown target rendered as %q", card.Fits)
	}
}

func TestWriteReportSVGDoesNotOverwriteWithoutForce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.svg")
	if err := writeReportSVG(path, "first", false); err != nil {
		t.Fatal(err)
	}
	if err := writeReportSVG(path, "second", false); err == nil {
		t.Error("second write without --force should fail")
	}
	b, _ := os.ReadFile(path)
	if string(b) != "first" {
		t.Fatalf("failed write changed existing report to %q", b)
	}
	if err := writeReportSVG(path, "second", true); err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(path)
	if string(b) != "second" {
		t.Fatalf("forced write = %q", b)
	}
}

func TestCmdReportSVGFromLedger(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")
	t.Setenv("LLAMACPP_HOST", "http://127.0.0.1:1")
	result := fitengine.Result{
		Policy:   fitengine.PolicyVersion,
		Artifact: fitengine.Artifact{Source: fitengine.SourceLocal, CanonicalID: `/private/alice/model-Q4_K_M.gguf`, Filename: "model-Q4_K_M.gguf", Quantization: "Q4_K_M"},
		Context:  32768, KVCacheType: "F16", ExpectedFootprintBytes: 7 * model.GiB,
		Targets: []fitengine.TargetResult{{Target: fitengine.Target{GPUName: "manual 24 GiB", MemoryKind: model.MemoryManual, CapacityBytes: 24 * model.GiB}, FitsOnDevice: fitengine.VerdictFits, FitsNow: fitengine.VerdictFits}},
	}
	rec, err := ledger.Save(result, "llama.cpp")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ledger.UpdateObservation(rec.ID, 7200*model.MiB, model.ProvenanceMeasured, "test fixture"); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "card.svg")
	out, err := capture(t, func() error {
		return cmdReport([]string{"--prediction", rec.ID, "--svg", "--static", "--output", path})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, path) {
		t.Errorf("report did not print output path: %q", out)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	svg := string(b)
	for _, want := range []string{"model-Q4_K_M.gguf", "manual 24 GiB", "32,768", "accuracy"} {
		if !strings.Contains(svg, want) {
			t.Errorf("report SVG missing %q", want)
		}
	}
	if strings.Contains(svg, "/private/alice") || strings.Contains(svg, "0001-01-01") {
		t.Error("report SVG leaked private path or zero timestamp")
	}
}

func TestParseByteSizeRejectsNonFiniteAndOverflow(t *testing.T) {
	for _, raw := range []string{"NaN", "+Inf", "-1GiB", "0", "nonsense", "18446744073709551615B"} {
		if _, err := parseByteSize(raw); err == nil {
			t.Errorf("parseByteSize(%q) should fail", raw)
		}
	}
	if got, err := parseByteSize("1.5GiB"); err != nil || got != 1536*model.MiB {
		t.Errorf("1.5GiB = %d, %v", got, err)
	}
	if _, err := parseByteSize(strings.Repeat("9", int(math.Log10(float64(^uint64(0))))+10) + "GiB"); err == nil {
		t.Error("overflowing size should fail")
	}
}

func TestDoctorAccelerationEvidence(t *testing.T) {
	if hasAccelerationEvidence(nil, []model.LoaderModel{{Name: "cpu-only"}}) {
		t.Error("model presence alone is not GPU evidence")
	}
	if !hasAccelerationEvidence(nil, []model.LoaderModel{{VRAMBytes: 1, VRAMSource: model.ProvenanceReported}}) {
		t.Error("loader-reported VRAM should be GPU evidence")
	}
	if hasAccelerationEvidence(nil, []model.LoaderModel{{VRAMBytes: 1, VRAMSource: model.ProvenanceEstimated}}) {
		t.Error("a model-derived footprint is not proof of GPU acceleration")
	}
	if !hasAccelerationEvidence([]model.GPU{{Procs: []model.Proc{{Name: "llama-server", UsedBytes: 1}}}}, []model.LoaderModel{{Name: "model", Loader: "llama.cpp"}}) {
		t.Error("driver process memory should be GPU evidence")
	}
	if hasAccelerationEvidence([]model.GPU{{Procs: []model.Proc{{Name: "browser", UsedBytes: 1}}}}, []model.LoaderModel{{Name: "model", Loader: "llama.cpp"}}) {
		t.Error("an unrelated GPU process is not loader acceleration evidence")
	}
}

func TestRecordMatchRejectsKnownDigestMismatch(t *testing.T) {
	rec := ledger.Record{Prediction: fitengine.Result{
		Artifact: fitengine.Artifact{Source: fitengine.SourceOllama, CanonicalID: "llama3.2:3b", Digest: "sha256:old", Quantization: "Q4_K_M"},
		Context:  4096,
	}}
	m := model.LoaderModel{Name: "llama3.2:3b", Digest: "sha256:new", Quantization: "Q4_K_M", ContextTokens: 4096}
	if recordMatchesModel(rec, m) {
		t.Error("same name with a different known digest must not match")
	}
	m.Digest = "sha256:old"
	if !recordMatchesModel(rec, m) {
		t.Error("matching identity, digest, quant, and context should match")
	}
	m.Digest = "old"
	if !recordMatchesModel(rec, m) {
		t.Error("sha256 prefix should not change digest identity")
	}
}

func TestLoaderVRAMProvenancePreservesEstimate(t *testing.T) {
	m := model.LoaderModel{VRAMBytes: 1, VRAMSource: model.ProvenanceEstimated}
	if got := loaderVRAMProvenance(m); got != model.ProvenanceEstimated {
		t.Fatalf("loader VRAM provenance = %q", got)
	}
}

func TestUsageContainsLaunchPositioning(t *testing.T) {
	want := "vramwatch — see why your local LLM ran out of GPU memory and determine what will fit before loading it."
	if !strings.HasPrefix(usage, want) {
		t.Fatalf("usage does not lead with positioning:\n%s", usage)
	}
}

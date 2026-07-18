package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/gpu"
	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/loader"
	"github.com/RamazanKara/vramwatch/internal/model"
)

type doctorCheck struct {
	ID          string `json:"id"`
	Layer       string `json:"layer"`
	Status      string `json:"status"`
	Summary     string `json:"summary"`
	Evidence    string `json:"evidence,omitempty"`
	Remediation string `json:"remediation,omitempty"`
}
type doctorEnvelope struct {
	SchemaVersion int           `json:"schema_version"`
	Command       string        `json:"command"`
	Checks        []doctorCheck `json:"checks"`
	Healthy       bool          `json:"healthy"`
}

func cmdDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "print stable machine-readable JSON")
	verbose := fs.Bool("verbose", false, "show provider error details")
	online := fs.Bool("online", false, "also test model metadata registries")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch doctor: diagnose driver, runtime, loader, and GPU detection\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return &usageError{fmt.Errorf("doctor takes no positional arguments")}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	checks := []doctorCheck{{ID: "system.platform", Layer: "system", Status: "pass", Summary: runtime.GOOS + "/" + runtime.GOARCH + " running " + runtime.Version()}}
	var detected int
	var sampledGPUs []model.GPU
	for _, p := range gpu.All() {
		id := "gpu." + p.Name()
		if !p.Available(ctx) {
			checks = append(checks, doctorCheck{ID: id, Layer: "driver", Status: "skip", Summary: p.Name() + " not detected", Remediation: gpuRemediation(p.Name())})
			continue
		}
		devices, err := p.Sample(ctx)
		if err != nil {
			ev := ""
			if *verbose {
				ev = err.Error()
			}
			checks = append(checks, doctorCheck{ID: id, Layer: "driver", Status: "fail", Summary: p.Name() + " is installed but its query failed", Evidence: ev, Remediation: gpuRemediation(p.Name())})
			continue
		}
		if len(devices) == 0 {
			checks = append(checks, doctorCheck{ID: id, Layer: "gpu", Status: "warn", Summary: p.Name() + " ran but returned no accelerator", Remediation: gpuRemediation(p.Name())})
			continue
		}
		detected += len(devices)
		sampledGPUs = append(sampledGPUs, devices...)
		status := "pass"
		summary := fmt.Sprintf("%s detected %d device(s)", p.Name(), len(devices))
		var unknownUsage int
		var unknownCapacity int
		for _, d := range devices {
			if d.UsageSource == model.ProvenanceAssumed {
				unknownUsage++
			}
			if d.TotalBytes == 0 && d.BudgetBytes == 0 {
				unknownCapacity++
			}
		}
		remediation := ""
		if unknownUsage > 0 {
			status = "warn"
			summary += fmt.Sprintf("; current free memory unavailable on %d", unknownUsage)
			remediation = gpuRemediation(p.Name())
		}
		if unknownCapacity > 0 {
			status = "warn"
			summary += fmt.Sprintf("; capacity unavailable on %d", unknownCapacity)
			remediation = gpuRemediation(p.Name())
		}
		checks = append(checks, doctorCheck{ID: id, Layer: "gpu", Status: status, Summary: summary, Evidence: devices[0].Name, Remediation: remediation})
	}
	if detected == 0 {
		checks = append(checks, doctorCheck{ID: "gpu.detected", Layer: "gpu", Status: "fail", Summary: "no supported accelerator was detected", Remediation: "Check the vendor driver and permissions, or use fit --vram for a manual target."})
	}
	loaderDetected := 0
	var residentModels []model.LoaderModel
	for _, p := range loader.All() {
		id := "loader." + p.Name()
		if !p.Available(ctx) {
			checks = append(checks, doctorCheck{ID: id, Layer: "loader", Status: "skip", Summary: p.Name() + " endpoint not reachable", Remediation: loaderRemediation(p.Name())})
			continue
		}
		loaderDetected++
		models, err := p.Models(ctx)
		if err != nil {
			ev := ""
			if *verbose {
				ev = err.Error()
			}
			checks = append(checks, doctorCheck{ID: id, Layer: "loader", Status: "fail", Summary: p.Name() + " answered its health probe but model inspection failed", Evidence: ev, Remediation: loaderRemediation(p.Name())})
			continue
		}
		residentModels = append(residentModels, models...)
		status := "pass"
		summary := fmt.Sprintf("%s endpoint healthy; %d resident model(s)", p.Name(), len(models))
		if len(models) == 0 {
			status = "warn"
			summary = p.Name() + " is healthy but no model is resident, so GPU acceleration cannot be confirmed"
		}
		checks = append(checks, doctorCheck{ID: id, Layer: "loader", Status: status, Summary: summary})
	}
	if loaderDetected == 0 {
		checks = append(checks, doctorCheck{ID: "loader.detected", Layer: "loader", Status: "fail", Summary: "neither Ollama nor llama.cpp is reachable", Remediation: "Start Ollama or llama-server, or set OLLAMA_HOST/LLAMACPP_HOST."})
	}
	if detected > 0 && loaderDetected > 0 {
		switch {
		case len(residentModels) == 0:
			checks = append(checks, doctorCheck{ID: "runtime.acceleration", Layer: "runtime", Status: "skip", Summary: "no model is resident; load one to verify GPU acceleration"})
		case hasAccelerationEvidence(sampledGPUs, residentModels):
			checks = append(checks, doctorCheck{ID: "runtime.acceleration", Layer: "runtime", Status: "pass", Summary: "resident model has GPU-memory evidence"})
		default:
			checks = append(checks, doctorCheck{ID: "runtime.acceleration", Layer: "runtime", Status: "warn", Summary: "a model is resident, but GPU acceleration could not be confirmed", Remediation: "Check loader GPU-offload settings and its CUDA, ROCm, Vulkan, or Metal backend logs."})
		}
	}
	state, err := ledger.StateDir()
	if err != nil {
		checks = append(checks, doctorCheck{ID: "state.ledger", Layer: "state", Status: "fail", Summary: "prediction ledger path cannot be resolved", Evidence: err.Error()})
	} else {
		records, listErr := ledger.List()
		if listErr != nil {
			checks = append(checks, doctorCheck{ID: "state.ledger", Layer: "state", Status: "fail", Summary: "prediction ledger cannot be read", Evidence: shortErr(listErr, *verbose), Remediation: "Check the state-directory permissions or set VRAMWATCH_STATE_DIR to a writable private directory."})
		} else {
			checks = append(checks, doctorCheck{ID: "state.ledger", Layer: "state", Status: "pass", Summary: fmt.Sprintf("prediction ledger ready; %d saved prediction(s)", len(records)), Evidence: state})
		}
	}
	if *online {
		for _, endpoint := range []struct{ id, url string }{{"registry.ollama", "https://registry.ollama.ai/v2/"}, {"registry.huggingface", "https://huggingface.co/api/models?limit=1"}} {
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.url, nil)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				checks = append(checks, doctorCheck{ID: endpoint.id, Layer: "network", Status: "fail", Summary: "metadata registry is unreachable", Evidence: shortErr(err, *verbose)})
				continue
			}
			resp.Body.Close()
			status := "pass"
			if resp.StatusCode/100 != 2 {
				status = "fail"
			}
			checks = append(checks, doctorCheck{ID: endpoint.id, Layer: "network", Status: status, Summary: fmt.Sprintf("metadata registry returned HTTP %d", resp.StatusCode)})
		}
	}
	healthy := true
	for _, c := range checks {
		if c.Status == "fail" {
			healthy = false
		}
	}
	if *asJSON {
		b, err := json.MarshalIndent(doctorEnvelope{1, "doctor", checks, healthy}, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
	} else {
		printDoctor(checks, cf.resolve())
	}
	if !healthy {
		return &exitError{code: 1, quiet: true, err: fmt.Errorf("doctor found required failures")}
	}
	return nil
}

func hasAccelerationEvidence(gpus []model.GPU, models []model.LoaderModel) bool {
	for _, m := range models {
		p := loaderVRAMProvenance(m)
		if m.VRAMBytes > 0 && (p == model.ProvenanceMeasured || p == model.ProvenanceReported) {
			return true
		}
	}
	for _, g := range gpus {
		for _, p := range g.Procs {
			if p.UsedBytes > 0 && processMatchesLoader(p.Name, models) {
				return true
			}
		}
	}
	return false
}

func processMatchesLoader(raw string, models []model.LoaderModel) bool {
	name := strings.ToLower(strings.TrimSuffix(filepath.Base(strings.ReplaceAll(raw, `\`, "/")), ".exe"))
	for _, m := range models {
		switch strings.ToLower(m.Loader) {
		case "ollama":
			if strings.HasPrefix(name, "ollama") {
				return true
			}
		case "llama.cpp":
			if strings.HasPrefix(name, "llama-") || name == "llama.cpp" {
				return true
			}
		}
	}
	return false
}

func printDoctor(checks []doctorCheck, color bool) {
	fmt.Println(bold(color, "vramwatch doctor"))
	for _, c := range checks {
		mark := c.Status
		switch c.Status {
		case "pass":
			mark = greenc(color, "PASS")
		case "fail":
			mark = redc(color, "FAIL")
		case "warn":
			mark = csi(color, "38;5;214", "WARN")
		default:
			mark = dimc(color, "SKIP")
		}
		fmt.Printf("  %-4s  %-8s  %s\n", mark, c.Layer, c.Summary)
		if c.Evidence != "" {
			fmt.Printf("        %s\n", dimc(color, c.Evidence))
		}
		if c.Remediation != "" && (c.Status == "fail" || c.Status == "warn") {
			fmt.Printf("        fix: %s\n", c.Remediation)
		}
	}
}
func gpuRemediation(name string) string {
	switch name {
	case "nvidia-smi":
		return "Install/repair the NVIDIA driver and ensure nvidia-smi is on PATH."
	case "amd-smi":
		return "Install AMD SMI/ROCm and verify access to /dev/dri and /dev/kfd."
	case "apple-metal":
		return "Run on Apple silicon with Metal support enabled."
	default:
		return "Verify the OS GPU driver and device permissions."
	}
}
func loaderRemediation(name string) string {
	if name == "ollama" {
		return "Start `ollama serve` or set OLLAMA_HOST to the active endpoint."
	}
	return "Start llama-server or set LLAMACPP_HOST to its endpoint."
}
func shortErr(err error, verbose bool) string {
	if verbose {
		return err.Error()
	}
	return "run with --verbose for the provider error"
}

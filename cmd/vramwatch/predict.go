package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
)

type targetFit struct {
	Context        int    `json:"context"`
	Fits           bool   `json:"fits"`
	NeededBytes    uint64 `json:"needed_bytes"`
	TotalBytes     uint64 `json:"total_bytes"`
	ExceedsTrained bool   `json:"exceeds_trained_context,omitempty"`
	TrainedContext int    `json:"trained_context,omitempty"`
}

type predictResult struct {
	GPU             int        `json:"gpu"`
	GPUName         string     `json:"gpu_name"`
	Model           string     `json:"model,omitempty"`
	KVBytesPerToken uint64     `json:"kv_bytes_per_token,omitempty"`
	HeadroomBytes   uint64     `json:"headroom_bytes"`
	MaxContextFits  int        `json:"max_context_fits,omitempty"`
	OOMRisk         bool       `json:"oom_risk"`
	Target          *targetFit `json:"target,omitempty"`
	Note            string     `json:"note,omitempty"`
}

func cmdPredict(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	src := fs.String("source", "", "data source: live | demo | mock:PATH | PATH.json")
	target := fs.Int("context", 0, "target context length in tokens to test for fit (0 to skip)")
	asJSON := fs.Bool("json", false, "print machine-readable JSON")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch predict — will a context fit, and what's the max before OOM?\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	snap, _, err := collect(context.Background(), *src)
	if err != nil {
		return err
	}

	var results []predictResult
	for _, bd := range snap.Breakdowns {
		r := predictResult{GPU: bd.GPU.Index, GPUName: bd.GPU.Name}
		if p := bd.Prediction; p != nil {
			r.Model = p.Model
			r.KVBytesPerToken = p.KVBytesPerToken
			r.HeadroomBytes = p.HeadroomBytes
			r.MaxContextFits = p.MaxContextFits
			r.OOMRisk = p.OOMRisk
		} else {
			r.HeadroomBytes = bd.GPU.FreeBytes
			r.Note = "no model with a known architecture is loaded on this GPU"
		}
		if *target > 0 {
			if fit, ok := engine.WillContextFit(bd.GPU, bd.Models, *target); ok {
				r.Target = &targetFit{
					Context: *target, Fits: fit.Fits, NeededBytes: fit.NeededBytes, TotalBytes: fit.TotalBytes,
					ExceedsTrained: fit.ExceedsTrained, TrainedContext: fit.ModelContextMax,
				}
			}
		}
		results = append(results, r)
	}

	if *asJSON {
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			return err
		}
		os.Stdout.Write(data)
		os.Stdout.Write([]byte("\n"))
		return nil
	}
	printPredict(results, cf.resolve())
	return nil
}

func printPredict(results []predictResult, color bool) {
	if len(results) == 0 {
		fmt.Println("no GPUs detected")
		return
	}
	for _, r := range results {
		fmt.Printf("%s  %s\n", bold(color, fmt.Sprintf("GPU %d", r.GPU)), r.GPUName)
		if r.Note != "" {
			fmt.Printf("  %s\n", dimc(color, r.Note))
			continue
		}
		fmt.Printf("  model: %s   ~%s/token\n", r.Model, model.HumanBytes(r.KVBytesPerToken))
		fmt.Printf("  headroom: %s\n", model.HumanBytes(r.HeadroomBytes))
		maxLine := fmt.Sprintf("  max context that fits: ~%s tokens", commas(r.MaxContextFits))
		if r.OOMRisk {
			maxLine += "   " + redc(color, "(OOM risk now)")
		}
		fmt.Println(maxLine)
		if r.Target != nil {
			t := r.Target
			if t.Fits {
				line := fmt.Sprintf("  target %s tokens: %s (needs %s of %s)",
					commas(t.Context), greenc(color, "FITS"), model.HumanBytes(t.NeededBytes), model.HumanBytes(t.TotalBytes))
				if t.ExceedsTrained {
					line += "  " + dimc(color, fmt.Sprintf("but exceeds trained context %s", commas(t.TrainedContext)))
				}
				fmt.Println(line)
			} else {
				fmt.Printf("  target %s tokens: %s (needs %s, card has %s)\n",
					commas(t.Context), redc(color, "WON'T FIT"), model.HumanBytes(t.NeededBytes), model.HumanBytes(t.TotalBytes))
			}
		}
	}
}

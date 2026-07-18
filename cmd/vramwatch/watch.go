package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"time"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/ledger"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
	"github.com/RamazanKara/vramwatch/internal/source"
)

func cmdWatch(args []string) error {
	fs := flag.NewFlagSet("watch", flag.ContinueOnError)
	src := fs.String("source", "", "data source: live | demo | mock:PATH | PATH.json")
	interval := fs.Duration("interval", time.Second, "refresh interval")
	barWidth := fs.Int("bar-width", 48, "width of the stacked bar in cells")
	once := fs.Bool("once", false, "render a single frame and exit (useful for captures/CI)")
	kvType := addKVFlag(fs)
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch watch: live stacked VRAM bar\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	kvBits, err := resolveKVBits(*kvType)
	if err != nil {
		return err
	}
	if *interval < 100*time.Millisecond {
		*interval = 100 * time.Millisecond
	}

	s, err := source.FromSpec(*src)
	if err != nil {
		return err
	}
	color := cf.resolve()
	opts := render.Options{Color: color, BarWidth: *barWidth}

	if *once {
		return renderFrame(context.Background(), s, opts, kvBits, false, "")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Hide cursor; restore on exit.
	fmt.Print("\x1b[?25l")
	defer fmt.Print("\x1b[?25h\n")

	footer := dimc(color, "updating every "+interval.String()+" • press Ctrl-C to quit")
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	if err := renderFrame(ctx, s, opts, kvBits, true, footer); err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := renderFrame(ctx, s, opts, kvBits, true, footer); err != nil {
				return err
			}
		}
	}
}

// renderFrame collects one snapshot and paints it. When clear is true it first
// clears the screen (the live loop); when false it prints in place (--once).
func renderFrame(ctx context.Context, s source.Source, opts render.Options, kvBits int, clear bool, footer string) error {
	gpus, models, err := s.Collect(ctx)
	if err != nil {
		return err
	}
	snap := engine.Build(gpus, models, engine.Options{Version: Version, Now: time.Now(), KVBits: kvBits})
	var out string
	if clear {
		out = "\x1b[H\x1b[2J" // home + clear
	}
	out += render.Table(snap, opts)
	if comparison := trackWatchPrediction(snap); comparison != "" {
		out += comparison + "\n"
	}
	if footer != "" {
		out += "\n" + footer + "\n"
	}
	os.Stdout.WriteString(out)
	return nil
}

type watchTrackState struct {
	id       string
	last     uint64
	stable   int
	recorded bool
}

var watchTrack watchTrackState

func trackWatchPrediction(snap model.Snapshot) string {
	records, err := ledger.List()
	if err != nil {
		return ""
	}
	for _, bd := range snap.Breakdowns {
		if len(bd.Models) != 1 {
			continue
		}
		for _, rec := range records {
			if !recordMatchesModel(rec, bd.Models[0]) {
				continue
			}
			return trackWatchBreakdown(rec, bd)
		}
	}
	return ""
}

func trackWatchBreakdown(rec ledger.Record, bd model.Breakdown) string {
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
	} else if bd.Models[0].VRAMBytes > 0 {
		fp = bd.Models[0].VRAMBytes
		prov = loaderVRAMProvenance(bd.Models[0])
		source = bd.Models[0].Loader + " model footprint"
	}
	if fp == 0 {
		return ""
	}
	if watchTrack.id != rec.ID {
		watchTrack = watchTrackState{id: rec.ID, last: fp, stable: 1}
	} else {
		delta := math.Abs(float64(fp)-float64(watchTrack.last)) / float64(fp)
		if delta <= 0.02 {
			watchTrack.stable++
		} else {
			watchTrack.stable = 1
			watchTrack.recorded = false
		}
		watchTrack.last = fp
	}
	if watchTrack.stable >= 3 && !watchTrack.recorded {
		if _, err := ledger.UpdateObservation(rec.ID, fp, prov, source); err == nil {
			watchTrack.recorded = true
		}
	}
	signed := 100 * (float64(rec.Prediction.ExpectedFootprintBytes) - float64(fp)) / float64(fp)
	return fmt.Sprintf("prediction %s  [E] %s expected · [%s] %s observed · error %+.1f%%", rec.ID, model.HumanBytes(rec.Prediction.ExpectedFootprintBytes), watchProvBadge(prov), model.HumanBytes(fp), signed)
}

func watchProvBadge(p model.Provenance) string {
	if p == model.ProvenanceMeasured {
		return "M"
	}
	if p == model.ProvenanceReported {
		return "R"
	}
	return "E"
}

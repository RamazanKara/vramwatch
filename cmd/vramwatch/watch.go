package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/RamazanKara/vramwatch/internal/engine"
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
		fmt.Fprintln(os.Stderr, "vramwatch watch — live stacked VRAM bar\n\nFLAGS")
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
	if footer != "" {
		out += "\n" + footer + "\n"
	}
	os.Stdout.WriteString(out)
	return nil
}

package main

import (
	"context"
	"flag"
	"os"
	"strconv"
	"time"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
	"github.com/RamazanKara/vramwatch/internal/source"
)

// colorFlags holds the shared colour-control flags for a subcommand.
type colorFlags struct {
	color   *bool
	noColor *bool
}

func addColorFlags(fs *flag.FlagSet) colorFlags {
	return colorFlags{
		color:   fs.Bool("color", false, "force ANSI colour even when stdout is not a TTY"),
		noColor: fs.Bool("no-color", false, "disable ANSI colour"),
	}
}

// resolve decides whether colour should be emitted.
func (c colorFlags) resolve() bool {
	if *c.noColor || os.Getenv("NO_COLOR") != "" {
		return false
	}
	if *c.color {
		return true
	}
	return isTerminal(os.Stdout)
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// collect resolves the source spec and builds a snapshot from it.
func collect(ctx context.Context, spec string) (model.Snapshot, source.Source, error) {
	src, err := source.FromSpec(spec)
	if err != nil {
		return model.Snapshot{}, nil, err
	}
	gpus, models, err := src.Collect(ctx)
	if err != nil {
		return model.Snapshot{}, src, err
	}
	snap := engine.Build(gpus, models, engine.Options{Version: Version, Now: time.Now()})
	return snap, src, nil
}

// renderConsole prints the standard console table.
func renderConsole(snap model.Snapshot, color bool) {
	os.Stdout.WriteString(render.Table(snap, render.Options{Color: color}))
}

// Small colour + formatting helpers for the ad-hoc command output (the table
// renderer has its own; these are for predict/devices).
func csi(color bool, params, s string) string {
	if !color || params == "" {
		return s
	}
	return "\x1b[" + params + "m" + s + "\x1b[0m"
}

func bold(color bool, s string) string   { return csi(color, "1", s) }
func dimc(color bool, s string) string   { return csi(color, "2", s) }
func redc(color bool, s string) string   { return csi(color, "38;5;203", s) }
func greenc(color bool, s string) string { return csi(color, "38;5;35", s) }

// commas formats a non-negative int with thousands separators.
func commas(n int) string {
	s := strconv.Itoa(n)
	neg := false
	if n < 0 {
		neg, s = true, s[1:]
	}
	var b []byte
	for i := 0; i < len(s); i++ {
		if i > 0 && (len(s)-i)%3 == 0 {
			b = append(b, ',')
		}
		b = append(b, s[i])
	}
	if neg {
		return "-" + string(b)
	}
	return string(b)
}

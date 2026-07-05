package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
	"github.com/RamazanKara/vramwatch/internal/source"
)

// usageError marks a command-line usage problem (bad flag) so main can exit
// with the conventional code 2 rather than 1.
type usageError struct{ err error }

func (e *usageError) Error() string { return e.err.Error() }
func (e *usageError) Unwrap() error { return e.err }

// parseFlags parses a subcommand's flags, mapping flag package outcomes onto
// our error convention: flag.ErrHelp propagates (main exits 0), any other parse
// failure becomes a usageError (main exits 2). flag has already printed the
// message/usage in both cases.
func parseFlags(fs *flag.FlagSet, args []string) error {
	err := fs.Parse(args)
	switch {
	case err == nil:
		return nil
	case errors.Is(err, flag.ErrHelp):
		return err
	default:
		return &usageError{err}
	}
}

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

// collect resolves the source spec and builds a snapshot from it. kvBits, when
// non-zero, overrides the KV cache element size (from --kv-cache-type).
func collect(ctx context.Context, spec string, kvBits int) (model.Snapshot, source.Source, error) {
	src, err := source.FromSpec(spec)
	if err != nil {
		return model.Snapshot{}, nil, err
	}
	gpus, models, err := src.Collect(ctx)
	if err != nil {
		return model.Snapshot{}, src, err
	}
	snap := engine.Build(gpus, models, engine.Options{Version: Version, Now: time.Now(), KVBits: kvBits})
	return snap, src, nil
}

// addKVFlag registers the shared --kv-cache-type flag.
func addKVFlag(fs *flag.FlagSet) *string {
	return fs.String("kv-cache-type", "",
		"KV cache dtype for an accurate estimate: f16|bf16|f32|q8_0|q5_0|q4_0 (or $VRAMWATCH_KV_CACHE_TYPE)")
}

// resolveKVBits maps a --kv-cache-type value (falling back to the env var) to a
// bit width; 0 means "leave each model's own dtype".
func resolveKVBits(flagVal string) (int, error) {
	if flagVal == "" {
		flagVal = os.Getenv("VRAMWATCH_KV_CACHE_TYPE")
	}
	switch strings.ToLower(strings.TrimSpace(flagVal)) {
	case "":
		return 0, nil
	case "f32", "fp32", "float32":
		return 32, nil
	case "f16", "fp16", "float16", "bf16", "bfloat16":
		return 16, nil
	case "q8_0", "q8":
		return 8, nil
	case "q5_0", "q5_1", "q5":
		return 5, nil
	case "q4_0", "q4_1", "q4", "iq4_nl":
		return 4, nil
	default:
		return 0, fmt.Errorf("unknown --kv-cache-type %q (try f16, q8_0, q5_0, q4_0, f32)", flagVal)
	}
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

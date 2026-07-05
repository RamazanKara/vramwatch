package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/RamazanKara/vramwatch/internal/render"
)

func cmdSnapshot(args []string) error {
	fs := flag.NewFlagSet("snapshot", flag.ContinueOnError)
	src := fs.String("source", "", "data source: live | demo | mock:PATH | PATH.json")
	asJSON := fs.Bool("json", false, "print machine-readable JSON")
	svgOut := fs.String("svg", "", "write an SVG scorecard to this path ('-' for stdout)")
	static := fs.Bool("static", false, "omit host/timestamp for reproducible output (docs, tests)")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch snapshot — one-shot VRAM breakdown\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}

	snap, _, err := collect(context.Background(), *src)
	if err != nil {
		return err
	}
	if *static {
		snap.Host = ""
		snap.Timestamp = time.Time{}
	}

	wrote := false
	if *asJSON {
		data, err := render.JSON(snap)
		if err != nil {
			return err
		}
		os.Stdout.Write(data)
		os.Stdout.Write([]byte("\n"))
		wrote = true
	}
	if *svgOut != "" {
		svg := render.SVG(snap)
		if *svgOut == "-" {
			os.Stdout.WriteString(svg + "\n")
		} else {
			if err := os.WriteFile(*svgOut, []byte(svg), 0o644); err != nil {
				return err
			}
			fmt.Fprintln(os.Stderr, "wrote", *svgOut)
		}
		wrote = true
	}
	if !wrote {
		renderConsole(snap, cf.resolve())
	}
	return nil
}

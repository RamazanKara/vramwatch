package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/RamazanKara/vramwatch/internal/gpu"
	"github.com/RamazanKara/vramwatch/internal/loader"
	"github.com/RamazanKara/vramwatch/internal/model"
)

func cmdDevices(args []string) error {
	fs := flag.NewFlagSet("devices", flag.ContinueOnError)
	src := fs.String("source", "", "data source to enumerate: live | demo | mock:PATH | PATH.json")
	cf := addColorFlags(fs)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "vramwatch devices: detected GPUs and inference loaders\n\nFLAGS")
		fs.PrintDefaults()
	}
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	color := cf.resolve()
	ctx := context.Background()

	// Live provider availability (always shown; a diagnostic).
	fmt.Println(bold(color, "GPU providers"))
	for _, p := range gpu.All() {
		fmt.Printf("  %-12s %s\n", p.Name(), availability(color, p.Available(ctx)))
	}
	fmt.Println(bold(color, "Loader providers"))
	for _, p := range loader.All() {
		fmt.Printf("  %-12s %s\n", p.Name(), availability(color, p.Available(ctx)))
	}

	// Devices from the chosen source.
	snap, s, err := collect(ctx, *src, 0)
	if err != nil {
		return err
	}
	fmt.Printf("\n%s %s\n", bold(color, "Devices"), dimc(color, "("+s.Describe()+")"))
	if len(snap.Breakdowns) == 0 {
		fmt.Println("  " + dimc(color, "none detected"))
		return nil
	}
	for _, bd := range snap.Breakdowns {
		g := bd.GPU
		fmt.Printf("  GPU %d  %s  [%s]  %s / %s\n", g.Index, g.Name, g.Vendor,
			model.HumanBytes(g.UsedBytes), model.HumanBytes(g.TotalBytes))
		for _, m := range bd.Models {
			fmt.Printf("           model: %s (%s)\n", m.Name, m.Loader)
		}
	}
	return nil
}

func availability(color bool, ok bool) string {
	if ok {
		return greenc(color, "available")
	}
	return dimc(color, "not detected")
}

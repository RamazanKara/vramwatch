// Command vramwatch explains local-LLM accelerator memory and predicts what
// fits before a model is downloaded or loaded.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
)

// Version is set at build time via -ldflags "-X main.Version=...".
var Version = "dev"

const usage = `vramwatch — see why your local LLM ran out of GPU memory and determine what will fit before loading it.

USAGE
  vramwatch <command> [flags]

COMMANDS
  fit        Predict whether a model + quant + context fits before downloading it
  watch      Live VRAM attribution with explicit measured/estimated provenance
  doctor     Diagnose drivers, runtimes, loaders, permissions, and GPU detection
  report     One-shot console/JSON report or privacy-safe SVG accuracy card
  version    Print version information
  help       Show this help

COMMON FLAGS
  --no-color            Disable ANSI colour   (also honours NO_COLOR)
  --color               Force ANSI colour even when not a TTY

EXAMPLES
  vramwatch fit ollama:llama3.2:3b-instruct --quant q4_k_m --context 32768
  vramwatch fit hf:bartowski/Llama-3.2-1B-Instruct-GGUF --quant q4_k_m --context 32768
  vramwatch watch
  vramwatch doctor
  vramwatch report --svg

Docs:        https://github.com/RamazanKara/vramwatch
Methodology: https://github.com/RamazanKara/vramwatch/blob/master/docs/METHODOLOGY.md
`

func main() {
	// Enable ANSI on legacy Windows consoles for every command that may print
	// colour (no-op elsewhere and on modern terminals).
	enableVT()

	if len(os.Args) < 2 {
		fmt.Print(usage)
		return
	}
	cmd, args := os.Args[1], os.Args[2:]

	var err error
	switch cmd {
	case "fit":
		err = cmdFit(args)
	case "watch":
		err = cmdWatch(args)
	case "doctor":
		err = cmdDoctor(args)
	case "report":
		err = cmdReport(args)
	case "predict":
		err = &usageError{fmt.Errorf("command predict was removed; use `vramwatch fit MODEL --context N`")}
	case "snapshot", "snap":
		err = &usageError{fmt.Errorf("command snapshot was removed; use `vramwatch report`")}
	case "devices", "detect":
		err = &usageError{fmt.Errorf("command devices was removed; use `vramwatch doctor`")}
	case "version", "--version", "-v":
		printVersion()
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "vramwatch: unknown command %q\n\n", cmd)
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	if err != nil {
		// A help request is not an error; flag already printed the usage.
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		var ee *exitError
		if errors.As(err, &ee) {
			if !ee.quiet && ee.err != nil {
				fmt.Fprintln(os.Stderr, "vramwatch:", ee.err)
			}
			os.Exit(ee.code)
		}
		// Usage errors (bad flags) exit 2; flag already printed the message.
		var ue *usageError
		if errors.As(err, &ue) {
			fmt.Fprintln(os.Stderr, "vramwatch:", ue)
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "vramwatch:", err)
		os.Exit(1)
	}
}

func printVersion() {
	fmt.Printf("vramwatch %s (%s %s/%s)\n", Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

// Command vramwatch live-traces where every megabyte of your local-LLM VRAM
// went (weights vs KV cache vs other apps) and predicts how much context
// fits before you run out of memory.
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

const usage = `vramwatch: the flame graph for "why won't this model fit"

USAGE
  vramwatch <command> [flags]

COMMANDS
  watch      Live TUI: a stacked VRAM bar that updates as the KV cache grows
  snapshot   One-shot breakdown (console, --json, or --svg scorecard)
  predict    Will a target context fit? What's the max context before OOM?
  devices    List detected GPUs and inference loaders (diagnostics)
  version    Print version information
  help       Show this help

COMMON FLAGS
  --source <spec>       Data source: "live" (default), "demo", "mock:PATH", or a .json path
  --kv-cache-type <t>   KV cache dtype for an accurate estimate: f16|bf16|f32|q8_0|q5_0|q4_0
                        (or set $VRAMWATCH_KV_CACHE_TYPE)
  --no-color            Disable ANSI colour   (also honours NO_COLOR)
  --color               Force ANSI colour even when not a TTY

EXAMPLES
  vramwatch watch
  vramwatch watch --kv-cache-type q8_0
  vramwatch snapshot --json
  vramwatch snapshot --svg card.svg
  vramwatch predict --context 32768
  vramwatch snapshot --source mock:testdata/scenarios/24gb-70b-oom.json

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
	case "watch":
		err = cmdWatch(args)
	case "snapshot", "report", "snap":
		err = cmdSnapshot(args)
	case "predict":
		err = cmdPredict(args)
	case "devices", "detect":
		err = cmdDevices(args)
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
		// Usage errors (bad flags) exit 2; flag already printed the message.
		var ue *usageError
		if errors.As(err, &ue) {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "vramwatch:", err)
		os.Exit(1)
	}
}

func printVersion() {
	fmt.Printf("vramwatch %s (%s %s/%s)\n", Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

// Command reportfixture renders the deterministic SVG committed in docs/sample.
package main

import (
	"fmt"

	"github.com/RamazanKara/vramwatch/internal/model"
	"github.com/RamazanKara/vramwatch/internal/render"
)

func main() {
	fmt.Println(render.ReportSVG(render.ReportCard{
		Version:               "v1.0.0-beta.1",
		PredictionID:          "8f31a42dc0e719b6",
		GPUName:               "NVIDIA GeForce RTX 4090",
		MemoryKind:            model.MemoryDedicated,
		CapacityBytes:         24 * model.GiB,
		Driver:                "590.48",
		Loader:                "ollama",
		Model:                 "llama3.1:8b-instruct-q4_K_M",
		Quant:                 "Q4_K_M",
		Context:               32768,
		KVType:                "F16",
		PredictedBytes:        8460 * model.MiB,
		ObservedBytes:         8652 * model.MiB,
		ObservationProvenance: model.ProvenanceMeasured,
		AbsoluteErrorPct:      2.2,
		SignedErrorPct:        -2.2,
		Fits:                  "fits",
	}))
}

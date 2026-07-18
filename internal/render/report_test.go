package render

import (
	"strings"
	"testing"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestReportSVGContainsRequiredFactsAndAccuracy(t *testing.T) {
	c := ReportCard{
		Version: "v1.0.0", GeneratedAt: time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC), PredictionID: "0123456789abcdef",
		GPUName: "GeForce RTX 4090", MemoryKind: model.MemoryDedicated, CapacityBytes: 24 * model.GiB, Driver: "999.1",
		Loader: "ollama", Model: "llama3.1:8b", Quant: "Q4_K_M", Context: 32768, KVType: "F16",
		PredictedBytes: 7 * model.GiB, ObservedBytes: 7200 * model.MiB, ObservationProvenance: model.ProvenanceMeasured,
		AbsoluteErrorPct: 0.4, SignedErrorPct: -0.4, Fits: "fits",
	}
	out := ReportSVG(c)
	for _, want := range []string{"GeForce RTX 4090", "dedicated VRAM", "llama3.1:8b", "32,768", "Q4_K_M", "within 0.4%", "[M]", "FITS", "2026-07-18"} {
		if !strings.Contains(out, want) {
			t.Errorf("report SVG missing %q\n%s", want, out)
		}
	}
	if !strings.HasPrefix(out, "<svg") || !strings.HasSuffix(out, "</svg>") {
		t.Error("report SVG is not wrapped")
	}
}

func TestReportSVGStaticOmitsTimestampAndEscapesText(t *testing.T) {
	out := ReportSVG(ReportCard{GPUName: `GPU <unsafe>`, Model: `model & friends`, Fits: "unknown"})
	if strings.Contains(out, "0001-01-01") {
		t.Error("static report leaked the zero timestamp")
	}
	if strings.Contains(out, "<unsafe>") || strings.Contains(out, "model & friends") {
		t.Error("report text was not XML escaped")
	}
	if !strings.Contains(out, "&lt;unsafe&gt;") || !strings.Contains(out, "model &amp; friends") {
		t.Error("escaped report values are missing")
	}
}

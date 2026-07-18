package render

import (
	"fmt"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// ReportCard is deliberately privacy-safe: it contains no host, path, PID,
// bus address, token, or serial-number field.
type ReportCard struct {
	Version               string
	GeneratedAt           time.Time
	PredictionID          string
	GPUName               string
	MemoryKind            model.MemoryKind
	CapacityBytes         uint64
	Driver                string
	Loader                string
	Model                 string
	Quant                 string
	Context               int
	KVType                string
	PredictedBytes        uint64
	ObservedBytes         uint64
	ObservationProvenance model.Provenance
	AbsoluteErrorPct      float64
	SignedErrorPct        float64
	Fits                  string
}

func ReportSVG(c ReportCard) string {
	const h = 430
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="%s">`, svgWidth, h, svgWidth, h, svgFont)
	fmt.Fprintf(&b, `<rect width="%d" height="%d" rx="14" fill="%s"/>`, svgWidth, h, svgBG)
	fmt.Fprintf(&b, `<text x="24" y="44" font-size="25" font-weight="700" fill="%s">vramwatch report</text>`, svgFG)
	if !c.GeneratedAt.IsZero() {
		fmt.Fprintf(&b, `<text x="736" y="43" text-anchor="end" font-size="13" fill="%s">%s</text>`, svgMuted, escapeXML(c.GeneratedAt.Format(time.RFC3339)))
	}
	fmt.Fprintf(&b, `<line x1="24" y1="62" x2="736" y2="62" stroke="%s"/>`, svgTrack)
	statusColor := "#8b949e"
	if strings.EqualFold(c.Fits, "fits") {
		statusColor = "#3fb950"
	} else if strings.Contains(strings.ToLower(c.Fits), "not") || strings.Contains(strings.ToLower(c.Fits), "unsupported") {
		statusColor = "#f85149"
	}
	status := strings.ToUpper(c.Fits)
	pillWidth := 126
	if need := len([]rune(status))*9 + 30; need > pillWidth {
		pillWidth = need
	}
	fmt.Fprintf(&b, `<rect x="24" y="82" width="%d" height="34" rx="17" fill="%s"/><text x="%d" y="105" text-anchor="middle" font-size="15" font-weight="700" fill="#0d1117">%s</text>`, pillWidth, statusColor, 24+pillWidth/2, escapeXML(status))
	row := func(y int, label, value string) {
		fmt.Fprintf(&b, `<text x="24" y="%d" font-size="12" fill="%s">%s</text><text x="170" y="%d" font-size="14" fill="%s">%s</text>`, y, svgMuted, escapeXML(label), y, svgFG, escapeXML(value))
	}
	row(150, "hardware", fmt.Sprintf("%s · %s · %s", c.GPUName, memoryKindName(c.MemoryKind), model.HumanBytes(c.CapacityBytes)))
	row(180, "driver / loader", strings.Trim(strings.Join([]string{c.Driver, c.Loader}, " · "), " ·"))
	row(210, "model", fmt.Sprintf("%s · %s", c.Model, c.Quant))
	row(240, "context", fmt.Sprintf("%s tokens · KV %s", formatInt(c.Context), c.KVType))
	fmt.Fprintf(&b, `<rect x="24" y="268" width="712" height="1" fill="%s"/>`, svgTrack)
	row(300, "predicted", model.HumanBytes(c.PredictedBytes)+" [E]")
	if c.ObservedBytes > 0 {
		row(330, "observed", fmt.Sprintf("%s [%s]", model.HumanBytes(c.ObservedBytes), provenanceShort(c.ObservationProvenance)))
		row(360, "accuracy", fmt.Sprintf("within %.1f%% · signed error %+.1f%%", c.AbsoluteErrorPct, c.SignedErrorPct))
	} else {
		row(330, "observed", "pending — load the model and run watch/report again")
		row(360, "accuracy", "pending")
	}
	fmt.Fprintf(&b, `<text x="24" y="400" font-size="11" fill="%s">[M] measured · [R] loader-reported · [E] model-estimated · prediction %s · vramwatch %s</text>`, svgMuted, escapeXML(c.PredictionID), escapeXML(c.Version))
	b.WriteString(`</svg>`)
	return b.String()
}

func memoryKindName(k model.MemoryKind) string {
	switch k {
	case model.MemoryUnified:
		return "unified memory"
	case model.MemoryManual:
		return "manual budget"
	case model.MemoryDedicated:
		return "dedicated VRAM"
	default:
		return "accelerator memory"
	}
}

func provenanceShort(p model.Provenance) string {
	switch p {
	case model.ProvenanceMeasured:
		return "M"
	case model.ProvenanceReported:
		return "R"
	default:
		return "E"
	}
}

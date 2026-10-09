package render

import (
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func ReportMarkdown(c ReportCard) string {
	var b strings.Builder
	b.WriteString("# vramwatch report\n\n| Field | Value |\n| --- | --- |\n")
	row := func(label, value string) {
		fmt.Fprintf(&b, "| %s | %s |\n", label, markdownCell(value))
	}
	row("Hardware", c.GPUName)
	row("Memory", memoryKindName(c.MemoryKind)+" · "+model.HumanBytes(c.CapacityBytes))
	row("Driver / loader", strings.Trim(strings.Join([]string{c.Driver, c.Loader}, " · "), " ·"))
	row("Model", c.Model)
	row("Quantization", c.Quant)
	row("Context", fmt.Sprintf("%s tokens · KV %s", formatInt(c.Context), c.KVType))
	row("Fit", strings.ToUpper(c.Fits))
	row("Predicted", model.HumanBytes(c.PredictedBytes)+" [E]")
	if c.ObservedBytes > 0 {
		row("Observed", fmt.Sprintf("%s [%s]", model.HumanBytes(c.ObservedBytes), provenanceShort(c.ObservationProvenance)))
		row("Accuracy", fmt.Sprintf("within %.1f%% · signed error %+.1f%%", c.AbsoluteErrorPct, c.SignedErrorPct))
	} else {
		row("Observed", "pending — load the model and run watch/report again")
		row("Accuracy", "pending")
	}
	row("Prediction", c.PredictionID)
	row("Version", c.Version)
	if !c.GeneratedAt.IsZero() {
		row("Generated", c.GeneratedAt.Format(time.RFC3339))
	}
	b.WriteString("\n[M] measured · [R] loader-reported · [E] model-estimated\n")
	return b.String()
}

func markdownCell(s string) string {
	s = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return ' '
		}
		return r
	}, s)
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", "|", "&#124;",
		"\\", "&#92;", "`", "&#96;", "*", "&#42;", "_", "&#95;",
		"[", "&#91;", "]", "&#93;", "!", "&#33;", "~", "&#126;",
	).Replace(s)
}

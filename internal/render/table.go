package render

import (
	"fmt"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// Options controls console/TUI rendering.
type Options struct {
	Color    bool
	BarWidth int // cells in the stacked bar; 0 => 48
}

func (o Options) barWidth() int {
	if o.BarWidth <= 0 {
		return 48
	}
	return o.BarWidth
}

const reset = "\x1b[0m"

func sgr(color bool, params, s string) string {
	if !color || params == "" {
		return s
	}
	return "\x1b[" + params + "m" + s + reset
}

func bold(color bool, s string) string   { return sgr(color, "1", s) }
func dim(color bool, s string) string    { return sgr(color, "2", s) }
func redSGR(color bool, s string) string { return sgr(color, "38;5;203", s) }

// Table renders a full snapshot as a coloured console report.
func Table(snap model.Snapshot, opts Options) string {
	var b strings.Builder
	header := "vramwatch"
	if snap.Version != "" {
		header += " " + snap.Version
	}
	meta := []string{}
	if snap.Host != "" {
		meta = append(meta, snap.Host)
	}
	if !snap.Timestamp.IsZero() {
		meta = append(meta, snap.Timestamp.Format(time.RFC3339))
	}
	fmt.Fprint(&b, bold(opts.Color, header))
	if len(meta) > 0 {
		fmt.Fprint(&b, dim(opts.Color, "  •  "+strings.Join(meta, "  •  ")))
	}
	b.WriteString("\n")

	if len(snap.Breakdowns) == 0 {
		b.WriteString(dim(opts.Color, "no GPUs detected\n"))
		return b.String()
	}

	for _, bd := range snap.Breakdowns {
		b.WriteString("\n")
		renderBreakdown(&b, bd, opts)
	}
	return b.String()
}

func renderBreakdown(b *strings.Builder, bd model.Breakdown, opts Options) {
	g := bd.GPU
	title := fmt.Sprintf("GPU %d  %s", g.Index, g.Name)
	sub := string(g.Vendor) + ", " + memoryKindName(g.MemoryKind)
	if g.Driver != "" {
		sub += ", driver " + g.Driver
	}
	fmt.Fprintf(b, "%s  %s\n", bold(opts.Color, title), dim(opts.Color, "("+sub+")"))

	segs := orderedSegments(bd)
	vals := make([]uint64, len(segs))
	for i, s := range segs {
		vals[i] = s.Bytes
	}
	cells := allocateCells(vals, opts.barWidth())

	// The stacked bar.
	var bar strings.Builder
	for i, s := range segs {
		st := styleFor(s.Kind)
		glyph := "█"
		if !opts.Color {
			glyph = string(st.glyph)
		}
		run := strings.Repeat(glyph, cells[i])
		bar.WriteString(sgr(opts.Color, st.ansi, run))
	}
	used := bd.Used()
	fmt.Fprintf(b, "[%s]  %s %s used / %s %s capacity\n", bar.String(), provenanceCode(g.UsageSource),
		model.HumanBytes(used), provenanceCode(g.CapacitySource), model.HumanBytes(g.TotalBytes))

	// Legend.
	for _, s := range segs {
		st := styleFor(s.Kind)
		swatch := sgr(opts.Color, st.ansi, "█")
		note := ""
		if s.Source != "" && s.Source != "driver" {
			note = s.Source
		}
		if s.Estimated {
			if note != "" {
				note += ", "
			}
			note += "estimated"
		}
		line := fmt.Sprintf("  %s %s %-11s %10s  %5.1f%%", swatch, provenanceBadge(s), s.Label, model.HumanBytes(s.Bytes), pct(s.Bytes, g.TotalBytes))
		if note != "" {
			line += "  " + dim(opts.Color, "("+note+")")
		}
		b.WriteString(line + "\n")
	}
	b.WriteString(dim(opts.Color, "  [M] measured  [R] loader-reported  [E] model-estimated  [A] assumed\n"))

	// Models.
	for _, m := range bd.Models {
		ctx := ""
		if m.ContextTokens > 0 {
			ctx = fmt.Sprintf("ctx %d", m.ContextTokens)
			if m.ContextMax > 0 {
				ctx += fmt.Sprintf("/%d", m.ContextMax)
			}
		}
		parts := []string{m.Name}
		if ctx != "" {
			parts = append(parts, ctx)
		}
		fmt.Fprintf(b, "  %s %s\n", dim(opts.Color, "[R] model:"), strings.Join(parts, "  "))
	}

	// Prediction.
	if p := bd.Prediction; p != nil {
		msg := fmt.Sprintf("%s headroom %s • [E] ~%s/token • max context ≈ %s tokens",
			provenanceCode(g.UsageSource), model.HumanBytes(p.HeadroomBytes), model.HumanBytes(p.KVBytesPerToken), formatInt(p.MaxContextFits))
		if p.OOMRisk {
			fmt.Fprintf(b, "  %s %s\n", redSGR(opts.Color, "⚠ OOM risk:"), msg)
		} else {
			fmt.Fprintf(b, "  %s %s\n", dim(opts.Color, "predict:"), msg)
		}
	}

	// Warnings.
	for _, w := range bd.Warnings {
		fmt.Fprintf(b, "  %s %s\n", dim(opts.Color, "note:"), w)
	}
}

func provenanceBadge(s model.Segment) string {
	p := s.Provenance
	if p == "" {
		if s.Estimated {
			p = model.ProvenanceEstimated
		} else if s.Source == "driver" {
			p = model.ProvenanceMeasured
		} else {
			p = model.ProvenanceReported
		}
	}
	return provenanceCode(p)
}

func provenanceCode(p model.Provenance) string {
	switch p {
	case model.ProvenanceMeasured:
		return "[M]"
	case model.ProvenanceReported:
		return "[R]"
	case model.ProvenanceAssumed:
		return "[A]"
	case model.ProvenanceUser:
		return "[U]"
	default:
		return "[E]"
	}
}

// formatInt renders an int with thousands separators.
func formatInt(n int) string {
	s := fmt.Sprintf("%d", n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}

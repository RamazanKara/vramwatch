// Package render turns a model.Snapshot into the three output forms vramwatch
// produces: a coloured console/TUI table, machine-readable JSON, and a branded
// SVG scorecard (the shareable artifact).
package render

import (
	"sort"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// kindOrder is the fixed left-to-right order segments are drawn and listed in.
var kindOrder = []model.SegmentKind{
	model.KindWeights,
	model.KindKVCache,
	model.KindCompute,
	model.KindOtherProcess,
	model.KindFree,
}

type style struct {
	ansi  string // ANSI SGR parameters, e.g. "38;5;39"
	hex   string // SVG fill colour
	glyph rune   // ASCII glyph for no-colour bars
}

var styles = map[model.SegmentKind]style{
	model.KindWeights:      {ansi: "38;5;39", hex: "#4C9BE8", glyph: '#'},
	model.KindKVCache:      {ansi: "38;5;214", hex: "#E8B84C", glyph: '+'},
	model.KindCompute:      {ansi: "38;5;43", hex: "#4CE0C0", glyph: '='},
	model.KindOtherProcess: {ansi: "38;5;244", hex: "#6E7681", glyph: ':'},
	model.KindFree:         {ansi: "38;5;35", hex: "#3FB950", glyph: '.'},
}

func styleFor(k model.SegmentKind) style {
	if s, ok := styles[k]; ok {
		return s
	}
	return style{ansi: "0", hex: "#8B949E", glyph: ' '}
}

// orderedSegments returns a breakdown's segments in canonical draw order,
// dropping zero-byte segments.
func orderedSegments(b model.Breakdown) []model.Segment {
	byKind := map[model.SegmentKind]model.Segment{}
	for _, s := range b.Segments {
		byKind[s.Kind] = s
	}
	var out []model.Segment
	for _, k := range kindOrder {
		if s, ok := byKind[k]; ok && s.Bytes > 0 {
			out = append(out, s)
		}
	}
	// Append any segment kinds not in kindOrder (future-proofing) in a stable order.
	if len(out) != len(b.Segments) {
		seen := map[model.SegmentKind]bool{}
		for _, s := range out {
			seen[s.Kind] = true
		}
		var extra []model.Segment
		for _, s := range b.Segments {
			if !seen[s.Kind] && s.Bytes > 0 {
				extra = append(extra, s)
			}
		}
		sort.SliceStable(extra, func(i, j int) bool { return extra[i].Kind < extra[j].Kind })
		out = append(out, extra...)
	}
	return out
}

// allocateCells distributes width cells across vals proportionally using the
// largest-remainder method, guaranteeing the cells sum to width and that every
// strictly-positive value receives at least one cell when width permits.
func allocateCells(vals []uint64, width int) []int {
	cells := make([]int, len(vals))
	if width <= 0 || len(vals) == 0 {
		return cells
	}
	var total uint64
	positive := 0
	for _, v := range vals {
		total += v
		if v > 0 {
			positive++
		}
	}
	if total == 0 {
		return cells
	}

	// Reserve one cell per positive value if there's room, so nothing vanishes.
	reserve := 0
	if positive <= width {
		reserve = positive
	}
	remaining := width - reserve

	type frac struct {
		i    int
		frac float64
	}
	var fracs []frac
	assigned := 0
	for i, v := range vals {
		exact := float64(v) / float64(total) * float64(remaining)
		n := int(exact)
		cells[i] = n
		assigned += n
		fracs = append(fracs, frac{i, exact - float64(n)})
	}
	// Hand out the leftover cells to the largest remainders.
	leftover := remaining - assigned
	sort.SliceStable(fracs, func(a, b int) bool { return fracs[a].frac > fracs[b].frac })
	for k := 0; k < leftover && k < len(fracs); k++ {
		cells[fracs[k].i]++
	}
	// Add the reserved cell back to each positive value.
	if reserve > 0 {
		for i, v := range vals {
			if v > 0 {
				cells[i]++
			}
		}
	}
	return cells
}

func pct(part, whole uint64) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole) * 100
}

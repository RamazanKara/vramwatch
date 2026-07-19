// gifgen renders the deterministic README walkthrough at docs/demo.gif.
//
// The animation is an illustrative scenario, not a hardware benchmark. Every
// value is composed deterministically and carries the same provenance badge the
// real CLI uses. The tool is a standalone module because the shipped vramwatch
// binary intentionally has no third-party Go dependencies.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

const (
	KiB = 1 << 10
	MiB = 1 << 20
	GiB = 1 << 30
)

// Logical canvas, upscaled 2x for a crisp 1040x600 README asset.
const (
	lw = 520
	lh = 300
)

var (
	cBG        = color.RGBA{0x09, 0x0d, 0x13, 0xff}
	cPanel     = color.RGBA{0x0d, 0x11, 0x17, 0xff}
	cPanelHi   = color.RGBA{0x16, 0x1b, 0x22, 0xff}
	cTrack     = color.RGBA{0x21, 0x26, 0x2d, 0xff}
	cBorder    = color.RGBA{0x30, 0x36, 0x3d, 0xff}
	cFG        = color.RGBA{0xe6, 0xed, 0xf3, 0xff}
	cMuted     = color.RGBA{0x8b, 0x94, 0x9e, 0xff}
	cBlue      = color.RGBA{0x58, 0xa6, 0xff, 0xff}
	cCyan      = color.RGBA{0x39, 0xd3, 0xbb, 0xff}
	cYellow    = color.RGBA{0xd2, 0xa8, 0x4a, 0xff}
	cPurple    = color.RGBA{0xbc, 0x8c, 0xff, 0xff}
	cGreen     = color.RGBA{0x3f, 0xb9, 0x50, 0xff}
	cGreenDark = color.RGBA{0x18, 0x4d, 0x28, 0xff}
	cRed       = color.RGBA{0xf8, 0x51, 0x49, 0xff}
	cRedDark   = color.RGBA{0x56, 0x20, 0x24, 0xff}
)

var palette = color.Palette{
	cBG, cPanel, cPanelHi, cTrack, cBorder, cFG, cMuted,
	cBlue, cCyan, cYellow, cPurple, cGreen, cGreenDark, cRed, cRedDark,
}

const (
	modelWeights    = uint64(2019377376)
	totalVRAM       = uint64(8 * GiB)
	availableVRAM   = uint64(7600 * MiB)
	runtimeCeiling  = uint64(304 * MiB)
	runtimeExpected = uint64(208 * MiB)
	safetyMargin    = uint64(512 * MiB)
	watchRuntime    = uint64(320 * MiB)
	watchOther      = uint64(512 * MiB)
	kvPerToken      = uint64(112 * KiB)
	targetContext   = 32768
)

func fillRect(img *image.Paletted, x, y, w, h int, c color.Color) {
	if w <= 0 || h <= 0 {
		return
	}
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			if image.Pt(xx, yy).In(img.Bounds()) {
				img.Set(xx, yy, c)
			}
		}
	}
}

func strokeRect(img *image.Paletted, x, y, w, h int, c color.Color) {
	fillRect(img, x, y, w, 1, c)
	fillRect(img, x, y+h-1, w, 1, c)
	fillRect(img, x, y, 1, h, c)
	fillRect(img, x+w-1, y, 1, h, c)
}

func drawText(img *image.Paletted, x, y int, s string, c color.Color) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
}

func textRight(img *image.Paletted, right, y int, s string, c color.Color) {
	drawText(img, right-len(s)*7, y, s, c)
}

func textCenter(img *image.Paletted, center, y int, s string, c color.Color) {
	drawText(img, center-len(s)*7/2, y, s, c)
}

func human(b uint64) string {
	switch {
	case b >= GiB:
		return fmt.Sprintf("%.2f GiB", float64(b)/GiB)
	case b >= MiB:
		return fmt.Sprintf("%.1f MiB", float64(b)/MiB)
	case b >= KiB:
		return fmt.Sprintf("%.1f KiB", float64(b)/KiB)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func commas(n int) string {
	s := fmt.Sprintf("%d", n)
	var out []byte
	for i := 0; i < len(s); i++ {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, s[i])
	}
	return string(out)
}

func drawBadge(img *image.Paletted, x, y, w int, label string, bg, fg color.Color) {
	fillRect(img, x, y, w, 18, bg)
	textCenter(img, x+w/2, y+13, label, fg)
}

func drawChrome(img *image.Paletted, active int) {
	fillRect(img, 0, 0, lw, lh, cBG)
	fillRect(img, 1, 1, lw-2, lh-2, cPanel)
	strokeRect(img, 0, 0, lw, lh, cBorder)

	drawText(img, 16, 24, "vramwatch", cFG)
	drawText(img, 88, 24, "know what fits before loading", cMuted)
	drawBadge(img, 430, 10, 74, "LOCAL LLM", cPanelHi, cCyan)

	tabs := []string{"1  FIT", "2  WATCH", "3  DOCTOR", "4  REPORT"}
	for i, tab := range tabs {
		x := 16 + i*122
		bg, fg := cPanelHi, cMuted
		if i == active {
			bg, fg = cBlue, cBG
		}
		drawBadge(img, x, 40, 112, tab, bg, fg)
	}

	fillRect(img, 16, 286, lw-32, 1, cBorder)
	drawText(img, 16, 298, "[M] measured  [R] loader-reported  [E] estimated  [A] assumed", cMuted)
}

func drawCommand(img *image.Paletted, lines ...string) {
	h := 12 + len(lines)*16
	fillRect(img, 16, 72, lw-32, h, cBG)
	strokeRect(img, 16, 72, lw-32, h, cBorder)
	for i, line := range lines {
		drawText(img, 27, 92+i*16, line, cFG)
	}
}

func drawFit(img *image.Paletted, progress int) {
	drawChrome(img, 0)
	drawCommand(img,
		"$ vramwatch fit ollama:llama3.2:3b-instruct \\",
		"    --quant q4_k_m --context 32768",
	)

	y := 126
	if progress == 0 {
		drawText(img, 27, y+16, "resolving bounded GGUF metadata...", cCyan)
		drawText(img, 27, y+34, "model tensors are not downloaded", cMuted)
		return
	}

	drawText(img, 27, y, "PREFLIGHT  model not downloaded or launched", cCyan)
	rows := []struct {
		label string
		value string
		col   color.Color
	}{
		{"[E] weights", human(modelWeights), cBlue},
		{"[E] KV cache", human(kvPerToken * targetContext), cYellow},
		{"[A] runtime ceiling", human(runtimeCeiling), cPurple},
		{"[A] safety margin", human(safetyMargin), cPurple},
	}
	for i, row := range rows {
		ry := y + 20 + i*17
		drawText(img, 27, ry, row.label, row.col)
		textRight(img, 245, ry, row.value, cFG)
	}

	if progress < 2 {
		return
	}
	boxY := 214
	fillRect(img, 16, boxY, lw-32, 66, cPanelHi)
	strokeRect(img, 16, boxY, lw-32, 66, cBorder)
	drawText(img, 27, boxY+17, "GPU 0  NVIDIA GeForce RTX 4060 Ti  (dedicated VRAM)", cFG)
	drawText(img, 27, boxY+35, "[M] 8.00 GiB total / [M] 7.42 GiB free now", cMuted)
	required := modelWeights + kvPerToken*targetContext + runtimeCeiling + safetyMargin
	textRight(img, lw-27, boxY+35, "required "+human(required), cMuted)
	if progress >= 3 {
		drawText(img, 27, boxY+54, "on device", cMuted)
		drawBadge(img, 104, boxY+39, 55, "FITS", cGreenDark, cGreen)
		drawText(img, 181, boxY+54, "right now", cMuted)
		drawBadge(img, 265, boxY+39, 55, "FITS", cGreenDark, cGreen)
		textRight(img, lw-27, boxY+54, human(availableVRAM-required)+" spare", cGreen)
	}
}

type barSegment struct {
	bytes uint64
	col   color.Color
}

func drawBar(img *image.Paletted, x, y, w, h int, total uint64, segs []barSegment) {
	fillRect(img, x, y, w, h, cTrack)
	cursor := x
	for i, seg := range segs {
		sw := int(float64(w) * float64(seg.bytes) / float64(total))
		if i == len(segs)-1 {
			sw = x + w - cursor
		}
		if cursor+sw > x+w {
			sw = x + w - cursor
		}
		fillRect(img, cursor, y, sw, h, seg.col)
		cursor += sw
	}
}

func drawLegendRow(img *image.Paletted, y int, badge, label string, bytes uint64, col color.Color) {
	fillRect(img, 27, y-10, 10, 10, col)
	drawText(img, 45, y, badge, col)
	drawText(img, 75, y, label, cFG)
	textRight(img, 315, y, human(bytes), cFG)
	textRight(img, lw-27, y, fmt.Sprintf("%5.1f%%", float64(bytes)/float64(totalVRAM)*100), cMuted)
}

func drawWatch(img *image.Paletted, tokens int) {
	drawChrome(img, 1)
	drawCommand(img, "$ vramwatch watch")

	kv := kvPerToken * uint64(tokens)
	used := modelWeights + kv + watchRuntime + watchOther
	if used > totalVRAM {
		used = totalVRAM
	}
	free := totalVRAM - used
	oom := free < 512*MiB

	drawText(img, 27, 122, "[R] model: llama3.2:3b-instruct-q4_K_M", cMuted)
	textRight(img, lw-27, 122, "ctx "+commas(tokens), cMuted)
	drawBar(img, 27, 133, lw-54, 24, totalVRAM, []barSegment{
		{modelWeights, cBlue},
		{kv, cYellow},
		{watchRuntime, cCyan},
		{watchOther, cPurple},
		{free, cGreen},
	})
	drawText(img, 27, 174, "[M] "+human(used)+" used", cFG)
	textRight(img, lw-27, 174, "[M] "+human(totalVRAM)+" capacity", cFG)

	drawLegendRow(img, 193, "[E]", "weights", modelWeights, cBlue)
	drawLegendRow(img, 210, "[E]", "KV cache", kv, cYellow)
	drawLegendRow(img, 227, "[E]", "runtime", watchRuntime, cCyan)
	drawLegendRow(img, 244, "[E]", "other apps", watchOther, cPurple)
	drawLegendRow(img, 261, "[M]", "free", free, cGreen)

	label := "[M] headroom " + human(free) + " / [E] ~112.0 KiB/token"
	if oom {
		fillRect(img, 16, 266, lw-32, 16, cRedDark)
		drawText(img, 27, 278, "OOM RISK  "+label, cRed)
	} else {
		drawText(img, 27, 278, label, cMuted)
	}
}

func drawDoctor(img *image.Paletted, progress int) {
	drawChrome(img, 2)
	drawCommand(img, "$ vramwatch doctor")

	drawText(img, 27, 122, "driver -> GPU -> loader -> runtime -> state", cCyan)
	checks := []struct {
		layer string
		text  string
	}{
		{"system", "linux/amd64"},
		{"driver", "nvidia-smi detected 1 device"},
		{"loader", "ollama healthy; 1 resident model"},
		{"runtime", "resident model has GPU-memory evidence"},
		{"state", "prediction ledger ready"},
	}
	for i, check := range checks {
		if progress <= i {
			break
		}
		y := 136 + i*27
		fillRect(img, 16, y, lw-32, 22, cPanelHi)
		drawBadge(img, 24, y+2, 48, "PASS", cGreenDark, cGreen)
		drawText(img, 86, y+16, check.layer, cMuted)
		drawText(img, 156, y+16, check.text, cFG)
	}
	if progress >= len(checks) {
		drawText(img, 27, 280, "healthy: true  /  no hidden provider failures", cGreen)
	}
}

func drawReport(img *image.Paletted) {
	drawChrome(img, 3)
	drawCommand(img, "$ vramwatch report --svg --output vramwatch-report.svg")

	cardX, cardY, cardW, cardH := 27, 113, lw-54, 164
	fillRect(img, cardX, cardY, cardW, cardH, cBG)
	strokeRect(img, cardX, cardY, cardW, cardH, cBorder)
	drawText(img, cardX+13, cardY+21, "vramwatch report", cFG)
	drawBadge(img, cardX+cardW-75, cardY+8, 62, "FITS", cGreenDark, cGreen)
	fillRect(img, cardX+13, cardY+30, cardW-26, 1, cBorder)

	expected := modelWeights + kvPerToken*targetContext + runtimeExpected
	observed := uint64(5856 * MiB)
	signed := 100 * (float64(expected) - float64(observed)) / float64(observed)
	absolute := math.Abs(signed)
	rows := []struct {
		label string
		value string
	}{
		{"hardware", "RTX 4060 Ti / dedicated VRAM / 8.00 GiB"},
		{"model", "llama3.2:3b-instruct / Q4_K_M"},
		{"context", "32,768 tokens / KV F16"},
		{"predicted", human(expected) + " [E]"},
		{"observed", human(observed) + " [M]"},
		{"accuracy", fmt.Sprintf("within %.1f%% / signed error %+.1f%%", absolute, signed)},
	}
	for i, row := range rows {
		y := cardY + 48 + i*18
		drawText(img, cardX+13, y, row.label, cMuted)
		drawText(img, cardX+104, y, row.value, cFG)
	}
	drawText(img, cardX+13, cardY+157, "shareable SVG / no hostname, PID, bus ID, or local path", cCyan)
}

func newFrame() *image.Paletted {
	return image.NewPaletted(image.Rect(0, 0, lw, lh), palette)
}

func upscale2x(src *image.Paletted) *image.Paletted {
	b := src.Bounds()
	dst := image.NewPaletted(image.Rect(0, 0, b.Dx()*2, b.Dy()*2), src.Palette)
	for y := 0; y < b.Dy()*2; y++ {
		for x := 0; x < b.Dx()*2; x++ {
			dst.SetColorIndex(x, y, src.ColorIndexAt(x/2, y/2))
		}
	}
	return dst
}

func appendFrame(g *gif.GIF, img *image.Paletted, delay int) {
	g.Image = append(g.Image, upscale2x(img))
	g.Delay = append(g.Delay, delay)
	g.Disposal = append(g.Disposal, gif.DisposalNone)
}

func main() {
	out := "docs/demo.gif"
	if len(os.Args) > 1 {
		out = os.Args[1]
	}

	g := &gif.GIF{LoopCount: 0}
	for progress := 0; progress <= 4; progress++ {
		img := newFrame()
		drawFit(img, progress)
		delay := 22
		if progress == 4 {
			delay = 110
		}
		appendFrame(g, img, delay)
	}
	for i, tokens := range []int{8192, 16384, 24576, 32768, 40960, 45056, 49152} {
		img := newFrame()
		drawWatch(img, tokens)
		delay := 18
		if i == 6 {
			delay = 110
		}
		appendFrame(g, img, delay)
	}
	for progress := 0; progress <= 5; progress++ {
		img := newFrame()
		drawDoctor(img, progress)
		delay := 22
		if progress == 5 {
			delay = 110
		}
		appendFrame(g, img, delay)
	}
	report := newFrame()
	drawReport(report)
	appendFrame(g, report, 180)

	f, err := os.Create(out)
	if err != nil {
		panic(err)
	}
	if err := gif.EncodeAll(f, g); err != nil {
		f.Close()
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s (%d frames, %dx%d)\n", out, len(g.Image), lw*2, lh*2)
}

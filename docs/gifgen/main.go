// gifgen renders docs/demo.gif for vramwatch: an animated stacked VRAM bar whose
// KV cache grows until the card OOMs, mirroring `vramwatch watch --source demo`.
// Standalone tool (uses x/image for a bitmap font); NOT part of the vramwatch module.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
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

// Logical canvas (upscaled 2x into the GIF).
const (
	lw = 440
	lh = 182
)

var (
	cBG      = color.RGBA{0x0d, 0x11, 0x17, 0xff}
	cTrack   = color.RGBA{0x16, 0x1b, 0x22, 0xff}
	cBorder  = color.RGBA{0x30, 0x36, 0x3d, 0xff}
	cFG      = color.RGBA{0xe6, 0xed, 0xf3, 0xff}
	cMuted   = color.RGBA{0x8b, 0x94, 0x9e, 0xff}
	cWeights = color.RGBA{0x4c, 0x9b, 0xe8, 0xff}
	cKV      = color.RGBA{0xe8, 0xb8, 0x4c, 0xff}
	cCompute = color.RGBA{0x4c, 0xe0, 0xc0, 0xff}
	cOther   = color.RGBA{0x6e, 0x76, 0x81, 0xff}
	cFree    = color.RGBA{0x3f, 0xb9, 0x50, 0xff}
	cRed     = color.RGBA{0xf8, 0x51, 0x49, 0xff}
)

var palette = color.Palette{cBG, cTrack, cBorder, cFG, cMuted, cWeights, cKV, cCompute, cOther, cFree, cRed}

func human(b uint64) string {
	switch {
	case b >= GiB:
		return fmt.Sprintf("%.2f GiB", float64(b)/GiB)
	case b >= MiB:
		return fmt.Sprintf("%.0f MiB", float64(b)/MiB)
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

func fillRect(img *image.Paletted, x, y, w, h int, c color.Color) {
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			img.Set(xx, yy, c)
		}
	}
}

func text(img *image.Paletted, x, y int, s string, c color.Color) {
	d := &font.Drawer{Dst: img, Src: image.NewUniform(c), Face: basicfont.Face7x13,
		Dot: fixed.P(x, y)}
	d.DrawString(s)
}

func textRight(img *image.Paletted, xRight, y int, s string, c color.Color) {
	text(img, xRight-len(s)*7, y, s, c)
}

type seg struct {
	bytes uint64
	col   color.Color
}

func drawFrame(tokens int) *image.Paletted {
	const total = 24 * GiB
	const other = 1 * GiB
	const weights = 5600 * MiB
	const overhead = 800 * MiB
	kvPerTok := uint64(2 * 32 * 8 * 128 * 16 / 8) // 128 KiB
	kv := kvPerTok * uint64(tokens)

	budget := uint64(total - other) // room for the inference process
	infProc := weights + overhead + kv
	if infProc > budget {
		infProc = budget
		kv = budget - weights - overhead
	}
	free := uint64(total) - other - infProc
	oom := free < 512*MiB

	img := image.NewPaletted(image.Rect(0, 0, lw, lh), palette)
	fillRect(img, 0, 0, lw, lh, cBG)
	// border
	fillRect(img, 0, 0, lw, 1, cBorder)
	fillRect(img, 0, lh-1, lw, 1, cBorder)
	fillRect(img, 0, 0, 1, lh, cBorder)
	fillRect(img, lw-1, 0, 1, lh, cBorder)

	text(img, 12, 20, "vramwatch", cFG)
	text(img, 12+len("vramwatch")*7+8, 20, "local-LLM VRAM, live", cMuted)
	usedTotal := fmt.Sprintf("%s / %s", human(uint64(total)-free), human(total))
	textRight(img, lw-12, 20, usedTotal, cMuted)

	text(img, 12, 38, "GPU 0  AMD Radeon RX 7900 XTX", cFG)

	// Stacked bar.
	bx, by, bw, bh := 12, 48, lw-24, 22
	fillRect(img, bx, by, bw, bh, cTrack)
	segs := []seg{
		{weights, cWeights},
		{kv, cKV},
		{overhead, cCompute},
		{other, cOther},
		{free, cFree},
	}
	x := bx
	acc := 0
	for i, s := range segs {
		w := int(float64(bw) * float64(s.bytes) / float64(total))
		if i == len(segs)-1 {
			w = bx + bw - x // last fills to the end
		}
		if w < 0 {
			w = 0
		}
		fillRect(img, x, by, w, bh, s.col)
		x += w
		acc += w
	}

	// Legend.
	rows := []struct {
		col   color.Color
		label string
		bytes uint64
	}{
		{cWeights, "weights", weights},
		{cKV, "KV cache", kv},
		{cCompute, "compute", overhead},
		{cOther, "other apps", other},
		{cFree, "free", free},
	}
	ly := 84
	for _, r := range rows {
		fillRect(img, 12, ly-8, 9, 9, r.col)
		text(img, 26, ly, r.label, cFG)
		textRight(img, 300, ly, human(r.bytes), cFG)
		pct := float64(r.bytes) / float64(total) * 100
		textRight(img, lw-14, ly, fmt.Sprintf("%.1f%%", pct), cMuted)
		ly += 15
	}

	// Model + prediction line.
	text(img, 12, ly+3, fmt.Sprintf("model: llama3:8b  ctx %s", commas(tokens)), cMuted)
	pline := fmt.Sprintf("free %s  -  128 KiB/token", human(free))
	if oom {
		textRight(img, lw-12, ly+3, "! OOM RISK", cRed)
	} else {
		textRight(img, lw-12, ly+3, pline, cMuted)
	}
	return img
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

func main() {
	out := "docs/demo.gif"
	if len(os.Args) > 1 {
		out = os.Args[1]
	}
	const N = 46
	g := &gif.GIF{LoopCount: 0}
	for i := 0; i < N; i++ {
		frac := float64(i) / float64(N-1)
		tokens := 2048 + int(frac*frac*(180000-2048)) // ease-in growth
		frame := upscale2x(drawFrame(tokens))
		g.Image = append(g.Image, frame)
		delay := 8
		if i == 0 {
			delay = 90 // hold the calm start
		}
		if i >= N-3 {
			delay = 60 // linger on the OOM
		}
		g.Delay = append(g.Delay, delay)
	}
	f, err := os.Create(out)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err := gif.EncodeAll(f, g); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s (%d frames)\n", out, N)
}

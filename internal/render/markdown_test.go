package render

import (
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestReportMarkdown(t *testing.T) {
	for _, tc := range []struct {
		name     string
		observed uint64
		prov     model.Provenance
		static   bool
		want     string
	}{
		{"pending", 0, "", true, "| Accuracy | pending |"},
		{"measured", 7 * model.GiB, model.ProvenanceMeasured, false, "7.00 GiB &#91;M&#93;"},
		{"reported", 7 * model.GiB, model.ProvenanceReported, true, "7.00 GiB &#91;R&#93;"},
		{"estimated", 7 * model.GiB, model.ProvenanceEstimated, true, "7.00 GiB &#91;E&#93;"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := ReportCard{Version: "dev", PredictionID: "0123456789abcdef",
				GPUName: "Example GPU", MemoryKind: model.MemoryUnified, CapacityBytes: 24 * model.GiB,
				Driver: "1.2", Loader: "ollama", Model: "model:8b", Quant: "Q4_K_M", Context: 32768, KVType: "F16",
				PredictedBytes: 7 * model.GiB, ObservedBytes: tc.observed, ObservationProvenance: tc.prov,
				AbsoluteErrorPct: 0.4, SignedErrorPct: -0.4, Fits: "fits"}
			if !tc.static {
				c.GeneratedAt = time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)
			}
			out := ReportMarkdown(c)
			for _, want := range []string{"# vramwatch report\n", "Example GPU", "unified memory", "24.00 GiB", "model:8b", "Q4&#95;K&#95;M", "32,768 tokens · KV F16", "FITS", c.PredictionID, tc.want} {
				if !strings.Contains(out, want) {
					t.Errorf("report missing %q:\n%s", want, out)
				}
			}
			if strings.Contains(out, "| Generated |") == tc.static || strings.Contains(out, "0001-01-01") {
				t.Errorf("incorrect timestamp in report:\n%s", out)
			}
			if tc.observed > 0 && !strings.Contains(out, "within 0.4% · signed error -0.4%") {
				t.Error("missing signed/absolute accuracy")
			}
			if ReportMarkdown(c) != out {
				t.Error("rendering is not deterministic")
			}
		})
	}
}

func TestMarkdownCell(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{"model | name", "model &#124; name"},
		{"<b>&amp;</b>", "&lt;b&gt;&amp;amp;&lt;/b&gt;"},
		{"a\r\nb\tc\x1bd", "a  b c d"},
		{"[x](url)!*a*_b_`c`~d~\\", "&#91;x&#93;(url)&#33;&#42;a&#42;&#95;b&#95;&#96;c&#96;&#126;d&#126;&#92;"},
		{"模型 α", "模型 α"},
	} {
		t.Run(tc.input, func(t *testing.T) {
			if got := markdownCell(tc.input); got != tc.want {
				t.Errorf("markdownCell(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func FuzzMarkdownCell(f *testing.F) {
	for _, seed := range []string{"model-Q4_K_M.gguf", "|\n<script>bad</script>", "![x](url)\\", "\x00\x1b[2J", ""} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, s string) {
		cell := markdownCell(s)
		if strings.ContainsAny(cell, "|<>\\`*_[]!~") {
			t.Fatalf("unescaped markup in %q", cell)
		}
		for _, r := range cell {
			if unicode.IsControl(r) || (unicode.IsSpace(r) && r != ' ') {
				t.Fatalf("control/line separator in %q", cell)
			}
		}
		if strings.Count(ReportMarkdown(ReportCard{Model: s}), "\n") != strings.Count(ReportMarkdown(ReportCard{}), "\n") {
			t.Fatal("model name changed the table structure")
		}
	})
}

package render

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

var updateGolden = flag.Bool("update-golden", false, "rewrite the JSON schema golden file")

// TestJSONSchemaStable pins the exact `--json` output for a fixed snapshot. Any
// change to the machine-readable schema (a field added, removed, renamed, or
// reformatted) fails this test, so a breaking change to the JSON contract can't
// land unnoticed. If a change is intentional, regenerate the golden:
//
//	go test ./internal/render -run JSONSchemaStable -update-golden
func TestJSONSchemaStable(t *testing.T) {
	data, err := JSON(sampleSnap()) // sampleSnap is fully deterministic
	if err != nil {
		t.Fatal(err)
	}
	const golden = "testdata/snapshot.golden.json"
	if *updateGolden {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(golden, data, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (generate it with -update-golden): %v", err)
	}
	if !bytes.Equal(data, want) {
		t.Errorf("--json schema changed. If this is intentional, regenerate the golden:\n"+
			"  go test ./internal/render -run JSONSchemaStable -update-golden\n\ngot:\n%s", data)
	}
}

func sampleSnap() model.Snapshot {
	return model.Snapshot{
		Version: "v0.1.0", Host: "bench",
		Timestamp: time.Date(2026, 7, 5, 12, 0, 0, 0, time.UTC),
		Breakdowns: []model.Breakdown{{
			GPU: model.GPU{Index: 0, Name: "AMD Radeon RX 7900 XTX", Vendor: model.VendorAMD, Driver: "6.10.5", TotalBytes: 24 * model.GiB},
			Segments: []model.Segment{
				{Kind: model.KindWeights, Label: "weights", Bytes: 19*model.GiB + 512*model.MiB, Source: "ollama", Estimated: true},
				{Kind: model.KindKVCache, Label: "KV cache", Bytes: 2*model.GiB + 512*model.MiB, Source: "ollama", Estimated: true},
				{Kind: model.KindOtherProcess, Label: "other apps", Bytes: 1*model.GiB + 768*model.MiB},
				{Kind: model.KindFree, Label: "free", Bytes: 256 * model.MiB},
			},
			Models:     []model.LoaderModel{{Loader: "ollama", Name: "llama3:70b-q2_K", ContextTokens: 8192, ContextMax: 8192}},
			Prediction: &model.Prediction{Model: "llama3:70b-q2_K", KVBytesPerToken: 327680, MaxContextFits: 8192, HeadroomBytes: 256 * model.MiB, OOMRisk: true},
		}},
	}
}

func TestAllocateCellsSumsExactly(t *testing.T) {
	cases := [][]uint64{
		{10, 20, 30, 40},
		{1, 1, 1, 1, 1},
		{100, 0, 0, 1},
		{24 * model.GiB, 2 * model.GiB, 1 * model.GiB, 256 * model.MiB},
	}
	for _, vals := range cases {
		for _, width := range []int{10, 48, 3, 5} {
			cells := allocateCells(vals, width)
			sum := 0
			for _, c := range cells {
				if c < 0 {
					t.Fatalf("negative cell for %v w=%d: %v", vals, width, cells)
				}
				sum += c
			}
			if sum != width {
				t.Fatalf("allocateCells(%v, %d) summed to %d: %v", vals, width, sum, cells)
			}
			// Positive values must get >=1 cell when there's room.
			positive := 0
			for _, v := range vals {
				if v > 0 {
					positive++
				}
			}
			if positive <= width {
				for i, v := range vals {
					if v > 0 && cells[i] == 0 {
						t.Fatalf("positive value %d got 0 cells in %v (w=%d)", v, cells, width)
					}
				}
			}
		}
	}
}

func TestTableContainsKeyFacts(t *testing.T) {
	out := Table(sampleSnap(), Options{Color: false})
	for _, want := range []string{"vramwatch", "RX 7900 XTX", "weights", "KV cache", "OOM risk", "llama3:70b-q2_K", "8,192"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q\n%s", want, out)
		}
	}
	// No-colour output must not contain ANSI escapes.
	if strings.Contains(out, "\x1b[") {
		t.Errorf("no-colour table contained ANSI escape:\n%q", out)
	}
}

func TestTableColorEmitsANSI(t *testing.T) {
	out := Table(sampleSnap(), Options{Color: true})
	if !strings.Contains(out, "\x1b[") {
		t.Error("colour table should contain ANSI escapes")
	}
}

func TestJSONRoundTrips(t *testing.T) {
	data, err := JSON(sampleSnap())
	if err != nil {
		t.Fatal(err)
	}
	var back model.Snapshot
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(back.Breakdowns) != 1 || back.Breakdowns[0].GPU.Name != "AMD Radeon RX 7900 XTX" {
		t.Errorf("round-trip lost data: %+v", back)
	}
}

func TestSVGWellFormed(t *testing.T) {
	out := SVG(sampleSnap())
	if !strings.HasPrefix(out, "<svg") || !strings.HasSuffix(out, "</svg>") {
		t.Fatal("SVG not wrapped in <svg> tags")
	}
	for _, want := range []string{"vramwatch", "RX 7900 XTX", "#E8B84C", "OOM risk", "clipPath"} {
		if !strings.Contains(out, want) {
			t.Errorf("SVG missing %q", want)
		}
	}
	if strings.Contains(out, "<text></text>") {
		t.Error("empty text node")
	}
}

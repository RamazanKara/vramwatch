package loader

import (
	"encoding/json"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/engine"
	"github.com/RamazanKara/vramwatch/internal/model"
)

const llama3ModelInfo = `{
  "general.architecture": "llama",
  "llama.block_count": 32,
  "llama.attention.head_count": 32,
  "llama.attention.head_count_kv": 8,
  "llama.attention.key_length": 128,
  "llama.embedding_length": 4096,
  "llama.context_length": 8192
}`

func TestParseOllamaArch(t *testing.T) {
	var info map[string]any
	if err := json.Unmarshal([]byte(llama3ModelInfo), &info); err != nil {
		t.Fatal(err)
	}
	ai := parseOllamaArch(info)
	want := model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16}
	if ai.arch != want {
		t.Errorf("arch = %+v, want %+v", ai.arch, want)
	}
	if ai.ctxMax != 8192 {
		t.Errorf("ctxMax = %d, want 8192", ai.ctxMax)
	}
	// Cross-check against the engine's known KV figure.
	if engine.KVBytesPerToken(ai.arch) != 131072 {
		t.Errorf("kv/token = %d, want 131072", engine.KVBytesPerToken(ai.arch))
	}
}

func TestParseOllamaArchFallbacks(t *testing.T) {
	// No head_count_kv (MHA) and no key_length: derive from head_count/embedding.
	info := map[string]any{
		"general.architecture":       "qwen2",
		"qwen2.block_count":          float64(28),
		"qwen2.attention.head_count": float64(16),
		"qwen2.embedding_length":     float64(2048),
		"qwen2.context_length":       float64(32768),
	}
	ai := parseOllamaArch(info)
	if ai.arch.KVHeads != 16 {
		t.Errorf("KVHeads fallback = %d, want 16", ai.arch.KVHeads)
	}
	if ai.arch.HeadDim != 128 { // 2048 / 16
		t.Errorf("HeadDim fallback = %d, want 128", ai.arch.HeadDim)
	}
}

func TestParseOllamaArchUnknown(t *testing.T) {
	if (parseOllamaArch(nil)).arch.KnownForKV() {
		t.Error("nil model_info should not yield a known arch")
	}
	if (parseOllamaArch(map[string]any{"foo": "bar"})).arch.KnownForKV() {
		t.Error("archless model_info should not yield a known arch")
	}
}

func TestParseOllamaPS(t *testing.T) {
	const ps = `{"models":[{"name":"llama3:70b","model":"llama3:70b","size":41000000000,"size_vram":40000000000,"context_length":8192,"details":{"parameter_size":"70.6B","quantization_level":"Q4_0","family":"llama"}}]}`
	var r psResponse
	if err := json.Unmarshal([]byte(ps), &r); err != nil {
		t.Fatal(err)
	}
	if len(r.Models) != 1 || r.Models[0].SizeVRAM != 40000000000 || r.Models[0].ContextLength != 8192 {
		t.Errorf("ps parse wrong: %+v", r.Models)
	}
}

func TestParseLlamaProps(t *testing.T) {
	var props propsResponse
	props.ModelPath = "/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
	props.DefaultGenerationSettings.NCtx = 8192
	ms := parseLlamaProps(props)
	if len(ms) != 1 {
		t.Fatalf("want 1 model, got %d", len(ms))
	}
	if ms[0].Name != "Meta-Llama-3-8B-Instruct.Q4_K_M" {
		t.Errorf("name = %q", ms[0].Name)
	}
	if ms[0].ContextTokens != 8192 {
		t.Errorf("ctx = %d", ms[0].ContextTokens)
	}
}

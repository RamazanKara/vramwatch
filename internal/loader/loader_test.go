package loader

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
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

func TestOllamaModelsOverHTTP(t *testing.T) {
	const psJSON = `{"models":[{"name":"llama3:8b","model":"llama3:8b","size":6000000000,"size_vram":5800000000,"context_length":0,"details":{"parameter_size":"8B","quantization_level":"Q5_K_M","family":"llama"}}]}`

	var showCalls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"version":"0.5.0"}`))
	})
	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(psJSON))
	})
	mux.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&showCalls, 1)
		w.Write([]byte(`{"model_info":` + llama3ModelInfo + `}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	o := NewOllama(srv.URL)
	ctx := context.Background()
	if !o.Available(ctx) {
		t.Fatal("Ollama should report available against the test server")
	}

	ms, err := o.Models(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 1 {
		t.Fatalf("want 1 model, got %d", len(ms))
	}
	m := ms[0]
	if m.Name != "llama3:8b" || m.VRAMBytes != 5800000000 {
		t.Errorf("model basics wrong: %+v", m)
	}
	want := model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16}
	if m.Arch != want {
		t.Errorf("arch = %+v, want %+v", m.Arch, want)
	}
	if m.ContextMax != 8192 {
		t.Errorf("ContextMax = %d, want 8192", m.ContextMax)
	}
	// /api/ps reported context_length 0, so we fall back to min(4096, ctxMax).
	if m.ContextTokens != 4096 {
		t.Errorf("ContextTokens fallback = %d, want 4096", m.ContextTokens)
	}
	// The engine can now compute a real KV split for this model.
	if engine.KVCacheBytes(m.Arch, m.ContextTokens) == 0 {
		t.Error("expected a non-zero KV estimate")
	}

	// Architecture is cached: a second Models() call must not re-hit /api/show.
	if _, err := o.Models(ctx); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt32(&showCalls); got != 1 {
		t.Errorf("/api/show called %d times, want 1 (arch should be cached)", got)
	}
}

func TestLlamaCppOverHTTP(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"default_generation_settings":{"n_ctx":8192},"model_path":"/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	l := NewLlamaCpp(srv.URL)
	ctx := context.Background()
	if !l.Available(ctx) {
		t.Fatal("llama.cpp should report available")
	}
	ms, err := l.Models(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 1 || ms[0].Name != "Meta-Llama-3-8B-Instruct.Q4_K_M" || ms[0].ContextTokens != 8192 {
		t.Errorf("llama.cpp model wrong: %+v", ms)
	}
}

// writeMinimalGGUF writes a tiny valid GGUF (llama arch) for the llama.cpp test.
func writeMinimalGGUF(t *testing.T) string {
	t.Helper()
	var b bytes.Buffer
	wu32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	wu64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	wstr := func(s string) { wu64(uint64(len(s))); b.WriteString(s) }
	kvU32 := func(k string, v uint32) { wstr(k); wu32(4); wu32(v) }
	kvStr := func(k, v string) { wstr(k); wu32(8); wstr(v) }

	b.WriteString("GGUF")
	wu32(3) // version
	wu64(0) // tensor_count
	wu64(6) // metadata_kv_count
	kvStr("general.architecture", "llama")
	kvU32("llama.block_count", 32)
	kvU32("llama.attention.head_count", 32)
	kvU32("llama.attention.head_count_kv", 8)
	kvU32("llama.embedding_length", 4096)
	kvU32("llama.context_length", 8192)

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLlamaCppWithGGUF(t *testing.T) {
	ggufPath := writeMinimalGGUF(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		// JSON-encode the model path so backslashes (Windows temp dirs) are escaped.
		mp, _ := json.Marshal(ggufPath)
		w.Write([]byte(`{"default_generation_settings":{"n_ctx":4096},"model_path":` + string(mp) + `}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	ms, err := NewLlamaCpp(srv.URL).Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 1 {
		t.Fatalf("want 1 model, got %d", len(ms))
	}
	m := ms[0]
	if !m.Arch.KnownForKV() || m.Arch.Layers != 32 || m.Arch.KVHeads != 8 || m.Arch.HeadDim != 128 {
		t.Errorf("GGUF arch not applied: %+v", m.Arch)
	}
	if m.WeightsBytes == 0 {
		t.Error("WeightsBytes should be the GGUF file size")
	}
	if m.ContextTokens != 4096 { // from /props
		t.Errorf("ContextTokens = %d, want 4096", m.ContextTokens)
	}
	if m.ContextMax != 8192 { // from GGUF
		t.Errorf("ContextMax = %d, want 8192", m.ContextMax)
	}
	// The engine can now compute a real KV cache for a llama.cpp model.
	if engine.KVCacheBytes(m.Arch, m.ContextTokens) == 0 {
		t.Error("expected a KV estimate for the GGUF-backed llama.cpp model")
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

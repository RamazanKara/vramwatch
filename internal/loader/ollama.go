package loader

import (
	"context"
	"path"
	"strings"
	"sync"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// Ollama queries an Ollama server. It is the first-class loader: /api/ps gives
// resident models and VRAM, and /api/show gives the architecture used to
// estimate the KV cache split.
type Ollama struct {
	Base string

	mu    sync.Mutex
	cache map[string]archInfo // model name -> arch (avoids refetching every tick)
}

type archInfo struct {
	arch   model.Arch
	ctxMax int
}

// NewOllama builds an Ollama provider. An empty base uses OLLAMA_HOST or the
// default 127.0.0.1:11434.
func NewOllama(base string) *Ollama {
	return &Ollama{Base: normalizeBase(base, "OLLAMA_HOST", "http://127.0.0.1:11434"), cache: map[string]archInfo{}}
}

func (o *Ollama) Name() string { return "ollama" }

func (o *Ollama) Available(ctx context.Context) bool {
	return probe(ctx, o.Base+"/api/version")
}

// psResponse is the subset of /api/ps we consume.
type psResponse struct {
	Models []psModel `json:"models"`
}

type psModel struct {
	Name          string `json:"name"`
	Model         string `json:"model"`
	Size          uint64 `json:"size"`
	SizeVRAM      uint64 `json:"size_vram"`
	ContextLength int    `json:"context_length"`
	Details       struct {
		ParameterSize     string `json:"parameter_size"`
		QuantizationLevel string `json:"quantization_level"`
		Family            string `json:"family"`
	} `json:"details"`
}

// showResponse is the subset of /api/show we consume.
type showResponse struct {
	ModelInfo map[string]any `json:"model_info"`
}

func (o *Ollama) Models(ctx context.Context) ([]model.LoaderModel, error) {
	var ps psResponse
	if err := getJSON(ctx, o.Base+"/api/ps", &ps); err != nil {
		return nil, err
	}
	out := make([]model.LoaderModel, 0, len(ps.Models))
	for _, m := range ps.Models {
		lm := model.LoaderModel{
			Loader:        "ollama",
			Name:          m.Name,
			GPUIndex:      -1,
			VRAMBytes:     m.SizeVRAM,
			ContextTokens: m.ContextLength,
			Estimated:     true,
		}
		ai := o.arch(ctx, m.Name)
		lm.Arch = ai.arch
		lm.ContextMax = ai.ctxMax
		if lm.ContextTokens == 0 && ai.ctxMax > 0 {
			// /api/ps didn't report the running context; assume the model's
			// configured default (Ollama's default is 4096) capped at trained max.
			lm.ContextTokens = min(4096, ai.ctxMax)
		}
		out = append(out, lm)
	}
	return out, nil
}

// arch returns cached architecture for a model, fetching /api/show on a miss.
func (o *Ollama) arch(ctx context.Context, name string) archInfo {
	o.mu.Lock()
	if ai, ok := o.cache[name]; ok {
		o.mu.Unlock()
		return ai
	}
	o.mu.Unlock()

	var show showResponse
	if err := postJSON(ctx, o.Base+"/api/show", map[string]string{"model": name}, &show); err != nil {
		return archInfo{}
	}
	ai := parseOllamaArch(show.ModelInfo)

	o.mu.Lock()
	o.cache[name] = ai
	o.mu.Unlock()
	return ai
}

// parseOllamaArch extracts KV-relevant architecture from an Ollama model_info
// map. Values are JSON numbers (float64) or strings.
func parseOllamaArch(info map[string]any) archInfo {
	if info == nil {
		return archInfo{}
	}
	archName, _ := info["general.architecture"].(string)
	if archName == "" {
		return archInfo{}
	}
	p := archName + "."
	layers := numAt(info, p+"block_count")
	headCount := numAt(info, p+"attention.head_count")
	kvHeads := numAt(info, p+"attention.head_count_kv")
	if kvHeads == 0 {
		kvHeads = headCount // multi-head attention: kv heads == query heads
	}
	keyLen := numAt(info, p+"attention.key_length")
	embLen := numAt(info, p+"embedding_length")
	ctxMax := numAt(info, p+"context_length")

	headDim := keyLen
	if headDim == 0 && headCount > 0 {
		headDim = embLen / headCount
	}

	return archInfo{
		arch: model.Arch{
			Name:       archName,
			Layers:     layers,
			KVHeads:    kvHeads,
			HeadDim:    headDim,
			KVTypeBits: 16, // Ollama does not expose the KV cache type; assume f16
		},
		ctxMax: ctxMax,
	}
}

// numAt reads a numeric model_info value, tolerating float64 or numeric string.
func numAt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		return atoi(strings.TrimSpace(v))
	}
	return 0
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// baseName is used by the llama.cpp provider but lives here to share path use.
// baseName returns the file name of a path, handling both Windows ("\") and
// POSIX ("/") separators, since a llama.cpp server on Windows reports Windows
// paths that path.Base alone would not split.
func baseName(p string) string { return path.Base(strings.ReplaceAll(p, `\`, "/")) }

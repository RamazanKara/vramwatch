package loader

import (
	"context"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// LlamaCpp queries a llama.cpp server. Support is best-effort: /props exposes
// the context length and model path but not VRAM or architecture, so vramwatch
// reports the model's presence and context without a weights/KV split.
type LlamaCpp struct {
	Base string
}

// NewLlamaCpp builds a llama.cpp provider. An empty base uses LLAMACPP_HOST or
// the default 127.0.0.1:8080.
func NewLlamaCpp(base string) *LlamaCpp {
	return &LlamaCpp{Base: normalizeBase(base, "LLAMACPP_HOST", "http://127.0.0.1:8080")}
}

func (l *LlamaCpp) Name() string { return "llama.cpp" }

func (l *LlamaCpp) Available(ctx context.Context) bool {
	return probe(ctx, l.Base+"/health")
}

type propsResponse struct {
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
	ModelPath string `json:"model_path"`
}

func (l *LlamaCpp) Models(ctx context.Context) ([]model.LoaderModel, error) {
	var props propsResponse
	if err := getJSON(ctx, l.Base+"/props", &props); err != nil {
		return nil, err
	}
	return parseLlamaProps(props), nil
}

func parseLlamaProps(props propsResponse) []model.LoaderModel {
	name := baseName(props.ModelPath)
	name = strings.TrimSuffix(name, ".gguf")
	if name == "" || name == "." {
		name = "llama.cpp model"
	}
	return []model.LoaderModel{{
		Loader:        "llama.cpp",
		Name:          name,
		GPUIndex:      -1,
		ContextTokens: props.DefaultGenerationSettings.NCtx,
		Estimated:     true,
	}}
}

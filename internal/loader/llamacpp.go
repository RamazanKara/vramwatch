package loader

import (
	"context"
	"net/url"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/gguf"
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
	models := parseLlamaProps(props)
	// Enrich with GGUF metadata. This is what gives llama.cpp a real weights/KV
	// split, since /props exposes neither VRAM nor architecture. Only attempt the
	// read when the server is on THIS host: props.ModelPath is a path as seen by
	// the server, so reading it locally for a remote server could open a
	// different (wrong) file entirely.
	if props.ModelPath != "" && isLocalURL(l.Base) {
		if info, err := gguf.Read(props.ModelPath); err == nil {
			for i := range models {
				models[i].ArtifactPath = props.ModelPath
				models[i].WeightsBytes = info.FileSize // ≈ weights (assumes full GPU offload)
				models[i].Quantization = info.Quantization()
				if info.ContextLength > 0 {
					models[i].ContextMax = info.ContextLength
				}
				if info.Architecture != "" {
					models[i].Arch = info.ToArch()
				}
			}
		}
	}
	return models, nil
}

// isLocalURL reports whether base points at the loopback host, so it's safe to
// read a server-supplied file path from the local filesystem.
func isLocalURL(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "localhost" || host == "::1" || strings.HasPrefix(host, "127.")
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
		VRAMSource:    model.ProvenanceEstimated,
	}}
}

package loader

import (
	"encoding/json"
	"testing"
)

func FuzzOllamaMetadata(f *testing.F) {
	f.Add(llama3ModelInfo, "FROM /models/fixture.gguf\n")
	f.Add(`{"general.architecture":"llama","llama.block_count":"32"}`, "FROM llama3:8b")
	f.Add("null", "")
	f.Fuzz(func(t *testing.T, metadata, modelfile string) {
		var info map[string]any
		if json.Unmarshal([]byte(metadata), &info) == nil {
			parseOllamaArch(info)
		}
		parseModelfileFrom(modelfile)
	})
}

func FuzzLlamaProps(f *testing.F) {
	f.Add(`{"model_path":"C:\\models\\fixture.gguf","default_generation_settings":{"n_ctx":4096}}`)
	f.Add(`{}`)
	f.Add("null")
	f.Fuzz(func(t *testing.T, data string) {
		var props propsResponse
		if json.Unmarshal([]byte(data), &props) != nil {
			return
		}
		models := parseLlamaProps(props)
		if len(models) != 1 || models[0].Name == "" {
			t.Fatalf("invalid model: %+v", models)
		}
	})
}

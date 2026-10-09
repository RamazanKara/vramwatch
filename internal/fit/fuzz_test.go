package fit

import "testing"

func FuzzArtifactReferences(f *testing.F) {
	f.Add("model-Q4_K_M-00001-of-00001.gguf", "q4_k_m")
	f.Add("llama3:8b-q4_K_M", "fp16")
	f.Add("", "")
	f.Fuzz(func(t *testing.T, name, quant string) {
		_, part, total, sharded := hfLogicalKey(name)
		group, err := selectHFGroup([]hfFile{{Name: name, Size: 1}})
		if err == nil && (len(group) != 1 || (sharded && (part != 1 || total != 1))) {
			t.Fatalf("accepted incomplete shard set: %q", name)
		}
		_, tag := splitOllama(name)
		rewriteOllamaQuant(tag, quant)
		q := quantFromName(name)
		if normalizeQuant(q) != q {
			t.Fatalf("quantization is not canonical: %q", q)
		}
	})
}

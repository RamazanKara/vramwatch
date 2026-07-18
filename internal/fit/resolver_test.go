package fit

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func fitGGUF(fileType uint32) []byte {
	var body bytes.Buffer
	putStringKV := func(key, value string) {
		putString(&body, key)
		putU32(&body, 8)
		putString(&body, value)
	}
	putUintKV := func(key string, value uint32) {
		putString(&body, key)
		putU32(&body, 4)
		putU32(&body, value)
	}
	putStringKV("general.name", "Fixture")
	putStringKV("general.architecture", "llama")
	putUintKV("general.file_type", fileType)
	putUintKV("llama.block_count", 32)
	putUintKV("llama.attention.head_count", 32)
	putUintKV("llama.attention.head_count_kv", 8)
	putUintKV("llama.attention.key_length", 128)
	putUintKV("llama.attention.value_length", 64)
	putUintKV("llama.context_length", 131072)

	var out bytes.Buffer
	out.WriteString("GGUF")
	putU32(&out, 3)
	putU64(&out, 0)
	putU64(&out, 9)
	out.Write(body.Bytes())
	return out.Bytes()
}

func putU32(w io.Writer, v uint32) { _ = binary.Write(w, binary.LittleEndian, v) }
func putU64(w io.Writer, v uint64) { _ = binary.Write(w, binary.LittleEndian, v) }
func putString(w io.Writer, s string) {
	putU64(w, uint64(len(s)))
	_, _ = io.WriteString(w, s)
}

func response(status int, body string, headers map[string]string) *http.Response {
	h := make(http.Header)
	for k, v := range headers {
		h.Set(k, v)
	}
	return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body))}
}

func TestResolveLocalGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	if err := os.WriteFile(path, fitGGUF(15), 0o644); err != nil {
		t.Fatal(err)
	}
	a, err := Resolve(context.Background(), path, ResolveOptions{Quant: "q4_k_m"})
	if err != nil {
		t.Fatal(err)
	}
	if a.Source != SourceLocal || a.Quantization != "Q4_K_M" || a.WeightBytes == 0 {
		t.Fatalf("artifact = %+v", a)
	}
	if a.Arch.ValueDim != 64 || a.ContextMax != 131072 || a.MetadataBytes != 0 {
		t.Fatalf("metadata = %+v", a)
	}
	if _, err := Resolve(context.Background(), path, ResolveOptions{Quant: "q8_0"}); err == nil {
		t.Error("quant mismatch should fail")
	}
}

func TestResolveMissingLocalGGUFFailsBeforeRegistryLookup(t *testing.T) {
	for _, ref := range []string{"missing.gguf", "./missing/model.gguf"} {
		_, err := Resolve(context.Background(), ref, ResolveOptions{})
		if err == nil || !strings.Contains(err.Error(), "local GGUF") {
			t.Errorf("Resolve(%q) error = %v", ref, err)
		}
	}
}

func TestResolveHuggingFaceShardsUsesOnlyMetadata(t *testing.T) {
	prefix := fitGGUF(15)
	const first = uint64(3 * model.GiB)
	const second = uint64(2 * model.GiB)
	apiJSON := fmt.Sprintf(`{"id":"owner/repo","sha":"abc123","siblings":[{"rfilename":"model-Q4_K_M-00001-of-00002.gguf","size":%d},{"rfilename":"model-Q4_K_M-00002-of-00002.gguf","size":%d},{"rfilename":"README.md","size":12}]}`, first, second)
	var requests []*http.Request
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requests = append(requests, req.Clone(req.Context()))
		switch {
		case strings.Contains(req.URL.Path, "/api/models/"):
			return response(http.StatusOK, apiJSON, nil), nil
		case strings.Contains(req.URL.Path, "/resolve/"):
			return response(http.StatusPartialContent, string(prefix), map[string]string{"Content-Range": fmt.Sprintf("bytes 0-%d/%d", len(prefix)-1, first)}), nil
		default:
			return nil, fmt.Errorf("unexpected URL %s", req.URL)
		}
	})}
	a, err := Resolve(context.Background(), "hf:owner/repo", ResolveOptions{Quant: "q4_k_m", Revision: "experiment", HFToken: "secret", Client: client})
	if err != nil {
		t.Fatal(err)
	}
	if a.WeightBytes != first+second || a.Digest != "abc123" || a.Revision != "abc123" {
		t.Fatalf("artifact identity/size = %+v", a)
	}
	if a.Arch.ValueDim != 64 || a.ContextMax != 131072 || a.MetadataBytes <= 0 || a.MetadataBytes >= int64(first) {
		t.Fatalf("metadata-only resolution failed: %+v", a)
	}
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want API + one range request", len(requests))
	}
	if !strings.Contains(requests[0].URL.Path, "/revision/experiment") {
		t.Errorf("revision API was not used: %s", requests[0].URL)
	}
	if requests[0].Header.Get("Authorization") != "Bearer secret" || requests[1].Header.Get("Authorization") != "Bearer secret" {
		t.Error("HF token was not applied to metadata requests")
	}
	if !strings.HasPrefix(requests[1].Header.Get("Range"), "bytes=0-") {
		t.Errorf("model request was not ranged: %q", requests[1].Header.Get("Range"))
	}
}

func TestResolveURLRequiresFullArtifactSize(t *testing.T) {
	prefix := string(fitGGUF(15))
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return response(http.StatusPartialContent, prefix, nil), nil
	})}
	_, err := Resolve(context.Background(), "https://example.test/model-Q4_K_M.gguf", ResolveOptions{Client: client})
	if err == nil || !strings.Contains(err.Error(), "did not report the model size") {
		t.Fatalf("missing Content-Range size error = %v", err)
	}
}

func TestResolveURLRejectsImpossibleArtifactSize(t *testing.T) {
	prefix := string(fitGGUF(15))
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return response(http.StatusPartialContent, prefix, map[string]string{"Content-Range": fmt.Sprintf("bytes 0-%d/1", len(prefix)-1)}), nil
	})}
	_, err := Resolve(context.Background(), "https://example.test/model-Q4_K_M.gguf", ResolveOptions{Client: client})
	if err == nil || !strings.Contains(err.Error(), "smaller than") {
		t.Fatalf("impossible size error = %v", err)
	}
}

func TestResolveURLStopsAfterArchitectureMetadata(t *testing.T) {
	prefix := fitGGUF(15)
	body := string(prefix) + strings.Repeat("tensor-data", 200000)
	const total = uint64(5 * model.GiB)
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return response(http.StatusPartialContent, body, map[string]string{
			"Content-Range": fmt.Sprintf("bytes 0-%d/%d", len(body)-1, total),
		}), nil
	})}
	a, err := Resolve(context.Background(), "https://example.test/model.gguf", ResolveOptions{Client: client})
	if err != nil {
		t.Fatal(err)
	}
	if a.WeightBytes != total || a.Quantization != "Q4_K_M" {
		t.Fatalf("resolved artifact = %+v", a)
	}
	if a.MetadataBytes >= int64(len(body))/10 {
		t.Fatalf("resolver read %d bytes from a %d-byte ranged body after metadata was complete", a.MetadataBytes, len(body))
	}
}

func TestResolveOllamaManifestAndRangedBlob(t *testing.T) {
	prefix := fitGGUF(15)
	const weight = uint64(5 * model.GiB)
	manifest := fmt.Sprintf(`{"config":{"digest":"sha256:config","size":99},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:model","size":%d}]}`, weight)
	manifestSum := fmt.Sprintf("%x", sha256.Sum256([]byte(manifest)))
	var requests []*http.Request
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requests = append(requests, req.Clone(req.Context()))
		switch {
		case strings.Contains(req.URL.Path, "/manifests/"):
			return response(http.StatusOK, manifest, nil), nil
		case strings.HasSuffix(req.URL.Path, "/sha256:config"):
			return response(http.StatusOK, `{"file_type":"Q4_K_M","model_family":"llama"}`, nil), nil
		case strings.HasSuffix(req.URL.Path, "/sha256:model"):
			return response(http.StatusPartialContent, string(prefix), map[string]string{"Content-Range": fmt.Sprintf("bytes 0-%d/%d", len(prefix)-1, weight)}), nil
		default:
			return nil, fmt.Errorf("unexpected URL %s", req.URL)
		}
	})}
	a, err := Resolve(context.Background(), "ollama:llama3.2:3b-instruct", ResolveOptions{Quant: "q4_k_m", Client: client})
	if err != nil {
		t.Fatal(err)
	}
	if a.Source != SourceOllama || a.CanonicalID != "llama3.2:3b-instruct-q4_K_M" || a.WeightBytes != weight || a.Digest != manifestSum {
		t.Fatalf("Ollama artifact = %+v", a)
	}
	if len(requests) != 3 || !strings.HasPrefix(requests[2].Header.Get("Range"), "bytes=0-") {
		t.Fatalf("Ollama metadata requests = %d, final Range=%q", len(requests), requests[len(requests)-1].Header.Get("Range"))
	}
	if !strings.HasSuffix(requests[0].URL.Path, "/3b-instruct-q4_K_M") {
		t.Fatalf("quant-specific manifest was not tried first: %s", requests[0].URL)
	}
}

func TestResolveRefusesIgnoredLargeRange(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		r := response(http.StatusOK, "", nil)
		r.ContentLength = DefaultMetadataLimit + 1
		return r, nil
	})}
	_, err := Resolve(context.Background(), "https://example.test/model.gguf", ResolveOptions{Client: client})
	if err == nil || !strings.Contains(err.Error(), "ignored Range") {
		t.Fatalf("ignored range error = %v", err)
	}
}

func TestSelectHFGroupValidatesShards(t *testing.T) {
	if _, err := selectHFGroup([]hfFile{{Name: "m-Q4_K_M-00001-of-00002.gguf", Size: 1}}); err == nil {
		t.Error("incomplete shard set should fail")
	}
	group, err := selectHFGroup([]hfFile{{Name: "m-Q4_K_M-00002-of-00002.gguf", Size: 2}, {Name: "m-Q4_K_M-00001-of-00002.gguf", Size: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(group[0].Name, "00001") || !strings.Contains(group[1].Name, "00002") {
		t.Fatalf("shards not sorted: %+v", group)
	}
	if _, err := selectHFGroup([]hfFile{{Name: "a-Q4_K_M.gguf"}, {Name: "b-Q4_K_M.gguf"}}); err == nil {
		t.Error("ambiguous files should fail")
	}
}

func TestQuantFromNameCoversCommonFamilies(t *testing.T) {
	cases := map[string]string{
		"model-Q4_K_M.gguf": "Q4_K_M", "model-Q2_K.gguf": "Q2_K", "model-IQ3_XXS.gguf": "IQ3_XXS", "model-BF16.gguf": "BF16",
	}
	for name, want := range cases {
		if got := quantFromName(name); got != want {
			t.Errorf("quantFromName(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestRewriteOllamaQuantUsesRegistryTagCasing(t *testing.T) {
	cases := map[string]string{
		"q4_k_m": "3b-instruct-q4_K_M",
		"q8_0":   "3b-instruct-q8_0",
		"f16":    "3b-instruct-fp16",
	}
	for quant, want := range cases {
		if got := rewriteOllamaQuant("3b-instruct", quant); got != want {
			t.Errorf("rewriteOllamaQuant(%q) = %q, want %q", quant, got, want)
		}
	}
	if got := rewriteOllamaQuant("3b-instruct-q4_K_S", "q4_k_m"); got != "3b-instruct-q4_K_M" {
		t.Errorf("replace existing quant = %q", got)
	}
}

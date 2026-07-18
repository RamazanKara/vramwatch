package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// --- minimal GGUF encoder (test-only) ---

func wU32(b *bytes.Buffer, v uint32) { binary.Write(b, binary.LittleEndian, v) }
func wU64(b *bytes.Buffer, v uint64) { binary.Write(b, binary.LittleEndian, v) }
func wStr(b *bytes.Buffer, s string) {
	wU64(b, uint64(len(s)))
	b.WriteString(s)
}
func kvU32(b *bytes.Buffer, key string, v uint32) {
	wStr(b, key)
	wU32(b, 4) // type: uint32
	wU32(b, v)
}
func kvStr(b *bytes.Buffer, key, val string) {
	wStr(b, key)
	wU32(b, 8) // type: string
	wStr(b, val)
}
func kvStrArray(b *bytes.Buffer, key string, vals []string) {
	wStr(b, key)
	wU32(b, 9) // type: array
	wU32(b, 8) // elem type: string
	wU64(b, uint64(len(vals)))
	for _, v := range vals {
		wStr(b, v)
	}
}

func writeGGUF(t *testing.T, kvs func(*bytes.Buffer) int) string {
	t.Helper()
	var body bytes.Buffer
	n := kvs(&body)

	var hdr bytes.Buffer
	hdr.WriteString("GGUF")
	wU32(&hdr, 3)         // version
	wU64(&hdr, 0)         // tensor_count
	wU64(&hdr, uint64(n)) // metadata_kv_count
	hdr.Write(body.Bytes())

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, hdr.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadGGUF(t *testing.T) {
	path := writeGGUF(t, func(b *bytes.Buffer) int {
		kvStr(b, "general.architecture", "llama")
		kvU32(b, "llama.block_count", 32)
		kvU32(b, "llama.attention.head_count", 32)
		kvU32(b, "llama.attention.head_count_kv", 8)
		kvU32(b, "llama.attention.key_length", 128)
		kvU32(b, "llama.embedding_length", 4096)
		kvU32(b, "llama.context_length", 8192)
		kvStrArray(b, "tokenizer.ggml.tokens", []string{"a", "b", "c"}) // must be skipped
		return 8
	})

	info, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Architecture != "llama" || info.Layers != 32 || info.HeadCountKV != 8 {
		t.Errorf("bad metadata: %+v", info)
	}
	if info.HeadDim() != 128 {
		t.Errorf("HeadDim = %d, want 128", info.HeadDim())
	}
	if info.ContextLength != 8192 {
		t.Errorf("ContextLength = %d", info.ContextLength)
	}
	if info.FileSize == 0 {
		t.Error("FileSize should be set")
	}
	arch := info.ToArch()
	want := model.Arch{Name: "llama", Layers: 32, KVHeads: 8, HeadDim: 128, ValueDim: 128, KVTypeBits: 16}
	if arch != want {
		t.Errorf("ToArch = %+v, want %+v", arch, want)
	}
}

func TestQuantizationUsesLlamaFileType(t *testing.T) {
	cases := map[int]string{0: "F32", 7: "Q8_0", 8: "Q5_0", 15: "Q4_K_M", 32: "BF16", 41: "Q2_0", 999: ""}
	for fileType, want := range cases {
		if got := (Info{FileType: fileType, FileTypeKnown: true}).Quantization(); got != want {
			t.Errorf("file type %d = %q, want %q", fileType, got, want)
		}
	}
	if got := (Info{}).Quantization(); got != "" {
		t.Errorf("missing file type = %q, want unknown", got)
	}
}

func TestReadGGUFHeadCountKVFallback(t *testing.T) {
	path := writeGGUF(t, func(b *bytes.Buffer) int {
		kvStr(b, "general.architecture", "gpt2")
		kvU32(b, "gpt2.block_count", 12)
		kvU32(b, "gpt2.attention.head_count", 12) // no head_count_kv (MHA)
		kvU32(b, "gpt2.embedding_length", 768)
		return 4
	})
	info, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.HeadCountKV != 12 {
		t.Errorf("HeadCountKV fallback = %d, want 12", info.HeadCountKV)
	}
	if info.HeadDim() != 64 { // 768 / 12
		t.Errorf("HeadDim = %d, want 64", info.HeadDim())
	}
}

func TestReadPrefixWaitsForLateKVMetadata(t *testing.T) {
	var body bytes.Buffer
	kvStr(&body, "general.architecture", "llama")
	kvU32(&body, "general.file_type", 15)
	kvU32(&body, "llama.block_count", 28)
	kvU32(&body, "llama.attention.head_count", 24)
	kvU32(&body, "llama.attention.key_length", 128)
	kvU32(&body, "llama.context_length", 131072) // core fields arrive first
	kvU32(&body, "llama.attention.head_count_kv", 8)
	kvU32(&body, "llama.attention.value_length", 64)
	kvStrArray(&body, "tokenizer.ggml.tokens", []string{"large", "array"})
	var hdr bytes.Buffer
	hdr.WriteString("GGUF")
	wU32(&hdr, 3)
	wU64(&hdr, 0)
	wU64(&hdr, 9)
	hdr.Write(body.Bytes())

	info, err := ReadPrefix(hdr.Bytes(), 2*model.GiB)
	if err != nil {
		t.Fatal(err)
	}
	if info.HeadCountKV != 8 || info.ValueLength != 64 {
		t.Fatalf("late KV metadata was lost: %+v", info)
	}
}

func TestReadPrefixStopsBeforeTokenizerArrayForMHA(t *testing.T) {
	var body bytes.Buffer
	kvStr(&body, "general.architecture", "gpt2")
	kvU32(&body, "gpt2.block_count", 12)
	kvU32(&body, "gpt2.attention.head_count", 12)
	kvU32(&body, "gpt2.embedding_length", 768)
	// Deliberately omit the array payload. A bounded remote parser should stop at
	// this boundary and use the conservative MHA/value-dimension fallbacks.
	wStr(&body, "tokenizer.ggml.tokens")
	wU32(&body, 9)
	var hdr bytes.Buffer
	hdr.WriteString("GGUF")
	wU32(&hdr, 3)
	wU64(&hdr, 0)
	wU64(&hdr, 5)
	hdr.Write(body.Bytes())

	info, err := ReadPrefix(hdr.Bytes(), model.GiB)
	if err != nil {
		t.Fatal(err)
	}
	if info.HeadCountKV != 12 || info.HeadDim() != 64 || info.ValueLength != 0 {
		t.Fatalf("MHA fallback = %+v", info)
	}
}

// validGGUFBytes returns the bytes of a small valid GGUF, for fuzz seeding.
func validGGUFBytes() []byte {
	var body bytes.Buffer
	kvStr(&body, "general.architecture", "llama")
	kvU32(&body, "llama.block_count", 32)
	kvU32(&body, "llama.attention.head_count", 32)
	var hdr bytes.Buffer
	hdr.WriteString("GGUF")
	wU32(&hdr, 3)
	wU64(&hdr, 0)
	wU64(&hdr, 3)
	hdr.Write(body.Bytes())
	return hdr.Bytes()
}

// FuzzReadGGUF asserts Read never panics on arbitrary (truncated/hostile) input.
func FuzzReadGGUF(f *testing.F) {
	f.Add(validGGUFBytes())
	f.Add([]byte("GGUF"))
	f.Add([]byte("GGUF\x03\x00\x00\x00"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		p := filepath.Join(t.TempDir(), "f.gguf")
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Skip()
		}
		_, _ = Read(p) // must return, never panic
	})
}

func TestReadGGUFRejectsDeepNesting(t *testing.T) {
	var body bytes.Buffer
	wStr(&body, "a")
	wU32(&body, 9) // KV value type: array
	for i := 0; i < 40; i++ {
		wU32(&body, 9) // elem type: array (each block descends one level)
		wU64(&body, 1) // count: 1
	}
	wU32(&body, 4) // terminal: empty uint32 array (never reached)
	wU64(&body, 0)

	var hdr bytes.Buffer
	hdr.WriteString("GGUF")
	wU32(&hdr, 3)
	wU64(&hdr, 0)
	wU64(&hdr, 1)
	hdr.Write(body.Bytes())

	path := filepath.Join(t.TempDir(), "deep.gguf")
	if err := os.WriteFile(path, hdr.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(path); err == nil {
		t.Error("expected an error for deeply nested arrays (stack-overflow guard)")
	}
}

func TestReadGGUFRejectsNonGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notgguf.bin")
	os.WriteFile(path, []byte("NOPExxxxxxxx"), 0o644)
	if _, err := Read(path); err == nil {
		t.Error("expected an error for a non-GGUF file")
	}
	if _, err := Read(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("expected an error for a missing file")
	}
}

// Package gguf reads the metadata header of a GGUF model file (the format used
// by llama.cpp and Ollama) to recover the architecture vramwatch needs for the
// weights/KV split. Only the header is read; tensor data is never touched.
package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

const magic = "GGUF"

// Limits guarding against corrupt/hostile headers.
const (
	maxKVCount     = 1 << 20 // metadata key/value pairs
	maxStringLen   = 1 << 26 // 64 MiB per string
	maxArrayCount  = 1 << 30
	maxArrayDepth  = 16 // real GGUF metadata is never deeply nested
	maxLocalHeader = 256 * model.MiB
)

// Info is the subset of GGUF metadata vramwatch consumes, plus the file size.
type Info struct {
	Name          string
	Architecture  string
	Layers        int
	HeadCount     int
	HeadCountKV   int
	KeyLength     int
	ValueLength   int
	EmbeddingLen  int
	ContextLength int
	FileType      int
	FileTypeKnown bool
	FileSize      uint64
}

// HeadDim returns the per-head dimension, preferring an explicit key_length and
// falling back to embedding_length / head_count.
func (i Info) HeadDim() int {
	if i.KeyLength > 0 {
		return i.KeyLength
	}
	if i.HeadCount > 0 {
		return i.EmbeddingLen / i.HeadCount
	}
	return 0
}

// ToArch converts the GGUF metadata into a model.Arch (f16 KV cache assumed).
func (i Info) ToArch() model.Arch {
	valueDim := i.ValueLength
	if valueDim == 0 {
		valueDim = i.HeadDim()
	}
	return model.Arch{
		Name:       i.Architecture,
		Layers:     i.Layers,
		KVHeads:    i.HeadCountKV,
		HeadDim:    i.HeadDim(),
		ValueDim:   valueDim,
		KVTypeBits: 16,
	}
}

// Read parses the metadata header of the GGUF file at path.
func Read(path string) (Info, error) {
	f, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return Info{}, err
	}
	var src io.Reader = f
	if st.Size() > int64(maxLocalHeader) {
		src = io.LimitReader(f, int64(maxLocalHeader))
	}
	info, err := readFrom(src, uint64(st.Size()), false)
	if err != nil && st.Size() > int64(maxLocalHeader) && (err == io.EOF || err == io.ErrUnexpectedEOF) {
		return info, fmt.Errorf("gguf: metadata header exceeds %s limit", model.HumanBytes(maxLocalHeader))
	}
	return info, err
}

// ReadPrefix parses architecture metadata from a bounded prefix fetched with
// HTTP Range. It stops as soon as the fields needed for fit prediction are
// known and never requires tensor data or tokenizer arrays.
func ReadPrefix(data []byte, fileSize uint64) (Info, error) {
	return readFrom(bytes.NewReader(data), fileSize, true)
}

func readFrom(src io.Reader, fileSize uint64, stopWhenReady bool) (Info, error) {
	info := Info{FileSize: fileSize}
	r := bufio.NewReaderSize(src, 1<<16)
	var mg [4]byte
	if _, err := io.ReadFull(r, mg[:]); err != nil {
		return info, err
	}
	if string(mg[:]) != magic {
		return info, fmt.Errorf("gguf: not a GGUF file (bad magic)")
	}
	version, err := readU32(r)
	if err != nil {
		return info, err
	}
	if version < 2 || version > 3 {
		return info, fmt.Errorf("gguf: unsupported version %d", version)
	}
	if _, err := readU64(r); err != nil { // tensor_count (unused)
		return info, err
	}
	kvCount, err := readU64(r)
	if err != nil {
		return info, err
	}
	if kvCount > maxKVCount {
		return info, fmt.Errorf("gguf: implausible metadata count %d", kvCount)
	}

	strs := map[string]string{}
	nums := map[string]int64{}
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r)
		if err != nil {
			return info, err
		}
		vtype, err := readU32(r)
		if err != nil {
			return info, err
		}
		// Architecture metadata is normally followed by tokenizer arrays. When
		// optional KV fields are absent (for example MHA omits head_count_kv), stop
		// at that clear boundary without consuming a potentially multi-megabyte
		// token list. Until the boundary, keep scanning so a late GQA/value field
		// cannot be mistaken for an absent one.
		if stopWhenReady && metadataBoundary(key, vtype, strs, nums) {
			break
		}
		if err := readTopValue(r, key, vtype, strs, nums); err != nil {
			return info, err
		}
		if stopWhenReady && coreMetadataReady(strs, nums, true) {
			break
		}
	}

	info.Name = strs["general.name"]
	info.Architecture = strs["general.architecture"]
	p := info.Architecture + "."
	info.Layers = int(nums[p+"block_count"])
	info.HeadCount = int(nums[p+"attention.head_count"])
	info.HeadCountKV = int(nums[p+"attention.head_count_kv"])
	if info.HeadCountKV == 0 {
		info.HeadCountKV = info.HeadCount // multi-head attention
	}
	info.KeyLength = int(nums[p+"attention.key_length"])
	info.ValueLength = int(nums[p+"attention.value_length"])
	info.EmbeddingLen = int(nums[p+"embedding_length"])
	info.ContextLength = int(nums[p+"context_length"])
	info.FileType = int(nums["general.file_type"])
	info.FileTypeKnown = hasNum(nums, "general.file_type")
	return info, nil
}

func coreMetadataReady(strs map[string]string, nums map[string]int64, requireOptional bool) bool {
	a := strs["general.architecture"]
	if a == "" {
		return false
	}
	p := a + "."
	base := nums[p+"block_count"] > 0 && nums[p+"attention.head_count"] > 0 &&
		(nums[p+"attention.key_length"] > 0 || nums[p+"embedding_length"] > 0)
	if !base || !requireOptional {
		return base
	}
	return hasNum(nums, p+"attention.head_count_kv") &&
		hasNum(nums, p+"attention.value_length") &&
		hasNum(nums, p+"context_length") &&
		hasNum(nums, "general.file_type")
}

func metadataBoundary(key string, vtype uint32, strs map[string]string, nums map[string]int64) bool {
	if !coreMetadataReady(strs, nums, false) {
		return false
	}
	a := strs["general.architecture"]
	if strings.HasPrefix(key, a+".") {
		return false
	}
	return strings.HasPrefix(key, "tokenizer.") || vtype == 9
}

func hasNum(m map[string]int64, key string) bool { _, ok := m[key]; return ok }

// Quantization returns the canonical GGUF quant name for common file types.
func (i Info) Quantization() string {
	if !i.FileTypeKnown {
		return ""
	}
	// general.file_type uses llama_ftype, not ggml_type. Keep this mapping in
	// lockstep with llama.cpp/include/llama.h.
	names := map[int]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
		10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
		16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
		22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
		28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
		37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
	}
	return names[i.FileType]
}

// readTopValue reads a metadata value, storing scalars/strings we care about
// and consuming (skipping) everything else.
func readTopValue(r *bufio.Reader, key string, vtype uint32, strs map[string]string, nums map[string]int64) error {
	switch vtype {
	case 8: // string
		s, err := readString(r)
		if err != nil {
			return err
		}
		strs[key] = s
		return nil
	case 0, 1, 2, 3, 4, 5, 10, 11: // integer types
		n, err := readInt(r, vtype)
		if err != nil {
			return err
		}
		nums[key] = n
		return nil
	case 9: // array
		return skipArray(r, 1)
	default: // float32/float64/bool: fixed-size, just skip
		return skipValue(r, vtype, 0)
	}
}

// readInt reads one integer value of the given GGUF type as int64.
func readInt(r *bufio.Reader, vtype uint32) (int64, error) {
	switch vtype {
	case 0: // uint8
		b, err := r.ReadByte()
		return int64(b), err
	case 1: // int8
		b, err := r.ReadByte()
		return int64(int8(b)), err
	case 2: // uint16
		v, err := readU16(r)
		return int64(v), err
	case 3: // int16
		v, err := readU16(r)
		return int64(int16(v)), err
	case 4: // uint32
		v, err := readU32(r)
		return int64(v), err
	case 5: // int32
		v, err := readU32(r)
		return int64(int32(v)), err
	case 10: // uint64
		v, err := readU64(r)
		return int64(v), err
	case 11: // int64
		v, err := readU64(r)
		return int64(v), err
	}
	return 0, fmt.Errorf("gguf: not an integer type %d", vtype)
}

// skipValue consumes one value of the given type. depth is the array-nesting
// level, bounded to stop a hostile file from overflowing the stack.
func skipValue(r *bufio.Reader, vtype uint32, depth int) error {
	switch vtype {
	case 7: // bool
		_, err := r.Discard(1)
		return err
	case 6: // float32
		_, err := r.Discard(4)
		return err
	case 12: // float64
		_, err := r.Discard(8)
		return err
	case 8: // string
		_, err := readString(r)
		return err
	case 9: // array
		return skipArray(r, depth+1)
	case 0, 1, 2, 3, 4, 5, 10, 11:
		_, err := readInt(r, vtype)
		return err
	}
	return fmt.Errorf("gguf: unknown value type %d", vtype)
}

func skipArray(r *bufio.Reader, depth int) error {
	if depth > maxArrayDepth {
		return fmt.Errorf("gguf: array nesting too deep")
	}
	elemType, err := readU32(r)
	if err != nil {
		return err
	}
	n, err := readU64(r)
	if err != nil {
		return err
	}
	if n > maxArrayCount {
		return fmt.Errorf("gguf: implausible array length %d", n)
	}
	for i := uint64(0); i < n; i++ {
		if err := skipValue(r, elemType, depth); err != nil {
			return err
		}
	}
	return nil
}

func readString(r *bufio.Reader) (string, error) {
	n, err := readU64(r)
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("gguf: string too long (%d bytes)", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readU16(r *bufio.Reader) (uint16, error) {
	var v uint16
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

func readU32(r *bufio.Reader) (uint32, error) {
	var v uint32
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

func readU64(r *bufio.Reader) (uint64, error) {
	var v uint64
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

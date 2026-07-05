// Package gguf reads the metadata header of a GGUF model file (the format used
// by llama.cpp and Ollama) to recover the architecture vramwatch needs for the
// weights/KV split. Only the header is read; tensor data is never touched.
package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/RamazanKara/vramwatch/internal/model"
)

const magic = "GGUF"

// Limits guarding against corrupt/hostile headers.
const (
	maxKVCount    = 1 << 20 // metadata key/value pairs
	maxStringLen  = 1 << 26 // 64 MiB per string
	maxArrayCount = 1 << 30
	maxArrayDepth = 16 // real GGUF metadata is never deeply nested
)

// Info is the subset of GGUF metadata vramwatch consumes, plus the file size.
type Info struct {
	Architecture  string
	Layers        int
	HeadCount     int
	HeadCountKV   int
	KeyLength     int
	EmbeddingLen  int
	ContextLength int
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
	return model.Arch{
		Name:       i.Architecture,
		Layers:     i.Layers,
		KVHeads:    i.HeadCountKV,
		HeadDim:    i.HeadDim(),
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
	info := Info{FileSize: uint64(st.Size())}

	r := bufio.NewReaderSize(f, 1<<16)
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
		if err := readTopValue(r, key, vtype, strs, nums); err != nil {
			return info, err
		}
	}

	info.Architecture = strs["general.architecture"]
	p := info.Architecture + "."
	info.Layers = int(nums[p+"block_count"])
	info.HeadCount = int(nums[p+"attention.head_count"])
	info.HeadCountKV = int(nums[p+"attention.head_count_kv"])
	if info.HeadCountKV == 0 {
		info.HeadCountKV = info.HeadCount // multi-head attention
	}
	info.KeyLength = int(nums[p+"attention.key_length"])
	info.EmbeddingLen = int(nums[p+"embedding_length"])
	info.ContextLength = int(nums[p+"context_length"])
	return info, nil
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

package model

import "testing"

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		in   uint64
		want string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{2 * KiB, "2.0 KiB"},
		{3 * MiB / 2, "1.5 MiB"},
		{24 * GiB, "24.00 GiB"},
		{1073741824, "1.00 GiB"},
	}
	for _, c := range cases {
		if got := HumanBytes(c.in); got != c.want {
			t.Errorf("HumanBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestArchKnownForKV(t *testing.T) {
	full := Arch{Layers: 32, KVHeads: 8, HeadDim: 128, KVTypeBits: 16}
	if !full.KnownForKV() {
		t.Fatal("expected full arch to be known for KV")
	}
	if (Arch{Layers: 32, KVHeads: 8, HeadDim: 128}).KnownForKV() {
		t.Fatal("arch without KVTypeBits must not be known for KV")
	}
}

func TestBreakdownUsedAndSegment(t *testing.T) {
	b := Breakdown{Segments: []Segment{
		{Kind: KindWeights, Bytes: 10 * GiB},
		{Kind: KindKVCache, Bytes: 4 * GiB},
		{Kind: KindFree, Bytes: 10 * GiB},
	}}
	if got := b.Used(); got != 14*GiB {
		t.Errorf("Used() = %d, want %d", got, uint64(14*GiB))
	}
	if s, ok := b.Segment(KindKVCache); !ok || s.Bytes != 4*GiB {
		t.Errorf("Segment(KVCache) = %+v, ok=%v", s, ok)
	}
	if _, ok := b.Segment(KindOtherProcess); ok {
		t.Error("expected no other_process segment")
	}
}

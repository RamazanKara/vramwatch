package main

import "testing"

func FuzzParseByteSize(f *testing.F) {
	for _, seed := range []string{"1.5GiB", "0", "NaN", "18446744073709551616B", ""} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw string) {
		value, err := parseByteSize(raw)
		if err != nil && value != 0 {
			t.Fatalf("invalid size returned %d bytes", value)
		}
	})
}

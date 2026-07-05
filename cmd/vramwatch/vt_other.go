//go:build !windows

package main

// enableVT is a no-op on non-Windows platforms, where ANSI escapes work
// natively.
func enableVT() {}

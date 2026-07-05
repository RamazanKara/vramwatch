//go:build !windows

package gpu

// platformProviders adds OS-specific GPU providers. Only Windows has one today.
func platformProviders() []Provider { return nil }

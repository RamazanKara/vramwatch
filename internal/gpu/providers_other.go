//go:build !windows && !darwin

package gpu

// This platform has no additional provider beyond the vendor command-line tools.
func platformProviders() []Provider { return nil }

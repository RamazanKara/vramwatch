//go:build darwin

package gpu

func platformProviders() []Provider { return []Provider{AppleMetal{}} }

// Package loader queries local inference servers (Ollama, llama.cpp) over HTTP
// to learn which models are resident and, where possible, their architecture —
// the information that lets the engine split VRAM into weights vs KV cache.
package loader

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// Provider reports the models an inference server currently holds in VRAM.
type Provider interface {
	Name() string
	Available(ctx context.Context) bool
	Models(ctx context.Context) ([]model.LoaderModel, error)
}

// All returns the built-in loader providers using their default endpoints.
func All() []Provider {
	return []Provider{NewOllama(""), NewLlamaCpp("")}
}

// DetectAvailable returns the loader providers that answer on this host.
func DetectAvailable(ctx context.Context) []Provider {
	var out []Provider
	for _, p := range All() {
		if p.Available(ctx) {
			out = append(out, p)
		}
	}
	return out
}

// Models runs every available loader and concatenates the resident models.
func Models(ctx context.Context) ([]model.LoaderModel, error) {
	var all []model.LoaderModel
	for _, p := range DetectAvailable(ctx) {
		m, err := p.Models(ctx)
		if err != nil {
			continue
		}
		all = append(all, m...)
	}
	return all, nil
}

// httpClient is shared and short-timeout; loaders are always local.
var httpClient = &http.Client{Timeout: 4 * time.Second}

func getJSON(ctx context.Context, url string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		return &httpError{url: url, code: resp.StatusCode}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func postJSON(ctx context.Context, url string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		return &httpError{url: url, code: resp.StatusCode}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// probe returns true if a short GET to url returns 200.
func probe(ctx context.Context, url string) bool {
	pctx, cancel := context.WithTimeout(ctx, 600*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(pctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

type httpError struct {
	url  string
	code int
}

func (e *httpError) Error() string { return e.url + ": HTTP " + itoa(e.code) }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// normalizeBase trims a trailing slash and applies a default when empty,
// honouring the given environment variable if set.
func normalizeBase(base, envVar, def string) string {
	if base == "" {
		if v := os.Getenv(envVar); v != "" {
			base = v
		} else {
			base = def
		}
	}
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		base = "http://" + base
	}
	return strings.TrimRight(base, "/")
}

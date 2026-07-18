// Package fit resolves local and remote GGUF model metadata and predicts
// accelerator fit without downloading model tensors.
package fit

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/gguf"
	"github.com/RamazanKara/vramwatch/internal/model"
)

const DefaultMetadataLimit int64 = 16 * model.MiB

type Source string

const (
	SourceLocal  Source = "local"
	SourceURL    Source = "url"
	SourceHF     Source = "huggingface"
	SourceOllama Source = "ollama"
)

// Artifact is the immutable metadata required to predict a model's footprint.
type Artifact struct {
	Reference         string     `json:"reference"`
	CanonicalID       string     `json:"canonical_id"`
	Source            Source     `json:"source"`
	Revision          string     `json:"revision,omitempty"`
	Filename          string     `json:"filename,omitempty"`
	Digest            string     `json:"digest,omitempty"`
	Quantization      string     `json:"quantization"`
	WeightBytes       uint64     `json:"weight_bytes"`
	Arch              model.Arch `json:"architecture"`
	ContextMax        int        `json:"context_max,omitempty"`
	MetadataBytes     int64      `json:"metadata_bytes_fetched"`
	WeightBasis       string     `json:"weight_basis"`
	ArchitectureBasis string     `json:"architecture_basis"`
}

type ResolveOptions struct {
	Quant            string
	Revision         string
	File             string
	HFToken          string
	MaxMetadataBytes int64
	Client           *http.Client
}

type resolver struct {
	opts    ResolveOptions
	client  *http.Client
	fetched int64
}

func Resolve(ctx context.Context, ref string, opts ResolveOptions) (Artifact, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return Artifact{}, errors.New("MODEL is required")
	}
	if opts.MaxMetadataBytes <= 0 {
		opts.MaxMetadataBytes = DefaultMetadataLimit
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	r := &resolver{opts: opts, client: client}
	var a Artifact
	var err error
	switch {
	case strings.HasPrefix(ref, "hf:"):
		a, err = r.resolveHF(ctx, strings.TrimPrefix(ref, "hf:"))
	case strings.HasPrefix(ref, "ollama:"):
		a, err = r.resolveOllama(ctx, strings.TrimPrefix(ref, "ollama:"))
	case strings.HasPrefix(ref, "https://"):
		a, err = r.resolveURL(ctx, ref)
	case fileExists(ref):
		a, err = resolveLocal(ref, opts.Quant)
	case looksLikeLocalPath(ref):
		return Artifact{}, fmt.Errorf("local GGUF %q does not exist or is not a file", ref)
	case strings.Contains(ref, "/"):
		a, err = r.resolveHF(ctx, ref)
	default:
		a, err = r.resolveOllama(ctx, ref)
	}
	if err != nil {
		return Artifact{}, err
	}
	a.Reference = ref
	a.MetadataBytes = r.fetched
	if !a.Arch.KnownForKV() {
		return Artifact{}, fmt.Errorf("model architecture is incomplete; refusing an optimistic fit prediction")
	}
	return a, nil
}

func fileExists(path string) bool { st, err := os.Stat(path); return err == nil && !st.IsDir() }

func looksLikeLocalPath(ref string) bool {
	if strings.EqualFold(filepath.Ext(ref), ".gguf") || filepath.IsAbs(ref) {
		return true
	}
	return strings.HasPrefix(ref, "./") || strings.HasPrefix(ref, "../") ||
		strings.HasPrefix(ref, `.\`) || strings.HasPrefix(ref, `..\`)
}

func resolveLocal(path, quant string) (Artifact, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return Artifact{}, err
	}
	info, err := gguf.Read(abs)
	if err != nil {
		return Artifact{}, fmt.Errorf("read GGUF metadata: %w", err)
	}
	q := info.Quantization()
	if q == "" {
		q = quantFromName(filepath.Base(abs))
	}
	if want := normalizeQuant(quant); want != "" {
		if q != "" && want != q {
			return Artifact{}, fmt.Errorf("requested quant %s, but %s is %s", want, filepath.Base(abs), q)
		}
		if q == "" {
			q = want
		}
	}
	return artifactFromInfo(SourceLocal, abs, filepath.Base(abs), q, info, "GGUF file size", "GGUF header"), nil
}

type hfModel struct {
	ID       string `json:"id"`
	SHA      string `json:"sha"`
	Siblings []struct {
		Name string `json:"rfilename"`
		Size uint64 `json:"size"`
		LFS  *struct {
			Size uint64 `json:"size"`
		} `json:"lfs"`
	} `json:"siblings"`
}

func (r *resolver) resolveHF(ctx context.Context, repo string) (Artifact, error) {
	repo = strings.Trim(repo, "/")
	if repo == "" || !strings.Contains(repo, "/") {
		return Artifact{}, fmt.Errorf("Hugging Face MODEL must be owner/repo")
	}
	api := "https://huggingface.co/api/models/" + escapeRepo(repo)
	if r.opts.Revision != "" {
		api += "/revision/" + url.PathEscape(r.opts.Revision)
	}
	api += "?blobs=true"
	var m hfModel
	if err := r.getJSON(ctx, api, &m, true); err != nil {
		return Artifact{}, fmt.Errorf("resolve Hugging Face model %s: %w", repo, err)
	}
	want := normalizeQuant(r.opts.Quant)
	var all []hfFile
	for _, f := range m.Siblings {
		if !strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			continue
		}
		sz := f.Size
		if sz == 0 && f.LFS != nil {
			sz = f.LFS.Size
		}
		all = append(all, hfFile{Name: f.Name, Size: sz})
	}
	var candidates []hfFile
	if r.opts.File != "" {
		var key string
		for _, f := range all {
			if strings.EqualFold(f.Name, r.opts.File) {
				key, _, _, _ = hfLogicalKey(f.Name)
				break
			}
		}
		if key == "" {
			return Artifact{}, fmt.Errorf("GGUF file %q was not found in %s", r.opts.File, repo)
		}
		for _, f := range all {
			k, _, _, _ := hfLogicalKey(f.Name)
			if k == key {
				candidates = append(candidates, f)
			}
		}
	} else {
		for _, f := range all {
			if want == "" || quantFromName(f.Name) == want {
				candidates = append(candidates, f)
			}
		}
	}
	if len(candidates) == 0 {
		return Artifact{}, fmt.Errorf("no GGUF file matching quant %q in %s", want, repo)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Name < candidates[j].Name })
	group, err := selectHFGroup(candidates)
	if err != nil {
		return Artifact{}, err
	}
	var total uint64
	for _, f := range group {
		if f.Size == 0 {
			return Artifact{}, fmt.Errorf("Hub did not report the size of %s; refusing an optimistic fit prediction", f.Name)
		}
		total = saturatingAdd(total, f.Size)
	}
	revision := m.SHA
	if revision == "" {
		revision = r.opts.Revision
		if revision == "" {
			revision = "main"
		}
	}
	fileURL := "https://huggingface.co/" + escapeRepo(repo) + "/resolve/" + url.PathEscape(revision) + "/" + escapePath(group[0].Name)
	info, _, err := r.readRemoteGGUF(ctx, fileURL, r.opts.HFToken, total)
	if err != nil {
		return Artifact{}, fmt.Errorf("read remote GGUF header: %w", err)
	}
	q := quantFromName(group[0].Name)
	if q == "" {
		q = info.Quantization()
	}
	if want != "" {
		if q != "" && q != want {
			return Artifact{}, fmt.Errorf("requested quant %s, but %s is %s", want, group[0].Name, q)
		}
		if q == "" {
			q = want
		}
	}
	a := artifactFromInfo(SourceHF, repo+"@"+revision, group[0].Name, q, info, "Hub file metadata", "ranged GGUF header")
	a.WeightBytes = total
	a.Revision = revision
	a.Digest = m.SHA
	return a, nil
}

type hfFile struct {
	Name string
	Size uint64
}

var shardRE = regexp.MustCompile(`(?i)-([0-9]{5})-of-([0-9]{5})\.gguf$`)

func selectHFGroup(files []hfFile) ([]hfFile, error) {
	groups := map[string][]hfFile{}
	for _, f := range files {
		key, _, _, _ := hfLogicalKey(f.Name)
		groups[key] = append(groups[key], f)
	}
	if len(groups) == 1 {
		for _, group := range groups {
			_, _, total, sharded := hfLogicalKey(group[0].Name)
			if sharded {
				if len(group) != total {
					return nil, fmt.Errorf("GGUF shard set is incomplete: found %d of %d files", len(group), total)
				}
				seen := make(map[int]bool, len(group))
				for _, f := range group {
					_, part, wantTotal, ok := hfLogicalKey(f.Name)
					if !ok || wantTotal != total || part < 1 || part > total || seen[part] {
						return nil, fmt.Errorf("GGUF shard set has invalid or duplicate part numbers")
					}
					seen[part] = true
				}
			}
			sort.Slice(group, func(i, j int) bool { return group[i].Name < group[j].Name })
			return group, nil
		}
	}
	var names []string
	for _, f := range files {
		names = append(names, f.Name)
	}
	return nil, fmt.Errorf("multiple GGUF files match; select one with --file: %s", strings.Join(names, ", "))
}

func hfLogicalKey(name string) (key string, part, total int, sharded bool) {
	m := shardRE.FindStringSubmatch(name)
	if len(m) != 3 {
		return strings.ToLower(name), 0, 0, false
	}
	part, _ = strconv.Atoi(m[1])
	total, _ = strconv.Atoi(m[2])
	base := shardRE.ReplaceAllString(name, "")
	return strings.ToLower(base) + "#" + m[2], part, total, true
}

type ociManifest struct {
	Config struct {
		Digest string `json:"digest"`
		Size   uint64 `json:"size"`
	} `json:"config"`
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      uint64 `json:"size"`
	} `json:"layers"`
}

type ollamaConfig struct {
	FileType    string `json:"file_type"`
	ModelFamily string `json:"model_family"`
}

func (r *resolver) resolveOllama(ctx context.Context, name string) (Artifact, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Artifact{}, fmt.Errorf("Ollama MODEL name is empty")
	}
	modelName, tag := splitOllama(name)
	want := normalizeQuant(r.opts.Quant)
	tags := []string{tag}
	if want != "" {
		// Registry shorthand such as 3b-instruct often has no manifest of its own;
		// the real artifact is 3b-instruct-q4_K_M. Try that exact quant tag first.
		if candidate := rewriteOllamaQuant(tag, want); candidate != tag {
			tags = []string{candidate, tag}
		}
	}
	var manifest ociManifest
	var canonical, got, manifestDigest string
	var attempts []string
	resolved := false
	for _, candidate := range tags {
		m, c, id, digest, err := r.ollamaManifest(ctx, modelName, candidate)
		if err != nil {
			attempts = append(attempts, id+" ("+err.Error()+")")
			continue
		}
		q := normalizeQuant(c.FileType)
		if want != "" && q != want {
			if q == "" {
				q = "unknown quant"
			}
			attempts = append(attempts, id+" ("+q+")")
			continue
		}
		manifest, canonical, got, manifestDigest = m, id, q, digest
		resolved = true
		break
	}
	if !resolved {
		if want != "" {
			return Artifact{}, fmt.Errorf("requested quant %s could not be resolved; tried %s", want, strings.Join(attempts, ", "))
		}
		return Artifact{}, fmt.Errorf("resolve Ollama model %s: %s", name, strings.Join(attempts, ", "))
	}
	var weight uint64
	var modelDigest string
	for _, l := range manifest.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" || strings.Contains(l.MediaType, "projector") {
			weight = saturatingAdd(weight, l.Size)
			if modelDigest == "" && l.MediaType == "application/vnd.ollama.image.model" {
				modelDigest = l.Digest
			}
		}
	}
	if modelDigest == "" || weight == 0 {
		return Artifact{}, errors.New("Ollama manifest has no model layer")
	}
	base := ollamaRegistryPath(modelName)
	blobURL := "https://registry.ollama.ai/v2/" + base + "/blobs/" + modelDigest
	info, _, err := r.readRemoteGGUF(ctx, blobURL, "", weight)
	if err != nil {
		return Artifact{}, fmt.Errorf("read Ollama GGUF header: %w", err)
	}
	a := artifactFromInfo(SourceOllama, canonical, canonical, got, info, "Ollama manifest layer sizes", "ranged GGUF header")
	a.WeightBytes = weight
	a.Digest = manifestDigest
	if a.Digest == "" {
		a.Digest = modelDigest
	}
	return a, nil
}

func (r *resolver) ollamaManifest(ctx context.Context, modelName, tag string) (ociManifest, ollamaConfig, string, string, error) {
	base := ollamaRegistryPath(modelName)
	canonical := modelName + ":" + tag
	var m ociManifest
	digest, err := r.getJSONDigest(ctx, "https://registry.ollama.ai/v2/"+base+"/manifests/"+url.PathEscape(tag), &m, false)
	if err != nil {
		return m, ollamaConfig{}, canonical, "", err
	}
	var cfg ollamaConfig
	if m.Config.Digest != "" {
		if err := r.getJSON(ctx, "https://registry.ollama.ai/v2/"+base+"/blobs/"+m.Config.Digest, &cfg, false); err != nil {
			return m, cfg, canonical, digest, err
		}
	}
	return m, cfg, canonical, digest, nil
}

func (r *resolver) resolveURL(ctx context.Context, raw string) (Artifact, error) {
	info, size, err := r.readRemoteGGUF(ctx, raw, "", 0)
	if err != nil {
		return Artifact{}, err
	}
	if size == 0 {
		return Artifact{}, errors.New("server did not report the model size; refusing an optimistic fit prediction")
	}
	name := filepath.Base(strings.Split(raw, "?")[0])
	q := quantFromName(name)
	if q == "" {
		q = info.Quantization()
	}
	if want := normalizeQuant(r.opts.Quant); want != "" {
		if q != "" && want != q {
			return Artifact{}, fmt.Errorf("requested quant %s, URL appears to be %s", want, q)
		}
		if q == "" {
			q = want
		}
	}
	return artifactFromInfo(SourceURL, raw, name, q, info, "HTTP Content-Range", "ranged GGUF header"), nil
}

func artifactFromInfo(source Source, id, filename, q string, info gguf.Info, weightBasis, archBasis string) Artifact {
	return Artifact{CanonicalID: id, Source: source, Filename: filename, Quantization: normalizeQuant(q), WeightBytes: info.FileSize,
		Arch: info.ToArch(), ContextMax: info.ContextLength, WeightBasis: weightBasis, ArchitectureBasis: archBasis}
}

func (r *resolver) getJSON(ctx context.Context, raw string, out any, hfAuth bool) error {
	_, err := r.getJSONDigest(ctx, raw, out, hfAuth)
	return err
}

func (r *resolver) getJSONDigest(ctx context.Context, raw string, out any, hfAuth bool) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return "", err
	}
	if hfAuth && r.opts.HFToken != "" {
		req.Header.Set("Authorization", "Bearer "+r.opts.HFToken)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	remaining := r.opts.MaxMetadataBytes - r.fetched
	if remaining <= 0 {
		return "", errors.New("metadata limit exceeded")
	}
	lr := &io.LimitedReader{R: resp.Body, N: remaining + 1}
	b, readErr := io.ReadAll(lr)
	used := remaining + 1 - lr.N
	r.fetched += used
	if used > remaining {
		return "", errors.New("metadata limit exceeded")
	}
	if readErr != nil {
		return "", readErr
	}
	if err := json.Unmarshal(b, out); err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum[:]), nil
}

func (r *resolver) readRemoteGGUF(ctx context.Context, raw, token string, artifactSize uint64) (gguf.Info, uint64, error) {
	remaining := r.opts.MaxMetadataBytes - r.fetched
	if remaining <= 0 {
		return gguf.Info{}, 0, errors.New("metadata limit exceeded")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return gguf.Info{}, 0, err
	}
	req.Header.Set("Range", "bytes=0-"+strconv.FormatInt(remaining-1, 10))
	if token != "" && strings.HasSuffix(req.URL.Hostname(), "huggingface.co") {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return gguf.Info{}, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return gguf.Info{}, 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusOK && resp.ContentLength > remaining {
		return gguf.Info{}, 0, errors.New("server ignored Range; refusing model download")
	}
	size := responseArtifactSize(resp)
	parseSize := artifactSize
	if parseSize == 0 {
		parseSize = size
	}
	var data []byte
	lr := &io.LimitedReader{R: resp.Body, N: remaining + 1}
	buf := make([]byte, 64*model.KiB)
	var parseErr error
	for {
		n, readErr := lr.Read(buf)
		if n > 0 {
			data = append(data, buf[:n]...)
			r.fetched += int64(n)
			if int64(len(data)) > remaining {
				return gguf.Info{}, size, errors.New("metadata limit exceeded")
			}
			if (size > 0 && size < uint64(len(data))) || (artifactSize > 0 && artifactSize < uint64(len(data))) {
				return gguf.Info{}, size, errors.New("reported model size is smaller than the ranged metadata")
			}
			info, err := gguf.ReadPrefix(data, parseSize)
			if err == nil {
				return info, size, nil
			}
			parseErr = err
			if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				return gguf.Info{}, size, err
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				return gguf.Info{}, size, readErr
			}
			if parseErr != nil {
				return gguf.Info{}, size, parseErr
			}
			return gguf.Info{}, size, io.ErrUnexpectedEOF
		}
		if n == 0 {
			return gguf.Info{}, size, io.ErrNoProgress
		}
	}
}

func responseArtifactSize(resp *http.Response) uint64 {
	var size uint64
	// For a 206 response Content-Length is only the returned prefix. The full
	// artifact size must come from Content-Range (or the Hub's linked-size
	// header), otherwise treating the prefix as the model would be dangerously
	// optimistic.
	if resp.StatusCode == http.StatusOK && resp.ContentLength > 0 {
		size = uint64(resp.ContentLength)
	}
	if cr := resp.Header.Get("Content-Range"); cr != "" {
		if slash := strings.LastIndex(cr, "/"); slash >= 0 {
			if n, err := strconv.ParseUint(cr[slash+1:], 10, 64); err == nil {
				size = n
			}
		}
	}
	if linked := resp.Header.Get("X-Linked-Size"); linked != "" {
		if n, err := strconv.ParseUint(linked, 10, 64); err == nil {
			size = n
		}
	}
	return size
}

func splitOllama(name string) (string, string) {
	if i := strings.LastIndex(name, ":"); i > strings.LastIndex(name, "/") {
		return name[:i], name[i+1:]
	}
	return name, "latest"
}

func ollamaRegistryPath(name string) string {
	if strings.Contains(name, "/") {
		return name
	}
	return "library/" + name
}

var quantSuffixRE = regexp.MustCompile(`(?i)(?:-|_)(?:bf16|f(?:p)?16|f32|q[1-8](?:_[01]|_k(?:_[sml])?)?|iq[1-4]_[a-z0-9_]+|tq[12]_0)$`)

func rewriteOllamaQuant(tag, q string) string {
	q = ollamaTagQuant(q)
	if tag == "latest" {
		return tag
	}
	if quantSuffixRE.MatchString(tag) {
		return quantSuffixRE.ReplaceAllString(tag, "-"+q)
	}
	return tag + "-" + q
}

// Ollama's published tags use a lower-case quant family with upper-case
// variants (for example q4_K_M), while GGUF metadata uses Q4_K_M. Preserve that
// registry spelling when deriving an alternate tag from --quant.
func ollamaTagQuant(q string) string {
	q = normalizeQuant(q)
	switch q {
	case "F16":
		return "fp16"
	case "F32":
		return "fp32"
	}
	if i := strings.IndexByte(q, '_'); i >= 0 {
		return strings.ToLower(q[:i]) + q[i:]
	}
	return strings.ToLower(q)
}

var quantNameRE = regexp.MustCompile(`(?i)(?:^|[-._])(bf16|fp16|f16|f32|q[1-8](?:_[01]|_k(?:_[sml])?)?|iq[1-4]_[a-z0-9_]+|tq[12]_0)(?:[-.]|$)`)

func quantFromName(name string) string {
	m := quantNameRE.FindStringSubmatch(name)
	if len(m) > 1 {
		return normalizeQuant(m[1])
	}
	return ""
}
func normalizeQuant(q string) string {
	q = strings.ToUpper(strings.TrimSpace(q))
	if q == "FP16" {
		return "F16"
	}
	return q
}
func escapeRepo(repo string) string {
	parts := strings.Split(repo, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}
func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

// Package ledger stores fit predictions and later observations locally so a
// report can show prediction error without telemetry or an account.
package ledger

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/RamazanKara/vramwatch/internal/fit"
	"github.com/RamazanKara/vramwatch/internal/model"
)

const SchemaVersion = 1
const maxRecordBytes = 4 * model.MiB

type Observation struct {
	ObservedAt       time.Time        `json:"observed_at"`
	FootprintBytes   uint64           `json:"footprint_bytes"`
	Provenance       model.Provenance `json:"provenance"`
	Source           string           `json:"source"`
	SignedErrorPct   float64          `json:"signed_error_percent"`
	AbsoluteErrorPct float64          `json:"absolute_error_percent"`
}

type Record struct {
	SchemaVersion int          `json:"schema_version"`
	ID            string       `json:"id"`
	CreatedAt     time.Time    `json:"created_at"`
	Loader        string       `json:"loader"`
	Prediction    fit.Result   `json:"prediction"`
	Observation   *Observation `json:"observation,omitempty"`
}

func StateDir() (string, error) {
	if d := os.Getenv("VRAMWATCH_STATE_DIR"); d != "" {
		return d, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	switch runtime.GOOS {
	case "windows":
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			return filepath.Join(d, "vramwatch"), nil
		}
		return filepath.Join(home, "AppData", "Local", "vramwatch"), nil
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "vramwatch"), nil
	default:
		if d := os.Getenv("XDG_STATE_HOME"); d != "" {
			return filepath.Join(d, "vramwatch"), nil
		}
		return filepath.Join(home, ".local", "state", "vramwatch"), nil
	}
}

func Save(result fit.Result, loader string) (Record, error) {
	id, err := newID()
	if err != nil {
		return Record{}, err
	}
	r := Record{SchemaVersion: SchemaVersion, ID: id, CreatedAt: time.Now().UTC(), Loader: loader, Prediction: result}
	return r, write(r)
}

func UpdateObservation(id string, footprint uint64, provenance model.Provenance, source string) (Record, error) {
	r, err := Load(id)
	if err != nil {
		return Record{}, err
	}
	if footprint == 0 {
		return Record{}, errors.New("observed footprint is zero")
	}
	if r.Observation != nil && provenanceRank(provenance) < provenanceRank(r.Observation.Provenance) {
		return r, nil
	}
	pred := r.Prediction.ExpectedFootprintBytes
	signed := 100 * (float64(pred) - float64(footprint)) / float64(footprint)
	abs := signed
	if abs < 0 {
		abs = -abs
	}
	r.Observation = &Observation{ObservedAt: time.Now().UTC(), FootprintBytes: footprint, Provenance: provenance, Source: source, SignedErrorPct: signed, AbsoluteErrorPct: abs}
	return r, write(r)
}

func provenanceRank(p model.Provenance) int {
	switch p {
	case model.ProvenanceMeasured:
		return 3
	case model.ProvenanceReported:
		return 2
	case model.ProvenanceEstimated:
		return 1
	default:
		return 0
	}
}

func Load(id string) (Record, error) {
	if !validID(id) {
		return Record{}, fmt.Errorf("invalid prediction ID %q", id)
	}
	id = strings.ToLower(id)
	d, err := StateDir()
	if err != nil {
		return Record{}, err
	}
	f, err := os.Open(filepath.Join(d, "predictions", id+".json"))
	if err != nil {
		return Record{}, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, int64(maxRecordBytes)+1))
	if err != nil {
		return Record{}, err
	}
	if len(b) > maxRecordBytes {
		return Record{}, fmt.Errorf("prediction record exceeds %s", model.HumanBytes(maxRecordBytes))
	}
	var r Record
	if err := json.Unmarshal(b, &r); err != nil {
		return Record{}, err
	}
	if r.SchemaVersion != SchemaVersion {
		return Record{}, fmt.Errorf("unsupported ledger schema %d", r.SchemaVersion)
	}
	if !validID(r.ID) || strings.ToLower(r.ID) != id {
		return Record{}, fmt.Errorf("prediction record ID does not match filename")
	}
	r.ID = id
	return r, nil
}

func validID(id string) bool {
	if len(id) != 16 {
		return false
	}
	for _, c := range strings.ToLower(id) {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func Latest() (Record, error) {
	list, err := List()
	if err != nil {
		return Record{}, err
	}
	if len(list) == 0 {
		return Record{}, os.ErrNotExist
	}
	return list[0], nil
}

func List() ([]Record, error) {
	d, err := StateDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(filepath.Join(d, "predictions"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Record
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		r, err := Load(e.Name()[:len(e.Name())-5])
		if err == nil {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out, nil
}

func write(r Record) error {
	if !validID(r.ID) {
		return fmt.Errorf("invalid prediction ID %q", r.ID)
	}
	d, err := StateDir()
	if err != nil {
		return err
	}
	d = filepath.Join(d, "predictions")
	if err := os.MkdirAll(d, 0o700); err != nil {
		return err
	}
	// MkdirAll leaves an existing directory's mode untouched. Tighten it as
	// well, because records can contain the original local path or URL.
	if err := os.Chmod(d, 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if len(b) > maxRecordBytes {
		return fmt.Errorf("prediction record exceeds %s", model.HumanBytes(maxRecordBytes))
	}
	tmp, err := os.CreateTemp(d, ".prediction-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	ok := false
	defer func() {
		tmp.Close()
		if !ok {
			os.Remove(name)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, filepath.Join(d, r.ID+".json")); err != nil {
		return err
	}
	ok = true
	return nil
}

func newID() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

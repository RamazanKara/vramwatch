package ledger

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/fit"
	"github.com/RamazanKara/vramwatch/internal/model"
)

func ledgerPrediction(bytes uint64) fit.Result {
	return fit.Result{
		Policy: fit.PolicyVersion,
		Artifact: fit.Artifact{
			CanonicalID: "owner/model@abc", Source: fit.SourceHF, Quantization: "Q4_K_M", WeightBytes: 4 * model.GiB,
		},
		Context:                8192,
		KVCacheType:            "F16",
		ExpectedFootprintBytes: bytes,
	}
}

func TestSaveLoadLatestAndObservation(t *testing.T) {
	state := t.TempDir()
	t.Setenv("VRAMWATCH_STATE_DIR", state)

	first, err := Save(ledgerPrediction(10*model.GiB), "ollama")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Save(ledgerPrediction(12*model.GiB), "llama.cpp")
	if err != nil {
		t.Fatal(err)
	}
	if first.ID == second.ID || !validID(first.ID) || !validID(second.ID) {
		t.Fatalf("bad IDs: %q %q", first.ID, second.ID)
	}
	loaded, err := Load(first.ID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Loader != "ollama" || loaded.Prediction.ExpectedFootprintBytes != 10*model.GiB {
		t.Fatalf("loaded record = %+v", loaded)
	}
	latest, err := Latest()
	if err != nil {
		t.Fatal(err)
	}
	if latest.ID != second.ID {
		t.Fatalf("latest = %s, want %s", latest.ID, second.ID)
	}

	updated, err := UpdateObservation(first.ID, 8*model.GiB, model.ProvenanceMeasured, "driver process memory")
	if err != nil {
		t.Fatal(err)
	}
	if updated.Observation == nil || updated.Observation.SignedErrorPct != 25 || updated.Observation.AbsoluteErrorPct != 25 {
		t.Fatalf("observation = %+v", updated.Observation)
	}
	reloaded, err := Load(first.ID)
	if err != nil || reloaded.Observation == nil {
		t.Fatalf("updated record was not persisted: %+v, %v", reloaded, err)
	}

	path := filepath.Join(state, "predictions", first.ID+".json")
	if st, err := os.Stat(path); err != nil {
		t.Fatal(err)
	} else if st.Mode().Perm()&0o077 != 0 {
		t.Errorf("ledger record permissions = %o, want private", st.Mode().Perm())
	}
}

func TestLedgerRejectsUnsafeIDsAndEmptyState(t *testing.T) {
	state := t.TempDir()
	t.Setenv("VRAMWATCH_STATE_DIR", state)
	if _, err := Load("../../secrets"); err == nil {
		t.Error("path-like prediction ID should fail")
	}
	if _, err := UpdateObservation("not-an-id", 1, model.ProvenanceMeasured, "test"); err == nil {
		t.Error("invalid update ID should fail")
	}
	if _, err := Latest(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("empty Latest error = %v, want os.ErrNotExist", err)
	}
	list, err := List()
	if err != nil || len(list) != 0 {
		t.Fatalf("empty List = %v, %v", list, err)
	}

	const filenameID = "0123456789abcdef"
	dir := filepath.Join(state, "predictions")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	bad := `{"schema_version":1,"id":"../../escape","created_at":"2026-01-01T00:00:00Z","loader":"ollama","prediction":{}}`
	if err := os.WriteFile(filepath.Join(dir, filenameID+".json"), []byte(bad), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(filenameID); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("embedded unsafe ID error = %v", err)
	}
}

func TestUpdateObservationRejectsZero(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	rec, err := Save(ledgerPrediction(model.GiB), "ollama")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := UpdateObservation(rec.ID, 0, model.ProvenanceMeasured, "test"); err == nil {
		t.Error("zero observation should fail")
	}
}

func TestObservationProvenanceCannotBeDowngraded(t *testing.T) {
	t.Setenv("VRAMWATCH_STATE_DIR", t.TempDir())
	rec, err := Save(ledgerPrediction(10*model.GiB), "ollama")
	if err != nil {
		t.Fatal(err)
	}
	measured, err := UpdateObservation(rec.ID, 8*model.GiB, model.ProvenanceMeasured, "driver")
	if err != nil {
		t.Fatal(err)
	}
	kept, err := UpdateObservation(rec.ID, 9*model.GiB, model.ProvenanceEstimated, "model math")
	if err != nil {
		t.Fatal(err)
	}
	if kept.Observation == nil || kept.Observation.FootprintBytes != measured.Observation.FootprintBytes || kept.Observation.Provenance != model.ProvenanceMeasured {
		t.Fatalf("strong observation was downgraded: %+v", kept.Observation)
	}
	updated, err := UpdateObservation(rec.ID, 7*model.GiB, model.ProvenanceMeasured, "new driver sample")
	if err != nil {
		t.Fatal(err)
	}
	if updated.Observation == nil || updated.Observation.FootprintBytes != 7*model.GiB {
		t.Fatalf("same-quality observation did not refresh: %+v", updated.Observation)
	}
}

func TestWriteRejectsOversizedRecordAndTightensDirectory(t *testing.T) {
	state := t.TempDir()
	t.Setenv("VRAMWATCH_STATE_DIR", state)
	dir := filepath.Join(state, "predictions")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	r := Record{
		SchemaVersion: SchemaVersion,
		ID:            "0123456789abcdef",
		Prediction: fit.Result{Artifact: fit.Artifact{
			CanonicalID: strings.Repeat("x", maxRecordBytes),
		}},
	}
	if err := write(r); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized write error = %v", err)
	}
	if st, err := os.Stat(dir); err != nil {
		t.Fatal(err)
	} else if st.Mode().Perm()&0o077 != 0 {
		t.Fatalf("ledger directory permissions = %o, want private", st.Mode().Perm())
	}
}

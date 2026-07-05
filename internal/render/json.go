package render

import (
	"encoding/json"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// JSON renders a snapshot as indented, machine-readable JSON.
func JSON(snap model.Snapshot) ([]byte, error) {
	return json.MarshalIndent(snap, "", "  ")
}

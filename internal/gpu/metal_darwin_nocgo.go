//go:build darwin && !cgo

package gpu

import (
	"context"
	"github.com/RamazanKara/vramwatch/internal/model"
)

// AppleMetal is unavailable in CGO-free builds; release artifacts are built
// natively against the system Metal framework.
type AppleMetal struct{}

func (AppleMetal) Name() string                                { return "apple-metal" }
func (AppleMetal) Vendor() model.Vendor                        { return model.VendorApple }
func (AppleMetal) Available(context.Context) bool              { return false }
func (AppleMetal) Sample(context.Context) ([]model.GPU, error) { return nil, nil }

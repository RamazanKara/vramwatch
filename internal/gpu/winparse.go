package gpu

import (
	"encoding/csv"
	"strconv"
	"strings"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// This file holds the pure parsers for the Windows GPU provider (registry +
// typeperf performance counters). They live in an untagged file so they can be
// unit-tested on any platform against captured fixtures.

// parseRegValues walks `reg query ... /s /v <name>` output and returns, per
// registry subkey path, the value of that name. When a subkey has several
// values (e.g. multiple MatchingDeviceId), a PCI\VEN_ value wins.
func parseRegValues(out, valueName string) map[string]string {
	res := map[string]string{}
	var subkey string
	for _, line := range strings.Split(out, "\n") {
		t := strings.TrimRight(line, "\r")
		if strings.HasPrefix(t, "HKEY") {
			subkey = strings.TrimSpace(t)
			continue
		}
		ts := strings.TrimSpace(t)
		if subkey == "" || !strings.HasPrefix(ts, valueName) || !strings.Contains(ts, "REG_") {
			continue
		}
		rest := ts[strings.Index(ts, "REG_"):]
		parts := strings.Fields(rest) // ["REG_QWORD","0x4ff000000"] / ["REG_SZ","AMD","Radeon",...]
		if len(parts) < 2 {
			continue
		}
		v := strings.Join(parts[1:], " ")
		if cur, ok := res[subkey]; !ok || (strings.Contains(v, "VEN_") && !strings.Contains(cur, "VEN_")) {
			res[subkey] = v
		}
	}
	return res
}

// parseRegUint parses a reg value that may be hex ("0x4ff000000") or decimal.
func parseRegUint(s string) uint64 {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		n, err := strconv.ParseUint(s[2:], 16, 64)
		if err != nil {
			return 0
		}
		return n
	}
	n, _ := strconv.ParseUint(s, 10, 64)
	return n
}

// vendorFromDeviceID maps a PCI device id to a GPU vendor.
func vendorFromDeviceID(s string) model.Vendor {
	u := strings.ToUpper(s)
	switch {
	case strings.Contains(u, "VEN_1002"):
		return model.VendorAMD
	case strings.Contains(u, "VEN_10DE"):
		return model.VendorNVIDIA
	case strings.Contains(u, "VEN_8086"):
		return model.VendorIntel
	}
	return model.VendorUnknown
}

// parseTypeperfAdapter parses `typeperf "\GPU Adapter Memory(*)\Dedicated Usage"`
// CSV into a map of adapter LUID -> dedicated bytes in use (the last sample).
func parseTypeperfAdapter(out string) map[string]uint64 {
	res := map[string]uint64{}
	var header, data []string
	for _, r := range readCSVRows(out) {
		if len(r) > 1 && strings.Contains(r[0], "PDH-CSV") {
			header, data = r, nil
			continue
		}
		if header != nil && len(r) == len(header) {
			data = r // keep the last matching data row
		}
	}
	if header == nil || data == nil {
		return res
	}
	for i := 1; i < len(header); i++ {
		inst, kind := parseGPUCounter(header[i])
		if kind != "adapter" {
			continue
		}
		luid := strings.TrimPrefix(inst, "luid_")
		res[luid] += parseFloatBytes(data[i])
	}
	return res
}

// parseGPUCounter extracts the instance name and kind (adapter/process) from a
// GPU performance-counter path.
func parseGPUCounter(col string) (instance, kind string) {
	lp := strings.Index(col, "(")
	rp := strings.LastIndex(col, ")")
	if lp < 0 || rp <= lp {
		return "", ""
	}
	instance = col[lp+1 : rp]
	switch {
	case strings.Contains(col[:lp], "GPU Adapter Memory"):
		kind = "adapter"
	case strings.Contains(col[:lp], "GPU Process Memory"):
		kind = "process"
	}
	return instance, kind
}

func parseFloatBytes(s string) uint64 {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || f < 0 {
		return 0
	}
	return uint64(f)
}

// readCSVRows parses typeperf/tasklist CSV output, tolerating the non-CSV
// trailer lines those tools print.
func readCSVRows(out string) [][]string {
	r := csv.NewReader(strings.NewReader(out))
	r.FieldsPerRecord = -1
	r.LazyQuotes = true
	var rows [][]string
	for {
		rec, err := r.Read()
		if err != nil {
			if len(rec) == 0 {
				break
			}
			continue
		}
		rows = append(rows, rec)
	}
	return rows
}

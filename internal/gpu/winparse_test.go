package gpu

import (
	"fmt"
	"testing"

	"github.com/RamazanKara/vramwatch/internal/model"
)

func TestParseTypeperfAdapterUsage(t *testing.T) {
	for _, tc := range []struct {
		value string
		known bool
		bytes uint64
	}{
		{"0.000000", true, 0},
		{" 1024.000000 ", true, 1024},
		{"", false, 0},
		{"N/A", false, 0},
		{"-1", false, 0},
		{"NaN", false, 0},
		{"+Inf", false, 0},
		{"1e100", false, 0},
		{"18446744073709551616", false, 0},
	} {
		t.Run(tc.value, func(t *testing.T) {
			csv := fmt.Sprintf("\"(PDH-CSV 4.0)\",\"GPU Adapter Memory(luid_gpu)\\Dedicated Usage\"\n\"time\",\"%s\"\n", tc.value)
			got, known := parseTypeperfAdapter(csv)["gpu"]
			if known != tc.known || got != tc.bytes {
				t.Errorf("usage = %d, known = %v; want %d, %v", got, known, tc.bytes, tc.known)
			}
		})
	}
}

// Fixtures below are real output captured from an AMD Radeon RX 7900 XT on
// Windows 11 (typeperf + reg query).

const typeperfAdapterFixture = `"(PDH-CSV 4.0)","\\HOMEPC\GPU Adapter Memory(luid_0x00000000_0x00017E08_phys_0)\Dedicated Usage","\\HOMEPC\GPU Adapter Memory(luid_0x00000000_0x679BDD95_phys_0)\Dedicated Usage"
"07/05/2026 12:34:28.096","0.000000","5764141056.000000"
Vorgang wird beendet...
Der Befehl wurde erfolgreich ausgefuehrt.
`

func TestParseTypeperfAdapter(t *testing.T) {
	a := parseTypeperfAdapter(typeperfAdapterFixture)
	if a["0x00000000_0x679BDD95_phys_0"] != 5764141056 {
		t.Errorf("AMD adapter usage = %d, want 5764141056", a["0x00000000_0x679BDD95_phys_0"])
	}
	if a["0x00000000_0x00017E08_phys_0"] != 0 {
		t.Errorf("software adapter usage = %d, want 0", a["0x00000000_0x00017E08_phys_0"])
	}
}

const regQwFixture = `
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001
    HardwareInformation.qwMemorySize    REG_QWORD    0x4ff000000

`

func TestParseRegValuesQw(t *testing.T) {
	m := parseRegValues(regQwFixture, "HardwareInformation.qwMemorySize")
	if len(m) != 1 {
		t.Fatalf("want 1 subkey, got %d: %v", len(m), m)
	}
	for _, v := range m {
		if parseRegUint(v) != 21458059264 { // 0x4ff000000 = 19.98 GiB
			t.Errorf("qwMemorySize = %d, want 21458059264", parseRegUint(v))
		}
	}
}

func TestParseRegValuesName(t *testing.T) {
	out := "HKEY_LOCAL_MACHINE\\...\\0001\n    DriverDesc    REG_SZ    AMD Radeon RX 7900 XT\n"
	m := parseRegValues(out, "DriverDesc")
	for _, v := range m {
		if v != "AMD Radeon RX 7900 XT" {
			t.Errorf("DriverDesc = %q", v)
		}
	}
}

func TestParseRegValuesPrefersVEN(t *testing.T) {
	// A subkey with several MatchingDeviceId values: the PCI\VEN one must win.
	out := "HKEY_LOCAL_MACHINE\\...\\0001\n" +
		"    MatchingDeviceId    REG_SZ    root\\sudomaker\\sudovda\n" +
		"    MatchingDeviceId    REG_SZ    PCI\\VEN_1002&DEV_744C&SUBSYS_79051EAE&REV_CC\n"
	m := parseRegValues(out, "MatchingDeviceId")
	for _, v := range m {
		if vendorFromDeviceID(v) != model.VendorAMD {
			t.Errorf("expected AMD from %q", v)
		}
	}
}

func TestVendorFromDeviceID(t *testing.T) {
	cases := map[string]model.Vendor{
		`PCI\VEN_1002&DEV_744C`:        model.VendorAMD,
		`PCI\VEN_10DE&DEV_2684`:        model.VendorNVIDIA,
		`PCI\VEN_8086&DEV_56A0`:        model.VendorIntel,
		`ROOT\MetaVirtualScreenDriver`: model.VendorUnknown,
	}
	for in, want := range cases {
		if got := vendorFromDeviceID(in); got != want {
			t.Errorf("vendorFromDeviceID(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLooksNVIDIA(t *testing.T) {
	yes := []string{"NVIDIA GeForce RTX 4090", "Quadro P2000", "Tesla T4", "nvidia geforce gtx 1080"}
	no := []string{"AMD Radeon RX 7900 XT", "Intel Arc A770", "Microsoft Basic Render Driver", ""}
	for _, n := range yes {
		if !looksNVIDIA(n) {
			t.Errorf("%q should look like NVIDIA", n)
		}
	}
	for _, n := range no {
		if looksNVIDIA(n) {
			t.Errorf("%q should NOT look like NVIDIA", n)
		}
	}
}

func TestParseRegUint(t *testing.T) {
	cases := map[string]uint64{
		"0x4ff000000": 21458059264,
		"0X10":        16,
		"1073741824":  1073741824,
		"":            0,
		"garbage":     0,
	}
	for in, want := range cases {
		if got := parseRegUint(in); got != want {
			t.Errorf("parseRegUint(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestSingleDedicatedAdapter(t *testing.T) {
	for _, tc := range []struct {
		name       string
		capacities map[string]string
		single     bool
	}{
		{"none", nil, false},
		{"AMD", map[string]string{"amd": "0x4ff000000"}, true},
		{"software adapter", map[string]string{"amd": "0x4ff000000", "software": "0"}, true},
		{"mixed vendors", map[string]string{"amd": "0x4ff000000", "nvidia": "0x600000000"}, false},
		{"two AMD cards", map[string]string{"amd0": "0x4ff000000", "amd1": "0x4ff000000"}, false},
		{"unknown capacity", map[string]string{"unknown": "N/A"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := singleDedicatedAdapter(tc.capacities); got != tc.single {
				t.Errorf("single dedicated adapter = %v, want %v", got, tc.single)
			}
		})
	}
}

package gpu

import "testing"

func FuzzNvidiaCSV(f *testing.F) {
	f.Add(nvGPUFixture, nvAppsFixture)
	f.Add("", "No running processes found")
	f.Fuzz(func(t *testing.T, devices, processes string) {
		gpus, indices := parseNvidiaGPUs(devices)
		for uuid, index := range indices {
			if index < 0 || index >= len(gpus) {
				t.Fatalf("invalid GPU index for %q: %d", uuid, index)
			}
		}
		parseNvidiaApps(processes, gpus, indices)
	})
}

func FuzzAMDSMI(f *testing.F) {
	f.Add(amdStaticFixture, amdMetricFixture)
	f.Add(`[{"gpu":0,"asic":"N/A","vram":"N/A"}]`, `[]`)
	f.Add("", "")
	f.Fuzz(func(t *testing.T, static, metric string) {
		gpus, err := parseAMDSMI(static, metric)
		if err != nil {
			return
		}
		for i := 1; i < len(gpus); i++ {
			if gpus[i-1].Index > gpus[i].Index {
				t.Fatal("GPU indices are not sorted")
			}
		}
	})
}

func FuzzWindowsOutput(f *testing.F) {
	f.Add(typeperfAdapterFixture, regQwFixture)
	f.Add("", "")
	f.Fuzz(func(t *testing.T, counters, registry string) {
		parseTypeperfAdapter(counters)
		for _, value := range parseRegValues(registry, "HardwareInformation.qwMemorySize") {
			parseRegUint(value)
		}
	})
}

func FuzzFdinfo(f *testing.F) {
	f.Add(drmFdinfo)
	f.Add("drm-driver: amdgpu\ndrm-resident-vram: 18446744073709551615 GiB\n")
	f.Add("")
	f.Fuzz(func(t *testing.T, content string) {
		parseFdinfo(content)
	})
}

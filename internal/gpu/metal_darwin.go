//go:build darwin && cgo

package gpu

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <stdint.h>
#include <string.h>

static int vmw_metal_probe(char *name, size_t name_len, uint64_t *physical,
                           uint64_t *available, uint64_t *recommended) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || !device.hasUnifiedMemory) return 0;
        const char *n = device.name.UTF8String;
        if (n != NULL && name_len > 0) {
            strncpy(name, n, name_len - 1);
            name[name_len - 1] = '\0';
        }
        *recommended = device.recommendedMaxWorkingSetSize;
        size_t sz = sizeof(*physical);
        if (sysctlbyname("hw.memsize", physical, &sz, NULL, 0) != 0) *physical = 0;
        vm_statistics64_data_t vm;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        mach_port_t host = mach_host_self();
        vm_size_t page = 0;
        kern_return_t page_result = host_page_size(host, &page);
        kern_return_t stats_result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
        mach_port_deallocate(mach_task_self(), host);
        if (page_result == KERN_SUCCESS && stats_result == KERN_SUCCESS) {
            // speculative_count is already included in free_count, and
            // purgeable pages can overlap other VM categories. free + inactive
            // is the conservative non-overlapping reclaimable estimate.
            *available = (uint64_t)(vm.free_count + vm.inactive_count) * page;
        } else {
            *available = 0;
        }
        return 1;
    }
}
*/
import "C"

import (
	"context"
	"unsafe"

	"github.com/RamazanKara/vramwatch/internal/model"
)

// AppleMetal reports Apple silicon's unified-memory pool. The in-principle
// accelerator budget is Metal's recommended working set; current availability
// comes from non-overlapping Mach free + inactive page counts, not this observer
// process's Metal allocations.
type AppleMetal struct{}

func (AppleMetal) Name() string                   { return "apple-metal" }
func (AppleMetal) Vendor() model.Vendor           { return model.VendorApple }
func (AppleMetal) Available(context.Context) bool { _, ok := metalSample(); return ok }
func (AppleMetal) Sample(context.Context) ([]model.GPU, error) {
	g, ok := metalSample()
	if !ok {
		return nil, nil
	}
	return []model.GPU{g}, nil
}

func metalSample() (model.GPU, bool) {
	name := make([]byte, 256)
	var physical, available, recommended C.uint64_t
	ok := C.vmw_metal_probe((*C.char)(unsafe.Pointer(&name[0])), C.size_t(len(name)), &physical, &available, &recommended)
	if ok == 0 {
		return model.GPU{}, false
	}
	end := 0
	for end < len(name) && name[end] != 0 {
		end++
	}
	p := uint64(physical)
	a := uint64(available)
	if a > p {
		a = p
	}
	budget := uint64(recommended)
	if budget == 0 || budget > p {
		budget = p
	}
	if budget == 0 {
		return model.GPU{}, false
	}
	if a > budget {
		a = budget
	}
	return model.GPU{Index: 0, Name: string(name[:end]), Vendor: model.VendorApple, TotalBytes: budget, UsedBytes: budget - a, FreeBytes: a, MemoryKind: model.MemoryUnified, BudgetBytes: budget, CapacitySource: model.ProvenanceMeasured, UsageSource: model.ProvenanceMeasured}, true
}

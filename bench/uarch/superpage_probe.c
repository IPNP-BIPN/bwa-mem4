// Does macOS on Apple Silicon actually honour VM_FLAGS_SUPERPAGE_SIZE_2MB?
// The constant is in the SDK headers, but the headers are shared across arches.
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/mach_error.h>
#include <stdio.h>
#include <string.h>

static void try_alloc(const char *label, int flags, size_t size) {
    mach_vm_address_t addr = 0;
    kern_return_t kr = mach_vm_allocate(mach_task_self(), &addr, size, flags);
    if (kr == KERN_SUCCESS) {
        // touch it: allocation succeeding is not the same as it being usable
        memset((void *)addr, 1, size);
        printf("  %-28s SUCCESS  addr=0x%llx  (2MB-aligned: %s)\n",
               label, (unsigned long long)addr,
               (addr % (2ull<<20)) == 0 ? "yes" : "NO");
        mach_vm_deallocate(mach_task_self(), addr, size);
    } else {
        printf("  %-28s FAILED   kr=%d (%s)\n", label, kr, mach_error_string(kr));
    }
}

int main(void) {
    printf("Apple Silicon superpage probe (mach_vm_allocate)\n");
    try_alloc("plain 2MB (control)",      VM_FLAGS_ANYWHERE, 2u<<20);
    try_alloc("SUPERPAGE_SIZE_2MB 2MB",   VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_2MB, 2u<<20);
    try_alloc("SUPERPAGE_SIZE_ANY 2MB",   VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_ANY, 2u<<20);
    try_alloc("SUPERPAGE_SIZE_2MB 64MB",  VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_2MB, 64u<<20);
    return 0;
}

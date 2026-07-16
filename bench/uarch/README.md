# Apple Silicon memory-subsystem probes

Two self-contained C probes measuring the constraints that actually bound this aligner at `-t8`.
Built ad hoc during the 2026-07-16 perf session; kept because their results are load-bearing for
`docs/optimization-roadmap.md` and cheap to re-run on new hardware.

```
cc -O2 -o /tmp/tlb tlb_probe.c && /tmp/tlb 1 && /tmp/tlb 8
cc -O2 -o /tmp/sp superpage_probe.c && /tmp/sp
```

## `superpage_probe.c` — are huge pages available?

**Measured on M4 Max / macOS 15 (Darwin 25.5): NO.** `mach_vm_allocate` with
`VM_FLAGS_SUPERPAGE_SIZE_2MB` (and `_SIZE_ANY`, and at 64 MB) returns `KERN_INVALID_ARGUMENT` (4),
while the identical allocation without the flag succeeds. The constant *is* present in the SDK
headers (`mach/vm_statistics.h`), which is an x86_64 legacy: the arm64 kernel rejects it. `sys/mman.h`
has no `MAP_HUGETLB` / `MAP_ALIGNED` equivalent either.

Base page size is **16 KB** (`getconf PAGESIZE`).

## `tlb_probe.c` — the TLB hierarchy and its cost

Chases a random cycle touching exactly **one cache line per 16 KB page**, so the *data* footprint
(pages x 128 B) stays cache-resident far longer than the *page* count stays TLB-resident. Any cliff
is therefore the TLB, not the cache. Compile with `-DPAGE=128` (see the sed in the session log) for a
**packed control** with the same line count but ~128x fewer pages: `paged / packed` at equal line
count isolates the TLB tax.

Measured, M4 Max, quiet host:

| span | 1 thread | 8 threads | contention |
|---|---|---|---|
| 2 MB | 0.7 ns | 1.0 ns | 1.4x |
| 64 MB | 17.3 ns | 20.2 ns | 1.2x |
| 256 MB | 18.4 ns | **37.6 ns** | **2.0x** |
| 512 MB | 16.5 ns | **60.4 ns** | **3.7x** |
| 1 GB | 40.8 ns | 95.8 ns | 2.3x |
| 2 GB | 44.0 ns | 115.9 ns | 2.6x |
| 4 GB | 108.2 ns | 128.8 ns | 1.2x |

TLB tax at equal cache footprint (packed vs page-strided, 1 thread): **~2.2-2.7x** once past TLB
reach, peaking at **7.2x** (128 KB of data: 0.8 ns packed vs 5.8 ns paged).

**Three conclusions this pins down:**
1. **L1 dTLB reach is ~128 entries = ~2 MB.** The 128 -> 256 page cliff (0.7 -> 3.1 ns) happens while
   the touched data (16 -> 32 KB) still fits L1d (128 KB), so it cannot be a cache effect.
2. **The `-t8` penalty is shared hardware, not data sharing.** Each thread chases its *own private
   mapping*; nothing is shared but the L2/SLC, the page walkers and DRAM. Yet 8 threads cost 2.0-3.7x
   per access in the 256 MB - 2 GB range. This reproduces, in 60 lines and without the aligner, the
   wall documented in `docs/optimization-roadmap.md`.
3. **Widening the TLB's reach is not available to us** (probe 1), so the only lever left is making the
   random accesses *local* rather than making the TLB bigger.

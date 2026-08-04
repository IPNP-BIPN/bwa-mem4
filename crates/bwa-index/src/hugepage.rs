//! Ask the kernel to back the index's big random-access arrays with huge pages.
//!
//! # Why this exists
//!
//! Seeding is ~78% of single-end wall, and it is a random walk over `cp_occ`: 6.2 GB on GRCh38,
//! touched one 64-byte cache line at a time at addresses the FM recurrence *computes*, so no
//! prefetcher can help and every access is a fresh page-table lookup. With Linux's 4 KiB base page
//! a 64-entry L1 dTLB covers **256 KiB** of that 6.2 GB array, i.e. essentially nothing: virtually
//! every `get_occ` pays a hardware page-table walk on top of its cache miss.
//!
//! This is also why the effect is invisible in this project's own history of measurements. It is
//! **latency**, not bandwidth (`docs/optimization-roadmap.md` measured 7.2 GB/s, ~20% of one core,
//! and concluded correctly that we are not bandwidth-bound; a TLB-walk stall consumes no
//! bandwidth), and it is **Linux-specific**: macOS on arm64 uses a 16 KiB base page, four times the
//! reach, and offers no `MADV_HUGEPAGE` equivalent. Every number in `docs/perf-levers.md` was taken
//! on an M-series Mac, so none of them could have shown it.
//!
//! It should also scale *against* us with thread count, which is the shape of the deficit we are
//! chasing: the seeding lockstep keeps `W = 16` FM walks in flight **per thread**, so `-t16` has
//! 256 concurrent page-table walks contending for a shared L2 TLB and a fixed number of hardware
//! page walkers, where `-t1` has 16 that the out-of-order engine hides.
//!
//! `fg-labs/bwa-mem3` does exactly this (`reference/bwa-mem3-cpp/src/bwa_madvise.h`, applied to
//! `cp_occ`, both sampled-SA arrays and `pac`), which is why it looked like a differentiator.
//!
//! # MEASURED: the huge pages matter enormously, and this hint is not what gets them
//!
//! Measured 2026-07-28 on Linux aarch64 (4 KiB base page, kernel 6.12), GRCh38, 500k pairs, PE,
//! `-K` 10M, the `align` stage from `BWA4_STAGE_TIME` (which is the stage this targets: the random
//! walk over `cp_occ`), best of 3:
//!
//! | THP mode | peak `AnonHugePages` | `align` at `-t12` |
//! |---|---|---|
//! | `never` | 0 MB | **5.714s** |
//! | `madvise`, hint ON | 9866 MB | 3.456s |
//! | `madvise`, hint OFF (`BWA4_NO_HUGEPAGE=1`) | **9866 MB** | **3.309s** |
//!
//! Two conclusions, and the second is the load-bearing one.
//!
//! **Huge pages are worth ~1.7x on seeding.** The TLB reasoning above is right, and on Linux this
//! is one of the largest single effects anywhere in the aligner. Anything that breaks THP on a
//! deployment (`transparent_hugepage=never`, or a fragmented host where compaction keeps failing)
//! costs that much.
//!
//! **This hint is not what obtains them.** `AnonHugePages` is identical to the kilobyte with the
//! hint on and off, because **mimalloc already hints THP on the arenas it serves these allocations
//! from** (`#[global_allocator]`, `crates/bwa-cli/src/main.rs`). The wall difference between the two
//! arms is noise, in both directions across reps. So the fork's `bwa_madvise.h` cannot explain any
//! part of its Graviton advantage either: it also ships mimalloc, so it was already getting huge
//! pages before that header existed.
//!
//! This is the fifth lever in this project killed by existing machinery having already eaten it
//! (after LISA, the flat SA, minibwa's 10-mer cache and the `get_sa_batch` prefetch).
//!
//! # Why the call is kept anyway
//!
//! It is retained, not deleted, for one reason: `bwa-mem4-index` is published as a LIBRARY, and a
//! consumer using the system allocator gets none of mimalloc's hinting. For them this is the 1.7x.
//! Inside our own binary it is one ignored syscall per array at load time and nothing else. Do not
//! expect it to move a number in `bwa-mem4` itself, and do not re-measure it hoping otherwise.
//!
//! # Why it cannot change output
//!
//! `madvise` is a hint to the virtual-memory subsystem about page *size*. It moves no data, changes
//! no virtual address, and changes no byte at any address: `cp_occ[i]` reads the identical value
//! from the identical pointer either way. Failure is ignored on purpose, because "this kernel has
//! no THP", "this range is not 2 MiB-aligned" and "this is a file mapping on a filesystem that
//! cannot do huge pages" are all normal, not errors. Byte-identity is therefore structural rather
//! than something the gate has to establish, though `scripts/oracle_diff.sh` covers it anyway.

/// `MADV_HUGEPAGE`, from `include/uapi/asm-generic/mman-common.h`. The value is 14 on every Linux
/// architecture that defines it (it lives in the *generic* header, not a per-arch one), so x86_64
/// and aarch64 agree and no per-target constant is needed.
#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

/// `BWA4_NO_HUGEPAGE=1` suppresses the hint, so the A/B can be run on ONE binary.
///
/// Without this there is no way to measure the lever: rebuilding to remove the call would confound
/// the hint with whatever else the compiler did differently. Presence-only, read once and cached,
/// matching every other `BWA4_*` gate in the tree. It is a MEASUREMENT switch and cannot change
/// output in either position, which is the whole point of the module.
#[cfg(target_os = "linux")]
fn disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("BWA4_NO_HUGEPAGE").is_some())
}

#[cfg(target_os = "linux")]
extern "C" {
    /// libc's `madvise(2)`. Declared here rather than pulled from the `libc` crate so this costs
    /// the workspace no new dependency; the signature is fixed by POSIX and the Linux uapi.
    fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
}

/// Hint that `[ptr, ptr + len)` should be backed by transparent huge pages.
///
/// Best-effort and infallible by design: the return value is discarded because every failure mode
/// is a legitimate configuration rather than a fault (see the module docs). On non-Linux targets
/// this compiles to nothing.
///
/// # Parameters
///
/// * `ptr`: start of the range. Need not be page-aligned; the kernel rounds down. Only the
///   2 MiB-aligned interior of the range can actually be promoted, which is why this is worth
///   calling on multi-gigabyte arrays and pointless on small ones.
/// * `len`: length in BYTES.
///
/// # Safety
///
/// `ptr` must point to `len` bytes of memory owned by the caller and valid for the duration of the
/// call. Nothing is read or written through it here: the pointer is passed to the kernel purely as
/// the identity of a VMA range.
// Rust: `unsafe fn` puts the obligation on the CALLER, and the `# Safety` section above is the
// contract it must satisfy. Every call site therefore needs its own `unsafe` block, which is the
// mechanism that makes the obligation visible where it is actually discharged rather than buried
// here. Note what this function does NOT do: it never reads or writes through the pointer. It hands
// the address and length to the kernel as the identity of a memory range, so the only thing the
// caller really promises is that the range exists for the duration of the call.
//
// The `#[cfg]` pair below means this compiles to an empty function on macOS: `madvise` with
// `MADV_HUGEPAGE` is Linux-only, and there is nothing to fall back to.
pub unsafe fn advise_hugepage(ptr: *const u8, len: usize) {
    #[cfg(target_os = "linux")]
    {
        if !ptr.is_null() && len > 0 && !disabled() {
            // Cast away const: `madvise` takes `void *` by POSIX convention even though this
            // particular advice does not modify the range.
            madvise(ptr as *mut core::ffi::c_void, len, MADV_HUGEPAGE);
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Referenced so the parameters are not dead on macOS/Windows.
        let _ = (ptr, len);
    }
}

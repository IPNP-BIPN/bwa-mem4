//! `BWA4_STAGE_ALLOC=1` probe: how many bytes each pipeline stage ALLOCATES, and how many are
//! live at the peak.
//!
//! Issue #25 asks why a batch costs us close to twice the fork's resident memory at the same `-K`.
//! Every measurement taken so far has been a SNAPSHOT: peak RSS from `/usr/bin/time`, or a periodic
//! sampler. Both were misleading, and the roadmap records exactly how. On macOS, pages returned
//! with `madvise(MADV_FREE)` stay counted until the system wants them back, so peak RSS there is
//! the ceiling of everything ever allocated rather than anything that was ever live at one moment;
//! two correctness-preserving fixes aimed at live memory (`shrink_to_fit`, then `reserve_exact`)
//! measured +0.95 GB and nil respectively, because they moved live bytes and the metric was not
//! measuring live bytes. On Linux the sawtooth returns and the gap against the fork is real
//! (1.71x at the default `-K`, 1.93x pinned), so the remaining question is narrow: WHICH stage
//! commits the bytes, and are they still live when the peak happens.
//!
//! This probe answers both, from inside the allocator rather than from outside the process:
//!
//! - **Volume, per stage.** Every `alloc` is credited to whichever stage the pipeline is in. This
//!   is the number the roadmap asked for, "un compteur d'octets alloues, pas des snapshots": it
//!   sees a buffer that is allocated and freed inside one stage, which no sampler can catch and
//!   which on macOS still raises the reported ceiling.
//! - **Live bytes, exactly.** `dealloc` gets the layout back, so allocated minus freed is the true
//!   live total at any instant, and its running maximum is the real peak. Comparing that peak with
//!   peak RSS separates "our structures are twice as big" from "the allocator is holding pages".
//!   Those two findings have completely different fixes, and no measurement so far has told them
//!   apart.
//! - **Size classes.** A log2 histogram of request sizes, so 53 bytes per input base can be read as
//!   "many small requests" or "a few large ones" without guessing.
//!
//! # What it costs, and why it is not a feature flag
//!
//! Disabled, every allocation pays one relaxed load of an `AtomicBool` and an untaken branch. That
//! sounds free and is not, because it is per ALLOCATION rather than per batch like the other
//! probes, and this pipeline allocates 167M times per 500k GIAB pairs: interleaved A/B, min of 6 on
//! a quiet host, 12.49 s -> 12.56 s real and 95.14 s -> 95.67 s user, i.e. +0.5%. So the wrapper
//! lives behind the `stage-alloc` feature and the shipped binary does not carry it; everything else
//! here is per stage or per batch and compiles in unconditionally, doing nothing.
//! Enabled the probe is far from free (four relaxed read-modify-writes per allocation, on a path the
//! pipeline takes tens of millions of times), so an instrumented run's WALL TIME is not a
//! measurement of anything. Its byte counts are.
//!
//! # The one behaviour change the probe forces
//!
//! Stage attribution uses a single process-wide tag, because the threads that allocate the most are
//! rayon workers inside a parallel stage and there is nowhere to thread a parameter through to
//! them. That tag is only unambiguous with ONE batch in flight, so `run_pipeline` retires each
//! batch before starting the next when this probe is on. Output is unaffected (the pipeline already
//! guarantees batch order; only the overlap is dropped), but it is a second reason not to read wall
//! time from an instrumented run.
//!
//! The reader and writer threads run concurrently with the stages regardless, so they are tagged by
//! THREAD instead: each sets a thread-local role on entry and every allocation it makes lands in
//! its own bucket. Work the reader hands to the rayon pool (parallel inflate) is the one known
//! blind spot: it lands in whatever stage the aligner is in, which inflates that stage's volume.
//!
//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `unsafe impl GlobalAlloc for T` | the trait a type implements to become the program's allocator. `unsafe` because the compiler cannot check the contract (return a block of the requested size and alignment, never return the same block twice). |
//! | `#[global_allocator]` | the attribute, applied at a `static`, that installs such a type process-wide. Declared in `main.rs`; this file only provides the type. |
//! | `Layout` | the size and alignment of one request. `dealloc` is handed the same `Layout` the allocation was made with, which is why freed bytes can be counted exactly without a side table. |
//! | `AtomicI64` | an integer that several threads may update at once without a lock. Signed here, not unsigned, because allocations made before the probe was armed are freed after it, so the live counter legitimately dips below zero. |
//! | `fetch_add` / `fetch_sub` / `fetch_max` | read-modify-write in one indivisible step. `fetch_max` is what keeps a running maximum without a compare-and-swap loop. |
//! | `Ordering::Relaxed` | the weakest memory ordering: the counter itself stays correct, but no other memory is synchronised by touching it. Right for statistics, wrong for locks. |
//! | `thread_local!` + `const { ... }` | per-thread storage, initialised at compile time. The `const` form matters here: a lazily initialised thread-local could ALLOCATE on first touch, and allocating inside the allocator is an infinite regress. |
//! | `try_with` | reads a thread-local without panicking if it has already been destroyed, which happens during thread teardown, i.e. exactly while the last deallocations are running. |
//! | `Drop` | the destructor. [`StageGuard`] uses it to restore the previous stage tag, so a stage cannot leak past its own scope even on an early return. |

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8};

use crate::stage_time::{Stage, N_STAGES};

/// Bucket for allocations made by the FASTQ reader thread.
const B_READER: u8 = N_STAGES as u8;
/// Bucket for allocations made by the SAM/BAM writer thread.
const B_WRITER: u8 = N_STAGES as u8 + 1;
/// Bucket for everything outside a named stage: index load, header emission, teardown, and any
/// thread that is neither the reader, the writer, nor inside a `stage_time::measure`.
const B_UNSTAGED: u8 = N_STAGES as u8 + 2;
/// Number of buckets, i.e. the accumulator arrays' length.
const N_BUCKETS: usize = N_STAGES + 3;

/// Bucket labels. The first [`N_STAGES`] mirror `stage_time`'s stage names so the two tables can be
/// read side by side; a mismatch would make one probe's `align` row mean something else in the
/// other, so they are asserted equal in the tests.
const EXTRA_NAMES: [&str; 3] = ["reader", "writer", "unstaged"];

/// Thread-local role, consulted before the global stage tag. `0` means "no role, use the stage".
const ROLE_NONE: u8 = 0;
/// Role value marking the reader thread.
pub const ROLE_READER: u8 = 1;
/// Role value marking the writer thread.
pub const ROLE_WRITER: u8 = 2;

/// Number of log2 size classes tracked. Class `k` holds requests whose bit length is `k`, i.e.
/// sizes in `[2^(k-1), 2^k)`. 31 covers 1 GB and above in the last bucket.
const N_CLASSES: usize = 32;

/// Whether the probe is armed. Set once by [`init`] from `BWA4_STAGE_ALLOC`, and read on every
/// allocation, so it is a plain atomic rather than a `OnceLock`: `OnceLock::get_or_init` would run
/// `std::env::var_os` on first touch, and that ALLOCATES, from inside the allocator.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether the pipeline must run one batch at a time. Cleared by `BWA4_STAGE_ALLOC=overlap`, which
/// trades correct per-stage attribution for a peak that reflects the SHIPPED pipeline.
static SERIALIZE: AtomicBool = AtomicBool::new(true);

/// The stage every non-reader, non-writer thread's allocations are credited to.
static STAGE: AtomicU8 = AtomicU8::new(B_UNSTAGED);

/// Bytes requested per bucket, summed over the run. Never decremented: this is volume, not
/// occupancy, and volume is what sets the ceiling on a platform that does not return pages.
static BYTES: [AtomicU64; N_BUCKETS] = [const { AtomicU64::new(0) }; N_BUCKETS];
/// Allocation calls per bucket, so the table can print a mean request size.
static CALLS: [AtomicU64; N_BUCKETS] = [const { AtomicU64::new(0) }; N_BUCKETS];
/// Bytes freed per bucket, credited to the bucket doing the FREEING, which is not in general the
/// one that allocated. Printed to show which stage hands memory back rather than to balance a
/// ledger, and the two columns deliberately do not have to match per bucket.
static FREED: [AtomicU64; N_BUCKETS] = [const { AtomicU64::new(0) }; N_BUCKETS];

/// Live bytes right now: everything allocated minus everything freed. Signed because allocations
/// predating [`init`] are freed after it.
static LIVE: AtomicI64 = AtomicI64::new(0);
/// Running maximum of [`LIVE`]. This is the number to compare against peak RSS.
static PEAK: AtomicI64 = AtomicI64::new(0);
/// Running maximum of [`LIVE`] observed while each bucket was current. The whole-run [`PEAK`] is
/// the largest of these; per bucket it answers "how high does live memory get during this stage",
/// which with [`BASELINE`] subtracted is the batch footprint the fork is being compared against.
static PEAK_IN: [AtomicI64; N_BUCKETS] = [const { AtomicI64::new(0) }; N_BUCKETS];

/// Live bytes at the moment [`mark_baseline`] was called, i.e. everything the index and startup
/// hold and no batch will ever free. Subtracted from the peaks so the table reports the part of
/// memory that scales with `-K`, which is the only part issue #25 is about.
static BASELINE: AtomicI64 = AtomicI64::new(0);

/// Which bucket was current when [`PEAK`] was last raised. Read without a lock alongside `PEAK`, so
/// it can lag by one update under contention; it names the stage the peak happened in, and that
/// answer does not change from one allocation to the next.
static PEAK_BUCKET: AtomicU8 = AtomicU8::new(B_UNSTAGED);

/// Requests per log2 size class, all buckets together.
static CLASS_CALLS: [AtomicU64; N_CLASSES] = [const { AtomicU64::new(0) }; N_CLASSES];
/// Bytes per log2 size class, all buckets together.
static CLASS_BYTES: [AtomicU64; N_CLASSES] = [const { AtomicU64::new(0) }; N_CLASSES];

/// Batches processed, so the table can print a per-batch mean.
static BATCHES: AtomicU64 = AtomicU64::new(0);
/// Input bases seen, so the table can print the bytes-per-base figure issue #25 is written in.
static BASES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// This thread's role, or [`ROLE_NONE`]. `const` initialised: a lazily initialised
    /// thread-local allocates on first touch, and this is read from inside the allocator.
    static ROLE: Cell<u8> = const { Cell::new(ROLE_NONE) };
}

/// Arm the probe if `BWA4_STAGE_ALLOC` is set in the environment.
///
/// Call once, as early in `main` as possible: allocations made before this point are not counted,
/// which for the `mem` path means clap's command line parsing and nothing else. It reads the
/// environment exactly once, on a thread where allocating is safe, which is the whole reason the
/// flag is not a `OnceLock` consulted lazily from the allocator.
///
/// Two modes, because they answer different questions and cannot both be answered by one run:
///
/// - `BWA4_STAGE_ALLOC=1` (or anything else): one batch in flight, so the per-stage columns are
///   exact. The peak-live column is then the footprint of ONE batch.
/// - `BWA4_STAGE_ALLOC=overlap`: the shipped two-batches-in-flight pipeline. The peak-live and
///   total columns are the real ones; the per-stage split is a blend of whatever two batches were
///   doing at the same time and must not be quoted.
pub fn init() {
    let Some(mode) = std::env::var_os("BWA4_STAGE_ALLOC") else {
        return;
    };
    // The counting allocator is only installed by `main.rs` under the `stage-alloc` feature, which
    // is off by default (it costs a measured +0.5% even disarmed). Arming the flag without it would
    // print a table of zeros and serialise the pipeline for nothing, so say so instead.
    if !cfg!(feature = "stage-alloc") {
        eprintln!(
            "[stage-alloc] BWA4_STAGE_ALLOC is set but this binary has no counting allocator; \
             rebuild with `cargo build --release -p bwa-mem4 --features stage-alloc`. Ignored."
        );
        return;
    }
    ENABLED.store(true, Relaxed);
    if mode == *"overlap" {
        SERIALIZE.store(false, Relaxed);
    }
}

/// Whether the probe is armed.
///
/// # Returns
///
/// True if allocations are being counted, i.e. `BWA4_STAGE_ALLOC` was set and [`init`] has run.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// Restores the previous stage tag when dropped. Returned by [`enter`]; not constructible
/// otherwise, so a stage tag cannot be set without a matching restore.
pub struct StageGuard {
    /// The tag to put back on drop.
    prev: u8,
    /// False when the probe is off, in which case dropping does nothing at all.
    active: bool,
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        if self.active {
            STAGE.store(self.prev, Relaxed);
        }
    }
}

/// Credit every subsequent allocation to `stage`, until the returned guard drops.
///
/// Called from `stage_time::measure`, so the two probes always agree on where a stage starts and
/// ends: one instrumentation point, two measurements. Nesting is safe, since the guard restores
/// whatever tag it displaced rather than clearing it.
///
/// # Parameters
///
/// - `stage`: the stage the calling thread is entering.
///
/// # Returns
///
/// A guard that restores the previous tag. Bind it (`let _tag = ...`), do not discard it with `_`,
/// or the tag is restored immediately.
#[inline]
pub fn enter(stage: Stage) -> StageGuard {
    if !enabled() {
        return StageGuard {
            prev: B_UNSTAGED,
            active: false,
        };
    }
    StageGuard {
        prev: STAGE.swap(stage as u8, Relaxed),
        active: true,
    }
}

/// Tag the CALLING thread with a role, so its allocations are credited to it whatever stage the
/// aligner happens to be in. Used by the reader and writer threads, which run concurrently with
/// every stage and would otherwise pollute whichever one was current.
///
/// # Parameters
///
/// - `role`: [`ROLE_READER`] or [`ROLE_WRITER`].
pub fn set_role(role: u8) {
    if !enabled() {
        return;
    }
    let _ = ROLE.try_with(|r| r.set(role));
}

/// Freeze the current live total as the run's baseline: the index, the reference and everything
/// else loaded once and held for the whole run.
///
/// Call it after the index load and before the first batch. Without it the peak column is 10 GB of
/// human genome on every row and says nothing about the per-batch structures, which are the entire
/// subject of issue #25.
pub fn mark_baseline() {
    if !enabled() {
        return;
    }
    BASELINE.store(LIVE.load(Relaxed), Relaxed);
}

/// Record that one more batch has been processed, so the table can divide by it.
pub fn count_batch() {
    if !enabled() {
        return;
    }
    BATCHES.fetch_add(1, Relaxed);
}

/// Record a batch's input size, so the table can print bytes allocated per input base, which is the
/// unit issue #25 and the roadmap's RAM model are both written in.
///
/// # Parameters
///
/// - `bases`: the batch's total sequence length, both mates included for paired-end.
pub fn note_bases(bases: u64) {
    if !enabled() {
        return;
    }
    BASES.fetch_add(bases, Relaxed);
}

/// Whether `run_pipeline` must retire each batch before starting the next.
///
/// The stage tag is process-wide, so two batches in flight would each be overwriting the other's
/// stage and every per-stage number would be a blend of two. Serialising costs wall time and no
/// bytes, which is the right trade for a probe that measures bytes.
///
/// # Returns
///
/// True when the probe is armed, in which case the pipeline drops its batch overlap.
pub fn serialize_batches() -> bool {
    enabled() && SERIALIZE.load(Relaxed)
}

/// Which bucket the calling thread's allocations belong to.
#[inline]
fn bucket() -> usize {
    let role = ROLE.try_with(|r| r.get()).unwrap_or(ROLE_NONE);
    match role {
        ROLE_READER => B_READER as usize,
        ROLE_WRITER => B_WRITER as usize,
        _ => STAGE.load(Relaxed) as usize,
    }
}

/// Log2 size class of a request: the number of significant bits in `size`, saturated at the last
/// class. Size 1 lands in class 1, 8 in class 4, and everything from 1 GB up shares class 31.
#[inline]
fn class_of(size: usize) -> usize {
    ((usize::BITS - size.leading_zeros()) as usize).min(N_CLASSES - 1)
}

/// Count one allocation of `size` bytes against the calling thread's bucket, and raise the live
/// peak if this request set a new one.
#[inline]
fn record_alloc(size: usize) {
    let b = bucket();
    BYTES[b].fetch_add(size as u64, Relaxed);
    CALLS[b].fetch_add(1, Relaxed);
    let c = class_of(size);
    CLASS_CALLS[c].fetch_add(1, Relaxed);
    CLASS_BYTES[c].fetch_add(size as u64, Relaxed);
    let live = LIVE.fetch_add(size as i64, Relaxed) + size as i64;
    PEAK_IN[b].fetch_max(live, Relaxed);
    if live > PEAK.fetch_max(live, Relaxed) {
        PEAK_BUCKET.store(b as u8, Relaxed);
    }
}

/// Count one deallocation of `size` bytes against the bucket doing the freeing.
#[inline]
fn record_free(size: usize) {
    FREED[bucket()].fetch_add(size as u64, Relaxed);
    LIVE.fetch_sub(size as i64, Relaxed);
}

/// The global allocator wrapper: forwards everything to `A` and counts it on the way through.
///
/// Generic over the inner allocator so the same wrapper serves the default build (mimalloc) and
/// `--no-default-features` (the system allocator), and so the probe cannot silently measure a
/// different allocator from the one that shipped.
pub struct Counting<A> {
    /// The allocator doing the actual work.
    inner: A,
}

impl<A> Counting<A> {
    /// Wrap `inner`. `const` so it can initialise a `static`, which is what `#[global_allocator]`
    /// requires.
    ///
    /// # Parameters
    ///
    /// - `inner`: the allocator to forward to.
    ///
    /// # Returns
    ///
    /// The wrapper, ready to be installed as the global allocator.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

// SAFETY: every method forwards to `self.inner`, which upholds the `GlobalAlloc` contract, and
// returns its pointer unchanged. The counting is pure arithmetic on atomics: it allocates nothing
// (a `const`-initialised thread-local and atomic adds, no formatting, no `getenv`), so it cannot
// re-enter the allocator, and it never inspects or writes the returned block.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = self.inner.alloc(layout);
        if enabled() && !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if enabled() {
            record_free(layout.size());
        }
        self.inner.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = self.inner.alloc_zeroed(layout);
        if enabled() && !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }

    // Counted as a free of the old block plus a fresh allocation of the whole new size, NOT as the
    // difference. That is deliberate and it is the pessimistic reading: a growing `Vec` normally
    // gets a new block and copies, so both sizes are committed at once, and it is precisely that
    // transient double footprint the roadmap suspects of setting the macOS ceiling. When the
    // allocator does grow in place the volume column overstates by the old size; the LIVE column is
    // exact either way, since it moves by the difference.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = self.inner.realloc(ptr, layout, new_size);
        if enabled() && !p.is_null() {
            record_free(layout.size());
            record_alloc(new_size);
        }
        p
    }
}

/// Print the per-bucket allocation table to stderr, once, at the end of a run. A no-op unless
/// [`enabled`].
///
/// Unlike `stage_time::dump` this may be called from any thread: every accumulator here is a plain
/// atomic, because the threads being measured are the rayon pool and not the caller.
pub fn dump() {
    if !enabled() {
        return;
    }
    let names: Vec<&str> = crate::stage_time::names()
        .iter()
        .copied()
        .chain(EXTRA_NAMES)
        .collect();
    let bytes: [u64; N_BUCKETS] = std::array::from_fn(|i| BYTES[i].load(Relaxed));
    let freed: [u64; N_BUCKETS] = std::array::from_fn(|i| FREED[i].load(Relaxed));
    let calls: [u64; N_BUCKETS] = std::array::from_fn(|i| CALLS[i].load(Relaxed));
    let total: u64 = bytes.iter().sum();
    let batches = BATCHES.load(Relaxed).max(1);
    let bases = BASES.load(Relaxed);
    let gb = |v: u64| v as f64 / 1e9;

    let baseline = BASELINE.load(Relaxed);
    if !SERIALIZE.load(Relaxed) {
        eprintln!(
            "[stage-alloc] overlap mode: two batches in flight, so the TOTAL and peak rows are the \
             shipped pipeline's and the per-stage split is a blend of two batches"
        );
    }
    eprintln!(
        "[stage-alloc] {batches} batches, {bases} input bases, peak live {:.3} GB (in {}), \
         of which {:.3} GB is the index and other run-long allocations",
        PEAK.load(Relaxed).max(0) as f64 / 1e9,
        names[(PEAK_BUCKET.load(Relaxed) as usize).min(N_BUCKETS - 1)],
        baseline.max(0) as f64 / 1e9,
    );
    eprintln!(
        "[stage-alloc] {:>13}  {:>10}  {:>6}  {:>10}  {:>12}  {:>9}  {:>9}  {:>10}",
        "bucket", "alloc_GB", "%", "freed_GB", "calls", "mean_B", "B_per_base", "peak_live_GB"
    );
    for (i, name) in names.iter().enumerate() {
        // A bucket the run never entered (the paired-end stages under single-end) says nothing.
        if bytes[i] == 0 && freed[i] == 0 {
            continue;
        }
        eprintln!(
            "[stage-alloc] {:>13}  {:>10.3}  {:>5.1}%  {:>10.3}  {:>12}  {:>9.0}  {:>9.1}  {:>10.3}",
            name,
            gb(bytes[i]),
            100.0 * bytes[i] as f64 / total.max(1) as f64,
            gb(freed[i]),
            calls[i],
            bytes[i] as f64 / calls[i].max(1) as f64,
            bytes[i] as f64 / bases.max(1) as f64,
            // Above the baseline, so this column is the batch's own footprint. A stage that never
            // ran leaves it at zero, and one that ran entirely below the baseline clamps there.
            (PEAK_IN[i].load(Relaxed) - baseline).max(0) as f64 / 1e9,
        );
    }
    eprintln!(
        "[stage-alloc] {:>13}  {:>10.3}  {:>5.1}%  {:>10.3}  {:>12}  {:>9.0}  {:>9.1}  {:>10.3}",
        "TOTAL",
        gb(total),
        100.0,
        gb(freed.iter().sum::<u64>()),
        calls.iter().sum::<u64>(),
        total as f64 / calls.iter().sum::<u64>().max(1) as f64,
        total as f64 / bases.max(1) as f64,
        (PEAK.load(Relaxed) - baseline).max(0) as f64 / 1e9,
    );
    // Size classes, so "53 bytes per base" can be read as a shape rather than a scalar. Only
    // non-empty classes print, and the row is the closed interval the class covers.
    eprintln!(
        "[stage-alloc] {:>13}  {:>10}  {:>6}  {:>12}",
        "size_class", "alloc_GB", "%", "calls"
    );
    for c in 0..N_CLASSES {
        let (b, n) = (CLASS_BYTES[c].load(Relaxed), CLASS_CALLS[c].load(Relaxed));
        if n == 0 {
            continue;
        }
        let lo = if c == 0 { 0 } else { 1usize << (c - 1) };
        eprintln!(
            "[stage-alloc] {:>13}  {:>10.3}  {:>5.1}%  {:>12}",
            format!("{}..{}", lo, (1usize << c) - 1),
            gb(b),
            100.0 * b as f64 / total.max(1) as f64,
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two probes must label the same stage the same way, or one table's `align` row means
    /// something else in the other.
    #[test]
    fn bucket_names_extend_the_stage_names() {
        assert_eq!(crate::stage_time::names().len(), N_STAGES);
        assert_eq!(N_BUCKETS, N_STAGES + EXTRA_NAMES.len());
    }

    /// Size classes are the bit length of the request, which is what makes the printed interval
    /// `2^(c-1)..2^c - 1` correct.
    #[test]
    fn size_classes_are_bit_lengths() {
        assert_eq!(class_of(1), 1);
        assert_eq!(class_of(2), 2);
        assert_eq!(class_of(3), 2);
        assert_eq!(class_of(8), 4);
        assert_eq!(class_of(usize::MAX), N_CLASSES - 1);
    }

    /// One test, not three: the accumulators are process-wide statics, so separate `#[test]`
    /// functions would race each other under the test harness's thread pool.
    ///
    /// It arms the probe by hand rather than through [`init`], which would need the environment
    /// variable set for the whole test binary and would then count the harness's own allocations.
    #[test]
    fn allocations_land_in_the_current_stage_and_the_peak_is_the_live_maximum() {
        ENABLED.store(true, Relaxed);

        // Outside any stage.
        record_alloc(1000);
        assert_eq!(BYTES[B_UNSTAGED as usize].load(Relaxed), 1000);

        {
            let _tag = enter(Stage::Align);
            record_alloc(500);
            assert_eq!(BYTES[Stage::Align as usize].load(Relaxed), 500);
            // Live is 1500 here, and that is the run's high-water mark.
            assert_eq!(PEAK.load(Relaxed), 1500);
            assert_eq!(PEAK_BUCKET.load(Relaxed), Stage::Align as u8);
        }
        // The guard put the previous tag back, so this lands outside the stage again.
        record_free(500);
        record_alloc(100);
        assert_eq!(BYTES[B_UNSTAGED as usize].load(Relaxed), 1100);
        assert_eq!(FREED[B_UNSTAGED as usize].load(Relaxed), 500);
        // 1000 + 500 - 500 + 100: freeing does not lower the peak.
        assert_eq!(LIVE.load(Relaxed), 1100);
        assert_eq!(PEAK.load(Relaxed), 1500);

        // A thread role wins over the stage tag, which is what keeps the reader out of `align`.
        {
            let _tag = enter(Stage::Align);
            set_role(ROLE_READER);
            record_alloc(7);
            assert_eq!(BYTES[B_READER as usize].load(Relaxed), 7);
            ROLE.with(|r| r.set(ROLE_NONE));
        }

        // Disarmed, `enter` is inert: the tag stays where it was, so the guard cannot move an
        // uninstrumented run's allocations into a stage bucket.
        ENABLED.store(false, Relaxed);
        let _tag = enter(Stage::Rescue);
        assert_eq!(STAGE.load(Relaxed), B_UNSTAGED);
    }
}

//! `BWA4_STAGE_TIME=1` probe: where one batch's WALL clock goes, stage by stage.
//!
//! The three probes that already exist (`BWA4_CHAIN_TIME`, `BWA4_MATESW_TIME`, `BWA4_TRAFFIC`)
//! all accumulate **CPU** nanoseconds summed across every rayon worker, via relaxed atomics on
//! the hot path. That answers "how much total work did stage X do", which is the right question
//! for a kernel. It cannot answer "was the pool idle", which is the only question that matters
//! for a *scaling* deficit: a stage that runs alone on the main thread costs the same CPU ns at
//! `-t1` and `-t16` while costing 16x the parallel-equivalent wall.
//!
//! So this probe is deliberately the opposite shape:
//!
//! - **Wall clock, not CPU.** One `Instant` pair per stage per batch.
//! - **Main thread only.** Every recorded stage is entered and left by the thread that runs
//!   `run_pipeline`'s loop, so the accumulators are a `thread_local` `Cell` array: no atomics,
//!   no contention, and nothing at all on any worker's hot path.
//! - **Whole-batch granularity.** A rayon `par_iter` shows up as the wall time of its fork/join,
//!   which is exactly the barrier cost we are hunting, stragglers included.
//!
//! Disabled (the default) each site costs one cached bool load and an untaken branch, so this
//! stays compiled into the shipped binary rather than living behind a feature flag.
//!
//! The stages tile the batch loop without gaps: `sum(stages) + epsilon == total wall`, where the
//! epsilon is index load and header emission outside the loop. A stage that a given path never
//! enters (the paired-end-only ones, under single-end) simply stays at zero and is not printed.
//!
//! # Rust mechanics used in this file
//!
//! The design note above rests on one language choice, so it is worth spelling out. The other
//! probes in this tree accumulate into ATOMIC counters, because rayon workers write them from many
//! threads at once, and an atomic add is the cheapest thing that is still correct there. This probe
//! is only ever touched by the main thread, so it uses `thread_local` `Cell`s instead: a `Cell` is
//! plain mutable storage with no synchronisation whatsoever, and `thread_local` gives each thread
//! its own copy. That is why the disabled cost really is one bool load and an untaken branch, with
//! nothing on any worker's hot path.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `thread_local!` | declares storage of which every thread gets its own private copy. No thread can see another's, so no locking or atomics are needed to touch it. |
//! | `Cell<T>` | a container whose contents can be replaced even through a shared borrow. Ordinary Rust forbids that; `Cell` permits it precisely because it is single-threaded and cannot be shared across threads. |
//! | `.get()` / `.set(v)` | read and overwrite a `Cell`. Both are plain loads and stores. |
//! | `Instant` | a monotonic clock reading, unaffected by the wall clock being adjusted. Two of them subtract to a `Duration`. |
//! | `.elapsed()` | time since an `Instant` was taken. |
//! | `Duration::as_nanos()` | that span in nanoseconds. |
//! | `std::env::var_os(...)` | reads an environment variable, here `BWA4_STAGE_TIME`, without requiring it to be valid text. |
//! | `OnceLock<bool>` + `.get_or_init(...)` | a slot filled at most once, on first use, from any thread. It is what turns the environment lookup into a single read for the whole process, so a disabled probe costs a cached bool load rather than a `getenv` at every stage boundary. |
//! | `const { ... }` in an array repeat | forces the repeated element to be built at compile time. It is what allows an array of `Cell`s to be declared, since `Cell` cannot be cloned the way an ordinary repeated initialiser would require. |

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// One measured segment of the per-batch loop.
///
/// The discriminants are the index into the accumulator array, so the order here is the order the
/// table prints in, which is deliberately the order the pipeline executes in: reading the table
/// top to bottom walks one batch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Blocked in `batch_rx.recv()`, i.e. the aligner is starved by the reader thread. A large
    /// value here means input (decompress + parse + per-read allocation) is the bottleneck and no
    /// amount of compute parallelism will help.
    WaitRead = 0,
    /// ASCII bases to nt4 codes. Serial on the main thread today, and O(`-K` bases), so it grows
    /// with `-t` under the default `-K = 10M * threads`.
    Encode = 1,
    /// `batched_regs`: seeding, `get_sa`, chaining and banded extension. The one genuinely
    /// parallel stage, and the one whose share should dominate if scaling were healthy.
    Align = 2,
    /// Paired-end de-interleave of the flat interleaved code/region lists back into pairs.
    Deinterleave = 3,
    /// Per-read dedup and primary patching (`mem_sort_dedup_patch` + `stamp_is_alt`), parallel.
    DedupPrep = 4,
    /// `mem_pestat`: the per-batch insert-size model. Serial, and it sorts the whole batch.
    Pestat = 5,
    /// Batched mate rescue. Parallel, and per `docs/perf-levers.md` it is 47-64% of paired-end
    /// compute, so its fork/join imbalance is worth this much attention.
    Rescue = 6,
    /// SAM record formatting (`finish_se` / `mem_sam_pe`), parallel.
    SamEmit = 7,
    /// Blocked in `sam_tx.send()`, i.e. the aligner is throttled by the writer thread.
    ///
    /// There is deliberately no `Concat` stage between this and [`Stage::SamEmit`]: the per-record
    /// pieces go to the writer as they are, so the single-threaded concatenation that used to sit
    /// here no longer exists (see `run_pipeline`).
    WaitWrite = 8,
}

/// Human-readable names, indexed by [`Stage`]'s discriminant.
const NAMES: [&str; N_STAGES] = [
    "wait_read",
    "encode",
    "align",
    "deinterleave",
    "dedup_prep",
    "pestat",
    "rescue",
    "sam_emit",
    "wait_write",
];

/// Number of stages, i.e. the accumulator array's length. Kept next to [`NAMES`] so a new stage
/// that forgets to add its name fails to compile rather than printing a wrong label.
///
/// Public because [`crate::stage_alloc`] tiles the same stages with three buckets of its own and
/// must agree with this count exactly.
pub const N_STAGES: usize = 9;

/// The stage labels, in discriminant order.
///
/// Exists so [`crate::stage_alloc`] prints the same name for the same stage rather than keeping a
/// second copy that could drift.
///
/// # Returns
///
/// The [`NAMES`] table, indexed by [`Stage`]'s discriminant.
pub fn names() -> &'static [&'static str; N_STAGES] {
    &NAMES
}

/// Nanoseconds accumulated per stage, across every thread.
///
/// This used to be a `thread_local`, on the belief that only the main thread records. That was
/// wrong and it silently lost almost everything: `run_pipeline` runs each batch's `process` on a
/// `scope.spawn`ed thread, so every stage between `encode` and `sam_emit` was credited to a thread
/// that then died, and the table showed only `wait_read`, `wait_write` and a giant `unaccounted`.
/// A run that spent 95% of its time somewhere reported that somewhere as "index load, header,
/// teardown".
///
/// Atomics rather than a lock: the probe is off unless `BWA4_STAGE_TIME` is set, and when it is on
/// each stage costs one relaxed add per batch, not per read.
static NS: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];

/// Batches processed, so the table can print a per-batch mean.
static BATCHES: AtomicU64 = AtomicU64::new(0);

/// Whether `BWA4_STAGE_TIME` is set. Read once and cached, matching `chain_time::enabled`: the
/// variable cannot be toggled mid-process and the disabled path is one atomic load.
///
/// # Returns
///
/// True if the probe should record and print timings.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_STAGE_TIME").is_some())
}

/// Add an already-measured duration to a stage. Used by the two call sites that time a blocking
/// channel operation, where wrapping the call in a closure would fight the borrow checker.
///
/// # Parameters
///
/// - `stage`: which accumulator to credit.
/// - `elapsed`: the measured wall time. Saturates rather than wrapping, which only matters for a
///   run longer than 584 years.
pub fn add(stage: Stage, elapsed: Duration) {
    if !enabled() {
        return;
    }
    NS[stage as usize].fetch_add(elapsed.as_nanos() as u64, Relaxed);
}

/// Time `f` and credit it to `stage`, returning whatever `f` returned.
///
/// This is the normal way to instrument a stage: it wraps the existing expression in place, so
/// the instrumented code keeps the same shape as the uninstrumented code and there is no
/// scope-based guard whose drop point has to be reasoned about.
///
/// When disabled it is a plain call to `f` with no `Instant` at all, so the probe cannot perturb
/// the measurement it is not taking.
///
/// # Parameters
///
/// - `stage`: which accumulator to credit.
/// - `f`: the work to measure. Runs exactly once either way.
///
/// # Returns
///
/// `f`'s return value, unchanged.
pub fn measure<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    // One instrumentation point, two probes: `BWA4_STAGE_ALLOC` credits this stage with every byte
    // allocated until the guard drops, so the wall-clock table and the allocation table can never
    // disagree about where a stage begins. Inert (one bool load) unless that probe is armed.
    let _tag = crate::stage_alloc::enter(stage);
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    add(stage, t0.elapsed());
    out
}

/// Record that one more batch has been processed, so the table can divide by it.
pub fn count_batch() {
    if !enabled() {
        return;
    }
    BATCHES.fetch_add(1, Relaxed);
}

/// Print the per-stage table to stderr, once, at the end of a run. A no-op unless [`enabled`].
///
/// Must be called from the SAME thread that recorded the stages (the main thread), because the
/// accumulators are thread-local. Call it after `run_pipeline` returns, so the reader and writer
/// have joined and the wall figure covers a finished run.
///
/// # Parameters
///
/// - `total_wall`: the whole `mem` run's elapsed time, used as the denominator for the share
///   column and to show how much of the run the loop did not account for (index load, header).
pub fn dump(total_wall: Duration) {
    if !enabled() {
        return;
    }
    // Per-stage seconds, snapshotted together so the table is internally consistent.
    let secs: [f64; N_STAGES] = std::array::from_fn(|i| NS[i].load(Relaxed) as f64 / 1e9);
    let batches = BATCHES.load(Relaxed).max(1);
    let wall = total_wall.as_secs_f64().max(1e-9);
    // Everything the batch loop accounted for; the remainder is startup (index load) and teardown.
    let accounted: f64 = secs.iter().sum();

    eprintln!("[stage-time] {batches} batches, {wall:.3}s total wall");
    eprintln!(
        "[stage-time] {:>13}  {:>9}  {:>7}  {:>11}",
        "stage", "wall_s", "%_run", "ms_per_batch"
    );
    for (i, name) in NAMES.iter().enumerate() {
        // Stages a given path never enters stay exactly zero; printing them would only pad the
        // table with rows that say nothing (all the paired-end rows, under single-end).
        if secs[i] == 0.0 {
            continue;
        }
        eprintln!(
            "[stage-time] {:>13}  {:>9.3}  {:>6.1}%  {:>11.1}",
            name,
            secs[i],
            100.0 * secs[i] / wall,
            1e3 * secs[i] / batches as f64,
        );
    }
    // Stages OVERLAP `wait_read`: the pipeline spawns batch N's `process` and immediately waits
    // for batch N+1, so the two run at once and the column can sum past 100%. The remainder is
    // therefore signed: positive means startup and teardown (chiefly the index load), negative
    // means the overlap exceeded them.
    eprintln!(
        "[stage-time] {:>13}  {:>9.3}  {:>6.1}%  (index load, header, teardown, minus overlap)",
        "unaccounted",
        wall - accounted,
        100.0 * (wall - accounted) / wall,
    );
}

/// `BWA4_BARRIER_TIME=1`: per-worker occupancy of each rayon fork/join region.
///
/// This exists because of one measurement. In Nils's regime (`-t16`, default `-K`, gzipped input,
/// 1M real GIAB pairs) we burn **21% less CPU than fg-labs/bwa-mem3** and yet finish in the same
/// wall time, 6 reps out of 6. Dividing CPU by wall gives the reason: the fork keeps ~13.4 of 16
/// threads busy, we keep ~10.8. Our per-core code is ahead and we hand the advantage back as idle
/// cores.
///
/// [`super::stage_time`] cannot see that: it times the MAIN thread, so a barrier where fifteen
/// workers finished early and one ran long looks exactly like a barrier where all sixteen ran the
/// whole time. This module times each WORKER inside the region and compares:
///
/// - `wall`: how long the region took.
/// - `busy_sum / workers`: what the region would have taken with perfect balance.
/// - `busy_max`: the slowest single worker, i.e. what the barrier actually waits for.
///
/// `busy_sum / (wall * workers)` is the occupancy of that region, and `1 - busy_max / wall` is how
/// much of the tail is scheduling rather than work. Both are ratios between workers inside ONE run,
/// so unlike wall-clock they are robust to the competing load on a developer machine.
pub mod barrier {
    use super::{Stage, N_STAGES};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    /// Upper bound on rayon workers this probe tracks; anything beyond falls in slot 0, which only
    /// blurs the probe, never the alignment.
    const MAX_WORKERS: usize = 128;

    /// Busy nanoseconds of each worker in the region currently being measured.
    static BUSY: [AtomicU64; MAX_WORKERS] = [const { AtomicU64::new(0) }; MAX_WORKERS];
    /// Summed region wall time per stage, over all batches.
    static WALL: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
    /// Summed worker-busy nanoseconds per stage (all workers, all batches).
    static SUM: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
    /// Summed slowest-worker nanoseconds per stage, one term per batch.
    static MAX: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];

    /// Whether `BWA4_BARRIER_TIME` is set. Read once and cached, so the hot path pays one bool load.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_BARRIER_TIME").is_some())
    }

    /// Time one worker's share of a parallel region and bank it against that worker's slot.
    ///
    /// # Parameters
    /// - `f`: the per-item body of the parallel iterator; its value is returned untouched.
    #[inline]
    pub fn worker<T>(f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let started = std::time::Instant::now();
        let out = f();
        let slot = rayon::current_thread_index()
            .unwrap_or(0)
            .min(MAX_WORKERS - 1);
        BUSY[slot].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        out
    }

    /// Run a whole parallel region, timing it and folding this batch's per-worker totals into the
    /// stage accumulators.
    ///
    /// # Parameters
    /// - `stage`: which region this is, for the report.
    /// - `f`: the region, normally a rayon iterator chain whose body is wrapped in [`worker`].
    pub fn region<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        for b in BUSY.iter() {
            b.store(0, Relaxed);
        }
        let started = std::time::Instant::now();
        let out = f();
        let wall = started.elapsed().as_nanos() as u64;
        let (mut sum, mut max) = (0u64, 0u64);
        for b in BUSY.iter() {
            let v = b.load(Relaxed);
            sum += v;
            max = max.max(v);
        }
        WALL[stage as usize].fetch_add(wall, Relaxed);
        SUM[stage as usize].fetch_add(sum, Relaxed);
        MAX[stage as usize].fetch_add(max, Relaxed);
        out
    }

    /// Print the per-region occupancy table once at end of run. No-op unless [`enabled`].
    pub fn dump() {
        if !enabled() {
            return;
        }
        let workers = rayon::current_num_threads().max(1) as f64;
        eprintln!(
            "[barrier] {:>13}  {:>9}  {:>9}  {:>9}  {:>6}  {:>6}",
            "region", "wall_s", "busy_s", "slowest_s", "occ%", "tail%"
        );
        for (i, name) in super::NAMES.iter().enumerate() {
            let (wall, sum, max) = (
                WALL[i].load(Relaxed) as f64 / 1e9,
                SUM[i].load(Relaxed) as f64 / 1e9,
                MAX[i].load(Relaxed) as f64 / 1e9,
            );
            if wall == 0.0 {
                continue;
            }
            eprintln!(
                "[barrier] {:>13}  {:>9.3}  {:>9.3}  {:>9.3}  {:>5.1}%  {:>5.1}%",
                name,
                wall,
                sum,
                max,
                // Occupancy: what fraction of the region's core-seconds were doing work.
                100.0 * sum / (wall * workers),
                // Tail: the part of the region nobody was working, even the slowest worker, i.e.
                // rayon scheduling and the join itself rather than imbalance.
                100.0 * (wall - max).max(0.0) / wall,
            );
        }
    }
}

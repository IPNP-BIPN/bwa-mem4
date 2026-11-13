//! Lane-batched local Smith-Waterman for **mate rescue** (`kswv` in bwa-mem2).
//!
//! Mate rescue realigns a mate read against an insert-size window when its pairing is missing. Each
//! such alignment is an independent full local SW ([`bwa_extend::ksw_align2`]) returning
//! `{score, qb, qe, tb, te, score2, te2}` (no CIGAR/traceback). bwa-mem2 vectorizes this
//! **inter-sequence**: many rescue jobs packed into SoA lanes (16 x u8 / 8 x i16 on NEON), each lane
//! a different job, length-sorted so lanes finish together. This mirrors [`crate::batched`] (seed
//! extension) but with the local-SW recurrence and the two-phase start recovery (`KSW_XSTART`).
//!
//! [`batched_ksw_align2`] returns one [`KswAlignResult`] per job, each byte-identical to
//! [`bwa_extend::ksw_align2`] on that job. The scalar per-job loop is the portable fallback and the
//! source of truth the NEON kernels are validated against (`matesw_equals_scalar`).
//!
//! # Rust mechanics used in this file
//!
//! Same machinery as [`crate::batched`] (see the crate note for `cfg` / feature detection /
//! `target_feature` / the kernel macro), with one addition that is worth calling out because it is a
//! precondition rather than a bound.
//!
//! `mat_is_standard(m, mat)` is checked BEFORE the vector path is taken. The scoring matrix is
//! supplied by the caller, and the SIMD kernels hard-code the standard bwa matrix's shape into their
//! arithmetic; on any other matrix the vector result would be wrong rather than merely slow. So the
//! dispatch is a conjunction: the ISA is present AND the matrix is the one the kernels assume. Fail
//! either and the scalar per-job loop runs, which is also the source of truth the kernels are
//! validated against (`matesw_equals_scalar`).
//!
//! The rest reads as it does in the sibling file: jobs are length-sorted so lanes finish together,
//! results are scattered back into input order, and every `unsafe` call carries a `SAFETY` note
//! naming both the detected feature and the width bound that keeps the lane arithmetic exact.
//!
//! # How this differs from [`crate::batched`], and why that is not an inconsistency
//!
//! Both files run affine-gap Smith-Waterman, but they port *different* C functions and the
//! differences are deliberate:
//!
//! | | [`crate::batched`] (seed extension) | this file (mate rescue) |
//! |---|---|---|
//! | C original | `ksw_extend2`, `ksw.cpp:432` | `ksw_u8`/`ksw_i16`, `ksw.cpp:111`/`:233` |
//! | banded? | yes, `[beg, end)` per row | no, the full query x target rectangle |
//! | gaps open from | `M` (`ksw.cpp:493-501`) | `H` (`ksw.cpp:168`, `:173`) |
//! | seeded with | `h0` from the seed | 0 (a true local alignment from nothing) |
//! | reports | `qle`/`tle`/`gscore` | `qb`/`qe`/`tb`/`te` plus a 2nd-best `score2`/`te2` |
//!
//! The gap-opening row is the trap. In seed extension, opening from `H` would let an insertion
//! abut a deletion for free and inflate `gscore`; here, opening from `H` is what the C actually
//! does (`h = _mm_subs_epu8(h, oe_del)` at `ksw.cpp:168`), so mirroring `ksw_extend2` instead would
//! be the bug. Each kernel mirrors its own original, and neither convention is "the right one".
//!
//! # Recurrence used here
//!
//! ```text
//!   H(i,j) = max( 0, H(i-1,j-1) + S(target[i], query[j]), E(i,j), F(i,j) )
//!   E(i+1,j) = max( 0, E(i,j) - e_del , H(i,j) - o_del - e_del )
//!   F(i,j+1) = max( 0, F(i,j) - e_ins , H(i,j) - o_ins - e_ins )
//! ```
//!
//! The `max(0, .)` on `H` is what makes it *local*: an alignment may start anywhere, so there is no
//! `h0` and no first-row initialization at all. Everything is non-negative, which is precisely what
//! lets the 16-lane u8 kernel below use saturating unsigned arithmetic with no bias term.
//!
//! # Order to read this file in
//!
//! 1. [`batched_ksw_align2`], the entry point: forward pass, then the reverse start-recovery pass.
//! 2. [`fwd_local_sw_scalar`], the readable reference for one forward pass. The two NEON kernels are
//!    this function with the lane loop replaced by vector lanes, so read it first.
//! 3. [`extract_group`], the shared per-lane output step (`qe` and the 2nd-best `score2`/`te2`),
//!    called identically by the scalar and vector paths so they cannot drift.
//! 4. [`fwd_local_sw_neon`] (i16, 8 lanes) and [`fwd_local_sw_neon_u8`] (u8, 16 lanes, the one that
//!    actually runs on stock settings).
//!
//! # Glossary: the short names kept from the C, in plain language
//!
//! | name | plain language |
//! |---|---|
//! | `i` | index of the current **target** (reference-window) base: the DP row |
//! | `j` | index of the current **query** (read) base: the DP column |
//! | `l` | which lane, i.e. which of the batched rescue jobs |
//! | `H` (`h`, `h_cur`, `h_prev`) | best score of any alignment ending at this cell |
//! | `E` (`e`) | best score ending here with a **deletion** open: a gap in the query, CIGAR `D` |
//! | `F` (`f`, `f_v`) | best score ending here with an **insertion** open: a gap in the target, CIGAR `I` |
//! | `imax` | this row's best `H`; `imax_col` the column where it occurred |
//! | `gmax` | best `H` over every row so far, i.e. the alignment score |
//! | `te` / `qe` | 0-based target / query position where the best alignment **ends** |
//! | `tb` / `qb` | where it **begins**, recovered by the reverse pass; `-1` means "not found" |
//! | `score2` / `te2` | best *rival* alignment elsewhere in the window, and its row |
//! | `minsc` | score below which a row max is not even a `score2` candidate (`KSW_XSUBO`) |
//! | `endsc` | score at which the pass may stop early (`KSW_XSTOP`) |
//! | `oe_del` / `oe_ins` | cost of a gap's *first* base: open + one extend |
//! | `b` | the running list of `score2` candidates, one entry per surviving row max |
//!
//! There is no `M` here and no `h0`, unlike [`crate::batched`]: this recurrence has no separate
//! diagonal surface (gaps open straight from `H`) and no seed score to start from. A `_v` suffix
//! always means "the vector register holding one of these per lane".

use bwa_extend::{KswAlignResult, SuboptimalTracker};
use std::sync::OnceLock;

use crate::rescue_prune;

/// One mate-rescue local-SW job: align `query` against `target` (both `0..=4` codes).
///
/// `query` is the unmapped mate's read (or its reverse complement), typically 100-150 bp. `target`
/// is the reference window where the insert-size distribution says the mate should be, which is
/// large (bwa opens `2 * max_insert` around the anchor), so the DP rectangle is heavily
/// target-dominated: the row loop is long and the column loop is short.
#[derive(Clone, Copy)]
pub struct KswJob<'a> {
    /// The read to place, one byte per base in 2-bit code (`0=A 1=C 2=G 3=T`, `4=N`); never ASCII.
    /// Typically 100-150 bp. Supplied by `mem_matesw`, already reverse-complemented if the pair
    /// orientation calls for it. Every byte must be `<= 4`: values `>= 5` collide with the [`ZPAD`]
    /// and [`PAD`] sentinels and would be scored as padding. May be empty, in which case the job
    /// contributes no DP work and yields the default `(0, -1, -1, -1, -1)` result.
    pub query: &'a [u8],
    /// The reference window to search, same 2-bit encoding and same `<= 4` precondition. Much longer
    /// than `query` (bwa opens roughly `2 * max_insert` around the anchor mate), so the DP rectangle
    /// is target-dominated: many rows, few columns. Its length is unconstrained by the kernel width
    /// because row indices are kept in scalar `i32`, never in a SIMD lane.
    pub target: &'a [u8],
}

/// One forward local-SW pass: the target is `target`, the query is `query`, and the pass reports the
/// max score reaching `>= minsc` and stops early once it reaches `endsc`. This is the unit the
/// vectorized kernel batches; [`batched_ksw_align2`] issues one batch for the forward pass and one for
/// the reverse (`KSW_XSTART`) start-recovery pass.
///
/// - `minsc`: the score below which a row max is not worth recording in the 2nd-best list
///   (`KSW_XSUBO`, `ksw.cpp:130`, `:194`). bwa passes `opt->min_seed_len * opt->a`. Set to
///   `i32::MAX` on the reverse pass, which suppresses the list entirely, since `score2` is only ever
///   taken from the forward pass.
/// - `endsc`: stop as soon as the running max reaches this (`KSW_XSTOP`, `ksw.cpp:131`, `:207`).
///   `i32::MAX` on the forward pass means "never stop early"; on the reverse pass it is the forward
///   score, because the reverse walk only needs to find where that same score began.
#[derive(Clone, Copy)]
pub(crate) struct FwdJob<'a> {
    /// DP columns: 2-bit base codes, `<= 4`. On the forward pass this is [`KswJob::query`] verbatim;
    /// on the reverse pass it is the owned reversed prefix `query[..=qe]`, so it is at most as long.
    pub(crate) query: &'a [u8],
    /// DP rows: 2-bit base codes, `<= 4`. Forward pass: [`KswJob::target`]. Reverse pass: the owned
    /// reversed prefix `target[..=te]`.
    pub(crate) target: &'a [u8],
    /// 2nd-best candidate cutoff, in score units. See the doc above for how each pass sets it.
    pub(crate) minsc: i32,
    /// Early-stop score, in score units. See the doc above for how each pass sets it.
    pub(crate) endsc: i32,
}

/// `BWA4_MATESW_TIME=1`: cells, jobs and wall for the rescue kernel, so its throughput can be
/// compared against the ISA's ceiling. The note calling this kernel "memory-bandwidth-bound"
/// predates the finding that this aligner uses ~20% of one core's DRAM bandwidth, and its rails are
/// only `qmax * LANES` bytes -- L1-resident. Measure before believing it.
pub mod cells {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    /// Running total of DP cells the caller *asked* for, summed as `qlen * tlen` per job over every
    /// thread. This is the nominal work, not the work performed: it counts neither the extra columns
    /// added by [`super::ksw_padded_qlen`] nor the rows skipped by the `endsc` early exit, and it
    /// does not include the reverse start-recovery pass. Accumulated only when [`enabled`].
    pub static CELLS: AtomicU64 = AtomicU64::new(0);
    /// Number of rescue jobs submitted to [`super::batched_ksw_align2`], forward pass only. The
    /// divisor for the per-job means printed by [`dump`].
    pub static JOBS: AtomicU64 = AtomicU64::new(0);
    /// Kernel calls, and the power-of-two histogram of `jobs.len()` per call (issue #53 step 0).
    /// A GPU launch only amortises from a few thousand jobs up, so this distribution, not the job
    /// total, decides whether a GPU backend can be one launch per call or needs a cross-thread
    /// aggregation queue. Counted here rather than at the call site so every caller is covered.
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub const BUCKETS: usize = 16;
    pub static CALL_HIST: [AtomicU64; BUCKETS] = [const { AtomicU64::new(0) }; BUCKETS];
    pub static JOB_HIST: [AtomicU64; BUCKETS] = [const { AtomicU64::new(0) }; BUCKETS];
    pub static CALL_MAX: AtomicU64 = AtomicU64::new(0);

    /// Bucket index of a batch size: `floor(log2(n))`, saturated at the last bucket.
    pub fn bucket(n: u64) -> usize {
        if n == 0 {
            return 0;
        }
        (63 - n.leading_zeros() as usize).min(BUCKETS - 1)
    }

    /// Record one kernel call carrying `n` jobs. No-op unless the probe is enabled.
    pub fn record_call(n: usize) {
        if !enabled() {
            return;
        }
        let n = n as u64;
        CALLS.fetch_add(1, Ordering::Relaxed);
        CALL_HIST[bucket(n)].fetch_add(1, Ordering::Relaxed);
        JOB_HIST[bucket(n)].fetch_add(n, Ordering::Relaxed);
        CALL_MAX.fetch_max(n, Ordering::Relaxed);
    }
    /// Summed wall time in nanoseconds spent inside [`super::batched_ksw_align2`], across all
    /// threads, so it is CPU time rather than elapsed time when `-t > 1`.
    pub static NS: AtomicU64 = AtomicU64::new(0);
    /// Summed query lengths in bases over all counted jobs; `QLEN / JOBS` is the mean read length.
    pub static QLEN: AtomicU64 = AtomicU64::new(0);
    /// Summed target lengths in bases over all counted jobs; `TLEN / JOBS` is the mean rescue window,
    /// the number that revealed the window (not the read) is what makes rescue expensive.
    pub static TLEN: AtomicU64 = AtomicU64::new(0);
    /// DP cells the kernel actually executes in the forward pass, i.e. summed over 16-lane groups in
    /// caller order, `16 * max(padded qlen) * max(tlen)` per group. Divided by [`CELLS`] this is the
    /// lane-divergence tax.
    pub static EXEC: AtomicU64 = AtomicU64::new(0);
    /// The same count under the counterfactual where each batch is length-sorted before grouping.
    /// `EXEC / EXEC_SORTED` is the speed-up length sorting could buy in the forward pass.
    pub static EXEC_SORTED: AtomicU64 = AtomicU64::new(0);
    /// Jobs inside one kernel call whose `(query, target)` byte slices are exactly equal to an
    /// earlier job's in the same call, hence recomputed for nothing.
    pub static DUP_JOBS: AtomicU64 = AtomicU64::new(0);
    /// Rescue DPs whose result was ACCEPTED (`score >= min_seed_len` and `qb >= 0`), and the total
    /// scored. The kernel is ~32% of a paired-end run's CPU; the ratio says how much of it produces
    /// nothing, which is the ceiling on what a sound pre-filter could remove.
    pub static ACCEPTED: AtomicU64 = AtomicU64::new(0);
    pub static SCORED: AtomicU64 = AtomicU64::new(0);
    /// Exact pruning ([`super::rescue_prune`]): jobs proven below `minsc` and skipped, jobs cut to
    /// a hull, jobs run in full, and the target rows of the hull jobs before and after the cut.
    pub static PRUNE_FAIL: AtomicU64 = AtomicU64::new(0);
    pub static PRUNE_HULL: AtomicU64 = AtomicU64::new(0);
    pub static PRUNE_BAND: AtomicU64 = AtomicU64::new(0);
    /// Jobs answered from an earlier job of the batch with the same mate and the same kept rows.
    pub static PRUNE_REUSED: AtomicU64 = AtomicU64::new(0);
    /// The banded path: kernel time, the rest of `Batch::run`, cell vectors computed, round-2 parents.
    pub static BAND_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
    pub static BAND_OTHER_NS: AtomicU64 = AtomicU64::new(0);
    pub static BAND_CELLS: AtomicU64 = AtomicU64::new(0);
    pub static BAND_R2: AtomicU64 = AtomicU64::new(0);
    pub static PRUNE_FULL: AtomicU64 = AtomicU64::new(0);
    pub static HULL_ROWS_IN: AtomicU64 = AtomicU64::new(0);
    pub static HULL_ROWS_KEPT: AtomicU64 = AtomicU64::new(0);
    /// Summed time inside the filter's decisions, all threads, in nanoseconds.
    pub static PRUNE_NS: AtomicU64 = AtomicU64::new(0);

    /// Record one rescue DP's outcome. No-op unless `BWA4_MATESW_TIME` is set.
    pub fn count_outcome(accepted: bool) {
        if !enabled() {
            return;
        }
        SCORED.fetch_add(1, Ordering::Relaxed);
        if accepted {
            ACCEPTED.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether `BWA4_MATESW_TIME` is set in the environment. Read once and cached, so setting the
    /// variable after the first call has no effect and the hot path pays only an atomic load.
    ///
    /// # Returns
    /// `true` if the counters above should be accumulated.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_MATESW_TIME").is_some())
    }
    /// Print the accumulated counters to stderr, once, at the end of a run. No-op unless
    /// [`enabled`]. Takes no parameters and reads the statics above with `Relaxed` ordering: the
    /// counts are diagnostics, so a slightly stale read from another thread is acceptable.
    ///
    /// # What the throughput figure should be compared against
    ///
    /// This line used to print "16 u8 lanes x ~3.5 GHz = ~56 Gcell/s if 1 cell/lane/cycle". That
    /// number is a width bound, not an achievable one, and it made the kernel look five times worse
    /// than it is. **Measured** on the M4 Max this was developed on:
    ///
    /// | | |
    /// |---|---|
    /// | peak NEON op throughput, no dependencies, no memory | 16.63 G ops/s, about 3.8 per cycle |
    /// | the kernel's own per-cell op sequence, registers only | **16.04 Gcell/s**, 93% of that peak |
    /// | the shipped kernel, real data | 10.34 Gcell/s, **64% of the ceiling above** |
    ///
    /// The reconciliation is that ONE vector operation advances 16 cells, because the layout is
    /// inter-sequence: sixteen lanes are sixteen different jobs. A cell costs about 15 operations
    /// per ROW, and a row covers 16 cells, so the true figure is ~0.98 operations per cell, not one
    /// cell per lane per cycle. Disassembling the shipped binary confirms it: the quad fast-column
    /// loop is 71 instructions for 64 cells, 63 of them vector, i.e. 1.11 instructions per cell.
    ///
    /// So the arithmetic is essentially done. What the 36% actually is, corrected on 2026-08-29
    /// after two of the four suspects above were measured and cleared:
    ///
    /// | | |
    /// |---|---|
    /// | measured gap to the ceiling | 1.62x |
    /// | lane divergence and padding, from this probe's own accounting | 1.08x |
    /// | leaves | 1.50x |
    /// | the DP loop's four memory accesses, which the ceiling does not pay | 1.31x |
    /// | genuinely unexplained | **1.15x** |
    ///
    /// **The ceiling was never reachable, and that is the first thing to know before chasing it.**
    /// It is measured registers-only; the shipped loop is 17 instructions of which 2 loads and 2
    /// stores, so a register-only twin would be 13, and 17/13 is 1.31x on its own. Every lane of an
    /// inter-sequence kernel needs its own H and E state across a 150-column query, and that state
    /// cannot live in registers. The memory is the layout, not slack in it.
    ///
    /// Two of the four suspects this note used to name have since been measured and are NOT where
    /// the time is. The row epilogue's scalar sixteen-lane loop was guarded with a vector test, and
    /// that was **1.1% slower** (see `docs/kernel-ceiling-and-dead-ends.md`): the horizontal
    /// reduction costs more than the well-predicted loop it skips. The `score2` walk inside
    /// `extract_group` had half its work removed by a provable skip, and this probe did not move.
    ///
    /// And then the two places left were measured too, so the accounting closes:
    ///
    /// | | |
    /// |---|---|
    /// | lane divergence and padding | 1.080x |
    /// | the four memory accesses the ceiling omits (17 instructions against 13) | 1.308x |
    /// | padded tail columns (10 of 160, at 1.28x each) | 1.018x |
    /// | group pack and extract (1.40% of the x86 profile against the kernel's 9.55%) | 1.147x |
    /// | **product** | **1.648x** |
    /// | measured gap | **1.617x** |
    ///
    /// **There is no unexplained slack in this kernel.** Every part of the 1.62x is a named,
    /// measured cost, and not one of them is removable without changing the layout: the memory is
    /// what an inter-sequence kernel needs to keep per-lane state across a 150-column query, the
    /// divergence is what batching unequal lengths costs, the tail columns are the padding the
    /// vector width requires, and the pack/extract is the price of the SoA transpose.
    ///
    /// So: do not try to shorten the cell recurrence, do not treat the register-only figure as a
    /// target, and do not come back here for a factor. The next factor is not on this CPU.
    pub fn dump() {
        if !enabled() {
            return;
        }
        let (cells, jobs, elapsed_ns) = (
            CELLS.load(Ordering::Relaxed),
            JOBS.load(Ordering::Relaxed),
            NS.load(Ordering::Relaxed),
        );
        let seconds = elapsed_ns as f64 / 1e9;
        eprintln!(
            "[matesw] {jobs} jobs, {cells} DP cells in {seconds:.2}s CPU -> {:.2} Gcell/s/thread\n\
             [matesw] measured ceiling on an M4 Max: ~16 Gcell/s/thread (see the note above)",
            cells as f64 / seconds.max(1e-9) / 1e9
        );
        let calls = CALLS.load(Ordering::Relaxed);
        eprintln!(
            "[matesw] {calls} kernel calls, mean {:.1} jobs/call, largest {}",
            jobs as f64 / calls.max(1) as f64,
            CALL_MAX.load(Ordering::Relaxed)
        );
        eprintln!(
            "[matesw] {:>12}  {:>10}  {:>12}  {:>7}",
            "jobs/call", "calls", "jobs", "%_jobs"
        );
        for b in 0..BUCKETS {
            let (c, j) = (
                CALL_HIST[b].load(Ordering::Relaxed),
                JOB_HIST[b].load(Ordering::Relaxed),
            );
            if c == 0 {
                continue;
            }
            let lo = 1u64 << b;
            let label = if b == BUCKETS - 1 {
                format!("{lo}+")
            } else {
                format!("{lo}-{}", (lo << 1) - 1)
            };
            eprintln!(
                "[matesw] {label:>12}  {c:>10}  {j:>12}  {:>6.1}%",
                100.0 * j as f64 / jobs.max(1) as f64
            );
        }
        let (acc, sc) = (
            ACCEPTED.load(Ordering::Relaxed),
            SCORED.load(Ordering::Relaxed),
        );
        if sc > 0 {
            eprintln!(
                "[matesw] accepted {acc} of {sc} scored rescue DPs ({:.1}%); {:.1}% produced nothing",
                100.0 * acc as f64 / sc as f64,
                100.0 * (sc - acc) as f64 / sc as f64
            );
        }
        let (query_bases, target_bases) =
            (QLEN.load(Ordering::Relaxed), TLEN.load(Ordering::Relaxed));
        eprintln!(
            "[matesw] mean query = {:.0} bp, mean target window = {:.0} bp -> {:.0} cells/job",
            query_bases as f64 / jobs.max(1) as f64,
            target_bases as f64 / jobs.max(1) as f64,
            cells as f64 / jobs.max(1) as f64
        );
        let (exec, exec_sorted, dup) = (
            EXEC.load(Ordering::Relaxed),
            EXEC_SORTED.load(Ordering::Relaxed),
            DUP_JOBS.load(Ordering::Relaxed),
        );
        eprintln!(
            "[matesw] lane divergence: executed {exec} cells vs {cells} nominal ({:.2}x tax); \
             length-sorted would execute {exec_sorted} ({:.2}x of nominal, {:.1}% saved)",
            exec as f64 / cells.max(1) as f64,
            exec_sorted as f64 / cells.max(1) as f64,
            100.0 * (exec as f64 - exec_sorted as f64) / exec.max(1) as f64,
        );
        eprintln!(
            "[matesw] duplicate jobs within a call: {dup} of {jobs} ({:.1}%)",
            100.0 * dup as f64 / jobs.max(1) as f64
        );
        let (fail, hull, full) = (
            PRUNE_FAIL.load(Ordering::Relaxed),
            PRUNE_HULL.load(Ordering::Relaxed),
            PRUNE_FULL.load(Ordering::Relaxed),
        );
        if fail + hull + full + PRUNE_BAND.load(Ordering::Relaxed) > 0 {
            let (rin, rkept) = (
                HULL_ROWS_IN.load(Ordering::Relaxed),
                HULL_ROWS_KEPT.load(Ordering::Relaxed),
            );
            let band = PRUNE_BAND.load(Ordering::Relaxed);
            eprintln!(
                "[matesw] reused: {} jobs answered from an identical kept window",
                PRUNE_REUSED.load(Ordering::Relaxed)
            );
            eprintln!(
                "[matesw] bands: kernel {:.2}s for {} cells ({:.2} Gcell/s), other {:.2}s, round 2 {}",
                BAND_KERNEL_NS.load(Ordering::Relaxed) as f64 / 1e9,
                BAND_CELLS.load(Ordering::Relaxed),
                BAND_CELLS.load(Ordering::Relaxed) as f64 / BAND_KERNEL_NS.load(Ordering::Relaxed).max(1) as f64,
                BAND_OTHER_NS.load(Ordering::Relaxed) as f64 / 1e9,
                BAND_R2.load(Ordering::Relaxed),
            );
            eprintln!(
                "[matesw] prune: {fail} proven failing, {hull} narrowed (rows kept {:.1}%), {band} banded, \
                 {full} full, of {} filtered, in {:.2}s CPU ({:.2} us/job)",
                100.0 * rkept as f64 / rin.max(1) as f64,
                fail + hull + full + band,
                PRUNE_NS.load(Ordering::Relaxed) as f64 / 1e9,
                PRUNE_NS.load(Ordering::Relaxed) as f64 / 1e3
                    / (fail + hull + full + band).max(1) as f64
            );
        }
    }
}

/// Batched local SW: `out[i]` equals [`bwa_extend::ksw_align2`] on `jobs[i]`. Structured exactly like
/// `ksw_align2`: a forward pass over every job, then a reverse pass over the truncated/reversed
/// prefixes of the qualifying jobs to recover the start coordinates. Both passes go through
/// [`fwd_local_sw_batch`], the single point the NEON kernel plugs into.
///
/// Why two passes: a scoring-only Smith-Waterman keeps no traceback, so it learns where the
/// alignment *ends* (`qe`, `te`) but not where it starts. bwa's `KSW_XSTART` trick (`ksw.h:87`,
/// `ksw.cpp:370`) is to re-run the same DP on the reversed prefixes `query[..=qe]` and
/// `target[..=te]`; by symmetry the best local alignment of the reversed pair is the same alignment
/// read backwards, so its end offsets are the original's start offsets, and `tb = te - rte`.
/// The reverse pass stops the moment it matches the forward score (`endsc`), which is why it is much
/// cheaper than the forward one.
///
/// Parameters: see the crate-level glossary for `m`/`mat`/`o_*`/`e_*`. `minsc` is the 2nd-best
/// cutoff described on [`FwdJob`]; `max_sc` is the best entry of `mat` (bwa's `opt->a`), used both
/// to pick the kernel width and to size the `score2` exclusion window in [`extract_group`].
///
/// # Parameters
/// - `jobs`: the rescue batch, in caller order. May be empty. Bases must be `<= 4`. Order is
///   load-bearing only in that `out[i]` corresponds to `jobs[i]`; the kernel additionally groups
///   jobs into consecutive chunks of 8 or 16, so unequal lengths inside a chunk cost padded work
///   (the caller is expected to have length-sorted, but correctness does not depend on it).
/// - `m`: side of the square substitution matrix. Always 5 in bwa (A/C/G/T/N); the NEON paths
///   require exactly 5 and fall back to scalar otherwise.
/// - `mat`: row-major `m * m` substitution scores in score units; `mat[t * m + q]` scores target
///   base `t` against query base `q`. Length must be at least `m * m`. bwa's form is `+a` on the
///   diagonal, `-b` off it, `-1` on every N row and column.
/// - `o_del`, `e_del`: deletion (gap in the query, CIGAR `D`) gap-open and gap-extend penalties, as
///   positive magnitudes in score units. A run of `k` deleted bases costs `o_del + k * e_del`.
/// - `o_ins`, `e_ins`: the same for insertions (gap in the target, CIGAR `I`).
/// - `minsc`: score units. A row max below this is not a `score2` candidate, and a job whose score
///   falls below it skips the reverse pass entirely (its `qb`/`tb` stay `-1`). bwa passes
///   `opt->min_seed_len * opt->a`.
/// - `max_sc`: the largest entry of `mat`, i.e. the match bonus `opt->a`, in score units and `> 0`.
///   Used to bound a job's reachable score when picking the kernel width, to compute the padded
///   query length ([`ksw_padded_qlen`]), and to size the `score2` exclusion window.
///
/// # Returns
/// One [`KswAlignResult`] per input job, in input order, each byte-identical to
/// [`bwa_extend::ksw_align2`] on that job. `qb`/`qe`/`tb`/`te` are inclusive 0-based positions, not
/// lengths. `te`/`qe` are `-1` when no alignment scored above 0; `qb`/`tb` are `-1` when the reverse
/// start-recovery pass was skipped or disagreed, which `mem_matesw` treats as "drop this rescue".
/// `score2`/`te2` are `-1` when no rival alignment qualified.
#[allow(clippy::too_many_arguments)]
pub fn batched_ksw_align2(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Vec<KswAlignResult> {
    batched_align(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, None, None,
    )
}

/// [`batched_ksw_align2`] for mate rescue, where only alignments scoring `>= minsc` are ever
/// consumed: windows are first run through the exact K-mer filter of [`rescue_prune`], which skips
/// the jobs it proves cannot reach `minsc` and narrows the others to the rows that can.
///
/// # Returns
/// One [`KswAlignResult`] per job, in input order. For a job whose `ksw_align2` score is
/// `>= minsc`, every field is byte-identical to [`bwa_extend::ksw_align2`]'s. For any other job,
/// `score < minsc` and `qb = tb = -1`, which is all `mem_matesw` reads of it (it accepts a rescue
/// only when `qb >= 0`, and `qb` is only ever set at `score >= minsc`); the other fields are
/// unspecified.
///
/// The filter is ON by default; `BWA4_RESCUE_PRUNE=0` turns it off, which makes this exactly
/// [`batched_ksw_align2`]. It costs about 1 us per job to gate and about 3 us more per job it
/// filters, against a ~19 us rescue DP on real reads, so it pays exactly as far as real windows are
/// prunable. On 2 M real GIAB HG002 pairs against hg38 (M4 Max), 15.1 % of the 6.9 M rescue jobs
/// are proven failing; at `-t1` on 300 k of those pairs, median user time is -1.6 % (hit gate
/// 400), and output is byte-identical to bwa-mem2. On simulated pairs, whose windows nearly always
/// hold the true mate, it is a small loss instead (+1 to +3 % on hg38 chr20 wgsim). The fork makes
/// the filter itself ~4x cheaper with SIMD and saves more per window with a banded DP; neither is
/// here yet.
#[allow(clippy::too_many_arguments)]
pub fn batched_mate_rescue(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Vec<KswAlignResult> {
    let merged = crate::merged_pair(m, mat);
    if let Some((x, y, uni, true)) = merged {
        let (buf, spans) = crate::rewrite_jobs(jobs.iter().map(|j| (j.query, j.target)), x, y);
        let conv: Vec<KswJob> = spans
            .iter()
            .map(|&(a, b, c)| KswJob {
                query: &buf[a..b],
                target: &buf[b..c],
            })
            .collect();
        return batched_mate_rescue(&conv, m, &uni, o_del, e_del, o_ins, e_ins, minsc, max_sc);
    }
    // `--meth` neutral / genomic (`merged_pair` without `exact`): the prune bounds the uniform
    // matrix with the freed cell counted a match (`rescue_prune::decide_rel`'s relation), which
    // scores every cell at least as high as `mat`, with the same gaps. Every local score, hence
    // every row maximum, is then at least as high, so a window proven below `minsc` under it is
    // below it here, and a hull holding every alignment reaching `minsc` holds every one here.
    let (prune, rel) = match merged {
        Some((x, y, uni, false)) => {
            let mm = uni[x as usize * 5 + y as usize];
            // The freed (target, query) cell; both freed would be the exact rewrite above.
            let rel = if mat[x as usize * 5 + y as usize] > mm {
                (x, y)
            } else {
                (y, x)
            };
            // shortcut: the relation's K-mer count is scalar, and on x86 it plus the band kernel
            // costs more than the 32/64-lane full SW it saves (TAPS -t4, EPYC 9V45: 3.81 s without,
            // 4.09-4.17 s with; 7763: equal, -2.5% at -t1), so x86 runs every job in full. Drop
            // the cfg once the relation count has an SSE form like `rescue_prune::sse::count_hits`.
            let prune = if cfg!(target_arch = "x86_64") {
                None
            } else {
                prune_params(m, &uni, o_del, e_del, o_ins, e_ins, minsc, max_sc)
            };
            (prune, Some(rel))
        }
        _ => (
            prune_params(m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc),
            None,
        ),
    };
    // `BWA4_RESCUE_DUMP=path`: append every rescue job (u32 target length, u32 query length,
    // target bytes, query bytes, little-endian) to `path`, the input of `rescue_prune`'s offline
    // bench. A debugging aid; off unless set.
    static DUMP: OnceLock<Option<std::sync::Mutex<std::fs::File>>> = OnceLock::new();
    if let Some(f) = DUMP.get_or_init(|| {
        std::env::var_os("BWA4_RESCUE_DUMP")
            .map(|p| std::sync::Mutex::new(std::fs::File::create(p).expect("BWA4_RESCUE_DUMP")))
    }) {
        use std::io::Write;
        let mut f = f.lock().unwrap();
        for j in jobs {
            for part in [
                &(j.target.len() as u32).to_le_bytes()[..],
                &(j.query.len() as u32).to_le_bytes()[..],
                j.target,
                j.query,
            ] {
                f.write_all(part).expect("BWA4_RESCUE_DUMP write");
            }
        }
    }
    batched_align(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, prune, rel,
    )
}

/// Three 8-byte words of a window (first, middle, last; all of it when shorter) folded into one:
/// equal windows get equal values, and different ones almost never do. A prefilter only, the
/// bytes are still compared.
fn window_fp(t: &[u8]) -> u64 {
    let n = t.len();
    if n < 8 {
        return t.iter().fold(n as u64, |h, &b| h << 8 | u64::from(b));
    }
    let w = |at: usize| u64::from_le_bytes(t[at..at + 8].try_into().unwrap());
    w(0) ^ w((n - 8) / 2).rotate_left(21) ^ w(n - 8).rotate_left(42)
}

/// A grouping key for a query's bytes: equal queries get equal keys. Collisions only cost a
/// cache miss in the filter, never a wrong result (`load_query` compares the bytes).
fn query_key(q: &[u8]) -> u64 {
    let mix = |h: u64, w: u64| (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    let (words, tail) = q.as_chunks::<8>();
    let mut h = words
        .iter()
        .fold(q.len() as u64, |h, w| mix(h, u64::from_le_bytes(*w)));
    // The tail as a zero-padded little-endian word, without the `memcpy` call per chunk that
    // `copy_from_slice` into a padded buffer made.
    if !tail.is_empty() {
        h = mix(h, tail.iter().rev().fold(0, |v, &b| v << 8 | u64::from(b)));
    }
    h
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
thread_local! {
    /// The banded-rescue state, reused across kernel calls.
    static BAND: std::cell::RefCell<crate::rescue_band::Batch> = Default::default();
}

/// The body of both entry points; `prune` is the pruning bound, `None` for the exact-everywhere path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_align(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
    prune: Option<rescue_prune::Params>,
    // The relation the prune decides under (`--meth`, see `rescue_prune::decide_rel`).
    rel: Option<(u8, u8)>,
) -> Vec<KswAlignResult> {
    // `Some(start instant)` only when BWA4_MATESW_TIME is set; `None` disables all accounting so the
    // stock path pays one cached bool load and nothing else.
    // NOTE the ordering: the probe below runs BEFORE the timer starts, because it is not cheap
    // (it length-sorts a counterfactual copy and hashes every job) and counting it as kernel time
    // would corrupt the very number it exists to explain.
    let probing = cells::enabled();
    if probing {
        cells::record_call(jobs.len());
        use std::sync::atomic::Ordering::Relaxed;
        // The DP is query x target per job; that is the work the kernel must actually do.
        let cell_count: u64 = jobs
            .iter()
            .map(|j| (j.query.len() * j.target.len()) as u64)
            .sum();
        cells::CELLS.fetch_add(cell_count, Relaxed);
        cells::JOBS.fetch_add(jobs.len() as u64, Relaxed);
        // 207k cells per job means a huge target window; record the dimensions to see which side it is.
        cells::QLEN.fetch_add(
            jobs.iter().map(|j| j.query.len() as u64).sum::<u64>(),
            Relaxed,
        );
        cells::TLEN.fetch_add(
            jobs.iter().map(|j| j.target.len() as u64).sum::<u64>(),
            Relaxed,
        );
        // Lane-divergence accounting. The kernels take `jobs.chunks(LANES)` in caller order and run
        // every group to `max(qpad) x max(tlen)` over its lanes, so a group holding one long window
        // pays for that window in all 16 lanes. EXEC counts what the kernel actually executes;
        // EXEC_SORTED counts what it would execute if the batch were length-sorted first, which is
        // legal because each job's result is independent of the others (the grouping only decides
        // which jobs share a vector). The ratio EXEC / EXEC_SORTED is the whole prize of sorting.
        const PROBE_LANES: usize = 16;
        let qpad: Vec<usize> = jobs
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .collect();
        let tl: Vec<usize> = jobs.iter().map(|j| j.target.len()).collect();
        let group_cost = |q: &[usize], t: &[usize]| -> u64 {
            q.chunks(PROBE_LANES)
                .zip(t.chunks(PROBE_LANES))
                .map(|(qs, ts)| {
                    (PROBE_LANES
                        * qs.iter().copied().max().unwrap_or(0)
                        * ts.iter().copied().max().unwrap_or(0)) as u64
                })
                .sum()
        };
        cells::EXEC.fetch_add(group_cost(&qpad, &tl), Relaxed);
        // Counterfactual: same jobs, sorted by (target length, padded query length).
        let mut ord: Vec<usize> = (0..jobs.len()).collect();
        ord.sort_unstable_by_key(|&i| (tl[i], qpad[i]));
        let (qs, ts): (Vec<usize>, Vec<usize>) = ord.iter().map(|&i| (qpad[i], tl[i])).unzip();
        cells::EXEC_SORTED.fetch_add(group_cost(&qs, &ts), Relaxed);
        // Exact duplicate jobs within one kernel call: same (query, target) bytes means the same
        // five outputs, so one of them is pure waste. Counted, not yet exploited.
        let mut seen = std::collections::HashSet::with_capacity(jobs.len());
        let dups = jobs
            .iter()
            .filter(|j| !seen.insert((j.query, j.target)))
            .count();
        cells::DUP_JOBS.fetch_add(dups as u64, Relaxed);
    }
    // `Some(start instant)` only when BWA4_MATESW_TIME is set; `None` disables all accounting so the
    // stock path pays one cached bool load and nothing else.
    let timer = probing.then(std::time::Instant::now);
    // ---- Exact pruning (`rescue_prune`): drop the jobs proven to score below `minsc`, and cut the
    //      others down to the target rows that can hold a row max reaching it. `run_of[k]` is the
    //      job the k-th forward job came from and `hb_of[k]` the first target row it kept; a job
    //      not in `run_of` keeps the "nothing found" result below, which `mem_matesw` drops exactly
    //      as it drops a rescue scoring under `minsc`. ----
    let mut run_of: Vec<u32> = Vec::with_capacity(jobs.len());
    let mut hb_of: Vec<u32> = Vec::with_capacity(jobs.len());
    // ---- Pass 1: forward over the surviving jobs. Finds the score and where each alignment ENDS.
    // The pass-specific stop conditions: collect `score2` candidates at `minsc`, and never stop
    // early (`endsc = i32::MAX`) because the forward pass does not yet know what score to aim for.
    let mut fwd_jobs: Vec<FwdJob> = Vec::with_capacity(jobs.len());
    // Banded rescue (`rescue_band`): a pruned job whose bands are cheaper than its hull leaves
    // `fwd_jobs` and is run after them, under the u8 kernel's own conditions.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let band_ok = prune.is_some()
        && !matches!(forced_rescue_tier(), RescueTier::Scalar)
        && crate::rescue_band::kernel_available()
        && crate::rescue_band::gaps_ok(o_del, e_del, o_ins, e_ins)
        && (mat_is_standard(m, mat) || mat5_u8_ok(m, mat))
        && u8_table_has_headroom(mat);
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let mut band = BAND.with(|c| c.take());
    // The filter caches its K-mer table per query (`Scratch::load_query`), and a batch holds ~17
    // jobs per distinct mate query, scattered by the caller's (pair, anchor) order: visit the jobs
    // grouped by query content so the table is built once per query. Free to reorder: every
    // per-job result below (`run_of`, `hb_of`, the band plan) is keyed by the job's own index.
    let mut ord: Vec<(u64, u32)> = (0..jobs.len() as u32)
        .map(|i| {
            (
                if prune.is_some() {
                    query_key(jobs[i as usize].query)
                } else {
                    0
                },
                i,
            )
        })
        .collect();
    if prune.is_some() {
        ord.sort_unstable();
    }
    // ---- Repeats: a job whose kept rows (hull or whole window) and mate equal an earlier job's
    //      has that job's forward result, rows shifted: a pruned job's forward pass is exactly the
    //      DP of its kept rows (on either kernel), a function of those bytes, the mate and `minsc`.
    //      Two anchors near one mate locus produce them: different windows, one hull. Only the
    //      forward pass is shared; the reverse pass still runs on the job's own window. The
    //      candidates are the earlier jobs of the current query group (`ord` groups by query).
    //      `alias`: (job, source job, row shift). ----
    let mut alias: Vec<(u32, u32, i32)> = Vec::new();
    // (job, first kept row, kept rows, `window_fp` of those rows)
    let mut group: Vec<(u32, u32, u32, u64)> = Vec::new();
    let mut group_q: &[u8] = &[];
    let kept = |d: &rescue_prune::Decision, len1: usize| match *d {
        rescue_prune::Decision::Hull { hb, he } => Some((hb, he + 1 - hb)),
        rescue_prune::Decision::Full => Some((0, len1)),
        rescue_prune::Decision::Fail => None,
    };
    // The source among `group` for job `j` with kept rows `(hb, n)` and fingerprint `fp`, if any.
    // The fingerprint turns most misses away without a `memcmp` call (one was 0.27% of M4 PE).
    let find = |group: &[(u32, u32, u32, u64)], j: &KswJob, hb: usize, n: usize, fp: u64| {
        group
            .iter()
            .find(|&&(s, shb, sn, sfp)| {
                sn as usize == n
                    && sfp == fp
                    && jobs[s as usize].target[shb as usize..shb as usize + n]
                        == j.target[hb..hb + n]
            })
            .map(|&(s, shb, _, _)| (s, hb as i32 - shb as i32))
    };
    for &(_, i) in &ord {
        let (i, j) = (i as usize, &jobs[i as usize]);
        // Jobs of one mate share its slice: the pointer test skips the byte compare for them.
        let same_q = std::ptr::eq(group_q, j.query) || group_q == j.query;
        if prune.is_some() && !same_q {
            group.clear();
            group_q = j.query;
        }
        let t_decide = probing.then(std::time::Instant::now);
        let decision = match prune {
            None => None,
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            Some(p) if band_ok && u8_job_fits(j.query.len(), j.target.len(), mat, max_sc) => {
                let quanta = ksw_padded_qlen(j.query.len(), max_sc);
                let (d, t1) = band.decide(j.target, j.query, &p, prune_max_hits(), rel);
                if let Some((hb, n)) = kept(&d, j.target.len()) {
                    let fp = window_fp(&j.target[hb..hb + n]);
                    if let Some((src, sh)) = find(&group, j, hb, n, fp) {
                        alias.push((i as u32, src, sh));
                        continue;
                    }
                    group.push((i as u32, hb as u32, n as u32, fp));
                }
                match band.commit(i, j.target, j.query, d, t1, &p, quanta) {
                    Some(d) => Some(d),
                    None => {
                        if let Some(t0) = t_decide {
                            use std::sync::atomic::Ordering::Relaxed;
                            cells::PRUNE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
                            cells::PRUNE_BAND.fetch_add(1, Relaxed);
                        }
                        continue;
                    }
                }
            }
            Some(p) => {
                let d = match rel {
                    Some(r) => rescue_prune::decide_rel(j.target, j.query, &p, prune_max_hits(), r),
                    None => rescue_prune::decide(j.target, j.query, &p, prune_max_hits()),
                };
                if let Some((hb, n)) = kept(&d, j.target.len()) {
                    let fp = window_fp(&j.target[hb..hb + n]);
                    if let Some((src, sh)) = find(&group, j, hb, n, fp) {
                        alias.push((i as u32, src, sh));
                        continue;
                    }
                    group.push((i as u32, hb as u32, n as u32, fp));
                }
                Some(d)
            }
        };
        if let Some(t0) = t_decide {
            use std::sync::atomic::Ordering::Relaxed;
            cells::PRUNE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            match decision {
                Some(rescue_prune::Decision::Fail) => cells::PRUNE_FAIL.fetch_add(1, Relaxed),
                Some(rescue_prune::Decision::Hull { hb, he }) => {
                    cells::HULL_ROWS_IN.fetch_add(j.target.len() as u64, Relaxed);
                    cells::HULL_ROWS_KEPT.fetch_add((he + 1 - hb) as u64, Relaxed);
                    cells::PRUNE_HULL.fetch_add(1, Relaxed)
                }
                Some(rescue_prune::Decision::Full) => cells::PRUNE_FULL.fetch_add(1, Relaxed),
                None => 0,
            };
        }
        let (hb, target) = match decision {
            Some(rescue_prune::Decision::Fail) => continue,
            Some(rescue_prune::Decision::Hull { hb, he }) => (hb, &j.target[hb..=he]),
            Some(rescue_prune::Decision::Full) | None => (0, j.target),
        };
        run_of.push(i as u32);
        hb_of.push(hb as u32);
        fwd_jobs.push(FwdJob {
            query: j.query,
            target,
            minsc,
            endsc: i32::MAX,
        });
    }
    // One `(score, te, qe, score2, te2)` per job, in job order.
    //
    // Length-sorted before the kernel sees it. The kernel takes `jobs.chunks(LANES)` in caller
    // order and runs each group to `max(padded qlen) x max(tlen)` over its lanes, so one long
    // window in a group is paid for by every lane of that group. Grouping jobs of similar shape
    // together is the whole fix, and it is result-preserving for the reason stated in this crate's
    // module docs: a job's alignment depends only on its own `(query, target, minsc, endsc)`, so
    // the order decides lane occupancy and nothing else.
    //
    // Measured with `BWA4_MATESW_TIME` on 2 M real GIAB pairs, whole genome:
    //
    // | input | lane tax | sorting saves |
    // |---|---|---|
    // | untrimmed, all 151 bp | 1.09x | 0.0% |
    // | fastp-trimmed, variable lengths | 1.15x | 4.5% |
    //
    // Which is why it was measured as worthless the first time and is worth taking now: with reads
    // all one length there is nothing to sort. Real input is trimmed.
    let fwd_results = if sort_jobs_enabled() && fwd_jobs.len() > LANES {
        // `ord[k]` is the original index of the job the kernel will run in slot `k`. Sorted by
        // target length first because it dominates the cell count (~1400 rows against ~150
        // columns), then by the PADDED query length, which is what the kernel actually walks.
        // Keys computed once, not per comparison: `(target_len << 32 | padded_qlen, index)`.
        let mut ord: Vec<(u64, u32)> = fwd_jobs
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let pad = ksw_padded_qlen(j.query.len(), max_sc) as u64;
                ((j.target.len() as u64) << 32 | pad, i as u32)
            })
            .collect();
        ord.sort_unstable();
        let sorted: Vec<FwdJob> = ord.iter().map(|&(_, i)| fwd_jobs[i as usize]).collect();
        let got = fwd_local_sw_batch(&sorted, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
        // Scatter back to caller order: every downstream index (`out`, pass 2, the caller's
        // orientation cursor) is in job order and must stay that way.
        let mut back = vec![(0i32, 0i32, 0i32, 0i32, 0i32); got.len()];
        for (slot, &(_, i)) in ord.iter().enumerate() {
            back[i as usize] = got[slot];
        }
        back
    } else {
        fwd_local_sw_batch(&fwd_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
    };

    // The final answers, complete except for the start coordinates: `qb`/`tb` stay at the -1 sentinel
    // until pass 2 fills them, and stay -1 forever for jobs pass 2 skips or disagrees with. Pruned
    // jobs keep the all-sentinel result of a window where nothing scored, and a hull's rows are
    // shifted back to the full window, so pass 2 and the caller see full-window coordinates.
    let mut out: Vec<KswAlignResult> = vec![
        KswAlignResult {
            score: 0,
            qb: -1,
            qe: -1,
            tb: -1,
            te: -1,
            score2: -1,
            te2: -1,
        };
        jobs.len()
    ];
    for (k, &(score, te, qe, score2, te2)) in fwd_results.iter().enumerate() {
        let hb = hb_of[k] as i32;
        let shift = |row: i32| if row >= 0 { row + hb } else { row };
        out[run_of[k] as usize] = KswAlignResult {
            score,
            qb: -1,
            qe,
            tb: -1,
            te: shift(te),
            score2,
            te2: shift(te2),
        };
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        if band_ok {
            // SAFETY: NEON, the standard table and the u8 ceiling were checked when planning.
            unsafe {
                band.run(
                    jobs, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, &mut out,
                );
            }
        }
        band.clear();
    }
    for &(i, src, sh) in &alias {
        let shift = |row: i32| if row >= 0 { row + sh } else { row };
        let r = out[src as usize];
        out[i as usize] = KswAlignResult {
            te: shift(r.te),
            te2: shift(r.te2),
            ..r
        };
    }
    if probing {
        cells::PRUNE_REUSED.fetch_add(alias.len() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    // ---- Pass 2 (KSW_XSTART): reversed prefixes, to recover where each alignment BEGINS. ----
    // Reverse pass (KSW_XSTART): for each qualifying job, align the reversed prefixes ending at
    // (qe, te) and stop at `score`; the reversed end offsets give the start coords.
    //
    // DISCREPANCY, flagged not fixed (comments-only change): bwa does
    // `revseq(r.qe + 1, query); revseq(r.te + 1, target);` **in place** and then calls the kernel
    // with the *full* `tlen` (`ksw.cpp:368-372`), so the target's tail past `te` is untouched but
    // still scanned. The scalar reference in `bwa-extend/src/sw.rs:831-833` reproduces that exactly
    // (`trev = target.to_vec(); trev[..=te].reverse();`), whereas this batched path truncates:
    // `j.target[..=te].iter().rev()`. The two agree whenever the reversed prefix on its own reaches
    // `score` and trips KSW_XSTOP before the tail is ever consulted, which is the overwhelmingly
    // common case and why `matesw_equals_scalar` passes. It is UNVERIFIED whether a job exists where
    // the tail is what reaches `score`; if one does, this path would leave `qb`/`tb` at -1 where bwa
    // sets them, and `mem_matesw` drops a rescue when `qb < 0`. Worth a targeted differential test.
    // Owned reversed prefixes, and the index of the job each one came from.
    // `rev_bufs[k] = (reversed query[..=qe], reversed target[..=te])` for the k-th *qualifying* job,
    // and `rev_of_job[k]` is that job's index in `jobs`/`out`. The two vectors are parallel and
    // shorter than `jobs`, since jobs below `minsc` are skipped; `rev_of_job` is the only way back.
    // ONE flat arena holding every reversed prefix back to back, rather than two `Vec<u8>` per
    // qualifying job. On a real paired-end run the rescue submits ~3.68M jobs and most of them
    // qualify, so the per-job form cost ~7M allocations and copied ~5 GB (mean target window is
    // 1395 bp). A `sample` profile attributes ~1.7% of busy time to mimalloc entry points; this is
    // one of the two big contributors (the other, the introsort stack, is fixed in `bwa-chain`).
    //
    // Byte-identical by construction: the same bytes in the same order, only their storage changes.
    // `qualifying` is computed first so the arena is sized exactly once and never reallocates
    // mid-fill, which is what would reintroduce the copying this removes.
    //
    // ISSUE #50A. The reverse target is `target[..=te]` reversed, `te + 1` rows, but the alignment
    // it has to rediscover cannot span more than `rev_span_bound` of them (see there for the proof).
    // Rows past that bound cannot participate in an alignment scoring `score`, so dropping them
    // cannot change `rscore`/`rte`/`rqe` for any job whose reverse pass reaches the forward score,
    // and that equality is the only thing the caller acts on. It shortens the arena, the SoA target
    // scatter, `tmax`, and the `rowmax` allocation that is sized by it.
    //
    // What it does NOT shorten is the DP itself: `endsc = score` always fires (the reverse pass's
    // `gmax` is exactly `score`, proved in the issue), so the row loop already stopped at the freeze.
    // This is buffer sizing, not fewer cells.
    let rev_bound = revbound_enabled();
    let rev_tlen = |i: usize| -> usize {
        let full = out[i].te as usize + 1;
        if rev_bound {
            full.min(rev_span_bound(
                out[i].qe as usize + 1,
                out[i].score,
                max_sc,
                o_del,
                e_del,
            ))
        } else {
            full
        }
    };
    let arena_bytes: usize = jobs
        .iter()
        .enumerate()
        .filter(|(i, _)| out[*i].score >= minsc && out[*i].qe >= 0)
        .map(|(i, _)| (out[i].qe as usize + 1) + rev_tlen(i))
        .sum();
    let mut rev_arena: Vec<u8> = Vec::with_capacity(arena_bytes);
    // `(q_off, q_len, t_off, t_len)` into `rev_arena` for the k-th qualifying job. Offsets, not
    // slices, because the arena is still being appended to while these are recorded.
    let mut rev_spans: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut rev_of_job: Vec<usize> = Vec::new();
    for (i, j) in jobs.iter().enumerate() {
        // The forward pass's verdict for job `i`: its score, and the inclusive 0-based positions
        // where the best alignment ended.
        let (score, te, qe) = (out[i].score, out[i].te, out[i].qe);
        // Only jobs that cleared `minsc` get start coordinates, matching the C's guard at
        // `bwamem_pair.cpp:667`: below the cutoff the caller discards the alignment anyway, so the
        // reverse pass would be wasted work. `qe >= 0` also rules out "no alignment found at all".
        if score >= minsc && qe >= 0 {
            // The prefixes ending at the alignment's end, reversed. Aligning these to each other
            // finds the same alignment read backwards, so its end offsets are distances back from
            // (qe, te) to the alignment's start.
            // Copied, then reversed in place: `slice::reverse` is vectorised, a `rev()` iterator
            // copy goes byte by byte.
            let q_off = rev_arena.len();
            rev_arena.extend_from_slice(&j.query[..=qe as usize]);
            rev_arena[q_off..].reverse();
            let t_off = rev_arena.len();
            // Reversed, so the FIRST `rev_tlen(i)` bytes of the reversed prefix are the ones nearest
            // `te`, which is where the alignment ends and therefore where it must start being read.
            let t = &j.target[..=te as usize];
            rev_arena.extend_from_slice(&t[t.len() - rev_tlen(i)..]);
            rev_arena[t_off..].reverse();
            rev_spans.push((q_off, t_off - q_off, t_off, rev_arena.len() - t_off));
            rev_of_job.push(i);
        }
    }
    debug_assert_eq!(rev_arena.len(), arena_bytes, "arena sizing must be exact");
    // The reverse batch, one per qualifying job: `minsc = i32::MAX` suppresses the `score2` list
    // (a 2nd-best is only ever taken from the forward pass), and `endsc = out[i].score` stops each
    // lane the instant it matches the forward score, which is the whole reason this pass is cheap.
    let rev_jobs: Vec<FwdJob> = rev_spans
        .iter()
        .zip(rev_of_job.iter())
        .map(|(&(q_off, q_len, t_off, t_len), &i)| FwdJob {
            query: &rev_arena[q_off..q_off + q_len],
            target: &rev_arena[t_off..t_off + t_len],
            minsc: i32::MAX,
            endsc: out[i].score,
        })
        .collect();
    // The reverse pass: banded around the diagonal where the band kernel runs (every qualifying
    // job is under the u8 ceiling, as its forward pass was), else the forward kernel on the whole
    // reversed prefixes.
    #[allow(unused_mut)]
    let mut rev_results: Vec<(i32, i32, i32, i32, i32)> = Vec::new();
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let mut banded_rev = false;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if band_ok && !rev_jobs.is_empty() {
        let rj: Vec<(&[u8], &[u8], i32)> = rev_jobs
            .iter()
            .map(|j| (j.query, j.target, j.endsc))
            .collect();
        let mut r3 = vec![(0i32, -1i32, -1i32); rj.len()];
        // SAFETY: as for `band.run`.
        unsafe {
            band.run_reverse(&rj, mat, o_del, e_del, o_ins, e_ins, max_sc, &mut r3);
        }
        rev_results = r3
            .into_iter()
            .map(|(s, te, qe)| (s, te, qe, -1, -1))
            .collect();
        banded_rev = true;
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        band.clear();
        BAND.with(|c| c.replace(band));
        if !banded_rev {
            rev_results = fwd_local_sw_batch(&rev_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rev_results = fwd_local_sw_batch(&rev_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
    }
    for (k, &i) in rev_of_job.iter().enumerate() {
        // `rscore` should equal the forward score; `rte`/`rqe` are end offsets *in the reversed*
        // coordinates, i.e. how many bases back from `te`/`qe` the alignment started. The two
        // discarded fields are `score2`/`te2`, meaningless here because `minsc` was `i32::MAX`.
        let (rscore, rte, rqe, _, _) = rev_results[k];
        // Only trust the reverse pass when it reproduced the forward score exactly; otherwise the
        // two passes found different alignments and the offsets would not correspond. bwa applies
        // the same equality guard, leaving `qb`/`tb` at -1 when it fails.
        if out[i].score == rscore {
            out[i].tb = out[i].te - rte;
            out[i].qb = out[i].qe - rqe;
        }
    }
    if let Some(started_at) = timer {
        cells::NS.fetch_add(
            started_at.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        for r in &out {
            cells::count_outcome(r.score >= minsc && r.qb >= 0);
        }
    }
    out
}

/// The pruning bound for this batch's scoring, or `None` when pruning is off (`BWA4_RESCUE_PRUNE=0`)
/// or the lemma does not cover the scoring. The lemma needs every A/C/G/T cell of `mat` to score at
/// most `max_sc` on the diagonal and at most `-b` off it; `b` is read off the matrix as its weakest
/// mismatch penalty, so a non-uniform matrix is bounded by its kindest mismatch. N cells do not
/// matter: a job holding an N is never pruned.
#[allow(clippy::too_many_arguments)]
fn prune_params(
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Option<rescue_prune::Params> {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_PRUNE").is_none_or(|v| v != "0")) || m < 4
    {
        return None;
    }
    let mut b = i32::MAX;
    for t in 0..4 {
        for q in 0..4 {
            let v = mat[t * m + q] as i32;
            if t == q {
                if v > max_sc {
                    return None;
                }
            } else {
                b = b.min(-v);
            }
        }
    }
    rescue_prune::Params::from(max_sc, b, o_del, e_del, o_ins, e_ins, minsc)
}

/// The hit gate of [`rescue_prune::decide`]: a window sharing more K-mer hits than this with its
/// mate runs in full, because enumerating them would cost more than the rows the bound could save.
/// 1000 is the best of {400, 1000, 2000} on real GIAB reads with the banded kernel (2M HG002 pairs,
/// M4: 90.6 s of mate-rescue CPU at 1000, 117.6 at 400, 93.7 at 2000; a 1.4 kb window shares ~600
/// 5-mer hits with its mate). Changes speed only, never output. `BWA4_RESCUE_PRUNE_MAX_HITS`
/// overrides it.
fn prune_max_hits() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BWA4_RESCUE_PRUNE_MAX_HITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000)
    })
}

/// Whether the reverse (`KSW_XSTART`) pass truncates its target by the alignment-span bound
/// (issue #50A). `BWA4_RESCUE_REVBOUND=0` keeps the full `target[..=te]` prefix. Read once, cached.
fn revbound_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_REVBOUND").is_none_or(|v| v != "0"))
}

/// The most target rows an alignment of at most `qlen` query bases can span while still scoring
/// `score`. Issue #50A's lemma, and the whole of its correctness.
///
/// An alignment covering `T` target rows and `Q <= qlen` query columns has `T - Q` net unmatched
/// target bases when `T > Q`. Those form at least one gap run in the query, so they cost at least
/// `o_del + (T - Q) * e_del`, and no cell can earn more than `max_sc`:
///
/// ```text
///   score <= max_sc * Q - o_del - (T - Q) * e_del
///         <= max_sc * qlen - o_del - (T - qlen) * e_del
///   =>  T <= qlen + (max_sc * qlen - score - o_del) / e_del
/// ```
///
/// When the bracket is negative the second case of the lemma applies (`T <= Q <= qlen`), so the
/// bound is `qlen` and the `max(0, .)` below is that case, not a defensive clamp. Integer division
/// floors, which is the safe direction: `T` is an integer bounded above by a real quantity.
///
/// # Parameters
/// - `qlen`: the reverse pass's query length, `qe + 1`, in bases.
/// - `score`: the score the reverse pass must reach, i.e. the forward pass's own score. Using the
///   job's score rather than `minsc` is what makes the bound tight; a higher score means a shorter
///   possible span.
/// - `max_sc`: the largest matrix entry, `> 0`. `o_del`, `e_del`: gap open and extend as positive
///   magnitudes; `e_del >= 1`.
///
/// # Returns
/// A row count `>= qlen`. Truncating the reverse target to it cannot remove any alignment that
/// scores `score`, and therefore cannot change `rscore`, `rte` or `rqe` for any job whose reverse
/// pass reaches the forward score, which is the only case the caller acts on.
fn rev_span_bound(qlen: usize, score: i32, max_sc: i32, o_del: i32, e_del: i32) -> usize {
    debug_assert!(max_sc > 0 && e_del >= 1);
    let slack = (max_sc * qlen as i32)
        .saturating_sub(score)
        .saturating_sub(o_del);
    qlen + (slack.max(0) / e_del) as usize
}

/// Whether the u8 rescue kernel may use its padding-free fast column range. `BWA4_RESCUE_FASTCOL=0`
/// forces `n_fast = 0`, which routes every column through the tail body and reproduces the kernel
/// exactly as it was before the split. Exists so the two can be A/B'd inside ONE binary, with
/// identical instrumentation on both arms; the value is read once and cached.
fn fastcol_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_FASTCOL").is_none_or(|v| v != "0"))
}

/// Both u8 kernels (NEON and AVX2) use it; the i16 kernels keep the one-row loop, see their notes.
/// Whether the u8 rescue kernel processes target rows in PAIRS. `BWA4_RESCUE_ROWPAIR=0` forces the
/// one-row-at-a-time loop, so the two can be A/B'd inside one binary. Read once and cached.
fn rowpair_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_ROWPAIR").is_none_or(|v| v != "0"))
}

/// Whether the NEON u8 rescue kernel processes target rows FOUR at a time. `BWA4_RESCUE_ROWQUAD=0`
/// drops back to the pair loop (itself still governed by [`rowpair_enabled`]), so the three widths
/// are A/B-able inside one binary. Read once and cached.
///
/// Same argument as the pair, taken one step further: a group of four rows loads the query column,
/// `e[j]` and `h_prev[j]` once and stores `e[j]` and `h_cur[j]` once, so five memory operations
/// cover four cells instead of the pair's five for two. The three interior E carries and the three
/// interior diagonals stay in registers and never reach memory at all.
///
/// NEON only for now, hence the `cfg`. The x86 u8 kernels keep the pair loop: AVX2 has 16 vector
/// registers against NEON's 32, and four rows need sixteen live accumulators (`f`, `d`, `imax`,
/// `col` per row) before the constants, so a quad body there would likely spill more than it saves.
/// Measure before porting.
#[cfg(target_arch = "aarch64")]
fn rowquad_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_ROWQUAD").is_none_or(|v| v != "0"))
}

/// Whether the NEON u8 rescue kernel gives the all-ZPAD query columns their own third body.
/// `BWA4_RESCUE_ZPADCOL=0` sends them back through the full tail body. Read once and cached.
///
/// Unlike the `USQADD` and `SHARE_OE` levers this is not a const parameter: it only moves the
/// `n_pad` boundary, computed once per group outside every loop, exactly as `n_fast` already is.
/// The AVX2 and AVX-512 u8 kernels read it too, for their two-row body.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn zpadcol_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_ZPADCOL").is_none_or(|v| v != "0"))
}

/// Whether the NEON u8 rescue kernel applies the substitution score with `USQADD` (`vsqaddq_u8`,
/// unsigned-saturating accumulate of a SIGNED addend) instead of the biased `vqadd` + de-biasing
/// `vqsub` pair. `BWA4_RESCUE_USQADD=0` restores the biased form. Read once and cached.
///
/// The two forms are separately monomorphised bodies rather than a branch inside the column loop,
/// for the reason issue #45 states and this project already paid for once: a loop-invariant `if`
/// inside the N repair was NOT unswitched by LLVM and cost 7%.
#[cfg(target_arch = "aarch64")]
fn usqadd_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_USQADD").is_none_or(|v| v != "0"))
}

/// Whether the NEON u8 rescue kernel shares ONE `vqsubq_u8(h, oe)` between the E and F recurrences.
/// `BWA4_RESCUE_SHAREOE=0` restores the split form. Read once and cached.
///
/// Legal only when `oe_del == oe_ins` (bwa's stock `-O 6,6 -E 1,1`), which the caller checks; the
/// dispatcher falls through to the split body otherwise. Same monomorphisation argument as
/// [`usqadd_enabled`].
#[cfg(target_arch = "aarch64")]
fn shareoe_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_SHAREOE").is_none_or(|v| v != "0"))
}

/// Lanes processed in lockstep per group. 8 = one NEON `int16x8`.
const LANES: usize = 8;

/// Query-column / target-row padding sentinel (`>= m`, so its cell score is forced very negative and
/// the padded cells stay `0` — neutral to the real lanes).
///
/// This fills two different holes. Along the query it marks columns past the *padded profile* of a
/// short lane; along the target it marks rows past a short lane's window. In both cases the cell is
/// killed outright (score `-30000` in i16, or a forced 0 in the u8 kernel where saturation makes any
/// large subtraction land on 0). Killing rather than merely masking matters because H is carried
/// diagonally: a padded cell that kept a live value would leak into the next real row.
///
/// 255 specifically because it must satisfy `> ZPAD` and `>= m` in *both* element types, and it is
/// the u8 kernel's largest value so `vcgtq_u8(q, zpad)` cannot be fooled.
const PAD: u8 = 255;
/// ksw's query-profile padding: `ksw_qinit` rounds the query up to a whole number of SIMD lanes and
/// fills the tail with **score 0**, so those columns leave `h = h_diag` (they carry a diagonal) and
/// still feed ksw's per-row max, hence `score2`. Distinct from [`PAD`], which marks cells past the
/// padded profile that must stay dead.
///
/// The C is explicit: `ksw.cpp:96` and `:105` write `(k >= qlen? 0 : ma[query[k]])`, i.e. score 0,
/// not a large negative sentinel. That choice is *observable*, which is why it has to be emulated
/// rather than optimized away. A padding column scoring 0 leaves `H(i,j) = H(i-1,j-1)`, so it copies
/// the diagonal one step further right and one step further down; the value can therefore reappear
/// in a later row's max and change `score2`, and in principle `gmax`/`te` too. Dropping the padding
/// (the "obvious" simplification, since these columns are past the end of the query) silently
/// changes output on some reads.
///
/// Value 5 = `m`, one past the largest real code (4 = N). It is distinguishable from both a real
/// base and from [`PAD`], which is what the three-way select in the kernels keys on.
const ZPAD: u8 = 5;

/// Code the NEON u8 rescue kernel substitutes for a TARGET N (bwa's code 4) inside its packed
/// `seq_t`, so that `target XOR query` lands on a table slot of its own in every case, including
/// both-N. See the score-table comment in `fwd_local_sw_neon_u8` for the full index map and for why
/// this removes four vector operations per column of a row quad.
///
/// 12 rather than any other free code because `12 XOR 0..3` is 12-15 and `12 XOR 4` is 8, so all
/// five reachable indices stay inside the 16-entry table `vqtbl1q_u8` can address, and none of them
/// collides with the 0-7 range the real bases already use. Purely internal to that one kernel: it is
/// written when the group is packed and never leaves it, so no caller and no other kernel sees it.
///
/// Shared by the NEON u8 kernel and, since issue #43, by the AVX2 and SSE4.1 u8 kernels, which now
/// use the same XOR-indexed table. `vpshufb` reads `tbl[idx & 15]` where `vqtbl1q_u8` zeroes any
/// index >= 16, but the two agree over every index this encoding can produce; the argument is spelled
/// out where the x86 table is built. The AVX-512 u8 kernel still scores with blends (issue #44).
const N_TARGET: u8 = 12;

/// Cell score written where a [`PAD`] position is scored, i.e. where the cell must come out dead.
///
/// Chosen to satisfy three constraints at once: negative enough to drive any reachable `H` to 0
/// through the `max(0, .)` clamp, small enough in magnitude that `h_diag + this` cannot overflow,
/// and inside `i16` so the vector kernel can use the identical value. The u8 kernel writes a plain
/// 0 instead, which is the same thing after the clamp. Not `i16::MIN`: repeated additions to it
/// would wrap.
const DEAD_CELL_SCORE: i32 = -30_000;

/// Score ceiling under which the 16-lane u8 kernel is exact. Also the query-length cap, because
/// that kernel keeps the argmax *column* in the same u8 lane as the scores.
///
/// bwa uses the same number for a related but distinct decision: `mem_matesw` picks its `KSW_XBYTE`
/// (u8) kernel when `read_len * a < 250` (`bwamem_pair.cpp:208`). See [`ksw_padded_qlen`], which
/// must reproduce bwa's choice rather than ours.
const U8_SCORE_LIMIT: i32 = 250;

/// Whether the u8 rescue kernels' substitution table can hold this scoring matrix without
/// saturating, i.e. whether the u8 path is legal AT ALL for these scores.
///
/// The u8 kernels apply the substitution score through a 16-entry table biased by the mismatch
/// penalty, so that every entry is a non-negative magnitude and one saturating add plus one
/// saturating subtract reproduce `max(0, h + S)`. The bias costs headroom: the largest value the
/// biased add can produce is `h + mispen + mtch`, and with `h` capped at [`U8_SCORE_LIMIT`] that
/// must stay inside a byte. Each u8 kernel asserts exactly this, and the assertion is hard rather
/// than `debug_assert` because the release build's correctness rests on it.
///
/// WHY IT IS A GATE AND NOT A PANIC: the score ceiling test next to this one is about the JOBS, and
/// a batch that fails it falls to the i16 kernel. This test is about the OPTIONS, and until it was
/// added a batch that failed it reached the assertion instead. `-x intractg` (`-B 9`) and any
/// `-A 3` or `-B 6` run therefore aborted the whole aligner on the first paired-end rescue batch,
/// where bwa-mem2 aligns them fine. Falling to i16 is free of output risk: which internal kernel
/// width runs a batch is invisible to the result, unlike [`ksw_padded_qlen`]'s `lanes`, which is
/// bwa's own u8/i16 choice and does move the bytes.
///
/// `mat` must already have passed [`mat_is_standard`], which is what makes `mat[0]`/`mat[1]` the
/// match bonus and the (negated) mismatch penalty for every cell of the matrix.
fn u8_table_has_headroom(mat: &[i8]) -> bool {
    // Match bonus `a` and mismatch penalty `b`, as the non-negative magnitudes the kernels use.
    let mtch = i32::from(mat[0]);
    let mispen = -i32::from(mat[1]);
    U8_SCORE_LIMIT + mispen + mtch <= 256
}

/// Score ceiling under which the 8-lane i16 kernel is exact, with headroom left for
/// [`DEAD_CELL_SCORE`] at the other end of the range. Jobs above this fall back to the scalar path.
///
/// Gated on aarch64/x86_64 because its only readers are the NEON and AVX2/AVX512 dispatches. Without
/// the gate a scalar-only build sees an unused constant, which is only a warning locally but a hard
/// error under CI's `-D warnings`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const I16_SCORE_LIMIT: i32 = 30_000;

/// Length ksw pads a query of `qlen` out to. The lane count is bwa's kernel choice in `mem_matesw`
/// (`l_ms * opt->a < 250 ? KSW_XBYTE : 0`), i.e. u8/16 lanes or i16/8 lanes -- not our SIMD width.
///
/// This is the subtlest coupling in the file. `ksw_qinit` segments the query into `slen = ceil(qlen/p)`
/// vectors of `p` values, where `p = 8 * (3 - size)` (`ksw.cpp:68-69`): `size = 1` (KSW_XBYTE, u8)
/// gives p = 16, `size = 2` (i16) gives p = 8. bwa picks `size` in `bwamem_pair.cpp:208` from
/// `l_ms * opt->a < 250 ? KSW_XBYTE : 0`, i.e. from the *mate read's length times the match score*.
///
/// So the amount of zero-padding, and therefore the result, depends on which kernel width bwa would
/// have chosen for this query, entirely independently of which kernel width *we* choose to run it
/// on. Deriving `lanes` from our own SIMD register instead would produce a different padded length
/// and a different `score2`. That is why the test in this file recomputes `lanes` the same way
/// before calling the scalar `ksw_align2`.
///
/// # Parameters
/// - `qlen`: the real query length in bases, before any padding.
/// - `max_sc`: the match bonus `opt->a` in score units, `> 0`. Only used to reproduce bwa's kernel
///   choice; it does not scale the result.
///
/// # Returns
/// `qlen` rounded up to a multiple of 16 (if `qlen * max_sc < 250`) or of 8 (otherwise). This is the
/// number of DP columns the pass must actually run: the extra columns score 0 rather than being
/// skipped, and they are observable in `score2`.
/// Whether to length-sort the rescue batch before the forward kernel. Default ON; set
/// `BWA4_MATESW_SORT=0` to compare against the caller-order grouping.
fn sort_jobs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_MATESW_SORT").is_none_or(|v| v != "0"))
}

fn ksw_padded_qlen(qlen: usize, max_sc: i32) -> usize {
    // Deliberately bwa's choice, not ours: this is `mem_matesw`'s `l_ms * opt->a < 250` test, and
    // the width it yields decides how much zero padding the query gets, which is observable.
    // 16 = ksw's u8 profile width, 8 = its i16 width; unrelated to the register we happen to use.
    let lanes = if qlen as i32 * max_sc < U8_SCORE_LIMIT {
        16
    } else {
        8
    };
    qlen.div_ceil(lanes) * lanes
}

/// Whether a job of these lengths runs on the u8 rescue kernels (with [`u8_table_has_headroom`]):
/// every cell under the score ceiling `min(qlen, tlen) * max_sc`, plus the biased table add's
/// `mispen + mtch`, fits a byte, and the padded query's columns are at most 256. The fork's
/// `matesw_use_u8` bound (`l a + shift <= 254`); at bwa's scores it admits the 250 bp mate the
/// historical `ceiling < 250` sent to i16. Which kernel runs never moves the bytes; the padding
/// stays [`ksw_padded_qlen`]'s, bwa's own.
fn u8_job_fits(qlen: usize, tlen: usize, mat: &[i8], max_sc: i32) -> bool {
    let ceiling = qlen.min(tlen) as i32 * max_sc;
    ceiling - i32::from(mat[1]) + i32::from(mat[0]) <= 255 && ksw_padded_qlen(qlen, max_sc) <= 256
}

/// Batched forward local-SW pass: `out[i] = (score, te, qe, score2, te2)` for `jobs[i]`, each equal to
/// [`ksw_local_fwd`]. Dispatches to the NEON i16 kernel where available, else the scalar lockstep.
///
/// # Parameters
/// Same meaning as on [`batched_ksw_align2`], except that the per-job `minsc`/`endsc` now travel
/// inside each [`FwdJob`] rather than being one value for the whole batch. `jobs` may mix the
/// forward and reverse conventions freely; the dispatch below only looks at the sequence lengths.
///
/// # Returns
/// `out[i] = (score, te, qe, score2, te2)` for `jobs[i]`, in input order. `score` is in score units
/// and `>= 0`; `te`/`qe` are inclusive 0-based row/column positions (`te = -1`, `qe = -1` when no
/// cell exceeded 0); `score2`/`te2` are the best rival alignment and its row, both `-1` when none
/// qualifies (always the case when the caller set `minsc = i32::MAX`). Note `qe` is a column of the
/// *padded* query, so it can in principle exceed the real query length.
///
/// The three implementations are interchangeable and must stay byte-identical; the choice between
/// them is purely which one is legal for these lengths and scores, never a quality tradeoff.
/// Runtime override of the rescue-kernel tier, mirroring the fork's `BWAMEM3_FORCE_TIER`
/// (fg-labs/bwa-mem3, `src/simd_dispatch.cpp`). Two uses: a same-host A/B that isolates what the
/// AVX2 / AVX-512 / scalar rescue kernel actually costs end-to-end (the whole-genome head-to-head
/// cannot attribute time to one stage), and a pin for a user whose CPU regresses on the widest tier.
/// The fork's own characterization is why the pin matters: AVX-512 at 512-bit width is only ~+2-4% on
/// Intel Sapphire Rapids and roughly break-even on AMD Zen 4, where every 512-bit op issues as 2x
/// 256-bit uops (`docs/src/whats-different/avx512-baseline.md` in the fork).
///
/// `BWA4_RESCUE_TIER=scalar|avx2|avx512`; anything else, or unset, means auto = widest available.
/// Every tier is byte-identical, so this changes speed only, never output. Read once and cached.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RescueTier {
    Auto,
    Scalar,
    Sse41,
    Avx2,
    Avx512,
}

/// Parse and cache `BWA4_RESCUE_TIER`. On first read it also prints the resolved override to stderr
/// (once) when one is set, mirroring the fork's `BWAMEM3_DEBUG_SIMD` confirmation so a benchmark run
/// can prove which path it took.
fn forced_rescue_tier() -> RescueTier {
    use std::sync::OnceLock;
    static TIER: OnceLock<RescueTier> = OnceLock::new();
    *TIER.get_or_init(|| match std::env::var("BWA4_RESCUE_TIER").ok().as_deref() {
        Some("scalar") => {
            eprintln!("[bwa-mem4] BWA4_RESCUE_TIER=scalar: forcing scalar mate-rescue kernel");
            RescueTier::Scalar
        }
        Some("sse41") => {
            eprintln!("[bwa-mem4] BWA4_RESCUE_TIER=sse41: capping mate-rescue kernel at SSE4.1");
            RescueTier::Sse41
        }
        Some("avx2") => {
            eprintln!("[bwa-mem4] BWA4_RESCUE_TIER=avx2: capping mate-rescue kernel at AVX2");
            RescueTier::Avx2
        }
        Some("avx512") => {
            eprintln!("[bwa-mem4] BWA4_RESCUE_TIER=avx512: requiring AVX-512 mate-rescue kernel");
            RescueTier::Avx512
        }
        _ => RescueTier::Auto,
    })
}

/// Whether the rescue kernels may take their AVX-512BW form: the CPU has it and
/// `BWA4_RESCUE_TIER` does not cap the tier below it.
#[cfg(target_arch = "x86_64")]
pub(crate) fn rescue_avx512_ok() -> bool {
    matches!(forced_rescue_tier(), RescueTier::Auto | RescueTier::Avx512)
        && bwa_extend::has_avx512()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_local_sw_batch(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    // A forced `scalar` tier bypasses every vector path on all architectures (the fork's
    // `BWAMEM3_FORCE_TIER=scalar`, and the `BWA3_SCALAR_RESCUE` toggle issue #12 measured with).
    let tier = forced_rescue_tier();
    if tier == RescueTier::Scalar {
        return fwd_local_sw_scalar(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // avx2/avx512 forces have no aarch64 meaning; NEON is the only vector tier here.
        let standard = mat_is_standard(m, mat);
        if std::arch::is_aarch64_feature_detected!("neon") && (standard || mat5_u8_ok(m, mat)) {
            // Max reachable score per job = min(len) * match. Only the SCORE cells (H/E/F) live in the
            // SIMD vector; positions/te/qe are scalar i32, so window length is unconstrained. If every
            // job's score ceiling fits u8, run 16 lanes; else the i16 kernel at 8 lanes. Mate-rescue
            // jobs (short reads, match ~1) fit u8.
            // A local alignment can match at most `min(qlen, tlen)` bases and only loses score to
            // mismatches and gaps, so `min(len) * max_sc` is a hard ceiling on every H/E/F cell.
            let score_ceiling = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            // u8 also holds the argmax query column, so the query must be < 256 too.
            if u8_table_has_headroom(mat)
                && jobs
                    .iter()
                    .all(|j| u8_job_fits(j.query.len(), j.target.len(), mat, max_sc))
            {
                // SAFETY: neon detected; every job passes `u8_job_fits`; standard
                // mat; and the biased substitution table has byte headroom for it.
                return unsafe {
                    fwd_local_sw_neon_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
            }
            if jobs.iter().all(|j| {
                score_ceiling(j) < I16_SCORE_LIMIT && j.target.len() < I16_SCORE_LIMIT as usize
            }) {
                // SAFETY: neon detected; i16 range guaranteed; standard or MAT5 mat.
                return unsafe {
                    if standard {
                        fwd_local_sw_neon::<false>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    } else {
                        fwd_local_sw_neon::<true>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    }
                };
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // A `--meth` matrix (`mat5_u8_ok`) takes the u8 and i16 kernels' `MAT5` form.
        // shortcut: a `MAT5` matrix without an add code (every `--meth` matrix has one) runs scalar,
        // AVX-512 with VBMI included, which would not need it; gate VBMI apart if one ever matters.
        let standard = mat_is_standard(m, mat);
        if standard || (mat5_u8_ok(m, mat) && mat5_add_code(mat).is_some()) {
            // Same score-ceiling test as the aarch64 arm above, verbatim: the threshold is bwa's
            // (`bwamem_pair.cpp:208`), not a property of the register we happen to run on, so it does
            // not change when the vector gets twice or four times as wide. See `ksw_padded_qlen` for
            // why deriving any of this from the SIMD width would alter the output.
            let score_ceiling = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            // shortcut: the x86 u8 kernels subtract the gap terms as signed bytes, so an open+extend
            // above 127 (never bwa's defaults) goes to i16 or scalar; widen the guard if that matters.
            let fits_u8 = u8_table_has_headroom(mat)
                && o_del + e_del <= 127
                && o_ins + e_ins <= 127
                && jobs
                    .iter()
                    .all(|j| u8_job_fits(j.query.len(), j.target.len(), mat, max_sc));
            let fits_i16 = jobs.iter().all(|j| {
                score_ceiling(j) < I16_SCORE_LIMIT && j.target.len() < I16_SCORE_LIMIT as usize
            });
            // The i16 kernels' standard or `MAT5` form.
            macro_rules! i16_kernel {
                ($k:ident) => {
                    if standard {
                        $k::<false>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    } else {
                        $k::<true>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    }
                };
            }
            // Prefer the widest legal vector: AVX-512BW (64 u8 / 32 i16 lanes) when the CPU has it,
            // else AVX2 (32 / 16). The width is purely a throughput choice; every kernel is
            // byte-identical, so the only thing the branch decides is how many lanes run at once.
            // `avx512bw` implies `avx512f`, which is what the word/byte ops need. `BWA4_RESCUE_TIER`
            // narrows the choice: `avx2` skips the AVX-512 branch, `avx512` skips the AVX2 one (so a
            // forced-avx512 run on a host without it falls to scalar, never silently to AVX2).
            let want_avx512 = matches!(tier, RescueTier::Auto | RescueTier::Avx512);
            // `Auto` here means "the tier the extension kernels calibrated to". Rescue and extension
            // are the same kind of work on the same registers, so a host where 256-bit is a bad deal
            // for one is a host where it is a bad deal for the other, and running a second timing
            // probe to rediscover that would only cost milliseconds to reach the same answer.
            let want_avx2 = match tier {
                RescueTier::Avx2 => true,
                RescueTier::Auto => crate::batched::prefers_wide_x86(),
                _ => false,
            };
            let want_sse41 = matches!(
                tier,
                RescueTier::Auto | RescueTier::Sse41 | RescueTier::Avx2
            );
            if want_avx512 && bwa_extend::has_avx512() {
                if fits_u8 {
                    // SAFETY: AVX-512 BW/VL/DQ detected; u8 preconditions hold; standard or MAT5 mat.
                    return unsafe {
                        fwd_local_sw_avx512_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: AVX-512 BW/VL/DQ detected; i16 range guaranteed; standard or MAT5 mat.
                    return unsafe { i16_kernel!(fwd_local_sw_avx512_i16) };
                }
            }
            if want_avx2 && std::arch::is_x86_feature_detected!("avx2") {
                if fits_u8 {
                    // SAFETY: avx2 detected; every job passes `u8_job_fits`; standard or MAT5 mat.
                    return unsafe {
                        fwd_local_sw_avx2_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: avx2 detected; i16 range guaranteed; standard or MAT5 mat.
                    return unsafe { i16_kernel!(fwd_local_sw_avx2_i16) };
                }
            }
            if want_sse41 && std::arch::is_x86_feature_detected!("sse4.1") {
                if fits_u8 {
                    // SAFETY: sse4.1 detected; u8 preconditions as for the AVX2 arm; standard or MAT5 mat.
                    return unsafe {
                        fwd_local_sw_sse41_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: sse4.1 detected; i16 range guaranteed; standard or MAT5 mat.
                    return unsafe { i16_kernel!(fwd_local_sw_sse41_i16) };
                }
            }
        }
    }
    fwd_local_sw_scalar(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
}

/// Whether the NEON u8 rescue kernel's 5x5 table form (`MAT5`) takes `mat`: a 5x5 matrix whose
/// every cell lies between the mismatch `mat[1]` and the match `mat[0]`, with the byte headroom the
/// standard form needs. `--meth`'s matrices are of this shape (the freed cell is 0 or `a`).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn mat5_u8_ok(m: usize, mat: &[i8]) -> bool {
    m == 5
        && mat[1] <= 0
        && mat[..25].iter().all(|v| (mat[1]..=mat[0]).contains(v))
        && u8_table_has_headroom(mat)
}

/// A code that scores a 5x5 `mat` with one 16-entry `pshufb` at `(t + q) & 15`: target base `b`
/// packs as `t[b]`, query base `b` as `q[b]`, and `tbl[(t[x] + q[y]) & 15] == mat[5 x + y]` for
/// every pair. So the x86 u8 kernels score a `--meth` matrix with one lookup per cell, as they do the
/// standard XOR table, instead of `5 t + q` over 25 entries (two lookups, a subtract and an xor).
/// Every code is below 16 (bit 7 stays `PAD`'s), and no query code is `ZPAD`.
///
/// XOR cannot do it: the nine N cells need nine slots of their own, and in GF(2)^4 no 4x4 block in
/// the seven left has a sum with one representation, which the freed cell needs. Addition mod 16
/// has a code for every `--meth` matrix (`mat5_add_code_covers_meth_matrices`). Depth-first over the
/// ten codes with `t[A] = 0`: adding a constant to every `t` and subtracting it from every `q`
/// changes nothing.
#[cfg(target_arch = "x86_64")]
fn mat5_add_code(mat: &[i8]) -> Option<([u8; 5], [u8; 5], [i8; 16])> {
    struct Search<'a> {
        mat: &'a [i8],
        t: [u8; 5],
        q: [u8; 5],
        val: [i8; 16],
        refs: [u8; 16],
        budget: u32,
    }
    // Level `k` sets `t[k / 2]` (even `k`) or `q[k / 2]` (odd) and places the cells it completes:
    // row `i` against `q[..i]`, or column `i` against `t[..=i]`.
    fn go(s: &mut Search, k: usize) -> bool {
        if k == 10 {
            return true;
        }
        let (is_t, i) = (k.is_multiple_of(2), k / 2);
        for c in 0..16u8 {
            // shortcut: a matrix the budget does not settle runs on the scalar kernel; no `--meth`
            // matrix comes near it (the search ends within a few hundred steps).
            if s.budget == 0 {
                return false;
            }
            s.budget -= 1;
            let taken = if is_t {
                s.t[..i].contains(&c)
            } else {
                s.q[..i].contains(&c)
            };
            if taken || (is_t && i == 0 && c != 0) || (!is_t && c == ZPAD) {
                continue;
            }
            let n = if is_t { i } else { i + 1 };
            let (mut slots, mut placed) = ([0usize; 5], 0);
            for o in 0..n {
                let (x, y, sum) = if is_t {
                    (i, o, c + s.q[o])
                } else {
                    (o, i, s.t[o] + c)
                };
                let (slot, v) = ((sum & 15) as usize, s.mat[5 * x + y]);
                if s.refs[slot] > 0 && s.val[slot] != v {
                    break;
                }
                s.val[slot] = v;
                s.refs[slot] += 1;
                slots[placed] = slot;
                placed += 1;
            }
            if placed == n {
                if is_t {
                    s.t[i] = c;
                } else {
                    s.q[i] = c;
                }
                if go(s, k + 1) {
                    return true;
                }
            }
            for &slot in &slots[..placed] {
                s.refs[slot] -= 1;
            }
        }
        false
    }
    let mut s = Search {
        mat,
        t: [0; 5],
        q: [0; 5],
        val: [0; 16],
        refs: [0; 16],
        budget: 1 << 20,
    };
    if !go(&mut s, 0) {
        return None;
    }
    let mut tbl = [0i8; 16];
    for (k, e) in tbl.iter_mut().enumerate() {
        if s.refs[k] > 0 {
            *e = s.val[k];
        }
    }
    Some((s.t, s.q, tbl))
}

/// Whether `mat` is bwa's standard 5x5 form (uniform match on the diagonal, uniform mismatch
/// off-diagonal for `0..4`, `-1` for any N row/col) so the NEON kernel can compute cell scores from
/// three scalars instead of a per-cell table lookup.
///
/// # Parameters
/// - `m`: matrix side. Anything other than 5 fails immediately, so `mat` is never indexed past 25.
/// - `mat`: the row-major `m * m` score matrix to test. Must hold at least 25 entries when `m == 5`.
///
/// # Returns
/// `true` when the NEON kernels' three-constant shortcut is exact for this matrix. A `false` is not
/// an error: the caller simply keeps the scalar path, which reads `mat` cell by cell and therefore
/// handles arbitrary matrices.
///
/// Gated on aarch64/x86_64: its only callers are the NEON and AVX2 dispatches, so elsewhere it is
/// dead code and `-D warnings` would reject it.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn mat_is_standard(m: usize, mat: &[i8]) -> bool {
    if m != 5 {
        return false;
    }
    // The two constants the NEON kernels broadcast: the diagonal (match) score and one off-diagonal
    // (mismatch) score, taken as representative and then verified against every other entry below.
    let (mtch, mis) = (mat[0], mat[1]);
    for i in 0..4 {
        for j in 0..4 {
            // The score this cell must hold for the shortcut to be exact.
            let want = if i == j { mtch } else { mis };
            if mat[i * 5 + j] != want {
                return false;
            }
        }
        if mat[i * 5 + 4] != -1 || mat[4 * 5 + i] != -1 {
            return false;
        }
    }
    mat[4 * 5 + 4] == -1
}

/// Scalar lockstep reference: processes `LANES` jobs with shared row/column loops and per-lane state
/// and masking, the structure the NEON kernel vectorizes. The scalar per-cell arithmetic here is the
/// byte-identity source of truth (`matesw_equals_scalar`).
///
/// It is "lockstep" rather than a plain per-job loop on purpose: every array here is indexed
/// `[position * LANES + lane]`, exactly the layout the NEON kernels load from, so the vector kernels
/// are this function with the `for l in 0..LANES` loop deleted and the body replaced by intrinsics.
/// Reading the two side by side is the intended way to check the vectorization.
///
/// # Parameters
/// Identical to [`fwd_local_sw_batch`]. Unlike the NEON paths this imposes no score or length
/// ceiling and accepts any `mat`, which is why it is the fallback.
///
/// # Returns
/// As [`fwd_local_sw_batch`]: `(score, te, qe, score2, te2)` per job, in input order.
#[allow(clippy::too_many_arguments)]
fn fwd_local_sw_scalar(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    // Cost of a gap's *first* base: open plus one extend. The recurrence never uses `o_*` alone, so
    // both are folded once here rather than per cell.
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    for (group_idx, group) in jobs.chunks(LANES).enumerate() {
        // Live lanes: `LANES` for every group but possibly the last, which is partly padding. Lanes
        // `n_lanes..LANES` still run the arithmetic (their sequences are all-[`PAD`], so their cells
        // die) but are skipped by every piece of bookkeeping and never written to `out`.
        let n_lanes = group.len();
        // The DP rectangle for the whole group: `qmax` padded query columns and `tmax` target rows,
        // both the max over the group's lanes so all lanes can share one pair of loops. A lane
        // shorter than the max wastes the difference, which is why the caller length-sorts.
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // --- group setup: SoA sequences (padded), per-lane bounds and stop conditions ---------
        // Interleaved (struct-of-arrays) sequences: `seq_q[j * LANES + l]` is query base `j` of lane
        // `l`, so one contiguous load gathers column `j` across every job at once, which is the
        // entire reason for this layout. Pre-filled with [`PAD`] so any cell a lane does not really
        // have is dead by default; the loops below overwrite only the live region.
        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        // Per-lane scalars, indexed by lane. `qlen`/`tlen` are the *real* (unpadded) lengths in
        // bases, used to tell a lane's own rectangle from the group's; `minsc`/`endsc` are that
        // job's cutoffs in score units. Dead lanes keep `0` lengths and `i32::MAX` cutoffs, so they
        // are excluded by the `i >= tlen[l]` test and can never trip `endsc`.
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES],
            [0usize; LANES],
            [i32::MAX; LANES],
            [i32::MAX; LANES],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES + l] = b;
            }
            // Columns from this lane's real length up to *its own* padded length get [`ZPAD`]
            // (score 0, carries the diagonal). Columns past that, out to the group's `qmax`, keep
            // the [`PAD`] fill and stay dead. Both regions exist and they behave differently.
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES + l] = ZPAD;
            }
            // The target gets no ZPAD region: ksw pads only the query profile.
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = b;
            }
        }

        // DP state (SoA over query columns). Two H rows are kept because the recurrence needs both
        // H(i-1, j) (to become the next cell's diagonal) and H(i, j) (being written); they are
        // swapped at the end of each row rather than copied. `e` persists across rows because E is
        // a per-column carry. There is no F array: F only moves rightwards within a row, so it lives
        // in a scalar (`f`) here and in a register in the SIMD kernels.
        // `h_prev[j * LANES + l]` = H(i-1, j) for lane `l`, the row just finished; `h_cur` = H(i, j),
        // the row being written. `e[j * LANES + l]` = E(i, j), the "a deletion is open in column j"
        // score, which survives the row swap because E propagates downwards. All three start at 0,
        // which is the correct local-SW initial row: there is no h0 and no first-row setup.
        let mut h_prev = vec![0i32; qmax * LANES];
        let mut h_cur = vec![0i32; qmax * LANES];
        let mut e = vec![0i32; qmax * LANES];
        // Every row's max is retained, not just the best: `score2` is chosen later from the list of
        // row maxima far enough from `te` (`ksw.cpp:194-202`), so the row maxima cannot be reduced
        // on the fly.
        let mut rowmax = vec![0i32; tmax * LANES]; // per-row imax, for score2
                                                   // Best H seen in any row so far, per lane, and where it was: `te` the target row (`-1` until
                                                   // some cell beats 0, which is what "no alignment" looks like), `qe` the padded query column
                                                   // in that row. `qe` starts at 0 rather than -1 because it is only ever read when `te >= 0`.
        let mut gmax = [0i32; LANES];
        let mut te = [-1i32; LANES];
        let mut qe = [0i32; LANES]; // argmax column at the best row
                                    // `limit` is where the C's row loop stopped: it truncates the `score2` candidate list to the
                                    // rows actually visited. Without it, an early-stopped lane would contribute row maxima that
                                    // bwa never computed.
        let mut limit = [-1i32; LANES]; // last processed row (inclusive)
                                        // A lane that hit `endsc` stops updating (bwa `break`s out of the row loop, `ksw.cpp:207`);
                                        // lanes cannot break individually, so it goes idle while its neighbours finish.
        let mut frozen = [false; LANES];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }

        // =====================================================================================
        // Main DP. Reminder: H = best score ending at this cell, E = with a deletion (gap in the
        // query) open, F = with an insertion (gap in the target) open. Both gaps open from H here.
        // =====================================================================================
        // Target rows, shared by all lanes and run to the longest window in the group.
        for i in 0..tmax {
            // F and the diagonal carry both reset at the start of every row: a local alignment
            // cannot have a gap hanging over from the previous row's right edge.
            //
            // Loop invariant, per lane, at the top of the column iteration for `j`:
            //   f[l]      = F(i, j),   the "an insertion is open" score for the cell about to be computed
            //   h_diag[l] = H(i-1, j-1), the diagonal predecessor of that same cell
            //   imax[l]   = max H(i, 0..j), and imax_col[l] the smallest column attaining it
            let mut f = [0i32; LANES];
            let mut h_diag = [0i32; LANES];
            let mut imax = [0i32; LANES];
            let mut imax_col = [0i32; LANES]; // min query column achieving this row's max
                                              // Query columns. Note this runs over the *padded* qmax, so the ZPAD columns are real
                                              // work that has to happen, not a rounding artifact (see `ksw_padded_qlen`).
            for j in 0..qmax {
                for l in 0..LANES {
                    // This lane's target base at row `i` and query base at column `j`: a code in
                    // `0..=4`, or `ZPAD` (5) for ksw profile padding, or `PAD` (255) for a position
                    // this lane does not have.
                    let t_code = seq_t[i * LANES + l] as usize;
                    let q_code = seq_q[j * LANES + l] as usize;
                    // Three-way cell score. Order matters: the PAD test comes first so a padded
                    // column can never be mistaken for profile padding.
                    // `sc` = S(target[i], query[j]) in score units, the substitution score added to
                    // the diagonal for this cell.
                    let sc = if t_code >= m || q_code > ZPAD as usize {
                        // Past the padded profile (or a dead row): kill the cell. See
                        // `DEAD_CELL_SCORE` for why this particular magnitude.
                        DEAD_CELL_SCORE
                    } else if q_code == ZPAD as usize {
                        0 // ksw profile padding: carries the diagonal
                    } else {
                        i32::from(mat[t_code * m + q_code])
                    };
                    // Flat SoA offset of cell (row `i`, column `j`) for this lane. The row is
                    // implicit: `h_cur`/`h_prev`/`e` each hold exactly one row at a time.
                    let cell = j * LANES + l;
                    // H(i,j) = max(0, H(i-1,j-1) + S, E(i,j), F(i,j)). The `max(0, .)` on the
                    // diagonal term is the local-alignment restart (`ksw.cpp:159-163`, where the
                    // unsigned saturating `subs_epu8` performs the clamp implicitly).
                    // Starts as the diagonal candidate, then absorbs E and F: by the end of the four
                    // statements below it is the final H(i, j) for this lane.
                    let mut h = h_diag[l] + sc;
                    if h < 0 {
                        h = 0;
                    }
                    if e[cell] > h {
                        h = e[cell];
                    }
                    if f[l] > h {
                        h = f[l];
                    }
                    // Row argmax with a **strict** `>`, so a tie keeps the *smallest* column. That
                    // reproduces `ksw.cpp:216-218`, which scans the saved H vector and only lowers
                    // `qe` on a tie. The opposite convention (as used in `crate::batched`, which
                    // mirrors a different C function) would shift `qb`/`qe` and change the CIGAR.
                    if h > imax[l] {
                        imax[l] = h;
                        imax_col[l] = j as i32;
                    }
                    h_cur[cell] = h;
                    // E(i+1,j): extend the column's deletion, or open one from H. Opening from **H**
                    // here, not from the diagonal term: `ksw.cpp:168` subtracts `oe_del` from `h`.
                    // This is the opposite convention to `crate::batched`, and it is correct in both
                    // places because they port different functions.
                    // `e_new` = E(i+1, j) under construction: first the "keep extending the deletion
                    // already open" branch, then the better of that and `open_del`, the cost of
                    // starting a fresh deletion here (H minus open plus one extend).
                    let mut e_new = e[cell] - e_del;
                    let open_del = h - oe_del;
                    if open_del > e_new {
                        e_new = open_del;
                    }
                    e[cell] = e_new.max(0);
                    // F(i,j+1): the same along the row (`ksw.cpp:172-174`).
                    // `f_new` = F(i, j+1): extend the insertion already open along this row, or open
                    // a new one from H. Assigned back into `f[l]`, which the next column reads.
                    let mut f_new = f[l] - e_ins;
                    let open_ins = h - oe_ins;
                    if open_ins > f_new {
                        f_new = open_ins;
                    }
                    f[l] = f_new.max(0);
                    // Load the diagonal for the *next* column before `h_prev` is swapped away. This
                    // is the same one-load-behind trick as `ksw.cpp:176` (`h = load(H0 + j)`).
                    h_diag[l] = h_prev[cell];
                }
            }
            // --- row epilogue: per-row bookkeeping (only lanes within target and not frozen) ---
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                rowmax[i * LANES + l] = imax[l];
                // Strict `>`: the *first* row reaching a score keeps `te` (`ksw.cpp:203`).
                if imax[l] > gmax[l] {
                    gmax[l] = imax[l];
                    te[l] = i as i32;
                    // bwa saves the whole H vector here and rescans it after the loop to find `qe`
                    // (`ksw.cpp:205-206`, `:216-218`). Tracking the argmax column inline is
                    // equivalent given the same tie-break, and it avoids copying a full H column
                    // every time the max improves.
                    qe[l] = imax_col[l];
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            // Swap instead of copy: this row's H becomes the next row's diagonal source.
            std::mem::swap(&mut h_prev, &mut h_cur);
        }

        extract_group(
            n_lanes, group_idx, LANES, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// Shared per-lane output extraction (`qe`, `score2`/`te2`) from a group's filled DP state, exactly
/// as [`ksw_local_fwd`]. Used by both the scalar and NEON DP paths so they cannot drift.
/// `rowmax` (per-row imax) is SoA `[row*lanes + lane]`; `qe` is the per-lane query end (the argmax
/// column at the best row), tracked inline in the DP so no H column has to be copied.
///
/// What `score2` is for: mate rescue needs to know whether the window contains a *second*, unrelated
/// place the read aligns nearly as well, because that is what makes the rescue ambiguous and drives
/// the mapping quality down. So bwa collects every row max at or above `minsc`, merges runs of
/// consecutive rows (they are almost certainly the same alignment sliding by a base), then reports
/// the best remaining candidate that is far enough from `te` to be a genuinely different alignment.
///
/// # Parameters
/// - `n_lanes`: how many lanes of this group hold real jobs, `1..=lanes`. Lanes at or above this
///   index are padding and are not written to `out`.
/// - `group_idx`: which chunk of the batch this is, so job `l` of this group is
///   `out[group_idx * lanes + l]`. Supplied by the DP's `chunks(..).enumerate()`.
/// - `lanes`: the chunk size the DP actually used, 8 ([`LANES`]) or 16 ([`LANES16`]). It is the
///   stride of `rowmax` and of the `out` index, so passing the wrong one silently misattributes
///   every result.
/// - `minsc`: per-lane 2nd-best cutoff in score units, indexed by lane. A row max below
///   `minsc[l]` is not a `score2` candidate; `i32::MAX` suppresses the list entirely.
/// - `max_sc`: the match bonus (largest matrix entry), `> 0`, used only to size the exclusion window
///   below. Must be positive or the ceiling division divides by zero.
/// - `gmax`: per-lane best H over the rows visited, in score units, `>= 0`. Becomes `score`.
/// - `te`: per-lane target row where `gmax` occurred, inclusive 0-based, `-1` if no cell beat 0.
/// - `qe`: per-lane padded-query column where `gmax` occurred, inclusive 0-based. Only meaningful
///   when `te[l] >= 0`.
/// - `limit`: per-lane last target row actually processed, inclusive; `-1` means no row was.
///   Truncates the candidate scan so rows an early-stopped lane never computed are not counted.
/// - `rowmax`: SoA `[row * lanes + lane]`, the max H of each target row, in score units. Rows past
///   a lane's `limit` are ignored, so their contents do not matter.
/// - `out`: destination, indexed `group_idx * lanes + l`. Must be at least that long. Only the
///   `n_lanes` live entries of this group are touched.
///
/// # Returns
/// Nothing; writes `(score, te, qe, score2, te2)` into `out` for each live lane.
#[allow(clippy::too_many_arguments)]
fn extract_group<R: Copy + Into<i32>>(
    n_lanes: usize,
    group_idx: usize,
    lanes: usize,
    minsc: &[i32],
    max_sc: i32,
    gmax: &[i32],
    te: &[i32],
    qe: &[i32],
    limit: &[i32],
    rowmax: &[R],
    out: &mut [(i32, i32, i32, i32, i32)],
) {
    let walks = |l: usize| limit[l] >= 0 && gmax[l] >= minsc[l];
    // The rows of each lane that clear its `minsc`, as `bits[blk * n_lanes + l]` bit `k` for row
    // `8 blk + k`: built with whole-row compares (all lanes of a row at once, the lane loop
    // vectorises), so each lane below visits only its own candidate rows instead of a strided load,
    // compare and branch per row. bwa-mem3's `kswv_score2_row_bits_u8`. Every visited row is
    // re-tested by `push_row`, and the rows skipped are exactly those it would drop, so `b` is
    // filled with the same entries in the same order.
    let nrows = (0..n_lanes)
        .filter(|&l| walks(l))
        .map(|l| limit[l] as usize + 1)
        .max()
        .unwrap_or(0);
    let mut bits = vec![0u8; nrows.div_ceil(8) * n_lanes];
    for (blk, acc) in bits.chunks_exact_mut(n_lanes.max(1)).enumerate() {
        for k in 0..(nrows - 8 * blk).min(8) {
            let row = &rowmax[(8 * blk + k) * lanes..][..n_lanes];
            for ((a, &v), &m) in acc.iter_mut().zip(row).zip(minsc) {
                *a |= u8::from(v.into() >= m) << k;
            }
        }
    }
    for l in 0..n_lanes {
        let best_score = gmax[l];
        let best_te = te[l];
        // No alignment found at all (te still -1) means there is no query end to report either.
        let best_qe = if best_te >= 0 { qe[l] } else { -1 };
        // The tracker streamed, skipping the exclusion zone around `te` where nearly every row
        // reaching `minsc` lies (`score2_lane`). The `BWA4_SUBOPT_SHAPE` probe counts the
        // tracker's candidates, so it keeps the walk below.
        if !bwa_extend::sw::subopt_shape::enabled() {
            let (score2, te2) = if walks(l) {
                crate::rescue_band::score2_lane(
                    rowmax,
                    lanes,
                    l,
                    limit[l] as usize + 1,
                    &bits,
                    n_lanes,
                    (best_score, best_te),
                    minsc[l],
                    max_sc,
                )
            } else {
                (-1, -1)
            };
            out[group_idx * lanes + l] = (best_score, best_te, best_qe, score2, te2);
            continue;
        }
        // score2: feed this lane's row maxima to ksw's `b` tracker, then take the best candidate
        // outside the exclusion window around `te`. The tracker is shared with the scalar
        // `ksw_local_fwd` precisely so the merge rule and the window cannot drift between them.
        let mut b = SuboptimalTracker::new();
        // `best_score` is the maximum of every row's maximum, so when it is below `minsc` no row can
        // clear `minsc` either, the candidate list is necessarily empty, and `finish` returns
        // `(-1, -1)` on its `self.b.is_empty()` arm. The whole walk is then provably pointless.
        //
        // It is not a rare case, it is the common one: `BWA4_SUBOPT_SHAPE` on 1M chr21 wgsim pairs
        // counts 4,388,797 jobs pushing 1,694,171,812 rows, and 2,212,629 of those jobs, 50.4%,
        // keep ZERO candidates. Half the rows pushed in a run are pushed only to be discarded one
        // at a time, at about 2.9 ns each.
        //
        // Provable rather than measured: the test is on the same `minsc` the pushes use, and the
        // skipped work has no effect other than filling `b`.
        if walks(l) {
            let last = limit[l] as usize;
            for blk in 0..=last / 8 {
                let mut m = bits[blk * n_lanes + l];
                while m != 0 {
                    let i = 8 * blk + m.trailing_zeros() as usize;
                    m &= m - 1;
                    // Best H anywhere in target row `i` for this lane. Rows past `limit[l]` were
                    // never processed by this lane, so they are not offered as candidates.
                    if i > last {
                        break;
                    }
                    b.push_row(i as i32, rowmax[i * lanes + l].into(), minsc[l]);
                }
            }
        }
        b.record_shape(if limit[l] >= 0 {
            limit[l] as u64 + 1
        } else {
            0
        });
        let (score2, te2) = b.finish(best_score, best_te, max_sc);
        out[group_idx * lanes + l] = (best_score, best_te, best_qe, score2, te2);
    }
}

/// NEON i16x8 forward local-SW. Vectorizes the [`fwd_local_sw_scalar`] control flow across `LANES`
/// jobs: the inner cell recurrence runs on `int16x8` (one lane per job), the per-row bookkeeping and
/// [`extract_group`] stay scalar. With the standard 5x5 `mat` a cell score is `match`/`mismatch`/
/// `-1(N)` chosen by compares; with `MAT5` (a `--meth` matrix, `mat5_u8_ok`) it is `mat[5t + q]`,
/// one table lookup on the byte codes. Every value must fit i16 (caller-guarded).
///
/// # Parameters
/// As [`fwd_local_sw_batch`]. Two extra preconditions the caller must have checked, because nothing
/// here re-checks them: `mat_is_standard(m, mat)` must hold, and every job's score ceiling
/// `min(qlen, tlen) * max_sc` must be under [`I16_SCORE_LIMIT`] with a target under that many bases.
///
/// # Returns
/// As [`fwd_local_sw_batch`], and byte-identical to [`fwd_local_sw_scalar`] on the same input.
///
/// # Safety
/// Caller must have confirmed NEON is available. All the loads and stores below use unchecked
/// pointer offsets whose bounds come from `qmax`/`tmax`, which are computed from the same buffers.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_neon<const MAT5: bool>(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::aarch64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // The whole standard matrix collapses to two numbers plus the fixed -1 for N: `mtch` is the
    // positive match bonus, `mis` the *signed* mismatch score (already negative, e.g. -4).
    let mtch = mat[0] as i16;
    let mis = mat[1] as i16;
    // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    // Broadcast constants. Each is the same scalar in all 8 lanes, hoisted out of both DP loops:
    // scores (`mtch_v`/`mis_v`/`n_v`/`dead_v`), gap costs (`e_*_v`/`oe_*_v`), and the three code
    // values the cell-score selects compare against (`four_v` = N, `m_v` = 5 = first dead code,
    // `zpad_v` = 5 = profile padding). `m_v` and `zpad_v` hold the same number but are compared
    // differently: `>= m_v` on the target versus `== zpad_v` / `> zpad_v` on the query.
    let zero = vdupq_n_s16(0);
    let one_v = vdupq_n_s16(1);
    let mtch_v = vdupq_n_s16(mtch);
    let mis_v = vdupq_n_s16(mis);
    let n_v = vdupq_n_s16(-1);
    let dead_v = vdupq_n_s16(DEAD_CELL_SCORE as i16);
    let four_v = vdupq_n_s16(4);
    let m_v = vdupq_n_s16(m as i16);
    let zpad_v = vdupq_n_s16(ZPAD as i16);
    let e_del_v = vdupq_n_s16(e_del as i16);
    let oe_del_v = vdupq_n_s16(oe_del as i16);
    let e_ins_v = vdupq_n_s16(e_ins as i16);
    let oe_ins_v = vdupq_n_s16(oe_ins as i16);
    // `MAT5`: the 25 cells, then zeros; the index `5t + q` of a real pair is under 25, N included.
    let mut tab = [0i8; 32];
    tab[..25].copy_from_slice(&mat[..25]);
    let tbl = int8x16x2_t(vld1q_s8(tab.as_ptr()), vld1q_s8(tab.as_ptr().add(16)));
    let five = vdup_n_u8(5);

    // Group setup is identical to `fwd_local_sw_scalar`; see there for what each variable holds.
    for (group_idx, group) in jobs.chunks(LANES).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // SoA sequences (u8, padded) + per-lane bounds/params.
        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES],
            [0usize; LANES],
            [i32::MAX; LANES],
            [i32::MAX; LANES],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = b;
            }
        }

        // i16 SoA DP state. Same meanings as the scalar path (`h_prev` = H(i-1, .), `h_cur` = H(i, .),
        // `e` = the per-column E carry), narrowed from i32 to i16 so one `vld1q_s16` fetches all 8
        // lanes of a column. `rowmax` stays i32: it is only read by scalar bookkeeping.
        let mut h_prev = vec![0i16; qmax * LANES];
        let mut h_cur = vec![0i16; qmax * LANES];
        let mut e = vec![0i16; qmax * LANES];
        let mut rowmax = vec![0i32; tmax * LANES];
        let mut gmax = [0i32; LANES];
        let mut te = [-1i32; LANES];
        let mut qe = [0i32; LANES];
        let mut limit = [-1i32; LANES];
        let mut frozen = [false; LANES];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // Widen 8 u8 codes at `off` into an int16x8 (lanes = jobs). The sequences are stored as u8
        // (they are 0..=5 plus the 255 sentinel) but the DP is i16, so each load is a 64-bit read
        // plus `vmovl_u8` (unsigned widen, so 255 becomes 255 and not -1) and a reinterpret, which
        // is a no-op at runtime and only changes the type for the signed compares below.
        // `buf` is `seq_q` or `seq_t`; `off` is the flat SoA offset of the column or row, i.e.
        // `index * LANES`. Returns one lane per job, lane `l` holding that job's base code.
        // Caller must keep `off + 8 <= buf.len()`; both call sites derive `off` from `qmax`/`tmax`.
        let load_codes = |buf: &[u8], off: usize| -> int16x8_t {
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(buf.as_ptr().add(off))))
        };
        // `MAT5`'s cell score at column `j` from the row's `5t`. ZPAD and PAD index garbage (in the
        // table or out of it, which reads 0); the tail's selects override both.
        let mat5_sc = |t5: uint8x8_t, j: usize| -> int16x8_t {
            vmovl_s8(vqtbl2_s8(
                tbl,
                vadd_u8(t5, vld1_u8(seq_q.as_ptr().add(j * LANES))),
            ))
        };

        // =====================================================================================
        // Main DP, one target row per iteration. Reminder: H = best score ending at this cell,
        // E = with a deletion (gap in query) open, F = with an insertion (gap in target) open.
        // Same structure as `fwd_local_sw_scalar`; only the per-cell arithmetic is vectorized.
        // =====================================================================================
        for i in 0..tmax {
            // The target base is fixed for the whole row, so this load is hoisted out of `j`.
            // Lane `l` = target base at row `i` of job `l`.
            let t_v = load_codes(&seq_t, i * LANES);
            // Row accumulators, one lane per job, with the same invariant as the scalar version at
            // the top of column `j`: lane `l` of `f_v` is F(i, j), of `h_diag_v` is H(i-1, j-1), of
            // `imax_v` is max H(i, 0..j), and of `imax_col_v` is the smallest column attaining it.
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
                                       // Carried column index and the row-invariant "target is N" mask, hoisted for the same
                                       // reasons as in `fwd_local_sw_neon_u8`.
            let mut j_v = zero;
            let t_is_n = vceqq_s16(t_v, four_v);
            let t5 = vmul_u8(vld1_u8(seq_t.as_ptr().add(i * LANES)), five);

            // Padding-free column range; see `fwd_local_sw_neon_u8` for why dropping the ZPAD and
            // PAD blends below `n_fast` is exact (no live lane shows either code there, and a dead
            // target row's cells stay inside their own lane, where nothing reads them).
            for j in 0..n_fast {
                let sc = if MAT5 {
                    mat5_sc(t5, j)
                } else {
                    let q_v = load_codes(&seq_q, j * LANES);
                    let eq = vceqq_s16(t_v, q_v);
                    let n_mask = vorrq_u16(t_is_n, vceqq_s16(q_v, four_v));
                    vbslq_s16(n_mask, n_v, vbslq_s16(eq, mtch_v, mis_v))
                };
                let e_v = vld1q_s16(e.as_ptr().add(j * LANES));
                let mut h_v = vaddq_s16(h_diag_v, sc);
                h_v = vmaxq_s16(h_v, zero);
                h_v = vmaxq_s16(h_v, e_v);
                let mfe = h_v;
                h_v = vmaxq_s16(h_v, f_v);
                let is_new_row_max = vcgtq_s16(h_v, imax_v);
                imax_col_v = vbslq_s16(is_new_row_max, j_v, imax_col_v);
                imax_v = vmaxq_s16(imax_v, h_v);
                vst1q_s16(h_cur.as_mut_ptr().add(j * LANES), h_v);
                let e_new = vmaxq_s16(vsubq_s16(e_v, e_del_v), vsubq_s16(h_v, oe_del_v));
                vst1q_s16(e.as_mut_ptr().add(j * LANES), vmaxq_s16(e_new, zero));
                let f_new = vmaxq_s16(vsubq_s16(f_v, e_ins_v), vsubq_s16(mfe, oe_ins_v));
                f_v = vmaxq_s16(f_new, zero);
                h_diag_v = vld1q_s16(h_prev.as_ptr().add(j * LANES));
                j_v = vaddq_s16(j_v, one_v);
            }

            // Tail: the columns where ZPAD / PAD can appear, full logic.
            for j in n_fast..qmax {
                // Lane `l` = query base at column `j` of job `l`.
                let q_v = load_codes(&seq_q, j * LANES);
                // Cell score: match/mismatch, then N override (-1), then padding override (very neg).
                // Four masks (all-ones / all-zero per lane), then four selects applied in increasing
                // order of priority. The order IS the semantics: profile padding must beat the
                // equality test (ZPAD equals nothing real, but PAD-vs-PAD would compare equal), and
                // dead padding must beat everything.
                // Each mask is all-ones in the lanes where it applies, all-zero elsewhere:
                //   eq        the two bases are the same code (a match)
                //   n_mask    either base is N (code 4)
                //   zpad_mask the query position is ksw profile padding (code 5)
                //   pad_mask  the cell is past a real position: dead target row, or query beyond
                //             the padded profile
                let eq = vceqq_s16(t_v, q_v);
                let n_mask = vorrq_u16(t_is_n, vceqq_s16(q_v, four_v));
                let zpad_mask = vceqq_s16(q_v, zpad_v);
                let pad_mask = vorrq_u16(vcgeq_s16(t_v, m_v), vcgtq_s16(q_v, zpad_v));
                // `sc` lane `l` = S(target[i], query[j]) for job `l`, built by four selects in
                // increasing priority; only the last write to a lane survives.
                let mut sc = if MAT5 {
                    mat5_sc(t5, j)
                } else {
                    // Either base is N: fixed -1 in bwa's matrix, not the mismatch penalty.
                    vbslq_s16(n_mask, n_v, vbslq_s16(eq, mtch_v, mis_v))
                };
                sc = vbslq_s16(zpad_mask, zero, sc); // ksw profile padding scores 0
                sc = vbslq_s16(pad_mask, dead_v, sc); // dead padding: kill the cell outright

                // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                let e_v = vld1q_s16(e.as_ptr().add(j * LANES));
                // H = max(0, H_diag + S, E, F). Plain (wrapping) adds are safe because the caller
                // proved every job's ceiling is under `I16_SCORE_LIMIT` and the kill score is only
                // `DEAD_CELL_SCORE`, so even the killed cells stay well inside i16.
                // Lane `l` becomes H(i, j) for job `l` after the three maxes below.
                let mut h_v = vaddq_s16(h_diag_v, sc);
                h_v = vmaxq_s16(h_v, zero);
                h_v = vmaxq_s16(h_v, e_v);
                // `mfe = max(0, H_diag + S, E)` is the part of H that does not depend on the serial F
                // carry; the F recurrence below reads it instead of the full `h_v` so the critical
                // column chain drops from f -> h -> f (3 ops) to f -> f (2 ops). See the u8 kernel.
                let mfe = h_v;
                h_v = vmaxq_s16(h_v, f_v);
                // Track the min column reaching a new row max (strict >, so ties keep the earlier j).
                // `is_new_row_max` is all-ones in the lanes whose job just beat its own row best.
                let is_new_row_max = vcgtq_s16(h_v, imax_v);
                imax_col_v = vbslq_s16(is_new_row_max, j_v, imax_col_v);
                imax_v = vmaxq_s16(imax_v, h_v);
                vst1q_s16(h_cur.as_mut_ptr().add(j * LANES), h_v);

                // E(i+1,j) = max(0, E - e_del, H - oe_del), stored back into the column carry.
                // Opening from H (not from the diagonal term) is `ksw.cpp:167-169`.
                // Lane `l` = E(i+1, j) before the floor at 0 is applied on the store.
                let e_new = vmaxq_s16(vsubq_s16(e_v, e_del_v), vsubq_s16(h_v, oe_del_v));
                vst1q_s16(e.as_mut_ptr().add(j * LANES), vmaxq_s16(e_new, zero));
                // F(i,j+1) = max(0, F - e_ins, H - oe_ins), kept in a register: this is the loop's
                // serial dependency, one sub + one max per column (`ksw.cpp:172-174`).
                // Lane `l` = F(i, j+1) before the floor at 0; `f_v` then carries it to column j+1.
                let f_new = vmaxq_s16(vsubq_s16(f_v, e_ins_v), vsubq_s16(mfe, oe_ins_v));
                f_v = vmaxq_s16(f_new, zero);
                // Preload the next column's diagonal from the previous row, mirroring `ksw.cpp:176`.
                h_diag_v = vld1q_s16(h_prev.as_ptr().add(j * LANES));
                j_v = vaddq_s16(j_v, one_v);
            }

            // Per-row bookkeeping (scalar per lane). Spill the two row accumulators to memory so the
            // lanes can be inspected individually; `imax_arr[l]` is job `l`'s max H in row `i` and
            // `col_arr[l]` the column where it occurred.
            let mut imax_arr = [0i16; LANES];
            let mut col_arr = [0i16; LANES];
            vst1q_s16(imax_arr.as_mut_ptr(), imax_v);
            vst1q_s16(col_arr.as_mut_ptr(), imax_col_v);
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                // Job `l`'s best H anywhere in target row `i`, widened out of the lane.
                let row_max = imax_arr[l] as i32;
                rowmax[i * LANES + l] = row_max;
                if row_max > gmax[l] {
                    gmax[l] = row_max;
                    te[l] = i as i32;
                    qe[l] = col_arr[l] as i32;
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once no lane can still advance. This is only a speed win, never a
            // correctness one: a frozen lane's later rows are discarded by `limit` anyway, so
            // stopping early cannot change any output. It matters because mate-rescue windows are
            // long and the reverse pass usually trips `endsc` in the first handful of rows.
            if (0..n_lanes).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n_lanes, group_idx, LANES, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// Lanes for the u8 kernel: one NEON `uint8x16`, twice the i16 width. Fixed by the register width,
/// not tunable: every load, store and array below is sized `* LANES16`, and it is also the `lanes`
/// stride [`extract_group`] must be told about.
///
/// Gated on aarch64 for the same reason as [`I16_SCORE_LIMIT`]: only the NEON kernels read it.
#[cfg(target_arch = "aarch64")]
const LANES16: usize = 16;

/// NEON u8x16 forward local-SW: same control flow as [`fwd_local_sw_neon`] but 16 lanes. Local
/// alignment keeps every H/E/F non-negative, so **saturating** u8 arithmetic (`vqadd`/`vqsub`)
/// realizes `max(0, .)` directly with no bias/shift: the caller guarantees each job's score ceiling
/// `min(len)*match` fits u8. Positions (`te`/`qe`/`rowmax`) stay scalar i32, so window length is
/// unconstrained. Byte-identical to the scalar path (validated by `matesw_equals_scalar`).
///
/// This is the kernel bwa itself uses for mate rescue on stock settings: `mem_matesw` requests
/// `KSW_XBYTE` whenever `read_len * a < 250` (`bwamem_pair.cpp:208`), which a 150 bp read at `a = 1`
/// easily satisfies. It is also the one that matters for throughput, being twice as wide as the i16
/// path, and mate rescue is a large fraction of paired-end runtime.
///
/// The unsigned/saturating choice is not a micro-optimization but the thing that makes u8 viable.
/// Local SW keeps every H, E and F in `[0, ceiling]`, and `vqsubq_u8` clamps at 0 rather than
/// wrapping, so each of the three `max(0, .)` clamps in the recurrence comes for free inside the
/// arithmetic instruction. bwa exploits exactly this (`ksw.cpp:159-174` is all `adds_epu8`/`subs_epu8`),
/// with one difference: bwa carries a `q->shift` bias so it can store negative profile scores in an
/// unsigned profile, while here the mismatch is applied as a saturating *subtract* instead, so no
/// bias is needed and no unshifting is required at the end.
///
/// # Parameters
/// As [`fwd_local_sw_batch`]. Preconditions the caller must have checked and this function does not:
/// `mat_is_standard(m, mat)`, every job's score ceiling `min(qlen, tlen) * max_sc` under
/// [`U8_SCORE_LIMIT`], and every query shorter than [`U8_SCORE_LIMIT`] bases (the argmax column
/// shares the u8 lane with the scores). Target length is unconstrained.
///
/// # Returns
/// As [`fwd_local_sw_batch`], and byte-identical to [`fwd_local_sw_scalar`] on the same input.
///
/// # Safety
/// Caller must have confirmed NEON is available. Loads and stores use unchecked pointer offsets
/// bounded by `qmax`/`tmax`, which are derived from the same buffers.
///
/// # Monomorphisation
/// Two independent micro-rewrites of the column body (issue #45) are compiled in as const generic
/// parameters rather than run-time branches, and selected once per call by [`fwd_local_sw_neon_u8`]:
///
/// - `USQADD`: apply the substitution score with one `USQADD` on a SIGNED table instead of the
///   biased `UQADD` + de-biasing `UQSUB` pair. Saves one op per cell and halves the diagonal
///   dependency chain (6 cycles to 3).
/// - `SHARE_OE`: compute `vqsubq_u8(h, oe)` once and feed both E and F, legal when
///   `oe_del == oe_ins`. Saves one more op per cell.
///
/// Both are byte-identical; see the notes at their use sites for the proofs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_neon_u8_impl<const USQADD: bool, const SHARE_OE: bool, const MAT5: bool>(
    jobs: &[FwdJob],
    // Matrix dimension (5). No longer read now that dead-cell detection keys on the PAD byte's high
    // bit rather than `>= m`; kept in the signature for parity with the i16 and AVX2 kernels.
    _m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::aarch64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as u8; // match bonus (>= 0)
    let mispen = (-mat[1]) as u8; // mismatch penalty b (mat[1] = -b)
                                  // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    // Broadcast constants, one scalar replicated across all 16 lanes. Note the scores are stored as
    // *magnitudes* here, not signed values, which is what lets the whole kernel stay unsigned.
    let zero = vdupq_n_u8(0);
    let zpad_v = vdupq_n_u8(ZPAD);
    let e_del_v = vdupq_n_u8(e_del as u8);
    let oe_del_v = vdupq_n_u8(oe_del as u8);
    let e_ins_v = vdupq_n_u8(e_ins as u8);
    let oe_ins_v = vdupq_n_u8(oe_ins as u8);

    // Substitution-score table, indexed by `target XOR query` and biased by `mispen` so every entry
    // is a non-negative magnitude: one saturating add then one de-biasing saturating subtract
    // reproduce `max(0, h_diag + S)` exactly, replacing the split form's three candidate adds and
    // four selects. Real base codes are 0-3, so a real-vs-real XOR is 0 (match, +a) or 1-3
    // (mismatch, -b); a query N (code 4) against a real base gives 4-7, scoring bwa's -1.
    //
    // TARGET N IS RE-ENCODED AS [`N_TARGET`] (12) WHEN `seq_t` IS FILLED, and that is what lets the
    // loop have no N repair at all. With target N left at 4 the both-N cell reads XOR 0, which is
    // the match slot, so every row needed a `t == 4` blend to put bwa's -1 back: four vector
    // operations per column of a quad, about 6% of the kernel. Re-encoded, the reachable indices are
    //
    //   t 0-3 vs q 0-3  -> 0-3    match / mismatch
    //   t 0-3 vs q 4    -> 4-7    one N
    //   t 12  vs q 0-3  -> 12-15  one N
    //   t 12  vs q 4    -> 8      both N, previously the wrong slot, now its own
    //   t 12  vs q 5    -> 9      ZPAD, overridden by the zpad blend, so a don't-care
    //   anything vs PAD -> >= 16  `vqtbl1q` returns 0 there, and the pad blend overrides anyway
    //
    // so slots 8 and 12-15 carry the N score alongside 4-7, and 10-11 are unreachable. The trick is
    // available because the table has 16 slots and only six were in use.
    //
    // The bias is `mispen`, the smallest value making the mismatch entry (bias - b) non-negative.
    // Correctness needs the biased add not to saturate early: `max h_diag + bias + mtch < 256`. The
    // caller caps the u8 kernel at a score ceiling under `U8_SCORE_LIMIT` (250), and for the DNA
    // scores that select this kernel `mispen + mtch` is small (5 for bwa's a=1,b=4), so 249+5 = 254
    // holds. Asserted so a future parameter set that breaks it fails loudly rather than silently.
    //
    // A HARD assert, not a `debug_assert`: with `USQADD` the release build's correctness rests on
    // this invariant too (see the `diag!` macro), so a parameter set that broke it must fail loudly
    // rather than silently saturate. It is one test per call, outside every loop.
    assert!(
        (U8_SCORE_LIMIT as u32) + mispen as u32 + mtch as u32 <= 256,
        "u8 rescue score table would saturate: mispen {mispen} + mtch {mtch} + ceiling too large"
    );
    // Under `USQADD` the table holds the SIGNED deltas themselves and there is no bias at all: the
    // single `USQADD Vd.16B, Vn.16B` is "unsigned saturating accumulate of a signed addend", which
    // is exactly `clamp(d + delta, 0, 255)` — the same value the biased pair produces, and with the
    // same clamp at 0. Bytes are stored here and reinterpreted as `int8x16_t` at the use site.
    let bias = if USQADD { 0 } else { mispen };
    let mut tbl = [0u8; 16];
    if USQADD {
        tbl = u8_score_table(mat);
    } else {
        tbl[0] = bias + mtch; // match: +a
                              // tbl[1..=3] = bias - b = 0 (mismatch), already zero
        for e in tbl.iter_mut().take(8).skip(4) {
            *e = bias - 1; // query N (code 4) against a real base: bwa scores -1
        }
        tbl[8] = bias - 1; // both N: N_TARGET XOR 4
        for e in tbl.iter_mut().take(16).skip(12) {
            *e = bias - 1; // target N against a real base: N_TARGET XOR 0-3
        }
    }
    let score_tbl = vld1q_u8(tbl.as_ptr());
    // `MAT5` (`--meth`'s matrices, whose freed C/T or G/A cell no `target XOR query` class can hold):
    // the matrix itself as a 32-entry table indexed by `5 t + q`, `seq_t` holding `5 t` (N lands on
    // 20). The index is a SATURATING add so a PAD byte (255) still reads past the table and scores
    // 0, as it does in the XOR form; a ZPAD index is overridden by the zpad blend in both forms.
    // Every cell lies in `[mat[1], mat[0]]` (`mat5_u8_ok`), so the biased entries stay >= 0.
    let mut tbl5 = [0u8; 32];
    for (e, &v) in tbl5.iter_mut().zip(&mat[..25]) {
        *e = (i32::from(bias) + i32::from(v)) as u8;
    }
    let score_tbl5 = vld1q_u8_x2(tbl5.as_ptr());
    macro_rules! score {
        ($t:expr, $q:expr) => {
            if MAT5 {
                vqtbl2q_u8(score_tbl5, vqaddq_u8($t, $q))
            } else {
                vqtbl1q_u8(score_tbl, veorq_u8($t, $q))
            }
        };
    }
    let bias_v = vdupq_n_u8(bias);
    let high_bit_v = vdupq_n_u8(0x80);

    // `max(0, d + S)` for one cell, in the form the `USQADD` const parameter selects. The two arms
    // agree on every index the output can depend on:
    //
    // - match:    biased is `qsub(qadd(d, b + a), b)`, and the assert above forbids the inner add
    //             from saturating, so it is `d + a`; `USQADD` gives `min(255, d + a)` = `d + a`.
    // - mismatch: biased is `qsub(qadd(d, 0), b)` = `max(0, d - b)`; `USQADD` is `max(0, d - b)`.
    // - either N: same, with 1 in place of `b`.
    //
    // They differ on ONE index, and only there: an out-of-range table read (target byte `PAD`, which
    // the FAST column body can see for a lane already past its `tlen`). `vqtbl1q` returns 0, so the
    // biased form yields `d - bias` and `USQADD` yields `d`. Those lanes are dead by construction:
    // `finish_row` skips `row >= tlen[l]`, `extract_group` stops at `limit[l] <= tlen[l] - 1`, and
    // every operation here is lane-local, so nothing the output reads can see the difference. Pinned
    // by `matesw_ragged_tlen_equals_scalar` rather than left to the argument.
    macro_rules! diag {
        ($d:expr, $s:expr) => {
            if USQADD {
                vsqaddq_u8($d, vreinterpretq_s8_u8($s))
            } else {
                vqsubq_u8(vqaddq_u8($d, $s), bias_v)
            }
        };
    }

    // The E and F recurrences for one cell, returned as `(E(i+1, j), F(i, j+1))`.
    //
    // Under `SHARE_OE` one `vqsubq_u8(h, oe)` feeds both, five ops where there were six. Exact, not
    // merely equal in score: unsigned saturating subtract is monotone and its clamp at 0 preserves
    // order, so `qsub(max(a, b), c) == max(qsub(a, c), qsub(b, c))`. With `h = max(mfe, f)` that
    // makes `qsub(h, oe_ins)` equal to `max(qsub(mfe, oe_ins), qsub(f, oe_ins))`, and the second
    // term is absorbed by the `qsub(f, e_ins)` already in the max because `oe_ins >= e_ins`. What is
    // left is precisely the split form's F. Requires `oe_del == oe_ins`, checked by the caller.
    //
    // And the subtract is taken off `mfe`, not `h` (bwa-mem3's FScan cell): F is unchanged by the
    // same absorption, while E loses only `qsub(f, oe)`, a vertical gap opened straight out of a
    // horizontal one. That path's twin (vertical gap first, then the horizontal one, between the
    // same two cells) costs the same and is still admitted, since F opens from `mfe`, which holds
    // E; so every H is unchanged, and H is all the kernel reads. What it buys is latency: F's
    // carry is `qsub` + `max` per column, off H, and E no longer waits on F.
    macro_rules! ef {
        ($e_prev:expr, $f_prev:expr, $mfe:expr, $h:expr) => {
            if SHARE_OE {
                let hg = vqsubq_u8($mfe, oe_del_v);
                (
                    vmaxq_u8(vqsubq_u8($e_prev, e_del_v), hg),
                    vmaxq_u8(vqsubq_u8($f_prev, e_ins_v), hg),
                )
            } else {
                (
                    vmaxq_u8(vqsubq_u8($e_prev, e_del_v), vqsubq_u8($h, oe_del_v)),
                    vmaxq_u8(vqsubq_u8($f_prev, e_ins_v), vqsubq_u8($mfe, oe_ins_v)),
                )
            }
        };
    }

    // Group setup is identical to `fwd_local_sw_scalar` at 16 lanes; see there for each variable.
    for (group_idx, group) in jobs.chunks(LANES16).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES16];
        let mut seq_t = vec![PAD; tmax * LANES16];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES16],
            [0usize; LANES16],
            [i32::MAX; LANES16],
            [i32::MAX; LANES16],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES16 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES16 + l] = ZPAD;
            }
            // The N re-encoding, applied once per target base here so the column loops never pay
            // for it. See the score-table comment above.
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = if MAT5 {
                    5 * b
                } else if b == 4 {
                    N_TARGET
                } else {
                    b
                };
            }
        }

        // Same DP state as the scalar path (H(i-1, .), H(i, .), the per-column E carry), narrowed to
        // u8 so a column of all 16 lanes is one `vld1q_u8`. Legal only because local SW keeps every
        // value in `[0, ceiling]` and the caller proved the ceiling is under 250.
        let mut h_prev = vec![0u8; qmax * LANES16];
        let mut h_cur = vec![0u8; qmax * LANES16];
        let mut e = vec![0u8; qmax * LANES16];
        // LazyQE (bwa-mem3): the column of a row's max is recovered after the row, not tracked per
        // cell. `hq` holds the rows a quad or pair otherwise keeps only in registers (its last row
        // is `h_cur`), `ck[r]` row `r`-of-the-iteration's running max at the end of each 16-column
        // block. Each is ONE buffer, rows interleaved per column (per block), so the quad body
        // addresses all of them off one register: separate buffers spilled the loop's constants.
        let nblk = qmax.div_ceil(16);
        let mut hq = vec![0u8; qmax * 3 * LANES16];
        let mut ck = vec![0u8; nblk * 4 * LANES16];
        let (hq_p, ck_p) = (hq.as_mut_ptr(), ck.as_mut_ptr());
        // Row maxima kept in the LANE's own width, not widened to i32. Two reasons, both measured:
        // the whole row can then be published with ONE `vst1q_u8` instead of 16 scalar stores inside
        // the per-lane epilogue, and the buffer shrinks 4x (22 KB instead of 89 KB for a 1400-row
        // group), which matters because it is walked again by `extract_group`.
        let mut rowmax = vec![0u8; tmax * LANES16];
        let mut gmax = [0i32; LANES16];
        let mut te = [-1i32; LANES16];
        let mut qe = [0i32; LANES16];
        let mut limit = [-1i32; LANES16];
        let mut frozen = [false; LANES16];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // `finish_row`'s state as lanes (bwa-mem3's row epilogue): the global max and its column
        // in bytes, its row in two u16 halves (-1 = none yet), the freeze mask, and the per-lane
        // bounds. A lane past the group has `tlen` 0 and is never live. `endsc` clamped to a byte
        // freezes the same lanes: an improved `gmax` is at least 1 and at most 254 (`u8_job_fits`).
        assert!(
            tmax < usize::from(u16::MAX),
            "u8 rescue window too long for u16 rows"
        );
        //
        // Kept in memory, `ep[EP_*]`, and moved in and out once per row by volatile accesses so
        // LLVM cannot promote it: held in registers across the column loops, these eight vectors
        // pushed the quad body's own constants out to the stack (5 reloads a column).
        const EP_GMAX: usize = 0;
        const EP_QE: usize = 1;
        const EP_FROZEN: usize = 2;
        const EP_TE: usize = 3; // and 4: the u16 row, low then high half
        const EP_TLEN: usize = 5; // and 6: likewise
        const EP_ENDSC: usize = 7;
        let mut ep = [zero; 8];
        {
            let t16: [u16; LANES16] = std::array::from_fn(|l| tlen[l] as u16);
            ep[EP_TLEN] = vreinterpretq_u8_u16(vld1q_u16(t16.as_ptr()));
            ep[EP_TLEN + 1] = vreinterpretq_u8_u16(vld1q_u16(t16.as_ptr().add(8)));
            let e8: [u8; LANES16] = std::array::from_fn(|l| endsc[l].clamp(0, 255) as u8);
            ep[EP_ENDSC] = vld1q_u8(e8.as_ptr());
            ep[EP_TE] = vdupq_n_u8(0xff);
            ep[EP_TE + 1] = vdupq_n_u8(0xff);
        }
        let ep_p = ep.as_mut_ptr();
        macro_rules! ep_ld {
            ($k:expr) => {
                std::ptr::read_volatile(ep_p.add($k))
            };
        }
        macro_rules! ep_st {
            ($k:expr, $v:expr) => {
                std::ptr::write_volatile(ep_p.add($k), $v)
            };
        }
        // Columns below the shortest live query: no lane can be showing ZPAD or PAD there, which is
        // what the fast body in the row loop relies on. Mate-rescue batches are whole reads of one
        // run, so in practice this covers ~92% of the padded columns (148 real of 160 padded).
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };
        // First column from which EVERY live lane holds ZPAD, or `qmax` if no such column exists.
        //
        // The columns `[n_pad, qmax)` are the ksw profile padding, and `ksw_padded_qlen` rounds each
        // query up to a whole vector, so there are 12 of them on the measured shape (148 real of 160
        // padded). They are NOT dead: a ZPAD column scores 0 and therefore carries the diagonal, so
        // its H can be the row maximum and set `qe` and `score2`. But when every live lane is showing
        // ZPAD, `zpad_mask` is all-ones and the tail body's whole score computation folds away to
        // `diag = d`: no query load, no `EOR`, no `TBL`, no substitution add, and none of the four
        // pad blends. That is the third column regime run below (issue #47).
        //
        // The range is `[max qlen, min padded_qlen)`, and it is non-empty exactly when every live
        // lane pads to the same `qmax`, which is the ordinary case of a batch of whole reads from one
        // run. When it is empty this reduces to today's two regimes with `n_pad == qmax`.
        let n_pad = if zpadcol_enabled()
            && n_lanes > 0
            && group
                .iter()
                .all(|j| ksw_padded_qlen(j.query.len(), max_sc) == qmax)
        {
            qlen[..n_lanes]
                .iter()
                .copied()
                .max()
                .unwrap_or(qmax)
                .max(n_fast)
                .min(qmax)
        } else {
            qmax
        };

        // =====================================================================================
        // Main DP. Two target rows per iteration when there are two left, one otherwise.
        //
        // WHY PAIRS. In the one-row form every cell pays five memory operations: load the query
        // column, load `e[j]`, load `h_prev[j]`, store `e[j]`, store `h_cur[j]`. Rows `i` and `i+1`
        // share all of them. Row `i+1`'s diagonal is `H(i, j-1)`, which row `i` just produced in a
        // register; row `i+1`'s E carry is `E(i+1, j)`, which row `i` computes and would otherwise
        // store only for row `i+1` to load back; and both rows read the SAME query column. So a
        // pair costs five memory ops for two cells instead of ten, and the two H chains are
        // independent, which gives the out-of-order engine something to interleave.
        //
        // Byte-identical: the arithmetic per cell is untouched and the row epilogues still run in
        // row order, so freezing at row `i` still suppresses row `i+1` exactly as before. `h_cur`
        // holds `H(i+1, .)`, which is what row `i+2` needs after the swap; `H(i, .)` never reaches
        // memory because nothing outside the pair reads it.
        // =====================================================================================
        // Per-row bookkeeping, identical in both paths: publish this row's max for `score2`, and
        // update the lane's global max / end coordinates / freeze state.
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $h:expr, $hs:expr, $r:expr) => {{
                let row = $row;
                let imax = $imax;
                // Publish the row in one store. Lanes that are out of target or already frozen get
                // a value written that the old code left untouched, and that is safe: a lane's row
                // maxima are read only for rows `0..=limit[l]`, and `limit[l]` is `tlen[l] - 1` or
                // the freeze row, so every slot this writes beyond the guard is one nothing reads.
                vst1q_u8(rowmax.as_mut_ptr().add(row * LANES16), imax);
                // A lane is live while `row < tlen` and not frozen; a live lane whose row max beats
                // its `gmax` takes this row's max, row and column. Branch-free: the scalar form's
                // per-lane `row_max > gmax` test was a data-dependent branch, ~10% of this kernel.
                let rv = vdupq_n_u16(row as u16);
                let inside = vcombine_u8(
                    vmovn_u16(vcltq_u16(rv, vreinterpretq_u16_u8(ep_ld!(EP_TLEN)))),
                    vmovn_u16(vcltq_u16(rv, vreinterpretq_u16_u8(ep_ld!(EP_TLEN + 1)))),
                );
                let frozen_v = ep_ld!(EP_FROZEN);
                let gmax_old = ep_ld!(EP_GMAX);
                let imp = vandq_u8(vcgtq_u8(imax, gmax_old), vbicq_u8(inside, frozen_v));
                let gmax_v = vbslq_u8(imp, imax, gmax_old);
                ep_st!(EP_GMAX, gmax_v);
                // The column, for improved lanes only (~44% of rows have one): the first column
                // holding the row max is in the first block whose running max reached it, so
                // find that block per lane from the checkpoints (the last block has none and is
                // reached by every lane still pending), then scan it backwards so the smallest
                // column wins. Same column as the per-cell strict `>` argmax it replaces.
                if vmaxvq_u8(imp) != 0 {
                    let (h, hs): (*const u8, usize) = ($h, $hs);
                    let (mut pending, mut col) = (imp, zero);
                    for blk in 0..nblk {
                        let reached = if blk + 1 < nblk {
                            vceqq_u8(vld1q_u8(ck_p.add((blk * 4 + $r) * LANES16)), imax)
                        } else {
                            vdupq_n_u8(0xff)
                        };
                        let here = vandq_u8(reached, pending);
                        if vmaxvq_u8(here) == 0 {
                            continue;
                        }
                        for j in (blk * 16..qmax.min(blk * 16 + 16)).rev() {
                            let eq =
                                vandq_u8(vceqq_u8(vld1q_u8(h.add(j * hs * LANES16)), imax), here);
                            col = vbslq_u8(eq, vdupq_n_u8(j as u8), col);
                        }
                        pending = vbicq_u8(pending, here);
                        if vmaxvq_u8(pending) == 0 {
                            break;
                        }
                    }
                    ep_st!(EP_QE, vbslq_u8(imp, col, ep_ld!(EP_QE)));
                }
                let rv8 = vreinterpretq_u8_u16(rv);
                ep_st!(EP_TE, vbslq_u8(vzip1q_u8(imp, imp), rv8, ep_ld!(EP_TE)));
                ep_st!(
                    EP_TE + 1,
                    vbslq_u8(vzip2q_u8(imp, imp), rv8, ep_ld!(EP_TE + 1))
                );
                // Freezing is rare (the reverse pass's stop score): lanes, then the arrays the
                // early exit and `extract_group` read.
                let newly = vandq_u8(imp, vcgeq_u8(gmax_v, ep_ld!(EP_ENDSC)));
                if vmaxvq_u8(newly) != 0 {
                    ep_st!(EP_FROZEN, vorrq_u8(frozen_v, newly));
                    let mut nw = [0u8; LANES16];
                    vst1q_u8(nw.as_mut_ptr(), newly);
                    for l in 0..n_lanes {
                        if nw[l] != 0 {
                            frozen[l] = true;
                            limit[l] = row as i32;
                        }
                    }
                }
            }};
        }

        let pair_rows = rowpair_enabled();
        let quad_rows = pair_rows && rowquad_enabled();
        let mut i = 0usize;
        while i < tmax {
            // Rows consumed by this iteration: 4 when a full quad is left, else 2, else 1. The
            // narrower bodies are the tail, and are also what `BWA4_RESCUE_ROWQUAD=0` falls back to.
            let rows = if quad_rows && i + 3 < tmax {
                4
            } else if pair_rows && i + 1 < tmax {
                2
            } else {
                1
            };
            if rows == 4 {
                let t0_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
                let t1_v = vld1q_u8(seq_t.as_ptr().add((i + 1) * LANES16));
                let t2_v = vld1q_u8(seq_t.as_ptr().add((i + 2) * LANES16));
                let t3_v = vld1q_u8(seq_t.as_ptr().add((i + 3) * LANES16));
                // Four independent sets of row accumulators. `d0` is the only diagonal that comes
                // from memory; `d1`/`d2`/`d3` are the previous column's H from the row above, handed
                // over in a register.
                let (mut f0, mut f1, mut f2, mut f3) = (zero, zero, zero, zero);
                let (mut d0, mut d1, mut d2, mut d3) = (zero, zero, zero, zero);
                let (mut imax0, mut imax1, mut imax2, mut imax3) = (zero, zero, zero, zero);

                // ---- Fast column range: no ZPAD, no PAD (see the one-row body below) ----------
                for j in 0..n_fast {
                    // The one load all four rows use.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));

                    // Row i. `e_v` is E(i, j), the only E that comes from memory.
                    let s0 = score!(t0_v, q_v);
                    let diag0 = diag!(d0, s0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    // Row i+1, whose diagonal H(i, j-1) is the previous column's `h0` and whose E
                    // carry row i just produced in a register.
                    let s1 = score!(t1_v, q_v);
                    let diag1 = diag!(d1, s1);
                    let mfe1 = vmaxq_u8(diag1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    // Row i+2.
                    let s2 = score!(t2_v, q_v);
                    let diag2 = diag!(d2, s2);
                    let mfe2 = vmaxq_u8(diag2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    // Row i+3, the only one of the four whose H and E reach memory.
                    let s3 = score!(t3_v, q_v);
                    let diag3 = diag!(d3, s3);
                    let mfe3 = vmaxq_u8(diag3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    vst1q_u8(hq_p.add((3 * j + 1) * LANES16), h1);
                    vst1q_u8(hq_p.add((3 * j + 2) * LANES16), h2);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 2) * LANES16), imax2);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 3) * LANES16), imax3);
                    }
                }

                // ---- Tail: ZPAD / PAD possible, full logic, still four rows at a time ----------
                // `q_pad` is hoisted out of the four PAD tests: the pair body recomputes
                // `vtstq_u8(vorrq_u8(t, q), high_bit)` per row, but the query half of that OR is the
                // same for all four, so it is tested once and OR-ed with each row's target test.
                for j in n_fast..n_pad {
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    let zpad_mask = vceqq_u8(q_v, zpad_v);
                    let q_pad = vtstq_u8(q_v, high_bit_v);

                    let s0 = score!(t0_v, q_v);
                    let scored0 = diag!(d0, s0);
                    let pad0 = vorrq_u8(q_pad, vtstq_u8(t0_v, high_bit_v));
                    let mut diag0 = vbslq_u8(zpad_mask, d0, scored0);
                    diag0 = vbslq_u8(pad0, zero, diag0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let s1 = score!(t1_v, q_v);
                    let scored1 = diag!(d1, s1);
                    let pad1 = vorrq_u8(q_pad, vtstq_u8(t1_v, high_bit_v));
                    let mut diag1 = vbslq_u8(zpad_mask, d1, scored1);
                    diag1 = vbslq_u8(pad1, zero, diag1);
                    let mfe1 = vmaxq_u8(diag1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    let s2 = score!(t2_v, q_v);
                    let scored2 = diag!(d2, s2);
                    let pad2 = vorrq_u8(q_pad, vtstq_u8(t2_v, high_bit_v));
                    let mut diag2 = vbslq_u8(zpad_mask, d2, scored2);
                    diag2 = vbslq_u8(pad2, zero, diag2);
                    let mfe2 = vmaxq_u8(diag2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    let s3 = score!(t3_v, q_v);
                    let scored3 = diag!(d3, s3);
                    let pad3 = vorrq_u8(q_pad, vtstq_u8(t3_v, high_bit_v));
                    let mut diag3 = vbslq_u8(zpad_mask, d3, scored3);
                    diag3 = vbslq_u8(pad3, zero, diag3);
                    let mfe3 = vmaxq_u8(diag3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    vst1q_u8(hq_p.add((3 * j + 1) * LANES16), h1);
                    vst1q_u8(hq_p.add((3 * j + 2) * LANES16), h2);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 2) * LANES16), imax2);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 3) * LANES16), imax3);
                    }
                }

                // ---- ksw profile padding: every live lane is ZPAD, so `diag = d` ---------------
                // The tail body with a provably all-ones `zpad_mask` folded in: the query column is
                // not even loaded, and `EOR`, `TBL`, the substitution add and the four pad blends
                // all disappear. What survives is the H/E/F recurrence carrying the diagonal.
                //
                // The pad blend is dropped on the same argument the fast body already makes: a lane
                // whose target is PAD here is a lane past its own `tlen`, every operation is
                // lane-local, and `finish_row` and `extract_group` both stop at that lane's `limit`.
                for j in n_pad..qmax {
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(d0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let mfe1 = vmaxq_u8(d1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    let mfe2 = vmaxq_u8(d2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    let mfe3 = vmaxq_u8(d3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    vst1q_u8(hq_p.add((3 * j + 1) * LANES16), h1);
                    vst1q_u8(hq_p.add((3 * j + 2) * LANES16), h2);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 2) * LANES16), imax2);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 3) * LANES16), imax3);
                    }
                }

                // Row order matters here: freezing at row `i` must suppress rows `i+1..=i+3`.
                finish_row!(i, imax0, hq_p, 3, 0);
                finish_row!(i + 1, imax1, hq_p.add(LANES16), 3, 1);
                finish_row!(i + 2, imax2, hq_p.add(2 * LANES16), 3, 2);
                finish_row!(i + 3, imax3, h_cur.as_ptr(), 1, 3);
            } else if rows == 2 {
                let t0_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
                let t1_v = vld1q_u8(seq_t.as_ptr().add((i + 1) * LANES16));
                // Two independent sets of row accumulators, one per row of the pair. `d0`/`d1` are
                // the diagonals: `d0` comes from `h_prev`, `d1` from the previous column's `h0`.
                let (mut f0, mut f1) = (zero, zero);
                let (mut d0, mut d1) = (zero, zero);
                let (mut imax0, mut imax1) = (zero, zero);

                // ---- Fast column range: no ZPAD, no PAD (see the one-row body below) ----------
                for j in 0..n_fast {
                    // The one load both rows use.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));

                    // Row i.
                    let s0 = score!(t0_v, q_v);
                    let diag0 = diag!(d0, s0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    // E(i+1, j). Handed straight to row i+1 in a register instead of being stored
                    // and reloaded, which is one of the two memory ops this pairing removes.
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    // Row i+1, whose diagonal H(i, j-1) is the previous column's `h0`.
                    let s1 = score!(t1_v, q_v);
                    let diag1 = diag!(d1, s1);
                    let mfe1 = vmaxq_u8(diag1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                    }
                }

                // ---- Tail: ZPAD / PAD possible, full logic, still two rows at a time -----------
                for j in n_fast..n_pad {
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    let zpad_mask = vceqq_u8(q_v, zpad_v);

                    let s0 = score!(t0_v, q_v);
                    let scored0 = diag!(d0, s0);
                    let pad0 = vtstq_u8(vorrq_u8(t0_v, q_v), high_bit_v);
                    let mut diag0 = vbslq_u8(zpad_mask, d0, scored0);
                    diag0 = vbslq_u8(pad0, zero, diag0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let s1 = score!(t1_v, q_v);
                    let scored1 = diag!(d1, s1);
                    let pad1 = vtstq_u8(vorrq_u8(t1_v, q_v), high_bit_v);
                    let mut diag1 = vbslq_u8(zpad_mask, d1, scored1);
                    diag1 = vbslq_u8(pad1, zero, diag1);
                    let mfe1 = vmaxq_u8(diag1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                    }
                }

                // ---- ksw profile padding: every live lane is ZPAD, so `diag = d` ---------------
                // See the quad body's version for why the score computation and the pad blend both
                // fold away here.
                for j in n_pad..qmax {
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(d0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let mfe1 = vmaxq_u8(d1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    vst1q_u8(hq_p.add(3 * j * LANES16), h0);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax0);
                        vst1q_u8(ck_p.add(((j >> 4) * 4 + 1) * LANES16), imax1);
                    }
                }

                // Row order matters here: freezing at row `i` must suppress row `i+1`.
                finish_row!(i, imax0, hq_p, 3, 0);
                finish_row!(i + 1, imax1, h_cur.as_ptr(), 1, 1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
                // Row accumulators, one lane per job, same invariant as the scalar version at the
                // top of column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j). The column of the max is
                // recovered by `finish_row` (LazyQE). A per-lane strided scalar scan of `h_cur` was
                // measured 18% slower; the counts that make the vector form pay are 44% of rows
                // with an improved lane and ~1.4 distinct 16-column blocks to scan per such row.
                let mut f_v = zero;
                let mut h_diag_v = zero;
                let mut imax_v = zero;
                // Hoisted out of the column loop: "is this row's target base an N", which depends
                // only on `i`. It was recomputed for all `qmax` columns of every row.

                // ---- Fast column range: no padding of any kind can appear ---------------------
                // For `j < n_fast` every live lane holds a real base (`n_fast` is the shortest live
                // query), so both padding blends are provably no-ops there and are not emitted.
                //
                // Dropping the PAD blend is safe even though `seq_t` really is `PAD` past a short
                // lane's window: those cells now carry a value instead of dying, but the value
                // lives in that lane alone (every op here is lane-local) and every reader of a
                // lane's results is already guarded by `i >= tlen[l]`. A `PAD` target against a
                // real query also indexes `score_tbl` out of range, and `vqtbl1q` returns 0 there,
                // so the cell decays rather than scoring a spurious match. The `q == PAD` case,
                // which WOULD read a false match at `xor == 0`, cannot occur below `n_fast`.
                for j in 0..n_fast {
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    let sbt = score!(t_v, q_v);
                    let diag_v = diag!(h_diag_v, sbt);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe = vmaxq_u8(diag_v, e_v);
                    let h_v = vmaxq_u8(mfe, f_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax_v);
                    }
                    let (e_new, f_next) = ef!(e_v, f_v, mfe, h_v);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new);
                    f_v = f_next;
                    h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
                }

                // ---- Tail: the columns where ZPAD / PAD can appear, full logic ----------------
                for j in n_fast..n_pad {
                    // Lane `l` = query base at column `j` of job `l`.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    // diag_v = max(0, h_diag + score). The substitution score comes from one table
                    // lookup on `target XOR query` (see `score_tbl`), biased so it applies as a
                    // single saturating add followed by a de-biasing saturating subtract. Both
                    // saturations floor the result at 0, so no explicit `max(0, .)` is needed.
                    let sbt = score!(t_v, q_v);
                    // A target N (code 4) scores -1, including the both-N cell the table read as a
                    // match (XOR 0). Only the target can be a real code XOR-ing to 0 with an N, so
                    // `t == 4` alone catches every case the table gets wrong; a query-side N
                    // already lands on a 4-7 slot and needs no fix.
                    let scored = diag!(h_diag_v, sbt);
                    // `zpad_mask`: the query column is ksw profile padding (score 0), so the cell
                    // carries the diagonal through unchanged. `pad_mask`: the cell is dead (a PAD
                    // byte, 255, the only value with bit 7 set), forced to 0.
                    let zpad_mask = vceqq_u8(q_v, zpad_v);
                    let pad_mask = vtstq_u8(vorrq_u8(t_v, q_v), high_bit_v);
                    // Lane `l` = max(0, H(i-1, j-1) + S) for job `l`, after the padding overrides.
                    let mut diag_v = vbslq_u8(zpad_mask, h_diag_v, scored);
                    diag_v = vbslq_u8(pad_mask, zero, diag_v);

                    // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                    // saturation, so the two maxes are the whole H recurrence. `mfe = max(diag, E)`
                    // is the part of H that does NOT depend on the row-serial F carry, so the F
                    // recurrence below never waits on H.
                    let mfe = vmaxq_u8(diag_v, e_v);
                    let h_v = vmaxq_u8(mfe, f_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax_v);
                    }

                    // e = max(0, e-e_del, h-oe_del); the saturating subs supply both inner clamps.
                    // F, the row's serial carry, reassociated off `mfe` instead of `h_v`.
                    // Byte-identical: `oe_ins >= e_ins` makes the `f - oe_ins` branch inside H
                    // dominated, so `max(f - e_ins, mfe - oe_ins)` equals the original. Under
                    // `SHARE_OE` the same argument runs the other way and F comes off `h_v` again,
                    // sharing E's subtract; see the `ef!` macro.
                    let (e_new, f_next) = ef!(e_v, f_v, mfe, h_v);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new);
                    f_v = f_next;
                    h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
                }

                // ---- ksw profile padding: every live lane is ZPAD, so `diag = h_diag` ----------
                // See the quad body's version for why the score computation and the pad blend both
                // fold away here.
                for j in n_pad..qmax {
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe = vmaxq_u8(h_diag_v, e_v);
                    let h_v = vmaxq_u8(mfe, f_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);
                    if j & 15 == 15 {
                        vst1q_u8(ck_p.add((j >> 4) * 4 * LANES16), imax_v);
                    }
                    let (e_new2, f_next2) = ef!(e_v, f_v, mfe, h_v);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new2);
                    f_v = f_next2;
                    h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
                }

                finish_row!(i, imax_v, h_cur.as_ptr(), 1, 0);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once every live lane is either frozen or out of target: purely a speed
            // win, as in `fwd_local_sw_neon`. See the longer note there.
            if (0..n_lanes).all(|l| frozen[l] || i + rows >= tlen[l]) {
                break;
            }
            i += rows;
        }

        {
            let (mut g, mut q, mut t) = ([0u8; LANES16], [0u8; LANES16], [0u16; LANES16]);
            vst1q_u8(g.as_mut_ptr(), ep_ld!(EP_GMAX));
            vst1q_u8(q.as_mut_ptr(), ep_ld!(EP_QE));
            vst1q_u16(t.as_mut_ptr(), vreinterpretq_u16_u8(ep_ld!(EP_TE)));
            vst1q_u16(
                t.as_mut_ptr().add(8),
                vreinterpretq_u16_u8(ep_ld!(EP_TE + 1)),
            );
            for l in 0..n_lanes {
                (gmax[l], qe[l], te[l]) = (i32::from(g[l]), i32::from(q[l]), t[l] as i16 as i32);
            }
        }
        extract_group(
            n_lanes, group_idx, LANES16, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// The x86 u8 kernels' lane-major sequences of one group: `(seq_q, seq_t)`, `seq_q[c * lanes + l]`
/// the query byte (ZPAD from its length to its padded one, PAD after) and `seq_t[r * lanes + l]`
/// the target byte through `enc`, PAD past its length and in the lanes past the group. Each lane is
/// written contiguous and the whole is transposed in 16x16 blocks (SSE2), where one strided byte
/// store per base cost ~12% of the AVX-512 kernel on Zen 5 (bwa-mem3's `x86_soa_pack`, ~2%). The
/// row counts are rounded up to 16 with PAD rows, which nothing reads.
#[cfg(target_arch = "x86_64")]
fn soa_group(
    group: &[FwdJob],
    lanes: usize,
    qmax: usize,
    tmax: usize,
    max_sc: i32,
    enc_t: impl Fn(u8) -> u8,
    enc_q: impl Fn(u8) -> u8,
) -> (Vec<u8>, Vec<u8>) {
    debug_assert!(lanes.is_multiple_of(16) && group.len() <= lanes);
    let (ql, tl) = (qmax.next_multiple_of(16), tmax.next_multiple_of(16));
    let (mut q, mut t) = (vec![PAD; lanes * ql], vec![PAD; lanes * tl]);
    for (l, j) in group.iter().enumerate() {
        let (qr, tr) = (&mut q[l * ql..][..ql], &mut t[l * tl..][..tl]);
        for (d, &b) in qr.iter_mut().zip(j.query) {
            *d = enc_q(b);
        }
        qr[j.query.len()..ksw_padded_qlen(j.query.len(), max_sc)].fill(ZPAD);
        for (d, &b) in tr.iter_mut().zip(j.target) {
            *d = enc_t(b);
        }
    }
    (transpose16(&q, lanes, ql), transpose16(&t, lanes, tl))
}

/// `out[r * lanes + l] = src[l * rows + r]`: `lanes` rows of `rows` bytes to `rows` rows of `lanes`,
/// both multiples of 16, one 16x16 block at a time in four rounds of SSE2 unpacks.
#[cfg(target_arch = "x86_64")]
fn transpose16(src: &[u8], lanes: usize, rows: usize) -> Vec<u8> {
    use std::arch::x86_64::*;
    assert!(lanes.is_multiple_of(16) && rows.is_multiple_of(16) && src.len() == lanes * rows);
    let mut out = vec![0u8; src.len()];
    for l0 in (0..lanes).step_by(16) {
        for r0 in (0..rows).step_by(16) {
            // SAFETY: SSE2 is x86_64's baseline; every load reads 16 bytes of lane `l0 + k`'s row
            // inside `src` (`r0 + 16 <= rows`), every store 16 bytes of row `r0 + c` inside `out`.
            unsafe {
                let v: [__m128i; 16] = std::array::from_fn(|k| {
                    _mm_loadu_si128(src.as_ptr().add((l0 + k) * rows + r0) as *const __m128i)
                });
                // Round 1: `a[i]` / `a[i + 8]` interleave lanes 2i, 2i+1, columns 0-7 / 8-15.
                let a: [__m128i; 16] = std::array::from_fn(|i| {
                    let (x, y) = (v[2 * (i % 8)], v[2 * (i % 8) + 1]);
                    if i < 8 {
                        _mm_unpacklo_epi8(x, y)
                    } else {
                        _mm_unpackhi_epi8(x, y)
                    }
                });
                // Round 2: `b[h * 8 + g * 4 + i]` = lanes 4i..4i+3, columns 8h + 4g .. + 3.
                let b: [__m128i; 16] = std::array::from_fn(|n| {
                    let (h, g, i) = (n / 8, n / 4 % 2, n % 4);
                    let (x, y) = (a[h * 8 + 2 * i], a[h * 8 + 2 * i + 1]);
                    if g == 0 {
                        _mm_unpacklo_epi16(x, y)
                    } else {
                        _mm_unpackhi_epi16(x, y)
                    }
                });
                // Round 3: `c[n * 4 + m]`, `n` = column quad, `m` = (lane half, column pair).
                let c: [__m128i; 16] = std::array::from_fn(|n| {
                    let (quad, half, pair) = (n / 4, n / 2 % 2, n % 2);
                    let (x, y) = (b[quad * 4 + 2 * half], b[quad * 4 + 2 * half + 1]);
                    if pair == 0 {
                        _mm_unpacklo_epi32(x, y)
                    } else {
                        _mm_unpackhi_epi32(x, y)
                    }
                });
                // Round 4: column `4 quad + 2 pair + odd`, lanes 0-7 then 8-15.
                for col in 0..16 {
                    let (quad, pair, odd) = (col / 4, col / 2 % 2, col % 2);
                    let (x, y) = (c[quad * 4 + pair], c[quad * 4 + 2 + pair]);
                    let o = if odd == 0 {
                        _mm_unpacklo_epi64(x, y)
                    } else {
                        _mm_unpackhi_epi64(x, y)
                    };
                    _mm_storeu_si128(
                        out.as_mut_ptr().add((r0 + col) * lanes + l0) as *mut __m128i,
                        o,
                    );
                }
            }
        }
    }
    out
}

/// The u8 kernels' signed substitution table, indexed by `target XOR query` (see the score-table
/// note in [`fwd_local_sw_neon_u8_impl`]): match `+a`, mismatch `-b`, any N `-1`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn u8_score_table(mat: &[i8]) -> [u8; 16] {
    let mut tbl = [0u8; 16];
    tbl[0] = mat[0] as u8; // match: +a
    for e in tbl.iter_mut().take(4).skip(1) {
        *e = mat[1] as u8; // mismatch: -b
    }
    for e in tbl.iter_mut().take(8).skip(4) {
        *e = 0xff; // query N (code 4) against a real base: bwa scores -1
    }
    tbl[8] = 0xff; // both N: N_TARGET XOR 4
    for e in tbl.iter_mut().take(16).skip(12) {
        *e = 0xff; // target N against a real base: N_TARGET XOR 0-3
    }
    tbl
}

/// Pick the monomorphised NEON u8 body and run it. The two const parameters of
/// [`fwd_local_sw_neon_u8_impl`] are resolved here, once per call, from two cached environment
/// toggles and one property of the scoring parameters:
///
/// - `BWA4_RESCUE_USQADD=0` restores the biased add/de-bias pair;
/// - `BWA4_RESCUE_SHAREOE=0`, or `oe_del != oe_ins`, restores the split E/F subtracts. The equality
///   holds for bwa's stock `-O 6,6 -E 1,1`, which is what selects this kernel in practice.
///
/// All four bodies are byte-identical, so the toggles change speed only. They exist so the levers
/// can be A/B'd inside ONE binary with identical instrumentation on every arm, which is the only
/// measurement this project accepts for an effect of a few percent.
///
/// # Safety
/// As [`fwd_local_sw_neon_u8_impl`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_neon_u8(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    let usqadd = usqadd_enabled();
    let share_oe =
        shareoe_enabled() && (o_del + e_del) == (o_ins + e_ins) && o_del >= 0 && o_ins >= 0;
    // A non-standard matrix (`--meth`) takes the 5x5 table form, always with `USQADD`: two bodies
    // rather than four, the toggle being an A/B aid for the standard path only.
    if !mat_is_standard(m, mat) {
        return if share_oe {
            fwd_local_sw_neon_u8_impl::<true, true, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            )
        } else {
            fwd_local_sw_neon_u8_impl::<true, false, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            )
        };
    }
    match (usqadd, share_oe) {
        (true, true) => fwd_local_sw_neon_u8_impl::<true, true, false>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (true, false) => fwd_local_sw_neon_u8_impl::<true, false, false>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (false, true) => fwd_local_sw_neon_u8_impl::<false, true, false>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (false, false) => fwd_local_sw_neon_u8_impl::<false, false, false>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
    }
}

/// Lanes for the AVX2 u8 kernel: one `__m256i`, twice the NEON u8 width. Fixed by the register, not
/// tunable: every load, store and per-lane array below is sized `* LANES32`, and it is the `lanes`
/// stride [`extract_group`] must be told about.
///
/// Widening the group from 16 to 32 jobs is observable only as extra padded work: `qmax`/`tmax` are
/// maxima over a bigger chunk, and the surplus columns/rows are [`PAD`] and therefore dead. It does
/// not move any threshold, because every threshold in this file is bwa's rather than the hardware's.
#[cfg(target_arch = "x86_64")]
const LANES32: usize = 32;

/// Columns per checkpoint of the AVX-512 u8 kernel's lazy query-end recovery (bwa-mem3's `QE_BLK`):
/// a lane's query end is searched only in the block where its row max is first reached. The AVX2
/// and SSE4.1 kernels keep the inline argmax: on EPYC 7763 the lazy form was 11% slower there.
#[cfg(target_arch = "x86_64")]
const QE_BLK: usize = 16;

/// AVX2 `u8x32` forward local-SW: [`fwd_local_sw_neon_u8`] transliterated to `__m256i`, so x86_64
/// stops falling back to [`fwd_local_sw_scalar`] (measured at 2.22x the CPU of the vector path on a
/// whole paired-end run). Scope is u8 only; an i16 batch still takes the scalar path.
///
/// The transliteration is mechanical because every operation this kernel needs is element-wise, and
/// AVX2's element-wise byte ops behave exactly like NEON's despite the 128-bit-lane split that
/// afflicts its *shuffles*. Two operations have no direct AVX2 spelling and are emulated below with
/// the same trick `crate::batched`'s AVX2 kernels use:
///
/// - **signed domain**: AVX2 has only *signed* `_mm256_cmpgt_epi8`, so H, E and F are held as
///   `H - 128` (XOR 0x80), where signed order is unsigned order. Every max, saturating subtract and
///   compare is then its `epi8` twin, saturation at -128 is the DP's clamp at 0, the diagonal adds an
///   unbiased table delta with one `adds_epi8`, and only the row max is XORed back: the fork's `Sgn`
///   u8 body (`kswv.cpp`). The strict `>` on the row argmax still keeps `qe` at the *smallest*
///   attaining column (`ksw.cpp:216-218`). Both gaps open from `mfe`, not `h` (the FScan
///   cell, see `ef!` in [`fwd_local_sw_neon_u8_impl`]), so under `SHARE` (`o_del + e_del == o_ins +
///   e_ins`) E and F share one subtract.
/// - **bit-select**: NEON `vbslq_u8(mask, a, b)` is `mask ? a : b`; `_mm256_blendv_epi8(a, b, mask)`
///   is `mask ? b : a`, i.e. the operands are the other way round.
///
/// No cross-lane reduction is needed anywhere: the row accumulators are spilled to arrays and read
/// per lane, exactly as the NEON kernel does, so `vmaxvq_u8` never appears and neither does the
/// `_mm256_extracti128_si256` dance it would require.
///
/// # Parameters
/// As [`fwd_local_sw_batch`], with the same unchecked preconditions as [`fwd_local_sw_neon_u8`]:
/// `mat_is_standard(m, mat)`, every job's score ceiling `min(qlen, tlen) * max_sc` under
/// [`U8_SCORE_LIMIT`], and every query shorter than [`U8_SCORE_LIMIT`] bases.
///
/// # Returns
/// As [`fwd_local_sw_batch`], and byte-identical to [`fwd_local_sw_scalar`] on the same input
/// (`avx2_verify::avx2_matesw_u8_matches_scalar`).
///
/// # Safety
/// Caller must have confirmed AVX2 is available. Loads and stores use unchecked pointer offsets
/// bounded by `qmax`/`tmax`, which are derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx2_u8_impl<const MAT5: bool, const SHARE: bool>(
    jobs: &[FwdJob],
    // Matrix dimension (5). No longer read now that dead-cell detection keys on the PAD
    // byte's high bit rather than `>= m`, exactly as in `fwd_local_sw_neon_u8`; kept in the
    // signature for parity with the other kernels.
    _m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    // NEON `vbslq_u8(mask, a, b)` = `mask ? a : b`. `_mm256_blendv_epi8(a, b, mask)` = `mask ? b : a`,
    // hence the swapped operands; only the top bit of each mask byte is consulted, which the
    // all-ones masks above satisfy.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bsl(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_blendv_epi8(b, a, mask)
    }

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as u8; // match bonus (>= 0)
    let mispen = (-mat[1]) as u8; // mismatch penalty b (mat[1] = -b)
                                  // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    // Broadcast constants; gap penalties are held as magnitudes (subtracted, not added).
    let zero = _mm256_setzero_si256();
    // The DP's 0 in the signed H domain (H - 128): saturation at -128 is the clamp at 0, and
    // every max, subtract and compare has its exact signed twin (the fork's `Sgn` u8 body).
    let hz = _mm256_set1_epi8(-128);
    // ISSUE #43: the XOR-indexed score table, ported from `fwd_local_sw_neon_u8`. See there for
    // why the target N is re-encoded as `N_TARGET` (12) and what each of the 16 slots means. The
    // saturation guard is the same, and it is a HARD assert because the release path depends on it.
    assert!(
        (U8_SCORE_LIMIT as u32) + mispen as u32 + mtch as u32 <= 256,
        "u8 rescue score table would saturate: mispen {mispen} + mtch {mtch} + ceiling too large"
    );
    let bias = mispen;
    let mut tbl = [0u8; 32];
    for r in 0..2 {
        let o = r * 16;
        tbl[o] = bias + mtch; // match: +a
                              // tbl[o + 1..=o + 3] = bias - b = 0 (mismatch), already zero
        for k in 4..8 {
            tbl[o + k] = bias - 1; // query N against a real base: bwa scores -1
        }
        tbl[o + 8] = bias - 1; // both N: N_TARGET XOR 4
        for k in 12..16 {
            tbl[o + k] = bias - 1; // target N against a real base
        }
    }
    // `vpshufb` is IN-LANE on AVX2, so the 16-byte table is written into both 128-bit halves; on
    // SSE4.1 there is a single lane and the loop above runs once.
    // H is held as H - 128 (signed), so the table holds the unbiased deltas.
    for v in &mut tbl {
        *v = v.wrapping_sub(bias);
    }
    let score_tbl = _mm256_loadu_si256(tbl.as_ptr() as *const __m256i);
    let bias_v = _mm256_set1_epi8(bias as i8);
    // `MAT5` (`--meth`'s matrices, `mat5_u8_ok`): one lookup at `(t + q) & 15` through
    // `mat5_add_code` (which the dispatch checked exists), `seq_t` / `seq_q` holding its codes.
    // Every cell lies in `[mat[1], mat[0]]`, a signed byte added straight to the `H - 128` domain.
    let (score_tbl, enc_t, enc_q) = if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        let lo = _mm_loadu_si128(t16.as_ptr() as *const __m128i);
        (_mm256_broadcastsi128_si256(lo), t, q)
    } else {
        (score_tbl, [0, 1, 2, 3, N_TARGET], [0, 1, 2, 3, 4])
    };
    // `vpshufb` zeroes a lane whose index has bit 7 set and otherwise reads `tbl[idx & 15]`, where
    // `vqtbl1q_u8` zeroes every index >= 16. They differ only on indices 16..127, and that range is
    // UNREACHABLE here: the only alphabet byte with bit 7 set is `PAD`, `seq_q` holds
    // {0,1,2,3,4,ZPAD=5,PAD} and `seq_t` holds {0,1,2,3,N_TARGET=12,PAD}, so with neither operand
    // PAD both are <= 12 and `t ^ q <= 15` (the maximum is `3 ^ 12`); with exactly one PAD the xor
    // has bit 7 set and both instructions give 0; with both PAD the xor is 0 and both give
    // `tbl[0]`, a cell the pad blend overrides anyway. So the NEON byte-identity argument carries
    // over unchanged.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn score_diag<const M5: bool>(
        tbl: __m256i,
        _bias_v: __m256i,
        t_v: __m256i,
        q_v: __m256i,
        d: __m256i,
    ) -> __m256i {
        // `M5`: `t + q < 32` in every cell a live lane keeps, so bit 7 is clear and `vpshufb`
        // reads slot `(t + q) & 15`. A `PAD` row is past the lane's target (`finish_row` and
        // `extract_group` stop at `tlen`) and a `ZPAD` / `PAD` column is overridden by the pad
        // blend, so what they read does not matter.
        let i = if M5 {
            _mm256_add_epi8(t_v, q_v)
        } else {
            _mm256_xor_si256(t_v, q_v)
        };
        _mm256_adds_epi8(d, _mm256_shuffle_epi8(tbl, i))
    }
    let zpad_v = _mm256_set1_epi8(ZPAD as i8);
    let e_del_v = _mm256_set1_epi8(e_del as i8);
    let oe_del_v = _mm256_set1_epi8(oe_del as i8);
    let e_ins_v = _mm256_set1_epi8(e_ins as i8);
    let oe_ins_v = _mm256_set1_epi8(oe_ins as i8);

    // Group setup is identical to `fwd_local_sw_scalar` at 32 lanes; see there for each variable.
    for (group_idx, group) in jobs.chunks(LANES32).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // The N re-encoding (or `MAT5`'s add code), applied once per base so the column loops
        // never pay for it (issues #43, #44).
        let (seq_q, seq_t) = soa_group(
            group,
            LANES32,
            qmax,
            tmax,
            max_sc,
            |b| enc_t[b as usize],
            |b| enc_q[b as usize],
        );
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES32],
            [0usize; LANES32],
            [i32::MAX; LANES32],
            [i32::MAX; LANES32],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
        }

        // Same DP state as the scalar path, narrowed to u8 so a column of all 32 lanes is one
        // `_mm256_loadu_si256`. Legal only because local SW keeps every value in `[0, ceiling]` and
        // the caller proved the ceiling is under 250.
        let mut h_prev = vec![0x80u8; qmax * LANES32];
        let mut h_cur = vec![0x80u8; qmax * LANES32];
        let mut e = vec![0x80u8; qmax * LANES32];
        // LazyQE (bwa-mem3), as in `fwd_local_sw_avx512_u8`: the column of a row's max is recovered
        // by `resolve_qe` instead of tracked per cell. `ck` holds each row's running max at the end
        // of every 16-column block, a pair's two rows interleaved, in the `H - 128` domain of `imax`.
        let nblk = qmax.div_ceil(16);
        let mut ck = vec![0x80u8; nblk * 2 * LANES32];
        let ck_p = ck.as_mut_ptr();
        // ISSUE #46C. `rowmax` is u8, not i32, and the row is published with ONE vector store
        // instead of `LANES` guarded scalar ones, exactly as `fwd_local_sw_neon_u8` already does.
        // Two reasons, both structural rather than micro:
        //
        //  - size. At AVX-512's mean shape the i32 buffer was 359 KB per group, out of L1 and into
        //    L2, and `extract_group`'s per-lane walk then touched a fresh cache line for every load,
        //    re-fetching each line once per lane. As u8 it is 90 KB and the stride is exactly one
        //    64-byte line per row of 64 lanes.
        //  - stores. The scalar form issued one guarded store per live lane per row, 32 or 64 of
        //    them; the vector form issues one.
        //
        // Writing it unconditionally is safe for the same reason it is on NEON: a lane's row maxima
        // are read only for rows `0..=limit[l]`, and `limit[l]` is `tlen[l] - 1` or the freeze row,
        // so every slot this writes beyond the old guard is one nothing reads. Values fit u8 because
        // the caller proved the score ceiling is under `U8_SCORE_LIMIT`.
        let mut rowmax = vec![0u8; tmax * LANES32];
        let mut gmax = [0i32; LANES32];
        // `te` as four vectors of eight i32 rows and `qe` as one of 32 columns, blended under the
        // row's improving lanes instead of written lane by lane.
        let mut te_v = [_mm256_set1_epi32(-1); 4];
        let mut qe_v = zero;
        let mut limit = [-1i32; LANES32];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // The row epilogue's per-lane state as vectors, so a row costs a few vector ops instead
        // of a branchy pass over 32 lanes (bwa-mem3's vector epilogue): `gmax_v` is the best row
        // max so far and `endsc_v` the freeze threshold clamped to a byte, both in the `H - 128`
        // domain (scores stay below `U8_SCORE_LIMIT`, so 255 is never reached), and `frozen_v`
        // the lanes that reached it.
        let mut gmax_v = hz;
        let mut frozen_v = zero;
        let endsc_v = {
            let mut a = [i8::MAX; LANES32];
            for l in 0..n_lanes {
                a[l] = (endsc[l].clamp(0, 255) - 128) as i8;
            }
            _mm256_loadu_si256(a.as_ptr() as *const __m256i)
        };
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };
        // From here every live lane holds ZPAD (the ksw profile padding): `diag = d`, no score and
        // no pad blend. Same third regime, and the same argument, as `fwd_local_sw_neon_u8`'s.
        let n_pad = if zpadcol_enabled()
            && group
                .iter()
                .all(|j| ksw_padded_qlen(j.query.len(), max_sc) == qmax)
        {
            qlen[..n_lanes]
                .iter()
                .copied()
                .max()
                .unwrap_or(qmax)
                .max(n_fast)
                .min(qmax)
        } else {
            qmax
        };

        // =====================================================================================
        // Main DP, two target rows per iteration when two are left. Same transformation, and the
        // same byte-identity argument, as `fwd_local_sw_neon_u8`: rows `i` and `i+1` share the
        // query-column load, `h_prev[j]`, `e[j]` and the `h_cur[j]` store, because row `i+1`'s
        // diagonal is row `i`'s previous-column H and its E carry is what row `i` just produced.
        // =====================================================================================
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $t:expr) => {{
                let row = $row;
                let imax_u = _mm256_xor_si256($imax, hz);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm256_storeu_si256(
                    rowmax.as_mut_ptr().add(row * LANES32) as *mut __m256i,
                    imax_u,
                );
                // The lanes this row improves: in their target (the row's base is not `PAD`,
                // which also leaves out the lanes past `n_lanes`), not frozen, strictly above.
                let need_v = _mm256_andnot_si256(
                    _mm256_or_si256(frozen_v, $t),
                    _mm256_cmpgt_epi8($imax, gmax_v),
                );
                let need = _mm256_movemask_epi8(need_v) as u32;
                if need != 0 {
                    gmax_v = bsl(need_v, $imax, gmax_v);
                    let froze_v = _mm256_andnot_si256(_mm256_cmpgt_epi8(endsc_v, gmax_v), need_v);
                    frozen_v = _mm256_or_si256(frozen_v, froze_v);
                    // Each 8-lane quarter of the byte mask, sign-extended to i32 (bit 7 is the
                    // mask, and the extension carries it into every byte of the word).
                    let row_v = _mm256_set1_epi32(row as i32);
                    for (k, t) in te_v.iter_mut().enumerate() {
                        let m = _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            (&need_v as *const __m256i as *const u8).add(8 * k) as *const __m128i,
                        ));
                        *t = bsl(m, row_v, *t);
                    }
                    let mut m = _mm256_movemask_epi8(froze_v) as u32;
                    while m != 0 {
                        let l = m.trailing_zeros() as usize;
                        m &= m - 1;
                        limit[l] = row as i32;
                    }
                }
                need_v
            }};
        }
        // The row epilogue's second half, as in `fwd_local_sw_avx512_u8`: `finish_row` returns the
        // lanes a row improves and this recovers their query end. A pair resolves row `i` only for
        // the lanes row `i+1` does not improve again, since `qe` is read only at the last one.
        macro_rules! resolve_qe {
            ($imax:expr, $h:expr, $r:expr, $need:expr) => {{
                let need_v: __m256i = $need;
                if _mm256_movemask_epi8(need_v) != 0 {
                    // The first column holding the row max, for these lanes only: the first block
                    // whose running max reached it (the last block has no checkpoint and is reached
                    // by every lane still pending), scanned backwards so the smallest column wins.
                    let h: *const u8 = $h;
                    let (mut pend_v, mut col) = (need_v, zero);
                    for blk in 0..nblk {
                        let here_v = if blk + 1 < nblk {
                            let c = ck_p.add((blk * 2 + $r) * LANES32) as *const __m256i;
                            _mm256_and_si256(
                                _mm256_cmpeq_epi8(_mm256_loadu_si256(c), $imax),
                                pend_v,
                            )
                        } else {
                            pend_v
                        };
                        if _mm256_movemask_epi8(here_v) == 0 {
                            continue;
                        }
                        for j in (blk * 16..qmax.min(blk * 16 + 16)).rev() {
                            let hj = _mm256_loadu_si256(h.add(j * LANES32) as *const __m256i);
                            let eq = _mm256_and_si256(_mm256_cmpeq_epi8(hj, $imax), here_v);
                            col = bsl(eq, _mm256_set1_epi8(j as i8), col);
                        }
                        pend_v = _mm256_andnot_si256(here_v, pend_v);
                        if _mm256_movemask_epi8(pend_v) == 0 {
                            break;
                        }
                    }
                    qe_v = bsl(need_v, col, qe_v);
                }
            }};
        }

        let pair_rows = rowpair_enabled();
        let mut i = 0usize;
        while i < tmax {
            // Rows consumed by this iteration: 2 when a full pair is left, else 1.
            let rows = if pair_rows && i + 1 < tmax { 2 } else { 1 };
            if rows == 2 {
                let t0_v = _mm256_loadu_si256(seq_t.as_ptr().add(i * LANES32) as *const __m256i);
                let t1_v =
                    _mm256_loadu_si256(seq_t.as_ptr().add((i + 1) * LANES32) as *const __m256i);
                let (mut f0, mut f1) = (hz, hz);
                let (mut d0, mut d1) = (hz, hz);
                let (mut imax0, mut imax1) = (hz, hz);

                for j in 0..n_fast {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);

                    let diag0 = score_diag::<MAT5>(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe0 = _mm256_max_epi8(diag0, e_v);
                    let h0 = _mm256_max_epi8(mfe0, f0);
                    imax0 = _mm256_max_epi8(imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid = _mm256_max_epi8(
                        _mm256_subs_epi8(e_v, e_del_v),
                        _mm256_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm256_max_epi8(
                        _mm256_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);

                    let diag1 = score_diag::<MAT5>(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm256_max_epi8(diag1, e_mid);
                    let h1 = _mm256_max_epi8(mfe1, f1);
                    imax1 = _mm256_max_epi8(imax1, h1);
                    _mm256_storeu_si256(
                        e.as_mut_ptr().add(j * LANES32) as *mut __m256i,
                        _mm256_max_epi8(
                            _mm256_subs_epi8(e_mid, e_del_v),
                            _mm256_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm256_max_epi8(
                        _mm256_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h1);
                    // Row i's H, into the `h_prev` slot `d0` already read (for `resolve_qe`).
                    _mm256_storeu_si256(h_prev.as_mut_ptr().add(j * LANES32) as *mut __m256i, h0);
                    if j & 15 == 15 {
                        let c = ck_p.add((j >> 4) * 2 * LANES32) as *mut __m256i;
                        _mm256_storeu_si256(c, imax0);
                        _mm256_storeu_si256(c.add(1), imax1);
                    }
                }

                for j in n_fast..n_pad {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
                    let zpad_mask = _mm256_cmpeq_epi8(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask in one
                    // instruction instead of three (issue #43).
                    let q_pad = _mm256_cmpgt_epi8(zero, q_v);

                    let mut diag0 = score_diag::<MAT5>(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = bsl(zpad_mask, d0, diag0);
                    diag0 = bsl(
                        _mm256_or_si256(_mm256_cmpgt_epi8(zero, t0_v), q_pad),
                        hz,
                        diag0,
                    );
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe0 = _mm256_max_epi8(diag0, e_v);
                    let h0 = _mm256_max_epi8(mfe0, f0);
                    imax0 = _mm256_max_epi8(imax0, h0);
                    let e_mid = _mm256_max_epi8(
                        _mm256_subs_epi8(e_v, e_del_v),
                        _mm256_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm256_max_epi8(
                        _mm256_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);

                    let mut diag1 = score_diag::<MAT5>(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = bsl(zpad_mask, d1, diag1);
                    diag1 = bsl(
                        _mm256_or_si256(_mm256_cmpgt_epi8(zero, t1_v), q_pad),
                        hz,
                        diag1,
                    );
                    let mfe1 = _mm256_max_epi8(diag1, e_mid);
                    let h1 = _mm256_max_epi8(mfe1, f1);
                    imax1 = _mm256_max_epi8(imax1, h1);
                    _mm256_storeu_si256(
                        e.as_mut_ptr().add(j * LANES32) as *mut __m256i,
                        _mm256_max_epi8(
                            _mm256_subs_epi8(e_mid, e_del_v),
                            _mm256_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm256_max_epi8(
                        _mm256_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h1);
                    // Row i's H, into the `h_prev` slot `d0` already read (for `resolve_qe`).
                    _mm256_storeu_si256(h_prev.as_mut_ptr().add(j * LANES32) as *mut __m256i, h0);
                    if j & 15 == 15 {
                        let c = ck_p.add((j >> 4) * 2 * LANES32) as *mut __m256i;
                        _mm256_storeu_si256(c, imax0);
                        _mm256_storeu_si256(c.add(1), imax1);
                    }
                }

                // Every live lane is ZPAD: `diag = d`, so no query load, score or pad blend. A lane
                // past its target is left un-killed, as in the fast body: `finish_row` skips it.
                for j in n_pad..qmax {
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe0 = _mm256_max_epi8(d0, e_v);
                    let h0 = _mm256_max_epi8(mfe0, f0);
                    imax0 = _mm256_max_epi8(imax0, h0);
                    let e_mid = _mm256_max_epi8(
                        _mm256_subs_epi8(e_v, e_del_v),
                        _mm256_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm256_max_epi8(
                        _mm256_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);

                    let mfe1 = _mm256_max_epi8(d1, e_mid);
                    let h1 = _mm256_max_epi8(mfe1, f1);
                    imax1 = _mm256_max_epi8(imax1, h1);
                    _mm256_storeu_si256(
                        e.as_mut_ptr().add(j * LANES32) as *mut __m256i,
                        _mm256_max_epi8(
                            _mm256_subs_epi8(e_mid, e_del_v),
                            _mm256_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm256_max_epi8(
                        _mm256_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h1);
                    // Row i's H, into the `h_prev` slot `d0` already read (for `resolve_qe`).
                    _mm256_storeu_si256(h_prev.as_mut_ptr().add(j * LANES32) as *mut __m256i, h0);
                    if j & 15 == 15 {
                        let c = ck_p.add((j >> 4) * 2 * LANES32) as *mut __m256i;
                        _mm256_storeu_si256(c, imax0);
                        _mm256_storeu_si256(c.add(1), imax1);
                    }
                }

                // Row order matters: freezing at row `i` must suppress row `i+1`.
                let need0 = finish_row!(i, imax0, t0_v);
                let need1 = finish_row!(i + 1, imax1, t1_v);
                resolve_qe!(imax0, h_prev.as_ptr(), 0, _mm256_andnot_si256(need1, need0));
                resolve_qe!(imax1, h_cur.as_ptr(), 1, need1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = _mm256_loadu_si256(seq_t.as_ptr().add(i * LANES32) as *const __m256i);
                // Row accumulators, one lane per job, same invariant as the scalar version at the top of
                // column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j); `finish_row` finds its column.
                let mut f_v = hz;
                let mut h_diag_v = hz;
                let mut imax_v = hz;

                // Padding-free column range: below `n_fast` no live lane shows ZPAD or PAD, so the two
                // padding masks and their selects are not emitted. Identical argument to the NEON u8
                // kernel, including why a dead target row may be left un-killed here.
                for j in 0..n_fast {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
                    let diag_v = score_diag::<MAT5>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe = _mm256_max_epi8(diag_v, e_v);
                    let h_v = _mm256_max_epi8(mfe, f_v);
                    imax_v = _mm256_max_epi8(imax_v, h_v);
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h_v);
                    if j & 15 == 15 {
                        _mm256_storeu_si256(
                            ck_p.add((j >> 4) * 2 * LANES32) as *mut __m256i,
                            imax_v,
                        );
                    }
                    let e_new = _mm256_max_epi8(
                        _mm256_subs_epi8(e_v, e_del_v),
                        _mm256_subs_epi8(mfe, oe_del_v),
                    );
                    _mm256_storeu_si256(e.as_mut_ptr().add(j * LANES32) as *mut __m256i, e_new);
                    f_v = _mm256_max_epi8(
                        _mm256_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v =
                        _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);
                }

                // Tail: the columns where ZPAD / PAD can appear, full logic.
                for j in n_fast..qmax {
                    // Lane `l` = query base at column `j` of job `l`.
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
                    // `zpad_mask`: the query column is ksw profile padding, so the cell carries the
                    // diagonal through. `pad_mask`: the cell is dead, forced to 0. Both dead tests are
                    // now the NEON high-bit form (`PAD` is the only byte with bit 7 set), because the
                    // old `t >= m` test would kill every real N target once N is re-encoded as 12.
                    let zpad_mask = _mm256_cmpeq_epi8(q_v, zpad_v);
                    let pad_mask =
                        _mm256_or_si256(_mm256_cmpgt_epi8(zero, t_v), _mm256_cmpgt_epi8(zero, q_v));
                    let mut diag_v = score_diag::<MAT5>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = bsl(zpad_mask, h_diag_v, diag_v); // score 0: diagonal passes through
                                                               // Dead padding: force 0 outright, as in the NEON u8 kernel.
                    diag_v = bsl(pad_mask, hz, diag_v);

                    // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                    // saturation, so the two maxes are the whole H recurrence.
                    // `mfe = max(diag, E)` excludes the serial F carry, so the F recurrence below can read
                    // it instead of the full `h_v` and stop waiting on H -- the same critical-chain
                    // shortening as the NEON u8 kernel (f -> h -> f becomes f -> f).
                    let mfe = _mm256_max_epi8(diag_v, e_v);
                    let h_v = _mm256_max_epi8(mfe, f_v);
                    imax_v = _mm256_max_epi8(imax_v, h_v);
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h_v);
                    if j & 15 == 15 {
                        _mm256_storeu_si256(
                            ck_p.add((j >> 4) * 2 * LANES32) as *mut __m256i,
                            imax_v,
                        );
                    }

                    // e = max(0, e-e_del, h-oe_del); both saturating subs supply their own clamp.
                    let e_new = _mm256_max_epi8(
                        _mm256_subs_epi8(e_v, e_del_v),
                        _mm256_subs_epi8(mfe, oe_del_v),
                    );
                    _mm256_storeu_si256(e.as_mut_ptr().add(j * LANES32) as *mut __m256i, e_new);
                    // F, the row's serial carry, reassociated off `mfe` (see the NEON u8 kernel for the
                    // byte-identity argument: oe_ins >= e_ins makes the `f - oe_ins` branch inside H
                    // dominated, so `max(f - e_ins, mfe - oe_ins)` equals the original).
                    f_v = _mm256_max_epi8(
                        _mm256_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm256_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm256_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v =
                        _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);
                }
                let need = finish_row!(i, imax_v, t_v);
                resolve_qe!(imax_v, h_cur.as_ptr(), 0, need);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once every live lane is either frozen or out of target: purely a speed win,
            // as in the NEON kernels. See the longer note in `fwd_local_sw_neon`. A target ends where
            // its bytes turn `PAD`, so the next row's bases say which lanes still run.
            i += rows;
            if i >= tmax
                || _mm256_movemask_epi8(_mm256_or_si256(
                    frozen_v,
                    _mm256_loadu_si256(seq_t.as_ptr().add(i * LANES32) as *const __m256i),
                )) == -1
            {
                break;
            }
        }
        let mut gmax_arr = [0u8; LANES32];
        _mm256_storeu_si256(
            gmax_arr.as_mut_ptr() as *mut __m256i,
            _mm256_xor_si256(gmax_v, hz),
        );
        for (g, &v) in gmax.iter_mut().zip(&gmax_arr) {
            *g = i32::from(v);
        }
        let mut te = [-1i32; LANES32];
        for (k, t) in te_v.iter().enumerate() {
            _mm256_storeu_si256(te.as_mut_ptr().add(8 * k) as *mut __m256i, *t);
        }
        let mut qe_arr = [0u8; LANES32];
        _mm256_storeu_si256(qe_arr.as_mut_ptr() as *mut __m256i, qe_v);
        let qe = qe_arr.map(i32::from);

        extract_group(
            n_lanes, group_idx, LANES32, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// [`fwd_local_sw_avx2_u8_impl`] in the form `mat` takes: the standard XOR table, or `MAT5`.
///
/// # Safety
/// As the impl; a non-standard `mat` must pass `mat5_u8_ok`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx2_u8(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    // SAFETY: the caller's.
    unsafe {
        let share = o_del + e_del == o_ins + e_ins;
        match (mat_is_standard(m, mat), share) {
            (true, false) => fwd_local_sw_avx2_u8_impl::<false, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (true, true) => fwd_local_sw_avx2_u8_impl::<false, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, false) => fwd_local_sw_avx2_u8_impl::<true, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, true) => fwd_local_sw_avx2_u8_impl::<true, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
        }
    }
}

/// Lanes for the AVX2 i16 kernel: one `__m256i` holds 16 signed words, half the u8 width. This is the
/// x86 counterpart of [`LANES16`] on aarch64, and the reason a job that overflows u8 no longer drops
/// to the scalar lockstep on x86_64. Same "wider group is only extra padded work" argument as
/// [`LANES32`].
#[cfg(target_arch = "x86_64")]
const LANES16: usize = 16;

/// AVX2 `i16x16` forward local-SW: [`fwd_local_sw_neon`] transliterated to `__m256i`, so a batch whose
/// score ceiling overflows u8 runs vectorised on x86_64 instead of falling through to
/// [`fwd_local_sw_scalar`]. This is the second half of closing the x86 rescue gap (#12/#20): the u8
/// kernel above already handles stock DNA scores, this one the high-scoring configs (large `a`, or the
/// NEON gate's `(10, 40)` matrix) where `min(len) * match` exceeds 250.
///
/// The transliteration is mechanical, exactly as for [`fwd_local_sw_avx2_u8`], because every operation
/// is element-wise and i16 signed arithmetic matches NEON's `int16x8` byte-for-byte. Two NEON spellings
/// have no one-instruction AVX2 form and are rebuilt here:
///
/// - **`vcgeq_s16(a, b)`** (signed `>=`): recovered as `max_epi16(a, b) == a`. Every value compared
///   this way is a base code (0..5) or [`PAD`] widened *unsigned* to a small positive i16, so the
///   signed max is exact.
/// - **`vbslq_s16(mask, a, b)`** = `mask ? a : b`: `_mm256_blendv_epi8(a, b, mask)` is `mask ? b : a`,
///   so the operands swap, same as the u8 kernel's `bsl`. The per-word masks are all-ones/all-zero, so
///   consulting the top bit of each byte is exact.
///
/// Unlike NEON's byte loads, widening a 16-code column to i16 is one `_mm_loadu_si128` (16 bytes) then
/// `_mm256_cvtepu8_epi16`, the unsigned widen that keeps [`PAD`] = 255 as +255 rather than -1.
///
/// # Parameters / Returns
/// As [`fwd_local_sw_batch`], with the same unchecked preconditions as [`fwd_local_sw_neon`]:
/// `mat_is_standard(m, mat)`, and every job's score ceiling `min(qlen, tlen) * max_sc` plus target
/// length under [`I16_SCORE_LIMIT`]. Byte-identical to [`fwd_local_sw_scalar`]
/// (`avx2_verify::avx2_matesw_i16_matches_scalar`).
///
/// # Safety
/// Caller must have confirmed AVX2 is available. Loads/stores use unchecked offsets bounded by
/// `qmax`/`tmax`, derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx2_i16<const MAT5: bool>(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    // `mask ? a : b`, per 16-bit lane; the all-ones/all-zero masks make the byte-granular blend exact.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bsl(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_blendv_epi8(b, a, mask)
    }
    // Signed `a >= b`, all-ones/all-zero per word. `max_epi16(a,b) == a` iff `a >= b`; used only on the
    // small positive code/PAD values, where signed and unsigned max coincide.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epi16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(_mm256_max_epi16(a, b), a)
    }

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // The standard matrix collapses to `mtch` (positive match) and `mis` (signed mismatch, e.g. -4),
    // plus the fixed -1 for N, exactly as in the NEON i16 kernel.
    let mtch = mat[0] as i16;
    let mis = mat[1] as i16;
    // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    let zero = _mm256_setzero_si256();
    // Increment for the carried column counter `j_v` (see the fast column range below).
    let one_v = _mm256_set1_epi16(1);
    let mtch_v = _mm256_set1_epi16(mtch);
    let mis_v = _mm256_set1_epi16(mis);
    let n_v = _mm256_set1_epi16(-1);
    let dead_v = _mm256_set1_epi16(DEAD_CELL_SCORE as i16);
    let four_v = _mm256_set1_epi16(4);
    // `MAT5` codes run to 15 (`mat5_add_code`), so the dead tests move above them.
    let m_v = _mm256_set1_epi16(if MAT5 { 16 } else { m as i16 });
    let zpad_v = _mm256_set1_epi16(ZPAD as i16);
    let qdead_v = _mm256_set1_epi16(if MAT5 { 15 } else { ZPAD as i16 });
    // `MAT5` (a `--meth` matrix): the cell score is one `pshufb` at `(t + q) & 15` on the encoded
    // byte codes, as in the u8 kernels; ZPAD (5, never an encoded query code) and PAD keep the
    // tail's selects.
    let (tbl, enc_t, enc_q) = if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        (_mm_loadu_si128(t16.as_ptr() as *const __m128i), t, q)
    } else {
        (_mm_setzero_si128(), [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    };
    let e_del_v = _mm256_set1_epi16(e_del as i16);
    let oe_del_v = _mm256_set1_epi16(oe_del as i16);
    let e_ins_v = _mm256_set1_epi16(e_ins as i16);
    let oe_ins_v = _mm256_set1_epi16(oe_ins as i16);

    for (group_idx, group) in jobs.chunks(LANES16).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES16];
        let mut seq_t = vec![PAD; tmax * LANES16];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES16],
            [0usize; LANES16],
            [i32::MAX; LANES16],
            [i32::MAX; LANES16],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES16 + l] = if MAT5 { enc_q[b as usize] } else { b };
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES16 + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = if MAT5 { enc_t[b as usize] } else { b };
            }
        }

        // i16 SoA DP state, one `__m256i` per column across the 16 lanes. `rowmax` stays i32; it is only
        // read by the scalar bookkeeping. Plain wrapping i16 arithmetic is safe because the caller
        // proved every ceiling is under `I16_SCORE_LIMIT` and the kill score is only `DEAD_CELL_SCORE`.
        let mut h_prev = vec![0i16; qmax * LANES16];
        let mut h_cur = vec![0i16; qmax * LANES16];
        let mut e = vec![0i16; qmax * LANES16];
        let mut rowmax = vec![0i32; tmax * LANES16];
        let mut gmax = [0i32; LANES16];
        let mut te = [-1i32; LANES16];
        let mut qe = [0i32; LANES16];
        let mut limit = [-1i32; LANES16];
        let mut frozen = [false; LANES16];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // Widen 16 u8 codes at `off` into an `i16x16`. `_mm256_cvtepu8_epi16` is the unsigned widen, so
        // PAD (255) stays +255 and not -1, matching NEON's `vmovl_u8`.
        let load_codes = |buf: &[u8], off: usize| -> __m256i {
            _mm256_cvtepu8_epi16(_mm_loadu_si128(buf.as_ptr().add(off) as *const __m128i))
        };
        // `MAT5`'s cell score at column `j`, from the row's byte codes `t_b`.
        let mat5_sc = |t_b: __m128i, j: usize| -> __m256i {
            let q_b = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(tbl, _mm_add_epi8(t_b, q_b)))
        };

        // =====================================================================================
        // Main DP, one target row per iteration. Structurally identical to `fwd_local_sw_neon`, at
        // 16 lanes; only the per-cell arithmetic is vectorised, the bookkeeping stays scalar.
        // =====================================================================================
        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES16);
            let t_b = _mm_loadu_si128(seq_t.as_ptr().add(i * LANES16) as *const __m128i);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
                                       // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm256_cmpeq_epi16(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    let q_v = load_codes(&seq_q, j * LANES16);
                    let eq = _mm256_cmpeq_epi16(t_v, q_v);
                    let n_mask = _mm256_or_si256(t_is_n, _mm256_cmpeq_epi16(q_v, four_v));
                    bsl(n_mask, n_v, bsl(eq, mtch_v, mis_v))
                };
                let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES16) as *const __m256i);
                let mut h_v = _mm256_add_epi16(h_diag_v, sc);
                h_v = _mm256_max_epi16(h_v, zero);
                h_v = _mm256_max_epi16(h_v, e_v);
                let mfe = h_v;
                h_v = _mm256_max_epi16(h_v, f_v);
                let is_new_row_max = _mm256_cmpgt_epi16(h_v, imax_v);
                imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                imax_v = _mm256_max_epi16(imax_v, h_v);
                _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m256i, h_v);
                let e_new = _mm256_max_epi16(
                    _mm256_sub_epi16(e_v, e_del_v),
                    _mm256_sub_epi16(h_v, oe_del_v),
                );
                _mm256_storeu_si256(
                    e.as_mut_ptr().add(j * LANES16) as *mut __m256i,
                    _mm256_max_epi16(e_new, zero),
                );
                let f_new = _mm256_max_epi16(
                    _mm256_sub_epi16(f_v, e_ins_v),
                    _mm256_sub_epi16(mfe, oe_ins_v),
                );
                f_v = _mm256_max_epi16(f_new, zero);
                h_diag_v = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES16) as *const __m256i);
                j_v = _mm256_add_epi16(j_v, one_v);
            }

            // Tail: the columns where ZPAD / PAD can appear, full logic.
            for j in n_fast..qmax {
                let q_v = load_codes(&seq_q, j * LANES16);
                // Four masks, four selects in increasing priority, exactly as the NEON i16 kernel:
                //   eq        the bases match; n_mask either is N; zpad_mask query is profile padding;
                //   pad_mask  cell is past a real position (dead target row, or query past the profile).
                let eq = _mm256_cmpeq_epi16(t_v, q_v);
                let n_mask = _mm256_or_si256(t_is_n, _mm256_cmpeq_epi16(q_v, four_v));
                let zpad_mask = _mm256_cmpeq_epi16(q_v, zpad_v);
                let pad_mask =
                    _mm256_or_si256(cge_epi16(t_v, m_v), _mm256_cmpgt_epi16(q_v, qdead_v));
                let mut sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    bsl(n_mask, n_v, bsl(eq, mtch_v, mis_v))
                };
                sc = bsl(zpad_mask, zero, sc); // ksw profile padding scores 0
                sc = bsl(pad_mask, dead_v, sc); // dead padding: kill the cell outright

                let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES16) as *const __m256i);
                // H = max(0, H_diag + S, E, F); wrapping adds are in range by the ceiling guarantee.
                let mut h_v = _mm256_add_epi16(h_diag_v, sc);
                h_v = _mm256_max_epi16(h_v, zero);
                h_v = _mm256_max_epi16(h_v, e_v);
                // `mfe = max(0, H_diag + S, E)` excludes the serial F carry, so the F recurrence below
                // reads it and the critical column chain stays f -> f (see the NEON i16 kernel).
                let mfe = h_v;
                h_v = _mm256_max_epi16(h_v, f_v);
                // Strict signed `>`, so a tie keeps the earlier column.
                let is_new_row_max = _mm256_cmpgt_epi16(h_v, imax_v);
                imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                imax_v = _mm256_max_epi16(imax_v, h_v);
                _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m256i, h_v);

                // E(i+1,j) = max(0, E - e_del, H - oe_del).
                let e_new = _mm256_max_epi16(
                    _mm256_sub_epi16(e_v, e_del_v),
                    _mm256_sub_epi16(h_v, oe_del_v),
                );
                _mm256_storeu_si256(
                    e.as_mut_ptr().add(j * LANES16) as *mut __m256i,
                    _mm256_max_epi16(e_new, zero),
                );
                // F(i,j+1) = max(0, F - e_ins, mfe - oe_ins), reassociated off `mfe`.
                let f_new = _mm256_max_epi16(
                    _mm256_sub_epi16(f_v, e_ins_v),
                    _mm256_sub_epi16(mfe, oe_ins_v),
                );
                f_v = _mm256_max_epi16(f_new, zero);
                h_diag_v = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES16) as *const __m256i);
                j_v = _mm256_add_epi16(j_v, one_v);
            }

            let mut imax_arr = [0i16; LANES16];
            let mut col_arr = [0i16; LANES16];
            _mm256_storeu_si256(imax_arr.as_mut_ptr() as *mut __m256i, imax_v);
            _mm256_storeu_si256(col_arr.as_mut_ptr() as *mut __m256i, imax_col_v);
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                let row_max = imax_arr[l] as i32;
                rowmax[i * LANES16 + l] = row_max;
                if row_max > gmax[l] {
                    gmax[l] = row_max;
                    te[l] = i as i32;
                    qe[l] = col_arr[l] as i32;
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            if (0..n_lanes).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n_lanes, group_idx, LANES16, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// Lanes for the AVX512 u8 kernel: one `__m512i` holds 64 bytes, twice the AVX2 u8 width. The i16
/// AVX512 kernel reuses [`LANES32`] (32 words). As with every lane constant here, a wider group only
/// adds padded work; no threshold moves, because every threshold is bwa's, not the register's.
#[cfg(target_arch = "x86_64")]
const LANES64: usize = 64;

// ---------------------------------------------------------------------------------------------
// SSE4.1 (128-bit) rescue kernels.
//
// Same two reasons as the extension kernels in `crate::batched::sse41`. Coverage: without these, an
// x86_64 host without AVX2, which includes VMs and container hosts that mask CPU features, ran mate
// rescue through the scalar path. And width: a 256-bit operation is only worth its encoding when
// the hardware executes it as one operation, which under emulation it does not.
//
// The transliteration from the AVX2 kernels is mechanical, operation for operation, because every
// intrinsic they use is element-wise and has an exact 128-bit spelling. The one exception is the
// widening load: AVX2 widens 16 codes with `_mm_loadu_si128` + `_mm256_cvtepu8_epi16`, and the
// 128-bit form widens 8 with `_mm_loadl_epi64` + `_mm_cvtepu8_epi16`. Byte-identity is not argued
// from that mechanical correspondence, it is tested: `sse41_verify` checks both kernels against the
// scalar reference on the same randomised jobs the AVX2 gate uses.
// ---------------------------------------------------------------------------------------------

/// SSE4.1's element-wise byte ops behave exactly like NEON's despite the 128-bit-lane split that
/// afflicts its *shuffles*. Two operations have no direct SSE4.1 spelling and are emulated below with
/// the same trick `crate::batched`'s SSE4.1 kernels use:
///
/// - **signed domain**: SSE4.1 has only *signed* `_mm_cmpgt_epi8`, so H, E and F are held as
///   `H - 128` (XOR 0x80), where signed order is unsigned order. Every max, saturating subtract and
///   compare is then its `epi8` twin, saturation at -128 is the DP's clamp at 0, the diagonal adds an
///   unbiased table delta with one `adds_epi8`, and only the row max is XORed back: the fork's `Sgn`
///   u8 body (`kswv.cpp`). The strict `>` on the row argmax still keeps `qe` at the *smallest*
///   attaining column (`ksw.cpp:216-218`). Both gaps open from `mfe`, not `h` (the FScan
///   cell, see `ef!` in [`fwd_local_sw_neon_u8_impl`]), so under `SHARE` (`o_del + e_del == o_ins +
///   e_ins`) E and F share one subtract.
/// - **bit-select**: NEON `vbslq_u8(mask, a, b)` is `mask ? a : b`; `_mm_blendv_epi8(a, b, mask)`
///   is `mask ? b : a`, i.e. the operands are the other way round.
///
/// No cross-lane reduction is needed anywhere: the row accumulators are spilled to arrays and read
/// per lane, exactly as the NEON kernel does, so `vmaxvq_u8` never appears and neither does the
/// `_mm_extracti128_si256` dance it would require.
///
/// # Parameters
/// As [`fwd_local_sw_batch`], with the same unchecked preconditions as [`fwd_local_sw_neon_u8`]:
/// `mat_is_standard(m, mat)`, every job's score ceiling `min(qlen, tlen) * max_sc` under
/// [`U8_SCORE_LIMIT`], and every query shorter than [`U8_SCORE_LIMIT`] bases.
///
/// # Returns
/// As [`fwd_local_sw_batch`], and byte-identical to [`fwd_local_sw_scalar`] on the same input
/// (`sse41_verify::sse41_matesw_u8_matches_scalar`).
///
/// # Safety
/// Caller must have confirmed SSE4.1 is available. Loads and stores use unchecked pointer offsets
/// bounded by `qmax`/`tmax`, which are derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_sse41_u8_impl<const MAT5: bool, const SHARE: bool>(
    jobs: &[FwdJob],
    // Matrix dimension (5). No longer read now that dead-cell detection keys on the PAD
    // byte's high bit rather than `>= m`, exactly as in `fwd_local_sw_neon_u8`; kept in the
    // signature for parity with the other kernels.
    _m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    // NEON `vbslq_u8(mask, a, b)` = `mask ? a : b`. `_mm_blendv_epi8(a, b, mask)` = `mask ? b : a`,
    // hence the swapped operands; only the top bit of each mask byte is consulted, which the
    // all-ones masks above satisfy.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn bsl(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_blendv_epi8(b, a, mask)
    }

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as u8; // match bonus (>= 0)
    let mispen = (-mat[1]) as u8; // mismatch penalty b (mat[1] = -b)
                                  // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    // Broadcast constants; gap penalties are held as magnitudes (subtracted, not added).
    let zero = _mm_setzero_si128();
    // The DP's 0 in the signed H domain (H - 128): saturation at -128 is the clamp at 0, and
    // every max, subtract and compare has its exact signed twin (the fork's `Sgn` u8 body).
    let hz = _mm_set1_epi8(-128);
    // ISSUE #43: the XOR-indexed score table, ported from `fwd_local_sw_neon_u8`. See there for
    // why the target N is re-encoded as `N_TARGET` (12) and what each of the 16 slots means. The
    // saturation guard is the same, and it is a HARD assert because the release path depends on it.
    assert!(
        (U8_SCORE_LIMIT as u32) + mispen as u32 + mtch as u32 <= 256,
        "u8 rescue score table would saturate: mispen {mispen} + mtch {mtch} + ceiling too large"
    );
    let bias = mispen;
    let mut tbl = [0u8; 16];
    for r in 0..1 {
        let o = r * 16;
        tbl[o] = bias + mtch; // match: +a
                              // tbl[o + 1..=o + 3] = bias - b = 0 (mismatch), already zero
        for k in 4..8 {
            tbl[o + k] = bias - 1; // query N against a real base: bwa scores -1
        }
        tbl[o + 8] = bias - 1; // both N: N_TARGET XOR 4
        for k in 12..16 {
            tbl[o + k] = bias - 1; // target N against a real base
        }
    }
    // `vpshufb` is IN-LANE on AVX2, so the 16-byte table is written into both 128-bit halves; on
    // SSE4.1 there is a single lane and the loop above runs once.
    // H is held as H - 128 (signed), so the table holds the unbiased deltas.
    for v in &mut tbl {
        *v = v.wrapping_sub(bias);
    }
    let score_tbl = _mm_loadu_si128(tbl.as_ptr() as *const __m128i);
    let bias_v = _mm_set1_epi8(bias as i8);
    // `MAT5` (`--meth`'s matrices, `mat5_u8_ok`): one lookup at `(t + q) & 15` through
    // `mat5_add_code` (which the dispatch checked exists), `seq_t` / `seq_q` holding its codes.
    // Every cell lies in `[mat[1], mat[0]]`, a signed byte added straight to the `H - 128` domain.
    let (score_tbl, enc_t, enc_q) = if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        (_mm_loadu_si128(t16.as_ptr() as *const __m128i), t, q)
    } else {
        (score_tbl, [0, 1, 2, 3, N_TARGET], [0, 1, 2, 3, 4])
    };
    // `vpshufb` zeroes a lane whose index has bit 7 set and otherwise reads `tbl[idx & 15]`, where
    // `vqtbl1q_u8` zeroes every index >= 16. They differ only on indices 16..127, and that range is
    // UNREACHABLE here: the only alphabet byte with bit 7 set is `PAD`, `seq_q` holds
    // {0,1,2,3,4,ZPAD=5,PAD} and `seq_t` holds {0,1,2,3,N_TARGET=12,PAD}, so with neither operand
    // PAD both are <= 12 and `t ^ q <= 15` (the maximum is `3 ^ 12`); with exactly one PAD the xor
    // has bit 7 set and both instructions give 0; with both PAD the xor is 0 and both give
    // `tbl[0]`, a cell the pad blend overrides anyway. So the NEON byte-identity argument carries
    // over unchanged.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn score_diag<const M5: bool>(
        tbl: __m128i,
        _bias_v: __m128i,
        t_v: __m128i,
        q_v: __m128i,
        d: __m128i,
    ) -> __m128i {
        // `M5`: the add code's slot `(t + q) & 15`; see the AVX2 kernel's `score_diag`.
        let i = if M5 {
            _mm_add_epi8(t_v, q_v)
        } else {
            _mm_xor_si128(t_v, q_v)
        };
        _mm_adds_epi8(d, _mm_shuffle_epi8(tbl, i))
    }
    let zpad_v = _mm_set1_epi8(ZPAD as i8);
    let e_del_v = _mm_set1_epi8(e_del as i8);
    let oe_del_v = _mm_set1_epi8(oe_del as i8);
    let e_ins_v = _mm_set1_epi8(e_ins as i8);
    let oe_ins_v = _mm_set1_epi8(oe_ins as i8);

    // Group setup is identical to `fwd_local_sw_scalar` at 16 lanes; see there for each variable.
    for (group_idx, group) in jobs.chunks(LANES16).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES16];
        let mut seq_t = vec![PAD; tmax * LANES16];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES16],
            [0usize; LANES16],
            [i32::MAX; LANES16],
            [i32::MAX; LANES16],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES16 + l] = enc_q[b as usize];
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES16 + l] = ZPAD;
            }
            // The N re-encoding (or `MAT5`'s add code), applied once per base so the column loops
            // never pay for it (issue #43). See `fwd_local_sw_neon_u8`'s score-table comment.
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = enc_t[b as usize];
            }
        }

        // Same DP state as the scalar path, narrowed to u8 so a column of all 16 lanes is one
        // `_mm_loadu_si128`. Legal only because local SW keeps every value in `[0, ceiling]` and
        // the caller proved the ceiling is under 250.
        let mut h_prev = vec![0x80u8; qmax * LANES16];
        let mut h_cur = vec![0x80u8; qmax * LANES16];
        let mut e = vec![0x80u8; qmax * LANES16];
        // LazyQE (bwa-mem3), as in `fwd_local_sw_avx512_u8`: no per-cell argmax. Each row's running
        // max at every 16th column, a pair's two rows interleaved, so `resolve_qe` finds the column
        // only for the lanes whose row max improves, and only inside the one block that reached it.
        let nblk = qmax.div_ceil(16);
        let mut ck = vec![0x80u8; nblk * 2 * LANES16];
        let ck_p = ck.as_mut_ptr();
        // ISSUE #46C. `rowmax` is u8, not i32, and the row is published with ONE vector store
        // instead of `LANES` guarded scalar ones, exactly as `fwd_local_sw_neon_u8` already does.
        // Two reasons, both structural rather than micro:
        //
        //  - size. At AVX-512's mean shape the i32 buffer was 359 KB per group, out of L1 and into
        //    L2, and `extract_group`'s per-lane walk then touched a fresh cache line for every load,
        //    re-fetching each line once per lane. As u8 it is 90 KB and the stride is exactly one
        //    64-byte line per row of 64 lanes.
        //  - stores. The scalar form issued one guarded store per live lane per row, 32 or 64 of
        //    them; the vector form issues one.
        //
        // Writing it unconditionally is safe for the same reason it is on NEON: a lane's row maxima
        // are read only for rows `0..=limit[l]`, and `limit[l]` is `tlen[l] - 1` or the freeze row,
        // so every slot this writes beyond the old guard is one nothing reads. Values fit u8 because
        // the caller proved the score ceiling is under `U8_SCORE_LIMIT`.
        let mut rowmax = vec![0u8; tmax * LANES16];
        let mut gmax = [0i32; LANES16];
        let mut te = [-1i32; LANES16];
        let mut qe = [0i32; LANES16];
        let mut limit = [-1i32; LANES16];
        let mut frozen = [false; LANES16];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // =====================================================================================
        // Main DP, two target rows per iteration when two are left. Same transformation, and the
        // same byte-identity argument, as `fwd_local_sw_neon_u8`: rows `i` and `i+1` share the
        // query-column load, `h_prev[j]`, `e[j]` and the `h_cur[j]` store, because row `i+1`'s
        // diagonal is row `i`'s previous-column H and its E carry is what row `i` just produced.
        // =====================================================================================
        // The row epilogue, in two halves as in `fwd_local_sw_avx512_u8`: `finish_row` folds the row
        // into `rowmax`, `gmax`, `te` and the freeze and returns the lanes it improves, and
        // `resolve_qe` recovers their query end. A pair resolves row `i` only for the lanes row
        // `i+1` does not improve again, since a lane's `qe` is read only at its last improving row.
        macro_rules! finish_row {
            ($row:expr, $imax:expr) => {{
                let row = $row;
                let mut imax_arr = [0u8; LANES16];
                let imax_u = _mm_xor_si128($imax, hz);
                _mm_storeu_si128(imax_arr.as_mut_ptr() as *mut __m128i, imax_u);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm_storeu_si128(
                    rowmax.as_mut_ptr().add(row * LANES16) as *mut __m128i,
                    imax_u,
                );
                let mut need = 0u32;
                for l in 0..n_lanes {
                    if row >= tlen[l] || frozen[l] {
                        continue;
                    }
                    // Job `l`'s best H anywhere in target row `row`, widened out of the lane.
                    let row_max = imax_arr[l] as i32;
                    if row_max > gmax[l] {
                        gmax[l] = row_max;
                        te[l] = row as i32;
                        need |= 1 << l;
                        if gmax[l] >= endsc[l] {
                            frozen[l] = true;
                            limit[l] = row as i32;
                        }
                    }
                }
                need
            }};
        }
        macro_rules! resolve_qe {
            ($row:expr, $h:expr, $r:expr, $need:expr) => {{
                let (row, h, mut need): (usize, *const u8, u32) = ($row, $h, $need);
                while need != 0 {
                    let l = need.trailing_zeros() as usize;
                    need &= need - 1;
                    // The first block whose running max reached the row max holds the first column
                    // attaining it (the last block, never checkpointed when partial, by
                    // elimination). Strict `>` in the old inline argmax kept that same column.
                    let m = rowmax[row * LANES16 + l] ^ 0x80;
                    let mut blk = 0;
                    while blk + 1 < nblk && *ck_p.add((blk * 2 + $r) * LANES16 + l) != m {
                        blk += 1;
                    }
                    let mut j = blk * 16;
                    while *h.add(j * LANES16 + l) != m {
                        j += 1;
                    }
                    qe[l] = j as i32;
                }
            }};
        }

        let pair_rows = rowpair_enabled();
        let mut i = 0usize;
        while i < tmax {
            // Rows consumed by this iteration: 2 when a full pair is left, else 1.
            let rows = if pair_rows && i + 1 < tmax { 2 } else { 1 };
            if rows == 2 {
                let t0_v = _mm_loadu_si128(seq_t.as_ptr().add(i * LANES16) as *const __m128i);
                let t1_v = _mm_loadu_si128(seq_t.as_ptr().add((i + 1) * LANES16) as *const __m128i);
                let (mut f0, mut f1) = (hz, hz);
                let (mut d0, mut d1) = (hz, hz);
                let (mut imax0, mut imax1) = (hz, hz);

                for j in 0..n_fast {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);

                    let diag0 = score_diag::<MAT5>(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe0 = _mm_max_epi8(diag0, e_v);
                    let h0 = _mm_max_epi8(mfe0, f0);
                    imax0 = _mm_max_epi8(imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid =
                        _mm_max_epi8(_mm_subs_epi8(e_v, e_del_v), _mm_subs_epi8(mfe0, oe_del_v));
                    f0 = _mm_max_epi8(
                        _mm_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);

                    let diag1 = score_diag::<MAT5>(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm_max_epi8(diag1, e_mid);
                    let h1 = _mm_max_epi8(mfe1, f1);
                    imax1 = _mm_max_epi8(imax1, h1);
                    _mm_storeu_si128(
                        e.as_mut_ptr().add(j * LANES16) as *mut __m128i,
                        _mm_max_epi8(_mm_subs_epi8(e_mid, e_del_v), _mm_subs_epi8(mfe1, oe_del_v)),
                    );
                    f1 = _mm_max_epi8(
                        _mm_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h1);
                    // Row i's H, into the `h_prev` slot `d0` already read (for `resolve_qe`).
                    _mm_storeu_si128(h_prev.as_mut_ptr().add(j * LANES16) as *mut __m128i, h0);
                    if j & 15 == 15 {
                        let c = ck_p.add((j >> 4) * 2 * LANES16) as *mut __m128i;
                        _mm_storeu_si128(c, imax0);
                        _mm_storeu_si128(c.add(1), imax1);
                    }
                }

                for j in n_fast..qmax {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
                    let zpad_mask = _mm_cmpeq_epi8(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask in one
                    // instruction instead of three (issue #43).
                    let q_pad = _mm_cmpgt_epi8(zero, q_v);

                    let mut diag0 = score_diag::<MAT5>(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = bsl(zpad_mask, d0, diag0);
                    diag0 = bsl(_mm_or_si128(_mm_cmpgt_epi8(zero, t0_v), q_pad), hz, diag0);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe0 = _mm_max_epi8(diag0, e_v);
                    let h0 = _mm_max_epi8(mfe0, f0);
                    imax0 = _mm_max_epi8(imax0, h0);
                    let e_mid =
                        _mm_max_epi8(_mm_subs_epi8(e_v, e_del_v), _mm_subs_epi8(mfe0, oe_del_v));
                    f0 = _mm_max_epi8(
                        _mm_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);

                    let mut diag1 = score_diag::<MAT5>(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = bsl(zpad_mask, d1, diag1);
                    diag1 = bsl(_mm_or_si128(_mm_cmpgt_epi8(zero, t1_v), q_pad), hz, diag1);
                    let mfe1 = _mm_max_epi8(diag1, e_mid);
                    let h1 = _mm_max_epi8(mfe1, f1);
                    imax1 = _mm_max_epi8(imax1, h1);
                    _mm_storeu_si128(
                        e.as_mut_ptr().add(j * LANES16) as *mut __m128i,
                        _mm_max_epi8(_mm_subs_epi8(e_mid, e_del_v), _mm_subs_epi8(mfe1, oe_del_v)),
                    );
                    f1 = _mm_max_epi8(
                        _mm_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h1);
                    // Row i's H, into the `h_prev` slot `d0` already read (for `resolve_qe`).
                    _mm_storeu_si128(h_prev.as_mut_ptr().add(j * LANES16) as *mut __m128i, h0);
                    if j & 15 == 15 {
                        let c = ck_p.add((j >> 4) * 2 * LANES16) as *mut __m128i;
                        _mm_storeu_si128(c, imax0);
                        _mm_storeu_si128(c.add(1), imax1);
                    }
                }

                // Row order matters: freezing at row `i` must suppress row `i+1`.
                let need0 = finish_row!(i, imax0);
                let need1 = finish_row!(i + 1, imax1);
                resolve_qe!(i, h_prev.as_ptr(), 0, need0 & !need1);
                resolve_qe!(i + 1, h_cur.as_ptr(), 1, need1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = _mm_loadu_si128(seq_t.as_ptr().add(i * LANES16) as *const __m128i);
                // Row accumulators, one lane per job, same invariant as the scalar version at the top of
                // column `j`: F(i, j), H(i-1, j-1) and max H(i, 0..j).
                let mut f_v = hz;
                let mut h_diag_v = hz;
                let mut imax_v = hz;

                // Padding-free column range: below `n_fast` no live lane shows ZPAD or PAD, so the two
                // padding masks and their selects are not emitted. Identical argument to the NEON u8
                // kernel, including why a dead target row may be left un-killed here.
                for j in 0..n_fast {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
                    let diag_v = score_diag::<MAT5>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe = _mm_max_epi8(diag_v, e_v);
                    let h_v = _mm_max_epi8(mfe, f_v);
                    imax_v = _mm_max_epi8(imax_v, h_v);
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h_v);
                    let e_new =
                        _mm_max_epi8(_mm_subs_epi8(e_v, e_del_v), _mm_subs_epi8(mfe, oe_del_v));
                    _mm_storeu_si128(e.as_mut_ptr().add(j * LANES16) as *mut __m128i, e_new);
                    f_v = _mm_max_epi8(
                        _mm_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);
                    if j & 15 == 15 {
                        _mm_storeu_si128(ck_p.add((j >> 4) * 2 * LANES16) as *mut __m128i, imax_v);
                    }
                }

                // Tail: the columns where ZPAD / PAD can appear, full logic.
                for j in n_fast..qmax {
                    // Lane `l` = query base at column `j` of job `l`.
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
                    // `zpad_mask`: the query column is ksw profile padding, so the cell carries the
                    // diagonal through. `pad_mask`: the cell is dead, forced to 0. Both dead tests are
                    // now the NEON high-bit form (`PAD` is the only byte with bit 7 set), because the
                    // old `t >= m` test would kill every real N target once N is re-encoded as 12.
                    let zpad_mask = _mm_cmpeq_epi8(q_v, zpad_v);
                    let pad_mask =
                        _mm_or_si128(_mm_cmpgt_epi8(zero, t_v), _mm_cmpgt_epi8(zero, q_v));
                    let mut diag_v = score_diag::<MAT5>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = bsl(zpad_mask, h_diag_v, diag_v); // score 0: diagonal passes through
                                                               // Dead padding: force 0 outright, as in the NEON u8 kernel.
                    diag_v = bsl(pad_mask, hz, diag_v);

                    // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                    // saturation, so the two maxes are the whole H recurrence.
                    // `mfe = max(diag, E)` excludes the serial F carry, so the F recurrence below can read
                    // it instead of the full `h_v` and stop waiting on H -- the same critical-chain
                    // shortening as the NEON u8 kernel (f -> h -> f becomes f -> f).
                    let mfe = _mm_max_epi8(diag_v, e_v);
                    let h_v = _mm_max_epi8(mfe, f_v);
                    imax_v = _mm_max_epi8(imax_v, h_v);
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h_v);

                    // e = max(0, e-e_del, h-oe_del); both saturating subs supply their own clamp.
                    let e_new =
                        _mm_max_epi8(_mm_subs_epi8(e_v, e_del_v), _mm_subs_epi8(mfe, oe_del_v));
                    _mm_storeu_si128(e.as_mut_ptr().add(j * LANES16) as *mut __m128i, e_new);
                    // F, the row's serial carry, reassociated off `mfe` (see the NEON u8 kernel for the
                    // byte-identity argument: oe_ins >= e_ins makes the `f - oe_ins` branch inside H
                    // dominated, so `max(f - e_ins, mfe - oe_ins)` equals the original).
                    f_v = _mm_max_epi8(
                        _mm_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);
                    if j & 15 == 15 {
                        _mm_storeu_si128(ck_p.add((j >> 4) * 2 * LANES16) as *mut __m128i, imax_v);
                    }
                }
                let need = finish_row!(i, imax_v);
                resolve_qe!(i, h_cur.as_ptr(), 0, need);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once every live lane is either frozen or out of target: purely a speed win,
            // as in the NEON kernels. See the longer note in `fwd_local_sw_neon`.
            if (0..n_lanes).all(|l| frozen[l] || i + rows >= tlen[l]) {
                break;
            }
            i += rows;
        }

        extract_group(
            n_lanes, group_idx, LANES16, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// [`fwd_local_sw_sse41_u8_impl`] in the form `mat` takes: the standard XOR table, or `MAT5`.
///
/// # Safety
/// As the impl; a non-standard `mat` must pass `mat5_u8_ok`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_sse41_u8(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    // SAFETY: the caller's.
    unsafe {
        let share = o_del + e_del == o_ins + e_ins;
        match (mat_is_standard(m, mat), share) {
            (true, false) => fwd_local_sw_sse41_u8_impl::<false, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (true, true) => fwd_local_sw_sse41_u8_impl::<false, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, false) => fwd_local_sw_sse41_u8_impl::<true, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, true) => fwd_local_sw_sse41_u8_impl::<true, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
        }
    }
}
/// SSE4.1 `i16x8` forward local-SW: [`fwd_local_sw_neon`] transliterated to `__m128i`, so a batch whose
/// score ceiling overflows u8 runs vectorised on x86_64 instead of falling through to
/// [`fwd_local_sw_scalar`]. This is the second half of closing the x86 rescue gap (#12/#20): the u8
/// kernel above already handles stock DNA scores, this one the high-scoring configs (large `a`, or the
/// NEON gate's `(10, 40)` matrix) where `min(len) * match` exceeds 250.
///
/// The transliteration is mechanical, exactly as for [`fwd_local_sw_sse41_u8`], because every operation
/// is element-wise and i16 signed arithmetic matches NEON's `int16x8` byte-for-byte. Two NEON spellings
/// have no one-instruction SSE4.1 form and are rebuilt here:
///
/// - **`vcgeq_s16(a, b)`** (signed `>=`): recovered as `max_epi16(a, b) == a`. Every value compared
///   this way is a base code (0..5) or [`PAD`] widened *unsigned* to a small positive i16, so the
///   signed max is exact.
/// - **`vbslq_s16(mask, a, b)`** = `mask ? a : b`: `_mm_blendv_epi8(a, b, mask)` is `mask ? b : a`,
///   so the operands swap, same as the u8 kernel's `bsl`. The per-word masks are all-ones/all-zero, so
///   consulting the top bit of each byte is exact.
///
/// Unlike NEON's byte loads, widening an 8-code column to i16 is one `_mm_loadl_epi64` (8 bytes) then
/// `_mm_cvtepu8_epi16`, the unsigned widen that keeps [`PAD`] = 255 as +255 rather than -1.
///
/// # Parameters / Returns
/// As [`fwd_local_sw_batch`], with the same unchecked preconditions as [`fwd_local_sw_neon`]:
/// `mat_is_standard(m, mat)`, and every job's score ceiling `min(qlen, tlen) * max_sc` plus target
/// length under [`I16_SCORE_LIMIT`]. Byte-identical to [`fwd_local_sw_scalar`]
/// (`sse41_verify::sse41_matesw_i16_matches_scalar`).
///
/// # Safety
/// Caller must have confirmed SSE4.1 is available. Loads/stores use unchecked offsets bounded by
/// `qmax`/`tmax`, derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_sse41_i16<const MAT5: bool>(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    // `mask ? a : b`, per 16-bit lane; the all-ones/all-zero masks make the byte-granular blend exact.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn bsl(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_blendv_epi8(b, a, mask)
    }
    // Signed `a >= b`, all-ones/all-zero per word. `max_epi16(a,b) == a` iff `a >= b`; used only on the
    // small positive code/PAD values, where signed and unsigned max coincide.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn cge_epi16(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi16(_mm_max_epi16(a, b), a)
    }

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // The standard matrix collapses to `mtch` (positive match) and `mis` (signed mismatch, e.g. -4),
    // plus the fixed -1 for N, exactly as in the NEON i16 kernel.
    let mtch = mat[0] as i16;
    let mis = mat[1] as i16;
    // (score, te, qe, score2, te2) seeded with ksw's `g_defr`: score2 defaults to -1, not 0.
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    let zero = _mm_setzero_si128();
    // Increment for the carried column counter `j_v` (see the fast column range below).
    let one_v = _mm_set1_epi16(1);
    let mtch_v = _mm_set1_epi16(mtch);
    let mis_v = _mm_set1_epi16(mis);
    let n_v = _mm_set1_epi16(-1);
    let dead_v = _mm_set1_epi16(DEAD_CELL_SCORE as i16);
    let four_v = _mm_set1_epi16(4);
    // `MAT5` codes run to 15 (`mat5_add_code`), so the dead tests move above them.
    let m_v = _mm_set1_epi16(if MAT5 { 16 } else { m as i16 });
    let zpad_v = _mm_set1_epi16(ZPAD as i16);
    let qdead_v = _mm_set1_epi16(if MAT5 { 15 } else { ZPAD as i16 });
    // `MAT5` (a `--meth` matrix): the cell score is one `pshufb` at `(t + q) & 15` on the encoded
    // byte codes, as in the u8 kernels; ZPAD (5, never an encoded query code) and PAD keep the
    // tail's selects.
    let (tbl, enc_t, enc_q) = if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        (_mm_loadu_si128(t16.as_ptr() as *const __m128i), t, q)
    } else {
        (_mm_setzero_si128(), [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    };
    let e_del_v = _mm_set1_epi16(e_del as i16);
    let oe_del_v = _mm_set1_epi16(oe_del as i16);
    let e_ins_v = _mm_set1_epi16(e_ins as i16);
    let oe_ins_v = _mm_set1_epi16(oe_ins as i16);

    for (group_idx, group) in jobs.chunks(LANES).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES],
            [0usize; LANES],
            [i32::MAX; LANES],
            [i32::MAX; LANES],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES + l] = if MAT5 { enc_q[b as usize] } else { b };
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = if MAT5 { enc_t[b as usize] } else { b };
            }
        }

        // i16 SoA DP state, one `__m128i` per column across the 16 lanes. `rowmax` stays i32; it is only
        // read by the scalar bookkeeping. Plain wrapping i16 arithmetic is safe because the caller
        // proved every ceiling is under `I16_SCORE_LIMIT` and the kill score is only `DEAD_CELL_SCORE`.
        let mut h_prev = vec![0i16; qmax * LANES];
        let mut h_cur = vec![0i16; qmax * LANES];
        let mut e = vec![0i16; qmax * LANES];
        let mut rowmax = vec![0i32; tmax * LANES];
        let mut gmax = [0i32; LANES];
        let mut te = [-1i32; LANES];
        let mut qe = [0i32; LANES];
        let mut limit = [-1i32; LANES];
        let mut frozen = [false; LANES];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // Widen 16 u8 codes at `off` into an `i16x8`. `_mm_cvtepu8_epi16` is the unsigned widen, so
        // PAD (255) stays +255 and not -1, matching NEON's `vmovl_u8`.
        let load_codes = |buf: &[u8], off: usize| -> __m128i {
            _mm_cvtepu8_epi16(_mm_loadl_epi64(buf.as_ptr().add(off) as *const __m128i))
        };
        // `MAT5`'s cell score at column `j`, from the row's byte codes `t_b`.
        let mat5_sc = |t_b: __m128i, j: usize| -> __m128i {
            let q_b = _mm_loadl_epi64(seq_q.as_ptr().add(j * LANES) as *const __m128i);
            _mm_cvtepi8_epi16(_mm_shuffle_epi8(tbl, _mm_add_epi8(t_b, q_b)))
        };

        // =====================================================================================
        // Main DP, one target row per iteration. Structurally identical to `fwd_local_sw_neon`, at
        // 8 lanes; only the per-cell arithmetic is vectorised, the bookkeeping stays scalar.
        // =====================================================================================
        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES);
            let t_b = _mm_loadl_epi64(seq_t.as_ptr().add(i * LANES) as *const __m128i);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
                                       // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm_cmpeq_epi16(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    let q_v = load_codes(&seq_q, j * LANES);
                    let eq = _mm_cmpeq_epi16(t_v, q_v);
                    let n_mask = _mm_or_si128(t_is_n, _mm_cmpeq_epi16(q_v, four_v));
                    bsl(n_mask, n_v, bsl(eq, mtch_v, mis_v))
                };
                let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES) as *const __m128i);
                let mut h_v = _mm_add_epi16(h_diag_v, sc);
                h_v = _mm_max_epi16(h_v, zero);
                h_v = _mm_max_epi16(h_v, e_v);
                let mfe = h_v;
                h_v = _mm_max_epi16(h_v, f_v);
                let is_new_row_max = _mm_cmpgt_epi16(h_v, imax_v);
                imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                imax_v = _mm_max_epi16(imax_v, h_v);
                _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES) as *mut __m128i, h_v);
                let e_new =
                    _mm_max_epi16(_mm_sub_epi16(e_v, e_del_v), _mm_sub_epi16(h_v, oe_del_v));
                _mm_storeu_si128(
                    e.as_mut_ptr().add(j * LANES) as *mut __m128i,
                    _mm_max_epi16(e_new, zero),
                );
                let f_new =
                    _mm_max_epi16(_mm_sub_epi16(f_v, e_ins_v), _mm_sub_epi16(mfe, oe_ins_v));
                f_v = _mm_max_epi16(f_new, zero);
                h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES) as *const __m128i);
                j_v = _mm_add_epi16(j_v, one_v);
            }

            // Tail: the columns where ZPAD / PAD can appear, full logic.
            for j in n_fast..qmax {
                let q_v = load_codes(&seq_q, j * LANES);
                // Four masks, four selects in increasing priority, exactly as the NEON i16 kernel:
                //   eq        the bases match; n_mask either is N; zpad_mask query is profile padding;
                //   pad_mask  cell is past a real position (dead target row, or query past the profile).
                let eq = _mm_cmpeq_epi16(t_v, q_v);
                let n_mask = _mm_or_si128(t_is_n, _mm_cmpeq_epi16(q_v, four_v));
                let zpad_mask = _mm_cmpeq_epi16(q_v, zpad_v);
                let pad_mask = _mm_or_si128(cge_epi16(t_v, m_v), _mm_cmpgt_epi16(q_v, qdead_v));
                let mut sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    bsl(n_mask, n_v, bsl(eq, mtch_v, mis_v))
                };
                sc = bsl(zpad_mask, zero, sc); // ksw profile padding scores 0
                sc = bsl(pad_mask, dead_v, sc); // dead padding: kill the cell outright

                let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES) as *const __m128i);
                // H = max(0, H_diag + S, E, F); wrapping adds are in range by the ceiling guarantee.
                let mut h_v = _mm_add_epi16(h_diag_v, sc);
                h_v = _mm_max_epi16(h_v, zero);
                h_v = _mm_max_epi16(h_v, e_v);
                // `mfe = max(0, H_diag + S, E)` excludes the serial F carry, so the F recurrence below
                // reads it and the critical column chain stays f -> f (see the NEON i16 kernel).
                let mfe = h_v;
                h_v = _mm_max_epi16(h_v, f_v);
                // Strict signed `>`, so a tie keeps the earlier column.
                let is_new_row_max = _mm_cmpgt_epi16(h_v, imax_v);
                imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                imax_v = _mm_max_epi16(imax_v, h_v);
                _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES) as *mut __m128i, h_v);

                // E(i+1,j) = max(0, E - e_del, H - oe_del).
                let e_new =
                    _mm_max_epi16(_mm_sub_epi16(e_v, e_del_v), _mm_sub_epi16(h_v, oe_del_v));
                _mm_storeu_si128(
                    e.as_mut_ptr().add(j * LANES) as *mut __m128i,
                    _mm_max_epi16(e_new, zero),
                );
                // F(i,j+1) = max(0, F - e_ins, mfe - oe_ins), reassociated off `mfe`.
                let f_new =
                    _mm_max_epi16(_mm_sub_epi16(f_v, e_ins_v), _mm_sub_epi16(mfe, oe_ins_v));
                f_v = _mm_max_epi16(f_new, zero);
                h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES) as *const __m128i);
                j_v = _mm_add_epi16(j_v, one_v);
            }

            let mut imax_arr = [0i16; LANES];
            let mut col_arr = [0i16; LANES];
            _mm_storeu_si128(imax_arr.as_mut_ptr() as *mut __m128i, imax_v);
            _mm_storeu_si128(col_arr.as_mut_ptr() as *mut __m128i, imax_col_v);
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                let row_max = imax_arr[l] as i32;
                rowmax[i * LANES + l] = row_max;
                if row_max > gmax[l] {
                    gmax[l] = row_max;
                    te[l] = i as i32;
                    qe[l] = col_arr[l] as i32;
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            if (0..n_lanes).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n_lanes, group_idx, LANES, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}
/// AVX512 `u8x64` forward local-SW: [`fwd_local_sw_avx2_u8`] widened to `__m512i`, 64 rescue jobs per
/// group instead of 32. Mate-rescue jobs are cross-lane independent and near-uniform in size (query =
/// read length, target = the insert-size window), so doubling the lane count keeps utilisation high;
/// the kernel is throughput-bound after the F reassociation, which is the regime a wider vector helps.
///
/// AVX-512 changes the *spelling* of two things relative to AVX2, and both make the code simpler, not
/// harder:
///
/// - **Comparisons return mask registers** (`__mmask64`), not vector masks.
/// - **Blends take the mask directly**: `_mm512_mask_blend_epi8(k, a, b)` is `k ? b : a`, so the NEON
///   `vbslq(mask, a, b) = mask ? a : b` becomes `_mm512_mask_blend_epi8(mask, b, a)` (operands swapped,
///   same convention as the AVX2 `bsl`).
///
/// Everything else is identical to the AVX2 u8 kernel: saturating byte arithmetic supplies the
/// `max(0, .)` clamps, scores live in the same signed `H - 128` domain (one `adds_epi8` per diagonal,
/// the `SHARE` F open), the F recurrence is reassociated off `mfe`, and the argmax column rides in
/// the same byte lane as the scores (so the caller's 250-base query cap still holds at 64 lanes).
///
/// # Parameters / Returns
/// As [`fwd_local_sw_batch`], same preconditions as [`fwd_local_sw_avx2_u8`]. Byte-identical to
/// [`fwd_local_sw_scalar`] (`avx512_verify::avx512_matesw_u8_matches_scalar`, which runs only where
/// `avx512bw` is present).
///
/// # Safety
/// Caller must have confirmed AVX-512BW is available. Loads/stores use unchecked offsets bounded by
/// `qmax`/`tmax`, derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx512_u8_impl<const MAT5: bool, const SHARE: bool, const VBMI: bool>(
    jobs: &[FwdJob],
    // Matrix dimension (5). No longer read: dead cells are detected by the PAD byte's high
    // bit, as in every other u8 kernel here. Kept for signature parity.
    _m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as u8;
    let mispen = (-mat[1]) as u8;
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    let zero = _mm512_setzero_si512();
    // The DP's 0 in the signed H domain (H - 128): saturation at -128 is the clamp at 0, and
    // every max, subtract and compare has its exact signed twin (the fork's `Sgn` u8 body).
    let hz = _mm512_set1_epi8(-128);
    // ISSUE #44 part A: the same XOR-indexed score table as `fwd_local_sw_neon_u8` and, since #43,
    // as the AVX2 and SSE4.1 kernels. `vpshufb` is IN-LANE, so the 16-byte table is broadcast to all
    // four 128-bit lanes; `vpermb` was considered and rejected, a 16-entry table does not need a
    // 64-entry lookup and `vpermb` costs more.
    //
    // Why it pays here specifically: on Golden Cove every 512-bit saturating-integer and max op is
    // p0-only at throughput 1.00, because at 512 bits port 1's vector ALU folds into p0. The kernel's
    // cost is then literally the count of p0-only ops, and this rewrite deletes two of them per row
    // (the N-penalty `subs` and the `korq`, which is p0 on SKX/ICL/EMR) while moving work onto ports
    // that sit idle: `vpxord` is p05 and `vpshufb` ZMM is p5.
    assert!(
        (U8_SCORE_LIMIT as u32) + mispen as u32 + mtch as u32 <= 256,
        "u8 rescue score table would saturate: mispen {mispen} + mtch {mtch} + ceiling too large"
    );
    let bias = mispen;
    let mut tbl = [0u8; 16];
    tbl[0] = bias + mtch; // match: +a
                          // tbl[1..=3] = bias - b = 0 (mismatch), already zero
    for k in 4..8 {
        tbl[k] = bias - 1; // query N against a real base: bwa scores -1
    }
    tbl[8] = bias - 1; // both N: N_TARGET XOR 4
    for k in 12..16 {
        tbl[k] = bias - 1; // target N against a real base
    }
    // H is held as H - 128 (signed), so the table holds the unbiased deltas.
    for v in &mut tbl {
        *v = v.wrapping_sub(bias);
    }
    let score_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tbl.as_ptr() as *const __m128i));
    let bias_v = _mm512_set1_epi8(bias as i8);
    // `MAT5` (`--meth`'s matrices, `mat5_u8_ok`): with VBMI the whole 5x5 matrix is one 64-byte
    // `vpermb` table at `5 t + q` (`seq_t` holds `5 t`, N as 20); without it, one `vpshufb` at
    // `(t + q) & 15` through `mat5_add_code`, as in the AVX2 kernel. Every cell lies in
    // `[mat[1], mat[0]]`, a signed byte added straight to the `H - 128` domain.
    let (score_tbl, enc_t, enc_q) = if MAT5 && VBMI {
        let mut t64 = [0u8; 64];
        for (e, &v) in t64.iter_mut().zip(&mat[..25]) {
            *e = v as u8;
        }
        let t64 = _mm512_loadu_si512(t64.as_ptr() as *const __m512i);
        (t64, [0, 5, 10, 15, 20], [0, 1, 2, 3, 4])
    } else if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        let lo = _mm_loadu_si128(t16.as_ptr() as *const __m128i);
        (_mm512_broadcast_i32x4(lo), t, q)
    } else {
        (score_tbl, [0, 1, 2, 3, N_TARGET], [0, 1, 2, 3, 4])
    };
    // The `vpshufb` / `vqtbl1q_u8` reachable-index argument is the one spelled out in
    // `fwd_local_sw_avx2_u8`; the encoding and therefore the argument are the same.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq")]
    #[inline]
    unsafe fn score_diag<const M5: bool, const VB: bool>(
        tbl: __m512i,
        _bias_v: __m512i,
        t_v: __m512i,
        q_v: __m512i,
        d: __m512i,
    ) -> __m512i {
        // `M5`: `i = 5 t + q <= 24` (VBMI) or the add code's `t + q < 32` in every cell a live lane
        // keeps. Plain (wrapping) adds, not saturating ones, which are p0-only at 512 bits on Golden
        // Cove: a `PAD` row is past the lane's target (`finish_row` and `extract_group` stop at
        // `tlen`) and a `ZPAD` / `PAD` column is overridden by the pad blend, so what they read
        // does not matter.
        let s = if M5 && VB {
            // `vpermb` reads `i & 63`.
            let i = _mm512_add_epi8(t_v, q_v);
            let s: __m512i;
            // SAFETY: the caller dispatched `VBMI` only where `avx512vbmi` was detected. Inline
            // asm because `target_feature` is per function: enabling VBMI on this impl would let
            // its non-VBMI instantiations use it too.
            std::arch::asm!("vpermb {s}, {i}, {t}", s = lateout(zmm_reg) s, i = in(zmm_reg) i,
                 t = in(zmm_reg) tbl, options(pure, nomem, nostack));
            s
        } else if M5 {
            _mm512_shuffle_epi8(tbl, _mm512_add_epi8(t_v, q_v))
        } else {
            _mm512_shuffle_epi8(tbl, _mm512_xor_si512(t_v, q_v))
        };
        _mm512_adds_epi8(d, s)
    }
    let zpad_v = _mm512_set1_epi8(ZPAD as i8);
    let e_del_v = _mm512_set1_epi8(e_del as i8);
    let oe_del_v = _mm512_set1_epi8(oe_del as i8);
    let e_ins_v = _mm512_set1_epi8(e_ins as i8);
    let oe_ins_v = _mm512_set1_epi8(oe_ins as i8);

    for (group_idx, group) in jobs.chunks(LANES64).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // The N re-encoding (or `MAT5`'s codes), applied once per base so the column loops never
        // pay for it (issues #43, #44).
        let (seq_q, seq_t) = soa_group(
            group,
            LANES64,
            qmax,
            tmax,
            max_sc,
            |b| enc_t[b as usize],
            |b| enc_q[b as usize],
        );
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES64],
            [0usize; LANES64],
            [i32::MAX; LANES64],
            [i32::MAX; LANES64],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
        }

        let mut h_prev = vec![0x80u8; qmax * LANES64];
        let mut h_cur = vec![0x80u8; qmax * LANES64];
        let mut e = vec![0x80u8; qmax * LANES64];
        // The running row max after every `QE_BLK` columns, per row of a pair (`finish_row`).
        let mut blk0 = vec![0u8; qmax.div_ceil(QE_BLK) * LANES64];
        let mut blk1 = vec![0u8; qmax.div_ceil(QE_BLK) * LANES64];
        // ISSUE #46C. `rowmax` is u8, not i32, and the row is published with ONE vector store
        // instead of `LANES` guarded scalar ones, exactly as `fwd_local_sw_neon_u8` already does.
        // Two reasons, both structural rather than micro:
        //
        //  - size. At AVX-512's mean shape the i32 buffer was 359 KB per group, out of L1 and into
        //    L2, and `extract_group`'s per-lane walk then touched a fresh cache line for every load,
        //    re-fetching each line once per lane. As u8 it is 90 KB and the stride is exactly one
        //    64-byte line per row of 64 lanes.
        //  - stores. The scalar form issued one guarded store per live lane per row, 32 or 64 of
        //    them; the vector form issues one.
        //
        // Writing it unconditionally is safe for the same reason it is on NEON: a lane's row maxima
        // are read only for rows `0..=limit[l]`, and `limit[l]` is `tlen[l] - 1` or the freeze row,
        // so every slot this writes beyond the old guard is one nothing reads. Values fit u8 because
        // the caller proved the score ceiling is under `U8_SCORE_LIMIT`.
        let mut rowmax = vec![0u8; tmax * LANES64];
        let mut gmax = [0i32; LANES64];
        // `te` as four vectors of 16 i32 rows and `qe` as one of 64 columns, blended under the
        // row's improving lanes instead of written lane by lane.
        let mut te_v = [_mm512_set1_epi32(-1); 4];
        let mut qe_v = zero;
        let mut limit = [-1i32; LANES64];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // The row epilogue's per-lane state as one vector and one mask, so a row costs a few mask
        // ops instead of a branchy pass over 64 lanes (bwa-mem3's vector epilogue): `gmax_v` is the
        // best row max so far (unsigned), `frozen` the lanes that reached `endsc`, `endsc_v` that
        // threshold clamped to a byte (scores stay below `U8_SCORE_LIMIT`, so 255 is never reached).
        let mut gmax_v = zero;
        let mut frozen = 0u64;
        let endsc_v = {
            let mut a = [u8::MAX; LANES64];
            for l in 0..n_lanes {
                a[l] = endsc[l].clamp(0, 255) as u8;
            }
            _mm512_loadu_si512(a.as_ptr() as *const __m512i)
        };
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };
        // From here every live lane holds ZPAD (the ksw profile padding): `diag = d`, no score and
        // no pad blend. Same third regime, and the same argument, as `fwd_local_sw_neon_u8`'s.
        let n_pad = if zpadcol_enabled()
            && group
                .iter()
                .all(|j| ksw_padded_qlen(j.query.len(), max_sc) == qmax)
        {
            qlen[..n_lanes]
                .iter()
                .copied()
                .max()
                .unwrap_or(qmax)
                .max(n_fast)
                .min(qmax)
        } else {
            qmax
        };

        // Two target rows per iteration when two are left; see `fwd_local_sw_neon_u8` for the
        // memory-traffic argument and the byte-identity argument.
        // The row epilogue, in two halves. `finish_row` folds the row into `rowmax`, `gmax`, `te`
        // and the freeze, and returns the lanes it improves; `resolve_qe` recovers their query end.
        // A pair resolves row `i` only for the lanes row `i+1` does not improve again: the lane's
        // `qe` is only ever read at its last improving row, so the skipped rescans are never
        // observed, and along an alignment that grows row by row they are most of them.
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $t:expr) => {{
                let row = $row;
                let imax_u = _mm512_xor_si512($imax, hz);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm512_storeu_si512(
                    rowmax.as_mut_ptr().add(row * LANES64) as *mut __m512i,
                    imax_u,
                );
                // The lanes this row improves are the only ones that need its query end. A lane
                // is in its target while the row's base is not `PAD` (bit 7), which also leaves
                // out the lanes past `n_lanes`.
                let need =
                    !_mm512_movepi8_mask($t) & !frozen & _mm512_cmpgt_epu8_mask(imax_u, gmax_v);
                if need != 0 {
                    gmax_v = _mm512_mask_blend_epi8(need, gmax_v, imax_u);
                    let froze = _mm512_mask_cmpge_epu8_mask(need, gmax_v, endsc_v);
                    frozen |= froze;
                    let row_v = _mm512_set1_epi32(row as i32);
                    for (k, t) in te_v.iter_mut().enumerate() {
                        *t = _mm512_mask_blend_epi32((need >> (16 * k)) as u16, *t, row_v);
                    }
                    let mut m = froze;
                    while m != 0 {
                        let l = m.trailing_zeros() as usize;
                        m &= m - 1;
                        limit[l] = row as i32;
                    }
                }
                need
            }};
        }
        macro_rules! resolve_qe {
            ($imax:expr, $hrow:expr, $blk:expr, $need:expr) => {{
                let need: u64 = $need;
                if need != 0 {
                    // The query end, recovered lazily (bwa-mem3's LazyQE): the smallest column
                    // whose stored H equals the row max, the column the inline strict `>` kept.
                    // The checkpoints are prefix maxima, so a lane's first block whose checkpoint
                    // reaches the max holds that column, and scanning the block last to first,
                    // the last select is the smallest one.
                    let (hrow, blk): (*const u8, *const u8) = ($hrow, $blk);
                    let mut pending = need;
                    let mut col = zero;
                    for b in 0..qmax.div_ceil(QE_BLK) {
                        let ck = _mm512_loadu_si512(blk.add(b * LANES64) as *const __m512i);
                        let hit = _mm512_mask_cmpeq_epi8_mask(pending, ck, $imax);
                        if hit == 0 {
                            continue;
                        }
                        for j in (b * QE_BLK..(b * QE_BLK + QE_BLK).min(qmax)).rev() {
                            let h = _mm512_loadu_si512(hrow.add(j * LANES64) as *const __m512i);
                            let eq = _mm512_mask_cmpeq_epi8_mask(hit, h, $imax);
                            col = _mm512_mask_blend_epi8(eq, col, _mm512_set1_epi8(j as i8));
                        }
                        pending &= !hit;
                        if pending == 0 {
                            break;
                        }
                    }
                    qe_v = _mm512_mask_blend_epi8(need, qe_v, col);
                }
            }};
        }

        let pair_rows = rowpair_enabled();
        let mut i = 0usize;
        while i < tmax {
            let rows = if pair_rows && i + 1 < tmax { 2 } else { 1 };
            if rows == 2 {
                let t0_v = _mm512_loadu_si512(seq_t.as_ptr().add(i * LANES64) as *const __m512i);
                let t1_v =
                    _mm512_loadu_si512(seq_t.as_ptr().add((i + 1) * LANES64) as *const __m512i);
                let (mut f0, mut f1) = (hz, hz);
                let (mut d0, mut d1) = (hz, hz);
                let (mut imax0, mut imax1) = (hz, hz);

                for j in 0..n_fast {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);

                    let diag0 = score_diag::<MAT5, VBMI>(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe0 = _mm512_max_epi8(diag0, e_v);
                    let h0 = _mm512_max_epi8(mfe0, f0);
                    imax0 = _mm512_max_epi8(imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid = _mm512_max_epi8(
                        _mm512_subs_epi8(e_v, e_del_v),
                        _mm512_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm512_max_epi8(
                        _mm512_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    // Row i's H, into the slot its diagonal just left (read by `finish_row`).
                    _mm512_storeu_si512(h_prev.as_mut_ptr().add(j * LANES64) as *mut __m512i, h0);

                    let diag1 = score_diag::<MAT5, VBMI>(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm512_max_epi8(diag1, e_mid);
                    let h1 = _mm512_max_epi8(mfe1, f1);
                    imax1 = _mm512_max_epi8(imax1, h1);
                    _mm512_storeu_si512(
                        e.as_mut_ptr().add(j * LANES64) as *mut __m512i,
                        _mm512_max_epi8(
                            _mm512_subs_epi8(e_mid, e_del_v),
                            _mm512_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm512_max_epi8(
                        _mm512_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h1);
                    if j % QE_BLK == QE_BLK - 1 {
                        _mm512_storeu_si512(
                            blk0.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax0,
                        );
                        _mm512_storeu_si512(
                            blk1.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax1,
                        );
                    }
                }

                for j in n_fast..n_pad {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);
                    let zpad_mask = _mm512_cmpeq_epi8_mask(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask.
                    let q_pad = _mm512_movepi8_mask(q_v);

                    let mut diag0 = score_diag::<MAT5, VBMI>(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = _mm512_mask_blend_epi8(zpad_mask, diag0, d0);
                    diag0 = _mm512_mask_blend_epi8(_mm512_movepi8_mask(t0_v) | q_pad, diag0, hz);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe0 = _mm512_max_epi8(diag0, e_v);
                    let h0 = _mm512_max_epi8(mfe0, f0);
                    imax0 = _mm512_max_epi8(imax0, h0);
                    let e_mid = _mm512_max_epi8(
                        _mm512_subs_epi8(e_v, e_del_v),
                        _mm512_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm512_max_epi8(
                        _mm512_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    // Row i's H, into the slot its diagonal just left (read by `finish_row`).
                    _mm512_storeu_si512(h_prev.as_mut_ptr().add(j * LANES64) as *mut __m512i, h0);

                    let mut diag1 = score_diag::<MAT5, VBMI>(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = _mm512_mask_blend_epi8(zpad_mask, diag1, d1);
                    diag1 = _mm512_mask_blend_epi8(_mm512_movepi8_mask(t1_v) | q_pad, diag1, hz);
                    let mfe1 = _mm512_max_epi8(diag1, e_mid);
                    let h1 = _mm512_max_epi8(mfe1, f1);
                    imax1 = _mm512_max_epi8(imax1, h1);
                    _mm512_storeu_si512(
                        e.as_mut_ptr().add(j * LANES64) as *mut __m512i,
                        _mm512_max_epi8(
                            _mm512_subs_epi8(e_mid, e_del_v),
                            _mm512_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm512_max_epi8(
                        _mm512_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h1);
                    if j % QE_BLK == QE_BLK - 1 {
                        _mm512_storeu_si512(
                            blk0.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax0,
                        );
                        _mm512_storeu_si512(
                            blk1.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax1,
                        );
                    }
                }

                // Every live lane is ZPAD: `diag = d`, so no query load, score or pad blend. A lane
                // past its target is left un-killed, as in the fast body: `finish_row` skips it.
                for j in n_pad..qmax {
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe0 = _mm512_max_epi8(d0, e_v);
                    let h0 = _mm512_max_epi8(mfe0, f0);
                    imax0 = _mm512_max_epi8(imax0, h0);
                    let e_mid = _mm512_max_epi8(
                        _mm512_subs_epi8(e_v, e_del_v),
                        _mm512_subs_epi8(mfe0, oe_del_v),
                    );
                    f0 = _mm512_max_epi8(
                        _mm512_subs_epi8(f0, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe0, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe0, oe_ins_v)
                        },
                    );
                    d0 = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    _mm512_storeu_si512(h_prev.as_mut_ptr().add(j * LANES64) as *mut __m512i, h0);

                    let mfe1 = _mm512_max_epi8(d1, e_mid);
                    let h1 = _mm512_max_epi8(mfe1, f1);
                    imax1 = _mm512_max_epi8(imax1, h1);
                    _mm512_storeu_si512(
                        e.as_mut_ptr().add(j * LANES64) as *mut __m512i,
                        _mm512_max_epi8(
                            _mm512_subs_epi8(e_mid, e_del_v),
                            _mm512_subs_epi8(mfe1, oe_del_v),
                        ),
                    );
                    f1 = _mm512_max_epi8(
                        _mm512_subs_epi8(f1, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe1, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe1, oe_ins_v)
                        },
                    );
                    d1 = h0;
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h1);
                    if j % QE_BLK == QE_BLK - 1 {
                        _mm512_storeu_si512(
                            blk0.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax0,
                        );
                        _mm512_storeu_si512(
                            blk1.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax1,
                        );
                    }
                }

                _mm512_storeu_si512(
                    blk0.as_mut_ptr().add((qmax - 1) / QE_BLK * LANES64) as *mut __m512i,
                    imax0,
                );
                _mm512_storeu_si512(
                    blk1.as_mut_ptr().add((qmax - 1) / QE_BLK * LANES64) as *mut __m512i,
                    imax1,
                );
                // Row order matters: freezing at row `i` must suppress row `i+1`.
                let need0 = finish_row!(i, imax0, t0_v);
                let need1 = finish_row!(i + 1, imax1, t1_v);
                resolve_qe!(imax0, h_prev.as_ptr(), blk0.as_ptr(), need0 & !need1);
                resolve_qe!(imax1, h_cur.as_ptr(), blk1.as_ptr(), need1);
            } else {
                let t_v = _mm512_loadu_si512(seq_t.as_ptr().add(i * LANES64) as *const __m512i);
                let mut f_v = hz;
                let mut h_diag_v = hz;
                let mut imax_v = hz;

                // Padding-free column range, same argument as every other kernel here.
                for j in 0..n_fast {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);
                    let diag_v = score_diag::<MAT5, VBMI>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe = _mm512_max_epi8(diag_v, e_v);
                    let h_v = _mm512_max_epi8(mfe, f_v);
                    imax_v = _mm512_max_epi8(imax_v, h_v);
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h_v);
                    let e_new = _mm512_max_epi8(
                        _mm512_subs_epi8(e_v, e_del_v),
                        _mm512_subs_epi8(mfe, oe_del_v),
                    );
                    _mm512_storeu_si512(e.as_mut_ptr().add(j * LANES64) as *mut __m512i, e_new);
                    f_v = _mm512_max_epi8(
                        _mm512_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v =
                        _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    if j % QE_BLK == QE_BLK - 1 {
                        _mm512_storeu_si512(
                            blk0.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax_v,
                        );
                    }
                }

                // Tail: the columns where ZPAD / PAD can appear, full logic.
                for j in n_fast..qmax {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);
                    // Same four masks as the AVX2 u8 kernel, now as `__mmask64`. Unsigned `>=` / `>` are
                    // native here, so no `max_epu8`-based recovery and no 127 hazard.
                    let zpad_mask = _mm512_cmpeq_epi8_mask(q_v, zpad_v);
                    // `PAD` is the only alphabet byte with bit 7 set, so the high-bit test replaces both
                    // `t >= m` (which would now kill every real N target, re-encoded as 12) and
                    // `q > ZPAD`.
                    let pad_mask = _mm512_movepi8_mask(t_v) | _mm512_movepi8_mask(q_v);
                    let mut diag_v =
                        score_diag::<MAT5, VBMI>(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = _mm512_mask_blend_epi8(zpad_mask, diag_v, h_diag_v);
                    diag_v = _mm512_mask_blend_epi8(pad_mask, diag_v, hz);

                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe = _mm512_max_epi8(diag_v, e_v);
                    let h_v = _mm512_max_epi8(mfe, f_v);
                    // Strict `>` on `H - 128`, so a tie keeps the earlier column.
                    imax_v = _mm512_max_epi8(imax_v, h_v);
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h_v);

                    let e_new = _mm512_max_epi8(
                        _mm512_subs_epi8(e_v, e_del_v),
                        _mm512_subs_epi8(mfe, oe_del_v),
                    );
                    _mm512_storeu_si512(e.as_mut_ptr().add(j * LANES64) as *mut __m512i, e_new);
                    f_v = _mm512_max_epi8(
                        _mm512_subs_epi8(f_v, e_ins_v),
                        if SHARE {
                            _mm512_subs_epi8(mfe, oe_del_v)
                        } else {
                            _mm512_subs_epi8(mfe, oe_ins_v)
                        },
                    );
                    h_diag_v =
                        _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    if j % QE_BLK == QE_BLK - 1 {
                        _mm512_storeu_si512(
                            blk0.as_mut_ptr().add(j / QE_BLK * LANES64) as *mut __m512i,
                            imax_v,
                        );
                    }
                }
                _mm512_storeu_si512(
                    blk0.as_mut_ptr().add((qmax - 1) / QE_BLK * LANES64) as *mut __m512i,
                    imax_v,
                );
                let need = finish_row!(i, imax_v, t_v);
                resolve_qe!(imax_v, h_cur.as_ptr(), blk0.as_ptr(), need);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Every lane frozen or out of target from the next row on: a target ends where its
            // bytes turn `PAD`, so the next row's bases say which lanes still run.
            i += rows;
            if i >= tmax
                || !_mm512_movepi8_mask(_mm512_loadu_si512(
                    seq_t.as_ptr().add(i * LANES64) as *const __m512i
                )) & !frozen
                    == 0
            {
                break;
            }
        }
        let mut gmax_arr = [0u8; LANES64];
        _mm512_storeu_si512(gmax_arr.as_mut_ptr() as *mut __m512i, gmax_v);
        for (g, &v) in gmax.iter_mut().zip(&gmax_arr) {
            *g = i32::from(v);
        }
        let mut te = [-1i32; LANES64];
        for (k, t) in te_v.iter().enumerate() {
            _mm512_storeu_si512(te.as_mut_ptr().add(16 * k) as *mut __m512i, *t);
        }
        let mut qe_arr = [0u8; LANES64];
        _mm512_storeu_si512(qe_arr.as_mut_ptr() as *mut __m512i, qe_v);
        let qe = qe_arr.map(i32::from);

        extract_group(
            n_lanes, group_idx, LANES64, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// [`fwd_local_sw_avx512_u8_impl`] in the form `mat` takes: the standard XOR table, or `MAT5`.
///
/// # Safety
/// As the impl; a non-standard `mat` must pass `mat5_u8_ok`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx512_u8(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    // SAFETY: the caller's.
    unsafe {
        let share = o_del + e_del == o_ins + e_ins;
        let vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
        match (mat_is_standard(m, mat), share, vbmi) {
            (true, false, _) => fwd_local_sw_avx512_u8_impl::<false, false, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (true, true, _) => fwd_local_sw_avx512_u8_impl::<false, true, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, false, false) => fwd_local_sw_avx512_u8_impl::<true, false, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, true, false) => fwd_local_sw_avx512_u8_impl::<true, true, false>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, false, true) => fwd_local_sw_avx512_u8_impl::<true, false, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
            (false, true, true) => fwd_local_sw_avx512_u8_impl::<true, true, true>(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
            ),
        }
    }
}

/// AVX512 `i16x32` forward local-SW: [`fwd_local_sw_avx2_i16`] widened to `__m512i`, 32 lanes. Selected
/// on x86_64 when `avx512bw` is present and the batch overflows u8. Same mask-register simplifications
/// as the AVX512 u8 kernel: `_mm512_cmp*_epi16_mask` yields `__mmask32`, `_mm512_mask_blend_epi16`
/// consumes it directly, so the AVX2 `blendv`/`max_epi16 == a` recoveries disappear.
///
/// # Parameters / Returns
/// As [`fwd_local_sw_batch`], same preconditions as [`fwd_local_sw_neon`]/[`fwd_local_sw_avx2_i16`].
/// Byte-identical to [`fwd_local_sw_scalar`] (`avx512_verify::avx512_matesw_i16_matches_scalar`).
///
/// # Safety
/// Caller must have confirmed AVX-512BW is available. Loads/stores use unchecked offsets bounded by
/// `qmax`/`tmax`, derived from the same buffers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx512_i16<const MAT5: bool>(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::x86_64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as i16;
    let mis = mat[1] as i16;
    let mut out = vec![(0i32, -1i32, -1i32, -1i32, -1i32); jobs.len()];

    let zero = _mm512_setzero_si512();
    // Increment for the carried column counter `j_v` (see the fast column range below).
    let one_v = _mm512_set1_epi16(1);
    let mtch_v = _mm512_set1_epi16(mtch);
    let mis_v = _mm512_set1_epi16(mis);
    let n_v = _mm512_set1_epi16(-1);
    let dead_v = _mm512_set1_epi16(DEAD_CELL_SCORE as i16);
    let four_v = _mm512_set1_epi16(4);
    // `MAT5` codes run to 15 (`mat5_add_code`), so the dead tests move above them.
    let m_v = _mm512_set1_epi16(if MAT5 { 16 } else { m as i16 });
    let zpad_v = _mm512_set1_epi16(ZPAD as i16);
    let qdead_v = _mm512_set1_epi16(if MAT5 { 15 } else { ZPAD as i16 });
    // `MAT5` (a `--meth` matrix): the cell score is one `pshufb` at `(t + q) & 15` on the encoded
    // byte codes, as in the u8 kernels; ZPAD (5, never an encoded query code) and PAD keep the
    // tail's selects.
    let (tbl, enc_t, enc_q) = if MAT5 {
        let (t, q, t16) = mat5_add_code(mat).expect("the dispatch checks mat5_add_code");
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(t16.as_ptr() as *const __m128i)),
            t,
            q,
        )
    } else {
        (_mm256_setzero_si256(), [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    };
    let e_del_v = _mm512_set1_epi16(e_del as i16);
    let oe_del_v = _mm512_set1_epi16(oe_del as i16);
    let e_ins_v = _mm512_set1_epi16(e_ins as i16);
    let oe_ins_v = _mm512_set1_epi16(oe_ins as i16);

    for (group_idx, group) in jobs.chunks(LANES32).enumerate() {
        let n_lanes = group.len();
        let qmax = group
            .iter()
            .map(|j| ksw_padded_qlen(j.query.len(), max_sc))
            .max()
            .unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES32];
        let mut seq_t = vec![PAD; tmax * LANES32];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES32],
            [0usize; LANES32],
            [i32::MAX; LANES32],
            [i32::MAX; LANES32],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES32 + l] = if MAT5 { enc_q[b as usize] } else { b };
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES32 + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES32 + l] = if MAT5 { enc_t[b as usize] } else { b };
            }
        }

        let mut h_prev = vec![0i16; qmax * LANES32];
        let mut h_cur = vec![0i16; qmax * LANES32];
        let mut e = vec![0i16; qmax * LANES32];
        let mut rowmax = vec![0i32; tmax * LANES32];
        let mut gmax = [0i32; LANES32];
        let mut te = [-1i32; LANES32];
        let mut qe = [0i32; LANES32];
        let mut limit = [-1i32; LANES32];
        let mut frozen = [false; LANES32];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // Widen 32 u8 codes at `off` into an `i16x32`. `_mm512_cvtepu8_epi16` is the unsigned widen, so
        // PAD (255) stays +255, matching NEON's `vmovl_u8`.
        let load_codes = |buf: &[u8], off: usize| -> __m512i {
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(buf.as_ptr().add(off) as *const __m256i))
        };
        // `MAT5`'s cell score at column `j`, from the row's byte codes `t_b`.
        let mat5_sc = |t_b: __m256i, j: usize| -> __m512i {
            let q_b = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
            _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(tbl, _mm256_add_epi8(t_b, q_b)))
        };

        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES32);
            let t_b = _mm256_loadu_si256(seq_t.as_ptr().add(i * LANES32) as *const __m256i);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero;
            // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm512_cmpeq_epi16_mask(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    let q_v = load_codes(&seq_q, j * LANES32);
                    let eq = _mm512_cmpeq_epi16_mask(t_v, q_v);
                    let n_mask = t_is_n | _mm512_cmpeq_epi16_mask(q_v, four_v);
                    _mm512_mask_blend_epi16(n_mask, _mm512_mask_blend_epi16(eq, mis_v, mtch_v), n_v)
                };
                let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES32) as *const __m512i);
                let mut h_v = _mm512_add_epi16(h_diag_v, sc);
                h_v = _mm512_max_epi16(h_v, zero);
                h_v = _mm512_max_epi16(h_v, e_v);
                let mfe = h_v;
                h_v = _mm512_max_epi16(h_v, f_v);
                let is_new_row_max = _mm512_cmpgt_epi16_mask(h_v, imax_v);
                imax_col_v = _mm512_mask_blend_epi16(is_new_row_max, imax_col_v, j_v);
                imax_v = _mm512_max_epi16(imax_v, h_v);
                _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m512i, h_v);
                let e_new = _mm512_max_epi16(
                    _mm512_sub_epi16(e_v, e_del_v),
                    _mm512_sub_epi16(h_v, oe_del_v),
                );
                _mm512_storeu_si512(
                    e.as_mut_ptr().add(j * LANES32) as *mut __m512i,
                    _mm512_max_epi16(e_new, zero),
                );
                let f_new = _mm512_max_epi16(
                    _mm512_sub_epi16(f_v, e_ins_v),
                    _mm512_sub_epi16(mfe, oe_ins_v),
                );
                f_v = _mm512_max_epi16(f_new, zero);
                h_diag_v = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES32) as *const __m512i);
                j_v = _mm512_add_epi16(j_v, one_v);
            }

            // Tail: the columns where ZPAD / PAD can appear, full logic.
            for j in n_fast..qmax {
                let q_v = load_codes(&seq_q, j * LANES32);
                let eq = _mm512_cmpeq_epi16_mask(t_v, q_v);
                let n_mask = t_is_n | _mm512_cmpeq_epi16_mask(q_v, four_v);
                let zpad_mask = _mm512_cmpeq_epi16_mask(q_v, zpad_v);
                let pad_mask =
                    _mm512_cmpge_epi16_mask(t_v, m_v) | _mm512_cmpgt_epi16_mask(q_v, qdead_v);
                // `mask ? a : b` = `mask_blend(mask, b, a)`, increasing priority.
                let mut sc = if MAT5 {
                    mat5_sc(t_b, j)
                } else {
                    _mm512_mask_blend_epi16(n_mask, _mm512_mask_blend_epi16(eq, mis_v, mtch_v), n_v)
                };
                sc = _mm512_mask_blend_epi16(zpad_mask, sc, zero);
                sc = _mm512_mask_blend_epi16(pad_mask, sc, dead_v);

                let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES32) as *const __m512i);
                let mut h_v = _mm512_add_epi16(h_diag_v, sc);
                h_v = _mm512_max_epi16(h_v, zero);
                h_v = _mm512_max_epi16(h_v, e_v);
                let mfe = h_v;
                h_v = _mm512_max_epi16(h_v, f_v);
                let is_new_row_max = _mm512_cmpgt_epi16_mask(h_v, imax_v);
                imax_col_v = _mm512_mask_blend_epi16(is_new_row_max, imax_col_v, j_v);
                imax_v = _mm512_max_epi16(imax_v, h_v);
                _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m512i, h_v);

                let e_new = _mm512_max_epi16(
                    _mm512_sub_epi16(e_v, e_del_v),
                    _mm512_sub_epi16(h_v, oe_del_v),
                );
                _mm512_storeu_si512(
                    e.as_mut_ptr().add(j * LANES32) as *mut __m512i,
                    _mm512_max_epi16(e_new, zero),
                );
                let f_new = _mm512_max_epi16(
                    _mm512_sub_epi16(f_v, e_ins_v),
                    _mm512_sub_epi16(mfe, oe_ins_v),
                );
                f_v = _mm512_max_epi16(f_new, zero);
                h_diag_v = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES32) as *const __m512i);
                j_v = _mm512_add_epi16(j_v, one_v);
            }

            let mut imax_arr = [0i16; LANES32];
            let mut col_arr = [0i16; LANES32];
            _mm512_storeu_si512(imax_arr.as_mut_ptr() as *mut __m512i, imax_v);
            _mm512_storeu_si512(col_arr.as_mut_ptr() as *mut __m512i, imax_col_v);
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                let row_max = imax_arr[l] as i32;
                rowmax[i * LANES32 + l] = row_max;
                if row_max > gmax[l] {
                    gmax[l] = row_max;
                    te[l] = i as i32;
                    qe[l] = col_arr[l] as i32;
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            if (0..n_lanes).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n_lanes, group_idx, LANES32, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
}

/// Force-run verification of the AVX2 mate-rescue kernel against `fwd_local_sw_scalar`, byte-for-byte.
///
/// On x86_64 the AVX2 path only *runs* when `is_x86_feature_detected!("avx2")`, which Rosetta reports
/// as `false` even though it *executes* AVX2 instructions. So this test calls the kernel directly,
/// bypassing detection, which is how the port is validated on this Apple-Silicon host via
/// `cargo test --target x86_64-apple-darwin`. On a native x86 runner (which has AVX2) it validates the
/// real dispatched path too. Requires an AVX2-capable executor.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_verify {
    use super::*;

    /// bwa 5x5 score matrix: match `a`, mismatch `-b`, N row/col `-1`. A copy of `tests::scmat`
    /// rather than a reference to it: sibling test modules cannot see each other's private items,
    /// and widening that one's visibility for a five-line helper is not worth it.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// Compare the 32-lane AVX2 u8 kernel against the 8-lane scalar lockstep on randomized
    /// mate-rescue-shaped jobs. Deliberately a *direct* kernel-vs-kernel comparison rather than a
    /// round trip through `batched_ksw_align2`: that would go through the dispatch, which Rosetta
    /// routes to the scalar path, and the test would pass without ever executing an AVX2 instruction.
    ///
    /// The differing group widths (32 vs 8) are part of what is being tested: a wider chunk raises
    /// `qmax`/`tmax` for the whole group, and every extra column and row must be provably inert.
    #[test]
    fn avx2_matesw_u8_matches_scalar() {
        // bwa's stock gap penalties as positive magnitudes: open 6, extend 1, same for both sides.
        // Equal open+extend sums take the shared-subtract body, unequal ones the split one.
        for &(o_del, e_del, o_ins, e_ins) in
            &[(6, 1, 6, 1), (5, 2, 7, 3), (1, 1, 4, 1), (4, 1, 1, 1)]
        {
            // Fixed-seed LCG so a failure is reproducible; `next()` yields the top 31 bits of the state,
            // the low bits of an LCG being the poorly distributed ones. Same generator and same job shape
            // as `tests::matesw_equals_scalar`, which is the NEON gate.
            let mut state = 0x1234_5678_9abc_def1u64;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                state >> 33
            };

            let mut qbufs: Vec<Vec<u8>> = Vec::new();
            let mut tbufs: Vec<Vec<u8>> = Vec::new();
            for _ in 0..2000 {
                let qlen = 5 + (next() % 146) as usize; // 5..=150 (varied lens exercise padding)
                let tlen = qlen + (next() % 500) as usize; // window >= query
                                                           // `% 5`, not `% 4`: code 4 is N, and it is the one symbol the score table gets wrong on
                                                           // its own (both-N reads as a match through XOR 0), so every kernel has to repair it with
                                                           // an explicit blend. Real reference has N runs, so this was always covered end to end,
                                                           // but drawing it here means a kernel that skips or misplaces the repair fails in
                                                           // `cargo test` instead of in a whole-genome md5.
                let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
                let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
                // 1 or 2 planted copies of the query; two copies is what gives `score2` something to find.
                let copies = 1 + (next() % 2);
                if next() % 5 != 0 {
                    for _ in 0..copies {
                        if tlen > qlen {
                            let at = (next() as usize) % (tlen - qlen + 1);
                            for k in 0..qlen {
                                t[at + k] = q[k];
                            }
                        }
                    }
                    // 0 to 3 substitutions so the alignment is not a perfect match and mismatch scoring
                    // is exercised.
                    for _ in 0..(next() % 4) {
                        let p = (next() as usize) % qlen;
                        q[p] = (next() % 4) as u8;
                    }
                }
                // ISSUE #43 strengthened this: EVERY job now carries an N in BOTH query and target,
                // not one in four. The x86 kernels now score through the same XOR table as NEON, whose
                // target N is re-encoded as `N_TARGET` (12), and the both-N cell is a slot of its own.
                // A port that kept `t == 4` anywhere, or that left the old `t >= m` dead-cell test in
                // place, breaks on exactly these bases and on nothing else. Extra draws stay random so
                // N runs and adjacent Ns are covered too.
                q[(next() as usize) % qlen] = 4;
                t[(next() as usize) % tlen] = 4;
                if next() % 3 == 0 {
                    q[(next() as usize) % qlen] = 4;
                }
                if next() % 3 == 0 {
                    t[(next() as usize) % tlen] = 4;
                }
                // And a both-N cell on the diagonal of a planted copy, which is the index (`4 ^ 12 = 8`)
                // the old blend form never had a slot for.
                let at = (next() as usize) % qlen;
                q[at] = 4;
                t[at.min(tlen - 1)] = 4;
                qbufs.push(q);
                tbufs.push(t);
            }

            // Both scoring configurations the NEON gate uses. Only the first fits u8, which is the whole
            // scope of this kernel; the second is kept as a *negative* check that the dispatch, not the
            // kernel, is what excludes it -- so it is compared through `fwd_local_sw_batch` instead.
            for &(a, b, minsc) in &[(1i8, 4i8, 19i32), (10, 40, 190)] {
                let mat = scmat(a, b);
                let max_sc = a as i32;
                let jobs: Vec<FwdJob> = qbufs
                    .iter()
                    .zip(tbufs.iter())
                    .map(|(q, t)| FwdJob {
                        query: q.as_slice(),
                        target: t.as_slice(),
                        minsc,
                        // Exercise the `endsc` freeze on half the jobs: without it `frozen`/`limit` (and
                        // therefore the `score2` row-scan truncation) are never driven at 32 lanes.
                        endsc: if t.len() % 2 == 0 { i32::MAX } else { 30 },
                    })
                    .collect();
                let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);

                let fits_u8 = jobs.iter().all(|j| {
                    (j.query.len().min(j.target.len()) as i32 * max_sc) < U8_SCORE_LIMIT
                        && j.query.len() < U8_SCORE_LIMIT as usize
                });
                let got = if fits_u8 {
                    // SAFETY: the u8 preconditions were just checked. AVX2 availability is *assumed*, not
                    // detected: under Rosetta detection lies (see this module's header).
                    unsafe {
                        fwd_local_sw_avx2_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
                    }
                } else {
                    fwd_local_sw_batch(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
                for i in 0..jobs.len() {
                    assert_eq!(
                        got[i],
                        want[i],
                        "job {i} (qlen {}, match {a}, u8 {fits_u8})",
                        jobs[i].query.len()
                    );
                }
            }
        }
    }

    /// Reproducible mate-rescue-shaped jobs, shared by the u8 and i16 verify tests: a random target,
    /// a random query planted 1-2 times with 0-3 substitutions, occasional N bases. Same generator and
    /// shape as `avx2_matesw_u8_matches_scalar` and the NEON gate `tests::matesw_equals_scalar`.
    pub(super) fn rescue_jobs(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut qbufs, mut tbufs) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let qlen = 5 + (next() % 146) as usize;
            let tlen = qlen + (next() % 500) as usize;
            // `% 5`, not `% 4`: code 4 is N, and it is the one symbol the score table gets wrong on
            // its own (both-N reads as a match through XOR 0), so every kernel has to repair it with
            // an explicit blend. Real reference has N runs, so this was always covered end to end,
            // but drawing it here means a kernel that skips or misplaces the repair fails in
            // `cargo test` instead of in a whole-genome md5.
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            let copies = 1 + (next() % 2);
            if next() % 5 != 0 {
                for _ in 0..copies {
                    if tlen > qlen {
                        let at = (next() as usize) % (tlen - qlen + 1);
                        for k in 0..qlen {
                            t[at + k] = q[k];
                        }
                    }
                }
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            // ISSUE #44 strengthened this, as #43 did for the AVX2/SSE4.1 generator: EVERY job now
            // carries an N in BOTH query and target, plus a both-N cell, because the AVX-512 kernel
            // now scores through the same XOR table with the target N re-encoded to `N_TARGET` (12)
            // and the both-N index `4 ^ 12 = 8` is a slot the old blend form never had.
            q[(next() as usize) % qlen] = 4;
            t[(next() as usize) % tlen] = 4;
            if next() % 3 == 0 {
                q[(next() as usize) % qlen] = 4;
            }
            if next() % 3 == 0 {
                t[(next() as usize) % tlen] = 4;
            }
            let at = (next() as usize) % qlen;
            q[at] = 4;
            t[at.min(tlen - 1)] = 4;
            qbufs.push(q);
            tbufs.push(t);
        }
        (qbufs, tbufs)
    }

    /// Compare the 16-lane AVX2 i16 kernel against the scalar lockstep, on the high-scoring `(10, 40)`
    /// matrix whose ceiling overflows u8 and so selects the i16 path. This is the kernel that closes the
    /// x86 gap for batches the u8 kernel cannot take (#12/#20): before it, they fell to the scalar DP.
    ///
    /// Like the u8 test, this calls the kernel *directly* rather than through `fwd_local_sw_batch`,
    /// because under Rosetta the dispatch's `is_x86_feature_detected!("avx2")` reads false and would
    /// route to scalar, so no AVX2 instruction would run.
    #[test]
    fn avx2_matesw_i16_matches_scalar() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (qbufs, tbufs) = rescue_jobs(2000);
        // match 10, mismatch -40: min(len) * 10 exceeds 250 for any query past ~25 bases, so the batch
        // overflows u8 and this is exactly the i16 kernel's domain. Every value still fits i16.
        let (a, b, minsc) = (10i8, 40i8, 190i32);
        let mat = scmat(a, b);
        let max_sc = a as i32;
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc,
                endsc: if t.len() % 2 == 0 { i32::MAX } else { 300 },
            })
            .collect();
        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // SAFETY: standard mat and every ceiling under I16_SCORE_LIMIT; AVX2 is assumed, not detected
        // (Rosetta lies), so the kernel is exercised on this Apple-Silicon host as well as native x86.
        let got = unsafe {
            fwd_local_sw_avx2_i16::<false>(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        };
        for i in 0..jobs.len() {
            assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
        }
    }
}

/// Byte-identity gate for the two 128-bit mate-rescue kernels, against `fwd_local_sw_scalar`.
///
/// Same jobs and same scoring pairs as the AVX2 gate above, on purpose: the SSE4.1 kernels are that
/// code transliterated operation for operation, so what has to be proved is that the group width
/// change (16 lanes instead of 32, 8 instead of 16) did not move a padded row or column that the
/// wider kernel happened to leave inert. Reusing the fixture makes the two tests comparable; a
/// different generator would have made a divergence look like a different test.
#[cfg(all(test, target_arch = "x86_64"))]
mod sse41_verify {
    use super::*;

    /// bwa 5x5 score matrix; a copy for the same reason `avx2_verify` keeps its own.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    #[test]
    fn sse41_matesw_u8_matches_scalar() {
        // Equal open+extend sums take the shared-subtract body, unequal ones the split one.
        for &(o_del, e_del, o_ins, e_ins) in
            &[(6, 1, 6, 1), (5, 2, 7, 3), (1, 1, 4, 1), (4, 1, 1, 1)]
        {
            let (qbufs, tbufs) = super::avx2_verify::rescue_jobs(2000);
            let (a, b, minsc) = (1i8, 4i8, 19i32);
            let mat = scmat(a, b);
            let max_sc = a as i32;
            let jobs: Vec<FwdJob> = qbufs
                .iter()
                .zip(tbufs.iter())
                .map(|(q, t)| FwdJob {
                    query: q.as_slice(),
                    target: t.as_slice(),
                    minsc,
                    // Half the jobs freeze on `endsc`, which is what drives the `score2` row-scan
                    // truncation; at 16 lanes it truncates at different jobs than at 32.
                    endsc: if t.len() % 2 == 0 { i32::MAX } else { 30 },
                })
                .collect();
            let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
            // SAFETY: standard mat, and every ceiling under U8_SCORE_LIMIT at match score 1. SSE4.1 is
            // assumed rather than detected, as in the AVX2 gate: under Rosetta detection lies, and this
            // kernel is exactly the one that host should be running.
            let got = unsafe {
                fwd_local_sw_sse41_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
            };
            for i in 0..jobs.len() {
                assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
            }
        }
    }

    #[test]
    fn sse41_matesw_i16_matches_scalar() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (qbufs, tbufs) = super::avx2_verify::rescue_jobs(2000);
        // match 10 / mismatch -40 overflows u8 past ~25 bases, which is the i16 kernel's domain.
        let (a, b, minsc) = (10i8, 40i8, 190i32);
        let mat = scmat(a, b);
        let max_sc = a as i32;
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc,
                endsc: if t.len() % 2 == 0 { i32::MAX } else { 300 },
            })
            .collect();
        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // SAFETY: standard mat, every ceiling under I16_SCORE_LIMIT; SSE4.1 assumed as above.
        let got = unsafe {
            fwd_local_sw_sse41_i16::<false>(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        };
        for i in 0..jobs.len() {
            assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
        }
    }
}

/// Byte-identity gate for the two AVX-512 mate-rescue kernels, against `fwd_local_sw_scalar`.
///
/// Unlike the AVX2 tests, these cannot run under Rosetta: it does not implement AVX-512, so executing
/// one of these kernels there faults. Each test therefore *self-skips* when `avx512bw` is absent -- so
/// on this Apple-Silicon host and on any AVX-512-less x86 runner it is a no-op, and it only truly
/// validates on a GitHub-hosted (or other) runner whose CPU exposes AVX-512BW. That is by design: the
/// kernels are gated the same way at runtime, so a machine that skips the test also never dispatches
/// to them. CI logs `/proc/cpuinfo` flags so a run where these actually executed is identifiable.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx512_verify {
    use super::*;

    /// Same bwa 5x5 matrix helper as the sibling test modules (they cannot see each other's privates).
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// Same reproducible mate-rescue job generator as `avx2_verify::rescue_jobs`, duplicated because
    /// sibling test modules cannot share private items (the established pattern in this file).
    fn rescue_jobs(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut qbufs, mut tbufs) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let qlen = 5 + (next() % 146) as usize;
            let tlen = qlen + (next() % 500) as usize;
            // `% 5`, not `% 4`: code 4 is N, and it is the one symbol the score table gets wrong on
            // its own (both-N reads as a match through XOR 0), so every kernel has to repair it with
            // an explicit blend. Real reference has N runs, so this was always covered end to end,
            // but drawing it here means a kernel that skips or misplaces the repair fails in
            // `cargo test` instead of in a whole-genome md5.
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            let copies = 1 + (next() % 2);
            if next() % 5 != 0 {
                for _ in 0..copies {
                    if tlen > qlen {
                        let at = (next() as usize) % (tlen - qlen + 1);
                        for k in 0..qlen {
                            t[at + k] = q[k];
                        }
                    }
                }
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            if next() % 4 == 0 {
                q[(next() as usize) % qlen] = 4;
            }
            if next() % 4 == 0 {
                t[(next() as usize) % tlen] = 4;
            }
            qbufs.push(q);
            tbufs.push(t);
        }
        (qbufs, tbufs)
    }

    /// The 64-lane AVX-512 u8 kernel vs the scalar lockstep, on stock DNA scores and on a `--meth`
    /// 5x5 table (the VBMI `vpermb` lookup under SDE `-spr`, two `pshufb` under `-skx`). Skipped where
    /// the CPU lacks AVX-512BW (including under Rosetta), so it is only a real check on AVX-512 hardware.
    #[test]
    fn avx512_matesw_u8_matches_scalar() {
        if !crate::avx512_testable() {
            eprintln!("skipping avx512_matesw_u8_matches_scalar: no avx512bw on this CPU");
            return;
        }
        // Equal open+extend sums take the shared-subtract body, unequal ones the split one.
        for (&(o_del, e_del, o_ins, e_ins), meth) in
            [(6, 1, 6, 1), (5, 2, 7, 3), (1, 1, 4, 1), (4, 1, 1, 1)]
                .iter()
                .flat_map(|g| [(g, false), (g, true)])
        {
            let (qbufs, tbufs) = rescue_jobs(2000);
            let (a, b, minsc) = (1i8, 4i8, 19i32);
            let mut mat = scmat(a, b);
            if meth {
                mat[5 + 3] = 0;
                assert!(!mat_is_standard(5, &mat));
            }
            let max_sc = a as i32;
            let jobs: Vec<FwdJob> = qbufs
                .iter()
                .zip(tbufs.iter())
                .map(|(q, t)| FwdJob {
                    query: q.as_slice(),
                    target: t.as_slice(),
                    minsc,
                    endsc: if t.len() % 2 == 0 { i32::MAX } else { 30 },
                })
                .collect();
            let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
            // SAFETY: AVX-512 BW/VL/DQ just confirmed; u8 preconditions hold for these stock-DNA-scored jobs.
            let got = unsafe {
                fwd_local_sw_avx512_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
            };
            for i in 0..jobs.len() {
                assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
            }
        }
    }

    /// The 32-lane AVX-512 i16 kernel vs the scalar lockstep, on the high-scoring `(10, 40)` matrix that
    /// overflows u8. Skipped without AVX-512BW.
    #[test]
    fn avx512_matesw_i16_matches_scalar() {
        if !crate::avx512_testable() {
            eprintln!("skipping avx512_matesw_i16_matches_scalar: no avx512bw on this CPU");
            return;
        }
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (qbufs, tbufs) = rescue_jobs(2000);
        let (a, b, minsc) = (10i8, 40i8, 190i32);
        let mat = scmat(a, b);
        let max_sc = a as i32;
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc,
                endsc: if t.len() % 2 == 0 { i32::MAX } else { 300 },
            })
            .collect();
        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // SAFETY: AVX-512 BW/VL/DQ just confirmed; every ceiling is under I16_SCORE_LIMIT and mat is standard.
        let got = unsafe {
            fwd_local_sw_avx512_i16::<false>(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        };
        for i in 0..jobs.len() {
            assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
        }
    }
}

/// Same-runner A/B throughput probe for the x86 rescue kernels: scalar vs AVX2 vs (where present)
/// AVX-512, on one set of mate-rescue-shaped jobs. `#[ignore]`d so it never runs in the normal test
/// pass; CI's `bench-x86` workflow runs it with `--ignored --nocapture`.
///
/// This is the *trustworthy* half of the x86 measurement story (issue #20). It is not a bwa-mem2
/// head-to-head, but it needs no reference genome and, crucially, times all three kernels on the
/// **same** runner in the **same** process, so the CPU model and the VM's noisy neighbours cancel in
/// the ratio. Absolute Gcell/s from a shared cloud runner is meaningless; the scalar-relative and
/// avx2-relative speedups are what to read. The end-to-end small-reference A/B in the same workflow
/// supplies the (deliberately caveated) whole-pipeline number.
#[cfg(all(test, target_arch = "x86_64"))]
mod bench {
    use super::*;
    use std::time::Instant;

    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// Near-uniform mate-rescue jobs: 150 bp query, ~500 bp window, the query planted once with a few
    /// substitutions. Uniform dimensions are the realistic case (query = read length, window = the
    /// insert-size interval) and the one where wide vectors keep every lane busy.
    fn bench_jobs(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut state = 0xdead_beef_1234_5678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut qbufs, mut tbufs) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let qlen = 150usize;
            let tlen = 500usize;
            // `% 5`, not `% 4`: code 4 is N, and it is the one symbol the score table gets wrong on
            // its own (both-N reads as a match through XOR 0), so every kernel has to repair it with
            // an explicit blend. Real reference has N runs, so this was always covered end to end,
            // but drawing it here means a kernel that skips or misplaces the repair fails in
            // `cargo test` instead of in a whole-genome md5.
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            let at = (next() as usize) % (tlen - qlen + 1);
            for k in 0..qlen {
                t[at + k] = q[k];
            }
            for _ in 0..3 {
                let p = (next() as usize) % qlen;
                q[p] = (next() % 4) as u8;
            }
            qbufs.push(q);
            tbufs.push(t);
        }
        (qbufs, tbufs)
    }

    #[test]
    #[ignore = "throughput probe, run explicitly in the bench-x86 CI workflow"]
    fn rescue_kernel_ab() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (a, b, minsc) = (1i8, 4i8, 19i32); // stock DNA scores => the u8 kernels
        let mat = scmat(a, b);
        let max_sc = a as i32;
        let (qbufs, tbufs) = bench_jobs(8192);
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc,
                endsc: i32::MAX,
            })
            .collect();
        // "Cells" as issue #12 counts them: real query x real target, summed over jobs.
        let cells: u64 = jobs
            .iter()
            .map(|j| (j.query.len() * j.target.len()) as u64)
            .sum();

        // Time the best of several reps (min wall = least noise), after one warm-up rep.
        let reps = 5;
        let bench = |label: &str, run: &dyn Fn() -> Vec<(i32, i32, i32, i32, i32)>| -> f64 {
            let _ = run(); // warm up
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let out = run();
                let dt = t0.elapsed().as_secs_f64();
                std::hint::black_box(&out);
                best = best.min(dt);
            }
            let gcell = cells as f64 / best / 1e9;
            eprintln!(
                "  {label:<10} {best_ms:>8.2} ms   {gcell:>6.3} Gcell/s",
                best_ms = best * 1e3
            );
            best
        };

        eprintln!(
            "rescue_kernel_ab: {} jobs, {} DP cells (150 bp query, 500 bp window), best of {reps}",
            jobs.len(),
            cells
        );
        let scalar = bench("scalar", &|| {
            fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        });
        // AVX2 is called directly so it is exercised even under Rosetta; on a native runner it is the
        // same code the dispatch selects.
        let avx2 = bench("avx2_u8", &|| unsafe {
            fwd_local_sw_avx2_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        });
        eprintln!("  avx2 vs scalar: {:.2}x", scalar / avx2);
        if bwa_extend::has_avx512() {
            let avx512 = bench("avx512_u8", &|| unsafe {
                fwd_local_sw_avx512_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
            });
            eprintln!(
                "  avx512 vs scalar: {:.2}x   avx512 vs avx2: {:.2}x",
                scalar / avx512,
                avx2 / avx512
            );
        } else {
            eprintln!("  avx512_u8  skipped (no avx512bw on this runner)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

    /// bwa 5x5 score matrix: match `a`, mismatch `-b`, N row/col `-1`.
    ///
    /// # Parameters
    /// - `a`: match bonus as a positive magnitude, written to the diagonal as `+a`.
    /// - `b`: mismatch penalty as a positive magnitude, written off-diagonal as `-b`.
    ///
    /// # Returns
    /// 25 entries, row-major, in the exact form [`mat_is_standard`] accepts.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// Scoring matrices whose match bonus plus mismatch penalty leaves the u8 kernels' biased
    /// substitution table no byte headroom must still align, on the i16 path, and agree with the
    /// scalar reference.
    ///
    /// This is the regression test for a crash, not for a score. `-x intractg` sets `-B 9`, and
    /// with the stock `-A 1` that makes `mispen + mtch` 10 where the biased table affords 6. Before
    /// [`u8_table_has_headroom`] gated the dispatch, such a batch reached the u8 kernel's hard
    /// assertion and aborted the process on the first paired-end rescue batch, so the whole
    /// `-x intractg` paired-end mode was unusable while bwa-mem2 aligns it fine. `-A 3 -B 4` and
    /// `-B 6` are the two other reachable ways to cross the same line, and they are included so a
    /// future gate that only special-cases `intractg` fails here.
    #[test]
    fn saturating_score_matrices_fall_back_instead_of_panicking() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        // Deterministic mate-rescue-shaped jobs: a short query planted inside a longer window, so
        // the local alignment is real and every field (score, te, qe, score2, te2) is exercised.
        let mut state = 0x0bad_c0de_dead_beefu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..64 {
            let qlen = 40 + (next() % 60) as usize;
            let tlen = qlen + (next() % 200) as usize;
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let at = (next() as usize) % (tlen - qlen + 1);
            t[at..at + qlen].copy_from_slice(&q);
            qbufs.push(q);
            tbufs.push(t);
        }

        // `(a, b)` pairs that all fail the headroom test (`250 + b + a > 256`): `intractg`'s own
        // scores, then the smallest crossing on each of the two axes.
        for &(a, b) in &[(1i8, 9i8), (3, 4), (1, 6)] {
            assert!(
                !u8_table_has_headroom(&scmat(a, b)),
                "the point of this case is that (a {a}, b {b}) has no u8 headroom"
            );
            let mat = scmat(a, b);
            let jobs: Vec<FwdJob> = qbufs
                .iter()
                .zip(tbufs.iter())
                .map(|(q, t)| FwdJob {
                    query: q.as_slice(),
                    target: t.as_slice(),
                    minsc: 1,
                    endsc: i32::MAX,
                })
                .collect();
            let want =
                fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, i32::from(a));
            // Before the fix this call aborted the process instead of returning.
            let got = fwd_local_sw_batch(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, i32::from(a));
            assert_eq!(got, want, "batched kernel diverged at (a {a}, b {b})");
        }

        // And the stock scores still take the u8 path, so the gate did not disable it outright.
        assert!(u8_table_has_headroom(&scmat(1, 4)));
    }

    /// Random mate-rescue-shaped jobs (short query, longer window, some shared substring so the SW
    /// finds a real local alignment), then assert the batched kernel matches per-job `ksw_align2` on
    /// every field. This is the byte-identity gate for the NEON kernels.
    #[test]
    fn matesw_equals_scalar() {
        // bwa's stock gap penalties as positive magnitudes: open 6, extend 1, same for both sides.
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

        // Fixed-seed LCG so a failure is reproducible; `next()` yields the top 31 bits of the state,
        // the low bits of an LCG being the poorly distributed ones.
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // Build a pool of jobs and their owned buffers.
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..2000 {
            let qlen = 5 + (next() % 146) as usize; // 5..=150 (varied lens exercise padding)
            let tlen = qlen + (next() % 500) as usize; // window >= query
                                                       // `% 5`, not `% 4`: code 4 is N, and it is the one symbol the score table gets wrong on
                                                       // its own (both-N reads as a match through XOR 0), so every kernel has to repair it with
                                                       // an explicit blend. Real reference has N runs, so this was always covered end to end,
                                                       // but drawing it here means a kernel that skips or misplaces the repair fails in
                                                       // `cargo test` instead of in a whole-genome md5.
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            // Embed one or two mutated copies of the query into the target so local alignments (and a
            // 2nd-best, for score2) exist.
            // 1 or 2 planted copies of the query; two copies is what gives `score2` something to find.
            let copies = 1 + (next() % 2);
            if next() % 5 != 0 {
                for _ in 0..copies {
                    if tlen > qlen {
                        // Start offset in the target where this copy is written; the two copies may
                        // land on top of each other, which is fine.
                        let at = (next() as usize) % (tlen - qlen + 1);
                        for k in 0..qlen {
                            t[at + k] = q[k];
                        }
                    }
                }
                // 0 to 3 substitutions in the query after planting, so the alignment is not a perfect
                // match and mismatch scoring is exercised.
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            // ISSUE #43 strengthened this: EVERY job now carries an N in BOTH query and target,
            // not one in four. The x86 kernels now score through the same XOR table as NEON, whose
            // target N is re-encoded as `N_TARGET` (12), and the both-N cell is a slot of its own.
            // A port that kept `t == 4` anywhere, or that left the old `t >= m` dead-cell test in
            // place, breaks on exactly these bases and on nothing else. Extra draws stay random so
            // N runs and adjacent Ns are covered too.
            q[(next() as usize) % qlen] = 4;
            t[(next() as usize) % tlen] = 4;
            if next() % 3 == 0 {
                q[(next() as usize) % qlen] = 4;
            }
            if next() % 3 == 0 {
                t[(next() as usize) % tlen] = 4;
            }
            // And a both-N cell on the diagonal of a planted copy, which is the index (`4 ^ 12 = 8`)
            // the old blend form never had a slot for.
            let at = (next() as usize) % qlen;
            q[at] = 4;
            t[at.min(tlen - 1)] = 4;
            qbufs.push(q);
            tbufs.push(t);
        }
        let jobs: Vec<KswJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| KswJob {
                query: q.as_slice(),
                target: t.as_slice(),
            })
            .collect();

        // (match, mismatch, minsc): match=1 -> scores fit u8 (16-lane kernel); match=10 -> scores
        // exceed 250 (8-lane i16 kernel). Cover both, both against per-job ksw_align2.
        for &(a, b, minsc) in &[(1i8, 4i8, 19i32), (10, 40, 190)] {
            let mat = scmat(a, b);
            let max_sc = a as i32;
            let batched =
                batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc);
            for (i, j) in jobs.iter().enumerate() {
                // Same kernel width mem_matesw would pick, since it changes the result.
                let lanes = if j.query.len() as i32 * max_sc < 250 {
                    16
                } else {
                    8
                };
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, lanes,
                );
                assert_eq!(
                    batched[i],
                    want,
                    "job {i} (qlen {}, match {a})",
                    j.query.len()
                );
            }
        }
    }

    /// The u8/i16 dispatch boundary, straddled deliberately (issue #54, trap 3).
    ///
    /// The u8 rescue kernel is exact only while every H/E/F cell stays under [`U8_SCORE_LIMIT`], and
    /// `fwd_local_sw_batch` routes a job to the 8-lane i16 kernel the moment its score ceiling
    /// `min(qlen, tlen) * max_sc` reaches it. A GPU port has to detect saturation at **exactly** the
    /// same threshold: one off-by-one and it returns a clipped score where the CPU replayed the job
    /// in a wider type, which is a wrong answer, not a slower one.
    ///
    /// Random sweeps never land on that edge. This one is built to sit on it: for each of several
    /// match bonuses `a`, query lengths are chosen so that `qlen * a` falls exactly on 248, 249, 250,
    /// 251, 254, 255 and 256, on both sides of the 250 limit and on both sides of the u8 lane's own
    /// 255 wrap. The second sweep straddles the other u8 precondition, `qlen < 250`, which exists
    /// because the argmax column shares the lane with the score.
    ///
    /// Both `score` and `score2` are compared, `score2` because `minsc` is set low enough that the
    /// rival-alignment list is populated: a saturation bug that spared the best alignment but
    /// clipped a runner-up would change MAPQ and nothing else.
    #[test]
    fn matesw_saturation_boundary_equals_scalar() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mut state = 0xB0DE_1234_5678_9AB1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // (match bonus, mismatch penalty). `a` values chosen so `qlen * a` can hit the targets below
        // exactly: 2 and 5 divide 250, 3 and 7 do not, so the sweep covers both aligned and
        // misaligned arithmetic on the threshold.
        for &(a, b) in &[(1i8, 4i8), (2, 8), (3, 12), (5, 20), (7, 28)] {
            let mat = scmat(a, b);
            let max_sc = a as i32;
            let mut qbufs: Vec<Vec<u8>> = Vec::new();
            let mut tbufs: Vec<Vec<u8>> = Vec::new();
            for &target_product in &[248i32, 249, 250, 251, 254, 255, 256] {
                // The qlen whose ceiling brackets the target product from below and above, so the
                // sweep covers the exact edge even when `a` does not divide it.
                for qlen in [
                    (target_product / max_sc) as usize,
                    (target_product / max_sc) as usize + 1,
                ] {
                    if qlen == 0 || qlen > 400 {
                        continue;
                    }
                    // Target longer than the query, so `min(qlen, tlen)` is the query and the
                    // ceiling is exactly `qlen * a`.
                    let tlen = qlen * 3 + 17;
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
                    let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
                    // Two planted copies: the first gives a perfect alignment that reaches the
                    // ceiling exactly, the second gives `score2` something at nearly the same score,
                    // which is where a clipped runner-up would show.
                    t[2..2 + qlen].copy_from_slice(&q);
                    let at = tlen - qlen - 3;
                    t[at..at + qlen].copy_from_slice(&q);
                    // One mismatch in the second copy, so the two are close but not equal.
                    t[at + qlen / 2] = (t[at + qlen / 2] + 1) % 4;
                    qbufs.push(q);
                    tbufs.push(t);
                }
            }
            // The other u8 precondition: the query must be under 250 BASES, because the argmax
            // column shares the lane with the score. Straddled with a=1 so the ceiling is not what
            // moves.
            if a == 1 {
                for &qlen in &[248usize, 249, 250, 251] {
                    let tlen = qlen + 40;
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
                    let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
                    t[10..10 + qlen].copy_from_slice(&q);
                    qbufs.push(q);
                    tbufs.push(t);
                }
            }

            let jobs: Vec<KswJob> = qbufs
                .iter()
                .zip(tbufs.iter())
                .map(|(q, t)| KswJob {
                    query: q.as_slice(),
                    target: t.as_slice(),
                })
                .collect();
            // `minsc` low enough that the `score2` list is really populated.
            let minsc = max_sc * 5;
            let batched =
                batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc);
            for (i, j) in jobs.iter().enumerate() {
                // The width bwa's own `KSW_XBYTE` test would pick, which is observable through the
                // padded query length and therefore through `score2`.
                let lanes = if j.query.len() as i32 * max_sc < 250 {
                    16
                } else {
                    8
                };
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, lanes,
                );
                assert_eq!(
                    batched[i],
                    want,
                    "boundary job {i}: a {a}, qlen {}, ceiling {}",
                    j.query.len(),
                    j.query.len().min(j.target.len()) as i32 * max_sc
                );
            }
        }
    }

    /// The rescue kernel's argmax tie rule, on inputs where ties are the norm (issue #54, trap 2).
    ///
    /// `matesw.rs` keeps the FIRST maximum (`vcgtq_u8`, strict `>`), the opposite of the extension
    /// kernel, which keeps the last. A port that swaps them still returns the right `score`; it
    /// returns different `te`/`qe`, and only on inputs where cells tie, which random sequence
    /// almost never produces. So this generates them: periodic targets against queries in phase.
    ///
    /// Written now, before any GPU kernel exists, because that is the point of #54.
    #[test]
    fn matesw_tie_rule_equals_scalar() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mat = scmat(1, 4);
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        // Period 1 is a homopolymer, where every cell of a row ties. 7 shares no factor with any
        // SIMD width here, so a tie can never coincide with a lane boundary by luck.
        for &period in &[1usize, 2, 3, 4, 7] {
            for &qlen in &[9usize, 32, 63, 100, 149] {
                for &tlen in &[qlen, qlen + 1, qlen * 3, qlen * 3 + 5] {
                    qbufs.push((0..qlen).map(|i| (i % period) as u8 % 4).collect());
                    tbufs.push((0..tlen).map(|i| (i % period) as u8 % 4).collect());
                }
            }
        }
        let jobs: Vec<KswJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| KswJob {
                query: q.as_slice(),
                target: t.as_slice(),
            })
            .collect();
        for &minsc in &[1i32, 19, 100] {
            let batched = batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, 1);
            for (i, j) in jobs.iter().enumerate() {
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, 1, 16,
                );
                assert_eq!(
                    batched[i],
                    want,
                    "tie job {i} (qlen {}, tlen {}, minsc {minsc})",
                    j.query.len(),
                    j.target.len()
                );
            }
        }
    }

    /// Every `--meth` matrix (`MethOpt::fill_mats` at `-A 1..=10`, `-B 1..=20`: OT and OB, neutral,
    /// genomic and collapsed) has an add code, and the code gives back every cell.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn mat5_add_code_covers_meth_matrices() {
        for a in 1..=10i8 {
            for b in 1..=20i8 {
                for (r, q) in [(1usize, 3usize), (2, 0)] {
                    for (v, mirror) in [(0, false), (a, false), (a, true)] {
                        let mut mat = scmat(a, b);
                        mat[r * 5 + q] = v;
                        if mirror {
                            mat[q * 5 + r] = a;
                        }
                        let (t, qc, tbl) = mat5_add_code(&mat)
                            .unwrap_or_else(|| panic!("no add code: a {a} b {b} cell {r}{q} {v}"));
                        assert!(t.iter().chain(&qc).all(|&c| c < 16) && !qc.contains(&ZPAD));
                        for x in 0..5 {
                            for y in 0..5 {
                                assert_eq!(tbl[((t[x] + qc[y]) & 15) as usize], mat[5 * x + y]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The 16x16 block transpose against the plain index map, on 3 x 2 blocks: each byte's index
    /// goes through as its low and its high byte, so every position is told apart.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn transpose16_matches_index_map() {
        let (lanes, rows) = (48, 32);
        for shift in [0, 8] {
            let src: Vec<u8> = (0..lanes * rows).map(|k| (k >> shift) as u8).collect();
            let got = transpose16(&src, lanes, rows);
            for l in 0..lanes {
                for r in 0..rows {
                    assert_eq!(got[r * lanes + l], src[l * rows + r], "lane {l} row {r}");
                }
            }
        }
    }

    /// `n` jobs `(query, target, minsc)` of one query length `qlen` and ragged windows (290 bp and
    /// up), each query planted at the very end of its window, so the best alignment ends on the last
    /// real row and its H then walks diagonally through the padded columns. The shape of the all-ZPAD
    /// column regime.
    ///
    /// The even jobs are the case where that walk decides `score2`, at `minsc` 10 and with an N-free
    /// query (so `score` is `qlen`): the query's last 14 bases end on the last row `score2` excludes
    /// below `te`, `te + qlen`. The padded columns carry that 14 down the next rows, and bwa's
    /// candidate list keeps a plateau's first row, so the carry starts a new candidate two rows
    /// down, just past the excluded rows, and it is `score2`. A kernel that drops or misplaces the
    /// carry has no candidate there (the deletion path, 14 - 7, misses `minsc`). Those two rows
    /// end the window, and the windows are `200 + 90 k` plus 0 to 3, so they fall on every row of
    /// a NEON quad or an x86 pair, and the longest lane of a group puts them in the row-pair and
    /// single-row tails too (NEON's groups of 16 end 1 and 2 rows past a quad, the x86 groups of 32
    /// and 64 on odd rows).
    fn uniform_qlen_jobs(qlen: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, i32)> {
        let mut state = 0xfeed_face_1234_5678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut jobs = Vec::new();
        for k in 1..=n {
            let tlen = 200 + k * 90 + (k / 2 + k / 16 + k / 64) % 4;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let q: Vec<u8> = (0..qlen)
                .map(|_| (next() % (4 + k as u64 % 2)) as u8)
                .collect();
            let minsc = if k % 2 == 0 {
                // Main alignment, `qlen - 14` random rows, the stray 14, the two carry rows.
                let te = tlen - 3 - qlen;
                t[te + 1 - qlen..=te].copy_from_slice(&q);
                t[te + qlen - 13..te + qlen + 1].copy_from_slice(&q[qlen - 14..]);
                10
            } else {
                t[tlen - qlen..].copy_from_slice(&q);
                19
            };
            jobs.push((q, t, minsc));
        }
        jobs
    }

    /// The even jobs of `uniform_qlen_jobs` whose `score2` is the pad carry's, at `te + qlen + 2`.
    /// Shows the test still reaches the carry.
    fn pad_carry_decided(
        bufs: &[(Vec<u8>, Vec<u8>, i32)],
        want: &[(i32, i32, i32, i32, i32)],
    ) -> usize {
        bufs.iter()
            .zip(want)
            .filter(|((q, _, minsc), w)| {
                *minsc == 10
                    && w.0 == q.len() as i32
                    && w.3 >= 14
                    && w.4 == w.1 + q.len() as i32 + 2
            })
            .count()
    }

    /// The all-ZPAD third column regime of the AVX2 and AVX-512 u8 kernels' two-row body, on the
    /// shape of `matesw_uniform_qlen_pad_columns_equal_scalar`: 147 bp queries padded to 160, ragged
    /// windows, an odd and an even endsc, standard and `--meth` matrices, shared and split gaps.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_avx2_matesw_uniform_qlen_pad_columns_equal_scalar() {
        let max_sc = 1i32;
        const QLEN: usize = 147;
        assert_eq!(ksw_padded_qlen(QLEN, max_sc), 160);
        let bufs = uniform_qlen_jobs(QLEN, 128);
        for (&(o_del, e_del, o_ins, e_ins), meth) in [(6, 1, 6, 1), (5, 2, 7, 3)]
            .iter()
            .flat_map(|g| [(g, false), (g, true)])
        {
            let mut mat = scmat(1, 4);
            if meth {
                mat[5 + 3] = 0;
            }
            let jobs: Vec<FwdJob> = bufs
                .iter()
                .map(|(q, t, minsc)| FwdJob {
                    query: q.as_slice(),
                    target: t.as_slice(),
                    minsc: *minsc,
                    endsc: if t.len() / 90 % 3 == 0 { 120 } else { i32::MAX },
                })
                .collect();
            let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
            assert!(
                want.iter().filter(|r| r.3 >= 0).count() >= 8,
                "score2 unpopulated"
            );
            assert!(
                pad_carry_decided(&bufs, &want) >= 30,
                "the pad carry decides no score2"
            );
            assert!(
                want.iter().any(|r| r.0 >= 120 && r.0 < QLEN as i32),
                "endsc never froze a job"
            );
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 just detected; 147 bp queries keep the u8 ceiling under 250.
                let got = unsafe {
                    fwd_local_sw_avx2_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
                assert_eq!(
                    got, want,
                    "avx2 meth={meth} gaps {o_del},{e_del},{o_ins},{e_ins}"
                );
            }
            if !crate::avx512_testable() {
                eprintln!("skipping avx512 uniform-qlen pad columns: no avx512bw on this CPU");
            } else {
                // SAFETY: AVX-512 BW/VL/DQ just confirmed; same u8 preconditions.
                let got = unsafe {
                    fwd_local_sw_avx512_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
                assert_eq!(
                    got, want,
                    "avx512 meth={meth} gaps {o_del},{e_del},{o_ins},{e_ins}"
                );
            }
        }
    }

    /// The all-ZPAD third column regime (issue #47), on groups whose lanes have a UNIFORM query
    /// length, which is the only shape that opens it.
    ///
    /// `matesw_ragged_tlen_equals_scalar` cannot reach this body: its query lengths differ, so
    /// `n_pad == qmax` and the padded columns keep going through the tail. Here every lane is 147 bp
    /// and pads to 160, so 13 of the 160 columns run the new body. Target lengths stay ragged, so a
    /// lane can be past its `tlen` while the group runs on, which is the case where the dropped PAD
    /// blend has to be provably invisible.
    ///
    /// What the padded columns can and cannot reach, which the first draft of this test got wrong:
    /// they can NEVER move `score`/`te`/`qe`. A padded column copies the diagonal, so its H equals
    /// some earlier row's H at a real column, which was already folded into `gmax` when that row
    /// finished; the `>` in `finish_row` then rejects it. They CAN move `score2`/`te2`, because
    /// those come from `rowmax`, which every column writes, and a padded column can raise a LATER
    /// row's maximum. So `score2` is the field carrying the evidence here, and the assertion below
    /// checks it is actually populated.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn matesw_uniform_qlen_pad_columns_equal_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mat = scmat(1, 4);
        let max_sc = 1i32;
        // 147 pads to 160 at 16 lanes: 13 padded columns, right at the shape the issue measured.
        const QLEN: usize = 147;
        assert_eq!(ksw_padded_qlen(QLEN, max_sc), 160);

        let bufs = uniform_qlen_jobs(QLEN, 32);
        let jobs: Vec<FwdJob> = bufs
            .iter()
            .map(|(q, t, minsc)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc: *minsc,
                endsc: i32::MAX,
            })
            .collect();

        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        assert!(
            pad_carry_decided(&bufs, &want) >= 10,
            "the pad carry decides no score2"
        );
        // SAFETY: neon detected above; 147 bp queries and a score ceiling of 147 both under 250;
        // `scmat` is standard.
        let got = unsafe {
            fwd_local_sw_neon_u8_impl::<true, true, false>(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
            )
        };
        assert_eq!(got, want);
        // `score2` really is in play, so the `rowmax` the padded columns feed is being compared and
        // not silently `-1` on every job.
        assert!(
            got.iter().filter(|r| r.3 >= 0).count() >= 8,
            "score2 is unpopulated, so the rowmax path the padded columns feed is untested"
        );
    }

    /// The four monomorphisations of the NEON u8 kernel, on groups with deliberately RAGGED target
    /// lengths, all against the scalar source of truth.
    ///
    /// This is the gate issue #45 asks for rather than an argument. `USQADD` and the biased add
    /// disagree on exactly one input, an out-of-range score-table read, which happens in the FAST
    /// column body when a lane is already past its own `tlen` while longer lanes in the same group
    /// keep the loop running. Those cells are supposed to be unreachable from the output; a group of
    /// sixteen windows spanning 100 to 1600 bases makes the longest lane run 15x past the shortest,
    /// so if the difference could leak, it leaks here.
    ///
    /// The four bodies are called directly instead of through `BWA4_RESCUE_*`: those toggles are
    /// cached in a `OnceLock`, so one test process could otherwise only ever exercise one of them.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn matesw_ragged_tlen_equals_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mat = scmat(1, 4);
        let max_sc = 1i32;

        let mut state = 0x0bad_c0de_dead_beefu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // Two full groups of 16, target lengths 100, 200, ... 3200: within one group the longest
        // window is 15x (then 2x) the shortest, so the fast body spends most of its rows with dead
        // lanes beside live ones.
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for k in 1..=32usize {
            let tlen = k * 100;
            // 40..=150, all under the u8 ceiling, and never longer than half the window.
            let qlen = (40 + (next() % 111) as usize).min(tlen / 2);
            // `% 5`, not `% 4`: code 4 is N, and the N slots are exactly the table entries the
            // signed rewrite has to reproduce. See `matesw_equals_scalar`.
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            // Plant a copy of the query late in the window, so a long lane's best alignment lies
            // beyond every short lane's last row.
            let at = tlen - qlen - (next() as usize % 20);
            t[at..at + qlen].copy_from_slice(&q);
            qbufs.push(q);
            tbufs.push(t);
        }
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc: 19,
                // A finite `endsc` so the freeze path runs too: freezing is what makes `limit[l]`
                // shorter than `tlen[l] - 1`, i.e. the other way a lane goes dead early.
                endsc: 60,
            })
            .collect();

        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // SAFETY: neon detected above; queries under 250 bases and score ceiling `min(len) * 1`
        // under 250; `scmat` is standard.
        unsafe {
            for (name, got) in [
                (
                    "usqadd+shareoe",
                    fwd_local_sw_neon_u8_impl::<true, true, false>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "usqadd",
                    fwd_local_sw_neon_u8_impl::<true, false, false>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "shareoe",
                    fwd_local_sw_neon_u8_impl::<false, true, false>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "baseline",
                    fwd_local_sw_neon_u8_impl::<false, false, false>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
            ] {
                for (i, j) in jobs.iter().enumerate() {
                    assert_eq!(
                        got[i],
                        want[i],
                        "{name}: job {i} (qlen {}, tlen {})",
                        j.query.len(),
                        j.target.len()
                    );
                }
            }
        }
    }

    /// `--meth` neutral / genomic rescue (the `MAT5` kernel, the prune deciding under the
    /// relation) against the unpruned `batched_ksw_align2`: equal on every job scoring `>= minsc`,
    /// and `qb = tb = -1` below it, the contract `mem_matesw` reads. Windows hold a converted
    /// mate, a random one, or a converted mate under heavy noise, so all three decisions occur.
    #[test]
    fn mate_rescue_meth_relation_prune_is_exact() {
        let (o_del, e_del, o_ins, e_ins, minsc) = (6, 1, 6, 1, 19);
        let mut state = 0x7a95_c0de_1357_9bdfu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for (r, q) in [(1usize, 3usize), (2, 0)] {
            for conv in [0i8, 1] {
                let mut mat = scmat(1, 4);
                mat[r * 5 + q] = conv;
                assert!(matches!(
                    crate::merged_pair(5, &mat),
                    Some((_, _, _, false))
                ));
                let (mut qbufs, mut tbufs) = (Vec::new(), Vec::new());
                for k in 0..300usize {
                    let qlen = 60 + (next() % 91) as usize;
                    let tlen = qlen + 50 + (next() % 900) as usize;
                    let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
                    let qv: Vec<u8> = if k % 3 == 1 {
                        (0..qlen).map(|_| (next() % 4) as u8).collect()
                    } else {
                        let at = next() as usize % (tlen - qlen);
                        let noise = if k % 3 == 2 { 4 } else { 40 };
                        t[at..at + qlen]
                            .iter()
                            .map(|&b| match next() % noise {
                                0 => (next() % 4) as u8,
                                _ if b as usize == r && next() % 2 == 0 => q as u8,
                                _ => b,
                            })
                            .collect()
                    };
                    if k % 50 == 0 {
                        t[next() as usize % tlen] = 4;
                    }
                    qbufs.push(qv);
                    tbufs.push(t);
                }
                let jobs: Vec<KswJob> = qbufs
                    .iter()
                    .zip(&tbufs)
                    .map(|(q, t)| KswJob {
                        query: q,
                        target: t,
                    })
                    .collect();
                let want = batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, 1);
                let got = batched_mate_rescue(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, 1);
                let mut low = 0;
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    if w.score >= minsc {
                        assert_eq!(g, w, "freed ({r},{q}) conv {conv} job {k}");
                    } else {
                        low += 1;
                        assert!(
                            g.score < minsc && g.qb == -1 && g.tb == -1,
                            "job {k}: {g:?}"
                        );
                    }
                }
                assert!((50..=250).contains(&low), "low {low}");
            }
        }
    }

    /// The u8 admission at its edge (`u8_job_fits`): 250 bp mates at bwa's scores, ceiling 250 and
    /// a third reaching it, some frozen there by `endsc`, under the standard and a `--meth` matrix,
    /// against the scalar kernel on every u8 kernel.
    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn matesw_u8_admission_edge_equals_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let (a, b, c, d) = (6, 1, 6, 1);
        let mut state = 0x0bad_5eed_1234_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut meth = scmat(1, 4);
        (meth[5 + 3], meth[3 * 5 + 1]) = (1, 1);
        for mat in [scmat(1, 4), meth] {
            let bufs: Vec<(Vec<u8>, Vec<u8>)> = (0..48usize)
                .map(|k| {
                    let tlen = 250 + k * 13;
                    let t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
                    let at = next() as usize % (tlen - 249);
                    let mut q = t[at..at + 250].to_vec();
                    if k % 3 != 0 {
                        for _ in 0..k % 5 {
                            let p = next() as usize % 250;
                            q[p] = (next() % 5) as u8;
                        }
                    }
                    (q, t)
                })
                .collect();
            let jobs: Vec<FwdJob> = bufs
                .iter()
                .enumerate()
                .map(|(k, (q, t))| FwdJob {
                    query: q,
                    target: t,
                    minsc: 30,
                    endsc: if k % 4 == 3 { 250 } else { i32::MAX },
                })
                .collect();
            assert!(u8_table_has_headroom(&mat));
            assert!(jobs
                .iter()
                .all(|j| u8_job_fits(j.query.len(), j.target.len(), &mat, 1)));
            let want = fwd_local_sw_scalar(&jobs, 5, &mat, a, b, c, d, 1);
            assert!(want.iter().filter(|w| w.0 == 250).count() >= 10);
            assert_eq!(fwd_local_sw_batch(&jobs, 5, &mat, a, b, c, d, 1), want);
            let standard = mat_is_standard(5, &mat);
            // SAFETY: neon detected; every job passes `u8_job_fits` and the table has headroom;
            // standard or `mat5_u8_ok` mat (MAT5 only with USQADD, as the dispatch).
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let mut got = vec![
                    fwd_local_sw_neon_u8_impl::<true, true, true>(&jobs, 5, &mat, a, b, c, d, 1),
                    fwd_local_sw_neon_u8_impl::<true, false, true>(&jobs, 5, &mat, a, b, c, d, 1),
                ];
                if standard {
                    got = vec![
                        fwd_local_sw_neon_u8_impl::<true, true, false>(
                            &jobs, 5, &mat, a, b, c, d, 1,
                        ),
                        fwd_local_sw_neon_u8_impl::<true, false, false>(
                            &jobs, 5, &mat, a, b, c, d, 1,
                        ),
                        fwd_local_sw_neon_u8_impl::<false, true, false>(
                            &jobs, 5, &mat, a, b, c, d, 1,
                        ),
                        fwd_local_sw_neon_u8_impl::<false, false, false>(
                            &jobs, 5, &mat, a, b, c, d, 1,
                        ),
                    ];
                }
                for (v, g) in got.iter().enumerate() {
                    assert_eq!(g, &want, "neon variant {v}, standard {standard}");
                }
            }
            // SAFETY: as in `matesw_mat5_meth_equals_scalar`, at the same u8 preconditions.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                assert_eq!(
                    fwd_local_sw_sse41_u8(&jobs, 5, &mat, a, b, c, d, 1),
                    want,
                    "sse41"
                );
                assert_eq!(
                    fwd_local_sw_avx2_u8(&jobs, 5, &mat, a, b, c, d, 1),
                    want,
                    "avx2"
                );
                if bwa_extend::has_avx512() {
                    let got = fwd_local_sw_avx512_u8(&jobs, 5, &mat, a, b, c, d, 1);
                    assert_eq!(got, want, "avx512 standard {standard}");
                }
            }
        }
    }

    /// The 5x5 table form (`MAT5`, `--meth`'s matrices) against the scalar kernel: OT and OB, with
    /// the freed cell neutral (0) and collapsed (`a`, mirrored), N on both sides, ragged windows,
    /// and queries planted with their C (or G) bisulfite-converted so the freed cell decides scores.
    /// `long` queries (250 bp and more) take the i16 kernels.
    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn matesw_mat5_meth_equals_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mut state = 0x5eed_0f_c0ffee_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // (ref base, read base) of the freed cell: OT C->T, OB G->A.
        for (r, q, long) in [
            (1usize, 3usize, false),
            (2, 0, false),
            (1, 3, true),
            (2, 0, true),
        ] {
            for collapsed in [false, true] {
                let mut mat = scmat(1, 4);
                mat[r * 5 + q] = if collapsed { 1 } else { 0 };
                if collapsed {
                    mat[q * 5 + r] = 1;
                }
                assert!(!mat_is_standard(5, &mat) && mat5_u8_ok(5, &mat));
                let mut qbufs: Vec<Vec<u8>> = Vec::new();
                let mut tbufs: Vec<Vec<u8>> = Vec::new();
                for k in 1..=40usize {
                    let (tlen, qlen) = if long {
                        (600 + k * 37, 250 + (next() % 151) as usize)
                    } else {
                        let tlen = 150 + k * 37;
                        (tlen, (40 + (next() % 111) as usize).min(tlen / 2))
                    };
                    let t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
                    let at = next() as usize % (tlen - qlen);
                    let qv: Vec<u8> = t[at..at + qlen]
                        .iter()
                        .map(|&b| {
                            if b as usize == r && next() % 2 == 0 {
                                q as u8
                            } else {
                                b
                            }
                        })
                        .collect();
                    qbufs.push(qv);
                    tbufs.push(t);
                }
                let jobs: Vec<FwdJob> = qbufs
                    .iter()
                    .zip(&tbufs)
                    .map(|(q, t)| FwdJob {
                        query: q,
                        target: t,
                        minsc: 19,
                        endsc: i32::MAX,
                    })
                    .collect();
                let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, 1);
                assert_eq!(
                    fwd_local_sw_batch(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, 1),
                    want
                );
                assert!(want.iter().filter(|w| w.0 >= 19).count() >= 30);
                let (a, b, c, d) = (o_del, e_del, o_ins, e_ins);
                if long {
                    // SAFETY: neon detected; i16 ceilings (queries under 401 at match 1);
                    // `mat5_u8_ok`.
                    #[cfg(target_arch = "aarch64")]
                    assert_eq!(
                        unsafe { fwd_local_sw_neon::<true>(&jobs, 5, &mat, a, b, c, d, 1) },
                        want
                    );
                    // Every x86 i16 tier. SAFETY: as for the u8 tiers below, with i16 ceilings.
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        assert_eq!(
                            fwd_local_sw_sse41_i16::<true>(&jobs, 5, &mat, a, b, c, d, 1),
                            want,
                            "sse41 i16"
                        );
                        assert_eq!(
                            fwd_local_sw_avx2_i16::<true>(&jobs, 5, &mat, a, b, c, d, 1),
                            want,
                            "avx2 i16"
                        );
                        if bwa_extend::has_avx512() {
                            let got =
                                fwd_local_sw_avx512_i16::<true>(&jobs, 5, &mat, a, b, c, d, 1);
                            assert_eq!(got, want, "avx512 i16");
                        }
                    }
                    continue;
                }
                // SAFETY: neon detected; queries and score ceilings under 250; `mat5_u8_ok`.
                #[cfg(target_arch = "aarch64")]
                let split = unsafe {
                    fwd_local_sw_neon_u8_impl::<true, false, true>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, 1,
                    )
                };
                #[cfg(target_arch = "aarch64")]
                assert_eq!(split, want);
                // Every x86 u8 tier, whatever the dispatch picks. SAFETY: SSE4.1 in every x86_64
                // this runs on, AVX2 in the executor (Rosetta is one), AVX-512 checked; the u8
                // preconditions as above.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    assert_eq!(
                        fwd_local_sw_sse41_u8(&jobs, 5, &mat, a, b, c, d, 1),
                        want,
                        "sse41"
                    );
                    assert_eq!(
                        fwd_local_sw_avx2_u8(&jobs, 5, &mat, a, b, c, d, 1),
                        want,
                        "avx2"
                    );
                    if bwa_extend::has_avx512() {
                        let got = fwd_local_sw_avx512_u8(&jobs, 5, &mat, a, b, c, d, 1);
                        assert_eq!(got, want, "avx512");
                    }
                }
            }
        }
    }
}

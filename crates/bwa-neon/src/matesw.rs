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
struct FwdJob<'a> {
    /// DP columns: 2-bit base codes, `<= 4`. On the forward pass this is [`KswJob::query`] verbatim;
    /// on the reverse pass it is the owned reversed prefix `query[..=qe]`, so it is at most as long.
    query: &'a [u8],
    /// DP rows: 2-bit base codes, `<= 4`. Forward pass: [`KswJob::target`]. Reverse pass: the owned
    /// reversed prefix `target[..=te]`.
    target: &'a [u8],
    /// 2nd-best candidate cutoff, in score units. See the doc above for how each pass sets it.
    minsc: i32,
    /// Early-stop score, in score units. See the doc above for how each pass sets it.
    endsc: i32,
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
    /// So the arithmetic is essentially done, and the remaining 36% is NOT in the DP body. It is in
    /// the per-row and per-group work around it: the 1.09x lane-divergence tax, the row epilogue's
    /// scalar sixteen-lane loop, the padded tail columns (1.42 instructions per cell against 1.11),
    /// and the group pack/extract. Anyone chasing the next percent should start there and should not
    /// try to shorten the cell recurrence.
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
    // ---- Pass 1: forward over all jobs. Finds the score and where each alignment ENDS. ----
    // Same sequences as `jobs`, with the pass-specific stop conditions attached: collect `score2`
    // candidates at `minsc`, and never stop early (`endsc = i32::MAX`) because the forward pass does
    // not yet know what score to aim for.
    let fwd_jobs: Vec<FwdJob> = jobs
        .iter()
        .map(|j| FwdJob {
            query: j.query,
            target: j.target,
            minsc,
            endsc: i32::MAX,
        })
        .collect();
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
        let mut ord: Vec<u32> = (0..fwd_jobs.len() as u32).collect();
        ord.sort_unstable_by_key(|&i| {
            let j = &fwd_jobs[i as usize];
            (j.target.len(), ksw_padded_qlen(j.query.len(), max_sc))
        });
        let sorted: Vec<FwdJob> = ord.iter().map(|&i| fwd_jobs[i as usize]).collect();
        let got = fwd_local_sw_batch(&sorted, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
        // Scatter back to caller order: every downstream index (`out`, pass 2, the caller's
        // orientation cursor) is in job order and must stay that way.
        let mut back = vec![(0i32, 0i32, 0i32, 0i32, 0i32); got.len()];
        for (slot, &i) in ord.iter().enumerate() {
            back[i as usize] = got[slot];
        }
        back
    } else {
        fwd_local_sw_batch(&fwd_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
    };

    // The final answers, complete except for the start coordinates: `qb`/`tb` stay at the -1 sentinel
    // until pass 2 fills them, and stay -1 forever for jobs pass 2 skips or disagrees with.
    let mut out: Vec<KswAlignResult> = fwd_results
        .iter()
        .map(|&(score, te, qe, score2, te2)| KswAlignResult {
            score,
            qb: -1,
            qe,
            tb: -1,
            te,
            score2,
            te2,
        })
        .collect();

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
            let q_off = rev_arena.len();
            rev_arena.extend(j.query[..=qe as usize].iter().rev().copied());
            let t_off = rev_arena.len();
            // Reversed, so the FIRST `rev_tlen(i)` bytes of the reversed prefix are the ones nearest
            // `te`, which is where the alignment ends and therefore where it must start being read.
            rev_arena.extend(
                j.target[..=te as usize]
                    .iter()
                    .rev()
                    .take(rev_tlen(i))
                    .copied(),
            );
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
    let rev_results = fwd_local_sw_batch(&rev_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
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
    }
    out
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
#[cfg(target_arch = "aarch64")]
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

#[allow(clippy::too_many_arguments)]
fn fwd_local_sw_batch(
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
        if std::arch::is_aarch64_feature_detected!("neon") && mat_is_standard(m, mat) {
            // Max reachable score per job = min(len) * match. Only the SCORE cells (H/E/F) live in the
            // SIMD vector; positions/te/qe are scalar i32, so window length is unconstrained. If every
            // job's score ceiling fits u8, run 16 lanes; else the i16 kernel at 8 lanes. Mate-rescue
            // jobs (short reads, match ~1) fit u8.
            // A local alignment can match at most `min(qlen, tlen)` bases and only loses score to
            // mismatches and gaps, so `min(len) * max_sc` is a hard ceiling on every H/E/F cell.
            let score_ceiling = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            // u8 also holds the argmax query column, so the query must be < 256 too.
            if jobs.iter().all(|j| {
                score_ceiling(j) < U8_SCORE_LIMIT && j.query.len() < U8_SCORE_LIMIT as usize
            }) {
                // SAFETY: neon detected; every H/E/F cell and query column < 250 fits u8; standard mat.
                return unsafe {
                    fwd_local_sw_neon_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
            }
            if jobs.iter().all(|j| {
                score_ceiling(j) < I16_SCORE_LIMIT && j.target.len() < I16_SCORE_LIMIT as usize
            }) {
                // SAFETY: neon detected; i16 range guaranteed; standard mat.
                return unsafe {
                    fwd_local_sw_neon(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if mat_is_standard(m, mat) {
            // Same score-ceiling test as the aarch64 arm above, verbatim: the threshold is bwa's
            // (`bwamem_pair.cpp:208`), not a property of the register we happen to run on, so it does
            // not change when the vector gets twice or four times as wide. See `ksw_padded_qlen` for
            // why deriving any of this from the SIMD width would alter the output.
            let score_ceiling = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            let fits_u8 = jobs.iter().all(|j| {
                score_ceiling(j) < U8_SCORE_LIMIT && j.query.len() < U8_SCORE_LIMIT as usize
            });
            let fits_i16 = jobs.iter().all(|j| {
                score_ceiling(j) < I16_SCORE_LIMIT && j.target.len() < I16_SCORE_LIMIT as usize
            });
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
            if want_avx512 && std::arch::is_x86_feature_detected!("avx512bw") {
                if fits_u8 {
                    // SAFETY: avx512bw detected; u8 preconditions hold; standard mat.
                    return unsafe {
                        fwd_local_sw_avx512_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: avx512bw detected; i16 range guaranteed; standard mat.
                    return unsafe {
                        fwd_local_sw_avx512_i16(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
            }
            if want_avx2 && std::arch::is_x86_feature_detected!("avx2") {
                if fits_u8 {
                    // SAFETY: avx2 detected; every H/E/F cell and query column < 250 fits u8; standard mat.
                    return unsafe {
                        fwd_local_sw_avx2_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: avx2 detected; i16 range guaranteed; standard mat.
                    return unsafe {
                        fwd_local_sw_avx2_i16(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
            }
            if want_sse41 && std::arch::is_x86_feature_detected!("sse4.1") {
                if fits_u8 {
                    // SAFETY: sse4.1 detected; u8 preconditions as for the AVX2 arm; standard mat.
                    return unsafe {
                        fwd_local_sw_sse41_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
                if fits_i16 {
                    // SAFETY: sse4.1 detected; i16 range guaranteed; standard mat.
                    return unsafe {
                        fwd_local_sw_sse41_i16(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                    };
                }
            }
        }
    }
    fwd_local_sw_scalar(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
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
fn mat_is_standard(m: usize, mat: &[i8]) -> bool {
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
    for l in 0..n_lanes {
        let best_score = gmax[l];
        let best_te = te[l];
        // No alignment found at all (te still -1) means there is no query end to report either.
        let best_qe = if best_te >= 0 { qe[l] } else { -1 };
        // score2: feed this lane's row maxima to ksw's `b` tracker, then take the best candidate
        // outside the exclusion window around `te`. The tracker is shared with the scalar
        // `ksw_local_fwd` precisely so the merge rule and the window cannot drift between them.
        let mut b = SuboptimalTracker::new();
        if limit[l] >= 0 {
            for i in 0..=limit[l] {
                // Best H anywhere in target row `i` for this lane. Rows past `limit[l]` were never
                // processed by this lane, so they are not offered as candidates.
                b.push_row(i, rowmax[i as usize * lanes + l].into(), minsc[l]);
            }
        }
        let (score2, te2) = b.finish(best_score, best_te, max_sc);
        out[group_idx * lanes + l] = (best_score, best_te, best_qe, score2, te2);
    }
}

/// NEON i16x8 forward local-SW. Vectorizes the [`fwd_local_sw_scalar`] control flow across `LANES`
/// jobs: the inner cell recurrence runs on `int16x8` (one lane per job), the per-row bookkeeping and
/// [`extract_group`] stay scalar. Requires the standard 5x5 `mat` (checked by the caller) so a cell
/// score is `match`/`mismatch`/`-1(N)` chosen by compares. Every value must fit i16 (caller-guarded).
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
unsafe fn fwd_local_sw_neon(
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

            // Padding-free column range; see `fwd_local_sw_neon_u8` for why dropping the ZPAD and
            // PAD blends below `n_fast` is exact (no live lane shows either code there, and a dead
            // target row's cells stay inside their own lane, where nothing reads them).
            for j in 0..n_fast {
                let q_v = load_codes(&seq_q, j * LANES);
                let eq = vceqq_s16(t_v, q_v);
                let n_mask = vorrq_u16(t_is_n, vceqq_s16(q_v, four_v));
                let mut sc = vbslq_s16(eq, mtch_v, mis_v);
                sc = vbslq_s16(n_mask, n_v, sc);
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
                let mut sc = vbslq_s16(eq, mtch_v, mis_v);
                // Either base is N: fixed -1 in bwa's matrix, not the mismatch penalty.
                sc = vbslq_s16(n_mask, n_v, sc);
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
unsafe fn fwd_local_sw_neon_u8_impl<const USQADD: bool, const SHARE_OE: bool>(
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
    // *magnitudes* here, not signed values: `mispen_v` and `one_v` are subtracted rather than added,
    // which is what lets the whole kernel stay unsigned.
    let zero = vdupq_n_u8(0);
    let one_v = vdupq_n_u8(1);
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
        tbl[0] = mtch; // match: +a
        for e in tbl.iter_mut().take(4).skip(1) {
            *e = mispen.wrapping_neg(); // mismatch: -b
        }
        for e in tbl.iter_mut().take(8).skip(4) {
            *e = 0xff; // query N (code 4) against a real base: bwa scores -1
        }
        tbl[8] = 0xff; // both N: N_TARGET XOR 4
        for e in tbl.iter_mut().take(16).skip(12) {
            *e = 0xff; // target N against a real base: N_TARGET XOR 0-3
        }
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
    macro_rules! ef {
        ($e_prev:expr, $f_prev:expr, $mfe:expr, $h:expr) => {
            if SHARE_OE {
                let hg = vqsubq_u8($h, oe_del_v);
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
                seq_t[r * LANES16 + l] = if b == 4 { N_TARGET } else { b };
            }
        }

        // Same DP state as the scalar path (H(i-1, .), H(i, .), the per-column E carry), narrowed to
        // u8 so a column of all 16 lanes is one `vld1q_u8`. Legal only because local SW keeps every
        // value in `[0, ceiling]` and the caller proved the ceiling is under 250.
        let mut h_prev = vec![0u8; qmax * LANES16];
        let mut h_cur = vec![0u8; qmax * LANES16];
        let mut e = vec![0u8; qmax * LANES16];
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
            ($row:expr, $imax:expr, $col:expr) => {{
                let row = $row;
                let mut imax_arr = [0u8; LANES16];
                let mut col_arr = [0u8; LANES16];
                vst1q_u8(imax_arr.as_mut_ptr(), $imax);
                vst1q_u8(col_arr.as_mut_ptr(), $col);
                // Publish the row in one store. Lanes that are out of target or already frozen get
                // a value written that the old code left untouched, and that is safe: a lane's row
                // maxima are read only for rows `0..=limit[l]`, and `limit[l]` is `tlen[l] - 1` or
                // the freeze row, so every slot this writes beyond the guard is one nothing reads.
                vst1q_u8(rowmax.as_mut_ptr().add(row * LANES16), $imax);
                for l in 0..n_lanes {
                    if row >= tlen[l] || frozen[l] {
                        continue;
                    }
                    // Job `l`'s best H anywhere in target row `row`, widened out of the lane.
                    let row_max = imax_arr[l] as i32;
                    if row_max > gmax[l] {
                        gmax[l] = row_max;
                        te[l] = row as i32;
                        qe[l] = col_arr[l] as i32;
                        if gmax[l] >= endsc[l] {
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
                let (mut col0, mut col1, mut col2, mut col3) = (zero, zero, zero, zero);
                let mut j_v = zero;

                // ---- Fast column range: no ZPAD, no PAD (see the one-row body below) ----------
                for j in 0..n_fast {
                    // The one load all four rows use.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));

                    // Row i. `e_v` is E(i, j), the only E that comes from memory.
                    let s0 = vqtbl1q_u8(score_tbl, veorq_u8(t0_v, q_v));
                    let diag0 = diag!(d0, s0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    // Row i+1, whose diagonal H(i, j-1) is the previous column's `h0` and whose E
                    // carry row i just produced in a register.
                    let s1 = vqtbl1q_u8(score_tbl, veorq_u8(t1_v, q_v));
                    let diag1 = diag!(d1, s1);
                    let mfe1 = vmaxq_u8(diag1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    // Row i+2.
                    let s2 = vqtbl1q_u8(score_tbl, veorq_u8(t2_v, q_v));
                    let diag2 = diag!(d2, s2);
                    let mfe2 = vmaxq_u8(diag2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    col2 = vbslq_u8(vcgtq_u8(h2, imax2), j_v, col2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    // Row i+3, the only one of the four whose H and E reach memory.
                    let s3 = vqtbl1q_u8(score_tbl, veorq_u8(t3_v, q_v));
                    let diag3 = diag!(d3, s3);
                    let mfe3 = vmaxq_u8(diag3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    col3 = vbslq_u8(vcgtq_u8(h3, imax3), j_v, col3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    j_v = vaddq_u8(j_v, one_v);
                }

                // ---- Tail: ZPAD / PAD possible, full logic, still four rows at a time ----------
                // `q_pad` is hoisted out of the four PAD tests: the pair body recomputes
                // `vtstq_u8(vorrq_u8(t, q), high_bit)` per row, but the query half of that OR is the
                // same for all four, so it is tested once and OR-ed with each row's target test.
                for j in n_fast..n_pad {
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    let zpad_mask = vceqq_u8(q_v, zpad_v);
                    let q_pad = vtstq_u8(q_v, high_bit_v);

                    let s0 = vqtbl1q_u8(score_tbl, veorq_u8(t0_v, q_v));
                    let scored0 = diag!(d0, s0);
                    let pad0 = vorrq_u8(q_pad, vtstq_u8(t0_v, high_bit_v));
                    let mut diag0 = vbslq_u8(zpad_mask, d0, scored0);
                    diag0 = vbslq_u8(pad0, zero, diag0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let s1 = vqtbl1q_u8(score_tbl, veorq_u8(t1_v, q_v));
                    let scored1 = diag!(d1, s1);
                    let pad1 = vorrq_u8(q_pad, vtstq_u8(t1_v, high_bit_v));
                    let mut diag1 = vbslq_u8(zpad_mask, d1, scored1);
                    diag1 = vbslq_u8(pad1, zero, diag1);
                    let mfe1 = vmaxq_u8(diag1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    let s2 = vqtbl1q_u8(score_tbl, veorq_u8(t2_v, q_v));
                    let scored2 = diag!(d2, s2);
                    let pad2 = vorrq_u8(q_pad, vtstq_u8(t2_v, high_bit_v));
                    let mut diag2 = vbslq_u8(zpad_mask, d2, scored2);
                    diag2 = vbslq_u8(pad2, zero, diag2);
                    let mfe2 = vmaxq_u8(diag2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    col2 = vbslq_u8(vcgtq_u8(h2, imax2), j_v, col2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    let s3 = vqtbl1q_u8(score_tbl, veorq_u8(t3_v, q_v));
                    let scored3 = diag!(d3, s3);
                    let pad3 = vorrq_u8(q_pad, vtstq_u8(t3_v, high_bit_v));
                    let mut diag3 = vbslq_u8(zpad_mask, d3, scored3);
                    diag3 = vbslq_u8(pad3, zero, diag3);
                    let mfe3 = vmaxq_u8(diag3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    col3 = vbslq_u8(vcgtq_u8(h3, imax3), j_v, col3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    j_v = vaddq_u8(j_v, one_v);
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
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e1, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let mfe1 = vmaxq_u8(d1, e1);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e2, f1_next) = ef!(e1, f1, mfe1, h1);
                    f1 = f1_next;
                    d1 = h0;

                    let mfe2 = vmaxq_u8(d2, e2);
                    let h2 = vmaxq_u8(mfe2, f2);
                    col2 = vbslq_u8(vcgtq_u8(h2, imax2), j_v, col2);
                    imax2 = vmaxq_u8(imax2, h2);
                    let (e3, f2_next) = ef!(e2, f2, mfe2, h2);
                    f2 = f2_next;
                    d2 = h1;

                    let mfe3 = vmaxq_u8(d3, e3);
                    let h3 = vmaxq_u8(mfe3, f3);
                    col3 = vbslq_u8(vcgtq_u8(h3, imax3), j_v, col3);
                    imax3 = vmaxq_u8(imax3, h3);
                    let (e_out, f3_next) = ef!(e3, f3, mfe3, h3);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f3 = f3_next;
                    d3 = h2;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h3);
                    j_v = vaddq_u8(j_v, one_v);
                }

                // Row order matters here: freezing at row `i` must suppress rows `i+1..=i+3`.
                finish_row!(i, imax0, col0);
                finish_row!(i + 1, imax1, col1);
                finish_row!(i + 2, imax2, col2);
                finish_row!(i + 3, imax3, col3);
            } else if rows == 2 {
                let t0_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
                let t1_v = vld1q_u8(seq_t.as_ptr().add((i + 1) * LANES16));
                // Two independent sets of row accumulators, one per row of the pair. `d0`/`d1` are
                // the diagonals: `d0` comes from `h_prev`, `d1` from the previous column's `h0`.
                let (mut f0, mut f1) = (zero, zero);
                let (mut d0, mut d1) = (zero, zero);
                let (mut imax0, mut imax1) = (zero, zero);
                let (mut col0, mut col1) = (zero, zero);
                let mut j_v = zero;

                // ---- Fast column range: no ZPAD, no PAD (see the one-row body below) ----------
                for j in 0..n_fast {
                    // The one load both rows use.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));

                    // Row i.
                    let s0 = vqtbl1q_u8(score_tbl, veorq_u8(t0_v, q_v));
                    let diag0 = diag!(d0, s0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    // E(i+1, j). Handed straight to row i+1 in a register instead of being stored
                    // and reloaded, which is one of the two memory ops this pairing removes.
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    // Row i+1, whose diagonal H(i, j-1) is the previous column's `h0`.
                    let s1 = vqtbl1q_u8(score_tbl, veorq_u8(t1_v, q_v));
                    let diag1 = diag!(d1, s1);
                    let mfe1 = vmaxq_u8(diag1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    j_v = vaddq_u8(j_v, one_v);
                }

                // ---- Tail: ZPAD / PAD possible, full logic, still two rows at a time -----------
                for j in n_fast..n_pad {
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    let zpad_mask = vceqq_u8(q_v, zpad_v);

                    let s0 = vqtbl1q_u8(score_tbl, veorq_u8(t0_v, q_v));
                    let scored0 = diag!(d0, s0);
                    let pad0 = vtstq_u8(vorrq_u8(t0_v, q_v), high_bit_v);
                    let mut diag0 = vbslq_u8(zpad_mask, d0, scored0);
                    diag0 = vbslq_u8(pad0, zero, diag0);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(diag0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let s1 = vqtbl1q_u8(score_tbl, veorq_u8(t1_v, q_v));
                    let scored1 = diag!(d1, s1);
                    let pad1 = vtstq_u8(vorrq_u8(t1_v, q_v), high_bit_v);
                    let mut diag1 = vbslq_u8(zpad_mask, d1, scored1);
                    diag1 = vbslq_u8(pad1, zero, diag1);
                    let mfe1 = vmaxq_u8(diag1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    j_v = vaddq_u8(j_v, one_v);
                }

                // ---- ksw profile padding: every live lane is ZPAD, so `diag = d` ---------------
                // See the quad body's version for why the score computation and the pad blend both
                // fold away here.
                for j in n_pad..qmax {
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe0 = vmaxq_u8(d0, e_v);
                    let h0 = vmaxq_u8(mfe0, f0);
                    col0 = vbslq_u8(vcgtq_u8(h0, imax0), j_v, col0);
                    imax0 = vmaxq_u8(imax0, h0);
                    let (e_mid, f0_next) = ef!(e_v, f0, mfe0, h0);
                    f0 = f0_next;
                    d0 = vld1q_u8(h_prev.as_ptr().add(j * LANES16));

                    let mfe1 = vmaxq_u8(d1, e_mid);
                    let h1 = vmaxq_u8(mfe1, f1);
                    col1 = vbslq_u8(vcgtq_u8(h1, imax1), j_v, col1);
                    imax1 = vmaxq_u8(imax1, h1);
                    let (e_out, f1_next) = ef!(e_mid, f1, mfe1, h1);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_out);
                    f1 = f1_next;
                    d1 = h0;
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h1);
                    j_v = vaddq_u8(j_v, one_v);
                }

                // Row order matters here: freezing at row `i` must suppress row `i+1`.
                finish_row!(i, imax0, col0);
                finish_row!(i + 1, imax1, col1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
                // Row accumulators, one lane per job, same invariant as the scalar version at the
                // top of column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j), and its smallest
                // attaining column.
                let mut f_v = zero;
                let mut h_diag_v = zero;
                let mut imax_v = zero;
                // Min query column achieving this row's max, tracked per cell. Recovering it lazily
                // instead (scan `h_cur` for the first cell equal to the row max, only in rows that
                // raise the lane's global max) was MEASURED and is 18% SLOWER: the global max climbs
                // in far more rows than the "few dozen" the idea assumed, and each recovery is a
                // 160-element strided scalar scan. Do not retry without first counting the rows
                // that improve `gmax`.
                let mut imax_col_v = zero;
                // Current query column, broadcast. Carried as a vector and incremented rather than
                // rebuilt with `vdupq_n_u8(j as u8)` per cell: the dup is a GPR-to-vector transfer
                // on the critical path of the argmax blend, the add is not.
                let mut j_v = zero;
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
                    let sbt = vqtbl1q_u8(score_tbl, veorq_u8(t_v, q_v));
                    let diag_v = diag!(h_diag_v, sbt);
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe = vmaxq_u8(diag_v, e_v);
                    let h_v = vmaxq_u8(mfe, f_v);
                    let is_new_row_max = vcgtq_u8(h_v, imax_v);
                    imax_col_v = vbslq_u8(is_new_row_max, j_v, imax_col_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);
                    let (e_new, f_next) = ef!(e_v, f_v, mfe, h_v);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new);
                    f_v = f_next;
                    h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
                    j_v = vaddq_u8(j_v, one_v);
                }

                // ---- Tail: the columns where ZPAD / PAD can appear, full logic ----------------
                for j in n_fast..n_pad {
                    // Lane `l` = query base at column `j` of job `l`.
                    let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                    // diag_v = max(0, h_diag + score). The substitution score comes from one table
                    // lookup on `target XOR query` (see `score_tbl`), biased so it applies as a
                    // single saturating add followed by a de-biasing saturating subtract. Both
                    // saturations floor the result at 0, so no explicit `max(0, .)` is needed.
                    let xor_v = veorq_u8(t_v, q_v);
                    let sbt = vqtbl1q_u8(score_tbl, xor_v);
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
                    // Strict `>`, so a tie keeps the earlier column. `j as u8` is why the caller
                    // caps the query at 250 bases: the column index shares the lane width with the
                    // scores.
                    let is_new_row_max = vcgtq_u8(h_v, imax_v);
                    imax_col_v = vbslq_u8(is_new_row_max, j_v, imax_col_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);

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
                    j_v = vaddq_u8(j_v, one_v);
                }

                // ---- ksw profile padding: every live lane is ZPAD, so `diag = h_diag` ----------
                // See the quad body's version for why the score computation and the pad blend both
                // fold away here.
                for j in n_pad..qmax {
                    let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                    let mfe = vmaxq_u8(h_diag_v, e_v);
                    let h_v = vmaxq_u8(mfe, f_v);
                    let is_new_row_max = vcgtq_u8(h_v, imax_v);
                    imax_col_v = vbslq_u8(is_new_row_max, j_v, imax_col_v);
                    imax_v = vmaxq_u8(imax_v, h_v);
                    vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);
                    let (e_new2, f_next2) = ef!(e_v, f_v, mfe, h_v);
                    vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new2);
                    f_v = f_next2;
                    h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
                    j_v = vaddq_u8(j_v, one_v);
                }

                finish_row!(i, imax_v, imax_col_v);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once every live lane is either frozen or out of target: purely a speed
            // win, as in `fwd_local_sw_neon`. See the longer note there.
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
    let share_oe = shareoe_enabled() && (o_del + e_del) == (o_ins + e_ins);
    match (usqadd, share_oe) {
        (true, true) => fwd_local_sw_neon_u8_impl::<true, true>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (true, false) => fwd_local_sw_neon_u8_impl::<true, false>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (false, true) => fwd_local_sw_neon_u8_impl::<false, true>(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc,
        ),
        (false, false) => fwd_local_sw_neon_u8_impl::<false, false>(
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

/// AVX2 `u8x32` forward local-SW: [`fwd_local_sw_neon_u8`] transliterated to `__m256i`, so x86_64
/// stops falling back to [`fwd_local_sw_scalar`] (measured at 2.22x the CPU of the vector path on a
/// whole paired-end run). Scope is u8 only; an i16 batch still takes the scalar path.
///
/// The transliteration is mechanical because every operation this kernel needs is element-wise, and
/// AVX2's element-wise byte ops behave exactly like NEON's despite the 128-bit-lane split that
/// afflicts its *shuffles*. Two operations have no direct AVX2 spelling and are emulated below with
/// the same trick `crate::batched`'s AVX2 kernels use:
///
/// - **unsigned compare**: AVX2 has only *signed* `_mm256_cmpgt_epi8`, which reads any score above
///   127 as negative. `cge_epu8(a, b)` is recovered as `max_epu8(a, b) == a`, and `cgt_epu8(a, b)`
///   as `!cge_epu8(b, a)`. Both are exact for all 256 byte values, including the ties that
///   `vcgtq_u8` must reject (the strict `>` on the row argmax is what keeps `qe` at the *smallest*
///   attaining column, `ksw.cpp:216-218`).
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
unsafe fn fwd_local_sw_avx2_u8(
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

    // --- the three NEON primitives AVX2 does not spell directly -------------------------------
    // `a >= b`, unsigned, as an all-ones/all-zero byte mask. `max_epu8` is a true unsigned max, so
    // `max(a,b) == a` is exactly `a >= b`; the signed `cmpgt_epi8` would misread scores above 127.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)
    }
    // `a > b`, unsigned: the complement of `b >= a`. `set1_epi8(-1)` is 0xFF per byte, i.e. an
    // all-ones vector, so the xor is a mask negation and not an arithmetic negation.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cgt_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(cge_epu8(b, a), _mm256_set1_epi8(-1))
    }
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

    // Broadcast constants, scores held as *magnitudes* (subtracted, not added) so the whole kernel
    // stays unsigned; see the NEON u8 kernel's header for why that is what makes u8 viable.
    let zero = _mm256_setzero_si256();
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
    let score_tbl = _mm256_loadu_si256(tbl.as_ptr() as *const __m256i);
    let bias_v = _mm256_set1_epi8(bias as i8);
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
    unsafe fn score_diag(
        tbl: __m256i,
        bias_v: __m256i,
        t_v: __m256i,
        q_v: __m256i,
        d: __m256i,
    ) -> __m256i {
        let s = _mm256_shuffle_epi8(tbl, _mm256_xor_si256(t_v, q_v));
        _mm256_subs_epu8(_mm256_adds_epu8(d, s), bias_v)
    }
    let one_v = _mm256_set1_epi8(1); // N penalty
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
                seq_q[c * LANES32 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES32 + l] = ZPAD;
            }
            // The N re-encoding, applied once per target base so the column loops never pay for
            // it (issue #43). See `fwd_local_sw_neon_u8`'s score-table comment.
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES32 + l] = if b == 4 { N_TARGET } else { b };
            }
        }

        // Same DP state as the scalar path, narrowed to u8 so a column of all 32 lanes is one
        // `_mm256_loadu_si256`. Legal only because local SW keeps every value in `[0, ceiling]` and
        // the caller proved the ceiling is under 250.
        let mut h_prev = vec![0u8; qmax * LANES32];
        let mut h_cur = vec![0u8; qmax * LANES32];
        let mut e = vec![0u8; qmax * LANES32];
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

        // =====================================================================================
        // Main DP, two target rows per iteration when two are left. Same transformation, and the
        // same byte-identity argument, as `fwd_local_sw_neon_u8`: rows `i` and `i+1` share the
        // query-column load, `h_prev[j]`, `e[j]` and the `h_cur[j]` store, because row `i+1`'s
        // diagonal is row `i`'s previous-column H and its E carry is what row `i` just produced.
        // =====================================================================================
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $col:expr) => {{
                let row = $row;
                let mut imax_arr = [0u8; LANES32];
                let mut col_arr = [0u8; LANES32];
                _mm256_storeu_si256(imax_arr.as_mut_ptr() as *mut __m256i, $imax);
                _mm256_storeu_si256(col_arr.as_mut_ptr() as *mut __m256i, $col);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm256_storeu_si256(
                    rowmax.as_mut_ptr().add(row * LANES32) as *mut __m256i,
                    $imax,
                );
                for l in 0..n_lanes {
                    if row >= tlen[l] || frozen[l] {
                        continue;
                    }
                    // Job `l`'s best H anywhere in target row `row`, widened out of the lane.
                    let row_max = imax_arr[l] as i32;
                    if row_max > gmax[l] {
                        gmax[l] = row_max;
                        te[l] = row as i32;
                        qe[l] = col_arr[l] as i32;
                        if gmax[l] >= endsc[l] {
                            frozen[l] = true;
                            limit[l] = row as i32;
                        }
                    }
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
                let (mut f0, mut f1) = (zero, zero);
                let (mut d0, mut d1) = (zero, zero);
                let (mut imax0, mut imax1) = (zero, zero);
                let (mut col0, mut col1) = (zero, zero);
                let mut j_v = zero;

                for j in 0..n_fast {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);

                    let diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe0 = _mm256_max_epu8(diag0, e_v);
                    let h0 = _mm256_max_epu8(mfe0, f0);
                    col0 = bsl(cgt_epu8(h0, imax0), j_v, col0);
                    imax0 = _mm256_max_epu8(imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid = _mm256_max_epu8(
                        _mm256_subs_epu8(e_v, e_del_v),
                        _mm256_subs_epu8(h0, oe_del_v),
                    );
                    f0 = _mm256_max_epu8(
                        _mm256_subs_epu8(f0, e_ins_v),
                        _mm256_subs_epu8(mfe0, oe_ins_v),
                    );
                    d0 = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);

                    let diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm256_max_epu8(diag1, e_mid);
                    let h1 = _mm256_max_epu8(mfe1, f1);
                    col1 = bsl(cgt_epu8(h1, imax1), j_v, col1);
                    imax1 = _mm256_max_epu8(imax1, h1);
                    _mm256_storeu_si256(
                        e.as_mut_ptr().add(j * LANES32) as *mut __m256i,
                        _mm256_max_epu8(
                            _mm256_subs_epu8(e_mid, e_del_v),
                            _mm256_subs_epu8(h1, oe_del_v),
                        ),
                    );
                    f1 = _mm256_max_epu8(
                        _mm256_subs_epu8(f1, e_ins_v),
                        _mm256_subs_epu8(mfe1, oe_ins_v),
                    );
                    d1 = h0;
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h1);
                    j_v = _mm256_add_epi8(j_v, one_v);
                }

                for j in n_fast..qmax {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
                    let zpad_mask = _mm256_cmpeq_epi8(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask in one
                    // instruction instead of three (issue #43).
                    let q_pad = _mm256_cmpgt_epi8(zero, q_v);

                    let mut diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = bsl(zpad_mask, d0, diag0);
                    diag0 = bsl(
                        _mm256_or_si256(_mm256_cmpgt_epi8(zero, t0_v), q_pad),
                        zero,
                        diag0,
                    );
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe0 = _mm256_max_epu8(diag0, e_v);
                    let h0 = _mm256_max_epu8(mfe0, f0);
                    col0 = bsl(cgt_epu8(h0, imax0), j_v, col0);
                    imax0 = _mm256_max_epu8(imax0, h0);
                    let e_mid = _mm256_max_epu8(
                        _mm256_subs_epu8(e_v, e_del_v),
                        _mm256_subs_epu8(h0, oe_del_v),
                    );
                    f0 = _mm256_max_epu8(
                        _mm256_subs_epu8(f0, e_ins_v),
                        _mm256_subs_epu8(mfe0, oe_ins_v),
                    );
                    d0 = _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);

                    let mut diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = bsl(zpad_mask, d1, diag1);
                    diag1 = bsl(
                        _mm256_or_si256(_mm256_cmpgt_epi8(zero, t1_v), q_pad),
                        zero,
                        diag1,
                    );
                    let mfe1 = _mm256_max_epu8(diag1, e_mid);
                    let h1 = _mm256_max_epu8(mfe1, f1);
                    col1 = bsl(cgt_epu8(h1, imax1), j_v, col1);
                    imax1 = _mm256_max_epu8(imax1, h1);
                    _mm256_storeu_si256(
                        e.as_mut_ptr().add(j * LANES32) as *mut __m256i,
                        _mm256_max_epu8(
                            _mm256_subs_epu8(e_mid, e_del_v),
                            _mm256_subs_epu8(h1, oe_del_v),
                        ),
                    );
                    f1 = _mm256_max_epu8(
                        _mm256_subs_epu8(f1, e_ins_v),
                        _mm256_subs_epu8(mfe1, oe_ins_v),
                    );
                    d1 = h0;
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h1);
                    j_v = _mm256_add_epi8(j_v, one_v);
                }

                // Row order matters: freezing at row `i` must suppress row `i+1`.
                finish_row!(i, imax0, col0);
                finish_row!(i + 1, imax1, col1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = _mm256_loadu_si256(seq_t.as_ptr().add(i * LANES32) as *const __m256i);
                // Row accumulators, one lane per job, same invariant as the scalar version at the top of
                // column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j), and its smallest attaining column.
                let mut f_v = zero;
                let mut h_diag_v = zero;
                let mut imax_v = zero;
                let mut imax_col_v = zero; // min query column achieving this row's max
                                           // Carried column index (bumped by `one_v`, which is set1_epi8(1)) and the row-invariant
                                           // "target is N" mask. See `fwd_local_sw_neon_u8` for the reasoning.
                let mut j_v = zero;

                // Padding-free column range: below `n_fast` no live lane shows ZPAD or PAD, so the two
                // padding masks and their selects are not emitted. Identical argument to the NEON u8
                // kernel, including why a dead target row may be left un-killed here.
                for j in 0..n_fast {
                    let q_v = _mm256_loadu_si256(seq_q.as_ptr().add(j * LANES32) as *const __m256i);
                    let diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    let mfe = _mm256_max_epu8(diag_v, e_v);
                    let h_v = _mm256_max_epu8(mfe, f_v);
                    let is_new_row_max = cgt_epu8(h_v, imax_v);
                    imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                    imax_v = _mm256_max_epu8(imax_v, h_v);
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h_v);
                    let e_new = _mm256_max_epu8(
                        _mm256_subs_epu8(e_v, e_del_v),
                        _mm256_subs_epu8(h_v, oe_del_v),
                    );
                    _mm256_storeu_si256(e.as_mut_ptr().add(j * LANES32) as *mut __m256i, e_new);
                    f_v = _mm256_max_epu8(
                        _mm256_subs_epu8(f_v, e_ins_v),
                        _mm256_subs_epu8(mfe, oe_ins_v),
                    );
                    h_diag_v =
                        _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);
                    j_v = _mm256_add_epi8(j_v, one_v);
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
                    let mut diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = bsl(zpad_mask, h_diag_v, diag_v); // score 0: diagonal passes through
                                                               // Dead padding: force 0 outright, as in the NEON u8 kernel.
                    diag_v = bsl(pad_mask, zero, diag_v);

                    // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                    let e_v = _mm256_loadu_si256(e.as_ptr().add(j * LANES32) as *const __m256i);
                    // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                    // saturation, so the two maxes are the whole H recurrence.
                    // `mfe = max(diag, E)` excludes the serial F carry, so the F recurrence below can read
                    // it instead of the full `h_v` and stop waiting on H -- the same critical-chain
                    // shortening as the NEON u8 kernel (f -> h -> f becomes f -> f).
                    let mfe = _mm256_max_epu8(diag_v, e_v);
                    let h_v = _mm256_max_epu8(mfe, f_v);
                    // Strict unsigned `>`, so a tie keeps the earlier `j`. This is the one place the
                    // signed `_mm256_cmpgt_epi8` would silently differ from NEON, for any score or column
                    // index above 127; `cgt_epu8` is why the caller's 250-base query cap still holds.
                    let is_new_row_max = cgt_epu8(h_v, imax_v);
                    imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                    imax_v = _mm256_max_epu8(imax_v, h_v);
                    _mm256_storeu_si256(h_cur.as_mut_ptr().add(j * LANES32) as *mut __m256i, h_v);

                    // e = max(0, e-e_del, h-oe_del); both saturating subs supply their own clamp.
                    let e_new = _mm256_max_epu8(
                        _mm256_subs_epu8(e_v, e_del_v),
                        _mm256_subs_epu8(h_v, oe_del_v),
                    );
                    _mm256_storeu_si256(e.as_mut_ptr().add(j * LANES32) as *mut __m256i, e_new);
                    // F, the row's serial carry, reassociated off `mfe` (see the NEON u8 kernel for the
                    // byte-identity argument: oe_ins >= e_ins makes the `f - oe_ins` branch inside H
                    // dominated, so `max(f - e_ins, mfe - oe_ins)` equals the original).
                    f_v = _mm256_max_epu8(
                        _mm256_subs_epu8(f_v, e_ins_v),
                        _mm256_subs_epu8(mfe, oe_ins_v),
                    );
                    h_diag_v =
                        _mm256_loadu_si256(h_prev.as_ptr().add(j * LANES32) as *const __m256i);
                    j_v = _mm256_add_epi8(j_v, one_v);
                }
                finish_row!(i, imax_v, imax_col_v);
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
            n_lanes, group_idx, LANES32, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
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
unsafe fn fwd_local_sw_avx2_i16(
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
    let m_v = _mm256_set1_epi16(m as i16);
    let zpad_v = _mm256_set1_epi16(ZPAD as i16);
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
                seq_q[c * LANES16 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES16 + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = b;
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

        // =====================================================================================
        // Main DP, one target row per iteration. Structurally identical to `fwd_local_sw_neon`, at
        // 16 lanes; only the per-cell arithmetic is vectorised, the bookkeeping stays scalar.
        // =====================================================================================
        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES16);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
                                       // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm256_cmpeq_epi16(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let q_v = load_codes(&seq_q, j * LANES16);
                let eq = _mm256_cmpeq_epi16(t_v, q_v);
                let n_mask = _mm256_or_si256(t_is_n, _mm256_cmpeq_epi16(q_v, four_v));
                let mut sc = bsl(eq, mtch_v, mis_v);
                sc = bsl(n_mask, n_v, sc);
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
                    _mm256_or_si256(cge_epi16(t_v, m_v), _mm256_cmpgt_epi16(q_v, zpad_v));
                let mut sc = bsl(eq, mtch_v, mis_v);
                sc = bsl(n_mask, n_v, sc);
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
/// - **unsigned compare**: SSE4.1 has only *signed* `_mm_cmpgt_epi8`, which reads any score above
///   127 as negative. `cge_epu8(a, b)` is recovered as `max_epu8(a, b) == a`, and `cgt_epu8(a, b)`
///   as `!cge_epu8(b, a)`. Both are exact for all 256 byte values, including the ties that
///   `vcgtq_u8` must reject (the strict `>` on the row argmax is what keeps `qe` at the *smallest*
///   attaining column, `ksw.cpp:216-218`).
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
unsafe fn fwd_local_sw_sse41_u8(
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

    // --- the three NEON primitives SSE4.1 does not spell directly -------------------------------
    // `a >= b`, unsigned, as an all-ones/all-zero byte mask. `max_epu8` is a true unsigned max, so
    // `max(a,b) == a` is exactly `a >= b`; the signed `cmpgt_epi8` would misread scores above 127.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn cge_epu8(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)
    }
    // `a > b`, unsigned: the complement of `b >= a`. `set1_epi8(-1)` is 0xFF per byte, i.e. an
    // all-ones vector, so the xor is a mask negation and not an arithmetic negation.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn cgt_epu8(a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(cge_epu8(b, a), _mm_set1_epi8(-1))
    }
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

    // Broadcast constants, scores held as *magnitudes* (subtracted, not added) so the whole kernel
    // stays unsigned; see the NEON u8 kernel's header for why that is what makes u8 viable.
    let zero = _mm_setzero_si128();
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
    let score_tbl = _mm_loadu_si128(tbl.as_ptr() as *const __m128i);
    let bias_v = _mm_set1_epi8(bias as i8);
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
    unsafe fn score_diag(
        tbl: __m128i,
        bias_v: __m128i,
        t_v: __m128i,
        q_v: __m128i,
        d: __m128i,
    ) -> __m128i {
        let s = _mm_shuffle_epi8(tbl, _mm_xor_si128(t_v, q_v));
        _mm_subs_epu8(_mm_adds_epu8(d, s), bias_v)
    }
    let one_v = _mm_set1_epi8(1); // N penalty
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
                seq_q[c * LANES16 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES16 + l] = ZPAD;
            }
            // The N re-encoding, applied once per target base so the column loops never pay for
            // it (issue #43). See `fwd_local_sw_neon_u8`'s score-table comment.
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = if b == 4 { N_TARGET } else { b };
            }
        }

        // Same DP state as the scalar path, narrowed to u8 so a column of all 16 lanes is one
        // `_mm_loadu_si128`. Legal only because local SW keeps every value in `[0, ceiling]` and
        // the caller proved the ceiling is under 250.
        let mut h_prev = vec![0u8; qmax * LANES16];
        let mut h_cur = vec![0u8; qmax * LANES16];
        let mut e = vec![0u8; qmax * LANES16];
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
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $col:expr) => {{
                let row = $row;
                let mut imax_arr = [0u8; LANES16];
                let mut col_arr = [0u8; LANES16];
                _mm_storeu_si128(imax_arr.as_mut_ptr() as *mut __m128i, $imax);
                _mm_storeu_si128(col_arr.as_mut_ptr() as *mut __m128i, $col);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm_storeu_si128(
                    rowmax.as_mut_ptr().add(row * LANES16) as *mut __m128i,
                    $imax,
                );
                for l in 0..n_lanes {
                    if row >= tlen[l] || frozen[l] {
                        continue;
                    }
                    // Job `l`'s best H anywhere in target row `row`, widened out of the lane.
                    let row_max = imax_arr[l] as i32;
                    if row_max > gmax[l] {
                        gmax[l] = row_max;
                        te[l] = row as i32;
                        qe[l] = col_arr[l] as i32;
                        if gmax[l] >= endsc[l] {
                            frozen[l] = true;
                            limit[l] = row as i32;
                        }
                    }
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
                let (mut f0, mut f1) = (zero, zero);
                let (mut d0, mut d1) = (zero, zero);
                let (mut imax0, mut imax1) = (zero, zero);
                let (mut col0, mut col1) = (zero, zero);
                let mut j_v = zero;

                for j in 0..n_fast {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);

                    let diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe0 = _mm_max_epu8(diag0, e_v);
                    let h0 = _mm_max_epu8(mfe0, f0);
                    col0 = bsl(cgt_epu8(h0, imax0), j_v, col0);
                    imax0 = _mm_max_epu8(imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid =
                        _mm_max_epu8(_mm_subs_epu8(e_v, e_del_v), _mm_subs_epu8(h0, oe_del_v));
                    f0 = _mm_max_epu8(_mm_subs_epu8(f0, e_ins_v), _mm_subs_epu8(mfe0, oe_ins_v));
                    d0 = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);

                    let diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm_max_epu8(diag1, e_mid);
                    let h1 = _mm_max_epu8(mfe1, f1);
                    col1 = bsl(cgt_epu8(h1, imax1), j_v, col1);
                    imax1 = _mm_max_epu8(imax1, h1);
                    _mm_storeu_si128(
                        e.as_mut_ptr().add(j * LANES16) as *mut __m128i,
                        _mm_max_epu8(_mm_subs_epu8(e_mid, e_del_v), _mm_subs_epu8(h1, oe_del_v)),
                    );
                    f1 = _mm_max_epu8(_mm_subs_epu8(f1, e_ins_v), _mm_subs_epu8(mfe1, oe_ins_v));
                    d1 = h0;
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h1);
                    j_v = _mm_add_epi8(j_v, one_v);
                }

                for j in n_fast..qmax {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
                    let zpad_mask = _mm_cmpeq_epi8(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask in one
                    // instruction instead of three (issue #43).
                    let q_pad = _mm_cmpgt_epi8(zero, q_v);

                    let mut diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = bsl(zpad_mask, d0, diag0);
                    diag0 = bsl(_mm_or_si128(_mm_cmpgt_epi8(zero, t0_v), q_pad), zero, diag0);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe0 = _mm_max_epu8(diag0, e_v);
                    let h0 = _mm_max_epu8(mfe0, f0);
                    col0 = bsl(cgt_epu8(h0, imax0), j_v, col0);
                    imax0 = _mm_max_epu8(imax0, h0);
                    let e_mid =
                        _mm_max_epu8(_mm_subs_epu8(e_v, e_del_v), _mm_subs_epu8(h0, oe_del_v));
                    f0 = _mm_max_epu8(_mm_subs_epu8(f0, e_ins_v), _mm_subs_epu8(mfe0, oe_ins_v));
                    d0 = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);

                    let mut diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = bsl(zpad_mask, d1, diag1);
                    diag1 = bsl(_mm_or_si128(_mm_cmpgt_epi8(zero, t1_v), q_pad), zero, diag1);
                    let mfe1 = _mm_max_epu8(diag1, e_mid);
                    let h1 = _mm_max_epu8(mfe1, f1);
                    col1 = bsl(cgt_epu8(h1, imax1), j_v, col1);
                    imax1 = _mm_max_epu8(imax1, h1);
                    _mm_storeu_si128(
                        e.as_mut_ptr().add(j * LANES16) as *mut __m128i,
                        _mm_max_epu8(_mm_subs_epu8(e_mid, e_del_v), _mm_subs_epu8(h1, oe_del_v)),
                    );
                    f1 = _mm_max_epu8(_mm_subs_epu8(f1, e_ins_v), _mm_subs_epu8(mfe1, oe_ins_v));
                    d1 = h0;
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h1);
                    j_v = _mm_add_epi8(j_v, one_v);
                }

                // Row order matters: freezing at row `i` must suppress row `i+1`.
                finish_row!(i, imax0, col0);
                finish_row!(i + 1, imax1, col1);
            } else {
                // Lane `l` = target base at row `i` of job `l`.
                let t_v = _mm_loadu_si128(seq_t.as_ptr().add(i * LANES16) as *const __m128i);
                // Row accumulators, one lane per job, same invariant as the scalar version at the top of
                // column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j), and its smallest attaining column.
                let mut f_v = zero;
                let mut h_diag_v = zero;
                let mut imax_v = zero;
                let mut imax_col_v = zero; // min query column achieving this row's max
                                           // Carried column index (bumped by `one_v`, which is set1_epi8(1)) and the row-invariant
                                           // "target is N" mask. See `fwd_local_sw_neon_u8` for the reasoning.
                let mut j_v = zero;

                // Padding-free column range: below `n_fast` no live lane shows ZPAD or PAD, so the two
                // padding masks and their selects are not emitted. Identical argument to the NEON u8
                // kernel, including why a dead target row may be left un-killed here.
                for j in 0..n_fast {
                    let q_v = _mm_loadu_si128(seq_q.as_ptr().add(j * LANES16) as *const __m128i);
                    let diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    let mfe = _mm_max_epu8(diag_v, e_v);
                    let h_v = _mm_max_epu8(mfe, f_v);
                    let is_new_row_max = cgt_epu8(h_v, imax_v);
                    imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                    imax_v = _mm_max_epu8(imax_v, h_v);
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h_v);
                    let e_new =
                        _mm_max_epu8(_mm_subs_epu8(e_v, e_del_v), _mm_subs_epu8(h_v, oe_del_v));
                    _mm_storeu_si128(e.as_mut_ptr().add(j * LANES16) as *mut __m128i, e_new);
                    f_v = _mm_max_epu8(_mm_subs_epu8(f_v, e_ins_v), _mm_subs_epu8(mfe, oe_ins_v));
                    h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);
                    j_v = _mm_add_epi8(j_v, one_v);
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
                    let mut diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = bsl(zpad_mask, h_diag_v, diag_v); // score 0: diagonal passes through
                                                               // Dead padding: force 0 outright, as in the NEON u8 kernel.
                    diag_v = bsl(pad_mask, zero, diag_v);

                    // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                    let e_v = _mm_loadu_si128(e.as_ptr().add(j * LANES16) as *const __m128i);
                    // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                    // saturation, so the two maxes are the whole H recurrence.
                    // `mfe = max(diag, E)` excludes the serial F carry, so the F recurrence below can read
                    // it instead of the full `h_v` and stop waiting on H -- the same critical-chain
                    // shortening as the NEON u8 kernel (f -> h -> f becomes f -> f).
                    let mfe = _mm_max_epu8(diag_v, e_v);
                    let h_v = _mm_max_epu8(mfe, f_v);
                    // Strict unsigned `>`, so a tie keeps the earlier `j`. This is the one place the
                    // signed `_mm_cmpgt_epi8` would silently differ from NEON, for any score or column
                    // index above 127; `cgt_epu8` is why the caller's 250-base query cap still holds.
                    let is_new_row_max = cgt_epu8(h_v, imax_v);
                    imax_col_v = bsl(is_new_row_max, j_v, imax_col_v);
                    imax_v = _mm_max_epu8(imax_v, h_v);
                    _mm_storeu_si128(h_cur.as_mut_ptr().add(j * LANES16) as *mut __m128i, h_v);

                    // e = max(0, e-e_del, h-oe_del); both saturating subs supply their own clamp.
                    let e_new =
                        _mm_max_epu8(_mm_subs_epu8(e_v, e_del_v), _mm_subs_epu8(h_v, oe_del_v));
                    _mm_storeu_si128(e.as_mut_ptr().add(j * LANES16) as *mut __m128i, e_new);
                    // F, the row's serial carry, reassociated off `mfe` (see the NEON u8 kernel for the
                    // byte-identity argument: oe_ins >= e_ins makes the `f - oe_ins` branch inside H
                    // dominated, so `max(f - e_ins, mfe - oe_ins)` equals the original).
                    f_v = _mm_max_epu8(_mm_subs_epu8(f_v, e_ins_v), _mm_subs_epu8(mfe, oe_ins_v));
                    h_diag_v = _mm_loadu_si128(h_prev.as_ptr().add(j * LANES16) as *const __m128i);
                    j_v = _mm_add_epi8(j_v, one_v);
                }
                finish_row!(i, imax_v, imax_col_v);
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
unsafe fn fwd_local_sw_sse41_i16(
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
    let m_v = _mm_set1_epi16(m as i16);
    let zpad_v = _mm_set1_epi16(ZPAD as i16);
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
                seq_q[c * LANES + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = b;
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

        // =====================================================================================
        // Main DP, one target row per iteration. Structurally identical to `fwd_local_sw_neon`, at
        // 8 lanes; only the per-cell arithmetic is vectorised, the bookkeeping stays scalar.
        // =====================================================================================
        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
                                       // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm_cmpeq_epi16(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let q_v = load_codes(&seq_q, j * LANES);
                let eq = _mm_cmpeq_epi16(t_v, q_v);
                let n_mask = _mm_or_si128(t_is_n, _mm_cmpeq_epi16(q_v, four_v));
                let mut sc = bsl(eq, mtch_v, mis_v);
                sc = bsl(n_mask, n_v, sc);
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
                let pad_mask = _mm_or_si128(cge_epi16(t_v, m_v), _mm_cmpgt_epi16(q_v, zpad_v));
                let mut sc = bsl(eq, mtch_v, mis_v);
                sc = bsl(n_mask, n_v, sc);
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
/// Whether this process runs the AVX-512 u8 rescue kernel with its leaf maxes spelled as
/// `vpcmpub` + `vpblendmb` instead of `vpmaxub` (ISSUE #44 part B), decided once by timing both.
///
/// The lever is real but **not portable**, which is why it is measured rather than assumed. On
/// Golden Cove every 512-bit saturating-integer and max op is single-ported (p0, TP 1.00), so the
/// kernel's cost is the count of p0-only ops and moving two of them per row onto p5/p05 is worth
/// about a sixth of the body. On Zen 5 the same rewrite is a loss: `vpmaxub` runs on all four FP
/// pipes at TP 0.25 while `vpcmpub` costs latency 6. A CPUID model match would have to be updated
/// for every part ever shipped; a stopwatch does not.
///
/// This mirrors `batched::extend_tier`'s calibration in shape and in caution: best of three on a
/// fixed synthetic batch, and the new spelling has to be more than 3% faster to be taken, so noise
/// on a normal host cannot flip a kernel that is already correct and fast. Both spellings are
/// byte-identical to the scalar reference (`avx512_verify::avx512_matesw_u8_matches_scalar` runs
/// both), so this decides speed and nothing else.
///
/// `BWA4_AVX512_CMPBLEND=1|0` skips the timing and names the spelling, for A/B runs.
///
/// # Returns
///
/// True to use the compare-and-blend spelling.
#[cfg(target_arch = "x86_64")]
fn avx512_leaf_cmp_blend() -> bool {
    use std::sync::OnceLock;
    static CHOICE: OnceLock<bool> = OnceLock::new();
    *CHOICE.get_or_init(|| {
        match std::env::var("BWA4_AVX512_CMPBLEND").as_deref() {
            Ok("1") => return true,
            Ok("0") => return false,
            _ => {}
        }
        if !std::arch::is_x86_feature_detected!("avx512bw") {
            return false;
        }
        // Mate-rescue shape, not a generic one: 150 bp query against a 500 bp insert window, with
        // the query planted in the target so the batch actually walks the full body instead of
        // dying to the score floor in the first rows.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let (qlen, tlen) = (150usize, 500usize);
        let mut qbufs: Vec<Vec<u8>> = Vec::with_capacity(64);
        let mut tbufs: Vec<Vec<u8>> = Vec::with_capacity(64);
        for _ in 0..64 {
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let at = next() % (tlen - qlen);
            t[at..at + qlen].copy_from_slice(&q);
            qbufs.push(q);
            tbufs.push(t);
        }
        // bwa's default DNA scores, so the calibration measures the scheme the run will use.
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { 1 } else { -4 };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        let jobs: Vec<FwdJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| FwdJob {
                query: q.as_slice(),
                target: t.as_slice(),
                minsc: 19,
                endsc: i32::MAX,
            })
            .collect();
        // Best of three, for the same reason the extension tier takes the best rather than the
        // mean: the fastest observation is the one least polluted by whatever else the box is doing.
        let time = |f: &dyn Fn()| {
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                f();
                best = best.min(t0.elapsed());
            }
            best
        };
        // SAFETY of both closures: avx512bw was just detected, and these are stock-DNA-scored jobs
        // whose ceiling is far below `U8_SCORE_LIMIT`, which is the u8 kernel's precondition.
        let plain = time(&|| {
            let out =
                unsafe { fwd_local_sw_avx512_u8_impl::<false>(&jobs, 5, &mat, 6, 1, 6, 1, 1) };
            std::hint::black_box(&out);
        });
        let blended = time(&|| {
            let out = unsafe { fwd_local_sw_avx512_u8_impl::<true>(&jobs, 5, &mat, 6, 1, 6, 1, 1) };
            std::hint::black_box(&out);
        });
        blended.as_secs_f64() < plain.as_secs_f64() * 0.97
    })
}

/// The AVX-512 u8 rescue kernel, in whichever spelling this host timed as faster.
///
/// # Parameters / Returns
/// As [`fwd_local_sw_avx512_u8_impl`], which does the work; this only picks `LEAF_CMP_BLEND`.
///
/// # Safety
/// Caller must have confirmed AVX-512BW is available, as for the implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
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
    if avx512_leaf_cmp_blend() {
        fwd_local_sw_avx512_u8_impl::<true>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
    } else {
        fwd_local_sw_avx512_u8_impl::<false>(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
    }
}

/// AVX512 `u8x64` forward local-SW: [`fwd_local_sw_avx2_u8`] widened to `__m512i`, 64 rescue jobs per
/// group instead of 32. Mate-rescue jobs are cross-lane independent and near-uniform in size (query =
/// read length, target = the insert-size window), so doubling the lane count keeps utilisation high;
/// the kernel is throughput-bound after the F reassociation, which is the regime a wider vector helps.
///
/// AVX-512 changes the *spelling* of two things relative to AVX2, and both make the code simpler, not
/// harder:
///
/// - **Comparisons return mask registers** (`__mmask64`), not vector masks. Unsigned compares are
///   therefore *native*: `_mm512_cmpge_epu8_mask` / `_mm512_cmpgt_epu8_mask` replace AVX2's
///   `max_epu8 == a` recovery, and there is no 127-boundary hazard to reason about.
/// - **Blends take the mask directly**: `_mm512_mask_blend_epi8(k, a, b)` is `k ? b : a`, so the NEON
///   `vbslq(mask, a, b) = mask ? a : b` becomes `_mm512_mask_blend_epi8(mask, b, a)` (operands swapped,
///   same convention as the AVX2 `bsl`).
///
/// Everything else is identical to the AVX2 u8 kernel: saturating byte arithmetic supplies the
/// `max(0, .)` clamps, scores are stored as magnitudes so the whole kernel stays unsigned, the F
/// recurrence is reassociated off `mfe`, and the argmax column rides in the same byte lane as the
/// scores (so the caller's 250-base query cap still holds at 64 lanes).
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
#[target_feature(enable = "avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx512_u8_impl<const LEAF_CMP_BLEND: bool>(
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
    let score_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tbl.as_ptr() as *const __m128i));
    let bias_v = _mm512_set1_epi8(bias as i8);
    // The `vpshufb` / `vqtbl1q_u8` reachable-index argument is the one spelled out in
    // `fwd_local_sw_avx2_u8`; the encoding and therefore the argument are the same.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn score_diag(
        tbl: __m512i,
        bias_v: __m512i,
        t_v: __m512i,
        q_v: __m512i,
        d: __m512i,
    ) -> __m512i {
        let s = _mm512_shuffle_epi8(tbl, _mm512_xor_si512(t_v, q_v));
        _mm512_subs_epu8(_mm512_adds_epu8(d, s), bias_v)
    }
    /// One unsigned byte max, in either of its two spellings (ISSUE #44 part B, the paid half).
    ///
    /// `vpmaxub` ZMM is one uop on **p0 only** at throughput 1.00 on Golden Cove, because at 512
    /// bits port 1's vector ALU folds into p0; `vpcmpub` is p5-only and `vpblendmb` is p05 at
    /// 2/cycle. So paying one extra instruction to move a max off the port that binds is a win
    /// wherever p0 is the constraint, and a loss where it is not: on Zen 5 all four FP pipes take
    /// `vpmaxub` at TP 0.25 while `vpcmpub` costs latency 6. Hence `LEAF_CMP_BLEND` is decided by
    /// timing, in [`avx512_leaf_cmp_blend`], never by a CPUID model match.
    ///
    /// Only ever applied to the E and F updates, which are LEAVES: their results feed the next
    /// column, not this cell's `h`. `max(diag, e)` and `max(mfe, f)` sit on the cross-row
    /// `h0 -> e_mid -> mfe1 -> h1` chain, where 3 + 3 cycles of compare-then-blend latency would
    /// replace `vpmaxub`'s 1 and lose whatever the ports gained.
    ///
    /// Byte-identical to `max_epu8` by definition: the strict `>` mask selects `a` exactly where
    /// `a > b`, and `b` (which is then `>= a`) everywhere else.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn leaf_max<const CMP_BLEND: bool>(a: __m512i, b: __m512i) -> __m512i {
        if CMP_BLEND {
            _mm512_mask_blend_epi8(_mm512_cmpgt_epu8_mask(a, b), b, a)
        } else {
            _mm512_max_epu8(a, b)
        }
    }
    let one_v = _mm512_set1_epi8(1);
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

        let mut seq_q = vec![PAD; qmax * LANES64];
        let mut seq_t = vec![PAD; tmax * LANES64];
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
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES64 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES64 + l] = ZPAD;
            }
            // The N re-encoding, once per target base (issue #44 part A).
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES64 + l] = if b == 4 { N_TARGET } else { b };
            }
        }

        let mut h_prev = vec![0u8; qmax * LANES64];
        let mut h_cur = vec![0u8; qmax * LANES64];
        let mut e = vec![0u8; qmax * LANES64];
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
        let mut te = [-1i32; LANES64];
        let mut qe = [0i32; LANES64];
        let mut limit = [-1i32; LANES64];
        let mut frozen = [false; LANES64];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }
        // Columns below the shortest live query: no live lane can be showing ZPAD or PAD there.
        let n_fast = if fastcol_enabled() {
            qlen[..n_lanes].iter().copied().min().unwrap_or(0).min(qmax)
        } else {
            0
        };

        // Two target rows per iteration when two are left; see `fwd_local_sw_neon_u8` for the
        // memory-traffic argument and the byte-identity argument.
        macro_rules! finish_row {
            ($row:expr, $imax:expr, $col:expr) => {{
                let row = $row;
                let mut imax_arr = [0u8; LANES64];
                let mut col_arr = [0u8; LANES64];
                _mm512_storeu_si512(imax_arr.as_mut_ptr() as *mut __m512i, $imax);
                _mm512_storeu_si512(col_arr.as_mut_ptr() as *mut __m512i, $col);
                // The whole row, all lanes, one store. See the note on `rowmax`.
                _mm512_storeu_si512(
                    rowmax.as_mut_ptr().add(row * LANES64) as *mut __m512i,
                    $imax,
                );
                for l in 0..n_lanes {
                    if row >= tlen[l] || frozen[l] {
                        continue;
                    }
                    let row_max = imax_arr[l] as i32;
                    if row_max > gmax[l] {
                        gmax[l] = row_max;
                        te[l] = row as i32;
                        qe[l] = col_arr[l] as i32;
                        if gmax[l] >= endsc[l] {
                            frozen[l] = true;
                            limit[l] = row as i32;
                        }
                    }
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
                let (mut f0, mut f1) = (zero, zero);
                let (mut d0, mut d1) = (zero, zero);
                let (mut imax0, mut imax1) = (zero, zero);
                let (mut col0, mut col1) = (zero, zero);
                let mut j_v = zero;

                for j in 0..n_fast {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);

                    let diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe0 = _mm512_max_epu8(diag0, e_v);
                    let h0 = _mm512_max_epu8(mfe0, f0);
                    // ISSUE #44 part B, the free swap: the strict `>` mask is already needed for `col0`, and
                    // `max_epu8(a, b)` IS `mask_blend(cmpgt_epu8(b, a), a, b)`, so reusing it costs no new
                    // instruction and moves the max off p0, which is the port that binds at 512 bits, onto
                    // p05 where `VPBLENDMB` runs at 2/cycle. Byte-identical by definition of max.
                    let gt = _mm512_cmpgt_epu8_mask(h0, imax0);
                    col0 = _mm512_mask_blend_epi8(gt, col0, j_v);
                    imax0 = _mm512_mask_blend_epi8(gt, imax0, h0);
                    // E(i+1, j), handed to row i+1 in a register rather than stored and reloaded.
                    let e_mid = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(e_v, e_del_v),
                        _mm512_subs_epu8(h0, oe_del_v),
                    );
                    f0 = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f0, e_ins_v),
                        _mm512_subs_epu8(mfe0, oe_ins_v),
                    );
                    d0 = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);

                    let diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    let mfe1 = _mm512_max_epu8(diag1, e_mid);
                    let h1 = _mm512_max_epu8(mfe1, f1);
                    // ISSUE #44 part B, the free swap: the strict `>` mask is already needed for `col1`, and
                    // `max_epu8(a, b)` IS `mask_blend(cmpgt_epu8(b, a), a, b)`, so reusing it costs no new
                    // instruction and moves the max off p0, which is the port that binds at 512 bits, onto
                    // p05 where `VPBLENDMB` runs at 2/cycle. Byte-identical by definition of max.
                    let gt = _mm512_cmpgt_epu8_mask(h1, imax1);
                    col1 = _mm512_mask_blend_epi8(gt, col1, j_v);
                    imax1 = _mm512_mask_blend_epi8(gt, imax1, h1);
                    _mm512_storeu_si512(
                        e.as_mut_ptr().add(j * LANES64) as *mut __m512i,
                        leaf_max::<LEAF_CMP_BLEND>(
                            _mm512_subs_epu8(e_mid, e_del_v),
                            _mm512_subs_epu8(h1, oe_del_v),
                        ),
                    );
                    f1 = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f1, e_ins_v),
                        _mm512_subs_epu8(mfe1, oe_ins_v),
                    );
                    d1 = h0;
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h1);
                    j_v = _mm512_add_epi8(j_v, one_v);
                }

                for j in n_fast..qmax {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);
                    let zpad_mask = _mm512_cmpeq_epi8_mask(q_v, zpad_v);
                    // `q > ZPAD` is true only for `PAD`, so the high-bit test is the same mask.
                    let q_pad = _mm512_movepi8_mask(q_v);

                    let mut diag0 = score_diag(score_tbl, bias_v, t0_v, q_v, d0);
                    diag0 = _mm512_mask_blend_epi8(zpad_mask, diag0, d0);
                    diag0 = _mm512_mask_blend_epi8(_mm512_movepi8_mask(t0_v) | q_pad, diag0, zero);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe0 = _mm512_max_epu8(diag0, e_v);
                    let h0 = _mm512_max_epu8(mfe0, f0);
                    // ISSUE #44 part B, the free swap: the strict `>` mask is already needed for `col0`, and
                    // `max_epu8(a, b)` IS `mask_blend(cmpgt_epu8(b, a), a, b)`, so reusing it costs no new
                    // instruction and moves the max off p0, which is the port that binds at 512 bits, onto
                    // p05 where `VPBLENDMB` runs at 2/cycle. Byte-identical by definition of max.
                    let gt = _mm512_cmpgt_epu8_mask(h0, imax0);
                    col0 = _mm512_mask_blend_epi8(gt, col0, j_v);
                    imax0 = _mm512_mask_blend_epi8(gt, imax0, h0);
                    let e_mid = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(e_v, e_del_v),
                        _mm512_subs_epu8(h0, oe_del_v),
                    );
                    f0 = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f0, e_ins_v),
                        _mm512_subs_epu8(mfe0, oe_ins_v),
                    );
                    d0 = _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);

                    let mut diag1 = score_diag(score_tbl, bias_v, t1_v, q_v, d1);
                    diag1 = _mm512_mask_blend_epi8(zpad_mask, diag1, d1);
                    diag1 = _mm512_mask_blend_epi8(_mm512_movepi8_mask(t1_v) | q_pad, diag1, zero);
                    let mfe1 = _mm512_max_epu8(diag1, e_mid);
                    let h1 = _mm512_max_epu8(mfe1, f1);
                    // ISSUE #44 part B, the free swap: the strict `>` mask is already needed for `col1`, and
                    // `max_epu8(a, b)` IS `mask_blend(cmpgt_epu8(b, a), a, b)`, so reusing it costs no new
                    // instruction and moves the max off p0, which is the port that binds at 512 bits, onto
                    // p05 where `VPBLENDMB` runs at 2/cycle. Byte-identical by definition of max.
                    let gt = _mm512_cmpgt_epu8_mask(h1, imax1);
                    col1 = _mm512_mask_blend_epi8(gt, col1, j_v);
                    imax1 = _mm512_mask_blend_epi8(gt, imax1, h1);
                    _mm512_storeu_si512(
                        e.as_mut_ptr().add(j * LANES64) as *mut __m512i,
                        leaf_max::<LEAF_CMP_BLEND>(
                            _mm512_subs_epu8(e_mid, e_del_v),
                            _mm512_subs_epu8(h1, oe_del_v),
                        ),
                    );
                    f1 = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f1, e_ins_v),
                        _mm512_subs_epu8(mfe1, oe_ins_v),
                    );
                    d1 = h0;
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h1);
                    j_v = _mm512_add_epi8(j_v, one_v);
                }

                finish_row!(i, imax0, col0);
                finish_row!(i + 1, imax1, col1);
            } else {
                let t_v = _mm512_loadu_si512(seq_t.as_ptr().add(i * LANES64) as *const __m512i);
                let mut f_v = zero;
                let mut h_diag_v = zero;
                let mut imax_v = zero;
                let mut imax_col_v = zero;
                // Carried column index (bumped by `one_v` = set1_epi8(1)) and the row-invariant "target
                // is N" mask; see `fwd_local_sw_neon_u8`.
                let mut j_v = zero;

                // Padding-free column range, same argument as every other kernel here.
                for j in 0..n_fast {
                    let q_v = _mm512_loadu_si512(seq_q.as_ptr().add(j * LANES64) as *const __m512i);
                    let diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe = _mm512_max_epu8(diag_v, e_v);
                    let h_v = _mm512_max_epu8(mfe, f_v);
                    // Issue #44 part B, the free swap; see the pair body.
                    let is_new_row_max = _mm512_cmpgt_epu8_mask(h_v, imax_v);
                    imax_col_v = _mm512_mask_blend_epi8(is_new_row_max, imax_col_v, j_v);
                    imax_v = _mm512_mask_blend_epi8(is_new_row_max, imax_v, h_v);
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h_v);
                    let e_new = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(e_v, e_del_v),
                        _mm512_subs_epu8(h_v, oe_del_v),
                    );
                    _mm512_storeu_si512(e.as_mut_ptr().add(j * LANES64) as *mut __m512i, e_new);
                    f_v = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f_v, e_ins_v),
                        _mm512_subs_epu8(mfe, oe_ins_v),
                    );
                    h_diag_v =
                        _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    j_v = _mm512_add_epi8(j_v, one_v);
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
                    let mut diag_v = score_diag(score_tbl, bias_v, t_v, q_v, h_diag_v);
                    diag_v = _mm512_mask_blend_epi8(zpad_mask, diag_v, h_diag_v);
                    diag_v = _mm512_mask_blend_epi8(pad_mask, diag_v, zero);

                    let e_v = _mm512_loadu_si512(e.as_ptr().add(j * LANES64) as *const __m512i);
                    let mfe = _mm512_max_epu8(diag_v, e_v);
                    let h_v = _mm512_max_epu8(mfe, f_v);
                    // Strict unsigned `>`, so a tie keeps the earlier column.
                    // Issue #44 part B, the free swap; see the pair body.
                    let is_new_row_max = _mm512_cmpgt_epu8_mask(h_v, imax_v);
                    imax_col_v = _mm512_mask_blend_epi8(is_new_row_max, imax_col_v, j_v);
                    imax_v = _mm512_mask_blend_epi8(is_new_row_max, imax_v, h_v);
                    _mm512_storeu_si512(h_cur.as_mut_ptr().add(j * LANES64) as *mut __m512i, h_v);

                    let e_new = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(e_v, e_del_v),
                        _mm512_subs_epu8(h_v, oe_del_v),
                    );
                    _mm512_storeu_si512(e.as_mut_ptr().add(j * LANES64) as *mut __m512i, e_new);
                    f_v = leaf_max::<LEAF_CMP_BLEND>(
                        _mm512_subs_epu8(f_v, e_ins_v),
                        _mm512_subs_epu8(mfe, oe_ins_v),
                    );
                    h_diag_v =
                        _mm512_loadu_si512(h_prev.as_ptr().add(j * LANES64) as *const __m512i);
                    j_v = _mm512_add_epi8(j_v, one_v);
                }
                finish_row!(i, imax_v, imax_col_v);
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            if (0..n_lanes).all(|l| frozen[l] || i + rows >= tlen[l]) {
                break;
            }
            i += rows;
        }

        extract_group(
            n_lanes, group_idx, LANES64, &minsc, max_sc, &gmax, &te, &qe, &limit, &rowmax, &mut out,
        );
    }
    out
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
#[target_feature(enable = "avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_avx512_i16(
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
    let m_v = _mm512_set1_epi16(m as i16);
    let zpad_v = _mm512_set1_epi16(ZPAD as i16);
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
                seq_q[c * LANES32 + l] = b;
            }
            for c in qlen[l]..ksw_padded_qlen(qlen[l], max_sc) {
                seq_q[c * LANES32 + l] = ZPAD;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES32 + l] = b;
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

        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES32);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero;
            // Carried column index and the row-invariant "target is N" mask; see the NEON u8 kernel.
            let mut j_v = zero;
            let t_is_n = _mm512_cmpeq_epi16_mask(t_v, four_v);

            // Padding-free column range, same argument as every other kernel here.
            for j in 0..n_fast {
                let q_v = load_codes(&seq_q, j * LANES32);
                let eq = _mm512_cmpeq_epi16_mask(t_v, q_v);
                let n_mask = t_is_n | _mm512_cmpeq_epi16_mask(q_v, four_v);
                let mut sc = _mm512_mask_blend_epi16(eq, mis_v, mtch_v);
                sc = _mm512_mask_blend_epi16(n_mask, sc, n_v);
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
                    _mm512_cmpge_epi16_mask(t_v, m_v) | _mm512_cmpgt_epi16_mask(q_v, zpad_v);
                // `mask ? a : b` = `mask_blend(mask, b, a)`, increasing priority.
                let mut sc = _mm512_mask_blend_epi16(eq, mis_v, mtch_v);
                sc = _mm512_mask_blend_epi16(n_mask, sc, n_v);
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
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

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
                unsafe { fwd_local_sw_avx2_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc) }
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
        let got =
            unsafe { fwd_local_sw_avx2_i16(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc) };
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
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
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
        let got =
            unsafe { fwd_local_sw_sse41_u8(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc) };
        for i in 0..jobs.len() {
            assert_eq!(got[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
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
        let got =
            unsafe { fwd_local_sw_sse41_i16(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc) };
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

    /// The 64-lane AVX-512 u8 kernel vs the scalar lockstep, on stock DNA scores. Skipped where the CPU
    /// lacks AVX-512BW (including under Rosetta), so it is only a real check on AVX-512 hardware.
    #[test]
    fn avx512_matesw_u8_matches_scalar() {
        if !crate::avx512_testable() {
            eprintln!("skipping avx512_matesw_u8_matches_scalar: no avx512bw on this CPU");
            return;
        }
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (qbufs, tbufs) = rescue_jobs(2000);
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
                endsc: if t.len() % 2 == 0 { i32::MAX } else { 30 },
            })
            .collect();
        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // BOTH spellings of #44 part B, not whichever this host's calibration prefers: the one that
        // is not selected here is the one that would otherwise ship untested, and it is selected on
        // somebody else's machine.
        // SAFETY: avx512bw just confirmed; u8 preconditions hold for these stock-DNA-scored jobs.
        let plain = unsafe {
            fwd_local_sw_avx512_u8_impl::<false>(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        };
        let blended = unsafe {
            fwd_local_sw_avx512_u8_impl::<true>(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc)
        };
        for i in 0..jobs.len() {
            assert_eq!(plain[i], want[i], "job {i} (qlen {})", jobs[i].query.len());
            assert_eq!(
                blended[i],
                want[i],
                "cmp+blend, job {i} (qlen {})",
                jobs[i].query.len()
            );
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
        // SAFETY: avx512bw just confirmed; every ceiling is under I16_SCORE_LIMIT and mat is standard.
        let got =
            unsafe { fwd_local_sw_avx512_i16(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc) };
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
        if std::arch::is_x86_feature_detected!("avx512bw") {
            let avx512 = bench("avx512_u8", &|| unsafe {
                fwd_local_sw_avx512_u8_impl::<false>(
                    &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                )
            });
            eprintln!(
                "  avx512 vs scalar: {:.2}x   avx512 vs avx2: {:.2}x",
                scalar / avx512,
                avx2 / avx512
            );
            // ISSUE #44 part B, the paid half. Timed here rather than trusted: it is a win only
            // where p0 binds, which is an Intel-at-512-bits property, and the run-time calibration
            // decides on exactly this comparison.
            let blended = bench("avx512_cmpblend", &|| unsafe {
                fwd_local_sw_avx512_u8_impl::<true>(
                    &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                )
            });
            eprintln!(
                "  part B cmp+blend vs max: {:.2}x   (calibration picked {})",
                avx512 / blended,
                if avx512_leaf_cmp_blend() {
                    "cmp+blend"
                } else {
                    "max"
                }
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

        let mut state = 0xfeed_face_1234_5678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for k in 1..=32usize {
            let tlen = 200 + k * 90; // ragged, 290 to 3080
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let q: Vec<u8> = (0..QLEN).map(|_| (next() % 5) as u8).collect();
            // Plant the query at the very end of the window, so the best alignment ends on the last
            // real row and its H then walks diagonally through the padded columns.
            let at = tlen - QLEN;
            t[at..].copy_from_slice(&q);
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
                endsc: i32::MAX,
            })
            .collect();

        let want = fwd_local_sw_scalar(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        // SAFETY: neon detected above; 147 bp queries and a score ceiling of 147 both under 250;
        // `scmat` is standard.
        let got = unsafe {
            fwd_local_sw_neon_u8_impl::<true, true>(
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
                    fwd_local_sw_neon_u8_impl::<true, true>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "usqadd",
                    fwd_local_sw_neon_u8_impl::<true, false>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "shareoe",
                    fwd_local_sw_neon_u8_impl::<false, true>(
                        &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc,
                    ),
                ),
                (
                    "baseline",
                    fwd_local_sw_neon_u8_impl::<false, false>(
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
}

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

use bwa_extend::KswAlignResult;

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

/// `BWA3_MATESW_TIME=1`: cells, jobs and wall for the rescue kernel, so its throughput can be
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
    /// Summed wall time in nanoseconds spent inside [`super::batched_ksw_align2`], across all
    /// threads, so it is CPU time rather than elapsed time when `-t > 1`.
    pub static NS: AtomicU64 = AtomicU64::new(0);
    /// Summed query lengths in bases over all counted jobs; `QLEN / JOBS` is the mean read length.
    pub static QLEN: AtomicU64 = AtomicU64::new(0);
    /// Summed target lengths in bases over all counted jobs; `TLEN / JOBS` is the mean rescue window,
    /// the number that revealed the window (not the read) is what makes rescue expensive.
    pub static TLEN: AtomicU64 = AtomicU64::new(0);
    /// Whether `BWA3_MATESW_TIME` is set in the environment. Read once and cached, so setting the
    /// variable after the first call has no effect and the hot path pays only an atomic load.
    ///
    /// # Returns
    /// `true` if the counters above should be accumulated.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA3_MATESW_TIME").is_some())
    }
    /// Print the accumulated counters to stderr, once, at the end of a run. No-op unless
    /// [`enabled`]. Takes no parameters and reads the statics above with `Relaxed` ordering: the
    /// counts are diagnostics, so a slightly stale read from another thread is acceptable.
    pub fn dump() {
        if !enabled() { return; }
        let (cells, jobs, elapsed_ns) = (CELLS.load(Ordering::Relaxed), JOBS.load(Ordering::Relaxed), NS.load(Ordering::Relaxed));
        let seconds = elapsed_ns as f64 / 1e9;
        eprintln!(
            "[matesw] {jobs} jobs, {cells} DP cells in {seconds:.2}s CPU -> {:.2} Gcell/s/thread\n\
             [matesw] ISA ceiling: 16 u8 lanes x ~3.5 GHz = ~56 Gcell/s if 1 cell/lane/cycle",
            cells as f64 / seconds.max(1e-9) / 1e9);
        let (query_bases, target_bases) = (QLEN.load(Ordering::Relaxed), TLEN.load(Ordering::Relaxed));
        eprintln!("[matesw] mean query = {:.0} bp, mean target window = {:.0} bp -> {:.0} cells/job",
                  query_bases as f64 / jobs.max(1) as f64,
                  target_bases as f64 / jobs.max(1) as f64,
                  cells as f64 / jobs.max(1) as f64);
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
    // `Some(start instant)` only when BWA3_MATESW_TIME is set; `None` disables all accounting so the
    // stock path pays one cached bool load and nothing else.
    let timer = cells::enabled().then(std::time::Instant::now);
    if timer.is_some() {
        use std::sync::atomic::Ordering::Relaxed;
        // The DP is query x target per job; that is the work the kernel must actually do.
        let cell_count: u64 = jobs.iter().map(|j| (j.query.len() * j.target.len()) as u64).sum();
        cells::CELLS.fetch_add(cell_count, Relaxed);
        cells::JOBS.fetch_add(jobs.len() as u64, Relaxed);
        // 207k cells per job means a huge target window; record the dimensions to see which side it is.
        cells::QLEN.fetch_add(jobs.iter().map(|j| j.query.len() as u64).sum::<u64>(), Relaxed);
        cells::TLEN.fetch_add(jobs.iter().map(|j| j.target.len() as u64).sum::<u64>(), Relaxed);
    }
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
    let fwd_results = fwd_local_sw_batch(&fwd_jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);

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
    let mut rev_bufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
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
            let qrev: Vec<u8> = j.query[..=qe as usize].iter().rev().copied().collect();
            let trev: Vec<u8> = j.target[..=te as usize].iter().rev().copied().collect();
            rev_bufs.push((qrev, trev));
            rev_of_job.push(i);
        }
    }
    // The reverse batch, one per qualifying job: `minsc = i32::MAX` suppresses the `score2` list
    // (a 2nd-best is only ever taken from the forward pass), and `endsc = out[i].score` stops each
    // lane the instant it matches the forward score, which is the whole reason this pass is cheap.
    let rev_jobs: Vec<FwdJob> = rev_bufs
        .iter()
        .zip(rev_of_job.iter())
        .map(|((q, t), &i)| FwdJob {
            query: q,
            target: t,
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
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && mat_is_standard(m, mat) {
            // Max reachable score per job = min(len) * match. Only the SCORE cells (H/E/F) live in the
            // SIMD vector; positions/te/qe are scalar i32, so window length is unconstrained. If every
            // job's score ceiling fits u8, run 16 lanes; else the i16 kernel at 8 lanes. Mate-rescue
            // jobs (short reads, match ~1) fit u8.
            // A local alignment can match at most `min(qlen, tlen)` bases and only loses score to
            // mismatches and gaps, so `min(len) * max_sc` is a hard ceiling on every H/E/F cell.
            let score_ceiling = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            // u8 also holds the argmax query column, so the query must be < 256 too.
            if jobs
                .iter()
                .all(|j| score_ceiling(j) < U8_SCORE_LIMIT && j.query.len() < U8_SCORE_LIMIT as usize)
            {
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
        let qmax = group.iter().map(|j| ksw_padded_qlen(j.query.len(), max_sc)).max().unwrap_or(0);
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
        let (mut qlen, mut tlen, mut minsc, mut endsc) =
            ([0usize; LANES], [0usize; LANES], [i32::MAX; LANES], [i32::MAX; LANES]);
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
fn extract_group(
    n_lanes: usize,
    group_idx: usize,
    lanes: usize,
    minsc: &[i32],
    max_sc: i32,
    gmax: &[i32],
    te: &[i32],
    qe: &[i32],
    limit: &[i32],
    rowmax: &[i32],
    out: &mut [(i32, i32, i32, i32, i32)],
) {
    for l in 0..n_lanes {
        let best_score = gmax[l];
        let best_te = te[l];
        // No alignment found at all (te still -1) means there is no query end to report either.
        let best_qe = if best_te >= 0 { qe[l] } else { -1 };
        // score2: rebuild ksw_local_fwd's `b` list (row-maxes >= minsc, consecutive rows merged
        // keeping the higher AND advancing the column only on an update), then take the best entry
        // whose column lies outside [te - w, te + w].
        let mut score2 = -1i32;
        let mut te2 = -1i32;
        // ksw's `b` array: surviving `(row max score, target row)` candidates in increasing row
        // order, runs of consecutive rows already merged. Never longer than the number of rows.
        let mut b: Vec<(i32, i32)> = Vec::new();
        if limit[l] >= 0 {
            for i in 0..=limit[l] {
                // Best H anywhere in target row `i` for this lane.
                let row_max = rowmax[i as usize * lanes + l];
                if row_max >= minsc[l] {
                    match b.last() {
                        // Row `i` directly follows the last entry's row: same alignment drifting, so
                        // merge rather than append. bwa keeps the higher score AND moves the stored
                        // row to `i` in the same step (`ksw.cpp:201`), which is why the row is only
                        // advanced inside the `<` branch. Advancing it unconditionally would look
                        // harmless and would change which entries the window below excludes.
                        // `col` is the last entry's stored *row*, despite the name kept from the C.
                        Some(&(_, col)) if col + 1 == i => {
                            if b.last().unwrap().0 < row_max {
                                *b.last_mut().unwrap() = (row_max, i);
                            }
                        }
                        // A gap in rows: a genuinely separate candidate (`ksw.cpp:195-200`).
                        _ => b.push((row_max, i)),
                    }
                }
            }
        }
        if best_score > 0 && !b.is_empty() {
            // ceil(score / max_sc) = the fewest columns an alignment scoring `score` can span, since
            // no cell contributes more than `max_sc`. Any candidate whose row lies within that many
            // rows of `te` could be the same alignment ending early or late, so it is excluded from
            // the 2nd-best search rather than counted as a rival (`ksw.cpp:221-222`).
            let exclusion_half_width = (best_score + max_sc - 1) / max_sc;
            // Inclusive band of target rows around `te` treated as "the same alignment"; candidates
            // inside it are ignored. May go negative or past the last row, which is harmless.
            let (low, high) = (
                best_te - exclusion_half_width,
                best_te + exclusion_half_width,
            );
            // `cand_score` is a surviving row max in score units and `cand_te` its target row.
            for &(cand_score, cand_te) in &b {
                if (cand_te < low || cand_te > high) && cand_score > score2 {
                    score2 = cand_score;
                    te2 = cand_te;
                }
            }
        }
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
        let qmax = group.iter().map(|j| ksw_padded_qlen(j.query.len(), max_sc)).max().unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // SoA sequences (u8, padded) + per-lane bounds/params.
        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        let (mut qlen, mut tlen, mut minsc, mut endsc) =
            ([0usize; LANES], [0usize; LANES], [i32::MAX; LANES], [i32::MAX; LANES]);
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
            for j in 0..qmax {
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
                let n_mask = vorrq_u16(vceqq_s16(t_v, four_v), vceqq_s16(q_v, four_v));
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
                h_v = vmaxq_s16(h_v, f_v);
                // Track the min column reaching a new row max (strict >, so ties keep the earlier j).
                // `is_new_row_max` is all-ones in the lanes whose job just beat its own row best.
                let is_new_row_max = vcgtq_s16(h_v, imax_v);
                imax_col_v = vbslq_s16(is_new_row_max, vdupq_n_s16(j as i16), imax_col_v);
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
                let f_new = vmaxq_s16(vsubq_s16(f_v, e_ins_v), vsubq_s16(h_v, oe_ins_v));
                f_v = vmaxq_s16(f_new, zero);
                // Preload the next column's diagonal from the previous row, mirroring `ksw.cpp:176`.
                h_diag_v = vld1q_s16(h_prev.as_ptr().add(j * LANES));
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
    let mtch_v = vdupq_n_u8(mtch);
    let mispen_v = vdupq_n_u8(mispen);
    let one_v = vdupq_n_u8(1); // N penalty
    let four_v = vdupq_n_u8(4);
    let m_v = vdupq_n_u8(m as u8);
    let zpad_v = vdupq_n_u8(ZPAD);
    let e_del_v = vdupq_n_u8(e_del as u8);
    let oe_del_v = vdupq_n_u8(oe_del as u8);
    let e_ins_v = vdupq_n_u8(e_ins as u8);
    let oe_ins_v = vdupq_n_u8(oe_ins as u8);

    // Group setup is identical to `fwd_local_sw_scalar` at 16 lanes; see there for each variable.
    for (group_idx, group) in jobs.chunks(LANES16).enumerate() {
        let n_lanes = group.len();
        let qmax = group.iter().map(|j| ksw_padded_qlen(j.query.len(), max_sc)).max().unwrap_or(0);
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

        // Same DP state as the scalar path (H(i-1, .), H(i, .), the per-column E carry), narrowed to
        // u8 so a column of all 16 lanes is one `vld1q_u8`. Legal only because local SW keeps every
        // value in `[0, ceiling]` and the caller proved the ceiling is under 250.
        let mut h_prev = vec![0u8; qmax * LANES16];
        let mut h_cur = vec![0u8; qmax * LANES16];
        let mut e = vec![0u8; qmax * LANES16];
        let mut rowmax = vec![0i32; tmax * LANES16];
        let mut gmax = [0i32; LANES16];
        let mut te = [-1i32; LANES16];
        let mut qe = [0i32; LANES16];
        let mut limit = [-1i32; LANES16];
        let mut frozen = [false; LANES16];
        for l in 0..n_lanes {
            limit[l] = tlen[l] as i32 - 1;
        }

        // =====================================================================================
        // Main DP, one target row per iteration. Identical in structure to `fwd_local_sw_neon`,
        // at 16 lanes instead of 8, with saturating u8 arithmetic supplying the `max(0, .)` clamps.
        // =====================================================================================
        for i in 0..tmax {
            // Lane `l` = target base at row `i` of job `l`.
            let t_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
            // Row accumulators, one lane per job, same invariant as the scalar version at the top of
            // column `j`: F(i, j), H(i-1, j-1), max H(i, 0..j), and its smallest attaining column.
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            let mut imax_col_v = zero; // min query column achieving this row's max
            for j in 0..qmax {
                // Lane `l` = query base at column `j` of job `l`.
                let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                // diag_v = max(0, h_diag + score): saturating add/sub floor at 0, no explicit max-0.
                // Masks, all-ones per lane where they apply: `eq` the bases match, `n_mask` either is
                // N, `zpad_mask` the query column is ksw profile padding, `pad_mask` the cell is
                // past a real position (dead row, or query past the padded profile).
                let eq = vceqq_u8(t_v, q_v);
                let n_mask = vorrq_u8(vceqq_u8(t_v, four_v), vceqq_u8(q_v, four_v));
                let zpad_mask = vceqq_u8(q_v, zpad_v);
                let pad_mask = vorrq_u8(vcgeq_u8(t_v, m_v), vcgtq_u8(q_v, zpad_v));
                // Rather than build a score and add it, each of the four cases is applied directly
                // to `h_diag` as a saturating add or subtract. That is what removes the need for a
                // signed intermediate: a mismatch at h_diag = 2 with penalty 4 saturates to 0, which
                // is exactly what `max(0, h_diag - 4)` should give, whereas a wrapping sub gives 254.
                // The three candidate values of `max(0, H(i-1,j-1) + S)`, computed unconditionally
                // for every lane and then selected between: the match case, the mismatch case, and
                // the N case. Lane `l` of each is job `l`'s value.
                let add_match = vqaddq_u8(h_diag_v, mtch_v);
                let sub_mis = vqsubq_u8(h_diag_v, mispen_v);
                // N scores -1 in bwa's matrix, not -b, hence a separate constant from `mispen_v`.
                let sub_n = vqsubq_u8(h_diag_v, one_v);
                // Lane `l` = max(0, H(i-1, j-1) + S) for job `l`, the diagonal arrival term, after
                // the four selects below resolve which case this cell is.
                let mut diag_v = vbslq_u8(eq, add_match, sub_mis);
                diag_v = vbslq_u8(n_mask, sub_n, diag_v);
                diag_v = vbslq_u8(zpad_mask, h_diag_v, diag_v); // score 0: diagonal passes through
                // Dead padding: force 0 outright. In the i16 kernel this is a `DEAD_CELL_SCORE`
                // that the `max(0, .)` then clamps; here 0 is written directly, same result.
                diag_v = vbslq_u8(pad_mask, zero, diag_v);

                // Lane `l` = E(i, j) for job `l`, the deletion carry left by the previous row.
                let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                // No explicit `max(0, .)` on H: `diag_v`, `e_v` and `f_v` are already >= 0 by
                // saturation, so the two maxes are the whole H recurrence.
                // Lane `l` becomes H(i, j) for job `l` after the second max.
                let mut h_v = vmaxq_u8(diag_v, e_v);
                h_v = vmaxq_u8(h_v, f_v);
                // Track the min column reaching a new row max (strict >, so ties keep the earlier j).
                // `is_new_row_max` is all-ones in the lanes whose job just beat its own row best.
                // `j as u8` is why the caller caps the query at 250 bases: the column index shares
                // the lane width with the scores, and a longer query would wrap the argmax. The
                // real constraint is the *padded* length: `ksw_padded_qlen` rounds up to a multiple
                // of 16 (or 8), so a query under 250 pads to at most 256 columns and the largest
                // `j` is 255, the last value u8 can hold. Raising the 250 cap past 256 padded
                // columns would wrap `imax_col_v` silently, with no overflow check anywhere.
                let is_new_row_max = vcgtq_u8(h_v, imax_v);
                imax_col_v = vbslq_u8(is_new_row_max, vdupq_n_u8(j as u8), imax_col_v);
                imax_v = vmaxq_u8(imax_v, h_v);
                vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);

                // e = max(max(0,e-e_del), max(0,h-oe_del)) = max(0, e-e_del, h-oe_del). The two
                // saturating subs supply both inner clamps, so the i16 kernel's trailing
                // `vmaxq(e_new, zero)` has no counterpart here: it would be a no-op.
                // Lane `l` = E(i+1, j), already floored at 0 by the saturating subtracts.
                let e_new = vmaxq_u8(vqsubq_u8(e_v, e_del_v), vqsubq_u8(h_v, oe_del_v));
                vst1q_u8(e.as_mut_ptr().add(j * LANES16), e_new);
                // Same for F, the row's serial carry.
                f_v = vmaxq_u8(vqsubq_u8(f_v, e_ins_v), vqsubq_u8(h_v, oe_ins_v));
                h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
            }

            // Spill the two row accumulators so lanes can be read individually: `imax_arr[l]` is job
            // `l`'s max H in row `i`, `col_arr[l]` the padded query column where it occurred.
            let mut imax_arr = [0u8; LANES16];
            let mut col_arr = [0u8; LANES16];
            vst1q_u8(imax_arr.as_mut_ptr(), imax_v);
            vst1q_u8(col_arr.as_mut_ptr(), imax_col_v);
            for l in 0..n_lanes {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                // Job `l`'s best H anywhere in target row `i`, widened out of the lane.
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

            // Early exit once every live lane is either frozen or out of target: purely a speed win,
            // as in `fwd_local_sw_neon`. See the longer note there.
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
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
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
            // Inject N bases (code 4) sometimes, in query and/or target.
            if next() % 4 == 0 {
                q[(next() as usize) % qlen] = 4;
            }
            if next() % 4 == 0 {
                t[(next() as usize) % tlen] = 4;
            }
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
                let lanes = if j.query.len() as i32 * max_sc < 250 { 16 } else { 8 };
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc, lanes,
                );
                assert_eq!(batched[i], want, "job {i} (qlen {}, match {a})", j.query.len());
            }
        }
    }
}

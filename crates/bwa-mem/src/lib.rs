//! Alignment core: chains -> scored alignment regions (`mem_chain2aln`) and the single-end driver,
//! mirroring bwa-mem2's `mem_chain2aln_across_reads_V2` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Phase 6 (first milestone): produce alignment regions and pick the best mapping position. Full
//! byte-identical SAM (dedup, primary marking, MAPQ, CIGAR, tags) is layered on top of this.
//!
//! # What this file is responsible for
//!
//! Given a read and the chains of seeds that were found for it, this module runs the dynamic
//! programming (DP) that grows each seed outward into a full alignment, and packages the answer as
//! a [`MemAlnReg`] ("alignment region"). It is the middle of the pipeline:
//!
//! ```text
//!   FASTQ read
//!     -> bwa-seed:   exact-match seeds looked up in the FM index
//!     -> bwa-chain:  collinear seeds grouped into chains, weak chains filtered
//!     -> THIS FILE:  each seed extended left and right by banded DP  => Vec<MemAlnReg>
//!     -> primary.rs: duplicate regions merged, primary/secondary marked, MAPQ computed
//!     -> pe.rs:      paired-end pairing, mate rescue, SAM record emission
//!     -> cigar.rs:   region -> CIGAR string + edit distance
//!     -> alt.rs:     the XA:Z: list of alternative placements
//! ```
//!
//! # Suggested reading order
//!
//! 1. [`MemAlnReg`], the struct everything downstream consumes.
//! 2. [`cal_max_gap`], a small self-contained helper: how wide a reference window an extension needs.
//! 3. [`extend_side`], one call into the DP kernel plus bwa's band-doubling retry rule.
//! 4. [`mem_chain2aln_meta`], the main event: window setup, seed ordering, left extension, right
//!    extension, seed coverage. [`mem_chain2aln`] is just a thin wrapper over it.
//! 5. [`align_read`] / [`align_read_se`], the per-read drivers that string the stages together.
//! 6. [`region_to_pos`], turning an internal coordinate back into a SAM contig plus position.
//!
//! Note that `across.rs` contains a SECOND implementation of step 4, batched across many reads so
//! the DP can be run with SIMD. The two must produce identical regions, so a change here almost
//! always needs the mirror change there.
//!
//! # Glossary of names kept identical to the C
//!
//! Diffing this Rust against `bwamem.cpp` line by line is the central workflow of this project, and
//! every parity bug so far was found that way. So the following names are deliberately NOT
//! renamed to something more readable. Their plain-language meanings:
//!
//! | Name       | Plain language                                                                  |
//! |------------|---------------------------------------------------------------------------------|
//! | `rb`, `re` | Reference begin / end of a region (end exclusive), in packed-genome coords.     |
//! | `qb`, `qe` | Query begin / end, i.e. which slice of the READ the region covers.              |
//! | `h0`       | The DP's starting score: points already banked before this extension starts.    |
//! | `qle`      | "Query length to end": how many read bases the best local alignment consumed.   |
//! | `tle`      | "Target length to end": how many reference bases that same alignment consumed.  |
//! | `gscore`   | Best score among alignments that reach the END of the read (`g` for global).    |
//! | `gtle`     | Reference bases consumed by the `gscore` alignment.                             |
//! | `w`        | DP band width: how far off the diagonal the alignment is allowed to wander.     |
//! | `prev`     | The previous band try's score, used to decide whether widening the band helped. |
//! | `qs`, `rs` | Query slice and reference slice handed to the DP, already oriented outward.     |
//! | `rmax0/1`  | The reference window `[rmax0, rmax1)` that all of a chain's extensions live in. |
//! | `l_pac`    | Length of the packed reference (see `rb` above for why it appears in bounds).   |
//! | `l_query`  | Length of the read in bases.                                                    |
//! | `truesc`   | "True score": the score of the alignment actually taken (see the field docs).   |
//! | `sub`      | Score of the best competing alignment, which is what drives MAPQ.               |
//! | `csub`     | Chain-level equivalent of `sub`.                                                |
//! | `seedcov`  | How many of the region's bases are backed by exact seed matches.                |
//! | `frac_rep` | Fraction of the region's seeds landing in repetitive reference.                 |
//! | `H0_`      | bwa's -99 sentinel for "this bound was never set" (here [`H0_SENTINEL`]).       |

use crate::across::RegMeta;
use bwa_chain::{build_chains, mem_chain_flt, MemChain};
use bwa_core::MemOpt;
use bwa_extend::ksw_extend2;
use bwa_index::{BntSeq, FmIndex};

pub mod across;
pub mod alt;
pub mod cigar;
pub mod pe;
pub mod primary;
pub use across::align_reads_batched;
pub use cigar::{cigar_string, reg2aln, MemAln};
pub use pe::{batch_mate_rescue, mem_pestat, mem_sam_pe, PairRescueData, PeStat};
pub use primary::{mem_approx_mapq_se, mem_mark_primary_se, mem_sort_dedup_patch};

/// Sentinel for uninitialized region bounds (bwa's `H0_`, `macro.h:44`: `#define H0_ -99`).
///
/// bwa-mem2 stamps all four bounds of a fresh `mem_alnreg_t` with it (`bwamem.cpp:2223`,
/// `a->rb = a->qb = a->re = a->qe = H0_`) and later tests `a->rb != H0_ && ...` to decide whether
/// a region was ever actually extended (`bwamem.cpp:2423`, `:2507`). It has to be a value no real
/// coordinate can take, hence a negative number rather than 0.
pub(crate) const H0_SENTINEL: i64 = -99;
/// bwa's `MAX_BAND_TRY` (`bwamem.cpp:51`): how many times a one-sided extension may re-run with a
/// doubled DP band before its result is accepted unconditionally. 2 means bands `opt.w` and
/// `opt.w << 1` are tried, no more.
pub(crate) const MAX_BAND_TRY: i32 = 2;

/// A scored alignment region (bwa-mem2's `mem_alnreg_t`, phase-6 subset).
///
/// One region is produced per extended seed, so a read normally carries many of them; the dedup,
/// primary-marking and pairing passes then collapse and rank the set. Field names are kept
/// identical to the C struct so the two can be diffed side by side when chasing a byte divergence.
#[derive(Debug, Clone)]
pub struct MemAlnReg {
    /// Reference begin, 0-based inclusive, in bwa's *concatenated* forward+reverse pac coordinate
    /// space: `[0, l_pac)` is the forward strand and `[l_pac, 2*l_pac)` the reverse. A region never
    /// straddles `l_pac` (the window clamp below enforces that).
    pub rb: i64,
    /// Reference end, exclusive, same coordinate space as `rb`.
    pub re: i64,
    /// Query begin, 0-based inclusive, in read coordinates as the read was seeded (already
    /// reverse-complemented if the seed hit the reverse strand).
    pub qb: i32,
    /// Query end, exclusive.
    pub qe: i32,
    /// Contig index into `BntSeq::contigs`, or -1 when the region does not resolve to a contig.
    pub rid: i32,
    /// Best banded local-alignment score for the region, as returned by the extension DP. This is
    /// the value ranked on by dedup, primary marking and MAPQ, and the one emitted as `AS`.
    pub score: i32,
    /// The score of the alignment that was actually *taken*. It equals `score` when extension chose
    /// to soft-clip, but becomes the global-extension score `gscore` when running to the end of the
    /// query beat clipping by more than `pen_clip` (`bwamem.cpp:2499-2505`). Kept separate because
    /// pairing scores pairs on `truesc` while ranking still uses `score`.
    pub truesc: i32,
    /// Score of the best *other* region overlapping this one, i.e. the suboptimal alignment score
    /// that drives MAPQ and the `XS` tag. 0 when there is none.
    pub sub: i32,
    /// Chain-level suboptimal score carried over from chaining, used by `mem_approx_mapq_se` as a
    /// second opinion on `sub`.
    pub csub: i32,
    /// Number of near-optimal alternative regions, feeding the MAPQ penalty for repetitive hits.
    pub sub_n: i32,
    /// Total seed length contained inside `[qb,qe) x [rb,re)`, a proxy for how much of the region
    /// is supported by exact seed matches. Used by MAPQ and by the dedup pass.
    pub seedcov: i32,
    /// Length of the seed this region was grown from, before any extension.
    pub seedlen0: i32,
    /// -1 for a primary region, otherwise the index of the region it is secondary to. The value
    /// -2 is used as a sentinel by the PE path to mark a record that must still be emitted.
    pub secondary: i32,
    /// Rank-preserving secondary index used only by `mem_gen_alt` (`get_pri_idx`). Equals
    /// `secondary` after marking on a no-ALT reference, but the PE primary/secondary swap mutates
    /// it independently of `secondary` (which the `-2` sentinel repurposes for the emitted record).
    pub secondary_all: i32,
    /// The DP band width that was finally accepted for this region. Starts at `opt.w` and is
    /// raised to the widest band any of the two extensions ended up using (`max_(a->w, w)` at
    /// `bwamem.cpp:2506`). Later re-used as the band for CIGAR generation, so it must survive.
    pub w: i32,
    /// Fraction of the region's seeds that fell in repetitive parts of the reference, inherited
    /// from the chain. `f32` and not `f64` on purpose: the C field is `float` and this value is
    /// multiplied into the MAPQ formula, where the narrower mantissa is observable.
    pub frac_rep: f32,
    /// Whether the region sits on an ALT contig, which suppresses it from primary consideration.
    pub is_alt: bool,
    /// Per-region hash used purely as a deterministic tie-break when two regions compare equal
    /// during primary marking, so that ordering does not depend on sort stability. Derived from
    /// the read id, hence the `read_id` parameter threaded through `align_read_se`.
    pub hash: u64,
    /// Number of chain components merged into this region by the dedup pass. Starts at 1.
    pub n_comp: i32,
}

/// Longest gap worth allowing for a `qlen`-base extension, i.e. how far the reference window must
/// reach beyond the seed. Port of `cal_max_gap` (`bwamem.cpp:66-75`).
///
/// The formula asks: with affine gap costs, how long can a gap get before its penalty eats the
/// entire match score `qlen * a` this extension could possibly earn? Solving
/// `o + e * l = qlen * a` for `l` gives `l = (qlen * a - o) / e`, computed separately for
/// deletions and insertions, and the larger of the two wins. The `+ 1.` then truncation is the C's
/// way of rounding up. The result is floored at 1 and capped at `opt.w << 1` (twice the band
/// width), because no DP band wider than that is ever run.
///
/// # Parameters
///
/// - `opt`: the read-group-wide scoring options, supplied by the CLI once per run and never mutated
///   here. Only five fields are read: `a` (match score, positive, default 1), `o_del`/`e_del` and
///   `o_ins`/`e_ins` (gap open and gap extend penalties for deletions and insertions, all strictly
///   positive; `e_del`/`e_ins` must be nonzero or the divisions below are undefined), and `w`
///   (the DP band width in bases, positive, default 100).
/// - `qlen`: the number of query bases this extension could still consume, i.e. the length of the
///   read prefix left of the seed or of the suffix right of it. Units: read bases. Range `>= 0`;
///   supplied by `mem_chain2aln_meta` from the seed's `qbeg` / `l_query - qbeg - len`. A `qlen` of
///   0 is legal and yields the floor value 1.
///
/// # Returns
///
/// A count of reference bases, always in `[1, opt.w << 1]`: how far past the seed the reference
/// window must reach so that no gap the DP could profitably open falls outside it.
///
/// C-QUIRK: the C evaluates `qlen * opt->a - opt->o_del` in `int` and only then casts to `double`;
/// we promote to `f64` before multiplying. The two agree for every `qlen` an int can hold without
/// the product overflowing (`qlen < 2^31 / a`), which no read approaches, so this is a widening
/// that cannot change output. The division and the truncating cast are kept in `f64`/`as i32`
/// exactly as the C does, because doing it in integers would round differently.
#[must_use]
pub(crate) fn cal_max_gap(opt: &MemOpt, qlen: i32) -> i32 {
    // `opt.a` is the per-base match score, the reward this extension could earn per query base.
    let match_score = f64::from(opt.a);
    // Longest DELETION (bases missing from the read) whose affine penalty is still repaid by the
    // `qlen * a` this extension could earn. May be negative before the `.max(1)` floor below when
    // `qlen` is small enough that even opening a gap costs more than the whole extension is worth.
    let max_deletion_len =
        ((f64::from(qlen) * match_score - f64::from(opt.o_del)) / f64::from(opt.e_del) + 1.0) as i32;
    // Same quantity for an INSERTION (extra bases in the read), which may use different open/extend
    // penalties, hence the separate computation rather than reusing the deletion answer.
    let max_insertion_len =
        ((f64::from(qlen) * match_score - f64::from(opt.o_ins)) / f64::from(opt.e_ins) + 1.0) as i32;
    // Whichever gap kind can stretch furthest decides the window; floored at 1 so a window is never
    // empty even when neither gap kind pays for itself.
    let max_gap = max_deletion_len.max(max_insertion_len).max(1);
    max_gap.min(opt.w << 1)
}

/// What one accepted call to `ksw_extend2` reports back, plus the band that produced it.
struct SideResult {
    /// Best *local* score reachable in this direction (soft-clipping the rest of the query).
    score: i32,
    /// Query length consumed at the `score` optimum ("query length to end").
    qle: i32,
    /// Reference length consumed at the `score` optimum ("target length to end").
    tle: i32,
    /// Best score that reaches the *end of the query* (global in the query, local in the
    /// reference). Compared against `score - pen_clip` to decide clip versus extend-to-end.
    gscore: i32,
    /// Reference length consumed by the `gscore` alignment.
    gtle: i32,
    /// The band width actually accepted, propagated into `MemAlnReg::w`.
    w: i32,
}

/// One-sided extension with bwa's `MAX_BAND_TRY` band-doubling acceptance
/// (`bwamem.cpp:2472-2497`, and the identical loop repeated for each of the other kernels).
///
/// WHY THE LOOP: a banded DP can only find gaps that fit inside the band. bwa runs the extension
/// at band `opt.w`, then decides whether the answer is trustworthy. It is trusted when either the
/// score did not improve over the previous, narrower band (`ext.score == prev`, so widening bought
/// nothing), or the optimum sat well inside the band (`max_off < (w>>1) + (w>>2)`, i.e. the best
/// cell's diagonal offset stayed under three quarters of the half-band, so the band never
/// constrained it). Otherwise the band is doubled and the DP re-run. `i + 1 == MAX_BAND_TRY`
/// forces acceptance after the second try no matter what, so this terminates in at most 2 DP runs.
///
/// OPERATIONS: `opt.w << i` is the band doubling; `(w >> 1) + (w >> 2)` is `0.75 * w` written as
/// shifts, exactly as the C writes it (integer truncation included, which matters for odd `w`).
/// `prev0` is the value the C reads out of the region before the first band try: `int prev =
/// a->score;` (`bwamem.cpp:2492`) runs on *every* iteration, the first included. For a left
/// extension the region still holds the -1 it was initialised with (`bwamem.cpp:2218`), so
/// `score == prev` can never fire at round 0. For a right extension it holds the left extension's
/// score, which is exactly this side's `h0`, so a right extension that gains nothing converges
/// immediately. Passing -1 on both sides is *nearly* equivalent (a right extension scoring exactly
/// `h0` leaves `max_off` at 0, because `ksw_extend2` only touches `max_off` inside `if (m > max)`
/// with `max` seeded to `h0`, so acceptance test 2 fires instead), but it breaks for `w <= 1`,
/// where test 2 reads `0 < 0` and is false. Mirroring the C removes the special case.
///
/// # Parameters
///
/// - `qs`: the slice of the READ still to be aligned on this side, as 2-bit codes (0..=3, 4 for
///   `N`), *already oriented* outward from the seed. The caller reverses it for a left extension
///   (see `mem_chain2aln_meta`), so index 0 is always the base adjacent to the seed. May be empty
///   only if the caller allows it; both call sites guard on a nonzero length first.
/// - `rs`: the corresponding reference slice, same 2-bit encoding, same outward orientation, drawn
///   from the chain's window `rseq`. Its length bounds how far the reference may be consumed.
/// - `opt`: scoring options. Read here: `mat` (the 5x5 substitution matrix), `o_del`/`e_del`,
///   `o_ins`/`e_ins`, `w` (the starting band width, doubled per retry) and `zdrop`.
/// - `pen_clip`: the soft-clip penalty for THIS side, `opt.pen_clip5` for a left extension and
///   `opt.pen_clip3` for a right one, in score units (non-negative). It does double duty: it is
///   passed to `ksw_extend2` in the `end_bonus` slot, not as a penalty (inside the DP a soft clip
///   is modelled as *forgoing a bonus* for reaching the query end rather than as a subtracted
///   cost), and the same number is then used as a genuine penalty in the caller's
///   `gscore <= score - pen_clip` comparison. Same double duty as the C, which also hands
///   `opt->pen_clip5` straight to `ksw_extend2`'s `end_bonus`.
/// - `h0`: the DP's starting score, i.e. the score already banked by the seed (left extension) or
///   by the seed plus the left extension (right extension). Passing it in is what makes the two
///   extensions compose into a single score rather than two independent ones, and it means the
///   returned `score` is CUMULATIVE, not this side's increment.
/// - `prev0`: the score attributed to the "previous, narrower band" on the very first try, so that
///   the `ext.score == prev` acceptance test behaves as the C's does at round 0. -1 for a left
///   extension (the region's freshly initialised `score`), the left extension's score for a right
///   one. See the OPERATIONS paragraph above for why passing -1 on both sides is not equivalent.
///
/// # Returns
///
/// A [`SideResult`] holding the accepted DP answer plus the band width `w` that produced it, which
/// the caller folds into `MemAlnReg::w` with a `max`.
#[allow(clippy::too_many_arguments)]
fn extend_side(
    qs: &[u8],
    rs: &[u8],
    opt: &MemOpt,
    pen_clip: i32,
    h0: i32,
    prev0: i32,
) -> SideResult {
    // `prev` is the score the previous (narrower) band produced; `w` is the band width; `qle`/`tle`
    // and `gscore`/`gtle` are the clipped and run-to-end answers. See the module-header glossary.
    //
    // LOOP INVARIANT, true at the top of every iteration: `prev` holds the score of the last band
    // already tried (or `prev0` before any has been), and `band_try` counts bands tried so far, so
    // it is in `[0, MAX_BAND_TRY)`. Widening is only worth another DP run while `prev` is still
    // being beaten, which is what the acceptance test checks.
    let mut prev = prev0;
    let mut band_try = 0;
    loop {
        // Band width for this attempt: `opt.w` doubled once per previous failed acceptance.
        let w = opt.w << band_try;
        // The raw DP answer at band `w`; `ext.max_off` (how far off-diagonal the optimum sat) is
        // used only by the acceptance test and is not propagated.
        let ext = ksw_extend2(
            qs, rs, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, w, pen_clip,
            opt.zdrop, h0,
        );
        // Accept when widening bought nothing, or the optimum sat well inside the band, or we have
        // used up our `MAX_BAND_TRY` attempts.
        if ext.score == prev || ext.max_off < (w >> 1) + (w >> 2) || band_try + 1 == MAX_BAND_TRY {
            return SideResult {
                score: ext.score,
                qle: ext.qle,
                tle: ext.tle,
                gscore: ext.gscore,
                gtle: ext.gtle,
                w,
            };
        }
        prev = ext.score;
        band_try += 1;
    }
}

/// Extend every seed of `chain` into an alignment region (one region per seed). Port of the
/// per-chain body of `mem_chain2aln_across_reads_V2` (`bwamem.cpp:2140` onward).
///
/// This is the scalar, one-read-at-a-time path. `across.rs` holds the batched path that bwa-mem2
/// actually uses in production (it collects extension jobs across many reads and feeds them to the
/// SIMD kernel). The two MUST produce identical regions, which is why several oddities below, the
/// contained-seed skip in particular, are duplicated verbatim in both.
///
/// # Parameters
///
/// - `fm`: the loaded FM index, used here only as a random-access reader of the 2-bit packed
///   genome via `fm.base(pos)` when materialising the chain's reference window.
/// - `bns`: the contig table, used to clamp that window to a single contig.
/// - `opt`: scoring and band-width options, shared read-only for the whole run.
/// - `codes`: the read as 2-bit codes (0..=3, 4 for `N`), in the orientation it was seeded in. Its
///   length is `l_query`.
/// - `chain`: one chain of collinear seeds from `bwa-chain`; must be nonempty to produce anything
///   (an empty chain returns immediately).
/// - `out`: the read's growing region vector. Regions are APPENDED, never cleared, so a caller can
///   accumulate every chain of a read into one vector; append order is load-bearing downstream.
pub fn mem_chain2aln(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    chain: &MemChain,
    out: &mut Vec<MemAlnReg>,
) {
    mem_chain2aln_meta(fm, bns, opt, codes, 0, chain, out, &mut Vec::new(), &mut Vec::new());
}

/// `mem_chain2aln`, additionally recording each emitted region's [`RegMeta`] so the caller can run
/// bwa-mem2's discard pass over the read's full region set once every chain has been extended.
///
/// # Parameters
///
/// - `fm`: the FM index, used here only for `fm.base(pos)` reads of the 2-bit packed genome.
/// - `bns`: the contig table; supplies `l_pac` (packed reference length, so `2*l_pac` is the
///   forward+reverse coordinate space) and `fetch_bounds` for trimming the window to one contig.
/// - `opt`: scoring options (`a`, gap penalties, `w`, `pen_clip5`/`pen_clip3`, `zdrop`, `mat`).
/// - `codes`: the read as 2-bit codes (0..=3, with 4 for `N`); `codes.len()` is `l_query`.
/// - `ci`: this chain's index within the read (0-based, in the post-`mem_chain_flt` chain order).
///   Stored into each `RegMeta` so the discard pass can reconstruct bwa-mem2's exact scan order.
/// - `chain`: the chain being extended. Read: `seeds` (each with `qbeg`, `rbeg`, `len`, `score`),
///   `rid`, `frac_rep` and `is_alt`, all copied into every region it produces. Returns immediately
///   if `seeds` is empty.
/// - `out`: the read's region vector, appended to (one entry per seed, including skipped seeds).
/// - `meta`: per-region provenance (chain index, position in the extension order, seed index),
///   appended in lockstep with `out` and consumed only by the discard pass.
/// - `preskip`: per-region flag, `true` when the region is a placeholder for a seed whose DP was
///   skipped as redundant, appended in lockstep with `out`.
///
/// All three of `out`, `meta` and `preskip` are grown, never read, by this function.
///
/// INVARIANT: `out.len() == meta.len() == preskip.len()` on entry and on exit. The discard pass
/// indexes all three by the same position, so a `continue` that pushes to one but not the others
/// would silently corrupt it. That is why the skip branch below still pushes a placeholder region.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mem_chain2aln_meta(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    ci: usize,
    chain: &MemChain,
    out: &mut Vec<MemAlnReg>,
    meta: &mut Vec<RegMeta>,
    preskip: &mut Vec<bool>,
) {
    if chain.seeds.is_empty() {
        return;
    }
    // Read length in bases, and the length of the packed forward reference. Every reference
    // coordinate below lives in `[0, 2*l_pac)`: forward strand first, reverse complement after.
    let l_query = codes.len() as i32;
    let l_pac = bns.l_pac;

    // ================= step 1: the reference window this chain may extend inside =================
    // Reference window spanning the chain (`bwamem.cpp:2145-2166`).
    //
    // The window must be wide enough that no extension can want a reference base outside it. For
    // each seed we push the left edge back by the unaligned query prefix (`qbeg`) plus the longest
    // gap that prefix could justify, and the right edge forward by the unaligned suffix plus its
    // own max gap. Taking min/max over all seeds of the chain gives one window covering all of
    // them. Seeded with the *inverted* extremes (`rmax0 = 2*l_pac`, `rmax1 = 0`) so the first
    // seed's own bounds win, which is how the C initialises it too.
    //
    // INVARIANT across the loop: `[rmax0, rmax1)` is the smallest window covering every seed
    // examined so far together with the reference each of them could still extend into. The
    // interval is empty (`rmax0 > rmax1`) only before the first seed is folded in.
    let mut rmax0 = l_pac << 1;
    let mut rmax1 = 0i64;
    for seed in &chain.seeds {
        // `qbeg` is the seed's start in the read, so it is also the length of the unaligned prefix.
        // Leftmost reference base a left extension of this seed could ever touch: the seed start
        // pulled back by the unaligned prefix plus the longest gap that prefix could justify.
        let leftmost_needed =
            seed.rbeg - (i64::from(seed.qbeg) + i64::from(cal_max_gap(opt, seed.qbeg)));
        // Read bases right of this seed, i.e. what a right extension still has to spend.
        let unaligned_suffix_len = l_query - seed.qbeg - seed.len;
        // Mirror of `leftmost_needed`: one past the rightmost reference base a right extension
        // could touch.
        let rightmost_needed = seed.rbeg
            + i64::from(seed.len)
            + (i64::from(unaligned_suffix_len)
                + i64::from(cal_max_gap(opt, unaligned_suffix_len)));
        rmax0 = rmax0.min(leftmost_needed);
        rmax1 = rmax1.max(rightmost_needed);
    }
    // Clamp to the concatenated forward+reverse pac, then refuse to straddle the strand boundary
    // at `l_pac`. Positions `[0, l_pac)` are the forward genome and `[l_pac, 2*l_pac)` its reverse
    // complement, so a window crossing `l_pac` would splice the end of the forward strand onto the
    // start of the reverse one and extend into nonsense. bwa resolves it by keeping whichever half
    // contains the chain's first seed (`bwamem.cpp:2162-2166`).
    rmax0 = rmax0.max(0);
    rmax1 = rmax1.min(l_pac << 1);
    if rmax0 < l_pac && l_pac < rmax1 {
        if chain.seeds[0].rbeg < l_pac {
            rmax1 = l_pac;
        } else {
            rmax0 = l_pac;
        }
    }
    // `bns_fetch_seq`: trim the window to the seed's contig so extension cannot run off its end
    // into the next contig's sequence (visible on the circular MT genome).
    // The final (shadowing, now immutable) window bounds. `_rid` is the contig the window was
    // trimmed to; unused here because the region takes its `rid` from the chain instead.
    let (rmax0, rmax1, _rid) = bns.fetch_bounds(rmax0, rmax1, chain.seeds[0].rbeg);
    // The window's reference bases, materialised once and shared by every seed of the chain.
    // Index `i` of `rseq` is absolute pac position `rmax0 + i`; that offset is the source of every
    // `- rmax0` / `+ rmax0` conversion below.
    let rseq: Vec<u8> = (rmax0..rmax1).map(|p| fm.base(p)).collect();

    // ===================== step 2: decide the order the seeds are extended in =====================
    // Seeds in descending (score, index) order (`bwamem.cpp:2190-2192` plus the `for (k = c->n-1;
    // k >= 0; k--)` walk at `:2205`).
    //
    // OPERATION: the C packs each seed into `(uint64_t)score << 32 | i`, introsorts ascending, then
    // iterates *backwards*. Packing the index into the low 32 bits makes the sort total, so equal
    // scores break ties by index and the result does not depend on the sort's stability. We get the
    // same order directly by sorting on `Reverse(score << 32 | i)`. The `score as u32` cast is
    // deliberate: it reproduces the C's reinterpretation of a signed seed score into the high half
    // of an unsigned key, so a hypothetical negative score would sort high in both.
    //
    // This order is load-bearing, not cosmetic: it fixes the order regions are appended to `out`,
    // and bwa-mem2's discard pass (`bwamem.cpp:2921-2985`) walks the region array in exactly that
    // order, with its `lim` counter advancing as it goes. Permuting equal-score seeds would change
    // which regions get purged, and so the SAM.
    // Seed indices into `chain.seeds`, permuted into the order the seeds will be extended in
    // (highest seed score first, ties broken by highest index).
    let mut extension_order: Vec<usize> = (0..chain.seeds.len()).collect();
    extension_order.sort_by_key(|&seed_idx| {
        std::cmp::Reverse((u64::from(chain.seeds[seed_idx].score as u32) << 32) | seed_idx as u64)
    });

    // Same contained-seed extension skip as the batched path (they must stay region-identical), and
    // the same mutual exclusion with the discard pass, which needs one slot per seed.
    //
    // WHY MUTUALLY EXCLUSIVE: bwa-mem2 moved the "this seed is already covered" test out of the
    // extension loop into a separate post-pass (`bwamem.cpp:2921`) that runs only after every chain
    // of every read has been extended, and purges by setting `ar->qb = ar->qe = -1`. That post-pass
    // indexes regions positionally, so it requires one region slot per seed to exist. The
    // `skip_contained` optimisation instead avoids the DP up front; it is a pure speed lever and is
    // only sound when the discard pass is off. Both are env-gated in `across.rs`.
    // True when the up-front "this seed is already covered" skip is enabled. Read once per chain
    // rather than per seed because it is a process-wide env gate that cannot change mid-run.
    let skip_contained = crate::across::skip_contained_enabled();

    // ============ step 3: extend each seed into a region, in the order chosen above ============
    // `order_pos` is the rank in the extension order (0 = highest-scoring seed) and is what the
    // discard pass replays; `seed_idx` is the seed's original index in `chain.seeds`. Both are
    // recorded in the `RegMeta` pushed for this region, in the skip branch as well as the normal one.
    for (order_pos, &seed_idx) in extension_order.iter().enumerate() {
        if skip_contained && crate::across::seed_ext_redundant(&chain.seeds, seed_idx) {
            // Keep the slot (the discard pass reproduces bwa-mem2's scan order), skip the DP.
            // The placeholder uses the same `qb = qe = -1` purge marker the C's discard pass
            // writes, so the `retain(|reg| reg.qe > reg.qb)` compaction downstream removes it
            // identically.
            meta.push(RegMeta {
                chain: ci as u32,
                pos: order_pos as u32,
                seed: seed_idx as u32,
            });
            preskip.push(true);
            out.push(MemAlnReg {
                rb: -1,
                re: -1,
                qb: -1,
                qe: -1,
                rid: chain.rid,
                score: -1,
                truesc: -1,
                sub: 0,
                csub: 0,
                sub_n: 0,
                seedcov: 0,
                seedlen0: chain.seeds[seed_idx].len,
                secondary: -1,
                secondary_all: -1,
                w: opt.w,
                frac_rep: chain.frac_rep,
                is_alt: chain.is_alt,
                hash: 0,
                n_comp: 1,
            });
            continue;
        }
        // The seed being grown, copied out by value: `qbeg` (start in the read), `rbeg` (start in
        // pac coordinates), `len` (exact-match length) and `score`.
        let seed = chain.seeds[seed_idx];
        // `reg` is the region under construction; its `rb`/`re`/`qb`/`qe` are the reference and
        // query bounds, and start out as the -99 "never set" sentinel. See the module glossary.
        let mut reg = MemAlnReg {
            rb: H0_SENTINEL,
            re: H0_SENTINEL,
            qb: H0_SENTINEL as i32,
            qe: H0_SENTINEL as i32,
            rid: chain.rid,
            score: -1,
            truesc: -1,
            sub: 0,
            csub: 0,
            sub_n: 0,
            seedcov: 0,
            seedlen0: seed.len,
            secondary: -1,
            secondary_all: -1,
            w: opt.w,
            frac_rep: chain.frac_rep,
            is_alt: chain.is_alt,
            hash: 0,
            n_comp: 1,
        };

        // ---------------------------- step 3a: extend leftwards ----------------------------
        // Left extension (`bwamem.cpp:2229` onward, accepted at `:2495`).
        //
        // Both sequences are REVERSED so the DP can run in its usual left-to-right direction while
        // actually growing leftwards from the seed. That is why `qle`/`tle` are then *subtracted*
        // from the seed's start to get `qb`/`rb`. `ref_bases_available` is how much reference lies
        // between the window start and the seed, so the reversed reference slice is exactly the
        // bases the extension may consume. `h0` seeds the DP with the score the exact seed match
        // already earned: `seed.len` matching bases at `opt.a` each.
        if seed.qbeg > 0 {
            // `qs`/`rs`: the query and reference slices for the DP, reversed here so that "forward"
            // for the kernel means "leftwards" for us.
            let qs: Vec<u8> = (0..seed.qbeg).rev().map(|i| codes[i as usize]).collect();
            // Reference bases lying between the window start and the seed, i.e. the most this
            // left extension could consume. Nonnegative because `rmax0 <= seed.rbeg` by construction.
            let ref_bases_available = (seed.rbeg - rmax0) as usize;
            let rs: Vec<u8> = (0..ref_bases_available).rev().map(|i| rseq[i]).collect();
            // Score already banked before this DP starts: the seed's `len` exact matches at `a` each.
            let h0 = seed.len * opt.a;
            // `prev0 = -1`: the region has not been scored yet on this side, which is the -1 the C
            // initialises `a->score` to (`bwamem.cpp:2218`).
            // The accepted left-extension answer (clipped optimum, run-to-end optimum, band width).
            let left = extend_side(&qs, &rs, opt, opt.pen_clip5, h0, -1);
            reg.score = left.score;
            // Clip or run to the read's 5' end? Take the to-end alignment only when it exists
            // (`gscore > 0`; `ksw_extend2` returns -1 when no path reached the query end) AND it
            // beats the clipped optimum by more than the clip penalty. Written as
            // `gscore <= score - pen_clip5` so the CLIPPED branch is the one taken on ties, exactly
            // as the C orders it at `bwamem.cpp:2499`.
            if left.gscore <= 0 || left.gscore <= left.score - opt.pen_clip5 {
                reg.qb = seed.qbeg - left.qle;
                reg.rb = seed.rbeg - i64::from(left.tle);
                reg.truesc = left.score;
            } else {
                reg.qb = 0;
                reg.rb = seed.rbeg - i64::from(left.gtle);
                reg.truesc = left.gscore;
            }
            reg.w = reg.w.max(left.w);
        } else {
            // The seed already starts at query base 0, so there is nothing to extend into. Score is
            // just the seed's own match score and the region begins exactly at the seed.
            reg.score = seed.len * opt.a;
            reg.truesc = reg.score;
            reg.qb = 0;
            reg.rb = seed.rbeg;
        }

        // ---------------------------- step 3b: extend rightwards ----------------------------
        // Right extension (`bwamem.cpp:2327` onward, accepted at `:2417`).
        //
        // No reversal here: the DP direction and the growth direction agree. `re` is the seed's end
        // expressed as an offset into `rseq` (hence the `- rmax0`), and `re_abs` converts it back to
        // an absolute pac coordinate for the result. Crucially `h0 = reg.score`, the score CARRIED
        // OVER from the left extension, which is what chains the two halves into one alignment.
        if seed.qbeg + seed.len != l_query {
            // `qe` here is the seed's end in the read; `re` is the seed's end as an offset into the
            // window slice `rseq` (not an absolute coordinate, hence `re_abs` below).
            let qe = seed.qbeg + seed.len;
            let re = seed.rbeg + i64::from(seed.len) - rmax0;
            // The read suffix and reference suffix past the seed: what this extension may consume.
            // No reversal, unlike the left side, because DP direction and growth direction agree.
            let qs: Vec<u8> = codes[qe as usize..].to_vec();
            let rs: Vec<u8> = rseq[re as usize..].to_vec();
            // Score carried over from the left extension (or from the bare seed if there was none):
            // starting here is what welds the two halves into a single cumulative alignment score.
            let h0 = reg.score;
            // `prev0 = reg.score`, which is also this side's `h0`: the C's `int prev = a->score;`
            // reads the left extension's result on the very first band try.
            // The accepted right-extension answer; its `score` is CUMULATIVE (it includes `h0`).
            let right = extend_side(&qs, &rs, opt, opt.pen_clip3, h0, reg.score);
            reg.score = right.score;
            // The seed's end back in absolute pac coordinates, the base `right.tle`/`right.gtle`
            // are measured from.
            let re_abs = rmax0 + re;
            // Same clip-versus-run-to-end decision as the left side, now against the 3' penalty.
            //
            // OPERATION: `truesc += right.score - h0` and not `= right.score`. Because `h0` was the
            // left side's score, `right.score` is a CUMULATIVE score for both halves; `- h0` peels
            // off the left contribution so only this side's increment is added to a `truesc` that
            // already holds the left side's own (possibly `gscore`-based) value. Assigning directly
            // would silently discard the left extension's global-versus-clipped choice.
            if right.gscore <= 0 || right.gscore <= right.score - opt.pen_clip3 {
                reg.qe = qe + right.qle;
                reg.re = re_abs + i64::from(right.tle);
                reg.truesc += right.score - h0;
            } else {
                reg.qe = l_query;
                reg.re = re_abs + i64::from(right.gtle);
                reg.truesc += right.gscore - h0;
            }
            reg.w = reg.w.max(right.w);
        } else {
            reg.qe = l_query;
            reg.re = seed.rbeg + i64::from(seed.len);
        }

        // ------------------- step 3c: how much of the region is seed-backed -------------------
        // Seed coverage within the region (`bwamem.cpp:2507-2513`).
        //
        // Sums the lengths of every seed of the chain that lies wholly inside the finished region,
        // in BOTH query and reference. It answers "how many of these bases are backed by an exact
        // seed match rather than by DP guesswork", and feeds MAPQ and the dedup pass. Note it
        // counts overlapping seeds twice, which is a quirk we keep because the C does.
        //
        // The `H0_SENTINEL` guard is the C's way of asking "was this region actually extended".
        // The `H0_SENTINEL as i32` cast is required only because the constant is typed `i64` here
        // while `qb` is `i32`; both sides are -99 either way.
        if reg.rb != H0_SENTINEL && reg.qb != H0_SENTINEL as i32 {
            reg.seedcov = 0;
            for other_seed in &chain.seeds {
                if other_seed.qbeg >= reg.qb
                    && other_seed.qbeg + other_seed.len <= reg.qe
                    && other_seed.rbeg >= reg.rb
                    && other_seed.rbeg + i64::from(other_seed.len) <= reg.re
                {
                    reg.seedcov += other_seed.len;
                }
            }
        }
        meta.push(RegMeta {
            chain: ci as u32,
            pos: order_pos as u32,
            seed: seed_idx as u32,
        });
        preskip.push(false);
        out.push(reg);
    }
}

/// Align one read (2-bit codes) through seeding -> chaining -> extension, returning all regions.
///
/// The three stages are bwa's own: `build_chains` collects SMEM seeds from the FM index and groups
/// collinear ones into chains, `mem_chain_flt` drops chains dominated by better ones, and
/// `mem_chain2aln_meta` grows every surviving seed into a scored region. No dedup, no primary
/// marking, no MAPQ: callers layer those on, because the PE path needs the raw region set.
///
/// # Parameters
///
/// - `fm`: the FM index, used both for seeding (SMEM search, SA lookups) and for reading reference
///   bases during extension.
/// - `bns`: the contig table, for window clamping and contig assignment.
/// - `opt`: the run's scoring, band and chaining options.
/// - `codes`: the read as 2-bit codes (0..=3, 4 for `N`). Must be the already-oriented read as it
///   came off the FASTQ; reverse-strand handling happens via pac coordinates, not by flipping this.
///
/// # Returns
///
/// Every surviving region for the read, in extension order with purged entries compacted out.
/// Possibly empty (an unmappable read). Not deduplicated and not primary-marked.
pub fn align_read(fm: &FmIndex, bns: &BntSeq, opt: &MemOpt, codes: &[u8]) -> Vec<MemAlnReg> {
    // The read's chains after weak ones are dropped. The trailing 0 to `build_chains` is the seed
    // batch/read id; unused on this scalar path.
    let chains = mem_chain_flt(opt, build_chains(fm, bns, opt, codes, 0));
    // Three parallel arrays filled across ALL chains of the read: the regions themselves, their
    // provenance, and the "was the DP skipped" flag. They stay index-aligned throughout, which is
    // what lets the discard pass below index all three by one position.
    let mut regs = Vec::new();
    let mut meta = Vec::new();
    let mut preskip = Vec::new();
    for (chain_idx, chain) in chains.iter().enumerate() {
        mem_chain2aln_meta(
            fm, bns, opt, codes, chain_idx, chain, &mut regs, &mut meta, &mut preskip,
        );
    }
    // bwa-mem2 purges covered seeds only once every chain of the read has been extended, so this
    // cannot live inside the per-chain body above.
    if crate::across::discard_enabled() {
        crate::across::discard_contained(
            opt,
            codes.len() as i32,
            &chains,
            &mut regs,
            &meta,
            &preskip,
        );
    }
    // Drop the regions the discard pass purged. bwa-mem2 marks them in place with
    // `ar->qb = ar->qe = -1` (`bwamem.cpp:2979`) rather than removing them, so `qe > qb` is exactly
    // the "survived" predicate. We must compact here and not leave the holes in, because the dedup
    // that follows sorts with an UNSTABLE sort (`ks_introsort`), and dead entries participating in
    // that sort would perturb the final order of the live ones. Same compaction, same reason, in
    // the batched path.
    regs.retain(|reg| reg.qe > reg.qb);
    regs
}

/// Align one read and deduplicate its regions (`mem_sort_dedup_patch`), WITHOUT primary marking.
/// This is the per-read input to paired-end statistics (`mem_pestat`) and pairing, which mark
/// primaries themselves.
///
/// # Parameters
///
/// Same as [`align_read`]: `fm` the FM index, `bns` the contig table, `opt` the scoring options,
/// `codes` the read as 2-bit codes. `codes` is forwarded to `mem_sort_dedup_patch` as well, which
/// re-scores merged regions and therefore needs the read bases again.
///
/// # Returns
///
/// The deduplicated region set, in `mem_sort_dedup_patch`'s output order.
pub fn align_read_dedup(fm: &FmIndex, bns: &BntSeq, opt: &MemOpt, codes: &[u8]) -> Vec<MemAlnReg> {
    // Raw regions before dedup, dumped first so a divergence can be attributed to extension rather
    // than to the dedup pass.
    let regs = align_read(fm, bns, opt, codes);
    if std::env::var_os("BWA3_DUMP_REGS").is_some() {
        dump_regs(bns, "pre-dedup", &regs);
    }
    let deduped = mem_sort_dedup_patch(fm, opt, codes, regs);
    if std::env::var_os("BWA3_DUMP_REGS").is_some() {
        dump_regs(bns, "post-dedup", &deduped);
    }
    deduped
}

/// Env-gated (`BWA3_DUMP_REGS`) diagnostic: print every region with its query span, reference
/// span, mapped position and scores. Used to compare our suboptimal-region set against the oracle.
///
/// # Parameters
///
/// - `bns`: the contig table, needed to turn each region's pac coordinates into a contig and a
///   1-based position.
/// - `tag`: a free-form label printed in the header line so several dumps of the same read (for
///   example `"pre-dedup"` and `"post-dedup"`) can be told apart in the log.
/// - `regs`: the regions to print, in their current order; the printed `#i` is that order's index,
///   which is what makes two dumps diffable line by line.
pub fn dump_regs(bns: &BntSeq, tag: &str, regs: &[MemAlnReg]) {
    eprintln!("--- regs [{}] n={} ---", tag, regs.len());
    for (i, r) in regs.iter().enumerate() {
        // Contig index, 1-based contig-local position and strand, as the SAM record would carry them.
        let (rid, pos, rev) = region_to_pos(bns, r);
        let strand = if rev { '-' } else { '+' };
        eprintln!(
            "  #{i} q[{},{}) r[{},{}) rid={rid} {strand}pos={pos} score={} truesc={} sub={} sub_n={} seedcov={} seedlen0={} frac_rep={}",
            r.qb, r.qe, r.rb, r.re, r.score, r.truesc, r.sub, r.sub_n, r.seedcov, r.seedlen0, r.frac_rep
        );
    }
}

/// Full single-end alignment for one read: extension regions, deduplicated and primary-marked
/// (`sub`/`sub_n` set for MAPQ). `read_id` is the global read index (for the `hash` tie-break).
///
/// # Parameters
///
/// - `fm`, `bns`, `opt`, `codes`: as in [`align_read`].
/// - `read_id`: the read's global 0-based index in the input file. It is not used for alignment at
///   all; `mem_mark_primary_se` mixes it into each region's `hash` so that equal-comparing regions
///   get a deterministic, read-dependent tie-break instead of one that depends on sort stability.
///   Callers must supply the same id for the same read on every run, or the SAM changes.
///
/// # Returns
///
/// The read's deduplicated, primary-marked regions with `sub`/`sub_n` filled in, ready for MAPQ
/// and SAM emission.
pub fn align_read_se(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    read_id: u64,
) -> Vec<MemAlnReg> {
    let regs = align_read(fm, bns, opt, codes);
    let mut regs = mem_sort_dedup_patch(fm, opt, codes, regs);
    mem_mark_primary_se(opt, &mut regs, read_id);
    regs
}

/// The 1-based mapping position of a region: `(rid, pos, is_rev)`, mirroring `mem_reg2aln`'s
/// coordinate derivation (`bns_depos` on `rb` for forward, `re-1` for reverse).
///
/// WHY THE ASYMMETRIC PROBE: a region on the reverse strand occupies `[rb, re)` in the *reverse*
/// half of the pac. Mapping it back to forward coordinates flips the interval, so the base that
/// becomes the leftmost forward position is the one at `re - 1`, not at `rb`. Probing `rb` for a
/// forward region and `re - 1` for a reverse one therefore picks the leftmost forward base in both
/// cases, which is what SAM's POS field means. `reg.rb < bns.l_pac` is the strand test, valid
/// because the window clamp above guarantees a region never straddles `l_pac`.
///
/// `bns.depos` returns the forward-strand position and the strand flag; `pos2rid` then locates the
/// contig, and subtracting that contig's `offset` converts a genome-wide offset into a
/// contig-local one. The `+ 1` is SAM's 1-based convention. `rid < 0` (position outside every
/// contig, which happens for the holes bwa inserts between contigs) falls back to offset 0 rather
/// than panicking, matching the C's lack of a bounds check.
///
/// # Parameters
///
/// - `bns`: the contig table, supplying `l_pac` (the strand-boundary test), `depos`, `pos2rid` and
///   the per-contig `offset`.
/// - `reg`: the region to place. Only `rb` and `re` are read, and they must be real coordinates:
///   a region still holding the [`H0_SENTINEL`] bounds, or one purged to `qb = qe = -1`, will
///   produce a meaningless position rather than an error.
///
/// # Returns
///
/// `(rid, pos, is_rev)`: the contig index (-1 when the position falls in an inter-contig hole), the
/// 1-based contig-local leftmost forward position as SAM's POS field wants it, and whether the
/// region aligned to the reverse strand.
pub fn region_to_pos(bns: &BntSeq, reg: &MemAlnReg) -> (i32, i64, bool) {
    // The single pac coordinate whose forward image is the region's leftmost forward base: `rb` on
    // the forward strand, `re - 1` on the reverse (see WHY THE ASYMMETRIC PROBE above).
    let probe = if reg.rb < bns.l_pac {
        reg.rb
    } else {
        reg.re - 1
    };
    // `fpos` is `probe` mapped into the forward half `[0, l_pac)`; `is_rev` records which half it
    // came from. `rid` is the contig containing `fpos`, or -1 in an inter-contig hole.
    let (fpos, is_rev) = bns.depos(probe);
    let rid = bns.pos2rid(fpos);
    // Genome-wide start of that contig, subtracted to make the position contig-local.
    let offset = if rid >= 0 {
        bns.contigs[rid as usize].offset
    } else {
        0
    };
    (rid, fpos - offset + 1, is_rev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Load the checked-in single-contig test reference (`testdata/tiny/tiny.fa`) and return its
    /// FM index plus contig table. Every test below aligns synthetic reads cut out of it.
    fn tiny() -> (FmIndex, BntSeq) {
        // Index path prefix; `FmIndex::load`/`BntSeq::load` append their own suffixes to it.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    /// The highest-scoring region of `regs`. A stand-in for primary marking, which these tests do
    /// not run; `regs` must be nonempty.
    fn best(regs: &[MemAlnReg]) -> &MemAlnReg {
        regs.iter().max_by_key(|r| r.score).unwrap()
    }

    #[test]
    fn forward_slice_maps_to_origin() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        // Cut a 150-base read verbatim out of the reference at pac offset 40000, so the only
        // correct answer is a full-length exact match at that position.
        let start = 40_000i64;
        let len = 150i64;
        let read: Vec<u8> = (0..len).map(|i| fm.base(start + i)).collect();
        let regs = align_read(&fm, &bns, &opt, &read);
        assert!(!regs.is_empty());
        let b = best(&regs);
        assert_eq!(
            b.score,
            (len * i64::from(opt.a)) as i32,
            "full-length exact score"
        );
        let (rid, pos, is_rev) = region_to_pos(&bns, b);
        assert_eq!(rid, 0);
        assert!(!is_rev);
        assert_eq!(pos, start + 1); // 1-based
    }

    #[test]
    fn reverse_complement_slice_maps_reverse() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        let start = 90_000i64;
        let len = 150i64;
        // Reverse-complement of the forward slice: should map to the reverse strand at `start`.
        let fwd: Vec<u8> = (0..len).map(|i| fm.base(start + i)).collect();
        // `3 - c` complements a 2-bit base (A=0 <-> T=3, C=1 <-> G=2); reversing as well gives the
        // reverse complement, which must map to the same POS with the strand flag set.
        let read: Vec<u8> = fwd.iter().rev().map(|&c| 3 - c).collect();
        let regs = align_read(&fm, &bns, &opt, &read);
        assert!(!regs.is_empty());
        let b = best(&regs);
        assert_eq!(b.score, (len * i64::from(opt.a)) as i32);
        let (rid, pos, is_rev) = region_to_pos(&bns, b);
        assert_eq!(rid, 0);
        assert!(is_rev);
        assert_eq!(pos, start + 1);
    }

    #[test]
    fn mismatch_read_still_maps_with_expected_score() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        let start = 60_000i64;
        let len = 150i64;
        let mut read: Vec<u8> = (0..len).map(|i| fm.base(start + i)).collect();
        // one mismatch in the middle: base 75 rotated to a different code, far enough from both
        // ends that seeding still finds the position and the mismatch costs exactly `-b`.
        read[75] = (read[75] + 1) % 4;
        let regs = align_read(&fm, &bns, &opt, &read);
        let b = best(&regs);
        // 149 matches (+1) minus a mismatch (-b): 150 - 1 - 4 = 145.
        assert_eq!(b.score, (len * i64::from(opt.a)) as i32 - 1 - opt.b);
        let (_, pos, is_rev) = region_to_pos(&bns, b);
        assert!(!is_rev);
        assert_eq!(pos, start + 1);
    }
}

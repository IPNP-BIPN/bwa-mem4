//! CIGAR / NM / MD generation from an alignment region, mirroring bwa-mem2's `bwa_gen_cigar2`
//! (`reference/bwa-mem2/src/bwa.cpp`) and the CIGAR assembly of `mem_reg2aln`
//! (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! # What this file is responsible for
//!
//! Earlier pipeline stages (seeding, chaining, extension) end up with, for each candidate hit, a
//! *region*: a pair of half-open intervals saying "read bases `[qb, qe)` line up with reference
//! bases `[rb, re)`, and that scored `score`". A region says where the alignment starts and ends
//! but not how the interior lines up. This file is the stage that materializes that interior: it
//! re-runs a banded *global* alignment between those already-fixed end points and turns the result
//! into the three things SAM wants: a CIGAR string, the `NM:i` edit distance and the `MD:Z` string.
//! Its output, [`MemAln`], is what the SAM writer formats into an output line.
//!
//! # What a CIGAR string is
//!
//! A CIGAR is SAM's compact description of how a read lines up with the reference, as a run-length
//! encoded list of operations read left to right along the reference, for example `12S88M2D30M`.
//! The operation codes that occur here:
//!
//! | Op | Name | Consumes read? | Consumes reference? | Meaning |
//! |----|------|----------------|---------------------|---------|
//! | `M` | match/mismatch | yes | yes | bases are aligned to each other; `M` does *not* claim they are equal, a mismatch is still `M` |
//! | `I` | insertion | yes | no | read has bases the reference does not |
//! | `D` | deletion | no | yes | reference has bases the read does not |
//! | `S` | soft clip | yes | no | read bases left unaligned but still stored in the SAM record |
//! | `H` | hard clip | no | no | read bases left unaligned and *not* stored, used on supplementary records |
//!
//! Internally a CIGAR is carried as a `Vec<u32>` in htslib's packed form, one `u32` per operation
//! holding `length << 4 | op_code`, with the op codes numbered `M=0, I=1, D=2, S=3, H=4`.
//!
//! Two sibling annotations travel with it. `NM:i` is the edit distance: mismatching bases plus the
//! total length of all gaps. `MD:Z` spells out the reference bases wherever the read disagrees, as
//! alternating match run lengths and reference letters (`10A5^AC6`), which lets a reader
//! reconstruct the reference sequence from the read plus the CIGAR alone.
//!
//! # Reading order
//!
//! 1. [`MemAln`], the output record, and [`MemAln::unmapped`].
//! 2. [`infer_bw`], the band width heuristic, small and self-contained.
//! 3. [`gen_cigar2`], the core: global alignment plus NM/MD construction.
//! 4. [`reg2aln`], the driver: band retry loop, edge-deletion squeeze, soft clips, position mapping.
//! 5. [`cigar_string`] / [`cigar_string_which`], the formatters.
//!
//! # Glossary of names kept from the C
//!
//! Byte-identical SAM output is a hard requirement here, so names that make the correspondence with
//! bwa-mem2's source auditable are kept verbatim rather than "improved". In plain language:
//!
//! | Name | Plain meaning |
//! |------|---------------|
//! | `qb`, `qe` | query begin / query end: half-open range of read bases covered by this alignment |
//! | `rb`, `re` | reference begin / end, in *packed* reference coordinates (see below) |
//! | `l_query` | length of the query (read), in bases |
//! | `l_pac` | length of the forward genome in the packed reference |
//! | `rlen` | length of the reference interval, `re - rb` |
//! | `score` | Smith-Waterman alignment score; reported as `AS:i` |
//! | `score2`, `sub`, `csub` | suboptimal scores, the best score of a *competing* alignment; reported as `XS:i` |
//! | `truesc` | the region's local alignment score, the target the global re-alignment is compared against |
//! | `w`, `w2`, `w_` | band width: how far off the main diagonal the alignment is allowed to wander |
//! | `qle`, `tle` | query-length-extended / target-length-extended: how far an extension ran on each side |
//! | `NM` / `nm` | edit distance (mismatches + gap bases) |
//! | `l_MD` | byte length of the MD string (a C bookkeeping quantity; we use a `String` instead) |
//! | `n_cigar` | number of CIGAR operations |
//! | `pos` | 0-based reference position; SAM `POS` is `pos + 1` |
//! | `flag` | SAM FLAG bit field (0x4 unmapped, 0x10 reverse strand, 0x100 secondary, ...) |
//! | `mapq` | mapping quality, 0..=60 |
//! | `rid` | reference (contig) id, an index into the contig table; `-1` means unmapped |
//! | `n_mm`, `n_gap` | counts of mismatching bases and of gap bases; `NM = n_mm + n_gap` |
//! | `mat` | the 5x5 nucleotide scoring matrix, row-major, indexed `ref_base * 5 + query_base` |
//! | `o_del`, `e_del`, `o_ins`, `e_ins` | gap open / extend penalties, for deletions and insertions |
//! | `infer_bw` | the C function name for the band width heuristic |
//! | nt4 code | a base encoded as a small integer: `A=0, C=1, G=2, T=3, N=4` |
//!
//! *Packed reference coordinates*: the reference is stored 2 bits per base as the forward genome
//! followed by its reverse complement, so coordinates run `[0, 2 * l_pac)` and a coordinate at or
//! above `l_pac` simply *is* a reverse-strand coordinate. There is no separate strand flag at this
//! level.

// ---------------------------------------------------------------------------------------------
// CIGAR operation codes, as packed into the low 4 bits of each `u32`. These are htslib's `BAM_C*`
// constants (`htslib/sam.h`), which bwa uses via the string "MIDSH": M=0, I=1, D=2, S=3, H=4.
// ---------------------------------------------------------------------------------------------

/// Low 4 bits of a packed CIGAR word hold the op; the rest is the run length.
const CIGAR_OP_MASK: u32 = 0xf;
/// Op 0: aligned columns, matching or mismatching.
const CIGAR_OP_MATCH: u32 = 0;
/// Op 1: insertion (consumes query only).
const CIGAR_OP_INS: u32 = 1;
/// Op 2: deletion (consumes reference only).
const CIGAR_OP_DEL: u32 = 2;
/// Op 3: soft clip.
const CIGAR_OP_SOFT_CLIP: u32 = 3;
/// Op 4: hard clip. Never stored in [`MemAln::cigar`]; only synthesized at emission time.
const CIGAR_OP_HARD_CLIP: u32 = 4;

/// SAM FLAG 0x4, "segment unmapped".
const SAM_FLAG_UNMAPPED: u32 = 0x4;
/// SAM FLAG 0x100, "secondary alignment".
const SAM_FLAG_SECONDARY: u32 = 0x100;

use bwa_core::MemOpt;
use bwa_extend::ksw_global2;
use bwa_index::{BntSeq, FmIndex};

use crate::primary::mem_approx_mapq_se;
use crate::MemAlnReg;

/// A finalized single-end alignment (bwa-mem2's `mem_aln_t`, phase-6/7 subset).
#[derive(Debug, Clone)]
pub struct MemAln {
    /// Index into `BntSeq::contigs` of the contig this alignment landed on, or `-1` when unmapped.
    /// `mem_reg2aln` derives it with `bns_pos2rid` (`bntseq.cpp:378`) and then asserts it equals the
    /// region's own `rid`, so by construction it agrees with `MemAlnReg::rid`.
    pub rid: i32,
    /// 0-based position within the contig (SAM POS is `pos + 1`).
    pub pos: i64,
    /// True when the region sits on the reverse strand of the packed reference, i.e. `rb >= l_pac`.
    /// The 2-bit pac holds the forward genome followed by its reverse complement, so "reverse" is a
    /// property of the coordinate, not of a separate flag (`bns_depos`, `bntseq.h:87`).
    pub is_rev: bool,
    /// SAM MAPQ, 0..=60. Zero for secondary hits; otherwise `mem_approx_mapq_se`.
    pub mapq: u32,
    /// SAM FLAG bits set so far (0x4 unmapped, 0x100 secondary). Strand/pair bits are added by the
    /// SAM writer.
    pub flag: u32,
    /// CIGAR, `len<<4 | op` (op 0=M/1=I/2=D/3=S). This is htslib's packed encoding and bwa carries
    /// it verbatim; note that the ops here are only ever M/I/D/S, hard clips (op 4) are synthesized
    /// at emission time by [`cigar_string_which`], never stored.
    pub cigar: Vec<u32>,
    /// `NM:i`, edit distance: mismatches plus total gap length. Note the asymmetry inherited from
    /// `bwa_gen_cigar2` (`bwa.cpp:326`): a leading or trailing D contributes to neither NM nor MD,
    /// because such a D is squeezed out of the CIGAR moments later in `reg2aln`.
    pub nm: i32,
    /// `MD:Z` string, without the tag prefix. Always non-empty: `bwa_gen_cigar2` terminates it with
    /// a final match run length, which is `0` when the alignment ends on a mismatch.
    pub md: String,
    /// Smith-Waterman score of the region (`AS:i`). Copied from `MemAlnReg::score`, i.e. the *local*
    /// score, not the global score that the band-retry loop below recomputes.
    pub score: i32,
    /// Suboptimal score (`XS:i`), `max(sub, csub)`. `-1` suppresses the tag (secondary hits).
    pub sub: i32,
    /// Alternate hits (`XA:Z:`), pre-formatted `rname,±pos,cigar,NM;`... or `None`. Set by the
    /// caller via `mem_gen_alt`; `reg2aln` leaves it `None`.
    pub xa: Option<String>,
}

impl MemAln {
    /// An unmapped alignment (`mem_reg2aln` with a null region): `rid=-1`, FLAG 0x4, no CIGAR.
    #[must_use]
    pub fn unmapped() -> Self {
        MemAln {
            rid: -1,
            pos: -1,
            is_rev: false,
            mapq: 0,
            flag: SAM_FLAG_UNMAPPED,
            cigar: Vec::new(),
            nm: 0,
            md: String::new(),
            // Zero, not -1: bwa builds this from `memset(&a, 0, sizeof(mem_aln_t))`, so an unmapped
            // record clears `mem_aln2sam`'s `score >= 0` / `sub >= 0` guards and carries
            // `AS:i:0 XS:i:0`. Signalling "absent" with -1 silently drops both tags.
            score: 0,
            sub: 0,
            xa: None,
        }
    }
}

/// Env-gated (`BWA3_DUMP_BW`) trace of the band-width retry loop, in bwa's `-v 4` format so the two
/// can be diffed directly. Cached: `reg2aln` runs per emitted alignment, so a `var_os` per call
/// would show up in the profile.
///
/// # Returns
///
/// `true` when the environment variable `BWA3_DUMP_BW` was set at the time of the *first* call.
/// The value is frozen from then on: setting or clearing the variable later in the process has no
/// effect. Read only by [`reg2aln`], purely to decide whether to `eprintln!` a trace line.
fn band_width_trace_enabled() -> bool {
    // Process-wide latch holding the one-time answer, so the env lookup happens at most once for
    // the whole run rather than once per emitted alignment.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BWA3_DUMP_BW").is_some())
}

/// Inferred band width, port of `infer_bw` (`bwamem.cpp:1811`).
///
/// Given that a local alignment of query length `l1` against reference length `l2` scored `score`,
/// this bounds how far off the main diagonal the global re-alignment could possibly need to stray.
/// `a` is the match score, `q` the gap-open penalty and `r` the gap-extend penalty (all positive
/// magnitudes as stored in `MemOpt`; the caller passes the del pair and the ins pair separately and
/// takes the max, because a band must accommodate whichever gap type is cheaper).
///
/// The reasoning behind the arithmetic: `l1 * a` is the score a perfect ungapped alignment would
/// get, so `l1 * a - score` is the score deficit. A gap of length `g` costs `q + g*r`, hence the
/// deficit can pay for at most `(deficit - q) / r` extending bases, plus 2 for slack. The `min(l1,
/// l2)` picks the shorter side because the gap cannot be longer than that.
///
/// The early return encodes: when the two lengths are already equal, any indel must be paid for
/// *twice* (one insertion and one deletion) to keep the lengths matched, so if the deficit is below
/// `2*(q + r - a)` no such pair is affordable and a zero-width band (pure diagonal) suffices. That
/// zero is load-bearing downstream: `gen_cigar2` only takes its no-DP fast path when `w_ == 0`.
///
/// Overflow note: `l1 * a` is `int` arithmetic in C and `i32` here, so both would wrap identically
/// in release mode, but read lengths make this unreachable in practice.
///
/// # Parameters
///
/// - `l1`: query-side length of the region in bases, i.e. `qe - qb`. Supplied by [`reg2aln`].
///   Expected positive; a zero or negative value is not rejected here but yields a meaningless band.
/// - `l2`: reference-side length of the region in bases, i.e. `(re - rb) as i32`. Same supplier.
/// - `score`: the region's local alignment score (`MemAlnReg::truesc`), in score units. This is the
///   score the end points were chosen by, and the whole heuristic is "how much score is missing
///   relative to a perfect ungapped alignment of length `l1`".
/// - `a`: match score, a positive magnitude (`MemOpt::a`, default 1).
/// - `q`: gap *open* penalty, a positive magnitude (`MemOpt::o_del` or `o_ins`, default 6).
/// - `r`: gap *extend* penalty, a positive magnitude (`MemOpt::e_del` or `e_ins`, default 1). Must
///   be non-zero: it is a divisor.
///
/// The caller invokes this twice, once with the deletion pair `(o_del, e_del)` and once with the
/// insertion pair, and takes the max.
///
/// # Returns
///
/// Band half-width in bases, always `>= 0`, and always `>= (l1 - l2).abs()` except in the
/// early-return case where it is exactly `0`. A returned `0` is a signal, not just a small number:
/// [`gen_cigar2`] takes its no-DP ungapped fast path only when it receives `w_ == 0`.
#[must_use]
fn infer_bw(l1: i32, l2: i32, score: i32, a: i32, q: i32, r: i32) -> i32 {
    if l1 == l2 && l1 * a - score < (q + r - a) << 1 {
        return 0;
    }
    // The f64 round-trip is not cosmetic: C computes `((double)(...) / r + 2.)` and then truncates
    // toward zero on assignment to `int`. Doing this in integer arithmetic would floor instead, so a
    // negative numerator (possible when the score exceeds the deficit budget) would land one apart.
    //
    // `w` holds the candidate band half-width: the number of extend-penalty units the score deficit
    // can pay for, plus 2 bases of slack. It is then raised to at least the length difference,
    // because a band narrower than `|l1 - l2|` cannot reach the far corner of the DP matrix at all.
    let mut w = (f64::from(l1.min(l2) * a - score - q) / f64::from(r) + 2.0) as i32;
    if w < (l1 - l2).abs() {
        w = (l1 - l2).abs();
    }
    w
}

/// Global-align `query_codes` (the seed-region query slice) to reference `[rb, re)` and compute the
/// CIGAR, NM and MD. Port of `bwa_gen_cigar2` (`bwa.cpp:260`).
///
/// The seeding/extension phase produced only end points and a score; this is where an actual base
/// level alignment is materialized, by running a *banded global* (Needleman-Wunsch) alignment
/// between those fixed end points. Global, not local, is the point: the end points are already
/// decided, so the only question left is how the interior lines up.
///
/// # Parameters
///
/// - `fm`: the loaded FM index. Used here only for two things: `l_pac()` (length of the forward
///   genome, to tell forward coordinates from reverse ones) and `base(p)` (the 2-bit reference base
///   at packed coordinate `p`). No suffix-array lookups happen in this function.
/// - `opt`: the run's alignment parameters. Only the scoring fields are read: `mat` (5x5 substitution
///   matrix) and the four gap penalties `o_del`/`e_del`/`o_ins`/`e_ins`.
/// - `w_`: maximum band half-width in bases, supplied by the caller's retry loop. `0` means the
///   caller proved no gap is affordable (see [`infer_bw`]) and requests the ungapped fast path.
///   Note it is an upper bound only in part: an internal floor (`min_w`) can widen past it.
/// - `l_query`: length of `query_codes`, i.e. `qe - qb`, in bases. Must be `> 0` or we bail.
/// - `query_codes`: nt4 codes (0..=3 for ACGT, 4 for N) for the query slice `[qb, qe)`, in *pac*
///   orientation, not sequencing orientation. Its length must equal `l_query`; the slicing is done
///   by the caller ([`reg2aln`]), not here.
/// - `rb`, `re`: half-open interval in packed-reference coordinates, which run `[0, 2*l_pac)` with
///   the reverse complement occupying the upper half. Precondition enforced below: `rb < re` and
///   the interval must not straddle `l_pac`.
///
/// # Returns
///
/// `Some((global_score, cigar, nm, md))`, or `None` for the degenerate inputs the C rejects
/// (`l_query <= 0`, an empty or inverted reference interval, or one straddling the pac midpoint).
/// The score returned is the *global* score and is generally lower than the region's local
/// `truesc`; the retry loop in [`reg2aln`] compares the two. `cigar` is packed `len<<4|op` and, at
/// this point, contains only M/I/D (no clips, those are added by the caller). `nm` is the edit
/// distance and `md` the MD:Z body, both excluding any leading or trailing deletion.
pub(crate) fn gen_cigar2(
    fm: &FmIndex,
    opt: &MemOpt,
    w_: i32,
    l_query: i32,
    query_codes: &[u8],
    rb: i64,
    re: i64,
) -> Option<(i32, Vec<u32>, i32, String)> {
    // -------------------------------------------------------------------------------------
    // 1. Reject degenerate inputs.
    // -------------------------------------------------------------------------------------
    // `l_pac`: length of the forward genome; `rb`/`re`: reference begin/end in packed coordinates.
    let l_pac = fm.l_pac();
    // `rb < l_pac && re > l_pac` means the interval straddles the forward/reverse boundary of the
    // packed reference, which is a coordinate artifact rather than a real alignment. bwa.cpp:271
    // rejects it with the same three-way test.
    if l_query <= 0 || rb >= re || (rb < l_pac && re > l_pac) {
        return None;
    }

    // -------------------------------------------------------------------------------------
    // 2. Materialize both sequences to be aligned, in the orientation the aligner wants.
    // -------------------------------------------------------------------------------------
    // `rlen`: length of the reference interval.
    let rlen = (re - rb) as i32;
    // Owned copy of the query codes, because the reverse-strand branch below mutates it in place.
    // Length `l_query`; indexed by the `query_pos` cursor during the NM/MD walk.
    let mut query: Vec<u8> = query_codes.to_vec();
    // The reference side of the alignment: `rlen` nt4 codes for packed positions `[rb, re)`. We
    // materialize the reference slice base by base rather than calling a `bns_get_seq`
    // equivalent; `fm.base(p)` already handles the reverse-strand half of the pac.
    let mut rseq: Vec<u8> = (rb..re).map(|p| fm.base(p)).collect();
    // Reverse both sides for reverse-strand hits (bwa.cpp:274). This is NOT a complement, only an
    // order reversal, and it exists purely so that ties in gap placement resolve to the leftmost
    // position *in forward-genome coordinates*. Dropping it still yields valid alignments, just
    // with indels shifted right on minus-strand reads: a byte-parity break, not a correctness one.
    if rb >= l_pac {
        query.reverse();
        rseq.reverse();
    }
    // `mat[0]` is the A/A cell of the 5x5 scoring matrix, i.e. the match score `opt.a`. bwa reads
    // it out of the matrix rather than using `opt->a` directly, and we copy that.
    let match_score = i32::from(opt.mat[0]);

    // -------------------------------------------------------------------------------------
    // 3. Produce the alignment itself: either the ungapped fast path or a banded global DP.
    // -------------------------------------------------------------------------------------
    // `score`: the global alignment score of the whole `[qb, qe)` x `[rb, re)` block, in score
    // units. `cigar`: the packed M/I/D operations describing that alignment, left to right along
    // the reference. Both come from one branch or the other and are final from here on; step 4 only
    // reads them.
    let (score, cigar) = if l_query == rlen && w_ == 0 {
        // Ungapped fast path (bwa.cpp:280): equal lengths and a proven-zero band mean the CIGAR is
        // a single M run and the score is just the sum of matrix cells down the diagonal. The
        // matrix is row-major 5x5 indexed `[ref_base * 5 + query_base]`; the operand order matters
        // only for asymmetric matrices, but we keep bwa's order regardless.
        //
        // Invariant: at the top of iteration `i`, `diagonal_score` is the summed substitution score
        // of columns `0..i`. There is no gap term because this path has no gaps by construction.
        let mut diagonal_score = 0;
        for i in 0..l_query as usize {
            diagonal_score += i32::from(opt.mat[rseq[i] as usize * 5 + query[i] as usize]);
        }
        // `(len << 4) | 0`: op 0 is M, so the `| 0` is elided.
        (diagonal_score, vec![(l_query as u32) << 4])
    } else {
        // Derive an internal band from affordability, then intersect it with the caller's `w_`.
        // `((l_query+1)>>1) * match_score` is the score of matching *half* the query: bwa's
        // heuristic is that a gap can never be worth more than half the read's best-case score.
        // Subtract the gap open penalty and divide by the extend penalty to get a maximum gap
        // length. Insertions and deletions get separate budgets because their penalties can differ.
        //
        // `max_ins` / `max_del`: longest insertion / deletion, in bases, that half the query's
        // best-case score could still pay for. Can come out zero or negative on very short queries,
        // which is what the `.max(1)` below is for.
        let max_ins = (f64::from(((l_query + 1) >> 1) * match_score - opt.o_ins)
            / f64::from(opt.e_ins)
            + 1.0) as i32;
        let max_del = (f64::from(((l_query + 1) >> 1) * match_score - opt.o_del)
            / f64::from(opt.e_del)
            + 1.0) as i32;
        // `max_gap`: the affordable gap length in bases, taking whichever gap type is cheaper.
        // `.max(1)`: a zero or negative budget would produce a nonsensical band.
        let max_gap = max_ins.max(max_del).max(1);
        // Half-width: the band must cover the length difference plus the affordable gap, and `>> 1`
        // converts a full width into a half-width. `+1` rounds up before the shift.
        //
        // `w` holds the band half-width in bases that will actually be handed to the DP, after the
        // two clamps below. It starts as this function's own estimate, is capped by the caller's
        // `w_`, then floored at `min_w`.
        let mut w = (max_gap + (rlen - l_query).abs() + 1) >> 1;
        // Clamp down to what the caller allows, then clamp back *up* to `min_w`. The order is not
        // interchangeable: `min_w` wins over `w_`, so a caller-supplied band is silently widened
        // when it is too narrow to even span the length difference. Without that floor the global
        // alignment would be infeasible (no path reaches the far corner) rather than merely poor.
        w = w.min(w_);
        // `min_w`: the narrowest band in which a global alignment is even feasible, the length
        // difference the band must absorb plus 3 bases of slack (bwa's constant, not derived).
        let min_w = (rlen - l_query).abs() + 3;
        w = w.max(min_w);
        ksw_global2(
            &query, &rseq, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, w,
        )
    };

    // -------------------------------------------------------------------------------------
    // 4. Derive NM (edit distance) and MD (reference-difference string) from that CIGAR.
    // -------------------------------------------------------------------------------------
    // NM + MD, walking the CIGAR we just produced (bwa.cpp:309). The C writes MD into the same
    // allocation as the CIGAR, immediately past the last op, which is why `mem_reg2aln` later has
    // to `memmove` an `l_MD` tail around whenever it edits the CIGAR. We keep MD in its own
    // `String`, so those moves have no counterpart here.
    // True when the whole interval lies in the forward half of the pac. Safe to decide from `rb`
    // alone: the straddling case was rejected in step 1, so `rb` and `re` are on the same side.
    // Read only by `base_char` below, to pick which alphabet MD letters are spelled with.
    let is_forward_strand = rb < l_pac;
    // MD reports reference bases as sequencing-orientation characters. For a reverse-strand hit the
    // stored pac codes are already the reverse-complement strand's, and `rseq` was order-reversed
    // above, so complementing the *alphabet* ("TGCAN" instead of "ACGTN") is what turns the codes
    // back into the letters bwa prints. Same trick as bwa.cpp:312's `int2base` selection.
    //
    // Maps one nt4 code (0..=4, anything higher clamped to N) to the single MD character for it.
    // Captures `is_forward_strand`; used for both mismatch letters and deleted-base runs.
    let base_char = |code: u8| -> char {
        let alphabet = if is_forward_strand {
            *b"ACGTN"
        } else {
            *b"TGCAN"
        };
        // `.min(4)`: pac codes above 3 (ambiguous bases) all collapse onto N.
        alphabet[code.min(4) as usize] as char
    };
    // The MD:Z body under construction, complete except for its final run length (appended after
    // the loop). Grows as alternating decimal run lengths and reference letters.
    let mut md = String::new();
    // Three cursors, named `x`, `y` and `u` in the C: the query position, the reference position,
    // and the length of the match run currently being accumulated (the integers that alternate with
    // the letters in an MD string).
    //
    // Invariant at the top of each loop iteration: `query_pos` and `ref_pos` are the number of
    // query and reference bases consumed by the ops already processed, so they index `query` and
    // `rseq` at the current op's start. `match_run` is the number of *equal* aligned bases seen
    // since the last thing written into `md`, and is reset to 0 every time something is written.
    let (mut query_pos, mut ref_pos, mut match_run) = (0usize, 0usize, 0i32);
    // `n_mm`: mismatching bases. `n_gap`: total gap length. Their sum is NM. Both are running
    // totals over the ops processed so far, and both deliberately exclude an edge deletion (see the
    // D branch), because such a deletion will not survive into the emitted CIGAR.
    let mut n_mm = 0i32;
    let mut n_gap = 0i32;
    // `n_cigar`: number of CIGAR operations. Captured once because the D branch needs to recognise
    // the last op by index.
    let n_cigar = cigar.len();
    for (op_idx, &packed_op) in cigar.iter().enumerate() {
        // Unpack this operation: `op` is the code (0=M, 1=I, 2=D), `len` its run length in bases.
        let op = packed_op & CIGAR_OP_MASK;
        let len = (packed_op >> 4) as usize;
        if op == CIGAR_OP_MATCH {
            // M covers both matches and mismatches, so we have to compare base by base. Note the
            // comparison is on nt4 codes, so an N in the query (code 4) never equals a reference
            // base and always scores as a mismatch in NM.
            for i in 0..len {
                if query[query_pos + i] != rseq[ref_pos + i] {
                    md.push_str(&match_run.to_string());
                    md.push(base_char(rseq[ref_pos + i]));
                    n_mm += 1;
                    match_run = 0;
                } else {
                    match_run += 1;
                }
            }
            query_pos += len;
            ref_pos += len;
        } else if op == CIGAR_OP_DEL {
            // Deletion. The `op_idx > 0 && op_idx < n_cigar - 1` guard skips a D that is the first
            // or last op (bwa.cpp:327): `mem_reg2aln` is about to squeeze exactly such a D out of
            // the CIGAR, so recording it in MD or NM would describe an alignment that will not be
            // emitted. `ref_pos` still advances in either case, keeping the reference cursor honest.
            if op_idx > 0 && op_idx < n_cigar - 1 {
                md.push_str(&match_run.to_string());
                md.push('^');
                for i in 0..len {
                    md.push(base_char(rseq[ref_pos + i]));
                }
                match_run = 0;
                n_gap += len as i32;
            }
            ref_pos += len;
        } else if op == CIGAR_OP_INS {
            // Insertion: consumes query only, and is invisible to MD by definition (MD describes
            // the reference). It does count toward NM.
            query_pos += len;
            n_gap += len as i32;
        }
    }
    // MD always ends with a run length, even when that run is empty ("0"), matching the final
    // `kputw(u, &str)` at bwa.cpp:336.
    md.push_str(&match_run.to_string());
    Some((score, cigar, n_mm + n_gap, md))
}

/// Turn an alignment region into a finalized alignment (CIGAR, NM, MD, position). Port of the CIGAR
/// assembly in `mem_reg2aln` (`bwamem.cpp:1732`): band retry, leading/trailing-D squeeze, soft-clip
/// addition.
///
/// `query_codes` is the *whole* read in nt4 codes and pac orientation, indexed by the region's
/// `qb`/`qe`; `l_query` is its length. The C converts ASCII to nt4 here via `nst_nt4_table`
/// (`bwamem.cpp:1748`); we require callers to hand us codes already, which is why there is no
/// equivalent conversion loop.
///
/// Precondition: `reg.rb >= 0 && reg.re >= 0`. The C folds the unmapped case in via an early
/// return; we split it out into [`MemAln::unmapped`] so callers decide explicitly.
///
/// # Parameters
///
/// - `fm`: the FM index, passed straight through to [`gen_cigar2`] for `l_pac()` and `base()`.
/// - `bns`: the contig table. Used for `l_pac` (the forward/reverse split point), `depos` (packed
///   coordinate to genome position plus strand) and `pos2rid` / `contigs[..].offset` (genome
///   position to contig id and contig-relative position).
/// - `opt`: run parameters. Read here for `a` (match score), `w` (the `-w` band width), the four
///   gap penalties, and indirectly for MAPQ via `mem_approx_mapq_se`.
/// - `l_query`: full read length in bases, *not* the aligned span. Soft-clip lengths are computed
///   against it, so passing the span instead would silently drop the 3' clip.
/// - `query_codes`: the whole read in nt4 codes (0..=3 ACGT, 4 N) and pac orientation. Must have
///   length `l_query`; it is indexed with `reg.qb`/`reg.qe`, which are offsets into this buffer.
/// - `reg`: the candidate region to finalize. Read fields: `qb`/`qe`/`rb`/`re` (the end points),
///   `truesc` (local score, the retry loop's target), `w` (the band the region was extended with),
///   `secondary` (`< 0` means primary), `score`, `sub`, `csub`.
///
/// # Returns
///
/// A [`MemAln`] whose `cigar`, `nm`, `md`, `pos`, `rid`, `is_rev`, `mapq` and `flag` are final
/// except for the strand and pairing FLAG bits, which the SAM writer adds, and `xa`, which the
/// caller fills in from `mem_gen_alt`.
pub fn reg2aln(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    l_query: i32,
    query_codes: &[u8],
    reg: &MemAlnReg,
) -> MemAln {
    // -------------------------------------------------------------------------------------
    // 1. Choose a band width for the global re-alignment.
    // -------------------------------------------------------------------------------------
    // `qb`/`qe`: aligned span within the read. `rb`/`re`: reference interval, packed coordinates.
    let (qb, qe, rb, re) = (reg.qb, reg.qe, reg.rb, reg.re);
    // Two band estimates, one assuming the length difference is explained by deletions and one by
    // insertions; the band has to be wide enough for whichever is cheaper, hence the max
    // (bwamem.cpp:1752-1754).
    //
    // `bw_del`: band half-width in bases sufficient if the length difference is all deletions.
    let bw_del = infer_bw(
        qe - qb,
        (re - rb) as i32,
        reg.truesc,
        opt.a,
        opt.o_del,
        opt.e_del,
    );
    // `w2`: the working band half-width, widened by the retry loop below.
    let mut w2 = infer_bw(
        qe - qb,
        (re - rb) as i32,
        reg.truesc,
        opt.a,
        opt.o_ins,
        opt.e_ins,
    )
    .max(bw_del);
    // `if (w2 > opt->w) w2 = w2 < ar->w? w2 : ar->w;` (bwamem.cpp:1756). Read carefully: the
    // inferred band is only allowed to exceed the command-line `-w` if the region's *own* recorded
    // band `reg.w` (set during extension) also allows it. So `opt.w` is not a hard cap, it is the
    // threshold past which `reg.w` becomes the cap. A region that was extended with a wide band
    // keeps its licence to use one here.
    if w2 > opt.w {
        w2 = w2.min(reg.w);
    }

    if band_width_trace_enabled() {
        eprintln!(
            "* Band width: inferred={w2}, cmd_opt={}, alnreg={}",
            opt.w, reg.w
        );
    }

    // -------------------------------------------------------------------------------------
    // 2. Global-align between the fixed end points, widening the band and retrying if the
    //    result scores implausibly far below the local score that found those end points.
    // -------------------------------------------------------------------------------------
    /// The C's `++i < 3`: at most 3 calls to `gen_cigar2` when neither early exit fires.
    const MAX_BAND_ATTEMPTS: i32 = 3;
    /// bwa's `-(1<<30)` sentinel for "no previous score"; any real score beats it, so the
    /// `sc == last_sc` convergence test cannot fire on the first pass.
    const NO_PREVIOUS_SCORE: i32 = -(1 << 30);

    // Band-doubling retry loop, a transliteration of the C `do { ... } while (++i < 3 && score <
    // ar->truesc - opt->a)` at bwamem.cpp:1758-1766. The global alignment between fixed end points
    // can score worse than the local alignment that found those end points, which is a symptom of
    // too narrow a band; widening and retrying recovers it. At most 3 more attempts.
    //
    // The `loop` shape is required to reproduce the C exactly: a do-while runs its body before
    // testing, and there are two *early* exits (bwamem.cpp:1763) that fire before the tail test.
    // Rewriting this as a `while` would change which CIGAR is kept.
    // `attempt`: number of band doublings performed so far, 0 on entry. `last_sc`: the global score
    // returned by the previous iteration, so the loop can detect that widening the band changed
    // nothing. Invariant at the top of each iteration: `w2` is the band about to be tried and is
    // strictly wider than the one that produced `last_sc`.
    let mut attempt = 0;
    let mut last_sc = NO_PREVIOUS_SCORE;
    // Whichever attempt the loop settled on. `_score` is the global score, discarded because AS:i
    // reports the region's local score instead; `cigar` is still M/I/D-only (clips added in step 4)
    // and `mut` because step 4 edits it in place.
    let (_score, mut cigar, nm, md) = loop {
        // Hard ceiling of 4x the command-line band, applied at the top of every iteration so that
        // the `w2 <<= 1` below can never overshoot it.
        w2 = w2.min(opt.w << 2);
        // This attempt's result: `sc` the global alignment score at band `w2`, and the CIGAR/NM/MD
        // that go with it. Kept only if one of the exits below breaks on this iteration.
        let (sc, cigar_attempt, nm_attempt, md_attempt) = gen_cigar2(
            fm,
            opt,
            w2,
            qe - qb,
            &query_codes[qb as usize..qe as usize],
            rb,
            re,
        )
        // `expect`: the C `assert(a.cigar != NULL)` at bwamem.cpp:1767. A region reaching here has
        // `qe > qb` and a non-straddling `[rb, re)`, so none of `gen_cigar2`'s reject conditions
        // can hold.
        .expect("gen_cigar2");
        if band_width_trace_enabled() {
            eprintln!(
                "* Final alignment: w2={w2}, global_sc={sc}, local_sc={}",
                reg.truesc
            );
        }
        // Early exits (bwamem.cpp:1763). `sc == last_sc` means doubling the band bought nothing, so
        // the score gap is genuine rather than a banding artifact; `w2 == opt.w << 2` means we are
        // already at the ceiling and cannot widen further.
        if sc == last_sc || w2 == opt.w << 2 {
            break (sc, cigar_attempt, nm_attempt, md_attempt);
        }
        last_sc = sc;
        w2 <<= 1;
        attempt += 1;
        // The do-while tail condition, negated. `sc < reg.truesc - opt.a` is the "still meaningfully
        // short of the local score" test, with one match score of tolerance: a global alignment
        // legitimately loses up to about that much to the forced end points, so insisting on exact
        // equality would retry forever on healthy alignments.
        //
        // Note `attempt` is incremented *before* the test, mirroring the C's prefix `++i < 3`,
        // which caps the loop at 3 total calls to `gen_cigar2` when neither early exit fires.
        if !(attempt < MAX_BAND_ATTEMPTS && sc < reg.truesc - opt.a) {
            break (sc, cigar_attempt, nm_attempt, md_attempt);
        }
    };

    // -------------------------------------------------------------------------------------
    // 3. Convert the packed reference coordinate into a strand plus a genome position.
    // -------------------------------------------------------------------------------------

    // Map the pac coordinate back to a forward-genome coordinate plus a strand flag. The probe
    // point differs by strand (bwamem.cpp:1770): on the forward strand the alignment's leftmost
    // forward-genome base is `rb`, but on the reverse strand the pac interval runs backwards
    // relative to the forward genome, so `re - 1` is the base that maps to the leftmost position.
    // Probing with `rb` on a reverse hit would yield the rightmost base and an off-by-length POS.
    let probe_pos = if rb < bns.l_pac { rb } else { re - 1 };
    // `pos`: 0-based genome position; SAM POS is `pos + 1` (and contig-relative, adjusted below).
    let (mut pos, is_rev) = bns.depos(probe_pos);

    // -------------------------------------------------------------------------------------
    // 4. Clean up the CIGAR for SAM: drop an edge deletion, add soft clips.
    // -------------------------------------------------------------------------------------
    // Squeeze a leading or trailing deletion (bwamem.cpp:1772). A D at either edge is meaningless
    // in SAM: it asserts reference bases outside the aligned span. A leading D is absorbed by
    // advancing POS past it; a trailing D is simply dropped. Note this is an `else if`, so a CIGAR
    // that is D-flanked on both ends keeps its trailing D. That is bwa's behavior, not an
    // oversight to fix: `gen_cigar2` already excluded both edge Ds from NM/MD, so removing both
    // here would still be self-consistent, but the emitted CIGAR would differ from bwa-mem2's.
    if !cigar.is_empty() {
        if cigar[0] & CIGAR_OP_MASK == CIGAR_OP_DEL {
            pos += i64::from(cigar[0] >> 4);
            cigar.remove(0);
        } else if cigar[cigar.len() - 1] & CIGAR_OP_MASK == CIGAR_OP_DEL {
            cigar.pop();
        }
    }

    // Soft-clips for the unaligned read ends (bwamem.cpp:1783). `[qb, qe)` is the aligned span
    // within the read, so anything outside it is clipped.
    if qb != 0 || qe != l_query {
        // The swap on `is_rev` is the crux: `qb`/`qe` are offsets into the read as it will be
        // *emitted* (pac orientation), but CIGAR is written 5'-to-3' along the *reference*. For a
        // reverse-strand hit the two run opposite ways, so the clip that is 5' in reference order
        // is the one at the read's 3' end, `l_query - qe`.
        let clip5 = if is_rev { l_query - qe } else { qb };
        let clip3 = if is_rev { qb } else { l_query - qe };
        if clip5 > 0 {
            cigar.insert(0, ((clip5 as u32) << 4) | CIGAR_OP_SOFT_CLIP);
        }
        if clip3 > 0 {
            cigar.push(((clip3 as u32) << 4) | CIGAR_OP_SOFT_CLIP);
        }
    }

    // -------------------------------------------------------------------------------------
    // 5. Assemble the output record.
    // -------------------------------------------------------------------------------------
    // `pos` is still a whole-genome offset; SAM POS is contig-relative, so subtract the contig's
    // start (bwamem.cpp:1798-1800). The C asserts `a.rid == ar->rid` here and indexes `bns->anns`
    // unconditionally; we tolerate `rid < 0` with a zero offset instead of trusting the assert.
    // `rid`: index into `bns.contigs` of the contig containing `pos`, or negative if none.
    let rid = bns.pos2rid(pos);
    // `contig_offset`: that contig's start in whole-genome coordinates, the amount to subtract to
    // make `pos` contig-relative. Zero when `rid < 0`, which leaves `pos` as the raw genome offset
    // rather than panicking on the index.
    let contig_offset = if rid >= 0 {
        bns.contigs[rid as usize].offset
    } else {
        0
    };
    MemAln {
        rid,
        pos: pos - contig_offset,
        is_rev,
        // Only primaries get a real MAPQ; secondaries are pinned to 0 (bwamem.cpp:1750). Computing
        // MAPQ for a secondary would be meaningless anyway, since its `sub` was never populated.
        mapq: if reg.secondary < 0 {
            mem_approx_mapq_se(opt, reg)
        } else {
            0
        },
        // 0x100 = SAM SECONDARY. The strand bit (0x10) and all pairing bits are added later by the
        // SAM writer, which is why `flag` is not final here.
        flag: if reg.secondary >= 0 {
            SAM_FLAG_SECONDARY
        } else {
            0
        },
        cigar,
        nm,
        md,
        // `ar->score`, the local score, deliberately not the global `score` the loop computed
        // (bwamem.cpp:1801). AS:i therefore reports the extension score, which can exceed what the
        // emitted CIGAR would rescore to.
        score: reg.score,
        // XS:i is the better of the two suboptimal estimates: `sub` from overlapping regions
        // (`mem_mark_primary_se`) and `csub` from within the same chain.
        sub: reg.sub.max(reg.csub),
        // Filled in by the caller from `mem_gen_alt`; `reg2aln` never produces XA itself, because
        // XA generation calls back into `reg2aln` and would recurse.
        xa: None,
    }
}

/// Format a CIGAR (`len<<4|op`) as a SAM string, or `*` when empty.
///
/// # Parameters
///
/// - `cigar`: packed operations in reference order, each `len << 4 | op` with `op` in 0..=4. Op
///   codes above 4 would panic on the `OPS` index; bwa never produces them. An empty slice is the
///   unmapped case.
///
/// # Returns
///
/// The SAM CIGAR field verbatim, e.g. `12S88M2D30M`, or the literal `*`.
#[must_use]
pub fn cigar_string(cigar: &[u32]) -> String {
    if cigar.is_empty() {
        return "*".to_string();
    }
    // Index-into-string decoding, exactly bwa's `"MIDSH"[c]`. Op codes above 4 (N, P, =, X) never
    // occur in bwa output and would panic here rather than silently mis-encode.
    // Op code to letter, indexed by the code itself: M=0, I=1, D=2, S=3, H=4. Reordering these
    // characters would mislabel every CIGAR in the output.
    const OPS: [char; 5] = ['M', 'I', 'D', 'S', 'H'];
    // The SAM CIGAR field being built, complete for the ops emitted so far.
    let mut s = String::new();
    for &packed_op in cigar {
        s.push_str(&(packed_op >> 4).to_string());
        s.push(OPS[(packed_op & CIGAR_OP_MASK) as usize]);
    }
    s
}

/// `add_cigar`: like [`cigar_string`], but rewrites clip ops for the record being emitted. A
/// supplementary record (`which != 0`) hard-clips what the primary soft-clips, so that the read's
/// bases are stored exactly once; `-Y` (`flags::SOFTCLIP`) and ALT hits keep soft clips.
///
/// Port of `add_cigar` (`bwamem.cpp:1579`). The clip rewrite is suppressed for ALT hits because an
/// ALT record is a re-report of bases that already appear on the primary assembly, so hard-clipping
/// it would lose them.
///
/// # Parameters
///
/// - `cigar`: packed operations, as for [`cigar_string`]. Not modified; the clip rewrite happens on
///   the way out, per operation.
/// - `which`: index of this record within the read's alignment list. `0` is the primary and keeps
///   soft clips; any non-zero value marks a supplementary record and turns clips into hard clips.
/// - `is_alt`: the region's ALT-contig flag, supplied by the caller from the contig table. `true`
///   suppresses the rewrite entirely.
/// - `softclip`: `opt.flag & MEM_F_SOFTCLIP` reduced to a bool, i.e. the `-Y` command-line switch.
///   `true` suppresses the rewrite, keeping soft clips on every record.
///
/// # Returns
///
/// The SAM CIGAR field for this particular record, or `*` when `cigar` is empty.
#[must_use]
pub fn cigar_string_which(cigar: &[u32], which: usize, is_alt: bool, softclip: bool) -> String {
    if cigar.is_empty() {
        return "*".to_string();
    }
    // Op code to letter, indexed by the code: M=0, I=1, D=2, S=3, H=4. Same table as in
    // [`cigar_string`]; kept local rather than shared to stay a direct transliteration of bwa.
    const OPS: [char; 5] = ['M', 'I', 'D', 'S', 'H'];
    // `op` is kept as `usize` here (it indexes `OPS` directly), so the op-code constants are cast
    // rather than used raw. Same values as `CIGAR_OP_SOFT_CLIP` (3) and `CIGAR_OP_HARD_CLIP` (4).
    const SOFT_CLIP: usize = CIGAR_OP_SOFT_CLIP as usize;
    const HARD_CLIP: usize = CIGAR_OP_HARD_CLIP as usize;
    // The SAM CIGAR field being built, with clip ops already rewritten for this `which`.
    let mut s = String::new();
    for &packed_op in cigar {
        // This operation's code, possibly rewritten below from S to H or back.
        let mut op = (packed_op & CIGAR_OP_MASK) as usize;
        // Both 3 (S) and 4 (H) are matched on input, not just 3: the same alignment can be emitted
        // more than once with different `which`, so a clip already rewritten to H must be able to
        // travel back to S. The rewrite is idempotent and reversible for that reason.
        if !softclip && !is_alt && (op == SOFT_CLIP || op == HARD_CLIP) {
            op = if which != 0 { HARD_CLIP } else { SOFT_CLIP };
        }
        s.push_str(&(packed_op >> 4).to_string());
        s.push(OPS[op]);
    }
    s
}

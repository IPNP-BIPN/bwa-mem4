//! Paired-end pairing, insert-size estimation and SAM emission, mirroring bwa-mem2's
//! `mem_pestat` / `mem_pair` / `mem_sam_pe` / `mem_aln2sam` (`reference/bwa-mem2/src/bwamem_pair.cpp`
//! and `bwamem.cpp`).
//!
//! Mate rescue (`mem_matesw`) uses `ksw_align2`; on concordant pairs all orientations are skipped
//! so it performs no Smith-Waterman and leaves the region set untouched.
//!
//! # What this file owns
//!
//! Everything that happens to a read *pair* after each end has been aligned on its own. The
//! single-end pipeline (seeding, chaining, extension, dedup, primary marking) lives in the sibling
//! modules and hands this file two score-sorted `Vec<MemAlnReg>`, one per end. From there:
//!
//! 1. **Insert-size estimation** over a whole batch ([`mem_pestat`]), producing four [`PeStat`]s.
//! 2. **Mate rescue** ([`mem_matesw`], or its batched split [`batch_mate_rescue`]), which can ADD
//!    regions to either end.
//! 3. **Pairing** ([`mem_pair`]), which picks the best proper pair and its MAPQ inputs.
//! 4. **SAM emission** ([`mem_sam_pe`] -> [`mem_reg2sam`] -> [`mem_aln2sam`]), byte for byte.
//!
//! # Reading order
//!
//! [`mem_infer_dir`] (the FF/FR/RF/RR encoding everything else indexes by), [`PeStat`],
//! [`mem_pestat`], [`cal_sub`], then the rescue group ([`bns_fetch_seq`], [`mem_matesw`], and only
//! afterwards its batched twin [`matesw_collect`] / [`rescue_jobs`] / [`matesw_apply`] /
//! [`batch_mate_rescue`]), then [`mem_pair`] with [`id_shift_c`] and [`raw_mapq`], and finally the
//! emitters bottom-up: [`add_cigar`], [`get_rlen`], [`mem_aln2sam`], [`mem_reg2sam`],
//! [`mem_sam_pe`]. `mem_sam_pe` is the entry point and reads best last.
//!
//! # Glossary of names kept from the C
//!
//! These are deliberately NOT renamed: the point of each is that it is the same identifier as in
//! `bwamem_pair.cpp` / `bwamem.cpp`, so a reader can diff the two sources line by line.
//!
//! | name | plain language |
//! |------|----------------|
//! | `pes` | the four per-orientation insert-size stats, indexed FF=0, FR=1, RF=2, RR=3 |
//! | `avg`, `std` | mean and (population) std.dev of the insert size for one orientation |
//! | `low`, `high` | the proper-pair insert window for one orientation, in reference bases |
//! | `l_pac` | packed reference length; coordinates live in the doubled space `[0, 2*l_pac)` |
//! | `rb`, `re` | region's reference begin/end, in that doubled space (half-open) |
//! | `qb`, `qe` | region's query (read) begin/end, half-open, in sequencing orientation |
//! | `qle`, `tle`, `gtle` | ksw's query/target extension lengths (local, and to the global end) |
//! | `gscore`, `score2` | ksw's end-to-end score, and its second-best (tandem-repeat) score |
//! | `h0`, `h1` | the two chosen alignments of a pair, read 1 and read 2 (also ksw's seed score) |
//! | `z` | index into each end's region vector of the chosen paired placement, `z[0]`/`z[1]` |
//! | `o` | one candidate rescue orientation's job (see [`Orient`]) |
//! | `subo` | the sub-optimal score the paired MAPQ is measured against |
//! | `n_sub` | how many runner-up pairs are effectively tied with the sub-optimal |
//! | `xs` | the `XS:i` tag's value, i.e. the reported sub-optimal alignment score |
//! | `kept`, `frac_rep` | region bookkeeping carried through from the single-end stage |
//! | `eh`, `mj` | ksw DP row state and its last non-zero column (not used in this file) |
//! | `csub` | second-best score at the SAME locus; caps MAPQ on internally repetitive loci |

//! # Rust mechanics used in this file
//!
//! | Construct | What it means, and why it is here |
//! |-----------|-----------------------------------|
//! | `pub struct PairRescueData<'a>` | the batched rescue path's input: two read slices and two MUTABLE borrows of the mates' region vectors, bundled in one value. The `'a` lifetime ties the bundle to the batch buffers it points into, so it cannot outlive them; the `&'a mut Vec<..>` fields mean rescue appends to the caller's lists in place, as the C does, with the compiler guaranteeing no other path holds a view of those lists at the same time. |
//! | `[Option<Orient>; 4]` with `.flatten()` | four orientation slots (FF, FR, RF, RR), each either used or not. `flatten()` on the array iterator yields the occupied ones in ASCENDING slot order, which is exactly the C's loop order, so the rescue calls happen in the same sequence and the regions are appended in the same order. |
//! | `#[derive(Debug, Clone, Copy)]` on `PeStat` | four numbers; `Copy` makes passing one around free and non-moving. |
//! | `impl Default for PeStat` (hand-written) | "no statistics yet" is not all-zeros in bwa's terms, so the default is written out rather than derived. |
//! | `samples.sort_unstable()` | legitimate here, and only here, because the samples are plain integers: equal elements are INDISTINGUISHABLE, so no permutation of them is observable. Contrast the dedup sort in `primary.rs`, where equal keys carry different payloads and the permutation is output-visible. |
//! | `f64` throughout the pairing scores | the C computes these in `double`. Narrowing to `f32` would round differently and flip pairing decisions on a small fraction of reads; the width is part of the contract, not a performance knob. |
//! | `std::io::Write` on a generic writer | SAM records are formatted into any sink implementing `Write`, so the same code serves a file, a pipe and an in-memory buffer in tests. The formatting is byte-exact, which is why the writer is buffered by the caller rather than here. |
//! | `wrapping_add` on the read id | the C adds an `int64_t` id that is allowed to wrap; asking for wrapping explicitly keeps debug and release builds identical. |

use std::io::{self, Write};

use bwa_core::MemOpt;
use bwa_extend::{ksw_align2, KswAlignResult};
use bwa_index::{BntSeq, FmIndex};
use bwa_neon::{batched_mate_rescue, batched_mate_rescue_lazy, KswJob, SCORE2_PENDING};

use crate::alt::mem_gen_alt;
use crate::cigar::{reg2aln, MemAln};
use crate::meth::MethEmit;
use crate::primary::{
    hash_64, insert_into_settled, mem_approx_mapq_se, mem_mark_primary_se, mem_sort_dedup_patch,
    same_region,
};
use crate::MemAlnReg;

extern "C" {
    /// System libm complementary error function, for bit-identical pairing scores.
    fn erfc(x: f64) -> f64;
}

// The six insert-size tuning constants of `mem_pestat`, verbatim from the `#define`s at
// `bwamem_pair.cpp:49-54`. They are not options: bwa hard-codes them, so they are `const` here.

/// A pair only feeds the insert-size histogram if its second-best score is at most 80% of its best
/// on *both* ends, i.e. both ends map close to uniquely (`MIN_RATIO`, `bwamem_pair.cpp:49`).
const MIN_RATIO: f64 = 0.8;
/// An orientation with fewer than 10 candidate unique pairs is declared `failed` and never used for
/// pairing or rescue (`MIN_DIR_CNT`, `bwamem_pair.cpp:50`).
const MIN_DIR_CNT: usize = 10;
/// An orientation is also failed if it holds under 5% of the most-populated orientation's count:
/// this is what kills the three spurious orientations on a normal FR library
/// (`MIN_DIR_RATIO`, `bwamem_pair.cpp:51`).
const MIN_DIR_RATIO: f64 = 0.05;
/// Tukey-style fence multiplier on the interquartile range used to pick the window over which the
/// mean and std.dev are computed (`OUTLIER_BOUND`, `bwamem_pair.cpp:52`).
const OUTLIER_BOUND: f64 = 2.0;
/// The wider IQR multiplier defining the final proper-pair window `[low, high]`
/// (`MAPPING_BOUND`, `bwamem_pair.cpp:53`).
const MAPPING_BOUND: f64 = 3.0;
/// The proper-pair window is additionally widened to at least +/- 4 std.dev around the mean
/// (`MAX_STDDEV`, `bwamem_pair.cpp:54`).
const MAX_STDDEV: f64 = 4.0;
/// C's `M_SQRT1_2` (1/sqrt(2)), used to turn a z-score into an `erfc` argument in `mem_pair`.
const M_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// The remaining literals below are bwa's, named here only so the arithmetic reads. Each keeps its
// exact C value and its C type, so substituting the name changes nothing.

/// bwa's `(l_ms * opt->a < 250? KSW_XBYTE : 0)` (`bwamem_pair.cpp:216`): 250 is a hair under the 255
/// an unsigned byte holds, so the test is "a perfect match of the whole mate provably cannot
/// saturate the byte kernel".
const KSW_XBYTE_SCORE_LIMIT: i32 = 250;
/// SIMD lanes of the ksw u8 kernel (one score per byte in a 128-bit vector). `usize`, matching
/// `ksw_align2`'s `lanes` parameter.
const KSW_LANES_U8: usize = 16;
/// SIMD lanes of the ksw i16 kernel, used when a score could exceed [`KSW_XBYTE_SCORE_LIMIT`].
const KSW_LANES_I16: usize = 8;
/// bwa's `raw_mapq` coefficient (`bwamem_pair.cpp:350`): `10 * log10(4)`, i.e. one extra matched
/// base is a 4-fold likelihood ratio, which is 6.02 phred units.
const MAPQ_PER_MATCHED_BASE: f64 = 6.02;
/// bwa's `0.721` (`bwamem_pair.cpp:317`), annotated there as `1/log(4)`: converts the insert-size
/// log-likelihood from nats into log-base-4 units, commensurate with a per-base score.
const INV_LOG4: f64 = 0.721;
/// bwa's `4.343` in `mem_sam_pe`: `10/ln(10)`, turning a natural log into decibans, so the `n_sub`
/// discount is `10*log10(n_sub+1)` phred units.
const DECIBANS_PER_NAT: f64 = 4.343;
/// The SAM MAPQ ceiling bwa clamps to (`q_pe = q_pe < 0? 0 : q_pe < 60? q_pe : 60`).
const MAX_MAPQ: i32 = 60;
/// bwa's `q_se + 40` cap: pairing evidence may never add more than 40 phred units on top of what
/// the single-end alignment alone justified.
const MAX_PAIRING_MAPQ_BOOST: i32 = 40;

/// Per-orientation insert-size statistics (`mem_pestat_t`), indexed by the direction code that
/// [`mem_infer_dir`] returns: 0 = FF, 1 = FR, 2 = RF, 3 = RR.
///
/// Produced once per read batch by [`mem_pestat`] and then read (never written) by mate rescue and
/// pairing. Because it is estimated per batch, a single difference in these five numbers moves the
/// rescue and pairing decisions of every pair in the batch, not of one read.
#[derive(Debug, Clone, Copy)]
pub struct PeStat {
    /// Smallest insert (in reference bases) still considered a proper pair. Clamped to >= 1 by
    /// `mem_pestat`, so it is never zero or negative even for tight libraries.
    pub low: i32,
    /// Largest insert still considered a proper pair. No upper clamp in the C.
    pub high: i32,
    /// True when this orientation has too few observations to trust. `mem_matesw` seeds `skip[r]`
    /// from it and `mem_pair` skips the direction entirely, so a failed orientation contributes
    /// neither rescue nor pairing.
    pub failed: bool,
    /// Mean insert size over the inliers, f64 (bwa's `double`; f32 here would change `erfc`).
    pub avg: f64,
    /// Population (not sample) std.dev over the same inliers: `mem_pestat` divides by `x`, not
    /// `x - 1`. Used as the denominator of the z-score in `mem_pair`, so it must not be zero.
    pub std: f64,
}

/// bwa starts from `memset(pes, 0, ...)`, i.e. `failed == 0`, and then sets `failed = 1` on every
/// under-populated orientation. We start from `failed: true` instead and clear it on the populated
/// ones (`mem_pestat` assigns `r.failed = false` explicitly), which lands on the same four values
/// while making "never analysed" the safe default for any caller that builds a `PeStat` by hand.
impl Default for PeStat {
    fn default() -> Self {
        PeStat {
            low: 0,
            high: 0,
            failed: true,
            avg: 0.0,
            std: 0.0,
        }
    }
}

/// Infer the relative orientation and distance of two 5' coordinates. Port of `mem_infer_dir`
/// (`bwamem_pair.cpp:58-65`).
///
/// bwa stores every alignment start in a doubled "forward-reverse" coordinate space of width
/// `2 * l_pac`: positions `[0, l_pac)` are the forward strand of the concatenated reference and
/// `[l_pac, 2*l_pac)` are its reverse complement, with forward position `p` mirroring to
/// `2*l_pac - 1 - p`. So `b >= l_pac` *is* the strand bit, which is why the C tests it directly
/// instead of carrying a separate flag.
///
/// PARAMETERS: `l_pac` is the packed reference length in bases (`bns.l_pac`). `b1`, `b2` are the
/// two regions' `rb` (leftmost reference coordinate in that doubled space), `b1` from read 1 and
/// `b2` from read 2; both must be in `[0, 2*l_pac)`.
///
/// RETURNS `(dir, dist)`, where `dir` indexes `pes[4]` as FF=0, FR=1, RF=2, RR=3 and `dist` is the
/// unsigned separation of the two 5' ends in reference bases.
#[must_use]
fn mem_infer_dir(l_pac: i64, b1: i64, b2: i64) -> (usize, i64) {
    // `r1`/`r2` are the C's names: 1 when that coordinate sits in the reverse half, i.e. the strand.
    let r1 = (b1 >= l_pac) as i64;
    let r2 = (b2 >= l_pac) as i64;
    // p2: read-2 coordinate projected onto read-1's strand. When the strands already agree the
    // coordinates are directly comparable; otherwise b2 is mirrored through the midpoint of the
    // doubled space so that the subtraction below measures a real genomic distance.
    let p2 = if r1 == r2 { b2 } else { (l_pac << 1) - 1 - b2 };
    let dist = if p2 > b1 { p2 - b1 } else { b1 - p2 };
    // The C is one expression: `(r1 == r2? 0 : 1) ^ (p2 > b1? 0 : 3)`. Bit 1 of the result says
    // "the two reads are on opposite strands", and XOR-ing with 3 when read 2 sits *before* read 1
    // flips both bits, which is exactly swapping the roles of the two reads: FF <-> RR (0 <-> 3)
    // and FR <-> RF (1 <-> 2). That is why the direction code is order-independent in the sense
    // that mem_infer_dir(b2, b1) yields the mirrored orientation rather than garbage.
    let opposite_strands_bit = if r1 == r2 { 0 } else { 1 };
    let dir = (opposite_strands_bit ^ if p2 > b1 { 0 } else { 3 }) as usize;
    (dir, dist)
}

/// Fetch reference `[rb, re)` (in 2*l_pac space) clamped to the contig containing `mid`, returning
/// the clamped bounds, the contig id, and the nt4 sequence (`.0123`, forward++reverse-complement).
/// Port of `bns_fetch_seq` (`bntseq.cpp:453-480`).
///
/// WHY the clamp: the mate-rescue window is built purely from insert-size arithmetic, so it can
/// easily run off the end of the contig that the anchor sits on and into the next one. Aligning
/// across a contig boundary would produce a hit spanning two chromosomes, so the C narrows the
/// window to the single contig containing the window's midpoint and then makes the caller check
/// `a->rid == rid`, discarding the whole attempt if the anchor lives on a different contig.
///
/// PARAMETERS: `fm` is the FM index, used only through `fm.base(p)` to unpack one 2-bit reference
/// base at doubled-space position `p`. `bns` supplies the contig annotation table (`bns.contigs`,
/// `bns.l_pac`) that the clamp is computed from; both are read-only and shared across threads.
/// `rb`/`re` are the requested half-open window in the doubled coordinate space and may
/// arrive in either order (the C swaps them with a three-way XOR; we swap with a tuple). `mid`
/// selects the contig and, through `bns.depos`, the strand; callers pass `(rb + re) >> 1`. The C
/// asserts `*beg <= mid && mid < *end`, which that midpoint satisfies whenever `rb < re`.
///
/// RETURNS the *clamped* `(rb, re)` (the caller needs them, since every returned coordinate is
/// relative to the clamped `rb`), the contig id `rid` (-1 if `mid` falls outside every contig), and
/// the fetched bases. `rid < 0` cannot match a real `a.rid`, so it fails the caller's guard.
pub(crate) fn bns_fetch_seq(
    fm: &FmIndex,
    bns: &BntSeq,
    rb: i64,
    mid: i64,
    re: i64,
) -> (i64, i64, i32, Vec<u8>) {
    let (mut rb, mut re) = if re < rb { (re, rb) } else { (rb, re) };
    // `depos` folds `mid` back onto the forward strand and reports whether it came from the reverse
    // half; `pos2rid` then binary-searches the contig offsets in that forward space.
    let (fwd_mid, is_rev) = bns.depos(mid);
    let rid = bns.pos2rid(fwd_mid);
    // The C indexes `bns->anns[*rid]` unconditionally and would read out of bounds on rid < 0; we
    // fall back to the whole doubled space, which imposes no clamp. That case is unreachable for
    // callers that pass a midpoint inside the reference, and it cannot change output because the
    // caller rejects `rid != a.rid` anyway.
    let (mut far_beg, mut far_end) = if rid >= 0 {
        let contig = &bns.contigs[rid as usize];
        (contig.offset, contig.offset + i64::from(contig.len))
    } else {
        (0, bns.l_pac << 1)
    };
    // The contig bounds came out of the forward-strand annotation table, but `rb`/`re` live on the
    // reverse half. Mirror the interval (not each endpoint independently, hence the temporary and
    // the swap) so the clamp below compares like with like.
    if is_rev {
        let old_far_beg = far_beg;
        far_beg = (bns.l_pac << 1) - far_end;
        far_end = (bns.l_pac << 1) - old_far_beg;
    }
    rb = rb.max(far_beg);
    re = re.min(far_end);
    // `bns_get_seq` unpacks 2-bit bases; `fm.base` does the same per position, including the
    // reverse-complement fold for `p >= l_pac`. The C asserts the returned length equals `re - rb`,
    // which holds here by construction.
    let seq: Vec<u8> = fm.bases(rb, re);
    (rb, re, rid, seq)
}

/// [`bns_fetch_seq`] appending the window to `out` instead of allocating it.
pub(crate) fn bns_fetch_seq_into(
    fm: &FmIndex,
    bns: &BntSeq,
    rb: i64,
    mid: i64,
    re: i64,
    out: &mut Vec<u8>,
) -> (i64, i64, i32) {
    let (mut rb, mut re) = if re < rb { (re, rb) } else { (rb, re) };
    let (fwd_mid, is_rev) = bns.depos(mid);
    let rid = bns.pos2rid(fwd_mid);
    let (mut far_beg, mut far_end) = if rid >= 0 {
        let contig = &bns.contigs[rid as usize];
        (contig.offset, contig.offset + i64::from(contig.len))
    } else {
        (0, bns.l_pac << 1)
    };
    if is_rev {
        let old_far_beg = far_beg;
        far_beg = (bns.l_pac << 1) - far_end;
        far_end = (bns.l_pac << 1) - old_far_beg;
    }
    rb = rb.max(far_beg);
    re = re.min(far_end);
    fm.bases_append(rb, re, out);
    (rb, re, rid)
}

/// Smith-Waterman mate rescue: given an anchor region `a` for one read, try to align the mate `ms`
/// (nt4) in each insert-consistent orientation not already satisfied, appending any hit to `ma`.
/// Port of `mem_matesw` (`bwamem_pair.cpp:150-283`, the `#if !MATE_SORT` branches; bwa-mem2 ships
/// with `MATE_SORT` off and the sorted variant would not be byte-identical to bwa).
///
/// WHY: seeding can miss a mate entirely (repetitive, low complexity, too many mismatches for a
/// seed to survive). If the other end is confidently placed, the insert-size distribution says
/// within a few hundred bases where the mate must be, and a full local SW over that small window
/// finds it. This is where a large fraction of PE runtime goes on real data.
///
/// PARAMETERS: `fm`/`bns` are the reference (window fetching and `l_pac`). `opt` supplies the
/// scoring matrix and gap penalties handed to ksw, plus `min_seed_len` (both the score floor
/// `min_seed_len * a` given to ksw and, separately, the acceptance threshold and the minimum usable
/// window width). `pes` is the batch's four insert-size stats, read for `failed`/`low`/`high` only.
/// `a` is the anchor, a region of the *other* read, used only for `a.rb` (window
/// centre), `a.rid` (contig guard) and `a.is_alt` (copied to the new region). `ms` is this read's
/// nt4 sequence in sequencing orientation. `ma` is this read's region vector, read to decide which
/// orientations are already covered and mutated in place with any accepted hit.
///
/// RETURNS the number of orientations on which SW actually ran (bwa sums this into `mem_sam_pe`'s
/// `n`, which is its return value and otherwise unused). It is *not* the number of hits added.
///
/// INVARIANT: `ma` is kept sorted by score descending throughout, both by the manual insertion
/// below and by `mem_sort_dedup_patch`. Later anchors' skip test therefore sees earlier inserts,
/// which is what forces the round-by-round structure of [`batch_mate_rescue`].
#[allow(clippy::too_many_arguments)]
fn mem_matesw(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &MemAlnReg,
    ms: &[u8],
    ma: &mut Vec<MemAlnReg>,
    // `--meth`: the mate's chemistry (read 1 = C->T); unused otherwise.
    mate_ot: bool,
) -> i32 {
    // `l_pac`: packed reference length, so the doubled coordinate space is [0, 2*l_pac).
    // `l_ms`: the mate's length in bases, the length of every query built below and the amount by
    // which the reference window is extended to allow for the mate's own span.
    let l_pac = bns.l_pac;
    let l_ms = ms.len() as i64;
    // ---- step 1: decide which of the four orientations still need a rescue attempt ----
    // `skip[r]` means "do not run SW for orientation r". Two reasons to skip: the orientation was
    // never characterised for this library (`failed`), or `ma` already holds a region that pairs
    // with the anchor at a plausible insert in that orientation, so there is nothing to rescue.
    let mut skip = [0i32; 4];
    for (r, skip_r) in skip.iter_mut().enumerate() {
        *skip_r = i32::from(pes[r].failed);
    }
    for existing in ma.iter() {
        let (r, dist) = mem_infer_dir(l_pac, a.rb, existing.rb);
        if dist >= i64::from(pes[r].low) && dist <= i64::from(pes[r].high) {
            skip[r] = 1;
        }
    }
    if dump_pestat() {
        eprintln!(
            "[MATESW] rb={} ma_n={} skip={}{}{}{}",
            a.rb,
            ma.len(),
            skip[0],
            skip[1],
            skip[2],
            skip[3]
        );
    }
    if skip.iter().sum::<i32>() == 4 {
        return 0; // a consistent pair already exists; no SW needed
    }

    // ---- step 2: for each surviving orientation, build the query and the reference window ----
    // `n` counts orientations that actually ran SW (this function's return value), not hits.
    let mut n = 0;
    for r in 0..4 {
        if skip[r] != 0 {
            continue;
        }
        // Decode the two bits of the orientation code r (bit 1 = anchor's strand, bit 0 = mate's
        // strand, per mem_infer_dir). `is_rev`: the two strands differ, so the mate must be
        // reverse-complemented before aligning it to the forward-space window. `is_larger`: the
        // anchor is on the forward strand (r>>1 == 0), so the mate lies at *larger* coordinates,
        // i.e. downstream of the anchor. bwa writes these as `r>>1 != (r&1)` and `!(r>>1)`
        // (`bwamem_pair.cpp:195-196`).
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;
        // `seq`: the SW query for this orientation, i.e. the mate laid out on the same strand as the
        // forward-space window fetched below. Owned (not borrowed from `ms`) because the reverse
        // orientations need a new buffer.
        let seq: Vec<u8> = if is_rev {
            // reverse complement of ms. Note nt4 code 4 (N, and anything above) maps to itself
            // rather than to 3 - 4; the C guards with `ms[i] < 4? 3 - ms[i] : 4`.
            ms.iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            ms.to_vec()
        };
        // The search window is the anchor's 5' coordinate offset by the proper-pair insert range,
        // then extended by the mate's length on whichever side the mate's own start can slide.
        // Downstream (`is_larger`) the mate starts in `[rb+low, rb+high]`; upstream it starts in
        // `[rb-high, rb-low]`. Which end gets the extra `l_ms` depends on whether the mate was
        // reverse-complemented, because that flips which end of `seq` is its 5' end
        // (`bwamem_pair.cpp:203-210`).
        //
        // `lo`/`hi`: this orientation's proper-pair insert window in reference bases, widened to i64
        // so the window arithmetic below cannot overflow near the ends of the doubled space.
        // `rb0`/`re0`: the resulting unclamped search window, half-open, doubled space.
        let (lo, hi) = (i64::from(pes[r].low), i64::from(pes[r].high));
        let (rb0, re0) = if !is_rev {
            let rb = if is_larger { a.rb + lo } else { a.rb - hi };
            let re = (if is_larger { a.rb + hi } else { a.rb - lo }) + l_ms;
            (rb, re)
        } else {
            let rb = (if is_larger { a.rb + lo } else { a.rb - hi }) - l_ms;
            let re = if is_larger { a.rb + hi } else { a.rb - lo };
            (rb, re)
        };
        // Clamp to the doubled coordinate space. `rb0 >= re0` after clamping means the window fell
        // entirely off the reference, so no SW runs and `n` is not incremented for this r.
        let rb0 = rb0.max(0);
        let re0 = re0.min(l_pac << 1);
        if rb0 < re0 {
            // `rb`/`re`: the window after `bns_fetch_seq` clamped it to one contig (so `re - rb` can
            // be much smaller than `re0 - rb0`). `rid`: that contig's id, -1 off the reference.
            // `refseq`: its nt4 bases, `refseq[i]` being doubled-space position `rb + i`.
            let (rb, re, rid, refseq) = bns_fetch_seq(fm, bns, rb0, (rb0 + re0) >> 1, re0);
            if dump_pestat() {
                eprintln!(
                    "[MSW_R] anchor={} r={r} rb={rb} re={re} rid={rid} a_rid={}",
                    a.rb, a.rid
                );
            }
            // "no funny things happening" in the C: refuse to rescue onto a different contig than
            // the anchor, and refuse windows the contig clamp shrank below one seed length.
            // ---- step 3: local Smith-Waterman of the mate against that window ----
            if a.rid == rid && re - rb >= i64::from(opt.min_seed_len) {
                // The score floor handed to ksw: below `min_seed_len * a` the hit is not worth
                // reporting. bwa packs it into the low 16 bits of `xtra` alongside the KSW_X* bits.
                let minsc = opt.min_seed_len * opt.a;
                // bwa's `xtra`: KSW_XBYTE (the u8 kernel, 16 lanes) when the mate cannot overflow a
                // byte, else i16 at 8 lanes. The width is part of the result, not just the speed:
                // ksw pads the query profile out to a whole number of lanes.
                //
                // The C is `(l_ms * opt->a < 250? KSW_XBYTE : 0)` (`bwamem_pair.cpp:216`): 250 is a
                // hair under the 255 an unsigned byte holds, and `l_ms * a` is the largest score a
                // perfect match of the whole mate could reach, so the test is "this alignment
                // provably cannot saturate the byte kernel". The other two `xtra` bits, KSW_XSUBO
                // (report the second-best score, which becomes `csub`) and KSW_XSTART (report start
                // positions, which we need for `qb`/`tb`), are unconditional in the C and implicit
                // in `ksw_align2`'s Rust signature, which always returns all of them.
                let lanes = if l_ms as i32 * opt.a < KSW_XBYTE_SCORE_LIMIT {
                    KSW_LANES_U8
                } else {
                    KSW_LANES_I16
                };
                // `aln`: ksw's verdict for this orientation. `score`/`score2` are the best and
                // second-best local scores, `qb`/`qe` and `tb`/`te` the CLOSED start/end offsets
                // into `seq` and `refseq`, and `qb < 0` means "nothing scored above `minsc`".
                // `--meth`: a mate aligned reverse-complemented reads its conversion on the other
                // strand, so it takes the other chemistry's matrix.
                let mat = match &opt.meth {
                    Some(meth) => &meth.mats[usize::from(mate_ot ^ is_rev)],
                    None => &opt.mat,
                };
                let aln = ksw_align2(
                    &seq, &refseq, 5, mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, minsc,
                    opt.a, lanes,
                );
                if dump_pestat() {
                    eprintln!(
                        "[RESCUE] anchor={} r={r} score={} score2={} qb={} qe={} tb={} te={} te2={} tlen={}",
                        a.rb, aln.score, aln.score2, aln.qb, aln.qe, aln.tb, aln.te, aln.te2,
                        refseq.len()
                    );
                }
                // Acceptance: note the C compares the score against `opt->min_seed_len` (a *length*,
                // typically 19), not against the `minsc = min_seed_len * a` it passed to ksw. That
                // asymmetry looks like a bug in bwa but it is load-bearing for byte-identity, so it
                // is reproduced verbatim. `qb < 0` is ksw's "no alignment found" sentinel.
                if aln.score >= opt.min_seed_len && aln.qb >= 0 {
                    // Map ksw's coordinates, which are relative to `seq` (already reverse
                    // complemented) and to the fetched window starting at `rb`, back onto the
                    // read's sequencing orientation and the doubled reference space. When `is_rev`,
                    // both axes flip: a query interval [qb, qe] becomes [l_ms-1-qe, l_ms-1-qb] in
                    // closed form, which with half-open ends is the `l_ms - (qe+1)` / `l_ms - qb`
                    // pair below, and a reference interval mirrors through `2*l_pac`.
                    //
                    // `qb`/`qe`: the new region's half-open query interval in the READ's sequencing
                    // orientation. `b_rb`/`b_re`: its half-open interval in the doubled reference
                    // space. These four become the corresponding fields of the region built below.
                    let qb = if is_rev {
                        l_ms as i32 - (aln.qe + 1)
                    } else {
                        aln.qb
                    };
                    let qe = if is_rev {
                        l_ms as i32 - aln.qb
                    } else {
                        aln.qe + 1
                    };
                    let b_rb = if is_rev {
                        (l_pac << 1) - (rb + i64::from(aln.te) + 1)
                    } else {
                        rb + i64::from(aln.tb)
                    };
                    let b_re = if is_rev {
                        (l_pac << 1) - (rb + i64::from(aln.tb))
                    } else {
                        rb + i64::from(aln.te) + 1
                    };
                    // bwa fabricates a seed coverage of "half the shorter of the two spans"
                    // (`bwamem_pair.cpp:236`) because a rescued region has no seeds behind it, yet
                    // `mem_approx_mapq_se` and the chain filters read `seedcov`. The `>> 1` is an
                    // arbitrary discount, not a derived quantity.
                    let seedcov = ((b_re - b_rb).min(i64::from(qe - qb)) >> 1) as i32;
                    // The C `memset`s the whole `mem_alnreg_t` to zero and then fills six fields,
                    // so every field left at 0 / -1 / 0.0 below is that memset, not an oversight:
                    // `truesc`, `sub`, `sub_n`, `seedlen0`, `secondary_all`, `w`, `frac_rep` and
                    // `hash` are all zero for a rescued region, and `secondary` is the
                    // one field the C sets to -1 explicitly. `csub` carries ksw's second-best score
                    // (KSW_XSUBO), which later caps the paired MAPQ at the tandem-repeat score.
                    let b = MemAlnReg {
                        rb: b_rb,
                        re: b_re,
                        qb,
                        qe,
                        rid: a.rid,
                        score: aln.score,
                        truesc: 0,
                        sub: 0,
                        csub: aln.score2,
                        sub_n: 0,
                        seedcov,
                        seedlen0: 0,
                        secondary: -1,
                        secondary_all: 0,
                        w: 0,
                        frac_rep: 0.0,
                        is_alt: a.is_alt,
                        // Set only by the ALT branch of mem_mark_primary_se; 0 everywhere it is constructed.
                        alt_sc: 0,
                    };
                    // ---- step 4: insert the accepted hit, keeping `ma` score-sorted ----
                    // Insert keeping `ma` sorted by score descending (bwa's manual insertion at
                    // `bwamem_pair.cpp:240-246`: push, scan for the first element with a *strictly
                    // smaller* score, shift the tail right by one, drop `b` in). The strict
                    // comparison means an equal-scoring existing region keeps its position and the
                    // newcomer lands after it, which our `>=` loop condition reproduces. Getting
                    // this tie-break backwards silently reorders XA hits.
                    let mut ins = 0;
                    while ins < ma.len() && ma[ins].score >= b.score {
                        ins += 1;
                    }
                    grow_one(ma);
                    ma.insert(ins, b);
                }
                n += 1;
            }
        }
        // ---- step 5: dedup, once per orientation ----
        // Dedup after each orientation, not once at the end, and it runs whenever *any* earlier
        // orientation did SW (`if (n) ...`, `bwamem_pair.cpp:277`), even if this one added nothing.
        // The empty query slice is bwa's `query = 0`, which switches `mem_sort_dedup_patch` from
        // "merge overlapping regions" to "drop contained duplicates only". Both the placement and
        // the null query change which regions survive, so neither can be hoisted or simplified.
        if n > 0 {
            let taken = std::mem::take(ma);
            *ma = mem_sort_dedup_patch(fm, opt, &[], taken);
        }
    }
    n
}

/// One insert-consistent orientation's mate-rescue local-SW job (built by [`matesw_collect`]): the
/// mate `query` in this orientation vs the `target` window, plus the coordinates needed to place the
/// hit ([`matesw_apply`]).
#[derive(Clone, Copy)]
struct Orient {
    /// The fetched reference window, already clamped to the anchor's contig by `bns_fetch_seq`:
    /// its `(offset, length)` in the round's arena. The mate in this orientation is `ms` as
    /// sequenced, or the call's `rc` when `is_rev`.
    target: (u32, u32),
    /// The window's start in the doubled coordinate space *after* clamping. Every ksw target
    /// coordinate is relative to this, so it must travel with the job.
    rb: i64,
    /// Whether `query` was reverse complemented, which decides how ksw's coordinates are mapped
    /// back in [`matesw_apply`].
    is_rev: bool,
}

/// One anchor's mate rescue, split into the SW-independent `collect` (which orientations to run) and
/// the SW-dependent `apply` (insert the hits). Mirrors [`mem_matesw`] exactly but lets the SW of many
/// anchors, across the whole pair batch, run through one vectorized [`batched_mate_rescue`].
/// What [`matesw_collect`] reads of an anchor region: its position, contig and ALT flag. The
/// anchors of a whole pair batch sit in one flat vector, which the rescue rounds index.
#[derive(Clone, Copy)]
pub(crate) struct Anchor {
    rb: i64,
    rid: i32,
    is_alt: bool,
}

#[derive(Clone, Copy)]
struct RescueCall {
    /// `skip[r] != 0` means orientation `r` runs no SW. `matesw_apply` re-walks this so its
    /// per-orientation dedup fires on exactly the same iterations as the C's loop.
    skip: [i32; 4],
    /// The job for each non-skipped orientation, or `None` when the window fell off the reference
    /// or landed on the wrong contig. `skip[r] == 0` with `per_r[r] == None` is the C's "window
    /// rejected": no SW, and `n` is not incremented.
    per_r: [Option<Orient>; 4],
    /// The mate's reverse complement in the round's arena, `(offset, length)`, built once per call
    /// when a reverse orientation runs (length 0 when none did). The windows live there too: one
    /// arena per round instead of two allocations per call.
    rc: (u32, u32),
    /// Mate length in bases, needed to flip query coordinates for the reverse orientations.
    l_ms: i64,
    /// The anchor's `rid` and `is_alt`, copied into every region this call produces. Captured at
    /// collect time because `apply` no longer has the anchor.
    ///
    /// `rid`: the contig every rescued region is placed on (the anchor's, by construction, since a
    /// window on any other contig was rejected at collect time).
    rid: i32,
    /// Whether the anchor sat on an ALT contig. Copied verbatim onto each rescued region, because a
    /// mate rescued next to an ALT anchor is itself an ALT hit. Always false in this non-ALT build.
    is_alt: bool,
}

/// Collect the orientations that would run mate rescue for anchor `a` against mate `ms`, reading the
/// current `ma`. Returns `None` when a consistent pair already exists (all four skipped, no SW), so
/// the caller records nothing. The SW jobs depend only on `(query, target)`, not on `ma`.
///
/// This is the first half of [`mem_matesw`], line for line, up to but not including the `ksw_align2`
/// call. It takes `ma` by shared reference precisely to make the split safe: everything it reads
/// from `ma` (the `skip` test) happens strictly before any SW, so hoisting the SW out of the loop
/// cannot observe a different `ma` than the per-pair path would.
///
/// PRECONDITION: `ma` must be in the exact state `mem_matesw` would see at this point, i.e. all
/// earlier anchors for this (pair, direction) must already have been applied. That is what the
/// round structure in [`batch_mate_rescue`] guarantees.
///
/// PARAMETERS: identical to [`mem_matesw`] except for `ma`, which is taken by shared slice here
/// because this half only reads it. `fm`/`bns` fetch and clamp the reference window, `opt` supplies
/// `min_seed_len` (the minimum usable window width), `pes` the per-orientation insert windows, `a`
/// the anchor region of the other read, `ms` this read's nt4 sequence, `ma` this read's current
/// regions (used solely for the `skip` test).
/// Fewer regions than this and the per-anchor scans stay linear: a view costs an allocation and a
/// sort, and a short list is walked faster than it is indexed.
const VIEW_MIN: usize = 16;

/// A rescue target's regions indexed for the two per-anchor scans, the skip test of
/// [`matesw_collect`] (any region at a proper-pair distance of the anchor) and the settled insert
/// of [`matesw_apply`] (any region overlapping the hit). Both are range queries on `rb`, so the
/// linear scans (1.4G rows per 2M pairs: repetitive mates carry a hundred regions and fifty
/// anchors) become binary searches.
#[derive(Default)]
pub(crate) struct RegionView {
    /// `(rb, re, qb, qe)` of every region, ascending by `rb` up to `sorted`, then the few pushed
    /// since, unordered. Meaningful only while `built`.
    by_rb: Vec<(i64, i64, i32, i32)>,
    sorted: usize,
    /// Longest `re - rb` in `by_rb`: how far left of a hit an overlapping region's `rb` can lie.
    max_len: i64,
    built: bool,
    /// Every region has `rid >= 0` and `re > rb`: what lets the insert scan visit only the regions
    /// overlapping the hit, since [`insert_into_settled`] declines on any other kind.
    clean: bool,
    /// Whether the list is what `mem_sort_dedup_patch` left (see [`insert_into_settled`]).
    pub(crate) settled: bool,
    /// Hits [`insert_settled_view`] appended past the list's sorted prefix, put in place by
    /// [`settle_order`] before anything that reads the order.
    pending: usize,
    /// Bumped by [`matesw_apply`] whenever a rescue hit may have changed the list: a skip set
    /// computed at an unchanged version still holds.
    version: u32,
}

thread_local! {
    /// [`RegionView::ensure`]'s tuples and sort keys, kept per thread: a view lives one batch, so
    /// buffers of its own were two allocations per view built.
    static VIEW_SCRATCH: std::cell::RefCell<ViewScratch> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// `(rb, re, qb, qe)` per region, then the packed sort keys.
type ViewScratch = (Vec<(i64, i64, i32, i32)>, Vec<u64>);

impl RegionView {
    /// Index `ma` unless it is short; returns whether the view is usable for it.
    fn ensure(&mut self, ma: &[MemAlnReg]) -> bool {
        if ma.len() < VIEW_MIN {
            return false;
        }
        if !self.built {
            let t_build = crate::rescue_split::start();
            if t_build.is_some() {
                crate::rescue_split::VIEW_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::rescue_split::VIEW_BUILD_ROWS
                    .fetch_add(ma.len() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            // One pass over the 96-byte regions for everything the view needs; the sort then works
            // on the 24-byte tuples alone. Three passes (keys, max length, cleanliness) plus a
            // gather back through `ma` in sorted order were most of `batch_mate_rescue` on x86.
            let (mut max_len, mut clean, mut packable) = (0i64, true, ma.len() < 1 << 24);
            VIEW_SCRATCH.with_borrow_mut(|(tuples, keys)| {
                tuples.clear();
                tuples.extend(ma.iter().map(|q| {
                    max_len = max_len.max(q.re - q.rb);
                    clean &= q.rid >= 0 && q.re > q.rb;
                    packable &= (0..1 << 40).contains(&q.rb);
                    (q.rb, q.re, q.qb, q.qe)
                }));
                self.by_rb.clear();
                // Sorted as packed `rb << 24 | index` words: a primitive sort, not a keyed tuple one.
                if packable {
                    keys.clear();
                    keys.extend(
                        tuples
                            .iter()
                            .enumerate()
                            .map(|(i, q)| (q.0 as u64) << 24 | i as u64),
                    );
                    keys.sort_unstable();
                    self.by_rb
                        .extend(keys.iter().map(|&k| tuples[(k & 0xff_ffff) as usize]));
                } else {
                    self.by_rb.extend_from_slice(tuples);
                    self.by_rb.sort_unstable_by_key(|q| q.0);
                }
            });
            self.sorted = self.by_rb.len();
            self.max_len = max_len;
            self.clean = clean;
            self.built = true;
            crate::rescue_split::stop(&crate::rescue_split::VIEW_BUILD_NS, t_build);
        }
        true
    }

    /// Whether some region's `rb` lies in `[lo, hi]` (none when `lo > hi`).
    fn any_rb_in(&self, lo: i64, hi: i64) -> bool {
        let (head, tail) = self.by_rb.split_at(self.sorted);
        let i = head.partition_point(|q| q.0 < lo);
        (i < head.len() && head[i].0 <= hi) || tail.iter().any(|q| (lo..=hi).contains(&q.0))
    }

    /// Mirror an insertion into `ma`.
    fn push(&mut self, q: &MemAlnReg) {
        if self.built {
            // Appended, and sorted in once more than 16 wait: a sorted insert per hit was a memmove.
            self.by_rb.push((q.rb, q.re, q.qb, q.qe));
            if self.by_rb.len() - self.sorted > 16 {
                // Stable: a sorted run plus a short tail is merged in near-linear time.
                self.by_rb.sort_by_key(|q| q.0);
                self.sorted = self.by_rb.len();
            }
            self.max_len = self.max_len.max(q.re - q.rb);
        }
    }
}

/// The skip scan of [`mem_matesw`] on the view: orientation `r` is already paired when some region
/// of the mate is at distance `[low, high]` of the anchor in that orientation. The linear loop's
/// `mem_infer_dir` test, inverted: the `p2` range that gives `r` with a distance in range, mapped
/// back to an `rb` range on the strand `r` names. `skip[r]` is only ever set, never cleared.
fn skip_from_view(
    view: &RegionView,
    l_pac: i64,
    a_rb: i64,
    pes: &[PeStat; 4],
    skip: &mut [i32; 4],
) {
    let a_rev = a_rb >= l_pac;
    for r in 0..4 {
        if skip[r] != 0 {
            continue;
        }
        let (lo, hi) = (i64::from(pes[r].low), i64::from(pes[r].high));
        // `mem_infer_dir`: `r = opposite_strands ^ (if p2 > b1 { 0 } else { 3 })`.
        let opposite = r == 1 || r == 2;
        let (p_lo, p_hi) = if r == 0 || r == 1 {
            (a_rb + lo.max(1), a_rb + hi)
        } else {
            (a_rb - hi, a_rb - lo.max(0))
        };
        // `p2` is `rb` on the anchor's strand and `2 * l_pac - 1 - rb` on the other.
        let (rb_lo, rb_hi) = if opposite {
            ((l_pac << 1) - 1 - p_hi, (l_pac << 1) - 1 - p_lo)
        } else {
            (p_lo, p_hi)
        };
        let (s_lo, s_hi) = if a_rev != opposite {
            (l_pac, (l_pac << 1) - 1)
        } else {
            (0, l_pac - 1)
        };
        if view.any_rb_in(rb_lo.max(s_lo), rb_hi.min(s_hi)) {
            skip[r] = 1;
        }
    }
}

/// [`insert_into_settled`] through a built, clean view: its overlap test needs `or_ > 0` when
/// `0 <= mask_level_redun`, so only the regions overlapping `b` in reference coordinates can pass
/// it or decline the insert, and the scan visits those alone. Same answer, same `ma` afterwards.
fn insert_settled_view(
    opt: &MemOpt,
    ma: &mut Vec<MemAlnReg>,
    view: &mut RegionView,
    b: MemAlnReg,
) -> Result<(), MemAlnReg> {
    let mask_level_redun = opt.mask_level_redun;
    if ma.is_empty() || b.rid < 0 || b.qe <= b.qb || b.re <= b.rb {
        return Err(b);
    }
    let mut identical = false;
    let (head, tail) = view.by_rb.split_at(view.sorted);
    let start = head.partition_point(|q| q.0 <= b.rb - view.max_len);
    let near = head[start..].iter().take_while(|q| q.0 < b.re);
    for &(rb, re, qb, qe) in near.chain(tail) {
        if rb >= b.re || re <= b.rb {
            continue;
        }
        let or_ = if re <= b.re { re - b.rb } else { b.re - rb };
        let oq = if re <= b.re {
            if qb < b.qb {
                qe - b.qb
            } else {
                b.qe - qb
            }
        } else if b.qb < qb {
            b.qe - qb
        } else {
            qe - b.qb
        };
        let mr = (re - rb).min(b.re - b.rb);
        let mq = (qe - qb).min(b.qe - b.qb);
        if or_ as f32 > mask_level_redun * mr as f32 && oq as f32 > mask_level_redun * mq as f32 {
            // The identity test needs the whole region; every `ma` entry with these coordinates
            // passed the overlap test above the same way.
            let same = ma
                .iter()
                .filter(|q| q.rb == rb && q.re == re && q.qb == qb && q.qe == qe)
                .all(|q| same_region(q, &b));
            if !same {
                return Err(b);
            }
            identical = true;
        }
    }
    if !identical {
        view.push(&b);
        // Appended, not inserted: a rescued hit lands near the front of a long score-descending
        // list, and shifting ~100 96-byte regions per hit was 2 s of memmove per 2M pairs. Until
        // `settle_order`, only order-blind readers see the list (the skip scan, this function).
        grow_one(ma);
        ma.push(b);
        view.pending += 1;
    }
    Ok(())
}

/// Put the `*pending` hits appended past `ma`'s sorted prefix where [`insert_into_settled`]'s
/// `partition_point` insert would have put them one by one: by `(score desc, rb, qb)`, each before
/// the regions with its key, so among equal keys the later hit first. One backward merge.
fn settle_order(ma: &mut Vec<MemAlnReg>, pending: &mut usize) {
    if *pending == 0 {
        return;
    }
    let key = |r: &MemAlnReg| (-i64::from(r.score), r.rb, r.qb);
    let mut a = ma.len() - *pending;
    *pending = 0;
    let mut tail = ma.split_off(a);
    // Stable on the reversed hits: equal keys come out later hit first.
    tail.reverse();
    tail.sort_by_key(key);
    ma.extend_from_slice(&tail);
    let mut k = ma.len();
    for t in tail.into_iter().rev() {
        while a > 0 && key(&ma[a - 1]) >= key(&t) {
            k -= 1;
            a -= 1;
            ma[k] = ma[a].clone();
        }
        k -= 1;
        ma[k] = t;
    }
}

/// Prefetch the reference windows [`matesw_collect`] would fetch for anchor `a`, in each
/// orientation the insert-size model has not failed. A hint: the contig clamp is left out.
fn prefetch_windows(fm: &FmIndex, l_pac: i64, pes: &[PeStat; 4], a: &Anchor, ms: &[u8]) {
    let l_ms = ms.len() as i64;
    for (r, pe) in pes.iter().enumerate() {
        if pe.failed {
            continue;
        }
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;
        let (lo, hi) = (i64::from(pe.low), i64::from(pe.high));
        let rb = if is_larger { a.rb + lo } else { a.rb - hi };
        let re = if is_larger { a.rb + hi } else { a.rb - lo };
        let (rb, re) = if is_rev {
            (rb - l_ms, re)
        } else {
            (rb, re + l_ms)
        };
        fm.prefetch_pac(rb.max(0), re.min(l_pac << 1));
    }
}

/// The orientations `r` that need no rescue for an anchor at `a_rb`: `pes[r]` failed, or some
/// region of `ma` already sits at a proper-pair distance of the anchor in that orientation.
fn skip_set(
    view: &mut RegionView,
    ma: &[MemAlnReg],
    l_pac: i64,
    a_rb: i64,
    pes: &[PeStat; 4],
) -> [i32; 4] {
    let mut skip = [0i32; 4];
    for (r, skip_r) in skip.iter_mut().enumerate() {
        *skip_r = i32::from(pes[r].failed);
    }
    let t_skip = crate::rescue_split::start();
    if view.ensure(ma) {
        skip_from_view(view, l_pac, a_rb, pes, &mut skip);
    } else {
        for existing in ma.iter() {
            let (r, dist) = mem_infer_dir(l_pac, a_rb, existing.rb);
            if dist >= i64::from(pes[r].low) && dist <= i64::from(pes[r].high) {
                skip[r] = 1;
            }
        }
    }
    if t_skip.is_some() {
        use std::sync::atomic::Ordering::Relaxed;
        crate::rescue_split::COLLECT_CALLS.fetch_add(1, Relaxed);
        crate::rescue_split::COLLECT_SKIP_ROWS.fetch_add(ma.len() as u64, Relaxed);
    }
    crate::rescue_split::stop(&crate::rescue_split::COLLECT_SKIP_NS, t_skip);
    skip
}

#[allow(clippy::too_many_arguments)]
fn matesw_collect(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &Anchor,
    ms: &[u8],
    ma: &[MemAlnReg],
    view: &mut RegionView,
    arena: &mut Vec<u8>,
) -> Option<RescueCall> {
    // `l_pac`: packed reference length (doubled space is [0, 2*l_pac)). `l_ms`: mate length in
    // bases, stored into the returned call because `matesw_apply` needs it to flip query
    // coordinates. `skip[r] != 0`: orientation `r` needs no rescue (failed, or already paired).
    let l_pac = bns.l_pac;
    let l_ms = ms.len() as i64;
    let skip = skip_set(view, ma, l_pac, a.rb, pes);
    if skip.iter().sum::<i32>() == 4 {
        return None;
    }
    let t_fetch = crate::rescue_split::start();
    // `per_r[r]`: the SW job for orientation `r`, left `None` when `r` is skipped or its window was
    // rejected. Every binding inside the loop below (`is_rev`, `is_larger`, `seq`, `lo`/`hi`,
    // `rb0`/`re0`, `rb`/`re`/`rid`/`refseq`) has the same meaning as in [`mem_matesw`], which
    // carries the explanations; this copy stops at building the job instead of calling ksw.
    let mut per_r: [Option<Orient>; 4] = [None, None, None, None];
    let mut rc = (0u32, 0u32);
    for r in 0..4 {
        if skip[r] != 0 {
            continue;
        }
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;
        let (lo, hi) = (i64::from(pes[r].low), i64::from(pes[r].high));
        let (rb0, re0) = if !is_rev {
            let rb = if is_larger { a.rb + lo } else { a.rb - hi };
            let re = (if is_larger { a.rb + hi } else { a.rb - lo }) + l_ms;
            (rb, re)
        } else {
            let rb = (if is_larger { a.rb + lo } else { a.rb - hi }) - l_ms;
            let re = if is_larger { a.rb + hi } else { a.rb - lo };
            (rb, re)
        };
        let rb0 = rb0.max(0);
        let re0 = re0.min(l_pac << 1);
        if rb0 < re0 {
            let off = arena.len();
            let (rb, re, rid) = bns_fetch_seq_into(fm, bns, rb0, (rb0 + re0) >> 1, re0, arena);
            if dump_pestat() {
                eprintln!(
                    "[MSW_R] anchor={} r={r} rb={rb} re={re} rid={rid} a_rid={}",
                    a.rb, a.rid
                );
            }
            if a.rid == rid && re - rb >= i64::from(opt.min_seed_len) {
                let wlen = (arena.len() - off) as u32;
                if is_rev && rc.1 == 0 {
                    rc = (arena.len() as u32, ms.len() as u32);
                    arena.extend(ms.iter().rev().map(|&b| if b < 4 { 3 - b } else { 4 }));
                }
                per_r[r] = Some(Orient {
                    target: (off as u32, wlen),
                    rb,
                    is_rev,
                });
            } else {
                arena.truncate(off);
            }
        }
    }
    crate::rescue_split::stop(&crate::rescue_split::COLLECT_FETCH_NS, t_fetch);
    Some(RescueCall {
        skip,
        per_r,
        rc,
        l_ms,
        rid: a.rid,
        is_alt: a.is_alt,
    })
}

/// The `KswJob`s of a `RescueCall`, in orientation order (the order `matesw_apply` consumes results).
///
/// `flatten()` on `[Option<Orient>; 4]` yields only the `Some` entries, ascending in `r`, and
/// `matesw_apply` walks `r` in the same order taking one result per `Some`. That shared ordering is
/// the entire contract between the two halves: there is no index stored in the job, so if either
/// side changed its traversal the results would be silently mismatched to the wrong orientation.
///
/// PARAMETERS: `call` is one anchor's collected rescue and `ms` the mate it was collected for,
/// borrowed for as long as the iterator lives (the jobs hold slices, they copy nothing).
///
/// RETURNS between 0 and 4 jobs, one per `Some` entry in `call.per_r`, ascending in orientation.
/// [`batched_mate_rescue`] over `jobs`, each scored with the matrix its mate gets in
/// [`mem_matesw`]: `opt.mat` outside `--meth`; under it `MethOpt::mats[group[i]]`, where
/// `group[i]` is the rescued mate's chemistry (read 1 is C->T) XOR the job's `is_rev`. The two
/// groups run as two kernel calls, results back in `jobs` order.
fn rescue_kernel(opt: &MemOpt, jobs: &[KswJob], group: &[bool]) -> Vec<KswAlignResult> {
    let run = |jobs: &[KswJob], mat: &[i8]| {
        batched_mate_rescue(
            jobs,
            5,
            mat,
            opt.o_del,
            opt.e_del,
            opt.o_ins,
            opt.e_ins,
            opt.min_seed_len * opt.a,
            opt.a,
        )
    };
    let Some(meth) = &opt.meth else {
        return run(jobs, &opt.mat);
    };
    let mut out: Vec<Option<KswAlignResult>> = vec![None; jobs.len()];
    for g in [false, true] {
        let idx: Vec<usize> = (0..jobs.len()).filter(|&i| group[i] == g).collect();
        if idx.is_empty() {
            continue;
        }
        let sub: Vec<KswJob> = idx.iter().map(|&i| jobs[i]).collect();
        for (i, r) in idx.into_iter().zip(run(&sub, &meth.mats[usize::from(g)])) {
            out[i] = Some(r);
        }
    }
    out.into_iter()
        .map(|r| r.expect("every job is in one group"))
        .collect()
}

fn rescue_jobs<'a>(
    call: &'a RescueCall,
    ms: &'a [u8],
    arena: &'a [u8],
) -> impl Iterator<Item = KswJob<'a>> {
    call.per_r.iter().flatten().map(move |o| KswJob {
        query: if o.is_rev {
            &arena[call.rc.0 as usize..(call.rc.0 + call.rc.1) as usize]
        } else {
            ms
        },
        target: &arena[o.target.0 as usize..(o.target.0 + o.target.1) as usize],
    })
}

/// Apply a `RescueCall`'s SW results (`alns`, one per collected orientation, in order) to `ma`,
/// inserting each accepted hit and deduping after each orientation, exactly as [`mem_matesw`] does.
///
/// The second half of [`mem_matesw`]: see that function for the meaning of every coordinate
/// transform, the `aln.score >= opt.min_seed_len` asymmetry, the fabricated `seedcov`, the
/// score-descending insertion tie-break and the placement of the dedup. This copy exists only so
/// the SW in between can be batched, and must be kept in lockstep with it.
///
/// PRECONDITION: `alns.len()` equals the number of `Some` entries in `call.per_r`, and the results
/// are in ascending orientation order (see [`rescue_jobs`]). The `alns[ai]` index is not bounds
/// checked beyond Rust's own panic.
///
/// PARAMETERS: `fm`/`opt` are needed only for the per-orientation `mem_sort_dedup_patch`, `bns` only
/// for `l_pac`. `call` is the matching [`matesw_collect`] output (its `skip` drives the loop, its
/// `per_r` the coordinate mapping, its `rid`/`is_alt` the new regions' fields). `alns` are the SW
/// results for that call's jobs, in orientation order. `ma` is the mate's region vector, mutated in
/// place and left sorted score-descending. `ma_settled` says whether `ma` is what
/// [`insert_into_settled`] requires, starts false for a list fresh from the first dedup, and is kept
/// up to date here. `pend`, when given, takes the jobs of results whose `score2` is pending (see
/// [`batch_mate_rescue`]), `read` naming the rescued end (0 or 1) for [`PendS2::fetch`].
#[allow(clippy::too_many_arguments)]
fn matesw_apply(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    call: &RescueCall,
    alns: &[KswAlignResult],
    ma: &mut Vec<MemAlnReg>,
    view: &mut RegionView,
    mut pend: Option<&mut Vec<PendS2>>,
    read: u8,
) {
    let l_pac = bns.l_pac;
    let l_ms = call.l_ms;
    // `n` counts orientations that actually ran SW, exactly as in the C, because the trailing
    // `if n > 0` dedup is keyed on it: once any orientation has run, every subsequent iteration of
    // this loop dedups, whether or not it produced a hit.
    // `ai`: cursor into `alns`. INVARIANT at the top of each iteration: `ai` is the number of
    // orientations below `r` that carried a job, so `alns[ai]` is this orientation's result. This is
    // the only link between a job and its orientation (see [`rescue_jobs`]).
    let mut n = 0;
    let mut ai = 0usize;
    // Whether `ma` has actually changed since the last `mem_sort_dedup_patch` below.
    //
    // `n` counts orientations that RAN, not orientations that INSERTED, so the C re-runs the whole
    // sort/dedup at the end of every subsequent orientation even when that orientation found
    // nothing and the vector is bit-for-bit what the previous dedup returned. On real data that is
    // most of the calls: `BWA4_DEDUP_SHAPE` measures 1.84M calls with n >= 65 regions coming from
    // this path, and a `sample` profile puts 13.3% of busy time in `mem_sort_dedup_patch`.
    //
    // Skipping the no-op re-runs is byte-identical because the function is a fixed point on its own
    // output. Its last pass sorts by `(score desc, rb, qb)` and then removes entries equal on all
    // three, so the comparator is a strict TOTAL order on whatever survives: the sorted result is
    // determined by the surviving set alone, not by the input permutation, even though the sort
    // itself is unstable. Feeding that result back in therefore cannot reorder it, and the
    // redundancy scan cannot kill anything either, since every surviving pair already failed that
    // test in the run that produced them.
    //
    // That fixed point holds only for this null-query mode's OWN output. A list that is not yet
    // settled comes from the single-end dedup, which ran WITH the query and so may have patched two
    // regions into one that now contains a third; the first null-query pass drops that third, so it
    // must run even when no hit was inserted (1 read in 173M of HG002 30x lost an XA hit to this).
    let mut dirty = !view.settled;
    for r in 0..4 {
        if call.skip[r] != 0 {
            continue;
        }
        if let Some(o) = &call.per_r[r] {
            // `aln`: this orientation's SW result (coordinates closed and relative to `o.target`).
            let aln = alns[ai];
            ai += 1;
            if aln.score >= opt.min_seed_len && aln.qb >= 0 {
                // Every change to `ma` below (insert, or the dedup that needs `n > 0`) follows this.
                view.version = view.version.wrapping_add(1);
                let t_ins = crate::rescue_split::start();
                if t_ins.is_some() {
                    crate::rescue_split::APPLY_HITS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                // `rb`: doubled-space start of the window `aln`'s target offsets are relative to.
                // `is_rev`: whether the query was reverse complemented, so both axes must be
                // mirrored below. Every binding that follows mirrors [`mem_matesw`] exactly.
                let (rb, is_rev) = (o.rb, o.is_rev);
                let qb = if is_rev {
                    l_ms as i32 - (aln.qe + 1)
                } else {
                    aln.qb
                };
                let qe = if is_rev {
                    l_ms as i32 - aln.qb
                } else {
                    aln.qe + 1
                };
                let b_rb = if is_rev {
                    (l_pac << 1) - (rb + i64::from(aln.te) + 1)
                } else {
                    rb + i64::from(aln.tb)
                };
                let b_re = if is_rev {
                    (l_pac << 1) - (rb + i64::from(aln.tb))
                } else {
                    rb + i64::from(aln.te) + 1
                };
                let seedcov = ((b_re - b_rb).min(i64::from(qe - qb)) >> 1) as i32;
                // A lazy kernel's pending `score2` names this window in the chunk's table.
                let csub = if aln.score2 == SCORE2_PENDING {
                    let pend = pend.as_deref_mut().expect("a pending score2 needs a table");
                    pend.push(PendS2 {
                        rb,
                        len: o.target.1,
                        rev: is_rev,
                        read,
                    });
                    crate::lazy_csub::encode(pend.len() - 1)
                } else {
                    aln.score2
                };
                let b = MemAlnReg {
                    rb: b_rb,
                    re: b_re,
                    qb,
                    qe,
                    rid: call.rid,
                    score: aln.score,
                    truesc: 0,
                    sub: 0,
                    csub,
                    sub_n: 0,
                    seedcov,
                    seedlen0: 0,
                    secondary: -1,
                    secondary_all: 0,
                    w: 0,
                    frac_rep: 0.0,
                    is_alt: call.is_alt,
                    // Set only by the ALT branch of mem_mark_primary_se; 0 everywhere it is constructed.
                    alt_sc: 0,
                };
                // A settled list can often take the hit without the full pass. See the function.
                let b = if view.settled {
                    let r = if view.ensure(ma)
                        && view.clean
                        && (0.0..1.0).contains(&opt.mask_level_redun)
                    {
                        insert_settled_view(opt, ma, view, b)
                    } else {
                        settle_order(ma, &mut view.pending);
                        insert_into_settled(opt, ma, b)
                    };
                    match r {
                        Ok(()) => {
                            n += 1;
                            crate::rescue_split::stop(&crate::rescue_split::APPLY_INSERT_NS, t_ins);
                            continue;
                        }
                        Err(b) => b,
                    }
                } else {
                    b
                };
                settle_order(ma, &mut view.pending);
                let mut ins = 0;
                while ins < ma.len() && ma[ins].score >= b.score {
                    ins += 1;
                }
                grow_one(ma);
                ma.insert(ins, b);
                dirty = true;
                crate::rescue_split::stop(&crate::rescue_split::APPLY_INSERT_NS, t_ins);
            }
            n += 1;
        }
        if n > 0 && dirty {
            let t_dd = crate::rescue_split::start();
            if t_dd.is_some() {
                use std::sync::atomic::Ordering::Relaxed;
                crate::rescue_split::APPLY_DEDUP_CALLS.fetch_add(1, Relaxed);
                crate::rescue_split::APPLY_DEDUP_ROWS.fetch_add(ma.len() as u64, Relaxed);
            }
            settle_order(ma, &mut view.pending);
            let taken = std::mem::take(ma);
            // Below two regions the pass returns its input as it found it, so that proves nothing.
            view.settled = taken.len() >= 2;
            view.built = false;
            *ma = mem_sort_dedup_patch(fm, opt, &[], taken);
            crate::rescue_split::stop(&crate::rescue_split::APPLY_DEDUP_NS, t_dd);
            dirty = false;
        }
    }
}

/// The rescue job behind a pending `csub` ([`crate::lazy_csub`]): the window `rb..rb + len` in the
/// doubled coordinate space, already clamped to its contig, and the mate that was aligned in it,
/// read `read` (0 or 1) as sequenced, reverse complemented when `rev`. Rerunning that job through
/// [`batched_mate_rescue`] gives the `score2` that becomes `csub`.
#[derive(Clone, Copy, Debug)]
pub struct PendS2 {
    pub rb: i64,
    pub len: u32,
    pub rev: bool,
    pub read: u8,
}

impl PendS2 {
    /// The job's query and target, appended to `arena`: `(query start, target start)`; the query
    /// ends where the target starts and the target runs to the arena's end.
    pub fn fetch(&self, fm: &FmIndex, seqs: &[&[u8]; 2], arena: &mut Vec<u8>) -> (usize, usize) {
        let q0 = arena.len();
        let ms = seqs[self.read as usize];
        if self.rev {
            arena.extend(ms.iter().rev().map(|&b| if b < 4 { 3 - b } else { 4 }));
        } else {
            arena.extend_from_slice(ms);
        }
        let t0 = arena.len();
        fm.bases_append(self.rb, self.rb + i64::from(self.len), arena);
        (q0, t0)
    }
}

/// `score2` of each pending job, by [`batched_mate_rescue`] over the same windows the rescue ran.
pub fn resolve_pending(fm: &FmIndex, opt: &MemOpt, jobs: &[(PendS2, [&[u8]; 2])]) -> Vec<i32> {
    let mut arena = Vec::new();
    let spans: Vec<(usize, usize, usize)> = jobs
        .iter()
        .map(|(p, seqs)| {
            let (q0, t0) = p.fetch(fm, seqs, &mut arena);
            (q0, t0, arena.len())
        })
        .collect();
    let kj: Vec<KswJob> = spans
        .iter()
        .map(|&(q0, t0, t1)| KswJob {
            query: &arena[q0..t0],
            target: &arena[t0..t1],
        })
        .collect();
    batched_mate_rescue(
        &kj,
        5,
        &opt.mat,
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.min_seed_len * opt.a,
        opt.a,
    )
    .into_iter()
    .map(|r| r.score2)
    .collect()
}

/// One read pair's rescue inputs for [`batch_mate_rescue`]: the two mates' nt4 codes and their
/// (mutable) dedup'd region vectors. `a0`/`seq0` is read 1, `a1`/`seq1` is read 2.
pub struct PairRescueData<'a> {
    /// Read 1's nt4 codes in sequencing orientation. Used as the *rescue query* for read-2 anchors.
    pub seq0: &'a [u8],
    /// Read 2's nt4 codes. Used as the rescue query for read-1 anchors.
    pub seq1: &'a [u8],
    /// Read 1's regions, sorted score-descending. Mutated in place: read-2 anchors insert here.
    pub a0: &'a mut Vec<MemAlnReg>,
    /// Read 2's regions. Mutated in place by read-1 anchors.
    pub a1: &'a mut Vec<MemAlnReg>,
}

/// Batched mate rescue across a whole pair batch: identical to running [`mem_matesw`] inside each
/// pair's `mem_sam_pe`, but the insert-window SWs of many pairs' anchors run in one vectorized
/// [`batched_mate_rescue`] call, filling the SIMD lanes that a single pair's <=4 orientations cannot.
///
/// bwa-mem2's rescue snapshots each read's near-best regions as anchors (before any rescue), then,
/// per anchor in order, SW-rescues the mate in each orientation that `skip` (a mate region already
/// at a proper-pair distance) leaves open. Later anchors' `skip` sees earlier anchors' inserts. This
/// runs bwa-mem2's own batched shape (`mem_sam_pe_batch_pre` / `_post`), per block of pairs:
/// 1. collect: every anchor of every pair in the block, `skip` read from the mate's list as it was
///    before any rescue;
/// 2. one kernel call for the whole block, identical windows of one (pair, direction) computed once;
/// 3. apply, anchor by anchor in bwa's order: `skip` is read again from the CURRENT list, and the
///    orientations it now closes drop their result. Skip sets only grow with inserts, so a job is
///    never missing; when a dedup removed a region and reopened an orientation, the anchor is
///    collected and computed again on the spot, which is exactly the per-pair path.
///
/// Blocks hold whole pairs and about [`RESCUE_BLOCK_ANCHORS`] anchors: the one-call batch is what
/// lets the K-mer filter reuse a mate's table across its anchors and the band kernel group jobs of
/// like width, and the bound keeps the windows' arena small. The two directions target disjoint
/// arrays (`a1` vs `a0`) and different pairs are independent, so only the per-(pair, direction)
/// anchor order matters, and it is kept.
///
/// PARAMETERS: `fm`/`bns` are the reference. `opt` supplies `max_matesw` (`-m`, the anchor cap),
/// `pen_unpaired` (`-U`, doubling as the anchor score window) and the ksw scoring parameters. `pes`
/// is the batch's insert-size estimate, which must already be final: it is read by every collect.
/// `pairs` is the whole batch, mutated in place, and must be the same pairs, in the same order, that
/// [`mem_sam_pe`] will later be called on with `rescue_done = true`. `pend`, when given, makes
/// `score2` lazy: a rescued region whose `score2` the kernel left pending gets
/// `csub = lazy_csub::encode(i)` with `pend[i]` the job to rerun, through [`resolve_pending`], if
/// SAM emit reads that `csub`.
pub fn batch_mate_rescue(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    pairs: &mut [PairRescueData],
    mut pend: Option<&mut Vec<PendS2>>,
) {
    // `-m` (`opt->max_matesw`, default 50): at most this many anchors per end. bwa expresses it as
    // the loop bound `j < opt->max_matesw`, so a value of 0 disables rescue entirely, and negative
    // values (impossible from the CLI, but the field is a plain `int`) behave the same. `max(0)`
    // avoids the `as usize` on a negative wrapping to a huge cap.
    let anchor_cap = opt.max_matesw.max(0) as usize;
    if anchor_cap == 0 {
        return;
    }
    // shortcut: `--meth` rescues eagerly; the lazy rerun (`resolve_pending`) scores with `opt.mat`,
    // not the per-strand matrices. Make it matrix-aware if --meth rescue time matters.
    if opt.meth.is_some() {
        pend = None;
    }
    // `-U` (`opt->pen_unpaired`, default 17): the phred-scaled penalty for not pairing. It doubles
    // here as the anchor window: any region within `pen` of the best is plausible enough that its
    // mate is worth rescuing.
    let pen = opt.pen_unpaired;

    // Snapshot near-best anchors per (pair, direction), before any rescue. dir 0: read-1 anchors that
    // rescue read 2 (`a1`); dir 1: read-2 anchors that rescue read 1 (`a0`). `spans[pi][dir]` is
    // that list's `(start, len)` in `anchors`, score-descending, at most `anchor_cap` long. Frozen
    // here: regions inserted by rescue must NOT themselves become anchors, and in bwa they cannot,
    // because the C snapshots `b[i]` before its own anchor loop too.
    // Sized for every region the cap lets in: grown by doubling it was a realloc and a copy.
    let most = |regs: &[MemAlnReg]| regs.len().min(anchor_cap);
    let mut anchors: Vec<Anchor> =
        Vec::with_capacity(pairs.iter().map(|p| most(p.a0) + most(p.a1)).sum());
    let mut spans: Vec<[(u32, u32); 2]> = Vec::with_capacity(pairs.len());
    for p in pairs.iter() {
        // Each side is snapshotted on its own: bwa gates only on MEM_F_NO_RESCUE, never on both
        // reads having regions. A read with no regions contributes no anchors but is still the
        // rescue target of the other read's anchors, which is how an unmapped mate gets rescued.
        let mut near_best = |regs: &[MemAlnReg]| -> (u32, u32) {
            let start = anchors.len() as u32;
            let Some(best) = regs.first().map(|r| r.score) else {
                return (start, 0);
            };
            // The vector is sorted score-descending, so the near-best regions are a prefix and
            // capping during the filter keeps the same set as bwa's cap on the anchor loop.
            anchors.extend(
                regs.iter()
                    .filter(|r| r.score >= best - pen)
                    .take(anchor_cap)
                    .map(|r| Anchor {
                        rb: r.rb,
                        rid: r.rid,
                        is_alt: r.is_alt,
                    }),
            );
            (start, anchors.len() as u32 - start)
        };
        let b0 = near_best(p.a0);
        let b1 = near_best(p.a1);
        spans.push([b0, b1]);
    }
    anchor_spread::record(&anchors, &spans);

    let mut views: Vec<[RegionView; 2]> = (0..pairs.len()).map(|_| Default::default()).collect();
    // `arena`: the block's windows and reverse-complemented mates, back to back. `calls`: what
    // collect made of each anchor. Both kept between calls (see [`RESCUE_BUFS`]).
    let (mut arena, mut calls) = RESCUE_BUFS.take();
    // The block's anchors in apply order, `(pair, dir, anchor index)`.
    let mut todo: Vec<(u32, u8, u32)> = Vec::new();
    // Per anchor, `(start, count)` in `job_of`; per collected orientation, its kernel job.
    let mut call_jobs: Vec<(u32, u32)> = Vec::new();
    // Each todo entry's view version at collect time.
    let mut vers: Vec<u32> = Vec::new();
    let mut job_of: Vec<u32> = Vec::new();
    let mut kept: Vec<KswAlignResult> = Vec::with_capacity(4);
    // Identical windows of one (pair, direction): two anchors in identical repeat copies fetch the
    // same bytes, and the query is that direction's mate, so the job is the same. Found through
    // `head` by (pair, dir, length, orientation, window hash), chained through `Seen::next`; a hash
    // match is only a candidate, confirmed byte for byte against the earlier window in the arena.
    let mut head: std::collections::HashMap<u64, u32, std::hash::BuildHasherDefault<PassHash>> =
        Default::default();
    let mut seen: Vec<Seen> = Vec::new();
    // dir 0 is bwa's `i = 0`: read-1 anchors, mate `s[!i]` = read 2.
    fn mate<'a>(p: &PairRescueData<'a>, dir: usize) -> &'a [u8] {
        if dir == 0 {
            p.seq1
        } else {
            p.seq0
        }
    }

    let block = rescue_block_anchors();
    let mut p0 = 0usize;
    while p0 < pairs.len() {
        // Whole pairs, until the block holds `block` anchors.
        todo.clear();
        let mut p1 = p0;
        while p1 < pairs.len() && todo.len() < block {
            for dir in 0..2 {
                todo.extend((0..spans[p1][dir].1).map(|k| (p1 as u32, dir as u8, k)));
            }
            p1 += 1;
        }
        let anchor_of = |&(pi, dir, k): &(u32, u8, u32)| -> &Anchor {
            &anchors[(spans[pi as usize][dir as usize].0 + k) as usize]
        };

        // ---- 1. collect, against each mate list as it was before any rescue ----
        let t_collect = crate::rescue_split::start();
        arena.clear();
        calls.clear();
        vers.clear();
        for (t, at) in todo.iter().enumerate() {
            // The windows two anchors ahead are random DRAM misses: hint them now.
            if let Some(ahead) = todo.get(t + 2) {
                let q = &pairs[ahead.0 as usize];
                prefetch_windows(
                    fm,
                    bns.l_pac,
                    pes,
                    anchor_of(ahead),
                    mate(q, ahead.1 as usize),
                );
            }
            let (pi, dir) = (at.0 as usize, at.1 as usize);
            let p = &pairs[pi];
            let target: &[MemAlnReg] = if dir == 0 { p.a1 } else { p.a0 };
            vers.push(views[pi][dir].version);
            calls.push(matesw_collect(
                fm,
                bns,
                opt,
                pes,
                anchor_of(at),
                mate(p, dir),
                target,
                &mut views[pi][dir],
                &mut arena,
            ));
        }
        // Every collected orientation's job, identical windows of one (pair, dir) merged.
        let mut jobs: Vec<KswJob> = Vec::new();
        // `--meth`: each job's matrix group (see `rescue_kernel`).
        let mut group: Vec<bool> = Vec::new();
        call_jobs.clear();
        job_of.clear();
        head.clear();
        seen.clear();
        for (at, call) in todo.iter().zip(&calls) {
            let start = job_of.len() as u32;
            if let Some(call) = call {
                let slot = 2 * at.0 as usize + at.1 as usize;
                let ms = mate(&pairs[at.0 as usize], at.1 as usize);
                for (job, o) in rescue_jobs(call, ms, &arena).zip(call.per_r.iter().flatten()) {
                    let (len, h) = (o.target.1, window_hash(job.target));
                    let key = seen_key(slot, len, o.is_rev, h);
                    let first = head.get(&key).copied().unwrap_or(u32::MAX);
                    let mut e = first;
                    let mut hit = None;
                    while e != u32::MAX {
                        let c = &seen[e as usize];
                        if c.slot == slot as u32
                            && c.len == len
                            && c.is_rev == o.is_rev
                            && c.hash == h
                            && (c.rb == o.rb
                                || arena[c.off as usize..(c.off + len) as usize] == *job.target)
                        {
                            hit = Some(c.job);
                            break;
                        }
                        e = c.next;
                    }
                    job_of.push(hit.unwrap_or_else(|| {
                        seen.push(Seen {
                            next: first,
                            slot: slot as u32,
                            len,
                            is_rev: o.is_rev,
                            hash: h,
                            rb: o.rb,
                            off: o.target.0,
                            job: jobs.len() as u32,
                        });
                        head.insert(key, (seen.len() - 1) as u32);
                        jobs.push(job);
                        group.push((at.1 == 1) ^ o.is_rev);
                        (jobs.len() - 1) as u32
                    }));
                }
            }
            call_jobs.push((start, job_of.len() as u32 - start));
        }
        crate::rescue_split::stop(&crate::rescue_split::COLLECT_NS, t_collect);

        // ---- 2. one kernel call for the block ----
        // No `lanes` argument, unlike the scalar [`mem_matesw`] call: `batched_mate_rescue` bins jobs
        // by length and its u8 kernel uses unsigned saturating lanes, documented exact in
        // `bwa-neon/src/batched.rs`, so width never changes the reported score.
        let t_kernel = crate::rescue_split::start();
        let alns = if jobs.is_empty() {
            Vec::new()
        } else {
            if pend.is_some() {
                batched_mate_rescue_lazy(
                    &jobs,
                    5,
                    &opt.mat,
                    opt.o_del,
                    opt.e_del,
                    opt.o_ins,
                    opt.e_ins,
                    opt.min_seed_len * opt.a,
                    opt.a,
                )
            } else {
                rescue_kernel(opt, &jobs, &group)
            }
        };
        drop(jobs);
        crate::rescue_split::stop(&crate::rescue_split::KERNEL_NS, t_kernel);

        // ---- 3. apply in bwa's order, `skip` re-read from the current lists ----
        let t_apply = crate::rescue_split::start();
        for (t, at) in todo.iter().enumerate() {
            let (pi, dir) = (at.0 as usize, at.1 as usize);
            let a = *anchor_of(at);
            let p = &mut pairs[pi];
            let ms = mate(p, dir);
            let target = if dir == 0 { &mut *p.a1 } else { &mut *p.a0 };
            let view = &mut views[pi][dir];
            // No hit applied to this list since its collect: the skip set collect computed holds.
            let skip = match &calls[t] {
                Some(c) if vers[t] == view.version => c.skip,
                _ => skip_set(view, target, bns.l_pac, a.rb, pes),
            };
            let was = calls[t].as_ref().map_or([1; 4], |c| c.skip);
            if (0..4).any(|r| skip[r] == 0 && was[r] != 0) {
                // A dedup removed the region that closed an orientation: the per-pair path, here.
                let mut own = Vec::new();
                if let Some(call) =
                    matesw_collect(fm, bns, opt, pes, &a, ms, target, view, &mut own)
                {
                    let jobs: Vec<KswJob> = rescue_jobs(&call, ms, &own).collect();
                    let group: Vec<bool> = call
                        .per_r
                        .iter()
                        .flatten()
                        .map(|o| (dir == 1) ^ o.is_rev)
                        .collect();
                    let alns = rescue_kernel(opt, &jobs, &group);
                    matesw_apply(fm, bns, opt, &call, &alns, target, view, None, 0);
                }
                continue;
            }
            let Some(call) = &calls[t] else {
                continue; // all four closed then and now
            };
            // This anchor's results in orientation order, without the ones `skip` has since closed.
            let (start, _) = call_jobs[t];
            kept.clear();
            let mut ai = start as usize;
            for r in 0..4 {
                if call.per_r[r].is_some() {
                    if skip[r] == 0 {
                        kept.push(alns[job_of[ai] as usize]);
                    }
                    ai += 1;
                }
            }
            let call = RescueCall { skip, ..*call };
            // dir 0 rescues read 2 (index 1), dir 1 read 1.
            let read = (dir == 0) as u8;
            matesw_apply(
                fm,
                bns,
                opt,
                &call,
                &kept,
                target,
                view,
                pend.as_deref_mut(),
                read,
            );
        }
        crate::rescue_split::stop(&crate::rescue_split::APPLY_NS, t_apply);
        p0 = p1;
    }
    for (p, v) in pairs.iter_mut().zip(&mut views) {
        settle_order(p.a1, &mut v[0].pending);
        settle_order(p.a0, &mut v[1].pending);
    }
    RESCUE_BUFS.set((arena, calls));
}

thread_local! {
    /// [`batch_mate_rescue`]'s window arena and per-anchor calls, kept between calls: grown from
    /// empty each time they were a sixth of the run's realloc copies (chr21, 200k pairs: 0.23 of
    /// 1.29 GB). Cleared per block before use.
    static RESCUE_BUFS: std::cell::Cell<(Vec<u8>, Vec<Option<RescueCall>>)> =
        const { std::cell::Cell::new((Vec::new(), Vec::new())) };
}

/// Anchors per [`batch_mate_rescue`] block; `BWA4_RESCUE_BLOCK` overrides it for measurement.
fn rescue_block_anchors() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BWA4_RESCUE_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(RESCUE_BLOCK_ANCHORS)
    })
}

/// See [`batch_mate_rescue`].
const RESCUE_BLOCK_ANCHORS: usize = 4096;

/// One collected rescue job of a [`batch_mate_rescue`] block, for the identical-window lookup: its
/// window's length, mate orientation, [`window_hash`], start and arena offset, its kernel job, and
/// the list's next entry.
#[derive(Clone, Copy)]
struct Seen {
    next: u32,
    slot: u32,
    len: u32,
    is_rev: bool,
    hash: u64,
    rb: i64,
    off: u32,
    job: u32,
}

/// Room for one more region, growing by an eighth rather than doubling. A batch holds every read's
/// regions until SAM emit, and doubling on the rescued hit left a few-region list half empty: 0.15
/// GB of chr21's 2.2 GB peak. An eighth keeps repeat-heavy lists (hundreds of regions, many rescued
/// hits each) amortized.
fn grow_one(ma: &mut Vec<MemAlnReg>) {
    if ma.len() == ma.capacity() {
        ma.reserve_exact(1 + ma.len() / 8);
    }
}

/// [`Seen`]'s lookup key: the slot, length and orientation mixed into the window hash.
fn seen_key(slot: usize, len: u32, is_rev: bool, h: u64) -> u64 {
    const M: u64 = 0x9E37_79B9_7F4A_7C15;
    (h ^ (slot as u64) << 33 ^ u64::from(len) << 1 ^ u64::from(is_rev)).wrapping_mul(M)
}

/// A hasher that passes an already mixed `u64` key through.
#[derive(Default)]
struct PassHash(u64);
impl std::hash::Hasher for PassHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ u64::from(x)).wrapping_mul(0x1000_0000_01b3);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

/// A 64-bit hash of a rescue window's bytes, four independent multiply lanes over 8-byte words.
/// Only selects candidates for [`batch_mate_rescue`]'s repeat lookup, which then compares bytes.
fn window_hash(b: &[u8]) -> u64 {
    const M: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = [b.len() as u64, 1, 2, 3];
    let (words, rest) = b.as_chunks::<32>();
    for w in words {
        for (l, x) in h.iter_mut().zip(w.as_chunks::<8>().0) {
            *l = (*l ^ u64::from_le_bytes(*x))
                .wrapping_mul(M)
                .rotate_left(31);
        }
    }
    let (tail_words, tail) = rest.as_chunks::<8>();
    for (l, x) in h.iter_mut().zip(tail_words) {
        *l = (*l ^ u64::from_le_bytes(*x))
            .wrapping_mul(M)
            .rotate_left(31);
    }
    if !tail.is_empty() {
        // The last eight bytes, overlapping the words before; the length is already in `h[0]`.
        let x = match b.len().checked_sub(8) {
            Some(at) => u64::from_le_bytes(b[at..].try_into().unwrap()),
            None => tail.iter().rev().fold(0, |v, &x| v << 8 | u64::from(x)),
        };
        h[3] = (h[3] ^ x).wrapping_mul(M).rotate_left(31);
    }
    (h[0] ^ h[1].rotate_left(16) ^ h[2].rotate_left(32) ^ h[3].rotate_left(48)).wrapping_mul(M)
}

/// bwa's `id << 8`, reproduced with its C integer width.
///
/// `mem_pair` is declared `int mem_pair(..., int id, ...)`, so the 64-bit pair id is **truncated to
/// a 32-bit signed int** at the call, and `id << 8` is evaluated in `int` arithmetic. From pair id
/// 2^23 that shift reaches the sign bit; the (now negative) `int` is then converted to `uint64_t`
/// for `p->y ^ id<<8`, which **sign-extends** it:
///
/// | pair id   | this fn / bwa        | a clean 64-bit `id << 8` |
/// |-----------|----------------------|--------------------------|
/// | 8_388_607 | `0x000000007fffff00` | `0x000000007fffff00`     |
/// | 8_388_608 | `0xffffffff80000000` | `0x0000000080000000`     |
///
/// Computing this in `u64` (as we used to) therefore feeds `hash_64` a different key for every pair
/// past 2^23, flipping the pairing tie-break on equally-scoring hits. On real 30x WGS that showed up
/// as ~3% of records differing -- all MAPQ 0 -- from read 8_388_608 onwards, and nowhere before.
/// The overflow is UB in C but deterministic two's-complement wrapping on every target bwa runs on,
/// so byte-identity requires reproducing it rather than "fixing" it.
///
/// PARAMETERS: `id` is the global 0-based read-pair index, as handed to [`mem_sam_pe`]. Any value is
/// accepted; only its low 32 bits survive.
///
/// RETURNS the salt to XOR into `hash_64`'s input, matching C's `(uint64_t)(id << 8)` bit for bit.
#[must_use]
fn id_shift_c(id: u64) -> u64 {
    ((id as u32 as i32).wrapping_shl(8)) as i64 as u64
}

/// Sub-optimal score of a read: the best rival placement of the *same* stretch of the read
/// (`r[0]` is the best hit overall). Port of `cal_sub` (`bwamem_pair.cpp:67-79`).
///
/// WHY: `mem_pestat` only wants pairs where both ends are placed near-uniquely, so it needs a
/// measure of "how good is the runner-up for this locus". A region covering a different part of the
/// query is not a runner-up, it is another piece of a split alignment, so the scan walks past those.
///
/// RETURNS the score of the first (hence highest-scoring, the vector being score-sorted) region
/// whose query interval overlaps `r[0]`'s by at least `mask_level` times the shorter of the two
/// spans. If no region qualifies, the floor `min_seed_len * a` is returned: the score of a bare
/// seed, i.e. "there is effectively no competitor".
///
/// PRECONDITION: `r` is score-sorted descending, so `r[0]` is the best hit. Empty `r` is not
/// possible here: `mem_pestat` checks both ends non-empty first.
///
/// PARAMETERS: `opt` supplies `mask_level` (the overlap fraction that makes a region count as a
/// rival, an f32 in [0, 1], default 0.50) and `min_seed_len * a` (the "no competitor" floor). `r` is
/// one read's dedup'd region vector, supplied by [`mem_pestat`] before any mate rescue has run.
#[must_use]
fn cal_sub(opt: &MemOpt, r: &[MemAlnReg]) -> i32 {
    // The C loop variable escapes the loop (`return j < r->n? ...`), so the "no competitor found"
    // case is signalled by `j` reaching the end rather than by a sentinel.
    let mut j = 1;
    while j < r.len() {
        // Intersection of the two query intervals: `[b_max, e_min)` is non-empty iff they overlap.
        let b_max = r[j].qb.max(r[0].qb);
        let e_min = r[j].qe.min(r[0].qe);
        if e_min > b_max {
            // Shorter of the two query spans, in read bases: the overlap is judged relative to it,
            // so a short region fully inside a long one always counts as a rival.
            let min_l = (r[j].qe - r[j].qb).min(r[0].qe - r[0].qb);
            // f32, not f64: `opt->mask_level` is a C `float`, so `min_l * opt->mask_level` promotes
            // the int to float and the comparison happens entirely in single precision. Widening to
            // f64 flips the boundary cases (0.5f is exact, but the default 0.50 is not the only
            // value users pass), which changes which region is reported as the sub-optimal score
            // and hence whether this pair feeds the insert-size histogram at all.
            //
            // Note the direction: the loop BREAKS on significant overlap, so the score returned is
            // that of the best-scoring region covering substantially the same part of the read as
            // `r[0]`. Regions elsewhere on the read are skipped, because they are a different piece
            // of a split alignment, not a rival placement of the same piece. The C comments this
            // loop "choose unique alignment".
            if (e_min - b_max) as f32 >= min_l as f32 * opt.mask_level {
                break;
            }
        }
        j += 1;
    }
    if j < r.len() {
        r[j].score
    } else {
        opt.min_seed_len * opt.a
    }
}

/// Env-gated (`BWA4_DUMP_PESTAT`) tracing of insert-size stats and every mate-rescue decision,
/// phrased in bwa's own `[PE]` wording so the two programs' stderr can be diffed directly.
///
/// This is a debugging lever, not part of the algorithm: bwa prints the same lines unconditionally
/// under `bwa_verbose >= 3`, and we gate them behind an env var so normal runs stay quiet. Worth
/// reaching for first when PE output diverges, because the insert-size distribution is estimated
/// per batch: one different number here moves the rescue and pairing decisions of every pair in the
/// batch, so a whole-batch diff is far more informative than staring at a single record.
///
/// The result is cached in a `OnceLock`, so setting the variable mid-process has no effect.
///
/// RETURNS true when `BWA4_DUMP_PESTAT` was set (to anything, including the empty string) at the
/// first call.
fn dump_pestat() -> bool {
    // Process-wide cache of the env lookup, so the check costs nothing on the hot rescue path.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_DUMP_PESTAT").is_some())
}

/// Estimate insert-size distributions for the four orientations over a whole batch. Port of
/// `mem_pestat` (`bwamem_pair.cpp:81-147`).
///
/// WHY per batch: bwa re-estimates the library's insert-size distribution from the reads it has in
/// hand rather than taking it as a parameter, so the same FASTQ split differently across batches
/// can give slightly different `pes` and hence different output. Byte-identity therefore requires
/// matching bwa's batch boundaries as well as this arithmetic.
///
/// PARAMETERS: `opt` supplies `mask_level` (via `cal_sub`) and `max_ins` (`-I`'s cap, the largest
/// insert allowed into the histogram). `l_pac` is the packed reference length. `regs` holds the
/// dedup'd, score-sorted regions of `2N` interleaved reads (`regs[2i]` = R1, `regs[2i+1]` = R2); an
/// odd length simply drops the trailing read, since the loop runs `regs.len() / 2` times.
///
/// RETURNS the four `PeStat`s by orientation code (FF, FR, RF, RR). Orientations with too little
/// support come back `failed`, which downstream code treats as "never pair or rescue this way".
pub fn mem_pestat(opt: &MemOpt, l_pac: i64, regs: &[&[MemAlnReg]]) -> [PeStat; 4] {
    mem_pestat_from(pestat_samples(opt, l_pac, regs))
}

/// [`mem_pestat`]'s step 1 alone: the insert sizes it fits, one list per orientation. A caller may
/// collect them over slices of the batch (an even number of reads each) and concatenate the lists
/// in any order, since [`mem_pestat_from`] sorts them.
pub fn pestat_samples(opt: &MemOpt, l_pac: i64, regs: &[&[MemAlnReg]]) -> [Vec<i64>; 4] {
    // One insert-size sample list per orientation. bwa uses `uint64_v`; the values are distances so
    // they are non-negative, and we keep them as i64 to avoid signed/unsigned comparison hazards in
    // the inlier tests below.
    // (bwa calls this array `isize`, which in Rust would shadow the primitive integer type.)
    let mut insert_sizes: [Vec<i64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    // ---- step 1: collect one insert size per near-uniquely-placed, same-contig pair ----
    // Number of complete pairs in the interleaved batch; a trailing unpaired read is ignored.
    let n_pairs = regs.len() / 2;
    for i in 0..n_pairs {
        // `r0`/`r1`: this pair's two ends' region vectors, score-sorted, pre-rescue.
        let r0 = regs[i << 1];
        let r1 = regs[(i << 1) | 1];
        // Four filters, all of them "this pair is not trustworthy enough to define the library":
        // either end unmapped, either end ambiguous (sub-optimal within 80% of best), or the two
        // ends on different chromosomes. Note bwa always uses `a[0]`, the top hit, never a rescued
        // or secondary one, and that this runs *before* mate rescue, so the distribution is
        // estimated only from pairs that seeding placed on its own.
        if r0.is_empty() || r1.is_empty() {
            continue;
        }
        // f64 here, unlike `cal_sub`'s internal f32 test: the C promotes the int score to `double`
        // against the `double` constant 0.8.
        if f64::from(cal_sub(opt, r0)) > MIN_RATIO * f64::from(r0[0].score) {
            continue;
        }
        if f64::from(cal_sub(opt, r1)) > MIN_RATIO * f64::from(r1[0].score) {
            continue;
        }
        if r0[0].rid != r1[0].rid {
            continue;
        }
        // `is == 0` (the two 5' ends coincide) is excluded, not just clamped: a zero insert is a
        // degenerate self-overlap, and `opt.max_ins` drops the long tail that would otherwise drag
        // the percentiles. `max_ins` is fixed at 10000 (`bwamem.cpp:127`) and exposed by no command
        // line flag in either bwa-mem2 or this port.
        // `ins_size` is bwa's `is`, the pair's insert size in reference bases.
        let (dir, ins_size) = mem_infer_dir(l_pac, r0[0].rb, r1[0].rb);
        if ins_size != 0 && ins_size <= i64::from(opt.max_ins) {
            insert_sizes[dir].push(ins_size);
        }
    }
    insert_sizes
}

/// [`mem_pestat`]'s step 2: the four orientations' fit, from [`pestat_samples`]' lists in any
/// order (each is sorted here, a no-op on a sorted list).
pub fn mem_pestat_from(mut insert_sizes: [Vec<i64>; 4]) -> [PeStat; 4] {
    let mut pes = [PeStat::default(); 4];
    if dump_pestat() {
        eprintln!(
            "[PE] # candidate unique pairs for (FF, FR, RF, RR): ({}, {}, {}, {})",
            insert_sizes[0].len(),
            insert_sizes[1].len(),
            insert_sizes[2].len(),
            insert_sizes[3].len()
        );
    }
    // ---- step 2: per orientation, fit low/high/avg/std to those samples ----
    for d in 0..4 {
        // `stat` is bwa's `r` (the `mem_pestat_t` being filled), `samples` its `q`.
        let stat = &mut pes[d];
        let samples = &mut insert_sizes[d];
        if samples.len() < MIN_DIR_CNT {
            stat.failed = true;
            continue;
        }
        // bwa sorts with `ks_introsort_64`. Sorting is only used for percentiles here and the values
        // are plain integers, so unstable vs stable cannot matter (unlike the region sorts in
        // `mem_sort_dedup_patch`, where the payload differs between equal keys).
        samples.sort_unstable();
        // Sample count for this orientation, >= MIN_DIR_CNT here; only used for the percentile
        // indices, whose exact expression is bwa's and must not be refactored.
        let n_samples = samples.len();
        // bwa's percentile: `q->a[(int)(.25 * q->n + .499)]`. This is not a standard quantile, there
        // is no interpolation, and the `+ .499` is a truncating round-half-up on the *index*. It can
        // index `n` itself for the 75th percentile only if `.75*n + .499 >= n`, i.e. n <= 1, which
        // `MIN_DIR_CNT` already excludes. Reproduce the expression literally: computing the index
        // any other way shifts `p25`/`p75` by one sample and moves every boundary below.
        let p25 = samples[(0.25 * n_samples as f64 + 0.499) as usize] as i32;
        let p75 = samples[(0.75 * n_samples as f64 + 0.499) as usize] as i32;
        stat.failed = false;
        // First pass: a Tukey fence at 2x IQR, used ONLY to choose which samples count as inliers
        // for the mean and std.dev. `low`/`high` are overwritten further down with the wider 3x
        // fence, so these two assignments never reach the caller.
        stat.low = (f64::from(p25) - OUTLIER_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        if stat.low < 1 {
            stat.low = 1;
        }
        stat.high = (f64::from(p75) + OUTLIER_BOUND * f64::from(p75 - p25) + 0.499) as i32;

        // Mean over the inliers. `n_inliers` is bwa's `x`; the C asserts it is non-zero, which holds
        // because `p25` and `p75` are themselves samples and always lie inside the 2x fence.
        // Two separate passes (mean, then variance) rather than a one-pass sum-of-squares: the
        // one-pass form gives different floating-point results, and `avg`/`std` feed `erfc` in
        // `mem_pair`, where a last-bit difference can flip a pairing score.
        let (mut sum, mut n_inliers) = (0.0f64, 0i64);
        for &v in samples.iter() {
            if v >= i64::from(stat.low) && v <= i64::from(stat.high) {
                sum += v as f64;
                n_inliers += 1;
            }
        }
        stat.avg = sum / n_inliers as f64;
        // Sum of squared deviations from `avg` over the same inliers (not yet divided).
        let mut var = 0.0f64;
        for &v in samples.iter() {
            if v >= i64::from(stat.low) && v <= i64::from(stat.high) {
                var += (v as f64 - stat.avg) * (v as f64 - stat.avg);
            }
        }
        // Population std.dev: divided by `n_inliers` (bwa's `x`), not `x - 1`. Not a bug to "fix".
        stat.std = (var / n_inliers as f64).sqrt();
        // Second pass: the fence that actually ships, at 3x IQR, then widened (never narrowed) to
        // at least mean +/- 4 std.dev. The two `if`s only ever push the bounds outward, so the
        // proper-pair window is the union of the IQR fence and the Gaussian fence. This is what
        // makes the window tolerant of a distribution whose IQR is tight but whose tails are not.
        stat.low = (f64::from(p25) - MAPPING_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        stat.high = (f64::from(p75) + MAPPING_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        if f64::from(stat.low) > stat.avg - MAX_STDDEV * stat.std {
            stat.low = (stat.avg - MAX_STDDEV * stat.std + 0.499) as i32;
        }
        if f64::from(stat.high) < stat.avg + MAX_STDDEV * stat.std {
            stat.high = (stat.avg + MAX_STDDEV * stat.std + 0.499) as i32;
        }
        // `+ 0.499` then `as i32` is C's round-half-up-via-truncation and it truncates toward zero,
        // so a negative `low` becomes a small negative int rather than rounding down. The clamp to
        // 1 makes that moot, and guarantees `low >= 1` for every non-failed orientation.
        if stat.low < 1 {
            stat.low = 1;
        }
        if dump_pestat() {
            eprintln!(
                "[PE] dir={d} (25, 50, 75) percentile: ({p25}, {}, {p75})",
                samples[(0.50 * n_samples as f64 + 0.499) as usize]
            );
            eprintln!(
                "[PE] dir={d} mean and std.dev: ({:.2}, {:.2})",
                stat.avg, stat.std
            );
            eprintln!(
                "[PE] dir={d} low and high boundaries for proper pairs: ({}, {})",
                stat.low, stat.high
            );
        }
    }

    // Final cross-orientation filter. On a normal FR library, FF/RF/RR pick up a scattering of
    // chimeric or mismapped pairs that can clear MIN_DIR_CNT on a big batch; this ratio test kills
    // them by comparing against the dominant orientation rather than against an absolute count.
    // Note it uses the raw sample counts, not the post-fence inlier counts, and that an orientation
    // already `failed` for being under-populated is left alone.
    // ---- step 3: fail any orientation dwarfed by the dominant one ----
    let max_dir_count = insert_sizes.iter().map(Vec::len).max().unwrap_or(0);
    for d in 0..4 {
        if !pes[d].failed && (insert_sizes[d].len() as f64) < max_dir_count as f64 * MIN_DIR_RATIO {
            pes[d].failed = true;
        }
    }
    pes
}

/// Result of `mem_pair`: combined score, sub-optimal, `n_sub`, and the best region index per read.
///
/// The C returns the score as its `int` return value and writes the rest through out-parameters,
/// using `return 0` as "no proper pair found". We return `Option<PairResult>` instead, so the
/// caller's `> 0` test becomes `if let Some(..)`. INVARIANT: `score > 0` whenever this is `Some`,
/// because `u` is only non-empty when at least one pair cleared the `q < 0 -> 0` clamp... except
/// that a pair whose `q` clamped exactly to 0 yields `score == 0`, which the C would treat as "no
/// pair" and fall through to `no_pairing`. UNVERIFIED: whether that zero-score case is reachable in
/// practice, and whether this port therefore takes the paired branch where bwa takes `no_pairing`.
struct PairResult {
    /// Best pair's combined score: the two regions' alignment scores plus the (negative)
    /// insert-size log-likelihood term, floored at 0.
    score: i32,
    /// Second-best pair's combined score, or 0 when only one pair was found. bwa's `*sub`.
    sub: i32,
    /// How many pairs (excluding the best) score within one cheapest-event margin (`mem_pair`'s
    /// `tie_margin`, bwa's `tmp`) of `sub`, i.e. how crowded the
    /// runner-up field is. Subtracted from the paired MAPQ by the caller.
    n_sub: i32,
    /// Index into each read's region vector of the chosen pair: `z[0]` for read 1, `z[1]` for
    /// read 2.
    z: [usize; 2],
}

/// Pair the two ends' regions and pick the best proper pair. Port of `mem_pair`
/// (`bwamem_pair.cpp:285-348`), non-ALT case, where bwa's `n_pri[r]` equals the region count.
///
/// ALGORITHM: build one flat list of all candidate placements from both reads, sort it by genomic
/// position, then sweep it left to right. For each placement, look back at recent placements of the
/// *other* read on a compatible strand and, for every one at a plausible insert distance, emit a
/// candidate pair scored by (sum of alignment scores) + (insert-size log-likelihood). The best pair
/// wins. bwa notes the look-back is O(n^2) worst case; it terminates early on distance.
///
/// PARAMETERS: `bns` supplies `l_pac` and the contig offsets that make positions comparable within
/// a chromosome. `opt` supplies `a`/`b`/`o_del`/`e_del`/`o_ins`/`e_ins`, used only to derive the
/// `tie_margin` that decides how many runner-up pairs count as tied. `a[0]` / `a[1]` are the two
/// reads' primary regions, score-sorted, and every region must have `rid >= 0`. `pes` gates which
/// orientations may pair at all. `id` is the global pair index, used only as a hash salt to break
/// ties deterministically (see [`id_shift_c`], and note it must be salted the way C does).
///
/// RETURNS `None` when no candidate pair existed at all (bwa's `u.n == 0`).
fn mem_pair(
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &[&[MemAlnReg]; 2],
    id: u64,
) -> Option<PairResult> {
    thread_local! {
        /// `v` and `u` below, kept per thread: two or three allocations per pair otherwise. And
        /// the insert-size penalties, kept across the pairs of a batch.
        static PAIR_LISTS: std::cell::RefCell<PairScratch> =
            const { std::cell::RefCell::new(([Vec::new(), Vec::new()], PairPenalty::new())) };
    }
    PAIR_LISTS.with(|c| {
        let ([v, u], pen) = &mut *c.borrow_mut();
        v.clear();
        u.clear();
        pen.reset_for(pes, opt.a);
        mem_pair_in(bns, opt, pes, a, id, [v, u], pen)
    })
}

/// [`mem_pair`]'s per-thread lists `v` and `u`, and its penalty memo.
type PairScratch = ([Vec<(u64, u64)>; 2], PairPenalty);

/// What [`PairPenalty`] was filled for: `(low, high, avg bits, std bits)` per orientation, and
/// `opt.a`.
type PairPenaltyKey = ([(i32, i32, u64, u64); 4], i32);

/// [`mem_pair`]'s insert-size penalty, `INV_LOG4 * ln(2 * erfc(|ns| / sqrt 2)) * a`, by orientation
/// and insert size, which is all it depends on: libm's `erfc` and `log` per candidate pair were
/// ~0.3% of an x86 PE run. Valid for one batch's statistics and one `opt.a` (`key`). A slot is NaN
/// until computed; a penalty that is NaN itself (`a == 0`) is simply recomputed each time.
struct PairPenalty {
    key: Option<PairPenaltyKey>,
    /// Per orientation, the penalty at insert size `low + k` in slot `k`.
    vals: [Vec<f64>; 4],
}

impl PairPenalty {
    /// Insert sizes per orientation the memo covers; a wider `[low, high]` computes the rest.
    const SPAN: i32 = 1 << 16;

    const fn new() -> Self {
        PairPenalty {
            key: None,
            vals: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Empty every slot unless `pes` and `a` are the ones the slots were computed for.
    fn reset_for(&mut self, pes: &[PeStat; 4], a: i32) {
        let key = (
            std::array::from_fn(|d| {
                let p = &pes[d];
                (p.low, p.high, p.avg.to_bits(), p.std.to_bits())
            }),
            a,
        );
        if self.key == Some(key) {
            return;
        }
        for (v, p) in self.vals.iter_mut().zip(pes) {
            v.clear();
            v.resize((p.high - p.low + 1).clamp(0, Self::SPAN) as usize, f64::NAN);
        }
        self.key = Some(key);
    }

    /// The penalty of an insert of `dist`, which lies in `[pes.low, pes.high]`, in orientation `dir`.
    fn get(&mut self, pes: &PeStat, dir: usize, dist: i64, a: i32) -> f64 {
        match self.vals[dir].get_mut((dist - i64::from(pes.low)) as usize) {
            Some(p) if !p.is_nan() => *p,
            slot => {
                // `ns` is bwa's name for the z-score of this insert size.
                let ns = (dist as f64 - pes.avg) / pes.std;
                // SAFETY: libm's `erfc` is a pure function of its argument.
                let erfc_term = unsafe { erfc(ns.abs() * M_SQRT1_2) };
                let pen = INV_LOG4 * (2.0 * erfc_term).ln() * f64::from(a);
                if let Some(p) = slot {
                    *p = pen;
                }
                pen
            }
        }
    }
}

/// [`mem_pair`] on the caller's two empty lists `v` and `u`.
fn mem_pair_in(
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &[&[MemAlnReg]; 2],
    id: u64,
    [v, u]: [&mut Vec<(u64, u64)>; 2],
    pen: &mut PairPenalty,
) -> Option<PairResult> {
    let l_pac = bns.l_pac;
    // `v`: every candidate placement of either read, packed into two u64s so that a plain sort by
    // `x` orders them by genomic position. bwa's `pair64_t` is exactly this pair of fields.
    //
    //   x = rid<<32 | forward-position-within-contig
    //   y = score<<32 | i<<2 | strand<<1 | read
    //
    // Putting `rid` in the high half of `x` makes the sort group by contig first, so the sweep's
    // distance test never straddles a chromosome boundary with a meaningless difference. The
    // low half is offset-relative so it fits in 32 bits for any real contig.
    //
    // `y` is a packed payload, not a sort key in its own right, but it is the tie-break: two
    // placements at the identical position order by score, then by region index. Its three low
    // fields are recovered later by masking (`y & 1` = read, `y >> 1 & 1` = strand,
    // `y << 32 >> 34` = region index).
    // ---- step 1: flatten both reads' regions into one position-sortable list ----
    for (r, regs) in a.iter().enumerate() {
        for (i, reg) in regs.iter().enumerate() {
            // Fold the doubled coordinate back onto the forward strand so both strands' placements
            // are comparable; the strand itself is preserved separately in bit 1 of `y`.
            let fpos = if reg.rb < l_pac {
                reg.rb
            } else {
                (l_pac << 1) - 1 - reg.rb
            };
            // Start of this region's contig in the concatenated forward reference, subtracted so the
            // low half of `x` is a within-contig offset and always fits in 32 bits.
            let off = bns.contigs[reg.rid as usize].offset;
            let x = (u64::from(reg.rid as u32) << 32) | (fpos - off) as u64;
            // `reg.score` is an i32 (negative only under --meth), so the `as u64` shift matches C's
            // `(uint64_t)e->score << 32`. `i << 2` leaves the two low bits for strand and read,
            // which caps the region index at 2^30 (never approached).
            let y = ((reg.score as u64) << 32)
                | ((i as u64) << 2)
                | (((reg.rb >= l_pac) as u64) << 1)
                | r as u64;
            v.push((x, y));
        }
    }
    // bwa's `ks_introsort_128` compares `x` then `y`, which is what Rust's tuple ordering does.
    // Unstable is safe: `(x, y)` pairs are distinct, since `y` carries the region index and read.
    v.sort_unstable();

    // `u` collects every candidate pair. `y[w]` is the index in `v` of the most recent placement
    // whose low two bits (`strand<<1 | read`) equal `w`, i.e. one bucket per (read, strand) combo.
    // It is the sweep's "last hit" memory, which is why the look-back can start at a known-relevant
    // element rather than at 0.
    //
    // ---- step 2: sweep left to right, emitting one scored candidate per compatible look-back ----
    let mut y: [i64; 4] = [-1, -1, -1, -1];
    for i in 0..v.len() {
        for r in 0..2u64 {
            // `r` here is NOT the read index: it is the loop over candidate orientations. `dir`
            // combines `r` with the *current* placement's strand to index `pes`; a failed
            // orientation is skipped outright, which is how `mem_pestat`'s verdict reaches pairing.
            let dir = (r << 1 | ((v[i].1 >> 1) & 1)) as usize;
            if pes[dir].failed {
                continue;
            }
            // The partner must come from the *other* read, hence `^ 1` on the read bit. `which`
            // selects the (read, strand) bucket to look back into.
            let which = (r << 1 | ((v[i].1 & 1) ^ 1)) as usize;
            if y[which] < 0 {
                continue;
            }
            // Walk backwards from the last placement in that bucket. Because `v` is position
            // sorted, `dist` grows monotonically as `k` decreases, so exceeding `high` can `break`
            // out entirely while a too-small `dist` only skips this one element.
            let mut k = y[which];
            while k >= 0 {
                let ku = k as usize;
                // `y[which]` only recorded the newest member of the bucket; older entries are
                // interleaved with other buckets, so each candidate is re-checked here.
                if (v[ku].1 & 3) as usize != which {
                    k -= 1;
                    continue;
                }
                // Both `x` values carry the same `rid` in their high half whenever the two are on
                // the same contig, so the subtraction cancels it and leaves the base distance. On
                // *different* contigs the difference is a multiple of 2^32, hence enormous, hence
                // caught by the `> high` break. That is the whole cross-contig guard.
                let dist = v[i].0 as i64 - v[ku].0 as i64;
                if dist > i64::from(pes[dir].high) {
                    break;
                }
                if dist < i64::from(pes[dir].low) {
                    k -= 1;
                    continue;
                }
                // Insert-size likelihood, in alignment-score units. `ns` is the z-score of this
                // insert under the fitted Gaussian; `2 * erfc(|ns| / sqrt(2))` is the two-sided tail
                // probability of a deviation at least this large. Its natural log is negative, so
                // the term is a *penalty* that grows as the insert strays from the mean, scaled to
                // score units by `opt.a` (the match score) and by [`INV_LOG4`] (bwa's 0.721), which
                // converts nats into log-base-4 units so the penalty is commensurate with a
                // per-base score.
                //
                // f64 throughout and the *system* libm `erfc`: this is the single most
                // parity-sensitive expression in the file. A Rust-side erfc implementation, or f32
                // anywhere in the chain, changes `q` by one on boundary pairs and silently reorders
                // `u`, picking a different "best pair". The memo holds that same f64.
                let penalty = pen.get(&pes[dir], dir, dist, opt.a);
                // `v[..].1 >> 32` recovers each end's alignment score from the packed `y`. The
                // `+ 0.499` is C's round-half-up, applied once to the whole sum, not per term; the
                // additions run in the C's order. Read the scores back signed: only --meth regions
                // score below zero, and the fork gives such a pair q = 0 (still pushed, so it counts
                // toward `sub` / `n_sub`).
                let (sc_i, sc_k) = ((v[i].1 >> 32) as u32 as i32, (v[ku].1 >> 32) as u32 as i32);
                let q = if sc_i < 0 || sc_k < 0 {
                    0
                } else {
                    (f64::from(sc_i) + f64::from(sc_k) + penalty + 0.499) as i64
                };
                // A pair whose insert penalty exceeds its alignment scores floors at 0 rather than
                // going negative, so it can still be selected if nothing better exists.
                let q = q.max(0) as u64;
                // Pack the pair: `py` remembers the two indices into `v` (partner in the high half,
                // current in the low half). `px` sorts by score in its high half, and its low half
                // is a *deterministic pseudo-random* tie-break: equal-scoring pairs are ordered by
                // `hash_64` of their index pair salted with the read-pair id. That is what makes
                // repeat-region pair choice reproducible across runs and thread counts. See
                // [`id_shift_c`] for why the salt must be computed in C's `int` width.
                let py = (k as u64) << 32 | i as u64;
                let px = (q << 32) | (hash_64(py ^ id_shift_c(id)) & 0xffff_ffff);
                u.push((px, py));
                k -= 1;
            }
        }
        // Record this placement as the newest member of its bucket, AFTER the look-back, so a
        // placement never pairs with itself and buckets only ever hold earlier positions.
        y[(v[i].1 & 3) as usize] = i as i64;
    }

    if u.is_empty() {
        return None;
    }
    // ---- step 3: pick the best candidate and measure how crowded the runner-up field is ----
    // `tie_margin` (bwa's `tmp`) is the score of the cheapest single event (one mismatch, or one
    // 1bp indel): the largest score gap that still counts as "essentially the same quality". Used
    // below to count how many runner-up pairs are effectively tied with `sub`.
    let mut tie_margin = opt.a + opt.b;
    tie_margin = tie_margin
        .max(opt.o_del + opt.e_del)
        .max(opt.o_ins + opt.e_ins);
    // The winning candidate pair (highest `q`, hash-tie-broken), the runner-up score and how many
    // others sit within `tie_margin` of it. `u` is non-empty by the check above.
    let (last, sub, n_sub) = best_pair(u, tie_margin);
    // Unpack the two `v` indices from `py`. `i` was the current placement, `k` the partner.
    let i = (last.1 >> 32) as usize;
    let k = (last.1 & 0xffff_ffff) as usize;
    // `v[..].1 & 1` is the read number, so each of the two placements files itself into its own
    // slot; the pair is guaranteed to have one from each read by the `^ 1` in the sweep.
    //
    // `y << 32 >> 34` on a u64 is C's idiom for "take bits 2..31 and shift them down by 2", i.e.
    // recover the region index `i` from `i << 2`: the left shift discards the score in the high
    // half, and the right shift of 34 = 32 + 2 undoes both it and the `<< 2`. It is a logical shift
    // in both languages here (`p->y` is `uint64_t`, ours is `u64`), so no sign extension.
    let mut z = [0usize; 2];
    z[(v[i].1 & 1) as usize] = (v[i].1 << 32 >> 34) as usize;
    z[(v[k].1 & 1) as usize] = (v[k].1 << 32 >> 34) as usize;
    // Winning pair's combined score, already floored at 0 when it was packed.
    let score = (last.0 >> 32) as i32;
    Some(PairResult {
        score,
        sub,
        n_sub,
        z,
    })
}

/// `mem_pair`'s pick from its non-empty candidate list `u`: the best `(px, py)`, the runner-up score
/// `sub` and `n_sub`, the count of other candidates within `tie_margin` of `sub`. bwa sorts `u`
/// ascending and reads `u[n-1]`, `u[n-2]` and every score below; nothing else sees the order, so
/// one scan for the maximum gives the same three answers.
///
/// Note `u` holds one entry per candidate *pairing*, so the runner-up is very often the same two
/// regions paired through a different intermediate, not a genuinely different locus. That is
/// intentional: the caller uses `sub` only to damp MAPQ.
fn best_pair(u: &[(u64, u64)], tie_margin: i32) -> ((u64, u64), i32, i32) {
    let best = (0..u.len()).max_by_key(|&j| u[j]).unwrap();
    let others = || {
        u.iter()
            .enumerate()
            .filter(move |&(j, _)| j != best)
            .map(|(_, c)| (c.0 >> 32) as i32)
    };
    // The sorted list's second-to-last entry, i.e. the top score among the others.
    let sub = others().max().unwrap_or(0);
    // bwa's loop over `u.a[0 .. u.n-2]`: every entry but the best, so `n_sub >= 1` once `n >= 2`.
    let n_sub = others().filter(|&q| sub - q <= tie_margin).count() as i32;
    (u[best], sub, n_sub)
}

/// Convert a score margin into a phred-scaled mapping quality. Port of the `raw_mapq` macro
/// (`bwamem_pair.cpp:350`).
///
/// `diff` is best-minus-suboptimal in alignment-score units and `a` is the match score (`opt.a`), so
/// `diff / a` is roughly "how many matched bases better the best hit is". 6.02 is `10 * log10(4)`:
/// each such base is a 4-fold likelihood ratio, which is 6.02 phred units. The trailing `+ 0.499`
/// is C's round-half-up before truncation, and it is applied inside the cast, so a `diff` of 0
/// yields 0 rather than rounding up.
///
/// f64 is required: the C macro's operands promote to `double` against the literal `6.02`.
/// No clamping happens here; callers clamp to [0, 60] themselves.
///
/// PARAMETERS: `diff` is a score margin (best minus sub-optimal) in alignment-score units and MAY be
/// negative, in which case the result is negative too. `a` is the match score `opt.a` (`-A`,
/// default 1) and must not be zero.
///
/// RETURNS an unclamped phred-scaled quality.
#[inline]
#[must_use]
fn raw_mapq(diff: i32, a: i32) -> i32 {
    (MAPQ_PER_MATCHED_BASE * f64::from(diff) / f64::from(a) + 0.499) as i32
}

/// Reference length consumed by a CIGAR (M/D ops). Port of `get_rlen` (`bwamem.cpp:1820-1829`).
///
/// CIGAR ops are packed bwa/BAM style: low 4 bits are the opcode, high 28 the length. The opcode
/// table is `MIDSH` = 0,1,2,3,4, so op 0 (M) and op 2 (D) are the two that advance along the
/// reference; I, S and H do not. Used only to find an alignment's rightmost reference base when
/// computing TLEN.
///
/// The C accumulates into an `int` and returns `int`; we widen to i64, which cannot differ for any
/// real read length.
///
/// PARAMETERS: `cigar` is one alignment's op list, bwa-packed (`len << 4 | op`), possibly empty.
///
/// RETURNS the number of reference bases the alignment spans (0 for an empty CIGAR).
#[must_use]
fn get_rlen(cigar: &[u32]) -> i64 {
    let mut rlen = 0i64;
    for &c in cigar {
        let op = c & 0xf;
        if op == 0 || op == 2 {
            rlen += i64::from(c >> 4);
        }
    }
    rlen
}

/// Append a CIGAR string, converting soft-clips to hard-clips when `hard_clip` says to. Port of
/// `add_cigar` (`bwamem.cpp:1579-1591`).
///
/// The C's condition is `p->n_cigar && which && !(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt`, i.e.
/// THREE things beyond being supplementary. It is passed in rather than derived from `which` here
/// because two of the three are invisible at this level: `-Y` lives in the options and `is_alt` on
/// the alignment, and the `MC:Z` caller needs the MATE's `is_alt` with the CURRENT record's
/// `which`. Deriving it from `which` alone was wrong for an ALT supplementary, which must keep its
/// SOFT clips: it is a parallel representation of the whole read, not a fragment of a split
/// alignment, so the primary's SEQ is not the rest of it.
///
/// PARAMETERS: `cigar` is bwa-packed (`len << 4 | op`). `which` is the alignment's index within its
/// read's list; index 0 is the primary and everything after it is supplementary. An EMPTY `cigar`
/// prints `*`, which is not the same as "unmapped": `mem_aln2sam` clears the CIGAR when it copies a
/// mate's coordinates onto an unmapped read, so the record has a position but no alignment.
/// `out` is the record buffer being built; this function only appends, never inspects it.
///
/// WHY hard-clip: SAM requires that only one record per read carry the full SEQ. bwa keeps SEQ on
/// the primary (soft-clipped) and hard-clips the supplementaries, whose SEQ is then trimmed to
/// match by the caller.
pub(crate) fn add_cigar(cigar: &[u32], hard_clip: bool, out: &mut Vec<u8>) {
    if cigar.is_empty() {
        out.push(b'*');
        return;
    }
    // bwa's `"MIDSH"[c]`. Op 3 is bwa's internal soft-clip marker in this table (SAM's own numbering
    // puts N at 3), so both 3 and 4 are clip ops here and both become H on a supplementary.
    const OPS: [u8; 5] = *b"MIDSH";
    for &c in cigar {
        let mut op = (c & 0xf) as usize;
        // The C writes `c = which? 4 : 3`, i.e. it also rewrites 4 back to 3 on the primary. Our
        // form only rewrites upward, which agrees on the primary as long as no clip op reaching
        // here is already 4. That holds: `cigar.rs` builds clips exclusively as `len << 4 | 3`
        // (the two `insert`/`push` sites around `crates/bwa-mem/src/cigar.rs:416`), so op 4 never
        // enters a CIGAR before this point.
        if (op == 3 || op == 4) && hard_clip {
            op = 4; // hard-clip on supplementary
        }
        crate::emit::push_int(out, i64::from(c >> 4));
        out.push(OPS[op]);
    }
}

/// Emit one SAM record for read `which` of `list`, with optional mate `m`. Port of `mem_aln2sam`
/// (`bwamem.cpp:1592-1727`), the SE/PE subset, non-ALT throughout, with no `pa:f` (which needs ALT
/// contigs). `seq` is nt4-encoded in sequencing
/// orientation.
///
/// PARAMETERS: `bns` resolves `rid` to a contig name for RNAME/RNEXT/SA:Z. `name` becomes QNAME
/// verbatim. `seq` is the read's nt4 codes (0..=4) in sequencing orientation, whose length also sets
/// the SEQ window before hard-clip trimming. `qual` is the raw ASCII quality string of the same
/// length, or `None`/empty to print `*`. `list` is *all* of this read's emitted alignments and
/// `which` selects the one to print, `which == 0` meaning primary (index 0) and anything else
/// supplementary (which drives hard-clipping). The whole list is needed, not just the selected
/// element, because the SA:Z tag enumerates the read's other primary hits. `m` is the mate's chosen
/// alignment, or `None` for single-end; passing `Some` is what sets FLAG 0x1 and populates
/// RNEXT/PNEXT/TLEN. `comment` is the FASTQ comment for `-C`. `out` receives exactly one
/// newline-terminated record, appended.
///
/// The function works on CLONES of `p` and `m` (bwa's `ptmp`/`mtmp`) because it mutates both: the
/// mate-coordinate copy below rewrites `rid`/`pos`/`is_rev` and clears CIGARs. Those edits must not
/// escape, since the same `MemAln` is emitted again for the other end of the pair.
#[allow(clippy::too_many_arguments)]
fn mem_aln2sam(
    bns: &BntSeq,
    name: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    comment: Option<&str>,
    list: &[MemAln],
    which: usize,
    m: Option<&MemAln>,
    // `-Y` (MEM_F_SOFTCLIP): keep soft clips on supplementary records instead of hard-clipping
    // them. Passed in rather than read from `MemOpt` because this function takes no options struct
    // by design; it is a formatter, and every other decision was already made by the caller.
    softclip: bool,
    // `-V` (MEM_F_REF_HDR): close the record with the contig's FASTA comment. Passed in for the same
    // reason as `softclip`.
    ref_hdr: bool,
    // `--meth`: the read's chemistry and what XM:Z needs; `None` for every other run.
    meth: Option<&MethEmit>,
    out: &mut Vec<u8>,
) {
    // `p`: the alignment being printed, and `m` the mate's. bwa works on private copies
    // (`ptmp`/`mtmp`) precisely because step 1 rewrites their `rid`/`pos`/`is_rev` and clears their
    // CIGARs, and this used to clone both. A `MemAln` owns a `Vec<u32>` CIGAR, an `md` String and
    // an `xa` String, so those two clones were up to six allocations PER RECORD, paid only to get
    // four mutable scalars and a "CIGAR is now empty" bit. The borrow below plus the shadow
    // variables carry exactly the same mutations without touching the heap; every read of a field
    // step 1 can rewrite goes through a shadow, and every field it cannot (`mapq`, `nm`, `md`,
    // `score`, `sub`, `alt_sc`, `is_alt`, `xa`) is read straight off the borrow.
    let p = &list[which];
    // The four mutable scalars of `ptmp`, plus `p_cigar`, which is `&p.cigar` until the mate-
    // coordinate copy empties it. Emptying a slice reference is bwa's `p->n_cigar = 0`: the storage
    // is untouched, only this record stops seeing it.
    let mut p_flag = p.flag;
    let mut p_rid = p.rid;
    let mut p_pos = p.pos;
    let mut p_is_rev = p.is_rev;
    let mut p_cigar: &[u32] = &p.cigar;
    // Same for `mtmp`. The values are meaningless when `m` is `None`; every use below is guarded by
    // `m.is_some()` or by an `if let`, exactly as the `Option` was.
    let (mut m_rid, mut m_pos, mut m_is_rev, mut m_cigar): (i32, i64, bool, &[u32]) = match m {
        Some(x) => (x.rid, x.pos, x.is_rev, &x.cigar),
        None => (0, 0, false, &[]),
    };

    // ---- step 1: FLAG bits and the unmapped-read mate coordinate copy ----
    // An unmapped read is placed AT ITS MATE'S COORDINATE (with an
    // empty CIGAR) rather than at `*`/0, which is what lets samtools sort keep the pair together.
    // Both directions are done, and the order matters: `p.rid < 0` is tested before `p` is possibly
    // overwritten, so a pair with both ends unmapped copies nothing.
    if m.is_some() {
        p_flag |= 0x1;
    }
    // 0x4 / 0x8 are decided BEFORE the coordinate copy below, so a read that ends up printing its
    // mate's RNAME/POS still carries the unmapped bit. That is deliberate and load-bearing.
    if p_rid < 0 {
        p_flag |= 0x4;
    }
    if m.is_some() && m_rid < 0 {
        p_flag |= 0x8;
    }
    if p_rid < 0 && m.is_some() && m_rid >= 0 {
        p_rid = m_rid;
        p_pos = m_pos;
        p_is_rev = m_is_rev;
        p_cigar = &[];
    }
    if p_rid >= 0 && m.is_some() && m_rid < 0 {
        m_rid = p_rid;
        m_pos = p_pos;
        m_is_rev = p_is_rev;
        m_cigar = &[];
    }
    if p_is_rev {
        p_flag |= 0x10;
    }
    if m.is_some() && m_is_rev {
        p_flag |= 0x20;
    }
    // `--meth`: a mapped record's conversion strand (XG:Z:CT when `top`), and the QC rules that
    // may change its FLAG and MAPQ before they are printed.
    let mut mapq = p.mapq;
    let top = meth.filter(|_| p.rid >= 0).map(|e| e.ot ^ p.is_rev);
    if let (Some(e), Some(top)) = (meth, top) {
        e.qc(top, &p.cigar, seq.len(), &mut p_flag, &mut mapq);
    }

    // ---- step 2: QNAME, FLAG ----
    out.extend_from_slice(name.as_bytes());
    out.push(b'\t');
    // Bit 0x10000 is bwa-internal, not SAM: `mem_reg2sam` sets it instead of 0x800 under `-M`
    // (MEM_F_NO_MULTI) to mark "supplementary, but report it as secondary". Here it is masked off
    // and translated into the real SAM 0x100 secondary bit.
    // `flag`: the value actually printed in column 2, i.e. `p.flag` with the internal bit translated.
    let flag = (p_flag & 0xffff) | if p_flag & 0x10000 != 0 { 0x100 } else { 0 };
    crate::emit::push_int(out, i64::from(flag));
    out.push(b'\t');

    // ---- step 3: RNAME, POS, MAPQ, CIGAR ----
    // `pos` is 0-based internally and SAM is 1-based, hence `+ 1`. The
    // unmapped branch writes the four fields at once as `*\t0\t0\t*` (bwa's `kputsn(..., 7, str)`).
    if p_rid >= 0 {
        out.extend_from_slice(bns.contigs[p_rid as usize].name.as_bytes());
        out.push(b'\t');
        crate::emit::push_int(out, p_pos + 1);
        out.push(b'\t');
        crate::emit::push_int(out, i64::from(mapq));
        out.push(b'\t');
        add_cigar(p_cigar, which != 0 && !softclip && !p.is_alt, out);
    } else {
        out.extend_from_slice(b"*\t0\t0\t*");
    }
    out.push(b'\t');

    // ---- step 4: RNEXT, PNEXT, TLEN ----
    match m {
        Some(_) if m_rid >= 0 => {
            if p_rid == m_rid {
                out.push(b'=');
            } else {
                out.extend_from_slice(bns.contigs[m_rid as usize].name.as_bytes());
            }
            out.push(b'\t');
            crate::emit::push_int(out, m_pos + 1);
            out.push(b'\t');
            // TLEN. `p0`/`p1` are each record's *outermost* base: the leftmost for a forward
            // alignment, the rightmost (`pos + rlen - 1`) for a reverse one. TLEN is their signed
            // separation, and the `sign` term widens it by one so the reported length counts both
            // endpoints inclusively. The leading unary minus is bwa's sign convention: the
            // leftmost-starting mate gets the positive value, and this expression is evaluated
            // independently for each record, so the two lines come out as exact negations.
            //
            // Either CIGAR being empty means one end is only borrowing the other's coordinate (see
            // the mate-copy above), so there is no real template and TLEN is 0.
            if p_rid == m_rid {
                // `p0`/`p1`: this record's and the mate's outermost reference base, 0-based.
                let p0 = p_pos + if p_is_rev { get_rlen(p_cigar) - 1 } else { 0 };
                let p1 = m_pos + if m_is_rev { get_rlen(m_cigar) - 1 } else { 0 };
                if m_cigar.is_empty() || p_cigar.is_empty() {
                    out.push(b'0');
                } else {
                    // +1, -1 or 0: widens the span by one so TLEN counts both endpoints inclusively.
                    let sign = match p0.cmp(&p1) {
                        std::cmp::Ordering::Greater => 1,
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                    };
                    crate::emit::push_int(out, -(p0 - p1 + sign));
                }
            } else {
                out.push(b'0');
            }
        }
        _ => out.extend_from_slice(b"*\t0\t0"),
    }
    out.push(b'\t');

    // ---- step 5: SEQ, QUAL ----
    // Secondary records (0x100) omit both, per SAM convention and to keep file size
    // sane; the read's bases are already on its primary record.
    // Where SEQ lands in `out`, for XM:Z. Empty when SEQ is `*`.
    let mut seq_span = 0..0;
    if p_flag & 0x100 != 0 {
        out.extend_from_slice(b"*\t*");
    } else {
        // `qb`/`qe`: the half-open slice of `seq` (and of `qual`) this record actually prints, in
        // sequencing orientation. Starts as the whole read and is narrowed below only when this
        // record's clips were rendered as hard clips.
        let (mut qb, mut qe) = (0usize, seq.len());
        // Hard-clip trimming for supplementary alignments: `add_cigar` turned this record's clips
        // into H, and H means the bases are ABSENT from SEQ, so the emitted range must be narrowed
        // to match or the record is self-inconsistent. This condition is EXACTLY `add_cigar`'s
        // (`bwamem.cpp:1655` repeats the same four terms), and the two must be kept in step: trim
        // without hard-clipping and the record claims bases it does not carry.
        if !p_cigar.is_empty() && which != 0 && !softclip && !p.is_alt {
            // Opcodes of the CIGAR's two outermost ops; 3 (S) or 4 (H) means that end is clipped.
            let first = p_cigar[0] & 0xf;
            let last = p_cigar[p_cigar.len() - 1] & 0xf;
            // `qb`/`qe` index `seq` in SEQUENCING orientation, but the CIGAR is in REFERENCE
            // orientation, so on a reverse-strand alignment the leading CIGAR clip corresponds to
            // the trailing end of `seq` and vice versa. That is the only difference between the two
            // branches below, and getting it backwards trims the wrong end of the read.
            if !p_is_rev {
                if first == 4 || first == 3 {
                    qb += (p_cigar[0] >> 4) as usize;
                }
                if last == 4 || last == 3 {
                    qe -= (p_cigar[p_cigar.len() - 1] >> 4) as usize;
                }
            } else {
                if first == 4 || first == 3 {
                    qe -= (p_cigar[0] >> 4) as usize;
                }
                if last == 4 || last == 3 {
                    qb += (p_cigar[p_cigar.len() - 1] >> 4) as usize;
                }
            }
        }
        // SAM stores SEQ on the reference forward strand, so a reverse-strand alignment emits the
        // reverse complement of the read and reverses QUAL with it. `FWD_BASE`/`REV_BASE` are bwa's `F`/`R`,
        // "ACGTN" and "TGCAN"; indexing `REV_BASE` in reverse order performs the complement,
        // since `REV_BASE[c]` is the complement base of `FWD_BASE[c]`. `c.min(4)` guards nt4 codes above 4 (bwa indexes unguarded).
        // Written a block at a time rather than a base at a time: see `crate::emit`, and the x86
        // profile that motivated it (`sam_emit` is 5.15x slower per pair there than on an M4, the
        // worst-scaling stage we have). Same bytes, same alphabets, same order.
        seq_span = out.len()..out.len() + (qe - qb);
        if !p_is_rev {
            crate::emit::push_seq_fwd(out, &seq[qb..qe]);
            out.push(b'\t');
            match qual {
                Some(qv) if !qv.is_empty() => out.extend_from_slice(&qv[qb..qe]),
                _ => out.push(b'*'),
            }
        } else {
            crate::emit::push_seq_rev(out, &seq[qb..qe]);
            out.push(b'\t');
            match qual {
                Some(qv) if !qv.is_empty() => crate::emit::push_qual_rev(out, &qv[qb..qe]),
                _ => out.push(b'*'),
            }
        }
    }

    // ---- step 6: optional tags ----
    // In bwa's exact emission order: NM, MD, MC, AS, XS, RG, SA, (pa), XA, comment.
    // SAM does not prescribe an order, but byte-identity does, so nothing here may be reordered.
    //
    // NM/MD are gated on a non-empty CIGAR, so a read placed at its mate's coordinate gets neither.
    if !p_cigar.is_empty() {
        out.extend_from_slice(b"\tNM:i:");
        crate::emit::push_int(out, i64::from(p.nm));
        out.extend_from_slice(b"\tMD:Z:");
        out.extend_from_slice(p.md.as_bytes());
    }
    // MC:Z is the MATE's CIGAR, but note it is rendered with THIS record's `which`
    // (`bwamem.cpp:1689` passes `which` straight through), so on a supplementary record the mate's
    // clips are printed as H even though the mate's own record prints them as S. That is bwa's
    // behaviour, not a transcription slip.
    if let Some(mate) = m {
        if !m_cigar.is_empty() {
            out.extend_from_slice(b"\tMC:Z:");
            add_cigar(m_cigar, which != 0 && !softclip && !mate.is_alt, out);
        }
        if meth.is_some() {
            out.extend_from_slice(b"\tMQ:i:");
            crate::emit::push_int(out, i64::from(mate.mapq));
        }
    }
    // AS/XS are gated on `>= 0`, not on `> 0`, which is why an unmapped record (built from a zeroed
    // `mem_aln_t`) still carries `AS:i:0 XS:i:0`. `sub` is set to -1 for secondaries to suppress XS.
    if p.score >= 0 {
        out.extend_from_slice(b"\tAS:i:");
        crate::emit::push_int(out, i64::from(p.score));
    }
    if p.sub >= 0 {
        out.extend_from_slice(b"\tXS:i:");
        crate::emit::push_int(out, i64::from(p.sub));
    }
    // `-R`: bwa emits RG:Z here, between XS and SA:Z.
    bwa_core::rg::append_rg_tag(out);
    // SA:Z (chimeric): the read's OTHER non-secondary hits, i.e. the other pieces of a split
    // alignment. Emitted only on non-secondary records, and only when at least one other such hit
    // exists (bwa's two-pass `for` + `if (i < n)`, which the `has_other` scan reproduces so the tag
    // prefix is not written for an empty list).
    //
    // Each entry is `rname,pos,strand,CIGAR,mapq,NM;` INCLUDING the trailing semicolon on the last
    // one. The CIGAR here is rendered with `which = 0`, i.e. always soft-clipped, regardless of
    // whether the referenced hit is itself a supplementary.
    if p_flag & 0x100 == 0 {
        // True when some OTHER non-secondary record exists for this read, i.e. the SA:Z tag will
        // have at least one entry. Scanned first so the `SA:Z:` prefix is never written empty.
        let has_other = list
            .iter()
            .enumerate()
            .any(|(i, r)| i != which && r.flag & 0x100 == 0);
        if has_other {
            out.extend_from_slice(b"\tSA:Z:");
            for (i, r) in list.iter().enumerate() {
                if i == which || r.flag & 0x100 != 0 {
                    continue;
                }
                out.extend_from_slice(bns.contigs[r.rid as usize].name.as_bytes());
                out.push(b',');
                crate::emit::push_int(out, r.pos + 1);
                out.push(b',');
                out.push(if r.is_rev { b'-' } else { b'+' });
                out.push(b',');
                add_cigar(&r.cigar, false, out);
                out.push(b',');
                crate::emit::push_int(out, i64::from(r.mapq));
                out.push(b',');
                crate::emit::push_int(out, i64::from(r.nm));
                out.push(b';');
            }
        }
    }
    // `pa:f` (`bwamem.cpp:1714`): emitted for a non-secondary record whenever `alt_sc` was set,
    // independently of whether an `SA:Z` was printed, and always before `XA:Z`. `alt_sc` is 0
    // unless the index has ALT contigs.
    if p_flag & 0x100 == 0 && p.alt_sc > 0 {
        out.extend_from_slice(b"\tpa:f:");
        out.extend_from_slice(crate::cigar::format_pa(p.score, p.alt_sc).as_bytes());
    }
    // XA:Z (alternate hits), after SA/pa per mem_aln2sam.
    if let Some(xa) = &p.xa {
        out.extend_from_slice(b"\tXA:Z:");
        out.extend_from_slice(xa.as_bytes());
    }
    if let Some(e) = meth {
        if p.hn >= 0 {
            out.extend_from_slice(b"\tHN:i:");
            crate::emit::push_int(out, i64::from(p.hn));
        }
        let emitted_seq = out[seq_span].to_vec();
        let hard = which != 0 && !softclip && !p.is_alt;
        e.push_tags(
            bns,
            out,
            top,
            p.rid,
            p.pos,
            &p.cigar,
            hard,
            &emitted_seq,
            comment,
        );
        out.push(b'\n');
        return;
    }
    // `-C`: the FASTQ comment closes the record, after every tag.
    bwa_core::rg::append_comment(out, comment);
    // `-V`, after even the comment (`bwamem.cpp:1721`). Tested on the rewritten `rid`, so an unmapped
    // read placed at its mate carries the mate's contig.
    if ref_hdr && p_rid >= 0 {
        crate::emit::push_xr(out, bns.contigs[p_rid as usize].comment());
    }
    out.push(b'\n');
}

/// Emit SAM for one read's regions (the `no_pairing` path). Port of `mem_reg2sam`
/// (`bwamem.cpp:1521-1577`), non-ALT, without `-a` (MEM_F_ALL) or `-M`/`-5`.
///
/// WHY it exists separately from the paired emitter: when pairing fails or is disabled, each end is
/// written as if single-end (possibly as several records, primary plus supplementaries), with only
/// the mate's coordinates borrowed for RNEXT/PNEXT. `extra_flag` is how the caller stamps the
/// paired-end bits (0x1, 0x40/0x80, and 0x2 when the two top hits happen to look proper) onto every
/// record this produces.
///
/// PARAMETERS: `fm`/`bns` are the reference, needed by `reg2aln` (CIGAR, NM, MD) and `mem_gen_alt`
/// (XA:Z). `opt` supplies `t` (`-T`, the minimum reportable score) and `drop_ratio`. `name`, `seq`,
/// `qual` and `comment` are the read's FASTQ fields, passed through unchanged to [`mem_aln2sam`].
/// `a` is the read's full region vector; this function decides which regions become
/// records. `extra_flag` is OR-ed into every record's FLAG (0x1 plus 0x40/0x80, and 0x2 when the
/// caller inferred a proper pair). `m` is the mate's chosen alignment for RNEXT/PNEXT/TLEN, or
/// `None`. `out` receives one or more complete records, appended.
///
/// `meth` is the `--meth` record context (`None` otherwise); `opt` must then carry the read's
/// chemistry matrix ([`crate::meth::read_opt`]).
#[allow(clippy::too_many_arguments)]
pub fn mem_reg2sam(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    name: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    comment: Option<&str>,
    a: &[MemAlnReg],
    extra_flag: u32,
    m: Option<&MemAln>,
    meth: Option<&MethEmit>,
    out: &mut Vec<u8>,
) {
    // `-Y` and `-V`, read once here and threaded into every `mem_aln2sam` call below.
    let softclip = opt.flag & bwa_core::opt::flags::SOFTCLIP != 0;
    let ref_hdr = opt.flag & bwa_core::opt::flags::REF_HDR != 0;
    // Shadowed hits are not emitted here; they surface as the primary's XA:Z, which this path used
    // to omit entirely (~1% of PE records carried no XA where bwa emits one).
    // XA is generated over the WHOLE region vector and indexed by `k`, so a region that is dropped
    // below still contributes to the surviving primary's XA:Z list. That is the point: shadowed
    // hits are reported as alternates, not as records.
    // `xa[k]`: the XA:Z string region `k` should carry, or `None`. Index-parallel with `a`.
    //
    // `-a` (MEM_F_ALL) suppresses XA entirely: the C guards the call with
    // `if (!(opt->flag & MEM_F_ALL)) XA = mem_gen_alt(...)` and leaves `XA` null otherwise
    // (`bwamem.cpp:1521`). The two are alternatives: `-a` emits every shadowed region as its own
    // record, so repeating those placements inside a tag would duplicate them. Generating it
    // unconditionally was a byte-parity bug on `-a`, which the option harness never saw because it
    // only ever ran `-a` single-end, where this path is not reached with shadowed regions present.
    // `-a` (MEM_F_ALL): emit every alignment, including the shadowed ones, and emit no XA.
    let all = opt.flag & bwa_core::opt::flags::ALL != 0;
    let xa = if all {
        crate::alt::XaTags::default()
    } else {
        mem_gen_alt(fm, bns, opt, a, seq.len() as i32, seq)
    };
    // `--meth` only: HN:i per region, absent under `-a` like XA.
    let hn = if meth.is_some() && !all {
        crate::alt::hit_counts(opt, a)
    } else {
        vec![-1; a.len()]
    };
    // ---- step 1: turn the surviving regions into records ----
    // `emitted` is bwa's `aa`, the records to emit; `n_emitted` is its `l`, the count accepted so
    // far, and is what makes "everything after the first" supplementary. It is a separate counter
    // from `k` because `k` walks skipped regions too.
    let mut emitted: Vec<MemAln> = Vec::new();
    let mut n_emitted = 0;
    for (k, p) in a.iter().enumerate() {
        // `-T` (`opt.t`): minimum score to report at all.
        if p.score < opt.t {
            continue;
        }
        // `bwamem.cpp:1541`. A secondary is dropped unless `-a` asked for every alignment, and an
        // ALT secondary is dropped even then: under `-a` the ALT hits are already reported as the
        // supplementary record the paired branch emits, so repeating them here would duplicate
        // them.
        if p.secondary >= 0 && (p.is_alt || !all) {
            continue;
        }
        // `bwamem.cpp:1543`. Drop a secondary that scores too far below the primary shadowing it.
        //
        // `p.secondary < i32::MAX` is NOT redundant, and leaving it out is a panic rather than a
        // wrong byte: `mem_mark_primary_se`'s ALT branch parks ALT hits at `secondary = INT_MAX`
        // as a sentinel, and indexing `a[INT_MAX]` would be out of bounds. The C guards the same
        // way and for the same reason.
        //
        // f32 throughout: `opt->drop_ratio` is a C `float`.
        if p.secondary >= 0
            && p.secondary < i32::MAX
            && (p.score as f32) < a[p.secondary as usize].score as f32 * opt.drop_ratio
        {
            continue;
        }
        // The printable form of region `p`: CIGAR, POS, MAPQ, NM and MD, all derived here.
        let mut aln = reg2aln(fm, bns, opt, seq.len() as i32, seq, p);
        aln.xa = xa.get(k);
        aln.hn = hn[k];
        aln.flag |= extra_flag;
        // Suppress XS on a secondary (`sub < 0` gates the tag in `mem_aln2sam`). Unreachable here
        // for the same reason as the drop-ratio test above.
        if p.secondary >= 0 {
            aln.sub = -1;
        }
        // Everything after the first accepted record is a supplementary. Under `-M`
        // (MEM_F_NO_MULTI) bwa sets its INTERNAL 0x10000 instead of 0x800 (`bwamem.cpp:1552`),
        // which `mem_aln2sam` then prints as SAM's 0x100 secondary bit: `-M` exists so that older
        // tools, which understand secondary but not supplementary, still see one primary per read.
        // This branch used to write 0x800 unconditionally, its own comment describing the `-M` case
        // it did not implement, so every supplementary this path emitted came out supplementary
        // under `-M` too.
        if n_emitted > 0 && p.secondary < 0 {
            aln.flag |= if opt.flag & bwa_core::opt::flags::NO_MULTI != 0 {
                0x10000 // bwa-internal; printed as 0x100
            } else {
                0x800 // supplementary
            };
        }
        // A supplementary must not claim higher confidence than the primary it came from, so its
        // MAPQ is capped at the first record's. Only lowered, never raised.
        //
        // `-q` (MEM_F_KEEP_SUPP_MAPQ), which `-5` also sets, suppresses the cap: for a chimeric
        // library each segment is its own real placement and capping it hides that. The C's guard
        // is the `#if V17` arm at `bwamem.cpp:1555`, and it was missing here, so `-q` and `-5`
        // capped anyway on this path while the single-end path honoured them.
        if opt.flag & bwa_core::opt::flags::KEEP_SUPP_MAPQ == 0
            && n_emitted > 0
            && !p.is_alt
            && aln.mapq > emitted[0].mapq
        {
            aln.mapq = emitted[0].mapq;
        }
        emitted.push(aln);
        n_emitted += 1;
    }
    // ---- step 2: write them, or one unmapped record if none survived ----
    // No region cleared `-T`: emit one unmapped record rather than nothing, so every input read
    // still produces exactly one line here. bwa builds it via `mem_reg2aln(..., 0)`, a zeroed
    // `mem_aln_t`, which is why `MemAln::unmapped` carries score 0 / sub 0 and thus `AS:i:0 XS:i:0`.
    if emitted.is_empty() {
        // A zeroed alignment: rid -1, no CIGAR, MAPQ 0, score 0. Printed as the read's only record.
        let mut unmapped = MemAln::unmapped();
        unmapped.flag |= extra_flag;
        mem_aln2sam(
            bns,
            name,
            seq,
            qual,
            comment,
            &[unmapped],
            0,
            m,
            softclip,
            ref_hdr,
            meth,
            out,
        );
    } else {
        for k in 0..emitted.len() {
            mem_aln2sam(
                bns, name, seq, qual, comment, &emitted, k, m, softclip, ref_hdr, meth, out,
            );
        }
    }
}

thread_local! {
    /// One SAM output buffer per thread, reused across every pair that thread emits.
    ///
    /// `mem_sam_pe` used to build each end's records in its own fresh `Vec`, so a pair paid two
    /// allocations plus the reallocations of growing from empty to a few hundred bytes, and then
    /// issued two writes. The buffer is taken out here and put back at the end; an early return
    /// between the two simply drops it and costs one allocation on the next pair, which makes this
    /// a performance path and never a correctness one.
    ///
    /// Per-THREAD and not per-call, so it cannot be shared between the rayon workers that run pairs
    /// concurrently. Both ends of a pair still go in before anything is written.
    static SCRATCH_SAM: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Full paired-end SAM for one read pair. Port of `mem_sam_pe` (`bwamem_pair.cpp:353-546`),
/// non-ALT, without `-a`/`-5`.
///
/// SHAPE: four stages, in this order and no other. (1) mate rescue, which can ADD regions;
/// (2) primary marking, which must therefore run after rescue; (3) proper pairing, which either
/// emits two records and returns or falls through; (4) the `no_pairing` fallback, which emits each
/// end independently. bwa reaches stage 4 by `goto no_pairing` from three different places, which
/// here is expressed as falling off the end of the `if` block.
///
/// PARAMETERS: `fm`/`bns` are the reference (rescue windows, contig names, `reg2aln`, `mem_gen_alt`).
/// `opt` supplies every tunable this file reads: `-U` `pen_unpaired`, `-m` `max_matesw`, `-T` `t`,
/// `-A` `a` and the gap penalties, plus `flag` for `-P`. `pes` is the batch's insert-size estimate,
/// which must be the same one [`batch_mate_rescue`] was given. `names`, `seqs` (nt4), `quals` and
/// `comments` are the two ends' FASTQ fields, index 0 = read 1, index 1 = read 2; `seqs[i]`,
/// `quals[i]` must be the same length. `a0`/`a1` are the two reads' dedup'd, score-sorted region
/// vectors, taken by `&mut`
/// because every stage mutates them (rescue inserts, primary marking rewrites `secondary`, pairing
/// rewrites `sub`/`secondary`/`secondary_all`). `id` is the global 0-based pair index, used as the
/// hash salt for tie-breaks; it must be the same value bwa would use, so batching must not
/// renumber. `rescue_done` is a port-only flag: true when the caller already ran
/// [`batch_mate_rescue`] over the whole batch, in which case stage 1 is skipped here.
///
/// Records are written to `w` read 1 first, then read 2. The whole pair is built in
/// [`SCRATCH_SAM`], this thread's reusable buffer, and written in ONE call, so a pair's lines
/// cannot interleave with another thread's.
///
/// RETURNS whatever `w` reported; the only failure mode is the write itself.
///
/// UNVERIFIED: the C aborts with `err_fatal` when the two ends' names differ ("paired reads have
/// different names"). No equivalent check was found anywhere in this workspace, so mismatched pairs
/// appear to be emitted silently rather than rejected. That affects diagnostics only, not the
/// output of well-formed input.
#[allow(clippy::too_many_arguments)]
pub fn mem_sam_pe<W: Write>(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    id: u64,
    names: &[&str; 2],
    seqs: &[&[u8]; 2],
    quals: &[Option<&[u8]>; 2],
    comments: &[Option<&str>; 2],
    a0: &mut Vec<MemAlnReg>,
    a1: &mut Vec<MemAlnReg>,
    rescue_done: bool,
    w: &mut W,
) -> io::Result<()> {
    // Mate rescue (mem_matesw), before primary marking. Snapshot each read's near-best regions as
    // anchors, then SW-rescue the other read's mate in any missing insert-consistent orientation.
    // On concordant pairs every orientation is skipped, so this is a no-op. Port of the non-
    // MATE_SORT rescue block in mem_sam_pe (bwamem_pair.cpp lines 378-414). Skipped when the caller
    // already ran it batched across the whole pair batch ([`batch_mate_rescue`]).
    // ---- stage 1: mate rescue ----
    // `--meth`: each end's own chemistry matrix (read 1 C->T, read 2 G->A) and record context.
    // Plain runs borrow `opt` and carry no context.
    let ropt = [
        crate::meth::read_opt(opt, true),
        crate::meth::read_opt(opt, false),
    ];
    let emit = opt.meth.as_ref().map_or([None, None], |meth| {
        [true, false].map(|ot| Some(MethEmit { fm, meth, ot }))
    });
    if !rescue_done {
        // `-U`: any region within this of the best is a plausible anchor. `-m`: anchor cap.
        let pen = opt.pen_unpaired;
        let anchor_cap = opt.max_matesw.max(0) as usize;
        // Each side is snapshotted on its own: bwa gates only on MEM_F_NO_RESCUE, never on both
        // reads having regions. A read with no regions contributes no anchors, yet is still the
        // *rescue target* of the other read's anchors -- which is exactly how an unmapped mate gets
        // rescued. Requiring both sides non-empty leaves it unmapped.
        let near_best = |regs: &[MemAlnReg]| -> Vec<MemAlnReg> {
            let Some(best) = regs.first().map(|r| r.score) else {
                return Vec::new();
            };
            regs.iter()
                .filter(|r| r.score >= best - pen)
                .cloned()
                .collect()
        };
        // `b0`/`b1`: each end's anchors (bwa's `b[0]`/`b[1]`), snapshotted BEFORE any rescue so a
        // region added by rescue can never itself become an anchor.
        let b0 = near_best(a0);
        let b1 = near_best(a1);
        for anchor in b0.iter().take(anchor_cap) {
            mem_matesw(fm, bns, opt, pes, anchor, seqs[1], a1, false);
        }
        for anchor in b1.iter().take(anchor_cap) {
            mem_matesw(fm, bns, opt, pes, anchor, seqs[0], a0, true);
        }
    }

    // ---- stage 2: primary marking ----
    // Runs AFTER rescue, because rescue can add regions that change which hit is
    // primary. The hash salt is `id<<1|end`, giving the two ends distinct but reproducible
    // tie-breaks; note this is a plain u64 shift, unlike `mem_pair`'s `id<<8` (see [`id_shift_c`]),
    // because the C passes `id` here as `int64_t` rather than truncating it to `int`.
    //
    // `n_pri` is the count of non-ALT ("primary") regions, which form a prefix of the vector. In
    // this non-ALT build it equals the vector length, but the code keeps the distinction so the
    // `n_pri < a.len()` ALT branches stay meaningful.
    let n_pri0 = mem_mark_primary_se(opt, a0, id << 1) as usize;
    let n_pri1 = mem_mark_primary_se(opt, a1, (id << 1) | 1) as usize;
    // `-5`: both ends, after the marking and before the `-P` test, exactly where
    // `bwamem_pair.cpp:419-423` puts it. `n_pri0`/`n_pri1` are computed FIRST and stay valid: the
    // reorder swaps two slots that were both non-secondary, so it cannot change how many there are.
    if opt.flag & bwa_core::opt::flags::PRIMARY5 != 0 {
        crate::primary::mem_reorder_primary5(opt.t, a0);
        crate::primary::mem_reorder_primary5(opt.t, a1);
    }
    // 0x1 (paired in sequencing) is stamped on every record either path emits.
    let extra_flag: u32 = 1;

    // ---- stage 3: proper pairing ----
    // `-P` (MEM_F_NOPAIRING) skips the whole block, exactly as bwa's
    // `if (!(opt->flag & MEM_F_NOPAIRING))` does, so both ends fall through to the no-pairing
    // emitter below. Mate rescue is unaffected: only `-S` suppresses that.
    if n_pri0 > 0 && n_pri1 > 0 && opt.flag & bwa_core::opt::flags::NOPAIRING == 0 {
        // `pr`: the best proper pair and its MAPQ inputs, or `None` if no candidate pair existed.
        // Computed in an inner scope so the two shared borrows of `a0`/`a1` end before the mutable
        // reborrows below.
        let pr = {
            let a: [&[MemAlnReg]; 2] = [&a0[..n_pri0], &a1[..n_pri1]];
            mem_pair(bns, opt, pes, &a, id)
        };
        // `.filter(score > 0)` is bwa's `(o = mem_pair(...)) > 0`, and it is not a nicety. A pair
        // whose best score is exactly ZERO is not a proper pair to the C, which falls straight
        // through to `no_pairing` -- where the insert window, not the pair score, decides the 0x2
        // bit. Testing only for `Some` took the paired branch instead, and that branch never
        // evaluates the `no_pairing` window test, so the pair lost its proper-pair bit.
        //
        // Zero is reachable, and reachable on ordinary data: the insert-size term is
        // `0.721 * ln(2 * erfc(|z| / sqrt(2))) * a`, and with a standard deviation of ZERO the
        // z-score is `0/0` for an insert exactly at the mean. That is NaN, `NaN as i64` is 0 here
        // and `(int)NaN` is 0 in the C on this machine, so both floor the pair at 0 -- and then only
        // the C drew the right conclusion from it. `-I 400,0` reproduces it on any read set, and so
        // does a library with a fixed insert size, where the inferred distribution has no spread at
        // all: 600 of 600 records differed on a simulated fixed-insert set before this.
        if let Some(pr) = pr.filter(|p| p.score > 0) {
            // Multiple sufficiently-good primary hits on either end -> fall back. bwa's rationale
            // (its own TODO) is that a split alignment would be mangled by forcing a single paired
            // record, so it prefers the per-end emitter. The scan starts at j = 1 because a[0] is
            // the best hit by construction and is not its own competitor.
            let is_multi = |a: &[MemAlnReg], n_pri: usize| -> bool {
                (1..n_pri).any(|j| a[j].secondary < 0 && a[j].score >= opt.t)
            };
            if !is_multi(a0, n_pri0) && !is_multi(a1, n_pri1) {
                // The score the pair would get if the two ends were reported independently: both
                // best scores, minus the `-U` unpaired penalty. This is the bar the proper pair has
                // to clear, both for MAPQ (as a floor on the sub-optimal) and, further down, for
                // deciding whether to use the paired placement at all.
                let score_un = a0[0].score + a1[0].score - opt.pen_unpaired;
                // `subo`: the score the paired MAPQ is measured against, the better of the runner-up
                // pair and the unpaired alternative. `q_pe`: the pair-level MAPQ, still unclamped
                // and undamped at this point, shared by both ends.
                let subo = pr.sub.max(score_un);
                let mut qdiff = pr.score - subo;
                // `--meth`: each end's second-best primary also caps the pair's margin, so a
                // repeat on either end deflates the pair MAPQ (fg-labs/bwa-mem3 0.16).
                if opt.meth.is_some() {
                    let second =
                        |a: &[MemAlnReg], n_pri: usize| if n_pri > 1 { a[1].score } else { 0 };
                    let cap = pr.score + 4 * opt.a - (second(a0, n_pri0) + second(a1, n_pri1));
                    qdiff = qdiff.min(cap).max(0);
                }
                let mut q_pe = raw_mapq(qdiff, opt.a);
                // Crowded runner-up field discounts the MAPQ, by `10*log10(n_sub+1)` phred units
                // (see [`DECIBANS_PER_NAT`]).
                if pr.n_sub > 0 {
                    q_pe -=
                        (DECIBANS_PER_NAT * crate::primary::ln_int(pr.n_sub + 1) + 0.499) as i32;
                }
                // Clamp BEFORE the repeat scaling, not after: bwa's order, and the two differ.
                q_pe = q_pe.clamp(0, MAX_MAPQ);
                // Damp by repetitiveness: `frac_rep` is each end's fraction of repetitive seeds, so
                // a fully repetitive pair (both at 1.0) is scaled by 0. f64 is required because the
                // C promotes the two `float` `frac_rep`s to `double` for this expression, but the
                // ADDITION `a0.frac_rep + a1.frac_rep` happens in f32 first (both operands are
                // `float`), which is why the sum is computed inside the `f64::from` here.
                q_pe = (f64::from(q_pe) * (1.0 - 0.5 * f64::from(a0[0].frac_rep + a1[0].frac_rep))
                    + 0.499) as i32;

                // `extra_flag`: shadows the outer 0x1 so 0x2 can be added on the proper-pair branch
                // only. `z[i]`: the index in end `i`'s vector of the region actually emitted, which
                // the `else` branch resets to that end's own best. `q_se[i]`: end `i`'s final MAPQ,
                // starting as its single-end value and ending as the blended, capped one.
                let mut extra_flag = extra_flag;
                let mut z = pr.z;
                let mut q_se = [0i32; 2];
                // The decision: is the paired placement actually better than reporting both ends
                // independently? If yes, keep `pr.z` and flag 0x2 (proper pair). If no, fall back
                // to each end's own best hit (`z = [0, 0]`) and DO NOT set 0x2, while still writing
                // the two records through the paired emitter.
                if pr.score > score_un {
                    // The C keeps pointers `c[0]`/`c[1]` into the two vectors and reads
                    // `c[i]->score` / `c[i]->csub` after the loop below; we copy the two values up
                    // front instead, purely to satisfy the borrow checker. Equivalent because that
                    // loop only writes `sub` and `secondary`, never `score` or `csub`.
                    let chosen_score = [a0[z[0]].score, a1[z[1]].score];
                    let chosen_csub = [
                        crate::lazy_csub::get(a0[z[0]].csub),
                        crate::lazy_csub::get(a1[z[1]].csub),
                    ];
                    for i in 0..2 {
                        let a = if i == 0 { &mut *a0 } else { &mut *a1 };
                        let zi = z[i];
                        // The paired region may have been marked secondary to some other hit. Since
                        // pairing has now promoted it, adopt that hit's score as its sub-optimal
                        // and set `secondary = -2`: a distinct sentinel from -1 meaning "primary,
                        // but only because pairing said so". `mem_approx_mapq_se` reads `sub`, so
                        // this must happen before the call.
                        if a[zi].secondary >= 0 {
                            a[zi].sub = a[a[zi].secondary as usize].score;
                            a[zi].secondary = -2;
                        }
                        q_se[i] = mem_approx_mapq_se(opt, &a[zi]) as i32;
                    }
                    // Blend the single-end and paired MAPQs. Reading the C's nested ternary
                    // (`q_se > q_pe? q_se : q_pe < q_se + 40? q_pe : q_se + 40`) as a whole: take
                    // the larger of the two, but never let the pairing evidence add more than 40
                    // phred units on top of what the single-end alignment alone justified. So a
                    // read that is genuinely ambiguous on its own cannot be rescued to MAPQ 60 by
                    // its mate.
                    for i in 0..2 {
                        q_se[i] = if q_se[i] > q_pe {
                            q_se[i]
                        } else if q_pe < q_se[i] + MAX_PAIRING_MAPQ_BOOST {
                            q_pe
                        } else {
                            q_se[i] + MAX_PAIRING_MAPQ_BOOST
                        };
                        // Tandem-repeat cap: `csub` is the second-best score found by the SAME
                        // local alignment (ksw's KSW_XSUBO), so a small `score - csub` means the
                        // locus itself is internally repetitive and the placement within it is
                        // arbitrary, however good the pairing looks.
                        let mapq_cap = raw_mapq(chosen_score[i] - chosen_csub[i], opt.a);
                        q_se[i] = q_se[i].min(mapq_cap);
                    }
                    // 0x2, proper pair. Only set on this branch.
                    extra_flag |= 2;
                } else {
                    z = [0, 0];
                    q_se[0] = mem_approx_mapq_se(opt, &a0[0]) as i32;
                    q_se[1] = mem_approx_mapq_se(opt, &a1[0]) as i32;
                }

                // Primary/secondary swap on `secondary_all` so `mem_gen_alt` attributes the
                // paired region z[i]'s siblings to it as XA hits (mem_sam_pe lines 474-483).
                // Both ends' primary-region counts, indexable by `i` inside the loop below.
                let n_pri = [n_pri0, n_pri1];
                for i in 0..2 {
                    let a = if i == 0 { &mut *a0 } else { &mut *a1 };
                    let zi = z[i] as i32;
                    // `secondary_all` is a second, ALT-blind shadowing relation, used only to build
                    // XA:Z. Before pairing, `z[i]` may be shadowed by some region `k`; now that
                    // pairing has chosen `z[i]`, the relation is INVERTED so `z[i]` becomes the
                    // head of that shadow group and `k` (plus everything that pointed at `k`)
                    // points at `z[i]` instead. Without this, `mem_gen_alt` would attribute the
                    // group's XA hits to `k`, whose record is never emitted, and the paired record
                    // would come out with no XA:Z at all.
                    //
                    // The C's loop is `if (a[j].secondary_all == k || j == k) a[j].secondary_all =
                    // z[i]`, which we split into the sweep plus the explicit `a[k]` assignment.
                    // The `a[z[i]].secondary_all = -1` last: `z[i]` itself matched `== k`'s group
                    // in the sweep, so it must be reset to "not shadowed" afterwards.
                    // `k`: the region currently heading `z[i]`'s XA shadow group, or < 0 if `z[i]`
                    // is not shadowed (in which case nothing is inverted).
                    let k = a[z[i]].secondary_all;
                    if k >= 0 && (k as usize) < n_pri[i] {
                        for r in a.iter_mut() {
                            if r.secondary_all == k {
                                r.secondary_all = zi;
                            }
                        }
                        a[k as usize].secondary_all = zi;
                        a[z[i]].secondary_all = -1;
                    }
                }
                // Per-region XA:Z strings for each end, index-parallel with `a0`/`a1`. Generated
                // AFTER the inversion above, which is the whole reason the inversion exists: only
                // then does `xa0[z[0]]` hold the shadow group's alternates.
                //
                // `-a` (MEM_F_ALL) suppresses XA ENTIRELY on this path: the C is
                // `if (!(opt->flag & MEM_F_ALL)) XA[i] = mem_gen_alt(...); else XA[0] = XA[1] = 0;`
                // (`bwamem_pair.cpp:484`). The two are alternatives, not complements: `-a` already
                // emits every shadowed region as its own record, so listing those same placements
                // again inside a tag would duplicate them. Generating XA here regardless was a real
                // byte-parity bug, invisible because `-a` was only ever exercised single-end.
                // `-Y` and `-V`, needed by every `mem_aln2sam` call in this branch.
                let softclip = opt.flag & bwa_core::opt::flags::SOFTCLIP != 0;
                let ref_hdr = opt.flag & bwa_core::opt::flags::REF_HDR != 0;
                let all = opt.flag & bwa_core::opt::flags::ALL != 0;
                let (mut xa0, mut xa1) = if all {
                    (crate::alt::XaTags::default(), crate::alt::XaTags::default())
                } else {
                    (
                        crate::emit_split::measure(
                            &crate::emit_split::ALT_NS,
                            &crate::emit_split::ALT_CALLS,
                            || mem_gen_alt(fm, bns, &ropt[0], a0, seqs[0].len() as i32, seqs[0]),
                        ),
                        crate::emit_split::measure(
                            &crate::emit_split::ALT_NS,
                            &crate::emit_split::ALT_CALLS,
                            || mem_gen_alt(fm, bns, &ropt[1], a1, seqs[1].len() as i32, seqs[1]),
                        ),
                    )
                };

                // One record per end WITHOUT ALT contigs (bwa's `n_aa[i]` stays 1), two when the end
                // also has a reportable ALT hit: see the `n_pri < n` block after `h0`/`h1` below.
                //
                // 0x40 = first in pair, 0x80 = second (bwa writes `0x40 << i`).
                //
                // UNVERIFIED: the `.max(0)` on MAPQ has no counterpart in the C, which assigns
                // straight into `mapq:8`, an UNSIGNED 8-bit bitfield (`bwamem.h:172`). A negative
                // `q_se` would therefore print as 256 + q_se in bwa but as 0 here. `q_se` looks
                // reachable-negative through the tandem-repeat cap above (`raw_mapq(score - csub)`
                // is negative whenever `csub > score`); whether that combination actually occurs,
                // and what bwa prints when it does, has not been checked against a real run.
                // `h0`/`h1`: the single record emitted for read 1 and read 2. Each is also handed to
                // the other's `mem_aln2sam` call as the mate.
                // `--meth` only: HN:i per region, absent under `-a` like XA.
                let hn = [&*a0, &*a1].map(|a| {
                    if opt.meth.is_some() && !all {
                        crate::alt::hit_counts(opt, a)
                    } else {
                        vec![-1; a.len()]
                    }
                });
                let h0 = {
                    let mut h =
                        reg2aln(fm, bns, &ropt[0], seqs[0].len() as i32, seqs[0], &a0[z[0]]);
                    h.mapq = q_se[0].max(0) as u32;
                    h.flag |= 0x40 | extra_flag;
                    // Moved, not cloned: `z[0] < n_pri0`, and `alt_extra` reads only `xa0[n_pri0]`.
                    h.xa = xa0.take(z[0]);
                    h.hn = hn[0][z[0]];
                    h
                };
                let h1 = {
                    let mut h =
                        reg2aln(fm, bns, &ropt[1], seqs[1].len() as i32, seqs[1], &a1[z[1]]);
                    h.mapq = q_se[1].max(0) as u32;
                    h.flag |= 0x80 | extra_flag;
                    // Moved, not cloned: `z[1] < n_pri1`, and `alt_extra` reads only `xa1[n_pri1]`.
                    h.xa = xa1.take(z[1]);
                    h.hn = hn[1][z[1]];
                    h
                };
                // ---- The ALT hit, if this end has one worth reporting (`bwamem_pair.cpp:496`) --
                // The paired record above is a primary-assembly hit. When the end ALSO hits an ALT
                // contig, bwa emits that as a second, SUPPLEMENTARY record (0x800) so the ALT
                // placement is not lost, while the pairing and the coordinate system stay anchored
                // on the primary assembly.
                //
                // Only ONE candidate is ever considered: `a[n_pri]`, the first region after the
                // primary-assembly prefix, i.e. the best-scoring ALT hit. It is dropped unless it
                // clears `-T`, is unshadowed, and really is ALT. Nothing is emitted at all when the
                // index has no ALT contigs, since then `n_pri == a.len()`.
                let alt_extra = |i: usize,
                                 a: &[MemAlnReg],
                                 n_pri: usize,
                                 xa: &crate::alt::XaTags,
                                 seq: &[u8],
                                 flag_bit: u32|
                 -> Option<MemAln> {
                    if n_pri >= a.len() {
                        return None;
                    }
                    let p = &a[n_pri];
                    if p.score < opt.t || p.secondary >= 0 || !p.is_alt {
                        return None;
                    }
                    let mut g = reg2aln(fm, bns, &ropt[i], seq.len() as i32, seq, p);
                    g.flag |= 0x800 | flag_bit | extra_flag;
                    g.xa = xa.get(n_pri);
                    g.hn = hn[i][n_pri];
                    Some(g)
                };
                // bwa's `aa[i]`: every record emitted for this end, primary first. Its LENGTH is
                // what makes the difference downstream, because `mem_aln2sam` uses it to decide
                // whether an `SA:Z` is owed and how to clip.
                let extra0 = alt_extra(0, a0, n_pri0, &xa0, seqs[0], 0x40);
                let extra1 = alt_extra(1, a1, n_pri1, &xa1, seqs[1], 0x80);
                // Backing storage for the TWO-record case only. On an index with no ALT contigs
                // `alt_extra` returns `None` for every read, so `aa0`/`aa1` borrow `h0`/`h1` in
                // place and nothing is allocated or copied. They used to be
                // `vec![h0.clone()]`, which on the common path was one `Vec` plus a `MemAln` clone
                // per end -- and a `MemAln` clone is its CIGAR `Vec`, its `md` String and its `xa`
                // String, so up to eight allocations per pair to build two one-element lists that
                // `mem_aln2sam` only ever reads.
                let (both0, both1);
                let aa0: &[MemAln] = match extra0 {
                    None => std::slice::from_ref(&h0),
                    Some(g) => {
                        both0 = [h0.clone(), g];
                        &both0
                    }
                };
                let aa1: &[MemAln] = match extra1 {
                    None => std::slice::from_ref(&h1),
                    Some(g) => {
                        both1 = [h1.clone(), g];
                        &both1
                    }
                };

                // The MATE handed to the other end's records is `h`, the paired primary record,
                // never the ALT supplementary one (the C passes `&h[1]` and `&h[0]`).
                // ONE buffer for both ends, taken from the per-thread scratch, so a pair costs no
                // allocation here at all after the first one on this thread. It was two `Vec`s
                // grown from empty per pair, each reallocating its way up to a few hundred bytes.
                // Both ends still land in the buffer before anything is written, and now in a
                // single `write_all` rather than two, so a pair's records cannot be split.
                let mut buf = SCRATCH_SAM.with(|c| std::mem::take(&mut *c.borrow_mut()));
                buf.clear();
                for which in 0..aa0.len() {
                    mem_aln2sam(
                        bns,
                        names[0],
                        seqs[0],
                        quals[0],
                        comments[0],
                        aa0,
                        which,
                        Some(&h1),
                        softclip,
                        ref_hdr,
                        emit[0].as_ref(),
                        &mut buf,
                    );
                }
                for which in 0..aa1.len() {
                    mem_aln2sam(
                        bns,
                        names[1],
                        seqs[1],
                        quals[1],
                        comments[1],
                        aa1,
                        which,
                        Some(&h0),
                        softclip,
                        ref_hdr,
                        emit[1].as_ref(),
                        &mut buf,
                    );
                }
                if opt.meth.as_ref().is_some_and(|m| m.may_qcfail()) {
                    crate::meth::propagate_qcfail(&mut buf);
                }
                let wrote = w.write_all(&buf);
                SCRATCH_SAM.with(|c| *c.borrow_mut() = buf);
                return wrote;
            }
        }
    }

    // ---- stage 4: the no_pairing fallback ----
    // Reached from four places, matching bwa's four paths to `no_pairing`:
    // `-P` was given, either end had no primary regions, `mem_pair` found no candidate pair, or one
    // end had multiple good hits (`is_multi`). Each end is then emitted as if single-end.
    //
    // Pick the representative hit for the mate fields. Prefer the overall best (index 0); failing
    // the `-T` threshold, fall back to the first ALT hit, which lives at index `n_pri`; failing
    // that, -1 for "unmapped". Note this only chooses the alignment `h0`/`h1` handed to the OTHER
    // end as its mate, not which records get written: `mem_reg2sam` re-scans the full vector.
    let pick = |a: &[MemAlnReg], n_pri: usize| -> i32 {
        if a.is_empty() {
            -1
        } else if a[0].score >= opt.t {
            0
        } else if n_pri < a.len() && a[n_pri].score >= opt.t {
            n_pri as i32
        } else {
            -1
        }
    };
    // `w0`/`w1`: index of each end's representative region, or -1 for "no reportable hit".
    let w0 = pick(a0, n_pri0);
    let w1 = pick(a1, n_pri1);
    // `h0`/`h1`: those regions rendered as alignments, used ONLY as the mate argument for the other
    // end's records. They are never emitted as records themselves on this path.
    let h0 = if w0 >= 0 {
        reg2aln(
            fm,
            bns,
            &ropt[0],
            seqs[0].len() as i32,
            seqs[0],
            &a0[w0 as usize],
        )
    } else {
        MemAln::unmapped()
    };
    let h1 = if w1 >= 0 {
        reg2aln(
            fm,
            bns,
            &ropt[1],
            seqs[1].len() as i32,
            seqs[1],
            &a1[w1 as usize],
        )
    } else {
        MemAln::unmapped()
    };
    // Shadows the outer 0x1 so 0x2 can be added below without disturbing the paired path's copy.
    let mut extra_flag = extra_flag;
    // bwa gates this proper-pair inference on `-P` too (`mem_sam_pe`: `if (!(opt->flag &
    // MEM_F_NOPAIRING) && h[0].rid == h[1].rid ...)`). Without the gate, `-P` still stamps 0x2 and
    // the FLAG comes out 83 where bwa emits 81.
    if opt.flag & bwa_core::opt::flags::NOPAIRING == 0 && h0.rid == h1.rid && h0.rid >= 0 {
        // Note the asymmetry, faithfully reproduced: the guard tests `h0`/`h1` (the picked hits),
        // but the geometry is measured on `a0[0]`/`a1[0]` (each end's overall best region). When
        // `pick` chose an ALT hit at index `n_pri`, those are different alignments. Indexing
        // `a0[0]` is also why this block is only safe under `h0.rid >= 0`, which implies both
        // vectors are non-empty.
        // `d`: the orientation code the two top hits form. `dist`: their 5'-to-5' separation in
        // reference bases, tested against that orientation's proper-pair window.
        let (d, dist) = mem_infer_dir(bns.l_pac, a0[0].rb, a1[0].rb);
        if !pes[d].failed && dist >= i64::from(pes[d].low) && dist <= i64::from(pes[d].high) {
            extra_flag |= 2;
        }
    }
    // 0x41 = 0x40 (first in pair) | 0x01 (paired), 0x81 = 0x80 (second) | 0x01. `extra_flag` also
    // carries 0x01 already, so the OR is redundant on that bit; it matters only for 0x2. Each end
    // is handed the OTHER end's picked alignment as its mate. Both buffers are built in full before
    // either is written so a pair's four-or-more lines never interleave with another thread's.
    let mut buf = SCRATCH_SAM.with(|c| std::mem::take(&mut *c.borrow_mut()));
    buf.clear();
    mem_reg2sam(
        fm,
        bns,
        &ropt[0],
        names[0],
        seqs[0],
        quals[0],
        comments[0],
        a0,
        0x41 | extra_flag,
        Some(&h1),
        emit[0].as_ref(),
        &mut buf,
    );
    mem_reg2sam(
        fm,
        bns,
        &ropt[1],
        names[1],
        seqs[1],
        quals[1],
        comments[1],
        a1,
        0x81 | extra_flag,
        Some(&h0),
        emit[1].as_ref(),
        &mut buf,
    );
    if opt.meth.as_ref().is_some_and(|m| m.may_qcfail()) {
        crate::meth::propagate_qcfail(&mut buf);
    }
    let wrote = w.write_all(&buf);
    SCRATCH_SAM.with(|c| *c.borrow_mut() = buf);
    wrote
}

#[cfg(test)]
mod tests {
    use super::{best_pair, mem_infer_dir};

    /// [`best_pair`] answers what bwa reads off the sorted candidate list, including heavy score ties.
    #[test]
    fn best_pair_matches_sorted_list() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for n in [1usize, 2, 3, 7, 40, 300] {
            for spread in [2u64, 9, 1000] {
                let u: Vec<(u64, u64)> = (0..n as u64)
                    .map(|j| ((next() % spread) << 32 | next() & 0xffff, j))
                    .collect();
                let mut s = u.clone();
                s.sort_unstable();
                let sub = if n > 1 { (s[n - 2].0 >> 32) as i32 } else { 0 };
                let n_sub = s[..n - 1]
                    .iter()
                    .filter(|c| sub - (c.0 >> 32) as i32 <= 5)
                    .count() as i32;
                assert_eq!(
                    best_pair(&u, 5),
                    (s[n - 1], sub, n_sub),
                    "n={n} spread={spread}"
                );
            }
        }
    }

    /// Pins the FF/FR/RF/RR encoding of [`mem_infer_dir`], which every `pes[4]` lookup in this file
    /// depends on. Getting the code wrong would silently swap two orientations' insert windows,
    /// which is the kind of bug that produces plausible-but-wrong output rather than a crash.
    #[test]
    fn infer_dir_orientations() {
        let l_pac = 1000i64;
        // Forward read at 100, mate reverse-strand mapping to forward pos 500 (rb = 2L-1-500).
        let (dir, dist) = mem_infer_dir(l_pac, 100, (l_pac << 1) - 1 - 500);
        assert_eq!(dir, 1, "forward-then-reverse is FR (1)");
        assert_eq!(dist, 400);
        // Both forward: FF (0).
        let (dir, _) = mem_infer_dir(l_pac, 100, 500);
        assert_eq!(dir, 0, "both forward is FF (0)");
        // Both reverse: RR (3).
        let (dir, _) = mem_infer_dir(l_pac, (l_pac << 1) - 1 - 100, (l_pac << 1) - 1 - 500);
        assert_eq!(dir, 3, "both reverse is RR (3)");
    }
}

#[cfg(test)]
mod pair_penalty_tests {
    use super::*;

    /// The memo returns exactly the expression it replaced, on a miss, on a hit, past its span and
    /// after the statistics change (a new batch), and `ln_int` is libm's `ln` bit for bit.
    #[test]
    fn memo_matches_direct_formula() {
        let direct = |p: &PeStat, dist: i64, a: i32| {
            let ns = (dist as f64 - p.avg) / p.std;
            INV_LOG4 * (2.0 * unsafe { erfc(ns.abs() * M_SQRT1_2) }).ln() * f64::from(a)
        };
        let mut pes = [PeStat::default(); 4];
        pes[1] = PeStat {
            low: 50,
            high: 900,
            failed: false,
            avg: 412.25,
            std: 61.5,
        };
        pes[2] = PeStat {
            low: 0,
            high: PairPenalty::SPAN + 10,
            failed: false,
            avg: 3e4,
            std: 9e3,
        };
        let mut memo = PairPenalty::new();
        for (round, a) in [(0, 1), (1, 1), (2, 2)] {
            if round == 1 {
                pes[1].avg += 1.0;
            }
            memo.reset_for(&pes, a);
            for _hit in 0..2 {
                for dist in [50, 51, 412, 899, 900] {
                    let (want, got) = (direct(&pes[1], dist, a), memo.get(&pes[1], 1, dist, a));
                    assert_eq!(got.to_bits(), want.to_bits(), "round {round} dist {dist}");
                }
                let far = i64::from(PairPenalty::SPAN) + 5;
                let (want, got) = (direct(&pes[2], far, a), memo.get(&pes[2], 2, far, a));
                assert_eq!(got.to_bits(), want.to_bits(), "past the span");
            }
        }
        for n in [-3, 0, 1, 2, 149, 150, 1023, 1024, 5000, i32::MAX] {
            let want = f64::from(n).ln();
            assert_eq!(
                crate::primary::ln_int(n).to_bits(),
                want.to_bits(),
                "ln_int({n})"
            );
        }
    }
}

#[cfg(test)]
mod id_shift_tests {
    use super::id_shift_c;

    /// Pins bwa's `int id; ... id << 8` overflow: identical below 2^23, sign-extended from 2^23 on.
    #[test]
    fn id_shift_matches_c_int_overflow() {
        assert_eq!(id_shift_c(1000), 0x0000_0000_0003_e800);
        assert_eq!(id_shift_c(8_388_607), 0x0000_0000_7fff_ff00);
        // 2^23: the shift reaches the sign bit and the int is sign-extended to uint64_t.
        assert_eq!(id_shift_c(8_388_608), 0xffff_ffff_8000_0000);
        assert_eq!(id_shift_c(9_000_000), 0xffff_ffff_8954_4000);
    }
}

/// `BWA4_ANCHOR_SPREAD=1`: do one read's rescue windows overlap each other?
///
/// A (pair, direction) runs one mate-rescue SW per anchor, up to `-m` (50), all with the SAME mate
/// query and a window derived from each anchor's `rb`. If those anchors cluster, the windows overlap
/// and most of that DP is recomputation: in local SW no positive-scoring alignment can span more
/// than `l_ms + (a * l_ms - o) / e` target rows (294 for a 150 bp mate under `a=1, o=6, e=1`), so two
/// windows that overlap by more than that agree EXACTLY on the shared rows past the burn-in. One
/// pass over the union would then answer every sub-window, per-row maxima included.
///
/// This probe prices that: `SPAN` is the sum of per-anchor window widths (what we compute today) and
/// `UNION` is the width of their union per (pair, direction). `SPAN / UNION` is the ceiling of the
/// idea; anything near 1.0 means the anchors are scattered and there is nothing to share.
pub mod anchor_spread {
    use super::Anchor;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;
    /// Summed per-anchor window widths, in bases.
    pub static SPAN: AtomicU64 = AtomicU64::new(0);
    /// Summed width of the union of one (pair, direction)'s windows, in bases.
    pub static UNION: AtomicU64 = AtomicU64::new(0);
    /// (pair, direction) groups with at least two anchors.
    pub static GROUPS: AtomicU64 = AtomicU64::new(0);
    /// Anchors counted.
    pub static ANCHORS: AtomicU64 = AtomicU64::new(0);
    /// Groups whose anchors all fall inside one window width, i.e. fully overlapping.
    pub static TIGHT: AtomicU64 = AtomicU64::new(0);
    /// Whether `BWA4_ANCHOR_SPREAD` is set; read once and cached.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_ANCHOR_SPREAD").is_some())
    }
    /// Accumulate one batch's anchor lists. No-op unless [`enabled`].
    ///
    /// The window width is approximated by a constant 1400 bases, the measured mean; the exact width
    /// depends on `pes[r].high - pes[r].low + l_ms` and does not change the shape of the answer.
    pub(crate) fn record(anchors: &[Anchor], spans: &[[(u32, u32); 2]]) {
        if !enabled() {
            return;
        }
        const W: i64 = 1400;
        for per_pair in spans {
            for &(start, len) in per_pair {
                let list = &anchors[start as usize..(start + len) as usize];
                if list.len() < 2 {
                    continue;
                }
                GROUPS.fetch_add(1, Relaxed);
                ANCHORS.fetch_add(list.len() as u64, Relaxed);
                SPAN.fetch_add((list.len() as u64) * W as u64, Relaxed);
                let mut starts: Vec<i64> = list.iter().map(|a| a.rb).collect();
                starts.sort_unstable();
                // Sweep the sorted window starts, merging overlaps.
                let (mut union, mut cur_end) = (0u64, i64::MIN);
                for s in starts.iter().copied() {
                    let (b, e) = (s, s + W);
                    if b > cur_end {
                        union += W as u64;
                        cur_end = e;
                    } else if e > cur_end {
                        union += (e - cur_end) as u64;
                        cur_end = e;
                    }
                }
                UNION.fetch_add(union, Relaxed);
                if union <= W as u64 {
                    TIGHT.fetch_add(1, Relaxed);
                }
            }
        }
    }
    /// Print the accumulated counters once at end of run. No-op unless [`enabled`].
    pub fn dump() {
        if !enabled() {
            return;
        }
        let (span, union, groups, anchors, tight) = (
            SPAN.load(Relaxed),
            UNION.load(Relaxed),
            GROUPS.load(Relaxed),
            ANCHORS.load(Relaxed),
            TIGHT.load(Relaxed),
        );
        eprintln!(
            "[anchor-spread] {groups} multi-anchor groups, {anchors} anchors ({:.1} per group); \
             window span {span} bases vs union {union} ({:.2}x redundancy); \
             {tight} groups ({:.1}%) fit inside a single window",
            anchors as f64 / groups.max(1) as f64,
            span as f64 / union.max(1) as f64,
            100.0 * tight as f64 / groups.max(1) as f64,
        );
    }
}

#[cfg(test)]
mod region_view_tests {
    use super::*;

    fn xorshift() -> impl FnMut(u64) -> u64 {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        }
    }

    /// Regions on both strands of a 100 kb reference, short and long, clustered so that overlaps
    /// and proper-pair distances are the common case.
    fn region(next: &mut dyn FnMut(u64) -> u64) -> MemAlnReg {
        let l_pac = 100_000i64;
        let rb = next(2) as i64 * l_pac + next(60) as i64 * 5 + next(3) as i64 * 400;
        let len = 20 + next(6) as i64 * 30;
        let qb = next(4) as i32 * 10;
        let qlen = (len as i32 + next(3) as i32 * 2).min(148 - qb);
        MemAlnReg {
            rb,
            re: rb + len,
            qb,
            qe: qb + qlen,
            rid: 0,
            score: 30 + next(5) as i32 * 5,
            truesc: 0,
            sub: 0,
            csub: next(3) as i32,
            sub_n: 0,
            seedcov: next(2) as i32,
            seedlen0: 0,
            secondary: -1,
            secondary_all: 0,
            w: 0,
            frac_rep: 0.0,
            is_alt: false,
            alt_sc: 0,
        }
    }

    #[test]
    fn skip_from_view_equals_the_linear_scan() {
        let l_pac = 100_000i64;
        let mut next = xorshift();
        let mut pes = [PeStat::default(); 4];
        for _ in 0..300 {
            for p in pes.iter_mut() {
                p.low = next(3) as i32 * 200 - 100;
                p.high = p.low + next(4) as i32 * 300;
                p.failed = next(5) == 0;
            }
            let n = VIEW_MIN + next(100) as usize;
            let ma: Vec<MemAlnReg> = (0..n).map(|_| region(&mut next)).collect();
            let mut view = RegionView::default();
            assert!(view.ensure(&ma));
            for _ in 0..8 {
                let a_rb = region(&mut next).rb;
                let mut want = [0i32; 4];
                for (r, s) in want.iter_mut().enumerate() {
                    *s = i32::from(pes[r].failed);
                }
                let mut got = want;
                for existing in &ma {
                    let (r, dist) = mem_infer_dir(l_pac, a_rb, existing.rb);
                    if dist >= i64::from(pes[r].low) && dist <= i64::from(pes[r].high) {
                        want[r] = 1;
                    }
                }
                skip_from_view(&view, l_pac, a_rb, &pes, &mut got);
                assert_eq!(got, want, "anchor {a_rb} pes {pes:?}");
            }
        }
    }

    #[test]
    fn insert_settled_view_equals_insert_into_settled() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(std::path::Path::new(prefix)).unwrap();
        let opt = MemOpt::default();
        let mut next = xorshift();
        let same = |x: &[MemAlnReg], y: &[MemAlnReg]| format!("{x:?}") == format!("{y:?}");
        let (mut taken, mut declined) = (0u32, 0u32);
        for _ in 0..400 {
            let n = VIEW_MIN + next(100) as usize;
            let pool: Vec<MemAlnReg> = (0..n).map(|_| region(&mut next)).collect();
            let settled = mem_sort_dedup_patch(&fm, &opt, &[], pool);
            if settled.len() < VIEW_MIN {
                continue;
            }
            let mut fast = settled.clone();
            let mut full = settled.clone();
            let mut view = RegionView::default();
            // Several inserts in a row: the view must follow its own pushes.
            for _ in 0..6 {
                let mut b = region(&mut next);
                if next(2) == 0 {
                    b = full[next(full.len() as u64) as usize].clone();
                    match next(4) {
                        0 => b.csub += 1,
                        1 => b.re += 1,
                        2 => b.score += 1,
                        _ => {}
                    }
                }
                assert!(view.ensure(&fast) && view.clean);
                let want = insert_into_settled(&opt, &mut full, b.clone());
                let got = insert_settled_view(&opt, &mut fast, &mut view, b.clone());
                assert_eq!(got.is_ok(), want.is_ok(), "b {b:?}");
                // `fast` keeps its hits pending across the inserts, as a rescue round does.
                let (mut chk, mut pending) = (fast.clone(), view.pending);
                settle_order(&mut chk, &mut pending);
                assert!(same(&chk, &full), "b {b:?}\nfast {chk:?}\nfull {full:?}");
                if got.is_ok() {
                    taken += 1;
                } else {
                    declined += 1;
                }
            }
        }
        assert!(
            taken > 100 && declined > 100,
            "taken {taken} declined {declined}"
        );
    }
}

#[cfg(test)]
mod matesw_apply_tests {
    use super::*;

    /// The C dedups the mate's list once any orientation has run SW, even when it found nothing
    /// (`if (n)`, `bwamem_pair.cpp:277`). A list fresh from the single-end dedup is not yet a fixed
    /// point of that null-query pass, so the batched path must run it too: here it drops a duplicate.
    #[test]
    fn unsettled_list_is_deduped_even_without_a_hit() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(std::path::Path::new(prefix)).unwrap();
        let bns = BntSeq::load(std::path::Path::new(prefix)).unwrap();
        let opt = MemOpt::default();
        let region = |score: i32| MemAlnReg {
            rb: 100,
            re: 200,
            qb: 0,
            qe: 100,
            rid: 0,
            score,
            truesc: score,
            sub: 0,
            csub: 0,
            sub_n: 0,
            seedcov: 50,
            seedlen0: 0,
            secondary: -1,
            secondary_all: 0,
            w: 0,
            frac_rep: 0.0,
            is_alt: false,
            alt_sc: 0,
        };
        let ma0 = vec![region(90), region(80)];
        let want = mem_sort_dedup_patch(&fm, &opt, &[], ma0.clone());
        assert_eq!(
            want.len(),
            1,
            "the case must be one the null-query pass changes"
        );
        let call = RescueCall {
            skip: [0, 1, 1, 1],
            per_r: [
                Some(Orient {
                    target: (0, 40),
                    rb: 0,
                    is_rev: false,
                }),
                None,
                None,
                None,
            ],
            rc: (0, 0),
            l_ms: 20,
            rid: 0,
            is_alt: false,
        };
        let miss = KswAlignResult {
            score: 0,
            qb: -1,
            qe: -1,
            tb: -1,
            te: -1,
            score2: -1,
            te2: -1,
        };
        let (mut ma, mut view) = (ma0, RegionView::default());
        matesw_apply(&fm, &bns, &opt, &call, &[miss], &mut ma, &mut view, None, 0);
        assert_eq!(format!("{ma:?}"), format!("{want:?}"));
    }

    /// Under --meth a region can score below zero. The fork scores a pair with such an end q = 0
    /// (bwamem_pair.cpp, mem_pair); reading the packed score back unsigned made it the best pair.
    #[test]
    fn negative_score_end_pairs_at_zero() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let bns = BntSeq::load(std::path::Path::new(prefix)).unwrap();
        let opt = MemOpt::default();
        let reg = |rb: i64, score: i32| MemAlnReg {
            rb,
            re: rb + 150,
            qb: 0,
            qe: 150,
            rid: 0,
            score,
            truesc: score,
            sub: 0,
            csub: 0,
            sub_n: 0,
            seedcov: 50,
            seedlen0: 0,
            secondary: -1,
            secondary_all: 0,
            w: 0,
            frac_rep: 0.0,
            is_alt: false,
            alt_sc: 0,
        };
        let pes = [PeStat {
            low: 1,
            high: 10_000,
            failed: false,
            avg: 300.0,
            std: 50.0,
        }; 4];
        // read 1 forward at 100, read 2 reverse ending at forward 400: one FR candidate.
        let rev = 2 * bns.l_pac - 400;
        let score = |sc2| {
            let (a0, a1) = ([reg(100, 20)], [reg(rev, sc2)]);
            mem_pair(&bns, &opt, &pes, &[&a0, &a1], 7).map(|p| p.score)
        };
        assert!(
            score(5).is_some_and(|q| q > 0),
            "the setup must form a pair"
        );
        assert_eq!(score(-5), Some(0));
    }
}

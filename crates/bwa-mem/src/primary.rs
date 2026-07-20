//! Region dedup, primary/secondary marking and MAPQ, mirroring bwa-mem2's
//! `mem_sort_dedup_patch`, `mem_mark_primary_se[_core]` and `mem_approx_mapq_se`
//! (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Includes the `mem_patch_reg` region-merge branch (collinear regions re-aligned into one),
//! which fires on split/long-indel alignments.
//!
//! # Where this sits in the pipeline
//!
//! Seeding and chaining produce a set of candidate alignment regions ([`MemAlnReg`]) per read,
//! each already extended and scored. This module is the stage that turns that raw candidate set
//! into the answer a SAM record needs: which regions are actually distinct alignments, which one
//! of them is *the* alignment (primary) with the others hanging off it as secondaries, and how
//! confident we are in that choice (MAPQ). CIGAR generation (`reg2aln`) and pairing run after.
//!
//! # Reading order
//!
//! 1. [`mem_sort_dedup_patch`]: the dedup/merge pass, run first on the raw region list. It calls
//!    `mem_patch_reg` for the collinear-merge branch.
//! 2. `mem_patch_reg`: the merge trial itself (a global re-alignment across two regions).
//! 3. [`mem_mark_primary_se`]: sorts best-first and delegates to `mark_primary_core`, which does
//!    the greedy primary/secondary sweep and fills in `sub`/`sub_n`.
//! 4. [`mem_approx_mapq_se`]: reads the fields the sweep filled in and produces the MAPQ.
//!
//! # Glossary
//!
//! Many identifiers below keep bwa's C spelling deliberately, because a reader checking parity
//! against `bwamem.cpp` needs to be able to match them line for line. What they mean:
//!
//! | Name | Meaning |
//! |------|---------|
//! | `a` | in the C, the array of regions for one read (and, in `mem_patch_reg`, the earlier of the two regions being merged) |
//! | `p` / `q` | the C's cursors inside the dedup scan: `p` is the later region `a[i]`, `q` the earlier `a[j]` |
//! | `rb` / `re` | reference begin / end of a region, in the concatenated forward+reverse pac coordinate space |
//! | `qb` / `qe` | query (read) begin / end of a region. `qe == qb` (zero length) is the "killed" marker used during dedup |
//! | `rid` | reference sequence (chromosome) id |
//! | `score` | the region's alignment score, possibly from a banded retry |
//! | `truesc` | the "true" score: the score of the alignment actually reported, used to decide whether a band retry converged |
//! | `sub` | best score among *overlapping competing* regions, i.e. the runner-up explanation of the same read bases. 0 means unset |
//! | `csub` | same idea but for the best competitor found within the same seed chain |
//! | `sub_n` | how many competitors scored close enough to the primary to dilute confidence; each one costs MAPQ |
//! | `seedcov` | number of read bases covered by seeds supporting this region |
//! | `seedlen0` | length of the seed the chain was founded on |
//! | `frac_rep` | fraction of the read covered by seeds that hit repetitive loci; scales MAPQ down at the end |
//! | `w` | band width (in bp) used for the banded dynamic programming of this region |
//! | `n_comp` | how many original regions were folded into this one by merging |
//! | `hash` | Thomas Wang hash of the global read id plus the region index: a deterministic, thread-count-independent tie-breaker for equal-scoring regions |
//! | `l` | in MAPQ, the alignment length (longer of the query and reference spans) |
//! | `identity` | in MAPQ, the fractional identity *inferred from the score*, not counted from a CIGAR |
//! | `mapq` | mapping quality, phred-scaled, clamped to 0..=60 |
//! | `or_` / `oq` | overlap length of two regions on the reference / on the query |
//! | `mr` / `mq` | the shorter of the two regions' lengths on the reference / on the query |
//! | `b_max` / `e_min` | begin and end of the intersection of two query intervals |
//! | `min_l` | the shorter of two regions' query lengths |
//! | `q_s` / `r_s` | in `mem_patch_reg`, the merged score predicted from query spans / from reference spans |

use bwa_chain::ks_introsort_by;
use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

use crate::cigar::gen_cigar2;
use crate::MemAlnReg;

/// Maximum tolerated *relative* band width for a merge, `bwamem.cpp:172`.
///
/// **Declared `f32` on purpose, and widened only at the comparison.** The C writes `0.05f`, a
/// `float` literal, and compares it against a `double` `r` (our `rel_band`); the usual arithmetic
/// conversions promote the literal, so the threshold the C actually applies is `(double)0.05f` =
/// 0.05000000074505806, not 0.05. Writing `0.05_f64` here (0.05000000000000000277) makes our
/// threshold ~7.5e-10 *smaller*, so a `rel_band` landing in that sliver is accepted by the C and
/// would be rejected here, dropping a merge bwa performs. `PATCH_MAX_R_BW * 2.0` has the same
/// story and must also be evaluated in `f32`: the C's `PATCH_MAX_R_BW*2` converts the `int` 2 to
/// `float` and multiplies in single precision, giving exactly `0.1f` = 0.100000001490116, where
/// f64 `0.05 * 2` gives 0.100000000000000006.
///
/// This is the same class of bug as the `mask_level`/`mask_level_redun` f32 mismatch fixed earlier
/// (see the dedup redundancy test and `mem_pair`'s `cal_sub`), so it is fixed the same way rather
/// than left as a documented hazard.
const PATCH_MAX_R_BW: f32 = 0.05;
/// Minimum ratio of the merged global score to the predicted score for the merge to be accepted,
/// `bwamem.cpp:173`. Same `f32` promotion rule as above, and the gap is 32x wider here:
/// `(double)0.90f` = 0.89999997615814209 against f64 0.9 = 0.90000000000000002, a 2.4e-8
/// difference. The comparison is `<`, so ratios in that sliver are accepted by the C.
const PATCH_MIN_SC_RATIO: f32 = 0.90;

/// bwa's round-to-nearest idiom: add this, then truncate with a C cast (`as i32` here).
///
/// It is `.499` in the C, not `.5`, at every site that uses it (`bwamem.cpp:213`, `:216`,
/// `:1484`, `:1488`, `:1492`), so a value whose fractional part is exactly 0.5 rounds *down*.
/// Reproducing the constant exactly is required for byte parity; "fixing" it to 0.5 shifts scores
/// and MAPQs by one at the ties.
const ROUND_BIAS: f64 = 0.499;

/// Scale factor from score margin to phred in the MAPQ formula (`bwamem.cpp:1484`).
///
/// Empirical in bwa: treat it as calibrated so that a one-match-score margin maps to a few phred
/// points. It is not derived from anything documented in the C.
const MAPQ_SCORE_COEF: f64 = 6.02;

/// `10 / ln(10)`, so `MAPQ_SUB_N_COEF * ln(sub_n + 1)` is `10 * log10(sub_n + 1)` phred points
/// (`bwamem.cpp:1487`). The C writes the rounded literal 4.343, not the exact ratio, so we do too.
const MAPQ_SUB_N_COEF: f64 = 4.343;

/// bwa's MAPQ ceiling (`bwamem.cpp:1488`). This is bwa's own cap, not anything the SAM spec
/// imposes.
const MAPQ_MAX: i32 = 60;

/// Try to merge collinear regions `left` (earlier `rb`) and `right` via a global re-alignment.
/// Returns `(merged_score, band)` if the merge is good enough. Port of `mem_patch_reg`
/// (`bwamem.cpp:175`), whose two region arguments are named `a` and `b` there.
///
/// This is what turns two separate hits into one long-indel or split alignment. Seeding and
/// chaining can leave a single true alignment broken into two regions when the intervening indel
/// is bigger than the extension band; if the two pieces are collinear on both query and reference,
/// re-aligning across the whole span with a generous band recovers the single alignment.
///
/// Preconditions asserted by the C (`bwamem.cpp:182`) and relied on here: `left.rid == right.rid`
/// and `left.rb <= right.rb`. The caller enforces both.
///
/// # Parameters
///
/// - `fm`: the loaded FM index. Read here only for `l_pac` (the forward-strand length, used to tell
///   the two strands apart) and, inside `gen_cigar2`, for the packed reference bases.
/// - `opt`: the alignment options. Only `w` (the `-w` band width, in bp) is read directly; the
///   scoring parameters are read by `gen_cigar2`.
/// - `codes`: the whole read in nt4 codes (0..=3 for ACGT, 4 for N), length = read length. Indexed
///   by query coordinates, so it must be the *same* read both regions came from. An empty slice is
///   the caller's signal that merging is disabled (`mem_matesw`'s dedup passes no query).
/// - `left`: the earlier region on the reference. Precondition: `left.rid == right.rid` and
///   `left.rb <= right.rb`; the caller's `a[j_idx].rb < p_rb` test establishes this.
/// - `right`: the later region on the reference, same read and same reference sequence as `left`.
///
/// # Returns
///
/// `None` when the merge is rejected for any reason (the C returns score 0, which the caller tests
/// as `> 0`), and `Some((score, w))` where `score` is the global alignment score of the merged span
/// and `w` the band width (bp) used, both of which the caller writes into the survivor.
#[must_use]
fn mem_patch_reg(
    fm: &FmIndex,
    opt: &MemOpt,
    codes: &[u8],
    left: &MemAlnReg,
    right: &MemAlnReg,
) -> Option<(i32, i32)> {
    // ---- Cheap rejections: same strand, collinear on both axes -------------------------------
    if codes.is_empty() {
        return None; // `query == 0` (mem_matesw's dedup): merging is disabled
    }
    // Length of the forward strand in the concatenated forward+reverse pac. Reference coordinates
    // below `l_pac` are forward-strand hits, at or above it reverse-strand ones, so it doubles as
    // the strand boundary.
    let l_pac = fm.l_pac();
    if left.rb < l_pac && right.rb >= l_pac {
        return None; // different strands
    }
    // Collinearity: `left` must start earlier and end earlier than `right` on *both* axes. Note
    // these are strict `>=` rejections, so identical coordinates on any axis also disqualify the
    // pair. (`rb`/`re` are reference begin/end, `qb`/`qe` query begin/end.)
    if left.qb >= right.qb || left.qe >= right.qe || left.re >= right.re {
        return None; // not collinear
    }

    // ---- How wide a band would the merged alignment need? -------------------------------------
    // Required band width: the gap between the two regions measured on the reference minus the same
    // gap measured on the query. Both `left.re - right.rb` and `left.qe - right.qb` can be negative
    // (the regions may not overlap at all), and the difference is exactly the indel length that a
    // merged alignment would have to absorb. `abs` because either strand of indel counts the same.
    //
    // Width note: C declares this as `int` (`bwamem.cpp:186`, its `w`) and truncates the int64
    // subtraction into it; we keep i64 throughout. Divergence would require an indel over 2^31 bp.
    let required_band = ((left.re - right.rb) - i64::from(left.qe - right.qb)).abs();
    // Relative band width (the C's `r`): the same discrepancy normalized by the merged span, so
    // that a long merged alignment is allowed a proportionally larger indel than a short one.
    // Computed as the difference of two fractions-of-span rather than `required_band / span`
    // because the query and reference spans differ. This is `double` in C and f64 here, matching.
    let rel_band = ((left.re - right.rb) as f64 / (right.re - left.rb) as f64
        - f64::from(left.qe - right.qb) / f64::from(right.qe - left.qb))
    .abs();
    // Two tiers of tolerance (`bwamem.cpp:197`): when the regions do not overlap on the reference
    // or the query, the evidence that they belong together is weaker, so demand a tight band
    // (2x `-w`). When they overlap on both, the overlap itself corroborates the merge and the
    // limits double to 4x `-w` and 2x the relative bound.
    // `f64::from` at the point of comparison is the C's float-to-double promotion, and the `* 2.0`
    // stays in f32 because the C multiplies before promoting. Both are load-bearing: see the
    // constants' docs.
    if left.re < right.rb || left.qe < right.qb {
        if required_band > i64::from(opt.w << 1) || rel_band >= f64::from(PATCH_MAX_R_BW) {
            return None;
        }
    } else if required_band > i64::from(opt.w << 2) || rel_band >= f64::from(PATCH_MAX_R_BW * 2.0) {
        return None;
    }
    // Band for the trial alignment: the required width plus each region's own extension band, since
    // the merged alignment must still be able to wander as much as either piece did. Capped at 4x
    // `-w` to bound the DP cost. (`w` is the C's name for a band width, kept here because the
    // caller stores it straight into the region's `w` field.)
    // `w`: band width in bp for the trial alignment below, and, on acceptance, the value stored
    // into the merged region's `w` field.
    let mut w = required_band as i32 + left.w + right.w;
    w = w.min(opt.w << 2);

    // ---- Trial global alignment across the union span -----------------------------------------
    // `[left.qb, right.qe)` x `[left.rb, right.re)`. Only the score is used; the CIGAR/NM/MD are
    // discarded, matching the C passing `0, 0` for `n_cigar` and `NM` (`bwamem.cpp:211`). The real
    // CIGAR is generated later by `reg2aln` on the merged region.
    // Query length of the merged span in bp: from the left region's start to the right region's
    // end, i.e. the number of read bases the trial alignment has to explain.
    let l_query = right.qe - left.qb;
    // Only `score` (the global alignment score over the merged span) is kept; the CIGAR, NM and MD
    // are discarded here and regenerated later by `reg2aln`.
    let (score, _c, _nm, _md) = gen_cigar2(
        fm,
        opt,
        w,
        l_query,
        &codes[left.qb as usize..right.qe as usize],
        left.rb,
        right.re,
    )?;

    // ---- Is the trial score close enough to what a real merge should score? -------------------
    // Predicted score of the merged alignment, computed two ways. If the two regions really are one
    // alignment, their combined score should scale with the merged span divided by the sum of the
    // individual spans: that ratio is below 1 when the regions overlap (shared bases were counted
    // twice) and above 1 when they are disjoint (the gap between them contributes nothing yet).
    // `q_s` uses query lengths, `r_s` reference lengths, and the stricter of the two is used below.
    // `ROUND_BIAS` before the `as i32` truncation is bwa's round-to-nearest idiom; see its doc.
    let q_s = ((f64::from(right.qe - left.qb)
        / f64::from((right.qe - right.qb) + (left.qe - left.qb)))
        * f64::from(right.score + left.score)
        + ROUND_BIAS) as i32;
    let r_s = (((right.re - left.rb) as f64 / ((right.re - right.rb) + (left.re - left.rb)) as f64)
        * f64::from(right.score + left.score)
        + ROUND_BIAS) as i32;
    // Accept only if the actual global score reaches 90% of the more optimistic prediction. Falling
    // short means the two regions do not in fact align through the interval between them.
    if f64::from(score) / f64::from(q_s.max(r_s)) < f64::from(PATCH_MIN_SC_RATIO) {
        return None;
    }
    Some((score, w))
}

/// Thomas Wang's 64-bit integer hash (`hash_64`), the deterministic tie-breaker for equal-scoring
/// regions.
///
/// Fed `id + i` where `id` is the global read index, so two equal-scoring regions of the same read
/// get an arbitrary but *reproducible* order that does not depend on how many threads ran or in
/// what order reads were batched. That reproducibility is the whole reason the hash exists; any
/// other mixing function would be equally valid as a hash but would produce different SAM output.
///
/// Every `+` is `wrapping_add` because the C relies on unsigned wraparound (defined behavior for
/// `uint64_t`); Rust would panic on these in a debug build otherwise. The shifts are logical
/// (`u64`, not `i64`), which matters for `>>` on values with the high bit set.
///
/// # Parameters
///
/// - `key`: the value to mix. Callers here always pass `global_read_id + region_index`; any `u64`
///   is valid. It is reassigned in place through the mixing steps, so partway down it holds the
///   partially mixed hash, not the original input.
///
/// # Returns
///
/// The mixed 64-bit hash, used only as a sort key (`alnreg_hlt`'s last tie-break). The exact bit
/// pattern is part of the output contract: change the mixing and a different one of two
/// equal-scoring regions becomes primary, changing the SAM bytes.
pub fn hash_64(mut key: u64) -> u64 {
    key = key.wrapping_add(!(key << 32));
    key ^= key >> 22;
    key = key.wrapping_add(!(key << 13));
    key ^= key >> 8;
    key = key.wrapping_add(key << 3);
    key ^= key >> 15;
    key = key.wrapping_add(!(key << 27));
    key ^= key >> 31;
    key
}

/// Remove redundant and identical alignment regions, merging collinear ones. Port of
/// `mem_sort_dedup_patch` (`bwamem.cpp:292`), including the `mem_patch_reg` merge branch.
///
/// Three passes, in this order and not interchangeable:
/// 1. Sort by reference END, then for each region look backwards at every region that could still
///    overlap it. A pair that overlaps heavily on both axes is redundant, so the lower-scoring one
///    is killed; a pair that is merely collinear is offered to [`mem_patch_reg`] for merging.
/// 2. Compact out the killed regions.
/// 3. Re-sort by score and drop exact coordinate duplicates.
///
/// `codes` is the whole read in nt4 codes, needed only by the merge branch. An empty slice
/// disables merging (see [`mem_patch_reg`]).
///
/// Regions are "killed" by setting `qe = qb`, i.e. giving them zero query length, rather than being
/// removed immediately. This is essential, not stylistic: the backward scan holds indices into `a`,
/// so removing an element mid-scan would invalidate them. The C does the same and tests `q->qe ==
/// q->qb` to skip already-dead entries (`bwamem.cpp:311`).
///
/// # Parameters
///
/// - `fm`: the loaded FM index, forwarded to [`mem_patch_reg`] for the merge trial's re-alignment.
/// - `opt`: the alignment options. Read here for `mask_level_redun` (redundancy overlap fraction)
///   and `max_chain_gap` (bp); the rest is forwarded.
/// - `codes`: the whole read in nt4 codes, length = read length. Used only by the merge branch; an
///   empty slice disables merging (see [`mem_patch_reg`]).
/// - `a`: the read's candidate regions, taken by value and returned. Any order on entry (this
///   function sorts it twice). All entries must belong to the same read as `codes`.
///
/// # Returns
///
/// The surviving regions, sorted by score descending then `rb` then `qb`, with the killed ones
/// compacted out and merged ones rewritten in place. Possibly shorter than the input, never longer.
pub fn mem_sort_dedup_patch(
    fm: &FmIndex,
    opt: &MemOpt,
    codes: &[u8],
    mut a: Vec<MemAlnReg>,
) -> Vec<MemAlnReg> {
    if a.len() <= 1 {
        return a;
    }
    // f32, not f64: `mask_level_redun` is a C `float`, and `or_ > opt->mask_level_redun * mr`
    // (`or_` is the reference overlap, `mr` the shorter reference span; see the module glossary)
    // promotes the int64 to float, so the whole comparison happens in single precision. Widening to
    // f64 changes the result at the boundary -- 0.95f is really 0.9499999880790710, so f32 rounds
    // 0.95f * 40 back up to exactly 38.0 (38 > 38.0 is false, hit kept) while f64 keeps
    // 37.99999952 (38 > that is true, hit dropped). One region's fate per boundary case.
    let mask_level_redun = opt.mask_level_redun;
    // Slack, in bp on the reference, for "these two regions could still be parts of one alignment".
    // Widened to i64 once here because every use compares it against int64 reference coordinates.
    let max_gap = i64::from(opt.max_chain_gap);

    // ---- Pass 1a: sort by reference end ------------------------------------------------------
    // Sort by END position on the reference (`alnreg_slt2`). bwa's `ks_introsort` is *unstable* and
    // the key is `re` alone, so ties are both common and consequential: the redundancy test below
    // drops `q` (the earlier entry) whenever the scores tie, meaning the order among equal-`re`
    // regions decides which alignment survives. A stable sort silently picks a different one.
    ks_introsort_by(&mut a, |x, y| x.re < y.re);
    // `n_comp` counts how many original regions were folded into this one; it starts at 1 (itself)
    // and the merge branch accumulates. Only `mem_patch_reg` merging changes it.
    for r in &mut a {
        r.n_comp = 1;
    }

    // ---- Pass 1b: backward scan, killing redundant regions and merging collinear ones ---------
    for i in 1..a.len() {
        // Cheap pre-filter (`bwamem.cpp:304`): if region `i` cannot even reach its immediate
        // predecessor, it cannot reach anything further back either, because the array is sorted by
        // `re` ascending and the scan only goes backwards. `max_chain_gap` is the slack: two pieces
        // of one alignment can be separated by up to that much reference without disqualifying.
        if a[i].rid != a[i - 1].rid || a[i].rb >= a[i - 1].re + max_gap {
            continue;
        }
        // Backward cursor over the regions preceding `i`. Signed because it is allowed to fall to
        // -1 to end the scan. Invariant at the top of each iteration: every region in `j+1..i` has
        // already been compared against `a[i]` and either killed, merged into it, or left alone,
        // and `a[i]` is still alive (the redundancy branch `break`s as soon as it dies).
        let mut j = i as i64 - 1;
        while j >= 0 {
            // Same cursor as an index; `j >= 0` is guaranteed by the loop condition.
            let j_idx = j as usize;
            // Read p (a[i]) fresh each iteration: a merge below updates it in place. `p` is the
            // C's name for the later region under test, `q` (below) for the earlier one.
            let (p_rid, p_rb, p_re, p_qb, p_qe, p_score) =
                (a[i].rid, a[i].rb, a[i].re, a[i].qb, a[i].qe, a[i].score);
            if a[j_idx].rid != p_rid || p_rb >= a[j_idx].re + max_gap {
                break;
            }
            // Already killed by an earlier iteration: skip, do not stop.
            if a[j_idx].qe == a[j_idx].qb {
                j -= 1;
                continue;
            }
            // `or_` / `oq`: overlap length on reference and on query. Both can go negative when the
            // regions are disjoint, which is fine because the comparisons below then fail. `oq`
            // needs the branch because which end contributes depends on which region starts first;
            // `or_` does not, since the sort guarantees `a[j_idx].re <= a[i].re`.
            let or_ = a[j_idx].re - p_rb;
            let oq = if a[j_idx].qb < p_qb {
                a[j_idx].qe - p_qb
            } else {
                p_qe - a[j_idx].qb
            };
            // `mr` / `mq`: the shorter of the two regions on each axis. Normalizing the overlap by
            // the *minimum* length (not the maximum, not the sum) means a short region fully
            // contained in a long one always reads as 100% overlapping, which is the intent.
            let mr = (a[j_idx].re - a[j_idx].rb).min(p_re - p_rb);
            let mq = (a[j_idx].qe - a[j_idx].qb).min(p_qe - p_qb);
            // Redundancy test: overlapping by more than `mask_level_redun` (0.95) of the shorter
            // region on both axes means these are two descriptions of one alignment. See the
            // `mask_level_redun` note above for why this is f32 arithmetic.
            if or_ as f32 > mask_level_redun * mr as f32 && oq as f32 > mask_level_redun * mq as f32
            {
                if p_score < a[j_idx].score {
                    // `p` (the later region) loses. `break`, not `continue`: `p` is dead, so there
                    // is nothing left to compare it against (`bwamem.cpp:320`).
                    a[i].qe = a[i].qb;
                    break;
                }
                // `q` loses, but `p` survives and keeps scanning further back.
                a[j_idx].qe = a[j_idx].qb;
            } else if a[j_idx].rb < p_rb {
                // Not redundant but `q` starts strictly earlier: candidate for a collinear merge.
                // The `rb` test is what establishes `mem_patch_reg`'s `left.rb <= right.rb`
                // precondition, with `q` as the left region and `p` as the right one.
                // Snapshots of the two candidates, cloned only to satisfy the borrow checker:
                // `mem_patch_reg` needs two shared references into `a` while `a` stays mutable.
                // `q` is the left (earlier `rb`) region and `p` the right one, matching
                // `mem_patch_reg`'s precondition.
                let q = a[j_idx].clone();
                let p = a[i].clone();
                if let Some((score, w)) = mem_patch_reg(fm, opt, codes, &q, &p) {
                    // Merge `q` into `p`: `p` inherits `q`'s start coordinates and the union's
                    // score, takes the max of the suboptimal-score fields (they describe evidence
                    // against, so the strongest counter-evidence survives), and `q` is killed.
                    // `truesc` and `score` both become the *global* merged score, so the retry loop
                    // in `reg2aln` will later compare that global score against itself and converge
                    // immediately: merged regions never trigger a band retry.
                    //
                    // `+ 1` in the `n_comp` update mirrors `bwamem.cpp:325` exactly. It appears to
                    // double count (`q` contributes its own `n_comp` and then one more), but it is
                    // the C's arithmetic and `n_comp` only feeds heuristics, never SAM bytes.
                    a[i].n_comp += a[j_idx].n_comp + 1;
                    a[i].seedcov = a[i].seedcov.max(a[j_idx].seedcov);
                    a[i].sub = a[i].sub.max(a[j_idx].sub);
                    a[i].csub = a[i].csub.max(a[j_idx].csub);
                    a[i].qb = a[j_idx].qb;
                    a[i].rb = a[j_idx].rb;
                    a[i].truesc = score;
                    a[i].score = score;
                    a[i].w = w;
                    // The C writes `q->qb = q->qe` here (`bwamem.cpp:332`), the opposite assignment
                    // to the redundancy branch's `q->qe = q->qb`. Either way the region becomes
                    // zero-length and is compacted away, and nothing reads `qb`/`qe` of a dead
                    // region in between, so the two forms are interchangeable. We use the `qe =
                    // qb` form uniformly.
                    a[j_idx].qe = a[j_idx].qb;
                }
            }
            j -= 1;
        }
    }
    // ---- Pass 2: compact out the killed regions -----------------------------------------------
    // Compact out everything killed above (`bwamem.cpp:336`). The C shifts survivors down in place
    // and keeps the relative order, which is exactly `retain`'s contract.
    a.retain(|r| r.qe > r.qb);

    // ---- Pass 3: re-sort by score and drop exact coordinate duplicates ------------------------
    // Sort by score desc, then rb, then qb (`alnreg_slt`), again with bwa's unstable introsort.
    ks_introsort_by(&mut a, |x, y| {
        x.score > y.score || (x.score == y.score && (x.rb < y.rb || (x.rb == y.rb && x.qb < y.qb)))
    });
    // Exact-duplicate removal (`bwamem.cpp:343`). After the score sort, identical regions are
    // adjacent, so a single linear pass comparing each entry to its predecessor suffices. Note the
    // key here is (score, rb, qb) with no `re`/`qe`: two regions agreeing on all three but ending
    // differently are still treated as duplicates and the later one dies.
    for i in 1..a.len() {
        if a[i].score == a[i - 1].score && a[i].rb == a[i - 1].rb && a[i].qb == a[i - 1].qb {
            a[i].qe = a[i].qb;
        }
    }
    // The C's second compaction starts at `i = 1, m = 1` (`bwamem.cpp:347`), unconditionally
    // keeping `a[0]`. That is safe to express as a plain `retain` because the marking loop above
    // also starts at 1 and therefore can never kill `a[0]`.
    a.retain(|r| r.qe > r.qb);
    a
}

/// Assign each region a primary, or promote it to primary itself. Port of
/// `mem_mark_primary_se_core` (`bwamem.cpp:1392`).
///
/// Precondition: `a` is already sorted best-first (by `alnreg_hlt`, done by the caller). The
/// algorithm is a single greedy sweep: maintain `primaries` (the C's `z`), the list of primaries
/// found so far, and for each next region check whether it overlaps any existing primary
/// substantially on the query. If it does, it is a secondary of the *first* such primary; if not,
/// it becomes a primary itself.
///
/// "Substantially" means the overlap is at least `mask_level` (0.5) of the shorter region. The
/// query axis is what matters here, not the reference: two regions covering the same read bases are
/// competing explanations of that read regardless of where they land in the genome.
///
/// Side effects on `a[j]` for each primary `j` that absorbs a secondary `i`:
/// - `sub` is set to the first (i.e. best, given the sort) competing score, if not already set.
/// - `sub_n` counts competitors whose score is close enough to matter; it feeds the MAPQ penalty.
///
/// # Parameters
///
/// - `opt`: the alignment options. Read for the scoring parameters `a`, `b`, `o_del`, `e_del`,
///   `o_ins`, `e_ins` (which set the near-tie window) and `mask_level` (the overlap fraction, an
///   f32 in 0.0..=1.0, default 0.5) that decides what counts as "the same read bases".
/// - `a`: the read's regions, sorted best-first by the caller. Mutated in place: `secondary`,
///   `sub` and `sub_n` are written. `sub` must have been zeroed by the caller, since 0 is the
///   "unset" sentinel this function relies on.
fn mark_primary_core(opt: &MemOpt, a: &mut [MemAlnReg]) {
    // Region count, hoisted only because `a` is borrowed mutably inside the loop.
    let n = a.len();
    // ---- The "close enough to count as a real competitor" score window (the C's `tmp`) --------
    // The largest of a mismatch's swing (`a + b`, since a match becomes a mismatch) and either
    // gap's minimum cost (`o + e`). In other words, one single sequencing error or one 1bp indel.
    // A competitor within that of the primary could plausibly be the true alignment, so it dilutes
    // MAPQ.
    let mut near_tie_window = opt.a + opt.b;
    near_tie_window = near_tie_window
        .max(opt.o_del + opt.e_del)
        .max(opt.o_ins + opt.e_ins);

    // ---- Greedy sweep ------------------------------------------------------------------------
    // Region 0 is unconditionally primary: it is the best-scoring one after the sort.
    // Indices into `a` of the primaries found so far, in discovery order, which (given the
    // best-first sort) is also descending score order. Invariant at the top of each iteration: it
    // holds every region in `0..i` that no earlier primary absorbed, and it is never empty.
    let mut primaries: Vec<usize> = vec![0];
    for i in 1..n {
        // Fields of the candidate `a[i]`, copied out so the loop below can borrow `a` immutably
        // and then write to `a[j]`: query bounds, score, and whether it is an ALT-contig hit.
        let (i_qb, i_qe, i_score, i_alt) = (a[i].qb, a[i].qe, a[i].score, a[i].is_alt);
        // Index of the primary that claims `a[i]` as its secondary, or `None` while no primary has
        // been found to overlap it. Set at most once (the `break` below).
        let mut absorbing_primary = None;
        // Scans `primaries` in insertion order, i.e. best primary first, and takes the *first*
        // match, not the largest-overlap match. That `break` is load-bearing for parity.
        for &j in &primaries {
            // Intersection of the two query intervals: `[max(qb), min(qe))`. `b_max`/`e_min` are
            // the C's names for its two ends.
            let b_max = a[j].qb.max(i_qb);
            let e_min = a[j].qe.min(i_qe);
            if e_min > b_max {
                // `min_l`: the shorter of the two regions' query lengths.
                let min_l = (i_qe - i_qb).min(a[j].qe - a[j].qb);
                // f32: `min_l * opt->mask_level` is float arithmetic in C (see the
                // `mask_level_redun` note in `mem_sort_dedup_patch`).
                if (e_min - b_max) as f32 >= min_l as f32 * opt.mask_level {
                    // Only the first competitor sets `sub`, and since the array is sorted
                    // best-first that first one is the best competitor. `sub == 0` is the
                    // "unset" sentinel (`mem_mark_primary_se` zeroes it beforehand), which
                    // conflates "no competitor" with "a competitor that scored exactly 0";
                    // harmless, since a zero-scoring region is not a competitor in practice.
                    if a[j].sub == 0 {
                        a[j].sub = i_score;
                    }
                    // `(a[j].is_alt || !i_alt)`: a primary-assembly hit is not diluted by an ALT
                    // competitor, because an ALT contig is a known alternative representation of
                    // the same locus rather than a genuinely ambiguous second placement. An ALT
                    // primary, by contrast, is diluted by anything.
                    if a[j].score - i_score <= near_tie_window && (a[j].is_alt || !i_alt) {
                        a[j].sub_n += 1;
                    }
                    absorbing_primary = Some(j);
                    break;
                }
            }
        }
        // The C expresses this as `if (k == z->n) kv_push(...) else a[i].secondary = z->a[k]`,
        // using the loop variable's terminal value to detect "no break" (`bwamem.cpp:1415`). The
        // `Option` carries the same information.
        match absorbing_primary {
            Some(j) => a[i].secondary = j as i32,
            None => primaries.push(i),
        }
    }
}

/// Stamp `is_alt` on every region from its OWN contig, immediately after the dedup pass. Port of
/// the loop at `bwamem.cpp:1161-1170`.
///
/// This looks redundant with the flag the chain already carries (`bwamem.cpp:948`) and is not.
/// Extension moves a region's boundaries, `mem_sort_dedup_patch` merges regions together, and the
/// surviving region's `rid` is therefore not guaranteed to be the `rid` of the chain that seeded
/// it. The C stamps again, from the region, and so must we.
///
/// Deliberately NOT folded into [`mem_sort_dedup_patch`]: mate rescue also calls that function, and
/// there the C copies the flag from the ANCHOR (`b.is_alt = a->is_alt`, `bwamem_pair.cpp:218`)
/// rather than reading it from the rescued region's own contig. Stamping inside the dedup would
/// overwrite that with a different answer whenever a rescued mate lands on a contig of the other
/// kind.
///
/// # Parameters
///
/// - `bns`: the contig table, the only place `is_alt` lives.
/// - `regs`: one read's regions, post-dedup. Mutated in place. Regions with `rid < 0` are left
///   alone, matching the C's `p->rid >= 0` guard.
pub fn stamp_is_alt(bns: &BntSeq, regs: &mut [MemAlnReg]) {
    for r in regs.iter_mut() {
        if r.rid >= 0 && bns.contigs[r.rid as usize].is_alt {
            r.is_alt = true;
        }
    }
}

/// Mark primary/secondary regions and set `sub`/`sub_n`. Port of `mem_mark_primary_se`
/// (`bwamem.cpp:1420`) for the primary-assembly (non-ALT) case.
///
/// `id` is the *global* read index, `n_processed + i`, supplied by the worker. It must be global
/// and not batch-local: it seeds the hash tie-breaker, so a batch-local id would make output depend
/// on batch boundaries and thus on thread count.
///
/// Returns `n_pri`, the number of non-ALT regions. Callers use it to decide whether the ALT
/// re-ranking machinery is needed.
///
/// Both branches at `bwamem.cpp:1440` are implemented. When the index has ALT contigs and at least
/// one region landed on one (`n_pri < n`), the function re-sorts with `alnreg_hlt2` so the
/// primary-assembly hits form the prefix `0..n_pri`, rebuilds `secondary_all` through a rank
/// permutation, sets `secondary = INT_MAX` on the ALT hits, and re-runs the core over that prefix
/// alone. Without ALT contigs `n_pri == n` and only the cheap `else` branch runs, so an index with
/// no `.alt` file pays nothing for any of this.
///
/// # Parameters
///
/// - `opt`: the alignment options, forwarded to `mark_primary_core` for the scoring parameters and
///   `mask_level`.
/// - `a`: one read's regions, typically the output of [`mem_sort_dedup_patch`]. Mutated in place
///   and reordered: this function overwrites `sub`, `secondary`, `hash` and `secondary_all`, then
///   sorts by `alnreg_hlt`. Safe to call more than once on the same slice (PE rescue does).
/// - `id`: the *global* read index, `n_processed + i`, supplied by the worker. Must be global, not
///   batch-local: it seeds the hash tie-breaker, so a batch-local value would make output depend
///   on batch boundaries and hence on thread count.
///
/// # Returns
///
/// `n_pri`, the number of non-ALT regions in `a`, in `0..=a.len()`.
pub fn mem_mark_primary_se(opt: &MemOpt, a: &mut [MemAlnReg], id: u64) -> i32 {
    if a.is_empty() {
        return 0;
    }
    // Running count of non-ALT (primary-assembly) regions seen so far; the return value. Counted
    // during the reset loop because that loop already visits every region.
    let mut n_pri = 0;
    for (i, r) in a.iter_mut().enumerate() {
        // Reset before marking: this function can run more than once over the same regions (PE
        // rescue re-marks), and stale `sub` would defeat `mark_primary_core`'s `sub == 0` sentinel.
        r.sub = 0;
        r.secondary = -1;
        // `hash_64(id + i)` (`bwamem.cpp:1428`). `wrapping_add` because the C adds an `int64_t` id
        // to an `int` index with no overflow check; at realistic read counts this never wraps, but
        // the wrapping form is the faithful one.
        r.hash = hash_64(id.wrapping_add(i as u64));
        if !r.is_alt {
            n_pri += 1;
        }
    }
    // Debug aid: set `BWA3_DUMP_PRESORT=<read id>` to dump one read's regions immediately before
    // the hash sort, which is where parity investigations usually start. Unlike
    // `band_width_trace_enabled` in
    // cigar.rs this is not cached, but it runs once per read rather than once per emitted
    // alignment, so the `var_os` cost is tolerable.
    if std::env::var_os("BWA3_DUMP_PRESORT").is_some_and(|v| v.to_string_lossy() == id.to_string())
    {
        eprintln!("PRESORT id={id} n={}", a.len());
        for (i, r) in a.iter().enumerate() {
            eprintln!(
                "  pre{i} q[{},{}) r[{},{}) score={} hash={}",
                r.qb, r.qe, r.rb, r.re, r.score, r.hash
            );
        }
    }
    // Sort by (score desc, is_alt asc, hash asc), i.e. `alnreg_hlt` at `bwamem.cpp:155`, using
    // bwa's unstable introsort. The three keys in order mean: best score wins; among equals, a
    // primary-assembly hit outranks an ALT hit; among those, the hash decides. `!x.is_alt &&
    // y.is_alt` is the bool spelling of the C's `(a).is_alt < (b).is_alt` on an integer field.
    //
    // The hash key is what makes the unstable sort harmless here: it gives a total order, so there
    // are no ties left for introsort's instability to resolve differently. Contrast `alnreg_slt2`
    // in `mem_sort_dedup_patch` above, which keys on `re` alone and where instability genuinely
    // decides which alignment survives.
    ks_introsort_by(a, |x, y| {
        x.score > y.score
            || (x.score == y.score
                && (!x.is_alt && y.is_alt || (x.is_alt == y.is_alt && x.hash < y.hash)))
    });
    mark_primary_core(opt, a);
    let n = a.len();
    // First-round bookkeeping (`bwamem.cpp:1433-1439`), done BEFORE any re-ranking:
    //   - `secondary_all` temporarily holds this region's rank in the first round. It is not the
    //     final value; the ALT branch below rewrites it through that rank. Overloading one field
    //     for two meanings is the C's design, kept because the permutation below depends on it.
    //   - `alt_sc` records the score of the ALT region shadowing a primary-assembly one, which is
    //     the numerator's partner in the `pa:f` tag.
    for i in 0..n {
        a[i].secondary_all = i as i32;
        let sec = a[i].secondary;
        if !a[i].is_alt && sec >= 0 && a[sec as usize].is_alt {
            a[i].alt_sc = a[sec as usize].score;
        }
    }
    if n_pri < n as i32 {
        // ---- There is at least one ALT hit. Rank the primary-assembly hits on their own -------
        // Re-sort with ALT-ness as the FIRST key (`alnreg_hlt2`, `bwamem.cpp:158`) rather than the
        // third, so that every non-ALT region moves ahead of every ALT one and the primary
        // assembly occupies the prefix `0..n_pri`. Skipped when n_pri == 0, exactly as the C skips
        // it: with no primary-assembly hit there is no prefix to isolate, and re-sorting would
        // reorder the ALT hits for nothing.
        if n_pri > 0 {
            ks_introsort_by(a, |x, y| {
                (!x.is_alt && y.is_alt)
                    || (x.is_alt == y.is_alt
                        && (x.score > y.score || (x.score == y.score && x.hash < y.hash)))
            });
        }
        // `z[first_round_rank] = index_after_the_re-sort`. Built from `secondary_all`, which each
        // region carried through the sort, so it inverts the permutation the sort just applied.
        let mut z = vec![0i32; n];
        for i in 0..n {
            z[a[i].secondary_all as usize] = i as i32;
        }
        // `secondary` still points at FIRST-ROUND indices, so it has to be mapped through `z` to
        // stay meaningful. `secondary_all` keeps the full ranking (ALT hits included), which is
        // what `mem_gen_alt` and the PE primary/secondary swap read; `secondary` is then reserved
        // for the primary-assembly-only ranking recomputed just below.
        for i in 0..n {
            if a[i].secondary >= 0 {
                a[i].secondary_all = z[a[i].secondary as usize];
                // An ALT hit is excluded from the primary-assembly ranking altogether. INT_MAX,
                // not -1: -1 means "unshadowed primary" and would promote the ALT hit instead.
                if a[i].is_alt {
                    a[i].secondary = i32::MAX;
                }
            } else {
                a[i].secondary_all = -1;
            }
        }
        if n_pri > 0 {
            // Re-run the marking over the primary-assembly prefix ONLY, so that `sub` (and hence
            // MAPQ) reflects competition within the primary assembly and is not diluted by an ALT
            // representation of the same locus. `sub_n` is deliberately NOT reset here: the C
            // resets only `sub` and `secondary` (`bwamem.cpp:1455`).
            for r in a[..n_pri as usize].iter_mut() {
                r.sub = 0;
                r.secondary = -1;
            }
            mark_primary_core(opt, &mut a[..n_pri as usize]);
        }
    } else {
        // No ALT contigs: `secondary_all` mirrors `secondary` (the C's else-branch at
        // `bwamem.cpp:1458`). `mem_gen_alt`/the PE swap read `secondary_all` separately.
        for r in a.iter_mut() {
            r.secondary_all = r.secondary;
        }
    }
    n_pri
}

/// Approximate mapping quality of a primary region, 0..=60. Port of `mem_approx_mapq_se`
/// (`bwamem.cpp:1470`).
///
/// MAPQ answers "how confident are we that this is the right locus", so the formula is driven by
/// the *margin* between this alignment's score and the best competing score, scaled down by how
/// good the alignment is in absolute terms and by how repetitive the seeds were.
///
/// # Parameters
///
/// - `opt`: the alignment options. Read for `min_seed_len` and `a` (which set the no-competitor
///   score floor), `b` (the mismatch penalty, used to convert a score deficit into an identity),
///   and `mapq_coef_len` / `mapq_coef_fac` (the length-discount curve, 50 and the *truncated
///   integer* 3 by default; see `mapq_uses_truncated_int_coef_fac`).
/// - `a`: the region to score. Read-only. Expects `sub`, `csub`, `sub_n` and `frac_rep` to have
///   been filled in by [`mem_mark_primary_se`] and the chaining stage; called on primary regions,
///   secondaries get their MAPQ elsewhere.
///
/// # Returns
///
/// The phred-scaled mapping quality, clamped to 0..=60. 0 means "this placement is arbitrary",
/// either because a competitor scored at least as well or because the repeat scaling zeroed it.
#[must_use]
pub fn mem_approx_mapq_se(opt: &MemOpt, a: &MemAlnReg) -> u32 {
    // ---- The competing score -----------------------------------------------------------------
    // `sub` is the best overlapping region's score (from
    // `mark_primary_core`); when there is none, fall back to `min_seed_len * a`, the score a bare
    // minimum-length exact seed would get. That floor matters: without it a read with no competitor
    // would get an unboundedly large margin and a maxed MAPQ purely for being unopposed.
    let mut sub = if a.sub != 0 {
        a.sub
    } else {
        opt.min_seed_len * opt.a
    };
    // `csub` is the best competitor from within the same chain; take whichever evidence is stronger.
    sub = sub.max(a.csub);
    // No margin at all: the competitor is as good as us, so the placement is arbitrary.
    if sub >= a.score {
        return 0;
    }
    // ---- Alignment length and score-derived identity -----------------------------------------
    // `l`: alignment length, taken as the longer of the query and reference spans.
    //
    // UNVERIFIED (narrow, believed unreachable): the C computes `a->qe - a->qb > a->re - a->rb`
    // with `a->re`/`a->rb` as `int64_t`, so the comparison happens in 64 bits and only the result
    // is narrowed to `int l`. Here the reference span is cast to `i32` *before* the comparison. The
    // two differ only if the reference span exceeds `i32::MAX`, which a read-length alignment
    // cannot produce.
    let l = (a.qe - a.qb).max((a.re - a.rb) as i32);
    // Fractional identity, derived from the score rather than from the alignment: `l * opt.a` is
    // the perfect score, so `l * opt.a - score` is the deficit, and dividing by `(a + b)` converts
    // that deficit into an equivalent number of mismatches (a mismatch costs `a + b`: you lose the
    // match reward and pay the mismatch penalty). Dividing again by `l` turns a mismatch count into
    // a mismatch rate. Gaps get charged at the mismatch rate too, which is why this is approximate.
    let identity = 1.0 - f64::from(l * opt.a - a.score) / f64::from(opt.a + opt.b) / f64::from(l);
    // ---- The MAPQ formula itself --------------------------------------------------------------
    // The running MAPQ in phred units. Unclamped and possibly negative until the `clamp` below,
    // because the `sub_n` penalty is subtracted first.
    let mut mapq: i32;
    if a.score == 0 {
        mapq = 0;
    } else {
        // `mapQ_coef_len > 0` always holds for the default options (it is set to 50 at
        // `bwamem.cpp:140`), so only the C's first branch (`bwamem.cpp:1480`) is ported. The
        // `else` branch at `bwamem.cpp:1485`, which uses `MEM_MAPQ_COEF` and `log(seedcov)`, is
        // dead code under any reachable option set.
        //
        // Length discount (the C's first `tmp`): short alignments (below 50bp) get no discount,
        // longer ones get `log(50) / log(l)`, which decays slowly. The intuition is that a long
        // alignment needs a proportionally larger score margin to be equally convincing.
        let len_discount = if f64::from(l) < opt.mapq_coef_len {
            1.0
        } else {
            opt.mapq_coef_fac / f64::from(l).ln()
        };
        // Squared here and squared again in the line below, so identity enters to the *fourth*
        // power and the length term to the second. That steep falloff is deliberate: a
        // low-identity alignment loses MAPQ fast. See the `mapq_uses_truncated_int_coef_fac` test
        // for why `mapq_coef_fac` must be the truncated integer 3 and not `ln(50) = 3.912`.
        //
        // The C reuses `tmp` for this product; we give the two values distinct names.
        let weight = len_discount * identity * identity;
        mapq = (MAPQ_SCORE_COEF * f64::from(a.score - sub) / f64::from(opt.a) * weight * weight
            + ROUND_BIAS) as i32;
    }
    // ---- Penalties and clamping ---------------------------------------------------------------
    // Penalty for multiple near-equal competitors: subtracts `10 * log10(sub_n + 1)` phred points,
    // so two competitors cost about 3 and ten cost 10. Applied before the clamp, so a large
    // penalty can drive `mapq` negative and the clamp catches it.
    if a.sub_n > 0 {
        mapq -= (MAPQ_SUB_N_COEF * f64::from(a.sub_n + 1).ln() + ROUND_BIAS) as i32;
    }
    // bwa's own ceiling, not a SAM one. Must happen before the `frac_rep` scaling below,
    // otherwise a large pre-clamp value would survive the multiplication (`bwamem.cpp:1490`).
    mapq = mapq.clamp(0, MAPQ_MAX);
    // Final scale-down by the fraction of the read covered by repetitive seeds: a read whose seeds
    // were all repeats gets MAPQ 0 no matter how clean the alignment looked.
    //
    // `frac_rep` is `float` in C and f32 here; `f64::from` widens it exactly, and the C's implicit
    // float-to-double promotion at `bwamem.cpp:1492` does the same, so this one is safe.
    mapq = (f64::from(mapq) * (1.0 - f64::from(a.frac_rep)) + ROUND_BIAS) as i32;
    mapq as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_64_is_wang() {
        // Distinct inputs give distinct, well-mixed outputs (sanity).
        assert_ne!(hash_64(0), hash_64(1));
        assert_ne!(hash_64(1000), hash_64(1001));
    }

    /// Build a minimal region for MAPQ tests: one full-length, forward-strand hit on rid 0 with no
    /// repeat content, so only the scoring fields vary.
    ///
    /// # Parameters
    ///
    /// - `score`: the region's alignment score, also copied into `truesc`.
    /// - `sub`: best competing overlapping score; 0 means "no competitor" (the sentinel).
    /// - `sub_n`: number of near-tie competitors, feeding the MAPQ penalty.
    /// - `seedcov`: read bases covered by supporting seeds. Unused by the reachable MAPQ branch,
    ///   set only so the struct is complete.
    /// - `len`: alignment length in bp, used as both the query span (`qb..qe`) and the reference
    ///   span (`rb..re`) so that MAPQ's `l` equals it exactly.
    fn reg(score: i32, sub: i32, sub_n: i32, seedcov: i32, len: i32) -> MemAlnReg {
        MemAlnReg {
            rb: 0,
            re: i64::from(len),
            qb: 0,
            qe: len,
            rid: 0,
            score,
            truesc: score,
            sub,
            csub: 0,
            sub_n,
            seedcov,
            seedlen0: 0,
            secondary: -1,
            secondary_all: -1,
            w: 0,
            frac_rep: 0.0,
            is_alt: false,
            alt_sc: 0,
            hash: 0,
            n_comp: 0,
        }
    }

    #[test]
    fn mapq_uses_truncated_int_coef_fac() {
        // Oracle read `_96f`: score=145 sub=140 sub_n=1 seedcov=323 l=150 frac_rep=0 -> mapq=8.
        // bwa-mem2's `mapQ_coef_fac` is an int = (int)log(50) = 3, not the float 3.912; the float
        // yields 15 here, the int yields 8.
        let opt = MemOpt::default();
        let a = reg(145, 140, 1, 323, 150);
        assert_eq!(mem_approx_mapq_se(&opt, &a), 8);
    }
}

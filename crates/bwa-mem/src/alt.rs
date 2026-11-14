//! Alternate-hit (`XA:Z:`) generation, port of `mem_gen_alt` (bwamem_extra.cpp).
//!
//! Runs after `mem_mark_primary_se`. For each region it decides, via `get_pri_idx`, whether the
//! region is a near-primary alternate (score within `xa_drop_ratio` of its primary) and, if so,
//! groups it under that primary. Primaries with too many alternates (> `max_xa_hits`) are dropped.
//! The reference has no ALT contigs, so `secondary_all == secondary` and `has_alt` is always false.
//!
//! # What this file is responsible for
//!
//! A read can often be placed in more than one spot in the genome. SAM reports ONE placement per
//! record, and lists the plausible runners-up in an optional tag called `XA:Z:`, a semicolon
//! separated list of `contig,±position,cigar,edit_distance` entries. This file decides which
//! regions become such runners-up and formats the tag. It runs late: after `primary.rs` has marked
//! which region is primary, and (for paired reads) after `pe.rs` has done its primary/secondary
//! swap, because both of those decide what "the primary" even is.
//!
//! # Suggested reading order
//!
//! 1. [`get_pri_idx`], the one-region test: is this region a plausible runner-up, and to whom?
//! 2. [`xa_group`], the two-pass grouping and the per-primary cap.
//! 3. [`mem_gen_alt`], which turns each group into the actual `XA:Z:` bytes.
//!
//! # Glossary of names kept identical to the C
//!
//! | Name             | Plain language                                                           |
//! |------------------|-------------------------------------------------------------------------|
//! | `secondary`      | Index of the region this one is a runner-up to, or -1 if it is primary.  |
//! | `secondary_all`  | Same idea, but the copy maintained for XA grouping (see [`get_pri_idx`]). |
//! | `cnt`            | Per-primary count of runners-up (the C's `cnt[]`).                       |
//! | `has_alt`        | Whether a primary has at least one runner-up on an ALT contig.           |
//! | `l_query`        | Read length in bases.                                                    |
//! | `NM` / `nm`      | Edit distance: mismatches plus inserted plus deleted bases.              |
//! | `rid`            | Contig index (a "reference id"), indexing `BntSeq::contigs`.             |

//! # Rust mechanics used in this file
//!
//! `get_pri_idx` returns `Option<usize>`: either this region hangs off a primary, and here is that
//! primary's index, or it does not. The index is into the SAME slice the function was given, which is
//! a convention a raw `usize` cannot express and the doc comment therefore states explicitly. A
//! reference to the primary would have been the C-shaped answer and is not available here: the slice
//! is being examined while the caller intends to mutate it, and Rust will not hand out a borrow that
//! outlives that intent. An index is the standard way through, and it is checked at every use.
//!
//! The tag builder returns `Vec<Option<String>>`, one slot per region, `None` where the region has no
//! alternates. Same reasoning as the CIGAR module's `xa` field: a missing tag and an empty tag are
//! different bytes in the output, so they must be different values in the program.
//!
//! `xa_drop_ratio` is an `f64` compared against a ratio of `i32` scores. The C does the same
//! comparison in `double`, and the cast happens at the same point here, because a ratio that lands
//! exactly on the threshold decides whether a record carries an alternate at all.

use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

use crate::cigar::reg2aln;
use crate::MemAlnReg;

/// `get_pri_idx`: the primary index region `i` is an XA alternate of, or `None`. Port of
/// `bwamem_extra.cpp:122-127`.
///
/// A region qualifies as an alternate of its primary when two things hold: it actually has one
/// (`secondary_all >= 0`, set by `mem_mark_primary_se`), and its score is still within
/// `xa_drop_ratio` of that primary's. The second test is what keeps `XA:Z:` from filling up with
/// junk: a hit scoring far below the primary is not a plausible alternative placement of the read,
/// it is just noise.
///
/// WHY `secondary_all` AND NOT `secondary`: the two fields agree after single-end marking on an
/// ALT-free index (`mem_mark_primary_se` copies one into the other), but they diverge afterwards
/// and each is maintained for a different consumer.
///
/// `secondary` is the *emission* ranking and gets overwritten downstream: `mem_mark_primary_se`'s
/// ALT branch stores `INT_MAX` in it for ALT hits (`bwamem.cpp:1450`), and the paired-end rescue
/// sets it to the sentinel `-2` on the region it promotes (`pe.rs:1640`). Both values would fail
/// this function's `primary_idx >= 0` test or index out of bounds.
///
/// `secondary_all` is the *grouping* ranking, maintained specifically so XA attribution stays
/// correct. It is not simply the frozen pre-swap value: the PE path deliberately rewrites it
/// (`pe.rs:1662-1670`) so that a region promoted to primary by pairing collects its former
/// siblings as XA hits. `mem_gen_alt` is the consumer that rewrite exists to serve.
///
/// # Parameters
///
/// - `xa_drop_ratio`: score fraction a region must keep, relative to its primary, to be worth
///   listing. Unitless, valid range (0, 1], default 0.8. Supplied by the caller as
///   `f64::from(opt.xa_drop_ratio)` (bwa's `-r`-adjacent `XA_drop_ratio`); already widened to `f64`
///   here so the comparison below matches the C's promotion exactly.
/// - `regs`: the read's alignment regions, in the order and with the `secondary_all`/`score` values
///   left by `mem_mark_primary_se` (and, for pairs, by the PE swap). Precondition: every
///   non-negative `secondary_all` in it is a valid index into this same slice; the cast on line 76
///   would panic otherwise.
/// - `i`: index into `regs` of the single region being tested. Range `0..regs.len()`.
///
/// # Returns
///
/// `Some(p)` where `p` indexes the SAME `regs` slice and names the primary this region should be
/// listed under, or `None` when the region is itself primary or scores too far below its primary.
///
/// PRECISION: the C compares `int >= int * double`, so the left side is promoted to `double` and
/// the product is evaluated in `double`. Converting both sides with `f64::from` reproduces that
/// exactly. Doing the comparison in `f32`, or rearranging it into an integer form, would move the
/// boundary cases and break byte parity.
#[must_use]
fn get_pri_idx(xa_drop_ratio: f64, regs: &[MemAlnReg], i: usize) -> Option<usize> {
    // Candidate primary for region `i`, as an `i32` because it is a sentinel-carrying field:
    // >= 0 is a real region index, < 0 means "region `i` is itself primary" (-1) or one of the
    // downstream sentinels described above. Kept signed until the >= 0 test has ruled those out.
    let primary_idx = regs[i].secondary_all;
    if primary_idx >= 0
        && f64::from(regs[i].score) >= f64::from(regs[primary_idx as usize].score) * xa_drop_ratio
    {
        Some(primary_idx as usize)
    } else {
        None
    }
}

/// For each region, the primary it is emitted under as an `XA` hit, or `None`. Pure port of
/// `mem_gen_alt`'s selection (the string formatting is done by the caller via `reg2aln`).
///
/// NOTATION: the C's `r` (the index of a primary region) is spelled `primary_idx` below; the quoted
/// C expressions in this comment keep the C's spelling so they can be grepped in the source.
///
/// TWO PASSES, and they cannot be fused (`bwamem_extra.cpp:140-153`). The cap test in the second
/// pass needs `cnt[r]`, the TOTAL number of alternates under primary `r`, which is only known once
/// every region has been visited. A single fused pass would emit the first few alternates of a
/// primary before discovering the primary blows the cap, and the C's semantics are all-or-nothing:
/// a primary over the cap contributes no `XA` hits at all rather than a truncated list.
///
/// THE CAP IS ASYMMETRIC by design. `max_xa_hits_alt` (bwa's `-h` second value, default 200) is the
/// generous limit; `max_xa_hits` (default 5) is the strict one. A primary that has at least one
/// alternate on an ALT contig (`has_alt[r]`) gets the generous limit, because ALT contigs are
/// expected to produce many near-identical placements and truncating them would hide real haplotype
/// alternatives. Everything else gets the strict limit. Note the C's operator precedence, preserved
/// here: `cnt[r] > max_xa_hits_alt || (!has_alt[r] && cnt[r] > max_xa_hits)`, so the generous limit
/// is a hard ceiling that applies even to ALT-bearing primaries.
///
/// INVARIANT: the returned vector has one entry per region, indexed the same way as `regs`. A
/// primary's hits are emitted in ascending region order, which is the order of the
/// semicolon-separated fields in its `XA:Z:` string, so it is output-visible.
///
/// The C additionally short-circuits on `tot == 0` (`bwamem_extra.cpp:147`); we simply return the
/// all-empty vector, which the caller skips region by region. Same result, no allocation to avoid.
///
/// # Parameters
///
/// - `opt`: the run's alignment options. Only three fields are read here: `xa_drop_ratio` (see
///   [`get_pri_idx`]), `max_xa_hits` (bwa `-h` first value, default 5, count of alternates) and
///   `max_xa_hits_alt` (bwa `-h` second value, default 200). Same object for every read, supplied
///   by the CLI once at startup.
/// - `regs`: the read's alignment regions after primary marking. Not mutated. Its length fixes the
///   length of every vector below and of the return value.
///
/// # Returns
///
/// One entry per region, indexed identically to `regs`: `Some(p)` when the region is emitted as an
/// `XA` hit under primary `p`, `None` when it is not a runner-up or its primary's group blew the
/// cap. Two vectors in all: a list per primary was one more allocation per primary.
pub fn xa_group(opt: &MemOpt, regs: &[MemAlnReg]) -> Vec<Option<usize>> {
    // Score-drop threshold, widened once here so both passes hand `get_pri_idx` the identical
    // `f64` and cannot disagree about a boundary case. `n_regs` is the region count, which is also
    // the length of `tally` and of the returned vector.
    let ratio = f64::from(opt.xa_drop_ratio);
    let n_regs = regs.len();
    // `tally[p]`: the C's `cnt[p]`, how many regions name `p` as their XA primary, and `has_alt[p]`,
    // whether any of them sits on an ALT contig, which selects the generous cap below. Written only
    // by pass 1, so no cap decision may read it before that loop ends. The count is `i32` because it
    // is compared against the `i32` option fields.
    let mut tally = vec![(0i32, false); n_regs];
    // ---- pass 1: each region's primary, and how many runners-up each primary has ----
    // `get_pri_idx` is a pure function of `regs`, so evaluating it once here is the C's second
    // evaluation too.
    let mut listed: Vec<Option<usize>> = (0..n_regs).map(|i| get_pri_idx(ratio, regs, i)).collect();
    for (i, p) in listed.iter().enumerate() {
        if let Some(p) = *p {
            tally[p].0 += 1;
            tally[p].1 |= regs[i].is_alt;
        }
    }
    // ---- pass 2: drop the runners-up of a primary over the cap, now that every count is final ----
    for p in &mut listed {
        if let Some(q) = *p {
            let (cnt, has_alt) = tally[q];
            if cnt > opt.max_xa_hits_alt || (!has_alt && cnt > opt.max_xa_hits) {
                *p = None;
            }
        }
    }
    listed
}

/// `HN:i` of each region under `--meth`: how many alternative hits fall within `xa_drop_ratio` of
/// it, counted before the `-h` cap drops them from XA (fg-labs/bwa-mem3 `mem_gen_alt`'s `cnt`).
pub fn hit_counts(opt: &MemOpt, regs: &[MemAlnReg]) -> Vec<i32> {
    let ratio = f64::from(opt.xa_drop_ratio);
    let mut cnt = vec![0i32; regs.len()];
    for i in 0..regs.len() {
        if let Some(primary_idx) = get_pri_idx(ratio, regs, i) {
            cnt[primary_idx] += 1;
        }
    }
    cnt
}

/// Per-region `XA:Z:` string (`rname,±pos,cigar,NM;`... concatenated), or `None` for regions that
/// are not XA primaries. Port of `mem_gen_alt`: groups near-primary alternates and formats each
/// via `reg2aln`. Runs after `mem_mark_primary_se` (and, in PE, after the primary/secondary swap).
///
/// FIELD LAYOUT of each semicolon-terminated record, matching `bwamem_extra.cpp:156-171` field for
/// field: contig name, comma, strand character, 1-based position, comma, CIGAR, comma, edit
/// distance `NM`, semicolon. There is a trailing `;` after the LAST hit too, not a separator, which
/// is why the loop pushes it unconditionally.
///
/// # Parameters
///
/// - `fm`: the FM index, needed only so `reg2aln` can resolve suffix-array positions back to
///   reference coordinates. Read-only, shared by all threads.
/// - `bns`: the reference metadata (contig names, lengths, offsets). Read for `contigs[rid].name`,
///   the first field of every emitted record, and by `reg2aln` to convert a concatenated-genome
///   coordinate into a contig plus offset.
/// - `opt`: the run's alignment options, forwarded to [`xa_group`] for the drop ratio and caps and
///   to `reg2aln` for the scoring parameters it re-aligns with.
/// - `regs`: the read's regions after `mem_mark_primary_se` (and, in PE, after the primary/secondary
///   swap). Precondition: primary marking has already run, otherwise `secondary_all` is meaningless.
/// - `l_query`: read length in bases, > 0. Forwarded untouched to `reg2aln`.
/// - `query`: the read's bases as 2-bit codes (0..=3, 4 for `N`), length `l_query`, forward strand.
///   Forwarded untouched. `reg2aln` needs it because CIGAR and NM are not cached on `MemAlnReg`, so
///   every XA hit costs one re-alignment.
///
/// # Returns
///
/// Per region, read by its index in `regs` ([`XaTags`]): `Some(string)` for a region that is an XA
/// primary with at least one surviving alternate, the string being the concatenated
/// semicolon-terminated records (the value of the `XA:Z:` tag, tag prefix not included); `None`
/// everywhere else.
///
/// `aln.pos + 1` converts `reg2aln`'s 0-based contig-local position to SAM's 1-based convention, the
/// same `kputl(p->pos + 1, &str)` the C writes.
///
/// The `which = 0` passed to `add_cigar` is not a placeholder, it is what makes the shared helper
/// agree with the C here. `mem_gen_alt` does NOT call the C's `add_cigar`; it inlines its own loop
/// over `"MIDSHN"` with no clip rewriting at all (`bwamem_extra.cpp:165-168`). Passing `which = 0`
/// disables our helper's soft-to-hard clip rewrite, so the two produce identical bytes for every
/// op that can actually occur (0..=4). XA hits are always written with soft clips, whatever the
/// enclosing record does.
///
/// The `expect("ASCII XA")` is sound because every byte pushed comes from a contig name, a digit, or
/// one of `+-,;MIDSHN`. It is an assertion about the reference's naming, not a fallible parse.
pub fn mem_gen_alt(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    regs: &[MemAlnReg],
    l_query: i32,
    query: &[u8],
) -> XaTags {
    // Selection is done first and in full: `xa_group` names, cap checked, the primary each region is
    // listed under. `out` is the answer under construction, one slot per region, filled only at the
    // indices that turn out to be XA primaries; every other slot stays `None`.
    // TRIED AND MEASURED AT ZERO (2026-09-10): skipping this whole function when no region has
    // `secondary_all >= 0` -- i.e. when the read's hits are all primaries, which is most reads, so
    // no region can appear under anyone's `XA:Z`. The scan is O(n) and it removes `xa_group`'s
    // three vectors plus `out`, about four allocations per read: `sam_emit` fell from 3,404,012
    // allocations to 2,023,572 on 200k pairs, -41%, and the whole run from 12.55M to 11.17M.
    //
    // The wall did not move. Five interleaved reps of the `sam_emit` stage, sorted: 0.282 0.290
    // 0.291 0.293 0.298 without it against 0.283 0.287 0.287 0.291 0.307 with it, which is a tie.
    // The allocator is not what this stage is spending its time on -- the -10.6% that the `ptmp`
    // clones and the per-pair buffers bought came from the COPYING and the growth memcpys, not from
    // the malloc calls. Reverted then because it returned a short vector, a trap for callers that
    // index it. Kept since 2026-11-10 in the full-length form below: on Linux (N2, EPYC) mimalloc
    // is ~1.3% of the profile against the fork's ~0.45%, and this is 3 of ~21 allocations per read.
    //
    // No region close enough to the one shadowing it (most reads): no XA anywhere, so skip
    // `xa_group`'s vectors. Its own test, so exact. The answer keeps one slot per region.
    let ratio = f64::from(opt.xa_drop_ratio);
    if (0..regs.len()).all(|i| get_pri_idx(ratio, regs, i).is_none()) {
        return XaTags::default();
    }
    // Every primary's tag is built in one pass over the regions: walking them in ascending order
    // appends each primary's records in ascending order, the C's order. Each tag holds complete
    // records, each ending in its own `;`, so a record can always simply be appended.
    let mut out: Vec<Option<Vec<u8>>> = vec![None; regs.len()];
    for (alt_idx, primary_idx) in xa_group(opt, regs).into_iter().enumerate() {
        let Some(primary_idx) = primary_idx else {
            continue;
        };
        let xa_bytes = out[primary_idx].get_or_insert_with(Vec::new);
        // `aln` carries the re-derived CIGAR, contig `rid`, 0-based `pos` and edit distance `nm`
        // for this one alternate placement.
        let aln = reg2aln(fm, bns, opt, l_query, query, &regs[alt_idx]);
        xa_bytes.extend_from_slice(bns.contigs[aln.rid as usize].name.as_bytes());
        xa_bytes.push(b',');
        xa_bytes.push(if aln.is_rev { b'-' } else { b'+' });
        crate::emit::push_int(xa_bytes, aln.pos + 1);
        xa_bytes.push(b',');
        crate::pe::add_cigar(&aln.cigar, false, xa_bytes);
        xa_bytes.push(b',');
        crate::emit::push_int(xa_bytes, i64::from(aln.nm));
        xa_bytes.push(b';');
    }
    // Same-layout map, so the vector is converted in place.
    XaTags(
        out.into_iter()
            .map(|b| b.map(|b| String::from_utf8(b).expect("ASCII XA")))
            .collect(),
    )
}

/// [`mem_gen_alt`]'s per-region `XA:Z` values, read by region index. Empty, with no allocation,
/// when no region has one (most reads), which is why it is read through [`get`](Self::get) and
/// [`take`](Self::take) and not indexed.
#[derive(Default)]
pub struct XaTags(Vec<Option<String>>);

impl XaTags {
    /// Region `i`'s value, copied.
    pub fn get(&self, i: usize) -> Option<String> {
        self.0.get(i).cloned().flatten()
    }

    /// Region `i`'s value, moved out.
    pub fn take(&mut self, i: usize) -> Option<String> {
        self.0.get_mut(i).and_then(Option::take)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal region carrying only the fields `xa_group` reads.
    ///
    /// # Parameters
    ///
    /// - `score`: the region's alignment score, the quantity the `xa_drop_ratio` test compares.
    ///   Also copied into `truesc` so the region is internally consistent.
    /// - `secondary`: index of the primary this region is a runner-up to, or -1 for a primary.
    ///   Stored into BOTH `secondary` and `secondary_all`; the test does not exercise the paths
    ///   where those two diverge. `xa_group` reads only `secondary_all`.
    ///
    /// Every other field is zeroed and is not read
    /// by `xa_group`; in particular `is_alt = false` means the strict `max_xa_hits` cap applies.
    fn reg(score: i32, secondary: i32) -> MemAlnReg {
        MemAlnReg {
            rb: 0,
            re: 0,
            qb: 0,
            qe: 0,
            rid: 0,
            score,
            truesc: score,
            sub: 0,
            csub: 0,
            sub_n: 0,
            seedcov: 0,
            seedlen0: 0,
            secondary,
            secondary_all: secondary,
            w: 0,
            frac_rep: 0.0,
            is_alt: false,
            alt_sc: 0,
        }
    }

    #[test]
    fn near_primary_alternate_grouped_distant_one_dropped() {
        // Read _217: primary score 140, alternate 116 (>= 140*0.8=112 -> XA),
        // third 98 (< 112 -> not XA). All secondary to region 0.
        let opt = MemOpt::default();
        let regs = [reg(140, -1), reg(116, 0), reg(98, 0)];
        assert_eq!(xa_group(&opt, &regs), [None, Some(0), None]);
    }

    #[test]
    fn too_many_hits_drops_all_of_that_primary() {
        // 6 alternates all within ratio -> cnt=6 > max_xa_hits(5) -> group dropped entirely.
        let opt = MemOpt::default();
        let mut regs = vec![reg(150, -1)];
        for _ in 0..6 {
            regs.push(reg(140, 0));
        }
        assert!(
            xa_group(&opt, &regs).iter().all(Option::is_none),
            "primary with >5 alternates emits no XA"
        );
    }
}

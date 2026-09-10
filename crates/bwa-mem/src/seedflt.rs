//! `mem_flt_chained_seeds` (`bwamem.cpp:472-508`) and its helper `mem_seed_sw`
//! (`bwamem.cpp:401-427`): the per-seed Smith-Waterman filter bwa runs over the surviving chains,
//! after `mem_chain_flt` and before any extension.
//!
//! WHY IT WAS MISSING, AND WHAT IT COST. The filter disables itself on short reads, and it does so
//! by a test on the read length alone: `min_l > MEM_SEEDSW_COEF * l_query`, i.e.
//! `5.5 * ln(l_query) > 0.05 * l_query`, which is true for every read below roughly 690 bp. All of
//! this project's fixtures are 150 bp, so no test could see the filter's absence, and none did:
//! it was found by aligning a 3 kb read and watching `XS` come out at 24 where both bwa-mem2 2.3
//! and bwa 0.7.19 print 0. The extra `XS` was a 47-base region built from a 20-base seed that bwa
//! had thrown away here, a locus this read does not belong to.
//!
//! WHAT IT DOES. A chained seed is an exact match, so its length is a lower bound on what it is
//! worth, but a SHORT exact match in a long read is usually noise: at 3 kb a 20-mer occurs by
//! chance. The filter re-scores every seed shorter than [`MEM_SHORT_LEN`] by a local alignment of
//! its own small neighbourhood (the seed plus [`MEM_SHORT_EXT`] on each side, query and reference)
//! and drops it unless that alignment reaches `opt.a * min_l`. Seeds too long to be re-scored, and
//! seeds whose neighbourhood grew past [`MEM_SHORT_LEN`], are kept unconditionally: `mem_seed_sw`
//! returns `-1` for them and `-1` is a keep, not a reject.
//!
//! IT ALSO WRITES `score`. A surviving seed keeps its SW score, or takes `len * opt.a` if the SW
//! was skipped. That is not bookkeeping: `mem_chain2aln` sorts a chain's seeds by
//! `score << 32 | index` (`bwamem.cpp:2189`), so the filter changes the ORDER seeds are extended
//! in as well as which ones exist. Before this ran, every seed carried `score = len` from chaining.

use bwa_chain::MemChain;
use bwa_core::MemOpt;
use bwa_extend::ksw_align2;
use bwa_index::{BntSeq, FmIndex};

/// `MEM_SHORT_EXT` (`bwamem.cpp:230`): how far past each end of a seed the re-scoring window
/// reaches, on the query and on the reference alike.
const MEM_SHORT_EXT: i64 = 50;

/// `MEM_SHORT_LEN` (`bwamem.cpp:231`): the length at which a seed, or a re-scoring window, is
/// considered long enough to trust without a Smith-Waterman.
const MEM_SHORT_LEN: i64 = 200;

/// `MEM_HSP_COEF` (`bwamem.cpp:233`), the multiplier on `-W` when the user sets one.
///
/// Declared `1.1f` in the C, so the value that reaches the arithmetic is the FLOAT nearest 1.1 and
/// not the double nearest it. Writing `1.1_f64` here would make `min_HSP_score` differ by one on
/// the exact `-W` values that sit on the rounding boundary, which is a byte-parity break for a
/// constant nobody would think to check. `f64::from(1.1_f32)` reproduces the C promotion exactly.
const MEM_HSP_COEF: f32 = 1.1;

/// `MEM_MINSC_COEF` (`bwamem.cpp:234`), the multiplier on `ln(l_query)` in the default case.
/// 5.5 is exact in both float and double, so the promotion is a no-op here; it is still written as
/// an `f32` so the three coefficients read as the three the C declares.
const MEM_MINSC_COEF: f32 = 5.5;

/// `MEM_SEEDSW_COEF` (`bwamem.cpp:235`): the filter runs only while `min_l` stays under this
/// fraction of the read length, which is what confines it to long reads. `0.05f` is NOT exact in
/// binary, so the same promotion argument as [`MEM_HSP_COEF`] applies to the comparison.
const MEM_SEEDSW_COEF: f32 = 0.05;

/// `ksw_align2`'s SIMD width for this call. `mem_seed_sw` passes `KSW_XSTART` with no `KSW_XBYTE`,
/// and `ksw_align2` reads that as `ksw_qinit(2, ...)` (`ksw.cpp:357`), the i16 kernel, which is
/// 8 lanes. The width is not a performance knob: it sets the query-profile padding and therefore
/// the score.
const KSW_I16_LANES: usize = 8;

/// Local-alignment score for one seed's neighbourhood, or `-1` when the seed is exempt.
/// Port of `mem_seed_sw` (`bwamem.cpp:401-427`).
///
/// # Parameters
///
/// - `fm`, `bns`: the reference; `fm` unpacks the window's bases, `bns` clamps it to one contig.
/// - `opt`: scoring matrix and gap penalties, handed straight to ksw.
/// - `query`: the whole read as nt4 codes; the window is sliced out of it here.
/// - `qbeg`, `rbeg`, `len`: the seed, as `MemSeed` holds it.
///
/// # Returns
///
/// The ksw score, or `-1` meaning "not scored, keep it". The two `-1` cases are bwa's: a seed at
/// least [`MEM_SHORT_LEN`] long, and a neighbourhood that grew to [`MEM_SHORT_LEN`] on either side.
fn mem_seed_sw(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    query: &[u8],
    qbeg: i32,
    rbeg: i64,
    len: i32,
) -> i32 {
    let l_query = query.len() as i64;
    let l_pac = bns.l_pac;
    if i64::from(len) >= MEM_SHORT_LEN {
        // "the seed is longer than the max-extend; no need to do SW"
        return -1;
    }
    // The seed itself, then the same interval widened by `MEM_SHORT_EXT` on all four sides and
    // clamped to the read and to the doubled reference space.
    let (mut qb, mut qe) = (i64::from(qbeg), i64::from(qbeg) + i64::from(len));
    let (mut rb, mut re) = (rbeg, rbeg + i64::from(len));
    // The midpoint of the UNWIDENED seed, which is what picks the contig below. Taken before the
    // widening, exactly as the C does.
    let mid = (rb + re) >> 1;
    qb = (qb - MEM_SHORT_EXT).max(0);
    qe = (qe + MEM_SHORT_EXT).min(l_query);
    rb = (rb - MEM_SHORT_EXT).max(0);
    re = (re + MEM_SHORT_EXT).min(l_pac << 1);
    // Do not let the window straddle the forward/reverse seam: cut it back to the half the seed's
    // midpoint lives in.
    if rb < l_pac && l_pac < re {
        if mid < l_pac {
            re = l_pac;
        } else {
            rb = l_pac;
        }
    }
    if qe - qb >= MEM_SHORT_LEN || re - rb >= MEM_SHORT_LEN {
        // "the seed seems good enough; no need to do SW"
        return -1;
    }
    // `bns_fetch_seq` narrows `rb`/`re` to the contig holding `mid`, and the ksw call below uses
    // the NARROWED width, because the C reads `re - rb` after the fetch has written through its
    // pointers.
    let (rb, re, _rid, rseq) = crate::pe::bns_fetch_seq(fm, bns, rb, mid, re);
    debug_assert_eq!(rseq.len() as i64, re - rb);
    // `minsc` 0: the C passes `KSW_XSTART` alone, so `xtra & 0xffff` is zero. Only `score` is read,
    // and `score` does not depend on the suboptimal bookkeeping our `ksw_align2` also does.
    ksw_align2(
        &query[qb as usize..qe as usize],
        &rseq,
        5,
        &opt.mat,
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        0,
        opt.a,
        KSW_I16_LANES,
    )
    .score
}

/// Drop the chained seeds that a local alignment says are noise, and write every survivor's
/// `score`. Port of `mem_flt_chained_seeds` (`bwamem.cpp:472-508`), run once per read between
/// `mem_chain_flt` and the extension (`bwamem.cpp:1084`).
///
/// # Parameters
///
/// - `fm`, `bns`: the reference, for the per-seed re-scoring window.
/// - `opt`: `a` and the gap penalties for ksw, plus `min_chain_weight` (`-W`), which replaces the
///   `ln(l_query)` term when the user sets it.
/// - `query`: the read as nt4 codes. Its LENGTH is what decides whether the filter runs at all.
/// - `chains`: this read's chains, filtered in place. A chain can come out with no seeds left;
///   bwa's `mem_chain2aln` skips such a chain (`if (c->n == 0) continue;`, `bwamem.cpp:2137`) and
///   so must every caller here.
///
/// # Note
///
/// The threshold test is per chain in bwa-mem2, not per read, even though every term in it is a
/// property of the read. That is a leftover from the merge of nh13's per-read function into a
/// per-batch loop (the original's early `return` is still there, commented out, above the loop it
/// became a `continue` in). It makes no difference, and it is transcribed as written.
pub fn mem_flt_chained_seeds(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    query: &[u8],
    chains: &mut [MemChain],
) {
    let l_query = query.len();
    if l_query == 0 {
        return;
    }
    // The score a seed must reach to survive, in bwa's two regimes: a multiple of `-W` when the
    // user pinned a minimum chain weight, else a multiple of the read length's logarithm.
    //
    // Hoisted out of the chain loop the C computes it in. Both terms are properties of the READ, so
    // every chain of a read gets the same two numbers and the same verdict; recomputing them, and
    // in particular calling `ln` again, once per chain would only cost a transcendental per chain
    // on every short read to reach a conclusion already known. See the note in this function's doc
    // comment for why the C has it inside the loop at all.
    let min_l = if opt.min_chain_weight != 0 {
        f64::from(MEM_HSP_COEF) * f64::from(opt.min_chain_weight)
    } else {
        f64::from(MEM_MINSC_COEF) * (l_query as f64).ln()
    };
    if min_l > f64::from(MEM_SEEDSW_COEF) * l_query as f64 {
        // "don't run the following for short reads" -- the whole filter, for this read. This is the
        // branch every 150 bp read takes, and taking it is the reason the filter's absence was
        // invisible for so long.
        return;
    }
    // `(int)(x + .499)` in the C: a truncation of a nudged value, not a round-half-even.
    let min_hsp_score = (f64::from(opt.a) * min_l + 0.499) as i32;
    for chain in chains.iter_mut() {
        chain.seeds.retain_mut(|s| {
            let score = mem_seed_sw(fm, bns, opt, query, s.qbeg, s.rbeg, s.len);
            if score < 0 || score >= min_hsp_score {
                // A skipped SW leaves the seed worth its length, in score units this time: the
                // chaining stage had written the bare length.
                s.score = if score < 0 { s.len * opt.a } else { score };
                true
            } else {
                false
            }
        });
    }
}

//! Alignment core: chains -> scored alignment regions (`mem_chain2aln`) and the single-end driver,
//! mirroring bwa-mem2's `mem_chain2aln_across_reads_V2` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Phase 6 (first milestone): produce alignment regions and pick the best mapping position. Full
//! byte-identical SAM (dedup, primary marking, MAPQ, CIGAR, tags) is layered on top of this.

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

/// Sentinel for uninitialized region bounds (bwa's `H0_`).
pub(crate) const H0_SENTINEL: i64 = -99;
/// bwa's `MAX_BAND_TRY`.
pub(crate) const MAX_BAND_TRY: i32 = 2;

/// A scored alignment region (bwa-mem2's `mem_alnreg_t`, phase-6 subset).
#[derive(Debug, Clone)]
pub struct MemAlnReg {
    pub rb: i64,
    pub re: i64,
    pub qb: i32,
    pub qe: i32,
    pub rid: i32,
    pub score: i32,
    pub truesc: i32,
    pub sub: i32,
    pub csub: i32,
    pub sub_n: i32,
    pub seedcov: i32,
    pub seedlen0: i32,
    pub secondary: i32,
    /// Rank-preserving secondary index used only by `mem_gen_alt` (`get_pri_idx`). Equals
    /// `secondary` after marking on a no-ALT reference, but the PE primary/secondary swap mutates
    /// it independently of `secondary` (which the `-2` sentinel repurposes for the emitted record).
    pub secondary_all: i32,
    pub w: i32,
    pub frac_rep: f32,
    pub is_alt: bool,
    pub hash: u64,
    pub n_comp: i32,
}

pub(crate) fn cal_max_gap(opt: &MemOpt, qlen: i32) -> i32 {
    let a = f64::from(opt.a);
    let l_del = ((f64::from(qlen) * a - f64::from(opt.o_del)) / f64::from(opt.e_del) + 1.0) as i32;
    let l_ins = ((f64::from(qlen) * a - f64::from(opt.o_ins)) / f64::from(opt.e_ins) + 1.0) as i32;
    let l = l_del.max(l_ins).max(1);
    l.min(opt.w << 1)
}

struct SideResult {
    score: i32,
    qle: i32,
    tle: i32,
    gscore: i32,
    gtle: i32,
    w: i32,
}

/// One-sided extension with bwa's `MAX_BAND_TRY` band-doubling acceptance.
#[allow(clippy::too_many_arguments)]
fn extend_side(qs: &[u8], rs: &[u8], opt: &MemOpt, pen_clip: i32, h0: i32) -> SideResult {
    let mut prev = -1i32;
    let mut i = 0;
    loop {
        let w = opt.w << i;
        let r = ksw_extend2(
            qs, rs, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, w, pen_clip,
            opt.zdrop, h0,
        );
        if r.score == prev || r.max_off < (w >> 1) + (w >> 2) || i + 1 == MAX_BAND_TRY {
            return SideResult {
                score: r.score,
                qle: r.qle,
                tle: r.tle,
                gscore: r.gscore,
                gtle: r.gtle,
                w,
            };
        }
        prev = r.score;
        i += 1;
    }
}

/// Extend every seed of `chain` into an alignment region (one region per seed). Port of the
/// per-chain body of `mem_chain2aln_across_reads_V2`.
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
    let l_query = codes.len() as i32;
    let l_pac = bns.l_pac;

    // Reference window spanning the chain.
    let mut rmax0 = l_pac << 1;
    let mut rmax1 = 0i64;
    for s in &chain.seeds {
        let b = s.rbeg - (i64::from(s.qbeg) + i64::from(cal_max_gap(opt, s.qbeg)));
        let tail = l_query - s.qbeg - s.len;
        let e = s.rbeg + i64::from(s.len) + (i64::from(tail) + i64::from(cal_max_gap(opt, tail)));
        rmax0 = rmax0.min(b);
        rmax1 = rmax1.max(e);
    }
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
    let (rmax0, rmax1, _rid) = bns.fetch_bounds(rmax0, rmax1, chain.seeds[0].rbeg);
    let rseq: Vec<u8> = (rmax0..rmax1).map(|p| fm.base(p)).collect();

    // Seeds in descending (score, index) order.
    let mut order: Vec<usize> = (0..chain.seeds.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse((u64::from(chain.seeds[i].score as u32) << 32) | i as u64)
    });

    // Same contained-seed extension skip as the batched path (they must stay region-identical), and
    // the same mutual exclusion with the discard pass, which needs one slot per seed.
    let skip_contained = crate::across::skip_contained_enabled();

    for (pos, &si) in order.iter().enumerate() {
        if skip_contained && crate::across::seed_ext_redundant(&chain.seeds, si) {
            // Keep the slot (the discard pass reproduces bwa-mem2's scan order), skip the DP.
            meta.push(RegMeta { chain: ci as u32, pos: pos as u32, seed: si as u32 });
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
                seedlen0: chain.seeds[si].len,
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
        let s = chain.seeds[si];
        let mut a = MemAlnReg {
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
            seedlen0: s.len,
            secondary: -1,
            secondary_all: -1,
            w: opt.w,
            frac_rep: chain.frac_rep,
            is_alt: chain.is_alt,
            hash: 0,
            n_comp: 1,
        };

        // Left extension.
        if s.qbeg > 0 {
            let qs: Vec<u8> = (0..s.qbeg).rev().map(|i| codes[i as usize]).collect();
            let rlen = (s.rbeg - rmax0) as usize;
            let rs: Vec<u8> = (0..rlen).rev().map(|i| rseq[i]).collect();
            let h0 = s.len * opt.a;
            let r = extend_side(&qs, &rs, opt, opt.pen_clip5, h0);
            a.score = r.score;
            if r.gscore <= 0 || r.gscore <= r.score - opt.pen_clip5 {
                a.qb = s.qbeg - r.qle;
                a.rb = s.rbeg - i64::from(r.tle);
                a.truesc = r.score;
            } else {
                a.qb = 0;
                a.rb = s.rbeg - i64::from(r.gtle);
                a.truesc = r.gscore;
            }
            a.w = a.w.max(r.w);
        } else {
            a.score = s.len * opt.a;
            a.truesc = a.score;
            a.qb = 0;
            a.rb = s.rbeg;
        }

        // Right extension.
        if s.qbeg + s.len != l_query {
            let qe = s.qbeg + s.len;
            let re = s.rbeg + i64::from(s.len) - rmax0;
            let qs: Vec<u8> = codes[qe as usize..].to_vec();
            let rs: Vec<u8> = rseq[re as usize..].to_vec();
            let h0 = a.score;
            let r = extend_side(&qs, &rs, opt, opt.pen_clip3, h0);
            a.score = r.score;
            let re_abs = rmax0 + re;
            if r.gscore <= 0 || r.gscore <= r.score - opt.pen_clip3 {
                a.qe = qe + r.qle;
                a.re = re_abs + i64::from(r.tle);
                a.truesc += r.score - h0;
            } else {
                a.qe = l_query;
                a.re = re_abs + i64::from(r.gtle);
                a.truesc += r.gscore - h0;
            }
            a.w = a.w.max(r.w);
        } else {
            a.qe = l_query;
            a.re = s.rbeg + i64::from(s.len);
        }

        // Seed coverage within the region.
        if a.rb != H0_SENTINEL && a.qb != H0_SENTINEL as i32 {
            a.seedcov = 0;
            for t in &chain.seeds {
                if t.qbeg >= a.qb
                    && t.qbeg + t.len <= a.qe
                    && t.rbeg >= a.rb
                    && t.rbeg + i64::from(t.len) <= a.re
                {
                    a.seedcov += t.len;
                }
            }
        }
        meta.push(RegMeta { chain: ci as u32, pos: pos as u32, seed: si as u32 });
        preskip.push(false);
        out.push(a);
    }
}

/// Align one read (2-bit codes) through seeding -> chaining -> extension, returning all regions.
pub fn align_read(fm: &FmIndex, bns: &BntSeq, opt: &MemOpt, codes: &[u8]) -> Vec<MemAlnReg> {
    let chains = mem_chain_flt(opt, build_chains(fm, bns, opt, codes, 0));
    let mut regs = Vec::new();
    let mut meta = Vec::new();
    let mut preskip = Vec::new();
    for (ci, c) in chains.iter().enumerate() {
        mem_chain2aln_meta(fm, bns, opt, codes, ci, c, &mut regs, &mut meta, &mut preskip);
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
    // Same compaction the batched path does, and for the same reason (the dedup's unstable sort).
    regs.retain(|a| a.qe > a.qb);
    regs
}

/// Align one read and deduplicate its regions (`mem_sort_dedup_patch`), WITHOUT primary marking.
/// This is the per-read input to paired-end statistics (`mem_pestat`) and pairing, which mark
/// primaries themselves.
pub fn align_read_dedup(fm: &FmIndex, bns: &BntSeq, opt: &MemOpt, codes: &[u8]) -> Vec<MemAlnReg> {
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
pub fn dump_regs(bns: &BntSeq, tag: &str, regs: &[MemAlnReg]) {
    eprintln!("--- regs [{}] n={} ---", tag, regs.len());
    for (i, r) in regs.iter().enumerate() {
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
pub fn region_to_pos(bns: &BntSeq, reg: &MemAlnReg) -> (i32, i64, bool) {
    let probe = if reg.rb < bns.l_pac {
        reg.rb
    } else {
        reg.re - 1
    };
    let (fpos, is_rev) = bns.depos(probe);
    let rid = bns.pos2rid(fpos);
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

    fn tiny() -> (FmIndex, BntSeq) {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    fn best(regs: &[MemAlnReg]) -> &MemAlnReg {
        regs.iter().max_by_key(|r| r.score).unwrap()
    }

    #[test]
    fn forward_slice_maps_to_origin() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
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
        // one mismatch in the middle
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

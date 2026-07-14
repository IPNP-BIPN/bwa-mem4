//! Alternate-hit (`XA:Z:`) generation, port of `mem_gen_alt` (bwamem_extra.cpp).
//!
//! Runs after `mem_mark_primary_se`. For each region it decides, via `get_pri_idx`, whether the
//! region is a near-primary alternate (score within `xa_drop_ratio` of its primary) and, if so,
//! groups it under that primary. Primaries with too many alternates (> `max_xa_hits`) are dropped.
//! The reference has no ALT contigs, so `secondary_all == secondary` and `has_alt` is always false.

use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

use crate::cigar::reg2aln;
use crate::MemAlnReg;

/// `get_pri_idx`: the primary index region `i` is an XA alternate of, or `None`.
fn get_pri_idx(xa_drop_ratio: f64, regs: &[MemAlnReg], i: usize) -> Option<usize> {
    let k = regs[i].secondary_all;
    if k >= 0 && f64::from(regs[i].score) >= f64::from(regs[k as usize].score) * xa_drop_ratio {
        Some(k as usize)
    } else {
        None
    }
}

/// For each region index, the list of alternate-region indices to emit under it as `XA` hits,
/// in region order. Empty for regions that are not XA primaries. Pure port of `mem_gen_alt`'s
/// selection (the string formatting is done by the caller via `reg2aln`).
pub fn xa_group(opt: &MemOpt, regs: &[MemAlnReg]) -> Vec<Vec<usize>> {
    let ratio = f64::from(opt.xa_drop_ratio);
    let n = regs.len();
    let mut cnt = vec![0i32; n];
    let mut has_alt = vec![false; n];
    for i in 0..n {
        if let Some(r) = get_pri_idx(ratio, regs, i) {
            cnt[r] += 1;
            if regs[i].is_alt {
                has_alt[r] = true;
            }
        }
    }
    let mut groups = vec![Vec::new(); n];
    for i in 0..n {
        if let Some(r) = get_pri_idx(ratio, regs, i) {
            if cnt[r] > opt.max_xa_hits_alt || (!has_alt[r] && cnt[r] > opt.max_xa_hits) {
                continue;
            }
            groups[r].push(i);
        }
    }
    groups
}

/// Per-region `XA:Z:` string (`rname,±pos,cigar,NM;`... concatenated), or `None` for regions that
/// are not XA primaries. Port of `mem_gen_alt`: groups near-primary alternates and formats each
/// via `reg2aln`. Runs after `mem_mark_primary_se` (and, in PE, after the primary/secondary swap).
pub fn mem_gen_alt(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    regs: &[MemAlnReg],
    l_query: i32,
    query: &[u8],
) -> Vec<Option<String>> {
    let groups = xa_group(opt, regs);
    let mut out = vec![None; regs.len()];
    for (r, alts) in groups.iter().enumerate() {
        if alts.is_empty() {
            continue;
        }
        let mut s = Vec::new();
        for &i in alts {
            let t = reg2aln(fm, bns, opt, l_query, query, &regs[i]);
            s.extend_from_slice(bns.contigs[t.rid as usize].name.as_bytes());
            s.push(b',');
            s.push(if t.is_rev { b'-' } else { b'+' });
            s.extend_from_slice((t.pos + 1).to_string().as_bytes());
            s.push(b',');
            crate::pe::add_cigar(&t.cigar, 0, &mut s);
            s.push(b',');
            s.extend_from_slice(t.nm.to_string().as_bytes());
            s.push(b';');
        }
        out[r] = Some(String::from_utf8(s).expect("ASCII XA"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal region carrying only the fields `xa_group` reads.
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
            hash: 0,
            n_comp: 1,
        }
    }

    #[test]
    fn near_primary_alternate_grouped_distant_one_dropped() {
        // Read _217: primary score 140, alternate 116 (>= 140*0.8=112 -> XA),
        // third 98 (< 112 -> not XA). All secondary to region 0.
        let opt = MemOpt::default();
        let regs = [reg(140, -1), reg(116, 0), reg(98, 0)];
        let groups = xa_group(&opt, &regs);
        assert_eq!(groups[0], vec![1]);
        assert!(groups[1].is_empty());
        assert!(groups[2].is_empty());
    }

    #[test]
    fn too_many_hits_drops_all_of_that_primary() {
        // 6 alternates all within ratio -> cnt=6 > max_xa_hits(5) -> group dropped entirely.
        let opt = MemOpt::default();
        let mut regs = vec![reg(150, -1)];
        for _ in 0..6 {
            regs.push(reg(140, 0));
        }
        let groups = xa_group(&opt, &regs);
        assert!(
            groups[0].is_empty(),
            "primary with >5 alternates emits no XA"
        );
    }
}

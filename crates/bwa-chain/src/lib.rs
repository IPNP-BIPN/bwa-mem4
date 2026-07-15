//! Seed chaining and chain filtering, mirroring bwa-mem2's `mem_chain` / `test_and_merge` /
//! `mem_chain_weight` / `mem_chain_flt` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Chains collinear seeds into candidate alignments, then filters by weight and overlap. The
//! end-to-end byte-identity gate is the SE SAM concordance in phase 6.

use std::collections::BTreeMap;

use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};
use bwa_seed::{mem_collect_smem, MemSeed};

/// A chain of collinear seeds (bwa-mem2's `mem_chain_t`).
#[derive(Debug, Clone)]
pub struct MemChain {
    pub seqid: i32,
    pub rid: i32,
    pub pos: i64,
    pub w: i32,
    pub kept: u8,
    pub is_alt: bool,
    pub first: i32,
    pub frac_rep: f32,
    pub seeds: Vec<MemSeed>,
}

#[inline]
fn chn_beg(c: &MemChain) -> i32 {
    c.seeds[0].qbeg
}
#[inline]
fn chn_end(c: &MemChain) -> i32 {
    let last = c.seeds.last().unwrap();
    last.qbeg + last.len
}

/// Try to absorb seed `p` (on contig `seed_rid`) into chain `c`; returns whether it was
/// merged/contained. Faithful port of `test_and_merge`.
fn test_and_merge(opt: &MemOpt, l_pac: i64, c: &mut MemChain, p: &MemSeed, seed_rid: i32) -> bool {
    let last = *c.seeds.last().unwrap();
    let qend = last.qbeg + last.len;
    let rend = last.rbeg + i64::from(last.len);

    if seed_rid != c.rid {
        return false;
    }
    if p.qbeg >= c.seeds[0].qbeg
        && p.qbeg + p.len <= qend
        && p.rbeg >= c.seeds[0].rbeg
        && p.rbeg + i64::from(p.len) <= rend
    {
        return true; // contained
    }
    if (last.rbeg < l_pac || c.seeds[0].rbeg < l_pac) && p.rbeg >= l_pac {
        return false; // different strand
    }
    let x = i64::from(p.qbeg - last.qbeg);
    let y = p.rbeg - last.rbeg;
    if y >= 0
        && x - y <= i64::from(opt.w)
        && y - x <= i64::from(opt.w)
        && x - i64::from(last.len) < i64::from(opt.max_chain_gap)
        && y - i64::from(last.len) < i64::from(opt.max_chain_gap)
    {
        c.seeds.push(*p);
        return true;
    }
    false
}

/// Chain weight = min of non-overlapping query and reference coverage. Port of `mem_chain_weight`.
pub fn mem_chain_weight(c: &MemChain) -> i32 {
    let mut w = 0i64;
    let mut end = 0i64;
    for s in &c.seeds {
        let (b, len) = (i64::from(s.qbeg), i64::from(s.len));
        if b >= end {
            w += len;
        } else if b + len > end {
            w += b + len - end;
        }
        end = end.max(b + len);
    }
    let tmp = w;
    w = 0;
    end = 0;
    for s in &c.seeds {
        let (b, len) = (s.rbeg, i64::from(s.len));
        if b >= end {
            w += len;
        } else if b + len > end {
            w += b + len - end;
        }
        end = end.max(b + len);
    }
    w = w.min(tmp);
    if w < (1 << 30) {
        w as i32
    } else {
        (1 << 30) - 1
    }
}

/// Build seed chains for a single read (2-bit codes) with the given `seqid`. Port of `mem_chain`
/// (round-1 seeds only for now; the btree "closest lower chain" is a `BTreeMap` keyed by position).
pub fn build_chains(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    seqid: i32,
) -> Vec<MemChain> {
    let smems = mem_collect_smem(fm, codes, opt);
    build_chains_from_smems(fm, bns, opt, codes, seqid, smems)
}

/// Build chains from **pre-computed** SMEMs (e.g. from batched lockstep seeding). Identical to
/// [`build_chains`] given the same seed set; only the SMEM source differs.
pub fn build_chains_from_smems(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    seqid: i32,
    mut smems: Vec<bwa_index::Smem>,
) -> Vec<MemChain> {
    // Intra-read SMEM order: by (m, n) ascending (bwa's `intv_lt1`).
    smems.sort_by_key(|s| (u64::from(s.m) << 32) | u64::from(s.n));

    // Repetitive length -> frac_rep.
    let mut l_rep = 0i64;
    let (mut b, mut e) = (0i64, 0i64);
    for p in &smems {
        if p.s <= i64::from(opt.max_occ) {
            continue;
        }
        let (sb, se) = (i64::from(p.m), i64::from(p.n) + 1);
        if sb > e {
            l_rep += e - b;
            b = sb;
            e = se;
        } else {
            e = e.max(se);
        }
    }
    l_rep += e - b;

    let l_pac = bns.l_pac;
    let max_occ = i64::from(opt.max_occ);
    let mut chains: Vec<MemChain> = Vec::new();
    let mut tree: BTreeMap<i64, usize> = BTreeMap::new();

    // Pass 1: gather every sampled occurrence position (in the exact order the merge consumes them)
    // and each SMEM's sampled count. Each `get_sa` is a random-access LF-walk, so resolving them all
    // in one lockstep+prefetch batch hides the DRAM latency (vs. one serial walk per occurrence).
    let mut positions: Vec<i64> = Vec::new();
    let mut counts: Vec<i64> = Vec::with_capacity(smems.len());
    for p in &smems {
        let step = if p.s > max_occ { p.s / max_occ } else { 1 };
        let mut k = 0i64;
        let mut count = 0i64;
        while k < p.s && count < max_occ {
            positions.push(p.k + k);
            k += step;
            count += 1;
        }
        counts.push(count);
    }
    let mut rbegs = vec![0i64; positions.len()];
    fm.get_sa_batch(&positions, &mut rbegs);

    // Pass 2: merge, replaying the original per-occurrence logic with precomputed `rbeg` values
    // (same values, same order -> byte-identical chains).
    let mut pi = 0usize;
    for (si, p) in smems.iter().enumerate() {
        let slen = (p.n + 1 - p.m) as i32;
        for _ in 0..counts[si] {
            let rbeg = rbegs[pi];
            pi += 1;
            let s = MemSeed {
                rbeg,
                qbeg: p.m as i32,
                len: slen,
                score: slen,
            };
            let rid = bns.intv2rid(rbeg, rbeg + i64::from(slen));
            if rid < 0 {
                continue;
            }
            let mut to_add = true;
            if let Some((_, &ci)) = tree.range(..=rbeg).next_back() {
                if test_and_merge(opt, l_pac, &mut chains[ci], &s, rid) {
                    to_add = false;
                }
            }
            if to_add {
                let idx = chains.len();
                chains.push(MemChain {
                    seqid,
                    rid,
                    pos: rbeg,
                    w: 0,
                    kept: 0,
                    is_alt: false,
                    first: -1,
                    frac_rep: 0.0,
                    seeds: vec![s],
                });
                tree.insert(rbeg, idx);
            }
        }
    }

    let frac = l_rep as f32 / codes.len() as f32;
    for c in &mut chains {
        c.frac_rep = frac;
    }
    // bwa-mem2 stores chains in a position-keyed kbtree and emits them via an in-order traversal,
    // so `mem_chain_flt` receives them sorted by `pos` ascending. We build in seed-occurrence order,
    // so re-sort to match; this ordering drives the (unstable) tie-break among equal-weight chains.
    chains.sort_by_key(|c| c.pos);
    chains
}

/// Insertion sort over `a`, moving an element left while it is strictly `lt` its predecessor.
/// Faithful port of klib's `__ks_insertsort` (stable for equal keys). `lt(x, y)` = `x < y`.
fn ks_insertsort_by<T>(a: &mut [T], lt: &impl Fn(&T, &T) -> bool) {
    for i in 1..a.len() {
        let mut j = i;
        while j > 0 && lt(&a[j], &a[j - 1]) {
            a.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Comb sort, klib's `ks_combsort` (introsort's depth-limit fallback), ending with an insertion
/// sort pass. `lt(x, y)` = `x < y`.
fn ks_combsort_by<T>(a: &mut [T], lt: &impl Fn(&T, &T) -> bool) {
    let n = a.len();
    if n == 0 {
        return;
    }
    const SHRINK: f64 = 1.2473309501039786540366528676643;
    let mut gap = n;
    loop {
        if gap > 2 {
            gap = (gap as f64 / SHRINK) as usize;
            if gap == 9 || gap == 10 {
                gap = 11;
            }
        }
        let mut do_swap = false;
        if gap < n {
            let mut i = 0;
            while i < n - gap {
                let j = i + gap;
                if lt(&a[j], &a[i]) {
                    a.swap(i, j);
                    do_swap = true;
                }
                i += 1;
            }
        }
        if !(do_swap || gap > 2) {
            break;
        }
    }
    if gap != 1 {
        ks_insertsort_by(a, lt);
    }
}

/// Introspective sort, a faithful port of klib's `ks_introsort` (median-of-3 quicksort with a
/// depth-limited comb-sort fallback and a final insertion-sort pass). This is deliberately the
/// exact same (unstable) permutation bwa-mem2 applies in `mem_chain_flt`, so equal-weight chains
/// resolve identically to the oracle. `lt(x, y)` = `x < y`.
fn ks_introsort_by<T>(a: &mut [T], lt: impl Fn(&T, &T) -> bool) {
    let n = a.len();
    if n < 1 {
        return;
    }
    if n == 2 {
        if lt(&a[1], &a[0]) {
            a.swap(0, 1);
        }
        return;
    }
    let mut dd = 2usize;
    while (1usize << dd) < n {
        dd += 1;
    }
    let mut stack: Vec<(usize, usize, i64)> = Vec::new();
    let mut s = 0usize;
    let mut t = n - 1;
    let mut d: i64 = (dd as i64) << 1;
    loop {
        if s < t {
            d -= 1;
            if d == 0 {
                ks_combsort_by(&mut a[s..=t], &lt);
                t = s;
                continue;
            }
            let mut i = s;
            let mut j = t;
            // Median-of-3 pivot selection (klib picks the middle-biased index k).
            let mut k = i + ((j - i) >> 1) + 1;
            if lt(&a[k], &a[i]) {
                if lt(&a[k], &a[j]) {
                    k = j;
                }
            } else {
                k = if lt(&a[j], &a[i]) { i } else { j };
            }
            if k != t {
                a.swap(k, t);
            }
            // Pivot value now lives at index `t` and is untouched until the final swap below.
            loop {
                loop {
                    i += 1;
                    if !lt(&a[i], &a[t]) {
                        break;
                    }
                }
                loop {
                    j -= 1;
                    if !(i <= j && lt(&a[t], &a[j])) {
                        break;
                    }
                }
                if j <= i {
                    break;
                }
                a.swap(i, j);
            }
            a.swap(i, t);
            let is = i - s;
            let ti = t - i;
            if is > ti {
                if is > 16 {
                    stack.push((s, i - 1, d));
                }
                s = if ti > 16 { i + 1 } else { t };
            } else {
                if ti > 16 {
                    stack.push((i + 1, t, d));
                }
                t = if is > 16 { i - 1 } else { s };
            }
        } else if let Some((l, r, dep)) = stack.pop() {
            s = l;
            t = r;
            d = dep;
        } else {
            ks_insertsort_by(a, &lt);
            return;
        }
    }
}

/// Filter chains for a single read: drop light chains, then prune overlapping ones. Port of
/// `mem_chain_flt` (single-read group).
pub fn mem_chain_flt(opt: &MemOpt, chains: Vec<MemChain>) -> Vec<MemChain> {
    if chains.is_empty() {
        return chains;
    }
    let mut a: Vec<MemChain> = Vec::new();
    for mut c in chains {
        c.first = -1;
        c.kept = 0;
        c.w = mem_chain_weight(&c);
        if c.w >= opt.min_chain_weight {
            a.push(c);
        }
    }
    if a.is_empty() {
        return a;
    }

    // Sort by weight descending with bwa-mem2's exact (unstable) `ks_introsort`, so equal-weight
    // overlapping chains are ordered identically to the oracle (`flt_lt(a, b) = a.w > b.w`).
    ks_introsort_by(&mut a, |x, y| x.w > y.w);

    a[0].kept = 3;
    let mut kept_idx: Vec<usize> = vec![0];
    for i in 1..a.len() {
        let (ib, ie, iw, ialt) = (chn_beg(&a[i]), chn_end(&a[i]), a[i].w, a[i].is_alt);
        let mut large_ovlp = false;
        let mut broke = false;
        for &j in &kept_idx {
            let (jb, je, jw, jalt) = (chn_beg(&a[j]), chn_end(&a[j]), a[j].w, a[j].is_alt);
            let b_max = jb.max(ib);
            let e_min = je.min(ie);
            if e_min > b_max && (!jalt || ialt) {
                let li = ie - ib;
                let lj = je - jb;
                let min_l = li.min(lj);
                if (e_min - b_max) as f32 >= min_l as f32 * opt.mask_level
                    && min_l < opt.max_chain_gap
                {
                    large_ovlp = true;
                    if a[j].first < 0 {
                        a[j].first = i as i32;
                    }
                    if (iw as f32) < jw as f32 * opt.drop_ratio && jw - iw >= opt.min_seed_len << 1
                    {
                        broke = true;
                        break;
                    }
                }
            }
        }
        if !broke {
            kept_idx.push(i);
            a[i].kept = if large_ovlp { 2 } else { 3 };
        }
    }
    for &ci in &kept_idx {
        let f = a[ci].first;
        if f >= 0 {
            a[f as usize].kept = 1;
        }
    }
    // max_chain_extend demotion (default is 1<<30, effectively never).
    let mut k = 0i32;
    let mut i = 0usize;
    while i < a.len() {
        if a[i].kept == 0 || a[i].kept == 3 {
            i += 1;
            continue;
        }
        k += 1;
        i += 1;
        if k >= opt.max_chain_extend {
            break;
        }
    }
    while i < a.len() {
        if a[i].kept < 3 {
            a[i].kept = 0;
        }
        i += 1;
    }
    a.retain(|c| c.kept != 0);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_index::BntSeq;
    use std::path::Path;

    fn tiny() -> (FmIndex, BntSeq) {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    // A brute-force reference introsort matching klib's structure is unnecessary; we assert the
    // two invariants that must hold for any faithful port: the result is fully sorted by the
    // comparator, and it is a permutation of the input. Byte-identity to the oracle's *unstable*
    // permutation is covered by the end-to-end SAM gate.
    #[test]
    fn introsort_sorts_and_permutes_various_sizes() {
        // Deterministic LCG so the test is reproducible without extra deps.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i32 % 100
        };
        for &n in &[0usize, 1, 2, 3, 5, 16, 17, 33, 64, 200, 1000] {
            let orig: Vec<i32> = (0..n).map(|_| next()).collect();
            let mut a = orig.clone();
            ks_introsort_by(&mut a, |x, y| x > y); // descending, like flt_lt
            for w in a.windows(2) {
                assert!(w[0] >= w[1], "not sorted desc at n={n}: {a:?}");
            }
            let mut s1 = orig.clone();
            let mut s2 = a.clone();
            s1.sort_unstable();
            s2.sort_unstable();
            assert_eq!(s1, s2, "not a permutation at n={n}");
        }
    }

    #[test]
    fn introsort_two_element_special_case() {
        let mut a = [1i32, 2];
        ks_introsort_by(&mut a, |x, y| x > y);
        assert_eq!(a, [2, 1]);
        let mut b = [5i32, 3];
        ks_introsort_by(&mut b, |x, y| x > y);
        assert_eq!(b, [5, 3]);
    }

    #[test]
    fn single_slice_makes_one_chain() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        let start = 40_000i64;
        let read: Vec<u8> = (0..150).map(|i| fm.base(start + i)).collect();
        let chains = build_chains(&fm, &bns, &opt, &read, 0);
        let chains = mem_chain_flt(&opt, chains);
        assert!(!chains.is_empty());
        // A chain whose first seed sits at the origin, covering the read.
        let c = chains
            .iter()
            .find(|c| c.seeds.iter().any(|s| s.rbeg == start))
            .expect("origin chain");
        assert_eq!(mem_chain_weight(c), 150);
        // Seeds within a chain are collinear and non-decreasing in qbeg.
        for w in c.seeds.windows(2) {
            assert!(w[1].qbeg >= w[0].qbeg);
        }
    }

    #[test]
    fn two_slices_make_two_chains() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        // Concatenate two distant reference slices -> two separate chains.
        let mut read: Vec<u8> = (0..80).map(|i| fm.base(10_000 + i)).collect();
        read.extend((0..80).map(|i| fm.base(120_000 + i)));
        let chains = build_chains(&fm, &bns, &opt, &read, 0);
        let chains = mem_chain_flt(&opt, chains);
        assert!(
            chains.len() >= 2,
            "expected >=2 chains, got {}",
            chains.len()
        );
    }
}

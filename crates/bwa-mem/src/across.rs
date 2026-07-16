//! Batched seed extension across a whole read batch, porting bwa-mem2's
//! `mem_chain2aln_across_reads_V2` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! bwa-mem2 does not extend one seed at a time: it collects every seed's **left** and **right**
//! one-sided extension across all reads into two job arrays, sorts each by length so a SIMD batch
//! packs similar-length alignments into its lanes (bwa-mem2 also bins int8/int16/scalar by length),
//! runs them through the vectorized banded Smith-Waterman (`bandedSWA`), and scatters the results
//! back with the exact `MAX_BAND_TRY` band-doubling acceptance logic. Right extensions use each
//! region's post-left score as `h0`, so left must complete before right.
//!
//! Because each region's extension result depends only on its own `(query, target, h0, w)`, batching
//! and length-sorting are **result-preserving**: [`align_reads_batched`] returns, for every read, the
//! same `Vec<MemAlnReg>` as calling [`crate::align_read`] per read (checked by an equivalence test),
//! while routing the DP through a batched [`SwBackend`] (the NEON kernel). The retry `prev` semantics
//! mirror the per-read [`crate::extend_side`] (`prev` reset to -1 per side) so the two paths are
//! bit-for-bit identical.

use crate::{cal_max_gap, MemAlnReg, H0_SENTINEL, MAX_BAND_TRY};
use bwa_chain::{build_chains_from_smems, mem_chain_flt, MemChain};
use bwa_core::MemOpt;
use bwa_extend::{ExtendJob, SwBackend};
use bwa_index::{BntSeq, FmIndex};
use bwa_seed::{mem_collect_smem_batched, MemSeed};

/// nh13's `mem_seed_ext_redundant` (`--skip-contained-ext`): true when seed `si` is strictly
/// contained, on the same diagonal, in a longer seed of the same chain, and no comparably long seed
/// interferes on a different diagonal. Skipping its banded-SW saves ~7.7% SE / ~5% PE.
///
/// **Off by default: it is not output-preserving.** The skipped seed still needs a region slot (the
/// discard pass reproduces bwa-mem2's scan order, which is slot-ordered), but with no DP that slot
/// has no real `rb`/`re`. `mem_sort_dedup_patch` sorts regions by `re` with an *unstable* introsort
/// and lets purged regions take part as `p`, so the placeholder bounds move real regions around in
/// the sort and change which alignment survives a score tie. bwa extends these seeds and purges them
/// afterwards, keeping their true `rb`/`re` -- values we cannot fabricate without doing the DP we are
/// trying to skip. Measured cost of enabling it: 2 extra diverging records per 100k real reads.
///
/// `BWA3_SKIP_CONTAINED=1` opts in, trading that exactness for the speed. Cached: the two extension
/// paths (batched [`align_reads_batched`] and per-read `mem_chain2aln`) must agree, so they share
/// this one decision.
pub(crate) fn skip_contained_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BWA3_SKIP_CONTAINED").is_some())
}

pub(crate) fn seed_ext_redundant(seeds: &[MemSeed], si: usize) -> bool {
    let s = seeds[si];
    let sd = s.rbeg - i64::from(s.qbeg);
    let mut has_container = false;
    for (j, t) in seeds.iter().enumerate() {
        if j == si || t.len <= s.len {
            continue; // must be strictly longer
        }
        if t.rbeg - i64::from(t.qbeg) != sd {
            continue; // must be the same diagonal
        }
        if s.qbeg >= t.qbeg && s.qbeg + s.len <= t.qbeg + t.len {
            has_container = true;
            break;
        }
    }
    if !has_container {
        return false;
    }
    // Interference guard (mirrors the PE18 purge): a seed >= 0.95*len overlapping s on a *different*
    // diagonal by >= s.len/4 could lead to a distinct alignment, so s must be extended after all.
    for (j, u) in seeds.iter().enumerate() {
        if j == si || (f64::from(u.len)) < f64::from(s.len) * 0.95 {
            continue;
        }
        if s.qbeg <= u.qbeg
            && s.qbeg + s.len - u.qbeg >= s.len >> 2
            && i64::from(u.qbeg) - i64::from(s.qbeg) != u.rbeg - s.rbeg
        {
            return false;
        }
        if u.qbeg <= s.qbeg
            && u.qbeg + u.len - s.qbeg >= s.len >> 2
            && i64::from(s.qbeg) - i64::from(u.qbeg) != s.rbeg - u.rbeg
        {
            return false;
        }
    }
    true
}

/// Where a region came from, so the discard pass can recover its seed. `pos` is the seed's index in
/// the chain's descending-score order, which is also the region's offset within its chain's block of
/// slots, so region `idx` of `(chain, q)` is `idx - pos + q`.
#[derive(Clone, Copy)]
pub(crate) struct RegMeta {
    pub chain: u32,
    pub pos: u32,
    pub seed: u32,
}

/// bwa-mem2's discard pass (the tail of `mem_chain2aln_across_reads_V2`, `bwamem.cpp:2895-2990`).
///
/// bwa-mem2 extends every seed up front, then walks each read's chains (seeds in descending score
/// order, the same order the collection pass emits slots in) and **purges** the region of any seed
/// that a previously-kept region already covers within the band -- unless a comparably long seed on
/// a different diagonal interferes, meaning the seed could still yield a distinct alignment. Purged
/// regions are marked `qb = qe = -1`, exactly as bwa-mem2 does; `mem_sort_dedup_patch`'s compaction
/// drops them later.
///
/// This is what keeps repeat-region reads from accumulating near-duplicate regions that survive the
/// dedup's redundancy test and inflate `sub` (hence collapse MAPQ). It is not an optimization: the
/// extensions have already run.
///
/// `lim` (seeds kept so far for this read) bounding the scan is bwa-mem2's own, and it is
/// load-bearing: it caps how many regions are examined, so the outcome depends on the **slot order**
/// of `regs[r]`. That order must therefore match bwa-mem2's `s->aln` slots, i.e. one slot per seed.
pub(crate) fn discard_contained(
    opt: &MemOpt,
    l_query: i32,
    chains: &[MemChain],
    regs: &mut Vec<MemAlnReg>,
    meta: &[RegMeta],
    preskip: &[bool],
) {
    let n = regs.len();
    let mut lim: i32 = 0;
    // bwa-mem2's `srt2[k] = UINT_MAX`. Seeds skipped up front (nh13's `seed_ext_redundant`) start
    // out purged: their slot exists to preserve scan order, but they were never extended.
    let mut purged: Vec<bool> = preskip.to_vec();
    for idx in 0..n {
        if purged[idx] {
            continue; // pre-skipped: never extended, and contributes no `lim`
        }
        let m = meta[idx];
        let c = &chains[m.chain as usize];
        let s = c.seeds[m.seed as usize];

        // "test whether extension has been made before": scan this read's regions in slot order,
        // stopping once `lim` non-purged ones have been examined without finding a container.
        let mut v: i32 = 0;
        let mut i = 0usize;
        while i < n && v < lim {
            let p = &regs[i];
            if p.qb == -1 && p.qe == -1 {
                i += 1;
                continue; // already purged: not counted against `lim`
            }
            if s.rbeg < p.rb
                || s.rbeg + i64::from(s.len) > p.re
                || s.qbeg < p.qb
                || s.qbeg + s.len > p.qe
            {
                v += 1;
                i += 1;
                continue; // not fully contained
            }
            if f64::from(s.len - p.seedlen0) > 0.1 * f64::from(l_query) {
                v += 1;
                i += 1;
                continue; // this seed may give a better alignment
            }
            // Ahead of the seed: is it "around" this hit, within the gap the band still allows?
            let qd = i64::from(s.qbeg - p.qb);
            let rd = s.rbeg - p.rb;
            let max_gap = i64::from(cal_max_gap(opt, qd.min(rd) as i32));
            let w = max_gap.min(i64::from(p.w));
            if qd - rd < w && rd - qd < w {
                break;
            }
            // Same test behind the seed.
            let qd = i64::from(p.qe - (s.qbeg + s.len));
            let rd = p.re - (s.rbeg + i64::from(s.len));
            let max_gap = i64::from(cal_max_gap(opt, qd.min(rd) as i32));
            let w = max_gap.min(i64::from(p.w));
            if qd - rd < w && rd - qd < w {
                break;
            }
            v += 1;
            i += 1;
        }

        if v < lim {
            // The seed is (almost) contained in an existing alignment. Confirm it cannot lead to a
            // different one: look for a comparably long, already-processed seed of the same chain
            // that overlaps it on a *different* diagonal. bwa scans `srt2[k+1..]` (higher scores,
            // i.e. our earlier positions); only whether one exists matters, not which.
            let first = idx - m.pos as usize; // this chain's first slot
            let mut interferes = false;
            for q in 0..m.pos as usize {
                let t_idx = first + q;
                if purged[t_idx] {
                    continue;
                }
                let t = c.seeds[meta[t_idx].seed as usize];
                if f64::from(t.len) < f64::from(s.len) * 0.95 {
                    continue;
                }
                if s.qbeg <= t.qbeg
                    && s.qbeg + s.len - t.qbeg >= s.len >> 2
                    && i64::from(t.qbeg - s.qbeg) != t.rbeg - s.rbeg
                {
                    interferes = true;
                    break;
                }
                if t.qbeg <= s.qbeg
                    && t.qbeg + t.len - s.qbeg >= s.len >> 2
                    && i64::from(s.qbeg - t.qbeg) != s.rbeg - t.rbeg
                {
                    interferes = true;
                    break;
                }
            }
            if !interferes {
                regs[idx].qb = -1;
                regs[idx].qe = -1;
                purged[idx] = true;
                continue; // purged seeds do not count towards `lim`
            }
        }
        lim += 1;
    }
}

/// One pending one-sided extension, with a back-pointer to the region it fills.
struct SideJob {
    read: usize,
    reg: usize,
    query: Vec<u8>,
    target: Vec<u8>,
    h0: i32,
    /// Previous round's score for the band-doubling acceptance test (`-1` before the first round).
    prev: i32,
    /// Still needs another (wider) band pass.
    active: bool,
}

/// Align a batch of reads (2-bit codes) through seeding, chaining, and **batched** extension,
/// returning each read's alignment regions (pre-dedup), byte-identical to [`crate::align_read`].
pub fn align_reads_batched<B: SwBackend>(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    reads: &[Vec<u8>],
    backend: &B,
) -> Vec<Vec<MemAlnReg>> {
    let l_pac = bns.l_pac;

    // Round-1 SMEMs for the whole batch in lockstep (hides FM-index latency across reads), then
    // chain each read. Result-identical to per-read `build_chains` (batched seeding is verified equal
    // to per-read `collect_smems`).
    let refs: Vec<&[u8]> = reads.iter().map(|c| c.as_slice()).collect();
    let per_read_smems = mem_collect_smem_batched(fm, &refs, opt);
    if std::env::var_os("BWA3_DUMP_SMEMS").is_some() {
        for sm in &per_read_smems {
            eprintln!("SMEM tot={}", sm.len());
            for p in sm {
                eprintln!("  smem q[{},{}) len={} s={} k={}", p.m, p.n + 1, p.n + 1 - p.m, p.s, p.k);
            }
        }
    }
    let dump_chains = std::env::var_os("BWA3_DUMP_CHAINS").is_some();
    let per_read_chains: Vec<Vec<MemChain>> = per_read_smems
        .into_iter()
        .zip(reads.iter())
        .map(|(smems, codes)| {
            let pre = build_chains_from_smems(fm, bns, opt, codes, 0, smems);
            if dump_chains {
                eprintln!("PRECHAIN nchains={}", pre.len());
                for (ci, c) in pre.iter().enumerate() {
                    eprintln!(
                        "  prechain{ci} pos={} nseed={} w={} qbeg={} qend={}",
                        c.pos,
                        c.seeds.len(),
                        bwa_chain::mem_chain_weight(c),
                        c.seeds.first().map_or(0, |s| s.qbeg),
                        c.seeds.last().map_or(0, |s| s.qbeg + s.len),
                    );
                }
            }
            let out = mem_chain_flt(opt, pre);
            if dump_chains {
                eprintln!("CHAIN nchains={}", out.len());
                for (ci, c) in out.iter().enumerate() {
                    eprintln!(
                        "  chain{ci} pos={} nseed={} w={} kept={}",
                        c.pos,
                        c.seeds.len(),
                        c.w,
                        c.kept
                    );
                }
            }
            out
        })
        .collect();

    let mut regs: Vec<Vec<MemAlnReg>> = vec![Vec::new(); reads.len()];
    // region -> owning chain index (for the final seedcov pass).
    let mut reg_chain: Vec<Vec<usize>> = vec![Vec::new(); reads.len()];
    // Per-region (chain, order position, seed) for the discard pass.
    let mut reg_meta: Vec<Vec<RegMeta>> = vec![Vec::new(); reads.len()];
    // Seeds whose extension nh13's skip elided: they keep a slot (so the discard pass reproduces
    // bwa-mem2's scan order) but start purged.
    let mut reg_preskip: Vec<Vec<bool>> = vec![Vec::new(); reads.len()];

    let mut left_jobs: Vec<SideJob> = Vec::new();
    let mut right_jobs: Vec<SideJob> = Vec::new();

    // Skip banded-SW for same-diagonal contained seeds (nh13 --skip-contained-ext). The discard pass
    // below needs one slot per seed to reproduce bwa-mem2's scan order, so a skipped seed would shift
    // every later slot; the two are mutually exclusive for now.
    let purge = discard_enabled();
    let skip_contained = skip_contained_enabled();

    // ---- collection pass: one region skeleton + up to one left and one right job per seed ----
    for (r, codes) in reads.iter().enumerate() {
        let l_query = codes.len() as i32;
        let chains = &per_read_chains[r];
        for (ci, chain) in chains.iter().enumerate() {
            if chain.seeds.is_empty() {
                continue;
            }
            // Reference window spanning the chain (mirrors mem_chain2aln).
            let mut rmax0 = l_pac << 1;
            let mut rmax1 = 0i64;
            for s in &chain.seeds {
                let b = s.rbeg - (i64::from(s.qbeg) + i64::from(cal_max_gap(opt, s.qbeg)));
                let tail = l_query - s.qbeg - s.len;
                let e = s.rbeg
                    + i64::from(s.len)
                    + (i64::from(tail) + i64::from(cal_max_gap(opt, tail)));
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

            for (pos, &si) in order.iter().enumerate() {
                if skip_contained && seed_ext_redundant(&chain.seeds, si) {
                    // Keep the slot, skip the DP: the discard pass would purge this seed anyway
                    // (its container is a longer same-diagonal seed, extended earlier).
                    reg_chain[r].push(ci);
                    reg_meta[r].push(RegMeta { chain: ci as u32, pos: pos as u32, seed: si as u32 });
                    reg_preskip[r].push(true);
                    regs[r].push(MemAlnReg {
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

                // Left extension job, or seed-terminal left edge.
                if s.qbeg > 0 {
                    let query: Vec<u8> = (0..s.qbeg).rev().map(|i| codes[i as usize]).collect();
                    let rlen = (s.rbeg - rmax0) as usize;
                    let target: Vec<u8> = (0..rlen).rev().map(|i| rseq[i]).collect();
                    a.qb = s.qbeg;
                    a.rb = s.rbeg;
                    left_jobs.push(SideJob {
                        read: r,
                        reg: regs[r].len(),
                        query,
                        target,
                        h0: s.len * opt.a,
                        prev: -1,
                        active: true,
                    });
                } else {
                    a.score = s.len * opt.a;
                    a.truesc = a.score;
                    a.qb = 0;
                    a.rb = s.rbeg;
                }

                // Right extension job, or seed-terminal right edge.
                if s.qbeg + s.len != l_query {
                    let qe = s.qbeg + s.len;
                    let re = s.rbeg + i64::from(s.len) - rmax0;
                    let query: Vec<u8> = codes[qe as usize..].to_vec();
                    let target: Vec<u8> = rseq[re as usize..].to_vec();
                    a.qe = qe;
                    a.re = rmax0 + re;
                    right_jobs.push(SideJob {
                        read: r,
                        reg: regs[r].len(),
                        query,
                        target,
                        h0: H0_SENTINEL as i32, // filled from a.score after left completes
                        prev: -1,
                        active: true,
                    });
                } else {
                    a.qe = l_query;
                    a.re = s.rbeg + i64::from(s.len);
                }

                reg_chain[r].push(ci);
                reg_meta[r].push(RegMeta { chain: ci as u32, pos: pos as u32, seed: si as u32 });
                reg_preskip[r].push(false);
                regs[r].push(a);
            }
        }
    }

    // ---- left extensions (h0 already set), then fill right h0 and run right extensions ----
    run_side(backend, opt, &mut left_jobs, &mut regs, opt.pen_clip5, true);

    for j in &mut right_jobs {
        j.h0 = regs[j.read][j.reg].score;
    }
    run_side(
        backend,
        opt,
        &mut right_jobs,
        &mut regs,
        opt.pen_clip3,
        false,
    );

    // ---- seedcov, per region, from final bounds (mirrors mem_chain2aln's tail) ----
    for r in 0..reads.len() {
        for (idx, a) in regs[r].iter_mut().enumerate() {
            if a.rb != H0_SENTINEL && a.qb != H0_SENTINEL as i32 && a.qb != -1 {
                let chain = &per_read_chains[r][reg_chain[r][idx]];
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
        }
    }

    // ---- bwa-mem2's discard pass, after every extension has landed ----
    if purge {
        for r in 0..reads.len() {
            let l_query = reads[r].len() as i32;
            discard_contained(
                opt,
                l_query,
                &per_read_chains[r],
                &mut regs[r],
                &reg_meta[r],
                &reg_preskip[r],
            );
        }
    }

    // Drop the purged regions before returning, as bwa-mem2 does between the discard pass and
    // `mem_sort_dedup_patch`. Not cosmetic: the dedup sorts by `re` with an *unstable* introsort, so
    // leaving purged entries in would change the array it partitions and hence the order of
    // equal-`re` regions -- which is exactly what decides who survives a score tie.
    for r in regs.iter_mut() {
        r.retain(|a| a.qe > a.qb);
    }

    regs
}

/// Whether bwa-mem2's post-extension discard pass runs (`BWA3_NO_DISCARD` opts out).
pub(crate) fn discard_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BWA3_NO_DISCARD").is_none())
}

/// Run one side (left or right) of all pending extensions through `MAX_BAND_TRY` band-doubling
/// rounds. Each round batches the still-active jobs (sorted by length so SIMD lanes pack tightly),
/// applies the exact `extend_side` acceptance test, and scatters accepted results into their region.
fn run_side<B: SwBackend>(
    backend: &B,
    opt: &MemOpt,
    jobs: &mut [SideJob],
    regs: &mut [Vec<MemAlnReg>],
    pen_clip: i32,
    is_left: bool,
) {
    for i in 0..MAX_BAND_TRY {
        // Collect active job indices, sorted by length to cluster similar sizes in a batch.
        let mut idxs: Vec<usize> = (0..jobs.len()).filter(|&k| jobs[k].active).collect();
        if idxs.is_empty() {
            break;
        }
        idxs.sort_by_key(|&k| std::cmp::Reverse(jobs[k].query.len().max(jobs[k].target.len())));

        let w = opt.w << i;
        let ejobs: Vec<ExtendJob> = idxs
            .iter()
            .map(|&k| ExtendJob {
                query: &jobs[k].query,
                target: &jobs[k].target,
                h0: jobs[k].h0,
            })
            .collect();
        let results = backend.extend_batch(
            &ejobs, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, w, pen_clip, opt.zdrop,
        );

        for (pos, &k) in idxs.iter().enumerate() {
            let res = &results[pos];
            let prev = jobs[k].prev;
            let score = res.score;
            let accept =
                score == prev || res.max_off < (w >> 1) + (w >> 2) || i + 1 == MAX_BAND_TRY;
            if !accept {
                jobs[k].prev = score;
                continue;
            }
            jobs[k].active = false;
            let a = &mut regs[jobs[k].read][jobs[k].reg];
            if is_left {
                a.score = score;
                if res.gscore <= 0 || res.gscore <= score - pen_clip {
                    a.qb -= res.qle;
                    a.rb -= i64::from(res.tle);
                    a.truesc = score;
                } else {
                    a.qb = 0;
                    a.rb -= i64::from(res.gtle);
                    a.truesc = res.gscore;
                }
            } else {
                let h0 = jobs[k].h0;
                a.score = score;
                if res.gscore <= 0 || res.gscore <= score - pen_clip {
                    a.qe += res.qle;
                    a.re += i64::from(res.tle);
                    a.truesc += score - h0;
                } else {
                    // qe = l_query: the region's read length. a.qe was set to (s.qbeg + s.len) at
                    // collection; the query slice length is (l_query - qe), so restore l_query.
                    a.qe += jobs[k].query.len() as i32;
                    a.re += i64::from(res.gtle);
                    a.truesc += res.gscore - h0;
                }
            }
            a.w = a.w.max(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align_read;
    use bwa_neon::NeonBackend;
    use std::path::Path;

    fn tiny() -> (FmIndex, BntSeq) {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    /// The batched across-reads path (through the NEON backend) must produce, for every read, the
    /// exact same regions as calling `align_read` per read. A diverse read set (forward / RC slices,
    /// mismatches, insertions, deletions, truncations) exercises left+right extension, band-doubling,
    /// and the gscore/z-drop branches.
    #[test]
    fn batched_across_reads_equals_per_read() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        let l_ref = bns.l_pac;

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..400 {
            let len = 60 + (next() % 120) as i64; // 60..180
            let start = (next() as i64) % (l_ref - len - 1).max(1);
            let mut r: Vec<u8> = (0..len).map(|i| fm.base(start + i)).collect();
            // Perturb: mismatches, an insertion, a deletion.
            let nmut = (next() % 6) as usize;
            for _ in 0..nmut {
                let p = (next() as usize) % r.len();
                r[p] = ((r[p] as u64 + 1 + next() % 3) % 4) as u8;
            }
            if next() % 3 == 0 && r.len() > 20 {
                let p = (next() as usize) % r.len();
                r.insert(p, (next() % 4) as u8); // insertion
            }
            if next() % 3 == 0 && r.len() > 20 {
                let p = (next() as usize) % r.len();
                r.remove(p); // deletion
            }
            // Reverse-complement half of them.
            if next() % 2 == 0 {
                r = r
                    .iter()
                    .rev()
                    .map(|&c| if c < 4 { 3 - c } else { c })
                    .collect();
            }
            reads.push(r);
        }

        let batched = align_reads_batched(&fm, &bns, &opt, &reads, &NeonBackend);
        assert_eq!(batched.len(), reads.len());
        for (i, codes) in reads.iter().enumerate() {
            let per_read = align_read(&fm, &bns, &opt, codes);
            assert_eq!(
                format!("{:?}", batched[i]),
                format!("{:?}", per_read),
                "read {i} (len {}) diverged between batched and per-read extension",
                codes.len()
            );
        }
    }
}

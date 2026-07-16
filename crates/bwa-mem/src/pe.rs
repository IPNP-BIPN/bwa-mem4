//! Paired-end pairing, insert-size estimation and SAM emission, mirroring bwa-mem2's
//! `mem_pestat` / `mem_pair` / `mem_sam_pe` / `mem_aln2sam` (`reference/bwa-mem2/src/bwamem_pair.cpp`
//! and `bwamem.cpp`).
//!
//! Mate rescue (`mem_matesw`) uses `ksw_align2`; on concordant pairs all orientations are skipped
//! so it performs no Smith-Waterman and leaves the region set untouched.

use std::io::{self, Write};

use bwa_core::MemOpt;
use bwa_extend::{ksw_align2, KswAlignResult};
use bwa_index::{BntSeq, FmIndex};
use bwa_neon::{batched_ksw_align2, KswJob};

use crate::alt::mem_gen_alt;
use crate::cigar::{reg2aln, MemAln};
use crate::primary::{hash_64, mem_approx_mapq_se, mem_mark_primary_se, mem_sort_dedup_patch};
use crate::MemAlnReg;

extern "C" {
    /// System libm complementary error function, for bit-identical pairing scores.
    fn erfc(x: f64) -> f64;
}

const MIN_RATIO: f64 = 0.8;
const MIN_DIR_CNT: usize = 10;
const MIN_DIR_RATIO: f64 = 0.05;
const OUTLIER_BOUND: f64 = 2.0;
const MAPPING_BOUND: f64 = 3.0;
const MAX_STDDEV: f64 = 4.0;
const M_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Per-orientation insert-size statistics (`mem_pestat_t`).
#[derive(Debug, Clone, Copy)]
pub struct PeStat {
    pub low: i32,
    pub high: i32,
    pub failed: bool,
    pub avg: f64,
    pub std: f64,
}

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

/// Infer the relative orientation and distance of two 5' coordinates. Port of `mem_infer_dir`.
fn mem_infer_dir(l_pac: i64, b1: i64, b2: i64) -> (usize, i64) {
    let r1 = (b1 >= l_pac) as i64;
    let r2 = (b2 >= l_pac) as i64;
    // p2: read-2 coordinate projected onto read-1's strand.
    let p2 = if r1 == r2 { b2 } else { (l_pac << 1) - 1 - b2 };
    let dist = if p2 > b1 { p2 - b1 } else { b1 - p2 };
    let base = if r1 == r2 { 0 } else { 1 };
    let dir = (base ^ if p2 > b1 { 0 } else { 3 }) as usize;
    (dir, dist)
}

/// Fetch reference `[rb, re)` (in 2*l_pac space) clamped to the contig containing `mid`, returning
/// the clamped bounds, the contig id, and the nt4 sequence (`.0123`, forward++reverse-complement).
/// Port of `bns_fetch_seq`.
fn bns_fetch_seq(
    fm: &FmIndex,
    bns: &BntSeq,
    rb: i64,
    mid: i64,
    re: i64,
) -> (i64, i64, i32, Vec<u8>) {
    let (mut rb, mut re) = if re < rb { (re, rb) } else { (rb, re) };
    let (dep, is_rev) = bns.depos(mid);
    let rid = bns.pos2rid(dep);
    let (mut far_beg, mut far_end) = if rid >= 0 {
        let c = &bns.contigs[rid as usize];
        (c.offset, c.offset + i64::from(c.len))
    } else {
        (0, bns.l_pac << 1)
    };
    if is_rev {
        let tmp = far_beg;
        far_beg = (bns.l_pac << 1) - far_end;
        far_end = (bns.l_pac << 1) - tmp;
    }
    rb = rb.max(far_beg);
    re = re.min(far_end);
    let seq: Vec<u8> = (rb..re).map(|p| fm.base(p)).collect();
    (rb, re, rid, seq)
}

/// Smith-Waterman mate rescue: given an anchor region `a` for one read, try to align the mate `ms`
/// (nt4) in each insert-consistent orientation not already satisfied, appending any hit to `ma`.
/// Port of `mem_matesw` (non-MATE_SORT path). Returns the number of SW attempts that ran.
fn mem_matesw(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &MemAlnReg,
    ms: &[u8],
    ma: &mut Vec<MemAlnReg>,
) -> i32 {
    let l_pac = bns.l_pac;
    let l_ms = ms.len() as i64;
    let mut skip = [0i32; 4];
    for (r, sk) in skip.iter_mut().enumerate() {
        *sk = i32::from(pes[r].failed);
    }
    for m in ma.iter() {
        let (r, dist) = mem_infer_dir(l_pac, a.rb, m.rb);
        if dist >= i64::from(pes[r].low) && dist <= i64::from(pes[r].high) {
            skip[r] = 1;
        }
    }
    if skip.iter().sum::<i32>() == 4 {
        return 0; // a consistent pair already exists; no SW needed
    }

    let mut n = 0;
    for r in 0..4 {
        if skip[r] != 0 {
            continue;
        }
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;
        let seq: Vec<u8> = if is_rev {
            // reverse complement of ms
            ms.iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            ms.to_vec()
        };
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
            let (rb, re, rid, refseq) = bns_fetch_seq(fm, bns, rb0, (rb0 + re0) >> 1, re0);
            if a.rid == rid && re - rb >= i64::from(opt.min_seed_len) {
                let minsc = opt.min_seed_len * opt.a;
                let aln = ksw_align2(
                    &seq, &refseq, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, minsc,
                    opt.a,
                );
                if dump_pestat() {
                    eprintln!(
                        "[RESCUE] score={} score2={} qb={} qe={} tb={} te={} te2={} tlen={}",
                        aln.score, aln.score2, aln.qb, aln.qe, aln.tb, aln.te, aln.te2,
                        refseq.len()
                    );
                }
                if aln.score >= opt.min_seed_len && aln.qb >= 0 {
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
                    let brb = if is_rev {
                        (l_pac << 1) - (rb + i64::from(aln.te) + 1)
                    } else {
                        rb + i64::from(aln.tb)
                    };
                    let bre = if is_rev {
                        (l_pac << 1) - (rb + i64::from(aln.tb))
                    } else {
                        rb + i64::from(aln.te) + 1
                    };
                    let seedcov = ((bre - brb).min(i64::from(qe - qb)) >> 1) as i32;
                    let b = MemAlnReg {
                        rb: brb,
                        re: bre,
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
                        hash: 0,
                        n_comp: 0,
                    };
                    // Insert keeping `ma` sorted by score descending (bwa's manual insertion).
                    let mut ins = 0;
                    while ins < ma.len() && ma[ins].score >= b.score {
                        ins += 1;
                    }
                    ma.insert(ins, b);
                }
                n += 1;
            }
        }
        // Dedup after each orientation, null query (merging disabled), per mem_matesw.
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
struct Orient {
    query: Vec<u8>,
    target: Vec<u8>,
    rb: i64,
    is_rev: bool,
}

/// One anchor's mate rescue, split into the SW-independent `collect` (which orientations to run) and
/// the SW-dependent `apply` (insert the hits). Mirrors [`mem_matesw`] exactly but lets the SW of many
/// anchors, across the whole pair batch, run through one vectorized [`batched_ksw_align2`].
struct RescueCall {
    skip: [i32; 4],
    per_r: [Option<Orient>; 4],
    l_ms: i64,
    rid: i32,
    is_alt: bool,
}

/// Collect the orientations that would run mate rescue for anchor `a` against mate `ms`, reading the
/// current `ma`. Returns `None` when a consistent pair already exists (all four skipped — no SW), so
/// the caller records nothing. The SW jobs depend only on `(query, target)`, not on `ma`.
fn matesw_collect(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &MemAlnReg,
    ms: &[u8],
    ma: &[MemAlnReg],
) -> Option<RescueCall> {
    let l_pac = bns.l_pac;
    let l_ms = ms.len() as i64;
    let mut skip = [0i32; 4];
    for (r, sk) in skip.iter_mut().enumerate() {
        *sk = i32::from(pes[r].failed);
    }
    for m in ma.iter() {
        let (r, dist) = mem_infer_dir(l_pac, a.rb, m.rb);
        if dist >= i64::from(pes[r].low) && dist <= i64::from(pes[r].high) {
            skip[r] = 1;
        }
    }
    if skip.iter().sum::<i32>() == 4 {
        return None;
    }
    let mut per_r: [Option<Orient>; 4] = [None, None, None, None];
    for r in 0..4 {
        if skip[r] != 0 {
            continue;
        }
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;
        let seq: Vec<u8> = if is_rev {
            ms.iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            ms.to_vec()
        };
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
            let (rb, re, rid, refseq) = bns_fetch_seq(fm, bns, rb0, (rb0 + re0) >> 1, re0);
            if a.rid == rid && re - rb >= i64::from(opt.min_seed_len) {
                per_r[r] = Some(Orient {
                    query: seq,
                    target: refseq,
                    rb,
                    is_rev,
                });
            }
        }
    }
    Some(RescueCall {
        skip,
        per_r,
        l_ms,
        rid: a.rid,
        is_alt: a.is_alt,
    })
}

/// The `KswJob`s of a `RescueCall`, in orientation order (the order `matesw_apply` consumes results).
fn rescue_jobs(call: &RescueCall) -> impl Iterator<Item = KswJob<'_>> {
    call.per_r.iter().flatten().map(|o| KswJob {
        query: &o.query,
        target: &o.target,
    })
}

/// Apply a `RescueCall`'s SW results (`alns`, one per collected orientation, in order) to `ma`,
/// inserting each accepted hit and deduping after each orientation, exactly as [`mem_matesw`] does.
fn matesw_apply(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    call: &RescueCall,
    alns: &[KswAlignResult],
    ma: &mut Vec<MemAlnReg>,
) {
    let l_pac = bns.l_pac;
    let l_ms = call.l_ms;
    let mut n = 0;
    let mut ai = 0usize;
    for r in 0..4 {
        if call.skip[r] != 0 {
            continue;
        }
        if let Some(o) = &call.per_r[r] {
            let aln = alns[ai];
            ai += 1;
            if aln.score >= opt.min_seed_len && aln.qb >= 0 {
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
                let brb = if is_rev {
                    (l_pac << 1) - (rb + i64::from(aln.te) + 1)
                } else {
                    rb + i64::from(aln.tb)
                };
                let bre = if is_rev {
                    (l_pac << 1) - (rb + i64::from(aln.tb))
                } else {
                    rb + i64::from(aln.te) + 1
                };
                let seedcov = ((bre - brb).min(i64::from(qe - qb)) >> 1) as i32;
                let b = MemAlnReg {
                    rb: brb,
                    re: bre,
                    qb,
                    qe,
                    rid: call.rid,
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
                    is_alt: call.is_alt,
                    hash: 0,
                    n_comp: 0,
                };
                let mut ins = 0;
                while ins < ma.len() && ma[ins].score >= b.score {
                    ins += 1;
                }
                ma.insert(ins, b);
            }
            n += 1;
        }
        if n > 0 {
            let taken = std::mem::take(ma);
            *ma = mem_sort_dedup_patch(fm, opt, &[], taken);
        }
    }
}

/// One read pair's rescue inputs for [`batch_mate_rescue`]: the two mates' nt4 codes and their
/// (mutable) dedup'd region vectors. `a0`/`seq0` is read 1, `a1`/`seq1` is read 2.
pub struct PairRescueData<'a> {
    pub seq0: &'a [u8],
    pub seq1: &'a [u8],
    pub a0: &'a mut Vec<MemAlnReg>,
    pub a1: &'a mut Vec<MemAlnReg>,
}

/// Batched mate rescue across a whole pair batch: identical to running [`mem_matesw`] inside each
/// pair's `mem_sam_pe`, but every anchor's insert-window SW (across all pairs) runs in one vectorized
/// [`batched_ksw_align2`], filling the SIMD lanes that a single pair's <=4 orientations cannot.
///
/// bwa-mem2's rescue snapshots each read's near-best regions as anchors (before any rescue), then, per
/// anchor, SW-rescues the mate in each missing orientation. Anchors of one read against one mate are
/// applied in order (later anchors' `skip` sees earlier inserts), so this proceeds in **rounds**: round
/// `k` collects the `k`-th anchor of every (pair, direction), batches their SW, then applies — keeping
/// the per-array insertion order byte-identical to the per-pair path. The two directions target
/// disjoint arrays (`a1` vs `a0`), and different pairs are independent, so a round batches freely.
pub fn batch_mate_rescue(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    pairs: &mut [PairRescueData],
) {
    let cap = opt.max_matesw.max(0) as usize;
    if cap == 0 {
        return;
    }
    let pen = opt.pen_unpaired;

    // Snapshot near-best anchors per (pair, direction), before any rescue. dir 0: read-1 anchors that
    // rescue read 2 (`a1`); dir 1: read-2 anchors that rescue read 1 (`a0`).
    let mut anchors: Vec<[Vec<MemAlnReg>; 2]> = Vec::with_capacity(pairs.len());
    let mut max_rounds = 0usize;
    for p in pairs.iter() {
        let a = if !p.a0.is_empty() && !p.a1.is_empty() {
            let best0 = p.a0[0].score;
            let best1 = p.a1[0].score;
            let b0: Vec<MemAlnReg> = p
                .a0
                .iter()
                .filter(|r| r.score >= best0 - pen)
                .take(cap)
                .cloned()
                .collect();
            let b1: Vec<MemAlnReg> = p
                .a1
                .iter()
                .filter(|r| r.score >= best1 - pen)
                .take(cap)
                .cloned()
                .collect();
            max_rounds = max_rounds.max(b0.len()).max(b1.len());
            [b0, b1]
        } else {
            [Vec::new(), Vec::new()]
        };
        anchors.push(a);
    }

    for round in 0..max_rounds {
        // Collect this round's rescue calls across all pairs and both directions.
        let mut calls: Vec<(usize, usize, RescueCall)> = Vec::new(); // (pair, dir, call)
        for (pi, p) in pairs.iter().enumerate() {
            for dir in 0..2 {
                let anc = &anchors[pi][dir];
                if round >= anc.len() {
                    continue;
                }
                let (ms, target): (&[u8], &Vec<MemAlnReg>) = if dir == 0 {
                    (p.seq1, p.a1)
                } else {
                    (p.seq0, p.a0)
                };
                if let Some(call) = matesw_collect(fm, bns, opt, pes, &anc[round], ms, target) {
                    calls.push((pi, dir, call));
                }
            }
        }
        if calls.is_empty() {
            continue;
        }
        // Flatten every call's orientation jobs into one batch (spans map each call to its slice).
        let mut jobs: Vec<KswJob> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (_, _, call) in &calls {
            let start = jobs.len();
            jobs.extend(rescue_jobs(call));
            spans.push((start, jobs.len() - start));
        }
        let alns = batched_ksw_align2(
            &jobs, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins,
            opt.min_seed_len * opt.a, opt.a,
        );
        for (idx, (pi, dir, call)) in calls.iter().enumerate() {
            let (start, count) = spans[idx];
            let target = if *dir == 0 {
                &mut *pairs[*pi].a1
            } else {
                &mut *pairs[*pi].a0
            };
            matesw_apply(fm, bns, opt, call, &alns[start..start + count], target);
        }
    }
}

/// Second-best non-overlapping score among a read's dedup'd regions (`a[0]` is the best). Port of
/// `cal_sub`.
fn cal_sub(opt: &MemOpt, r: &[MemAlnReg]) -> i32 {
    let mut j = 1;
    while j < r.len() {
        let b_max = r[j].qb.max(r[0].qb);
        let e_min = r[j].qe.min(r[0].qe);
        if e_min > b_max {
            let min_l = (r[j].qe - r[j].qb).min(r[0].qe - r[0].qb);
            if f64::from(e_min - b_max) >= f64::from(min_l) * f64::from(opt.mask_level) {
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

/// Estimate insert-size distributions for the four orientations over a whole batch. `regs` holds the
/// dedup'd regions of `2N` interleaved reads (`regs[2i]`=R1, `regs[2i+1]`=R2). Port of `mem_pestat`.
/// Env-gated (`BWA3_DUMP_PESTAT`) insert-size stats, in bwa's `[PE]` wording so the two diff
/// directly. The distribution is per batch, so a difference here moves rescue decisions for every
/// pair in it, not just one read.
fn dump_pestat() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA3_DUMP_PESTAT").is_some())
}

pub fn mem_pestat(opt: &MemOpt, l_pac: i64, regs: &[&[MemAlnReg]]) -> [PeStat; 4] {
    let mut pes = [PeStat::default(); 4];
    let mut isize: [Vec<i64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    let n_pairs = regs.len() / 2;
    for i in 0..n_pairs {
        let r0 = regs[i << 1];
        let r1 = regs[(i << 1) | 1];
        if r0.is_empty() || r1.is_empty() {
            continue;
        }
        if f64::from(cal_sub(opt, r0)) > MIN_RATIO * f64::from(r0[0].score) {
            continue;
        }
        if f64::from(cal_sub(opt, r1)) > MIN_RATIO * f64::from(r1[0].score) {
            continue;
        }
        if r0[0].rid != r1[0].rid {
            continue;
        }
        let (dir, is) = mem_infer_dir(l_pac, r0[0].rb, r1[0].rb);
        if is != 0 && is <= i64::from(opt.max_ins) {
            isize[dir].push(is);
        }
    }

    if dump_pestat() {
        eprintln!(
            "[PE] # candidate unique pairs for (FF, FR, RF, RR): ({}, {}, {}, {})",
            isize[0].len(),
            isize[1].len(),
            isize[2].len(),
            isize[3].len()
        );
    }
    for d in 0..4 {
        let r = &mut pes[d];
        let q = &mut isize[d];
        if q.len() < MIN_DIR_CNT {
            r.failed = true;
            continue;
        }
        q.sort_unstable();
        let n = q.len();
        let p25 = q[(0.25 * n as f64 + 0.499) as usize] as i32;
        let p75 = q[(0.75 * n as f64 + 0.499) as usize] as i32;
        r.failed = false;
        r.low = (f64::from(p25) - OUTLIER_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        if r.low < 1 {
            r.low = 1;
        }
        r.high = (f64::from(p75) + OUTLIER_BOUND * f64::from(p75 - p25) + 0.499) as i32;

        let (mut sum, mut x) = (0.0f64, 0i64);
        for &v in q.iter() {
            if v >= i64::from(r.low) && v <= i64::from(r.high) {
                sum += v as f64;
                x += 1;
            }
        }
        r.avg = sum / x as f64;
        let mut var = 0.0f64;
        for &v in q.iter() {
            if v >= i64::from(r.low) && v <= i64::from(r.high) {
                var += (v as f64 - r.avg) * (v as f64 - r.avg);
            }
        }
        r.std = (var / x as f64).sqrt();
        r.low = (f64::from(p25) - MAPPING_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        r.high = (f64::from(p75) + MAPPING_BOUND * f64::from(p75 - p25) + 0.499) as i32;
        if f64::from(r.low) > r.avg - MAX_STDDEV * r.std {
            r.low = (r.avg - MAX_STDDEV * r.std + 0.499) as i32;
        }
        if f64::from(r.high) < r.avg + MAX_STDDEV * r.std {
            r.high = (r.avg + MAX_STDDEV * r.std + 0.499) as i32;
        }
        if r.low < 1 {
            r.low = 1;
        }
        if dump_pestat() {
            eprintln!(
                "[PE] dir={d} (25, 50, 75) percentile: ({p25}, {}, {p75})",
                q[(0.50 * n as f64 + 0.499) as usize]
            );
            eprintln!("[PE] dir={d} mean and std.dev: ({:.2}, {:.2})", r.avg, r.std);
            eprintln!("[PE] dir={d} low and high boundaries for proper pairs: ({}, {})", r.low, r.high);
        }
    }

    let max = isize.iter().map(Vec::len).max().unwrap_or(0);
    for d in 0..4 {
        if !pes[d].failed && (isize[d].len() as f64) < max as f64 * MIN_DIR_RATIO {
            pes[d].failed = true;
        }
    }
    pes
}

/// Result of `mem_pair`: combined score, sub-optimal, `n_sub`, and the best region index per read.
struct PairResult {
    score: i32,
    sub: i32,
    n_sub: i32,
    z: [usize; 2],
}

/// Pair the two ends' regions and pick the best proper pair. Port of `mem_pair` (non-ALT: `n_pri`
/// == region count). `id` is the global pair index.
fn mem_pair(
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    a: &[&[MemAlnReg]; 2],
    id: u64,
) -> Option<PairResult> {
    let l_pac = bns.l_pac;
    // v: (x = rid<<32 | fwd-pos-in-contig, y = score<<32 | i<<2 | strand<<1 | read).
    let mut v: Vec<(u64, u64)> = Vec::new();
    for (r, ar) in a.iter().enumerate() {
        for (i, e) in ar.iter().enumerate() {
            let fpos = if e.rb < l_pac {
                e.rb
            } else {
                (l_pac << 1) - 1 - e.rb
            };
            let off = bns.contigs[e.rid as usize].offset;
            let x = (u64::from(e.rid as u32) << 32) | (fpos - off) as u64;
            let y = ((e.score as u64) << 32)
                | ((i as u64) << 2)
                | (((e.rb >= l_pac) as u64) << 1)
                | r as u64;
            v.push((x, y));
        }
    }
    v.sort_unstable();

    let mut u: Vec<(u64, u64)> = Vec::new();
    let mut y: [i64; 4] = [-1, -1, -1, -1];
    for i in 0..v.len() {
        for r in 0..2u64 {
            let dir = (r << 1 | ((v[i].1 >> 1) & 1)) as usize;
            if pes[dir].failed {
                continue;
            }
            let which = (r << 1 | ((v[i].1 & 1) ^ 1)) as usize;
            if y[which] < 0 {
                continue;
            }
            let mut k = y[which];
            while k >= 0 {
                let ku = k as usize;
                if (v[ku].1 & 3) as usize != which {
                    k -= 1;
                    continue;
                }
                let dist = v[i].0 as i64 - v[ku].0 as i64;
                if dist > i64::from(pes[dir].high) {
                    break;
                }
                if dist < i64::from(pes[dir].low) {
                    k -= 1;
                    continue;
                }
                let ns = (dist as f64 - pes[dir].avg) / pes[dir].std;
                let erfc_term = unsafe { erfc(ns.abs() * M_SQRT1_2) };
                let q = ((v[i].1 >> 32) as f64
                    + (v[ku].1 >> 32) as f64
                    + 0.721 * (2.0 * erfc_term).ln() * f64::from(opt.a)
                    + 0.499) as i64;
                let q = q.max(0) as u64;
                let py = (k as u64) << 32 | i as u64;
                let px = (q << 32) | (hash_64(py ^ (id << 8)) & 0xffff_ffff);
                u.push((px, py));
                k -= 1;
            }
        }
        y[(v[i].1 & 3) as usize] = i as i64;
    }

    if u.is_empty() {
        return None;
    }
    let mut tmp = opt.a + opt.b;
    tmp = tmp.max(opt.o_del + opt.e_del).max(opt.o_ins + opt.e_ins);
    u.sort_unstable();
    let last = *u.last().unwrap();
    let i = (last.1 >> 32) as usize;
    let k = (last.1 & 0xffff_ffff) as usize;
    let mut z = [0usize; 2];
    z[(v[i].1 & 1) as usize] = (v[i].1 << 32 >> 34) as usize;
    z[(v[k].1 & 1) as usize] = (v[k].1 << 32 >> 34) as usize;
    let score = (last.0 >> 32) as i32;
    let sub = if u.len() > 1 {
        (u[u.len() - 2].0 >> 32) as i32
    } else {
        0
    };
    let mut n_sub = 0;
    if u.len() >= 2 {
        for e in u[..u.len() - 1].iter().rev() {
            if sub - (e.0 >> 32) as i32 <= tmp {
                n_sub += 1;
            }
        }
    }
    Some(PairResult {
        score,
        sub,
        n_sub,
        z,
    })
}

#[inline]
fn raw_mapq(diff: i32, a: i32) -> i32 {
    (6.02 * f64::from(diff) / f64::from(a) + 0.499) as i32
}

/// Reference length consumed by a CIGAR (M/D ops). Port of `get_rlen`.
fn get_rlen(cigar: &[u32]) -> i64 {
    let mut l = 0i64;
    for &c in cigar {
        let op = c & 0xf;
        if op == 0 || op == 2 {
            l += i64::from(c >> 4);
        }
    }
    l
}

/// Append a CIGAR string, converting soft-clips to hard-clips for supplementary alignments
/// (`which != 0`). Port of `add_cigar` (non-ALT, no `-Y` soft-clip flag).
pub(crate) fn add_cigar(cigar: &[u32], which: usize, out: &mut Vec<u8>) {
    if cigar.is_empty() {
        out.push(b'*');
        return;
    }
    const OPS: [u8; 5] = [b'M', b'I', b'D', b'S', b'H'];
    for &c in cigar {
        let mut op = (c & 0xf) as usize;
        if (op == 3 || op == 4) && which != 0 {
            op = 4; // hard-clip on supplementary
        }
        out.extend_from_slice((c >> 4).to_string().as_bytes());
        out.push(OPS[op]);
    }
}

/// Emit one SAM record for read `which` of `list`, with optional mate `m`. Port of `mem_aln2sam`
/// (SE/PE subset: no RG/XR/XA/pa, non-ALT). `seq` is nt4-encoded in sequencing orientation.
#[allow(clippy::too_many_arguments)]
fn mem_aln2sam(
    bns: &BntSeq,
    name: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    list: &[MemAln],
    which: usize,
    m: Option<&MemAln>,
    out: &mut Vec<u8>,
) {
    let mut p = list[which].clone();
    let mut m = m.cloned();

    // Flags + mate coordinate copy.
    if m.is_some() {
        p.flag |= 0x1;
    }
    if p.rid < 0 {
        p.flag |= 0x4;
    }
    if m.as_ref().map(|x| x.rid < 0).unwrap_or(false) {
        p.flag |= 0x8;
    }
    if p.rid < 0 {
        if let Some(mm) = m.as_ref() {
            if mm.rid >= 0 {
                p.rid = mm.rid;
                p.pos = mm.pos;
                p.is_rev = mm.is_rev;
                p.cigar.clear();
            }
        }
    }
    if p.rid >= 0 {
        if let Some(mm) = m.as_mut() {
            if mm.rid < 0 {
                mm.rid = p.rid;
                mm.pos = p.pos;
                mm.is_rev = p.is_rev;
                mm.cigar.clear();
            }
        }
    }
    if p.is_rev {
        p.flag |= 0x10;
    }
    if m.as_ref().map(|x| x.is_rev).unwrap_or(false) {
        p.flag |= 0x20;
    }

    // QNAME, FLAG.
    out.extend_from_slice(name.as_bytes());
    out.push(b'\t');
    let flag = (p.flag & 0xffff) | if p.flag & 0x10000 != 0 { 0x100 } else { 0 };
    out.extend_from_slice(flag.to_string().as_bytes());
    out.push(b'\t');

    // RNAME, POS, MAPQ, CIGAR.
    if p.rid >= 0 {
        out.extend_from_slice(bns.contigs[p.rid as usize].name.as_bytes());
        out.push(b'\t');
        out.extend_from_slice((p.pos + 1).to_string().as_bytes());
        out.push(b'\t');
        out.extend_from_slice(p.mapq.to_string().as_bytes());
        out.push(b'\t');
        add_cigar(&p.cigar, which, out);
    } else {
        out.extend_from_slice(b"*\t0\t0\t*");
    }
    out.push(b'\t');

    // RNEXT, PNEXT, TLEN.
    match m.as_ref() {
        Some(mm) if mm.rid >= 0 => {
            if p.rid == mm.rid {
                out.push(b'=');
            } else {
                out.extend_from_slice(bns.contigs[mm.rid as usize].name.as_bytes());
            }
            out.push(b'\t');
            out.extend_from_slice((mm.pos + 1).to_string().as_bytes());
            out.push(b'\t');
            if p.rid == mm.rid {
                let p0 = p.pos + if p.is_rev { get_rlen(&p.cigar) - 1 } else { 0 };
                let p1 = mm.pos
                    + if mm.is_rev {
                        get_rlen(&mm.cigar) - 1
                    } else {
                        0
                    };
                if mm.cigar.is_empty() || p.cigar.is_empty() {
                    out.push(b'0');
                } else {
                    let sign = match p0.cmp(&p1) {
                        std::cmp::Ordering::Greater => 1,
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                    };
                    out.extend_from_slice((-(p0 - p1 + sign)).to_string().as_bytes());
                }
            } else {
                out.push(b'0');
            }
        }
        _ => out.extend_from_slice(b"*\t0\t0"),
    }
    out.push(b'\t');

    // SEQ, QUAL.
    if p.flag & 0x100 != 0 {
        out.extend_from_slice(b"*\t*");
    } else {
        let (mut qb, mut qe) = (0usize, seq.len());
        // Hard-clip trimming for supplementary alignments.
        if !p.cigar.is_empty() && which != 0 {
            let first = p.cigar[0] & 0xf;
            let last = p.cigar[p.cigar.len() - 1] & 0xf;
            if !p.is_rev {
                if first == 4 || first == 3 {
                    qb += (p.cigar[0] >> 4) as usize;
                }
                if last == 4 || last == 3 {
                    qe -= (p.cigar[p.cigar.len() - 1] >> 4) as usize;
                }
            } else {
                if first == 4 || first == 3 {
                    qe -= (p.cigar[0] >> 4) as usize;
                }
                if last == 4 || last == 3 {
                    qb += (p.cigar[p.cigar.len() - 1] >> 4) as usize;
                }
            }
        }
        if !p.is_rev {
            const F: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];
            for &c in &seq[qb..qe] {
                out.push(F[c.min(4) as usize]);
            }
            out.push(b'\t');
            match qual {
                Some(qv) if !qv.is_empty() => out.extend_from_slice(&qv[qb..qe]),
                _ => out.push(b'*'),
            }
        } else {
            const R: [u8; 5] = [b'T', b'G', b'C', b'A', b'N'];
            for &c in seq[qb..qe].iter().rev() {
                out.push(R[c.min(4) as usize]);
            }
            out.push(b'\t');
            match qual {
                Some(qv) if !qv.is_empty() => out.extend(qv[qb..qe].iter().rev()),
                _ => out.push(b'*'),
            }
        }
    }

    // Optional tags: NM/MD, MC, AS, XS, SA.
    if !p.cigar.is_empty() {
        out.extend_from_slice(b"\tNM:i:");
        out.extend_from_slice(p.nm.to_string().as_bytes());
        out.extend_from_slice(b"\tMD:Z:");
        out.extend_from_slice(p.md.as_bytes());
    }
    if let Some(mm) = m.as_ref() {
        if !mm.cigar.is_empty() {
            out.extend_from_slice(b"\tMC:Z:");
            add_cigar(&mm.cigar, which, out);
        }
    }
    if p.score >= 0 {
        out.extend_from_slice(b"\tAS:i:");
        out.extend_from_slice(p.score.to_string().as_bytes());
    }
    if p.sub >= 0 {
        out.extend_from_slice(b"\tXS:i:");
        out.extend_from_slice(p.sub.to_string().as_bytes());
    }
    // SA:Z (chimeric): other primary hits in `list`.
    if p.flag & 0x100 == 0 {
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
                out.extend_from_slice((r.pos + 1).to_string().as_bytes());
                out.push(b',');
                out.push(if r.is_rev { b'-' } else { b'+' });
                out.push(b',');
                add_cigar(&r.cigar, 0, out);
                out.push(b',');
                out.extend_from_slice(r.mapq.to_string().as_bytes());
                out.push(b',');
                out.extend_from_slice(r.nm.to_string().as_bytes());
                out.push(b';');
            }
        }
    }
    // XA:Z (alternate hits), after SA/pa per mem_aln2sam. `pa` needs ALT contigs, so never emitted.
    if let Some(xa) = &p.xa {
        out.extend_from_slice(b"\tXA:Z:");
        out.extend_from_slice(xa.as_bytes());
    }
    out.push(b'\n');
}

/// Emit SAM for one read's regions (the `no_pairing` path). Port of `mem_reg2sam` (non-ALT).
#[allow(clippy::too_many_arguments)]
fn mem_reg2sam(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    name: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    a: &[MemAlnReg],
    extra_flag: u32,
    m: Option<&MemAln>,
    out: &mut Vec<u8>,
) {
    // Shadowed hits are not emitted here; they surface as the primary's XA:Z, which this path used
    // to omit entirely (~1% of PE records carried no XA where bwa emits one).
    let xa = mem_gen_alt(fm, bns, opt, a, seq.len() as i32, seq);
    let mut aa: Vec<MemAln> = Vec::new();
    let mut l = 0;
    for (k, p) in a.iter().enumerate() {
        if p.score < opt.t {
            continue;
        }
        if p.secondary >= 0 {
            continue; // !MEM_F_ALL: drop all secondaries
        }
        if p.secondary >= 0
            && p.score < (a[p.secondary as usize].score as f32 * opt.drop_ratio) as i32
        {
            continue;
        }
        let mut q = reg2aln(fm, bns, opt, seq.len() as i32, seq, p);
        q.xa = xa[k].clone();
        q.flag |= extra_flag;
        if p.secondary >= 0 {
            q.sub = -1;
        }
        if l > 0 && p.secondary < 0 {
            q.flag |= 0x800; // supplementary
        }
        if l > 0 && !p.is_alt && q.mapq > aa[0].mapq {
            q.mapq = aa[0].mapq;
        }
        aa.push(q);
        l += 1;
    }
    if aa.is_empty() {
        let mut t = MemAln::unmapped();
        t.flag |= extra_flag;
        mem_aln2sam(bns, name, seq, qual, &[t], 0, m, out);
    } else {
        for k in 0..aa.len() {
            mem_aln2sam(bns, name, seq, qual, &aa, k, m, out);
        }
    }
}

/// Full paired-end SAM for one read pair. Port of `mem_sam_pe` (no mate rescue / ALT / XA yet).
/// `a0`/`a1` are dedup'd region vectors; they are re-marked (`mem_mark_primary_se`) here. Returns
/// the two reads' SAM records (read-1 lines, then read-2 lines).
#[allow(clippy::too_many_arguments)]
pub fn mem_sam_pe<W: Write>(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    pes: &[PeStat; 4],
    id: u64,
    names: &[String; 2],
    seqs: &[&[u8]; 2],
    quals: &[Option<&[u8]>; 2],
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
    if !rescue_done && !a0.is_empty() && !a1.is_empty() {
        let pen = opt.pen_unpaired;
        let cap = opt.max_matesw.max(0) as usize;
        let (best0, best1) = (a0[0].score, a1[0].score);
        let b0: Vec<MemAlnReg> = a0
            .iter()
            .filter(|r| r.score >= best0 - pen)
            .cloned()
            .collect();
        let b1: Vec<MemAlnReg> = a1
            .iter()
            .filter(|r| r.score >= best1 - pen)
            .cloned()
            .collect();
        for anchor in b0.iter().take(cap) {
            mem_matesw(fm, bns, opt, pes, anchor, seqs[1], a1);
        }
        for anchor in b1.iter().take(cap) {
            mem_matesw(fm, bns, opt, pes, anchor, seqs[0], a0);
        }
    }

    let n_pri0 = mem_mark_primary_se(opt, a0, id << 1) as usize;
    let n_pri1 = mem_mark_primary_se(opt, a1, (id << 1) | 1) as usize;
    let extra_flag: u32 = 1;

    // Try proper pairing.
    if n_pri0 > 0 && n_pri1 > 0 {
        let pr = {
            let a: [&[MemAlnReg]; 2] = [&a0[..n_pri0], &a1[..n_pri1]];
            mem_pair(bns, opt, pes, &a, id)
        };
        if let Some(pr) = pr {
            // Multiple sufficiently-good primary hits on either end -> fall back.
            let is_multi = |a: &[MemAlnReg], n_pri: usize| -> bool {
                (1..n_pri).any(|j| a[j].secondary < 0 && a[j].score >= opt.t)
            };
            if !is_multi(a0, n_pri0) && !is_multi(a1, n_pri1) {
                let score_un = a0[0].score + a1[0].score - opt.pen_unpaired;
                let subo = pr.sub.max(score_un);
                let mut q_pe = raw_mapq(pr.score - subo, opt.a);
                if pr.n_sub > 0 {
                    q_pe -= (4.343 * f64::from(pr.n_sub + 1).ln() + 0.499) as i32;
                }
                q_pe = q_pe.clamp(0, 60);
                q_pe = (f64::from(q_pe) * (1.0 - 0.5 * f64::from(a0[0].frac_rep + a1[0].frac_rep))
                    + 0.499) as i32;

                let mut extra_flag = extra_flag;
                let mut z = pr.z;
                let mut q_se = [0i32; 2];
                if pr.score > score_un {
                    let cscore = [a0[z[0]].score, a1[z[1]].score];
                    let ccsub = [a0[z[0]].csub, a1[z[1]].csub];
                    for i in 0..2 {
                        let a = if i == 0 { &mut *a0 } else { &mut *a1 };
                        let zi = z[i];
                        if a[zi].secondary >= 0 {
                            a[zi].sub = a[a[zi].secondary as usize].score;
                            a[zi].secondary = -2;
                        }
                        q_se[i] = mem_approx_mapq_se(opt, &a[zi]) as i32;
                    }
                    for i in 0..2 {
                        q_se[i] = if q_se[i] > q_pe {
                            q_se[i]
                        } else if q_pe < q_se[i] + 40 {
                            q_pe
                        } else {
                            q_se[i] + 40
                        };
                        let cap = raw_mapq(cscore[i] - ccsub[i], opt.a);
                        q_se[i] = q_se[i].min(cap);
                    }
                    extra_flag |= 2;
                } else {
                    z = [0, 0];
                    q_se[0] = mem_approx_mapq_se(opt, &a0[0]) as i32;
                    q_se[1] = mem_approx_mapq_se(opt, &a1[0]) as i32;
                }

                // Primary/secondary swap on `secondary_all` so `mem_gen_alt` attributes the
                // paired region z[i]'s siblings to it as XA hits (mem_sam_pe lines 474-483).
                let n_pri = [n_pri0, n_pri1];
                for i in 0..2 {
                    let a = if i == 0 { &mut *a0 } else { &mut *a1 };
                    let zi = z[i] as i32;
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
                let xa0 = mem_gen_alt(fm, bns, opt, a0, seqs[0].len() as i32, seqs[0]);
                let xa1 = mem_gen_alt(fm, bns, opt, a1, seqs[1].len() as i32, seqs[1]);

                let h0 = {
                    let mut h = reg2aln(fm, bns, opt, seqs[0].len() as i32, seqs[0], &a0[z[0]]);
                    h.mapq = q_se[0].max(0) as u32;
                    h.flag |= 0x40 | extra_flag;
                    h.xa = xa0[z[0]].clone();
                    h
                };
                let h1 = {
                    let mut h = reg2aln(fm, bns, opt, seqs[1].len() as i32, seqs[1], &a1[z[1]]);
                    h.mapq = q_se[1].max(0) as u32;
                    h.flag |= 0x80 | extra_flag;
                    h.xa = xa1[z[1]].clone();
                    h
                };
                let mut buf0 = Vec::new();
                mem_aln2sam(
                    bns,
                    &names[0],
                    seqs[0],
                    quals[0],
                    std::slice::from_ref(&h0),
                    0,
                    Some(&h1),
                    &mut buf0,
                );
                let mut buf1 = Vec::new();
                mem_aln2sam(
                    bns,
                    &names[1],
                    seqs[1],
                    quals[1],
                    std::slice::from_ref(&h1),
                    0,
                    Some(&h0),
                    &mut buf1,
                );
                w.write_all(&buf0)?;
                w.write_all(&buf1)?;
                return Ok(());
            }
        }
    }

    // no_pairing fallback.
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
    let w0 = pick(a0, n_pri0);
    let w1 = pick(a1, n_pri1);
    let h0 = if w0 >= 0 {
        reg2aln(
            fm,
            bns,
            opt,
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
            opt,
            seqs[1].len() as i32,
            seqs[1],
            &a1[w1 as usize],
        )
    } else {
        MemAln::unmapped()
    };
    let mut extra_flag = extra_flag;
    if h0.rid == h1.rid && h0.rid >= 0 {
        let (d, dist) = mem_infer_dir(bns.l_pac, a0[0].rb, a1[0].rb);
        if !pes[d].failed && dist >= i64::from(pes[d].low) && dist <= i64::from(pes[d].high) {
            extra_flag |= 2;
        }
    }
    let mut buf0 = Vec::new();
    mem_reg2sam(
        fm,
        bns,
        opt,
        &names[0],
        seqs[0],
        quals[0],
        a0,
        0x41 | extra_flag,
        Some(&h1),
        &mut buf0,
    );
    let mut buf1 = Vec::new();
    mem_reg2sam(
        fm,
        bns,
        opt,
        &names[1],
        seqs[1],
        quals[1],
        a1,
        0x81 | extra_flag,
        Some(&h0),
        &mut buf1,
    );
    w.write_all(&buf0)?;
    w.write_all(&buf1)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::mem_infer_dir;

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

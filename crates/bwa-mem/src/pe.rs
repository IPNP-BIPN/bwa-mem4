//! Paired-end pairing, insert-size estimation and SAM emission, mirroring bwa-mem2's
//! `mem_pestat` / `mem_pair` / `mem_sam_pe` / `mem_aln2sam` (`reference/bwa-mem2/src/bwamem_pair.cpp`
//! and `bwamem.cpp`).
//!
//! Mate rescue (`mem_matesw`, needing `ksw_align2`) is deferred; on concordant pairs it performs no
//! Smith-Waterman, so the pairing/SAM path is exercised first.

use std::io::{self, Write};

use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

use crate::cigar::{reg2aln, MemAln};
use crate::primary::{hash_64, mem_approx_mapq_se, mem_mark_primary_se};
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
fn add_cigar(cigar: &[u32], which: usize, out: &mut Vec<u8>) {
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
    out.push(b'\n');
}

/// Emit SAM for one read's regions (the `no_pairing` path). Port of `mem_reg2sam` (non-ALT, no XA).
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
    let mut aa: Vec<MemAln> = Vec::new();
    let mut l = 0;
    for p in a {
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
        q.flag |= extra_flag;
        if p.secondary >= 0 {
            q.sub = -1;
        }
        if l > 0 && p.secondary < 0 {
            q.flag |= 0x800; // supplementary
        }
        if l > 0 && q.mapq > aa[0].mapq {
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
    w: &mut W,
) -> io::Result<()> {
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

                let h0 = {
                    let mut h = reg2aln(fm, bns, opt, seqs[0].len() as i32, seqs[0], &a0[z[0]]);
                    h.mapq = q_se[0].max(0) as u32;
                    h.flag |= 0x40 | extra_flag;
                    h
                };
                let h1 = {
                    let mut h = reg2aln(fm, bns, opt, seqs[1].len() as i32, seqs[1], &a1[z[1]]);
                    h.mapq = q_se[1].max(0) as u32;
                    h.flag |= 0x80 | extra_flag;
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

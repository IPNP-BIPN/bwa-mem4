//! CIGAR / NM / MD generation from an alignment region, mirroring bwa-mem2's `bwa_gen_cigar2`
//! (`reference/bwa-mem2/src/bwa.cpp`) and the CIGAR assembly of `mem_reg2aln`
//! (`reference/bwa-mem2/src/bwamem.cpp`).

use bwa_core::MemOpt;
use bwa_extend::ksw_global2;
use bwa_index::{BntSeq, FmIndex};

use crate::primary::mem_approx_mapq_se;
use crate::MemAlnReg;

/// A finalized single-end alignment (bwa-mem2's `mem_aln_t`, phase-6/7 subset).
#[derive(Debug, Clone)]
pub struct MemAln {
    pub rid: i32,
    /// 0-based position within the contig (SAM POS is `pos + 1`).
    pub pos: i64,
    pub is_rev: bool,
    pub mapq: u32,
    /// SAM FLAG bits set so far (0x4 unmapped, 0x100 secondary). Strand/pair bits are added by the
    /// SAM writer.
    pub flag: u32,
    /// CIGAR, `len<<4 | op` (op 0=M/1=I/2=D/3=S).
    pub cigar: Vec<u32>,
    pub nm: i32,
    pub md: String,
    pub score: i32,
    /// Suboptimal score (`XS:i`), `max(sub, csub)`. `-1` suppresses the tag (secondary hits).
    pub sub: i32,
    /// Alternate hits (`XA:Z:`), pre-formatted `rname,±pos,cigar,NM;`... or `None`. Set by the
    /// caller via `mem_gen_alt`; `reg2aln` leaves it `None`.
    pub xa: Option<String>,
}

impl MemAln {
    /// An unmapped alignment (`mem_reg2aln` with a null region): `rid=-1`, FLAG 0x4, no CIGAR.
    pub fn unmapped() -> Self {
        MemAln {
            rid: -1,
            pos: -1,
            is_rev: false,
            mapq: 0,
            flag: 0x4,
            cigar: Vec::new(),
            nm: 0,
            md: String::new(),
            // Zero, not -1: bwa builds this from `memset(&a, 0, sizeof(mem_aln_t))`, so an unmapped
            // record clears `mem_aln2sam`'s `score >= 0` / `sub >= 0` guards and carries
            // `AS:i:0 XS:i:0`. Signalling "absent" with -1 silently drops both tags.
            score: 0,
            sub: 0,
            xa: None,
        }
    }
}

/// Env-gated (`BWA3_DUMP_BW`) trace of the band-width retry loop, in bwa's `-v 4` format so the two
/// can be diffed directly. Cached: `reg2aln` runs per emitted alignment, so a `var_os` per call
/// would show up in the profile.
fn dump_bw() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA3_DUMP_BW").is_some())
}

/// Inferred band width, port of `infer_bw`.
fn infer_bw(l1: i32, l2: i32, score: i32, a: i32, q: i32, r: i32) -> i32 {
    if l1 == l2 && l1 * a - score < (q + r - a) << 1 {
        return 0;
    }
    let mut w = (f64::from(l1.min(l2) * a - score - q) / f64::from(r) + 2.0) as i32;
    if w < (l1 - l2).abs() {
        w = (l1 - l2).abs();
    }
    w
}

/// Global-align `query_codes` (the seed-region query slice) to reference `[rb, re)` and compute the
/// CIGAR, NM and MD. Port of `bwa_gen_cigar2`.
pub(crate) fn gen_cigar2(
    fm: &FmIndex,
    opt: &MemOpt,
    w_: i32,
    l_query: i32,
    query_codes: &[u8],
    rb: i64,
    re: i64,
) -> Option<(i32, Vec<u32>, i32, String)> {
    let l_pac = fm.l_pac();
    if l_query <= 0 || rb >= re || (rb < l_pac && re > l_pac) {
        return None;
    }
    let rlen = (re - rb) as i32;
    let mut query: Vec<u8> = query_codes.to_vec();
    let mut rseq: Vec<u8> = (rb..re).map(|p| fm.base(p)).collect();
    if rb >= l_pac {
        query.reverse();
        rseq.reverse();
    }
    let mat0 = i32::from(opt.mat[0]);

    let (score, cigar) = if l_query == rlen && w_ == 0 {
        let mut sc = 0;
        for i in 0..l_query as usize {
            sc += i32::from(opt.mat[rseq[i] as usize * 5 + query[i] as usize]);
        }
        (sc, vec![(l_query as u32) << 4])
    } else {
        let max_ins = (f64::from(((l_query + 1) >> 1) * mat0 - opt.o_ins) / f64::from(opt.e_ins)
            + 1.0) as i32;
        let max_del = (f64::from(((l_query + 1) >> 1) * mat0 - opt.o_del) / f64::from(opt.e_del)
            + 1.0) as i32;
        let max_gap = max_ins.max(max_del).max(1);
        let mut w = (max_gap + (rlen - l_query).abs() + 1) >> 1;
        w = w.min(w_);
        let min_w = (rlen - l_query).abs() + 3;
        w = w.max(min_w);
        ksw_global2(
            &query, &rseq, 5, &opt.mat, opt.o_del, opt.e_del, opt.o_ins, opt.e_ins, w,
        )
    };

    // NM + MD.
    let fwd = rb < l_pac;
    let base_char = |c: u8| -> char {
        let t = if fwd {
            [b'A', b'C', b'G', b'T', b'N']
        } else {
            [b'T', b'G', b'C', b'A', b'N']
        };
        t[c.min(4) as usize] as char
    };
    let mut md = String::new();
    let (mut x, mut y, mut u) = (0usize, 0usize, 0i32);
    let mut n_mm = 0i32;
    let mut n_gap = 0i32;
    let n_cigar = cigar.len();
    for (k, &c) in cigar.iter().enumerate() {
        let op = c & 0xf;
        let len = (c >> 4) as usize;
        if op == 0 {
            for i in 0..len {
                if query[x + i] != rseq[y + i] {
                    md.push_str(&u.to_string());
                    md.push(base_char(rseq[y + i]));
                    n_mm += 1;
                    u = 0;
                } else {
                    u += 1;
                }
            }
            x += len;
            y += len;
        } else if op == 2 {
            if k > 0 && k < n_cigar - 1 {
                md.push_str(&u.to_string());
                md.push('^');
                for i in 0..len {
                    md.push(base_char(rseq[y + i]));
                }
                u = 0;
                n_gap += len as i32;
            }
            y += len;
        } else if op == 1 {
            x += len;
            n_gap += len as i32;
        }
    }
    md.push_str(&u.to_string());
    Some((score, cigar, n_mm + n_gap, md))
}

/// Turn an alignment region into a finalized alignment (CIGAR, NM, MD, position). Port of the CIGAR
/// assembly in `mem_reg2aln` (band retry, leading/trailing-D squeeze, soft-clip addition).
pub fn reg2aln(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    l_query: i32,
    query_codes: &[u8],
    reg: &MemAlnReg,
) -> MemAln {
    let (qb, qe, rb, re) = (reg.qb, reg.qe, reg.rb, reg.re);
    let tmp = infer_bw(
        qe - qb,
        (re - rb) as i32,
        reg.truesc,
        opt.a,
        opt.o_del,
        opt.e_del,
    );
    let mut w2 = infer_bw(
        qe - qb,
        (re - rb) as i32,
        reg.truesc,
        opt.a,
        opt.o_ins,
        opt.e_ins,
    )
    .max(tmp);
    if w2 > opt.w {
        w2 = w2.min(reg.w);
    }

    if dump_bw() {
        eprintln!("* Band width: inferred={w2}, cmd_opt={}, alnreg={}", opt.w, reg.w);
    }
    let mut i = 0;
    let mut last_sc = -(1 << 30);
    let (_score, mut cigar, nm, md) = loop {
        w2 = w2.min(opt.w << 2);
        let (sc, cg, nm_, md_) = gen_cigar2(
            fm,
            opt,
            w2,
            qe - qb,
            &query_codes[qb as usize..qe as usize],
            rb,
            re,
        )
        .expect("gen_cigar2");
        if dump_bw() {
            eprintln!("* Final alignment: w2={w2}, global_sc={sc}, local_sc={}", reg.truesc);
        }
        if sc == last_sc || w2 == opt.w << 2 {
            break (sc, cg, nm_, md_);
        }
        last_sc = sc;
        w2 <<= 1;
        i += 1;
        if !(i < 3 && sc < reg.truesc - opt.a) {
            break (sc, cg, nm_, md_);
        }
    };

    let probe = if rb < bns.l_pac { rb } else { re - 1 };
    let (mut pos, is_rev) = bns.depos(probe);

    // Squeeze a leading or trailing deletion.
    if !cigar.is_empty() {
        if cigar[0] & 0xf == 2 {
            pos += i64::from(cigar[0] >> 4);
            cigar.remove(0);
        } else if cigar[cigar.len() - 1] & 0xf == 2 {
            cigar.pop();
        }
    }

    // Soft-clips for the unaligned read ends.
    if qb != 0 || qe != l_query {
        let clip5 = if is_rev { l_query - qe } else { qb };
        let clip3 = if is_rev { qb } else { l_query - qe };
        if clip5 > 0 {
            cigar.insert(0, ((clip5 as u32) << 4) | 3);
        }
        if clip3 > 0 {
            cigar.push(((clip3 as u32) << 4) | 3);
        }
    }

    let rid = bns.pos2rid(pos);
    let offset = if rid >= 0 {
        bns.contigs[rid as usize].offset
    } else {
        0
    };
    MemAln {
        rid,
        pos: pos - offset,
        is_rev,
        mapq: if reg.secondary < 0 {
            mem_approx_mapq_se(opt, reg)
        } else {
            0
        },
        flag: if reg.secondary >= 0 { 0x100 } else { 0 },
        cigar,
        nm,
        md,
        score: reg.score,
        sub: reg.sub.max(reg.csub),
        xa: None,
    }
}

/// Format a CIGAR (`len<<4|op`) as a SAM string, or `*` when empty.
pub fn cigar_string(cigar: &[u32]) -> String {
    if cigar.is_empty() {
        return "*".to_string();
    }
    const OPS: [char; 5] = ['M', 'I', 'D', 'S', 'H'];
    let mut s = String::new();
    for &c in cigar {
        s.push_str(&(c >> 4).to_string());
        s.push(OPS[(c & 0xf) as usize]);
    }
    s
}

/// `add_cigar`: like [`cigar_string`], but rewrites clip ops for the record being emitted. A
/// supplementary record (`which != 0`) hard-clips what the primary soft-clips, so that the read's
/// bases are stored exactly once; `-Y` (`flags::SOFTCLIP`) and ALT hits keep soft clips.
pub fn cigar_string_which(cigar: &[u32], which: usize, is_alt: bool, softclip: bool) -> String {
    if cigar.is_empty() {
        return "*".to_string();
    }
    const OPS: [char; 5] = ['M', 'I', 'D', 'S', 'H'];
    let mut s = String::new();
    for &c in cigar {
        let mut op = (c & 0xf) as usize;
        if !softclip && !is_alt && (op == 3 || op == 4) {
            op = if which != 0 { 4 } else { 3 };
        }
        s.push_str(&(c >> 4).to_string());
        s.push(OPS[op]);
    }
    s
}

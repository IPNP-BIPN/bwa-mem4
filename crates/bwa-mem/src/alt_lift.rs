//! `mem --alt-project keep-alt|primary` and `--alt-project-exclude-bed` (fg-labs/bwa-mem3 0.18,
//! #775 and #802): lift alignment regions on ALT contigs onto the primary assembly through the
//! `.alt` sidecar's alignments, before primary marking, pestat, rescue, pairing and MAPQ, so an
//! ALT haplotype can decide where a read is placed. Port of the fork's `src/alt_lift.cpp`; the
//! comments there carry the reasoning, the ones here say where the port has to be careful.
//!
//! The developer-only `--alt-project-min-gap`, `BWAMEM3_ALT_PROJECT_PAIR_GAP` and
//! `BWAMEM3_ALT_PROJECT_TRACE` are not ported (default 0 / unset in the fork).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use bwa_core::alt_lift::{AltLift, AltLift1};
use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

use crate::cigar::{gen_cigar_band, reg2aln};
use crate::MemAlnReg;

/// The widest band floor a lifted region carries (the fork's `MEM_ALT_BW_MAX`, 10 bits less one):
/// a hit whose ALT indels need more stays on the ALT whatever `-w` is.
pub const ALT_BW_MAX: i64 = (1 << 10) - 2;

// BAM CIGAR ops.
const OP_INS: u32 = 1;
const OP_SOFT: u32 = 4;
const OP_HARD: u32 = 5;
const OP_PAD: u32 = 6;
const OP_BACK: u32 = 9;

fn op_is_aligned(op: u32) -> bool {
    matches!(op, 0 | 7 | 8)
}
fn op_is_clip(op: u32) -> bool {
    op == OP_SOFT || op == OP_HARD
}
fn op_eats_query(op: u32) -> bool {
    matches!(op, 0 | 1 | 4 | 5 | 7 | 8)
}
fn op_eats_ref(op: u32) -> bool {
    matches!(op, 0 | 2 | 3 | 7 | 8)
}

/// `parse_cigar`: digits before every op, lengths in `[1, 2^28)`, P dropped. `(ops, qlen)`.
fn parse_cigar(s: &[u8]) -> Option<(Vec<u32>, i64)> {
    let mut ops = Vec::new();
    let mut qlen = 0i64;
    let mut i = 0;
    while i < s.len() {
        let d0 = i;
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
        }
        if i == d0 || i == s.len() || i - d0 > 12 {
            return None;
        }
        let len: i64 = std::str::from_utf8(&s[d0..i]).ok()?.parse().ok()?;
        if len <= 0 || len >= 1 << 28 {
            return None;
        }
        let op = b"MIDNSHP=XB".iter().position(|&c| c == s[i])? as u32;
        i += 1;
        if op == OP_BACK {
            return None;
        }
        if op == OP_PAD {
            continue;
        }
        ops.push((len as u32) << 4 | op);
        if op_eats_query(op) {
            qlen += len;
        }
    }
    (!ops.is_empty() && qlen <= i64::from(i32::MAX)).then_some((ops, qlen))
}

/// `parse_uint_field`: a whole non-negative decimal (FLAG or POS).
fn parse_uint(s: &[u8]) -> Option<i64> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

/// What `--alt-project-exclude-bed` read and did (the fork's `alt_lift_exclude_stats_t`).
#[derive(Debug, Default, Clone, Copy)]
pub struct ExcludeStats {
    /// Nonempty validated rows before the merge.
    pub input_intervals: i64,
    /// Intervals after merging overlapping and adjacent rows.
    pub merged_intervals: i64,
    /// Rows with start == end (ignored).
    pub zero_width_intervals: i64,
    /// Otherwise-usable `.alt` lines whose primary footprint overlaps an interval.
    pub excluded: i64,
}

/// Which input a load failed on, so the caller names the right option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailedInput {
    Bed,
    Sidecar,
}

/// Merged BED intervals per primary contig id.
type ExcludeBed = HashMap<usize, Vec<(i64, i64)>>;

/// Gzip or plain, by the magic bytes (`gzopen` reads both).
fn read_maybe_gz(path: &Path) -> std::io::Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&raw[..]).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

/// `load_exclude_bed`: validated, sorted and merged, or the fork's diagnostic.
fn load_exclude_bed(
    path: &Path,
    bns: &BntSeq,
    stats: &mut ExcludeStats,
) -> Result<ExcludeBed, String> {
    let p = path.display();
    let text = read_maybe_gz(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound
            || e.kind() == std::io::ErrorKind::PermissionDenied
        {
            format!("{p}: cannot open BED: {e}")
        } else {
            format!("{p}: BED decompression/read failure: {e}")
        }
    })?;
    let mut names: HashMap<&[u8], usize> = HashMap::new();
    for (i, c) in bns.contigs.iter().enumerate() {
        names.entry(c.name.as_bytes()).or_insert(i);
    }
    let directive = |l: &[u8], w: &[u8]| {
        l.starts_with(w) && (l.len() == w.len() || l[w.len()] == b' ' || l[w.len()] == b'\t')
    };
    let coord = |t: &[u8]| -> Option<i64> {
        if t.is_empty() {
            return None;
        }
        t.iter().try_fold(0i64, |n, &c| {
            c.is_ascii_digit()
                .then(|| n.checked_mul(10)?.checked_add(i64::from(c - b'0')))
                .flatten()
        })
    };
    let mut bed = ExcludeBed::new();
    let mut lines: Vec<&[u8]> = text.split(|&c| c == b'\n').collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    for (n, line) in lines.into_iter().enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.iter().all(|&c| matches!(c, b' ' | b'\t' | b'\r'))
            || line[0] == b'#'
            || directive(line, b"track")
            || directive(line, b"browser")
        {
            continue;
        }
        let f: Vec<&[u8]> = line.splitn(4, |&c| c == b'\t').collect();
        let reason = if line.contains(&0) {
            Some("embedded NUL")
        } else if f.len() < 3 {
            Some("expected tab-delimited CHROM, start, end")
        } else {
            let end_field = f[2];
            match (coord(f[1]), coord(end_field)) {
                (Some(s), Some(e)) if s <= e => match names.get(f[0]) {
                    None => Some("unknown reference contig"),
                    Some(&r) if bns.contigs[r].is_alt => Some("BED contig is an ALT contig"),
                    Some(&r) if e > i64::from(bns.contigs[r].len) => {
                        Some("interval exceeds reference contig length")
                    }
                    Some(&r) => {
                        if s == e {
                            stats.zero_width_intervals += 1;
                        } else {
                            bed.entry(r).or_default().push((s, e));
                            stats.input_intervals += 1;
                        }
                        None
                    }
                },
                _ => Some("expected decimal coordinates with 0 <= start <= end"),
            }
        };
        if let Some(r) = reason {
            return Err(format!("{p} line {}: {r}", n + 1));
        }
    }
    for iv in bed.values_mut() {
        iv.sort_unstable();
        let mut kept: Vec<(i64, i64)> = Vec::with_capacity(iv.len());
        for &(s, e) in iv.iter() {
            match kept.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => kept.push((s, e)),
            }
        }
        stats.merged_intervals += kept.len() as i64;
        *iv = kept;
    }
    Ok(bed)
}

/// Whether `[start, end)` on `rid` overlaps a merged interval by at least 1 bp.
fn excluded_footprint(bed: &ExcludeBed, rid: usize, start: i64, end: i64) -> bool {
    if end <= start {
        return false;
    }
    bed.get(&rid).is_some_and(|iv| {
        let k = iv.partition_point(|&(_, e)| e <= start);
        k < iv.len() && iv[k].0 < end
    })
}

/// `alt_lift_load` / `alt_lift_load_excluded`: parse `path` against `bns`, skipping lines whose
/// primary footprint overlaps `bed` (when given). Warnings on the skipped lines go to stderr. On
/// failure, which input failed and the fork's diagnostic.
pub fn load(
    path: &Path,
    bns: &BntSeq,
    bed_path: Option<&Path>,
) -> Result<(AltLift, ExcludeStats), (FailedInput, String)> {
    let mut stats = ExcludeStats::default();
    let bed = match bed_path {
        Some(b) => Some(load_exclude_bed(b, bns, &mut stats).map_err(|e| (FailedInput::Bed, e))?),
        None => None,
    };
    let text = std::fs::read(path).map_err(|e| {
        let msg = if e.kind() == std::io::ErrorKind::NotFound
            || e.kind() == std::io::ErrorKind::PermissionDenied
        {
            format!("{}: cannot open sidecar: {e}", path.display())
        } else {
            format!("{}: sidecar read/close failure", path.display())
        };
        (FailedInput::Sidecar, msg)
    })?;
    let mut names: HashMap<&[u8], usize> = HashMap::new();
    for (i, c) in bns.contigs.iter().enumerate() {
        // `kh_put` on a name already present overwrites its value: the last contig wins.
        names.insert(c.name.as_bytes(), i);
    }
    let mut l = AltLift {
        n_lift: 0,
        a: vec![
            AltLift1 {
                trid: -1,
                ..AltLift1::default()
            };
            bns.contigs.len()
        ],
    };
    // no_ctg, no_target, alt_target, unmapped, not_primary, dup, bad
    let mut sk = [0i64; 7];
    let shown = path.display();
    let bad = |sk: &mut [i64; 7], lineno: usize, what: String| {
        if sk[6] < 10 {
            eprintln!("[W::alt_lift_load] {shown} line {lineno}: skipping {what}");
        }
        sk[6] += 1;
    };
    for (n, line) in text.split(|&c| c == b'\n').enumerate() {
        let mut line = line;
        while let Some(t) = line.strip_suffix(b"\r") {
            line = t;
        }
        if line.is_empty() || line[0] == b'@' {
            continue;
        }
        let lineno = n + 1;
        let f: Vec<&[u8]> = line.splitn(7, |&c| c == b'\t').take(6).collect();
        if f.len() < 6 {
            bad(&mut sk, lineno, "a line with fewer than 6 fields".into());
            continue;
        }
        let name = String::from_utf8_lossy(f[0]);
        let Some(&qrid) = names.get(f[0]) else {
            sk[0] += 1;
            continue;
        };
        let (Some(flag), Some(pos)) = (parse_uint(f[1]), parse_uint(f[3])) else {
            bad(
                &mut sk,
                lineno,
                format!("{name} (FLAG or POS is not a non-negative integer)"),
            );
            continue;
        };
        if flag & 0x4 != 0 || pos < 1 || f[2] == b"*" || f[5] == b"*" {
            sk[3] += 1;
            continue;
        }
        if flag & 0x900 != 0 {
            sk[4] += 1;
            continue;
        }
        let Some(&trid) = names.get(f[2]) else {
            sk[1] += 1;
            continue;
        };
        let tlen = i64::from(bns.contigs[trid].len);
        if bns.contigs[trid].is_alt {
            sk[2] += 1;
            continue;
        }
        if l.a[qrid].trid >= 0 {
            sk[5] += 1;
            continue;
        }
        let tname = &bns.contigs[trid].name;
        if pos > tlen {
            bad(
                &mut sk,
                lineno,
                format!("{name} (POS {pos} is past the end of {tname}, length {tlen})"),
            );
            continue;
        }
        let Some((cigar, qlen)) = parse_cigar(f[5]) else {
            bad(&mut sk, lineno, format!("{name} (bad CIGAR)"));
            continue;
        };
        let rspan: i64 = cigar
            .iter()
            .filter(|&&c| op_eats_ref(c & 0xf))
            .map(|&c| i64::from(c >> 4))
            .sum();
        let clen = i64::from(bns.contigs[qrid].len);
        if qlen != clen {
            bad(
                &mut sk,
                lineno,
                format!("{name} (its CIGAR has {qlen} query bases, the contig has {clen})"),
            );
            continue;
        }
        if rspan > tlen - (pos - 1) {
            bad(
                &mut sk,
                lineno,
                format!(
                    "{name} (its alignment ends at {}, past the end of {tname}, length {tlen})",
                    pos - 1 + rspan
                ),
            );
            continue;
        }
        if bed
            .as_ref()
            .is_some_and(|b| excluded_footprint(b, trid, pos - 1, pos - 1 + rspan))
        {
            stats.excluded += 1;
            continue;
        }
        let mut qoff = vec![0i64; cigar.len() + 1];
        let mut roff = vec![pos - 1; cigar.len() + 1];
        for (i, &c) in cigar.iter().enumerate() {
            let len = i64::from(c >> 4);
            qoff[i + 1] = qoff[i] + if op_eats_query(c & 0xf) { len } else { 0 };
            roff[i + 1] = roff[i] + if op_eats_ref(c & 0xf) { len } else { 0 };
        }
        l.a[qrid] = AltLift1 {
            trid: trid as i32,
            rev: flag & 0x10 != 0,
            tpos: pos - 1,
            tlen,
            qlen,
            cigar,
            qoff,
            roff,
        };
        l.n_lift += 1;
    }
    if sk[6] > 10 {
        eprintln!(
            "[W::alt_lift_load] {shown}: {} more malformed line(s) not named above",
            sk[6] - 10
        );
    }
    let n_skip: i64 = sk.iter().sum();
    if n_skip > 0 {
        eprintln!(
            "[W::alt_lift_load] {shown}: skipped {n_skip} line(s): {} ALT contig not in the index, \
             {} target not in the index, {} target is an ALT contig, {} unmapped, \
             {} secondary/supplementary, {} duplicate, {} malformed",
            sk[0], sk[1], sk[2], sk[3], sk[4], sk[5], sk[6]
        );
    }
    Ok((l, stats))
}

/// One lifted interval (the fork's `alt_lift_iv_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiftIv {
    pub trid: i32,
    pub rev: bool,
    /// `[ps, pe)`: 0-based forward interval on `trid`.
    pub ps: i64,
    pub pe: i64,
    /// Query bases of the interval in M/=/X ops.
    pub n_aln: i64,
    /// Query bases of the interval in S/H ops.
    pub n_clip: i64,
    /// Widest |inserted - deleted| between the interval's start and any point inside it.
    pub max_diag: i64,
    /// The interval's first or last base lies in an I op.
    pub edge_ins: bool,
}

/// `find_op`: the first op `i` with `qoff[i + 1] > x` (strict) or `>= x`.
fn find_op(a: &AltLift1, x: i64, strict: bool) -> usize {
    a.qoff[1..].partition_point(|&q| if strict { q <= x } else { q < x })
}

/// `alt_lift_interval`: lift `[s, e)` (forward coordinates on ALT contig `rid`), or `None`.
pub fn lift_interval(l: &AltLift, rid: usize, s: i64, e: i64) -> Option<LiftIv> {
    let a = l.a.get(rid)?;
    if a.trid < 0 || a.tlen < 1 || e <= s || s < 0 || e > a.qlen {
        return None;
    }
    let (qs, qe) = if a.rev {
        (a.qlen - e, a.qlen - s)
    } else {
        (s, e)
    };
    let (i0, i1) = (find_op(a, qs, true), find_op(a, qe, false));
    assert!(
        i0 <= i1 && i1 < a.cigar.len(),
        "lift_interval: outside the .alt CIGAR"
    );
    let ref_at = |i: usize, x: i64| {
        a.roff[i]
            + if op_is_aligned(a.cigar[i] & 0xf) {
                x - a.qoff[i]
            } else {
                0
            }
    };
    let mut b = ref_at(i0, qs);
    let mut en = ref_at(i1, qe);
    if b >= a.tlen {
        b = a.tlen - 1;
    }
    if en <= b {
        en = b + 1;
    }
    let (mut d, mut max_diag, mut n_aln, mut n_clip) = (0i64, 0i64, 0i64, 0i64);
    for k in i0..=i1 {
        let op = a.cigar[k] & 0xf;
        if op_eats_query(op) {
            let lo = qs.max(a.qoff[k]);
            let hi = qe.min(a.qoff[k + 1]);
            if op_is_aligned(op) {
                n_aln += hi - lo;
                continue;
            }
            if op_is_clip(op) {
                n_clip += hi - lo;
            }
            d += hi - lo;
        } else if op_eats_ref(op) {
            d -= i64::from(a.cigar[k] >> 4);
        }
        max_diag = max_diag.max(d.abs());
    }
    Some(LiftIv {
        trid: a.trid,
        rev: a.rev,
        ps: b,
        pe: en,
        n_aln,
        n_clip,
        max_diag,
        edge_ins: a.cigar[i0] & 0xf == OP_INS || a.cigar[i1] & 0xf == OP_INS,
    })
}

/// `mem_alnreg_score_lt`: dedup's by-score order (score descending, rb, qb).
fn score_lt(a: &MemAlnReg, b: &MemAlnReg) -> bool {
    a.score > b.score || (a.score == b.score && (a.rb < b.rb || (a.rb == b.rb && a.qb < b.qb)))
}

/// `mem_overlap_redundant`, in the C's float arithmetic.
fn redundant(opt: &MemOpt, ovl: i64, min_len: i64) -> bool {
    (ovl as f32) > opt.mask_level_redun * min_len as f32
}

/// `alt_project_same_locus`.
fn same_locus(opt: &MemOpt, x: &MemAlnReg, y: &MemAlnReg) -> bool {
    if x.rid != y.rid {
        return false;
    }
    let or = x.re.min(y.re) - x.rb.max(y.rb);
    let mr = (x.re - x.rb).min(y.re - y.rb);
    if or <= 0 || !redundant(opt, or, mr) {
        return false;
    }
    let oq = x.qe.min(y.qe) - x.qb.max(y.qb);
    let mq = (x.qe - x.qb).min(y.qe - y.qb);
    oq > 0 && 2 * oq > mq
}

/// `alt_project_replaces`: may lifted `l` replace unlifted `p` at its locus?
fn replaces(opt: &MemOpt, l: &MemAlnReg, p: &MemAlnReg) -> bool {
    let n_out = i64::from((l.qb - p.qb).max(0) + (p.qe - l.qe).max(0));
    i64::from(l.score) > i64::from(p.score) + i64::from(opt.a) * n_out
}

const UNLIFTED: u8 = 0;
const CAN_REPLACE: u8 = 1;
const CANNOT_REPLACE: u8 = 2;

/// `mem_alt_project`: lift the read's ALT regions, drop the ones a lift made redundant, and leave
/// the survivors in priority order. A no-op without `--alt-project` or when nothing lifts.
pub fn project(opt: &MemOpt, bns: &BntSeq, regs: &mut Vec<MemAlnReg>) {
    let Some(l) = opt.alt_lift.as_deref() else {
        return;
    };
    let l_pac = bns.l_pac;
    // (index, rid before the lift, kind); empty until the first lift.
    let mut e: Vec<(usize, i32, u8)> = Vec::new();
    for i in 0..regs.len() {
        let p = &regs[i];
        if !p.is_alt || p.rid < 0 {
            continue;
        }
        let is_rev = p.rb >= l_pac;
        let (fb, fe) = if is_rev {
            ((l_pac << 1) - p.re, (l_pac << 1) - p.rb)
        } else {
            (p.rb, p.re)
        };
        let off = bns.contigs[p.rid as usize].offset;
        let Some(iv) = lift_interval(l, p.rid as usize, fb - off, fe - off) else {
            continue;
        };
        if iv.n_clip > 0
            || iv.edge_ins
            || 2 * iv.n_aln < fe - fb
            || iv.max_diag > i64::from(opt.w)
            || iv.max_diag > ALT_BW_MAX
        {
            continue;
        }
        // The narrowest band the re-alignment can use (a wider one never narrows it).
        if i64::from(gen_cigar_band(
            opt,
            iv.max_diag as i32,
            p.qe - p.qb,
            (iv.pe - iv.ps) as i32,
        )) < iv.max_diag
        {
            continue;
        }
        if e.is_empty() {
            e = regs
                .iter()
                .enumerate()
                .map(|(k, r)| (k, r.rid, UNLIFTED))
                .collect();
        }
        let toff = bns.contigs[iv.trid as usize].offset;
        let p = &mut regs[i];
        if is_rev ^ iv.rev {
            p.rb = (l_pac << 1) - (toff + iv.pe);
            p.re = (l_pac << 1) - (toff + iv.ps);
        } else {
            p.rb = toff + iv.ps;
            p.re = toff + iv.pe;
        }
        p.seedlen0 = p.rid;
        p.rid = iv.trid;
        p.is_alt = false;
        p.alt_bw = (iv.max_diag + 1) as u16;
        e[i].2 = CAN_REPLACE;
    }
    if e.is_empty() {
        return;
    }
    let n = regs.len();
    // Step 0: a lifted region can replace only if it replaces every unlifted one at its locus.
    for i in 0..n {
        if e[i].2 == UNLIFTED {
            continue;
        }
        if (0..n).any(|j| {
            e[j].2 == UNLIFTED
                && same_locus(opt, &regs[i], &regs[j])
                && !replaces(opt, &regs[i], &regs[j])
        }) {
            e[i].2 = CANNOT_REPLACE;
        }
    }
    e.sort_by(|x, y| {
        let (p, q) = (&regs[x.0], &regs[y.0]);
        if score_lt(p, q) {
            std::cmp::Ordering::Less
        } else if score_lt(q, p) {
            std::cmp::Ordering::Greater
        } else {
            x.1.cmp(&y.1)
        }
    });
    let mut keep = vec![false; n];
    // Step 1.
    for r in 0..n {
        if e[r].2 == CANNOT_REPLACE {
            continue;
        }
        keep[r] = !(0..r).any(|k| {
            keep[k] && e[k].2 == CAN_REPLACE && same_locus(opt, &regs[e[k].0], &regs[e[r].0])
        });
    }
    // Step 2.
    for r in 0..n {
        if e[r].2 != CANNOT_REPLACE {
            continue;
        }
        keep[r] =
            !(0..n).any(|k| k != r && keep[k] && same_locus(opt, &regs[e[k].0], &regs[e[r].0]));
    }
    let out: Vec<MemAlnReg> = (0..n)
        .filter(|&r| keep[r])
        .map(|r| regs[e[r].0].clone())
        .collect();
    *regs = out;
}

/// `mem_alt_drop_unlifted`: remove every region still on an ALT contig and return the best one
/// scoring at least `-T` (score, then rid, rb, qb), marked primary with `sub`/`sub_n` set among the
/// read's other ALT regions, as the `OA:Z` candidate.
pub fn drop_unlifted(opt: &MemOpt, regs: &mut Vec<MemAlnReg>) -> Option<MemAlnReg> {
    let mut best: Option<usize> = None;
    for (i, p) in regs.iter().enumerate() {
        if !p.is_alt || p.score < opt.t {
            continue;
        }
        best = match best {
            Some(b) => {
                let b_ = &regs[b];
                let better = p.score > b_.score
                    || (p.score == b_.score
                        && (p.rid < b_.rid
                            || (p.rid == b_.rid
                                && (p.rb < b_.rb || (p.rb == b_.rb && p.qb < b_.qb)))));
                Some(if better { i } else { b })
            }
            None => Some(i),
        };
    }
    let oa = best.map(|best| {
        let tmp = (opt.a + opt.b)
            .max(opt.o_del + opt.e_del)
            .max(opt.o_ins + opt.e_ins);
        let mut oa = regs[best].clone();
        oa.sub = 0;
        oa.sub_n = 0;
        oa.alt_sc = 0;
        oa.secondary = -1;
        oa.secondary_all = -1;
        for (j, r) in regs.iter().enumerate() {
            if j == best || !r.is_alt {
                continue;
            }
            let b_max = r.qb.max(oa.qb);
            let e_min = r.qe.min(oa.qe);
            if e_min <= b_max {
                continue;
            }
            let min_l = (r.qe - r.qb).min(oa.qe - oa.qb);
            if ((e_min - b_max) as f32) < min_l as f32 * opt.mask_level {
                continue;
            }
            oa.sub = oa.sub.max(r.score);
            if oa.score - r.score <= tmp {
                oa.sub_n += 1;
            }
        }
        oa
    });
    regs.retain(|r| !r.is_alt);
    oa
}

/// `--alt-project` on one read's deduped, stamped regions: the lift, then under `primary` the
/// drop. Returns the `OA:Z` value the read's record carries if it ends up unmapped: computed only
/// when no surviving region reaches `-T` (a PE end can still be rescued; its record is then mapped
/// and the value unused). `codes` is the read as aligned.
pub fn project_read(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    regs: &mut Vec<MemAlnReg>,
) -> Option<Vec<u8>> {
    project(opt, bns, regs);
    if !opt.alt_project_primary {
        return None;
    }
    let oa = drop_unlifted(opt, regs)?;
    if regs.iter().any(|r| r.score >= opt.t) {
        return None;
    }
    Some(oa_string(fm, bns, opt, codes, &oa))
}

/// `mem_aln_oa_string`: `RNAME,POS,strand,CIGAR,MAPQ,NM;` of `reg` on its ALT contig.
pub fn oa_string(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    reg: &MemAlnReg,
) -> Vec<u8> {
    let o = reg2aln(fm, bns, opt, codes.len() as i32, codes, reg);
    let mut s = Vec::new();
    s.extend_from_slice(bns.contigs[o.rid as usize].name.as_bytes());
    s.push(b',');
    crate::emit::push_int(&mut s, o.pos + 1);
    s.extend_from_slice(if o.is_rev { b",-," } else { b",+," });
    for &c in &o.cigar {
        crate::emit::push_int(&mut s, i64::from(c >> 4));
        s.push(b"MIDSH"[(c & 0xf) as usize]);
    }
    if o.cigar.is_empty() {
        s.push(b'*');
    }
    s.push(b',');
    crate::emit::push_int(&mut s, i64::from(o.mapq));
    s.push(b',');
    crate::emit::push_int(&mut s, i64::from(o.nm));
    s.push(b';');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(cigar: &str, rev: bool, tpos: i64) -> AltLift {
        let (cigar, qlen) = parse_cigar(cigar.as_bytes()).unwrap();
        let mut qoff = vec![0; cigar.len() + 1];
        let mut roff = vec![tpos; cigar.len() + 1];
        for (i, &c) in cigar.iter().enumerate() {
            let len = i64::from(c >> 4);
            qoff[i + 1] = qoff[i] + if op_eats_query(c & 0xf) { len } else { 0 };
            roff[i + 1] = roff[i] + if op_eats_ref(c & 0xf) { len } else { 0 };
        }
        AltLift {
            n_lift: 1,
            a: vec![AltLift1 {
                trid: 0,
                rev,
                tpos,
                tlen: 100_000,
                qlen,
                cigar,
                qoff,
                roff,
            }],
        }
    }

    #[test]
    fn cigar_parse_rules() {
        assert_eq!(parse_cigar(b"10M2P5I").unwrap().1, 15);
        for bad in [
            &b"10M+5I"[..],
            b"10M 5I",
            b"0M",
            b"10",
            b"M",
            b"10B",
            b"",
            b"10Q",
        ] {
            assert!(
                parse_cigar(bad).is_none(),
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn lift_through_deletion_insertion_and_clip() {
        // 1500M3D497M at 3000: a read across the deletion moves 3 bases on.
        let l = table("1500M3D497M", false, 3000);
        let iv = lift_interval(&l, 0, 1450, 1600).unwrap();
        assert_eq!((iv.ps, iv.pe, iv.n_aln, iv.max_diag), (4450, 4603, 150, 3));
        // Reverse-complemented: [s, e) mirrors to [qlen - e, qlen - s).
        let r = table("1500M3D497M", true, 3000);
        let iv = lift_interval(&r, 0, 1997 - 1600, 1997 - 1450).unwrap();
        assert_eq!((iv.ps, iv.pe, iv.rev), (4450, 4603, true));
        // An interval ending inside an insertion; one inside a clip.
        let i = table("100M5I100M", false, 0);
        assert!(lift_interval(&i, 0, 50, 103).unwrap().edge_ins);
        let c = table("300S1000M", false, 0);
        assert_eq!(lift_interval(&c, 0, 250, 400).unwrap().n_clip, 50);
        assert!(lift_interval(&c, 0, 0, 1301).is_none());
    }

    #[test]
    fn bed_footprint_overlap() {
        let mut bed = ExcludeBed::new();
        bed.insert(0, vec![(10, 20), (30, 40)]);
        assert!(excluded_footprint(&bed, 0, 19, 25));
        assert!(!excluded_footprint(&bed, 0, 20, 30));
        assert!(excluded_footprint(&bed, 0, 0, 11));
        assert!(!excluded_footprint(&bed, 1, 0, 100));
    }
}

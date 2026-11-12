//! `--meth` record extras: `XR:Z`/`XG:Z`/`XM:Z`, `MQ:i`, `HN:i`, the FASTQ comment re-encoded as
//! SAM tags, and the QC-fail rules.
//!
//! Ported from fg-labs/bwa-mem3 0.16 (`meth_bam.cpp`, `meth_xm.cpp`, `bam_writer.cpp`), Copyright
//! (C) 2026 Fulcrum Genomics LLC, MIT licensed; the notice is retained in the repository `LICENSE`.
//! The fork writes these records through htslib; the text here is what htslib prints for them.

use std::borrow::Cow;

use bwa_core::meth::{Chem, MethOpt, TAG_XG, TAG_XM, TAG_XR};
use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};

/// `opt` with the scoring matrix of a read of chemistry `ot` (read 1 / single-end C->T when true,
/// read 2 G->A otherwise): what extension, CIGAR, NM and XA of that read score with. `opt` itself
/// outside `--meth`.
pub fn read_opt(opt: &MemOpt, ot: bool) -> Cow<'_, MemOpt> {
    match &opt.meth {
        None => Cow::Borrowed(opt),
        Some(meth) => {
            let mut o = opt.clone();
            o.mat = meth.mats[usize::from(ot)];
            Cow::Owned(o)
        }
    }
}

/// What `mem_aln2sam` needs to write one read's records the `--meth` way.
pub struct MethEmit<'a> {
    pub fm: &'a FmIndex,
    pub meth: &'a MethOpt,
    /// The read's chemistry: read 1 / single-end (C->T) is `true`, read 2 (G->A) `false`.
    pub ot: bool,
}

/// bwameth's chimera rule: the longest M run must cover at least 44% of the read.
const MIN_LONGEST_M_PCT: usize = 44;

impl MethEmit<'_> {
    /// Apply `--set-as-failed` and `--chimera-qc` to a mapped record's FLAG and MAPQ.
    /// `top` is its conversion strand (XG:Z:CT), `cigar` its mem-op CIGAR.
    pub fn qc(&self, top: bool, cigar: &[u32], l_seq: usize, flag: &mut u32, mapq: &mut u32) {
        if self.meth.set_as_failed == Some(if top { b'f' } else { b'r' }) {
            *flag |= 0x200;
        }
        if self.meth.chimera_qc && !cigar.is_empty() && l_seq > 0 {
            let longest_m = cigar
                .iter()
                .filter(|&&c| c & 0xf == 0)
                .map(|&c| (c >> 4) as usize)
                .max()
                .unwrap_or(0);
            if 100 * longest_m < MIN_LONGEST_M_PCT * l_seq {
                *flag = (*flag | 0x200) & !0x2;
                *mapq = (*mapq).min(1);
            }
        }
    }

    /// The tags after `XA:Z`/`HN:i`: `XR:Z` (every record), `XG:Z` and `XM:Z` (mapped records;
    /// XM only when SEQ was written), then the FASTQ comment's SAM tags.
    ///
    /// `top` is `Some(conversion strand)` for a mapped record; `seq` is the emitted SEQ text,
    /// `hard` whether the CIGAR's clips were printed as H.
    #[allow(clippy::too_many_arguments)]
    pub fn push_tags(
        &self,
        bns: &BntSeq,
        out: &mut Vec<u8>,
        top: Option<bool>,
        rid: i32,
        pos: i64,
        cigar: &[u32],
        hard: bool,
        seq: &[u8],
        comment: Option<&str>,
    ) {
        let tags = self.meth.tags;
        if tags & TAG_XR != 0 {
            out.extend_from_slice(if self.ot { b"\tXR:Z:CT" } else { b"\tXR:Z:GA" });
        }
        if let Some(top) = top {
            if tags & TAG_XG != 0 {
                out.extend_from_slice(if top { b"\tXG:Z:CT" } else { b"\tXG:Z:GA" });
            }
            if tags & TAG_XM != 0 && !seq.is_empty() && !cigar.is_empty() {
                let xm = build_xm(
                    self.fm,
                    bns,
                    rid,
                    pos,
                    top,
                    cigar,
                    hard,
                    seq,
                    self.meth.chem,
                );
                out.extend_from_slice(b"\tXM:Z:");
                out.extend_from_slice(&xm);
            }
        }
        push_comment_tags(out, comment);
    }
}

/// `meth_build_xm`: Bismark's per-base call string over the emitted SEQ (forward orientation).
///
/// `.` everywhere except at a reference C (top strand; G on the bottom strand) read as C or T
/// (G or A), where the letter gives the context (Z/z CpG, X/x CHG, H/h CHH, U/u unknown, from the
/// next two reference bases, or the previous two on the bottom strand) and its case the call:
/// upper case methylated. Which read base means methylated depends on the chemistry.
#[allow(clippy::too_many_arguments)]
pub fn build_xm(
    fm: &FmIndex,
    bns: &BntSeq,
    rid: i32,
    pos: i64,
    top: bool,
    cigar: &[u32],
    hard: bool,
    seq: &[u8],
    chem: Chem,
) -> Vec<u8> {
    let rlen: i64 = cigar
        .iter()
        .filter(|&&c| c & 0xf == 0 || c & 0xf == 2)
        .map(|&c| i64::from(c >> 4))
        .sum();
    let refs = ref_slice(fm, bns, rid, pos - 2, pos + rlen + 2);
    xm_calls(&refs, top, cigar, hard, seq, chem)
}

/// [`build_xm`] on a reference slice already cut: `refs` holds the forward bases from two before
/// the alignment start to two past its end.
fn xm_calls(refs: &[u8], top: bool, cigar: &[u32], hard: bool, seq: &[u8], chem: Chem) -> Vec<u8> {
    let (c_marker, cpg_marker) = if top { (b'C', b'G') } else { (b'G', b'C') };
    // (methylated, unmethylated) read base: EM-seq keeps a methylated C, TAPS converts it.
    let (meth_b, unmeth_b) = match (chem, top) {
        (Chem::EmSeq, true) => (b'C', b'T'),
        (Chem::EmSeq, false) => (b'G', b'A'),
        (Chem::Taps, true) => (b'T', b'C'),
        (Chem::Taps, false) => (b'A', b'G'),
    };
    let at = |i: i64| {
        refs.get(i as usize)
            .copied()
            .filter(|_| i >= 0)
            .unwrap_or(b'N')
    };
    let mut xm = Vec::with_capacity(seq.len());
    let mut t = 0i64;
    for &c in cigar {
        let (op, len) = (c & 0xf, (c >> 4) as usize);
        match op {
            0 => {
                for _ in 0..len {
                    if xm.len() >= seq.len() {
                        break;
                    }
                    let ri = t + 2;
                    let read_b = seq[xm.len()];
                    let methylated = if read_b == meth_b {
                        Some(true)
                    } else if read_b == unmeth_b {
                        Some(false)
                    } else {
                        None
                    };
                    let call = match methylated {
                        Some(m) if at(ri) == c_marker => {
                            let (ctx1, ctx2) = if top {
                                (at(ri + 1), at(ri + 2))
                            } else {
                                (at(ri - 1), at(ri - 2))
                            };
                            let letter = if ctx1 == b'N' {
                                b'U'
                            } else if ctx1 == cpg_marker {
                                b'Z'
                            } else if ctx2 == b'N' {
                                b'U'
                            } else if ctx2 == cpg_marker {
                                b'X'
                            } else {
                                b'H'
                            };
                            if m {
                                letter
                            } else {
                                letter.to_ascii_lowercase()
                            }
                        }
                        _ => b'.',
                    };
                    xm.push(call);
                    t += 1;
                }
            }
            1 => xm.extend(std::iter::repeat_n(
                b'.',
                len.min(seq.len().saturating_sub(xm.len())),
            )),
            3 if !hard => xm.extend(std::iter::repeat_n(
                b'.',
                len.min(seq.len().saturating_sub(xm.len())),
            )),
            2 => t += len as i64,
            _ => {}
        }
    }
    xm.resize(seq.len(), b'.');
    xm
}

/// `orig_ref_slice`: contig `rid`'s forward bases `[st, en)` (contig coordinates) as ASCII, `N`
/// outside the contig and over the ambiguous runs the index replaced with random bases.
fn ref_slice(fm: &FmIndex, bns: &BntSeq, rid: i32, st: i64, en: i64) -> Vec<u8> {
    let contig = &bns.contigs[rid as usize];
    let (off, len) = (contig.offset, i64::from(contig.len));
    let mut out: Vec<u8> = (st..en)
        .map(|p| {
            if p < 0 || p >= len {
                b'N'
            } else {
                b"ACGT"[fm.base(off + p) as usize & 3]
            }
        })
        .collect();
    let (lo, hi) = (off + st.max(0), off + en.min(len));
    for amb in &bns.ambs {
        let (a, b) = (amb.offset, amb.offset + i64::from(amb.len));
        if b <= lo || a >= hi {
            continue;
        }
        for p in a.max(lo)..b.min(hi) {
            out[(p - off - st) as usize] = b'N';
        }
    }
    out
}

/// `append_sam_aux_tokens`: the FASTQ comment's `XX:T:value` tokens as htslib prints them once
/// stored (`A` first character, `i` as int32, `f` via `%g`, `Z`/`H` verbatim); any other token is
/// dropped, as the fork does.
pub fn push_comment_tags(out: &mut Vec<u8>, comment: Option<&str>) {
    let Some(comment) = comment else { return };
    for tok in comment.split('\t') {
        let b = tok.as_bytes();
        if b.len() < 6 || b[2] != b':' || b[4] != b':' {
            continue;
        }
        let (tag, typ, val) = (&tok[..2], b[3], &tok[5..]);
        let text = match typ {
            b'A' => val.chars().next().map(String::from),
            b'i' => leading_int(val).map(|v| (v as i32).to_string()),
            b'f' => leading_float(val).map(|v| format_g(v as f32)),
            b'Z' | b'H' => Some(val.to_string()),
            _ => None,
        };
        if let Some(text) = text {
            out.push(b'\t');
            out.extend_from_slice(tag.as_bytes());
            out.extend_from_slice(&[b':', typ, b':']);
            out.extend_from_slice(text.as_bytes());
        }
    }
}

/// `strtoll(val, &end, 10)` that succeeded: the longest leading `[+-]digits` (after whitespace).
fn leading_int(val: &str) -> Option<i64> {
    let v = val.trim_start();
    let end = v
        .char_indices()
        .find(|&(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '+' || c == '-'))))
        .map_or(v.len(), |(i, _)| i);
    let digits = &v[..end];
    digits.parse::<i64>().ok().or_else(|| {
        // Out of range: strtoll saturates.
        digits[1.min(digits.len())..]
            .bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| {
                if digits.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                }
            })
            .filter(|_| digits.bytes().any(|c| c.is_ascii_digit()))
    })
}

/// `strtof` that succeeded, on the decimal forms a FASTQ comment carries.
// shortcut: hex floats, inf/nan spellings and trailing junk handling are approximated by the
// longest parseable prefix; upgrade if a real comment ever carries them.
fn leading_float(val: &str) -> Option<f64> {
    let v = val.trim_start();
    (1..=v.len())
        .rev()
        .filter(|&i| v.is_char_boundary(i))
        .find_map(|i| v[..i].parse::<f64>().ok())
}

/// C's `%g` (6 significant digits, trailing zeros dropped), which htslib uses for `f` tags.
fn format_g(v: f32) -> String {
    let v = f64::from(v);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let exp = format!("{v:.5e}");
    let (mant, e) = exp.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if (-4..6).contains(&e) {
        trim(&format!("{v:.*}", (5 - e) as usize))
    } else {
        format!(
            "{}e{}{:02}",
            trim(mant),
            if e < 0 { '-' } else { '+' },
            e.abs()
        )
    }
}

/// `meth_bam_group_propagate_qcfail` on one read's or pair's SAM lines: once any record is QC-fail
/// (0x200), all of them are, and none is a proper pair (0x2).
pub fn propagate_qcfail(buf: &mut Vec<u8>) {
    let flag_of = |line: &[u8]| -> Option<(usize, usize, u32)> {
        let start = line.iter().position(|&c| c == b'\t')? + 1;
        let len = line[start..].iter().position(|&c| c == b'\t')?;
        let flag = std::str::from_utf8(&line[start..start + len])
            .ok()?
            .parse()
            .ok()?;
        Some((start, len, flag))
    };
    let lines: Vec<&[u8]> = buf.split_inclusive(|&c| c == b'\n').collect();
    if !lines
        .iter()
        .any(|l| flag_of(l).is_some_and(|(_, _, f)| f & 0x200 != 0))
    {
        return;
    }
    let mut out = Vec::with_capacity(buf.len() + 8);
    for line in lines {
        match flag_of(line) {
            Some((start, len, flag)) => {
                out.extend_from_slice(&line[..start]);
                out.extend_from_slice(((flag | 0x200) & !0x2).to_string().as_bytes());
                out.extend_from_slice(&line[start + len..]);
            }
            None => out.extend_from_slice(line),
        }
    }
    *buf = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comment_tokens_reencode_like_htslib() {
        let mut out = Vec::new();
        push_comment_tags(
            &mut out,
            Some("BC:Z:ACGT\tXi:i:007\tbad\tXf:f:0.50\tXa:A:xy\tXB:B:c,1\tXh:H:1AE3"),
        );
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\tBC:Z:ACGT\tXi:i:7\tXf:f:0.5\tXa:A:x\tXh:H:1AE3"
        );
    }

    #[test]
    fn percent_g_matches_c() {
        for (v, s) in [
            (1.0, "1"),
            (0.1, "0.1"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5, "-2.5"),
        ] {
            assert_eq!(format_g(v), s, "{v}");
        }
    }

    /// bwa's internal CIGAR op: length << 4 | op, with ops `MIDSH` = 0..4.
    fn op(len: u32, op: u32) -> u32 {
        len << 4 | op
    }

    #[test]
    fn xm_letters_follow_context_strand_and_chemistry() {
        // Top strand: CpG at 0, CHG at 3, CHH at 6, and at 9 a C whose second context base lies
        // past the contig (N), so unknown.
        let refs = b"TTCGACAGCTTCANN";
        let read = b"CGATAGCTTTA";
        let m11 = [op(11, 0)];
        assert_eq!(
            xm_calls(refs, true, &m11, false, read, Chem::EmSeq),
            b"Z..x..H..u."
        );
        assert_eq!(
            xm_calls(refs, true, &m11, false, read, Chem::Taps),
            b"z..X..h..U."
        );
        // A read A over a reference C is neither call (a SNP, not a conversion).
        assert_eq!(
            xm_calls(refs, true, &m11, false, b"AGATAGCTTTA", Chem::EmSeq),
            b"...x..H..u."
        );
        // Bottom strand: G is the marker and the context runs leftwards (CpG at 1, CHH at 4).
        let m5 = [op(5, 0)];
        assert_eq!(
            xm_calls(b"TTCGAAGTT", false, &m5, false, b"CGAAA", Chem::EmSeq),
            b".Z..h"
        );
        // A soft clip and an insertion take '.', a deletion skips reference only.
        let cigar = [op(1, 3), op(2, 0), op(1, 1), op(1, 2), op(3, 0)];
        let got = xm_calls(b"NNCGACGTNN", true, &cigar, false, b"ACGTTGT", Chem::EmSeq);
        assert_eq!(got, b".Z..z..");
        // A hard clip is not in SEQ, so it gets no letter.
        let got = xm_calls(b"NNCGACGTNN", true, &cigar, true, b"CGTTGT", Chem::EmSeq);
        assert_eq!(got, b"Z..z..");
    }

    #[test]
    fn qcfail_spreads_to_the_whole_pair() {
        let mut buf = b"r\t99\tc\n r\t147\tc\n".to_vec();
        propagate_qcfail(&mut buf);
        assert_eq!(buf, b"r\t99\tc\n r\t147\tc\n");
        let mut buf = b"r\t611\tc\nr\t147\tc\n".to_vec();
        propagate_qcfail(&mut buf);
        assert_eq!(buf, b"r\t609\tc\nr\t657\tc\n");
    }
}

//! Reference metadata parsing (`.ann` + `.amb`), mirroring bwa-mem2's `bntseq_t`.
//!
//! Formats (text), from `reference/bwa-mem2/src/bntseq.cpp`:
//! - `.ann`: line 1 `l_pac n_seqs seed`; then per contig two lines: `gi name anno`
//!   and `offset len n_ambs`. `anno` is literally `(null)` when absent.
//! - `.amb`: line 1 `l_pac n_seqs n_holes`; then per hole `offset len amb_char`.

use bwa_core::{Error, Result};
use std::path::{Path, PathBuf};

/// One reference contig (from `.ann`).
#[derive(Debug, Clone)]
pub struct Contig {
    pub gi: i64,
    pub name: String,
    pub anno: String,
    pub offset: i64,
    pub len: i32,
    pub n_ambs: i32,
}

/// An ambiguous-base run (from `.amb`).
#[derive(Debug, Clone)]
pub struct Amb {
    pub offset: i64,
    pub len: i32,
    pub amb: char,
}

/// Reference metadata parsed from `<prefix>.ann` and `<prefix>.amb`.
#[derive(Debug, Clone)]
pub struct BntSeq {
    pub l_pac: i64,
    pub n_seqs: i32,
    pub seed: u32,
    pub contigs: Vec<Contig>,
    pub n_holes: i32,
    pub ambs: Vec<Amb>,
}

impl BntSeq {
    /// Load `<prefix>.ann` and `<prefix>.amb`. `prefix` is the FASTA path passed to `index`; the
    /// index files are siblings named `<prefix>.ann`, `<prefix>.amb`, ...
    pub fn load(prefix: &Path) -> Result<Self> {
        let ann = std::fs::read_to_string(sibling(prefix, "ann"))?;
        let amb = std::fs::read_to_string(sibling(prefix, "amb"))?;
        let mut s = parse_ann(&ann)?;
        parse_amb(&amb, &mut s)?;
        Ok(s)
    }

    /// Map a 2L-space position to `(forward_pos, is_rev)`, mirroring `bns_depos`.
    #[inline]
    pub fn depos(&self, pos: i64) -> (i64, bool) {
        if pos >= self.l_pac {
            ((self.l_pac << 1) - 1 - pos, true)
        } else {
            (pos, false)
        }
    }

    /// Clamp a 2L-space reference window `[beg, end)` to the contig containing `mid`, mirroring the
    /// clamp inside `bns_fetch_seq`. Returns `(beg, end, rid)`.
    ///
    /// This is what stops an extension from running off the end of a contig into whatever sequence
    /// the packed reference happens to hold next: bwa resolves the seed's contig, then trims the
    /// window to that contig's span (flipped onto the reverse strand when the seed is reverse). It
    /// matters wherever a read reaches a contig edge, most visibly on the circular MT genome.
    pub fn fetch_bounds(&self, beg: i64, end: i64, mid: i64) -> (i64, i64, i32) {
        let (mid_f, is_rev) = self.depos(mid);
        let rid = self.pos2rid(mid_f);
        if rid < 0 {
            return (beg, end, rid);
        }
        let c = &self.contigs[rid as usize];
        let (mut far_beg, mut far_end) = (c.offset, c.offset + i64::from(c.len));
        if is_rev {
            let tmp = far_beg;
            far_beg = (self.l_pac << 1) - far_end;
            far_end = (self.l_pac << 1) - tmp;
        }
        (beg.max(far_beg), end.min(far_end), rid)
    }

    /// Contig index containing forward position `pos_f`, or -1, mirroring `bns_pos2rid`.
    pub fn pos2rid(&self, pos_f: i64) -> i32 {
        if pos_f >= self.l_pac {
            return -1;
        }
        let mut left = 0i32;
        let mut mid = 0i32;
        let mut right = self.n_seqs;
        while left < right {
            mid = (left + right) >> 1;
            if pos_f >= self.contigs[mid as usize].offset {
                if mid == self.n_seqs - 1 {
                    break;
                }
                if pos_f < self.contigs[(mid + 1) as usize].offset {
                    break;
                }
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        mid
    }

    /// Contig index for the interval `[rb, re)` in 2L space, or a negative code if it bridges a
    /// contig boundary (-1) or the forward/reverse midpoint (-2). Mirrors `bns_intv2rid`.
    pub fn intv2rid(&self, rb: i64, re: i64) -> i32 {
        if rb < self.l_pac && re > self.l_pac {
            return -2;
        }
        let rid_b = self.pos2rid(self.depos(rb).0);
        let rid_e = if rb < re {
            self.pos2rid(self.depos(re - 1).0)
        } else {
            rid_b
        };
        if rid_b == rid_e {
            rid_b
        } else {
            -1
        }
    }
}

fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn fmt_err(what: &str) -> Error {
    Error::IndexFormat(what.to_string())
}

fn parse_ann(text: &str) -> Result<BntSeq> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| fmt_err(".ann: empty file"))?;
    let mut h = header.split_whitespace();
    let l_pac: i64 = next_parse(&mut h, ".ann l_pac")?;
    let n_seqs: i32 = next_parse(&mut h, ".ann n_seqs")?;
    let seed: u32 = next_parse(&mut h, ".ann seed")?;

    let mut contigs = Vec::with_capacity(n_seqs.max(0) as usize);
    for i in 0..n_seqs {
        let l1 = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".ann: missing name line for contig {i}")))?;
        // "<gi> <name> <anno...>": contig names contain no spaces, anno is the remainder.
        let mut p = l1.splitn(3, ' ');
        let gi: i64 = p
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| fmt_err(".ann: bad gi"))?;
        let name = p
            .next()
            .ok_or_else(|| fmt_err(".ann: bad name"))?
            .to_string();
        let anno = p.next().unwrap_or("").to_string();

        let l2 = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".ann: missing offset line for contig {i}")))?;
        let mut q = l2.split_whitespace();
        let offset: i64 = next_parse(&mut q, ".ann offset")?;
        let len: i32 = next_parse(&mut q, ".ann len")?;
        let n_ambs: i32 = next_parse(&mut q, ".ann n_ambs")?;

        contigs.push(Contig {
            gi,
            name,
            anno,
            offset,
            len,
            n_ambs,
        });
    }

    Ok(BntSeq {
        l_pac,
        n_seqs,
        seed,
        contigs,
        n_holes: 0,
        ambs: Vec::new(),
    })
}

fn parse_amb(text: &str, s: &mut BntSeq) -> Result<()> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| fmt_err(".amb: empty file"))?;
    let mut h = header.split_whitespace();
    let _l_pac: i64 = next_parse(&mut h, ".amb l_pac")?;
    let _n_seqs: i32 = next_parse(&mut h, ".amb n_seqs")?;
    let n_holes: i32 = next_parse(&mut h, ".amb n_holes")?;
    s.n_holes = n_holes;
    for i in 0..n_holes {
        let l = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".amb: missing hole line {i}")))?;
        let mut q = l.split_whitespace();
        let offset: i64 = next_parse(&mut q, ".amb offset")?;
        let len: i32 = next_parse(&mut q, ".amb len")?;
        let amb = q.next().and_then(|x| x.chars().next()).unwrap_or('N');
        s.ambs.push(Amb { offset, len, amb });
    }
    Ok(())
}

fn next_parse<'a, T, I>(it: &mut I, what: &str) -> Result<T>
where
    T: std::str::FromStr,
    I: Iterator<Item = &'a str>,
{
    it.next()
        .ok_or_else(|| fmt_err(&format!("{what}: missing field")))?
        .parse::<T>()
        .map_err(|_| fmt_err(&format!("{what}: parse error")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_committed_tiny_index() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let bns = BntSeq::load(Path::new(prefix)).unwrap();
        assert_eq!(bns.l_pac, 200_001);
        assert_eq!(bns.n_seqs, 1);
        assert_eq!(bns.seed, 11);
        assert_eq!(bns.n_holes, 0);
        assert_eq!(bns.contigs.len(), 1);
        assert_eq!(bns.contigs[0].name, "20:2000000-2200000");
        assert_eq!(bns.contigs[0].len, 200_001);
        assert_eq!(bns.contigs[0].offset, 0);
    }
}

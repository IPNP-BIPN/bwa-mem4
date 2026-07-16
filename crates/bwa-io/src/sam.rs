//! Hand-formatted SAM output (header + records), byte-controlled for bit-identity.

use std::io::{self, Write};

/// One `@SQ` reference-sequence line: name and length.
pub struct SqRecord {
    pub name: String,
    pub len: i64,
}

/// Write the SAM header. bwa-mem2 emits NO `@HD` line; it starts with `@SQ` (one per contig) then
/// a single `@PG`. We reproduce that exactly for the `@SQ` lines (gated for byte-identity); `@PG`
/// is ours by design and excluded from the gate.
pub fn write_header<W: Write>(
    w: &mut W,
    sqs: &[SqRecord],
    pg_id: &str,
    pg_pn: &str,
    pg_vn: &str,
    pg_cl: &str,
) -> io::Result<()> {
    for sq in sqs {
        writeln!(w, "@SQ\tSN:{}\tLN:{}", sq.name, sq.len)?;
    }
    writeln!(w, "@PG\tID:{pg_id}\tPN:{pg_pn}\tVN:{pg_vn}\tCL:{pg_cl}")?;
    Ok(())
}

/// Write one unmapped SE record (FLAG 4). SEQ is emitted as sequenced; QUAL falls back to `*`.
///
/// The trailing `AS:i:0 XS:i:0` are not cosmetic: bwa builds an unmapped record from a zeroed
/// `mem_aln_t` (`mem_reg2aln` with a null region), so its `score`/`sub` are 0 rather than negative,
/// and `mem_aln2sam` emits both tags under `if (p->score >= 0)` / `if (p->sub >= 0)`.
pub fn write_unmapped<W: Write>(
    w: &mut W,
    qname: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
) -> io::Result<()> {
    w.write_all(qname.as_bytes())?;
    w.write_all(b"\t4\t*\t0\t0\t*\t*\t0\t0\t")?;
    w.write_all(seq)?;
    w.write_all(b"\t")?;
    match qual {
        Some(q) if !q.is_empty() => w.write_all(q)?,
        _ => w.write_all(b"*")?,
    }
    w.write_all(b"\tAS:i:0\tXS:i:0\n")?;
    Ok(())
}

/// Write one mapped single-end record. CIGAR/MAPQ are provided by the caller; RNEXT/PNEXT/TLEN
/// are the SE defaults (`* 0 0`).
#[allow(clippy::too_many_arguments)]
pub fn write_mapped_se<W: Write>(
    w: &mut W,
    qname: &str,
    flag: u32,
    rname: &str,
    pos: i64,
    mapq: u32,
    cigar: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    tags: &str,
) -> io::Result<()> {
    write!(
        w,
        "{qname}\t{flag}\t{rname}\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t"
    )?;
    w.write_all(seq)?;
    w.write_all(b"\t")?;
    match qual {
        Some(q) if !q.is_empty() => w.write_all(q)?,
        _ => w.write_all(b"*")?,
    }
    if !tags.is_empty() {
        w.write_all(b"\t")?;
        w.write_all(tags.as_bytes())?;
    }
    w.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_has_no_hd_and_exact_sq() {
        let mut buf = Vec::new();
        let sqs = vec![SqRecord {
            name: "20:2000000-2200000".into(),
            len: 200_001,
        }];
        write_header(
            &mut buf,
            &sqs,
            "bwa-mem3",
            "bwa-mem3",
            "0.0.0",
            "bwa-mem3 mem x",
        )
        .unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("@SQ\tSN:20:2000000-2200000\tLN:200001\n"));
        assert!(!s.contains("@HD"));
        assert!(s.contains("\n@PG\tID:bwa-mem3\t"));
    }

    #[test]
    fn unmapped_record_shape() {
        let mut buf = Vec::new();
        write_unmapped(&mut buf, "r1", b"ACGT", Some(b"IIII")).unwrap();
        assert_eq!(&buf, b"r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\tAS:i:0\tXS:i:0\n");
    }

    #[test]
    fn unmapped_missing_qual_is_star() {
        let mut buf = Vec::new();
        write_unmapped(&mut buf, "r1", b"ACGT", None).unwrap();
        assert_eq!(&buf, b"r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t*\tAS:i:0\tXS:i:0\n");
    }
}

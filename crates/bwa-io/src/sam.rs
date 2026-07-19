//! Hand-formatted SAM output (header + records), byte-controlled for bit-identity.
//!
//! # The SAM record, for readers who have not met it
//!
//! SAM is a tab-separated text format: a header of `@`-prefixed lines, then one line per alignment.
//! Every alignment line has the same 11 mandatory fields in this fixed order, followed by any
//! number of optional `TAG:TYPE:VALUE` tags:
//!
//! | # | Field | Meaning | Value here |
//! |---|-------|---------|------------|
//! | 1 | QNAME | read name | FASTQ header up to the first space, `/1`-`/2` stripped |
//! | 2 | FLAG  | bitfield | see [`write_mapped_se`] and `cmd_mem::write_aln_se` |
//! | 3 | RNAME | contig name | `*` if unmapped |
//! | 4 | POS   | leftmost position, **1-based** | 0 if unmapped; callers add 1 to their 0-based `pos` |
//! | 5 | MAPQ  | mapping quality, 0..=60 | 0 for secondary and for unmapped |
//! | 6 | CIGAR | alignment ops | `*` if unmapped |
//! | 7 | RNEXT | mate's contig | `*` single-end, `=` when the mate is on the same contig |
//! | 8 | PNEXT | mate's position | 0 single-end |
//! | 9 | TLEN  | observed template length | 0 single-end |
//! | 10 | SEQ  | read bases | reverse-complemented when FLAG 0x10 is set; `*` for secondary |
//! | 11 | QUAL | Phred+33 qualities | `*` when the input was FASTA or the read is secondary |
//!
//! MAPQ is Phred-scaled: 60 means bwa estimates a 1-in-10^6 chance the position is wrong, 0 means
//! the read maps equally well elsewhere. SEQ is stored relative to the REFERENCE strand, not as
//! sequenced, which is why the reverse-strand path reverse-complements the bases and reverses (but
//! does not complement) the qualities.
//!
//! # Why hand-formatted
//!
//! Byte-identity with bwa-mem2 is the project's gate, and it is a property of the exact bytes: field
//! order, tag order, `*`-vs-empty conventions, and whether a value is printed as `0` or omitted. A
//! SAM library would make its own defensible choices about all of these. Writing the bytes here
//! makes every one of those choices explicit and reviewable.
//!
//! # Glossary
//!
//! | Term | Plain language |
//! |------|----------------|
//! | contig | one named sequence of the reference (a chromosome, a scaffold, a patch) |
//! | `rid` | contig id: index into the contig table; the caller turns it into the RNAME string |
//! | CIGAR | run-length list of alignment ops: `M` aligned, `I`/`D` insert/delete, `S`/`H` clipped |
//! | soft clip (`S`) | read bases not aligned but still stored in SEQ |
//! | hard clip (`H`) | read bases not aligned and NOT stored, because another record holds them |
//! | tag | an optional `TAG:TYPE:VALUE` field after column 11, e.g. `NM:i:2` |
//!
//! Reading order: [`write_header`] (once per file), then [`write_unmapped`] and
//! [`write_mapped_se`] (once per emitted record). None of the three makes a decision: the caller
//! (`bwa-cli::cmd_mem::write_aln_se`) has already chosen every field and every tag.

use std::io::{self, Write};

/// SAM's placeholder for "no value here". Used for RNAME/CIGAR of an unmapped read, for QUAL of a
/// FASTA-sourced read, and for SEQ/QUAL of a secondary record whose bases live on the primary.
///
/// The single byte `*` is fixed by the SAM spec for every column that can be absent. Only the QUAL
/// (column 11) fallback goes through this constant here; the `*` values in the hard-coded unmapped
/// middle and in the SE mate defaults are baked into their literals. Changing it produces a file no
/// SAM parser accepts, and breaks byte-identity on every FASTA-input and every secondary record.
const SAM_MISSING: &[u8] = b"*";

/// One `@SQ` reference-sequence line: a reference contig's name and length.
///
/// `name` becomes `SN:` and must not contain whitespace (it is the same string later used as
/// RNAME); `len` becomes `LN:` and is the contig length in bases. Supplied by the caller from the
/// loaded index's contig table, in index order, which is the order they must be emitted in.
pub struct SqRecord {
    /// The contig name, written as `SN:<name>` in the `@SQ` line. The very same string is later
    /// written as SAM column 3 (RNAME) on every record aligned to this contig, and inside SA:Z /
    /// XA:Z tag groups, so any whitespace in it would corrupt the tab-separated line.
    pub name: String,
    /// Contig length in BASES, written as `LN:<len>`. Signed because the index stores 64-bit signed
    /// offsets; always positive in practice. Read only by [`write_header`].
    pub len: i64,
}

/// Write the SAM header. bwa-mem2 emits NO `@HD` line; it starts with `@SQ` (one per contig), then
/// `hdr_lines` (whatever `-H` inserted, with the `-R` `@RG` line appended to it), then a single
/// `@PG`. We reproduce that order exactly (`bwa_print_sam_hdr`); `@PG` is ours by design and
/// excluded from the byte-identity gate.
///
/// As in the C, supplying `@SQ` lines through `-H` suppresses the generated ones.
///
/// # Why no `@HD`
///
/// The SAM spec makes `@HD` (format version, sort order) optional, and bwa simply never writes one:
/// `bwa_print_sam_hdr` (`bwa.cpp:523`) goes straight to the `@SQ` loop. Since bwa emits records in
/// input order, there is no sort order it could honestly declare. Adding an `@HD` would be more
/// spec-complete and would break byte-identity, so we do not.
///
/// # Why `-H` @SQ lines suppress the generated ones
///
/// The C counts line-initial `@SQ\t` occurrences in `hdr_line` into `n_SQ` and only runs the
/// generating loop `if (n_SQ == 0)`. The reasoning is that a user supplying their own reference
/// dictionary means to replace ours, not to be given two conflicting ones. Note the C merely warns
/// (at `bwa_verbose >= 2`) when the counts disagree with the index and continues anyway; we do not
/// emit that warning, which is a stderr-only difference.
///
/// # Parameters
///
/// - `w`: the output sink, at byte 0 of the SAM stream (stdout or the `-o` file, usually behind a
///   `BufWriter`). Nothing has been written to it yet; this call must produce the whole header
///   before any record.
/// - `sqs`: contigs in index order. Emitted verbatim unless suppressed.
/// - `hdr_lines`: the `-H` insertions with the `-R` `@RG` line already appended, newline-joined, or
///   `None`. Written as one block, so its internal order is the caller's responsibility.
/// - `pg_id`/`pg_pn`/`pg_vn`/`pg_cl`: `@PG` program record: ID, program name, version, and the full
///   command line. `pg_cl` is raw argv joined by spaces; it is not re-quoted or escaped, matching
///   bwa, so a shell-quoted argument containing a space will not round-trip.
///
/// Emission order is exactly `@SQ*`, `hdr_lines`, `@PG`, and that order is part of the gate.
///
/// # Returns
///
/// Only propagates `w`'s I/O errors; the formatting itself cannot fail.
pub fn write_header<W: Write>(
    w: &mut W,
    sqs: &[SqRecord],
    hdr_lines: Option<&str>,
    pg_id: &str,
    pg_pn: &str,
    pg_vn: &str,
    pg_cl: &str,
) -> io::Result<()> {
    // The C scans for `@SQ\t` at a line start (`p == hdr_line || *(p-1) == '\n'`); splitting on '\n'
    // and testing each line's prefix is the same predicate, since the block is newline-joined.
    // ---- @SQ: one line per reference contig, unless the user supplied their own ----
    // True when the `-H` block already contains at least one line-initial `@SQ\t`, i.e. the user
    // brought their own reference dictionary; the C's `n_SQ != 0`. Suppresses OUR `@SQ` loop only,
    // never the user's lines, which are written unchanged further down.
    let user_supplied_sq =
        hdr_lines.is_some_and(|block| block.split('\n').any(|line| line.starts_with("@SQ\t")));
    if !user_supplied_sq {
        for sq in sqs {
            writeln!(w, "@SQ\tSN:{}\tLN:{}", sq.name, sq.len)?;
        }
    }
    // ---- The -H insertions with the -R @RG line already appended, verbatim ----
    if let Some(header_block) = hdr_lines {
        writeln!(w, "{header_block}")?;
    }
    // ---- @PG: how this file was produced. Ours by design, outside the byte-identity gate ----
    writeln!(w, "@PG\tID:{pg_id}\tPN:{pg_pn}\tVN:{pg_vn}\tCL:{pg_cl}")?;
    Ok(())
}

/// Write one unmapped SE record (FLAG 4). SEQ is emitted as sequenced; QUAL falls back to `*`.
///
/// A read with no alignment reaching `-T` is still written: SAM keeps unmapped reads so the file
/// remains a complete record of the input. The fixed middle of the line is
/// `4  *  0  0  *  *  0  0`, i.e. FLAG=4 (unmapped), RNAME/CIGAR `*`, POS/MAPQ 0, and the SE mate
/// defaults. SEQ is written as sequenced (there is no strand to flip to).
///
/// # Parameters
///
/// - `w`: the SAM output sink, positioned after the header and any previously emitted records.
/// - `qname`: SAM column 1, `Record::name` (already `/1`-`/2`-stripped). Must not contain a tab.
/// - `seq`: SAM column 10, the ASCII bases as sequenced. Never reverse-complemented here: an
///   unmapped record has no reference strand to be relative to.
/// - `qual`: SAM column 11, Phred+33, same length and same order as `seq`. `None` (FASTA input) or
///   an empty slice both write `*`.
/// - `comment`: `Record::comment`, the FASTQ header past the QNAME. Appended as the last
///   tab-separated field, and only when `-C` was given; `append_comment` makes that decision, so
///   passing `Some(..)` without `-C` still emits nothing.
///
/// # Returns
///
/// Only propagates `w`'s I/O errors; no field is validated and no formatting can fail.
///
/// The trailing `AS:i:0 XS:i:0` are not cosmetic: bwa builds an unmapped record from a zeroed
/// `mem_aln_t` (`mem_reg2aln` with a null region), so its `score`/`sub` are 0 rather than negative,
/// and `mem_aln2sam` emits both tags under `if (p->score >= 0)` / `if (p->sub >= 0)`.
pub fn write_unmapped<W: Write>(
    w: &mut W,
    qname: &str,
    seq: &[u8],
    qual: Option<&[u8]>,
    comment: Option<&str>,
) -> io::Result<()> {
    // ---- Columns 1..9: QNAME, then the fixed unmapped middle (FLAG 4, no RNAME/POS/MAPQ/CIGAR,
    //      SE mate defaults) ----
    w.write_all(qname.as_bytes())?;
    w.write_all(b"\t4\t*\t0\t0\t*\t*\t0\t0\t")?;
    // ---- Columns 10-11: SEQ as sequenced, QUAL or `*` ----
    w.write_all(seq)?;
    w.write_all(b"\t")?;
    match qual {
        Some(quals) if !quals.is_empty() => w.write_all(quals)?,
        _ => w.write_all(SAM_MISSING)?,
    }
    // ---- Tags, in bwa's fixed order: AS, XS, then RG:Z, then the -C comment last ----
    w.write_all(b"\tAS:i:0\tXS:i:0")?;
    // Everything after AS/XS, built up in bwa's emission order (RG:Z, then the `-C` comment) and
    // written as one block. Stays empty when neither `-R` nor `-C` is in effect, in which case the
    // line ends right after XS:i:0.
    let mut trailing_tags = Vec::new();
    // `-R`: bwa stamps RG:Z on every record, unmapped ones included.
    bwa_core::rg::append_rg_tag(&mut trailing_tags);
    // `-C`: the comment is last, after RG:Z.
    bwa_core::rg::append_comment(&mut trailing_tags, comment);
    w.write_all(&trailing_tags)?;
    w.write_all(b"\n")?;
    Ok(())
}

/// Write one mapped single-end record. CIGAR/MAPQ are provided by the caller; RNEXT/PNEXT/TLEN
/// are the SE defaults (`* 0 0`).
///
/// Pure formatting: every decision (which bits are in `flag`, whether `seq` is the forward or
/// reverse-complemented strand, which slice of the read it is, what is in `tags` and in what order)
/// has already been made by `cmd_mem::write_aln_se`. Nothing is validated here.
///
/// # Parameters
///
/// - `w`: the SAM output sink, positioned after the header and any previously emitted records. In
///   the threaded path this is a per-batch `Vec<u8>` that is concatenated in input order later, so
///   nothing here may depend on what other threads wrote.
/// - `qname`: SAM column 1, `Record::name`. Every record of one read (primary, secondary,
///   supplementary) repeats it unchanged; that is how a reader groups them.
/// - `rname`: SAM column 3, the contig name looked up by the caller from `aln.rid`. Never `*`:
///   an unmapped read goes to [`write_unmapped`] instead.
/// - `flag`: SAM column 2, the FINAL 16-bit SAM flag. The caller must already have collapsed bwa's internal
///   `0x10000` pseudo-bit; passing it through would emit an out-of-spec value.
/// - `pos`: SAM column 4, the **1-based** leftmost reference position, relative to the start of
///   `rname`'s contig. Callers hold 0-based positions and add 1. Must be >= 1.
/// - `mapq`: SAM column 5, 0..=60. 0 is meaningful (maps equally well elsewhere), not "unknown"; SAM spells
///   unknown as 255, which bwa never emits. The caller forces 0 on secondary records.
/// - `cigar`: SAM column 6, preformatted, already S/H-rewritten for this record's role, `*` if none.
///   Its consumed-query length must match `seq.len()` (hard clips excluded) or the record is
///   internally inconsistent; that is not checked here.
/// - `seq`: SAM column 10, bases in REFERENCE orientation, or `b"*"` for a secondary record. When
///   FLAG 0x10 is set the caller has already reverse-complemented it, and soft-clip trimming (the
///   `qb..qe` slice) has already been applied.
/// - `qual`: SAM column 11, same orientation and slice as `seq` (REVERSED, not complemented, on the
///   reverse strand); `None` or empty writes `*`.
/// - `tags`: everything after column 11, preformatted tab-separated by `cmd_mem::write_aln_se` in
///   bwa's fixed order (NM:i, AS:i, XS:i, RG:Z, SA:Z, XA:Z, then the `-C` comment), WITHOUT a
///   leading tab (added here). Empty is allowed and writes nothing, which is why an empty string
///   must not be `"\t"`.
///
/// # Returns
///
/// Only propagates `w`'s I/O errors.
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
    // ---- Columns 1-6 (QNAME..CIGAR), then 7-9 hard-coded to the single-end mate defaults
    //      (RNEXT `*`, PNEXT 0, TLEN 0); the paired-end path formats its own line ----
    write!(
        w,
        "{qname}\t{flag}\t{rname}\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t"
    )?;
    // ---- Columns 10-11: SEQ (already in reference orientation) and QUAL ----
    w.write_all(seq)?;
    w.write_all(b"\t")?;
    match qual {
        Some(quals) if !quals.is_empty() => w.write_all(quals)?,
        _ => w.write_all(SAM_MISSING)?,
    }
    // ---- Optional tags, preformatted by the caller; the leading tab belongs to us ----
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
            None,
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

    /// `-R`/`-H` land between the generated `@SQ` lines and `@PG`, in that order
    /// (`bwa_print_sam_hdr`), and user-supplied `@SQ` lines suppress the generated ones.
    #[test]
    fn header_inserts_rg_and_hdr_lines() {
        let sqs = vec![SqRecord {
            name: "chr1".into(),
            len: 100,
        }];
        let mut buf = Vec::new();
        write_header(
            &mut buf,
            &sqs,
            Some("@RG\tID:foo\tSM:bar"),
            "bwa-mem3",
            "bwa-mem3",
            "0.0.0",
            "cl",
        )
        .unwrap();
        let s = String::from_utf8(buf).unwrap();
        let sq = s.find("@SQ").unwrap();
        let rg = s.find("@RG").unwrap();
        let pg = s.find("@PG").unwrap();
        assert!(sq < rg && rg < pg, "order must be @SQ, @RG, @PG: {s}");

        // user-provided @SQ suppresses the generated ones
        let mut buf2 = Vec::new();
        write_header(
            &mut buf2,
            &sqs,
            Some("@SQ\tSN:other\tLN:7"),
            "bwa-mem3",
            "bwa-mem3",
            "0.0.0",
            "cl",
        )
        .unwrap();
        let s2 = String::from_utf8(buf2).unwrap();
        assert!(
            !s2.contains("SN:chr1"),
            "generated @SQ should be suppressed: {s2}"
        );
        assert!(s2.contains("SN:other"));
    }

    #[test]
    fn unmapped_record_shape() {
        let mut buf = Vec::new();
        write_unmapped(&mut buf, "r1", b"ACGT", Some(b"IIII"), None).unwrap();
        assert_eq!(
            &buf,
            b"r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\tAS:i:0\tXS:i:0\n"
        );
    }

    #[test]
    fn unmapped_missing_qual_is_star() {
        let mut buf = Vec::new();
        write_unmapped(&mut buf, "r1", b"ACGT", None, None).unwrap();
        assert_eq!(
            &buf,
            b"r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t*\tAS:i:0\tXS:i:0\n"
        );
    }
}

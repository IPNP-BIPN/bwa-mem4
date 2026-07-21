//! FASTQ reading with bwa-compatible QNAME derivation and fixed-size (`-K`) batching.
//!
//! # FASTQ, briefly
//!
//! Four lines per read: `@name optional comment`, the bases, a `+` separator, then one Phred+33
//! quality character per base. FASTA (no qualities) is also accepted, in which case `qual` is
//! `None` and SAM gets `*`. Files may be gzipped; needletail sniffs that for us.
//!
//! # Why batching is not just a performance detail
//!
//! Reads are handed downstream in batches sized by cumulative BASES, not by read count, because
//! bwa's `-K` is in bases. This matters beyond load balancing: for paired-end input the insert-size
//! distribution is re-estimated once per batch, so where the boundaries fall is visible in the
//! output. Fixing `-K` is what makes a run reproducible across thread counts.
//!
//! # Glossary
//!
//! | Term | Plain language |
//! |------|----------------|
//! | read | one short stretch of DNA as reported by the sequencer, typically 100-150 bases |
//! | QNAME | the read's name, SAM column 1; both mates of a pair share one |
//! | mate / pair | the same DNA fragment sequenced from each end, so the two reads face each other |
//! | Phred+33 | quality as one ASCII character per base: `chr(33 + q)`, where q is `-10 log10(P(wrong))` |
//! | `-K` | batch size in BASES (not reads), because that is bwa's unit |
//!
//! Reading order: [`Record`] (what comes out), [`FastqReader`] (one file), then the two paired
//! variants, then the two private header-splitting helpers at the bottom.

use bwa_core::{Error, Result};
use needletail::{parse_fastx_file, FastxReader};

/// One read: SAM QNAME, sequence, and (optional) quality string.
///
/// `name` is already bwa-normalised (see [`qname_from_id`]). `seq` is ASCII bases as sequenced, in
/// the orientation they came off the instrument, and is NOT uppercased or validated here: `dna::nt4`
/// maps anything that is not ACGT/acgt to 4 (N) at use site. `qual`, when present, is Phred+33 and
/// has the same length as `seq`, index for index.
pub struct Record {
    /// SAM column 1 (QNAME). FASTQ line 1 (`@...`) up to the first whitespace, with a trailing
    /// `/1`/`/2` trimmed. Set once at parse time by [`qname_from_id`]; read by every SAM emitter
    /// and, for `-p` input, by [`InterleavedFastqReader::next_batch`] to recognise mates.
    pub name: String,
    /// FASTQ line 2: the bases as ASCII, in the orientation they came off the instrument. Length is
    /// the read length (typically 100-150). Not uppercased, not validated, not 2-bit packed here:
    /// any byte outside ACGT/acgt becomes N (code 4) when `dna::nt4` is applied downstream. Feeds
    /// seeding, and (possibly reverse-complemented and soft-clip-sliced) SAM column 10 (SEQ).
    pub seq: Vec<u8>,
    /// FASTQ line 4: Phred+33 quality, one byte per base, so `qual.len() == seq.len()` index for
    /// index. `None` for FASTA input (no line 3/4 at all), which makes SAM column 11 (QUAL) `*`.
    /// Byte value range in practice 33..=73 (`!` to `I`, q = 0..=40).
    pub qual: Option<Vec<u8>>,
    /// Everything after the first whitespace of the header, as kseq's `comment`. Only emitted when
    /// `-C` is given (bwa frees it otherwise), so carrying it always costs one `Option` per read.
    ///
    /// From FASTQ line 1, after the QNAME field. `None` when the header is a bare name. Under `-C`
    /// it is appended verbatim as the LAST tab-separated field of the SAM line, after every tag
    /// including RG:Z/SA:Z/XA:Z; it is not a typed `TAG:TYPE:VALUE` tag.
    pub comment: Option<String>,
}

/// Streaming FASTQ reader.
pub struct FastqReader {
    /// needletail's parser, owning the file handle, its buffer and (for `.gz` input) the
    /// decompressor. Boxed because the concrete parser type depends on the compression needletail
    /// sniffed from the file's magic bytes. Advanced only by [`FastqReader::next_record`]; holds
    /// all the reader's position state, so this struct has no offset of its own.
    inner: Box<dyn FastxReader>,
}

impl FastqReader {
    /// Open a FASTQ (optionally gzipped) file.
    ///
    /// # Parameters
    ///
    /// - `path`: filesystem path supplied by the caller from argv. Plain or gzipped FASTQ, or
    ///   FASTA; the format and the compression are both sniffed by needletail, not from the
    ///   extension. Must exist and be readable, otherwise [`Error::Fastq`] is returned.
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let inner = parse_fastx_file(path).map_err(|e| Error::Fastq(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Pull the next read, or `None` at EOF.
    ///
    /// # Returns
    ///
    /// `Ok(None)` exactly at end of file (not an error: it is the stop condition), `Ok(Some(rec))`
    /// with all four FASTQ lines already split into [`Record`] fields, or [`Error::Fastq`] on a
    /// malformed record.
    pub fn next_record(&mut self) -> Result<Option<Record>> {
        match self.inner.next() {
            None => Ok(None),
            Some(rec) => {
                // One parsed record, borrowing needletail's internal buffer: everything taken from
                // it below must be copied out before the next `self.inner.next()`.
                let rec = rec.map_err(|e| Error::Fastq(e.to_string()))?;
                // `rec.id()` is FASTQ line 1 WITHOUT the leading `@`, i.e. name plus any comment.
                // Split into the two halves: `name` becomes SAM QNAME, `comment` the `-C` trailer.
                let name = qname_from_id(rec.id());
                let comment = comment_from_id(rec.id());
                // FASTQ line 2 (bases) and line 4 (Phred+33), both owned copies. `qual` is `None`
                // for FASTA input, where lines 3 and 4 do not exist.
                let seq = rec.seq().into_owned();
                let qual = rec.qual().map(<[u8]>::to_vec);
                Ok(Some(Record {
                    name,
                    seq,
                    qual,
                    comment,
                }))
            }
        }
    }

    /// Read a batch whose cumulative sequence length reaches at least `k_batch` bytes (or EOF).
    /// Fixing this boundary (via `-K`) is what makes downstream per-batch statistics reproducible.
    ///
    /// `k_batch` is in BASES, not reads or bytes-on-disk. The loop tests `bases < k_batch` before
    /// pulling, so a batch overshoots by at most one read: boundaries are deterministic given the
    /// input and `-K`, which is the property the byte-identity gate rests on. An empty return means
    /// EOF and is the pipeline's stop condition, so callers must not treat it as an error.
    ///
    /// # Parameters
    ///
    /// - `k_batch`: target batch size in BASES (summed FASTQ line 2 lengths), from `-K` (or
    ///   `-K`'s default scaled by the thread count in `cmd_mem`). Must be > 0, else every batch
    ///   comes back empty and the pipeline stops immediately.
    ///
    /// # Returns
    ///
    /// The reads of one batch, in file order. An empty vector means EOF.
    pub fn next_batch(&mut self, k_batch: usize) -> Result<Vec<Record>> {
        // Accumulators. Invariant at the top of each iteration: `batch` holds every read consumed
        // so far this call, in file order, and `bases_so_far` is exactly the sum of their
        // `seq.len()`, which is still strictly below `k_batch` (else we would have exited).
        let mut batch = Vec::new();
        let mut bases_so_far = 0usize;
        while bases_so_far < k_batch {
            match self.next_record()? {
                None => break,
                Some(rec) => {
                    bases_so_far += rec.seq.len();
                    batch.push(rec);
                }
            }
        }
        Ok(batch)
    }
}

/// Streaming reader over a pair of FASTQ files (R1, R2) advanced in lockstep.
///
/// The standard paired-end layout: the same fragment sequenced from both ends, with the two reads
/// at the same ordinal position in the two files. Pairing is therefore POSITIONAL, not by name; the
/// only name check is the length mismatch caught in [`Self::next_batch`]. That matches bwa, which
/// also trusts file order here.
pub struct PairedFastqReader {
    /// The R1 (first-in-pair, SAM FLAG 0x40) file. Read one record per pair.
    r1: FastqReader,
    /// The R2 (second-in-pair, SAM FLAG 0x80) file. Advanced in lockstep with `r1`: the two are
    /// always at the same ordinal record, which is the only thing that makes the pairing correct.
    r2: FastqReader,
}

impl PairedFastqReader {
    /// Open the two mate files.
    ///
    /// # Parameters
    ///
    /// - `p1`: path to the R1 FASTQ, `p2`: path to the R2 FASTQ, both from argv in that order.
    ///   They must hold the same number of records, in the same order; nothing is checked here
    ///   (the length mismatch surfaces later, in [`Self::next_batch`]).
    pub fn from_paths<P: AsRef<std::path::Path>>(p1: P, p2: P) -> Result<Self> {
        Ok(Self {
            r1: FastqReader::from_path(p1)?,
            r2: FastqReader::from_path(p2)?,
        })
    }

    /// Read a batch of read pairs whose cumulative sequence length (both mates) reaches at least
    /// `k_batch` bytes, or EOF. Boundaries fall on pair granularity so per-batch statistics
    /// (`mem_pestat`) are reproducible under a fixed `-K`.
    ///
    /// # Parameters
    ///
    /// - `k_batch`: target size in BASES, counting BOTH mates, from `-K`. Same units and same
    ///   role as [`FastqReader::next_batch`]'s.
    ///
    /// # Returns
    ///
    /// One `(R1, R2)` tuple per fragment, in file order; empty at EOF. [`Error::Fastq`] if one
    /// file runs out before the other.
    pub fn next_batch(&mut self, k_batch: usize) -> Result<Vec<(Record, Record)>> {
        // Invariant at the top of each iteration: `batch` holds the pairs read so far (both files
        // sit at the same ordinal record), `bases_so_far` is the sum of both mates' `seq.len()`
        // over those pairs, and is still below `k_batch`.
        let mut batch = Vec::new();
        let mut bases_so_far = 0usize;
        while bases_so_far < k_batch {
            match (self.r1.next_record()?, self.r2.next_record()?) {
                // Both files yielded: `mate1` is the R1 read, `mate2` the R2 read of one fragment.
                // They are paired by POSITION; their names are not compared here.
                (Some(mate1), Some(mate2)) => {
                    bases_so_far += mate1.seq.len() + mate2.seq.len();
                    batch.push((mate1, mate2));
                }
                // Clean EOF: both files ended on the same record boundary.
                (None, None) => break,
                // Exactly one file ended: the positional pairing is broken from here on, so every
                // remaining pair would be wrong. Refuse rather than emit mis-mated records.
                _ => return Err(Error::Fastq("paired FASTQ files differ in length".into())),
            }
        }
        Ok(batch)
    }
}

/// Streaming reader over ONE file holding both mates interleaved (`-p`, smart pairing).
///
/// bwa's smart pairing runs `bseq_classify`: consecutive reads sharing a QNAME form a pair, and
/// anything left over is realigned single-end and merged back in input order. We implement the
/// genuinely-interleaved case and *refuse* the mixed one rather than silently mis-pairing reads:
/// a singleton here would otherwise be paired with its neighbour and produce confidently wrong
/// mate coordinates. `qname_from_id` has already stripped `/1` and `/2`, so mates compare equal.
pub struct InterleavedFastqReader {
    /// The single input file, holding R1 and R2 alternating. Read twice per pair, so a pair
    /// boundary is simply an even record index; there is no separate parity flag to keep in sync.
    inner: FastqReader,
}

impl InterleavedFastqReader {
    /// Open the interleaved file.
    ///
    /// # Parameters
    ///
    /// - `p`: path to the one FASTQ holding both mates alternating (R1, R2, R1, R2, ...), from
    ///   argv under `-p`. Must contain an even number of records with mates adjacent.
    pub fn from_path<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        Ok(Self {
            inner: FastqReader::from_path(p)?,
        })
    }

    /// Read a batch of pairs, on the same `-K` cumulative-bases rule as [`PairedFastqReader`].
    ///
    /// # Parameters
    ///
    /// - `k_batch`: target size in BASES over both mates, from `-K`, as in the other two readers.
    ///
    /// # Returns
    ///
    /// One `(R1, R2)` tuple per fragment in file order; empty at EOF. [`Error::Fastq`] on an odd
    /// trailing record or on two adjacent records whose QNAMEs differ.
    pub fn next_batch(&mut self, k_batch: usize) -> Result<Vec<(Record, Record)>> {
        // Invariant at the top of each iteration: `inner` sits on an EVEN record index (a pair
        // boundary), `batch` holds the pairs consumed so far, and `bases_so_far` is their summed
        // two-mate base count, still below `k_batch`.
        let mut batch = Vec::new();
        let mut bases_so_far = 0usize;
        while bases_so_far < k_batch {
            // Even record: the R1 of the next pair. Absent means a clean EOF on a pair boundary.
            let Some(mate1) = self.inner.next_record()? else {
                break;
            };
            // Odd record: the R2 that must accompany it. Absent means the file ended mid-pair.
            let Some(mate2) = self.inner.next_record()? else {
                return Err(Error::Fastq(format!(
                    "-p: interleaved input ended on an unpaired read ('{}'). bwa would realign it \
                     single-end (bseq_classify); bwa-mem4 refuses rather than mis-pair it.",
                    mate1.name
                )));
            };
            // `qname_from_id` already stripped any `/1`/`/2`, so genuine mates compare equal here.
            if mate1.name != mate2.name {
                return Err(Error::Fastq(format!(
                    "-p: consecutive reads '{}' and '{}' are not mates. bwa would split these into \
                     a single-end pass (bseq_classify); bwa-mem4 refuses rather than mis-pair them.",
                    mate1.name, mate2.name
                )));
            }
            bases_so_far += mate1.seq.len() + mate2.seq.len();
            batch.push((mate1, mate2));
        }
        Ok(batch)
    }
}

/// Derive the SAM QNAME from a FASTQ id line, mirroring bwa: take the field up to the first
/// whitespace, then trim a trailing `/<digit>` (bwa's `trim_readno`).
/// kseq's `comment`: everything after the first run of whitespace in the header, or `None` when the
/// header is just a name. bwa appends it verbatim at the very end of the SAM record under `-C`.
///
/// `id` is the header line WITHOUT its leading `@`. Returns `None` when there is no whitespace, or
/// only trailing whitespace, so a header of `read1` and one of `read1   ` both yield no comment.
///
/// # Parameters
///
/// - `id`: FASTQ line 1 minus the `@`, exactly as needletail hands it over (name plus comment, no
///   trailing newline). Not required to be valid UTF-8: invalid bytes are lossily replaced.
///
/// # Returns
///
/// The comment, which under `-C` becomes the final field of the SAM line. `None` means no comment.
fn comment_from_id(id: &[u8]) -> Option<String> {
    // Byte offset of the first whitespace, i.e. one past the end of the QNAME field. `?` returns
    // `None` for a bare-name header, which has no comment by definition.
    let first_space = id.iter().position(u8::is_ascii_whitespace)?;
    // The tail starting AT that whitespace, so offsets found in it are relative to `first_space`.
    let after_name = &id[first_space..];
    // Skip the whole run of whitespace, not just one byte; `None` here means the header ended in
    // whitespace and so carries no comment.
    // Absolute offset into `id` of the comment's first non-whitespace byte; everything from there
    // to the end of the line, whitespace included, is the comment.
    let comment_start = after_name
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .map(|offset| first_space + offset)?;
    Some(String::from_utf8_lossy(&id[comment_start..]).into_owned())
}

/// The QNAME half of the split described above: header up to the first whitespace, minus a trailing
/// `/1` or `/2`.
///
/// Stripping the read-number suffix is what makes the two mates of a pair share one QNAME, as SAM
/// requires, and it is also what lets [`InterleavedFastqReader`] recognise mates by name equality.
/// The `s.len() > 2` guard means a read literally named `/1` keeps its name rather than becoming
/// empty. Note the test is on the LAST two bytes only, so `read/12` is left alone (bwa behaves the
/// same way: `trim_readno` checks a single digit).
///
/// # Parameters
///
/// - `id`: FASTQ line 1 minus the `@`, the same slice [`comment_from_id`] is given.
///
/// # Returns
///
/// The string written as SAM column 1 (QNAME) on every record for this read, and (for `-p` input)
/// compared against the neighbouring record's to confirm the two are mates.
fn qname_from_id(id: &[u8]) -> String {
    // One past the last QNAME byte: the first whitespace, or the whole slice for a bare-name
    // header (which is the common case for simulated reads).
    let name_end = id
        .iter()
        .position(u8::is_ascii_whitespace)
        .unwrap_or(id.len());
    // The QNAME candidate, narrowed by the read-number trim below before being copied out.
    let mut name = &id[..name_end];
    // Trailing `/<digit>` (bwa's `trim_readno`). READNO_SUFFIX_LEN is the `/` plus the digit.
    if name.len() > READNO_SUFFIX_LEN
        && name[name.len() - 2] == b'/'
        && name[name.len() - 1].is_ascii_digit()
    {
        name = &name[..name.len() - READNO_SUFFIX_LEN];
    }
    String::from_utf8_lossy(name).into_owned()
}

/// Length of the read-number suffix bwa trims: the `/` and the single digit after it.
///
/// Fixed at 2 by the `/<digit>` convention in bwa's `trim_readno`, which tests exactly one digit.
/// It is used both as the length to cut and as the minimum name length to allow cutting, so
/// raising it would strip multi-digit suffixes bwa keeps (`read/12`) and break the two mates'
/// QNAMEs apart, which is byte-visible in SAM column 1 and, under `-p`, breaks mate detection.
const READNO_SUFFIX_LEN: usize = 2;

#[cfg(test)]
mod tests {
    use super::qname_from_id;

    #[test]
    fn qname_strips_comment_and_readno() {
        assert_eq!(qname_from_id(b"read1 some comment"), "read1");
        assert_eq!(qname_from_id(b"read1/1"), "read1");
        assert_eq!(qname_from_id(b"read1/2 desc"), "read1");
        assert_eq!(
            qname_from_id(b"20:2000000-2200000_50861_51313_0:0:0_0:1:0_0"),
            "20:2000000-2200000_50861_51313_0:0:0_0:1:0_0"
        );
    }
}

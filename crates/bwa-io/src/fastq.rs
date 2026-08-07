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
//!
//! # Rust mechanics used in this file
//!
//! `Result`, `?` and `Option` are glossed in full in `bwa_core::error` and `bwa_core::cpu`; this
//! table covers what is new here. The two ideas worth the most attention are the trait object that
//! hides which decompressor is in use, and the borrowed-versus-owned distinction that forces every
//! field of [`Record`] to be copied out of the parser's buffer.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `Box<dyn FastxReader>` | a value on the heap whose exact type is not known at compile time, only the set of operations it supports (the "trait"). Here it may be a plain-file parser or a gzip-decompressing one; the code below calls `.next()` on it without caring which. `Box` is needed because the two possibilities have different sizes. |
//! | `dyn` | marks that dispatch happens at run time, through a hidden pointer to the right implementation, rather than being resolved by the compiler. |
//! | `impl` | a block attaching methods to a type. `FastqReader` gets three; they are called as `reader.next_batch(...)`. |
//! | `&mut self` | the method borrows the reader exclusively and advances it. That is why reading is a mutation: position lives inside the parser, not in a returned cursor. |
//! | `P: AsRef<Path>` | a generic parameter with a BOUND: accept any type that can be viewed as a filesystem path. It is what lets callers pass a `&str`, a `String` or a `PathBuf` without converting first. The compiler generates a version per type actually used, so this costs nothing at run time. |
//! | `Self` | inside an `impl`, the type being implemented. `Ok(Self { inner })` builds one and wraps it in a success. |
//! | `Self { inner }` | field shorthand: when a local variable has the same name as the field, writing it once means `inner: inner`. |
//! | `Vec<u8>` vs `&[u8]` | owned bytes versus a borrowed window onto somebody else's. [`Record`] holds owned copies because the parser reuses one internal buffer for every record; a borrow would dangle as soon as the next read is pulled. |
//! | `.into_owned()` | turns a maybe-borrowed value into a definitely-owned one, copying only if it was still a borrow. Used to lift text out of the parser's buffer. |
//! | `String::from_utf8_lossy` | interprets bytes as text, substituting a replacement character for anything invalid instead of failing. Read names are not guaranteed to be valid UTF-8, and refusing a file over a stray byte in a name would be worse than mangling that one name. |
//! | `match (a, b)` | matching on a PAIR of values at once. The paired reader uses it to handle all four end-of-file combinations of two files in one place, and the compiler checks that none is forgotten. |
//! | `_ => ...` | the catch-all arm, here standing for "exactly one of the two files ended". |
//! | `let Some(x) = ... else { ... }` | bind and continue, or run a block that must leave the function. Used for the two mid-pair end-of-file checks. |
//! | `format!(...)` | builds an owned `String` from a template, `{}` marking where each argument goes. |
//! | `.into()` | converts into whatever the surrounding code needs, here a string literal into the owned `String` the error variant carries. |
//! | `.iter().position(...)` | walks and returns the index of the first element passing a test, or `None`. |
//! | `u8::is_ascii_whitespace` | a method passed BY NAME as the test, instead of writing `\|c\| c.is_ascii_whitespace()`. Identical meaning, shorter. |
//! | `&id[a..b]`, `&id[a..]` | a slice of an existing buffer, with no copy. An omitted bound means "to the end". |

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

/// Open a FASTQ/FASTA file and return the parser for it, whatever its compression.
///
/// Plain and gzip go straight to needletail, which sniffs the magic bytes and builds its own
/// decompressor. That path is untouched and stays on flate2 with the `zlib-rs` backend, which is
/// where the measured inflate speed lives.
///
/// With `--features multi-format`, anything needletail does not recognise is offered to niffler
/// first, which adds bzip2, xz and zstd. The order matters and is not arbitrary: sniffing here
/// rather than dispatching on the file extension means a `.fq` that is really zstd still works, and
/// a gzip file never leaves the fast path because it is recognised before niffler is consulted.
///
/// # Parameters
/// * `path`: the file, from argv. Format and compression are both detected from content.
///
/// # Errors
/// [`Error::Fastq`] if the file cannot be opened, is empty, or is not a format any enabled backend
/// recognises. The message names the path, since a user who passed the wrong file gets nothing else
/// to go on.
fn open_reader(path: &std::path::Path) -> Result<Box<dyn FastxReader>> {
    // The first two bytes decide: 0x1f 0x8b is gzip, and needletail handles both that and plain
    // text. Read them without consuming the file, since needletail opens it again by path.
    let magic = {
        use std::io::Read;
        let mut f = std::fs::File::open(path)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        let mut buf = [0u8; 2];
        // A file shorter than two bytes cannot be a FASTQ; let needletail produce that error.
        let n = f.read(&mut buf).unwrap_or(0);
        if n == 2 {
            Some(buf)
        } else {
            None
        }
    };
    let is_gzip = magic == Some([0x1f, 0x8b]);
    let looks_like_text = magic.is_some_and(|m| m[0] == b'@' || m[0] == b'>');

    if is_gzip || looks_like_text || cfg!(not(feature = "multi-format")) {
        return parse_fastx_file(path).map_err(|e| Error::Fastq(e.to_string()));
    }

    #[cfg(feature = "multi-format")]
    {
        // niffler picks the decompressor from the same magic bytes and hands back a `Read`;
        // needletail then parses the decompressed stream exactly as it would a plain file.
        let (reader, _format) = niffler::send::from_path(path)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        needletail::parse_fastx_reader(reader).map_err(|e| Error::Fastq(e.to_string()))
    }
    #[cfg(not(feature = "multi-format"))]
    unreachable!("the branch above returns when the feature is off")
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
        Ok(Self {
            inner: open_reader(path.as_ref())?,
        })
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
                //
                // Rust: this is where the borrowed-to-owned copy happens, and it is not optional.
                // `.into_owned()` and `.to_vec()` each allocate a fresh buffer, because `rec` is a
                // window onto memory needletail will overwrite on the next call. `.map(...)` on the
                // quality applies the copy only when there is one, leaving `None` alone, which is
                // how FASTA input (no quality line at all) passes through untouched.
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
            // Rust: both files are advanced, then the PAIR of answers is matched at once. The three
            // arms below cover all four combinations, and the compiler refuses the code if one is
            // left out. That exhaustiveness is what guarantees the "one file ran out early" case
            // cannot be forgotten, which is the case that would otherwise emit mis-mated records.
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
    //
    // Rust: `.position(...)` walks until the test passes and yields that index, or `None` if it
    // never does. The method is passed by name rather than wrapped in a closure. The trailing `?`
    // turns "no whitespace anywhere" into an immediate `None` return from this function.
    let first_space = id.iter().position(u8::is_ascii_whitespace)?;
    // The tail starting AT that whitespace, so offsets found in it are relative to `first_space`.
    let after_name = &id[first_space..];
    // Skip the whole run of whitespace, not just one byte; `None` here means the header ended in
    // whitespace and so carries no comment.
    // Absolute offset into `id` of the comment's first non-whitespace byte; everything from there
    // to the end of the line, whitespace included, is the comment.
    //
    // Rust: the `.map(...)` in the middle is a coordinate fix, not a search. `.position` measured
    // from the start of `after_name`, but the slice on the next line indexes into `id`, so the two
    // origins have to be reconciled. Getting this wrong would silently truncate every comment by
    // the length of the read name. The `?` after it returns `None` for a header that was nothing
    // but trailing whitespace.
    let comment_start = after_name
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .map(|offset| first_space + offset)?;
    // Copy the tail out as owned text. `from_utf8_lossy` tolerates invalid bytes rather than
    // rejecting the record, and `.into_owned()` makes the result independent of `id`.
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
    //
    // Rust: `.unwrap_or(v)` supplies a fallback when the search found nothing, so "no whitespace"
    // becomes "the name runs to the end of the line" instead of an error. This is the common case
    // for simulated reads, not an edge case.
    let name_end = id
        .iter()
        .position(u8::is_ascii_whitespace)
        .unwrap_or(id.len());
    // The QNAME candidate, narrowed by the read-number trim below before being copied out.
    //
    // Rust: `mut` here makes the BORROW re-pointable, not the bytes writable. The line below can
    // therefore aim `name` at a shorter window of the same buffer, and `id` itself is never
    // modified. No copy happens until the `from_utf8_lossy` at the end.
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

#[cfg(all(test, feature = "multi-format"))]
mod multi_format_tests {
    use super::*;
    use std::io::Write;

    /// The same reads, compressed four ways, must parse to the same records.
    ///
    /// This is the whole promise of the feature: a `.zst` FASTQ is not a different input, it is the
    /// same input spelled differently, and the aligner downstream cannot tell. Building the fixtures
    /// here rather than committing four binaries keeps the test honest about what it compresses.
    #[test]
    fn every_compression_yields_the_same_records() {
        let plain = b"@r1 comment\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nTTTTGGGGCC\n+\nJJJJJJJJJJ\n";
        let dir = std::env::temp_dir().join(format!("bwa4_multifmt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let write = |name: &str, bytes: &[u8]| -> std::path::PathBuf {
            let p = dir.join(name);
            std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
            p
        };
        // niffler's writer is the same dispatcher the reader uses, so the fixtures are compressed by
        // the code path the test is about rather than by whatever tool happens to be installed.
        let compress = |name: &str, fmt: niffler::compression::Format| -> std::path::PathBuf {
            let p = dir.join(name);
            let mut w = niffler::to_path(&p, fmt, niffler::Level::One).unwrap();
            w.write_all(plain).unwrap();
            drop(w);
            p
        };

        let paths = [
            write("reads.fq", plain),
            compress("reads.fq.gz", niffler::compression::Format::Gzip),
            compress("reads.fq.bz2", niffler::compression::Format::Bzip),
            compress("reads.fq.zst", niffler::compression::Format::Zstd),
            compress("reads.fq.xz", niffler::compression::Format::Lzma),
        ];

        // Records from the uncompressed file, the reference every other spelling must match.
        let read_all = |p: &std::path::Path| -> Vec<(String, Vec<u8>, Option<Vec<u8>>)> {
            let mut r = FastqReader::from_path(p).unwrap();
            let mut out = Vec::new();
            while let Some(rec) = r.next_record().unwrap() {
                out.push((rec.name.clone(), rec.seq.clone(), rec.qual.clone()));
            }
            out
        };
        let expected = read_all(&paths[0]);
        assert_eq!(expected.len(), 2, "fixture should hold two reads");
        for p in &paths[1..] {
            assert_eq!(
                read_all(p),
                expected,
                "{} parsed differently from the uncompressed fixture",
                p.display()
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

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

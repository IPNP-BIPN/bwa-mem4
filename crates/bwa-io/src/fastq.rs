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
//! Reading order: [`Record`] (what comes out), [`FastqReader`] (one file), then `RecordStream`
//! (one file on its own thread) and the two paired variants, then the two private header-splitting
//! helpers at the bottom.
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
/// `name()` is already bwa-normalised (see [`split_id`]). `seq()` is ASCII bases as sequenced, in
/// the orientation they came off the instrument, and is NOT uppercased or validated here: `dna::nt4`
/// maps anything that is not ACGT/acgt to 4 (N) at use site. `qual()`, when present, is Phred+33 and
/// has the same length as `seq()`, index for index.
///
/// # Why one buffer instead of four fields
///
/// This used to be `String` + `Vec<u8>` + `Option<Vec<u8>>` + `Option<String>`, so every read cost
/// three heap allocations and three copies, and the struct was 96 bytes. Measured on 1 M real GIAB
/// reads, parsing with needletail and building this record: the parser alone takes 0.048 s and the
/// copy-out takes it to 0.113 s. The copy-out was more expensive than the parse.
///
/// That also ruled out the obvious move of swapping parsers. seq_io measures 0.049 s on the same
/// file, i.e. the same as needletail, and paraseq says of itself that it "matches the performance of
/// the zero-copy parsers"; its gain is over ONE-COPY parsers, which is what this type was.
///
/// One allocation per read, laid out `name | seq | qual | comment`, brings that to 0.067 s, and the
/// struct to 32 bytes. The size matters twice over: the reader moves every record twice (out of the
/// per-file chunk channel, then into the batch vector), and at 1 M pairs that is 192 MB of moves
/// instead of 576 MB.
pub struct Record {
    /// `name | seq | qual | comment`, concatenated. Exactly the size of its contents: built with
    /// `Vec::with_capacity` of the summed lengths, so `into_boxed_slice` never reallocates.
    buf: Box<[u8]>,
    /// Bytes of `buf` holding SAM column 1 (QNAME), from offset 0. FASTQ line 1 (`@...`) up to the
    /// first whitespace, with a trailing `/1`/`/2` trimmed, lossily converted to UTF-8. Read by
    /// every SAM emitter and, for `-p` input, by [`InterleavedFastqReader::next_batch`] to
    /// recognise mates.
    name_len: u32,
    /// Bytes of `buf` holding FASTQ line 2, the bases as ASCII in the orientation they came off the
    /// instrument, starting at `name_len`. Typically 100-150. Not uppercased, not validated, not
    /// 2-bit packed here: any byte outside ACGT/acgt becomes N (code 4) when `dna::nt4` is applied
    /// downstream. Feeds seeding, and (possibly reverse-complemented and soft-clip-sliced) SAM
    /// column 10 (SEQ).
    seq_len: u32,
    /// Bytes of `buf` holding the kseq `comment`, at the end. 0 means no comment, which is the same
    /// thing here: [`split_id`] only reports a comment when it has at least one non-whitespace byte.
    ///
    /// From FASTQ line 1, after the QNAME field. Under `-C` it is appended verbatim as the LAST
    /// tab-separated field of the SAM line, after every tag including RG:Z/SA:Z/XA:Z; it is not a
    /// typed `TAG:TYPE:VALUE` tag.
    comment_len: u32,
    /// Whether FASTQ line 4 is present, i.e. whether `seq_len` bytes of Phred+33 quality follow the
    /// sequence. A flag rather than a length because quality is always exactly as long as the
    /// sequence, and rather than `comment_len`'s zero-means-absent trick because a zero-length read
    /// would then be indistinguishable from a FASTA one. `false` makes SAM column 11 `*`.
    has_qual: bool,
}

impl Record {
    /// SAM column 1 (QNAME).
    ///
    /// # Panics
    /// Never in practice: the bytes were produced by `String::from_utf8_lossy` at parse time, so
    /// they are valid UTF-8 by construction. The check is kept rather than elided with `unsafe`
    /// because it costs a scan of ~40 bytes per call against the ~150-byte copies this type exists
    /// to remove.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.name_len as usize])
            .expect("record name is UTF-8 by construction (from_utf8_lossy at parse time)")
    }

    /// FASTQ line 2: the ASCII bases.
    #[inline]
    #[must_use]
    pub fn seq(&self) -> &[u8] {
        let start = self.name_len as usize;
        &self.buf[start..start + self.seq_len as usize]
    }

    /// FASTQ line 4: Phred+33 quality, one byte per base, or `None` for FASTA input.
    #[inline]
    #[must_use]
    pub fn qual(&self) -> Option<&[u8]> {
        if !self.has_qual {
            return None;
        }
        let start = self.name_len as usize + self.seq_len as usize;
        Some(&self.buf[start..start + self.seq_len as usize])
    }

    /// The kseq `comment`, or `None` for a bare-name header.
    ///
    /// # Panics
    /// Never in practice, for the same reason as [`Record::name`].
    #[inline]
    #[must_use]
    pub fn comment(&self) -> Option<&str> {
        if self.comment_len == 0 {
            return None;
        }
        let end = self.buf.len();
        Some(
            std::str::from_utf8(&self.buf[end - self.comment_len as usize..])
                .expect("record comment is UTF-8 by construction (from_utf8_lossy at parse time)"),
        )
    }
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
    // Whether the path can be opened a second time and read from the start. A regular file can; a
    // FIFO, a process substitution (`<(zcat r1.gz)`) or a character device cannot, because the two
    // magic bytes read below are consumed from the stream and never come back.
    //
    // This distinction used to be missing, and the consequence was silent: sniffing consumed two
    // bytes, `parse_fastx_file` reopened the path, the parser saw a stream already past its header
    // and produced NO records, so `bwa-mem4 mem ref <(zcat r1.gz)` wrote a header and zero
    // alignments with no error at all. bwa-mem2 reads that input, so this was a parity gap as well
    // as a wrong answer. Non-seekable input now keeps its first two bytes and never reopens.
    let seekable = std::fs::metadata(path).is_ok_and(|m| m.is_file());
    let mut opened =
        std::fs::File::open(path).map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
    // The first two bytes decide: 0x1f 0x8b is gzip, and needletail handles both that and plain
    // text.
    let magic = {
        use std::io::Read;
        let mut buf = [0u8; 2];
        // A file shorter than two bytes cannot be a FASTQ; let needletail produce that error.
        let n = opened.read(&mut buf).unwrap_or(0);
        if n == 2 {
            Some(buf)
        } else {
            None
        }
    };
    let is_gzip = magic == Some([0x1f, 0x8b]);
    let looks_like_text = magic.is_some_and(|m| m[0] == b'@' || m[0] == b'>');

    // Non-seekable: hand needletail the bytes already taken, chained back in front of the rest, and
    // let it sniff compression itself. Sequential by construction, which is also what any parallel
    // decoder would have to fall back to on a stream it cannot seek in.
    if !seekable {
        let head = std::io::Cursor::new(magic.map(Vec::from).unwrap_or_default());
        let stream = std::io::Read::chain(head, opened);
        return needletail::parse_fastx_reader(stream).map_err(|e| Error::Fastq(e.to_string()));
    }

    // Gzipped input, parallel path: rapidgzip decodes ONE stream on several threads and hands back
    // a `Read`, so needletail parses plain text exactly as it would from an uncompressed file. A
    // pipe or FIFO is not seekable and the crate falls back to sequential decoding on its own, so
    // there is no input to special-case here. See `gzip_threads` for the budget.
    #[cfg(feature = "parallel-gzip")]
    if is_gzip {
        let dec = rapidgzip_core::Decoder::builder()
            .decoder_threads(gzip_threads())
            .build()
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        let stream = dec
            .open(path)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        return needletail::parse_fastx_reader(stream).map_err(|e| Error::Fastq(e.to_string()));
    }

    // Gzipped input, single-thread fallback: inflate on a thread of its own and hand needletail the
    // decompressed bytes, so that inflating block N+1 overlaps parsing block N instead of following
    // it. See [`spawn_inflate`] for the measurement that motivates it.
    #[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
    if is_gzip {
        let blocks = spawn_inflate(path)?;
        return needletail::parse_fastx_reader(blocks).map_err(|e| Error::Fastq(e.to_string()));
    }

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

/// Decoder threads per gzipped input file, as set by the binary from `-t`. `None` until then.
#[cfg(feature = "parallel-gzip")]
static GZIP_THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Tell the reader how many threads one gzipped file's decoder may use.
///
/// Called once by the binary before any input is opened, with `-t`. The value is halved because a
/// paired-end run opens TWO files and each gets its own decoder, so the two together stay inside
/// the thread budget the user asked for. It is not a throughput decision: the decoders matter
/// during the FIRST batch, when every aligner worker is idle because no batch has arrived yet.
///
/// # Parameters
/// * `threads`: the run's `-t`. 0 is treated as 1.
#[cfg(feature = "parallel-gzip")]
pub fn set_gzip_threads(threads: usize) {
    let _ = GZIP_THREADS.set(threads.max(1).div_ceil(2));
}

/// Threads for one file's decoder: what the binary set, overridden by `BWA4_GZIP_THREADS`.
///
/// `BWA4_GZIP_THREADS` exists so the budget can be swept on a new machine rather than argued about;
/// 1 reproduces the single-thread inflater's throughput without rebuilding. A value that does not
/// parse, or 0, is ignored rather than clamped silently, so a typo in a benchmark script cannot
/// quietly change what is being measured.
#[cfg(feature = "parallel-gzip")]
fn gzip_threads() -> usize {
    if let Ok(s) = std::env::var("BWA4_GZIP_THREADS") {
        if let Ok(n) = s.trim().parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    // No setter call (a library user, or a unit test): fall back to the machine rather than to 1,
    // since the whole point is not to leave the decoder single-threaded by accident.
    *GZIP_THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .div_ceil(2)
    })
}

/// Decompressed bytes handed over per channel message. 4 MiB is about 12 000 reads at 150 bp, so
/// the channel is touched ~90 times for a 1 M-pair mate file, and three of them in flight cost
/// 12 MiB, which is noise next to a batch of records.
#[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
const INFLATE_BLOCK: usize = 4 << 20;

/// Blocks the inflater may run ahead. One in flight plus one being parsed is the double buffering
/// that makes the overlap work; the third absorbs a slow parse without stalling the inflater.
#[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
const INFLATE_DEPTH: usize = 3;

/// The decompressed side of a gzipped FASTQ, as a plain [`std::io::Read`] fed by an inflater thread.
///
/// Why this exists. `parse_fastx_file` builds `MultiGzDecoder` INSIDE the parser, so one thread
/// alternates between inflating and parsing: a file's reader time is `inflate + parse` rather than
/// `max(inflate, parse)`. That difference is invisible when the run has many batches (the reader is
/// then fully hidden behind the aligner) and expensive when it does not. Measured on 1 M real GIAB
/// pairs against GRCh38, `wait_read` (the main thread blocked on the reader) is:
///
/// | `-K` | batches | `wait_read` gzipped | plain |
/// |---|---|---|---|
/// | 20 M | 15 | 0.097 s | 0.000 s |
/// | 40 M | 8 | 0.184 s | |
/// | 160 M (the `-t16` default) | 2 | **0.682 s, 4.2 % of the run** | 0.219 s |
///
/// The default `-K` is `10M * threads`, so the batch grows with `-t` while the file does not: at
/// `-t16` a 1 M-pair run is two batches and the FIRST one is exposed in full, since it has no
/// predecessor to overlap with. The index load, the other thing it could hide behind, is 0.28 s on
/// a warm page cache and is already overlapped.
///
/// This cannot move a byte of output: inflate is a bijection, so the parser sees the same bytes in
/// the same order, hence the same records and the same `-K` boundaries.
///
/// Not a parallel decoder. One stream is still inflated by one thread, at the 898 MB/s this
/// machine measures for `zlib-rs` (`gzcat` on Apple's zlib does 1137 MB/s on the same file). What
/// is bought here is the overlap, not the throughput.
#[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
struct InflatedBlocks {
    /// Blocks from the inflater thread, or the first I/O error it hit.
    rx: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    /// The block currently being handed out.
    cur: Vec<u8>,
    /// How much of `cur` has already been returned by `read`.
    pos: usize,
    /// Set once the stream has ended or reported its error, so a later `read` cannot block on a
    /// channel whose sender is gone.
    done: bool,
}

#[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
impl std::io::Read for InflatedBlocks {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.pos < self.cur.len() {
                let n = (self.cur.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.cur[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(Ok(block)) => {
                    self.cur = block;
                    self.pos = 0;
                }
                Ok(Err(e)) => {
                    self.done = true;
                    return Err(e);
                }
                // Every sender gone: the inflater returned, which is end of file.
                Err(_) => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

/// Open `path`, inflate it on a fresh thread, and return the decompressed stream.
///
/// The thread is not joined: dropping the returned value drops the receiver, the next `send` fails
/// and the thread returns. That is the whole shutdown protocol, and it is why no handle is kept.
#[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
fn spawn_inflate(path: &std::path::Path) -> Result<InflatedBlocks> {
    use std::io::Read;
    let file =
        std::fs::File::open(path).map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::io::Result<Vec<u8>>>(INFLATE_DEPTH);
    std::thread::spawn(move || {
        // `MultiGzDecoder`, not `GzDecoder`: a FASTQ may be several gzip members concatenated, and
        // stopping at the first one would silently truncate the file. Same choice needletail makes.
        let mut dec =
            flate2::read::MultiGzDecoder::new(std::io::BufReader::with_capacity(1 << 20, file));
        loop {
            let mut block = vec![0u8; INFLATE_BLOCK];
            // Fill the block completely unless the stream ends, so the parser is not handed a
            // trickle of short reads at whatever size the decoder felt like returning.
            let mut filled = 0;
            let mut failed = None;
            while filled < INFLATE_BLOCK {
                match dec.read(&mut block[filled..]) {
                    Ok(0) => break,
                    Ok(k) => filled += k,
                    Err(e) => {
                        failed = Some(e);
                        break;
                    }
                }
            }
            let last = filled < INFLATE_BLOCK;
            block.truncate(filled);
            if filled > 0 && tx.send(Ok(block)).is_err() {
                return; // consumer went away
            }
            if let Some(e) = failed {
                let _ = tx.send(Err(e));
                return;
            }
            if last {
                return;
            }
        }
    });
    Ok(InflatedBlocks {
        rx,
        cur: Vec::new(),
        pos: 0,
        done: false,
    })
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
                // Split into the two halves: the first becomes SAM QNAME, the second the `-C`
                // trailer. Both are still borrowed from needletail's buffer at this point.
                let (name_bytes, comment_bytes) = split_id(rec.id());
                // Lossy UTF-8, as before, and free in the normal case: `from_utf8_lossy` borrows
                // when the bytes are already valid and only allocates for a header with a stray
                // byte, which is the rare path this tolerates rather than rejecting the file over.
                let name = String::from_utf8_lossy(name_bytes);
                let comment = comment_bytes.map(String::from_utf8_lossy);
                // FASTQ line 2 (bases) and line 4 (Phred+33). `qual` is `None` for FASTA input,
                // where lines 3 and 4 do not exist.
                let seq = rec.seq();
                let qual = rec.qual();
                // The copy out of needletail's buffer is not optional (it reuses that buffer for the
                // next record), but it is ONE allocation sized exactly right, not three. See the
                // type's own docs for what that is worth.
                let comment_len = comment.as_ref().map_or(0, |c| c.len());
                let mut buf = Vec::with_capacity(
                    name.len() + seq.len() + qual.map_or(0, <[u8]>::len) + comment_len,
                );
                buf.extend_from_slice(name.as_bytes());
                buf.extend_from_slice(&seq);
                if let Some(q) = qual {
                    buf.extend_from_slice(q);
                }
                if let Some(c) = &comment {
                    buf.extend_from_slice(c.as_bytes());
                }
                Ok(Some(Record {
                    name_len: name.len() as u32,
                    seq_len: seq.len() as u32,
                    comment_len: comment_len as u32,
                    has_qual: qual.is_some(),
                    buf: buf.into_boxed_slice(),
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
                    bases_so_far += rec.seq().len();
                    batch.push(rec);
                }
            }
        }
        Ok(batch)
    }
}

/// How many records one background mate-reader hands over per channel message.
///
/// Big enough that the channel is touched once per few thousand reads instead of once per read
/// (about 2 MB of records at 150 bp), small enough that the two streams stay within a few
/// milliseconds of each other rather than one running a whole batch ahead of the other.
const STREAM_CHUNK_RECORDS: usize = 8192;

/// Chunks a mate reader may have ready before it blocks. One in flight plus one being consumed is
/// double buffering; deeper only buys memory, exactly as for the batch queue in `cmd_mem`.
const STREAM_CHUNK_DEPTH: usize = 2;

/// One FASTQ file being parsed on its own thread, delivered in chunks.
///
/// This exists because a paired-end run reads two INDEPENDENT files, and doing so on one thread
/// makes their decompression and parsing add up on the serial path in front of the aligner. Under
/// the default `-K` (10 M bases per thread) a 500k-pair run is a single batch, so that serial time
/// is not overlapped with anything at all: measured at `-t16` on chr21, `wait_read` is 0.428 s for
/// gzipped input against 0.153 s for plain, and every millisecond of it is wall clock the sixteen
/// workers spend idle.
///
/// Splitting per file cannot move a byte of output: the records are the same records in the same
/// order, and the batch boundary is still decided by the consumer applying the same cumulative-base
/// rule to the same interleaved sequence.
struct RecordStream {
    /// Chunks from the reader thread, or the first error it hit. `None` once dropped, which is how
    /// [`Drop`] releases the thread before joining it.
    rx: Option<std::sync::mpsc::Receiver<Result<Vec<Record>>>>,
    /// The chunk currently being handed out, record by record.
    cur: std::vec::IntoIter<Record>,
    /// Set once the stream has yielded its last record or reported its error, so a second call
    /// cannot block on a channel whose sender is gone.
    done: bool,
    /// The reader thread, joined on drop after `rx` is released.
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RecordStream {
    /// Open `path` on a fresh thread and start filling the channel.
    ///
    /// Opening happens on the thread too, so an unreadable file surfaces as the stream's first
    /// item rather than at construction; the caller sees it on the first [`Self::next_record`].
    fn spawn(path: std::path::PathBuf) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<Record>>>(STREAM_CHUNK_DEPTH);
        let handle = std::thread::spawn(move || {
            let mut reader = match FastqReader::from_path(&path) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            loop {
                let mut chunk = Vec::with_capacity(STREAM_CHUNK_RECORDS);
                let mut failed = None;
                while chunk.len() < STREAM_CHUNK_RECORDS {
                    match reader.next_record() {
                        Ok(Some(rec)) => chunk.push(rec),
                        Ok(None) => break,
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                // A short chunk means the file ended or the parse failed; either way this is the
                // last message. Send what was parsed first so a mid-file error still reports the
                // records before it, which is what makes the pair-length check meaningful.
                let last = chunk.len() < STREAM_CHUNK_RECORDS;
                if !chunk.is_empty() && tx.send(Ok(chunk)).is_err() {
                    return; // consumer went away
                }
                if let Some(e) = failed {
                    let _ = tx.send(Err(e));
                    return;
                }
                if last {
                    return;
                }
            }
        });
        Self {
            rx: Some(rx),
            cur: Vec::new().into_iter(),
            done: false,
            handle: Some(handle),
        }
    }

    /// The next record of this file, or `None` at end of file.
    fn next_record(&mut self) -> Result<Option<Record>> {
        loop {
            if let Some(rec) = self.cur.next() {
                return Ok(Some(rec));
            }
            if self.done {
                return Ok(None);
            }
            // `recv` failing means every sender is gone, i.e. the thread returned: end of file.
            match self.rx.as_ref().map(std::sync::mpsc::Receiver::recv) {
                Some(Ok(Ok(chunk))) => self.cur = chunk.into_iter(),
                Some(Ok(Err(e))) => {
                    self.done = true;
                    return Err(e);
                }
                _ => {
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }
}

impl Drop for RecordStream {
    fn drop(&mut self) {
        // Order matters: releasing the receiver is what unblocks a thread parked in `send`, so it
        // has to happen before the join or an early return from `next_batch` would deadlock.
        self.rx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Streaming reader over a pair of FASTQ files (R1, R2) advanced in lockstep.
///
/// The standard paired-end layout: the same fragment sequenced from both ends, with the two reads
/// at the same ordinal position in the two files. Pairing is therefore POSITIONAL, not by name; the
/// only name check is the length mismatch caught in [`Self::next_batch`]. That matches bwa, which
/// also trusts file order here.
///
/// Each file is parsed on its own thread ([`RecordStream`]); this struct is the consumer that
/// interleaves them and applies the `-K` rule. See [`RecordStream`] for why.
pub struct PairedFastqReader {
    /// The R1 (first-in-pair, SAM FLAG 0x40) file. Read one record per pair.
    r1: RecordStream,
    /// The R2 (second-in-pair, SAM FLAG 0x80) file. Advanced in lockstep with `r1`: the two are
    /// always at the same ordinal record, which is the only thing that makes the pairing correct.
    r2: RecordStream,
}

impl PairedFastqReader {
    /// Open the two mate files.
    ///
    /// # Parameters
    ///
    /// - `p1`: path to the R1 FASTQ, `p2`: path to the R2 FASTQ, both from argv in that order.
    ///   They must hold the same number of records, in the same order; nothing is checked here
    ///   (the length mismatch surfaces later, in [`Self::next_batch`]).
    ///
    /// # Errors
    ///
    /// Never fails here: each file is opened on its reader thread, so a missing or malformed file
    /// is reported by the first [`Self::next_batch`] instead. The `Result` is kept so callers do
    /// not have to change and so the signature can go back to eager opening.
    pub fn from_paths<P: AsRef<std::path::Path>>(p1: P, p2: P) -> Result<Self> {
        Ok(Self {
            r1: RecordStream::spawn(p1.as_ref().to_path_buf()),
            r2: RecordStream::spawn(p2.as_ref().to_path_buf()),
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
                    bases_so_far += mate1.seq().len() + mate2.seq().len();
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
                    mate1.name()
                )));
            };
            // `qname_from_id` already stripped any `/1`/`/2`, so genuine mates compare equal here.
            if mate1.name() != mate2.name() {
                return Err(Error::Fastq(format!(
                    "-p: consecutive reads '{}' and '{}' are not mates. bwa would split these into \
                     a single-end pass (bseq_classify); bwa-mem4 refuses rather than mis-pair them.",
                    mate1.name(), mate2.name()
                )));
            }
            bases_so_far += mate1.seq().len() + mate2.seq().len();
            batch.push((mate1, mate2));
        }
        Ok(batch)
    }
}

/// Both halves of a FASTQ header at once, as BORROWED slices of it.
///
/// The record builder needs the two together and must not allocate for either: it copies them
/// straight into [`Record`]'s single buffer. [`qname_from_id`] and [`comment_from_id`] are thin
/// owned-string wrappers over this, kept for the tests and for callers outside the hot path.
///
/// # Parameters
///
/// - `id`: FASTQ line 1 minus the `@`, exactly as needletail hands it over (name plus comment, no
///   trailing newline). Not required to be valid UTF-8; that is the caller's problem.
///
/// # Returns
///
/// `(name, comment)`. `name` is the header up to the first whitespace with a trailing `/<digit>`
/// trimmed, which is what makes the two mates of a pair share one QNAME. `comment` is everything
/// from the first non-whitespace byte after that field to the end of the line, or `None` when the
/// header is a bare name or ends in whitespace.
fn split_id(id: &[u8]) -> (&[u8], Option<&[u8]>) {
    // One past the last QNAME byte: the first whitespace, or the whole slice for a bare-name
    // header (which is the common case for simulated reads).
    let name_end = id
        .iter()
        .position(u8::is_ascii_whitespace)
        .unwrap_or(id.len());
    let mut name = &id[..name_end];
    // Trailing `/<digit>` (bwa's `trim_readno`). READNO_SUFFIX_LEN is the `/` plus the digit. The
    // length guard means a read literally named `/1` keeps its name rather than becoming empty, and
    // the test is on the LAST two bytes only, so `read/12` is left alone exactly as bwa leaves it.
    if name.len() > READNO_SUFFIX_LEN
        && name[name.len() - 2] == b'/'
        && name[name.len() - 1].is_ascii_digit()
    {
        name = &name[..name.len() - READNO_SUFFIX_LEN];
    }
    // Skip the whole run of whitespace after the name, not just one byte. Nothing left means the
    // header ended in whitespace and so carries no comment.
    let comment = id[name_end..]
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .map(|offset| &id[name_end + offset..]);
    (name, comment)
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
                out.push((
                    rec.name().to_owned(),
                    rec.seq().to_vec(),
                    rec.qual().map(<[u8]>::to_vec),
                ));
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
    use super::{split_id, FastqReader, PairedFastqReader};
    use std::io::Write;

    /// Build a FASTQ of `n` records named `<prefix><i>` with a 4-base sequence, and return its path.
    fn write_fastq(
        dir: &std::path::Path,
        file: &str,
        prefix: &str,
        n: usize,
    ) -> std::path::PathBuf {
        let p = dir.join(file);
        let mut f = std::fs::File::create(&p).unwrap();
        for i in 0..n {
            writeln!(f, "@{prefix}{i}\nACGT\n+\nIIII").unwrap();
        }
        p
    }

    /// The two mate files are parsed on separate threads, so the property that matters is that the
    /// consumer still sees exactly the same interleaving: pair `i` is record `i` of each file, in
    /// order, with the `-K` boundary falling on the same cumulative base count as a one-thread read
    /// would put it. A chunk boundary is crossed on purpose (more records than
    /// `STREAM_CHUNK_RECORDS`) so the refill path is exercised rather than a single chunk.
    #[test]
    fn paired_reader_preserves_order_and_batch_boundaries() {
        let dir = std::env::temp_dir().join(format!("bwa4_pe_order_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = super::STREAM_CHUNK_RECORDS * 2 + 17;
        let p1 = write_fastq(&dir, "r1.fq", "read", n);
        let p2 = write_fastq(&dir, "r2.fq", "read", n);

        let mut r = PairedFastqReader::from_paths(&p1, &p2).unwrap();
        // 8 bases per pair (4 per mate), so a 800-base batch is exactly 100 pairs.
        let mut seen = 0usize;
        loop {
            let batch = r.next_batch(800).unwrap();
            if batch.is_empty() {
                break;
            }
            if seen + batch.len() <= n {
                // Every full batch lands on the same boundary a single-threaded read would pick.
                assert!(
                    batch.len() == 100 || seen + batch.len() == n,
                    "batch {} at {seen}",
                    batch.len()
                );
            }
            for (k, (m1, m2)) in batch.iter().enumerate() {
                assert_eq!(m1.name(), format!("read{}", seen + k));
                assert_eq!(m2.name(), format!("read{}", seen + k));
            }
            seen += batch.len();
        }
        assert_eq!(seen, n);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Files of different lengths must still be refused rather than mis-paired, which is the one
    /// safety property the split could plausibly have broken: the two threads reach end of file
    /// independently, so the check has to live in the consumer.
    #[test]
    fn paired_reader_refuses_unequal_files() {
        let dir = std::env::temp_dir().join(format!("bwa4_pe_uneven_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p1 = write_fastq(&dir, "r1.fq", "read", 10);
        let p2 = write_fastq(&dir, "r2.fq", "read", 7);
        let mut r = PairedFastqReader::from_paths(&p1, &p2).unwrap();
        assert!(r.next_batch(usize::MAX).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A missing file is now reported by the first read rather than by `from_paths`, since opening
    /// moved onto the reader thread. It still has to be an error and not a silent empty run.
    #[test]
    fn paired_reader_reports_a_missing_file() {
        let dir = std::env::temp_dir().join(format!("bwa4_pe_missing_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p1 = write_fastq(&dir, "r1.fq", "read", 4);
        let p2 = dir.join("nope.fq");
        let mut r = PairedFastqReader::from_paths(&p1, &p2).unwrap();
        assert!(r.next_batch(usize::MAX).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A named pipe is not seekable, so the two magic bytes read for format sniffing can never be
    /// re-read. Before the `seekable` split in [`super::open_reader`] the sniff consumed them and
    /// the parser then reopened the path, which on a FIFO yielded a stream already past its header
    /// and therefore ZERO records, silently: `bwa-mem4 mem ref <(zcat r1.gz)` wrote a SAM header
    /// and no alignments, with no error. bwa-mem2 reads that input, so it was a parity gap too.
    #[test]
    #[cfg(unix)]
    fn reads_from_a_non_seekable_fifo() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("bwa4_fifo_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("reads.fq");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .is_ok_and(|s| s.success());
        assert!(made, "mkfifo failed");

        // The writer must run concurrently: opening a FIFO for reading blocks until a writer
        // appears, and opening it for writing blocks until a reader does.
        let w = fifo.clone();
        let writer = std::thread::spawn(move || {
            let mut f = std::fs::File::create(&w).unwrap();
            for i in 0..2500 {
                writeln!(f, "@read{i}\nACGTACGTAC\n+\nIIIIIIIIII").unwrap();
            }
        });

        let mut r = FastqReader::from_path(&fifo).unwrap();
        let mut n = 0usize;
        while let Some(rec) = r.next_record().unwrap() {
            assert_eq!(rec.name(), format!("read{n}"));
            n += 1;
        }
        writer.join().unwrap();
        assert_eq!(n, 2500, "records read from a FIFO");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The header split, on the four shapes that matter: a comment, both read-number suffixes, and
    /// a bare name full of the punctuation simulated reads use.
    #[test]
    fn split_id_strips_comment_and_readno() {
        assert_eq!(
            split_id(b"read1 some comment"),
            (&b"read1"[..], Some(&b"some comment"[..]))
        );
        assert_eq!(split_id(b"read1/1"), (&b"read1"[..], None));
        assert_eq!(
            split_id(b"read1/2 desc"),
            (&b"read1"[..], Some(&b"desc"[..]))
        );
        assert_eq!(
            split_id(b"20:2000000-2200000_50861_51313_0:0:0_0:1:0_0"),
            (&b"20:2000000-2200000_50861_51313_0:0:0_0:1:0_0"[..], None)
        );
        // Trailing whitespace is not a comment, and `/12` is not a read number.
        assert_eq!(split_id(b"read1   "), (&b"read1"[..], None));
        assert_eq!(split_id(b"read/12"), (&b"read/12"[..], None));
    }
}

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
    /// Bases in FASTQ line 2, in the orientation they came off the instrument, held from offset
    /// `name_len` in the form `bases` says. Typically 100-150. Feeds seeding, and (possibly
    /// reverse-complemented and soft-clip-sliced) SAM column 10 (SEQ).
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
    /// How `buf` holds the bases. [`FastqReader::nt4`] picks the coded forms.
    bases: Bases,
}

/// The form a [`Record`]'s bases take in its buffer.
///
/// `Packed` is a memory measure. The aligner holds every record of two batches (the one aligning
/// and the one the reader has ready), and a 150 bp read with a 40-byte name was 340 bytes in the
/// allocator's 384-byte class. Four bases a byte makes it 228 bytes in the 256 class: chr21, 1M
/// pairs, -t4, peak RSS 1030 -> 960 MB, and more at higher `-t`, where batches scale with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bases {
    /// ASCII as read, `seq_len` bytes, not uppercased or validated.
    Ascii,
    /// nt4 codes (A=0, C=1, G=2, T=3, anything else 4), one a byte: a read holding a base that is
    /// not ACGT/acgt, which two bits cannot hold.
    Codes,
    /// nt4 codes 0..=3, four a byte, base `i` in bits `2 * (i % 4)` of byte `i / 4`.
    Packed,
}

/// The nt4 code of an A/C/G/T byte in either case: bits 1 and 2 of the ASCII, XORed. Only defined
/// on those eight bytes; [`is_acgt`] says whether a byte is one of them.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn acgt_code(b: u8) -> u8 {
    ((b >> 1) ^ (b >> 2)) & 3
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn is_acgt(b: u8) -> bool {
    matches!(b & 0xDF, b'A' | b'C' | b'G' | b'T')
}

/// Pack 64 bases four a byte, base `i` in bits `2 * (i % 4)` of byte `i / 4`, and say whether all
/// were A/C/G/T in either case (else `dst` is meaningless). NEON: the code is bits 1 and 2 of the
/// ASCII XORed, and `ld4` de-interleaves so each output byte is one lane.
#[cfg(target_arch = "aarch64")]
#[inline]
fn pack64(src: &[u8; 64], dst: &mut [u8; 16]) -> bool {
    use std::arch::aarch64::*;
    // SAFETY: NEON is baseline on aarch64; reads 64 bytes of `src`, writes 16 of `dst`.
    unsafe {
        let (m3, mdf) = (vdupq_n_u8(3), vdupq_n_u8(0xDF));
        let hit = |v| {
            let u = vandq_u8(v, mdf);
            let ac = vorrq_u8(vceqq_u8(u, vdupq_n_u8(b'A')), vceqq_u8(u, vdupq_n_u8(b'C')));
            vorrq_u8(
                ac,
                vorrq_u8(vceqq_u8(u, vdupq_n_u8(b'G')), vceqq_u8(u, vdupq_n_u8(b'T'))),
            )
        };
        let code = |v| vandq_u8(veorq_u8(vshrq_n_u8::<1>(v), vshrq_n_u8::<2>(v)), m3);
        // Lane `j` of `q.k` is base `4j + k`.
        let q = vld4q_u8(src.as_ptr());
        let all = vandq_u8(vandq_u8(hit(q.0), hit(q.1)), vandq_u8(hit(q.2), hit(q.3)));
        let lo = vorrq_u8(code(q.0), vshlq_n_u8::<2>(code(q.1)));
        let hi = vorrq_u8(vshlq_n_u8::<4>(code(q.2)), vshlq_n_u8::<6>(code(q.3)));
        vst1q_u8(dst.as_mut_ptr(), vorrq_u8(lo, hi));
        vminvq_u8(all) == 0xFF
    }
}

/// [`pack64`] in SSE2: codes byte by byte, then each 32-bit lane's four into its low byte, then two
/// narrowing packs.
#[cfg(target_arch = "x86_64")]
#[inline]
fn pack64(src: &[u8; 64], dst: &mut [u8; 16]) -> bool {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is baseline on x86-64; reads 64 bytes of `src`, writes 16 of `dst`.
    unsafe {
        let (m3, mdf, mff) = (
            _mm_set1_epi8(3),
            _mm_set1_epi8(0xDF_u8 as i8),
            _mm_set1_epi32(0xFF),
        );
        let eq = |u, b: u8| _mm_cmpeq_epi8(u, _mm_set1_epi8(b as i8));
        let mut all = _mm_set1_epi8(-1);
        let mut w = [_mm_setzero_si128(); 4];
        for (k, wk) in w.iter_mut().enumerate() {
            let v = _mm_loadu_si128(src.as_ptr().add(16 * k).cast());
            let u = _mm_and_si128(v, mdf);
            let ac = _mm_or_si128(eq(u, b'A'), eq(u, b'C'));
            all = _mm_and_si128(
                all,
                _mm_or_si128(ac, _mm_or_si128(eq(u, b'G'), eq(u, b'T'))),
            );
            // 16-bit shifts: the bits a byte borrows from its neighbour are masked off.
            let c = _mm_and_si128(
                _mm_xor_si128(_mm_srli_epi16::<1>(v), _mm_srli_epi16::<2>(v)),
                m3,
            );
            // c0 | c1 << 2 | c2 << 4 | c3 << 6 in each 32-bit lane's low byte.
            let lo = _mm_or_si128(c, _mm_srli_epi32::<6>(c));
            let hi = _mm_or_si128(_mm_srli_epi32::<12>(c), _mm_srli_epi32::<18>(c));
            *wk = _mm_and_si128(_mm_or_si128(lo, hi), mff);
        }
        let b = _mm_packus_epi16(_mm_packs_epi32(w[0], w[1]), _mm_packs_epi32(w[2], w[3]));
        _mm_storeu_si128(dst.as_mut_ptr().cast(), b);
        _mm_movemask_epi8(all) == 0xFFFF
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn pack64(src: &[u8; 64], dst: &mut [u8; 16]) -> bool {
    for (d, q) in dst.iter_mut().zip(src.chunks_exact(4)) {
        *d = acgt_code(q[0]) | acgt_code(q[1]) << 2 | acgt_code(q[2]) << 4 | acgt_code(q[3]) << 6;
    }
    src.iter().all(|&b| is_acgt(b))
}

/// Write the 64 codes 16 packed bytes hold to `dst`. NEON: an interleaving `st4` puts lane `j` of
/// shift `k` at byte `4j + k`.
///
/// # Safety
///
/// `dst` must be valid for 64 bytes of writes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn unpack16(src: &[u8; 16], dst: *mut u8) {
    use std::arch::aarch64::*;
    // SAFETY: NEON is baseline on aarch64; reads 16 bytes of `src`, writes the 64 the caller vouches for.
    unsafe {
        let (v, m3) = (vld1q_u8(src.as_ptr()), vdupq_n_u8(3));
        let codes = uint8x16x4_t(
            vandq_u8(v, m3),
            vandq_u8(vshrq_n_u8::<2>(v), m3),
            vandq_u8(vshrq_n_u8::<4>(v), m3),
            vshrq_n_u8::<6>(v),
        );
        vst4q_u8(dst, codes);
    }
}

/// [`unpack16`] in SSE2: byte then 16-bit unpacks do the interleave.
///
/// # Safety
///
/// `dst` must be valid for 64 bytes of writes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn unpack16(src: &[u8; 16], dst: *mut u8) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is baseline on x86-64; reads 16 bytes of `src`, writes the 64 the caller vouches for.
    unsafe {
        let (v, m3) = (_mm_loadu_si128(src.as_ptr().cast()), _mm_set1_epi8(3));
        let c0 = _mm_and_si128(v, m3);
        let c1 = _mm_and_si128(_mm_srli_epi16::<2>(v), m3);
        let c2 = _mm_and_si128(_mm_srli_epi16::<4>(v), m3);
        let c3 = _mm_and_si128(_mm_srli_epi16::<6>(v), m3);
        let (lo01, hi01) = (_mm_unpacklo_epi8(c0, c1), _mm_unpackhi_epi8(c0, c1));
        let (lo23, hi23) = (_mm_unpacklo_epi8(c2, c3), _mm_unpackhi_epi8(c2, c3));
        let p = dst.cast::<__m128i>();
        _mm_storeu_si128(p, _mm_unpacklo_epi16(lo01, lo23));
        _mm_storeu_si128(p.add(1), _mm_unpackhi_epi16(lo01, lo23));
        _mm_storeu_si128(p.add(2), _mm_unpacklo_epi16(hi01, hi23));
        _mm_storeu_si128(p.add(3), _mm_unpackhi_epi16(hi01, hi23));
    }
}

/// # Safety
///
/// `dst` must be valid for 64 bytes of writes.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
unsafe fn unpack16(src: &[u8; 16], dst: *mut u8) {
    for (j, &b) in src.iter().enumerate() {
        // SAFETY: bytes 4j..4j + 4 of the 64 the caller vouches for.
        unsafe {
            dst.add(4 * j).cast::<[u8; 4]>().write_unaligned([
                b & 3,
                b >> 2 & 3,
                b >> 4 & 3,
                b >> 6,
            ])
        };
    }
}

/// Append `seq` packed four codes a byte, and say whether every byte of it is A/C/G/T in either
/// case. On `false` the bytes appended are meaningless and the caller drops them. A short last
/// block is padded with `A`, so it takes the SIMD path too.
fn pack_acgt(seq: &[u8], out: &mut Vec<u8>) -> bool {
    let mut ok = true;
    let mut packed = [0u8; 16];
    for block in seq.chunks(64) {
        let mut pad = [b'A'; 64];
        let src: &[u8; 64] = match block.try_into() {
            Ok(full) => full,
            Err(_) => {
                pad[..block.len()].copy_from_slice(block);
                &pad
            }
        };
        ok &= pack64(src, &mut packed);
        out.extend_from_slice(&packed[..block.len().div_ceil(4)]);
    }
    ok
}

/// Append the first `n` codes `packed` holds (four a byte, low bits first; `packed` may run past
/// them). Every read is unpacked three times (seeding, mate rescue, SAM), so this is SIMD all the
/// way, straight into `out`'s spare capacity: a short last block is padded.
fn unpack_codes(packed: &[u8], n: usize, out: &mut Vec<u8>) {
    assert!(n <= 4 * packed.len());
    let at = out.len();
    out.reserve(64 * packed.len().div_ceil(16));
    let mut pad = [0u8; 16];
    for (k, block) in packed.chunks(16).enumerate() {
        let src: &[u8; 16] = match block.try_into() {
            Ok(full) => full,
            Err(_) => {
                pad[..block.len()].copy_from_slice(block);
                &pad
            }
        };
        // SAFETY: the reserve left 64 bytes of spare capacity per block past `at`.
        unsafe { unpack16(src, out.as_mut_ptr().add(at + 64 * k)) };
    }
    // SAFETY: the blocks wrote every byte below `at + 4 * packed.len()`, which covers `at + n`.
    unsafe { out.set_len(at + n) };
}

impl Record {
    /// SAM column 1 (QNAME), as raw bytes: a FASTQ header need not be valid UTF-8, and bwa copies it
    /// through unchanged, so converting it here would change the output.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.buf[..self.name_len as usize]
    }

    /// FASTQ line 2 as ASCII: a record from a reader without [`FastqReader::nt4`].
    ///
    /// # Panics
    ///
    /// On a record whose bases are coded; [`Record::codes_into`] reads those.
    #[inline]
    #[must_use]
    pub fn seq(&self) -> &[u8] {
        assert!(
            self.bases == Bases::Ascii,
            "coded bases: read them with Record::codes_into"
        );
        let start = self.name_len as usize;
        &self.buf[start..start + self.seq_len as usize]
    }

    /// Number of bases.
    #[inline]
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len as usize
    }

    /// Append the bases as nt4 codes (A=0, C=1, G=2, T=3, anything else 4), bwa's encoding.
    pub fn codes_into(&self, out: &mut Vec<u8>) {
        let n = self.seq_len as usize;
        let start = self.name_len as usize;
        match self.bases {
            Bases::Ascii => out.extend(
                self.buf[start..start + n]
                    .iter()
                    .map(|&b| bwa_core::dna::nt4(b)),
            ),
            Bases::Codes => out.extend_from_slice(&self.buf[start..start + n]),
            Bases::Packed => {
                // Whole 16-byte blocks where the buffer has them (the quality string follows), so
                // the last block needs no padding copy; the codes past `n` it yields are dropped.
                let end = (start + n.div_ceil(4).next_multiple_of(16)).min(self.buf.len());
                unpack_codes(&self.buf[start..end], n, out);
            }
        }
    }

    /// Bytes of `buf` the bases take.
    #[inline]
    fn seq_bytes(&self) -> usize {
        match self.bases {
            Bases::Packed => (self.seq_len as usize).div_ceil(4),
            Bases::Ascii | Bases::Codes => self.seq_len as usize,
        }
    }

    /// FASTQ line 4: Phred+33 quality, one byte per base, or `None` for FASTA input.
    #[inline]
    #[must_use]
    pub fn qual(&self) -> Option<&[u8]> {
        if !self.has_qual {
            return None;
        }
        let start = self.name_len as usize + self.seq_bytes();
        Some(&self.buf[start..start + self.seq_len as usize])
    }

    /// The kseq `comment`, or `None` for a bare-name header. Raw bytes, as [`Record::name`].
    #[inline]
    #[must_use]
    pub fn comment(&self) -> Option<&[u8]> {
        if self.comment_len == 0 {
            return None;
        }
        Some(&self.buf[self.buf.len() - self.comment_len as usize..])
    }
}

/// Streaming FASTQ reader.
pub struct FastqReader {
    /// needletail's parser, owning the file handle, its buffer and (for `.gz` input) the
    /// decompressor. Boxed because the concrete parser type depends on the compression needletail
    /// sniffed from the file's magic bytes. Advanced only by [`FastqReader::next_record`]; holds
    /// all the reader's position state, so this struct has no offset of its own.
    /// `None` for an empty input: it yields no records.
    inner: Option<Box<dyn FastxReader>>,
    /// Whether records come out with their bases nt4-coded (see [`FastqReader::nt4`]).
    nt4: bool,
}

/// Bytes peeked from the head of an input to decide whether it needs [`UnwrapFastq`].
///
/// 64 KiB is about 200 records of 150 bp, which is ample evidence for a file that is homogeneous --
/// and real FASTQ is: a writer that wraps wraps everything. It is also small enough to disappear
/// into the noise. At one megabyte the peek was measurable: `wait_read` went from 20 ms to 34 ms on
/// a two-file paired run, because the first batch cannot start until both peeks are done, and total
/// CPU rose 0.7%. At 64 KiB both are back in the noise.
const WRAP_PEEK: usize = 64 << 10;

/// Whether the input starting at `head` is FASTQ that needs [`UnwrapFastq`] in front of it.
///
/// TRUE ONLY ON PROOF. The answer is `true` only when the first record parses cleanly under kseq's
/// rules AND takes more than four lines; everything else -- FASTA, four-line FASTQ, junk, and
/// anything malformed -- is `false` and goes to needletail untouched.
///
/// That asymmetry is the point, and it was learned the hard way. An earlier version answered "is
/// this already four-line?" and so returned "needs unwrapping" for MALFORMED files too, since a
/// quality string of the wrong length looks exactly like a wrapped one from the third line. The
/// unwrapper then failed on them and needletail reported `Failed to read the first two bytes. Is
/// the file empty?` for a file that was neither empty nor unreadable, in place of the precise
/// `Sequence length is 150 but quality length is 100` it gives when it parses the bytes itself.
/// Diagnosing broken input is needletail's job and it is good at it; this function's only job is to
/// recognise the one shape it cannot handle.
///
/// # Parameters
///
/// - `head`: the first bytes of the input.
/// - `complete`: true when `head` is the WHOLE input rather than a prefix of it. It decides what a
///   record running off the end means: in a complete file that is malformed input, and in a prefix
///   it is just the peek ending, so the two cases must not be conflated.
///
/// # Returns
///
/// Whether to insert [`UnwrapFastq`].
fn needs_unwrapping(head: &[u8], complete: bool) -> bool {
    // Tolerate CRLF: a trailing carriage return would otherwise hide the `+` and make every record
    // look wrapped.
    let mut lines = head
        .split(|&b| b == b'\n')
        .map(|l| l.strip_suffix(b"\r").unwrap_or(l));
    let Some(first) = lines.find(|l| !l.is_empty()) else {
        return false; // nothing to judge
    };
    if first[0] != b'@' {
        return false; // FASTA, whose wrapping needletail handles, or not sequence data at all
    }
    // Sequence lines, up to the `+` separator. More than one of them is the wrapped shape.
    let (mut seq_len, mut seq_lines) = (0usize, 0usize);
    loop {
        let Some(line) = lines.next() else {
            // Ran out before the separator. In a complete file that is malformed; in a prefix it
            // only means the peek ended, and a sequence already spanning two lines is wrapped.
            return !complete && seq_lines > 1;
        };
        if line.first() == Some(&b'+') {
            break;
        }
        // An empty piece is the trailing newline's split artifact as often as it is a blank line, so
        // it is treated as the input ending here: malformed in a complete file, just the peek
        // running out in a prefix. Reading it as a blank line inside a record instead made every
        // wrapped file whose peek ended on a newline look unwrapped.
        if line.is_empty() {
            return !complete && seq_lines > 1;
        }
        seq_len += line.len();
        seq_lines += 1;
    }
    if seq_lines == 0 {
        return false; // `@name` followed immediately by `+`
    }
    // Quality lines, until quality is as long as the sequence. kseq's rule, and the reason a
    // quality line may legally start with `@`.
    let mut qual_lines = 0usize;
    let mut qual_len = 0usize;
    while qual_len < seq_len {
        let Some(line) = lines.next() else {
            return !complete && qual_lines > 1;
        };
        if line.is_empty() {
            return !complete && qual_lines > 1;
        }
        qual_len += line.len();
        qual_lines += 1;
    }
    // Overshooting means the quality string is longer than the sequence, which is malformed rather
    // than wrapped, and needletail says so far better than this could.
    if qual_len != seq_len {
        return false;
    }
    seq_lines > 1 || qual_lines > 1
}

/// A `Read` that rewrites multi-line FASTQ into the four-line form, and passes everything else
/// through untouched.
///
/// WHY IT EXISTS. bwa reads FASTQ with klib's `kseq`, which accumulates sequence lines until it
/// meets a line starting with `+` and then accumulates quality lines until quality is as long as
/// sequence. So `bwa-mem2` aligns a wrapped FASTQ perfectly happily, and its output is
/// byte-identical to the same reads unwrapped (measured: 2000 records, identical). needletail's
/// FASTQ parser requires exactly four lines per record and errors on anything else, so until this
/// existed `bwa-mem4` produced ZERO records and a parse error on a file bwa aligns.
///
/// It is placed in front of the parser only when [`needs_unwrapping`] proves the input needs it, so
/// the ordinary case pays one megabyte of peek at open and nothing per byte after that.
///
/// KSEQ'S QUALITY RULE IS THE POINT. Quality is accumulated until it is as long as the sequence,
/// NOT until a line starting with `@` is seen -- because `@` is a legal quality character and a
/// quality line may well start with one. Terminating on `@` is the classic FASTQ parsing bug, and
/// this reproduces bwa's rule rather than inventing one.
struct UnwrapFastq<R: std::io::BufRead> {
    /// The wrapped input, read a line at a time.
    inner: R,
    /// Normalised bytes not yet handed to the caller, and how far into them we are. Holds at most
    /// one record.
    out: Vec<u8>,
    /// Read cursor into `out`.
    pos: usize,
    /// Set once `inner` reports end of input, so `read` stops asking for more.
    done: bool,
}

impl<R: std::io::BufRead> UnwrapFastq<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            out: Vec::with_capacity(1024),
            pos: 0,
            done: false,
        }
    }

    /// Read one line, stripped of its newline (and of a CR before it). `Ok(None)` at end of input.
    fn line(&mut self, buf: &mut Vec<u8>) -> std::io::Result<bool> {
        buf.clear();
        if self.inner.read_until(b'\n', buf)? == 0 {
            return Ok(false);
        }
        if buf.last() == Some(&b'\n') {
            buf.pop();
        }
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
        Ok(true)
    }

    /// Normalise the next record into `out`. `Ok(false)` at end of input.
    fn fill(&mut self) -> std::io::Result<bool> {
        self.out.clear();
        self.pos = 0;
        let mut buf = Vec::with_capacity(256);
        // The header, skipping blank lines before it.
        loop {
            if !self.line(&mut buf)? {
                return Ok(false);
            }
            if !buf.is_empty() {
                break;
            }
        }
        if buf.first() != Some(&b'@') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected a FASTQ header starting with '@', found {:?}",
                    String::from_utf8_lossy(&buf[..buf.len().min(32)])
                ),
            ));
        }
        self.out.extend_from_slice(&buf);
        self.out.push(b'\n');
        // Sequence lines, up to the '+' separator.
        // `seq_len` is what the quality accumulation below has to match, which is kseq's rule.
        let mut seq_len = 0usize;
        loop {
            if !self.line(&mut buf)? {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "FASTQ input ended inside a record's sequence",
                ));
            }
            if buf.first() == Some(&b'+') {
                break;
            }
            seq_len += buf.len();
            self.out.extend_from_slice(&buf);
        }
        // The separator is emitted bare: its optional repeat of the name is not data, and bwa does
        // not read it either.
        self.out.extend_from_slice(b"\n+\n");
        // Quality lines, until quality is as long as sequence. See the note above on why the test is
        // a length and not a leading character.
        let mut qual_len = 0usize;
        while qual_len < seq_len {
            if !self.line(&mut buf)? {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "FASTQ input ended inside a record's quality string",
                ));
            }
            qual_len += buf.len();
            self.out.extend_from_slice(&buf);
        }
        self.out.push(b'\n');
        Ok(true)
    }
}

impl<R: std::io::BufRead> std::io::Read for UnwrapFastq<R> {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        while self.pos == self.out.len() {
            if self.done {
                return Ok(0);
            }
            if !self.fill()? {
                self.done = true;
                return Ok(0);
            }
        }
        let n = (self.out.len() - self.pos).min(dst.len());
        dst[..n].copy_from_slice(&self.out[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Put [`UnwrapFastq`] in front of `r` if, and only if, its first [`WRAP_PEEK`] bytes are FASTQ that
/// is not already in four-line form.
///
/// The peeked bytes are chained back in front of the stream either way, so nothing is consumed: the
/// same trick `open_reader` already uses to keep the two magic bytes it sniffed.
///
/// # Parameters
///
/// - `r`: the input, decompressed if it was compressed. Taken by value and returned inside the
///   result, so callers hand over ownership.
///
/// # Returns
///
/// A `Read` yielding either exactly `r`'s bytes or its records normalised to four lines each.
fn unwrap_multiline_fastq<R: std::io::Read + Send + 'static>(
    mut r: R,
) -> Result<Box<dyn std::io::Read + Send>> {
    use std::io::Read;
    // `take` bounds the peek; a short input simply yields fewer bytes.
    let mut head = Vec::with_capacity(WRAP_PEEK);
    (&mut r)
        .take(WRAP_PEEK as u64)
        .read_to_end(&mut head)
        .map_err(|e| Error::Fastq(e.to_string()))?;
    // `head` shorter than the cap means `read_to_end` hit the end of the input, so this IS the whole
    // file; that distinction is what lets a record running off the end be read as malformed input
    // rather than as the peek ending. Decided before `head` is moved into the cursor, so the peek is
    // not copied.
    let complete = head.len() < WRAP_PEEK;
    let wrapped = needs_unwrapping(&head, complete);
    let rest = std::io::Read::chain(std::io::Cursor::new(head), r);
    if !wrapped {
        return Ok(Box::new(rest));
    }
    Ok(Box::new(UnwrapFastq::new(
        std::io::BufReader::with_capacity(1 << 16, rest),
    )))
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
/// needletail's parser, or `None` for an input with no bytes at all. bwa-mem2 treats an empty FASTQ
/// (or an empty `.gz`) as zero reads, writes the header and exits 0; needletail refuses it with
/// `EmptyFile`, which would fail a pipeline that feeds us one empty chunk among many.
fn parsed(
    r: std::result::Result<Box<dyn FastxReader>, needletail::errors::ParseError>,
) -> Result<Option<Box<dyn FastxReader>>> {
    match r {
        Ok(r) => Ok(Some(r)),
        Err(e) if e.kind == needletail::errors::ParseErrorKind::EmptyFile => Ok(None),
        Err(e) => Err(Error::Fastq(e.to_string())),
    }
}

fn open_reader(path: &std::path::Path) -> Result<Option<Box<dyn FastxReader>>> {
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
        let mut buf = Vec::with_capacity(2);
        // `take(2).read_to_end`, not one `read`: a pipe can hand over a single byte first, and
        // that byte must still go back in front of the stream below.
        (&mut opened)
            .take(2)
            .read_to_end(&mut buf)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        buf
    };
    let head_bytes = magic.clone();
    let magic: Option<[u8; 2]> = magic.try_into().ok();
    let is_gzip = magic == Some([0x1f, 0x8b]);
    let looks_like_text = magic.is_some_and(|m| m[0] == b'@' || m[0] == b'>');

    // Non-seekable: hand needletail the bytes already taken, chained back in front of the rest, and
    // let it sniff compression itself. Sequential by construction, which is also what any parallel
    // decoder would have to fall back to on a stream it cannot seek in.
    if !seekable {
        let head = std::io::Cursor::new(head_bytes);
        let stream = std::io::Read::chain(head, opened);
        let stream = unwrap_multiline_fastq(stream)?;
        return parsed(needletail::parse_fastx_reader(stream));
    }

    // Gzipped input, rapidgzip path: it decodes ONE stream on `gzip_threads` threads and hands back
    // a `Read`, so needletail parses plain text exactly as it would from an uncompressed file. A
    // pipe or FIFO is not seekable and the crate falls back to sequential decoding on its own, so
    // there is no input to special-case here. See `gzip_threads` for the budget.
    #[cfg(feature = "parallel-gzip")]
    if is_gzip {
        // 256 KiB decoded chunks, one queued per thread: the default 4 MiB with two spare queued
        // cost 50 MB of a -t1 run (chr21 gzip: 443 -> 393 MB), and the first batch's wait is
        // unchanged at -t8 and -t16 (78 and 158 ms).
        let dec = rapidgzip_core::Decoder::builder()
            .decoder_threads(gzip_threads())
            .decoded_chunk_size(256 << 10)
            .in_flight_chunks(gzip_threads() + 1)
            .build()
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        let stream = dec
            .open(path)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        let stream = unwrap_multiline_fastq(stream)?;
        return parsed(needletail::parse_fastx_reader(stream));
    }

    // Gzipped input, single-thread fallback: inflate on a thread of its own and hand needletail the
    // decompressed bytes, so that inflating block N+1 overlaps parsing block N instead of following
    // it. See [`spawn_inflate`] for the measurement that motivates it.
    #[cfg(all(feature = "fast-gzip", not(feature = "parallel-gzip")))]
    if is_gzip {
        let blocks = unwrap_multiline_fastq(spawn_inflate(path)?)?;
        return parsed(needletail::parse_fastx_reader(blocks));
    }

    // PLAIN TEXT, AND IT GETS THE SAME READ-AHEAD THREAD THE COMPRESSED PATHS GET.
    //
    // `parse_fastx_file` opens the file itself and parses off its own buffer, so reading block N+1
    // and parsing block N are the same thread's work, one after the other. Every compressed path
    // above already avoids that: `rapidgzip` decodes on its own threads, and `spawn_inflate` exists
    // precisely so "inflating block N+1 overlaps parsing block N".
    //
    // The asymmetry is measurable and it runs against us. On a four-core runner with the same 5.45M
    // real pairs, the fork leads by 2.3% when the input is plain FASTQ (3.5 GB) and trails by 3.1%
    // when it is gzipped (900 MB). Same reads, same binaries; the input format flips the verdict,
    // and the difference is that our compressed path overlaps its I/O and our plain path does not.
    //
    // Byte-identical by construction: the parser sees the same bytes in the same order, only
    // delivered by a different thread.
    if !is_gzip && looks_like_text {
        let blocks = unwrap_multiline_fastq(spawn_plain(path)?)?;
        return parsed(needletail::parse_fastx_reader(blocks));
    }

    if is_gzip || looks_like_text || cfg!(not(feature = "multi-format")) {
        return parsed(parse_fastx_file(path));
    }

    #[cfg(feature = "multi-format")]
    {
        // niffler picks the decompressor from the same magic bytes and hands back a `Read`;
        // needletail then parses the decompressed stream exactly as it would a plain file.
        let (reader, _format) = niffler::send::from_path(path)
            .map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
        parsed(needletail::parse_fastx_reader(reader))
    }
    #[cfg(not(feature = "multi-format"))]
    unreachable!("the branch above returns when the feature is off")
}

/// Threads for one gzipped file's decoder: 1, or `BWA4_GZIP_THREADS`.
///
/// One thread inflates a 150 bp FASTQ at about 1 GB/s, and only the FIRST batch ever waits on it
/// (behind it the aligner is the bottleneck). Every extra thread keeps decoded chunks in flight for
/// the whole run. chr21 sim pairs, gzip -1, M4, `-t/2` threads a file against one: -t4 +170 MB and
/// a slower first batch (wait 67 vs 34 ms), -t8 +60 MB for the same wait, -t16 +200 MB to save 47
/// ms once. A value that does not parse, or 0, is ignored rather than clamped silently, so a typo
/// in a benchmark script cannot quietly change what is being measured.
#[cfg(feature = "parallel-gzip")]
fn gzip_threads() -> usize {
    std::env::var("BWA4_GZIP_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

/// Decompressed bytes handed over per channel message. 1 MiB is about 3 000 reads at 150 bp, so
/// the channel is touched ~360 times for a 1 M-pair mate file. Behind a slow aligner every queue
/// fills, so a file holds `INFLATE_DEPTH + 2` blocks: at 4 MiB and depth 3 that was 40 MiB per
/// paired run, more than a whole `-t1` batch of records.
const INFLATE_BLOCK: usize = 1 << 20;

/// Blocks the inflater may run ahead. One in flight plus one being parsed is the double buffering
/// that makes the overlap work.
const INFLATE_DEPTH: usize = 2;

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

/// The plain-text twin of [`spawn_inflate`]: read the file on a thread of its own and hand the
/// parser the bytes, so that reading block N+1 overlaps parsing block N.
///
/// Same block size, same queue depth and the same `InflatedBlocks` consumer as the gzip path, for
/// the same reason: the point is the overlap, not the decompression. See the call site for the
/// measurement that motivates it.
fn spawn_plain(path: &std::path::Path) -> Result<InflatedBlocks> {
    use std::io::Read;
    let file =
        std::fs::File::open(path).map_err(|e| Error::Fastq(format!("{}: {e}", path.display())))?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::io::Result<Vec<u8>>>(INFLATE_DEPTH);
    std::thread::spawn(move || {
        let mut src = std::io::BufReader::with_capacity(1 << 20, file);
        loop {
            let mut block = vec![0u8; INFLATE_BLOCK];
            // Fill each block completely unless the file ends, so the parser is handed whole blocks
            // rather than whatever length a single `read` happened to return.
            let mut filled = 0;
            let mut failed = None;
            while filled < INFLATE_BLOCK {
                match src.read(&mut block[filled..]) {
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
            nt4: false,
        })
    }

    /// Hand out records whose bases are already nt4 codes, packed four a byte when the read is all
    /// A/C/G/T: [`Record::codes_into`] reads them, [`Record::seq`] no longer can. bwa codes them the
    /// same way before aligning and prints SAM SEQ back out of the codes, so nothing is lost.
    #[must_use]
    pub fn nt4(mut self) -> Self {
        self.nt4 = true;
        self
    }

    /// Pull the next read, or `None` at EOF.
    ///
    /// # Returns
    ///
    /// `Ok(None)` exactly at end of file (not an error: it is the stop condition), `Ok(Some(rec))`
    /// with all four FASTQ lines already split into [`Record`] fields, or [`Error::Fastq`] on a
    /// malformed record.
    pub fn next_record(&mut self) -> Result<Option<Record>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        match inner.next() {
            None => Ok(None),
            Some(rec) => {
                // One parsed record, borrowing needletail's internal buffer: everything taken from
                // it below must be copied out before the next `self.inner.next()`.
                let rec = rec.map_err(|e| Error::Fastq(e.to_string()))?;
                // `rec.id()` is FASTQ line 1 WITHOUT the leading `@`, i.e. name plus any comment.
                // Split into the two halves: the first becomes SAM QNAME, the second the `-C`
                // trailer. Both are still borrowed from needletail's buffer at this point.
                let (name, comment) = split_id(rec.id());
                // FASTQ line 2 (bases) and line 4 (Phred+33). `qual` is `None` for FASTA input,
                // where lines 3 and 4 do not exist.
                let seq = rec.seq();
                let qual = rec.qual();
                // The copy out of needletail's buffer is not optional (it reuses that buffer for the
                // next record), but it is ONE allocation sized exactly right, not three. See the
                // type's own docs for what that is worth.
                let comment_len = comment.map_or(0, <[u8]>::len);
                let seq_bytes = if self.nt4 {
                    seq.len().div_ceil(4)
                } else {
                    seq.len()
                };
                let mut buf = Vec::with_capacity(
                    name.len() + seq_bytes + qual.map_or(0, <[u8]>::len) + comment_len,
                );
                buf.extend_from_slice(name);
                let bases = if !self.nt4 {
                    buf.extend_from_slice(&seq);
                    Bases::Ascii
                } else if pack_acgt(&seq, &mut buf) {
                    Bases::Packed
                } else {
                    // A base two bits cannot hold (N, IUPAC, junk): one code a byte instead.
                    buf.truncate(name.len());
                    buf.reserve_exact(seq.len() - seq_bytes);
                    buf.extend_from_slice(&seq);
                    bwa_core::dna::nt4_in_place(&mut buf[name.len()..]);
                    Bases::Codes
                };
                if let Some(q) = qual {
                    buf.extend_from_slice(q);
                }
                if let Some(c) = comment {
                    buf.extend_from_slice(c);
                }
                Ok(Some(Record {
                    name_len: name.len() as u32,
                    seq_len: seq.len() as u32,
                    comment_len: comment_len as u32,
                    has_qual: qual.is_some(),
                    bases,
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
                    bases_so_far += rec.seq_len();
                    batch.push(rec);
                }
            }
        }
        Ok(batch)
    }
}

/// How many records one background mate-reader hands over per channel message.
///
/// Big enough that the channel is touched once per two thousand reads instead of once per read
/// (about 0.8 MB of records at 150 bp), small enough that the two streams stay within a few
/// milliseconds of each other. Behind a slow aligner each file holds three chunks: 8192 records a
/// chunk was 20 MB of a paired run's resident set, 11 MB more than now at `-t1`.
const STREAM_CHUNK_RECORDS: usize = 2048;

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
    fn spawn(path: std::path::PathBuf, nt4: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<Record>>>(STREAM_CHUNK_DEPTH);
        let handle = std::thread::spawn(move || {
            let mut reader = match FastqReader::from_path(&path) {
                Ok(r) if nt4 => r.nt4(),
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
/// at the same ordinal position in the two files. Pairing is POSITIONAL, and [`Self::next_batch`]
/// then refuses a pair whose names differ, as bwa does (`mem_sam_pe`).
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
    /// - `nt4`: code the bases as [`FastqReader::nt4`] does.
    ///
    /// # Errors
    ///
    /// Never fails here: each file is opened on its reader thread, so a missing or malformed file
    /// is reported by the first [`Self::next_batch`] instead. The `Result` is kept so callers do
    /// not have to change and so the signature can go back to eager opening.
    pub fn from_paths<P: AsRef<std::path::Path>>(p1: P, p2: P, nt4: bool) -> Result<Self> {
        Ok(Self {
            r1: RecordStream::spawn(p1.as_ref().to_path_buf(), nt4),
            r2: RecordStream::spawn(p2.as_ref().to_path_buf(), nt4),
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
    /// file runs out before the other, or if the two reads of a pair have different names.
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
                // They are paired by POSITION, so a lost record shows up only here, as a name
                // mismatch. Same refusal and wording as bwa's `mem_sam_pe`.
                (Some(mate1), Some(mate2)) => {
                    if mate1.name() != mate2.name() {
                        return Err(Error::Fastq(format!(
                            "paired reads have different names: \"{}\", \"{}\"",
                            String::from_utf8_lossy(mate1.name()),
                            String::from_utf8_lossy(mate2.name())
                        )));
                    }
                    bases_so_far += mate1.seq_len() + mate2.seq_len();
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
    /// - `nt4`: code the bases as [`FastqReader::nt4`] does.
    pub fn from_path<P: AsRef<std::path::Path>>(p: P, nt4: bool) -> Result<Self> {
        let inner = FastqReader::from_path(p)?;
        Ok(Self {
            inner: if nt4 { inner.nt4() } else { inner },
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
                    String::from_utf8_lossy(mate1.name())
                )));
            };
            // `qname_from_id` already stripped any `/1`/`/2`, so genuine mates compare equal here.
            if mate1.name() != mate2.name() {
                return Err(Error::Fastq(format!(
                    "-p: consecutive reads '{}' and '{}' are not mates. bwa would split these into \
                     a single-end pass (bseq_classify); bwa-mem4 refuses rather than mis-pair them.",
                    String::from_utf8_lossy(mate1.name()),
                    String::from_utf8_lossy(mate2.name())
                )));
            }
            bases_so_far += mate1.seq_len() + mate2.seq_len();
            batch.push((mate1, mate2));
        }
        Ok(batch)
    }
}

/// Index of the first byte `<= b' '` in `b` (every ASCII whitespace byte is one), or `b.len()`.
fn first_le_space(b: &[u8]) -> usize {
    const ONES: u64 = 0x0101_0101_0101_0101;
    let (chunks, rest) = b.as_chunks::<8>();
    for (k, c) in chunks.iter().enumerate() {
        let x = u64::from_le_bytes(*c);
        // The high bit of each byte below 0x21, plus maybe some above a flagged one (borrows only
        // run upward), so the lowest flag is exact.
        let m = x.wrapping_sub(ONES * 0x21) & !x & (ONES * 0x80);
        if m != 0 {
            return 8 * k + (m.trailing_zeros() / 8) as usize;
        }
    }
    b.len() - rest.len() + rest.iter().position(|&c| c <= b' ').unwrap_or(rest.len())
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
    // Eight bytes a step to each byte at or below a space, then the exact test there: the byte loop
    // was a sixth of the parse on GIAB headers (~37 printable bytes before the first space).
    let mut name_end = 0;
    loop {
        name_end += first_le_space(&id[name_end..]);
        if name_end == id.len() || id[name_end].is_ascii_whitespace() {
            break;
        }
        name_end += 1;
    }
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
    use super::{
        first_le_space, needs_unwrapping, pack_acgt, split_id, unpack_codes, FastqReader,
        PairedFastqReader, UnwrapFastq,
    };
    use std::io::Write;

    /// Packing round-trips to bwa's nt4 codes at every length across the SIMD block edges (64
    /// bases packing, 16 packed bytes unpacking), and a single non-ACGT byte anywhere, in a SIMD
    /// block or in the scalar tail, sends the read to one code a byte.
    #[test]
    fn packed_bases_round_trip_and_refuse_non_acgt() {
        let alphabet = b"ACGTacgtTGCAtgca";
        for n in 0..=200 {
            let seq: Vec<u8> = (0..n).map(|i| alphabet[(i * 7 + i / 5) % 16]).collect();
            let mut packed = vec![0xAA];
            assert!(pack_acgt(&seq, &mut packed), "n={n}");
            assert_eq!(packed.len(), 1 + n.div_ceil(4), "n={n}");
            let mut codes = vec![9];
            unpack_codes(&packed[1..], n, &mut codes);
            let want: Vec<u8> = seq.iter().map(|&b| bwa_core::dna::nt4(b)).collect();
            assert_eq!(codes[1..], want[..], "n={n}");
            // Bytes past the bases (the quality string, in a record) decode to codes past `n`,
            // which must be dropped.
            let mut slack = packed[1..].to_vec();
            slack.extend([0xFF; 20]);
            let mut codes = Vec::new();
            unpack_codes(&slack, n, &mut codes);
            assert_eq!(codes, want, "n={n} with slack");
            for bad in [b'N', b'n', b'R', b'.', 0] {
                for at in [0, n / 2, n.saturating_sub(1)]
                    .into_iter()
                    .filter(|&a| a < n)
                {
                    let mut s = seq.clone();
                    s[at] = bad;
                    assert!(!pack_acgt(&s, &mut Vec::new()), "n={n} at={at} byte={bad}");
                }
            }
        }
    }

    /// An empty FASTQ, plain or gzipped, is zero reads (bwa-mem2 exits 0 on it), not an error.
    #[test]
    fn empty_input_is_zero_records() {
        let dir = std::env::temp_dir().join(format!("bwa4_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("e.fq");
        std::fs::write(&plain, b"").unwrap();
        // A valid gzip stream of zero bytes (`gzip -n < /dev/null`): header, empty block, trailer.
        let gz = dir.join("e.fq.gz");
        std::fs::write(
            &gz,
            [
                0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        for p in [&plain, &gz] {
            let mut r = FastqReader::from_path(p).unwrap();
            assert!(r.next_record().unwrap().is_none(), "{}", p.display());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// bwa copies a header's bytes through, so a byte that is not UTF-8 survives unchanged in the
    /// name and the comment, and a clean header comes through as is.
    #[test]
    fn non_utf8_header_keeps_its_bytes() {
        let dir = std::env::temp_dir().join(format!("bwa4_lossy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("l.fq");
        std::fs::write(
            &p,
            b"@r\xffx c\xfe\nACGT\n+\nIIII\n@ok tag\nACGT\n+\nIIII\n",
        )
        .unwrap();
        let mut r = FastqReader::from_path(&p).unwrap();
        let rec = r.next_record().unwrap().unwrap();
        assert_eq!(rec.name(), b"r\xffx");
        assert_eq!(rec.comment(), Some(&b"c\xfe"[..]));
        let rec = r.next_record().unwrap().unwrap();
        assert_eq!((rec.name(), rec.comment()), (&b"ok"[..], Some(&b"tag"[..])));
        std::fs::remove_dir_all(&dir).unwrap();
    }

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

        let mut r = PairedFastqReader::from_paths(&p1, &p2, false).unwrap();
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
                assert_eq!(m1.name(), format!("read{}", seen + k).as_bytes());
                assert_eq!(m2.name(), format!("read{}", seen + k).as_bytes());
            }
            seen += batch.len();
        }
        assert_eq!(seen, n);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Files of different lengths must still be refused rather than mis-paired, which is the one
    /// safety property the split could plausibly have broken: the two threads reach end of file
    /// independently, so the check has to live in the consumer.
    /// Same record count, but R2 lost its first read and gained one at the end: only the names show
    /// it. Every pair would otherwise be false, with a clean exit.
    #[test]
    fn paired_reader_refuses_different_names() {
        let dir = std::env::temp_dir().join(format!("bwa4_pe_names_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p1 = dir.join("r1.fq");
        let p2 = dir.join("r2.fq");
        std::fs::write(&p1, "@a/1\nACGT\n+\nIIII\n@b/1\nACGT\n+\nIIII\n").unwrap();
        std::fs::write(&p2, "@b/2\nACGT\n+\nIIII\n@a/2\nACGT\n+\nIIII\n").unwrap();
        let err = PairedFastqReader::from_paths(&p1, &p2, false)
            .unwrap()
            .next_batch(usize::MAX)
            .err()
            .expect("different names must be refused");
        assert!(err.to_string().contains("different names"), "{err}");
        // `/1` and `/2` are not part of the name, so genuine mates still pair.
        std::fs::write(&p2, "@a/2\nACGT\n+\nIIII\n@b/2\nACGT\n+\nIIII\n").unwrap();
        let batch = PairedFastqReader::from_paths(&p1, &p2, false)
            .unwrap()
            .next_batch(usize::MAX)
            .unwrap();
        assert_eq!(batch.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn paired_reader_refuses_unequal_files() {
        let dir = std::env::temp_dir().join(format!("bwa4_pe_uneven_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p1 = write_fastq(&dir, "r1.fq", "read", 10);
        let p2 = write_fastq(&dir, "r2.fq", "read", 7);
        let mut r = PairedFastqReader::from_paths(&p1, &p2, false).unwrap();
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
        let mut r = PairedFastqReader::from_paths(&p1, &p2, false).unwrap();
        assert!(r.next_batch(usize::MAX).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The detector answers "does this need unwrapping", and it must answer NO unless it can prove
    /// YES.
    ///
    /// A wrong yes is the expensive direction: the unwrapper then meets input it cannot rewrite,
    /// and needletail's precise complaint about the real problem is replaced by a claim that the
    /// file was empty. That is exactly what an earlier version did to three malformed shapes, and
    /// the last group below is those three.
    #[test]
    fn unwrap_detection() {
        // Four-line FASTQ, FASTA and junk all pass straight through.
        assert!(!needs_unwrapping(
            b"@r1\nACGT\n+\nIIII\n@r2\nACGT\n+\nIIII\n",
            true
        ));
        assert!(
            !needs_unwrapping(b"@r1\r\nACGT\r\n+\r\nIIII\r\n", true),
            "CRLF"
        );
        assert!(
            !needs_unwrapping(b"@r1\nACGT\n+r1\nIIII\n", true),
            "separator repeats the name"
        );
        assert!(!needs_unwrapping(b"", true), "nothing to judge");
        assert!(
            !needs_unwrapping(b">r1\nACGT\nACGT\n", true),
            "FASTA is needletail's own job"
        );

        // Genuinely wrapped, on either half of the record.
        assert!(
            needs_unwrapping(b"@r1\nAC\nGT\n+\nIIII\n", true),
            "wrapped sequence"
        );
        assert!(
            needs_unwrapping(b"@r1\nACGT\n+\nII\nII\n", true),
            "wrapped quality"
        );
        assert!(
            needs_unwrapping(b"@r1\r\nAC\r\nGT\r\n+\r\nIIII\r\n", true),
            "wrapped, CRLF"
        );

        // MALFORMED, and every one of these must be handed to needletail rather than rewritten.
        assert!(
            !needs_unwrapping(b"@a\nACGTACGT\n+\nIIII\n", true),
            "quality shorter than sequence"
        );
        assert!(
            !needs_unwrapping(b"@a\nACGT\n+\nIIIIIIII\n", true),
            "quality longer than sequence"
        );
        assert!(
            !needs_unwrapping(b"@a\nACGT\n+\n", true),
            "truncated before quality"
        );
        assert!(
            !needs_unwrapping(b"@a\nACGT\nIIII\n", true),
            "no separator at all"
        );

        // `complete` is what separates "the file ends here" from "the peek ends here". The same
        // bytes mean malformed input in a whole file and an unfinished record in a prefix.
        assert!(
            !needs_unwrapping(b"@a\nAC\nGT\n", true),
            "whole file, no separator: malformed"
        );
        assert!(
            needs_unwrapping(b"@a\nAC\nGT\n", false),
            "prefix, sequence already wrapped"
        );
        assert!(
            !needs_unwrapping(b"@a\nACGT\n", false),
            "prefix, nothing proven yet"
        );
    }

    /// Wrapped FASTQ is rewritten into the four-line form bwa's `kseq` accepts and needletail
    /// requires.
    ///
    /// The `@`-quality case is the one worth pinning. `@` is a legal quality character, so a parser
    /// that ends the quality block at the first line starting with `@` splits one record into two
    /// and silently corrupts everything after it. kseq's rule, reproduced here, is to accumulate
    /// quality until it is as long as the sequence.
    #[test]
    fn unwrap_rewrites_wrapped_records() {
        let read = |src: &[u8]| {
            use std::io::Read;
            let mut out = Vec::new();
            UnwrapFastq::new(std::io::BufReader::new(std::io::Cursor::new(src.to_vec())))
                .read_to_end(&mut out)
                .unwrap();
            out
        };

        assert_eq!(
            read(b"@r1\nAC\nGT\n+\nII\nII\n@r2\nACGT\n+\nIIII\n"),
            b"@r1\nACGT\n+\nIIII\n@r2\nACGT\n+\nIIII\n"
        );
        // Already four-line input passes through unchanged apart from the bare separator.
        assert_eq!(read(b"@r1\nACGT\n+r1\nIIII\n"), b"@r1\nACGT\n+\nIIII\n");
        // CRLF, and a quality string that both starts with `@` and is wrapped.
        assert_eq!(
            read(b"@r1\r\nACGTAC\r\nGT\r\n+\r\n@@@@\r\n@@@@\r\n"),
            b"@r1\nACGTACGT\n+\n@@@@@@@@\n"
        );
        assert_eq!(read(b""), b"");
    }

    /// A named pipe is not seekable, so the two magic bytes read for format sniffing can never be
    /// re-read. Before the `seekable` split in [`super::open_reader`] the sniff consumed them and
    /// the parser then reopened the path, which on a FIFO yielded a stream already past its header
    /// and therefore ZERO records, silently: `bwa-mem4 mem ref <(zcat r1.gz)` wrote a SAM header
    /// and no alignments, with no error. bwa-mem2 reads that input, so it was a parity gap too.
    ///
    /// WHY THE TIMEOUT, AND WHY THE PATH CARRIES A CLOCK READING. This test could hang forever, and
    /// it did: once in a `cargo test --workspace` run, blocking the whole gate until it was killed
    /// by hand. Sampled in that state the reader had consumed all 83890 bytes, the process held one
    /// fd on the FIFO (`4r`, read-only), and the writer thread no longer existed -- so every writer
    /// had closed and `read` should have returned 0 rather than blocking. What is known about it:
    ///
    /// - it needs the rest of the binary running alongside. 300 runs of this test ALONE, debug, and
    ///   150 release: no hang. 80 runs of the whole test binary, debug: one hang.
    /// - it is not the pattern. The same shape in C -- one thread writing 2500 records into a FIFO
    ///   and closing, the main thread reading to EOF -- reached EOF 400 times out of 400.
    ///
    /// That is not enough to name a cause, and the cause may not be ours. What IS ours is that a
    /// test must not be able to hang a CI job indefinitely, so the read runs on its own thread and
    /// the assertion waits with a bound. On a timeout the reader thread stays blocked and is
    /// abandoned; that is safe, because the harness's exit tears the process down regardless.
    ///
    /// The path gets a nanosecond stamp alongside the pid because a killed run leaves its FIFO
    /// behind, and the next process to be handed the same pid then fails at `mkfifo` for a reason
    /// that has nothing to do with what this test checks.
    #[test]
    #[cfg(unix)]
    fn reads_from_a_non_seekable_fifo() {
        // A timeout is retried on a fresh FIFO: the rare hang described above is not a wrong answer
        // and has not been traced to this reader, while a real regression (wrong count, wrong name,
        // an error) fails on the first attempt, and a reader that truly hangs fails all three.
        for attempt in 1..=3 {
            match fifo_attempt() {
                Some(Ok(n)) => return assert_eq!(n, 2500, "records read from a FIFO"),
                Some(Err(e)) => panic!("{e}"),
                None => eprintln!("FIFO attempt {attempt} timed out after 20s, retrying"),
            }
        }
        panic!("FIFO reader did not finish in 3 attempts of 20s (see this test's doc comment)");
    }

    /// One run of [`reads_from_a_non_seekable_fifo`]: `None` when the reader did not finish.
    #[cfg(unix)]
    fn fifo_attempt() -> Option<std::result::Result<usize, String>> {
        use std::io::Write;
        // Nanoseconds since the epoch, purely to make the directory name unrepeatable. A pid alone
        // is not: pids are recycled, and a hung run leaves its FIFO in place.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("bwa4_fifo_{}_{stamp}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("reads.fq");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .is_ok_and(|s| s.success());
        assert!(made, "mkfifo failed");

        // The writer must run concurrently: opening a FIFO for reading blocks until a writer
        // appears, and opening it for writing blocks until a reader does. It also cannot be joined
        // before the read, because 2500 records are more than a pipe buffer holds.
        let w = fifo.clone();
        let writer = std::thread::spawn(move || {
            let mut f = std::fs::File::create(&w).unwrap();
            for i in 0..2500 {
                writeln!(f, "@read{i}\nACGTACGTAC\n+\nIIIIIIIIII").unwrap();
            }
        });

        // The reader, on its own thread, reporting either the record count or the first name that
        // came out wrong. Anything it can block on is on the far side of this channel.
        let (tx, rx) = std::sync::mpsc::channel::<std::result::Result<usize, String>>();
        let path = fifo.clone();
        std::thread::spawn(move || {
            let mut r = match FastqReader::from_path(&path) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(format!("from_path: {e}")));
                    return;
                }
            };
            let mut n = 0usize;
            loop {
                match r.next_record() {
                    Ok(Some(rec)) => {
                        if rec.name() != format!("read{n}").as_bytes() {
                            let _ = tx.send(Err(format!(
                                "record {n} is named {:?}, not read{n}",
                                rec.name()
                            )));
                            return;
                        }
                        n += 1;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx.send(Err(format!("record {n}: {e}")));
                        return;
                    }
                }
            }
            let _ = tx.send(Ok(n));
        });

        // Generous: the work is 83890 bytes through a pipe, which takes milliseconds. The bound is
        // here to convert a hang into a failure, not to measure anything. On a timeout the writer
        // and reader threads are abandoned (see the doc comment).
        let got = rx.recv_timeout(std::time::Duration::from_secs(20)).ok();
        std::fs::remove_dir_all(&dir).ok();
        if got.is_some() {
            writer.join().unwrap();
        }
        got
    }

    /// The word-at-a-time scan against the byte loop it replaced, around every chunk boundary.
    #[test]
    fn first_le_space_matches_the_byte_scan() {
        for len in 0..40 {
            for at in 0..=len {
                for c in [b' ', b'\t', b'\x0c', b'\x01', 0x20, 0x21, 0x7f, 0x80, 0xff] {
                    let mut v = vec![b'A'; len];
                    if at < len {
                        v[at] = c;
                    }
                    let want = v.iter().position(|&x| x <= b' ').unwrap_or(len);
                    assert_eq!(first_le_space(&v), want, "len {len} at {at} byte {c:#x}");
                }
            }
        }
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

//! FMD index construction, byte-identical to `bwa-mem2 index`.
//!
//! Mirrors `bns_fasta2bntseq` (`.pac`/`.ann`/`.amb`) and `FMI_search::build_index`
//! (`.0123`/`.bwt.2bit.64`) from `reference/bwa-mem2/src`. See `docs`/the plan for the exact
//! byte layout. The suffix array is built with our own SA-IS (`crate::sais`).
//!
//! # Vocabulary used throughout this crate
//!
//! * `L` (`l_pac` in the C): the number of bases in the FORWARD reference, i.e. every contig of
//!   the FASTA concatenated head to tail with no separator. Contig `i` occupies
//!   `[anns[i].offset, anns[i].offset + anns[i].len)`; contig boundaries exist only in `.ann`, the
//!   sequence itself has no delimiter. That is exactly why `bns_fetch_seq` has to clamp an
//!   extension window to the seed's OWN contig: nothing in the packed bases stops a read at a
//!   contig edge from extending straight into the next contig.
//! * "2L space": the aligner works on `forward ++ reverse_complement(forward)`, length `2L`.
//!   Position `p >= L` denotes the reverse strand and maps back to forward coordinate
//!   `2L - 1 - p` (`bns_depos`). A hit found on the reverse half is the same hit as on the
//!   forward half, read the other way, so a single forward search finds both strands.
//! * `N = 2L + 1`: the BWT/suffix-array length. The `+1` is the sentinel, the empty suffix, which
//!   SA-IS places first (`sa[0] = 2L`, mirroring the C's `suffix_array[0] = pac_len` at
//!   `FMI_search.cpp:365`). `ref_seq_len` on disk is this `N`, not `2L`.
//! * "base codes": A=0, C=1, G=2, T=3, and 4 for anything else (`nst_nt4_table`, `bntseq.cpp:40`).
//!   Code 4 never survives into `.pac`/`.0123`: ambiguous bases are replaced by a random base
//!   (see `SEED` below). Code 4 is reused as the BWT sentinel symbol and 6 (`DUMMY_CHAR`,
//!   `FMI_search.h:44`) as the tail padding value.
//!
//! # The five files this module writes
//!
//! | file | written by | contents |
//! |---|---|---|
//! | `.pac` | [`write_pac`] | forward reference only, 2 bits/base, `L/4 + 2` bytes |
//! | `.ann` | [`write_ann`] | text: contig names, offsets, lengths |
//! | `.amb` | [`write_amb`] | text: ambiguous-base (N) runs |
//! | `.0123` | [`build_index`] | `forward ++ revcomp`, ONE byte per base, `2L` bytes |
//! | `.bwt.2bit.64` | [`write_fm_index`] | the FM index: counts, occ checkpoints, sampled SA |
//!
//! # Reading order for this file
//!
//! 1. [`build_index`]: the five-step pipeline, top to bottom. Each step's output feeds the next.
//! 2. [`write_pac`], [`write_ann`], [`write_amb`]: the three simple formats.
//! 3. [`write_fm_index`]: the one complex format, and the only place bit layouts matter.
//!
//! Every rule here exists because the index must be byte-identical to the C's. Where a comment
//! says a value is "load-bearing", changing it produces a file that still loads and still aligns
//! reads, just not the same reads.
//!
//! `.pac` is packed 4x but `.0123` is not: `.pac` is the archival copy that `bns_get_seq` unpacks
//! for the SAM MD/CIGAR path, while `.0123` is read on the hot alignment path where a shift-and-
//! mask per base would cost more than the extra 4.6 GB of address space (it is memory-mapped, see
//! `fmindex::FmIndex::load`).

use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bwa_core::{dna, Error, Result};
use rayon::prelude::*;

use crate::rand48::Rand48;
use crate::sais::suffix_array_inplace;

/// bwa-mem2's fixed RNG seed for ambiguous-base randomization, also echoed as the `.ann` third
/// field. Hard-coded to 11 in `bns_fasta2bntseq` (`bntseq.cpp:311`: `bns->seed = 11; srand48(...)`).
/// It must be 11, and the generator must be glibc's exactly (see [`crate::rand48`]), or every N in
/// the reference gets a different random replacement base and `.pac`/`.0123`/the whole FM index
/// diverge byte-wise from the C's.
const SEED: u32 = 11;

/// Rows per occurrence-checkpoint block, the C's `CP_BLOCK_SIZE` (`FMI_search.h:49`). The record
/// is also 64 BYTES, so one block is one cache line: that is why the stride is 64 and not some
/// other power of two. Baked into the output filename (`.bwt.2bit.64`).
const CP_BLOCK_SIZE: usize = 64;

/// Suffix-array sampling stride, `2^SA_COMPX` with `SA_COMPX = 03` octal = 3 (`macro.h:65`). Only
/// every 8th entry is stored; the reader LF-walks back to the nearest sample.
const SA_SAMPLE_STRIDE: usize = 8;

/// BWT symbol for the sentinel row: the row whose suffix is the empty one, which has no preceding
/// character. Outside `0..4`, so it sets no bit in any of the four one-hot bitmaps, which is
/// exactly the condition `FmIndex::get_sa` uses to stop its LF-walk.
const SENTINEL_CODE: u8 = 4;

/// Padding symbol for BWT rows past the end of the last checkpoint block: the C's `DUMMY_CHAR`
/// (`FMI_search.h:44`). Like the sentinel it sets no bit anywhere, so padded rows are simply
/// absent from every base's bitmap.
const DUMMY_CHAR: u8 = 6;

/// One FASTA record as read: `>name comment` plus the concatenated sequence lines (whitespace
/// stripped, case and IUPAC codes preserved, since `add1` needs the RAW byte to decide whether an
/// N-run continues).
struct FastaContig {
    /// The header's first whitespace-delimited token, with the `>` stripped: the contig name as it
    /// will appear in `.ann` and in the SAM `@SQ SN:` field. Never contains a space or a tab.
    name: String,
    /// Everything on the header line after the first space/tab, verbatim (may itself contain
    /// spaces, may be empty when the header was just `>name`). Empty is turned into the literal
    /// `(null)` when it is copied into [`AnnRec::anno`], never here.
    comment: String,
    /// The contig's sequence lines concatenated with all ASCII whitespace removed, still as RAW
    /// FASTA bytes: mixed case and IUPAC ambiguity codes are preserved, NOT decoded to 0..4 yet.
    /// [`build_index`] needs the raw byte because an `.amb` hole continues only while the literal
    /// character repeats, so `N` and `R` must stay distinguishable.
    seq: Vec<u8>,
}

/// One `.ann` contig record, i.e. the C's `bntann1_t` minus the always-zero `gi`.
struct AnnRec {
    /// Contig name, copied from [`FastaContig::name`]. Written as the second field of the `.ann`
    /// name line and read back by `bns_restore_core` to populate the SAM `@SQ` records.
    name: String,
    /// The FASTA comment, or the literal string `(null)` when there was none. The C stores
    /// `strdup("(null)")` (`bntseq.cpp:261`) rather than an empty string, so the field is never
    /// empty and `bns_dump`'s `if (p->anno[0])` branch always takes the "print it" path. Emitting a
    /// bare empty string here would drop a space and break `.ann` byte-identity.
    anno: String,
    /// Start of this contig in forward (1L) coordinates, i.e. the sum of all previous contig
    /// lengths. Contig 0 starts at 0. Range `[0, l_pac)`. This is a reference POSITION, not a
    /// suffix-array row. Set once when the contig's bases have all been encoded; read back by
    /// `bns_pos2rid`'s binary search, which needs these to be strictly increasing.
    offset: i64,
    /// Contig length in bases. `i32` because the C's `bntann1_t.len` is `int32_t`, which caps a
    /// single contig at 2^31-1 bases (the whole reference is `i64`, only the per-contig field is
    /// 32-bit). Always `> 0` for a contig that had any sequence lines.
    len: i32,
    /// Number of ambiguous-base RUNS (not bases) inside this contig: the count of [`AmbRec`]s whose
    /// offset falls in this contig's span. Purely informational in `.ann`; the authoritative hole
    /// list is `.amb`.
    n_ambs: i32,
}

/// One `.amb` ambiguous-base run ("hole"), i.e. the C's `bntamb1_t`.
struct AmbRec {
    /// Start of the run in forward (1L) coordinates, absolute across the whole reference (NOT
    /// relative to the containing contig, and not a suffix-array row). Range `[0, l_pac)`.
    /// Holes are pushed in ascending offset order and never overlap.
    offset: i64,
    /// Run length in bases, `>= 1`. Grown in place by the `last.len += 1` branch while the same
    /// raw character keeps repeating.
    len: i32,
    /// The literal FASTA character that formed the run (`N`, but also `R`, `Y`, ... since anything
    /// outside ACGT is ambiguous). A run only continues while the raw character is IDENTICAL, so
    /// `NNRR` is two holes, not one.
    amb: char,
}

/// Build the five index files (`.pac`, `.ann`, `.amb`, `.0123`, `.bwt.2bit.64`) next to `fasta`.
///
/// `fasta` is the path to the plain-text reference; every output is `<fasta>.<ext>`, matching the
/// C's convention that the "prefix" passed to `bwa-mem2 index` is the FASTA path itself.
///
/// Peak memory is dominated by step 4/5: `bref` is `2L` bytes and the suffix array is `8 * (2L+1)`
/// bytes, so roughly 10 bytes per forward base (about 31 GB for hg38). The `drop`s below are load-
/// bearing for that, not cosmetic.
///
/// # Parameters
///
/// * `fasta`: path to an existing plain-text (NOT gzipped) FASTA, supplied by the CLI's `index`
///   subcommand. Used both as the input to read and as the PREFIX for all five outputs, which are
///   created/truncated beside it. The caller must have write permission on the containing
///   directory. The file is read fully into memory, so it must fit in RAM.
///
/// # Returns
///
/// `Ok(())` once all five files are written and flushed. `Err` if the FASTA cannot be read, if it
/// parsed to zero contigs, or if any output file cannot be created or written.
pub fn build_index(fasta: &Path) -> Result<()> {
    build_index_with_prefix(fasta, fasta)
}

/// Build the index from `fasta` but name the output files after `prefix`, for `index -p`.
///
/// bwa-mem2's `index -p prefix` does exactly this: the five side files become `<prefix>.pac`,
/// `<prefix>.ann` and so on, and `<prefix>` is then what `mem` is given. The default is `prefix ==
/// fasta`, which is why every path in this project doubles as both the FASTA and the index prefix.
///
/// The CONTENTS are unaffected: the `.ann` records contig names taken from the FASTA, never the
/// file's own path, so an index built with `-p` is byte-identical to one built without it.
///
/// # Parameters
///
/// - `fasta`: the reference to read.
/// - `prefix`: the path the five output files are named after. Its directory must exist and be
///   writable; nothing here creates it.
pub fn build_index_with_prefix(fasta: &Path, prefix: &Path) -> Result<()> {
    let contigs = read_fasta(fasta)?;
    if contigs.is_empty() {
        return Err(Error::Other("empty FASTA".into()));
    }

    // 1. Encode the forward reference to 2-bit codes, randomizing N via lrand48, and collect the
    //    contig annotations and ambiguous-base holes (as bwa-mem2's add1 does).
    //
    // ORDER IS LOAD-BEARING: the C draws one lrand48() per ambiguous base, in reference order, in
    // the same pass that records the holes (`bntseq.cpp:283-290`). Drawing them in a different
    // order, or drawing extras, shifts the whole random stream and changes every later N's
    // replacement base. That is why this loop is strictly sequential while steps 4/5 are not.
    // `rng` is the single glibc-compatible LCG stream for the WHOLE reference: it is created once
    // here and never reset per contig, so draw number k is the k-th ambiguous base in reference
    // order. `forward` accumulates base codes 0..=3, one byte per base, and its length at any
    // moment is the absolute forward (1L) position of the next base to be written. `anns` gets one
    // record per contig, `ambs` one per ambiguous RUN, both in ascending position order.
    let mut rng = Rand48::srand48(SEED as i64);
    let mut forward: Vec<u8> = Vec::new();
    let mut anns: Vec<AnnRec> = Vec::new();
    let mut ambs: Vec<AmbRec> = Vec::new();

    // Loop invariant at the top of each contig iteration: `forward` holds the codes for every base
    // of every PRECEDING contig and nothing else; `ambs` holds every hole that ended before this
    // contig; `rng` has been advanced exactly once per ambiguous base seen so far.
    for c in &contigs {
        // `start` is this contig's first forward (1L) position, which becomes its `AnnRec::offset`.
        let start = forward.len();
        // Ambiguous RUNS seen so far in THIS contig (reset per contig, unlike `ambs`).
        let mut n_ambs = 0i32;
        // `lasts` resets per contig (bwa-mem2's `add1` does `for (i = lasts = 0; ...)`), so an
        // N-run at a contig boundary is split into one hole per contig, never merged across.
        // `lasts` holds the PREVIOUS raw FASTA byte (0 at a contig start, a value no FASTA
        // character takes), the run-continuation test's left operand.
        let mut lasts: u8 = 0;
        // Loop invariant per base: `forward.len()` is this base's absolute forward (1L) position,
        // and if the previous base was ambiguous then `ambs.last()` is its still-open run.
        for &raw in &c.seq {
            // `nt4` is `nst_nt4_table` (`bntseq.cpp:40`): ACGT/acgt -> 0..3, everything else -> 4.
            let mut code = dna::nt4(raw);
            if code >= 4 {
                // `pos` is the ABSOLUTE forward offset (across all contigs so far), because the C
                // records `(*q)->offset = p->offset + i` with `p->offset` the contig start.
                let pos = forward.len() as i64;
                // A hole continues only while the RAW character repeats (`lasts == seq->seq.s[i]`,
                // `bntseq.cpp:266`). Comparing decoded codes instead would wrongly merge `N` with
                // `R`, and comparing against 0 at a contig start is what keeps holes from spanning
                // contigs (see the `lasts` note above and the unit test at the bottom).
                if raw != lasts {
                    ambs.push(AmbRec {
                        offset: pos,
                        len: 1,
                        amb: raw as char,
                    });
                    n_ambs += 1;
                } else if let Some(last) = ambs.last_mut() {
                    last.len += 1;
                }
                // The ambiguous base is REPLACED by a random ACGT, so the packed reference contains
                // no code-4 holes at all (`if (c >= 4) c = lrand48()&3;`, `bntseq.cpp:285`). The
                // `.amb` file is the only record that the base was ever ambiguous; downstream code
                // consults it to avoid calling variants inside a hole.
                code = (rng.lrand48() & 3) as u8;
            }
            forward.push(code);
            lasts = raw;
        }
        // `len` is this contig's base count: every raw byte produced exactly one code, including
        // the ambiguous ones (they were replaced, not dropped), so this equals `c.seq.len()`.
        let len = (forward.len() - start) as i32;
        let anno = if c.comment.is_empty() {
            "(null)".to_string()
        } else {
            c.comment.clone()
        };
        anns.push(AnnRec {
            name: c.name.clone(),
            anno,
            offset: start as i64,
            len,
            n_ambs,
        });
    }

    // `l_pac` is L: the total forward reference length in bases, the sum of all contig lengths.
    // `n_seqs` is the contig count. Both are header fields of `.ann` and `.amb`.
    let l_pac = forward.len() as i64;
    let n_seqs = anns.len() as i32;
    // `contigs` is no longer needed (names/annotations are copied into `anns`); drop it before the
    // memory-heavy suffix-array stage.
    drop(contigs);

    // 2. Write .pac (2-bit packed forward reference).
    write_pac(&sibling(prefix, "pac"), &forward)?;

    // 3. Write .ann and .amb.
    write_ann(&sibling(prefix, "ann"), l_pac, n_seqs, &anns)?;
    write_amb(&sibling(prefix, "amb"), l_pac, n_seqs, &ambs)?;

    // 4. Build the forward++reverse-complement binary reference and write .0123. The RC half is
    //    filled in parallel: bref[L + i] = complement(forward[L - 1 - i]).
    //
    // This is the "2L space" the aligner searches. The C reaches the same array the long way
    // round: `pac2nt` (`FMI_search.cpp:80-140`) unpacks `.pac` into an ASCII string and then
    // appends the complement walking backwards, so `reference_seq[L + i]` is the complement of
    // `reference_seq[L - 1 - i]`. Reversing AND complementing is what makes position `p >= L`
    // decode to forward position `2L - 1 - p` (`bns_depos`), and it is why a plain forward
    // backward-search over this array finds reverse-strand hits for free.
    //
    // Worked micro-example, forward = ACGT (codes 0,1,2,3, L = 4):
    //   bref = [0,1,2,3,  0,1,2,3]   because complement(T)=A, complement(G)=C, ...
    // and 2L space position 5 maps to forward position 2*4-1-5 = 2, the `G`, complemented to `C`.
    //
    // The `if c < 4` guard is dead in practice (step 1 removed every code 4), but is kept so the
    // transform is total; the C's `pac2nt` has the same shape via its switch's missing default.
    // `forward_len` is L again (kept as a `usize` for indexing). `bref` is the 2L-space array:
    // `bref[0..L)` is the forward strand, `bref[L..2L)` the reverse complement. Indices into it are
    // 2L-space POSITIONS, never suffix-array rows.
    let forward_len = forward.len();
    let mut bref = vec![0u8; 2 * forward_len];
    bref[..forward_len].copy_from_slice(&forward);
    // `fwd_half` aliases `bref[0..L)` (already filled, read-only from here), `rc_half` the empty
    // `bref[L..2L)` being written. Splitting is what lets the RC half be filled in parallel.
    let (fwd_half, rc_half) = bref.split_at_mut(forward_len);
    rc_half.par_iter_mut().enumerate().for_each(|(i, dst)| {
        // `i` is the offset WITHIN the RC half, so the 2L-space position being written is `L + i`
        // and it mirrors forward position `L - 1 - i`. `code` is that forward base's code.
        let code = fwd_half[forward_len - 1 - i];
        *dst = if code < 4 { 3 - code } else { code };
    });
    // `forward` is now captured in bref; free it before the suffix array (~3 GB at genome scale).
    drop(forward);
    File::create(sibling(prefix, "0123")).and_then(|f| BufWriter::new(f).write_all(&bref))?;

    // 5. Build and write the FM-index (.bwt.2bit.64).
    write_fm_index(&sibling(prefix, "bwt.2bit.64"), &bref)?;

    Ok(())
}

/// Append `.<ext>` to a path, producing one of the five output filenames.
///
/// # Parameters
///
/// * `prefix`: the index prefix, which for `bwa-mem2 index` is the FASTA path itself (`ref.fa`).
/// * `ext`: the extension WITHOUT its dot (`"pac"`, `"ann"`, `"amb"`, `"0123"`, `"bwt.2bit.64"`).
///   May itself contain dots, as the last one does.
///
/// # Returns
///
/// `<prefix>.<ext>`. This APPENDS rather than replacing any existing extension, so `ref.fa` becomes
/// `ref.fa.pac`, not `ref.pac`. Operates on `OsString` bytes so non-UTF-8 paths survive intact.
fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    let mut s: OsString = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Minimal plain-text FASTA reader (name, comment, concatenated sequence).
///
/// # Parameters
///
/// * `path`: the FASTA to read. Must be plain text (no gzip/bgzf support here) and must fit in
///   memory: the whole file is slurped, then borrowed line by line.
///
/// # Returns
///
/// One [`FastaContig`] per `>` header, in FILE order, which is the order that fixes every contig's
/// forward offset. Sequence lines appearing before any header are silently dropped (there is no
/// `contigs.last_mut()` to append to). An empty or headerless file yields an empty vector, which
/// [`build_index`] rejects. Records with a header but no sequence lines are kept with `seq` empty.
fn read_fasta(path: &Path) -> Result<Vec<FastaContig>> {
    let text = std::fs::read_to_string(path)?;
    let mut contigs: Vec<FastaContig> = Vec::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix('>') {
            // `header` is the line minus the leading `>`. Split at the FIRST space or tab: name
            // before it, comment after it (comments may contain further spaces, names may not).
            let (name, comment) = match header.find([' ', '\t']) {
                Some(i) => (header[..i].to_string(), header[i + 1..].to_string()),
                None => (header.to_string(), String::new()),
            };
            contigs.push(FastaContig {
                name,
                comment,
                seq: Vec::new(),
            });
        } else if let Some(cur) = contigs.last_mut() {
            cur.seq
                .extend(line.bytes().filter(|b| !b.is_ascii_whitespace()));
        }
    }
    Ok(contigs)
}

/// `.pac`: 2-bit packed FORWARD reference (first base in the high bits of each byte); the file is
/// always `floor(L/4)+2` bytes, and the final byte encodes `L mod 4`.
///
/// # Format (written by `bns_fasta2bntseq`, `bntseq.cpp:337-351`; read by `bns_restore_core`)
///
/// ```text
///   byte 0            byte 1                 ...   [pad]  [last byte]
///   b0 b1 b2 b3       b4 b5 b6 b7                          L mod 4
///   ^^ high bits = FIRST base
/// ```
///
/// * Base `i` lives in byte `i >> 2`, at bit offset `(3 - (i & 3)) * 2` counting from the LSB,
///   which is the C's `_set_pac` macro `pac[l>>2] |= c << ((~l & 3) << 1)` (`bntseq.cpp:246`):
///   `~l & 3` and `3 - (l & 3)` are the same value for the low two bits. So the first base of a
///   byte sits in bits 7..6, the fourth in bits 1..0. Big-endian WITHIN the byte, which is why you
///   cannot just `u32::from_le_bytes` a `.pac` and shift.
/// * Worked micro-example, forward = A C G T = 0,1,2,3:
///   `0<<6 | 1<<4 | 2<<2 | 3<<0` = `0b00_01_10_11` = `0x1B`, one byte, and `L mod 4 == 0`.
///   The file is then `0x1B, 0x00, 0x00`: the payload byte, the extra zero byte, the remainder.
/// * There is NO magic number and NO length field. The reader recovers `L` from the file size and
///   the last byte: `L = (size - 1) * 4 - (4 - last) mod 4`, hence the padding rule below.
/// * Only the forward strand is stored. `bwa-mem2 index` calls `bns_fasta2bntseq(fp, prefix, 1)`
///   (`bwtindex.cpp:72`) with `for_only = 1`, so the C's reverse-complement-appending branch is
///   skipped and `.pac` holds `L` bases, not `2L`. The 2L array lives in `.0123` instead. (Plain
///   bwa-0.7 passes `for_only = 0` and gets a 2L `.pac`; the two tools' `.pac` files differ.)
///
/// INVARIANT: `forward` contains only codes 0..=3. A code >= 4 would spill into the neighbouring
/// base's bits via the `|=` and silently corrupt two bases at once.
///
/// # Parameters
///
/// * `path`: the `.pac` output path from [`sibling`]. Created, truncated if it exists.
/// * `forward`: the FORWARD reference only, one base code per byte, each in `0..=3`. Length is `L`.
///   Index `i` is a forward (1L) reference POSITION. Passing the 2L array here would produce a
///   `.pac` twice the correct size (that is plain bwa-0.7's layout, not bwa-mem2's).
fn write_pac(path: &Path, forward: &[u8]) -> Result<()> {
    // `n_bases` is L. `pac` is the packed buffer, sized `L/4 + 1` bytes so that the final partial
    // base group always has a byte to land in; index `i >> 2` is a BYTE offset, not a base index.
    let n_bases = forward.len();
    let mut pac = vec![0u8; (n_bases >> 2) + 1];
    // `i` is the forward base position; `code` its 2-bit value. Invariant per iteration: every bit
    // group for bases `< i` is already OR-ed in, and this base's group is still zero.
    for (i, &code) in forward.iter().enumerate() {
        pac[i >> 2] |= code << ((3 - (i & 3)) << 1);
    }
    // `payload_bytes` = the number of bytes actually holding bases: `L/4` whole bytes plus one
    // partial byte when `L` is not a multiple of 4. Mirrors the C's
    // `(l_pac>>2) + ((l_pac&3) == 0 ? 0 : 1)` (`bntseq.cpp:344`).
    let payload_bytes = (n_bases >> 2) + usize::from(!n_bases.is_multiple_of(4));
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&pac[..payload_bytes])?;
    // The C's comment: "the following codes make the pac file size always (l_pac/4+1+1)". When `L`
    // is a multiple of 4 there is no partial byte, so an explicit zero byte is emitted to keep the
    // size formula (and therefore the reader's length arithmetic) uniform. Dropping this byte makes
    // the file one shorter and the reader computes an `L` that is 4 too small.
    if n_bases.is_multiple_of(4) {
        out.write_all(&[0u8])?;
    }
    // Trailer: how many bases the last payload byte really holds, 0 meaning "a full 4".
    out.write_all(&[(n_bases % 4) as u8])?;
    out.flush()?;
    Ok(())
}

/// `.ann` (text): `l_pac n_seqs seed`, then per contig `gi name anno` and `offset len n_ambs`.
///
/// # Format (written by `bns_dump`, `bntseq.cpp:73-101`; read by `bns_restore_core`)
///
/// ```text
///   <l_pac> <n_seqs> <seed>            // %lld %d %u
///   <gi> <name> <anno>                 // %d %s [ %s], repeated n_seqs times
///   <offset> <len> <n_ambs>            // %lld %d %d
/// ```
///
/// Whitespace-separated ASCII, LF line endings, no trailing blank line. Byte-identity notes:
/// * `gi` is always the literal `0` (`p->gi = 0` at `bntseq.cpp:262`); bwa never populates it.
/// * `anno` is always present because the C substitutes `(null)` for a missing comment, so the
///   space before it is always emitted. See [`AnnRec::anno`].
/// * `seed` is always 11, see [`SEED`]. Readers use it for nothing; it only documents which RNG
///   stream produced the N replacements.
/// * The parser side splits the name line into at most 3 fields, because a FASTA comment may
///   contain spaces while a contig name may not.
///
/// INVARIANT: `anns[i].offset == anns[i-1].offset + anns[i-1].len` and `anns[0].offset == 0`, so
/// the offsets tile `[0, l_pac)` with no gaps. `bns_pos2rid` binary-searches on exactly that
/// assumption and returns garbage if it is broken.
///
/// # Parameters
///
/// * `path`: the `.ann` output path.
/// * `l_pac`: `L`, the forward reference length in bases. Header field 1. Must equal
///   `anns.last().offset + anns.last().len`.
/// * `n_seqs`: the contig count. Header field 2. Must equal `anns.len()`; it is passed separately
///   only to mirror the C's signature, and a mismatch would make the file unparseable.
/// * `anns`: one record per contig, in reference order (the order that defines the offsets).
fn write_ann(path: &Path, l_pac: i64, n_seqs: i32, anns: &[AnnRec]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{l_pac} {n_seqs} {SEED}")?;
    for a in anns {
        // gi is always 0; anno is always non-empty ("(null)" when absent) so it is always printed.
        writeln!(w, "0 {} {}", a.name, a.anno)?;
        writeln!(w, "{} {} {}", a.offset, a.len, a.n_ambs)?;
    }
    w.flush()?;
    Ok(())
}

/// `.amb` (text): `l_pac n_seqs n_holes`, then per hole `offset len amb_char`.
///
/// # Format (written by `bns_dump`, `bntseq.cpp:102-114`)
///
/// ```text
///   <l_pac> <n_seqs> <n_holes>   // %lld %d %u  (third field is n_holes, NOT the seed)
///   <offset> <len> <amb>         // %lld %d %c, repeated n_holes times
/// ```
///
/// The first two header fields duplicate `.ann` exactly; only the third differs. Holes are in
/// ascending `offset` order and never overlap, since they are emitted by a single forward scan.
/// `offset` is in forward (1L) coordinates, absolute across the whole reference, so a hole never
/// straddles a contig boundary (the C resets its run tracker per contig, see [`build_index`]).
///
/// `sum(len)` over all holes is the total ambiguous base count, which is also the exact number of
/// `lrand48()` draws made while building `.pac`.
///
/// # Parameters
///
/// * `path`: the `.amb` output path.
/// * `l_pac`: `L`, the forward reference length in bases. Header field 1, identical to `.ann`'s.
/// * `n_seqs`: the contig count. Header field 2, identical to `.ann`'s.
/// * `ambs`: every ambiguous run in the whole reference, in ascending `offset` order. The slice
///   LENGTH is header field 3 (`n_holes`), so this is not the place `.ann`'s per-contig `n_ambs`
///   comes from. May be empty for an N-free reference, giving a header-only file.
fn write_amb(path: &Path, l_pac: i64, n_seqs: i32, ambs: &[AmbRec]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{} {} {}", l_pac, n_seqs, ambs.len())?;
    for a in ambs {
        writeln!(w, "{} {} {}", a.offset, a.len, a.amb)?;
    }
    w.flush()?;
    Ok(())
}

/// `.bwt.2bit.64`: `[ref_seq_len:i64 | count[5]:i64 | CP_OCC[] | sa_ms_byte[]:i8 | sa_ls_word[]:u32
/// | sentinel_index:i64]`, all little-endian. `bref` is the 2L forward++RC binary reference.
///
/// # Format (written by `FMI_search::build_fm_index`, `FMI_search.cpp:142-300`; read by
/// `FMI_search::load_index`, `FMI_search.cpp:384-450`)
///
/// Despite the name there is no magic number and no version field: the ".64" in the FILENAME is
/// the format version, being `CP_BLOCK_SIZE` (`CP_FILENAME_SUFFIX ".bwt.2bit.64"`,
/// `FMI_search.h:50`). A build compiled with a different block size writes a differently named
/// file, so a mismatched index fails to open rather than being misparsed. Every integer is native
/// little-endian (the C `fwrite`s raw structs; the format is therefore x86/ARM-LE only).
///
/// Byte layout, in order, with `N = ref_seq_len = 2L + 1`:
///
/// ```text
///   off  0                 : ref_seq_len : i64          = N = 2L + 1
///   off  8                 : count[0..5] : i64 x 5      cumulative base counts, see below
///   off 48                 : cp_occ[cp_size]            cp_size = (N >> 6) + 1, 64 bytes each
///   off 48 + 64*cp_size    : sa_ms_byte[sa_size] : i8   sa_size = (N >> 3) + 1
///   ... + sa_size          : sa_ls_word[sa_size] : u32
///   ... + 4*sa_size        : sentinel_index : i64
/// ```
///
/// Total file size is exactly `48 + 64*cp_size + 5*sa_size + 8`; `FmIndex::load` relies on that to
/// justify its bulk copies. Note the OFF-BY-ONE trap: the header stores `N = 2L+1`, one more than
/// the length of `.0123`, because the BWT has one extra row for the sentinel suffix.
///
/// ## `count[5]`
///
/// `count[c]` is the number of bases strictly less than `c` in the 2L reference, i.e. the classic
/// FM-index `C[]` array: `{0, #A, #A+#C, #A+#C+#G, 2L}`. `count[4]` is the total. `load_index`
/// then adds 1 to ALL FIVE entries (`FMI_search.cpp:424`) to make room for the sentinel row at BWT
/// index 0, so the on-disk values and the in-memory values differ by one. Writing the incremented
/// values here would double-count.
///
/// ## `CP_OCC` (the occurrence checkpoints), one per 64 BWT rows
///
/// ```text
///   struct CP_OCC {            // FMI_search.h:55, 64 bytes, _mm_malloc'd 64-aligned
///       int64_t  cp_count[4];      // occurrences of A,C,G,T in bwt[0 .. block_start)
///       uint64_t one_hot_bwt_str[4]; // bit b of word c is 1 iff bwt[block_start + b] == c
///   }
/// ```
///
/// The stride is 64 rows and the record is 64 bytes, which is the whole point: counts and bitmaps
/// for a block live in ONE cache line, so an occurrence query is a single memory touch (see
/// `FmIndex::get_occ`). A stride of 64 costs `8 * (2L+1) / 64 * 8 = 1` byte per base of index.
///
/// Bit order inside `one_hot`: the block is shifted in MSB-first (`one_hot <<= 1` then `|= 1`,
/// `FMI_search.cpp:234-244`), so row `block_start + j` is bit `63 - j`. That convention is what
/// makes `get_occ`'s mask a simple "top `y` bits set" constant, and it is why `get_sa`'s LF-walk
/// tests bit `63 - (sp & 63)`.
///
/// Worked micro-example, the first block of a BWT starting `A C A ...` (codes 0,1,0):
///   `one_hot[0] = 0b101000...0` (bits 63 and 61), `one_hot[1] = 0b010000...0` (bit 62).
///   Occurrences of A in `bwt[0..2)` = `cp_count[0] + popcount(one_hot[0] & 0b11000...)` = 0 + 1.
///
/// ## Sampled suffix array
///
/// `SA_COMPRESSION` is on and `SA_COMPX = 03` octal = 3 (`macro.h:64-66`), so only every 8th
/// suffix-array entry is stored: `sa_size = (N >> 3) + 1` samples, entry `k` holding `sa[8k]`.
/// Each value is split across two arrays rather than stored as one `i64`: the low 32 bits in
/// `sa_ls_word` and bits 32..40 in `sa_ms_byte`, giving 40 usable bits (`< 2^40`, ample for
/// `2L + 1` on any real genome). Splitting saves 3 bytes per sample versus `i64` and keeps the two
/// arrays separately contiguous. Rows not sampled are recovered by LF-walking backwards to the
/// next multiple of 8, see `FmIndex::get_sa`.
///
/// ## `sentinel_index`
///
/// The BWT row whose suffix is the empty one, i.e. the unique `i` with `sa[i] == 0`. It is stored
/// because `backward_ext` needs it to keep the bidirectional interval's reverse start `l` correct
/// (the sentinel occupies a row but is not one of the 4 bases, so it shifts `l` by one whenever it
/// falls inside the interval).
///
/// `bref` must be the `2L` forward++RC array from step 4, containing only codes 0..=3.
///
/// # Parameters
///
/// * `path`: the `.bwt.2bit.64` output path. The ".64" in it is the format version and must track
///   [`CP_BLOCK_SIZE`].
/// * `bref`: the 2L-space reference (`forward ++ revcomp(forward)`), one code per byte, every byte
///   in `0..=3`, length exactly `2 * l_pac` and even. Indices into it are 2L-space POSITIONS. A
///   code >= 4 here would be counted into no histogram bucket and would silently desynchronize
///   `count[]` from the BWT.
fn write_fm_index(path: &Path, bref: &[u8]) -> Result<()> {
    // ---- Step A: suffix array ----------------------------------------------------------------
    // `two_l` = 2L, the number of real bases (no sentinel). `sa` maps ROW -> 2L-space POSITION:
    // `sa[row]` is where the suffix ranked `row`-th starts. `n` = N = 2L+1 is the ROW count, one
    // more than the position count because of the empty (sentinel) suffix at row 0.
    let two_l = bref.len(); // 2L
    let sa = suffix_array_inplace(bref); // length N = 2L+1, sa[0] = 2L (memory-efficient SA-IS)
    let n = sa.len(); // reference_seq_len = 2L + 1

    // ---- Step B: count[5], the cumulative base counts ----------------------------------------
    // count[5] = {0, #A, #A+#C, #A+#C+#G, 2L} over the 2L binary bases (parallel histogram reduce).
    // The C builds this in `build_index` while transcoding the reference (`FMI_search.cpp:345-360`)
    // and does the same rotate-into-cumulative shuffle. Note it counts the 2L reference, NOT the
    // N-length BWT: the sentinel is excluded here and compensated by load_index's `+1` per entry.
    // Because the second half is the reverse complement of the first, #A == #T and #C == #G exactly.
    // `hist[c]` is the raw occurrence count of base code `c` in the 2L reference (not cumulative).
    // The fold/reduce pair keeps a private 4-slot histogram per rayon chunk, then sums them.
    let hist = bref
        .par_iter()
        .fold(
            || [0i64; 4],
            |mut h, &c| {
                h[c as usize] += 1;
                h
            },
        )
        .reduce(
            || [0i64; 4],
            |mut a, b| {
                for k in 0..4 {
                    a[k] += b[k];
                }
                a
            },
        );
    // `count[c]` = number of bases strictly less than `c`, i.e. the exclusive prefix sum of `hist`
    // with a final total. These are ROW offsets in the sorted BWT (where each base's block of rows
    // begins), on-disk values, one less per entry than what `load_index` keeps in memory.
    let count: [i64; 5] = [
        0,
        hist[0],
        hist[0] + hist[1],
        hist[0] + hist[1] + hist[2],
        two_l as i64,
    ];

    // BWT: bwt[i] = 4 (sentinel) if sa[i]==0 else bref[sa[i]-1]. Each entry is independent, so fill
    // in parallel; the sentinel row (the unique sa[i]==0) is found by a parallel search.
    //
    // This is the textbook Burrows-Wheeler transform: row `i` of the sorted-rotations matrix ends
    // with the character PRECEDING suffix `sa[i]`, and `sa[i] == 0` has no predecessor, hence the
    // sentinel symbol 4. Mirrors `FMI_search.cpp:170-199`.
    //
    // ---- Step C: the BWT itself ---------------------------------------------------------------
    // The initial fill value 6 is [`DUMMY_CHAR`] (`FMI_search.h:44`), the C's padding for the tail of
    // its 64-aligned `bwt` buffer. Every one of the `n` entries is overwritten below, so the 6 only
    // matters as documentation of intent; the real padding is applied in the checkpoint loop, which
    // substitutes 6 for out-of-range rows so the last partial block matches the C byte for byte.
    // `bwt` is indexed by ROW (length N), holding the base code that PRECEDES that row's suffix.
    let mut bwt = vec![DUMMY_CHAR; n];
    bwt.par_iter_mut()
        .zip(sa.par_iter())
        .for_each(|(bwt_symbol, &sa_value)| {
            *bwt_symbol = if sa_value == 0 {
                SENTINEL_CODE
            } else {
                bref[(sa_value - 1) as usize]
            };
        });
    // `sentinel_index` is a ROW, not a position: the unique row whose suffix starts at 2L-space
    // position 0, hence has no preceding character. `position_any` is safe despite its unordered
    // name because exactly one element satisfies the predicate.
    let sentinel_index = sa
        .par_iter()
        .position_any(|&sa_value| sa_value == 0)
        .expect("suffix array must contain the sentinel (0)") as i64;

    // CP_OCC checkpoints, one per 64-base block. `one_hot[block_index]` is independent per block; `cp_count[block_index]`
    // is the running base-count prefix over prior blocks. Compute both per block in parallel, then a
    // cheap sequential prefix sum turns per-block counts into the running totals.
    //
    // `cp_count` is a PREFIX count (bases in `bwt[0 .. block_index*64)`, exclusive of the block itself), so
    // the query is `cp_count + popcount(bits before the row)`. The C achieves this by snapshotting
    // its running `cp_count[]` at the top of each block, BEFORE folding the block in
    // (`FMI_search.cpp:218-249`); we instead accumulate per-block counts in parallel and shift them
    // by one block in the sequential pass below, which yields the identical array.
    //
    // Block count is `(N >> 6) + 1`, matching `cp_occ_size` at `FMI_search.cpp:207`. The `+1` makes
    // the array one block longer than strictly needed, so a query at `pp == N` (a half-open
    // interval end landing exactly on the last row) still indexes a valid block instead of running
    // off the end. INVARIANT: callers may pass `pp` in `[0, N]` inclusive, and only that.
    // ---- Step D: the occurrence checkpoints ----------------------------------------------------
    // `n_blocks` is the checkpoint-record count (one per 64 ROWS, plus the guard block).
    // `cp_count[b][c]` starts as the count of base `c` WITHIN block `b` and is rewritten below into
    // the prefix count before block `b`. `one_hot[b][c]` is the block's 64-row bitmap for base `c`.
    let n_blocks = (n >> 6) + 1;
    let mut cp_count = vec![[0i64; 4]; n_blocks];
    let mut one_hot = vec![[0u64; 4]; n_blocks];
    cp_count
        .par_iter_mut()
        .zip(one_hot.par_iter_mut())
        .enumerate()
        .for_each(|(block_index, (block_count, block_one_hot))| {
            // `block_first_row` is this block's first BWT ROW. Loop invariant at the top of
            // iteration `j`: the four words hold rows `block_first_row .. block_first_row + j`
            // left-aligned at bit 63, and `block_count` counts those same rows.
            let block_first_row = block_index * CP_BLOCK_SIZE;
            for j in 0..CP_BLOCK_SIZE {
                // `idx` is the absolute BWT ROW for slot `j`; it may exceed `n` in the last block.
                let idx = block_first_row + j;
                // Past the end of the BWT, feed `DUMMY_CHAR` (6) so those bit positions stay 0 in
                // all four words, exactly as the C reads its 6-padded aligned buffer.
                let c = if idx < n { bwt[idx] } else { DUMMY_CHAR };
                // Shift ALL FOUR words every row, then set the one bit for this row's base. Shifting
                // first is what makes row `j` land at bit `63 - j` (MSB-first), the convention
                // `get_occ`'s mask table and `get_sa`'s `63 - (sp & 63)` both depend on. The
                // sentinel (4) and the padding (6) set no bit anywhere, so they are simply absent
                // from every base's bitmap, which is exactly the semantics `get_sa` exploits when it
                // detects "no bit set in any word" and stops the LF-walk.
                for word in block_one_hot.iter_mut() {
                    *word <<= 1;
                }
                if (c as usize) < 4 {
                    block_one_hot[c as usize] |= 1;
                    // Per-block count, converted to a running prefix below.
                    block_count[c as usize] += 1;
                }
            }
        });
    // Sequential prefix sum over blocks (n/64 iterations): cp_count[block_index] becomes the count of each
    // base in bwt[0..bi*64), matching the original running-total semantics.
    // `running[c]` is the invariant: at the top of iteration `b` it holds the number of `c`s in
    // `bwt[0 .. b*64)`, which is exactly the value block `b` must store.
    let mut running = [0i64; 4];
    for block_count in cp_count.iter_mut() {
        // Save this block's OWN counts before overwriting the slot with the prefix.
        let block = *block_count;
        *block_count = running;
        for k in 0..4 {
            running[k] += block[k];
        }
    }

    // Compressed suffix array: sample every 8th entry (N is odd, so exactly (N>>3)+1 samples).
    // Stride 8 is `SA_COMPX = 03` octal (`macro.h:65`), i.e. 2^3. `N = 2L+1` is odd, so `N-1` is the
    // largest row and `(N>>3)*8 <= N-1`: every sampled index is in range and no `+1` slot is wasted.
    // Splitting each value into a low u32 and a signed-i8 high byte costs 5 bytes per sample rather
    // than 8; `sa_ms_byte` is `int8_t` in the C, which is harmless because bit 39 of a genome-scale
    // position is never set, but it does mean a reader MUST sign-extend consistently (we mirror it
    // with `i8` in `FmIndex`, not `u8`).
    // ---- Step E: the sampled suffix array ------------------------------------------------------
    // `sa_count` is the number of SAMPLES, not rows: sample `k` corresponds to ROW `8k`.
    let sa_count = (n >> 3) + 1;
    let (sa_ls_word, sa_ms_byte): (Vec<u32>, Vec<u8>) = (0..sa_count)
        .into_par_iter()
        .map(|k| {
            // `v` is a 2L-space reference POSITION (the suffix start for row `8k`), not a row.
            let v = sa[k * SA_SAMPLE_STRIDE] as u64;
            ((v & 0xFFFF_FFFF) as u32, ((v >> 32) & 0xFF) as u8)
        })
        .unzip();
    debug_assert_eq!(sa_ms_byte.len(), sa_count);

    // Serialize. Field ORDER here is the format: header, counts, all checkpoints, then the two SA
    // arrays as separate blocks (ms_byte first, then ls_word: the C writes them in that order at
    // `FMI_search.cpp:271-273`, and `load_index` reads them back in that order), then the sentinel
    // last. Writing `cp_count` and `one_hot` interleaved per block, as below, reproduces the C's
    // `fwrite` of the packed `CP_OCC` struct: 4 x i64 then 4 x u64, no padding (both members are
    // 8-byte aligned so `align(64)` adds none), 64 bytes exactly.
    // ---- Step F: serialize --------------------------------------------------------------------
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&(n as i64).to_le_bytes())?;
    for c in count {
        out.write_all(&c.to_le_bytes())?;
    }
    for (block_count, block_one_hot) in cp_count.iter().zip(one_hot.iter()) {
        for &v in block_count {
            out.write_all(&v.to_le_bytes())?;
        }
        for &v in block_one_hot {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.write_all(&sa_ms_byte)?;
    for &v in &sa_ls_word {
        out.write_all(&v.to_le_bytes())?;
    }
    out.write_all(&sentinel_index.to_le_bytes())?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An N-run must NOT be merged across a contig boundary: bwa-mem2's `add1` resets its
    /// `lasts` tracker to 0 at the start of every contig, so N at the end of one contig and N at
    /// the start of the next are recorded as two separate `.amb` holes, not one.
    #[test]
    fn amb_holes_not_merged_across_contig_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "bwa3_amb_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fasta = dir.join("t.fa");
        // c1 ends with NN (pos 4,5); c2 starts with NN (pos 6,7). l_pac = 12.
        std::fs::write(&fasta, ">c1\nACGTNN\n>c2\nNNACGT\n").unwrap();

        build_index(&fasta).unwrap();
        let amb = std::fs::read_to_string(dir.join("t.fa.amb")).unwrap();
        let lines: Vec<&str> = amb.lines().collect();

        assert_eq!(lines[0], "12 2 2", "header l_pac n_seqs n_holes");
        assert_eq!(lines[1], "4 2 N", "chr1 trailing NN, its own hole");
        assert_eq!(lines[2], "6 2 N", "chr2 leading NN, a separate hole");

        std::fs::remove_dir_all(&dir).ok();
    }
}

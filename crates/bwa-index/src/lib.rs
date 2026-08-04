//! FMD index construction and loading.
//!
//! Phase 0 implements only reference-metadata parsing (`.ann`/`.amb`) needed for the SAM header.
//! Index construction (`build`) and the FM traversal (`fmindex`) arrive in phases 1-2.
//!
//! # Map of this crate
//!
//! | module | role | C counterpart |
//! |---|---|---|
//! | [`build`] | writes all five index files; START HERE for the on-disk formats | `bns_fasta2bntseq`, `FMI_search::build_index` |
//! | [`fmindex`] | loads `.bwt.2bit.64`/`.0123` and runs the FM search | `FMI_search::load_index`, `backwardExt`, `GET_OCC` |
//! | [`bntseq`] | reads `.ann`/`.amb`; contig lookup and coordinate mapping | `bns_restore_core`, `bns_pos2rid`, `bns_depos` |
//! | [`sais`] | suffix-array construction | `saisxx` (`sais.h`) |
//! | [`rand48`] | glibc LCG, needed only to reproduce N-base randomization | `srand48`/`lrand48` |
//! | [`lisa`], [`rmi`], [`packed`] | learned-index experiment, no C counterpart | none |
//!
//! # Rust mechanics used in this crate
//!
//! This crate is where the tree's `unsafe` lives, so it is worth saying up front what that keyword
//! does and does not mean. `unsafe` does NOT switch off the compiler's checks or make the code
//! faster. It marks a block in which a few extra operations become available (dereferencing a raw
//! pointer, calling into the operating system, reinterpreting a buffer's bytes as another type) and
//! declares that the author, not the compiler, has checked the conditions those operations require.
//! Every such block in this crate is accompanied by the argument for why it is sound; that argument
//! IS the safety, and reviewing it is the point.
//!
//! There are exactly three reasons `unsafe` appears here, and none of them is performance in the
//! sense of skipping a bounds check:
//!
//! 1. **Memory mapping.** `Mmap::map` is unsafe because the operating system, not Rust, owns the
//!    pages, and another process truncating the file underneath would break the buffer's promises.
//!    The crate assumes an index file is not modified while in use, exactly as bwa-mem2 does.
//! 2. **Reading a file straight into a typed, aligned buffer.** The index files hold arrays of
//!    fixed-layout records. Reading bytes into a `Vec<u8>` and converting afterwards would double
//!    peak memory on a 19 GB index, so the buffer is allocated with the right type and alignment
//!    and the read is aimed at its bytes directly. That view is built by hand, and the accompanying
//!    comments are what establish that its length and alignment are right.
//! 3. **Advising the kernel.** `madvise` is a raw system call taking a pointer and a length.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `unsafe { ... }` | a block where the author takes responsibility for conditions the compiler cannot check. Always paired here with the reasoning that discharges them. |
//! | `unsafe fn` | a function whose CALLER must uphold something. Calling one requires an `unsafe` block at the call site, which is what makes the obligation visible there. |
//! | `Mmap` | a file mapped into memory so it can be read as a byte slice, with the operating system paging it in on demand. Used for the 6.2 GB packed reference, which no run touches in full. |
//! | `*const u8` / `*mut u8` | raw pointers. Unlike references they carry no lifetime and no aliasing promise, which is exactly why using one requires `unsafe`. |
//! | `std::slice::from_raw_parts_mut(p, n)` | builds a writable slice of `n` bytes starting at `p`. Sound only if those bytes are really allocated, really writable, and not aliased; the surrounding comments state why each holds. |
//! | `.as_mut_ptr() as *mut u8` | takes the address of a typed buffer and views it as bytes, so a file can be read into it without an intermediate copy. |
//! | `Vec::with_capacity` + `set_len` | reserve memory, then declare it initialised once the read has filled it. This is the pattern the point above describes, and the reason it is written out rather than hidden. |
//! | `#[repr(C)]` | lays a struct out exactly as C would, field by field with C's padding rules. Required for any type read directly from a file bwa-mem2 wrote. |
//! | `u64::from_le_bytes` | assembles an integer from bytes in little-endian order explicitly, rather than assuming the host's byte order. |
//! | `.par_iter()` (rayon) | a parallel walk, used in index construction where the work is embarrassingly parallel and the output order is fixed by index rather than by completion. |
//!
//! # Crate-wide glossary
//!
//! These names appear across several modules and are all inherited from bwa-mem2's C. They are
//! deliberately NOT renamed, because diffing this Rust against the C line by line is how every
//! parity bug in this project has been found.
//!
//! | name | plain-language meaning |
//! |---|---|
//! | `l_pac` (also written `L`) | Length of the FORWARD reference in bases: every contig of the FASTA concatenated head to tail with NO separator between them. |
//! | "2L space" | The `forward ++ reverse_complement(forward)` array the aligner actually searches, of length `2 * l_pac`. A position `>= l_pac` is a reverse-strand hit and maps back to forward coordinate `2*l_pac - 1 - p`. One forward-only search structure therefore covers both strands. |
//! | `ref_seq_len` (`N`) | `2 * l_pac + 1`: the number of BWT/suffix-array rows. The `+1` is the sentinel, the empty suffix, which sorts before everything and occupies row 0. |
//! | base codes | `A=0 C=1 G=2 T=3`; `4` means anything else. Code 4 never survives into the packed reference (ambiguous bases are replaced by a random base and recorded in `.amb`), so it is reused as the BWT sentinel symbol. `6` is the tail-padding symbol. |
//! | `rid` | Reference-sequence index: which contig a position falls in. Negative means "no single contig". |
//! | FM interval `(k, l, s)` | A contiguous range of suffix-array ROWS sharing a prefix. `k` is its first row, `s` its size (= the pattern's occurrence count), and `l` the first row of the same pattern reverse-complemented (bookkeeping for bidirectional search only). |
//! | `occ` | "How many times does base `c` appear in the BWT before row `p`". The one primitive that makes each search step constant time. |
//! | `sa` | The suffix array: `sa[row]` is the reference position where the suffix at that row starts. |
//!
//! Because the project requires a BYTE-IDENTICAL index on disk, the formats are not merely
//! compatible: the exact field widths, the padding rules, the sampling strides, and even the order
//! of `lrand48()` draws must match bwa-mem2. `build` documents each of those; `fmindex` documents
//! the read side. Anything marked DIVERGENCE in this crate has been checked to be output-neutral.

/// `.ann`/`.amb` reference metadata: contigs, ambiguous-base holes, coordinate mapping.
pub mod bntseq;
/// Index construction and the authoritative description of every on-disk format.
pub mod build;
/// FM-index loading and backward search.
pub mod fmindex;
/// Transparent-huge-page hint for the multi-gigabyte, randomly-walked index arrays.
pub mod hugepage;
/// Learned suffix array (BWA-MEME style). Experimental, no bwa-mem2 counterpart.
pub mod lisa;
/// 5-byte-per-element integer array used by the learned index.
pub mod packed;
/// glibc `srand48`/`lrand48` reproduction, needed for byte-identical `.pac` on N-containing FASTAs.
pub mod rand48;
/// Recursive model index backing [`lisa`].
pub mod rmi;
/// Suffix array construction by induced sorting.
pub mod sais;

// Re-exports: the types callers outside this crate actually touch. `FmIndex` + `BntSeq` together
// are what the aligner needs to turn a read into a reference coordinate.
pub use bntseq::{Amb, BntSeq, Contig};
pub use build::{build_index, build_index_with_prefix};
pub use fmindex::{traffic, FmIndex, Smem};
pub use lisa::LearnedSa;

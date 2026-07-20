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

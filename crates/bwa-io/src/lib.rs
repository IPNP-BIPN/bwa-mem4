//! Sequence I/O: FASTQ input (needletail) and hand-formatted SAM output.
//!
//! SAM is written by hand (not via a library) so we control the exact bytes, which is required for
//! the bit-identity goal against bwa-mem2.
//!
//! The crate is the aligner's two ends and nothing in between: [`fastq`] turns files into
//! [`Record`]s, [`sam`] turns finished alignments into bytes. It holds no alignment logic and knows
//! nothing about the index, so every formatting decision that affects output bytes is either here
//! or explicitly passed in preformatted by the caller (see [`sam::write_mapped_se`]).

pub mod fastq;
pub mod sam;

pub use fastq::{FastqReader, InterleavedFastqReader, PairedFastqReader, Record};
pub use sam::{write_header, write_unmapped, SqRecord};

//! Sequence I/O: FASTQ input (needletail) and hand-formatted SAM output.
//!
//! SAM is written by hand (not via a library) so we control the exact bytes, which is required for
//! the bit-identity goal against bwa-mem2.
//!
//! The crate is the aligner's two ends and nothing in between: [`fastq`] turns files into
//! [`Record`]s, [`sam`] turns finished alignments into bytes. It holds no alignment logic and knows
//! nothing about the index, so every formatting decision that affects output bytes is either here
//! or explicitly passed in preformatted by the caller (see [`sam::write_mapped_se`]).
//!
//! # Rust mechanics used in this file
//!
//! A crate root, so it holds a map rather than logic. `bwa_core::lib` glosses `pub mod` and
//! `pub use` in full; the one thing to notice here is what the re-export list does NOT contain.
//! [`sam::write_mapped_se`] is deliberately absent, so it must be named through its module. Adding
//! it to the list below would be a purely cosmetic change with a real consequence: a shortcut at
//! the crate root reads as "this is the ordinary way in", and for that function it is not.

pub mod fastq;
pub mod sam;

pub use fastq::{FastqReader, InterleavedFastqReader, PairedFastqReader, Record};
pub use sam::{write_header, write_unmapped, SqRecord};

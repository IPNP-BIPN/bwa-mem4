//! Crate-wide error type.

use thiserror::Error;

/// Errors produced across bwa-mem4 crates.
///
/// Deliberately coarse. These are the failure modes a user can act on (bad index, bad FASTQ, disk),
/// not a taxonomy of internal states. Programmer errors stay panics: the aligner's invariants are
/// not recoverable conditions, and turning them into `Err` would only push the crash further from
/// its cause. The CLI wraps these in `anyhow` for context.
#[derive(Debug, Error)]
pub enum Error {
    /// Underlying I/O failure: file missing, unreadable, short read, disk full on the SAM writer.
    ///
    /// Holds the original [`std::io::Error`] unchanged, so the OS errno survives to the CLI. The
    /// `#[from]` means any `io::Error` in a `Result`-returning function converts with `?`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A malformed or unexpected index file: bad magic, truncated `.bwt`/`.pac`/`.ann`, a field
    /// whose value contradicts the header.
    ///
    /// Holds a human-readable description naming what was expected and what was found. Raised by
    /// the loaders in `bwa-index`; it means the index must be rebuilt, not that the read is bad.
    #[error("index format error: {0}")]
    IndexFormat(String),
    /// A malformed FASTQ/FASTA input: missing `@`/`>`, SEQ and QUAL of differing length, a record
    /// truncated at end of file.
    ///
    /// Holds a description that normally names the offending record. Raised by the readers in
    /// `bwa-io`, i.e. it is the user's input at fault rather than the index.
    #[error("sequence input error: {0}")]
    Fastq(String),
    /// Anything else, with context: the escape hatch for failures that fit none of the above.
    ///
    /// Holds the full message, already phrased for the end user, since the `Display` impl adds no
    /// prefix of its own. Prefer a specific variant when one applies.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias: `Result<T>` means `std::result::Result<T, Error>`.
///
/// `T` is whatever the fallible function yields on success. Shadows the prelude `Result` inside
/// this crate's modules and in downstream crates that `use bwa_core::Result`, which is why the two
/// argument form is spelled out explicitly wherever both are needed (see `rg::set_rg`).
pub type Result<T> = std::result::Result<T, Error>;

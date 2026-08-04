//! Crate-wide error type.
//!
//! # Rust mechanics used in this file
//!
//! This table is for readers who know the pipeline but not the language. Everything it glosses
//! recurs across the whole tree, so it is worth reading once: `Result`, `?` and this `Error` enum
//! are how every fallible operation in bwa-mem4 reports failure.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `enum` | a value that is exactly ONE of a listed set of shapes. Unlike a C enum, each shape may carry its own data, so `Error` is "either an I/O failure holding an OS error, or an index problem holding a message, or ...". |
//! | `String` | owned, growable text. The variants below hold one so the error message survives after the code that produced it has returned. |
//! | `Result<T, E>` | the return type of anything that can fail: either `Ok(value)` or `Err(error)`. Rust has no exceptions, so failure is an ordinary returned value that the caller cannot ignore by accident. |
//! | `?` | the propagation operator. Written after a fallible call, it means "if this failed, stop here and return that failure to MY caller; otherwise unwrap the success and carry on". It is what keeps error handling from burying the logic. |
//! | `#[derive(...)]` | asks the compiler to write an implementation for you. `Debug` generates the developer-facing printout; `Error` (from the `thiserror` crate) generates the boilerplate that makes this type a real error type. |
//! | `#[error("...")]` | `thiserror`'s instruction for the user-facing message of a variant. `{0}` interpolates that variant's first payload field. |
//! | `#[from]` | tells `thiserror` to also generate an automatic conversion INTO this variant. That conversion is what lets `?` turn a bare `std::io::Error` into an `Error::Io` with no code at the call site. |
//! | `pub type` | an alias, a second name for an existing type. It creates no new type and costs nothing at run time. |
//! | `<T>` | a type parameter, a placeholder filled in by whoever uses the alias. `Result<Vec<u8>>` and `Result<()>` are the same alias with different `T`. |
//!
//! Read the enum below as the four answers this codebase is willing to give to "why did that fail",
//! and the alias at the bottom as the shorthand every function signature in the tree uses.

use thiserror::Error;

/// Errors produced across bwa-mem4 crates.
///
/// Deliberately coarse. These are the failure modes a user can act on (bad index, bad FASTQ, disk),
/// not a taxonomy of internal states. Programmer errors stay panics: the aligner's invariants are
/// not recoverable conditions, and turning them into `Err` would only push the crash further from
/// its cause. The CLI wraps these in `anyhow` for context.
// Rust: `Debug` gives the developer printout (`{:?}`), `Error` is `thiserror`'s macro, which reads
// the `#[error(...)]` lines below and writes the user-facing `Display` text plus the standard error
// plumbing. Neither is hand-written; both are generated at compile time from these annotations.
#[derive(Debug, Error)]
pub enum Error {
    /// Underlying I/O failure: file missing, unreadable, short read, disk full on the SAM writer.
    ///
    /// Holds the original [`std::io::Error`] unchanged, so the OS errno survives to the CLI. The
    /// `#[from]` means any `io::Error` in a `Result`-returning function converts with `?`.
    //
    // Rust: the parentheses make this a variant that CARRIES a value, here the original OS error.
    // `{0}` in the message above prints that carried value. `#[from]` generates the conversion, so
    // writing `File::open(path)?` inside a function returning our `Result` silently produces an
    // `Error::Io`: that one annotation is why no call site in the tree does the wrapping by hand.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A malformed or unexpected index file: bad magic, truncated `.bwt`/`.pac`/`.ann`, a field
    /// whose value contradicts the header.
    ///
    /// Holds a human-readable description naming what was expected and what was found. Raised by
    /// the loaders in `bwa-index`; it means the index must be rebuilt, not that the read is bad.
    //
    // Rust: carries an owned `String` rather than a borrowed `&str`, because the error travels up
    // out of the function that built the message. A borrowed string would point at a buffer that
    // has already gone away by then, and the compiler rejects exactly that.
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
// Rust: `<T>` is a blank the caller fills in. `Result<Vec<Record>>` means "either a vector of
// records or an `Error`"; `Result<()>` means "either nothing in particular or an `Error`", `()`
// being the empty type used where C would return `void`. Because the alias pins the second slot to
// our `Error`, a signature only ever has to name the success type, which is why `-> Result<T>` with
// one argument is what you see everywhere in the tree.
pub type Result<T> = std::result::Result<T, Error>;

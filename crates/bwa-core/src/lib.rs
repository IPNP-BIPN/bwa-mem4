//! Types, constants and alignment options shared across bwa-mem3-rs.
//!
//! Everything here mirrors bwa-mem2 data structures so downstream crates can reason directly
//! against the reference source in `reference/bwa-mem2`.

pub mod dna;
pub mod error;
pub mod opt;
pub mod sysram;

pub use error::{Error, Result};
pub use opt::MemOpt;

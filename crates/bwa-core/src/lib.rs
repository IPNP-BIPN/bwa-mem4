//! Types, constants and alignment options shared across bwa-mem3-rs.
//!
//! Everything here mirrors bwa-mem2 data structures so downstream crates can reason directly
//! against the reference source in `reference/bwa-mem2`.

pub mod arch;
pub mod dna;
pub mod error;
pub mod opt;

pub use arch::{detect_simd, recommended_seed_engine, SeedEngine, Simd};
pub use error::{Error, Result};
pub use opt::MemOpt;

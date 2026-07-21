//! Types, constants and alignment options shared across bwa-mem3-rs.
//!
//! Everything here mirrors bwa-mem2 data structures so downstream crates can reason directly
//! against the reference source in `reference/bwa-mem2`.
//!
//! This is the leaf of the crate graph: it depends on no other bwa-mem3 crate, so the option struct
//! and the nucleotide tables can be shared by the CLI, the index, the aligner and the SIMD backends
//! without a cycle. Nothing here does I/O or alignment work.
//!
//! - [`opt`]: [`MemOpt`], the port of `mem_opt_t`, plus the `MEM_F_*` flag bits. Start here.
//! - [`dna`]: ASCII <-> 2-bit base encoding (`nst_nt4_table`).
//! - [`rg`]: process-wide `-R` read-group and `-C` copy-comment state, mirroring bwa's globals.
//! - [`error`]: the shared error type.
//! - [`sysram`]: host RAM detection, used only by the learned-index auto-select.

pub mod cpu;
pub mod dna;
pub mod error;
pub mod opt;
pub mod rg;
pub mod sysram;

// Re-exported at the crate root because these three are what downstream crates name constantly:
// `bwa_core::MemOpt` and `bwa_core::Result` read better than the module-qualified paths. Everything
// else stays behind its module, since the module name is the useful context there (`dna::nt4`,
// `rg::set_rg`).
pub use error::{Error, Result};
pub use opt::MemOpt;

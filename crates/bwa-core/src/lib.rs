//! Types, constants and alignment options shared across bwa-mem4.
//!
//! Everything here mirrors bwa-mem2 data structures so downstream crates can reason directly
//! against the reference source in `reference/bwa-mem2`.
//!
//! This is the leaf of the crate graph: it depends on no other bwa-mem4 crate, so the option struct
//! and the nucleotide tables can be shared by the CLI, the index, the aligner and the SIMD backends
//! without a cycle. Nothing here does I/O or alignment work.
//!
//! - [`opt`]: [`MemOpt`], the port of `mem_opt_t`, plus the `MEM_F_*` flag bits. Start here.
//! - [`dna`]: ASCII <-> 2-bit base encoding (`nst_nt4_table`).
//! - [`rg`]: process-wide `-R` read-group and `-C` copy-comment state, mirroring bwa's globals.
//! - [`error`]: the shared error type.
//! - [`sysram`]: host RAM detection, used only by the learned-index auto-select.
//!
//! # Rust mechanics used in this file
//!
//! This file is the crate's front door and contains no logic at all, only a map. The two keywords
//! below are the whole of it.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `//!` | a comment documenting the thing it is INSIDE (here, the whole crate), as opposed to `///`, which documents the item that follows it. That is why the file opens with `//!` prose and no code above it. |
//! | `lib.rs` | by convention the root of a library crate. It is not the entry point in the sense of a `main`; nothing runs here. Its job is to declare which files are part of the crate and what the outside world may see. |
//! | `pub mod name;` | "there is a file `name.rs` next to this one, make it part of this crate, and let outsiders reach it". Without `pub` the module would exist but be private to the crate. Without the line at all the file would simply not be compiled. |
//! | `pub use path;` | a re-export: a shortcut that makes an item reachable under a shorter name, in addition to its real one. It moves no code and costs nothing at run time. |
//! | `{A, B}` | brace grouping in a path, importing several names from the same place in one line. |
//!
//! So the six `pub mod` lines say what this crate is made of, and the three `pub use` lines say
//! which handful of names are common enough to deserve a shortcut at the top level.

pub mod alloc_probe;
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

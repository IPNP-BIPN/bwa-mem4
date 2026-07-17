//! FMD index construction and loading.
//!
//! Phase 0 implements only reference-metadata parsing (`.ann`/`.amb`) needed for the SAM header.
//! Index construction (`build`) and the FM traversal (`fmindex`) arrive in phases 1-2.

pub mod bntseq;
pub mod build;
pub mod fmindex;
pub mod lisa;
pub mod packed;
pub mod rand48;
pub mod rmi;
pub mod sais;

pub use bntseq::{Amb, BntSeq, Contig};
pub use build::build_index;
pub use fmindex::{traffic, FmIndex, Smem};
pub use lisa::LearnedSa;

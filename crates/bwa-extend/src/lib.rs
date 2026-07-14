//! Seed extension via banded Smith-Waterman.
//!
//! The scalar seed-extension kernel (`sw::ksw_extend2`) is the bit-identity source of truth;
//! NEON/Metal backends must reproduce its integer results.

pub mod sw;

pub use sw::{ksw_align2, ksw_extend2, ksw_global2, ExtendResult, KswAlignResult};

/// A batched banded Smith-Waterman backend.
///
/// The scalar backend is authoritative; SIMD (NEON) and GPU (Metal) backends must produce
/// byte-identical integer results to it.
pub trait SwBackend {
    /// Short backend name, e.g. `"scalar"`, `"neon"`, `"metal"`.
    fn name(&self) -> &'static str;
}

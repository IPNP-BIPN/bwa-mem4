//! `BWA4_STAGE_ALLOC`: the binary's half of the allocation probe.
//!
//! The counting itself lives in [`bwa_core::alloc_probe`], because three crates have to agree on
//! one bucket namespace: this one tags the pipeline stages, `bwa-mem4-mem` tags the sub-phases of
//! `align`, and `bwa-core` is the only crate both depend on. What stays here is what only the
//! binary knows: the environment variable, whether the counting allocator was actually compiled in,
//! and the mapping from [`Stage`] to a bucket index.
//!
//! See [`bwa_core::alloc_probe`] for what the probe measures and why issue #25 needs a counter
//! inside the allocator rather than another peak-RSS snapshot.

use crate::stage_time::Stage;
use bwa_core::alloc_probe;

pub use bwa_core::alloc_probe::{count_batch, dump, enabled, mark_baseline, note_bases};
pub use bwa_core::alloc_probe::{serialize_batches, Counting, StageGuard};

/// Arm the probe if `BWA4_STAGE_ALLOC` is set in the environment.
///
/// Call once, as early in `main` as possible: allocations before it are uncounted, which for the
/// `mem` path is clap's parsing and nothing else.
///
/// Two modes, because they answer different questions and one run cannot answer both:
///
/// - `BWA4_STAGE_ALLOC=1` (or anything else): one batch in flight, so the per-stage columns are
///   exact. The peak-live column is then the footprint of ONE batch.
/// - `BWA4_STAGE_ALLOC=overlap`: the shipped two-batches-in-flight pipeline. The peak and total
///   rows are the real ones; the per-stage split is a blend of whatever two batches were doing at
///   the same time and must not be quoted.
pub fn init() {
    let Some(mode) = std::env::var_os("BWA4_STAGE_ALLOC") else {
        return;
    };
    // The counting allocator is only installed by `main.rs` under the `stage-alloc` feature, which
    // is off by default (it costs a measured +0.5% even disarmed). Arming without it would print a
    // table of zeros and serialise the pipeline for nothing, so say so instead.
    if !cfg!(feature = "stage-alloc") {
        eprintln!(
            "[stage-alloc] BWA4_STAGE_ALLOC is set but this binary has no counting allocator; \
             rebuild with `cargo build --release -p bwa-mem4 --features stage-alloc`. Ignored."
        );
        return;
    }
    alloc_probe::arm(mode != *"overlap");
}

/// Credit every subsequent allocation on this thread to `stage`, until the returned guard drops.
///
/// Called from [`crate::stage_time::measure`], so the two probes cannot disagree about where a
/// stage begins: one instrumentation point, two measurements.
///
/// # Parameters
///
/// - `stage`: the stage the calling thread is entering.
///
/// # Returns
///
/// A guard that restores the previous bucket. Bind it (`let _tag = ...`), do not discard it with
/// `_`, or the tag is restored immediately.
#[inline]
pub fn enter(stage: Stage) -> StageGuard {
    // The discriminants ARE the bucket indices for the first nine buckets; the test below is what
    // keeps that true.
    alloc_probe::enter(stage as u8)
}

/// Tag the calling thread as the FASTQ reader, so its allocations are credited to `reader` whatever
/// stage the aligner is in.
pub fn set_role_reader() {
    alloc_probe::set_role(alloc_probe::ROLE_READER);
}

/// Tag the calling thread as the SAM/BAM writer, for the same reason as [`set_role_reader`].
pub fn set_role_writer() {
    alloc_probe::set_role(alloc_probe::ROLE_WRITER);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `enter` casts a `Stage` straight to a bucket index, so the two name lists must agree
    /// entry for entry. If they ever drift, one probe's `align` row is the other's `rescue`.
    #[test]
    fn stage_discriminants_are_bucket_indices() {
        let stages = crate::stage_time::names();
        assert_eq!(stages.len(), alloc_probe::N_STAGES);
        for (i, name) in stages.iter().enumerate() {
            assert_eq!(*name, alloc_probe::BUCKET_NAMES[i]);
        }
    }
}

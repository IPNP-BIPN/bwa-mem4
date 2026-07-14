//! NEON (Apple Silicon / AArch64) backend for the batched banded Smith-Waterman kernel (phase 9a).
//!
//! # Status: integration seam in place, NEON DP kernel pending
//!
//! [`NeonBackend`] implements [`bwa_extend::SwBackend`] and **currently delegates to the scalar
//! kernel** ([`bwa_extend::ksw_extend2`]). It therefore passes
//! [`bwa_extend::assert_backend_matches_scalar`] and `assert_backend_batch_matches_scalar` trivially,
//! which keeps CI green and gives the NEON kernel a verified drop-in point: as the vectorized
//! `extend_batch` is filled in, those same gates prove it stays byte-identical to scalar.
//!
//! # Porting plan (from nh13's `fg-labs/bwa-mem3`, credited in `DEPENDENCIES.md`)
//!
//! bwa-mem2 / bwa-mem3-cpp accelerate seed extension by **inter-sequence batching**: `bandedSWA`
//! packs `SIMD_WIDTH` independent alignments across NEON lanes (8 for int16, 16 for int8 on a
//! 128-bit register), one alignment per lane. Each lane runs the *same* integer recurrence as the
//! scalar [`bwa_extend::ksw_extend2`], so the batched result is byte-identical by construction.
//!
//! The real kernel (to implement here, behind `#[cfg(target_arch = "aarch64")]` with a scalar
//! fallback) vectorizes [`SwBackend::extend_batch`]:
//! - lay out per-lane query/target/`h0`, padding to the batch's max length and masking inactive
//!   lanes and out-of-band cells;
//! - carry the `H`/`E`/`F` recurrence and the affine-gap updates across lanes (int16 lanes, with an
//!   int8 fast path where scores fit, matching `SIMD_WIDTH8`/`SIMD_WIDTH16`);
//! - reproduce, per lane, the band tightening, z-drop termination, and max tracking
//!   (`qle`/`tle`/`gtle`/`gscore`/`max_off`) exactly.
//! - port nh13's native NEON tweaks: the `vbslq` blendv in `bandedSWA` and the native NEON `kswv`.
//!
//! Acceptance: `assert_backend_batch_matches_scalar(&NeonBackend)` must pass, then a measured
//! speedup over the scalar backend on a realistic batch.

use bwa_extend::{ksw_extend2, ExtendResult, SwBackend};

/// The NEON seed-extension backend. See the module docs: it delegates to the scalar kernel today
/// and is the drop-in point for the lane-parallel NEON DP (phase 9a).
#[derive(Debug, Default, Clone, Copy)]
pub struct NeonBackend;

impl SwBackend for NeonBackend {
    fn name(&self) -> &'static str {
        "neon"
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w: i32,
        end_bonus: i32,
        zdrop: i32,
        h0: i32,
    ) -> ExtendResult {
        // TODO(phase9a): NEON lane-parallel DP. Delegates to scalar until byte-identity is verified.
        ksw_extend2(
            query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        )
    }

    // extend_batch uses the default (loops `extend`) for now; the NEON kernel will override it to
    // process SIMD_WIDTH lanes at once.
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::{assert_backend_batch_matches_scalar, assert_backend_matches_scalar};

    #[test]
    fn neon_backend_matches_scalar() {
        // Passes today via delegation; remains the byte-identity gate once the NEON DP lands.
        assert_backend_matches_scalar(&NeonBackend);
        assert_backend_batch_matches_scalar(&NeonBackend);
        assert_eq!(NeonBackend.name(), "neon");
    }
}

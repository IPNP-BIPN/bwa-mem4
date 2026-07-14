//! Seed extension via banded Smith-Waterman.
//!
//! The scalar seed-extension kernel (`sw::ksw_extend2`) is the bit-identity source of truth;
//! NEON/Metal backends must reproduce its integer results.

pub mod sw;

pub use sw::{ksw_align2, ksw_extend2, ksw_global2, ExtendResult, KswAlignResult};

/// A banded Smith-Waterman seed-extension backend.
///
/// The scalar backend ([`ScalarBackend`], delegating to [`ksw_extend2`]) is authoritative; SIMD
/// (NEON) and GPU (Metal) backends must return **integer-identical** [`ExtendResult`]s to it for
/// every input, so byte-identity of the SAM output is preserved. Use
/// [`assert_backend_matches_scalar`] as the acceptance gate when adding a backend.
pub trait SwBackend {
    /// Short backend name, e.g. `"scalar"`, `"neon"`, `"metal"`.
    fn name(&self) -> &'static str;

    /// One banded local seed extension, mirroring [`ksw_extend2`] exactly.
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
    ) -> ExtendResult;
}

/// The authoritative scalar backend: `extend` delegates straight to [`ksw_extend2`]. Every other
/// backend is validated against this one.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarBackend;

impl SwBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
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
        ksw_extend2(
            query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        )
    }
}

/// Acceptance gate for any non-scalar [`SwBackend`]: assert `backend.extend` returns exactly the
/// same [`ExtendResult`] as [`ksw_extend2`] over a deterministic sweep of random DNA seed
/// extensions (varied lengths, band widths, gap/z-drop settings). Panics on the first mismatch.
/// NEON (phase 9a) and Metal (phase 9b) backends must pass this.
pub fn assert_backend_matches_scalar<B: SwBackend>(backend: &B) {
    // 5x5 bwa-style scoring matrix (match a, mismatch -b, N = -1).
    let (a, b) = (1i8, 4i8);
    let mut mat = vec![0i8; 25];
    let mut k = 0;
    for i in 0..4 {
        for j in 0..4 {
            mat[k] = if i == j { a } else { -b };
            k += 1;
        }
        mat[k] = -1;
        k += 1;
    }
    for _ in 0..5 {
        mat[k] = -1;
        k += 1;
    }

    // Deterministic LCG for reproducible random cases (no extra deps).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    for case in 0..2000u32 {
        let qlen = 1 + (next() % 60) as usize;
        let tlen = 1 + (next() % 60) as usize;
        let query: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
        let target: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
        let w = 1 + (next() % 30) as i32;
        let zdrop = (next() % 120) as i32;
        let end_bonus = (next() % 10) as i32;
        // ksw_extend2 requires a positive initial score (extension starts from a seed).
        let h0 = 1 + (next() % 20) as i32;
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

        let expected = ksw_extend2(
            &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        );
        let got = backend.extend(
            &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        );
        assert_eq!(
            got,
            expected,
            "backend {:?} diverged from scalar on case {case} (qlen={qlen} tlen={tlen} w={w} zdrop={zdrop} end_bonus={end_bonus} h0={h0})",
            backend.name()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_backend_matches_kernel() {
        // The harness itself must pass for the authoritative backend (and becomes the NEON/Metal
        // acceptance gate). ScalarBackend delegates, so this also self-checks the harness runs.
        assert_backend_matches_scalar(&ScalarBackend);
        assert_eq!(ScalarBackend.name(), "scalar");
    }
}

//! Seed extension via banded Smith-Waterman.
//!
//! The scalar seed-extension kernel (`sw::ksw_extend2`) is the bit-identity source of truth;
//! NEON/Metal backends must reproduce its integer results.

pub mod sw;

pub use sw::{
    ksw_align2, ksw_align2_with, ksw_extend2, ksw_global2, local_fwd_dp, local_fwd_finish,
    ExtendResult, KswAlignResult, LocalFwdDp, LocalFwdKernel, ScalarFwd,
};

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

    /// Extend a batch of seed extensions that share the same scoring (`m`, `mat`, gap penalties,
    /// band `w`, `end_bonus`, `zdrop`) but differ in query/target/`h0`. This is the entry point a
    /// SIMD backend accelerates: bwa-mem2's `bandedSWA` packs `SIMD_WIDTH` such alignments across
    /// vector lanes (8 for int16, 16 for int8), one independent alignment per lane. The result at
    /// index `k` must equal `extend(jobs[k]...)` exactly. The default loops the scalar `extend`.
    #[allow(clippy::too_many_arguments)]
    fn extend_batch(
        &self,
        jobs: &[ExtendJob],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w: i32,
        end_bonus: i32,
        zdrop: i32,
    ) -> Vec<ExtendResult> {
        jobs.iter()
            .map(|j| {
                self.extend(
                    j.query, j.target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                    j.h0,
                )
            })
            .collect()
    }
}

/// One seed extension in a batch: the per-alignment inputs that vary across lanes (the scoring
/// scheme and band are shared, passed to [`SwBackend::extend_batch`]).
#[derive(Debug, Clone, Copy)]
pub struct ExtendJob<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
    pub h0: i32,
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

/// Batch-mode acceptance gate: assert `backend.extend_batch` returns, for every job, exactly the
/// [`ksw_extend2`] result. Batches share scoring/band (as the API requires) and vary in
/// query/target/`h0`, across a range of batch sizes (including sizes that don't fill a SIMD width).
/// A SIMD backend must pass this in addition to [`assert_backend_matches_scalar`].
pub fn assert_backend_batch_matches_scalar<B: SwBackend>(backend: &B) {
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

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    // Many rounds, each a batch with shared scoring/band (per the contract) but varied content,
    // sizes straddling common SIMD widths (8, 16) with partial tails, and varied gap penalties.
    for round in 0..200u32 {
        let batch_size = *[1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 20, 31]
            .get((next() % 12) as usize)
            .unwrap();
        let w = 1 + (next() % 30) as i32;
        let zdrop = (next() % 120) as i32;
        let end_bonus = (next() % 10) as i32;
        // Vary affine gaps too (kept >= 1 and with o+e sane).
        let o_del = 4 + (next() % 4) as i32;
        let e_del = 1 + (next() % 2) as i32;
        let o_ins = 4 + (next() % 4) as i32;
        let e_ins = 1 + (next() % 2) as i32;

        let queries: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| {
                let qlen = 1 + (next() % 80) as usize;
                (0..qlen).map(|_| (next() % 4) as u8).collect()
            })
            .collect();
        let targets: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| {
                let tlen = 1 + (next() % 80) as usize;
                (0..tlen).map(|_| (next() % 4) as u8).collect()
            })
            .collect();
        let h0s: Vec<i32> = (0..batch_size).map(|_| 1 + (next() % 20) as i32).collect();

        let jobs: Vec<ExtendJob> = (0..batch_size)
            .map(|i| ExtendJob {
                query: &queries[i],
                target: &targets[i],
                h0: h0s[i],
            })
            .collect();

        let got = backend.extend_batch(
            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
        );
        assert_eq!(got.len(), batch_size);
        for (i, g) in got.iter().enumerate() {
            let expected = ksw_extend2(
                &queries[i],
                &targets[i],
                5,
                &mat,
                o_del,
                e_del,
                o_ins,
                e_ins,
                w,
                end_bonus,
                zdrop,
                h0s[i],
            );
            assert_eq!(
                *g, expected,
                "backend {:?} extend_batch diverged at round {round} job {i}/{batch_size} (w={w} zdrop={zdrop} gaps={o_del},{e_del},{o_ins},{e_ins})",
                backend.name()
            );
        }
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
        assert_backend_batch_matches_scalar(&ScalarBackend);
        assert_eq!(ScalarBackend.name(), "scalar");
    }
}

//! NEON (Apple Silicon / AArch64) backend for the batched banded Smith-Waterman kernel (phase 9a).
//!
//! # Status: NEON int16x8 DP kernel live (step 2b-ii)
//!
//! [`NeonBackend`] implements [`bwa_extend::SwBackend`]. `extend` (single) still delegates to the
//! scalar [`bwa_extend::ksw_extend2`]; `extend_batch` runs the vectorized lane-parallel DP in
//! [`batched`]. Both the shared gate ([`bwa_extend::assert_backend_batch_matches_scalar`]) and a
//! read-sized property test prove the batched path is **byte-identical** to scalar, and the
//! `bench_batch` example measures the speedup (~1.4x on mixed lengths, ~2.1x when lengths are
//! uniform, over the scalar per-lane loop).
//!
//! # Design (following bwa-mem2 `bandedSWA` and nh13's `fg-labs/bwa-mem3`, credited in `DEPENDENCIES.md`)
//!
//! bwa-mem2 / bwa-mem3-cpp accelerate seed extension by **inter-sequence batching**: `bandedSWA`
//! packs `SIMD_WIDTH` independent alignments across NEON lanes (8 for int16, 16 for int8 on a
//! 128-bit register), one alignment per lane, using an SoA layout `[column*SIMD_WIDTH + lane]`. Each
//! lane runs the *same* integer recurrence as the scalar [`bwa_extend::ksw_extend2`], so the batched
//! result is byte-identical by construction.
//!
//! [`batched::batched_extend`] reproduces this: 8 int16 lanes in the same SoA layout, `vbslq` blendv
//! for the per-lane band mask (matching nh13's `neon_utils.h` `NEON_BLENDV`), and per-lane band
//! tightening / z-drop / max tracking. It carries the `H`/`E`/`F` recurrence with non-saturating
//! `vaddq_s16`/`vsubq_s16`, exact because a local extension's cell values are bounded well inside the
//! int16 range (nh13's kernel uses saturating ops with `MAX_SEQ_LEN16` length-binning for the same
//! guarantee). A scalar reference ([`batched::batched_extend_scalar`]) is the portable fallback.

use bwa_extend::{ksw_extend2, ExtendJob, ExtendResult, SwBackend};

mod batched;

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
        // Step 2b-ii: NEON int16x8 lane-parallel DP (byte-identical to ksw_extend2 per job), with a
        // scalar fallback for non-aarch64 or out-of-int16-range batches.
        batched::batched_extend(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::{assert_backend_batch_matches_scalar, assert_backend_matches_scalar};

    #[test]
    fn neon_backend_matches_scalar() {
        // The shared gate (qlen/tlen <= 80). Exercises the NEON int16 kernel on aarch64.
        assert_backend_matches_scalar(&NeonBackend);
        assert_backend_batch_matches_scalar(&NeonBackend);
        assert_eq!(NeonBackend.name(), "neon");
    }

    /// Byte-identity at realistic short-read sizes: query up to a full 150 bp read side and
    /// reference windows up to ~300 bp, i.e. the range the aligner pipeline actually feeds
    /// `extend_batch`. Confirms the int16 kernel stays exact well above the shared gate's 80 bp.
    #[test]
    fn neon_batch_matches_scalar_read_sized() {
        use bwa_extend::{ksw_extend2, ExtendJob};
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

        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        for round in 0..120u32 {
            let batch_size = *[1usize, 5, 8, 9, 16, 17, 24, 40]
                .get((next() % 8) as usize)
                .unwrap();
            let w = 50 + (next() % 120) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            let o_del = 4 + (next() % 4) as i32;
            let e_del = 1 + (next() % 2) as i32;
            let o_ins = 4 + (next() % 4) as i32;
            let e_ins = 1 + (next() % 2) as i32;

            // Correlated query/target so the DP runs deep (not an all-mismatch early z-drop).
            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch_size {
                let qlen = 60 + (next() % 200) as usize; // up to ~260
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                let tlen = qlen + (next() % 60) as usize;
                let mut t: Vec<u8> = Vec::with_capacity(tlen);
                let mut qi = 0usize;
                while t.len() < tlen {
                    if qi < q.len() && next() % 100 >= 4 {
                        t.push(q[qi]);
                        qi += 1;
                    } else {
                        t.push((next() % 4) as u8);
                        if next() % 2 == 0 {
                            qi += 1;
                        }
                    }
                }
                queries.push(q);
                targets.push(t);
                h0s.push(20 + (next() % 30) as i32);
            }
            let jobs: Vec<ExtendJob> = (0..batch_size)
                .map(|i| ExtendJob {
                    query: &queries[i],
                    target: &targets[i],
                    h0: h0s[i],
                })
                .collect();

            let got = NeonBackend.extend_batch(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
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
                    *g,
                    expected,
                    "NEON diverged at round {round} job {i}/{batch_size} qlen={} tlen={} (w={w})",
                    queries[i].len(),
                    targets[i].len(),
                );
            }
        }
    }

    /// Exercise the ungapped diagonal HIT fast path directly: many perfect diagonals (which trigger
    /// the closed form) plus near-diagonals (0-2 substitutions/indels) and short targets, asserting
    /// the batched NEON result equals `ksw_extend2` for every job. Guards the HIT closed form against
    /// any divergence from the real DP on the exact cases it fires.
    #[test]
    fn neon_ungapped_hit_matches_scalar() {
        use bwa_extend::{ksw_extend2, ExtendJob};
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

        let mut state = 0x51ED_5EED_D1A6_0000u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        for round in 0..300u32 {
            let batch = *[1usize, 4, 8, 13, 16, 24]
                .get((next() % 6) as usize)
                .unwrap();
            let w = 1 + (next() % 150) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch {
                let qlen = 1 + (next() % 200) as usize;
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                // Target = query diagonal, then a mode: perfect / longer / shorter / few edits / N.
                let mode = next() % 5;
                let extra = (next() % 40) as usize;
                let mut t: Vec<u8> = q.clone();
                match mode {
                    0 => t.extend((0..extra).map(|_| (next() % 4) as u8)), // perfect, longer target
                    1 => {}                                                // perfect, equal length
                    2 => {
                        t.truncate(qlen.saturating_sub(1 + (next() % 3) as usize)); // shorter target
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                    3 => {
                        // 1-2 substitutions
                        for _ in 0..(1 + next() % 2) {
                            if !t.is_empty() {
                                let p = (next() as usize) % t.len();
                                t[p] = ((t[p] + 1 + (next() % 3) as u8) % 4) as u8;
                            }
                        }
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                    _ => {
                        // inject an N somewhere
                        if !t.is_empty() {
                            let p = (next() as usize) % t.len();
                            t[p] = 4;
                        }
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                }
                queries.push(q);
                targets.push(t);
                h0s.push(1 + (next() % 40) as i32);
            }
            let jobs: Vec<ExtendJob> = (0..batch)
                .map(|i| ExtendJob {
                    query: &queries[i],
                    target: &targets[i],
                    h0: h0s[i],
                })
                .collect();
            let got = NeonBackend.extend_batch(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
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
                    *g,
                    expected,
                    "ungapped HIT diverged at round {round} job {i} qlen={} tlen={} h0={}",
                    queries[i].len(),
                    targets[i].len(),
                    h0s[i],
                );
            }
        }
    }
}

//! Vectorized local Smith-Waterman for **mate rescue** (`ksw_align2`), the `kswv` equivalent.
//!
//! Mate rescue (`bwa_mem::pe::mem_matesw`) is the single largest CPU cost in `mem` on real paired
//! reads, and it runs the scalar full-matrix `ksw_local_fwd` once per rescue attempt. This module
//! provides a **striped** (Farrar) NEON kernel for the forward DP pass. Only the recurrence is
//! vectorized: it emits the same [`LocalFwdDp`] (per-row max + the `H` row at the best target end)
//! that the scalar [`bwa_extend::local_fwd_dp`] does, and the fiddly tail (`qe` argmax, `score2`,
//! `KSW_XSTART` start recovery) is the shared scalar [`bwa_extend::local_fwd_finish`] /
//! [`bwa_extend::ksw_align2_with`] — so byte-identity to the scalar oracle cannot drift.
//!
//! Range: the `u8` kernel (16 lanes) runs when the score ceiling fits (`qlen * max_sc < 256`, always
//! true for ≤150 bp Illumina with `a=1`) and the matrix is the uniform bwa DNA matrix; anything else
//! falls back to the scalar `local_fwd_dp`, so every input is handled correctly.

use bwa_extend::{ksw_align2_with, local_fwd_dp, KswAlignResult, LocalFwdDp, LocalFwdKernel};

/// The NEON forward-DP kernel: striped `u8` when eligible, scalar otherwise.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeonFwd;

/// The uniform bwa DNA matrix: diagonal `a` (0..4), off-diagonal `mm`, row/col 4 (`N`) = `npen`.
fn is_uniform_dna(mat: &[i8], m: usize) -> bool {
    if m == 0 || mat.len() < m * m {
        return false;
    }
    let a = mat[0];
    let mm = mat[1];
    let npen = mat[m - 1];
    for i in 0..m {
        for j in 0..m {
            let v = mat[i * m + j];
            let want = if i == m - 1 || j == m - 1 {
                npen
            } else if i == j {
                a
            } else {
                mm
            };
            if v != want {
                return false;
            }
        }
    }
    a >= 0 && mm <= 0 && npen <= 0
}

impl LocalFwdKernel for NeonFwd {
    #[allow(clippy::too_many_arguments)]
    fn dp(
        &self,
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        endsc: i32,
    ) -> LocalFwdDp {
        let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
        let qlen = query.len();
        // u8 eligibility: uniform DNA matrix + score ceiling and coordinates fit unsigned 8-bit.
        // (Every cell value is <= the best local score <= qlen*max_sc, so this bound is exact.)
        let u8_ok = qlen > 0
            && !target.is_empty()
            && is_uniform_dna(mat, m)
            && qlen.saturating_mul(max_sc as usize) < 256;
        #[cfg(target_arch = "aarch64")]
        if u8_ok && std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: `neon` is available; the kernel only does fixed-width NEON ops on owned buffers.
            return unsafe {
                neon::striped_local_fwd_u8(query, target, m, mat, o_del, e_del, o_ins, e_ins, endsc)
            };
        }
        let _ = u8_ok;
        local_fwd_dp(query, target, m, mat, o_del, e_del, o_ins, e_ins, endsc)
    }
}

/// Vectorized `ksw_align2` for mate rescue. Byte-identical to [`ksw_align2`]; routes through the
/// striped NEON forward DP where eligible and the scalar DP otherwise.
#[allow(clippy::too_many_arguments)]
pub fn kswv(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> KswAlignResult {
    ksw_align2_with(
        &NeonFwd, query, target, m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc,
    )
}

// Keep the scalar entry point reachable for callers/tests that want to force it.
#[allow(unused_imports)]
pub use bwa_extend::ksw_align2 as kswv_scalar;

#[cfg(target_arch = "aarch64")]
mod neon {
    use bwa_extend::LocalFwdDp;
    use std::arch::aarch64::*;

    /// Striped (Farrar) local-SW forward pass in unsigned `u8` (16 lanes). Query position `p` lives at
    /// stripe `p % segLen`, lane `p / segLen`. Emits per-row max (`row_imax`) and `gmax`/`te`/`hmax_col`
    /// with the `endsc` early-stop, matching `bwa_extend::local_fwd_dp` exactly in the u8 range.
    ///
    /// # Safety
    /// Requires the `neon` target feature (checked by the caller).
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn striped_local_fwd_u8(
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        endsc: i32,
    ) -> LocalFwdDp {
        const LANES: usize = 16;
        let qlen = query.len();
        let tlen = target.len();
        let seg_len = qlen.div_ceil(LANES);

        let a = mat[0] as u8;
        let mm_mag = (-i32::from(mat[1])) as u8;
        let npen_mag = (-i32::from(mat[m - 1])) as u8;
        let a_v = vdupq_n_u8(a);
        let mm_v = vdupq_n_u8(mm_mag);
        let npen_v = vdupq_n_u8(npen_mag);
        let amb_v = vdupq_n_u8(4); // code 4 = N
        let zero_v = vdupq_n_u8(0);
        let oe_del_v = vdupq_n_u8((o_del + e_del) as u8);
        let oe_ins_v = vdupq_n_u8((o_ins + e_ins) as u8);
        let e_del_v = vdupq_n_u8(e_del as u8);
        let e_ins_v = vdupq_n_u8(e_ins as u8);

        // Striped query codes (padding lanes past `qlen` get code 5 => treated as N, so any real
        // target base mismatches them), plus a per-stripe mask of the real (`p < qlen`) lanes. The
        // cross-lane diagonal shift keeps padding leaks confined to padding cells, so real H is exact;
        // the mask keeps padding out of the per-row max (`row_imax`), which feeds `score2`.
        let mut qcode: Vec<uint8x16_t> = vec![zero_v; seg_len];
        let mut real_mask: Vec<uint8x16_t> = vec![zero_v; seg_len];
        {
            let mut buf = [5u8; LANES];
            let mut msk = [0u8; LANES];
            for (v, slot) in qcode.iter_mut().enumerate() {
                for (l, (b, mk)) in buf.iter_mut().zip(msk.iter_mut()).enumerate() {
                    let p = l * seg_len + v;
                    let real = p < qlen;
                    *b = if real { query[p] } else { 5 };
                    *mk = if real { 0xff } else { 0x00 };
                }
                *slot = vld1q_u8(buf.as_ptr());
                real_mask[v] = vld1q_u8(msk.as_ptr());
            }
        }

        // H (store/load, double-buffered) and E, one 16-lane vector per stripe.
        let mut h_store: Vec<uint8x16_t> = vec![zero_v; seg_len];
        let mut h_load: Vec<uint8x16_t> = vec![zero_v; seg_len];
        let mut e_arr: Vec<uint8x16_t> = vec![zero_v; seg_len];

        let mut gmax = 0i32;
        let mut te = -1i32;
        let mut hmax_col = vec![0i32; qlen];
        let mut row_imax: Vec<i32> = Vec::with_capacity(tlen);

        for (i, &t) in target.iter().enumerate() {
            let t_v = vdupq_n_u8(t);
            let t_is_n = vcgeq_u8(t_v, amb_v);

            // diagonal H(i-1, p-1): prev row's last stripe shifted up one lane, 0 into lane 0.
            let mut h_diag = vextq_u8(zero_v, h_store[seg_len - 1], 15);
            std::mem::swap(&mut h_store, &mut h_load); // h_load = previous row's H
            let mut f_v = zero_v;
            let mut colmax_v = zero_v;

            for v in 0..seg_len {
                let q_v = qcode[v];
                let is_eq = vceqq_u8(t_v, q_v);
                let is_n = vorrq_u8(t_is_n, vcgeq_u8(q_v, amb_v));
                let pos = vbslq_u8(is_n, zero_v, vbslq_u8(is_eq, a_v, zero_v));
                let neg = vbslq_u8(is_n, npen_v, vbslq_u8(is_eq, zero_v, mm_v));
                // h = max(0, h_diag + score); saturating u8 gives the local floor.
                let mut h = vqsubq_u8(vqaddq_u8(h_diag, pos), neg);
                let e_v = e_arr[v];
                h = vmaxq_u8(h, e_v);
                h = vmaxq_u8(h, f_v);
                h_store[v] = h;
                colmax_v = vmaxq_u8(colmax_v, vandq_u8(h, real_mask[v]));

                // E' = max(E - e_del, H - oe_del); F' = max(F - e_ins, H - oe_ins)
                let h_oe_del = vqsubq_u8(h, oe_del_v);
                e_arr[v] = vmaxq_u8(vqsubq_u8(e_v, e_del_v), h_oe_del);
                let h_oe_ins = vqsubq_u8(h, oe_ins_v);
                f_v = vmaxq_u8(vqsubq_u8(f_v, e_ins_v), h_oe_ins);

                h_diag = h_load[v]; // H(i-1, p) becomes the next stripe's diagonal
            }

            // Lazy-F: propagate F across the stripe boundary until a full pass raises no H.
            'lazy: for _ in 0..=seg_len {
                f_v = vextq_u8(zero_v, f_v, 15);
                let mut changed = 0u8;
                for v in 0..seg_len {
                    let h_old = h_store[v];
                    let h = vmaxq_u8(h_old, f_v);
                    changed |= vmaxvq_u8(vcgtq_u8(h, h_old));
                    h_store[v] = h;
                    colmax_v = vmaxq_u8(colmax_v, vandq_u8(h, real_mask[v]));
                    f_v = vmaxq_u8(vqsubq_u8(f_v, e_ins_v), vqsubq_u8(h, oe_ins_v));
                }
                if changed == 0 {
                    break 'lazy;
                }
            }

            let imax = vmaxvq_u8(colmax_v) as i32;
            row_imax.push(imax);
            if imax > gmax {
                gmax = imax;
                te = i as i32;
                // destripe H into hmax_col for the real query positions.
                let mut buf = [0u8; LANES];
                for (v, &hv) in h_store.iter().enumerate() {
                    vst1q_u8(buf.as_mut_ptr(), hv);
                    for (l, &b) in buf.iter().enumerate() {
                        let p = l * seg_len + v;
                        if p < qlen {
                            hmax_col[p] = i32::from(b);
                        }
                    }
                }
                if gmax >= endsc {
                    break;
                }
            }
        }

        LocalFwdDp {
            gmax,
            te,
            hmax_col,
            row_imax,
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

    /// bwa's uniform DNA matrix for the given (a, mm, npen).
    fn dna_matrix(a: i8, mm: i8, npen: i8) -> Vec<i8> {
        let m = 5;
        let mut mat = vec![0i8; m * m];
        for i in 0..m {
            for j in 0..m {
                mat[i * m + j] = if i == m - 1 || j == m - 1 {
                    npen
                } else if i == j {
                    a
                } else {
                    mm
                };
            }
        }
        mat
    }

    // Small deterministic LCG so the test needs no rng dependency and is reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 16
        }
        fn range(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn rand_seq(rng: &mut Lcg, len: usize, n_pct: usize) -> Vec<u8> {
        (0..len)
            .map(|_| {
                if n_pct > 0 && rng.range(100) < n_pct {
                    4
                } else {
                    rng.range(4) as u8
                }
            })
            .collect()
    }

    #[test]
    fn kswv_matches_scalar_ksw_align2_randomized() {
        // bwa mem defaults: a=1, mm=-4, npen=-1, o_del=o_ins=6, e_del=e_ins=1.
        let mat = dna_matrix(1, -4, -1);
        let (m, a) = (5usize, 1i32);
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let minsc = 17; // ~ min_seed_len * a, like mem_matesw
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let mut checked = 0;
        for _ in 0..4000 {
            // qlen small enough that qlen*a < 256 (u8 path); target a bit longer, like a rescue window.
            let qlen = 1 + rng.range(200);
            let tlen = 1 + rng.range(400);
            let n_pct = if rng.range(3) == 0 { 8 } else { 0 };
            let q = rand_seq(&mut rng, qlen, n_pct);
            let t = rand_seq(&mut rng, tlen, n_pct);
            let want = ksw_align2(&q, &t, m, &mat, o_del, e_del, o_ins, e_ins, minsc, a);
            let got = kswv(&q, &t, m, &mat, o_del, e_del, o_ins, e_ins, minsc, a);
            assert_eq!(
                got, want,
                "mismatch qlen={qlen} tlen={tlen} n_pct={n_pct}\nq={q:?}\nt={t:?}"
            );
            checked += 1;
        }
        assert_eq!(checked, 4000);
    }

    #[test]
    fn kswv_matches_scalar_with_embedded_homology() {
        // Plant a copy of the query inside the target so rescue actually finds a strong hit.
        let mat = dna_matrix(1, -4, -1);
        let (m, a) = (5usize, 1i32);
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let minsc = 17;
        let mut rng = Lcg(0xdead_beef_0000_0001);
        for _ in 0..2000 {
            let qlen = 20 + rng.range(180);
            let q = rand_seq(&mut rng, qlen, 0);
            // target = random flank + (possibly mutated) query + random flank
            let lf = rng.range(150);
            let rf = rng.range(150);
            let mut t = rand_seq(&mut rng, lf, 0);
            let mut mid = q.clone();
            let nmut = rng.range(qlen / 4 + 1);
            for _ in 0..nmut {
                let p = rng.range(qlen);
                mid[p] = rng.range(4) as u8;
            }
            t.extend_from_slice(&mid);
            t.extend_from_slice(&rand_seq(&mut rng, rf, 0));
            let want = ksw_align2(&q, &t, m, &mat, o_del, e_del, o_ins, e_ins, minsc, a);
            let got = kswv(&q, &t, m, &mat, o_del, e_del, o_ins, e_ins, minsc, a);
            assert_eq!(
                got, want,
                "mismatch qlen={qlen} lf={lf} rf={rf} nmut={nmut}"
            );
        }
    }
}

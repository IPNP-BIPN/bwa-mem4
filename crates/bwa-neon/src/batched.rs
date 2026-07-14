//! Lane-batched banded Smith-Waterman seed extension (phase 9a).
//!
//! Processes several independent alignments in lockstep: a shared target-row loop `i` and a shared
//! query loop `j` over the union band, each lane masked to its own band `[beg, end)` and its own
//! termination (row all-zero or z-drop). Two implementations, both **byte-identical** to
//! [`bwa_extend::ksw_extend2`] run per lane:
//!
//! - [`batched_extend_scalar`]: the portable reference (scalar per-lane cell arithmetic, step 2b-i).
//! - `batched_extend_neon` (aarch64 only, step 2b-ii): the same recurrence with the inner cell
//!   arithmetic vectorized across 8 lanes with NEON `int16x8` ops. int16 is exact here because every
//!   `H`/`E`/`F` value in a local extension is clamped `>= 0` and bounded by
//!   `h0 + qlen*max_score + end_bonus` (~195 for a 150 bp read), far inside the int16 range; a guard
//!   falls back to the scalar path if a batch could ever exceed that (never happens for short reads).
//!
//! [`batched_extend`] dispatches to the NEON path on aarch64 when the int16 bound holds, else scalar.

use bwa_extend::{ExtendJob, ExtendResult};

/// Lanes processed per group. 8 = one NEON `int16x8` register (the vectorized path); the scalar
/// reference works for any value.
const LANES: usize = 8;

/// Batched banded local extension. Returns one [`ExtendResult`] per job, each equal to
/// [`bwa_extend::ksw_extend2`] on that job. Dispatches to the NEON kernel where available.
#[allow(clippy::too_many_arguments)]
pub fn batched_extend(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && fits_i16(jobs, mat, m, end_bonus) {
            // SAFETY: NEON is available (checked above) and the int16 bound holds (fits_i16).
            return unsafe {
                neon::batched_extend_neon(
                    jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
                )
            };
        }
    }
    batched_extend_scalar(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    )
}

/// True when every cell value in every job's extension is provably within the int16 range, so the
/// NEON int16 kernel is byte-identical. Bound: `h0 + qlen*max_score + end_bonus` (the local optimum
/// ceiling), with margin. Always true for short reads.
#[cfg(target_arch = "aarch64")]
fn fits_i16(jobs: &[ExtendJob], mat: &[i8], m: usize, end_bonus: i32) -> bool {
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
    const GUARD: i32 = 30_000; // < i16::MAX (32767), leaves headroom for intermediates.
    jobs.iter().all(|j| {
        let bound = j.h0 + (j.query.len() as i32) * max_sc.max(0) + end_bonus.max(0);
        bound < GUARD
    })
}

/// Portable scalar reference (step 2b-i): lane-batched control flow, scalar per-cell arithmetic.
/// This is the byte-identity source of truth the NEON kernel is validated against.
#[allow(clippy::too_many_arguments)]
pub fn batched_extend_scalar(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    let mut out = vec![default_result(); jobs.len()];

    for chunk_start in (0..jobs.len()).step_by(LANES) {
        let nlane = (jobs.len() - chunk_start).min(LANES);

        // Per-lane inputs.
        let mut qlen = [0usize; LANES];
        let mut tlen = [0usize; LANES];
        let mut h0 = [0i32; LANES];
        let mut w = [0i32; LANES];
        for l in 0..nlane {
            let job = &jobs[chunk_start + l];
            qlen[l] = job.query.len();
            tlen[l] = job.target.len();
            h0[l] = job.h0;
            w[l] = clamp_band(w0, qlen[l], max_sc, end_bonus, o_ins, e_ins, o_del, e_del);
        }
        let max_q = qlen[..nlane].iter().copied().max().unwrap_or(0);
        let max_t = tlen[..nlane].iter().copied().max().unwrap_or(0);

        // Per-lane DP state, indexed [lane * (max_q + 1) + j].
        let stride = max_q + 1;
        let mut eh_h = vec![0i32; LANES * stride];
        let mut eh_e = vec![0i32; LANES * stride];

        let mut beg = [0i32; LANES];
        let mut end = [0i32; LANES];
        let mut max = [0i32; LANES];
        let mut max_i = [-1i32; LANES];
        let mut max_j = [-1i32; LANES];
        let mut max_ie = [-1i32; LANES];
        let mut gscore = [-1i32; LANES];
        let mut max_off = [0i32; LANES];
        let mut done = [true; LANES];

        for l in 0..nlane {
            eh_h[l * stride] = h0[l];
            if qlen[l] >= 1 {
                eh_h[l * stride + 1] = if h0[l] > oe_ins { h0[l] - oe_ins } else { 0 };
            }
            let mut j = 2usize;
            while j <= qlen[l] && eh_h[l * stride + j - 1] > e_ins {
                eh_h[l * stride + j] = eh_h[l * stride + j - 1] - e_ins;
                j += 1;
            }
            max[l] = h0[l];
            end[l] = qlen[l] as i32;
            done[l] = false;
        }

        for i in 0..max_t as i32 {
            let mut h1 = [0i32; LANES];
            let mut f = [0i32; LANES];
            let mut row_max = [0i32; LANES];
            let mut mj = [-1i32; LANES];
            let mut active = [false; LANES];
            let mut gbeg = i32::MAX;
            let mut gend = 0i32;
            for l in 0..nlane {
                if done[l] || i >= tlen[l] as i32 {
                    continue;
                }
                active[l] = true;
                if beg[l] < i - w[l] {
                    beg[l] = i - w[l];
                }
                if end[l] > i + w[l] + 1 {
                    end[l] = i + w[l] + 1;
                }
                if end[l] > qlen[l] as i32 {
                    end[l] = qlen[l] as i32;
                }
                h1[l] = if beg[l] == 0 {
                    (h0[l] - (o_del + e_del * (i + 1))).max(0)
                } else {
                    0
                };
                gbeg = gbeg.min(beg[l]);
                gend = gend.max(end[l]);
            }

            for j in gbeg..gend {
                let ju = j as usize;
                for l in 0..nlane {
                    if !active[l] || j < beg[l] || j >= end[l] {
                        continue;
                    }
                    let base = l * stride;
                    let mut big_m = eh_h[base + ju];
                    let mut e = eh_e[base + ju];
                    eh_h[base + ju] = h1[l];
                    let score = i32::from(
                        mat[jobs[chunk_start + l].target[i as usize] as usize * m
                            + jobs[chunk_start + l].query[ju] as usize],
                    );
                    big_m = if big_m != 0 { big_m + score } else { 0 };
                    let mut h = big_m.max(e);
                    h = h.max(f[l]);
                    h1[l] = h;
                    if row_max[l] <= h {
                        mj[l] = j;
                        row_max[l] = h;
                    }
                    let t = (big_m - oe_del).max(0);
                    e = (e - e_del).max(t);
                    eh_e[base + ju] = e;
                    let t = (big_m - oe_ins).max(0);
                    f[l] = (f[l] - e_ins).max(t);
                }
            }

            for l in 0..nlane {
                if !active[l] {
                    continue;
                }
                let base = l * stride;
                eh_h[base + end[l] as usize] = h1[l];
                eh_e[base + end[l] as usize] = 0;
                if end[l] == qlen[l] as i32 && gscore[l] <= h1[l] {
                    max_ie[l] = i;
                    gscore[l] = h1[l];
                }
                if row_max[l] == 0 {
                    done[l] = true;
                    continue;
                }
                if row_max[l] > max[l] {
                    max[l] = row_max[l];
                    max_i[l] = i;
                    max_j[l] = mj[l];
                    let off = (mj[l] - i).abs();
                    if off > max_off[l] {
                        max_off[l] = off;
                    }
                } else if zdrop > 0 {
                    let drop = if i - max_i[l] > mj[l] - max_j[l] {
                        max[l] - row_max[l] - ((i - max_i[l]) - (mj[l] - max_j[l])) * e_del
                    } else {
                        max[l] - row_max[l] - ((mj[l] - max_j[l]) - (i - max_i[l])) * e_ins
                    };
                    if drop > zdrop {
                        done[l] = true;
                        continue;
                    }
                }
                let mut jb = beg[l];
                while jb < end[l] && eh_h[base + jb as usize] == 0 && eh_e[base + jb as usize] == 0
                {
                    jb += 1;
                }
                beg[l] = jb;
                let mut je = end[l];
                while je >= beg[l] && eh_h[base + je as usize] == 0 && eh_e[base + je as usize] == 0
                {
                    je -= 1;
                }
                end[l] = if je + 2 < qlen[l] as i32 {
                    je + 2
                } else {
                    qlen[l] as i32
                };
            }
        }

        for l in 0..nlane {
            out[chunk_start + l] = ExtendResult {
                score: max[l],
                qle: max_j[l] + 1,
                tle: max_i[l] + 1,
                gtle: max_ie[l] + 1,
                gscore: gscore[l],
                max_off: max_off[l],
            };
        }
    }

    out
}

fn default_result() -> ExtendResult {
    ExtendResult {
        score: 0,
        qle: 0,
        tle: 0,
        gtle: 0,
        gscore: 0,
        max_off: 0,
    }
}

/// Per-lane band clamp, mirroring the `w` adjustments in `ksw_extend2`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn clamp_band(
    w0: i32,
    qlen: usize,
    max_sc: i32,
    end_bonus: i32,
    o_ins: i32,
    e_ins: i32,
    o_del: i32,
    e_del: i32,
) -> i32 {
    let mut wl = w0;
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    wl = wl.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    wl = wl.min(max_del.max(1));
    wl
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{clamp_band, default_result, LANES};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::aarch64::*;

    /// NEON int16x8 batched banded local extension (step 2b-ii). Byte-identical to the scalar
    /// reference for every job (the caller guarantees the int16 bound via `fits_i16`).
    ///
    /// # Safety
    /// Requires the `neon` target feature. All vector loads/stores use fixed-size `[i16; 8]` /
    /// `[u16; 8]` scratch arrays, so they are in-bounds by construction.
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn batched_extend_neon(
        jobs: &[ExtendJob],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w0: i32,
        end_bonus: i32,
        zdrop: i32,
    ) -> Vec<ExtendResult> {
        let oe_del = o_del + e_del;
        let oe_ins = o_ins + e_ins;
        let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

        let mut out = vec![default_result(); jobs.len()];

        // Broadcast scoring constants (fit i16 by the caller's guard / small gap penalties).
        let oe_del_v = vdupq_n_s16(oe_del as i16);
        let oe_ins_v = vdupq_n_s16(oe_ins as i16);
        let e_del_v = vdupq_n_s16(e_del as i16);
        let e_ins_v = vdupq_n_s16(e_ins as i16);
        let zero_v = vdupq_n_s16(0);

        for chunk_start in (0..jobs.len()).step_by(LANES) {
            let nlane = (jobs.len() - chunk_start).min(LANES);

            let mut qlen = [0usize; LANES];
            let mut tlen = [0usize; LANES];
            let mut h0 = [0i32; LANES];
            let mut w = [0i32; LANES];
            for l in 0..nlane {
                let job = &jobs[chunk_start + l];
                qlen[l] = job.query.len();
                tlen[l] = job.target.len();
                h0[l] = job.h0;
                w[l] = clamp_band(w0, qlen[l], max_sc, end_bonus, o_ins, e_ins, o_del, e_del);
            }
            let max_q = qlen[..nlane].iter().copied().max().unwrap_or(0);
            let max_t = tlen[..nlane].iter().copied().max().unwrap_or(0);

            // Interleaved DP state: eh_h[j*LANES + lane] so the 8 lanes at column j are one vector.
            let stride = max_q + 1;
            let mut eh_h = vec![0i16; stride * LANES];
            let mut eh_e = vec![0i16; stride * LANES];

            // Per-lane query padded to `stride` (0 past qlen) so the score gather is branchless and
            // never indexes out of bounds; interleaved as qpad[j*LANES + lane].
            let mut qpad = vec![0usize; stride * LANES];
            for l in 0..nlane {
                let q = jobs[chunk_start + l].query;
                for (ju, &c) in q.iter().enumerate() {
                    qpad[ju * LANES + l] = c as usize;
                }
            }

            let mut beg = [0i32; LANES];
            let mut end = [0i32; LANES];
            let mut max = [0i32; LANES];
            let mut max_i = [-1i32; LANES];
            let mut max_j = [-1i32; LANES];
            let mut max_ie = [-1i32; LANES];
            let mut gscore = [-1i32; LANES];
            let mut max_off = [0i32; LANES];
            let mut done = [true; LANES];

            for l in 0..nlane {
                eh_h[l] = h0[l] as i16; // column 0
                if qlen[l] >= 1 {
                    eh_h[LANES + l] = if h0[l] > oe_ins {
                        (h0[l] - oe_ins) as i16
                    } else {
                        0
                    };
                }
                let mut j = 2usize;
                while j <= qlen[l] && i32::from(eh_h[(j - 1) * LANES + l]) > e_ins {
                    eh_h[j * LANES + l] = eh_h[(j - 1) * LANES + l] - e_ins as i16;
                    j += 1;
                }
                max[l] = h0[l];
                end[l] = qlen[l] as i32;
                done[l] = false;
            }

            for i in 0..max_t as i32 {
                let mut h1 = [0i16; LANES];
                let mut active = [false; LANES];
                let mut gbeg = i32::MAX;
                let mut gend = 0i32;
                for l in 0..nlane {
                    if done[l] || i >= tlen[l] as i32 {
                        continue;
                    }
                    active[l] = true;
                    if beg[l] < i - w[l] {
                        beg[l] = i - w[l];
                    }
                    if end[l] > i + w[l] + 1 {
                        end[l] = i + w[l] + 1;
                    }
                    if end[l] > qlen[l] as i32 {
                        end[l] = qlen[l] as i32;
                    }
                    h1[l] = if beg[l] == 0 {
                        (h0[l] - (o_del + e_del * (i + 1))).max(0) as i16
                    } else {
                        0
                    };
                    gbeg = gbeg.min(beg[l]);
                    gend = gend.max(end[l]);
                }

                let mut h1_v = vld1q_s16(h1.as_ptr());
                let mut f_v = zero_v;
                let mut rowmax_v = zero_v;
                let mut mj_v = vdupq_n_s16(-1);

                // Per-row band bounds and active mask as vectors (band test is branchless below).
                let mut beg16 = [0i16; LANES];
                let mut end16 = [0i16; LANES];
                let mut act16 = [0u16; LANES];
                // Per-lane target base for this row (for the score gather).
                let mut tbase = [0usize; LANES];
                for l in 0..nlane {
                    if active[l] {
                        beg16[l] = beg[l] as i16;
                        end16[l] = end[l] as i16;
                        act16[l] = 0xFFFF;
                        tbase[l] = jobs[chunk_start + l].target[i as usize] as usize * m;
                    }
                }
                let beg_v = vld1q_s16(beg16.as_ptr());
                let end_v = vld1q_s16(end16.as_ptr());
                let active_v = vld1q_u16(act16.as_ptr());

                for j in gbeg..gend {
                    let ju = j as usize;
                    let jrow = ju * LANES;
                    // band = active & (j >= beg) & (j < end), branchless.
                    let j_v = vdupq_n_s16(j as i16);
                    let band = vandq_u16(
                        active_v,
                        vandq_u16(vcgeq_s16(j_v, beg_v), vcltq_s16(j_v, end_v)),
                    );
                    // Branchless score gather (qpad padded with 0, so always in bounds).
                    let mut score_arr = [0i16; LANES];
                    for l in 0..nlane {
                        score_arr[l] = i16::from(mat[tbase[l] + qpad[jrow + l]]);
                    }
                    let score_v = vld1q_s16(score_arr.as_ptr());

                    let col = jrow;
                    let m_v = vld1q_s16(eh_h.as_ptr().add(col)); // H(i-1, j-1)
                    let e_v = vld1q_s16(eh_e.as_ptr().add(col)); // E(i-1, j)

                    // eh_h[j] <- h1 (old), for in-band lanes; out-of-band keep old m_v.
                    vst1q_s16(eh_h.as_mut_ptr().add(col), vbslq_s16(band, h1_v, m_v));

                    // big_m = if m != 0 { m + score } else { 0 }
                    let zero_mask = vceqzq_s16(m_v);
                    let bigm_v = vbslq_s16(zero_mask, zero_v, vaddq_s16(m_v, score_v));

                    // h = max(big_m, e, f)
                    let h_v = vmaxq_s16(vmaxq_s16(bigm_v, e_v), f_v);
                    // h1 <- h (in-band)
                    h1_v = vbslq_s16(band, h_v, h1_v);

                    // if row_max <= h { mj = j; row_max = h }  (in-band, ties take larger j)
                    let ge = vcgeq_s16(h_v, rowmax_v);
                    let upd = vandq_u16(band, ge);
                    rowmax_v = vbslq_s16(upd, h_v, rowmax_v);
                    mj_v = vbslq_s16(upd, vdupq_n_s16(j as i16), mj_v);

                    // e = max(e - e_del, max(big_m - oe_del, 0));  store (in-band)
                    let t1 = vmaxq_s16(vsubq_s16(bigm_v, oe_del_v), zero_v);
                    let e_new = vmaxq_s16(vsubq_s16(e_v, e_del_v), t1);
                    vst1q_s16(eh_e.as_mut_ptr().add(col), vbslq_s16(band, e_new, e_v));

                    // f = max(f - e_ins, max(big_m - oe_ins, 0))  (in-band)
                    let t2 = vmaxq_s16(vsubq_s16(bigm_v, oe_ins_v), zero_v);
                    let f_new = vmaxq_s16(vsubq_s16(f_v, e_ins_v), t2);
                    f_v = vbslq_s16(band, f_new, f_v);
                }

                // Extract per-lane vector state for the scalar epilogue.
                let mut h1o = [0i16; LANES];
                let mut rmo = [0i16; LANES];
                let mut mjo = [0i16; LANES];
                vst1q_s16(h1o.as_mut_ptr(), h1_v);
                vst1q_s16(rmo.as_mut_ptr(), rowmax_v);
                vst1q_s16(mjo.as_mut_ptr(), mj_v);

                for l in 0..nlane {
                    if !active[l] {
                        continue;
                    }
                    let h1l = i32::from(h1o[l]);
                    let row_max_l = i32::from(rmo[l]);
                    let mj_l = i32::from(mjo[l]);
                    eh_h[end[l] as usize * LANES + l] = h1l as i16;
                    eh_e[end[l] as usize * LANES + l] = 0;
                    if end[l] == qlen[l] as i32 && gscore[l] <= h1l {
                        max_ie[l] = i;
                        gscore[l] = h1l;
                    }
                    if row_max_l == 0 {
                        done[l] = true;
                        continue;
                    }
                    if row_max_l > max[l] {
                        max[l] = row_max_l;
                        max_i[l] = i;
                        max_j[l] = mj_l;
                        let off = (mj_l - i).abs();
                        if off > max_off[l] {
                            max_off[l] = off;
                        }
                    } else if zdrop > 0 {
                        let drop = if i - max_i[l] > mj_l - max_j[l] {
                            max[l] - row_max_l - ((i - max_i[l]) - (mj_l - max_j[l])) * e_del
                        } else {
                            max[l] - row_max_l - ((mj_l - max_j[l]) - (i - max_i[l])) * e_ins
                        };
                        if drop > zdrop {
                            done[l] = true;
                            continue;
                        }
                    }
                    let mut jb = beg[l];
                    while jb < end[l]
                        && eh_h[jb as usize * LANES + l] == 0
                        && eh_e[jb as usize * LANES + l] == 0
                    {
                        jb += 1;
                    }
                    beg[l] = jb;
                    let mut je = end[l];
                    while je >= beg[l]
                        && eh_h[je as usize * LANES + l] == 0
                        && eh_e[je as usize * LANES + l] == 0
                    {
                        je -= 1;
                    }
                    end[l] = if je + 2 < qlen[l] as i32 {
                        je + 2
                    } else {
                        qlen[l] as i32
                    };
                }
            }

            for l in 0..nlane {
                out[chunk_start + l] = ExtendResult {
                    score: max[l],
                    qle: max_j[l] + 1,
                    tle: max_i[l] + 1,
                    gtle: max_ie[l] + 1,
                    gscore: gscore[l],
                    max_off: max_off[l],
                };
            }
        }

        out
    }
}

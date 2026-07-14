//! Lane-batched banded Smith-Waterman seed extension (phase 9a).
//!
//! Processes several independent alignments in lockstep: a shared target-row loop `i` and a shared
//! query loop `j` over the union band, each lane masked to its own band `[beg, end)` and its own
//! termination (row all-zero or z-drop). Two implementations, both **byte-identical** to
//! [`bwa_extend::ksw_extend2`] run per lane:
//!
//! - [`batched_extend_scalar`]: the portable reference (scalar per-lane cell arithmetic, step 2b-i).
//! - `neon::batched_extend_neon_i8` / `_i16` (aarch64 only, step 2b-ii): the same recurrence with the
//!   inner cell arithmetic vectorized across NEON lanes (16 int8 lanes or 8 int16 lanes), generated
//!   from one macro so the two widths cannot drift. Both are exact because every `H`/`E`/`F` value in
//!   a local extension is clamped `>= 0` and bounded by `h0 + qlen*max_score + end_bonus`, so a
//!   per-job length bound decides int8 (bound < 120) vs int16 (< 30000) vs the scalar fallback.
//!
//! [`batched_extend`] dispatches on aarch64 to [`neon_dispatch`], which bins each job by that bound
//! into int8 (16 lanes) / int16 (8 lanes) / scalar and scatters results back. This is bwa-mem2's
//! `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` length-binning: the int8 path packs twice the lanes for short pairs.

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
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon_dispatch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
    }
    batched_extend_scalar(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    )
}

/// Per-job cell-value ceiling for a local extension: `h0 + qlen*max_score + end_bonus`. Every
/// `H`/`E`/`F` value stays in `[0, bound]`, so `bound < i8/i16::MAX` (minus a small margin for
/// negative gap intermediates) proves the corresponding NEON kernel is byte-identical.
#[cfg(target_arch = "aarch64")]
#[inline]
fn cell_bound(j: &ExtendJob, max_sc: i32, end_bonus: i32) -> i32 {
    j.h0 + (j.query.len() as i32) * max_sc.max(0) + end_bonus.max(0)
}

/// NEON dispatch: bin each job by length into int8 (16 lanes), int16 (8 lanes), or the scalar
/// fallback, process each bin with the matching kernel, and scatter results back by index. This is
/// bwa-mem2's `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` length-binning: the int8 path packs twice the lanes for
/// short pairs. Binning is result-preserving (each job's extension depends only on its own inputs).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn neon_dispatch(
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
    // int8 guard: cell values in [0, bound] must fit i8, and the DP indices (j, beg, end, mj, i) must
    // too; `bound`, qlen and tlen all below 120 keeps everything inside int8 with margin.
    const GUARD8: i32 = 120; // < i8::MAX (127)
    const GUARD16: i32 = 30_000; // < i16::MAX (32767)
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    let (mut i8_idx, mut i16_idx, mut sc_idx) = (Vec::new(), Vec::new(), Vec::new());
    for (k, j) in jobs.iter().enumerate() {
        let bound = cell_bound(j, max_sc, end_bonus);
        if bound < GUARD8 && j.query.len() < 120 && j.target.len() < 120 {
            i8_idx.push(k);
        } else if bound < GUARD16 {
            i16_idx.push(k);
        } else {
            sc_idx.push(k);
        }
    }

    // Homogeneous fast path: whole batch in one bin -> run the kernel on `jobs` with no gather/scatter.
    let n = jobs.len();
    if i8_idx.len() == n {
        // SAFETY: neon available (checked by caller); GUARD8 bounds keep all values inside i8.
        return unsafe {
            neon::batched_extend_neon_i8(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            )
        };
    }
    if i16_idx.len() == n {
        // SAFETY: neon available; GUARD16 bounds keep all values inside i16.
        return unsafe {
            neon::batched_extend_neon_i16(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            )
        };
    }
    if sc_idx.len() == n {
        return batched_extend_scalar(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }

    let mut out = vec![default_result(); n];

    let mut run = |idx: &[usize], f: &dyn Fn(&[ExtendJob]) -> Vec<ExtendResult>| {
        if idx.is_empty() {
            return;
        }
        let sub: Vec<ExtendJob> = idx.iter().map(|&k| jobs[k]).collect();
        let res = f(&sub);
        for (p, &k) in idx.iter().enumerate() {
            out[k] = res[p];
        }
    };
    run(&i8_idx, &|s| unsafe {
        // SAFETY: neon available (checked by caller); GUARD8 bounds keep all values inside i8.
        neon::batched_extend_neon_i8(s, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop)
    });
    run(&i16_idx, &|s| unsafe {
        // SAFETY: neon available; GUARD16 bounds keep all values inside i16.
        neon::batched_extend_neon_i16(s, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop)
    });
    run(&sc_idx, &|s| {
        batched_extend_scalar(s, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop)
    });
    out
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
    use super::{clamp_band, default_result};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::aarch64::*;

    /// Generate a NEON batched banded-SW kernel for a given lane element type. Both instances
    /// (`i16` x8 and `i8` x16) are byte-identical to the scalar reference for every job when the
    /// caller guarantees the element range (`fits_i16` / `fits_i8`); using one macro keeps the two
    /// width variants from drifting. Layout is SoA `[column*LANES + lane]` (as bwa-mem2's `bandedSWA`
    /// / nh13's `getScores16`/`getScores8`), the band mask is `vbslq` blendv, and the recurrence uses
    /// non-saturating add/sub (exact within the guaranteed range).
    macro_rules! define_neon_kernel {
        (
            $name:ident, $elem:ty, $umask:ty, $lanes:expr,
            dup = $dup:path, lds = $lds:path, sts = $sts:path, ldu = $ldu:path,
            add = $add:path, sub = $sub:path, max = $max:path,
            ceqz = $ceqz:path, cge = $cge:path, clt = $clt:path,
            andu = $andu:path, bsl = $bsl:path
        ) => {
            /// # Safety
            /// Requires the `neon` target feature; all vector loads/stores use fixed-size
            /// `[$elem; LANES]` / `[$umask; LANES]` scratch arrays, in-bounds by construction.
            #[target_feature(enable = "neon")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $name(
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
                const LANES: usize = $lanes;
                let oe_del = o_del + e_del;
                let oe_ins = o_ins + e_ins;
                let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

                let mut out = vec![default_result(); jobs.len()];

                let oe_del_v = $dup(oe_del as $elem);
                let oe_ins_v = $dup(oe_ins as $elem);
                let e_del_v = $dup(e_del as $elem);
                let e_ins_v = $dup(e_ins as $elem);
                let zero_v = $dup(0);

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
                        w[l] =
                            clamp_band(w0, qlen[l], max_sc, end_bonus, o_ins, e_ins, o_del, e_del);
                    }
                    let max_q = qlen[..nlane].iter().copied().max().unwrap_or(0);
                    let max_t = tlen[..nlane].iter().copied().max().unwrap_or(0);

                    let stride = max_q + 1;
                    let mut eh_h = vec![0 as $elem; stride * LANES];
                    let mut eh_e = vec![0 as $elem; stride * LANES];

                    // Per-lane query padded to `stride` (0 past qlen): branchless, in-bounds gather.
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
                        eh_h[l] = h0[l] as $elem; // column 0
                        if qlen[l] >= 1 {
                            eh_h[LANES + l] = if h0[l] > oe_ins {
                                (h0[l] - oe_ins) as $elem
                            } else {
                                0
                            };
                        }
                        let mut j = 2usize;
                        while j <= qlen[l] && i32::from(eh_h[(j - 1) * LANES + l]) > e_ins {
                            eh_h[j * LANES + l] = eh_h[(j - 1) * LANES + l] - e_ins as $elem;
                            j += 1;
                        }
                        max[l] = h0[l];
                        end[l] = qlen[l] as i32;
                        done[l] = false;
                    }

                    for i in 0..max_t as i32 {
                        let mut h1 = [0 as $elem; LANES];
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
                                (h0[l] - (o_del + e_del * (i + 1))).max(0) as $elem
                            } else {
                                0
                            };
                            gbeg = gbeg.min(beg[l]);
                            gend = gend.max(end[l]);
                        }

                        let mut h1_v = $lds(h1.as_ptr());
                        let mut f_v = zero_v;
                        let mut rowmax_v = zero_v;
                        let mut mj_v = $dup(-1);

                        let mut beg_a = [0 as $elem; LANES];
                        let mut end_a = [0 as $elem; LANES];
                        let mut act_a = [0 as $umask; LANES];
                        let mut tbase = [0usize; LANES];
                        for l in 0..nlane {
                            if active[l] {
                                beg_a[l] = beg[l] as $elem;
                                end_a[l] = end[l] as $elem;
                                act_a[l] = <$umask>::MAX;
                                tbase[l] = jobs[chunk_start + l].target[i as usize] as usize * m;
                            }
                        }
                        let beg_v = $lds(beg_a.as_ptr());
                        let end_v = $lds(end_a.as_ptr());
                        let active_v = $ldu(act_a.as_ptr());

                        for j in gbeg..gend {
                            let jrow = j as usize * LANES;
                            let j_v = $dup(j as $elem);
                            let band = $andu(active_v, $andu($cge(j_v, beg_v), $clt(j_v, end_v)));

                            let mut score_arr = [0 as $elem; LANES];
                            for l in 0..nlane {
                                score_arr[l] = mat[tbase[l] + qpad[jrow + l]] as $elem;
                            }
                            let score_v = $lds(score_arr.as_ptr());

                            let m_v = $lds(eh_h.as_ptr().add(jrow)); // H(i-1, j-1)
                            let e_v = $lds(eh_e.as_ptr().add(jrow)); // E(i-1, j)

                            // eh_h[j] <- h1 (old) for in-band lanes; out-of-band keep old m_v.
                            $sts(eh_h.as_mut_ptr().add(jrow), $bsl(band, h1_v, m_v));

                            // big_m = if m != 0 { m + score } else { 0 }
                            let zero_mask = $ceqz(m_v);
                            let bigm_v = $bsl(zero_mask, zero_v, $add(m_v, score_v));

                            let h_v = $max($max(bigm_v, e_v), f_v);
                            h1_v = $bsl(band, h_v, h1_v);

                            // if row_max <= h { mj = j; row_max = h } (ties take larger j)
                            let upd = $andu(band, $cge(h_v, rowmax_v));
                            rowmax_v = $bsl(upd, h_v, rowmax_v);
                            mj_v = $bsl(upd, $dup(j as $elem), mj_v);

                            // e = max(e - e_del, max(big_m - oe_del, 0)); store in-band
                            let t1 = $max($sub(bigm_v, oe_del_v), zero_v);
                            let e_new = $max($sub(e_v, e_del_v), t1);
                            $sts(eh_e.as_mut_ptr().add(jrow), $bsl(band, e_new, e_v));

                            // f = max(f - e_ins, max(big_m - oe_ins, 0)) in-band
                            let t2 = $max($sub(bigm_v, oe_ins_v), zero_v);
                            let f_new = $max($sub(f_v, e_ins_v), t2);
                            f_v = $bsl(band, f_new, f_v);
                        }

                        let mut h1o = [0 as $elem; LANES];
                        let mut rmo = [0 as $elem; LANES];
                        let mut mjo = [0 as $elem; LANES];
                        $sts(h1o.as_mut_ptr(), h1_v);
                        $sts(rmo.as_mut_ptr(), rowmax_v);
                        $sts(mjo.as_mut_ptr(), mj_v);

                        for l in 0..nlane {
                            if !active[l] {
                                continue;
                            }
                            let h1l = i32::from(h1o[l]);
                            let row_max_l = i32::from(rmo[l]);
                            let mj_l = i32::from(mjo[l]);
                            eh_h[end[l] as usize * LANES + l] = h1l as $elem;
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
                                    max[l]
                                        - row_max_l
                                        - ((i - max_i[l]) - (mj_l - max_j[l])) * e_del
                                } else {
                                    max[l]
                                        - row_max_l
                                        - ((mj_l - max_j[l]) - (i - max_i[l])) * e_ins
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
        };
    }

    define_neon_kernel!(
        batched_extend_neon_i16,
        i16,
        u16,
        8,
        dup = vdupq_n_s16,
        lds = vld1q_s16,
        sts = vst1q_s16,
        ldu = vld1q_u16,
        add = vaddq_s16,
        sub = vsubq_s16,
        max = vmaxq_s16,
        ceqz = vceqzq_s16,
        cge = vcgeq_s16,
        clt = vcltq_s16,
        andu = vandq_u16,
        bsl = vbslq_s16
    );

    define_neon_kernel!(
        batched_extend_neon_i8,
        i8,
        u8,
        16,
        dup = vdupq_n_s8,
        lds = vld1q_s8,
        sts = vst1q_s8,
        ldu = vld1q_u8,
        add = vaddq_s8,
        sub = vsubq_s8,
        max = vmaxq_s8,
        ceqz = vceqzq_s8,
        cge = vcgeq_s8,
        clt = vcltq_s8,
        andu = vandq_u8,
        bsl = vbslq_s8
    );
}

//! Lane-batched banded Smith-Waterman seed extension (phase 9a, step 2b-i).
//!
//! This processes `LANES` independent alignments in lockstep: a shared target-row loop `i` and a
//! shared query loop `j` over the union band, with each lane masked to its own band `[beg, end)`
//! and its own termination (row all-zero or z-drop). The per-cell arithmetic is scalar here and is
//! **byte-identical** to [`bwa_extend::ksw_extend2`] run per lane; step 2b-ii replaces the inner
//! `for lane` arithmetic with NEON int32x4 ops (add/sub/max/select), which is the whole point of
//! this SoA layout. Everything downstream (band tightening, max tracking) already operates on
//! per-lane arrays, so the vectorization is a localized change.

use bwa_extend::{ExtendJob, ExtendResult};

/// Lanes processed per group. 4 matches a NEON int32x4 register (the target for step 2b-ii); the
/// result is independent of this value.
const LANES: usize = 4;

/// Batched banded local extension. Returns one [`ExtendResult`] per job, each equal to
/// [`bwa_extend::ksw_extend2`] on that job.
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
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    let mut out = vec![
        ExtendResult {
            score: 0,
            qle: 0,
            tle: 0,
            gtle: 0,
            gscore: 0,
            max_off: 0,
        };
        jobs.len()
    ];

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
            // Per-lane band clamp (ksw_extend2 lines adjusting w by max_ins/max_del).
            let mut wl = w0;
            let max_ins = (((qlen[l] as f64 * f64::from(max_sc) + f64::from(end_bonus)
                - f64::from(o_ins))
                / f64::from(e_ins))
                + 1.0) as i32;
            wl = wl.min(max_ins.max(1));
            let max_del = (((qlen[l] as f64 * f64::from(max_sc) + f64::from(end_bonus)
                - f64::from(o_del))
                / f64::from(e_del))
                + 1.0) as i32;
            wl = wl.min(max_del.max(1));
            w[l] = wl;
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
            // eh_h[0] = h0; eh_h[1] = h0 - oe_ins (>=0); then a decaying insertion prefix.
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
            // Per-lane row setup: activate lanes still running, tighten band, init h1.
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

            // Shared query loop over the union band; each lane masked to its own [beg, end).
            for j in gbeg..gend {
                let ju = j as usize;
                for l in 0..nlane {
                    if !active[l] || j < beg[l] || j >= end[l] {
                        continue;
                    }
                    let base = l * stride;
                    // ---- inner cell recurrence (NEON int32x4 target in step 2b-ii) ----
                    let mut big_m = eh_h[base + ju]; // H(i-1, j-1)
                    let mut e = eh_e[base + ju]; // E(i-1, j)
                    eh_h[base + ju] = h1[l]; // H(i, j-1) for the next row
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
                    // ------------------------------------------------------------------
                }
            }

            // Per-lane row epilogue: mirrors ksw_extend2 after the j-loop.
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
                // Shrink the band around still-live cells.
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

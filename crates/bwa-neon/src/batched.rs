//! Lane-batched banded Smith-Waterman seed extension (phase 9a).
//!
//! Processes several independent alignments in lockstep: a shared target-row loop `i` and a shared
//! query loop `j` over the union band, each lane masked to its own band `[beg, end)` and its own
//! termination (row all-zero or z-drop). Two implementations, both **byte-identical** to
//! [`bwa_extend::ksw_extend2`] run per lane:
//!
//! - [`batched_extend_scalar`]: the portable reference (scalar per-lane cell arithmetic, step 2b-i).
//! - The SIMD kernels: one `define_sw_kernel!` macro, instantiated per ISA + lane width, so the
//!   variants cannot drift. **NEON** (`mod neon`, aarch64): u8 x16 / i16 x8. **AVX2** (`mod avx2`,
//!   x86_64): u8 x32 / i16 x16 (256-bit, twice NEON's width). All are exact because a local
//!   extension's `H`/`E`/`F` stay in `[0, minval]` with `minval = h0 + min(len)*a`, so the per-job
//!   bound decides u8 (`minval < 256`) vs i16 (`< 32768`) vs the scalar fallback.
//!
//! [`batched_extend`] dispatches to [`simd_dispatch`] when the ISA feature is present (NEON on
//! aarch64, AVX2 on x86_64): the ungapped-diagonal HIT fast path, then bwa-mem2's
//! `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` length-binning (the u8 path packs twice the lanes of i16). AVX-512
//! (64 u8 lanes) is a follow-up (needs the k-mask blend variant). The AVX2 kernels are validated
//! byte-identical to `ksw_extend2` by a force-run test executed under Rosetta (`avx2_verify`).

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
            return simd_dispatch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return simd_dispatch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
    }
    batched_extend_scalar(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    )
}

#[cfg(target_arch = "x86_64")]
use avx2::{batched_extend_avx2_i16 as sw_kernel_i16, batched_extend_avx2_u8 as sw_kernel_u8};
/// Element type / kernel alias for the current SIMD ISA: NEON on aarch64, AVX2 on x86_64.
#[cfg(target_arch = "aarch64")]
use neon::{batched_extend_neon_i16 as sw_kernel_i16, batched_extend_neon_u8 as sw_kernel_u8};

/// True if `mat` is the standard bwa `m x m` DNA matrix: a constant `a` on the diagonal, a constant
/// `mm` off-diagonal among the 4 concrete bases, and a constant `npen` on every ambiguous row/column
/// (index `m-1`). The NEON kernels score with a vector compare that relies on exactly this shape;
/// anything else falls back to the scalar reference (which reads `mat` directly).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn is_uniform_dna(mat: &[i8], m: usize) -> bool {
    if m < 2 || mat.len() < m * m {
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
    true
}

/// bwa-mem2's exact per-pair score ceiling (`sort_classify` in `bwamem.cpp`):
/// `minval = h0 + min(len1, len2) * score_a`. Because a local extension matches at most
/// `min(qlen, tlen)` bases and only *loses* score to mismatches/gaps, every `H`/`E`/`F` value (and
/// the intermediate `M = m + score` that becomes an `H`) stays in `[<0-ish>, minval]`. So
/// `minval < MAX_SEQ_LEN8 (128)` proves the signed-int8 kernel is exact (values fit `[0,127]`), and
/// `< MAX_SEQ_LEN16 (32768)` proves the int16 kernel is exact. This matches the oracle's own binning,
/// so the classification never changes results — only which (exact) kernel runs.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn cell_bound(j: &ExtendJob, max_sc: i32) -> i32 {
    j.h0 + (j.query.len().min(j.target.len()) as i32) * max_sc.max(0)
}

/// Ungapped diagonal fast path (nh13's `ungapped_analyze` HIT case, `bwamem.cpp`): when the whole
/// query matches the target on the diagonal with **no** mismatch and no ambiguous base (and the
/// target is at least as long, `h0 > 0`), the banded Smith-Waterman result is the closed form
/// `score = gscore = h0 + n*a`, `qle = tle = gtle = n`, `max_off = 0` — so the DP is skipped
/// entirely. Returns `None` (fall through to the kernel) for any mismatch/ambiguous/short-target
/// case. This is byte-identical to [`bwa_extend::ksw_extend2`] for the jobs it accepts (gated by the
/// property test), matching bwa-mem2's own output. `a` is the match score (`mat[0]`).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn ungapped_hit(j: &ExtendJob, a: i32) -> Option<ExtendResult> {
    let n = j.query.len();
    if n == 0 || j.h0 <= 0 || j.target.len() < n {
        return None;
    }
    for k in 0..n {
        // SAFETY of indexing: target.len() >= n checked above.
        let (q, t) = (j.query[k], j.target[k]);
        if q >= 4 || t >= 4 || q != t {
            return None;
        }
    }
    let score = j.h0 + n as i32 * a;
    Some(ExtendResult {
        score,
        qle: n as i32,
        tle: n as i32,
        gtle: n as i32,
        gscore: score,
        max_off: 0,
    })
}

/// SIMD dispatch (NEON on aarch64, AVX2 on x86_64): ungapped-HIT fast path, then bin each remaining
/// job by length into the u8 (16/32 lanes) / i16 (8/16 lanes) / scalar kernel and scatter back. This
/// is bwa-mem2's `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` binning. Result-preserving (each job's extension
/// depends only on its own inputs).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn simd_dispatch(
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
    // The NEON kernels score DNA with a vector compare that assumes the uniform bwa matrix
    // (diagonal `a`, off-diagonal `mm`, ambiguous row/col = `npen`). Any other matrix -> scalar.
    if !is_uniform_dna(mat, m) {
        return batched_extend_scalar(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }

    let a = mat[0] as i32; // match score for the uniform matrix

    // Ungapped diagonal HIT fast path: a perfect-diagonal extension gets its result in closed form,
    // skipping banded SW. Common on clean reads, so this removes a large slice of DP work while
    // staying byte-identical (see `ungapped_hit`). Non-HIT jobs fall through to the kernel binning.
    let mut out = vec![default_result(); jobs.len()];
    let mut rest: Vec<usize> = Vec::new();
    for (k, j) in jobs.iter().enumerate() {
        match ungapped_hit(j, a) {
            Some(r) => out[k] = r,
            None => rest.push(k),
        }
    }
    if rest.is_empty() {
        return out;
    }
    if rest.len() == jobs.len() {
        // No HITs: bin `jobs` directly (avoid the gather).
        return dispatch_bins(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }
    let sub: Vec<ExtendJob> = rest.iter().map(|&k| jobs[k]).collect();
    let res = dispatch_bins(
        &sub, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    );
    for (p, &k) in rest.iter().enumerate() {
        out[k] = res[p];
    }
    out
}

/// Length/score binning + kernel dispatch for the jobs that are not ungapped HITs. Bins each into
/// int8 (16 lanes) / int16 (8 lanes) / scalar, runs each bin, scatters back. This is bwa-mem2's
/// `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` binning; the 8-bit path packs twice the lanes for short pairs.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn dispatch_bins(
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
    // The 8-bit kernel is **unsigned u8** (values 0..=255, positions <256), so it takes any job whose
    // score ceiling `minval` and both lengths fit under 256 — twice the reach of bwa-mem2's signed
    // `MAX_SEQ_LEN8=128`, hence far more jobs run in 16 lanes vs int16's 8. Both kernels are exact in
    // range, so this only changes which one runs, never the result.
    const U8_LEN: usize = 256;
    const MAX_SEQ_LEN16: usize = 32768;
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    let (mut u8_idx, mut i16_idx, mut sc_idx) = (Vec::new(), Vec::new(), Vec::new());
    for (k, j) in jobs.iter().enumerate() {
        let minval = cell_bound(j, max_sc);
        let (ql, tl) = (j.query.len(), j.target.len());
        if ql < U8_LEN && tl < U8_LEN && minval < U8_LEN as i32 {
            u8_idx.push(k);
        } else if ql < MAX_SEQ_LEN16 && tl < MAX_SEQ_LEN16 && minval < MAX_SEQ_LEN16 as i32 {
            i16_idx.push(k);
        } else {
            sc_idx.push(k);
        }
    }

    // Homogeneous fast path: whole batch in one bin -> run the kernel on `jobs` with no gather/scatter.
    let n = jobs.len();
    if u8_idx.len() == n {
        // SAFETY: neon available (checked by caller); U8_LEN bounds keep all values/positions in u8.
        return unsafe {
            sw_kernel_u8(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            )
        };
    }
    if i16_idx.len() == n {
        // SAFETY: neon available; MAX_SEQ_LEN16 bounds keep all values inside i16.
        return unsafe {
            sw_kernel_i16(
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
    run(&u8_idx, &|s| unsafe {
        // SAFETY: neon available (checked by caller); U8_LEN bounds keep all values/positions in u8.
        sw_kernel_u8(s, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop)
    });
    run(&i16_idx, &|s| unsafe {
        // SAFETY: neon available; MAX_SEQ_LEN16 bounds keep all values inside i16.
        sw_kernel_i16(s, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop)
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
                    // Gaps open from H (= max(M, E, F)), matching bandedSWA / `ksw_extend2`.
                    let t = (h - oe_del).max(0);
                    e = (e - e_del).max(t);
                    eh_e[base + ju] = e;
                    let t = (h - oe_ins).max(0);
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

/// Generate a batched banded-SW kernel for a lane element type + target-feature string. One macro
/// serves NEON (aarch64), AVX2 and AVX-512 (x86_64): every SIMD op is a parameter, so the width
/// variants and the ISAs cannot drift. Layout is SoA `[column*LANES + lane]` (bwa-mem2's `bandedSWA`);
/// the band mask is a blendv select; the recurrence uses saturating (u8) or plain (i16) add/sub,
/// exact within the caller-guaranteed range. Expanded inside a per-ISA module that imports the
/// intrinsics and `super::{clamp_band, default_result}` / `bwa_extend::{ExtendJob, ExtendResult}`.
macro_rules! define_sw_kernel {
    (
            $name:ident, $elem:ty, $umask:ty, $lanes:expr, feat = $feat:literal,
            dup = $dup:path, lds = $lds:path, sts = $sts:path, ldu = $ldu:path,
            add = $add:path, sub = $sub:path, max = $max:path,
            ceqz = $ceqz:path, cge = $cge:path, clt = $clt:path,
            ceq = $ceq:path, orru = $orru:path,
            andu = $andu:path, bsl = $bsl:path
        ) => {
        /// # Safety
        /// Requires the `$feat` target feature; all vector loads/stores use fixed-size
        /// `[$elem; LANES]` / `[$umask; LANES]` scratch arrays, in-bounds by construction.
        #[target_feature(enable = $feat)]
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
            // DNA score as a vector compare (no per-cell gather): the caller (neon_dispatch) only
            // reaches here for the uniform bwa matrix. The signed substitution score
            // (`N ? npen : (t==q ? a : mm)`) is kept as its **positive** parts so the recurrence
            // works in unsigned-saturating u8 as well as signed i16: `sbt_pos` (match bonus `a`)
            // via saturating add, `sbt_neg` (mismatch `|mm|` / ambiguous `|npen|`) via saturating
            // sub. For the signed kernels `(m + pos) - neg == m + score` exactly (no wrap), so this
            // is byte-identical; for u8 it is bwa-mem2's `MAIN_CODE8_CORE`.
            let a_pos_v = $dup(mat[0] as $elem); // a >= 0
            let mm_mag_v = $dup((-(i32::from(mat[1]))) as $elem); // |mm|
            let npen_mag_v = $dup((-(i32::from(mat[m - 1]))) as $elem); // |npen|
            let amb_v = $dup(4); // code 4 = N; codes are 0..=4

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

                let stride = max_q + 1;
                let mut eh_h = vec![0 as $elem; stride * LANES];
                let mut eh_e = vec![0 as $elem; stride * LANES];

                // Per-lane query/target codes padded (0 past len) for branchless, in-bounds
                // vector loads of the 8/16 lanes' base at column j (query) or row i (target).
                let mut qcode = vec![0 as $elem; stride * LANES];
                let mut tcode = vec![0 as $elem; (max_t + 1) * LANES];
                for l in 0..nlane {
                    for (ju, &c) in jobs[chunk_start + l].query.iter().enumerate() {
                        qcode[ju * LANES + l] = c as $elem;
                    }
                    for (iu, &c) in jobs[chunk_start + l].target.iter().enumerate() {
                        tcode[iu * LANES + l] = c as $elem;
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
                    let mut mj_v = $dup((-1i32) as $elem);

                    let mut beg_a = [0 as $elem; LANES];
                    let mut end_a = [0 as $elem; LANES];
                    let mut act_a = [0 as $umask; LANES];
                    for l in 0..nlane {
                        if active[l] {
                            beg_a[l] = beg[l] as $elem;
                            end_a[l] = end[l] as $elem;
                            act_a[l] = <$umask>::MAX;
                        }
                    }
                    let beg_v = $lds(beg_a.as_ptr());
                    let end_v = $lds(end_a.as_ptr());
                    let active_v = $ldu(act_a.as_ptr());
                    let t_v = $lds(tcode.as_ptr().add(i as usize * LANES)); // this row's target base per lane
                    let t_is_n = $cge(t_v, amb_v); // target base is N — constant across the row

                    for j in gbeg..gend {
                        let jrow = j as usize * LANES;
                        let j_v = $dup(j as $elem);
                        let band = $andu(active_v, $andu($cge(j_v, beg_v), $clt(j_v, end_v)));

                        // DNA substitution score split into positive parts (no gather):
                        //   sbt_pos = (t==q && !N) ? a : 0 ;  sbt_neg = N ? |npen| : (t==q ? 0 : |mm|)
                        let q_v = $lds(qcode.as_ptr().add(jrow));
                        let is_eq = $ceq(t_v, q_v);
                        let is_n = $orru(t_is_n, $cge(q_v, amb_v));
                        let sbt_pos = $bsl(is_n, zero_v, $bsl(is_eq, a_pos_v, zero_v));
                        let sbt_neg = $bsl(is_n, npen_mag_v, $bsl(is_eq, zero_v, mm_mag_v));

                        let m_v = $lds(eh_h.as_ptr().add(jrow)); // H(i-1, j-1)
                        let e_v = $lds(eh_e.as_ptr().add(jrow)); // E(i-1, j)

                        // eh_h[j] <- h1 (old) for in-band lanes; out-of-band keep old m_v.
                        $sts(eh_h.as_mut_ptr().add(jrow), $bsl(band, h1_v, m_v));

                        // M = ((m + sbt_pos) - sbt_neg); m==0 -> local restart (0). Saturating for
                        // u8, plain for signed; identical either way in the guaranteed range.
                        let bigm_pre = $sub($add(m_v, sbt_pos), sbt_neg);
                        let bigm_v = $bsl($ceqz(m_v), zero_v, bigm_pre);

                        let h_v = $max($max(bigm_v, e_v), f_v);
                        h1_v = $bsl(band, h_v, h1_v);

                        // if row_max <= h { mj = j; row_max = h } (ties take larger j)
                        let upd = $andu(band, $cge(h_v, rowmax_v));
                        rowmax_v = $bsl(upd, h_v, rowmax_v);
                        mj_v = $bsl(upd, $dup(j as $elem), mj_v);

                        // e = max(e - e_del, max(h - oe_del, 0)); gaps open from H (bandedSWA).
                        let t1 = $max($sub(h_v, oe_del_v), zero_v);
                        let e_new = $max($sub(e_v, e_del_v), t1);
                        $sts(eh_e.as_mut_ptr().add(jrow), $bsl(band, e_new, e_v));

                        // f = max(f - e_ins, max(h - oe_ins, 0)) in-band
                        let t2 = $max($sub(h_v, oe_ins_v), zero_v);
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
    };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{clamp_band, default_result};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::aarch64::*;

    define_sw_kernel!(
        batched_extend_neon_i16,
        i16,
        u16,
        8,
        feat = "neon",
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
        ceq = vceqq_s16,
        orru = vorrq_u16,
        andu = vandq_u16,
        bsl = vbslq_s16
    );

    // 8-bit kernel: **unsigned** u8 [0,255] with saturating add/sub, so a local extension whose score
    // ceiling lands in [128,255] (bwa-mem2 would route it to int16) still runs in 16 lanes. Values are
    // non-negative and positions are `< 256`, so u8 holds both. This is bwa-mem2's `smithWaterman*_8`.
    define_sw_kernel!(
        batched_extend_neon_u8,
        u8,
        u8,
        16,
        feat = "neon",
        dup = vdupq_n_u8,
        lds = vld1q_u8,
        sts = vst1q_u8,
        ldu = vld1q_u8,
        add = vqaddq_u8,
        sub = vqsubq_u8,
        max = vmaxq_u8,
        ceqz = vceqzq_u8,
        cge = vcgeq_u8,
        clt = vcltq_u8,
        ceq = vceqq_u8,
        orru = vorrq_u8,
        andu = vandq_u8,
        bsl = vbslq_u8
    );
}

/// AVX2 (x86_64) instantiations of [`define_sw_kernel`]: 32 u8 lanes / 16 i16 lanes (256-bit, twice
/// NEON's width). x86 SIMD lacks unsigned integer compares and its blend has a different argument
/// order than NEON's `vbslq`, so a handful of `#[target_feature("avx2")]` wrappers adapt the ops to
/// the macro's interface. Byte-identical to the scalar reference by the same construction as NEON
/// (verified via the force-run property test compiled to x86 and executed under Rosetta).
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{clamp_band, default_result};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::x86_64::*;

    // set1 (dup): the macro passes an `$elem`-typed scalar; reinterpret to the epi lane type.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn set1_u8(x: u8) -> __m256i {
        _mm256_set1_epi8(x as i8)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn set1_i16(x: i16) -> __m256i {
        _mm256_set1_epi16(x)
    }

    // loads/stores: the macro hands typed element pointers into `[$elem; LANES]` / `[$umask; LANES]`
    // scratch; a 256-bit unaligned move covers the whole array (32 u8 or 16 u16/i16).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_u8(p: *const u8) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_u16(p: *const u16) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_i16(p: *const i16) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn storeu_u8(p: *mut u8, v: __m256i) {
        _mm256_storeu_si256(p as *mut __m256i, v)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn storeu_i16(p: *mut i16, v: __m256i) {
        _mm256_storeu_si256(p as *mut __m256i, v)
    }

    // x == 0.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceqz8(v: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(v, _mm256_setzero_si256())
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceqz16(v: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(v, _mm256_setzero_si256())
    }

    // a >= b via `max(a,b) == a` (works unsigned via max_epu8, signed via max_epi16); a < b = !(a>=b).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn clt_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(cge_epu8(a, b), _mm256_set1_epi8(-1))
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epi16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(_mm256_max_epi16(a, b), a)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn clt_epi16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(cge_epi16(a, b), _mm256_set1_epi8(-1))
    }

    // blend select: NEON `vbslq(mask, a, b)` = mask ? a : b; AVX2 `blendv_epi8(a, b, mask)` = mask ? b : a.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bsl256(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_blendv_epi8(b, a, mask)
    }

    define_sw_kernel!(
        batched_extend_avx2_i16,
        i16,
        u16,
        16,
        feat = "avx2",
        dup = set1_i16,
        lds = loadu_i16,
        sts = storeu_i16,
        ldu = loadu_u16,
        add = _mm256_add_epi16,
        sub = _mm256_sub_epi16,
        max = _mm256_max_epi16,
        ceqz = ceqz16,
        cge = cge_epi16,
        clt = clt_epi16,
        ceq = _mm256_cmpeq_epi16,
        orru = _mm256_or_si256,
        andu = _mm256_and_si256,
        bsl = bsl256
    );

    define_sw_kernel!(
        batched_extend_avx2_u8,
        u8,
        u8,
        32,
        feat = "avx2",
        dup = set1_u8,
        lds = loadu_u8,
        sts = storeu_u8,
        ldu = loadu_u8,
        add = _mm256_adds_epu8,
        sub = _mm256_subs_epu8,
        max = _mm256_max_epu8,
        ceqz = ceqz8,
        cge = cge_epu8,
        clt = clt_epu8,
        ceq = _mm256_cmpeq_epi8,
        orru = _mm256_or_si256,
        andu = _mm256_and_si256,
        bsl = bsl256
    );
}

/// Force-run verification of the AVX2 kernels against the scalar `ksw_extend2`, byte-for-byte.
///
/// On x86_64 the AVX2 path only *runs* when `is_x86_feature_detected!("avx2")`, which Rosetta
/// reports as `false` even though it *executes* AVX2 instructions. So this test calls the AVX2
/// kernels directly (bypassing detection), which is how the port is validated on this Apple-Silicon
/// host via `cargo test --target x86_64-apple-darwin` (Rosetta). On a native x86 CI runner (which has
/// AVX2) it validates the real path. Requires an AVX2-capable executor.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_verify {
    use bwa_extend::{ksw_extend2, ExtendJob};

    fn scoring() -> Vec<i8> {
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
        mat
    }

    #[test]
    fn avx2_u8_and_i16_match_scalar() {
        let mat = scoring();
        let mut state = 0xA7C2_0000_0000_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..200u32 {
            let w = 1 + (next() % 150) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            let batch = *[1usize, 8, 16, 17, 32, 33, 48]
                .get((next() % 7) as usize)
                .unwrap();

            // Keep both lengths and minval < 256 so the u8 kernel is exact, plus a longer-length set
            // (minval up to a few thousand) for the i16 kernel.
            for &big in &[false, true] {
                let mut queries: Vec<Vec<u8>> = Vec::new();
                let mut targets: Vec<Vec<u8>> = Vec::new();
                let mut h0s: Vec<i32> = Vec::new();
                for _ in 0..batch {
                    let qlen = if big {
                        200 + (next() % 400) as usize
                    } else {
                        1 + (next() % 90) as usize
                    };
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                    let tlen = qlen + (next() % 30) as usize;
                    let mut t: Vec<u8> = Vec::with_capacity(tlen);
                    let mut qi = 0usize;
                    while t.len() < tlen {
                        if qi < q.len() && next() % 100 >= 5 {
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
                    h0s.push(1 + (next() % 20) as i32);
                }
                let jobs: Vec<ExtendJob> = (0..batch)
                    .map(|i| ExtendJob {
                        query: &queries[i],
                        target: &targets[i],
                        h0: h0s[i],
                    })
                    .collect();
                // SAFETY: this test requires an AVX2-capable executor (native x86 CI or Rosetta).
                let got = unsafe {
                    if big {
                        super::avx2::batched_extend_avx2_i16(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    } else {
                        super::avx2::batched_extend_avx2_u8(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    }
                };
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
                        "AVX2 {} diverged round {round} job {i} qlen={} tlen={}",
                        if big { "i16" } else { "u8" },
                        queries[i].len(),
                        targets[i].len()
                    );
                }
            }
        }
    }
}

/// Native-NEON byte-identity gate, runnable on this Apple-Silicon host (`cargo test -p bwa-neon`).
///
/// The `avx2_verify` test validates the shared `define_sw_kernel!` logic, but only on x86 (under
/// Rosetta / native x86 CI). This module exercises the actual NEON u8 (16-lane) and i16 (8-lane)
/// kernels that ship on aarch64, asserting every `ExtendResult` field matches the scalar
/// `ksw_extend2` reference across randomized short/long jobs and batch sizes that straddle the lane
/// boundary. It is the on-device counterpart to `avx2_verify`.
#[cfg(all(test, target_arch = "aarch64"))]
mod neon_verify {
    use super::neon::{batched_extend_neon_i16, batched_extend_neon_u8};
    use bwa_extend::{ksw_extend2, ExtendJob};

    fn scoring() -> Vec<i8> {
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
        mat
    }

    #[test]
    fn neon_u8_and_i16_match_scalar() {
        let mat = scoring();
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..400u32 {
            let w = 1 + (next() % 150) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            // Straddle the 8/16-lane boundaries: partial and empty tail lanes, exact multiples.
            let batch = *[1usize, 7, 8, 9, 16, 17, 31, 32, 33, 48]
                .get((next() % 10) as usize)
                .unwrap();

            for &big in &[false, true] {
                let mut queries: Vec<Vec<u8>> = Vec::new();
                let mut targets: Vec<Vec<u8>> = Vec::new();
                let mut h0s: Vec<i32> = Vec::new();
                for _ in 0..batch {
                    // u8 kernel: lengths + score ceiling < 256; i16 kernel: a longer-length set.
                    let qlen = if big {
                        200 + (next() % 400) as usize
                    } else {
                        1 + (next() % 90) as usize
                    };
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                    let tlen = qlen + (next() % 30) as usize;
                    let mut t: Vec<u8> = Vec::with_capacity(tlen);
                    let mut qi = 0usize;
                    while t.len() < tlen {
                        if qi < q.len() && next() % 100 >= 5 {
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
                    h0s.push(1 + (next() % 20) as i32);
                }
                let jobs: Vec<ExtendJob> = (0..batch)
                    .map(|i| ExtendJob {
                        query: &queries[i],
                        target: &targets[i],
                        h0: h0s[i],
                    })
                    .collect();
                // SAFETY: this host has NEON (aarch64).
                let got = unsafe {
                    if big {
                        batched_extend_neon_i16(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    } else {
                        batched_extend_neon_u8(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    }
                };
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
                        "NEON {} diverged round {round} job {i} (batch {batch}) qlen={} tlen={}",
                        if big { "i16" } else { "u8" },
                        queries[i].len(),
                        targets[i].len()
                    );
                }
            }
        }
    }
}

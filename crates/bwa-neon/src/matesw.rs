//! Lane-batched local Smith-Waterman for **mate rescue** (`kswv` in bwa-mem2).
//!
//! Mate rescue realigns a mate read against an insert-size window when its pairing is missing. Each
//! such alignment is an independent full local SW ([`bwa_extend::ksw_align2`]) returning
//! `{score, qb, qe, tb, te, score2, te2}` (no CIGAR/traceback). bwa-mem2 vectorizes this
//! **inter-sequence**: many rescue jobs packed into SoA lanes (16 x u8 / 8 x i16 on NEON), each lane
//! a different job, length-sorted so lanes finish together. This mirrors [`crate::batched`] (seed
//! extension) but with the local-SW recurrence and the two-phase start recovery (`KSW_XSTART`).
//!
//! [`batched_ksw_align2`] returns one [`KswAlignResult`] per job, each byte-identical to
//! [`bwa_extend::ksw_align2`] on that job. The scalar per-job loop is the portable fallback and the
//! source of truth the NEON kernels are validated against (`matesw_equals_scalar`).

use bwa_extend::KswAlignResult;

/// One mate-rescue local-SW job: align `query` against `target` (both `0..=4` codes).
#[derive(Clone, Copy)]
pub struct KswJob<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
}

/// One forward local-SW pass: the target is `target`, the query is `query`, and the pass reports the
/// max score reaching `>= minsc` and stops early once it reaches `endsc`. This is the unit the
/// vectorized kernel batches; [`batched_ksw_align2`] issues one batch for the forward pass and one for
/// the reverse (`KSW_XSTART`) start-recovery pass.
#[derive(Clone, Copy)]
struct FwdJob<'a> {
    query: &'a [u8],
    target: &'a [u8],
    minsc: i32,
    endsc: i32,
}

/// Batched local SW: `out[i]` equals [`bwa_extend::ksw_align2`] on `jobs[i]`. Structured exactly like
/// `ksw_align2`: a forward pass over every job, then a reverse pass over the truncated/reversed
/// prefixes of the qualifying jobs to recover the start coordinates. Both passes go through
/// [`fwd_local_sw_batch`], the single point the NEON kernel plugs into.
#[allow(clippy::too_many_arguments)]
pub fn batched_ksw_align2(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Vec<KswAlignResult> {
    // Forward pass over all jobs.
    let fwd: Vec<FwdJob> = jobs
        .iter()
        .map(|j| FwdJob {
            query: j.query,
            target: j.target,
            minsc,
            endsc: i32::MAX,
        })
        .collect();
    let fr = fwd_local_sw_batch(&fwd, m, mat, o_del, e_del, o_ins, e_ins, max_sc);

    let mut out: Vec<KswAlignResult> = fr
        .iter()
        .map(|&(score, te, qe, score2, te2)| KswAlignResult {
            score,
            qb: -1,
            qe,
            tb: -1,
            te,
            score2,
            te2,
        })
        .collect();

    // Reverse pass (KSW_XSTART): for each qualifying job, align the reversed prefixes ending at
    // (qe, te) and stop at `score`; the reversed end offsets give the start coords.
    let mut rbufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut ridx: Vec<usize> = Vec::new();
    for (i, j) in jobs.iter().enumerate() {
        let (score, te, qe) = (out[i].score, out[i].te, out[i].qe);
        if score >= minsc && qe >= 0 {
            let qrev: Vec<u8> = j.query[..=qe as usize].iter().rev().copied().collect();
            let trev: Vec<u8> = j.target[..=te as usize].iter().rev().copied().collect();
            rbufs.push((qrev, trev));
            ridx.push(i);
        }
    }
    let rjobs: Vec<FwdJob> = rbufs
        .iter()
        .zip(ridx.iter())
        .map(|((q, t), &i)| FwdJob {
            query: q,
            target: t,
            minsc: i32::MAX,
            endsc: out[i].score,
        })
        .collect();
    let rr = fwd_local_sw_batch(&rjobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
    for (k, &i) in ridx.iter().enumerate() {
        let (rscore, rte, rqe, _, _) = rr[k];
        if out[i].score == rscore {
            out[i].tb = out[i].te - rte;
            out[i].qb = out[i].qe - rqe;
        }
    }
    out
}

/// Lanes processed in lockstep per group. 8 = one NEON `int16x8`.
const LANES: usize = 8;

/// Query-column / target-row padding sentinel (`>= m`, so its cell score is forced very negative and
/// the padded cells stay `0` — neutral to the real lanes).
const PAD: u8 = 255;

/// Batched forward local-SW pass: `out[i] = (score, te, qe, score2, te2)` for `jobs[i]`, each equal to
/// [`ksw_local_fwd`]. Dispatches to the NEON i16 kernel where available, else the scalar lockstep.
#[allow(clippy::too_many_arguments)]
fn fwd_local_sw_batch(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && mat_is_standard(m, mat) {
            // Max reachable score per job = min(len) * match. Only the SCORE cells (H/E/F) live in the
            // SIMD vector; positions/te/qe are scalar i32, so window length is unconstrained. If every
            // job's score ceiling fits u8, run 16 lanes; else the i16 kernel at 8 lanes. Mate-rescue
            // jobs (short reads, match ~1) fit u8.
            let bound = |j: &FwdJob| j.query.len().min(j.target.len()) as i32 * max_sc;
            if jobs.iter().all(|j| bound(j) < 250) {
                // SAFETY: neon detected; every H/E/F cell < 250 fits u8; standard mat.
                return unsafe {
                    fwd_local_sw_neon_u8(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
            }
            if jobs.iter().all(|j| bound(j) < 30000 && j.target.len() < 30000) {
                // SAFETY: neon detected; i16 range guaranteed; standard mat.
                return unsafe {
                    fwd_local_sw_neon(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
                };
            }
        }
    }
    fwd_local_sw_scalar(jobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc)
}

/// Whether `mat` is bwa's standard 5x5 form (uniform match on the diagonal, uniform mismatch
/// off-diagonal for `0..4`, `-1` for any N row/col) so the NEON kernel can compute cell scores from
/// three scalars instead of a per-cell table lookup.
fn mat_is_standard(m: usize, mat: &[i8]) -> bool {
    if m != 5 {
        return false;
    }
    let (mtch, mis) = (mat[0], mat[1]);
    for i in 0..4 {
        for j in 0..4 {
            let want = if i == j { mtch } else { mis };
            if mat[i * 5 + j] != want {
                return false;
            }
        }
        if mat[i * 5 + 4] != -1 || mat[4 * 5 + i] != -1 {
            return false;
        }
    }
    mat[4 * 5 + 4] == -1
}

/// Scalar lockstep reference: processes `LANES` jobs with shared row/column loops and per-lane state
/// and masking, the structure the NEON kernel vectorizes. The scalar per-cell arithmetic here is the
/// byte-identity source of truth (`matesw_equals_scalar`).
#[allow(clippy::too_many_arguments)]
fn fwd_local_sw_scalar(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mut out = vec![(0i32, -1i32, -1i32, 0i32, -1i32); jobs.len()];

    for (g, group) in jobs.chunks(LANES).enumerate() {
        let n = group.len();
        let qmax = group.iter().map(|j| j.query.len()).max().unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // SoA sequences (padded), per-lane bounds/params.
        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        let (mut qlen, mut tlen, mut minsc, mut endsc) =
            ([0usize; LANES], [0usize; LANES], [i32::MAX; LANES], [i32::MAX; LANES]);
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES + l] = b;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = b;
            }
        }

        // DP state (SoA over query columns).
        let mut h_prev = vec![0i32; qmax * LANES];
        let mut h_cur = vec![0i32; qmax * LANES];
        let mut e = vec![0i32; qmax * LANES];
        let mut hmax_col = vec![0i32; qmax * LANES];
        let mut rowmax = vec![0i32; tmax * LANES]; // per-row imax, for score2
        let mut gmax = [0i32; LANES];
        let mut te = [-1i32; LANES];
        let mut limit = [-1i32; LANES]; // last processed row (inclusive)
        let mut frozen = [false; LANES];
        for l in 0..n {
            limit[l] = tlen[l] as i32 - 1;
        }

        for i in 0..tmax {
            let mut f = [0i32; LANES];
            let mut h_diag = [0i32; LANES];
            let mut imax = [0i32; LANES];
            for j in 0..qmax {
                for l in 0..LANES {
                    let t = seq_t[i * LANES + l] as usize;
                    let q = seq_q[j * LANES + l] as usize;
                    let sc = if t >= m || q >= m {
                        -30000
                    } else {
                        i32::from(mat[t * m + q])
                    };
                    let idx = j * LANES + l;
                    let mut h = h_diag[l] + sc;
                    if h < 0 {
                        h = 0;
                    }
                    if e[idx] > h {
                        h = e[idx];
                    }
                    if f[l] > h {
                        h = f[l];
                    }
                    if h > imax[l] {
                        imax[l] = h;
                    }
                    h_cur[idx] = h;
                    let mut en = e[idx] - e_del;
                    let td = h - oe_del;
                    if td > en {
                        en = td;
                    }
                    e[idx] = en.max(0);
                    let mut fnv = f[l] - e_ins;
                    let ti = h - oe_ins;
                    if ti > fnv {
                        fnv = ti;
                    }
                    f[l] = fnv.max(0);
                    h_diag[l] = h_prev[idx];
                }
            }
            // Per-row bookkeeping (only active lanes: within target and not frozen).
            for l in 0..n {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                rowmax[i * LANES + l] = imax[l];
                if imax[l] > gmax[l] {
                    gmax[l] = imax[l];
                    te[l] = i as i32;
                    for j in 0..qmax {
                        hmax_col[j * LANES + l] = h_cur[j * LANES + l];
                    }
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);
        }

        extract_group(
            n, g, LANES, &qlen, &minsc, max_sc, &gmax, &te, &limit, &rowmax, &hmax_col, &mut out,
        );
    }
    out
}

/// Shared per-lane output extraction (`qe`, `score2`/`te2`) from a group's filled DP state, exactly
/// as [`ksw_local_fwd`]. Used by both the scalar and NEON DP paths so they cannot drift. `rowmax`
/// (per-row imax) and `hmax_col` (H column at the best row) are SoA `[pos*LANES + lane]`.
#[allow(clippy::too_many_arguments)]
fn extract_group(
    n: usize,
    g: usize,
    lanes: usize,
    qlen: &[usize],
    minsc: &[i32],
    max_sc: i32,
    gmax: &[i32],
    te: &[i32],
    limit: &[i32],
    rowmax: &[i32],
    hmax_col: &[i32],
    out: &mut [(i32, i32, i32, i32, i32)],
) {
    for l in 0..n {
        let g_ = gmax[l];
        let te_ = te[l];
        // qe = smallest query column reaching the H max at the best target row (min index on tie).
        let mut qe = -1i32;
        if te_ >= 0 {
            let mut mx = -1i32;
            for j in 0..qlen[l] {
                let v = hmax_col[j * lanes + l];
                if v > mx {
                    mx = v;
                    qe = j as i32;
                }
            }
        }
        // score2: rebuild ksw_local_fwd's `b` list (row-maxes >= minsc, consecutive rows merged
        // keeping the higher AND advancing the column only on an update), then take the best entry
        // whose column lies outside [te - w, te + w].
        let mut score2 = 0i32;
        let mut te2 = -1i32;
        let mut b: Vec<(i32, i32)> = Vec::new();
        if limit[l] >= 0 {
            for i in 0..=limit[l] {
                let im = rowmax[i as usize * lanes + l];
                if im >= minsc[l] {
                    match b.last() {
                        Some(&(_, col)) if col + 1 == i => {
                            if b.last().unwrap().0 < im {
                                *b.last_mut().unwrap() = (im, i);
                            }
                        }
                        _ => b.push((im, i)),
                    }
                }
            }
        }
        if g_ > 0 && !b.is_empty() {
            let w = (g_ + max_sc - 1) / max_sc;
            let (low, high) = (te_ - w, te_ + w);
            for &(bs, bc) in &b {
                if (bc < low || bc > high) && bs > score2 {
                    score2 = bs;
                    te2 = bc;
                }
            }
        }
        out[g * lanes + l] = (g_, te_, qe, score2, te2);
    }
}

/// NEON i16x8 forward local-SW. Vectorizes the [`fwd_local_sw_scalar`] control flow across `LANES`
/// jobs: the inner cell recurrence runs on `int16x8` (one lane per job), the per-row bookkeeping and
/// [`extract_group`] stay scalar. Requires the standard 5x5 `mat` (checked by the caller) so a cell
/// score is `match`/`mismatch`/`-1(N)` chosen by compares. Every value must fit i16 (caller-guarded).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_neon(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::aarch64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as i16;
    let mis = mat[1] as i16;
    let mut out = vec![(0i32, -1i32, -1i32, 0i32, -1i32); jobs.len()];

    // Broadcast constants.
    let zero = vdupq_n_s16(0);
    let mtch_v = vdupq_n_s16(mtch);
    let mis_v = vdupq_n_s16(mis);
    let n_v = vdupq_n_s16(-1);
    let neg_v = vdupq_n_s16(-30000);
    let four_v = vdupq_n_s16(4);
    let m_v = vdupq_n_s16(m as i16);
    let e_del_v = vdupq_n_s16(e_del as i16);
    let oe_del_v = vdupq_n_s16(oe_del as i16);
    let e_ins_v = vdupq_n_s16(e_ins as i16);
    let oe_ins_v = vdupq_n_s16(oe_ins as i16);

    for (g, group) in jobs.chunks(LANES).enumerate() {
        let n = group.len();
        let qmax = group.iter().map(|j| j.query.len()).max().unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        // SoA sequences (u8, padded) + per-lane bounds/params.
        let mut seq_q = vec![PAD; qmax * LANES];
        let mut seq_t = vec![PAD; tmax * LANES];
        let (mut qlen, mut tlen, mut minsc, mut endsc) =
            ([0usize; LANES], [0usize; LANES], [i32::MAX; LANES], [i32::MAX; LANES]);
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES + l] = b;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES + l] = b;
            }
        }

        // i16 SoA DP state.
        let mut h_prev = vec![0i16; qmax * LANES];
        let mut h_cur = vec![0i16; qmax * LANES];
        let mut e = vec![0i16; qmax * LANES];
        let mut hmax_col = vec![0i32; qmax * LANES];
        let mut rowmax = vec![0i32; tmax * LANES];
        let mut gmax = [0i32; LANES];
        let mut te = [-1i32; LANES];
        let mut limit = [-1i32; LANES];
        let mut frozen = [false; LANES];
        for l in 0..n {
            limit[l] = tlen[l] as i32 - 1;
        }

        // Widen 8 u8 codes at `off` into an int16x8 (lanes = jobs).
        let load_codes = |buf: &[u8], off: usize| -> int16x8_t {
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(buf.as_ptr().add(off))))
        };

        for i in 0..tmax {
            let t_v = load_codes(&seq_t, i * LANES);
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            for j in 0..qmax {
                let q_v = load_codes(&seq_q, j * LANES);
                // Cell score: match/mismatch, then N override (-1), then padding override (very neg).
                let eq = vceqq_s16(t_v, q_v);
                let n_mask = vorrq_u16(vceqq_s16(t_v, four_v), vceqq_s16(q_v, four_v));
                let pad_mask = vorrq_u16(vcgeq_s16(t_v, m_v), vcgeq_s16(q_v, m_v));
                let mut sc = vbslq_s16(eq, mtch_v, mis_v);
                sc = vbslq_s16(n_mask, n_v, sc);
                sc = vbslq_s16(pad_mask, neg_v, sc);

                let e_v = vld1q_s16(e.as_ptr().add(j * LANES));
                let mut h_v = vaddq_s16(h_diag_v, sc);
                h_v = vmaxq_s16(h_v, zero);
                h_v = vmaxq_s16(h_v, e_v);
                h_v = vmaxq_s16(h_v, f_v);
                imax_v = vmaxq_s16(imax_v, h_v);
                vst1q_s16(h_cur.as_mut_ptr().add(j * LANES), h_v);

                let en = vmaxq_s16(vsubq_s16(e_v, e_del_v), vsubq_s16(h_v, oe_del_v));
                vst1q_s16(e.as_mut_ptr().add(j * LANES), vmaxq_s16(en, zero));
                let fnv = vmaxq_s16(vsubq_s16(f_v, e_ins_v), vsubq_s16(h_v, oe_ins_v));
                f_v = vmaxq_s16(fnv, zero);
                h_diag_v = vld1q_s16(h_prev.as_ptr().add(j * LANES));
            }

            // Per-row bookkeeping (scalar per lane).
            let mut imax_arr = [0i16; LANES];
            vst1q_s16(imax_arr.as_mut_ptr(), imax_v);
            for l in 0..n {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                let im = imax_arr[l] as i32;
                rowmax[i * LANES + l] = im;
                if im > gmax[l] {
                    gmax[l] = im;
                    te[l] = i as i32;
                    for j in 0..qmax {
                        hmax_col[j * LANES + l] = h_cur[j * LANES + l] as i32;
                    }
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            // Early exit once no lane can still advance.
            if (0..n).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n, g, LANES, &qlen, &minsc, max_sc, &gmax, &te, &limit, &rowmax, &hmax_col, &mut out,
        );
    }
    out
}

/// Lanes for the u8 kernel: one NEON `uint8x16`, twice the i16 width.
const LANES16: usize = 16;

/// NEON u8x16 forward local-SW: same control flow as [`fwd_local_sw_neon`] but 16 lanes. Local
/// alignment keeps every H/E/F non-negative, so **saturating** u8 arithmetic (`vqadd`/`vqsub`)
/// realizes `max(0, .)` directly with no bias/shift: the caller guarantees each job's score ceiling
/// `min(len)*match` fits u8. Positions (`te`/`qe`/`rowmax`) stay scalar i32, so window length is
/// unconstrained. Byte-identical to the scalar path (validated by `matesw_equals_scalar`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fwd_local_sw_neon_u8(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    use std::arch::aarch64::*;

    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let mtch = mat[0] as u8; // match bonus (>= 0)
    let mispen = (-mat[1]) as u8; // mismatch penalty b (mat[1] = -b)
    let mut out = vec![(0i32, -1i32, -1i32, 0i32, -1i32); jobs.len()];

    let zero = vdupq_n_u8(0);
    let mtch_v = vdupq_n_u8(mtch);
    let mispen_v = vdupq_n_u8(mispen);
    let one_v = vdupq_n_u8(1); // N penalty
    let four_v = vdupq_n_u8(4);
    let m_v = vdupq_n_u8(m as u8);
    let e_del_v = vdupq_n_u8(e_del as u8);
    let oe_del_v = vdupq_n_u8(oe_del as u8);
    let e_ins_v = vdupq_n_u8(e_ins as u8);
    let oe_ins_v = vdupq_n_u8(oe_ins as u8);

    for (g, group) in jobs.chunks(LANES16).enumerate() {
        let n = group.len();
        let qmax = group.iter().map(|j| j.query.len()).max().unwrap_or(0);
        let tmax = group.iter().map(|j| j.target.len()).max().unwrap_or(0);
        if qmax == 0 || tmax == 0 {
            continue;
        }

        let mut seq_q = vec![PAD; qmax * LANES16];
        let mut seq_t = vec![PAD; tmax * LANES16];
        let (mut qlen, mut tlen, mut minsc, mut endsc) = (
            [0usize; LANES16],
            [0usize; LANES16],
            [i32::MAX; LANES16],
            [i32::MAX; LANES16],
        );
        for (l, j) in group.iter().enumerate() {
            qlen[l] = j.query.len();
            tlen[l] = j.target.len();
            minsc[l] = j.minsc;
            endsc[l] = j.endsc;
            for (c, &b) in j.query.iter().enumerate() {
                seq_q[c * LANES16 + l] = b;
            }
            for (r, &b) in j.target.iter().enumerate() {
                seq_t[r * LANES16 + l] = b;
            }
        }

        let mut h_prev = vec![0u8; qmax * LANES16];
        let mut h_cur = vec![0u8; qmax * LANES16];
        let mut e = vec![0u8; qmax * LANES16];
        let mut hmax_col = vec![0i32; qmax * LANES16];
        let mut rowmax = vec![0i32; tmax * LANES16];
        let mut gmax = [0i32; LANES16];
        let mut te = [-1i32; LANES16];
        let mut limit = [-1i32; LANES16];
        let mut frozen = [false; LANES16];
        for l in 0..n {
            limit[l] = tlen[l] as i32 - 1;
        }

        for i in 0..tmax {
            let t_v = vld1q_u8(seq_t.as_ptr().add(i * LANES16));
            let mut f_v = zero;
            let mut h_diag_v = zero;
            let mut imax_v = zero;
            for j in 0..qmax {
                let q_v = vld1q_u8(seq_q.as_ptr().add(j * LANES16));
                // dsc = max(0, h_diag + score): saturating add/sub floor at 0, so no explicit max-0.
                let eq = vceqq_u8(t_v, q_v);
                let n_mask = vorrq_u8(vceqq_u8(t_v, four_v), vceqq_u8(q_v, four_v));
                let pad_mask = vorrq_u8(vcgeq_u8(t_v, m_v), vcgeq_u8(q_v, m_v));
                let add_match = vqaddq_u8(h_diag_v, mtch_v);
                let sub_mis = vqsubq_u8(h_diag_v, mispen_v);
                let sub_n = vqsubq_u8(h_diag_v, one_v);
                let mut dsc = vbslq_u8(eq, add_match, sub_mis);
                dsc = vbslq_u8(n_mask, sub_n, dsc);
                dsc = vbslq_u8(pad_mask, zero, dsc);

                let e_v = vld1q_u8(e.as_ptr().add(j * LANES16));
                let mut h_v = vmaxq_u8(dsc, e_v);
                h_v = vmaxq_u8(h_v, f_v);
                imax_v = vmaxq_u8(imax_v, h_v);
                vst1q_u8(h_cur.as_mut_ptr().add(j * LANES16), h_v);

                // e = max(max(0,e-e_del), max(0,h-oe_del)) = max(0, e-e_del, h-oe_del).
                let en = vmaxq_u8(vqsubq_u8(e_v, e_del_v), vqsubq_u8(h_v, oe_del_v));
                vst1q_u8(e.as_mut_ptr().add(j * LANES16), en);
                f_v = vmaxq_u8(vqsubq_u8(f_v, e_ins_v), vqsubq_u8(h_v, oe_ins_v));
                h_diag_v = vld1q_u8(h_prev.as_ptr().add(j * LANES16));
            }

            let mut imax_arr = [0u8; LANES16];
            vst1q_u8(imax_arr.as_mut_ptr(), imax_v);
            for l in 0..n {
                if i >= tlen[l] || frozen[l] {
                    continue;
                }
                let im = imax_arr[l] as i32;
                rowmax[i * LANES16 + l] = im;
                if im > gmax[l] {
                    gmax[l] = im;
                    te[l] = i as i32;
                    for j in 0..qmax {
                        hmax_col[j * LANES16 + l] = h_cur[j * LANES16 + l] as i32;
                    }
                    if gmax[l] >= endsc[l] {
                        frozen[l] = true;
                        limit[l] = i as i32;
                    }
                }
            }
            std::mem::swap(&mut h_prev, &mut h_cur);

            if (0..n).all(|l| frozen[l] || i + 1 >= tlen[l]) {
                break;
            }
        }

        extract_group(
            n, g, LANES16, &qlen, &minsc, max_sc, &gmax, &te, &limit, &rowmax, &hmax_col, &mut out,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

    /// bwa 5x5 score matrix: match `a`, mismatch `-b`, N row/col `-1`.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
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

    /// Random mate-rescue-shaped jobs (short query, longer window, some shared substring so the SW
    /// finds a real local alignment), then assert the batched kernel matches per-job `ksw_align2` on
    /// every field. This is the byte-identity gate for the NEON kernels.
    #[test]
    fn matesw_equals_scalar() {
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // Build a pool of jobs and their owned buffers.
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..2000 {
            let qlen = 5 + (next() % 146) as usize; // 5..=150 (varied lens exercise padding)
            let tlen = qlen + (next() % 500) as usize; // window >= query
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            // Embed one or two mutated copies of the query into the target so local alignments (and a
            // 2nd-best, for score2) exist.
            let copies = 1 + (next() % 2);
            if next() % 5 != 0 {
                for _ in 0..copies {
                    if tlen > qlen {
                        let at = (next() as usize) % (tlen - qlen + 1);
                        for k in 0..qlen {
                            t[at + k] = q[k];
                        }
                    }
                }
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            // Inject N bases (code 4) sometimes, in query and/or target.
            if next() % 4 == 0 {
                q[(next() as usize) % qlen] = 4;
            }
            if next() % 4 == 0 {
                t[(next() as usize) % tlen] = 4;
            }
            qbufs.push(q);
            tbufs.push(t);
        }
        let jobs: Vec<KswJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| KswJob {
                query: q.as_slice(),
                target: t.as_slice(),
            })
            .collect();

        // (match, mismatch, minsc): match=1 -> scores fit u8 (16-lane kernel); match=10 -> scores
        // exceed 250 (8-lane i16 kernel). Cover both, both against per-job ksw_align2.
        for &(a, b, minsc) in &[(1i8, 4i8, 19i32), (10, 40, 190)] {
            let mat = scmat(a, b);
            let max_sc = a as i32;
            let batched =
                batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc);
            for (i, j) in jobs.iter().enumerate() {
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc,
                );
                assert_eq!(batched[i], want, "job {i} (qlen {}, match {a})", j.query.len());
            }
        }
    }
}

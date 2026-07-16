//! Scalar banded Smith-Waterman seed extension, a faithful port of bwa's `ksw_extend2`
//! (`reference/bwa-mem2/src/ksw.cpp`): local extension from an initial score `h0` with affine
//! gaps, a band `w`, and z-drop early termination. This is the bit-identity source of truth for
//! seed extension; SIMD/GPU backends must reproduce its integer results.

/// Result of a seed extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtendResult {
    /// Best local score (the return value of `ksw_extend2`).
    pub score: i32,
    /// Query length of the best local alignment (`max_j + 1`).
    pub qle: i32,
    /// Target length of the best local alignment (`max_i + 1`).
    pub tle: i32,
    /// Target length when the alignment reaches the query end (`max_ie + 1`).
    pub gtle: i32,
    /// Score when the alignment reaches the query end.
    pub gscore: i32,
    /// Maximum band offset observed.
    pub max_off: i32,
}

/// Extend `query` against `target` from initial score `h0`. `m` is the alphabet size and `mat` the
/// `m*m` scoring matrix (row-major, `mat[a*m + b]`). Faithful port of `ksw_extend2`.
#[allow(clippy::too_many_arguments)]
pub fn ksw_extend2(
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
    let qlen = query.len();
    let tlen = target.len();
    debug_assert!(h0 > 0);
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    // Query profile: qp[c*qlen + j] = score of target base c against query base j.
    let mut qp = vec![0i8; qlen * m];
    let mut idx = 0;
    for k in 0..m {
        let row = &mat[k * m..k * m + m];
        for &qb in query {
            qp[idx] = row[qb as usize];
            idx += 1;
        }
    }

    // Score arrays: eh_h = H, eh_e = E.
    let mut eh_h = vec![0i32; qlen + 1];
    let mut eh_e = vec![0i32; qlen + 1];
    eh_h[0] = h0;
    eh_h[1] = if h0 > oe_ins { h0 - oe_ins } else { 0 };
    {
        let mut j = 2;
        while j <= qlen && eh_h[j - 1] > e_ins {
            eh_h[j] = eh_h[j - 1] - e_ins;
            j += 1;
        }
    }

    // Adjust the band by the maximum feasible insertion/deletion.
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
    let mut w = w;
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    let max_ins = max_ins.max(1);
    w = w.min(max_ins);
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    let max_del = max_del.max(1);
    w = w.min(max_del);

    let mut max = h0;
    let mut max_i = -1i32;
    let mut max_j = -1i32;
    let mut max_ie = -1i32;
    let mut gscore = -1i32;
    let mut max_off = 0i32;
    let mut beg = 0i32;
    let mut end = qlen as i32;

    for i in 0..tlen as i32 {
        let mut f = 0i32;
        let mut row_max = 0i32;
        let mut mj = -1i32;
        let tc = target[i as usize] as usize;
        let q = &qp[tc * qlen..tc * qlen + qlen];

        if beg < i - w {
            beg = i - w;
        }
        if end > i + w + 1 {
            end = i + w + 1;
        }
        if end > qlen as i32 {
            end = qlen as i32;
        }
        let mut h1 = if beg == 0 {
            let v = h0 - (o_del + e_del * (i + 1));
            v.max(0)
        } else {
            0
        };

        let mut j = beg;
        while j < end {
            let ju = j as usize;
            let big_m = eh_h[ju]; // H(i-1, j-1)
            let mut e = eh_e[ju]; // E(i-1, j)
            eh_h[ju] = h1; // H(i, j-1) for next row
            let big_m = if big_m != 0 {
                big_m + i32::from(q[ju])
            } else {
                0
            };
            let mut h = if big_m > e { big_m } else { e };
            h = if h > f { h } else { f };
            h1 = h;
            mj = if row_max > h { mj } else { j };
            row_max = if row_max > h { row_max } else { h };
            // Gaps open from M (the diagonal score), not from H = max(M, E, F). Both `ksw_extend2`
            // and the vectorized `bandedSWA` do this: MAIN_CODE16 subtracts oe_ins/oe_del from
            // `m11`, not `h11`. Using H instead only agrees while H == M, which holds on ordinary
            // extensions but not inside satellite repeats, where it silently turns a local extension
            // into a to-end one (wrong gscore => wrong qe/re, and a lost supplementary record).
            let mut t = big_m - oe_del;
            t = t.max(0);
            e -= e_del;
            e = if e > t { e } else { t };
            eh_e[ju] = e;
            let mut t = big_m - oe_ins;
            t = t.max(0);
            f -= e_ins;
            f = if f > t { f } else { t };
            j += 1;
        }
        eh_h[end as usize] = h1;
        eh_e[end as usize] = 0;
        if j == qlen as i32 && gscore <= h1 {
            max_ie = i;
            gscore = h1;
        }
        if row_max == 0 {
            break;
        }
        if row_max > max {
            max = row_max;
            max_i = i;
            max_j = mj;
            let off = (mj - i).abs();
            if off > max_off {
                max_off = off;
            }
        } else if zdrop > 0 {
            if i - max_i > mj - max_j {
                if max - row_max - ((i - max_i) - (mj - max_j)) * e_del > zdrop {
                    break;
                }
            } else if max - row_max - ((mj - max_j) - (i - max_i)) * e_ins > zdrop {
                break;
            }
        }

        // Shrink the band around the still-live cells.
        let mut jb = beg;
        while jb < end && eh_h[jb as usize] == 0 && eh_e[jb as usize] == 0 {
            jb += 1;
        }
        beg = jb;
        let mut je = end;
        while je >= beg && eh_h[je as usize] == 0 && eh_e[je as usize] == 0 {
            je -= 1;
        }
        end = if je + 2 < qlen as i32 {
            je + 2
        } else {
            qlen as i32
        };
    }

    ExtendResult {
        score: max,
        qle: max_j + 1,
        tle: max_i + 1,
        gtle: max_ie + 1,
        gscore,
        max_off,
    }
}

const MINUS_INF: i32 = -0x4000_0000;

/// Append `(op, len)` to a CIGAR (op-merged), mirroring bwa's `push_cigar`. Ops: 0=M, 1=I, 2=D.
fn push_cigar(cigar: &mut Vec<u32>, op: u32, len: u32) {
    if let Some(last) = cigar.last_mut() {
        if (*last & 0xf) == op {
            *last += len << 4;
            return;
        }
    }
    cigar.push((len << 4) | op);
}

/// Banded global alignment with traceback, a faithful port of `ksw_global2`. Returns the global
/// score and the CIGAR (`len<<4 | op`, op 0=M/1=I/2=D).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn ksw_global2(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
) -> (i32, Vec<u32>) {
    let qlen = query.len();
    let tlen = target.len();
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    let w = w.max(0) as usize;
    let n_col = qlen.min(2 * w + 1);

    let mut z = vec![0u8; n_col * tlen];
    let mut qp = vec![0i8; qlen * m];
    let mut idx = 0;
    for k in 0..m {
        let row = &mat[k * m..k * m + m];
        for &qb in query {
            qp[idx] = row[qb as usize];
            idx += 1;
        }
    }

    let mut eh_h = vec![MINUS_INF; qlen + 1];
    let mut eh_e = vec![MINUS_INF; qlen + 1];
    eh_h[0] = 0;
    for j in 1..=qlen.min(w) {
        eh_h[j] = -(o_ins + e_ins * j as i32);
    }

    for i in 0..tlen {
        let mut f = MINUS_INF;
        let beg = i.saturating_sub(w);
        let end = (i + w + 1).min(qlen);
        let mut h1 = if beg == 0 {
            -(o_del + e_del * (i as i32 + 1))
        } else {
            MINUS_INF
        };
        let tc = target[i] as usize;
        let q = &qp[tc * qlen..tc * qlen + qlen];
        let zoff = i * n_col;
        for j in beg..end {
            let mut mm = eh_h[j];
            let mut e = eh_e[j];
            eh_h[j] = h1;
            mm += i32::from(q[j]);
            let mut d: u8 = u8::from(mm < e);
            let mut h = if mm >= e { mm } else { e };
            d = if h >= f { d } else { 2 };
            h = if h >= f { h } else { f };
            h1 = h;
            let t = mm - oe_del;
            e -= e_del;
            d |= if e > t { 1 << 2 } else { 0 };
            e = if e > t { e } else { t };
            eh_e[j] = e;
            let t = mm - oe_ins;
            f -= e_ins;
            d |= if f > t { 2 << 4 } else { 0 };
            f = if f > t { f } else { t };
            z[zoff + (j - beg)] = d;
        }
        eh_h[end] = h1;
        eh_e[end] = MINUS_INF;
    }
    let score = eh_h[qlen];

    // Traceback from the last cell.
    let mut cigar: Vec<u32> = Vec::new();
    let mut i = tlen as i64 - 1;
    let mut k = (tlen as i64 - 1 + w as i64 + 1).min(qlen as i64) - 1;
    let mut which = 0u8;
    while i >= 0 && k >= 0 {
        let beg = (i as usize).saturating_sub(w);
        let d = z[i as usize * n_col + (k as usize - beg)];
        which = (d >> (which << 1)) & 3;
        if which == 0 {
            push_cigar(&mut cigar, 0, 1);
            i -= 1;
            k -= 1;
        } else if which == 1 {
            push_cigar(&mut cigar, 2, 1);
            i -= 1;
        } else {
            push_cigar(&mut cigar, 1, 1);
            k -= 1;
        }
    }
    if i >= 0 {
        push_cigar(&mut cigar, 2, (i + 1) as u32);
    }
    if k >= 0 {
        push_cigar(&mut cigar, 1, (k + 1) as u32);
    }
    cigar.reverse();
    (score, cigar)
}

/// Result of `ksw_align2`: a local (Smith-Waterman) alignment with start/end coordinates and the
/// 2nd-best score. Mirrors bwa's `kswr_t` (the fields mem_matesw uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KswAlignResult {
    pub score: i32,
    /// 0-based query end / start of the best alignment (`qb <= qe`).
    pub qb: i32,
    pub qe: i32,
    /// 0-based target end / start of the best alignment (`tb <= te`).
    pub tb: i32,
    pub te: i32,
    /// 2nd-best score on a target-end column far from `te`, and its column (`-1`/`0` if none).
    pub score2: i32,
    pub te2: i32,
}

/// One forward local-SW pass. Returns `(score, te, qe, score2, te2)`. `minsc` gates the
/// suboptimal (`b`-array) tracking (`KSW_XSUBO`); `endsc` stops early once a column max reaches it
/// (`KSW_XSTOP`, use `i32::MAX` to disable). `max_sc` is the maximum single-cell match score.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
/// Forward local-SW pass returning `(score, te, qe, score2, te2)` (no start coords). This is the
/// per-lane semantics a batched/vectorized mate-rescue kernel must reproduce: [`ksw_align2`] is just
/// this forward pass plus a second, reversed forward pass to recover `qb`/`tb` (`KSW_XSTART`).
///
/// `lanes` is ksw's SIMD width (16 for the u8 kernel, 8 for i16), and it is **not** a performance
/// knob: `ksw_qinit` rounds the query profile up to `slen * lanes` columns and fills the tail with
/// score 0 (`(k >= qlen? 0 : ma[query[k]]) + shift`). A zero-score column leaves `h = h_diag`, so
/// the padding carries a diagonal forward and its cells land in ksw's per-row `max` -- which feeds
/// the `b` array and hence `score2`. Dropping the padding makes row maxima decay where bwa's plateau,
/// so `score2` (and the `csub` mate rescue derives from it) comes out too low.
#[allow(clippy::too_many_arguments)]
pub fn ksw_local_fwd(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    endsc: i32,
    max_sc: i32,
    lanes: usize,
) -> (i32, i32, i32, i32, i32) {
    let qlen_real = query.len();
    // `slen * lanes` columns, the tail scoring 0 (see above).
    let slen = qlen_real.div_ceil(lanes);
    let qlen = slen * lanes;
    let tlen = target.len();
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    let mut h_prev = vec![0i32; qlen]; // H(i-1, .)
    let mut h_cur = vec![0i32; qlen]; // H(i, .)
    let mut e = vec![0i32; qlen]; // E(i, j), persists across target rows
    let mut hmax_col = vec![0i32; qlen]; // H column at the best target end `te`
    let mut gmax = 0i32;
    let mut te = -1i32;
    // Suboptimal tracker `b`: (column max score, column), consecutive columns merged (keep higher).
    let mut b: Vec<(i32, i32)> = Vec::new();

    for i in 0..tlen {
        let s_row = target[i] as usize * m;
        let mut f = 0i32;
        let mut h_diag = 0i32; // H(i-1, -1) = 0
        let mut imax = 0i32;
        for j in 0..qlen {
            let sc = if j < qlen_real {
                i32::from(mat[s_row + query[j] as usize])
            } else {
                0 // padding column: h = h_diag, carrying the diagonal forward
            };
            // H(i,j) = max{0, H(i-1,j-1)+s, E(i,j), F(i,j)}  (E,F are already >= 0)
            let mut h = h_diag + sc;
            if h < 0 {
                h = 0;
            }
            if e[j] > h {
                h = e[j];
            }
            if f > h {
                h = f;
            }
            if h > imax {
                imax = h;
            }
            h_cur[j] = h;
            // E(i+1,j) = max{0, E(i,j)-e_del, H(i,j)-o_del-e_del}
            let mut en = e[j] - e_del;
            let td = h - oe_del;
            if td > en {
                en = td;
            }
            e[j] = en.max(0);
            // F(i,j+1) = max{0, F(i,j)-e_ins, H(i,j)-o_ins-e_ins}
            let mut fn_ = f - e_ins;
            let ti = h - oe_ins;
            if ti > fn_ {
                fn_ = ti;
            }
            f = fn_.max(0);
            h_diag = h_prev[j];
        }
        if imax >= minsc {
            match b.last() {
                Some(&(_, col)) if col + 1 == i as i32 => {
                    if b.last().unwrap().0 < imax {
                        *b.last_mut().unwrap() = (imax, i as i32);
                    }
                }
                _ => b.push((imax, i as i32)),
            }
        }
        if imax > gmax {
            gmax = imax;
            te = i as i32;
            hmax_col.copy_from_slice(&h_cur);
            if gmax >= endsc {
                break;
            }
        }
        std::mem::swap(&mut h_prev, &mut h_cur);
    }

    // Query end: smallest query column reaching the max at `te` (ksw scans Hmax in *striped byte*
    // order, mapping byte i to column `i/lanes + i%lanes*slen`, but only the min-on-tie survives, so
    // the order does not matter -- the padded columns being in range does).
    let mut qe = -1i32;
    if te >= 0 {
        let mut mx = -1i32;
        for i in 0..qlen {
            let col = i / lanes + (i % lanes) * slen;
            let v = hmax_col[col];
            if v > mx {
                mx = v;
                qe = col as i32;
            } else if v == mx && (col as i32) < qe {
                qe = col as i32;
            }
        }
    }

    // 2nd-best score: best `b` entry whose column lies outside [te - w, te + w], w = ceil(score/max).
    // Starts at -1, not 0: ksw returns `g_defr = {0, -1, -1, -1, -1, -1, -1}` when nothing qualifies,
    // and mem_matesw copies it straight into `csub`. The sign matters downstream -- mem_sam_pe caps
    // MAPQ with `raw_mapq(score - csub, a)`, so csub = -1 yields score+1 where 0 yields score.
    let mut score2 = -1i32;
    let mut te2 = -1i32;
    if gmax > 0 && !b.is_empty() {
        let w = (gmax + max_sc - 1) / max_sc;
        let (low, high) = (te - w, te + w);
        for &(bs, bc) in &b {
            if (bc < low || bc > high) && bs > score2 {
                score2 = bs;
                te2 = bc;
            }
        }
    }
    (gmax, te, qe, score2, te2)
}

/// Local Smith-Waterman with affine gaps, returning best-alignment coords and the 2nd-best score.
/// Faithful scalar port of `ksw_align2` (with `KSW_XSTART | KSW_XSUBO`). `max_sc` is the maximum
/// single match score (`opt.a`), used for the suboptimal-window width. `lanes` selects ksw's kernel
/// width (16 = u8 / `KSW_XBYTE`, 8 = i16); both passes use the same one, and it changes the result
/// (see [`ksw_local_fwd`]).
#[allow(clippy::too_many_arguments)]
pub fn ksw_align2(
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
    lanes: usize,
) -> KswAlignResult {
    let (score, te, qe, score2, te2) = ksw_local_fwd(
        query,
        target,
        m,
        mat,
        o_del,
        e_del,
        o_ins,
        e_ins,
        minsc,
        i32::MAX,
        max_sc,
        lanes,
    );
    let mut r = KswAlignResult {
        score,
        qb: -1,
        qe,
        tb: -1,
        te,
        score2,
        te2,
    };
    // KSW_XSTART: recover the start by aligning the reversed prefixes and stopping at `score`.
    if score < minsc || qe < 0 {
        return r;
    }
    // bwa does `revseq(r.qe + 1, query); revseq(r.te + 1, target);` -- both *in place* -- then
    // re-inits the query profile at length `qe + 1` but calls the kernel with the **full** `tlen`.
    // So the query really is truncated, while the target is not: only its first `te + 1` bases are
    // reversed and the untouched tail still gets scanned. That matters, because the pass stops via
    // KSW_XSTOP once it reaches `score`, and if the reversed prefix alone never gets there the tail
    // can still reach it -- which sets qb/tb, and mem_matesw drops the rescue when qb < 0.
    let qrev: Vec<u8> = query[..=qe as usize].iter().rev().copied().collect();
    let mut trev: Vec<u8> = target.to_vec();
    trev[..=te as usize].reverse();
    let (rscore, rte, rqe, _, _) = ksw_local_fwd(
        &qrev,
        &trev,
        m,
        mat,
        o_del,
        e_del,
        o_ins,
        e_ins,
        i32::MAX,
        score,
        max_sc,
        lanes,
    );
    if score == rscore {
        r.tb = te - rte;
        r.qb = qe - rqe;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    // 5x5 matrix like bwa: match a, mismatch -b, N row/col -1.
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

    #[test]
    fn ksw_align2_perfect_match() {
        // query == target: full-length local alignment, score = len*a, spanning both from 0.
        let mat = scmat(1, 4);
        let q = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let r = ksw_align2(&q, &q, 5, &mat, 6, 1, 6, 1, 4, 1, 16);
        assert_eq!(r.score, 8);
        assert_eq!((r.qb, r.qe), (0, 7));
        assert_eq!((r.tb, r.te), (0, 7));
    }

    #[test]
    fn ksw_align2_local_trims_flanks() {
        // A perfect core flanked by mismatches: local alignment picks the core only.
        // query core AACC at query[2..6]; target has the core at target[1..5].
        let mat = scmat(1, 4);
        //            0  1  2  3  4  5  6  7
        let q = [3u8, 3, 0, 0, 1, 1, 3, 3];
        let t = [2u8, 0, 0, 1, 1, 2];
        let r = ksw_align2(&q, &t, 5, &mat, 6, 1, 6, 1, 1, 1, 16);
        assert_eq!(r.score, 4); // 4 matching bases
        assert_eq!((r.qb, r.qe), (2, 5));
        assert_eq!((r.tb, r.te), (1, 4));
    }

    /// Unbanded reference of the same recurrence (no band, no zdrop, no zero-row break), for
    /// validating the core DP where those heuristics don't fire.
    #[allow(clippy::too_many_arguments)]
    fn ref_extend(
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        h0: i32,
    ) -> i32 {
        let qlen = query.len();
        let tlen = target.len();
        let oe_del = o_del + e_del;
        let oe_ins = o_ins + e_ins;
        let mut eh_h = vec![0i32; qlen + 1];
        let mut eh_e = vec![0i32; qlen + 1];
        eh_h[0] = h0;
        eh_h[1] = if h0 > oe_ins { h0 - oe_ins } else { 0 };
        let mut j = 2;
        while j <= qlen && eh_h[j - 1] > e_ins {
            eh_h[j] = eh_h[j - 1] - e_ins;
            j += 1;
        }
        let mut max = h0;
        for i in 0..tlen {
            let mut f = 0i32;
            let mut h1 = (h0 - (o_del + e_del * (i as i32 + 1))).max(0);
            for j in 0..qlen {
                let sc = i32::from(mat[target[i] as usize * m + query[j] as usize]);
                let big_m = eh_h[j];
                let mut e = eh_e[j];
                eh_h[j] = h1;
                let big_m = if big_m != 0 { big_m + sc } else { 0 };
                let mut h = big_m.max(e).max(f);
                h1 = h;
                if h > max {
                    max = h;
                }
                let t = (big_m - oe_del).max(0);
                e = (e - e_del).max(t);
                eh_e[j] = e;
                let t = (big_m - oe_ins).max(0);
                f = (f - e_ins).max(t);
                let _ = &mut h;
            }
            eh_h[qlen] = h1;
            eh_e[qlen] = 0;
        }
        max
    }

    fn call(query: &[u8], target: &[u8], mat: &[i8], h0: i32) -> ExtendResult {
        ksw_extend2(query, target, 5, mat, 6, 1, 6, 1, 100, 0, 100, h0)
    }

    #[test]
    fn exact_match_scores_full_length() {
        let mat = scmat(1, 4);
        let s: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let r = call(&s, &s, &mat, 1);
        // h0 + qlen matches of score 1.
        assert_eq!(r.score, 1 + s.len() as i32);
        assert_eq!(r.qle, s.len() as i32);
        assert_eq!(r.tle, s.len() as i32);
        assert_eq!(r.gscore, 1 + s.len() as i32);
    }

    #[test]
    fn matches_unbanded_reference() {
        let mat = scmat(1, 4);
        let mut state: u64 = 0xa5a5_1234_9999_0001;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..500 {
            // Build a target that shares a positive-scoring prefix with the query so the local
            // extension never hits a zero row (keeping band/zdrop inactive for the comparison).
            let len = 20 + (next() % 40) as usize;
            let base: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            let mut target = base.clone();
            // introduce a couple of mismatches late, staying positive
            if len > 25 {
                let p = len - 3 - (next() % 3) as usize;
                target[p] = (target[p] + 1) % 4;
            }
            let got = ksw_extend2(&base, &target, 5, &mat, 6, 1, 6, 1, 1000, 0, 1_000_000, 1);
            let want = ref_extend(&base, &target, 5, &mat, 6, 1, 6, 1, 1);
            assert_eq!(got.score, want, "len={len}");
        }
    }

    #[test]
    fn global_exact_match_is_all_m() {
        let mat = scmat(1, 4);
        let s: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let (score, cigar) = ksw_global2(&s, &s, 5, &mat, 6, 1, 6, 1, 100);
        assert_eq!(score, s.len() as i32);
        assert_eq!(cigar, vec![(s.len() as u32) << 4]); // "<len>M"
    }

    #[test]
    fn global_single_deletion() {
        let mat = scmat(1, 4);
        // target has one extra base vs query -> a 1bp deletion (D) in the CIGAR.
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let mut target = query.clone();
        target.insert(4, 2); // extra base in the middle of target
        let (_score, cigar) = ksw_global2(&query, &target, 5, &mat, 6, 1, 6, 1, 100);
        // total reference length consumed == target length; exactly one D of length 1.
        let dsum: u32 = cigar.iter().filter(|c| *c & 0xf == 2).map(|c| c >> 4).sum();
        assert_eq!(dsum, 1);
        let msum: u32 = cigar.iter().filter(|c| *c & 0xf == 0).map(|c| c >> 4).sum();
        assert_eq!(msum, query.len() as u32);
    }

    #[test]
    fn zdrop_stops_runaway_extension() {
        let mat = scmat(1, 4);
        // A short match then a long mismatched tail: zdrop caps the target length used.
        let mut query = vec![0u8; 10];
        query.extend(vec![1u8; 40]);
        let mut target = vec![0u8; 10];
        target.extend(vec![2u8; 40]); // tail all mismatched vs query tail
        let r = ksw_extend2(&query, &target, 5, &mat, 6, 1, 6, 1, 100, 0, 100, 1);
        assert_eq!(r.score, 1 + 10); // only the 10 matching bases contribute
        assert_eq!(r.tle, 10);
    }
}

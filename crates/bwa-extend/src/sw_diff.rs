//! LEVER 4 spike — difference-recurrence (Suzuki-Kasahara / KSW2) feasibility probe.
//!
//! The difference recurrence expresses the DP in terms of adjacent-cell *differences* (u/v/x/y),
//! which is what lets KSW2 keep values in int8 and shorten the per-cell critical path. Its algebra
//! requires a **non-local** (extension) recurrence: `M(i,j) = H(i-1,j-1) + s`. bwa-mem2's
//! `ksw_extend2` is **local** — `M(i,j) = (H(i-1,j-1) == 0) ? 0 : H(i-1,j-1) + s` (a Smith-Waterman
//! restart) — and that `== 0` test is an *absolute* condition that pure differences cannot represent.
//!
//! This module implements the extension recurrence the difference form computes (no restart), so the
//! `scalar == diff` gate can measure *exactly where and how often* it diverges from `ksw_extend2`.

use crate::sw::ExtendResult;

/// Affine banded extension DP **without** the local `M==0` restart — i.e. the recurrence a
/// difference-recurrence (KSW2) kernel computes. Same band / gap / end-bonus / z-drop handling and
/// the same output extraction as `ksw_extend2`, so any divergence is attributable solely to the
/// missing local restart (the feature the difference algebra cannot express).
#[allow(clippy::too_many_arguments)]
pub fn extend_no_restart(
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
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    let mut qp = vec![0i8; qlen * m];
    let mut idx = 0;
    for k in 0..m {
        let row = &mat[k * m..k * m + m];
        for &qb in query {
            qp[idx] = row[qb as usize];
            idx += 1;
        }
    }

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

    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
    let mut w = w;
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    w = w.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    w = w.min(max_del.max(1));

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
            (h0 - (o_del + e_del * (i + 1))).max(0)
        } else {
            0
        };

        let mut j = beg;
        while j < end {
            let ju = j as usize;
            let big_m = eh_h[ju];
            let mut e = eh_e[ju];
            eh_h[ju] = h1;
            // *** The only difference from ksw_extend2: no `big_m != 0` local restart. ***
            let big_m = big_m + i32::from(q[ju]);
            let mut h = if big_m > e { big_m } else { e };
            h = if h > f { h } else { f };
            h1 = h;
            mj = if row_max > h { mj } else { j };
            row_max = if row_max > h { row_max } else { h };
            let mut t = h - oe_del;
            t = t.max(0);
            e -= e_del;
            e = if e > t { e } else { t };
            eh_e[ju] = e;
            let mut t = h - oe_ins;
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

#[cfg(test)]
mod tests {
    use super::extend_no_restart;
    use crate::sw::ksw_extend2;

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

    /// Quantify how often the difference-recurrence (non-local) DP diverges from bwa's local
    /// `ksw_extend2` over the acceptance gate's parameter space. Prints the divergence rate; this is
    /// the LEVER 4 STOP evidence (byte-identity is impossible when the local restart triggers).
    #[test]
    fn diff_recurrence_divergence_rate() {
        let mat = scoring();
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let mut diverged = 0u32;
        let n = 20000u32;
        for _ in 0..n {
            let qlen = 1 + (next() % 60) as usize;
            let tlen = 1 + (next() % 60) as usize;
            let query: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            let target: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let w = 1 + (next() % 30) as i32;
            let zdrop = (next() % 120) as i32;
            let end_bonus = (next() % 10) as i32;
            let h0 = 1 + (next() % 20) as i32;
            let expected = ksw_extend2(
                &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
            );
            let got = extend_no_restart(
                &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
            );
            if got != expected {
                diverged += 1;
            }
        }
        let pct = 100.0 * f64::from(diverged) / f64::from(n);
        println!(
            "LEVER4 diff-recurrence (non-local) vs ksw_extend2: diverged {diverged}/{n} ({pct:.1}%)"
        );
        // Assert it DOES diverge — that is the finding (local restart is inexpressible in differences).
        assert!(diverged > 0, "expected divergence from the local restart");
    }
}

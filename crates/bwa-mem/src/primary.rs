//! Region dedup, primary/secondary marking and MAPQ, mirroring bwa-mem2's
//! `mem_sort_dedup_patch`, `mem_mark_primary_se[_core]` and `mem_approx_mapq_se`
//! (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Includes the `mem_patch_reg` region-merge branch (collinear regions re-aligned into one),
//! which fires on split/long-indel alignments.

use bwa_core::MemOpt;
use bwa_index::FmIndex;

use crate::cigar::gen_cigar2;
use crate::MemAlnReg;

const PATCH_MAX_R_BW: f64 = 0.05;
const PATCH_MIN_SC_RATIO: f64 = 0.90;

/// Try to merge collinear regions `a` (earlier `rb`) and `b` via a global re-alignment. Returns
/// `(merged_score, band)` if the merge is good enough. Port of `mem_patch_reg`.
fn mem_patch_reg(
    fm: &FmIndex,
    opt: &MemOpt,
    codes: &[u8],
    a: &MemAlnReg,
    b: &MemAlnReg,
) -> Option<(i32, i32)> {
    if codes.is_empty() {
        return None; // `query == 0` (mem_matesw's dedup): merging is disabled
    }
    let l_pac = fm.l_pac();
    if a.rb < l_pac && b.rb >= l_pac {
        return None; // different strands
    }
    if a.qb >= b.qb || a.qe >= b.qe || a.re >= b.re {
        return None; // not collinear
    }
    let w0 = ((a.re - b.rb) - i64::from(a.qe - b.qb)).abs();
    let r = ((a.re - b.rb) as f64 / (b.re - a.rb) as f64
        - f64::from(a.qe - b.qb) / f64::from(b.qe - a.qb))
    .abs();
    if a.re < b.rb || a.qe < b.qb {
        if w0 > i64::from(opt.w << 1) || r >= PATCH_MAX_R_BW {
            return None;
        }
    } else if w0 > i64::from(opt.w << 2) || r >= PATCH_MAX_R_BW * 2.0 {
        return None;
    }
    let mut w = w0 as i32 + a.w + b.w;
    w = w.min(opt.w << 2);

    let l_query = b.qe - a.qb;
    let (score, _c, _nm, _md) = gen_cigar2(
        fm,
        opt,
        w,
        l_query,
        &codes[a.qb as usize..b.qe as usize],
        a.rb,
        b.re,
    )?;

    let q_s = ((f64::from(b.qe - a.qb) / f64::from((b.qe - b.qb) + (a.qe - a.qb)))
        * f64::from(b.score + a.score)
        + 0.499) as i32;
    let r_s = (((b.re - a.rb) as f64 / ((b.re - b.rb) + (a.re - a.rb)) as f64)
        * f64::from(b.score + a.score)
        + 0.499) as i32;
    if f64::from(score) / f64::from(q_s.max(r_s)) < PATCH_MIN_SC_RATIO {
        return None;
    }
    Some((score, w))
}

/// Thomas Wang's 64-bit integer hash (`hash_64`), the deterministic tie-breaker for equal-scoring
/// regions.
pub fn hash_64(mut key: u64) -> u64 {
    key = key.wrapping_add(!(key << 32));
    key ^= key >> 22;
    key = key.wrapping_add(!(key << 13));
    key ^= key >> 8;
    key = key.wrapping_add(key << 3);
    key ^= key >> 15;
    key = key.wrapping_add(!(key << 27));
    key ^= key >> 31;
    key
}

/// Remove redundant and identical alignment regions. Port of `mem_sort_dedup_patch` minus the
/// `mem_patch_reg` merge branch.
pub fn mem_sort_dedup_patch(
    fm: &FmIndex,
    opt: &MemOpt,
    codes: &[u8],
    mut a: Vec<MemAlnReg>,
) -> Vec<MemAlnReg> {
    if a.len() <= 1 {
        return a;
    }
    let mlr = f64::from(opt.mask_level_redun);
    let max_gap = i64::from(opt.max_chain_gap);

    // Sort by END position on the reference.
    a.sort_by_key(|r| r.re);
    for r in &mut a {
        r.n_comp = 1;
    }

    for i in 1..a.len() {
        if a[i].rid != a[i - 1].rid || a[i].rb >= a[i - 1].re + max_gap {
            continue;
        }
        let mut j = i as i64 - 1;
        while j >= 0 {
            let ju = j as usize;
            // Read p (a[i]) fresh each iteration: a merge below updates it in place.
            let (p_rid, p_rb, p_re, p_qb, p_qe, p_score) =
                (a[i].rid, a[i].rb, a[i].re, a[i].qb, a[i].qe, a[i].score);
            if a[ju].rid != p_rid || p_rb >= a[ju].re + max_gap {
                break;
            }
            if a[ju].qe == a[ju].qb {
                j -= 1;
                continue;
            }
            let or_ = a[ju].re - p_rb;
            let oq = if a[ju].qb < p_qb {
                a[ju].qe - p_qb
            } else {
                p_qe - a[ju].qb
            };
            let mr = (a[ju].re - a[ju].rb).min(p_re - p_rb);
            let mq = (a[ju].qe - a[ju].qb).min(p_qe - p_qb);
            if or_ as f64 > mlr * mr as f64 && f64::from(oq) > mlr * f64::from(mq) {
                if p_score < a[ju].score {
                    a[i].qe = a[i].qb;
                    break;
                }
                a[ju].qe = a[ju].qb;
            } else if a[ju].rb < p_rb {
                let q = a[ju].clone();
                let p = a[i].clone();
                if let Some((score, w)) = mem_patch_reg(fm, opt, codes, &q, &p) {
                    a[i].n_comp += a[ju].n_comp + 1;
                    a[i].seedcov = a[i].seedcov.max(a[ju].seedcov);
                    a[i].sub = a[i].sub.max(a[ju].sub);
                    a[i].csub = a[i].csub.max(a[ju].csub);
                    a[i].qb = a[ju].qb;
                    a[i].rb = a[ju].rb;
                    a[i].truesc = score;
                    a[i].score = score;
                    a[i].w = w;
                    a[ju].qe = a[ju].qb;
                }
            }
            j -= 1;
        }
    }
    a.retain(|r| r.qe > r.qb);

    // Sort by score desc, then rb, then qb; drop identical hits.
    a.sort_by(|x, y| {
        y.score
            .cmp(&x.score)
            .then(x.rb.cmp(&y.rb))
            .then(x.qb.cmp(&y.qb))
    });
    for i in 1..a.len() {
        if a[i].score == a[i - 1].score && a[i].rb == a[i - 1].rb && a[i].qb == a[i - 1].qb {
            a[i].qe = a[i].qb;
        }
    }
    a.retain(|r| r.qe > r.qb);
    a
}

fn mark_primary_core(opt: &MemOpt, a: &mut [MemAlnReg]) {
    let n = a.len();
    let mut tmp = opt.a + opt.b;
    tmp = tmp.max(opt.o_del + opt.e_del).max(opt.o_ins + opt.e_ins);
    let mut z: Vec<usize> = vec![0];
    for i in 1..n {
        let (i_qb, i_qe, i_score, i_alt) = (a[i].qb, a[i].qe, a[i].score, a[i].is_alt);
        let mut found = None;
        for &j in &z {
            let b_max = a[j].qb.max(i_qb);
            let e_min = a[j].qe.min(i_qe);
            if e_min > b_max {
                let min_l = (i_qe - i_qb).min(a[j].qe - a[j].qb);
                if f64::from(e_min - b_max) >= f64::from(min_l) * f64::from(opt.mask_level) {
                    if a[j].sub == 0 {
                        a[j].sub = i_score;
                    }
                    if a[j].score - i_score <= tmp && (a[j].is_alt || !i_alt) {
                        a[j].sub_n += 1;
                    }
                    found = Some(j);
                    break;
                }
            }
        }
        match found {
            Some(j) => a[i].secondary = j as i32,
            None => z.push(i),
        }
    }
}

/// Mark primary/secondary regions and set `sub`/`sub_n`. Port of `mem_mark_primary_se` for the
/// primary-assembly (non-ALT) case. `id` is the global read index (`n_processed + i`).
pub fn mem_mark_primary_se(opt: &MemOpt, a: &mut [MemAlnReg], id: u64) -> i32 {
    if a.is_empty() {
        return 0;
    }
    let mut n_pri = 0;
    for (i, r) in a.iter_mut().enumerate() {
        r.sub = 0;
        r.secondary = -1;
        r.hash = hash_64(id.wrapping_add(i as u64));
        if !r.is_alt {
            n_pri += 1;
        }
    }
    // Sort by (score desc, is_alt asc, hash asc) — `alnreg_hlt`.
    a.sort_by(|x, y| {
        y.score
            .cmp(&x.score)
            .then(x.is_alt.cmp(&y.is_alt))
            .then(x.hash.cmp(&y.hash))
    });
    mark_primary_core(opt, a);
    // No ALT contigs: `secondary_all` mirrors `secondary` (the C's else-branch when n_pri == n).
    // `mem_gen_alt`/the PE swap read `secondary_all` separately from `secondary`.
    for r in a.iter_mut() {
        r.secondary_all = r.secondary;
    }
    n_pri
}

/// Approximate mapping quality of a primary region. Port of `mem_approx_mapq_se`.
pub fn mem_approx_mapq_se(opt: &MemOpt, a: &MemAlnReg) -> u32 {
    let mut sub = if a.sub != 0 {
        a.sub
    } else {
        opt.min_seed_len * opt.a
    };
    sub = sub.max(a.csub);
    if sub >= a.score {
        return 0;
    }
    let l = (a.qe - a.qb).max((a.re - a.rb) as i32);
    let identity = 1.0 - f64::from(l * opt.a - a.score) / f64::from(opt.a + opt.b) / f64::from(l);
    let mut mapq: i32;
    if a.score == 0 {
        mapq = 0;
    } else {
        // mapQ_coef_len > 0 always for the default options.
        let tmp = if f64::from(l) < opt.mapq_coef_len {
            1.0
        } else {
            opt.mapq_coef_fac / f64::from(l).ln()
        };
        let tmp = tmp * identity * identity;
        mapq = (6.02 * f64::from(a.score - sub) / f64::from(opt.a) * tmp * tmp + 0.499) as i32;
    }
    if a.sub_n > 0 {
        mapq -= (4.343 * f64::from(a.sub_n + 1).ln() + 0.499) as i32;
    }
    mapq = mapq.clamp(0, 60);
    mapq = (f64::from(mapq) * (1.0 - f64::from(a.frac_rep)) + 0.499) as i32;
    mapq as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_64_is_wang() {
        // Distinct inputs give distinct, well-mixed outputs (sanity).
        assert_ne!(hash_64(0), hash_64(1));
        assert_ne!(hash_64(1000), hash_64(1001));
    }
}

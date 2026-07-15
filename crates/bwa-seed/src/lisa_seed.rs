//! LISA/BWA-MEME learned-index seeding: the same SMEM collection as [`crate`]'s FM-index path, but
//! every interval is obtained from a [`LearnedSa`] (plain `[fwd][rc]` suffix array + P-RMI) instead
//! of the FM-index `backward_ext`/`get_occ`.
//!
//! **Why this is byte-identical.** bwa-mem2's SMEM driver (`smems_from_pos` etc.) branches *only* on
//! the interval size `s` (against `min_intv`/`curr_s`) and on span lengths; the reverse-complement
//! start `l` is pure internal bookkeeping and the seed output reads only `k` and `s`. And
//! [`LearnedSa::bi_interval`] is proven (`bidirectional_interval_matches_fmindex`) to return the same
//! `(k, s)` as walking `backward_ext` over the same pattern. So we mirror the driver structure exactly
//! and replace each `backward_ext` with a span lookup `interval_of(codes[m..=n])`: the control flow —
//! and therefore the emitted SMEM set and the seeds derived from it — is identical to the FM path.
//! `l` is never needed and is left at 0.
//!
//! This is the correctness-first form (each interval is a from-scratch learned-index search). The
//! forward phase can later narrow incrementally ([`LearnedSa::narrow`]); the backward phase re-searches
//! (prepending does not nest in a suffix array). Byte-identity is validated against the FM path below.

use crate::MemSeed;
use bwa_core::MemOpt;
use bwa_index::lisa::LearnedSa;
use bwa_index::Smem;

/// The `(k, s)` of the exact match `codes[m..=n]` over the learned suffix array: `k` = forward SA
/// interval start, `s` = interval size. Same values as `FmIndex::backward_ext` walked over that
/// pattern (proven). `l` is not computed (the driver never reads it).
#[inline]
fn interval_of(lsa: &LearnedSa, codes: &[u8], m: usize, n: usize) -> (i64, i64) {
    let (lo, hi) = lsa.exact_interval(&codes[m..=n]);
    (lo as i64, (hi - lo) as i64)
}

/// One-position SMEM search starting at `x` (LISA analog of [`crate::smems_from_pos`], line-for-line
/// with `backward_ext` replaced by [`interval_of`]). Appends SMEMs to `out`, returns `next_x`.
fn smems_from_pos_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    x: usize,
    min_seed_len: i32,
    min_intv: i64,
    prev: &mut [Smem],
    out: &mut Vec<Smem>,
) -> usize {
    let readlength = codes.len();
    let n_sa = lsa.len();
    let mut next_x = x + 1;
    let a = codes[x];
    if a >= 4 {
        return next_x;
    }

    // Initial single-base interval, span [x, x]. Appending a base always nests within the current
    // interval, so the whole forward extension is a sequence of `narrow` calls (each two partition
    // points over the shrinking interval) instead of a from-scratch search per step — identical
    // result (`narrow` is proven to reproduce `exact_interval` for every prefix length).
    let (mut lo, mut hi) = lsa.narrow(0, n_sa, 0, a);
    let mut smem = Smem {
        rid: 0,
        m: x as u32,
        n: x as u32,
        k: lo as i64,
        l: 0,
        s: (hi - lo) as i64,
    };
    let mut num_prev = 0usize;

    // Forward extension: span [x, j], j increasing (append codes[j] at column j-x).
    let mut j = x + 1;
    while j < readlength {
        let aj = codes[j];
        next_x = j + 1;
        if aj >= 4 {
            break;
        }
        let (nlo, nhi) = lsa.narrow(lo, hi, j - x, aj);
        let new_smem = Smem {
            rid: 0,
            m: x as u32,
            n: j as u32,
            k: nlo as i64,
            l: 0,
            s: (nhi - nlo) as i64,
        };

        prev[num_prev] = smem;
        if new_smem.s != smem.s {
            num_prev += 1;
        }
        if new_smem.s < min_intv {
            next_x = j;
            break;
        }
        smem = new_smem;
        lo = nlo;
        hi = nhi;
        j += 1;
    }
    if smem.s >= min_intv {
        prev[num_prev] = smem;
        num_prev += 1;
    }

    prev[..num_prev].reverse();

    // Backward extension: span [jj, sm.n].
    let mut jj = x as i64 - 1;
    while jj >= 0 {
        let a = codes[jj as usize];
        if a > 3 {
            break;
        }
        let mut num_curr = 0usize;
        let mut curr_s = -1i64;

        let mut p = 0usize;
        while p < num_prev {
            let sm = prev[p];
            let (k, s) = interval_of(lsa, codes, jj as usize, sm.n as usize);
            let new_smem = Smem {
                rid: 0,
                m: jj as u32,
                n: sm.n,
                k,
                l: 0,
                s,
            };
            if new_smem.s < min_intv
                && (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len)
            {
                out.push(sm);
                break;
            }
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
                break;
            }
            p += 1;
        }
        p += 1;
        while p < num_prev {
            let sm = prev[p];
            let (k, s) = interval_of(lsa, codes, jj as usize, sm.n as usize);
            let new_smem = Smem {
                rid: 0,
                m: jj as u32,
                n: sm.n,
                k,
                l: 0,
                s,
            };
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
            }
            p += 1;
        }
        num_prev = num_curr;
        if num_curr == 0 {
            break;
        }
        jj -= 1;
    }
    if num_prev != 0 {
        let sm = prev[0];
        if (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len) {
            out.push(sm);
        }
    }
    next_x
}

/// Collect all round-1 SMEMs of `codes` via the learned index (LISA analog of [`crate::collect_smems`]).
pub fn collect_smems_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    min_seed_len: i32,
    min_intv: i64,
) -> Vec<Smem> {
    let mut out = Vec::new();
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    let mut x = 0usize;
    while x < codes.len() {
        x = smems_from_pos_lsa(lsa, codes, x, min_seed_len, min_intv, &mut scratch, &mut out);
    }
    out
}

/// Round-3 forward-only seeding (LISA analog of `bwt_seed_strategy`): emit a seed when the interval
/// first drops below `max_intv` and the seed is at least `min_seed_len` long.
fn bwt_seed_strategy_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    let readlength = codes.len();
    let n_sa = lsa.len();
    let mut x = 0usize;
    while x < readlength {
        let mut next_x = x + 1;
        if codes[x] < 4 {
            // Forward-only: fully incremental narrowing (append codes[j] at column j-x).
            let (mut lo, mut hi) = lsa.narrow(0, n_sa, 0, codes[x]);
            let mut j = x + 1;
            while j < readlength {
                next_x = j + 1;
                let aj = codes[j];
                if aj >= 4 {
                    break;
                }
                let (nlo, nhi) = lsa.narrow(lo, hi, j - x, aj);
                let s = (nhi - nlo) as i64;
                let smem = Smem {
                    rid: 0,
                    m: x as u32,
                    n: j as u32,
                    k: nlo as i64,
                    l: 0,
                    s,
                };
                if smem.s < max_intv
                    && (i64::from(smem.n) - i64::from(smem.m) + 1) >= i64::from(min_seed_len)
                {
                    if smem.s > 0 {
                        out.push(smem);
                    }
                    break;
                }
                lo = nlo;
                hi = nhi;
                j += 1;
            }
        }
        x = next_x;
    }
}

/// Round 2: re-seed each long, non-repetitive round-1 SMEM from its midpoint (LISA analog of
/// `smem_round_2`).
fn smem_round_2_lsa(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    let num1 = smems.len();
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    for idx in 0..num1 {
        let p = smems[idx];
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        let x = ((end + start) >> 1) as usize;
        smems_from_pos_lsa(lsa, codes, x, opt.min_seed_len, p.s + 1, &mut scratch, smems);
    }
}

/// Collect SMEMs across bwa-mem2's three rounds via the learned index (LISA analog of
/// [`crate::mem_collect_smem`]). Byte-identical SMEM set to the FM path.
pub fn mem_collect_smem_lsa(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = collect_smems_lsa(lsa, codes, opt.min_seed_len, 1);
    smem_round_2_lsa(lsa, codes, opt, &mut smems);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_lsa(lsa, codes, opt.max_mem_intv, opt.min_seed_len + 1, &mut smems);
    }
    smems
}

/// Turn one SMEM into reference-coordinate seeds using the learned suffix array (LISA analog of
/// [`crate::seeds_from_smem`]). `lsa.sa()[j]` equals `fm.get_sa(j)`, so the seeds are byte-identical.
pub fn seeds_from_smem_lsa(lsa: &LearnedSa, smem: &Smem, max_occ: i32) -> Vec<MemSeed> {
    let len = (i64::from(smem.n) - i64::from(smem.m) + 1) as i32;
    let max_occ = i64::from(max_occ);
    let step = if smem.s > max_occ { smem.s / max_occ } else { 1 };
    let sa = lsa.sa();
    let mut seeds = Vec::new();
    let mut c = 0i64;
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            rbeg: sa[j as usize],
            qbeg: smem.m as i32,
            len,
            score: len,
        });
        j += step;
        c += 1;
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_index::FmIndex;
    use std::path::Path;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// The LISA SMEM set must byte-match the FM path on real reads over the tiny reference, at both
    /// the SMEM level (m, n, k, s) and the derived seed level (rbeg, qbeg, len).
    #[test]
    fn lisa_seeding_matches_fmindex() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let opt = MemOpt::default();
        let l_pac = fm.l_pac() as usize;

        let mut seed = 0x51_5a_51_5a_1234_5678u64;
        for _ in 0..400 {
            // A read = a real substring of the forward reference (so it has genuine SMEMs), with a
            // few random mismatches sprinkled in to create multiple SMEMs.
            let rlen = 40 + (lcg(&mut seed) as usize % 120);
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            let n_mm = lcg(&mut seed) as usize % 4;
            for _ in 0..n_mm {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }
            // Occasional N.
            if lcg(&mut seed) % 5 == 0 {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = 4;
            }

            let fm_smems = crate::mem_collect_smem(&fm, &codes, &opt);
            let lsa_smems = mem_collect_smem_lsa(&lsa, &codes, &opt);

            // Compare SMEM sets on (m, n, k, s) — l is internal, rid is 0 during seeding.
            let key = |v: &[Smem]| -> Vec<(u32, u32, i64, i64)> {
                v.iter().map(|s| (s.m, s.n, s.k, s.s)).collect()
            };
            assert_eq!(
                key(&fm_smems),
                key(&lsa_smems),
                "SMEM mismatch: read start {start} len {rlen}"
            );

            // Compare the derived seeds too.
            for (a, b) in fm_smems.iter().zip(lsa_smems.iter()) {
                let sa_seeds = crate::seeds_from_smem(&fm, a, opt.max_occ);
                let lsa_seeds = seeds_from_smem_lsa(&lsa, b, opt.max_occ);
                assert_eq!(sa_seeds, lsa_seeds, "seed mismatch: read start {start}");
            }
        }
    }
}

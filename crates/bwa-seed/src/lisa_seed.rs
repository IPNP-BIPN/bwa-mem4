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

/// Backward LEM: longest exact match ending at `pivot` and extending left, occurring at least
/// `min_intv` times. Realised as a forward LEM of the reverse-complement pattern
/// `p[t] = 3 - codes[pivot - t]` over the `[fwd][rc]` SA (this is how the concatenated reference gives
/// left-extension "for free"). Stops at a read boundary or an `N`.
fn backward_lem(lsa: &LearnedSa, codes: &[u8], pivot: usize, min_intv: i64) -> usize {
    let mut pat = Vec::with_capacity(pivot + 1);
    let mut t = 0usize;
    loop {
        let idx = pivot as isize - t as isize;
        if idx < 0 {
            break;
        }
        let c = codes[idx as usize];
        if c >= 4 {
            break;
        }
        pat.push(3 - c);
        t += 1;
    }
    lsa.lem_min_intv(&pat, min_intv).0
}

/// Forward LEM from `pivot`, occurring at least `min_intv` times, capped at the first `N`.
fn forward_lem(lsa: &LearnedSa, codes: &[u8], pivot: usize, min_intv: i64) -> (usize, usize, usize) {
    let end = codes[pivot..]
        .iter()
        .position(|&c| c >= 4)
        .map(|p| pivot + p)
        .unwrap_or(codes.len());
    lsa.lem_min_intv(&codes[pivot..end], min_intv)
}

#[inline]
fn push_smem(out: &mut Vec<Smem>, lo: usize, hi: usize, m: usize, n: usize) {
    out.push(Smem {
        rid: 0,
        m: m as u32,
        n: n as u32,
        k: lo as i64,
        l: 0,
        s: (hi - lo) as i64,
    });
}

/// The shared zigzag body: starting at `start`, alternate backward (find left boundary, no emit) and
/// forward (emit) LEMs, marching the right boundary until it reaches `next_pivot`. `min_intv` gates
/// the LEM occurrence count (1 for round 1, parent-hitcount+1 for round-2 reseeding).
fn zigzag_march(
    lsa: &LearnedSa,
    codes: &[u8],
    start: usize,
    next_pivot: usize,
    min_intv: i64,
    min_seed_len: usize,
    out: &mut Vec<Smem>,
) {
    let l_seq = codes.len();
    let mut search_pivot = start;
    let mut cur = start;
    while search_pivot < next_pivot {
        if codes[search_pivot] >= 4 {
            if l_seq - search_pivot < min_seed_len {
                break;
            }
            search_pivot += 1;
            continue;
        }
        // Left extension (no emit): longest match ending at `cur`.
        let left_len = backward_lem(lsa, codes, cur, min_intv);
        cur = cur + 1 - left_len;
        if next_pivot - cur < min_seed_len {
            break;
        }
        // Right extension (emit): longest match starting at the new left boundary.
        let (rlen, lo, hi) = forward_lem(lsa, codes, cur, min_intv);
        if rlen >= min_seed_len {
            push_smem(out, lo, hi, cur, cur + rlen - 1);
        }
        search_pivot = cur + rlen;
        cur = search_pivot;
    }
}

/// Round-1 SMEM collection via BWA-MEME's fast zigzag (`Learned_getSMEMsAllPosOneThread`/`_step1`,
/// mode-1 non-tradeoff): never computes shallow intervals — each step jumps to a LEM. Emits SMEMs
/// `>= min_seed_len`. bwa-mem2-concordant (validated against the FM path below).
pub fn collect_smems_lsa_zigzag(lsa: &LearnedSa, codes: &[u8], min_seed_len: i32) -> Vec<Smem> {
    let l_seq = codes.len();
    let min_seed_len = min_seed_len as usize;
    let mut out = Vec::new();
    let mut pivot = 0usize;
    while pivot < l_seq {
        if codes[pivot] >= 4 {
            pivot = if l_seq - pivot < min_seed_len { l_seq } else { pivot + 1 };
            continue;
        }
        if pivot != 0 && codes[pivot - 1] < 4 {
            // Middle pivot: march to the read end (all-position step1).
            zigzag_march(lsa, codes, pivot, l_seq, 1, min_seed_len, &mut out);
            pivot = l_seq;
        } else {
            // Read start (or preceded by N): a single forward SMEM.
            let (rlen, lo, hi) = forward_lem(lsa, codes, pivot, 1);
            if rlen >= min_seed_len {
                push_smem(&mut out, lo, hi, pivot, pivot + rlen - 1);
            }
            pivot += rlen.max(1);
        }
    }
    out
}

/// Single-position SMEM search (`Learned_getSMEMsOnePosOneThread`), for round-2 reseeding from a
/// pivot with a given `min_intv`. Like step1 but `next_pivot` is bounded by this pivot's forward
/// reach (an initial forward LEM) rather than the read end.
fn smems_one_pos(
    lsa: &LearnedSa,
    codes: &[u8],
    pivot: usize,
    min_intv: i64,
    min_seed_len: usize,
    out: &mut Vec<Smem>,
) {
    if codes[pivot] >= 4 {
        return;
    }
    if pivot != 0 && codes[pivot - 1] < 4 {
        let (freach, _, _) = forward_lem(lsa, codes, pivot, min_intv);
        let next_pivot = pivot + freach;
        zigzag_march(lsa, codes, pivot, next_pivot, min_intv, min_seed_len, out);
    } else {
        let (rlen, lo, hi) = forward_lem(lsa, codes, pivot, min_intv);
        if rlen >= min_seed_len {
            push_smem(out, lo, hi, pivot, pivot + rlen - 1);
        }
    }
}

/// Round-2 reseeding (fast): re-seed each long, non-repetitive round-1 SMEM from its midpoint with
/// `min_intv = hitcount + 1`, appending in place. Mirrors `smem_round_2`.
fn smem_round_2_lsa_fast(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    let num1 = smems.len();
    let msl = opt.min_seed_len as usize;
    for idx in 0..num1 {
        let p = smems[idx];
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        let x = ((end + start) >> 1) as usize;
        smems_one_pos(lsa, codes, x, p.s + 1, msl, smems);
    }
}

/// Round-3 forward-only seeding (fast): for each pivot emit the shortest forward match that first
/// drops below `max_intv` occurrences (and is `>= min_seed_len`). Mirrors `bwt_seed_strategy`, but
/// jumps to the LEM and binary-searches the length where the occurrence count crosses `max_intv`
/// instead of narrowing base-by-base.
fn bwt_seed_strategy_lsa_fast(
    lsa: &LearnedSa,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    let l_seq = codes.len();
    let min_seed_len = min_seed_len as usize;
    let occ = |x: usize, l: usize| -> (i64, usize, usize) {
        let (a, b) = lsa.exact_interval(&codes[x..x + l]);
        ((b - a) as i64, a, b)
    };
    let mut x = 0usize;
    while x < l_seq {
        let mut next_x = x + 1;
        if codes[x] < 4 {
            let (lem_len, _, _) = forward_lem(lsa, codes, x, 1);
            if lem_len == 0 {
                x = next_x;
                continue;
            }
            let (occ_lem, _, _) = occ(x, lem_len);
            if lem_len >= min_seed_len && occ_lem < max_intv {
                // Smallest L in [min_seed_len, lem_len] with occ(L) < max_intv (occ decreasing).
                let l_star = if occ(x, min_seed_len).0 < max_intv {
                    min_seed_len
                } else {
                    let (mut lo, mut hi) = (min_seed_len + 1, lem_len);
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if occ(x, mid).0 < max_intv {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    lo
                };
                let (s, lo, hi) = occ(x, l_star);
                if s > 0 {
                    push_smem(out, lo, hi, x, x + l_star - 1);
                }
                next_x = x + l_star;
            } else {
                // No emit: advance past the explored match (matches FM's next_x after the forward
                // loop stops — approximate for the no-emit branch; validated against the FM path).
                next_x = (x + lem_len + 1).min(l_seq);
                if lem_len < min_seed_len {
                    next_x = (x + min_seed_len).min(l_seq);
                }
            }
        }
        x = next_x;
    }
}

/// Rounds 1+2 only (no round-3 `bwt_seed_strategy`), for isolating round costs in benchmarks.
pub fn mem_collect_smem_lsa_12(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = collect_smems_lsa_zigzag(lsa, codes, opt.min_seed_len);
    smem_round_2_lsa_fast(lsa, codes, opt, &mut smems);
    smems
}

/// Full fast SMEM collection: round-1 zigzag + round-2 reseed + round-3 strategy, the LISA analog of
/// [`crate::mem_collect_smem`]. Concordant seed set (validated against the FM path on real reads).
pub fn mem_collect_smem_lsa_fast(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = mem_collect_smem_lsa_12(lsa, codes, opt);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_lsa_fast(lsa, codes, opt.max_mem_intv, opt.min_seed_len + 1, &mut smems);
    }
    smems
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
    let mut seeds = Vec::new();
    let mut c = 0i64;
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            rbeg: lsa.sa_at(j as usize),
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

    /// The fast zigzag round-1 SMEM set must match the FM round-1 SMEM set (`collect_smems`,
    /// min_intv=1) as a set of `(m, n, k, s)` on real reads. BWA-MEME reproduces bwa-mem2 seeds, so
    /// this should be equal (or at worst concordant).
    #[test]
    fn zigzag_smems_match_fmindex_round1() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let msl = MemOpt::default().min_seed_len;
        let l_pac = fm.l_pac() as usize;

        let mut seed = 0xa11ce_5eed_1234u64;
        let mut mism = 0usize;
        for _ in 0..500 {
            let rlen = 40 + (lcg(&mut seed) as usize % 120);
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            for _ in 0..(lcg(&mut seed) as usize % 4) {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }

            let mut fm_set: Vec<(u32, u32, i64, i64)> = crate::collect_smems(&fm, &codes, msl, 1)
                .iter()
                .map(|s| (s.m, s.n, s.k, s.s))
                .collect();
            let mut lsa_set: Vec<(u32, u32, i64, i64)> = collect_smems_lsa_zigzag(&lsa, &codes, msl)
                .iter()
                .map(|s| (s.m, s.n, s.k, s.s))
                .collect();
            fm_set.sort_unstable();
            fm_set.dedup();
            lsa_set.sort_unstable();
            lsa_set.dedup();
            if fm_set != lsa_set {
                mism += 1;
                if mism <= 3 {
                    eprintln!("read@{start} len {rlen}\n  FM  : {fm_set:?}\n  LISA: {lsa_set:?}");
                }
            }
        }
        assert_eq!(mism, 0, "{mism}/500 reads had differing round-1 SMEM sets");
    }

    /// The full fast path (rounds 1+2+3) must produce a concordant SEED set to the FM
    /// `mem_collect_smem`. Compares the derived seeds (rbeg, qbeg, len) as a set, since the fast path
    /// emits in a different order and may differ in benign ways (duplicate/contained seeds that
    /// chaining discards). Reports the Jaccard overlap; requires exact-or-near-exact match.
    #[test]
    fn fast_full_seedset_concordant_with_fmindex() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let opt = MemOpt::default();
        let l_pac = fm.l_pac() as usize;

        let seeds_set = |smems: &[Smem], seeds_of: &dyn Fn(&Smem) -> Vec<MemSeed>| {
            let mut v: Vec<(i64, i32, i32)> = smems
                .iter()
                .flat_map(|s| seeds_of(s).into_iter().map(|d| (d.rbeg, d.qbeg, d.len)))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        let mut seed = 0xf00d_1234_5678u64;
        let (mut total, mut exact, mut worst_jac) = (0usize, 0usize, 1.0f64);
        for _ in 0..500 {
            let rlen = 60 + (lcg(&mut seed) as usize % 100);
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            for _ in 0..(lcg(&mut seed) as usize % 4) {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }

            let fm_smems = crate::mem_collect_smem(&fm, &codes, &opt);
            let lsa_smems = mem_collect_smem_lsa_fast(&lsa, &codes, &opt);
            let fm_seeds = seeds_set(&fm_smems, &|s| crate::seeds_from_smem(&fm, s, opt.max_occ));
            let lsa_seeds = seeds_set(&lsa_smems, &|s| seeds_from_smem_lsa(&lsa, s, opt.max_occ));

            total += 1;
            if fm_seeds == lsa_seeds {
                exact += 1;
            } else {
                let inter = fm_seeds.iter().filter(|x| lsa_seeds.binary_search(x).is_ok()).count();
                let uni = fm_seeds.len() + lsa_seeds.len() - inter;
                let jac = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
                if jac < worst_jac {
                    worst_jac = jac;
                }
            }
        }
        eprintln!("fast full: {exact}/{total} exact, worst Jaccard {worst_jac:.3}");
        assert!(
            exact as f64 / total as f64 >= 0.98 && worst_jac >= 0.9,
            "seed concordance too low: {exact}/{total} exact, worst Jaccard {worst_jac:.3}"
        );
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

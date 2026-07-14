//! SMEM seeding via the FMD index, mirroring bwa-mem2's `getSMEMsOnePosOneThread` /
//! `getSMEMsAllPosOneThread` (`reference/bwa-mem2/src/FMI_search.cpp`) and the seed derivation of
//! `get_sa_entries`.
//!
//! Phase 3 implements round 1 (all-position SMEM collection, `min_intv = 1`) and turns SMEM
//! intervals into reference-coordinate seeds. Reseeding rounds 2/3 (`getSMEMsOnePos` re-seeding of
//! long/repetitive SMEMs and `bwtSeedStrategy`) are layered on later; the end-to-end byte-identity
//! gate for seeding is the SE SAM concordance in phase 6.

use bwa_core::MemOpt;
use bwa_index::{FmIndex, Smem};

/// A seed: an exact match between the read and the reference (bwa-mem2's `mem_seed_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemSeed {
    /// Reference begin, in the 2L forward++RC coordinate space.
    pub rbeg: i64,
    /// Query begin (0-based).
    pub qbeg: i32,
    /// Seed length.
    pub len: i32,
    /// Seed score (length, for exact matches).
    pub score: i32,
}

/// Collect all round-1 SMEMs of `codes` (2-bit encoded read, N = 4) with the given seed length and
/// minimum interval size. Mirrors `getSMEMsAllPosOneThread` (repeatedly calling the one-position
/// routine, advancing the start to `next_x`).
pub fn collect_smems(fm: &FmIndex, codes: &[u8], min_seed_len: i32, min_intv: i64) -> Vec<Smem> {
    let mut out = Vec::new();
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    let mut x = 0usize;
    while x < codes.len() {
        x = smems_from_pos(fm, codes, x, min_seed_len, min_intv, &mut scratch, &mut out);
    }
    out
}

/// One-position SMEM search starting at `x`, appending SMEMs to `out` and returning `next_x`.
/// Faithful port of `getSMEMsOnePosOneThread`'s inner body.
fn smems_from_pos(
    fm: &FmIndex,
    codes: &[u8],
    x: usize,
    min_seed_len: i32,
    min_intv: i64,
    prev: &mut [Smem],
    out: &mut Vec<Smem>,
) -> usize {
    let readlength = codes.len();
    let counts = fm.counts();
    let mut next_x = x + 1;
    let a = codes[x];
    if a >= 4 {
        return next_x;
    }

    // Initial single-base interval.
    let a = a as usize;
    let mut smem = Smem {
        rid: 0,
        m: x as u32,
        n: x as u32,
        k: counts[a],
        l: counts[3 - a],
        s: counts[a + 1] - counts[a],
    };
    let mut num_prev = 0usize;

    // Forward extension (backward extension on the RC via swapped k/l and complemented base).
    let mut j = x + 1;
    while j < readlength {
        let aj = codes[j];
        next_x = j + 1;
        if aj >= 4 {
            break;
        }
        let mut fwd = smem;
        std::mem::swap(&mut fwd.k, &mut fwd.l);
        let ext = fm.backward_ext(fwd, 3 - aj as usize);
        let mut new_smem = ext;
        std::mem::swap(&mut new_smem.k, &mut new_smem.l);
        new_smem.n = j as u32;

        prev[num_prev] = smem;
        if new_smem.s != smem.s {
            num_prev += 1;
        }
        if new_smem.s < min_intv {
            next_x = j;
            break;
        }
        smem = new_smem;
        j += 1;
    }
    if smem.s >= min_intv {
        prev[num_prev] = smem;
        num_prev += 1;
    }

    prev[..num_prev].reverse();

    // Backward extension.
    let mut jj = x as i64 - 1;
    while jj >= 0 {
        let a = codes[jj as usize];
        if a > 3 {
            break;
        }
        let a = a as usize;
        let mut num_curr = 0usize;
        let mut curr_s = -1i64;

        let mut p = 0usize;
        while p < num_prev {
            let sm = prev[p];
            let mut new_smem = fm.backward_ext(sm, a);
            new_smem.m = jj as u32;
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
                // Prefetch the checkpoint blocks the next SMEM step's backward_ext on this kept
                // interval will touch, one step ahead (bwa-mem2 / nh13 `ENABLE_PREFETCH`).
                fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
                break;
            }
            p += 1;
        }
        p += 1;
        while p < num_prev {
            let sm = prev[p];
            let mut new_smem = fm.backward_ext(sm, a);
            new_smem.m = jj as u32;
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
                fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
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

/// Round-3 forward-only seeding (`bwtSeedStrategyAllPosOneThread`): emit a seed when the interval
/// first drops below `max_intv` and the seed is at least `min_seed_len` long.
fn bwt_seed_strategy(
    fm: &FmIndex,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    let counts = fm.counts();
    let readlength = codes.len();
    let mut x = 0usize;
    while x < readlength {
        let mut next_x = x + 1;
        if codes[x] < 4 {
            let a = codes[x] as usize;
            let mut smem = Smem {
                rid: 0,
                m: x as u32,
                n: x as u32,
                k: counts[a],
                l: counts[3 - a],
                s: counts[a + 1] - counts[a],
            };
            let mut j = x + 1;
            while j < readlength {
                next_x = j + 1;
                let aj = codes[j];
                if aj >= 4 {
                    break;
                }
                let mut fwd = smem;
                std::mem::swap(&mut fwd.k, &mut fwd.l);
                let ext = fm.backward_ext(fwd, 3 - aj as usize);
                let mut new_smem = ext;
                std::mem::swap(&mut new_smem.k, &mut new_smem.l);
                new_smem.n = j as u32;
                smem = new_smem;
                if smem.s < max_intv
                    && (i64::from(smem.n) - i64::from(smem.m) + 1) >= i64::from(min_seed_len)
                {
                    if smem.s > 0 {
                        out.push(smem);
                    }
                    break;
                }
                j += 1;
            }
        }
        x = next_x;
    }
}

/// Collect SMEMs across bwa-mem2's three rounds (`mem_collect_smem`): round-1 all-position SMEMs,
/// round-2 re-seeding of long non-repetitive SMEMs from their midpoint, and round-3 interval-capped
/// forward seeding. This is the full seed set feeding chaining.
pub fn mem_collect_smem(fm: &FmIndex, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;

    // Round 1.
    let mut smems = collect_smems(fm, codes, opt.min_seed_len, 1);
    let num1 = smems.len();

    // Round 2: re-seed each long, non-repetitive round-1 SMEM from its midpoint.
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    for idx in 0..num1 {
        let p = smems[idx];
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        let x = ((end + start) >> 1) as usize;
        smems_from_pos(
            fm,
            codes,
            x,
            opt.min_seed_len,
            p.s + 1,
            &mut scratch,
            &mut smems,
        );
    }

    // Round 3.
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy(
            fm,
            codes,
            opt.max_mem_intv,
            opt.min_seed_len + 1,
            &mut smems,
        );
    }

    smems
}

/// Turn one SMEM into reference-coordinate seeds, sampling up to `max_occ` occurrences
/// (bwa-mem2's `get_sa_entries` stride sampling).
pub fn seeds_from_smem(fm: &FmIndex, smem: &Smem, max_occ: i32) -> Vec<MemSeed> {
    let len = (i64::from(smem.n) - i64::from(smem.m) + 1) as i32;
    let max_occ = i64::from(max_occ);
    let step = if smem.s > max_occ {
        smem.s / max_occ
    } else {
        1
    };
    let mut seeds = Vec::new();
    let mut c = 0i64;
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            rbeg: fm.get_sa(j),
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
    use std::path::Path;

    fn tiny() -> FmIndex {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    /// Occurrences of `pat` in the binary reference (both strands, since .0123 is fwd++RC).
    fn naive_occ(reference: &[u8], pat: &[u8]) -> i64 {
        reference.windows(pat.len()).filter(|w| *w == pat).count() as i64
    }

    #[test]
    fn smems_are_exact_matches() {
        let fm = tiny();
        // Read = a 120bp forward slice of the reference; must seed back to its origin.
        let start = 50_000i64;
        let len = 120usize;
        let read: Vec<u8> = (0..len).map(|i| fm.base(start + i as i64)).collect();

        let smems = collect_smems(&fm, &read, 19, 1);
        assert!(!smems.is_empty(), "no SMEMs found");

        // Each SMEM's read substring occurs exactly `s` times in the reference.
        for sm in &smems {
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(
                sm.s,
                naive_occ(fm.reference(), sub),
                "SMEM interval size wrong"
            );
            assert!((sm.n - sm.m + 1) as i32 >= 19);
        }

        // The full-length SMEM must exist and seed to the origin position.
        let full = smems.iter().find(|s| s.m == 0 && s.n as usize == len - 1);
        let full = full.expect("no full-length SMEM covering the read");
        let seeds = seeds_from_smem(&fm, full, 500);
        assert!(
            seeds
                .iter()
                .any(|s| s.rbeg == start && s.qbeg == 0 && s.len == len as i32),
            "no seed mapping back to the origin at {start}"
        );
    }

    #[test]
    fn smems_cover_repeated_region() {
        let fm = tiny();
        // A short read is still collected as an SMEM if >= min_seed_len.
        let start = 123_456i64.min(fm.l_pac() - 60);
        let read: Vec<u8> = (0..60).map(|i| fm.base(start + i as i64)).collect();
        let smems = collect_smems(&fm, &read, 19, 1);
        for sm in &smems {
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(sm.s, naive_occ(fm.reference(), sub));
        }
    }
}

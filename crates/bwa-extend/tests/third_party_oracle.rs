//! Cross-check our local Smith-Waterman score against an INDEPENDENT implementation.
//!
//! Every other parity test in this workspace compares our SIMD kernels against our own scalar
//! `ksw_local_fwd`. That catches a vectorisation bug and cannot, by construction, catch a bug the
//! scalar reference shares with them: a misread of `ksw.cpp` would be reproduced faithfully by all
//! six kernels and every test would stay green. This test closes that hole with a second opinion
//! from code that has never seen ours.
//!
//! The oracle is [hyalite](https://github.com/Psy-Fer/hyalite), a Rust reimplementation of Opal
//! (Rognes-style inter-sequence SIMD Smith-Waterman), MIT-licensed, dev-dependency only. It is a
//! deliberately narrow check, and the narrowness is the point: it validates exactly the one quantity
//! both implementations agree on the meaning of.
//!
//! # What this test does NOT check, and why
//!
//! `ksw_align2` returns seven values; only `score` is comparable.
//!
//! - `qe`/`te`/`qb`/`tb` depend on bwa's tie-breaking among equally scoring alignments and on the
//!   `KSW_XSTART` reverse pass over reversed prefixes. hyalite reports its own end positions under
//!   its own rules, so a mismatch there would mean nothing.
//! - `score2`/`te2` come from the per-row maxima above `minsc`, outside an exclusion window around
//!   `te`. hyalite does not compute per-row maxima at all (there is no such quantity anywhere in its
//!   API), which is also why it cannot replace our mate-rescue kernel.
//!
//! # Two conventions that must be translated, not assumed
//!
//! 1. **Gap cost.** hyalite follows Opal: a gap of length `n` costs `open + (n - 1) * ext`. ksw and
//!    we charge `o + n * e`. They are the same affine function under `open = o + e`, `ext = e`, and
//!    that substitution is what this test passes. Getting it wrong would produce a plausible-looking
//!    failure that has nothing to do with either implementation being wrong.
//! 2. **Matrix indexing.** hyalite reads `matrix[q * alphabet_len + t]`, bwa reads `mat[t * m + q]`.
//!    bwa's DNA matrix is symmetric, so the transposition is a no-op here; it is written out anyway
//!    so a future asymmetric matrix does not silently transpose.
//!
//! hyalite carries a single gap-penalty pair, so it cannot express `o_del != o_ins`. The test
//! therefore runs symmetric penalties, which is bwa's default (`-O 6 -E 1`).

use bwa_mem4_extend::ksw_align2;
use hyalite::{align_pair, Mode, Scoring, SearchType};

/// bwa's default DNA matrix, built as `bwa_fill_scmat` does.
///
/// # Returns
/// A 25-entry (`m = 5`) row-major matrix: `+a` on the diagonal, `-b` off-diagonal among the four
/// concrete bases, `-1` on the whole N row and N column.
fn bwa_matrix(a: i8, b: i8) -> Vec<i8> {
    let mut mat = vec![0i8; 25];
    // Write cursor walking the matrix in row-major order: four rows of (4 base scores + the N
    // column), then the final all-N row.
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

/// The same matrix in hyalite's layout and element type.
///
/// # Parameters
/// - `mat`: bwa's `m * m` matrix, indexed `mat[t * m + q]`.
/// - `m`: alphabet size (5 here).
///
/// # Returns
/// An `m * m` `i32` matrix indexed `out[q * m + t]`, i.e. the transpose, widened.
fn to_hyalite_matrix(mat: &[i8], m: usize) -> Vec<i32> {
    let mut out = vec![0i32; m * m];
    for t in 0..m {
        for q in 0..m {
            out[q * m + t] = i32::from(mat[t * m + q]);
        }
    }
    out
}

/// Our best local-alignment score for one pair, with the `score2` machinery suppressed.
///
/// # Parameters
/// - `query`, `target`: nt4 codes, `0..=4`.
/// - `mat`, `o`, `e`: bwa's matrix and its symmetric gap open / extend magnitudes.
///
/// # Returns
/// `KswAlignResult::score`, the only field this test compares.
fn our_score(query: &[u8], target: &[u8], mat: &[i8], o: i32, e: i32) -> i32 {
    // `minsc = i32::MAX` suppresses the 2nd-best tracker (nothing can clear it), and `lanes` only
    // sets the query-profile padding, which is observable in `score2` and never in `score`.
    ksw_align2(
        query,
        target,
        5,
        mat,
        o,
        e,
        o,
        e,
        i32::MAX,
        i32::from(mat[0]),
        16,
    )
    .score
}

/// hyalite's best local-alignment score for the same pair.
///
/// # Returns
/// `BestHit::score` from `Mode::Sw` (the only local mode it offers), under the translated gap
/// convention described in the module docs.
fn oracle_score(query: &[u8], target: &[u8], mat: &[i8], o: i32, e: i32) -> i32 {
    // `open = o + e` and `ext = e`: hyalite charges `open + (n-1)*ext` where we charge `o + n*e`.
    let scoring = Scoring::new(5, to_hyalite_matrix(mat, 5), o + e, e)
        .expect("bwa's matrix and penalties are a valid hyalite scoring scheme");
    align_pair(query, target, &scoring, Mode::Sw, SearchType::Score)
        .expect("codes are all inside the 5-symbol alphabet")
        .score
}

/// Our local SW score must equal an independent implementation's, over randomised pairs.
///
/// The pairs are built the way the kernel tests build them: mostly a copy of the query so the
/// alignment is a noisy relative rather than unrelated noise, since two unrelated sequences score
/// near zero and would exercise nothing. `N` is injected on both sides so the `-1` row and column of
/// the matrix are covered.
#[test]
fn local_sw_score_matches_independent_implementation() {
    // bwa's `-A 1 -B 4 -O 6 -E 1`, plus a second scheme so the check is not tied to one point in
    // parameter space.
    for &(a, b, o, e) in &[(1i8, 4i8, 6i32, 1i32), (2, 3, 5, 2)] {
        let mat = bwa_matrix(a, b);
        // Deterministic LCG (Numerical Recipes 64-bit constants), top 31 bits taken so the
        // low-order-bit weakness never reaches the small moduli below. Fixed seed: a failure is
        // reproducible and the assert prints the round.
        let mut state = 0x51D3_9CF1_0BAD_C0DEu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for round in 0..300u32 {
            let qlen = 1 + (next() % 160) as usize;
            // Query over the full alphabet, N included (code 4) at ~3%.
            let q: Vec<u8> = (0..qlen)
                .map(|_| {
                    if next() % 100 < 3 {
                        4
                    } else {
                        (next() % 4) as u8
                    }
                })
                .collect();
            // Target: a noisy relative of the query, with room to run off either end.
            let tlen = qlen + (next() % 200) as usize;
            let mut t: Vec<u8> = Vec::with_capacity(tlen);
            // Read cursor into `q`: mostly copy the next query base, otherwise emit a random base
            // and coin-flip whether to also advance, giving mismatches and indels respectively.
            let mut qi = 0usize;
            while t.len() < tlen {
                if qi < q.len() && next() % 100 >= 8 {
                    t.push(q[qi]);
                    qi += 1;
                } else if next() % 100 < 3 {
                    t.push(4);
                    if next() % 2 == 0 {
                        qi += 1;
                    }
                } else {
                    t.push((next() % 4) as u8);
                    if next() % 2 == 0 {
                        qi += 1;
                    }
                }
            }

            let ours = our_score(&q, &t, &mat, o, e);
            let theirs = oracle_score(&q, &t, &mat, o, e);
            assert_eq!(
                ours, theirs,
                "local SW score disagrees with hyalite at A={a} B={b} O={o} E={e} round {round}, \
                 qlen={qlen} tlen={tlen}"
            );
        }
    }
}

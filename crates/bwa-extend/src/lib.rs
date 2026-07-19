//! Seed extension via banded Smith-Waterman.
//!
//! The scalar seed-extension kernel (`sw::ksw_extend2`) is the bit-identity source of truth;
//! NEON/Metal backends must reproduce its integer results.
//!
//! # What this crate is for
//!
//! The aligner turns each short exact match ("seed") into a real alignment by extending it outwards
//! with dynamic programming. This crate owns that step twice over: [`sw`] holds the actual
//! arithmetic (the portable, authoritative version), and this file holds the *plug interface* plus
//! the acceptance tests that every faster reimplementation has to survive.
//!
//! # Order to read this file in
//!
//! 1. [`SwBackend`], the one-method-per-mode trait an accelerated implementation fills in, and
//!    [`ExtendJob`], one alignment's worth of per-job input.
//! 2. [`ScalarBackend`], the trivial implementation that just calls [`ksw_extend2`]. It is the
//!    definition of "correct" for everything else.
//! 3. [`assert_backend_matches_scalar`], the single-alignment acceptance gate, and its long comment
//!    about why the *scoring* is swept and not just the sequence geometry.
//! 4. [`assert_backend_batch_matches_scalar`], the batch gate, which hunts a different bug class
//!    (SIMD lanes leaking into each other).
//!
//! # Vocabulary used throughout
//!
//! `query` is the read, `target` the stretch of reference it is being compared against, both encoded
//! as one byte per base (`0=A 1=C 2=G 3=T`, `4=N`). `h0` is the score the seed already earned.
//! `qle`/`tle` are how many query/target bases the best local alignment consumed; `gscore`/`gtle`
//! are the score and target length of the best alignment that instead consumes the *whole* query.
//! The full glossary of the short C-derived names lives in the [`sw`] module header.

pub mod sw;

/// Multiplier of the 64-bit linear congruential generator the acceptance gates use for their
/// deterministic pseudo-random cases. This value and [`LCG_INCREMENT`] are the widely used pair
/// from Knuth's MMIX (also the LCG stage inside PCG). They are here only so the two gates below
/// generate reproducible input, not because bwa uses them anywhere.
const LCG_MULTIPLIER: u64 = 6364136223846793005;
/// Increment of the acceptance gates' LCG. See [`LCG_MULTIPLIER`].
const LCG_INCREMENT: u64 = 1442695040888963407;

pub use sw::{ksw_align2, ksw_extend2, ksw_global2, ksw_local_fwd, ExtendResult, KswAlignResult};

/// A banded Smith-Waterman seed-extension backend.
///
/// The scalar backend ([`ScalarBackend`], delegating to [`ksw_extend2`]) is authoritative; SIMD
/// (NEON) and GPU (Metal) backends must return **integer-identical** [`ExtendResult`]s to it for
/// every input, so byte-identity of the SAM output is preserved. Use
/// [`assert_backend_matches_scalar`] as the acceptance gate when adding a backend.
///
/// "Integer-identical" rather than "close enough" is affordable here because the recurrence is
/// entirely integer: there is no floating point in the DP, so a correct reimplementation on any
/// hardware gives bit-for-bit the same answer. Any difference is a bug, never a rounding artifact.
/// (The one `f64` in [`ksw_extend2`] is the band clamp, and backends are expected to compute it on
/// the CPU in `f64` and pass the result in, rather than re-derive it.)
pub trait SwBackend {
    /// Short backend name, e.g. `"scalar"`, `"neon"`, `"metal"`.
    fn name(&self) -> &'static str;

    /// One banded local seed extension, mirroring [`ksw_extend2`] exactly.
    ///
    /// # Parameters
    ///
    /// Every parameter has the same meaning, encoding and units as the identically named parameter
    /// of [`ksw_extend2`], and an implementation must honour the same preconditions:
    ///
    /// * `query`, `target`: the read and the reference stretch, one byte per base in 2-bit codes
    ///   (`0=A 1=C 2=G 3=T`, `4=N`), already oriented so extension runs left to right from the seed.
    ///   Both are non-empty. Lengths are independent and unpadded.
    /// * `m`: alphabet size, always `5` in bwa. `mat`: the `m*m` row-major substitution matrix,
    ///   `mat[t*m + q]`, in score points.
    /// * `o_del`, `e_del`, `o_ins`, `e_ins`: affine gap open and extend penalties for deletions and
    ///   insertions, positive magnitudes in score points that get subtracted. Extends must be `>= 1`
    ///   (they divide in the band clamp).
    /// * `w`: half band width in cells, `>= 1`. `end_bonus`: query-end bonus in score points, used
    ///   only to widen the band clamp. `zdrop`: early-exit threshold in score points, `<= 0`
    ///   disables it.
    /// * `h0`: the seed's already-earned score, the DP's starting value. Must be `> 0`.
    ///
    /// All of them are supplied by the aligner's chain-to-alignment step (`mem_chain2aln`), which
    /// reads the scoring and band out of `mem_opt_t`.
    ///
    /// # Returns
    ///
    /// The [`ExtendResult`] that [`ksw_extend2`] would return for these arguments, field for field,
    /// as exact integers. Not "within a point": see the trait docs.
    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
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
    ) -> ExtendResult;

    /// Extend a batch of seed extensions that share the same scoring (`m`, `mat`, gap penalties,
    /// band `w`, `end_bonus`, `zdrop`) but differ in query/target/`h0`. This is the entry point a
    /// SIMD backend accelerates: bwa-mem2's `bandedSWA` packs `SIMD_WIDTH` such alignments across
    /// vector lanes (8 for int16, 16 for int8), one independent alignment per lane. The result at
    /// index `k` must equal `extend(jobs[k]...)` exactly. The default loops the scalar `extend`.
    ///
    /// # Parameters
    ///
    /// * `jobs`: the per-alignment inputs, one [`ExtendJob`] per lane. May be any length, including
    ///   lengths that are not a multiple of the backend's SIMD width: a partial final vector is the
    ///   implementation's problem, not the caller's. An empty slice yields an empty result.
    /// * `m`, `mat`, `o_del`, `e_del`, `o_ins`, `e_ins`, `w`, `end_bonus`, `zdrop`: exactly as in
    ///   [`SwBackend::extend`], but shared by every job in the batch. Sharing them is what makes
    ///   lane packing possible; `h0` and the sequences are the only things allowed to vary, and they
    ///   live in `jobs`.
    ///
    /// # Returns
    ///
    /// One [`ExtendResult`] per job, in job order, with `result.len() == jobs.len()`. Result `k`
    /// must be integer-identical to `extend(jobs[k].query, jobs[k].target, .., jobs[k].h0)`; in
    /// particular the longest job in the batch must not influence any other job's band, z-drop or
    /// termination.
    #[allow(clippy::too_many_arguments)]
    fn extend_batch(
        &self,
        jobs: &[ExtendJob],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w: i32,
        end_bonus: i32,
        zdrop: i32,
    ) -> Vec<ExtendResult> {
        jobs.iter()
            .map(|j| {
                self.extend(
                    j.query, j.target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                    j.h0,
                )
            })
            .collect()
    }
}

/// One seed extension in a batch: the per-alignment inputs that vary across lanes (the scoring
/// scheme and band are shared, passed to [`SwBackend::extend_batch`]).
#[derive(Debug, Clone, Copy)]
pub struct ExtendJob<'a> {
    /// The read side of this alignment: 2-bit base codes (`0=A 1=C 2=G 3=T`, `4=N`), one byte per
    /// base, oriented so extension runs left to right from the seed end. Non-empty, unpadded, and
    /// borrowed from the caller's read buffer, so a batched backend must not assume any particular
    /// length relative to its neighbours in the batch.
    pub query: &'a [u8],
    /// The reference side, same encoding and orientation as `query`. Its length is independent of
    /// `query`'s.
    pub target: &'a [u8],
    /// The score this seed had already earned before extension, in score points. Strictly positive
    /// (the recurrence's "zero means unreachable" sentinel depends on it). It is the DP's value at
    /// the origin, so the returned `score` is always `>= h0`. This is the field that most often
    /// differs between lanes of one batch.
    pub h0: i32,
}

/// The authoritative scalar backend: `extend` delegates straight to [`ksw_extend2`]. Every other
/// backend is validated against this one.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarBackend;

impl SwBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
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
        ksw_extend2(
            query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        )
    }
}

/// Acceptance gate for any non-scalar [`SwBackend`]: assert `backend.extend` returns exactly the
/// same [`ExtendResult`] as [`ksw_extend2`] over a deterministic sweep of random DNA seed
/// extensions. Panics on the first mismatch. NEON (phase 9a) and Metal (phase 9b) backends must
/// pass this.
///
/// # What this actually guarantees
///
/// All six fields of [`ExtendResult`] are compared with `assert_eq!`, not just `score`. That is
/// the point: `qle`/`tle`/`gtle`/`gscore`/`max_off` decide soft-clipping and band retries, so a
/// backend that agrees on score alone still produces different SAM. The comparison is exact integer
/// equality, so passing this gate means the backend is byte-identical **on the sampled inputs**,
/// not merely concordant.
///
/// The generator is a fixed-seed LCG, so the 2000 cases are the same on every run and on every
/// machine: a failure is always reproducible, and a backend cannot pass by luck on one run. The
/// panic message prints the full case (lengths, band, zdrop, end_bonus, h0 **and** the scoring) so
/// a failure is directly replayable against [`ksw_extend2`].
///
/// # Do not narrow this sweep. It was blind once, and something got through.
///
/// This gate originally pinned the scoring to bwa's defaults: `(a, b) = (1, 4)` and gaps
/// `(6, 1, 6, 1)`, varying only the geometry (lengths, `w`, `zdrop`, `end_bonus`, `h0`). Under
/// symmetric default penalties the E/F recurrence's gap-open source is nearly unobservable, because
/// `H == M` at almost every cell that ends up on the optimal path. A backend that opened gaps from
/// `H = max(M, E, F)` instead of from `M` therefore passed all 2000 cases while being wrong.
///
/// That is exactly what had happened: the Metal shader in `bwa-gpu` computed its E/F floor from
/// `h - oe_del` / `h - oe_ins` where `ksw_extend2` (`ksw.cpp:493`, `498`) and bwa-mem2's vectorized
/// `bandedSWA` (`MAIN_CODE16`, `bandedSWA.cpp:327-342`, subtracting from `m11`) both use `M`.
/// Adding the `(a, b, o_del, e_del, o_ins, e_ins)` sweep below surfaced it immediately.
///
/// So: the scoring sweep is load-bearing coverage, not decoration. Anyone tempted to pin the
/// penalties back to defaults (to make a failure go away, to speed the test up, to "simplify")
/// would restore precisely the blind spot that let a real divergence ship. Case 0 already preserves
/// the historical default-scoring case; add cases, do not remove the variation.
///
/// Two limits worth stating honestly, so nobody over-reads a green run:
/// * Sequences are uniform random DNA with `qlen`/`tlen <= 60`. Real reads are longer and contain
///   satellite repeats, which is the other regime where H-vs-M diverges. Callers with a GPU/SIMD
///   backend should also run a read-sized gate (see `bwa-gpu`'s `metal_matches_scalar_read_sized`).
/// * A backend that silently falls back to scalar passes vacuously. Backends must assert
///   separately that their real kernel was actually built and taken.
///
/// # Parameters
///
/// * `backend`: the implementation under test. Only [`SwBackend::extend`] and [`SwBackend::name`]
///   are exercised (the latter only to name the backend in the panic message);
///   [`SwBackend::extend_batch`] is covered by [`assert_backend_batch_matches_scalar`]. Passing
///   [`ScalarBackend`] is a self-test of the harness and always succeeds.
///
/// # Returns
///
/// Nothing: it either returns normally (all 2000 cases matched) or panics on the first mismatch.
pub fn assert_backend_matches_scalar<B: SwBackend>(backend: &B) {
    // Deterministic LCG for reproducible random cases (no extra deps).
    // `rng_state` is the LCG's 64-bit internal state, seeded with a fixed constant so the sweep
    // replays identically on every machine and every run. It is captured by `next_random` and never
    // read directly: only the closure's return value (the top 31 bits, which are the well-mixed
    // ones of an LCG) is consumed as the random stream.
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_random = move || {
        rng_state = rng_state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        rng_state >> 33
    };

    for case in 0..2000u32 {
        // Lengths start at 1, never 0: ksw_extend2 indexes eh_h[1] unconditionally.
        let qlen = 1 + (next_random() % 60) as usize;
        let tlen = 1 + (next_random() % 60) as usize;
        let query: Vec<u8> = (0..qlen).map(|_| (next_random() % 4) as u8).collect();
        let target: Vec<u8> = (0..tlen).map(|_| (next_random() % 4) as u8).collect();
        // Bands from 1 to 30 against lengths up to 60, so the band genuinely bites on most cases
        // (a `w` that always exceeded the sequence would never exercise the adaptive shrink).
        let w = 1 + (next_random() % 30) as i32;
        // zdrop 0 is included on purpose: 0 disables the z-drop branch entirely, so both the
        // early-exit path and the run-to-completion path get covered.
        let zdrop = (next_random() % 120) as i32;
        let end_bonus = (next_random() % 10) as i32;
        // ksw_extend2 requires a positive initial score (extension starts from a seed).
        let h0 = 1 + (next_random() % 20) as i32;
        // Sweep the SCORING too, not just the geometry. Pinning (a,b) = (1,4) and the gaps to
        // (6,1,6,1) -- as this gate used to -- makes it structurally blind to every bug that only
        // shows up under `-A/-B/-O/-E`, which is exactly how a real gap-vs-mismatch divergence
        // survived here. Case 0 keeps bwa's defaults so the historical coverage is preserved.
        //
        // Ranges chosen to break the symmetries the default scheme happens to have: `o_del` and
        // `o_ins` (and `e_del`/`e_ins`) are drawn independently, so deletion and insertion penalties
        // usually differ, which is what makes an E-vs-F mix-up or an H-vs-M gap open observable.
        // Extends stay >= 1 because ksw_extend2's band clamp divides by them; opens stay >= 1 and
        // mismatch >= 1 so penalties are real penalties. See this function's docs: do not shrink
        // these ranges.
        let (a, b, o_del, e_del, o_ins, e_ins) = if case == 0 {
            (1i8, 4i8, 6i32, 1i32, 6i32, 1i32)
        } else {
            (
                1 + (next_random() % 4) as i8,
                1 + (next_random() % 6) as i8,
                1 + (next_random() % 8) as i32,
                1 + (next_random() % 3) as i32,
                1 + (next_random() % 8) as i32,
                1 + (next_random() % 3) as i32,
            )
        };
        // The 5x5 scoring matrix this case will be run under, built from the (a, b) drawn above.
        let mat = fill_scmat_vec(a, b);

        // `expected` is the authoritative answer (the scalar kernel), `got` the backend's; the two
        // calls take byte-for-byte the same arguments, so any difference between them is a backend
        // bug rather than a difference of input.
        let expected = ksw_extend2(
            &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        );
        let got = backend.extend(
            &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        );
        assert_eq!(
            got,
            expected,
            "backend {:?} diverged from scalar on case {case} (qlen={qlen} tlen={tlen} w={w} zdrop={zdrop} end_bonus={end_bonus} h0={h0} a={a} b={b} O={o_del},{o_ins} E={e_del},{e_ins})",
            backend.name()
        );
    }
}

/// 5x5 bwa-style scoring matrix (match `a`, mismatch `-b`, N = -1), as `bwa_fill_scmat` builds it.
/// Row-major over the code alphabet `A,C,G,T,N`, so `mat[t*5 + q]`. The `N` row and column are a
/// flat `-1` regardless of `a`/`b`: bwa neither rewards nor heavily punishes an ambiguous base.
///
/// Coverage gap, stated so it is not mistaken for coverage: both gates above draw bases with
/// `next_random() % 4`, so code `4` never appears in a query or target and the `N` row/column of this
/// matrix is never actually scored. A backend that hard-coded the ambiguity penalty, or dropped it,
/// would still pass. Real reads do contain `N`.
///
/// # Parameters
///
/// * `a`: the match score, a positive magnitude in score points, written on the diagonal. bwa's
///   `-A`, default `1`. Only the ratios between `a`, `b` and the gap penalties affect an alignment,
///   which is why the gates sweep small values (1..=4) rather than realistic ones.
/// * `b`: the mismatch penalty, a positive magnitude written as `-b` off the diagonal. bwa's `-B`,
///   default `4`. Should be `>= 1` so a mismatch is genuinely a penalty.
///
/// # Returns
///
/// A 25-element row-major `5x5` matrix indexable as `mat[t*5 + q]` for target code `t` and query
/// code `q`, suitable to pass straight as the `mat` argument of [`ksw_extend2`].
fn fill_scmat_vec(a: i8, b: i8) -> Vec<i8> {
    let mut mat = vec![0i8; 25];
    // Cursor into `mat`: the flat index of the next entry to write, so it walks the 25 cells in
    // row-major order and ends at 25.
    let mut write_pos = 0;
    // Rows 0..4 are the concrete bases A/C/G/T: `a` where the two bases agree, `-b` where they do
    // not, then a trailing `-1` for the N column of that row.
    for target_code in 0..4 {
        for query_code in 0..4 {
            mat[write_pos] = if target_code == query_code { a } else { -b };
            write_pos += 1;
        }
        mat[write_pos] = -1;
        write_pos += 1;
    }
    // Row 4 is the N row: `-1` against everything, including another N.
    for _ in 0..5 {
        mat[write_pos] = -1;
        write_pos += 1;
    }
    mat
}

/// Batch-mode acceptance gate: assert `backend.extend_batch` returns, for every job, exactly the
/// [`ksw_extend2`] result. Batches share scoring/band (as the API requires) and vary in
/// query/target/`h0`, across a range of batch sizes (including sizes that don't fill a SIMD width).
/// A SIMD backend must pass this in addition to [`assert_backend_matches_scalar`].
///
/// This catches a different failure class from the single-alignment gate: **lane cross-talk**. A
/// vector backend packs several independent alignments side by side, so it can produce correct
/// results one at a time yet leak state between lanes, mis-handle a partial final vector, or let
/// the longest job in a batch dictate another job's band or termination. Hence the batch sizes
/// deliberately straddle the SIMD widths in use (8 and 16): 7/8/9 and 15/16/17 exercise the
/// under-full, exactly-full and one-over cases, and the jobs inside a batch get independently
/// random lengths so no lane sees the same geometry as its neighbour.
///
/// Note this gate keeps `(a, b)` at bwa's defaults and only varies the gaps. It is a *lane*
/// correctness gate; the scoring sweep that catches recurrence bugs lives in
/// [`assert_backend_matches_scalar`], and both must be run.
///
/// # Parameters
///
/// * `backend`: the implementation under test. Exercises [`SwBackend::extend_batch`] only (plus
///   [`SwBackend::name`] for the panic message). A backend that has not overridden `extend_batch`
///   inherits the default scalar loop and passes vacuously.
///
/// # Returns
///
/// Nothing: returns normally when all 200 rounds matched, panics on the first divergent job.
pub fn assert_backend_batch_matches_scalar<B: SwBackend>(backend: &B) {
    // bwa's default match/mismatch, held fixed on purpose: this gate hunts lane cross-talk, and the
    // scoring sweep that hunts recurrence bugs lives in `assert_backend_matches_scalar`.
    let (a, b) = (1i8, 4i8);
    let mat = fill_scmat_vec(a, b);

    // LCG state for this gate. A different seed from the single-alignment gate, so the two sweeps
    // cover different geometries rather than replaying the same numbers.
    let mut rng_state = 0x1234_5678_9ABC_DEF0u64;
    let mut next_random = move || {
        rng_state = rng_state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        rng_state >> 33
    };

    // Many rounds, each a batch with shared scoring/band (per the contract) but varied content,
    // sizes straddling common SIMD widths (8, 16) with partial tails, and varied gap penalties.
    for round in 0..200u32 {
        // Number of jobs (lanes) in this round's batch, drawn from a fixed list that straddles the
        // SIMD widths in use: 7/8/9 and 15/16/17 give an under-full, an exactly-full and a
        // one-over-plus-tail vector for widths 8 and 16 respectively.
        let batch_size = *[1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 20, 31]
            .get((next_random() % 12) as usize)
            .unwrap();
        // Band, z-drop and query-end bonus shared by every job in this batch, as the API requires.
        // Same ranges and rationale as the single-alignment gate: `w` genuinely bites against the
        // lengths below, and `zdrop == 0` (which disables z-drop) is inside the range on purpose.
        let w = 1 + (next_random() % 30) as i32;
        let zdrop = (next_random() % 120) as i32;
        let end_bonus = (next_random() % 10) as i32;
        // Vary affine gaps too (kept >= 1 and with o+e sane).
        let o_del = 4 + (next_random() % 4) as i32;
        let e_del = 1 + (next_random() % 2) as i32;
        let o_ins = 4 + (next_random() % 4) as i32;
        let e_ins = 1 + (next_random() % 2) as i32;

        // One random query and one random target per lane, with *independently* drawn lengths (up
        // to 80) so no lane sees the same geometry as its neighbour: that is what makes it visible
        // if the longest job in a batch dictates another job's band or termination. `h0s[k]` is
        // lane k's seed score, kept >= 1 as ksw_extend2 requires. These three vectors own the
        // storage that `jobs` borrows, which is why they are materialised before `jobs` is built.
        let queries: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| {
                let qlen = 1 + (next_random() % 80) as usize;
                (0..qlen).map(|_| (next_random() % 4) as u8).collect()
            })
            .collect();
        let targets: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| {
                let tlen = 1 + (next_random() % 80) as usize;
                (0..tlen).map(|_| (next_random() % 4) as u8).collect()
            })
            .collect();
        let h0s: Vec<i32> = (0..batch_size).map(|_| 1 + (next_random() % 20) as i32).collect();

        // The batch handed to the backend: job k borrows queries[k]/targets[k] and carries h0s[k].
        let jobs: Vec<ExtendJob> = (0..batch_size)
            .map(|i| ExtendJob {
                query: &queries[i],
                target: &targets[i],
                h0: h0s[i],
            })
            .collect();

        // The backend's answers for the whole batch, one per job in job order.
        let got = backend.extend_batch(
            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
        );
        assert_eq!(got.len(), batch_size);
        for (i, got_result) in got.iter().enumerate() {
            // The authoritative answer for lane `i`, computed one job at a time: running the scalar
            // kernel in isolation is precisely what makes cross-lane leakage detectable.
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
                *got_result, expected,
                "backend {:?} extend_batch diverged at round {round} job {i}/{batch_size} (w={w} zdrop={zdrop} gaps={o_del},{e_del},{o_ins},{e_ins})",
                backend.name()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_backend_matches_kernel() {
        // The harness itself must pass for the authoritative backend (and becomes the NEON/Metal
        // acceptance gate). ScalarBackend delegates, so this also self-checks the harness runs.
        assert_backend_matches_scalar(&ScalarBackend);
        assert_backend_batch_matches_scalar(&ScalarBackend);
        assert_eq!(ScalarBackend.name(), "scalar");
    }
}

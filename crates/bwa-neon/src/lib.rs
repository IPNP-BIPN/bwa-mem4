//! NEON (Apple Silicon / AArch64) backend for the batched banded Smith-Waterman kernel (phase 9a).
//!
//! # Status: NEON int16x8 DP kernel live (step 2b-ii)
//!
//! [`NeonBackend`] implements [`bwa_extend::SwBackend`]. `extend` (single) still delegates to the
//! scalar [`bwa_extend::ksw_extend2`]; `extend_batch` runs the vectorized lane-parallel DP in
//! [`batched`]. Both the shared gate ([`bwa_extend::assert_backend_batch_matches_scalar`]) and a
//! read-sized property test prove the batched path is **byte-identical** to scalar, and the
//! `bench_batch` example measures the speedup (~1.4x on mixed lengths, ~2.1x when lengths are
//! uniform, over the scalar per-lane loop).
//!
//! # Design (following bwa-mem2 `bandedSWA` and nh13's `fg-labs/bwa-mem3`, credited in `DEPENDENCIES.md`)
//!
//! bwa-mem2 / bwa-mem3-cpp accelerate seed extension by **inter-sequence batching**: `bandedSWA`
//! packs `SIMD_WIDTH` independent alignments across NEON lanes (8 for int16, 16 for int8 on a
//! 128-bit register), one alignment per lane, using an SoA layout `[column*SIMD_WIDTH + lane]`. Each
//! lane runs the *same* integer recurrence as the scalar [`bwa_extend::ksw_extend2`], so the batched
//! result is byte-identical by construction.
//!
//! [`batched::batched_extend`] reproduces this: 8 int16 lanes in the same SoA layout, `vbslq` blendv
//! for the per-lane band mask (matching nh13's `neon_utils.h` `NEON_BLENDV`), and per-lane band
//! tightening / z-drop / max tracking. It carries the `H`/`E`/`F` recurrence with non-saturating
//! `vaddq_s16`/`vsubq_s16`, exact because a local extension's cell values are bounded well inside the
//! int16 range (nh13's kernel uses saturating ops with `MAX_SEQ_LEN16` length-binning for the same
//! guarantee). A scalar reference ([`batched::batched_extend_scalar`]) is the portable fallback.
//!
//! # Where this sits, and the order to read it in
//!
//! The aligner extends each seed into a full alignment ([`batched`]) and, for paired reads whose
//! partner did not map, realigns that partner inside an insert-size window ([`matesw`]). Both are
//! dynamic programming over a query x target grid, and both are accelerated the same way: run many
//! *independent* alignments side by side, one per SIMD lane, rather than trying to vectorize a
//! single alignment.
//!
//! 1. This file: [`NeonBackend`], a thin adapter that plugs into `bwa-extend`'s [`SwBackend`] trait.
//!    There is no arithmetic here.
//! 2. [`batched`]: seed extension. Start with its `batched_extend_scalar`, the readable reference;
//!    the vector kernels are the same code with the lane loop replaced by register lanes.
//! 3. [`matesw`]: mate rescue. Different C original, different conventions, documented in its header.
//!
//! Both modules carry their own glossary of the short C-derived names (`h`, `e`, `f`, `mj`, `beg`,
//! `qle`, ...). The parameter glossary below covers the arguments they share.
//!
//! # Parameter glossary (shared by every function in this crate)
//!
//! Every kernel here takes the same scoring block, which comes straight from bwa's `mem_opt_t`
//! (`bwamem.h`) and is passed down unchanged by the aligner. Units are alignment score points.
//!
//! - `query`, `target`: base codes, **2-bit-packed values 0..=3 for A/C/G/T and 4 for N/ambiguous**
//!   (bwa's `nst_nt4_table` encoding), one byte per base, not ASCII. `target` is the reference side.
//!   For seed extension both are already oriented so that the extension runs left-to-right from the
//!   seed end (the caller reverses the left-extension buffers), which is why the kernels only ever
//!   walk forward.
//! - `m`: the score-matrix order, always 5 for DNA (4 bases + N). `mat` must be `m * m` entries.
//! - `mat`: row-major substitution matrix, `mat[t * m + q]`. bwa builds it as `a` on the diagonal,
//!   `-b` off-diagonal among 0..=3 and `-1` on every N row/column (`bwa_fill_scmat` in `bwa.cpp`).
//!   Defaults `a = 1`, `b = 4`.
//! - `o_del`/`e_del`: deletion (reference-consuming gap) open and extend penalties, **positive
//!   magnitudes**; a deletion of length `k` costs `o_del + k * e_del`. Defaults 6 and 1.
//! - `o_ins`/`e_ins`: the same for insertions (query-consuming gaps). Defaults 6 and 1.
//! - `w`/`w0`: band half-width in cells. Only cells with `|i - j| <= w` are computed. bwa's `-w`,
//!   default 100. Each lane re-clamps it per job via [`batched::clamp_band`]-equivalent logic.
//! - `end_bonus`: bwa's `-L` clipping penalty, used here only inside the band clamp (it enlarges the
//!   score a full-length alignment could reach, hence the maximum useful gap length). Default 5.
//! - `zdrop`: bwa's `-d`, default 100. The DP abandons a row once the running score has fallen
//!   `zdrop` below the best seen, after correcting for the diagonal drift (see the z-drop block in
//!   [`batched::batched_extend_scalar`]). `zdrop <= 0` disables the test.
//! - `h0`: the score already accumulated by the seed being extended (bwa passes `seedlen * a`).
//!   `ksw_extend2` asserts `h0 > 0` (`ksw.cpp:437`); it seeds cell `H(-1,-1)` and the whole first row.
//!
//! All lengths are in bases; all returned coordinates (`qle`, `tle`, `gtle`, `te`, `qe`, ...) are
//! 0-based *lengths consumed* or end offsets exactly as the C returns them, so they can be compared
//! byte-for-byte against bwa-mem2's.

use bwa_extend::{ksw_extend2, ExtendJob, ExtendResult, SwBackend};

/// Seed extension: the lane-parallel `ksw_extend2` equivalent plus its scalar reference. Private
/// because callers reach it through [`NeonBackend`]'s [`SwBackend`] impl, never directly.
mod batched;
/// Mate rescue: the lane-parallel `ksw_align2` equivalent. Public because the paired-end code calls
/// it directly (there is no `SwBackend` method for whole-read local alignment).
pub mod matesw;

/// Mate-rescue entry points, re-exported at the crate root so `bwa-mem`'s paired-end path can say
/// `bwa_neon::batched_ksw_align2` without naming the module. [`batched_ksw_align2`] takes a slice of
/// [`KswJob`] (one unmapped mate against one reference window each) and returns one alignment per
/// job. Note this kernel opens gaps from `H`, not from `M` like [`batched`]: that asymmetry is
/// deliberate and mirrors the two different C originals (`ksw_u8` vs `ksw_extend2`).
pub use matesw::{batched_ksw_align2, KswJob};

/// The NEON seed-extension backend. See the module docs: it delegates to the scalar kernel today
/// and is the drop-in point for the lane-parallel NEON DP (phase 9a).
#[derive(Debug, Default, Clone, Copy)]
pub struct NeonBackend;

impl SwBackend for NeonBackend {
    /// # Returns
    ///
    /// The literal `"neon"`, this backend's identifier in the [`SwBackend`] registry. Used by the
    /// CLI's backend-selection message and by the acceptance gates' panic text, never by any
    /// dispatch decision (dispatch is by CPU feature detection inside [`batched`]), so it is a
    /// label only. It is asserted in this crate's tests, so changing it breaks them.
    fn name(&self) -> &'static str {
        "neon"
    }

    /// One seed extension: the banded local DP of `ksw_extend2` (`ksw.cpp:432`) for a single
    /// query/target pair. See the crate-level parameter glossary for every argument. Returns
    /// `{score, qle, tle, gtle, gscore, max_off}`: the best local score and where it ended
    /// (`qle`/`tle` = query/target bases consumed), plus the best *global* (query-exhausting) score
    /// `gscore` and its target end `gtle`, which is what lets bwa decide between soft-clipping and
    /// extending to the read end. `max_off` is the largest `|i - j|` the optimum path visited, which
    /// the caller uses to decide whether the band was wide enough and the alignment must be redone.
    ///
    /// There is no per-pair NEON win to have here: a single alignment has no lanes to fill, so this
    /// deliberately calls the scalar kernel and all the vectorization lives in [`Self::extend_batch`].
    ///
    /// # Parameters
    ///
    /// Every argument is forwarded verbatim to [`ksw_extend2`]; the crate-level glossary is the
    /// authority, repeated here in argument order so the signature can be read on its own.
    ///
    /// - `query`: the read side being extended, one byte per base in 2-bit code (`0=A 1=C 2=G 3=T`,
    ///   `4=N`), not ASCII. Supplied by the aligner, already oriented so the extension runs forward
    ///   from the seed end. Length in bases; may be empty only if the caller tolerates the trivial
    ///   result. Indexes the *column* axis of the DP.
    /// - `target`: the reference stretch to extend into, same encoding and same orientation rule.
    ///   Indexes the *row* axis. Usually a little longer than `query` to leave room for gaps.
    /// - `m`: order of the score matrix, always 5 for DNA (4 bases plus N). Fixes `mat.len()`.
    /// - `mat`: row-major `m * m` substitution matrix, `mat[t * m + q]` scoring target base `t`
    ///   against query base `q`, in score points. Built once per run by `bwa_fill_scmat`.
    /// - `o_del`, `e_del`: deletion (gap in the query, CIGAR `D`, consumes target) open and extend
    ///   penalties as **positive magnitudes**, score points. A `k`-base deletion costs
    ///   `o_del + k * e_del`. Defaults 6 and 1; `e_del >= 1` is required because the band clamp
    ///   divides by it.
    /// - `o_ins`, `e_ins`: the same for insertions (gap in the target, CIGAR `I`, consumes query).
    /// - `w`: band half-width in cells; only cells with `|i - j| <= w` are evaluated. bwa's `-w`,
    ///   default 100, then clamped downward per call from the score geometry.
    /// - `end_bonus`: bwa's `-L`, score points awarded for reaching the query end rather than
    ///   soft-clipping. Default 5. Affects both the band clamp and the local-vs-global choice.
    /// - `zdrop`: bwa's `-d`, default 100, score points. The DP abandons the extension once the
    ///   running score has fallen `zdrop` below its best, corrected for diagonal drift.
    ///   `zdrop <= 0` disables the test.
    /// - `h0`: score points the seed has already earned, the DP's starting value (bwa passes
    ///   `seedlen * a`). Must be `> 0`: `ksw_extend2` asserts it.
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
        // TODO(phase9a): NEON lane-parallel DP. Delegates to scalar until byte-identity is verified.
        ksw_extend2(
            query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
        )
    }

    /// Many independent seed extensions at once. `out[i]` is exactly what [`Self::extend`] would
    /// return for `jobs[i]`: the batching is *inter-sequence* (one alignment per SIMD lane), never
    /// intra-sequence, so no job can observe another and the results cannot depend on the batch
    /// composition. That property is what makes byte-identity with bwa-mem2 provable rather than
    /// merely tested: the scoring arguments are shared across the batch (they are per-run options),
    /// only `query`/`target`/`h0` vary per job.
    ///
    /// # Parameters
    ///
    /// - `jobs`: the per-alignment inputs, one [`ExtendJob`] per extension: its `query`, `target`
    ///   (2-bit base codes as in [`Self::extend`]) and `h0` (the seed's earned score). Any length,
    ///   including 0 and lengths that do not fill a SIMD width: the kernel pads the final partial
    ///   vector with inert lanes. Jobs need not share lengths; the kernel bins them by length so a
    ///   long job does not force short ones through a wider element type.
    /// - `m`, `mat`, `o_del`, `e_del`, `o_ins`, `e_ins`, `w`, `end_bonus`, `zdrop`: exactly as in
    ///   [`Self::extend`], with the same units, defaults and preconditions. They are per-run options,
    ///   so they are passed once for the whole batch rather than per job. Note `h0` is *not* here: it
    ///   is per job and lives in [`ExtendJob`].
    ///
    /// # Returns
    ///
    /// One [`ExtendResult`] per job, in input order, so `out[k]` is what `extend(jobs[k]...)` would
    /// have returned, field for field (`score`, `qle`, `tle`, `gtle`, `gscore`, `max_off`). The
    /// length always equals `jobs.len()`.
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
        // Step 2b-ii: NEON int16x8 lane-parallel DP (byte-identical to ksw_extend2 per job), with a
        // scalar fallback for non-aarch64 or out-of-int16-range batches.
        batched::batched_extend(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::{assert_backend_batch_matches_scalar, assert_backend_matches_scalar};

    /// The two shared acceptance gates from `bwa-extend`, run against this backend. They take no
    /// arguments: each generates its own fixed-seed sweep and panics on the first field-level
    /// divergence from `ksw_extend2`.
    #[test]
    fn neon_backend_matches_scalar() {
        // The shared gate (qlen/tlen <= 80). Exercises the NEON int16 kernel on aarch64.
        assert_backend_matches_scalar(&NeonBackend);
        assert_backend_batch_matches_scalar(&NeonBackend);
        assert_eq!(NeonBackend.name(), "neon");
    }

    /// Byte-identity at realistic short-read sizes: query up to a full 150 bp read side and
    /// reference windows up to ~300 bp, i.e. the range the aligner pipeline actually feeds
    /// `extend_batch`. Confirms the int16 kernel stays exact well above the shared gate's 80 bp.
    #[test]
    fn neon_batch_matches_scalar_read_sized() {
        use bwa_extend::{ksw_extend2, ExtendJob};
        // bwa's default scoring: `a` = +1 per match, `b` = 4 (applied as -4) per mismatch.
        let (a, b) = (1i8, 4i8);
        // The 5x5 row-major substitution matrix, `mat[t * 5 + q]`, built exactly as `bwa_fill_scmat`
        // does: `a` on the diagonal, `-b` off it, and a flat -1 on the whole N row and N column.
        // `k` is the write cursor walking the 25 entries in row-major order.
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        // Rows 0..4, the concrete bases A/C/G/T: four scored entries then the row's N column.
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        // Row 4, the N row: -1 against every base including another N.
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }

        // Fixed-seed 64-bit LCG (the MMIX multiplier/increment pair) so this sweep is identical on
        // every run and machine: any failure is replayable from the printed round index. `state` is
        // the generator's internal word; `next()` advances it and returns the top 31 bits, whose
        // low bits are well mixed (the raw low bits of an LCG are not).
        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        for round in 0..120u32 {
            // Number of independent alignments packed into one `extend_batch` call. The set
            // straddles the SIMD widths in play (8 int16 lanes, 16 int8 lanes) with under-full,
            // exactly-full and one-over sizes, which is what exercises partial-vector handling.
            let batch_size = *[1usize, 5, 8, 9, 16, 17, 24, 40]
                .get((next() % 8) as usize)
                .unwrap();
            // Band half-width in cells, 50..=169: wide enough that the DP is not trivially clipped
            // at these read-scale lengths, narrow enough that the band still bites.
            let w = 50 + (next() % 120) as i32;
            // Score points the running score may fall below its best before the row is abandoned,
            // 0..=199. 0 is included deliberately: it disables the z-drop branch entirely.
            let zdrop = (next() % 200) as i32;
            // bwa's `-L` end bonus, 0..=11 score points.
            let end_bonus = (next() % 12) as i32;
            // Affine gap penalties as positive magnitudes: opens 4..=7, extends 1..=2. Deletion and
            // insertion are drawn independently so they usually differ, which is what makes an
            // E-vs-F (deletion-vs-insertion) mix-up in a lane observable at all.
            let o_del = 4 + (next() % 4) as i32;
            let e_del = 1 + (next() % 2) as i32;
            let o_ins = 4 + (next() % 4) as i32;
            let e_ins = 1 + (next() % 2) as i32;

            // Correlated query/target so the DP runs deep (not an all-mismatch early z-drop).
            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            // Per-job inputs, index-aligned across the three vectors: job `i` is
            // (queries[i], targets[i], h0s[i]). Owned separately from `jobs` below because
            // `ExtendJob` only borrows.
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch_size {
                let qlen = 60 + (next() % 200) as usize; // up to ~260
                                                         // The read side: uniform random 2-bit base codes 0..=3 (never 4/N here).
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                // Reference window, 0..=59 bases longer than the query so gaps have room.
                let tlen = qlen + (next() % 60) as usize;
                let mut t: Vec<u8> = Vec::with_capacity(tlen);
                // Cursor into `q` for the mutation walk. Loop invariant at the top of each
                // iteration: `t` is a ~96%-faithful copy of `q[..qi]`, so target position
                // `t.len()` is the mutated image of query position `qi`, and the two stay near the
                // main diagonal.
                let mut qi = 0usize;
                while t.len() < tlen {
                    // 96% of the time copy the query base through (a match); otherwise emit a
                    // random base and, half the time, also skip the query base, which yields a
                    // substitution or an insertion respectively.
                    if qi < q.len() && next() % 100 >= 4 {
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
                // Seed score already earned, 20..=49 points; must be > 0 for `ksw_extend2`.
                h0s.push(20 + (next() % 30) as i32);
            }
            let jobs: Vec<ExtendJob> = (0..batch_size)
                .map(|i| ExtendJob {
                    query: &queries[i],
                    target: &targets[i],
                    h0: h0s[i],
                })
                .collect();

            // The batched NEON results, one per job in input order.
            let got = NeonBackend.extend_batch(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
            for (i, g) in got.iter().enumerate() {
                // The authoritative scalar answer for this job alone, computed with no batch
                // context at all: comparing against it is what proves the lanes stayed independent.
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
                    "NEON diverged at round {round} job {i}/{batch_size} qlen={} tlen={} (w={w})",
                    queries[i].len(),
                    targets[i].len(),
                );
            }
        }
    }

    /// Exercise the ungapped diagonal HIT fast path directly: many perfect diagonals (which trigger
    /// the closed form) plus near-diagonals (0-2 substitutions/indels) and short targets, asserting
    /// the batched NEON result equals `ksw_extend2` for every job. Guards the HIT closed form against
    /// any divergence from the real DP on the exact cases it fires.
    #[test]
    fn neon_ungapped_hit_matches_scalar() {
        use bwa_extend::{ksw_extend2, ExtendJob};
        // bwa's default scoring, and the same `bwa_fill_scmat` 5x5 matrix as the test above:
        // `mat[t * 5 + q]`, `a` on the diagonal, `-b` off it, -1 on the N row and column. `k` is
        // the row-major write cursor. The standard shape matters here specifically: the ungapped
        // fast path only fires when the matrix is recognised as uniform DNA.
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

        // Independent fixed seed from the test above, so the two sweeps explore different cases.
        let mut state = 0x51ED_5EED_D1A6_0000u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        for round in 0..300u32 {
            // Alignments per `extend_batch` call, again straddling the 8 and 16 lane widths.
            let batch = *[1usize, 4, 8, 13, 16, 24]
                .get((next() % 6) as usize)
                .unwrap();
            // Band half-width 1..=150. Starting at 1 matters here: a band of 1 is the regime where
            // the DP is closest to a bare diagonal, i.e. exactly what the fast path claims to
            // compute in closed form.
            let w = 1 + (next() % 150) as i32;
            // z-drop threshold 0..=199 (0 disables) and `-L` end bonus 0..=11, both score points.
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            // Gap penalties pinned to bwa's defaults: this test targets the ungapped path, and the
            // gap-penalty sweep lives in the read-sized test above.
            let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);

            // Per-job inputs, index-aligned: job `i` is (queries[i], targets[i], h0s[i]).
            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch {
                // 1..=200 bases; length 1 is included since the fast path must handle it too.
                let qlen = 1 + (next() % 200) as usize;
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                // Target = query diagonal, then a mode: perfect / longer / shorter / few edits / N.
                // Which of the five target shapes below to build, 0..=4.
                let mode = next() % 5;
                // Bases of random trailing target appended past the query's end, 0..=39: the
                // reference window normally overruns the query, and the tail must not change the
                // best local alignment.
                let extra = (next() % 40) as usize;
                let mut t: Vec<u8> = q.clone();
                match mode {
                    0 => t.extend((0..extra).map(|_| (next() % 4) as u8)), // perfect, longer target
                    1 => {}                                                // perfect, equal length
                    2 => {
                        t.truncate(qlen.saturating_sub(1 + (next() % 3) as usize)); // shorter target
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                    3 => {
                        // 1-2 substitutions
                        for _ in 0..(1 + next() % 2) {
                            if !t.is_empty() {
                                // Target offset to corrupt; the base there is bumped by 1..=3
                                // modulo 4, which is guaranteed to change it to a different base.
                                let p = (next() as usize) % t.len();
                                t[p] = (t[p] + 1 + (next() % 3) as u8) % 4;
                            }
                        }
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                    _ => {
                        // inject an N somewhere
                        if !t.is_empty() {
                            // Target offset that becomes code 4 (N). This is the one generator
                            // here that produces an ambiguous base, so it is the only coverage of
                            // the matrix's N row on the fast path.
                            let p = (next() as usize) % t.len();
                            t[p] = 4;
                        }
                        t.extend((0..extra).map(|_| (next() % 4) as u8));
                    }
                }
                queries.push(q);
                targets.push(t);
                // Seed score already earned, 1..=40 points (must be > 0).
                h0s.push(1 + (next() % 40) as i32);
            }
            let jobs: Vec<ExtendJob> = (0..batch)
                .map(|i| ExtendJob {
                    query: &queries[i],
                    target: &targets[i],
                    h0: h0s[i],
                })
                .collect();
            // The batched NEON results, one per job in input order.
            let got = NeonBackend.extend_batch(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
            for (i, g) in got.iter().enumerate() {
                // The authoritative scalar answer for this job alone, computed with no batch
                // context at all: comparing against it is what proves the lanes stayed independent.
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
                    "ungapped HIT diverged at round {round} job {i} qlen={} tlen={} h0={}",
                    queries[i].len(),
                    targets[i].len(),
                    h0s[i],
                );
            }
        }
    }
}

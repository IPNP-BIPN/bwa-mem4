//! Lane-batched banded Smith-Waterman seed extension (phase 9a).
//!
//! Processes several independent alignments in lockstep: a shared target-row loop `i` and a shared
//! query loop `j` over the union band, each lane masked to its own band `[beg, end)` and its own
//! termination (row all-zero or z-drop). Two implementations, both **byte-identical** to
//! [`bwa_extend::ksw_extend2`] run per lane:
//!
//! - [`batched_extend_scalar`]: the portable reference (scalar per-lane cell arithmetic, step 2b-i).
//! - The SIMD kernels: one `define_sw_kernel!` macro, instantiated per ISA + lane width, so the
//!   variants cannot drift. **NEON** (`mod neon`, aarch64): u8 x16 / i16 x8. **AVX2** (`mod avx2`,
//!   x86_64): u8 x32 / i16 x16 (256-bit, twice NEON's width). All are exact because a local
//!   extension's `H`/`E`/`F` stay in `[0, minval]` with `minval = h0 + min(len)*a`, so the per-job
//!   bound decides u8 (`minval < 256`) vs i16 (`< 32768`) vs the scalar fallback.
//!
//! [`batched_extend`] dispatches to [`simd_dispatch`] when the ISA feature is present (NEON on
//! aarch64, AVX2 on x86_64): the ungapped-diagonal HIT fast path, then bwa-mem2's
//! `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` length-binning (the u8 path packs twice the lanes of i16). AVX-512
//! (64 u8 lanes) is a follow-up (needs the k-mask blend variant). The AVX2 kernels are validated
//! byte-identical to `ksw_extend2` by a force-run test executed under Rosetta (`avx2_verify`).
//!
//! # Order to read this file in
//!
//! 1. [`batched_extend`], the entry point: it only picks a path.
//! 2. [`batched_extend_scalar`], the readable reference. It is `ksw_extend2` with a lane loop `l`
//!    wrapped around every statement. Understand this one and the SIMD kernels follow, because they
//!    are the same code with the `l` loop replaced by the lanes of a vector register.
//! 3. [`ungapped_hit`] and [`simd_dispatch`] / `dispatch_bins`, the routing: which alignments can
//!    skip the DP entirely, and which integer width the rest run at.
//! 4. `define_sw_kernel!`, the macro holding the actual vector kernel, instantiated four times at
//!    the bottom (NEON u8/i16, AVX2 u8/i16).
//!
//! # Glossary: the short names kept from the C, in plain language
//!
//! These mirror `reference/bwa-mem2/src/ksw.cpp` on purpose. Diffing this file against the C line by
//! line is how parity bugs get found here, so the names stay.
//!
//! | name | plain language |
//! |---|---|
//! | `i` | index of the current **target** (reference) base: the DP row |
//! | `j` | index of the current **query** (read) base: the DP column |
//! | `l` | which lane, i.e. which of the batched alignments (scalar path only) |
//! | `H` (`h`, `h1`, `h1_v`) | best score of any alignment ending at this cell; `h1` is the cell to the left, carried right |
//! | `M` (`big_m`, `bigm_v`, `m11`) | best score ending here **on the diagonal**, i.e. as an aligned pair (CIGAR `M`) |
//! | `E` (`e`, `eh_e`, `e_v`) | best score ending here with a **deletion** open: a gap in the query, CIGAR `D` |
//! | `F` (`f`, `f_v`) | best score ending here with an **insertion** open: a gap in the target, CIGAR `I` |
//! | `h0` | score the seed already earned; the DP starts from it |
//! | `mj` | column where this row's best `H` was found |
//! | `oe_del` / `oe_ins` | cost of a gap's *first* base: open + one extend |
//! | `beg` / `end` | half-open range of columns still worth evaluating, per lane |
//! | `qle` / `tle` | query / target bases consumed by the best **local** alignment |
//! | `gtle` / `gscore` | target length and score of the best **query-exhausting** alignment |
//! | `max_off` | furthest any row's best cell strayed from the main diagonal |
//! | `zdrop` | how far the score may fall below its best before the DP gives up |
//!
//! `m` is overloaded exactly as in the C: as a parameter it is the alphabet size (always 5), while
//! `M` inside the recurrence is the diagonal score. This file writes the latter `big_m`/`bigm_v`.
//! A `_v` suffix always means "the vector register holding one of these per lane".
//!
//! # The recurrence, once, for the whole file
//!
//! Every kernel below computes the same three affine-gap Smith-Waterman arrays, over a target row
//! index `i` and a query column index `j` (`ksw.cpp:479-501`, the comment block bwa itself keeps):
//!
//! ```text
//!   M(i,j) = H(i-1,j-1) + S(target[i], query[j])   , clamped to 0 when H(i-1,j-1) == 0
//!   H(i,j) = max( M(i,j), E(i,j), F(i,j) )
//!   E(i+1,j) = max( E(i,j) - e_del , M(i,j) - o_del - e_del , 0 )   // gap in the query (deletion)
//!   F(i,j+1) = max( F(i,j) - e_ins , M(i,j) - o_ins - e_ins , 0 )   // gap in the target (insertion)
//! ```
//!
//! Variable naming used consistently in the code: `big_m` / `bigm_v` is `M`, `eh_h[j]` holds
//! `H(i-1,j-1)` on entry to the cell and is overwritten with `H(i,j-1)`, `eh_e[j]` holds `E(i,j)`,
//! `f` / `f_v` is the carried `F` (it lives in a register because `F(i,j+1)` depends only on the
//! cell to its left), and `h1` / `h1_v` carries `H(i,j)` into the next column.
//!
//! Two details that are load-bearing for byte-identity and easy to get wrong:
//!
//! 1. **Gaps open from `M`, not from `H`.** `E` and `F` above subtract from `M`, not from
//!    `max(M,E,F)`. bwa spells out why at `ksw.cpp:487`: "separating H and M to disallow a cigar
//!    like `100M3I3D20M`". bwa-mem2's vector kernel does the same (`bandedSWA.cpp:338` and `:342`
//!    subtract `oe_ins256`/`oe_del256` from `m11`, never from `h11`). Opening from `H` still
//!    produces a valid-looking alignment but silently inflates `gscore` inside repeats, because an
//!    insertion immediately followed by a deletion becomes free.
//! 2. **The `M(i,j) = 0` when `H(i-1,j-1) == 0` clamp is the local-alignment restart.** A dead
//!    diagonal must not be resurrected by a negative-scoring match; `ksw.cpp:487`'s `M = M? M + q[j] : 0`
//!    is exactly this and the vector form is a compare-against-zero plus a select.

use bwa_extend::{ExtendJob, ExtendResult};

/// Lanes processed per group. 8 = one NEON `int16x8` register (the vectorized path); the scalar
/// reference works for any value.
///
/// This constant only governs [`batched_extend_scalar`]. The SIMD kernels get their lane count from
/// the `define_sw_kernel!` invocation instead (16 for NEON u8, 8 for NEON i16, 32/16 for AVX2),
/// because there the number is dictated by `register width / element width` and must match the
/// intrinsics being passed in.
const LANES: usize = 8;

/// Batched banded local extension. Returns one [`ExtendResult`] per job, each equal to
/// [`bwa_extend::ksw_extend2`] on that job. Dispatches to the NEON kernel where available.
///
/// See the crate-level parameter glossary for `m`/`mat`/`o_*`/`e_*`/`w0`/`end_bonus`/`zdrop`; each
/// job carries its own `query`, `target` and `h0`. Jobs may have wildly different lengths: the
/// kernels run every lane for `max(tlen)` rows and mask the finished ones, so a batch of similar
/// lengths is faster but no batch is ever wrong.
///
/// # Parameters
///
/// - `jobs`: the batch, supplied by the caller in `bwa-mem`'s seed-extension loop. Each entry holds
///   `query` and `target` as 2-bit base codes (`0=A 1=C 2=G 3=T`, `4=N`, one byte per base, **not**
///   ASCII) plus `h0`, the score the seed already earned, which is where this extension's DP starts.
///   Any length is accepted, including an empty batch; jobs are independent of one another.
/// - `m`: alphabet size / order of the score matrix. Always 5 for DNA (4 bases + N).
/// - `mat`: row-major `m * m` substitution matrix; `mat[t * m + q]` scores target base `t` against
///   query base `q`. Must have at least `m * m` entries. The SIMD kernels additionally require the
///   uniform bwa shape (see [`is_uniform_dna`]); anything else silently routes to the scalar path.
/// - `o_del` / `e_del`: deletion (gap in the query, CIGAR `D`) open and extend penalties as
///   **positive magnitudes**; a deletion of length `k` costs `o_del + k * e_del`. bwa defaults 6, 1.
/// - `o_ins` / `e_ins`: the same for an insertion (gap in the target, CIGAR `I`). Defaults 6, 1.
///   `e_ins`/`e_del` must be non-zero: [`clamp_band`] divides by them.
/// - `w0`: the user's `-w`, band half-width in cells, before the per-job clamp in [`clamp_band`].
///   Only cells with `|i - j| <= w` are evaluated. Default 100.
/// - `end_bonus`: bwa's `-L` clipping penalty. Used here *only* to widen the band clamp, never added
///   to a score. Default 5.
/// - `zdrop`: bwa's `-d`, default 100. A lane gives up once its score has fallen `zdrop` below its
///   own best (after the diagonal-drift correction). `zdrop <= 0` disables the test entirely.
///
/// # Returns
///
/// One [`ExtendResult`] per job, in input order (`out[k]` corresponds to `jobs[k]`), each holding
/// `score`/`qle`/`tle` for the best local alignment, `gscore`/`gtle` for the best query-exhausting
/// one, and `max_off`, the largest diagonal excursion of any row's best cell.
#[allow(clippy::too_many_arguments)]
pub fn batched_extend(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return simd_dispatch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return simd_dispatch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
    }
    batched_extend_scalar(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    )
}

#[cfg(target_arch = "x86_64")]
use avx2::{batched_extend_avx2_i16 as sw_kernel_i16, batched_extend_avx2_u8 as sw_kernel_u8};
/// Element type / kernel alias for the current SIMD ISA: NEON on aarch64, AVX2 on x86_64.
#[cfg(target_arch = "aarch64")]
use neon::{batched_extend_neon_i16 as sw_kernel_i16, batched_extend_neon_u8 as sw_kernel_u8};

/// True if `mat` is the standard bwa `m x m` DNA matrix: a constant `a` on the diagonal, a constant
/// `mm` off-diagonal among the 4 concrete bases, and a constant `npen` on every ambiguous row/column
/// (index `m-1`). The NEON kernels score with a vector compare that relies on exactly this shape;
/// anything else falls back to the scalar reference (which reads `mat` directly).
///
/// # Parameters
///
/// - `mat`: the candidate row-major substitution matrix, as handed to [`batched_extend`]. Shorter
///   than `m * m` entries is not an error here, it just answers `false`.
/// - `m`: claimed alphabet size (5 for DNA). `m < 2` answers `false`, since the "off-diagonal
///   among concrete bases" and "ambiguous row/column" categories would not both exist.
///
/// # Returns
///
/// `true` only if every entry equals the value its category demands: `mat[0]` on the diagonal,
/// `mat[1]` off-diagonal, `mat[m-1]` on row or column `m-1`. `true` is the precondition for the
/// SIMD kernels, which reconstruct the score from those three numbers instead of loading `mat`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn is_uniform_dna(mat: &[i8], m: usize) -> bool {
    if m < 2 || mat.len() < m * m {
        return false;
    }
    // The three numbers the whole matrix must be built from, read out of the cells that define
    // them: `a` = match score (diagonal, positive), `mm` = mismatch score (off-diagonal, negative),
    // `npen` = the ambiguous-base score on row/column m-1 (bwa uses -1). Everything else is then
    // checked against these rather than against hardcoded constants.
    let a = mat[0];
    let mm = mat[1];
    let npen = mat[m - 1];
    for i in 0..m {
        for j in 0..m {
            let v = mat[i * m + j];
            // The value this cell would hold if the matrix really has the uniform bwa shape.
            let want = if i == m - 1 || j == m - 1 {
                npen
            } else if i == j {
                a
            } else {
                mm
            };
            if v != want {
                return false;
            }
        }
    }
    true
}

/// bwa-mem2's exact per-pair score ceiling (`sort_classify` in `bwamem.cpp`):
/// `minval = h0 + min(len1, len2) * score_a`. Because a local extension matches at most
/// `min(qlen, tlen)` bases and only *loses* score to mismatches/gaps, every `H`/`E`/`F` value (and
/// the intermediate `M = m + score` that becomes an `H`) stays in `[<0-ish>, minval]`. So
/// `minval < MAX_SEQ_LEN8 (128)` proves the signed-int8 kernel is exact (values fit `[0,127]`), and
/// `< MAX_SEQ_LEN16 (32768)` proves the int16 kernel is exact. This matches the oracle's own binning,
/// so the classification never changes results — only which (exact) kernel runs.
///
/// # Parameters
///
/// - `job`: the alignment being classified. Only `h0` and the two sequence *lengths* are read; the
///   base codes are irrelevant to a worst-case bound.
/// - `max_sc`: the largest entry of `mat` (the match score `a` for the standard DNA matrix), in
///   score units. Supplied by [`dispatch_bins`], which scans `mat` once per batch. Negative values
///   are clamped to 0 by the `.max(0)`, so a degenerate all-negative matrix cannot shrink the bound
///   below `h0`.
///
/// # Returns
///
/// `minval`, the inclusive upper bound on every DP value this job can produce, in score units.
/// [`dispatch_bins`] compares it against 256 (u8 kernel) and 32768 (i16 kernel).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn cell_bound(job: &ExtendJob, max_sc: i32) -> i32 {
    job.h0 + (job.query.len().min(job.target.len()) as i32) * max_sc.max(0)
}

/// Ungapped diagonal fast path (nh13's `ungapped_analyze`, FP_STATUS_HIT; `bwamem.cpp`): when a
/// seed extension stays on the diagonal with few enough mismatches, the ungapped alignment is
/// *provably* optimal and the banded Smith-Waterman is skipped.
///
/// The proof: any gapped alternative sitting `B` off the diagonal needs at least `B` gaps, costing
/// `B * (o_min + e_min)`, while its score is capped at `min(len1, len2) * a`. So it can only beat an
/// ungapped walk carrying `X` mismatches while `X <= o_min / (a + b - e_min)` -- which is
/// `6 / (1 + 4 - 1) = 1` for bwa's defaults, i.e. a single mismatch. `x_threshold` carries that
/// bound; above it we fall through to the kernel (bwa-mem4-cpp instead narrows the band, which buys
/// us nothing: our SIMD kernel walks every column under a mask, so a tighter band does not shorten
/// the inner loop).
///
/// With no mismatch the walk is deterministic and collapses to `score = gscore = h0 + n*a`,
/// `qle = tle = gtle = n`. With 1..=x_threshold mismatches it needs the scalar walk below, whose
/// tie-break is `ksw_extend2`'s: `if (m > max)` is **strict**, so a row that only ties the running
/// max leaves `max_i` alone and the extension reports `qle = 0`. (nh13's walk uses `>=` instead,
/// mirroring bandedSWA's `maxRS`, which does move on ties -- each fork must mirror its own kernel.
/// Ours is ksw_extend2-derived, and the property test below pins that.)
///
/// Z-drop has to be modelled too. ksw breaks the row loop once the score falls `zdrop` below the
/// running max; on a diagonal `i - max_i == mj - max_j`, so its test collapses to `max - m > zdrop`.
/// A mismatch-free walk never descends (`max == m`) and is immune, which is why the closed form
/// needs no check -- but a mismatch drops the score by `b`, so a small `zdrop` would make ksw stop
/// where this walk carries on. bwa's default zdrop is 100 against a mismatch penalty of 4, so this
/// never fires in production; it fires under the property test's randomized zdrop, which is the
/// point of testing it.
///
/// `tle == qle` because the walk never leaves the diagonal, and `max_off == 0` for the same reason.
/// The `h0 > 0` guard preserves parity with the DP: at `h0 == 0` the `score == 0` early-skip inhibits
/// every update, leaving the running best at 0, where the closed form would claim `n*a`.
///
/// Byte-identical to [`bwa_extend::ksw_extend2`] on the jobs it accepts (gated by the property test).
/// `a` is the match score (`mat[0]`), `b` the mismatch penalty magnitude.
///
/// # Parameters
///
/// - `job`: the single alignment to try in closed form. `query` and `target` are 2-bit base codes;
///   `target` must be at least as long as `query` (a shorter target declines). `h0` must be `> 0`
///   (see the parity note above).
/// - `a`: match score, a positive magnitude, `mat[0]` of the uniform matrix.
/// - `b`: mismatch penalty as a **positive magnitude**, i.e. `-mat[1]`. The walk subtracts it.
/// - `x_threshold`: the largest number of mismatches an ungapped diagonal may carry and still be
///   provably optimal, computed once per batch by [`simd_dispatch`]. `1` at bwa's defaults. A
///   negative value (degenerate scoring) makes this function always decline.
/// - `zdrop`: bwa's `-d`, in score units, same meaning as everywhere else. Used only to detect the
///   case where the real DP would have stopped early, which this walk declines rather than models.
///
/// # Returns
///
/// `Some(result)` with exactly the fields `ksw_extend2` would have produced, or `None` meaning
/// "cannot prove it, run the real DP" (too many mismatches, an N base, a walk longer than the
/// 128-column mismatch bitmap, a z-drop that would have fired, or a diagonal that died).
fn ungapped_hit(
    job: &ExtendJob,
    a: i32,
    b: i32,
    x_threshold: i32,
    zdrop: i32,
) -> Option<ExtendResult> {
    // Number of cells on the diagonal walk = the whole query, since an ungapped extension consumes
    // one target base per query base. Also the `n` in the closed forms below.
    let diag_len = job.query.len();
    if diag_len == 0 || job.h0 <= 0 || job.target.len() < diag_len || x_threshold < 0 {
        return None;
    }
    // Widest diagonal this fast path can describe, because the mismatch positions are recorded as
    // set bits of a single `u128` (nh13 calls the same limit FP_N_MAX). Longer walks fall through
    // to the real DP.
    const MISMATCH_BITMAP_BITS: usize = 128;
    // Walk the diagonal once, recording mismatch columns. Any ambiguous base bails: N scores -1
    // rather than -b, which the walk below does not model.
    // `mismatch_bits` bit `c` is set iff diagonal position `c` is a mismatch (so only positions
    // `< 128` are representable); `total_mis` counts them, including any beyond the bitmap, and is
    // what gets tested against `x_threshold`.
    let mut mismatch_bits: u128 = 0;
    let mut total_mis = 0i32;
    for col in 0..diag_len {
        // SAFETY of indexing: target.len() >= diag_len checked above.
        let (query_base, target_base) = (job.query[col], job.target[col]);
        if query_base >= 4 || target_base >= 4 {
            return None;
        }
        if query_base != target_base {
            total_mis += 1;
            if total_mis > x_threshold {
                return None; // a gapped alignment could win: let the kernel decide
            }
            if col >= MISMATCH_BITMAP_BITS {
                return None; // past the mismatch bitmap
            }
            mismatch_bits |= 1u128 << col;
        }
    }
    if total_mis == 0 {
        // A mismatch-free walk only ever climbs, so its last cell is also its best: local and
        // query-exhausting alignments coincide and both consume all `diag_len` bases.
        let score = job.h0 + diag_len as i32 * a;
        return Some(ExtendResult {
            score,
            qle: diag_len as i32,
            tle: diag_len as i32,
            gtle: diag_len as i32,
            gscore: score,
            max_off: 0,
        });
    }
    if diag_len > MISMATCH_BITMAP_BITS {
        return None; // bitmap holds only the first 128 columns
    }
    // 1..=x_threshold mismatches: the walk is cheap but no longer closed-form. `running` is the
    // score at the current column, `best_score`/`best_end` the best cell seen and where it was
    // (these are ksw_extend2's `max`/`max_i`, tracked along the single diagonal).
    let (mut running, mut best_score, mut best_end) = (job.h0, job.h0, 0i32);
    for col in 0..diag_len {
        if running == 0 {
            continue; // local SW: once the diagonal dies it stays dead (no e/f restart)
        }
        if mismatch_bits >> col & 1 == 0 {
            running += a;
        } else {
            running -= b;
            if running < 0 {
                running = 0;
            }
        }
        if running > best_score {
            best_score = running;
            best_end = col as i32 + 1;
        } else if zdrop > 0 && best_score - running > zdrop {
            return None; // ksw would break the row loop here; this walk does not model that
        }
    }
    if running == 0 {
        // The diagonal died. `gscore`/`gtle` then depend on cells this walk does not model (ksw
        // breaks the row loop at `m == 0`, leaving max_ie on the last live row), so decline the HIT
        // rather than report a field we cannot prove. nh13 returns gtle = N here and relies on the
        // caller ignoring it when gscore <= 0; we would rather not ship a struct that lies.
        return None;
    }
    Some(ExtendResult {
        score: best_score,
        qle: best_end,
        tle: best_end,
        gtle: diag_len as i32,
        gscore: running,
        max_off: 0,
    })
}

/// SIMD dispatch (NEON on aarch64, AVX2 on x86_64): ungapped-HIT fast path, then bin each remaining
/// job by length into the u8 (16/32 lanes) / i16 (8/16 lanes) / scalar kernel and scatter back. This
/// is bwa-mem2's `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` binning. Result-preserving (each job's extension
/// depends only on its own inputs).
///
/// # Parameters
///
/// Identical in meaning, units and provenance to [`batched_extend`]'s, passed straight through:
/// `jobs`, `m`, `mat`, `o_del`, `e_del`, `o_ins`, `e_ins`, `w0`, `end_bonus`, `zdrop`. The one
/// added precondition is the caller's: the ISA feature (NEON or AVX2) has been detected, which is
/// what makes the `unsafe` kernel calls downstream sound.
///
/// # Returns
///
/// One [`ExtendResult`] per job in input order, whichever path produced it.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn simd_dispatch(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    // The NEON kernels score DNA with a vector compare that assumes the uniform bwa matrix
    // (diagonal `a`, off-diagonal `mm`, ambiguous row/col = `npen`). Any other matrix -> scalar.
    if !is_uniform_dna(mat, m) {
        return batched_extend_scalar(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }

    let a = mat[0] as i32; // match score for the uniform matrix
    let b = -(mat[1] as i32); // mismatch penalty magnitude
                              // How many mismatches an ungapped diagonal can carry and still be provably optimal (see
                              // `ungapped_hit`). The cheapest gapped alternative costs at least `o_min + e_min` and can at
                              // best convert every one of the `X` mismatches into a match, gaining `X * (a + b)`. So the
                              // ungapped walk is safe only while it wins *strictly*: `X * (a + b) < o_min + e_min`. The
                              // largest such X is `(o_min + e_min - 1) / (a + b)` in integer arithmetic.
                              //
                              // Strictness is not pedantry. On an exact tie the gapped path scores the same, and
                              // `ksw_extend2` breaks ties with `if (m > max)` -- whichever cell is reached first keeps the
                              // max -- so the DP can return the gapped CIGAR while this fast path returns the ungapped one.
                              // `-B 2 -E 3` is such a case: 3*(1+2) == 6+3.
                              //
                              // This used to read `o_min / (a + b - e_min)`, which coincides with the correct bound at bwa's
                              // defaults (6/4 = 1, and (6+1-1)/5 = 1) and is too permissive everywhere else: at `-B 3` it
                              // allowed X = 2, but 2*(1+3) = 8 > 6+1, so a single deletion beats that diagonal and the fast
                              // path returned an ungapped alignment where bwa-mem2 emits `4M1D146M`. Defaults are unchanged,
                              // so this costs no fast-path coverage on a stock run.
                              // Cheapest gap the scoring scheme allows: whichever of insertion/deletion opens and extends
                              // more cheaply. Using the minimum is what makes the bound conservative for both gap kinds.
    let (o_min, e_min) = (o_del.min(o_ins), e_del.min(e_ins));
    // Score swing from turning one mismatch into a match: `+a` gained, `+b` no longer lost. Zero or
    // negative only for a degenerate matrix, which disables the fast path via `x_threshold = -1`.
    let denom = a + b;
    let x_threshold = if denom > 0 {
        (o_min + e_min - 1) / denom
    } else {
        -1
    };

    // Ungapped diagonal HIT fast path: a perfect-diagonal extension gets its result in closed form,
    // skipping banded SW. Common on clean reads, so this removes a large slice of DP work while
    // staying byte-identical (see `ungapped_hit`). Non-HIT jobs fall through to the kernel binning.
    let mut out = vec![default_result(); jobs.len()];
    // Indices of the jobs the fast path declined, i.e. the ones that still need the real DP.
    let mut needs_dp: Vec<usize> = Vec::new();
    for (k, job) in jobs.iter().enumerate() {
        match ungapped_hit(job, a, b, x_threshold, zdrop) {
            Some(result) => out[k] = result,
            None => needs_dp.push(k),
        }
    }
    if needs_dp.is_empty() {
        return out;
    }
    if needs_dp.len() == jobs.len() {
        // No HITs: bin `jobs` directly (avoid the gather).
        return dispatch_bins(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }
    // Gather the declined jobs into a dense batch (kernels want full lanes, and a sparse batch
    // would waste them), then scatter the results back to their original slots below. `dp_jobs[s]`
    // is `jobs[needs_dp[s]]`, so `dp_results[s]` belongs at `out[needs_dp[s]]`.
    let dp_jobs: Vec<ExtendJob> = needs_dp.iter().map(|&k| jobs[k]).collect();
    let dp_results = dispatch_bins(
        &dp_jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
    );
    for (dp_slot, &k) in needs_dp.iter().enumerate() {
        out[k] = dp_results[dp_slot];
    }
    out
}

/// Length/score binning + kernel dispatch for the jobs that are not ungapped HITs. Bins each into
/// int8 (16 lanes) / int16 (8 lanes) / scalar, runs each bin, scatters back. This is bwa-mem2's
/// `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` binning; the 8-bit path packs twice the lanes for short pairs.
///
/// # Parameters
///
/// Same meaning, units and provenance as [`batched_extend`]'s: `jobs` (here, only the jobs the
/// ungapped fast path declined), `m`, `mat`, `o_del`, `e_del`, `o_ins`, `e_ins`, `w0`, `end_bonus`,
/// `zdrop`. Preconditions inherited from [`simd_dispatch`]: the ISA feature is present and `mat` is
/// the uniform DNA matrix, both of which the SIMD kernels rely on.
///
/// # Returns
///
/// One [`ExtendResult`] per job in input order. Which bin ran a job is not observable in the
/// output: all three kernels are exact inside the range that put a job in their bin.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn dispatch_bins(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    // The 8-bit kernel is **unsigned u8** (values 0..=255, positions <256), so it takes any job whose
    // score ceiling `minval` and both lengths fit under 256 — twice the reach of bwa-mem2's signed
    // `MAX_SEQ_LEN8=128`, hence far more jobs run in 16 lanes vs int16's 8. Both kernels are exact in
    // range, so this only changes which one runs, never the result.
    // Exclusive limits, in the same units for both lengths (bases) and scores. Raising `U8_LEN`
    // past 256 would let a score or a column index wrap in a u8 lane and silently corrupt results;
    // `MAX_SEQ_LEN16` is bwa-mem2's own name and value (`bandedSWA.h`) for the signed-i16 ceiling.
    const U8_LEN: usize = 256;
    const MAX_SEQ_LEN16: usize = 32768;
    // Largest single-cell score the matrix can award, feeding `cell_bound`'s worst case.
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    // Job indices per bin, in ascending order, partitioning `0..jobs.len()` exactly once.
    let (mut u8_idx, mut i16_idx, mut sc_idx) = (Vec::new(), Vec::new(), Vec::new());
    for (k, job) in jobs.iter().enumerate() {
        // Upper bound on every DP value this job can reach; must fit the lane type, as must both
        // lengths (column and row indices are stored in lanes too).
        let minval = cell_bound(job, max_sc);
        let (qlen, tlen) = (job.query.len(), job.target.len());
        if qlen < U8_LEN && tlen < U8_LEN && minval < U8_LEN as i32 {
            u8_idx.push(k);
        } else if qlen < MAX_SEQ_LEN16 && tlen < MAX_SEQ_LEN16 && minval < MAX_SEQ_LEN16 as i32 {
            i16_idx.push(k);
        } else {
            sc_idx.push(k);
        }
    }

    // Homogeneous fast path: whole batch in one bin -> run the kernel on `jobs` with no gather/scatter.
    // `n` is the batch size, so `x_idx.len() == n` means bin `x` took every job.
    let n = jobs.len();
    if u8_idx.len() == n {
        // SAFETY: neon available (checked by caller); U8_LEN bounds keep all values/positions in u8.
        return unsafe {
            sw_kernel_u8(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            )
        };
    }
    if i16_idx.len() == n {
        // SAFETY: neon available; MAX_SEQ_LEN16 bounds keep all values inside i16.
        return unsafe {
            sw_kernel_i16(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            )
        };
    }
    if sc_idx.len() == n {
        return batched_extend_scalar(
            jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        );
    }

    let mut out = vec![default_result(); n];

    // Gather the jobs of one bin into a contiguous batch, run the bin's kernel, scatter back.
    // `idx` = this bin's job indices into `jobs`; `kernel` = the batch routine to run them with.
    let mut run_bin = |idx: &[usize], kernel: &dyn Fn(&[ExtendJob]) -> Vec<ExtendResult>| {
        if idx.is_empty() {
            return;
        }
        // Dense copy of the bin, so `bin_results[s]` corresponds to original job `idx[s]`.
        let bin_jobs: Vec<ExtendJob> = idx.iter().map(|&k| jobs[k]).collect();
        let bin_results = kernel(&bin_jobs);
        for (bin_slot, &k) in idx.iter().enumerate() {
            out[k] = bin_results[bin_slot];
        }
    };
    run_bin(&u8_idx, &|bin| unsafe {
        // SAFETY: neon available (checked by caller); U8_LEN bounds keep all values/positions in u8.
        sw_kernel_u8(
            bin, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        )
    });
    run_bin(&i16_idx, &|bin| unsafe {
        // SAFETY: neon available; MAX_SEQ_LEN16 bounds keep all values inside i16.
        sw_kernel_i16(
            bin, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        )
    });
    run_bin(&sc_idx, &|bin| {
        batched_extend_scalar(
            bin, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
        )
    });
    out
}

/// Portable scalar reference (step 2b-i): lane-batched control flow, scalar per-cell arithmetic.
/// This is the byte-identity source of truth the NEON kernel is validated against.
///
/// Read this function first: it is `ksw_extend2` (`ksw.cpp:432-533`) with an extra `l` loop wrapped
/// around every statement, and the SIMD kernels below are a mechanical translation of it where the
/// `l` loop becomes lanes of one register. Nothing here is an optimization; every branch exists
/// because the C has it. Arithmetic is `i32`, so this path has no saturation preconditions at all
/// and is also the fallback for jobs too long/high-scoring for the u8 and i16 kernels.
///
/// # Parameters
///
/// Same meaning, units and provenance as [`batched_extend`]'s: `jobs`, `m`, `mat`, `o_del`,
/// `e_del`, `o_ins`, `e_ins`, `w0`, `end_bonus`, `zdrop`. Unlike the SIMD kernels this path reads
/// `mat` cell by cell, so **any** matrix shape is accepted, and every length and score is fine
/// because the arithmetic is `i32` throughout.
///
/// # Returns
///
/// One [`ExtendResult`] per job in input order, each field defined exactly as `ksw_extend2`'s.
#[allow(clippy::too_many_arguments)]
pub fn batched_extend_scalar(
    jobs: &[ExtendJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w0: i32,
    end_bonus: i32,
    zdrop: i32,
) -> Vec<ExtendResult> {
    // Cost of the *first* base of a gap: open + one extension. bwa charges `o + k*e` for length k,
    // so opening is `oe` and each further base `e` (`ksw.cpp:436`).
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // Best single-cell score in the matrix (`a` for the standard DNA matrix). Used only for the band
    // clamp, mirroring the `max` scan at `ksw.cpp:453-455`.
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

    let mut out = vec![default_result(); jobs.len()];

    // One chunk = one register's worth of lanes in the SIMD kernels. Chunks are independent.
    for chunk_start in (0..jobs.len()).step_by(LANES) {
        // The final chunk may be partial; lanes `nlane..LANES` are never marked active and their
        // garbage never reaches `out`.
        let nlane = (jobs.len() - chunk_start).min(LANES);

        // Per-lane inputs, all indexed by lane `l` in `0..nlane`: query and target length in bases,
        // the seed's starting score `h0`, and `w`, this lane's band half-width in cells after the
        // per-job clamp. Lanes `nlane..LANES` keep their zero initializers and are never active.
        let mut qlen = [0usize; LANES];
        let mut tlen = [0usize; LANES];
        let mut h0 = [0i32; LANES];
        let mut w = [0i32; LANES];
        for l in 0..nlane {
            let job = &jobs[chunk_start + l];
            qlen[l] = job.query.len();
            tlen[l] = job.target.len();
            h0[l] = job.h0;
            w[l] = clamp_band(w0, qlen[l], max_sc, end_bonus, o_ins, e_ins, o_del, e_del);
        }
        // Longest query and target in this chunk, in bases. `max_q` sizes the shared DP row and
        // `max_t` is how many rows the lockstep loop runs: every lane pays for the longest target.
        let max_q = qlen[..nlane].iter().copied().max().unwrap_or(0);
        let max_t = tlen[..nlane].iter().copied().max().unwrap_or(0);

        // Per-lane DP state, indexed [lane * (max_q + 1) + j]. This is AoS (lane-major) because the
        // scalar path indexes one lane at a time; the SIMD kernels flip it to SoA
        // (`[j * LANES + lane]`) so that one vector load grabs column `j` of all lanes at once.
        // `+1` column: `ksw.cpp:441` allocates `qlen + 1` cells because the row epilogue writes
        // `eh[end]` with `end == qlen` possible.
        let stride = max_q + 1;
        // eh_h[j] = H(i-1, j-1) on entry to cell j, H(i, j-1) on exit. eh_e[j] = E(i, j).
        let mut eh_h = vec![0i32; LANES * stride];
        let mut eh_e = vec![0i32; LANES * stride];

        // Per-lane band and result trackers, all mirroring `ksw_extend2`'s locals (`ksw.cpp:436`,
        // `:463-465`). `beg`/`end` are the half-open live column range for this lane's next row;
        // `max`/`max_i`/`max_j` the best local cell and where it was; `max_ie`/`gscore` the best
        // *query-exhausting* row and its score; `max_off` the largest diagonal excursion seen;
        // `done` the per-lane loop break (the C's `break` out of the `i` loop).
        let mut beg = [0i32; LANES];
        let mut end = [0i32; LANES];
        let mut max = [0i32; LANES];
        let mut max_i = [-1i32; LANES];
        let mut max_j = [-1i32; LANES];
        let mut max_ie = [-1i32; LANES];
        // `-1` not 0: bwa initializes gscore to -1 so that "no full-length alignment found" is
        // distinguishable from "found one scoring 0" (`ksw.cpp:463`), and the caller tests `gscore <= 0`.
        let mut gscore = [-1i32; LANES];
        let mut max_off = [0i32; LANES];
        let mut done = [true; LANES];

        // First row (i = -1): the alignment may start with an insertion, so H(-1,j) is h0 decayed by
        // one gap open plus j-1 extensions, truncated as soon as it would go non-positive. Exactly
        // `ksw.cpp:449-451`; the `> e_ins` loop condition (not `> 0`) is the C's, and reproducing it
        // literally matters because it decides how many leading columns start alive.
        for l in 0..nlane {
            eh_h[l * stride] = h0[l];
            if qlen[l] >= 1 {
                eh_h[l * stride + 1] = if h0[l] > oe_ins { h0[l] - oe_ins } else { 0 };
            }
            let mut j = 2usize;
            while j <= qlen[l] && eh_h[l * stride + j - 1] > e_ins {
                eh_h[l * stride + j] = eh_h[l * stride + j - 1] - e_ins;
                j += 1;
            }
            // The seed itself already scores h0, so the running max starts there (`ksw.cpp:463`).
            max[l] = h0[l];
            end[l] = qlen[l] as i32;
            done[l] = false;
        }

        // ===================================================================================
        // Main DP. Reminder: H = best score ending at this cell, M = ending on the diagonal,
        // E = with a deletion (gap in query) open, F = with an insertion (gap in target) open.
        // ===================================================================================
        // Target rows. All lanes share the row counter and run to the longest target in the chunk;
        // a lane that has finished (or whose target is shorter) is simply not `active` this row.
        //
        // Invariant at the top of each iteration, for every lane `l` still live: `eh_h[base + j]`
        // holds H(i-1, j-1) and `eh_e[base + j]` holds E(i, j) for every column `j` in
        // `[beg[l], end[l]]`; `max`/`max_i`/`max_j`, `max_ie`/`gscore` and `max_off` summarize rows
        // `0..i` only; `beg`/`end` are the previous row's tightened live range, not yet intersected
        // with row `i`'s band.
        for i in 0..max_t as i32 {
            // --- row prologue: per-lane band intersection and left-edge cell -----------------
            // Row-local state, all per lane: `h1` = H(i, j-1) carried rightwards, `f` = the carried
            // F recurrence, `row_max`/`mj` = this row's best cell and its column (the C's `m`/`mj`).
            let mut h1 = [0i32; LANES];
            let mut f = [0i32; LANES];
            let mut row_max = [0i32; LANES];
            let mut mj = [-1i32; LANES];
            let mut active = [false; LANES];
            // Union of the active lanes' bands. The shared `j` loop must cover every lane's band;
            // each lane is then masked back to its own `[beg, end)`. This is the price of lockstep:
            // a lane whose band is narrow still pays for the widest lane's columns.
            let mut gbeg = i32::MAX;
            let mut gend = 0i32;
            for l in 0..nlane {
                if done[l] || i >= tlen[l] as i32 {
                    continue;
                }
                active[l] = true;
                // Apply the band to this row: only columns within `w` of the diagonal survive
                // (`ksw.cpp:470-472`). `beg`/`end` also carry over the previous row's tightening, so
                // these are `max`/`min` against the incoming values rather than assignments.
                if beg[l] < i - w[l] {
                    beg[l] = i - w[l];
                }
                if end[l] > i + w[l] + 1 {
                    end[l] = i + w[l] + 1;
                }
                if end[l] > qlen[l] as i32 {
                    end[l] = qlen[l] as i32;
                }
                // Column -1 of this row: reachable only by a pure deletion of length i+1 from the
                // seed, hence `h0 - (o_del + e_del*(i+1))`, floored at 0 (`ksw.cpp:474-477`). Once
                // the band has moved off column 0 that cell is outside it, so h1 starts at 0.
                h1[l] = if beg[l] == 0 {
                    (h0[l] - (o_del + e_del * (i + 1))).max(0)
                } else {
                    0
                };
                gbeg = gbeg.min(beg[l]);
                gend = gend.max(end[l]);
            }

            // --- inner loop: one cell per (column, lane) -------------------------------------
            // Invariant at the top of each `j` iteration, per in-band lane: `h1[l]` = H(i, j-1),
            // `f[l]` = F(i, j) (the insertion carried in from the left), `row_max[l]`/`mj[l]` = the
            // best H seen in columns `beg..j` of this row and its column.
            for j in gbeg..gend {
                let ju = j as usize;
                for l in 0..nlane {
                    if !active[l] || j < beg[l] || j >= end[l] {
                        continue;
                    }
                    // Start of lane `l`'s row inside the lane-major `eh_h`/`eh_e` buffers.
                    let base = l * stride;
                    // On entry eh_h[j] = H(i-1, j-1) and eh_e[j] = E(i, j) (`ksw.cpp:485`).
                    let mut big_m = eh_h[base + ju];
                    let mut e = eh_e[base + ju];
                    // Publish H(i, j-1) into the array before overwriting: the *next* row will read
                    // this slot as its diagonal predecessor (`ksw.cpp:486`).
                    eh_h[base + ju] = h1[l];
                    // S(target[i], query[j]) from the substitution matrix. The SIMD kernels replace
                    // this gather with compares because they are restricted to the uniform matrix.
                    let score = i32::from(
                        mat[jobs[chunk_start + l].target[i as usize] as usize * m
                            + jobs[chunk_start + l].query[ju] as usize],
                    );
                    // M = H(i-1,j-1) + S, or 0 if the diagonal was already dead: the local restart
                    // (`ksw.cpp:487`). Note this is `!= 0`, not `> 0`: H is never negative here.
                    big_m = if big_m != 0 { big_m + score } else { 0 };
                    // H = max(M, E, F). E and F are >= 0 by construction, so H >= 0 even when M < 0
                    // (`ksw.cpp:488`), which is what lets the u8 kernel below use unsigned lanes.
                    let mut h = big_m.max(e);
                    h = h.max(f[l]);
                    h1[l] = h;
                    // Row argmax. `<=` means a tie moves `mj` to the larger column, matching
                    // `ksw.cpp:491-492` (`mj = m > h? mj : j`), where the condition is likewise not
                    // strict. This tie-break is observable: it sets `qle`, hence the CIGAR.
                    if row_max[l] <= h {
                        mj[l] = j;
                        row_max[l] = h;
                    }
                    // Gaps open from M, not H: bandedSWA's MAIN_CODE16 subtracts oe_ins/oe_del
                    // from `m11` (`bandedSWA.cpp:338`, `:342`), and `ksw_extend2` from `M`
                    // (`ksw.cpp:493-501`). Using `h` here instead would make an insertion followed
                    // immediately by a deletion cost nothing extra, inflating gscore in repeats.
                    // E(i+1,j): either extend the deletion already open in this column, or open a
                    // fresh one out of M. The `.max(0)` on the open term is the C's `t = t > 0? t : 0`,
                    // and it is also what keeps E non-negative for the unsigned kernel.
                    let t = (big_m - oe_del).max(0);
                    e = (e - e_del).max(t);
                    eh_e[base + ju] = e;
                    // F(i,j+1): the same for the insertion running along this row. `f` stays in a
                    // scalar/register because it only ever moves one column to the right.
                    let t = (big_m - oe_ins).max(0);
                    f[l] = (f[l] - e_ins).max(t);
                }
            }

            // --- row epilogue: to-end score, termination tests, band tightening (per lane) ---
            for l in 0..nlane {
                if !active[l] {
                    continue;
                }
                let base = l * stride;
                // Close the row: the cell just past the band gets the carried H and a zeroed E
                // (`ksw.cpp:503`). E is zeroed rather than carried because bwa disallows an
                // insertion directly adjacent to a deletion.
                eh_h[base + end[l] as usize] = h1[l];
                eh_e[base + end[l] as usize] = 0;
                // If the band reached the end of the query, this row has a *global* (query-exhausting)
                // alignment scoring h1. `<=` so the last such row wins ties, matching
                // `ksw.cpp:504-507` (`max_ie = gscore > h1? max_ie : i`). The C tests `j == qlen`
                // after the loop, which is the same condition as `end == qlen` here.
                if end[l] == qlen[l] as i32 && gscore[l] <= h1[l] {
                    max_ie[l] = i;
                    gscore[l] = h1[l];
                }
                // Whole row scored 0: every path through it is dead, so no later row can revive it.
                // `ksw.cpp:508` breaks out of the row loop; a lane cannot break, so it goes idle.
                if row_max[l] == 0 {
                    done[l] = true;
                    continue;
                }
                if row_max[l] > max[l] {
                    max[l] = row_max[l];
                    max_i[l] = i;
                    max_j[l] = mj[l];
                    // How far off the main diagonal the best cell sits. The caller compares this
                    // against the band to decide whether to redo the alignment wider
                    // (`ksw.cpp:509-511`).
                    let off = (mj[l] - i).abs();
                    if off > max_off[l] {
                        max_off[l] = off;
                    }
                } else if zdrop > 0 {
                    // Z-drop (`ksw.cpp:512-517`): stop when the score has fallen `zdrop` below the
                    // best seen, but only after crediting the gap that the drift away from the best
                    // cell would have cost. `(i - max_i)` rows and `(mj - max_j)` columns advanced;
                    // their difference is the net indel length, charged at `e_del` when the target
                    // ran ahead and `e_ins` when the query did. Without that correction a long, good
                    // gapped alignment would be cut short. `zdrop <= 0` disables the test entirely.
                    let drop = if i - max_i[l] > mj[l] - max_j[l] {
                        max[l] - row_max[l] - ((i - max_i[l]) - (mj[l] - max_j[l])) * e_del
                    } else {
                        max[l] - row_max[l] - ((mj[l] - max_j[l]) - (i - max_i[l])) * e_ins
                    };
                    if drop > zdrop {
                        done[l] = true;
                        continue;
                    }
                }
                // Band tightening (`ksw.cpp:519-523`): drop the all-zero prefix and suffix of the row,
                // since a cell with H == 0 and E == 0 can contribute nothing to the next row. This is
                // pure work reduction in the C, but it is *not* optional here: `beg`/`end` feed the
                // `beg == 0` first-column test and the `end == qlen` gscore test above, so a
                // different band gives different output. `last_live + 2` (not `+ 1`) keeps one
                // column of slack because the next row's diagonal reads one column further left.
                // `first_live` scans up to the leftmost column with a non-zero H or E, `last_live`
                // scans down to the rightmost one; both are column indices in this lane's row.
                let mut first_live = beg[l];
                while first_live < end[l]
                    && eh_h[base + first_live as usize] == 0
                    && eh_e[base + first_live as usize] == 0
                {
                    first_live += 1;
                }
                beg[l] = first_live;
                let mut last_live = end[l];
                while last_live >= beg[l]
                    && eh_h[base + last_live as usize] == 0
                    && eh_e[base + last_live as usize] == 0
                {
                    last_live -= 1;
                }
                end[l] = if last_live + 2 < qlen[l] as i32 {
                    last_live + 2
                } else {
                    qlen[l] as i32
                };
            }
        }

        // `+1` everywhere: the C returns *lengths consumed*, and the trackers hold 0-based indices
        // seeded at -1, so an untouched tracker yields 0 (`ksw.cpp:527-531`).
        for l in 0..nlane {
            out[chunk_start + l] = ExtendResult {
                score: max[l],
                qle: max_j[l] + 1,
                tle: max_i[l] + 1,
                gtle: max_ie[l] + 1,
                gscore: gscore[l],
                max_off: max_off[l],
            };
        }
    }

    out
}

/// All-zero placeholder used to pre-size the output vector before the bins scatter their real
/// results back into it. Every slot is overwritten before [`batched_extend`] returns, so these
/// values never reach a caller.
///
/// # Returns
///
/// An [`ExtendResult`] with every field 0. Note this is *not* the "no alignment" encoding the DP
/// itself produces: a real empty extension reports `gscore = -1`, not 0. Nothing may read these
/// zeros as if they meant something.
fn default_result() -> ExtendResult {
    ExtendResult {
        score: 0,
        qle: 0,
        tle: 0,
        gtle: 0,
        gscore: 0,
        max_off: 0,
    }
}

/// Per-lane band clamp, mirroring the `w` adjustments in `ksw_extend2` (`ksw.cpp:456-461`).
///
/// A band wider than the longest gap that could ever pay for itself is wasted work, and bwa shrinks
/// `w` to that bound. The longest useful insertion is the point where the gap cost `o_ins + k*e_ins`
/// eats the entire score an alignment could earn (`qlen * max_sc + end_bonus`), so
/// `k = (qlen*max_sc + end_bonus - o_ins) / e_ins + 1`; likewise for deletions. `.max(1)` keeps the
/// band from collapsing to zero on very short queries.
///
/// This must stay in `f64`, not integer division: the C computes it in `double` and truncates once
/// at the cast, and integer division would round differently for negative numerators (short query,
/// large `o_ins`), producing a different band and therefore a different alignment.
///
/// Returns the clamped half-width in cells. `w0` is the user's `-w`; the result is never larger.
///
/// # Parameters
///
/// - `w0`: the requested band half-width in cells (bwa's `-w`, default 100). The result is
///   `min(w0, useful_ins, useful_del)`, so it is never widened.
/// - `qlen`: this job's query length in bases. Longer queries can pay for longer gaps, so this
///   scales the bound linearly.
/// - `max_sc`: the largest entry of `mat` (the match score `a`), in score units: the most a single
///   aligned base can earn, hence `qlen * max_sc` is the most the whole alignment can earn.
/// - `end_bonus`: bwa's `-L`. Its only role in the whole file is right here, as extra score the
///   alignment could earn by reaching the query end, which lets a slightly longer gap pay off.
/// - `o_ins` / `e_ins`, `o_del` / `e_del`: gap open and extend penalties as positive magnitudes.
///   `e_ins` and `e_del` must be non-zero (they are divisors).
///
/// # Returns
///
/// The band half-width in cells to use for this job, always `>= 1` and `<= w0`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn clamp_band(
    w0: i32,
    qlen: usize,
    max_sc: i32,
    end_bonus: i32,
    o_ins: i32,
    e_ins: i32,
    o_del: i32,
    e_del: i32,
) -> i32 {
    let mut clamped = w0;
    // Longest insertion, in bases, that could still pay for itself: the point where
    // `o_ins + k*e_ins` consumes the whole score the alignment could earn. Same below for `max_del`
    // with the deletion penalties. Both can come out negative or zero on a very short query, which
    // is what the `.max(1)` guards against.
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    clamped = clamped.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    clamped = clamped.min(max_del.max(1));
    clamped
}

/// Generate a batched banded-SW kernel for a lane element type + target-feature string. One macro
/// serves NEON (aarch64), AVX2 and AVX-512 (x86_64): every SIMD op is a parameter, so the width
/// variants and the ISAs cannot drift. Layout is SoA `[column*LANES + lane]` (bwa-mem2's `bandedSWA`);
/// the band mask is a blendv select; the recurrence uses saturating (u8) or plain (i16) add/sub,
/// exact within the caller-guaranteed range. Expanded inside a per-ISA module that imports the
/// intrinsics and `super::{clamp_band, default_result}` / `bwa_extend::{ExtendJob, ExtendResult}`.
///
/// # Macro parameters
///
/// - `$name`: generated function name. `$elem`: the lane element type (`u8` or `i16`); every DP
///   value lives in this type, so it sets the saturation regime. `$umask`: the unsigned integer of
///   the same width, used for compare results and masks (NEON compares return unsigned vectors).
/// - `$lanes`: elements per register = `register_bits / 8*sizeof($elem)`. Must agree with the
///   intrinsics passed in, and it is also the SoA stride, so a mismatch corrupts memory rather than
///   just slowing things down.
/// - `$feat`: the `#[target_feature]` string the generated `unsafe fn` requires of its caller.
/// - Op parameters, all semantic rather than syntactic so the two ISAs can be adapted:
///   `$dup` broadcast a scalar to every lane; `$lds`/`$sts` load/store a vector of `$elem`;
///   `$ldu` load a vector of `$umask` (mask arrays); `$add`/`$sub` lane-wise add/subtract
///   (**saturating for u8**, wrapping for i16, see the invariants below); `$max` lane-wise maximum
///   (unsigned for u8, signed for i16); `$ceqz` "lane == 0" to an all-ones/all-zero mask;
///   `$cge`/`$clt`/`$ceq` the >=, <, == compares in the element's signedness; `$orru`/`$andu`
///   bitwise or/and on masks; `$bsl(mask, a, b)` = per-lane `mask ? a : b`.
///
/// # Invariants the caller must uphold
///
/// - Every DP value (`H`, `E`, `F`, and the intermediate `M`) and every column index the kernel
///   stores must fit `$elem`. `dispatch_bins` enforces this with the `cell_bound` score ceiling and
///   the length limits; violating it silently truncates scores instead of trapping.
/// - `mat` must be the uniform DNA matrix (`is_uniform_dna`), because the substitution score is
///   recomputed from three broadcast constants instead of being read from `mat`.
/// - `query`/`target` codes must be in `0..=4`. Code 4 (N) is detected by a `>= 4` compare, so a
///   stray code 5 would be scored as N rather than rejected.
/// - For the u8 instantiation, `$add`/`$sub` must be the *saturating* unsigned intrinsics: the
///   kernel relies on `sub` clamping at 0 to realize the `max(0, .)` in the E/F recurrences, and on
///   `add` never wrapping past 255. With wrapping ops a mismatch at score 0 would wrap to 251.
/// - No alignment requirement: all vector loads/stores here are the unaligned forms, and the SoA
///   buffers are plain `Vec`s. Their length is always a multiple of `$lanes`, which is what keeps
///   the pointer arithmetic (`ptr.add(j * LANES)`) in bounds for a full-width access.
macro_rules! define_sw_kernel {
    (
            $name:ident, $elem:ty, $umask:ty, $lanes:expr, feat = $feat:literal,
            dup = $dup:path, lds = $lds:path, sts = $sts:path, ldu = $ldu:path,
            add = $add:path, sub = $sub:path, max = $max:path,
            ceqz = $ceqz:path, cge = $cge:path, clt = $clt:path,
            ceq = $ceq:path, orru = $orru:path,
            andu = $andu:path, bsl = $bsl:path
        ) => {
        /// # Safety
        /// Requires the `$feat` target feature; all vector loads/stores use fixed-size
        /// `[$elem; LANES]` / `[$umask; LANES]` scratch arrays, in-bounds by construction.
        ///
        /// # Parameters
        ///
        /// Same meaning, units and provenance as `batched_extend`'s: `jobs` (2-bit base codes plus
        /// `h0` per job), `m` (alphabet size, 5), `mat` (row-major `m*m` matrix), `o_del`/`e_del`
        /// and `o_ins`/`e_ins` (gap open/extend penalties as positive magnitudes), `w0` (band
        /// half-width before the per-job clamp), `end_bonus` (band-clamp input only) and `zdrop`.
        ///
        /// Beyond the target feature, the caller must also uphold the range and matrix-shape
        /// invariants listed on `define_sw_kernel!`: every DP value and column index must fit
        /// `$elem`, and `mat` must be the uniform DNA matrix. `dispatch_bins` is what establishes
        /// both. Violating either produces wrong scores silently, not a panic.
        ///
        /// # Returns
        ///
        /// One `ExtendResult` per job in input order, byte-identical to `ksw_extend2` per job.
        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            jobs: &[ExtendJob],
            m: usize,
            mat: &[i8],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            w0: i32,
            end_bonus: i32,
            zdrop: i32,
        ) -> Vec<ExtendResult> {
            // Lanes per vector register: 16 for NEON u8, 8 for NEON i16, 32/16 for AVX2. Also the
            // SoA stride, so it must match the intrinsics the macro was instantiated with.
            const LANES: usize = $lanes;
            // Cost of a gap's *first* base (open + one extend), as in the scalar path.
            let oe_del = o_del + e_del;
            let oe_ins = o_ins + e_ins;
            // Best single-cell score in the matrix, used only by `clamp_band`.
            let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

            // One slot per job, overwritten chunk by chunk before returning.
            let mut out = vec![default_result(); jobs.len()];

            // Loop-invariant broadcast constants: every lane of each of these holds the same
            // scalar, so a vector op applies the same penalty to all lanes at once. `zero_v` is the
            // recurrence's floor and the identity for the `max(0, .)` clamps.
            let oe_del_v = $dup(oe_del as $elem);
            let oe_ins_v = $dup(oe_ins as $elem);
            let e_del_v = $dup(e_del as $elem);
            let e_ins_v = $dup(e_ins as $elem);
            let zero_v = $dup(0);
            // DNA score as a vector compare (no per-cell gather): the caller (neon_dispatch) only
            // reaches here for the uniform bwa matrix. The signed substitution score
            // (`N ? npen : (t==q ? a : mm)`) is kept as its **positive** parts so the recurrence
            // works in unsigned-saturating u8 as well as signed i16: `sbt_pos` (match bonus `a`)
            // via saturating add, `sbt_neg` (mismatch `|mm|` / ambiguous `|npen|`) via saturating
            // sub. For the signed kernels `(m + pos) - neg == m + score` exactly (no wrap), so this
            // is byte-identical; for u8 it is bwa-mem2's `MAIN_CODE8_CORE`.
            let a_pos_v = $dup(mat[0] as $elem); // a >= 0
            let mm_mag_v = $dup((-(i32::from(mat[1]))) as $elem); // |mm|
            let npen_mag_v = $dup((-(i32::from(mat[m - 1]))) as $elem); // |npen|
            let amb_v = $dup(4); // code 4 = N; codes are 0..=4

            // Per-chunk DP scratch, allocated once and reused across chunks (clear+resize keeps the
            // capacity, so it grows to the largest chunk's size then stops reallocating). Byte-safe:
            // resize-from-empty zero-fills, identical to the old per-chunk `vec![0; ..]`.
            let mut eh_h: Vec<$elem> = Vec::new();
            let mut eh_e: Vec<$elem> = Vec::new();
            let mut qcode: Vec<$elem> = Vec::new();
            let mut tcode: Vec<$elem> = Vec::new();

            // =======================================================================
            // One chunk = one register's worth of independent alignments.
            // =======================================================================
            for chunk_start in (0..jobs.len()).step_by(LANES) {
                // Jobs in this chunk: LANES except for a partial final chunk. Lanes
                // `nlane..LANES` are never marked active, so their garbage never reaches `out`.
                let nlane = (jobs.len() - chunk_start).min(LANES);

                // --- chunk setup: per-lane lengths, h0 and band --------------------
                // Indexed by lane `l`: sequence lengths in bases, the seed's starting score, and
                // this lane's clamped band half-width in cells.
                let mut qlen = [0usize; LANES];
                let mut tlen = [0usize; LANES];
                let mut h0 = [0i32; LANES];
                let mut w = [0i32; LANES];
                for l in 0..nlane {
                    let job = &jobs[chunk_start + l];
                    qlen[l] = job.query.len();
                    tlen[l] = job.target.len();
                    h0[l] = job.h0;
                    w[l] = clamp_band(w0, qlen[l], max_sc, end_bonus, o_ins, e_ins, o_del, e_del);
                }
                // Longest query and target in this chunk, in bases: `max_q` sizes the shared row,
                // `max_t` is the number of lockstep row iterations every lane pays for.
                let max_q = qlen[..nlane].iter().copied().max().unwrap_or(0);
                let max_t = tlen[..nlane].iter().copied().max().unwrap_or(0);

                // Columns per SoA buffer: `max_q + 1` because the row epilogue writes `eh_h[end]`
                // with `end == qlen` possible. Buffers are therefore `stride * LANES` elements,
                // element `j * LANES + l` being column `j` of lane `l`.
                let stride = max_q + 1;
                eh_h.clear();
                eh_h.resize(stride * LANES, 0 as $elem);
                eh_e.clear();
                eh_e.resize(stride * LANES, 0 as $elem);

                // Transpose the per-job sequences into SoA: `qcode[j * LANES + l]` is job `l`'s base
                // at query column `j`. One vector load then yields "column j of all lanes", which is
                // the whole reason for the layout (bwa-mem2's `seq1SoA`/`seq2SoA`, `bandedSWA.cpp:449`).
                // Padding is code 0 (an ordinary base), not a sentinel: padded lanes are masked out
                // of every write by `band`, so whatever score they compute is discarded. The point of
                // padding is only that the load stays in bounds and branch-free.
                qcode.clear();
                qcode.resize(stride * LANES, 0 as $elem);
                tcode.clear();
                tcode.resize((max_t + 1) * LANES, 0 as $elem);
                for l in 0..nlane {
                    for (ju, &c) in jobs[chunk_start + l].query.iter().enumerate() {
                        qcode[ju * LANES + l] = c as $elem;
                    }
                    for (iu, &c) in jobs[chunk_start + l].target.iter().enumerate() {
                        tcode[iu * LANES + l] = c as $elem;
                    }
                }

                // --- per-lane band and result trackers (same meaning as the scalar path) ---
                let mut beg = [0i32; LANES];
                let mut end = [0i32; LANES];
                let mut max = [0i32; LANES];
                let mut max_i = [-1i32; LANES];
                let mut max_j = [-1i32; LANES];
                let mut max_ie = [-1i32; LANES];
                let mut gscore = [-1i32; LANES];
                let mut max_off = [0i32; LANES];
                let mut done = [true; LANES];

                // --- row -1: the leading-insertion run that seeds the first real row ---
                for l in 0..nlane {
                    eh_h[l] = h0[l] as $elem; // column 0
                    if qlen[l] >= 1 {
                        eh_h[LANES + l] = if h0[l] > oe_ins {
                            (h0[l] - oe_ins) as $elem
                        } else {
                            0
                        };
                    }
                    let mut j = 2usize;
                    while j <= qlen[l] && i32::from(eh_h[(j - 1) * LANES + l]) > e_ins {
                        eh_h[j * LANES + l] = eh_h[(j - 1) * LANES + l] - e_ins as $elem;
                        j += 1;
                    }
                    max[l] = h0[l];
                    end[l] = qlen[l] as i32;
                    done[l] = false;
                }

                // =================================================================
                // Main DP. Reminder: H = best score ending at this cell, M = ending
                // on the diagonal, E = deletion open (gap in query), F = insertion
                // open (gap in target). `_v` = the vector holding one per lane.
                // =================================================================
                // Invariant at the top of each row iteration, for every live lane `l`: SoA column
                // `j` holds H(i-1, j-1) in `eh_h` and E(i, j) in `eh_e` across `[beg[l], end[l]]`;
                // the i32 trackers summarize rows `0..i`; `beg`/`end` are the previous row's
                // tightened range, not yet intersected with row `i`'s band.
                for i in 0..max_t as i32 {
                    // --- row prologue (scalar, per lane): band intersection, left edge ---
                    // `h1[l]` = H(i, beg[l]-1), the value carried into the row's first column.
                    let mut h1 = [0 as $elem; LANES];
                    // `active[l]`: this lane has a row `i` and has not terminated. `gbeg`/`gend`:
                    // the union of the active lanes' bands, i.e. the half-open column range the
                    // shared vector loop below must cover. They stay `i32::MAX`/`0` (an empty
                    // range) if no lane is active this row.
                    let mut active = [false; LANES];
                    let mut gbeg = i32::MAX;
                    let mut gend = 0i32;
                    for l in 0..nlane {
                        if done[l] || i >= tlen[l] as i32 {
                            continue;
                        }
                        active[l] = true;
                        if beg[l] < i - w[l] {
                            beg[l] = i - w[l];
                        }
                        if end[l] > i + w[l] + 1 {
                            end[l] = i + w[l] + 1;
                        }
                        if end[l] > qlen[l] as i32 {
                            end[l] = qlen[l] as i32;
                        }
                        h1[l] = if beg[l] == 0 {
                            (h0[l] - (o_del + e_del * (i + 1))).max(0) as $elem
                        } else {
                            0
                        };
                        gbeg = gbeg.min(beg[l]);
                        gend = gend.max(end[l]);
                    }

                    // Move the row-local scalars into vectors. `h1_v` was just computed per lane
                    // (the first-column value), the rest start at their identity.
                    let mut h1_v = $lds(h1.as_ptr());
                    let mut f_v = zero_v;
                    let mut rowmax_v = zero_v;
                    // `mj = -1` as the C does; for the u8 kernel this is 255, which is fine because
                    // it is only ever read after `rowmax > 0` has proved some column updated it.
                    let mut mj_v = $dup((-1i32) as $elem);

                    // Build the per-lane band as vectors so the inner loop can mask with two
                    // compares instead of a branch per lane. `beg`/`end` are column indices, so they
                    // must fit `$elem` too, which is the second reason the u8 kernel caps lengths at
                    // 256. An inactive lane gets an all-zero mask (`beg = end = 0` would also make
                    // the range empty, but the explicit `active_v` is what keeps a *finished* lane's
                    // stale state from being rewritten).
                    let mut beg_lane = [0 as $elem; LANES];
                    let mut end_lane = [0 as $elem; LANES];
                    let mut active_lane = [0 as $umask; LANES];
                    for l in 0..nlane {
                        if active[l] {
                            beg_lane[l] = beg[l] as $elem;
                            end_lane[l] = end[l] as $elem;
                            // All-ones = "true" in the blend/and convention used throughout.
                            active_lane[l] = <$umask>::MAX;
                        }
                    }
                    let beg_v = $lds(beg_lane.as_ptr());
                    let end_v = $lds(end_lane.as_ptr());
                    let active_v = $ldu(active_lane.as_ptr());
                    let t_v = $lds(tcode.as_ptr().add(i as usize * LANES)); // this row's target base per lane
                    let t_is_n = $cge(t_v, amb_v); // target base is N — constant across the row

                    // The shared column loop over the union band. Every lane executes every column;
                    // `band` decides whether the lane keeps the result. This is the core trade of
                    // inter-sequence batching: uniform control flow at the cost of some wasted lanes.
                    // --- inner loop (vector): one column of all lanes per iteration ---
                    // Invariant at the top of each `j` iteration: lane `l` of `h1_v` holds
                    // H(i, j-1), lane `l` of `f_v` holds F(i, j) carried from the left, and lane
                    // `l` of `rowmax_v`/`mj_v` holds the best H in columns `beg[l]..j` of this row
                    // and its column index (only meaningful for lanes that were in band).
                    for j in gbeg..gend {
                        // Byte offset of column `j` in every SoA buffer (all lanes of that column
                        // sit contiguously, which is the whole point of the layout).
                        let col_base = j as usize * LANES;
                        // The column index itself broadcast to every lane, so the band compares
                        // and the `mj` update can be done lane-wise against per-lane bounds.
                        let j_v = $dup(j as $elem);
                        // band = active && beg <= j < end, as an all-ones/all-zero mask per lane.
                        // Two compares rather than a shift-based window because `beg`/`end` differ
                        // per lane.
                        let band = $andu(active_v, $andu($cge(j_v, beg_v), $clt(j_v, end_v)));

                        // DNA substitution score split into positive parts (no gather):
                        //   sbt_pos = (t==q && !N) ? a : 0 ;  sbt_neg = N ? |npen| : (t==q ? 0 : |mm|)
                        // Lane `l` = job `l`'s query base code at column `j` (0..=4). One load, one
                        // base per lane, which is what the SoA transpose above was built for.
                        let q_v = $lds(qcode.as_ptr().add(col_base));
                        // Three masks, each all-ones or all-zero per lane: bases equal, and either
                        // base ambiguous (code >= 4). `t_is_n` was hoisted out of the loop since the
                        // target base is fixed for the whole row.
                        let is_eq = $ceq(t_v, q_v);
                        let is_n = $orru(t_is_n, $cge(q_v, amb_v));
                        // Nested selects, innermost first: match -> a, mismatch -> 0 on the positive
                        // side and 0 / |mm| on the negative side; then the N case overrides both.
                        // Order matters: N wins over equality, because two N bases compare equal but
                        // must still score `npen`, not `a`.
                        let sbt_pos = $bsl(is_n, zero_v, $bsl(is_eq, a_pos_v, zero_v));
                        let sbt_neg = $bsl(is_n, npen_mag_v, $bsl(is_eq, zero_v, mm_mag_v));

                        let m_v = $lds(eh_h.as_ptr().add(col_base)); // H(i-1, j-1)
                                                                     // E(i, j), one lane per job. Written while row i-1 was walked, but it is the
                                                                     // deletion state ENTERING cell (i, j), so it is indexed (i, j) here and in
                                                                     // the scalar path. Same convention as the C ("eh[j] = { H(i-1,j-1), E(i,j) }",
                                                                     // `ksw.cpp:479`); labelling it E(i-1, j) would make the two paths look like
                                                                     // they hold different values when they hold the same one.
                        let e_v = $lds(eh_e.as_ptr().add(col_base)); // E(i, j)

                        // eh_h[j] <- h1 (old) for in-band lanes; out-of-band keep old m_v.
                        // The mask has to be applied on the *store*, not skipped around, because all
                        // lanes share the store instruction: writing back the value just loaded is
                        // how an out-of-band lane is left untouched.
                        $sts(eh_h.as_mut_ptr().add(col_base), $bsl(band, h1_v, m_v));

                        // M = ((m + sbt_pos) - sbt_neg); m==0 -> local restart (0). Saturating for
                        // u8, plain for signed; identical either way in the guaranteed range.
                        let bigm_pre = $sub($add(m_v, sbt_pos), sbt_neg);
                        // The local restart: `ceqz` finds the lanes whose diagonal predecessor was
                        // already 0 and forces M back to 0 there (`ksw.cpp:487`,
                        // `bandedSWA.cpp:334-335` which does the same `cmpeq h00, zero` + blend).
                        let bigm_v = $bsl($ceqz(m_v), zero_v, bigm_pre);

                        // max(M, E) is only an intermediate on the way to H; gaps open from M, not
                        // from this (see below).
                        let m_or_e_v = $max(bigm_v, e_v);
                        let h_v = $max(m_or_e_v, f_v);
                        h1_v = $bsl(band, h_v, h1_v);

                        // if row_max <= h { mj = j; row_max = h } (ties take larger j)
                        // `$cge` gives the non-strict compare, matching the scalar `row_max <= h`
                        // and `ksw.cpp:491`. Because `j` increases monotonically, the last lane to
                        // satisfy it wins, which reproduces the C's "later column keeps the tie".
                        // All-ones in the lanes that are in band *and* whose new H ties or beats
                        // this row's running best, i.e. exactly the lanes whose `rowmax`/`mj` the
                        // two blends below should replace.
                        let upd = $andu(band, $cge(h_v, rowmax_v));
                        rowmax_v = $bsl(upd, h_v, rowmax_v);
                        mj_v = $bsl(upd, $dup(j as $elem), mj_v);

                        // e = max(e - e_del, max(M - oe_del, 0)); bandedSWA's MAIN_CODE16 opens the
                        // gap from `m11`, not `h11` (`bandedSWA.cpp:342-345`). The explicit
                        // `max(., zero)` is redundant for the u8 kernel (saturating `sub` already
                        // floors at 0) but free, and required for i16 where `sub` wraps.
                        // `open_del_v` lane l = the score of opening a fresh deletion here,
                        // M - (o_del + e_del), floored at 0; `e_new` lane l = E(i+1, j), the better
                        // of extending the deletion already open in this column and that open.
                        let open_del_v = $max($sub(bigm_v, oe_del_v), zero_v);
                        let e_new = $max($sub(e_v, e_del_v), open_del_v);
                        $sts(eh_e.as_mut_ptr().add(col_base), $bsl(band, e_new, e_v));

                        // f = max(f - e_ins, max(M - oe_ins, 0)). M does not depend on the carried f,
                        // so the carried chain is just sub+max (~2 ops) with no f->h->f dependency.
                        // This is the whole reason the F recurrence needs no lazy-F fixup loop of the
                        // kind striped kernels use (`ksw.cpp:179-190`): striped layouts break the
                        // left-to-right order of F, inter-sequence layouts do not. F is genuinely
                        // sequential along the row, and that sequential chain is this kernel's
                        // critical path, which is why shortening it to sub+max was a measurable win.
                        // Mirror of the pair above for the insertion: `open_ins_v` lane l is the
                        // cost-adjusted score of opening one here, `f_new` lane l is F(i, j+1),
                        // which the next iteration will read as its carried `f_v`.
                        let open_ins_v = $max($sub(bigm_v, oe_ins_v), zero_v);
                        let f_new = $max($sub(f_v, e_ins_v), open_ins_v);
                        f_v = $bsl(band, f_new, f_v);
                    }

                    // --- row epilogue (scalar, per lane): mirrors batched_extend_scalar ---
                    // Spill the three row results back to scalars. The row epilogue below is kept
                    // scalar on purpose: it is per-row (not per-cell) work, it is full of data
                    // dependent branching (z-drop, band tightening, early termination), and
                    // vectorizing it would buy nothing while making the tie-breaks much harder to
                    // keep bit-exact against the C.
                    let mut h1_out = [0 as $elem; LANES];
                    let mut rowmax_out = [0 as $elem; LANES];
                    let mut mj_out = [0 as $elem; LANES];
                    $sts(h1_out.as_mut_ptr(), h1_v);
                    $sts(rowmax_out.as_mut_ptr(), rowmax_v);
                    $sts(mj_out.as_mut_ptr(), mj_v);

                    for l in 0..nlane {
                        if !active[l] {
                            continue;
                        }
                        // Widen back to i32 before any comparison: the trackers (`max`, `gscore`,
                        // `max_i`, ...) are i32 and hold values like -1 that `$elem` cannot express.
                        // From here on the logic is identical to `batched_extend_scalar`'s epilogue,
                        // and the comments there apply line for line.
                        let h1_l = i32::from(h1_out[l]);
                        let row_max_l = i32::from(rowmax_out[l]);
                        let mj_l = i32::from(mj_out[l]);
                        eh_h[end[l] as usize * LANES + l] = h1_l as $elem;
                        eh_e[end[l] as usize * LANES + l] = 0;
                        if end[l] == qlen[l] as i32 && gscore[l] <= h1_l {
                            max_ie[l] = i;
                            gscore[l] = h1_l;
                        }
                        if row_max_l == 0 {
                            done[l] = true;
                            continue;
                        }
                        if row_max_l > max[l] {
                            max[l] = row_max_l;
                            max_i[l] = i;
                            max_j[l] = mj_l;
                            let off = (mj_l - i).abs();
                            if off > max_off[l] {
                                max_off[l] = off;
                            }
                        } else if zdrop > 0 {
                            let drop = if i - max_i[l] > mj_l - max_j[l] {
                                max[l] - row_max_l - ((i - max_i[l]) - (mj_l - max_j[l])) * e_del
                            } else {
                                max[l] - row_max_l - ((mj_l - max_j[l]) - (i - max_i[l])) * e_ins
                            };
                            if drop > zdrop {
                                done[l] = true;
                                continue;
                            }
                        }
                        let mut first_live = beg[l];
                        while first_live < end[l]
                            && eh_h[first_live as usize * LANES + l] == 0
                            && eh_e[first_live as usize * LANES + l] == 0
                        {
                            first_live += 1;
                        }
                        beg[l] = first_live;
                        let mut last_live = end[l];
                        while last_live >= beg[l]
                            && eh_h[last_live as usize * LANES + l] == 0
                            && eh_e[last_live as usize * LANES + l] == 0
                        {
                            last_live -= 1;
                        }
                        end[l] = if last_live + 2 < qlen[l] as i32 {
                            last_live + 2
                        } else {
                            qlen[l] as i32
                        };
                    }
                }

                for l in 0..nlane {
                    out[chunk_start + l] = ExtendResult {
                        score: max[l],
                        qle: max_j[l] + 1,
                        tle: max_i[l] + 1,
                        gtle: max_ie[l] + 1,
                        gscore: gscore[l],
                        max_off: max_off[l],
                    };
                }
            }

            out
        }
    };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{clamp_band, default_result};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::aarch64::*;

    // 16-bit kernel: **signed** i16 with plain (wrapping) add/sub, 8 lanes in a 128-bit `int16x8_t`.
    // Signed because the intermediate `M = H + score` can legitimately go negative before the
    // `max(M, E, F)` pulls it back up (`ksw.cpp:488` says as much), and only signed arithmetic gets
    // that right without a bias. Non-saturating is safe because `dispatch_bins` has already proved
    // every value stays under 32768. This is the fallback width: it takes the jobs the u8 kernel
    // cannot, at half the throughput.
    define_sw_kernel!(
        batched_extend_neon_i16,
        i16,
        u16,
        8,
        feat = "neon",
        dup = vdupq_n_s16,
        lds = vld1q_s16,
        sts = vst1q_s16,
        ldu = vld1q_u16,
        add = vaddq_s16,
        sub = vsubq_s16,
        max = vmaxq_s16,
        ceqz = vceqzq_s16,
        cge = vcgeq_s16,
        clt = vcltq_s16,
        ceq = vceqq_s16,
        orru = vorrq_u16,
        andu = vandq_u16,
        bsl = vbslq_s16
    );

    // 8-bit kernel: **unsigned** u8 [0,255] with saturating add/sub, so a local extension whose score
    // ceiling lands in [128,255] (bwa-mem2 would route it to int16) still runs in 16 lanes. Values are
    // non-negative and positions are `< 256`, so u8 holds both. This is bwa-mem2's `smithWaterman*_8`.
    //
    // Unsigned works here only because the split into `sbt_pos`/`sbt_neg` above avoids ever forming a
    // negative intermediate: `(M + a) - |mm|` saturates at 0 on the way down, which is exactly the
    // `max(0, .)` the recurrence wants, whereas a signed `M + (-b)` would need a real signed type.
    // The payoff is 16 lanes instead of 8, and mate-rescue-sized and read-sized jobs almost always
    // qualify, so this is the kernel that actually runs in production.
    //
    // bwa-mem2 keeps its 8-bit path *signed* and therefore caps at `MAX_SEQ_LEN8 = 128`
    // (`bandedSWA.h:82`); going unsigned doubles the reach to 256 and pulls many more jobs into the
    // wide kernel. Both are exact in their range, so this is a scheduling change, not a scoring one.
    define_sw_kernel!(
        batched_extend_neon_u8,
        u8,
        u8,
        16,
        feat = "neon",
        dup = vdupq_n_u8,
        lds = vld1q_u8,
        sts = vst1q_u8,
        ldu = vld1q_u8,
        add = vqaddq_u8,
        sub = vqsubq_u8,
        max = vmaxq_u8,
        ceqz = vceqzq_u8,
        cge = vcgeq_u8,
        clt = vcltq_u8,
        ceq = vceqq_u8,
        orru = vorrq_u8,
        andu = vandq_u8,
        bsl = vbslq_u8
    );
}

/// AVX2 (x86_64) instantiations of [`define_sw_kernel`]: 32 u8 lanes / 16 i16 lanes (256-bit, twice
/// NEON's width). x86 SIMD lacks unsigned integer compares and its blend has a different argument
/// order than NEON's `vbslq`, so a handful of `#[target_feature("avx2")]` wrappers adapt the ops to
/// the macro's interface. Byte-identical to the scalar reference by the same construction as NEON
/// (verified via the force-run property test compiled to x86 and executed under Rosetta).
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{clamp_band, default_result};
    use bwa_extend::{ExtendJob, ExtendResult};
    use std::arch::x86_64::*;

    // set1 (dup): the macro passes an `$elem`-typed scalar; reinterpret to the epi lane type.
    /// # Parameters
    /// - `x`: the value to place in all 32 byte lanes. The `as i8` is a pure bit reinterpretation:
    ///   the kernel treats these lanes as unsigned, so a value above 127 is intended, not a bug.
    ///
    /// # Returns
    /// A 256-bit vector whose every byte lane equals `x`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn set1_u8(x: u8) -> __m256i {
        _mm256_set1_epi8(x as i8)
    }
    /// # Parameters
    /// - `x`: the value to place in all 16 signed 16-bit lanes; negative values (`-1` for the
    ///   initial `mj`) are meaningful here, unlike in the u8 kernel.
    ///
    /// # Returns
    /// A 256-bit vector whose every 16-bit lane equals `x`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn set1_i16(x: i16) -> __m256i {
        _mm256_set1_epi16(x)
    }

    // loads/stores: the macro hands typed element pointers into `[$elem; LANES]` / `[$umask; LANES]`
    // scratch; a 256-bit unaligned move covers the whole array (32 u8 or 16 u16/i16).
    //
    // In all five: `p` is the address of the first element of a run of at least LANES elements of
    // that type (a scratch array, or `buf.as_ptr().add(j * LANES)` inside an SoA `Vec`), which the
    // kernel guarantees is in bounds. No alignment requirement: these are the unaligned forms. `v`
    // is the vector to write. Reads/writes exactly 32 bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_u8(p: *const u8) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_u16(p: *const u16) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu_i16(p: *const i16) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn storeu_u8(p: *mut u8, v: __m256i) {
        _mm256_storeu_si256(p as *mut __m256i, v)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn storeu_i16(p: *mut i16, v: __m256i) {
        _mm256_storeu_si256(p as *mut __m256i, v)
    }

    // x == 0.
    //
    // Both take `v`, a vector of DP values (in practice the diagonal predecessor H(i-1,j-1)), and
    // return the all-ones/all-zero mask "this lane is zero", which drives the local restart.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceqz8(v: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(v, _mm256_setzero_si256())
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceqz16(v: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(v, _mm256_setzero_si256())
    }

    // a >= b via `max(a,b) == a` (works unsigned via max_epu8, signed via max_epi16); a < b = !(a>=b).
    // The detour exists because AVX2 only offers *signed* integer compares (`cmpgt_epi8`), which
    // would misread any u8 lane above 127 as negative. `max_epu8` is a true unsigned max, so
    // comparing its result against `a` recovers an unsigned >=. The `xor` with all-ones is the mask
    // negation (`set1_epi8(-1)` is 0xFF in every byte, i.e. an all-ones vector, not the number -1).
    //
    // All four take `a` and `b`, two vectors of the same lane type (in the kernel: a broadcast
    // column index against a per-lane band bound, or an H against the running row max), and return
    // an all-ones/all-zero mask per lane. `cge` is `a >= b`, `clt` is `a < b`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn clt_epu8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(cge_epu8(a, b), _mm256_set1_epi8(-1))
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cge_epi16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi16(_mm256_max_epi16(a, b), a)
    }
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn clt_epi16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(cge_epi16(a, b), _mm256_set1_epi8(-1))
    }

    // blend select: NEON `vbslq(mask, a, b)` = mask ? a : b; AVX2 `blendv_epi8(a, b, mask)` = mask ? b : a.
    //
    // `mask` must be a per-lane all-ones/all-zero mask as produced by the compares above (only the
    // top bit of each byte is actually consulted, which is why the 16-bit kernel can reuse the
    // byte-granular blend: its masks are all-ones across both bytes of a lane). `a` is the value
    // taken where the mask is set, `b` where it is clear.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bsl256(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_blendv_epi8(b, a, mask)
    }

    define_sw_kernel!(
        batched_extend_avx2_i16,
        i16,
        u16,
        16,
        feat = "avx2",
        dup = set1_i16,
        lds = loadu_i16,
        sts = storeu_i16,
        ldu = loadu_u16,
        add = _mm256_add_epi16,
        sub = _mm256_sub_epi16,
        max = _mm256_max_epi16,
        ceqz = ceqz16,
        cge = cge_epi16,
        clt = clt_epi16,
        ceq = _mm256_cmpeq_epi16,
        orru = _mm256_or_si256,
        andu = _mm256_and_si256,
        bsl = bsl256
    );

    define_sw_kernel!(
        batched_extend_avx2_u8,
        u8,
        u8,
        32,
        feat = "avx2",
        dup = set1_u8,
        lds = loadu_u8,
        sts = storeu_u8,
        ldu = loadu_u8,
        add = _mm256_adds_epu8,
        sub = _mm256_subs_epu8,
        max = _mm256_max_epu8,
        ceqz = ceqz8,
        cge = cge_epu8,
        clt = clt_epu8,
        ceq = _mm256_cmpeq_epi8,
        orru = _mm256_or_si256,
        andu = _mm256_and_si256,
        bsl = bsl256
    );
}

/// Force-run verification of the AVX2 kernels against the scalar `ksw_extend2`, byte-for-byte.
///
/// On x86_64 the AVX2 path only *runs* when `is_x86_feature_detected!("avx2")`, which Rosetta
/// reports as `false` even though it *executes* AVX2 instructions. So this test calls the AVX2
/// kernels directly (bypassing detection), which is how the port is validated on this Apple-Silicon
/// host via `cargo test --target x86_64-apple-darwin` (Rosetta). On a native x86 CI runner (which has
/// AVX2) it validates the real path. Requires an AVX2-capable executor.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_verify {
    use bwa_extend::{ksw_extend2, ExtendJob};

    /// bwa's default DNA scoring matrix, built the way `bwa_fill_scmat` does.
    ///
    /// # Returns
    /// A 25-entry (`m = 5`) row-major matrix: `+1` on the diagonal (match), `-4` off-diagonal among
    /// the concrete bases (mismatch), `-1` on the whole N row and N column. This is exactly the
    /// uniform shape `is_uniform_dna` accepts, which is what routes the tests through the SIMD
    /// kernels rather than the scalar fallback.
    fn scoring() -> Vec<i8> {
        // Match bonus and mismatch penalty magnitude: bwa's `-A 1 -B 4` defaults.
        let (a, b) = (1i8, 4i8);
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

    #[test]
    fn avx2_u8_and_i16_match_scalar() {
        let mat = scoring();
        let mut state = 0xA7C2_0000_0000_0001u64;
        // Deterministic LCG (the classic Numerical Recipes 64-bit multiplier/increment), returning
        // the top 31 bits so the low-order-bit weakness of an LCG does not leak into the small
        // moduli used below. A fixed seed keeps a failure reproducible: the assert prints the round.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // bwa's default gap penalties, held fixed so the randomization varies only band/zdrop/data.
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..200u32 {
            // Randomized DP shape for this round: band half-width in cells (1..=150, so both
            // "tight enough to clip the alignment" and "wider than the sequences" occur), z-drop in
            // score units (0 included, which disables the test), and the band-clamp end bonus.
            let w = 1 + (next() % 150) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            // Jobs per call, chosen to straddle the 16/32-lane chunk boundaries: exact multiples,
            // one over, and a lone job, so partial final chunks are exercised.
            let batch = *[1usize, 8, 16, 17, 32, 33, 48]
                .get((next() % 7) as usize)
                .unwrap();

            // Keep both lengths and minval < 256 so the u8 kernel is exact, plus a longer-length set
            // (minval up to a few thousand) for the i16 kernel.
            for &big in &[false, true] {
                // Owned backing storage for the batch, one entry per job: 2-bit base codes for the
                // query and target, and the seed score the extension starts from. `ExtendJob`
                // borrows from these, so they must outlive the `jobs` vector built below.
                let mut queries: Vec<Vec<u8>> = Vec::new();
                let mut targets: Vec<Vec<u8>> = Vec::new();
                let mut h0s: Vec<i32> = Vec::new();
                for _ in 0..batch {
                    let qlen = if big {
                        200 + (next() % 400) as usize
                    } else {
                        1 + (next() % 90) as usize
                    };
                    // Random query over codes 0..=3 only (no N, so the fast path is not trivially
                    // declined), and a target a little longer so the extension has room to run off
                    // the end of the query.
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                    let tlen = qlen + (next() % 30) as usize;
                    let mut t: Vec<u8> = Vec::with_capacity(tlen);
                    // Read cursor into `q` while synthesizing the target: mostly copy the next
                    // query base (95%), otherwise emit a random base and coin-flip whether to also
                    // advance `qi`. Copying gives a mismatch, advancing as well gives an indel, so
                    // the target is a noisy relative of the query rather than unrelated noise. That
                    // matters: an unrelated target dies to z-drop in a few rows and would never
                    // reach the deep-DP code paths this test exists to check.
                    let mut qi = 0usize;
                    while t.len() < tlen {
                        if qi < q.len() && next() % 100 >= 5 {
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
                    // h0 >= 1: a zero starting score would make every cell dead from the start.
                    h0s.push(1 + (next() % 20) as i32);
                }
                let jobs: Vec<ExtendJob> = (0..batch)
                    .map(|i| ExtendJob {
                        query: &queries[i],
                        target: &targets[i],
                        h0: h0s[i],
                    })
                    .collect();
                // SAFETY: this test requires an AVX2-capable executor (native x86 CI or Rosetta).
                let got = unsafe {
                    if big {
                        super::avx2::batched_extend_avx2_i16(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    } else {
                        super::avx2::batched_extend_avx2_u8(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    }
                };
                // `got[i]` is the kernel's result for job `i`; `expected` is the authoritative
                // scalar `ksw_extend2` run on that same job alone. Every field must match exactly:
                // this is the byte-identity gate, not an approximate-score check.
                for (i, g) in got.iter().enumerate() {
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
                        "AVX2 {} diverged round {round} job {i} qlen={} tlen={}",
                        if big { "i16" } else { "u8" },
                        queries[i].len(),
                        targets[i].len()
                    );
                }
            }
        }
    }
}

/// Native-NEON byte-identity gate, runnable on this Apple-Silicon host (`cargo test -p bwa-neon`).
///
/// The `avx2_verify` test validates the shared `define_sw_kernel!` logic, but only on x86 (under
/// Rosetta / native x86 CI). This module exercises the actual NEON u8 (16-lane) and i16 (8-lane)
/// kernels that ship on aarch64, asserting every `ExtendResult` field matches the scalar
/// `ksw_extend2` reference across randomized short/long jobs and batch sizes that straddle the lane
/// boundary. It is the on-device counterpart to `avx2_verify`.
#[cfg(all(test, target_arch = "aarch64"))]
mod neon_verify {
    use super::neon::{batched_extend_neon_i16, batched_extend_neon_u8};
    use bwa_extend::{ksw_extend2, ExtendJob};

    /// bwa's default DNA scoring matrix, built the way `bwa_fill_scmat` does.
    ///
    /// # Returns
    /// A 25-entry (`m = 5`) row-major matrix: `+1` on the diagonal (match), `-4` off-diagonal among
    /// the concrete bases (mismatch), `-1` on the whole N row and N column. This is exactly the
    /// uniform shape `is_uniform_dna` accepts, which is what routes the tests through the SIMD
    /// kernels rather than the scalar fallback.
    fn scoring() -> Vec<i8> {
        // Match bonus and mismatch penalty magnitude: bwa's `-A 1 -B 4` defaults.
        let (a, b) = (1i8, 4i8);
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

    #[test]
    fn neon_u8_and_i16_match_scalar() {
        let mat = scoring();
        let mut state = 0x1234_5678_9abc_def1u64;
        // Deterministic LCG (the classic Numerical Recipes 64-bit multiplier/increment), returning
        // the top 31 bits so the low-order-bit weakness of an LCG does not leak into the small
        // moduli used below. A fixed seed keeps a failure reproducible: the assert prints the round.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // bwa's default gap penalties, held fixed so the randomization varies only band/zdrop/data.
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..400u32 {
            // Randomized DP shape for this round: band half-width in cells (1..=150, so both
            // "tight enough to clip the alignment" and "wider than the sequences" occur), z-drop in
            // score units (0 included, which disables the test), and the band-clamp end bonus.
            let w = 1 + (next() % 150) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            // Straddle the 8/16-lane boundaries: partial and empty tail lanes, exact multiples.
            let batch = *[1usize, 7, 8, 9, 16, 17, 31, 32, 33, 48]
                .get((next() % 10) as usize)
                .unwrap();

            for &big in &[false, true] {
                // Owned backing storage for the batch, one entry per job: 2-bit base codes for the
                // query and target, and the seed score the extension starts from. `ExtendJob`
                // borrows from these, so they must outlive the `jobs` vector built below.
                let mut queries: Vec<Vec<u8>> = Vec::new();
                let mut targets: Vec<Vec<u8>> = Vec::new();
                let mut h0s: Vec<i32> = Vec::new();
                for _ in 0..batch {
                    // u8 kernel: lengths + score ceiling < 256; i16 kernel: a longer-length set.
                    let qlen = if big {
                        200 + (next() % 400) as usize
                    } else {
                        1 + (next() % 90) as usize
                    };
                    // Random query over codes 0..=3 only (no N, so the fast path is not trivially
                    // declined), and a target a little longer so the extension has room to run off
                    // the end of the query.
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                    let tlen = qlen + (next() % 30) as usize;
                    let mut t: Vec<u8> = Vec::with_capacity(tlen);
                    // Read cursor into `q` while synthesizing the target: mostly copy the next
                    // query base (95%), otherwise emit a random base and coin-flip whether to also
                    // advance `qi`. Copying gives a mismatch, advancing as well gives an indel, so
                    // the target is a noisy relative of the query rather than unrelated noise. That
                    // matters: an unrelated target dies to z-drop in a few rows and would never
                    // reach the deep-DP code paths this test exists to check.
                    let mut qi = 0usize;
                    while t.len() < tlen {
                        if qi < q.len() && next() % 100 >= 5 {
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
                    // h0 >= 1: a zero starting score would make every cell dead from the start.
                    h0s.push(1 + (next() % 20) as i32);
                }
                let jobs: Vec<ExtendJob> = (0..batch)
                    .map(|i| ExtendJob {
                        query: &queries[i],
                        target: &targets[i],
                        h0: h0s[i],
                    })
                    .collect();
                // SAFETY: this host has NEON (aarch64).
                let got = unsafe {
                    if big {
                        batched_extend_neon_i16(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    } else {
                        batched_extend_neon_u8(
                            &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                        )
                    }
                };
                // `got[i]` is the kernel's result for job `i`; `expected` is the authoritative
                // scalar `ksw_extend2` run on that same job alone. Every field must match exactly:
                // this is the byte-identity gate, not an approximate-score check.
                for (i, g) in got.iter().enumerate() {
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
                        "NEON {} diverged round {round} job {i} (batch {batch}) qlen={} tlen={}",
                        if big { "i16" } else { "u8" },
                        queries[i].len(),
                        targets[i].len()
                    );
                }
            }
        }
    }
}

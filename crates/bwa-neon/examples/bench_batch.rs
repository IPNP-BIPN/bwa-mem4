//! Micro-benchmark: batched NEON extend vs the scalar per-lane loop, on realistic
//! seed-extension sizes (query up to a 150 bp read side, reference window a bit longer).
//!
//! Run: `cargo run --release -p bwa-neon --example bench_batch`
//!
//! This is a perf probe for phase 9a (the gate requires a *measured* speedup); it does not
//! assert byte-identity (the `assert_backend_batch_matches_scalar` gate does that).

use bwa_extend::{ExtendJob, ScalarBackend, SwBackend};
use bwa_neon::NeonBackend;
use std::time::Instant;

/// Build the 5x5 substitution matrix both backends are benchmarked with, exactly as bwa's
/// `bwa_fill_scmat` does at its defaults.
///
/// # Parameters
///
/// None: the scoring is hard-coded to bwa's defaults so the two backends are always compared under
/// the same, representative scheme.
///
/// # Returns
///
/// 25 entries, row-major over the code alphabet `A,C,G,T,N`, indexed `mat[t * 5 + q]` to score
/// target base `t` against query base `q` in score points: `+1` on the diagonal (match), `-4` off
/// it (mismatch), `-1` on the whole N row and N column.
fn build_scoring() -> Vec<i8> {
    // Match reward and mismatch magnitude, bwa's `-A 1` and `-B 4`.
    let (a, b) = (1i8, 4i8);
    // `k` is the row-major write cursor over the 25 entries.
    let mut mat = vec![0i8; 25];
    let mut k = 0;
    // Rows 0..4 are A/C/G/T: four scored entries, then that row's N column.
    for i in 0..4 {
        for j in 0..4 {
            mat[k] = if i == j { a } else { -b };
            k += 1;
        }
        mat[k] = -1;
        k += 1;
    }
    // Row 4 is the N row: -1 against everything, including another N.
    for _ in 0..5 {
        mat[k] = -1;
        k += 1;
    }
    mat
}

fn main() {
    let mat = build_scoring();
    // Affine gap penalties as positive magnitudes, bwa's defaults `-O 6 -E 1` for both deletions
    // (gap in the query) and insertions (gap in the target): a k-base gap costs `6 + k * 1` points.
    let (o_del, e_del, o_ins, e_ins) = (6i32, 1i32, 6i32, 1i32);
    // Band half-width in cells (`-w`), end bonus in score points (`-L`), and the z-drop threshold in
    // score points (`-d`), all at bwa's defaults so the measured cost is the cost the aligner pays.
    let (w, end_bonus, zdrop) = (100i32, 5i32, 100i32);

    // Deterministic LCG. `state` is the generator word; `next()` advances it and returns the top 31
    // bits. The fixed seed means both backends see byte-identical inputs on every run, so a
    // reported speedup is not an artifact of one backend getting easier work.
    let mut state = 0xDEAD_BEEF_1234_5678u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    // A large pool of realistic seed-extension jobs: query ~ one side of a 150 bp read,
    // target the reference window (query length + gap slack), correlated content so the DP
    // does real work (not an all-mismatch early z-drop).
    // Number of synthetic seed extensions in the pool. Large enough that the per-batch call
    // overhead is amortised and the working set does not fit in L1/L2 (so the measurement includes
    // realistic memory behaviour), small enough to build in well under a second.
    const N_JOBS: usize = 20_000;
    // Per-job inputs, index-aligned: job `i` is (queries[i], targets[i], h0s[i]). Owned here
    // because `ExtendJob` only borrows them.
    let mut queries: Vec<Vec<u8>> = Vec::with_capacity(N_JOBS);
    let mut targets: Vec<Vec<u8>> = Vec::with_capacity(N_JOBS);
    let mut h0s: Vec<i32> = Vec::with_capacity(N_JOBS);
    // Set the `UNIFORM` env var (to anything) to pin every query to 150 bp. Length uniformity is
    // the dominant factor in the batched speedup: with mixed lengths a batch runs until its longest
    // job finishes, so short lanes idle. The two modes bracket the real speedup (~1.4x mixed,
    // ~2.1x uniform).
    let uniform = std::env::var("UNIFORM").is_ok();
    for _ in 0..N_JOBS {
        let qlen = if uniform {
            150
        } else {
            40 + (next() % 110) as usize
        }; // 40..150
        let tlen = qlen + (next() % 40) as usize; // a bit longer than the query
                                                  // The read side: uniform random 2-bit base codes 0..=3 (this generator never emits 4/N).
        let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
        // Target = query with ~3% substitutions and occasional indels, padded to tlen.
        let mut t: Vec<u8> = Vec::with_capacity(tlen);
        // Cursor into `q` for the mutation walk. Invariant at the top of each iteration: `t` is a
        // ~95%-faithful copy of `q[..qi]`, so target offset `t.len()` is the mutated image of query
        // offset `qi` and the alignment stays near the main diagonal, which is what makes the DP
        // run deep instead of z-dropping out in the first few rows.
        let mut qi = 0usize;
        while t.len() < tlen {
            if qi < q.len() {
                // Percentile draw 0..=99 selecting the edit to apply at this position: 3%
                // substitution, 1% deletion, 1% insertion, 95% exact copy.
                let r = next() % 100;
                if r < 3 {
                    t.push((next() % 4) as u8); // substitution
                    qi += 1;
                } else if r < 4 {
                    qi += 1; // deletion (skip a query base)
                } else if r < 5 {
                    t.push((next() % 4) as u8); // insertion (extra target base)
                } else {
                    t.push(q[qi]);
                    qi += 1;
                }
            } else {
                // Query exhausted: pad out to `tlen` with unrelated reference bases, the trailing
                // window the real aligner also has to reject.
                t.push((next() % 4) as u8);
            }
        }
        queries.push(q);
        targets.push(t);
        h0s.push(20 + (next() % 20) as i32); // seedlen*a-ish
    }

    // The borrowed view handed to both backends: identical slice contents for each, so the only
    // difference between the two timings is the kernel.
    let jobs: Vec<ExtendJob> = (0..N_JOBS)
        .map(|i| ExtendJob {
            query: &queries[i],
            target: &targets[i],
            h0: h0s[i],
        })
        .collect();

    // The baseline (a plain per-job loop over `ksw_extend2`) and the backend under test.
    let scalar = ScalarBackend;
    let neon = NeonBackend;

    // Jobs handed to one `extend_batch` call. 8 is exactly one int16 vector; the larger sizes let
    // the kernel's length-binning group similar jobs together, which is where the extra speedup
    // comes from, at the cost of more per-call buffering.
    for &batch in &[8usize, 16, 32, 64] {
        // Warm up + correctness spot check on the first batch.
        // `s0` / `n0`: the two backends' results for the same first `batch` jobs, compared below.
        // Not a substitute for the byte-identity gate, just a guard against benchmarking a kernel
        // that is fast because it is wrong.
        let s0 = scalar.extend_batch(
            &jobs[..batch],
            5,
            &mat,
            o_del,
            e_del,
            o_ins,
            e_ins,
            w,
            end_bonus,
            zdrop,
        );
        let n0 = neon.extend_batch(
            &jobs[..batch],
            5,
            &mat,
            o_del,
            e_del,
            o_ins,
            e_ins,
            w,
            end_bonus,
            zdrop,
        );
        assert_eq!(s0, n0, "batched result diverged from scalar");

        // Full passes over the 20 000-job pool per timing, averaged in the printout. Enough to
        // smooth out scheduler noise without making the run long.
        let reps = 4;
        // Wall-clock `Duration` for `reps` complete scalar passes over the pool, measured with a
        // monotonic clock. Includes the per-call allocation of the result vectors, which is
        // deliberate: the aligner pays that too.
        let t_scalar = {
            let start = Instant::now();
            // Sum of every returned alignment score. Meaningless as a number; it exists so the
            // `black_box` below can make the results observable, stopping the optimiser from
            // deleting the DP calls whose cost is the whole point of the measurement.
            let mut acc = 0i64;
            for _ in 0..reps {
                for chunk in jobs.chunks(batch) {
                    let r = scalar.extend_batch(
                        chunk, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                    );
                    acc += r.iter().map(|x| i64::from(x.score)).sum::<i64>();
                }
            }
            std::hint::black_box(acc);
            start.elapsed()
        };
        // The same measurement for the NEON backend: `reps` passes over the identical job pool at
        // the identical batch size, so `t_scalar / t_neon` is a like-for-like ratio.
        let t_neon = {
            let start = Instant::now();
            // Same anti-dead-code accumulator as above.
            let mut acc = 0i64;
            for _ in 0..reps {
                for chunk in jobs.chunks(batch) {
                    let r = neon.extend_batch(
                        chunk, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                    );
                    acc += r.iter().map(|x| i64::from(x.score)).sum::<i64>();
                }
            }
            std::hint::black_box(acc);
            start.elapsed()
        };

        // Speedup, dimensionless: scalar time divided by NEON time, so > 1 means NEON is faster.
        // The `reps` division cancels, hence the raw totals here and the per-pass ms below.
        let sp = t_scalar.as_secs_f64() / t_neon.as_secs_f64();
        // One line per batch size. Both times are milliseconds for a single pass over all
        // `N_JOBS` extensions (`as_secs_f64() * 1e3` converts to ms, `/ reps` averages the passes).
        println!(
            "batch={batch:>3}  scalar={:>8.2}ms  neon={:>8.2}ms  speedup={sp:>5.2}x",
            t_scalar.as_secs_f64() * 1e3 / f64::from(reps),
            t_neon.as_secs_f64() * 1e3 / f64::from(reps),
        );
    }
}

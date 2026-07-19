//! Does `get_sa_batch`'s BATCH SIZE explain why get_sa costs ~177 ns/lookup when the hardware
//! sustains random gathers into 16 GB at ~4 ns?
//!
//! `build_chains_from_smems` is called per read, so it hands `get_sa_batch` only ~42 positions
//! (21.3M lookups / 500k reads). With W=32 that is 1.3 windows: the lockstep pipeline never reaches
//! steady state and the core's ~28 memory-level-parallelism lanes sit idle.
//!
//! This calls the SAME function with the SAME total work, chunked differently. Any difference is
//! batch size alone.
//!
//! Reading the output. One row per chunk size, plus a final `ALL` row (one call for all 2M lookups):
//!   - `chunk`     : positions handed to a single `get_sa_batch` call.
//!   - `wall(s)`   : time for the whole 2M-lookup sweep at that chunking.
//!   - `ns/lookup` : the number to compare against the ~177 ns the aligner actually pays, and against
//!                   the ~4 ns floor for a single random gather into 16 GB on this host. A `get_sa`
//!                   is ~7 DEPENDENT misses when the position is not an 1-in-8 SA sample, so ~28 ns
//!                   is the latency-bound floor and anything near it means the pipeline is full.
//!   - `vs 42`     : speedup over the chunk=42 row, i.e. over what the production caller supplies.
//!
//! What the answer looks like. If `vs 42` climbs materially with chunk size and plateaus, batch size
//! IS the lever and the fix is to hoist `get_sa_batch` above the per-read loop so it sees thousands
//! of positions. If the column stays near 1.00x, the 177 ns is NOT a pipeline-fill problem and this
//! whole line of attack is dead: look at the LF-walk depth or the SA sampling rate instead.
//!
//! The last two blocks are correctness gates, not timing: this example doubles as the only coverage
//! `get_sa_batch` has at genome scale.
//!
//!   cargo run --release -p bwa-index --example sa_batch_bench -- work/genome.fa
//!
//! The argument is the INDEX PREFIX (the FASTA path the index files sit next to), not the FASTA
//! content: only `<prefix>.bwt.2bit.64` and `<prefix>.0123` are read.
use bwa_index::FmIndex;
use std::time::Instant;

fn main() {
    // argv[1]: the index PREFIX, i.e. the path the index side-files sit next to (typically the FASTA
    // path itself). Supplied by the operator on the command line; no default, we abort if absent.
    let prefix = std::env::args()
        .nth(1)
        .expect("usage: sa_batch_bench <index prefix>");
    // The loaded FM index. `load` mmaps `<prefix>.bwt.2bit.64` (BWT + checkpointed occurrence table
    // + the 1-in-8 sampled suffix array) and `<prefix>.0123` (the 2-bit forward+RC reference).
    // Owned for the whole run; every timed call below borrows it.
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    // Length of the FMD reference in BASES, counting the forward strand and its reverse complement,
    // so ~6.2e9 for GRCh38. It is also the number of ROWS in the BWT / suffix array, which is why it
    // is the correct modulus for generating random SA rows below.
    let n = fm.ref_seq_len;
    println!("ref_seq_len = {n}");

    // Positions an aligner actually asks for are SA-interval rows, i.e. essentially uniform over the
    // BWT. A cheap xorshift keeps generation out of the timed region.
    // 2M lookups is ~10% of a 500k-read run's 21.3M, big enough that per-call overhead and the timer
    // are noise, small enough that the position array (16 MB) does not itself perturb the cache.
    // TOTAL: number of SA lookups performed by EVERY row of the sweep. Fixed across chunk sizes so
    // that the only variable is chunking. Raising it costs 8 B/entry in each of three i64 vectors
    // plus proportional wall time; lowering it below ~1e5 lets timer granularity and per-call
    // overhead leak into the ns/lookup figure.
    const TOTAL: usize = 2_000_000;
    // xorshift64 state. The literal is the first 64 bits of the fractional part of pi (a standard
    // arbitrary nonzero seed); it is FIXED so the position list, and therefore the run, is
    // reproducible. Any nonzero seed would do: nothing here depends on the specific stream.
    let mut s = 0x243f_6a88_85a3_08d3u64;
    // The query set: TOTAL suffix-array ROW indices (0..n), NOT reference positions. A row is a rank
    // in the BWT; the reference position it denotes is precisely what `get_sa` computes. Generated
    // once, outside every timed region, and reused verbatim by every chunk size.
    let positions: Vec<i64> = (0..TOTAL)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % (n as u64)) as i64
        })
        .collect();
    // Destination buffer, parallel to `positions`: out[i] receives the reference POSITION (in bases,
    // 0..n, forward+RC coordinate space) that SA row positions[i] points at. Preallocated so no
    // timed region ever pays an allocation.
    let mut out = vec![0i64; TOTAL];

    // Warm: fault the pages and settle the frequency before ANY timing. Skipping this is exactly the
    // cold-start trap that produced two bogus 14-20x "speedups" earlier in this session.
    // 200_000 = 10% of TOTAL: enough touched pages and enough elapsed time (tens of ms) for the core
    // to reach its sustained clock, while costing a tenth of one sweep row. The result is discarded.
    fm.get_sa_batch(&positions[..200_000], &mut out[..200_000]);

    println!(
        "\n{:<12} {:>12} {:>14} {:>10}",
        "chunk", "wall(s)", "ns/lookup", "vs 42"
    );
    // ns/lookup of the chunk=42 row, i.e. the production caller's cost. Set on the first iteration
    // and read by every later one as the denominator of the `vs 42` column. The 0.0 initializer is
    // never observed, because 42 is the first element of the sweep array below.
    let mut base = 0f64;
    // 42 must stay FIRST: it is the production caller's batch size and `base` is captured on it, so
    // the `vs 42` column is "how much would we gain by batching harder than we do today".
    for &chunk in &[42usize, 128, 512, 2048, 8192, 32768, TOTAL] {
        // UNVERIFIED: this comment claims an interleaved reference run, but no such run exists in the
        // loop below; each chunk size is timed once, in ascending order. Drift over the sweep would
        // therefore bias the later (larger) chunks. Treat a `vs 42` under ~1.1x as inside the noise.
        // Timed region: the whole TOTAL-lookup sweep including per-call entry/exit, excluding the
        // position generation and the allocation of `out`. Invariant at the top of the inner loop:
        // rows positions[0..c] have been resolved into out[0..c], and `c` is a multiple of `chunk`.
        let t = Instant::now();
        for c in (0..TOTAL).step_by(chunk) {
            // Last chunk is short whenever `chunk` does not divide TOTAL, hence the clamp.
            let hi = (c + chunk).min(TOTAL);
            fm.get_sa_batch(&positions[c..hi], &mut out[c..hi]);
        }
        // TOTAL-lookup wall time in seconds, then the same figure amortized to nanoseconds per
        // single SA-row lookup. `ns` is what is compared against the aligner's ~177 ns and against
        // the ~28 ns dependent-miss floor.
        let secs = t.elapsed().as_secs_f64();
        let ns = secs * 1e9 / TOTAL as f64;
        if chunk == 42 {
            base = ns;
        }
        println!(
            "{:<12} {:>12.3} {:>14.1} {:>9.2}x",
            if chunk == TOTAL {
                "ALL".to_string()
            } else {
                chunk.to_string()
            },
            secs,
            ns,
            base / ns
        );
    }
    // get_sa_batch has no unit test, and it carries a lockstep walk plus a deferred-read pipeline.
    // Gate it against the per-position reference on real genome-scale positions.
    // Arm under test: reference positions produced by ONE batched call over all TOTAL rows.
    let mut batched = vec![0i64; TOTAL];
    fm.get_sa_batch(&positions, &mut batched);
    // Count of rows where the batched answer differs from the scalar `get_sa` reference. Must end at
    // 0; the first 3 offenders are printed so a failure is diagnosable without a rerun.
    let mut bad = 0usize;
    for (i, &p) in positions.iter().enumerate() {
        if batched[i] != fm.get_sa(p) {
            if bad < 3 {
                eprintln!(
                    "  MISMATCH at pos {p}: batch={} vs get_sa={}",
                    batched[i],
                    fm.get_sa(p)
                );
            }
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "get_sa_batch disagrees with get_sa on {bad}/{TOTAL} positions"
    );
    println!("\nget_sa_batch == per-position get_sa on {TOTAL} genome positions: OK");

    // Prove the arms agree: same values regardless of chunking (byte-identity of the lever).
    // Distinct from the check above. That one gates batch-vs-scalar; this one gates the specific
    // property the proposed optimization depends on, namely that re-chunking is observationally
    // free. `get_sa_batch` retires slots out of order as their LF-walks terminate, so a bug in the
    // slot-compaction bookkeeping would show up here as a chunk-size-dependent value.
    // Control: positions resolved in a single maximal-width call (chunk = TOTAL).
    let mut ref_out = vec![0i64; TOTAL];
    fm.get_sa_batch(&positions, &mut ref_out);
    // Variable: the same rows resolved 42 at a time, the production chunking. `ref_out` and
    // `chunked` must be element-wise equal, which is the property "re-chunking is observationally
    // free" that the proposed optimization rests on.
    let mut chunked = vec![0i64; TOTAL];
    for c in (0..TOTAL).step_by(42) {
        let hi = (c + 42).min(TOTAL);
        fm.get_sa_batch(&positions[c..hi], &mut chunked[c..hi]);
    }
    assert_eq!(
        ref_out, chunked,
        "chunking changed a value -- the lever is NOT byte-identical"
    );
    println!("\nvalues identical across chunk sizes: OK ({TOTAL} lookups)");
}

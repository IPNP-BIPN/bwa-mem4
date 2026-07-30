//! Byte-identity + speed/RAM spike for a CaPS-SA suffix-array backend (`capsa` feature).
//!
//! A suffix array is UNIQUE, so CaPS-SA can only be a drop-in for the in-tree SA-IS if it produces
//! exactly the same array. This builds a synthetic `bref` (codes 0..=3, the same alphabet the real 2L
//! reference uses), runs the in-tree SA-IS as ground truth, then CaPS-SA in BOTH modes:
//!   * `build_in_memory` (parallel sample-sort, in RAM), and
//!   * `build_ext_mem`    (external-memory, bounded RAM),
//! asserts each is identical element-for-element to SA-IS, and times all three. If parity holds, every
//! downstream index file (derived deterministically from `sa`) is identical too.
//!
//!   cargo run --release -p bwa-mem4-index --features capsa --example capsa_parity -- 100000000
//!
//! The argument is the number of bases (default 20M). Use a large value on a QUIET machine for a
//! representative speed ratio; small values just prove parity. CaPS-SA returns the standard lexicographic
//! SA with NO sentinel, so we prepend `sa[0] = n` to match SA-IS's layout (length `n+1`, `sa[0] = n`).
use bwa_index::sais::suffix_array_inplace;
use std::hint::black_box;
use std::time::Instant;

/// RAM-isolation mode: build ONE constructor's SA and hold it live, so an external `/usr/bin/time -l`
/// attributes peak RSS to that constructor (over the common `bref` baseline). Returns the elapsed time.
/// `which` in {sais, libsais, capsa-mem, capsa-ext}. Prints time; caller reads max RSS from the wrapper.
fn run_one(bref: &[u8], which: &str) {
    let n = bref.len();
    let t = Instant::now();
    // Each backend black_boxes its NATIVE output (i64 for SA-IS/libsais, u64 for CaPS-SA), both 8
    // bytes/entry, so peak RSS reflects the constructor's own footprint, not an extra i64 copy. The
    // `capsa-ext` collect-into-Vec is only for this harness; the algorithm itself never holds the whole
    // SA (it spills to temp files), so its true bounded-RAM win needs the emit callback to stream to
    // disk rather than to a Vec. Here the Vec dominates its RSS, so `capsa-ext` is an UPPER bound.
    match which {
        "sais" => {
            let sa = suffix_array_inplace(bref);
            report(which, n, t);
            black_box(&sa);
        }
        #[cfg(feature = "libsais")]
        "libsais" => {
            let mut sa = vec![0i64; n + 1];
            let ret = libsais_rs::libsais64::libsais64(bref, &mut sa[1..], 0, None);
            assert_eq!(ret, 0, "libsais64 failed");
            sa[0] = n as i64;
            report(which, n, t);
            black_box(&sa);
        }
        "capsa-mem" => {
            let sa: Vec<u64> = caps_sa::build_in_memory(bref);
            report(which, n, t);
            black_box(&sa);
        }
        "capsa-ext" => {
            let opts = caps_sa::ExtMemOpts::default();
            let mut sa: Vec<u64> = Vec::with_capacity(n);
            caps_sa::build_ext_mem(bref, &opts, |pos| {
                sa.push(pos);
                Ok(())
            })
            .expect("build_ext_mem failed");
            report(which, n, t);
            black_box(&sa);
        }
        // True bounded-RAM: the emit callback checksums positions instead of storing them, so RSS is the
        // algorithm's own working set (bref + subproblem buffers), independent of n. This is what wiring
        // ext-mem straight into the `.sa` file writer would look like: stream SA entries to disk, never
        // materialize the whole array. Not parity-checked here (parity is proven in full-compare mode).
        "capsa-ext-stream" => {
            let opts = caps_sa::ExtMemOpts::default();
            let mut checksum: u64 = 0;
            caps_sa::build_ext_mem(bref, &opts, |pos| {
                checksum = checksum.wrapping_add(pos).rotate_left(1);
                Ok(())
            })
            .expect("build_ext_mem failed");
            report(which, n, t);
            black_box(checksum);
        }
        other => {
            eprintln!("unknown backend {other:?}; use sais|libsais|capsa-mem|capsa-ext");
            std::process::exit(2);
        }
    }
}

fn report(which: &str, n: usize, t: Instant) {
    println!("[{which}] {n} bases: {:.3}s", t.elapsed().as_secs_f64());
}

/// Wrap a CaPS-SA `Vec<u64>` SA (length n, no sentinel) into SA-IS's `Vec<i64>` layout: length n+1 with
/// the empty/sentinel suffix at row 0 (`sa[0] = n`), `sa[1..]` the suffix array of `bref`.
fn with_sentinel(caps: Vec<u64>, n: usize) -> Vec<i64> {
    let mut sa = vec![0i64; n + 1];
    sa[0] = n as i64;
    for (dst, &src) in sa[1..].iter_mut().zip(caps.iter()) {
        *dst = src as i64;
    }
    sa
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    // Synthetic DNA: xorshift64 -> 2 bits per base, codes 0..=3. Deterministic so runs are comparable
    // (identical generator to libsais_parity so the two spikes are directly comparable).
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let bref: Vec<u8> = (0..n).map(|_| (rng() & 3) as u8).collect();

    // RAM-isolation mode: `capsa_parity <n> <backend>` runs ONE constructor so an external
    // `/usr/bin/time -l` reads its peak RSS. No second arg -> full parity + speed comparison.
    if let Some(which) = std::env::args().nth(2) {
        run_one(&bref, &which);
        return;
    }
    println!("bref: {n} bases (codes 0..=3)");

    let t = Instant::now();
    let sa_ours = suffix_array_inplace(&bref);
    let dt_ours = t.elapsed();
    println!("SA-IS (in-tree):        {:.3}s", dt_ours.as_secs_f64());

    // CaPS-SA in-memory (parallel sample-sort). I = u64 to cover 2L > 2^32 at full-genome scale.
    let t = Instant::now();
    let sa_mem_raw: Vec<u64> = caps_sa::build_in_memory(&bref);
    let dt_mem = t.elapsed();
    let sa_mem = with_sentinel(sa_mem_raw, n);
    println!("CaPS-SA in-memory:      {:.3}s", dt_mem.as_secs_f64());

    assert_eq!(sa_ours.len(), sa_mem.len(), "in-memory length mismatch");
    assert!(
        sa_ours == sa_mem,
        "CaPS-SA in-memory SA differs from SA-IS -> NOT byte-identical"
    );
    println!("  in-memory parity vs SA-IS: OK");

    // CaPS-SA external-memory (bounded RAM): streams SA positions in lexicographic order through an
    // emit callback, spilling to temp files (default `ExtMemOpts` -> system temp dir). We collect them
    // back into a Vec purely to check parity; the point of ext-mem is that the algorithm itself never
    // holds the whole SA in RAM. Report the error rather than panic so the spike still yields the
    // in-memory numbers if ext-mem needs setup.
    let opts = caps_sa::ExtMemOpts::default();
    let mut sa_ext_raw: Vec<u64> = Vec::with_capacity(n);
    let t = Instant::now();
    let res = caps_sa::build_ext_mem(&bref, &opts, |pos| {
        sa_ext_raw.push(pos);
        Ok(())
    });
    match res {
        Ok(()) => {
            let dt_ext = t.elapsed();
            let sa_ext = with_sentinel(sa_ext_raw, n);
            println!("CaPS-SA external-memory: {:.3}s", dt_ext.as_secs_f64());
            assert_eq!(sa_ours.len(), sa_ext.len(), "ext-mem length mismatch");
            assert!(
                sa_ours == sa_ext,
                "CaPS-SA ext-mem SA differs from SA-IS -> NOT byte-identical"
            );
            println!("  external-memory parity vs SA-IS: OK");
        }
        Err(e) => {
            println!("CaPS-SA external-memory: skipped ({e})");
        }
    }

    println!(
        "speedup (SA-IS / CaPS-SA in-memory): {:.2}x",
        dt_ours.as_secs_f64() / dt_mem.as_secs_f64()
    );
}

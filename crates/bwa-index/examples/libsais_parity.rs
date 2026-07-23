//! Byte-identity + speed gate for the `libsais` feature's suffix-array path.
//!
//! A suffix array is UNIQUE, so `--features libsais` can only be byte-identical if libsais-rs produces
//! exactly the array the in-tree SA-IS does. This builds a synthetic `bref` (codes 0..=3, the same
//! alphabet the real 2L reference uses), runs BOTH constructors on it, asserts the results are
//! identical element-for-element, and times each so the build-speed ratio is visible. If this passes,
//! every downstream index file (derived deterministically from `sa`) is identical too.
//!
//!   cargo run --release -p bwa-mem4-index --features libsais --example libsais_parity -- 100000000
//!
//! The argument is the number of bases (default 20M). Use a large value on a QUIET machine for a
//! representative speed ratio; small values just prove parity.
use bwa_index::sais::suffix_array_inplace;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    // Synthetic DNA: xorshift64 -> 2 bits per base, codes 0..=3. Deterministic so runs are comparable.
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let bref: Vec<u8> = (0..n).map(|_| (rng() & 3) as u8).collect();
    println!("bref: {n} bases (codes 0..=3)");

    let t = Instant::now();
    let sa_ours = suffix_array_inplace(&bref);
    let dt_ours = t.elapsed();
    println!("SA-IS (in-tree):  {:.3}s", dt_ours.as_secs_f64());

    // libsais writes the n-length SA into sa[1..]; sa[0] = n is the empty/sentinel suffix, matching
    // suffix_array_inplace's layout exactly.
    let t = Instant::now();
    let mut sa_lib = vec![0i64; n + 1];
    let ret = libsais_rs::libsais64::libsais64(&bref, &mut sa_lib[1..], 0, None);
    assert_eq!(ret, 0, "libsais64 failed");
    sa_lib[0] = n as i64;
    let dt_lib = t.elapsed();
    println!("libsais-rs:       {:.3}s", dt_lib.as_secs_f64());

    assert_eq!(sa_ours.len(), sa_lib.len(), "length mismatch");
    let identical = sa_ours == sa_lib;
    println!(
        "suffix arrays identical: {}",
        if identical { "OK" } else { "MISMATCH" }
    );
    assert!(
        identical,
        "libsais SA differs from SA-IS -> NOT byte-identical"
    );
    println!(
        "speedup (SA-IS / libsais): {:.2}x",
        dt_ours.as_secs_f64() / dt_lib.as_secs_f64()
    );
}

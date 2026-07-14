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

fn build_scoring() -> Vec<i8> {
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
    mat
}

fn main() {
    let mat = build_scoring();
    let (o_del, e_del, o_ins, e_ins) = (6i32, 1i32, 6i32, 1i32);
    let (w, end_bonus, zdrop) = (100i32, 5i32, 100i32);

    // Deterministic LCG.
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
    const N_JOBS: usize = 20_000;
    let mut queries: Vec<Vec<u8>> = Vec::with_capacity(N_JOBS);
    let mut targets: Vec<Vec<u8>> = Vec::with_capacity(N_JOBS);
    let mut h0s: Vec<i32> = Vec::with_capacity(N_JOBS);
    let uniform = std::env::var("UNIFORM").is_ok();
    for _ in 0..N_JOBS {
        let qlen = if uniform {
            150
        } else {
            40 + (next() % 110) as usize
        }; // 40..150
        let tlen = qlen + (next() % 40) as usize; // a bit longer than the query
        let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
        // Target = query with ~3% substitutions and occasional indels, padded to tlen.
        let mut t: Vec<u8> = Vec::with_capacity(tlen);
        let mut qi = 0usize;
        while t.len() < tlen {
            if qi < q.len() {
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
                t.push((next() % 4) as u8);
            }
        }
        queries.push(q);
        targets.push(t);
        h0s.push(20 + (next() % 20) as i32); // seedlen*a-ish
    }

    let jobs: Vec<ExtendJob> = (0..N_JOBS)
        .map(|i| ExtendJob {
            query: &queries[i],
            target: &targets[i],
            h0: h0s[i],
        })
        .collect();

    let scalar = ScalarBackend;
    let neon = NeonBackend;

    for &batch in &[8usize, 16, 32, 64] {
        // Warm up + correctness spot check on the first batch.
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

        let reps = 4;
        let t_scalar = {
            let start = Instant::now();
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
        let t_neon = {
            let start = Instant::now();
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

        let sp = t_scalar.as_secs_f64() / t_neon.as_secs_f64();
        println!(
            "batch={batch:>3}  scalar={:>8.2}ms  neon={:>8.2}ms  speedup={sp:>5.2}x",
            t_scalar.as_secs_f64() * 1e3 / f64::from(reps),
            t_neon.as_secs_f64() * 1e3 / f64::from(reps),
        );
    }
}

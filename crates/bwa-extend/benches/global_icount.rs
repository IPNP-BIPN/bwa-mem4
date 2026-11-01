//! Instruction counts for `ksw_global2`, with and without `tightened_global_band`. Exact and
//! independent of host load, so a 1% change is visible where a wall-clock A/B cannot see it.
use bwa_mem4_extend::{ksw_global2, tightened_global_band};
use gungraun::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

fn mat() -> [i8; 25] {
    let mut m = [-4i8; 25];
    for i in 0..4 {
        m[i * 5 + i] = 1;
    }
    for i in 0..5 {
        m[i * 5 + 4] = -1;
        m[4 * 5 + i] = -1;
    }
    m
}

/// A 150 bp pair with `subs` mismatches and, if `del`, a 2 bp deletion from the read.
fn pair(subs: usize, del: bool) -> (Vec<u8>, Vec<u8>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let t: Vec<u8> = (0..150)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s & 3) as u8
        })
        .collect();
    let mut q = t.clone();
    for k in 0..subs {
        let p = 11 + k * 130 / subs.max(1);
        q[p] = (q[p] + 1) % 4;
    }
    if del {
        q.drain(70..72);
    }
    (q, t)
}

#[library_benchmark]
#[bench::three_subs(pair(3, false), 11)]
#[bench::six_subs(pair(6, false), 26)]
#[bench::deletion(pair(1, true), 8)]
fn full_band(p: (Vec<u8>, Vec<u8>), w: i32) -> i32 {
    let m = mat();
    black_box(ksw_global2(&p.0, &p.1, 5, &m, 6, 1, 6, 1, w)).0
}

#[library_benchmark]
#[bench::three_subs(pair(3, false), 11)]
#[bench::six_subs(pair(6, false), 26)]
#[bench::deletion(pair(1, true), 8)]
fn tightened(p: (Vec<u8>, Vec<u8>), w: i32) -> i32 {
    let m = mat();
    let (w, _) = tightened_global_band(&p.0, &p.1, 5, &m, 6, 1, 6, 1, w);
    black_box(ksw_global2(&p.0, &p.1, 5, &m, 6, 1, 6, 1, w)).0
}

library_benchmark_group!(name = global; benchmarks = full_band, tightened);
main!(library_benchmark_groups = global);

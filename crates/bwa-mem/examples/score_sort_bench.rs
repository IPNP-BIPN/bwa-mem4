//! Times the score sort's two key formulations against each other, on whatever ISA it is built for.
//!
//! WHY THIS EXISTS. `mem_sort_dedup_patch`'s pass 3 was changed from a `(i32, i64, i32)` tuple key
//! to one packed u128, measured at -28% on an Apple M4. That measurement was taken on ONE
//! architecture, and the change is about how a comparator compiles, which is exactly the kind of
//! thing that does not transfer. x86-64 has no 128-bit ALU, so the packed key's construction costs
//! more there than it does on aarch64, while its comparison costs less than a three-field branchy
//! one. Which wins is a question for a stopwatch on each target.
//!
//!   cargo run --release -p bwa-mem4-mem --example score_sort_bench
//!
//! Both arms must produce the same permutation; the example asserts that before reporting any time.
use bwa_chain::ks_introsort_by_key;
use std::time::Instant;

/// A stand-in for the three fields of `MemAlnReg` the sort reads, plus an id so the permutation
/// can be compared between the two arms.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Row {
    score: i32,
    rb: i64,
    qb: i32,
    id: u32,
}

fn key_u128(r: &Row) -> u128 {
    let score = !((r.score as u32) ^ 0x8000_0000) as u128;
    let rb = ((r.rb as u64) ^ 0x8000_0000_0000_0000) as u128;
    let qb = ((r.qb as u32) ^ 0x8000_0000) as u128;
    (score << 96) | (rb << 32) | qb
}

/// A third formulation: two 64-bit words instead of one 128-bit one, compared lexicographically.
/// x86-64 has no 128-bit ALU, so the packed u128's construction costs two multi-word shifts per
/// element; a pair of u64s is built with ordinary 64-bit operations and still compares in one or
/// two instructions.
fn key_u64x2(r: &Row) -> (u64, u64) {
    let score = !((r.score as u32) ^ 0x8000_0000) as u64;
    let rb = (r.rb as u64) ^ 0x8000_0000_0000_0000;
    let qb = ((r.qb as u32) ^ 0x8000_0000) as u64;
    ((score << 32) | (rb >> 32), (rb << 32) | qb)
}

fn main() {
    // The shape the probe measured on a 1M-pair wgsim run: 2.18M calls averaging 92 regions.
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(92);
    let calls: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    // Scores repeat heavily in real data, which is what makes the second and third key fields do
    // any work at all; a uniform 32-bit score would never reach them.
    let rows: Vec<Row> = (0..n)
        .map(|i| Row {
            score: (rng() % 40) as i32,
            rb: (rng() % 46_000_000) as i64,
            qb: (rng() % 150) as i32,
            id: i as u32,
        })
        .collect();

    let mut a = rows.clone();
    let mut b = rows.clone();
    let (mut perm_t, mut perm_u, mut spare) = (Vec::new(), Vec::new(), Vec::new());
    ks_introsort_by_key(
        &mut a,
        &mut perm_t,
        &mut spare,
        |r| (r.score, r.rb, r.qb),
        |x, y| x.0 > y.0 || (x.0 == y.0 && (x.1 < y.1 || (x.1 == y.1 && x.2 < y.2))),
    );
    let mut spare2 = Vec::new();
    ks_introsort_by_key(&mut b, &mut perm_u, &mut spare2, key_u128, |x, y| x < y);
    assert_eq!(
        a.iter().map(|r| r.id).collect::<Vec<_>>(),
        b.iter().map(|r| r.id).collect::<Vec<_>>(),
        "the two keys produce different permutations"
    );

    let mut c = rows.clone();
    let (mut perm_p, mut spare3) = (Vec::new(), Vec::new());
    ks_introsort_by_key(&mut c, &mut perm_p, &mut spare3, key_u64x2, |x, y| x < y);
    assert_eq!(
        a.iter().map(|r| r.id).collect::<Vec<_>>(),
        c.iter().map(|r| r.id).collect::<Vec<_>>(),
        "the u64 pair produces a different permutation"
    );

    let mut best = [f64::MAX; 3];
    for _ in 0..3 {
        // Tuple key.
        let t = Instant::now();
        for _ in 0..calls {
            let mut v = rows.clone();
            ks_introsort_by_key(
                &mut v,
                &mut perm_t,
                &mut spare,
                |r| (r.score, r.rb, r.qb),
                |x, y| x.0 > y.0 || (x.0 == y.0 && (x.1 < y.1 || (x.1 == y.1 && x.2 < y.2))),
            );
            std::hint::black_box(&v);
        }
        let dt = t.elapsed().as_secs_f64();
        if dt < best[0] {
            best[0] = dt;
        }
        // Packed key.
        let t = Instant::now();
        for _ in 0..calls {
            let mut v = rows.clone();
            ks_introsort_by_key(&mut v, &mut perm_u, &mut spare2, key_u128, |x, y| x < y);
            std::hint::black_box(&v);
        }
        let dt = t.elapsed().as_secs_f64();
        if dt < best[1] {
            best[1] = dt;
        }
        // Pair of u64s.
        let t = Instant::now();
        for _ in 0..calls {
            let mut v = rows.clone();
            ks_introsort_by_key(&mut v, &mut perm_p, &mut spare3, key_u64x2, |x, y| x < y);
            std::hint::black_box(&v);
        }
        let dt = t.elapsed().as_secs_f64();
        if dt < best[2] {
            best[2] = dt;
        }
    }
    println!("n={n} calls={calls}");
    println!("tuple key: {:.3}s", best[0]);
    println!("u128 key:  {:.3}s", best[1]);
    println!("u64 pair:  {:.3}s", best[2]);
    println!("u128 is {:.3}x the cost of the tuple", best[1] / best[0]);
    println!(
        "u64 pair is {:.3}x the cost of the tuple",
        best[2] / best[0]
    );
}

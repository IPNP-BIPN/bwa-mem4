//! Times the two formulations of the SEQ emitter's block loop against each other.
//!
//! WHY THIS EXISTS. `push_seq_fwd` was changed from `chunks_exact(BLOCK)` to `as_chunks::<BLOCK>()`
//! to satisfy clippy's `chunks_exact_to_as_chunks` on Rust 1.98, and the commit claimed the new form
//! "hands the compiler `&[u8; BLOCK]` instead of a slice whose length it has to re-prove". That was
//! an assumption. It was never timed, and this function runs once per emitted record on a field the
//! length of the read, so an assumption is not good enough.
//!
//!   cargo run --release -p bwa-mem4-mem --example seq_emit_bench
//!
//! Both arms produce identical bytes; the example asserts that before reporting any time.
use std::time::Instant;

const BLOCK: usize = 16;

const fn base_table(alpha: &[u8; 5]) -> [u8; 256] {
    let mut t = [b'N'; 256];
    let mut i = 0;
    while i < 5 {
        t[i] = alpha[i];
        i += 1;
    }
    t
}
static FWD: [u8; 256] = base_table(b"ACGTN");

/// The formulation that shipped before 1.98's lint. The lint is allowed here on purpose: the whole
/// point of this file is to time the form clippy rejects against the form it asks for.
// `unknown_lints` first, and it is not decoration: `chunks_exact_to_as_chunks` does not exist in
// clippy 1.96.1, which is the toolchain CI pins, so naming it there is itself a denied lint. This is
// the same blind spot as the one this branch fixed in the other direction, met from the other side.
#[allow(unknown_lints)]
#[allow(clippy::chunks_exact_to_as_chunks)]
fn push_chunks_exact(out: &mut Vec<u8>, codes: &[u8]) {
    out.reserve(codes.len());
    let mut block = [0u8; BLOCK];
    let mut chunks = codes.chunks_exact(BLOCK);
    for chunk in &mut chunks {
        for (dst, &c) in block.iter_mut().zip(chunk) {
            *dst = FWD[c as usize];
        }
        out.extend_from_slice(&block);
    }
    for &c in chunks.remainder() {
        out.push(FWD[c as usize]);
    }
}

/// The formulation clippy asked for.
fn push_as_chunks(out: &mut Vec<u8>, codes: &[u8]) {
    out.reserve(codes.len());
    let mut block = [0u8; BLOCK];
    let (chunks, remainder) = codes.as_chunks::<BLOCK>();
    for chunk in chunks {
        for (dst, &c) in block.iter_mut().zip(chunk) {
            *dst = FWD[c as usize];
        }
        out.extend_from_slice(&block);
    }
    for &c in remainder {
        out.push(FWD[c as usize]);
    }
}

fn main() {
    let len: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(151);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let codes: Vec<u8> = (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 5) as u8
        })
        .collect();

    let (mut a, mut b) = (Vec::with_capacity(len), Vec::with_capacity(len));
    push_chunks_exact(&mut a, &codes);
    push_as_chunks(&mut b, &codes);
    assert_eq!(a, b, "the two formulations disagree");

    // Interleaved, each arm best of three, so an ordering artefact cannot decide it.
    let mut best = [f64::MAX; 2];
    for _ in 0..3 {
        for (arm, f) in [
            (0usize, push_chunks_exact as fn(&mut Vec<u8>, &[u8])),
            (1usize, push_as_chunks as fn(&mut Vec<u8>, &[u8])),
        ] {
            let mut out = Vec::with_capacity(len);
            let t = Instant::now();
            for _ in 0..reps {
                out.clear();
                f(&mut out, &codes);
            }
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if dt < best[arm] {
                best[arm] = dt;
            }
        }
    }
    println!("len={len} reps={reps}");
    println!("chunks_exact: {:.3}s", best[0]);
    println!("as_chunks:    {:.3}s", best[1]);
    println!(
        "as_chunks is {:.3}x the cost of chunks_exact",
        best[1] / best[0]
    );
}

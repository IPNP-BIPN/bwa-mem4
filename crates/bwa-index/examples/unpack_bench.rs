//! Times the 2-bit reference unpack: the scalar tail against the SSSE3 block loop.
//!
//! WHY THIS EXISTS. `unpack_pac_fwd` had a NEON path and no x86 one, so every x86 host unpacked one
//! base at a time; an SSSE3 path was added on the strength of a profile. That is a claim about
//! codegen, and this project has just been bitten by shipping exactly such a claim unmeasured (the
//! u128 sort key, a win on aarch64 and a 1.27x loss on x86). So the fix gets its own stopwatch.
//!
//!   cargo run --release -p bwa-mem4-index --example unpack_bench
//!
//! Both arms must produce the same bytes; the example asserts that before timing either.
use std::time::Instant;

/// The scalar loop every x86 build ran before the fix.
fn unpack_scalar(pac: &[u8], start: i64, len: usize, out: &mut Vec<u8>) {
    out.clear();
    let mut pos = start;
    let end = start + len as i64;
    while pos < end {
        out.push((pac[(pos >> 2) as usize] >> ((3 - (pos & 3)) << 1)) & 3);
        pos += 1;
    }
}

/// The block loop the fix adds, in the same shape as the one in `fmindex.rs`.
#[cfg(target_arch = "x86_64")]
fn unpack_vector(pac: &[u8], start: i64, len: usize, out: &mut Vec<u8>) {
    use std::arch::x86_64::*;
    out.clear();
    out.reserve(len + 16);
    let mut pos = start;
    let end = start + len as i64;
    while pos < end && (pos & 3) != 0 {
        out.push((pac[(pos >> 2) as usize] >> ((3 - (pos & 3)) << 1)) & 3);
        pos += 1;
    }
    if std::is_x86_feature_detected!("ssse3") {
        unsafe {
            let spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
            let three = _mm_set1_epi8(3);
            let m0 = _mm_setr_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
            let m1 = _mm_setr_epi8(0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0);
            let m2 = _mm_setr_epi8(0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0);
            let m3 = _mm_setr_epi8(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
            while pos + 16 <= end {
                let word = (pac.as_ptr().add((pos >> 2) as usize) as *const u32).read_unaligned();
                let v = _mm_shuffle_epi8(_mm_set1_epi32(word as i32), spread);
                let s6 = _mm_and_si128(_mm_srli_epi16(v, 6), m0);
                let s4 = _mm_and_si128(_mm_srli_epi16(v, 4), m1);
                let s2 = _mm_and_si128(_mm_srli_epi16(v, 2), m2);
                let s0 = _mm_and_si128(v, m3);
                let out_v = _mm_and_si128(
                    _mm_or_si128(_mm_or_si128(s6, s4), _mm_or_si128(s2, s0)),
                    three,
                );
                let n = out.len();
                _mm_storeu_si128(out.as_mut_ptr().add(n) as *mut __m128i, out_v);
                out.set_len(n + 16);
                pos += 16;
            }
        }
    }
    while pos < end {
        out.push((pac[(pos >> 2) as usize] >> ((3 - (pos & 3)) << 1)) & 3);
        pos += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn unpack_vector(pac: &[u8], start: i64, len: usize, out: &mut Vec<u8>) {
    unpack_scalar(pac, start, len, out)
}

fn main() {
    // A chain window's worth of reference. The collection pass fetches one per chain, and the
    // profile that motivated the fix was taken on 150 bp reads whose windows run a few hundred bases.
    let len: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let pac: Vec<u8> = (0..(len / 4 + 64))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();

    let (mut a, mut b) = (Vec::new(), Vec::new());
    for start in 0..8i64 {
        unpack_scalar(&pac, start, len, &mut a);
        unpack_vector(&pac, start, len, &mut b);
        assert_eq!(a, b, "the two unpacks disagree at start {start}");
    }

    let mut best = [f64::MAX; 2];
    for _ in 0..3 {
        for (arm, f) in [
            (0usize, unpack_scalar as fn(&[u8], i64, usize, &mut Vec<u8>)),
            (1usize, unpack_vector as fn(&[u8], i64, usize, &mut Vec<u8>)),
        ] {
            let mut out = Vec::with_capacity(len + 16);
            let t = Instant::now();
            for _ in 0..reps {
                f(&pac, 0, len, &mut out);
            }
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if dt < best[arm] {
                best[arm] = dt;
            }
        }
    }
    println!("len={len} reps={reps}");
    println!("scalar: {:.3}s", best[0]);
    println!("vector: {:.3}s", best[1]);
    println!("the vector path is {:.2}x faster", best[0] / best[1]);
}

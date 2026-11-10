//! Anti-diagonal (wavefront) `ksw_global2` in int16 lanes, NEON or SSE4.1: fg-labs/bwa-mem3's `ksw_g2_wave16`
//! (`ksw_global2_wave.h`, with its LEMMA 1-10 proof of byte-identity to the scalar `ksw_global2`).
//!
//! Cells on one anti-diagonal `s = i + j` depend only on diagonals `s - 1` and `s - 2`, so a vector's
//! worth of rows of a diagonal (8, or 16 under AVX2) go in one vector. H/E/F live in rings indexed by row (`+1` store offset), the
//! direction bytes in diagonal-major order, and the traceback walks them exactly as the scalar walks
//! its row-major bytes. Entered only where [`safe`] proves every value fits int16 and the band meets
//! the kernel's floor `w >= |tlen - qlen| + 3`.
//!
//! One kernel body ([`wave_fill`]) for every ISA: on x86_64 the NEON intrinsics it uses are the
//! SSE4.1 shims of [`sse`] (same 8 x int16 lanes, same names), the AVX2 ones of [`avx2`] (16
//! lanes, as the fork's AVX2 kernel) or the AVX-512BW ones of [`avx512`] (32 lanes, as the fork's
//! AVX-512 kernel), and the score lookup and the direction-byte store, the two steps without a
//! one-instruction twin, are [`score_lanes`] and [`store_dirs`]. A diagonal holds about `w + 1`
//! in-band cells, so the 16-lane kernel takes bands of [`WIDE_MIN`] and up, which need two 8-lane
//! chunks per diagonal, and the 32-lane one bands of [`wide512_min`] and up.

#[cfg(target_arch = "x86_64")]
use sse::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// The 5 x 5 score table as two 16-byte halves.
#[cfg(target_arch = "aarch64")]
type Table = uint8x16x2_t;
#[cfg(target_arch = "aarch64")]
unsafe fn load_table(tab: &[u8; 32]) -> Table {
    uint8x16x2_t(vld1q_u8(tab.as_ptr()), vld1q_u8(tab.as_ptr().add(16)))
}

/// Lane `l`'s score `mat[t[l] * 5 + q[7 - l]]`, sign-extended: `tp` and `qp` point at 8 bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn score_lanes(table: &Table, tp: *const u8, qp: *const u8) -> int16x8_t {
    let idx = vmla_u8(vrev64_u8(vld1_u8(qp)), vld1_u8(tp), vdup_n_u8(5));
    vmovl_s8(vreinterpret_s8_u8(vqtbl2_u8(*table, idx)))
}

/// The eight direction bytes of `d` (each lane `< 64`) at `p`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_dirs(p: *mut u8, d: int16x8_t) {
    vst1_u8(p, vmovn_u16(vreinterpretq_u16_s16(d)));
}

/// SSE4.1 under the NEON names the kernel uses; masks are all-ones / all-zeros int16 lanes.
#[cfg(target_arch = "x86_64")]
mod sse {
    pub(super) use std::arch::x86_64::*;
    type V = __m128i;
    macro_rules! shim {
        ($($name:ident($($a:ident: $t:ty),*) -> $r:ty $body:block)*) => {$(
            #[inline]
            #[target_feature(enable = "sse4.1")]
            pub(super) unsafe fn $name($($a: $t),*) -> $r $body
        )*};
    }
    shim! {
        vdupq_n_s16(x: i16) -> V { _mm_set1_epi16(x) }
        vld1q_s16(p: *const i16) -> V { _mm_loadu_si128(p.cast()) }
        vst1q_s16(p: *mut i16, v: V) -> () { _mm_storeu_si128(p.cast(), v) }
        vaddq_s16(a: V, b: V) -> V { _mm_add_epi16(a, b) }
        vsubq_s16(a: V, b: V) -> V { _mm_sub_epi16(a, b) }
        vmulq_s16(a: V, b: V) -> V { _mm_mullo_epi16(a, b) }
        vmaxq_s16(a: V, b: V) -> V { _mm_max_epi16(a, b) }
        vminq_s16(a: V, b: V) -> V { _mm_min_epi16(a, b) }
        vceqq_s16(a: V, b: V) -> V { _mm_cmpeq_epi16(a, b) }
        vcgtq_s16(a: V, b: V) -> V { _mm_cmpgt_epi16(a, b) }
        vcltq_s16(a: V, b: V) -> V { _mm_cmpgt_epi16(b, a) }
        vcgeq_s16(a: V, b: V) -> V { _mm_cmpeq_epi16(_mm_max_epi16(a, b), a) }
        vbslq_s16(m: V, a: V, b: V) -> V { _mm_blendv_epi8(b, a, m) }
        vandq_u16(a: V, b: V) -> V { _mm_and_si128(a, b) }
        vorrq_s16(a: V, b: V) -> V { _mm_or_si128(a, b) }
    }
}

#[cfg(target_arch = "x86_64")]
type Table = (__m128i, __m128i);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn load_table(tab: &[u8; 32]) -> Table {
    (
        _mm_loadu_si128(tab.as_ptr().cast()),
        _mm_loadu_si128(tab.as_ptr().add(16).cast()),
    )
}

/// As the NEON one: indices `t * 5 + q < 25` look up both halves with `pshufb`, which reads the
/// low four bits, and the half is picked by `idx > 15`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn score_lanes(table: &Table, tp: *const u8, qp: *const u8) -> __m128i {
    let rev = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1);
    let q = _mm_shuffle_epi8(_mm_loadl_epi64(qp.cast()), rev);
    let t = _mm_cvtepu8_epi16(_mm_loadl_epi64(tp.cast()));
    let i16x = _mm_add_epi16(_mm_mullo_epi16(t, _mm_set1_epi16(5)), _mm_cvtepu8_epi16(q));
    let idx = _mm_packus_epi16(i16x, i16x);
    let hi = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
    let s = _mm_blendv_epi8(
        _mm_shuffle_epi8(table.0, idx),
        _mm_shuffle_epi8(table.1, idx),
        hi,
    );
    _mm_cvtepi8_epi16(s)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn store_dirs(p: *mut u8, d: __m128i) {
    _mm_storel_epi64(p.cast(), _mm_packus_epi16(d, d));
}

/// Int16 "minus infinity": its own decrement cannot wrap and no feasible score reaches it ([`safe`]).
const VMINF: i16 = -30000;
/// Lane numbers, read as the first 8, 16 or 32 of them.
const IOTA: [i16; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];
/// Narrowest band the 16-lane AVX2 kernel takes: its diagonals hold up to `w + 1 >= 9` cells.
#[cfg(target_arch = "x86_64")]
const WIDE_MIN: usize = 8;

/// The widest kernel this host runs, in int16 lanes: 32 with AVX-512BW, 16 with AVX2, else 8.
/// `BWA4_EXTEND_TIER=avx2` or `sse41` caps it as it caps the extension kernels.
#[cfg(target_arch = "x86_64")]
fn max_lanes() -> usize {
    use std::sync::OnceLock;
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        let tier = std::env::var("BWA4_EXTEND_TIER");
        if std::arch::is_x86_feature_detected!("avx512bw")
            && !matches!(tier.as_deref(), Ok("avx2" | "sse41"))
        {
            32
        } else if std::arch::is_x86_feature_detected!("avx2") && tier.as_deref() != Ok("sse41") {
            16
        } else {
            8
        }
    })
}

/// Narrowest band the 32-lane kernel takes: below it a diagonal fits one 16-lane chunk.
/// `BWA4_WAVE512_MIN` overrides it; a huge value keeps those bands on the 16-lane kernel.
#[cfg(target_arch = "x86_64")]
fn wide512_min() -> usize {
    use std::sync::OnceLock;
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("BWA4_WAVE512_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
            .max(WIDE_MIN)
    })
}
/// Bytes of padding on both sides of the query and target copies (the vector loads overhang by 7).
const PAD: usize = 64;

/// Narrowest band routed here (the fork's `KSW_WAVE16_WMIN`: 4 on x86, 5 on NEON, where 4
/// measured neutral). On x86 the `w = 4` bands, the 1 bp indel floor, run faster as one 8-lane
/// diagonal than scalar: ksw_global2 -1.5..-3% on EPYC 7763 and 9V74. `BWA4_WAVE_WMIN`
/// overrides it; a huge value keeps every call on the scalar kernel.
fn wmin() -> usize {
    use std::sync::OnceLock;
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("BWA4_WAVE_WMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if cfg!(target_arch = "x86_64") { 4 } else { 5 })
            .max(3)
    })
}

/// The fork's `ksw_g2_wave16_safe`: every H/E/F the int16 kernel can produce stays clear of
/// [`VMINF`] and of int16 overflow, and the index math fits int16.
#[allow(clippy::too_many_arguments)]
fn safe(
    qlen: usize,
    tlen: usize,
    w: usize,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    mat: &[i8],
) -> bool {
    let (mut a, mut b) = (0i64, 0i64);
    for &v in &mat[..25] {
        a = a.max(i64::from(v));
        b = b.max(-i64::from(v));
    }
    let (o_max, e_max) = (i64::from(o_del.max(o_ins)), i64::from(e_del.max(e_ins)));
    let (q, t, w) = (qlen as i64, tlen as i64, w as i64);
    let l = q.min(t);
    let slack = 64;
    let h_upper = q * a + a;
    let h_lower = -(l * b) - (o_max + e_max * (2 * w + 2)) - b;
    q < 32000
        && t < 32000
        && t + w <= 32000
        && h_upper <= 32767 - slack
        && h_lower >= i64::from(VMINF) + o_max + e_max + slack
        && i64::from(VMINF) - (o_max + e_max) > -32768 + slack
}

#[derive(Default)]
struct Scratch {
    tpad: Vec<u8>,
    qpad: Vec<u8>,
    hb: Vec<i16>,
    eb: Vec<i16>,
    fb: Vec<i16>,
    z: Vec<u8>,
    zoff: Vec<usize>,
    zis: Vec<usize>,
    stride: usize,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// [`super::sw::ksw_global2`]'s result from the wavefront kernel, or `None` when the call is
/// outside what the kernel is proven for (or below the width crossover).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_wave16(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
) -> Option<(i32, Vec<u32>)> {
    let (qlen, tlen) = (query.len(), target.len());
    if m != 5 || mat.len() < 25 || qlen == 0 || tlen == 0 || w < 0 {
        return None;
    }
    let w = w as usize;
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("sse4.1") {
        return None;
    }
    if w < wmin()
        || w < qlen.abs_diff(tlen) + 3
        || !safe(qlen, tlen, w, o_del, e_del, o_ins, e_ins, mat)
    {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    let lanes = match max_lanes() {
        32 if w >= wide512_min() => 32,
        16 | 32 if w >= WIDE_MIN => 16,
        _ => 8,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let lanes = 8;
    Some(SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        // SAFETY: NEON is the AArch64 baseline; SSE4.1 was detected above on x86_64, and
        // `max_lanes` detected AVX2 for 16 lanes and AVX-512BW for 32.
        unsafe {
            wave16(
                &mut s, query, target, mat, o_del, e_del, o_ins, e_ins, w, lanes,
            )
        }
    }))
}

/// The forward pass in `L`-row chunks of each anti-diagonal (`L` = the vector's int16 lanes): fills
/// `sc.z`, `sc.zoff` and `sc.zis` for [`traceback`] and returns the score. [`wave16`] has sized the
/// scratch and copied the sequences. Expanded once per vector width, each with its own
/// [`score_lanes`], [`store_dirs`] and lane ops in scope.
macro_rules! wave_fill {
    ($(#[$m:meta])* $vis:vis $name:ident, $lanes:expr) => {
        $(#[$m])*
        #[allow(clippy::too_many_arguments)]
        $vis unsafe fn $name(
            sc: &mut Scratch,
            qlen: usize,
            tlen: usize,
            mat: &[i8],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            w: usize,
        ) -> i32 {
            const L: usize = $lanes;
            let sdim = tlen + qlen - 1;
            let stride = sc.stride;
            let (oe_del, oe_ins) = (o_del + e_del, o_ins + e_ins);
            let mut score = if qlen <= w {
                -(o_ins + e_ins * qlen as i32)
            } else {
                crate::sw::MINUS_INF
            };
            let mut tab = [0u8; 32];
            for (d, &v) in tab.iter_mut().zip(&mat[..25]) {
                *d = v as u8;
            }
            let (w16, q16) = (w as i16, qlen as i16);
            let mut roff = 0usize;
            // SAFETY: the ISA is the caller's contract (see above). Every pointer below stays inside its buffer: ring rows are
            // `stride >= maxlen + 33` long and lanes reach at most row `tlen + L <= tlen + 16`; the
            // query/target copies have `PAD` bytes on both sides of the at most `L - 1`-byte overhang; `z`
            // holds `zneed` bytes, the in-band cell count plus 64 for the last chunk's overhang.
            unsafe {
                let table = load_table(&tab);
                let vminf = vdupq_n_s16(VMINF);
                let laneid = vld1q_s16(IOTA.as_ptr());
                let (hb, eb, fb) = (sc.hb.as_mut_ptr(), sc.eb.as_mut_ptr(), sc.fb.as_mut_ptr());
                let (tp, qp, z) = (
                    sc.tpad.as_ptr().add(PAD),
                    sc.qpad.as_ptr().add(PAD),
                    sc.z.as_mut_ptr(),
                );
                let (mut hc, mut hd2, mut efc) = (0usize, 1usize, 0usize);
                for s in 0..sdim {
                    let (hcur, hdiag2) = (hb.add(hc * stride), hb.add(hd2 * stride));
                    let (ecur, ep1) = (eb.add(efc * stride), eb.add((efc ^ 1) * stride));
                    let (fcur, fp1) = (fb.add(efc * stride), fb.add((efc ^ 1) * stride));
                    let si = s as isize;
                    let i_lo = (si - (qlen as isize - 1))
                        .max((si - w as isize + 1) >> 1)
                        .max(0) as usize;
                    let i_hi = (s.min((s + w) >> 1)).min(tlen - 1);
                    sc.zoff[s] = roff;
                    sc.zis[s] = i_lo;
                    let zbase = roff;
                    if i_hi >= i_lo {
                        roff += i_hi - i_lo + 1;
                    }
                    let mut i0 = i_lo;
                    while i0 <= i_hi {
                        let (hdiag, e, f);
                        if i0 > i_lo && i0 + (L - 1) < i_hi {
                            hdiag = vld1q_s16(hdiag2.add(i0));
                            e = vld1q_s16(ep1.add(i0));
                            f = vld1q_s16(fp1.add(i0 + 1));
                        } else if i0 > 0 && s - i0 > L - 1 {
                            // Band-edge chunk (LEMMA 8b): no lane on row 0 or column 0; only the lane on the
                            // band's lower edge loses E and the one on its upper edge loses F.
                            hdiag = vld1q_s16(hdiag2.add(i0));
                            let (mut ev, mut fv) = (vld1q_s16(ep1.add(i0)), vld1q_s16(fp1.add(i0 + 1)));
                            if (s + w) & 1 == 0 {
                                let el = (((s as isize - w as isize) / 2) - i0 as isize) as i16;
                                let fl = (((s + w) / 2) as isize - i0 as isize) as i16;
                                ev = vbslq_s16(vceqq_s16(laneid, vdupq_n_s16(el)), vminf, ev);
                                fv = vbslq_s16(vceqq_s16(laneid, vdupq_n_s16(fl)), vminf, fv);
                            }
                            e = ev;
                            f = fv;
                        } else {
                            let iv = vaddq_s16(vdupq_n_s16(i0 as i16), laneid);
                            let jv = vsubq_s16(vdupq_n_s16((s - i0) as i16), laneid);
                            let beg = vmaxq_s16(vdupq_n_s16(0), vsubq_s16(iv, vdupq_n_s16(w16)));
                            let mut hd = vld1q_s16(hdiag2.add(i0));
                            let i0m = vceqq_s16(iv, vdupq_n_s16(0));
                            let j0m = vceqq_s16(jv, vdupq_n_s16(0));
                            let hi0 = vbslq_s16(
                                j0m,
                                vdupq_n_s16(0),
                                vsubq_s16(
                                    vdupq_n_s16(-o_ins as i16),
                                    vmulq_s16(vdupq_n_s16(e_ins as i16), jv),
                                ),
                            );
                            let hj0 = vsubq_s16(
                                vdupq_n_s16(-o_del as i16),
                                vmulq_s16(vdupq_n_s16(e_del as i16), iv),
                            );
                            hd = vbslq_s16(j0m, hj0, hd);
                            hd = vbslq_s16(i0m, hi0, hd);
                            hdiag = hd;
                            let begm1 = vmaxq_s16(vdupq_n_s16(0), vsubq_s16(iv, vdupq_n_s16(w16 + 1)));
                            let endm1 = vminq_s16(vdupq_n_s16(q16), vaddq_s16(iv, vdupq_n_s16(w16)));
                            let evalid = vandq_u16(
                                vcgeq_s16(iv, vdupq_n_s16(1)),
                                vandq_u16(vcgeq_s16(jv, begm1), vcltq_s16(jv, endm1)),
                            );
                            e = vbslq_s16(evalid, vld1q_s16(ep1.add(i0)), vminf);
                            f = vbslq_s16(vceqq_s16(jv, beg), vminf, vld1q_s16(fp1.add(i0 + 1)));
                        }
                        // Score index t*5 + q per byte lane; lane l reads query column s - i0 - l.
                        let sv = score_lanes(&table, tp.add(i0), qp.offset((s - i0) as isize - (L as isize - 1)));
                        let mm = vaddq_s16(hdiag, sv);
                        let h_me = vmaxq_s16(mm, e);
                        let h = vmaxq_s16(h_me, f);
                        let t1 = vsubq_s16(mm, vdupq_n_s16(oe_del as i16));
                        let em = vsubq_s16(e, vdupq_n_s16(e_del as i16));
                        let t2 = vsubq_s16(mm, vdupq_n_s16(oe_ins as i16));
                        let fm = vsubq_s16(f, vdupq_n_s16(e_ins as i16));
                        vst1q_s16(hcur.add(i0 + 1), h);
                        vst1q_s16(ecur.add(i0 + 1), vmaxq_s16(em, t1));
                        vst1q_s16(fcur.add(i0 + 1), vmaxq_s16(fm, t2));
                        let mut d = vbslq_s16(vcgeq_s16(mm, e), vdupq_n_s16(0), vdupq_n_s16(1));
                        d = vbslq_s16(vcgeq_s16(h_me, f), d, vdupq_n_s16(2));
                        d = vorrq_s16(
                            d,
                            vbslq_s16(vcgtq_s16(em, t1), vdupq_n_s16(1 << 2), vdupq_n_s16(0)),
                        );
                        d = vorrq_s16(
                            d,
                            vbslq_s16(vcgtq_s16(fm, t2), vdupq_n_s16(2 << 4), vdupq_n_s16(0)),
                        );
                        store_dirs(z.add(zbase + (i0 - i_lo)), d);
                        i0 += L;
                    }
                    // The terminal column (LEMMA 9): in band on this diagonal iff its row is `i_lo`.
                    if s + 1 == i_lo + qlen && i_lo <= i_hi {
                        score = i32::from(*hcur.add(i_lo + 1));
                    }
                    hc = if hc == 2 { 0 } else { hc + 1 };
                    hd2 = if hd2 == 2 { 0 } else { hd2 + 1 };
                    efc ^= 1;
                }
            }
            score
        }
    };
}

wave_fill!(
    /// [`wave_fill`] at eight lanes: NEON, or the SSE4.1 shims on x86_64.
    ///
    /// # Safety
    /// On x86_64 the CPU must have SSE4.1.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse4.1"))]
    fill,
    8
);

/// The 16-lane kernel: AVX2 shims under the NEON names, the 16-lane score lookup and direction
/// store, and [`wave_fill`] expanded over them.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Scratch, IOTA, PAD, VMINF};
    use std::arch::x86_64::*;
    type V = __m256i;
    macro_rules! shim {
        ($($name:ident($($a:ident: $t:ty),*) -> $r:ty $body:block)*) => {$(
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn $name($($a: $t),*) -> $r $body
        )*};
    }
    shim! {
        vdupq_n_s16(x: i16) -> V { _mm256_set1_epi16(x) }
        vld1q_s16(p: *const i16) -> V { _mm256_loadu_si256(p.cast()) }
        vst1q_s16(p: *mut i16, v: V) -> () { _mm256_storeu_si256(p.cast(), v) }
        vaddq_s16(a: V, b: V) -> V { _mm256_add_epi16(a, b) }
        vsubq_s16(a: V, b: V) -> V { _mm256_sub_epi16(a, b) }
        vmulq_s16(a: V, b: V) -> V { _mm256_mullo_epi16(a, b) }
        vmaxq_s16(a: V, b: V) -> V { _mm256_max_epi16(a, b) }
        vminq_s16(a: V, b: V) -> V { _mm256_min_epi16(a, b) }
        vceqq_s16(a: V, b: V) -> V { _mm256_cmpeq_epi16(a, b) }
        vcgtq_s16(a: V, b: V) -> V { _mm256_cmpgt_epi16(a, b) }
        vcltq_s16(a: V, b: V) -> V { _mm256_cmpgt_epi16(b, a) }
        vcgeq_s16(a: V, b: V) -> V { _mm256_cmpeq_epi16(_mm256_max_epi16(a, b), a) }
        vbslq_s16(m: V, a: V, b: V) -> V { _mm256_blendv_epi8(b, a, m) }
        vandq_u16(a: V, b: V) -> V { _mm256_and_si256(a, b) }
        vorrq_s16(a: V, b: V) -> V { _mm256_or_si256(a, b) }
    }

    type Table = (__m128i, __m128i);
    #[target_feature(enable = "avx2")]
    unsafe fn load_table(tab: &[u8; 32]) -> Table {
        (
            _mm_loadu_si128(tab.as_ptr().cast()),
            _mm_loadu_si128(tab.as_ptr().add(16).cast()),
        )
    }

    /// Lane `l`'s score `mat[t[l] * 5 + q[15 - l]]`, sign-extended: `tp` and `qp` point at 16 bytes.
    /// Codes are at most 4, so `t * 5 + q < 25` fits a byte and the SSE4.1 lookup applies as is.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn score_lanes(table: &Table, tp: *const u8, qp: *const u8) -> V {
        let rev = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let q = _mm_shuffle_epi8(_mm_loadu_si128(qp.cast()), rev);
        let t = _mm_loadu_si128(tp.cast());
        let t4 = _mm_add_epi8(_mm_add_epi8(t, t), _mm_add_epi8(t, t));
        let idx = _mm_add_epi8(_mm_add_epi8(t4, t), q);
        let hi = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15));
        let s = _mm_blendv_epi8(
            _mm_shuffle_epi8(table.0, idx),
            _mm_shuffle_epi8(table.1, idx),
            hi,
        );
        _mm256_cvtepi8_epi16(s)
    }

    /// The sixteen direction bytes of `d` (each lane `< 64`) at `p`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_dirs(p: *mut u8, d: V) {
        let b = _mm_packus_epi16(_mm256_castsi256_si128(d), _mm256_extracti128_si256::<1>(d));
        _mm_storeu_si128(p.cast(), b);
    }

    wave_fill!(
        /// [`wave_fill`] at sixteen lanes.
        ///
        /// # Safety
        /// The CPU must have AVX2.
        #[target_feature(enable = "avx2")]
        pub(super) fill,
        16
    );
}

/// The 32-lane kernel: AVX-512BW shims under the NEON names (compares widened to vector masks,
/// which LLVM folds back into mask registers), the score lookup as one word permute over the 25
/// scores, and [`wave_fill`] expanded over them.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{Scratch, IOTA, PAD, VMINF};
    use std::arch::x86_64::*;
    type V = __m512i;
    macro_rules! shim {
        ($($name:ident($($a:ident: $t:ty),*) -> $r:ty $body:block)*) => {$(
            #[inline]
            #[target_feature(enable = "avx512bw")]
            unsafe fn $name($($a: $t),*) -> $r $body
        )*};
    }
    shim! {
        vdupq_n_s16(x: i16) -> V { _mm512_set1_epi16(x) }
        vld1q_s16(p: *const i16) -> V { _mm512_loadu_si512(p.cast()) }
        vst1q_s16(p: *mut i16, v: V) -> () { _mm512_storeu_si512(p.cast(), v) }
        vaddq_s16(a: V, b: V) -> V { _mm512_add_epi16(a, b) }
        vsubq_s16(a: V, b: V) -> V { _mm512_sub_epi16(a, b) }
        vmulq_s16(a: V, b: V) -> V { _mm512_mullo_epi16(a, b) }
        vmaxq_s16(a: V, b: V) -> V { _mm512_max_epi16(a, b) }
        vminq_s16(a: V, b: V) -> V { _mm512_min_epi16(a, b) }
        vceqq_s16(a: V, b: V) -> V { _mm512_movm_epi16(_mm512_cmpeq_epi16_mask(a, b)) }
        vcgtq_s16(a: V, b: V) -> V { _mm512_movm_epi16(_mm512_cmpgt_epi16_mask(a, b)) }
        vcltq_s16(a: V, b: V) -> V { _mm512_movm_epi16(_mm512_cmplt_epi16_mask(a, b)) }
        vcgeq_s16(a: V, b: V) -> V { _mm512_movm_epi16(_mm512_cmpge_epi16_mask(a, b)) }
        vbslq_s16(m: V, a: V, b: V) -> V { _mm512_mask_blend_epi16(_mm512_movepi16_mask(m), b, a) }
        vandq_u16(a: V, b: V) -> V { _mm512_and_si512(a, b) }
        vorrq_s16(a: V, b: V) -> V { _mm512_or_si512(a, b) }
    }

    /// The 25 scores sign-extended to words (the rest of the 32 bytes are zero).
    type Table = V;
    #[target_feature(enable = "avx512bw")]
    unsafe fn load_table(tab: &[u8; 32]) -> Table {
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(tab.as_ptr().cast()))
    }

    /// Word `l` is `31 - l`: the permute that reverses the query's 32 lanes.
    const REV: [i16; 32] = [
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
        8, 7, 6, 5, 4, 3, 2, 1, 0,
    ];

    /// Lane `l`'s score `mat[t[l] * 5 + q[31 - l]]`: `tp` and `qp` point at 32 bytes. Codes are at
    /// most 4, so the index is below 25 and `vpermw` (which reads it modulo 32) is exact.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn score_lanes(table: &Table, tp: *const u8, qp: *const u8) -> V {
        let q = _mm512_permutexvar_epi16(
            _mm512_loadu_si512(REV.as_ptr().cast()),
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(qp.cast())),
        );
        let t = _mm512_cvtepu8_epi16(_mm256_loadu_si256(tp.cast()));
        let idx = _mm512_add_epi16(_mm512_add_epi16(_mm512_slli_epi16::<2>(t), t), q);
        _mm512_permutexvar_epi16(idx, *table)
    }

    /// The 32 direction bytes of `d` (each lane `< 64`) at `p`.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn store_dirs(p: *mut u8, d: V) {
        _mm256_storeu_si256(p.cast(), _mm512_cvtepi16_epi8(d));
    }

    wave_fill!(
        /// [`wave_fill`] at 32 lanes.
        ///
        /// # Safety
        /// The CPU must have AVX-512BW.
        #[target_feature(enable = "avx512bw")]
        pub(super) fill,
        32
    );
}

/// # Safety
/// On x86_64 the CPU must have SSE4.1, AVX2 for 16 `lanes` and AVX-512BW for 32.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse4.1"))]
unsafe fn wave16(
    sc: &mut Scratch,
    query: &[u8],
    target: &[u8],
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: usize,
    lanes: usize,
) -> (i32, Vec<u32>) {
    let (qlen, tlen) = (query.len(), target.len());
    let maxlen = qlen.max(tlen);
    if sc.stride < maxlen + 1 + 32 {
        // Grown, never cleared between calls: no in-band cell reads a value this call did not write
        // (stale ring lanes are masked by `evalid`/`freset`, LEMMA 7).
        sc.stride = maxlen + 1 + 32;
        sc.hb = vec![VMINF; 3 * sc.stride];
        sc.eb = vec![VMINF; 2 * sc.stride];
        sc.fb = vec![VMINF; 2 * sc.stride];
        sc.tpad = vec![0; maxlen + 2 * PAD];
        sc.qpad = vec![0; maxlen + 2 * PAD];
    }
    let sdim = tlen + qlen - 1;
    let zneed = qlen.min(2 * w + 1) * (tlen + qlen) + 64;
    if sc.z.len() < zneed {
        sc.z.resize(zneed, 0);
    }
    if sc.zoff.len() < sdim {
        sc.zoff.resize(sdim, 0);
        sc.zis.resize(sdim, 0);
    }
    sc.tpad[PAD..PAD + tlen].copy_from_slice(target);
    sc.qpad[PAD..PAD + qlen].copy_from_slice(query);
    #[cfg(target_arch = "x86_64")]
    let score = match lanes {
        32 => avx512::fill(sc, qlen, tlen, mat, o_del, e_del, o_ins, e_ins, w),
        16 => avx2::fill(sc, qlen, tlen, mat, o_del, e_del, o_ins, e_ins, w),
        _ => fill(sc, qlen, tlen, mat, o_del, e_del, o_ins, e_ins, w),
    };
    #[cfg(not(target_arch = "x86_64"))]
    let score = {
        let _ = lanes;
        fill(sc, qlen, tlen, mat, o_del, e_del, o_ins, e_ins, w)
    };
    (score, traceback(sc, qlen, tlen, w))
}

/// The traceback (LEMMA 10): the scalar's walk over the direction bytes [`wave_fill`] stored,
/// diagonal-major. Independent of the lane width.
fn traceback(sc: &Scratch, qlen: usize, tlen: usize, w: usize) -> Vec<u32> {
    let mut cigar: Vec<u32> = Vec::new();
    let mut i = tlen as i64 - 1;
    let mut k = (tlen as i64 + w as i64).min(qlen as i64) - 1;
    let mut which = 0u8;
    while i >= 0 && k >= 0 {
        let s = (i + k) as usize;
        let d = sc.z[sc.zoff[s] + (i as usize - sc.zis[s])];
        which = (d >> (which << 1)) & 3;
        match which {
            0 => {
                crate::sw::push_cigar(&mut cigar, 0, 1);
                i -= 1;
                k -= 1;
            }
            1 => {
                crate::sw::push_cigar(&mut cigar, 2, 1);
                i -= 1;
            }
            _ => {
                crate::sw::push_cigar(&mut cigar, 1, 1);
                k -= 1;
            }
        }
    }
    if i >= 0 {
        crate::sw::push_cigar(&mut cigar, 2, (i + 1) as u32);
    }
    if k >= 0 {
        crate::sw::push_cigar(&mut cigar, 1, (k + 1) as u32);
    }
    cigar.reverse();
    cigar
}

/// The lane counts whose kernels can run here (only x86_64 has the 16- and 32-lane ones).
#[cfg(test)]
fn lanes_here() -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    return [
        (8, true),
        (16, std::arch::is_x86_feature_detected!("avx2")),
        (32, std::arch::is_x86_feature_detected!("avx512bw")),
    ]
    .into_iter()
    .filter_map(|(l, ok)| ok.then_some(l))
    .collect();
    #[cfg(not(target_arch = "x86_64"))]
    vec![8]
}

#[cfg(test)]
fn isa_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    return std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(not(target_arch = "x86_64"))]
    true
}

#[cfg(test)]
mod tests {
    /// The wavefront kernel against the scalar `ksw_global2` on random inputs at every width it
    /// may be given (`w >= |tlen - qlen| + 3`), including tie-heavy scoring.
    #[test]
    fn wave16_matches_scalar() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st % n
        };
        let mut checked = 0;
        for case in 0..30000 {
            let qlen = 1 + rnd(if case % 5 == 0 { 400 } else { 160 }) as usize;
            let dl = rnd(12) as i64 - 6;
            let tlen = (qlen as i64 + dl).max(1) as usize;
            let query: Vec<u8> = (0..qlen)
                .map(|_| if rnd(40) == 0 { 4 } else { rnd(4) as u8 })
                .collect();
            // Mostly a mutated copy, so real alignments with indels dominate, as in production.
            let mut target: Vec<u8> = Vec::with_capacity(tlen);
            let mut j = 0;
            while target.len() < tlen {
                let r = rnd(100);
                if j < qlen && r < 85 {
                    target.push(query[j]);
                    j += 1;
                } else if r < 92 {
                    target.push(rnd(5) as u8);
                    j += 1;
                } else if r < 96 {
                    j += 1 + rnd(3) as usize;
                } else {
                    target.push(rnd(4) as u8);
                }
            }
            let (a, b) = if case % 3 == 0 {
                (1i8, 1i8)
            } else {
                (1 + rnd(3) as i8, 1 + rnd(5) as i8)
            };
            let mut mat = [0i8; 25];
            for x in 0..5 {
                for y in 0..5 {
                    mat[x * 5 + y] = if x == 4 || y == 4 {
                        -1
                    } else if x == y {
                        a
                    } else {
                        -b
                    };
                }
            }
            let (o_del, e_del, o_ins, e_ins) = if case % 4 == 0 {
                (
                    1 + rnd(8) as i32,
                    1 + rnd(3) as i32,
                    1 + rnd(8) as i32,
                    1 + rnd(3) as i32,
                )
            } else {
                (6, 1, 6, 1)
            };
            let w = (qlen.abs_diff(tlen) + 3 + rnd(40) as usize) as i32;
            if !super::safe(qlen, tlen, w as usize, o_del, e_del, o_ins, e_ins, &mat) {
                continue;
            }
            if !super::isa_ok() {
                return;
            }
            let scalar = crate::sw::ksw_global2_scalar(
                &query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w,
            );
            for lanes in super::lanes_here() {
                // SAFETY: `isa_ok` checked SSE4.1, `lanes_here` AVX2 and AVX-512BW.
                let wave = super::SCRATCH.with(|s| unsafe {
                    super::wave16(
                        &mut s.borrow_mut(),
                        &query,
                        &target,
                        &mat,
                        o_del,
                        e_del,
                        o_ins,
                        e_ins,
                        w as usize,
                        lanes,
                    )
                });
                assert_eq!(
                    wave, scalar,
                    "case {case}: qlen {qlen} tlen {tlen} w {w} lanes {lanes}"
                );
            }
            checked += 1;
        }
        assert!(checked > 20000, "only {checked} cases reached the kernel");
        // Read by the SDE job, which must see the 32-lane kernel among them.
        println!("wave16 lanes checked: {:?}", super::lanes_here());
    }
}

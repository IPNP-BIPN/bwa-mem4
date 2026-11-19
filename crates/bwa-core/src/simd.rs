//! Every SIMD memory access in the workspace, behind safe, array-typed functions.
//!
//! The vector kernels are safe `#[target_feature]` functions: their arithmetic intrinsics are safe
//! to call there. What stays `unsafe` is the handful of intrinsics that read or write through a
//! pointer. Each one is wrapped here exactly once, as `ld*(s, i)` / `st*(s, i, v)`: the vector at
//! element offset `i` of slice `s`. The slice `s[i..i + width]` is taken first, so an out-of-range
//! access panics like indexing does, and the `unsafe` block holds a single call on a slice that
//! is exactly the width the intrinsic touches. That one bounds check is the only cost, and LLVM
//! hoists it out of loops whose range it can see.
//!
//! The x86 helpers are generic over [`Int`]: any plain integer element type, which is what makes it
//! sound to store arbitrary vector bits into the array.

/// Integer element types: every bit pattern is a valid value, and there is no padding.
pub trait Int: Copy + private::Sealed {}
mod private {
    pub trait Sealed {}
}
macro_rules! int {
    ($($t:ty)*) => { $(impl private::Sealed for $t {} impl Int for $t {})* };
}
int!(u8 i8 u16 i16 u32 i32 u64 i64);

/// Prefetch the cache line holding `r` for a read (`prfm pldl1keep` / `prefetcht0`). A hint: it
/// changes no result, and a no-op elsewhere.
#[inline(always)]
pub fn prefetch<T>(r: &T) {
    let p = (r as *const T).cast::<u8>();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a prefetch reads nothing architecturally and never faults; `p` is a live reference.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.cast());
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

#[cfg(target_arch = "aarch64")]
pub use neon::*;
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    macro_rules! ld {
        ($($name:ident: [$t:ty; $n:literal] -> $v:ty = $f:ident;)*) => {$(
            #[inline(always)]
            pub fn $name(s: &[$t], i: usize) -> $v {
                let a = &s[i..i + $n];
                // SAFETY: `a` is exactly the width `$f` reads; NEON is aarch64 baseline.
                unsafe { $f(a.as_ptr()) }
            }
        )*};
    }
    macro_rules! st {
        ($($name:ident: [$t:ty; $n:literal] <- $v:ty = $f:ident;)*) => {$(
            #[inline(always)]
            pub fn $name(s: &mut [$t], i: usize, v: $v) {
                let a = &mut s[i..i + $n];
                // SAFETY: `a` is exactly the width `$f` writes; NEON is aarch64 baseline.
                unsafe { $f(a.as_mut_ptr(), v) }
            }
        )*};
    }
    ld! {
        ld_u8: [u8; 16] -> uint8x16_t = vld1q_u8;
        ld_s8: [i8; 16] -> int8x16_t = vld1q_s8;
        ld_u16: [u16; 8] -> uint16x8_t = vld1q_u16;
        ld_s16: [i16; 8] -> int16x8_t = vld1q_s16;
        ld_u32: [u32; 4] -> uint32x4_t = vld1q_u32;
        ld_s32: [i32; 4] -> int32x4_t = vld1q_s32;
        ld_u8x8: [u8; 8] -> uint8x8_t = vld1_u8;
        ld_u8_x2: [u8; 32] -> uint8x16x2_t = vld1q_u8_x2;
        ld2_u8: [u8; 32] -> uint8x16x2_t = vld2q_u8;
    }
    st! {
        st_u8: [u8; 16] <- uint8x16_t = vst1q_u8;
        st_s8: [i8; 16] <- int8x16_t = vst1q_s8;
        st_u16: [u16; 8] <- uint16x8_t = vst1q_u16;
        st_s16: [i16; 8] <- int16x8_t = vst1q_s16;
        st_u32: [u32; 4] <- uint32x4_t = vst1q_u32;
        st_s32: [i32; 4] <- int32x4_t = vst1q_s32;
        st_u8x8: [u8; 8] <- uint8x8_t = vst1_u8;
        st4_u8: [u8; 64] <- uint8x16x4_t = vst4q_u8;
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::*;
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Int;
    use std::arch::x86_64::*;
    use std::mem::size_of;

    macro_rules! ldst {
        ($($feat:literal $bytes:literal $v:ty: $ld:ident = $fl:ident, $st:ident = $fs:ident;)*) => {$(
            /// # Safety
            /// The CPU must have the target feature: free to call from a function that enables it.
            #[inline]
            #[target_feature(enable = $feat)]
            pub fn $ld<T: Int>(s: &[T], i: usize) -> $v {
                let a = &s[i..i + $bytes / size_of::<T>()];
                // SAFETY: `a` is exactly the bytes the unaligned load reads.
                unsafe { $fl(a.as_ptr().cast()) }
            }
            /// # Safety
            /// The CPU must have the target feature: free to call from a function that enables it.
            #[inline]
            #[target_feature(enable = $feat)]
            pub fn $st<T: Int>(s: &mut [T], i: usize, v: $v) {
                let a = &mut s[i..i + $bytes / size_of::<T>()];
                // SAFETY: `a` is exactly the bytes the unaligned store writes, and `T: Int`
                // accepts any bit pattern.
                unsafe { $fs(a.as_mut_ptr().cast(), v) }
            }
        )*};
    }
    ldst! {
        "sse2" 8 __m128i: ld64 = _mm_loadl_epi64, st64 = _mm_storel_epi64;
        "sse2" 16 __m128i: ld128 = _mm_loadu_si128, st128 = _mm_storeu_si128;
        "avx" 32 __m256i: ld256 = _mm256_loadu_si256, st256 = _mm256_storeu_si256;
        "avx512f" 64 __m512i: ld512 = _mm512_loadu_si512, st512 = _mm512_storeu_si512;
    }
}

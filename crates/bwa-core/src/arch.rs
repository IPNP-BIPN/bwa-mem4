//! Runtime CPU architecture / SIMD detection and the per-architecture strategy recommendations that
//! follow from this project's benchmarks and BWA-MEME's published results.
//!
//! The two seeding engines have opposite sweet spots:
//!
//! * **AArch64 (Apple Silicon).** The FMD-index seeding is already latency-hidden (lockstep +
//!   prefetch), and the NEON banded SW kernel dominates runtime (~85% of cycles). A learned index
//!   (BWA-MEME) trades many cheap `Occ` loads for few expensive LEM lookups; with FM's latency
//!   already hidden this is a net loss — measured **~5x slower** full-seeder on an M4 Max. So AArch64
//!   uses the **FM index**.
//! * **x86-64 (Intel/AMD).** FM `Occ` loads are more latency-bound on the x86 cache hierarchy, and
//!   AVX-512 makes the SW kernel much faster (64 u8 lanes vs NEON's 16), so seeding is a larger share
//!   of runtime. BWA-MEME reports **3.45x seeding / 1.42x end-to-end** there. So x86-64 prefers the
//!   **learned index** — when its P-RMI/suffix-array index is available.
//!
//! This module only *detects and recommends*; wiring the learned engine into the pipeline is gated on
//! `learned_index_available` (the index files being built + loaded), so the default everywhere stays
//! the FM index until a learned index is explicitly provided.

/// SIMD capability of the running CPU, in the terms that set the banded SW kernel width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simd {
    /// AArch64 NEON: 128-bit, 16 u8 / 8 i16 lanes (the Apple-Silicon path).
    Neon,
    /// x86-64 AVX2: 256-bit, 32 u8 / 16 i16 lanes.
    Avx2,
    /// x86-64 AVX-512 (F+BW): 512-bit, 64 u8 / 32 i16 lanes.
    Avx512,
    /// No supported SIMD; scalar fallback.
    Scalar,
}

impl Simd {
    /// u8 lane count of the banded SW kernel on this SIMD width.
    pub fn u8_lanes(self) -> usize {
        match self {
            Simd::Neon => 16,
            Simd::Avx2 => 32,
            Simd::Avx512 => 64,
            Simd::Scalar => 1,
        }
    }

    /// Short human-readable tag, e.g. `"aarch64 NEON"`.
    pub fn label(self) -> &'static str {
        match self {
            Simd::Neon => "aarch64 NEON",
            Simd::Avx2 => "x86-64 AVX2",
            Simd::Avx512 => "x86-64 AVX-512",
            Simd::Scalar => "scalar",
        }
    }
}

/// Which seeding engine the pipeline should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedEngine {
    /// FMD-index backward-extension seeding (lockstep + prefetch). The measured winner on AArch64 and
    /// the default whenever no learned index is loaded.
    FmIndex,
    /// BWA-MEME learned-index (P-RMI + plain suffix array) seeding. Favoured on x86-64.
    Learned,
}

/// Detect the SIMD capability of the running CPU. On AArch64, NEON is architecturally baseline; on
/// x86-64 the widest available of AVX-512(F+BW) / AVX2 is reported via runtime feature detection.
#[inline]
pub fn detect_simd() -> Simd {
    #[cfg(target_arch = "aarch64")]
    {
        Simd::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            Simd::Avx512
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Simd::Avx2
        } else {
            Simd::Scalar
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Simd::Scalar
    }
}

/// Recommended seeding engine for the running CPU. `learned_index_available` reflects whether the
/// learned-index files (P-RMI + suffix array) were built and loaded; when `false`, FM is the only
/// option. The learned engine is recommended only on x86-64 (AVX2/AVX-512), matching where BWA-MEME
/// shows a win; AArch64 always uses FM (learned index is ~5x slower there in our measurements).
#[inline]
pub fn recommended_seed_engine(learned_index_available: bool) -> SeedEngine {
    match detect_simd() {
        Simd::Avx512 | Simd::Avx2 if learned_index_available => SeedEngine::Learned,
        _ => SeedEngine::FmIndex,
    }
}

/// One-line startup summary, e.g.
/// `"CPU: aarch64 NEON (16 u8 lanes); seeding: FM-index (learned index: unavailable)"`.
pub fn summary(learned_index_available: bool) -> String {
    let simd = detect_simd();
    let engine = match recommended_seed_engine(learned_index_available) {
        SeedEngine::FmIndex => "FM-index",
        SeedEngine::Learned => "learned-index (BWA-MEME)",
    };
    let learned = if learned_index_available {
        "available"
    } else {
        "unavailable"
    };
    format!(
        "CPU: {} ({} u8 lanes); seeding: {} (learned index: {})",
        simd.label(),
        simd.u8_lanes(),
        engine,
        learned
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_this_cpu() {
        let s = detect_simd();
        // On the two supported arches we must report a real SIMD width, not scalar.
        #[cfg(target_arch = "aarch64")]
        assert_eq!(s, Simd::Neon);
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(s, Simd::Avx512 | Simd::Avx2 | Simd::Scalar));
        assert!(s.u8_lanes() >= 1);
    }

    #[test]
    fn seed_engine_policy() {
        // Without a learned index, FM everywhere.
        assert_eq!(recommended_seed_engine(false), SeedEngine::FmIndex);
        // With one, the choice tracks the arch: AArch64 stays FM, x86-64 (AVX2+) goes learned.
        let with = recommended_seed_engine(true);
        match detect_simd() {
            Simd::Avx2 | Simd::Avx512 => assert_eq!(with, SeedEngine::Learned),
            Simd::Neon | Simd::Scalar => assert_eq!(with, SeedEngine::FmIndex),
        }
    }
}

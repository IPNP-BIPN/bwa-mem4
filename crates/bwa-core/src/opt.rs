//! Alignment options, mirroring bwa-mem2's `mem_opt_t` and its `mem_opt_init()` defaults.

/// `MemOpt::flag` bits, mirroring bwa-mem2's `MEM_F_*` (`reference/bwa-mem2/src/bwamem.h`).
pub mod flags {
    /// Paired-end input (`-p`/two input files).
    pub const PE: i32 = 0x2;
    /// Output all found alignments, secondary ones included (`-a`).
    pub const ALL: i32 = 0x8;
    /// Mark shadowed hits 0x100 rather than 0x800 (`-M`).
    pub const NO_MULTI: i32 = 0x10;
    /// Soft-clip supplementary alignments too, instead of hard-clipping them (`-Y`).
    pub const SOFTCLIP: i32 = 0x200;
}

/// Alignment parameters. Field names and default values mirror bwa-mem2's `mem_opt_t`
/// (see `reference/bwa-mem2/src/bwamem.cpp::mem_opt_init`).
#[derive(Debug, Clone)]
pub struct MemOpt {
    /// Match score.
    pub a: i32,
    /// Mismatch penalty.
    pub b: i32,
    /// Gap open penalty (deletion / insertion).
    pub o_del: i32,
    pub o_ins: i32,
    /// Gap extension penalty (deletion / insertion).
    pub e_del: i32,
    pub e_ins: i32,
    /// Penalty for an unpaired read pair.
    pub pen_unpaired: i32,
    /// 5'/3' clipping penalty.
    pub pen_clip5: i32,
    pub pen_clip3: i32,
    /// Band width.
    pub w: i32,
    /// Off-diagonal X-dropoff (Z-drop).
    pub zdrop: i32,
    /// Minimum score to output (`-T`).
    pub t: i32,
    /// Behaviour flags.
    pub flag: i32,
    /// Minimum seed length (`-k`).
    pub min_seed_len: i32,
    /// Minimum chain weight (`-W`).
    pub min_chain_weight: i32,
    /// Max chain extension.
    pub max_chain_extend: i32,
    /// Re-seed trigger factor (`-r`).
    pub split_factor: f32,
    /// Re-seed occurrence threshold.
    pub split_width: i32,
    /// Skip seeds with more than this many occurrences (`-c`).
    pub max_occ: i32,
    /// Max chain gap.
    pub max_chain_gap: i32,
    /// Number of threads.
    pub n_threads: i32,
    /// Bases processed per batch (`-K`).
    pub chunk_size: i64,
    /// Redundancy mask level.
    pub mask_level: f32,
    /// Chain drop ratio (`-D`).
    pub drop_ratio: f32,
    /// XA drop ratio.
    pub xa_drop_ratio: f32,
    /// Redundant-hit mask level.
    pub mask_level_redun: f32,
    /// MAPQ coefficient length.
    pub mapq_coef_len: f64,
    /// MAPQ coefficient factor. bwa-mem2 declares this `int` and sets it to `(int)log(mapq_coef_len)`,
    /// so the fractional part is truncated (`log(50) = 3.912` becomes `3`). We keep it as an `f64`
    /// holding that already-truncated integer value; the truncation matters for borderline MAPQs.
    pub mapq_coef_fac: f64,
    /// Max insert size.
    pub max_ins: i32,
    /// Max rounds of mate-SW rescue (`-m`).
    pub max_matesw: i32,
    /// Seed occurrence for the 3rd round (`-y`).
    pub max_mem_intv: i64,
    /// Max XA hits.
    pub max_xa_hits: i32,
    /// Max XA hits for ALT contigs.
    pub max_xa_hits_alt: i32,
    /// 5x5 scoring matrix (`bwa_fill_scmat`).
    pub mat: [i8; 25],
}

impl Default for MemOpt {
    fn default() -> Self {
        let a = 1i32;
        let b = 4i32;
        let mut mat = [0i8; 25];
        fill_scmat(a, b, &mut mat);
        MemOpt {
            a,
            b,
            o_del: 6,
            o_ins: 6,
            e_del: 1,
            e_ins: 1,
            pen_unpaired: 17,
            pen_clip5: 5,
            pen_clip3: 5,
            w: 100,
            zdrop: 100,
            t: 30,
            flag: 0,
            min_seed_len: 19,
            min_chain_weight: 0,
            max_chain_extend: 1 << 30,
            split_factor: 1.5,
            split_width: 10,
            max_occ: 500,
            max_chain_gap: 10000,
            n_threads: 1,
            chunk_size: 10_000_000,
            mask_level: 0.50,
            drop_ratio: 0.50,
            xa_drop_ratio: 0.80,
            mask_level_redun: 0.95,
            mapq_coef_len: 50.0,
            // (int)log(50) = 3, matching bwa-mem2's integer `mapQ_coef_fac`.
            mapq_coef_fac: (50.0f64.ln() as i32) as f64,
            max_ins: 10000,
            max_matesw: 50,
            max_mem_intv: 20,
            max_xa_hits: 5,
            max_xa_hits_alt: 200,
            mat,
        }
    }
}

/// Fill the 5x5 scoring matrix, mirroring bwa's `bwa_fill_scmat(a, b, mat)`.
fn fill_scmat(a: i32, b: i32, mat: &mut [i8; 25]) {
    let mut k = 0usize;
    for i in 0..4 {
        for j in 0..4 {
            mat[k] = if i == j { a as i8 } else { -(b as i8) };
            k += 1;
        }
        mat[k] = -1; // ambiguous base
        k += 1;
    }
    for _ in 0..5 {
        mat[k] = -1;
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_bwa_mem2() {
        let o = MemOpt::default();
        assert_eq!((o.a, o.b), (1, 4));
        assert_eq!((o.o_del, o.e_del), (6, 1));
        assert_eq!(o.min_seed_len, 19);
        assert_eq!(o.max_occ, 500);
        assert_eq!(o.chunk_size, 10_000_000);
        assert_eq!(o.max_mem_intv, 20);
        // scoring matrix: diagonal = a, off-diagonal = -b, N row/col = -1
        assert_eq!(o.mat[0], 1); // A/A
        assert_eq!(o.mat[1], -4); // A/C
        assert_eq!(o.mat[4], -1); // A/N
        assert_eq!(o.mat[24], -1); // N/N
    }
}

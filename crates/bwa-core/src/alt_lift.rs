//! `mem --alt-project` (fg-labs/bwa-mem3 0.18, #775): the ALT -> primary-assembly liftover table,
//! one entry per contig, built from `<idxbase>.alt` by `bwa_mem::alt_lift::load`. It lives here
//! only because [`crate::opt::MemOpt`] holds it; the parsing and the lift are in `bwa-mem`.

/// One ALT contig's alignment to the primary assembly (the fork's `alt_lift1_t`).
#[derive(Debug, Clone, Default)]
pub struct AltLift1 {
    /// Primary-assembly contig this ALT aligns to; -1 when it has no usable `.alt` line.
    pub trid: i32,
    /// The `.alt` line has FLAG 16: the contig is the reverse complement of its primary span.
    pub rev: bool,
    /// 0-based start of the alignment on `trid`.
    pub tpos: i64,
    /// Length of `trid` (lifted intervals are clamped to it).
    pub tlen: i64,
    /// Query bases the CIGAR consumes (M/I/S/H/=/X): the ALT contig's length.
    pub qlen: i64,
    /// BAM-style ops, `len << 4 | op` (0 M, 1 I, 2 D, 3 N, 4 S, 5 H, 7 =, 8 X); P is dropped.
    pub cigar: Vec<u32>,
    /// Query bases before op `i`, `cigar.len() + 1` entries.
    pub qoff: Vec<i64>,
    /// Reference position (on `trid`) before op `i`, `cigar.len() + 1` entries.
    pub roff: Vec<i64>,
}

/// The table, indexed by contig id (the fork's `alt_lift_t`).
#[derive(Debug, Clone, Default)]
pub struct AltLift {
    /// ALT contigs with a usable alignment.
    pub n_lift: usize,
    /// One entry per contig of the index.
    pub a: Vec<AltLift1>,
}

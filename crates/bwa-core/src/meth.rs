//! Methylation (`--meth`) mode: bisulfite, EM-seq, TAPS and Illumina 5-base reads.
//!
//! Ported from fg-labs/bwa-mem3 0.16 (`meth_opts.cpp`, `mem_opt_fill_meth_mat`,
//! `mem_opt_apply_meth_defaults`, the read conversion in `fastmap.cpp`), Copyright (C) 2026
//! Fulcrum Genomics LLC, MIT licensed; the notice is retained in the repository `LICENSE`.
//!
//! The model, in one paragraph: read 1 (and single-end reads) is seeded with every C turned into
//! T, read 2 with every G turned into A, against an index of the genome converted both ways
//! (`<ref>.meth`, contigs `r<name>` G->A then `f<name>` C->T). Seeds are mapped back to the
//! original genome and everything after seeding (extension, pairing, CIGAR, output) runs on the
//! ORIGINAL read against the ORIGINAL genome, with a scoring matrix that frees the one conversion
//! the read's chemistry can produce. See `docs/meth-mode.md`.

/// Which bases the chemistry converts, hence the polarity of the XM:Z calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chem {
    /// Bisulfite / EM-seq: UNmethylated C reads as T.
    EmSeq,
    /// TAPS / Illumina 5-base: METHYLATED C reads as T.
    Taps,
}

/// How the freed conversion scores (`--meth-scoring`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scoring {
    /// C/T and G/A interchangeable both ways, scored as matches (bwameth-like).
    Collapsed,
    /// Only the conversion direction is free, scored as a match.
    Genomic,
    /// Only the conversion direction is free, scored 0.
    Neutral,
}

/// SMEM pruning before suffix-array resolution (`--meth-seed-prune`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prune {
    /// Per-read two-regime rule (the default under `--meth`).
    Spec30,
    /// Drop seeds shorter than 25 that occur more than once.
    Baseline,
    /// No pruning.
    Off,
}

/// `--meth-tags` bits.
pub const TAG_XR: u8 = 1;
/// `XG:Z`, the genome strand of the conversion.
pub const TAG_XG: u8 = 2;
/// `XM:Z`, the per-base methylation call string.
pub const TAG_XM: u8 = 4;
/// Every tag, the default.
pub const TAG_ALL: u8 = TAG_XR | TAG_XG | TAG_XM;

/// The fork's spec30 thresholds (`BWAMEM3_METH_PRUNE_{T,L,U}` defaults).
const PRUNE_T: i32 = 30;
const PRUNE_L: i32 = 25;
const PRUNE_U: i32 = 22;
/// The baseline rule's occurrence bound (`BWAMEM3_METH_PRUNE_KB`).
const PRUNE_KB: i64 = 1;

/// Everything `--meth` adds to [`crate::MemOpt`]. `None` there means the plain bwa pipeline.
#[derive(Debug, Clone)]
pub struct MethOpt {
    pub chem: Chem,
    pub scoring: Scoring,
    /// OR of `TAG_*`.
    pub tags: u8,
    pub prune: Prune,
    /// `--set-as-failed f|r`: flag alignments of that conversion strand QC-fail (0x200).
    pub set_as_failed: Option<u8>,
    /// `--chimera-qc`: bwameth's longest-M < 44% rule.
    pub chimera_qc: bool,
    /// Scoring matrices by hypothesis, `[ref * 5 + read]`: `[0]` frees G->A (read 2, OB),
    /// `[1]` frees C->T (read 1, OT). Filled by [`MethOpt::fill_mats`] from the plain matrix.
    pub mats: [[i8; 25]; 2],
}

impl MethOpt {
    /// `--meth[=CHEM]` defaults: collapsed for EM-seq, neutral for TAPS, spec30 pruning, all tags.
    pub fn new(chem: Chem) -> Self {
        MethOpt {
            chem,
            scoring: if chem == Chem::Taps {
                Scoring::Neutral
            } else {
                Scoring::Collapsed
            },
            tags: TAG_ALL,
            prune: Prune::Spec30,
            set_as_failed: None,
            chimera_qc: false,
            mats: [[0; 25]; 2],
        }
    }

    /// `mem_opt_fill_meth_mat`: the two matrices from the plain one (`mat`, after `-A`/`-B`).
    pub fn fill_mats(&mut self, mat: &[i8; 25], a: i32) {
        let conv = if self.scoring == Scoring::Neutral {
            0
        } else {
            a as i8
        };
        let (mut ot, mut ob) = (*mat, *mat);
        ot[5 + 3] = conv; // ref C, read T
        ob[2 * 5] = conv; // ref G, read A
        if self.scoring == Scoring::Collapsed {
            ot[3 * 5 + 1] = a as i8; // ref T, read C
            ob[2] = a as i8; // ref A, read G
        }
        self.mats = [ob, ot];
    }

    /// `meth_prune_smems` keep rule for one SMEM of a read whose longest SMEM is `max_len`.
    /// The caller always keeps the first SMEM of that maximal length.
    pub fn prune_keeps(&self, len: i32, occ: i64, max_len: i32) -> bool {
        match self.prune {
            Prune::Off => true,
            Prune::Spec30 => max_len < PRUNE_T || len >= PRUNE_L || (occ == 1 && len >= PRUNE_U),
            Prune::Baseline => len >= PRUNE_L || occ <= PRUNE_KB,
        }
    }
}

/// `--meth[=CHEM]`. Case-sensitive, as in the fork.
pub fn parse_chem(s: &str) -> Result<Chem, String> {
    match s {
        "emseq" | "em-seq" | "bisulfite" => Ok(Chem::EmSeq),
        "taps" | "5base" | "5-base" => Ok(Chem::Taps),
        _ => Err(format!(
            "--meth accepts 'emseq' (default; aliases em-seq, bisulfite), 'taps', or '5base' \
             (alias 5-base), got '{s}'"
        )),
    }
}

/// `--meth-scoring`.
pub fn parse_scoring(s: &str) -> Result<Scoring, String> {
    match s {
        "collapsed" => Ok(Scoring::Collapsed),
        "genomic" => Ok(Scoring::Genomic),
        "neutral" => Ok(Scoring::Neutral),
        _ => Err("--meth-scoring requires 'collapsed', 'genomic', or 'neutral'".into()),
    }
}

/// `--meth-seed-prune`.
pub fn parse_prune(s: &str) -> Result<Prune, String> {
    match s {
        "spec30" => Ok(Prune::Spec30),
        "baseline" => Ok(Prune::Baseline),
        "off" => Ok(Prune::Off),
        _ => Err("--meth-seed-prune accepts 'spec30' (default), 'baseline', or 'off'".into()),
    }
}

/// `mem_opt_parse_meth_tags`: `all`, `none`, `XR,XG`, or exclusions `^XM` / `-XM` (case-insensitive,
/// inclusions and exclusions not mixed).
pub fn parse_tags(spec: &str) -> Result<u8, String> {
    if spec.is_empty() {
        return Err("empty tag spec".into());
    }
    if spec.eq_ignore_ascii_case("all") {
        return Ok(TAG_ALL);
    }
    if spec.eq_ignore_ascii_case("none") {
        return Ok(0);
    }
    if spec.ends_with(',') {
        return Err("trailing comma".into());
    }
    let (mut inc, mut exc, mut n_inc, mut n_exc) = (0u8, 0u8, 0, 0);
    for tok in spec.split(',') {
        if tok.is_empty() {
            return Err("empty tag in list".into());
        }
        let (excl, name) = match tok.strip_prefix(['^', '-']) {
            Some(rest) => (true, rest),
            None => (false, tok),
        };
        let bit = match name.to_ascii_uppercase().as_str() {
            "XR" => TAG_XR,
            "XG" => TAG_XG,
            "XM" => TAG_XM,
            _ => return Err("unknown tag (expected XR, XG or XM)".into()),
        };
        if excl {
            exc |= bit;
            n_exc += 1;
        } else {
            inc |= bit;
            n_inc += 1;
        }
    }
    if n_inc > 0 && n_exc > 0 {
        return Err("cannot mix included and ^excluded tags in one spec".into());
    }
    Ok(if n_exc > 0 { TAG_ALL & !exc } else { inc })
}

/// The seeding copy of a read in nt4 codes: C->T for read 1 / single-end (`ot`), G->A for read 2.
/// nt4 already folds case, so this is the fork's `c|C -> T` / `g|G -> A`.
pub fn convert_codes(codes: &[u8], ot: bool) -> Vec<u8> {
    let (from, to) = if ot { (1, 3) } else { (2, 0) };
    codes
        .iter()
        .map(|&c| if c == from { to } else { c })
        .collect()
}

/// One FASTA record of `<ref>.meth.fa`: `>{prefix}{name}`, uppercased, `from` turned into `to`,
/// 100 bases per line.
pub fn write_converted(
    out: &mut impl std::io::Write,
    prefix: char,
    name: &str,
    seq: &[u8],
    from: u8,
    to: u8,
) -> std::io::Result<()> {
    writeln!(out, ">{prefix}{name}")?;
    for line in seq.chunks(100) {
        let conv: Vec<u8> = line
            .iter()
            .map(|&c| {
                let c = c.to_ascii_uppercase();
                if c == from {
                    to
                } else {
                    c
                }
            })
            .collect();
        out.write_all(&conv)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_follows_read_number() {
        // A C G T N
        let codes = [0, 1, 2, 3, 4];
        assert_eq!(convert_codes(&codes, true), [0, 3, 2, 3, 4]);
        assert_eq!(convert_codes(&codes, false), [0, 1, 0, 3, 4]);
    }

    #[test]
    fn matrices_free_only_the_chemistry() {
        let mut mat = [0i8; 25];
        for r in 0..5 {
            for q in 0..5 {
                mat[r * 5 + q] = if r == 4 || q == 4 {
                    -1
                } else if r == q {
                    1
                } else {
                    -4
                };
            }
        }
        let mut m = MethOpt::new(Chem::EmSeq);
        m.fill_mats(&mat, 1);
        let [ob, ot] = m.mats;
        assert_eq!((ot[5 + 3], ot[3 * 5 + 1], ot[2 * 5]), (1, 1, -4));
        assert_eq!((ob[2 * 5], ob[2], ob[5 + 3]), (1, 1, -4));
        m.scoring = Scoring::Neutral;
        m.fill_mats(&mat, 1);
        assert_eq!((m.mats[1][5 + 3], m.mats[1][3 * 5 + 1]), (0, -4));
        assert_eq!(MethOpt::new(Chem::Taps).scoring, Scoring::Neutral);
    }

    #[test]
    fn tag_specs_parse_like_the_fork() {
        assert_eq!(parse_tags("all"), Ok(TAG_ALL));
        assert_eq!(parse_tags("NONE"), Ok(0));
        assert_eq!(parse_tags("XR,xg"), Ok(TAG_XR | TAG_XG));
        assert_eq!(parse_tags("^XM"), Ok(TAG_XR | TAG_XG));
        assert_eq!(parse_tags("-XM,-XR"), Ok(TAG_XG));
        assert!(parse_tags("XR,^XM").is_err());
        assert!(parse_tags("XR,").is_err());
        assert!(parse_tags("XR,,XG").is_err());
        assert!(parse_tags("YY").is_err());
    }

    #[test]
    fn spec30_prunes_short_repeats_only_when_a_long_seed_exists() {
        let m = MethOpt::new(Chem::EmSeq);
        assert!(m.prune_keeps(19, 50, 29)); // regime A: longest < 30 keeps all
        assert!(!m.prune_keeps(24, 2, 40));
        assert!(m.prune_keeps(22, 1, 40));
        assert!(m.prune_keeps(25, 9, 40));
    }

    #[test]
    fn converted_fasta_is_uppercased_and_wrapped() {
        let mut out = Vec::new();
        let seq: Vec<u8> = b"acgtn".repeat(41);
        write_converted(&mut out, 'f', "chr1", &seq, b'C', b'T').unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], ">fchr1");
        assert_eq!(lines[1].len(), 100);
        assert_eq!(lines[3].len(), 5);
        assert!(lines[1].starts_with("ATGTNATGTN"));
    }
}

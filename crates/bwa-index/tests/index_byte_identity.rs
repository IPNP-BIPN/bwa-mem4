//! Byte-identity gate for the indexer: building the tiny fixture must reproduce the five index
//! files produced by `bwa-mem2 index` exactly.
//!
//! The oracle is `testdata/tiny/tiny.fa.{pac,ann,amb,0123,bwt.2bit.64}`: files checked into the repo
//! that were produced by the real C `bwa-mem2 index` on `testdata/tiny/tiny.fa` (a single 200,001 bp
//! contig, chr20:2000000-2200000). They are the *entire* output set of `bwa-mem2 index`, so passing
//! this test means our indexer is a drop-in replacement on that input: an index we build can be read
//! by C bwa-mem2 and vice versa.
//!
//! Why byte-identity of the index and not just "an index that works": the aligner's own SAM-level
//! byte-identity requirement is only meaningful if both sides search the same structure. A `.bwt.2bit.64`
//! that differs anywhere (different SA order for equal suffixes, a different sentinel row, a different
//! checkpoint layout) can still align reads correctly while silently changing seed ORDER, and seed
//! order propagates into chain order, into `mem_alnreg_v` order, and out to the SAM. So the index is
//! gated at the byte level rather than at the "same alignments" level.
//!
//! What a failure means, per extension:
//!   - `.pac`   : the 2-bit packing or the trailing `L mod 4` byte is wrong (`bns_fasta2bntseq`).
//!   - `.ann`   : contig name/comment splitting, the `(null)` placeholder, offsets, or the `n_ambs`
//!                per-contig count drifted.
//!   - `.amb`   : ambiguous-base hole detection differs (run merging, or the per-contig `lasts` reset).
//!   - `.0123`  : the forward++reverse-complement binary reference is mis-built (complement rule, or
//!                the RC half's index arithmetic).
//!   - `.bwt.2bit.64`: the deep one. Either SA-IS produced a different suffix array (our `crate::sais`
//!                vs the C's), or the BWT/CP_OCC/compressed-SA serialization drifted. The reported
//!                first-differing byte offset localizes it: the header is 48 B, then CP_OCC blocks of
//!                64 B, then `sa_ms_byte`, then `sa_ls_word`, then the sentinel index.
//!
//! Coverage caveat: this fixture has ONE contig and ZERO ambiguous bases (`tiny.fa.amb` is the
//! 3-number header and nothing else). So it does not exercise the `lrand48` N-randomization path nor
//! the cross-contig hole-splitting rule. Those are covered separately by the unit test
//! `build::tests::amb_holes_not_merged_across_contig_boundary`, which builds a synthetic 2-contig
//! FASTA with N-runs on the boundary.

use std::path::{Path, PathBuf};

/// # Returns
/// Absolute path to `testdata/tiny`, the directory holding both the input FASTA and the five oracle
/// files produced by C `bwa-mem2 index`. Derived from `CARGO_MANIFEST_DIR` (this crate's root,
/// baked in at compile time) so the test is independent of the working directory the runner uses.
fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny")
}

/// Builds the tiny fixture with our indexer and asserts all five outputs match the checked-in C
/// `bwa-mem2 index` oracle byte for byte. Panics on the first extension that differs, naming the
/// offset; see the module docs for what each extension's failure implicates.
#[test]
fn tiny_index_is_byte_identical() {
    // The fixture FASTA: one 200,001 bp contig (chr20:2000000-2200000), no ambiguous bases.
    let src = tiny_dir().join("tiny.fa");
    // Build into a scratch dir, not `testdata/`: `build_index` writes its five outputs as siblings of
    // the FASTA, which would clobber the very oracle files this test compares against.
    // Scratch build directory, named with the PID so two concurrent `cargo test` processes cannot
    // collide. Removed on both the success and the failure path.
    let dir = std::env::temp_dir().join(format!("bwamem3_idx_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The FASTA copy inside the scratch dir. `build_index` writes its outputs next to this path, so
    // the copy is what keeps the oracle files untouched.
    let dst = dir.join("tiny.fa");
    std::fs::copy(&src, &dst).unwrap();

    bwa_index::build_index(&dst).expect("build_index");

    // These five extensions are the COMPLETE output set of `bwa-mem2 index`; comparing all of them is
    // what makes passing this test equivalent to being a drop-in replacement on this input.
    for ext in ["pac", "ann", "amb", "0123", "bwt.2bit.64"] {
        // `ours`: the file our `build_index` just wrote into the scratch dir.
        // `oracle`: the same-named file in `testdata/tiny`, produced by the real C `bwa-mem2 index`
        // and checked into the repo. Both are read whole; the largest is a few MB at this size.
        let ours = std::fs::read(dir.join(format!("tiny.fa.{ext}"))).unwrap();
        let oracle = std::fs::read(tiny_dir().join(format!("tiny.fa.{ext}"))).unwrap();
        if ours != oracle {
            // Report the first differing byte rather than dumping megabytes: with the layouts noted in
            // the module docs, the offset alone usually names the guilty stage. `pos` is `None` when
            // one file is a strict prefix of the other, in which case the two lengths are the signal.
            let pos = ours.iter().zip(&oracle).position(|(a, b)| a != b);
            // Clean up before panicking: `remove_dir_all` after the `panic!` would never run.
            std::fs::remove_dir_all(&dir).ok();
            panic!(
                ".{ext}: differs at byte {:?} (len ours={}, oracle={})",
                pos,
                ours.len(),
                oracle.len()
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

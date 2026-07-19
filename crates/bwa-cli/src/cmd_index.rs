//! `bwa-mem3 index` subcommand: build the FMD index (byte-identical to `bwa-mem2 index`).

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

// `bwa-mem3 index`'s option set: one positional and nothing else.
//
// `//` rather than `///`: clap can surface an args struct's doc comment in the subcommand's help,
// and the `index` help text must stay exactly as it is. The per-field `///` below is the intended
// help string.
#[derive(Args)]
pub struct IndexArgs {
    // The only argument. bwa-mem2's `index` accepts `-p prefix` to place the output elsewhere; this
    // port does not, so the side files are always written next to the FASTA and the FASTA path
    // doubles as the index prefix passed to `mem`.
    /// FASTA reference to index.
    pub fasta: PathBuf,
}

/// Build the index and report elapsed time on stderr.
///
/// One-shot and expensive: indexing a human genome takes minutes and tens of GB, but is done once
/// per reference and reused by every subsequent `mem` run. Writes several side files derived from
/// the FASTA path (the FM index itself, the packed 2-bit reference, and the contig dictionary);
/// `bwa_index::build_index` owns the exact names and formats, which are byte-compatible with
/// `bwa-mem2 index`. Overwrites any existing index without prompting.
///
/// # Parameters
///
/// - `args`: the parsed command line, supplied by `main`'s dispatch. Its single field `fasta` is
///   the reference to read; it must exist and be readable FASTA (plain or gzipped, per
///   `bwa_index`). It doubles as the index prefix, so the caller must have write permission in the
///   containing directory.
///
/// # Returns
///
/// `Ok(())` once every side file has been written. Errors from `build_index` (missing FASTA,
/// unwritable directory, malformed input) propagate unchanged; a failure can leave partially
/// written side files behind.
pub fn run(args: IndexArgs) -> anyhow::Result<()> {
    // Wall-clock origin for the one stderr progress line below. Not used for anything else.
    let t0 = Instant::now();
    bwa_index::build_index(&args.fasta)?;
    eprintln!(
        "[bwa-mem3 index] built index for {} in {:.3}s",
        args.fasta.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

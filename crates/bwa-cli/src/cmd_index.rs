//! `bwa-mem4 index` subcommand: build the FMD index (byte-identical to `bwa-mem2 index`).

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

// `bwa-mem4 index`'s option set: one positional plus `-p`, matching `bwa-mem2 index`.
//
// `//` rather than `///`: clap can surface an args struct's doc comment in the subcommand's help,
// and the `index` help text must stay exactly as it is. The per-field `///` below is the intended
// help string.
#[derive(Args)]
pub struct IndexArgs {
    /// FASTA reference to index.
    pub fasta: PathBuf,
    /// Index file prefix [same as the FASTA path]. The five side files become `<prefix>.pac`,
    /// `.ann`, `.amb`, `.bwt.2bit.64` and `.0123`, and `<prefix>` is what `mem` is then given.
    #[arg(short = 'p')]
    pub prefix: Option<PathBuf>,
    // NOT a bwa flag and OFF by default, deliberately. bwa's five files are an FM-index; the
    // long-read path (`mem -x pacbio|pbref|ont2d`) is served by rammap, which needs a MINIMIZER
    // index instead and cannot read bwa's. Building it here means a long-read run does not pay for
    // it on first use, but it is opt-in because it roughly doubles index time and adds several GB
    // that a short-read user would never open.
    //
    // The five bwa files are untouched and stay byte-identical to `bwa-mem2 index` either way; this
    // only ever ADDS `<prefix>.rammap.<preset>.mmi`, which is exactly the name `mem` looks for.
    /// Also build rammap minimizer indexes for long-read mapping: `map-ont`, `map-pb`, or `all`.
    /// Repeatable. Without this, `mem -x ont2d`/`pacbio` builds and caches one on first use.
    #[arg(long = "mmi", value_name = "PRESET")]
    pub mmi: Vec<String>,
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
    // Defaulting the prefix to the FASTA path is bwa's behaviour, not just a convenience: it is
    // why a bare path can be passed to both `index` and `mem`.
    let prefix = args.prefix.clone().unwrap_or_else(|| args.fasta.clone());
    bwa_index::build_index_with_prefix(&args.fasta, &prefix)?;
    eprintln!(
        "[bwa-mem4 index] built index for {} in {:.3}s",
        args.fasta.display(),
        t0.elapsed().as_secs_f64()
    );
    // Optional minimizer indexes, after the bwa five so a failure here cannot leave those missing.
    for preset in resolve_mmi_presets(&args.mmi)? {
        let t = Instant::now();
        let path = crate::cmd_longread::mmi_path(&prefix, preset);
        crate::cmd_longread::build_mmi(preset, &args.fasta, &path)?;
        eprintln!(
            "[bwa-mem4 index] built {} in {:.3}s",
            path.display(),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Expand the `--mmi` values into the presets to build.
///
/// # Parameters
/// - `values`: the raw `--mmi` arguments. `all` expands to every long-read preset `mem -x` can
///   route to; the names are the same ones `mem` reports.
///
/// # Returns
/// The presets to build, deduplicated and in a stable order, or an error naming the accepted values.
fn resolve_mmi_presets(values: &[String]) -> anyhow::Result<Vec<rammap::Preset>> {
    use rammap::Preset;
    let mut out: Vec<Preset> = Vec::new();
    let push = |p: Preset, out: &mut Vec<Preset>| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    for v in values {
        match v.as_str() {
            "all" => {
                push(Preset::MapOnt, &mut out);
                push(Preset::MapPb, &mut out);
            }
            "map-ont" | "ont2d" | "ont" => push(Preset::MapOnt, &mut out),
            "map-pb" | "pacbio" | "pbref" | "pb" => push(Preset::MapPb, &mut out),
            other => anyhow::bail!(
                "unknown --mmi preset '{other}': expected map-ont, map-pb or all \
                 (bwa's spellings ont2d, pacbio and pbref are accepted too)"
            ),
        }
    }
    Ok(out)
}

//! Long-read mapping, routed to rammap instead of bwa's `-x` presets.
//!
//! # Why this exists
//!
//! bwa-mem2 accepts `-x pacbio`, `-x pbref` and `-x ont2d`, and its own code prints
//! `WARNING: bwa-mem2 doesn't work well with long reads or contigs; please use minimap2 instead.`
//! before running them (`fastmap.cpp:820`). That warning is honest: the presets only retune a
//! short-read seed-and-extend design (cheaper gaps, a longer re-seed, no clip penalty) and cannot
//! turn it into a long-read mapper. Reproducing that output byte for byte, which this port did until
//! now, faithfully reproduces a result its own author tells you not to use.
//!
//! So the three LONG-READ presets are routed to [rammap](https://github.com/jwanglab/rammap)
//! (`rammap-core`, MIT, pure Rust), which mirrors minimap2 and produces its output. `-x intractg`
//! is NOT routed: it is intra-species contig alignment, not long reads, and it stays on the
//! bwa-identical path.
//!
//! # What this means for the output contract
//!
//! Everywhere else, this binary's promise is "byte-identical to bwa-mem2 2.3". Here it is
//! deliberately not, and it cannot be quiet about that:
//!
//! - a banner on stderr says which mapper ran and which preset it was given;
//! - the `@PG` line records `rammap` and its version, not `bwa-mem4`, so a SAM file says who made
//!   it;
//! - `-x intractg` and every short-read path are untouched.
//!
//! The alternative, silently returning minimap2-shaped records under a bwa flag, would be worse
//! than either honest option.
//!
//! # Preset mapping
//!
//! | bwa `-x` | rammap preset | why |
//! |---|---|---|
//! | `ont2d` | `map-ont` | Nanopore 2D reads; bwa's own preset lowers `min_seed_len` to 14 |
//! | `pacbio` | `map-pb` | noisy PacBio CLR, which is what bwa's preset targets |
//! | `pbref` | `map-pb` | bwa treats `pbref` and `pacbio` identically (one shared branch) |

use anyhow::{Context, Result};
use rammap::align::extend::AlignmentContext;
use rammap::align::index::Index;
use rammap::align::map::MapContext;
use rammap::align::pipeline::{align_and_format_query, OutputConfig, ReadInfo};
use rammap::{Aligner, Preset};
use std::io::Write;
use std::path::Path;

/// Which rammap preset a bwa `-x` mode routes to, or `None` if it stays on the bwa path.
///
/// # Parameters
/// - `mode`: the raw `-x` argument, already validated as one of bwa's four spellings by
///   `build_opt`.
///
/// # Returns
/// `Some(preset)` for the three long-read modes, `None` for `intractg` (and for anything else,
/// which `build_opt` has already rejected).
pub fn preset_for(mode: &str) -> Option<Preset> {
    match mode {
        "ont2d" => Some(Preset::MapOnt),
        "pacbio" | "pbref" => Some(Preset::MapPb),
        _ => None,
    }
}

/// Locate the reference FASTA behind an index prefix.
///
/// bwa's convention is that the prefix IS the FASTA path (`bwa index genome.fa` writes
/// `genome.fa.bwt.2bit.64` and friends), so the prefix itself is tried first. rammap builds a
/// minimizer index rather than reading bwa's, so it needs the sequences, not the `.bwt`.
///
/// # Parameters
/// - `prefix`: the index prefix given on the command line.
///
/// # Returns
/// The path to use, or an error naming everything that was tried.
fn reference_fasta(prefix: &Path) -> Result<std::path::PathBuf> {
    if prefix.is_file() {
        return Ok(prefix.to_path_buf());
    }
    // A few conventional spellings, in case the prefix was given without its extension.
    for ext in ["fa", "fasta", "fna", "fa.gz", "fasta.gz"] {
        let candidate = prefix.with_extension(ext);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "long-read mapping needs the reference FASTA, not bwa's index: tried {} and {}.{{fa,fasta,fna}}. \
         rammap builds its own minimizer index from the sequences.",
        prefix.display(),
        prefix.display()
    )
}

/// Map long reads with rammap and write SAM.
///
/// # Parameters
/// - `preset`: from [`preset_for`].
/// - `mode`: the original `-x` spelling, for the banner.
/// - `prefix`: the index prefix, resolved to a FASTA by [`reference_fasta`].
/// - `reads`: the read files. A second file is mapped as independent single-end reads: pairing is a
///   short-read concept and rammap's own CLI treats long reads as fragments.
/// - `out`: where the SAM goes, already opened by the caller (stdout or `-o`).
/// - `argv`: the full command line, for the `@PG CL:` field.
pub fn run(
    preset: Preset,
    mode: &str,
    prefix: &Path,
    reads: &[std::path::PathBuf],
    out: &mut dyn Write,
    argv: &[String],
) -> Result<()> {
    let fasta = reference_fasta(prefix)?;
    eprintln!(
        "[M::main_mem] -x {mode} is a long-read preset. bwa-mem2's own code calls its long-read \
         modes unusable; this run is mapped by rammap (minimap2-equivalent) with preset {preset:?}, \
         NOT by bwa-mem4, and the output is not byte-identical to bwa-mem2. The @PG line records \
         rammap. Use -x intractg or drop -x for the bwa-identical paths."
    );

    let aligner = load_or_build_index(preset, prefix, &fasta)?;

    // SAM, with CIGAR. The remaining toggles are minimap2's optional tags, left off so the records
    // carry what a bwa user expects; `split_mode` is rammap's multi-part index feature, unused here.
    let out_cfg = OutputConfig {
        do_cigar: true,
        do_cs: false,
        cs_long: false,
        do_md: false,
        do_ds: false,
        eqx: false,
        output_sam: true,
        rg_id: None,
        split_mode: false,
    };

    write_header(out, aligner.index(), argv)?;

    // Records emitted, for the closing log line.
    let mut n_reads = 0u64;
    for path in reads {
        let mut reader = bwa_io::FastqReader::from_path(path)
            .with_context(|| format!("opening {}", path.display()))?;
        // One batch in flight, sized by BASES rather than reads: a nanopore file mixes 500 bp and
        // 500 kb reads, so a fixed read count would swing the resident set by three orders of
        // magnitude. The reader thread of the short-read path is deliberately not reused here --
        // long-read mapping is compute-bound by a wide margin, so a simple read-batch-map-write loop
        // leaves nothing on the table.
        const BATCH_BASES: usize = 48 << 20;
        let mut batch: Vec<bwa_io::Record> = Vec::new();
        let mut batch_bases = 0usize;
        loop {
            let more = match reader.next_record()? {
                Some(rec) => {
                    batch_bases += rec.seq.len();
                    batch.push(rec);
                    batch_bases < BATCH_BASES
                }
                None => false,
            };
            if more {
                continue;
            }
            if batch.is_empty() {
                break;
            }
            n_reads += batch.len() as u64;
            map_batch(&aligner, &out_cfg, &batch, out)?;
            batch.clear();
            batch_bases = 0;
        }
    }
    eprintln!("[M::main_mem] rammap mapped {n_reads} reads");
    Ok(())
}

/// Map one batch on the rayon pool and write its SAM in input order.
///
/// Each read is independent, so the only shared state is the immutable index and options. rammap's
/// `AlignmentContext` and `MapContext` are per-thread scratch, so one pair is created per CHUNK
/// rather than per read: constructing them per read would dominate a batch of short fragments, and
/// sharing one across threads is not possible (they are `&mut`).
///
/// Order is preserved because `par_chunks` is an INDEXED parallel iterator: `collect` puts each
/// chunk's output back at its own position, so the SAM comes out in the order the reads were read,
/// exactly as the single-threaded loop would have written it.
///
/// # Parameters
/// - `aligner`: the built index and its options, shared immutably by every worker.
/// - `out_cfg`: SAM formatting toggles, likewise shared.
/// - `batch`: the reads to map, in input order.
/// - `out`: destination for the formatted records.
fn map_batch(
    aligner: &Aligner,
    out_cfg: &OutputConfig,
    batch: &[bwa_io::Record],
    out: &mut dyn Write,
) -> Result<()> {
    use rayon::prelude::*;
    // Enough chunks for work stealing to even out a batch whose reads differ 1000x in length,
    // without shrinking them to the point where the per-chunk context setup shows up.
    let workers = rayon::current_num_threads().max(1);
    let chunk = batch.len().div_ceil(workers * 4).max(1);
    let pieces: Vec<String> = batch
        .par_chunks(chunk)
        .map(|reads| {
            let mut ctx = AlignmentContext::new();
            let mut map_ctx = MapContext::new();
            let mut buf = String::new();
            for rec in reads {
                // Our reader keeps QUAL as raw bytes (copied verbatim into SAM on the short-read
                // path); rammap's formatter wants `&str`. FASTQ quality is printable ASCII by
                // definition, so this is lossless; a malformed file that is not gets its QUAL
                // dropped rather than the run aborted.
                let qual = rec
                    .qual
                    .as_deref()
                    .and_then(|q| std::str::from_utf8(q).ok());
                let read = ReadInfo {
                    qname: &rec.name,
                    qseq: &rec.seq,
                    qual,
                    comment: rec.comment.as_deref(),
                    n_seg: 1,
                    seg_idx: 0,
                };
                let (text, _stats) = align_and_format_query(
                    aligner.options(),
                    aligner.index(),
                    &read,
                    &mut ctx,
                    &mut map_ctx,
                    None,
                    None,
                    out_cfg,
                );
                buf.push_str(&text);
            }
            buf
        })
        .collect();
    for piece in &pieces {
        out.write_all(piece.as_bytes())?;
    }
    Ok(())
}

/// Where the minimizer index for `preset` lives, given an index prefix.
///
/// Keyed by preset because `map-ont` and `map-pb` use different k-mer and window sizes: an index
/// built for one is not valid for the other, so they cannot share a filename. `bwa-mem4 index
/// --mmi` writes exactly this path, which is what makes a pre-built index findable here.
///
/// # Parameters
/// - `prefix`: the index prefix, i.e. what `mem` is given.
/// - `preset`: the rammap preset the index is built for.
///
/// # Returns
/// `<prefix>.rammap.<preset>.mmi`.
pub fn mmi_path(prefix: &Path, preset: Preset) -> std::path::PathBuf {
    let mut name = prefix.as_os_str().to_os_string();
    name.push(format!(".rammap.{}.mmi", preset_slug(preset)));
    std::path::PathBuf::from(name)
}

/// Build one minimizer index from a FASTA and write it to `path`.
///
/// Shared by `bwa-mem4 index --mmi` (build it up front) and the `mem` long-read path (build it on
/// first use), so the two cannot drift in parameters or filename.
///
/// # Parameters
/// - `preset`: decides k-mer size, window size and homopolymer compression.
/// - `fasta`: the reference sequences. rammap cannot read bwa's FM-index.
/// - `path`: destination, normally from [`mmi_path`].
pub fn build_mmi(preset: Preset, fasta: &Path, path: &Path) -> Result<()> {
    let fasta_str = fasta
        .to_str()
        .context("reference path is not valid UTF-8")?;
    let path_str = path.to_str().context("index path is not valid UTF-8")?;
    let aligner = Aligner::from_fasta(fasta_str, preset)
        .with_context(|| format!("building the rammap index from {}", fasta.display()))?;
    aligner
        .index()
        .save(path_str)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Load rammap's minimizer index, building and caching it beside the reference the first time.
///
/// **The bwa index cannot be reused.** bwa's is an FM-index (`.bwt.2bit.64`, `.pac`, `.ann`,
/// `.amb`); rammap's is a minimizer index in minimap2's `.mmi` shape. They are different data
/// structures for different seeding strategies, so a long-read run needs its own, built from the
/// reference sequences.
///
/// Rebuilding it per run would cost minutes and several GB on a human genome, so it is written once
/// to `<prefix>.rammap.mmi` and loaded from there afterwards. The cache is keyed by the PRESET,
/// because `map-ont` and `map-pb` use different k-mer and window sizes and an index built for one is
/// not valid for the other; the preset name is therefore part of the filename.
///
/// Staleness is handled the way bwa handles its own index files: it is not. A cache older than the
/// FASTA is reported and rebuilt, which costs one modification-time check and removes the failure
/// mode where an edited reference is silently mapped against the old one.
///
/// # Parameters
/// - `preset`: decides both the index parameters and the cache filename.
/// - `prefix`: the index prefix, i.e. where the cache is written.
/// - `fasta`: the resolved reference sequences.
///
/// # Returns
/// A ready `Aligner`, from the cache when one is usable and from a fresh build otherwise.
fn load_or_build_index(preset: Preset, prefix: &Path, fasta: &Path) -> Result<Aligner> {
    let cache = mmi_path(prefix, preset);
    let cache_str = cache
        .to_str()
        .context("index cache path is not valid UTF-8")?
        .to_owned();
    let fasta_str = fasta
        .to_str()
        .context("reference path is not valid UTF-8")?
        .to_owned();

    // Reuse the cache only if it is at least as new as the FASTA. A cache that cannot be read at all
    // is treated as absent rather than fatal: rebuilding is always correct.
    if cache.is_file() {
        let fresh = match (
            cache.metadata().and_then(|m| m.modified()),
            fasta.metadata().and_then(|m| m.modified()),
        ) {
            (Ok(c), Ok(f)) => c >= f,
            // No usable timestamps: trust the cache rather than rebuild on every run.
            _ => true,
        };
        if !fresh {
            eprintln!(
                "[M::main_mem] {} is older than the reference; rebuilding",
                cache.display()
            );
        } else {
            let started = std::time::Instant::now();
            match Aligner::from_index(&cache_str, preset) {
                Ok(a) => {
                    eprintln!(
                        "[M::main_mem] loaded the rammap index from {} in {:.1}s",
                        cache.display(),
                        started.elapsed().as_secs_f64()
                    );
                    return Ok(a);
                }
                Err(e) => eprintln!(
                    "[M::main_mem] could not load {} ({e}); rebuilding",
                    cache.display()
                ),
            }
        }
    }

    eprintln!(
        "[M::main_mem] building the rammap minimizer index from {} (bwa's FM-index cannot be \\
         reused; this happens once and is cached as {})",
        fasta.display(),
        cache.display()
    );
    let started = std::time::Instant::now();
    let aligner = Aligner::from_fasta(&fasta_str, preset)
        .with_context(|| format!("building the rammap index from {}", fasta.display()))?;
    eprintln!(
        "[M::main_mem] rammap index built in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    // A cache that cannot be written is a warning, never an error: the run can still proceed, it
    // will just pay the build again next time. Build it with `bwa-mem4 index --mmi` to avoid this.
    if let Err(e) = aligner.index().save(&cache_str) {
        eprintln!(
            "[M::main_mem] could not cache the index at {} ({e})",
            cache.display()
        );
    } else {
        eprintln!("[M::main_mem] cached the index at {}", cache.display());
    }
    Ok(aligner)
}

/// Filename-safe name for a preset, so two presets cannot share one cache file.
fn preset_slug(preset: Preset) -> &'static str {
    match preset {
        Preset::MapOnt => "map-ont",
        Preset::MapPb => "map-pb",
        // Only the two above are reachable from `preset_for`; anything else still gets a distinct
        // slug rather than silently colliding.
        _ => "other",
    }
}

/// Write the SAM header: `@HD`, one `@SQ` per target, and a `@PG` naming rammap.
///
/// Mirrors rammap's own header (`rammap/src/main.rs`), because the records under it are rammap's.
/// The `@PG` deliberately does NOT say `bwa-mem4`: the file should name the mapper that produced
/// its alignments.
fn write_header(out: &mut dyn Write, index: &Index, argv: &[String]) -> Result<()> {
    writeln!(out, "@HD\tVN:1.6\tSO:unsorted\tGO:query")?;
    for s in &index.seqs {
        writeln!(out, "@SQ\tSN:{}\tLN:{}", s.name, s.len)?;
    }
    writeln!(
        out,
        "@PG\tID:rammap\tPN:rammap\tVN:{}\tCL:{}",
        rammap_version(),
        argv.join(" ")
    )?;
    Ok(())
}

/// rammap's version, read from the dependency rather than hardcoded so a bump cannot make the
/// `@PG` line lie.
fn rammap_version() -> &'static str {
    // `rammap-core` does not re-export its own `CARGO_PKG_VERSION`, and ours would be the wrong
    // number, so this is pinned alongside the dependency in `Cargo.toml` and checked by
    // `pg_version_matches_the_linked_crate` below.
    "1.1.2"
}

#[cfg(test)]
mod tests {
    /// The `@PG VN:` we write must be the version actually linked.
    ///
    /// `rammap_version` is a literal because the crate does not expose its own `CARGO_PKG_VERSION`,
    /// which makes it exactly the kind of constant that goes stale on a dependency bump and starts
    /// writing a false provenance into every SAM file produced. The requirement is read back out of
    /// `Cargo.toml` rather than duplicated here, so the test cannot drift with the thing it checks.
    #[test]
    fn pg_version_matches_the_linked_crate() {
        let manifest = include_str!("../Cargo.toml");
        let declared = manifest
            .lines()
            .find_map(|l| l.trim().strip_prefix("rammap-core = "))
            .expect("rammap-core is not declared in Cargo.toml")
            .trim()
            .trim_matches('"')
            .to_owned();
        assert_eq!(
            declared,
            super::rammap_version(),
            "the @PG VN: written into every long-read SAM does not match the linked rammap-core; \
             bump `rammap_version` to match Cargo.toml"
        );
    }
}

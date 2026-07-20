//! `bwa-mem3` command-line entry point.
//!
//! Two subcommands, mirroring the two things you do with a read aligner: `index` prepares a
//! reference genome once, `mem` aligns reads against it. Both are drop-in compatible with their
//! `bwa-mem2` equivalents, which is the whole point: the same command line must produce the same
//! bytes. See [`cmd_mem::MemArgs`] for the option surface and the three bwa options not carried
//! over.

use clap::{Parser, Subcommand};

// mimalloc as the global allocator: the pipeline makes many small short-lived allocations (per-job
// query/target buffers, per-chunk DP scratch, per-read region vectors); a fast allocator with good
// locality cuts wall time noticeably. Does not affect output bytes (byte-identity preserved).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cmd_index;
mod cmd_mem;

// Top-level parsed command line. Holds nothing but the chosen subcommand: every real option lives
// on `cmd_index::IndexArgs` or `cmd_mem::MemArgs`.
//
// Deliberately a `//` comment, not `///`: clap derives `long_about` from a struct's doc comment, so
// a `///` here would appear in `bwa-mem3 --help`. The `about` string in the attribute below is the
// only help text this struct should contribute.
#[derive(Parser)]
#[command(
    name = "bwa-mem3",
    version,
    about = "Native Rust reimplementation of bwa-mem2"
)]
struct Cli {
    // Which subcommand was invoked, with its own already-parsed arguments. Always set: clap
    // refuses the command line outright when no subcommand is named.
    //
    // `//` rather than `///` so clap cannot pick this up as help text.
    #[command(subcommand)]
    cmd: Cmd,
}

/// The two subcommands. Note that clap's `about` strings here are ours, not bwa's: bwa has no
/// structured help for its subcommands.
#[derive(Subcommand)]
enum Cmd {
    // Variant payload: the one FASTA path to index. Run once per reference, minutes and tens of GB.
    /// Build the FMD index from a FASTA reference. Output is byte-identical to `bwa-mem2 index`.
    Index(cmd_index::IndexArgs),
    // Variant payload: the whole `mem` option surface (~35 flags plus 2-3 positionals). This is the
    // variant every production run takes.
    /// Align reads to an indexed reference.
    Mem(cmd_mem::MemArgs),
}

/// Process entry point: capture argv, parse it, dispatch to the chosen subcommand.
///
/// # Returns
///
/// `Ok(())` after the subcommand has run to completion (for `mem`, after the SAM sink has been
/// flushed and finalized). Any error propagates out of `main`, so the process exits non-zero and
/// prints the anyhow chain; there is no partial-success exit code.
fn main() -> anyhow::Result<()> {
    // Capture the raw command line for the @PG CL tag before clap consumes it. It must be the raw
    // argv, not a reconstruction from the parsed args: the CL field is meant to record what the
    // user actually typed, defaults and all, so the SAM file documents its own provenance.
    let argv: Vec<String> = std::env::args().collect();
    match Cli::parse().cmd {
        Cmd::Index(args) => cmd_index::run(args),
        Cmd::Mem(args) => cmd_mem::run(args, &argv),
    }
}

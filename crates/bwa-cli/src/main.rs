//! `bwa-mem4` command-line entry point.
//!
//! Two subcommands, mirroring the two things you do with a read aligner: `index` prepares a
//! reference genome once, `mem` aligns reads against it. Both are drop-in compatible with their
//! `bwa-mem2` equivalents, which is the whole point: the same command line must produce the same
//! bytes. See [`cmd_mem::MemArgs`] for the option surface and the three bwa options not carried
//! over.

//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `main.rs` | by convention the EXECUTABLE's entry point, as opposed to `lib.rs`, which is a library's root. This is the function the operating system starts. |
//! | `#[derive(Parser)]` | generates a whole command-line parser from a struct definition, using the `clap` crate. The fields become options and the doc comments become the help text, so the declaration and the documentation cannot drift apart. |
//! | `#[derive(Subcommand)]` | the same for an enum, where each variant is one subcommand (`index`, `mem`). |
//! | `#[global_allocator]` | designates which allocator the whole program uses for every heap allocation. Declaring one replaces the system default process-wide, including inside dependencies. |
//! | `static GLOBAL: ... = ...;` | a single value existing for the program's whole run. Here it is the allocator itself. |
//! | `use a::{B, C};` | imports two names from one module, so they can be written unqualified below. |

use clap::{Parser, Subcommand};

// mimalloc as the global allocator: the pipeline makes many small short-lived allocations (per-job
// query/target buffers, per-chunk DP scratch, per-read region vectors); a fast allocator with good
// locality cuts wall time noticeably. Does not affect output bytes (byte-identity preserved).
//
// Behind a feature because a global allocator is a whole-program decision and
// this crate is also a library now; see Cargo.toml. On by default, so this
// binary is unchanged.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// The command implementations live in the library target so an embedder can
// call them without spawning this binary. Declared there, used here, so there
// is exactly one copy of each.
use bwa_mem4::{cmd_index, cmd_mem};

// Top-level parsed command line. Holds nothing but the chosen subcommand: every real option lives
// on `cmd_index::IndexArgs` or `cmd_mem::MemArgs`.
//
// Deliberately a `//` comment, not `///`: clap derives `long_about` from a struct's doc comment, so
// a `///` here would appear in `bwa-mem4 --help`. The `about` string in the attribute below is the
// only help text this struct should contribute.
#[derive(Parser)]
#[command(
    name = "bwa-mem4",
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
    // No payload. Exists purely because `bwa-mem2 version` is a SUBCOMMAND, not a flag, and a
    // drop-in replacement that only answers `--version` breaks any script that calls the former.
    // It prints the bare version and nothing else, as bwa-mem2 does ("2.3"), so a caller can use it
    // unquoted; `--version` keeps clap's "bwa-mem4 4.3.3" form for humans.
    /// Print the version number, as `bwa-mem2 version` does.
    Version,
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
        Cmd::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}
